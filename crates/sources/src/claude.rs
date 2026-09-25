//! Claude Code transcripts: `~/.claude/projects/<dir>/<session>.jsonl`, plus subagent transcripts
//! in `<dir>/<session>/subagents/agent-*.jsonl`. One JSON event per line.

use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::time::UNIX_EPOCH;

use serde_json::Value;

use crate::tools::summarise;
use crate::{Locator, Result, Session, SessionRef, Source, SourceError, Turn, io_err, project, push_unique};

/// Lists every main and subagent transcript under `projects_dir` (normally `~/.claude/projects`).
pub fn discover(projects_dir: &Path) -> Result<Vec<SessionRef>> {
    let mut refs = Vec::new();
    for project in read_dir(projects_dir)? {
        if !project.is_dir() {
            continue;
        }
        for entry in read_dir(&project)? {
            if is_jsonl(&entry) {
                let id = stem(&entry);
                refs.push(session_ref(id, &entry)?);
            } else if entry.is_dir() {
                let subagents = entry.join("subagents");
                if subagents.is_dir() {
                    for sub in read_dir(&subagents)? {
                        if is_jsonl(&sub) {
                            let id = format!("{}/{}", stem(&entry), stem(&sub));
                            refs.push(session_ref(id, &sub)?);
                        }
                    }
                }
            }
        }
    }
    refs.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(refs)
}

fn read_dir(dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    let mut out: Vec<_> = fs::read_dir(dir)
        .map_err(io_err(dir))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    out.sort();
    Ok(out)
}

fn is_jsonl(p: &Path) -> bool {
    p.is_file() && p.extension().is_some_and(|e| e == "jsonl")
}

fn stem(p: &Path) -> String {
    p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
}

fn session_ref(id: String, path: &Path) -> Result<SessionRef> {
    let meta = fs::metadata(path).map_err(io_err(path))?;
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_millis());
    Ok(SessionRef {
        source: Source::Claude,
        id,
        fingerprint: format!("{}-{mtime}", meta.len()),
        locator: Locator::File(path.to_path_buf()),
    })
}

pub fn load(r: &SessionRef) -> Result<Session> {
    let Locator::File(path) = &r.locator else {
        return Err(SourceError::NotFound(r.id.clone()));
    };
    let file = File::open(path).map_err(io_err(path))?;
    parse(&r.id, BufReader::new(file))
}

/// Parses one transcript. Unparseable lines are skipped: the last line of a live session is
/// often half-written, and one bad line must not hide the rest of the session.
pub fn parse(id: &str, reader: impl Read) -> Result<Session> {
    let parent_id = id.split_once('/').map(|(parent, _)| parent.to_owned());
    let mut b = Builder::default();
    for line in BufReader::new(reader).lines() {
        let Ok(line) = line else { continue };
        let Ok(event) = serde_json::from_str::<Value>(&line) else { continue };
        b.event(&event);
    }
    b.finish(id, parent_id)
}

#[derive(Default)]
struct Builder {
    turns: Vec<Turn>,
    current: Option<Turn>,
    title: Option<String>,
    cwd: Option<String>,
    branch: Option<String>,
    first_ts: Option<i64>,
    last_ts: Option<i64>,
}

impl Builder {
    fn event(&mut self, e: &Value) {
        let ts = e.get("timestamp").and_then(Value::as_str).and_then(parse_ts);
        if let Some(ts) = ts {
            self.first_ts.get_or_insert(ts);
            self.last_ts = Some(ts);
        }
        match e.get("type").and_then(Value::as_str) {
            Some("ai-title") => {
                // Claude re-emits the title as the session evolves; the last one wins.
                if let Some(t) = e.get("aiTitle").and_then(Value::as_str) {
                    self.title = Some(t.to_owned());
                }
            }
            Some("user") => {
                if let Some(text) = prompt_text(e) {
                    self.start_turn(e, text, ts.unwrap_or(0));
                }
            }
            Some("assistant") => self.assistant(e),
            _ => {}
        }
    }

    fn start_turn(&mut self, e: &Value, text: String, ts: i64) {
        if self.cwd.is_none() {
            self.cwd = e.get("cwd").and_then(Value::as_str).map(str::to_owned);
        }
        if self.branch.is_none() {
            self.branch = e.get("gitBranch").and_then(Value::as_str).filter(|b| !b.is_empty()).map(str::to_owned);
        }
        self.flush();
        let id = e.get("uuid").and_then(Value::as_str).unwrap_or_default().to_owned();
        self.current = Some(Turn { id, index: self.turns.len(), ts, user: text, ..Default::default() });
    }

    fn assistant(&mut self, e: &Value) {
        let Some(turn) = self.current.as_mut() else { return };
        let Some(blocks) = e.pointer("/message/content").and_then(Value::as_array) else { return };
        for block in blocks {
            match block.get("type").and_then(Value::as_str) {
                Some("text") => {
                    if let Some(t) = block.get("text").and_then(Value::as_str).map(str::trim).filter(|t| !t.is_empty()) {
                        if !turn.assistant.is_empty() {
                            turn.assistant.push_str("\n\n");
                        }
                        turn.assistant.push_str(t);
                    }
                }
                Some("tool_use") => {
                    let name = block.get("name").and_then(Value::as_str).unwrap_or("tool");
                    let call = summarise(name, block.get("input").unwrap_or(&Value::Null));
                    turn.tools.push(call.summary);
                    if let Some(f) = call.edited_file {
                        push_unique(&mut turn.files, f);
                    }
                }
                _ => {} // thinking: private reasoning, not indexed
            }
        }
    }

    fn flush(&mut self) {
        if let Some(t) = self.current.take() {
            self.turns.push(t);
        }
    }

    fn finish(mut self, id: &str, parent_id: Option<String>) -> Result<Session> {
        self.flush();
        if self.turns.is_empty() {
            return Err(SourceError::Empty(id.to_owned()));
        }
        let cwd = self.cwd.unwrap_or_default();
        Ok(Session {
            source: Source::Claude,
            id: id.to_owned(),
            parent_id,
            project: project::key(&cwd),
            cwd,
            title: self.title,
            branch: self.branch,
            started_at: self.first_ts.unwrap_or(0),
            updated_at: self.last_ts.unwrap_or(0),
            turns: self.turns,
        })
    }
}

/// The prompt text if this `user` event is something a person (or a parent agent) asked;
/// `None` for tool results, harness notices, slash-command echoes and compaction summaries.
fn prompt_text(e: &Value) -> Option<String> {
    let flag = |k: &str| e.get(k).and_then(Value::as_bool).unwrap_or(false);
    if flag("isMeta") || flag("isCompactSummary") {
        return None;
    }
    // Subagent prompts carry no origin; anything else non-human is automation.
    if let Some(kind) = e.pointer("/origin/kind").and_then(Value::as_str) {
        if kind != "human" {
            return None;
        }
    }
    let raw = match e.pointer("/message/content")? {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => {
            if blocks.iter().any(|b| b.get("type").and_then(Value::as_str) == Some("tool_result")) {
                return None;
            }
            blocks
                .iter()
                .filter_map(|b| b.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n")
        }
        _ => return None,
    };
    let text = strip_tag(&raw, "system-reminder");
    let text = text.trim();
    const HARNESS: [&str; 4] = ["<command-name>", "<local-command", "<task-notification>", "[Request interrupted"];
    if text.is_empty() || HARNESS.iter().any(|p| text.starts_with(p)) {
        return None;
    }
    Some(text.to_owned())
}

/// Removes every `<tag>…</tag>` block; an unclosed tag drops the rest of the text.
fn strip_tag(text: &str, tag: &str) -> String {
    let (open, close) = (format!("<{tag}>"), format!("</{tag}>"));
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find(&open) {
        out.push_str(&rest[..start]);
        match rest[start..].find(&close) {
            Some(end) => rest = &rest[start + end + close.len()..],
            None => return out,
        }
    }
    out.push_str(rest);
    out
}

fn parse_ts(s: &str) -> Option<i64> {
    s.parse::<jiff::Timestamp>().ok().map(|t| t.as_millisecond())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn jsonl(events: &[Value]) -> String {
        events.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("\n")
    }

    fn prompt(uuid: &str, ts: &str, text: &str) -> Value {
        json!({"type": "user", "uuid": uuid, "timestamp": ts, "cwd": "Z:\\no\\such\\Proj",
               "gitBranch": "main", "origin": {"kind": "human"},
               "message": {"role": "user", "content": text}})
    }

    fn assistant(blocks: Value) -> Value {
        json!({"type": "assistant", "timestamp": "2026-09-25T10:00:05Z",
               "message": {"role": "assistant", "content": blocks}})
    }

    fn tool_result() -> Value {
        json!({"type": "user", "timestamp": "2026-09-25T10:00:06Z",
               "message": {"content": [{"type": "tool_result", "content": "huge output"}]}})
    }

    fn parse_events(events: &[Value]) -> Session {
        parse("s1", jsonl(events).as_bytes()).unwrap()
    }

    #[test]
    fn groups_prompt_with_following_assistant_text_and_tools() {
        let s = parse_events(&[
            prompt("u1", "2026-09-25T10:00:00Z", "fix the login bug"),
            assistant(json!([{"type": "thinking", "thinking": "secret plan"},
                             {"type": "text", "text": "Looking at auth."},
                             {"type": "tool_use", "name": "Edit", "input": {"file_path": "src/auth.rs"}}])),
            tool_result(),
            assistant(json!([{"type": "text", "text": "Fixed."}])),
            prompt("u2", "2026-09-25T10:05:00Z", "now add a test"),
            assistant(json!([{"type": "tool_use", "name": "Write", "input": {"file_path": "tests/auth.rs"}}])),
        ]);
        assert_eq!(s.turns.len(), 2);
        let t = &s.turns[0];
        assert_eq!((t.id.as_str(), t.index), ("u1", 0));
        assert_eq!(t.user, "fix the login bug");
        assert_eq!(t.assistant, "Looking at auth.\n\nFixed.");
        assert_eq!(t.tools, ["Edit src/auth.rs"]);
        assert_eq!(t.files, ["src/auth.rs"]);
        assert!(!t.assistant.contains("secret plan"), "thinking must not be indexed");
        assert_eq!(s.turns[1].files, ["tests/auth.rs"]);
        assert_eq!(s.turns[1].index, 1);
    }

    #[test]
    fn session_metadata_comes_from_events() {
        let s = parse_events(&[
            json!({"type": "ai-title", "aiTitle": "Old title"}),
            prompt("u1", "2026-09-25T10:00:00Z", "hello"),
            json!({"type": "ai-title", "aiTitle": "Login bug fix"}),
            assistant(json!([{"type": "text", "text": "hi"}])),
        ]);
        assert_eq!(s.title.as_deref(), Some("Login bug fix"));
        assert_eq!(s.cwd, "Z:\\no\\such\\Proj");
        assert_eq!(s.project, "z:/no/such/proj");
        assert_eq!(s.branch.as_deref(), Some("main"));
        assert_eq!(s.started_at, 1_790_330_400_000);
        assert_eq!(s.updated_at, 1_790_330_405_000);
        assert_eq!(s.turns[0].ts, 1_790_330_400_000);
        assert!(!s.is_subagent());
    }

    #[test]
    fn harness_and_automation_messages_are_not_prompts() {
        let meta = json!({"type": "user", "isMeta": true, "message": {"content": "<local-command-caveat>x"}});
        let compact = json!({"type": "user", "isCompactSummary": true, "message": {"content": "This session is being continued"}});
        let notif = json!({"type": "user", "origin": {"kind": "task-notification"}, "message": {"content": "done"}});
        let cmd = json!({"type": "user", "message": {"content": "<command-name>/model</command-name>"}});
        let interrupted = json!({"type": "user", "message": {"content": [{"type": "text", "text": "[Request interrupted by user]"}]}});
        let only_reminder = json!({"type": "user", "message": {"content": "<system-reminder>ctx</system-reminder>  "}});
        let s = parse_events(&[
            prompt("u1", "2026-09-25T10:00:00Z", "real question"),
            meta, compact, notif, cmd, interrupted, only_reminder,
            assistant(json!([{"type": "text", "text": "answer"}])),
        ]);
        assert_eq!(s.turns.len(), 1);
        assert_eq!(s.turns[0].assistant, "answer");
    }

    #[test]
    fn system_reminders_are_stripped_from_prompts() {
        let s = parse_events(&[prompt(
            "u1",
            "2026-09-25T10:00:00Z",
            "<system-reminder>internal</system-reminder>please deploy<system-reminder>more</system-reminder>",
        )]);
        assert_eq!(s.turns[0].user, "please deploy");
    }

    #[test]
    fn prompt_without_origin_is_accepted_for_subagents() {
        let sub = json!({"type": "user", "uuid": "a1", "isSidechain": true, "cwd": "Z:\\p",
                         "message": {"content": "Explore the repo"}});
        let s = parse("parent-sid/agent-abc", jsonl(&[sub, assistant(json!([{"type": "text", "text": "found it"}]))]).as_bytes()).unwrap();
        assert_eq!(s.parent_id.as_deref(), Some("parent-sid"));
        assert!(s.is_subagent());
        assert_eq!(s.turns[0].user, "Explore the repo");
    }

    #[test]
    fn assistant_output_before_first_prompt_is_ignored() {
        let s = parse_events(&[
            assistant(json!([{"type": "text", "text": "orphan"}])),
            prompt("u1", "2026-09-25T10:00:00Z", "q"),
        ]);
        assert_eq!(s.turns.len(), 1);
        assert_eq!(s.turns[0].assistant, "");
    }

    #[test]
    fn corrupt_and_truncated_lines_are_skipped() {
        let text = format!(
            "{}\nnot json\n{}\n{{\"type\": \"assistant\", \"mess",
            prompt("u1", "2026-09-25T10:00:00Z", "q"),
            assistant(json!([{"type": "text", "text": "a"}]))
        );
        let s = parse("s1", text.as_bytes()).unwrap();
        assert_eq!(s.turns[0].assistant, "a");
    }

    #[test]
    fn session_without_prompts_is_empty_error() {
        let e = parse("s1", jsonl(&[json!({"type": "mode"})]).as_bytes()).unwrap_err();
        assert!(matches!(e, SourceError::Empty(id) if id == "s1"));
    }

    #[test]
    fn files_are_deduplicated_in_first_touch_order() {
        let s = parse_events(&[
            prompt("u1", "2026-09-25T10:00:00Z", "q"),
            assistant(json!([
                {"type": "tool_use", "name": "Edit", "input": {"file_path": "b.rs"}},
                {"type": "tool_use", "name": "Edit", "input": {"file_path": "a.rs"}},
                {"type": "tool_use", "name": "Edit", "input": {"file_path": "b.rs"}}])),
        ]);
        assert_eq!(s.turns[0].files, ["b.rs", "a.rs"]);
        assert_eq!(s.turns[0].tools.len(), 3);
    }

    #[test]
    fn strip_tag_handles_unclosed_tag() {
        assert_eq!(strip_tag("keep<x>drop", "x"), "keep");
        assert_eq!(strip_tag("a<x>1</x>b<x>2</x>c", "x"), "abc");
        assert_eq!(strip_tag("no tags", "x"), "no tags");
    }

    #[test]
    fn discover_finds_main_and_subagent_transcripts_with_fingerprints() {
        let tmp = tempfile::tempdir().unwrap();
        let proj = tmp.path().join("D--dev-x");
        fs::create_dir_all(proj.join("sid1/subagents")).unwrap();
        fs::create_dir_all(proj.join("sid1/tool-results")).unwrap();
        fs::create_dir_all(proj.join("memory")).unwrap();
        fs::write(proj.join("sid1.jsonl"), "{}").unwrap();
        fs::write(proj.join("sid1/subagents/agent-a.jsonl"), "{}").unwrap();
        fs::write(proj.join("sid1/subagents/agent-a.meta.json"), "{}").unwrap();
        fs::write(proj.join("memory/MEMORY.md"), "x").unwrap();

        let refs = discover(tmp.path()).unwrap();
        let ids: Vec<_> = refs.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(ids, ["sid1", "sid1/agent-a"]);
        assert!(refs[0].fingerprint.starts_with("2-"));

        fs::write(proj.join("sid1.jsonl"), "{}\n{}").unwrap();
        let again = discover(tmp.path()).unwrap();
        assert_ne!(again[0].fingerprint, refs[0].fingerprint, "growth must change the fingerprint");
    }

    #[test]
    fn load_reads_the_file_behind_a_ref() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("s.jsonl");
        fs::write(&p, jsonl(&[prompt("u1", "2026-09-25T10:00:00Z", "q")])).unwrap();
        let r = SessionRef { source: Source::Claude, id: "s".into(), fingerprint: String::new(), locator: Locator::File(p) };
        assert_eq!(load(&r).unwrap().turns[0].user, "q");
    }
}
