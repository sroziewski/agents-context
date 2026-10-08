//! The `/clear` handoff (DEV-58) and the status line (DEV-59) through the real `ctx` binary, fed
//! the JSON Claude Code sends: SessionEnd(clear) then SessionStart(clear) must hand the cleared
//! session over; the status line must render from its stdin.

use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use serde_json::{Value, json};

struct Env {
    tmp: tempfile::TempDir,
}

impl Env {
    fn new() -> Self {
        Self { tmp: tempfile::tempdir().unwrap() }
    }

    fn data(&self) -> std::path::PathBuf {
        self.tmp.path().join("data")
    }

    /// The binary isolated from the real install: its own data dir, no daemon, no Qdrant, no Teams.
    fn ctx(&self) -> Command {
        let mut c = Command::new(env!("CARGO_BIN_EXE_ctx"));
        c.env("CTX_DATA_DIR", self.data())
            .env("CTX_EMBED_URL", "http://127.0.0.1:9")
            .env("CTX_QDRANT_URL", "http://127.0.0.1:9")
            .env("CTX_TEAMS_DB", self.tmp.path().join("no-teams.db"))
            .env("CTX_REDACT_FILE", self.tmp.path().join("no-redact.txt"))
            .env_remove("CTX_SUMMARIZER")
            .env_remove("CTX_HANDOFF")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        c
    }

    fn run(&self, args: &[&str], stdin: &str, extra_env: &[(&str, &str)]) -> (i32, String) {
        let mut cmd = self.ctx();
        cmd.args(args);
        for (k, v) in extra_env {
            cmd.env(k, v);
        }
        let mut child = cmd.spawn().unwrap();
        child.stdin.take().unwrap().write_all(stdin.as_bytes()).unwrap();
        let out = child.wait_with_output().unwrap();
        (out.status.code().unwrap_or(-1), String::from_utf8_lossy(&out.stdout).into_owned())
    }
}

/// A Claude Code transcript as written on disk: prompts, an edit, a final answer.
fn transcript(path: &Path, cwd: &str) {
    let lines = [
        json!({"type": "user", "uuid": "u1", "timestamp": "2026-10-08T10:00:00Z", "cwd": cwd, "message": {"role": "user", "content": "fix the FTS fallback in teams search"}}),
        json!({"type": "assistant", "timestamp": "2026-10-08T10:01:00Z", "message": {"content": [
            {"type": "tool_use", "name": "Edit", "input": {"file_path": "D:\\work\\p\\src\\teams.rs", "old_string": "a", "new_string": "b"}},
            {"type": "text", "text": "Fixed the fallback; tests pass."}]}}),
        json!({"type": "user", "uuid": "u2", "timestamp": "2026-10-08T10:05:00Z", "cwd": cwd, "message": {"role": "user", "content": "now add a test for the unclosed quote case"}}),
        json!({"type": "user", "timestamp": "2026-10-08T10:05:10Z", "message": {"content": [{"type": "tool_result", "content": "ok"}]}}),
        json!({"type": "assistant", "timestamp": "2026-10-08T10:06:00Z", "message": {"content": [{"type": "text", "text": "Added test_unclosed_quote. Open: the NEAR( case is untested."}]}}),
    ];
    let text: Vec<String> = lines.iter().map(Value::to_string).collect();
    std::fs::write(path, text.join("\n")).unwrap();
}

fn event(name: &str, kind_key: &str, kind: &str, session: &str, transcript: &Path, cwd: &str) -> String {
    json!({"hook_event_name": name, kind_key: kind, "session_id": session, "transcript_path": transcript, "cwd": cwd}).to_string()
}

#[test]
fn clear_hands_the_cleared_session_over_to_the_new_one() {
    let e = Env::new();
    let proj = e.tmp.path().join("projects").join("D--work-p");
    std::fs::create_dir_all(&proj).unwrap();
    let (old, new) = (proj.join("old.jsonl"), proj.join("new.jsonl"));
    let cwd = r"D:\work\p";
    transcript(&old, cwd);
    std::fs::write(&new, "").unwrap();

    let (code, out) = e.run(&["hook", "--source", "claude"], &event("SessionEnd", "reason", "clear", "old", &old, cwd), &[]);
    assert_eq!((code, out.as_str()), (0, ""), "the end hook prints nothing");
    assert!(e.data().join("handoff.json").is_file(), "the clear is recorded in the data dir");

    let (code, out) = e.run(&["hook", "--source", "claude"], &event("SessionStart", "source", "clear", "new", &new, cwd), &[]);
    assert_eq!(code, 0);
    let v: Value = serde_json::from_str(out.trim()).unwrap_or_else(|err| panic!("{err}: {out}"));
    assert_eq!(v["hookSpecificOutput"]["hookEventName"], "SessionStart");
    let ctx = v["hookSpecificOutput"]["additionalContext"].as_str().unwrap();
    assert!(ctx.starts_with("<previous-session>") && ctx.ends_with("</previous-session>"), "{ctx}");
    assert!(ctx.contains("Session old ·") && ctx.contains("2 turns"), "{ctx}");
    assert!(ctx.contains("- fix the FTS fallback in teams search\n- now add a test for the unclosed quote case\n"), "{ctx}");
    assert!(ctx.contains("teams.rs"), "edited files: {ctx}");
    assert!(ctx.contains("Open: the NEAR( case is untested."), "the last answer: {ctx}");
    assert!(!ctx.contains("ctx summary"), "Qdrant is unreachable, so no summary — and no failure");
}

#[test]
fn only_clear_starts_get_a_handoff_and_it_can_be_turned_off() {
    let e = Env::new();
    let proj = e.tmp.path().join("p");
    std::fs::create_dir_all(&proj).unwrap();
    let (old, new) = (proj.join("old.jsonl"), proj.join("new.jsonl"));
    transcript(&old, "D:\\w");
    std::fs::write(&new, "").unwrap();
    for source in ["startup", "resume", "compact"] {
        let (code, out) = e.run(&["hook", "--source", "claude"], &event("SessionStart", "source", source, "new", &new, "D:\\w"), &[]);
        assert_eq!((code, out.as_str()), (0, ""), "{source}");
    }
    // Without a record the newest sibling transcript is used...
    let start = event("SessionStart", "source", "clear", "new", &new, "D:\\w");
    let (_, out) = e.run(&["hook", "--source", "claude"], &start, &[]);
    assert!(out.contains("Session old"), "{out}");
    // ...unless the user turned handoff off.
    let (code, out) = e.run(&["hook", "--source", "claude"], &start, &[("CTX_HANDOFF", "off")]);
    assert_eq!((code, out.as_str()), (0, ""));
}

#[test]
fn hook_survives_garbage_and_a_missing_transcript() {
    let e = Env::new();
    for stdin in ["not json", "", r#"{"hook_event_name":"SessionStart","source":"clear","transcript_path":"Z:\\nope\\x.jsonl"}"#] {
        let (code, out) = e.run(&["hook", "--source", "claude"], stdin, &[]);
        assert_eq!((code, out.as_str()), (0, ""), "{stdin}");
    }
}

#[test]
fn statusline_renders_usage_and_flags_the_down_daemon() {
    let e = Env::new();
    let input = r#"{"model":{"display_name":"Opus 5.5"},"context_window":{"used_percentage":81.2}}"#;
    let (code, out) = e.run(&["statusline"], input, &[]);
    assert_eq!(code, 0);
    let line = out.trim_end();
    assert!(line.starts_with("Opus 5.5 · \x1b[31mctx ▰▰▰▰▰▰▰▰▱▱ 81%"), "{line:?}");
    assert!(line.contains("/clear → ctx hands over") && line.contains("ctx daemon down"), "{line:?}");
    assert!(!line.contains("Teams"), "no teams.db, no Teams marker: {line:?}");

    let (_, quiet) = e.run(&["statusline"], input, &[("CTX_STATUS_CRIT", "90"), ("CTX_STATUS_WARN", "85"), ("CTX_HANDOFF", "off")]);
    assert!(quiet.starts_with("Opus 5.5 · ctx ▰▰▰▰▰▰▰▰▱▱ 81%"), "thresholds come from config: {quiet:?}");
    let (code, out) = e.run(&["statusline"], "garbage", &[]);
    assert_eq!((code, out.trim_end()), (0, "\x1b[31mctx daemon down\x1b[0m"));
}
