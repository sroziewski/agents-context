//! Queries over the sessions collection, shared by the CLI and the MCP tools.

use std::sync::{Arc, Mutex};

use embed::Embedder;
use serde_json::Value;
use store::{Hit, SearchFilter, SearchMode, Store};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Scope {
    /// One project key (see `sources::project::key`).
    Project(String),
    All,
}

#[derive(Debug, Clone)]
pub struct Query {
    pub text: String,
    pub scope: Scope,
    pub source: Option<String>,
    pub since: Option<i64>,
    pub until: Option<i64>,
    pub limit: u64,
    /// Subagent transcripts are mostly exploration noise, so they are opt-in.
    pub include_subagents: bool,
}

#[derive(Clone)]
pub struct Context {
    pub store: Arc<Store>,
    pub embedder: Arc<Mutex<Box<dyn Embedder>>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SessionSummary {
    pub source: String,
    pub id: String,
    pub project: String,
    pub title: String,
    pub ts: i64,
}

/// Search results: matching session summaries (the gist) and matching turns (the detail).
#[derive(Debug, Clone, Default)]
pub struct Results {
    pub summaries: Vec<Hit>,
    pub turns: Vec<Hit>,
}

/// Summaries shown above the turns; a few are enough to orient, the turns carry the detail.
const MAX_SUMMARIES: u64 = 3;

impl Context {
    pub async fn search(&self, q: &Query) -> anyhow::Result<Results> {
        let embedder = self.embedder.clone();
        let text = q.text.clone();
        // Embedding blocks (HTTP to the daemon or a local model); keep it off async workers.
        let query = tokio::task::spawn_blocking(move || {
            embedder.lock().unwrap_or_else(|p| p.into_inner()).embed_one(&text)
        })
        .await??;
        let filter_for = |kind: &str| {
            let mut f = scope_filter(&q.scope, kind).between(q.since, q.until);
            if let Some(source) = &q.source {
                f = f.eq("source", source.clone());
            }
            if !q.include_subagents {
                f = f.flag("subagent", false);
            }
            f
        };
        let (summary_filter, turn_filter) = (filter_for("summary"), filter_for("turn"));
        let summaries = self.store.search(&query, &summary_filter, MAX_SUMMARIES.min(q.limit), SearchMode::Hybrid);
        let turns = self.store.search(&query, &turn_filter, q.limit, SearchMode::Hybrid);
        let (summaries, turns) = tokio::try_join!(summaries, turns)?;
        Ok(Results { summaries, turns })
    }

    /// All chunks of one session in conversation order. A session id is explicit, so it is not
    /// limited to the current project.
    pub async fn session(&self, id: &str) -> anyhow::Result<Vec<Hit>> {
        let mut hits = self.store.scroll(&SearchFilter::default().eq("session_id", id), 2000).await?;
        hits.sort_by_key(|h| (int(&h.payload, "turn_index"), int(&h.payload, "chunk_index")));
        Ok(hits)
    }

    /// Sessions newest first, one entry per session (its first chunk).
    pub async fn sessions(&self, scope: &Scope, limit: usize) -> anyhow::Result<Vec<SessionSummary>> {
        let filter = scope_filter(scope, "turn").int("turn_index", 0).int("chunk_index", 0);
        let mut out: Vec<SessionSummary> = self
            .store
            .scroll(&filter, 5000)
            .await?
            .into_iter()
            .map(|h| SessionSummary {
                source: string(&h.payload, "source"),
                id: string(&h.payload, "session_id"),
                project: string(&h.payload, "project"),
                title: string(&h.payload, "title"),
                ts: int(&h.payload, "ts"),
            })
            .collect();
        out.sort_by(|a, b| b.ts.cmp(&a.ts).then_with(|| a.id.cmp(&b.id)));
        out.truncate(limit);
        Ok(out)
    }
}

fn scope_filter(scope: &Scope, kind: &str) -> SearchFilter {
    let f = SearchFilter::default().eq("kind", kind);
    match scope {
        Scope::Project(p) => f.eq("project", p.clone()),
        Scope::All => f,
    }
}

fn string(p: &serde_json::Map<String, Value>, k: &str) -> String {
    p.get(k).and_then(Value::as_str).unwrap_or_default().to_owned()
}

fn int(p: &serde_json::Map<String, Value>, k: &str) -> i64 {
    p.get(k).and_then(Value::as_i64).unwrap_or_default()
}

/// `YYYY-MM-DD` in local time → epoch ms at the start of that day, or at its last millisecond
/// when `end_of_day` (so `until` includes the whole day).
pub fn parse_date(s: &str, end_of_day: bool) -> anyhow::Result<i64> {
    let date: jiff::civil::Date = s.trim().parse().map_err(|_| anyhow::anyhow!("invalid date {s:?}, expected YYYY-MM-DD"))?;
    let start = date.to_zoned(jiff::tz::TimeZone::system())?.timestamp().as_millisecond();
    Ok(if end_of_day { start + 86_400_000 - 1 } else { start })
}

/// Summaries first under their own heading, then turns. Without summaries the output is just
/// the numbered turns.
pub fn format_results(r: &Results) -> String {
    if r.summaries.is_empty() {
        return format_hits(&r.turns);
    }
    let summaries = r
        .summaries
        .iter()
        .enumerate()
        .map(|(i, h)| {
            let p = &h.payload;
            format!("[S{}] {} · {} · session {}\n{}", i + 1, string(p, "source"), string(p, "project"), string(p, "session_id"), string(p, "text").trim())
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    let turns = if r.turns.is_empty() { "No matching turns.".to_owned() } else { format_hits(&r.turns) };
    format!("## Session summaries (machine-written; verify against the turns)\n\n{summaries}\n\n## Matching turns\n\n{turns}")
}

pub fn format_hits(hits: &[Hit]) -> String {
    if hits.is_empty() {
        return "No matching sessions.".into();
    }
    hits.iter()
        .enumerate()
        .map(|(i, h)| {
            let p = &h.payload;
            format!(
                "[{}] {} · {} · session {} · turn {}\n{}",
                i + 1,
                string(p, "source"),
                string(p, "project"),
                string(p, "session_id"),
                int(p, "turn_index") + 1,
                string(p, "text").trim()
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// The session as one text, capped at `max_chars` so a 400-turn session cannot flood a context
/// window; the cut is reported so the caller can narrow down with search instead.
pub fn format_session(hits: &[Hit], max_chars: usize) -> String {
    if hits.is_empty() {
        return "Session not found.".into();
    }
    let mut out = String::new();
    for (i, h) in hits.iter().enumerate() {
        let label = if string(&h.payload, "kind") == "summary" {
            "summary".to_owned()
        } else {
            format!("turn {}", int(&h.payload, "turn_index") + 1)
        };
        let piece = format!("--- {label} ---\n{}\n\n", string(&h.payload, "text").trim());
        if out.len() + piece.len() > max_chars {
            out.push_str(&format!("[truncated: {} of {} chunks shown; use search_sessions to find the rest]", i, hits.len()));
            return out;
        }
        out.push_str(&piece);
    }
    out.trim_end().to_owned()
}

pub fn format_sessions(list: &[SessionSummary]) -> String {
    if list.is_empty() {
        return "No sessions indexed for this scope.".into();
    }
    list.iter()
        .map(|s| {
            let date = jiff::Timestamp::from_millisecond(s.ts).map(|t| t.strftime("%Y-%m-%d").to_string()).unwrap_or_default();
            let title = if s.title.is_empty() { "(untitled)" } else { &s.title };
            format!("{date}  {:<8}  {title}  [{}]  {}", s.source, s.id, s.project)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn hit(v: Value) -> Hit {
        Hit { id: uuid::Uuid::nil(), score: 1.0, payload: v.as_object().cloned().unwrap() }
    }

    #[test]
    fn parse_date_spans_the_whole_local_day() {
        let start = parse_date("2026-09-25", false).unwrap();
        let end = parse_date("2026-09-25", true).unwrap();
        assert_eq!(end - start, 86_400_000 - 1);
        assert!(parse_date("25.09.2026", false).is_err());
        assert!(parse_date(" 2026-09-25 ", false).is_ok());
    }

    #[test]
    fn format_hits_numbers_results_with_provenance() {
        let out = format_hits(&[hit(json!({"source": "claude", "project": "d:/p", "session_id": "s1", "turn_index": 2, "text": "[T · 2026-09-25]\nUser: q"}))]);
        assert_eq!(out, "[1] claude · d:/p · session s1 · turn 3\n[T · 2026-09-25]\nUser: q");
        assert_eq!(format_hits(&[]), "No matching sessions.");
    }

    #[test]
    fn format_session_truncates_with_notice() {
        let hits: Vec<Hit> = (0..10).map(|i| hit(json!({"turn_index": i, "text": "x".repeat(100)}))).collect();
        let out = format_session(&hits, 350);
        assert!(out.contains("--- turn 1 ---"));
        assert!(out.contains("[truncated: 2 of 10 chunks shown"));
        let full = format_session(&hits[..1], 10_000);
        assert!(!full.contains("truncated"));
        assert_eq!(format_session(&[], 10), "Session not found.");
    }

    #[test]
    fn format_sessions_marks_untitled() {
        let s = SessionSummary { source: "claude".into(), id: "s1".into(), project: "d:/p".into(), title: String::new(), ts: 0 };
        assert!(format_sessions(&[s]).contains("(untitled)"));
        assert_eq!(format_sessions(&[]), "No sessions indexed for this scope.");
    }

    #[test]
    fn scope_filter_restricts_to_one_kind() {
        assert_eq!(scope_filter(&Scope::All, "turn").equals, [("kind".to_string(), "turn".to_string())]);
        assert_eq!(scope_filter(&Scope::Project("d:/p".into()), "summary").equals.len(), 2);
    }

    #[test]
    fn results_without_summaries_look_like_plain_hits() {
        let turns = vec![hit(json!({"source": "claude", "project": "d:/p", "session_id": "s1", "turn_index": 0, "text": "t"}))];
        let r = Results { summaries: vec![], turns: turns.clone() };
        assert_eq!(format_results(&r), format_hits(&turns));
    }

    #[test]
    fn summaries_come_first_under_their_own_heading() {
        let r = Results {
            summaries: vec![hit(json!({"source": "claude", "project": "d:/p", "session_id": "s1", "text": "Session summary\n## Goal\n- x"}))],
            turns: vec![],
        };
        let out = format_results(&r);
        assert!(out.starts_with("## Session summaries (machine-written; verify against the turns)\n\n[S1] claude · d:/p · session s1\nSession summary"), "{out}");
        assert!(out.ends_with("## Matching turns\n\nNo matching turns."), "{out}");
    }

    #[test]
    fn session_text_labels_the_summary() {
        let hits = vec![
            hit(json!({"kind": "summary", "turn_index": -1, "text": "gist"})),
            hit(json!({"kind": "turn", "turn_index": 0, "text": "first"})),
        ];
        let out = format_session(&hits, 10_000);
        assert!(out.starts_with("--- summary ---\ngist\n\n--- turn 1 ---\nfirst"), "{out}");
    }
}
