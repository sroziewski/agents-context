//! Jira issues → chunks for the `jira` collection, indexed incrementally by `updated` watermark.
//!
//! One `issue` chunk (key, type, status, summary, description) answers "what is this issue
//! about"; one `comment` chunk per comment, prefixed with the issue title, answers "who said
//! what on it" — the same split as Teams threads/messages. All text is redacted.

use embed::Embedder;
use serde_json::{Map, Value, json};
use sources::jira::{Jira, JiraIssue};
use store::{Field, SearchFilter, Store};

use crate::chunk::{Chunk, pack, split};
use crate::indexer::{IndexConfig, IndexError, Report, write_chunks};
use crate::state::{Record, State};
use crate::tokens::TokenCounter;

pub const SOURCE: &str = "jira";
/// Meta key for the delta watermark (max `updated` epoch ms indexed so far).
const WATERMARK: &str = "jira/watermark_ms";
/// JQL `updated` is minute-granular; widen the delta window so a same-minute edit is not missed.
const MARGIN_MS: i64 = 120_000;
const PAGE: u32 = 50;
const CONTEXT_CHARS: usize = 150;

/// Payload fields the jira collection indexes. `project` is the Jira key (HD, SEDN26), so search
/// can scope by it like the session project tenant.
pub const JIRA_FIELDS: &[Field] = &[
    Field::Tenant("project"),
    Field::Keyword("kind"),
    Field::Keyword("issue_key"),
    Field::Keyword("status"),
    Field::Keyword("issue_type"),
    Field::Keyword("author"),
];

fn first_line(text: &str, max: usize) -> String {
    let line = text.lines().find(|l| !l.trim().is_empty()).unwrap_or("").trim();
    if line.chars().count() > max { format!("{}…", line.chars().take(max).collect::<String>()) } else { line.to_owned() }
}

/// Best-effort parse of a Jira `updated` timestamp (e.g. `2026-09-27T10:00:00.000+0200`) to
/// epoch ms. `None` on an unexpected shape — the watermark just does not advance for it.
pub fn updated_ms(s: &str) -> Option<i64> {
    for fmt in ["%Y-%m-%dT%H:%M:%S%.f%z", "%Y-%m-%dT%H:%M:%S%z"] {
        if let Ok(z) = jiff::Zoned::strptime(fmt, s) {
            return Some(z.timestamp().as_millisecond());
        }
    }
    None
}

/// `project in (...)` plus, when a watermark exists, `updated >= "<watermark − margin>"` in the
/// local calendar (Jira compares in the user's tz; same machine, so the system tz matches).
pub fn sync_jql(keys: &[String], since_ms: Option<i64>) -> String {
    let scope = format!("project in ({})", keys.join(", "));
    let time = since_ms
        .and_then(|ms| jiff::Timestamp::from_millisecond((ms - MARGIN_MS).max(0)).ok())
        .map(|ts| format!(" AND updated >= \"{}\"", ts.to_zoned(jiff::tz::TimeZone::system()).strftime("%Y-%m-%d %H:%M")))
        .unwrap_or_default();
    format!("{scope}{time} ORDER BY updated ASC")
}

fn base_payload(issue: &JiraIssue, kind: &str, text: &str) -> Map<String, Value> {
    let project = issue.key.split('-').next().unwrap_or("").to_owned();
    let v = json!({
        "kind": kind,
        "source": SOURCE,
        "project": project,
        "issue_key": issue.key,
        "status": issue.status,
        "issue_type": issue.issue_type,
        "text": text,
        "ts": updated_ms(&issue.updated).unwrap_or(0),
    });
    let Value::Object(m) = v else { unreachable!("json! object") };
    m
}

/// All chunks of one issue, redacted. Returns `(chunks, redactions)`.
pub fn jira_chunks(issue: &JiraIssue, counter: &dyn TokenCounter, cfg: &IndexConfig) -> (Vec<Chunk>, usize) {
    let redactor = &cfg.chunk.redactor;
    let mut redactions = 0;
    let mut scrub = |s: &str| {
        let (clean, n) = redactor.scrub(s);
        redactions += n;
        clean
    };
    let summary = scrub(&issue.summary);
    let description = scrub(&issue.description);
    let header = format!("[{} · {} · {}] {}\n", issue.key, issue.issue_type, issue.status, summary);
    let budget = cfg.chunk.max_tokens.saturating_sub(counter.count(&header)).max(16);

    let mut chunks = Vec::new();
    // Issue chunk(s): header + description.
    let body = if description.trim().is_empty() { summary.clone() } else { description };
    let pieces = pack(split(body.trim(), budget, counter), budget, cfg.chunk.overlap_tokens, counter);
    let total = pieces.len().max(1);
    for (i, piece) in pieces.into_iter().enumerate() {
        let text = format!("{header}{piece}");
        let mut p = base_payload(issue, "issue", &text);
        p.insert("chunk_index".into(), json!(i));
        p.insert("chunk_count".into(), json!(total));
        chunks.push(Chunk { id: store::point_id(SOURCE, &format!("{}/i{i}", issue.key)), tokens: counter.count(&text), payload: p, text });
    }

    // Comment chunks, each with the issue title as context.
    let title = first_line(&summary, CONTEXT_CHARS);
    for (k, c) in issue.comments.iter().enumerate() {
        let clean = scrub(&c.body);
        let lead = format!("{} · comment by {}: ", header.trim_end(), c.author);
        let context = format!("Issue: {title}\n");
        let room = cfg.chunk.max_tokens.saturating_sub(counter.count(&lead) + counter.count(&context)).max(16);
        for (i, piece) in pack(split(clean.trim(), room, counter), room, 0, counter).into_iter().enumerate() {
            let text = format!("{lead}\n{context}{piece}");
            let mut p = base_payload(issue, "comment", &text);
            p.insert("author".into(), json!(c.author));
            p.insert("chunk_index".into(), json!(i));
            chunks.push(Chunk { id: store::point_id(SOURCE, &format!("{}/c{k}/{i}", issue.key)), tokens: counter.count(&text), payload: p, text });
        }
    }
    (chunks, redactions)
}

/// Indexes issues changed since the watermark (deltas only), across `keys`, paginating the Jira
/// search. Unchanged issues (same `updated`) are skipped. The watermark advances to the newest
/// `updated` seen. One failing issue is reported and retried next run.
#[allow(clippy::too_many_arguments)]
pub async fn sync_jira(
    store: &Store,
    embedder: &mut dyn Embedder,
    counter: &dyn TokenCounter,
    state: &State,
    client: &Jira,
    keys: &[String],
    config: &IndexConfig,
    now_ms: i64,
) -> Result<Report, IndexError> {
    if keys.is_empty() {
        return Ok(Report::default());
    }
    let since = state.get_meta(WATERMARK).ok().flatten().and_then(|s| s.parse::<i64>().ok());
    let jql = sync_jql(keys, since);
    let mut report = Report::default();
    let mut max_updated = since.unwrap_or(0);
    let mut start = 0u32;
    loop {
        let page = client.search_full(&jql, start, PAGE).map_err(|e| IndexError::Source(e.to_string()))?;
        if page.issues.is_empty() {
            break;
        }
        let fetched = page.issues.len() as u32;
        for issue in &page.issues {
            report.scanned += 1;
            max_updated = max_updated.max(updated_ms(&issue.updated).unwrap_or(0));
            let fingerprint = issue.updated.clone();
            if state.get(SOURCE, &issue.key)?.is_some_and(|i| i.fingerprint == fingerprint) {
                report.unchanged += 1;
                continue;
            }
            let (chunks, redactions) = jira_chunks(issue, counter, config);
            let scope = SearchFilter::default().eq("source", SOURCE).eq("issue_key", issue.key.clone());
            let n = match write_chunks(store, embedder, config, &scope, &chunks).await {
                Ok(n) => n,
                Err(e) => {
                    report.failed.push((issue.key.clone(), e.to_string()));
                    continue;
                }
            };
            state.record(
                &Record { source: SOURCE, session_id: &issue.key, fingerprint: &fingerprint, project: "", chunks: n, redactions, turns: issue.comments.len(), subagent: false },
                now_ms,
            )?;
            report.indexed += 1;
            report.chunks += n;
            report.redactions += redactions;
        }
        start += fetched;
        if start >= page.total {
            break;
        }
    }
    // Persist the watermark only after a clean pass, so a mid-run failure re-scans the tail.
    if report.failed.is_empty() && max_updated > 0 {
        let _ = state.set_meta(WATERMARK, &max_updated.to_string());
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokens::WordCounter;
    use sources::jira::JiraComment;

    fn issue(key: &str, comments: &[(&str, &str)]) -> JiraIssue {
        JiraIssue {
            key: key.into(),
            summary: "Login fails after deploy".into(),
            description: "Users cannot log in on prod after the release.".into(),
            status: "Open".into(),
            issue_type: "Bug".into(),
            assignee: "Anna".into(),
            updated: "2026-09-27T10:00:00.000+0200".into(),
            url: format!("https://j/browse/{key}"),
            comments: comments.iter().map(|(a, b)| JiraComment { author: (*a).into(), created: "2026-09-27T11:00:00.000+0200".into(), body: (*b).into() }).collect(),
        }
    }

    #[test]
    fn one_issue_chunk_and_one_per_comment_with_context() {
        let (chunks, _) = jira_chunks(&issue("HD-1", &[("Piotr", "restarted the service"), ("Anna", "ok, works")]), &WordCounter, &IndexConfig::default());
        let kinds: Vec<&str> = chunks.iter().map(|c| c.payload["kind"].as_str().unwrap()).collect();
        assert_eq!(kinds, ["issue", "comment", "comment"]);
        assert_eq!(chunks[0].payload["project"], "HD");
        assert_eq!(chunks[0].payload["issue_key"], "HD-1");
        assert!(chunks[0].text.starts_with("[HD-1 · Bug · Open] Login fails"));
        assert!(chunks[1].text.contains("Issue: Login fails after deploy") && chunks[1].text.contains("restarted the service"));
        assert_eq!(chunks[1].payload["author"], "Piotr");
    }

    #[test]
    fn text_is_redacted() {
        let mut cfg = IndexConfig::default();
        cfg.chunk.redactor = crate::Redactor::new(["Tajne123".to_string()]);
        let mut i = issue("HD-2", &[("A", "hasło Tajne123 tutaj")]);
        i.description = "secret Tajne123 in body".into();
        let (chunks, n) = jira_chunks(&i, &WordCounter, &cfg);
        assert!(n >= 2);
        assert!(chunks.iter().all(|c| !c.text.contains("Tajne123")));
    }

    #[test]
    fn ids_are_stable_and_distinct() {
        let (a, _) = jira_chunks(&issue("HD-3", &[("A", "x")]), &WordCounter, &IndexConfig::default());
        let (b, _) = jira_chunks(&issue("HD-3", &[("A", "x")]), &WordCounter, &IndexConfig::default());
        assert_eq!(a.iter().map(|c| c.id).collect::<Vec<_>>(), b.iter().map(|c| c.id).collect::<Vec<_>>());
        let mut ids: Vec<_> = a.iter().map(|c| c.id).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), a.len());
    }

    #[test]
    fn jql_is_scoped_and_delta_filtered() {
        assert_eq!(sync_jql(&["HD".into(), "SEDN26".into()], None), "project in (HD, SEDN26) ORDER BY updated ASC");
        let ms = updated_ms("2026-09-27T10:00:00.000+0200").unwrap();
        let jql = sync_jql(&["HD".into()], Some(ms));
        assert!(jql.starts_with("project in (HD) AND updated >= \"2026-09-27 ") && jql.ends_with("ORDER BY updated ASC"), "{jql}");
    }

    #[test]
    fn updated_ms_parses_jira_offsets() {
        assert!(updated_ms("2026-09-27T10:00:00.000+0200").is_some());
        assert!(updated_ms("2026-09-27T10:00:00+0000").is_some());
        assert_eq!(updated_ms("garbage"), None);
    }
}
