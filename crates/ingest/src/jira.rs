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
use crate::state::JIRA_WATERMARK as WATERMARK;
/// Meta key for when indexed issues were last checked against Jira for deletions and moves.
const RECONCILED: &str = "jira/reconciled_ms";
const RECONCILE_EVERY_MS: i64 = 24 * 3_600_000;
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

/// Describes an image (e.g. a screenshot) as text so its content is searchable. Implemented in
/// `ctx` by a headless vision model; kept abstract here so the indexer can be tested with a fake.
pub trait ImageDescriber: Send + Sync {
    fn describe(&self, bytes: &[u8], filename: &str, mime: &str) -> Result<String, String>;
}

/// How `sync_jira` turns image attachments into text: the describer and the per-issue / per-file
/// caps. Absent ⇒ images are not fetched or described at all (the default).
pub struct ImageOpts<'a> {
    pub describer: &'a dyn ImageDescriber,
    pub max_images: usize,
    pub max_bytes: usize,
}

/// FNV-1a 64-bit, stable forever and dependency-free — a recorded guard for a cached description,
/// not a security hash.
fn fnv1a_hex(bytes: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}

/// A description for one image: the cache first (no download on a hit), otherwise `fetch` the
/// bytes and `describe` them, caching the result. `None` (logged) on a fetch/describe failure, so
/// a bad image never fails its issue. `fetch` is a closure so this is testable without Jira.
fn resolve_description(
    state: &State,
    describer: &dyn ImageDescriber,
    id: &str,
    filename: &str,
    mime: &str,
    now_ms: i64,
    fetch: impl FnOnce() -> Result<Vec<u8>, String>,
) -> Option<String> {
    if let Ok(Some(desc)) = state.get_image_desc(id) {
        return Some(desc);
    }
    let bytes = match fetch() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[jira-image] download {filename} failed: {e}");
            return None;
        }
    };
    match describer.describe(&bytes, filename, mime) {
        Ok(desc) => {
            let desc = desc.trim().to_owned();
            if desc.is_empty() {
                return None;
            }
            // The cache sits outside the index; keep secrets read off a screenshot out of it too.
            let desc = crate::scrub::scrub(&desc).0;
            let _ = state.put_image_desc(id, &fnv1a_hex(&bytes), &desc, now_ms);
            Some(desc)
        }
        Err(e) => {
            eprintln!("[jira-image] describe {filename} failed: {e}");
            None
        }
    }
}

/// `filename: description` lines for an issue's image attachments (cache-first), capped at
/// `max_images`. Empty when there are no images or nothing could be described.
fn image_notes(issue: &JiraIssue, client: &Jira, opts: &ImageOpts<'_>, state: &State, now_ms: i64) -> String {
    let mut lines = Vec::new();
    for a in issue.attachments.iter().filter(|a| a.is_image()).take(opts.max_images) {
        let fetch = || client.download(&a.content_url, opts.max_bytes).map_err(|e| e.to_string());
        if let Some(desc) = resolve_description(state, opts.describer, &a.id, &a.filename, &a.mime_type, now_ms, fetch) {
            lines.push(format!("{}: {}", a.filename, desc));
        }
    }
    lines.join("\n")
}

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

/// All chunks of one issue, redacted. `image_notes` (vision descriptions of image attachments,
/// may be empty) is appended to the issue body and redacted with everything else. Returns
/// `(chunks, redactions)`.
pub fn jira_chunks(issue: &JiraIssue, counter: &dyn TokenCounter, cfg: &IndexConfig, image_notes: &str) -> (Vec<Chunk>, usize) {
    let redactor = &cfg.chunk.redactor;
    let mut redactions = 0;
    let mut scrub = |s: &str| {
        let (clean, n) = redactor.scrub(s);
        redactions += n;
        clean
    };
    let summary = scrub(&issue.summary);
    let description = scrub(&issue.description);
    let notes = scrub(image_notes);
    let header = format!("[{} · {} · {}] {}\n", issue.key, issue.issue_type, issue.status, summary);
    let budget = cfg.chunk.max_tokens.saturating_sub(counter.count(&header)).max(16);

    let mut chunks = Vec::new();
    // Issue chunk(s): header + description + image descriptions.
    let mut body = if description.trim().is_empty() { summary.clone() } else { description };
    if !notes.trim().is_empty() {
        if !body.trim().is_empty() {
            body.push_str("\n\n");
        }
        body.push_str("Obrazy:\n");
        body.push_str(notes.trim());
    }
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
    images: Option<&ImageOpts<'_>>,
    now_ms: i64,
) -> Result<Report, IndexError> {
    if keys.is_empty() {
        return Ok(Report::default());
    }
    let since = state.get_meta(WATERMARK).ok().flatten().and_then(|s| s.parse::<i64>().ok());
    let mut report = Report::default();
    let mut max_updated = since.unwrap_or(0);
    // Pages restart from the newest `updated` seen rather than continuing by offset: an issue
    // edited during the run moves to the end of the `updated` order, and an offset would then
    // step over the issue that slid into its place, for good once the watermark passed it.
    let mut cursor = since;
    let mut start = 0u32;
    // Key → `updated` already handled this run. Restarted pages overlap the previous one, and a
    // repeat must be neither counted again nor retried; an issue edited mid-run comes back with
    // a new `updated` and is handled again.
    let mut seen: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    loop {
        let jql = sync_jql(keys, cursor);
        let page = client.search_full(&jql, start, PAGE).map_err(|e| IndexError::Source(e.to_string()))?;
        if page.issues.is_empty() {
            break;
        }
        let fetched = page.issues.len() as u32;
        let page_max = page.issues.iter().filter_map(|i| updated_ms(&i.updated)).max();
        for issue in &page.issues {
            max_updated = max_updated.max(updated_ms(&issue.updated).unwrap_or(0));
            match seen.insert(issue.key.clone(), issue.updated.clone()) {
                Some(prev) if prev == issue.updated => continue,
                Some(_) => {}
                None => report.scanned += 1,
            }
            let fingerprint = issue.updated.clone();
            if state.get(SOURCE, &issue.key)?.is_some_and(|i| i.fingerprint == fingerprint) {
                report.unchanged += 1;
                continue;
            }
            let notes = images.map(|o| image_notes(issue, client, o, state, now_ms)).unwrap_or_default();
            let (chunks, redactions) = jira_chunks(issue, counter, config, &notes);
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
        if start + fetched >= page.total {
            break;
        }
        match page_max {
            // The re-query overlaps the page by the JQL margin; those issues are unchanged.
            Some(m) if cursor.is_none_or(|c| m > c) => {
                cursor = Some(m);
                start = 0;
            }
            // A whole page within one cursor window: only an offset can move past it.
            _ => start += fetched,
        }
    }
    // Persist the watermark only after a clean pass, so a mid-run failure re-scans the tail.
    if report.failed.is_empty() && max_updated > 0 {
        let _ = state.set_meta(WATERMARK, &max_updated.to_string());
    }
    let last = state.get_meta(RECONCILED).ok().flatten().and_then(|s| s.parse::<i64>().ok());
    if last.is_none_or(|l| now_ms - l >= RECONCILE_EVERY_MS) {
        report.removed = reconcile(store, state, client, keys).await?;
        state.set_meta(RECONCILED, &now_ms.to_string())?;
    }
    Ok(report)
}

/// Removes indexed issues of the mapped projects that Jira no longer has under that key: deleted,
/// or moved to another key or project. The delta sync only ever sees issues that still match,
/// so without this they would stay searchable forever. A key missing from the listing is removed
/// only once Jira confirms it, so a listing that shifted while it was paged deletes nothing.
/// Returns how many issues were removed.
async fn reconcile(store: &Store, state: &State, client: &Jira, keys: &[String]) -> Result<usize, IndexError> {
    let listed: std::collections::HashSet<String> = client
        .all_keys(&format!("project in ({}) ORDER BY key ASC", keys.join(", ")))
        .map_err(|e| IndexError::Source(e.to_string()))?
        .into_iter()
        .collect();
    let indexed = SearchFilter::default().eq("source", SOURCE).eq("kind", "issue").any_of("project", keys.to_vec());
    let mut removed = 0;
    for key in store.distinct_payload(&indexed, "issue_key", 10_000_000).await? {
        if listed.contains(&key) {
            continue;
        }
        let gone = match client.issue(&key) {
            Err(sources::jira::JiraError::NotFound(_)) => true,
            Ok(issue) => issue.key != key,
            Err(e) => {
                eprintln!("[jira] could not check {key}, kept: {e}");
                false
            }
        };
        if gone {
            store.delete_matching(&SearchFilter::default().eq("source", SOURCE).eq("issue_key", key.clone())).await?;
            state.forget(SOURCE, &key)?;
            removed += 1;
        }
    }
    Ok(removed)
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
            attachments: Vec::new(),
        }
    }

    #[test]
    fn one_issue_chunk_and_one_per_comment_with_context() {
        let (chunks, _) = jira_chunks(&issue("HD-1", &[("Piotr", "restarted the service"), ("Anna", "ok, works")]), &WordCounter, &IndexConfig::default(), "");
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
        let (chunks, n) = jira_chunks(&i, &WordCounter, &cfg, "");
        assert!(n >= 2);
        assert!(chunks.iter().all(|c| !c.text.contains("Tajne123")));
    }

    #[test]
    fn ids_are_stable_and_distinct() {
        let (a, _) = jira_chunks(&issue("HD-3", &[("A", "x")]), &WordCounter, &IndexConfig::default(), "");
        let (b, _) = jira_chunks(&issue("HD-3", &[("A", "x")]), &WordCounter, &IndexConfig::default(), "");
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

    struct FakeDescriber {
        calls: std::sync::atomic::AtomicUsize,
        out: Result<String, String>,
    }
    impl ImageDescriber for FakeDescriber {
        fn describe(&self, _bytes: &[u8], _filename: &str, _mime: &str) -> Result<String, String> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.out.clone()
        }
    }

    #[test]
    fn image_notes_land_in_the_issue_chunk_and_are_redacted() {
        let mut cfg = IndexConfig::default();
        cfg.chunk.redactor = crate::Redactor::new(["Tajne123".to_string()]);
        let (chunks, n) = jira_chunks(&issue("HD-7", &[]), &WordCounter, &cfg, "screen.png: error 500 token Tajne123 visible");
        assert!(n >= 1);
        let issue_chunk = chunks.iter().find(|c| c.payload["kind"] == "issue").unwrap();
        assert!(issue_chunk.text.contains("Obrazy:") && issue_chunk.text.contains("screen.png: error 500"), "{}", issue_chunk.text);
        assert!(!issue_chunk.text.contains("Tajne123"), "the image description is redacted too");
    }

    #[test]
    fn description_is_cached_after_the_first_describe() {
        use std::sync::atomic::Ordering;
        let state = State::in_memory().unwrap();
        let d = FakeDescriber { calls: std::sync::atomic::AtomicUsize::new(0), out: Ok("a login error screen".into()) };
        let fetched = std::cell::Cell::new(0);
        let r = resolve_description(&state, &d, "att-1", "s.png", "image/png", 0, || {
            fetched.set(fetched.get() + 1);
            Ok(vec![1u8, 2, 3])
        });
        assert_eq!(r.as_deref(), Some("a login error screen"));
        // Second time: cache hit — neither fetch nor describe runs again.
        let r2 = resolve_description(&state, &d, "att-1", "s.png", "image/png", 0, || {
            fetched.set(fetched.get() + 1);
            Ok(vec![1u8, 2, 3])
        });
        assert_eq!(r2.as_deref(), Some("a login error screen"));
        assert_eq!(d.calls.load(Ordering::SeqCst), 1, "described exactly once");
        assert_eq!(fetched.get(), 1, "fetched exactly once");
    }

    #[test]
    fn a_failed_fetch_or_describe_yields_no_note_and_caches_nothing() {
        let state = State::in_memory().unwrap();
        let boom = FakeDescriber { calls: std::sync::atomic::AtomicUsize::new(0), out: Err("model down".into()) };
        assert!(resolve_description(&state, &boom, "att-2", "s.png", "image/png", 0, || Ok(vec![9])).is_none());
        assert!(state.get_image_desc("att-2").unwrap().is_none(), "a failed describe is not cached");
        let ok = FakeDescriber { calls: std::sync::atomic::AtomicUsize::new(0), out: Ok("x".into()) };
        assert!(resolve_description(&state, &ok, "att-3", "s.png", "image/png", 0, || Err("404".into())).is_none());
        assert_eq!(ok.calls.load(std::sync::atomic::Ordering::SeqCst), 0, "no describe when the fetch fails");
    }

    #[test]
    fn a_secret_read_off_a_screenshot_never_reaches_the_cache() {
        let state = State::in_memory().unwrap();
        let d = FakeDescriber {
            calls: std::sync::atomic::AtomicUsize::new(0),
            out: Ok("  terminal shows ghp_abcdefghijklmnopqrstuvwxyz0123456789AB and a 401  ".into()),
        };
        let want = "terminal shows [REDACTED:github-token] and a 401";
        let r = resolve_description(&state, &d, "att-9", "s.png", "image/png", 0, || Ok(vec![4u8, 5]));
        assert_eq!(r.as_deref(), Some(want));
        assert_eq!(state.get_image_desc("att-9").unwrap().as_deref(), Some(want));
    }
}
