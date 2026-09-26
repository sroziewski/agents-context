//! Teams threads → chunks for the shared `teams` collection.
//!
//! Two kinds per thread. `thread` chunks hold the conversation (root and replies) and answer
//! "where was X discussed". `message` chunks hold one message with a context prefix (the
//! thread's opening line and the previous reply) and answer "who said X": a reply such as
//! "ok, zrobione" means nothing alone.

use embed::Embedder;
use serde_json::{Map, Value, json};
use sources::teams::{TeamsDb, Thread};
use store::{Field, SearchFilter, Store};

use crate::chunk::{Chunk, pack, split};
use crate::indexer::{IndexConfig, IndexError, Report, write_chunks};
use crate::state::{Record, State};
use crate::tokens::TokenCounter;

pub const SOURCE: &str = "teams";

/// Payload fields the teams collection indexes. No project tenant: Teams is shared.
pub const TEAMS_FIELDS: &[Field] = &[
    Field::Keyword("kind"),
    Field::Keyword("thread_id"),
    Field::Keyword("channel_id"),
    Field::Keyword("author"),
];

/// Context lines in a message chunk are cut to this many characters.
const CONTEXT_CHARS: usize = 150;

fn date(ts: i64, with_time: bool) -> String {
    let fmt = if with_time { "%Y-%m-%d %H:%M" } else { "%Y-%m-%d" };
    jiff::Timestamp::from_millisecond(ts)
        .map(|t| t.to_zoned(jiff::tz::TimeZone::system()).strftime(fmt).to_string())
        .unwrap_or_default()
}

fn first_line(text: &str, max: usize) -> String {
    let line = text.lines().find(|l| !l.trim().is_empty()).unwrap_or("").trim();
    if line.chars().count() > max { format!("{}…", line.chars().take(max).collect::<String>()) } else { line.to_owned() }
}

/// All chunks of one thread, with message texts redacted. Returns `(chunks, redactions)`.
pub fn thread_chunks(t: &Thread, counter: &dyn TokenCounter, cfg: &IndexConfig) -> (Vec<Chunk>, usize) {
    let redactor = &cfg.chunk.redactor;
    let mut redactions = 0;
    let texts: Vec<String> = t
        .messages
        .iter()
        .map(|m| {
            let (clean, n) = redactor.scrub(&m.text);
            redactions += n;
            clean
        })
        .collect();
    let header = format!("[{} / {} · {}]\n", t.channel.team, t.channel.name, date(t.messages.first().map_or(0, |m| m.ts), false));
    let budget = cfg.chunk.max_tokens.saturating_sub(counter.count(&header)).max(16);
    let mut chunks = Vec::new();

    // Thread chunks: every message as a line, packed into window-sized pieces.
    let lines: Vec<String> =
        t.messages.iter().zip(&texts).map(|(m, text)| format!("{} ({}): {}", m.author, date(m.ts, true), text.trim())).collect();
    let pieces = pack(split(&lines.join("\n\n"), budget, counter), budget, cfg.chunk.overlap_tokens, counter);
    let mut authors: Vec<&str> = t.messages.iter().map(|m| m.author.as_str()).collect();
    authors.sort_unstable();
    authors.dedup();
    let total = pieces.len();
    for (i, piece) in pieces.into_iter().enumerate() {
        let text = format!("{header}{piece}");
        let mut p = base_payload(t, "thread", &text);
        p.insert("chunk_index".into(), json!(i));
        p.insert("chunk_count".into(), json!(total));
        p.insert("first_ts".into(), json!(t.messages.first().map_or(0, |m| m.ts)));
        p.insert("ts".into(), json!(t.last_ts()));
        p.insert("authors".into(), json!(authors));
        p.insert("message_count".into(), json!(t.messages.len()));
        chunks.push(Chunk { id: store::point_id(SOURCE, &format!("{}/t{i}", t.id)), tokens: counter.count(&text), payload: p, text });
    }

    // Message chunks, each with the thread's opening line and the previous message as context.
    let opening = texts.first().map(|t| first_line(t, CONTEXT_CHARS)).unwrap_or_default();
    for (k, (m, text)) in t.messages.iter().zip(&texts).enumerate() {
        let mut context = String::new();
        if k > 0 {
            context.push_str(&format!("Thread: {opening}\n"));
        }
        if k > 1 {
            let prev = &t.messages[k - 1];
            context.push_str(&format!("Previous: {}: {}\n", prev.author, first_line(&texts[k - 1], CONTEXT_CHARS)));
        }
        let lead = format!("{header}{context}{} ({}): ", m.author, date(m.ts, true));
        let room = cfg.chunk.max_tokens.saturating_sub(counter.count(&lead)).max(16);
        let pieces = pack(split(text.trim(), room, counter), room, 0, counter);
        for (i, piece) in pieces.into_iter().enumerate() {
            let text = format!("{lead}{piece}");
            let mut p = base_payload(t, "message", &text);
            p.insert("message_id".into(), json!(m.id));
            p.insert("author".into(), json!(m.author));
            p.insert("ts".into(), json!(m.ts));
            p.insert("chunk_index".into(), json!(i));
            chunks.push(Chunk {
                id: store::point_id(SOURCE, &format!("{}/m{}/{i}", t.id, m.id)),
                tokens: counter.count(&text),
                payload: p,
                text,
            });
        }
    }
    (chunks, redactions)
}

fn base_payload(t: &Thread, kind: &str, text: &str) -> Map<String, Value> {
    let v = json!({
        "kind": kind,
        "source": SOURCE,
        "thread_id": t.id,
        "channel_id": t.channel.id,
        "team": t.channel.team,
        "channel": t.channel.name,
        "text": text,
    });
    let Value::Object(m) = v else { unreachable!("json! object") };
    m
}

/// Indexes every thread whose fingerprint changed. Threads that disappear from `teams.db` keep
/// their points, like sessions: the index outlives the app's lookback window.
pub async fn sync_teams(
    store: &Store,
    embedder: &mut dyn Embedder,
    counter: &dyn TokenCounter,
    state: &State,
    db: &TeamsDb,
    config: &IndexConfig,
    now_ms: i64,
) -> Result<Report, IndexError> {
    let threads = db.threads().map_err(|e| IndexError::Source(e.to_string()))?;
    let mut report = Report { scanned: threads.len(), ..Default::default() };
    for t in &threads {
        let fingerprint = t.fingerprint();
        if state.get(SOURCE, &t.id)?.is_some_and(|i| i.fingerprint == fingerprint) {
            report.unchanged += 1;
            continue;
        }
        let (chunks, redactions) = thread_chunks(t, counter, config);
        let scope = SearchFilter::default().eq("source", SOURCE).eq("thread_id", t.id.clone());
        // One failing thread must not keep every thread after it from being indexed; it is
        // reported and retried on the next run (its fingerprint is not recorded).
        let n = match write_chunks(store, embedder, config, &scope, &chunks).await {
            Ok(n) => n,
            Err(e) => {
                report.failed.push((t.id.clone(), e.to_string()));
                continue;
            }
        };
        state.record(
            &Record {
                source: SOURCE,
                session_id: &t.id,
                fingerprint: &fingerprint,
                project: "",
                chunks: n,
                redactions,
                turns: t.messages.len(),
                subagent: false,
            },
            now_ms,
        )?;
        report.indexed += 1;
        report.chunks += n;
        report.redactions += redactions;
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokens::WordCounter;
    use sources::teams::{Channel, Message};

    fn thread(texts: &[(&str, &str)]) -> Thread {
        Thread {
            id: "100".into(),
            channel: Channel { id: "c1".into(), team: "ZSUN-Helpdesk".into(), name: "Analiza_HD".into() },
            messages: texts
                .iter()
                .enumerate()
                .map(|(i, (author, text))| Message {
                    id: format!("{}", 100 + i),
                    thread_id: "100".into(),
                    author: (*author).into(),
                    ts: 1_790_330_400_000 + i as i64 * 60_000,
                    text: (*text).into(),
                })
                .collect(),
        }
    }

    #[test]
    fn short_thread_gives_one_thread_chunk_and_one_chunk_per_message() {
        let t = thread(&[("Anna", "Logowanie przez SSO nie działa"), ("Piotr", "sprawdzam"), ("Anna", "ok, zrobione")]);
        let (chunks, _) = thread_chunks(&t, &WordCounter, &IndexConfig::default());
        let kinds: Vec<&str> = chunks.iter().map(|c| c.payload["kind"].as_str().unwrap()).collect();
        assert_eq!(kinds, ["thread", "message", "message", "message"]);
        let th = &chunks[0];
        assert!(th.text.starts_with("[ZSUN-Helpdesk / Analiza_HD · "));
        assert!(th.text.contains("Anna (") && th.text.contains("ok, zrobione"));
        assert_eq!(th.payload["authors"], json!(["Anna", "Piotr"]));
        assert_eq!(th.payload["message_count"], 3);
    }

    #[test]
    fn replies_carry_the_threads_opening_line_and_the_previous_message() {
        let t = thread(&[("Anna", "Logowanie przez SSO nie działa\nszczegóły niżej"), ("Piotr", "sprawdzam"), ("Anna", "ok, zrobione")]);
        let (chunks, _) = thread_chunks(&t, &WordCounter, &IndexConfig::default());
        let last = chunks.iter().rfind(|c| c.payload["kind"] == "message").unwrap();
        assert!(last.text.contains("Thread: Logowanie przez SSO nie działa\n"), "{}", last.text);
        assert!(last.text.contains("Previous: Piotr: sprawdzam\n"), "{}", last.text);
        assert!(last.text.ends_with("ok, zrobione"));
        assert_eq!(last.payload["author"], "Anna");
        assert_eq!(last.payload["message_id"], "102");
        let root = chunks.iter().find(|c| c.payload.get("message_id") == Some(&json!("100"))).unwrap();
        assert!(!root.text.contains("Thread:"), "the root needs no context line");
    }

    #[test]
    fn long_threads_and_long_messages_are_split_within_the_budget() {
        let long = vec!["słowo"; 900].join(" ");
        let t = thread(&[("Anna", &long), ("Piotr", "krótko")]);
        let cfg = IndexConfig::default();
        let (chunks, _) = thread_chunks(&t, &WordCounter, &cfg);
        assert!(chunks.iter().all(|c| c.tokens <= cfg.chunk.max_tokens), "{:?}", chunks.iter().map(|c| c.tokens).collect::<Vec<_>>());
        assert!(chunks.iter().filter(|c| c.payload["kind"] == "thread").count() >= 2);
        assert!(chunks.iter().filter(|c| c.payload.get("message_id") == Some(&json!("100"))).count() >= 2, "a long message spans several chunks");
    }

    #[test]
    fn ids_are_stable_and_distinct() {
        let t = thread(&[("Anna", "a"), ("Piotr", "b")]);
        let (a, _) = thread_chunks(&t, &WordCounter, &IndexConfig::default());
        let (b, _) = thread_chunks(&t, &WordCounter, &IndexConfig::default());
        assert_eq!(a.iter().map(|c| c.id).collect::<Vec<_>>(), b.iter().map(|c| c.id).collect::<Vec<_>>());
        let mut ids: Vec<_> = a.iter().map(|c| c.id).collect();
        ids.dedup();
        assert_eq!(ids.len(), a.len());
    }

    #[test]
    fn message_text_is_redacted_everywhere_it_appears() {
        let t = thread(&[("Anna", "hasło do testowej bazy to Tajne123!x"), ("Piotr", "dzięki")]);
        let (chunks, n) = thread_chunks(&t, &WordCounter, &IndexConfig::default());
        assert!(n >= 1);
        assert!(chunks.iter().all(|c| !c.text.contains("Tajne123")), "thread, message and context lines are all redacted");
    }
}
