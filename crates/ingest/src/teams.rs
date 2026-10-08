//! Teams threads → chunks for the shared `teams` collection.
//!
//! Two kinds per thread. `thread` chunks hold the conversation (root and replies) and answer
//! "where was X discussed". `message` chunks hold one message with a context prefix (the
//! thread's opening line and the previous reply) and answer "who said X": a reply such as
//! "ok, zrobione" means nothing alone.

use embed::Embedder;
use serde_json::{Map, Value, json};
use sources::teams::{Meeting, TeamsDb, Thread, TranscriptLine};
use store::{Field, SearchFilter, Store};

use crate::chunk::{Chunk, pack, split};
use crate::indexer::{IndexConfig, IndexError, Report, write_chunks};
use crate::state::{Record, State};
use crate::tokens::TokenCounter;

pub const SOURCE: &str = "teams";

/// Payload fields the teams collection indexes. No project tenant: Teams is shared. `meeting_id`
/// scopes meeting/transcript chunks (thread/message chunks simply never set it, and vice versa).
/// `team` lets inject leave private chats (team "Chat") out without opening teams.db; every chunk
/// has always carried it, so adding the index needed no re-index.
pub const TEAMS_FIELDS: &[Field] = &[
    Field::Keyword("kind"),
    Field::Keyword("team"),
    Field::Keyword("thread_id"),
    Field::Keyword("channel_id"),
    Field::Keyword("author"),
    Field::Keyword("meeting_id"),
];

/// Payload fields for the meetings collection (transcript/summary/meeting-card chunks). Separate
/// from `teams` so channel-thread search and meeting search never surface each other's chunks.
pub const MEETING_FIELDS: &[Field] = &[Field::Keyword("kind"), Field::Keyword("meeting_id")];

/// Context lines in a message chunk are cut to this many characters.
const CONTEXT_CHARS: usize = 150;

/// The meeting card shows only a short head of the summary, so the card stays one small chunk;
/// the full summary is in the `summary` chunks.
const CARD_SUMMARY_HEAD: usize = 80;

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

/// Bump when `meeting_chunks`' layout *or target collection* changes, so already-indexed meetings
/// re-chunk/re-write even though their content fingerprint is unchanged (see `sync_meetings`).
/// Narrower than `CHUNKER_VERSION`, which would re-embed every session too. 3: moved meeting
/// chunks to their own `meetings` collection (DEV-51).
pub const MEETING_LAYOUT: u32 = 3;

/// Chunks for one meeting. `transcript` chunks pack the spoken lines into window-sized pieces for
/// topic search ("what was discussed"); `summary` chunks hold the full AI summary, split so all of
/// it is embedded; one compact `meeting` card chunk carries the title, participants and the head of
/// the summary for meeting-level matching. Per-line speaker precision ("who said X") is left to the
/// FTS half of the search, so the multi-speaker vector windows stay simple. Returns
/// `(chunks, redactions)`.
pub fn meeting_chunks(m: &Meeting, lines: &[TranscriptLine], counter: &dyn TokenCounter, cfg: &IndexConfig) -> (Vec<Chunk>, usize) {
    let redactor = &cfg.chunk.redactor;
    let mut redactions = 0;
    let header = format!("[{} · {}]\n", m.title, date(m.started_at, false));
    let budget = cfg.chunk.max_tokens.saturating_sub(counter.count(&header)).max(16);

    let participants = sources::teams::participants(lines);
    let mut rendered: Vec<String> = Vec::new();
    for l in lines {
        let (clean, n) = redactor.scrub(&l.text);
        redactions += n;
        // System lines (e.g. "started transcription") have no speaker; show a neutral marker.
        let who = l.speaker.as_deref().unwrap_or("—");
        rendered.push(format!("{who} ({}): {}", date(l.ts, true), clean.trim()));
    }

    let mut chunks = Vec::new();

    // Transcript window chunks.
    if !rendered.is_empty() {
        let pieces = pack(split(&rendered.join("\n"), budget, counter), budget, cfg.chunk.overlap_tokens, counter);
        let total = pieces.len();
        let last_ts = lines.last().map_or(m.started_at, |l| l.ts);
        for (i, piece) in pieces.into_iter().enumerate() {
            let text = format!("{header}{piece}");
            let mut p = meeting_payload(m, "transcript", &text);
            p.insert("chunk_index".into(), json!(i));
            p.insert("chunk_count".into(), json!(total));
            p.insert("first_ts".into(), json!(m.started_at));
            p.insert("ts".into(), json!(last_ts));
            chunks.push(Chunk { id: store::point_id(SOURCE, &format!("meeting/{}/tr{i}", m.id)), tokens: counter.count(&text), payload: p, text });
        }
    }

    // The AI summary, scrubbed once, split like the transcript so all of it is embedded — a
    // ~2k-token summary in one chunk would otherwise be truncated by the embedder to its head.
    let summary = m.summary.as_deref().filter(|_| m.has_summary()).map(|s| {
        let (clean, n) = redactor.scrub(s);
        redactions += n;
        clean.trim().to_owned()
    });
    if let Some(sum) = summary.as_deref().filter(|s| !s.is_empty()) {
        let sheader = format!("[{} · {}] Podsumowanie\n", m.title, date(m.started_at, false));
        let sbudget = cfg.chunk.max_tokens.saturating_sub(counter.count(&sheader)).max(16);
        let pieces = pack(split(sum, sbudget, counter), sbudget, cfg.chunk.overlap_tokens, counter);
        let total = pieces.len();
        for (i, piece) in pieces.into_iter().enumerate() {
            let text = format!("{sheader}{piece}");
            let mut p = meeting_payload(m, "summary", &text);
            p.insert("chunk_index".into(), json!(i));
            p.insert("chunk_count".into(), json!(total));
            p.insert("ts".into(), json!(m.started_at));
            chunks.push(Chunk { id: store::point_id(SOURCE, &format!("meeting/{}/sum{i}", m.id)), tokens: counter.count(&text), payload: p, text });
        }
    }

    // One compact meeting "card": title, participants, and just the head of the summary — small
    // enough to stay a single chunk, for meeting-level matching. The whole summary is in the
    // `summary` chunks above.
    let mut body = String::new();
    if !participants.is_empty() {
        body.push_str(&format!("\nUczestnicy: {}", participants.join(", ")));
    }
    if let Some(sum) = summary.as_deref().filter(|s| !s.is_empty()) {
        let head = split(sum, CARD_SUMMARY_HEAD, counter).into_iter().next().unwrap_or_default();
        body.push_str(&format!("\nPodsumowanie: {head}"));
    }
    let text = format!("[{} · {}–{}]{body}", m.title, date(m.started_at, true), date(m.ended_at, true));
    let mut p = meeting_payload(m, "meeting", &text);
    p.insert("ts".into(), json!(m.started_at));
    p.insert("first_ts".into(), json!(m.started_at));
    p.insert("participants".into(), json!(participants));
    chunks.push(Chunk { id: store::point_id(SOURCE, &format!("meeting/{}/card", m.id)), tokens: counter.count(&text), payload: p, text });

    (chunks, redactions)
}

fn meeting_payload(m: &Meeting, kind: &str, text: &str) -> Map<String, Value> {
    let v = json!({
        "kind": kind,
        "source": SOURCE,
        "meeting_id": m.id,
        "title": m.title,
        "text": text,
    });
    let Value::Object(map) = v else { unreachable!("json! object") };
    map
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

/// Indexes every meeting whose transcript changed, into the same `teams` collection. Meeting
/// state ids are prefixed `meeting:` and point ids `meeting/`, so they never collide with the
/// thread/message ids that share the collection.
pub async fn sync_meetings(
    store: &Store,
    embedder: &mut dyn Embedder,
    counter: &dyn TokenCounter,
    state: &State,
    db: &TeamsDb,
    config: &IndexConfig,
    now_ms: i64,
) -> Result<Report, IndexError> {
    let meetings = db.meetings().map_err(|e| IndexError::Source(e.to_string()))?;
    let mut report = Report { scanned: meetings.len(), ..Default::default() };
    for m in &meetings {
        let lines = db.transcript(&m.id).map_err(|e| IndexError::Source(e.to_string()))?;
        // The layout tag forces a re-chunk when meeting_chunks changes shape, independent of the
        // meeting's own content fingerprint.
        let fingerprint = format!("{}-l{MEETING_LAYOUT}", m.fingerprint(&lines));
        let key = format!("meeting:{}", m.id);
        if state.get(SOURCE, &key)?.is_some_and(|i| i.fingerprint == fingerprint) {
            report.unchanged += 1;
            continue;
        }
        let (chunks, redactions) = meeting_chunks(m, &lines, counter, config);
        let scope = SearchFilter::default().eq("source", SOURCE).eq("meeting_id", m.id.clone());
        let n = match write_chunks(store, embedder, config, &scope, &chunks).await {
            Ok(n) => n,
            Err(e) => {
                report.failed.push((m.id.clone(), e.to_string()));
                continue;
            }
        };
        state.record(
            &Record { source: SOURCE, session_id: &key, fingerprint: &fingerprint, project: "", chunks: n, redactions, turns: lines.len(), subagent: false },
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
    use sources::teams::{Channel, Message, Meeting, TranscriptLine};

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

    fn meeting_fx(lines: &[(Option<&str>, &str)], status: &str) -> (Meeting, Vec<TranscriptLine>) {
        let m = Meeting {
            id: "m1".into(),
            chat: "c".into(),
            title: "Standup".into(),
            occurrence: "o".into(),
            started_at: 1_790_330_400_000,
            ended_at: 1_790_330_400_000 + 900_000,
            summary: None,
            summary_status: "unavailable".into(),
            transcript_status: status.into(),
        };
        let lines = lines
            .iter()
            .enumerate()
            .map(|(i, (sp, txt))| TranscriptLine {
                meeting_id: "m1".into(),
                idx: i as i64,
                speaker: sp.map(str::to_string),
                offset_s: Some(i as i64 * 5),
                ts: 1_790_330_400_000 + i as i64 * 5000,
                text: (*txt).into(),
            })
            .collect();
        (m, lines)
    }

    #[test]
    fn short_meeting_gives_transcript_and_meeting_chunks() {
        let (m, lines) = meeting_fx(&[(None, "started transcription"), (Some("Tomasz"), "Zrobiłem migrację"), (Some("Anna"), "ok")], "ok");
        let (chunks, _) = meeting_chunks(&m, &lines, &WordCounter, &IndexConfig::default());
        let kinds: Vec<&str> = chunks.iter().map(|c| c.payload["kind"].as_str().unwrap()).collect();
        assert!(kinds.contains(&"transcript") && kinds.contains(&"meeting"), "{kinds:?}");
        let tr = chunks.iter().find(|c| c.payload["kind"] == "transcript").unwrap();
        assert!(tr.text.starts_with("[Standup · "));
        assert!(tr.text.contains("— ("), "null speaker rendered as a dash");
        assert!(tr.text.contains("Tomasz (") && tr.text.contains("Zrobiłem migrację"));
        let mtg = chunks.iter().find(|c| c.payload["kind"] == "meeting").unwrap();
        assert_eq!(mtg.payload["meeting_id"], "m1");
        assert_eq!(mtg.payload["participants"], json!(["Anna", "Tomasz"]), "sorted, system line excluded");
    }

    #[test]
    fn summary_is_indexed_only_when_the_exporter_captured_it() {
        let (mut m, lines) = meeting_fx(&[(Some("A"), "x")], "ok");
        m.summary = Some("Ustalono deploy na piątek".into());
        m.summary_status = "ok".into();
        let (chunks, _) = meeting_chunks(&m, &lines, &WordCounter, &IndexConfig::default());
        assert!(chunks.iter().find(|c| c.payload["kind"] == "meeting").unwrap().text.contains("Podsumowanie: Ustalono deploy"));

        let (mut m2, lines2) = meeting_fx(&[(Some("A"), "x")], "ok");
        m2.summary = Some("secret plan".into());
        m2.summary_status = "unavailable".into();
        let (chunks2, _) = meeting_chunks(&m2, &lines2, &WordCounter, &IndexConfig::default());
        assert!(!chunks2.iter().any(|c| c.text.contains("secret plan")), "summary omitted unless the exporter marked it ok");

        let (mut m3, lines3) = meeting_fx(&[(Some("A"), "x")], "ok");
        m3.summary = Some("Ustalono release".into());
        m3.summary_status = "available".into();
        let (chunks3, _) = meeting_chunks(&m3, &lines3, &WordCounter, &IndexConfig::default());
        assert!(chunks3.iter().any(|c| c.text.contains("Podsumowanie: Ustalono release")), "legacy status still accepted");
    }

    #[test]
    fn long_transcript_splits_within_budget() {
        let long = vec!["słowo"; 1200].join(" ");
        let (m, lines) = meeting_fx(&[(Some("Anna"), long.as_str())], "ok");
        let cfg = IndexConfig::default();
        let (chunks, _) = meeting_chunks(&m, &lines, &WordCounter, &cfg);
        assert!(chunks.iter().all(|c| c.tokens <= cfg.chunk.max_tokens), "{:?}", chunks.iter().map(|c| c.tokens).collect::<Vec<_>>());
        assert!(chunks.iter().filter(|c| c.payload["kind"] == "transcript").count() >= 2, "long transcript spans several windows");
    }

    #[test]
    fn meeting_ids_are_stable_distinct_and_redacted() {
        let (m, lines) = meeting_fx(&[(Some("Anna"), "hasło do bazy to Tajne123!x"), (Some("Piotr"), "dzięki")], "ok");
        let (a, n) = meeting_chunks(&m, &lines, &WordCounter, &IndexConfig::default());
        let (b, _) = meeting_chunks(&m, &lines, &WordCounter, &IndexConfig::default());
        assert_eq!(a.iter().map(|c| c.id).collect::<Vec<_>>(), b.iter().map(|c| c.id).collect::<Vec<_>>(), "stable across runs");
        let mut ids: Vec<_> = a.iter().map(|c| c.id).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), a.len(), "distinct");
        assert!(n >= 1 && a.iter().all(|c| !c.text.contains("Tajne123")), "transcript text redacted everywhere");
    }

    #[test]
    fn fingerprint_changes_when_a_line_is_appended() {
        let (m, mut lines) = meeting_fx(&[(Some("A"), "one")], "ok");
        let before = m.fingerprint(&lines);
        lines.push(TranscriptLine { meeting_id: "m1".into(), idx: 1, speaker: Some("B".into()), offset_s: Some(5), ts: 1_790_330_405_000, text: "two".into() });
        assert_ne!(before, m.fingerprint(&lines));
    }

    #[test]
    fn long_summary_is_split_into_summary_chunks_and_the_card_stays_small() {
        let (mut m, lines) = meeting_fx(&[(Some("Anna"), "x")], "ok");
        m.summary = Some(vec!["ustalenie"; 1500].join(" "));
        m.summary_status = "ok".into();
        let cfg = IndexConfig::default();
        let (chunks, _) = meeting_chunks(&m, &lines, &WordCounter, &cfg);
        let summaries: Vec<_> = chunks.iter().filter(|c| c.payload["kind"] == "summary").collect();
        assert!(summaries.len() >= 2, "a long summary spans several chunks, got {}", summaries.len());
        assert!(summaries.iter().all(|c| c.tokens <= cfg.chunk.max_tokens), "summary chunks within budget");
        assert!(summaries.iter().all(|c| c.text.starts_with("[Standup · ") && c.payload["meeting_id"] == "m1"));
        let card = chunks.iter().find(|c| c.payload["kind"] == "meeting").unwrap();
        assert!(card.tokens <= cfg.chunk.max_tokens, "card stays within budget, {} tokens", card.tokens);
        assert!(card.text.contains("Podsumowanie: ustalenie") && card.text.contains("Uczestnicy: Anna"));
    }

    #[test]
    fn participants_are_cleaned_and_deduped_in_the_card() {
        let (m, lines) = meeting_fx(&[(None, "start"), (Some("Anna Nowak 5 minutes"), "a"), (Some("Anna Nowak"), "b"), (Some("Piotr Kowalski"), "c")], "ok");
        let (chunks, _) = meeting_chunks(&m, &lines, &WordCounter, &IndexConfig::default());
        let card = chunks.iter().find(|c| c.payload["kind"] == "meeting").unwrap();
        assert_eq!(card.payload["participants"], json!(["Anna Nowak", "Piotr Kowalski"]), "duration-variant merged, system line skipped");
    }
}
