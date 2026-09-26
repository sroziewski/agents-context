//! Turns sessions into embeddable chunks: one chunk per turn when it fits the model window,
//! otherwise paragraph-aligned pieces with a small overlap so a statement cut at a boundary is
//! still whole in one of the two neighbours.

use serde_json::{Map, Value, json};
use sources::{Session, Turn};
use uuid::Uuid;

use crate::scrub::Redactor;
use crate::tokens::TokenCounter;

/// Bump whenever chunk text or payload layout changes: the indexer re-embeds everything on a
/// version change, because old and new chunks would otherwise coexist under different ids.
pub const CHUNKER_VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Hard cap per chunk including the header; must be <= the embedder's max length.
    pub max_tokens: usize,
    /// A trailing segment up to this size is repeated at the start of the next chunk.
    pub overlap_tokens: usize,
    /// Tool-call lines kept per turn; long agentic turns can have hundreds.
    pub max_tools: usize,
    pub redactor: Redactor,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        // 512 is bge-m3's configured window; the margin absorbs token merges at segment joins.
        Self { max_tokens: 480, overlap_tokens: 64, max_tools: 40, redactor: Redactor::default() }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    pub id: Uuid,
    pub text: String,
    pub tokens: usize,
    pub payload: Map<String, Value>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ChunkStats {
    pub redactions: usize,
}

pub fn chunk_session(s: &Session, counter: &dyn TokenCounter, cfg: &ChunkConfig) -> (Vec<Chunk>, ChunkStats) {
    let mut chunks = Vec::new();
    let mut stats = ChunkStats::default();
    // The title is user text too (OpenCode titles are often the first prompt) and goes into
    // every chunk header and the payload, so it is redacted like the turns.
    let title = s.title.as_deref().map(|t| cfg.redactor.scrub(t).0);
    for turn in &s.turns {
        let (doc, redactions) = cfg.redactor.scrub(&turn_document(turn, cfg.max_tools));
        stats.redactions += redactions;
        let header = header(title.as_deref(), turn);
        let budget = cfg.max_tokens.saturating_sub(counter.count(&header)).max(16);
        let pieces = pack(split(&doc, budget, counter), budget, cfg.overlap_tokens, counter);
        let total = pieces.len();
        for (i, piece) in pieces.into_iter().enumerate() {
            let text = format!("{header}{piece}");
            let turn_key = if turn.id.is_empty() { turn.index.to_string() } else { turn.id.clone() };
            chunks.push(Chunk {
                id: store::point_id(s.source.as_str(), &format!("{}/{turn_key}/{i}", s.id)),
                tokens: counter.count(&text),
                payload: payload(s, title.as_deref(), turn, i, total, &text),
                text,
            });
        }
    }
    (chunks, stats)
}

fn turn_document(t: &Turn, max_tools: usize) -> String {
    let mut doc = format!("User: {}", t.user.trim());
    if !t.assistant.trim().is_empty() {
        doc.push_str("\n\nAssistant: ");
        doc.push_str(t.assistant.trim());
    }
    if !t.tools.is_empty() {
        let shown = &t.tools[..t.tools.len().min(max_tools)];
        doc.push_str("\n\nTools: ");
        doc.push_str(&shown.join("; "));
        if t.tools.len() > shown.len() {
            doc.push_str(&format!("; … {} more", t.tools.len() - shown.len()));
        }
    }
    if !t.files.is_empty() {
        doc.push_str("\n\nFiles: ");
        doc.push_str(&t.files.join(", "));
    }
    doc
}

/// Session title and date on every chunk: they carry meaning ("Login bug fix") that a middle
/// chunk of a long turn would otherwise lack.
fn header(title: Option<&str>, t: &Turn) -> String {
    let date = jiff::Timestamp::from_millisecond(t.ts)
        .map(|ts| ts.strftime("%Y-%m-%d").to_string())
        .unwrap_or_default();
    match title.map(str::trim).filter(|t| !t.is_empty()) {
        Some(title) => format!("[{title} · {date}]\n"),
        None => format!("[{date}]\n"),
    }
}

fn payload(s: &Session, title: Option<&str>, t: &Turn, index: usize, total: usize, text: &str) -> Map<String, Value> {
    let v = json!({
        "kind": "turn",
        "source": s.source.as_str(),
        "project": s.project,
        "cwd": s.cwd,
        "session_id": s.id,
        "parent_id": s.parent_id,
        "subagent": s.is_subagent(),
        "title": title,
        "branch": s.branch,
        "turn_id": t.id,
        "turn_index": t.index,
        "chunk_index": index,
        "chunk_count": total,
        "ts": t.ts,
        "files": t.files,
        "text": text,
    });
    let Value::Object(mut m) = v else { unreachable!("json! object") };
    m.retain(|_, v| !v.is_null());
    m
}

/// Breaks text into segments that each fit `budget`: paragraphs, then lines, then hard cuts.
fn split(text: &str, budget: usize, counter: &dyn TokenCounter) -> Vec<String> {
    let mut out = Vec::new();
    for para in text.split("\n\n").filter(|p| !p.trim().is_empty()) {
        if counter.count(para) <= budget {
            out.push(para.to_owned());
            continue;
        }
        for line in para.lines().filter(|l| !l.trim().is_empty()) {
            if counter.count(line) <= budget {
                out.push(line.to_owned());
            } else {
                out.extend(hard_split(line, budget, counter));
            }
        }
    }
    out
}

/// Cuts an over-long line (minified JSON, a pasted log) by characters, shrinking each piece
/// until it fits.
fn hard_split(line: &str, budget: usize, counter: &dyn TokenCounter) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest: Vec<char> = line.chars().collect();
    while !rest.is_empty() {
        let tokens = counter.count(&rest.iter().collect::<String>()).max(1);
        let mut n = if tokens <= budget { rest.len() } else { (rest.len() * budget / tokens).max(1) };
        loop {
            let piece: String = rest[..n].iter().collect();
            if n == 1 || counter.count(&piece) <= budget {
                out.push(piece);
                break;
            }
            n = (n * 4 / 5).max(1);
        }
        rest.drain(..n);
    }
    out
}

/// Greedily joins segments into chunks under `budget`, re-counting the joined text because
/// tokens can merge across a join.
fn pack(segments: Vec<String>, budget: usize, overlap: usize, counter: &dyn TokenCounter) -> Vec<String> {
    let mut chunks: Vec<String> = Vec::new();
    let mut current: Vec<String> = Vec::new();
    for seg in segments {
        current.push(seg);
        if current.len() > 1 && counter.count(&current.join("\n\n")) > budget {
            let seg = current.pop().expect("just pushed");
            let tail = current.last().cloned();
            chunks.push(current.join("\n\n"));
            current.clear();
            if let Some(tail) = tail.filter(|t| counter.count(t) <= overlap) {
                current.push(tail);
            }
            current.push(seg);
            // The overlap tail plus the new segment may itself be too big; drop the overlap.
            if current.len() > 1 && counter.count(&current.join("\n\n")) > budget {
                current.remove(0);
            }
        }
    }
    if !current.is_empty() {
        chunks.push(current.join("\n\n"));
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokens::WordCounter;
    use sources::Source;

    fn session(turns: Vec<Turn>) -> Session {
        Session {
            source: Source::Claude,
            id: "sid".into(),
            parent_id: None,
            project: "d:/dev/proj".into(),
            cwd: "D:\\dev\\proj".into(),
            title: Some("Login bug fix".into()),
            branch: Some("main".into()),
            started_at: 0,
            updated_at: 0,
            turns,
        }
    }

    fn turn(id: &str, user: &str, assistant: &str) -> Turn {
        Turn { id: id.into(), ts: 1_790_330_400_000, user: user.into(), assistant: assistant.into(), ..Default::default() }
    }

    fn cfg(max: usize, overlap: usize) -> ChunkConfig {
        ChunkConfig { max_tokens: max, overlap_tokens: overlap, max_tools: 3, ..Default::default() }
    }

    #[test]
    fn short_turn_is_one_chunk_with_header_and_labelled_parts() {
        let mut t = turn("u1", "fix login", "Done, see auth.rs");
        t.tools = vec!["Edit auth.rs".into()];
        t.files = vec!["auth.rs".into()];
        let (chunks, _) = chunk_session(&session(vec![t]), &WordCounter, &ChunkConfig::default());
        assert_eq!(chunks.len(), 1);
        assert_eq!(
            chunks[0].text,
            "[Login bug fix · 2026-09-25]\nUser: fix login\n\nAssistant: Done, see auth.rs\n\nTools: Edit auth.rs\n\nFiles: auth.rs"
        );
    }

    #[test]
    fn payload_carries_project_and_session_identity() {
        let (chunks, _) = chunk_session(&session(vec![turn("u1", "q", "a")]), &WordCounter, &ChunkConfig::default());
        let p = &chunks[0].payload;
        assert_eq!(p["project"], "d:/dev/proj");
        assert_eq!(p["source"], "claude");
        assert_eq!(p["session_id"], "sid");
        assert_eq!(p["turn_id"], "u1");
        assert_eq!(p["kind"], "turn");
        assert_eq!(p["subagent"], false);
        assert_eq!(p["ts"], 1_790_330_400_000_i64);
        assert_eq!(p["text"], Value::String(chunks[0].text.clone()));
        assert!(!p.contains_key("parent_id"), "nulls are dropped");
    }

    #[test]
    fn ids_are_stable_and_unique_per_chunk() {
        let s = session(vec![turn("u1", "q", "a"), turn("u2", "q2", "a2")]);
        let (a, _) = chunk_session(&s, &WordCounter, &ChunkConfig::default());
        let (b, _) = chunk_session(&s, &WordCounter, &ChunkConfig::default());
        assert_eq!(a.iter().map(|c| c.id).collect::<Vec<_>>(), b.iter().map(|c| c.id).collect::<Vec<_>>());
        assert_ne!(a[0].id, a[1].id);
    }

    #[test]
    fn long_turn_splits_on_paragraphs_within_budget() {
        let para = |w: &str| vec![w; 30].join(" ");
        let assistant = [para("alpha"), para("beta"), para("gamma"), para("delta")].join("\n\n");
        let (chunks, _) = chunk_session(&session(vec![turn("u1", "q", &assistant)]), &WordCounter, &cfg(70, 0));
        assert!(chunks.len() >= 2);
        for c in &chunks {
            assert!(c.tokens <= 70, "{} tokens: {}", c.tokens, c.text);
            assert!(c.text.starts_with("[Login bug fix"), "every chunk carries the header");
        }
        let all: String = chunks.iter().map(|c| c.text.as_str()).collect();
        for w in ["alpha", "beta", "gamma", "delta"] {
            assert!(all.contains(w), "{w} lost");
        }
        assert_eq!(chunks[0].payload["chunk_count"], chunks.len());
        assert_eq!(chunks.last().unwrap().payload["chunk_index"], chunks.len() - 1);
    }

    #[test]
    fn small_trailing_segment_is_repeated_as_overlap() {
        let segs = vec!["a a a a".to_owned(), "b b".to_owned(), "c c c c c".to_owned()];
        let chunks = pack(segs, 7, 3, &WordCounter);
        assert_eq!(chunks, ["a a a a\n\nb b", "b b\n\nc c c c c"]);
    }

    #[test]
    fn overlap_is_dropped_when_it_would_overflow() {
        let segs = vec!["a a".to_owned(), "b b b b b b".to_owned()];
        assert_eq!(pack(segs, 6, 3, &WordCounter), ["a a", "b b b b b b"]);
    }

    #[test]
    fn oversized_line_is_hard_split_without_losing_characters() {
        let line = "x".repeat(10) + " " + &"word ".repeat(50);
        let pieces = hard_split(&line, 8, &WordCounter);
        assert!(pieces.iter().all(|p| WordCounter.count(p) <= 8));
        assert_eq!(pieces.concat(), line);
    }

    #[test]
    fn hard_split_handles_multibyte_characters() {
        let line = "żółć ".repeat(40);
        let pieces = hard_split(&line, 5, &WordCounter);
        assert_eq!(pieces.concat(), line);
    }

    #[test]
    fn secrets_are_redacted_before_chunking_and_counted() {
        let t = turn("u1", "use DB_PASSWORD=hunter2hunter please", "ok");
        let (chunks, stats) = chunk_session(&session(vec![t]), &WordCounter, &ChunkConfig::default());
        assert!(!chunks[0].text.contains("hunter2"));
        assert!(!chunks[0].payload["text"].as_str().unwrap().contains("hunter2"));
        assert_eq!(stats.redactions, 1);
    }

    #[test]
    fn listed_literals_are_redacted_from_chunks() {
        let t = turn("u1", "the SOCN password is now plainsecret1", "ok");
        let cfg = ChunkConfig { redactor: Redactor::new(["plainsecret1".to_owned()]), ..Default::default() };
        let (chunks, stats) = chunk_session(&session(vec![t]), &WordCounter, &cfg);
        assert!(!chunks[0].text.contains("plainsecret1"), "{}", chunks[0].text);
        assert!(stats.redactions >= 1);
    }

    #[test]
    fn secret_in_the_session_title_is_redacted_in_header_and_payload() {
        let mut s = session(vec![turn("u1", "q", "a")]);
        s.title = Some("rotate plainsecret1 today".into());
        let cfg = ChunkConfig { redactor: Redactor::new(["plainsecret1".to_owned()]), ..Default::default() };
        let (chunks, _) = chunk_session(&s, &WordCounter, &cfg);
        assert!(!chunks[0].text.contains("plainsecret1"), "{}", chunks[0].text);
        assert_eq!(chunks[0].payload["title"], "rotate [REDACTED:listed] today");
    }

    #[test]
    fn tool_list_is_capped_with_remainder_count() {
        let mut t = turn("u1", "q", "");
        t.tools = (0..5).map(|i| format!("Read f{i}")).collect();
        let (chunks, _) = chunk_session(&session(vec![t]), &WordCounter, &cfg(480, 64));
        assert!(chunks[0].text.contains("Tools: Read f0; Read f1; Read f2; … 2 more"));
        assert!(!chunks[0].text.contains("Assistant:"), "empty assistant text is omitted");
    }

    #[test]
    fn untitled_session_header_has_date_only() {
        let mut s = session(vec![turn("u1", "q", "a")]);
        s.title = None;
        let (chunks, _) = chunk_session(&s, &WordCounter, &ChunkConfig::default());
        assert!(chunks[0].text.starts_with("[2026-09-25]\nUser: q"));
    }

    #[test]
    fn turn_without_id_falls_back_to_index_for_its_point_id() {
        let mut s = session(vec![turn("", "q", "a"), turn("", "q", "a")]);
        s.turns[1].index = 1;
        let (chunks, _) = chunk_session(&s, &WordCounter, &ChunkConfig::default());
        assert_ne!(chunks[0].id, chunks[1].id);
    }
}
