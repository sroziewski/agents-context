//! summarize_pending end to end: real transcripts → sync → Qdrant → fake LLM → summary points.

use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Duration;

use embed::HashEmbedder;
use ingest::{IndexConfig, Indexer, SESSION_FIELDS, State, SummaryConfig, Summarizer, WordCounter, summarize_pending};
use serde_json::json;
use sources::claude;
use store::{SearchFilter, Store};

const DIM: usize = 64;
const CWD: &str = r"Z:\work\alpha";

/// Answers with a fixed summary (or error) and records every prompt it was given.
struct Fake {
    reply: Result<String, String>,
    prompts: Mutex<Vec<String>>,
}

impl Fake {
    fn ok(body: &str) -> Self {
        Self { reply: Ok(body.to_owned()), prompts: Mutex::default() }
    }
    fn calls(&self) -> usize {
        self.prompts.lock().unwrap().len()
    }
}

impl Summarizer for Fake {
    fn id(&self) -> &str {
        "fake"
    }
    fn summarize(&self, prompt: &str, timeout: Duration) -> Result<String, String> {
        assert!(timeout > Duration::ZERO && timeout <= Duration::from_secs(60), "{timeout:?}");
        self.prompts.lock().unwrap().push(prompt.to_owned());
        self.reply.clone()
    }
}

struct Env {
    store: Store,
    state: State,
    root: PathBuf,
    _tmp: tempfile::TempDir,
    _qdrant: testkit::Qdrant,
}

fn turns(n: usize, extra: &str) -> Vec<String> {
    (0..n)
        .flat_map(|i| {
            [
                json!({"type": "user", "uuid": format!("u{i}"), "timestamp": "2026-09-26T10:00:00Z", "cwd": CWD,
                       "origin": {"kind": "human"}, "message": {"content": format!("step {i} {extra}")}})
                .to_string(),
                json!({"type": "assistant", "timestamp": "2026-09-26T10:00:05Z",
                       "message": {"content": [{"type": "text", "text": format!("did step {i}")}]}})
                .to_string(),
            ]
        })
        .collect()
}

async fn env() -> Env {
    let qdrant = testkit::qdrant().await;
    let store = Store::connect(&qdrant.url, None, "sessions", DIM).unwrap();
    store.ensure_collection(SESSION_FIELDS).await.unwrap();
    let tmp = tempfile::tempdir().unwrap();
    Env { store, state: State::in_memory().unwrap(), root: tmp.path().join("projects"), _tmp: tmp, _qdrant: qdrant }
}

impl Env {
    fn write(&self, sid: &str, lines: &[String]) {
        let d = self.root.join("alpha");
        fs::create_dir_all(&d).unwrap();
        fs::write(d.join(format!("{sid}.jsonl")), lines.join("\n")).unwrap();
    }

    async fn sync(&self) {
        let refs = claude::discover(&self.root).unwrap();
        let mut e = HashEmbedder::new(DIM);
        let mut ix = Indexer { store: &self.store, embedder: &mut e, counter: &WordCounter, state: &self.state, config: IndexConfig::default() };
        ix.sync(&refs, claude::load, |_| {}, 1).await.unwrap();
    }

    async fn summarize(&self, s: &dyn Summarizer, cfg: &SummaryConfig) -> ingest::SummaryReport {
        summarize_pending(&self.store, &mut HashEmbedder::new(DIM), s, &self.state, cfg, 2).await.unwrap()
    }

    async fn summaries(&self) -> Vec<store::Hit> {
        self.store.scroll(&SearchFilter::default().eq("kind", "summary"), 100).await.unwrap()
    }
}

const BODY: &str = "## Goal\n- ship the api\n## Decisions\n- Qdrant\n## Dead ends\n- none\n## Open\n- none\n## Files\n- api.rs";

#[tokio::test]
async fn main_sessions_get_one_summary_point_each() {
    let e = env().await;
    e.write("multi", &turns(3, ""));
    e.write("single", &turns(1, ""));
    e.sync().await;
    let fake = Fake::ok(BODY);
    let r = e.summarize(&fake, &SummaryConfig::default()).await;
    assert_eq!((r.written, r.failed.len(), r.deferred), (1, 0, 0), "one-turn session is skipped");

    let s = e.summaries().await;
    assert_eq!(s.len(), 1);
    let p = &s[0].payload;
    assert_eq!(p["session_id"], "multi");
    assert_eq!(p["project"], "z:/work/alpha");
    assert_eq!(p["turn_index"], -1);
    assert_eq!(p["turn_count"], 3);
    assert_eq!(p["summarizer"], "fake");
    let text = p["text"].as_str().unwrap();
    assert!(text.contains("Session summary (3 turns)") && text.contains("## Goal\n- ship the api"), "{text}");

    let prompt = &fake.prompts.lock().unwrap()[0];
    assert!(prompt.contains("step 0") && prompt.contains("did step 2"), "prompt carries the transcript");
}

#[tokio::test]
async fn summarised_sessions_are_not_redone_until_they_grow() {
    let e = env().await;
    e.write("s", &turns(3, ""));
    e.sync().await;
    let fake = Fake::ok(BODY);
    e.summarize(&fake, &SummaryConfig::default()).await;
    e.summarize(&fake, &SummaryConfig::default()).await;
    assert_eq!(fake.calls(), 1, "unchanged session is not re-summarised");

    e.write("s", &turns(8, ""));
    e.sync().await;
    e.summarize(&fake, &SummaryConfig::default()).await;
    assert_eq!(fake.calls(), 2, "5 new turns trigger a re-summary");
    let s = e.summaries().await;
    assert_eq!(s.len(), 1, "the summary is overwritten, not duplicated");
    assert_eq!(s[0].payload["turn_count"], 8);
}

#[tokio::test]
async fn rejected_output_is_not_stored_and_is_retried() {
    let e = env().await;
    e.write("s", &turns(2, ""));
    e.sync().await;
    let chatter = Fake { reply: Ok("Would you like me to save this to memory?".into()), prompts: Mutex::default() };
    let r = e.summarize(&chatter, &SummaryConfig::default()).await;
    assert_eq!(r.written, 0);
    assert_eq!(r.failed.len(), 1);
    assert!(e.summaries().await.is_empty());

    let r = e.summarize(&Fake::ok(BODY), &SummaryConfig::default()).await;
    assert_eq!(r.written, 1, "retried on the next pass");
}

#[tokio::test]
async fn summarizer_error_is_reported_per_session() {
    let e = env().await;
    e.write("a", &turns(2, ""));
    e.write("b", &turns(2, ""));
    e.sync().await;
    let broken = Fake { reply: Err("claude exited with 1: not logged in".into()), prompts: Mutex::default() };
    let r = e.summarize(&broken, &SummaryConfig::default()).await;
    assert_eq!(r.failed.len(), 2);
    assert!(r.failed[0].1.contains("not logged in"));
}

#[tokio::test]
async fn exhausted_budget_defers_the_rest() {
    let e = env().await;
    for i in 0..3 {
        e.write(&format!("s{i}"), &turns(2, ""));
    }
    e.sync().await;
    let cfg = SummaryConfig { budget: Duration::ZERO, ..Default::default() };
    let r = e.summarize(&Fake::ok(BODY), &cfg).await;
    assert_eq!((r.written, r.deferred), (0, 3));
}

#[tokio::test]
async fn secrets_never_reach_the_summarizer_or_the_summary() {
    let e = env().await;
    e.write("s", &turns(2, "with token ghp_abcdefghijklmnopqrstuvwxyz0123456789AB"));
    e.sync().await;
    let leaky = Fake::ok(&format!("{BODY}\n- also DB_PASSWORD=hunter2hunter"));
    e.summarize(&leaky, &SummaryConfig::default()).await;
    assert!(!leaky.prompts.lock().unwrap()[0].contains("ghp_"), "input comes from redacted chunks");
    let text = e.summaries().await[0].payload["text"].as_str().unwrap().to_owned();
    assert!(!text.contains("hunter2"), "output is redacted too: {text}");
}

#[tokio::test]
async fn version_reset_forgets_summaries() {
    let e = env().await;
    e.write("s", &turns(2, ""));
    e.sync().await;
    let fake = Fake::ok(BODY);
    e.summarize(&fake, &SummaryConfig::default()).await;
    e.state.ensure_version("v-next").unwrap();
    e.sync().await;
    e.summarize(&fake, &SummaryConfig::default()).await;
    assert_eq!(fake.calls(), 2, "a model or chunker change re-summarises");
}

#[tokio::test]
async fn reindexing_a_growing_session_keeps_its_summary() {
    let e = env().await;
    e.write("s", &turns(4, ""));
    e.sync().await;
    e.summarize(&Fake::ok(BODY), &SummaryConfig::default()).await;
    for n in 5..=8 {
        // 1-4 new turns: re-indexed each time, not yet due for a new summary.
        e.write("s", &turns(n, ""));
        e.sync().await;
        assert_eq!(e.summaries().await.len(), 1, "summary survives re-index at {n} turns");
    }
    let turns_left = e.store.count(&SearchFilter::default().eq("session_id", "s").eq("kind", "turn")).await.unwrap();
    assert_eq!(turns_left, 8, "turn chunks are still pruned and upserted normally");
}

#[tokio::test]
async fn repeatedly_failing_session_stops_being_retried() {
    let e = env().await;
    e.write("s", &turns(3, ""));
    e.sync().await;
    let broken = Fake { reply: Err("timeout".into()), prompts: Mutex::default() };
    for _ in 0..5 {
        e.summarize(&broken, &SummaryConfig::default()).await;
    }
    assert_eq!(broken.calls(), 3, "given up after max_attempts");
    e.write("s", &turns(8, ""));
    e.sync().await;
    e.summarize(&broken, &SummaryConfig::default()).await;
    assert_eq!(broken.calls(), 4, "tried again once the session grew");
}

#[tokio::test]
async fn session_whose_transcript_is_gone_gets_its_turns_backfilled() {
    let e = env().await;
    e.write("s", &turns(3, ""));
    e.sync().await;
    // Simulate a row written before turn counts existed, whose transcript was cleaned up.
    let r = ingest::state::Record { source: "claude", session_id: "s", fingerprint: "", project: "z:/work/alpha", chunks: 3, ..Default::default() };
    e.state.record(&r, 1).unwrap();
    fs::remove_file(e.root.join("alpha/s.jsonl")).unwrap();
    let report = e.summarize(&Fake::ok(BODY), &SummaryConfig::default()).await;
    assert_eq!(report.written, 1);
    assert_eq!(e.summaries().await[0].payload["turn_count"], 3);
}

#[tokio::test]
async fn call_timeout_never_exceeds_the_remaining_budget() {
    struct Clock(Mutex<Vec<Duration>>);
    impl Summarizer for Clock {
        fn id(&self) -> &str {
            "clock"
        }
        fn summarize(&self, _: &str, timeout: Duration) -> Result<String, String> {
            self.0.lock().unwrap().push(timeout);
            Ok(BODY.into())
        }
    }
    let e = env().await;
    e.write("s", &turns(2, ""));
    e.sync().await;
    let clock = Clock(Mutex::default());
    let cfg = SummaryConfig { budget: Duration::from_secs(20), call_timeout: Duration::from_secs(60), ..Default::default() };
    e.summarize(&clock, &cfg).await;
    let t = clock.0.lock().unwrap()[0];
    assert!(t <= Duration::from_secs(20), "{t:?}");
}

#[tokio::test]
async fn listed_secrets_are_redacted_from_summaries() {
    let e = env().await;
    e.write("s", &turns(2, ""));
    e.sync().await;
    let leaky = Fake::ok(&format!("{BODY}\n- SOCN value is plainsecret1"));
    let cfg = SummaryConfig { redactor: ingest::Redactor::new(["plainsecret1".to_owned()]), ..Default::default() };
    e.summarize(&leaky, &cfg).await;
    let text = e.summaries().await[0].payload["text"].as_str().unwrap().to_owned();
    assert!(!text.contains("plainsecret1") && text.contains("[REDACTED:listed]"), "{text}");
}
