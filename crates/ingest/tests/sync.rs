//! End-to-end: real Claude transcript files on disk → sync → real Qdrant (testcontainers).

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use embed::{Embedder, Embedding, HashEmbedder};
use ingest::{IndexConfig, Indexer, SESSION_FIELDS, State, WordCounter};
use serde_json::json;
use sources::{SessionRef, SourceError, claude};
use store::{SearchFilter, SearchMode, Store};

const DIM: usize = 64;

/// HashEmbedder that counts how many texts it embedded, to prove unchanged sessions are skipped.
#[derive(Clone)]
struct Counting {
    inner: HashEmbedder,
    texts: Arc<AtomicUsize>,
}

impl Embedder for Counting {
    fn dim(&self) -> usize {
        DIM
    }
    fn embed(&mut self, texts: &[&str]) -> embed::Result<Vec<Embedding>> {
        self.texts.fetch_add(texts.len(), Ordering::SeqCst);
        self.inner.embed(texts)
    }
}

struct Env {
    store: Store,
    state: State,
    embedder: Counting,
    root: PathBuf,
    _tmp: tempfile::TempDir,
    _qdrant: testkit::Qdrant,
}

async fn env() -> Env {
    let qdrant = testkit::qdrant().await;
    let store = Store::connect(&qdrant.url, None, "sessions", DIM).unwrap();
    store.ensure_collection(SESSION_FIELDS).await.unwrap();
    let tmp = tempfile::tempdir().unwrap();
    Env {
        store,
        state: State::in_memory().unwrap(),
        embedder: Counting { inner: HashEmbedder::new(DIM), texts: Arc::default() },
        root: tmp.path().join("projects"),
        _tmp: tmp,
        _qdrant: qdrant,
    }
}

fn prompt(uuid: &str, cwd: &str, text: &str) -> String {
    json!({"type": "user", "uuid": uuid, "timestamp": "2026-09-25T10:00:00Z", "cwd": cwd,
           "origin": {"kind": "human"}, "message": {"content": text}})
    .to_string()
}

fn reply(text: &str) -> String {
    json!({"type": "assistant", "timestamp": "2026-09-25T10:00:05Z",
           "message": {"content": [{"type": "text", "text": text}]}})
    .to_string()
}

impl Env {
    fn write_session(&self, dir: &str, sid: &str, lines: &[String]) -> PathBuf {
        let d = self.root.join(dir);
        fs::create_dir_all(&d).unwrap();
        let p = d.join(format!("{sid}.jsonl"));
        fs::write(&p, lines.join("\n")).unwrap();
        p
    }

    fn refs(&self) -> Vec<SessionRef> {
        claude::discover(&self.root).unwrap()
    }

    async fn sync_with(&mut self, refs: &[SessionRef], load: impl Fn(&SessionRef) -> sources::Result<sources::Session>) -> ingest::Report {
        let mut indexer = Indexer {
            store: &self.store,
            embedder: &mut self.embedder,
            counter: &WordCounter,
            state: &self.state,
            config: IndexConfig::default(),
        };
        indexer.sync(refs, load, |_| {}, 0).await.unwrap()
    }

    async fn sync(&mut self) -> ingest::Report {
        let refs = self.refs();
        self.sync_with(&refs, claude::load).await
    }

    async fn count(&self, f: SearchFilter) -> u64 {
        self.store.count(&f).await.unwrap()
    }

    fn embedded(&self) -> usize {
        self.embedder.texts.load(Ordering::SeqCst)
    }

    async fn search_projects(&mut self, query: &str, project: Option<&str>) -> Vec<String> {
        let q = self.embedder.inner.embed_one(query).unwrap();
        let mut f = SearchFilter::default();
        if let Some(p) = project {
            f = f.eq("project", p);
        }
        self.store
            .search(&q, &f, 20, SearchMode::Hybrid)
            .await
            .unwrap()
            .into_iter()
            .map(|h| h.payload["project"].as_str().unwrap().to_owned())
            .collect()
    }
}

const ALPHA: &str = r"Z:\work\alpha";
const BETA: &str = r"Z:\work\beta";

#[tokio::test]
async fn first_sync_indexes_every_session_and_records_state() {
    let mut e = env().await;
    e.write_session("a", "s1", &[prompt("u1", ALPHA, "deploy the api"), reply("deployed")]);
    e.write_session("a", "s2", &[prompt("u1", ALPHA, "fix tests"), reply("fixed"), prompt("u2", ALPHA, "again"), reply("ok")]);
    e.write_session("b", "s3", &[prompt("u1", BETA, "write docs"), reply("written")]);

    let r = e.sync().await;
    assert_eq!((r.scanned, r.indexed, r.unchanged, r.chunks), (3, 3, 0, 4));
    assert!(r.failed.is_empty());
    assert_eq!(e.count(SearchFilter::default()).await, 4);
    assert_eq!(e.count(SearchFilter::default().eq("project", "z:/work/alpha")).await, 3);
    assert_eq!(e.state.get("claude", "s2").unwrap().unwrap().chunks, 2);
}

#[tokio::test]
async fn unchanged_sessions_are_not_re_embedded() {
    let mut e = env().await;
    e.write_session("a", "s1", &[prompt("u1", ALPHA, "q"), reply("a")]);
    e.sync().await;
    let embedded = e.embedded();

    let r = e.sync().await;
    assert_eq!((r.indexed, r.unchanged), (0, 1));
    assert_eq!(e.embedded(), embedded, "no embedding calls for unchanged sessions");
}

#[tokio::test]
async fn grown_session_is_reindexed_without_duplicates() {
    let mut e = env().await;
    let first = [prompt("u1", ALPHA, "step one"), reply("done one")];
    e.write_session("a", "s1", &first);
    e.sync().await;

    let mut grown = first.to_vec();
    grown.extend([prompt("u2", ALPHA, "step two"), reply("done two")]);
    e.write_session("a", "s1", &grown);
    let r = e.sync().await;
    assert_eq!(r.indexed, 1);
    assert_eq!(e.count(SearchFilter::default().eq("session_id", "s1")).await, 2);
}

#[tokio::test]
async fn rewritten_session_prunes_chunks_that_no_longer_exist() {
    let mut e = env().await;
    e.write_session("a", "s1", &[prompt("u1", ALPHA, "one"), reply("1"), prompt("u2", ALPHA, "two"), reply("2")]);
    e.sync().await;
    assert_eq!(e.count(SearchFilter::default().eq("session_id", "s1")).await, 2);

    e.write_session("a", "s1", &[prompt("u1", ALPHA, "one, edited"), reply("1")]);
    e.sync().await;
    let left = e.store.scroll(&SearchFilter::default().eq("session_id", "s1"), 10).await.unwrap();
    assert_eq!(left.len(), 1);
    assert!(left[0].payload["text"].as_str().unwrap().contains("one, edited"));
}

#[tokio::test]
async fn project_filter_isolates_projects() {
    let mut e = env().await;
    e.write_session("a", "s1", &[prompt("u1", ALPHA, "database migration failed"), reply("timeout")]);
    e.write_session("b", "s2", &[prompt("u1", BETA, "database migration failed"), reply("timeout")]);
    e.sync().await;

    let scoped = e.search_projects("database migration", Some("z:/work/alpha")).await;
    assert_eq!(scoped, ["z:/work/alpha"]);
    let mut all = e.search_projects("database migration", None).await;
    all.sort();
    assert_eq!(all, ["z:/work/alpha", "z:/work/beta"]);
}

#[tokio::test]
async fn deleted_transcript_keeps_its_points() {
    let mut e = env().await;
    let path = e.write_session("a", "s1", &[prompt("u1", ALPHA, "remember this"), reply("ok")]);
    e.sync().await;
    fs::remove_file(&path).unwrap();

    let r = e.sync().await;
    assert_eq!(r.scanned, 0);
    assert_eq!(e.count(SearchFilter::default().eq("session_id", "s1")).await, 1, "index outlives Claude's cleanup");
}

#[tokio::test]
async fn version_change_reindexes_everything() {
    let mut e = env().await;
    e.write_session("a", "s1", &[prompt("u1", ALPHA, "q"), reply("a")]);
    e.state.ensure_version("v1").unwrap();
    e.sync().await;
    assert!(e.state.ensure_version("v2").unwrap());

    let r = e.sync().await;
    assert_eq!(r.indexed, 1);
    assert_eq!(e.count(SearchFilter::default()).await, 1, "same ids overwrite, no duplicates");
}

#[tokio::test]
async fn unreadable_session_is_reported_and_others_still_index() {
    let mut e = env().await;
    e.write_session("a", "good", &[prompt("u1", ALPHA, "fine"), reply("ok")]);
    e.write_session("a", "bad", &[prompt("u1", ALPHA, "fine"), reply("ok")]);
    let refs = e.refs();
    let mut events = Vec::new();
    let r = {
        let mut indexer = Indexer {
            store: &e.store,
            embedder: &mut e.embedder,
            counter: &WordCounter,
            state: &e.state,
            config: IndexConfig::default(),
        };
        indexer
            .sync(
                &refs,
                |r| if r.id == "bad" { Err(SourceError::NotFound(r.id.clone())) } else { claude::load(r) },
                |p| events.push(format!("{p:?}")),
                0,
            )
            .await
            .unwrap()
    };
    assert_eq!(r.indexed, 1);
    assert_eq!(r.failed.len(), 1);
    assert_eq!(r.failed[0].0, "bad");
    assert!(events.iter().any(|ev| ev.starts_with("Failed")));
    assert!(e.state.get("claude", "bad").unwrap().is_none(), "failed session is retried next sync");
}

#[tokio::test]
async fn empty_session_is_remembered_not_retried() {
    let mut e = env().await;
    e.write_session("a", "noprompts", &[json!({"type": "mode"}).to_string()]);
    let r = e.sync().await;
    assert_eq!(r.empty, 1);
    let r = e.sync().await;
    assert_eq!((r.empty, r.unchanged), (0, 1));
}

#[tokio::test]
async fn secrets_never_reach_the_store() {
    let mut e = env().await;
    e.write_session(
        "a",
        "s1",
        &[prompt("u1", ALPHA, "connect with postgres://app:SuperSecret99@db:5432/x and token ghp_abcdefghijklmnopqrstuvwxyz0123456789AB"), reply("connected")],
    );
    let r = e.sync().await;
    assert_eq!(r.redactions, 2);
    for hit in e.store.scroll(&SearchFilter::default(), 100).await.unwrap() {
        let text = hit.payload["text"].as_str().unwrap();
        assert!(!text.contains("SuperSecret99") && !text.contains("ghp_"), "{text}");
    }
}

#[tokio::test]
async fn subagent_transcripts_are_indexed_and_flagged() {
    let mut e = env().await;
    e.write_session("a", "s1", &[prompt("u1", ALPHA, "main work"), reply("ok")]);
    let sub = e.root.join("a/s1/subagents");
    fs::create_dir_all(&sub).unwrap();
    let explore = json!({"type": "user", "uuid": "x1", "cwd": ALPHA, "message": {"content": "explore the repo"}}).to_string();
    fs::write(sub.join("agent-1.jsonl"), [explore, reply("found it")].join("\n")).unwrap();
    e.sync().await;

    assert_eq!(e.count(SearchFilter::default().flag("subagent", true)).await, 1);
    assert_eq!(e.count(SearchFilter::default().flag("subagent", false)).await, 1);
    let sub_hit = e.store.scroll(&SearchFilter::default().flag("subagent", true), 1).await.unwrap();
    assert_eq!(sub_hit[0].payload["parent_id"], "s1");
}

#[test]
fn fixture_paths_are_not_inside_a_git_repo() {
    // Project keys in these tests assume no enclosing repo; Z: must not exist or be a repo.
    assert!(!Path::new(r"Z:\work\.git").exists());
}

#[tokio::test]
async fn grown_session_embeds_only_new_chunks() {
    let mut e = env().await;
    let mut lines = vec![prompt("u1", ALPHA, "one"), reply("1"), prompt("u2", ALPHA, "two"), reply("2")];
    e.write_session("a", "s1", &lines);
    e.sync().await;
    let before = e.embedded();
    assert_eq!(before, 2);

    lines.extend([prompt("u3", ALPHA, "three"), reply("3")]);
    e.write_session("a", "s1", &lines);
    e.sync().await;
    assert_eq!(e.embedded() - before, 1, "only the appended turn is embedded");
    assert_eq!(e.count(SearchFilter::default().eq("session_id", "s1")).await, 3);
}

/// HashEmbedder that fails on any batch holding a text with "POISON" while `poisoned` is set.
struct Poisonable {
    inner: HashEmbedder,
    poisoned: Arc<std::sync::atomic::AtomicBool>,
}

impl Embedder for Poisonable {
    fn dim(&self) -> usize {
        DIM
    }
    fn embed(&mut self, texts: &[&str]) -> embed::Result<Vec<Embedding>> {
        if self.poisoned.load(Ordering::SeqCst) && texts.iter().any(|t| t.contains("POISON")) {
            return Err(embed::EmbedError::Model("cuda out of memory".into()));
        }
        self.inner.embed(texts)
    }
}

#[tokio::test]
async fn a_session_whose_indexing_fails_is_reported_and_retried_and_later_ones_still_index() {
    let mut e = env().await;
    e.write_session("a", "s1-fine", &[prompt("u1", ALPHA, "first fine session"), reply("ok")]);
    e.write_session("a", "s2-bad", &[prompt("u1", ALPHA, "POISON pill prompt"), reply("ok")]);
    e.write_session("a", "s3-fine", &[prompt("u1", ALPHA, "third fine session"), reply("ok")]);
    let mut refs = e.refs();
    refs.sort_by(|a, b| a.id.cmp(&b.id));
    assert_eq!(refs.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(), ["s1-fine", "s2-bad", "s3-fine"], "the failing session is in the middle");
    let poisoned = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let mut embedder = Poisonable { inner: HashEmbedder::new(DIM), poisoned: poisoned.clone() };

    let mut events = Vec::new();
    let r = {
        let mut indexer = Indexer { store: &e.store, embedder: &mut embedder, counter: &WordCounter, state: &e.state, config: IndexConfig::default() };
        indexer.sync(&refs, claude::load, |p| events.push(format!("{p:?}")), 0).await.unwrap()
    };

    assert_eq!((r.scanned, r.indexed, r.chunks), (3, 2, 2));
    assert_eq!(r.failed, [("s2-bad".to_owned(), "model error: cuda out of memory".to_owned())]);
    assert_eq!(
        events,
        [
            "Indexed { id: \"s1-fine\", chunks: 1 }",
            "Failed { id: \"s2-bad\", error: \"model error: cuda out of memory\" }",
            "Indexed { id: \"s3-fine\", chunks: 1 }",
        ]
    );
    assert!(e.state.get("claude", "s2-bad").unwrap().is_none(), "not recorded, so the next sync retries it");
    assert_eq!(e.count(SearchFilter::default().eq("session_id", "s3-fine")).await, 1, "the session after the failure is indexed");
    assert_eq!(e.count(SearchFilter::default().eq("session_id", "s2-bad")).await, 0);

    poisoned.store(false, Ordering::SeqCst);
    let again = {
        let mut indexer = Indexer { store: &e.store, embedder: &mut embedder, counter: &WordCounter, state: &e.state, config: IndexConfig::default() };
        indexer.sync(&refs, claude::load, |_| {}, 0).await.unwrap()
    };
    assert_eq!((again.indexed, again.unchanged, again.failed.len()), (1, 2, 0), "only the failed session is indexed again");
    assert_eq!(e.count(SearchFilter::default().eq("session_id", "s2-bad")).await, 1);
}

fn prompt_at(uuid: &str, at: &str, text: &str) -> String {
    json!({"type": "user", "uuid": uuid, "timestamp": at, "cwd": ALPHA, "origin": {"kind": "human"}, "message": {"content": text}}).to_string()
}

fn ms(at: &str) -> i64 {
    at.parse::<jiff::Timestamp>().unwrap().as_millisecond()
}

/// `(turn_index, ts)` of every stored turn of `session`, sorted.
async fn stored_turns(e: &Env, session: &str) -> Vec<(i64, i64)> {
    let mut v: Vec<(i64, i64)> = e
        .store
        .scroll(&SearchFilter::default().eq("session_id", session), 50)
        .await
        .unwrap()
        .iter()
        .map(|h| (h.payload["turn_index"].as_i64().unwrap(), h.payload["ts"].as_i64().unwrap()))
        .collect();
    v.sort();
    v
}

#[tokio::test]
async fn chunks_at_or_before_the_prune_cutoff_are_not_written_and_leave_the_scope() {
    let mut e = env().await;
    let (t0, t1, t2) = ("2025-03-01T10:00:00Z", "2026-02-01T10:00:00Z", "2026-09-01T10:00:00Z");
    let lines = [prompt_at("u0", t0, "an old question"), reply("old"), prompt_at("u1", t1, "the cutoff question"), reply("mid"), prompt_at("u2", t2, "a new question"), reply("new")];
    e.write_session("a", "s1", &lines);
    e.sync().await;
    assert_eq!(stored_turns(&e, "s1").await, [(0, ms(t0)), (1, ms(t1)), (2, ms(t2))], "precondition: all three turns stored");

    // The cutoff equals turn 1's time: turn 0 is before it, turn 1 at it, both go.
    e.state.forget("claude", "s1").unwrap();
    let refs = e.refs();
    let config = IndexConfig { prune_before_ms: Some(ms(t1)), ..IndexConfig::default() };
    let before = e.embedded();
    let r = {
        let mut indexer = Indexer { store: &e.store, embedder: &mut e.embedder, counter: &WordCounter, state: &e.state, config };
        indexer.sync(&refs, claude::load, |_| {}, 0).await.unwrap()
    };

    assert_eq!((r.indexed, r.chunks), (1, 1));
    assert_eq!(stored_turns(&e, "s1").await, [(2, ms(t2))], "the old turns left the session's points");
    assert_eq!(e.embedded(), before, "the kept turn was unchanged, so nothing was embedded");
}

#[tokio::test]
async fn a_session_entirely_before_the_cutoff_is_written_as_nothing() {
    let mut e = env().await;
    e.write_session("a", "old", &[prompt_at("u0", "2025-03-01T10:00:00Z", "an old question"), reply("old")]);
    let refs = e.refs();
    let config = IndexConfig { prune_before_ms: Some(ms("2026-01-01T00:00:00Z")), ..IndexConfig::default() };
    let r = {
        let mut indexer = Indexer { store: &e.store, embedder: &mut e.embedder, counter: &WordCounter, state: &e.state, config };
        indexer.sync(&refs, claude::load, |_| {}, 0).await.unwrap()
    };
    assert_eq!((r.indexed, r.chunks, r.failed.len()), (1, 0, 0));
    assert_eq!(e.count(SearchFilter::default()).await, 0);
    assert_eq!(e.embedded(), 0);
}

#[tokio::test]
async fn edited_turn_is_re_embedded_and_unchanged_turns_are_not() {
    let mut e = env().await;
    e.write_session("a", "s1", &[prompt("u1", ALPHA, "one"), reply("1"), prompt("u2", ALPHA, "two"), reply("2")]);
    e.sync().await;
    let before = e.embedded();

    e.write_session("a", "s1", &[prompt("u1", ALPHA, "one"), reply("1"), prompt("u2", ALPHA, "two"), reply("2, with more detail")]);
    e.sync().await;
    assert_eq!(e.embedded() - before, 1);
    let texts: Vec<String> = e
        .store
        .scroll(&SearchFilter::default().eq("session_id", "s1"), 10)
        .await
        .unwrap()
        .iter()
        .map(|h| h.payload["text"].as_str().unwrap().to_owned())
        .collect();
    assert!(texts.iter().any(|t| t.contains("with more detail")));
}
