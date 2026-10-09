//! Hook → daemon → queue → indexer → Qdrant, end to end over real HTTP and a real Qdrant.

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use ctx::config::Config;
use ctx::embed_server::{SharedEmbedder, service_router};
use ctx::hook::{Outcome, notify};
use ctx::queue::{IndexQueue, Status, Target};
use embed::HashEmbedder;
use ingest::WordCounter;
use serde_json::json;
use store::{SearchFilter, Store};

const DIM: usize = 64;

struct Env {
    url: String,
    store: Arc<Store>,
    transcript: PathBuf,
    _tmp: tempfile::TempDir,
    _qdrant: testkit::Qdrant,
}

fn turn(uuid: &str, text: &str, answer: &str) -> String {
    [
        json!({"type": "user", "uuid": uuid, "timestamp": "2026-09-26T10:00:00Z", "cwd": r"Z:\work\alpha",
               "origin": {"kind": "human"}, "message": {"content": text}})
        .to_string(),
        json!({"type": "assistant", "timestamp": "2026-09-26T10:00:05Z",
               "message": {"content": [{"type": "text", "text": answer}]}})
        .to_string(),
    ]
    .join("\n")
}

async fn env() -> Env {
    env_with(Duration::from_millis(20)).await
}

async fn env_with(debounce: Duration) -> Env {
    let qdrant = testkit::qdrant().await;
    let tmp = tempfile::tempdir().unwrap();
    let projects = tmp.path().join("projects");
    fs::create_dir_all(projects.join("alpha")).unwrap();
    let transcript = projects.join("alpha/s1.jsonl");
    fs::write(&transcript, turn("u1", "deploy the api", "deployed")).unwrap();

    let cfg = Config {
        claude_dir: projects,
        opencode_db: tmp.path().join("missing-opencode.db"),
        state_path: tmp.path().join("state.db"),
        ..Config::from_env(|_| None)
    };
    let store = Arc::new(Store::connect(&qdrant.url, None, "sessions", DIM).unwrap());
    let embedder = SharedEmbedder::new(Box::new(HashEmbedder::new(DIM)));
    let runner = ctx::daemon_runner(cfg, store.clone(), Arc::new(WordCounter), embedder.clone(), "hash".into(), None, None, None, None, None);
    let queue = IndexQueue::start(runner, debounce);
    let app = service_router(embedder, "hash", Some(queue));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    Env { url, store, transcript, _tmp: tmp, _qdrant: qdrant }
}

impl Env {
    async fn hook(&self, target: Target) {
        let url = self.url.clone();
        let outcome = tokio::task::spawn_blocking(move || notify(&url, target, Duration::from_secs(2), |_| None)).await.unwrap();
        assert_eq!(outcome, Outcome::Requested);
    }

    async fn status(&self) -> Status {
        let url = format!("{}/index/status", self.url);
        tokio::task::spawn_blocking(move || ureq::get(&url).call().unwrap().body_mut().read_json().unwrap()).await.unwrap()
    }

    async fn wait_for_runs(&self, runs: u64) -> Status {
        for _ in 0..500 {
            let s = self.status().await;
            if s.runs >= runs && !s.running {
                return s;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("index run {runs} never finished: {:?}", self.status().await);
    }

    async fn chunks(&self) -> u64 {
        self.store.count(&SearchFilter::default().eq("session_id", "s1")).await.unwrap()
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn hook_triggers_indexing_in_the_daemon() {
    let e = env().await;
    e.hook(Target::Claude).await;
    let s = e.wait_for_runs(1).await;
    assert!(s.last_error.is_none(), "{s:?}");
    assert!(s.last_report.as_deref().unwrap().contains("indexed 1"), "{s:?}");
    assert_eq!(s.last_target, Some(Target::Claude));
    assert_eq!(e.chunks().await, 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn new_turn_becomes_searchable_after_the_next_hook() {
    let e = env().await;
    e.hook(Target::Claude).await;
    e.wait_for_runs(1).await;

    let grown = format!("{}\n{}", fs::read_to_string(&e.transcript).unwrap(), turn("u2", "add a health check", "added /health"));
    fs::write(&e.transcript, grown).unwrap();
    e.hook(Target::Claude).await;
    let s = e.wait_for_runs(2).await;
    assert!(s.last_report.as_deref().unwrap().contains("indexed 1"), "{s:?}");
    assert_eq!(e.chunks().await, 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn hook_with_nothing_changed_is_a_cheap_no_op() {
    let e = env().await;
    e.hook(Target::All).await;
    e.wait_for_runs(1).await;
    e.hook(Target::All).await;
    let s = e.wait_for_runs(2).await;
    assert!(s.last_report.as_deref().unwrap().contains("indexed 0 · unchanged 1"), "{s:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn burst_of_hooks_is_coalesced() {
    // Concurrent hooks inside one debounce window, like Stop + SessionEnd firing together.
    let e = Arc::new(env_with(Duration::from_millis(400)).await);
    let hooks: Vec<_> = (0..8)
        .map(|_| {
            let e = e.clone();
            tokio::spawn(async move { e.hook(Target::Claude).await })
        })
        .collect();
    for h in hooks {
        h.await.unwrap();
    }
    e.wait_for_runs(1).await;
    tokio::time::sleep(Duration::from_millis(600)).await;
    let s = e.status().await;
    assert_eq!(s.runs, 1, "8 concurrent hooks must be one run: {s:?}");
    assert_eq!(e.chunks().await, 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn missing_claude_dir_is_skipped_not_an_error() {
    let e = env().await;
    fs::remove_dir_all(e.transcript.parent().unwrap().parent().unwrap()).unwrap();
    e.hook(Target::Claude).await;
    let s = e.wait_for_runs(1).await;
    assert!(s.last_error.is_none(), "{s:?}");
    assert!(s.last_report.as_deref().unwrap().starts_with("scanned 0"), "{s:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn unreachable_qdrant_is_reported_and_the_queue_keeps_working() {
    let tmp = tempfile::tempdir().unwrap();
    let projects = tmp.path().join("projects/alpha");
    fs::create_dir_all(&projects).unwrap();
    fs::write(projects.join("s1.jsonl"), turn("u1", "q", "a")).unwrap();
    let cfg = Config {
        claude_dir: tmp.path().join("projects"),
        opencode_db: tmp.path().join("none.db"),
        state_path: tmp.path().join("state.db"),
        ..Config::from_env(|_| None)
    };
    let store = Arc::new(Store::connect("http://127.0.0.1:9", None, "sessions", DIM).unwrap());
    let embedder = SharedEmbedder::new(Box::new(HashEmbedder::new(DIM)));
    let queue = IndexQueue::start(ctx::daemon_runner(cfg, store, Arc::new(WordCounter), embedder.clone(), "hash".into(), None, None, None, None, None), Duration::from_millis(10));
    for run in 1..=2 {
        queue.request(Target::Claude);
        for _ in 0..500 {
            if queue.status().runs == run && !queue.status().running {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let s = queue.status();
        assert_eq!(s.runs, run, "run {run} must still happen after a failure: {s:?}");
        assert!(s.last_error.is_some(), "{s:?}");
        assert!(s.last_report.is_none());
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn embed_is_served_while_an_index_run_is_in_progress() {
    use ctx::embed_server::RemoteEmbedder;
    use embed::Embedder;
    let gate = Arc::new(tokio::sync::Notify::new());
    let g = gate.clone();
    let runner: ctx::queue::Runner = Arc::new(move |_| {
        let g = g.clone();
        Box::pin(async move {
            g.notified().await;
            Ok("done".into())
        })
    });
    let queue = IndexQueue::start(runner, Duration::from_millis(10));
    let app = service_router(SharedEmbedder::new(Box::new(HashEmbedder::new(DIM))), "hash", Some(queue.clone()));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    queue.request(Target::All);
    while !queue.status().running {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let u = url.clone();
    let embedded = tokio::time::timeout(
        Duration::from_secs(5),
        tokio::task::spawn_blocking(move || RemoteEmbedder::connect(&u).unwrap().embed(&["query during indexing"]).unwrap().len()),
    )
    .await
    .expect("/embed must not wait for the index run")
    .unwrap();
    assert_eq!(embedded, 1);
    assert!(queue.status().running, "the run was still in progress");
    gate.notify_one();
}

#[tokio::test(flavor = "multi_thread")]
async fn cli_index_is_delegated_to_the_daemon() {
    let e = env().await;
    let url = e.url.clone();
    let status = tokio::task::spawn_blocking(move || ctx::hook::delegate_index(&url, Target::Claude, Duration::from_secs(30)))
        .await
        .unwrap()
        .unwrap()
        .expect("daemon is up");
    assert!(status.last_report.unwrap().contains("indexed 1"));
    assert_eq!(e.chunks().await, 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn cli_index_falls_back_when_no_daemon_is_running() {
    let outcome = tokio::task::spawn_blocking(|| ctx::hook::delegate_index("http://127.0.0.1:9", Target::All, Duration::from_secs(5)))
        .await
        .unwrap()
        .unwrap();
    assert!(outcome.is_none());
}
