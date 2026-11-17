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

/// A daemon whose run `n` ends as `outcome(n)`; run 1 blocks until the returned gate is notified.
/// No Qdrant behind it: the runs are scripted.
async fn scripted_daemon(debounce: Duration, outcome: fn(u64) -> Result<String, String>) -> (String, IndexQueue, Arc<tokio::sync::Notify>) {
    let gate = Arc::new(tokio::sync::Notify::new());
    let g = gate.clone();
    let started = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let runner: ctx::queue::Runner = Arc::new(move |_| {
        let g = g.clone();
        let n = started.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
        Box::pin(async move {
            if n == 1 {
                g.notified().await;
            }
            outcome(n).map_err(anyhow::Error::msg)
        })
    });
    let queue = IndexQueue::start(runner, debounce);
    let app = service_router(SharedEmbedder::new(Box::new(HashEmbedder::new(DIM))), "hash", Some(queue.clone()));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (url, queue, gate)
}

/// Calls `delegate_index` while run 1 is in progress and lets run 1 finish only once the
/// CLI's request has been queued behind it. The long debounce leaves the daemon idle with
/// `runs` already increased between run 1 and the CLI's own run 2.
async fn delegate_to_a_busy_daemon(outcome: fn(u64) -> Result<String, String>) -> Status {
    let (url, queue, gate) = scripted_daemon(Duration::from_millis(700), outcome).await;
    assert_eq!(queue.request(Target::All), 1);
    wait_until("run 1 starts", || queue.status().running).await;
    let call = tokio::task::spawn_blocking(move || ctx::hook::delegate_index(&url, Target::Claude, Duration::from_secs(30)));
    wait_until("the CLI's request is queued", || queue.ticket(queue.epoch(), 2) == ctx::queue::Ticket::Pending).await;
    gate.notify_one();
    call.await.unwrap().unwrap().expect("daemon is up")
}

async fn wait_until(what: &str, cond: impl Fn() -> bool) {
    for _ in 0..1000 {
        if cond() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("never happened: {what}");
}

#[tokio::test(flavor = "multi_thread")]
async fn cli_index_against_a_busy_daemon_reports_its_own_run_not_the_one_in_progress() {
    let status = delegate_to_a_busy_daemon(|n| Ok(format!("run {n}"))).await;
    assert_eq!((status.last_report.as_deref(), status.last_error.as_deref()), (Some("run 2"), None));
    assert_eq!((status.running, status.runs, status.last_target), (false, 2, Some(Target::Claude)), "the status is already that of the finished run");
}

#[tokio::test(flavor = "multi_thread")]
async fn cli_index_against_a_busy_daemon_reports_its_own_runs_failure() {
    let status = delegate_to_a_busy_daemon(|n| if n == 2 { Err("run 2 failed: qdrant down".to_owned()) } else { Ok(format!("run {n}")) }).await;
    assert_eq!((status.last_report.as_deref(), status.last_error.as_deref()), (None, Some("run 2 failed: qdrant down")), "run 1's report is not this request's");
    assert_eq!((status.running, status.runs, status.last_target), (false, 2, Some(Target::Claude)));
}

/// A stand-in daemon answering from `reply`, which sees the request line (`METHOD path`) and how
/// many requests came before it. Returns the URL and the log of request lines.
async fn mock_daemon(reply: fn(&str, usize) -> axum::response::Response) -> (String, Arc<std::sync::Mutex<Vec<String>>>) {
    let log = Arc::new(std::sync::Mutex::new(Vec::new()));
    let l = log.clone();
    // The body is read to its end: answering a POST without reading it lets the server close the
    // connection under the client that is still sending.
    let app = axum::Router::new().fallback(move |method: axum::http::Method, uri: axum::http::Uri, _body: String| {
        let l = l.clone();
        async move {
            let line = format!("{method} {}", uri.path());
            let earlier = {
                let mut log = l.lock().unwrap();
                log.push(line.clone());
                log.len() - 1
            };
            reply(&line, earlier)
        }
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (url, log)
}

#[tokio::test(flavor = "multi_thread")]
async fn cli_index_fails_promptly_when_the_daemon_does_not_know_its_ticket() {
    use axum::response::IntoResponse;
    // A daemon that restarted after accepting: it still answers, but the ticket is not its own.
    let (url, log) = mock_daemon(|line, _| match line {
        "GET /index/status" => axum::Json(Status { runs: 4, ..Status::default() }).into_response(),
        "POST /index" => (axum::http::StatusCode::ACCEPTED, axum::Json(ctx::embed_server::IndexAccepted { epoch: 7, ticket: 3 })).into_response(),
        "GET /index/ticket/7/3" => axum::Json(ctx::queue::Ticket::Unknown).into_response(),
        _ => axum::Json(ctx::queue::Ticket::Pending).into_response(),
    })
    .await;
    let start = std::time::Instant::now();
    let err = tokio::task::spawn_blocking(move || ctx::hook::delegate_index(&url, Target::All, Duration::from_secs(30))).await.unwrap().unwrap_err();
    let took = start.elapsed();
    assert_eq!(err.to_string(), "the daemon restarted before the requested index run finished; run the command again");
    assert!(took < Duration::from_secs(5), "must not poll until the 30 s deadline: {took:?}");
    assert_eq!(*log.lock().unwrap(), ["GET /index/status", "POST /index", "GET /index/ticket/7/3"]);
}

#[tokio::test(flavor = "multi_thread")]
async fn cli_index_still_works_against_a_daemon_that_answers_202_without_a_ticket() {
    use axum::response::IntoResponse;
    fn status(runs: u64, running: bool, report: &str) -> Status {
        Status { running, runs, last_target: Some(Target::Opencode), last_report: Some(report.to_owned()), last_error: None, last_finished_ms: Some(1_790_000_000_000 + runs as i64) }
    }
    // The daemon before tickets: an empty 202, then the run shows up in /index/status only.
    let (url, log) = mock_daemon(|line, earlier| match (line, earlier) {
        ("GET /index/status", 0) => axum::Json(status(4, false, "run 4")).into_response(),
        ("POST /index", _) => axum::http::StatusCode::ACCEPTED.into_response(),
        ("GET /index/status", 2) => axum::Json(status(4, true, "run 4")).into_response(),
        ("GET /index/status", _) => axum::Json(status(5, false, "run 5")).into_response(),
        _ => axum::http::StatusCode::NOT_FOUND.into_response(),
    })
    .await;
    let got = tokio::task::spawn_blocking(move || ctx::hook::delegate_index(&url, Target::Opencode, Duration::from_secs(30))).await.unwrap().unwrap();
    assert_eq!(got, Some(status(5, false, "run 5")));
    assert_eq!(*log.lock().unwrap(), ["GET /index/status", "POST /index", "GET /index/status", "GET /index/status"], "the ticket route is never asked");
}

fn mock_status(runs: u64, report: &str) -> Status {
    Status { running: false, runs, last_target: Some(Target::Opencode), last_report: Some(report.to_owned()), last_error: None, last_finished_ms: Some(1_790_000_000_000 + runs as i64) }
}

fn mock_done(report: &str) -> ctx::queue::Ticket {
    ctx::queue::Ticket::Done(ctx::queue::Outcome { report: Some(report.to_owned()), error: None })
}

#[tokio::test(flavor = "multi_thread")]
async fn cli_index_keeps_polling_when_the_ticket_route_hiccups() {
    use axum::response::IntoResponse;
    // The ticket route fails once, then answers something that is not JSON, then the outcome.
    let (url, log) = mock_daemon(|line, earlier| match (line, earlier) {
        ("GET /index/status", 0) => axum::Json(mock_status(4, "run 4")).into_response(),
        ("POST /index", _) => (axum::http::StatusCode::ACCEPTED, axum::Json(ctx::embed_server::IndexAccepted { epoch: 7, ticket: 3 })).into_response(),
        ("GET /index/ticket/7/3", 2) => axum::http::StatusCode::INTERNAL_SERVER_ERROR.into_response(),
        ("GET /index/ticket/7/3", 3) => "<html>busy</html>".into_response(),
        ("GET /index/ticket/7/3", _) => axum::Json(mock_done("the ticket's run")).into_response(),
        ("GET /index/status", _) => axum::Json(mock_status(5, "the status's run")).into_response(),
        _ => axum::http::StatusCode::NOT_FOUND.into_response(),
    })
    .await;
    let got = tokio::task::spawn_blocking(move || ctx::hook::delegate_index(&url, Target::Opencode, Duration::from_secs(30))).await.unwrap().unwrap();
    assert_eq!(got, Some(Status { last_report: Some("the ticket's run".into()), ..mock_status(5, "") }), "the status of the daemon with the ticket's own outcome");
    assert_eq!(
        *log.lock().unwrap(),
        ["GET /index/status", "POST /index", "GET /index/ticket/7/3", "GET /index/ticket/7/3", "GET /index/ticket/7/3", "GET /index/status"]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn cli_index_gives_up_when_its_ticket_is_still_pending_at_the_deadline() {
    use axum::response::IntoResponse;
    let (url, log) = mock_daemon(|line, _| match line {
        "GET /index/status" => axum::Json(mock_status(4, "run 4")).into_response(),
        "POST /index" => (axum::http::StatusCode::ACCEPTED, axum::Json(ctx::embed_server::IndexAccepted { epoch: 7, ticket: 3 })).into_response(),
        "GET /index/ticket/7/3" => axum::Json(ctx::queue::Ticket::Pending).into_response(),
        _ => axum::http::StatusCode::NOT_FOUND.into_response(),
    })
    .await;
    let err = tokio::task::spawn_blocking(move || ctx::hook::delegate_index(&url, Target::All, Duration::from_millis(600))).await.unwrap().unwrap_err();
    assert_eq!(err.to_string(), "daemon accepted the request but did not finish within 600ms; check `/index/status`");
    let log = log.lock().unwrap();
    assert_eq!(log[..3], ["GET /index/status", "POST /index", "GET /index/ticket/7/3"]);
    assert!(log[3..].iter().all(|line| line == "GET /index/ticket/7/3"), "only the ticket is polled: {log:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn cli_index_returns_its_tickets_outcome_even_when_the_status_can_no_longer_be_read() {
    use axum::response::IntoResponse;
    let (url, log) = mock_daemon(|line, earlier| match (line, earlier) {
        ("GET /index/status", 0) => axum::Json(mock_status(4, "run 4")).into_response(),
        ("POST /index", _) => (axum::http::StatusCode::ACCEPTED, axum::Json(ctx::embed_server::IndexAccepted { epoch: 7, ticket: 3 })).into_response(),
        ("GET /index/ticket/7/3", _) => axum::Json(ctx::queue::Ticket::Done(ctx::queue::Outcome { report: None, error: Some("the ticket's run failed".into()) })).into_response(),
        _ => axum::http::StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    })
    .await;
    let got = tokio::task::spawn_blocking(move || ctx::hook::delegate_index(&url, Target::Opencode, Duration::from_secs(30))).await.unwrap().unwrap().expect("the run finished");
    assert_eq!((got.last_report.as_deref(), got.last_error.as_deref()), (None, Some("the ticket's run failed")), "not run 4's report from the earlier status");
    assert_eq!(*log.lock().unwrap(), ["GET /index/status", "POST /index", "GET /index/ticket/7/3", "GET /index/status"]);
}

#[tokio::test(flavor = "multi_thread")]
async fn cli_index_falls_back_when_the_daemon_has_no_index_queue() {
    let app = service_router(SharedEmbedder::new(Box::new(HashEmbedder::new(DIM))), "hash", None);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let outcome = tokio::task::spawn_blocking(move || ctx::hook::delegate_index(&url, Target::All, Duration::from_secs(5))).await.unwrap().unwrap();
    assert_eq!(outcome, None);
}

#[tokio::test(flavor = "multi_thread")]
async fn cli_index_falls_back_when_no_daemon_is_running() {
    let outcome = tokio::task::spawn_blocking(|| ctx::hook::delegate_index("http://127.0.0.1:9", Target::All, Duration::from_secs(5)))
        .await
        .unwrap()
        .unwrap();
    assert!(outcome.is_none());
}
