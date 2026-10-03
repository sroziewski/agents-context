//! Summaries through the real surfaces: the daemon's index run writes them, and MCP
//! `search_sessions` / `get_session` show them first, scoped by project like everything else.

use std::fs;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ctx::config::Config;
use ctx::embed_server::SharedEmbedder;
use ctx::mcp::ContextServer;
use ctx::queue::{IndexQueue, Target};
use ctx::search::Context;
use embed::{Embedder, HashEmbedder};
use ingest::{Summarizer, WordCounter};
use rmcp::ServiceExt;
use rmcp::model::CallToolRequestParams;
use serde_json::{Value, json};
use store::{SearchFilter, Store};

const DIM: usize = 64;
const ALPHA: &str = r"Z:\work\alpha";
const BETA: &str = r"Z:\work\beta";

/// Summarises by echoing the first user prompt, so each session's summary is recognisable.
struct EchoSummarizer;

impl Summarizer for EchoSummarizer {
    fn id(&self) -> &str {
        "echo"
    }
    fn summarize(&self, prompt: &str, _timeout: std::time::Duration) -> Result<String, String> {
        let first = prompt.lines().find(|l| l.starts_with("User: ")).unwrap_or("User: ?");
        Ok(format!("## Goal\n- {}\n## Decisions\n- none\n## Dead ends\n- none\n## Open\n- none\n## Files\n- none", &first[6..]))
    }
}

fn session(cwd: &str, prompts: &[&str]) -> String {
    prompts
        .iter()
        .enumerate()
        .flat_map(|(i, p)| {
            [
                json!({"type": "user", "uuid": format!("u{i}"), "timestamp": "2026-09-26T10:00:00Z", "cwd": cwd,
                       "origin": {"kind": "human"}, "message": {"content": p}})
                .to_string(),
                json!({"type": "assistant", "timestamp": "2026-09-26T10:00:05Z",
                       "message": {"content": [{"type": "text", "text": format!("done: {p}")}]}})
                .to_string(),
            ]
        })
        .collect::<Vec<_>>()
        .join("\n")
}

struct Env {
    store: Arc<Store>,
    cfg: Config,
    _tmp: tempfile::TempDir,
    _qdrant: testkit::Qdrant,
}

async fn env() -> Env {
    let qdrant = testkit::qdrant().await;
    let tmp = tempfile::tempdir().unwrap();
    let projects = tmp.path().join("projects");
    for (dir, sid, cwd, prompts) in [
        ("a", "alpha-db", ALPHA, vec!["migrate the database schema", "add rollback"]),
        ("b", "beta-db", BETA, vec!["migrate the database schema too", "fix indexes"]),
    ] {
        fs::create_dir_all(projects.join(dir)).unwrap();
        fs::write(projects.join(dir).join(format!("{sid}.jsonl")), session(cwd, &prompts)).unwrap();
    }
    let cfg = Config {
        claude_dir: projects,
        opencode_db: tmp.path().join("none.db"),
        state_path: tmp.path().join("state.db"),
        ..Config::from_env(|_| None)
    };
    let store = Arc::new(Store::connect(&qdrant.url, None, "sessions", DIM).unwrap());
    Env { store, cfg, _tmp: tmp, _qdrant: qdrant }
}

impl Env {
    /// One daemon index run with the echo summarizer; returns its status line.
    async fn daemon_run(&self) -> String {
        let embedder = SharedEmbedder::new(Box::new(HashEmbedder::new(DIM)));
        let runner = ctx::daemon_runner(
            self.cfg.clone(),
            self.store.clone(),
            Arc::new(WordCounter),
            embedder,
            "hash".into(),
            Some(Arc::new(EchoSummarizer)),
            None,
            None,
            None,
        );
        let queue = IndexQueue::start(runner, Duration::from_millis(10));
        queue.request(Target::All);
        for _ in 0..500 {
            let s = queue.status();
            if s.runs == 1 && !s.running {
                assert!(s.last_error.is_none(), "{s:?}");
                return s.last_report.unwrap();
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("run did not finish");
    }

    async fn mcp(&self, current: &str) -> rmcp::service::RunningService<rmcp::RoleClient, ()> {
        let context = Context {
            store: self.store.clone(),
            embedder: Arc::new(Mutex::new(Box::new(HashEmbedder::new(DIM)) as Box<dyn Embedder>)),
        };
        let server = ContextServer::new(context, sources::project::key(current), self.cfg.state_path.clone());
        let (server_io, client_io) = tokio::io::duplex(1 << 16);
        tokio::spawn(async move {
            let running = server.serve(server_io).await.unwrap();
            let _ = running.waiting().await;
        });
        ().serve(client_io).await.unwrap()
    }
}

async fn call(client: &rmcp::service::RunningService<rmcp::RoleClient, ()>, tool: &'static str, args: Value) -> String {
    let r = client.call_tool(CallToolRequestParams::new(tool).with_arguments(args.as_object().cloned().unwrap())).await.unwrap();
    assert_ne!(r.is_error, Some(true), "{r:?}");
    r.content.iter().filter_map(|c| c.as_text().map(|t| t.text.clone())).collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn daemon_run_writes_summaries_after_indexing() {
    let e = env().await;
    let line = e.daemon_run().await;
    assert!(line.contains("indexed 2") && line.contains("summaries 2 written, 0 failed, 0 deferred"), "{line}");
    let s = e.store.scroll(&SearchFilter::default().eq("kind", "summary"), 10).await.unwrap();
    assert_eq!(s.len(), 2);
    let second = e.daemon_run().await;
    assert!(second.contains("summaries 0 written"), "unchanged sessions are not re-summarised: {second}");
}

#[tokio::test(flavor = "multi_thread")]
async fn search_shows_this_projects_summary_first() {
    let e = env().await;
    e.daemon_run().await;
    let client = e.mcp(ALPHA).await;
    let out = call(&client, "search_sessions", json!({"query": "migrate the database schema"})).await;
    assert!(out.starts_with("## Session summaries (machine-written; verify against the turns)\n\n[S1] claude · z:/work/alpha · session alpha-db"), "{out}");
    assert!(out.contains("## Matching turns"), "{out}");
    assert!(!out.contains("z:/work/beta"), "other project's summary leaked: {out}");

    let all = call(&client, "search_sessions", json!({"query": "migrate the database schema", "scope": "all"})).await;
    assert!(all.contains("session alpha-db") && all.contains("session beta-db"), "{all}");
}

#[tokio::test(flavor = "multi_thread")]
async fn get_session_starts_with_the_summary() {
    let e = env().await;
    e.daemon_run().await;
    let client = e.mcp(ALPHA).await;
    let out = call(&client, "get_session", json!({"session_id": "alpha-db"})).await;
    assert!(out.starts_with("--- summary ---\n"), "{out}");
    assert!(out.contains("## Goal\n- migrate the database schema"), "{out}");
    assert!(out.contains("--- turn 1 ---"), "{out}");
}

#[tokio::test(flavor = "multi_thread")]
async fn list_sessions_is_not_polluted_by_summary_points() {
    let e = env().await;
    e.daemon_run().await;
    let client = e.mcp(ALPHA).await;
    let out = call(&client, "list_sessions", json!({})).await;
    assert_eq!(out.lines().count(), 1, "{out}");
}
