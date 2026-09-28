//! MCP protocol end-to-end: fixture sessions from two projects are indexed into a real Qdrant
//! (testcontainers), then a real MCP client talks to the server over an in-memory duplex pipe.

use std::fs;
use std::sync::{Arc, Mutex};

use ctx::mcp::ContextServer;
use ctx::search::Context;
use embed::{Embedder, HashEmbedder};
use ingest::{State, WordCounter};
use rmcp::model::{CallToolRequestParams, CallToolResult};
use rmcp::service::RunningService;
use rmcp::{RoleClient, ServiceExt};
use serde_json::{Value, json};
use store::Store;

const DIM: usize = 64;
const ALPHA: &str = r"Z:\work\alpha";
const BETA: &str = r"Z:\work\beta";

struct Env {
    client: RunningService<RoleClient, ()>,
    _tmp: tempfile::TempDir,
    _qdrant: testkit::Qdrant,
}

fn line(v: Value) -> String {
    v.to_string()
}

fn session(cwd: &str, turns: &[(&str, &str)]) -> String {
    let mut out = vec![line(json!({"type": "ai-title", "aiTitle": format!("{} work", if cwd == ALPHA { "Alpha" } else { "Beta" })}))];
    for (i, (q, a)) in turns.iter().enumerate() {
        out.push(line(json!({"type": "user", "uuid": format!("u{i}"), "timestamp": format!("2026-09-2{}T10:00:00Z", i + 1),
                             "cwd": cwd, "origin": {"kind": "human"}, "message": {"content": q}})));
        out.push(line(json!({"type": "assistant", "timestamp": format!("2026-09-2{}T10:00:05Z", i + 1),
                             "message": {"content": [{"type": "text", "text": a}]}})));
    }
    out.join("\n")
}

async fn env() -> Env {
    let qdrant = testkit::qdrant().await;
    let store = Store::connect(&qdrant.url, None, "sessions", DIM).unwrap();

    let tmp = tempfile::tempdir().unwrap();
    let projects = tmp.path().join("projects");
    for (dir, sid, cwd, turns) in [
        ("a", "alpha-1", ALPHA, vec![("database migration failed with timeout", "raised the lock timeout"), ("add retry to migration", "added retry with backoff")]),
        ("a", "alpha-2", ALPHA, vec![("style the dashboard", "new palette")]),
        ("b", "beta-1", BETA, vec![("database migration failed again", "dropped the index first")]),
    ] {
        fs::create_dir_all(projects.join(dir)).unwrap();
        fs::write(projects.join(dir).join(format!("{sid}.jsonl")), session(cwd, &turns)).unwrap();
    }
    let state_path = tmp.path().join("state.db");
    let state = State::open(&state_path).unwrap();
    let refs = sources::claude::discover(&projects).unwrap();
    let mut embedder = HashEmbedder::new(DIM);
    let report = ctx::run_index(&store, &mut embedder, "hash", &WordCounter, &state, &refs, &ingest::Redactor::default(), |_| {}).await.unwrap();
    assert_eq!(report.indexed, 3);

    let context = Context {
        store: Arc::new(store),
        embedder: Arc::new(Mutex::new(Box::new(HashEmbedder::new(DIM)) as Box<dyn Embedder>)),
    };
    let server = ContextServer::new(context, sources::project::key(ALPHA), state_path);
    let (server_io, client_io) = tokio::io::duplex(1 << 16);
    tokio::spawn(async move {
        let running = server.serve(server_io).await.unwrap();
        let _ = running.waiting().await;
    });
    let client = ().serve(client_io).await.unwrap();
    Env { client, _tmp: tmp, _qdrant: qdrant }
}

impl Env {
    async fn call(&self, tool: &'static str, args: Value) -> CallToolResult {
        let params = CallToolRequestParams::new(tool).with_arguments(args.as_object().cloned().unwrap());
        self.client.call_tool(params).await.unwrap()
    }

    async fn text(&self, tool: &'static str, args: Value) -> String {
        let r = self.call(tool, args).await;
        assert_ne!(r.is_error, Some(true), "tool error: {r:?}");
        text_of(&r)
    }
}

fn text_of(r: &CallToolResult) -> String {
    r.content.iter().filter_map(|c| c.as_text().map(|t| t.text.clone())).collect::<Vec<_>>().join("\n")
}

#[tokio::test]
async fn lists_the_tools_with_schemas_and_instructions() {
    let e = env().await;
    let tools = e.client.list_all_tools().await.unwrap();
    let mut names: Vec<_> = tools.iter().map(|t| t.name.to_string()).collect();
    names.sort();
    assert_eq!(names, ["deep_search_sessions", "get_jira_issue", "get_session", "get_teams_thread", "list_projects", "list_sessions", "search_jira", "search_sessions", "search_teams"]);
    let search = tools.iter().find(|t| t.name == "search_sessions").unwrap();
    let props = search.input_schema.get("properties").unwrap();
    assert!(props.get("query").is_some() && props.get("scope").is_some());
    let info = e.client.peer_info().unwrap();
    assert!(info.instructions.as_deref().unwrap().contains("z:/work/alpha"));
}

#[tokio::test]
async fn search_defaults_to_the_current_project() {
    let e = env().await;
    let out = e.text("search_sessions", json!({"query": "database migration failed"})).await;
    assert!(out.contains("z:/work/alpha"), "{out}");
    assert!(!out.contains("z:/work/beta"), "other project leaked into default scope:\n{out}");
    assert!(out.starts_with("[1] claude · z:/work/alpha · session alpha-1"), "{out}");
}

#[tokio::test]
async fn scope_all_and_explicit_project_reach_other_projects() {
    let e = env().await;
    let all = e.text("search_sessions", json!({"query": "database migration failed", "scope": "all"})).await;
    assert!(all.contains("z:/work/alpha") && all.contains("z:/work/beta"), "{all}");
    let beta = e.text("search_sessions", json!({"query": "database migration", "project": r"Z:\work\BETA"})).await;
    assert!(beta.contains("z:/work/beta") && !beta.contains("z:/work/alpha"), "{beta}");
}

#[tokio::test]
async fn date_filters_restrict_turns() {
    let e = env().await;
    // alpha-1 turn 1 is dated 2026-09-21, turn 2 2026-09-22 (UTC).
    let out = e.text("search_sessions", json!({"query": "migration", "since": "2026-09-22", "until": "2026-09-22"})).await;
    assert!(out.contains("add retry"), "{out}");
    assert!(!out.contains("lock timeout"), "{out}");
}

#[tokio::test]
async fn invalid_arguments_return_tool_errors_not_crashes() {
    let e = env().await;
    for args in [
        json!({"query": "x", "since": "21.09.2026"}),
        json!({"query": "x", "scope": "everything"}),
        json!({"query": "x", "source": "cursor"}),
        json!({"query": "   "}),
    ] {
        let r = e.call("search_sessions", args.clone()).await;
        assert_eq!(r.is_error, Some(true), "{args} should be a tool error");
    }
    // The server keeps working after errors.
    assert!(e.text("list_sessions", json!({})).await.contains("alpha"));
}

#[tokio::test]
async fn get_session_returns_turns_in_order() {
    let e = env().await;
    let out = e.text("get_session", json!({"session_id": "alpha-1"})).await;
    let first = out.find("database migration failed with timeout").unwrap();
    let second = out.find("add retry to migration").unwrap();
    assert!(first < second, "{out}");
    assert!(out.starts_with("--- turn 1 ---"));
    assert_eq!(e.text("get_session", json!({"session_id": "nope"})).await, "Session not found.");
}

#[tokio::test]
async fn list_sessions_is_scoped_and_newest_first() {
    let e = env().await;
    let out = e.text("list_sessions", json!({})).await;
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines.len(), 2, "{out}");
    assert!(lines.iter().all(|l| l.contains("Alpha work")));
    let all = e.text("list_sessions", json!({"scope": "all"})).await;
    assert_eq!(all.lines().count(), 3);
}

#[tokio::test]
async fn list_projects_marks_the_current_one() {
    let e = env().await;
    let out = e.text("list_projects", json!({})).await;
    assert!(out.starts_with("current project: z:/work/alpha"));
    assert!(out.contains("z:/work/alpha (current): 2 sessions, 3 chunks"), "{out}");
    assert!(out.contains("z:/work/beta: 1 sessions, 1 chunks"), "{out}");
}
