//! Fixtures shared by the P4 integration tests: Claude session files with chosen timestamps,
//! indexed and summarised by one real daemon run into a throwaway Qdrant.

#![allow(dead_code)]

use std::fs;
use std::sync::Arc;
use std::time::Duration;

use ctx::config::Config;
use ctx::embed_server::SharedEmbedder;
use ctx::queue::{IndexQueue, Target};
use embed::HashEmbedder;
use ingest::{Summarizer, WordCounter};
use serde_json::json;
use store::Store;

pub const DIM: usize = 64;
pub const ALPHA: &str = r"Z:\work\alpha";
pub const BETA: &str = r"Z:\work\beta";

/// Summarises by echoing the first user prompt, so each session's summary is recognisable.
pub struct EchoSummarizer;

impl Summarizer for EchoSummarizer {
    fn id(&self) -> &str {
        "echo"
    }
    fn summarize(&self, prompt: &str, _timeout: Duration) -> Result<String, String> {
        let first = prompt.lines().find(|l| l.starts_with("User: ")).unwrap_or("User: ?");
        Ok(format!("## Goal\n- {}", &first[6..]))
    }
}

pub struct Fixture {
    pub id: &'static str,
    pub cwd: &'static str,
    /// RFC 3339 time of every turn.
    pub at: &'static str,
    pub prompts: &'static [&'static str],
}

fn session(f: &Fixture) -> String {
    f.prompts
        .iter()
        .enumerate()
        .flat_map(|(i, p)| {
            [
                json!({"type": "user", "uuid": format!("u{i}"), "timestamp": f.at, "cwd": f.cwd,
                       "origin": {"kind": "human"}, "message": {"content": p}})
                .to_string(),
                json!({"type": "assistant", "timestamp": f.at,
                       "message": {"content": [{"type": "text", "text": format!("done: {p}")}]}})
                .to_string(),
            ]
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub struct Env {
    pub store: Arc<Store>,
    pub cfg: Config,
    pub tmp: tempfile::TempDir,
    _qdrant: testkit::Qdrant,
}

/// Writes the sessions, indexes and summarises them with one daemon run.
pub async fn indexed(fixtures: &[Fixture]) -> Env {
    let qdrant = testkit::qdrant().await;
    let tmp = tempfile::tempdir().unwrap();
    let projects = tmp.path().join("projects");
    for (i, f) in fixtures.iter().enumerate() {
        let dir = projects.join(format!("p{i}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join(format!("{}.jsonl", f.id)), session(f)).unwrap();
    }
    let cfg = Config {
        claude_dir: projects,
        opencode_db: tmp.path().join("none.db"),
        state_path: tmp.path().join("state.db"),
        redact_file: tmp.path().join("redact.txt"),
        teams_db: tmp.path().join("no-teams.db"),
        ..Config::from_env(|_| None)
    };
    let store = Arc::new(Store::connect(&qdrant.url, None, "sessions", DIM).unwrap());
    let runner = ctx::daemon_runner(
        cfg.clone(),
        store.clone(),
        Arc::new(WordCounter),
        SharedEmbedder::new(Box::new(HashEmbedder::new(DIM))),
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
            return Env { store, cfg, tmp, _qdrant: qdrant };
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("index run did not finish");
}
