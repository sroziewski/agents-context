//! Memory suggestions end to end: summaries in a real Qdrant → fake LLM → candidates → accept.

use std::sync::Mutex;
use std::time::Duration;

use ctx::memories::{accept, suggest};
use embed::{Embedder, HashEmbedder};
use ingest::{Redactor, SESSION_FIELDS, Summarizer};
use serde_json::json;
use store::{Doc, Store, point_id};

const DIM: usize = 64;

struct Fake {
    reply: String,
    prompts: Mutex<Vec<String>>,
}

impl Summarizer for Fake {
    fn id(&self) -> &str {
        "fake"
    }
    fn summarize(&self, prompt: &str, _: Duration) -> Result<String, String> {
        self.prompts.lock().unwrap().push(prompt.to_owned());
        Ok(self.reply.clone())
    }
}

async fn store() -> (Store, testkit::Qdrant) {
    let qdrant = testkit::qdrant().await;
    let store = Store::connect(&qdrant.url, None, "sessions", DIM).unwrap();
    store.ensure_collection(SESSION_FIELDS).await.unwrap();
    let mut e = HashEmbedder::new(DIM);
    let docs = [
        ("a-old", "d:/dev/a", 1, "## Goal\n- old alpha work"),
        ("a-new", "d:/dev/a", 2, "## Goal\n- new alpha work\n## Decisions\n- always run tests with testcontainers"),
        ("b", "d:/dev/b", 3, "## Goal\n- beta work"),
    ]
    .into_iter()
    .map(|(sid, project, ts, text)| Doc {
        id: point_id("claude", &format!("{sid}/summary")),
        embedding: e.embed_one(text).unwrap(),
        payload: json!({"kind": "summary", "source": "claude", "project": project, "session_id": sid, "ts": ts, "text": text})
            .as_object()
            .cloned()
            .unwrap(),
    })
    .collect();
    store.upsert(docs).await.unwrap();
    (store, qdrant)
}

const REPLY: &str = r#"[{"name": "testcontainers-for-integration", "type": "feedback", "description": "integration tests use testcontainers",
  "body": "Run integration tests against real services via testcontainers.\n**Why:** user asked.\n**How to apply:** every change.", "session": "a-new"}]"#;

#[tokio::test]
async fn suggestions_use_only_this_projects_summaries_newest_first() {
    let (store, _q) = store().await;
    let fake = Fake { reply: REPLY.into(), prompts: Mutex::default() };
    let c = suggest(&store, &fake, &Redactor::default(), "d:/dev/a", "### feedback_x.md\nknown fact", Duration::from_secs(5)).await.unwrap();

    assert_eq!(c.len(), 1);
    assert_eq!(c[0].name, "testcontainers-for-integration");
    let prompt = &fake.prompts.lock().unwrap()[0];
    assert!(prompt.contains("new alpha work") && prompt.contains("old alpha work"));
    assert!(!prompt.contains("beta work"), "other project's summaries leaked");
    assert!(prompt.find("new alpha").unwrap() < prompt.find("old alpha").unwrap(), "newest first");
    assert!(prompt.contains("known fact"), "existing memories are included to avoid repeats");
}

#[tokio::test]
async fn project_without_summaries_is_a_clear_error() {
    let (store, _q) = store().await;
    let fake = Fake { reply: "[]".into(), prompts: Mutex::default() };
    let e = suggest(&store, &fake, &Redactor::default(), "d:/dev/none", "", Duration::from_secs(5)).await.unwrap_err();
    assert!(e.to_string().contains("no session summaries"));
    assert!(fake.prompts.lock().unwrap().is_empty(), "no LLM call without input");
}

#[tokio::test]
async fn accepted_suggestion_becomes_a_memory_file_nothing_else_is_written() {
    let (store, _q) = store().await;
    let fake = Fake { reply: REPLY.into(), prompts: Mutex::default() };
    let c = suggest(&store, &fake, &Redactor::default(), "d:/dev/a", "", Duration::from_secs(5)).await.unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("memory");
    assert!(!dir.exists(), "suggest alone writes nothing");
    accept(&dir, &c).unwrap();
    let file = std::fs::read_to_string(dir.join("feedback_testcontainers-for-integration.md")).unwrap();
    assert!(file.starts_with("---\nname: testcontainers-for-integration\n"));
    assert!(file.contains("**How to apply:** every change."));
    assert!(std::fs::read_to_string(dir.join("MEMORY.md")).unwrap().contains("feedback_testcontainers-for-integration.md"));
}
