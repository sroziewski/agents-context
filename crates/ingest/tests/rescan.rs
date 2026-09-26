//! rescan against a real Qdrant: stored text written before a rule or listed secret existed.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use embed::{Embedder, Embedding, HashEmbedder};
use ingest::{Redactor, SESSION_FIELDS, rescan};
use serde_json::json;
use store::{Doc, SearchFilter, Store, point_id};

const DIM: usize = 64;

struct Counting(HashEmbedder, Arc<AtomicUsize>);

impl Embedder for Counting {
    fn dim(&self) -> usize {
        DIM
    }
    fn embed(&mut self, texts: &[&str]) -> embed::Result<Vec<Embedding>> {
        self.1.fetch_add(texts.len(), Ordering::SeqCst);
        self.0.embed(texts)
    }
}

async fn store_with(texts: &[(&str, &str, &str)]) -> (Store, testkit::Qdrant) {
    let qdrant = testkit::qdrant().await;
    let store = Store::connect(&qdrant.url, None, "sessions", DIM).unwrap();
    store.ensure_collection(SESSION_FIELDS).await.unwrap();
    let mut e = HashEmbedder::new(DIM);
    // Written raw, bypassing redaction, as an older version of the indexer would have.
    let docs = texts
        .iter()
        .map(|(key, kind, text)| Doc {
            id: point_id("claude", key),
            embedding: e.embed_one(text).unwrap(),
            payload: json!({"kind": kind, "session_id": "s", "project": "p", "text": text}).as_object().cloned().unwrap(),
        })
        .collect();
    store.upsert(docs).await.unwrap();
    (store, qdrant)
}

async fn texts(store: &Store) -> Vec<String> {
    let mut t: Vec<String> = store
        .scroll(&SearchFilter::default(), 100)
        .await
        .unwrap()
        .into_iter()
        .map(|h| h.payload["text"].as_str().unwrap().to_owned())
        .collect();
    t.sort();
    t
}

#[tokio::test]
async fn listed_secret_is_removed_from_chunks_and_summaries_only_where_present() {
    let (store, _q) = store_with(&[
        ("a", "turn", "User: set SOCN to plainsecret1 please"),
        ("b", "summary", "## Decisions\n- SOCN value plainsecret1"),
        ("c", "turn", "User: unrelated work"),
    ])
    .await;
    let calls = Arc::new(AtomicUsize::new(0));
    let mut e = Counting(HashEmbedder::new(DIM), calls.clone());
    let r = rescan(&store, &mut e, &Redactor::new(["plainsecret1".to_owned()])).await.unwrap();

    assert_eq!((r.checked, r.changed, r.redactions), (3, 2, 2));
    assert_eq!(calls.load(Ordering::SeqCst), 2, "only changed points are re-embedded");
    let all = texts(&store).await;
    assert!(all.iter().all(|t| !t.contains("plainsecret1")), "{all:?}");
    assert!(all.contains(&"User: unrelated work".to_owned()));
    let summary = store.scroll(&SearchFilter::default().eq("kind", "summary"), 10).await.unwrap();
    assert_eq!(summary[0].payload["session_id"], "s", "other payload fields are kept");
}

#[tokio::test]
async fn new_builtin_rule_applies_to_old_points() {
    let (store, _q) = store_with(&[("a", "turn", "Update the system password to bcrypt 2a hash `Xq7dHgwQaKi82p` now")]).await;
    let r = rescan(&store, &mut HashEmbedder::new(DIM), &Redactor::default()).await.unwrap();
    assert_eq!(r.changed, 1);
    assert!(!texts(&store).await[0].contains("Xq7dHgwQaKi82p"));
}

#[tokio::test]
async fn rescan_is_idempotent() {
    let (store, _q) = store_with(&[("a", "turn", "value plainsecret1 here")]).await;
    let redactor = Redactor::new(["plainsecret1".to_owned()]);
    rescan(&store, &mut HashEmbedder::new(DIM), &redactor).await.unwrap();
    let again = rescan(&store, &mut HashEmbedder::new(DIM), &redactor).await.unwrap();
    assert_eq!((again.changed, again.redactions), (0, 0));
}

#[tokio::test]
async fn many_changed_points_are_all_rewritten_in_small_batches() {
    let owned: Vec<(String, String)> = (0..11).map(|i| (format!("k{i}"), format!("turn {i} plainsecret1"))).collect();
    let refs: Vec<(&str, &str, &str)> = owned.iter().map(|(k, t)| (k.as_str(), "turn", t.as_str())).collect();
    let (store, _q) = store_with(&refs).await;
    let r = rescan(&store, &mut HashEmbedder::new(DIM), &Redactor::new(["plainsecret1".to_owned()])).await.unwrap();
    assert_eq!(r.changed, 11);
    assert!(texts(&store).await.iter().all(|t| t.ends_with("[REDACTED:listed]")));
}

#[tokio::test]
async fn titles_are_redacted_too() {
    let qdrant = testkit::qdrant().await;
    let store = Store::connect(&qdrant.url, None, "sessions", DIM).unwrap();
    store.ensure_collection(SESSION_FIELDS).await.unwrap();
    let doc = Doc {
        id: point_id("claude", "t"),
        embedding: HashEmbedder::new(DIM).embed_one("clean text").unwrap(),
        payload: json!({"kind": "turn", "session_id": "s", "title": "set plainsecret1", "text": "clean text"}).as_object().cloned().unwrap(),
    };
    store.upsert(vec![doc]).await.unwrap();
    let r = rescan(&store, &mut HashEmbedder::new(DIM), &Redactor::new(["plainsecret1".to_owned()])).await.unwrap();
    assert_eq!(r.changed, 1);
    let hit = &store.scroll(&SearchFilter::default(), 10).await.unwrap()[0];
    assert_eq!(hit.payload["title"], "set [REDACTED:listed]");
    assert_eq!(hit.payload["text"], "clean text");
}
