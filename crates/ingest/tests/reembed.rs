//! `reembed_stale` against a real Qdrant: points not stamped with the current index version are
//! re-embedded from their stored text and stamped, everything else in their payload is kept.

use std::sync::{Arc, Mutex};

use embed::{Embedder, Embedding, HashEmbedder};
use ingest::{SESSION_FIELDS, VERSION_FIELD, reembed_stale};
use serde_json::{Map, Value, json};
use store::{Doc, SearchFilter, SearchMode, Store, point_id};

const DIM: usize = 64;
/// The vector every fixture point starts with: the embedding of a text none of them holds, so a
/// point still carrying it was not re-embedded.
const STALE_VECTOR_TEXT: &str = "a vector from an older model";

/// HashEmbedder that records every text it embeds.
#[derive(Clone, Default)]
struct Recording {
    texts: Arc<Mutex<Vec<String>>>,
}

impl Embedder for Recording {
    fn dim(&self) -> usize {
        DIM
    }
    fn embed(&mut self, texts: &[&str]) -> embed::Result<Vec<Embedding>> {
        self.texts.lock().unwrap().extend(texts.iter().map(|t| (*t).to_owned()));
        HashEmbedder::new(DIM).embed(texts)
    }
}

impl Recording {
    fn sorted(&self) -> Vec<String> {
        let mut t = self.texts.lock().unwrap().clone();
        t.sort();
        t
    }
}

fn payload(v: Value) -> Map<String, Value> {
    let Value::Object(m) = v else { unreachable!("object") };
    m
}

async fn store_with(qdrant: &testkit::Qdrant, points: &[Map<String, Value>]) -> Store {
    let store = Store::connect(&qdrant.url, None, "sessions", DIM).unwrap();
    store.ensure_collection(SESSION_FIELDS).await.unwrap();
    let stale = HashEmbedder::new(DIM).embed_one(STALE_VECTOR_TEXT).unwrap();
    let docs = points
        .iter()
        .map(|p| Doc { id: point_id("claude", p["session_id"].as_str().unwrap()), embedding: stale.clone(), payload: p.clone() })
        .collect();
    store.upsert(docs).await.unwrap();
    store
}

/// Dense similarity of the point of `session` to the embedding of `text`.
async fn score(store: &Store, session: &str, text: &str) -> f32 {
    let q = HashEmbedder::new(DIM).embed_one(text).unwrap();
    let hits = store.search(&q, &SearchFilter::default().eq("session_id", session), 1, SearchMode::Dense).await.unwrap();
    assert_eq!(hits.len(), 1, "{session}");
    hits[0].score
}

async fn payload_of(store: &Store, session: &str) -> Map<String, Value> {
    let hits = store.scroll(&SearchFilter::default().eq("session_id", session), 10).await.unwrap();
    assert_eq!(hits.len(), 1, "{session}");
    hits[0].payload.clone()
}

#[tokio::test(flavor = "multi_thread")]
async fn stale_and_unstamped_points_are_re_embedded_from_their_text_and_stamped() {
    let q = testkit::qdrant().await;
    let old = payload(json!({"session_id": "s-old", "project": "z:/work/alpha", "turn_index": 3, "text": "alpha deploy notes", VERSION_FIELD: "chunker-3/m1"}));
    let none = payload(json!({"session_id": "s-none", "project": "z:/work/beta", "turn_index": 5, "text": "beta migration log"}));
    let current = payload(json!({"session_id": "s-cur", "project": "z:/work/gamma", "turn_index": 7, "text": "gamma current text", VERSION_FIELD: "chunker-3/m2"}));
    let store = store_with(&q, &[old.clone(), none.clone(), current.clone()]).await;
    let mut emb = Recording::default();

    let n = reembed_stale(&store, &mut emb, "chunker-3/m2").await.unwrap();

    assert_eq!(n, 2);
    assert_eq!(emb.sorted(), ["alpha deploy notes", "beta migration log"], "embedded from the stored text, the current point left out");
    for (session, before) in [("s-old", old), ("s-none", none)] {
        let mut want = before;
        want.insert(VERSION_FIELD.into(), json!("chunker-3/m2"));
        assert_eq!(payload_of(&store, session).await, want, "{session}: payload kept, version stamped");
    }
    assert_eq!(payload_of(&store, "s-cur").await, current);
    assert!(score(&store, "s-old", "alpha deploy notes").await > 0.999, "the vector is now the text's");
    assert!(score(&store, "s-none", "beta migration log").await > 0.999);
    assert!(score(&store, "s-cur", STALE_VECTOR_TEXT).await > 0.999, "a current point keeps its vector");
    assert_eq!(store.count(&SearchFilter::default()).await.unwrap(), 3, "updated in place, no new points");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_second_pass_finds_nothing_to_do() {
    let q = testkit::qdrant().await;
    let store = store_with(&q, &[payload(json!({"session_id": "s1", "text": "one", VERSION_FIELD: "v1"}))]).await;
    assert_eq!(reembed_stale(&store, &mut Recording::default(), "v2").await.unwrap(), 1);
    let mut again = Recording::default();
    assert_eq!(reembed_stale(&store, &mut again, "v2").await.unwrap(), 0);
    assert_eq!(again.sorted(), Vec::<String>::new());
}

#[tokio::test(flavor = "multi_thread")]
async fn more_points_than_one_page_are_all_re_embedded_and_the_pass_ends() {
    let q = testkit::qdrant().await;
    // 70: more than one 64-point page, and not a multiple of the 4-text embedding batch.
    let points: Vec<_> = (0..70).map(|i| payload(json!({"session_id": format!("s{i:02}"), "text": format!("turn text {i:02}"), VERSION_FIELD: "v1"}))).collect();
    let store = store_with(&q, &points).await;
    let mut emb = Recording::default();

    assert_eq!(reembed_stale(&store, &mut emb, "v2").await.unwrap(), 70);

    let want: Vec<String> = (0..70).map(|i| format!("turn text {i:02}")).collect();
    assert_eq!(emb.sorted(), want, "each point embedded exactly once");
    assert_eq!(store.count(&SearchFilter::default().ne(VERSION_FIELD, "v2")).await.unwrap(), 0, "every point stamped");
    assert_eq!(store.count(&SearchFilter::default()).await.unwrap(), 70);
}

#[tokio::test(flavor = "multi_thread")]
async fn an_empty_collection_is_a_no_op() {
    let q = testkit::qdrant().await;
    let store = store_with(&q, &[]).await;
    let mut emb = Recording::default();
    assert_eq!(reembed_stale(&store, &mut emb, "v2").await.unwrap(), 0);
    assert_eq!(emb.sorted(), Vec::<String>::new());
}
