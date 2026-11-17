//! `reembed_if_needed` against a real Qdrant: when a collection is re-embedded after a model or
//! chunker change, and that the version it was brought to is recorded per collection.

use std::sync::{Arc, Mutex};

use embed::{Embedder, Embedding, HashEmbedder};
use ingest::{SESSION_FIELDS, State, VERSION_FIELD};
use serde_json::{Value, json};
use store::{Doc, SearchFilter, Store, point_id};

const DIM: usize = 64;

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

/// A collection named `name` with two points: one stamped `chunker-3/m1`, one never stamped.
async fn collection(q: &testkit::Qdrant, name: &str) -> Store {
    let store = Store::connect(&q.url, None, name, DIM).unwrap();
    store.ensure_collection(SESSION_FIELDS).await.unwrap();
    let mut emb = HashEmbedder::new(DIM);
    let docs = [json!({"session_id": "a", "text": format!("{name} stamped"), VERSION_FIELD: "chunker-3/m1"}), json!({"session_id": "b", "text": format!("{name} unstamped")})]
        .into_iter()
        .map(|v| {
            let Value::Object(payload) = v else { unreachable!() };
            Doc { id: point_id("claude", payload["session_id"].as_str().unwrap()), embedding: emb.embed_one("old vector").unwrap(), payload }
        })
        .collect();
    store.upsert(docs).await.unwrap();
    store
}

/// How the state looks before the call.
#[derive(Debug, Clone, Copy)]
enum Before {
    /// `index_version/<collection>` holds this.
    Recorded(&'static str),
    /// Nothing recorded; the index version has changed at some point (`ensure_version` twice).
    NoneAfterAChange,
    /// Nothing recorded; the index version never changed (first `ensure_version` only).
    NoneNeverChanged,
}

#[tokio::test(flavor = "multi_thread")]
async fn re_embeds_exactly_when_the_recorded_version_differs_or_is_unknown_after_a_change() {
    let q = testkit::qdrant().await;
    let rows: [(&str, Before, &[&str]); 4] = [
        ("same", Before::Recorded("chunker-3/m2"), &[]),
        ("differs", Before::Recorded("chunker-3/m1"), &["differs stamped", "differs unstamped"]),
        ("unknown-changed", Before::NoneAfterAChange, &["unknown-changed stamped", "unknown-changed unstamped"]),
        ("unknown-never", Before::NoneNeverChanged, &[]),
    ];
    for (name, before, want) in rows {
        let store = collection(&q, name).await;
        let state = State::in_memory().unwrap();
        state.ensure_version("chunker-3/m1").unwrap();
        match before {
            Before::Recorded(v) => state.set_meta(&format!("index_version/{name}"), v).unwrap(),
            Before::NoneAfterAChange => {
                state.ensure_version("chunker-3/m2").unwrap();
            }
            Before::NoneNeverChanged => {}
        }
        let mut emb = Recording::default();

        ctx::reembed_if_needed(&store, &mut emb, &state, "chunker-3/m2").await.unwrap();

        let mut embedded = emb.texts.lock().unwrap().clone();
        embedded.sort();
        assert_eq!(embedded, want, "{name}: {before:?}");
        assert_eq!(state.get_meta(&format!("index_version/{name}")).unwrap().as_deref(), Some("chunker-3/m2"), "{name}: version recorded afterwards");
        let stale = store.count(&SearchFilter::default().ne(VERSION_FIELD, "chunker-3/m2")).await.unwrap();
        assert_eq!(stale, if want.is_empty() { 2 } else { 0 }, "{name}: stamped points");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn run_index_re_embeds_a_point_of_the_previous_model_and_records_the_new_version() {
    let q = testkit::qdrant().await;
    let store = Store::connect(&q.url, None, "sessions", DIM).unwrap();
    store.ensure_collection(SESSION_FIELDS).await.unwrap();
    let (old, new) = (ctx::index_version("m1"), ctx::index_version("m2"));
    let Value::Object(payload) = json!({"session_id": "kept-1", "kind": "turn", "text": "a turn from the m1 model", VERSION_FIELD: old}) else { unreachable!() };
    store.upsert(vec![Doc { id: point_id("claude", "kept-1/0"), embedding: HashEmbedder::new(DIM).embed_one("old vector").unwrap(), payload }]).await.unwrap();
    let state = State::in_memory().unwrap();
    state.ensure_version(&old).unwrap();
    state.set_meta("index_version/sessions", &old).unwrap();
    let mut emb = Recording::default();

    ctx::run_index(&store, &mut emb, "m2", &ingest::WordCounter, &state, &[], &ingest::Redactor::default(), |_| {}).await.unwrap();

    assert_eq!(*emb.texts.lock().unwrap(), ["a turn from the m1 model"]);
    let hits = store.scroll(&SearchFilter::default().eq("session_id", "kept-1"), 10).await.unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].payload[VERSION_FIELD], json!(new));
    assert_eq!(state.get_meta("index_version/sessions").unwrap(), Some(new));
}

#[tokio::test(flavor = "multi_thread")]
async fn the_version_is_kept_per_collection() {
    let q = testkit::qdrant().await;
    let sessions = collection(&q, "sessions").await;
    let teams = collection(&q, "teams").await;
    let state = State::in_memory().unwrap();
    state.ensure_version("chunker-3/m1").unwrap();
    state.set_meta("index_version/sessions", "chunker-3/m1").unwrap();
    state.set_meta("index_version/teams", "chunker-3/m2").unwrap();
    let mut emb = Recording::default();

    ctx::reembed_if_needed(&sessions, &mut emb, &state, "chunker-3/m2").await.unwrap();
    ctx::reembed_if_needed(&teams, &mut emb, &state, "chunker-3/m2").await.unwrap();

    let mut embedded = emb.texts.lock().unwrap().clone();
    embedded.sort();
    assert_eq!(embedded, ["sessions stamped", "sessions unstamped"], "teams was already current");
    assert_eq!(teams.count(&SearchFilter::default().ne(VERSION_FIELD, "chunker-3/m2")).await.unwrap(), 2, "teams untouched");

    // Recorded now, so a second call re-embeds nothing.
    let mut again = Recording::default();
    ctx::reembed_if_needed(&sessions, &mut again, &state, "chunker-3/m2").await.unwrap();
    assert_eq!(*again.texts.lock().unwrap(), Vec::<String>::new());
}
