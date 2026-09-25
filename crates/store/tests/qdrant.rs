//! Integration tests against a real Qdrant started by testcontainers (Docker must be running).
//! Each test gets its own container so tests stay independent and can run in parallel.

use embed::{Embedder, Embedding, HashEmbedder, SparseVector};
use serde_json::{Map, Value, json};
use store::{Doc, Field, SearchFilter, SearchMode, Store, StoreError, point_id};

const DIM: usize = 64;
const FIELDS: &[Field] = &[Field::Keyword("source"), Field::Keyword("channel"), Field::Tenant("project"), Field::Bool("subagent")];

struct Env {
    store: Store,
    embedder: HashEmbedder,
    _qdrant: testkit::Qdrant,
}

async fn env() -> Env {
    let qdrant = testkit::qdrant().await;
    let store = Store::connect(&qdrant.url, None, "test", DIM).unwrap();
    store.ensure_collection(FIELDS).await.unwrap();
    Env { store, embedder: HashEmbedder::new(DIM), _qdrant: qdrant }
}

fn payload(v: Value) -> Map<String, Value> {
    v.as_object().cloned().expect("object")
}

impl Env {
    fn doc(&mut self, key: &str, text: &str, extra: Value) -> Doc {
        let mut p = payload(json!({ "text": text, "key": key }));
        p.extend(payload(extra));
        Doc { id: point_id("test", key), embedding: self.embedder.embed_one(text).unwrap(), payload: p }
    }

    async fn search(&mut self, q: &str, filter: SearchFilter, mode: SearchMode) -> Vec<String> {
        let q = self.embedder.embed_one(q).unwrap();
        self.store
            .search(&q, &filter, 10, mode)
            .await
            .unwrap()
            .into_iter()
            .map(|h| h.payload["key"].as_str().unwrap().to_owned())
            .collect()
    }
}

#[tokio::test]
async fn ensure_collection_is_idempotent() {
    let e = env().await;
    e.store.ensure_collection(FIELDS).await.unwrap();
    e.store.ensure_collection(&FIELDS[..1]).await.unwrap();
    assert_eq!(e.store.count(&SearchFilter::default()).await.unwrap(), 0);
}

#[tokio::test]
async fn hybrid_search_ranks_matching_document_first_and_returns_payload() {
    let mut e = env().await;
    let docs = vec![
        e.doc("db", "database migration failed with timeout on port 5439", json!({"source": "claude"})),
        e.doc("css", "restyled dashboard buttons with new palette", json!({"source": "claude"})),
        e.doc("lunch", "lunch at noon in the kitchen", json!({"source": "teams"})),
    ];
    e.store.upsert(docs).await.unwrap();

    let q = e.embedder.embed_one("why did the database migration fail").unwrap();
    let hits = e.store.search(&q, &SearchFilter::default(), 3, SearchMode::Hybrid).await.unwrap();
    assert_eq!(hits[0].id, point_id("test", "db"));
    assert_eq!(hits[0].payload["source"], "claude");
    assert_eq!(hits[0].payload["text"], "database migration failed with timeout on port 5439");
    assert!(hits.windows(2).all(|w| w[0].score >= w[1].score), "scores must be descending");
}

#[tokio::test]
async fn upserting_same_id_overwrites_instead_of_duplicating() {
    let mut e = env().await;
    let first = e.doc("k", "first version", json!({"rev": 1}));
    e.store.upsert(vec![first]).await.unwrap();
    let second = e.doc("k", "second version", json!({"rev": 2}));
    e.store.upsert(vec![second]).await.unwrap();

    assert_eq!(e.store.count(&SearchFilter::default()).await.unwrap(), 1);
    let q = e.embedder.embed_one("second version").unwrap();
    let hits = e.store.search(&q, &SearchFilter::default(), 5, SearchMode::Hybrid).await.unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].payload["rev"], 2);
}

#[tokio::test]
async fn keyword_filter_restricts_results_and_count() {
    let mut e = env().await;
    let docs = vec![
        e.doc("a", "deploy failed", json!({"source": "claude", "channel": "ops"})),
        e.doc("b", "deploy failed again", json!({"source": "teams", "channel": "ops"})),
        e.doc("c", "deploy failed in staging", json!({"source": "teams", "channel": "dev"})),
    ];
    e.store.upsert(docs).await.unwrap();

    let teams = SearchFilter::default().eq("source", "teams");
    let mut keys = e.search("deploy failed", teams.clone(), SearchMode::Hybrid).await;
    keys.sort();
    assert_eq!(keys, ["b", "c"]);
    assert_eq!(e.store.count(&teams).await.unwrap(), 2);

    let teams_ops = teams.eq("channel", "ops");
    assert_eq!(e.search("deploy failed", teams_ops, SearchMode::Hybrid).await, ["b"]);
}

#[tokio::test]
async fn timestamp_range_is_inclusive_on_both_ends() {
    let mut e = env().await;
    let docs = vec![
        e.doc("t100", "status report", json!({"ts": 100})),
        e.doc("t200", "status report", json!({"ts": 200})),
        e.doc("t300", "status report", json!({"ts": 300})),
    ];
    e.store.upsert(docs).await.unwrap();

    let mut keys = e
        .search("status report", SearchFilter::default().between(Some(100), Some(200)), SearchMode::Hybrid)
        .await;
    keys.sort();
    assert_eq!(keys, ["t100", "t200"]);
    let since = SearchFilter::default().between(Some(250), None);
    assert_eq!(e.search("status report", since, SearchMode::Dense).await, ["t300"]);
}

#[tokio::test]
async fn hybrid_fuses_documents_found_by_only_one_leg() {
    let e = env().await;
    // Hand-built vectors: `dense_only` matches the query's dense vector but shares no sparse
    // index; `sparse_only` is the reverse. Only fusion can surface both.
    let axis = |i: usize| {
        let mut v = vec![0.0; DIM];
        v[i] = 1.0;
        v
    };
    let sparse = |i: u32| SparseVector::new(vec![i], vec![1.0]).unwrap();
    let doc = |key: &str, dense: Vec<f32>, sp: SparseVector| Doc {
        id: point_id("test", key),
        embedding: Embedding { dense, sparse: sp },
        payload: payload(json!({"key": key})),
    };
    e.store
        .upsert(vec![
            doc("dense_only", axis(0), sparse(900)),
            doc("sparse_only", axis(1), sparse(7)),
        ])
        .await
        .unwrap();
    let query = Embedding { dense: axis(0), sparse: sparse(7) };
    let keys = |hits: Vec<store::Hit>| -> Vec<String> {
        hits.into_iter().map(|h| h.payload["key"].as_str().unwrap().to_owned()).collect()
    };
    let none = SearchFilter::default();

    let mut hybrid = keys(e.store.search(&query, &none, 10, SearchMode::Hybrid).await.unwrap());
    hybrid.sort();
    assert_eq!(hybrid, ["dense_only", "sparse_only"]);
    assert_eq!(keys(e.store.search(&query, &none, 10, SearchMode::Sparse).await.unwrap()), ["sparse_only"]);
    assert_eq!(keys(e.store.search(&query, &none, 10, SearchMode::Dense).await.unwrap())[0], "dense_only");
}

#[tokio::test]
async fn delete_removes_points_and_empty_calls_are_no_ops() {
    let mut e = env().await;
    let docs = vec![e.doc("x", "alpha", json!({})), e.doc("y", "beta", json!({}))];
    e.store.upsert(docs).await.unwrap();
    e.store.upsert(vec![]).await.unwrap();
    e.store.delete(&[]).await.unwrap();
    assert_eq!(e.store.count(&SearchFilter::default()).await.unwrap(), 2);

    e.store.delete(&[point_id("test", "x")]).await.unwrap();
    assert_eq!(e.store.count(&SearchFilter::default()).await.unwrap(), 1);
    assert_eq!(e.search("alpha beta", SearchFilter::default(), SearchMode::Hybrid).await, ["y"]);
}

#[tokio::test]
async fn wrong_dimension_is_rejected_before_reaching_qdrant() {
    let e = env().await;
    let bad = Embedding { dense: vec![1.0; DIM + 1], sparse: SparseVector::default() };
    let doc = Doc { id: point_id("test", "bad"), embedding: bad.clone(), payload: Map::new() };
    assert!(matches!(e.store.upsert(vec![doc]).await, Err(StoreError::Dimension { expected: DIM, got: 65 })));
    assert!(matches!(
        e.store.search(&bad, &SearchFilter::default(), 5, SearchMode::Hybrid).await,
        Err(StoreError::Dimension { .. })
    ));
}

#[tokio::test]
async fn collection_can_be_dropped_and_rebuilt() {
    let mut e = env().await;
    let d = e.doc("x", "alpha", json!({}));
    e.store.upsert(vec![d]).await.unwrap();
    e.store.drop_collection().await.unwrap();
    e.store.drop_collection().await.unwrap();
    e.store.ensure_collection(&[]).await.unwrap();
    assert_eq!(e.store.count(&SearchFilter::default()).await.unwrap(), 0);
}

#[tokio::test]
async fn polish_diacritics_survive_payload_round_trip() {
    let mut e = env().await;
    let text = "Zażółć gęślą jaźń — migracja bazy nie powiodła się";
    let d = e.doc("pl", text, json!({"author": "Łukasz Żółkiewski"}));
    e.store.upsert(vec![d]).await.unwrap();
    let q = e.embedder.embed_one("migracja bazy").unwrap();
    let hits = e.store.search(&q, &SearchFilter::default(), 1, SearchMode::Hybrid).await.unwrap();
    assert_eq!(hits[0].payload["text"], text);
    assert_eq!(hits[0].payload["author"], "Łukasz Żółkiewski");
}

#[tokio::test]
async fn delete_matching_except_prunes_only_stale_points_of_that_session() {
    let mut e = env().await;
    let docs = vec![
        e.doc("s1-a", "keep me", json!({"source": "claude", "project": "p", "session_id": "s1"})),
        e.doc("s1-b", "stale", json!({"source": "claude", "project": "p", "session_id": "s1"})),
        e.doc("s2-a", "other session", json!({"source": "claude", "project": "p", "session_id": "s2"})),
    ];
    e.store.upsert(docs).await.unwrap();
    let s1 = SearchFilter::default().eq("session_id", "s1");
    e.store.delete_matching_except(&s1, &[point_id("test", "s1-a")]).await.unwrap();

    assert_eq!(e.store.count(&s1).await.unwrap(), 1);
    assert_eq!(e.store.count(&SearchFilter::default().eq("session_id", "s2")).await.unwrap(), 1);
}

#[tokio::test]
async fn delete_matching_except_refuses_an_empty_filter() {
    let mut e = env().await;
    let d = e.doc("x", "alpha", json!({}));
    e.store.upsert(vec![d]).await.unwrap();
    assert!(e.store.delete_matching_except(&SearchFilter::default(), &[]).await.is_err());
    assert_eq!(e.store.count(&SearchFilter::default()).await.unwrap(), 1);
}

#[tokio::test]
async fn scroll_returns_all_matching_payloads() {
    let mut e = env().await;
    let docs = (0..5)
        .map(|i| {
            let session = if i < 3 { "s1" } else { "s2" };
            e.doc(&format!("k{i}"), "same text", json!({"session_id": session, "turn_index": i}))
        })
        .collect();
    e.store.upsert(docs).await.unwrap();
    let mut turns: Vec<i64> = e
        .store
        .scroll(&SearchFilter::default().eq("session_id", "s1"), 100)
        .await
        .unwrap()
        .iter()
        .map(|h| h.payload["turn_index"].as_i64().unwrap())
        .collect();
    turns.sort();
    assert_eq!(turns, [0, 1, 2]);
}

#[tokio::test]
async fn tenant_field_filters_like_a_keyword() {
    let mut e = env().await;
    let docs = vec![
        e.doc("a", "deploy notes", json!({"project": "d:/dev/a"})),
        e.doc("b", "deploy notes", json!({"project": "d:/dev/b"})),
    ];
    e.store.upsert(docs).await.unwrap();
    let only_a = SearchFilter::default().eq("project", "d:/dev/a");
    assert_eq!(e.search("deploy notes", only_a, SearchMode::Hybrid).await, ["a"]);
}

#[tokio::test]
async fn bool_flag_filter_excludes_subagent_points() {
    let mut e = env().await;
    let docs = vec![
        e.doc("main", "found the bug", json!({"subagent": false})),
        e.doc("sub", "found the bug", json!({"subagent": true})),
    ];
    e.store.upsert(docs).await.unwrap();
    let f = SearchFilter::default().flag("subagent", false);
    assert_eq!(e.search("found the bug", f, SearchMode::Hybrid).await, ["main"]);
}

#[tokio::test]
async fn integer_equality_filter_selects_first_chunks() {
    let mut e = env().await;
    let docs = (0..3).map(|i| e.doc(&format!("c{i}"), "chunk", json!({"chunk_index": i}))).collect();
    e.store.upsert(docs).await.unwrap();
    let firsts = e.store.scroll(&SearchFilter::default().int("chunk_index", 0), 10).await.unwrap();
    assert_eq!(firsts.len(), 1);
    assert_eq!(firsts[0].payload["key"], "c0");
}
