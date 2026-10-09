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
async fn delete_matching_prunes_by_filter_and_refuses_an_empty_one() {
    let mut e = env().await;
    let docs = vec![
        e.doc("a1", "alpha one", json!({"project": "A", "session_id": "a1", "ts": 100})),
        e.doc("a2", "alpha two", json!({"project": "A", "session_id": "a1", "ts": 100})),
        e.doc("b1", "beta one", json!({"project": "B", "session_id": "b1", "ts": 5000})),
    ];
    e.store.upsert(docs).await.unwrap();

    // Empty filter is refused (would wipe the collection).
    assert!(e.store.delete_matching(&SearchFilter::default()).await.is_err());
    assert_eq!(e.store.count(&SearchFilter::default()).await.unwrap(), 3);

    // distinct_payload gathers the ids about to go.
    let ids = e.store.distinct_payload(&SearchFilter::default().eq("project", "A"), "session_id", 100).await.unwrap();
    assert_eq!(ids, ["a1"], "two chunks, one distinct session");

    // Prune project A: only A's points go.
    e.store.delete_matching(&SearchFilter::default().eq("project", "A")).await.unwrap();
    assert_eq!(e.store.count(&SearchFilter::default()).await.unwrap(), 1);
    assert_eq!(e.store.count(&SearchFilter::default().eq("project", "B")).await.unwrap(), 1);

    // Prune older-than: ts <= cutoff removes the old one, keeps the newer (b1 ts=5000).
    let old = e.doc("a3", "gamma", json!({"project": "A", "session_id": "a3", "ts": 50}));
    e.store.upsert(vec![old]).await.unwrap();
    assert_eq!(e.store.count(&SearchFilter::default().between(None, Some(60))).await.unwrap(), 1, "only a3 is old");
    e.store.delete_matching(&SearchFilter::default().between(None, Some(60))).await.unwrap();
    assert_eq!(e.store.count(&SearchFilter::default()).await.unwrap(), 1, "old removed");
    assert_eq!(e.store.count(&SearchFilter::default().eq("project", "B")).await.unwrap(), 1, "newer B kept");
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

#[tokio::test]
async fn scroll_pages_through_more_points_than_one_page() {
    let mut e = env().await;
    // Payloads large enough that one unpaged reply of all of them would be several MiB.
    let big = "x".repeat(3000);
    let docs: Vec<_> = (0..700).map(|i| e.doc(&format!("p{i}"), &format!("doc {i} {big}"), json!({"n": i}))).collect();
    for chunk in docs.chunks(100) {
        e.store.upsert(chunk.to_vec()).await.unwrap();
    }
    let all = e.store.scroll(&SearchFilter::default(), 10_000).await.unwrap();
    assert_eq!(all.len(), 700);
    let mut ids: Vec<_> = all.iter().map(|h| h.id).collect();
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), 700, "no duplicates across pages");
    assert_eq!(e.store.scroll(&SearchFilter::default(), 300).await.unwrap().len(), 300, "limit still applies");

    let mut pages = 0;
    e.store
        .scroll_pages(&SearchFilter::default(), 10_000, |p| {
            assert!(p.len() <= 256);
            pages += 1;
            std::future::ready(Ok(()))
        })
        .await
        .unwrap();
    assert!(pages >= 3, "{pages}");
}

#[tokio::test]
async fn any_of_matches_listed_values_and_an_empty_list_matches_nothing() {
    let mut e = env().await;
    let docs = vec![
        e.doc("a", "status update", json!({"channel": "c1"})),
        e.doc("b", "status update", json!({"channel": "c2"})),
        e.doc("c", "status update", json!({"channel": "c3"})),
    ];
    e.store.upsert(docs).await.unwrap();
    let two = SearchFilter::default().any_of("channel", vec!["c1".into(), "c3".into()]);
    let mut keys = e.search("status update", two.clone(), SearchMode::Hybrid).await;
    keys.sort();
    assert_eq!(keys, ["a", "c"]);
    assert_eq!(e.store.count(&two).await.unwrap(), 2);
    let none = SearchFilter::default().any_of("channel", vec![]);
    assert!(e.search("status update", none.clone(), SearchMode::Hybrid).await.is_empty());
    assert_eq!(e.store.count(&none).await.unwrap(), 0);
    assert!(e.store.scroll(&none, 10).await.unwrap().is_empty());
}

/// Each probe's own point as found by `mode`: its score and payload. The test embedder maps
/// some texts to identical vectors, so the probe's own point is looked up among the ties.
async fn own_hits(s: &Store, probes: &[(String, Embedding)], mode: SearchMode) -> Vec<(f32, Map<String, Value>)> {
    let mut out = Vec::new();
    for (key, p) in probes {
        let hits = s.search(p, &SearchFilter::default(), 20, mode).await.unwrap();
        let h = hits.into_iter().find(|h| h.payload["key"] == key.as_str()).expect("own point among the top hits");
        out.push((h.score, h.payload));
    }
    out
}

fn keys(hits: Vec<store::Hit>) -> Vec<String> {
    hits.into_iter().map(|h| h.payload["key"].as_str().unwrap().to_owned()).collect()
}

#[tokio::test]
async fn bm25_finds_the_exact_token_and_respects_filters() {
    let qdrant = testkit::qdrant().await;
    let store = Store::connect(&qdrant.url, None, "lex", DIM).unwrap().with_lexical();
    store.ensure_collection(FIELDS).await.unwrap();
    let mut emb = HashEmbedder::new(DIM);
    let mut doc = |key: &str, text: &str, project: &str| Doc {
        id: point_id("t", key),
        embedding: emb.embed_one(text).unwrap(),
        payload: payload(json!({"text": text, "key": key, "project": project})),
    };
    store
        .upsert(vec![
            doc("a", "we fixed the build and the build passed", "p"),
            doc("b", "the build broke in SEDN26-1910 token_stream", "p"),
            doc("c", "SEDN26-1910 in another project", "q"),
        ])
        .await
        .unwrap();
    let in_p = SearchFilter::default().eq("project", "p");
    assert_eq!(keys(store.search_bm25("co z SEDN26-1910?", &in_p, 10).await.unwrap()), ["b"]);
    assert_eq!(keys(store.search_bm25("token_stream", &SearchFilter::default(), 10).await.unwrap()), ["b"]);
    assert_eq!(keys(store.search_bm25("sedn26 1910", &SearchFilter::default(), 10).await.unwrap()).len(), 2);
    assert!(store.search_bm25("?!", &in_p, 10).await.unwrap().is_empty(), "no tokens, no hits");
    // Dense and sparse legs are untouched by the extra vector.
    let q = HashEmbedder::new(DIM).embed_one("build passed").unwrap();
    assert_eq!(keys(store.search(&q, &in_p, 1, SearchMode::Dense).await.unwrap()), ["a"]);
}

#[tokio::test]
async fn bm25_search_on_a_collection_without_it_is_an_error_not_a_panic() {
    let mut e = env().await;
    let d = e.doc("a", "token_stream", json!({}));
    e.store.upsert(vec![d]).await.unwrap();
    assert!(e.store.search_bm25("token_stream", &SearchFilter::default(), 10).await.is_err());
}

/// An old collection (no BM25) is migrated in place: same points, same stored vectors and
/// payloads, plus BM25; the copy aside is gone afterwards; a second call is a no-op.
#[tokio::test]
async fn migration_adds_bm25_and_keeps_every_point_and_vector() {
    let qdrant = testkit::qdrant().await;
    let old = Store::connect(&qdrant.url, None, "sessions", DIM).unwrap();
    old.ensure_collection(FIELDS).await.unwrap();
    let mut emb = HashEmbedder::new(DIM);
    // More than one scroll page (256), so paging is exercised.
    let docs: Vec<Doc> = (0..300)
        .map(|i| {
            let text = format!("turn {i} mentions ident_{i}");
            Doc { id: point_id("t", &i.to_string()), embedding: emb.embed_one(&text).unwrap(), payload: payload(json!({"text": text, "key": i.to_string(), "project": "p"})) }
        })
        .collect();
    old.upsert(docs).await.unwrap();
    // Each probe is a stored point's own embedding; the same score for its own point before and
    // after, per leg, shows the vectors were copied unchanged.
    let probes: Vec<_> = [0, 7, 150, 299].iter().map(|i| (i.to_string(), emb.embed_one(&format!("turn {i} mentions ident_{i}")).unwrap())).collect();
    let before = (own_hits(&old, &probes, SearchMode::Dense).await, own_hits(&old, &probes, SearchMode::Sparse).await);

    let new = Store::connect(&qdrant.url, None, "sessions", DIM).unwrap().with_lexical();
    new.ensure_collection(FIELDS).await.unwrap();
    assert_eq!(new.count(&SearchFilter::default()).await.unwrap(), 300);
    let after = (own_hits(&new, &probes, SearchMode::Dense).await, own_hits(&new, &probes, SearchMode::Sparse).await);
    assert_eq!(after, before, "stored dense and sparse vectors and payloads are copied unchanged");
    assert_eq!(keys(new.search_bm25("ident_7", &SearchFilter::default(), 5).await.unwrap())[0], "7");
    let aside = Store::connect(&qdrant.url, None, "sessions__migrate", DIM).unwrap();
    assert!(aside.search_bm25("x", &SearchFilter::default(), 1).await.is_err(), "the copy aside was dropped");

    new.ensure_collection(FIELDS).await.unwrap();
    assert_eq!(new.count(&SearchFilter::default()).await.unwrap(), 300, "idempotent");
    new.ensure_collection(FIELDS).await.unwrap();
    let filtered = SearchFilter::default().eq("project", "p");
    assert_eq!(new.count(&filtered).await.unwrap(), 300, "payload indexes still work");
}

/// A crash after the original was dropped leaves only the copy aside; the next start finishes.
#[tokio::test]
async fn migration_resumes_from_the_copy_aside() {
    let qdrant = testkit::qdrant().await;
    let aside = Store::connect(&qdrant.url, None, "sessions__migrate", DIM).unwrap();
    aside.ensure_collection(&[]).await.unwrap();
    let mut emb = HashEmbedder::new(DIM);
    let text = "the only turn about RDE_2772";
    aside
        .upsert(vec![Doc { id: point_id("t", "only"), embedding: emb.embed_one(text).unwrap(), payload: payload(json!({"text": text, "key": "only"})) }])
        .await
        .unwrap();

    let store = Store::connect(&qdrant.url, None, "sessions", DIM).unwrap().with_lexical();
    store.ensure_collection(FIELDS).await.unwrap();
    assert_eq!(keys(store.search_bm25("RDE_2772", &SearchFilter::default(), 5).await.unwrap()), ["only"]);
    assert!(aside.count(&SearchFilter::default()).await.is_err(), "copy aside dropped once copied back");
}

/// The original was dropped mid-migration and an older build recreated it (no BM25) with one
/// re-indexed point. The larger copy aside is the truth; the newer point is folded in.
#[tokio::test]
async fn migration_prefers_the_complete_copy_over_an_original_recreated_by_an_older_build() {
    let qdrant = testkit::qdrant().await;
    let mut emb = HashEmbedder::new(DIM);
    let mut doc = |key: &str, text: &str| Doc { id: point_id("t", key), embedding: emb.embed_one(text).unwrap(), payload: payload(json!({"text": text, "key": key})) };
    let aside = Store::connect(&qdrant.url, None, "sessions__migrate", DIM).unwrap();
    aside.ensure_collection(&[]).await.unwrap();
    aside.upsert(vec![doc("a", "old text of a"), doc("b", "only b mentions RDE_2772"), doc("c", "c")]).await.unwrap();
    let older_build = Store::connect(&qdrant.url, None, "sessions", DIM).unwrap();
    older_build.ensure_collection(&[]).await.unwrap();
    older_build.upsert(vec![doc("a", "new text of a mentions token_stream")]).await.unwrap();

    let store = Store::connect(&qdrant.url, None, "sessions", DIM).unwrap().with_lexical();
    store.ensure_collection(FIELDS).await.unwrap();
    assert_eq!(store.count(&SearchFilter::default()).await.unwrap(), 3, "nothing from the copy aside is lost");
    assert_eq!(keys(store.search_bm25("RDE_2772", &SearchFilter::default(), 5).await.unwrap()), ["b"]);
    assert_eq!(keys(store.search_bm25("token_stream", &SearchFilter::default(), 5).await.unwrap()), ["a"], "the newer write wins");
    assert!(aside.count(&SearchFilter::default()).await.is_err(), "copy aside dropped");
}


fn plain_doc(key: &str, text: Option<&str>, ts: i64) -> Doc {
    let mut p = payload(json!({"key": key, "ts": ts}));
    if let Some(t) = text {
        p.insert("text".into(), json!(t));
    }
    Doc { id: point_id("t", key), embedding: HashEmbedder::new(DIM).embed_one(text.unwrap_or(key)).unwrap(), payload: p }
}

/// A crash after the full copy aside but before the original was dropped: equal counts, the
/// original still without BM25. The next start restarts cleanly from the original.
#[tokio::test]
async fn migration_restarts_when_the_copy_aside_is_complete_but_the_original_untouched() {
    let qdrant = testkit::qdrant().await;
    for name in ["sessions", "sessions__migrate"] {
        let s = Store::connect(&qdrant.url, None, name, DIM).unwrap();
        s.ensure_collection(&[]).await.unwrap();
        s.upsert(vec![plain_doc("a", Some("alpha RDE_2772"), 1), plain_doc("b", Some("beta"), 2)]).await.unwrap();
    }
    let store = Store::connect(&qdrant.url, None, "sessions", DIM).unwrap().with_lexical();
    store.ensure_collection(FIELDS).await.unwrap();
    assert_eq!(store.count(&SearchFilter::default()).await.unwrap(), 2);
    assert_eq!(keys(store.search_bm25("RDE_2772", &SearchFilter::default(), 5).await.unwrap()), ["a"]);
}

/// A crash part-way through copying back: the original already has BM25 and one point, since
/// rewritten (redacted). Resuming adds what is missing and keeps the rewrite.
#[tokio::test]
async fn resumed_copy_back_never_overwrites_newer_points() {
    let qdrant = testkit::qdrant().await;
    let store = Store::connect(&qdrant.url, None, "sessions", DIM).unwrap().with_lexical();
    store.ensure_collection(&[]).await.unwrap();
    store.upsert(vec![plain_doc("a", Some("a was [REDACTED]"), 1)]).await.unwrap();
    let aside = Store::connect(&qdrant.url, None, "sessions__migrate", DIM).unwrap();
    aside.ensure_collection(&[]).await.unwrap();
    aside.upsert(vec![plain_doc("a", Some("a was hunter2"), 1), plain_doc("b", Some("b mentions RDE_2772"), 2)]).await.unwrap();

    store.ensure_collection(FIELDS).await.unwrap();
    assert_eq!(store.count(&SearchFilter::default()).await.unwrap(), 2);
    assert!(store.search_bm25("hunter2", &SearchFilter::default(), 5).await.unwrap().is_empty(), "the redaction stays");
    assert_eq!(keys(store.search_bm25("RDE_2772", &SearchFilter::default(), 5).await.unwrap()), ["b"]);
    assert!(aside.count(&SearchFilter::default()).await.is_err(), "copy aside dropped");
}

/// Points without text survive the migration (they just get no BM25), and integer payloads
/// keep working with range filters afterwards.
#[tokio::test]
async fn migration_keeps_textless_points_and_integer_payloads() {
    let qdrant = testkit::qdrant().await;
    let old = Store::connect(&qdrant.url, None, "sessions", DIM).unwrap();
    old.ensure_collection(FIELDS).await.unwrap();
    old.upsert(vec![plain_doc("no-text", None, 10), plain_doc("t", Some("RDE_2772 here"), 20)]).await.unwrap();

    let store = Store::connect(&qdrant.url, None, "sessions", DIM).unwrap().with_lexical();
    store.ensure_collection(FIELDS).await.unwrap();
    assert_eq!(store.count(&SearchFilter::default()).await.unwrap(), 2);
    assert_eq!(store.count(&SearchFilter::default().between(Some(5), Some(15))).await.unwrap(), 1, "ts is still an integer");
    assert_eq!(keys(store.search_bm25("RDE_2772", &SearchFilter::default().between(Some(15), None), 5).await.unwrap()), ["t"]);
}

/// While one process holds the migration lock, another fails instead of interleaving drops.
#[tokio::test]
async fn a_second_migrator_fails_while_the_lock_is_held() {
    let qdrant = testkit::qdrant().await;
    let old = Store::connect(&qdrant.url, None, "sessions", DIM).unwrap();
    old.ensure_collection(FIELDS).await.unwrap();
    old.upsert(vec![plain_doc("a", Some("RDE_2772"), 1)]).await.unwrap();
    let dir = std::env::temp_dir().join(format!("ctx-lock-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let lock = dir.join("migrate.lock");
    let held = std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(&lock).unwrap();
    held.lock().unwrap();

    let store = Store::connect(&qdrant.url, None, "sessions", DIM).unwrap().with_lexical().with_migration_lock(&lock);
    let err = store.ensure_collection(FIELDS).await.unwrap_err();
    assert!(matches!(err, StoreError::Migration(_)), "{err}");
    assert_eq!(old.count(&SearchFilter::default()).await.unwrap(), 1, "untouched");

    drop(held);
    store.ensure_collection(FIELDS).await.unwrap();
    assert_eq!(keys(store.search_bm25("RDE_2772", &SearchFilter::default(), 5).await.unwrap()), ["a"]);
    let _ = std::fs::remove_dir_all(dir);
}
