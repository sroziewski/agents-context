//! Routed lexical search (DEV-61..63) against a real Qdrant: identifier queries get dense fused
//! with BM25 and no summaries, every leg keeps the scope filters, deep search reranks the routed
//! pool, and collections written before BM25 keep answering and migrate on the indexing path.

mod common;

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use common::{ALPHA, BETA, DIM, Fixture, indexed};
use ctx::config::Config;
use ctx::lexical::Lexical;
use ctx::mcp::ContextServer;
use ctx::search::{Context, Query, Results, Scope};
use embed::{Embedder, HashEmbedder};
use ingest::{State, WordCounter};
use rmcp::ServiceExt;
use rmcp::model::CallToolRequestParams;
use serde_json::{Value, json};
use store::{SearchFilter, SearchMode, Store};

fn context(store: Arc<Store>) -> Context {
    Context { store, embedder: Arc::new(Mutex::new(Box::new(HashEmbedder::new(DIM)) as Box<dyn Embedder>)) }
}

fn sessions(r: &Results) -> Vec<String> {
    r.turns.iter().map(|h| h.payload["session_id"].as_str().unwrap().to_owned()).collect()
}

fn alpha() -> Scope {
    Scope::Project(sources::project::key(ALPHA))
}

fn query(text: &str, scope: Scope, lexical: Lexical) -> Query {
    let mut q = Query::new(text, scope, 5);
    q.mode = SearchMode::Dense;
    q.lexical = lexical;
    q
}

/// Every turn starts with the same five words, so BM25 gives them almost no weight while the
/// hash embedder still counts them: `plain` (nothing else) is the dense favourite, `ident` alone
/// names the identifier but is diluted by unrelated words, and the plans share the query's
/// moderately rare `rollout`, so BM25 puts `plain` below them.
const RANKING: &[Fixture] = &[
    Fixture { id: "plain", cwd: ALPHA, at: "2026-09-20T10:00:00Z", prompts: &["please check what we did", "thanks"] },
    Fixture { id: "ident", cwd: ALPHA, at: "2026-09-20T10:00:00Z", prompts: &["please check what we did with token_stream_v2 yesterday evening near harbour lights while seagulls circled over fishing boats at low tide", "thanks"] },
    Fixture { id: "plan-a", cwd: ALPHA, at: "2026-09-20T10:00:00Z", prompts: &["please check what we did on the rollout plan for marketing budget spreadsheets quarterly review meeting with finance team leads plus regional directors next thursday afternoon in room seven", "thanks"] },
    Fixture { id: "plan-b", cwd: ALPHA, at: "2026-09-20T10:00:00Z", prompts: &["please check what we did on the rollout plan for sales budget spreadsheets yearly review meeting with finance team leads plus regional directors next thursday afternoon in room seven", "thanks"] },
    Fixture { id: "lunch-a", cwd: ALPHA, at: "2026-09-20T10:00:00Z", prompts: &["please check what we did for lunch menu options", "thanks"] },
    Fixture { id: "lunch-b", cwd: ALPHA, at: "2026-09-20T10:00:00Z", prompts: &["please check what we did for dinner menu options", "thanks"] },
    Fixture { id: "beta", cwd: BETA, at: "2026-09-20T10:00:00Z", prompts: &["please check what we did with token_stream_v2", "thanks"] },
];
const ID_QUERY: &str = "please check what we did about token_stream_v2 and the rollout";
const NATURAL_QUERY: &str = "please check what we did on the rollout plan";

#[tokio::test(flavor = "multi_thread")]
async fn bm25_puts_the_turn_naming_the_identifier_first_where_dense_does_not() {
    let e = indexed(RANKING).await;
    let ctx = context(e.store.clone());

    let dense = ctx.search(&query(ID_QUERY, alpha(), Lexical::Off)).await.unwrap();
    assert_eq!(sessions(&dense)[..2], ["plain", "ident"], "dense alone prefers the shorter turn");

    let routed = ctx.search(&query(ID_QUERY, alpha(), Lexical::Bm25)).await.unwrap();
    assert_eq!(sessions(&routed)[0], "ident", "{:?}", sessions(&routed));
    assert_eq!(routed.turns.len(), 5, "cut to the limit");
    assert!(routed.turns.windows(2).all(|w| w[0].score >= w[1].score), "scored by the fused rank");
    assert!(!sessions(&routed).contains(&"beta".to_owned()), "BM25's best match is in another project");

    let all = ctx.search(&query(ID_QUERY, Scope::All, Lexical::Bm25)).await.unwrap();
    assert!(sessions(&all)[..2].contains(&"beta".to_owned()), "found once the scope allows it: {:?}", sessions(&all));
}

#[tokio::test(flavor = "multi_thread")]
async fn sparse_and_bm25_legs_together_also_route_the_identifier_turn_first() {
    let e = indexed(RANKING).await;
    let r = context(e.store.clone()).search(&query(ID_QUERY, alpha(), Lexical::SparseBm25)).await.unwrap();
    assert_eq!(r.turns.len(), 5);
    assert_eq!(sessions(&r)[0], "ident", "{:?}", sessions(&r));
    assert!(r.summaries.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn identifier_queries_return_turns_without_summaries_and_natural_ones_keep_them() {
    let e = indexed(RANKING).await;
    let ctx = context(e.store.clone());
    assert!(e.store.count(&SearchFilter::default().eq("kind", "summary")).await.unwrap() > 0, "the fixtures are summarised");
    for lexical in [Lexical::Off, Lexical::Bm25, Lexical::SparseBm25] {
        let id = ctx.search(&query(ID_QUERY, alpha(), lexical)).await.unwrap();
        if lexical == Lexical::Off {
            assert!(!id.summaries.is_empty(), "off restores the old layout, summaries first");
        } else {
            assert!(id.summaries.is_empty(), "{lexical:?}: {:?}", id.summaries);
        }
        assert!(!id.turns.is_empty(), "{lexical:?}");

        let natural = ctx.search(&query(NATURAL_QUERY, alpha(), lexical)).await.unwrap();
        assert!(!natural.summaries.is_empty(), "{lexical:?}: natural questions still get the gist");
        assert!(natural.summaries.iter().all(|h| h.payload["kind"] == "summary" && h.payload["project"] == "z:/work/alpha"));
        assert!(!natural.turns.is_empty(), "{lexical:?}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn natural_queries_ignore_the_lexical_setting() {
    let e = indexed(RANKING).await;
    let ctx = context(e.store.clone());
    let off = ctx.search(&query(NATURAL_QUERY, alpha(), Lexical::Off)).await.unwrap();
    let bm25 = ctx.search(&query(NATURAL_QUERY, alpha(), Lexical::Bm25)).await.unwrap();
    assert_eq!(sessions(&off), sessions(&bm25));
    let scores = |r: &Results| r.turns.iter().map(|h| h.score).collect::<Vec<_>>();
    assert_eq!(scores(&off), scores(&bm25), "plain dense scores, not fused ranks");
}

#[tokio::test(flavor = "multi_thread")]
async fn deep_search_reranks_the_routed_pool() {
    let e = indexed(RANKING).await;
    let ctx = context(e.store.clone());
    let rr = || -> Arc<Mutex<Box<dyn embed::Reranker>>> { Arc::new(Mutex::new(Box::new(embed::HashReranker))) };

    // With a one-turn pool the reranker can only return what the first stage put there.
    let mut q = query(ID_QUERY, alpha(), Lexical::Off);
    q.limit = 1;
    assert_eq!(sessions(&ctx.search_reranked(&q, rr(), 1).await.unwrap()), ["plain"]);
    q.lexical = Lexical::Bm25;
    let deep = ctx.search_reranked(&q, rr(), 1).await.unwrap();
    assert_eq!(sessions(&deep), ["ident"], "the pool came from the routed search");
    assert!(deep.summaries.is_empty());

    let mut q = query(ID_QUERY, alpha(), Lexical::Bm25);
    q.limit = 3;
    let deep = ctx.search_reranked(&q, rr(), 20).await.unwrap();
    assert_eq!(deep.turns.len(), 3);
    assert_eq!(sessions(&deep)[0], "ident", "most words in common with the query: {:?}", sessions(&deep));
    assert!(deep.summaries.is_empty());
    assert!(!sessions(&deep).contains(&"beta".to_owned()));
}

async fn call(client: &rmcp::service::RunningService<rmcp::RoleClient, ()>, tool: &'static str, args: Value) -> String {
    let r = client.call_tool(CallToolRequestParams::new(tool).with_arguments(args.as_object().cloned().unwrap())).await.unwrap();
    assert_ne!(r.is_error, Some(true), "{r:?}");
    r.content.iter().filter_map(|c| c.as_text().map(|t| t.text.clone())).collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn mcp_search_routes_identifier_queries_without_a_summary_section() {
    let e = indexed(RANKING).await;
    let server = ContextServer::new(context(e.store.clone()), sources::project::key(ALPHA), e.cfg.state_path.clone()).with_lexical(Lexical::Bm25);
    let (server_io, client_io) = tokio::io::duplex(1 << 16);
    tokio::spawn(async move {
        let running = server.serve(server_io).await.unwrap();
        let _ = running.waiting().await;
    });
    let client = ().serve(client_io).await.unwrap();

    let out = call(&client, "search_sessions", json!({"query": ID_QUERY})).await;
    assert!(out.starts_with("[1] claude · z:/work/alpha · session ident · turn 1\n"), "{out}");
    assert!(!out.contains("## Session summaries") && !out.contains("[S1]"), "{out}");
    assert!(!out.contains("z:/work/beta"), "{out}");

    let natural = call(&client, "search_sessions", json!({"query": NATURAL_QUERY})).await;
    assert!(natural.starts_with("## Session summaries"), "{natural}");
}

/// Same identifier everywhere: an old turn, a new one, another project's and a subagent's.
const SCOPED: &[Fixture] = &[
    Fixture { id: "old", cwd: ALPHA, at: "2025-01-10T10:00:00Z", prompts: &["deploy RDE_2772 to staging", "thanks"] },
    Fixture { id: "new", cwd: ALPHA, at: "2026-09-20T10:00:00Z", prompts: &["deploy RDE_2772 to production", "thanks"] },
    Fixture { id: "other", cwd: BETA, at: "2026-09-20T10:00:00Z", prompts: &["deploy RDE_2772 to beta", "thanks"] },
    Fixture { id: "notes", cwd: ALPHA, at: "2026-09-21T10:00:00Z", prompts: &["write the release notes", "thanks"] },
];

#[tokio::test(flavor = "multi_thread")]
async fn routed_legs_keep_project_subagent_and_date_filters() {
    let e = indexed(SCOPED).await;
    // A subagent of `new` (fixture 1, so directory p1) that also names the identifier, indexed
    // by a second run.
    let dir = e.cfg.claude_dir.join("p1").join("new").join("subagents");
    fs::create_dir_all(&dir).unwrap();
    let sub = [
        json!({"type": "user", "uuid": "a1", "isSidechain": true, "cwd": ALPHA, "timestamp": "2026-09-20T11:00:00Z",
               "message": {"content": "grep for RDE_2772 in the deploy scripts"}}),
        json!({"type": "assistant", "timestamp": "2026-09-20T11:00:05Z", "message": {"content": [{"type": "text", "text": "RDE_2772 is in deploy.sh"}]}}),
    ]
    .map(|v| v.to_string())
    .join("\n");
    fs::write(dir.join("agent-x.jsonl"), sub).unwrap();
    let refs = sources::claude::discover(&e.cfg.claude_dir).unwrap();
    let state = State::open(&e.cfg.state_path).unwrap();
    let report = ctx::run_index(&e.store, &mut HashEmbedder::new(DIM), "hash", &WordCounter, &state, &refs, &ingest::Redactor::default(), |_| {}).await.unwrap();
    assert_eq!(report.indexed, 1, "only the subagent is new");

    let ctx = context(e.store.clone());
    // Each session's set, as the dense leg also returns the identifier-free turns in scope.
    let named = |r: &Results| sessions(r).into_iter().filter(|s| s != "notes").collect::<BTreeSet<_>>();
    let mut q = query("RDE_2772", alpha(), Lexical::Bm25);
    q.limit = 10;
    assert_eq!(named(&ctx.search(&q).await.unwrap()), set(&["new", "old"]), "no other project, no subagent by default");

    q.include_subagents = true;
    assert_eq!(named(&ctx.search(&q).await.unwrap()), set(&["new", "new/agent-x", "old"]));

    q.include_subagents = false;
    q.since = Some(ctx::search::parse_date("2026-01-01", false).unwrap());
    assert_eq!(named(&ctx.search(&q).await.unwrap()), set(&["new"]));
    q.since = None;
    q.until = Some(ctx::search::parse_date("2025-12-31", true).unwrap());
    assert_eq!(named(&ctx.search(&q).await.unwrap()), set(&["old"]), "the date filter applies to every leg");

    let mut q = query("RDE_2772", Scope::All, Lexical::Bm25);
    q.limit = 10;
    assert_eq!(named(&ctx.search(&q).await.unwrap()), set(&["new", "old", "other"]));
}

fn set(ids: &[&str]) -> BTreeSet<String> {
    ids.iter().map(|s| (*s).to_owned()).collect()
}

struct Plain {
    url: String,
    tmp: tempfile::TempDir,
    _qdrant: testkit::Qdrant,
}

impl Plain {
    fn state_path(&self) -> PathBuf {
        self.tmp.path().join("state.db")
    }

    async fn index(&self, store: &Store) -> ingest::Report {
        let refs = sources::claude::discover(&self.tmp.path().join("projects")).unwrap();
        let state = State::open(&self.state_path()).unwrap();
        ctx::run_index(store, &mut HashEmbedder::new(DIM), "hash", &WordCounter, &state, &refs, &ingest::Redactor::default(), |_| {}).await.unwrap()
    }
}

/// The fixtures indexed by a store without BM25, as every build before DEV-61 wrote them.
async fn plain_indexed(fixtures: &[Fixture]) -> (Plain, Store) {
    let qdrant = testkit::qdrant().await;
    let tmp = tempfile::tempdir().unwrap();
    for (i, f) in fixtures.iter().enumerate() {
        let dir = tmp.path().join("projects").join(format!("p{i}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join(format!("{}.jsonl", f.id)), common::session(f)).unwrap();
    }
    let plain = Plain { url: qdrant.url.clone(), tmp, _qdrant: qdrant };
    let store = Store::connect(&plain.url, None, "sessions", DIM).unwrap();
    assert_eq!(plain.index(&store).await.indexed, fixtures.len());
    (plain, store)
}

#[tokio::test(flavor = "multi_thread")]
async fn a_collection_without_bm25_still_answers_identifier_queries_densely() {
    let (_plain, store) = plain_indexed(RANKING).await;
    let turns = SearchFilter::default().eq("kind", "turn");
    assert!(store.search_bm25("token_stream_v2", &turns, 5).await.is_err(), "really an old collection");
    let ctx = context(Arc::new(store));

    let dense = ctx.search(&query(ID_QUERY, alpha(), Lexical::Off)).await.unwrap();
    for lexical in [Lexical::Bm25, Lexical::SparseBm25] {
        let r = ctx.search(&query(ID_QUERY, alpha(), lexical)).await.expect("falls back instead of failing");
        assert_eq!(sessions(&r), sessions(&dense), "{lexical:?}: the dense results");
        assert!(r.summaries.is_empty());
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn the_indexing_path_migrates_an_old_collection_without_losing_points() {
    let (plain, old) = plain_indexed(RANKING).await;
    let all = SearchFilter::default();
    let before = old.count(&all).await.unwrap();
    assert_eq!(before, 14, "two turns per fixture");

    let store = Store::connect(&plain.url, None, "sessions", DIM).unwrap().with_lexical();
    let report = plain.index(&store).await;
    assert_eq!((report.indexed, report.unchanged), (0, RANKING.len()), "nothing re-embedded");
    assert_eq!(store.count(&all).await.unwrap(), before);
    let turns = SearchFilter::default().eq("kind", "turn").eq("project", "z:/work/alpha");
    let bm25 = store.search_bm25("token_stream_v2", &turns, 5).await.unwrap();
    assert_eq!(bm25[0].payload["session_id"], "ident");

    let r = context(Arc::new(store)).search(&query(ID_QUERY, alpha(), Lexical::Bm25)).await.unwrap();
    assert_eq!(sessions(&r)[0], "ident", "routed once migrated: {:?}", sessions(&r));
}

#[test]
fn ctx_lexical_selects_the_legs_with_bm25_by_default() {
    let cfg = |v: Option<&str>| Config::from_env(|k| if k == "CTX_LEXICAL" { v.map(str::to_owned) } else { None }).lexical;
    assert_eq!(cfg(None), Lexical::Bm25);
    assert_eq!(cfg(Some("off")), Lexical::Off);
    assert_eq!(cfg(Some("bm25")), Lexical::Bm25);
    assert_eq!(cfg(Some("sparse+bm25")), Lexical::SparseBm25);
}
