//! The eval harness on a synthetic query set against a real Qdrant: session-level ranking,
//! the recency variant preferring the newer of two conflicting decisions, and the inject sweep.

mod common;

use std::sync::{Arc, Mutex};

use common::{ALPHA, BETA, DIM, Fixture, indexed};
use ctx::eval::{self, Baseline, PRODUCTION};
use ctx::inject::{InjectConfig, Injector};
use ctx::rank::Recency;
use ctx::search::Context;
use embed::{Embedder, HashEmbedder};
use ingest::WordCounter;

const FIXTURES: &[Fixture] = &[
    Fixture { id: "model-old", cwd: ALPHA, at: "2025-01-10T10:00:00Z", prompts: &["which embedding model: we choose the int8 model", "thanks"] },
    Fixture { id: "model-new", cwd: ALPHA, at: "2026-09-20T10:00:00Z", prompts: &["which embedding model: we choose the fp32 model", "thanks"] },
    Fixture { id: "css", cwd: ALPHA, at: "2026-09-21T10:00:00Z", prompts: &["style the login button colours", "thanks"] },
    Fixture { id: "beta-deploy", cwd: BETA, at: "2026-09-22T10:00:00Z", prompts: &["jenkins deploy pipeline to production", "thanks"] },
];

const CASES: &str = r#"
# synthetic set: one superseded decision, one cross-project question, one negative
{"id": "model", "query": "which embedding model did we choose", "project": "Z:\\work\\alpha", "expect": ["model-new", "model-old"], "latest": true, "lang": "en"}
{"id": "deploy", "query": "jenkins deploy pipeline", "expect": ["beta-deploy"], "expect_turns": [{"session": "beta-deploy", "turn": 1}], "lang": "en"}
{"id": "css", "query": "login button colours", "project": "Z:\\work\\alpha", "expect": ["css"], "lang": "en"}
{"id": "none", "query": "kubernetes ingress certificates rotation", "project": "Z:\\work\\alpha", "lang": "en"}
"#;

#[tokio::test(flavor = "multi_thread")]
async fn eval_measures_variants_recency_and_inject_thresholds() {
    let e = indexed(FIXTURES).await;
    let cases = eval::parse_cases(CASES).unwrap();
    let ctx = Context { store: e.store.clone(), embedder: Arc::new(Mutex::new(Box::new(HashEmbedder::new(DIM)) as Box<dyn Embedder>)) };
    let vectors = eval::embed_queries(&ctx, &cases).await.unwrap();

    let recency = Recency { weight: 0.5, half_life_days: 30.0 };
    let reports = eval::run_retrieval(&ctx, &cases, &eval::variants(store::SearchMode::Hybrid, recency, ctx::lexical::Lexical::Off), &vectors).await.unwrap();
    assert_eq!(reports.len(), 6);
    let production = reports.iter().find(|r| r.name == PRODUCTION).unwrap();
    assert_eq!(production.cases.len(), 3, "negatives are not retrieval cases");
    assert_eq!(production.recall_at_5, 1.0, "{}", eval::format_misses(production, &cases));
    assert_eq!(production.latest, (1, 1), "the newer decision ranks first with recency");
    assert_eq!(production.turn_hit_at_5, (1, 1), "the one turn-targeted case finds its turn");
    let table = eval::format_retrieval(&reports);
    assert!(table.contains(PRODUCTION) && table.contains("turn@5"), "{table}");

    // --compare path: dense vs production, with a bootstrap CI and no panics on real data.
    let two = eval::variants(store::SearchMode::Hybrid, recency, ctx::lexical::Lexical::Off);
    let dense = two.iter().find(|v| v.name == "dense").copied().unwrap();
    let prod = two.iter().find(|v| v.name == PRODUCTION).copied().unwrap();
    let cmp = eval::run_retrieval(&ctx, &cases, &[dense, prod], &vectors).await.unwrap();
    let out = eval::format_compare(&cmp[0], &cmp[1], &cases);
    assert!(out.starts_with("compare: dense -> production"), "{out}");
    assert!(out.contains("R@1") && out.contains("MRR") && out.contains("turn-MRR"), "{out}");

    let injector = Injector::new(
        e.store.clone(),
        None,
        ctx.embedder.clone(),
        Arc::new(WordCounter),
        Recency::OFF,
        InjectConfig::default(),
        e.cfg.redact_file.clone(),
    );
    let points = eval::run_inject(&injector, &cases, &vectors, &[-1.0, 1.01]).await.unwrap();
    assert_eq!((points[0].positives, points[0].negatives), (2, 1), "only project-scoped cases");
    assert_eq!(points[0].false_injects, 1, "no threshold injects on the negative too");
    assert_eq!(points[0].hits, 2);
    assert_eq!((points[1].hits, points[1].false_injects), (0, 0), "an impossible threshold injects nothing");

    // Lexical legs on the hand-written set: project directories resolve through `Case::scope`,
    // and the cross-project case (no project) searches every project.
    let turns = ctx::eval_identifiers::load_turns(&e.store).await.unwrap();
    let bm25 = ctx::eval_identifiers::Bm25::new(&turns);
    let positives: Vec<eval::Case> = cases.iter().filter(|c| !c.expect.is_empty()).cloned().collect();
    let fused = ctx::eval_identifiers::run_fused(&ctx, &turns, &bm25, &positives, &vectors, &ctx::eval_identifiers::fused_variants(), &eval::Case::scope)
        .await
        .unwrap();
    assert_eq!(fused.len(), 6);
    let dense_only = fused.iter().find(|r| r.name == "routed dense+sparse+bm25").unwrap();
    let plain_dense = eval::run_retrieval(&ctx, &cases, &[dense], &vectors).await.unwrap().remove(0);
    let ranks = |r: &eval::VariantReport| r.cases.iter().map(|c| c.rank).collect::<Vec<_>>();
    assert_eq!(ranks(dense_only), ranks(&plain_dense), "no identifier in these queries, so routing keeps dense alone");
    let bm = fused.iter().find(|r| r.name == "bm25").unwrap();
    assert_eq!(bm.cases.iter().find(|c| c.id == "deploy").unwrap().rank, Some(1), "found across projects");
    assert_eq!(bm.cases.iter().find(|c| c.id == "css").unwrap().rank, Some(1), "found inside its project");

    let baseline = Baseline::from_report(production);
    assert!(!eval::compare(&baseline, production).is_regression());

    // Reranker spike path with the deterministic HashReranker (word overlap): it runs over a
    // real dense pool and produces a comparable report and per-query latencies.
    let mut reranker = embed::HashReranker;
    let (reranked, latency) = eval::run_reranked(&ctx, &cases, &vectors, &mut reranker, 20, ctx::lexical::Lexical::Off, &eval::Case::scope, "dense+rerank").await.unwrap();
    assert_eq!(reranked.cases.len(), 3, "reranked over the positive cases");
    assert_eq!(latency.len(), 3, "one timing per query");
    assert_eq!(reranked.turn_hit_at_5.1, 1, "the turn-targeted case is still counted");
    // The overlap reranker surfaces the jenkins turn for its query.
    assert!(reranked.cases.iter().any(|c| c.id == "deploy" && c.rank == Some(1)), "{:?}", reranked.cases);
    assert!(eval::format_latency("rerank", latency).contains("p50"));

    // Deep search (opt-in): the same reordering, but summaries stay first and the turns are cut
    // to the caller's limit. Uses a local HashReranker, no daemon.
    let rr: Arc<Mutex<Box<dyn embed::Reranker>>> = Arc::new(Mutex::new(Box::new(embed::HashReranker)));
    let mut q = ctx::search::Query::new("jenkins deploy pipeline to production", ctx::search::Scope::All, 3);
    q.recency = Recency::OFF;
    let deep = ctx.search_reranked(&q, rr, 20).await.unwrap();
    assert!(deep.turns.len() <= 3, "cut to the limit");
    let top = deep.turns[0].payload.get("session_id").and_then(|v| v.as_str()).unwrap();
    assert_eq!(top, "beta-deploy", "the overlap reranker puts the jenkins turn first");
}

const ID_FIXTURES: &[Fixture] = &[
    Fixture { id: "lexer", cwd: ALPHA, at: "2026-09-20T10:00:00Z", prompts: &["the parser chokes, look at token_stream_v2 please", "thanks"] },
    Fixture { id: "ticket", cwd: ALPHA, at: "2026-09-21T10:00:00Z", prompts: &["move ticket SEDN26-1910 to review", "thanks"] },
    Fixture { id: "shared", cwd: ALPHA, at: "2026-09-22T10:00:00Z", prompts: &["edit config_loader.rs", "and config_loader.rs again"] },
    Fixture { id: "shared-too", cwd: ALPHA, at: "2026-09-23T10:00:00Z", prompts: &["config_loader.rs is broken", "thanks"] },
    Fixture { id: "beta", cwd: BETA, at: "2026-09-22T10:00:00Z", prompts: &["deploy RDE_2772 to staging", "thanks"] },
];

/// `ctx eval --target identifiers` end to end: cases generated from the indexed turns, the
/// production variants through the real search path, and the in-memory BM25 fusions.
#[tokio::test(flavor = "multi_thread")]
async fn identifier_eval_generates_cases_from_the_index_and_scores_every_leg() {
    use ctx::eval_identifiers as ids;

    let e = indexed(ID_FIXTURES).await;
    let ctx = Context { store: e.store.clone(), embedder: Arc::new(Mutex::new(Box::new(HashEmbedder::new(DIM)) as Box<dyn Embedder>)) };
    let turns = ids::load_turns(&e.store).await.unwrap();
    assert!(turns.len() >= 10, "every turn of every fixture: {}", turns.len());

    let (picked, _) = ids::pick(&turns, 100);
    let mut found: Vec<(&str, &str)> = picked.iter().map(|p| (p.identifier.as_str(), p.session.as_str())).collect();
    found.sort_unstable();
    assert_eq!(found, [("RDE_2772", "beta"), ("SEDN26-1910", "ticket"), ("token_stream_v2", "lexer")], "config_loader.rs is in two sessions");
    let beta = picked.iter().find(|p| p.session == "beta").unwrap();
    assert_eq!(beta.project, "z:/work/beta");
    assert_eq!(beta.turns, [0]);

    let (bare, sentence) = ids::cases(&picked);
    let all: Vec<eval::Case> = bare.iter().chain(&sentence).cloned().collect();
    let vectors = eval::embed_queries(&ctx, &all).await.unwrap();
    let bm25 = ids::Bm25::new(&turns);
    let variants = eval::variants(store::SearchMode::Dense, Recency::OFF, ctx::lexical::Lexical::Off);
    let fused = ids::fused_variants();
    let reports = ids::run(&ctx, &turns, &bm25, &bare, &vectors, &variants, &fused).await.unwrap();
    assert_eq!(reports.len(), variants.len() + fused.len());
    for r in &reports {
        assert_eq!(r.cases.len(), 3, "{}", r.name);
    }
    let by = |name: &str| reports.iter().find(|r| r.name == name).unwrap();
    assert_eq!(by("bm25").recall_at_1, 1.0, "a unique exact token is BM25's best case");
    assert_eq!(by("dense+sparse+bm25").recall_at_5, 1.0);
    assert_eq!(by("bm25").turn_hit_at_5, (3, 3), "the turn that mentions it is shown");

    let reports = ids::run(&ctx, &turns, &bm25, &sentence, &vectors, &variants, &fused).await.unwrap();
    let table = eval::format_retrieval(&reports);
    assert!(table.contains("dense+sparse+bm25") && table.contains("pl R@5") && table.contains("en R@5"), "{table}");
}
