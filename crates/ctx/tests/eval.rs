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
    let reports = eval::run_retrieval(&ctx, &cases, &eval::variants(store::SearchMode::Hybrid, recency), &vectors).await.unwrap();
    assert_eq!(reports.len(), 6);
    let production = reports.iter().find(|r| r.name == PRODUCTION).unwrap();
    assert_eq!(production.cases.len(), 3, "negatives are not retrieval cases");
    assert_eq!(production.recall_at_5, 1.0, "{}", eval::format_misses(production, &cases));
    assert_eq!(production.latest, (1, 1), "the newer decision ranks first with recency");
    assert_eq!(production.turn_hit_at_5, (1, 1), "the one turn-targeted case finds its turn");
    let table = eval::format_retrieval(&reports);
    assert!(table.contains(PRODUCTION) && table.contains("turn@5"), "{table}");

    // --compare path: dense vs production, with a bootstrap CI and no panics on real data.
    let two = eval::variants(store::SearchMode::Hybrid, recency);
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

    let baseline = Baseline::from_report(production);
    assert!(!eval::compare(&baseline, production).is_regression());

    // Reranker spike path with the deterministic HashReranker (word overlap): it runs over a
    // real dense pool and produces a comparable report and per-query latencies.
    let mut reranker = embed::HashReranker;
    let (reranked, latency) = eval::run_reranked(&ctx, &cases, &vectors, &mut reranker, 20, "dense+rerank").await.unwrap();
    assert_eq!(reranked.cases.len(), 3, "reranked over the positive cases");
    assert_eq!(latency.len(), 3, "one timing per query");
    assert_eq!(reranked.turn_hit_at_5.1, 1, "the turn-targeted case is still counted");
    // The overlap reranker surfaces the jenkins turn for its query.
    assert!(reranked.cases.iter().any(|c| c.id == "deploy" && c.rank == Some(1)), "{:?}", reranked.cases);
    assert!(eval::format_latency("rerank", latency).contains("p50"));
}
