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
{"id": "deploy", "query": "jenkins deploy pipeline", "expect": ["beta-deploy"], "lang": "en"}
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
    assert!(eval::format_retrieval(&reports).contains(PRODUCTION));

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
}
