//! Optional prune end to end against a real Qdrant: dry-run vs apply, project isolation, age
//! cutoff, and re-index restoring a still-present source.

mod common;

use common::{ALPHA, BETA, Fixture, indexed};
use ctx::prune::{prune_older_than, prune_project};
use ingest::State;
use store::SearchFilter;

const FIXTURES: &[Fixture] = &[
    Fixture { id: "old-a", cwd: ALPHA, at: "2025-01-10T10:00:00Z", prompts: &["alpha old migrate the database", "thanks"] },
    Fixture { id: "new-a", cwd: ALPHA, at: "2026-09-20T10:00:00Z", prompts: &["alpha new add rollback", "thanks"] },
    Fixture { id: "new-b", cwd: BETA, at: "2026-09-21T10:00:00Z", prompts: &["beta jenkins deploy", "thanks"] },
];

fn key(dir: &str) -> String {
    sources::project::key(dir)
}

#[tokio::test(flavor = "multi_thread")]
async fn prune_project_is_dry_by_default_then_removes_only_that_project() {
    let e = indexed(FIXTURES).await;
    let any = SearchFilter::default();
    let all = || e.store.count(&any);
    let before = all().await.unwrap();
    assert!(before > 0);
    let a = SearchFilter::default().eq("project", key(ALPHA));
    let a_before = e.store.count(&a).await.unwrap();
    assert!(a_before > 0);

    // Dry run: reports, deletes nothing.
    let dry = prune_project(&e.store, &e.cfg.state_path, &key(ALPHA), false).await.unwrap();
    assert!(!dry.applied && dry.points == a_before && dry.items == 2, "{dry:?}");
    assert_eq!(all().await.unwrap(), before, "dry run changed nothing");

    // Apply: only ALPHA goes, from Qdrant and from state; BETA stays.
    let done = prune_project(&e.store, &e.cfg.state_path, &key(ALPHA), true).await.unwrap();
    assert!(done.applied && done.points == a_before);
    assert_eq!(e.store.count(&a).await.unwrap(), 0, "alpha points gone");
    assert_eq!(e.store.count(&SearchFilter::default().eq("project", key(BETA))).await.unwrap(), before - a_before, "beta kept");
    let state = State::open(&e.cfg.state_path).unwrap();
    assert!(state.get("claude", "old-a").unwrap().is_none() && state.get("claude", "new-a").unwrap().is_none());
    assert!(state.get("claude", "new-b").unwrap().is_some(), "beta state kept");
    drop(state);

    // Re-index: ALPHA's transcripts still exist, so it comes back (point-in-time semantics).
    let refs = ctx::discover(&e.cfg, ctx::Sources::All).unwrap();
    let mut emb = embed::HashEmbedder::new(common::DIM);
    let state = State::open(&e.cfg.state_path).unwrap();
    ctx::run_index(&e.store, &mut emb, "hash", &ingest::WordCounter, &state, &refs, &ingest::Redactor::default(), |_| {}).await.unwrap();
    assert!(e.store.count(&a).await.unwrap() > 0, "alpha re-indexed since its source is still present");
}

#[tokio::test(flavor = "multi_thread")]
async fn prune_project_refuses_empty_key_and_older_than_cuts_by_age() {
    let e = indexed(FIXTURES).await;
    assert!(prune_project(&e.store, &e.cfg.state_path, "", true).await.is_err(), "empty key refused");

    // Cutoff between 2025 (old-a) and 2026 (new-*): only old-a is older.
    let cutoff = jiff::civil::date(2026, 1, 1).to_zoned(jiff::tz::TimeZone::UTC).unwrap().timestamp().as_millisecond();
    let dry = prune_older_than(&e.store, &e.cfg.state_path, "session_id", cutoff, false).await.unwrap();
    assert_eq!((dry.items, dry.applied), (1, false), "one old session, dry: {dry:?}");
    let old_points = dry.points;

    let done = prune_older_than(&e.store, &e.cfg.state_path, "session_id", cutoff, true).await.unwrap();
    assert!(done.applied && done.points == old_points);
    assert_eq!(e.store.count(&SearchFilter::default().between(None, Some(cutoff))).await.unwrap(), 0, "old points gone");
    assert!(e.store.count(&SearchFilter::default()).await.unwrap() > 0, "newer kept");
    let state = State::open(&e.cfg.state_path).unwrap();
    assert!(state.get("claude", "old-a").unwrap().is_none(), "old session state removed");
    assert!(state.get("claude", "new-a").unwrap().is_some(), "newer session state kept");
}
