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

fn day_ms(y: i16, m: i8, d: i8) -> i64 {
    jiff::civil::date(y, m, d).to_zoned(jiff::tz::TimeZone::UTC).unwrap().timestamp().as_millisecond()
}

#[test]
fn the_cutoff_key_names_the_collection() {
    assert_eq!(ctx::prune::cutoff_key("sessions"), "prune/before_ms/sessions");
    assert_eq!(ctx::prune::cutoff_key("meetings"), "prune/before_ms/meetings");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_applied_prune_records_the_highest_cutoff_per_collection_and_a_dry_run_nothing() {
    let q = testkit::qdrant().await;
    let sessions = store::Store::connect(&q.url, None, "sessions", common::DIM).unwrap();
    sessions.ensure_collection(ingest::SESSION_FIELDS).await.unwrap();
    let teams = store::Store::connect(&q.url, None, "teams", common::DIM).unwrap();
    teams.ensure_collection(ingest::SESSION_FIELDS).await.unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("state.db");
    let recorded = |collection: &str| State::open(&path).unwrap().get_meta(&ctx::prune::cutoff_key(collection)).unwrap();
    let (jan, mar, jun) = (day_ms(2026, 1, 1), day_ms(2026, 3, 1), day_ms(2026, 6, 1));

    let dry = prune_older_than(&sessions, &path, "session_id", mar, false).await.unwrap();
    assert_eq!((dry.points, dry.applied), (0, false));
    assert_eq!(recorded("sessions"), None, "a dry run records nothing");

    let done = prune_older_than(&sessions, &path, "session_id", mar, true).await.unwrap();
    assert_eq!((done.points, done.items, done.applied), (0, 0, true), "nothing matched");
    assert_eq!(recorded("sessions"), Some(mar.to_string()), "recorded even when nothing matched");

    prune_older_than(&sessions, &path, "session_id", jan, true).await.unwrap();
    assert_eq!(recorded("sessions"), Some(mar.to_string()), "a narrower prune does not lower it");

    prune_older_than(&sessions, &path, "session_id", jun, true).await.unwrap();
    assert_eq!(recorded("sessions"), Some(jun.to_string()), "a wider one raises it");

    assert_eq!(recorded("teams"), None, "per collection");
    prune_older_than(&teams, &path, "thread_id", jan, true).await.unwrap();
    assert_eq!(recorded("teams"), Some(jan.to_string()));
    assert_eq!(recorded("sessions"), Some(jun.to_string()));
}

#[test]
fn index_config_reads_its_collections_cutoff_and_carries_the_redactor() {
    let state = State::in_memory().unwrap();
    // Never contacted: index_config only reads the collection's name.
    let sessions = store::Store::connect("http://127.0.0.1:9", None, "sessions", common::DIM).unwrap();
    let jira = store::Store::connect("http://127.0.0.1:9", None, "jira", common::DIM).unwrap();
    let teams = store::Store::connect("http://127.0.0.1:9", None, "teams", common::DIM).unwrap();
    state.set_meta(&ctx::prune::cutoff_key("sessions"), "1767225600000").unwrap();
    state.set_meta(&ctx::prune::cutoff_key("teams"), "not a number").unwrap();
    let redactor = ingest::Redactor::new(["plainsecret1".to_owned()]);

    let config = ctx::index_config(&sessions, &state, &redactor).unwrap();
    assert_eq!(config.prune_before_ms, Some(1_767_225_600_000));
    assert_eq!(config.chunk.redactor.scrub("x plainsecret1 y").0, "x [REDACTED:listed] y");
    assert_eq!(ctx::index_config(&jira, &state, &redactor).unwrap().prune_before_ms, None, "no prune recorded for jira");
    assert_eq!(ctx::index_config(&teams, &state, &redactor).unwrap().prune_before_ms, None, "an unreadable value is no cutoff");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_re_index_after_prune_older_than_does_not_bring_the_old_turns_back() {
    let e = indexed(FIXTURES).await;
    // A session spanning the cutoff: its first turn before it, its second after.
    let dir = e.cfg.claude_dir.join("p9");
    std::fs::create_dir_all(&dir).unwrap();
    let mixed = [
        serde_json::json!({"type": "user", "uuid": "u0", "timestamp": "2025-02-01T10:00:00Z", "cwd": ALPHA, "origin": {"kind": "human"}, "message": {"content": "mixed old turn"}}),
        serde_json::json!({"type": "assistant", "timestamp": "2025-02-01T10:00:05Z", "message": {"content": [{"type": "text", "text": "old answer"}]}}),
        serde_json::json!({"type": "user", "uuid": "u1", "timestamp": "2026-09-22T10:00:00Z", "cwd": ALPHA, "origin": {"kind": "human"}, "message": {"content": "mixed new turn"}}),
        serde_json::json!({"type": "assistant", "timestamp": "2026-09-22T10:00:05Z", "message": {"content": [{"type": "text", "text": "new answer"}]}}),
    ];
    std::fs::write(dir.join("mixed-a.jsonl"), mixed.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("\n")).unwrap();
    let reindex = || async {
        let refs = ctx::discover(&e.cfg, ctx::Sources::All).unwrap();
        let state = State::open(&e.cfg.state_path).unwrap();
        let mut emb = embed::HashEmbedder::new(common::DIM);
        ctx::run_index(&e.store, &mut emb, "hash", &ingest::WordCounter, &state, &refs, &ingest::Redactor::default(), |_| {}).await.unwrap()
    };
    reindex().await;
    let store = &e.store;
    let turns_of = move |id: &'static str| async move {
        let mut v: Vec<i64> = store.scroll(&SearchFilter::default().eq("session_id", id).eq("kind", "turn"), 50).await.unwrap().iter().map(|h| h.payload["turn_index"].as_i64().unwrap()).collect();
        v.sort();
        v
    };
    assert_eq!(turns_of("mixed-a").await, [0, 1], "precondition: both turns indexed");
    let new_b_points = e.store.count(&SearchFilter::default().eq("session_id", "new-b")).await.unwrap();

    let cutoff = day_ms(2026, 1, 1);
    prune_older_than(&e.store, &e.cfg.state_path, "session_id", cutoff, true).await.unwrap();
    let r = reindex().await;

    assert!(r.indexed >= 2, "old-a and mixed-a were re-read: {r:?}");
    assert_eq!(e.store.count(&SearchFilter::default().between(None, Some(cutoff))).await.unwrap(), 0, "no point at or before the cutoff came back");
    assert_eq!(turns_of("old-a").await, Vec::<i64>::new());
    assert_eq!(turns_of("mixed-a").await, [1], "only the turn after the cutoff");
    assert_eq!(e.store.count(&SearchFilter::default().eq("session_id", "new-b")).await.unwrap(), new_b_points, "newer sessions untouched");
}
