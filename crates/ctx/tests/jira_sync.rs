//! `sync_jira` against a stateful mock Jira and a real Qdrant: cursor paging by `updated`, the
//! watermark, the daily reconcile that drops deleted and moved issues, and the prune cutoff.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use axum::extract::{Query, State as AxState};
use axum::http::{StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::{Json, Router};
use ctx::jira::Jira;
use embed::{Embedder, Embedding, HashEmbedder};
use ingest::{Report, State, WordCounter};
use serde_json::{Value, json};
use store::{SearchFilter, Store};

const DIM: usize = 64;
const DAY_MS: i64 = 24 * 3_600_000;

#[derive(Clone)]
struct Issue {
    key: String,
    updated: String,
    summary: String,
    comments: Vec<String>,
}

enum IssueGet {
    NotFound,
    Moved(&'static str),
    Fails,
}

/// A Jira that filters and orders like the real one for the JQL `sync_jira` sends, answering at
/// most `cap` issues per response, and records every request.
#[derive(Default)]
struct Mock {
    issues: Vec<Issue>,
    cap: usize,
    /// Overrides `total` in the full-issue search replies.
    total_lie: Option<u64>,
    /// What the key listing (`fields=key`) lists; the issues' keys when unset.
    listing: Option<Vec<String>>,
    /// Overrides `total` in the key listing replies.
    listing_total: Option<u64>,
    /// Answers every key listing request as is.
    listing_reply: Option<Value>,
    gets: HashMap<String, IssueGet>,
    /// Before full search number N (1-based): set the issue's `updated`.
    edit: Option<(usize, String, String)>,
    /// `(jql, startAt)` of every full-issue search.
    searches: Vec<(String, String)>,
    /// `(jql, startAt)` of every key listing.
    listings: Vec<(String, String)>,
    issue_gets: Vec<String>,
}

type Shared = Arc<Mutex<Mock>>;

fn updated_ms(s: &str) -> i64 {
    ingest::jira::updated_ms(s).unwrap()
}

/// The `updated >= "…"` bound of `jql` in epoch ms, read the way Jira reads it (local time,
/// minute precision); `None` without one.
fn since(jql: &str) -> Option<i64> {
    let at = jql.find("updated >= \"")? + "updated >= \"".len();
    let local = jiff::civil::DateTime::strptime("%Y-%m-%d %H:%M", &jql[at..at + 16]).unwrap();
    Some(local.to_zoned(jiff::tz::TimeZone::system()).unwrap().timestamp().as_millisecond())
}

fn issue_json(i: &Issue, key: &str) -> Value {
    let comments: Vec<Value> =
        i.comments.iter().map(|c| json!({"author": {"displayName": "Bob"}, "created": "2026-10-01T12:00:00.000+0000", "body": c})).collect();
    json!({"key": key, "fields": {"summary": i.summary, "description": format!("details of {}", i.key), "status": {"name": "Open"},
        "issuetype": {"name": "Task"}, "assignee": {"displayName": "Anna"}, "updated": i.updated, "comment": {"comments": comments}}})
}

fn page<T: Clone>(all: &[T], start: usize, max: usize, cap: usize) -> Vec<T> {
    all.iter().skip(start).take(max.min(cap)).cloned().collect()
}

async fn handle(AxState(m): AxState<Shared>, uri: Uri, Query(q): Query<HashMap<String, String>>) -> Response {
    let mut m = m.lock().unwrap();
    let path = uri.path().to_owned();
    if path == "/rest/api/2/search" {
        let jql = q.get("jql").cloned().unwrap_or_default();
        let start_at = q.get("startAt").cloned().unwrap_or_default();
        let (start, max): (usize, usize) = (start_at.parse().unwrap(), q["maxResults"].parse().unwrap());
        if m.searches.len() + m.listings.len() > 60 {
            return (StatusCode::INTERNAL_SERVER_ERROR, "too many requests: the sync does not terminate").into_response();
        }
        if q.get("fields").map(String::as_str) == Some("key") {
            m.listings.push((jql, start_at));
            if let Some(reply) = &m.listing_reply {
                return Json(reply.clone()).into_response();
            }
            let mut keys = m.listing.clone().unwrap_or_else(|| m.issues.iter().map(|i| i.key.clone()).collect());
            keys.sort();
            let total = m.listing_total.unwrap_or(keys.len() as u64);
            let issues: Vec<Value> = page(&keys, start, max, m.cap).into_iter().map(|k| json!({"key": k})).collect();
            return Json(json!({"startAt": start, "total": total, "issues": issues})).into_response();
        }
        m.searches.push((jql.clone(), start_at));
        let n = m.searches.len();
        if let Some((at, key, updated)) = m.edit.clone()
            && at == n
        {
            m.issues.iter_mut().find(|i| i.key == key).unwrap().updated = updated;
        }
        let from = since(&jql).unwrap_or(i64::MIN);
        let mut matching: Vec<Issue> = m.issues.iter().filter(|i| updated_ms(&i.updated) >= from).cloned().collect();
        matching.sort_by(|a, b| (updated_ms(&a.updated), &a.key).cmp(&(updated_ms(&b.updated), &b.key)));
        let total = m.total_lie.unwrap_or(matching.len() as u64);
        let issues: Vec<Value> = page(&matching, start, max, m.cap).iter().map(|i| issue_json(i, &i.key)).collect();
        return Json(json!({"startAt": start, "total": total, "issues": issues})).into_response();
    }
    if let Some(key) = path.strip_prefix("/rest/api/2/issue/") {
        let key = key.to_owned();
        m.issue_gets.push(key.clone());
        let found = m.issues.iter().find(|i| i.key == key).cloned();
        return match (m.gets.get(&key), found) {
            (Some(IssueGet::NotFound), _) | (None, None) => StatusCode::NOT_FOUND.into_response(),
            (Some(IssueGet::Fails), _) => (StatusCode::INTERNAL_SERVER_ERROR, "boom").into_response(),
            (Some(IssueGet::Moved(to)), Some(i)) => Json(issue_json(&i, to)).into_response(),
            (Some(IssueGet::Moved(_)), None) => panic!("a moved issue must exist in the fixture"),
            (None, Some(i)) => Json(issue_json(&i, &key)).into_response(),
        };
    }
    StatusCode::NOT_FOUND.into_response()
}

async fn serve(mock: Mock) -> (String, Shared) {
    let shared: Shared = Arc::new(Mutex::new(mock));
    let app = Router::new().fallback(handle).with_state(shared.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (url, shared)
}

/// `updated` of an issue edited `minutes` after 2026-10-01 10:00 UTC.
fn at(minutes: i64) -> String {
    format!("2026-10-01T{:02}:{:02}:00.000+0000", 10 + minutes / 60, minutes % 60)
}

fn issue(key: &str, updated: &str) -> Issue {
    Issue { key: key.into(), updated: updated.into(), summary: format!("summary of {key}"), comments: vec![] }
}

/// HD-1..HD-n edited 10 minutes apart, HD-1 first.
fn spaced(n: usize) -> Vec<Issue> {
    (1..=n).map(|i| issue(&format!("HD-{i}"), &at(10 * i as i64))).collect()
}

fn jql(since: Option<&str>) -> String {
    ingest::jira::sync_jql(&["HD".to_owned()], since.map(updated_ms))
}

/// HashEmbedder that fails on a batch holding "POISON" while `poisoned` is set.
struct Poisonable(Arc<AtomicBool>);

impl Embedder for Poisonable {
    fn dim(&self) -> usize {
        DIM
    }
    fn embed(&mut self, texts: &[&str]) -> embed::Result<Vec<Embedding>> {
        if self.0.load(Ordering::SeqCst) && texts.iter().any(|t| t.contains("POISON")) {
            return Err(embed::EmbedError::Model("cuda out of memory".into()));
        }
        HashEmbedder::new(DIM).embed(texts)
    }
}

struct Env {
    store: Store,
    state_path: std::path::PathBuf,
    client: Jira,
    url: String,
    mock: Shared,
    poisoned: Arc<AtomicBool>,
    _tmp: tempfile::TempDir,
    _q: testkit::Qdrant,
}

async fn env(mock: Mock) -> Env {
    let q = testkit::qdrant().await;
    let store = Store::connect(&q.url, None, "jira", DIM).unwrap();
    store.ensure_collection(ingest::jira::JIRA_FIELDS).await.unwrap();
    let (url, mock) = serve(mock).await;
    let tmp = tempfile::tempdir().unwrap();
    Env { store, state_path: tmp.path().join("state.db"), client: Jira::new(&url, "tok"), url, mock, poisoned: Arc::default(), _tmp: tmp, _q: q }
}

impl Env {
    fn state(&self) -> State {
        State::open(&self.state_path).unwrap()
    }

    async fn try_sync(&self, now_ms: i64) -> Result<Report, ingest::IndexError> {
        let state = self.state();
        let config = ctx::index_config(&self.store, &state, &ingest::Redactor::default()).unwrap();
        let mut emb = Poisonable(self.poisoned.clone());
        ingest::jira::sync_jira(&self.store, &mut emb, &WordCounter, &state, &self.client, &["HD".to_owned()], &config, None, now_ms).await
    }

    async fn sync(&self, now_ms: i64) -> Report {
        self.try_sync(now_ms).await.unwrap()
    }

    fn searches(&self) -> Vec<(String, String)> {
        std::mem::take(&mut self.mock.lock().unwrap().searches)
    }

    fn listings(&self) -> Vec<(String, String)> {
        std::mem::take(&mut self.mock.lock().unwrap().listings)
    }

    fn issue_gets(&self) -> Vec<String> {
        let mut g = std::mem::take(&mut self.mock.lock().unwrap().issue_gets);
        g.sort();
        g
    }

    async fn points(&self, key: &str) -> u64 {
        self.store.count(&SearchFilter::default().eq("issue_key", key.to_owned())).await.unwrap()
    }

    fn fingerprint(&self, key: &str) -> Option<String> {
        self.state().get("jira", key).unwrap().map(|i| i.fingerprint)
    }

    fn meta(&self, key: &str) -> Option<String> {
        self.state().get_meta(key).unwrap()
    }
}

fn s(jql: String, start: &str) -> (String, String) {
    (jql, start.to_owned())
}

const NOW: i64 = 1_790_000_000_000;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn each_full_page_restarts_at_offset_zero_from_the_newest_updated_seen() {
    let issues = spaced(5);
    let e = env(Mock { issues: issues.clone(), cap: 2, ..Mock::default() }).await;

    let r = e.sync(NOW).await;

    assert_eq!(
        e.searches(),
        [s(jql(None), "0"), s(jql(Some(&at(20))), "0"), s(jql(Some(&at(30))), "0"), s(jql(Some(&at(40))), "0")]
    );
    // HD-2..HD-4 come twice with the same `updated`: skipped, not counted as unchanged.
    assert_eq!((r.scanned, r.indexed, r.unchanged, r.failed.len()), (5, 5, 0, 0), "{r:?}");
    for i in &issues {
        assert_eq!(e.fingerprint(&i.key), Some(i.updated.clone()), "{}", i.key);
    }
    assert_eq!(e.meta("jira/watermark_ms"), Some(updated_ms(&at(50)).to_string()));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_full_page_inside_one_cursor_window_continues_by_offset() {
    let issues = vec![issue("HD-1", &at(10)), issue("HD-2", &at(10)), issue("HD-3", &at(10)), issue("HD-4", &at(40)), issue("HD-5", &at(50))];
    let e = env(Mock { issues: issues.clone(), cap: 2, ..Mock::default() }).await;

    let r = e.sync(NOW).await;

    let t1 = jql(Some(&at(10)));
    assert_eq!(e.searches(), [s(jql(None), "0"), s(t1.clone(), "0"), s(t1, "2"), s(jql(Some(&at(40))), "0")]);
    assert_eq!((r.scanned, r.indexed, r.unchanged), (5, 5, 0), "{r:?}");
    for i in &issues {
        assert_eq!(e.fingerprint(&i.key), Some(i.updated.clone()), "{}", i.key);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_issue_edited_mid_run_does_not_make_a_later_one_skipped() {
    // HD-1 is edited after the first page; by offset the second page would start at HD-4.
    let e = env(Mock { issues: spaced(5), cap: 2, edit: Some((2, "HD-1".into(), at(60))), ..Mock::default() }).await;

    let r = e.sync(NOW).await;

    for (key, updated) in [("HD-1", at(60)), ("HD-2", at(20)), ("HD-3", at(30)), ("HD-4", at(40)), ("HD-5", at(50))] {
        assert_eq!(e.fingerprint(key), Some(updated), "{key}");
        assert!(e.points(key).await > 0, "{key} indexed");
    }
    assert_eq!(r.scanned, 5, "each key counted once although HD-1 came twice: {r:?}");
    assert_eq!((r.indexed, r.unchanged), (6, 0), "HD-1 is indexed again with its new `updated`: {r:?}");
    assert_eq!(e.meta("jira/watermark_ms"), Some(updated_ms(&at(60)).to_string()));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn paging_stops_at_an_empty_page_when_total_overstates() {
    let e = env(Mock { issues: spaced(3), cap: 2, total_lie: Some(10), ..Mock::default() }).await;

    let r = e.sync(NOW).await;

    let t3 = jql(Some(&at(30)));
    assert_eq!(e.searches(), [s(jql(None), "0"), s(jql(Some(&at(20))), "0"), s(t3.clone(), "0"), s(t3, "1")]);
    assert_eq!((r.scanned, r.indexed, r.unchanged), (3, 3, 0), "{r:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_issue_failing_on_overlapping_pages_is_reported_once() {
    // HD-2 is on the first page and again on the restarted second one, same `updated`.
    let mut issues = spaced(4);
    issues[1].summary = "POISON in the summary".into();
    let e = env(Mock { issues, cap: 2, ..Mock::default() }).await;
    e.poisoned.store(true, Ordering::SeqCst);

    let r = e.sync(NOW).await;

    assert_eq!(e.searches(), [s(jql(None), "0"), s(jql(Some(&at(20))), "0"), s(jql(Some(&at(30))), "0")]);
    assert_eq!(r.failed, [("HD-2".to_owned(), "model error: cuda out of memory".to_owned())]);
    assert_eq!((r.scanned, r.indexed, r.unchanged), (4, 3, 0), "{r:?}");
    assert_eq!(e.fingerprint("HD-2"), None);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_watermark_advances_only_after_a_run_without_failures() {
    let mut issues = spaced(3);
    issues[1].summary = "POISON in the summary".into();
    let e = env(Mock { issues, cap: 50, ..Mock::default() }).await;
    let before = updated_ms(&at(0)).to_string();
    e.state().set_meta("jira/watermark_ms", &before).unwrap();
    e.poisoned.store(true, Ordering::SeqCst);

    let failed = e.sync(NOW).await;

    assert_eq!(failed.failed, [("HD-2".to_owned(), "model error: cuda out of memory".to_owned())]);
    assert_eq!((failed.indexed, failed.scanned), (2, 3));
    assert_eq!(e.meta("jira/watermark_ms"), Some(before), "kept, so the next run scans the failed issue again");
    assert_eq!(e.fingerprint("HD-2"), None);

    e.poisoned.store(false, Ordering::SeqCst);
    let clean = e.sync(NOW + 1).await;

    assert_eq!((clean.indexed, clean.failed.len()), (1, 0), "{clean:?}");
    assert_eq!(e.fingerprint("HD-2"), Some(at(20)));
    assert_eq!(e.meta("jira/watermark_ms"), Some(updated_ms(&at(30)).to_string()));
}

fn other_projects_point() -> store::Doc {
    let text = "[XX-9 · Task · Open] another project";
    let Value::Object(payload) = json!({"kind": "issue", "source": "jira", "project": "XX", "issue_key": "XX-9", "text": text, "ts": 1}) else { unreachable!() };
    store::Doc { id: store::point_id("jira", "XX-9/i0"), embedding: HashEmbedder::new(DIM).embed_one(text).unwrap(), payload }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_daily_reconcile_removes_deleted_and_moved_issues_and_keeps_the_rest() {
    let mut issues = spaced(5);
    issues[1].comments = vec!["a comment on HD-2".into()];
    let e = env(Mock { issues, cap: 50, ..Mock::default() }).await;
    e.store.upsert(vec![other_projects_point()]).await.unwrap();

    let first = e.sync(NOW).await;
    assert_eq!((first.indexed, first.removed), (5, 0));
    assert_eq!(e.listings(), [s("project in (HD) ORDER BY key ASC".into(), "0")], "the first run reconciles");
    assert_eq!(e.issue_gets(), Vec::<String>::new(), "every indexed key was listed");
    assert_eq!(e.meta("jira/reconciled_ms"), Some(NOW.to_string()));
    assert_eq!(e.points("HD-2").await, 2, "precondition: HD-2 has an issue and a comment chunk");
    {
        let mut m = e.mock.lock().unwrap();
        m.listing = Some(vec!["HD-1".into()]);
        m.gets.insert("HD-2".into(), IssueGet::NotFound);
        m.gets.insert("HD-3".into(), IssueGet::Moved("OPS-3"));
        m.gets.insert("HD-4".into(), IssueGet::Fails);
        // HD-5 is left out of the listing but still answers 200 under its own key.
    }

    let early = e.sync(NOW + DAY_MS - 1).await;
    assert_eq!(early.removed, 0);
    assert_eq!(e.listings(), Vec::<(String, String)>::new(), "not again within 24 h");

    let r = e.sync(NOW + DAY_MS).await;

    assert_eq!(r.removed, 2, "{r:?}");
    assert_eq!(e.listings(), [s("project in (HD) ORDER BY key ASC".into(), "0")]);
    assert_eq!(e.issue_gets(), ["HD-2", "HD-3", "HD-4", "HD-5"], "only unlisted keys are checked, XX-9 never");
    for (key, points, recorded) in [("HD-1", 1, true), ("HD-2", 0, false), ("HD-3", 0, false), ("HD-4", 1, true), ("HD-5", 1, true), ("XX-9", 1, false)] {
        assert_eq!(e.points(key).await, points, "{key}");
        assert_eq!(e.fingerprint(key).is_some(), recorded, "{key} state row");
    }
    assert_eq!(e.meta("jira/reconciled_ms"), Some((NOW + DAY_MS).to_string()));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_listing_error_fails_the_sync_and_deletes_nothing() {
    let e = env(Mock { issues: spaced(2), cap: 50, ..Mock::default() }).await;
    e.sync(NOW).await;
    {
        let mut m = e.mock.lock().unwrap();
        m.listing_reply = Some(json!({"issues": []}));
        m.gets.insert("HD-1".into(), IssueGet::NotFound);
        m.gets.insert("HD-2".into(), IssueGet::NotFound);
    }

    let err = e.try_sync(NOW + DAY_MS).await.unwrap_err();

    assert_eq!(err.to_string(), "source: Jira response could not be parsed: search reply without total");
    assert_eq!((e.points("HD-1").await, e.points("HD-2").await), (1, 1));
    assert_eq!(e.issue_gets(), Vec::<String>::new());
    assert_eq!(e.meta("jira/reconciled_ms"), Some(NOW.to_string()), "not marked done, so the next run tries again");
}

async fn all_keys(mock: Mock) -> (Result<Vec<String>, String>, Vec<(String, String)>) {
    let (url, shared) = serve(mock).await;
    let r = tokio::task::spawn_blocking(move || Jira::new(&url, "tok").all_keys("project in (HD) ORDER BY key ASC").map_err(|e| e.to_string())).await.unwrap();
    let listings = std::mem::take(&mut shared.lock().unwrap().listings);
    (r, listings)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn all_keys_walks_every_page() {
    let keys: Vec<String> = (1..=5).map(|i| format!("HD-{i}")).collect();
    let (r, listings) = all_keys(Mock { listing: Some(keys.clone()), cap: 2, ..Mock::default() }).await;
    assert_eq!(r, Ok(keys));
    let starts: Vec<&str> = listings.iter().map(|(_, s)| s.as_str()).collect();
    assert_eq!(starts, ["0", "2", "4"]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn all_keys_is_strict_about_the_reply() {
    let rows = [
        ("no total", json!({"issues": [{"key": "HD-1"}]}), "Jira response could not be parsed: search reply without total"),
        ("no issues", json!({"total": 1}), "Jira response could not be parsed: search reply without issues"),
        ("an issue without key", json!({"total": 2, "issues": [{"key": "HD-1"}, {"fields": {}}]}), "Jira response could not be parsed: issue without a key"),
        ("an empty key", json!({"total": 1, "issues": [{"key": ""}]}), "Jira response could not be parsed: issue without a key"),
    ];
    for (name, reply, want) in rows {
        let (r, _) = all_keys(Mock { listing_reply: Some(reply), cap: 50, ..Mock::default() }).await;
        assert_eq!(r, Err(want.to_owned()), "{name}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn all_keys_fails_when_the_listing_ends_short_of_total() {
    let keys: Vec<String> = (1..=3).map(|i| format!("HD-{i}")).collect();
    let (r, _) = all_keys(Mock { listing: Some(keys), listing_total: Some(5), cap: 2, ..Mock::default() }).await;
    assert_eq!(r, Err("Jira response could not be parsed: issue listing ended after 3 of 5".to_owned()));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_full_rescan_after_prune_older_than_does_not_bring_the_old_issue_back() {
    let mut old = issue("HD-1", "2025-03-01T10:00:00.000+0000");
    old.comments = vec!["an old comment".into()];
    let e = env(Mock { issues: vec![old, issue("HD-2", "2026-10-01T10:00:00.000+0000")], cap: 50, ..Mock::default() }).await;
    e.state().ensure_version("chunker-3/m1").unwrap();
    e.sync(NOW).await;
    assert_eq!((e.points("HD-1").await, e.points("HD-2").await), (2, 1), "precondition");

    let cutoff = jiff::civil::date(2026, 1, 1).to_zoned(jiff::tz::TimeZone::UTC).unwrap().timestamp().as_millisecond();
    ctx::prune::prune_older_than(&e.store, &e.state_path, "issue_key", cutoff, true).await.unwrap();
    // A version change drops the watermark: the next run reads every issue again.
    assert!(e.state().ensure_version("chunker-3/m2").unwrap());
    let r = e.sync(NOW + 1).await;

    assert_eq!(e.searches().last().unwrap().0, jql(None), "a full rescan");
    assert_eq!((r.scanned, r.failed.len()), (2, 0), "{r:?}");
    assert_eq!(e.points("HD-1").await, 0, "the pruned issue stays out");
    assert_eq!(e.points("HD-2").await, 1);
}

impl Env {
    /// `ctx::run_jira_index` with Jira configured at the mock and HD mapped.
    async fn run_jira_index(&self, model: &str) -> Report {
        let mapping = self._tmp.path().join("jira-projects.json");
        std::fs::write(&mapping, r#"{"work/desk": ["HD"]}"#).unwrap();
        let cfg = ctx::config::Config {
            jira_url: Some(self.url.clone()),
            jira_token: Some("tok".into()),
            jira_projects: mapping,
            jira_image_describe: false,
            ..ctx::config::Config::from_env(|_| None)
        };
        let mut emb = HashEmbedder::new(DIM);
        ctx::run_jira_index(&cfg, &self.store, &mut emb, model, &WordCounter, &self.state(), &ingest::Redactor::default()).await.unwrap()
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn run_jira_index_re_embeds_a_point_of_the_previous_model_and_records_the_new_version() {
    let e = env(Mock { issues: spaced(1), cap: 50, ..Mock::default() }).await;
    let (old, new) = (ctx::index_version("m1"), ctx::index_version("m2"));
    // An unmapped project's issue: neither the sync nor the reconcile touch it.
    let mut seeded = other_projects_point();
    seeded.payload.insert(ingest::VERSION_FIELD.into(), json!(old));
    e.store.upsert(vec![seeded]).await.unwrap();
    e.state().ensure_version(&old).unwrap();
    e.state().set_meta("index_version/jira", &old).unwrap();

    let r = e.run_jira_index("m2").await;

    assert_eq!((r.indexed, r.failed.len()), (1, 0), "{r:?}");
    let hits = e.store.scroll(&SearchFilter::default().eq("issue_key", "XX-9"), 10).await.unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].payload[ingest::VERSION_FIELD], json!(new));
    assert_eq!(e.meta("index_version/jira"), Some(new));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn run_jira_index_honours_the_recorded_prune_cutoff() {
    let old = issue("HD-1", "2025-03-01T10:00:00.000+0000");
    let e = env(Mock { issues: vec![old, issue("HD-2", "2026-10-01T10:00:00.000+0000")], cap: 50, ..Mock::default() }).await;
    let cutoff = jiff::civil::date(2026, 1, 1).to_zoned(jiff::tz::TimeZone::UTC).unwrap().timestamp().as_millisecond();
    e.state().set_meta(&ctx::prune::cutoff_key("jira"), &cutoff.to_string()).unwrap();

    let r = e.run_jira_index("m1").await;

    assert_eq!((r.scanned, r.failed.len()), (2, 0), "{r:?}");
    assert_eq!(e.points("HD-1").await, 0, "edited before the cutoff: not written");
    assert_eq!(e.points("HD-2").await, 1);
}
