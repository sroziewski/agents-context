//! Jira client against a mock HTTP server: the Bearer header, the endpoints, parsing and the
//! auth-error path — no live Jira.

mod common;

use std::sync::{Arc, Mutex};

use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::routing::get;
use axum::{Json, Router};
use ctx::jira::{Jira, JiraError};
use serde_json::json;

#[derive(Clone, Default)]
struct Seen {
    auth: Arc<Mutex<Vec<String>>>,
}

async fn serve(fail_auth: bool) -> (String, Seen) {
    let seen = Seen::default();
    let issue = |headers: HeaderMap, State(s): State<Seen>, Path(key): Path<String>| async move {
        s.auth.lock().unwrap().push(headers.get("authorization").and_then(|h| h.to_str().ok()).unwrap_or("").to_owned());
        Json(json!({
            "key": key,
            "fields": {
                "summary": "Login fails",
                "description": "detail",
                "status": {"name": "Open"},
                "issuetype": {"name": "Bug"},
                "assignee": {"displayName": "Anna"},
                "updated": "2026-09-27T10:00:00.000+0200",
                "comment": {"comments": [{"author": {"displayName": "Bob"}, "created": "2026-09-27T11:00:00Z", "body": "hi"}]}
            }
        }))
    };
    let search = |State(s): State<Seen>| async move {
        let _ = &s;
        Json(json!({"total": 1, "issues": [{"key": "HD-1", "fields": {"summary": "s", "status": {"name": "Open"}, "issuetype": {"name": "Task"}, "updated": "2026-09-27T10:00:00.000+0200"}}]}))
    };
    let app: Router = if fail_auth {
        Router::new().route("/rest/api/2/issue/{key}", get(|| async { (axum::http::StatusCode::UNAUTHORIZED, "nope") }))
    } else {
        Router::new()
            .route("/rest/api/2/issue/{key}", get(issue))
            .route("/rest/api/2/search", get(search))
            .with_state(seen.clone())
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (url, seen)
}

#[tokio::test(flavor = "multi_thread")]
async fn issue_and_search_parse_and_send_the_bearer_token() {
    let (url, seen) = serve(false).await;
    let (issue, issues) = tokio::task::spawn_blocking(move || {
        let j = Jira::new(&url, "tok123");
        (j.issue("HD-1").unwrap(), j.search("project = HD", 5).unwrap())
    })
    .await
    .unwrap();
    assert_eq!(issue.key, "HD-1");
    assert_eq!(issue.summary, "Login fails");
    assert_eq!(issue.comments.len(), 1);
    assert_eq!(issues.len(), 1);
    assert_eq!(issues[0].key, "HD-1");
    assert_eq!(*seen.auth.lock().unwrap(), ["Bearer tok123"], "the token is sent as a Bearer header");
}

/// A server counting POSTs to the comment endpoint, to prove writes are/aren't sent.
async fn serve_writes() -> (String, Arc<Mutex<u32>>) {
    let posts = Arc::new(Mutex::new(0u32));
    let p = posts.clone();
    let comment = move |State(s): State<Arc<Mutex<u32>>>| {
        let s = s.clone();
        async move {
            *s.lock().unwrap() += 1;
            (axum::http::StatusCode::CREATED, Json(json!({"id": "1"})))
        }
    };
    let app: Router = Router::new().route("/rest/api/2/issue/{key}/comment", axum::routing::post(comment)).with_state(p);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (url, posts)
}

fn search(url: &str, write_allowed: bool) -> ctx::jira::JiraSearch {
    let tmp = std::env::temp_dir().join(format!("ctx-jira-test-{}", std::process::id()));
    ctx::jira::JiraSearch {
        jira: Jira::new(url, "tok"),
        projects: tmp.join("jira-projects.json"),
        redact: tmp.join("redact.txt"),
        write_allowed,
        write_log: tmp.join("jira-writes.log"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn writes_need_both_the_config_lock_and_confirm() {
    let (url, posts) = serve_writes().await;
    let sent = || *posts.lock().unwrap();

    // Config lock off: refused even with confirm, nothing sent.
    let u = url.clone();
    let off = tokio::task::spawn_blocking(move || search(&u, false).comment("HD-1", "hi", true)).await.unwrap();
    assert!(off.is_err() && off.unwrap_err().to_string().contains("JIRA_WRITE"));
    assert_eq!(sent(), 0, "nothing sent when the config lock is off");

    // Config on but no confirm: dry run, still nothing sent.
    let u2 = url.clone();
    let dry = tokio::task::spawn_blocking(move || search(&u2, true).comment("HD-1", "hi", false)).await.unwrap().unwrap();
    assert!(dry.contains("DRY RUN"));
    assert_eq!(sent(), 0, "dry run sends nothing");

    // Both locks satisfied: the POST is sent.
    let u3 = url.clone();
    let done = tokio::task::spawn_blocking(move || search(&u3, true).comment("HD-1", "hi", true)).await.unwrap().unwrap();
    assert!(done.contains("Comment added"));
    assert_eq!(sent(), 1, "sent exactly once when allowed and confirmed");
}

/// A mock Jira /search returning a fixed set of issues (ignoring the JQL), for the indexer.
async fn serve_search(updated: &'static str) -> String {
    let search = move || async move {
        Json(json!({
            "total": 1,
            "issues": [{
                "key": "HD-1",
                "fields": {"summary": "Login fails", "description": "detail", "status": {"name": "Open"},
                    "issuetype": {"name": "Bug"}, "assignee": {"displayName": "Anna"}, "updated": updated,
                    "comment": {"comments": []}}
            }]
        }))
    };
    let app: Router = Router::new().route("/rest/api/2/search", get(search));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    url
}

#[tokio::test(flavor = "multi_thread")]
async fn sync_jira_indexes_then_skips_unchanged_and_reindexes_changed() {
    use ingest::{IndexConfig, State};
    let qdrant = testkit::qdrant().await;
    let store = store::Store::connect(&qdrant.url, None, "jira", 64).unwrap();
    store.ensure_collection(ingest::jira::JIRA_FIELDS).await.unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let state = State::open(&tmp.path().join("state.db")).unwrap();
    let keys = vec!["HD".to_string()];
    let cfg = IndexConfig::default();

    let run = |url: String| {
        let store = &store;
        let state = &state;
        let keys = &keys;
        let cfg = &cfg;
        async move {
            let mut emb = embed::HashEmbedder::new(64);
            let client = Jira::new(&url, "tok");
            ingest::jira::sync_jira(store, &mut emb, &ingest::WordCounter, state, &client, keys, cfg, None, 0).await.unwrap()
        }
    };

    let first = run(serve_search("2026-09-27T10:00:00.000+0200").await).await;
    assert_eq!((first.indexed, first.unchanged), (1, 0), "first run indexes the issue");
    assert!(first.chunks >= 1);

    let second = run(serve_search("2026-09-27T10:00:00.000+0200").await).await;
    assert_eq!((second.indexed, second.unchanged), (0, 1), "same `updated` → skipped, not re-embedded");

    let third = run(serve_search("2026-09-28T09:00:00.000+0200").await).await;
    assert_eq!((third.indexed, third.unchanged), (1, 0), "changed `updated` → re-indexed");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_bare_number_fetches_the_current_projects_issue() {
    let (url, _) = serve(false).await;
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("jira-projects.json"), r#"{"sedn/sedn26": ["SEDN26"], "helpdesk": ["HD"]}"#).unwrap();
    let js = ctx::jira::JiraSearch { projects: tmp.path().join("jira-projects.json"), ..search(&url, false) };
    let sedn = sources::project::key("d:/dev/sedn/sedn26");
    let unmapped = sources::project::key("d:/dev/other");

    let (searched, fetched, kept, unscoped) = tokio::task::spawn_blocking(move || {
        (
            js.search("1910", Some(&sedn), false, false, 8).unwrap(),
            js.issue_in("#1910", &sedn).unwrap(),
            js.issue_in("HD-5", &sedn).unwrap(),
            js.search("1910", Some(&unmapped), false, false, 8).unwrap(),
        )
    })
    .await
    .unwrap();
    assert!(searched.contains("SEDN26-1910") && searched.contains("Login fails"), "search fetches the issue itself:\n{searched}");
    assert!(fetched.contains("SEDN26-1910"), "{fetched}");
    assert!(kept.contains("HD-5") && !kept.contains("SEDN26"), "a full key is used as given:\n{kept}");
    assert!(unscoped.contains("HD-1") && !unscoped.contains("SEDN26"), "no mapping → plain text search:\n{unscoped}");
}

#[test]
fn server_instructions_name_the_projects_jira_key() {
    let one = ctx::mcp::jira_instructions(&["SEDN26".into()]);
    assert!(one.contains("SEDN26-1910") && one.contains("get_jira_issue"), "{one}");
    let many = ctx::mcp::jira_instructions(&["HD".into(), "SEDN26".into()]);
    assert!(many.contains("HD, SEDN26") && many.contains("ask which"), "{many}");
    assert_eq!(ctx::mcp::jira_instructions(&[]), "");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_401_maps_to_the_auth_error() {
    let (url, _) = serve(true).await;
    let err = tokio::task::spawn_blocking(move || Jira::new(&url, "bad").issue("HD-1")).await.unwrap().unwrap_err();
    assert!(matches!(err, JiraError::Auth), "{err}");
    assert!(err.to_string().contains("JIRA_TOKEN"));
}

use common::jira_mock::{self, IssueReply, JiraMock, issue_text, search_text};
use ctx::jira::JiraSearch;

struct Fx {
    mock: JiraMock,
    js: JiraSearch,
    _tmp: tempfile::TempDir,
}

/// A recording mock Jira and a `JiraSearch` on it. One project maps to exactly one Jira key,
/// one to two, and `unmapped()` to none.
async fn fx(issue: IssueReply, write_allowed: bool) -> Fx {
    let mock = jira_mock::serve(issue).await;
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("jira-projects.json"), r#"{"sedn/sedn26": ["SEDN26"], "shared/mono": ["AA", "BB"], "legacy/desk": ["hd"], "broken/map": ["A/B"]}"#).unwrap();
    let js = JiraSearch {
        jira: Jira::new(&mock.url, "tok"),
        projects: tmp.path().join("jira-projects.json"),
        redact: tmp.path().join("redact.txt"),
        write_allowed,
        write_log: tmp.path().join("jira-writes.log"),
    };
    Fx { mock, js, _tmp: tmp }
}

fn one_key() -> String {
    sources::project::key("d:/dev/sedn/sedn26")
}

fn two_keys() -> String {
    sources::project::key("d:/dev/shared/mono")
}

fn unmapped() -> String {
    sources::project::key("d:/dev/other")
}

/// Mapped to the one key `hd`, written in lower case in the mapping file.
fn lower_case_key() -> String {
    sources::project::key("d:/dev/legacy/desk")
}

/// Mapped to the one value `A/B`, which cannot be part of an issue key.
fn malformed_key() -> String {
    sources::project::key("d:/dev/broken/map")
}

/// What one call returned and what it made the mock receive.
#[derive(Debug, PartialEq)]
struct Sent {
    result: Result<String, String>,
    requests: Vec<String>,
    bodies: Vec<serde_json::Value>,
}

fn observe(mock: &JiraMock, call: impl FnOnce() -> anyhow::Result<String>) -> Sent {
    let result = call().map_err(|e| e.to_string());
    Sent { result, requests: mock.take_requests(), bodies: mock.take_bodies() }
}

fn nothing_sent(error: String) -> Sent {
    Sent { result: Err(error), requests: vec![], bodies: vec![] }
}

async fn blocking<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    tokio::task::spawn_blocking(f).await.unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_key_or_a_resolvable_number_reaches_jira_as_a_well_formed_key() {
    let f = fx(IssueReply::Found, true).await;
    let rows = [
        ("hd-12", unmapped(), "HD-12"),
        ("  Sedn26-7 ", unmapped(), "SEDN26-7"),
        ("1910", one_key(), "SEDN26-1910"),
        ("#1911", one_key(), "SEDN26-1911"),
        ("HD-5", one_key(), "HD-5"),
        ("aa-3", two_keys(), "AA-3"),
    ];
    for (input, project, key) in rows {
        let (js, mock) = (f.js.clone(), f.mock.clone());
        let (issue, comment, dry_comment, transition, dry_transition) = blocking(move || {
            (
                observe(&mock, || js.issue_in(input, &project)),
                observe(&mock, || js.comment_in(input, &project, "deployed to staging", true)),
                observe(&mock, || js.comment_in(input, &project, "deployed to staging", false)),
                observe(&mock, || js.transition_in(input, &project, " in review ", true)),
                observe(&mock, || js.transition_in(input, &project, "DONE", false)),
            )
        })
        .await;
        let path = format!("/rest/api/2/issue/{key}");
        assert_eq!(issue, Sent { result: Ok(issue_text(&f.mock.url, key)), requests: vec![format!("GET {path}")], bodies: vec![] }, "issue_in({input:?})");
        assert_eq!(
            comment,
            Sent { result: Ok(format!("Comment added to {key}.")), requests: vec![format!("POST {path}/comment")], bodies: vec![json!({"body": "deployed to staging"})] },
            "comment_in({input:?})"
        );
        assert_eq!(
            dry_comment,
            Sent {
                result: Ok(format!("DRY RUN — would POST a comment to {key}:\ndeployed to staging\n\n(pass the confirm/--write flag to actually send it)")),
                requests: vec![],
                bodies: vec![]
            },
            "comment_in({input:?}) without confirm"
        );
        assert_eq!(
            transition,
            Sent {
                result: Ok(format!("{key} transitioned to \"In Review\".")),
                requests: vec![format!("GET {path}/transitions"), format!("POST {path}/transitions")],
                bodies: vec![json!({"transition": {"id": "31"}})]
            },
            "transition_in({input:?})"
        );
        assert_eq!(
            dry_transition,
            Sent {
                result: Ok(format!("DRY RUN — would transition {key} to \"Done\" (id 41).\n(pass the confirm/--write flag to actually send it)")),
                requests: vec![format!("GET {path}/transitions")],
                bodies: vec![]
            },
            "transition_in({input:?}) without confirm reads the transitions but posts nothing"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn what_names_no_issue_in_the_project_is_refused_without_a_request() {
    let f = fx(IssueReply::Found, true).await;
    let rows = [
        ("12345", unmapped()),
        ("#1910", unmapped()),
        ("1910", two_keys()),
        ("#1910", two_keys()),
        ("HD-1/../../x", unmapped()),
        ("HD-1?x=1", unmapped()),
        ("HD-1#c", unmapped()),
        ("HD 1", unmapped()),
        ("H-1", unmapped()),
        ("ABCDEFGHIJK-1", unmapped()),
        ("HD-1-2", unmapped()),
        ("hd-", unmapped()),
        ("", unmapped()),
        ("   ", unmapped()),
        ("", one_key()),
        ("#", one_key()),
        ("19 10", one_key()),
        ("-1910", one_key()),
        ("+1910", one_key()),
        ("1910.0", one_key()),
        ("1910/../x", one_key()),
        ("1910?x=1", one_key()),
        ("HD-1/../../x", one_key()),
        ("SEDN26-", one_key()),
    ];
    for (input, project) in rows {
        let (js, mock) = (f.js.clone(), f.mock.clone());
        let p = project.clone();
        let (issue, comment, transition) = blocking(move || {
            (
                observe(&mock, || js.issue_in(input, &p)),
                observe(&mock, || js.comment_in(input, &p, "deployed to staging", true)),
                observe(&mock, || js.transition_in(input, &p, "done", true)),
            )
        })
        .await;
        let refused = || nothing_sent(format!("{input:?} is not a Jira issue key; pass a full key such as HD-123 (a bare number works only where the current project maps to exactly one Jira project)"));
        assert_eq!(issue, refused(), "issue_in({input:?}) in {project}");
        assert_eq!(comment, refused(), "comment_in({input:?}) in {project}");
        assert_eq!(transition, refused(), "transition_in({input:?}) in {project}");
    }
    assert!(!f.js.write_log.exists(), "a refused write is not logged as sent");
}

#[tokio::test(flavor = "multi_thread")]
async fn the_keyed_calls_take_a_full_key_in_any_case_and_nothing_else() {
    let f = fx(IssueReply::Found, true).await;
    let (js, mock) = (f.js.clone(), f.mock.clone());
    let (issue, comment, transition) = blocking(move || {
        (observe(&mock, || js.issue("hd-3")), observe(&mock, || js.comment("hd-4", "note", true)), observe(&mock, || js.transition("hd-5", "Done", true)))
    })
    .await;
    assert_eq!(issue, Sent { result: Ok(issue_text(&f.mock.url, "HD-3")), requests: vec!["GET /rest/api/2/issue/HD-3".into()], bodies: vec![] });
    assert_eq!(comment, Sent { result: Ok("Comment added to HD-4.".into()), requests: vec!["POST /rest/api/2/issue/HD-4/comment".into()], bodies: vec![json!({"body": "note"})] });
    assert_eq!(
        transition,
        Sent {
            result: Ok("HD-5 transitioned to \"Done\".".into()),
            requests: vec!["GET /rest/api/2/issue/HD-5/transitions".into(), "POST /rest/api/2/issue/HD-5/transitions".into()],
            bodies: vec![json!({"transition": {"id": "41"}})]
        }
    );

    for input in ["1910", "#1910", "HD-1/../../x", "HD-1?x=1", "HD 1", "HD-1/comment", ""] {
        let (js, mock) = (f.js.clone(), f.mock.clone());
        let (issue, comment, dry_comment, transition) = blocking(move || {
            (
                observe(&mock, || js.issue(input)),
                observe(&mock, || js.comment(input, "note", true)),
                observe(&mock, || js.comment(input, "note", false)),
                observe(&mock, || js.transition(input, "Done", true)),
            )
        })
        .await;
        let refused = || nothing_sent(format!("{input:?} is not a Jira issue key; pass a full key such as HD-123"));
        assert_eq!(issue, refused(), "issue({input:?})");
        assert_eq!(comment, refused(), "comment({input:?})");
        assert_eq!(dry_comment, refused(), "comment({input:?}) without confirm");
        assert_eq!(transition, refused(), "transition({input:?})");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn with_the_config_lock_off_no_write_path_sends_anything() {
    let f = fx(IssueReply::Found, false).await;
    for (input, project) in [("HD-1", unmapped()), ("1910", one_key())] {
        for confirm in [true, false] {
            let (js, mock, p) = (f.js.clone(), f.mock.clone(), project.clone());
            let (comment, transition) = blocking(move || {
                (observe(&mock, || js.comment_in(input, &p, "deployed to staging", confirm)), observe(&mock, || js.transition_in(input, &p, "done", confirm)))
            })
            .await;
            let off = || nothing_sent("Jira writes are off; set JIRA_WRITE=on in the config .env to allow them (they still need an explicit confirm)".to_owned());
            assert_eq!(comment, off(), "comment_in({input:?}, confirm={confirm})");
            assert_eq!(transition, off(), "transition_in({input:?}, confirm={confirm})");
        }
    }
    // Reading stays allowed with the lock off.
    let (js, mock) = (f.js.clone(), f.mock.clone());
    let read = blocking(move || observe(&mock, || js.issue_in("1910", &one_key()))).await;
    assert_eq!(read, Sent { result: Ok(issue_text(&f.mock.url, "SEDN26-1910")), requests: vec!["GET /rest/api/2/issue/SEDN26-1910".into()], bodies: vec![] });
}

#[tokio::test(flavor = "multi_thread")]
async fn a_key_shaped_query_jira_does_not_know_becomes_a_text_search() {
    let f = fx(IssueReply::Status(404), false).await;
    let rows = [
        ("UTF-8", None, "UTF-8", "text ~ \"UTF-8\" ORDER BY updated DESC"),
        ("UTF-8", Some(one_key()), "UTF-8", "project in (SEDN26) AND text ~ \"UTF-8\" ORDER BY updated DESC"),
        ("424242", Some(one_key()), "SEDN26-424242", "project in (SEDN26) AND text ~ \"424242\" ORDER BY updated DESC"),
    ];
    for (query, project, tried, jql) in rows {
        let (js, mock) = (f.js.clone(), f.mock.clone());
        let p = project.clone();
        let (sent, searched) = blocking(move || {
            let sent = observe(&mock, || js.search(query, p.as_deref(), false, false, 8));
            (sent, mock.take_jql())
        })
        .await;
        assert_eq!(
            sent,
            Sent { result: Ok(search_text(&f.mock.url)), requests: vec![format!("GET /rest/api/2/issue/{tried}"), "GET /rest/api/2/search".into()], bodies: vec![] },
            "search({query:?}, {project:?})"
        );
        assert_eq!(searched, [jql], "search({query:?}, {project:?})");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_search_that_names_no_well_formed_key_sends_only_the_text_search() {
    let f = fx(IssueReply::Found, false).await;
    let rows = [
        ("lower-case text is prose", "hd-77", None, "text ~ \"hd-77\" ORDER BY updated DESC"),
        ("lower-case text is prose, in a mapped project too", "hd-77", Some(one_key()), "project in (SEDN26) AND text ~ \"hd-77\" ORDER BY updated DESC"),
        ("a number the mapping cannot turn into a key", "1910", Some(malformed_key()), "project in (A/B) AND text ~ \"1910\" ORDER BY updated DESC"),
    ];
    for (name, query, project, jql) in rows {
        let (js, mock) = (f.js.clone(), f.mock.clone());
        let (sent, searched) = blocking(move || {
            let sent = observe(&mock, || js.search(query, project.as_deref(), false, false, 8));
            (sent, mock.take_jql())
        })
        .await;
        assert_eq!(sent, Sent { result: Ok(search_text(&f.mock.url)), requests: vec!["GET /rest/api/2/search".into()], bodies: vec![] }, "{name}");
        assert_eq!(searched, [jql], "{name}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_key_from_the_mapping_is_normalised_before_it_reaches_an_issue_url() {
    let f = fx(IssueReply::Found, true).await;
    let (js, mock) = (f.js.clone(), f.mock.clone());
    let (searched, fetched, commented) = blocking(move || {
        let p = lower_case_key();
        (
            observe(&mock, || js.search("1910", Some(&p), false, false, 8)),
            observe(&mock, || js.issue_in("#1911", &p)),
            observe(&mock, || js.comment_in("1912", &p, "note", true)),
        )
    })
    .await;
    assert_eq!(searched, Sent { result: Ok(issue_text(&f.mock.url, "HD-1910")), requests: vec!["GET /rest/api/2/issue/HD-1910".into()], bodies: vec![] });
    assert_eq!(fetched, Sent { result: Ok(issue_text(&f.mock.url, "HD-1911")), requests: vec!["GET /rest/api/2/issue/HD-1911".into()], bodies: vec![] });
    assert_eq!(commented, Sent { result: Ok("Comment added to HD-1912.".into()), requests: vec!["POST /rest/api/2/issue/HD-1912/comment".into()], bodies: vec![json!({"body": "note"})] });

    let (js, mock) = (f.js.clone(), f.mock.clone());
    let (fetched, commented, moved) = blocking(move || {
        let p = malformed_key();
        (
            observe(&mock, || js.issue_in("1910", &p)),
            observe(&mock, || js.comment_in("1910", &p, "note", true)),
            observe(&mock, || js.transition_in("1910", &p, "done", true)),
        )
    })
    .await;
    let refused = || nothing_sent("\"A/B-1910\" is not a Jira issue key; pass a full key such as HD-123".to_owned());
    assert_eq!(fetched, refused());
    assert_eq!(commented, refused());
    assert_eq!(moved, refused());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_key_lookup_that_fails_for_another_reason_is_still_an_error() {
    for (status, error) in [
        (401, "Jira authentication failed (401/403); check JIRA_TOKEN in the config .env".to_owned()),
        (403, "Jira authentication failed (401/403); check JIRA_TOKEN in the config .env".to_owned()),
        (500, format!("Jira request failed: {}", ureq::Error::StatusCode(500))),
    ] {
        let f = fx(IssueReply::Status(status), false).await;
        let (js, mock) = (f.js.clone(), f.mock.clone());
        let (sent, searched) = blocking(move || {
            let sent = observe(&mock, || js.search("UTF-8", None, false, false, 8));
            (sent, mock.take_jql())
        })
        .await;
        assert_eq!(sent, Sent { result: Err(error), requests: vec!["GET /rest/api/2/issue/UTF-8".into()], bodies: vec![] }, "issue lookup answered {status}");
        assert_eq!(searched, Vec::<String>::new(), "no text search after a {status}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_query_that_is_an_existing_key_is_fetched_directly() {
    let f = fx(IssueReply::Found, false).await;
    let (js, mock) = (f.js.clone(), f.mock.clone());
    let (sent, searched) = blocking(move || {
        let sent = observe(&mock, || js.search("HD-77", Some(&one_key()), false, false, 8));
        (sent, mock.take_jql())
    })
    .await;
    assert_eq!(sent, Sent { result: Ok(issue_text(&f.mock.url, "HD-77")), requests: vec!["GET /rest/api/2/issue/HD-77".into()], bodies: vec![] });
    assert_eq!(searched, Vec::<String>::new());
}
