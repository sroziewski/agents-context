//! Jira client against a mock HTTP server: the Bearer header, the endpoints, parsing and the
//! auth-error path — no live Jira.

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

#[tokio::test(flavor = "multi_thread")]
async fn a_401_maps_to_the_auth_error() {
    let (url, _) = serve(true).await;
    let err = tokio::task::spawn_blocking(move || Jira::new(&url, "bad").issue("HD-1")).await.unwrap().unwrap_err();
    assert!(matches!(err, JiraError::Auth), "{err}");
    assert!(err.to_string().contains("JIRA_TOKEN"));
}
