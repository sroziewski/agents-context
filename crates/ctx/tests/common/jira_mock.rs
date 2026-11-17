//! A mock Jira REST API that records every request it receives, so a test can assert exactly
//! which paths were asked for — and that nothing was sent at all.

use std::sync::{Arc, Mutex};

use axum::extract::State;
use axum::http::{Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::{Json, Router};
use serde_json::{Value, json};

/// How `GET /rest/api/2/issue/{key}` answers.
#[derive(Clone, Copy)]
pub enum IssueReply {
    Found,
    Status(u16),
}

#[derive(Clone)]
pub struct JiraMock {
    pub url: String,
    requests: Arc<Mutex<Vec<String>>>,
    jql: Arc<Mutex<Vec<String>>>,
    bodies: Arc<Mutex<Vec<Value>>>,
    issue: IssueReply,
}

impl JiraMock {
    /// `METHOD path` of every request since the last call, in order.
    pub fn take_requests(&self) -> Vec<String> {
        std::mem::take(&mut self.requests.lock().unwrap())
    }

    /// The decoded `jql` of every search since the last call.
    pub fn take_jql(&self) -> Vec<String> {
        std::mem::take(&mut self.jql.lock().unwrap())
    }

    /// The JSON body of every POST since the last call.
    pub fn take_bodies(&self) -> Vec<Value> {
        std::mem::take(&mut self.bodies.lock().unwrap())
    }
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let hex = (bytes[i] == b'%').then(|| s.get(i + 1..i + 3).and_then(|h| u8::from_str_radix(h, 16).ok())).flatten();
        match hex {
            Some(b) => {
                out.push(b);
                i += 3;
            }
            None => {
                out.push(bytes[i]);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

async fn handle(State(m): State<JiraMock>, method: Method, uri: Uri, body: String) -> Response {
    m.requests.lock().unwrap().push(format!("{method} {}", uri.path()));
    if method == Method::POST {
        m.bodies.lock().unwrap().push(serde_json::from_str(&body).unwrap_or(Value::Null));
    }
    let rest = uri.path().strip_prefix("/rest/api/2/").unwrap_or_default();
    let segments: Vec<&str> = rest.split('/').collect();
    match (method.as_str(), segments.as_slice()) {
        ("GET", ["search"]) => {
            let jql = uri.query().unwrap_or_default().split('&').find_map(|p| p.strip_prefix("jql=")).map(percent_decode).unwrap_or_default();
            m.jql.lock().unwrap().push(jql);
            Json(json!({"total": 1, "issues": [{"key": "TXT-9", "fields": {
                "summary": "Encoding breaks on import", "status": {"name": "In Progress"}, "issuetype": {"name": "Task"},
                "assignee": {"displayName": "Celina"}, "updated": "2026-10-03T08:30:00.000+0200"}}]}))
            .into_response()
        }
        ("GET", ["issue", key]) => match m.issue {
            IssueReply::Found => Json(json!({"key": key, "fields": {
                "summary": "Login fails", "description": "detail", "status": {"name": "Open"}, "issuetype": {"name": "Bug"},
                "assignee": {"displayName": "Anna"}, "updated": "2026-09-27T10:00:00.000+0200",
                "comment": {"comments": []}}}))
            .into_response(),
            IssueReply::Status(code) => StatusCode::from_u16(code).unwrap().into_response(),
        },
        ("GET", ["issue", _, "transitions"]) => Json(json!({"transitions": [
            {"id": "31", "name": "Send to review", "to": {"name": "In Review"}},
            {"id": "41", "name": "Close", "to": {"name": "Done"}}]}))
        .into_response(),
        ("POST", ["issue", _, "transitions"]) => StatusCode::NO_CONTENT.into_response(),
        ("POST", ["issue", _, "comment"]) => (StatusCode::CREATED, Json(json!({"id": "10001"}))).into_response(),
        _ => StatusCode::NOT_FOUND.into_response(),
    }
}

/// `ctx`'s rendering of the issue this mock serves for `key`.
pub fn issue_text(base: &str, key: &str) -> String {
    format!("{key} · Bug · Open · assignee Anna · updated 2026-09-27T10:00:00.000+0200\n{base}/browse/{key}\n\nLogin fails\n\ndetail")
}

/// `ctx`'s rendering of the one search result this mock serves.
pub fn search_text(base: &str) -> String {
    format!("[1] TXT-9 · Task · In Progress · Celina · updated 2026-10-03\nEncoding breaks on import\n{base}/browse/TXT-9")
}

pub async fn serve(issue: IssueReply) -> JiraMock {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let mock = JiraMock { url: format!("http://{}", listener.local_addr().unwrap()), requests: Arc::default(), jql: Arc::default(), bodies: Arc::default(), issue };
    let app = Router::new().fallback(handle).with_state(mock.clone());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    mock
}
