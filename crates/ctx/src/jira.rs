//! Read-only Jira client (Jira Data Center, REST v2) and the project→key mapping.
//!
//! Auth is a Bearer Personal Access Token from `JIRA_TOKEN`; the base URL is `JIRA_URL`. This
//! module never writes — writes live behind the DEV-38 gate. Nothing here logs the token.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

/// A Jira read failure, distinguishing auth from a missing issue from transport, so the CLI can
/// say something useful (e.g. "check JIRA_TOKEN").
#[derive(Debug)]
pub enum JiraError {
    /// 401/403 — the token is missing, wrong or lacks permission.
    Auth,
    /// 404 — no such issue/project.
    NotFound(String),
    /// Any other HTTP status or a transport error.
    Http(String),
    Parse(String),
}

impl std::fmt::Display for JiraError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auth => write!(f, "Jira authentication failed (401/403); check JIRA_TOKEN in the config .env"),
            Self::NotFound(what) => write!(f, "Jira: {what} not found (404)"),
            Self::Http(e) => write!(f, "Jira request failed: {e}"),
            Self::Parse(e) => write!(f, "Jira response could not be parsed: {e}"),
        }
    }
}

impl std::error::Error for JiraError {}

pub type Result<T> = std::result::Result<T, JiraError>;

#[derive(Debug, Clone, PartialEq)]
pub struct JiraComment {
    pub author: String,
    pub created: String,
    pub body: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct JiraIssue {
    pub key: String,
    pub summary: String,
    pub description: String,
    pub status: String,
    pub issue_type: String,
    pub assignee: String,
    pub updated: String,
    pub url: String,
    /// Empty for search results (only the summary fields are fetched there).
    pub comments: Vec<JiraComment>,
}

/// Read-only Jira Data Center client. Blocking `ureq`, like the other service clients.
pub struct Jira {
    agent: ureq::Agent,
    base: String,
    token: String,
}

impl Jira {
    pub fn new(base: &str, token: &str) -> Self {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(std::time::Duration::from_secs(20)))
            .build()
            .into();
        Self { agent, base: base.trim_end_matches('/').to_owned(), token: token.to_owned() }
    }

    fn get_json(&self, path: &str) -> Result<serde_json::Value> {
        let url = format!("{}{path}", self.base);
        match self.agent.get(&url).header("Authorization", format!("Bearer {}", self.token)).header("Accept", "application/json").call() {
            Ok(mut resp) => resp.body_mut().read_json().map_err(|e| JiraError::Parse(e.to_string())),
            Err(ureq::Error::StatusCode(401 | 403)) => Err(JiraError::Auth),
            Err(ureq::Error::StatusCode(404)) => Err(JiraError::NotFound(path.to_owned())),
            Err(e) => Err(JiraError::Http(e.to_string())),
        }
    }

    /// One issue with its description and comments.
    pub fn issue(&self, key: &str) -> Result<JiraIssue> {
        let v = self.get_json(&format!("/rest/api/2/issue/{key}?fields=summary,description,status,issuetype,assignee,updated,comment"))?;
        Ok(self.parse_issue(&v))
    }

    /// Issues matching a JQL query; summary fields only (no comments), newest changes first if
    /// the JQL does not order.
    pub fn search(&self, jql: &str, limit: u32) -> Result<Vec<JiraIssue>> {
        let q = url_encode(jql);
        let path = format!("/rest/api/2/search?jql={q}&maxResults={limit}&fields=summary,status,issuetype,assignee,updated");
        let v = self.get_json(&path)?;
        Ok(v.get("issues").and_then(|i| i.as_array()).map(|arr| arr.iter().map(|i| self.parse_issue(i)).collect()).unwrap_or_default())
    }

    /// `(key, name)` for every visible project.
    pub fn projects(&self) -> Result<Vec<(String, String)>> {
        let v = self.get_json("/rest/api/2/project")?;
        Ok(v.as_array()
            .map(|arr| arr.iter().map(|p| (string(p, "key"), string(p, "name"))).filter(|(k, _)| !k.is_empty()).collect())
            .unwrap_or_default())
    }

    fn parse_issue(&self, v: &serde_json::Value) -> JiraIssue {
        let key = string(v, "key");
        let f = v.get("fields").cloned().unwrap_or(serde_json::Value::Null);
        let named = |field: &str| f.get(field).map(|n| string(n, "name")).unwrap_or_default();
        let comments = f
            .pointer("/comment/comments")
            .and_then(|c| c.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|c| JiraComment {
                        author: c.get("author").map(|a| string(a, "displayName")).unwrap_or_default(),
                        created: string(c, "created"),
                        body: string(c, "body"),
                    })
                    .collect()
            })
            .unwrap_or_default();
        JiraIssue {
            url: format!("{}/browse/{key}", self.base),
            summary: string(&f, "summary"),
            // DC returns description as plain text/wiki markup (not Cloud's ADF object).
            description: f.get("description").and_then(|d| d.as_str()).unwrap_or_default().to_owned(),
            status: named("status"),
            issue_type: named("issuetype"),
            assignee: f.get("assignee").filter(|a| !a.is_null()).map(|a| string(a, "displayName")).unwrap_or_else(|| "Unassigned".to_owned()),
            updated: string(&f, "updated"),
            comments,
            key,
        }
    }
}

fn string(v: &serde_json::Value, k: &str) -> String {
    v.get(k).and_then(|x| x.as_str()).unwrap_or_default().to_owned()
}

/// Minimal percent-encoding for a JQL query in a URL query string.
fn url_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 2);
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Loads `jira-projects.json` (a `{ "project dir": ["JIRA-KEY", …] }` map), keyed by normalised
/// project key like the Teams mapping. Missing file → empty (integration simply unscoped).
pub fn load_project_keys(path: &Path) -> anyhow::Result<HashMap<String, Vec<String>>> {
    match std::fs::read_to_string(path) {
        Ok(text) => {
            let raw: HashMap<String, Vec<String>> = serde_json::from_str(text.trim_start_matches('\u{feff}'))?;
            Ok(raw.into_iter().map(|(dir, keys)| (sources::project::key(&dir), keys)).collect())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(HashMap::new()),
        Err(e) => Err(e.into()),
    }
}

/// The Jira keys mapped to `project_key`: the values of every mapping entry whose (normalised)
/// directory is a substring of the project, so `d:/dev/helpdesk/zsun-help-backend` matches a
/// `helpdesk` entry. Deduped, order-stable.
pub fn keys_for_project(project_key: &str, map: &HashMap<String, Vec<String>>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut entries: Vec<(&String, &Vec<String>)> = map.iter().collect();
    entries.sort_by_key(|(dir, _)| *dir);
    for (dir, keys) in entries {
        if !dir.is_empty() && project_key.contains(dir.as_str()) {
            for k in keys {
                if !out.contains(k) {
                    out.push(k.clone());
                }
            }
        }
    }
    out
}

/// A bare Jira issue key in `text` (e.g. `HD-123`), so a search query that is really an issue
/// reference can be fetched directly. Requires an uppercase project code and a number.
pub fn detect_issue_key(text: &str) -> Option<String> {
    let t = text.trim();
    let bytes = t.as_bytes();
    let dash = t.find('-')?;
    let (code, num) = (&t[..dash], &t[dash + 1..]);
    let code_ok = (2..=10).contains(&code.len()) && code.bytes().all(|b| b.is_ascii_uppercase() || b.is_ascii_digit()) && code.bytes().next().is_some_and(|b| b.is_ascii_uppercase());
    let num_ok = !num.is_empty() && num.bytes().all(|b| b.is_ascii_digit());
    let _ = bytes;
    (code_ok && num_ok).then(|| t.to_owned())
}

/// Deserialization sanity for a captured search page (used only in tests / future callers).
#[derive(Debug, Deserialize)]
pub struct SearchPage {
    #[serde(default)]
    pub total: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn jira() -> Jira {
        Jira::new("https://jira.example.org/", "tok")
    }

    #[test]
    fn parses_a_full_issue_with_comments() {
        let v = json!({
            "key": "HD-1410",
            "fields": {
                "summary": "Login fails after deploy",
                "description": "Users cannot log in on prod.",
                "status": {"name": "In Progress"},
                "issuetype": {"name": "Bug"},
                "assignee": {"displayName": "Szymon Roziewski"},
                "updated": "2026-09-27T10:00:00.000+0200",
                "comment": {"comments": [
                    {"author": {"displayName": "Anna W"}, "created": "2026-09-27T11:00:00.000+0200", "body": "restarted the service"}
                ]}
            }
        });
        let i = jira().parse_issue(&v);
        assert_eq!(i.key, "HD-1410");
        assert_eq!(i.summary, "Login fails after deploy");
        assert_eq!((i.status.as_str(), i.issue_type.as_str(), i.assignee.as_str()), ("In Progress", "Bug", "Szymon Roziewski"));
        assert_eq!(i.url, "https://jira.example.org/browse/HD-1410");
        assert_eq!(i.comments, [JiraComment { author: "Anna W".into(), created: "2026-09-27T11:00:00.000+0200".into(), body: "restarted the service".into() }]);
    }

    #[test]
    fn missing_or_null_fields_degrade_gracefully() {
        let i = jira().parse_issue(&json!({"key": "SEDN26-1", "fields": {"summary": "x", "status": {"name": "Open"}, "assignee": null}}));
        assert_eq!(i.assignee, "Unassigned");
        assert_eq!(i.issue_type, "");
        assert!(i.description.is_empty() && i.comments.is_empty());
    }

    #[test]
    fn detects_issue_keys_but_not_prose() {
        assert_eq!(detect_issue_key("HD-1410"), Some("HD-1410".into()));
        assert_eq!(detect_issue_key("  SEDN26-42 "), Some("SEDN26-42".into()));
        assert_eq!(detect_issue_key("how did we fix HD-1410"), None, "only a bare key");
        assert_eq!(detect_issue_key("lowercase-12"), None);
        assert_eq!(detect_issue_key("HD-"), None);
    }

    #[test]
    fn maps_project_dirs_to_keys_by_substring() {
        let map: HashMap<String, Vec<String>> = [
            (sources::project::key("helpdesk"), vec!["HD".to_string()]),
            (sources::project::key("zsun-help-backend"), vec!["HD".to_string()]),
            (sources::project::key("sedn26"), vec!["SEDN26".to_string()]),
        ]
        .into_iter()
        .collect();
        assert_eq!(keys_for_project(&sources::project::key("d:/dev/helpdesk"), &map), ["HD"]);
        assert_eq!(keys_for_project(&sources::project::key("d:/dev/sedn/sedn26"), &map), ["SEDN26"]);
        assert!(keys_for_project(&sources::project::key("d:/dev/unrelated"), &map).is_empty());
    }

    #[test]
    #[ignore = "hits the real Jira; run with JIRA_URL/JIRA_TOKEN set"]
    fn live_smoke() {
        let (Ok(url), Ok(token)) = (std::env::var("JIRA_URL"), std::env::var("JIRA_TOKEN")) else {
            return;
        };
        let j = Jira::new(&url, &token);
        let issues = j.search("project = HD ORDER BY updated DESC", 1).unwrap();
        assert!(!issues.is_empty(), "HD has issues");
        let full = j.issue(&issues[0].key).unwrap();
        assert_eq!(full.key, issues[0].key);
        assert!(!full.summary.is_empty());
    }

    #[test]
    fn url_encoding_covers_jql_punctuation() {
        assert_eq!(url_encode("project = HD ORDER BY updated"), "project%20%3D%20HD%20ORDER%20BY%20updated");
        assert_eq!(url_encode("text ~ \"db\""), "text%20~%20%22db%22");
    }
}
