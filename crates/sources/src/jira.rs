//! Read/write Jira Data Center client (REST v2, Bearer PAT). Lives in `sources` so both the
//! indexer (`ingest`) and the search layer (`ctx`) can use it. Writes here are unconditional —
//! the read-only-by-default policy is enforced by the caller (`ctx`).

use serde_json::Value;

/// A Jira failure, distinguishing auth from a missing issue from transport.
#[derive(Debug)]
pub enum JiraError {
    /// 401/403 — token missing, wrong, or lacking permission.
    Auth,
    /// 404 — no such issue/project.
    NotFound(String),
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
pub struct JiraAttachment {
    pub id: String,
    pub filename: String,
    pub mime_type: String,
    pub size: u64,
    /// Absolute URL to the raw bytes (Jira returns a full URL here); fetch with [`Jira::download`].
    pub content_url: String,
}

impl JiraAttachment {
    pub fn is_image(&self) -> bool {
        self.mime_type.to_ascii_lowercase().starts_with("image/")
    }
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
    pub comments: Vec<JiraComment>,
    pub attachments: Vec<JiraAttachment>,
}

/// One page of a search: the issues plus the total matching, for pagination.
#[derive(Debug, Clone)]
pub struct SearchPage {
    pub issues: Vec<JiraIssue>,
    pub total: u32,
}

/// Fields fetched when indexing (full text incl. description and comments), so a page has
/// everything and the indexer needs no per-issue follow-up call.
const FULL_FIELDS: &str = "summary,description,status,issuetype,assignee,updated,comment,attachment";
/// Fields for a quick list (no comments/description).
const LIST_FIELDS: &str = "summary,status,issuetype,assignee,updated";

#[derive(Clone)]
pub struct Jira {
    agent: ureq::Agent,
    base: String,
    token: String,
}

impl Jira {
    pub fn new(base: &str, token: &str) -> Self {
        let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(std::time::Duration::from_secs(30))).build().into();
        Self { agent, base: base.trim_end_matches('/').to_owned(), token: token.to_owned() }
    }

    pub fn base(&self) -> &str {
        &self.base
    }

    fn get_json(&self, path: &str) -> Result<Value> {
        let url = format!("{}{path}", self.base);
        match self.agent.get(&url).header("Authorization", format!("Bearer {}", self.token)).header("Accept", "application/json").call() {
            Ok(mut resp) => resp.body_mut().read_json().map_err(|e| JiraError::Parse(e.to_string())),
            Err(ureq::Error::StatusCode(401 | 403)) => Err(JiraError::Auth),
            Err(ureq::Error::StatusCode(404)) => Err(JiraError::NotFound(path.to_owned())),
            Err(e) => Err(JiraError::Http(e.to_string())),
        }
    }

    fn post_json(&self, path: &str, body: Value) -> Result<()> {
        let url = format!("{}{path}", self.base);
        match self.agent.post(&url).header("Authorization", format!("Bearer {}", self.token)).header("Accept", "application/json").send_json(&body) {
            Ok(_) => Ok(()),
            Err(ureq::Error::StatusCode(401 | 403)) => Err(JiraError::Auth),
            Err(ureq::Error::StatusCode(404)) => Err(JiraError::NotFound(path.to_owned())),
            Err(e) => Err(JiraError::Http(e.to_string())),
        }
    }

    /// One issue with description and comments.
    pub fn issue(&self, key: &str) -> Result<JiraIssue> {
        let v = self.get_json(&format!("/rest/api/2/issue/{key}?fields={FULL_FIELDS}"))?;
        Ok(self.parse_issue(&v))
    }

    /// Issues matching JQL — summary fields only (fast list, no comments).
    pub fn search(&self, jql: &str, limit: u32) -> Result<Vec<JiraIssue>> {
        Ok(self.search_page(jql, 0, limit, LIST_FIELDS)?.issues)
    }

    /// One page of full issues (description + comments) for indexing, from `start_at`.
    pub fn search_full(&self, jql: &str, start_at: u32, max_results: u32) -> Result<SearchPage> {
        self.search_page(jql, start_at, max_results, FULL_FIELDS)
    }

    fn search_page(&self, jql: &str, start_at: u32, max_results: u32, fields: &str) -> Result<SearchPage> {
        let path = format!("/rest/api/2/search?jql={}&startAt={start_at}&maxResults={max_results}&fields={fields}", url_encode(jql));
        let v = self.get_json(&path)?;
        let total = v.get("total").and_then(Value::as_u64).unwrap_or(0) as u32;
        let issues = v.get("issues").and_then(|i| i.as_array()).map(|arr| arr.iter().map(|i| self.parse_issue(i)).collect()).unwrap_or_default();
        Ok(SearchPage { issues, total })
    }

    /// Every issue key matching `jql`, across all pages. Strict where [`Self::search`] is lenient:
    /// a reply without `issues` or `total`, or a listing that ends short of `total`, is an error,
    /// because callers remove what is missing from this list.
    pub fn all_keys(&self, jql: &str) -> Result<Vec<String>> {
        const PAGE: usize = 1000;
        let mut keys: Vec<String> = Vec::new();
        loop {
            let path = format!("/rest/api/2/search?jql={}&startAt={}&maxResults={PAGE}&fields=key", url_encode(jql), keys.len());
            let v = self.get_json(&path)?;
            let total = v.get("total").and_then(Value::as_u64).ok_or_else(|| JiraError::Parse("search reply without total".into()))?;
            let issues = v.get("issues").and_then(Value::as_array).ok_or_else(|| JiraError::Parse("search reply without issues".into()))?;
            if issues.is_empty() {
                if (keys.len() as u64) < total {
                    return Err(JiraError::Parse(format!("issue listing ended after {} of {total}", keys.len())));
                }
                return Ok(keys);
            }
            for issue in issues {
                let key = issue.get("key").and_then(Value::as_str).filter(|k| !k.is_empty());
                keys.push(key.ok_or_else(|| JiraError::Parse("issue without a key".into()))?.to_owned());
            }
            if keys.len() as u64 >= total {
                return Ok(keys);
            }
        }
    }

    /// `(key, name)` for every visible project.
    pub fn projects(&self) -> Result<Vec<(String, String)>> {
        let v = self.get_json("/rest/api/2/project")?;
        Ok(v.as_array().map(|arr| arr.iter().map(|p| (string(p, "key"), string(p, "name"))).filter(|(k, _)| !k.is_empty()).collect()).unwrap_or_default())
    }

    /// Appends a comment. Write op — the policy gate lives in the caller.
    pub fn add_comment(&self, key: &str, body: &str) -> Result<()> {
        self.post_json(&format!("/rest/api/2/issue/{key}/comment"), serde_json::json!({ "body": body }))
    }

    /// Raw bytes of an attachment from its absolute `content` URL, capped at `max_bytes` (a larger
    /// body is refused without being fully read). Bearer PAT like every other call.
    pub fn download(&self, content_url: &str, max_bytes: usize) -> Result<Vec<u8>> {
        use std::io::Read;
        match self.agent.get(content_url).header("Authorization", format!("Bearer {}", self.token)).call() {
            Ok(mut resp) => {
                let mut buf = Vec::new();
                resp.body_mut()
                    .as_reader()
                    .take(max_bytes as u64 + 1)
                    .read_to_end(&mut buf)
                    .map_err(|e| JiraError::Http(e.to_string()))?;
                if buf.len() > max_bytes {
                    return Err(JiraError::Http(format!("attachment exceeds {max_bytes} bytes")));
                }
                Ok(buf)
            }
            Err(ureq::Error::StatusCode(401 | 403)) => Err(JiraError::Auth),
            Err(ureq::Error::StatusCode(404)) => Err(JiraError::NotFound(content_url.to_owned())),
            Err(e) => Err(JiraError::Http(e.to_string())),
        }
    }

    /// Available transitions for an issue as `(id, name)`.
    pub fn transitions(&self, key: &str) -> Result<Vec<(String, String)>> {
        let v = self.get_json(&format!("/rest/api/2/issue/{key}/transitions"))?;
        Ok(v.get("transitions")
            .and_then(|t| t.as_array())
            .map(|arr| arr.iter().map(|t| (string(t, "id"), t.get("to").map(|to| string(to, "name")).unwrap_or_else(|| string(t, "name")))).collect())
            .unwrap_or_default())
    }

    /// Applies a transition by id. Write op — gated by the caller.
    pub fn do_transition(&self, key: &str, transition_id: &str) -> Result<()> {
        self.post_json(&format!("/rest/api/2/issue/{key}/transitions"), serde_json::json!({ "transition": { "id": transition_id } }))
    }

    fn parse_issue(&self, v: &Value) -> JiraIssue {
        let key = string(v, "key");
        let f = v.get("fields").cloned().unwrap_or(Value::Null);
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
        let attachments = f
            .get("attachment")
            .and_then(|a| a.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|a| JiraAttachment {
                        id: string(a, "id"),
                        filename: string(a, "filename"),
                        mime_type: string(a, "mimeType"),
                        size: a.get("size").and_then(Value::as_u64).unwrap_or(0),
                        content_url: string(a, "content"),
                    })
                    .collect()
            })
            .unwrap_or_default();
        JiraIssue {
            url: format!("{}/browse/{key}", self.base),
            summary: string(&f, "summary"),
            description: f.get("description").and_then(|d| d.as_str()).unwrap_or_default().to_owned(),
            status: named("status"),
            issue_type: named("issuetype"),
            assignee: f.get("assignee").filter(|a| !a.is_null()).map(|a| string(a, "displayName")).unwrap_or_else(|| "Unassigned".to_owned()),
            updated: string(&f, "updated"),
            comments,
            attachments,
            key,
        }
    }
}

fn string(v: &Value, k: &str) -> String {
    v.get(k).and_then(|x| x.as_str()).unwrap_or_default().to_owned()
}

/// Minimal percent-encoding for a JQL query in a URL query string.
pub fn url_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 2);
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
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
                "comment": {"comments": [{"author": {"displayName": "Anna W"}, "created": "2026-09-27T11:00:00.000+0200", "body": "restarted"}]}
            }
        });
        let i = jira().parse_issue(&v);
        assert_eq!((i.key.as_str(), i.status.as_str(), i.issue_type.as_str(), i.assignee.as_str()), ("HD-1410", "In Progress", "Bug", "Szymon Roziewski"));
        assert_eq!(i.url, "https://jira.example.org/browse/HD-1410");
        assert_eq!(i.comments.len(), 1);
    }

    #[test]
    fn parses_attachments_and_detects_images() {
        let v = json!({
            "key": "HD-9", "fields": {
                "summary": "x", "status": {"name": "Open"},
                "attachment": [
                    {"id": "10001", "filename": "screen.png", "mimeType": "image/png", "size": 4533, "content": "https://jira.example.org/secure/attachment/10001/screen.png"},
                    {"id": "10002", "filename": "log.txt", "mimeType": "text/plain", "size": 20, "content": "https://jira.example.org/secure/attachment/10002/log.txt"}
                ]
            }
        });
        let i = jira().parse_issue(&v);
        assert_eq!(i.attachments.len(), 2);
        let png = &i.attachments[0];
        assert_eq!((png.id.as_str(), png.filename.as_str(), png.size), ("10001", "screen.png", 4533));
        assert!(png.content_url.ends_with("/10001/screen.png"));
        assert!(png.is_image(), "image/png is an image");
        assert!(!i.attachments[1].is_image(), "text/plain is not");
    }

    #[test]
    fn missing_attachment_field_is_empty() {
        let i = jira().parse_issue(&json!({"key": "HD-8", "fields": {"summary": "x", "status": {"name": "Open"}}}));
        assert!(i.attachments.is_empty());
    }

    #[test]
    fn null_and_missing_fields_degrade() {
        let i = jira().parse_issue(&json!({"key": "SEDN26-1", "fields": {"summary": "x", "status": {"name": "Open"}, "assignee": null}}));
        assert_eq!(i.assignee, "Unassigned");
        assert!(i.description.is_empty() && i.comments.is_empty() && i.issue_type.is_empty());
    }

    #[test]
    fn url_encoding_covers_jql_punctuation() {
        assert_eq!(url_encode("project = HD"), "project%20%3D%20HD");
    }
}
