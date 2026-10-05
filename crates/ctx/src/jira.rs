//! Jira search, formatting and the project→key mapping for `ctx`. The read/write client and the
//! types live in `sources::jira`; here we add JQL building, redacted formatting, project scoping
//! and the read-only-by-default write gate. Nothing here logs the token.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub use sources::jira::{Jira, JiraComment, JiraError, JiraIssue};

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

/// The Jira keys mapped to `project_key`: values of every entry whose (normalised) directory is
/// a substring of the project. Deduped, order-stable.
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

/// Every distinct Jira key across the whole mapping (what the indexer covers). Sorted.
pub fn all_keys(map: &HashMap<String, Vec<String>>) -> Vec<String> {
    let mut out: Vec<String> = map.values().flatten().cloned().collect();
    out.sort();
    out.dedup();
    out
}

/// A bare Jira issue key in `text` (e.g. `HD-123`), so a search query that is really an issue
/// reference can be fetched directly.
pub fn detect_issue_key(text: &str) -> Option<String> {
    let t = text.trim();
    let dash = t.find('-')?;
    let (code, num) = (&t[..dash], &t[dash + 1..]);
    let code_ok = (2..=10).contains(&code.len())
        && code.bytes().all(|b| b.is_ascii_uppercase() || b.is_ascii_digit())
        && code.bytes().next().is_some_and(|b| b.is_ascii_uppercase());
    let num_ok = !num.is_empty() && num.bytes().all(|b| b.is_ascii_digit());
    (code_ok && num_ok).then(|| t.to_owned())
}

/// Builds a JQL query from plain text. Scopes to `keys` (`project in (…)`) when any are given,
/// matches with `~`, and orders by most recently updated. Quotes are escaped so text cannot
/// break out of the JQL string literal.
pub fn build_jql(text: &str, keys: &[String]) -> String {
    let escaped = text.replace('\\', "\\\\").replace('"', "\\\"");
    let text_clause = format!("text ~ \"{escaped}\"");
    let scoped = if keys.is_empty() { text_clause } else { format!("project in ({}) AND {text_clause}", keys.join(", ")) };
    format!("{scoped} ORDER BY updated DESC")
}

/// Read-only Jira search + formatting plus the gated writes, redacted on the way out. The CLI
/// and MCP tools call this so query/scoping/redaction/gate logic lives in one place.
#[derive(Clone)]
pub struct JiraSearch {
    pub jira: Jira,
    /// `jira-projects.json`, for default project scoping.
    pub projects: PathBuf,
    /// The user's literal redaction list; issue/comment text is scrubbed before display.
    pub redact: PathBuf,
    /// Config lock (`JIRA_WRITE`): writes are refused unless true, whatever the caller asks.
    pub write_allowed: bool,
    /// Append-only local record of writes actually sent.
    pub write_log: PathBuf,
}

impl JiraSearch {
    /// Text or JQL search. A bare issue key (`HD-123`) is fetched directly. Scoped to
    /// `project_key`'s mapped Jira keys unless `all` or `raw_jql`.
    pub fn search(&self, query: &str, project_key: Option<&str>, raw_jql: bool, all: bool, limit: u32) -> anyhow::Result<String> {
        if !raw_jql && let Some(key) = detect_issue_key(query) {
            return self.issue(&key);
        }
        let jql = if raw_jql {
            query.to_owned()
        } else {
            let keys = match (all, project_key) {
                (false, Some(p)) => keys_for_project(p, &load_project_keys(&self.projects)?),
                _ => Vec::new(),
            };
            build_jql(query, &keys)
        };
        let issues = self.jira.search(&jql, limit).map_err(anyhow::Error::msg)?;
        let redactor = ingest::Redactor::from_file(&self.redact)?;
        Ok(format_issues(&issues, &jql, &redactor))
    }

    /// One issue with description and comments, redacted.
    pub fn issue(&self, key: &str) -> anyhow::Result<String> {
        let issue = self.jira.issue(key).map_err(anyhow::Error::msg)?;
        let redactor = ingest::Redactor::from_file(&self.redact)?;
        Ok(format_issue(&issue, &redactor))
    }

    fn require_write_allowed(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.write_allowed, "Jira writes are off; set JIRA_WRITE=on in the config .env to allow them (they still need an explicit confirm)");
        Ok(())
    }

    /// Adds a comment. Two locks: `JIRA_WRITE=on` (config) AND `confirm` (explicit intent).
    /// Without `confirm` it is a dry run.
    pub fn comment(&self, key: &str, body: &str, confirm: bool) -> anyhow::Result<String> {
        self.require_write_allowed()?;
        if !confirm {
            return Ok(format!("DRY RUN — would POST a comment to {key}:\n{body}\n\n(pass the confirm/--write flag to actually send it)"));
        }
        self.jira.add_comment(key, body).map_err(anyhow::Error::msg)?;
        self.log(&format!("comment {key}: {} chars", body.len()));
        Ok(format!("Comment added to {key}."))
    }

    /// Transitions an issue to the named status (case-insensitive), under the same two locks.
    pub fn transition(&self, key: &str, to: &str, confirm: bool) -> anyhow::Result<String> {
        self.require_write_allowed()?;
        let available = self.jira.transitions(key).map_err(anyhow::Error::msg)?;
        let Some((id, name)) = available.iter().find(|(_, name)| name.eq_ignore_ascii_case(to.trim())) else {
            let names: Vec<&str> = available.iter().map(|(_, n)| n.as_str()).collect();
            anyhow::bail!("no transition to {to:?} from {key}'s current status; available: {}", if names.is_empty() { "(none)".to_owned() } else { names.join(", ") });
        };
        if !confirm {
            return Ok(format!("DRY RUN — would transition {key} to \"{name}\" (id {id}).\n(pass the confirm/--write flag to actually send it)"));
        }
        self.jira.do_transition(key, id).map_err(anyhow::Error::msg)?;
        self.log(&format!("transition {key} -> {name}"));
        Ok(format!("{key} transitioned to \"{name}\"."))
    }

    fn log(&self, what: &str) {
        use std::io::Write;
        if let Some(dir) = self.write_log.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&self.write_log) {
            let _ = writeln!(f, "{}  {what}", jiff::Timestamp::now());
        }
    }
}

/// Numbered search results with key, status and URL, redacted.
pub fn format_issues(issues: &[JiraIssue], jql: &str, redactor: &ingest::Redactor) -> String {
    if issues.is_empty() {
        return format!("No Jira issues for: {jql}");
    }
    let scrub = |s: &str| redactor.scrub(s).0;
    issues
        .iter()
        .enumerate()
        .map(|(i, x)| {
            format!(
                "[{}] {} · {} · {} · {} · updated {}\n{}\n{}",
                i + 1,
                x.key,
                x.issue_type,
                x.status,
                x.assignee,
                x.updated.split('T').next().unwrap_or(&x.updated),
                scrub(&x.summary),
                x.url
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// One issue in full: header, description, then comments — all redacted.
pub fn format_issue(x: &JiraIssue, redactor: &ingest::Redactor) -> String {
    let scrub = |s: &str| redactor.scrub(s).0;
    let mut out = format!("{} · {} · {} · assignee {} · updated {}\n{}\n\n{}", x.key, x.issue_type, x.status, x.assignee, x.updated, x.url, scrub(&x.summary));
    if !x.description.trim().is_empty() {
        out.push_str(&format!("\n\n{}", scrub(x.description.trim())));
    }
    for c in &x.comments {
        out.push_str(&format!("\n\n--- {} · {} ---\n{}", c.author, c.created.split('T').next().unwrap_or(&c.created), scrub(c.body.trim())));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_issue_keys_but_not_prose() {
        assert_eq!(detect_issue_key("HD-1410"), Some("HD-1410".into()));
        assert_eq!(detect_issue_key("  SEDN26-42 "), Some("SEDN26-42".into()));
        assert_eq!(detect_issue_key("how did we fix HD-1410"), None);
        assert_eq!(detect_issue_key("lowercase-12"), None);
        assert_eq!(detect_issue_key("HD-"), None);
    }

    #[test]
    fn maps_project_dirs_to_keys_and_lists_all() {
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
        assert_eq!(all_keys(&map), ["HD", "SEDN26"]);
    }

    #[test]
    fn build_jql_scopes_and_escapes() {
        assert_eq!(build_jql("login fails", &[]), "text ~ \"login fails\" ORDER BY updated DESC");
        assert_eq!(build_jql("login", &["HD".into(), "SEDN26".into()]), "project in (HD, SEDN26) AND text ~ \"login\" ORDER BY updated DESC");
        assert!(build_jql("say \"hi\"", &[]).contains("text ~ \"say \\\"hi\\\"\""));
    }

    #[test]
    fn formats_issues_and_one_issue_redacted() {
        let red = ingest::Redactor::new(["s3cr3tp4ss".to_string()]);
        let base = JiraIssue {
            key: "HD-1".into(),
            summary: "leak s3cr3tp4ss here".into(),
            description: String::new(),
            status: "Open".into(),
            issue_type: "Bug".into(),
            assignee: "Anna".into(),
            updated: "2026-09-27T10:00:00.000+0200".into(),
            url: "https://j/browse/HD-1".into(),
            comments: vec![],
            attachments: vec![],
        };
        let out = format_issues(std::slice::from_ref(&base), "jql", &red);
        assert!(out.contains("[1] HD-1 · Bug · Open · Anna · updated 2026-09-27") && !out.contains("s3cr3tp4ss"), "{out}");
        assert_eq!(format_issues(&[], "project = X", &red), "No Jira issues for: project = X");
        let full = JiraIssue { description: "detail s3cr3tp4ss".into(), comments: vec![JiraComment { author: "Bob".into(), created: "2026-09-27T11:00:00Z".into(), body: "note s3cr3tp4ss".into() }], ..base };
        let one = format_issue(&full, &red);
        assert!(!one.contains("s3cr3tp4ss") && one.contains("--- Bob · 2026-09-27 ---"), "{one}");
    }

    #[test]
    #[ignore = "hits the real Jira; run with JIRA_URL/JIRA_TOKEN set"]
    fn live_smoke() {
        let (Ok(url), Ok(token)) = (std::env::var("JIRA_URL"), std::env::var("JIRA_TOKEN")) else {
            return;
        };
        let j = Jira::new(&url, &token);
        let issues = j.search("project = HD ORDER BY updated DESC", 1).unwrap();
        assert!(!issues.is_empty());
        assert_eq!(j.issue(&issues[0].key).unwrap().key, issues[0].key);
    }
}
