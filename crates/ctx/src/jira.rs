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
            // normalise, not key: a mapping name is not a working directory, and key() resolves a
            // relative name like "sedn26" against the git repo the process happens to run in.
            Ok(raw.into_iter().map(|(dir, keys)| (sources::project::normalise(&dir), keys)).collect())
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

/// The issue key `text` refers to: a full key as is, or a bare number (`1910`, `#1910`) completed
/// with the project's Jira key. A bare number is completed only when the project maps to exactly
/// one key; with several it is ambiguous and stays a text search.
pub fn resolve_issue_key(text: &str, keys: &[String]) -> Option<String> {
    if let Some(key) = detect_issue_key(text) {
        return Some(key);
    }
    let num = text.trim().trim_start_matches('#');
    match keys {
        [only] if !num.is_empty() && num.bytes().all(|b| b.is_ascii_digit()) => Some(format!("{only}-{num}")),
        _ => None,
    }
}

/// A full issue key in any case (`hd-12` is `HD-12`); `None` for anything that is not one. Used
/// where the caller names one issue, so nothing but a key ever reaches an issue URL.
pub fn normalize_issue_key(text: &str) -> Option<String> {
    detect_issue_key(&text.trim().to_ascii_uppercase())
}

fn checked_key(key: &str) -> anyhow::Result<String> {
    normalize_issue_key(key).ok_or_else(|| anyhow::anyhow!("{key:?} is not a Jira issue key; pass a full key such as HD-123"))
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
        let jql = if raw_jql {
            query.to_owned()
        } else {
            let keys = match (all, project_key) {
                (false, Some(p)) => self.keys_for(p)?,
                _ => Vec::new(),
            };
            if let Some(key) = resolve_issue_key(query, &keys).and_then(|key| normalize_issue_key(&key)) {
                match self.jira.issue(&key) {
                    Ok(issue) => return Ok(format_issue(&issue, &ingest::Redactor::from_file(&self.redact)?)),
                    Err(sources::jira::JiraError::NotFound(_)) => {}
                    Err(e) => return Err(anyhow::Error::msg(e)),
                }
            }
            build_jql(query, &keys)
        };
        let issues = self.jira.search(&jql, limit).map_err(anyhow::Error::msg)?;
        let redactor = ingest::Redactor::from_file(&self.redact)?;
        Ok(format_issues(&issues, &jql, &redactor))
    }

    /// The Jira keys mapped to `project_key` in `jira-projects.json`.
    pub fn keys_for(&self, project_key: &str) -> anyhow::Result<Vec<String>> {
        Ok(keys_for_project(project_key, &load_project_keys(&self.projects)?))
    }

    /// One issue by key, where a bare number is completed with `project_key`'s Jira key.
    pub fn issue_in(&self, key: &str, project_key: &str) -> anyhow::Result<String> {
        self.issue(&self.key_in(key, project_key)?)
    }

    /// The full key `key` names inside `project_key`: a key as is, or a bare number completed
    /// with the project's only Jira key. Anything else is an error, never a guess.
    fn key_in(&self, key: &str, project_key: &str) -> anyhow::Result<String> {
        if let Some(full) = normalize_issue_key(key) {
            return Ok(full);
        }
        resolve_issue_key(key, &self.keys_for(project_key)?).ok_or_else(|| {
            anyhow::anyhow!("{key:?} is not a Jira issue key; pass a full key such as HD-123 (a bare number works only where the current project maps to exactly one Jira project)")
        })
    }

    /// [`Self::comment`] on the issue `key` names inside `project_key` (see [`Self::issue_in`]).
    pub fn comment_in(&self, key: &str, project_key: &str, body: &str, confirm: bool) -> anyhow::Result<String> {
        self.comment(&self.key_in(key, project_key)?, body, confirm)
    }

    /// [`Self::transition`] on the issue `key` names inside `project_key`.
    pub fn transition_in(&self, key: &str, project_key: &str, to: &str, confirm: bool) -> anyhow::Result<String> {
        self.transition(&self.key_in(key, project_key)?, to, confirm)
    }

    /// One issue with description and comments, redacted.
    pub fn issue(&self, key: &str) -> anyhow::Result<String> {
        let key = &checked_key(key)?;
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
        let key = &checked_key(key)?;
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
        let key = &checked_key(key)?;
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
    fn relative_mapping_names_stay_apart_inside_a_git_repo() {
        // Tests run with the crate dir as cwd, inside this repo: exactly where key() collapsed
        // every relative name to "" and only one mapping entry survived.
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("m.json");
        std::fs::write(&path, r#"{"sedn26": ["SEDN26"], "helpdesk": ["HD"], "ai4sd": ["AI"]}"#).unwrap();
        let map = load_project_keys(&path).unwrap();
        assert_eq!(all_keys(&map), ["AI", "HD", "SEDN26"]);
        assert_eq!(keys_for_project("d:/dev/sedn/sedn26", &map), ["SEDN26"]);
        assert_eq!(keys_for_project("d:/dev/atlassian-gatewa-ai4sd", &map), ["AI"]);
    }

    #[test]
    fn a_bare_number_takes_the_projects_only_key() {
        let one = ["SEDN26".to_string()];
        assert_eq!(resolve_issue_key("1910", &one), Some("SEDN26-1910".into()));
        assert_eq!(resolve_issue_key(" #1910 ", &one), Some("SEDN26-1910".into()));
        assert_eq!(resolve_issue_key("HD-7", &one), Some("HD-7".into()), "a full key is kept, even from another project");
        assert_eq!(resolve_issue_key("1910", &[]), None, "unmapped project: no key to complete with");
        assert_eq!(resolve_issue_key("1910", &["HD".into(), "SEDN26".into()]), None, "two keys: ambiguous");
        assert_eq!(resolve_issue_key("19 10", &one), None);
        assert_eq!(resolve_issue_key("#", &one), None);
        assert_eq!(resolve_issue_key("login 1910", &one), None);
    }

    #[test]
    fn normalize_accepts_full_keys_in_any_case_and_nothing_else() {
        let rows: &[(&str, Option<&str>)] = &[
            ("HD-1410", Some("HD-1410")),
            ("hd-12", Some("HD-12")),
            ("Sedn26-42", Some("SEDN26-42")),
            ("  hd-7\t", Some("HD-7")),
            ("AB-0", Some("AB-0")),
            ("ABCDEFGHIJ-1", Some("ABCDEFGHIJ-1")),
            ("ABCDEFGHIJK-1", None),
            ("H-1", None),
            ("2HD-1", None),
            ("1910", None),
            ("#1910", None),
            ("-1910", None),
            ("HD-", None),
            ("HD", None),
            ("HD-1/../../x", None),
            ("HD-1?x=1", None),
            ("HD-1#frag", None),
            ("HD 1", None),
            ("HD -1", None),
            ("HD- 1", None),
            ("HD-1 extra", None),
            ("HD-1-2", None),
            ("HD--1", None),
            ("HD-1a", None),
            ("HD_X-1", None),
            ("HD-1\nHD-2", None),
            ("ÄB-1", None),
            ("HD-１", None),
            ("", None),
            ("   ", None),
        ];
        let wrong: Vec<String> = rows
            .iter()
            .filter(|(text, want)| normalize_issue_key(text).as_deref() != *want)
            .map(|(text, want)| format!("{text:?}: got {:?}, want {want:?}", normalize_issue_key(text)))
            .collect();
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }

    #[test]
    fn maps_project_dirs_to_keys_and_lists_all() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("jira-projects.json");
        std::fs::write(&file, r#"{"Helpdesk": ["HD"], "zsun-help-backend": ["HD"], "sedn26": ["SEDN26"]}"#).unwrap();
        let map = load_project_keys(&file).unwrap();
        let expected: HashMap<String, Vec<String>> = [
            ("helpdesk".to_string(), vec!["HD".to_string()]),
            ("zsun-help-backend".to_string(), vec!["HD".to_string()]),
            ("sedn26".to_string(), vec!["SEDN26".to_string()]),
        ]
        .into_iter()
        .collect();
        assert_eq!(map, expected, "mapping names are normalised, never resolved against the process cwd");
        assert_eq!(keys_for_project("d:/dev/helpdesk", &map), ["HD"]);
        assert_eq!(keys_for_project("d:/dev/sedn/sedn26", &map), ["SEDN26"]);
        assert!(keys_for_project("d:/dev/unrelated", &map).is_empty());
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
