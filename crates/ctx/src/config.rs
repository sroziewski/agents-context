use std::path::PathBuf;
use std::time::Duration;

/// Runtime configuration from environment variables, after loading
/// `%APPDATA%\agents-context\.env` if it exists. That file matters for the MCP server: Claude
/// Code and OpenCode start it from arbitrary project directories with a minimal environment.
#[derive(Debug, Clone)]
pub struct Config {
    pub qdrant_url: String,
    pub qdrant_api_key: Option<String>,
    pub collection: String,
    pub embed_url: String,
    pub state_path: PathBuf,
    /// Local bge-m3 export; its `tokenizer.json` is also what chunking counts tokens with.
    pub model_dir: PathBuf,
    pub claude_dir: PathBuf,
    pub opencode_db: PathBuf,
    /// `CTX_SUMMARIES=off` disables session summaries.
    pub summaries: bool,
    pub summary_model: String,
    pub claude_exe: PathBuf,
    /// Working directory for `claude -p` summarizer runs.
    pub summarizer_dir: PathBuf,
    /// Literal secrets to redact everywhere, one per line (see `ctx redact add`).
    pub redact_file: PathBuf,
    /// Local working data (state DB, summarizer dir, pending memory suggestions).
    pub data_dir: PathBuf,
    /// The Teams app's store, read-only; Teams indexing is skipped while it does not exist.
    pub teams_db: PathBuf,
    /// The Teams app's who-is-who file (groups, roles, VIPs).
    pub teams_people: PathBuf,
    pub teams_collection: String,
    /// Qdrant collection for indexed Teams meeting transcripts, separate from channel threads so
    /// the two never leak into each other's search.
    pub meetings_collection: String,
    /// Project → Teams groups whose channels rank higher when searching from that project.
    pub teams_projects: PathBuf,
    /// Re-run the Teams scraper when its store goes stale (`CTX_TEAMS_SYNC_CMD` and friends). Off
    /// unless the command is set.
    pub teams_watch: crate::teams_watch::WatchConfig,
    pub recency: crate::rank::Recency,
    /// How session search ranks (`CTX_SEARCH_MODE`: dense, hybrid, sparse).
    pub search_mode: store::SearchMode,
    /// Lexical legs for identifier queries (`CTX_LEXICAL`: bm25, sparse+bm25, off).
    pub lexical: crate::lexical::Lexical,
    /// Deep search on or off (`CTX_RERANK`, default off). Off: the daemon never loads the
    /// reranker, MCP does not offer `deep_search_sessions`, `ctx search --rerank` searches
    /// normally. DEV-63 measured it at p95 ~7 s per query at a pool of 10 on CPU.
    pub rerank: bool,
    /// Deep-search reranker: which CPU model, its thread cap, and the candidate pool. Opt-in;
    /// normal search never loads it (DEV-28 found it ~16 s/query, too slow for the default).
    pub rerank_model: String,
    pub rerank_threads: Option<usize>,
    pub rerank_pool: u64,
    /// Where the reranker model is cached (HF-hub layout). Stable, independent of the process's
    /// working directory.
    pub rerank_cache_dir: PathBuf,
    pub inject: crate::inject::InjectConfig,
    /// After `/clear`, hand the cleared session over to the new one (`CTX_HANDOFF`, default on).
    pub handoff: bool,
    /// Status-line context thresholds in percent (`CTX_STATUS_WARN` / `CTX_STATUS_CRIT`).
    pub status_warn: f64,
    pub status_crit: f64,
    /// Model that fact-checks memory suggestions against the sessions they came from.
    pub verify_model: String,
    /// The retrieval eval's query set. It names real projects and topics, so it lives with the
    /// user's config rather than in the repository.
    pub eval_file: PathBuf,
    pub eval_baseline: PathBuf,
    /// Jira Data Center base URL and Personal Access Token (`JIRA_URL` / `JIRA_TOKEN`, in the
    /// config .env). The integration is off when either is unset.
    pub jira_url: Option<String>,
    pub jira_token: Option<String>,
    pub jira_projects: PathBuf,
    /// Writes to Jira are refused unless this is on (`JIRA_WRITE`); read-only by default.
    pub jira_write: bool,
    /// Qdrant collection for indexed Jira issues.
    pub jira_collection: String,
    /// Local times to index Jira at (`CTX_JIRA_INDEX_TIMES`, e.g. "10:00,17:00").
    pub jira_index_times: Vec<(u32, u32)>,
    /// Describe Jira image attachments with a vision model so screenshots are searchable
    /// (`CTX_JIRA_IMAGE_DESCRIBE`). Off by default: it sends images to the model and costs per
    /// image. `vision_model` picks the model; the caps bound cost per issue and per file.
    pub jira_image_describe: bool,
    pub vision_model: String,
    pub jira_image_max: usize,
    pub jira_image_max_bytes: usize,
    /// OpenAI-compatible vision endpoint (`CTX_VISION_URL`, e.g. OPI's GLM at
    /// `http://host:8000/v1`). When set, image describing uses it instead of the local `claude`
    /// CLI; `vision_model` is the model id and `vision_api_key` the optional bearer token.
    pub vision_url: Option<String>,
    pub vision_api_key: Option<String>,
    /// Confluence Data Center base URL and Personal Access Token (`CONFLUENCE_URL` /
    /// `CONFLUENCE_TOKEN`, in the config .env). The integration is off when either is unset.
    pub confluence_url: Option<String>,
    pub confluence_token: Option<String>,
    /// `confluence-projects.json`: project directory → the space keys indexed and searched for it.
    pub confluence_projects: PathBuf,
    pub confluence_collection: String,
    /// Local times to index Confluence at (`CTX_CONFLUENCE_INDEX_TIMES`).
    pub confluence_index_times: Vec<(u32, u32)>,
    /// Background replay of past Claude prompts to OpenCode (`CTX_REPLAY*`, off by default).
    pub replay: crate::replay::ReplayConfig,
}

/// Off: on the real query set (2026-09-27, 33 queries) every weight tried lowered MRR and none
/// fixed a superseded-decision case, because the only such cases are a day apart. Re-run
/// `ctx eval --sweep` once there are sessions that overturn older ones.
pub const DEFAULT_RECENCY_WEIGHT: f64 = 0.0;
pub const DEFAULT_HALF_LIFE_DAYS: f64 = 60.0;

impl Config {
    pub fn config_dir() -> PathBuf {
        dirs::config_dir().unwrap_or_else(|| PathBuf::from(".")).join("agents-context")
    }

    pub fn env_file() -> PathBuf {
        Self::config_dir().join(".env")
    }

    /// A read-only Jira client, or `None` when the URL or token is unset (integration off).
    pub fn jira(&self) -> Option<crate::jira::Jira> {
        match (&self.jira_url, &self.jira_token) {
            (Some(url), Some(token)) => Some(crate::jira::Jira::new(url, token)),
            _ => None,
        }
    }

    /// A Confluence client, or `None` when the URL or token is unset (integration off).
    pub fn confluence(&self) -> Option<crate::confluence::Confluence> {
        match (&self.confluence_url, &self.confluence_token) {
            (Some(url), Some(token)) => Some(crate::confluence::Confluence::new(url, token)),
            _ => None,
        }
    }

    pub fn load() -> Self {
        // dotenvy stops at the first bad line and drops every setting after it, so a silent
        // failure looks like settings that "don't apply". The classic cause is an unquoted
        // Windows path: backslashes are escapes there, `'D:\x'` is literal.
        if let Err(e) = dotenvy::from_path(Self::env_file())
            && !e.not_found()
        {
            eprintln!("warning: {}: {e}; the lines after it are ignored (quote Windows paths with single quotes)", Self::env_file().display());
        }
        Self::from_env(|k| std::env::var(k).ok())
    }

    pub fn from_env(get: impl Fn(&str) -> Option<String>) -> Self {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        let path = |key: &str, default: PathBuf| get(key).map(PathBuf::from).unwrap_or(default);
        let data = path("CTX_DATA_DIR", dirs::data_local_dir().unwrap_or_else(|| home.clone()).join("agents-context"));
        let number = |key: &str, default: f64| get(key).and_then(|v| v.trim().parse::<f64>().ok()).filter(|v| v.is_finite()).unwrap_or(default);
        let on = |key: &str, default: bool| match get(key).map(|v| v.trim().to_ascii_lowercase()) {
            Some(v) if matches!(v.as_str(), "on" | "1" | "true" | "yes") => true,
            Some(v) if matches!(v.as_str(), "off" | "0" | "false" | "no") => false,
            _ => default,
        };
        Self {
            qdrant_url: get("CTX_QDRANT_URL").unwrap_or_else(|| "http://127.0.0.1:6334".into()),
            qdrant_api_key: get("QDRANT_API_KEY").filter(|k| !k.is_empty()),
            collection: get("CTX_COLLECTION").unwrap_or_else(|| "sessions".into()),
            embed_url: get("CTX_EMBED_URL").unwrap_or_else(|| "http://127.0.0.1:7997".into()),
            state_path: path("CTX_STATE", data.join("state.db")),
            model_dir: path("CTX_MODEL_DIR", PathBuf::from(r"D:\dev\my\agents-context\models\bge-m3-fp32")),
            claude_dir: path("CTX_CLAUDE_DIR", home.join(".claude").join("projects")),
            opencode_db: path("CTX_OPENCODE_DB", home.join(".local/share/opencode/opencode.db")),
            summaries: !get("CTX_SUMMARIES").is_some_and(|v| v.eq_ignore_ascii_case("off")),
            summary_model: get("CTX_SUMMARY_MODEL").unwrap_or_else(|| "haiku".into()),
            claude_exe: path("CTX_CLAUDE_EXE", PathBuf::from("claude")),
            summarizer_dir: data.join("summarizer"),
            redact_file: path("CTX_REDACT_FILE", Self::config_dir().join("redact.txt")),
            data_dir: data.clone(),
            teams_db: path("CTX_TEAMS_DB", PathBuf::from(r"D:\dev\my\teams\data\teams.db")),
            teams_people: path("CTX_TEAMS_PEOPLE", PathBuf::from(r"D:\dev\my\teams\people.json")),
            teams_collection: get("CTX_TEAMS_COLLECTION").unwrap_or_else(|| "teams".into()),
            meetings_collection: get("CTX_MEETINGS_COLLECTION").unwrap_or_else(|| "meetings".into()),
            teams_projects: path("CTX_TEAMS_PROJECTS", Self::config_dir().join("teams-projects.json")),
            teams_watch: crate::teams_watch::WatchConfig {
                cmd: get("CTX_TEAMS_SYNC_CMD").map(|s| s.trim().to_owned()).filter(|s| !s.is_empty()),
                stale_after: Duration::from_secs_f64(number("CTX_TEAMS_STALE_HOURS", 3.0).max(0.0) * 3600.0),
                cooldown: Duration::from_secs_f64(number("CTX_TEAMS_SYNC_COOLDOWN_MINS", 90.0).max(0.0) * 60.0),
                hours: crate::teams_watch::parse_hours(get("CTX_TEAMS_SYNC_HOURS").as_deref().unwrap_or("8-18")),
            },
            recency: crate::rank::Recency {
                weight: number("CTX_RECENCY_WEIGHT", DEFAULT_RECENCY_WEIGHT),
                half_life_days: number("CTX_RECENCY_HALF_LIFE_DAYS", DEFAULT_HALF_LIFE_DAYS),
            },
            search_mode: parse_mode(get("CTX_SEARCH_MODE").as_deref()),
            lexical: crate::lexical::Lexical::parse(get("CTX_LEXICAL").as_deref()),
            rerank: on("CTX_RERANK", false),
            rerank_model: get("CTX_RERANK_MODEL").unwrap_or_else(|| "jina-v2".into()),
            rerank_threads: get("CTX_RERANK_THREADS").and_then(|v| v.trim().parse().ok()),
            rerank_pool: get("CTX_RERANK_POOL").and_then(|v| v.trim().parse().ok()).unwrap_or(30),
            rerank_cache_dir: path("CTX_RERANK_CACHE_DIR", data.join("fastembed-cache")),
            inject: crate::inject::InjectConfig {
                enabled: on("CTX_INJECT", false),
                enabled_claude: on("CTX_INJECT_CLAUDE", on("CTX_INJECT", false)),
                enabled_opencode: on("CTX_INJECT_OPENCODE", on("CTX_INJECT", false)),
                teams: on("CTX_INJECT_TEAMS", false),
                chats: on("CTX_INJECT_CHATS", false),
                min_score: number("CTX_INJECT_MIN_SCORE", crate::inject::DEFAULT_MIN_SCORE as f64) as f32,
                teams_min_score: number("CTX_INJECT_TEAMS_MIN_SCORE", crate::inject::DEFAULT_TEAMS_MIN_SCORE as f64) as f32,
                max_tokens: number("CTX_INJECT_MAX_TOKENS", crate::inject::DEFAULT_MAX_TOKENS as f64) as usize,
            },
            handoff: on("CTX_HANDOFF", true),
            status_warn: number("CTX_STATUS_WARN", 50.0),
            status_crit: number("CTX_STATUS_CRIT", 75.0),
            verify_model: get("CTX_VERIFY_MODEL").unwrap_or_else(|| "sonnet".into()),
            eval_file: path("CTX_EVAL_FILE", Self::config_dir().join("eval").join("sessions.jsonl")),
            eval_baseline: path("CTX_EVAL_BASELINE", data.join("eval-baseline.json")),
            jira_url: get("JIRA_URL").map(|u| u.trim_end_matches('/').to_owned()).filter(|u| !u.is_empty()),
            jira_token: get("JIRA_TOKEN").filter(|t| !t.is_empty()),
            jira_projects: path("JIRA_PROJECTS", Self::config_dir().join("jira-projects.json")),
            jira_write: on("JIRA_WRITE", false),
            jira_collection: get("CTX_JIRA_COLLECTION").unwrap_or_else(|| "jira".into()),
            jira_index_times: parse_times(get("CTX_JIRA_INDEX_TIMES").as_deref().unwrap_or("10:00,17:00")),
            jira_image_describe: on("CTX_JIRA_IMAGE_DESCRIBE", false),
            vision_model: get("CTX_VISION_MODEL").unwrap_or_else(|| "haiku".into()),
            jira_image_max: get("CTX_JIRA_IMAGE_MAX").and_then(|v| v.trim().parse().ok()).unwrap_or(6),
            jira_image_max_bytes: get("CTX_JIRA_IMAGE_MAX_BYTES").and_then(|v| v.trim().parse().ok()).unwrap_or(5 * 1024 * 1024),
            vision_url: get("CTX_VISION_URL").map(|u| u.trim().trim_end_matches('/').to_owned()).filter(|u| !u.is_empty()),
            vision_api_key: get("CTX_VISION_API_KEY").filter(|k| !k.is_empty()),
            confluence_url: get("CONFLUENCE_URL").map(|u| u.trim().trim_end_matches('/').to_owned()).filter(|u| !u.is_empty()),
            confluence_token: get("CONFLUENCE_TOKEN").filter(|t| !t.is_empty()),
            confluence_projects: path("CONFLUENCE_PROJECTS", Self::config_dir().join("confluence-projects.json")),
            confluence_collection: get("CTX_CONFLUENCE_COLLECTION").unwrap_or_else(|| "confluence".into()),
            // Half an hour after Jira's slots, so the two runs do not queue behind each other.
            confluence_index_times: parse_times(get("CTX_CONFLUENCE_INDEX_TIMES").as_deref().unwrap_or("10:30,17:30")),
            replay: crate::replay::ReplayConfig {
                enabled: on("CTX_REPLAY", false),
                projects: crate::replay::parse_projects(get("CTX_REPLAY_PROJECTS").as_deref()),
                every: Duration::from_secs_f64(number("CTX_REPLAY_EVERY_MINS", 60.0).max(1.0) * 60.0),
                daily_max: number("CTX_REPLAY_DAILY_MAX", 10.0).max(0.0) as u32,
                next: crate::replay::parse_range(get("CTX_REPLAY_NEXT").as_deref(), (2, 5)),
                agent: get("CTX_REPLAY_AGENT").map(|a| a.trim().to_owned()).filter(|a| !a.is_empty()).unwrap_or_else(|| "ctx-replay".into()),
                agents_dir: path("CTX_OPENCODE_AGENTS_DIR", home.join(".config").join("opencode").join("agents")),
                // Unset: OpenCode's own global model and provider.
                model: get("CTX_REPLAY_MODEL").map(|m| m.trim().to_owned()).filter(|m| !m.is_empty()),
                exe: path("CTX_OPENCODE_EXE", PathBuf::from("opencode")),
                timeout: Duration::from_secs_f64(number("CTX_REPLAY_TIMEOUT_MINS", 20.0).max(0.1) * 60.0),
                min_chars: number("CTX_REPLAY_MIN_CHARS", 40.0).max(1.0) as usize,
                log: path("CTX_REPLAY_LOG", data.join("logs").join("opencode.log")),
            },
        }
    }
}

/// Dense by default: on the real query set it found every expected session in the top 5 and
/// ranked 88% first, against 97% / 79% for dense+sparse RRF. bge-m3's sparse weights miss
/// Polish inflections (12 of 18 Polish queries), and fusion lets that leg push good dense hits
/// down. Qdrant's weighted RRF does not help: weighting a leg flattens its rank differences,
/// so the other leg decides; DBSF fusion ranked exactly like RRF.
pub fn parse_mode(v: Option<&str>) -> store::SearchMode {
    match v.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
        Some("hybrid") => store::SearchMode::Hybrid,
        Some("sparse") => store::SearchMode::Sparse,
        _ => store::SearchMode::Dense,
    }
}

/// The inject settings as the `.env` file says now, over the process environment. The daemon
/// loaded the file once at start, and dotenvy never overrides a variable already set, so
/// re-loading would keep the old values.
pub fn fresh_inject(env_file: &std::path::Path) -> crate::inject::InjectConfig {
    let text = std::fs::read_to_string(env_file).unwrap_or_default();
    let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
    let file: std::collections::HashMap<String, String> = dotenvy::from_read_iter(text.as_bytes()).filter_map(Result::ok).collect();
    Config::from_env(|k| file.get(k).cloned().or_else(|| std::env::var(k).ok())).inject
}

/// Parses "HH:MM,HH:MM" into `(hour, minute)` pairs, dropping malformed entries. Sorted and
/// deduped so the scheduler can pick the next slot deterministically.
pub fn parse_times(s: &str) -> Vec<(u32, u32)> {
    let mut out: Vec<(u32, u32)> = s
        .split(',')
        .filter_map(|t| {
            let (h, m) = t.trim().split_once(':')?;
            let (h, m) = (h.trim().parse::<u32>().ok()?, m.trim().parse::<u32>().ok()?);
            (h < 24 && m < 60).then_some((h, m))
        })
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// Sets `key=value` in a `.env` file, replacing an existing assignment (also `export KEY=`)
/// and keeping every other line (the API key lives there) and the file's line endings. The
/// file is replaced by a rename, so a crash mid-write cannot leave it half written.
pub fn set_env_value(path: &std::path::Path, key: &str, value: &str) -> std::io::Result<()> {
    edit_env_values(path, &[(key, Some(value))])
}

/// Removes every assignment of `key` from a `.env` file, keeping the rest as [`set_env_value`]
/// does. A file without the key (or no file) is left untouched.
pub fn remove_env_value(path: &std::path::Path, key: &str) -> std::io::Result<()> {
    edit_env_values(path, &[(key, None)])
}

/// Turns auto-inject on or off in a `.env` file. For one `client` (`claude` / `opencode`) only
/// that client's key is written. Without a client the global `CTX_INJECT` is written and both
/// per-client keys are dropped, since a per-client key overrides the global one and would keep
/// its client on the old setting. One rewrite of the file, so it cannot stop half way.
pub fn switch_inject(path: &std::path::Path, client: Option<&str>, on: bool) -> anyhow::Result<()> {
    let value = Some(if on { "on" } else { "off" });
    let edits: &[(&str, Option<&str>)] = match client {
        None => &[("CTX_INJECT", value), ("CTX_INJECT_CLAUDE", None), ("CTX_INJECT_OPENCODE", None)],
        Some("claude") => &[("CTX_INJECT_CLAUDE", value)],
        Some("opencode") => &[("CTX_INJECT_OPENCODE", value)],
        Some(other) => anyhow::bail!("unknown client {other:?}; use claude or opencode"),
    };
    Ok(edit_env_values(path, edits)?)
}

/// Applies every `(key, value)` edit in one rewrite: `Some` sets the key, `None` removes it.
fn edit_env_values(path: &std::path::Path, edits: &[(&str, Option<&str>)]) -> std::io::Result<()> {
    let text = std::fs::read_to_string(path).or_else(|e| if e.kind() == std::io::ErrorKind::NotFound { Ok(String::new()) } else { Err(e) })?;
    let (bom, text) = match text.strip_prefix('\u{feff}') {
        Some(rest) => ("\u{feff}", rest),
        None => ("", text.as_str()),
    };
    let eol = if text.contains("\r\n") { "\r\n" } else { "\n" };
    let assigns = |line: &str, key: &str| {
        let l = line.trim_start();
        let l = l.strip_prefix("export ").map_or(l, str::trim_start);
        l.strip_prefix(key).is_some_and(|rest| rest.trim_start().starts_with('='))
    };
    let mut found = vec![false; edits.len()];
    let mut lines: Vec<String> = Vec::new();
    for line in text.lines() {
        match edits.iter().position(|(key, _)| assigns(line, key)) {
            None => lines.push(line.to_owned()),
            Some(i) if !found[i] => {
                found[i] = true;
                let (key, value) = edits[i];
                lines.extend(value.map(|value| format!("{key}={value}")));
            }
            Some(_) => {}
        }
    }
    let mut changed = found.contains(&true);
    for (&(key, value), _) in edits.iter().zip(&found).filter(|(_, found)| !**found) {
        if let Some(value) = value {
            lines.push(format!("{key}={value}"));
            changed = true;
        }
    }
    if !changed {
        return Ok(());
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("env.tmp");
    std::fs::write(&tmp, format!("{bom}{}{eol}", lines.join(eol)))?;
    std::fs::rename(&tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn defaults_point_at_local_services() {
        let c = Config::from_env(|_| None);
        assert_eq!(c.qdrant_url, "http://127.0.0.1:6334");
        assert_eq!(c.embed_url, "http://127.0.0.1:7997");
        assert_eq!(c.collection, "sessions");
        assert!(c.qdrant_api_key.is_none());
        assert!(c.summaries);
        assert_eq!(c.summary_model, "haiku");
        assert!(c.claude_dir.ends_with(".claude/projects") || c.claude_dir.ends_with(r".claude\projects"));
    }

    #[test]
    fn deep_search_is_off_unless_switched_on() {
        assert!(!Config::from_env(|_| None).rerank);
        assert!(!Config::from_env(|k| (k == "CTX_RERANK_MODEL").then(|| "jina-v2".into())).rerank, "a model alone does not enable it");
        assert!(Config::from_env(|k| (k == "CTX_RERANK").then(|| "on".into())).rerank);
    }

    #[test]
    fn environment_overrides_defaults_and_blank_key_is_none() {
        let env: HashMap<&str, &str> = [("CTX_QDRANT_URL", "http://q:1"), ("QDRANT_API_KEY", ""), ("CTX_STATE", "x.db")].into();
        let c = Config::from_env(|k| env.get(k).map(|v| v.to_string()));
        assert_eq!(c.qdrant_url, "http://q:1");
        assert!(c.qdrant_api_key.is_none());
        assert_eq!(c.state_path, PathBuf::from("x.db"));
    }

    #[test]
    fn inject_is_off_by_default_and_numbers_fall_back_on_garbage() {
        let c = Config::from_env(|_| None);
        assert!(!c.inject.enabled && !c.inject.teams && !c.inject.chats);
        assert_eq!(c.recency.weight, DEFAULT_RECENCY_WEIGHT);
        let env: HashMap<&str, &str> = [("CTX_INJECT", "ON"), ("CTX_RECENCY_WEIGHT", "abc"), ("CTX_INJECT_MIN_SCORE", " 0.7 ")].into();
        let c = Config::from_env(|k| env.get(k).map(|v| v.to_string()));
        assert!(c.inject.enabled);
        assert_eq!(c.recency.weight, DEFAULT_RECENCY_WEIGHT);
        assert!((c.inject.min_score - 0.7).abs() < 1e-6);
    }

    #[test]
    fn search_mode_defaults_to_dense() {
        assert_eq!(parse_mode(None), store::SearchMode::Dense);
        assert_eq!(parse_mode(Some(" Hybrid ")), store::SearchMode::Hybrid);
        assert_eq!(parse_mode(Some("nonsense")), store::SearchMode::Dense);
    }

    #[test]
    fn set_env_value_replaces_in_place_and_keeps_other_lines() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join(".env");
        std::fs::write(&path, "QDRANT_API_KEY=abc=def\n# comment\nCTX_INJECT=off\nCTX_INJECTX=1\nCTX_INJECT = on\n").unwrap();
        set_env_value(&path, "CTX_INJECT", "on").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "QDRANT_API_KEY=abc=def\n# comment\nCTX_INJECT=on\nCTX_INJECTX=1\n");
        let fresh = tmp.path().join("new/.env");
        set_env_value(&fresh, "CTX_INJECT", "off").unwrap();
        assert_eq!(std::fs::read_to_string(&fresh).unwrap(), "CTX_INJECT=off\n");
    }

    #[test]
    fn set_env_value_handles_export_bom_and_crlf() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join(".env");
        std::fs::write(&path, "\u{feff}export CTX_INJECT=off\r\nQDRANT_API_KEY=k\r\n").unwrap();
        set_env_value(&path, "CTX_INJECT", "on").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "\u{feff}CTX_INJECT=on\r\nQDRANT_API_KEY=k\r\n");
        assert!(!tmp.path().join(".env.tmp").exists() && !path.with_extension("env.tmp").exists());
        assert!(fresh_inject(&path).enabled, "the daemon's fresh read sees the change");
    }

    #[test]
    fn set_env_value_keeps_windows_paths_verbatim() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join(".env");
        let line = r"CTX_MODEL_DIR=D:\dev\my\agents-context\models\bge-m3-fp32";
        std::fs::write(&path, format!("{line}\nCTX_INJECT=off\n")).unwrap();
        set_env_value(&path, "CTX_INJECT", "on").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), format!("{line}\nCTX_INJECT=on\n"));
    }

    #[test]
    fn remove_env_value_drops_every_assignment_of_the_key_and_nothing_else() {
        let rows = [
            (
                "plain, export, spaced and duplicate assignments; prefix keys, comments and values stay",
                "CTX_INJECT",
                "QDRANT_API_KEY=abc=def\n# CTX_INJECT=on\nCTX_INJECT=off\nCTX_INJECT_CLAUDE=on\nexport CTX_INJECT=on\n  CTX_INJECT = on\nCTX_INJECTX=1\nOTHER=CTX_INJECT=on\nCTX_INJECT_OPENCODE=off\n",
                "QDRANT_API_KEY=abc=def\n# CTX_INJECT=on\nCTX_INJECT_CLAUDE=on\nCTX_INJECTX=1\nOTHER=CTX_INJECT=on\nCTX_INJECT_OPENCODE=off\n",
            ),
            (
                "a per-client key goes, the global key and the other client stay",
                "CTX_INJECT_CLAUDE",
                "CTX_INJECT=off\nCTX_INJECT_CLAUDE=on\nCTX_INJECT_OPENCODE=on\n",
                "CTX_INJECT=off\nCTX_INJECT_OPENCODE=on\n",
            ),
            (
                "BOM and CRLF survive",
                "CTX_INJECT_CLAUDE",
                "\u{feff}QDRANT_API_KEY=k\r\nexport CTX_INJECT_CLAUDE=on\r\nCTX_INJECT=off\r\nCTX_INJECT_CLAUDE=off\r\n",
                "\u{feff}QDRANT_API_KEY=k\r\nCTX_INJECT=off\r\n",
            ),
            (
                "the key on the BOM line is found",
                "CTX_INJECT_CLAUDE",
                "\u{feff}CTX_INJECT_CLAUDE=on\nQDRANT_API_KEY=k\n",
                "\u{feff}QDRANT_API_KEY=k\n",
            ),
            ("a Windows path next to the key is kept verbatim", "CTX_INJECT_OPENCODE", "CTX_MODEL_DIR=D:\\dev\\models\\bge-m3-fp32\nCTX_INJECT_OPENCODE=on\n", "CTX_MODEL_DIR=D:\\dev\\models\\bge-m3-fp32\n"),
        ];
        for (name, key, before, after) in rows {
            let tmp = tempfile::tempdir().unwrap();
            let path = tmp.path().join(".env");
            std::fs::write(&path, before).unwrap();
            remove_env_value(&path, key).unwrap();
            assert_eq!(std::fs::read_to_string(&path).unwrap(), after, "{name}");
            assert!(!path.with_extension("env.tmp").exists(), "{name}: no temp file is left behind");
        }
    }

    #[test]
    fn remove_env_value_leaves_a_file_without_the_key_byte_for_byte() {
        // No trailing newline and mixed line endings: a rewrite would normalise both.
        for before in ["A=1\r\nCTX_INJECT=on\nCTX_INJECT_CLAUDEX=1", "\u{feff}# CTX_INJECT_CLAUDE=on\n\n\nB=2\n\n", ""] {
            let tmp = tempfile::tempdir().unwrap();
            let path = tmp.path().join(".env");
            std::fs::write(&path, before).unwrap();
            remove_env_value(&path, "CTX_INJECT_CLAUDE").unwrap();
            assert_eq!(std::fs::read(&path).unwrap(), before.as_bytes(), "{before:?}");
            assert!(!path.with_extension("env.tmp").exists());
        }
    }

    #[test]
    fn remove_env_value_does_not_create_a_missing_file() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("new/.env");
        remove_env_value(&path, "CTX_INJECT_CLAUDE").unwrap();
        assert!(!path.exists());
        assert!(!tmp.path().join("new").exists(), "not even its directory");
    }

    /// The inject settings a `.env` file alone gives, without the process environment.
    fn inject_from(path: &std::path::Path) -> crate::inject::InjectConfig {
        let text = std::fs::read_to_string(path).unwrap();
        let file: HashMap<String, String> = dotenvy::from_read_iter(text.as_bytes()).map(Result::unwrap).collect();
        Config::from_env(|k| file.get(k).cloned()).inject
    }

    #[test]
    fn the_global_switch_wins_over_an_earlier_per_client_switch() {
        // (client, its switch, the global switch that follows)
        let rows = [("claude", true, false), ("claude", false, true), ("opencode", true, false), ("opencode", false, true)];
        for (client, client_on, global_on) in rows {
            let name = format!("{client} {client_on}, then all {global_on}");
            let tmp = tempfile::tempdir().unwrap();
            let path = tmp.path().join(".env");
            std::fs::write(&path, "QDRANT_API_KEY=k\n").unwrap();
            switch_inject(&path, Some(client), client_on).unwrap();
            let before = inject_from(&path);
            // The other client still follows the unset global switch, which is off.
            let (this, other) = if client == "claude" { (before.enabled_claude, before.enabled_opencode) } else { (before.enabled_opencode, before.enabled_claude) };
            assert_eq!((this, other), (client_on, false), "{name}: the per-client switch applies to its client only");

            switch_inject(&path, None, global_on).unwrap();
            let value = if global_on { "on" } else { "off" };
            assert_eq!(std::fs::read_to_string(&path).unwrap(), format!("QDRANT_API_KEY=k\nCTX_INJECT={value}\n"), "{name}");
            let i = inject_from(&path);
            assert_eq!((i.enabled, i.enabled_claude, i.enabled_opencode), (global_on, global_on, global_on), "{name}");
        }
    }

    #[test]
    fn the_global_switch_drops_both_per_client_keys_in_every_spelling() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join(".env");
        std::fs::write(&path, "\u{feff}CTX_INJECT_OPENCODE=on\r\nCTX_INJECT=on\r\nQDRANT_API_KEY=k\r\nexport CTX_INJECT_CLAUDE=on\r\nCTX_INJECT_CLAUDE = on\r\nCTX_INJECT_TEAMS=on\r\n").unwrap();
        switch_inject(&path, None, false).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "\u{feff}CTX_INJECT=off\r\nQDRANT_API_KEY=k\r\nCTX_INJECT_TEAMS=on\r\n");
        assert!(!path.with_extension("env.tmp").exists());
    }

    #[test]
    fn a_per_client_switch_leaves_the_global_and_the_other_clients_key_alone() {
        // (file, client, on, file afterwards, expected (global, claude, opencode))
        let rows = [
            ("CTX_INJECT=off\nCTX_INJECT_OPENCODE=on\n", "claude", true, "CTX_INJECT=off\nCTX_INJECT_OPENCODE=on\nCTX_INJECT_CLAUDE=on\n", (false, true, true)),
            ("CTX_INJECT=off\nCTX_INJECT_OPENCODE=on\nCTX_INJECT_CLAUDE=on\n", "claude", false, "CTX_INJECT=off\nCTX_INJECT_OPENCODE=on\nCTX_INJECT_CLAUDE=off\n", (false, false, true)),
            ("CTX_INJECT=on\nCTX_INJECT_CLAUDE=off\n", "opencode", false, "CTX_INJECT=on\nCTX_INJECT_CLAUDE=off\nCTX_INJECT_OPENCODE=off\n", (true, false, false)),
            ("CTX_INJECT_CLAUDE=on\nCTX_INJECT=off\nCTX_INJECT_OPENCODE=off\n", "opencode", true, "CTX_INJECT_CLAUDE=on\nCTX_INJECT=off\nCTX_INJECT_OPENCODE=on\n", (false, true, true)),
            ("CTX_INJECT=on\nCTX_INJECT_OPENCODE=off\n", "claude", false, "CTX_INJECT=on\nCTX_INJECT_OPENCODE=off\nCTX_INJECT_CLAUDE=off\n", (true, false, false)),
            ("CTX_INJECT=off\nCTX_INJECT_CLAUDE=on\n", "opencode", false, "CTX_INJECT=off\nCTX_INJECT_CLAUDE=on\nCTX_INJECT_OPENCODE=off\n", (false, true, false)),
        ];
        for (before, client, on, after, enabled) in rows {
            let tmp = tempfile::tempdir().unwrap();
            let path = tmp.path().join(".env");
            std::fs::write(&path, before).unwrap();
            switch_inject(&path, Some(client), on).unwrap();
            assert_eq!(std::fs::read_to_string(&path).unwrap(), after, "{client} {on} on {before:?}");
            let i = inject_from(&path);
            assert_eq!((i.enabled, i.enabled_claude, i.enabled_opencode), enabled, "{client} {on} on {before:?}");
        }
    }

    #[test]
    fn an_unknown_client_is_an_error_and_the_file_is_not_touched() {
        // CRLF and no final newline: any rewrite would show.
        let before = "CTX_INJECT=off\r\nCTX_INJECT_CLAUDE=on\r\nCTX_INJECT_OPENCODE=on";
        for client in ["cursor", "Claude", "claude ", "all", ""] {
            for on in [true, false] {
                let tmp = tempfile::tempdir().unwrap();
                let path = tmp.path().join(".env");
                std::fs::write(&path, before).unwrap();
                let err = switch_inject(&path, Some(client), on).unwrap_err();
                assert_eq!(err.to_string(), format!("unknown client {client:?}; use claude or opencode"));
                assert_eq!(std::fs::read(&path).unwrap(), before.as_bytes(), "{client:?}");

                let missing = tmp.path().join("new/.env");
                switch_inject(&missing, Some(client), on).unwrap_err();
                assert!(!tmp.path().join("new").exists(), "{client:?}: nothing is created either");
            }
        }
    }

    #[test]
    fn parse_times_reads_and_sorts_valid_slots() {
        assert_eq!(parse_times("10:00,17:00"), [(10, 0), (17, 0)]);
        assert_eq!(parse_times("17:30, 09:05"), [(9, 5), (17, 30)]);
        assert_eq!(parse_times("25:00,10:00,nonsense,10:00"), [(10, 0)], "bad/duplicate dropped");
        assert!(parse_times("").is_empty());
        let c = Config::from_env(|k| (k == "CTX_JIRA_INDEX_TIMES").then(|| "08:15".to_owned()));
        assert_eq!(c.jira_index_times, [(8, 15)]);
        assert_eq!(Config::from_env(|_| None).jira_index_times, [(10, 0), (17, 0)]);
    }

    #[test]
    fn confluence_is_off_until_both_url_and_token_are_set() {
        assert!(Config::from_env(|_| None).confluence().is_none());
        let url_only = Config::from_env(|k| (k == "CONFLUENCE_URL").then(|| "https://confluence.example.org/".to_owned()));
        assert_eq!(url_only.confluence_url.as_deref(), Some("https://confluence.example.org"));
        assert!(url_only.confluence().is_none());
        let both = Config::from_env(|k| match k {
            "CONFLUENCE_URL" => Some("https://confluence.example.org".into()),
            "CONFLUENCE_TOKEN" => Some("tok".into()),
            _ => None,
        });
        assert!(both.confluence().is_some());
        assert_eq!(both.confluence_collection, "confluence");
        assert_eq!(both.confluence_index_times, [(10, 30), (17, 30)]);
    }

    #[test]
    fn non_finite_numbers_fall_back_to_defaults() {
        let c = Config::from_env(|k| (k == "CTX_RECENCY_WEIGHT").then(|| "NaN".to_owned()));
        assert_eq!(c.recency.weight, DEFAULT_RECENCY_WEIGHT);
    }

    #[test]
    fn replay_is_off_by_default_with_documented_defaults() {
        let data = PathBuf::from(r"Z:\ctx-data");
        let r = Config::from_env(|k| (k == "CTX_DATA_DIR").then(|| data.to_string_lossy().into_owned())).replay;
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        assert!(!r.enabled);
        assert_eq!(r.projects, Vec::<String>::new());
        assert_eq!(r.every, Duration::from_secs(60 * 60));
        assert_eq!(r.daily_max, 10);
        assert_eq!(r.next, (2, 5));
        assert_eq!(r.timeout, Duration::from_secs(20 * 60));
        assert_eq!(r.min_chars, 40);
        assert_eq!(r.agent, "ctx-replay");
        assert_eq!(r.agents_dir, home.join(".config").join("opencode").join("agents"));
        assert_eq!(r.model, None);
        assert_eq!(r.exe, PathBuf::from("opencode"));
        assert_eq!(r.log, data.join("logs").join("opencode.log"));
    }

    #[test]
    fn replay_reads_every_key() {
        let env: HashMap<&str, &str> = [
            ("CTX_REPLAY", "on"),
            ("CTX_REPLAY_PROJECTS", " helpdesk , ,sedn26,"),
            ("CTX_REPLAY_EVERY_MINS", "15"),
            ("CTX_REPLAY_DAILY_MAX", "3"),
            ("CTX_REPLAY_NEXT", "4-1"),
            ("CTX_REPLAY_TIMEOUT_MINS", "2.5"),
            ("CTX_REPLAY_MIN_CHARS", "25"),
            ("CTX_REPLAY_MODEL", " opi/glm-5 "),
            ("CTX_REPLAY_AGENT", " ro-agent "),
            ("CTX_OPENCODE_AGENTS_DIR", r"Z:\oc\agents"),
            ("CTX_OPENCODE_EXE", r"Z:\bin\opencode.cmd"),
            ("CTX_REPLAY_LOG", r"Z:\logs\replay.log"),
        ]
        .into();
        let r = Config::from_env(|k| env.get(k).map(|v| v.to_string())).replay;
        assert!(r.enabled);
        assert_eq!(r.projects, ["helpdesk", "sedn26"]);
        assert_eq!(r.every, Duration::from_secs(15 * 60));
        assert_eq!(r.daily_max, 3);
        assert_eq!(r.next, (1, 4), "a reversed range is put in order");
        assert_eq!(r.timeout, Duration::from_secs(150));
        assert_eq!(r.min_chars, 25);
        assert_eq!(r.model.as_deref(), Some("opi/glm-5"));
        assert_eq!(r.agent, "ro-agent");
        assert_eq!(r.agents_dir, PathBuf::from(r"Z:\oc\agents"));
        assert_eq!(r.exe, PathBuf::from(r"Z:\bin\opencode.cmd"));
        assert_eq!(r.log, PathBuf::from(r"Z:\logs\replay.log"));
    }

    #[test]
    fn replay_numbers_at_or_below_zero_are_clamped() {
        for value in ["0", "-5"] {
            let env: HashMap<&str, &str> = [
                ("CTX_REPLAY_EVERY_MINS", value),
                ("CTX_REPLAY_DAILY_MAX", value),
                ("CTX_REPLAY_TIMEOUT_MINS", value),
                ("CTX_REPLAY_MIN_CHARS", value),
            ]
            .into();
            let r = Config::from_env(|k| env.get(k).map(|v| v.to_string())).replay;
            assert_eq!((r.every, r.daily_max, r.timeout, r.min_chars), (Duration::from_secs(60), 0, Duration::from_secs(6), 1), "{value}");
        }
    }

    #[test]
    fn replay_next_falls_back_on_garbage_and_blank_model_and_agent_mean_the_defaults() {
        let rows: [(&str, (u32, u32)); 4] = [("3", (3, 3)), ("2-5", (2, 5)), ("two to five", (2, 5)), ("", (2, 5))];
        for (next, want) in rows {
            assert_eq!(Config::from_env(|k| (k == "CTX_REPLAY_NEXT").then(|| next.to_owned())).replay.next, want, "{next:?}");
        }
        let env: HashMap<&str, &str> = [("CTX_REPLAY_MODEL", "  "), ("CTX_REPLAY_AGENT", " "), ("CTX_REPLAY", "maybe")].into();
        let r = Config::from_env(|k| env.get(k).map(|v| v.to_string())).replay;
        assert_eq!((r.model, r.agent.as_str(), r.enabled), (None, "ctx-replay", false));
    }

    #[test]
    fn summaries_can_be_switched_off() {
        assert!(!Config::from_env(|k| (k == "CTX_SUMMARIES").then(|| "OFF".to_owned())).summaries);
        assert!(Config::from_env(|k| (k == "CTX_SUMMARIES").then(|| "on".to_owned())).summaries);
    }
}
