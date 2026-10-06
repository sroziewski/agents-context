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
    /// Deep-search reranker: which CPU model, its thread cap, and the candidate pool. Opt-in;
    /// normal search never loads it (DEV-28 found it ~16 s/query, too slow for the default).
    pub rerank_model: String,
    pub rerank_threads: Option<usize>,
    pub rerank_pool: u64,
    /// Where the reranker model is cached (HF-hub layout). Stable, independent of the process's
    /// working directory.
    pub rerank_cache_dir: PathBuf,
    pub inject: crate::inject::InjectConfig,
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
        let data = dirs::data_local_dir().unwrap_or_else(|| home.clone()).join("agents-context");
        let path = |key: &str, default: PathBuf| get(key).map(PathBuf::from).unwrap_or(default);
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
                stale_after: Duration::from_secs_f64(number("CTX_TEAMS_STALE_HOURS", 6.0).max(0.0) * 3600.0),
                cooldown: Duration::from_secs_f64(number("CTX_TEAMS_SYNC_COOLDOWN_MINS", 90.0).max(0.0) * 60.0),
                hours: crate::teams_watch::parse_hours(get("CTX_TEAMS_SYNC_HOURS").as_deref().unwrap_or("8-18")),
            },
            recency: crate::rank::Recency {
                weight: number("CTX_RECENCY_WEIGHT", DEFAULT_RECENCY_WEIGHT),
                half_life_days: number("CTX_RECENCY_HALF_LIFE_DAYS", DEFAULT_HALF_LIFE_DAYS),
            },
            search_mode: parse_mode(get("CTX_SEARCH_MODE").as_deref()),
            rerank_model: get("CTX_RERANK_MODEL").unwrap_or_else(|| "jina-v2".into()),
            rerank_threads: get("CTX_RERANK_THREADS").and_then(|v| v.trim().parse().ok()),
            rerank_pool: get("CTX_RERANK_POOL").and_then(|v| v.trim().parse().ok()).unwrap_or(30),
            rerank_cache_dir: path("CTX_RERANK_CACHE_DIR", data.join("fastembed-cache")),
            inject: crate::inject::InjectConfig {
                enabled: on("CTX_INJECT", false),
                enabled_claude: on("CTX_INJECT_CLAUDE", on("CTX_INJECT", false)),
                enabled_opencode: on("CTX_INJECT_OPENCODE", on("CTX_INJECT", false)),
                teams: on("CTX_INJECT_TEAMS", false),
                min_score: number("CTX_INJECT_MIN_SCORE", crate::inject::DEFAULT_MIN_SCORE as f64) as f32,
                teams_min_score: number("CTX_INJECT_TEAMS_MIN_SCORE", crate::inject::DEFAULT_TEAMS_MIN_SCORE as f64) as f32,
                max_tokens: number("CTX_INJECT_MAX_TOKENS", crate::inject::DEFAULT_MAX_TOKENS as f64) as usize,
            },
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
    let text = std::fs::read_to_string(path).or_else(|e| if e.kind() == std::io::ErrorKind::NotFound { Ok(String::new()) } else { Err(e) })?;
    let (bom, text) = match text.strip_prefix('\u{feff}') {
        Some(rest) => ("\u{feff}", rest),
        None => ("", text.as_str()),
    };
    let eol = if text.contains("\r\n") { "\r\n" } else { "\n" };
    let assigns = |line: &str| {
        let l = line.trim_start();
        let l = l.strip_prefix("export ").map_or(l, str::trim_start);
        l.strip_prefix(key).is_some_and(|rest| rest.trim_start().starts_with('='))
    };
    let mut found = false;
    let mut lines: Vec<String> = Vec::new();
    for line in text.lines() {
        if !assigns(line) {
            lines.push(line.to_owned());
        } else if !found {
            found = true;
            lines.push(format!("{key}={value}"));
        }
    }
    if !found {
        lines.push(format!("{key}={value}"));
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
        assert!(!c.inject.enabled && !c.inject.teams);
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
    fn non_finite_numbers_fall_back_to_defaults() {
        let c = Config::from_env(|k| (k == "CTX_RECENCY_WEIGHT").then(|| "NaN".to_owned()));
        assert_eq!(c.recency.weight, DEFAULT_RECENCY_WEIGHT);
    }

    #[test]
    fn summaries_can_be_switched_off() {
        assert!(!Config::from_env(|k| (k == "CTX_SUMMARIES").then(|| "OFF".to_owned())).summaries);
        assert!(Config::from_env(|k| (k == "CTX_SUMMARIES").then(|| "on".to_owned())).summaries);
    }
}
