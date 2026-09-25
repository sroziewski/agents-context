use std::path::PathBuf;

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
}

impl Config {
    pub fn config_dir() -> PathBuf {
        dirs::config_dir().unwrap_or_else(|| PathBuf::from(".")).join("agents-context")
    }

    pub fn load() -> Self {
        let _ = dotenvy::from_path(Self::config_dir().join(".env"));
        Self::from_env(|k| std::env::var(k).ok())
    }

    pub fn from_env(get: impl Fn(&str) -> Option<String>) -> Self {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        let data = dirs::data_local_dir().unwrap_or_else(|| home.clone()).join("agents-context");
        let path = |key: &str, default: PathBuf| get(key).map(PathBuf::from).unwrap_or(default);
        Self {
            qdrant_url: get("CTX_QDRANT_URL").unwrap_or_else(|| "http://127.0.0.1:6334".into()),
            qdrant_api_key: get("QDRANT_API_KEY").filter(|k| !k.is_empty()),
            collection: get("CTX_COLLECTION").unwrap_or_else(|| "sessions".into()),
            embed_url: get("CTX_EMBED_URL").unwrap_or_else(|| "http://127.0.0.1:7997".into()),
            state_path: path("CTX_STATE", data.join("state.db")),
            model_dir: path("CTX_MODEL_DIR", PathBuf::from(r"D:\dev\my\agents-context\models\bge-m3-fp32")),
            claude_dir: path("CTX_CLAUDE_DIR", home.join(".claude").join("projects")),
            opencode_db: path("CTX_OPENCODE_DB", home.join(".local/share/opencode/opencode.db")),
        }
    }
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
}
