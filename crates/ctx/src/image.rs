//! Vision descriptions of images (Jira screenshots) via a headless `claude`.
//!
//! Unlike the summarizer, this run must *read a file*, so it does not disable tools — it leaves the
//! default tools available (Read presents the image to the model) and passes
//! `--dangerously-skip-permissions` to auto-approve reading the single image in a throwaway
//! workdir. Hooks and project MCP are still disabled and the summarizer guard env is set, so the
//! run cannot recurse back into ctx.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use ingest::jira::ImageDescriber;

use crate::hook::SUMMARIZER_ENV;
use crate::summarizer::run_claude;

const VISION_PROMPT: &str = "Read the image file in the current directory and describe it for a search index. \
First output its visible text verbatim, then one or two sentences on what it shows (UI, diagram, chart, error, table, etc.). \
No preamble and no questions.";

/// Distinguishes concurrent describe calls' workdirs within one process.
static SEQ: AtomicU64 = AtomicU64::new(0);

pub struct ClaudeImageDescriber {
    pub exe: PathBuf,
    pub model: String,
    /// Base dir; each `describe` gets a unique throwaway subdir holding only the one image.
    pub workdir: PathBuf,
    pub timeout: Duration,
}

impl ClaudeImageDescriber {
    pub fn new(exe: PathBuf, model: &str, workdir: PathBuf, timeout: Duration) -> Self {
        Self { exe, model: model.to_owned(), workdir, timeout }
    }

    fn args(&self) -> Vec<String> {
        [
            "-p",
            "--model",
            &self.model,
            "--no-session-persistence",
            // No hooks and no project settings/MCP, so the run can't trigger ctx's own hooks.
            "--setting-sources",
            "",
            "--strict-mcp-config",
            "--settings",
            r#"{"autoMemoryEnabled":false}"#,
            // Read is the only tool: the screenshot is untrusted, and text in it must not be
            // able to make the run write files or run commands.
            "--tools",
            "Read",
            "--allowedTools",
            "Read",
            "--output-format",
            "json",
        ]
        .map(str::to_owned)
        .to_vec()
    }
}

impl ImageDescriber for ClaudeImageDescriber {
    fn describe(&self, bytes: &[u8], filename: &str, mime: &str) -> Result<String, String> {
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        let dir = self.workdir.join(format!("img-{nanos}-{seq}"));
        std::fs::create_dir_all(&dir).map_err(|e| format!("image workdir: {e}"))?;
        let path = dir.join(format!("image.{}", extension(mime, filename)));
        let result = std::fs::write(&path, bytes)
            .map_err(|e| format!("writing image: {e}"))
            .and_then(|()| run_claude(&self.exe, &self.args(), &dir, VISION_PROMPT, &[(SUMMARIZER_ENV, "1")], self.timeout));
        // Always clean up the image, whatever happened.
        let _ = std::fs::remove_dir_all(&dir);
        result
    }
}

/// Describes an image via an OpenAI-compatible chat-completions endpoint (e.g. OPI's GLM), sending
/// the image inline as a base64 data URL. Blocking (ureq), which fits the sync `ImageDescriber`
/// trait called from the indexer's blocking thread.
pub struct OpenAiImageDescriber {
    agent: ureq::Agent,
    /// Base URL ending in `/v1` (no trailing slash).
    url: String,
    model: String,
    api_key: Option<String>,
}

impl OpenAiImageDescriber {
    pub fn new(base_url: &str, model: &str, api_key: Option<String>, timeout: Duration) -> Self {
        let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(timeout)).build().into();
        Self {
            agent,
            url: base_url.trim_end_matches('/').to_owned(),
            model: model.to_owned(),
            // "no-key-needed" is OpenCode's placeholder for an unauthenticated endpoint.
            api_key: api_key.filter(|k| !k.is_empty() && k != "no-key-needed"),
        }
    }
}

impl ImageDescriber for OpenAiImageDescriber {
    fn describe(&self, bytes: &[u8], _filename: &str, mime: &str) -> Result<String, String> {
        let mime = if mime.trim().is_empty() { "image/png" } else { mime };
        let data_url = format!("data:{mime};base64,{}", base64(bytes));
        let body = serde_json::json!({
            "model": self.model,
            "temperature": 0,
            "max_tokens": 400,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": VISION_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]}],
        });
        let mut req = self.agent.post(&format!("{}/chat/completions", self.url));
        if let Some(k) = &self.api_key {
            req = req.header("Authorization", format!("Bearer {k}"));
        }
        let mut resp = req.send_json(&body).map_err(|e| format!("vision request failed: {e}"))?;
        let v: serde_json::Value = resp.body_mut().read_json().map_err(|e| format!("vision response parse: {e}"))?;
        let content = v.pointer("/choices/0/message/content").and_then(|c| c.as_str()).unwrap_or_default().trim().to_owned();
        if content.is_empty() {
            return Err("vision endpoint returned empty content".into());
        }
        Ok(content)
    }
}

/// Standard base64 (with padding) — a few lines, so no dependency just to inline an image.
fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let n = ((chunk[0] as u32) << 16) | ((*chunk.get(1).unwrap_or(&0) as u32) << 8) | (*chunk.get(2).unwrap_or(&0) as u32);
        out.push(T[(n >> 18 & 63) as usize] as char);
        out.push(T[(n >> 12 & 63) as usize] as char);
        out.push(if chunk.len() > 1 { T[(n >> 6 & 63) as usize] as char } else { '=' });
        out.push(if chunk.len() > 2 { T[(n & 63) as usize] as char } else { '=' });
    }
    out
}

/// An extension the model will recognise: from the mime type, else the filename's, else png.
fn extension(mime: &str, filename: &str) -> String {
    match mime.to_ascii_lowercase().as_str() {
        "image/png" => "png",
        "image/jpeg" | "image/jpg" => "jpg",
        "image/gif" => "gif",
        "image/webp" => "webp",
        "image/bmp" => "bmp",
        _ => filename
            .rsplit_once('.')
            .map(|(_, ext)| ext)
            .filter(|e| !e.is_empty() && e.len() <= 5 && e.chars().all(|c| c.is_ascii_alphanumeric()))
            .unwrap_or("png"),
    }
    .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extension_prefers_mime_then_filename_then_png() {
        assert_eq!(extension("image/png", "x.bin"), "png");
        assert_eq!(extension("image/jpeg", "x"), "jpg");
        assert_eq!(extension("application/octet-stream", "shot.PNG"), "PNG");
        assert_eq!(extension("application/octet-stream", "noext"), "png");
        assert_eq!(extension("", "a.verylongext"), "png", "implausible extension falls back");
    }

    #[test]
    fn args_allow_only_read_and_disable_hooks_settings_mcp_and_persistence() {
        let d = ClaudeImageDescriber::new("claude".into(), "sonnet-x", ".".into(), Duration::from_secs(1));
        assert_eq!(
            d.args(),
            [
                "-p",
                "--model",
                "sonnet-x",
                "--no-session-persistence",
                "--setting-sources",
                "",
                "--strict-mcp-config",
                "--settings",
                r#"{"autoMemoryEnabled":false}"#,
                "--tools",
                "Read",
                "--allowedTools",
                "Read",
                "--output-format",
                "json",
            ]
        );
    }

    #[test]
    fn base64_matches_known_vectors() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"hi"), "aGk=");
        assert_eq!(base64(b"abc"), "YWJj");
    }

    #[test]
    fn openai_describer_drops_placeholder_and_empty_keys() {
        let d = OpenAiImageDescriber::new("http://h:8000/v1/", "opi-glm-4.6", Some("no-key-needed".into()), Duration::from_secs(1));
        assert_eq!(d.url, "http://h:8000/v1", "trailing slash trimmed");
        assert!(d.api_key.is_none(), "the no-key-needed placeholder is not sent as a bearer token");
        assert!(OpenAiImageDescriber::new("http://h/v1", "m", Some(String::new()), Duration::from_secs(1)).api_key.is_none());
        assert_eq!(OpenAiImageDescriber::new("http://h/v1", "m", Some("real".into()), Duration::from_secs(1)).api_key.as_deref(), Some("real"));
    }
}
