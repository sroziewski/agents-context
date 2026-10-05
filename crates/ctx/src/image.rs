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
            // Tools are left enabled (Read must see the image); skip the per-read permission prompt
            // for the single image in the throwaway workdir.
            "--dangerously-skip-permissions",
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
    fn args_allow_reading_but_disable_hooks_and_persistence() {
        let d = ClaudeImageDescriber::new("claude".into(), "haiku", ".".into(), Duration::from_secs(1));
        let a = d.args();
        for flag in ["-p", "--dangerously-skip-permissions", "--setting-sources", "--strict-mcp-config", "--no-session-persistence"] {
            assert!(a.iter().any(|x| x == flag), "missing {flag}");
        }
        assert!(!a.iter().any(|x| x == "--tools"), "tools stay enabled so Read can see the image");
        let model = a.iter().position(|x| x == "--model").unwrap();
        assert_eq!(a[model + 1], "haiku");
    }
}
