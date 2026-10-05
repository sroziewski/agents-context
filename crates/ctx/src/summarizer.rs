//! Session summaries written by Claude Code in headless mode (`claude -p`).
//!
//! The run is locked down so it cannot act or recurse: no tools, no settings (hence no hooks and
//! no MCP servers), auto-memory off, no transcript written, and `CTX_SUMMARIZER` set so even a
//! hook that did run would be a no-op. The custom system prompt replaces Claude Code's default
//! one, which otherwise makes it offer follow-ups ("save this to memory?") inside the summary.

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use ingest::Summarizer;

use crate::hook::SUMMARIZER_ENV;

const SYSTEM_PROMPT: &str = "You write terse factual summaries of software-engineering sessions. \
Output only the summary in the requested format. Never ask questions, never offer follow-ups, never mention memory.";

pub struct ClaudeCli {
    pub exe: PathBuf,
    pub model: String,
    /// Where `claude` runs. A dedicated directory, so nothing it might create lands in a project.
    pub workdir: PathBuf,
    pub timeout: Duration,
    id: String,
}

impl ClaudeCli {
    pub fn new(exe: PathBuf, model: &str, workdir: PathBuf, timeout: Duration) -> Self {
        let id = format!("claude-cli/{model}");
        Self { exe, model: model.to_owned(), workdir, timeout, id }
    }

    fn args(&self) -> Vec<String> {
        [
            "-p",
            "--model",
            &self.model,
            "--no-session-persistence",
            "--tools",
            "",
            "--setting-sources",
            "",
            "--strict-mcp-config",
            "--settings",
            r#"{"autoMemoryEnabled":false}"#,
            "--system-prompt",
            SYSTEM_PROMPT,
            "--output-format",
            "json",
        ]
        .map(str::to_owned)
        .to_vec()
    }
}

impl Summarizer for ClaudeCli {
    fn id(&self) -> &str {
        &self.id
    }

    fn summarize(&self, prompt: &str, timeout: Duration) -> Result<String, String> {
        run_claude(&self.exe, &self.args(), &self.workdir, prompt, &[(SUMMARIZER_ENV, "1")], self.timeout.min(timeout))
    }
}

/// Runs `claude` headless, feeding `prompt` on stdin, and returns its parsed `result`. Shared by
/// the summarizer and the image describer: it drains both pipes on their own threads (a child
/// blocked writing a full pipe would otherwise hang), enforces `timeout` by killing the whole
/// process tree, and reports a non-zero exit with the stderr tail.
pub(crate) fn run_claude(exe: &std::path::Path, args: &[String], cwd: &std::path::Path, prompt: &str, env: &[(&str, &str)], timeout: Duration) -> Result<String, String> {
    let _ = std::fs::create_dir_all(cwd);
    let mut cmd = Command::new(exe);
    cmd.args(args).current_dir(cwd).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    for (k, v) in env {
        cmd.env(k, v);
    }
    let mut child = cmd.spawn().map_err(|e| format!("cannot start {}: {e}", exe.display()))?;

    let mut stdin = child.stdin.take().expect("piped");
    let input = prompt.to_owned();
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(input.as_bytes());
    });
    let mut out = child.stdout.take().expect("piped");
    let mut err = child.stderr.take().expect("piped");
    let (out_tx, out_rx) = std::sync::mpsc::channel();
    let (err_tx, err_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut s = String::new();
        let _ = out.read_to_string(&mut s);
        let _ = out_tx.send(s);
    });
    std::thread::spawn(move || {
        let mut s = String::new();
        let _ = err.read_to_string(&mut s);
        let _ = err_tx.send(s);
    });

    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() >= deadline => {
                kill_tree(&mut child);
                return Err(format!("claude timed out after {timeout:?}"));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(100)),
            Err(e) => return Err(format!("waiting for claude: {e}")),
        }
    };
    let _ = writer.join();
    // Bounded: a grandchild that inherited the pipes could keep them open after exit.
    let stdout = out_rx.recv_timeout(Duration::from_secs(5)).unwrap_or_default();
    let stderr = err_rx.recv_timeout(Duration::from_secs(5)).unwrap_or_default();
    if !status.success() {
        let tail: String = stderr.trim().chars().rev().take(300).collect::<Vec<_>>().into_iter().rev().collect();
        return Err(format!("claude exited with {status}: {tail}"));
    }
    parse_output(&stdout)
}

/// Kills the child and everything it started. `Child::kill` ends only the direct child: with a
/// `claude.cmd` shim that is cmd.exe, and the real CLI underneath would keep running (and
/// billing) and keep the output pipes open.
fn kill_tree(child: &mut std::process::Child) {
    let _ = Command::new("taskkill")
        .args(["/T", "/F", "/PID", &child.id().to_string()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    let _ = child.kill();
    let _ = child.wait();
}

/// `--output-format json` prints one object with `result` and `is_error`.
fn parse_output(stdout: &str) -> Result<String, String> {
    let v: serde_json::Value = serde_json::from_str(stdout.trim()).map_err(|e| format!("unparseable claude output ({e}): {:.200}", stdout.trim()))?;
    let result = v.get("result").and_then(|r| r.as_str()).unwrap_or_default().to_owned();
    if v.get("is_error").and_then(|b| b.as_bool()).unwrap_or(false) {
        return Err(format!("claude reported an error: {result:.300}"));
    }
    if result.trim().is_empty() {
        return Err("claude returned an empty result".into());
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in `claude`: a batch file with the given body (Windows runs .cmd via cmd.exe).
    fn fake(dir: &tempfile::TempDir, body: &str) -> ClaudeCli {
        let exe = dir.path().join("fake-claude.cmd");
        std::fs::write(&exe, format!("@echo off\r\n{body}\r\n")).unwrap();
        ClaudeCli::new(exe, "haiku", dir.path().join("work"), Duration::from_secs(10))
    }

    #[test]
    fn parses_successful_json_result() {
        let json = "{\"is_error\":false,\"result\":\"## Goal\\n- x\"}";
        assert_eq!(parse_output(json).unwrap(), "## Goal\n- x");
    }

    #[test]
    fn error_flag_empty_result_and_garbage_are_errors() {
        assert!(parse_output(r#"{"is_error":true,"result":"Credit balance too low"}"#).unwrap_err().contains("Credit"));
        assert!(parse_output(r#"{"is_error":false,"result":"  "}"#).is_err());
        assert!(parse_output("Error: not logged in").unwrap_err().contains("not logged in"));
    }

    #[test]
    fn arguments_lock_the_run_down() {
        let c = ClaudeCli::new("claude".into(), "haiku", ".".into(), Duration::from_secs(1));
        let a = c.args();
        for flag in ["-p", "--no-session-persistence", "--strict-mcp-config", "--tools", "--setting-sources", "--system-prompt"] {
            assert!(a.iter().any(|x| x == flag), "{flag}");
        }
        let tools = a.iter().position(|x| x == "--tools").unwrap();
        assert_eq!(a[tools + 1], "", "no tools at all");
        assert_eq!(c.id(), "claude-cli/haiku");
    }

    #[test]
    fn runs_the_executable_and_reads_its_json() {
        let dir = tempfile::tempdir().unwrap();
        let c = fake(&dir, "echo {\"is_error\":false,\"result\":\"## Goal - fake\"}");
        assert_eq!(c.summarize("prompt", Duration::from_secs(10)).unwrap(), "## Goal - fake");
        assert!(dir.path().join("work").is_dir(), "runs inside its own working directory");
    }

    #[test]
    fn summarizer_guard_env_is_set_for_the_child() {
        let dir = tempfile::tempdir().unwrap();
        let c = fake(&dir, r#"echo {"is_error":false,"result":"guard=%CTX_SUMMARIZER%"}"#);
        assert_eq!(c.summarize("p", Duration::from_secs(10)).unwrap(), "guard=1");
    }

    #[test]
    fn non_zero_exit_is_an_error_with_stderr() {
        let dir = tempfile::tempdir().unwrap();
        let c = fake(&dir, "echo not logged in 1>&2\r\nexit /b 1");
        let e = c.summarize("p", Duration::from_secs(10)).unwrap_err();
        assert!(e.contains("not logged in"), "{e}");
    }

    /// PIDs of running processes with this image name.
    fn pids(image: &str) -> Vec<u32> {
        let out = Command::new("tasklist").args(["/FI", &format!("IMAGENAME eq {image}"), "/FO", "CSV", "/NH"]).output().unwrap();
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter_map(|l| l.split(',').nth(1)?.trim_matches('"').parse().ok())
            .collect()
    }

    #[test]
    fn hanging_child_and_its_grandchildren_are_killed_at_the_timeout() {
        let dir = tempfile::tempdir().unwrap();
        // cmd.exe (the child) runs PING.EXE (a grandchild), like a claude.cmd shim runs the CLI.
        let c = fake(&dir, "ping -n 60 127.0.0.1 >nul");
        let before = pids("PING.EXE");
        let start = Instant::now();
        let err = c.summarize("p", Duration::from_millis(1500)).unwrap_err();
        assert!(err.contains("timed out"), "{err}");
        assert!(start.elapsed() < Duration::from_secs(10), "returned in {:?}", start.elapsed());
        std::thread::sleep(Duration::from_millis(500));
        let leaked: Vec<u32> = pids("PING.EXE").into_iter().filter(|p| !before.contains(p)).collect();
        assert!(leaked.is_empty(), "grandchild survived the timeout: {leaked:?}");
    }

    #[test]
    fn caller_timeout_caps_the_configured_one() {
        let dir = tempfile::tempdir().unwrap();
        let c = fake(&dir, "ping -n 60 127.0.0.1 >nul");
        let start = Instant::now();
        assert!(c.summarize("p", Duration::from_millis(800)).unwrap_err().contains("800ms"));
        assert!(start.elapsed() < Duration::from_secs(6));
    }

    #[test]
    fn missing_executable_is_a_clear_error() {
        let c = ClaudeCli::new("Z:/no/such/claude.exe".into(), "haiku", std::env::temp_dir(), Duration::from_secs(1));
        assert!(c.summarize("p", Duration::from_secs(10)).unwrap_err().contains("cannot start"));
    }

    #[test]
    #[ignore = "calls the real claude CLI (subscription, ~10-25 s)"]
    fn real_claude_writes_a_valid_summary() {
        let c = ClaudeCli::new("claude".into(), "haiku", std::env::temp_dir().join("ctx-summarizer-test"), Duration::from_secs(120));
        let prompt = ingest::summary::prompt("Qdrant choice", "User: Qdrant or pgvector?\nAssistant: Qdrant for native hybrid search.");
        let raw = c.summarize(&prompt, Duration::from_secs(120)).unwrap();
        let body = ingest::summary::validate(&raw).unwrap();
        assert!(body.contains("## Decisions"), "{body}");
        assert!(!body.to_lowercase().contains("memory"), "no chatter: {body}");
    }
}
