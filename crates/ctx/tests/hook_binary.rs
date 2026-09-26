//! Exit codes of the real `ctx` binary as agent hooks see them. Claude Code treats exit code 2
//! from a Stop/PreCompact hook as a blocking error, so `ctx hook` must exit 0 no matter what.

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn ctx() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_ctx"));
    // A port nothing listens on: the daemon is "down".
    c.env("CTX_EMBED_URL", "http://127.0.0.1:9").stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::piped());
    c
}

#[test]
fn hook_with_daemon_down_exits_zero_quickly() {
    let start = Instant::now();
    let out = ctx().args(["hook", "--source", "claude"]).output().unwrap();
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(start.elapsed() < Duration::from_secs(2), "took {:?}", start.elapsed());
}

#[test]
fn hook_with_bad_arguments_still_exits_zero() {
    for args in [&["hook", "--source", "claud"][..], &["hook", "--bogus"], &["hook", "extra-positional"]] {
        let out = ctx().args(args).output().unwrap();
        assert_eq!(out.status.code(), Some(0), "{args:?}: {}", String::from_utf8_lossy(&out.stderr));
    }
}

#[test]
fn hook_reads_and_ignores_a_json_payload_on_stdin() {
    use std::io::Write;
    let mut child = ctx().args(["hook"]).stdin(Stdio::piped()).spawn().unwrap();
    child.stdin.take().unwrap().write_all(br#"{"session_id":"s","hook_event_name":"Stop"}"#).unwrap();
    assert_eq!(child.wait().unwrap().code(), Some(0));
}

#[test]
fn hook_help_still_works() {
    let out = ctx().args(["hook", "--help"]).stdout(Stdio::piped()).output().unwrap();
    assert_eq!(out.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&out.stdout).contains("re-index"));
}

#[test]
fn other_commands_keep_normal_usage_errors() {
    let out = ctx().args(["no-such-command"]).output().unwrap();
    assert_eq!(out.status.code(), Some(2), "usage errors outside hooks must stay visible");
}
