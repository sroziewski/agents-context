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
    // The first launch of a freshly built exe includes the antivirus scan (seconds on Windows);
    // that is a one-off per build, not hook latency, so it is excluded from the measurement.
    ctx().arg("--help").output().unwrap();
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

async fn mock_daemon(delay: Duration) -> (String, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let seen = calls.clone();
    let app = axum::Router::new().route(
        "/context",
        axum::routing::post(move || {
            seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            async move {
                tokio::time::sleep(delay).await;
                axum::Json(serde_json::json!({"context": "session mock: the fix"}))
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (url, calls)
}

fn run_inject_hook(url: &str) -> (std::process::Output, Duration) {
    use std::io::Write;
    ctx().arg("--help").output().unwrap();
    let start = Instant::now();
    let mut child = Command::new(env!("CARGO_BIN_EXE_ctx"))
        .args(["hook", "--inject"])
        .env("CTX_EMBED_URL", url)
        .env("CTX_INJECT_CLAUDE", "on")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(br#"{"session_id":"s","cwd":"D:\\x","prompt":"how did we fix the directml gpu hang timeout","hook_event_name":"UserPromptSubmit"}"#)
        .unwrap();
    let out = child.wait_with_output().unwrap();
    (out, start.elapsed())
}

#[tokio::test(flavor = "multi_thread")]
async fn inject_hook_prints_context_from_a_responsive_daemon() {
    let (url, calls) = mock_daemon(Duration::from_millis(50)).await;
    let (out, _) = tokio::task::spawn_blocking(move || run_inject_hook(&url)).await.unwrap();
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains(r#""additionalContext":"session mock: the fix""#), "{stdout}");
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn inject_hook_gives_up_on_a_slow_daemon_within_its_budget() {
    let (url, calls) = mock_daemon(Duration::from_secs(10)).await;
    let (out, took) = tokio::task::spawn_blocking(move || run_inject_hook(&url)).await.unwrap();
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(out.stdout.is_empty(), "{}", String::from_utf8_lossy(&out.stdout));
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1, "the call was made, then abandoned");
    assert!(took < ctx::hook::INJECT_BUDGET + Duration::from_millis(700), "the whole process stays near the 3 s budget: {took:?}");
}
