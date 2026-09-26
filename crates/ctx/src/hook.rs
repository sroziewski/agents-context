//! `ctx hook`: what agent hooks run. It asks the daemon to index and returns at once.
//!
//! A hook runs inside the agent's own lifecycle — at session end, Claude Code is exiting and
//! tears down its child processes — so the actual indexing happens in the daemon, which lives
//! outside any agent session. The hook must never block or fail the session: every error is
//! swallowed and the next hook or the daemon's periodic run catches up.

use std::io::Read;
use std::time::Duration;

use crate::queue::Target;

/// Set in the environment of `claude -p` summarizer runs so their own hooks are no-ops.
pub const SUMMARIZER_ENV: &str = "CTX_SUMMARIZER";

#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    Requested,
    SkippedSummarizer,
    DaemonUnavailable(String),
}

pub fn notify(embed_url: &str, target: Target, timeout: Duration, env: impl Fn(&str) -> Option<String>) -> Outcome {
    if env(SUMMARIZER_ENV).is_some() {
        return Outcome::SkippedSummarizer;
    }
    // Windows retries a refused localhost connect for ~2 s; with the daemon down that would be
    // added to every single turn, so connecting gets a much shorter budget than the request.
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(timeout))
        .timeout_connect(Some(timeout.min(Duration::from_millis(300))))
        .build()
        .into();
    let url = format!("{}/index", embed_url.trim_end_matches('/'));
    match agent.post(&url).send_json(serde_json::json!({ "source": target })) {
        Ok(_) => Outcome::Requested,
        Err(e) => Outcome::DaemonUnavailable(e.to_string()),
    }
}

/// Runs an index through the daemon and waits for it, so a manual `ctx index` never races the
/// daemon's own runs over the same state DB and points. `None` when no daemon with an index
/// queue is reachable; the caller then indexes in-process.
pub fn delegate_index(embed_url: &str, target: Target, wait: Duration) -> anyhow::Result<Option<crate::queue::Status>> {
    let base = embed_url.trim_end_matches('/');
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(5)))
        .timeout_connect(Some(Duration::from_millis(300)))
        .build()
        .into();
    let status = |agent: &ureq::Agent| -> Option<crate::queue::Status> {
        agent.get(&format!("{base}/index/status")).call().ok()?.body_mut().read_json().ok()
    };
    let Some(before) = status(&agent) else { return Ok(None) };
    agent.post(&format!("{base}/index")).send_json(serde_json::json!({ "source": target }))?;
    let deadline = std::time::Instant::now() + wait;
    while std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(250));
        if let Some(s) = status(&agent).filter(|s| s.runs > before.runs && !s.running) {
            return Ok(Some(s));
        }
    }
    anyhow::bail!("daemon accepted the request but did not finish within {wait:?}; check `/index/status`")
}

/// Drains the hook payload from stdin. Claude Code writes JSON there and may block if the pipe
/// is never read; its content is not needed because the daemon re-scans incrementally.
/// Skipped on a console: nothing would ever close it and the hook would hang until Claude's
/// hook timeout.
pub fn drain_stdin() {
    use std::io::IsTerminal;
    if std::io::stdin().is_terminal() {
        return;
    }
    let mut sink = Vec::new();
    let _ = std::io::stdin().take(1 << 20).read_to_end(&mut sink);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed_server::{SharedEmbedder, service_router};
    use crate::queue::{IndexQueue, Runner};
    use embed::HashEmbedder;
    use std::sync::{Arc, Mutex};

    #[tokio::test(flavor = "multi_thread")]
    async fn hook_reaches_the_daemon_queue() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let s = seen.clone();
        let runner: Runner = Arc::new(move |t| {
            s.lock().unwrap().push(t);
            Box::pin(async { Ok(String::new()) })
        });
        let queue = IndexQueue::start(runner, Duration::from_millis(10));
        let app = service_router(SharedEmbedder::new(Box::new(HashEmbedder::new(8))), "m", Some(queue.clone()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let outcome = tokio::task::spawn_blocking(move || notify(&url, Target::Claude, Duration::from_secs(2), |_| None))
            .await
            .unwrap();
        assert_eq!(outcome, Outcome::Requested);
        for _ in 0..100 {
            if queue.status().runs == 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(*seen.lock().unwrap(), [Target::Claude]);
    }

    #[test]
    fn unreachable_daemon_returns_quickly_without_error() {
        let start = std::time::Instant::now();
        let outcome = notify("http://127.0.0.1:9", Target::Claude, Duration::from_secs(2), |_| None);
        assert!(matches!(outcome, Outcome::DaemonUnavailable(_)), "{outcome:?}");
        // Well under the 2 s Windows connect-retry: the short connect timeout must apply.
        assert!(start.elapsed() < Duration::from_millis(1500), "took {:?}", start.elapsed());
    }

    #[test]
    fn summarizer_runs_do_not_trigger_indexing() {
        let outcome = notify("http://127.0.0.1:9", Target::Claude, Duration::from_secs(2), |k| {
            (k == SUMMARIZER_ENV).then(|| "1".to_owned())
        });
        assert_eq!(outcome, Outcome::SkippedSummarizer);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn daemon_without_queue_rejects_index_requests() {
        let app = crate::embed_server::router(Box::new(HashEmbedder::new(8)), "m");
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let outcome = tokio::task::spawn_blocking(move || notify(&url, Target::All, Duration::from_secs(2), |_| None))
            .await
            .unwrap();
        assert!(matches!(outcome, Outcome::DaemonUnavailable(e) if e.contains("404")));
    }
}
