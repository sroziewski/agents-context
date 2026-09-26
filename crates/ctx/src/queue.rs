//! Background index runs inside the daemon, fed by agent hooks and a timer.
//!
//! Hooks fire in bursts (every answer, compaction, session end), so requests are coalesced: a
//! request waits `debounce` for others to join it, and everything that arrives while a run is in
//! progress becomes exactly one follow-up run. One run at a time — they share the GPU and the
//! state DB.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::Sources;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Target {
    Claude,
    Opencode,
    All,
}

impl Target {
    fn merge(self, other: Target) -> Target {
        if self == other { self } else { Target::All }
    }

    pub fn sources(self) -> Sources {
        match self {
            Target::Claude => Sources::Claude,
            Target::Opencode => Sources::OpenCode,
            Target::All => Sources::All,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Status {
    pub running: bool,
    pub runs: u64,
    pub last_target: Option<Target>,
    /// One-line report of the last successful run.
    pub last_report: Option<String>,
    pub last_error: Option<String>,
    pub last_finished_ms: Option<i64>,
}

pub type RunFuture = Pin<Box<dyn Future<Output = anyhow::Result<String>> + Send>>;
/// Performs one index run and returns its one-line report.
pub type Runner = Arc<dyn Fn(Target) -> RunFuture + Send + Sync>;

#[derive(Clone)]
pub struct IndexQueue {
    tx: mpsc::UnboundedSender<Target>,
    status: Arc<Mutex<Status>>,
}

impl IndexQueue {
    pub fn start(runner: Runner, debounce: Duration) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let status = Arc::new(Mutex::new(Status::default()));
        tokio::spawn(worker(rx, runner, debounce, status.clone()));
        Self { tx, status }
    }

    pub fn request(&self, target: Target) {
        // Only fails once the worker is gone, i.e. during shutdown.
        let _ = self.tx.send(target);
    }

    pub fn status(&self) -> Status {
        self.status.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// Requests `Target::All` every `every`, so sources without hooks still get indexed.
    pub fn schedule(&self, every: Duration) {
        let q = self.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(every);
            tick.tick().await; // the first tick fires immediately; the startup run is separate
            loop {
                tick.tick().await;
                q.request(Target::All);
            }
        });
    }
}

async fn worker(mut rx: mpsc::UnboundedReceiver<Target>, runner: Runner, debounce: Duration, status: Arc<Mutex<Status>>) {
    while let Some(first) = rx.recv().await {
        let mut target = first;
        tokio::time::sleep(debounce).await;
        while let Ok(more) = rx.try_recv() {
            target = target.merge(more);
        }
        set(&status, |s| s.running = true);
        // Its own task, so a panicking run (sync or async part) becomes an error on this run
        // instead of killing the worker and leaving every later request silently dropped.
        let run = runner.clone();
        let result = match tokio::spawn(async move { run(target).await }).await {
            Ok(r) => r,
            Err(e) => Err(anyhow::anyhow!("index run panicked: {e}")),
        };
        let now = jiff::Timestamp::now().as_millisecond();
        set(&status, |s| {
            s.running = false;
            s.runs += 1;
            s.last_target = Some(target);
            s.last_finished_ms = Some(now);
            match result {
                Ok(report) => {
                    s.last_report = Some(report);
                    s.last_error = None;
                }
                Err(e) => s.last_error = Some(format!("{e:#}")),
            }
        });
    }
}

fn set(status: &Mutex<Status>, f: impl FnOnce(&mut Status)) {
    f(&mut status.lock().unwrap_or_else(|p| p.into_inner()));
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::Notify;

    /// Records every run's target; each run blocks until `release` is notified when gated.
    fn recording(gate: Option<Arc<Notify>>) -> (Runner, Arc<Mutex<Vec<Target>>>) {
        let runs = Arc::new(Mutex::new(Vec::new()));
        let r = runs.clone();
        let runner: Runner = Arc::new(move |t| {
            let r = r.clone();
            let gate = gate.clone();
            Box::pin(async move {
                r.lock().unwrap().push(t);
                if let Some(g) = gate {
                    g.notified().await;
                }
                Ok(format!("ran {t:?}"))
            })
        });
        (runner, runs)
    }

    async fn settle(q: &IndexQueue, runs: u64) {
        for _ in 0..200 {
            if q.status().runs >= runs && !q.status().running {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("queue did not settle: {:?}", q.status());
    }

    #[test]
    fn merge_widens_to_all() {
        assert_eq!(Target::Claude.merge(Target::Claude), Target::Claude);
        assert_eq!(Target::Claude.merge(Target::Opencode), Target::All);
        assert_eq!(Target::All.merge(Target::Claude), Target::All);
    }

    #[test]
    fn target_serialises_lowercase() {
        assert_eq!(serde_json::to_string(&Target::Opencode).unwrap(), "\"opencode\"");
        assert_eq!(serde_json::from_str::<Target>("\"claude\"").unwrap(), Target::Claude);
    }

    #[tokio::test]
    async fn burst_of_requests_becomes_one_run() {
        let (runner, runs) = recording(None);
        let q = IndexQueue::start(runner, Duration::from_millis(50));
        for _ in 0..10 {
            q.request(Target::Claude);
        }
        settle(&q, 1).await;
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(*runs.lock().unwrap(), [Target::Claude]);
        assert_eq!(q.status().last_report.as_deref(), Some("ran Claude"));
    }

    #[tokio::test]
    async fn mixed_sources_in_one_window_run_as_all() {
        let (runner, runs) = recording(None);
        let q = IndexQueue::start(runner, Duration::from_millis(50));
        q.request(Target::Claude);
        q.request(Target::Opencode);
        settle(&q, 1).await;
        assert_eq!(*runs.lock().unwrap(), [Target::All]);
    }

    #[tokio::test]
    async fn requests_during_a_run_cause_exactly_one_follow_up() {
        let gate = Arc::new(Notify::new());
        let (runner, runs) = recording(Some(gate.clone()));
        let q = IndexQueue::start(runner, Duration::from_millis(20));
        q.request(Target::Claude);
        while !q.status().running {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        for _ in 0..5 {
            q.request(Target::Claude);
        }
        gate.notify_one();
        // Wait for the follow-up to start, then release it.
        while runs.lock().unwrap().len() < 2 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        gate.notify_one();
        settle(&q, 2).await;
        tokio::time::sleep(Duration::from_millis(80)).await;
        assert_eq!(runs.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn failed_run_is_reported_and_the_queue_keeps_working() {
        let calls = Arc::new(Mutex::new(0));
        let c = calls.clone();
        let runner: Runner = Arc::new(move |_| {
            let c = c.clone();
            Box::pin(async move {
                let mut n = c.lock().unwrap();
                *n += 1;
                if *n == 1 { Err(anyhow::anyhow!("qdrant down")) } else { Ok("ok".into()) }
            })
        });
        let q = IndexQueue::start(runner, Duration::from_millis(10));
        q.request(Target::All);
        settle(&q, 1).await;
        assert_eq!(q.status().last_error.as_deref(), Some("qdrant down"));
        q.request(Target::All);
        settle(&q, 2).await;
        let s = q.status();
        assert!(s.last_error.is_none());
        assert_eq!(s.last_report.as_deref(), Some("ok"));
    }

    #[tokio::test]
    async fn panicking_run_is_an_error_and_the_worker_survives() {
        let calls = Arc::new(Mutex::new(0));
        let c = calls.clone();
        let runner: Runner = Arc::new(move |_| {
            let first = {
                let mut n = c.lock().unwrap();
                *n += 1;
                *n == 1
            };
            if first {
                panic!("boom in the synchronous part");
            }
            Box::pin(async { Ok("recovered".into()) })
        });
        let q = IndexQueue::start(runner, Duration::from_millis(10));
        q.request(Target::All);
        settle(&q, 1).await;
        let s = q.status();
        assert!(!s.running, "running must be cleared after a panic");
        assert!(s.last_error.as_deref().unwrap().contains("panicked"), "{s:?}");
        q.request(Target::All);
        settle(&q, 2).await;
        assert_eq!(q.status().last_report.as_deref(), Some("recovered"));
    }

    #[tokio::test]
    async fn panic_inside_the_async_part_is_also_contained() {
        let runner: Runner = Arc::new(|_| Box::pin(async { panic!("boom after await") }));
        let q = IndexQueue::start(runner, Duration::from_millis(10));
        q.request(Target::Claude);
        settle(&q, 1).await;
        assert!(q.status().last_error.unwrap().contains("panicked"));
    }

    #[tokio::test(start_paused = true)]
    async fn schedule_requests_all_periodically() {
        let (runner, runs) = recording(None);
        let q = IndexQueue::start(runner, Duration::from_millis(1));
        q.schedule(Duration::from_secs(900));
        tokio::time::sleep(Duration::from_secs(1)).await;
        assert!(runs.lock().unwrap().is_empty(), "no run before the first interval");
        tokio::time::sleep(Duration::from_secs(900)).await;
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert_eq!(*runs.lock().unwrap(), [Target::All]);
    }
}
