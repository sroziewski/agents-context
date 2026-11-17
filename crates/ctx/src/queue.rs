//! Background index runs inside the daemon, fed by agent hooks and a timer.
//!
//! Hooks fire in bursts (every answer, compaction, session end), so requests are coalesced: a
//! request waits `debounce` for others to join it, and everything that arrives while a run is in
//! progress becomes exactly one follow-up run. One run at a time — they share the GPU and the
//! state DB.

use std::future::Future;
use std::pin::Pin;
use std::collections::{BTreeSet, VecDeque};
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
    Teams,
    /// Jira only. Deliberately not part of `All`: Jira is indexed on its own twice-daily
    /// schedule, not on the frequent session/Teams runs.
    Jira,
    /// Confluence only; like Jira, on its own schedule and never part of `All`.
    Confluence,
    All,
}

impl Target {
    fn merge(self, other: Target) -> Target {
        if self == other { self } else { Target::All }
    }

    /// Session sources this run covers; `None` for a Teams- or Jira-only run.
    pub fn sessions(self) -> Option<Sources> {
        match self {
            Target::Claude => Some(Sources::Claude),
            Target::Opencode => Some(Sources::OpenCode),
            Target::Teams | Target::Jira | Target::Confluence => None,
            Target::All => Some(Sources::All),
        }
    }

    pub fn includes_teams(self) -> bool {
        matches!(self, Target::Teams | Target::All)
    }

    /// Jira runs only when explicitly targeted (the twice-daily scheduler or `ctx index
    /// --source jira`), never as part of `All`.
    pub fn includes_jira(self) -> bool {
        matches!(self, Target::Jira)
    }

    pub fn includes_confluence(self) -> bool {
        matches!(self, Target::Confluence)
    }

    /// Targets that always run on their own instead of merging with others into `All`.
    fn standalone(self) -> bool {
        matches!(self, Target::Jira | Target::Confluence)
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

/// How the run that served one request ended.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Outcome {
    pub report: Option<String>,
    pub error: Option<String>,
}

/// Where one request stands. `Unknown` is a ticket this queue never issued (another daemon
/// instance did) or one old enough that its outcome is no longer kept.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "state", rename_all = "lowercase")]
pub enum Ticket {
    Pending,
    Done(Outcome),
    Unknown,
}

/// Finished runs whose outcome is kept for lookup; a waiting caller polls within milliseconds
/// of its run ending. Counted in runs, so every request one run served stays answerable however
/// many were coalesced into it.
const KEPT_RUNS: usize = 256;

#[derive(Default)]
struct Tickets {
    issued: u64,
    /// Issued and not yet served by a finished run.
    waiting: BTreeSet<u64>,
    /// The newest finished runs, oldest first: the tickets each served and how it ended.
    finished: VecDeque<(Vec<u64>, Outcome)>,
}

impl Tickets {
    fn issue(&mut self) -> u64 {
        self.issued += 1;
        self.waiting.insert(self.issued);
        self.issued
    }

    fn finish(&mut self, tickets: &[u64], outcome: Outcome) {
        for t in tickets {
            self.waiting.remove(t);
        }
        self.finished.push_back((tickets.to_vec(), outcome));
        while self.finished.len() > KEPT_RUNS {
            self.finished.pop_front();
        }
    }

    fn state(&self, ticket: u64) -> Ticket {
        if self.waiting.contains(&ticket) {
            return Ticket::Pending;
        }
        self.finished.iter().rev().find(|(served, _)| served.contains(&ticket)).map_or(Ticket::Unknown, |(_, outcome)| Ticket::Done(outcome.clone()))
    }
}

pub type RunFuture = Pin<Box<dyn Future<Output = anyhow::Result<String>> + Send>>;
/// Performs one index run and returns its one-line report.
pub type Runner = Arc<dyn Fn(Target) -> RunFuture + Send + Sync>;

#[derive(Clone)]
pub struct IndexQueue {
    tx: mpsc::UnboundedSender<(Target, u64)>,
    status: Arc<Mutex<Status>>,
    tickets: Arc<Mutex<Tickets>>,
    epoch: u64,
}

impl IndexQueue {
    pub fn start(runner: Runner, debounce: Duration) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let status = Arc::new(Mutex::new(Status::default()));
        let tickets = Arc::new(Mutex::new(Tickets::default()));
        tokio::spawn(worker(rx, runner, debounce, status.clone(), tickets.clone()));
        Self { tx, status, tickets, epoch: jiff::Timestamp::now().as_millisecond() as u64 }
    }

    /// Queues a run for `target` and returns the ticket to ask [`Self::ticket`] about.
    pub fn request(&self, target: Target) -> u64 {
        let ticket = self.tickets.lock().unwrap_or_else(|p| p.into_inner()).issue();
        // Only fails once the worker is gone, i.e. during shutdown.
        if self.tx.send((target, ticket)).is_err() {
            self.tickets.lock().unwrap_or_else(|p| p.into_inner()).waiting.remove(&ticket);
        }
        ticket
    }

    /// Identifies this queue instance: tickets restart from 1 with every daemon start, so a
    /// ticket only means something together with the epoch it was issued under.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// The state of a ticket issued under `epoch`. It is `Done` only once a run that started
    /// after the request has finished, with that run's outcome.
    pub fn ticket(&self, epoch: u64, ticket: u64) -> Ticket {
        if epoch != self.epoch {
            return Ticket::Unknown;
        }
        self.tickets.lock().unwrap_or_else(|p| p.into_inner()).state(ticket)
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

    /// Requests `target` at each given local `(hour, minute)` — for Jira's twice-daily index.
    /// Sleeps until the next slot, fires, repeats; robust across days and DST via jiff.
    pub fn schedule_daily(&self, target: Target, times: Vec<(u32, u32)>) {
        if times.is_empty() {
            return;
        }
        let q = self.clone();
        tokio::spawn(async move {
            loop {
                let wait = duration_until_next(&times, jiff::Zoned::now());
                tokio::time::sleep(wait).await;
                q.request(target);
                // Guard against firing twice within the same minute after a near-zero sleep.
                tokio::time::sleep(Duration::from_secs(61)).await;
            }
        });
    }
}

/// Time from `now` to the next `(hour, minute)` slot in `now`'s time zone.
fn duration_until_next(times: &[(u32, u32)], now: jiff::Zoned) -> Duration {
    let (date, tz, now_ts) = (now.date(), now.time_zone().clone(), now.timestamp());
    let next = times
        .iter()
        .flat_map(|&(h, m)| {
            // today and tomorrow, so a slot earlier today rolls to tomorrow.
            let tz = tz.clone();
            [0i64, 1].into_iter().filter_map(move |day| {
                date.checked_add(jiff::Span::new().days(day)).ok()?.at(h as i8, m as i8, 0, 0).to_zoned(tz.clone()).ok()
            })
        })
        .map(|z| z.timestamp())
        .filter(|&t| t > now_ts)
        .min();
    match next {
        Some(t) => Duration::from_millis((t.as_millisecond() - now_ts.as_millisecond()).max(0) as u64),
        None => Duration::from_secs(3600),
    }
}

async fn worker(
    mut rx: mpsc::UnboundedReceiver<(Target, u64)>,
    runner: Runner,
    debounce: Duration,
    status: Arc<Mutex<Status>>,
    tickets: Arc<Mutex<Tickets>>,
) {
    while let Some(first) = rx.recv().await {
        tokio::time::sleep(debounce).await;
        // Drain the debounce window. Jira is kept separate from the session/Teams merge: merging
        // it into `All` would drop it (All excludes Jira), so a Jira request coalesced with an
        // All request would silently never run. Instead each side is its own run.
        let mut sessions_teams: Option<(Target, Vec<u64>)> = None;
        let mut standalone: Vec<(Target, Vec<u64>)> = Vec::new();
        let mut consider = |(t, ticket): (Target, u64)| {
            if t.standalone() {
                match standalone.iter_mut().find(|(s, _)| *s == t) {
                    Some((_, served)) => served.push(ticket),
                    None => standalone.push((t, vec![ticket])),
                }
            } else {
                let (acc, mut served) = sessions_teams.take().map_or((t, Vec::new()), |(acc, served)| (acc.merge(t), served));
                served.push(ticket);
                sessions_teams = Some((acc, served));
            }
        };
        consider(first);
        while let Ok(more) = rx.try_recv() {
            consider(more);
        }
        for (target, served) in sessions_teams.into_iter().chain(standalone) {
            run_and_record(&runner, target, &status, &tickets, &served).await;
        }
    }
}

async fn run_and_record(runner: &Runner, target: Target, status: &Arc<Mutex<Status>>, tickets: &Mutex<Tickets>, served: &[u64]) {
    set(status, |s| s.running = true);
    // Its own task, so a panicking run (sync or async part) becomes an error on this run
    // instead of killing the worker and leaving every later request silently dropped.
    let run = runner.clone();
    let result = match tokio::spawn(async move { run(target).await }).await {
        Ok(r) => r,
        Err(e) => Err(anyhow::anyhow!("index run panicked: {e}")),
    };
    let now = jiff::Timestamp::now().as_millisecond();
    let outcome = match &result {
        Ok(report) => Outcome { report: Some(report.clone()), error: None },
        Err(e) => Outcome { report: None, error: Some(format!("{e:#}")) },
    };
    set(status, |s| {
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
    tickets.lock().unwrap_or_else(|p| p.into_inner()).finish(served, outcome);
}

fn set(status: &Mutex<Status>, f: impl FnOnce(&mut Status)) {
    f(&mut status.lock().unwrap_or_else(|p| p.into_inner()));
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::Notify;

    #[test]
    fn next_daily_slot_is_computed_across_the_day() {
        let times = [(10, 0), (17, 0)];
        let at = |s: &str| jiff::Zoned::strptime("%Y-%m-%dT%H:%M:%S[%Q]", s).unwrap();
        let hours = |z: jiff::Zoned| duration_until_next(&times, z).as_secs_f64() / 3600.0;
        assert!((hours(at("2026-09-28T09:00:00[Europe/Warsaw]")) - 1.0).abs() < 0.01, "09:00 → 10:00");
        assert!((hours(at("2026-09-28T12:00:00[Europe/Warsaw]")) - 5.0).abs() < 0.01, "12:00 → 17:00");
        assert!((hours(at("2026-09-28T18:00:00[Europe/Warsaw]")) - 16.0).abs() < 0.01, "18:00 → next 10:00");
        assert_eq!(duration_until_next(&[], at("2026-09-28T09:00:00[Europe/Warsaw]")).as_secs(), 3600, "no times → idle hour");
    }

    #[test]
    fn jira_target_is_isolated_from_all() {
        assert!(Target::Jira.includes_jira() && !Target::All.includes_jira());
        assert!(!Target::Jira.includes_teams());
        assert!(Target::Jira.sessions().is_none());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn all_and_jira_in_one_window_both_run() {
        let (runner, runs) = recording(None);
        let q = IndexQueue::start(runner, Duration::from_millis(30));
        q.request(Target::All);
        q.request(Target::Jira); // would be lost if merged into All
        settle(&q, 2).await;
        let got = runs.lock().unwrap().clone();
        assert!(got.contains(&Target::All) && got.contains(&Target::Jira), "both ran: {got:?}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn confluence_and_jira_each_run_once_beside_all() {
        let (runner, runs) = recording(None);
        let q = IndexQueue::start(runner, Duration::from_millis(30));
        q.request(Target::Confluence);
        q.request(Target::All);
        q.request(Target::Jira);
        q.request(Target::Confluence);
        settle(&q, 3).await;
        tokio::time::sleep(Duration::from_millis(80)).await;
        assert_eq!(*runs.lock().unwrap(), [Target::All, Target::Confluence, Target::Jira]);
        assert!(Target::Confluence.includes_confluence() && Target::Confluence.sessions().is_none());
        assert!(!Target::All.includes_confluence() && !Target::Confluence.includes_jira() && !Target::Confluence.includes_teams());
    }

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
    fn teams_runs_cover_no_sessions_and_all_covers_everything() {
        assert_eq!(Target::Teams.sessions(), None);
        assert!(Target::Teams.includes_teams() && Target::All.includes_teams());
        assert!(!Target::Claude.includes_teams());
        assert_eq!(Target::Teams.merge(Target::Claude), Target::All);
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

    /// Run `n` reports `run {n} {target:?}`, so an outcome names the run it came from. Returns
    /// the number of runs started; with a gate every run blocks until it is notified.
    fn numbered(gate: Option<Arc<Notify>>) -> (Runner, Arc<Mutex<u64>>) {
        let started = Arc::new(Mutex::new(0u64));
        let s = started.clone();
        let runner: Runner = Arc::new(move |t| {
            let s = s.clone();
            let gate = gate.clone();
            Box::pin(async move {
                let n = {
                    let mut n = s.lock().unwrap();
                    *n += 1;
                    *n
                };
                if let Some(g) = gate {
                    g.notified().await;
                }
                Ok(format!("run {n} {t:?}"))
            })
        });
        (runner, started)
    }

    async fn until(what: &str, cond: impl Fn() -> bool) {
        for _ in 0..1000 {
            if cond() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        panic!("never happened: {what}");
    }

    fn done(report: &str) -> Ticket {
        Ticket::Done(Outcome { report: Some(report.to_owned()), error: None })
    }

    #[tokio::test]
    async fn a_request_made_during_a_run_is_pending_until_its_own_run_finishes() {
        let gate = Arc::new(Notify::new());
        let (runner, started) = numbered(Some(gate.clone()));
        let q = IndexQueue::start(runner, Duration::from_millis(20));
        let e = q.epoch();
        let first = q.request(Target::Claude);
        until("run 1 starts", || q.status().running).await;
        let second = q.request(Target::Opencode);
        assert_eq!((first, second), (1, 2), "tickets count up from 1");
        assert_eq!(q.ticket(e, first), Ticket::Pending);
        assert_eq!(q.ticket(e, second), Ticket::Pending);

        gate.notify_one();
        until("run 1 finishes", || q.status().runs == 1).await;
        assert_eq!(q.ticket(e, first), done("run 1 Claude"));
        assert_eq!(q.ticket(e, second), Ticket::Pending, "run 1 started before this request, so it does not answer it");

        until("run 2 starts", || *started.lock().unwrap() == 2).await;
        assert_eq!(q.ticket(e, second), Ticket::Pending, "its own run is still in progress");
        gate.notify_one();
        settle(&q, 2).await;
        assert_eq!(q.ticket(e, second), done("run 2 Opencode"));
        assert_eq!(q.ticket(e, first), done("run 1 Claude"), "a later run does not rewrite an earlier outcome");
    }

    #[tokio::test]
    async fn requests_in_one_window_get_the_outcome_of_the_run_that_served_them() {
        let started = Arc::new(Mutex::new(0u64));
        let s = started.clone();
        let runner: Runner = Arc::new(move |t| {
            let s = s.clone();
            Box::pin(async move {
                let n = {
                    let mut n = s.lock().unwrap();
                    *n += 1;
                    *n
                };
                if t == Target::Jira { Err(anyhow::anyhow!("jira unreachable in run {n}")) } else { Ok(format!("run {n} {t:?}")) }
            })
        });
        let q = IndexQueue::start(runner, Duration::from_millis(30));
        let e = q.epoch();
        // Jira and Confluence each asked for twice: the second request joins the first one's run.
        let jira_first = q.request(Target::Jira);
        let sessions = q.request(Target::Claude);
        let confluence_first = q.request(Target::Confluence);
        let (jira, confluence) = ([jira_first, q.request(Target::Jira)], [confluence_first, q.request(Target::Confluence)]);
        assert_eq!((jira, sessions, confluence), ([1, 4], 2, [3, 5]));
        settle(&q, 3).await;
        tokio::time::sleep(Duration::from_millis(80)).await;
        assert_eq!(*started.lock().unwrap(), 3, "one run for the sessions, one for Jira, one for Confluence");
        assert_eq!(q.ticket(e, sessions), done("run 1 Claude"));
        for t in jira {
            assert_eq!(q.ticket(e, t), Ticket::Done(Outcome { report: None, error: Some("jira unreachable in run 2".into()) }), "Jira ticket {t}");
        }
        for t in confluence {
            assert_eq!(q.ticket(e, t), done("run 3 Confluence"), "Confluence ticket {t}");
        }
    }

    #[tokio::test]
    async fn requests_coalesced_into_one_run_all_get_that_runs_outcome() {
        let (runner, started) = numbered(None);
        let q = IndexQueue::start(runner, Duration::from_millis(30));
        let e = q.epoch();
        let tickets = [q.request(Target::Claude), q.request(Target::Opencode), q.request(Target::Claude)];
        settle(&q, 1).await;
        tokio::time::sleep(Duration::from_millis(80)).await;
        assert_eq!(*started.lock().unwrap(), 1, "one run for the whole window");
        assert_eq!(tickets, [1, 2, 3]);
        for t in tickets {
            assert_eq!(q.ticket(e, t), done("run 1 All"), "ticket {t}");
        }
    }

    #[tokio::test]
    async fn a_failed_run_is_done_with_an_error_and_no_report() {
        let calls = Arc::new(Mutex::new(0));
        let c = calls.clone();
        let runner: Runner = Arc::new(move |_| {
            let c = c.clone();
            Box::pin(async move {
                let mut n = c.lock().unwrap();
                *n += 1;
                if *n == 2 { Err(anyhow::anyhow!("qdrant down")) } else { Ok(format!("ok {n}")) }
            })
        });
        let q = IndexQueue::start(runner, Duration::from_millis(10));
        let e = q.epoch();
        let ok = q.request(Target::All);
        settle(&q, 1).await;
        let failed = q.request(Target::All);
        settle(&q, 2).await;
        assert_eq!(q.status().last_report.as_deref(), Some("ok 1"), "the status still shows the last successful run");
        assert_eq!(q.ticket(e, failed), Ticket::Done(Outcome { report: None, error: Some("qdrant down".into()) }));
        assert_eq!(q.ticket(e, ok), done("ok 1"));
    }

    #[tokio::test]
    async fn a_panicking_run_is_done_with_an_error_and_no_report() {
        let runner: Runner = Arc::new(|_| Box::pin(async { panic!("boom after await") }));
        let q = IndexQueue::start(runner, Duration::from_millis(10));
        let t = q.request(Target::Claude);
        settle(&q, 1).await;
        let Ticket::Done(outcome) = q.ticket(q.epoch(), t) else { panic!("not done: {:?}", q.ticket(q.epoch(), t)) };
        assert_eq!(outcome.report, None);
        // The text after the prefix is tokio's JoinError (it carries a task id), so only the prefix is fixed.
        let error = outcome.error.expect("a panic is an error");
        assert!(error.starts_with("index run panicked: "), "{error}");
    }

    #[tokio::test]
    async fn tickets_this_queue_did_not_issue_are_unknown() {
        let (runner, _) = numbered(None);
        let q = IndexQueue::start(runner, Duration::from_millis(10));
        let e = q.epoch();
        assert_eq!(q.ticket(e, 1), Ticket::Unknown, "nothing issued yet");
        let issued = q.request(Target::Claude);
        settle(&q, 1).await;
        assert_eq!(q.ticket(e, issued), done("run 1 Claude"));
        for (epoch, ticket, why) in [
            (e, 0, "ticket 0 is never issued"),
            (e, issued + 1, "the next ticket has not been issued"),
            (e, u64::MAX, "far beyond the last ticket"),
            (e + 1, issued, "a later daemon's epoch"),
            (e - 1, issued, "an earlier daemon's epoch"),
            (0, issued, "epoch 0"),
        ] {
            assert_eq!(q.ticket(epoch, ticket), Ticket::Unknown, "{why}");
        }
    }

    #[tokio::test]
    async fn outcomes_are_kept_for_the_newest_256_finished_runs() {
        let (runner, _) = numbered(None);
        let q = IndexQueue::start(runner, Duration::from_millis(1));
        let e = q.epoch();
        for run in 1..=256 {
            assert_eq!(q.request(Target::Claude), run);
            settle(&q, run).await;
        }
        assert_eq!(q.ticket(e, 1), done("run 1 Claude"), "256 finished runs are all kept");

        assert_eq!(q.request(Target::Claude), 257);
        settle(&q, 257).await;
        assert_eq!(q.ticket(e, 1), Ticket::Unknown, "the 257th run pushes the oldest out");
        for run in 2..=257 {
            assert_eq!(q.ticket(e, run), done(&format!("run {run} Claude")), "run {run} is among the newest 256");
        }
    }

    #[tokio::test]
    async fn every_ticket_of_a_kept_run_stays_done_however_many_it_served() {
        let (runner, started) = numbered(None);
        let q = IndexQueue::start(runner, Duration::from_millis(30));
        let e = q.epoch();
        let tickets: Vec<u64> = (0..300).map(|_| q.request(Target::Claude)).collect();
        settle(&q, 1).await;
        tokio::time::sleep(Duration::from_millis(80)).await;
        assert_eq!(*started.lock().unwrap(), 1, "300 requests in one window are one run");
        assert_eq!((tickets[0], tickets[299]), (1, 300));
        for &t in &tickets {
            assert_eq!(q.ticket(e, t), done("run 1 Claude"), "ticket {t}");
        }

        let later = q.request(Target::Opencode);
        settle(&q, 2).await;
        assert_eq!(q.ticket(e, later), done("run 2 Opencode"));
        assert_eq!(q.ticket(e, tickets[0]), done("run 1 Claude"), "a later run does not push out tickets of a kept run");
    }

    #[tokio::test]
    async fn a_ticket_stays_pending_behind_another_run_of_its_window_however_many_tickets_that_run_served() {
        let gate = Arc::new(Notify::new());
        let (runner, started) = numbered(Some(gate.clone()));
        let q = IndexQueue::start(runner, Duration::from_millis(30));
        let e = q.epoch();
        // One window: the oldest ticket is the Jira request, then 257 session requests. The
        // session run goes first; its finishing with more tickets than runs are kept must not
        // make the lower-numbered Jira ticket look forgotten while the Jira run is still to end.
        let jira = q.request(Target::Jira);
        let sessions: Vec<u64> = (0..257).map(|_| q.request(Target::Claude)).collect();
        until("the session run starts", || *started.lock().unwrap() == 1).await;
        assert_eq!(q.ticket(e, jira), Ticket::Pending, "the session run of its window is in progress");
        gate.notify_one();
        until("the Jira run starts", || *started.lock().unwrap() == 2).await;
        assert_eq!((q.ticket(e, sessions[0]), q.ticket(e, sessions[256])), (done("run 1 Claude"), done("run 1 Claude")));
        let while_running = q.ticket(e, jira);
        gate.notify_one();
        settle(&q, 2).await;
        assert_eq!(while_running, Ticket::Pending, "the Jira run was still in progress");
        assert_eq!(q.ticket(e, jira), done("run 2 Jira"));
    }

    #[test]
    fn a_request_made_after_the_worker_is_gone_is_unknown_not_pending_forever() {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let q = rt.block_on(async {
            let (runner, _) = numbered(None);
            let q = IndexQueue::start(runner, Duration::from_millis(10));
            let served = q.request(Target::Claude);
            settle(&q, 1).await;
            assert_eq!(q.ticket(q.epoch(), served), done("run 1 Claude"));
            q
        });
        // Dropping the runtime drops the worker task, as a daemon shutting down does.
        drop(rt);
        let late = q.request(Target::Claude);
        assert_eq!(late, 2);
        assert_eq!(q.ticket(q.epoch(), late), Ticket::Unknown, "no run will ever serve it");
        assert_eq!(q.ticket(q.epoch(), 1), done("run 1 Claude"), "what finished before stays answered");
    }
}
