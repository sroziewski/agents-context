//! Notice when the Teams scraper has stopped and re-trigger it.
//!
//! ctx only reads `teams.db`; the scraper app (`D:\dev\my\teams`) fills it. When that scraper
//! stops, `teams.db` freezes and Teams search silently goes stale — there is no error on the ctx
//! side, because ctx is faithfully indexing everything that exists. The scraper's scheduled task
//! is Interactive and, when the session is not present, fails to launch the process at all (its
//! own log never even opens). ctx's daemon, by contrast, runs inside the user's live session, so
//! it is well placed both to notice the freeze and to spawn the sync itself.
//!
//! Staleness is read from the app's own `last_synced_at` heartbeat, not from message timestamps:
//! a healthy scraper on a quiet channel still advances the heartbeat, so this tells a stopped
//! scraper apart from a quiet one.
//!
//! Opt-in and self-limiting: nothing runs unless `CTX_TEAMS_SYNC_CMD` names the command. A
//! cooldown stops a sync that takes minutes from being launched again, and a work-hours window
//! keeps it off at night. A launched sync that keeps failing (e.g. expired Teams auth that needs
//! an interactive `npm run login`) is surfaced in the log rather than looped on — ctx cannot fix
//! an auth failure, only report it.

use std::path::Path;
use std::time::Duration;

use ingest::State;
use jiff::Zoned;
use sources::teams::TeamsDb;

/// State-meta key holding the epoch-ms of the last sync we triggered (the cooldown anchor).
const TRIGGER_KEY: &str = "teams/last_sync_trigger_ms";

#[derive(Debug, Clone)]
pub struct WatchConfig {
    /// The shell command line that runs one sync. `None` disables the whole feature.
    pub cmd: Option<String>,
    /// `teams.db` counts as stale once its heartbeat is older than this.
    pub stale_after: Duration,
    /// Minimum gap between two triggers, so a slow sync is not launched on top of itself.
    pub cooldown: Duration,
    /// Local `[start, end)` hours, on weekdays only, within which a trigger may fire.
    pub hours: (u32, u32),
}

impl WatchConfig {
    pub fn enabled(&self) -> bool {
        self.cmd.is_some()
    }
}

/// Why a check did or did not fire, for logging and tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// No command configured.
    Disabled,
    /// The heartbeat is recent enough.
    Fresh,
    /// Weekend, or outside the configured hours.
    OutsideHours,
    /// Stale, but a trigger fired too recently.
    CoolingDown,
    /// Stale and eligible: the scraper should be run.
    Trigger,
    /// A sync this process launched is still running.
    Running,
}

/// "8-18" → `(8, 18)`. Malformed, reversed or out-of-range input falls back to `(8, 18)`. The
/// window is half-open `[start, end)`.
pub fn parse_hours(s: &str) -> (u32, u32) {
    let parsed = s.split_once('-').and_then(|(a, b)| Some((a.trim().parse::<u32>().ok()?, b.trim().parse::<u32>().ok()?)));
    match parsed {
        Some((a, b)) if a < b && b <= 24 => (a, b),
        _ => (8, 18),
    }
}

fn within_hours(now: &Zoned, (start, end): (u32, u32)) -> bool {
    use jiff::civil::Weekday::{Saturday, Sunday};
    if matches!(now.weekday(), Saturday | Sunday) {
        return false;
    }
    let hour = now.hour() as u32;
    hour >= start && hour < end
}

/// Time inside the weekday `[start, end)` window between `since_ms` and `now`. The scraper only
/// syncs in working hours, so a night or a weekend without a sync is not a missed sync: counting
/// wall-clock time made "Teams stale" show every evening and all weekend while nothing was wrong.
pub fn working_ms_since(since_ms: i64, now: &Zoned, (start, end): (u32, u32)) -> i64 {
    use jiff::civil::Weekday::{Saturday, Sunday};
    let now_ms = now.timestamp().as_millisecond();
    if since_ms >= now_ms {
        return 0;
    }
    // Weeks of silence are stale whatever the exact count; this also bounds the day loop.
    const HORIZON_MS: i64 = 30 * 24 * 3_600_000;
    let since_ms = since_ms.max(now_ms - HORIZON_MS);
    let tz = now.time_zone().clone();
    let Ok(since) = jiff::Timestamp::from_millisecond(since_ms) else { return now_ms - since_ms };
    let mut day = since.to_zoned(tz.clone()).date();
    let mut total = 0;
    while day <= now.date() {
        if !matches!(day.weekday(), Saturday | Sunday) {
            let at = |h: u32| -> Option<i64> {
                let civil = if h >= 24 { day.tomorrow().ok()?.at(0, 0, 0, 0) } else { day.at(h as i8, 0, 0, 0) };
                Some(civil.to_zoned(tz.clone()).ok()?.timestamp().as_millisecond())
            };
            if let (Some(open), Some(close)) = (at(start), at(end)) {
                total += (close.min(now_ms) - open.max(since_ms)).max(0);
            }
        }
        let Ok(next) = day.tomorrow() else { break };
        day = next;
    }
    total
}

/// Whether the scraper has missed its syncs: more than `stale_after` of working time since the
/// last heartbeat. A db that never synced is stale.
pub fn is_stale(cfg: &WatchConfig, last_synced_ms: Option<i64>, now: &Zoned) -> bool {
    last_synced_ms.is_none_or(|synced| working_ms_since(synced, now, cfg.hours) >= cfg.stale_after.as_millis() as i64)
}

/// Decide what to do from the freshness inputs alone — pure, so the policy is testable without a
/// database, a clock or a process.
pub fn decide(cfg: &WatchConfig, last_synced_ms: Option<i64>, last_trigger_ms: Option<i64>, now: &Zoned) -> Decision {
    if cfg.cmd.is_none() {
        return Decision::Disabled;
    }
    if !within_hours(now, cfg.hours) {
        return Decision::OutsideHours;
    }
    let now_ms = now.timestamp().as_millisecond();
    if !is_stale(cfg, last_synced_ms, now) {
        return Decision::Fresh;
    }
    if let Some(triggered) = last_trigger_ms
        && now_ms.saturating_sub(triggered) < cfg.cooldown.as_millis() as i64
    {
        return Decision::CoolingDown;
    }
    Decision::Trigger
}

/// Read freshness and the cooldown marker, decide, and on a trigger record the attempt and hand
/// back the command to run. Spawning is left to [`check_and_trigger`], so this whole path is
/// testable without launching a process.
pub fn check(cfg: &WatchConfig, teams_db: &Path, state: &State, now: &Zoned) -> anyhow::Result<(Decision, Option<String>)> {
    // A missing or unreadable db reads as "never synced": stale, so a stopped scraper that never
    // wrote the file still triggers.
    let last_synced = TeamsDb::open(teams_db).ok().and_then(|db| db.last_synced_at().ok().flatten());
    let last_trigger = state.get_meta(TRIGGER_KEY)?.and_then(|s| s.trim().parse::<i64>().ok());
    let decision = decide(cfg, last_synced, last_trigger, now);
    if decision == Decision::Trigger {
        state.set_meta(TRIGGER_KEY, &now.timestamp().as_millisecond().to_string())?;
        return Ok((decision, cfg.cmd.clone()));
    }
    Ok((decision, None))
}

/// The command a user's `CTX_TEAMS_SYNC_CMD` names, run through the platform shell so it may be a
/// full command line (arguments, redirects, `&&`).
pub fn build_command(cmd: &str) -> std::process::Command {
    if cfg!(windows) {
        let mut c = std::process::Command::new("cmd");
        c.args(["/C", cmd]);
        c
    } else {
        let mut c = std::process::Command::new("sh");
        c.args(["-c", cmd]);
        c
    }
}

/// Daemon entry point: [`check`], then run the sync in a detached thread. The sync can take
/// minutes (it drives a headless browser), so its outcome is logged, never awaited — indexing
/// must not block on it, and the cooldown already prevents a second launch.
pub fn check_and_trigger(cfg: &WatchConfig, teams_db: &Path, state: &State, now: &Zoned) -> anyhow::Result<Decision> {
    // The cooldown alone does not stop a sync that outlives it from being launched again.
    let Some(guard) = SyncGuard::acquire() else {
        return Ok(Decision::Running);
    };
    let (decision, cmd) = check(cfg, teams_db, state, now)?;
    if let Some(cmd) = cmd {
        std::thread::spawn(move || {
            let _guard = guard;
            let started = std::time::Instant::now();
            eprintln!("[teams-sync] scraper looks stopped; running: {cmd}");
            match run_sync(&cmd, SYNC_TIMEOUT) {
                Ok(()) => eprintln!("[teams-sync] sync finished in {:.0?}", started.elapsed()),
                Err(e) => eprintln!("[teams-sync] sync FAILED ({e}); if this persists the Teams login has likely expired — run `npm run login`"),
            }
        });
    }
    Ok(decision)
}

/// Longest a triggered sync may run before it is killed; a healthy one takes minutes.
pub const SYNC_TIMEOUT: Duration = Duration::from_secs(30 * 60);

static SYNC_RUNNING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Held while a triggered sync runs; at most one exists per process.
pub struct SyncGuard(());

impl SyncGuard {
    pub fn acquire() -> Option<Self> {
        use std::sync::atomic::Ordering;
        SYNC_RUNNING.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_ok().then(|| Self(()))
    }
}

impl Drop for SyncGuard {
    fn drop(&mut self) {
        SYNC_RUNNING.store(false, std::sync::atomic::Ordering::Release);
    }
}

/// Runs `cmd` through the shell and waits for it, killing its whole process tree after
/// `timeout`: the scraper drives a headless browser, which a hung run would otherwise keep alive.
pub fn run_sync(cmd: &str, timeout: Duration) -> Result<(), String> {
    use std::io::Read;
    let mut child = build_command(cmd)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not launch the scraper ({e}); check CTX_TEAMS_SYNC_CMD"))?;
    let mut err = child.stderr.take().expect("piped");
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut s = String::new();
        let _ = err.read_to_string(&mut s);
        let _ = tx.send(s);
    });
    let deadline = std::time::Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if std::time::Instant::now() >= deadline => {
                crate::summarizer::kill_tree(&mut child);
                return Err(format!("killed after {timeout:?}"));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(200)),
            Err(e) => return Err(format!("waiting for the scraper: {e}")),
        }
    };
    if status.success() {
        return Ok(());
    }
    // Bounded: a grandchild that inherited the pipe could keep it open after exit.
    let stderr = rx.recv_timeout(Duration::from_secs(5)).unwrap_or_default();
    let tail: String = stderr.trim().chars().rev().take(400).collect::<String>().chars().rev().collect();
    Err(format!("{status}; stderr: {tail}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(s: &str) -> Zoned {
        jiff::Zoned::strptime("%Y-%m-%dT%H:%M:%S[%Q]", s).unwrap()
    }

    fn cfg(cmd: Option<&str>) -> WatchConfig {
        WatchConfig {
            cmd: cmd.map(str::to_owned),
            stale_after: Duration::from_secs(6 * 3600),
            cooldown: Duration::from_secs(90 * 60),
            hours: (8, 18),
        }
    }

    fn ms(z: &Zoned) -> i64 {
        z.timestamp().as_millisecond()
    }

    #[test]
    fn parse_hours_reads_window_and_falls_back() {
        assert_eq!(parse_hours("8-18"), (8, 18));
        assert_eq!(parse_hours(" 9 - 17 "), (9, 17));
        assert_eq!(parse_hours("18-8"), (8, 18), "reversed → default");
        assert_eq!(parse_hours("garbage"), (8, 18));
        assert_eq!(parse_hours("0-24"), (0, 24));
        assert_eq!(parse_hours("8-25"), (8, 18), "end out of range → default");
    }

    #[test]
    fn disabled_without_a_command() {
        let now = at("2026-09-28T10:00:00[Europe/Warsaw]"); // a Monday, work hours
        assert_eq!(decide(&cfg(None), None, None, &now), Decision::Disabled);
    }

    #[test]
    fn fresh_heartbeat_does_not_trigger() {
        let now = at("2026-09-28T10:00:00[Europe/Warsaw]");
        let synced = ms(&at("2026-09-28T09:00:00[Europe/Warsaw]")); // 1h ago < 6h
        assert_eq!(decide(&cfg(Some("sync")), Some(synced), None, &now), Decision::Fresh);
    }

    #[test]
    fn stale_heartbeat_triggers_and_never_synced_counts_as_stale() {
        let now = at("2026-09-28T10:00:00[Europe/Warsaw]");
        let synced = ms(&at("2026-09-25T11:00:00[Europe/Warsaw]")); // Fri 11:00: 7h Fri + 2h Mon > 6h
        assert_eq!(decide(&cfg(Some("sync")), Some(synced), None, &now), Decision::Trigger);
        assert_eq!(decide(&cfg(Some("sync")), None, None, &now), Decision::Trigger, "never synced");
    }

    #[test]
    fn only_working_hours_count_towards_staleness() {
        let h = (8, 18);
        let friday_15 = ms(&at("2026-09-25T15:09:00[Europe/Warsaw]"));
        // Friday evening and the whole weekend add nothing: 2h51m on Friday, then Monday from 08.
        assert_eq!(working_ms_since(friday_15, &at("2026-09-26T23:00:00[Europe/Warsaw]"), h), (2 * 60 + 51) * 60_000);
        assert_eq!(working_ms_since(friday_15, &at("2026-09-28T09:00:00[Europe/Warsaw]"), h), (3 * 60 + 51) * 60_000);
        // Inside one working day it is plain elapsed time; a heartbeat in the future is 0.
        let monday_9 = ms(&at("2026-09-28T09:00:00[Europe/Warsaw]"));
        assert_eq!(working_ms_since(monday_9, &at("2026-09-28T12:30:00[Europe/Warsaw]"), h), 210 * 60_000);
        assert_eq!(working_ms_since(monday_9 + 1, &at("2026-09-28T09:00:00[Europe/Warsaw]"), h), 0);
        // Before the window opens and across a DST change (2026-10-25, Sunday).
        assert_eq!(working_ms_since(monday_9, &at("2026-09-29T07:00:00[Europe/Warsaw]"), h), 9 * 3_600_000);
        let fri_17 = ms(&at("2026-10-23T17:00:00[Europe/Warsaw]"));
        assert_eq!(working_ms_since(fri_17, &at("2026-10-26T09:00:00[Europe/Warsaw]"), h), 2 * 3_600_000);
        // A 0-24 window counts whole weekdays.
        assert_eq!(working_ms_since(monday_9, &at("2026-09-29T09:00:00[Europe/Warsaw]"), (0, 24)), 24 * 3_600_000);
    }

    #[test]
    fn evenings_and_weekends_are_not_stale_but_a_missed_working_day_is() {
        let c = WatchConfig { stale_after: Duration::from_secs(3 * 3600), ..cfg(Some("sync")) };
        let friday_15 = ms(&at("2026-09-25T15:09:00[Europe/Warsaw]"));
        assert!(!is_stale(&c, Some(friday_15), &at("2026-09-26T23:00:00[Europe/Warsaw]")), "Saturday night");
        assert!(!is_stale(&c, Some(friday_15), &at("2026-09-28T08:05:00[Europe/Warsaw]")), "Monday before the first sync");
        assert!(is_stale(&c, Some(friday_15), &at("2026-09-28T09:00:00[Europe/Warsaw]")), "Monday 09:00: 3h51m of working time");
        assert!(is_stale(&c, None, &at("2026-09-26T23:00:00[Europe/Warsaw]")), "never synced");
        assert!(is_stale(&c, Some(0), &at("2026-09-28T09:00:00[Europe/Warsaw]")), "decades ago, bounded loop");
    }

    #[test]
    fn cooldown_holds_back_a_repeat_trigger() {
        let now = at("2026-09-28T10:00:00[Europe/Warsaw]");
        let stale = ms(&at("2026-09-24T10:00:00[Europe/Warsaw]")); // Thursday
        let recent = ms(&at("2026-09-28T09:30:00[Europe/Warsaw]")); // 30m ago < 90m
        assert_eq!(decide(&cfg(Some("sync")), Some(stale), Some(recent), &now), Decision::CoolingDown);
        let old = ms(&at("2026-09-28T08:00:00[Europe/Warsaw]")); // 2h ago > 90m
        assert_eq!(decide(&cfg(Some("sync")), Some(stale), Some(old), &now), Decision::Trigger);
    }

    #[test]
    fn outside_hours_and_weekends_never_fire_even_when_stale() {
        let stale = ms(&at("2026-09-20T10:00:00[Europe/Warsaw]"));
        let night = at("2026-09-28T03:00:00[Europe/Warsaw]"); // Monday, before 08
        assert_eq!(decide(&cfg(Some("sync")), Some(stale), None, &night), Decision::OutsideHours);
        let saturday = at("2026-09-26T10:00:00[Europe/Warsaw]"); // weekend, work hours
        assert_eq!(decide(&cfg(Some("sync")), Some(stale), None, &saturday), Decision::OutsideHours);
        let after = at("2026-09-28T18:00:00[Europe/Warsaw]"); // end is exclusive
        assert_eq!(decide(&cfg(Some("sync")), Some(stale), None, &after), Decision::OutsideHours);
    }

    #[test]
    fn check_records_the_trigger_and_then_cools_down() {
        let dir = tempfile::tempdir().unwrap();
        let state = State::open(&dir.path().join("state.db")).unwrap();
        let missing_db = dir.path().join("nope.db"); // never synced → stale
        let now = at("2026-09-28T10:00:00[Europe/Warsaw]");
        let c = cfg(Some("cd .")); // harmless, but never spawned on this path

        let (decision, cmd) = check(&c, &missing_db, &state, &now).unwrap();
        assert_eq!((decision, cmd.as_deref()), (Decision::Trigger, Some("cd .")));
        assert_eq!(state.get_meta(TRIGGER_KEY).unwrap().as_deref(), Some(ms(&now).to_string().as_str()), "marker written");

        // A second check a few minutes later is inside the cooldown: no re-trigger, marker kept.
        let soon = at("2026-09-28T10:05:00[Europe/Warsaw]");
        let (decision, cmd) = check(&c, &missing_db, &state, &soon).unwrap();
        assert_eq!((decision, cmd), (Decision::CoolingDown, None));
        assert_eq!(state.get_meta(TRIGGER_KEY).unwrap().as_deref(), Some(ms(&now).to_string().as_str()), "marker unchanged");
    }

    /// The guard is one process-wide flag: every test that takes it goes through this lock.
    static GUARD_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn wait_for_guard_release(within: Duration) -> SyncGuard {
        let deadline = std::time::Instant::now() + within;
        loop {
            if let Some(g) = SyncGuard::acquire() {
                return g;
            }
            assert!(std::time::Instant::now() < deadline, "guard still held after {within:?}");
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    #[test]
    fn sync_guard_is_exclusive_and_released_on_drop() {
        let _serial = GUARD_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        let first = SyncGuard::acquire().expect("nothing holds the guard at the start");
        assert!(SyncGuard::acquire().is_none(), "a second guard while the first is held");
        assert!(SyncGuard::acquire().is_none(), "still held on a repeated attempt");
        drop(first);
        let again = SyncGuard::acquire().expect("released on drop");
        drop(again);
    }

    #[test]
    fn check_and_trigger_reports_running_without_touching_the_marker_while_a_sync_runs() {
        let _serial = GUARD_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let state = State::open(&dir.path().join("state.db")).unwrap();
        let missing_db = dir.path().join("nope.db"); // never synced: stale, so `check` would trigger
        let monday_10 = at("2026-09-28T10:00:00[Europe/Warsaw]");
        // Runs ~2 s, long enough to observe the guard held by the launched sync.
        let c = cfg(Some("ping -n 3 127.0.0.1 >nul"));

        let held = SyncGuard::acquire().expect("guard free at start");
        assert_eq!(check_and_trigger(&c, &missing_db, &state, &monday_10).unwrap(), Decision::Running);
        assert_eq!(state.get_meta(TRIGGER_KEY).unwrap(), None, "no cooldown marker while a sync runs");
        assert!(SyncGuard::acquire().is_none(), "a `Running` answer leaves the guard with its holder");
        drop(held);

        assert_eq!(check_and_trigger(&c, &missing_db, &state, &monday_10).unwrap(), Decision::Trigger);
        assert_eq!(state.get_meta(TRIGGER_KEY).unwrap(), Some(ms(&monday_10).to_string()));

        // The launched sync holds the guard: every check past the cooldown is still `Running`,
        // and the marker keeps the first trigger's time.
        for minute in ["12:00", "12:01"] {
            let past_cooldown = at(&format!("2026-09-28T{minute}:00[Europe/Warsaw]"));
            assert_eq!(check_and_trigger(&c, &missing_db, &state, &past_cooldown).unwrap(), Decision::Running, "{minute}");
            assert_eq!(state.get_meta(TRIGGER_KEY).unwrap(), Some(ms(&monday_10).to_string()), "{minute}");
        }

        // ... and gives it back when the sync ends.
        drop(wait_for_guard_release(Duration::from_secs(20)));
    }

    #[test]
    fn run_sync_succeeds_on_exit_zero() {
        assert_eq!(run_sync("exit 0", Duration::from_secs(20)), Ok(()));
    }

    #[test]
    fn run_sync_reports_the_exit_status_and_stderr_on_failure() {
        assert_eq!(run_sync(">&2 echo login expired& exit 3", Duration::from_secs(20)), Err("exit code: 3; stderr: login expired".to_owned()));
    }

    #[test]
    fn run_sync_keeps_only_the_last_400_characters_of_stderr() {
        // 120 lines "line1".."line120" on stderr, CRLF-separated: well over 400 characters.
        let r = run_sync("(for /L %i in (1,1,120) do @(>&2 echo line%i)) & exit 5", Duration::from_secs(20));
        let all = (1..=120).map(|i| format!("line{i}")).collect::<Vec<_>>().join("\r\n");
        let tail: String = all.chars().skip(all.chars().count() - 400).collect();
        assert_eq!(r, Err(format!("exit code: 5; stderr: {tail}")));
    }

    #[test]
    fn run_sync_kills_the_whole_tree_on_timeout_and_returns_promptly() {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("marker.txt");
        let script = dir.path().join("grandchild.bat");
        // A grandchild that outlives a killed parent would write the marker after ~3 s.
        std::fs::write(&script, format!("@ping -n 4 127.0.0.1 >nul\r\n@echo done> \"{}\"\r\n", marker.display())).unwrap();
        assert!(!script.to_string_lossy().contains(' '), "the command line below assumes no spaces: {}", script.display());

        let started = std::time::Instant::now();
        let r = run_sync(&format!("cmd /C {}", script.display()), Duration::from_millis(1500));
        let took = started.elapsed();

        assert_eq!(r, Err("killed after 1.5s".to_owned()));
        assert!(took < Duration::from_secs(6), "returned after {took:?}");
        std::thread::sleep(Duration::from_secs(5));
        assert!(!marker.exists(), "the grandchild survived the kill and finished its work");
    }
}
