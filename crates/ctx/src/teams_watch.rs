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
    let stale_ms = cfg.stale_after.as_millis() as i64;
    // A never-synced db (None) is treated as stale.
    if let Some(synced) = last_synced_ms
        && now_ms.saturating_sub(synced) < stale_ms
    {
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
    let (decision, cmd) = check(cfg, teams_db, state, now)?;
    if let Some(cmd) = cmd {
        std::thread::spawn(move || {
            let started = std::time::Instant::now();
            eprintln!("[teams-sync] scraper looks stopped; running: {cmd}");
            match build_command(&cmd).output() {
                Ok(out) if out.status.success() => eprintln!("[teams-sync] sync finished in {:.0?}", started.elapsed()),
                Ok(out) => {
                    let err = String::from_utf8_lossy(&out.stderr);
                    let tail: String = err.trim().chars().rev().take(400).collect::<String>().chars().rev().collect();
                    eprintln!("[teams-sync] sync FAILED ({}); if this persists the Teams login has likely expired — run `npm run login`. stderr: {tail}", out.status);
                }
                Err(e) => eprintln!("[teams-sync] could not launch the scraper ({e}); check CTX_TEAMS_SYNC_CMD"),
            }
        });
    }
    Ok(decision)
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
        let synced = ms(&at("2026-09-28T02:00:00[Europe/Warsaw]")); // 8h ago > 6h
        assert_eq!(decide(&cfg(Some("sync")), Some(synced), None, &now), Decision::Trigger);
        assert_eq!(decide(&cfg(Some("sync")), None, None, &now), Decision::Trigger, "never synced");
    }

    #[test]
    fn cooldown_holds_back_a_repeat_trigger() {
        let now = at("2026-09-28T10:00:00[Europe/Warsaw]");
        let stale = ms(&at("2026-09-27T10:00:00[Europe/Warsaw]"));
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
}
