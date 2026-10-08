//! `ctx statusline` (DEV-59): Claude Code's status line, showing how full the context is and
//! nudging toward `/clear` before the session gets expensive and long-winded.
//!
//! Claude Code cannot be cleared from outside, so a timely nudge is the closest thing to
//! automatic context management; with the `/clear` handoff (DEV-58) acting on it loses nothing.
//! Claude Code runs this on every status update, so it must be fast and must never fail: missing
//! fields just drop their part of the line.

use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::time::Duration;

use serde_json::Value;

/// What the status line shows; gathered by the caller, rendered by [`render`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Status {
    pub model: Option<String>,
    /// `context_window.used_percentage`; absent before the first API call.
    pub used_pct: Option<f64>,
    /// `None` when not checked.
    pub daemon_up: Option<bool>,
    pub teams_stale: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct Thresholds {
    pub warn: f64,
    pub crit: f64,
    /// Whether `/clear` hands the session over (CTX_HANDOFF); changes the hint.
    pub handoff: bool,
}

const BAR: usize = 10;
const YELLOW: &str = "\x1b[33m";
const RED: &str = "\x1b[31m";
const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

/// The parts of Claude Code's status-line JSON this uses. Anything malformed is just absent.
pub fn parse(input: &str) -> Status {
    let v: Value = serde_json::from_str(input).unwrap_or(Value::Null);
    Status {
        model: v.pointer("/model/display_name").and_then(Value::as_str).map(str::to_owned),
        used_pct: v.pointer("/context_window/used_percentage").and_then(Value::as_f64).filter(|p| p.is_finite()),
        ..Default::default()
    }
}

pub fn render(s: &Status, t: Thresholds) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(m) = &s.model {
        parts.push(m.clone());
    }
    if let Some(pct) = s.used_pct {
        let pct = pct.clamp(0.0, 100.0);
        let filled = ((pct / 100.0) * BAR as f64).round() as usize;
        let bar = format!("{}{}", "▰".repeat(filled), "▱".repeat(BAR - filled));
        let text = format!("ctx {bar} {pct:.0}%");
        parts.push(if pct >= t.crit {
            let hint = if t.handoff { "/clear → ctx hands over" } else { "/clear or /compact" };
            format!("{RED}{text}{RESET} · {RED}{hint}{RESET}")
        } else if pct >= t.warn {
            format!("{YELLOW}{text}{RESET}")
        } else {
            text
        });
    }
    if s.daemon_up == Some(false) {
        parts.push(format!("{RED}ctx daemon down{RESET}"));
    }
    if s.teams_stale {
        parts.push(format!("{DIM}Teams stale{RESET}"));
    }
    parts.join(" · ")
}

/// Whether something accepts connections at the daemon URL's host:port, within `timeout`.
/// A plain TCP connect: an HTTP round-trip would cost more than the status line can afford.
pub fn daemon_up(embed_url: &str, timeout: Duration) -> bool {
    let host = embed_url.split("://").nth(1).unwrap_or(embed_url).split('/').next().unwrap_or_default();
    let addrs: Vec<SocketAddr> = host.to_socket_addrs().map(|a| a.collect()).unwrap_or_default();
    addrs.iter().any(|a| TcpStream::connect_timeout(a, timeout).is_ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    const T: Thresholds = Thresholds { warn: 50.0, crit: 75.0, handoff: true };

    #[test]
    fn parses_the_documented_fields_and_tolerates_nulls_and_garbage() {
        let s = parse(r#"{"model":{"display_name":"Opus 5.5"},"context_window":{"used_percentage":42.4}}"#);
        assert_eq!((s.model.as_deref(), s.used_pct), (Some("Opus 5.5"), Some(42.4)));
        let before_first_call = parse(r#"{"model":{"display_name":"Opus 5.5"},"context_window":{"used_percentage":null}}"#);
        assert_eq!(before_first_call.used_pct, None);
        assert_eq!(parse("not json"), Status::default());
    }

    #[test]
    fn plain_below_warn_yellow_at_warn_red_with_hint_at_crit() {
        let at = |pct| render(&Status { model: Some("M".into()), used_pct: Some(pct), ..Default::default() }, T);
        assert_eq!(at(34.0), "M · ctx ▰▰▰▱▱▱▱▱▱▱ 34%");
        assert_eq!(at(50.0), format!("M · {YELLOW}ctx ▰▰▰▰▰▱▱▱▱▱ 50%{RESET}"));
        let crit = at(80.0);
        assert!(crit.starts_with(&format!("M · {RED}ctx ▰▰▰▰▰▰▰▰▱▱ 80%{RESET}")), "{crit}");
        assert!(crit.ends_with(&format!("{RED}/clear → ctx hands over{RESET}")), "{crit}");
        let no_handoff = render(&Status { used_pct: Some(90.0), ..Default::default() }, Thresholds { handoff: false, ..T });
        assert!(no_handoff.contains("/clear or /compact"), "{no_handoff}");
        assert_eq!(at(140.0), at(100.0), "clamped");
    }

    #[test]
    fn health_markers_and_empty_input() {
        let s = Status { daemon_up: Some(false), teams_stale: true, ..Default::default() };
        assert_eq!(render(&s, T), format!("{RED}ctx daemon down{RESET} · {DIM}Teams stale{RESET}"));
        assert_eq!(render(&Status { daemon_up: Some(true), ..Default::default() }, T), "");
    }

    #[test]
    fn daemon_probe_sees_a_listener_and_not_a_closed_port() {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", l.local_addr().unwrap());
        assert!(daemon_up(&url, Duration::from_millis(500)));
        drop(l);
        assert!(!daemon_up(&url, Duration::from_millis(200)));
        assert!(!daemon_up("garbage", Duration::from_millis(50)));
    }
}
