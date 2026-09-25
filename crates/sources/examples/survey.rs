//! Loads every real session and prints per-project totals: `cargo run -p sources --example survey`.

use std::collections::BTreeMap;

use sources::{SessionRef, claude, opencode};

fn main() {
    let home = std::env::var("USERPROFILE").expect("USERPROFILE");
    let mut refs: Vec<SessionRef> = claude::discover(format!("{home}/.claude/projects").as_ref()).unwrap();
    refs.extend(opencode::discover(format!("{home}/.local/share/opencode/opencode.db").as_ref()).unwrap());

    let mut per_project: BTreeMap<String, (usize, usize, usize)> = BTreeMap::new();
    let (mut ok, mut empty, mut failed) = (0, 0, 0);
    for r in &refs {
        let loaded = match r.source {
            sources::Source::Claude => claude::load(r),
            sources::Source::OpenCode => opencode::load(r),
        };
        match loaded {
            Ok(s) => {
                ok += 1;
                let chars: usize = s.turns.iter().map(|t| t.user.len() + t.assistant.len() + t.tools.join("\n").len()).sum();
                let e = per_project.entry(format!("{:<8} {}", s.source.as_str(), s.project)).or_default();
                e.0 += 1;
                e.1 += s.turns.len();
                e.2 += chars;
            }
            Err(sources::SourceError::Empty(_)) => empty += 1,
            Err(e) => {
                failed += 1;
                eprintln!("{}: {e}", r.id);
            }
        }
    }
    println!("{:<50} {:>8} {:>6} {:>9}", "source / project", "sessions", "turns", "KB text");
    let mut total = (0, 0, 0);
    for (p, (s, t, c)) in &per_project {
        println!("{p:<50} {s:>8} {t:>6} {:>9}", c / 1024);
        total = (total.0 + s, total.1 + t, total.2 + c);
    }
    println!("{:<50} {:>8} {:>6} {:>9}", "TOTAL", total.0, total.1, total.2 / 1024);
    println!("refs={} loaded={ok} empty={empty} failed={failed}", refs.len());
}
