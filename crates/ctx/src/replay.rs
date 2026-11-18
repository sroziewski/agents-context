//! Replays past Claude Code prompts to OpenCode in the background, so projects routed to
//! OpenCode get used on real questions.
//!
//! The daemon's periodic run picks one prompt the user once sent Claude in an allowed project
//! (`CTX_REPLAY_PROJECTS`), sends it with `opencode run` in that project's directory, then
//! follows up 2-5 times with the answer's own `Next:` suggestion, as the `/next` command of the
//! user's OpenCode setup does. Every exchange is appended to `logs/opencode.log`.
//!
//! Prompts come from the index, not the transcripts: they are already redacted, and the index
//! keeps them after Claude Code has deleted the transcript. Runs go through a dedicated
//! read-only agent, because an old prompt ("commit and push", "delete the branch") replayed by
//! OpenCode's default agent would act on the real repository.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use ingest::State;
use store::{SearchFilter, Store};

/// Titles of replay sessions start with this; the OpenCode source skips them when indexing, so
/// replays never come back as "past sessions".
pub const TITLE_PREFIX: &str = sources::opencode::REPLAY_TITLE_PREFIX;

const LAST_RUN: &str = "replay/last_ms";
const DAY_COUNT: &str = "replay/day";
const MAX_SUGGESTION: usize = 400;
/// The prompt travels as a command-line argument; Windows caps a whole command line at 32767.
const MAX_PROMPT_CHARS: usize = 12_000;
const LOG_MAX_BYTES: u64 = 10 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct ReplayConfig {
    pub enabled: bool,
    /// Project names; a project matches when any segment of its path equals one of them.
    pub projects: Vec<String>,
    pub every: Duration,
    pub daily_max: u32,
    /// Inclusive range of `/next` follow-ups after the replayed prompt.
    pub next: (u32, u32),
    pub agent: String,
    /// Where OpenCode reads agent files from; the run is refused when the agent is not there,
    /// since OpenCode would otherwise fall back to its default agent, which may edit and run.
    pub agents_dir: PathBuf,
    pub model: Option<String>,
    pub exe: PathBuf,
    /// Per `opencode run` call.
    pub timeout: Duration,
    /// Shorter prompts ("ok", "commit") mean nothing without the conversation around them.
    pub min_chars: usize,
    /// Lowercase words that keep a prompt or follow-up out of replays; see [`excluded`].
    pub skip_words: Vec<String>,
    pub log: PathBuf,
}

/// Prompts about databases ("check on prod db …") ask for access the replay agent does not
/// have, so they are not replayed. Polish forms of "baza" are listed because prompts mix both.
pub const DEFAULT_SKIP_WORDS: &str = "db,dbs,database,databases,sql,psql,sqlplus,baza,bazy,bazie,bazę,bazą,bazach,bazami,bazo,baz";

/// Comma-separated words, lowercased; unset gives [`DEFAULT_SKIP_WORDS`], an empty value none.
pub fn parse_skip_words(v: Option<&str>) -> Vec<String> {
    parse_projects(Some(v.unwrap_or(DEFAULT_SKIP_WORDS))).into_iter().map(|w| w.to_lowercase()).collect()
}

/// Why `text` must not be sent to OpenCode, if it must not: nothing mentioning Claude is ever
/// sent (anywhere in the text, so `CLAUDE.md` and `.claude/` count too), and nothing with one of
/// `skip_words` as a whole word, words being split at anything but letters and digits.
pub fn excluded(text: &str, skip_words: &[String]) -> Option<String> {
    let lower = text.to_lowercase();
    if lower.contains("claude") {
        return Some("mentions Claude".to_owned());
    }
    lower.split(|c: char| !c.is_alphanumeric()).find(|w| !w.is_empty() && skip_words.iter().any(|s| s == w)).map(|w| format!("has the skip word `{w}`"))
}

/// `"2-5"` → (2, 5), `"3"` → (3, 3). Unparseable values give `default`; a reversed range is
/// put in order.
pub fn parse_range(v: Option<&str>, default: (u32, u32)) -> (u32, u32) {
    let Some(v) = v.map(str::trim).filter(|v| !v.is_empty()) else { return default };
    let parsed: Option<(u32, u32)> = match v.split_once('-') {
        Some((a, b)) => a.trim().parse().ok().zip(b.trim().parse().ok()),
        None => v.parse().ok().map(|n| (n, n)),
    };
    match parsed {
        Some((a, b)) => (a.min(b), a.max(b)),
        None => default,
    }
}

/// Comma-separated names, trimmed, blanks dropped.
pub fn parse_projects(v: Option<&str>) -> Vec<String> {
    v.unwrap_or_default().split(',').map(str::trim).filter(|s| !s.is_empty()).map(str::to_owned).collect()
}

/// Whether the project key (a normalised path such as `d:/dev/sedn/sedn26`) belongs to one of
/// the named projects: some path segment equals a name, ignoring case. A name therefore also
/// covers the project's sub-directories, which are indexed as projects of their own.
pub fn project_matches(project: &str, names: &[String]) -> bool {
    project.split(['/', '\\']).any(|seg| !seg.is_empty() && names.iter().any(|n| seg.eq_ignore_ascii_case(n.trim())))
}

/// The user's prompt inside a stored turn chunk: `[title · date]\nUser: <prompt>\n\nAssistant: …`.
/// A long prompt split over several chunks yields its first part.
pub fn prompt_of(text: &str) -> Option<String> {
    let body = match text.strip_prefix('[') {
        Some(_) => text.split_once('\n')?.1,
        None => text,
    };
    let rest = body.strip_prefix("User: ")?;
    let end = ["\n\nAssistant: ", "\n\nTools: ", "\n\nFiles: "].iter().filter_map(|m| rest.find(m)).min().unwrap_or(rest.len());
    let prompt = rest[..end].trim();
    (!prompt.is_empty()).then(|| prompt.to_owned())
}

/// Worth replaying on its own: long enough to carry a question, and not a slash command, shell
/// escape or harness message.
pub fn usable(prompt: &str, min_chars: usize) -> bool {
    let p = prompt.trim_start();
    p.chars().count() >= min_chars && !p.starts_with('/') && !p.starts_with('!') && !p.contains("<command-") && !p.starts_with("[Request interrupted")
}

/// The answer's trailing `Next: <prompt>` suggestion, as the user's OpenCode `next-step` plugin
/// reads it: the last `Next:` / `Next action:` line outside code fences, markdown bold allowed,
/// with a closing "want me to?" or "on your go" dropped.
pub fn extract_next(text: &str) -> Option<String> {
    let mut fence: Option<&str> = None;
    let mut found = None;
    for line in text.lines() {
        let t = line.trim();
        if let Some(open) = fence {
            if t.starts_with(open) {
                fence = None;
            }
            continue;
        }
        if let Some(marker) = ["```", "~~~"].into_iter().find(|m| t.starts_with(m)) {
            fence = Some(marker);
            continue;
        }
        if let Some(s) = next_line(t) {
            found = Some(s);
        }
    }
    let s = found?;
    let s = clean_suggestion(&s);
    (!s.is_empty() && s.chars().count() <= MAX_SUGGESTION).then_some(s)
}

fn next_line(t: &str) -> Option<String> {
    let t = t.strip_prefix("**").unwrap_or(t);
    let head = t.get(..4)?;
    if !head.eq_ignore_ascii_case("next") {
        return None;
    }
    let mut rest = &t[4..];
    if rest.get(..7).is_some_and(|h| h.eq_ignore_ascii_case(" action")) {
        rest = &rest[7..];
    }
    let rest = rest.strip_prefix("**").unwrap_or(rest);
    let rest = rest.strip_prefix(':')?;
    let rest = rest.strip_prefix("**").unwrap_or(rest);
    Some(rest.trim().to_owned())
}

fn clean_suggestion(s: &str) -> String {
    let mut s = s.trim().trim_start_matches("**").trim_end_matches("**").trim().to_owned();
    loop {
        s = s.trim_end_matches(['.', ',']).trim_end().to_owned();
        let lower = s.to_ascii_lowercase();
        let tail = ["want me to?", "want me to", "shall i?", "shall i", "should i?", "should i", "ok?", "on your go", "when you're ready", "when youre ready", "awaiting your approval", "if you approve"]
            .iter()
            // At a word boundary only: "update the webhook?" must keep its "ok?".
            .find(|t| lower.ends_with(*t) && s.len() > t.len() && !s[..s.len() - t.len()].chars().next_back().is_some_and(char::is_alphanumeric));
        let Some(tail) = tail else { break };
        s.truncate(s.len() - tail.len());
        s = s.trim_end().trim_end_matches(['—', '–', '-', ',']).trim_end().to_owned();
    }
    s.trim_end_matches(['.', ',']).trim().to_owned()
}

/// Whether a replay may start now: on, with projects named, under today's cap, and `every`
/// since the last start or the last scan that found nothing to replay.
pub fn due(cfg: &ReplayConfig, last_ms: Option<i64>, today_count: u32, now_ms: i64) -> bool {
    cfg.enabled && !cfg.projects.is_empty() && today_count < cfg.daily_max && last_ms.is_none_or(|l| now_ms - l >= cfg.every.as_millis() as i64)
}

/// splitmix64: enough to pick a prompt and a follow-up count, without a dependency.
pub struct Rng(u64);

impl Rng {
    pub fn seeded(seed: u64) -> Self {
        Self(seed)
    }

    pub fn from_time() -> Self {
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
        Self(nanos ^ (std::process::id() as u64).rotate_left(32))
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n`; `n` must be positive.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// Uniform in `lo..=hi`.
    pub fn between(&mut self, lo: u32, hi: u32) -> u32 {
        lo + self.below((hi - lo) as usize + 1) as u32
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub session_id: String,
    /// The turn's id, or its index when the source has no ids.
    pub turn: String,
    pub project: String,
    pub cwd: PathBuf,
    pub prompt: String,
}

impl Candidate {
    fn done_key(&self) -> String {
        done_key(&self.session_id, &self.turn)
    }
}

pub fn done_key(session_id: &str, turn: &str) -> String {
    format!("replay/done/{session_id}/{turn}")
}

/// A random prompt from the allowed projects that has not been replayed yet: the first chunk of
/// a main-session Claude turn whose directory still exists.
pub async fn pick(store: &Store, state: &State, cfg: &ReplayConfig, rng: &mut Rng) -> anyhow::Result<Option<Candidate>> {
    let candidates = candidates(store, state, cfg).await?;
    Ok((!candidates.is_empty()).then(|| candidates[rng.below(candidates.len())].clone()))
}

/// Every prompt [`pick`] could choose, in index order.
pub async fn candidates(store: &Store, state: &State, cfg: &ReplayConfig) -> anyhow::Result<Vec<Candidate>> {
    let projects: Vec<String> = state.per_project()?.into_iter().map(|(p, ..)| p).filter(|p| project_matches(p, &cfg.projects)).collect();
    if projects.is_empty() {
        return Ok(Vec::new());
    }
    let filter = SearchFilter::default()
        .eq("kind", "turn")
        .eq("source", "claude")
        .flag("subagent", false)
        .int("chunk_index", 0)
        .any_of("project", projects);
    let mut out = Vec::new();
    for hit in store.scroll(&filter, 1_000_000).await? {
        let p = &hit.payload;
        let str_of = |k: &str| p.get(k).and_then(|v| v.as_str()).unwrap_or_default().to_owned();
        let Some(prompt) = prompt_of(&str_of("text")).filter(|t| usable(t, cfg.min_chars) && excluded(t, &cfg.skip_words).is_none()) else { continue };
        let turn = match p.get("turn_id").and_then(|v| v.as_str()).filter(|t| !t.is_empty()) {
            Some(id) => id.to_owned(),
            None => p.get("turn_index").and_then(|v| v.as_i64()).unwrap_or_default().to_string(),
        };
        let (session_id, project) = (str_of("session_id"), str_of("project"));
        if state.get_meta(&done_key(&session_id, &turn))?.is_some() {
            continue;
        }
        // OpenCode puts its working directory into the system prompt, so a `.claude/…` one
        // would send Claude's name after all.
        if project.to_lowercase().contains("claude") {
            continue;
        }
        let Some(cwd) = [str_of("cwd"), project.clone()]
            .into_iter()
            .filter(|d| !d.to_lowercase().contains("claude"))
            .map(PathBuf::from)
            .find(|d| !d.as_os_str().is_empty() && d.is_dir())
        else {
            continue;
        };
        let prompt: String = prompt.chars().take(MAX_PROMPT_CHARS).collect();
        out.push(Candidate { session_id, turn, project, cwd, prompt });
    }
    Ok(out)
}

/// One answer from the agent.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Reply {
    pub session_id: String,
    pub text: String,
    pub tokens_in: u64,
    pub tokens_out: u64,
}

/// Something that answers a prompt in a session; [`OpencodeCli`] in production.
pub trait Agent {
    /// `session` continues an existing session; `None` starts one titled `title`.
    fn send(&self, cwd: &Path, session: Option<&str>, title: &str, prompt: &str) -> Result<Reply, String>;
}

pub struct OpencodeCli {
    pub exe: PathBuf,
    pub agent: String,
    pub model: Option<String>,
    pub timeout: Duration,
}

impl OpencodeCli {
    pub fn from_config(cfg: &ReplayConfig) -> Self {
        Self { exe: cfg.exe.clone(), agent: cfg.agent.clone(), model: cfg.model.clone(), timeout: cfg.timeout }
    }

    pub fn args(&self, session: Option<&str>, title: &str, prompt: &str) -> Vec<String> {
        let mut a: Vec<String> = ["run", "--format", "json", "--agent", &self.agent].map(str::to_owned).to_vec();
        if let Some(m) = &self.model {
            a.extend(["--model".to_owned(), m.clone()]);
        }
        match session {
            Some(s) => a.extend(["--session".to_owned(), s.to_owned()]),
            None => a.extend(["--archived".to_owned(), "--title".to_owned(), title.to_owned()]),
        }
        // A prompt starting with `-` would be read as a flag.
        a.push(if prompt.starts_with('-') { format!(" {prompt}") } else { prompt.to_owned() });
        a
    }
}

impl Agent for OpencodeCli {
    fn send(&self, cwd: &Path, session: Option<&str>, title: &str, prompt: &str) -> Result<Reply, String> {
        let mut child = Command::new(&self.exe)
            .args(self.args(session, title, prompt))
            .current_dir(cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("cannot start {} ({e}); check CTX_OPENCODE_EXE", self.exe.display()))?;
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
        let deadline = Instant::now() + self.timeout;
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) if Instant::now() >= deadline => {
                    crate::summarizer::kill_tree(&mut child);
                    return Err(format!("opencode killed after {:?}", self.timeout));
                }
                Ok(None) => std::thread::sleep(Duration::from_millis(200)),
                Err(e) => return Err(format!("waiting for opencode: {e}")),
            }
        };
        // Bounded: a grandchild that inherited the pipes could keep them open after exit.
        let stderr = err_rx.recv_timeout(Duration::from_secs(5)).unwrap_or_default();
        let stdout = out_rx.recv_timeout(Duration::from_secs(5)).ok();
        if !status.success() {
            // OpenCode prints a failed request ("Model not found") as an `error` event on stdout
            // and leaves stderr empty.
            let reason = stdout.as_deref().and_then(reported_error).unwrap_or_else(|| tail(&stderr, 400));
            return Err(format!("opencode exited with {status}: {reason}"));
        }
        let Some(stdout) = stdout else {
            return Err("opencode exited but a process it started kept its output open; the answer was lost".into());
        };
        parse_reply(&stdout).map_err(|e| if stderr.trim().is_empty() { e } else { format!("{e}; stderr: {}", tail(&stderr, 400)) })
    }
}

fn tail(s: &str, n: usize) -> String {
    let s = s.trim();
    let skip = s.chars().count().saturating_sub(n);
    s.chars().skip(skip).collect()
}

/// `opencode run --format json` prints one event per line: `text` parts carry the answer,
/// `step_finish` parts the token counts, every event the session id.
pub fn parse_reply(stdout: &str) -> Result<Reply, String> {
    let mut reply = Reply::default();
    let mut texts = Vec::new();
    let mut errors = Vec::new();
    for line in stdout.lines().map(str::trim).filter(|l| l.starts_with('{')) {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else { continue };
        if reply.session_id.is_empty()
            && let Some(id) = v.get("sessionID").and_then(|s| s.as_str())
        {
            reply.session_id = id.to_owned();
        }
        let part = v.get("part");
        match v.get("type").and_then(|t| t.as_str()) {
            Some("text") => {
                if let Some(t) = part.and_then(|p| p.get("text")).and_then(|t| t.as_str()).filter(|t| !t.trim().is_empty()) {
                    texts.push(t.trim().to_owned());
                }
            }
            Some("step_finish") => {
                let tokens = part.and_then(|p| p.get("tokens"));
                let n = |k: &str| tokens.and_then(|t| t.get(k)).and_then(|v| v.as_u64()).unwrap_or(0);
                reply.tokens_in += n("input");
                reply.tokens_out += n("output") + n("reasoning");
            }
            Some("error") => errors.push(v.get("error").map(error_text).unwrap_or_else(|| line.to_owned())),
            _ => {}
        }
    }
    reply.text = texts.join("\n\n");
    if !errors.is_empty() && reply.text.is_empty() {
        return Err(format!("opencode reported an error: {}", tail(&errors.join("; "), 400)));
    }
    if reply.session_id.is_empty() {
        return Err(format!("no session in opencode output: {}", tail(stdout, 300)));
    }
    Ok(reply)
}

/// The `message` of an `error` event's error, or the whole error when it has none.
fn error_text(e: &serde_json::Value) -> String {
    match e.get("message").or_else(|| e.get("data").and_then(|d| d.get("message"))).and_then(|m| m.as_str()) {
        Some(m) => m.to_owned(),
        None => e.to_string(),
    }
}

/// The errors of the `error` events in `opencode run --format json` output, if any.
fn reported_error(stdout: &str) -> Option<String> {
    let errors: Vec<String> = stdout
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with('{'))
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter(|v| v.get("type").and_then(|t| t.as_str()) == Some("error"))
        .map(|v| v.get("error").map(error_text).unwrap_or_else(|| v.to_string()))
        .collect();
    (!errors.is_empty()).then(|| tail(&errors.join("; "), 400))
}

/// Totals of one replayed conversation.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Outcome {
    pub session_id: Option<String>,
    /// Answers received (the prompt's plus each follow-up's).
    pub exchanges: u32,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub error: Option<String>,
}

/// The replay log: plain text, appended per event so it can be followed live, rotated to
/// `<name>.1` once it passes `max_bytes`.
pub struct ReplayLog {
    path: PathBuf,
    max_bytes: u64,
}

impl ReplayLog {
    pub fn new(path: PathBuf) -> Self {
        Self::with_limit(path, LOG_MAX_BYTES)
    }

    pub fn with_limit(path: PathBuf, max_bytes: u64) -> Self {
        Self { path, max_bytes }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Moves a full log aside; called before a conversation, so one never straddles two files.
    pub fn rotate_if_full(&self) {
        if std::fs::metadata(&self.path).is_ok_and(|m| m.len() >= self.max_bytes) {
            let mut old = self.path.clone().into_os_string();
            old.push(".1");
            let _ = std::fs::rename(&self.path, old);
        }
    }

    pub fn write(&self, text: &str) {
        if let Some(dir) = self.path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        match std::fs::OpenOptions::new().create(true).append(true).open(&self.path) {
            Ok(mut f) => {
                let _ = f.write_all(text.as_bytes());
            }
            Err(e) => eprintln!("[replay] cannot write {}: {e}", self.path.display()),
        }
    }
}

fn now_local() -> String {
    jiff::Zoned::now().strftime("%Y-%m-%d %H:%M:%S").to_string()
}

/// Sends the prompt, then up to `follow_ups` `Next:` suggestions in the same session. Stops
/// early when an answer has no suggestion, the suggestion is [`excluded`] by `skip_words`, or a
/// call fails.
pub fn converse(agent: &dyn Agent, cand: &Candidate, follow_ups: u32, skip_words: &[String], log: &ReplayLog) -> Outcome {
    log.rotate_if_full();
    let first_line: String = cand.prompt.lines().next().unwrap_or_default().chars().take(60).collect();
    let title = format!("{TITLE_PREFIX}{first_line}");
    let total = follow_ups + 1;
    log.write(&format!(
        "\n===== {} · {} · replay of {}/{} · up to {} /next =====\n",
        now_local(),
        cand.project,
        cand.session_id,
        cand.turn,
        follow_ups
    ));
    let mut outcome = Outcome::default();
    let mut prompt = cand.prompt.clone();
    for step in 1..=total {
        let label = if step == 1 { "prompt".to_owned() } else { "/next".to_owned() };
        log.write(&format!("--- [{step}/{total}] {label}\n{}\n", prompt.trim_end()));
        let started = Instant::now();
        match agent.send(&cand.cwd, outcome.session_id.as_deref(), &title, &prompt) {
            Ok(r) => {
                outcome.exchanges += 1;
                outcome.tokens_in += r.tokens_in;
                outcome.tokens_out += r.tokens_out;
                log.write(&format!(
                    "--- [{step}/{total}] response · {:.1} s · in {} · out {} tokens · {}\n{}\n",
                    started.elapsed().as_secs_f64(),
                    r.tokens_in,
                    r.tokens_out,
                    r.session_id,
                    r.text.trim_end()
                ));
                outcome.session_id.get_or_insert(r.session_id);
                if step == total {
                    break;
                }
                match extract_next(&r.text) {
                    Some(next) => match excluded(&next, skip_words) {
                        None => prompt = next,
                        Some(why) => {
                            log.write(&format!("--- the `Next:` line {why}; not sent, stopping\n"));
                            break;
                        }
                    },
                    None => {
                        log.write("--- no `Next:` line in the answer; stopping\n");
                        break;
                    }
                }
            }
            Err(e) => {
                log.write(&format!("--- [{step}/{total}] FAILED after {:.1} s: {e}\n", started.elapsed().as_secs_f64()));
                outcome.error = Some(e);
                break;
            }
        }
    }
    log.write(&format!(
        "===== end · {} answers · in {} · out {} tokens{} =====\n",
        outcome.exchanges,
        outcome.tokens_in,
        outcome.tokens_out,
        if outcome.error.is_some() { " · failed" } else { "" }
    ));
    outcome
}

static RUNNING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Held while a replay runs; at most one exists per process.
pub struct ReplayGuard(());

impl ReplayGuard {
    pub fn acquire() -> Option<Self> {
        use std::sync::atomic::Ordering;
        RUNNING.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_ok().then(|| Self(()))
    }
}

impl Drop for ReplayGuard {
    fn drop(&mut self) {
        RUNNING.store(false, std::sync::atomic::Ordering::Release);
    }
}

/// Today's replay count from the `replay/day` meta value `"<date> <count>"`; another day's
/// value counts as zero.
pub fn today_count(value: Option<&str>, today: &str) -> u32 {
    value.and_then(|v| v.split_once(' ')).filter(|(d, _)| *d == today).and_then(|(_, n)| n.trim().parse().ok()).unwrap_or(0)
}

/// Refuses to run without the read-only agent: `opencode run --agent` with an unknown name
/// falls back to the default agent, which edits files and runs commands.
pub fn check_agent(cfg: &ReplayConfig) -> anyhow::Result<()> {
    let file = cfg.agents_dir.join(format!("{}.md", cfg.agent));
    anyhow::ensure!(file.is_file(), "replay agent {} not found at {}; run scripts/install.ps1", cfg.agent, file.display());
    Ok(())
}

/// Records a start (time, today's count, the prompt as done) before the conversation runs, so a
/// failing or hanging prompt is not picked again and a crash does not exceed the daily cap.
pub fn record_start(state: &State, cand: &Candidate, now: &jiff::Zoned) -> anyhow::Result<()> {
    let today = now.strftime("%Y-%m-%d").to_string();
    let count = today_count(state.get_meta(DAY_COUNT)?.as_deref(), &today);
    state.set_meta(DAY_COUNT, &format!("{today} {}", count + 1))?;
    state.set_meta(LAST_RUN, &now.timestamp().as_millisecond().to_string())?;
    state.set_meta(&cand.done_key(), &now.timestamp().as_millisecond().to_string())?;
    Ok(())
}

/// `ctx replay`: one replay now, in the foreground, whatever the pacing says (the projects and
/// the agent check still apply). `dry_run` only reports what would be sent.
pub async fn run_once(cfg: &ReplayConfig, store: &Store, state: &State, agent: &dyn Agent, dry_run: bool) -> anyhow::Result<String> {
    anyhow::ensure!(!cfg.projects.is_empty(), "no projects to replay; set CTX_REPLAY_PROJECTS (e.g. helpdesk,sedn26)");
    let all = candidates(store, state, cfg).await?;
    let mut rng = Rng::from_time();
    if all.is_empty() {
        return Ok(format!("nothing to replay in {}", cfg.projects.join(", ")));
    }
    let cand = all[rng.below(all.len())].clone();
    let follow_ups = rng.between(cfg.next.0, cfg.next.1);
    if dry_run {
        return Ok(format!(
            "{} prompts left in {}\nwould replay in {} ({}) with up to {follow_ups} /next:\n{}",
            all.len(),
            cfg.projects.join(", "),
            cand.project,
            cand.cwd.display(),
            cand.prompt
        ));
    }
    check_agent(cfg)?;
    record_start(state, &cand, &jiff::Zoned::now())?;
    let log = ReplayLog::new(cfg.log.clone());
    let o = converse(agent, &cand, follow_ups, &cfg.skip_words, &log);
    let summary = format!("{}: {} answers, in {} · out {} tokens; log: {}", cand.project, o.exchanges, o.tokens_in, o.tokens_out, log.path().display());
    match o.error {
        Some(e) => anyhow::bail!("{summary}\nfailed: {e}"),
        None => Ok(summary),
    }
}

/// Called by the daemon's periodic run. Starts one replay on its own thread when one is due and
/// none is running; returns a line for the run report when it started one.
pub async fn maybe_start(cfg: &ReplayConfig, store: &Store, state: &State, now: &jiff::Zoned) -> anyhow::Result<Option<String>> {
    if !cfg.enabled {
        return Ok(None);
    }
    let Some(guard) = ReplayGuard::acquire() else { return Ok(None) };
    let last = state.get_meta(LAST_RUN)?.and_then(|v| v.parse::<i64>().ok());
    let today = now.strftime("%Y-%m-%d").to_string();
    let count = today_count(state.get_meta(DAY_COUNT)?.as_deref(), &today);
    if !due(cfg, last, count, now.timestamp().as_millisecond()) {
        return Ok(None);
    }
    check_agent(cfg)?;
    let mut rng = Rng::from_time();
    let Some(cand) = pick(store, state, cfg, &mut rng).await? else {
        // Counts as a start for pacing: the scan reads every turn of the allowed projects, and
        // index runs (one after every answer) would otherwise repeat it until a prompt appears.
        state.set_meta(LAST_RUN, &now.timestamp().as_millisecond().to_string())?;
        return Ok(None);
    };
    record_start(state, &cand, now)?;
    let follow_ups = rng.between(cfg.next.0, cfg.next.1);
    let line = format!("replay: {} with up to {follow_ups} /next", cand.project);
    let cfg = cfg.clone();
    std::thread::spawn(move || {
        let _guard = guard;
        let log = ReplayLog::new(cfg.log.clone());
        let o = converse(&OpencodeCli::from_config(&cfg), &cand, follow_ups, &cfg.skip_words, &log);
        match o.error {
            Some(e) => eprintln!("[replay] {} failed after {} answers: {e}", cand.project, o.exchanges),
            None => eprintln!("[replay] {}: {} answers, in {} · out {} tokens", cand.project, o.exchanges, o.tokens_in, o.tokens_out),
        }
    });
    Ok(Some(line))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::Mutex;

    fn cfg() -> ReplayConfig {
        ReplayConfig {
            enabled: true,
            projects: vec!["helpdesk".into()],
            every: Duration::from_secs(60 * 60),
            daily_max: 3,
            next: (2, 5),
            agent: "ctx-replay".into(),
            agents_dir: PathBuf::from("Z:/no/such/agents"),
            model: None,
            exe: PathBuf::from("Z:/no/such/opencode.exe"),
            timeout: Duration::from_secs(7),
            min_chars: 40,
            skip_words: parse_skip_words(None),
            log: PathBuf::from("Z:/no/such/opencode.log"),
        }
    }

    fn words(w: &[&str]) -> Vec<String> {
        w.iter().map(|s| (*s).to_owned()).collect()
    }

    mod parse_skip_words {
        use super::*;

        #[test]
        fn unset_gives_the_database_words_and_a_value_replaces_them() {
            let defaults: &[&str] = &["db", "dbs", "database", "databases", "sql", "psql", "sqlplus", "baza", "bazy", "bazie", "bazę", "bazą", "bazach", "bazami", "bazo", "baz"];
            let rows: [(&str, Option<&str>, &[&str]); 7] = [
                ("unset", None, defaults),
                ("empty means none", Some(""), &[]),
                ("only blanks and commas", Some(" , ,"), &[]),
                ("one word", Some("staging"), &["staging"]),
                ("trimmed, blanks dropped, lowercased", Some(" Prod , ,STAGING,,"), &["prod", "staging"]),
                ("non-ASCII capitals are lowercased", Some("BAZĘ,Łódź"), &["bazę", "łódź"]),
                ("a value does not add the defaults", Some("mongo"), &["mongo"]),
            ];
            for (name, input, want) in rows {
                assert_eq!(parse_skip_words(input), want, "{name}: {input:?}");
            }
        }
    }

    mod excluded {
        use super::*;

        const CLAUDE: Option<&str> = Some("mentions Claude");

        #[test]
        fn anything_mentioning_claude_is_excluded_whatever_the_skip_words() {
            let long_tail = format!("{} see CLAUDE.md", "ż".repeat(12_001));
            // `None`: the default skip words.
            let rows: [(&str, &str, Option<&[&str]>); 9] = [
                ("the name", "ask Claude about the parser", Some(&[])),
                ("upper case file", "update CLAUDE.md with the rule", Some(&[])),
                ("a dot directory", "edit .claude/settings.json hooks", Some(&["staging"])),
                ("hyphenated", "how do claude-code hooks run", Some(&[])),
                ("inside a longer word", "the xclaudex tool", Some(&[])),
                ("at the very start", "Claude: why", Some(&[])),
                ("past the 12000 chars a replay keeps", &long_tail, Some(&[])),
                ("wins over a skip word earlier in the text", "the db that Claude made", Some(&["db"])),
                ("with the default skip words", "ask claude", None),
            ];
            for (name, text, skip) in rows {
                let skip = skip.map_or_else(|| parse_skip_words(None), words);
                assert_eq!(excluded(text, &skip).as_deref(), CLAUDE, "{name}: {text:.60}");
            }
        }

        #[test]
        fn a_skip_word_counts_only_as_a_whole_word_in_any_case() {
            let defaults = parse_skip_words(None);
            let rows: [(&str, &str, Option<&str>); 22] = [
                ("a plain word", "check on prod db", Some("has the skip word `db`")),
                ("joined by an underscore", "prod_db is slow", Some("has the skip word `db`")),
                ("upper case", "DB_HOST is wrong", Some("has the skip word `db`")),
                ("before a question mark", "is the database up?", Some("has the skip word `database`")),
                ("joined by a hyphen", "sql-injection risk here", Some("has the skip word `sql`")),
                ("in parentheses", "run it (psql) first", Some("has the skip word `psql`")),
                ("a Polish form, whole", "Sprawdź bazę danych", Some("has the skip word `bazę`")),
                ("a Polish form in capitals", "SPRAWDŹ BAZĄ", Some("has the skip word `bazą`")),
                ("the first skip word in the text is reported", "the sql query on the db", Some("has the skip word `sql`")),
                ("at the end after a newline", "one\ntwo db", Some("has the skip word `db`")),
                ("a longer word ending in one", "mongodb is slow", None),
                ("a longer word starting with one", "dbg build fails", None),
                ("sqlite", "sqlite file locked", None),
                ("digits are part of the word", "db2 driver", None),
                ("Polish letters are part of the word", "zażółćdb gęśl", None),
                ("a Polish stem inside a longer word", "bazooka", None),
                ("a Polish letter inside keeps the word whole", "bazęx", None),
                ("no skip word", "why does login fail", None),
                ("empty text", "", None),
                ("only separators", " ,.;", None),
                ("baz is a word on its own", "baz", Some("has the skip word `baz`")),
                ("bazy", "dwie bazy", Some("has the skip word `bazy`")),
            ];
            for (name, text, want) in rows {
                assert_eq!(excluded(text, &defaults).as_deref(), want, "{name}: {text}");
            }
        }

        #[test]
        fn the_given_skip_words_replace_the_defaults() {
            let rows: [(&str, &str, &[&str], Option<&str>); 5] = [
                ("a custom word", "deploy to staging now", &["staging"], Some("has the skip word `staging`")),
                ("the second custom word", "deploy to prod now", &["staging", "prod"], Some("has the skip word `prod`")),
                ("a default word is not one of them", "check the db", &["staging"], None),
                ("no skip words", "check the db", &[], None),
                ("a blank skip word never matches the gap between two separators", "a  b", &[""], None),
            ];
            for (name, text, skip, want) in rows {
                assert_eq!(excluded(text, &words(skip)).as_deref(), want, "{name}");
            }
        }
    }

    mod parse_range {
        use super::*;

        #[test]
        fn reads_a_range_a_single_number_and_falls_back_on_garbage() {
            // Default (7, 9) differs from every parsed value, so a fallback cannot pass as a parse.
            let rows: [(&str, Option<&str>, (u32, u32)); 14] = [
                ("a range", Some("2-5"), (2, 5)),
                ("a single number", Some("3"), (3, 3)),
                ("a reversed range is put in order", Some("5-2"), (2, 5)),
                ("spaces around the parts", Some(" 1 - 4 "), (1, 4)),
                ("equal ends", Some("4-4"), (4, 4)),
                ("zero", Some("0"), (0, 0)),
                ("unset", None, (7, 9)),
                ("empty", Some(""), (7, 9)),
                ("blank", Some("   "), (7, 9)),
                ("words", Some("two-five"), (7, 9)),
                ("trailing separator", Some("2-"), (7, 9)),
                ("leading separator", Some("-2"), (7, 9)),
                ("extra parts", Some("2-5-7"), (7, 9)),
                ("a decimal", Some("2.5"), (7, 9)),
            ];
            for (name, input, want) in rows {
                assert_eq!(parse_range(input, (7, 9)), want, "{name}: {input:?}");
            }
        }
    }

    mod parse_projects {
        use super::*;

        #[test]
        fn trims_names_and_drops_blanks() {
            let rows: [(Option<&str>, &[&str]); 6] = [
                (None, &[]),
                (Some(""), &[]),
                (Some(" , ,"), &[]),
                (Some("sedn26"), &["sedn26"]),
                (Some(" helpdesk , ,Sedn26,,"), &["helpdesk", "Sedn26"]),
                (Some("a b,c"), &["a b", "c"]),
            ];
            for (input, want) in rows {
                assert_eq!(parse_projects(input), want, "{input:?}");
            }
        }
    }

    mod project_matches {
        use super::*;

        fn names(n: &[&str]) -> Vec<String> {
            n.iter().map(|s| (*s).to_owned()).collect()
        }

        #[test]
        fn a_project_matches_when_some_path_segment_equals_a_name() {
            let rows: [(&str, &str, &[&str], bool); 13] = [
                ("a middle segment", "d:/dev/sedn/sedn26/sedn-data", &["sedn26"], true),
                ("the last segment", "d:/dev/sedn/sedn26", &["sedn26"], true),
                ("a longer segment is no match", "d:/dev/sedn26x", &["sedn26"], false),
                ("a shorter segment is no match", "d:/dev/sedn2", &["sedn26"], false),
                ("case of the project is ignored", "d:/dev/SEDN26/api", &["sedn26"], true),
                ("case of the name is ignored", "d:/dev/sedn26/api", &["SeDn26"], true),
                ("backslash separators", r"D:\dev\helpdesk\api", &["helpdesk"], true),
                ("the second name matches", "d:/dev/helpdesk", &["sedn26", "helpdesk"], true),
                ("no name matches", "d:/dev/other", &["sedn26", "helpdesk"], false),
                ("a blank name never matches", "d:/dev//x", &[""], false),
                ("a whitespace name never matches", "d:/dev/ /x", &["  "], false),
                ("a name with spaces around it", "d:/dev/helpdesk", &[" helpdesk "], true),
                ("no names", "d:/dev/helpdesk", &[], false),
            ];
            for (name, project, ns, want) in rows {
                assert_eq!(project_matches(project, &names(ns)), want, "{name}: {project} vs {ns:?}");
            }
        }
    }

    mod prompt_of {
        use super::*;

        #[test]
        fn takes_the_users_prompt_out_of_a_turn_chunk() {
            let rows: [(&str, &str, Option<&str>); 13] = [
                ("header and assistant", "[Login fix · 2026-09-20]\nUser: why does login fail?\n\nAssistant: because", Some("why does login fail?")),
                ("no header", "User: plain question\n\nAssistant: x", Some("plain question")),
                ("ends at Tools", "[t · d]\nUser: list the files\n\nTools: read a.rs", Some("list the files")),
                ("ends at Files", "[t · d]\nUser: edit it\n\nFiles: a.rs", Some("edit it")),
                ("the first marker wins", "[t · d]\nUser: first\n\nFiles: a.rs\n\nAssistant: later", Some("first")),
                ("no marker: the rest", "[t · d]\nUser: the whole rest", Some("the whole rest")),
                ("paragraphs inside the prompt stay", "[t · d]\nUser: para one\nline two\n\npara two\n\nAssistant: x", Some("para one\nline two\n\npara two")),
                ("a marker word without the blank line stays", "User: what does Assistant: mean here", Some("what does Assistant: mean here")),
                ("trimmed", "[t · d]\nUser:   spaced out  \n\nAssistant: x", Some("spaced out")),
                ("no User prefix", "[t · d]\nAssistant: only an answer", None),
                ("an empty prompt", "[t · d]\nUser: \n\nAssistant: x", None),
                ("a header without a newline", "[only a header", None),
                ("empty text", "", None),
            ];
            for (name, text, want) in rows {
                assert_eq!(prompt_of(text).as_deref(), want, "{name}");
            }
        }
    }

    mod usable {
        use super::*;

        #[test]
        fn needs_min_chars_and_no_command_or_harness_message() {
            // Polish letters are two bytes each: a byte count would call the 9-char one long enough.
            let rows: [(&str, &str, bool); 13] = [
                ("exactly min_chars", "abcdefghij", true),
                ("one short", "abcdefghi", false),
                ("multibyte, exactly min_chars", "zażółćgęśl", true),
                ("multibyte, one short (18 bytes)", "zażółćgęś", false),
                ("a slash command", "/review this pull request", false),
                ("a slash command after spaces", "   /review this pull request", false),
                ("a shell escape", "!git status --short --branch", false),
                ("a command tag inside", "please <command-name>x</command-name> run", false),
                ("an interruption", "[Request interrupted by user for tool use]", false),
                ("a slash later is fine", "explain the / operator in rust", true),
                ("a bang later is fine", "why does it print hello! twice", true),
                ("another bracket is fine", "[note] why does the parser fail", true),
                ("empty", "", false),
            ];
            for (name, prompt, want) in rows {
                assert_eq!(usable(prompt, 10), want, "{name}: {prompt:?}");
            }
        }
    }

    mod extract_next {
        use super::*;

        #[test]
        fn finds_the_last_next_line_outside_fences() {
            let rows: [(&str, &str, Option<&str>); 24] = [
                ("plain", "Done.\nNext: run the tests", Some("run the tests")),
                ("Polish: 'ó' sits on bytes 6..8 after Next, where the action check slices", "Gotowe.\nNext: zażółć gęślą jaźń", Some("zażółć gęślą jaźń")),
                ("Next and Polish text: 'ź' across byte 7 after Next is no label", "Nawigacja:\nNext łódź\nkoniec", None),
                ("the last one wins", "Next: first idea\nmore text\nNext: second idea\nbye", Some("second idea")),
                ("Next action", "Next action: deploy it", Some("deploy it")),
                ("bold label", "**Next:** bold one", Some("bold one")),
                ("bold word, colon outside", "**Next**: bold two", Some("bold two")),
                ("bold body", "Next: **bold body**", Some("bold body")),
                ("upper case", "NEXT: upper case", Some("upper case")),
                ("lower case action", "next action: lower case", Some("lower case")),
                ("mixed case action", "Next Action: mixed case", Some("mixed case")),
                ("indented", "   Next: indented one", Some("indented one")),
                ("inside ``` it does not count", "Next: outside\n```\nNext: inside\n```", Some("outside")),
                ("inside ~~~ it does not count", "Next: outside\n~~~\nNext: inside\n~~~", Some("outside")),
                ("after a closed fence it counts", "```rust\nNext: inside\n```\nNext: after", Some("after")),
                ("~~~ does not close ```", "```\n~~~\nNext: inside\n```\nNext: after", Some("after")),
                ("an unclosed fence hides the rest", "Next: before\n```\nNext: inside", Some("before")),
                ("the colon is required", "Next steps are unclear", None),
                ("a longer word is no label", "Nextly: x", None),
                ("empty suggestion", "Next:", None),
                ("blank suggestion", "Next:    ", None),
                ("no Next line", "Just an answer.", None),
                ("empty text", "", None),
                ("a fenced-only Next", "```\nNext: inside\n```", None),
            ];
            for (name, text, want) in rows {
                assert_eq!(extract_next(text).as_deref(), want, "{name}");
            }
        }

        #[test]
        fn drops_a_closing_question_and_trailing_punctuation() {
            let rows: [(&str, &str, &str); 20] = [
                ("want me to? after an em dash", "Next: run the migration — want me to?", "run the migration"),
                ("want me to after a comma", "Next: run the migration, want me to", "run the migration"),
                ("shall I?", "Next: add the index, shall I?", "add the index"),
                ("shall i after a dash", "Next: add the index - shall i", "add the index"),
                ("Should I? after a full stop", "Next: drop the column. Should I?", "drop the column"),
                ("ok?", "Next: commit the fix, ok?", "commit the fix"),
                ("on your go after an en dash", "Next: deploy to staging – on your go", "deploy to staging"),
                ("on your go before a full stop", "Next: deploy to staging — on your go.", "deploy to staging"),
                ("when you're ready", "Next: rerun it when you're ready", "rerun it"),
                ("when youre ready", "Next: rerun it when youre ready", "rerun it"),
                ("awaiting your approval", "Next: merge it, awaiting your approval", "merge it"),
                ("if you approve", "Next: merge it if you approve", "merge it"),
                ("two tails", "Next: ship it, if you approve, ok?", "ship it"),
                ("a trailing full stop", "Next: run cargo test.", "run cargo test"),
                ("a trailing comma", "Next: run cargo test,", "run cargo test"),
                ("ok? inside a word is kept: webhook?", "Next: update the webhook?", "update the webhook?"),
                ("ok? inside a word is kept: runbook?", "Next: check the runbook?", "check the runbook?"),
                ("ok? right after a dash", "Next: update the webhook -ok?", "update the webhook"),
                ("ok? right after a comma", "Next: update the webhook,ok?", "update the webhook"),
                ("shall I? right after a dash", "Next: rotate the key—shall I?", "rotate the key"),
            ];
            for (name, text, want) in rows {
                assert_eq!(extract_next(text).as_deref(), Some(want), "{name}: {text}");
            }
        }

        #[test]
        fn a_suggestion_over_400_chars_is_none() {
            // Counted in chars: 400 chars here are 792 bytes.
            let body = |n: usize| format!("abcdefgh{}", "ż".repeat(n - 8));
            assert_eq!(extract_next(&format!("Next: {}", body(400))), Some(body(400)), "400 chars is allowed");
            assert_eq!(extract_next(&format!("Next: {}", body(401))), None);
        }
    }

    mod due {
        use super::*;

        const HOUR_MS: i64 = 60 * 60 * 1000;
        const NOW: i64 = 1_794_907_800_000;

        #[test]
        fn starts_only_when_on_named_under_the_cap_and_every_has_passed() {
            let off = ReplayConfig { enabled: false, ..cfg() };
            let unnamed = ReplayConfig { projects: Vec::new(), ..cfg() };
            let zero_cap = ReplayConfig { daily_max: 0, ..cfg() };
            let rows: [(&str, ReplayConfig, Option<i64>, u32, bool); 9] = [
                ("never started", cfg(), None, 0, true),
                ("every passed exactly", cfg(), Some(NOW - HOUR_MS), 0, true),
                ("every passed long ago", cfg(), Some(NOW - 5 * HOUR_MS), 2, true),
                ("1 ms short of every", cfg(), Some(NOW - HOUR_MS + 1), 0, false),
                ("at the daily cap", cfg(), None, 3, false),
                ("one below the cap", cfg(), None, 2, true),
                ("a zero cap", zero_cap, None, 0, false),
                ("off", off, None, 0, false),
                ("no projects", unnamed, None, 0, false),
            ];
            for (name, c, last, count, want) in rows {
                assert_eq!(due(&c, last, count, NOW), want, "{name}");
            }
        }
    }

    mod today_count {
        use super::*;

        #[test]
        fn reads_todays_count_and_zero_otherwise() {
            let rows: [(Option<&str>, u32); 7] = [
                (Some("2026-11-17 4"), 4),
                (Some("2026-11-17  7"), 7),
                (Some("2026-11-16 4"), 0),
                (None, 0),
                (Some("garbage"), 0),
                (Some("2026-11-17 x"), 0),
                (Some("2026-11-17"), 0),
            ];
            for (value, want) in rows {
                assert_eq!(today_count(value, "2026-11-17"), want, "{value:?}");
            }
        }
    }

    #[test]
    fn done_key_names_the_session_and_turn() {
        assert_eq!(done_key("ses-a", "u7"), "replay/done/ses-a/u7");
    }

    mod rng {
        use super::*;

        #[test]
        fn is_splitmix64() {
            // Reference outputs of splitmix64 for seed 0.
            let mut r = Rng::seeded(0);
            assert_eq!([r.next(), r.next(), r.next()], [0xE220_A839_7B1D_CDAF, 0x6E78_9E6A_A1B9_65F4, 0x06C4_5D18_8009_454F]);
        }

        #[test]
        fn the_same_seed_gives_the_same_draws_and_another_seed_others() {
            let draws = |seed| {
                let mut r = Rng::seeded(seed);
                (0..20).map(|_| r.below(1000)).collect::<Vec<_>>()
            };
            assert_eq!(draws(42), draws(42));
            assert_ne!(draws(42), draws(43));
        }

        #[test]
        fn below_and_between_stay_in_range_and_reach_both_ends() {
            let mut r = Rng::seeded(7);
            let mut below: Vec<usize> = (0..500).map(|_| r.below(3)).collect();
            below.sort();
            below.dedup();
            assert_eq!(below, [0, 1, 2]);
            let mut between: Vec<u32> = (0..500).map(|_| r.between(2, 5)).collect();
            between.sort();
            between.dedup();
            assert_eq!(between, [2, 3, 4, 5]);
            assert_eq!((0..20).map(|_| r.between(4, 4)).collect::<Vec<_>>(), [4; 20]);
        }
    }

    mod args {
        use super::*;

        fn cli(model: Option<&str>) -> OpencodeCli {
            OpencodeCli { exe: "opencode".into(), agent: "ro-agent".into(), model: model.map(str::to_owned), timeout: Duration::from_secs(1) }
        }

        #[test]
        fn new_sessions_are_archived_and_titled_continued_ones_name_the_session_and_the_prompt_is_last() {
            let rows: [(&str, Option<&str>, Option<&str>, &str, &[&str]); 4] = [
                ("new, no model", None, None, "why?", &["run", "--format", "json", "--agent", "ro-agent", "--archived", "--title", "ctx-replay · t", "why?"]),
                ("new, with a model", Some("opi/glm-5"), None, "why?", &["run", "--format", "json", "--agent", "ro-agent", "--model", "opi/glm-5", "--archived", "--title", "ctx-replay · t", "why?"]),
                ("continued, with a model", Some("opi/glm-5"), Some("ses_7"), "go on", &["run", "--format", "json", "--agent", "ro-agent", "--model", "opi/glm-5", "--session", "ses_7", "go on"]),
                ("continued, no model", None, Some("ses_8"), "go deeper", &["run", "--format", "json", "--agent", "ro-agent", "--session", "ses_8", "go deeper"]),
            ];
            for (name, model, session, prompt, want) in rows {
                assert_eq!(cli(model).args(session, "ctx-replay · t", prompt), want, "{name}");
            }
        }

        #[test]
        fn a_prompt_starting_with_a_dash_gets_a_leading_space() {
            assert_eq!(cli(None).args(Some("s"), "t", "-v means verbose?").last().unwrap(), " -v means verbose?");
            assert_eq!(cli(None).args(Some("s"), "t", "a-b means?").last().unwrap(), "a-b means?");
        }
    }

    mod parse_reply {
        use super::*;

        const SAMPLE: &str = r#"{"type":"step_start","timestamp":1,"sessionID":"ses_X","part":{"type":"step-start"}}
{"type":"text","timestamp":2,"sessionID":"ses_X","part":{"type":"text","text":"ok\nNext: say hi"}}
{"type":"step_finish","timestamp":3,"sessionID":"ses_X","part":{"type":"step-finish","reason":"stop","tokens":{"input":86,"output":8,"reasoning":0,"cache":{"read":0,"write":20352}}}}
"#;

        fn reply(session: &str, text: &str, tokens_in: u64, tokens_out: u64) -> Reply {
            Reply { session_id: session.into(), text: text.into(), tokens_in, tokens_out }
        }

        #[test]
        fn reads_the_real_sample() {
            assert_eq!(parse_reply(SAMPLE), Ok(reply("ses_X", "ok\nNext: say hi", 86, 8)));
        }

        #[test]
        fn joins_texts_and_sums_tokens_over_steps_with_reasoning_as_output() {
            let out = r#"INFO starting opencode
{"type":"text","sessionID":"ses_A","part":{"type":"text","text":" first part \n"}}
{"type":"step_finish","sessionID":"ses_B","part":{"tokens":{"input":10,"output":3,"reasoning":4}}}
{broken json
{"type":"text","sessionID":"ses_B","part":{"type":"text","text":"   "}}
{"type":"text","sessionID":"ses_B","part":{"type":"text","text":"second part"}}
{"type":"step_finish","sessionID":"ses_B","part":{"tokens":{"input":20,"output":5,"reasoning":6}}}
{"type":"step_finish","sessionID":"ses_B","part":{"type":"step-finish"}}
"#;
            assert_eq!(parse_reply(out), Ok(reply("ses_A", "first part\n\nsecond part", 30, 18)));
        }

        #[test]
        fn the_session_comes_from_the_first_event_that_has_one() {
            let out = "{\"type\":\"step_start\"}\n{\"type\":\"text\",\"sessionID\":\"ses_late\",\"part\":{\"text\":\"hi\"}}\n{\"type\":\"text\",\"sessionID\":\"ses_later\",\"part\":{\"text\":\"there\"}}";
            assert_eq!(parse_reply(out), Ok(reply("ses_late", "hi\n\nthere", 0, 0)));
        }

        #[test]
        fn an_error_without_text_is_an_err() {
            // The error value is reported as JSON, so a string keeps its quotes.
            let out = "{\"type\":\"step_start\",\"sessionID\":\"ses_E\"}\n{\"type\":\"error\",\"sessionID\":\"ses_E\",\"error\":\"rate limited\"}";
            assert_eq!(parse_reply(out), Err("opencode reported an error: \"rate limited\"".to_owned()));
        }

        #[test]
        fn an_error_without_text_reports_the_errors_message() {
            let start = "{\"type\":\"step_start\",\"sessionID\":\"ses_M\"}\n";
            let model_not_found = "{\"type\":\"error\",\"timestamp\":5,\"sessionID\":\"ses_M\",\"error\":{\"type\":\"provider.invalid-request\",\"message\":\"Model not found\",\"status\":400,\"providerID\":\"opi\"}}";
            let rows: [(&str, String, &str); 5] = [
                ("message, as OpenCode 2.2.7 prints it", format!("{start}{model_not_found}"), "Model not found"),
                ("data.message", format!("{start}{{\"type\":\"error\",\"error\":{{\"name\":\"APIError\",\"data\":{{\"message\":\"Rate limit hit\"}}}}}}"), "Rate limit hit"),
                (
                    "message wins over data.message",
                    format!("{start}{{\"type\":\"error\",\"error\":{{\"message\":\"outer\",\"data\":{{\"message\":\"inner\"}}}}}}"),
                    "outer",
                ),
                ("no message: the whole error", format!("{start}{{\"type\":\"error\",\"error\":{{\"name\":\"UnknownError\"}}}}"), "{\"name\":\"UnknownError\"}"),
                (
                    "two errors, in order",
                    format!("{start}{{\"type\":\"error\",\"error\":{{\"message\":\"first failure\"}}}}\n{{\"type\":\"error\",\"error\":{{\"message\":\"second failure\"}}}}"),
                    "first failure; second failure",
                ),
            ];
            for (name, out, want) in rows {
                assert_eq!(parse_reply(&out), Err(format!("opencode reported an error: {want}")), "{name}");
            }
        }

        #[test]
        fn an_error_event_without_an_error_field_reports_its_line() {
            let line = "{\"type\":\"error\",\"sessionID\":\"ses_E\"}";
            assert_eq!(parse_reply(line), Err(format!("opencode reported an error: {line}")));
        }

        #[test]
        fn an_error_next_to_text_keeps_the_text() {
            let out = "{\"type\":\"text\",\"sessionID\":\"ses_T\",\"part\":{\"text\":\"partial answer\"}}\n{\"type\":\"error\",\"sessionID\":\"ses_T\",\"error\":\"aborted\"}";
            assert_eq!(parse_reply(out), Ok(reply("ses_T", "partial answer", 0, 0)));
        }

        #[test]
        fn no_session_is_an_err_with_the_output_tail() {
            let no_id = "{\"type\":\"text\",\"part\":{\"text\":\"hi\"}}";
            assert_eq!(parse_reply(no_id), Err(format!("no session in opencode output: {no_id}")));
            assert_eq!(parse_reply(""), Err("no session in opencode output: ".to_owned()));
            let long = format!("{}{}", "x".repeat(250), "y".repeat(100));
            assert_eq!(parse_reply(&long), Err(format!("no session in opencode output: {}{}", "x".repeat(200), "y".repeat(100))));
        }

        #[test]
        fn tail_keeps_the_last_n_chars_of_the_trimmed_text() {
            assert_eq!(tail("  abcdef \r\n", 3), "def");
            assert_eq!(tail("żółw", 2), "łw");
            assert_eq!(tail("ab", 5), "ab");
            assert_eq!(tail("abc", 3), "abc");
        }
    }

    mod send {
        use super::*;

        const SAMPLE: &str = "{\"type\":\"step_start\",\"timestamp\":1,\"sessionID\":\"ses_X\",\"part\":{\"type\":\"step-start\"}}\r\n{\"type\":\"text\",\"timestamp\":2,\"sessionID\":\"ses_X\",\"part\":{\"type\":\"text\",\"text\":\"ok\\nNext: say hi\"}}\r\n{\"type\":\"step_finish\",\"timestamp\":3,\"sessionID\":\"ses_X\",\"part\":{\"type\":\"step-finish\",\"reason\":\"stop\",\"tokens\":{\"input\":86,\"output\":8,\"reasoning\":0,\"cache\":{\"read\":0,\"write\":20352}}}}\r\n";

        /// A stand-in `opencode`: a batch file with the given body (Windows runs .cmd via cmd.exe).
        fn fake(dir: &Path, body: &str, model: Option<&str>, timeout: Duration) -> OpencodeCli {
            let exe = dir.join("fake-opencode.cmd");
            std::fs::write(&exe, format!("@echo off\r\n{body}\r\n")).unwrap();
            OpencodeCli { exe, agent: "ctx-replay".into(), model: model.map(str::to_owned), timeout }
        }

        /// A project directory holding `reply.jsonl`, which the fake can only find from there.
        fn project(dir: &Path) -> PathBuf {
            let p = dir.join("project");
            std::fs::create_dir_all(&p).unwrap();
            std::fs::write(p.join("reply.jsonl"), SAMPLE).unwrap();
            p
        }

        #[test]
        fn runs_in_the_cwd_and_parses_exit_0() {
            let tmp = tempfile::tempdir().unwrap();
            let c = fake(tmp.path(), "type reply.jsonl", None, Duration::from_secs(20));
            let r = c.send(&project(tmp.path()), None, "ctx-replay-t", "hello there");
            assert_eq!(r, Ok(Reply { session_id: "ses_X".into(), text: "ok\nNext: say hi".into(), tokens_in: 86, tokens_out: 8 }));
        }

        #[test]
        fn passes_the_arguments_to_the_executable() {
            let rows: [(Option<&str>, Option<&str>, &str); 2] = [
                (None, None, "run --format json --agent ctx-replay --archived --title ctx-replay-t \"hello there\""),
                (Some("opi/glm-5"), Some("ses_prev"), "run --format json --agent ctx-replay --model opi/glm-5 --session ses_prev \"hello there\""),
            ];
            for (model, session, want) in rows {
                let tmp = tempfile::tempdir().unwrap();
                let c = fake(tmp.path(), "echo %*> \"%~dp0args.txt\"\r\ntype reply.jsonl", model, Duration::from_secs(20));
                c.send(&project(tmp.path()), session, "ctx-replay-t", "hello there").unwrap();
                assert_eq!(std::fs::read_to_string(tmp.path().join("args.txt")).unwrap(), format!("{want}\r\n"), "{session:?}");
            }
        }

        #[test]
        fn a_non_zero_exit_is_an_err_with_the_stderr_tail() {
            let tmp = tempfile::tempdir().unwrap();
            let c = fake(tmp.path(), "type reply.jsonl\r\n>&2 echo auth expired\r\nexit /b 3", None, Duration::from_secs(20));
            assert_eq!(c.send(&project(tmp.path()), None, "t", "p"), Err("opencode exited with exit code: 3: auth expired".to_owned()));
        }

        #[test]
        fn a_non_zero_exit_reports_the_error_events_on_stdout_before_stderr() {
            let not_found = "{\"type\":\"error\",\"timestamp\":5,\"sessionID\":\"ses_M\",\"error\":{\"type\":\"provider.invalid-request\",\"message\":\"Model not found\",\"status\":400,\"providerID\":\"opi\"}}";
            let quota = "{\"type\":\"error\",\"error\":{\"name\":\"APIError\",\"data\":{\"message\":\"Quota exceeded\"}}}";
            let long = format!("{{\"type\":\"error\",\"error\":{{\"message\":\"{}{}\"}}}}", "a".repeat(100), "b".repeat(400));
            // (name, stdout, stderr line or none, exit code, the reason after `exit code: N: `)
            let rows: [(&str, String, Option<&str>, u8, String); 7] = [
                ("OpenCode 2.2.7: the event on stdout, stderr empty", format!("INFO starting\r\n{not_found}\r\n"), None, 1, "Model not found".into()),
                ("the event wins over stderr", format!("{not_found}\r\n"), Some("auth expired"), 4, "Model not found".into()),
                ("data.message", format!("{quota}\r\n"), None, 2, "Quota exceeded".into()),
                ("two events joined in order, broken lines skipped", format!("{{broken\r\n{not_found}\r\n{quota}\r\n"), None, 1, "Model not found; Quota exceeded".into()),
                ("an event without an error field: the event", "{\"sessionID\":\"ses_E\",\"type\":\"error\"}\r\n".into(), None, 1, "{\"sessionID\":\"ses_E\",\"type\":\"error\"}".into()),
                ("only the last 400 chars", format!("{long}\r\n"), None, 1, "b".repeat(400)),
                ("other events only: the stderr tail", SAMPLE.to_owned(), Some("auth expired"), 5, "auth expired".into()),
            ];
            for (name, stdout, stderr, code, want) in rows {
                let tmp = tempfile::tempdir().unwrap();
                let p = project(tmp.path());
                std::fs::write(p.join("out.jsonl"), &stdout).unwrap();
                let err = stderr.map(|e| format!(">&2 echo {e}\r\n")).unwrap_or_default();
                let c = fake(tmp.path(), &format!("type out.jsonl\r\n{err}exit /b {code}"), None, Duration::from_secs(20));
                assert_eq!(c.send(&p, None, "t", "p"), Err(format!("opencode exited with exit code: {code}: {want}")), "{name}");
            }
        }

        #[test]
        fn unparseable_output_is_an_err_with_stderr_when_there_is_any() {
            let rows = [("echo not json", "no session in opencode output: not json"), ("echo not json\r\n>&2 echo warn: slow", "no session in opencode output: not json; stderr: warn: slow")];
            for (body, want) in rows {
                let tmp = tempfile::tempdir().unwrap();
                let c = fake(tmp.path(), body, None, Duration::from_secs(20));
                assert_eq!(c.send(&project(tmp.path()), None, "t", "p"), Err(want.to_owned()), "{body}");
            }
        }

        #[test]
        fn the_timeout_kills_the_whole_tree_and_returns_promptly() {
            let tmp = tempfile::tempdir().unwrap();
            let marker = tmp.path().join("marker.txt");
            let script = tmp.path().join("grandchild.bat");
            // A grandchild that outlives a killed parent would write the marker after ~3 s.
            std::fs::write(&script, format!("@ping -n 4 127.0.0.1 >nul\r\n@echo done> \"{}\"\r\n", marker.display())).unwrap();
            assert!(!script.to_string_lossy().contains(' '), "the command line below assumes no spaces: {}", script.display());
            let c = fake(tmp.path(), &format!("cmd /C {}", script.display()), None, Duration::from_millis(1500));

            let started = Instant::now();
            let r = c.send(&project(tmp.path()), None, "t", "p");
            let took = started.elapsed();

            assert_eq!(r, Err("opencode killed after 1.5s".to_owned()));
            assert!(took < Duration::from_secs(6), "returned after {took:?}");
            std::thread::sleep(Duration::from_secs(5));
            assert!(!marker.exists(), "the grandchild survived the kill and finished its work");
        }

        #[test]
        fn output_held_open_by_a_grandchild_after_exit_is_an_err() {
            let tmp = tempfile::tempdir().unwrap();
            // The answer is printed, then a background ping (~14 s) inherits stdout and keeps it
            // open after the shim exits 0.
            let c = fake(tmp.path(), "type reply.jsonl\r\nstart /b \"\" ping -n 15 127.0.0.1", None, Duration::from_secs(20));

            let r = c.send(&project(tmp.path()), None, "t", "p");

            assert_eq!(r, Err("opencode exited but a process it started kept its output open; the answer was lost".to_owned()));
        }

        #[test]
        fn a_missing_executable_names_ctx_opencode_exe() {
            let tmp = tempfile::tempdir().unwrap();
            let c = OpencodeCli { exe: "Z:/no/such/opencode.exe".into(), agent: "a".into(), model: None, timeout: Duration::from_secs(5) };
            let e = c.send(tmp.path(), None, "t", "p").unwrap_err();
            // The middle is the OS error text.
            assert!(e.starts_with("cannot start Z:/no/such/opencode.exe ("), "{e}");
            assert!(e.ends_with("); check CTX_OPENCODE_EXE"), "{e}");
        }
    }

    /// An `Agent` that answers from a script and records every call.
    struct Scripted {
        replies: Mutex<VecDeque<Result<Reply, String>>>,
        calls: Mutex<Vec<(PathBuf, Option<String>, String, String)>>,
    }

    impl Scripted {
        fn new(replies: Vec<Result<Reply, String>>) -> Self {
            Self { replies: Mutex::new(replies.into()), calls: Mutex::default() }
        }
        fn calls(&self) -> Vec<(PathBuf, Option<String>, String, String)> {
            self.calls.lock().unwrap().clone()
        }
        fn prompts(&self) -> Vec<String> {
            self.calls().into_iter().map(|c| c.3).collect()
        }
        fn sessions(&self) -> Vec<Option<String>> {
            self.calls().into_iter().map(|c| c.1).collect()
        }
    }

    impl Agent for Scripted {
        fn send(&self, cwd: &Path, session: Option<&str>, title: &str, prompt: &str) -> Result<Reply, String> {
            self.calls.lock().unwrap().push((cwd.to_path_buf(), session.map(str::to_owned), title.to_owned(), prompt.to_owned()));
            self.replies.lock().unwrap().pop_front().unwrap_or_else(|| Err("no scripted reply left".into()))
        }
    }

    fn ok(session: &str, text: &str, tokens_in: u64, tokens_out: u64) -> Result<Reply, String> {
        Ok(Reply { session_id: session.into(), text: text.into(), tokens_in, tokens_out })
    }

    const PROMPT: &str = "Why does the login form reject valid passwords after the upgrade?";

    fn cand(cwd: &Path) -> Candidate {
        Candidate { session_id: "ses-src".into(), turn: "u42".into(), project: "d:/dev/helpdesk".into(), cwd: cwd.to_path_buf(), prompt: PROMPT.into() }
    }

    /// The log with its wall-clock parts (start time, seconds per call) replaced by `<ts>` and
    /// `<s>`, after checking their format.
    fn normalized(path: &Path) -> String {
        let log = std::fs::read_to_string(path).unwrap();
        let secs = |line: &str, after: &str, before: &str| -> String {
            let start = line.find(after).unwrap() + after.len();
            let end = start + line[start..].find(before).unwrap();
            let n = &line[start..end];
            assert!(n.parse::<f64>().is_ok() && n.split_once('.').is_some_and(|(_, d)| d.len() == 1), "seconds {n:?} in {line:?}");
            format!("{}<s>{}", &line[..start], &line[end..])
        };
        log.split_inclusive('\n')
            .map(|line| {
                if line.starts_with("===== ") && !line.starts_with("===== end") {
                    let ts = &line[6..25];
                    assert!(jiff::civil::DateTime::strptime("%Y-%m-%d %H:%M:%S", ts).is_ok(), "start time {ts:?}");
                    format!("===== <ts>{}", &line[25..])
                } else if line.contains("] response · ") {
                    secs(line, "] response · ", " s · ")
                } else if line.contains("] FAILED after ") {
                    secs(line, "] FAILED after ", " s: ")
                } else {
                    line.to_owned()
                }
            })
            .collect()
    }

    mod converse {
        use super::*;

        #[test]
        fn follows_each_next_line_in_the_first_replys_session_and_sums_tokens() {
            let tmp = tempfile::tempdir().unwrap();
            let log = ReplayLog::new(tmp.path().join("logs").join("opencode.log"));
            let agent = Scripted::new(vec![
                ok("ses_1", "first answer\nNext: check the hash migration", 11, 1),
                ok("ses_2", "second answer\nNext: run the auth tests", 22, 2),
                ok("ses_3", "third answer\nNext: not taken", 33, 4),
            ]);

            let o = converse(&agent, &cand(tmp.path()), 2, &parse_skip_words(None), &log);

            assert_eq!(o, Outcome { session_id: Some("ses_1".into()), exchanges: 3, tokens_in: 66, tokens_out: 7, error: None });
            // PROMPT is 65 chars; the title keeps the first 60.
            let title = "ctx-replay · Why does the login form reject valid passwords after the upg".to_owned();
            assert_eq!(
                agent.calls(),
                [
                    (tmp.path().to_path_buf(), None, title.clone(), PROMPT.to_owned()),
                    (tmp.path().to_path_buf(), Some("ses_1".to_owned()), title.clone(), "check the hash migration".to_owned()),
                    (tmp.path().to_path_buf(), Some("ses_1".to_owned()), title, "run the auth tests".to_owned()),
                ]
            );
            assert_eq!(
                normalized(log.path()),
                format!(
                    "\n===== <ts> · d:/dev/helpdesk · replay of ses-src/u42 · up to 2 /next =====\n\
                     --- [1/3] prompt\n{PROMPT}\n\
                     --- [1/3] response · <s> s · in 11 · out 1 tokens · ses_1\nfirst answer\nNext: check the hash migration\n\
                     --- [2/3] /next\ncheck the hash migration\n\
                     --- [2/3] response · <s> s · in 22 · out 2 tokens · ses_2\nsecond answer\nNext: run the auth tests\n\
                     --- [3/3] /next\nrun the auth tests\n\
                     --- [3/3] response · <s> s · in 33 · out 4 tokens · ses_3\nthird answer\nNext: not taken\n\
                     ===== end · 3 answers · in 66 · out 7 tokens =====\n"
                )
            );
        }

        #[test]
        fn an_excluded_next_line_is_not_sent_and_stops_the_replay() {
            // `None`: the default skip words.
            let rows: [(&str, Option<&[&str]>, &str, &str); 5] = [
                ("a default skip word", None, "check the prod db", "has the skip word `db`"),
                ("a Polish default skip word", None, "sprawdź bazę na produkcji", "has the skip word `bazę`"),
                ("a mention of Claude", None, "update CLAUDE.md with the rule", "mentions Claude"),
                ("a mention of Claude without skip words", Some(&[]), "add it to .claude/settings.json", "mentions Claude"),
                ("a given skip word, not a default one", Some(&["staging"]), "deploy it to staging", "has the skip word `staging`"),
            ];
            for (name, skip, next, why) in rows {
                let tmp = tempfile::tempdir().unwrap();
                let log = ReplayLog::new(tmp.path().join("o.log"));
                let agent = Scripted::new(vec![
                    ok("ses_1", "first answer\nNext: run the auth tests", 11, 1),
                    ok("ses_1", &format!("second answer\nNext: {next}"), 22, 2),
                    ok("ses_1", "never", 1, 1),
                ]);
                let skip = skip.map_or_else(|| parse_skip_words(None), words);

                let o = converse(&agent, &cand(tmp.path()), 3, &skip, &log);

                assert_eq!(o, Outcome { session_id: Some("ses_1".into()), exchanges: 2, tokens_in: 33, tokens_out: 3, error: None }, "{name}");
                assert_eq!(agent.prompts(), [PROMPT, "run the auth tests"], "{name}");
                assert_eq!(
                    normalized(log.path()),
                    format!(
                        "\n===== <ts> · d:/dev/helpdesk · replay of ses-src/u42 · up to 3 /next =====\n\
                         --- [1/4] prompt\n{PROMPT}\n\
                         --- [1/4] response · <s> s · in 11 · out 1 tokens · ses_1\nfirst answer\nNext: run the auth tests\n\
                         --- [2/4] /next\nrun the auth tests\n\
                         --- [2/4] response · <s> s · in 22 · out 2 tokens · ses_1\nsecond answer\nNext: {next}\n\
                         --- the `Next:` line {why}; not sent, stopping\n\
                         ===== end · 2 answers · in 33 · out 3 tokens =====\n"
                    ),
                    "{name}"
                );
            }
        }

        #[test]
        fn an_excluded_next_line_in_the_first_answer_leaves_one_call() {
            let tmp = tempfile::tempdir().unwrap();
            let log = ReplayLog::new(tmp.path().join("o.log"));
            let agent = Scripted::new(vec![ok("ses_7", "an answer\nNext: ask Claude to review it", 4, 9), ok("ses_7", "never", 1, 1)]);

            let o = converse(&agent, &cand(tmp.path()), 2, &parse_skip_words(None), &log);

            assert_eq!(o, Outcome { session_id: Some("ses_7".into()), exchanges: 1, tokens_in: 4, tokens_out: 9, error: None });
            assert_eq!(agent.prompts(), [PROMPT]);
            assert_eq!(
                normalized(log.path()),
                format!(
                    "\n===== <ts> · d:/dev/helpdesk · replay of ses-src/u42 · up to 2 /next =====\n\
                     --- [1/3] prompt\n{PROMPT}\n\
                     --- [1/3] response · <s> s · in 4 · out 9 tokens · ses_7\nan answer\nNext: ask Claude to review it\n\
                     --- the `Next:` line mentions Claude; not sent, stopping\n\
                     ===== end · 1 answers · in 4 · out 9 tokens =====\n"
                )
            );
        }

        #[test]
        fn zero_follow_ups_is_one_call() {
            let tmp = tempfile::tempdir().unwrap();
            let agent = Scripted::new(vec![ok("ses_1", "answer\nNext: would go on", 5, 6), ok("ses_1", "never", 1, 1)]);
            let o = converse(&agent, &cand(tmp.path()), 0, &parse_skip_words(None), &ReplayLog::new(tmp.path().join("o.log")));
            assert_eq!(o, Outcome { session_id: Some("ses_1".into()), exchanges: 1, tokens_in: 5, tokens_out: 6, error: None });
            assert_eq!(agent.prompts(), [PROMPT]);
        }

        #[test]
        fn stops_when_an_answer_has_no_next_line() {
            let tmp = tempfile::tempdir().unwrap();
            let log = ReplayLog::new(tmp.path().join("o.log"));
            let agent = Scripted::new(vec![ok("ses_1", "a\nNext: go deeper", 3, 4), ok("ses_1", "all done", 5, 6), ok("ses_1", "never", 1, 1)]);

            let o = converse(&agent, &cand(tmp.path()), 3, &parse_skip_words(None), &log);

            assert_eq!(o, Outcome { session_id: Some("ses_1".into()), exchanges: 2, tokens_in: 8, tokens_out: 10, error: None });
            assert_eq!(agent.prompts(), [PROMPT, "go deeper"]);
            assert_eq!(
                normalized(log.path()),
                format!(
                    "\n===== <ts> · d:/dev/helpdesk · replay of ses-src/u42 · up to 3 /next =====\n\
                     --- [1/4] prompt\n{PROMPT}\n\
                     --- [1/4] response · <s> s · in 3 · out 4 tokens · ses_1\na\nNext: go deeper\n\
                     --- [2/4] /next\ngo deeper\n\
                     --- [2/4] response · <s> s · in 5 · out 6 tokens · ses_1\nall done\n\
                     --- no `Next:` line in the answer; stopping\n\
                     ===== end · 2 answers · in 8 · out 10 tokens =====\n"
                )
            );
        }

        #[test]
        fn an_error_stops_and_keeps_the_exchanges_so_far() {
            let tmp = tempfile::tempdir().unwrap();
            let log = ReplayLog::new(tmp.path().join("o.log"));
            let agent = Scripted::new(vec![ok("ses_1", "a\nNext: go deeper", 3, 4), Err("opencode killed after 7s".into()), ok("ses_1", "never", 1, 1)]);

            let o = converse(&agent, &cand(tmp.path()), 3, &parse_skip_words(None), &log);

            assert_eq!(o, Outcome { session_id: Some("ses_1".into()), exchanges: 1, tokens_in: 3, tokens_out: 4, error: Some("opencode killed after 7s".into()) });
            assert_eq!(agent.sessions(), [None, Some("ses_1".to_owned())]);
            assert_eq!(
                normalized(log.path()),
                format!(
                    "\n===== <ts> · d:/dev/helpdesk · replay of ses-src/u42 · up to 3 /next =====\n\
                     --- [1/4] prompt\n{PROMPT}\n\
                     --- [1/4] response · <s> s · in 3 · out 4 tokens · ses_1\na\nNext: go deeper\n\
                     --- [2/4] /next\ngo deeper\n\
                     --- [2/4] FAILED after <s> s: opencode killed after 7s\n\
                     ===== end · 1 answers · in 3 · out 4 tokens · failed =====\n"
                )
            );
        }

        #[test]
        fn an_error_on_the_first_call_has_no_session() {
            let tmp = tempfile::tempdir().unwrap();
            let agent = Scripted::new(vec![Err("cannot start opencode".into())]);
            let o = converse(&agent, &cand(tmp.path()), 2, &parse_skip_words(None), &ReplayLog::new(tmp.path().join("o.log")));
            assert_eq!(o, Outcome { session_id: None, exchanges: 0, tokens_in: 0, tokens_out: 0, error: Some("cannot start opencode".into()) });
            assert_eq!(agent.prompts(), [PROMPT]);
        }

        #[test]
        fn the_title_is_the_prefix_and_the_first_60_chars_of_the_first_line() {
            let tmp = tempfile::tempdir().unwrap();
            let first = format!("{}{}", "ż".repeat(59), "abc");
            let c = Candidate { prompt: format!("{first}\nsecond line"), ..cand(tmp.path()) };
            let agent = Scripted::new(vec![ok("ses_1", "no next", 1, 1)]);
            converse(&agent, &c, 1, &parse_skip_words(None), &ReplayLog::new(tmp.path().join("o.log")));
            assert_eq!(agent.calls()[0].2, format!("ctx-replay · {}a", "ż".repeat(59)));
        }
    }

    mod replay_log {
        use super::*;

        fn one_conversation(log: &ReplayLog, dir: &Path) {
            converse(&Scripted::new(vec![ok("ses_1", "no next", 1, 1)]), &cand(dir), 0, &parse_skip_words(None), log);
        }

        #[test]
        fn a_full_log_is_moved_to_dot_1_before_a_conversation() {
            let tmp = tempfile::tempdir().unwrap();
            let path = tmp.path().join("opencode.log");
            std::fs::write(&path, "0123456789").unwrap();
            std::fs::write(tmp.path().join("opencode.log.1"), "an older rotation").unwrap();

            one_conversation(&ReplayLog::with_limit(path.clone(), 10), tmp.path());

            assert_eq!(std::fs::read_to_string(tmp.path().join("opencode.log.1")).unwrap(), "0123456789");
            let now = std::fs::read_to_string(&path).unwrap();
            assert!(now.starts_with("\n===== ") && now.ends_with("===== end · 1 answers · in 1 · out 1 tokens =====\n"), "{now:?}");
        }

        #[test]
        fn a_log_under_the_limit_is_appended_to() {
            let tmp = tempfile::tempdir().unwrap();
            let path = tmp.path().join("opencode.log");
            std::fs::write(&path, "012345678").unwrap();

            let log = ReplayLog::with_limit(path.clone(), 10);
            one_conversation(&log, tmp.path());

            assert!(!tmp.path().join("opencode.log.1").exists());
            let now = std::fs::read_to_string(&path).unwrap();
            assert!(now.starts_with("012345678\n===== "), "{now:?}");
            assert_eq!(now.matches("===== end").count(), 1);
        }

        #[test]
        fn the_default_limit_is_10_mib() {
            for (len, rotated) in [(10 * 1024 * 1024, true), (10 * 1024 * 1024 - 1, false)] {
                let tmp = tempfile::tempdir().unwrap();
                let path = tmp.path().join("opencode.log");
                std::fs::File::create(&path).unwrap().set_len(len).unwrap();

                ReplayLog::new(path.clone()).rotate_if_full();

                assert_eq!(tmp.path().join("opencode.log.1").exists(), rotated, "{len} bytes");
                assert_eq!(path.exists(), !rotated, "{len} bytes");
            }
        }

        #[test]
        fn the_parent_directory_is_created_and_a_missing_log_is_not_rotated() {
            let tmp = tempfile::tempdir().unwrap();
            let path = tmp.path().join("nested").join("logs").join("opencode.log");
            let log = ReplayLog::with_limit(path.clone(), 10);
            log.rotate_if_full();
            assert!(!tmp.path().join("nested").exists());

            one_conversation(&log, tmp.path());

            assert!(std::fs::read_to_string(&path).unwrap().starts_with("\n===== "));
            assert!(!tmp.path().join("nested").join("logs").join("opencode.log.1").exists());
        }
    }

    #[test]
    fn the_guard_is_exclusive_survives_a_failed_acquire_and_is_released_on_drop() {
        let held = ReplayGuard::acquire().expect("free at the start");
        assert!(ReplayGuard::acquire().is_none(), "held");
        assert!(ReplayGuard::acquire().is_none(), "a failed acquire did not release it");
        drop(held);
        let again = ReplayGuard::acquire();
        assert!(again.is_some(), "released on drop");
    }

    mod check_agent {
        use super::*;

        #[test]
        fn passes_only_when_the_agent_file_is_there() {
            let tmp = tempfile::tempdir().unwrap();
            let c = ReplayConfig { agents_dir: tmp.path().to_path_buf(), ..cfg() };
            let file = tmp.path().join("ctx-replay.md");
            let missing = format!("replay agent ctx-replay not found at {}; run scripts/install.ps1", file.display());
            assert_eq!(check_agent(&c).unwrap_err().to_string(), missing, "no file");

            std::fs::create_dir(&file).unwrap();
            assert_eq!(check_agent(&c).unwrap_err().to_string(), missing, "a directory of that name");
            std::fs::remove_dir(&file).unwrap();

            std::fs::write(&file, "---\nmode: primary\n---\n").unwrap();
            check_agent(&c).unwrap();

            let other = ReplayConfig { agent: "other-agent".into(), ..c };
            assert_eq!(
                check_agent(&other).unwrap_err().to_string(),
                format!("replay agent other-agent not found at {}; run scripts/install.ps1", tmp.path().join("other-agent.md").display()),
                "another agent's file does not count"
            );
        }
    }

    mod record_start {
        use super::*;

        /// 2026-11-17T09:30:00Z
        const MS: i64 = 1_794_907_800_000;

        fn at(ms: i64, offset_hours: i8) -> jiff::Zoned {
            jiff::Timestamp::from_millisecond(ms).unwrap().to_zoned(jiff::tz::TimeZone::fixed(jiff::tz::offset(offset_hours)))
        }

        #[test]
        fn records_the_time_the_done_key_and_todays_count() {
            let rows: [(&str, Option<&str>, &str); 4] = [
                ("first ever", None, "2026-11-17 1"),
                ("later the same day", Some("2026-11-17 4"), "2026-11-17 5"),
                ("a new day restarts at 1", Some("2026-11-16 9"), "2026-11-17 1"),
                ("garbage restarts at 1", Some("nonsense"), "2026-11-17 1"),
            ];
            for (name, before, after) in rows {
                let state = State::in_memory().unwrap();
                if let Some(b) = before {
                    state.set_meta("replay/day", b).unwrap();
                }
                let c = Candidate { session_id: "ses-9".into(), turn: "u3".into(), ..cand(Path::new("Z:/x")) };

                record_start(&state, &c, &at(MS, 1)).unwrap();

                assert_eq!(state.get_meta("replay/day").unwrap().as_deref(), Some(after), "{name}");
                assert_eq!(state.get_meta("replay/last_ms").unwrap(), Some(MS.to_string()), "{name}");
                assert_eq!(state.get_meta("replay/done/ses-9/u3").unwrap(), Some(MS.to_string()), "{name}");
            }
        }

        #[test]
        fn the_day_is_the_local_date_of_now() {
            let state = State::in_memory().unwrap();
            state.set_meta("replay/day", "2026-11-17 2").unwrap();
            // 23:30 UTC is already 00:30 on the 18th at +01:00.
            let late = MS + 14 * 60 * 60 * 1000;
            record_start(&state, &cand(Path::new("Z:/x")), &at(late, 1)).unwrap();
            assert_eq!(state.get_meta("replay/day").unwrap().as_deref(), Some("2026-11-18 1"));
            assert_eq!(state.get_meta("replay/last_ms").unwrap(), Some(late.to_string()));
        }
    }
}
