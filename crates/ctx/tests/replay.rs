//! `ctx replay` against a real Qdrant: which indexed prompts qualify, the `ctx replay` command
//! (`run_once`) and the daemon's `maybe_start` with a fake `opencode.cmd`.

use std::collections::{BTreeSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use ctx::replay::{self, Agent, Candidate, ReplayConfig, ReplayGuard, Reply, Rng};
use embed::{Embedder, HashEmbedder};
use ingest::State;
use ingest::state::Record;
use serde_json::{Map, Value, json};
use store::{Doc, Store, point_id};

const DIM: usize = 64;

/// `maybe_start` takes the process-wide replay guard; the tests that call it take turns.
static GUARD: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn key(dir: &Path) -> String {
    sources::project::normalise(&dir.to_string_lossy())
}

/// Project directories under one temp root. `sedn26` and `gone` are never created.
struct Dirs {
    tmp: tempfile::TempDir,
    helpdesk: PathBuf,
    api: PathBuf,
    other: PathBuf,
    sedn26: PathBuf,
    unindexed: PathBuf,
    gone: PathBuf,
}

impl Dirs {
    fn new() -> Self {
        let tmp = tempfile::tempdir().unwrap();
        let helpdesk = tmp.path().join("helpdesk");
        let api = helpdesk.join("api");
        let other = tmp.path().join("other");
        let unindexed = tmp.path().join("x").join("helpdesk");
        for d in [&api, &other, &unindexed] {
            std::fs::create_dir_all(d).unwrap();
        }
        let sedn26 = tmp.path().join("sedn26");
        let gone = tmp.path().join("deleted-worktree");
        Self { tmp, helpdesk, api, other, sedn26, unindexed, gone }
    }

    /// A State whose indexed sessions cover every project except `unindexed`.
    fn state(&self) -> State {
        let state = State::in_memory().unwrap();
        for (i, dir) in [&self.helpdesk, &self.api, &self.other, &self.sedn26].into_iter().enumerate() {
            let project = key(dir);
            let session = format!("indexed-{i}");
            state.record(&Record { source: "claude", session_id: &session, fingerprint: "f", project: &project, chunks: 1, ..Default::default() }, 0).unwrap();
        }
        state
    }
}

fn cfg(projects: &[&str], dirs: &Dirs) -> ReplayConfig {
    ReplayConfig {
        enabled: true,
        projects: projects.iter().map(|p| (*p).to_owned()).collect(),
        every: Duration::from_secs(60 * 60),
        daily_max: 3,
        next: (2, 2),
        agent: "ctx-replay".into(),
        agents_dir: dirs.tmp.path().join("agents"),
        model: None,
        exe: PathBuf::from("Z:/no/such/opencode.exe"),
        timeout: Duration::from_secs(60),
        min_chars: 40,
        skip_words: replay::parse_skip_words(None),
        log: dirs.tmp.path().join("logs").join("opencode.log"),
    }
}

fn install_agent(c: &ReplayConfig) {
    std::fs::create_dir_all(&c.agents_dir).unwrap();
    std::fs::write(c.agents_dir.join(format!("{}.md", c.agent)), "---\nmode: primary\n---\n").unwrap();
}

fn prompt(name: &str) -> String {
    format!("Why does the {name} case fail after the parser upgrade?")
}

/// The first chunk of a main-session Claude turn, shaped like `ingest::chunk::payload`.
fn turn(session: &str, turn_id: Option<&str>, turn_index: i64, project: &Path, cwd: Option<&Path>, prompt: &str) -> Map<String, Value> {
    let Value::Object(mut m) = json!({
        "kind": "turn",
        "source": "claude",
        "project": key(project),
        "session_id": session,
        "subagent": false,
        "title": "Login bug",
        "branch": "main",
        "turn_index": turn_index,
        "chunk_index": 0,
        "chunk_count": 1,
        "ts": 1_790_330_400_000_i64,
        "text": format!("[Login bug · 2026-09-25]\nUser: {prompt}\n\nAssistant: an answer\n\nTools: read a.rs"),
    }) else {
        unreachable!("json! object")
    };
    if let Some(id) = turn_id {
        m.insert("turn_id".into(), json!(id));
    }
    if let Some(c) = cwd {
        m.insert("cwd".into(), json!(c.to_string_lossy()));
    }
    m
}

fn with(mut m: Map<String, Value>, field: &str, value: Value) -> Map<String, Value> {
    m.insert(field.into(), value);
    m
}

async fn sessions(q: &testkit::Qdrant) -> Store {
    let store = Store::connect(&q.url, None, "sessions", DIM).unwrap();
    store.ensure_collection(ingest::SESSION_FIELDS).await.unwrap();
    store
}

async fn seed(store: &Store, points: Vec<(&str, Map<String, Value>)>) {
    let mut emb = HashEmbedder::new(DIM);
    let docs = points.into_iter().map(|(name, payload)| Doc { id: point_id("claude", name), embedding: emb.embed_one(name).unwrap(), payload }).collect();
    store.upsert(docs).await.unwrap();
}

/// `opencode run --format json` output (a real sample), served by the fake `opencode.cmd`.
const REPLY: &str = "{\"type\":\"step_start\",\"timestamp\":1,\"sessionID\":\"ses_X\",\"part\":{\"type\":\"step-start\"}}\r\n{\"type\":\"text\",\"timestamp\":2,\"sessionID\":\"ses_X\",\"part\":{\"type\":\"text\",\"text\":\"ok\\nNext: say hi\"}}\r\n{\"type\":\"step_finish\",\"timestamp\":3,\"sessionID\":\"ses_X\",\"part\":{\"type\":\"step-finish\",\"reason\":\"stop\",\"tokens\":{\"input\":86,\"output\":8,\"reasoning\":0,\"cache\":{\"read\":0,\"write\":20352}}}}\r\n";

/// A fake `opencode.cmd` in `bin` that appends its arguments to `calls.txt` and prints `REPLY`.
fn fake_opencode(bin: &Path) -> PathBuf {
    std::fs::create_dir_all(bin).unwrap();
    let exe = bin.join("opencode.cmd");
    std::fs::write(bin.join("reply.jsonl"), REPLY).unwrap();
    std::fs::write(&exe, "@echo off\r\necho %*>> \"%~dp0calls.txt\"\r\ntype \"%~dp0reply.jsonl\"\r\n").unwrap();
    exe
}

/// Polls until no replay thread holds the guard.
async fn wait_for_the_replay_thread() {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if let Some(g) = ReplayGuard::acquire() {
            drop(g);
            return;
        }
        assert!(Instant::now() < deadline, "the replay thread never finished");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Never contacted: the paths under test return before they reach the store.
fn unreachable_store() -> Store {
    Store::connect("http://127.0.0.1:9", None, "sessions", DIM).unwrap()
}

fn candidate(session: &str, turn: &str, project: &Path, cwd: PathBuf, prompt: String) -> Candidate {
    Candidate { session_id: session.into(), turn: turn.into(), project: key(project), cwd, prompt }
}

fn sorted(mut c: Vec<Candidate>) -> Vec<Candidate> {
    c.sort_by(|a, b| a.session_id.cmp(&b.session_id));
    c
}

/// The log with its wall-clock parts replaced by `<ts>` and `<s>`, after checking their format.
fn normalized(log: &str) -> String {
    let secs = |line: &str, i: usize, after: &str, before: &str| -> String {
        let start = i + after.len();
        let end = start + line[start..].find(before).unwrap();
        assert!(line[start..end].parse::<f64>().is_ok(), "seconds in {line:?}");
        format!("{}<s>{}", &line[..start], &line[end..])
    };
    log.split_inclusive('\n')
        .map(|line| {
            if line.starts_with("===== ") && !line.starts_with("===== end") {
                let ts = &line[6..25];
                assert!(jiff::civil::DateTime::strptime("%Y-%m-%d %H:%M:%S", ts).is_ok(), "start time {ts:?}");
                format!("===== <ts>{}", &line[25..])
            } else if let Some(i) = line.find("] response · ") {
                secs(line, i, "] response · ", " s · ")
            } else if let Some(i) = line.find("] FAILED after ") {
                secs(line, i, "] FAILED after ", " s: ")
            } else {
                line.to_owned()
            }
        })
        .collect()
}

mod candidates {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    async fn only_usable_main_turn_prompts_of_named_indexed_projects_with_a_directory_qualify() {
        let q = testkit::qdrant().await;
        let store = sessions(&q).await;
        let d = Dirs::new();
        let state = d.state();
        state.set_meta(&replay::done_key("ses-f", "u7"), "1").unwrap();
        state.set_meta(&replay::done_key("ses-g", "3"), "1").unwrap();
        let long: String = "ż".repeat(12_001);
        let base = |s: &str, name: &str| turn(s, Some(name), 1, &d.helpdesk, Some(&d.helpdesk), &prompt(name));
        seed(
            &store,
            vec![
                ("cwd-preferred", turn("ses-a", Some("u1"), 1, &d.helpdesk, Some(&d.api), &prompt("cwd-preferred"))),
                ("cwd-gone", turn("ses-b", Some("u2"), 2, &d.helpdesk, Some(&d.gone), &prompt("cwd-gone"))),
                ("both-gone", turn("ses-c", Some("u3"), 3, &d.sedn26, Some(&d.gone), &prompt("both-gone"))),
                ("no-cwd", turn("ses-c2", Some("u3b"), 4, &d.api, None, &prompt("no-cwd"))),
                ("by-index", turn("ses-d", None, 5, &d.helpdesk, Some(&d.helpdesk), &prompt("by-index"))),
                ("empty-id", turn("ses-e", Some(""), 6, &d.helpdesk, Some(&d.helpdesk), &prompt("empty-id"))),
                ("done", base("ses-f", "u7")),
                ("done-by-index", turn("ses-g", None, 3, &d.helpdesk, Some(&d.helpdesk), &prompt("done-by-index"))),
                ("other-turn-of-done-session", base("ses-f", "u8")),
                ("summary", with(base("ses-i", "u10"), "kind", json!("summary"))),
                ("opencode", with(base("ses-j", "u11"), "source", json!("opencode"))),
                ("subagent", with(base("ses-k", "u12"), "subagent", json!(true))),
                ("second-chunk", with(base("ses-l", "u13"), "chunk_index", json!(1))),
                ("other-project", turn("ses-m", Some("u14"), 1, &d.other, Some(&d.other), &prompt("other-project"))),
                ("unindexed", turn("ses-n", Some("u15"), 1, &d.unindexed, Some(&d.unindexed), &prompt("unindexed"))),
                ("short", turn("ses-o", Some("u16"), 1, &d.helpdesk, Some(&d.helpdesk), "fix it now please")),
                ("slash", turn("ses-p", Some("u17"), 1, &d.helpdesk, Some(&d.helpdesk), &format!("/review {}", prompt("slash")))),
                ("no-user", with(base("ses-q", "u18"), "text", json!("[Login bug · 2026-09-25]\nAssistant: an answer without a prompt that is long"))),
                ("long", turn("ses-h", Some("u9"), 1, &d.helpdesk, Some(&d.helpdesk), &long)),
            ],
        )
        .await;
        let c = cfg(&["helpdesk", "sedn26"], &d);

        let got = sorted(replay::candidates(&store, &state, &c).await.unwrap());

        assert_eq!(
            got,
            [
                candidate("ses-a", "u1", &d.helpdesk, d.api.clone(), prompt("cwd-preferred")),
                candidate("ses-b", "u2", &d.helpdesk, PathBuf::from(key(&d.helpdesk)), prompt("cwd-gone")),
                candidate("ses-c2", "u3b", &d.api, PathBuf::from(key(&d.api)), prompt("no-cwd")),
                candidate("ses-d", "5", &d.helpdesk, d.helpdesk.clone(), prompt("by-index")),
                candidate("ses-e", "6", &d.helpdesk, d.helpdesk.clone(), prompt("empty-id")),
                candidate("ses-f", "u8", &d.helpdesk, d.helpdesk.clone(), prompt("u8")),
                candidate("ses-h", "u9", &d.helpdesk, d.helpdesk.clone(), "ż".repeat(12_000)),
            ]
        );

        // pick: always one of them, and over many seeds every one of them.
        let ids: BTreeSet<String> = BTreeSet::from_iter(got.iter().map(|c| c.session_id.clone()));
        let mut picked = BTreeSet::new();
        for seed in 0..200 {
            let p = replay::pick(&store, &state, &c, &mut Rng::seeded(seed)).await.unwrap().expect("candidates exist");
            assert!(got.contains(&p), "seed {seed}: {p:?}");
            picked.insert(p.session_id);
        }
        assert_eq!(picked, ids);

        let none = cfg(&["nomatch"], &d);
        assert_eq!(replay::candidates(&store, &state, &none).await.unwrap(), Vec::<Candidate>::new());
        assert_eq!(replay::pick(&store, &state, &none, &mut Rng::seeded(1)).await.unwrap(), None);
    }

    const CLAUDE_PROMPTS: [(&str, &str); 4] = [
        ("skip-claude-name", "Why does Claude keep failing on the parser upgrade?"),
        ("skip-claude-md", "Please update CLAUDE.md with the parser upgrade rules"),
        ("skip-claude-dir", "Why is .claude/settings.json ignored after the upgrade?"),
        ("skip-claude-code", "How do claude-code hooks run after the parser upgrade?"),
    ];
    const DB_PROMPTS: [(&str, &str); 5] = [
        ("skip-db-plain", "check on prod db whether the parser upgrade broke it"),
        ("skip-db-underscore", "Why is prod_db unreachable after the parser upgrade?"),
        ("skip-db-upper", "DB_HOST points nowhere after the parser upgrade, why?"),
        ("skip-db-polish", "Sprawdź bazę danych po aktualizacji parsera, proszę"),
        ("skip-db-sql", "Run the SQL migration again after the parser upgrade"),
    ];
    const KEPT_PROMPTS: [(&str, &str); 4] = [
        ("skip-kept-mongodb", "Why does the mongodb driver fail after the parser upgrade?"),
        ("skip-kept-dbg", "Why does the dbg build fail after the parser upgrade?"),
        ("skip-kept-sqlite", "Why is the sqlite file locked after the parser upgrade?"),
        ("skip-kept-plain", "Why does the login form fail after the parser upgrade?"),
    ];

    /// Each prompt as a turn of its own session; point and session are `<prefix><name>`, so tests
    /// sharing the collection do not overwrite each other's points.
    async fn seed_prompts(store: &Store, d: &Dirs, prefix: &str, prompts: &[(&str, &str)]) {
        let names: Vec<String> = prompts.iter().map(|(n, _)| format!("{prefix}{n}")).collect();
        seed(store, names.iter().zip(prompts).map(|(name, (_, p))| (name.as_str(), turn(name, Some("u1"), 1, &d.helpdesk, Some(&d.helpdesk), p))).collect()).await;
    }

    fn expected(d: &Dirs, prefix: &str, prompts: &[&[(&str, &str)]]) -> Vec<Candidate> {
        sorted(prompts.iter().flat_map(|ps| ps.iter()).map(|(n, p)| candidate(&format!("{prefix}{n}"), "u1", &d.helpdesk, d.helpdesk.clone(), (*p).to_owned())).collect())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn prompts_mentioning_claude_or_with_a_skip_word_never_qualify() {
        let q = testkit::qdrant().await;
        let store = sessions(&q).await;
        let d = Dirs::new();
        let state = d.state();
        let all: Vec<(&str, &str)> = [&CLAUDE_PROMPTS[..], &DB_PROMPTS[..], &KEPT_PROMPTS[..]].concat();
        seed_prompts(&store, &d, "", &all).await;

        let defaults = cfg(&["helpdesk"], &d);
        let got = sorted(replay::candidates(&store, &state, &defaults).await.unwrap());
        assert_eq!(got, expected(&d, "", &[&KEPT_PROMPTS]), "the default skip words");
        for seed in 0..50 {
            let p = replay::pick(&store, &state, &defaults, &mut Rng::seeded(seed)).await.unwrap().expect("candidates exist");
            assert!(got.contains(&p), "seed {seed}: {p:?}");
        }

        // No skip words: the database prompts qualify, the Claude ones still do not.
        let no_words = ReplayConfig { skip_words: Vec::new(), ..cfg(&["helpdesk"], &d) };
        assert_eq!(sorted(replay::candidates(&store, &state, &no_words).await.unwrap()), expected(&d, "", &[&DB_PROMPTS, &KEPT_PROMPTS]), "no skip words");

        // Given words replace the defaults.
        let custom = ReplayConfig { skip_words: vec!["mongodb".into()], ..cfg(&["helpdesk"], &d) };
        assert_eq!(sorted(replay::candidates(&store, &state, &custom).await.unwrap()), expected(&d, "", &[&DB_PROMPTS, &KEPT_PROMPTS[1..]]), "given skip words");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_claude_directory_is_never_the_working_directory_and_a_claude_project_never_qualifies() {
        let q = testkit::qdrant().await;
        let store = sessions(&q).await;
        let d = Dirs::new();
        let state = d.state();
        let skills = d.helpdesk.join(".Claude").join("skills");
        std::fs::create_dir_all(&skills).unwrap();
        let claude_project = d.helpdesk.join(".Claude");
        std::fs::create_dir_all(&claude_project).unwrap();
        let claude_key = key(&claude_project);
        state.record(&Record { source: "claude", session_id: "indexed-dot-claude", fingerprint: "f", project: &claude_key, chunks: 1, ..Default::default() }, 0).unwrap();
        seed(
            &store,
            vec![
                // The cwd exists but is under `.claude`: the project directory is used instead.
                ("dot-dir-cwd", turn("ses-dd1", Some("u1"), 1, &d.helpdesk, Some(&skills), &prompt("dot-dir-cwd"))),
                // Same cwd, and the project directory does not exist: nothing to run in.
                ("dot-dir-no-fallback", turn("ses-dd2", Some("u2"), 2, &d.sedn26, Some(&skills), &prompt("dot-dir-no-fallback"))),
                // A usable cwd, but the project itself is a `.Claude` directory.
                ("dot-dir-project", turn("ses-dd3", Some("u3"), 3, &claude_project, Some(&d.api), &prompt("dot-dir-project"))),
            ],
        )
        .await;

        let got = sorted(replay::candidates(&store, &state, &cfg(&["helpdesk", "sedn26"], &d)).await.unwrap());

        assert_eq!(got, [candidate("ses-dd1", "u1", &d.helpdesk, PathBuf::from(key(&d.helpdesk)), prompt("dot-dir-cwd"))]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn with_only_excluded_prompts_neither_a_dry_nor_a_real_run_sends_or_logs_anything() {
        let q = testkit::qdrant().await;
        let store = sessions(&q).await;
        let d = Dirs::new();
        let state = d.state();
        seed_prompts(&store, &d, "only-", &[&CLAUDE_PROMPTS[..], &DB_PROMPTS[..]].concat()).await;
        let c = cfg(&["helpdesk"], &d);
        let agent = Scripted::new(vec![ok("ses_never", "never", 1, 1)]);

        assert_eq!(replay::run_once(&c, &store, &state, &agent, true).await.unwrap(), "nothing to replay in helpdesk", "dry run");
        install_agent(&c);
        assert_eq!(replay::run_once(&c, &store, &state, &agent, false).await.unwrap(), "nothing to replay in helpdesk", "real run");

        assert_eq!(agent.calls(), []);
        assert!(!c.log.exists(), "no log written");
        assert_eq!(records(&state, &[]), none_recorded(&[]));
    }
}

/// An `Agent` that answers from a script and records every call.
struct Scripted {
    replies: Mutex<VecDeque<Result<Reply, String>>>,
    calls: Mutex<Vec<(PathBuf, Option<String>, String)>>,
}

impl Scripted {
    fn new(replies: Vec<Result<Reply, String>>) -> Self {
        Self { replies: Mutex::new(replies.into()), calls: Mutex::default() }
    }
    fn calls(&self) -> Vec<(PathBuf, Option<String>, String)> {
        self.calls.lock().unwrap().clone()
    }
}

impl Agent for Scripted {
    fn send(&self, cwd: &Path, session: Option<&str>, _title: &str, prompt: &str) -> Result<Reply, String> {
        self.calls.lock().unwrap().push((cwd.to_path_buf(), session.map(str::to_owned), prompt.to_owned()));
        self.replies.lock().unwrap().pop_front().unwrap_or_else(|| Err("no scripted reply left".into()))
    }
}

fn ok(session: &str, text: &str, tokens_in: u64, tokens_out: u64) -> Result<Reply, String> {
    Ok(Reply { session_id: session.into(), text: text.into(), tokens_in, tokens_out })
}

/// Every replay record in the State, as (key, value).
fn records(state: &State, done: &[(&str, &str)]) -> Vec<(String, Option<String>)> {
    let mut keys: Vec<String> = vec!["replay/day".into(), "replay/last_ms".into()];
    keys.extend(done.iter().map(|(s, t)| replay::done_key(s, t)));
    keys.into_iter().map(|k| (k.clone(), state.get_meta(&k).unwrap())).collect()
}

fn none_recorded(done: &[(&str, &str)]) -> Vec<(String, Option<String>)> {
    let mut keys: Vec<String> = vec!["replay/day".into(), "replay/last_ms".into()];
    keys.extend(done.iter().map(|(s, t)| replay::done_key(s, t)));
    keys.into_iter().map(|k| (k, None)).collect()
}

fn now_ms() -> i64 {
    jiff::Timestamp::now().as_millisecond()
}

mod run_once {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    async fn refuses_without_projects_and_says_when_there_is_nothing_to_replay() {
        let d = Dirs::new();
        let state = d.state();
        let agent = Scripted::new(vec![]);
        let store = unreachable_store();

        let err = replay::run_once(&cfg(&[], &d), &store, &state, &agent, false).await.unwrap_err();
        assert_eq!(err.to_string(), "no projects to replay; set CTX_REPLAY_PROJECTS (e.g. helpdesk,sedn26)");

        let msg = replay::run_once(&cfg(&["nomatch", "neither"], &d), &store, &state, &agent, false).await.unwrap();
        assert_eq!(msg, "nothing to replay in nomatch, neither");
        assert_eq!(agent.calls(), []);
        assert_eq!(records(&state, &[]), none_recorded(&[]));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn dry_run_then_missing_agent_then_a_failing_and_a_successful_replay() {
        let q = testkit::qdrant().await;
        let store = sessions(&q).await;
        let d = Dirs::new();
        let state = d.state();
        let c = cfg(&["helpdesk", "sedn26"], &d);
        let first = prompt("first");
        seed(&store, vec![("first", turn("ses-1", Some("u1"), 1, &d.helpdesk, Some(&d.api), &first))]).await;
        let done = [("ses-1", "u1"), ("ses-2", "u2")];

        // Dry run: reports, sends and records nothing.
        let agent = Scripted::new(vec![]);
        let msg = replay::run_once(&c, &store, &state, &agent, true).await.unwrap();
        assert_eq!(msg, format!("1 prompts left in helpdesk, sedn26\nwould replay in {} ({}) with up to 2 /next:\n{first}", key(&d.helpdesk), d.api.display()));
        assert_eq!(agent.calls(), []);
        assert_eq!(records(&state, &done), none_recorded(&done));
        assert!(!c.log.exists(), "no log written");

        // No agent file: refused before anything is recorded or sent.
        let err = replay::run_once(&c, &store, &state, &agent, false).await.unwrap_err();
        assert_eq!(err.to_string(), format!("replay agent ctx-replay not found at {}; run scripts/install.ps1", c.agents_dir.join("ctx-replay.md").display()));
        assert_eq!(agent.calls(), []);
        assert_eq!(records(&state, &done), none_recorded(&done));

        // The agent fails on the first /next: an error with the summary, the start recorded.
        install_agent(&c);
        let today = jiff::Zoned::now().strftime("%Y-%m-%d").to_string();
        let failing = Scripted::new(vec![ok("ses_r1", "answer one\nNext: dig into the parser", 12, 3), Err("opencode exited with exit code: 1: boom".into())]);
        let before = now_ms();
        let err = replay::run_once(&c, &store, &state, &failing, false).await.unwrap_err();
        let after = now_ms();
        assert_eq!(
            err.to_string(),
            format!("{}: 1 answers, in 12 · out 3 tokens; log: {}\nfailed: opencode exited with exit code: 1: boom", key(&d.helpdesk), c.log.display())
        );
        assert_eq!(failing.calls(), [(d.api.clone(), None, first.clone()), (d.api.clone(), Some("ses_r1".to_owned()), "dig into the parser".to_owned())]);
        let r = records(&state, &done);
        assert_eq!(r[0], ("replay/day".to_owned(), Some(format!("{today} 1"))));
        let last: i64 = r[1].1.as_deref().unwrap().parse().unwrap();
        assert!((before..=after).contains(&last), "{before} <= {last} <= {after}");
        assert_eq!(r[2], (replay::done_key("ses-1", "u1"), Some(last.to_string())));
        assert_eq!(r[3], (replay::done_key("ses-2", "u2"), None));
        let log = std::fs::read_to_string(&c.log).unwrap();
        assert!(log.contains("--- [2/3] FAILED after ") && log.ends_with("===== end · 1 answers · in 12 · out 3 tokens · failed =====\n"), "{log}");

        // The failed prompt is done; with nothing else left there is nothing to replay.
        let idle = Scripted::new(vec![]);
        assert_eq!(replay::run_once(&c, &store, &state, &idle, false).await.unwrap(), "nothing to replay in helpdesk, sedn26");

        // A new prompt: three answers (the prompt and 2 /next), counted as today's second.
        let second = prompt("second");
        seed(&store, vec![("second", turn("ses-2", Some("u2"), 2, &d.api, None, &second))]).await;
        let good = Scripted::new(vec![
            ok("ses_r2", "a\nNext: step two", 10, 1),
            ok("ses_r2", "b\nNext: step three", 20, 2),
            ok("ses_r2", "c\nNext: step four", 30, 4),
            ok("ses_r2", "never", 1, 1),
        ]);
        let msg = replay::run_once(&c, &store, &state, &good, false).await.unwrap();
        assert_eq!(msg, format!("{}: 3 answers, in 60 · out 7 tokens; log: {}", key(&d.api), c.log.display()));
        let cwd = PathBuf::from(key(&d.api));
        assert_eq!(
            good.calls(),
            [
                (cwd.clone(), None, second.clone()),
                (cwd.clone(), Some("ses_r2".to_owned()), "step two".to_owned()),
                (cwd, Some("ses_r2".to_owned()), "step three".to_owned()),
            ]
        );
        assert_eq!(state.get_meta("replay/day").unwrap(), Some(format!("{today} 2")));
        let second_start = state.get_meta("replay/last_ms").unwrap();
        assert_ne!(second_start, Some(last.to_string()), "a new start time");
        assert_eq!(state.get_meta(&replay::done_key("ses-2", "u2")).unwrap(), second_start);
        assert_eq!(std::fs::read_to_string(&c.log).unwrap().matches("===== end").count(), 2, "both replays logged");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn the_configured_skip_words_stop_at_a_next_line() {
        let q = testkit::qdrant().await;
        let store = sessions(&q).await;
        let d = Dirs::new();
        let state = d.state();
        // "hi" is no default skip word, so only the configured list can stop at `Next: say hi`.
        let c = ReplayConfig { skip_words: vec!["hi".into()], ..cfg(&["helpdesk"], &d) };
        install_agent(&c);
        let p = prompt("pass-through");
        seed(&store, vec![("pass-through-run-once", turn("ses-pt1", Some("u1"), 1, &d.helpdesk, Some(&d.api), &p))]).await;
        let agent = Scripted::new(vec![ok("ses_pt", "answer\nNext: say hi", 5, 6), ok("ses_pt", "never", 1, 1)]);

        let msg = replay::run_once(&c, &store, &state, &agent, false).await.unwrap();

        assert_eq!(msg, format!("{}: 1 answers, in 5 · out 6 tokens; log: {}", key(&d.helpdesk), c.log.display()));
        assert_eq!(agent.calls(), [(d.api.clone(), None, p.clone())]);
        assert_eq!(
            normalized(&std::fs::read_to_string(&c.log).unwrap()),
            format!(
                "\n===== <ts> · {} · replay of ses-pt1/u1 · up to 2 /next =====\n\
                 --- [1/3] prompt\n{p}\n\
                 --- [1/3] response · <s> s · in 5 · out 6 tokens · ses_pt\nanswer\nNext: say hi\n\
                 --- the `Next:` line has the skip word `hi`; not sent, stopping\n\
                 ===== end · 1 answers · in 5 · out 6 tokens =====\n",
                key(&d.helpdesk)
            )
        );
    }
}

mod opencode_error {
    use super::*;
    use ctx::replay::OpencodeCli;

    /// What OpenCode 2.2.7 prints for an unknown `--model`: an `error` event on stdout, nothing
    /// on stderr, exit code 1.
    const MODEL_NOT_FOUND: &str = "{\"type\":\"error\",\"timestamp\":1763460000000,\"sessionID\":\"ses_F\",\"error\":{\"type\":\"provider.invalid-request\",\"message\":\"Model not found\",\"status\":400,\"providerID\":\"opi\",\"modelID\":\"no-such-model\"}}\r\n";

    /// A fake `opencode.cmd` in `bin` that appends its arguments to `calls.txt`, prints
    /// `MODEL_NOT_FOUND` and exits 1.
    fn failing_opencode(bin: &Path) -> PathBuf {
        std::fs::create_dir_all(bin).unwrap();
        let exe = bin.join("opencode.cmd");
        std::fs::write(bin.join("error.jsonl"), MODEL_NOT_FOUND).unwrap();
        std::fs::write(&exe, "@echo off\r\necho %*>> \"%~dp0calls.txt\"\r\ntype \"%~dp0error.jsonl\"\r\nexit /b 1\r\n").unwrap();
        exe
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn an_error_event_on_stdout_with_exit_1_ends_the_replay_with_its_message() {
        let q = testkit::qdrant().await;
        let store = sessions(&q).await;
        let d = Dirs::new();
        let state = d.state();
        let bin = d.tmp.path().join("bin");
        let c = ReplayConfig { model: Some("opi/no-such-model".into()), exe: failing_opencode(&bin), ..cfg(&["helpdesk"], &d) };
        install_agent(&c);
        let p = prompt("model-not-found");
        seed(&store, vec![("model-not-found", turn("ses-mnf", Some("u6"), 6, &d.helpdesk, Some(&d.api), &p))]).await;

        let err = replay::run_once(&c, &store, &state, &OpencodeCli::from_config(&c), false).await.unwrap_err();

        let reason = "opencode exited with exit code: 1: Model not found";
        assert_eq!(err.to_string(), format!("{}: 0 answers, in 0 · out 0 tokens; log: {}\nfailed: {reason}", key(&d.helpdesk), c.log.display()));
        assert_eq!(
            normalized(&std::fs::read_to_string(&c.log).unwrap()),
            format!(
                "\n===== <ts> · {} · replay of ses-mnf/u6 · up to 2 /next =====\n\
                 --- [1/3] prompt\n{p}\n\
                 --- [1/3] FAILED after <s> s: {reason}\n\
                 ===== end · 0 answers · in 0 · out 0 tokens · failed =====\n",
                key(&d.helpdesk)
            )
        );
        let calls = std::fs::read_to_string(bin.join("calls.txt")).unwrap();
        assert_eq!(calls.lines().count(), 1, "no follow-up after the failure: {calls}");
        assert!(calls.starts_with("run --format json --agent ctx-replay --model opi/no-such-model --archived --title "), "{calls}");
    }
}

mod maybe_start {
    use super::*;

    /// 2026-11-17T09:30:00Z, at +01:00.
    const MS: i64 = 1_794_907_800_000;
    const HOUR_MS: i64 = 60 * 60 * 1000;

    fn now() -> jiff::Zoned {
        jiff::Timestamp::from_millisecond(MS).unwrap().to_zoned(jiff::tz::TimeZone::fixed(jiff::tz::offset(1)))
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn off_returns_none_and_records_nothing() {
        let _turn = GUARD.lock().await;
        let d = Dirs::new();
        let state = d.state();
        let c = ReplayConfig { enabled: false, ..cfg(&["helpdesk"], &d) };
        install_agent(&c);
        assert_eq!(replay::maybe_start(&c, &unreachable_store(), &state, &now()).await.unwrap(), None);
        assert_eq!(records(&state, &[]), none_recorded(&[]));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_held_guard_returns_none_and_stays_held() {
        let _turn = GUARD.lock().await;
        let d = Dirs::new();
        let state = d.state();
        let c = cfg(&["helpdesk"], &d);
        let held = ReplayGuard::acquire().expect("free");

        // Due, but the agent file is missing: an Err would show the guard was not checked first.
        assert_eq!(replay::maybe_start(&c, &unreachable_store(), &state, &now()).await.unwrap(), None);

        assert!(ReplayGuard::acquire().is_none(), "still held");
        assert_eq!(records(&state, &[]), none_recorded(&[]));
        drop(held);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn not_due_returns_none_before_the_agent_check_and_records_nothing() {
        let _turn = GUARD.lock().await;
        let rows: [(&str, &[&str], Option<String>, Option<&str>); 3] = [
            ("every not passed", &["helpdesk"], Some((MS - HOUR_MS + 1).to_string()), None),
            ("daily cap reached", &["helpdesk"], None, Some("2026-11-17 3")),
            ("no projects", &[], None, None),
        ];
        for (name, projects, last, day) in rows {
            let d = Dirs::new();
            let state = d.state();
            let c = cfg(projects, &d);
            if let Some(l) = &last {
                state.set_meta("replay/last_ms", l).unwrap();
            }
            if let Some(v) = day {
                state.set_meta("replay/day", v).unwrap();
            }

            assert_eq!(replay::maybe_start(&c, &unreachable_store(), &state, &now()).await.unwrap(), None, "{name}");

            assert_eq!(state.get_meta("replay/last_ms").unwrap(), last, "{name}");
            assert_eq!(state.get_meta("replay/day").unwrap().as_deref(), day, "{name}");
            assert!(ReplayGuard::acquire().is_some(), "{name}: guard released");
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn due_without_the_agent_file_is_an_err_and_records_nothing() {
        let _turn = GUARD.lock().await;
        let d = Dirs::new();
        let state = d.state();
        let c = cfg(&["helpdesk"], &d);
        state.set_meta("replay/last_ms", &(MS - HOUR_MS).to_string()).unwrap();

        let err = replay::maybe_start(&c, &unreachable_store(), &state, &now()).await.unwrap_err();

        assert_eq!(err.to_string(), format!("replay agent ctx-replay not found at {}; run scripts/install.ps1", c.agents_dir.join("ctx-replay.md").display()));
        assert_eq!(state.get_meta("replay/last_ms").unwrap(), Some((MS - HOUR_MS).to_string()));
        assert_eq!(state.get_meta("replay/day").unwrap(), None);
        assert!(ReplayGuard::acquire().is_some(), "guard released");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn an_empty_pick_records_the_scan_time_so_the_next_scan_waits_every() {
        let _turn = GUARD.lock().await;
        let d = Dirs::new();
        let state = d.state();
        let none_match = cfg(&["nomatch"], &d);
        install_agent(&none_match);

        assert_eq!(replay::maybe_start(&none_match, &unreachable_store(), &state, &now()).await.unwrap(), None);

        assert_eq!(state.get_meta("replay/last_ms").unwrap(), Some(MS.to_string()));
        assert_eq!(state.get_meta("replay/day").unwrap(), None, "not counted as a replay");
        assert!(ReplayGuard::acquire().is_some(), "guard released");

        // "helpdesk" is indexed, so a scan would reach the unreachable store and fail.
        let matching = cfg(&["helpdesk"], &d);
        let at = |ms: i64| jiff::Timestamp::from_millisecond(ms).unwrap().to_zoned(jiff::tz::TimeZone::fixed(jiff::tz::offset(1)));
        assert_eq!(replay::maybe_start(&matching, &unreachable_store(), &state, &at(MS + HOUR_MS - 1)).await.unwrap(), None, "within every: no scan");
        assert_eq!(state.get_meta("replay/last_ms").unwrap(), Some(MS.to_string()));
        assert!(replay::maybe_start(&matching, &unreachable_store(), &state, &at(MS + HOUR_MS)).await.is_err(), "after every it scans (and hits the unreachable store)");
        assert!(ReplayGuard::acquire().is_some(), "guard released");
    }


    #[tokio::test(flavor = "multi_thread")]
    async fn due_starts_the_real_cli_on_a_thread_and_records_the_start() {
        let _turn = GUARD.lock().await;
        let q = testkit::qdrant().await;
        let store = sessions(&q).await;
        let d = Dirs::new();
        let state = d.state();
        state.set_meta("replay/day", "2026-11-17 2").unwrap();
        let bin = d.tmp.path().join("bin");
        let c = ReplayConfig { next: (1, 1), exe: fake_opencode(&bin), ..cfg(&["helpdesk"], &d) };
        install_agent(&c);
        let p = prompt("daemon");
        seed(&store, vec![("daemon", turn("ses-d1", Some("u5"), 5, &d.helpdesk, Some(&d.api), &p))]).await;

        let line = replay::maybe_start(&c, &store, &state, &now()).await.unwrap();

        assert_eq!(line, Some(format!("replay: {} with up to 1 /next", key(&d.helpdesk))));
        assert_eq!(state.get_meta("replay/last_ms").unwrap(), Some(MS.to_string()));
        assert_eq!(state.get_meta("replay/day").unwrap().as_deref(), Some("2026-11-17 3"));
        assert_eq!(state.get_meta(&replay::done_key("ses-d1", "u5")).unwrap(), Some(MS.to_string()));

        // The conversation runs on its own thread, holding the guard until it ends.
        wait_for_the_replay_thread().await;
        let log = std::fs::read_to_string(&c.log).unwrap();
        assert_eq!(
            normalized(&log),
            format!(
                "\n===== <ts> · {} · replay of ses-d1/u5 · up to 1 /next =====\n\
                 --- [1/2] prompt\n{p}\n\
                 --- [1/2] response · <s> s · in 86 · out 8 tokens · ses_X\nok\nNext: say hi\n\
                 --- [2/2] /next\nsay hi\n\
                 --- [2/2] response · <s> s · in 86 · out 8 tokens · ses_X\nok\nNext: say hi\n\
                 ===== end · 2 answers · in 172 · out 16 tokens =====\n",
                key(&d.helpdesk)
            )
        );
        let calls = std::fs::read_to_string(bin.join("calls.txt")).unwrap();
        let calls: Vec<&str> = calls.lines().collect();
        assert_eq!(calls.len(), 2, "{calls:?}");
        // The title carries a non-ASCII `·`, which cmd's code page may re-encode; checked around it.
        assert!(calls[0].starts_with("run --format json --agent ctx-replay --archived --title \"ctx-replay "), "{}", calls[0]);
        assert!(calls[0].ends_with(&format!(" \"{p}\"")), "{}", calls[0]);
        assert_eq!(calls[1], "run --format json --agent ctx-replay --session ses_X \"say hi\"");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn the_daemon_thread_stops_at_a_next_line_with_a_configured_skip_word() {
        let _turn = GUARD.lock().await;
        let q = testkit::qdrant().await;
        let store = sessions(&q).await;
        let d = Dirs::new();
        let state = d.state();
        let bin = d.tmp.path().join("bin");
        // REPLY ends with `Next: say hi`; "hi" is no default skip word.
        let c = ReplayConfig { next: (1, 1), skip_words: vec!["hi".into()], exe: fake_opencode(&bin), ..cfg(&["helpdesk"], &d) };
        install_agent(&c);
        let p = prompt("daemon-pass-through");
        seed(&store, vec![("pass-through-daemon", turn("ses-pt2", Some("u2"), 2, &d.helpdesk, Some(&d.api), &p))]).await;

        let line = replay::maybe_start(&c, &store, &state, &now()).await.unwrap();

        assert_eq!(line, Some(format!("replay: {} with up to 1 /next", key(&d.helpdesk))));
        wait_for_the_replay_thread().await;
        assert_eq!(
            normalized(&std::fs::read_to_string(&c.log).unwrap()),
            format!(
                "\n===== <ts> · {} · replay of ses-pt2/u2 · up to 1 /next =====\n\
                 --- [1/2] prompt\n{p}\n\
                 --- [1/2] response · <s> s · in 86 · out 8 tokens · ses_X\nok\nNext: say hi\n\
                 --- the `Next:` line has the skip word `hi`; not sent, stopping\n\
                 ===== end · 1 answers · in 86 · out 8 tokens =====\n",
                key(&d.helpdesk)
            )
        );
        let calls = std::fs::read_to_string(bin.join("calls.txt")).unwrap();
        assert_eq!(calls.lines().count(), 1, "{calls}");
        assert!(calls.trim_end().ends_with(&format!(" \"{p}\"")), "{calls}");
    }
}

mod daemon {
    use super::*;
    use std::sync::Arc;

    use ctx::config::Config;
    use ctx::embed_server::SharedEmbedder;
    use ctx::queue::{Runner, Target};

    fn runner(cfg: Config, store: Arc<Store>) -> Runner {
        ctx::daemon_runner(cfg, store, Arc::new(ingest::WordCounter), SharedEmbedder::new(Box::new(HashEmbedder::new(DIM))), "hash".into(), None, None, None, None, None)
    }

    /// One Claude session in `claude/p0` whose single prompt was sent in `cwd`.
    fn claude_session(claude: &Path, cwd: &Path, prompt: &str) {
        let dir = claude.join("p0");
        std::fs::create_dir_all(&dir).unwrap();
        let lines = [
            json!({"type": "user", "uuid": "u0", "timestamp": "2026-11-16T10:00:00Z", "cwd": cwd.to_string_lossy(), "origin": {"kind": "human"}, "message": {"content": prompt}}),
            json!({"type": "assistant", "timestamp": "2026-11-16T10:00:05Z", "message": {"content": [{"type": "text", "text": "an answer"}]}}),
        ];
        std::fs::write(dir.join("ses-daemon.jsonl"), lines.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("\n")).unwrap();
    }

    fn replay_records(state_path: &Path) -> (Option<String>, Option<String>) {
        let state = State::open(state_path).unwrap();
        (state.get_meta("replay/last_ms").unwrap(), state.get_meta("replay/day").unwrap())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_sessions_run_starts_a_replay_and_teams_only_or_switched_off_runs_do_not() {
        let _turn = GUARD.lock().await;
        let q = testkit::qdrant().await;
        let tmp = tempfile::tempdir().unwrap();
        let project = tmp.path().join("helpdesk");
        std::fs::create_dir_all(&project).unwrap();
        let claude = tmp.path().join("claude");
        let p = prompt("daemon wiring");
        claude_session(&claude, &project, &p);
        let agents = tmp.path().join("agents");
        std::fs::create_dir_all(&agents).unwrap();
        std::fs::write(agents.join("ctx-replay.md"), "---\nmode: primary\n---\n").unwrap();
        let log = tmp.path().join("logs").join("opencode.log");
        let replay_cfg = |enabled: bool| ReplayConfig {
            enabled,
            projects: vec!["helpdesk".into()],
            every: Duration::from_secs(60 * 60),
            daily_max: 3,
            next: (0, 0),
            agent: "ctx-replay".into(),
            agents_dir: agents.clone(),
            model: None,
            exe: fake_opencode(&tmp.path().join("bin")),
            timeout: Duration::from_secs(60),
            min_chars: 40,
            skip_words: replay::parse_skip_words(None),
            log: log.clone(),
        };
        let state_path = tmp.path().join("state.db");
        let config = |enabled: bool| Config {
            claude_dir: claude.clone(),
            opencode_db: tmp.path().join("none.db"),
            state_path: state_path.clone(),
            redact_file: tmp.path().join("redact.txt"),
            teams_db: tmp.path().join("no-teams.db"),
            replay: replay_cfg(enabled),
            ..Config::from_env(|_| None)
        };
        let store = Arc::new(Store::connect(&q.url, None, "sessions", DIM).unwrap().with_lexical());

        // Off: the sessions run indexes the prompt but starts nothing.
        let off = runner(config(false), store.clone())(Target::Claude).await.unwrap();
        assert!(!off.contains("replay:"), "{off}");
        assert_eq!(replay_records(&state_path), (None, None), "off");

        // On, Teams only (no Teams store): an empty report and nothing recorded.
        let on = runner(config(true), store.clone());
        assert_eq!(on(Target::Teams).await.unwrap(), "");
        assert_eq!(replay_records(&state_path), (None, None), "teams only");

        // On, sessions: the replay line closes the report and the start is recorded.
        let today = jiff::Zoned::now().strftime("%Y-%m-%d").to_string();
        let before = now_ms();
        let report = on(Target::Claude).await.unwrap();
        let after = now_ms();
        // The index part before it carries timings.
        assert!(report.ends_with(&format!(" · replay: {} with up to 0 /next", key(&project))), "{report}");
        let (last, day) = replay_records(&state_path);
        let last: i64 = last.expect("replay/last_ms recorded").parse().unwrap();
        assert!((before..=after).contains(&last), "{before} <= {last} <= {after}");
        assert_eq!(day, Some(format!("{today} 1")));

        wait_for_the_replay_thread().await;
        let calls = std::fs::read_to_string(tmp.path().join("bin").join("calls.txt")).unwrap();
        assert_eq!(calls.lines().count(), 1, "{calls}");
        assert!(calls.trim_end().ends_with(&format!(" \"{p}\"")), "{calls}");
        assert!(std::fs::read_to_string(&log).unwrap().ends_with("===== end · 1 answers · in 86 · out 8 tokens =====\n"));
    }
}
