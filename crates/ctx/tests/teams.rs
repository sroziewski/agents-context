//! Teams end to end: fixture teams.db + people.json → daemon Teams run → real Qdrant →
//! MCP search_teams / get_teams_thread over a real MCP client.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ctx::config::Config;
use ctx::embed_server::SharedEmbedder;
use ctx::mcp::ContextServer;
use ctx::queue::{IndexQueue, Target};
use ctx::search::Context;
use ctx::teams::TeamsSearch;
use embed::{Embedder, HashEmbedder};
use ingest::WordCounter;
use rmcp::ServiceExt;
use rmcp::model::{CallToolRequestParams, CallToolResult};
use rusqlite::{Connection, params};
use serde_json::{Value, json};
use store::Store;

const DIM: usize = 64;
const HELPDESK: &str = r"Z:\work\helpdesk";

fn fixture_db(path: &Path) {
    let c = Connection::open(path).unwrap();
    c.execute_batch(
        "create table channels (id text primary key, team text not null, name text not null, last_synced_at integer);
         create table messages (rowid integer primary key, id text not null unique, channel_id text not null,
             thread_id text not null, author text not null, ts integer not null, text text not null, indexed_at integer not null);
         create virtual table messages_fts using fts5(text, author, content='messages', content_rowid='rowid',
             tokenize='unicode61 remove_diacritics 2');
         create trigger messages_ai after insert on messages begin
             insert into messages_fts(rowid, text, author) values (new.rowid, new.text, new.author); end;
         create table meetings (id text primary key, chat text not null, title text not null, occurrence text not null,
             started_at integer not null, ended_at integer not null, summary text, summary_status text not null,
             transcript_status text not null, indexed_at integer not null);
         create table transcript_lines (rowid integer primary key, meeting_id text not null references meetings(id),
             idx integer not null, speaker text, offset_s integer, ts integer not null, text text not null,
             indexed_at integer not null, unique(meeting_id, idx));
         create virtual table transcript_fts using fts5(text, speaker, content='transcript_lines', content_rowid='rowid',
             tokenize='unicode61 remove_diacritics 2');
         create trigger transcript_ai after insert on transcript_lines begin
             insert into transcript_fts(rowid, text, speaker) values (new.rowid, new.text, coalesce(new.speaker, '')); end;
         insert into channels values ('hd', 'ZSUN-Helpdesk', 'Analiza_HD', null), ('sd', 'SEDN', 'daily_standup', null);
         insert into meetings values
             ('mtg1', 'chat', 'Standup 29 Sep', 'occ', 6000, 6600, null, 'unavailable', 'available', 0),
             ('mtg2', 'chat', 'Retro 30 Sep', 'occ', 7000, 7600,
              'Na retrospektywie ustalono: wdrazamy feature flagi w pazdzierniku. Zespol skraca standupy.', 'ok', 'ok', 0);",
    )
    .unwrap();
    let msg = |id: &str, ch: &str, th: &str, author: &str, ts: i64, text: &str| {
        c.execute("insert into messages (id, channel_id, thread_id, author, ts, text, indexed_at) values (?1,?2,?3,?4,?5,?6,0)", params![id, ch, th, author, ts, text])
            .unwrap();
    };
    msg("1", "hd", "1", "Anna Fiałkowska", 1_000, "Logowanie przez SSO nie działa od rana");
    msg("2", "hd", "1", "Tomasz Chmiel", 2_000, "ok, zrobione, restart usługi pomógł");
    // Same wording in two groups, so only the project boost decides the order.
    msg("10", "hd", "10", "Anna Fiałkowska", 3_000, "migracja bazy danych w piątek");
    msg("20", "sd", "20", "Piotr Nowak", 4_000, "migracja bazy danych w piątek");
    // A listed secret in raw teams.db text: search output must not show it.
    msg("30", "hd", "30", "Piotr Nowak", 5_000, "tymczasowy klucz plainsecret1 do konta serwisowego");

    let line = |idx: i64, speaker: Option<&str>, ts: i64, text: &str| {
        c.execute(
            "insert into transcript_lines (meeting_id, idx, speaker, offset_s, ts, text, indexed_at) values ('mtg1', ?1, ?2, null, ?3, ?4, 0)",
            params![idx, speaker, ts, text],
        )
        .unwrap();
    };
    line(0, None, 6_000, "started transcription");
    line(1, Some("Tomasz Chmiel"), 6_100, "zrobiłem migrację bazy danych na produkcji");
    line(2, Some("Anna Fiałkowska"), 6_200, "ja testuję nowy deployment");
    // A listed secret spoken aloud: transcript search/output must not show it.
    line(3, Some("Piotr Nowak"), 6_300, "klucz plainsecret1 wrzuciłem do vaulta");

    // mtg2 has a summary and a duration-suffixed speaker variant, to exercise summary chunks and
    // participant cleaning.
    let line2 = |idx: i64, speaker: Option<&str>, ts: i64, text: &str| {
        c.execute(
            "insert into transcript_lines (meeting_id, idx, speaker, offset_s, ts, text, indexed_at) values ('mtg2', ?1, ?2, null, ?3, ?4, 0)",
            params![idx, speaker, ts, text],
        )
        .unwrap();
    };
    line2(0, Some("Jan Nowak 5 minutes"), 7_010, "zaczynamy retro");
    line2(1, Some("Jan Nowak"), 7_020, "co poszło dobrze");
    line2(2, Some("Ewa Lis"), 7_030, "deploy był płynny");
}

struct Env {
    teams: TeamsSearch,
    ctx: Context,
    cfg: Config,
    _tmp: tempfile::TempDir,
    _q: testkit::Qdrant,
}

async fn env() -> Env {
    let q = testkit::qdrant().await;
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("teams.db");
    fixture_db(&db);
    let people = tmp.path().join("people.json");
    std::fs::write(
        &people,
        r#"{"groups": {"Helpdesk": {"channels": ["ZSUN-Helpdesk/"], "roles": {"Tomasz Chmiel": "leader"}},
                       "SEDN": {"channels": ["SEDN/"]}},
            "people": {"Tomasz Chmiel": {"title": "kierownik", "vip": true}}}"#,
    )
    .unwrap();
    let projects = tmp.path().join("teams-projects.json");
    std::fs::write(&projects, json!({ HELPDESK: ["Helpdesk"] }).to_string()).unwrap();
    let cfg = Config {
        claude_dir: tmp.path().join("no-claude"),
        opencode_db: tmp.path().join("no-opencode.db"),
        state_path: tmp.path().join("state.db"),
        teams_db: db.clone(),
        teams_people: people.clone(),
        teams_projects: projects.clone(),
        ..Config::from_env(|_| None)
    };
    let sessions = Arc::new(Store::connect(&q.url, None, "sessions", DIM).unwrap());
    let teams_store = Arc::new(Store::connect(&q.url, None, "teams", DIM).unwrap());
    let shared = SharedEmbedder::new(Box::new(HashEmbedder::new(DIM)));
    let runner = ctx::daemon_runner(cfg.clone(), sessions.clone(), Arc::new(WordCounter), shared, "hash".into(), None, Some(teams_store.clone()), None);
    let queue = IndexQueue::start(runner, Duration::from_millis(10));
    queue.request(Target::Teams);
    for _ in 0..500 {
        let s = queue.status();
        if s.runs == 1 && !s.running {
            assert!(s.last_error.is_none(), "{s:?}");
            assert!(s.last_report.as_deref().unwrap().contains("teams: 6 items (threads+meetings), indexed 6, failed 0"), "{s:?}");
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let embedder: Arc<Mutex<Box<dyn Embedder>>> = Arc::new(Mutex::new(Box::new(HashEmbedder::new(DIM))));
    let redact = tmp.path().join("redact.txt");
    std::fs::write(&redact, "plainsecret1
").unwrap();
    let teams = TeamsSearch { store: teams_store, embedder: embedder.clone(), db, people, projects, redact };
    Env { teams, ctx: Context { store: sessions, embedder }, cfg, _tmp: tmp, _q: q }
}

impl Env {
    async fn client(&self, project: &str, with_teams: bool) -> rmcp::service::RunningService<rmcp::RoleClient, ()> {
        let mut server = ContextServer::new(self.ctx.clone(), sources::project::key(project), self.cfg.state_path.clone());
        if with_teams {
            server = server.with_teams(self.teams.clone());
        }
        let (a, b) = tokio::io::duplex(1 << 16);
        tokio::spawn(async move {
            let running = server.serve(a).await.unwrap();
            let _ = running.waiting().await;
        });
        ().serve(b).await.unwrap()
    }
}

async fn call(c: &rmcp::service::RunningService<rmcp::RoleClient, ()>, tool: &'static str, args: Value) -> CallToolResult {
    c.call_tool(CallToolRequestParams::new(tool).with_arguments(args.as_object().cloned().unwrap())).await.unwrap()
}

fn text(r: &CallToolResult) -> String {
    r.content.iter().filter_map(|c| c.as_text().map(|t| t.text.clone())).collect()
}

async fn ok(c: &rmcp::service::RunningService<rmcp::RoleClient, ()>, tool: &'static str, args: Value) -> String {
    let r = call(c, tool, args).await;
    assert_ne!(r.is_error, Some(true), "{r:?}");
    text(&r)
}

#[tokio::test(flavor = "multi_thread")]
async fn keyword_prefix_match_from_fts_finds_the_thread_with_author_roles() {
    let e = env().await;
    let c = e.client(r"Z:\work\other", true).await;
    // "logow" only matches as an FTS prefix; the hash embedder alone would miss it.
    let out = ok(&c, "search_teams", json!({"query": "logow SSO"})).await;
    assert!(out.starts_with("[1] ZSUN-Helpdesk / Analiza_HD · thread 1 · 2 messages"), "{out}");
    let thread = ok(&c, "get_teams_thread", json!({"thread_id": "1"})).await;
    assert!(thread.contains("Tomasz Chmiel [VIP, kierownik; Helpdesk leader]"), "{thread}");
    assert!(thread.find("Logowanie").unwrap() < thread.find("restart").unwrap(), "oldest first");
}

#[tokio::test(flavor = "multi_thread")]
async fn project_mapping_ranks_its_groups_higher_without_hiding_others() {
    let e = env().await;
    let q = ctx::teams::TeamsQuery { text: "migracja bazy danych".into(), limit: 5, ..Default::default() };

    let plain = e.teams.search(&q, Some(&sources::project::key(r"Z:\work\other"))).await.unwrap();
    let order: Vec<&str> = plain.hits.iter().map(|h| h.thread_id.as_str()).collect();
    assert_eq!(&order[..2], ["20", "10"], "equal matches: the newer thread first");
    assert!(plain.boosted_groups.is_empty());

    let mapped = e.teams.search(&q, Some(&sources::project::key(HELPDESK))).await.unwrap();
    let order: Vec<&str> = mapped.hits.iter().map(|h| h.thread_id.as_str()).collect();
    assert_eq!(&order[..2], ["10", "20"], "the Helpdesk thread is ranked higher inside the helpdesk project");
    assert!(mapped.hits[0].boosted && !mapped.hits[1].boosted);
    assert_eq!(mapped.boosted_groups, ["Helpdesk"]);

    let c = e.client(HELPDESK, true).await;
    let out = ok(&c, "search_teams", json!({"query": "migracja bazy danych"})).await;
    assert!(out.starts_with("(threads in Helpdesk are ranked higher for this project)"), "{out}");
}

#[tokio::test(flavor = "multi_thread")]
async fn group_team_channel_and_author_filters_match_the_apps_rules() {
    let e = env().await;
    let c = e.client(r"Z:\work\other", true).await;
    let only_sedn = ok(&c, "search_teams", json!({"query": "migracja", "group": "sedn"})).await;
    assert!(only_sedn.contains("SEDN / daily_standup") && !only_sedn.contains("ZSUN-Helpdesk"), "{only_sedn}");
    let by_channel = ok(&c, "search_teams", json!({"query": "migracja", "channel": "ANALIZA"})).await;
    assert!(by_channel.contains("Analiza_HD") && !by_channel.contains("SEDN"), "{by_channel}");
    let by_author = ok(&c, "search_teams", json!({"query": "migracja", "author": "nowak"})).await;
    assert!(by_author.contains("thread 20") && !by_author.contains("thread 10"), "{by_author}");
    let nobody = ok(&c, "search_teams", json!({"query": "migracja", "author": "no such person"})).await;
    assert_eq!(nobody, "No matching Teams threads.");

    let unknown = call(&c, "search_teams", json!({"query": "x", "group": "Marketing"})).await;
    assert_eq!(unknown.is_error, Some(true));
    assert!(text(&unknown).contains("known groups: Helpdesk, SEDN"), "{}", text(&unknown));
}

#[tokio::test(flavor = "multi_thread")]
async fn teams_tools_explain_when_teams_is_not_configured() {
    let e = env().await;
    let c = e.client(r"Z:\work\other", false).await;
    let r = call(&c, "search_teams", json!({"query": "x"})).await;
    assert_eq!(r.is_error, Some(true));
    assert!(text(&r).contains("not configured"));

    let mut missing = e.teams.clone();
    missing.db = e.cfg.state_path.with_file_name("gone.db");
    let err = missing.search(&ctx::teams::TeamsQuery { text: "x".into(), limit: 1, ..Default::default() }, None).await.unwrap_err();
    assert!(err.to_string().contains("not set up"), "{err}");
}

#[tokio::test(flavor = "multi_thread")]
async fn listed_secrets_never_appear_in_search_or_thread_output() {
    let e = env().await;
    let c = e.client(r"Z:\work\other", true).await;
    let found = ok(&c, "search_teams", json!({"query": "klucz konta serwisowego"})).await;
    assert!(found.contains("thread 30"), "{found}");
    assert!(!found.contains("plainsecret1") && found.contains("[REDACTED:listed]"), "{found}");
    let thread = ok(&c, "get_teams_thread", json!({"thread_id": "30"})).await;
    assert!(!thread.contains("plainsecret1"), "{thread}");
}

#[tokio::test(flavor = "multi_thread")]
async fn invalid_fts_syntax_from_an_llm_still_returns_results() {
    let e = env().await;
    let c = e.client(r"Z:\work\other", true).await;
    for q in ["\"Logowanie przez SSO", "SSO OR - what's wrong?", "NEAR(logowanie"] {
        let out = ok(&c, "search_teams", json!({"query": q})).await;
        assert!(out.contains("thread 1 "), "{q}: {out}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn missing_teams_collection_degrades_to_keyword_search() {
    let e = env().await;
    let mut broken = e.teams.clone();
    broken.store = Arc::new(Store::connect("http://127.0.0.1:9", None, "teams", DIM).unwrap());
    let r = broken.search(&ctx::teams::TeamsQuery { text: "logowanie".into(), limit: 3, ..Default::default() }, None).await.unwrap();
    assert_eq!(r.hits[0].thread_id, "1", "FTS alone still finds it");
    assert!(r.notes[0].starts_with("semantic search unavailable"), "{:?}", r.notes);
}

#[tokio::test(flavor = "multi_thread")]
async fn thread_gone_from_teams_db_is_still_found_from_the_index() {
    let e = env().await;
    let c = Connection::open(&e.teams.db).unwrap();
    c.execute("delete from messages where thread_id = '20'", []).unwrap();
    drop(c);
    let r = e.teams.search(&ctx::teams::TeamsQuery { text: "migracja bazy danych".into(), limit: 5, ..Default::default() }, None).await.unwrap();
    let gone = r.hits.iter().find(|h| h.thread_id == "20").expect("kept in the index, so still a result");
    assert_eq!((gone.team.as_str(), gone.channel.as_str()), ("SEDN", "daily_standup"));
    assert!(gone.text.contains("migracja"), "{}", gone.text);
}

#[tokio::test(flavor = "multi_thread")]
async fn meeting_search_finds_transcript_and_filters_by_speaker() {
    let e = env().await;
    let c = e.client(r"Z:\work\other", true).await;
    // Prefix terms that survive diacritics folding (migracj- matches "migrację").
    let out = ok(&c, "search_meetings", json!({"query": "migracj baz"})).await;
    assert!(out.contains("Standup 29 Sep") && out.contains("meeting mtg1"), "{out}");
    assert!(out.contains("Tomasz Chmiel"), "best matching line is Tomasz's: {out}");
    // Speaker filter narrows the keyword half to Anna's line.
    let anna = ok(&c, "search_meetings", json!({"query": "testuje deployment", "speaker": "fialkowska"})).await;
    assert!(anna.contains("Anna Fiałkowska"), "{anna}");
}

#[tokio::test(flavor = "multi_thread")]
async fn get_meeting_assembles_the_transcript_speaker_labelled_and_ordered() {
    let e = env().await;
    let c = e.client(r"Z:\work\other", true).await;
    let t = ok(&c, "get_meeting", json!({"meeting_id": "mtg1"})).await;
    assert!(t.starts_with("Standup 29 Sep · "), "{t}");
    assert!(t.contains("— ("), "the system line is shown with a neutral speaker");
    assert!(t.find("migrację").unwrap() < t.find("deployment").unwrap(), "lines in spoken order");
}

#[tokio::test(flavor = "multi_thread")]
async fn listed_secrets_never_appear_in_meeting_output() {
    let e = env().await;
    let c = e.client(r"Z:\work\other", true).await;
    let found = ok(&c, "search_meetings", json!({"query": "klucz vault"})).await;
    assert!(found.contains("mtg1") && !found.contains("plainsecret1"), "{found}");
    let t = ok(&c, "get_meeting", json!({"meeting_id": "mtg1"})).await;
    assert!(!t.contains("plainsecret1"), "{t}");
}

#[tokio::test(flavor = "multi_thread")]
async fn meeting_summary_is_indexed_as_its_own_chunks() {
    let e = env().await;
    // mtg2 has an "ok" summary, so it is indexed as at least one `summary` chunk (not folded into
    // the oversized single chunk the old code produced).
    let n = e.teams.store.count(&store::SearchFilter::default().eq("kind", "summary")).await.unwrap();
    assert!(n >= 1, "summary indexed as its own chunk(s), got {n}");
}

#[tokio::test(flavor = "multi_thread")]
async fn get_meeting_leads_with_cleaned_participants_and_the_summary() {
    let e = env().await;
    let c = e.client(r"Z:\work\other", true).await;
    let t = ok(&c, "get_meeting", json!({"meeting_id": "mtg2"})).await;
    // The participants line is the cleaned roster (the raw per-line transcript labels stay faithful).
    let roster = t.lines().find(|l| l.starts_with("Uczestnicy:")).expect("participants line");
    assert_eq!(roster, "Uczestnicy: Ewa Lis, Jan Nowak", "cleaned, deduped, sorted");
    assert!(t.contains("Podsumowanie:") && t.contains("feature flagi w pazdzierniku"), "{t}");
    assert!(t.find("Uczestnicy:").unwrap() < t.find("Podsumowanie:").unwrap(), "participants first");
    assert!(t.find("Podsumowanie:").unwrap() < t.find("Transkrypcja:").unwrap(), "summary leads the transcript");
}
