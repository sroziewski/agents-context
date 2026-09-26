//! sync_teams against a fixture teams.db (the app's schema) and a real Qdrant.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use embed::{Embedder, Embedding, HashEmbedder};
use ingest::teams::{TEAMS_FIELDS, sync_teams};
use ingest::{IndexConfig, State, WordCounter};
use rusqlite::{Connection, params};
use sources::teams::TeamsDb;
use store::{SearchFilter, Store};

const DIM: usize = 64;

struct Counting(HashEmbedder, Arc<AtomicUsize>);
impl Embedder for Counting {
    fn dim(&self) -> usize {
        DIM
    }
    fn embed(&mut self, texts: &[&str]) -> embed::Result<Vec<Embedding>> {
        self.1.fetch_add(texts.len(), Ordering::SeqCst);
        self.0.embed(texts)
    }
}

fn create(path: &Path) -> Connection {
    let c = Connection::open(path).unwrap();
    c.execute_batch(
        "pragma journal_mode = wal;
         create table channels (id text primary key, team text not null, name text not null, last_synced_at integer);
         create table messages (rowid integer primary key, id text not null unique, channel_id text not null,
             thread_id text not null, author text not null, ts integer not null, text text not null, indexed_at integer not null);
         create virtual table messages_fts using fts5(text, author, content='messages', content_rowid='rowid',
             tokenize='unicode61 remove_diacritics 2');
         create trigger messages_ai after insert on messages begin
             insert into messages_fts(rowid, text, author) values (new.rowid, new.text, new.author); end;
         insert into channels values ('c1', 'ZSUN-Helpdesk', 'Analiza_HD', null), ('c2', 'SEDN', 'daily_standup', null);",
    )
    .unwrap();
    c
}

fn msg(c: &Connection, id: &str, ch: &str, thread: &str, author: &str, ts: i64, text: &str) {
    c.execute("insert into messages (id, channel_id, thread_id, author, ts, text, indexed_at) values (?1,?2,?3,?4,?5,?6,0)", params![id, ch, thread, author, ts, text]).unwrap();
}

struct Env {
    store: Store,
    state: State,
    db_path: PathBuf,
    calls: Arc<AtomicUsize>,
    _tmp: tempfile::TempDir,
    _q: testkit::Qdrant,
}

async fn env() -> Env {
    let q = testkit::qdrant().await;
    let store = Store::connect(&q.url, None, "teams", DIM).unwrap();
    store.ensure_collection(TEAMS_FIELDS).await.unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let db_path = tmp.path().join("teams.db");
    let c = create(&db_path);
    msg(&c, "100", "c1", "100", "Anna", 100, "Logowanie przez SSO nie działa");
    msg(&c, "101", "c1", "100", "Piotr", 101, "ok, zrobione");
    msg(&c, "200", "c2", "200", "Łukasz", 200, "Migracja bazy w piątek");
    Env { store, state: State::in_memory().unwrap(), db_path, calls: Arc::default(), _tmp: tmp, _q: q }
}

impl Env {
    async fn sync(&self) -> ingest::Report {
        let db = TeamsDb::open(&self.db_path).unwrap();
        let mut e = Counting(HashEmbedder::new(DIM), self.calls.clone());
        sync_teams(&self.store, &mut e, &WordCounter, &self.state, &db, &IndexConfig::default(), 1).await.unwrap()
    }

    async fn count(&self, f: SearchFilter) -> u64 {
        self.store.count(&f).await.unwrap()
    }
}

#[tokio::test]
async fn first_sync_writes_thread_and_message_chunks_per_thread() {
    let e = env().await;
    let r = e.sync().await;
    assert_eq!((r.scanned, r.indexed, r.unchanged), (2, 2, 0));
    assert_eq!(e.count(SearchFilter::default().eq("kind", "thread")).await, 2);
    assert_eq!(e.count(SearchFilter::default().eq("kind", "message")).await, 3);
    assert_eq!(e.count(SearchFilter::default().eq("thread_id", "100")).await, 3);
    let st = e.state.get("teams", "100").unwrap().unwrap();
    assert_eq!((st.fingerprint.as_str(), st.chunks), ("2-101-ZSUN-Helpdesk/Analiza_HD", 3));
}

#[tokio::test]
async fn unchanged_threads_are_skipped_and_a_new_reply_reindexes_only_its_thread() {
    let e = env().await;
    e.sync().await;
    let after_first = e.calls.load(Ordering::SeqCst);
    let r = e.sync().await;
    assert_eq!((r.indexed, r.unchanged), (0, 2));
    assert_eq!(e.calls.load(Ordering::SeqCst), after_first, "nothing re-embedded");

    let c = Connection::open(&e.db_path).unwrap();
    msg(&c, "102", "c1", "100", "Anna", 102, "działa już, dzięki");
    drop(c);
    let r = e.sync().await;
    assert_eq!((r.indexed, r.unchanged), (1, 1));
    // The thread chunk text changed and there is one new message chunk; the two old message
    // chunks for "100" and "101" gain no new context, the new one references its predecessor.
    assert_eq!(e.calls.load(Ordering::SeqCst) - after_first, 2);
    assert_eq!(e.count(SearchFilter::default().eq("thread_id", "100").eq("kind", "message")).await, 3);
}

#[tokio::test]
async fn teams_state_rows_never_become_summary_candidates() {
    let e = env().await;
    e.sync().await;
    assert!(e.state.pending_summaries(1, 5, 3, 100).unwrap().is_empty(), "project '' rows are not sessions");
    assert!(e.state.sessions_missing_turns().unwrap().is_empty(), "turn counts are recorded for teams rows");
}

#[tokio::test]
async fn threads_that_leave_the_database_keep_their_points() {
    let e = env().await;
    e.sync().await;
    let c = Connection::open(&e.db_path).unwrap();
    c.execute("delete from messages where thread_id = '200'", []).unwrap();
    drop(c);
    e.sync().await;
    assert_eq!(e.count(SearchFilter::default().eq("thread_id", "200")).await, 2, "outlives the app's lookback window");
}
