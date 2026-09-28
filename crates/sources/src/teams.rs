//! Read-only access to the Teams app's store (`D:\dev\my\teams\data\teams.db`): channels,
//! messages grouped into threads, and its FTS5 index.
//!
//! The Teams app owns this database and writes it hourly (WAL), so it is only ever opened
//! read-only. Filter semantics copy the app's `scope()`: team, channel and author filters are
//! case- and diacritics-insensitive substring matches (see [`name_key`]), so `search_teams`
//! agrees with the app's own `search_messages`.

use std::collections::BTreeMap;
use std::path::Path;

use rusqlite::{Connection, OpenFlags, params_from_iter};
use unicode_normalization::UnicodeNormalization;

use crate::{Result, SourceError};

/// The app's `nameKey`: NFD, drop combining marks, `ł`→`l` (it has no decomposition),
/// lower-case, collapse whitespace. "Fialkowski" matches "Fiałkowski".
pub fn name_key(s: &str) -> String {
    let folded: String = s
        .nfd()
        .filter(|c| !unicode_normalization::char::is_combining_mark(*c))
        .map(|c| match c {
            'ł' => 'l',
            'Ł' => 'L',
            c => c,
        })
        .collect::<String>()
        .to_lowercase();
    folded.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Channel {
    pub id: String,
    pub team: String,
    pub name: String,
}

impl Channel {
    /// `Team/Channel`, the form `people.json` groups match against.
    pub fn path(&self) -> String {
        format!("{}/{}", self.team, self.name)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub id: String,
    pub thread_id: String,
    pub author: String,
    pub ts: i64,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Thread {
    pub id: String,
    pub channel: Channel,
    /// Oldest first. The first is the root post when the sync reached it; a thread whose root
    /// is older than the app's lookback window starts with its oldest known reply.
    pub messages: Vec<Message>,
}

impl Thread {
    /// Changes whenever a reply is added (messages are never edited in the app's store).
    /// Channel names are included: the app updates them on rename, and they are part of every
    /// chunk's header and payload.
    pub fn fingerprint(&self) -> String {
        let last = self.messages.last().map_or(0, |m| m.ts);
        format!("{}-{last}-{}", self.messages.len(), self.channel.path())
    }

    pub fn last_ts(&self) -> i64 {
        self.messages.last().map_or(0, |m| m.ts)
    }
}

/// One FTS5 hit.
#[derive(Debug, Clone, PartialEq)]
pub struct FtsHit {
    pub message_id: String,
    pub thread_id: String,
    pub channel_id: String,
    pub author: String,
    pub ts: i64,
    pub text: String,
}

pub struct TeamsDb {
    conn: Connection,
}

impl TeamsDb {
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX)?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        Ok(Self { conn })
    }

    pub fn channels(&self) -> Result<Vec<Channel>> {
        let mut stmt = self.conn.prepare("select id, team, name from channels order by team, name")?;
        let rows = stmt.query_map([], |r| Ok(Channel { id: r.get(0)?, team: r.get(1)?, name: r.get(2)? }))?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// The app's own sync heartbeat: the most recent `last_synced_at` across channels, in epoch
    /// ms, or `None` when no channel has ever synced. This tracks whether the scraper is running,
    /// not whether new messages arrived — a quiet but healthy scraper still advances it, so it
    /// tells a stopped scraper apart from a quiet channel.
    pub fn last_synced_at(&self) -> Result<Option<i64>> {
        Ok(self.conn.query_row("select max(last_synced_at) from channels", [], |r| r.get::<_, Option<i64>>(0))?)
    }

    pub fn authors(&self) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare("select distinct author from messages order by author")?;
        let rows = stmt.query_map([], |r| r.get(0))?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Every thread with its messages, oldest message first.
    pub fn threads(&self) -> Result<Vec<Thread>> {
        let channels: BTreeMap<String, Channel> = self.channels()?.into_iter().map(|c| (c.id.clone(), c)).collect();
        let mut stmt = self
            .conn
            .prepare("select id, thread_id, channel_id, author, ts, text from messages order by thread_id, ts, rowid")?;
        let mut threads: BTreeMap<String, Thread> = BTreeMap::new();
        let rows = stmt.query_map([], |r| {
            Ok((
                Message { id: r.get(0)?, thread_id: r.get(1)?, author: r.get(3)?, ts: r.get(4)?, text: r.get(5)? },
                r.get::<_, String>(2)?,
            ))
        })?;
        for row in rows {
            let (m, channel_id) = row?;
            let Some(channel) = channels.get(&channel_id) else { continue };
            threads
                .entry(m.thread_id.clone())
                .or_insert_with(|| Thread { id: m.thread_id.clone(), channel: channel.clone(), messages: Vec::new() })
                .messages
                .push(m);
        }
        Ok(threads.into_values().collect())
    }

    pub fn thread(&self, thread_id: &str) -> Result<Thread> {
        self.threads()?
            .into_iter()
            .find(|t| t.id == thread_id)
            .ok_or_else(|| SourceError::NotFound(thread_id.to_owned()))
    }

    /// FTS5 BM25 search, best first, restricted to `channel_ids` / `authors` when given (an
    /// empty list matches nothing, like an empty group in the app).
    /// A query using FTS5 syntax is passed through, like the app does; if it is not valid FTS5
    /// (an unclosed quote, a dangling `OR`, common in LLM-written queries), it is retried as
    /// plain words instead of failing the search.
    pub fn fts(&self, query: &str, filter: &Filter, limit: usize) -> Result<Vec<FtsHit>> {
        match self.fts_with(&to_fts_query(query), filter, limit) {
            Err(SourceError::Sqlite(_)) => self.fts_with(&plain_fts_query(query), filter, limit),
            other => other,
        }
    }

    fn fts_with(&self, fts_query: &str, filter: &Filter, limit: usize) -> Result<Vec<FtsHit>> {
        let mut sql = String::from(
            "select m.id, m.thread_id, m.channel_id, m.author, m.ts, m.text
             from messages_fts f join messages m on m.rowid = f.rowid
             where messages_fts match ?",
        );
        let mut args: Vec<rusqlite::types::Value> = vec![fts_query.to_owned().into()];
        if let Some(ids) = &filter.channel_ids {
            push_in(&mut sql, &mut args, "m.channel_id", ids);
        }
        if let Some(names) = &filter.authors {
            push_in(&mut sql, &mut args, "m.author", names);
        }
        if let Some(since) = filter.since {
            sql.push_str(" and m.ts >= ?");
            args.push(since.into());
        }
        if let Some(until) = filter.until {
            sql.push_str(" and m.ts <= ?");
            args.push(until.into());
        }
        sql.push_str(" order by bm25(messages_fts) limit ?");
        args.push((limit as i64).into());
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(args), |r| {
            Ok(FtsHit { message_id: r.get(0)?, thread_id: r.get(1)?, channel_id: r.get(2)?, author: r.get(3)?, ts: r.get(4)?, text: r.get(5)? })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Channel ids matching the app's filters: `team`/`channel` are substrings of the team and
    /// channel names, `paths` (a group) substrings of `Team/Channel`; all folded by [`name_key`].
    pub fn resolve_channels(&self, team: Option<&str>, channel: Option<&str>, paths: Option<&[String]>) -> Result<Vec<String>> {
        let contains = |hay: &str, needle: &str| name_key(hay).contains(&name_key(needle));
        Ok(self
            .channels()?
            .into_iter()
            .filter(|c| team.is_none_or(|t| contains(&c.team, t)))
            .filter(|c| channel.is_none_or(|n| contains(&c.name, n)))
            .filter(|c| paths.is_none_or(|ps| ps.iter().any(|p| contains(&c.path(), p))))
            .map(|c| c.id)
            .collect())
    }

    /// Authors whose name contains `needle` (folded).
    pub fn resolve_authors(&self, needle: &str) -> Result<Vec<String>> {
        let k = name_key(needle);
        Ok(self.authors()?.into_iter().filter(|a| name_key(a).contains(&k)).collect())
    }
}

/// Query-side restrictions. `None` means unrestricted; `Some(empty)` matches nothing.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Filter {
    pub channel_ids: Option<Vec<String>>,
    pub authors: Option<Vec<String>>,
    pub since: Option<i64>,
    pub until: Option<i64>,
}

fn push_in(sql: &mut String, args: &mut Vec<rusqlite::types::Value>, col: &str, values: &[String]) {
    if values.is_empty() {
        sql.push_str(" and 0");
        return;
    }
    sql.push_str(&format!(" and {col} in ({})", vec!["?"; values.len()].join(",")));
    args.extend(values.iter().cloned().map(Into::into));
}

/// The app's `toFtsQuery`: plain words become an AND of prefix terms, so LLM-written queries
/// never hit FTS5 syntax errors; queries already using FTS5 operators pass through.
pub fn to_fts_query(q: &str) -> String {
    if q.contains('"') || q.split_whitespace().any(|w| matches!(w, "AND" | "OR" | "NOT" | "NEAR")) {
        return q.to_owned();
    }
    plain_fts_query(q)
}

/// Every word as a quoted prefix term, operators and quotes dropped; always valid FTS5.
pub fn plain_fts_query(q: &str) -> String {
    let terms: Vec<String> = q
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|t| !t.is_empty())
        .map(|t| format!("\"{t}\"*"))
        .collect();
    if terms.is_empty() { "\"\"".into() } else { terms.join(" ") }
}

#[cfg(test)]
pub(crate) mod fixture {
    use rusqlite::{Connection, params};

    /// The app's schema (src/db.ts), with its FTS triggers.
    pub fn create(path: &std::path::Path) -> Connection {
        let c = Connection::open(path).unwrap();
        c.execute_batch(
            "pragma journal_mode = wal;
             create table channels (id text primary key, team text not null, name text not null, last_synced_at integer);
             create table messages (rowid integer primary key, id text not null unique, channel_id text not null references channels(id),
                 thread_id text not null, author text not null, ts integer not null, text text not null, indexed_at integer not null);
             create table threads (id text primary key, reply_total integer not null);
             create virtual table messages_fts using fts5(text, author, content='messages', content_rowid='rowid',
                 tokenize='unicode61 remove_diacritics 2');
             create trigger messages_ai after insert on messages begin
                 insert into messages_fts(rowid, text, author) values (new.rowid, new.text, new.author); end;",
        )
        .unwrap();
        c
    }

    pub fn channel(c: &Connection, id: &str, team: &str, name: &str) {
        c.execute("insert into channels (id, team, name) values (?1, ?2, ?3)", params![id, team, name]).unwrap();
    }

    pub fn synced(c: &Connection, id: &str, at: i64) {
        c.execute("update channels set last_synced_at = ?2 where id = ?1", params![id, at]).unwrap();
    }

    pub fn message(c: &Connection, id: &str, channel: &str, thread: &str, author: &str, ts: i64, text: &str) {
        c.execute(
            "insert into messages (id, channel_id, thread_id, author, ts, text, indexed_at) values (?1, ?2, ?3, ?4, ?5, ?6, 0)",
            params![id, channel, thread, author, ts, text],
        )
        .unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::fixture::*;
    use super::*;

    fn db() -> (tempfile::TempDir, TeamsDb) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("teams.db");
        let c = create(&path);
        channel(&c, "c1", "ZSUN-Helpdesk", "Analiza_HD");
        channel(&c, "c2", "SEDN", "daily_standup");
        message(&c, "100", "c1", "100", "Tomasz Chmiel", 100, "Logowanie przez SSO nie działa od rana");
        message(&c, "101", "c1", "100", "Anna Fiałkowska", 101, "ok, zrobione");
        message(&c, "200", "c2", "200", "Łukasz Nowak", 200, "Migracja bazy przesunięta na piątek");
        message(&c, "301", "c2", "300", "Anna Fiałkowska", 300, "reply whose root is outside the lookback");
        drop(c);
        let db = TeamsDb::open(&path).unwrap();
        (dir, db)
    }

    #[test]
    fn name_key_matches_the_apps_folding() {
        assert_eq!(name_key("  Anna   Fiałkowska "), "anna fialkowska");
        assert_eq!(name_key("ŁUKASZ Żółć"), "lukasz zolc");
        assert_eq!(name_key("Rozwój_HD"), "rozwoj_hd");
    }

    #[test]
    fn threads_group_messages_oldest_first_with_their_channel() {
        let (_d, db) = db();
        let threads = db.threads().unwrap();
        assert_eq!(threads.len(), 3);
        let t = threads.iter().find(|t| t.id == "100").unwrap();
        assert_eq!(t.channel.path(), "ZSUN-Helpdesk/Analiza_HD");
        assert_eq!(t.messages.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(), ["100", "101"]);
        assert_eq!(t.fingerprint(), "2-101-ZSUN-Helpdesk/Analiza_HD");
        let orphan = threads.iter().find(|t| t.id == "300").unwrap();
        assert_eq!(orphan.messages[0].id, "301", "thread without its root starts at the oldest reply");
    }

    #[test]
    fn fingerprint_changes_when_a_reply_arrives() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("teams.db");
        let c = create(&path);
        channel(&c, "c1", "T", "C");
        message(&c, "1", "c1", "1", "A", 1, "root");
        let before = TeamsDb::open(&path).unwrap().threads().unwrap()[0].fingerprint();
        message(&c, "2", "c1", "1", "B", 2, "reply");
        let after = TeamsDb::open(&path).unwrap().threads().unwrap()[0].fingerprint();
        assert_ne!(before, after);
    }

    #[test]
    fn last_synced_at_is_the_newest_heartbeat_or_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("teams.db");
        let c = create(&path);
        channel(&c, "c1", "T", "C1");
        channel(&c, "c2", "T", "C2");
        assert_eq!(TeamsDb::open(&path).unwrap().last_synced_at().unwrap(), None, "never synced");
        synced(&c, "c1", 500);
        synced(&c, "c2", 900);
        assert_eq!(TeamsDb::open(&path).unwrap().last_synced_at().unwrap(), Some(900), "newest across channels");
    }

    #[test]
    fn channel_resolution_uses_folded_substrings_like_the_app() {
        let (_d, db) = db();
        assert_eq!(db.resolve_channels(Some("helpdesk"), None, None).unwrap(), ["c1"]);
        assert_eq!(db.resolve_channels(None, Some("STANDUP"), None).unwrap(), ["c2"]);
        let group = vec!["ZSUN-Helpdesk/".to_owned()];
        assert_eq!(db.resolve_channels(None, None, Some(&group)).unwrap(), ["c1"]);
        assert!(db.resolve_channels(None, None, Some(&[])).unwrap().is_empty(), "empty group matches nothing");
        assert_eq!(db.resolve_channels(None, None, None).unwrap().len(), 2);
    }

    #[test]
    fn author_resolution_is_diacritics_insensitive() {
        let (_d, db) = db();
        assert_eq!(db.resolve_authors("fialkowska").unwrap(), ["Anna Fiałkowska"]);
        assert_eq!(db.resolve_authors("lukasz").unwrap(), ["Łukasz Nowak"]);
    }

    #[test]
    fn fts_finds_prefix_terms_without_diacritics_and_respects_filters() {
        let (_d, db) = db();
        let hits = db.fts("logowanie dzia", &Filter::default(), 10).unwrap();
        assert_eq!(hits.len(), 1, "prefix terms");
        assert_eq!(hits[0].thread_id, "100");
        // remove_diacritics folds letters that decompose (ę → e) but not ł, exactly as in the
        // app's own search; the semantic side of search_teams covers "dziala" for "działa".
        assert_eq!(db.fts("przesuniet", &Filter::default(), 10).unwrap().len(), 1);
        assert!(db.fts("dziala", &Filter::default(), 10).unwrap().is_empty());
        let only_sedn = Filter { channel_ids: Some(vec!["c2".into()]), ..Default::default() };
        assert!(db.fts("logowanie", &only_sedn, 10).unwrap().is_empty());
        let nobody = Filter { authors: Some(vec![]), ..Default::default() };
        assert!(db.fts("logowanie", &nobody, 10).unwrap().is_empty());
        let late = Filter { since: Some(150), ..Default::default() };
        assert_eq!(db.fts("migracja", &late, 10).unwrap().len(), 1);
    }

    #[test]
    fn fts_query_conversion_matches_the_app() {
        assert_eq!(to_fts_query("SSO nie działa!"), "\"SSO\"* \"nie\"* \"działa\"*");
        assert_eq!(to_fts_query("a OR b"), "a OR b");
        assert_eq!(to_fts_query("\"exact phrase\""), "\"exact phrase\"");
        assert_eq!(to_fts_query("!!!"), "\"\"");
    }

    #[test]
    fn llm_written_punctuation_never_breaks_fts() {
        let (_d, db) = db();
        for q in ["what's up? (SSO)", "a*b", "-x", "col:value", "\"Invalid token", "SSO OR SAML - what's wrong?", "NEAR(x", "AND"] {
            assert!(db.fts(q, &Filter::default(), 5).is_ok(), "{q}");
        }
    }

    #[test]
    fn invalid_fts_syntax_falls_back_to_plain_words() {
        let (_d, db) = db();
        let hits = db.fts("\"logowanie SSO", &Filter::default(), 10).unwrap();
        assert_eq!(hits.len(), 1, "unclosed quote retried as plain words");
        assert_eq!(plain_fts_query("a \"b\" OR c"), "\"a\"* \"b\"* \"OR\"* \"c\"*");
    }

    #[test]
    fn database_is_opened_read_only() {
        let (_d, db) = db();
        assert!(db.conn.execute("delete from messages", []).is_err());
    }

    #[test]
    fn unknown_thread_is_not_found() {
        let (_d, db) = db();
        assert!(matches!(db.thread("nope"), Err(SourceError::NotFound(_))));
        assert_eq!(db.thread("200").unwrap().messages.len(), 1);
    }
}
