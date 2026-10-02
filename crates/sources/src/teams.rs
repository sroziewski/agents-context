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

/// Teams sometimes suffixes a transcript speaker with a running duration, e.g.
/// "Anna Wierzbicka 19 minutes". Strip a trailing " <n> <time-unit>" so the same person is one
/// participant; anything else is returned with its whitespace normalised.
pub fn clean_speaker(name: &str) -> String {
    let words: Vec<&str> = name.split_whitespace().collect();
    if words.len() >= 3 {
        let unit = words[words.len() - 1].to_ascii_lowercase();
        let num = words[words.len() - 2];
        if num.chars().all(|c| c.is_ascii_digit())
            && matches!(unit.trim_end_matches('s'), "minute" | "second" | "hour" | "min" | "sec" | "hr" | "minut" | "sekund" | "godzin")
        {
            return words[..words.len() - 2].join(" ");
        }
    }
    words.join(" ")
}

/// Distinct participant display names from a transcript: null-speaker (system) lines skipped,
/// each name cleaned by [`clean_speaker`], deduplicated by folded name, sorted.
pub fn participants(lines: &[TranscriptLine]) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut out: Vec<String> = Vec::new();
    for l in lines {
        if let Some(s) = l.speaker.as_deref() {
            let name = clean_speaker(s);
            if !name.is_empty() && seen.insert(name_key(&name)) {
                out.push(name);
            }
        }
    }
    out.sort_unstable();
    out
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

/// A recorded meeting (a standup, etc.). Its transcript lives in `transcript_lines`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Meeting {
    pub id: String,
    pub chat: String,
    pub title: String,
    pub occurrence: String,
    pub started_at: i64,
    pub ended_at: i64,
    /// The AI "Podsumowanie": often absent (licence-gated), so optional.
    pub summary: Option<String>,
    pub summary_status: String,
    pub transcript_status: String,
}

impl Meeting {
    /// Changes when the transcript grows or its status changes (lines are append-only, never
    /// edited), or when a summary arrives: Claude summarises after the transcript is stored, so
    /// without it a late summary would never reach the index. An unchanged meeting is skipped.
    pub fn fingerprint(&self, lines: &[TranscriptLine]) -> String {
        let last = lines.last();
        let summary = if self.has_summary() { self.summary.as_deref().map_or(0, |s| s.trim().len()) } else { 0 };
        format!("{}-{}-{}-{}-s{summary}", lines.len(), last.map_or(0, |l| l.idx), last.map_or(0, |l| l.ts), self.transcript_status)
    }

    /// The exporter marks a captured summary "ok"; "available" is what earlier builds expected.
    pub fn has_summary(&self) -> bool {
        self.summary.as_deref().is_some_and(|s| !s.trim().is_empty())
            && (self.summary_status.eq_ignore_ascii_case("ok") || self.summary_status.eq_ignore_ascii_case("available"))
    }
}

/// One line of a meeting transcript. `speaker`/`offset_s` are absent on system lines (e.g. the
/// opening "started transcription"). `ts` is the absolute time the line was spoken.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptLine {
    pub meeting_id: String,
    pub idx: i64,
    pub speaker: Option<String>,
    pub offset_s: Option<i64>,
    pub ts: i64,
    pub text: String,
}

/// One transcript FTS5 hit.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptHit {
    pub meeting_id: String,
    pub idx: i64,
    pub speaker: Option<String>,
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

    /// Every meeting, newest first.
    pub fn meetings(&self) -> Result<Vec<Meeting>> {
        let mut stmt = self.conn.prepare(
            "select id, chat, title, occurrence, started_at, ended_at, summary, summary_status, transcript_status
             from meetings order by started_at desc",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(Meeting {
                id: r.get(0)?,
                chat: r.get(1)?,
                title: r.get(2)?,
                occurrence: r.get(3)?,
                started_at: r.get(4)?,
                ended_at: r.get(5)?,
                summary: r.get(6)?,
                summary_status: r.get(7)?,
                transcript_status: r.get(8)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// One meeting's transcript, in spoken order.
    pub fn transcript(&self, meeting_id: &str) -> Result<Vec<TranscriptLine>> {
        let mut stmt =
            self.conn.prepare("select meeting_id, idx, speaker, offset_s, ts, text from transcript_lines where meeting_id = ? order by idx")?;
        let rows = stmt.query_map([meeting_id], |r| {
            Ok(TranscriptLine { meeting_id: r.get(0)?, idx: r.get(1)?, speaker: r.get(2)?, offset_s: r.get(3)?, ts: r.get(4)?, text: r.get(5)? })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub fn meeting(&self, meeting_id: &str) -> Result<Meeting> {
        self.meetings()?.into_iter().find(|m| m.id == meeting_id).ok_or_else(|| SourceError::NotFound(meeting_id.to_owned()))
    }

    /// FTS5 BM25 over transcript lines, best first, like [`Self::fts`]: valid FTS5 passes through,
    /// otherwise it is retried as plain prefix words. Restricted to `meeting_ids` / `speakers`
    /// when given (an empty list matches nothing).
    pub fn transcript_fts(&self, query: &str, filter: &TranscriptFilter, limit: usize) -> Result<Vec<TranscriptHit>> {
        match self.transcript_fts_with(&to_fts_query(query), filter, limit) {
            Err(SourceError::Sqlite(_)) => self.transcript_fts_with(&plain_fts_query(query), filter, limit),
            other => other,
        }
    }

    fn transcript_fts_with(&self, fts_query: &str, filter: &TranscriptFilter, limit: usize) -> Result<Vec<TranscriptHit>> {
        let mut sql = String::from(
            "select l.meeting_id, l.idx, l.speaker, l.ts, l.text
             from transcript_fts f join transcript_lines l on l.rowid = f.rowid
             where transcript_fts match ?",
        );
        let mut args: Vec<rusqlite::types::Value> = vec![fts_query.to_owned().into()];
        if let Some(ids) = &filter.meeting_ids {
            push_in(&mut sql, &mut args, "l.meeting_id", ids);
        }
        if let Some(names) = &filter.speakers {
            push_in(&mut sql, &mut args, "l.speaker", names);
        }
        if let Some(since) = filter.since {
            sql.push_str(" and l.ts >= ?");
            args.push(since.into());
        }
        if let Some(until) = filter.until {
            sql.push_str(" and l.ts <= ?");
            args.push(until.into());
        }
        sql.push_str(" order by bm25(transcript_fts) limit ?");
        args.push((limit as i64).into());
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(args), |r| {
            Ok(TranscriptHit { meeting_id: r.get(0)?, idx: r.get(1)?, speaker: r.get(2)?, ts: r.get(3)?, text: r.get(4)? })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Distinct transcript speakers whose name contains `needle` (folded). NULL (system) lines
    /// are excluded.
    pub fn resolve_speakers(&self, needle: &str) -> Result<Vec<String>> {
        let k = name_key(needle);
        let mut stmt = self.conn.prepare("select distinct speaker from transcript_lines where speaker is not null order by speaker")?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?.into_iter().filter(|s| name_key(s).contains(&k)).collect())
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

/// Query-side restrictions for a transcript search. `None` means unrestricted; `Some(empty)`
/// matches nothing.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TranscriptFilter {
    pub meeting_ids: Option<Vec<String>>,
    pub speakers: Option<Vec<String>>,
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
                 insert into transcript_fts(rowid, text, speaker) values (new.rowid, new.text, coalesce(new.speaker, '')); end;",
        )
        .unwrap();
        c
    }

    pub fn meeting(c: &Connection, id: &str, title: &str, started: i64, ended: i64, transcript_status: &str) {
        c.execute(
            "insert into meetings (id, chat, title, occurrence, started_at, ended_at, summary, summary_status, transcript_status, indexed_at)
             values (?1, 'chat', ?2, 'occ', ?3, ?4, null, 'unavailable', ?5, 0)",
            params![id, title, started, ended, transcript_status],
        )
        .unwrap();
    }

    pub fn line(c: &Connection, meeting: &str, idx: i64, speaker: Option<&str>, ts: i64, text: &str) {
        c.execute(
            "insert into transcript_lines (meeting_id, idx, speaker, offset_s, ts, text, indexed_at) values (?1, ?2, ?3, null, ?4, ?5, 0)",
            params![meeting, idx, speaker, ts, text],
        )
        .unwrap();
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
    fn meetings_are_newest_first_with_their_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("teams.db");
        let c = create(&path);
        meeting(&c, "m1", "Standup 8 Sep", 800, 900, "available");
        meeting(&c, "m2", "Standup 29 Sep", 2900, 3000, "available");
        line(&c, "m2", 0, None, 2900, "started transcription");
        line(&c, "m2", 1, Some("Tomasz Chmiel"), 2905, "Zrobiłem migrację bazy");
        line(&c, "m2", 2, Some("Anna Fiałkowska"), 2910, "ok, dzięki");
        drop(c);
        let db = TeamsDb::open(&path).unwrap();
        assert_eq!(db.meetings().unwrap().iter().map(|m| m.id.as_str()).collect::<Vec<_>>(), ["m2", "m1"], "newest first");
        let lines = db.transcript("m2").unwrap();
        assert_eq!(lines.iter().map(|l| l.idx).collect::<Vec<_>>(), [0, 1, 2], "idx order");
        assert_eq!(lines[0].speaker, None, "system line kept, no speaker");
        assert_eq!(lines[1].speaker.as_deref(), Some("Tomasz Chmiel"));
        assert_eq!(db.meeting("m2").unwrap().title, "Standup 29 Sep");
        assert!(db.meeting("nope").is_err());
    }

    #[test]
    fn transcript_fts_finds_polish_without_diacritics_and_filters_speaker() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("teams.db");
        let c = create(&path);
        meeting(&c, "m1", "Standup", 100, 200, "available");
        line(&c, "m1", 0, Some("Tomasz Chmiel"), 105, "Zrobiłem migrację bazy danych");
        line(&c, "m1", 1, Some("Anna Fiałkowska"), 110, "ja testuję deployment");
        drop(c);
        let db = TeamsDb::open(&path).unwrap();
        // remove_diacritics 2 folds ę→e, so the prefix "migracj" matches "migrację".
        let hits = db.transcript_fts("migracj baz", &TranscriptFilter::default(), 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].speaker.as_deref(), Some("Tomasz Chmiel"));
        let anna = TranscriptFilter { speakers: Some(vec!["Anna Fiałkowska".into()]), ..Default::default() };
        assert_eq!(db.transcript_fts("deployment testuje", &anna, 10).unwrap().len(), 1, "speaker filter keeps Anna");
        let tomasz = TranscriptFilter { speakers: Some(vec!["Tomasz Chmiel".into()]), ..Default::default() };
        assert!(db.transcript_fts("deployment", &tomasz, 10).unwrap().is_empty(), "speaker filter excludes others");
    }

    #[test]
    fn resolve_speakers_is_folded_substring_and_skips_system_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("teams.db");
        let c = create(&path);
        meeting(&c, "m1", "Standup", 100, 200, "available");
        line(&c, "m1", 0, Some("Tomasz Chmiel"), 105, "x");
        line(&c, "m1", 1, None, 106, "started transcription");
        drop(c);
        let db = TeamsDb::open(&path).unwrap();
        assert_eq!(db.resolve_speakers("chmiel").unwrap(), ["Tomasz Chmiel"]);
        assert!(db.resolve_speakers("nobody").unwrap().is_empty());
    }

    #[test]
    fn clean_speaker_strips_the_duration_suffix_and_participants_dedup() {
        assert_eq!(clean_speaker("Anna Wierzbicka 19 minutes"), "Anna Wierzbicka");
        assert_eq!(clean_speaker("Anna 1 minute"), "Anna");
        assert_eq!(clean_speaker("Tomasz Chmiel"), "Tomasz Chmiel", "no suffix untouched");
        assert_eq!(clean_speaker("Jan 3 Kowalski"), "Jan 3 Kowalski", "digit not at the tail is left alone");
        let line = |sp: Option<&str>| TranscriptLine { meeting_id: "m".into(), idx: 0, speaker: sp.map(str::to_string), offset_s: None, ts: 0, text: "x".into() };
        let lines = [line(None), line(Some("Anna Wierzbicka")), line(Some("Anna Wierzbicka 20 minutes")), line(Some("Tomasz Chmiel"))];
        assert_eq!(participants(&lines), ["Anna Wierzbicka", "Tomasz Chmiel"], "system line skipped, duration-variant merged, sorted");
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

    fn standup(summary: Option<&str>, summary_status: &str) -> Meeting {
        Meeting {
            id: "m1".into(),
            chat: "c".into(),
            title: "Standup".into(),
            occurrence: "o".into(),
            started_at: 1,
            ended_at: 2,
            summary: summary.map(str::to_string),
            summary_status: summary_status.into(),
            transcript_status: "ok".into(),
        }
    }

    #[test]
    fn summary_counts_when_the_exporter_marks_it_ok_or_the_legacy_available() {
        assert!(standup(Some("Ustalono deploy"), "ok").has_summary());
        assert!(standup(Some("Ustalono deploy"), "available").has_summary());
        assert!(!standup(Some("Ustalono deploy"), "unavailable").has_summary());
        assert!(!standup(Some("   "), "ok").has_summary());
        assert!(!standup(None, "ok").has_summary());
    }

    #[test]
    fn fingerprint_changes_when_a_summary_arrives_after_the_transcript() {
        let line = TranscriptLine { meeting_id: "m1".into(), idx: 0, speaker: Some("A".into()), offset_s: Some(0), ts: 1, text: "x".into() };
        let before = standup(None, "unavailable").fingerprint(std::slice::from_ref(&line));
        let after = standup(Some("Ustalono deploy"), "ok").fingerprint(std::slice::from_ref(&line));
        assert_ne!(before, after);
        assert_eq!(after, standup(Some("Ustalono deploy"), "ok").fingerprint(std::slice::from_ref(&line)), "stable once summarised");
    }
}
