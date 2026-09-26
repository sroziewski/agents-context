//! What has been indexed, so a sync only re-embeds sessions whose fingerprint changed.

use std::path::Path;

use rusqlite::{Connection, OptionalExtension, params};

pub struct State {
    conn: Connection,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Indexed {
    pub fingerprint: String,
    pub chunks: usize,
}

/// One indexed session as recorded after a sync.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Record<'a> {
    pub source: &'a str,
    pub session_id: &'a str,
    pub fingerprint: &'a str,
    pub project: &'a str,
    pub chunks: usize,
    pub redactions: usize,
    pub turns: usize,
    pub subagent: bool,
}

/// A session due for a (re-)summary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingSummary {
    pub source: String,
    pub session_id: String,
    pub turns: usize,
}

impl State {
    pub fn open(path: &Path) -> rusqlite::Result<Self> {
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        Self::init(Connection::open(path)?)
    }

    pub fn in_memory() -> rusqlite::Result<Self> {
        Self::init(Connection::open_in_memory()?)
    }

    fn init(conn: Connection) -> rusqlite::Result<Self> {
        conn.execute_batch(
            "pragma journal_mode = wal;
             create table if not exists meta (key text primary key, value text not null);
             create table if not exists indexed (
                 source text not null,
                 session_id text not null,
                 fingerprint text not null,
                 project text not null,
                 chunks integer not null,
                 redactions integer not null,
                 indexed_at integer not null,
                 primary key (source, session_id)
             );
             create table if not exists summaries (
                 source text not null,
                 session_id text not null,
                 turns integer not null,          -- turn count at the last successful summary
                 summarized_at integer not null,
                 attempts integer not null default 0,     -- consecutive failures since then
                 failed_turns integer not null default 0, -- turn count at the last failure
                 primary key (source, session_id)
             );",
        )?;
        let state = Self { conn };
        state.migrate()?;
        Ok(state)
    }

    /// Adds columns introduced after the first release. Rows written before they existed get
    /// their fingerprint cleared, so the next sync re-reads those sessions and fills them in;
    /// unchanged chunks keep their vectors, so that costs parsing, not embedding.
    ///
    /// One immediate transaction: a crash between adding the columns and clearing fingerprints
    /// would otherwise leave the columns in place and the re-read never happening, and the CLI
    /// and the daemon opening the DB at once would both try to add the same column.
    fn migrate(&self) -> rusqlite::Result<()> {
        self.conn.execute_batch("begin immediate")?;
        let result = (|| {
            let mut added = false;
            for (column, ddl) in [
                ("turns", "alter table indexed add column turns integer not null default 0"),
                ("subagent", "alter table indexed add column subagent integer not null default 0"),
            ] {
                if !self.has_column("indexed", column)? {
                    self.conn.execute(ddl, [])?;
                    added = true;
                }
            }
            if added {
                self.conn.execute("update indexed set fingerprint = ''", [])?;
            }
            Ok(())
        })();
        match result {
            Ok(()) => self.conn.execute_batch("commit"),
            Err(e) => {
                let _ = self.conn.execute_batch("rollback");
                Err(e)
            }
        }
    }

    fn has_column(&self, table: &str, column: &str) -> rusqlite::Result<bool> {
        let mut stmt = self.conn.prepare(&format!("pragma table_info({table})"))?;
        let names = stmt.query_map([], |r| r.get::<_, String>(1))?.collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(names.iter().any(|n| n == column))
    }

    /// Records `version` (chunker + model identity). If it differs from the stored one, every
    /// session is forgotten so the next sync re-embeds all of them; returns whether that happened.
    /// Vectors from different models or chunkers are not comparable, so they must not mix.
    pub fn ensure_version(&self, version: &str) -> rusqlite::Result<bool> {
        let current: Option<String> = self
            .conn
            .query_row("select value from meta where key = 'version'", [], |r| r.get(0))
            .optional()?;
        if current.as_deref() == Some(version) {
            return Ok(false);
        }
        self.conn.execute("delete from indexed", [])?;
        self.conn.execute("delete from summaries", [])?;
        self.conn.execute(
            "insert into meta (key, value) values ('version', ?1)
             on conflict(key) do update set value = excluded.value",
            params![version],
        )?;
        Ok(current.is_some())
    }

    pub fn get(&self, source: &str, session_id: &str) -> rusqlite::Result<Option<Indexed>> {
        self.conn
            .query_row(
                "select fingerprint, chunks from indexed where source = ?1 and session_id = ?2",
                params![source, session_id],
                |r| Ok(Indexed { fingerprint: r.get(0)?, chunks: r.get::<_, i64>(1)? as usize }),
            )
            .optional()
    }

    pub fn record(&self, r: &Record<'_>, now_ms: i64) -> rusqlite::Result<()> {
        self.conn.execute(
            "insert into indexed (source, session_id, fingerprint, project, chunks, redactions, indexed_at, turns, subagent)
             values (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
             on conflict(source, session_id) do update set
                 fingerprint = excluded.fingerprint, project = excluded.project, chunks = excluded.chunks,
                 redactions = excluded.redactions, indexed_at = excluded.indexed_at,
                 turns = excluded.turns, subagent = excluded.subagent",
            params![
                r.source,
                r.session_id,
                r.fingerprint,
                r.project,
                r.chunks as i64,
                r.redactions as i64,
                now_ms,
                r.turns as i64,
                r.subagent
            ],
        )?;
        Ok(())
    }

    /// Main sessions with at least `min_turns` turns that need a summary: none yet, grown by
    /// `regrow` turns since the last one, or failed fewer than `max_attempts` times in a row.
    /// A session that keeps failing is set aside until it grows by `regrow` turns again, so a
    /// transcript the model cannot handle does not take every run's budget forever. Sessions
    /// without failures come first, then oldest-changed first.
    pub fn pending_summaries(
        &self,
        min_turns: usize,
        regrow: usize,
        max_attempts: usize,
        limit: usize,
    ) -> rusqlite::Result<Vec<PendingSummary>> {
        let mut stmt = self.conn.prepare(
            "select i.source, i.session_id, i.turns from indexed i
             left join summaries s on s.source = i.source and s.session_id = i.session_id
             where i.subagent = 0 and i.project != '' and i.turns >= ?1
               and (s.session_id is null
                    or (s.attempts = 0 and i.turns - s.turns >= ?2)
                    or (s.attempts > 0 and s.attempts < ?3)
                    or (s.attempts >= ?3 and i.turns - s.failed_turns >= ?2))
             order by coalesce(s.attempts, 0), i.indexed_at, i.session_id
             limit ?4",
        )?;
        let rows = stmt.query_map(params![min_turns as i64, regrow as i64, max_attempts as i64, limit as i64], |r| {
            Ok(PendingSummary { source: r.get(0)?, session_id: r.get(1)?, turns: r.get::<_, i64>(2)? as usize })
        })?;
        rows.collect()
    }

    pub fn record_summary(&self, source: &str, session_id: &str, turns: usize, now_ms: i64) -> rusqlite::Result<()> {
        self.conn.execute(
            "insert into summaries (source, session_id, turns, summarized_at, attempts, failed_turns) values (?1, ?2, ?3, ?4, 0, 0)
             on conflict(source, session_id) do update set
                 turns = excluded.turns, summarized_at = excluded.summarized_at, attempts = 0, failed_turns = 0",
            params![source, session_id, turns as i64, now_ms],
        )?;
        Ok(())
    }

    /// Counts a failed summary attempt; an earlier successful summary (and its turn count) is kept.
    pub fn record_summary_failure(&self, source: &str, session_id: &str, turns: usize) -> rusqlite::Result<()> {
        self.conn.execute(
            "insert into summaries (source, session_id, turns, summarized_at, attempts, failed_turns) values (?1, ?2, 0, 0, 1, ?3)
             on conflict(source, session_id) do update set attempts = attempts + 1, failed_turns = excluded.failed_turns",
            params![source, session_id, turns as i64],
        )?;
        Ok(())
    }

    /// Sessions with chunks but an unknown turn count: indexed before turn counts were recorded
    /// and not re-readable because their transcript was cleaned up since.
    pub fn sessions_missing_turns(&self) -> rusqlite::Result<Vec<(String, String)>> {
        let mut stmt = self.conn.prepare("select source, session_id from indexed where turns = 0 and chunks > 0")?;
        let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
        rows.collect()
    }

    pub fn set_turns(&self, source: &str, session_id: &str, turns: usize, subagent: bool) -> rusqlite::Result<()> {
        self.conn.execute(
            "update indexed set turns = ?3, subagent = ?4 where source = ?1 and session_id = ?2",
            params![source, session_id, turns as i64, subagent],
        )?;
        Ok(())
    }

    /// Makes the next sync re-read every session from its source, e.g. after redaction rules
    /// changed. Unchanged chunks keep their vectors, so this costs parsing, not embedding.
    pub fn force_reread(&self) -> rusqlite::Result<usize> {
        self.conn.execute("update indexed set fingerprint = ''", [])
    }

    pub fn summary_count(&self) -> rusqlite::Result<usize> {
        self.conn.query_row("select count(*) from summaries", [], |r| r.get::<_, i64>(0)).map(|n| n as usize)
    }

    /// `(project, sessions, chunks)` per project, for `stats`.
    pub fn per_project(&self) -> rusqlite::Result<Vec<(String, usize, usize)>> {
        let mut stmt = self.conn.prepare(
            "select project, count(*), sum(chunks) from indexed group by project order by project",
        )?;
        let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get::<_, i64>(1)? as usize, r.get::<_, i64>(2)? as usize)))?;
        rows.collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_session_has_no_state() {
        assert_eq!(State::in_memory().unwrap().get("claude", "x").unwrap(), None);
    }

    fn rec<'a>(source: &'a str, id: &'a str, fp: &'a str, project: &'a str, chunks: usize) -> Record<'a> {
        Record { source, session_id: id, fingerprint: fp, project, chunks, ..Default::default() }
    }

    #[test]
    fn record_then_get_and_overwrite() {
        let s = State::in_memory().unwrap();
        s.record(&rec("claude", "s1", "fp1", "p", 3), 1).unwrap();
        assert_eq!(s.get("claude", "s1").unwrap(), Some(Indexed { fingerprint: "fp1".into(), chunks: 3 }));
        s.record(&rec("claude", "s1", "fp2", "p", 5), 2).unwrap();
        assert_eq!(s.get("claude", "s1").unwrap().unwrap().fingerprint, "fp2");
        assert_eq!(s.get("opencode", "s1").unwrap(), None, "sources are separate namespaces");
    }

    #[test]
    fn first_version_is_not_a_reset_but_a_change_is() {
        let s = State::in_memory().unwrap();
        assert!(!s.ensure_version("v1").unwrap());
        s.record(&rec("claude", "s1", "fp", "p", 1), 0).unwrap();
        s.record_summary("claude", "s1", 3, 0).unwrap();
        assert!(!s.ensure_version("v1").unwrap(), "same version keeps state");
        assert!(s.get("claude", "s1").unwrap().is_some());
        assert!(s.ensure_version("v2").unwrap());
        assert!(s.get("claude", "s1").unwrap().is_none(), "version change forgets sessions");
        assert_eq!(s.summary_count().unwrap(), 0, "and their summaries");
    }

    #[test]
    fn per_project_aggregates_sessions_and_chunks() {
        let s = State::in_memory().unwrap();
        s.record(&rec("claude", "a", "f", "p1", 3), 0).unwrap();
        s.record(&rec("opencode", "b", "f", "p1", 2), 0).unwrap();
        s.record(&rec("claude", "c", "f", "p2", 7), 0).unwrap();
        assert_eq!(s.per_project().unwrap(), [("p1".into(), 2, 5), ("p2".into(), 1, 7)]);
    }

    #[test]
    fn state_persists_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested/state.db");
        State::open(&path).unwrap().record(&rec("claude", "s", "fp", "p", 1), 0).unwrap();
        assert!(State::open(&path).unwrap().get("claude", "s").unwrap().is_some());
    }

    fn session(s: &State, id: &str, turns: usize, subagent: bool, at: i64) {
        let r = Record { source: "claude", session_id: id, fingerprint: "f", project: "p", turns, subagent, ..Default::default() };
        s.record(&r, at).unwrap();
    }

    #[test]
    fn pending_summaries_select_main_sessions_with_enough_turns() {
        let s = State::in_memory().unwrap();
        session(&s, "one-turn", 1, false, 1);
        session(&s, "sub", 9, true, 2);
        session(&s, "main-b", 3, false, 4);
        session(&s, "main-a", 2, false, 3);
        s.record(&Record { source: "claude", session_id: "empty", fingerprint: "f", project: "", turns: 5, ..Default::default() }, 5).unwrap();
        let ids: Vec<_> = s.pending_summaries(2, 5, 3, 10).unwrap().into_iter().map(|p| p.session_id).collect();
        assert_eq!(ids, ["main-a", "main-b"], "oldest first; subagents, 1-turn and empty sessions skipped");
    }

    #[test]
    fn summarised_session_is_pending_again_only_after_regrowing() {
        let s = State::in_memory().unwrap();
        session(&s, "m", 4, false, 1);
        s.record_summary("claude", "m", 4, 1).unwrap();
        assert!(s.pending_summaries(2, 5, 3, 10).unwrap().is_empty());
        session(&s, "m", 8, false, 2);
        assert!(s.pending_summaries(2, 5, 3, 10).unwrap().is_empty(), "4 new turns < 5");
        session(&s, "m", 9, false, 3);
        assert_eq!(s.pending_summaries(2, 5, 3, 10).unwrap()[0].turns, 9);
    }

    #[test]
    fn pending_summaries_respects_limit() {
        let s = State::in_memory().unwrap();
        for i in 0..5 {
            session(&s, &format!("m{i}"), 3, false, i);
        }
        assert_eq!(s.pending_summaries(2, 5, 3, 2).unwrap().len(), 2);
    }

    #[test]
    fn failing_session_is_retried_then_set_aside_until_it_grows() {
        let s = State::in_memory().unwrap();
        session(&s, "bad", 3, false, 1);
        session(&s, "good", 3, false, 2);
        for attempt in 1..=3 {
            let first = &s.pending_summaries(2, 5, 3, 10).unwrap()[0].session_id;
            if attempt == 1 {
                assert_eq!(first, "bad", "oldest first while nothing has failed");
            } else {
                assert_eq!(first, "good", "a failing session sorts after fresh ones");
            }
            s.record_summary_failure("claude", "bad", 3).unwrap();
        }
        let ids: Vec<_> = s.pending_summaries(2, 5, 3, 10).unwrap().into_iter().map(|p| p.session_id).collect();
        assert_eq!(ids, ["good"], "after 3 failures it is set aside");
        session(&s, "bad", 8, false, 3);
        let ids: Vec<_> = s.pending_summaries(2, 5, 3, 10).unwrap().into_iter().map(|p| p.session_id).collect();
        assert!(ids.contains(&"bad".to_owned()), "5 more turns give it another chance");
    }

    #[test]
    fn failure_after_a_success_keeps_the_successful_turn_count() {
        let s = State::in_memory().unwrap();
        session(&s, "m", 4, false, 1);
        s.record_summary("claude", "m", 4, 1).unwrap();
        session(&s, "m", 9, false, 2);
        s.record_summary_failure("claude", "m", 9).unwrap();
        assert_eq!(s.pending_summaries(2, 5, 3, 10).unwrap().len(), 1, "retried while under the attempt limit");
        s.record_summary("claude", "m", 9, 3).unwrap();
        assert!(s.pending_summaries(2, 5, 3, 10).unwrap().is_empty(), "success resets attempts");
    }

    #[test]
    fn sessions_missing_turns_can_be_backfilled() {
        let s = State::in_memory().unwrap();
        s.record(&Record { source: "claude", session_id: "gone", fingerprint: "", project: "p", chunks: 5, ..Default::default() }, 0).unwrap();
        s.record(&Record { source: "claude", session_id: "empty", fingerprint: "", project: "", ..Default::default() }, 0).unwrap();
        assert_eq!(s.sessions_missing_turns().unwrap(), [("claude".to_owned(), "gone".to_owned())]);
        s.set_turns("claude", "gone", 4, false).unwrap();
        assert!(s.sessions_missing_turns().unwrap().is_empty());
        assert_eq!(s.pending_summaries(2, 5, 3, 10).unwrap()[0].turns, 4);
    }

    #[test]
    fn force_reread_clears_every_fingerprint() {
        let s = State::in_memory().unwrap();
        session(&s, "a", 2, false, 1);
        session(&s, "b", 2, false, 1);
        assert_eq!(s.force_reread().unwrap(), 2);
        assert_eq!(s.get("claude", "a").unwrap().unwrap().fingerprint, "");
    }

    #[test]
    fn migration_adds_columns_and_forces_a_reread() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("old.db");
        {
            // Schema as shipped before turns/subagent existed.
            let c = Connection::open(&path).unwrap();
            c.execute_batch(
                "create table indexed (source text not null, session_id text not null, fingerprint text not null,
                     project text not null, chunks integer not null, redactions integer not null,
                     indexed_at integer not null, primary key (source, session_id));
                 insert into indexed values ('claude', 's', 'fp-old', 'p', 3, 0, 0);",
            )
            .unwrap();
        }
        let s = State::open(&path).unwrap();
        assert_eq!(s.get("claude", "s").unwrap().unwrap().fingerprint, "", "cleared so the next sync re-reads it");
        session(&s, "s", 4, false, 1);
        drop(s);
        let again = State::open(&path).unwrap();
        assert_eq!(again.get("claude", "s").unwrap().unwrap().fingerprint, "f", "a second open does not clear again");
    }
}
