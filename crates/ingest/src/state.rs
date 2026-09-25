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
             );",
        )?;
        Ok(Self { conn })
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

    pub fn record(
        &self,
        source: &str,
        session_id: &str,
        fingerprint: &str,
        project: &str,
        chunks: usize,
        redactions: usize,
        now_ms: i64,
    ) -> rusqlite::Result<()> {
        self.conn.execute(
            "insert into indexed (source, session_id, fingerprint, project, chunks, redactions, indexed_at)
             values (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             on conflict(source, session_id) do update set
                 fingerprint = excluded.fingerprint, project = excluded.project, chunks = excluded.chunks,
                 redactions = excluded.redactions, indexed_at = excluded.indexed_at",
            params![source, session_id, fingerprint, project, chunks as i64, redactions as i64, now_ms],
        )?;
        Ok(())
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

    #[test]
    fn record_then_get_and_overwrite() {
        let s = State::in_memory().unwrap();
        s.record("claude", "s1", "fp1", "p", 3, 0, 1).unwrap();
        assert_eq!(s.get("claude", "s1").unwrap(), Some(Indexed { fingerprint: "fp1".into(), chunks: 3 }));
        s.record("claude", "s1", "fp2", "p", 5, 1, 2).unwrap();
        assert_eq!(s.get("claude", "s1").unwrap().unwrap().fingerprint, "fp2");
        assert_eq!(s.get("opencode", "s1").unwrap(), None, "sources are separate namespaces");
    }

    #[test]
    fn first_version_is_not_a_reset_but_a_change_is() {
        let s = State::in_memory().unwrap();
        assert!(!s.ensure_version("v1").unwrap());
        s.record("claude", "s1", "fp", "p", 1, 0, 0).unwrap();
        assert!(!s.ensure_version("v1").unwrap(), "same version keeps state");
        assert!(s.get("claude", "s1").unwrap().is_some());
        assert!(s.ensure_version("v2").unwrap());
        assert!(s.get("claude", "s1").unwrap().is_none(), "version change forgets sessions");
    }

    #[test]
    fn per_project_aggregates_sessions_and_chunks() {
        let s = State::in_memory().unwrap();
        s.record("claude", "a", "f", "p1", 3, 0, 0).unwrap();
        s.record("opencode", "b", "f", "p1", 2, 0, 0).unwrap();
        s.record("claude", "c", "f", "p2", 7, 0, 0).unwrap();
        assert_eq!(s.per_project().unwrap(), [("p1".into(), 2, 5), ("p2".into(), 1, 7)]);
    }

    #[test]
    fn state_persists_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested/state.db");
        State::open(&path).unwrap().record("claude", "s", "fp", "p", 1, 0, 0).unwrap();
        assert!(State::open(&path).unwrap().get("claude", "s").unwrap().is_some());
    }
}
