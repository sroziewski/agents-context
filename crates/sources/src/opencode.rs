//! OpenCode sessions from its SQLite store (`~/.local/share/opencode/opencode.db`):
//! `session` → `message` (role in JSON `data`) → `part` (text / tool / reasoning … in JSON `data`).

use std::collections::HashMap;
use std::path::Path;

use rusqlite::{Connection, OpenFlags, params};
use serde_json::Value;

use crate::tools::summarise;
use crate::{Locator, Result, Session, SessionRef, Source, SourceError, Turn, project, push_unique};

/// Read-only: OpenCode may be writing concurrently (WAL), and this tool must never modify it.
fn open(db: &Path) -> Result<Connection> {
    let conn = Connection::open_with_flags(db, OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX)?;
    conn.busy_timeout(std::time::Duration::from_secs(5))?;
    Ok(conn)
}

/// Title prefix of the sessions `ctx replay` creates. They replay prompts that are already
/// indexed from Claude Code, so indexing them, or the subagent sessions under them, would only
/// duplicate those as "past sessions".
pub const REPLAY_TITLE_PREFIX: &str = "ctx-replay · ";

pub fn discover(db: &Path) -> Result<Vec<SessionRef>> {
    let conn = open(db)?;
    // session.time_updated alone can lag behind streamed parts, so the part count and newest
    // part time are folded into the fingerprint too.
    let mut stmt = conn.prepare(
        "with recursive replay(id) as (
             select id from session where substr(title, 1, length(?1)) = ?1
             union select s.id from session s join replay r on s.parent_id = r.id
         )
         select s.id, s.time_updated,
                (select count(*) from part p where p.session_id = s.id),
                (select coalesce(max(p.time_updated), 0) from part p where p.session_id = s.id)
         from session s
         where s.id not in (select id from replay)
         order by s.id",
    )?;
    let rows = stmt.query_map([REPLAY_TITLE_PREFIX], |r| {
        Ok(SessionRef {
            source: Source::OpenCode,
            id: r.get(0)?,
            fingerprint: format!("{}-{}-{}", r.get::<_, i64>(1)?, r.get::<_, i64>(2)?, r.get::<_, i64>(3)?),
            locator: Locator::OpenCode { db: db.to_path_buf() },
        })
    })?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

pub fn load(r: &SessionRef) -> Result<Session> {
    let Locator::OpenCode { db } = &r.locator else {
        return Err(SourceError::NotFound(r.id.clone()));
    };
    load_from(&open(db)?, &r.id)
}

pub fn load_from(conn: &Connection, id: &str) -> Result<Session> {
    let (parent_id, directory, title, created, updated) = conn
        .query_row(
            "select parent_id, directory, title, time_created, time_updated from session where id = ?1",
            params![id],
            |r| Ok((r.get::<_, Option<String>>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?, r.get::<_, i64>(3)?, r.get::<_, i64>(4)?)),
        )
        .map_err(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => SourceError::NotFound(id.to_owned()),
            e => e.into(),
        })?;

    let mut parts: HashMap<String, Vec<Value>> = HashMap::new();
    let mut stmt = conn.prepare("select message_id, data from part where session_id = ?1 order by time_created, id")?;
    for row in stmt.query_map(params![id], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))? {
        let (message_id, data) = row?;
        if let Ok(v) = serde_json::from_str(&data) {
            parts.entry(message_id).or_default().push(v);
        }
    }

    let mut turns: Vec<Turn> = Vec::new();
    let mut current: Option<Turn> = None;
    let mut stmt = conn.prepare("select id, time_created, data from message where session_id = ?1 order by time_created, id")?;
    for row in stmt.query_map(params![id], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?, r.get::<_, String>(2)?)))? {
        let (message_id, ts, data) = row?;
        let Ok(message) = serde_json::from_str::<Value>(&data) else { continue };
        let message_parts = parts.get(&message_id).map(Vec::as_slice).unwrap_or_default();
        match message.get("role").and_then(Value::as_str) {
            Some("user") => {
                let text = user_text(message_parts);
                if !text.is_empty() {
                    turns.extend(current.take());
                    current = Some(Turn { id: message_id, index: turns.len(), ts, user: text, ..Default::default() });
                }
            }
            Some("assistant") => {
                if let Some(turn) = current.as_mut() {
                    add_assistant(turn, message_parts);
                }
            }
            _ => {}
        }
    }
    turns.extend(current);
    if turns.is_empty() {
        return Err(SourceError::Empty(id.to_owned()));
    }
    Ok(Session {
        source: Source::OpenCode,
        id: id.to_owned(),
        parent_id,
        project: project::key(&directory),
        cwd: directory,
        title: Some(title).filter(|t| !t.is_empty()),
        branch: None,
        started_at: created,
        updated_at: updated,
        turns,
    })
}

/// Text the person typed. `synthetic` parts are OpenCode's own nudges ("Continue if you have
/// next steps…"), not the user.
fn user_text(parts: &[Value]) -> String {
    parts
        .iter()
        .filter(|p| p.get("type").and_then(Value::as_str) == Some("text"))
        .filter(|p| !p.get("synthetic").and_then(Value::as_bool).unwrap_or(false))
        .filter_map(|p| p.get("text").and_then(Value::as_str).map(str::trim))
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn add_assistant(turn: &mut Turn, parts: &[Value]) {
    for p in parts {
        match p.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(t) = p.get("text").and_then(Value::as_str).map(str::trim).filter(|t| !t.is_empty()) {
                    if !turn.assistant.is_empty() {
                        turn.assistant.push_str("\n\n");
                    }
                    turn.assistant.push_str(t);
                }
            }
            Some("tool") => {
                let name = p.get("tool").and_then(Value::as_str).unwrap_or("tool");
                let call = summarise(name, p.pointer("/state/input").unwrap_or(&Value::Null));
                turn.tools.push(call.summary);
                if let Some(f) = call.edited_file {
                    push_unique(&mut turn.files, f);
                }
            }
            _ => {} // reasoning, step markers, compaction
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The subset of OpenCode's schema this reader touches.
    fn fixture() -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("opencode.db");
        let c = Connection::open(&path).unwrap();
        c.execute_batch(
            "create table session (id text primary key, project_id text not null, parent_id text,
                 slug text not null default '', directory text not null, title text not null,
                 version text not null default '1', time_created integer not null, time_updated integer not null);
             create table message (id text primary key, session_id text not null,
                 time_created integer not null, time_updated integer not null, data text not null);
             create table part (id text primary key, message_id text not null, session_id text not null,
                 time_created integer not null, time_updated integer not null, data text not null);",
        )
        .unwrap();
        (dir, path)
    }

    struct Db(Connection);

    impl Db {
        fn session(&self, id: &str, parent: Option<&str>, dir: &str, title: &str) -> &Self {
            self.0
                .execute(
                    "insert into session (id, project_id, parent_id, directory, title, time_created, time_updated)
                     values (?1, 'global', ?2, ?3, ?4, 1000, 2000)",
                    params![id, parent, dir, title],
                )
                .unwrap();
            self
        }

        fn message(&self, sid: &str, id: &str, ts: i64, role: &str, parts: &[Value]) -> &Self {
            let data = json!({"role": role, "time": {"created": ts}}).to_string();
            self.0
                .execute("insert into message values (?1, ?2, ?3, ?3, ?4)", params![id, sid, ts, data])
                .unwrap();
            for (i, p) in parts.iter().enumerate() {
                self.0
                    .execute(
                        "insert into part values (?1, ?2, ?3, ?4, ?4, ?5)",
                        params![format!("{id}-p{i}"), id, sid, ts + i as i64, p.to_string()],
                    )
                    .unwrap();
            }
            self
        }
    }

    fn text(t: &str) -> Value {
        json!({"type": "text", "text": t})
    }

    #[test]
    fn builds_turns_from_user_and_assistant_messages() {
        let (_d, path) = fixture();
        let db = Db(Connection::open(&path).unwrap());
        db.session("ses1", None, "Z:/no/such/Sedn26", "Requirements question")
            .message("ses1", "m1", 100, "user", &[text("Czy requireNonBlank dotyczy K1?")])
            .message("ses1", "m2", 110, "assistant", &[
                json!({"type": "step-start"}),
                json!({"type": "reasoning", "text": "private"}),
                json!({"type": "tool", "tool": "read", "state": {"input": {"filePath": "K1.java"}, "output": "huge"}}),
                text("Tak, dotyczy K1."),
            ])
            .message("ses1", "m3", 120, "assistant", &[
                json!({"type": "tool", "tool": "edit", "state": {"input": {"filePath": "K1.java"}}}),
                text("Poprawione."),
            ])
            .message("ses1", "m4", 200, "user", &[text("a K2?")]);

        let s = load_from(&db.0, "ses1").unwrap();
        assert_eq!(s.source, Source::OpenCode);
        assert_eq!(s.project, "z:/no/such/sedn26");
        assert_eq!(s.title.as_deref(), Some("Requirements question"));
        assert_eq!((s.started_at, s.updated_at), (1000, 2000));
        assert_eq!(s.turns.len(), 2);
        let t = &s.turns[0];
        assert_eq!((t.id.as_str(), t.ts, t.index), ("m1", 100, 0));
        assert_eq!(t.assistant, "Tak, dotyczy K1.\n\nPoprawione.");
        assert_eq!(t.tools, ["read K1.java", "edit K1.java"]);
        assert_eq!(t.files, ["K1.java"]);
        assert!(!t.assistant.contains("private"));
        assert_eq!(s.turns[1].user, "a K2?");
    }

    #[test]
    fn synthetic_nudges_do_not_start_turns() {
        let (_d, path) = fixture();
        let db = Db(Connection::open(&path).unwrap());
        db.session("ses1", None, "Z:/p", "t")
            .message("ses1", "m1", 100, "user", &[text("real")])
            .message("ses1", "m2", 110, "assistant", &[text("one")])
            .message("ses1", "m3", 120, "user", &[json!({"type": "text", "text": "Continue if you have next steps", "synthetic": true})])
            .message("ses1", "m4", 130, "assistant", &[text("two")]);
        let s = load_from(&db.0, "ses1").unwrap();
        assert_eq!(s.turns.len(), 1);
        assert_eq!(s.turns[0].assistant, "one\n\ntwo");
    }

    #[test]
    fn child_session_is_a_subagent() {
        let (_d, path) = fixture();
        let db = Db(Connection::open(&path).unwrap());
        db.session("parent", None, "Z:/p", "p")
            .session("child", Some("parent"), "Z:/p", "")
            .message("child", "m1", 100, "user", &[text("explore")]);
        let s = load_from(&db.0, "child").unwrap();
        assert_eq!(s.parent_id.as_deref(), Some("parent"));
        assert!(s.title.is_none(), "empty title maps to None");
    }

    #[test]
    fn unknown_and_empty_sessions_are_errors() {
        let (_d, path) = fixture();
        let db = Db(Connection::open(&path).unwrap());
        db.session("empty", None, "Z:/p", "t").message("empty", "m1", 1, "assistant", &[text("x")]);
        assert!(matches!(load_from(&db.0, "nope"), Err(SourceError::NotFound(_))));
        assert!(matches!(load_from(&db.0, "empty"), Err(SourceError::Empty(_))));
    }

    #[test]
    fn corrupt_part_json_is_skipped() {
        let (_d, path) = fixture();
        let db = Db(Connection::open(&path).unwrap());
        db.session("s", None, "Z:/p", "t").message("s", "m1", 100, "user", &[text("q")]);
        db.0.execute("insert into part values ('bad', 'm1', 's', 101, 101, '{not json')", []).unwrap();
        assert_eq!(load_from(&db.0, "s").unwrap().turns[0].user, "q");
    }

    #[test]
    fn discover_fingerprint_changes_when_parts_are_added() {
        let (_d, path) = fixture();
        let db = Db(Connection::open(&path).unwrap());
        db.session("s", None, "Z:/p", "t").message("s", "m1", 100, "user", &[text("q")]);
        let before = discover(&path).unwrap();
        assert_eq!(before.len(), 1);
        assert_eq!(before[0].id, "s");
        db.message("s", "m2", 300, "assistant", &[text("a")]);
        let after = discover(&path).unwrap();
        assert_ne!(before[0].fingerprint, after[0].fingerprint);
        assert_eq!(load(&after[0]).unwrap().turns[0].assistant, "a");
    }

    #[test]
    fn discover_skips_replay_sessions_and_all_their_descendants_only() {
        let (_d, path) = fixture();
        let db = Db(Connection::open(&path).unwrap());
        db.session("s1-replay", None, "Z:/p", "ctx-replay · Why does login fail")
            .session("s2-replay-child", Some("s1-replay"), "Z:/p", "explore auth (@explore subagent)")
            .session("s2b-replay-grandchild", Some("s2-replay-child"), "Z:/p", "general (@general subagent)")
            .session("s3-mentions", None, "Z:/p", "Debugging ctx-replay titles")
            .session("s4-plain", None, "Z:/p", "Login bug")
            .session("s5-plain-child", Some("s4-plain"), "Z:/p", "explore")
            .session("s6-orphan-child", Some("gone"), "Z:/p", "")
            .session("s7-prefix-only", None, "Z:/p", "ctx-replay")
            .session("s8-design", None, "Z:/p", "ctx-replay-design notes")
            .session("s8b-design-child", Some("s8-design"), "Z:/p", "explore")
            .session("s9-upper", None, "Z:/p", "CTX-REPLAY · upper case");
        let all = [
            "s1-replay",
            "s2-replay-child",
            "s2b-replay-grandchild",
            "s3-mentions",
            "s4-plain",
            "s5-plain-child",
            "s6-orphan-child",
            "s7-prefix-only",
            "s8-design",
            "s8b-design-child",
            "s9-upper",
        ];
        for id in all {
            db.message(id, &format!("{id}-m1"), 100, "user", &[text("q")]);
        }
        let ids: Vec<String> = discover(&path).unwrap().into_iter().map(|r| r.id).collect();
        assert_eq!(ids, ["s3-mentions", "s4-plain", "s5-plain-child", "s6-orphan-child", "s7-prefix-only", "s8-design", "s8b-design-child", "s9-upper"]);
    }

    #[test]
    fn reader_never_writes_to_the_database() {
        let (_d, path) = fixture();
        Db(Connection::open(&path).unwrap()).session("s", None, "Z:/p", "t").message("s", "m1", 1, "user", &[text("q")]);
        let conn = open(&path).unwrap();
        assert!(conn.execute("delete from session", []).is_err(), "connection must be read-only");
    }
}
