//! Readers for agent session stores, normalised into [`Session`]s of prompt/response [`Turn`]s.
//!
//! Only conversation substance is kept: user prompts, assistant prose and a one-line summary per
//! tool call. Tool output, thinking and harness chatter are dropped here, before anything is
//! embedded — they are ~90% of the bytes and almost all of the noise.

pub mod claude;
pub mod jira;
pub mod opencode;
pub mod project;
pub mod teams;
mod tools;

use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    Claude,
    OpenCode,
}

impl Source {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Claude => "claude",
            Self::OpenCode => "opencode",
        }
    }
}

/// Cheap handle to a session: enough to decide whether it changed since the last index run
/// without parsing it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionRef {
    pub source: Source,
    pub id: String,
    /// Changes whenever the session content changes (file size + mtime, or `time_updated`).
    pub fingerprint: String,
    pub locator: Locator,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Locator {
    File(PathBuf),
    OpenCode { db: PathBuf },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Session {
    pub source: Source,
    pub id: String,
    /// Set for subagent sessions: the session that spawned them.
    pub parent_id: Option<String>,
    /// Stable project identity, see [`project::key`].
    pub project: String,
    pub cwd: String,
    pub title: Option<String>,
    pub branch: Option<String>,
    pub started_at: i64,
    pub updated_at: i64,
    pub turns: Vec<Turn>,
}

impl Session {
    pub fn is_subagent(&self) -> bool {
        self.parent_id.is_some()
    }
}

/// One user prompt and everything the agent said and did until the next prompt.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Turn {
    /// Id of the prompt message in the source store; unique within the session.
    pub id: String,
    pub index: usize,
    /// Epoch milliseconds of the prompt.
    pub ts: i64,
    pub user: String,
    pub assistant: String,
    /// One line per tool call, e.g. `Edit src/db.ts`.
    pub tools: Vec<String>,
    /// Files the agent edited or wrote, in first-touch order.
    pub files: Vec<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum SourceError {
    #[error("{path}: {source}")]
    Io { path: PathBuf, source: std::io::Error },
    #[error("sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("session {0} not found")]
    NotFound(String),
    #[error("session {0} has no usable turns")]
    Empty(String),
}

pub type Result<T> = std::result::Result<T, SourceError>;

pub(crate) fn io_err(path: &std::path::Path) -> impl FnOnce(std::io::Error) -> SourceError + '_ {
    move |source| SourceError::Io { path: path.to_path_buf(), source }
}

pub(crate) fn push_unique(files: &mut Vec<String>, file: String) {
    if !files.contains(&file) {
        files.push(file);
    }
}
