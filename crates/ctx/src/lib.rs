//! `ctx`: local memory over agent sessions — embedding daemon, indexer CLI and MCP server.

pub mod config;
pub mod embed_server;
pub mod mcp;
pub mod search;

use embed::Embedder;
use ingest::{IndexConfig, Indexer, Progress, Report, State, TokenCounter};
use sources::{Session, SessionRef, Source, claude, opencode};
use store::Store;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sources {
    Claude,
    OpenCode,
    All,
}

pub fn discover(cfg: &config::Config, which: Sources) -> anyhow::Result<Vec<SessionRef>> {
    let mut refs = Vec::new();
    if matches!(which, Sources::Claude | Sources::All) && cfg.claude_dir.is_dir() {
        refs.extend(claude::discover(&cfg.claude_dir)?);
    }
    if matches!(which, Sources::OpenCode | Sources::All) && cfg.opencode_db.is_file() {
        refs.extend(opencode::discover(&cfg.opencode_db)?);
    }
    Ok(refs)
}

pub fn load(r: &SessionRef) -> sources::Result<Session> {
    match r.source {
        Source::Claude => claude::load(r),
        Source::OpenCode => opencode::load(r),
    }
}

/// Identity of everything that shapes stored vectors; a change forces a full re-index.
pub fn index_version(model: &str) -> String {
    format!("chunker-{}/{model}", ingest::CHUNKER_VERSION)
}

pub async fn run_index(
    store: &Store,
    embedder: &mut dyn Embedder,
    model: &str,
    counter: &dyn TokenCounter,
    state: &State,
    refs: &[SessionRef],
    mut on_progress: impl FnMut(Progress<'_>),
) -> anyhow::Result<Report> {
    if state.ensure_version(&index_version(model))? {
        eprintln!("model or chunker changed: re-indexing every session");
    }
    store.ensure_collection(ingest::SESSION_FIELDS).await?;
    let mut indexer = Indexer { store, embedder, counter, state, config: IndexConfig::default() };
    let now = jiff::Timestamp::now().as_millisecond();
    Ok(indexer.sync(refs, load, |p| on_progress(p), now).await?)
}
