//! `ctx`: local memory over agent sessions — embedding daemon, indexer CLI and MCP server.

pub mod config;
pub mod embed_server;
pub mod hook;
pub mod mcp;
pub mod queue;
pub mod search;
pub mod summarizer;

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

pub fn format_report(r: &Report, elapsed: std::time::Duration) -> String {
    format!(
        "scanned {} · indexed {} · unchanged {} · empty {} · failed {} · {} chunks · {} secrets redacted · {:.1?}",
        r.scanned,
        r.indexed,
        r.unchanged,
        r.empty,
        r.failed.len(),
        r.chunks,
        r.redactions,
        elapsed
    )
}

/// One index run inside the daemon, embedding in-process through the shared model.
///
/// With a summarizer, each run then summarises sessions that are new or have grown, within the
/// summary budget; the rest wait for the next run.
pub fn daemon_runner(
    cfg: config::Config,
    store: std::sync::Arc<Store>,
    counter: std::sync::Arc<dyn TokenCounter + Send + Sync>,
    embedder: embed_server::SharedEmbedder,
    model: String,
    summarizer: Option<std::sync::Arc<dyn ingest::Summarizer>>,
) -> queue::Runner {
    std::sync::Arc::new(move |target: queue::Target| {
        let (cfg, store, counter, mut embedder, model) = (cfg.clone(), store.clone(), counter.clone(), embedder.clone(), model.clone());
        let summarizer = summarizer.clone();
        Box::pin(async move {
            let handle = tokio::runtime::Handle::current();
            // The indexer embeds synchronously; run it on a blocking thread so HTTP handlers
            // (search queries) keep being served in between batches.
            tokio::task::spawn_blocking(move || {
                handle.block_on(async move {
                    let t = std::time::Instant::now();
                    let refs = discover(&cfg, target.sources())?;
                    let state = State::open(&cfg.state_path)?;
                    let report = run_index(&store, &mut embedder, &model, counter.as_ref(), &state, &refs, |_| {}).await?;
                    let mut line = format_report(&report, t.elapsed());
                    for (id, error) in &report.failed {
                        eprintln!("[index {target:?}]   FAILED {id}: {error}");
                    }
                    if let Some(s) = summarizer {
                        let now = jiff::Timestamp::now().as_millisecond();
                        let r = ingest::summarize_pending(&store, &mut embedder, s.as_ref(), &state, &ingest::SummaryConfig::default(), now).await?;
                        line.push_str(&format!(" · summaries {} written, {} failed, {} deferred", r.written, r.failed.len(), r.deferred));
                        for (id, error) in &r.failed {
                            eprintln!("[summary]   FAILED {id}: {error}");
                        }
                    }
                    eprintln!("[index {target:?}] {line}");
                    Ok(line)
                })
            })
            .await?
        })
    })
}
