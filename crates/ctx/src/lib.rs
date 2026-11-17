//! `ctx`: local memory over agent sessions — embedding daemon, indexer CLI and MCP server.

pub mod config;
pub mod confluence;
pub mod embed_server;
pub mod eval;
pub mod eval_identifiers;
pub mod factcheck;
pub mod handoff;
pub mod hook;
pub mod image;
pub mod inject;
pub mod jira;
pub mod lexical;
pub mod mcp;
pub mod prune;
pub mod memories;
pub mod queue;
pub mod rank;
pub mod redact;
pub mod search;
pub mod statusline;
pub mod summarizer;
pub mod teams;
pub mod teams_watch;

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

/// Index settings for writing into `store`: the redaction list, and the `prune older-than` cutoff
/// recorded for that collection.
pub fn index_config(store: &Store, state: &State, redactor: &ingest::Redactor) -> anyhow::Result<IndexConfig> {
    let mut config = IndexConfig::default();
    config.chunk.redactor = redactor.clone();
    config.prune_before_ms = state.get_meta(&prune::cutoff_key(store.collection()))?.and_then(|v| v.parse().ok());
    Ok(config)
}

/// Brings `store` to the current index version: after a model or chunker change every point is
/// re-embedded from its stored text, once per collection. The version a collection was last
/// brought to is kept in the state DB. A collection with none recorded is taken to be current
/// unless the version has changed since this was introduced, so an upgrade does not re-embed.
pub async fn reembed_if_needed(store: &Store, embedder: &mut dyn Embedder, state: &State, version: &str) -> anyhow::Result<()> {
    let key = format!("index_version/{}", store.collection());
    let recorded = state.get_meta(&key)?;
    let current = match recorded.as_deref() {
        Some(v) => v == version,
        None => !state.version_ever_changed()?,
    };
    if !current {
        let n = ingest::reembed_stale(store, embedder, version).await?;
        eprintln!("[{}] re-embedded {n} points for {version}", store.collection());
    }
    state.set_meta(&key, version)?;
    Ok(())
}

pub async fn run_index(
    store: &Store,
    embedder: &mut dyn Embedder,
    model: &str,
    counter: &dyn TokenCounter,
    state: &State,
    refs: &[SessionRef],
    redactor: &ingest::Redactor,
    mut on_progress: impl FnMut(Progress<'_>),
) -> anyhow::Result<Report> {
    let version = index_version(model);
    if state.ensure_version(&version)? {
        eprintln!("model or chunker changed: re-indexing every session");
    }
    store.ensure_collection(ingest::SESSION_FIELDS).await?;
    reembed_if_needed(store, embedder, state, &version).await?;
    let config = index_config(store, state, redactor)?;
    let mut indexer = Indexer { store, embedder, counter, state, config };
    let now = jiff::Timestamp::now().as_millisecond();
    Ok(indexer.sync(refs, load, |p| on_progress(p), now).await?)
}

/// Incremental index of the Teams app's threads into the shared teams collection.
pub async fn run_teams_index(
    cfg: &config::Config,
    store: &Store,
    meetings_store: Option<&Store>,
    embedder: &mut dyn Embedder,
    model: &str,
    counter: &dyn TokenCounter,
    state: &State,
    redactor: &ingest::Redactor,
) -> anyhow::Result<Report> {
    let version = index_version(model);
    if state.ensure_version(&version)? {
        eprintln!("model or chunker changed: re-indexing everything");
    }
    store.ensure_collection(ingest::teams::TEAMS_FIELDS).await?;
    reembed_if_needed(store, embedder, state, &version).await?;
    let db = sources::teams::TeamsDb::open(&cfg.teams_db)?;
    let config = index_config(store, state, redactor)?;
    let now = jiff::Timestamp::now().as_millisecond();
    // Channel threads go to the teams collection; meeting transcripts to their own `meetings`
    // collection so the two never surface each other's chunks. A teams.db from an older scraper
    // without the meetings tables must not fail thread indexing, so the meeting sync is isolated.
    let mut report = ingest::teams::sync_teams(store, embedder, counter, state, &db, &config, now).await?;
    if let Some(meetings) = meetings_store {
        meetings.ensure_collection(ingest::teams::MEETING_FIELDS).await?;
        reembed_if_needed(meetings, embedder, state, &version).await?;
        let config = index_config(meetings, state, redactor)?;
        match ingest::teams::sync_meetings(meetings, embedder, counter, state, &db, &config, now).await {
            Ok(m) => {
                report.scanned += m.scanned;
                report.indexed += m.indexed;
                report.unchanged += m.unchanged;
                report.empty += m.empty;
                report.chunks += m.chunks;
                report.redactions += m.redactions;
                report.failed.extend(m.failed);
            }
            Err(e) => eprintln!("[meetings] FAILED: {e:#}"),
        }
    }
    Ok(report)
}

/// Delta index of Jira issues (the keys mapped in `jira-projects.json`) into the jira collection.
pub async fn run_jira_index(
    cfg: &config::Config,
    store: &Store,
    embedder: &mut dyn Embedder,
    model: &str,
    counter: &dyn TokenCounter,
    state: &State,
    redactor: &ingest::Redactor,
) -> anyhow::Result<Report> {
    let version = index_version(model);
    if state.ensure_version(&version)? {
        eprintln!("model or chunker changed: re-indexing everything");
    }
    let jira = cfg.jira().ok_or_else(|| anyhow::anyhow!("Jira is not configured (JIRA_URL / JIRA_TOKEN)"))?;
    let keys = jira::all_keys(&jira::load_project_keys(&cfg.jira_projects)?);
    anyhow::ensure!(!keys.is_empty(), "no Jira keys mapped in {}", cfg.jira_projects.display());
    store.ensure_collection(ingest::jira::JIRA_FIELDS).await?;
    reembed_if_needed(store, embedder, state, &version).await?;
    let config = index_config(store, state, redactor)?;
    let now = jiff::Timestamp::now().as_millisecond();
    // Opt-in image descriptions: screenshots on issues become searchable text. Off by default.
    // A configured OpenAI-compatible vision endpoint (CTX_VISION_URL, e.g. OPI's GLM) is preferred
    // over the local claude CLI.
    let timeout = std::time::Duration::from_secs(120);
    let describer: Option<Box<dyn ingest::jira::ImageDescriber>> = if !cfg.jira_image_describe {
        None
    } else if let Some(url) = &cfg.vision_url {
        Some(Box::new(crate::image::OpenAiImageDescriber::new(url, &cfg.vision_model, cfg.vision_api_key.clone(), timeout)))
    } else {
        Some(Box::new(crate::image::ClaudeImageDescriber::new(cfg.claude_exe.clone(), &cfg.vision_model, cfg.data_dir.join("vision"), timeout)))
    };
    let opts = describer
        .as_deref()
        .map(|d| ingest::jira::ImageOpts { describer: d, max_images: cfg.jira_image_max, max_bytes: cfg.jira_image_max_bytes });
    Ok(ingest::jira::sync_jira(store, embedder, counter, state, &jira, &keys, &config, opts.as_ref(), now).await?)
}

/// Index of the Confluence spaces mapped in `confluence-projects.json`: pages with a new version
/// only, and pages gone from a space removed.
pub async fn run_confluence_index(
    cfg: &config::Config,
    store: &Store,
    embedder: &mut dyn Embedder,
    model: &str,
    counter: &dyn TokenCounter,
    state: &State,
    redactor: &ingest::Redactor,
) -> anyhow::Result<ingest::confluence::ConfluenceReport> {
    let version = index_version(model);
    if state.ensure_version(&version)? {
        eprintln!("model or chunker changed: re-indexing everything");
    }
    let client = cfg.confluence().ok_or_else(|| anyhow::anyhow!("Confluence is not configured (CONFLUENCE_URL / CONFLUENCE_TOKEN)"))?;
    let units = confluence::all_units(&cfg.confluence_projects)?;
    anyhow::ensure!(!units.is_empty(), "no Confluence spaces or pages mapped in {}", cfg.confluence_projects.display());
    store.ensure_collection(ingest::confluence::CONFLUENCE_FIELDS).await?;
    reembed_if_needed(store, embedder, state, &version).await?;
    let config = index_config(store, state, redactor)?;
    let now = jiff::Timestamp::now().as_millisecond();
    // The client is blocking (ureq); the caller already runs this on a blocking thread.
    Ok(ingest::confluence::sync_confluence(store, embedder, counter, state, &client, &units, &config, now).await?)
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
#[allow(clippy::too_many_arguments)]
pub fn daemon_runner(
    cfg: config::Config,
    store: std::sync::Arc<Store>,
    counter: std::sync::Arc<dyn TokenCounter + Send + Sync>,
    embedder: embed_server::SharedEmbedder,
    model: String,
    summarizer: Option<std::sync::Arc<dyn ingest::Summarizer>>,
    teams_store: Option<std::sync::Arc<Store>>,
    meetings_store: Option<std::sync::Arc<Store>>,
    jira_store: Option<std::sync::Arc<Store>>,
    confluence_store: Option<std::sync::Arc<Store>>,
) -> queue::Runner {
    std::sync::Arc::new(move |target: queue::Target| {
        let (cfg, store, counter, mut embedder, model) = (cfg.clone(), store.clone(), counter.clone(), embedder.clone(), model.clone());
        let summarizer = summarizer.clone();
        let teams_store = teams_store.clone();
        let meetings_store = meetings_store.clone();
        let jira_store = jira_store.clone();
        let confluence_store = confluence_store.clone();
        Box::pin(async move {
            let handle = tokio::runtime::Handle::current();
            // The indexer embeds synchronously; run it on a blocking thread so HTTP handlers
            // (search queries) keep being served in between batches.
            tokio::task::spawn_blocking(move || {
                handle.block_on(async move {
                    let t = std::time::Instant::now();
                    let state = State::open(&cfg.state_path)?;
                    // Re-read every run, so `ctx redact add` applies without a daemon restart.
                    let redactor = ingest::Redactor::from_file(&cfg.redact_file)?;
                    let mut parts = Vec::new();
                    // Isolated like the sources below (a stuck collection migration would
                    // otherwise stop every source), but still the run's error at the end.
                    let mut sessions_error = None;
                    if let Some(which) = target.sessions() {
                        let refs = discover(&cfg, which)?;
                        match run_index(&store, &mut embedder, &model, counter.as_ref(), &state, &refs, &redactor, |_| {}).await {
                            Ok(report) => {
                                for (id, error) in &report.failed {
                                    eprintln!("[index {target:?}]   FAILED {id}: {error}");
                                }
                                parts.push(format_report(&report, t.elapsed()));
                            }
                            Err(e) => {
                                eprintln!("[index {target:?}] sessions FAILED: {e:#}");
                                parts.push(format!("sessions: FAILED ({e})"));
                                sessions_error = Some(e);
                            }
                        }
                    }
                    if let (true, Some(teams)) = (target.includes_teams(), &teams_store) {
                        // Before indexing, notice if the upstream scraper has stopped and, if
                        // configured, re-run it. ctx runs in the user's live session, so it can
                        // launch the sync the Interactive scheduled task keeps failing to start.
                        if cfg.teams_watch.enabled() {
                            match crate::teams_watch::check_and_trigger(&cfg.teams_watch, &cfg.teams_db, &state, &jiff::Zoned::now()) {
                                Ok(crate::teams_watch::Decision::Trigger) => parts.push("teams: scraper stale, re-running".into()),
                                Ok(_) => {}
                                Err(e) => eprintln!("[teams-sync] check failed: {e:#}"),
                            }
                        }
                        if cfg.teams_db.is_file() {
                            // Isolated: a locked or broken teams.db must not stop session
                            // indexing or the summaries below.
                            match run_teams_index(&cfg, teams, meetings_store.as_deref(), &mut embedder, &model, counter.as_ref(), &state, &redactor).await {
                                Ok(r) => {
                                    for (id, error) in &r.failed {
                                        eprintln!("[teams]   FAILED thread {id}: {error}");
                                    }
                                    parts.push(format!(
                                        "teams: {} items (threads+meetings), indexed {}, failed {}, {} chunks",
                                        r.scanned,
                                        r.indexed,
                                        r.failed.len(),
                                        r.chunks
                                    ));
                                }
                                Err(e) => {
                                    eprintln!("[teams] FAILED: {e:#}");
                                    parts.push(format!("teams: FAILED ({e})"));
                                }
                            }
                        }
                    }
                    if let (true, Some(jira)) = (target.includes_jira(), &jira_store) {
                        // Isolated like Teams: a Jira outage must not affect anything else.
                        match run_jira_index(&cfg, jira, &mut embedder, &model, counter.as_ref(), &state, &redactor).await {
                            Ok(r) => {
                                for (id, error) in &r.failed {
                                    eprintln!("[jira]   FAILED {id}: {error}");
                                }
                                parts.push(format!("jira: scanned {} · indexed {} · unchanged {} · removed {} · failed {} · {} chunks", r.scanned, r.indexed, r.unchanged, r.removed, r.failed.len(), r.chunks));
                            }
                            Err(e) => {
                                eprintln!("[jira] FAILED: {e:#}");
                                parts.push(format!("jira: FAILED ({e})"));
                            }
                        }
                    }
                    if let (true, Some(confluence)) = (target.includes_confluence(), &confluence_store) {
                        match run_confluence_index(&cfg, confluence, &mut embedder, &model, counter.as_ref(), &state, &redactor).await {
                            Ok(c) => {
                                for (id, error) in &c.report.failed {
                                    eprintln!("[confluence]   FAILED {id}: {error}");
                                }
                                let r = &c.report;
                                parts.push(format!(
                                    "confluence: scanned {} · indexed {} · unchanged {} · removed {} · failed {} · {} chunks",
                                    r.scanned,
                                    r.indexed,
                                    r.unchanged,
                                    c.removed,
                                    r.failed.len(),
                                    r.chunks
                                ));
                            }
                            Err(e) => {
                                eprintln!("[confluence] FAILED: {e:#}");
                                parts.push(format!("confluence: FAILED ({e})"));
                            }
                        }
                    }
                    let mut line = parts.join(" · ");
                    if let Some(e) = sessions_error {
                        eprintln!("[index {target:?}] {line}");
                        return Err(e.context(line));
                    }
                    if let (Some(s), Some(_)) = (summarizer, target.sessions()) {
                        let now = jiff::Timestamp::now().as_millisecond();
                        let cfg = ingest::SummaryConfig { redactor, ..Default::default() };
                        let r = ingest::summarize_pending(&store, &mut embedder, s.as_ref(), &state, &cfg, now).await?;
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
