use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Context as _;
use clap::{Parser, Subcommand, ValueEnum};
use ctx::config::Config;
use ctx::embed_server::{self, RemoteEmbedder, SharedEmbedder};
use ctx::hook;
use ctx::queue::{IndexQueue, Target};
use ctx::search::{Context, Query, Scope, format_results, format_sessions, parse_date};
use ctx::{Sources, mcp};
use embed::{Bgem3Embedder, Bgem3Options, Embedder, ModelSource, Provider};
use ingest::{HfTokenizer, Progress, State};
use store::Store;

#[derive(Parser)]
#[command(name = "ctx", about = "Local semantic memory over Claude Code and OpenCode sessions")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the embedding daemon (holds bge-m3 on the GPU for every other command).
    EmbedServer {
        #[arg(long, default_value = "127.0.0.1:7997")]
        addr: SocketAddr,
        #[arg(long, default_value = "directml")]
        provider: Provider,
        /// Model directory; defaults to CTX_MODEL_DIR.
        #[arg(long)]
        model_dir: Option<PathBuf>,
        /// Minutes between background index runs of all sources; 0 disables them. Hooks
        /// trigger runs on top of this.
        #[arg(long, default_value_t = 15)]
        index_every: u64,
    },
    /// Called by agent hooks: asks the daemon to re-index and exits 0 immediately, always.
    Hook {
        #[arg(long, value_enum, default_value_t = SourceArg::All)]
        source: SourceArg,
    },
    /// Index new or changed sessions (incremental; safe to run any time).
    Index {
        #[arg(long, value_enum, default_value_t = SourceArg::All)]
        source: SourceArg,
        /// Index in this process even when the daemon is running (by default the daemon does it,
        /// so two runs never race over the same state).
        #[arg(long)]
        local: bool,
    },
    /// Search sessions from the terminal.
    Search {
        query: String,
        /// Search every project instead of the current directory's.
        #[arg(long)]
        all: bool,
        /// Search this project directory instead of the current one.
        #[arg(long)]
        project: Option<String>,
        #[arg(long)]
        since: Option<String>,
        #[arg(long)]
        until: Option<String>,
        #[arg(long, default_value_t = 6)]
        limit: u64,
        #[arg(long)]
        subagents: bool,
    },
    /// Search Teams channel messages by meaning and keywords.
    SearchTeams {
        query: String,
        #[arg(long)]
        group: Option<String>,
        #[arg(long)]
        team: Option<String>,
        #[arg(long)]
        channel: Option<String>,
        #[arg(long)]
        author: Option<String>,
        #[arg(long)]
        since: Option<String>,
        #[arg(long)]
        until: Option<String>,
        #[arg(long, default_value_t = 6)]
        limit: usize,
    },
    /// Print a whole Teams thread.
    TeamsThread { thread_id: String },
    /// List indexed sessions, newest first.
    Sessions {
        #[arg(long)]
        all: bool,
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// Per-project index statistics.
    Stats,
    /// Manage redaction of secrets in the index.
    Redact {
        #[command(subcommand)]
        action: RedactAction,
    },
    /// Propose Claude Code memories from this project's session summaries, for review.
    Memories {
        #[command(subcommand)]
        action: MemoriesAction,
        /// Project directory (default: the current one).
        #[arg(long, global = true)]
        project: Option<String>,
    },
    /// MCP server over stdio (register with `claude mcp add` / OpenCode config).
    Mcp,
}

#[derive(Subcommand)]
enum RedactAction {
    /// Add a secret to the redaction list, read from stdin (never an argument, which would land
    /// in shell history), then redact it everywhere in the index.
    Add,
    /// Apply the current rules and list to everything already indexed.
    Rescan,
}

#[derive(Subcommand)]
enum MemoriesAction {
    /// Ask for new suggestions (replaces the pending list); nothing is written to memory.
    Suggest,
    /// Show pending suggestions.
    List,
    /// Write the chosen suggestions (numbers from `list`) as memory files.
    Accept { numbers: Vec<usize> },
    /// Drop all pending suggestions.
    Discard,
}

#[derive(Clone, Copy, ValueEnum)]
enum SourceArg {
    Claude,
    Opencode,
    Teams,
    All,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        // Claude Code treats exit code 2 (clap's usage error) from a Stop/PreCompact hook as a
        // blocking error and forces the turn to continue; a typo or stale binary must not do that.
        Err(e) if std::env::args().nth(1).as_deref() == Some("hook") && e.use_stderr() => return Ok(()),
        Err(e) => e.exit(),
    };
    let cfg = Config::load();
    match cli.command {
        Command::EmbedServer { addr, provider, model_dir, index_every } => {
            let dir = model_dir.unwrap_or_else(|| cfg.model_dir.clone());
            let model = model_name(&dir);
            // Before the slow GPU load, and from the same directory as the model.
            let counter = HfTokenizer::from_file(&dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
            let t = Instant::now();
            let embedder = tokio::task::spawn_blocking(move || {
                Bgem3Embedder::new(Bgem3Options {
                    model: ModelSource::OnnxDir(dir),
                    provider,
                    ..Default::default()
                })
            })
            .await??;
            eprintln!("loaded {model} on {provider:?} in {:.1?}", t.elapsed());
            let shared = SharedEmbedder::new(Box::new(embedder));
            let summarizer: Option<Arc<dyn ingest::Summarizer>> = cfg.summaries.then(|| {
                Arc::new(ctx::summarizer::ClaudeCli::new(
                    cfg.claude_exe.clone(),
                    &cfg.summary_model,
                    cfg.summarizer_dir.clone(),
                    Duration::from_secs(180),
                )) as Arc<dyn ingest::Summarizer>
            });
            let teams_store = Arc::new(Store::connect(&cfg.qdrant_url, cfg.qdrant_api_key.clone(), &cfg.teams_collection, embed::BGEM3_DIM)?);
            let runner = ctx::daemon_runner(
                cfg.clone(),
                Arc::new(connect_store(&cfg)?),
                Arc::new(counter),
                shared.clone(),
                model.clone(),
                summarizer,
                Some(teams_store),
            );
            let queue = IndexQueue::start(runner, Duration::from_secs(20));
            if !addr.ip().is_loopback() {
                eprintln!("warning: {addr} is not a loopback address; the daemon has no authentication");
            }
            // Catch up on whatever changed while the daemon was down (reboot, crash).
            queue.request(Target::All);
            if index_every > 0 {
                queue.schedule(Duration::from_secs(index_every * 60));
            }
            embed_server::serve(embed_server::service_router(shared, &model, Some(queue)), addr).await
        }
        Command::Hook { source } => {
            hook::drain_stdin();
            let target = to_target(source);
            let url = cfg.embed_url.clone();
            let _ = tokio::task::spawn_blocking(move || {
                hook::notify(&url, target, Duration::from_secs(2), |k| std::env::var(k).ok())
            })
            .await;
            Ok(())
        }
        Command::Index { source, local } => {
            if !local {
                let url = cfg.embed_url.clone();
                let target = to_target(source);
                let delegated =
                    tokio::task::spawn_blocking(move || hook::delegate_index(&url, target, Duration::from_secs(3600))).await??;
                if let Some(status) = delegated {
                    return match (status.last_report, status.last_error) {
                        (_, Some(err)) => Err(anyhow::anyhow!("daemon index run failed: {err}")),
                        (Some(report), None) => {
                            println!("{report}  (run by the daemon)");
                            Ok(())
                        }
                        (None, None) => Ok(()),
                    };
                }
            }
            let which = match source {
                SourceArg::Claude => Sources::Claude,
                SourceArg::Opencode => Sources::OpenCode,
                SourceArg::Teams => anyhow::bail!("indexing Teams needs the daemon (`ctx embed-server`); run without --local"),
                SourceArg::All => Sources::All,
            };
            let refs = ctx::discover(&cfg, which)?;
            let remote = connect_embedder(&cfg).await?;
            let model = remote.model().to_owned();
            let counter = HfTokenizer::from_file(&cfg.model_dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
            let state = State::open(&cfg.state_path)?;
            let store = connect_store(&cfg)?;
            let redactor = ingest::Redactor::from_file(&cfg.redact_file)?;
            let t = Instant::now();
            // Embedding over blocking HTTP inside the async indexer is fine for a CLI run.
            let report = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    let mut remote = remote;
                    ctx::run_index(&store, &mut remote, &model, &counter, &state, &refs, &redactor, |p| match p {
                        Progress::Indexed { id, chunks } => eprintln!("  indexed {id} ({chunks} chunks)"),
                        Progress::Failed { id, error } => eprintln!("  FAILED {id}: {error}"),
                    })
                    .await
                })
            })?;
            println!("{}", ctx::format_report(&report, t.elapsed()));
            Ok(())
        }
        Command::Search { query, all, project, since, until, limit, subagents } => {
            let ctx = context(&cfg).await?;
            let q = Query {
                text: query,
                scope: scope(all, project),
                source: None,
                since: since.as_deref().map(|d| parse_date(d, false)).transpose()?,
                until: until.as_deref().map(|d| parse_date(d, true)).transpose()?,
                limit,
                include_subagents: subagents,
            };
            println!("{}", format_results(&ctx.search(&q).await?));
            Ok(())
        }
        Command::SearchTeams { query, group, team, channel, author, since, until, limit } => {
            let teams = teams_search(&cfg, &context(&cfg).await?);
            let q = ctx::teams::TeamsQuery {
                text: query,
                group,
                team,
                channel,
                author,
                since: since.as_deref().map(|d| parse_date(d, false)).transpose()?,
                until: until.as_deref().map(|d| parse_date(d, true)).transpose()?,
                limit,
            };
            let project = sources::project::key(&std::env::current_dir()?.to_string_lossy());
            println!("{}", ctx::teams::format_results(&teams.search(&q, Some(&project)).await?));
            Ok(())
        }
        Command::TeamsThread { thread_id } => {
            let store = Arc::new(connect_store(&cfg)?);
            let ctx = Context { store, embedder: Arc::new(Mutex::new(Box::new(embed::HashEmbedder::new(1)))) };
            println!("{}", teams_search(&cfg, &ctx).thread(&thread_id, usize::MAX)?);
            Ok(())
        }
        Command::Sessions { all, limit } => {
            let store = connect_store(&cfg)?;
            let ctx = Context { store: Arc::new(store), embedder: Arc::new(Mutex::new(Box::new(embed::HashEmbedder::new(1)))) };
            println!("{}", format_sessions(&ctx.sessions(&scope(all, None), limit).await?));
            Ok(())
        }
        Command::Stats => {
            let state = State::open(&cfg.state_path)?;
            let (mut sessions, mut chunks) = (0, 0);
            for (project, s, c) in state.per_project()?.into_iter().filter(|(p, ..)| !p.is_empty()) {
                println!("{project:<50} {s:>5} sessions {c:>7} chunks");
                sessions += s;
                chunks += c;
            }
            println!("{:<50} {sessions:>5} sessions {chunks:>7} chunks", "TOTAL");
            Ok(())
        }
        Command::Redact { action } => {
            if let RedactAction::Add = action {
                use std::io::{IsTerminal, Read};
                if std::io::stdin().is_terminal() {
                    eprintln!("type the secret, then Enter, Ctrl+Z, Enter (the console shows what you type; it is not logged):");
                }
                let mut input = String::new();
                std::io::stdin().read_to_string(&mut input)?;
                let entries = ctx::redact::append_literal(&cfg.redact_file, input.trim())?;
                println!("redaction list has {entries} entries ({})", cfg.redact_file.display());
            }
            // 1. Re-read every session from source with the current rules; this also restores
            //    text an older, stricter rule redacted. It runs as a daemon job, which re-reads
            //    the list at its start, and is waited for, so any run that was already in
            //    progress with the old list has finished before step 2.
            let n = State::open(&cfg.state_path)?.force_reread()?;
            let url = cfg.embed_url.clone();
            match tokio::task::spawn_blocking(move || hook::delegate_index(&url, Target::All, Duration::from_secs(3600))).await?? {
                Some(status) => println!("re-read {n} sessions: {}", status.last_report.or(status.last_error).unwrap_or_default()),
                None => anyhow::bail!("the embedding daemon is not running; start it and run `ctx redact rescan`"),
            }
            // 2. Last, apply the rules to everything stored, which covers sessions whose
            //    transcript is gone and anything an older run wrote in between.
            let redactor = ingest::Redactor::from_file(&cfg.redact_file)?;
            let store = connect_store(&cfg)?;
            let mut embedder = connect_embedder(&cfg).await?;
            let report = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(ingest::rescan(&store, &mut embedder, &redactor))
            })?;
            println!(
                "checked {} points · rewrote {} · {} secrets redacted",
                report.checked, report.changed, report.redactions
            );
            Ok(())
        }
        Command::Memories { action, project } => {
            let key = match scope(false, project) {
                Scope::Project(k) => k,
                Scope::All => unreachable!("scope(false, _) is always a project"),
            };
            let pending = ctx::memories::pending_path(&cfg.data_dir, &key);
            let memory_dir = || {
                ctx::memories::memory_dir_for(&key, &cfg.claude_dir)
                    .ok_or_else(|| anyhow::anyhow!("no Claude Code project directory for {key} under {}", cfg.claude_dir.display()))
            };
            match action {
                MemoriesAction::Suggest => {
                    let existing = memory_dir().map(|d| ctx::memories::existing_memories(&d)).unwrap_or_default();
                    let summarizer = ctx::summarizer::ClaudeCli::new(cfg.claude_exe.clone(), &cfg.summary_model, cfg.summarizer_dir.clone(), Duration::from_secs(180));
                    let redactor = ingest::Redactor::from_file(&cfg.redact_file)?;
                    let store = connect_store(&cfg)?;
                    let candidates = tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(ctx::memories::suggest(&store, &summarizer, &redactor, &key, &existing, Duration::from_secs(180)))
                    })?;
                    ctx::memories::save_pending(&pending, &candidates)?;
                    println!("{}", ctx::memories::format_candidates(&candidates));
                    if !candidates.is_empty() {
                        println!("\naccept with: ctx memories accept <numbers>   (nothing is written until then)");
                    }
                }
                MemoriesAction::List => println!("{}", ctx::memories::format_candidates(&ctx::memories::load_pending(&pending))),
                MemoriesAction::Accept { numbers } => {
                    let list = ctx::memories::load_pending(&pending);
                    anyhow::ensure!(!numbers.is_empty(), "give the numbers to accept, e.g. `ctx memories accept 1 3`");
                    let mut chosen = Vec::new();
                    for n in &numbers {
                        chosen.push(list.get(n.wrapping_sub(1)).cloned().ok_or_else(|| anyhow::anyhow!("no suggestion {n} (see `ctx memories list`)"))?);
                    }
                    let dir = memory_dir()?;
                    let written = ctx::memories::accept(&dir, &chosen)?;
                    for w in &written {
                        println!("wrote {}", w.display());
                    }
                    if written.len() < chosen.len() {
                        println!("{} skipped: a memory file with that name already exists", chosen.len() - written.len());
                    }
                    let rest: Vec<_> = list.into_iter().filter(|c| !chosen.contains(c)).collect();
                    ctx::memories::save_pending(&pending, &rest)?;
                }
                MemoriesAction::Discard => {
                    let _ = std::fs::remove_file(&pending);
                    println!("pending suggestions discarded");
                }
            }
            Ok(())
        }
        Command::Mcp => {
            let embedder: Box<dyn Embedder> = Box::new(embed_server::LazyRemote::new(&cfg.embed_url, embed::BGEM3_DIM));
            let ctx = Context { store: Arc::new(connect_store(&cfg)?), embedder: Arc::new(Mutex::new(embedder)) };
            let teams = teams_search(&cfg, &ctx);
            let server = mcp::ContextServer::new(ctx, mcp::current_project(), cfg.state_path.clone()).with_teams(teams);
            mcp::serve_stdio(server).await
        }
    }
}

/// Teams search sharing the context's embedder, over the teams collection.
fn teams_search(cfg: &Config, ctx: &Context) -> ctx::teams::TeamsSearch {
    let store = Store::connect(&cfg.qdrant_url, cfg.qdrant_api_key.clone(), &cfg.teams_collection, embed::BGEM3_DIM)
        .expect("Qdrant client construction does not connect");
    ctx::teams::TeamsSearch {
        store: Arc::new(store),
        embedder: ctx.embedder.clone(),
        db: cfg.teams_db.clone(),
        people: cfg.teams_people.clone(),
        projects: cfg.teams_projects.clone(),
        redact: cfg.redact_file.clone(),
    }
}

fn to_target(source: SourceArg) -> Target {
    match source {
        SourceArg::Claude => Target::Claude,
        SourceArg::Opencode => Target::Opencode,
        SourceArg::Teams => Target::Teams,
        SourceArg::All => Target::All,
    }
}

fn scope(all: bool, project: Option<String>) -> Scope {
    if all {
        Scope::All
    } else {
        let dir = project.unwrap_or_else(|| std::env::current_dir().map(|d| d.to_string_lossy().into_owned()).unwrap_or_default());
        Scope::Project(sources::project::key(&dir))
    }
}

/// Model identity recorded in the index version: the export's directory name.
fn model_name(dir: &std::path::Path) -> String {
    dir.file_name().map_or_else(|| "bge-m3".into(), |n| n.to_string_lossy().into_owned())
}

fn connect_store(cfg: &Config) -> anyhow::Result<Store> {
    Store::connect(&cfg.qdrant_url, cfg.qdrant_api_key.clone(), &cfg.collection, embed::BGEM3_DIM)
        .with_context(|| format!("connecting to Qdrant at {}", cfg.qdrant_url))
}

async fn connect_embedder(cfg: &Config) -> anyhow::Result<RemoteEmbedder> {
    let url = cfg.embed_url.clone();
    tokio::task::spawn_blocking(move || RemoteEmbedder::connect(&url)).await?
}

async fn context(cfg: &Config) -> anyhow::Result<Context> {
    let embedder: Box<dyn Embedder> = Box::new(connect_embedder(cfg).await?);
    Ok(Context { store: Arc::new(connect_store(cfg)?), embedder: Arc::new(Mutex::new(embedder)) })
}
