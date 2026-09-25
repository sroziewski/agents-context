use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::Context as _;
use clap::{Parser, Subcommand, ValueEnum};
use ctx::config::Config;
use ctx::embed_server::{self, RemoteEmbedder};
use ctx::search::{Context, Query, Scope, format_hits, format_sessions, parse_date};
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
    },
    /// Index new or changed sessions (incremental; safe to run any time).
    Index {
        #[arg(long, value_enum, default_value_t = SourceArg::All)]
        source: SourceArg,
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
    /// List indexed sessions, newest first.
    Sessions {
        #[arg(long)]
        all: bool,
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// Per-project index statistics.
    Stats,
    /// MCP server over stdio (register with `claude mcp add` / OpenCode config).
    Mcp,
}

#[derive(Clone, Copy, ValueEnum)]
enum SourceArg {
    Claude,
    Opencode,
    All,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let cfg = Config::load();
    match cli.command {
        Command::EmbedServer { addr, provider, model_dir } => {
            let dir = model_dir.unwrap_or_else(|| cfg.model_dir.clone());
            let model = model_name(&dir);
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
            embed_server::serve(embed_server::router(Box::new(embedder), &model), addr).await
        }
        Command::Index { source } => {
            let which = match source {
                SourceArg::Claude => Sources::Claude,
                SourceArg::Opencode => Sources::OpenCode,
                SourceArg::All => Sources::All,
            };
            let refs = ctx::discover(&cfg, which)?;
            let remote = connect_embedder(&cfg).await?;
            let model = remote.model().to_owned();
            let counter = HfTokenizer::from_file(&cfg.model_dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
            let state = State::open(&cfg.state_path)?;
            let store = connect_store(&cfg)?;
            let t = Instant::now();
            // Embedding over blocking HTTP inside the async indexer is fine for a CLI run.
            let report = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    let mut remote = remote;
                    ctx::run_index(&store, &mut remote, &model, &counter, &state, &refs, |p| match p {
                        Progress::Indexed { id, chunks } => eprintln!("  indexed {id} ({chunks} chunks)"),
                        Progress::Failed { id, error } => eprintln!("  FAILED {id}: {error}"),
                    })
                    .await
                })
            })?;
            println!(
                "scanned {} · indexed {} · unchanged {} · empty {} · failed {} · {} chunks · {} secrets redacted · {:.1?}",
                report.scanned,
                report.indexed,
                report.unchanged,
                report.empty,
                report.failed.len(),
                report.chunks,
                report.redactions,
                t.elapsed()
            );
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
            println!("{}", format_hits(&ctx.search(&q).await?));
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
        Command::Mcp => {
            let embedder: Box<dyn Embedder> = Box::new(embed_server::LazyRemote::new(&cfg.embed_url, embed::BGEM3_DIM));
            let ctx = Context { store: Arc::new(connect_store(&cfg)?), embedder: Arc::new(Mutex::new(embedder)) };
            let server = mcp::ContextServer::new(ctx, mcp::current_project(), cfg.state_path.clone());
            mcp::serve_stdio(server).await
        }
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
