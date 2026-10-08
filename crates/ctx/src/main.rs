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
    /// On `/clear` it also hands the cleared session over to the new one (SessionEnd records it,
    /// SessionStart prints it; see `ctx::handoff`).
    Hook {
        #[arg(long, value_enum, default_value_t = SourceArg::All)]
        source: SourceArg,
        /// UserPromptSubmit: print matching past-session context for the prompt instead of
        /// indexing (only while auto-inject is on, see `ctx inject`).
        #[arg(long)]
        inject: bool,
    },
    /// Claude Code status line: model, context usage with a /clear nudge, daemon and Teams
    /// health. Reads the status JSON on stdin.
    Statusline,
    /// Auto-inject of past-session summaries into Claude Code prompts.
    Inject {
        #[command(subcommand)]
        action: InjectAction,
    },
    /// Measure retrieval quality on the query set (default: the eval file in the config dir).
    Eval {
        #[arg(long)]
        file: Option<PathBuf>,
        /// Record the production variant's ranks as the baseline.
        #[arg(long)]
        save_baseline: bool,
        /// Exit with an error if the production variant lost top-5 cases against the baseline.
        #[arg(long)]
        check: bool,
        /// Also try recency weights and half-lives, to tune them.
        #[arg(long)]
        sweep: bool,
        /// Compare two variants by name (e.g. `--compare dense+summaries production`): the
        /// per-case rank moves and bootstrap CIs on R@1, MRR and turn-MRR.
        #[arg(long, num_args = 2, value_names = ["A", "B"])]
        compare: Vec<String>,
        /// DEV-28 reranker spike: rerank a dense turn pool with this CPU cross-encoder and
        /// compare it to the dense baseline (jina-v2, bge-v2-m3 or bge-base). Downloads the
        /// model on first use.
        #[arg(long)]
        rerank_model: Option<String>,
        /// Candidates fed to the reranker (dense top-N).
        #[arg(long, default_value_t = 30)]
        rerank_pool: u64,
        /// Cap ONNX Runtime CPU threads for the reranker (default: all cores).
        #[arg(long)]
        rerank_threads: Option<usize>,
        /// What to evaluate: "sessions" (default) or "teams".
        #[arg(long, default_value = "sessions")]
        target: String,
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
        /// Deep search: rerank the results with the daemon's CPU cross-encoder. Slower
        /// (seconds), off by default; falls back to the normal order if the daemon has no
        /// reranker.
        #[arg(long)]
        rerank: bool,
    },
    /// Search Teams channel messages by meaning and keywords.
    SearchTeams {
        query: String,
        /// "chats", "channels" or "all".
        #[arg(long, default_value = "all")]
        kind: String,
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
    /// Search Teams meeting transcripts by meaning and keywords.
    SearchMeetings {
        query: String,
        #[arg(long)]
        speaker: Option<String>,
        #[arg(long)]
        since: Option<String>,
        #[arg(long)]
        until: Option<String>,
        #[arg(long, default_value_t = 6)]
        limit: usize,
    },
    /// Print a whole Teams meeting transcript.
    Meeting {
        meeting_id: String,
        /// Transcript around this point ("mm:ss", as cited in the summary).
        #[arg(long)]
        at: Option<String>,
        /// The whole transcript instead of the summary (paged with --from-line).
        #[arg(long)]
        transcript: bool,
        #[arg(long, default_value_t = 0)]
        from_line: usize,
    },
    /// Read Jira issues (read-only).
    Jira {
        #[command(subcommand)]
        action: JiraAction,
    },
    /// Remove indexed data (optional, manual; dry-run unless --yes).
    Prune {
        #[command(subcommand)]
        action: PruneAction,
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
enum JiraAction {
    /// Search issues. Plain text unless --jql; a bare key (HD-123) is fetched directly.
    Search {
        query: String,
        /// Scope to this project directory's mapped Jira keys instead of the current one.
        #[arg(long)]
        project: Option<String>,
        /// Treat the query as raw JQL.
        #[arg(long)]
        jql: bool,
        /// Search all projects, ignoring the project→key mapping.
        #[arg(long)]
        all: bool,
        #[arg(long, default_value_t = 8)]
        limit: u32,
    },
    /// Print one issue with its description and comments.
    Issue { key: String },
    /// Add a comment. Dry-run unless --write, and refused unless JIRA_WRITE=on.
    Comment {
        key: String,
        #[arg(long)]
        body: String,
        /// Actually send it (the explicit-intent lock); without it, dry-run.
        #[arg(long)]
        write: bool,
    },
    /// Move an issue to a status (case-insensitive). Dry-run unless --write; needs JIRA_WRITE=on.
    Transition {
        key: String,
        #[arg(long)]
        to: String,
        #[arg(long)]
        write: bool,
    },
}

#[derive(Subcommand)]
enum PruneAction {
    /// Drop all indexed points for a project (sessions collection), by its directory.
    Project {
        dir: String,
        /// Actually delete; without it, a dry run reporting what would go.
        #[arg(long)]
        yes: bool,
    },
    /// Remove points older than N days (by their timestamp) from a collection.
    OlderThan {
        days: u32,
        #[arg(long, value_enum, default_value_t = PruneSource::Sessions)]
        source: PruneSource,
        #[arg(long)]
        yes: bool,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum PruneSource {
    Sessions,
    Teams,
    Jira,
    All,
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
    /// Ask for new suggestions and fact-check them (replaces the pending list); nothing is
    /// written to memory.
    Suggest,
    /// Fact-check the pending suggestions again (after new sessions, or ones never checked).
    Verify,
    /// Show pending suggestions with their verdicts.
    List,
    /// Write the chosen suggestions (numbers from `list`) as memory files.
    Accept {
        numbers: Vec<usize>,
        /// Also write suggestions that were rejected, unverified or failed a check.
        #[arg(long)]
        force: bool,
    },
    /// Drop the given pending suggestions, or all of them without numbers.
    Discard { numbers: Vec<usize> },
}

#[derive(Subcommand)]
enum InjectAction {
    /// Turn auto-inject on (takes effect on the next prompt). With --client, only for that
    /// agent; without, the global default for both.
    On {
        #[arg(long)]
        client: Option<String>,
    },
    /// Turn it off (globally, or for one --client).
    Off {
        #[arg(long)]
        client: Option<String>,
    },
    /// Show whether it is on and its settings.
    Status,
    /// Show what would be injected for a prompt asked from the current directory.
    Preview { prompt: String },
}

#[derive(Clone, Copy, ValueEnum)]
enum SourceArg {
    Claude,
    Opencode,
    Teams,
    Jira,
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
    // Box the dispatch future onto the heap: in a debug build this match compiles to a very large
    // async state machine, and on Windows the main thread's stack is only 1 MB, so leaving it
    // inline overflows the stack even for a trivial `ctx hook`.
    Box::pin(async move {
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
            let meetings_store = Arc::new(Store::connect(&cfg.qdrant_url, cfg.qdrant_api_key.clone(), &cfg.meetings_collection, embed::BGEM3_DIM)?);
            // Jira store only when Jira is configured; the twice-daily schedule drives it.
            let jira_store = cfg.jira().map(|_| Store::connect(&cfg.qdrant_url, cfg.qdrant_api_key.clone(), &cfg.jira_collection, embed::BGEM3_DIM)).transpose()?.map(Arc::new);
            let sessions_store = Arc::new(connect_store(&cfg)?);
            let counter: Arc<dyn ingest::TokenCounter + Send + Sync> = Arc::new(counter);
            // Always attached: whether to inject is decided by the hook on every prompt, so
            // `ctx inject on` needs no daemon restart.
            let injector = Arc::new(ctx::inject::Injector::new(
                sessions_store.clone(),
                Some(teams_store.clone()),
                shared.0.clone(),
                counter.clone(),
                cfg.recency,
                cfg.inject,
                cfg.redact_file.clone(),
            ).with_env_file(Config::env_file()));
            let runner = ctx::daemon_runner(
                cfg.clone(),
                sessions_store,
                counter,
                shared.clone(),
                model.clone(),
                summarizer,
                Some(teams_store),
                Some(meetings_store),
                jira_store.clone(),
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
            // Jira: only on its own twice-daily schedule (not part of All), when configured.
            if jira_store.is_some() {
                eprintln!("jira indexing scheduled at {:?} local", cfg.jira_index_times);
                queue.schedule_daily(Target::Jira, cfg.jira_index_times.clone());
            }
            // Lazily-built CPU reranker for opt-in deep search; loaded only on the first
            // /rerank call, never for a normal search.
            let (rerank_model, rerank_threads, rerank_cache) = (cfg.rerank_model.clone(), cfg.rerank_threads, cfg.rerank_cache_dir.clone());
            let reranker: embed_server::RerankerFactory = Arc::new(move || {
                let which: embed::RerankModel = rerank_model.parse().map_err(embed::EmbedError::Model)?;
                Ok(Box::new(embed::FastembedReranker::hub(which, 512, rerank_threads, 32, Some(rerank_cache.clone()))?) as Box<dyn embed::Reranker>)
            });
            embed_server::serve(embed_server::daemon_router(shared, &model, Some(queue), Some(injector), Some(reranker)), addr).await
        }
        Command::Hook { inject: true, .. } => {
            let payload = hook::read_stdin();
            // Fast path: skip the HTTP call when Claude inject is off. The daemon gates per
            // client too, so this only saves a round-trip.
            if cfg.inject.enabled_claude {
                let url = cfg.embed_url.clone();
                let out = tokio::task::spawn_blocking(move || hook::inject(&url, &payload, Duration::from_secs(3), |k| std::env::var(k).ok())).await;
                if let Ok(Some(json)) = out {
                    println!("{json}");
                }
            }
            Ok(())
        }
        Command::Statusline => {
            let mut s = ctx::statusline::parse(&hook::read_stdin());
            s.daemon_up = Some(ctx::statusline::daemon_up(&cfg.embed_url, Duration::from_millis(150)));
            let synced = cfg.teams_db.is_file().then(|| sources::teams::TeamsDb::open(&cfg.teams_db).ok()?.last_synced_at().ok()?).flatten();
            s.teams_stale = synced.is_some_and(|ms| ctx::handoff::now_ms() - ms > cfg.teams_watch.stale_after.as_millis() as i64);
            let t = ctx::statusline::Thresholds { warn: cfg.status_warn, crit: cfg.status_crit, handoff: cfg.handoff };
            println!("{}", ctx::statusline::render(&s, t));
            Ok(())
        }
        Command::Inject { action } => {
            let env_file = Config::env_file();
            match action {
                InjectAction::On { ref client } | InjectAction::Off { ref client } => {
                    let on = matches!(action, InjectAction::On { .. });
                    let key = match client.as_deref() {
                        None => "CTX_INJECT",
                        Some("claude") => "CTX_INJECT_CLAUDE",
                        Some("opencode") => "CTX_INJECT_OPENCODE",
                        Some(other) => anyhow::bail!("unknown client {other:?}; use claude or opencode"),
                    };
                    ctx::config::set_env_value(&env_file, key, if on { "on" } else { "off" })?;
                    println!("auto-inject {} for {} ({})", if on { "on" } else { "off" }, client.as_deref().unwrap_or("all clients"), env_file.display());
                    if on {
                        println!("Claude needs the UserPromptSubmit hook (`ctx.exe hook --inject`); OpenCode needs the ctx-inject plugin.");
                    }
                }
                InjectAction::Status => {
                    let i = cfg.inject;
                    let onoff = |b| if b { "on" } else { "off" };
                    println!(
                        "auto-inject: claude {} · opencode {} · teams {} · min_score {:.2} · teams_min_score {:.2} · max {} tokens · recency weight {:.2}, half-life {} days",
                        onoff(i.enabled_claude),
                        onoff(i.enabled_opencode),
                        onoff(i.teams),
                        i.min_score,
                        i.teams_min_score,
                        i.max_tokens,
                        cfg.recency.weight,
                        cfg.recency.half_life_days
                    );
                }
                InjectAction::Preview { prompt } => {
                    // Show what would be injected even when inject is currently off.
                    let mut cfg = cfg.clone();
                    cfg.inject.enabled_claude = true;
                    let injector = injector(&cfg).await?;
                    let req = ctx::inject::ContextRequest {
                        prompt,
                        cwd: std::env::current_dir()?.to_string_lossy().into_owned(),
                        session_id: String::new(),
                        client: "claude".into(),
                    };
                    let resp = injector.context(&req).await?;
                    if resp.context.is_empty() {
                        println!("nothing would be injected (no summary of this project scores {:.2} or more, or the prompt is too short)", cfg.inject.min_score);
                    } else {
                        println!("{}\n\n({} tokens max; enabled: {})", resp.context, cfg.inject.max_tokens, cfg.inject.enabled);
                    }
                }
            }
            Ok(())
        }
        Command::Eval { file, save_baseline, check, sweep, compare, rerank_model, rerank_pool, rerank_threads, target } => {
            run_eval(cfg, file, save_baseline, check, sweep, compare, rerank_model, rerank_pool, rerank_threads, target).await
        }
        Command::Hook { source, .. } => {
            let payload = hook::read_stdin();
            if std::env::var(hook::SUMMARIZER_ENV).is_ok() {
                return Ok(());
            }
            let event: serde_json::Value = serde_json::from_str(&payload).unwrap_or_default();
            let field = |k: &str| event.get(k).and_then(serde_json::Value::as_str).unwrap_or_default().to_owned();
            match field("hook_event_name").as_str() {
                // Nothing to index at the start of a session; only a /clear handoff to give.
                "SessionStart" => {
                    if cfg.handoff && field("source") == "clear" {
                        let redactor = ingest::Redactor::from_file(&cfg.redact_file).unwrap_or_default();
                        let store = connect_store(&cfg).ok();
                        if let Some(json) = ctx::handoff::on_session_start(&event, &cfg.data_dir, &redactor, store.as_ref()).await {
                            println!("{json}");
                        }
                    }
                    return Ok(());
                }
                "SessionEnd" if cfg.handoff && field("reason") == "clear" => {
                    let cleared = ctx::handoff::Cleared {
                        session_id: field("session_id"),
                        transcript: PathBuf::from(field("transcript_path")),
                        at_ms: ctx::handoff::now_ms(),
                    };
                    let file = ctx::handoff::records_path(&cfg.data_dir);
                    // Best effort: without the record the new session falls back to the newest transcript.
                    let _ = ctx::handoff::record_clear(&file, &sources::project::key(&field("cwd")), cleared);
                }
                _ => {}
            }
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
                SourceArg::Jira => anyhow::bail!("indexing Jira needs the daemon (`ctx embed-server`); run without --local"),
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
        Command::Search { query, all, project, since, until, limit, subagents, rerank } => {
            let ctx = context(&cfg).await?;
            let mut q = Query::new(query, scope(all, project), limit);
            q.since = since.as_deref().map(|d| parse_date(d, false)).transpose()?;
            q.until = until.as_deref().map(|d| parse_date(d, true)).transpose()?;
            q.include_subagents = subagents;
            q.recency = cfg.recency;
            q.mode = cfg.search_mode;
            let results = if rerank {
                let rr: Arc<Mutex<Box<dyn embed::Reranker>>> = Arc::new(Mutex::new(Box::new(embed_server::RemoteReranker::new(&cfg.embed_url))));
                match ctx.search_reranked(&q, rr, cfg.rerank_pool).await {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("deep search unavailable ({e}); showing the normal order");
                        ctx.search(&q).await?
                    }
                }
            } else {
                ctx.search(&q).await?
            };
            println!("{}", format_results(&results));
            Ok(())
        }
        Command::SearchTeams { query, kind, group, team, channel, author, since, until, limit } => {
            let teams = teams_search(&cfg, &context(&cfg).await?);
            let q = ctx::teams::TeamsQuery {
                text: query,
                kind: ctx::teams::Conversations::parse(&kind)?,
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
        Command::SearchMeetings { query, speaker, since, until, limit } => {
            let teams = teams_search(&cfg, &context(&cfg).await?);
            let q = ctx::teams::MeetingQuery {
                text: query,
                speaker,
                since: since.as_deref().map(|d| parse_date(d, false)).transpose()?,
                until: until.as_deref().map(|d| parse_date(d, true)).transpose()?,
                limit,
            };
            println!("{}", ctx::teams::format_meetings(&teams.search_meetings(&q).await?));
            Ok(())
        }
        Command::Meeting { meeting_id, at, transcript, from_line } => {
            let view = match (at, transcript) {
                (Some(at), _) => ctx::teams::MeetingView::Around { offset_s: ctx::teams::parse_offset(&at)? },
                (None, true) => ctx::teams::MeetingView::Transcript { from_line },
                (None, false) => ctx::teams::MeetingView::Summary,
            };
            let store = Arc::new(connect_store(&cfg)?);
            let ctx = Context { store, embedder: Arc::new(Mutex::new(Box::new(embed::HashEmbedder::new(1)))) };
            println!("{}", teams_search(&cfg, &ctx).meeting(&meeting_id, view)?);
            Ok(())
        }
        Command::Jira { action } => {
            let js = jira_search(&cfg)?;
            match action {
                JiraAction::Search { query, project, jql, all, limit } => {
                    let project_key = if all {
                        None
                    } else {
                        let dir = project.unwrap_or_else(|| std::env::current_dir().map(|d| d.to_string_lossy().into_owned()).unwrap_or_default());
                        Some(sources::project::key(&dir))
                    };
                    println!("{}", js.search(&query, project_key.as_deref(), jql, all, limit)?);
                }
                JiraAction::Issue { key } => println!("{}", js.issue(&key)?),
                JiraAction::Comment { key, body, write } => println!("{}", js.comment(&key, &body, write)?),
                JiraAction::Transition { key, to, write } => println!("{}", js.transition(&key, &to, write)?),
            }
            Ok(())
        }
        Command::Prune { action } => {
            match action {
                PruneAction::Project { dir, yes } => {
                    let key = sources::project::key(&dir);
                    let store = connect_store(&cfg)?;
                    let r = ctx::prune::prune_project(&store, &cfg.state_path, &key, yes).await?;
                    println!("{}", ctx::prune::format_report(&format!("project {key}"), &r));
                    if !yes && r.points > 0 {
                        println!("re-run with --yes to apply. Note: a project whose sessions still exist is re-indexed on the next run.");
                    }
                }
                PruneAction::OlderThan { days, source, yes } => {
                    let cutoff = jiff::Timestamp::now().as_millisecond() - i64::from(days) * 86_400_000;
                    for (label, collection, id_field) in prune_targets(source, &cfg) {
                        let store = Store::connect(&cfg.qdrant_url, cfg.qdrant_api_key.clone(), &collection, embed::BGEM3_DIM)?;
                        let r = ctx::prune::prune_older_than(&store, &cfg.state_path, id_field, cutoff, yes).await?;
                        println!("{}", ctx::prune::format_report(&format!("{label} older than {days}d"), &r));
                    }
                    if !yes {
                        println!("re-run with --yes to apply.");
                    }
                }
            }
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
            let project_dir = project.clone().map(PathBuf::from);
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
                MemoriesAction::Suggest | MemoriesAction::Verify => {
                    let dir = memory_dir().ok();
                    let redactor = ingest::Redactor::from_file(&cfg.redact_file)?;
                    let store = connect_store(&cfg)?;
                    let candidates = if matches!(action, MemoriesAction::Suggest) {
                        let existing = dir.as_deref().map(ctx::memories::existing_memories).unwrap_or_default();
                        let summarizer = ctx::summarizer::ClaudeCli::new(cfg.claude_exe.clone(), &cfg.summary_model, cfg.summarizer_dir.clone(), Duration::from_secs(180));
                        let fresh = tokio::task::block_in_place(|| {
                            tokio::runtime::Handle::current().block_on(ctx::memories::suggest(&store, &summarizer, &redactor, &key, &existing, Duration::from_secs(180)))
                        })?;
                        // Saved unverified first: a failing fact-check must not throw away a
                        // three-minute suggestion run; `ctx memories verify` retries it.
                        ctx::memories::save_pending(&pending, &fresh)?;
                        fresh
                    } else {
                        ctx::memories::load_pending(&pending)
                    };
                    eprintln!("fact-checking {} suggestions against the session transcripts ({})…", candidates.len(), cfg.verify_model);
                    let verifier = ctx::summarizer::ClaudeCli::new(cfg.claude_exe.clone(), &cfg.verify_model, cfg.summarizer_dir.clone(), Duration::from_secs(300));
                    let root = project_dir.clone().unwrap_or_else(|| std::env::current_dir().unwrap_or_default());
                    let repo = ctx::factcheck::Repo::load(&root);
                    let embedder: Box<dyn Embedder> = Box::new(connect_embedder(&cfg).await?);
                    let env = ctx::factcheck::VerifyEnv {
                        store: &store,
                        embedder: Arc::new(Mutex::new(embedder)),
                        verifier: &verifier,
                        redactor: &redactor,
                        project_key: &key,
                        repo: &repo,
                        memory_dir: dir.as_deref(),
                        timeout: Duration::from_secs(300),
                    };
                    let checked = tokio::task::block_in_place(|| tokio::runtime::Handle::current().block_on(ctx::factcheck::verify(&env, candidates)))?;
                    ctx::memories::save_pending(&pending, &checked)?;
                    println!("{}", ctx::memories::format_candidates(&checked));
                    if !checked.is_empty() {
                        println!("\naccept with: ctx memories accept <numbers>   discard with: ctx memories discard [numbers]   (nothing is written until accepted)");
                    }
                }
                MemoriesAction::List => println!("{}", ctx::memories::format_candidates(&ctx::memories::load_pending(&pending))),
                MemoriesAction::Accept { numbers, force } => {
                    let list = ctx::memories::load_pending(&pending);
                    anyhow::ensure!(!numbers.is_empty(), "give the numbers to accept, e.g. `ctx memories accept 1 3`");
                    let mut chosen = Vec::new();
                    for n in &numbers {
                        let c = list.get(n.wrapping_sub(1)).cloned().ok_or_else(|| anyhow::anyhow!("no suggestion {n} (see `ctx memories list`)"))?;
                        if !force {
                            ctx::memories::acceptable(&c).map_err(anyhow::Error::msg)?;
                        }
                        chosen.push(c);
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
                MemoriesAction::Discard { numbers } if numbers.is_empty() => {
                    let _ = std::fs::remove_file(&pending);
                    println!("pending suggestions discarded");
                }
                MemoriesAction::Discard { numbers } => {
                    let rest = ctx::memories::remove_numbers(&ctx::memories::load_pending(&pending), &numbers).map_err(anyhow::Error::msg)?;
                    ctx::memories::save_pending(&pending, &rest)?;
                    println!("discarded {}; {} pending", numbers.len(), rest.len());
                }
            }
            Ok(())
        }
        Command::Mcp => {
            let embedder: Box<dyn Embedder> = Box::new(embed_server::LazyRemote::new(&cfg.embed_url, embed::BGEM3_DIM));
            let ctx = Context { store: Arc::new(connect_store(&cfg)?), embedder: Arc::new(Mutex::new(embedder)) };
            let teams = teams_search(&cfg, &ctx);
            let mut server = mcp::ContextServer::new(ctx, mcp::current_project(), cfg.state_path.clone())
                .with_teams(teams)
                .with_ranking(cfg.search_mode, cfg.recency)
                .with_rerank(cfg.embed_url.clone(), cfg.rerank_pool);
            if let Ok(js) = jira_search(&cfg) {
                server = server.with_jira(js);
            }
            mcp::serve_stdio(server).await
        }
    }
    })
    .await
}

/// Teams search sharing the context's embedder, over the teams collection.
fn teams_search(cfg: &Config, ctx: &Context) -> ctx::teams::TeamsSearch {
    let connect = |collection: &str| {
        Store::connect(&cfg.qdrant_url, cfg.qdrant_api_key.clone(), collection, embed::BGEM3_DIM).expect("Qdrant client construction does not connect")
    };
    ctx::teams::TeamsSearch {
        store: Arc::new(connect(&cfg.teams_collection)),
        meetings_store: Arc::new(connect(&cfg.meetings_collection)),
        embedder: ctx.embedder.clone(),
        db: cfg.teams_db.clone(),
        people: cfg.teams_people.clone(),
        projects: cfg.teams_projects.clone(),
        redact: cfg.redact_file.clone(),
    }
}

/// A read-only Jira search, or a clear error when Jira is not configured.
fn jira_search(cfg: &Config) -> anyhow::Result<ctx::jira::JiraSearch> {
    let jira = cfg
        .jira()
        .ok_or_else(|| anyhow::anyhow!("Jira is not configured; set JIRA_URL and JIRA_TOKEN in {}", Config::env_file().display()))?;
    Ok(ctx::jira::JiraSearch {
        jira,
        projects: cfg.jira_projects.clone(),
        redact: cfg.redact_file.clone(),
        write_allowed: cfg.jira_write,
        write_log: cfg.data_dir.join("jira-writes.log"),
    })
}

/// Collections a prune targets: `(label, collection, id-field)`.
fn prune_targets(source: PruneSource, cfg: &Config) -> Vec<(&'static str, String, &'static str)> {
    let sessions = ("sessions", cfg.collection.clone(), "session_id");
    let teams = ("teams", cfg.teams_collection.clone(), "thread_id");
    let jira = ("jira", cfg.jira_collection.clone(), "issue_key");
    match source {
        PruneSource::Sessions => vec![sessions],
        PruneSource::Teams => vec![teams],
        PruneSource::Jira => vec![jira],
        PruneSource::All => vec![sessions, teams, jira],
    }
}

fn to_target(source: SourceArg) -> Target {
    match source {
        SourceArg::Claude => Target::Claude,
        SourceArg::Opencode => Target::Opencode,
        SourceArg::Teams => Target::Teams,
        SourceArg::Jira => Target::Jira,
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

/// An injector for CLI use (preview, eval), embedding through the daemon.
async fn injector(cfg: &Config) -> anyhow::Result<ctx::inject::Injector> {
    let embedder: Box<dyn Embedder> = Box::new(connect_embedder(cfg).await?);
    let counter = HfTokenizer::from_file(&cfg.model_dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
    let teams = Store::connect(&cfg.qdrant_url, cfg.qdrant_api_key.clone(), &cfg.teams_collection, embed::BGEM3_DIM)?;
    Ok(ctx::inject::Injector::new(
        Arc::new(connect_store(cfg)?),
        Some(Arc::new(teams)),
        Arc::new(Mutex::new(embedder)),
        Arc::new(counter),
        cfg.recency,
        cfg.inject,
        cfg.redact_file.clone(),
    ))
}

async fn context(cfg: &Config) -> anyhow::Result<Context> {
    let embedder: Box<dyn Embedder> = Box::new(connect_embedder(cfg).await?);
    Ok(Context { store: Arc::new(connect_store(cfg)?), embedder: Arc::new(Mutex::new(embedder)) })
}


/// The `ctx eval` handler, split out of `main` so the async main state machine stays small
/// enough not to overflow the stack in debug builds.
#[allow(clippy::too_many_arguments)]
async fn run_eval(
    cfg: Config, file: Option<PathBuf>, save_baseline: bool, check: bool, sweep: bool,
    compare: Vec<String>, rerank_model: Option<String>, rerank_pool: u64, rerank_threads: Option<usize>, target: String,
) -> anyhow::Result<()> {
            use ctx::eval;
    let ctx = context(&cfg).await?;
    if target == "teams" {
        let path = file.unwrap_or_else(|| cfg.eval_file.parent().unwrap_or(std::path::Path::new(".")).join("teams.jsonl"));
        let mut cases = eval::load_cases(&path)?;
        for c in &mut cases { c.target = eval::Target::Teams; }
        let positives = cases.iter().filter(|c| !c.expect.is_empty()).count();
        println!("{} teams cases ({positives} positive, {} negatives) from {}
", cases.len(), cases.len() - positives, path.display());
        let teams = teams_search(&cfg, &ctx);
        let reports = eval::run_teams(&teams, &cases, &eval::teams_variants()).await?;
        println!("{}
", eval::format_retrieval(&reports));
        let production = reports.iter().find(|r| r.name == "fused").expect("fused variant is always run");
        println!("{}", eval::format_misses(production, &cases));
        let vectors = eval::embed_queries(&ctx, &cases).await?;
        let inj = injector(&cfg).await?;
        let points = eval::run_teams_inject(&inj, &cases, &vectors, &[0.45, 0.50, 0.55, 0.60, 0.65, 0.70]).await?;
        println!("\nteams auto-inject (current teams_min_score {:.2}):\n{}\n", cfg.inject.teams_min_score, eval::format_inject(&points));
        let baseline_path = cfg.eval_baseline.with_file_name(format!("{}-teams.json", cfg.eval_baseline.file_stem().and_then(|s| s.to_str()).unwrap_or("eval-baseline")));
        if check {
            let text = std::fs::read_to_string(&baseline_path).with_context(|| format!("no teams baseline at {}; run `ctx eval --target teams --save-baseline` first", baseline_path.display()))?;
            let c = eval::compare(&serde_json::from_str(&text)?, production);
            println!("\nagainst baseline: lost {:?} · gained {:?} · new cases {:?}", c.lost, c.gained, c.new);
            anyhow::ensure!(!c.is_regression(), "teams retrieval regressed against the baseline");
        }
        if save_baseline {
            std::fs::create_dir_all(baseline_path.parent().unwrap_or(std::path::Path::new(".")))?;
            std::fs::write(&baseline_path, serde_json::to_string_pretty(&eval::Baseline::from_report(production))?)?;
            println!("\nteams baseline saved to {}", baseline_path.display());
        }
        return Ok(());
    }
            let path = file.unwrap_or_else(|| cfg.eval_file.clone());
            let cases = eval::load_cases(&path)?;
            let vectors = eval::embed_queries(&ctx, &cases).await?;
            let positives = cases.iter().filter(|c| !c.expect.is_empty()).count();
            println!("{} cases ({positives} with expected sessions, {} negatives) from {}\n", cases.len(), cases.len() - positives, path.display());
            if let Some(model) = rerank_model {
                let which: embed::RerankModel = model.parse().map_err(anyhow::Error::msg)?;
                eprintln!("loading reranker {model} on CPU (first run downloads it)…");
                let cache = cfg.rerank_cache_dir.clone();
                let mut reranker = tokio::task::spawn_blocking(move || embed::FastembedReranker::hub(which, 512, rerank_threads, 32, Some(cache)))
                    .await?
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                // Baseline: dense turns, no summaries — the same pool the reranker reorders.
                let base = eval::Variant { name: "dense-turns", mode: store::SearchMode::Dense, summaries: false, recency: ctx::rank::Recency::OFF };
                let baseline = eval::run_retrieval(&ctx, &cases, &[base], &vectors).await?.remove(0);
                let (reranked, latency) = eval::run_reranked(&ctx, &cases, &vectors, &mut reranker, rerank_pool, "dense+rerank").await?;
                println!("reranker: {model}, pool N={rerank_pool}, threads={}\n", rerank_threads.map_or("all".to_owned(), |t| t.to_string()));
                println!("{}\n", eval::format_retrieval(&[baseline.clone(), reranked.clone()]));
                println!("{}\n", eval::format_compare(&baseline, &reranked, &cases));
                println!("{}", eval::format_latency("rerank added latency", latency));
                return Ok(());
            }
            let all_variants = eval::variants(cfg.search_mode, cfg.recency);
            if let [a, b] = compare.as_slice() {
                let pick = |name: &str| {
                    all_variants
                        .iter()
                        .find(|v| v.name == name)
                        .copied()
                        .ok_or_else(|| anyhow::anyhow!("unknown variant {name:?}; valid: {}", all_variants.iter().map(|v| v.name).collect::<Vec<_>>().join(", ")))
                };
                let two = [pick(a)?, pick(b)?];
                let reports = eval::run_retrieval(&ctx, &cases, &two, &vectors).await?;
                println!("{}", eval::format_compare(&reports[0], &reports[1], &cases));
                return Ok(());
            }
            let reports = eval::run_retrieval(&ctx, &cases, &all_variants, &vectors).await?;
            println!("production = {:?} + summaries, recency weight {:.2}", cfg.search_mode, cfg.recency.weight);
            println!("{}\n", eval::format_retrieval(&reports));
            let production = reports.iter().find(|r| r.name == eval::PRODUCTION).expect("production variant is always run");
            println!("{}\n", eval::format_misses(production, &cases));
            if sweep {
                let mut tries = Vec::new();
                for weight in [0.1, 0.2, 0.3, 0.5] {
                    for half_life_days in [30.0, 60.0, 120.0] {
                        tries.push(ctx::rank::Recency { weight, half_life_days });
                    }
                }
                let mut rows = Vec::new();
                for r in tries {
                    let v = eval::Variant { name: "sweep", mode: cfg.search_mode, summaries: true, recency: r };
                    let rep = eval::run_retrieval(&ctx, &cases, &[v], &vectors).await?.remove(0);
                    rows.push(format!(
                        "weight {:.1} half-life {:>3} d: R@5 {:.2} MRR {:.3} latest {}/{}",
                        r.weight, r.half_life_days, rep.recall_at_5, rep.mrr, rep.latest.0, rep.latest.1
                    ));
                }
                println!("recency sweep (production mode + summaries):\n  {}\n", rows.join("\n  "));
            }
            let injector = injector(&cfg).await?;
            let thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70];
            let points = eval::run_inject(&injector, &cases, &vectors, &thresholds).await?;
            println!("auto-inject (current min_score {:.2}):\n{}\n", cfg.inject.min_score, eval::format_inject(&points));
            // Checked before saving, so `--check --save-baseline` compares against the old one.
            if check {
                let text = std::fs::read_to_string(&cfg.eval_baseline)
                    .with_context(|| format!("no baseline at {}; run `ctx eval --save-baseline` first", cfg.eval_baseline.display()))?;
                let b: eval::Baseline = serde_json::from_str(&text)?;
                let c = eval::compare(&b, production);
                println!("against baseline: lost {:?} · gained {:?} · new cases {:?}", c.lost, c.gained, c.new);
                anyhow::ensure!(!c.is_regression(), "retrieval regressed against the baseline");
            }
            if save_baseline {
                let b = eval::Baseline::from_report(production);
                std::fs::create_dir_all(cfg.eval_baseline.parent().unwrap_or(std::path::Path::new(".")))?;
                std::fs::write(&cfg.eval_baseline, serde_json::to_string_pretty(&b)?)?;
                println!("baseline saved to {}", cfg.eval_baseline.display());
            }
            Ok(())
}