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
        /// UserPromptSubmit: print matching past-session context for the prompt instead of
        /// indexing (only while auto-inject is on, see `ctx inject`).
        #[arg(long)]
        inject: bool,
    },
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
    /// Turn auto-inject on (takes effect on the next prompt).
    On,
    /// Turn it off.
    Off,
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
            embed_server::serve(embed_server::daemon_router(shared, &model, Some(queue), Some(injector)), addr).await
        }
        Command::Hook { inject: true, .. } => {
            let payload = hook::read_stdin();
            if cfg.inject.enabled {
                let url = cfg.embed_url.clone();
                let out = tokio::task::spawn_blocking(move || hook::inject(&url, &payload, Duration::from_secs(3), |k| std::env::var(k).ok())).await;
                if let Ok(Some(json)) = out {
                    println!("{json}");
                }
            }
            Ok(())
        }
        Command::Inject { action } => {
            let env_file = Config::env_file();
            match action {
                InjectAction::On | InjectAction::Off => {
                    let on = matches!(action, InjectAction::On);
                    ctx::config::set_env_value(&env_file, "CTX_INJECT", if on { "on" } else { "off" })?;
                    println!("auto-inject {} ({})", if on { "on" } else { "off" }, env_file.display());
                    if on {
                        println!("it needs the UserPromptSubmit hook in ~/.claude/settings.json: `ctx.exe hook --inject`");
                    }
                }
                InjectAction::Status => {
                    let i = cfg.inject;
                    println!(
                        "auto-inject {} · teams {} · min_score {:.2} · max {} tokens · recency weight {:.2}, half-life {} days",
                        if i.enabled { "on" } else { "off" },
                        if i.teams { "on" } else { "off" },
                        i.min_score,
                        i.max_tokens,
                        cfg.recency.weight,
                        cfg.recency.half_life_days
                    );
                }
                InjectAction::Preview { prompt } => {
                    let injector = injector(&cfg).await?;
                    let req = ctx::inject::ContextRequest {
                        prompt,
                        cwd: std::env::current_dir()?.to_string_lossy().into_owned(),
                        session_id: String::new(),
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
        Command::Eval { file, save_baseline, check, sweep } => {
            use ctx::eval;
            let path = file.unwrap_or_else(|| cfg.eval_file.clone());
            let cases = eval::load_cases(&path)?;
            let ctx = context(&cfg).await?;
            let vectors = eval::embed_queries(&ctx, &cases).await?;
            let positives = cases.iter().filter(|c| !c.expect.is_empty()).count();
            println!("{} cases ({positives} with expected sessions, {} negatives) from {}\n", cases.len(), cases.len() - positives, path.display());
            let reports = eval::run_retrieval(&ctx, &cases, &eval::variants(cfg.search_mode, cfg.recency), &vectors).await?;
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
        Command::Hook { source, .. } => {
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
            let mut q = Query::new(query, scope(all, project), limit);
            q.since = since.as_deref().map(|d| parse_date(d, false)).transpose()?;
            q.until = until.as_deref().map(|d| parse_date(d, true)).transpose()?;
            q.include_subagents = subagents;
            q.recency = cfg.recency;
            q.mode = cfg.search_mode;
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
            let server = mcp::ContextServer::new(ctx, mcp::current_project(), cfg.state_path.clone()).with_teams(teams).with_ranking(cfg.search_mode, cfg.recency);
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
