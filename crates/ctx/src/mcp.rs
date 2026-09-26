//! MCP server exposing indexed session history to Claude Code and OpenCode over stdio.

use std::path::PathBuf;

use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{ServerCapabilities, ServerConfig};
use rmcp::{ServerHandler, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::search::{Context, Query, Scope, format_results, format_session, format_sessions, parse_date};

/// Largest `get_session` answer: ~10k tokens, enough for a long session's gist.
const MAX_SESSION_CHARS: usize = 40_000;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchArgs {
    /// What to look for, in Polish or English: a question, topic, error message, file or ticket id.
    pub query: String,
    /// "project" (default) searches only the current project; "all" searches every project.
    pub scope: Option<String>,
    /// Search this project directory instead of the current one (e.g. "D:\\dev\\helpdesk").
    pub project: Option<String>,
    /// Restrict to one agent: "claude" or "opencode".
    pub source: Option<String>,
    /// Only turns on or after this date, YYYY-MM-DD.
    pub since: Option<String>,
    /// Only turns on or before this date, YYYY-MM-DD.
    pub until: Option<String>,
    /// Number of results, 1-20 (default 6).
    pub limit: Option<u32>,
    /// Also search subagent transcripts (exploration runs); default false.
    pub include_subagents: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SessionArgs {
    /// Session id as shown in search results.
    pub session_id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListArgs {
    /// "project" (default) or "all".
    pub scope: Option<String>,
    /// List this project directory instead of the current one.
    pub project: Option<String>,
    /// Number of sessions, default 20.
    pub limit: Option<u32>,
}

#[derive(Clone)]
pub struct ContextServer {
    ctx: Context,
    /// Project key of the directory the agent was started in.
    current_project: String,
    state_path: PathBuf,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl ContextServer {
    pub fn new(ctx: Context, current_project: String, state_path: PathBuf) -> Self {
        Self { ctx, current_project, state_path, tool_router: Self::tool_router() }
    }

    fn scope(&self, scope: Option<&str>, project: Option<&str>) -> Result<Scope, String> {
        match (scope.map(str::to_ascii_lowercase).as_deref(), project) {
            (Some("all"), _) => Ok(Scope::All),
            (None | Some("project"), Some(p)) => Ok(Scope::Project(sources::project::key(p))),
            (None | Some("project"), None) => Ok(Scope::Project(self.current_project.clone())),
            (Some(other), _) => Err(format!("unknown scope {other:?}; use \"project\" or \"all\"")),
        }
    }

    /// Semantic + keyword search over past Claude Code and OpenCode sessions: matching session
    /// summaries first, then matching turns. Defaults to the current project only; pass scope
    /// "all" or a project path to look elsewhere.
    #[tool(name = "search_sessions")]
    pub async fn search_sessions(&self, Parameters(a): Parameters<SearchArgs>) -> Result<String, String> {
        if a.query.trim().is_empty() {
            return Err("query must not be empty".into());
        }
        let source = match a.source.as_deref() {
            None => None,
            Some(s @ ("claude" | "opencode")) => Some(s.to_owned()),
            Some(other) => return Err(format!("unknown source {other:?}; use \"claude\" or \"opencode\"")),
        };
        let q = Query {
            text: a.query,
            scope: self.scope(a.scope.as_deref(), a.project.as_deref())?,
            source,
            since: a.since.as_deref().map(|d| parse_date(d, false)).transpose().map_err(|e| e.to_string())?,
            until: a.until.as_deref().map(|d| parse_date(d, true)).transpose().map_err(|e| e.to_string())?,
            limit: u64::from(a.limit.unwrap_or(6).clamp(1, 20)),
            include_subagents: a.include_subagents.unwrap_or(false),
        };
        let results = self.ctx.search(&q).await.map_err(|e| e.to_string())?;
        Ok(format_results(&results))
    }

    /// Full text of one indexed session, in order. Use after search_sessions to read around a hit.
    #[tool(name = "get_session")]
    pub async fn get_session(&self, Parameters(a): Parameters<SessionArgs>) -> Result<String, String> {
        let hits = self.ctx.session(a.session_id.trim()).await.map_err(|e| e.to_string())?;
        Ok(format_session(&hits, MAX_SESSION_CHARS))
    }

    /// Recent sessions (date, agent, title, id), newest first. Defaults to the current project.
    #[tool(name = "list_sessions")]
    pub async fn list_sessions(&self, Parameters(a): Parameters<ListArgs>) -> Result<String, String> {
        let scope = self.scope(a.scope.as_deref(), a.project.as_deref())?;
        let list = self.ctx.sessions(&scope, a.limit.unwrap_or(20).clamp(1, 200) as usize).await.map_err(|e| e.to_string())?;
        Ok(format_sessions(&list))
    }

    /// Indexed projects with session and chunk counts, and which one is current.
    #[tool(name = "list_projects")]
    pub async fn list_projects(&self) -> Result<String, String> {
        let state = ingest::State::open(&self.state_path).map_err(|e| e.to_string())?;
        let rows = state.per_project().map_err(|e| e.to_string())?;
        let mut out = format!("current project: {}\n", self.current_project);
        for (project, sessions, chunks) in rows.into_iter().filter(|(p, ..)| !p.is_empty()) {
            let mark = if project == self.current_project { " (current)" } else { "" };
            out.push_str(&format!("{project}{mark}: {sessions} sessions, {chunks} chunks\n"));
        }
        Ok(out.trim_end().to_owned())
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for ContextServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build()).with_instructions(format!(
            "Memory of past Claude Code and OpenCode sessions (prompts, answers, tool summaries). \
             Current project: {}. Searches stay inside the current project unless you pass scope \"all\" \
             or another project path — only do that when the user asks about other projects. \
             Results cite session id, agent and date: quote them when you rely on a result, and \
             prefer recent sessions when results conflict.",
            self.current_project
        ))
    }
}

pub async fn serve_stdio(server: ContextServer) -> anyhow::Result<()> {
    use rmcp::ServiceExt;
    let running = server.serve(rmcp::transport::stdio()).await?;
    running.waiting().await?;
    Ok(())
}

/// Project key for the MCP process: `CTX_PROJECT` if set, else the working directory the agent
/// launched it in.
pub fn current_project() -> String {
    let dir = std::env::var("CTX_PROJECT")
        .ok()
        .filter(|p| !p.is_empty())
        .or_else(|| std::env::current_dir().ok().map(|d| d.to_string_lossy().into_owned()))
        .unwrap_or_default();
    sources::project::key(&dir)
}
