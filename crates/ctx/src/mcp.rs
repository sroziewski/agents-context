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
#[derive(Debug, Deserialize, JsonSchema)]
pub struct TeamsArgs {
    /// What to look for, in Polish or English; matched by meaning and by keywords.
    pub query: String,
    /// A people.json group (e.g. "SEDN"): only its channels.
    pub group: Option<String>,
    /// Substring of the Teams team name (case- and accent-insensitive).
    pub team: Option<String>,
    /// Substring of the channel name.
    pub channel: Option<String>,
    /// Substring of the author's name.
    pub author: Option<String>,
    /// Only messages on or after this date, YYYY-MM-DD.
    pub since: Option<String>,
    /// Only messages on or before this date, YYYY-MM-DD.
    pub until: Option<String>,
    /// Number of threads, 1-20 (default 6).
    pub limit: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct JiraSearchArgs {
    /// Text to match, or a bare issue key (HD-123), or raw JQL when jql=true.
    pub query: String,
    /// Scope to this project directory's mapped Jira keys instead of the current project.
    pub project: Option<String>,
    /// Treat query as raw JQL.
    pub jql: Option<bool>,
    /// Search all projects, ignoring the project→key mapping.
    pub all: Option<bool>,
    /// Number of issues, 1-30 (default 8).
    pub limit: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct JiraIssueArgs {
    /// Issue key, e.g. HD-1410.
    pub key: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct JiraCommentArgs {
    /// Issue key, e.g. HD-1410.
    pub key: String,
    /// Comment text.
    pub body: String,
    /// Must be true to actually send; omitted/false is a dry run.
    pub confirm: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct JiraTransitionArgs {
    /// Issue key, e.g. HD-1410.
    pub key: String,
    /// Target status name (case-insensitive), e.g. "In Progress".
    pub to: String,
    /// Must be true to actually send; omitted/false is a dry run.
    pub confirm: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct TeamsThreadArgs {
    /// Thread id as shown in search_teams results.
    pub thread_id: String,
}

/// Largest `get_teams_thread` answer.
const MAX_THREAD_CHARS: usize = 40_000;
/// Largest `get_meeting` answer. Bigger than a thread: a standup transcript runs long, and the
/// whole meeting is the point of the tool (summary + participants lead, so the tail is what's cut).
const MAX_MEETING_CHARS: usize = 120_000;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct MeetingArgs {
    /// What to look for in the meeting transcripts.
    pub query: String,
    /// Restrict to a speaker (matched loosely, diacritics-insensitive), e.g. "Chmiel".
    pub speaker: Option<String>,
    /// Earliest date, YYYY-MM-DD.
    pub since: Option<String>,
    /// Latest date, YYYY-MM-DD.
    pub until: Option<String>,
    /// Max meetings to return (default 6).
    pub limit: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct MeetingGetArgs {
    /// Meeting id as shown in search_meetings results.
    pub meeting_id: String,
}

#[derive(Clone)]
pub struct ContextServer {
    ctx: Context,
    /// Project key of the directory the agent was started in.
    current_project: String,
    state_path: PathBuf,
    teams: Option<crate::teams::TeamsSearch>,
    mode: store::SearchMode,
    recency: crate::rank::Recency,
    /// Daemon URL + candidate pool for the opt-in `deep_search_sessions`; `None` disables it.
    rerank: Option<(String, u64)>,
    /// Read-only Jira; `None` when JIRA_URL/JIRA_TOKEN are unset.
    jira: Option<crate::jira::JiraSearch>,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl ContextServer {
    pub fn new(ctx: Context, current_project: String, state_path: PathBuf) -> Self {
        Self { ctx, current_project, state_path, teams: None, mode: store::SearchMode::Hybrid, recency: crate::rank::Recency::OFF, rerank: None, jira: None, tool_router: Self::tool_router() }
    }

    /// Enable the read-only Jira tools.
    pub fn with_jira(mut self, jira: crate::jira::JiraSearch) -> Self {
        self.jira = Some(jira);
        self
    }

    /// Enable `deep_search_sessions`, which reranks through the daemon at `embed_url`.
    pub fn with_rerank(mut self, embed_url: String, pool: u64) -> Self {
        self.rerank = Some((embed_url, pool));
        self
    }

    pub fn with_ranking(mut self, mode: store::SearchMode, recency: crate::rank::Recency) -> Self {
        self.mode = mode;
        self.recency = recency;
        self
    }

    pub fn with_teams(mut self, teams: crate::teams::TeamsSearch) -> Self {
        self.teams = Some(teams);
        self
    }

    fn teams(&self) -> Result<&crate::teams::TeamsSearch, String> {
        self.teams.as_ref().ok_or_else(|| "Teams search is not configured".to_owned())
    }

    /// Search the organisation's Microsoft Teams channel messages (shared, not project-scoped),
    /// by meaning and by keywords; results are threads with their best matching message and
    /// author roles. Threads in the current project's groups rank higher. Open one with
    /// get_teams_thread.
    #[tool(name = "search_teams")]
    pub async fn search_teams(&self, Parameters(a): Parameters<TeamsArgs>) -> Result<String, String> {
        let teams = self.teams()?;
        if a.query.trim().is_empty() {
            return Err("query must not be empty".into());
        }
        let q = crate::teams::TeamsQuery {
            text: a.query,
            group: a.group,
            team: a.team,
            channel: a.channel,
            author: a.author,
            since: a.since.as_deref().map(|d| parse_date(d, false)).transpose().map_err(|e| e.to_string())?,
            until: a.until.as_deref().map(|d| parse_date(d, true)).transpose().map_err(|e| e.to_string())?,
            limit: a.limit.unwrap_or(6).clamp(1, 20) as usize,
        };
        let r = teams.search(&q, Some(&self.current_project)).await.map_err(|e| e.to_string())?;
        Ok(crate::teams::format_results(&r))
    }

    /// A whole Teams thread (root and replies) with author roles.
    #[tool(name = "get_teams_thread")]
    pub async fn get_teams_thread(&self, Parameters(a): Parameters<TeamsThreadArgs>) -> Result<String, String> {
        let teams = self.teams()?.clone();
        let id = a.thread_id.trim().to_owned();
        tokio::task::spawn_blocking(move || teams.thread(&id, MAX_THREAD_CHARS))
            .await
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())
    }

    /// Search the organisation's Teams meeting transcripts (standups etc.) by meaning and
    /// keywords; results are meetings with their best matching line and speaker. Filter by
    /// speaker or date. Open the full transcript with get_meeting.
    #[tool(name = "search_meetings")]
    pub async fn search_meetings(&self, Parameters(a): Parameters<MeetingArgs>) -> Result<String, String> {
        let teams = self.teams()?;
        if a.query.trim().is_empty() {
            return Err("query must not be empty".into());
        }
        let q = crate::teams::MeetingQuery {
            text: a.query,
            speaker: a.speaker,
            since: a.since.as_deref().map(|d| parse_date(d, false)).transpose().map_err(|e| e.to_string())?,
            until: a.until.as_deref().map(|d| parse_date(d, true)).transpose().map_err(|e| e.to_string())?,
            limit: a.limit.unwrap_or(6).clamp(1, 20) as usize,
        };
        let r = teams.search_meetings(&q).await.map_err(|e| e.to_string())?;
        Ok(crate::teams::format_meetings(&r))
    }

    /// A whole Teams meeting transcript, every line speaker-labelled (redacted).
    #[tool(name = "get_meeting")]
    pub async fn get_meeting(&self, Parameters(a): Parameters<MeetingGetArgs>) -> Result<String, String> {
        let teams = self.teams()?.clone();
        let id = a.meeting_id.trim().to_owned();
        if id.is_empty() {
            return Err("meeting_id must not be empty".into());
        }
        tokio::task::spawn_blocking(move || teams.meeting(&id, MAX_MEETING_CHARS)).await.map_err(|e| e.to_string())?.map_err(|e| e.to_string())
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
        let mut q = Query::new(a.query, self.scope(a.scope.as_deref(), a.project.as_deref())?, u64::from(a.limit.unwrap_or(6).clamp(1, 20)));
        q.source = source;
        q.since = a.since.as_deref().map(|d| parse_date(d, false)).transpose().map_err(|e| e.to_string())?;
        q.until = a.until.as_deref().map(|d| parse_date(d, true)).transpose().map_err(|e| e.to_string())?;
        q.include_subagents = a.include_subagents.unwrap_or(false);
        q.recency = self.recency;
        q.mode = self.mode;
        let results = self.ctx.search(&q).await.map_err(|e| e.to_string())?;
        Ok(format_results(&results))
    }

    /// Like search_sessions but reorders the matching turns with a cross-encoder reranker for
    /// higher precision. Slower — seconds, not milliseconds — so use it only when the plain
    /// search_sessions ranking is not good enough, not as the default. Same arguments.
    #[tool(name = "deep_search_sessions")]
    pub async fn deep_search_sessions(&self, Parameters(a): Parameters<SearchArgs>) -> Result<String, String> {
        let Some((url, pool)) = self.rerank.clone() else {
            return Err("deep search is not enabled in this daemon; use search_sessions".into());
        };
        if a.query.trim().is_empty() {
            return Err("query must not be empty".into());
        }
        let source = match a.source.as_deref() {
            None => None,
            Some(s @ ("claude" | "opencode")) => Some(s.to_owned()),
            Some(other) => return Err(format!("unknown source {other:?}; use \"claude\" or \"opencode\"")),
        };
        let mut q = Query::new(a.query, self.scope(a.scope.as_deref(), a.project.as_deref())?, u64::from(a.limit.unwrap_or(6).clamp(1, 20)));
        q.source = source;
        q.since = a.since.as_deref().map(|d| parse_date(d, false)).transpose().map_err(|e| e.to_string())?;
        q.until = a.until.as_deref().map(|d| parse_date(d, true)).transpose().map_err(|e| e.to_string())?;
        q.include_subagents = a.include_subagents.unwrap_or(false);
        let reranker: std::sync::Arc<std::sync::Mutex<Box<dyn embed::Reranker>>> =
            std::sync::Arc::new(std::sync::Mutex::new(Box::new(crate::embed_server::RemoteReranker::new(&url))));
        let results = self.ctx.search_reranked(&q, reranker, pool).await.map_err(|e| e.to_string())?;
        Ok(format_results(&results))
    }

    /// Search the organisation's Jira issues (read-only, Jira Data Center). Plain text becomes a
    /// JQL text match; a bare issue key (HD-123) is fetched directly; pass jql=true for raw JQL.
    /// Scoped to the current project's Jira keys by default (set project, or all=true for every
    /// project). Results carry the key and URL so you can cite and open them.
    #[tool(name = "search_jira")]
    pub async fn search_jira(&self, Parameters(a): Parameters<JiraSearchArgs>) -> Result<String, String> {
        let jira = self.jira.clone().ok_or("Jira is not configured (JIRA_URL / JIRA_TOKEN unset)")?;
        if a.query.trim().is_empty() {
            return Err("query must not be empty".into());
        }
        let all = a.all.unwrap_or(false);
        let project_key = if all { None } else { Some(a.project.map_or_else(|| self.current_project.clone(), |p| sources::project::key(&p))) };
        let (jql, limit) = (a.jql.unwrap_or(false), a.limit.unwrap_or(8).clamp(1, 30));
        let query = a.query;
        tokio::task::spawn_blocking(move || jira.search(&query, project_key.as_deref(), jql, all, limit))
            .await
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())
    }

    /// One Jira issue in full — description and comments (redacted), with a link. Read-only.
    #[tool(name = "get_jira_issue")]
    pub async fn get_jira_issue(&self, Parameters(a): Parameters<JiraIssueArgs>) -> Result<String, String> {
        let jira = self.jira.clone().ok_or("Jira is not configured (JIRA_URL / JIRA_TOKEN unset)")?;
        let key = a.key.trim().to_owned();
        if key.is_empty() {
            return Err("key must not be empty".into());
        }
        tokio::task::spawn_blocking(move || jira.issue(&key)).await.map_err(|e| e.to_string())?.map_err(|e| e.to_string())
    }

    /// Add a comment to a Jira issue. A WRITE — off unless the daemon allows Jira writes
    /// (JIRA_WRITE=on) AND you pass confirm=true. Without confirm it is a dry run: it returns
    /// what would be sent and changes nothing. Use it only when the user clearly asked to
    /// comment on the issue.
    #[tool(name = "jira_add_comment")]
    pub async fn jira_add_comment(&self, Parameters(a): Parameters<JiraCommentArgs>) -> Result<String, String> {
        let jira = self.jira.clone().ok_or("Jira is not configured (JIRA_URL / JIRA_TOKEN unset)")?;
        let (key, body, confirm) = (a.key.trim().to_owned(), a.body, a.confirm.unwrap_or(false));
        if key.is_empty() || body.trim().is_empty() {
            return Err("key and body must not be empty".into());
        }
        tokio::task::spawn_blocking(move || jira.comment(&key, &body, confirm)).await.map_err(|e| e.to_string())?.map_err(|e| e.to_string())
    }

    /// Move a Jira issue to a status by name. A WRITE — same gate as jira_add_comment
    /// (JIRA_WRITE=on and confirm=true); dry run otherwise. Only when the user asked to change
    /// the issue's status.
    #[tool(name = "jira_transition_issue")]
    pub async fn jira_transition_issue(&self, Parameters(a): Parameters<JiraTransitionArgs>) -> Result<String, String> {
        let jira = self.jira.clone().ok_or("Jira is not configured (JIRA_URL / JIRA_TOKEN unset)")?;
        let (key, to, confirm) = (a.key.trim().to_owned(), a.to, a.confirm.unwrap_or(false));
        if key.is_empty() || to.trim().is_empty() {
            return Err("key and to must not be empty".into());
        }
        tokio::task::spawn_blocking(move || jira.transition(&key, &to, confirm)).await.map_err(|e| e.to_string())?.map_err(|e| e.to_string())
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
             prefer recent sessions when results conflict. search_teams / get_teams_thread cover the \
             organisation's Teams channels, which are shared by all projects; author roles in brackets \
             come from the team directory. search_meetings / get_meeting cover Teams meeting transcripts \
             (standups etc.), filterable by speaker and date.",
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
