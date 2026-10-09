//! Teams search: semantic hits from the teams collection fused with the app's own FTS5 index,
//! ranked per thread, authors labelled from people.json.
//!
//! Filters resolve against teams.db with the app's folding rules, so `search_teams` narrows
//! exactly like the app's `search_messages` does. Either half may be unavailable (no teams
//! collection yet, embedding daemon down, a query FTS5 cannot parse); the search then answers
//! from the other half and says so.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use embed::Embedder;
use indexmap::IndexMap;
use ingest::Redactor;
use serde::Deserialize;
use serde_json::{Map, Value};
use sources::teams::{CHAT_TEAM, Filter, FtsHit, Meeting, TeamsDb, Thread, TranscriptFilter, TranscriptHit, name_key};
use store::{SearchFilter, SearchMode, Store};

/// Reciprocal rank fusion constant: the usual 60, so neither source dominates on rank 1.
const RRF_K: f64 = 60.0;
/// Score multiplier for threads in the current project's mapped groups (DEV-20): enough to lift
/// a comparable thread above another project's, not enough to bury a clearly better match.
pub const PROJECT_BOOST: f64 = 1.5;
/// Candidates taken from each source before fusion, per result asked for.
const DEPTH: usize = 5;
const EVIDENCE_CHARS: usize = 500;
/// How far around a meeting its chat is read for `get_meeting`: links and notes are often posted
/// just before it starts or right after it ends.
const MEETING_CHAT_MARGIN_MS: i64 = 15 * 60 * 1000;
/// One transcript page of `get_meeting`: ~8k tokens, under Claude Code's 10k MCP-output warning.
pub const TRANSCRIPT_PAGE_CHARS: usize = 30_000;
/// `get_meeting` at a cited `[mm:ss]`: a minute of lead-in, two minutes after.
const AROUND_BEFORE_S: i64 = 60;
const AROUND_AFTER_S: i64 = 120;

/// What `get_meeting` returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeetingView {
    /// Participants, the AI summary, the meeting chat; the transcript only when there is no summary.
    Summary,
    /// The transcript around this many seconds into the meeting.
    Around { offset_s: i64 },
    /// The transcript, one page from this line (0-based).
    Transcript { from_line: usize },
}

/// "39:20", "1:02:03" or plain seconds "2360" → seconds into the meeting.
pub fn parse_offset(s: &str) -> anyhow::Result<i64> {
    let parts: Vec<&str> = s.trim().trim_matches(|c| c == '[' || c == ']').split(':').collect();
    let nums: Option<Vec<i64>> = parts.iter().map(|p| p.trim().parse::<i64>().ok().filter(|n| *n >= 0)).collect();
    match nums.as_deref() {
        Some([s]) => Ok(*s),
        Some([m, s]) if *s < 60 => Ok(m * 60 + s),
        Some([h, m, s]) if *m < 60 && *s < 60 => Ok(h * 3600 + m * 60 + s),
        _ => anyhow::bail!("invalid time {s:?}; use mm:ss as cited in the summary, e.g. \"39:20\""),
    }
}

fn mmss(secs: i64) -> String {
    let secs = secs.max(0);
    if secs >= 3600 { format!("{}:{:02}:{:02}", secs / 3600, secs / 60 % 60, secs % 60) } else { format!("{:02}:{:02}", secs / 60, secs % 60) }
}

/// Appends transcript lines from `from_line` until the page is full, then where to continue.
fn page(out: &mut String, lines: &[sources::teams::TranscriptLine], from_line: usize, render: &dyn Fn(&sources::teams::TranscriptLine) -> String) {
    if from_line >= lines.len() && !lines.is_empty() {
        out.push_str(&format!("(from_line {from_line} is past the end: {} lines)", lines.len()));
        return;
    }
    let mut used = 0;
    for (i, l) in lines.iter().enumerate().skip(from_line) {
        let line = render(l);
        if used + line.len() > TRANSCRIPT_PAGE_CHARS && used > 0 {
            out.push_str(&format!("[lines {from_line}–{} of {}; continue with transcript=true, from_line={i}]", i - 1, lines.len()));
            return;
        }
        used += line.len();
        out.push_str(&line);
    }
}

/// people.json. Insertion-ordered like the app's objects, so a person's tags come out in the
/// same order as in the app.
#[derive(Debug, Default, Deserialize)]
pub struct Directory {
    #[serde(default)]
    pub groups: IndexMap<String, Group>,
    #[serde(default)]
    pub people: IndexMap<String, Person>,
}

#[derive(Debug, Default, Deserialize)]
pub struct Group {
    #[serde(default)]
    pub channels: Vec<String>,
    #[serde(default)]
    pub roles: IndexMap<String, String>,
    #[serde(default)]
    pub members: Vec<String>,
}

#[derive(Debug, Default, Deserialize)]
pub struct Person {
    pub title: Option<String>,
    #[serde(default)]
    pub vip: bool,
}

impl Directory {
    /// A missing file is an empty directory, as in the app.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(text) => Ok(serde_json::from_str(text.trim_start_matches('\u{feff}'))?),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e.into()),
        }
    }

    /// The app's `People` tags: VIP and title first, then group roles, then memberships (a
    /// member listed with a role in the same group, by exact name, is not tagged twice).
    pub fn tags(&self, name: &str) -> Vec<String> {
        let key = name_key(name);
        let mut tags = Vec::new();
        for (n, p) in &self.people {
            if name_key(n) == key {
                match (p.vip, &p.title) {
                    (true, Some(t)) => tags.push(format!("VIP, {t}")),
                    (true, None) => tags.push("VIP".into()),
                    (false, Some(t)) => tags.push(t.clone()),
                    (false, None) => {}
                }
            }
        }
        for (group, g) in &self.groups {
            for (n, role) in &g.roles {
                if name_key(n) == key {
                    tags.push(format!("{group} {role}"));
                }
            }
            for n in &g.members {
                if name_key(n) == key && !g.roles.contains_key(n) {
                    tags.push(format!("{group} member"));
                }
            }
        }
        tags
    }

    /// "Tomasz Chmiel [Technet leader]"
    pub fn label(&self, name: &str) -> String {
        let t = self.tags(name);
        if t.is_empty() { name.to_owned() } else { format!("{name} [{}]", t.join("; ")) }
    }

    pub fn group(&self, name: &str) -> Option<(&str, &[String])> {
        self.groups.iter().find(|(g, _)| name_key(g) == name_key(name)).map(|(g, v)| (g.as_str(), v.channels.as_slice()))
    }
}

/// `teams-projects.json`: project directory → Teams group names.
pub fn load_project_groups(path: &Path) -> anyhow::Result<HashMap<String, Vec<String>>> {
    match std::fs::read_to_string(path) {
        Ok(text) => {
            let raw: HashMap<String, Vec<String>> = serde_json::from_str(text.trim_start_matches('\u{feff}'))?;
            // normalise, not key: see `jira::load_project_keys`.
            Ok(raw.into_iter().map(|(dir, groups)| (sources::project::normalise(&dir), groups)).collect())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(HashMap::new()),
        Err(e) => Err(e.into()),
    }
}

/// Which conversations a Teams search covers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Conversations {
    #[default]
    All,
    /// Only the user's private chats (1:1, group and meeting chats).
    Chats,
    /// Only team channels.
    Channels,
}

impl Conversations {
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "all" => Ok(Self::All),
            "chats" | "chat" => Ok(Self::Chats),
            "channels" | "channel" => Ok(Self::Channels),
            other => anyhow::bail!("unknown kind {other:?}; use \"chats\", \"channels\" or \"all\""),
        }
    }

    fn only_chats(self) -> Option<bool> {
        match self {
            Self::All => None,
            Self::Chats => Some(true),
            Self::Channels => Some(false),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct TeamsQuery {
    pub text: String,
    pub kind: Conversations,
    pub group: Option<String>,
    pub team: Option<String>,
    pub channel: Option<String>,
    pub author: Option<String>,
    pub since: Option<i64>,
    pub until: Option<i64>,
    pub limit: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ThreadHit {
    pub thread_id: String,
    pub score: f64,
    pub boosted: bool,
    pub team: String,
    pub channel: String,
    pub message_count: usize,
    pub last_ts: i64,
    /// Best matching message: author (labelled), time, text (redacted).
    pub author: String,
    pub ts: i64,
    pub text: String,
}

/// The knobs of Teams ranking, so the eval (DEV-30) can compare fusion choices without copying
/// the fusion code. [`TeamsRanking::default`] is the shipped behaviour.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TeamsRanking {
    /// Include the semantic (Qdrant) half.
    pub use_semantic: bool,
    /// How the semantic half searches Qdrant (Hybrid today; Dense may suit Polish better).
    pub semantic_mode: SearchMode,
    /// RRF constant fusing the semantic and keyword ranks.
    pub rrf_k: f64,
    /// Project-group score multiplier (DEV-20).
    pub boost: f64,
    /// Include the FTS5 keyword half.
    pub use_fts: bool,
}

impl Default for TeamsRanking {
    fn default() -> Self {
        // Dense semantic half: DEV-30 measured fused-dense MRR 0.942 vs 0.936 for hybrid on the
        // Teams set — bge-m3's sparse weights don't help Polish, and the FTS half already covers
        // keywords. k=60 and boost 1.5 kept: the eval showed no difference from changing them
        // (re-check once the set has boost-discriminating cases).
        Self { use_semantic: true, semantic_mode: SearchMode::Dense, rrf_k: RRF_K, boost: PROJECT_BOOST, use_fts: true }
    }
}

#[derive(Debug, Clone, Default)]
pub struct TeamsResults {
    pub hits: Vec<ThreadHit>,
    /// Groups whose channels were ranked higher for the current project.
    pub boosted_groups: Vec<String>,
    /// Half of the search that was unavailable, and why.
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct MeetingQuery {
    pub text: String,
    /// Restrict to a speaker (folded substring); only narrows the keyword half.
    pub speaker: Option<String>,
    pub since: Option<i64>,
    pub until: Option<i64>,
    pub limit: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeetingHit {
    pub meeting_id: String,
    pub title: String,
    pub score: f64,
    pub started_at: i64,
    pub ended_at: i64,
    /// Best matching line: speaker (None for a system line), time, text (redacted).
    pub speaker: Option<String>,
    pub ts: i64,
    pub text: String,
}

#[derive(Debug, Clone, Default)]
pub struct MeetingResults {
    pub hits: Vec<MeetingHit>,
    pub notes: Vec<String>,
}

#[derive(Clone)]
pub struct TeamsSearch {
    /// The `teams` collection: channel threads/messages.
    pub store: Arc<Store>,
    /// The `meetings` collection: meeting transcript/summary/card chunks (searched by
    /// `search_meetings`), kept separate so it never mixes with channel search.
    pub meetings_store: Arc<Store>,
    pub embedder: Arc<Mutex<Box<dyn Embedder>>>,
    pub db: PathBuf,
    pub people: PathBuf,
    pub projects: PathBuf,
    /// The user's literal redaction list. Results carry raw teams.db text (keyword hits, whole
    /// threads), so they are redacted on the way out, not only at indexing.
    pub redact: PathBuf,
}

struct Evidence {
    author: String,
    ts: i64,
    text: String,
}

/// Everything read from teams.db for one search, gathered on a blocking thread.
struct Local {
    dir: Directory,
    channel_ids: Option<Vec<String>>,
    authors: Option<Vec<String>>,
    fts: Result<Vec<FtsHit>, String>,
    boost_channels: HashSet<String>,
    boosted_groups: Vec<String>,
}

impl TeamsSearch {
    pub async fn search(&self, q: &TeamsQuery, current_project: Option<&str>) -> anyhow::Result<TeamsResults> {
        self.search_with(q, current_project, TeamsRanking::default()).await
    }

    /// [`Self::search`] with an explicit ranking, used by the eval to compare variants.
    pub async fn search_with(&self, q: &TeamsQuery, current_project: Option<&str>, ranking: TeamsRanking) -> anyhow::Result<TeamsResults> {
        anyhow::ensure!(self.db.is_file(), "Teams is not set up: {} does not exist", self.db.display());
        let redactor = Redactor::from_file(&self.redact)?;
        let depth = q.limit.max(1) * DEPTH;

        let (this, query, project) = (self.clone(), q.clone(), current_project.map(str::to_owned));
        let local = tokio::task::spawn_blocking(move || this.read_local(&query, project.as_deref(), depth)).await??;

        // Semantic half; unavailable is not fatal.
        let mut notes = Vec::new();
        let semantic = if !ranking.use_semantic {
            Vec::new()
        } else {
            match self.semantic(q, &local, depth, ranking.semantic_mode).await {
                Ok(hits) => hits,
                Err(e) => {
                    notes.push(format!("semantic search unavailable ({e}); keyword matches only"));
                    Vec::new()
                }
            }
        };
        let fts = if !ranking.use_fts {
            Vec::new()
        } else {
            match local.fts {
                Ok(hits) => hits,
                Err(e) => {
                    notes.push(format!("keyword search unavailable ({e}); semantic matches only"));
                    Vec::new()
                }
            }
        };

        let mut score: HashMap<String, f64> = HashMap::new();
        let mut evidence: HashMap<String, Evidence> = HashMap::new();
        let mut stored: HashMap<String, Map<String, Value>> = HashMap::new();
        // Ranks count threads, not chunks: a thread with five matching chunks must not push the
        // next thread down to rank six.
        let mut rank = 0usize;
        let mut seen: HashSet<String> = HashSet::new();
        for h in &semantic {
            let p = &h.payload;
            let tid = string(p, "thread_id");
            if seen.insert(tid.clone()) {
                rank += 1;
                *score.entry(tid.clone()).or_default() += 1.0 / (ranking.rrf_k + rank as f64);
                stored.insert(tid.clone(), p.clone());
            }
            if string(p, "kind") == "message" && !evidence.contains_key(&tid) {
                evidence.insert(tid, Evidence { author: string(p, "author"), ts: int(p, "ts"), text: message_body(&string(p, "text")) });
            }
        }
        let mut rank = 0usize;
        let mut seen: HashSet<String> = HashSet::new();
        for h in &fts {
            if seen.insert(h.thread_id.clone()) {
                rank += 1;
                *score.entry(h.thread_id.clone()).or_default() += 1.0 / (ranking.rrf_k + rank as f64);
            }
            evidence.entry(h.thread_id.clone()).or_insert_with(|| Evidence { author: h.author.clone(), ts: h.ts, text: h.text.clone() });
        }

        let ids: HashSet<String> = score.keys().cloned().collect();
        let db = self.db.clone();
        let threads: HashMap<String, Thread> = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
            if ids.is_empty() {
                return Ok(HashMap::new());
            }
            let ids: Vec<String> = ids.into_iter().collect();
            Ok(TeamsDb::open(&db)?.threads_by_ids(&ids)?.into_iter().map(|t| (t.id.clone(), t)).collect())
        })
        .await??;

        let mut hits: Vec<ThreadHit> = score
            .into_iter()
            .filter_map(|(tid, s)| {
                // A thread no longer in teams.db (past the app's lookback) is kept in the index
                // on purpose; it is described from its stored point instead of being dropped.
                let (team, channel, channel_id, message_count, last_ts, fallback) = match threads.get(&tid) {
                    Some(t) => {
                        let m = &t.messages[0];
                        (
                            t.channel.team.clone(),
                            t.channel.name.clone(),
                            t.channel.id.clone(),
                            t.messages.len(),
                            t.last_ts(),
                            Evidence { author: m.author.clone(), ts: m.ts, text: m.text.clone() },
                        )
                    }
                    None => {
                        let p = stored.get(&tid)?;
                        let text = message_body(&string(p, "text"));
                        (
                            string(p, "team"),
                            string(p, "channel"),
                            string(p, "channel_id"),
                            p.get("message_count").and_then(Value::as_u64).unwrap_or(1) as usize,
                            int(p, "ts"),
                            Evidence { author: string(p, "author"), ts: int(p, "ts"), text },
                        )
                    }
                };
                let boosted = local.boost_channels.contains(&channel_id);
                let ev = evidence.remove(&tid).unwrap_or(fallback);
                Some(ThreadHit {
                    thread_id: tid,
                    score: if boosted { s * ranking.boost } else { s },
                    boosted,
                    team,
                    channel,
                    message_count,
                    last_ts,
                    author: local.dir.label(&ev.author),
                    ts: ev.ts,
                    text: redactor.scrub(&ev.text).0,
                })
            })
            .collect();
        hits.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| b.last_ts.cmp(&a.last_ts)));
        hits.truncate(q.limit.max(1));
        Ok(TeamsResults { hits, boosted_groups: local.boosted_groups, notes })
    }

    fn read_local(&self, q: &TeamsQuery, current_project: Option<&str>, depth: usize) -> anyhow::Result<Local> {
        let dir = Directory::load(&self.people)?;
        let db = TeamsDb::open(&self.db)?;
        let group_paths: Option<Vec<String>> = match &q.group {
            None => None,
            Some(g) => {
                let (_, channels) = dir.group(g).ok_or_else(|| {
                    anyhow::anyhow!("unknown group {g:?}; known groups: {}", dir.groups.keys().cloned().collect::<Vec<_>>().join(", "))
                })?;
                Some(channels.to_vec())
            }
        };
        let chats = q.kind.only_chats();
        let channel_ids = (q.team.is_some() || q.channel.is_some() || group_paths.is_some() || chats.is_some())
            .then(|| db.resolve_channels(q.team.as_deref(), q.channel.as_deref(), group_paths.as_deref(), chats))
            .transpose()?;
        let authors = q.author.as_deref().map(|a| db.resolve_authors(a)).transpose()?;
        let fts = db
            .fts(&q.text, &Filter { channel_ids: channel_ids.clone(), authors: authors.clone(), since: q.since, until: q.until }, depth)
            .map_err(|e| e.to_string());

        // DEV-20: the current project's groups rank higher; nothing is filtered out.
        let mut boost_channels = HashSet::new();
        let mut boosted_groups = Vec::new();
        if let Some(project) = current_project
            && let Some(groups) = load_project_groups(&self.projects)?.get(project)
        {
            for g in groups {
                if let Some((name, paths)) = dir.group(g) {
                    boost_channels.extend(db.resolve_channels(None, None, Some(paths), None)?);
                    boosted_groups.push(name.to_owned());
                }
            }
        }
        Ok(Local { dir, channel_ids, authors, fts, boost_channels, boosted_groups })
    }

    async fn semantic(&self, q: &TeamsQuery, local: &Local, depth: usize, mode: SearchMode) -> anyhow::Result<Vec<store::Hit>> {
        // Thread chunks carry the thread's last message time as `ts`, so a date window selects
        // threads active in it; message chunks carry their own time. Restrict to channel kinds so
        // any stray meeting chunk in the collection can never surface as an id-less "thread" hit.
        let mut f = SearchFilter::default().between(q.since, q.until).any_of("kind", vec!["thread".to_owned(), "message".to_owned()]);
        if let Some(ids) = &local.channel_ids {
            f = f.any_of("channel_id", ids.clone());
        }
        if let Some(a) = &local.authors {
            // Author is only on message chunks, so an author filter also narrows to those.
            f = f.any_of("author", a.clone());
        }
        let embedder = self.embedder.clone();
        let text = q.text.clone();
        let query = tokio::task::spawn_blocking(move || embedder.lock().unwrap_or_else(|p| p.into_inner()).embed_one(&text)).await??;
        Ok(self.store.search(&query, &f, depth as u64, mode).await?)
    }

    /// The whole thread, authors labelled, redacted, capped at `max_chars`.
    pub fn thread(&self, thread_id: &str, max_chars: usize) -> anyhow::Result<String> {
        let dir = Directory::load(&self.people)?;
        let redactor = Redactor::from_file(&self.redact)?;
        let db = TeamsDb::open(&self.db)?;
        let t = db.thread(thread_id)?;
        let mut out = format!("{} · thread {} · {} messages\n", place(&t.channel.team, &t.channel.name), t.id, t.messages.len());
        if t.channel.is_chat() {
            // A chat is stored one thread per day, so a day alone often starts mid-conversation.
            let days = db.channel_thread_ids(&t.channel.id)?;
            if let Some(i) = days.iter().position(|d| *d == t.id) {
                let mut nav = Vec::new();
                if let Some(prev) = i.checked_sub(1).and_then(|p| days.get(p)) {
                    nav.push(format!("previous day: thread {prev}"));
                }
                if let Some(next) = days.get(i + 1) {
                    nav.push(format!("next day: thread {next}"));
                }
                if !nav.is_empty() {
                    out.push_str(&format!("(private chat, one thread per day; {})\n", nav.join(" · ")));
                }
            }
        }
        out.push('\n');
        for (i, m) in t.messages.iter().enumerate() {
            let line = format!("{} ({}): {}\n\n", dir.label(&m.author), datetime(m.ts), redactor.scrub(m.text.trim()).0);
            if out.len() + line.len() > max_chars {
                out.push_str(&format!("[truncated: {} of {} messages shown]", i, t.messages.len()));
                return Ok(out);
            }
            out.push_str(&line);
        }
        Ok(out.trim_end().to_owned())
    }

    /// Search meeting transcripts: semantic (Qdrant `transcript`/`meeting` chunks) fused with
    /// transcript FTS (speaker-filterable) per meeting. Either half may be unavailable; the
    /// search then answers from the other and says so.
    pub async fn search_meetings(&self, q: &MeetingQuery) -> anyhow::Result<MeetingResults> {
        anyhow::ensure!(self.db.is_file(), "Teams is not set up: {} does not exist", self.db.display());
        let redactor = Redactor::from_file(&self.redact)?;
        let limit = q.limit.max(1);
        let depth = limit * DEPTH;

        // Local half: FTS lines + meeting metadata, gathered on a blocking thread.
        let (db_path, text, speaker, since, until) = (self.db.clone(), q.text.clone(), q.speaker.clone(), q.since, q.until);
        let (fts, meetings) = tokio::task::spawn_blocking(move || -> anyhow::Result<(Result<Vec<TranscriptHit>, String>, HashMap<String, Meeting>)> {
            let db = TeamsDb::open(&db_path)?;
            let speakers = speaker.as_deref().map(|s| db.resolve_speakers(s)).transpose()?;
            let filter = TranscriptFilter { speakers, meeting_ids: None, since, until };
            let fts = db.transcript_fts(&text, &filter, depth).map_err(|e| e.to_string());
            let meetings = db.meetings()?.into_iter().map(|m| (m.id.clone(), m)).collect();
            Ok((fts, meetings))
        })
        .await??;

        let mut notes = Vec::new();
        let semantic = match self.semantic_meetings(q, depth).await {
            Ok(h) => h,
            Err(e) => {
                notes.push(format!("semantic search unavailable ({e}); keyword matches only"));
                Vec::new()
            }
        };
        let fts = match fts {
            Ok(h) => h,
            Err(e) => {
                notes.push(format!("keyword search unavailable ({e}); semantic matches only"));
                Vec::new()
            }
        };

        let mut score: HashMap<String, f64> = HashMap::new();
        // Evidence per meeting: an FTS line (its speaker/time/text) is preferred; a semantic
        // chunk is the fallback for a meeting that matched only semantically.
        let mut evidence: HashMap<String, (Option<String>, i64, String)> = HashMap::new();

        let mut rank = 0usize;
        let mut seen: HashSet<String> = HashSet::new();
        for h in &semantic {
            let mid = string(&h.payload, "meeting_id");
            if mid.is_empty() {
                continue;
            }
            if seen.insert(mid.clone()) {
                rank += 1;
                *score.entry(mid.clone()).or_default() += 1.0 / (RRF_K + rank as f64);
            }
            evidence.entry(mid).or_insert((None, int(&h.payload, "ts"), strip_header(&string(&h.payload, "text"))));
        }
        let mut rank = 0usize;
        let mut seen: HashSet<String> = HashSet::new();
        for h in &fts {
            if seen.insert(h.meeting_id.clone()) {
                rank += 1;
                *score.entry(h.meeting_id.clone()).or_default() += 1.0 / (RRF_K + rank as f64);
                // Best (first) BM25 line for this meeting, overriding any semantic fallback.
                evidence.insert(h.meeting_id.clone(), (h.speaker.clone(), h.ts, h.text.clone()));
            }
        }

        let mut hits: Vec<MeetingHit> = score
            .into_iter()
            .filter_map(|(mid, s)| {
                let m = meetings.get(&mid)?;
                let (speaker, ts, text) = evidence.remove(&mid).unwrap_or((None, m.started_at, String::new()));
                Some(MeetingHit {
                    meeting_id: mid,
                    title: m.title.clone(),
                    score: s,
                    started_at: m.started_at,
                    ended_at: m.ended_at,
                    speaker,
                    ts,
                    text: redactor.scrub(&text).0,
                })
            })
            .collect();
        hits.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| b.started_at.cmp(&a.started_at)));
        hits.truncate(limit);
        Ok(MeetingResults { hits, notes })
    }

    async fn semantic_meetings(&self, q: &MeetingQuery, depth: usize) -> anyhow::Result<Vec<store::Hit>> {
        // Dense (DEV-30: dense ≥ hybrid for Polish Teams text); meeting chunk kinds only.
        let f = SearchFilter::default()
            .between(q.since, q.until)
            .any_of("kind", vec!["transcript".to_owned(), "meeting".to_owned(), "summary".to_owned()]);
        let embedder = self.embedder.clone();
        let text = q.text.clone();
        let query = tokio::task::spawn_blocking(move || embedder.lock().unwrap_or_else(|p| p.into_inner()).embed_one(&text)).await??;
        Ok(self.meetings_store.search(&query, &f, depth as u64, SearchMode::Dense).await?)
    }

    /// One meeting, in the requested [`MeetingView`], redacted.
    ///
    /// The default view is the AI summary, not the transcript (DEV-60): a standup transcript runs to
    /// ~35k tokens, while what was discussed, decided and assigned is in the summary, checked
    /// against the transcripts to be faithful (its `[mm:ss]` citations all land inside the meeting,
    /// and it reconstructs task ids the speech-to-text splits up). The transcript stays the source
    /// for exact wording, reached cheaply through a cited `[mm:ss]` or page by page.
    pub fn meeting(&self, meeting_id: &str, view: MeetingView) -> anyhow::Result<String> {
        let redactor = Redactor::from_file(&self.redact)?;
        let db = TeamsDb::open(&self.db)?;
        let m = db.meeting(meeting_id)?;
        let lines = db.transcript(meeting_id)?;
        let offset = |l: &sources::teams::TranscriptLine| l.offset_s.unwrap_or((l.ts - m.started_at) / 1000).max(0);
        let render_line = |l: &sources::teams::TranscriptLine| {
            format!("[{}] {}: {}\n", mmss(offset(l)), l.speaker.as_deref().unwrap_or("—"), redactor.scrub(l.text.trim()).0)
        };
        let mut out = format!("{} · {} – {} · meeting {} · {} transcript lines\n", m.title, datetime(m.started_at), datetime(m.ended_at), m.id, lines.len());
        let summary = m.summary.as_deref().filter(|_| m.has_summary());

        match view {
            MeetingView::Summary => {
                let participants = sources::teams::participants(&lines);
                if !participants.is_empty() {
                    out.push_str(&format!("Uczestnicy: {}\n", participants.join(", ")));
                }
                if let Some(sum) = summary {
                    out.push_str(&format!(
                        "\nPodsumowanie (AI-written; [mm:ss] point into the transcript):\n{}\n",
                        redactor.scrub(sum.trim()).0
                    ));
                }
                // The chat is where the links and ticket ids mentioned aloud were actually posted.
                let chat = db.chat_messages(&m.chat, m.started_at - MEETING_CHAT_MARGIN_MS, m.ended_at + MEETING_CHAT_MARGIN_MS)?;
                if !chat.is_empty() {
                    out.push_str(&format!("\nCzat spotkania ({}):\n", m.chat));
                    for c in &chat {
                        out.push_str(&format!("{} ({}): {}\n", c.author, datetime(c.ts), redactor.scrub(c.text.trim()).0));
                    }
                }
                if summary.is_some() {
                    out.push_str(&format!(
                        "\nTranskrypcja: {} lines, not shown. To quote or check exact wording, call get_meeting with \
                         at=\"mm:ss\" (that moment) or transcript=true (paged); don't quote the summary.",
                        lines.len()
                    ));
                } else {
                    // Without a summary the transcript is all there is: show its first page.
                    out.push_str("\nTranskrypcja (no AI summary for this meeting):\n");
                    page(&mut out, &lines, 0, &render_line);
                }
            }
            MeetingView::Around { offset_s } => {
                let (from, to) = (offset_s - AROUND_BEFORE_S, offset_s + AROUND_AFTER_S);
                out.push_str(&format!("\nTranskrypcja {}–{}:\n", mmss(from.max(0)), mmss(to)));
                let window: Vec<_> = lines.iter().filter(|l| (from..=to).contains(&offset(l))).collect();
                if window.is_empty() {
                    let last = lines.last().map_or(0, offset);
                    out.push_str(&format!("(no transcript lines there; the transcript runs 00:00–{})", mmss(last)));
                }
                for l in window {
                    out.push_str(&render_line(l));
                }
            }
            MeetingView::Transcript { from_line } => {
                out.push_str("\nTranskrypcja:\n");
                page(&mut out, &lines, from_line, &render_line);
            }
        }
        Ok(out.trim_end().to_owned())
    }
}

fn string(p: &Map<String, Value>, k: &str) -> String {
    p.get(k).and_then(Value::as_str).unwrap_or_default().to_owned()
}

fn int(p: &Map<String, Value>, k: &str) -> i64 {
    p.get(k).and_then(Value::as_i64).unwrap_or_default()
}

/// A message chunk is `[header]\n[Thread: …\n][Previous: …\n]Author (time): text`; the
/// evidence shown is just the message itself.
fn message_body(chunk: &str) -> String {
    let last = chunk
        .lines()
        .skip_while(|l| l.starts_with('[') || l.starts_with("Thread: ") || l.starts_with("Previous: "))
        .collect::<Vec<_>>()
        .join("\n");
    match last.split_once("): ") {
        Some((_, body)) => body.to_owned(),
        None => last,
    }
}

/// "Team / Channel", or "Chat: <name>" for a private chat, whose team is only the scraper's marker.
pub fn place(team: &str, channel: &str) -> String {
    if team == CHAT_TEAM { format!("{CHAT_TEAM}: {channel}") } else { format!("{team} / {channel}") }
}

fn datetime(ts: i64) -> String {
    jiff::Timestamp::from_millisecond(ts)
        .map(|t| t.to_zoned(jiff::tz::TimeZone::system()).strftime("%Y-%m-%d %H:%M").to_string())
        .unwrap_or_default()
}

fn truncate(s: &str, max: usize) -> String {
    let s = s.trim();
    if s.chars().count() > max { format!("{}…", s.chars().take(max).collect::<String>()) } else { s.to_owned() }
}

/// A transcript chunk's text is `[title · date]\n<lines>`; the evidence snippet drops that
/// leading header line.
fn strip_header(text: &str) -> String {
    text.lines().skip_while(|l| l.trim_start().starts_with('[')).collect::<Vec<_>>().join("\n").trim().to_owned()
}

pub fn format_meetings(r: &MeetingResults) -> String {
    let mut out = String::new();
    for n in &r.notes {
        out.push_str(&format!("({n})\n"));
    }
    if r.hits.is_empty() {
        out.push_str("No matching meetings.");
        return out;
    }
    if !out.is_empty() {
        out.push('\n');
    }
    let lines: Vec<String> = r
        .hits
        .iter()
        .enumerate()
        .map(|(i, h)| {
            let who = h.speaker.as_deref().unwrap_or("—");
            format!(
                "[{}] {} · {} – {} · meeting {}\n    {} ({}): {}",
                i + 1,
                h.title,
                datetime(h.started_at),
                datetime(h.ended_at),
                h.meeting_id,
                who,
                datetime(h.ts),
                truncate(&h.text, EVIDENCE_CHARS).replace('\n', "\n    ")
            )
        })
        .collect();
    out.push_str(&lines.join("\n\n"));
    out
}

pub fn format_results(r: &TeamsResults) -> String {
    let mut out = String::new();
    for n in &r.notes {
        out.push_str(&format!("({n})\n"));
    }
    if r.hits.is_empty() {
        out.push_str("No matching Teams threads.");
        return out;
    }
    if !r.boosted_groups.is_empty() {
        out.push_str(&format!("(threads in {} are ranked higher for this project)\n", r.boosted_groups.join(", ")));
    }
    if !out.is_empty() {
        out.push('\n');
    }
    let lines: Vec<String> = r
        .hits
        .iter()
        .enumerate()
        .map(|(i, h)| {
            format!(
                "[{}] {} · thread {} · {} messages · last {}\n    {} ({}): {}",
                i + 1,
                place(&h.team, &h.channel),
                h.thread_id,
                h.message_count,
                datetime(h.last_ts),
                h.author,
                datetime(h.ts),
                truncate(&h.text, EVIDENCE_CHARS).replace('\n', "\n    ")
            )
        })
        .collect();
    out.push_str(&lines.join("\n\n"));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir() -> Directory {
        serde_json::from_str(
            r#"{"groups": {"SEDN": {"channels": ["SEDN/"], "roles": {"Tomasz Chmiel": "leader"}, "members": ["Anna Fiałkowska", "Tomasz Chmiel"]},
                           "Helpdesk": {"channels": ["ZSUN-Helpdesk/"], "members": ["Tomasz Chmiel"]}},
                "people": {"Tomasz Chmiel": {"title": "kierownik", "vip": true}}}"#,
        )
        .unwrap()
    }

    #[test]
    fn labels_match_the_apps_format_and_group_order() {
        let d = dir();
        // SEDN comes before Helpdesk in the file, as in the app (not alphabetical).
        assert_eq!(d.label("Tomasz Chmiel"), "Tomasz Chmiel [VIP, kierownik; SEDN leader; Helpdesk member]");
        assert_eq!(d.label("anna fialkowska"), "anna fialkowska [SEDN member]", "folded match");
        assert_eq!(d.label("Nobody"), "Nobody");
    }

    #[test]
    fn member_with_a_role_is_matched_by_exact_name_like_the_app() {
        let d: Directory = serde_json::from_str(r#"{"groups": {"G": {"roles": {"Anna Nowak": "lead"}, "members": ["anna nowak"]}}}"#).unwrap();
        // Different spelling: the app tags both, since its role lookup is by exact key.
        assert_eq!(d.tags("Anna Nowak"), ["G lead", "G member"]);
    }

    #[test]
    fn group_lookup_is_folded() {
        let d = dir();
        assert_eq!(d.group("sedn").unwrap().1, ["SEDN/".to_owned()]);
        assert!(d.group("Technet").is_none());
    }

    #[test]
    fn missing_people_and_projects_files_are_empty() {
        assert!(Directory::load(Path::new("Z:/none/people.json")).unwrap().groups.is_empty());
        assert!(load_project_groups(Path::new("Z:/none/teams-projects.json")).unwrap().is_empty());
    }

    #[test]
    fn project_groups_are_keyed_by_normalised_project() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("teams-projects.json");
        std::fs::write(&p, "\u{feff}{\"Z:\\\\no\\\\Helpdesk\": [\"Helpdesk\"]}").unwrap();
        let m = load_project_groups(&p).unwrap();
        assert_eq!(m["z:/no/helpdesk"], ["Helpdesk"]);
    }

    #[test]
    fn message_body_strips_header_and_context() {
        let chunk = "[T / C · 2026-09-20]\nThread: opening\nPrevious: Piotr: sprawdzam\nAnna (2026-09-20 10:00): ok, zrobione";
        assert_eq!(message_body(chunk), "ok, zrobione");
    }

    #[test]
    fn offsets_parse_as_cited_and_render_back() {
        assert_eq!(parse_offset("39:20").unwrap(), 2360);
        assert_eq!(parse_offset(" [1:02:03] ").unwrap(), 3723);
        assert_eq!(parse_offset("95").unwrap(), 95);
        for bad in ["", "1:75", "a:b", "-3", "1:2:3:4", "1:60:00"] {
            assert!(parse_offset(bad).is_err(), "{bad}");
        }
        assert_eq!((mmss(2360), mmss(5), mmss(3723), mmss(-4)), ("39:20".into(), "00:05".into(), "1:02:03".into(), "00:00".into()));
    }

    #[test]
    fn kind_parses_loosely_and_rejects_typos() {
        assert_eq!(Conversations::parse(" Chats ").unwrap(), Conversations::Chats);
        assert_eq!(Conversations::parse("channel").unwrap(), Conversations::Channels);
        assert_eq!(Conversations::parse("").unwrap(), Conversations::All);
        assert!(Conversations::parse("dms").unwrap_err().to_string().contains("\"chats\""));
    }

    #[test]
    fn chats_are_labelled_by_name_not_as_a_team() {
        assert_eq!(place("Chat", "Joanna Raj"), "Chat: Joanna Raj");
        assert_eq!(place("SEDN", "daily"), "SEDN / daily");
    }

    #[test]
    fn formatting_shows_notes_boost_and_hits() {
        assert_eq!(format_results(&TeamsResults::default()), "No matching Teams threads.");
        let hit = ThreadHit {
            thread_id: "1".into(),
            score: 1.0,
            boosted: true,
            team: "T".into(),
            channel: "C".into(),
            message_count: 3,
            last_ts: 0,
            author: "Anna".into(),
            ts: 0,
            text: "hello".into(),
        };
        let r = TeamsResults { hits: vec![hit], boosted_groups: vec!["Helpdesk".into()], notes: vec!["semantic search unavailable (x)".into()] };
        let out = format_results(&r);
        assert!(out.starts_with("(semantic search unavailable (x))\n(threads in Helpdesk are ranked higher for this project)\n\n[1] T / C"), "{out}");
    }
}
