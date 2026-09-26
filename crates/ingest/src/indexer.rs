use std::collections::HashMap;

use embed::Embedder;
use sources::{Session, SessionRef, SourceError};
use store::{Doc, SearchFilter, Store};
use uuid::Uuid;

use crate::batch::batches;
use crate::chunk::{Chunk, ChunkConfig, chunk_session};

/// Upper bound when reading a session's stored chunks; the largest real session has ~1,300.
const MAX_SESSION_CHUNKS: u32 = 20_000;
use crate::state::{Record, State};
use crate::tokens::TokenCounter;

#[derive(Debug, Clone)]
pub struct IndexConfig {
    pub chunk: ChunkConfig,
    /// Padded tokens per embedding call; see [`batches`].
    pub max_padded_tokens: usize,
    pub max_batch: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        // ~2000 padded tokens took ~1 s per call on the GTX 1650 in the spike, half the TDR limit.
        Self { chunk: ChunkConfig::default(), max_padded_tokens: 2000, max_batch: 8 }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    #[error(transparent)]
    Embed(#[from] embed::EmbedError),
    #[error(transparent)]
    Store(#[from] store::StoreError),
    #[error("state: {0}")]
    State(#[from] rusqlite::Error),
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Report {
    pub scanned: usize,
    pub unchanged: usize,
    pub indexed: usize,
    pub empty: usize,
    pub chunks: usize,
    pub redactions: usize,
    /// `(session id, error)` for sessions that could not be read; the rest of the sync proceeds.
    pub failed: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Progress<'a> {
    Indexed { id: &'a str, chunks: usize },
    Failed { id: &'a str, error: &'a str },
}

pub struct Indexer<'a> {
    pub store: &'a Store,
    pub embedder: &'a mut dyn Embedder,
    pub counter: &'a dyn TokenCounter,
    pub state: &'a State,
    pub config: IndexConfig,
}

impl Indexer<'_> {
    /// Re-indexes one session: embeds every chunk, upserts, then prunes the session's chunks that
    /// no longer exist. Returns `(chunks, redactions)`.
    pub async fn index_session(&mut self, s: &Session) -> Result<(usize, usize), IndexError> {
        let (chunks, stats) = chunk_session(s, self.counter, &self.config.chunk);
        // Turn chunks only: the session's summary point is owned by the summary pass, and pruning
        // it here would delete it on every re-index of a growing session.
        let session = SearchFilter::default()
            .eq("source", s.source.as_str())
            .eq("session_id", s.id.clone())
            .eq("kind", "turn");

        // An active session grows by a turn at a time; re-embedding all of it on every sync made a
        // re-run of one long session take minutes. Chunks whose id and text are already stored
        // keep their vectors. The payload is derived from the same turn as the text, so equal
        // text means an equal point.
        let stored: HashMap<Uuid, String> = self
            .store
            .scroll(&session, MAX_SESSION_CHUNKS)
            .await?
            .into_iter()
            .filter_map(|h| h.payload.get("text").and_then(|t| t.as_str()).map(|t| (h.id, t.to_owned())))
            .collect();
        let todo: Vec<&Chunk> = chunks.iter().filter(|c| stored.get(&c.id) != Some(&c.text)).collect();

        let tokens: Vec<usize> = todo.iter().map(|c| c.tokens).collect();
        let mut docs = Vec::with_capacity(todo.len());
        for range in batches(&tokens, self.config.max_padded_tokens, self.config.max_batch) {
            let texts: Vec<&str> = todo[range.clone()].iter().map(|c| c.text.as_str()).collect();
            let embeddings = self.embedder.embed(&texts)?;
            for (chunk, embedding) in todo[range].iter().zip(embeddings) {
                docs.push(Doc { id: chunk.id, embedding, payload: chunk.payload.clone() });
            }
        }
        self.store.upsert(docs).await?;
        let ids: Vec<Uuid> = chunks.iter().map(|c| c.id).collect();
        self.store.delete_matching_except(&session, &ids).await?;
        Ok((ids.len(), stats.redactions))
    }

    /// Indexes every ref whose fingerprint changed since the last run.
    ///
    /// Sessions that disappeared from the source are deliberately kept: Claude Code deletes
    /// transcripts after `cleanupPeriodDays` (30 by default), and then the index is the only copy.
    pub async fn sync(
        &mut self,
        refs: &[SessionRef],
        load: impl Fn(&SessionRef) -> sources::Result<Session>,
        mut progress: impl FnMut(Progress<'_>),
        now_ms: i64,
    ) -> Result<Report, IndexError> {
        let mut report = Report { scanned: refs.len(), ..Default::default() };
        for r in refs {
            let source = r.source.as_str();
            if self.state.get(source, &r.id)?.is_some_and(|i| i.fingerprint == r.fingerprint) {
                report.unchanged += 1;
                continue;
            }
            let session = match load(r) {
                Ok(s) => s,
                Err(SourceError::Empty(_)) => {
                    // Remember it so an idle empty session isn't re-read on every sync.
                    let empty = Record { source, session_id: &r.id, fingerprint: &r.fingerprint, ..Default::default() };
                    self.state.record(&empty, now_ms)?;
                    report.empty += 1;
                    continue;
                }
                Err(e) => {
                    let error = e.to_string();
                    progress(Progress::Failed { id: &r.id, error: &error });
                    report.failed.push((r.id.clone(), error));
                    continue;
                }
            };
            let (chunks, redactions) = self.index_session(&session).await?;
            let record = Record {
                source,
                session_id: &r.id,
                fingerprint: &r.fingerprint,
                project: &session.project,
                chunks,
                redactions,
                turns: session.turns.len(),
                subagent: session.is_subagent(),
            };
            self.state.record(&record, now_ms)?;
            progress(Progress::Indexed { id: &r.id, chunks });
            report.indexed += 1;
            report.chunks += chunks;
            report.redactions += redactions;
        }
        Ok(report)
    }
}
