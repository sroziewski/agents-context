//! Re-applies redaction to text already in the index.
//!
//! New rules or a newly listed secret must also reach points written before them, including
//! sessions whose transcript Claude Code has since deleted: for those the stored text is the
//! only copy, so re-indexing from source cannot fix them.

use embed::Embedder;
use serde_json::{Map, Value};
use store::{Doc, SearchFilter, Store};

use crate::indexer::IndexError;
use crate::scrub::Redactor;

/// Points per embedding call: stored texts are up to ~480 tokens, and 4 x 480 stays under the
/// padded-token budget the GPU finishes within Windows' TDR limit.
const BATCH: usize = 4;
/// Upper bound on points read in one rescan.
const MAX_POINTS: u32 = 10_000_000;
/// Payload fields that can carry user text: the embedded `text` and the session title shown by
/// `list_sessions`.
const TEXT_FIELDS: [&str; 2] = ["text", "title"];

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RescanReport {
    pub checked: usize,
    /// Points whose text or title changed; those were rewritten (and re-embedded if the text
    /// changed).
    pub changed: usize,
    pub redactions: usize,
}

pub async fn rescan(store: &Store, embedder: &mut dyn Embedder, redactor: &Redactor) -> Result<RescanReport, IndexError> {
    let mut report = RescanReport::default();
    // Pages are scanned and only changed points kept, which are few; the collection itself is
    // never held in memory.
    let mut changed: Vec<(uuid::Uuid, Map<String, Value>)> = Vec::new();
    store
        .scroll_pages(&SearchFilter::default(), MAX_POINTS, |page| {
            report.checked += page.len();
            for mut h in page {
                let mut n_total = 0;
                for field in TEXT_FIELDS {
                    if let Some(Value::String(v)) = h.payload.get(field) {
                        let (clean, n) = redactor.scrub(v);
                        if n > 0 && clean != *v {
                            n_total += n;
                            h.payload.insert(field.into(), Value::String(clean));
                        }
                    }
                }
                if n_total > 0 {
                    report.redactions += n_total;
                    changed.push((h.id, h.payload));
                }
            }
            std::future::ready(Ok(()))
        })
        .await?;

    for batch in changed.chunks(BATCH) {
        let texts: Vec<&str> = batch.iter().map(|(_, p)| p.get("text").and_then(Value::as_str).unwrap_or_default()).collect();
        let embeddings = embedder.embed(&texts)?;
        let docs = batch
            .iter()
            .zip(embeddings)
            .map(|((id, payload), embedding)| Doc { id: *id, embedding, payload: payload.clone() })
            .collect();
        store.upsert(docs).await?;
        report.changed += batch.len();
    }
    Ok(report)
}
