//! Optional, explicit prune of indexed data: drop a project, or remove points older than N days.
//! Never automatic — retention is off by default (the index deliberately outlives its sources).
//! Both are dry-run unless `apply`, and both keep the state DB consistent with Qdrant.

use std::path::Path;

use ingest::State;
use store::{SearchFilter, Store};

#[derive(Debug, Clone, PartialEq)]
pub struct PruneReport {
    /// Qdrant points matched (chunks).
    pub points: u64,
    /// Distinct sessions/threads/issues matched.
    pub items: usize,
    /// Whether the deletion was actually performed (vs. a dry run).
    pub applied: bool,
}

/// Points belonging to one project (turns + summaries share the `project` payload). `State` is
/// opened only after the async store work, so the future stays `Send`.
pub async fn prune_project(store: &Store, state_path: &Path, project_key: &str, apply: bool) -> anyhow::Result<PruneReport> {
    anyhow::ensure!(!project_key.trim().is_empty(), "refusing to prune an empty project key (would match nothing/everything)");
    let filter = SearchFilter::default().eq("project", project_key.to_owned());
    let points = store.count(&filter).await?;
    let items = store.distinct_payload(&filter, "session_id", 200_000).await?.len();
    if apply && points > 0 {
        store.delete_matching(&filter).await?;
        State::open(state_path)?.delete_project(project_key)?;
    }
    Ok(PruneReport { points, items, applied: apply })
}

/// Points whose `ts` is at or before `cutoff_ms`, from one collection. `id_field` is the payload
/// field naming the item (session_id / thread_id / issue_key), used to prune matching state rows.
pub async fn prune_older_than(store: &Store, state_path: &Path, id_field: &str, cutoff_ms: i64, apply: bool) -> anyhow::Result<PruneReport> {
    let filter = SearchFilter::default().between(None, Some(cutoff_ms));
    let points = store.count(&filter).await?;
    let ids = store.distinct_payload(&filter, id_field, 500_000).await?;
    if apply && points > 0 {
        store.delete_matching(&filter).await?;
        State::open(state_path)?.delete_ids(&ids)?;
    }
    Ok(PruneReport { points, items: ids.len(), applied: apply })
}

pub fn format_report(label: &str, r: &PruneReport) -> String {
    let verb = if r.applied { "removed" } else { "would remove (dry run)" };
    format!("{label}: {verb} {} points across {} item(s)", r.points, r.items)
}
