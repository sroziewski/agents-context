//! Confluence for `ctx`: the project → space mapping (`confluence-projects.json`, the same shape
//! as `jira-projects.json`), space discovery for filling it in, and search over the indexed pages.
//! The client lives in `sources::confluence`, the indexer in `ingest::confluence`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use embed::Embedder;
use serde_json::Value;
use store::{Hit, SearchFilter, SearchMode, Store};

pub use sources::confluence::{Confluence, Space};

/// The units mapped to `project_key` (a project directory's key): space keys, and `p<id>` for a
/// page with everything below it (see `ingest::confluence::unit_key`).
pub fn units_for_project(project_key: &str, mapping: &Path) -> anyhow::Result<Vec<String>> {
    Ok(to_units(crate::jira::keys_for_project(project_key, &crate::jira::load_project_keys(mapping)?)))
}

/// Every mapped unit, which is what the indexer covers.
pub fn all_units(mapping: &Path) -> anyhow::Result<Vec<String>> {
    Ok(to_units(crate::jira::all_keys(&crate::jira::load_project_keys(mapping)?)))
}

/// Mapping entries as unit keys, deduplicated in order; entries that are neither a space key nor
/// a page are dropped.
fn to_units(entries: Vec<String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for unit in entries.iter().filter_map(|e| ingest::confluence::unit_key(e)) {
        if !out.contains(&unit) {
            out.push(unit);
        }
    }
    out
}

/// Spaces whose key or name contains every word of `filter` (case-insensitive); all without one.
pub fn matching_spaces<'a>(spaces: &'a [Space], filter: Option<&str>) -> Vec<&'a Space> {
    let words: Vec<String> = filter.unwrap_or_default().split_whitespace().map(str::to_lowercase).collect();
    spaces
        .iter()
        .filter(|s| {
            let hay = format!("{} {}", s.key, s.name).to_lowercase();
            words.iter().all(|w| hay.contains(w.as_str()))
        })
        .collect()
}

pub fn format_spaces(spaces: &[&Space]) -> String {
    if spaces.is_empty() {
        return "no matching spaces".into();
    }
    let width = spaces.iter().map(|s| s.key.chars().count()).max().unwrap_or(0);
    spaces.iter().map(|s| format!("{:width$}  {}", s.key, s.name)).collect::<Vec<_>>().join("\n")
}

/// Characters of page text shown per hit.
const EXCERPT_CHARS: usize = 500;

/// Semantic search over the `confluence` collection, scoped to a project's spaces.
#[derive(Clone)]
pub struct ConfluenceSearch {
    pub store: Arc<Store>,
    pub embedder: Arc<Mutex<Box<dyn Embedder>>>,
    /// `confluence-projects.json`.
    pub projects: PathBuf,
    pub redact: PathBuf,
}

impl ConfluenceSearch {
    /// The best pages for `query`, one hit per page. Scoped to `project_key`'s units, or
    /// everything indexed when `None`.
    pub async fn search(&self, query: &str, project_key: Option<&str>, limit: usize) -> anyhow::Result<String> {
        let mut filter = SearchFilter::default().eq("kind", "page");
        if let Some(project) = project_key {
            let units = units_for_project(project, &self.projects)?;
            if units.is_empty() {
                return Ok(format!(
                    "No Confluence spaces or pages are mapped to this project ({project}); add them to {} or search all spaces.",
                    self.projects.display()
                ));
            }
            filter = filter.any_of("unit", units);
        }
        let embedder = self.embedder.clone();
        let text = query.to_owned();
        let q = tokio::task::spawn_blocking(move || embedder.lock().unwrap_or_else(|p| p.into_inner()).embed_one(&text)).await??;
        // Several chunks of one page often rank together; fetch extra so `limit` distinct pages remain.
        let hits = self.store.search(&q, &filter, (limit * 4) as u64, SearchMode::Dense).await?;
        let redactor = ingest::Redactor::from_file(&self.redact)?;
        Ok(format_hits(&best_per_page(hits, limit), &redactor))
    }
}

/// The best-scoring chunk of each page, in score order, at most `limit` pages.
pub fn best_per_page(hits: Vec<Hit>, limit: usize) -> Vec<Hit> {
    let mut seen = HashSet::new();
    hits.into_iter().filter(|h| seen.insert(field(h, "page_id"))).take(limit).collect()
}

fn field(h: &Hit, k: &str) -> String {
    h.payload.get(k).and_then(Value::as_str).unwrap_or_default().to_owned()
}

pub fn format_hits(hits: &[Hit], redactor: &ingest::Redactor) -> String {
    if hits.is_empty() {
        return "no matching Confluence pages".into();
    }
    hits.iter()
        .map(|h| {
            let text = field(h, "text");
            // The first line is the `[SPACE · Title] breadcrumb` header, already in the heading.
            let body = text.split_once('\n').map_or("", |(_, b)| b).trim();
            let excerpt: String = body.chars().take(EXCERPT_CHARS).collect();
            let more = if body.chars().count() > EXCERPT_CHARS { "…" } else { "" };
            let day = jiff::Timestamp::from_millisecond(h.payload.get("ts").and_then(Value::as_i64).unwrap_or(0))
                .map(|t| t.strftime("%Y-%m-%d").to_string())
                .unwrap_or_default();
            let (excerpt, _) = redactor.scrub(&format!("{excerpt}{more}"));
            format!(
                "[{}] {} · score {:.2}\n{}\nedited {day} by {}\n{}",
                field(h, "project"),
                field(h, "title"),
                h.score,
                field(h, "url"),
                field(h, "author"),
                excerpt
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn hit(page: &str, score: f32, text: &str) -> Hit {
        let Value::Object(payload) = json!({
            "page_id": page, "project": "SEDN", "title": "Runbook", "url": format!("https://c/p/{page}"),
            "author": "Anna W", "ts": 1_759_305_600_000i64, "text": text,
        }) else {
            unreachable!()
        };
        Hit { id: store::point_id("confluence", &format!("{page}/0")), score, payload }
    }

    #[test]
    fn spaces_match_on_key_or_name_words() {
        let all = [
            Space { key: "SEDN".into(), name: "System Elektronicznych Dyplomów".into() },
            Space { key: "HD".into(), name: "Helpdesk ZSUN".into() },
            Space { key: "AI4SD".into(), name: "AI for Software Development".into() },
        ];
        let keys = |f: Option<&str>| matching_spaces(&all, f).iter().map(|s| s.key.clone()).collect::<Vec<_>>();
        assert_eq!(keys(Some("sedn")), ["SEDN"]);
        assert_eq!(keys(Some("helpdesk")), ["HD"]);
        assert_eq!(keys(Some("software development")), ["AI4SD"]);
        assert_eq!(keys(None).len(), 3);
        assert!(keys(Some("nothing")).is_empty());
    }

    #[test]
    fn one_hit_per_page_in_score_order() {
        let hits = vec![hit("1", 0.9, "a"), hit("1", 0.8, "b"), hit("2", 0.7, "c"), hit("3", 0.6, "d")];
        let best = best_per_page(hits, 2);
        assert_eq!(best.iter().map(|h| (field(h, "page_id"), h.score)).collect::<Vec<_>>(), [("1".into(), 0.9), ("2".into(), 0.7)]);
    }

    #[test]
    fn hits_show_where_the_page_is_and_an_excerpt_without_the_header() {
        let out = format_hits(&[hit("1", 0.9, "[SEDN · Runbook] Home › Ops\nrestart the pod")], &ingest::Redactor::default());
        assert!(out.starts_with("[SEDN] Runbook · score 0.90\nhttps://c/p/1\nedited 2025-10-01 by Anna W\nrestart the pod"), "{out}");
        assert_eq!(format_hits(&[], &ingest::Redactor::default()), "no matching Confluence pages");
    }

    #[test]
    fn excerpts_are_redacted_with_the_current_list() {
        let red = ingest::Redactor::new(["Tajne123".to_string()]);
        let out = format_hits(&[hit("1", 0.9, "[SEDN · Runbook]\nhasło Tajne123")], &red);
        assert!(!out.contains("Tajne123"), "{out}");
    }
}
