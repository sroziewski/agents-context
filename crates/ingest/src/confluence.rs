//! Confluence pages → chunks for the `confluence` collection.
//!
//! What is indexed comes in units: a whole space (`SEDN`) or a page with everything below it
//! (`p310000001`), for projects documented inside a big shared space.
//!
//! Change detection is the page version: each run lists every current page of a unit with its
//! version number only (no bodies), fetches and re-chunks just the pages whose version moved, and
//! within those re-embeds only the chunks whose text changed. Pages a complete listing no longer
//! contains (deleted, or moved out) are dropped from the index.

use std::collections::HashSet;

use embed::Embedder;
use serde_json::{Map, Value, json};
use sources::confluence::{Confluence, Page, PageRef};
use store::{Field, SearchFilter, Store};

use crate::chunk::{Chunk, pack, split};
use crate::indexer::{IndexConfig, IndexError, Report, write_chunks};
use crate::state::{Record, State};
use crate::tokens::TokenCounter;

pub const SOURCE: &str = "confluence";

/// Part of every page's fingerprint next to its version. Bump it when the page-to-text conversion
/// changes, so the next run re-indexes every page once instead of only edited ones.
/// 2: table rows as one labelled line.
const TEXT_FORMAT: u32 = 2;

/// Payload fields the confluence collection indexes. `unit` is what search scopes by; `project`
/// is the page's space key.
pub const CONFLUENCE_FIELDS: &[Field] =
    &[Field::Keyword("unit"), Field::Tenant("project"), Field::Keyword("kind"), Field::Keyword("page_id"), Field::Keyword("author")];

/// The unit key of a mapping entry: a space key as is, a page id or a page URL
/// (`…/pages/310000001/Title`) as `p310000001`. `None` for an entry that is neither.
pub fn unit_key(entry: &str) -> Option<String> {
    let e = entry.trim();
    let id = e.split_once("/pages/").map(|(_, rest)| rest.split('/').next().unwrap_or("")).or_else(|| e.strip_prefix('p')).unwrap_or(e);
    if !id.is_empty() && id.bytes().all(|b| b.is_ascii_digit()) {
        return Some(format!("p{id}"));
    }
    (!e.is_empty() && !e.contains(['/', ':', ' ', '?'])).then(|| e.to_owned())
}

/// The root page id of a page-tree unit, `None` for a space.
fn tree_root(unit: &str) -> Option<&str> {
    unit.strip_prefix('p').filter(|id| !id.is_empty() && id.bytes().all(|b| b.is_ascii_digit()))
}

/// What the sync needs from Confluence; the real client implements it, tests use a fake.
pub trait PageSource {
    fn page_refs(&self, space: &str) -> Result<Vec<PageRef>, String>;
    /// A page and every page below it.
    fn tree_refs(&self, root: &str) -> Result<Vec<PageRef>, String>;
    fn page(&self, id: &str) -> Result<Page, String>;
}

impl PageSource for Confluence {
    fn page_refs(&self, space: &str) -> Result<Vec<PageRef>, String> {
        Confluence::page_refs(self, space).map_err(|e| e.to_string())
    }

    fn tree_refs(&self, root: &str) -> Result<Vec<PageRef>, String> {
        Confluence::tree_refs(self, root).map_err(|e| e.to_string())
    }

    fn page(&self, id: &str) -> Result<Page, String> {
        Confluence::page(self, id).map_err(|e| e.to_string())
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ConfluenceReport {
    pub report: Report,
    /// Pages dropped because their unit no longer lists them.
    pub removed: usize,
}

/// The state key of a page: its unit first, so one unit's pages can be listed by prefix.
fn state_key(unit: &str, page_id: &str) -> String {
    format!("{unit}/{page_id}")
}

fn page_scope(unit: &str, page_id: &str) -> SearchFilter {
    SearchFilter::default().eq("source", SOURCE).eq("unit", unit.to_owned()).eq("page_id", page_id.to_owned())
}

/// Epoch ms of a Confluence `version.when` (RFC 3339), or 0 when it has an unexpected shape.
pub fn updated_ms(s: &str) -> i64 {
    s.parse::<jiff::Timestamp>().map(|t| t.as_millisecond()).unwrap_or(0)
}

/// All chunks of one page, redacted, each led by `[SPACE · Title] breadcrumb` so a chunk from the
/// middle of a long page still says where it comes from. Returns `(chunks, redactions)`.
pub fn page_chunks(page: &Page, unit: &str, counter: &dyn TokenCounter, cfg: &IndexConfig) -> (Vec<Chunk>, usize) {
    let redactor = &cfg.chunk.redactor;
    let (title, n_title) = redactor.scrub(&page.title);
    let (body, n_body) = redactor.scrub(&page.text);
    let crumbs = if page.ancestors.is_empty() { String::new() } else { format!(" {}", page.ancestors.join(" › ")) };
    let header = format!("[{} · {title}]{crumbs}\n", page.space);
    let budget = cfg.chunk.max_tokens.saturating_sub(counter.count(&header)).max(16);
    let mut pieces = pack(split(body.trim(), budget, counter), budget, cfg.chunk.overlap_tokens, counter);
    if pieces.is_empty() {
        // An empty page (a folder-like parent) is still findable by its title.
        pieces.push(String::new());
    }
    let total = pieces.len();
    let chunks = pieces
        .into_iter()
        .enumerate()
        .map(|(i, piece)| {
            let text = format!("{header}{piece}").trim_end().to_owned();
            let mut p = payload(page, unit, &title, &text);
            p.insert("chunk_index".into(), json!(i));
            p.insert("chunk_count".into(), json!(total));
            // The unit is part of the id: a page inside two mapped units is two sets of points,
            // and dropping it from one must not touch the other.
            Chunk { id: store::point_id(SOURCE, &format!("{unit}/{}/{i}", page.id)), tokens: counter.count(&text), payload: p, text }
        })
        .collect();
    (chunks, n_title + n_body)
}

fn payload(page: &Page, unit: &str, title: &str, text: &str) -> Map<String, Value> {
    let v = json!({
        "kind": "page",
        "source": SOURCE,
        "unit": unit,
        "project": page.space,
        "page_id": page.id,
        // The state key, so `ctx prune` can forget pruned pages and the next sync restores them.
        "page_key": state_key(unit, &page.id),
        "title": title,
        "author": page.author,
        "url": page.url,
        "version": page.version,
        "text": text,
        "ts": updated_ms(&page.updated),
    });
    let Value::Object(m) = v else { unreachable!("json! object") };
    m
}

/// The page-level fields of every chunk's payload, which an edit can change without changing a
/// chunk's text.
fn page_fields(page: &Page, chunk_count: usize) -> Map<String, Value> {
    let v = json!({
        "author": page.author,
        "url": page.url,
        "version": page.version,
        "ts": updated_ms(&page.updated),
        "chunk_count": chunk_count,
    });
    let Value::Object(m) = v else { unreachable!("json! object") };
    m
}

/// Indexes the pages of `units` (see [`unit_key`]) that are new or have a new version, and drops
/// pages a unit no longer lists. A unit whose listing fails is skipped whole (nothing is dropped
/// on a partial view); a page that fails is reported and retried next run, since its version is
/// not recorded.
#[allow(clippy::too_many_arguments)]
pub async fn sync_confluence(
    store: &Store,
    embedder: &mut dyn Embedder,
    counter: &dyn TokenCounter,
    state: &State,
    client: &dyn PageSource,
    units: &[String],
    config: &IndexConfig,
    now_ms: i64,
) -> Result<ConfluenceReport, IndexError> {
    let mut out = ConfluenceReport::default();
    for unit in units {
        let listing = match tree_root(unit) {
            Some(root) => client.tree_refs(root),
            None => client.page_refs(unit),
        };
        let refs = match listing {
            Ok(refs) => refs,
            Err(e) => {
                out.report.failed.push((format!("unit {unit}"), e));
                continue;
            }
        };
        let mut listed = HashSet::new();
        for r in &refs {
            out.report.scanned += 1;
            let key = state_key(unit, &r.id);
            listed.insert(key.clone());
            let fingerprint = format!("{}+text{TEXT_FORMAT}", r.version);
            if state.get(SOURCE, &key)?.is_some_and(|i| i.fingerprint == fingerprint) {
                out.report.unchanged += 1;
                continue;
            }
            let page = match client.page(&r.id) {
                Ok(p) => p,
                Err(e) => {
                    out.report.failed.push((key, e));
                    continue;
                }
            };
            let (chunks, redactions) = page_chunks(&page, unit, counter, config);
            let scope = page_scope(unit, &page.id);
            // write_chunks skips chunks whose text is unchanged, but the editor, time and version
            // are not in the text: refresh them on every chunk, without re-embedding.
            let written = async {
                let n = write_chunks(store, embedder, config, &scope, &chunks).await?;
                store.set_payload(&scope, page_fields(&page, n)).await?;
                Ok::<_, IndexError>(n)
            };
            let n = match written.await {
                Ok(n) => n,
                Err(e) => {
                    out.report.failed.push((key, e.to_string()));
                    continue;
                }
            };
            // The listed version, not the fetched one: an edit in between is caught next run.
            state.record(&Record { source: SOURCE, session_id: &key, fingerprint: &fingerprint, chunks: n, redactions, ..Default::default() }, now_ms)?;
            out.report.indexed += 1;
            out.report.chunks += n;
            out.report.redactions += redactions;
        }
        for key in state.ids_with_prefix(SOURCE, &format!("{unit}/"))? {
            if !listed.contains(&key) {
                drop_page(store, state, &key).await?;
                out.removed += 1;
            }
        }
    }
    // Units taken out of the mapping: nothing would ever refresh or remove their pages.
    for key in state.ids_with_prefix(SOURCE, "")? {
        let unit = key.split_once('/').map_or("", |(u, _)| u);
        if !units.iter().any(|u| u == unit) {
            drop_page(store, state, &key).await?;
            out.removed += 1;
        }
    }
    Ok(out)
}

/// Removes one page's points and state row. The unit comes from the key itself, never from the
/// caller, so only this key's copy of the page goes.
async fn drop_page(store: &Store, state: &State, key: &str) -> Result<(), IndexError> {
    let (unit, page_id) = key.split_once('/').unwrap_or(("", key));
    store.delete_matching(&page_scope(unit, page_id)).await?;
    state.forget(SOURCE, key)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokens::WordCounter;

    fn page(id: &str, text: &str) -> Page {
        Page {
            id: id.into(),
            space: "SEDN".into(),
            title: "Runbook wdrożenia".into(),
            version: 3,
            updated: "2026-10-01T10:00:00.000+02:00".into(),
            author: "Anna W".into(),
            ancestors: vec!["Home".into(), "Ops".into()],
            text: text.into(),
            url: format!("https://c/pages/viewpage.action?pageId={id}"),
        }
    }

    #[test]
    fn every_chunk_carries_the_space_title_and_breadcrumb() {
        let mut cfg = IndexConfig::default();
        cfg.chunk.max_tokens = 40;
        cfg.chunk.overlap_tokens = 0;
        let long = (0..60).map(|i| format!("krok {i} restart poda.")).collect::<Vec<_>>().join("\n");
        let (chunks, _) = page_chunks(&page("1", &long), "SEDN", &WordCounter, &cfg);
        assert!(chunks.len() > 1, "a long page is split");
        for c in &chunks {
            assert!(c.text.starts_with("[SEDN · Runbook wdrożenia] Home › Ops\n"), "{}", c.text);
            assert_eq!(c.payload["project"], "SEDN");
            assert_eq!(c.payload["page_id"], "1");
            assert_eq!(c.payload["chunk_count"], chunks.len());
        }
        assert_eq!(chunks[0].payload["ts"], updated_ms("2026-10-01T10:00:00.000+02:00"));
    }

    #[test]
    fn an_empty_page_is_one_chunk_with_its_title() {
        let (chunks, _) = page_chunks(&page("2", ""), "SEDN", &WordCounter, &IndexConfig::default());
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "[SEDN · Runbook wdrożenia] Home › Ops");
    }

    #[test]
    fn title_and_body_are_redacted() {
        let mut cfg = IndexConfig::default();
        cfg.chunk.redactor = crate::Redactor::new(["Tajne123".to_string()]);
        let mut p = page("3", "hasło Tajne123 do bazy");
        p.title = "Dostęp Tajne123".into();
        let (chunks, n) = page_chunks(&p, "SEDN", &WordCounter, &cfg);
        assert!(n >= 2);
        assert!(chunks.iter().all(|c| !c.text.contains("Tajne123") && c.payload["title"].as_str().is_some_and(|t| !t.contains("Tajne123"))));
    }

    #[test]
    fn ids_are_stable_per_page_and_chunk() {
        let (a, _) = page_chunks(&page("4", "x"), "SEDN", &WordCounter, &IndexConfig::default());
        let (b, _) = page_chunks(&page("4", "y"), "SEDN", &WordCounter, &IndexConfig::default());
        assert_eq!(a[0].id, b[0].id, "an edit keeps the point id, so the old vector is replaced");
        let (c, _) = page_chunks(&page("5", "x"), "SEDN", &WordCounter, &IndexConfig::default());
        assert_ne!(a[0].id, c[0].id);
    }

    #[test]
    fn mapping_entries_become_space_or_page_tree_units() {
        assert_eq!(unit_key("SEDN").as_deref(), Some("SEDN"));
        assert_eq!(unit_key(" zsunl ").as_deref(), Some("zsunl"));
        assert_eq!(unit_key("310000001").as_deref(), Some("p310000001"));
        assert_eq!(unit_key("p310000001").as_deref(), Some("p310000001"));
        assert_eq!(unit_key("https://confluence.example.org/spaces/DOC/pages/310000001/Project").as_deref(), Some("p310000001"));
        assert_eq!(unit_key("https://confluence.example.org/spaces/HD/pages/2400001/Help+desk+guide").as_deref(), Some("p2400001"));
        assert_eq!(unit_key("https://confluence.example.org/display/X"), None);
        assert_eq!(unit_key(""), None);
        assert_eq!(tree_root("p2400001"), Some("2400001"));
        assert_eq!((tree_root("SEDN"), tree_root("pmo"), tree_root("p")), (None, None, None));
    }

    #[test]
    fn one_page_in_two_units_gets_two_sets_of_ids() {
        let (a, _) = page_chunks(&page("4", "x"), "SEDN", &WordCounter, &IndexConfig::default());
        let (b, _) = page_chunks(&page("4", "x"), "p1", &WordCounter, &IndexConfig::default());
        assert_ne!(a[0].id, b[0].id);
        assert_eq!((a[0].payload["unit"].as_str(), b[0].payload["unit"].as_str()), (Some("SEDN"), Some("p1")));
        assert_eq!(b[0].payload["page_key"], "p1/4");
    }

    #[test]
    fn updated_ms_reads_rfc3339_and_degrades() {
        assert!(updated_ms("2026-10-01T10:00:00.000+02:00") > 0);
        assert_eq!(updated_ms("garbage"), 0);
    }
}
