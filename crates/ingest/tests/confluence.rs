//! sync_confluence against a fake Confluence and a real Qdrant: unchanged pages are neither
//! fetched nor embedded again, edited ones are, and pages gone from a space leave the index.

use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use embed::{Embedder, Embedding, HashEmbedder};
use ingest::confluence::{CONFLUENCE_FIELDS, PageSource, sync_confluence};
use ingest::{IndexConfig, State, WordCounter};
use sources::confluence::{Page, PageRef};
use store::{SearchFilter, Store};

const DIM: usize = 64;

struct Counting(HashEmbedder, Arc<AtomicUsize>);
impl Embedder for Counting {
    fn dim(&self) -> usize {
        DIM
    }
    fn embed(&mut self, texts: &[&str]) -> embed::Result<Vec<Embedding>> {
        self.1.fetch_add(texts.len(), Ordering::SeqCst);
        self.0.embed(texts)
    }
}

/// Space key → `(page id, version, text)`.
/// Ordered, so a page id set under two spellings of a key resolves to the upper-case one, as
/// Confluence does; a HashMap made that test pass or fail by chance.
type Spaces = BTreeMap<String, Vec<(String, u64, String)>>;

/// Spaces → pages, page trees (root → the ids in it), and a log of body fetches.
#[derive(Default)]
struct FakeConfluence {
    spaces: Mutex<Spaces>,
    trees: Mutex<HashMap<String, Vec<String>>>,
    fetches: Mutex<Vec<String>>,
    failing_space: Mutex<Option<String>>,
}

impl FakeConfluence {
    fn set(&self, space: &str, pages: &[(&str, u64, &str)]) {
        self.spaces.lock().unwrap().insert(space.into(), pages.iter().map(|(i, v, t)| ((*i).into(), *v, (*t).into())).collect());
    }

    fn set_tree(&self, root: &str, ids: &[&str]) {
        self.trees.lock().unwrap().insert(root.into(), ids.iter().map(|i| (*i).to_owned()).collect());
    }

    fn take_fetches(&self) -> Vec<String> {
        std::mem::take(&mut *self.fetches.lock().unwrap())
    }
}

impl PageSource for FakeConfluence {
    fn page_refs(&self, space: &str) -> Result<Vec<PageRef>, String> {
        if self.failing_space.lock().unwrap().as_deref() == Some(space) {
            return Err("503 Service Unavailable".into());
        }
        let spaces = self.spaces.lock().unwrap();
        Ok(spaces.get(space).into_iter().flatten().map(|(id, v, _)| PageRef { id: id.clone(), title: format!("Page {id}"), version: *v }).collect())
    }

    fn tree_refs(&self, root: &str) -> Result<Vec<PageRef>, String> {
        let trees = self.trees.lock().unwrap();
        let ids = trees.get(root).ok_or_else(|| format!("404 {root}"))?;
        let spaces = self.spaces.lock().unwrap();
        let all: Vec<&(String, u64, String)> = spaces.values().flatten().collect();
        Ok(ids
            .iter()
            .filter_map(|id| all.iter().find(|(pid, ..)| pid == id))
            .map(|(id, v, _)| PageRef { id: id.clone(), title: format!("Page {id}"), version: *v })
            .collect())
    }

    fn page(&self, id: &str) -> Result<Page, String> {
        self.fetches.lock().unwrap().push(id.into());
        let spaces = self.spaces.lock().unwrap();
        let (space, (_, version, text)) = spaces
            .iter()
            .find_map(|(s, pages)| pages.iter().find(|(pid, ..)| pid == id).map(|p| (s.clone(), p.clone())))
            .ok_or_else(|| format!("404 {id}"))?;
        Ok(Page {
            id: id.into(),
            space,
            title: format!("Page {id}"),
            version,
            updated: "2026-10-01T10:00:00.000+02:00".into(),
            author: "Anna W".into(),
            ancestors: vec!["Home".into()],
            text,
            url: format!("https://c/pages/viewpage.action?pageId={id}"),
        })
    }
}

struct Env {
    store: Store,
    state: State,
    calls: Arc<AtomicUsize>,
    _q: testkit::Qdrant,
}

async fn env() -> Env {
    let q = testkit::qdrant().await;
    let store = Store::connect(&q.url, None, "confluence", DIM).unwrap();
    store.ensure_collection(CONFLUENCE_FIELDS).await.unwrap();
    Env { store, state: State::in_memory().unwrap(), calls: Arc::new(AtomicUsize::new(0)), _q: q }
}

impl Env {
    async fn sync(&self, client: &FakeConfluence, spaces: &[&str]) -> ingest::confluence::ConfluenceReport {
        let mut emb = Counting(HashEmbedder::new(DIM), self.calls.clone());
        let spaces: Vec<String> = spaces.iter().map(|s| (*s).to_owned()).collect();
        sync_confluence(&self.store, &mut emb, &WordCounter, &self.state, client, &spaces, &IndexConfig::default(), 0).await.unwrap()
    }

    async fn sync_with(&self, client: &dyn PageSource, spaces: &[&str]) -> ingest::confluence::ConfluenceReport {
        let mut emb = Counting(HashEmbedder::new(DIM), self.calls.clone());
        let spaces: Vec<String> = spaces.iter().map(|s| (*s).to_owned()).collect();
        sync_confluence(&self.store, &mut emb, &WordCounter, &self.state, client, &spaces, &IndexConfig::default(), 0).await.unwrap()
    }

    fn embedded(&self) -> usize {
        self.calls.swap(0, Ordering::SeqCst)
    }

    async fn pages_in(&self, space: &str) -> Vec<String> {
        let hits = self.store.scroll(&SearchFilter::default().eq("project", space.to_owned()), 1000).await.unwrap();
        let mut ids: Vec<String> = hits.iter().map(|h| h.payload["page_id"].as_str().unwrap().to_owned()).collect();
        ids.sort();
        ids.dedup();
        ids
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn only_new_and_edited_pages_are_fetched_and_embedded() {
    let e = env().await;
    let c = FakeConfluence::default();
    c.set("SEDN", &[("1", 1, "restart the pod with kubectl"), ("2", 4, "deploy runbook for prod")]);

    let first = e.sync(&c, &["SEDN"]).await;
    assert_eq!((first.report.scanned, first.report.indexed, first.report.unchanged), (2, 2, 0));
    assert_eq!(c.take_fetches().len(), 2);
    assert!(e.embedded() >= 2);
    assert_eq!(e.pages_in("SEDN").await, ["1", "2"]);

    let again = e.sync(&c, &["SEDN"]).await;
    assert_eq!((again.report.indexed, again.report.unchanged), (0, 2), "same versions: nothing re-indexed");
    assert!(c.take_fetches().is_empty(), "an unchanged page's body is not even fetched");
    assert_eq!(e.embedded(), 0);

    c.set("SEDN", &[("1", 1, "restart the pod with kubectl"), ("2", 5, "deploy runbook for prod, now with a rollback step")]);
    let edited = e.sync(&c, &["SEDN"]).await;
    assert_eq!((edited.report.indexed, edited.report.unchanged), (1, 1));
    assert_eq!(c.take_fetches(), ["2"], "only the edited page is fetched");
    assert!(e.embedded() >= 1);
    let text = e.store.scroll(&SearchFilter::default().eq("page_id", "2"), 10).await.unwrap()[0].payload["text"].as_str().unwrap().to_owned();
    assert!(text.contains("rollback step"), "the index holds the new text: {text}");
}

#[tokio::test(flavor = "multi_thread")]
async fn deleted_and_moved_pages_leave_their_old_space() {
    let e = env().await;
    let c = FakeConfluence::default();
    c.set("SEDN", &[("1", 1, "a"), ("2", 1, "b"), ("3", 1, "c")]);
    c.set("HD", &[]);
    e.sync(&c, &["SEDN", "HD"]).await;
    assert_eq!(e.pages_in("SEDN").await, ["1", "2", "3"]);

    // Page 2 is deleted; page 3 moves to HD.
    c.set("SEDN", &[("1", 1, "a")]);
    c.set("HD", &[("3", 2, "c")]);
    let r = e.sync(&c, &["SEDN", "HD"]).await;
    assert_eq!(r.removed, 2, "2 deleted, 3 gone from SEDN");
    assert_eq!(e.pages_in("SEDN").await, ["1"]);
    assert_eq!(e.pages_in("HD").await, ["3"]);
    assert!(e.state.get("confluence", "SEDN/2").unwrap().is_none());
    assert!(e.state.get("confluence", "HD/3").unwrap().is_some());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_failed_listing_drops_nothing_and_other_spaces_still_sync() {
    let e = env().await;
    let c = FakeConfluence::default();
    c.set("SEDN", &[("1", 1, "a")]);
    c.set("HD", &[("7", 1, "helpdesk faq")]);
    e.sync(&c, &["SEDN", "HD"]).await;

    *c.failing_space.lock().unwrap() = Some("SEDN".into());
    c.set("HD", &[("7", 2, "helpdesk faq, updated")]);
    let r = e.sync(&c, &["SEDN", "HD"]).await;
    assert_eq!(r.report.failed.len(), 1, "{:?}", r.report.failed);
    assert_eq!(r.removed, 0, "an unreadable space is not mistaken for an empty one");
    assert_eq!(e.pages_in("SEDN").await, ["1"]);
    assert_eq!(r.report.indexed, 1, "HD's edit still indexed");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_space_taken_out_of_the_mapping_leaves_the_index() {
    let e = env().await;
    let c = FakeConfluence::default();
    c.set("SEDN", &[("1", 1, "a")]);
    c.set("OLD", &[("8", 1, "retired docs"), ("9", 1, "more")]);
    e.sync(&c, &["SEDN", "OLD"]).await;
    assert_eq!(e.pages_in("OLD").await, ["8", "9"]);

    let r = e.sync(&c, &["SEDN"]).await;
    assert_eq!(r.removed, 2);
    assert!(e.pages_in("OLD").await.is_empty());
    assert!(e.state.ids_with_prefix("confluence", "OLD/").unwrap().is_empty());
    assert_eq!(e.pages_in("SEDN").await, ["1"], "the mapped space is untouched");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_page_that_fails_to_fetch_is_retried_next_run() {
    let e = env().await;
    let c = FakeConfluence::default();
    c.set("SEDN", &[("1", 1, "a")]);
    // Listed but not fetchable: the listing and the fetch disagree, as during a delete.
    let broken = BrokenFetch(&c);
    let r = {
        let mut emb = Counting(HashEmbedder::new(DIM), e.calls.clone());
        sync_confluence(&e.store, &mut emb, &WordCounter, &e.state, &broken, &["SEDN".to_owned()], &IndexConfig::default(), 0).await.unwrap()
    };
    assert_eq!((r.report.failed.len(), r.report.indexed), (1, 0));
    assert!(e.state.get("confluence", "SEDN/1").unwrap().is_none(), "not recorded, so it is tried again");
    let ok = e.sync(&c, &["SEDN"]).await;
    assert_eq!(ok.report.indexed, 1);
}

struct BrokenFetch<'a>(&'a FakeConfluence);
impl PageSource for BrokenFetch<'_> {
    fn page_refs(&self, space: &str) -> Result<Vec<PageRef>, String> {
        self.0.page_refs(space)
    }
    fn tree_refs(&self, root: &str) -> Result<Vec<PageRef>, String> {
        self.0.tree_refs(root)
    }
    fn page(&self, id: &str) -> Result<Page, String> {
        Err(format!("timeout fetching {id}"))
    }
}

fn long_text(paragraphs: usize) -> String {
    (0..paragraphs).map(|i| format!("Akapit {i}: procedura restartu usługi dyplomów krok po kroku, wariant {i}, z weryfikacją logów.")).collect::<Vec<_>>().join("\n")
}

async fn chunk_texts(e: &Env, page_id: &str) -> Vec<String> {
    let mut hits = e.store.scroll(&SearchFilter::default().eq("page_id", page_id.to_owned()), 1000).await.unwrap();
    hits.sort_by_key(|h| h.payload["chunk_index"].as_u64().unwrap());
    hits.iter().map(|h| h.payload["text"].as_str().unwrap().to_owned()).collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_page_that_shrinks_loses_its_extra_chunks() {
    let e = env().await;
    let c = FakeConfluence::default();
    let long = long_text(400);
    c.set("SEDN", &[("1", 1, &long), ("2", 1, "neighbour page")]);
    let first = e.sync(&c, &["SEDN"]).await;
    let before = chunk_texts(&e, "1").await.len();
    assert!(before >= 5, "a long page spans many chunks: {before}");
    assert_eq!(first.report.chunks, before + 1);

    c.set("SEDN", &[("1", 2, "Strona skrócona do jednego akapitu."), ("2", 1, "neighbour page")]);
    let shrunk = e.sync(&c, &["SEDN"]).await;
    assert_eq!(shrunk.report.chunks, 1);
    let after = chunk_texts(&e, "1").await;
    assert_eq!(after.len(), 1, "stale tail chunks are deleted: {after:?}");
    assert!(after[0].contains("skrócona"), "{after:?}");
    assert_eq!(chunk_texts(&e, "2").await.len(), 1, "the other page is untouched");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_edit_at_the_end_of_a_long_page_re_embeds_only_the_changed_chunks() {
    let e = env().await;
    let c = FakeConfluence::default();
    let long = long_text(400);
    c.set("SEDN", &[("1", 1, &long)]);
    e.sync(&c, &["SEDN"]).await;
    let total = e.embedded();
    assert!(total >= 5);

    c.set("SEDN", &[("1", 2, &format!("{long}\nNowy krok: powiadom helpdesk."))]);
    let r = e.sync(&c, &["SEDN"]).await;
    assert_eq!(r.report.indexed, 1);
    let re = e.embedded();
    assert!(re >= 1 && re < total, "only the tail is re-embedded: {re} of {total}");
    assert!(chunk_texts(&e, "1").await.last().unwrap().contains("powiadom helpdesk"));
}

#[tokio::test(flavor = "multi_thread")]
async fn an_empty_space_indexes_nothing_and_emptying_one_drops_its_pages() {
    let e = env().await;
    let c = FakeConfluence::default();
    c.set("EMPTY", &[]);
    let r = e.sync(&c, &["EMPTY"]).await;
    assert_eq!((r.report.scanned, r.report.indexed, r.removed, r.report.failed.len()), (0, 0, 0, 0));
    assert_eq!(e.store.count(&SearchFilter::default()).await.unwrap(), 0);

    c.set("SEDN", &[("1", 1, "a"), ("2", 1, "b")]);
    e.sync(&c, &["SEDN"]).await;
    c.set("SEDN", &[]);
    let emptied = e.sync(&c, &["SEDN"]).await;
    assert_eq!(emptied.removed, 2, "a successful empty listing means the pages are gone");
    assert!(e.pages_in("SEDN").await.is_empty());
    assert!(e.state.ids_with_prefix("confluence", "SEDN/").unwrap().is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_page_moved_into_a_space_synced_first_survives_its_old_space_cleanup() {
    let e = env().await;
    let c = FakeConfluence::default();
    c.set("SEDN", &[("1", 1, "a"), ("3", 1, "moving page")]);
    c.set("HD", &[]);
    e.sync(&c, &["HD", "SEDN"]).await;

    c.set("SEDN", &[("1", 1, "a")]);
    c.set("HD", &[("3", 2, "moving page")]);
    let r = e.sync(&c, &["HD", "SEDN"]).await;
    assert_eq!(r.removed, 1);
    assert_eq!(e.pages_in("HD").await, ["3"], "the new space's points are not deleted by the old space's cleanup");
    assert_eq!(e.pages_in("SEDN").await, ["1"]);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_space_key_differing_only_in_case_does_not_lose_pages() {
    let e = env().await;
    let c = FakeConfluence::default();
    c.set("sedn", &[("1", 1, "restart the pod")]);
    c.set("SEDN", &[("1", 1, "restart the pod")]);
    e.sync(&c, &["sedn"]).await;

    // The mapping is corrected to the canonical upper-case key.
    e.sync(&c, &["SEDN"]).await;
    assert_eq!(e.pages_in("SEDN").await, ["1"], "the page must be indexed under the corrected key");
    let again = e.sync(&c, &["SEDN"]).await;
    assert_eq!(e.pages_in("SEDN").await, ["1"], "and stay indexed: {again:?}");
}

/// Stamps each fetched page with an editor and edit time derived from its version, as Confluence does.
struct Stamped<'a>(&'a FakeConfluence);
impl PageSource for Stamped<'_> {
    fn page_refs(&self, space: &str) -> Result<Vec<PageRef>, String> {
        self.0.page_refs(space)
    }
    fn tree_refs(&self, root: &str) -> Result<Vec<PageRef>, String> {
        self.0.tree_refs(root)
    }
    fn page(&self, id: &str) -> Result<Page, String> {
        let p = self.0.page(id)?;
        Ok(Page { author: format!("editor v{}", p.version), updated: format!("2026-10-{:02}T10:00:00.000+02:00", p.version), ..p })
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn after_an_edit_every_chunk_carries_the_new_version_editor_and_time() {
    let e = env().await;
    let c = FakeConfluence::default();
    let long = long_text(400);
    c.set("SEDN", &[("1", 1, &long)]);
    e.sync_with(&Stamped(&c), &["SEDN"]).await;
    c.set("SEDN", &[("1", 2, &format!("{long}
Nowy krok."))]);
    e.sync_with(&Stamped(&c), &["SEDN"]).await;

    let hits = e.store.scroll(&SearchFilter::default().eq("page_id", "1"), 1000).await.unwrap();
    let n = hits.len();
    let stale: Vec<_> = hits
        .iter()
        .filter(|h| h.payload["version"] != 2 || h.payload["author"] != "editor v2" || h.payload["chunk_count"] != n)
        .map(|h| (h.payload["chunk_index"].clone(), h.payload["version"].clone(), h.payload["author"].clone()))
        .collect();
    assert!(stale.is_empty(), "unchanged chunks keep the old page metadata: {stale:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_page_tree_indexes_only_that_part_of_a_shared_space() {
    let e = env().await;
    let c = FakeConfluence::default();
    c.set("ZA", &[("100", 1, "System zgłoszeń HELPDESK"), ("101", 1, "Kategorie zgłoszeń"), ("200", 1, "unrelated team notes")]);
    c.set_tree("100", &["100", "101"]);

    let r = e.sync(&c, &["p100"]).await;
    assert_eq!((r.report.indexed, r.report.failed.len()), (2, 0));
    assert_eq!(e.pages_in("ZA").await, ["100", "101"], "the rest of the space stays out");
    assert!(e.state.get("confluence", "p100/101").unwrap().is_some());

    // A page leaves the tree: it leaves the index; the tree's other pages are not refetched.
    c.take_fetches();
    c.set_tree("100", &["100"]);
    let r = e.sync(&c, &["p100"]).await;
    assert_eq!((r.removed, r.report.unchanged), (1, 1));
    assert!(c.take_fetches().is_empty());
    assert_eq!(e.pages_in("ZA").await, ["100"]);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_page_in_both_a_space_and_a_tree_survives_dropping_either() {
    let e = env().await;
    let c = FakeConfluence::default();
    c.set("ZA", &[("100", 1, "helpdesk")]);
    c.set_tree("100", &["100"]);
    e.sync(&c, &["ZA", "p100"]).await;
    assert_eq!(units_of(&e, "100").await, ["ZA", "p100"]);
    e.sync(&c, &["p100"]).await;
    assert_eq!(units_of(&e, "100").await, ["p100"], "dropping the space keeps the tree's copy");
}

async fn units_of(e: &Env, page_id: &str) -> Vec<String> {
    let hits = e.store.scroll(&SearchFilter::default().eq("page_id", page_id.to_owned()), 100).await.unwrap();
    let mut u: Vec<String> = hits.iter().map(|h| h.payload["unit"].as_str().unwrap().to_owned()).collect();
    u.sort();
    u
}