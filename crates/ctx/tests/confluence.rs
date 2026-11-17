//! The Confluence client against a mock HTTP server (pagination, Bearer header, 401), and search
//! over indexed pages scoped by the project → space mapping, against a real Qdrant.

use std::sync::{Arc, Mutex};

use axum::extract::{Path, Query, State};
use axum::http::HeaderMap;
use axum::routing::get;
use axum::{Json, Router};
use ctx::confluence::{Confluence, ConfluenceSearch};
use embed::{Embedder, HashEmbedder};
use serde_json::{Value, json};
use sources::confluence::ConfluenceError;
use store::Store;

#[derive(Clone, Default)]
struct Seen {
    auth: Arc<Mutex<Vec<String>>>,
    starts: Arc<Mutex<Vec<String>>>,
}

/// 5 pages in SEDN served 2 per response regardless of the limit asked, like a server cap.
async fn serve() -> (String, Seen) {
    let seen = Seen::default();
    let content = |headers: HeaderMap, State(s): State<Seen>, Query(q): Query<std::collections::HashMap<String, String>>| async move {
        s.auth.lock().unwrap().push(headers.get("authorization").and_then(|h| h.to_str().ok()).unwrap_or("").to_owned());
        s.starts.lock().unwrap().push(q.get("start").cloned().unwrap_or_default());
        let start: usize = q.get("start").and_then(|v| v.parse().ok()).unwrap_or(0);
        let all: Vec<Value> = (1..=5).map(|i| json!({"id": i.to_string(), "title": format!("P{i}"), "version": {"number": i * 10}})).collect();
        let page: Vec<Value> = all.iter().skip(start).take(2).cloned().collect();
        let mut body = json!({"results": page, "start": start, "limit": 2, "size": page.len()});
        if start + 2 < all.len() {
            body["_links"] = json!({"next": format!("/rest/api/content?start={}", start + 2)});
        }
        Json(body)
    };
    let one = |Path(id): Path<String>| async move {
        Json(json!({
            "id": id, "title": "Runbook", "space": {"key": "SEDN"},
            "version": {"number": 3, "when": "2026-10-01T10:00:00.000+02:00", "by": {"displayName": "Anna W"}},
            "ancestors": [{"title": "Home"}],
            "body": {"storage": {"value": "<p>restart &amp; check</p>"}},
            "_links": {"webui": format!("/pages/viewpage.action?pageId={id}")}
        }))
    };
    let spaces = || async {
        Json(json!({"results": [{"key": "SEDN", "name": "System Elektronicznych Dyplomów"}, {"key": "HD", "name": "Helpdesk"}], "size": 2}))
    };
    let app: Router = Router::new()
        .route("/rest/api/content", get(content))
        .route("/rest/api/content/{id}", get(one))
        .route(
            "/rest/api/content/search",
            get(|Query(q): Query<std::collections::HashMap<String, String>>| async move {
                assert_eq!(q.get("cql").map(String::as_str), Some("ancestor=4 and type=page"));
                Json(json!({"results": [{"id": "41", "title": "Child", "version": {"number": 2}}, {"id": "42", "title": "Grandchild", "version": {"number": 5}}], "size": 2}))
            }),
        )
        .route("/rest/api/space", get(spaces))
        .with_state(seen.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (url, seen)
}

#[tokio::test(flavor = "multi_thread")]
async fn lists_every_page_across_capped_responses_with_the_bearer_token() {
    let (url, seen) = serve().await;
    let (refs, page, spaces) = tokio::task::spawn_blocking(move || {
        let c = Confluence::new(&url, "pat123");
        (c.page_refs("SEDN").unwrap(), c.page("4").unwrap(), c.spaces().unwrap())
    })
    .await
    .unwrap();
    assert_eq!(refs.iter().map(|r| (r.id.as_str(), r.version)).collect::<Vec<_>>(), [("1", 10), ("2", 20), ("3", 30), ("4", 40), ("5", 50)]);
    assert_eq!(*seen.starts.lock().unwrap(), ["0", "2", "4"], "walks on by what came back, not by the limit asked");
    assert!(seen.auth.lock().unwrap().iter().all(|a| a == "Bearer pat123"));
    assert_eq!((page.id.as_str(), page.text.as_str(), page.version), ("4", "restart & check", 3));
    assert!(page.url.ends_with("/pages/viewpage.action?pageId=4"));
    assert_eq!(spaces.iter().map(|s| s.key.as_str()).collect::<Vec<_>>(), ["SEDN", "HD"]);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_page_tree_is_the_root_and_all_its_descendants() {
    let (url, _) = serve().await;
    let tree = tokio::task::spawn_blocking(move || Confluence::new(&url, "pat").tree_refs("4").unwrap()).await.unwrap();
    assert_eq!(tree.iter().map(|r| (r.id.as_str(), r.version)).collect::<Vec<_>>(), [("4", 3), ("41", 2), ("42", 5)]);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_401_maps_to_the_auth_error() {
    let app: Router = Router::new().route("/rest/api/space", get(|| async { (axum::http::StatusCode::UNAUTHORIZED, "nope") }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let err = tokio::task::spawn_blocking(move || Confluence::new(&url, "bad").spaces()).await.unwrap().unwrap_err();
    assert!(matches!(err, ConfluenceError::Auth), "{err}");
    assert!(err.to_string().contains("CONFLUENCE_TOKEN"));
}

#[tokio::test(flavor = "multi_thread")]
async fn search_is_scoped_to_the_projects_spaces() {
    const DIM: usize = 64;
    let q = testkit::qdrant().await;
    let store = Store::connect(&q.url, None, "confluence", DIM).unwrap();
    store.ensure_collection(ingest::confluence::CONFLUENCE_FIELDS).await.unwrap();
    let mut emb = HashEmbedder::new(DIM);
    let docs: Vec<store::Doc> = [("SEDN", "SEDN", "1", "restart the diploma service pod"), ("HD", "HD", "2", "restart the diploma service pod"), ("p77", "ZA", "77", "restart the diploma service pod")]
        .iter()
        .map(|(unit, space, id, body)| {
            let text = format!("[{space} · Runbook {id}]\n{body}");
            let Value::Object(payload) = json!({"kind": "page", "source": "confluence", "unit": unit, "project": space, "page_id": id, "title": format!("Runbook {id}"),
                "url": format!("https://c/{id}"), "author": "Anna W", "ts": 0, "text": text}) else { unreachable!() };
            store::Doc { id: store::point_id("confluence", &format!("{id}/0")), embedding: emb.embed_one(&text).unwrap(), payload }
        })
        .collect();
    store.upsert(docs).await.unwrap();

    let tmp = tempfile::tempdir().unwrap();
    let mapping = tmp.path().join("confluence-projects.json");
    std::fs::write(&mapping, r#"{"sedn/sedn26": ["SEDN", "https://confluence.example.org/spaces/ZA/pages/77/Helpdesk", "77"]}"#).unwrap();
    let search = ConfluenceSearch {
        store: Arc::new(store),
        embedder: Arc::new(Mutex::new(Box::new(HashEmbedder::new(DIM)) as Box<dyn Embedder>)),
        projects: mapping,
        redact: tmp.path().join("redact.txt"),
    };
    let sedn = sources::project::key("d:/dev/sedn/sedn26");
    let scoped = search.search("restart the diploma service", Some(&sedn), 5).await.unwrap();
    assert!(scoped.contains("[SEDN] Runbook 1") && scoped.contains("[ZA] Runbook 77") && !scoped.contains("[HD]"), "a space and a page tree: {scoped}");
    let all = search.search("restart the diploma service", None, 5).await.unwrap();
    assert!(all.contains("[SEDN]") && all.contains("[HD]"), "{all}");
    let unmapped = search.search("restart", Some(&sources::project::key("d:/dev/other")), 5).await.unwrap();
    assert!(unmapped.contains("No Confluence spaces or pages are mapped"), "{unmapped}");
}

#[test]
fn server_instructions_name_the_projects_spaces() {
    let s = ctx::mcp::confluence_instructions(&["SEDN".into(), "SEDNDOC".into()]);
    assert!(s.contains("SEDN, SEDNDOC") && s.contains("search_confluence"), "{s}");
    assert_eq!(ctx::mcp::confluence_instructions(&[]), "");
    let mixed = ctx::mcp::confluence_instructions(&["SEDN".into(), "p77".into(), "p78".into()]);
    assert!(mixed.contains("space(s) SEDN and 2 selected page tree(s)") && !mixed.contains("p77"), "{mixed}");
}

/// A space of `total` pages served `per_response` at a time; `_links.next` only when `with_next`
/// and more remain.
async fn serve_listing(total: usize, per_response: usize, with_next: bool) -> (String, Seen) {
    let seen = Seen::default();
    let content = move |State(s): State<Seen>, Query(q): Query<std::collections::HashMap<String, String>>| async move {
        s.starts.lock().unwrap().push(q.get("start").cloned().unwrap_or_default());
        let start: usize = q.get("start").and_then(|v| v.parse().ok()).unwrap_or(0);
        let page: Vec<Value> = (start..total.min(start + per_response)).map(|i| json!({"id": i.to_string(), "title": format!("P{i}"), "version": {"number": 1}})).collect();
        let mut body = json!({"results": page, "start": start, "size": page.len()});
        if with_next && start + per_response < total {
            body["_links"] = json!({"next": "/rest/api/content?next"});
        }
        Json(body)
    };
    let app: Router = Router::new().route("/rest/api/content", get(content)).with_state(seen.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (url, seen)
}

#[tokio::test(flavor = "multi_thread")]
async fn a_full_response_without_a_next_link_is_the_last_one() {
    let (url, seen) = serve_listing(100, 100, false).await;
    let refs = tokio::task::spawn_blocking(move || Confluence::new(&url, "t").page_refs("SEDN").unwrap()).await.unwrap();
    assert_eq!(refs.len(), 100);
    assert_eq!(*seen.starts.lock().unwrap(), ["0"], "no next link: no second request");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_listing_spanning_full_responses_is_walked_to_the_end() {
    let (url, seen) = serve_listing(250, 100, true).await;
    let refs = tokio::task::spawn_blocking(move || Confluence::new(&url, "t").page_refs("SEDN").unwrap()).await.unwrap();
    assert_eq!(refs.len(), 250);
    assert_eq!(refs.iter().map(|r| r.id.as_str()).collect::<std::collections::HashSet<_>>().len(), 250, "no page listed twice");
    assert_eq!(*seen.starts.lock().unwrap(), ["0", "100", "200"]);
}

#[tokio::test(flavor = "multi_thread")]
async fn an_empty_space_lists_no_pages_in_one_request() {
    let (url, seen) = serve_listing(0, 100, true).await;
    let refs = tokio::task::spawn_blocking(move || Confluence::new(&url, "t").page_refs("EMPTY").unwrap()).await.unwrap();
    assert!(refs.is_empty());
    assert_eq!(seen.starts.lock().unwrap().len(), 1);
}

/// `/rest/api/content` answers one full page with a `next` link, then a reply without `results`;
/// `/rest/api/space` answers without `results` at once.
async fn serve_without_results() -> (String, Seen) {
    let seen = Seen::default();
    let content = |State(s): State<Seen>, Query(q): Query<std::collections::HashMap<String, String>>| async move {
        let start = q.get("start").cloned().unwrap_or_default();
        s.starts.lock().unwrap().push(start.clone());
        if start == "0" {
            Json(json!({"results": [{"id": "7", "title": "P7", "version": {"number": 1}}], "size": 1, "_links": {"next": "/rest/api/content?start=1"}}))
        } else {
            Json(json!({"size": 0, "message": "temporarily unavailable"}))
        }
    };
    let app: Router = Router::new()
        .route("/rest/api/content", get(content))
        .route("/rest/api/space", get(|| async { Json(json!({"size": 0})) }))
        .with_state(seen.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (url, seen)
}

#[tokio::test(flavor = "multi_thread")]
async fn a_reply_without_results_is_a_parse_error_not_an_empty_listing() {
    let (url, _) = serve_without_results().await;
    let err = tokio::task::spawn_blocking(move || Confluence::new(&url, "t").spaces()).await.unwrap().unwrap_err();
    assert!(matches!(err, ConfluenceError::Parse(_)), "{err:?}");
    assert_eq!(err.to_string(), "Confluence response could not be parsed: /rest/api/space?type=global: reply without a results list");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_later_page_without_results_fails_the_whole_listing() {
    let (url, seen) = serve_without_results().await;
    let err = tokio::task::spawn_blocking(move || Confluence::new(&url, "t").page_refs("SEDN")).await.unwrap().unwrap_err();
    assert!(matches!(err, ConfluenceError::Parse(_)), "{err:?}");
    assert_eq!(
        err.to_string(),
        "Confluence response could not be parsed: /rest/api/content?spaceKey=SEDN&type=page&status=current&expand=version: reply without a results list"
    );
    assert_eq!(*seen.starts.lock().unwrap(), ["0", "1"], "the first page was read, then the reply without results");
}

/// Two pages of SEDN: one last edited in 2024, one in 2026.
struct AgedPages;
impl ingest::confluence::PageSource for AgedPages {
    fn page_refs(&self, _space: &str) -> Result<Vec<sources::confluence::PageRef>, String> {
        Ok(["old", "new"].iter().map(|id| sources::confluence::PageRef { id: (*id).into(), title: (*id).into(), version: 1 }).collect())
    }
    fn tree_refs(&self, root: &str) -> Result<Vec<sources::confluence::PageRef>, String> {
        Err(format!("no tree {root}"))
    }
    fn page(&self, id: &str) -> Result<sources::confluence::Page, String> {
        let updated = if id == "old" { "2024-03-01T10:00:00.000+01:00" } else { "2026-09-01T10:00:00.000+02:00" };
        Ok(sources::confluence::Page {
            id: id.into(),
            space: "SEDN".into(),
            title: format!("Runbook {id}"),
            version: 1,
            updated: updated.into(),
            author: "Anna W".into(),
            ancestors: vec![],
            text: format!("procedure {id}"),
            url: format!("https://c/{id}"),
        })
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn the_prune_cutoff_keeps_pages_edited_before_it_out_of_the_index() {
    const DIM: usize = 64;
    let q = testkit::qdrant().await;
    let store = Store::connect(&q.url, None, "confluence", DIM).unwrap();
    store.ensure_collection(ingest::confluence::CONFLUENCE_FIELDS).await.unwrap();
    let state = ingest::State::in_memory().unwrap();
    let cutoff = jiff::civil::date(2025, 1, 1).to_zoned(jiff::tz::TimeZone::UTC).unwrap().timestamp().as_millisecond();
    let config = ingest::IndexConfig { prune_before_ms: Some(cutoff), ..ingest::IndexConfig::default() };
    let mut emb = HashEmbedder::new(DIM);

    let r = ingest::confluence::sync_confluence(&store, &mut emb, &ingest::WordCounter, &state, &AgedPages, &["SEDN".to_owned()], &config, 0).await.unwrap();

    assert_eq!(r.report.failed, Vec::<(String, String)>::new());
    let page = |id: &'static str| store::SearchFilter::default().eq("page_id", id);
    assert_eq!(store.count(&page("old")).await.unwrap(), 0, "edited in 2024, before the cutoff");
    assert_eq!(store.count(&page("new")).await.unwrap(), 1, "edited in 2026");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn run_confluence_index_re_embeds_a_point_of_the_previous_model_and_records_the_new_version() {
    const DIM: usize = 64;
    let (url, _) = serve().await;
    let q = testkit::qdrant().await;
    let store = Store::connect(&q.url, None, "confluence", DIM).unwrap();
    store.ensure_collection(ingest::confluence::CONFLUENCE_FIELDS).await.unwrap();
    let (old, new) = (ctx::index_version("m1"), ctx::index_version("m2"));
    // A page of a unit that is not mapped: no listing covers it, only the re-embed pass.
    let Value::Object(payload) =
        json!({"kind": "page", "source": "confluence", "unit": "OTHER", "page_id": "seeded", "text": "an m1 page", "ts": 1, ingest::VERSION_FIELD: old})
    else {
        unreachable!()
    };
    store.upsert(vec![store::Doc { id: store::point_id("confluence", "OTHER/seeded/0"), embedding: HashEmbedder::new(DIM).embed_one("old vector").unwrap(), payload }]).await.unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let mapping = tmp.path().join("confluence-projects.json");
    std::fs::write(&mapping, r#"{"work/sedn": ["SEDN"]}"#).unwrap();
    let cfg = ctx::config::Config {
        confluence_url: Some(url),
        confluence_token: Some("pat".into()),
        confluence_projects: mapping,
        ..ctx::config::Config::from_env(|_| None)
    };
    let state = ingest::State::open(&tmp.path().join("state.db")).unwrap();
    state.ensure_version(&old).unwrap();
    state.set_meta("index_version/confluence", &old).unwrap();

    let mut emb = HashEmbedder::new(DIM);
    let r = ctx::run_confluence_index(&cfg, &store, &mut emb, "m2", &ingest::WordCounter, &state, &ingest::Redactor::default()).await.unwrap();

    assert_eq!((r.report.indexed, r.report.failed.len()), (5, 0), "{:?}", r.report);
    let hits = store.scroll(&store::SearchFilter::default().eq("page_id", "seeded"), 10).await.unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].payload[ingest::VERSION_FIELD], json!(new));
    assert_eq!(state.get_meta("index_version/confluence").unwrap(), Some(new));
}

#[tokio::test(flavor = "multi_thread")]
async fn pruned_pages_stay_out_after_the_next_sync() {
    const DIM: usize = 64;
    let q = testkit::qdrant().await;
    let store = Store::connect(&q.url, None, "confluence", DIM).unwrap();
    store.ensure_collection(ingest::confluence::CONFLUENCE_FIELDS).await.unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let state_path = tmp.path().join("state.db");
    let sync = || async {
        let state = ingest::State::open(&state_path).unwrap();
        let mut emb = HashEmbedder::new(DIM);
        let config = ctx::index_config(&store, &state, &ingest::Redactor::default()).unwrap();
        ingest::confluence::sync_confluence(&store, &mut emb, &ingest::WordCounter, &state, &AgedPages, &["SEDN".to_owned()], &config, 0).await.unwrap()
    };
    let page = |id: &'static str| store::SearchFilter::default().eq("page_id", id);
    sync().await;

    let cutoff = jiff::civil::date(2025, 1, 1).to_zoned(jiff::tz::TimeZone::UTC).unwrap().timestamp().as_millisecond();
    let done = ctx::prune::prune_older_than(&store, &state_path, "page_key", cutoff, true).await.unwrap();
    assert_eq!(done.items, 1, "{done:?}");
    assert_eq!(store.count(&page("old")).await.unwrap(), 0);
    assert!(store.count(&page("new")).await.unwrap() > 0);
    let state = ingest::State::open(&state_path).unwrap();
    assert!(state.get("confluence", "SEDN/old").unwrap().is_none(), "the pruned page's state is forgotten");
    assert!(state.get("confluence", "SEDN/new").unwrap().is_some());
    drop(state);

    let again = sync().await;
    assert_eq!(again.report.failed, Vec::<(String, String)>::new());
    assert_eq!(store.count(&page("old")).await.unwrap(), 0, "the recorded cutoff keeps the pruned page out");
    assert!(store.count(&page("new")).await.unwrap() > 0);
}
