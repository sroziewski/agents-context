//! Local embedding daemon. One process owns the model: every Claude/OpenCode session starts its
//! own MCP server, and each loading 2.3 GB of weights would not fit the 4 GB GPU twice.

use std::sync::{Arc, Mutex};

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use embed::{Embedder, Embedding, SparseVector};
use serde::{Deserialize, Serialize};

use crate::queue::{IndexQueue, Status, Target};

/// Upper bound per request so one client cannot pin the GPU for minutes; the indexer sends
/// small token-capped batches anyway.
pub const MAX_TEXTS: usize = 64;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Health {
    pub model: String,
    pub dim: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbedRequest {
    pub texts: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WireEmbedding {
    pub dense: Vec<f32>,
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

impl From<Embedding> for WireEmbedding {
    fn from(e: Embedding) -> Self {
        let (indices, values) = e.sparse.into_parts();
        Self { dense: e.dense, indices, values }
    }
}

impl TryFrom<WireEmbedding> for Embedding {
    type Error = embed::EmbedError;

    fn try_from(w: WireEmbedding) -> Result<Self, Self::Error> {
        Ok(Embedding { dense: w.dense, sparse: SparseVector::new(w.indices, w.values)? })
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbedResponse {
    pub model: String,
    pub embeddings: Vec<WireEmbedding>,
}

/// Embedder shared by the HTTP handlers and the background indexer. The lock is taken per
/// `embed` call, so a search query waits for at most one indexing batch, not a whole run.
#[derive(Clone)]
pub struct SharedEmbedder(pub Arc<Mutex<Box<dyn Embedder>>>);

impl SharedEmbedder {
    pub fn new(e: Box<dyn Embedder>) -> Self {
        Self(Arc::new(Mutex::new(e)))
    }
}

impl Embedder for SharedEmbedder {
    fn dim(&self) -> usize {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).dim()
    }

    fn embed(&mut self, texts: &[&str]) -> embed::Result<Vec<Embedding>> {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).embed(texts)
    }
}

/// Builds the reranker on first use. Returns a boxed model so the daemon and tests share the
/// route: production passes a factory that loads the CPU cross-encoder; tests pass one that
/// returns a [`embed::HashReranker`].
pub type RerankerFactory = Arc<dyn Fn() -> embed::Result<Box<dyn embed::Reranker>> + Send + Sync>;

/// Lazily-loaded reranker, kept behind its own lock so it never waits on the embedder mutex the
/// indexer holds. Loading is slow (model download + init), so it happens on the first `/rerank`.
struct RerankerState {
    make: RerankerFactory,
    slot: Mutex<Option<Box<dyn embed::Reranker>>>,
}

#[derive(Clone)]
struct AppState {
    embedder: SharedEmbedder,
    health: Health,
    queue: Option<IndexQueue>,
    injector: Option<Arc<crate::inject::Injector>>,
    reranker: Option<Arc<RerankerState>>,
}

/// Largest candidate set one `/rerank` call scores; deep search sends a few dozen.
pub const MAX_RERANK_DOCS: usize = 64;

#[derive(Debug, Serialize, Deserialize)]
pub struct RerankRequest {
    pub query: String,
    pub docs: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RerankResponse {
    /// One score per input document, in order; higher is more relevant.
    pub scores: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IndexRequest {
    pub source: Target,
}

pub fn router(embedder: Box<dyn Embedder>, model: &str) -> Router {
    service_router(SharedEmbedder::new(embedder), model, None)
}

/// Full daemon API. `/index` exists only when an index queue is attached.
pub fn service_router(embedder: SharedEmbedder, model: &str, queue: Option<IndexQueue>) -> Router {
    daemon_router(embedder, model, queue, None, None)
}

/// [`service_router`] plus `/context` (auto-inject) when an injector is attached and `/rerank`
/// (deep search) when a reranker factory is attached.
pub fn daemon_router(
    embedder: SharedEmbedder,
    model: &str,
    queue: Option<IndexQueue>,
    injector: Option<Arc<crate::inject::Injector>>,
    reranker: Option<RerankerFactory>,
) -> Router {
    let health = Health { model: model.to_owned(), dim: embedder.dim() };
    let reranker = reranker.map(|make| Arc::new(RerankerState { make, slot: Mutex::new(None) }));
    let state = AppState { embedder, health, queue, injector, reranker };
    Router::new()
        .route("/health", get(|State(s): State<AppState>| async move { Json(s.health) }))
        .route("/embed", post(embed))
        .route("/index", post(request_index))
        .route("/index/status", get(index_status))
        .route("/context", post(context))
        .route("/rerank", post(rerank))
        .layer(axum::middleware::from_fn(loopback_host_only))
        .with_state(state)
}

/// Scores `docs` against `query` with the daemon's reranker, loading it on the first call.
/// 404 when no reranker is configured; the CLI/MCP then fall back to the dense order.
async fn rerank(State(s): State<AppState>, Json(req): Json<RerankRequest>) -> Result<Json<RerankResponse>, (StatusCode, String)> {
    let rs = s.reranker.ok_or((StatusCode::NOT_FOUND, "reranking is not set up in this daemon".to_owned()))?;
    if req.docs.len() > MAX_RERANK_DOCS {
        return Err((StatusCode::PAYLOAD_TOO_LARGE, format!("at most {MAX_RERANK_DOCS} documents per request")));
    }
    tokio::task::spawn_blocking(move || {
        let mut slot = rs.slot.lock().unwrap_or_else(|p| p.into_inner());
        if slot.is_none() {
            *slot = Some((rs.make)().map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("loading reranker: {e}")))?);
        }
        let reranker = slot.as_mut().expect("just loaded");
        let refs: Vec<&str> = req.docs.iter().map(String::as_str).collect();
        let scores = reranker.rerank(&req.query, &refs).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        Ok(Json(RerankResponse { scores }))
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
}

/// Rejects requests whose `Host` is not a loopback name. Binding to 127.0.0.1 keeps other
/// machines out, but a web page can still reach it through DNS rebinding (a hostname that
/// resolves to 127.0.0.1); browsers then send that hostname, which this refuses.
async fn loopback_host_only(req: axum::extract::Request, next: axum::middleware::Next) -> axum::response::Response {
    use axum::response::IntoResponse;
    let host = req.headers().get(axum::http::header::HOST).and_then(|h| h.to_str().ok()).unwrap_or("");
    if is_loopback_host(host) {
        next.run(req).await
    } else {
        (StatusCode::FORBIDDEN, "only loopback clients are served").into_response()
    }
}

fn is_loopback_host(host: &str) -> bool {
    let name = match host.strip_prefix('[') {
        Some(v6) => v6.split(']').next().unwrap_or(""),
        None => host.rsplit_once(':').map_or(host, |(h, _)| h),
    };
    matches!(name.to_ascii_lowercase().as_str(), "127.0.0.1" | "localhost" | "::1")
}

async fn request_index(State(s): State<AppState>, Json(req): Json<IndexRequest>) -> StatusCode {
    match s.queue {
        Some(q) => {
            q.request(req.source);
            StatusCode::ACCEPTED
        }
        None => StatusCode::NOT_FOUND,
    }
}

async fn index_status(State(s): State<AppState>) -> Result<Json<Status>, StatusCode> {
    s.queue.map(|q| Json(q.status())).ok_or(StatusCode::NOT_FOUND)
}

async fn context(
    State(s): State<AppState>,
    Json(req): Json<crate::inject::ContextRequest>,
) -> Result<Json<crate::inject::ContextResponse>, (StatusCode, String)> {
    let injector = s.injector.ok_or((StatusCode::NOT_FOUND, "auto-inject is not set up in this daemon".to_owned()))?;
    let resp = injector.context(&req).await.map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")))?;
    if !resp.sessions.is_empty() || !resp.threads.is_empty() {
        eprintln!("[inject] {} sessions, {} threads for {}", resp.sessions.len(), resp.threads.len(), sources::project::key(&req.cwd));
    }
    Ok(Json(resp))
}

async fn embed(
    State(s): State<AppState>,
    Json(req): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, (StatusCode, String)> {
    if req.texts.len() > MAX_TEXTS {
        return Err((StatusCode::PAYLOAD_TOO_LARGE, format!("at most {MAX_TEXTS} texts per request")));
    }
    let mut embedder = s.embedder.clone();
    // Inference is CPU/GPU-bound and synchronous; keep it off the async workers.
    let out = tokio::task::spawn_blocking(move || {
        let refs: Vec<&str> = req.texts.iter().map(String::as_str).collect();
        embedder.embed(&refs)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(EmbedResponse { model: s.health.model, embeddings: out.into_iter().map(Into::into).collect() }))
}

/// [`Embedder`] backed by the daemon over HTTP. Blocking, like every other embedder.
pub struct RemoteEmbedder {
    agent: ureq::Agent,
    base: String,
    health: Health,
}

impl RemoteEmbedder {
    pub fn connect(base: &str) -> anyhow::Result<Self> {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(std::time::Duration::from_secs(120)))
            .build()
            .into();
        let base = base.trim_end_matches('/').to_owned();
        let health: Health = agent
            .get(&format!("{base}/health"))
            .call()
            .map_err(|e| anyhow::anyhow!("embedding daemon at {base} is not reachable ({e}); start it with `ctx embed-server`"))?
            .body_mut()
            .read_json()?;
        Ok(Self { agent, base, health })
    }

    pub fn model(&self) -> &str {
        &self.health.model
    }
}

impl Embedder for RemoteEmbedder {
    fn dim(&self) -> usize {
        self.health.dim
    }

    fn embed(&mut self, texts: &[&str]) -> embed::Result<Vec<Embedding>> {
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(MAX_TEXTS) {
            let req = EmbedRequest { texts: chunk.iter().map(|t| t.to_string()).collect() };
            let resp: EmbedResponse = self
                .agent
                .post(&format!("{}/embed", self.base))
                .send_json(&req)
                .and_then(|mut r| r.body_mut().read_json())
                .map_err(|e| embed::EmbedError::Model(format!("embedding daemon: {e}")))?;
            if resp.model != self.health.model {
                // The daemon was restarted with another model mid-run; vectors would not mix.
                return Err(embed::EmbedError::Model(format!(
                    "daemon model changed from {} to {}",
                    self.health.model, resp.model
                )));
            }
            if resp.embeddings.len() != chunk.len() {
                return Err(embed::EmbedError::CountMismatch { expected: chunk.len(), got: resp.embeddings.len() });
            }
            for w in resp.embeddings {
                out.push(w.try_into()?);
            }
        }
        Ok(out)
    }
}

/// [`RemoteEmbedder`] that connects on first use and reconnects after a failure. The MCP server
/// uses it so that a daemon which is down (not started yet after a reboot, or restarting) only
/// fails the calls that need embeddings instead of killing the server at startup; the agents
/// never retry a server that exited during initialisation.
pub struct LazyRemote {
    url: String,
    dim: usize,
    inner: Option<RemoteEmbedder>,
}

impl LazyRemote {
    pub fn new(url: &str, dim: usize) -> Self {
        Self { url: url.to_owned(), dim, inner: None }
    }
}

impl Embedder for LazyRemote {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&mut self, texts: &[&str]) -> embed::Result<Vec<Embedding>> {
        if self.inner.is_none() {
            let remote = RemoteEmbedder::connect(&self.url).map_err(|e| embed::EmbedError::Model(e.to_string()))?;
            self.inner = Some(remote);
        }
        let result = self.inner.as_mut().expect("connected above").embed(texts);
        if result.is_err() {
            self.inner = None;
        }
        result
    }
}

/// [`embed::Reranker`] backed by the daemon's `/rerank`. Used only by the opt-in deep search, so
/// its timeout is generous — a CPU cross-encoder over a few dozen candidates takes seconds.
pub struct RemoteReranker {
    agent: ureq::Agent,
    url: String,
}

impl RemoteReranker {
    pub fn new(base: &str) -> Self {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(std::time::Duration::from_secs(60)))
            .build()
            .into();
        Self { agent, url: format!("{}/rerank", base.trim_end_matches('/')) }
    }
}

impl embed::Reranker for RemoteReranker {
    fn rerank(&mut self, query: &str, docs: &[&str]) -> embed::Result<Vec<f32>> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }
        let req = RerankRequest { query: query.to_owned(), docs: docs.iter().map(|d| d.to_string()).collect() };
        let resp: RerankResponse = self
            .agent
            .post(&self.url)
            .send_json(&req)
            .and_then(|mut r| r.body_mut().read_json())
            .map_err(|e| embed::EmbedError::Model(format!("rerank daemon: {e}")))?;
        if resp.scores.len() != docs.len() {
            return Err(embed::EmbedError::CountMismatch { expected: docs.len(), got: resp.scores.len() });
        }
        Ok(resp.scores)
    }
}

pub async fn serve(router: Router, addr: std::net::SocketAddr) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!("embedding daemon listening on http://{addr}");
    axum::serve(listener, router).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use embed::{HashEmbedder, Reranker};

    async fn spawn(model: &str) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = router(Box::new(HashEmbedder::new(16)), model);
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{addr}")
    }

    /// ureq is blocking; run client calls off the runtime like real callers do.
    async fn blocking<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
        tokio::task::spawn_blocking(f).await.unwrap()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn remote_embedder_matches_local_embedder() {
        let base = spawn("hash-16").await;
        let (remote, dim, model) = blocking(move || {
            let mut r = RemoteEmbedder::connect(&base).unwrap();
            (r.embed(&["hello world", "zażółć gęślą"]).unwrap(), r.dim(), r.model().to_owned())
        })
        .await;
        assert_eq!((dim, model.as_str()), (16, "hash-16"));
        let local = HashEmbedder::new(16).embed(&["hello world", "zażółć gęślą"]).unwrap();
        assert_eq!(remote, local);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn more_than_max_texts_are_split_across_requests() {
        let base = spawn("hash-16").await;
        let n = blocking(move || {
            let texts: Vec<String> = (0..MAX_TEXTS * 2 + 3).map(|i| format!("t{i}")).collect();
            let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            RemoteEmbedder::connect(&base).unwrap().embed(&refs).unwrap().len()
        })
        .await;
        assert_eq!(n, MAX_TEXTS * 2 + 3);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn oversized_request_is_rejected_by_the_server() {
        let base = spawn("m").await;
        let status = blocking(move || {
            let req = EmbedRequest { texts: vec!["x".into(); MAX_TEXTS + 1] };
            match ureq::post(&format!("{base}/embed")).send_json(&req) {
                Err(ureq::Error::StatusCode(code)) => code,
                other => panic!("expected status error, got {other:?}"),
            }
        })
        .await;
        assert_eq!(status, 413);
    }

    async fn spawn_with_reranker() -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let factory: RerankerFactory = Arc::new(|| Ok(Box::new(embed::HashReranker)));
        let app = daemon_router(SharedEmbedder::new(Box::new(HashEmbedder::new(16))), "m", None, None, Some(factory));
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{addr}")
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn remote_reranker_scores_via_the_daemon() {
        let base = spawn_with_reranker().await;
        let scores = blocking(move || {
            RemoteReranker::new(&base).rerank("database migration", &["unrelated text", "database migration failed"]).unwrap()
        })
        .await;
        // HashReranker scores by shared-word count: doc 1 shares two words, doc 0 none.
        assert_eq!(scores, vec![0.0, 2.0]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn rerank_is_404_without_a_factory_and_413_when_oversized() {
        // No factory: /rerank absent.
        let plain = spawn("m").await;
        let no_reranker = blocking(move || match ureq::post(&format!("{plain}/rerank")).send_json(&RerankRequest { query: "q".into(), docs: vec!["d".into()] }) {
            Err(ureq::Error::StatusCode(code)) => code,
            other => panic!("expected status error, got {other:?}"),
        })
        .await;
        assert_eq!(no_reranker, 404);

        let base = spawn_with_reranker().await;
        let oversized = blocking(move || match ureq::post(&format!("{base}/rerank")).send_json(&RerankRequest { query: "q".into(), docs: vec!["d".into(); MAX_RERANK_DOCS + 1] }) {
            Err(ureq::Error::StatusCode(code)) => code,
            other => panic!("expected status error, got {other:?}"),
        })
        .await;
        assert_eq!(oversized, 413);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn unreachable_daemon_gives_actionable_error() {
        let err = blocking(|| RemoteEmbedder::connect("http://127.0.0.1:9").err().unwrap().to_string()).await;
        assert!(err.contains("ctx embed-server"), "{err}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn lazy_remote_survives_a_daemon_that_starts_late() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        drop(listener);
        let lazy = std::sync::Arc::new(Mutex::new(LazyRemote::new(&url, 16)));

        let l = lazy.clone();
        let down = blocking(move || l.lock().unwrap().embed(&["x"]).err().unwrap().to_string()).await;
        assert!(down.contains("ctx embed-server"), "{down}");

        let listener = tokio::net::TcpListener::bind(url.trim_start_matches("http://")).await.unwrap();
        tokio::spawn(async move { axum::serve(listener, router(Box::new(HashEmbedder::new(16)), "m")).await.unwrap() });
        let l = lazy.clone();
        let up = blocking(move || l.lock().unwrap().embed(&["x"]).map(|v| v.len())).await;
        assert_eq!(up.unwrap(), 1, "reconnects once the daemon is up");
    }

    #[test]
    fn loopback_host_names_are_recognised() {
        for ok in ["127.0.0.1:7997", "localhost:7997", "LOCALHOST", "[::1]:7997", "127.0.0.1"] {
            assert!(is_loopback_host(ok), "{ok}");
        }
        for bad in ["evil.example:7997", "127.0.0.1.evil.example", "", "10.0.0.5:7997", "[::2]:1"] {
            assert!(!is_loopback_host(bad), "{bad}");
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn rebinding_host_header_is_forbidden() {
        let base = spawn("m").await;
        let status = blocking(move || {
            match ureq::get(&format!("{base}/health")).header("Host", "attacker.example:7997").call() {
                Err(ureq::Error::StatusCode(code)) => code,
                other => panic!("expected 403, got {other:?}"),
            }
        })
        .await;
        assert_eq!(status, 403);
    }

    #[test]
    fn lazy_remote_reports_dim_without_connecting() {
        assert_eq!(LazyRemote::new("http://127.0.0.1:9", 1024).dim(), 1024);
    }

    #[test]
    fn wire_format_round_trips_and_validates_sparse() {
        let e = HashEmbedder::new(8).embed_one("a b c").unwrap();
        let back: Embedding = WireEmbedding::from(e.clone()).try_into().unwrap();
        assert_eq!(back, e);
        let bad = WireEmbedding { dense: vec![1.0], indices: vec![2, 1], values: vec![1.0, 1.0] };
        assert!(Embedding::try_from(bad).is_err());
    }
}
