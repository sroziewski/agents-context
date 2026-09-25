//! Local embedding daemon. One process owns the model: every Claude/OpenCode session starts its
//! own MCP server, and each loading 2.3 GB of weights would not fit the 4 GB GPU twice.

use std::sync::{Arc, Mutex};

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use embed::{Embedder, Embedding, SparseVector};
use serde::{Deserialize, Serialize};

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

#[derive(Clone)]
struct AppState {
    embedder: Arc<Mutex<Box<dyn Embedder>>>,
    health: Health,
}

pub fn router(embedder: Box<dyn Embedder>, model: &str) -> Router {
    let health = Health { model: model.to_owned(), dim: embedder.dim() };
    let state = AppState { embedder: Arc::new(Mutex::new(embedder)), health };
    Router::new()
        .route("/health", get(|State(s): State<AppState>| async move { Json(s.health) }))
        .route("/embed", post(embed))
        .with_state(state)
}

async fn embed(
    State(s): State<AppState>,
    Json(req): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, (StatusCode, String)> {
    if req.texts.len() > MAX_TEXTS {
        return Err((StatusCode::PAYLOAD_TOO_LARGE, format!("at most {MAX_TEXTS} texts per request")));
    }
    let embedder = s.embedder.clone();
    // Inference is CPU/GPU-bound and synchronous; keep it off the async workers.
    let out = tokio::task::spawn_blocking(move || {
        let refs: Vec<&str> = req.texts.iter().map(String::as_str).collect();
        let mut guard = embedder.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.embed(&refs)
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

pub async fn serve(router: Router, addr: std::net::SocketAddr) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!("embedding daemon listening on http://{addr}");
    axum::serve(listener, router).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use embed::HashEmbedder;

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
