//! Text embeddings for the context index: one dense and one sparse (lexical) vector per text.
//!
//! [`Bgem3Embedder`] is the production implementation (bge-m3, local ONNX). [`HashEmbedder`] is a
//! deterministic, model-free stand-in so crates above this one can be tested without a 570 MB
//! model download.

#[cfg(feature = "bgem3")]
mod bgem3;
mod hash;
mod vector;

#[cfg(feature = "bgem3")]
pub use bgem3::{BGEM3_DIM, Bgem3Embedder, Bgem3Options, ModelSource, Provider};
pub use hash::HashEmbedder;
pub use vector::{Embedding, SparseVector, cosine};

#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    #[error("model error: {0}")]
    Model(String),
    #[error("invalid sparse vector: {0}")]
    InvalidSparse(String),
    #[error("model returned {got} embeddings for {expected} inputs")]
    CountMismatch { expected: usize, got: usize },
    #[error("execution provider {0} is not compiled in; rebuild with `--features {1}`")]
    ProviderUnavailable(&'static str, &'static str),
}

pub type Result<T> = std::result::Result<T, EmbedError>;

pub trait Embedder: Send {
    /// Dimension of every dense vector this embedder returns.
    fn dim(&self) -> usize;

    /// Embeds `texts` in order; the result has exactly one entry per input.
    fn embed(&mut self, texts: &[&str]) -> Result<Vec<Embedding>>;

    fn embed_one(&mut self, text: &str) -> Result<Embedding> {
        let mut out = self.embed(&[text])?;
        out.pop().ok_or(EmbedError::CountMismatch { expected: 1, got: 0 })
    }
}
