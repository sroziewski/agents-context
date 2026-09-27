//! Cross-encoder reranking of retrieval candidates.
//!
//! A reranker reads the query and a candidate document together and scores their relevance —
//! more accurate than the bi-encoder cosine used for retrieval, but too slow to run over the
//! whole index, so it only reorders a shortlist. Per DEV-28 it runs CPU-only: bge-m3 keeps the
//! GPU, and scoring a few dozen candidates on CPU is within the search budget.

use crate::Result;

/// Scores each document against the query; higher is more relevant. One score per document, in
/// input order. Blocking, like [`crate::Embedder`].
pub trait Reranker: Send {
    fn rerank(&mut self, query: &str, docs: &[&str]) -> Result<Vec<f32>>;

    /// Indices of `docs`, best first. Ties keep the input order (stable sort).
    fn order(&mut self, query: &str, docs: &[&str]) -> Result<Vec<usize>> {
        let scores = self.rerank(query, docs)?;
        let mut idx: Vec<usize> = (0..docs.len()).collect();
        idx.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
        Ok(idx)
    }
}

fn words(text: &str) -> std::collections::HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(str::to_lowercase).collect()
}

/// Model-free reranker for tests above this crate: score = number of distinct lowercase words a
/// document shares with the query. Deterministic, no ONNX. Never a real reranker.
#[derive(Debug, Clone, Default)]
pub struct HashReranker;

impl Reranker for HashReranker {
    fn rerank(&mut self, query: &str, docs: &[&str]) -> Result<Vec<f32>> {
        let q = words(query);
        Ok(docs.iter().map(|d| words(d).intersection(&q).count() as f32).collect())
    }
}

#[cfg(feature = "bgem3")]
pub use fb::{FastembedReranker, RerankModel};

#[cfg(feature = "bgem3")]
mod fb {
    use std::str::FromStr;

    use fastembed::{RerankInitOptions, RerankerModel, TextRerank};

    use crate::{EmbedError, Reranker, Result};

    /// The fastembed reranker models worth considering (DEV-28). jina-v2 is the main candidate:
    /// multilingual (covers Polish) and half the size of bge-v2-m3.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum RerankModel {
        JinaV2Multilingual,
        BgeV2M3,
        /// English/Chinese only — a control, not for Polish.
        BgeBase,
    }

    impl FromStr for RerankModel {
        type Err = String;

        fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
            match s.to_ascii_lowercase().as_str() {
                "jina-v2" | "jina" => Ok(Self::JinaV2Multilingual),
                "bge-v2-m3" | "bge-m3" => Ok(Self::BgeV2M3),
                "bge-base" => Ok(Self::BgeBase),
                other => Err(format!("unknown reranker {other:?}; expected jina-v2, bge-v2-m3 or bge-base")),
            }
        }
    }

    impl RerankModel {
        fn to_fastembed(self) -> RerankerModel {
            match self {
                Self::JinaV2Multilingual => RerankerModel::JINARerankerV2BaseMultiligual,
                Self::BgeV2M3 => RerankerModel::BGERerankerV2M3,
                Self::BgeBase => RerankerModel::BGERerankerBase,
            }
        }
    }

    /// A fastembed cross-encoder on CPU. The model is fetched from Hugging Face on first use and
    /// cached; loading is slow (seconds), so construct once and reuse.
    pub struct FastembedReranker {
        model: TextRerank,
        batch: usize,
    }

    impl FastembedReranker {
        /// `intra_threads` caps ONNX Runtime's CPU threads (None = all cores).
        pub fn hub(model: RerankModel, max_length: usize, intra_threads: Option<usize>, batch: usize) -> Result<Self> {
            // Default execution providers = ORT's CPU provider, which is what we want here.
            let mut opts = RerankInitOptions::new(model.to_fastembed()).with_max_length(max_length).with_show_download_progress(true);
            if let Some(t) = intra_threads {
                opts = opts.with_intra_threads(t);
            }
            let model = TextRerank::try_new(opts).map_err(|e| EmbedError::Model(e.to_string()))?;
            Ok(Self { model, batch: batch.max(1) })
        }
    }

    impl Reranker for FastembedReranker {
        fn rerank(&mut self, query: &str, docs: &[&str]) -> Result<Vec<f32>> {
            if docs.is_empty() {
                return Ok(Vec::new());
            }
            // return_documents=false: we only need the scores, matched back to input order by index.
            let results = self.model.rerank(query, docs, false, Some(self.batch)).map_err(|e| EmbedError::Model(e.to_string()))?;
            let mut scores = vec![f32::MIN; docs.len()];
            for r in results {
                *scores.get_mut(r.index).ok_or_else(|| EmbedError::Model(format!("reranker returned out-of-range index {}", r.index)))? = r.score;
            }
            Ok(scores)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_reranker_scores_and_orders_by_word_overlap() {
        let mut r = HashReranker;
        let docs = ["nothing in common", "the database migration failed", "database rollback added"];
        let scores = r.rerank("database migration rollback", &docs).unwrap();
        assert_eq!(scores, vec![0.0, 2.0, 2.0]);
        // best first; ties (docs 1 and 2) keep input order.
        assert_eq!(r.order("database migration rollback", &docs).unwrap(), vec![1, 2, 0]);
        assert_eq!(r.rerank("x", &[]).unwrap(), Vec::<f32>::new());
    }
}
