use crate::{Embedder, Embedding, Result, SparseVector};

/// Deterministic bag-of-words embedder with no model: texts sharing words get similar dense
/// vectors and overlapping sparse indices. For tests of indexing and search plumbing only — it has
/// no notion of meaning, so it must never back a real index.
#[derive(Debug, Clone)]
pub struct HashEmbedder {
    dim: usize,
}

impl HashEmbedder {
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "dimension must be positive");
        Self { dim }
    }

    fn embed_text(&self, text: &str) -> Result<Embedding> {
        let mut dense = vec![0.0f32; self.dim];
        let mut pairs = Vec::new();
        for token in tokens(text) {
            let h = fnv1a(&token);
            let slot = (h % self.dim as u64) as usize;
            dense[slot] += if h & (1 << 63) == 0 { 1.0 } else { -1.0 };
            pairs.push(((h >> 32) as u32, 1.0));
        }
        let norm = dense.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm == 0.0 {
            // Cosine against a zero vector is undefined, so text without words gets a fixed axis.
            dense[0] = 1.0;
        } else {
            dense.iter_mut().for_each(|x| *x /= norm);
        }
        Ok(Embedding { dense, sparse: SparseVector::from_pairs(pairs)? })
    }
}

impl Embedder for HashEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&mut self, texts: &[&str]) -> Result<Vec<Embedding>> {
        texts.iter().map(|t| self.embed_text(t)).collect()
    }
}

fn tokens(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
}

// std's DefaultHasher is not guaranteed stable across Rust releases; tests pin exact vectors.
fn fnv1a(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::cosine;

    #[test]
    fn output_has_one_entry_per_input_in_order() {
        let mut e = HashEmbedder::new(64);
        let out = e.embed(&["alpha", "beta", "alpha"]).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], out[2]);
        assert_ne!(out[0], out[1]);
    }

    #[test]
    fn empty_input_gives_empty_output() {
        assert!(HashEmbedder::new(8).embed(&[]).unwrap().is_empty());
    }

    #[test]
    fn dense_vectors_are_unit_length_with_requested_dim() {
        let mut e = HashEmbedder::new(32);
        for text in ["one two three", "", "!!!", "Zażółć gęślą jaźń"] {
            let v = e.embed_one(text).unwrap();
            assert_eq!(v.dense.len(), 32);
            let norm = v.dense.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "norm {norm} for {text:?}");
        }
    }

    #[test]
    fn is_case_and_punctuation_insensitive() {
        let mut e = HashEmbedder::new(64);
        assert_eq!(e.embed_one("Login, FAILED!").unwrap(), e.embed_one("login failed").unwrap());
    }

    #[test]
    fn shared_words_score_higher_than_disjoint_words() {
        let mut e = HashEmbedder::new(256);
        let q = e.embed_one("database migration timeout").unwrap();
        let near = e.embed_one("the database migration hit a timeout").unwrap();
        let far = e.embed_one("css button colour").unwrap();
        assert!(cosine(&q.dense, &near.dense) > cosine(&q.dense, &far.dense));
        assert!(q.sparse.dot(&near.sparse) > q.sparse.dot(&far.sparse));
    }

    #[test]
    fn sparse_has_one_entry_per_distinct_word() {
        let v = HashEmbedder::new(16).embed_one("a b a c b a").unwrap();
        assert_eq!(v.sparse.len(), 3);
        assert!(v.sparse.values().iter().all(|&w| w == 1.0));
    }

    #[test]
    fn hashing_is_stable_across_runs() {
        // Published FNV-1a 64-bit test vectors.
        assert_eq!(fnv1a(""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a("a"), 0xaf63_dc4c_8601_ec8c);
    }

    #[test]
    #[should_panic(expected = "dimension must be positive")]
    fn zero_dimension_is_rejected() {
        HashEmbedder::new(0);
    }
}
