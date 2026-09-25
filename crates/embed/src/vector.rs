use crate::{EmbedError, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct Embedding {
    pub dense: Vec<f32>,
    pub sparse: SparseVector,
}

/// Sparse vector keyed by token id. Invariant: indices strictly ascending, same length as values,
/// all values finite — Qdrant rejects duplicate indices, and sorted order makes equality stable.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SparseVector {
    indices: Vec<u32>,
    values: Vec<f32>,
}

impl SparseVector {
    pub fn new(indices: Vec<u32>, values: Vec<f32>) -> Result<Self> {
        if indices.len() != values.len() {
            return Err(EmbedError::InvalidSparse(format!(
                "{} indices but {} values",
                indices.len(),
                values.len()
            )));
        }
        if let Some(w) = indices.windows(2).find(|w| w[0] >= w[1]) {
            return Err(EmbedError::InvalidSparse(format!(
                "indices not strictly ascending at {} -> {}",
                w[0], w[1]
            )));
        }
        if let Some(v) = values.iter().find(|v| !v.is_finite()) {
            return Err(EmbedError::InvalidSparse(format!("non-finite value {v}")));
        }
        Ok(Self { indices, values })
    }

    /// Builds from unordered `(index, weight)` pairs; a repeated index keeps its maximum weight,
    /// matching how bge-m3 pools lexical weights for a token that occurs several times.
    pub fn from_pairs(pairs: impl IntoIterator<Item = (u32, f32)>) -> Result<Self> {
        let mut pairs: Vec<(u32, f32)> = pairs.into_iter().collect();
        pairs.sort_unstable_by_key(|&(i, _)| i);
        pairs.dedup_by(|later, kept| {
            if later.0 == kept.0 {
                kept.1 = kept.1.max(later.1);
                true
            } else {
                false
            }
        });
        let (indices, values) = pairs.into_iter().unzip();
        Self::new(indices, values)
    }

    #[cfg(feature = "bgem3")]
    pub(crate) fn from_fastembed(s: fastembed::SparseEmbedding) -> Result<Self> {
        let indices = s
            .indices
            .into_iter()
            .map(|i| {
                u32::try_from(i)
                    .map_err(|_| EmbedError::InvalidSparse(format!("token id {i} exceeds u32")))
            })
            .collect::<Result<Vec<_>>>()?;
        Self::from_pairs(indices.into_iter().zip(s.values))
    }

    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn into_parts(self) -> (Vec<u32>, Vec<f32>) {
        (self.indices, self.values)
    }

    /// Dot product over shared indices — the score Qdrant computes for sparse vectors.
    pub fn dot(&self, other: &SparseVector) -> f32 {
        let (mut i, mut j, mut acc) = (0, 0, 0.0);
        while i < self.indices.len() && j < other.indices.len() {
            match self.indices[i].cmp(&other.indices[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    acc += self.values[i] * other.values[j];
                    i += 1;
                    j += 1;
                }
            }
        }
        acc
    }
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_accepts_valid_vector() {
        let v = SparseVector::new(vec![1, 5, 9], vec![0.1, 0.2, 0.3]).unwrap();
        assert_eq!(v.indices(), &[1, 5, 9]);
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn new_rejects_length_mismatch() {
        assert!(SparseVector::new(vec![1, 2], vec![0.1]).is_err());
    }

    #[test]
    fn new_rejects_unsorted_and_duplicate_indices() {
        assert!(SparseVector::new(vec![2, 1], vec![0.1, 0.2]).is_err());
        assert!(SparseVector::new(vec![1, 1], vec![0.1, 0.2]).is_err());
    }

    #[test]
    fn new_rejects_non_finite_values() {
        assert!(SparseVector::new(vec![1], vec![f32::NAN]).is_err());
        assert!(SparseVector::new(vec![1], vec![f32::INFINITY]).is_err());
    }

    #[test]
    fn empty_vector_is_valid() {
        let v = SparseVector::new(vec![], vec![]).unwrap();
        assert!(v.is_empty());
    }

    #[test]
    fn from_pairs_sorts_and_keeps_max_of_duplicates() {
        let v = SparseVector::from_pairs([(7, 0.2), (3, 0.5), (7, 0.9), (7, 0.1)]).unwrap();
        assert_eq!(v.indices(), &[3, 7]);
        assert_eq!(v.values(), &[0.5, 0.9]);
    }

    #[cfg(feature = "bgem3")]
    #[test]
    fn from_fastembed_rejects_ids_beyond_u32() {
        let s = fastembed::SparseEmbedding {
            indices: vec![u32::MAX as usize + 1],
            values: vec![1.0],
        };
        assert!(matches!(
            SparseVector::from_fastembed(s),
            Err(EmbedError::InvalidSparse(_))
        ));
    }

    #[cfg(feature = "bgem3")]
    #[test]
    fn from_fastembed_normalises_order() {
        let s = fastembed::SparseEmbedding {
            indices: vec![10, 2],
            values: vec![0.3, 0.4],
        };
        let v = SparseVector::from_fastembed(s).unwrap();
        assert_eq!(v.indices(), &[2, 10]);
        assert_eq!(v.values(), &[0.4, 0.3]);
    }

    #[test]
    fn dot_uses_only_shared_indices() {
        let a = SparseVector::new(vec![1, 3, 5], vec![1.0, 2.0, 3.0]).unwrap();
        let b = SparseVector::new(vec![3, 4, 5], vec![10.0, 100.0, 1.0]).unwrap();
        assert_eq!(a.dot(&b), 2.0 * 10.0 + 3.0 * 1.0);
        assert_eq!(a.dot(&SparseVector::default()), 0.0);
    }

    #[test]
    fn cosine_handles_zero_vectors() {
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
        assert!((cosine(&[1.0, 0.0], &[2.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[-1.0, 0.0]) < -0.99);
    }
}
