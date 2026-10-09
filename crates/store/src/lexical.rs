//! BM25 as a Qdrant sparse vector: documents carry tf-saturated term weights, Qdrant's IDF
//! modifier supplies the corpus half at query time, and a query is its distinct terms at weight
//! 1. Score = Σ idf(t) · tf·(k1+1) / (tf + k1·(1 − b + b·len/avg)), Okapi BM25.
//!
//! Terms are the words SQLite FTS5's `unicode61` tokenizer yields, minus diacritic folding
//! (lowercase runs of letters and digits, so `SEDN26-1910` gives `sedn26` + `1910`), plus each
//! joined token whole (`sedn26-1910`, `token_stream`). The words let a partial spelling match;
//! the whole token makes the exact identifier outrank turns that merely share its parts. No
//! stemming: this leg exists for exact tokens, and the dense leg covers inflected Polish.

use std::collections::BTreeMap;

const K1: f32 = 1.2;
const B: f32 = 0.75;
/// Average document length in tokens. A constant rather than a corpus statistic, so a document's
/// weights never change after it is written; it only sets how strongly length is normalised.
const AVG_LEN: f32 = 256.0;

pub fn tokens(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(str::to_lowercase)
}

/// Sentence punctuation around a joined token, not part of it.
const EDGES: &[char] = &['.', ',', ';', ':', '!', '?', '(', ')', '[', ']', '{', '}', '<', '>', '"', '\'', '`', '*'];

/// Whitespace-separated tokens that join several words (`token_stream`, `sedn26-1910`,
/// `crate::eval`), lowercased; plain words are already covered by [`tokens`].
fn compounds(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split_whitespace()
        .map(|t| t.trim_matches(EDGES))
        .filter(|t| t.chars().any(char::is_alphanumeric) && t.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).count() > 1)
        .map(str::to_lowercase)
}

fn terms(text: &str) -> impl Iterator<Item = String> + '_ {
    tokens(text).chain(compounds(text))
}

/// Term id: FNV-1a, stable across processes and versions (unlike `std`'s hasher), so stored
/// vectors and queries always agree. 32 bits: collisions are rare and only blur two terms'
/// statistics; ids stay unique within a vector.
fn term_id(token: &str) -> u32 {
    token.bytes().fold(0x811c_9dc5, |h, b| (h ^ u32::from(b)).wrapping_mul(0x0100_0193))
}

/// Document vector as sorted `(indices, values)`; empty for a text without tokens.
pub fn document(text: &str) -> (Vec<u32>, Vec<f32>) {
    let mut tf: BTreeMap<u32, f32> = BTreeMap::new();
    let mut len = 0usize;
    for t in terms(text) {
        *tf.entry(term_id(&t)).or_default() += 1.0;
        len += 1;
    }
    let norm = K1 * (1.0 - B + B * len as f32 / AVG_LEN);
    tf.into_iter().map(|(id, f)| (id, f * (K1 + 1.0) / (f + norm))).unzip()
}

/// Query vector: each distinct term once, weight 1, so Qdrant's IDF does the ranking.
pub fn query(text: &str) -> (Vec<u32>, Vec<f32>) {
    let ids: std::collections::BTreeSet<u32> = terms(text).map(|t| term_id(&t)).collect();
    let n = ids.len();
    (ids.into_iter().collect(), vec![1.0; n])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_split_identifiers_like_fts5() {
        let t: Vec<String> = tokens("Moved SEDN26-1910 to `token_stream.rs`; zażółć!").collect();
        assert_eq!(t, ["moved", "sedn26", "1910", "to", "token", "stream", "rs", "zażółć"]);
    }

    #[test]
    fn joined_tokens_are_also_terms_whole() {
        let t: Vec<String> = terms("see `token_stream`, SEDN26-1910 and crate::eval. Done.").collect();
        for whole in ["token_stream", "sedn26-1910", "crate::eval"] {
            assert!(t.contains(&whole.to_owned()), "{whole}: {t:?}");
        }
        assert_eq!(t.iter().filter(|x| *x == "done").count(), 1, "a plain word is not doubled");
        assert_eq!(query("token_stream").0.len(), 3, "token, stream, token_stream");
    }

    #[test]
    fn document_weights_saturate_and_are_sorted() {
        let (ids, values) = document("build build build parser");
        assert_eq!(ids.len(), 2);
        assert!(ids.windows(2).all(|w| w[0] < w[1]), "sorted, as Qdrant expects");
        let weight = |w: &str| values[ids.iter().position(|&i| i == term_id(w)).unwrap()];
        assert!(weight("build") > weight("parser"), "more occurrences weigh more");
        assert!(weight("build") < 3.0 * weight("parser"), "but saturate");
        assert!(weight("build") < K1 + 1.0);
    }

    #[test]
    fn longer_documents_weigh_a_term_less() {
        let weight = |text: &str| {
            let (ids, values) = document(text);
            values[ids.iter().position(|&i| i == term_id("parser")).unwrap()]
        };
        assert!(weight("parser") > weight(&format!("parser {}", "word ".repeat(500))));
    }

    #[test]
    fn query_is_distinct_terms_at_weight_one() {
        let (ids, values) = query("token token STREAM");
        assert_eq!(ids.len(), 2);
        assert_eq!(values, [1.0, 1.0]);
        assert_eq!(query("Token").0, query("token").0, "case-insensitive");
        assert!(query("  ").0.is_empty() && document("").0.is_empty());
    }
}
