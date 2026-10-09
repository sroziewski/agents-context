//! Routing between dense search and the lexical legs (DEV-61). Identifier queries (file names,
//! Jira keys, function names) are where dense search is weakest and BM25 strongest; on natural
//! questions every static fusion with a lexical leg lost (`ctx eval`: R@1 0.85 → 0.75 for
//! dense+BM25, Polish worst, since BM25 does not stem). So the lexical legs run only when the
//! query contains an identifier, and natural questions stay dense.

use std::collections::{HashMap, HashSet};

use uuid::Uuid;

/// The usual RRF constant, the one the evals measured. Qdrant's built-in fusion uses a smaller
/// one, which is why the legs are fused here rather than in Qdrant.
pub const RRF_K: f64 = 60.0;

/// Which legs join dense for an identifier query (`CTX_LEXICAL`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Lexical {
    /// Search exactly as before DEV-61/62: dense, summaries first, for every query.
    Off,
    #[default]
    Bm25,
    /// BM25 plus bge-m3's learned sparse weights. Measured even with `Bm25` (within noise), so
    /// not the default; worth re-checking with `ctx eval --target identifiers` as projects grow.
    SparseBm25,
}

impl Lexical {
    /// `off`, `bm25` (default), `sparse+bm25`; anything else (a typo) means the default.
    pub fn parse(v: Option<&str>) -> Self {
        match v.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
            Some("off" | "none" | "dense") => Self::Off,
            Some("sparse+bm25" | "bm25+sparse") => Self::SparseBm25,
            _ => Self::Bm25,
        }
    }
}

/// Characters that end a token besides whitespace. Path separators split a path into its
/// segments: people recall `eval.rs`, not `crates/ctx/src/eval.rs`.
const SEPARATORS: &[char] = &[
    '`', '\'', '"', '(', ')', '[', ']', '{', '}', '<', '>', ',', ';', '|', '=', '*', '/', '\\', '#', '!', '?', '&', '$', '@', '%', '^', '~', '+',
];
/// Sentence punctuation and flag dashes stripped from both ends; inside a token they belong to
/// it (`bge-m3`, `crate::eval`, `odpis.pdf`).
const EDGES: &[char] = &['.', ':', '-'];

/// Product names with an inner capital. Spelled like `TechFileEntity` but ordinary words in a
/// question ("why did GitHub actions fail"), where routing would cost a natural query its
/// summaries and add a leg that hurts natural questions.
const PRODUCT_NAMES: &[&str] = &[
    "github", "gitlab", "bitbucket", "directml", "powershell", "javascript", "typescript", "coffeescript", "postgresql", "mysql",
    "sqlite", "mongodb", "openai", "chatgpt", "youtube", "linkedin", "iphone", "ipad", "macos", "ios", "devops", "fastapi",
    "pytorch", "langchain", "sharepoint", "onedrive", "wordpress", "intellij", "webstorm", "pycharm", "opencode", "testcontainers",
    "keycloak", "graphql", "nestjs", "nextjs", "vscode", "dockerhub", "openshift", "rabbitmq", "elasticsearch", "opensearch",
];

/// Domain endings: `github.com` from a pasted URL is not a file name.
const DOMAIN_ENDINGS: &[&str] = &["com", "org", "net", "io", "pl", "eu", "dev", "app", "gov", "edu", "uk", "de", "info"];

/// Whether a token looks like something recalled by its exact spelling.
pub fn is_identifier(t: &str) -> bool {
    if PRODUCT_NAMES.contains(&t.to_ascii_lowercase().as_str()) {
        return false;
    }
    if !t.is_ascii() || !(4..=64).contains(&t.len()) || t.starts_with(|c: char| c.is_ascii_digit()) {
        return false;
    }
    let letters = t.bytes().filter(u8::is_ascii_alphabetic).count();
    let digits = t.bytes().filter(u8::is_ascii_digit).count();
    if letters < 2 {
        return false;
    }
    // Commit hashes and UUIDs are unique, but nobody finds a session by one. Short hex-looking
    // keys (`ED-850`) must survive, hence the length bounds.
    let hexish = t.bytes().all(|b| b.is_ascii_hexdigit() || b == b'-');
    if hexish && digits > 0 && (t.len() >= 32 || (!t.contains('-') && t.len() >= 7)) {
        return false;
    }
    let camel = t.as_bytes().windows(2).any(|w| w[0].is_ascii_lowercase() && w[1].is_ascii_uppercase());
    let file = t.rsplit_once('.').is_some_and(|(stem, ext)| {
        !stem.is_empty()
            && (1..=5).contains(&ext.len())
            && ext.bytes().all(|b| b.is_ascii_alphanumeric())
            && ext.bytes().any(|b| b.is_ascii_alphabetic())
            && !DOMAIN_ENDINGS.contains(&ext.to_ascii_lowercase().as_str())
    });
    digits > 0 || t.contains('_') || t.contains("::") || camel || file
}

/// Distinct identifiers in a text, first spelling kept, compared case-insensitively.
pub fn identifiers(text: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    text.split(|c: char| c.is_whitespace() || SEPARATORS.contains(&c))
        .map(|t| t.trim_matches(EDGES))
        .filter(|t| is_identifier(t))
        .filter(|t| seen.insert(t.to_ascii_lowercase()))
        .map(str::to_owned)
        .collect()
}

/// Whether a query names something by its exact spelling: the routing signal for the lexical
/// legs (DEV-61) and for leaving summaries out (DEV-62).
pub fn has_identifier(query: &str) -> bool {
    !identifiers(query).is_empty()
}

/// Reciprocal rank fusion of ranked id lists, best first, with the fused score; ties keep
/// first appearance.
pub fn rrf(legs: &[Vec<Uuid>], k: f64) -> Vec<(Uuid, f64)> {
    let mut order: Vec<Uuid> = Vec::new();
    let mut score: HashMap<Uuid, f64> = HashMap::new();
    for leg in legs {
        for (rank, id) in leg.iter().enumerate() {
            let s = score.entry(*id).or_insert_with(|| {
                order.push(*id);
                0.0
            });
            *s += 1.0 / (k + rank as f64 + 1.0);
        }
    }
    order.sort_by(|a, b| score[b].total_cmp(&score[a]));
    order.into_iter().map(|id| (id, score[&id])).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifiers_are_recalled_by_spelling() {
        for yes in ["DEV-41", "SEDN26-1910", "ED-850", "token_stream", "parser.rs", "crate::eval", "mObywatel", "bge-m3", "RDE_2772", "arm64_v8"] {
            assert!(is_identifier(yes), "{yes}");
        }
        for no in [
            "parser", "DEEPR", "97f4735", "0a1b2c3d-4e5f-6789-abcd-ef0123456789", "2026-10-09", "500ms", "v2", "e.g", "zażółć1", "x86_64",
            "GitHub", "DirectML", "PowerShell", "TypeScript", "OpenAI", "iPhone", "github.com", "ASP.NET", "confluence.example.org",
        ] {
            assert!(!is_identifier(no), "{no}");
        }
    }

    #[test]
    fn identifiers_split_paths_and_strip_sentence_punctuation() {
        let ids = identifiers("Fixed `crates/ctx/src/eval.rs` for DEV-41. See (token_stream), then DEV-41 again and --rerank_pool.");
        assert_eq!(ids, ["eval.rs", "DEV-41", "token_stream", "rerank_pool"]);
    }

    #[test]
    fn routing_fires_on_identifiers_only() {
        assert!(has_identifier("what does boundObjectId mean in the login payload"));
        assert!(has_identifier("co z SEDN26-1910?"));
        assert!(!has_identifier("jak działa wdrożenie na produkcję"));
        assert!(!has_identifier("which embedding model did we choose"));
        assert!(!has_identifier("how did we fix the DirectML hang last time"));
        assert!(!has_identifier("why did GitHub actions fail on https://github.com/x"));
        assert!(has_identifier("why did TechFileEntity fail"), "a class name still routes");
    }

    #[test]
    fn rrf_rewards_agreement_between_legs() {
        let (a, b, c) = (Uuid::from_u128(1), Uuid::from_u128(2), Uuid::from_u128(3));
        let ids = |r: Vec<(Uuid, f64)>| r.into_iter().map(|(id, _)| id).collect::<Vec<_>>();
        assert_eq!(ids(rrf(&[vec![a, b], vec![b, c]], 60.0)), [b, a, c]);
        assert_eq!(ids(rrf(&[vec![a, b]], 60.0)), [a, b], "one leg keeps its order");
        assert!(rrf(&[], 60.0).is_empty());
        let fused = rrf(&[vec![a], vec![a]], 60.0);
        assert!((fused[0].1 - 2.0 / 61.0).abs() < 1e-12);
    }

    #[test]
    fn lexical_setting_parses_with_bm25_as_default() {
        assert_eq!(Lexical::parse(None), Lexical::Bm25);
        assert_eq!(Lexical::parse(Some(" OFF ")), Lexical::Off);
        assert_eq!(Lexical::parse(Some("sparse+bm25")), Lexical::SparseBm25);
        assert_eq!(Lexical::parse(Some("nonsense")), Lexical::Bm25);
        assert_eq!(Lexical::default(), Lexical::Bm25, "the same default as the setting");
    }
}
