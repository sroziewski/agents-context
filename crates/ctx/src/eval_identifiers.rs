//! Identifier eval: the queries the natural-language set is blind to. File names, Jira keys,
//! function names and error codes are rare exact tokens, the classic weakness of a dense model
//! (in the atlassian gateway's measurement dense bge-m3 alone reached 38% recall@20 on them).
//!
//! Cases are generated from the live index rather than written by hand: every identifier that
//! occurs in exactly one session of its project becomes two cases, the bare token and the token
//! inside a sentence (half Polish, half English). Besides the production variants (the shipped
//! search, BM25 leg included), an in-memory BM25 and RRF fusions with it are scored, to compare
//! leg combinations without re-indexing. The in-memory BM25 is an approximation of the shipped
//! leg (corpus average length, turn-only IDF, words only), so only `production` measures it.

use std::collections::{HashMap, HashSet};

use embed::Embedding;
use store::{SearchFilter, SearchMode, Store};
use uuid::Uuid;

use crate::eval::{self, Case, TurnRef, Variant, VariantReport};
use crate::lexical::{RRF_K, has_identifier, identifiers, rrf};
use crate::search::{Context, Query, Scope};

/// Turns each leg contributes to a fusion; deep enough that a turn ranked moderately by every
/// leg can still win.
pub const POOL: u64 = 50;
/// Sentence forms, alternated so the set is half Polish, half English.
const SENTENCES: &[(&str, &str)] = &[
    ("en", "what did we change around {}?"),
    ("pl", "co ustaliliśmy w sprawie {}?"),
    ("en", "why did {} come up and how was it resolved"),
    ("pl", "gdzie poprawialiśmy {} i dlaczego"),
];

#[derive(Debug, Clone, PartialEq)]
pub struct Turn {
    pub id: Uuid,
    pub project: String,
    pub session: String,
    /// 0-based, as stored (`turn_index`).
    pub turn: usize,
    pub text: String,
}

/// Every main-agent turn in the index; subagent turns are excluded as in a normal search.
pub async fn load_turns(store: &Store) -> anyhow::Result<Vec<Turn>> {
    let filter = SearchFilter::default().eq("kind", "turn").flag("subagent", false);
    let hits = store.scroll(&filter, u32::MAX).await?;
    let s = |h: &store::Hit, k: &str| h.payload.get(k).and_then(|v| v.as_str()).unwrap_or_default().to_owned();
    Ok(hits
        .iter()
        .map(|h| Turn {
            id: h.id,
            project: s(h, "project"),
            session: s(h, "session_id"),
            turn: h.payload.get("turn_index").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
            text: s(h, "text"),
        })
        .collect())
}

/// One identifier unique to one session of its project.
#[derive(Debug, Clone, PartialEq)]
pub struct Picked {
    pub identifier: String,
    pub project: String,
    pub session: String,
    /// 0-based turns of that session that contain it.
    pub turns: Vec<usize>,
}

/// Up to `n` identifiers that occur in exactly one session of their project, in a fixed
/// pseudo-random order (so reruns on the same index give the same set), plus how many token-level
/// candidates there were. Uniqueness is checked by substring, not token, so `parser.rs` is
/// rejected when another session mentions `src/parser.rs`.
pub fn pick(turns: &[Turn], n: usize) -> (Vec<Picked>, usize) {
    let mut sessions_of: HashMap<(&str, String), (String, HashSet<&str>)> = HashMap::new();
    for t in turns {
        for id in identifiers(&t.text) {
            let entry = sessions_of.entry((t.project.as_str(), id.to_ascii_lowercase())).or_insert_with(|| (id, HashSet::new()));
            entry.1.insert(t.session.as_str());
        }
    }
    let mut candidates: Vec<(&str, String, String)> =
        sessions_of.into_iter().filter(|(_, (_, s))| s.len() == 1).map(|((p, lower), (original, _))| (p, lower, original)).collect();
    let total = candidates.len();
    candidates.sort_by_key(|(p, lower, _)| (fnv1a(lower), lower.clone(), p.to_string()));

    let lowered: Vec<String> = turns.iter().map(|t| t.text.to_lowercase()).collect();
    let mut by_project: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, t) in turns.iter().enumerate() {
        by_project.entry(t.project.as_str()).or_default().push(i);
    }
    let mut out = Vec::new();
    for (project, lower, original) in candidates {
        if out.len() >= n {
            break;
        }
        let containing: Vec<usize> = by_project[project].iter().copied().filter(|&i| lowered[i].contains(&lower)).collect();
        let sessions: HashSet<&str> = containing.iter().map(|&i| turns[i].session.as_str()).collect();
        if sessions.len() != 1 {
            continue;
        }
        let mut hit_turns: Vec<usize> = containing.iter().map(|&i| turns[i].turn).collect();
        hit_turns.sort_unstable();
        hit_turns.dedup();
        out.push(Picked { identifier: original, project: project.to_owned(), session: turns[containing[0]].session.clone(), turns: hit_turns });
    }
    (out, total)
}

/// The bare and the sentence case for each picked identifier. Bare cases carry lang `id`.
pub fn cases(picked: &[Picked]) -> (Vec<Case>, Vec<Case>) {
    let case = |id: String, query: String, lang: &str, p: &Picked| Case {
        id,
        query,
        target: eval::Target::Sessions,
        project: Some(p.project.clone()),
        group: None,
        team: None,
        channel: None,
        author: None,
        expect: vec![p.session.clone()],
        latest: false,
        expect_turns: p.turns.iter().map(|&t| TurnRef { session: p.session.clone(), turn: t + 1 }).collect(),
        lang: lang.to_owned(),
    };
    let bare = picked.iter().enumerate().map(|(i, p)| case(format!("bare:{i}"), p.identifier.clone(), "id", p)).collect();
    let sentence = picked
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let (lang, template) = SENTENCES[i % SENTENCES.len()];
            case(format!("sentence:{i}"), template.replace("{}", &p.identifier), lang, p)
        })
        .collect();
    (bare, sentence)
}

fn fnv1a(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
}

/// In-memory Okapi BM25 over the turns, OR semantics: any query word scores. Lets the eval
/// compare leg combinations without touching the index.
pub struct Bm25 {
    docs: Vec<(Uuid, String, HashMap<String, u32>, usize)>,
    postings: HashMap<String, Vec<usize>>,
    avg_len: f64,
}

impl Bm25 {
    const K1: f64 = 1.2;
    const B: f64 = 0.75;

    pub fn new(turns: &[Turn]) -> Self {
        let mut docs = Vec::with_capacity(turns.len());
        let mut postings: HashMap<String, Vec<usize>> = HashMap::new();
        let mut total = 0usize;
        for (i, t) in turns.iter().enumerate() {
            let mut tf: HashMap<String, u32> = HashMap::new();
            let mut len = 0;
            for w in store::lexical::tokens(&t.text) {
                *tf.entry(w).or_default() += 1;
                len += 1;
            }
            for w in tf.keys() {
                postings.entry(w.clone()).or_default().push(i);
            }
            total += len;
            docs.push((t.id, t.project.clone(), tf, len));
        }
        let avg_len = if docs.is_empty() { 1.0 } else { (total as f64 / docs.len() as f64).max(1.0) };
        Self { docs, postings, avg_len }
    }

    /// Best `limit` turns of `project` (every project when `None`), best first; ties keep
    /// index order.
    pub fn search(&self, query: &str, project: Option<&str>, limit: usize) -> Vec<Uuid> {
        let n = self.docs.len() as f64;
        let terms: HashSet<String> = store::lexical::tokens(query).collect();
        let mut scores: HashMap<usize, f64> = HashMap::new();
        for term in &terms {
            let Some(list) = self.postings.get(term) else { continue };
            let df = list.len() as f64;
            let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
            for &d in list {
                let (_, p, tf, len) = &self.docs[d];
                if project.is_some_and(|want| want != p) {
                    continue;
                }
                let f = f64::from(tf[term]);
                let norm = f + Self::K1 * (1.0 - Self::B + Self::B * *len as f64 / self.avg_len);
                *scores.entry(d).or_default() += idf * f * (Self::K1 + 1.0) / norm;
            }
        }
        let mut ranked: Vec<(usize, f64)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        ranked.into_iter().take(limit).map(|(d, _)| self.docs[d].0).collect()
    }
}

/// Which legs a fused variant combines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Legs {
    pub name: &'static str,
    pub dense: bool,
    pub sparse: bool,
    pub bm25: bool,
    /// Use these legs only for a query that contains an identifier, dense alone otherwise. No
    /// static fusion serves both query shapes: lexical legs win on identifiers and lose on
    /// natural questions (Polish ones most, BM25 has no stemming).
    pub routed: bool,
}

pub fn fused_variants() -> Vec<Legs> {
    let l = |name, dense, sparse, bm25, routed| Legs { name, dense, sparse, bm25, routed };
    vec![
        l("bm25", false, false, true, false),
        l("dense+bm25", true, false, true, false),
        l("dense+sparse (k60)", true, true, false, false),
        l("dense+sparse+bm25", true, true, true, false),
        l("routed dense+bm25", true, false, true, true),
        l("routed dense+sparse+bm25", true, true, true, true),
    ]
}

pub fn scope(c: &Case) -> Scope {
    // The generator stores the project key itself; re-deriving it from the filesystem could
    // land on a different git root.
    c.project.clone().map_or(Scope::All, Scope::Project)
}

/// The production variants (through the real search path) and the fused ones, over `cases`.
pub async fn run(
    ctx: &Context,
    turns: &[Turn],
    bm25: &Bm25,
    cases: &[Case],
    vectors: &HashMap<String, Embedding>,
    variants: &[Variant],
    fused: &[Legs],
) -> anyhow::Result<Vec<VariantReport>> {
    let mut reports = Vec::new();
    for v in variants {
        let mut results = Vec::new();
        for c in cases {
            let mut q = Query::new(c.query.clone(), scope(c), eval::DEPTH);
            q.mode = v.mode;
            q.recency = v.recency;
            q.lexical = v.lexical;
            let r = ctx.search_embedded(&vectors[&c.id], &q).await?;
            results.push(eval::judge_turns(c, &eval::session_order(&r, v.summaries), &eval::turn_order(&r)));
        }
        reports.push(eval::summarize(v.name, cases, results));
    }
    reports.extend(run_fused(ctx, turns, bm25, cases, vectors, fused, &scope).await?);
    Ok(reports)
}

/// The fused variants alone, turns only (no summaries), judged like any variant. `scope_of`
/// says where each case searches: the identifier cases carry a project key, the hand-written
/// set a project directory.
pub async fn run_fused(
    ctx: &Context,
    turns: &[Turn],
    bm25: &Bm25,
    cases: &[Case],
    vectors: &HashMap<String, Embedding>,
    fused: &[Legs],
    scope_of: &dyn Fn(&Case) -> Scope,
) -> anyhow::Result<Vec<VariantReport>> {
    let mut reports = Vec::new();
    let located: HashMap<Uuid, (&str, usize)> = turns.iter().map(|t| (t.id, (t.session.as_str(), t.turn + 1))).collect();
    let mut legs_of: HashMap<&str, [Vec<Uuid>; 3]> = HashMap::new();
    for c in cases {
        let leg = |mode| {
            let mut q = Query::new(c.query.clone(), scope_of(c), POOL);
            q.mode = mode;
            q
        };
        let dense = ctx.search_embedded(&vectors[&c.id], &leg(SearchMode::Dense)).await?.turns.iter().map(|h| h.id).collect();
        let sparse = ctx.search_embedded(&vectors[&c.id], &leg(SearchMode::Sparse)).await?.turns.iter().map(|h| h.id).collect();
        let project = match scope_of(c) {
            Scope::Project(p) => Some(p),
            Scope::All => None,
        };
        let lexical = bm25.search(&c.query, project.as_deref(), POOL as usize);
        legs_of.insert(c.id.as_str(), [dense, sparse, lexical]);
    }
    for f in fused {
        let mut results = Vec::new();
        for c in cases {
            let [dense, sparse, lexical] = &legs_of[c.id.as_str()];
            let lexical_on = !f.routed || has_identifier(&c.query);
            let chosen: Vec<Vec<Uuid>> = [(f.dense, dense), (f.sparse && lexical_on, sparse), (f.bm25 && lexical_on, lexical)]
                .into_iter()
                .filter(|(on, _)| *on)
                .map(|(_, l)| l.clone())
                .collect();
            let shown: Vec<(String, usize)> = rrf(&chosen, RRF_K)
                .into_iter()
                .filter_map(|(id, _)| located.get(&id).map(|&(s, t)| (s.to_owned(), t)))
                .take(eval::DEPTH as usize)
                .collect();
            let mut seen = HashSet::new();
            let sessions: Vec<String> = shown.iter().filter(|(s, _)| seen.insert(s.clone())).map(|(s, _)| s.clone()).collect();
            results.push(eval::judge_turns(c, &sessions, &shown));
        }
        reports.push(eval::summarize(f.name, cases, results));
    }
    Ok(reports)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(n: u128, project: &str, session: &str, idx: usize, text: &str) -> Turn {
        Turn { id: Uuid::from_u128(n), project: project.into(), session: session.into(), turn: idx, text: text.into() }
    }

    #[test]
    fn pick_keeps_identifiers_unique_to_one_session_of_their_project() {
        let turns = [
            turn(1, "p", "s1", 0, "renamed token_stream and fixed lexer.rs"),
            turn(2, "p", "s1", 3, "token_stream again"),
            turn(3, "p", "s2", 0, "touched src/lexer.rs too"),
            turn(4, "q", "s3", 0, "token_stream in another project does not count"),
            turn(5, "p", "s2", 1, "the old_token_stream_v2 helper"),
        ];
        let (picked, candidates) = pick(&turns, 10);
        let p: Vec<&str> = picked.iter().filter(|x| x.project == "p").map(|x| x.identifier.as_str()).collect();
        assert!(candidates >= 3, "{candidates}");
        // lexer.rs is in two sessions by token; token_stream is a substring of s2's helper name.
        assert!(!p.contains(&"lexer.rs") && !p.contains(&"token_stream"), "{p:?}");
        assert!(p.contains(&"old_token_stream_v2"), "{p:?}");
        let s3 = picked.iter().find(|x| x.project == "q").expect("unique within its own project");
        assert_eq!((s3.identifier.as_str(), s3.session.as_str(), s3.turns.as_slice()), ("token_stream", "s3", &[0][..]));
        assert_eq!(pick(&turns, 10), pick(&turns, 10), "deterministic");
        assert_eq!(pick(&turns, 1).0.len(), 1, "capped at n");
    }

    #[test]
    fn pick_records_every_turn_that_mentions_it() {
        let turns = [turn(1, "p", "s1", 2, "see DEV-77"), turn(2, "p", "s1", 0, "dev-77 first"), turn(3, "p", "s1", 1, "nothing")];
        let (picked, _) = pick(&turns, 5);
        assert_eq!(picked[0].turns, [0, 2]);
    }

    #[test]
    fn cases_come_bare_and_in_a_sentence_half_polish() {
        let picked: Vec<Picked> =
            (0..4).map(|i| Picked { identifier: format!("ID_{i}"), project: "p".into(), session: format!("s{i}"), turns: vec![1] }).collect();
        let (bare, sentence) = cases(&picked);
        assert_eq!(bare[0].query, "ID_0");
        assert_eq!(bare[0].lang, "id");
        assert_eq!(bare[0].expect_turns, [TurnRef { session: "s0".into(), turn: 2 }], "turns are shown 1-based");
        assert!(sentence.iter().zip(&picked).all(|(c, p)| c.query.contains(&p.identifier) && c.query != p.identifier));
        assert_eq!(sentence.iter().filter(|c| c.lang == "pl").count(), 2);
        assert!(sentence[1].query.starts_with("co ustaliliśmy"), "{}", sentence[1].query);
    }

    #[test]
    fn bm25_prefers_the_rare_word_and_stays_in_its_project() {
        let turns = [
            turn(1, "p", "s1", 0, "we fixed the build and the build passed"),
            turn(2, "p", "s2", 0, "the build broke in token_stream"),
            turn(3, "q", "s3", 0, "token_stream token_stream"),
        ];
        let bm = Bm25::new(&turns);
        assert_eq!(bm.search("token_stream build", Some("p"), 10), [Uuid::from_u128(2), Uuid::from_u128(1)]);
        assert_eq!(bm.search("TOKEN_STREAM", Some("q"), 10), [Uuid::from_u128(3)], "case-insensitive");
        assert!(bm.search("absent", Some("p"), 10).is_empty());
        assert_eq!(bm.search("token_stream", None, 10), [Uuid::from_u128(3), Uuid::from_u128(2)], "every project");
    }
}
