//! Retrieval eval: real questions with the sessions that answer them, run against the live
//! index in every search variant, so ranking changes (recency, summaries, auto-inject
//! thresholds) are measured instead of guessed.
//!
//! The query set is JSON lines, one [`Case`] per line. It names real projects and topics, so
//! it lives in the user's config directory, not in the repository; the repository tests the
//! harness on a synthetic set.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use embed::Embedding;
use serde::{Deserialize, Serialize};
use store::SearchMode;

use crate::inject::Injector;
use crate::rank::Recency;
use crate::search::{Context, Query, Results, Scope};

/// Results fetched per search; MRR is measured to this depth.
const DEPTH: u64 = 10;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Case {
    pub id: String,
    pub query: String,
    /// Project directory the question is asked from; absent means across all projects.
    #[serde(default)]
    pub project: Option<String>,
    /// Sessions that answer it, the preferred (newest) one first. Empty means nothing in the
    /// index answers it: a negative, where auto-inject must stay silent.
    #[serde(default)]
    pub expect: Vec<String>,
    /// A later session superseded the others: `expect[0]` must outrank the rest.
    #[serde(default)]
    pub latest: bool,
    #[serde(default)]
    pub lang: String,
}

impl Case {
    pub fn scope(&self) -> Scope {
        match &self.project {
            Some(dir) => Scope::Project(sources::project::key(dir)),
            None => Scope::All,
        }
    }
}

/// Parses the query set: JSON lines, blank lines and `#` comments skipped, ids unique.
pub fn parse_cases(text: &str) -> anyhow::Result<Vec<Case>> {
    let mut cases: Vec<Case> = Vec::new();
    let mut ids = HashSet::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let case: Case = serde_json::from_str(line).map_err(|e| anyhow::anyhow!("line {}: {e}", n + 1))?;
        anyhow::ensure!(!case.query.trim().is_empty(), "line {}: empty query", n + 1);
        anyhow::ensure!(ids.insert(case.id.clone()), "line {}: duplicate id {:?}", n + 1, case.id);
        anyhow::ensure!(!case.latest || case.expect.len() >= 2, "line {}: `latest` needs at least two expected sessions", n + 1);
        cases.push(case);
    }
    Ok(cases)
}

pub fn load_cases(path: &Path) -> anyhow::Result<Vec<Case>> {
    let text = std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))?;
    parse_cases(&text)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Variant {
    pub name: &'static str,
    pub mode: SearchMode,
    /// Session summaries are listed before the turns, as `search_sessions` shows them.
    pub summaries: bool,
    pub recency: Recency,
}

/// What `search_sessions` serves with the current configuration; baselines are kept for it.
pub const PRODUCTION: &str = "production";

/// The alternatives worth comparing, then the configured one.
pub fn variants(mode: SearchMode, recency: Recency) -> Vec<Variant> {
    let v = |name, mode, summaries, recency| Variant { name, mode, summaries, recency };
    vec![
        v("dense", SearchMode::Dense, false, Recency::OFF),
        v("sparse", SearchMode::Sparse, false, Recency::OFF),
        v("hybrid", SearchMode::Hybrid, false, Recency::OFF),
        v("hybrid+summaries", SearchMode::Hybrid, true, Recency::OFF),
        v("dense+summaries", SearchMode::Dense, true, Recency::OFF),
        v(PRODUCTION, mode, true, recency),
    ]
}

/// Distinct sessions in the order an agent reads them: summaries, then turns.
pub fn session_order(r: &Results, summaries: bool) -> Vec<String> {
    let mut seen = HashSet::new();
    let hits = if summaries { r.summaries.iter().chain(&r.turns).collect::<Vec<_>>() } else { r.turns.iter().collect() };
    hits.into_iter()
        .filter_map(|h| h.payload.get("session_id").and_then(|v| v.as_str()))
        .filter(|id| seen.insert(id.to_string()))
        .map(str::to_owned)
        .collect()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseResult {
    pub id: String,
    /// 1-based rank of the first expected session, if it was found within the depth.
    pub rank: Option<usize>,
    /// For `latest` cases: whether the preferred session came before the superseded ones.
    pub latest_ok: Option<bool>,
}

pub fn judge(case: &Case, order: &[String]) -> CaseResult {
    let rank = order.iter().position(|s| case.expect.contains(s)).map(|i| i + 1);
    let latest_ok = case.latest.then(|| rank.is_some_and(|r| order[r - 1] == case.expect[0]));
    CaseResult { id: case.id.clone(), rank, latest_ok }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VariantReport {
    pub name: String,
    pub cases: Vec<CaseResult>,
    pub recall_at_1: f64,
    pub recall_at_5: f64,
    pub mrr: f64,
    /// (preferred first, latest cases found at all).
    pub latest: (usize, usize),
    /// Language → (found in top 5, cases).
    pub by_lang: BTreeMap<String, (usize, usize)>,
}

pub fn summarize(name: &str, cases: &[Case], results: Vec<CaseResult>) -> VariantReport {
    let n = results.len().max(1) as f64;
    let within = |k: usize| results.iter().filter(|r| r.rank.is_some_and(|x| x <= k)).count() as f64 / n;
    let mrr = results.iter().map(|r| r.rank.map_or(0.0, |x| 1.0 / x as f64)).sum::<f64>() / n;
    let latest = results.iter().filter_map(|r| r.latest_ok).fold((0, 0), |(ok, all), x| (ok + usize::from(x), all + 1));
    let lang: HashMap<&str, &str> = cases.iter().map(|c| (c.id.as_str(), c.lang.as_str())).collect();
    let mut by_lang: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for r in &results {
        let entry = by_lang.entry(lang.get(r.id.as_str()).copied().unwrap_or_default().to_owned()).or_default();
        entry.0 += usize::from(r.rank.is_some_and(|x| x <= 5));
        entry.1 += 1;
    }
    VariantReport { name: name.to_owned(), recall_at_1: within(1), recall_at_5: within(5), mrr, latest, by_lang, cases: results }
}

/// Embeds every query once; the variants reuse the vectors.
pub async fn embed_queries(ctx: &Context, cases: &[Case]) -> anyhow::Result<HashMap<String, Embedding>> {
    let embedder = ctx.embedder.clone();
    let texts: Vec<(String, String)> = cases.iter().map(|c| (c.id.clone(), c.query.clone())).collect();
    tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
        let mut e = embedder.lock().unwrap_or_else(|p| p.into_inner());
        let refs: Vec<&str> = texts.iter().map(|(_, q)| q.as_str()).collect();
        let vectors = e.embed(&refs)?;
        Ok(texts.into_iter().map(|(id, _)| id).zip(vectors).collect())
    })
    .await?
}

/// Retrieval metrics per variant over the positive cases.
pub async fn run_retrieval(ctx: &Context, cases: &[Case], variants: &[Variant], vectors: &HashMap<String, Embedding>) -> anyhow::Result<Vec<VariantReport>> {
    let positives: Vec<&Case> = cases.iter().filter(|c| !c.expect.is_empty()).collect();
    let mut reports = Vec::new();
    for v in variants {
        let mut results = Vec::new();
        for c in &positives {
            let mut q = Query::new(c.query.clone(), c.scope(), DEPTH);
            q.mode = v.mode;
            q.recency = v.recency;
            let r = ctx.search_embedded(&vectors[&c.id], &q).await?;
            results.push(judge(c, &session_order(&r, v.summaries)));
        }
        reports.push(summarize(v.name, cases, results));
    }
    Ok(reports)
}

#[derive(Debug, Clone, PartialEq)]
pub struct InjectPoint {
    pub min_score: f32,
    /// Positives where an expected session was injected.
    pub hits: usize,
    pub positives: usize,
    /// Negatives where anything was injected.
    pub false_injects: usize,
    pub negatives: usize,
    /// Share of injected sessions (on positives) that were expected ones.
    pub precision: f64,
}

/// Auto-inject at each threshold, over the cases asked from a project (injection is always
/// project-scoped).
pub async fn run_inject(injector: &Injector, cases: &[Case], vectors: &HashMap<String, Embedding>, thresholds: &[f32]) -> anyhow::Result<Vec<InjectPoint>> {
    let now = jiff::Timestamp::now().as_millisecond();
    let scoped: Vec<(&Case, String)> = cases.iter().filter_map(|c| c.project.as_ref().map(|p| (c, sources::project::key(p)))).collect();
    let mut points = Vec::new();
    for &t in thresholds {
        let mut p = InjectPoint { min_score: t, hits: 0, positives: 0, false_injects: 0, negatives: 0, precision: 0.0 };
        let (mut expected, mut injected) = (0usize, 0usize);
        for (c, project) in &scoped {
            let sel = injector.select(&vectors[&c.id], project, "", t, false, now, &|_| false).await?;
            let ids: Vec<String> = sel.sessions.iter().filter_map(|h| h.payload.get("session_id")?.as_str().map(str::to_owned)).collect();
            if c.expect.is_empty() {
                p.negatives += 1;
                p.false_injects += usize::from(!ids.is_empty());
            } else {
                p.positives += 1;
                p.hits += usize::from(ids.iter().any(|s| c.expect.contains(s)));
                injected += ids.len();
                expected += ids.iter().filter(|s| c.expect.contains(s)).count();
            }
        }
        p.precision = if injected == 0 { 0.0 } else { expected as f64 / injected as f64 };
        points.push(p);
    }
    Ok(points)
}

pub fn format_retrieval(reports: &[VariantReport]) -> String {
    let langs: Vec<String> = reports.first().map(|r| r.by_lang.keys().cloned().collect()).unwrap_or_default();
    let mut out = format!("{:<26} {:>5} {:>5} {:>7} {:>7}", "variant", "R@1", "R@5", "MRR", "latest");
    for l in &langs {
        out.push_str(&format!(" {:>8}", format!("{} R@5", if l.is_empty() { "?" } else { l })));
    }
    for r in reports {
        out.push_str(&format!(
            "\n{:<26} {:>5.2} {:>5.2} {:>7.3} {:>7}",
            r.name,
            r.recall_at_1,
            r.recall_at_5,
            r.mrr,
            if r.latest.1 == 0 { "-".to_owned() } else { format!("{}/{}", r.latest.0, r.latest.1) }
        ));
        for l in &langs {
            let (hit, all) = r.by_lang.get(l).copied().unwrap_or_default();
            out.push_str(&format!(" {:>8}", format!("{hit}/{all}")));
        }
    }
    out
}

pub fn format_misses(report: &VariantReport, cases: &[Case]) -> String {
    let query: HashMap<&str, &str> = cases.iter().map(|c| (c.id.as_str(), c.query.as_str())).collect();
    let misses: Vec<String> = report
        .cases
        .iter()
        .filter(|r| r.rank.is_none_or(|x| x > 5) || r.latest_ok == Some(false))
        .map(|r| {
            let what = match (r.rank, r.latest_ok) {
                (Some(x), Some(false)) if x <= 5 => "superseded session ranked first".to_owned(),
                (Some(x), _) => format!("rank {x}"),
                (None, _) => "not found".to_owned(),
            };
            format!("  {:<14} {what:<32} {}", r.id, query.get(r.id.as_str()).copied().unwrap_or_default())
        })
        .collect();
    if misses.is_empty() { format!("{}: no misses", report.name) } else { format!("{} misses:\n{}", report.name, misses.join("\n")) }
}

pub fn format_inject(points: &[InjectPoint]) -> String {
    let mut out = format!("{:>9} {:>9} {:>10} {:>13}", "min_score", "hit rate", "precision", "false inject");
    for p in points {
        out.push_str(&format!(
            "\n{:>9.2} {:>9} {:>10.2} {:>13}",
            p.min_score,
            format!("{}/{}", p.hits, p.positives),
            p.precision,
            format!("{}/{}", p.false_injects, p.negatives)
        ));
    }
    out
}

/// The production variant's per-case ranks, kept to detect regressions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Baseline {
    pub variant: String,
    pub recall_at_5: f64,
    pub mrr: f64,
    pub ranks: BTreeMap<String, Option<usize>>,
}

impl Baseline {
    pub fn from_report(r: &VariantReport) -> Self {
        Self { variant: r.name.clone(), recall_at_5: r.recall_at_5, mrr: r.mrr, ranks: r.cases.iter().map(|c| (c.id.clone(), c.rank)).collect() }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Comparison {
    /// Cases in the top 5 at baseline that fell out.
    pub lost: Vec<String>,
    /// Cases that entered the top 5.
    pub gained: Vec<String>,
    /// Cases only in the current set; they do not count either way until a new baseline.
    pub new: Vec<String>,
}

impl Comparison {
    /// A regression is a net loss of top-5 cases among the cases both runs share; a swap of
    /// one case for another is reported but tolerated, since both are real queries. Comparing
    /// aggregate recall instead would flag every newly added hard case, and hide losses when
    /// cases are removed.
    pub fn is_regression(&self) -> bool {
        self.lost.len() > self.gained.len()
    }
}

pub fn compare(baseline: &Baseline, now: &VariantReport) -> Comparison {
    let top5 = |r: Option<usize>| r.is_some_and(|x| x <= 5);
    let mut c = Comparison::default();
    for case in &now.cases {
        match baseline.ranks.get(&case.id) {
            Some(before) if top5(*before) && !top5(case.rank) => c.lost.push(case.id.clone()),
            Some(before) if !top5(*before) && top5(case.rank) => c.gained.push(case.id.clone()),
            Some(_) => {}
            None => c.new.push(case.id.clone()),
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use store::Hit;

    fn case(id: &str, expect: &[&str], latest: bool, lang: &str) -> Case {
        Case { id: id.into(), query: "q".into(), project: None, expect: expect.iter().map(|s| s.to_string()).collect(), latest, lang: lang.into() }
    }

    fn hit(session: &str) -> Hit {
        Hit { id: uuid::Uuid::nil(), score: 1.0, payload: json!({ "session_id": session }).as_object().cloned().unwrap() }
    }

    #[test]
    fn parse_skips_comments_and_rejects_bad_lines() {
        let text = "# real queries\n\n{\"id\": \"a\", \"query\": \"x\", \"expect\": [\"s1\"]}\n{\"id\": \"n\", \"query\": \"y\"}\n";
        let cases = parse_cases(text).unwrap();
        assert_eq!(cases.len(), 2);
        assert!(cases[1].expect.is_empty(), "no expect is a negative");
        let dup = "{\"id\": \"a\", \"query\": \"x\"}\n{\"id\": \"a\", \"query\": \"y\"}";
        assert!(parse_cases(dup).unwrap_err().to_string().contains("line 2: duplicate id"));
        assert!(parse_cases("{\"id\": \"a\", \"query\": \" \"}").is_err());
        assert!(parse_cases("{\"id\": \"a\", \"query\": \"x\", \"expect\": [\"s\"], \"latest\": true}").is_err());
        assert!(parse_cases("not json").unwrap_err().to_string().starts_with("line 1"));
    }

    #[test]
    fn session_order_dedups_and_puts_summaries_first() {
        let r = Results { summaries: vec![hit("s3")], turns: vec![hit("s1"), hit("s1"), hit("s2"), hit("s3")] };
        assert_eq!(session_order(&r, true), ["s3", "s1", "s2"]);
        assert_eq!(session_order(&r, false), ["s1", "s2", "s3"]);
    }

    #[test]
    fn judge_ranks_by_session_and_checks_the_latest() {
        let c = case("a", &["new", "old"], true, "pl");
        let order = |ids: &[&str]| ids.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(judge(&c, &order(&["x", "new", "old"])), CaseResult { id: "a".into(), rank: Some(2), latest_ok: Some(true) });
        assert_eq!(judge(&c, &order(&["old", "new"])).latest_ok, Some(false));
        assert_eq!(judge(&c, &order(&["x"])), CaseResult { id: "a".into(), rank: None, latest_ok: Some(false) });
        assert_eq!(judge(&case("b", &["s"], false, ""), &order(&["s"])).latest_ok, None);
    }

    #[test]
    fn summarize_computes_recall_mrr_and_languages() {
        let cases = [case("a", &["s"], false, "pl"), case("b", &["s"], false, "en"), case("c", &["s"], false, "en")];
        let results = vec![
            CaseResult { id: "a".into(), rank: Some(1), latest_ok: None },
            CaseResult { id: "b".into(), rank: Some(4), latest_ok: Some(true) },
            CaseResult { id: "c".into(), rank: None, latest_ok: Some(false) },
        ];
        let r = summarize("v", &cases, results);
        assert!((r.recall_at_1 - 1.0 / 3.0).abs() < 1e-9);
        assert!((r.recall_at_5 - 2.0 / 3.0).abs() < 1e-9);
        assert!((r.mrr - (1.0 + 0.25) / 3.0).abs() < 1e-9);
        assert_eq!(r.latest, (1, 2));
        assert_eq!(r.by_lang["en"], (1, 2));
        assert_eq!(r.by_lang["pl"], (1, 1));
        let table = format_retrieval(std::slice::from_ref(&r));
        assert!(table.contains("en R@5") && table.contains("1/2"), "{table}");
        assert!(format_misses(&r, &cases).contains("not found"));
    }

    #[test]
    fn a_swap_is_reported_but_only_a_net_loss_is_a_regression() {
        let cases = [case("a", &["s"], false, ""), case("b", &["s"], false, "")];
        let before = summarize(PRODUCTION, &cases, vec![
            CaseResult { id: "a".into(), rank: Some(1), latest_ok: None },
            CaseResult { id: "b".into(), rank: None, latest_ok: None },
        ]);
        let swapped = summarize(PRODUCTION, &cases, vec![
            CaseResult { id: "a".into(), rank: Some(7), latest_ok: None },
            CaseResult { id: "b".into(), rank: Some(2), latest_ok: None },
        ]);
        let c = compare(&Baseline::from_report(&before), &swapped);
        assert_eq!((c.lost.as_slice(), c.gained.as_slice()), (&["a".to_string()][..], &["b".to_string()][..]));
        assert!(!c.is_regression());
        let worse = summarize(PRODUCTION, &cases, vec![
            CaseResult { id: "a".into(), rank: None, latest_ok: None },
            CaseResult { id: "b".into(), rank: None, latest_ok: None },
        ]);
        assert!(compare(&Baseline::from_report(&before), &worse).is_regression());
    }

    #[test]
    fn a_new_hard_case_is_not_a_regression() {
        let cases = [case("a", &["s"], false, ""), case("new", &["s"], false, "")];
        let before = summarize(PRODUCTION, &cases[..1], vec![CaseResult { id: "a".into(), rank: Some(1), latest_ok: None }]);
        let now = summarize(PRODUCTION, &cases, vec![
            CaseResult { id: "a".into(), rank: Some(1), latest_ok: None },
            CaseResult { id: "new".into(), rank: None, latest_ok: None },
        ]);
        let c = compare(&Baseline::from_report(&before), &now);
        assert!(!c.is_regression(), "aggregate recall dropped, but no shared case was lost");
        assert_eq!(c.new, ["new"]);
    }

    #[test]
    fn baseline_round_trips_through_json() {
        let r = summarize(PRODUCTION, &[case("a", &["s"], false, "")], vec![CaseResult { id: "a".into(), rank: Some(3), latest_ok: None }]);
        let b = Baseline::from_report(&r);
        let back: Baseline = serde_json::from_str(&serde_json::to_string(&b).unwrap()).unwrap();
        assert_eq!(back, b);
    }
}
