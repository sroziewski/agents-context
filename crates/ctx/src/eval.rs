//! Retrieval eval: real questions with the sessions that answer them, run against the live
//! index in every search variant, so ranking changes (recency, summaries, auto-inject
//! thresholds) are measured instead of guessed.
//!
//! The query set is JSON lines, one [`Case`] per line. It names real projects and topics, so
//! it lives in the user's config directory, not in the repository; the repository tests the
//! harness on a synthetic set.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::time::Instant;

use embed::{Embedding, Reranker};
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
    /// Optional turn-level target: the specific turns that answer the query, as shown in
    /// results (1-based, `session` + `turn`). Lets the eval measure a reranker, which reorders
    /// turns, not just whether the right session appears. A case with none is session-only.
    #[serde(default)]
    pub expect_turns: Vec<TurnRef>,
    #[serde(default)]
    pub lang: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnRef {
    pub session: String,
    /// 1-based, matching the `turn N` shown in search results (payload `turn_index` + 1).
    pub turn: usize,
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

/// The turns shown, in order, as `(session_id, displayed_turn)` where `displayed_turn` is the
/// 1-based number a reader sees. This is what a reranker reorders, so turn metrics use it, not
/// the summaries.
pub fn turn_order(r: &Results) -> Vec<(String, usize)> {
    r.turns
        .iter()
        .map(|h| {
            let sid = h.payload.get("session_id").and_then(|v| v.as_str()).unwrap_or_default().to_owned();
            let turn = h.payload.get("turn_index").and_then(|v| v.as_i64()).unwrap_or(-1) + 1;
            (sid, turn.max(0) as usize)
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseResult {
    pub id: String,
    /// 1-based rank of the first expected session, if it was found within the depth.
    pub rank: Option<usize>,
    /// For `latest` cases: whether the preferred session came before the superseded ones.
    pub latest_ok: Option<bool>,
    /// 1-based rank, among the turns shown, of the first turn in `expect_turns`; `None` when
    /// the case has no turn targets or none were found. This is the reranker's metric.
    #[serde(default)]
    pub turn_rank: Option<usize>,
}

pub fn judge(case: &Case, order: &[String]) -> CaseResult {
    judge_turns(case, order, &[])
}

/// Session judgement plus, when the case has `expect_turns`, the rank of the first matching
/// turn in `turns` (the shown turn list from [`turn_order`]).
pub fn judge_turns(case: &Case, order: &[String], turns: &[(String, usize)]) -> CaseResult {
    let rank = order.iter().position(|s| case.expect.contains(s)).map(|i| i + 1);
    let latest_ok = case.latest.then(|| rank.is_some_and(|r| order[r - 1] == case.expect[0]));
    let turn_rank = (!case.expect_turns.is_empty())
        .then(|| turns.iter().position(|(s, t)| case.expect_turns.iter().any(|e| &e.session == s && e.turn == *t)).map(|i| i + 1))
        .flatten();
    CaseResult { id: case.id.clone(), rank, latest_ok, turn_rank }
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
    /// (turn found in top 5, cases that carry turn targets). `(0, 0)` when none do.
    pub turn_hit_at_5: (usize, usize),
    /// Language → (found in top 5, cases).
    pub by_lang: BTreeMap<String, (usize, usize)>,
}

pub fn summarize(name: &str, cases: &[Case], results: Vec<CaseResult>) -> VariantReport {
    let n = results.len().max(1) as f64;
    let within = |k: usize| results.iter().filter(|r| r.rank.is_some_and(|x| x <= k)).count() as f64 / n;
    let mrr = results.iter().map(|r| r.rank.map_or(0.0, |x| 1.0 / x as f64)).sum::<f64>() / n;
    let latest = results.iter().filter_map(|r| r.latest_ok).fold((0, 0), |(ok, all), x| (ok + usize::from(x), all + 1));
    // Only cases with turn targets count toward turn hit@5; a session-only case is not a miss.
    let turn_targeted: HashSet<&str> = cases.iter().filter(|c| !c.expect_turns.is_empty()).map(|c| c.id.as_str()).collect();
    let turn_hit_at_5 = results
        .iter()
        .filter(|r| turn_targeted.contains(r.id.as_str()))
        .fold((0, 0), |(hit, all), r| (hit + usize::from(r.turn_rank.is_some_and(|x| x <= 5)), all + 1));
    let lang: HashMap<&str, &str> = cases.iter().map(|c| (c.id.as_str(), c.lang.as_str())).collect();
    let mut by_lang: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for r in &results {
        let entry = by_lang.entry(lang.get(r.id.as_str()).copied().unwrap_or_default().to_owned()).or_default();
        entry.0 += usize::from(r.rank.is_some_and(|x| x <= 5));
        entry.1 += 1;
    }
    VariantReport { name: name.to_owned(), recall_at_1: within(1), recall_at_5: within(5), mrr, latest, turn_hit_at_5, by_lang, cases: results }
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
            results.push(judge_turns(c, &session_order(&r, v.summaries), &turn_order(&r)));
        }
        reports.push(summarize(v.name, cases, results));
    }
    Ok(reports)
}

/// DEV-28 spike: a dense pool of `pool` turns per query, reordered by the cross-encoder, then
/// judged like any variant. Returns the report and the per-query rerank latency in ms (the
/// added cost on top of retrieval). Retrieval stays dense (production); summaries are left out
/// so the metric is purely the turn ranking the reranker produces.
pub async fn run_reranked(
    ctx: &Context,
    cases: &[Case],
    vectors: &HashMap<String, Embedding>,
    reranker: &mut dyn Reranker,
    pool: u64,
    name: &'static str,
) -> anyhow::Result<(VariantReport, Vec<f64>)> {
    let positives: Vec<&Case> = cases.iter().filter(|c| !c.expect.is_empty()).collect();
    let mut results = Vec::new();
    let mut latency_ms = Vec::new();
    for c in &positives {
        let mut q = Query::new(c.query.clone(), c.scope(), pool);
        q.mode = SearchMode::Dense;
        let r = ctx.search_embedded(&vectors[&c.id], &q).await?;
        let texts: Vec<&str> = r.turns.iter().map(|h| h.payload.get("text").and_then(|v| v.as_str()).unwrap_or_default()).collect();
        // The cross-encoder is blocking CPU work; keep it off the async workers.
        let t0 = Instant::now();
        let order = tokio::task::block_in_place(|| reranker.order(&c.query, &texts))?;
        latency_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        let reordered = Results { summaries: Vec::new(), turns: order.iter().map(|&i| r.turns[i].clone()).collect() };
        results.push(judge_turns(c, &session_order(&reordered, false), &turn_order(&reordered)));
    }
    Ok((summarize(name, cases, results), latency_ms))
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((p * (sorted.len() - 1) as f64).round() as usize).min(sorted.len() - 1)]
}

pub fn format_latency(label: &str, mut ms: Vec<f64>) -> String {
    ms.sort_by(f64::total_cmp);
    let mean = if ms.is_empty() { 0.0 } else { ms.iter().sum::<f64>() / ms.len() as f64 };
    format!("{label}: p50 {:.0} ms · p95 {:.0} ms · max {:.0} ms · mean {:.0} ms (n={})", percentile(&ms, 0.5), percentile(&ms, 0.95), percentile(&ms, 1.0), mean, ms.len())
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
    let turns = reports.iter().any(|r| r.turn_hit_at_5.1 > 0);
    let mut out = format!("{:<26} {:>5} {:>5} {:>7} {:>7}", "variant", "R@1", "R@5", "MRR", "latest");
    if turns {
        out.push_str(&format!(" {:>8}", "turn@5"));
    }
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
        if turns {
            let (hit, all) = r.turn_hit_at_5;
            out.push_str(&format!(" {:>8}", if all == 0 { "-".to_owned() } else { format!("{hit}/{all}") }));
        }
        for l in &langs {
            let (hit, all) = r.by_lang.get(l).copied().unwrap_or_default();
            out.push_str(&format!(" {:>8}", format!("{hit}/{all}")));
        }
    }
    out
}

/// Paired-bootstrap 95% CI for a per-case metric's delta (variant `b` minus `a`), over the
/// cases both share. Resampling cases together (paired) cancels the shared per-case difficulty,
/// so the CI reflects the ranking change, not which questions are hard. Deterministic: a fixed
/// seed and a fixed resample count, so the same reports always give the same interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ci {
    pub delta: f64,
    pub lo: f64,
    pub hi: f64,
}

impl Ci {
    /// The interval excludes 0, i.e. the change is beyond bootstrap noise.
    pub fn significant(&self) -> bool {
        self.lo > 0.0 || self.hi < 0.0
    }
}

/// SplitMix64 — a tiny deterministic PRNG, so the CI needs no external rng crate and is stable
/// across platforms.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

const BOOTSTRAP_RESAMPLES: usize = 1000;
const BOOTSTRAP_SEED: u64 = 0x5EED_5EED_5EED_5EED;

/// Per-case `(a, b)` metric pairs from two reports, matched by id.
fn paired(a: &VariantReport, b: &VariantReport, metric: impl Fn(&CaseResult) -> f64) -> Vec<(f64, f64)> {
    let bm: HashMap<&str, &CaseResult> = b.cases.iter().map(|c| (c.id.as_str(), c)).collect();
    a.cases.iter().filter_map(|ca| bm.get(ca.id.as_str()).map(|cb| (metric(ca), metric(cb)))).collect()
}

fn bootstrap_ci(pairs: &[(f64, f64)]) -> Ci {
    let mean_delta = |sel: &dyn Fn(usize) -> usize| {
        let n = pairs.len();
        (0..n).map(|i| { let (a, b) = pairs[sel(i)]; b - a }).sum::<f64>() / n.max(1) as f64
    };
    let delta = mean_delta(&|i| i);
    if pairs.is_empty() {
        return Ci { delta: 0.0, lo: 0.0, hi: 0.0 };
    }
    let mut rng = SplitMix64(BOOTSTRAP_SEED);
    let n = pairs.len();
    let mut deltas: Vec<f64> = (0..BOOTSTRAP_RESAMPLES)
        .map(|_| {
            let s: f64 = (0..n).map(|_| { let (a, b) = pairs[rng.below(n)]; b - a }).sum();
            s / n as f64
        })
        .collect();
    deltas.sort_by(f64::total_cmp);
    let pct = |p: f64| deltas[((p * (BOOTSTRAP_RESAMPLES - 1) as f64).round()) as usize];
    Ci { delta, lo: pct(0.025), hi: pct(0.975) }
}

fn rr(r: &CaseResult) -> f64 {
    r.rank.map_or(0.0, |x| 1.0 / x as f64)
}
fn hit1(r: &CaseResult) -> f64 {
    f64::from(u8::from(r.rank == Some(1)))
}
fn turn_rr(r: &CaseResult) -> f64 {
    r.turn_rank.map_or(0.0, |x| 1.0 / x as f64)
}

/// `b` vs `a`: the per-case rank moves plus bootstrap CIs on R@1, MRR and (when present)
/// turn-MRR. This is what `ctx eval --compare a b` prints.
pub fn format_compare(a: &VariantReport, b: &VariantReport, cases: &[Case]) -> String {
    let query: HashMap<&str, &str> = cases.iter().map(|c| (c.id.as_str(), c.query.as_str())).collect();
    let am: HashMap<&str, &CaseResult> = a.cases.iter().map(|c| (c.id.as_str(), c)).collect();
    let mut moved: Vec<String> = Vec::new();
    for cb in &b.cases {
        let Some(ca) = am.get(cb.id.as_str()) else { continue };
        if ca.rank != cb.rank {
            let show = |r: Option<usize>| r.map_or("—".to_owned(), |x| x.to_string());
            moved.push(format!("  {:<16} {:>3} -> {:<3}  {}", cb.id, show(ca.rank), show(cb.rank), query.get(cb.id.as_str()).copied().unwrap_or_default()));
        }
    }
    moved.sort();
    let ci_line = |label: &str, ci: Ci| {
        let mark = if ci.significant() { "significant" } else { "within noise" };
        format!("  {label:<9} Δ {:+.3}  95% CI [{:+.3}, {:+.3}]  {mark}", ci.delta, ci.lo, ci.hi)
    };
    let mut out = format!("compare: {} -> {}\n", a.name, b.name);
    out.push_str(&ci_line("R@1", bootstrap_ci(&paired(a, b, hit1))));
    out.push('\n');
    out.push_str(&ci_line("MRR", bootstrap_ci(&paired(a, b, rr))));
    if a.turn_hit_at_5.1 > 0 || b.turn_hit_at_5.1 > 0 {
        out.push('\n');
        out.push_str(&ci_line("turn-MRR", bootstrap_ci(&paired(a, b, turn_rr))));
    }
    out.push_str(&format!("\n\nranks moved ({}):\n{}", moved.len(), if moved.is_empty() { "  (none)".to_owned() } else { moved.join("\n") }));
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
        Case { id: id.into(), query: "q".into(), project: None, expect: expect.iter().map(|s| s.to_string()).collect(), latest, expect_turns: vec![], lang: lang.into() }
    }

    fn hit(session: &str) -> Hit {
        Hit { id: uuid::Uuid::nil(), score: 1.0, payload: json!({ "session_id": session }).as_object().cloned().unwrap() }
    }

    fn turn_hit(session: &str, turn: i64) -> Hit {
        Hit { id: uuid::Uuid::nil(), score: 1.0, payload: json!({ "session_id": session, "turn_index": turn - 1 }).as_object().cloned().unwrap() }
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
        assert_eq!(judge(&c, &order(&["x", "new", "old"])), CaseResult { id: "a".into(), rank: Some(2), latest_ok: Some(true), turn_rank: None });
        assert_eq!(judge(&c, &order(&["old", "new"])).latest_ok, Some(false));
        assert_eq!(judge(&c, &order(&["x"])), CaseResult { id: "a".into(), rank: None, latest_ok: Some(false), turn_rank: None });
        assert_eq!(judge(&case("b", &["s"], false, ""), &order(&["s"])).latest_ok, None);
    }

    #[test]
    fn turn_order_and_turn_rank() {
        let r = Results { summaries: vec![hit("s9")], turns: vec![turn_hit("s1", 3), turn_hit("s1", 4), turn_hit("s2", 1)] };
        assert_eq!(turn_order(&r), [("s1".into(), 3), ("s1".into(), 4), ("s2".into(), 1)]);
        let mut c = case("t", &["s1"], false, "");
        c.expect_turns = vec![TurnRef { session: "s2".into(), turn: 1 }];
        assert_eq!(judge_turns(&c, &session_order(&r, true), &turn_order(&r)).turn_rank, Some(3));
        c.expect_turns = vec![TurnRef { session: "s2".into(), turn: 9 }];
        assert_eq!(judge_turns(&c, &[], &turn_order(&r)).turn_rank, None, "target turn not shown");
        assert_eq!(judge_turns(&case("s", &["s1"], false, ""), &[], &turn_order(&r)).turn_rank, None, "no turn target");
    }

    #[test]
    fn summarize_computes_recall_mrr_and_languages() {
        let cases = [case("a", &["s"], false, "pl"), case("b", &["s"], false, "en"), case("c", &["s"], false, "en")];
        let results = vec![
            CaseResult { id: "a".into(), rank: Some(1), latest_ok: None, turn_rank: None },
            CaseResult { id: "b".into(), rank: Some(4), latest_ok: Some(true), turn_rank: None },
            CaseResult { id: "c".into(), rank: None, latest_ok: Some(false), turn_rank: None },
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
            CaseResult { id: "a".into(), rank: Some(1), latest_ok: None, turn_rank: None },
            CaseResult { id: "b".into(), rank: None, latest_ok: None, turn_rank: None },
        ]);
        let swapped = summarize(PRODUCTION, &cases, vec![
            CaseResult { id: "a".into(), rank: Some(7), latest_ok: None, turn_rank: None },
            CaseResult { id: "b".into(), rank: Some(2), latest_ok: None, turn_rank: None },
        ]);
        let c = compare(&Baseline::from_report(&before), &swapped);
        assert_eq!((c.lost.as_slice(), c.gained.as_slice()), (&["a".to_string()][..], &["b".to_string()][..]));
        assert!(!c.is_regression());
        let worse = summarize(PRODUCTION, &cases, vec![
            CaseResult { id: "a".into(), rank: None, latest_ok: None, turn_rank: None },
            CaseResult { id: "b".into(), rank: None, latest_ok: None, turn_rank: None },
        ]);
        assert!(compare(&Baseline::from_report(&before), &worse).is_regression());
    }

    #[test]
    fn a_new_hard_case_is_not_a_regression() {
        let cases = [case("a", &["s"], false, ""), case("new", &["s"], false, "")];
        let before = summarize(PRODUCTION, &cases[..1], vec![CaseResult { id: "a".into(), rank: Some(1), latest_ok: None, turn_rank: None }]);
        let now = summarize(PRODUCTION, &cases, vec![
            CaseResult { id: "a".into(), rank: Some(1), latest_ok: None, turn_rank: None },
            CaseResult { id: "new".into(), rank: None, latest_ok: None, turn_rank: None },
        ]);
        let c = compare(&Baseline::from_report(&before), &now);
        assert!(!c.is_regression(), "aggregate recall dropped, but no shared case was lost");
        assert_eq!(c.new, ["new"]);
    }

    #[test]
    fn baseline_round_trips_through_json() {
        let r = summarize(PRODUCTION, &[case("a", &["s"], false, "")], vec![CaseResult { id: "a".into(), rank: Some(3), latest_ok: None, turn_rank: None }]);
        let b = Baseline::from_report(&r);
        let back: Baseline = serde_json::from_str(&serde_json::to_string(&b).unwrap()).unwrap();
        assert_eq!(back, b);
    }

    #[test]
    fn bootstrap_ci_flags_a_real_gain_and_ignores_noise() {
        let uniform = |v: f64| (0..40).map(|_| (0.0, v)).collect::<Vec<_>>();
        let ci = bootstrap_ci(&uniform(1.0));
        assert!((ci.delta - 1.0).abs() < 1e-9 && ci.lo > 0.0 && ci.significant(), "{ci:?}");
        let zero = bootstrap_ci(&(0..40).map(|_| (0.5, 0.5)).collect::<Vec<_>>());
        assert_eq!((zero.delta, zero.lo, zero.hi), (0.0, 0.0, 0.0));
        assert!(!zero.significant());
        assert_eq!(bootstrap_ci(&[]).delta, 0.0);
        let mixed: Vec<(f64, f64)> = (0..40).map(|i| (0.0, f64::from(i % 3 == 0))).collect();
        assert_eq!(bootstrap_ci(&mixed), bootstrap_ci(&mixed), "deterministic across runs");
        let n = bootstrap_ci(&mixed);
        assert!(n.lo <= n.delta && n.delta <= n.hi);
    }

    #[test]
    fn latency_percentiles() {
        let out = format_latency("rerank", vec![10.0, 20.0, 30.0, 40.0, 100.0]);
        assert!(out.contains("p50 30 ms") && out.contains("p95 100 ms") && out.contains("max 100 ms") && out.contains("n=5"), "{out}");
        assert!(format_latency("x", vec![]).contains("n=0"));
    }

    #[test]
    fn compare_reports_rank_moves_and_cis() {
        let cases = [case("a", &["s"], false, ""), case("b", &["s"], false, "")];
        let cr = |id: &str, rank| CaseResult { id: id.into(), rank, latest_ok: None, turn_rank: None };
        let a = summarize("dense", &cases, vec![cr("a", Some(3)), cr("b", Some(2))]);
        let b = summarize("prod", &cases, vec![cr("a", Some(1)), cr("b", Some(2))]);
        let out = format_compare(&a, &b, &cases);
        assert!(out.starts_with("compare: dense -> prod"), "{out}");
        assert!(out.contains("3 -> 1"), "{out}");
        assert!(out.contains("ranks moved (1)"), "{out}");
        assert!(out.contains("R@1") && out.contains("MRR") && !out.contains("turn-MRR"), "{out}");
    }
}
