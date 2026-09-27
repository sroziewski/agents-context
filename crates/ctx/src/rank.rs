//! Recency-aware re-ranking: newer sessions supersede older decisions, so among comparably
//! relevant results the newer one should come first.
//!
//! The decay multiplies the fused score instead of adding to it. RRF scores are rank-based
//! and tiny (~1/60), so an additive term would either be noise or swamp relevance entirely;
//! a factor in `[1 - weight, 1]` keeps a strong old match ahead of a weak new one.

use store::Hit;

const DAY_MS: f64 = 86_400_000.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Recency {
    /// Share of the score that decays with age, 0 (off) to 1.
    pub weight: f64,
    /// Age at which the decaying share has halved.
    pub half_life_days: f64,
}

impl Recency {
    pub const OFF: Recency = Recency { weight: 0.0, half_life_days: 90.0 };

    pub fn is_off(&self) -> bool {
        self.weight <= 0.0 || self.half_life_days <= 0.0
    }

    /// `1 - weight` for something infinitely old, 1 for something from `now`. Timestamps in
    /// the future (clock skew between machines) count as new, not as a bonus.
    pub fn factor(&self, ts_ms: i64, now_ms: i64) -> f64 {
        if self.is_off() {
            return 1.0;
        }
        let w = self.weight.clamp(0.0, 1.0);
        let age_days = ((now_ms - ts_ms).max(0) as f64) / DAY_MS;
        (1.0 - w) + w * 0.5f64.powf(age_days / self.half_life_days)
    }

    /// How many candidates to fetch so a newer hit ranked a little lower by relevance can move
    /// into the top `limit`.
    pub fn pool(&self, limit: u64) -> u64 {
        if self.is_off() { limit } else { (limit * 3).max(20) }
    }

    /// Re-scores `hits` by their `ts` payload field and keeps the best `limit`. Ties keep the
    /// relevance order.
    pub fn rerank(&self, mut hits: Vec<Hit>, now_ms: i64, limit: usize) -> Vec<Hit> {
        if !self.is_off() {
            for h in &mut hits {
                let ts = h.payload.get("ts").and_then(serde_json::Value::as_i64).unwrap_or(0);
                h.score = (f64::from(h.score) * self.factor(ts, now_ms)) as f32;
            }
            hits.sort_by(|a, b| b.score.total_cmp(&a.score));
        }
        hits.truncate(limit);
        hits
    }
}

impl Default for Recency {
    fn default() -> Self {
        Self::OFF
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const NOW: i64 = 1_790_000_000_000;

    fn hit(score: f32, age_days: i64) -> Hit {
        let ts = NOW - age_days * 86_400_000;
        Hit { id: uuid::Uuid::from_u128(ts as u128), score, payload: json!({ "ts": ts }).as_object().cloned().unwrap() }
    }

    #[test]
    fn factor_halves_the_decaying_share_every_half_life() {
        let r = Recency { weight: 0.4, half_life_days: 30.0 };
        assert!((r.factor(NOW, NOW) - 1.0).abs() < 1e-9);
        assert!((r.factor(NOW - 30 * 86_400_000, NOW) - 0.8).abs() < 1e-9);
        assert!((r.factor(0, NOW) - 0.6).abs() < 1e-3, "floor is 1 - weight");
        assert_eq!(r.factor(NOW + 86_400_000, NOW), 1.0, "future timestamps get no bonus");
    }

    #[test]
    fn newer_hit_overtakes_a_slightly_better_older_one_but_not_a_much_better_one() {
        let r = Recency { weight: 0.3, half_life_days: 60.0 };
        let out = r.rerank(vec![hit(1.0, 120), hit(0.95, 1)], NOW, 2);
        assert_eq!(out[0].payload["ts"], json!(NOW - 86_400_000));
        let out = r.rerank(vec![hit(1.0, 120), hit(0.5, 1)], NOW, 2);
        assert_eq!(out[0].payload["ts"], json!(NOW - 120 * 86_400_000), "relevance still dominates");
    }

    #[test]
    fn off_keeps_order_and_scores_and_only_truncates() {
        let hits = vec![hit(0.2, 500), hit(0.9, 0), hit(0.1, 1)];
        let out = Recency::OFF.rerank(hits.clone(), NOW, 2);
        assert_eq!(out, hits[..2]);
        assert_eq!(Recency::OFF.pool(6), 6);
        assert_eq!(Recency { weight: 0.3, half_life_days: 60.0 }.pool(6), 20);
    }

    #[test]
    fn hit_without_timestamp_counts_as_oldest() {
        let r = Recency { weight: 0.5, half_life_days: 10.0 };
        let mut undated = hit(1.0, 0);
        undated.payload.remove("ts");
        let out = r.rerank(vec![undated, hit(0.8, 0)], NOW, 2);
        assert!(out[0].payload.contains_key("ts"));
    }
}
