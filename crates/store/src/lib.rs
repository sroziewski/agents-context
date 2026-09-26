//! Qdrant-backed hybrid index: each point carries a dense (`dense`) and a sparse (`sparse`) vector
//! and search fuses both rankings with reciprocal rank fusion.
//!
//! Qdrant is a derived index here — the source stores (session logs, teams.db) remain the truth —
//! so every write is an idempotent upsert keyed by [`point_id`] and a collection can be dropped
//! and rebuilt at any time.

use std::collections::HashMap;

use embed::Embedding;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    Condition, CountPointsBuilder, CreateCollectionBuilder, CreateFieldIndexCollectionBuilder,
    DeletePointsBuilder, Distance, FieldType, Filter, Fusion, KeywordIndexParamsBuilder, NamedVectors,
    PointId, PointStruct, PrefetchQueryBuilder, Query, QueryPointsBuilder, Range, RetrievedPoint,
    ScoredPoint, ScrollPointsBuilder, SparseVectorParamsBuilder, SparseVectorsConfigBuilder,
    UpsertPointsBuilder, Vector, VectorInput, VectorParamsBuilder, VectorsConfigBuilder,
    point_id::PointIdOptions,
};
use serde_json::{Map, Value};
use uuid::Uuid;

pub const DENSE: &str = "dense";
pub const SPARSE: &str = "sparse";

/// Fixed namespace so the same `(source, key)` maps to the same point id on every machine and run.
const ID_NAMESPACE: Uuid = Uuid::from_u128(0x6a1c_2f0e_8d3b_4c57_9e21_7b44_d0a5_c3f9);

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("qdrant: {0}")]
    Qdrant(#[from] qdrant_client::QdrantError),
    #[error("payload must be a JSON object: {0}")]
    Payload(String),
    #[error("dense vector has {got} dimensions, collection expects {expected}")]
    Dimension { expected: usize, got: usize },
    #[error("point returned without a UUID id")]
    MissingId,
}

pub type Result<T> = std::result::Result<T, StoreError>;

/// Stable point id for a record: re-indexing the same record overwrites instead of duplicating.
pub fn point_id(source: &str, key: &str) -> Uuid {
    // The separator can't occur in `source` names we control, so "a\x1fb" never collides with
    // a different split of the same characters.
    Uuid::new_v5(&ID_NAMESPACE, format!("{source}\x1f{key}").as_bytes())
}

#[derive(Debug, Clone)]
pub struct Doc {
    pub id: Uuid,
    pub embedding: Embedding,
    pub payload: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub id: Uuid,
    pub score: f32,
    pub payload: Map<String, Value>,
}

/// A payload field to index. `Tenant` is a keyword field that partitions the data (the project);
/// Qdrant then co-locates each tenant's points, which keeps filtered search fast as projects grow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Field {
    Keyword(&'static str),
    Tenant(&'static str),
    Integer(&'static str),
    Bool(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchMode {
    #[default]
    Hybrid,
    Dense,
    Sparse,
}

/// Payload filter. All conditions must hold. `ts_from`/`ts_to` are inclusive epoch milliseconds
/// on the `ts` payload field.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SearchFilter {
    pub equals: Vec<(String, String)>,
    pub flags: Vec<(String, bool)>,
    pub ints: Vec<(String, i64)>,
    /// `field` equals any of the values. An empty list matches nothing.
    pub any: Vec<(String, Vec<String>)>,
    pub ts_from: Option<i64>,
    pub ts_to: Option<i64>,
}

impl SearchFilter {
    pub fn eq(mut self, field: impl Into<String>, value: impl Into<String>) -> Self {
        self.equals.push((field.into(), value.into()));
        self
    }

    pub fn any_of(mut self, field: impl Into<String>, values: Vec<String>) -> Self {
        self.any.push((field.into(), values));
        self
    }

    /// True when some `any_of` list is empty, i.e. the filter can match nothing; callers skip
    /// the query instead of relying on how Qdrant treats an empty keyword list.
    pub fn matches_nothing(&self) -> bool {
        self.any.iter().any(|(_, v)| v.is_empty())
    }

    pub fn int(mut self, field: impl Into<String>, value: i64) -> Self {
        self.ints.push((field.into(), value));
        self
    }

    pub fn flag(mut self, field: impl Into<String>, value: bool) -> Self {
        self.flags.push((field.into(), value));
        self
    }

    pub fn between(mut self, from: Option<i64>, to: Option<i64>) -> Self {
        self.ts_from = from;
        self.ts_to = to;
        self
    }

    fn to_qdrant(&self) -> Option<Filter> {
        let mut conditions: Vec<Condition> = self
            .equals
            .iter()
            .map(|(field, value)| Condition::matches(field.clone(), value.clone()))
            .chain(self.flags.iter().map(|(field, value)| Condition::matches(field.clone(), *value)))
            .chain(self.ints.iter().map(|(field, value)| Condition::matches(field.clone(), *value)))
            .chain(self.any.iter().map(|(field, values)| Condition::matches(field.clone(), values.clone())))
            .collect();
        if self.ts_from.is_some() || self.ts_to.is_some() {
            conditions.push(Condition::range(
                "ts",
                Range {
                    gte: self.ts_from.map(|v| v as f64),
                    lte: self.ts_to.map(|v| v as f64),
                    ..Default::default()
                },
            ));
        }
        (!conditions.is_empty()).then(|| Filter::must(conditions))
    }
}

pub struct Store {
    client: Qdrant,
    collection: String,
    dim: usize,
}

impl Store {
    pub fn connect(url: &str, api_key: Option<String>, collection: &str, dim: usize) -> Result<Self> {
        let client = Qdrant::from_url(url).api_key(api_key).skip_compatibility_check().build()?;
        Ok(Self { client, collection: collection.to_owned(), dim })
    }

    pub fn collection(&self) -> &str {
        &self.collection
    }

    /// Creates the collection and payload indexes if missing; safe to call on every start.
    /// `ts` always gets an integer index because [`SearchFilter`] ranges over it.
    pub async fn ensure_collection(&self, fields: &[Field]) -> Result<()> {
        if !self.client.collection_exists(&self.collection).await? {
            let mut dense = VectorsConfigBuilder::default();
            dense.add_named_vector_params(DENSE, VectorParamsBuilder::new(self.dim as u64, Distance::Cosine));
            // No IDF modifier: bge-m3 sparse weights are already learned term importances, and
            // re-weighting them by corpus IDF would double-count rarity.
            let mut sparse = SparseVectorsConfigBuilder::default();
            sparse.add_named_vector_params(SPARSE, SparseVectorParamsBuilder::default());
            self.client
                .create_collection(
                    CreateCollectionBuilder::new(&self.collection)
                        .vectors_config(dense)
                        .sparse_vectors_config(sparse),
                )
                .await?;
        }
        for field in fields.iter().chain([&Field::Integer("ts")]) {
            self.index_field(*field).await?;
        }
        Ok(())
    }

    async fn index_field(&self, field: Field) -> Result<()> {
        // Creating an index that already exists with the same type is a no-op in Qdrant.
        let req = match field {
            Field::Keyword(name) => CreateFieldIndexCollectionBuilder::new(&self.collection, name, FieldType::Keyword),
            Field::Integer(name) => CreateFieldIndexCollectionBuilder::new(&self.collection, name, FieldType::Integer),
            Field::Bool(name) => CreateFieldIndexCollectionBuilder::new(&self.collection, name, FieldType::Bool),
            Field::Tenant(name) => CreateFieldIndexCollectionBuilder::new(&self.collection, name, FieldType::Keyword)
                .field_index_params(KeywordIndexParamsBuilder::default().is_tenant(true)),
        };
        self.client.create_field_index(req.wait(true)).await?;
        Ok(())
    }

    pub async fn upsert(&self, docs: Vec<Doc>) -> Result<()> {
        if docs.is_empty() {
            return Ok(());
        }
        let points = docs
            .into_iter()
            .map(|d| self.to_point(d))
            .collect::<Result<Vec<_>>>()?;
        self.client
            .upsert_points(UpsertPointsBuilder::new(&self.collection, points).wait(true))
            .await?;
        Ok(())
    }

    fn to_point(&self, doc: Doc) -> Result<PointStruct> {
        self.check_dim(&doc.embedding)?;
        let (indices, values) = doc.embedding.sparse.into_parts();
        let vectors = NamedVectors::default()
            .add_vector(DENSE, doc.embedding.dense)
            .add_vector(SPARSE, Vector::new_sparse(indices, values));
        let payload = qdrant_client::Payload::try_from(Value::Object(doc.payload))
            .map_err(|e| StoreError::Payload(e.to_string()))?;
        Ok(PointStruct::new(doc.id.to_string(), vectors, payload))
    }

    fn check_dim(&self, e: &Embedding) -> Result<()> {
        if e.dense.len() == self.dim {
            Ok(())
        } else {
            Err(StoreError::Dimension { expected: self.dim, got: e.dense.len() })
        }
    }

    pub async fn delete(&self, ids: &[Uuid]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let ids: Vec<PointId> = ids.iter().map(|id| id.to_string().into()).collect();
        self.client
            .delete_points(DeletePointsBuilder::new(&self.collection).points(ids).wait(true))
            .await?;
        Ok(())
    }

    /// Deletes every point matching `filter` except `keep`. Used after re-upserting a session:
    /// upsert-then-prune never leaves a window where the session is missing from the index.
    pub async fn delete_matching_except(&self, filter: &SearchFilter, keep: &[Uuid]) -> Result<()> {
        let Some(mut f) = filter.to_qdrant() else {
            // An empty filter would prune the whole collection; never what a caller means.
            return Err(StoreError::Payload("refusing to prune with an empty filter".into()));
        };
        if !keep.is_empty() {
            let ids: Vec<PointId> = keep.iter().map(|id| id.to_string().into()).collect();
            f.must_not.push(Condition::has_id(ids));
        }
        self.client
            .delete_points(DeletePointsBuilder::new(&self.collection).points(f).wait(true))
            .await?;
        Ok(())
    }

    /// All points matching `filter`, up to `limit`, without vectors; order is unspecified.
    pub async fn scroll(&self, filter: &SearchFilter, limit: u32) -> Result<Vec<Hit>> {
        let mut out = Vec::new();
        self.scroll_pages(filter, limit, |page| {
            out.extend(page);
            std::future::ready(Ok(()))
        })
        .await?;
        Ok(out)
    }

    /// Visits up to `limit` matching points page by page. One gRPC reply is capped at 4 MiB by
    /// default, which a single scroll of a few thousand chunk payloads exceeds, and callers
    /// that rewrite points should not hold the whole collection in memory either.
    pub async fn scroll_pages<F, Fut>(&self, filter: &SearchFilter, limit: u32, mut visit: F) -> Result<()>
    where
        F: FnMut(Vec<Hit>) -> Fut,
        Fut: std::future::Future<Output = Result<()>>,
    {
        const PAGE: u32 = 256;
        if filter.matches_nothing() {
            return Ok(());
        }
        let qfilter = filter.to_qdrant();
        let mut offset: Option<PointId> = None;
        let mut seen: u32 = 0;
        while seen < limit {
            let mut req = ScrollPointsBuilder::new(&self.collection).limit(PAGE.min(limit - seen)).with_payload(true);
            if let Some(f) = qfilter.clone() {
                req = req.filter(f);
            }
            if let Some(o) = offset.take() {
                req = req.offset(o);
            }
            let resp = self.client.scroll(req).await?;
            let page = resp.result.into_iter().map(retrieved_to_hit).collect::<Result<Vec<_>>>()?;
            seen += page.len() as u32;
            let done = page.is_empty();
            visit(page).await?;
            match resp.next_page_offset {
                Some(next) if !done => offset = Some(next),
                _ => break,
            }
        }
        Ok(())
    }

    pub async fn count(&self, filter: &SearchFilter) -> Result<u64> {
        if filter.matches_nothing() {
            return Ok(0);
        }
        let mut req = CountPointsBuilder::new(&self.collection).exact(true);
        if let Some(f) = filter.to_qdrant() {
            req = req.filter(f);
        }
        Ok(self.client.count(req).await?.result.map_or(0, |r| r.count))
    }

    pub async fn search(
        &self,
        query: &Embedding,
        filter: &SearchFilter,
        limit: u64,
        mode: SearchMode,
    ) -> Result<Vec<Hit>> {
        self.check_dim(query)?;
        if filter.matches_nothing() {
            return Ok(Vec::new());
        }
        let filter = filter.to_qdrant();
        let dense = VectorInput::new_dense(query.dense.clone());
        let sparse = VectorInput::new_sparse(query.sparse.indices().to_vec(), query.sparse.values().to_vec());

        let mut req = QueryPointsBuilder::new(&self.collection).limit(limit).with_payload(true);
        match mode {
            SearchMode::Dense => req = req.using(DENSE).query(Query::new_nearest(dense)),
            SearchMode::Sparse => req = req.using(SPARSE).query(Query::new_nearest(sparse)),
            SearchMode::Hybrid => {
                // Each leg over-fetches so a document ranked moderately by both legs can still
                // win the fusion; the filter must sit on the legs, since fusion only re-ranks.
                let depth = (limit * 4).max(20);
                let leg = |using: &str, input: VectorInput| {
                    let mut p = PrefetchQueryBuilder::default()
                        .using(using)
                        .query(Query::new_nearest(input))
                        .limit(depth);
                    if let Some(f) = filter.clone() {
                        p = p.filter(f);
                    }
                    p
                };
                req = req
                    .add_prefetch(leg(DENSE, dense))
                    .add_prefetch(leg(SPARSE, sparse))
                    .query(Query::new_fusion(Fusion::Rrf));
            }
        }
        if let Some(f) = filter {
            req = req.filter(f);
        }
        self.client.query(req).await?.result.into_iter().map(to_hit).collect()
    }

    pub async fn drop_collection(&self) -> Result<()> {
        if self.client.collection_exists(&self.collection).await? {
            self.client.delete_collection(&self.collection).await?;
        }
        Ok(())
    }
}

fn to_hit(p: ScoredPoint) -> Result<Hit> {
    let id = match p.id.and_then(|id| id.point_id_options) {
        Some(PointIdOptions::Uuid(s)) => Uuid::parse_str(&s).map_err(|_| StoreError::MissingId)?,
        _ => return Err(StoreError::MissingId),
    };
    Ok(Hit { id, score: p.score, payload: payload_to_json(p.payload) })
}

fn retrieved_to_hit(p: RetrievedPoint) -> Result<Hit> {
    let id = match p.id.and_then(|id| id.point_id_options) {
        Some(PointIdOptions::Uuid(s)) => Uuid::parse_str(&s).map_err(|_| StoreError::MissingId)?,
        _ => return Err(StoreError::MissingId),
    };
    Ok(Hit { id, score: 0.0, payload: payload_to_json(p.payload) })
}

fn payload_to_json(p: HashMap<String, qdrant_client::qdrant::Value>) -> Map<String, Value> {
    p.into_iter().map(|(k, v)| (k, Value::from(v))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use qdrant_client::qdrant::condition::ConditionOneOf;

    #[test]
    fn point_id_is_deterministic() {
        assert_eq!(point_id("claude", "abc"), point_id("claude", "abc"));
    }

    #[test]
    fn point_id_differs_by_source_and_key() {
        let a = point_id("claude", "abc");
        assert_ne!(a, point_id("opencode", "abc"));
        assert_ne!(a, point_id("claude", "abd"));
    }

    #[test]
    fn point_id_does_not_collide_on_concatenation() {
        assert_ne!(point_id("ab", "c"), point_id("a", "bc"));
    }

    #[test]
    fn point_id_is_uuid_v5() {
        assert_eq!(point_id("x", "y").get_version_num(), 5);
    }

    #[test]
    fn empty_filter_maps_to_none() {
        assert!(SearchFilter::default().to_qdrant().is_none());
    }

    #[test]
    fn filter_combines_equals_and_range_with_must() {
        let f = SearchFilter::default()
            .eq("source", "teams")
            .eq("channel", "General")
            .between(Some(10), None)
            .to_qdrant()
            .unwrap();
        assert_eq!(f.must.len(), 3);
        assert!(f.should.is_empty() && f.must_not.is_empty());
        let Some(ConditionOneOf::Field(range)) = &f.must[2].condition_one_of else {
            panic!("expected field condition");
        };
        assert_eq!(range.key, "ts");
        let r = range.range.as_ref().unwrap();
        assert_eq!((r.gte, r.lte), (Some(10.0), None));
    }

    #[test]
    fn upper_bound_only_is_a_range() {
        let f = SearchFilter::default().between(None, Some(99)).to_qdrant().unwrap();
        assert_eq!(f.must.len(), 1);
    }

    #[test]
    fn ints_become_integer_match_conditions() {
        let f = SearchFilter::default().int("turn_index", 0).int("chunk_index", 0).to_qdrant().unwrap();
        assert_eq!(f.must.len(), 2);
    }

    #[test]
    fn flags_become_bool_match_conditions() {
        let f = SearchFilter::default().flag("subagent", false).to_qdrant().unwrap();
        assert_eq!(f.must.len(), 1);
    }
}
