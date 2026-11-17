//! Qdrant-backed hybrid index: each point carries a dense (`dense`) and a sparse (`sparse`) vector
//! and search fuses both rankings with reciprocal rank fusion.
//!
//! Qdrant is a derived index here — the source stores (session logs, teams.db) remain the truth —
//! so every write is an idempotent upsert keyed by [`point_id`] and a collection can be dropped
//! and rebuilt at any time.

pub mod lexical;

use std::collections::HashMap;

use embed::Embedding;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    Condition, CountPointsBuilder, CreateCollectionBuilder, CreateFieldIndexCollectionBuilder,
    DeletePointsBuilder, Distance, FieldType, Filter, Fusion, KeywordIndexParamsBuilder, Modifier, NamedVectors,
    PointId, PointStruct, PrefetchQueryBuilder, Query, QueryPointsBuilder, Range, RetrievedPoint,
    GetPointsBuilder, ScoredPoint, ScrollPointsBuilder, SparseVectorParamsBuilder, SparseVectorsConfigBuilder,
    UpsertPointsBuilder, Vector, VectorInput, VectorParamsBuilder, VectorsConfigBuilder,
    point_id::PointIdOptions,
};
use serde_json::{Map, Value};
use uuid::Uuid;

pub const DENSE: &str = "dense";
pub const SPARSE: &str = "sparse";
/// BM25 term weights (see [`lexical`]); only on stores opened [`Store::with_lexical`].
pub const BM25: &str = "bm25";

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
    #[error("BM25 migration: {0}")]
    Migration(String),
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
/// on the `ts` payload field. `not_equals` excludes points whose field has that value; a point
/// without the field is kept.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SearchFilter {
    pub equals: Vec<(String, String)>,
    pub flags: Vec<(String, bool)>,
    pub ints: Vec<(String, i64)>,
    /// `field` equals any of the values. An empty list matches nothing.
    pub any: Vec<(String, Vec<String>)>,
    pub not_equals: Vec<(String, String)>,
    pub ts_from: Option<i64>,
    pub ts_to: Option<i64>,
}

impl SearchFilter {
    pub fn eq(mut self, field: impl Into<String>, value: impl Into<String>) -> Self {
        self.equals.push((field.into(), value.into()));
        self
    }

    pub fn ne(mut self, field: impl Into<String>, value: impl Into<String>) -> Self {
        self.not_equals.push((field.into(), value.into()));
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
        let excluded: Vec<Condition> =
            self.not_equals.iter().map(|(field, value)| Condition::matches(field.clone(), value.clone())).collect();
        if conditions.is_empty() && excluded.is_empty() {
            return None;
        }
        let mut f = Filter::must(conditions);
        f.must_not = excluded;
        Some(f)
    }
}

/// Timeout for every Qdrant request.
pub const REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

pub struct Store {
    client: Qdrant,
    collection: String,
    dim: usize,
    lexical: bool,
    migration_lock: Option<std::path::PathBuf>,
}

impl Store {
    pub fn connect(url: &str, api_key: Option<String>, collection: &str, dim: usize) -> Result<Self> {
        // The client's 5 s default fails index writes and scrolls of large sessions on a busy
        // Qdrant; latency-bound callers (the inject hook) keep their own shorter budget.
        let client = Qdrant::from_url(url).api_key(api_key).timeout(REQUEST_TIMEOUT).skip_compatibility_check().build()?;
        Ok(Self { client, collection: collection.to_owned(), dim, lexical: false, migration_lock: None })
    }

    /// Points also carry a [`BM25`] vector computed from their payload `text`, and
    /// [`Self::ensure_collection`] migrates a collection created without it.
    pub fn with_lexical(mut self) -> Self {
        self.lexical = true;
        self
    }

    /// File locked for the whole migration, so two processes (the daemon and `ctx index
    /// --local`) can never interleave its drops. An OS lock, released if the process dies.
    pub fn with_migration_lock(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.migration_lock = Some(path.into());
        self
    }

    pub fn collection(&self) -> &str {
        &self.collection
    }

    /// Creates the collection and payload indexes if missing; safe to call on every start.
    /// `ts` always gets an integer index because [`SearchFilter`] ranges over it.
    ///
    /// On a lexical store this also migrates a collection created without the [`BM25`] vector:
    /// Qdrant cannot add a named vector to an existing collection, so the points are copied
    /// aside, the collection is recreated and they are copied back with BM25 added. Stored
    /// vectors are copied, not recomputed, so nothing is re-embedded or re-summarised. Two
    /// processes must not migrate at once: only the indexing path calls it, and
    /// [`Self::with_migration_lock`] makes a second one fail instead of interleaving.
    pub async fn ensure_collection(&self, fields: &[Field]) -> Result<()> {
        if self.lexical {
            self.migrate_to_lexical().await?;
        }
        if !self.client.collection_exists(&self.collection).await? {
            self.create(&self.collection, self.lexical).await?;
        }
        for field in fields.iter().chain([&Field::Integer("ts")]) {
            self.index_field(*field).await?;
        }
        Ok(())
    }

    async fn create(&self, name: &str, bm25: bool) -> Result<()> {
        let mut dense = VectorsConfigBuilder::default();
        dense.add_named_vector_params(DENSE, VectorParamsBuilder::new(self.dim as u64, Distance::Cosine));
        // No IDF modifier: bge-m3 sparse weights are already learned term importances, and
        // re-weighting them by corpus IDF would double-count rarity.
        let mut sparse = SparseVectorsConfigBuilder::default();
        sparse.add_named_vector_params(SPARSE, SparseVectorParamsBuilder::default());
        if bm25 {
            // The stored BM25 weights are the term-frequency half only; IDF comes from Qdrant.
            sparse.add_named_vector_params(BM25, SparseVectorParamsBuilder::default().modifier(Modifier::Idf));
        }
        self.client
            .create_collection(CreateCollectionBuilder::new(name).vectors_config(dense).sparse_vectors_config(sparse))
            .await?;
        Ok(())
    }

    fn migration_collection(&self) -> String {
        format!("{}__migrate", self.collection)
    }

    async fn has_bm25(&self, name: &str) -> Result<bool> {
        let info = self.client.collection_info(name).await?;
        Ok(info
            .result
            .and_then(|i| i.config)
            .and_then(|c| c.params)
            .and_then(|p| p.sparse_vectors_config)
            .is_some_and(|s| s.map.contains_key(BM25)))
    }

    /// The migration as a resumable state machine. The original collection is dropped only
    /// once the copy aside holds every point, and the copy aside is dropped only once the
    /// recreated collection holds them all again, so a crash at any step loses nothing:
    /// - original without BM25 and no larger copy aside: (re)start from scratch; a leftover
    ///   copy is from an interrupted first copy and is discarded;
    /// - original without BM25 but a larger copy aside: the original was dropped and something
    ///   older (a previous ctx build) recreated it with a few points; the copy is the truth;
    /// - copy aside present and the original missing or already with BM25: finish copying back.
    ///   Only points the original lacks are copied, so anything written to it since (a
    ///   re-index, a redaction rewrite) is never overwritten by the older copy.
    async fn migrate_to_lexical(&self) -> Result<()> {
        let _lock = match &self.migration_lock {
            Some(path) => Some(lock_file(path)?),
            None => None,
        };
        let aside = self.migration_collection();
        let original_exists = self.client.collection_exists(&self.collection).await?;
        let aside_exists = self.client.collection_exists(&aside).await?;
        let mut original_ready = original_exists && self.has_bm25(&self.collection).await?;
        if original_exists && !original_ready && aside_exists && self.count_in(&aside).await? > self.count_in(&self.collection).await? {
            // What the older build wrote is newer than the copy: fold it in before dropping.
            self.copy(&self.collection, &aside, false, false).await?;
            self.client.delete_collection(&self.collection).await?;
            self.create(&self.collection, true).await?;
            original_ready = true;
        }
        if original_exists && !original_ready {
            if aside_exists {
                self.client.delete_collection(&aside).await?;
            }
            self.create(&aside, false).await?;
            self.copy(&self.collection, &aside, false, false).await?;
            if self.count_in(&aside).await? != self.count_in(&self.collection).await? {
                return Err(StoreError::Migration(format!("the copy in {aside} does not match {}", self.collection)));
            }
            self.client.delete_collection(&self.collection).await?;
        } else if !aside_exists {
            return Ok(());
        }
        if !self.client.collection_exists(&self.collection).await? {
            self.create(&self.collection, true).await?;
        }
        self.copy(&aside, &self.collection, true, true).await?;
        // At least as many: the original may also hold points written after the copy aside.
        if self.count_in(&self.collection).await? < self.count_in(&aside).await? {
            return Err(StoreError::Migration(format!("{} is missing points of {aside}", self.collection)));
        }
        self.client.delete_collection(&aside).await?;
        Ok(())
    }

    async fn count_in(&self, name: &str) -> Result<u64> {
        Ok(self.client.count(CountPointsBuilder::new(name).exact(true)).await?.result.map_or(0, |r| r.count))
    }

    /// Copies every point with its stored vectors and payload, adding BM25 when `bm25`; with
    /// `keep_existing`, points `to` already has are left as they are.
    async fn copy(&self, from: &str, to: &str, bm25: bool, keep_existing: bool) -> Result<()> {
        const PAGE: u32 = 256;
        let mut offset: Option<PointId> = None;
        loop {
            let mut req = ScrollPointsBuilder::new(from).limit(PAGE).with_payload(true).with_vectors(true);
            if let Some(o) = offset.take() {
                req = req.offset(o);
            }
            let resp = self.client.scroll(req).await?;
            let mut page = resp.result;
            if keep_existing && !page.is_empty() {
                let ids: Vec<PointId> = page.iter().filter_map(|p| p.id.clone()).collect();
                let existing = self.client.get_points(GetPointsBuilder::new(to, ids).with_payload(false).with_vectors(false)).await?;
                let existing: std::collections::HashSet<PointId> = existing.result.into_iter().filter_map(|p| p.id).collect();
                page.retain(|p| p.id.as_ref().is_none_or(|id| !existing.contains(id)));
            }
            let points = page.into_iter().map(|p| copied_point(p, bm25)).collect::<Result<Vec<_>>>()?;
            if !points.is_empty() {
                self.client.upsert_points(UpsertPointsBuilder::new(to, points).wait(true)).await?;
            }
            match resp.next_page_offset {
                Some(next) => offset = Some(next),
                None => return Ok(()),
            }
        }
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
        let mut vectors = NamedVectors::default()
            .add_vector(DENSE, doc.embedding.dense)
            .add_vector(SPARSE, Vector::new_sparse(indices, values));
        if self.lexical {
            vectors = with_bm25(vectors, &doc.payload);
        }
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
        // An exclusion alone still selects nearly everything, so it counts as empty here.
        let Some(mut f) = filter.to_qdrant().filter(|f| !f.must.is_empty()) else {
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

    /// Deletes every point matching `filter`. Refuses an empty filter (that would wipe the whole
    /// collection). For the optional prune tools; callers count first for a dry run.
    pub async fn delete_matching(&self, filter: &SearchFilter) -> Result<()> {
        self.delete_matching_except(filter, &[]).await
    }

    /// Sets `fields` on every point matching `filter`, keeping their vectors and other fields.
    /// Refuses an empty filter, like the deletes.
    pub async fn set_payload(&self, filter: &SearchFilter, fields: Map<String, Value>) -> Result<()> {
        let Some(f) = filter.to_qdrant().filter(|f| !f.must.is_empty()) else {
            return Err(StoreError::Payload("refusing to update with an empty filter".into()));
        };
        let payload = qdrant_client::Payload::try_from(Value::Object(fields)).map_err(|e| StoreError::Payload(e.to_string()))?;
        self.client
            .set_payload(qdrant_client::qdrant::SetPayloadPointsBuilder::new(&self.collection, payload).points_selector(f).wait(true))
            .await?;
        Ok(())
    }

    /// Distinct string values of `field` across points matching `filter` (e.g. the `session_id`s
    /// about to be pruned, so their state rows can be removed too).
    pub async fn distinct_payload(&self, filter: &SearchFilter, field: &str, limit: u32) -> Result<Vec<String>> {
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for h in self.scroll(filter, limit).await? {
            if let Some(v) = h.payload.get(field).and_then(Value::as_str) {
                if seen.insert(v.to_owned()) {
                    out.push(v.to_owned());
                }
            }
        }
        Ok(out)
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

    /// The BM25 leg alone (needs a lexical collection): best `limit` points matching `filter`.
    /// A query without any token matches nothing.
    pub async fn search_bm25(&self, text: &str, filter: &SearchFilter, limit: u64) -> Result<Vec<Hit>> {
        let (indices, values) = lexical::query(text);
        if indices.is_empty() || filter.matches_nothing() {
            return Ok(Vec::new());
        }
        let mut req = QueryPointsBuilder::new(&self.collection)
            .using(BM25)
            .query(Query::new_nearest(VectorInput::new_sparse(indices, values)))
            .limit(limit)
            .with_payload(true);
        if let Some(f) = filter.to_qdrant() {
            req = req.filter(f);
        }
        self.client.query(req).await?.result.into_iter().map(to_hit).collect()
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

fn lock_file(path: &std::path::Path) -> Result<std::fs::File> {
    let busy = || StoreError::Migration(format!("another process holds {}; try again when it is done", path.display()));
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let file = std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(path).map_err(|_| busy())?;
    file.try_lock().map_err(|_| busy())?;
    Ok(file)
}

/// BM25 from the payload's `text`; a point without text gets no BM25 vector.
fn with_bm25(vectors: NamedVectors, payload: &Map<String, Value>) -> NamedVectors {
    match payload.get("text").and_then(Value::as_str).map(lexical::document) {
        Some((indices, values)) if !indices.is_empty() => vectors.add_vector(BM25, Vector::new_sparse(indices, values)),
        _ => vectors,
    }
}

/// A scrolled point as an upsert, its stored vectors unchanged; any old BM25 is recomputed.
fn copied_point(p: RetrievedPoint, bm25: bool) -> Result<PointStruct> {
    use qdrant_client::qdrant::vector_output::Vector as Out;
    use qdrant_client::qdrant::vectors_output::VectorsOptions;
    let id = p.id.ok_or(StoreError::MissingId)?;
    let mut vectors = NamedVectors::default();
    let Some(VectorsOptions::Vectors(named)) = p.vectors.and_then(|v| v.vectors_options) else {
        return Err(StoreError::Migration(format!("point {id:?} has no named vectors")));
    };
    if !named.vectors.contains_key(DENSE) {
        return Err(StoreError::Migration(format!("point {id:?} has no {DENSE} vector")));
    }
    {
        for (name, v) in named.vectors {
            if name == BM25 {
                continue;
            }
            vectors = match v.into_vector() {
                Out::Dense(d) => vectors.add_vector(name, d.data),
                Out::Sparse(s) => vectors.add_vector(name, Vector::new_sparse(s.indices, s.values)),
                Out::MultiDense(_) => return Err(StoreError::Migration(format!("unexpected multi-vector {name}"))),
            };
        }
    }
    let payload = payload_to_json(p.payload);
    if bm25 {
        vectors = with_bm25(vectors, &payload);
    }
    let payload = qdrant_client::Payload::try_from(Value::Object(payload)).map_err(|e| StoreError::Payload(e.to_string()))?;
    Ok(PointStruct::new(id, vectors, payload))
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
    fn not_equals_becomes_must_not_alone_or_beside_must() {
        let alone = SearchFilter::default().ne("team", "Chat").to_qdrant().unwrap();
        assert!(alone.must.is_empty());
        assert_eq!(alone.must_not.len(), 1);
        let both = SearchFilter::default().eq("kind", "thread").ne("team", "Chat").to_qdrant().unwrap();
        assert_eq!((both.must.len(), both.must_not.len()), (1, 1));
    }

    #[test]
    fn flags_become_bool_match_conditions() {
        let f = SearchFilter::default().flag("subagent", false).to_qdrant().unwrap();
        assert_eq!(f.must.len(), 1);
    }
}
