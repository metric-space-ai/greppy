//! Persistent vector embeddings and exact nearest-neighbor search.
//!
//! R5 intentionally starts with an exact scan. It is deterministic, easy to
//! verify, and gives the EmbeddingGemma integration a real vector-search
//! substrate before ANN/index acceleration is introduced.

use std::collections::HashMap;

use rusqlite::{params, types::Value as SqlValue, OptionalExtension};

use crate::store::Store;
use crate::store_error::{Error, Result};
use crate::workspace_state::now_iso8601;

/// Input row for inserting or updating one embedding.
#[derive(Debug, Clone, PartialEq)]
pub struct NewVectorEmbedding {
    pub project: String,
    pub model_id: String,
    pub prompt_version: String,
    pub task: String,
    pub node_id: Option<i64>,
    pub chunk_idx: i64,
    pub qualified_name: String,
    pub file_path: String,
    pub start_line: i64,
    pub end_line: i64,
    pub content_sha256: String,
    pub graph_generation: u64,
    pub vector: Vec<f32>,
}

/// One persisted embedding row plus its decoded vector.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorEmbedding {
    pub id: i64,
    pub project: String,
    pub model_id: String,
    pub prompt_version: String,
    pub task: String,
    pub node_id: Option<i64>,
    pub chunk_idx: i64,
    pub qualified_name: String,
    pub file_path: String,
    pub start_line: i64,
    pub end_line: i64,
    pub content_sha256: String,
    pub graph_generation: u64,
    pub dim: usize,
    pub vector_norm: f32,
    pub vector: Vec<f32>,
    pub created_at: String,
}

/// Identity of an unchanged embedding chunk eligible for cross-generation
/// reuse. Every field is part of the semantic cache contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReusableVectorEmbeddingKey<'a> {
    pub project: &'a str,
    pub model_id: &'a str,
    pub prompt_version: &'a str,
    pub task: &'a str,
    pub qualified_name: &'a str,
    pub chunk_idx: i64,
    pub content_sha256: &'a str,
}

/// Scope and ranking policy for exact vector search.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorSearchQuery<'a> {
    pub project: &'a str,
    pub model_id: &'a str,
    pub prompt_version: &'a str,
    pub task: &'a str,
    /// When set, only embeddings from the current graph snapshot are searched.
    /// Passing `None` is allowed for diagnostics, never for visible augment
    /// decisions.
    pub graph_generation: Option<u64>,
    pub file_path: Option<&'a str>,
    pub limit: usize,
    pub min_score: Option<f32>,
}

/// One nearest-neighbor hit.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorSearchHit {
    pub embedding: VectorEmbedding,
    pub score: f32,
}

impl Store {
    /// Insert or update one embedding row.
    ///
    /// The uniqueness key includes `content_sha256`: a changed code span creates
    /// a new row, while query paths filter by `graph_generation` so stale rows
    /// cannot surface. Cleanup is handled by `prune_vector_embeddings_before_generation`.
    pub fn upsert_vector_embedding(&mut self, e: &NewVectorEmbedding) -> Result<i64> {
        validate_embedding_input(e)?;
        let dim = e.vector.len();
        let norm = vector_norm(&e.vector);
        let blob = encode_f32_le(&e.vector);
        // int8 candidate copy (see migration 0011): 4x smaller, used for
        // candidate SELECTION in large scans; winners are re-scored from
        // the exact f32 blob.
        let (i8_blob, i8_scale) = quantize_i8(&e.vector);
        let created_at = now_iso8601();
        let tx = self.transaction()?;
        let id = tx
            .raw()
            .prepare_cached(
                "INSERT INTO main.vector_embeddings
                   (project, model_id, prompt_version, task, node_id, chunk_idx,
                    qualified_name, file_path, start_line, end_line,
                    content_sha256, graph_generation, dim, vector_norm,
                    vector, created_at, vector_i8, i8_scale)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)
                 ON CONFLICT(project, model_id, prompt_version, task, qualified_name, chunk_idx, content_sha256)
                 DO UPDATE SET
                    node_id = excluded.node_id,
                    chunk_idx = excluded.chunk_idx,
                    file_path = excluded.file_path,
                    start_line = excluded.start_line,
                    end_line = excluded.end_line,
                    graph_generation = excluded.graph_generation,
                    dim = excluded.dim,
                    vector_norm = excluded.vector_norm,
                    vector = excluded.vector,
                    created_at = excluded.created_at,
                    vector_i8 = excluded.vector_i8,
                    i8_scale = excluded.i8_scale
                 RETURNING id",
            )?
            .query_row(
                params![
                    e.project,
                    e.model_id,
                    e.prompt_version,
                    e.task,
                    e.node_id,
                    e.chunk_idx,
                    e.qualified_name,
                    e.file_path,
                    e.start_line,
                    e.end_line,
                    e.content_sha256,
                    e.graph_generation as i64,
                    dim as i64,
                    norm as f64,
                    blob,
                    created_at,
                    i8_blob,
                    i8_scale as f64,
                ],
                |row| row.get(0),
            )?;
        tx.commit()?;
        Ok(id)
    }

    /// Fetch an embedding by primary key.
    pub fn get_vector_embedding(&self, id: i64) -> Result<Option<VectorEmbedding>> {
        self.conn()
            .query_row(
                "SELECT id, project, model_id, prompt_version, task, node_id, chunk_idx,
                        qualified_name, file_path, start_line, end_line,
                        content_sha256, graph_generation, dim, vector_norm,
                        vector, created_at
                 FROM vector_embeddings WHERE id = ?1",
                params![id],
                row_to_vector_embedding,
            )
            .optional()
            .map_err(Error::Sqlite)
    }

    /// Find a byte-identical chunk embedded with the same model and prompt.
    /// Snapshot builders use this to advance unchanged rows to a new graph
    /// generation without running inference again.
    pub fn find_reusable_vector_embedding(
        &self,
        key: &ReusableVectorEmbeddingKey<'_>,
    ) -> Result<Option<VectorEmbedding>> {
        self.conn()
            .query_row(
                "SELECT id, project, model_id, prompt_version, task, node_id, chunk_idx,
                        qualified_name, file_path, start_line, end_line,
                        content_sha256, graph_generation, dim, vector_norm,
                        vector, created_at
                 FROM vector_embeddings
                 WHERE project = ?1 AND model_id = ?2 AND prompt_version = ?3
                   AND task = ?4 AND qualified_name = ?5 AND chunk_idx = ?6
                   AND content_sha256 = ?7
                 ORDER BY graph_generation DESC LIMIT 1",
                params![
                    key.project,
                    key.model_id,
                    key.prompt_version,
                    key.task,
                    key.qualified_name,
                    key.chunk_idx,
                    key.content_sha256
                ],
                row_to_vector_embedding,
            )
            .optional()
            .map_err(Error::Sqlite)
    }

    /// Count embedding rows for a vector-search scope.
    pub fn count_vector_embeddings(
        &self,
        project: &str,
        model_id: &str,
        prompt_version: &str,
        task: &str,
        graph_generation: Option<u64>,
    ) -> Result<i64> {
        let generation = if self.is_overlay() {
            None
        } else {
            graph_generation.map(|g| g as i64)
        };
        let n = self.conn().query_row(
            "SELECT COUNT(*)
             FROM vector_embeddings
             WHERE project = ?1
               AND model_id = ?2
               AND prompt_version = ?3
               AND task = ?4
               AND (?5 IS NULL OR graph_generation = ?5)",
            params![project, model_id, prompt_version, task, generation],
            |row| row.get(0),
        )?;
        Ok(n)
    }

    /// Distinct embedding model ids present for `project`, regardless of
    /// generation. Used by the inline auto-reindex to detect that a store
    /// HAD code-span vectors (so it must rebuild them for the new
    /// generation instead of silently stranding them).
    pub fn vector_model_ids(&self, project: &str) -> Result<Vec<String>> {
        let conn = self.conn();
        let mut stmt = conn.prepare(
            "SELECT DISTINCT model_id FROM vector_embeddings WHERE project = ?1 ORDER BY model_id",
        )?;
        let rows = stmt.query_map(params![project], |row| row.get::<_, String>(0))?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Delete all embeddings for a file. Called by node/file reindex paths so
    /// stale vectors do not survive symbol deletion.
    pub fn delete_vector_embeddings_for_file(
        &mut self,
        project: &str,
        file_path: &str,
    ) -> Result<usize> {
        let tx = self.transaction()?;
        let n = tx.raw().execute(
            "DELETE FROM main.vector_embeddings WHERE project = ?1 AND file_path = ?2",
            params![project, file_path],
        )?;
        tx.commit()?;
        Ok(n)
    }

    /// Delete embeddings older than the given graph generation for a project.
    pub fn prune_vector_embeddings_before_generation(
        &mut self,
        project: &str,
        graph_generation: u64,
    ) -> Result<usize> {
        let tx = self.transaction()?;
        let n = tx.raw().execute(
            "DELETE FROM main.vector_embeddings
             WHERE project = ?1 AND graph_generation < ?2",
            params![project, graph_generation as i64],
        )?;
        tx.commit()?;
        Ok(n)
    }

    /// Exact cosine nearest-neighbor search over persisted embeddings.
    ///
    /// This is real vector search, not token lookup: the query vector is compared
    /// numerically against stored vectors. Ranking is total and deterministic:
    /// score descending, then `qualified_name`, then row id.
    pub fn vector_search_exact(
        &self,
        query_vector: &[f32],
        q: &VectorSearchQuery<'_>,
    ) -> Result<Vec<VectorSearchHit>> {
        validate_query_vector(query_vector)?;
        if q.limit == 0 {
            return Ok(Vec::new());
        }
        let query_norm = vector_norm(query_vector);
        // Base and Delta generations are layer-local. Visibility already
        // selects the current rows, so applying either layer's generation to
        // the union would incorrectly discard the other layer.
        let generation = if self.is_overlay() {
            None
        } else {
            q.graph_generation.map(|g| g as i64)
        };
        let file = q.file_path.unwrap_or("");

        // ---------------------------------------------------- pass 1: score
        // The old single-pass scan fully decoded EVERY candidate row (~10
        // string columns + an owned Vec<f32>) before its score was known, so
        // row materialisation — not the dot products — dominated large scans.
        // Pass 1 touches only the three columns scoring needs and reuses one
        // f32 buffer (zero per-row allocations); pass 2 fully decodes just
        // the winners. A brute-force scan structured this way runs at memory
        // bandwidth — which is also why a kd-tree is NOT the lever here: at
        // 768 dims the measured pruning rate is 0% (distance concentration),
        // i.e. a kd-tree visits every point AND pays pointer-chasing on top.
        struct Cand {
            id: i64,
            node_id: Option<i64>,
            score: f32,
            qualified_name: String,
        }
        // Pool by node during the scan so many matching chunks from one long
        // definition cannot consume the top-k. We still overfetch distinct
        // nodes before the final f32 rescore because pass 1 may use the i8 copy
        // as an approximate candidate selector.
        let overfetch = q.limit.saturating_mul(8).saturating_add(32);
        // The i8 blob is 4x smaller than the f32 blob, which is the actual
        // scaling fix: 200k f32 rows are ~615 MB of blob reads per query
        // (page-cache thrashing made the measured scan superlinear); the i8
        // copy keeps the scan cache-resident far past that.
        let (query_i8, query_scale) = quantize_i8(query_vector);
        // Pass-1 min_score filtering uses a small epsilon so a true match
        // sitting marginally above the caller's floor cannot be dropped by
        // quantization error; pass 2 re-applies the exact floor.
        let approx_min = q.min_score.map(|m| m - 0.01);
        let mut best_by_node: HashMap<String, Cand> = HashMap::new();
        let mut buf: Vec<f32> = Vec::with_capacity(query_vector.len());
        {
            let mut stmt = self.conn().prepare_cached(
                "SELECT id, node_id, qualified_name, dim, vector_norm, i8_scale,
                        COALESCE(vector_i8, vector)
                 FROM vector_embeddings
                 WHERE project = ?1
                   AND model_id = ?2
                   AND prompt_version = ?3
                   AND task = ?4
                   AND (?5 IS NULL OR graph_generation = ?5)
                   AND (?6 = '' OR file_path = ?6)",
            )?;
            let mut rows = stmt.query(params![
                q.project,
                q.model_id,
                q.prompt_version,
                q.task,
                generation,
                file
            ])?;
            while let Some(row) = rows.next()? {
                let id: i64 = row.get(0)?;
                let node_id: Option<i64> = row.get(1)?;
                let qualified_name: String = row.get(2)?;
                let dim = row.get::<_, i64>(3)? as usize;
                let norm = row.get::<_, f64>(4)? as f32;
                if dim != query_vector.len() {
                    return Err(Error::Invalid(format!(
                        "stored vector dimension mismatch for {}: stored {}, query {}",
                        self.vector_row_qname(id),
                        dim,
                        query_vector.len()
                    )));
                }
                if norm <= 0.0 || !norm.is_finite() {
                    return Err(Error::Invalid(format!(
                        "stored vector has invalid norm for {}",
                        self.vector_row_qname(id)
                    )));
                }
                let i8_scale: Option<f64> = row.get(5)?;
                let blob = match row.get_ref(6)? {
                    rusqlite::types::ValueRef::Blob(b) => b,
                    _ => {
                        return Err(Error::Invalid(format!(
                            "stored vector is not a blob for {}",
                            self.vector_row_qname(id)
                        )))
                    }
                };
                let score = if let Some(vscale) = i8_scale {
                    // i8 candidate path: blob is the quantized copy.
                    if blob.len() != dim {
                        return Err(Error::Invalid(format!(
                            "vector_i8 blob length mismatch for {}: bytes {}, dim {}",
                            self.vector_row_qname(id),
                            blob.len(),
                            dim
                        )));
                    }
                    let idot = dot_i8(&query_i8, blob) as f32;
                    (idot * query_scale * vscale as f32) / (query_norm * norm)
                } else {
                    // Legacy row without an i8 copy: exact f32 path.
                    if !decode_f32_le_into(blob, dim, &mut buf) {
                        return Err(Error::Invalid(format!(
                            "vector blob length mismatch for {}: bytes {}, dim {}",
                            self.vector_row_qname(id),
                            blob.len(),
                            dim
                        )));
                    }
                    dot(query_vector, &buf) / (query_norm * norm)
                };
                if let Some(min) = approx_min {
                    if score < min {
                        continue;
                    }
                }
                let key = vector_dedup_key(node_id, &qualified_name);
                match best_by_node.get_mut(&key) {
                    Some(best) => {
                        if score > best.score || (score == best.score && id < best.id) {
                            *best = Cand {
                                id,
                                node_id,
                                score,
                                qualified_name,
                            };
                        }
                    }
                    None => {
                        best_by_node.insert(
                            key,
                            Cand {
                                id,
                                node_id,
                                score,
                                qualified_name,
                            },
                        );
                    }
                }
            }
        }
        let mut top = best_by_node.into_values().collect::<Vec<_>>();
        top.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.qualified_name.cmp(&b.qualified_name))
                .then_with(|| a.id.cmp(&b.id))
        });
        top.truncate(overfetch);
        if top.is_empty() {
            return Ok(Vec::new());
        }

        // -------------------------------------------- pass 2: decode winners
        let mut node_ids = Vec::new();
        let mut qnames_without_node = Vec::new();
        for cand in &top {
            match cand.node_id {
                Some(id) => node_ids.push(id),
                None => qnames_without_node.push(cand.qualified_name.clone()),
            }
        }
        node_ids.sort_unstable();
        node_ids.dedup();
        qnames_without_node.sort();
        qnames_without_node.dedup();

        let mut key_clauses = Vec::new();
        if !node_ids.is_empty() {
            key_clauses.push(format!(
                "node_id IN ({})",
                vec!["?"; node_ids.len()].join(",")
            ));
        }
        if !qnames_without_node.is_empty() {
            key_clauses.push(format!(
                "(node_id IS NULL AND qualified_name IN ({}))",
                vec!["?"; qnames_without_node.len()].join(",")
            ));
        }
        let sql = format!(
            "SELECT id, project, model_id, prompt_version, task, node_id, chunk_idx,
                    qualified_name, file_path, start_line, end_line,
                    content_sha256, graph_generation, dim, vector_norm,
                    vector, created_at
             FROM vector_embeddings
             WHERE project = ?
               AND model_id = ?
               AND prompt_version = ?
               AND task = ?
               AND (? IS NULL OR graph_generation = ?)
               AND (? = '' OR file_path = ?)
               AND ({})",
            key_clauses.join(" OR ")
        );
        let mut stmt = self.conn().prepare(&sql)?;
        let mut values = vec![
            SqlValue::from(q.project.to_string()),
            SqlValue::from(q.model_id.to_string()),
            SqlValue::from(q.prompt_version.to_string()),
            SqlValue::from(q.task.to_string()),
            generation.map_or(SqlValue::Null, SqlValue::from),
            generation.map_or(SqlValue::Null, SqlValue::from),
            SqlValue::from(file.to_string()),
            SqlValue::from(file.to_string()),
        ];
        values.extend(node_ids.into_iter().map(SqlValue::from));
        values.extend(qnames_without_node.into_iter().map(SqlValue::from));
        let mut rows = stmt.query(rusqlite::params_from_iter(values))?;
        let mut best_exact: HashMap<String, VectorSearchHit> = HashMap::new();
        while let Some(row) = rows.next()? {
            let mut embedding = row_to_vector_embedding(row)?;
            self.canonicalize_vector_embedding(&mut embedding)?;
            // Exact re-score from the f32 blob: the pass-1 (possibly i8)
            // score only chose the candidates. The returned ranking and the
            // caller's min_score floor are always computed at full precision.
            let score = dot(query_vector, &embedding.vector) / (query_norm * embedding.vector_norm);
            if let Some(min) = q.min_score {
                if score < min {
                    continue;
                }
            }
            let key = vector_dedup_key(embedding.node_id, &embedding.qualified_name);
            let hit = VectorSearchHit { embedding, score };
            match best_exact.get_mut(&key) {
                Some(best) => {
                    if vector_hit_rank_cmp(&hit, best).is_lt() {
                        *best = hit;
                    }
                }
                None => {
                    best_exact.insert(key, hit);
                }
            }
        }
        let mut hits = best_exact.into_values().collect::<Vec<_>>();
        hits.sort_by(vector_hit_rank_cmp);
        hits.truncate(q.limit);
        Ok(hits)
    }

    /// Qualified name for error messages on the score-only scan path (which
    /// deliberately does not decode names). Best-effort: falls back to the
    /// row id when the lookup itself fails.
    fn vector_row_qname(&self, id: i64) -> String {
        self.conn()
            .query_row(
                "SELECT qualified_name FROM vector_embeddings WHERE id = ?1",
                params![id],
                |r| r.get::<_, String>(0),
            )
            .unwrap_or_else(|_| format!("vector row {id}"))
    }

    fn canonicalize_vector_embedding(&self, embedding: &mut VectorEmbedding) -> Result<()> {
        let Some(node_id) = embedding.node_id else {
            return Ok(());
        };
        if let Some(node) = self.get_node(node_id)? {
            embedding.qualified_name = node.qualified_name;
            embedding.file_path = node.file_path;
            embedding.start_line = node.start_line;
            embedding.end_line = node.end_line;
        }
        Ok(())
    }
}

fn vector_hit_rank_cmp(a: &VectorSearchHit, b: &VectorSearchHit) -> std::cmp::Ordering {
    b.score
        .total_cmp(&a.score)
        .then_with(|| a.embedding.qualified_name.cmp(&b.embedding.qualified_name))
        .then_with(|| a.embedding.chunk_idx.cmp(&b.embedding.chunk_idx))
        .then_with(|| a.embedding.id.cmp(&b.embedding.id))
}

fn vector_dedup_key(node_id: Option<i64>, qualified_name: &str) -> String {
    match node_id {
        Some(id) => format!("node:{id}"),
        None => format!("name:{qualified_name}"),
    }
}

fn validate_embedding_input(e: &NewVectorEmbedding) -> Result<()> {
    if e.project.trim().is_empty() {
        return Err(Error::Invalid(
            "vector embedding project must not be empty".into(),
        ));
    }
    if e.model_id.trim().is_empty() {
        return Err(Error::Invalid(
            "vector embedding model_id must not be empty".into(),
        ));
    }
    if e.prompt_version.trim().is_empty() {
        return Err(Error::Invalid(
            "vector embedding prompt_version must not be empty".into(),
        ));
    }
    if e.task.trim().is_empty() {
        return Err(Error::Invalid(
            "vector embedding task must not be empty".into(),
        ));
    }
    if e.qualified_name.trim().is_empty() {
        return Err(Error::Invalid(
            "vector embedding qualified_name must not be empty".into(),
        ));
    }
    if e.chunk_idx < 0 {
        return Err(Error::Invalid(
            "vector embedding chunk_idx must be non-negative".into(),
        ));
    }
    if e.content_sha256.len() != 64 || !e.content_sha256.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(Error::Invalid(
            "vector embedding content_sha256 must be a 64-character hex digest".into(),
        ));
    }
    validate_query_vector(&e.vector)?;
    Ok(())
}

fn validate_query_vector(v: &[f32]) -> Result<()> {
    if v.is_empty() {
        return Err(Error::Invalid("vector must not be empty".into()));
    }
    if !v.iter().all(|x| x.is_finite()) {
        return Err(Error::Invalid("vector contains non-finite values".into()));
    }
    let norm = vector_norm(v);
    if norm <= 0.0 || !norm.is_finite() {
        return Err(Error::Invalid("vector norm must be positive".into()));
    }
    Ok(())
}

fn vector_norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    // Four independent accumulators break the loop-carried dependency so the
    // autovectorizer emits packed fused multiply-adds — several times faster
    // than the naive zip/sum on 768-dim vectors, which matters because the
    // exact scan is a straight line of these dots. (Summation order changes
    // vs the naive loop; cosine scores move by ~1e-7, well below ranking
    // relevance.)
    let ca = a.chunks_exact(4);
    let cb = b.chunks_exact(4);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    let mut s = [0.0f32; 4];
    for (x, y) in ca.zip(cb) {
        s[0] = x[0].mul_add(y[0], s[0]);
        s[1] = x[1].mul_add(y[1], s[1]);
        s[2] = x[2].mul_add(y[2], s[2]);
        s[3] = x[3].mul_add(y[3], s[3]);
    }
    let mut t = (s[0] + s[1]) + (s[2] + s[3]);
    for (x, y) in ra.iter().zip(rb) {
        t = x.mul_add(*y, t);
    }
    t
}

/// Symmetric int8 quantization for the candidate-scan copy: one scale per
/// vector (max-abs / 127). Selection-only — winners are re-scored exactly
/// from the f32 blob, so the ~0.3% quantization error never affects the
/// final ranking, only which ~3x-overfetched candidates get re-scored.
fn quantize_i8(v: &[f32]) -> (Vec<u8>, f32) {
    let max_abs = v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    if max_abs == 0.0 || !max_abs.is_finite() {
        return (vec![0u8; v.len()], 1.0);
    }
    let scale = max_abs / 127.0;
    let inv = 127.0 / max_abs;
    let out = v
        .iter()
        .map(|x| ((x * inv).round().clamp(-127.0, 127.0) as i8) as u8)
        .collect();
    (out, scale)
}

/// Unrolled i8·i8 → i32 dot over the raw blob bytes (reinterpreted as i8).
fn dot_i8(a: &[u8], b: &[u8]) -> i64 {
    let ca = a.chunks_exact(4);
    let cb = b.chunks_exact(4);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    let mut s = [0i32; 4];
    for (x, y) in ca.zip(cb) {
        s[0] += (x[0] as i8 as i32) * (y[0] as i8 as i32);
        s[1] += (x[1] as i8 as i32) * (y[1] as i8 as i32);
        s[2] += (x[2] as i8 as i32) * (y[2] as i8 as i32);
        s[3] += (x[3] as i8 as i32) * (y[3] as i8 as i32);
    }
    let mut t = (s[0] as i64 + s[1] as i64) + (s[2] as i64 + s[3] as i64);
    for (x, y) in ra.iter().zip(rb) {
        t += (*x as i8 as i64) * (*y as i8 as i64);
    }
    t
}

fn encode_f32_le(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(std::mem::size_of_val(v));
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Buffer-reusing variant of [`decode_f32_le`] for the hot scan path: no
/// per-row allocation. Returns `false` on a length mismatch.
fn decode_f32_le_into(bytes: &[u8], dim: usize, out: &mut Vec<f32>) -> bool {
    if bytes.len() != dim * std::mem::size_of::<f32>() {
        return false;
    }
    out.clear();
    out.extend(
        bytes
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
    );
    true
}

fn decode_f32_le(bytes: &[u8], dim: usize, qname: &str) -> rusqlite::Result<Vec<f32>> {
    if bytes.len() != dim * std::mem::size_of::<f32>() {
        return Err(rusqlite::Error::FromSqlConversionFailure(
            14,
            rusqlite::types::Type::Blob,
            format!(
                "vector blob length mismatch for {qname}: bytes {}, dim {}",
                bytes.len(),
                dim
            )
            .into(),
        ));
    }
    let mut out = Vec::with_capacity(dim);
    for chunk in bytes.chunks_exact(std::mem::size_of::<f32>()) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn row_to_vector_embedding(row: &rusqlite::Row<'_>) -> rusqlite::Result<VectorEmbedding> {
    let dim_i64: i64 = row.get(13)?;
    let dim = usize::try_from(dim_i64).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(13, rusqlite::types::Type::Integer, Box::new(e))
    })?;
    let qname: String = row.get(7)?;
    let blob: Vec<u8> = row.get(15)?;
    let vector = decode_f32_le(&blob, dim, &qname)?;
    Ok(VectorEmbedding {
        id: row.get(0)?,
        project: row.get(1)?,
        model_id: row.get(2)?,
        prompt_version: row.get(3)?,
        task: row.get(4)?,
        node_id: row.get(5)?,
        chunk_idx: row.get(6)?,
        qualified_name: qname,
        file_path: row.get(8)?,
        start_line: row.get(9)?,
        end_line: row.get(10)?,
        content_sha256: row.get(11)?,
        graph_generation: row.get::<_, i64>(12)? as u64,
        dim,
        vector_norm: row.get::<_, f64>(14)? as f32,
        vector,
        created_at: row.get(16)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NewNode, Project};

    fn store_with_project(name: &str) -> Store {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: name.into(),
            indexed_at: "2026-07-01T00:00:00Z".into(),
            root_path: format!("/repos/{name}"),
        })
        .unwrap();
        s
    }

    fn node(project: &str, qname: &str, file: &str) -> NewNode {
        NewNode {
            project: project.into(),
            label: "Function".into(),
            name: qname.rsplit('.').next().unwrap_or(qname).into(),
            qualified_name: qname.into(),
            file_path: file.into(),
            start_line: 1,
            end_line: 4,
            properties: serde_json::json!({}),
        }
    }

    fn embedding(
        project: &str,
        node_id: Option<i64>,
        qname: &str,
        file: &str,
        generation: u64,
        content_sha256: &str,
        vector: Vec<f32>,
    ) -> NewVectorEmbedding {
        NewVectorEmbedding {
            project: project.into(),
            model_id: "google/embeddinggemma-300m-q4".into(),
            prompt_version: "embeddinggemma-code-retrieval-st-v2".into(),
            task: "retrieval_document".into(),
            node_id,
            chunk_idx: 0,
            qualified_name: qname.into(),
            file_path: file.into(),
            start_line: 1,
            end_line: 4,
            content_sha256: content_sha256.into(),
            graph_generation: generation,
            vector,
        }
    }

    fn query<'a>(project: &'a str, generation: Option<u64>, limit: usize) -> VectorSearchQuery<'a> {
        VectorSearchQuery {
            project,
            model_id: "google/embeddinggemma-300m-q4",
            prompt_version: "embeddinggemma-code-retrieval-st-v2",
            task: "retrieval_document",
            graph_generation: generation,
            file_path: None,
            limit,
            min_score: None,
        }
    }

    #[test]
    fn upsert_get_and_count_round_trip() {
        let mut s = store_with_project("p");
        let node_id = s
            .insert_node(&node("p", "p.payments.refund", "src/pay.rs"))
            .unwrap();
        let id = s
            .upsert_vector_embedding(&embedding(
                "p",
                Some(node_id),
                "p.payments.refund",
                "src/pay.rs",
                7,
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                vec![1.0, 0.0, 0.0],
            ))
            .unwrap();

        let got = s.get_vector_embedding(id).unwrap().unwrap();
        assert_eq!(got.qualified_name, "p.payments.refund");
        assert_eq!(got.chunk_idx, 0);
        assert_eq!(got.dim, 3);
        assert_eq!(got.vector, vec![1.0, 0.0, 0.0]);
        assert_eq!(
            s.count_vector_embeddings(
                "p",
                "google/embeddinggemma-300m-q4",
                "embeddinggemma-code-retrieval-st-v2",
                "retrieval_document",
                Some(7)
            )
            .unwrap(),
            1
        );
    }

    #[test]
    fn upsert_allows_distinct_chunks_for_same_content_hash() {
        let mut s = store_with_project("p");
        let node_id = s.insert_node(&node("p", "p.long", "src/long.rs")).unwrap();
        let mut first = embedding(
            "p",
            Some(node_id),
            "p.long",
            "src/long.rs",
            4,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            vec![1.0, 0.0],
        );
        let mut second = first.clone();
        second.chunk_idx = 1;
        second.start_line = 10;
        second.end_line = 20;
        second.vector = vec![0.0, 1.0];

        let id0 = s.upsert_vector_embedding(&first).unwrap();
        let id1 = s.upsert_vector_embedding(&second).unwrap();
        assert_ne!(id0, id1);
        assert_eq!(
            s.count_vector_embeddings(
                "p",
                "google/embeddinggemma-300m-q4",
                "embeddinggemma-code-retrieval-st-v2",
                "retrieval_document",
                Some(4),
            )
            .unwrap(),
            2
        );

        first.vector = vec![0.5, 0.5];
        let id0_again = s.upsert_vector_embedding(&first).unwrap();
        assert_eq!(id0_again, id0);
        assert_eq!(s.get_vector_embedding(id1).unwrap().unwrap().chunk_idx, 1);
    }

    #[test]
    fn exact_cosine_search_ranks_by_vector_score_not_tokens() {
        let mut s = store_with_project("p");
        s.upsert_vector_embedding(&embedding(
            "p",
            None,
            "p.alpha",
            "src/a.rs",
            3,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            vec![1.0, 0.0],
        ))
        .unwrap();
        s.upsert_vector_embedding(&embedding(
            "p",
            None,
            "p.beta",
            "src/b.rs",
            3,
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            vec![0.8, 0.6],
        ))
        .unwrap();
        s.upsert_vector_embedding(&embedding(
            "p",
            None,
            "p.gamma",
            "src/c.rs",
            3,
            "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            vec![0.0, 1.0],
        ))
        .unwrap();

        let hits = s
            .vector_search_exact(&[0.0, 1.0], &query("p", Some(3), 2))
            .unwrap();
        assert_eq!(
            hits.iter()
                .map(|h| h.embedding.qualified_name.as_str())
                .collect::<Vec<_>>(),
            vec!["p.gamma", "p.beta"]
        );
        assert!(hits[0].score > hits[1].score);
    }

    #[test]
    fn vector_search_dedups_chunks_by_node_and_fills_limit() {
        let mut s = store_with_project("p");
        let alpha_id = s.insert_node(&node("p", "p.alpha", "src/a.rs")).unwrap();
        let beta_id = s.insert_node(&node("p", "p.beta", "src/b.rs")).unwrap();
        let gamma_id = s.insert_node(&node("p", "p.gamma", "src/c.rs")).unwrap();

        let mut alpha0 = embedding(
            "p",
            Some(alpha_id),
            "p.alpha",
            "src/a.rs",
            8,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            vec![0.6, 0.8],
        );
        alpha0.start_line = 10;
        alpha0.end_line = 20;
        let mut alpha1 = embedding(
            "p",
            Some(alpha_id),
            "p.alpha",
            "src/a.rs",
            8,
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            vec![1.0, 0.0],
        );
        alpha1.chunk_idx = 1;
        alpha1.start_line = 40;
        alpha1.end_line = 50;
        s.upsert_vector_embedding(&alpha0).unwrap();
        s.upsert_vector_embedding(&alpha1).unwrap();
        s.upsert_vector_embedding(&embedding(
            "p",
            Some(beta_id),
            "p.beta",
            "src/b.rs",
            8,
            "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            vec![0.8, 0.2],
        ))
        .unwrap();
        s.upsert_vector_embedding(&embedding(
            "p",
            Some(gamma_id),
            "p.gamma",
            "src/c.rs",
            8,
            "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            vec![0.7, 0.3],
        ))
        .unwrap();

        let hits = s
            .vector_search_exact(&[1.0, 0.0], &query("p", Some(8), 3))
            .unwrap();
        assert_eq!(
            hits.iter()
                .map(|h| h.embedding.qualified_name.as_str())
                .collect::<Vec<_>>(),
            vec!["p.alpha", "p.beta", "p.gamma"]
        );
        assert_eq!(hits[0].embedding.chunk_idx, 1);
        assert_eq!(hits[0].embedding.start_line, 1);
        assert_eq!(hits[0].embedding.end_line, 4);
        assert!(hits[0].score > hits[1].score);
    }

    #[test]
    fn vector_search_streaming_top_k_keeps_late_best_candidate() {
        let mut s = store_with_project("p");
        for i in 0..20 {
            s.upsert_vector_embedding(&embedding(
                "p",
                None,
                &format!("p.poor{i:02}"),
                "src/poor.rs",
                5,
                &format!("{:064x}", i + 1),
                vec![0.0, 1.0],
            ))
            .unwrap();
        }
        s.upsert_vector_embedding(&embedding(
            "p",
            None,
            "p.best_late",
            "src/best.rs",
            5,
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            vec![1.0, 0.0],
        ))
        .unwrap();

        let hits = s
            .vector_search_exact(&[1.0, 0.0], &query("p", Some(5), 3))
            .unwrap();
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].embedding.qualified_name, "p.best_late");
        assert!(hits[0].score > hits[1].score);
    }

    #[test]
    fn vector_search_filters_stale_generation() {
        let mut s = store_with_project("p");
        s.upsert_vector_embedding(&embedding(
            "p",
            None,
            "p.old",
            "src/old.rs",
            1,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            vec![1.0, 0.0],
        ))
        .unwrap();
        s.upsert_vector_embedding(&embedding(
            "p",
            None,
            "p.current",
            "src/new.rs",
            2,
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            vec![1.0, 0.0],
        ))
        .unwrap();

        let current = s
            .vector_search_exact(&[1.0, 0.0], &query("p", Some(2), 10))
            .unwrap();
        assert_eq!(current.len(), 1);
        assert_eq!(current[0].embedding.qualified_name, "p.current");

        let all = s
            .vector_search_exact(&[1.0, 0.0], &query("p", None, 10))
            .unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn vector_search_tie_breaks_deterministically() {
        let mut s = store_with_project("p");
        s.upsert_vector_embedding(&embedding(
            "p",
            None,
            "p.zeta",
            "src/z.rs",
            1,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            vec![1.0, 0.0],
        ))
        .unwrap();
        s.upsert_vector_embedding(&embedding(
            "p",
            None,
            "p.alpha",
            "src/a.rs",
            1,
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            vec![1.0, 0.0],
        ))
        .unwrap();

        let hits = s
            .vector_search_exact(&[1.0, 0.0], &query("p", Some(1), 10))
            .unwrap();
        assert_eq!(
            hits.iter()
                .map(|h| h.embedding.qualified_name.as_str())
                .collect::<Vec<_>>(),
            vec!["p.alpha", "p.zeta"]
        );
    }

    #[test]
    fn vector_embedding_rejects_empty_zero_and_non_finite_vectors() {
        let mut s = store_with_project("p");
        for bad in [vec![], vec![0.0, 0.0], vec![1.0, f32::NAN]] {
            let err = s
                .upsert_vector_embedding(&embedding(
                    "p",
                    None,
                    "p.bad",
                    "src/bad.rs",
                    1,
                    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    bad,
                ))
                .unwrap_err();
            assert!(matches!(err, Error::Invalid(_)));
        }
    }

    #[test]
    fn prune_and_delete_file_remove_stale_vectors() {
        let mut s = store_with_project("p");
        s.upsert_vector_embedding(&embedding(
            "p",
            None,
            "p.old",
            "src/old.rs",
            1,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            vec![1.0, 0.0],
        ))
        .unwrap();
        s.upsert_vector_embedding(&embedding(
            "p",
            None,
            "p.current",
            "src/current.rs",
            3,
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            vec![1.0, 0.0],
        ))
        .unwrap();
        assert_eq!(
            s.prune_vector_embeddings_before_generation("p", 3).unwrap(),
            1
        );
        assert_eq!(
            s.count_vector_embeddings(
                "p",
                "google/embeddinggemma-300m-q4",
                "embeddinggemma-code-retrieval-st-v2",
                "retrieval_document",
                None
            )
            .unwrap(),
            1
        );
        assert_eq!(
            s.delete_vector_embeddings_for_file("p", "src/current.rs")
                .unwrap(),
            1
        );
        assert_eq!(
            s.count_vector_embeddings(
                "p",
                "google/embeddinggemma-300m-q4",
                "embeddinggemma-code-retrieval-st-v2",
                "retrieval_document",
                None
            )
            .unwrap(),
            0
        );
    }

    #[test]
    fn overlay_vector_search_merges_layers_and_suppresses_dirty_base_paths() {
        let tmp = tempfile::tempdir().unwrap();
        let base_path = tmp.path().join("base.db");
        let delta_path = tmp.path().join("delta.db");
        {
            let mut base = Store::open(&base_path).unwrap();
            base.upsert_project(&Project {
                name: "p".into(),
                indexed_at: "x".into(),
                root_path: "/base".into(),
            })
            .unwrap();
            let clean = base
                .insert_node(&node("p", "p.clean", "src/clean.rs"))
                .unwrap();
            let dirty = base
                .insert_node(&node("p", "p.old_dirty", "src/dirty.rs"))
                .unwrap();
            base.upsert_vector_embedding(&embedding(
                "p",
                Some(clean),
                "p.clean",
                "src/clean.rs",
                1,
                &"a".repeat(64),
                vec![1.0, 0.0],
            ))
            .unwrap();
            base.upsert_vector_embedding(&embedding(
                "p",
                Some(dirty),
                "p.old_dirty",
                "src/dirty.rs",
                1,
                &"b".repeat(64),
                vec![0.99, 0.01],
            ))
            .unwrap();
        }
        {
            let mut delta = Store::open(&delta_path).unwrap();
            delta
                .upsert_project(&Project {
                    name: "p".into(),
                    indexed_at: "x".into(),
                    root_path: "/delta".into(),
                })
                .unwrap();
            let node_id = delta
                .insert_node(&node("p", "p.new_dirty", "src/dirty.rs"))
                .unwrap();
            delta
                .upsert_vector_embedding(&embedding(
                    "p",
                    Some(node_id),
                    "p.new_dirty",
                    "src/dirty.rs",
                    2,
                    &"c".repeat(64),
                    vec![0.8, 0.2],
                ))
                .unwrap();
        }

        let visibility = crate::VisibilityIndex::new(["src/dirty.rs".into()], []).unwrap();
        let overlay = Store::open_overlay(&base_path, &delta_path, &visibility).unwrap();
        let hits = overlay
            .vector_search_exact(&[1.0, 0.0], &query("p", Some(2), 10))
            .unwrap();
        let qnames = hits
            .iter()
            .map(|hit| hit.embedding.qualified_name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(qnames, vec!["p.clean", "p.new_dirty"]);
        assert!(hits[0].embedding.node_id.is_some_and(|id| id < 0));
        assert!(hits[1].embedding.node_id.is_some_and(|id| id > 0));
        assert_eq!(
            overlay
                .count_vector_embeddings(
                    "p",
                    "google/embeddinggemma-300m-q4",
                    "embeddinggemma-code-retrieval-st-v2",
                    "retrieval_document",
                    Some(2),
                )
                .unwrap(),
            2
        );
    }

    #[test]
    fn ten_agents_reuse_base_embedding_without_private_duplicates() {
        let tmp = tempfile::tempdir().unwrap();
        let base_path = tmp.path().join("base.db");
        {
            let mut base = Store::open(&base_path).unwrap();
            base.upsert_project(&Project {
                name: "p".into(),
                indexed_at: "x".into(),
                root_path: "/base".into(),
            })
            .unwrap();
            let node_id = base
                .insert_node(&node("p", "p.shared", "src/shared.rs"))
                .unwrap();
            base.upsert_vector_embedding(&embedding(
                "p",
                Some(node_id),
                "p.shared",
                "src/shared.rs",
                1,
                &"d".repeat(64),
                vec![1.0, 0.0],
            ))
            .unwrap();
        }
        let before = std::fs::read(&base_path).unwrap();
        let mut permissions = std::fs::metadata(&base_path).unwrap().permissions();
        permissions.set_readonly(true);
        std::fs::set_permissions(&base_path, permissions).unwrap();

        let mut agents = Vec::new();
        for index in 0..10 {
            let base_path = base_path.clone();
            let delta_path = tmp.path().join(format!("delta-{index}.db"));
            agents.push(std::thread::spawn(move || {
                let visibility = crate::VisibilityIndex::default();
                let overlay = Store::open_overlay(&base_path, &delta_path, &visibility).unwrap();
                let hits = overlay
                    .vector_search_exact(&[1.0, 0.0], &query("p", Some(1), 10))
                    .unwrap();
                assert_eq!(hits.len(), 1);
                assert_eq!(hits[0].embedding.qualified_name, "p.shared");
                assert!(hits[0].embedding.node_id.is_some_and(|id| id < 0));
                overlay
                    .conn()
                    .query_row("SELECT COUNT(*) FROM main.vector_embeddings", [], |row| {
                        row.get::<_, i64>(0)
                    })
                    .unwrap()
            }));
        }
        for agent in agents {
            assert_eq!(agent.join().unwrap(), 0);
        }
        assert_eq!(std::fs::read(&base_path).unwrap(), before);
    }
}
