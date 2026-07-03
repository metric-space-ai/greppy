//! Persistent vector embeddings and exact nearest-neighbor search.
//!
//! R5 intentionally starts with an exact scan. It is deterministic, easy to
//! verify, and gives the EmbeddingGemma integration a real vector-search
//! substrate before ANN/index acceleration is introduced.

use rusqlite::{params, OptionalExtension};

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
        let created_at = now_iso8601();
        let tx = self.transaction()?;
        let id = tx
            .raw()
            .prepare_cached(
                "INSERT INTO vector_embeddings
                   (project, model_id, prompt_version, task, node_id,
                    qualified_name, file_path, start_line, end_line,
                    content_sha256, graph_generation, dim, vector_norm,
                    vector, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)
                 ON CONFLICT(project, model_id, prompt_version, task, qualified_name, content_sha256)
                 DO UPDATE SET
                    node_id = excluded.node_id,
                    file_path = excluded.file_path,
                    start_line = excluded.start_line,
                    end_line = excluded.end_line,
                    graph_generation = excluded.graph_generation,
                    dim = excluded.dim,
                    vector_norm = excluded.vector_norm,
                    vector = excluded.vector,
                    created_at = excluded.created_at
                 RETURNING id",
            )?
            .query_row(
                params![
                    e.project,
                    e.model_id,
                    e.prompt_version,
                    e.task,
                    e.node_id,
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
                "SELECT id, project, model_id, prompt_version, task, node_id,
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

    /// Count embedding rows for a vector-search scope.
    pub fn count_vector_embeddings(
        &self,
        project: &str,
        model_id: &str,
        prompt_version: &str,
        task: &str,
        graph_generation: Option<u64>,
    ) -> Result<i64> {
        let generation = graph_generation.map(|g| g as i64);
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

    /// Delete all embeddings for a file. Called by node/file reindex paths so
    /// stale vectors do not survive symbol deletion.
    pub fn delete_vector_embeddings_for_file(
        &mut self,
        project: &str,
        file_path: &str,
    ) -> Result<usize> {
        let tx = self.transaction()?;
        let n = tx.raw().execute(
            "DELETE FROM vector_embeddings WHERE project = ?1 AND file_path = ?2",
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
            "DELETE FROM vector_embeddings
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
        let generation = q.graph_generation.map(|g| g as i64);
        let file = q.file_path.unwrap_or("");
        let mut stmt = self.conn().prepare_cached(
            "SELECT id, project, model_id, prompt_version, task, node_id,
                    qualified_name, file_path, start_line, end_line,
                    content_sha256, graph_generation, dim, vector_norm,
                    vector, created_at
             FROM vector_embeddings
             WHERE project = ?1
               AND model_id = ?2
               AND prompt_version = ?3
               AND task = ?4
               AND (?5 IS NULL OR graph_generation = ?5)
               AND (?6 = '' OR file_path = ?6)
             ORDER BY qualified_name, id",
        )?;
        let mut rows = stmt.query(params![
            q.project,
            q.model_id,
            q.prompt_version,
            q.task,
            generation,
            file
        ])?;
        let mut hits: Vec<VectorSearchHit> = Vec::with_capacity(q.limit);
        while let Some(row) = rows.next()? {
            let embedding = row_to_vector_embedding(row)?;
            if embedding.dim != query_vector.len() {
                return Err(Error::Invalid(format!(
                    "stored vector dimension mismatch for {}: stored {}, query {}",
                    embedding.qualified_name,
                    embedding.dim,
                    query_vector.len()
                )));
            }
            if embedding.vector_norm <= 0.0 || !embedding.vector_norm.is_finite() {
                return Err(Error::Invalid(format!(
                    "stored vector has invalid norm for {}",
                    embedding.qualified_name
                )));
            }
            let score = dot(query_vector, &embedding.vector) / (query_norm * embedding.vector_norm);
            if let Some(min) = q.min_score {
                if score < min {
                    continue;
                }
            }
            let candidate = VectorSearchHit { embedding, score };
            if hits.len() < q.limit {
                hits.push(candidate);
                continue;
            }
            let worst_idx = hits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| vector_hit_rank_cmp(a, b))
                .map(|(idx, _)| idx)
                .expect("non-empty hits when limit > 0");
            if vector_hit_rank_cmp(&candidate, &hits[worst_idx]).is_lt() {
                hits[worst_idx] = candidate;
            }
        }
        hits.sort_by(vector_hit_rank_cmp);
        Ok(hits)
    }
}

fn vector_hit_rank_cmp(a: &VectorSearchHit, b: &VectorSearchHit) -> std::cmp::Ordering {
    b.score
        .total_cmp(&a.score)
        .then_with(|| a.embedding.qualified_name.cmp(&b.embedding.qualified_name))
        .then_with(|| a.embedding.id.cmp(&b.embedding.id))
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
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn encode_f32_le(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * std::mem::size_of::<f32>());
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
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
    let dim_i64: i64 = row.get(12)?;
    let dim = usize::try_from(dim_i64).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(12, rusqlite::types::Type::Integer, Box::new(e))
    })?;
    let qname: String = row.get(6)?;
    let blob: Vec<u8> = row.get(14)?;
    let vector = decode_f32_le(&blob, dim, &qname)?;
    Ok(VectorEmbedding {
        id: row.get(0)?,
        project: row.get(1)?,
        model_id: row.get(2)?,
        prompt_version: row.get(3)?,
        task: row.get(4)?,
        node_id: row.get(5)?,
        qualified_name: qname,
        file_path: row.get(7)?,
        start_line: row.get(8)?,
        end_line: row.get(9)?,
        content_sha256: row.get(10)?,
        graph_generation: row.get::<_, i64>(11)? as u64,
        dim,
        vector_norm: row.get::<_, f64>(13)? as f32,
        vector,
        created_at: row.get(15)?,
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
            prompt_version: "embeddinggemma-code-retrieval-st-v1".into(),
            task: "retrieval_document".into(),
            node_id,
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
            prompt_version: "embeddinggemma-code-retrieval-st-v1",
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
        assert_eq!(got.dim, 3);
        assert_eq!(got.vector, vec![1.0, 0.0, 0.0]);
        assert_eq!(
            s.count_vector_embeddings(
                "p",
                "google/embeddinggemma-300m-q4",
                "embeddinggemma-code-retrieval-st-v1",
                "retrieval_document",
                Some(7)
            )
            .unwrap(),
            1
        );
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
                "embeddinggemma-code-retrieval-st-v1",
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
                "embeddinggemma-code-retrieval-st-v1",
                "retrieval_document",
                None
            )
            .unwrap(),
            0
        );
    }
}
