//! User-global, content-addressed document embedding cache.
//!
//! Workspace graph stores remain isolated, but an identical model prompt must
//! not consume embedding compute once per worktree. This standalone SQLite-WAL
//! cache is keyed by the exact prompt input plus the complete model contract.

use std::path::PathBuf;

use rusqlite::{params, Connection, OptionalExtension};
use sha2::{Digest, Sha256};

use crate::store_error::{Error, Result};

const CACHE_DB_FILE: &str = "document-embeddings.db";
const DEFAULT_MAX_ENTRIES: i64 = 100_000;
const TRIM_TARGET_NUMERATOR: i64 = 9;
const TRIM_TARGET_DENOMINATOR: i64 = 10;

#[derive(Debug, Clone, PartialEq)]
pub struct CachedDocumentEmbedding {
    pub token_len: usize,
    pub vector: Option<Vec<f32>>,
}

#[derive(Debug)]
pub struct EmbeddingContentCache {
    conn: Connection,
}

impl EmbeddingContentCache {
    pub fn open_global() -> Result<Self> {
        let directory = greppy_core::cache::inference_cache_root();
        std::fs::create_dir_all(&directory).map_err(|error| {
            Error::Store(format!(
                "create global inference cache {}: {error}",
                directory.display()
            ))
        })?;
        Self::open(directory)
    }

    /// Open an isolated cache in `directory`; primarily useful for hermetic
    /// providers and integration tests.
    pub fn open(directory: impl Into<PathBuf>) -> Result<Self> {
        let directory = directory.into();
        std::fs::create_dir_all(&directory).map_err(|error| {
            Error::Store(format!(
                "create embedding cache directory {}: {error}",
                directory.display()
            ))
        })?;
        let path = directory.join(CACHE_DB_FILE);
        let conn = Connection::open(&path).map_err(|error| {
            Error::Store(format!(
                "open global embedding cache {}: {error}",
                path.display()
            ))
        })?;
        conn.busy_timeout(std::time::Duration::from_secs(2))
            .map_err(|error| Error::Store(format!("embedding cache busy_timeout: {error}")))?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             CREATE TABLE IF NOT EXISTS document_embeddings (
                model_id TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                task_profile TEXT NOT NULL,
                input_sha256 TEXT NOT NULL,
                token_len INTEGER NOT NULL,
                vector BLOB,
                vector_dim INTEGER,
                created_at INTEGER NOT NULL,
                last_accessed INTEGER NOT NULL,
                PRIMARY KEY(model_id, prompt_version, task_profile, input_sha256)
             );
             CREATE INDEX IF NOT EXISTS document_embeddings_lru
             ON document_embeddings(last_accessed);",
        )
        .map_err(|error| Error::Store(format!("create embedding cache schema: {error}")))?;
        Ok(Self { conn })
    }

    pub fn input_sha256(prompt_input: &str) -> String {
        format!("{:x}", Sha256::digest(prompt_input.as_bytes()))
    }

    pub fn get(
        &self,
        model_id: &str,
        prompt_version: &str,
        task_profile: &str,
        input_sha256: &str,
    ) -> Result<Option<CachedDocumentEmbedding>> {
        let row = self
            .conn
            .query_row(
                "SELECT token_len, vector, vector_dim
                 FROM document_embeddings
                 WHERE model_id=?1 AND prompt_version=?2 AND task_profile=?3 AND input_sha256=?4",
                params![model_id, prompt_version, task_profile, input_sha256],
                |row| {
                    let token_len: i64 = row.get(0)?;
                    let blob: Option<Vec<u8>> = row.get(1)?;
                    let dim: Option<i64> = row.get(2)?;
                    Ok((token_len, blob, dim))
                },
            )
            .optional()
            .map_err(|error| Error::Store(format!("embedding cache get: {error}")))?;
        let Some((token_len, blob, dim)) = row else {
            return Ok(None);
        };
        let vector = match (blob, dim) {
            (Some(blob), Some(dim)) => Some(decode_vector(&blob, dim)?),
            (None, None) => None,
            _ => return Err(Error::Store("embedding cache row is inconsistent".into())),
        };
        let _ = self.conn.execute(
            "UPDATE document_embeddings SET last_accessed=?5
             WHERE model_id=?1 AND prompt_version=?2 AND task_profile=?3 AND input_sha256=?4",
            params![
                model_id,
                prompt_version,
                task_profile,
                input_sha256,
                unix_now_secs()
            ],
        );
        Ok(Some(CachedDocumentEmbedding {
            token_len: usize::try_from(token_len)
                .map_err(|_| Error::Store("embedding cache token length is invalid".into()))?,
            vector,
        }))
    }

    pub fn put_token_len(
        &self,
        model_id: &str,
        prompt_version: &str,
        task_profile: &str,
        input_sha256: &str,
        token_len: usize,
    ) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO document_embeddings
                 (model_id,prompt_version,task_profile,input_sha256,token_len,vector,vector_dim,created_at,last_accessed)
                 VALUES(?1,?2,?3,?4,?5,NULL,NULL,?6,?6)
                 ON CONFLICT(model_id,prompt_version,task_profile,input_sha256)
                 DO UPDATE SET token_len=excluded.token_len,last_accessed=excluded.last_accessed",
                params![
                    model_id,
                    prompt_version,
                    task_profile,
                    input_sha256,
                    i64::try_from(token_len).unwrap_or(i64::MAX),
                    unix_now_secs()
                ],
            )
            .map_err(|error| Error::Store(format!("embedding cache put token length: {error}")))?;
        Ok(())
    }

    pub fn put_vector(
        &self,
        model_id: &str,
        prompt_version: &str,
        task_profile: &str,
        input_sha256: &str,
        token_len: usize,
        vector: &[f32],
    ) -> Result<()> {
        if vector.is_empty() {
            return Err(Error::Store("refuse to cache an empty embedding".into()));
        }
        let blob = vector
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let now = unix_now_secs();
        self.conn
            .execute(
                "INSERT OR REPLACE INTO document_embeddings
                 (model_id,prompt_version,task_profile,input_sha256,token_len,vector,vector_dim,created_at,last_accessed)
                 VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?8)",
                params![
                    model_id,
                    prompt_version,
                    task_profile,
                    input_sha256,
                    i64::try_from(token_len).unwrap_or(i64::MAX),
                    blob,
                    i64::try_from(vector.len()).unwrap_or(i64::MAX),
                    now
                ],
            )
            .map_err(|error| Error::Store(format!("embedding cache put vector: {error}")))?;
        self.prune_to_budget()
    }

    fn prune_to_budget(&self) -> Result<()> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static PUTS_SINCE_PROCESS_START: AtomicUsize = AtomicUsize::new(0);
        if !PUTS_SINCE_PROCESS_START
            .fetch_add(1, Ordering::Relaxed)
            .is_multiple_of(256)
        {
            return Ok(());
        }
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM document_embeddings", [], |row| {
                row.get(0)
            })
            .map_err(|error| Error::Store(format!("embedding cache count: {error}")))?;
        let page_count: i64 = self
            .conn
            .query_row("PRAGMA page_count", [], |row| row.get(0))
            .unwrap_or(0);
        let page_size: i64 = self
            .conn
            .query_row("PRAGMA page_size", [], |row| row.get(0))
            .unwrap_or(4096);
        let bytes = page_count.saturating_mul(page_size);
        let max_mib = std::env::var("GREPPY_EMBEDDING_CACHE_MAX_MIB")
            .ok()
            .and_then(|value| value.trim().parse::<i64>().ok())
            .unwrap_or(greppy_core::cache::DEFAULT_EMBEDDING_CACHE_MAX_MIB as i64);
        let max_bytes = max_mib.saturating_mul(1024 * 1024);
        if count <= DEFAULT_MAX_ENTRIES && (max_bytes <= 0 || bytes <= max_bytes) {
            return Ok(());
        }
        let by_entries = DEFAULT_MAX_ENTRIES * TRIM_TARGET_NUMERATOR / TRIM_TARGET_DENOMINATOR;
        let by_bytes = if bytes > 0 && max_bytes > 0 {
            count
                .saturating_mul(max_bytes)
                .saturating_mul(TRIM_TARGET_NUMERATOR)
                / bytes
                / TRIM_TARGET_DENOMINATOR
        } else {
            by_entries
        };
        let keep = by_entries.min(by_bytes).max(1);
        self.conn
            .execute(
                "DELETE FROM document_embeddings WHERE rowid IN (
                    SELECT rowid FROM document_embeddings
                    ORDER BY last_accessed ASC, rowid ASC LIMIT ?1
                 )",
                params![count.saturating_sub(keep)],
            )
            .map_err(|error| Error::Store(format!("prune embedding cache: {error}")))?;
        Ok(())
    }
}

fn decode_vector(blob: &[u8], dim: i64) -> Result<Vec<f32>> {
    let dim = usize::try_from(dim)
        .map_err(|_| Error::Store("embedding cache vector dimension is invalid".into()))?;
    if blob.len() != dim.saturating_mul(4) {
        return Err(Error::Store(
            "embedding cache vector length is invalid".into(),
        ));
    }
    Ok(blob
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn unix_now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .try_into()
        .unwrap_or(i64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_contract_and_prompt_hash_control_reuse() {
        let temp = tempfile::tempdir().unwrap();
        let cache = EmbeddingContentCache::open(temp.path()).unwrap();
        let input = EmbeddingContentCache::input_sha256("title\ncontent");
        cache.put_token_len("m", "p", "t", &input, 7).unwrap();
        assert_eq!(
            cache.get("m", "p", "t", &input).unwrap().unwrap().token_len,
            7
        );
        cache
            .put_vector("m", "p", "t", &input, 7, &[1.0, -2.5])
            .unwrap();
        assert_eq!(
            cache.get("m", "p", "t", &input).unwrap().unwrap().vector,
            Some(vec![1.0, -2.5])
        );
        assert!(cache.get("other", "p", "t", &input).unwrap().is_none());
    }
}
