//! Query-embedding cache: skip EmbeddingGemma inference entirely for
//! repeated fuzzy queries.
//!
//! The cache is a SMALL standalone SQLite database (`query_cache.db`)
//! that lives in the same per-workspace store directory as `graph.db`
//! (so it respects `GREPPLUS_STORE_DIR`), deliberately NOT a table in
//! `graph.db` itself:
//!
//! * query commands open `graph.db` READ-ONLY by design (skipping
//!   `migrate()` and the O(db-size) `integrity_check` was the fix for
//!   multi-second query opens on large repos) — a cache write from the
//!   query path would need a read-write open and re-pay all of that;
//! * writers to `graph.db` must hold the crash-safe advisory lock; a
//!   `semantic` query must never contend with a running indexer;
//! * `grepplus index` publishes a brand-new `graph.db` via atomic
//!   rename, which would discard in-DB cache rows on every re-index —
//!   query embeddings depend only on (model, query), not on the graph
//!   generation, so they should survive re-indexing.
//!
//! Keying: `model_key` is built by the caller from the logical model id,
//! prompt version, task profile and a fingerprint (len+mtime) of the
//! model source files, so swapping the GGUF/tokenizer invalidates cached
//! vectors. `query_text` is the normalized query (see
//! [`normalize_query_text`]).
//!
//! All operations are best-effort from the caller's perspective: cache
//! failures must never fail a search, so the CLI treats every error here
//! as a cache miss.

use std::path::{Path, PathBuf};

use rusqlite::{Connection, OptionalExtension};

use crate::store_error::{Error, Result};

/// File name of the cache database inside the workspace store dir.
pub const QUERY_CACHE_DB_FILE: &str = "query_cache.db";

/// Standalone query-embedding cache connection.
#[derive(Debug)]
pub struct QueryEmbeddingCache {
    conn: Connection,
}

impl QueryEmbeddingCache {
    /// Open (creating if needed) the cache DB in `store_dir`.
    pub fn open(store_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(store_dir)
            .map_err(|e| Error::Store(format!("create store dir for query cache: {e}")))?;
        let path: PathBuf = store_dir.join(QUERY_CACHE_DB_FILE);
        let conn = Connection::open(&path)
            .map_err(|e| Error::Store(format!("open query cache {}: {e}", path.display())))?;
        // Single-shot CLI: contention is rare and losing a cache write is
        // fine — keep the timeout short so the cache can never stall a
        // query noticeably.
        conn.busy_timeout(std::time::Duration::from_millis(200))
            .map_err(|e| Error::Store(format!("query cache busy_timeout: {e}")))?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS query_embeddings (
                model_key  TEXT    NOT NULL,
                query_text TEXT    NOT NULL,
                dim        INTEGER NOT NULL,
                vector     BLOB    NOT NULL,
                created_at TEXT    NOT NULL,
                PRIMARY KEY (model_key, query_text)
            );",
        )
        .map_err(|e| Error::Store(format!("create query cache schema: {e}")))?;
        Ok(Self { conn })
    }

    /// Look up a cached embedding.
    pub fn get(&self, model_key: &str, query_text: &str) -> Result<Option<Vec<f32>>> {
        let row: Option<(i64, Vec<u8>)> = self
            .conn
            .query_row(
                "SELECT dim, vector FROM query_embeddings
                 WHERE model_key = ?1 AND query_text = ?2",
                rusqlite::params![model_key, query_text],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .map_err(|e| Error::Store(format!("query cache get: {e}")))?;
        let Some((dim, blob)) = row else {
            return Ok(None);
        };
        let dim = usize::try_from(dim)
            .map_err(|_| Error::Store(format!("query cache row has negative dim {dim}")))?;
        if blob.len() != dim * std::mem::size_of::<f32>() {
            return Err(Error::Store(format!(
                "query cache blob length mismatch: bytes {}, dim {dim}",
                blob.len()
            )));
        }
        let mut out = Vec::with_capacity(dim);
        for chunk in blob.chunks_exact(std::mem::size_of::<f32>()) {
            out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(Some(out))
    }

    /// Insert or replace a cached embedding.
    pub fn put(&self, model_key: &str, query_text: &str, vector: &[f32]) -> Result<()> {
        let mut blob = Vec::with_capacity(vector.len() * std::mem::size_of::<f32>());
        for x in vector {
            blob.extend_from_slice(&x.to_le_bytes());
        }
        self.conn
            .execute(
                "INSERT OR REPLACE INTO query_embeddings
                 (model_key, query_text, dim, vector, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![
                    model_key,
                    query_text,
                    vector.len() as i64,
                    blob,
                    crate::workspace_state::now_iso8601(),
                ],
            )
            .map_err(|e| Error::Store(format!("query cache put: {e}")))?;
        Ok(())
    }
}

/// Normalize a query for cache keying: trim and collapse every internal
/// whitespace run to a single space. Case is preserved — EmbeddingGemma
/// embeddings are case-sensitive, so `Foo` and `foo` are different
/// queries.
pub fn normalize_query_text(q: &str) -> String {
    let mut out = String::with_capacity(q.len());
    let mut in_ws = false;
    for c in q.trim().chars() {
        if c.is_whitespace() {
            in_ws = true;
        } else {
            if in_ws && !out.is_empty() {
                out.push(' ');
            }
            in_ws = false;
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "grepplus-querycache-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn roundtrip_and_miss() {
        let dir = tmp_dir();
        let cache = QueryEmbeddingCache::open(&dir).unwrap();
        let v = vec![0.25f32, -1.5, 3.0];
        cache.put("model-a", "reverse linked list", &v).unwrap();
        assert_eq!(
            cache.get("model-a", "reverse linked list").unwrap(),
            Some(v.clone())
        );
        // Different model key or query text misses.
        assert_eq!(cache.get("model-b", "reverse linked list").unwrap(), None);
        assert_eq!(cache.get("model-a", "reverse linked lists").unwrap(), None);
        // Persistence across re-open.
        drop(cache);
        let cache = QueryEmbeddingCache::open(&dir).unwrap();
        assert_eq!(cache.get("model-a", "reverse linked list").unwrap(), Some(v));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn put_replaces_existing_row() {
        let dir = tmp_dir();
        let cache = QueryEmbeddingCache::open(&dir).unwrap();
        cache.put("m", "q", &[1.0]).unwrap();
        cache.put("m", "q", &[2.0, 3.0]).unwrap();
        assert_eq!(cache.get("m", "q").unwrap(), Some(vec![2.0, 3.0]));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn normalize_collapses_whitespace() {
        assert_eq!(
            normalize_query_text("  reverse   linked\t\nlist "),
            "reverse linked list"
        );
        assert_eq!(normalize_query_text(""), "");
        assert_eq!(normalize_query_text("   "), "");
        assert_eq!(normalize_query_text("Foo"), "Foo");
    }
}
