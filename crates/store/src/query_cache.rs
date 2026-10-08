//! Query-embedding cache: skip EmbeddingGemma inference entirely for
//! repeated fuzzy queries.
//!
//! The cache is a SMALL standalone SQLite database (`query_cache.db`)
//! that lives in the same per-workspace store directory as `graph.db`
//! (so it respects `GREPPY_STORE_DIR`), deliberately NOT a table in
//! `graph.db` itself:
//!
//! * query commands open `graph.db` READ-ONLY by design (skipping
//!   `migrate()` and the O(db-size) `integrity_check` was the fix for
//!   multi-second query opens on large repos) — a cache write from the
//!   query path would need a read-write open and re-pay all of that;
//! * writers to `graph.db` must hold the crash-safe advisory lock; a
//!   `semantic` query must never contend with a running indexer;
//! * `greppy index` publishes a brand-new `graph.db` via atomic
//!   rename, which would discard in-DB cache rows on every re-index —
//!   query embeddings depend only on (model, query), not on the graph
//!   generation, so they should survive re-indexing.
//!
//! Keying: `model_key` is built by the caller from the logical model id,
//! prompt version, task profile and a content digest of the
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
pub const QUERY_CACHE_MAX_ENTRIES: i64 = 10_000;
const QUERY_CACHE_TRIM_ENTRIES: i64 = 8_000;

/// Standalone query-embedding cache connection.
#[derive(Debug)]
pub struct QueryEmbeddingCache {
    conn: Connection,
}

impl QueryEmbeddingCache {
    /// Open (creating if needed) the cache DB in `store_dir`.
    pub fn open(store_dir: &Path) -> Result<Self> {
        greppy_core::workspace::ensure_store_dir(store_dir)
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
                last_accessed INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (model_key, query_text)
            );",
        )
        .map_err(|e| Error::Store(format!("create query cache schema: {e}")))?;
        // Upgrade standalone caches created before the bounded-LRU schema.
        if !column_exists(&conn, "query_embeddings", "last_accessed")? {
            conn.execute(
                "ALTER TABLE query_embeddings ADD COLUMN last_accessed INTEGER NOT NULL DEFAULT 0",
                [],
            )
            .map_err(|e| Error::Store(format!("upgrade query cache schema: {e}")))?;
        }
        greppy_core::workspace::ensure_db_mode(&path)
            .map_err(|e| Error::Store(format!("set query cache mode: {e}")))?;
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
        let _ = self.conn.execute(
            "UPDATE query_embeddings SET last_accessed = ?3
             WHERE model_key = ?1 AND query_text = ?2",
            rusqlite::params![model_key, query_text, unix_now_secs()],
        );
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
        let mut blob = Vec::with_capacity(std::mem::size_of_val(vector));
        for x in vector {
            blob.extend_from_slice(&x.to_le_bytes());
        }
        self.conn
            .execute(
                "INSERT OR REPLACE INTO query_embeddings
                 (model_key, query_text, dim, vector, created_at, last_accessed)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                rusqlite::params![
                    model_key,
                    query_text,
                    vector.len() as i64,
                    blob,
                    crate::workspace_state::now_iso8601(),
                    unix_now_secs(),
                ],
            )
            .map_err(|e| Error::Store(format!("query cache put: {e}")))?;
        self.prune_to_budget()
    }

    /// Bound an actively-used workspace's query cache independently of whole-
    /// store TTL/LRU eviction. Deletes least-recently-used rows and compacts
    /// only when a configured limit is exceeded.
    pub fn prune_to_budget(&self) -> Result<()> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM query_embeddings", [], |r| r.get(0))
            .map_err(|e| Error::Store(format!("query cache count: {e}")))?;
        let page_count: i64 = self
            .conn
            .query_row("PRAGMA page_count", [], |r| r.get(0))
            .unwrap_or(0);
        let page_size: i64 = self
            .conn
            .query_row("PRAGMA page_size", [], |r| r.get(0))
            .unwrap_or(4096);
        let bytes = page_count.saturating_mul(page_size);
        let max_mib = std::env::var("GREPPY_QUERY_CACHE_MAX_MIB")
            .ok()
            .and_then(|v| v.trim().parse::<i64>().ok())
            .unwrap_or(greppy_core::cache::DEFAULT_QUERY_CACHE_MAX_MIB as i64);
        let max_bytes = max_mib.saturating_mul(1024 * 1024);
        let over_entries = count > QUERY_CACHE_MAX_ENTRIES;
        let over_bytes = max_bytes > 0 && bytes > max_bytes;
        if !over_entries && !over_bytes {
            return Ok(());
        }
        let mut keep = QUERY_CACHE_TRIM_ENTRIES.min(count);
        if over_bytes && bytes > 0 {
            let byte_target = count
                .saturating_mul(max_bytes.saturating_mul(8) / 10)
                .checked_div(bytes)
                .unwrap_or(0);
            keep = keep.min(byte_target.max(1));
        }
        let remove = count.saturating_sub(keep);
        if remove > 0 {
            self.conn
                .execute(
                    "DELETE FROM query_embeddings WHERE rowid IN (
                        SELECT rowid FROM query_embeddings
                        ORDER BY last_accessed ASC, created_at ASC, rowid ASC
                        LIMIT ?1
                    )",
                    rusqlite::params![remove],
                )
                .map_err(|e| Error::Store(format!("prune query cache: {e}")))?;
            // VACUUM is intentionally only paid after crossing the hard cap.
            let _ = self.conn.execute_batch("VACUUM");
        }
        Ok(())
    }
}

fn column_exists(conn: &Connection, table: &str, column: &str) -> Result<bool> {
    let sql = format!("PRAGMA table_info({table})");
    let mut stmt = conn
        .prepare(&sql)
        .map_err(|e| Error::Store(format!("inspect query cache schema: {e}")))?;
    let names = stmt
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(|e| Error::Store(format!("inspect query cache columns: {e}")))?;
    for name in names {
        if name.map_err(Error::Sqlite)? == column {
            return Ok(true);
        }
    }
    Ok(false)
}

fn unix_now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
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
            "greppy-querycache-{}-{}",
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
        assert_eq!(
            cache.get("model-a", "reverse linked list").unwrap(),
            Some(v)
        );
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

    #[test]
    fn byte_budget_prunes_least_recently_used_rows() {
        let dir = tmp_dir();
        std::env::set_var("GREPPY_QUERY_CACHE_MAX_MIB", "1");
        let cache = QueryEmbeddingCache::open(&dir).unwrap();
        let vector = vec![0.25f32; 4096];
        for index in 0..100 {
            cache
                .put("model", &format!("query-{index:03}"), &vector)
                .unwrap();
        }
        let count: i64 = cache
            .conn
            .query_row("SELECT COUNT(*) FROM query_embeddings", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert!(count < 100, "byte cap must evict old entries");
        assert!(cache.get("model", "query-099").unwrap().is_some());
        std::env::remove_var("GREPPY_QUERY_CACHE_MAX_MIB");
        drop(cache);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[cfg(unix)]
    #[test]
    fn open_sets_directory_0700_and_database_0600() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tmp_dir();
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();
        let cache = QueryEmbeddingCache::open(&dir).unwrap();
        drop(cache);
        let mode = |path: &Path| std::fs::metadata(path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(&dir), 0o700);
        assert_eq!(mode(&dir.join(QUERY_CACHE_DB_FILE)), 0o600);
        std::fs::remove_dir_all(&dir).ok();
    }
}
