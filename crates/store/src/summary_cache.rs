//! Purpose-summary cache: skip Qwen3.5 generation entirely for repeated
//! unchanged source spans.
//!
//! The cache is a SMALL standalone SQLite database (`summary_cache.db`)
//! that lives in the same per-workspace store directory as `graph.db`
//! (so it respects `GREPPY_STORE_DIR`), deliberately NOT a table in
//! `graph.db` itself:
//!
//! * query commands open `graph.db` READ-ONLY by design, while summary cache
//!   hits update LRU state and misses populate the cache;
//! * writers to `graph.db` must hold the crash-safe advisory lock; summary
//!   generation must never contend with a running indexer;
//! * `greppy index` publishes a brand-new `graph.db` via atomic rename, which
//!   would discard in-DB cache rows on every re-index — summaries depend only
//!   on (model, file path, source span), not on the graph generation, so they
//!   should survive re-indexing.
//!
//! All operations are best-effort from the caller's perspective: cache
//! failures must never fail a command, so the CLI treats every error here as
//! a cache miss.

use std::path::{Path, PathBuf};

use rusqlite::{Connection, OptionalExtension};
use sha2::{Digest, Sha256};

use crate::store_error::{Error, Result};

/// File name of the cache database inside the workspace store dir.
pub const SUMMARY_CACHE_DB_FILE: &str = "summary_cache.db";
pub const SUMMARY_CACHE_MAX_ENTRIES: i64 = 10_000;
const SUMMARY_CACHE_TRIM_ENTRIES: i64 = 8_000;

/// Standalone purpose-summary cache connection.
#[derive(Debug)]
pub struct SummaryCache {
    conn: Connection,
}

impl SummaryCache {
    /// Open (creating if needed) the cache DB in `store_dir`.
    pub fn open(store_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(store_dir)
            .map_err(|e| Error::Store(format!("create store dir for summary cache: {e}")))?;
        let path: PathBuf = store_dir.join(SUMMARY_CACHE_DB_FILE);
        let conn = Connection::open(&path)
            .map_err(|e| Error::Store(format!("open summary cache {}: {e}", path.display())))?;
        // Single-shot CLI: contention is rare and losing a cache write is
        // fine — keep the timeout short so the cache can never stall a
        // command noticeably.
        conn.busy_timeout(std::time::Duration::from_millis(200))
            .map_err(|e| Error::Store(format!("summary cache busy_timeout: {e}")))?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS summaries (
                model_key  TEXT    NOT NULL,
                span_hash  TEXT    NOT NULL,
                bullets    TEXT    NOT NULL,
                created_at TEXT    NOT NULL,
                last_accessed INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (model_key, span_hash)
            );",
        )
        .map_err(|e| Error::Store(format!("create summary cache schema: {e}")))?;
        Ok(Self { conn })
    }

    /// Look up a cached purpose summary.
    pub fn get(&self, model_key: &str, span_hash: &str) -> Result<Option<Vec<String>>> {
        let row: Option<String> = self
            .conn
            .query_row(
                "SELECT bullets FROM summaries
                 WHERE model_key = ?1 AND span_hash = ?2",
                rusqlite::params![model_key, span_hash],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| Error::Store(format!("summary cache get: {e}")))?;
        let Some(bullets) = row else {
            return Ok(None);
        };
        let _ = self.conn.execute(
            "UPDATE summaries SET last_accessed = ?3
             WHERE model_key = ?1 AND span_hash = ?2",
            rusqlite::params![model_key, span_hash, unix_now_secs()],
        );
        serde_json::from_str(&bullets)
            .map(Some)
            .map_err(|e| Error::Store(format!("summary cache decode: {e}")))
    }

    /// Insert or replace a cached purpose summary. Empty summaries are not
    /// useful cache entries and are deliberately ignored.
    pub fn put(&self, model_key: &str, span_hash: &str, bullets: &[String]) -> Result<()> {
        if bullets.is_empty() {
            return Ok(());
        }
        let bullets = serde_json::to_string(bullets)
            .map_err(|e| Error::Store(format!("summary cache encode: {e}")))?;
        self.conn
            .execute(
                "INSERT OR REPLACE INTO summaries
                 (model_key, span_hash, bullets, created_at, last_accessed)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![
                    model_key,
                    span_hash,
                    bullets,
                    crate::workspace_state::now_iso8601(),
                    unix_now_secs(),
                ],
            )
            .map_err(|e| Error::Store(format!("summary cache put: {e}")))?;
        self.prune_to_budget()
    }

    /// Bound an actively-used workspace's summary cache independently of whole-
    /// store TTL/LRU eviction. Deletes least-recently-used rows and compacts
    /// only when a configured limit is exceeded.
    pub fn prune_to_budget(&self) -> Result<()> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM summaries", [], |r| r.get(0))
            .map_err(|e| Error::Store(format!("summary cache count: {e}")))?;
        let page_count: i64 = self
            .conn
            .query_row("PRAGMA page_count", [], |r| r.get(0))
            .unwrap_or(0);
        let page_size: i64 = self
            .conn
            .query_row("PRAGMA page_size", [], |r| r.get(0))
            .unwrap_or(4096);
        let bytes = page_count.saturating_mul(page_size);
        let max_mib = std::env::var("GREPPY_SUMMARY_CACHE_MAX_MIB")
            .ok()
            .and_then(|v| v.trim().parse::<i64>().ok())
            .unwrap_or(greppy_core::cache::DEFAULT_SUMMARY_CACHE_MAX_MIB as i64);
        let max_bytes = max_mib.saturating_mul(1024 * 1024);
        let over_entries = count > SUMMARY_CACHE_MAX_ENTRIES;
        let over_bytes = max_bytes > 0 && bytes > max_bytes;
        if !over_entries && !over_bytes {
            return Ok(());
        }
        let mut keep = SUMMARY_CACHE_TRIM_ENTRIES.min(count);
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
                    "DELETE FROM summaries WHERE rowid IN (
                        SELECT rowid FROM summaries
                        ORDER BY last_accessed ASC, created_at ASC, rowid ASC
                        LIMIT ?1
                    )",
                    rusqlite::params![remove],
                )
                .map_err(|e| Error::Store(format!("prune summary cache: {e}")))?;
            // VACUUM is intentionally only paid after crossing the hard cap.
            let _ = self.conn.execute_batch("VACUUM");
        }
        Ok(())
    }
}

fn unix_now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Hash a summary input using its prompt-visible file path and source bytes.
pub fn span_hash(file_path: &str, source: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(file_path.as_bytes());
    hasher.update([0]);
    hasher.update(source.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "greppy-summarycache-{}-{}",
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
        let cache = SummaryCache::open(&dir).unwrap();
        let bullets = vec!["Parses the request.".into(), "Returns a response.".into()];
        cache.put("model-a", "span-a", &bullets).unwrap();
        assert_eq!(
            cache.get("model-a", "span-a").unwrap(),
            Some(bullets.clone())
        );
        // Different model key or span hash misses.
        assert_eq!(cache.get("model-b", "span-a").unwrap(), None);
        assert_eq!(cache.get("model-a", "span-b").unwrap(), None);
        // Persistence across re-open.
        drop(cache);
        let cache = SummaryCache::open(&dir).unwrap();
        assert_eq!(cache.get("model-a", "span-a").unwrap(), Some(bullets));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn put_replaces_existing_row() {
        let dir = tmp_dir();
        let cache = SummaryCache::open(&dir).unwrap();
        cache.put("m", "s", &["old".into()]).unwrap();
        cache
            .put("m", "s", &["new one".into(), "new two".into()])
            .unwrap();
        assert_eq!(
            cache.get("m", "s").unwrap(),
            Some(vec!["new one".into(), "new two".into()])
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn byte_budget_prunes_least_recently_used_rows() {
        let dir = tmp_dir();
        std::env::set_var("GREPPY_SUMMARY_CACHE_MAX_MIB", "1");
        let cache = SummaryCache::open(&dir).unwrap();
        let bullet = "purpose ".repeat(2048);
        for index in 0..100 {
            cache
                .put(
                    "model",
                    &format!("span-{index:03}"),
                    std::slice::from_ref(&bullet),
                )
                .unwrap();
        }
        let count: i64 = cache
            .conn
            .query_row("SELECT COUNT(*) FROM summaries", [], |row| row.get(0))
            .unwrap();
        assert!(count < 100, "byte cap must evict old entries");
        assert!(cache.get("model", "span-099").unwrap().is_some());
        std::env::remove_var("GREPPY_SUMMARY_CACHE_MAX_MIB");
        drop(cache);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn span_hash_is_deterministic_and_path_sensitive() {
        let source = "fn answer() -> u32 { 42 }";
        let first = span_hash("src/lib.rs", source);
        assert_eq!(first, span_hash("src/lib.rs", source));
        assert_ne!(first, span_hash("src/other.rs", source));
        assert_eq!(first.len(), 64);
        assert!(first
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()));
    }
}
