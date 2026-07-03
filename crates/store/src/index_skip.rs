//! Persistent skipped-file metadata.
//!
//! R3 diagnostics require more than aggregate counters: a review harness must
//! be able to see which files were skipped and why.

use rusqlite::{params, OptionalExtension};
use serde::{Deserialize, Serialize};

use crate::store::Store;
use crate::store_error::Result;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexSkip {
    pub project: String,
    pub rel_path: String,
    pub language: String,
    pub reason: String,
    pub detail: String,
    pub size: i64,
    pub mtime_ns: i64,
    pub last_indexed_generation: u64,
    pub updated_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexSkipReasonCount {
    pub reason: String,
    pub count: i64,
}

impl Store {
    pub fn upsert_index_skip(&mut self, skip: &IndexSkip) -> Result<()> {
        let tx = self.transaction()?;
        tx.raw().execute(
            "INSERT INTO index_skips
                  (project, rel_path, language, reason, detail, size, mtime_ns,
                   last_indexed_generation, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
                 ON CONFLICT(project, rel_path) DO UPDATE SET
                   language = excluded.language,
                   reason = excluded.reason,
                   detail = excluded.detail,
                   size = excluded.size,
                   mtime_ns = excluded.mtime_ns,
                   last_indexed_generation = excluded.last_indexed_generation,
                   updated_at = excluded.updated_at",
            params![
                skip.project,
                skip.rel_path,
                skip.language,
                skip.reason,
                skip.detail,
                skip.size,
                skip.mtime_ns,
                skip.last_indexed_generation as i64,
                skip.updated_at,
            ],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn get_index_skip(&self, project: &str, rel_path: &str) -> Result<Option<IndexSkip>> {
        if !self.index_skips_table_exists()? {
            return Ok(None);
        }
        let row = self
            .conn()
            .query_row(
                "SELECT project, rel_path, language, reason, detail, size, mtime_ns,
                        last_indexed_generation, updated_at
                 FROM index_skips WHERE project = ?1 AND rel_path = ?2",
                params![project, rel_path],
                row_to_index_skip,
            )
            .optional()?;
        Ok(row)
    }

    pub fn delete_index_skip(&mut self, project: &str, rel_path: &str) -> Result<()> {
        if !self.index_skips_table_exists()? {
            return Ok(());
        }
        let tx = self.transaction()?;
        tx.raw().execute(
            "DELETE FROM index_skips WHERE project = ?1 AND rel_path = ?2",
            params![project, rel_path],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn list_index_skips(&self, project: &str) -> Result<Vec<IndexSkip>> {
        if !self.index_skips_table_exists()? {
            return Ok(Vec::new());
        }
        let mut stmt = self.conn().prepare(
            "SELECT project, rel_path, language, reason, detail, size, mtime_ns,
                    last_indexed_generation, updated_at
             FROM index_skips
             WHERE project = ?1
             ORDER BY rel_path",
        )?;
        let rows = stmt
            .query_map(params![project], row_to_index_skip)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    pub fn index_skip_counts_by_reason(&self, project: &str) -> Result<Vec<IndexSkipReasonCount>> {
        if !self.index_skips_table_exists()? {
            return Ok(Vec::new());
        }
        let mut stmt = self.conn().prepare(
            "SELECT reason, COUNT(*)
             FROM index_skips
             WHERE project = ?1
             GROUP BY reason
             ORDER BY reason",
        )?;
        let rows = stmt
            .query_map(params![project], |row| {
                Ok(IndexSkipReasonCount {
                    reason: row.get(0)?,
                    count: row.get(1)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    fn index_skips_table_exists(&self) -> Result<bool> {
        let n: i64 = self.conn().query_row(
            "SELECT COUNT(*) FROM sqlite_master
             WHERE type='table' AND name='index_skips'",
            [],
            |row| row.get(0),
        )?;
        Ok(n > 0)
    }
}

fn row_to_index_skip(row: &rusqlite::Row<'_>) -> rusqlite::Result<IndexSkip> {
    Ok(IndexSkip {
        project: row.get(0)?,
        rel_path: row.get(1)?,
        language: row.get(2)?,
        reason: row.get(3)?,
        detail: row.get(4)?,
        size: row.get(5)?,
        mtime_ns: row.get(6)?,
        last_indexed_generation: row.get::<_, i64>(7).unwrap_or(0) as u64,
        updated_at: row.get(8)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{workspace_state as ws, Project};

    fn store_with_project() -> Store {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "x".into(),
            root_path: "/p".into(),
        })
        .unwrap();
        s
    }

    fn skip(rel_path: &str, reason: &str) -> IndexSkip {
        IndexSkip {
            project: "p".into(),
            rel_path: rel_path.into(),
            language: "rust".into(),
            reason: reason.into(),
            detail: "test detail".into(),
            size: 123,
            mtime_ns: 456,
            last_indexed_generation: 7,
            updated_at: ws::now_iso8601(),
        }
    }

    #[test]
    fn index_skip_round_trips_and_counts_by_reason() {
        let mut s = store_with_project();
        s.upsert_index_skip(&skip("a.rs", "oversize")).unwrap();
        s.upsert_index_skip(&skip("b.rs", "oversize")).unwrap();
        s.upsert_index_skip(&skip("c.txt", "unsupported_language"))
            .unwrap();

        let got = s.get_index_skip("p", "a.rs").unwrap().unwrap();
        assert_eq!(got.reason, "oversize");
        assert_eq!(got.size, 123);

        let rows = s.list_index_skips("p").unwrap();
        assert_eq!(
            rows.iter().map(|s| s.rel_path.as_str()).collect::<Vec<_>>(),
            vec!["a.rs", "b.rs", "c.txt"]
        );
        assert_eq!(
            s.index_skip_counts_by_reason("p").unwrap(),
            vec![
                IndexSkipReasonCount {
                    reason: "oversize".into(),
                    count: 2,
                },
                IndexSkipReasonCount {
                    reason: "unsupported_language".into(),
                    count: 1,
                },
            ]
        );
    }

    #[test]
    fn delete_index_skip_clears_stale_reason() {
        let mut s = store_with_project();
        s.upsert_index_skip(&skip("a.rs", "oversize")).unwrap();
        s.delete_index_skip("p", "a.rs").unwrap();
        assert!(s.get_index_skip("p", "a.rs").unwrap().is_none());
    }
}
