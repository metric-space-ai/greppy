//! File-content FTS row CRUD (R-011 / WP-R011).
//!
//! Each row represents one indexed line of one file. Since the 0004
//! migration, `file_content_fts` is an **external-content** FTS5
//! table (`content = 'file_content'`) kept in sync by AFTER
//! INSERT/UPDATE/DELETE triggers on `file_content`. The application
//! therefore writes ONLY to `file_content`; the triggers mirror every
//! change into the FTS index (using the FTS5 `'delete'` command for
//! removals). Manually writing to `file_content_fts` would corrupt an
//! external-content index, which is exactly the bug RV-009 fixes — so
//! we never touch it directly.
//!
//! Re-indexing a file MUST be paired with `delete_for_file` first so
//! that previously-indexed lines do not persist after rename/remove
//! (R-018 / WP-R018). Because deletes fire the FTS prune trigger, the
//! mirror can never accumulate orphan rows (RV-009).

use rusqlite::params;

use crate::store::Store;
use crate::store_error::{Error, Result};

/// A single line of indexed file content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContentRow {
    pub line: u32,
    pub snippet: String,
}

impl Store {
    /// Delete every `file_content` row for `(project, rel_path)`.
    /// The matching FTS index entries are pruned automatically by the
    /// `file_content_fts_ad` AFTER DELETE trigger (migration 0004), so
    /// no FTS row is ever orphaned (RV-009). We only ever write to the
    /// content table — never to the external-content FTS mirror.
    pub fn delete_file_content(&mut self, project: &str, rel_path: &str) -> Result<usize> {
        let tx = self.transaction()?;
        let n = tx.raw().execute(
            "DELETE FROM file_content WHERE project = ?1 AND rel_path = ?2",
            params![project, rel_path],
        )?;
        tx.commit()?;
        Ok(n)
    }

    /// Insert (or upsert) the file-content rows for one file. Each
    /// row's `file_path` mirrors `rel_path` so the FTS5 external-
    /// content table can read the path column. The FTS index is kept
    /// in sync entirely by the migration-0004 triggers, so this method
    /// writes only to `file_content` — touching `file_content_fts`
    /// directly would corrupt the external-content index (RV-009).
    pub fn insert_file_content_rows(
        &mut self,
        project: &str,
        rel_path: &str,
        rows: &[ContentRow],
    ) -> Result<usize> {
        if rows.is_empty() {
            return Ok(0);
        }
        let tx = self.transaction()?;
        let mut inserted = 0usize;
        {
            // Prepare the statement ONCE and reuse it for every row. The old
            // code called `execute(sql, …)` per row, which re-parses the SQL
            // string on every one of the (up to ~20 K) content lines — that
            // re-parse was a large share of the cold-index cost (content-FTS
            // indexing measured at ~64 % of the whole index). `prepare_cached`
            // parses once and binds per row; the FTS mirror trigger still
            // fires identically, so behaviour is byte-for-byte unchanged.
            let mut stmt = tx
                .raw()
                .prepare_cached(
                    "INSERT INTO file_content
                       (project, rel_path, file_path, line, snippet)
                     VALUES (?1, ?2, ?2, ?3, ?4)
                     ON CONFLICT(project, rel_path, line) DO UPDATE SET
                       file_path = excluded.file_path,
                       snippet = excluded.snippet",
                )
                .map_err(Error::Sqlite)?;
            for row in rows {
                // The UNIQUE constraint is (project, rel_path, line), so
                // ON CONFLICT means an upsert. INSERT fires the AFTER
                // INSERT trigger; the upsert path fires AFTER UPDATE —
                // both keep the FTS mirror in lock-step.
                stmt.execute(params![project, rel_path, row.line as i64, row.snippet])
                    .map_err(Error::Sqlite)?;
                inserted += 1;
            }
        }
        tx.commit()?;
        Ok(inserted)
    }

    /// Insert content for MANY files in a SINGLE transaction.
    ///
    /// Indexing measured content-FTS as ~64% of a cold index; with one
    /// transaction (and one fsync / WAL flush) per file, a 423-file repo paid
    /// 423 commits just for content. This batches them all into one
    /// transaction, reusing one prepared statement — same rows, same FTS
    /// trigger behaviour, far fewer commits. Intended for the FULL index path,
    /// where no prior content exists to delete (the incremental path keeps the
    /// per-file delete-then-insert via [`Self::insert_file_content_rows`]).
    pub fn insert_file_content_batch(
        &mut self,
        project: &str,
        files: &[(String, Vec<ContentRow>)],
    ) -> Result<usize> {
        let tx = self.transaction()?;
        let mut inserted = 0usize;
        {
            let mut stmt = tx
                .raw()
                .prepare_cached(
                    "INSERT INTO file_content
                       (project, rel_path, file_path, line, snippet)
                     VALUES (?1, ?2, ?2, ?3, ?4)
                     ON CONFLICT(project, rel_path, line) DO UPDATE SET
                       file_path = excluded.file_path,
                       snippet = excluded.snippet",
                )
                .map_err(Error::Sqlite)?;
            for (rel_path, rows) in files {
                for row in rows {
                    stmt.execute(params![project, rel_path, row.line as i64, row.snippet])
                        .map_err(Error::Sqlite)?;
                    inserted += 1;
                }
            }
        }
        tx.commit()?;
        Ok(inserted)
    }

    /// Look up the file content row for a given rowid. Used by
    /// `search_file_content` to recover the canonical hit tuple.
    pub fn file_content_row(&self, id: i64) -> Result<Option<(String, u32, String)>> {
        let row = self
            .conn()
            .query_row(
                "SELECT rel_path, line, snippet FROM file_content WHERE id = ?1",
                params![id],
                |row| {
                    let rel_path: String = row.get(0)?;
                    let line: i64 = row.get(1)?;
                    let snippet: String = row.get(2)?;
                    Ok((rel_path, line as u32, snippet))
                },
            )
            .ok();
        Ok(row)
    }

    /// Find snippets matching `query`. Returns `(rel_path, line, snippet)`
    /// tuples ranked by BM25.
    pub fn search_file_content(
        &self,
        project: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<FileContentHit>> {
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }
        // Escape any FTS5 syntax in the query — the simplest safe
        // form is to bracket each token in double quotes and AND
        // them. A normal grep pattern like "processOrder" becomes
        // `"processOrder"`.
        let fts_query = build_fts_query(query);
        let mut stmt = self.conn().prepare(
            "SELECT c.rel_path, c.line, c.snippet, bm25(file_content_fts) AS rank
             FROM file_content_fts
             JOIN file_content c ON c.id = file_content_fts.rowid
             WHERE file_content_fts MATCH ?1
               AND c.project = ?2
             ORDER BY rank
             LIMIT ?3",
        )?;
        let hits = stmt
            .query_map(params![fts_query, project, limit as i64], |row| {
                let line: i64 = row.get(1)?;
                Ok(FileContentHit {
                    rel_path: row.get(0)?,
                    line: line as u32,
                    snippet: row.get(2)?,
                    rank: row.get(3)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(hits)
    }

    /// Count every file-content FTS hit for `query` within `project`.
    ///
    /// Used by machine-readable CLI output so `total_exact`, `shown`,
    /// `omitted` and `truncated` are not inferred from a limited result set.
    pub fn count_file_content_matches(&self, project: &str, query: &str) -> Result<usize> {
        if query.trim().is_empty() {
            return Ok(0);
        }
        let fts_query = build_fts_query(query);
        let total: i64 = self.conn().query_row(
            "SELECT COUNT(*)
             FROM file_content_fts
             JOIN file_content c ON c.id = file_content_fts.rowid
             WHERE file_content_fts MATCH ?1
               AND c.project = ?2",
            params![fts_query, project],
            |row| row.get(0),
        )?;
        Ok(total as usize)
    }
}

/// One row returned from `search_file_content`.
#[derive(Debug, Clone, PartialEq)]
pub struct FileContentHit {
    pub rel_path: String,
    pub line: u32,
    pub snippet: String,
    pub rank: f64,
}

/// Escape `query` into a safe FTS5 MATCH expression: each whitespace-
/// separated token is wrapped in double quotes and joined by AND.
fn build_fts_query(query: &str) -> String {
    query
        .split_whitespace()
        .filter(|t| !t.is_empty())
        .map(|t| format!("\"{}\"", t.replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join(" AND ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project::Project;

    fn store_with_project(name: &str) -> Store {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: name.into(),
            indexed_at: "2026-06-29T00:00:00Z".into(),
            root_path: format!("/r/{name}"),
        })
        .unwrap();
        s
    }

    #[test]
    fn insert_and_search_round_trip() {
        let mut s = store_with_project("p");
        s.insert_file_content_rows(
            "p",
            "src/lib.rs",
            &[
                ContentRow {
                    line: 1,
                    snippet: "fn hello() {}".into(),
                },
                ContentRow {
                    line: 2,
                    snippet: "let x = processOrder();".into(),
                },
                ContentRow {
                    line: 3,
                    snippet: "// see also brokenOrder".into(),
                },
            ],
        )
        .unwrap();
        let hits = s.search_file_content("p", "processOrder", 10).unwrap();
        assert_eq!(hits.len(), 1, "expected exactly one hit, got {hits:?}");
        assert_eq!(hits[0].rel_path, "src/lib.rs");
        assert_eq!(hits[0].line, 2);
        assert!(hits[0].snippet.contains("processOrder"));
    }

    #[test]
    fn search_and_count_are_project_scoped_before_limit() {
        let mut s = store_with_project("p1");
        s.upsert_project(&Project {
            name: "p2".into(),
            indexed_at: "2026-06-29T00:00:00Z".into(),
            root_path: "/r/p2".into(),
        })
        .unwrap();
        s.insert_file_content_rows(
            "p2",
            "src/noise.rs",
            &[
                ContentRow {
                    line: 1,
                    snippet: "needle from other project one".into(),
                },
                ContentRow {
                    line: 2,
                    snippet: "needle from other project two".into(),
                },
            ],
        )
        .unwrap();
        s.insert_file_content_rows(
            "p1",
            "src/lib.rs",
            &[
                ContentRow {
                    line: 1,
                    snippet: "needle first project hit one".into(),
                },
                ContentRow {
                    line: 2,
                    snippet: "needle first project hit two".into(),
                },
            ],
        )
        .unwrap();

        let hits = s.search_file_content("p1", "needle", 1).unwrap();
        assert_eq!(
            hits.len(),
            1,
            "limit=1 must still return a hit from p1, not be consumed by p2"
        );
        assert_eq!(hits[0].rel_path, "src/lib.rs");
        assert_eq!(s.count_file_content_matches("p1", "needle").unwrap(), 2);
        assert_eq!(s.count_file_content_matches("p2", "needle").unwrap(), 2);
    }

    #[test]
    fn reindex_overwrites_prior_content_for_same_file() {
        let mut s = store_with_project("p");
        s.insert_file_content_rows(
            "p",
            "src/lib.rs",
            &[ContentRow {
                line: 1,
                snippet: "fn old_name()".into(),
            }],
        )
        .unwrap();
        // R-018: delete-for-file then re-insert.
        let _ = s.delete_file_content("p", "src/lib.rs").unwrap();
        s.insert_file_content_rows(
            "p",
            "src/lib.rs",
            &[ContentRow {
                line: 1,
                snippet: "fn new_name()".into(),
            }],
        )
        .unwrap();
        assert!(
            s.search_file_content("p", "old_name", 10)
                .unwrap()
                .is_empty(),
            "stale content must not persist"
        );
        let hits = s.search_file_content("p", "new_name", 10).unwrap();
        assert_eq!(hits.len(), 1, "fresh content must be searchable");
    }

    /// Count rows in the FTS mirror. With an external-content table the
    /// index has no addressable row store, but the `*_data` shadow table
    /// is what leaked before the fix; the canonical, version-independent
    /// way to assert "no orphans" is that a full MATCH-less scan of the
    /// content table and the FTS index agree on row count. We compare the
    /// FTS index against `file_content` directly.
    fn fts_row_count(s: &Store) -> i64 {
        s.conn()
            .query_row("SELECT count(*) FROM file_content_fts", [], |r| r.get(0))
            .unwrap()
    }

    fn content_row_count(s: &Store) -> i64 {
        s.conn()
            .query_row("SELECT count(*) FROM file_content", [], |r| r.get(0))
            .unwrap()
    }

    #[test]
    fn reindex_prunes_orphaned_file_content_fts_rows() {
        // RV-009: re-indexing the same single-line file N times must NOT
        // accumulate orphan rows in the FTS mirror. The triggers added in
        // migration 0004 prune the FTS index on every delete, so after N
        // delete+reinsert cycles the FTS index and the content table hold
        // exactly the same number of rows.
        let mut s = store_with_project("p");

        const N: usize = 6;
        for i in 0..N {
            // R-018 re-index sequence: delete-for-file, then re-insert.
            s.delete_file_content("p", "src/lib.rs").unwrap();
            s.insert_file_content_rows(
                "p",
                "src/lib.rs",
                &[ContentRow {
                    line: 1,
                    snippet: format!("the_only_line_marker_xyz iteration {i}"),
                }],
            )
            .unwrap();

            // Invariant after every cycle: the contentless mirror is in
            // sync with the content table — never more, never fewer.
            assert_eq!(
                content_row_count(&s),
                1,
                "content table must hold exactly one row after cycle {i}"
            );
            assert_eq!(
                fts_row_count(&s),
                content_row_count(&s),
                "FTS mirror must match content row count (no orphans) after cycle {i}"
            );
        }

        // Final state: one content row, one FTS row, and search returns
        // that row exactly once (not N times).
        assert_eq!(content_row_count(&s), 1);
        assert_eq!(
            fts_row_count(&s),
            1,
            "no orphan FTS rows after {N} re-indexes"
        );

        let hits = s.search_file_content("p", "marker_xyz", 10).unwrap();
        assert_eq!(
            hits.len(),
            1,
            "search must return the row exactly once, got {hits:?}"
        );
        assert_eq!(hits[0].line, 1);
        assert!(hits[0].snippet.contains("marker_xyz"));

        // Stale content from earlier iterations must be gone, and only
        // the most recent iteration's content remains searchable.
        assert!(
            s.search_file_content("p", "iteration 0", 10)
                .unwrap()
                .is_empty(),
            "stale content from iteration 0 must not persist"
        );
        assert_eq!(
            s.search_file_content("p", "iteration 5", 10).unwrap().len(),
            1,
            "the most recent iteration's content must be searchable exactly once"
        );

        // Deleting once more drains both tables to zero — proving the
        // delete trigger prunes the FTS mirror rather than orphaning it.
        s.delete_file_content("p", "src/lib.rs").unwrap();
        assert_eq!(content_row_count(&s), 0);
        assert_eq!(
            fts_row_count(&s),
            0,
            "delete must leave zero orphan FTS rows"
        );
    }

    #[test]
    fn empty_query_yields_no_hits() {
        let mut s = store_with_project("p");
        s.insert_file_content_rows(
            "p",
            "src/lib.rs",
            &[ContentRow {
                line: 1,
                snippet: "x".into(),
            }],
        )
        .unwrap();
        let hits = s.search_file_content("p", "   ", 10).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn bm25_ranks_more_relevant_higher() {
        let mut s = store_with_project("p");
        s.insert_file_content_rows(
            "p",
            "src/lib.rs",
            &[
                ContentRow {
                    line: 1,
                    snippet: "fn hello".into(),
                },
                ContentRow {
                    line: 2,
                    snippet: "// hello world".into(),
                },
                ContentRow {
                    line: 3,
                    snippet: "fn greet".into(),
                },
            ],
        )
        .unwrap();
        let hits = s.search_file_content("p", "hello", 10).unwrap();
        assert!(
            hits.len() >= 2,
            "expected at least two hits for 'hello', got {hits:?}"
        );
        for h in &hits {
            assert!(
                h.snippet.contains("hello"),
                "every hit must contain the query, got {h:?}"
            );
        }
    }
}
