//! Database maintenance: VACUUM, ANALYZE, and FTS5 index optimisation.
//!
//! These are the housekeeping operations a long-lived store wants to run
//! periodically (e.g. after a large reindex): reclaim free pages, refresh the
//! query planner's statistics, and merge the FTS5 b-tree segments so MATCH
//! queries stay fast. They are thin, explicit wrappers so callers (the CLI's
//! `maintain` / `optimize` paths) do not have to embed raw SQL.

use crate::store::Store;
use crate::store_error::{Error, Result};

impl Store {
    /// Reclaim unused pages by rewriting the database file (`VACUUM`).
    ///
    /// VACUUM cannot run inside an open transaction and rewrites the whole
    /// file, so it is comparatively expensive — intended for occasional
    /// maintenance, not the hot path. Returns once the rebuild completes.
    pub fn vacuum(&self) -> Result<()> {
        self.conn()
            .execute_batch("VACUUM")
            .map_err(|e| Error::Store(format!("vacuum: {e}")))
    }

    /// Refresh the query planner's statistics (`ANALYZE`).
    ///
    /// ANALYZE gathers index selectivity into `sqlite_stat1` so the planner
    /// picks the better index for `(project, name)` / `(project, label)`
    /// lookups. Cheap relative to VACUUM; safe to run after a bulk insert.
    pub fn analyze(&self) -> Result<()> {
        self.conn()
            .execute_batch("ANALYZE")
            .map_err(|e| Error::Store(format!("analyze: {e}")))
    }

    /// Run both maintenance passes in the order that compounds best:
    /// `ANALYZE` first (so the planner stats reflect current data), then
    /// `VACUUM` to compact. A convenience for a CLI `maintain` command.
    pub fn vacuum_and_analyze(&self) -> Result<()> {
        self.analyze()?;
        self.vacuum()
    }

    /// Merge the FTS5 index segments for `nodes_fts` (`'optimize'`).
    ///
    /// FTS5 accumulates b-tree segments as rows are inserted/deleted; the
    /// special `INSERT INTO nodes_fts(nodes_fts) VALUES('optimize')` command
    /// merges them into a single segment, which shrinks the index and speeds
    /// up subsequent MATCH queries. Run after a large reindex churns the
    /// contentless index. No-op-safe to call repeatedly.
    pub fn optimize_fts(&self) -> Result<()> {
        self.conn()
            .execute("INSERT INTO nodes_fts(nodes_fts) VALUES('optimize')", [])
            .map(|_| ())
            .map_err(|e| Error::Store(format!("optimize nodes_fts: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::NewNode;
    use crate::project::Project;

    fn store_with_data() -> Store {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "2026-06-28T20:00:00Z".into(),
            root_path: "/repos/p".into(),
        })
        .unwrap();
        for i in 0..50 {
            s.insert_node(&NewNode {
                project: "p".into(),
                label: "Function".into(),
                name: format!("fn{i}"),
                qualified_name: format!("p.fn{i}"),
                file_path: "src/lib.rs".into(),
                start_line: 1,
                end_line: 2,
                properties: serde_json::json!({}),
            })
            .unwrap();
        }
        s
    }

    #[test]
    fn analyze_then_vacuum_keep_db_usable() {
        let s = store_with_data();
        s.analyze().unwrap();
        // ANALYZE must have populated planner stats.
        let stat_rows: i64 = s
            .conn()
            .query_row("SELECT COUNT(*) FROM sqlite_stat1", [], |r| r.get(0))
            .unwrap();
        assert!(stat_rows > 0, "ANALYZE must populate sqlite_stat1");

        s.vacuum().unwrap();
        // Data survives the rebuild and the DB still passes integrity.
        let n: i64 = s
            .conn()
            .query_row("SELECT COUNT(*) FROM nodes", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 50);
        s.integrity_check().unwrap();
    }

    #[test]
    fn vacuum_and_analyze_convenience_runs_both() {
        let s = store_with_data();
        s.vacuum_and_analyze().unwrap();
        let stat_rows: i64 = s
            .conn()
            .query_row("SELECT COUNT(*) FROM sqlite_stat1", [], |r| r.get(0))
            .unwrap();
        assert!(stat_rows > 0);
        s.integrity_check().unwrap();
    }

    #[test]
    fn optimize_fts_keeps_search_working() {
        let s = store_with_data();
        s.optimize_fts().unwrap();
        // Idempotent: running again is fine.
        s.optimize_fts().unwrap();
        // The contentless FTS index still answers MATCH after optimisation.
        let hits = crate::fts::search_fts(&s, "fn1", 10).unwrap();
        assert!(!hits.is_empty(), "FTS must still match after optimize");
        s.integrity_check().unwrap();
    }
}
