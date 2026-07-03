//! `project_summaries` CRUD.
//!
//! Mirrors upstream's `project_summaries` table (declared in
//! `internal/cbm/sqlite_writer.c`): one row per project holding a generated
//! natural-language summary plus the `source_hash` of the inputs it was
//! derived from, and `created_at` / `updated_at` timestamps.
//!
//! The table is keyed by `project` and is FK-cascaded by the schema
//! (`projects(name) ON DELETE CASCADE`), so deleting a project drops its
//! summary when foreign keys are enforced; `delete_project_summary` is the
//! explicit path.
//!
//! Upsert semantics match the rest of the store: an insert preserves the
//! original `created_at` and only bumps `updated_at` on conflict, so the
//! creation time is stable across regenerations.

use rusqlite::{params, OptionalExtension};

use crate::store::Store;
use crate::store_error::Result;
use crate::workspace_state::now_iso8601;

/// One row of the `project_summaries` table.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectSummary {
    pub project: String,
    pub summary: String,
    /// Hash of the inputs the summary was derived from, so a caller can
    /// detect a stale summary without re-reading the source.
    pub source_hash: String,
    pub created_at: String,
    pub updated_at: String,
}

impl Store {
    /// Insert or update a project summary.
    ///
    /// On first insert, `created_at` and `updated_at` are both set to now.
    /// On conflict (same `project`) the `summary`/`source_hash` are
    /// replaced and `updated_at` is refreshed, while `created_at` is
    /// preserved from the original row.
    pub fn upsert_project_summary(
        &mut self,
        project: &str,
        summary: &str,
        source_hash: &str,
    ) -> Result<()> {
        let now = now_iso8601();
        let tx = self.transaction()?;
        tx.raw().execute(
            "INSERT INTO project_summaries
               (project, summary, source_hash, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?4)
             ON CONFLICT(project) DO UPDATE SET
               summary = excluded.summary,
               source_hash = excluded.source_hash,
               updated_at = excluded.updated_at",
            params![project, summary, source_hash, now],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Fetch a project summary by project name.
    pub fn get_project_summary(&self, project: &str) -> Result<Option<ProjectSummary>> {
        let row = self
            .conn()
            .query_row(
                "SELECT project, summary, source_hash, created_at, updated_at
                 FROM project_summaries WHERE project = ?1",
                params![project],
                row_to_summary,
            )
            .optional()?;
        Ok(row)
    }

    /// Delete a project summary. Returns `true` if a row was removed.
    pub fn delete_project_summary(&mut self, project: &str) -> Result<bool> {
        let tx = self.transaction()?;
        let n = tx.raw().execute(
            "DELETE FROM project_summaries WHERE project = ?1",
            params![project],
        )?;
        tx.commit()?;
        Ok(n > 0)
    }
}

fn row_to_summary(row: &rusqlite::Row<'_>) -> rusqlite::Result<ProjectSummary> {
    Ok(ProjectSummary {
        project: row.get(0)?,
        summary: row.get(1)?,
        source_hash: row.get(2)?,
        created_at: row.get(3)?,
        updated_at: row.get(4)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project::Project;

    fn store_with_project(name: &str) -> Store {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: name.into(),
            indexed_at: "2026-06-28T20:00:00Z".into(),
            root_path: format!("/repos/{name}"),
        })
        .unwrap();
        s
    }

    #[test]
    fn upsert_then_get_round_trip() {
        let mut s = store_with_project("demo");
        s.upsert_project_summary("demo", "A small demo crate.", "hash-1")
            .unwrap();
        let got = s.get_project_summary("demo").unwrap().unwrap();
        assert_eq!(got.project, "demo");
        assert_eq!(got.summary, "A small demo crate.");
        assert_eq!(got.source_hash, "hash-1");
        assert!(!got.created_at.is_empty());
        assert_eq!(
            got.created_at, got.updated_at,
            "first insert: created == updated"
        );
    }

    #[test]
    fn get_missing_returns_none() {
        let s = store_with_project("demo");
        assert!(s.get_project_summary("demo").unwrap().is_none());
        assert!(s.get_project_summary("nope").unwrap().is_none());
    }

    #[test]
    fn upsert_replaces_summary_and_preserves_created_at() {
        let mut s = store_with_project("demo");
        s.upsert_project_summary("demo", "v1", "hash-1").unwrap();
        let first = s.get_project_summary("demo").unwrap().unwrap();

        s.upsert_project_summary("demo", "v2", "hash-2").unwrap();
        let second = s.get_project_summary("demo").unwrap().unwrap();

        assert_eq!(second.summary, "v2");
        assert_eq!(second.source_hash, "hash-2");
        assert_eq!(
            second.created_at, first.created_at,
            "created_at must be preserved across upsert"
        );
        // Exactly one row exists (upsert, not a second insert).
        let count: i64 = s
            .conn()
            .query_row("SELECT COUNT(*) FROM project_summaries", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn delete_removes_row() {
        let mut s = store_with_project("demo");
        s.upsert_project_summary("demo", "v1", "hash-1").unwrap();
        assert!(s.delete_project_summary("demo").unwrap());
        assert!(s.get_project_summary("demo").unwrap().is_none());
        // Deleting again is a no-op returning false.
        assert!(!s.delete_project_summary("demo").unwrap());
    }

    #[test]
    fn summaries_are_per_project() {
        let mut s = store_with_project("a");
        s.upsert_project(&Project {
            name: "b".into(),
            indexed_at: "2026-06-28T20:00:00Z".into(),
            root_path: "/repos/b".into(),
        })
        .unwrap();
        s.upsert_project_summary("a", "summary of a", "ha").unwrap();
        s.upsert_project_summary("b", "summary of b", "hb").unwrap();
        assert_eq!(
            s.get_project_summary("a").unwrap().unwrap().summary,
            "summary of a"
        );
        assert_eq!(
            s.get_project_summary("b").unwrap().unwrap().summary,
            "summary of b"
        );
        // Deleting a's summary leaves b's intact.
        s.delete_project_summary("a").unwrap();
        assert!(s.get_project_summary("a").unwrap().is_none());
        assert!(s.get_project_summary("b").unwrap().is_some());
    }
}
