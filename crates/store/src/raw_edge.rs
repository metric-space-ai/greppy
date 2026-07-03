//! Raw (unresolved) edge CRUD against the store-owned `raw_edges` table.
//!
//! The indexer extracts edges as `(source_qname, target_qname, edge_type,
//! properties)` tuples *before* it can resolve a qualified-name to a node id
//! — the resolution pass runs project-wide once every file is parsed. Today
//! the indexer persists those tuples in an ad-hoc `indexer_raw_edges` sidecar
//! it creates via raw `conn()` DDL. This module provides the typed,
//! store-owned replacement (migration 0007) so a future wave can switch the
//! indexer onto the store API instead of hand-rolled SQL.
//!
//! Rows are keyed by `(project, file_path)`: a file's contribution is
//! replaced wholesale with [`Store::delete_raw_edges_for_file`] before its
//! freshly-extracted edges are re-inserted, mirroring the per-file
//! delete-then-insert the indexer already does for nodes (R-018).
//!
//! This is **purely additive**: the indexer and its existing
//! `indexer_raw_edges` table are untouched.

use rusqlite::{params, OptionalExtension};

use crate::store::Store;
use crate::store_error::{Error, Result};

/// One row of the `raw_edges` table plus its parsed JSON properties.
#[derive(Debug, Clone, PartialEq)]
pub struct RawEdge {
    pub id: i64,
    pub project: String,
    pub file_path: String,
    pub source_qname: String,
    pub target_qname: String,
    pub edge_type: String,
    pub properties: serde_json::Value,
}

/// Input for inserting a raw edge. `id` is assigned by SQLite on insert.
#[derive(Debug, Clone)]
pub struct NewRawEdge {
    pub project: String,
    pub file_path: String,
    pub source_qname: String,
    pub target_qname: String,
    pub edge_type: String,
    pub properties: serde_json::Value,
}

impl Store {
    /// Insert many raw edges inside a SINGLE transaction (one fsync for the
    /// whole batch, mirroring [`Store::insert_nodes`]). Returns the assigned
    /// ids in input order. An empty slice is a no-op that returns an empty
    /// vec without opening a transaction.
    ///
    /// Unlike nodes/edges this is a plain append (no upsert): the indexer's
    /// contract is delete-then-insert per file, so duplicate suppression is
    /// the caller's job (call [`Store::delete_raw_edges_for_file`] first).
    pub fn insert_raw_edges(&mut self, edges: &[NewRawEdge]) -> Result<Vec<i64>> {
        if edges.is_empty() {
            return Ok(Vec::new());
        }
        let tx = self.transaction()?;
        let mut ids = Vec::with_capacity(edges.len());
        {
            let raw = tx.raw();
            let mut stmt = raw.prepare_cached(
                "INSERT INTO raw_edges
                   (project, file_path, source_qname, target_qname, edge_type, properties)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                 RETURNING id",
            )?;
            for e in edges {
                let props_str = serde_json::to_string(&e.properties)?;
                let id: i64 = stmt
                    .query_row(
                        params![
                            e.project,
                            e.file_path,
                            e.source_qname,
                            e.target_qname,
                            e.edge_type,
                            props_str,
                        ],
                        |row| row.get(0),
                    )
                    .map_err(Error::Sqlite)?;
                ids.push(id);
            }
        }
        tx.commit()?;
        Ok(ids)
    }

    /// List every raw edge for `project` in a deterministic order
    /// (`file_path`, then `id`, so a file's edges keep their insert order).
    /// This is the project-wide raw-edge set a resolution pass runs over.
    pub fn list_raw_edges(&self, project: &str) -> Result<Vec<RawEdge>> {
        let mut stmt = self.conn().prepare_cached(
            "SELECT id, project, file_path, source_qname, target_qname, edge_type, properties
             FROM raw_edges WHERE project = ?1
             ORDER BY file_path, id",
        )?;
        let rows = stmt
            .query_map(params![project], row_to_raw_edge)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// List the raw edges contributed by a single `(project, file_path)`,
    /// ordered by `id`. Useful for verifying a file's contribution.
    pub fn list_raw_edges_for_file(&self, project: &str, file_path: &str) -> Result<Vec<RawEdge>> {
        let mut stmt = self.conn().prepare_cached(
            "SELECT id, project, file_path, source_qname, target_qname, edge_type, properties
             FROM raw_edges WHERE project = ?1 AND file_path = ?2
             ORDER BY id",
        )?;
        let rows = stmt
            .query_map(params![project, file_path], row_to_raw_edge)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Delete every raw edge for `(project, file_path)` and return the number
    /// of rows removed. Called before re-inserting a re-extracted file's
    /// edges and for deleted files (per-file delete-then-insert).
    pub fn delete_raw_edges_for_file(&mut self, project: &str, file_path: &str) -> Result<usize> {
        let n = self
            .conn()
            .execute(
                "DELETE FROM raw_edges WHERE project = ?1 AND file_path = ?2",
                params![project, file_path],
            )
            .map_err(Error::Sqlite)?;
        Ok(n)
    }

    /// Count the raw edges stored for `project`.
    pub fn count_raw_edges(&self, project: &str) -> Result<i64> {
        let n: i64 = self.conn().query_row(
            "SELECT COUNT(*) FROM raw_edges WHERE project = ?1",
            params![project],
            |row| row.get(0),
        )?;
        Ok(n)
    }

    /// Fetch a single raw edge by id (primarily for tests / diagnostics).
    pub fn get_raw_edge(&self, id: i64) -> Result<Option<RawEdge>> {
        let row = self
            .conn()
            .query_row(
                "SELECT id, project, file_path, source_qname, target_qname, edge_type, properties
                 FROM raw_edges WHERE id = ?1",
                params![id],
                row_to_raw_edge,
            )
            .optional()?;
        Ok(row)
    }
}

fn row_to_raw_edge(row: &rusqlite::Row<'_>) -> rusqlite::Result<RawEdge> {
    let props_str: String = row.get(6)?;
    let properties: serde_json::Value =
        serde_json::from_str(&props_str).unwrap_or(serde_json::Value::Null);
    Ok(RawEdge {
        id: row.get(0)?,
        project: row.get(1)?,
        file_path: row.get(2)?,
        source_qname: row.get(3)?,
        target_qname: row.get(4)?,
        edge_type: row.get(5)?,
        properties,
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

    fn new_raw_edge(project: &str, file: &str, src: &str, tgt: &str, ty: &str) -> NewRawEdge {
        NewRawEdge {
            project: project.into(),
            file_path: file.into(),
            source_qname: src.into(),
            target_qname: tgt.into(),
            edge_type: ty.into(),
            properties: serde_json::json!({"line": 1}),
        }
    }

    #[test]
    fn insert_then_list_round_trip() {
        let mut s = store_with_project("p");
        let ids = s
            .insert_raw_edges(&[
                new_raw_edge("p", "a.rs", "p.a", "p.b", "CALLS"),
                new_raw_edge("p", "a.rs", "p.a", "p.c", "CALLS"),
            ])
            .unwrap();
        assert_eq!(ids.len(), 2);
        assert_eq!(s.count_raw_edges("p").unwrap(), 2);

        let all = s.list_raw_edges("p").unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].source_qname, "p.a");
        assert_eq!(all[0].target_qname, "p.b");
        assert_eq!(all[0].edge_type, "CALLS");
        assert_eq!(all[0].properties["line"], 1);

        let one = s.get_raw_edge(ids[0]).unwrap().unwrap();
        assert_eq!(one, all[0]);
    }

    #[test]
    fn empty_batch_is_noop() {
        let mut s = store_with_project("p");
        assert!(s.insert_raw_edges(&[]).unwrap().is_empty());
        assert_eq!(s.count_raw_edges("p").unwrap(), 0);
    }

    #[test]
    fn list_is_deterministic_by_file_then_id() {
        let mut s = store_with_project("p");
        // Insert out of file order; list must come back file-then-id sorted.
        s.insert_raw_edges(&[
            new_raw_edge("p", "z.rs", "p.z", "p.a", "CALLS"),
            new_raw_edge("p", "a.rs", "p.a", "p.b", "CALLS"),
            new_raw_edge("p", "a.rs", "p.a", "p.c", "IMPORTS"),
        ])
        .unwrap();
        let all = s.list_raw_edges("p").unwrap();
        let order: Vec<(&str, &str)> = all
            .iter()
            .map(|e| (e.file_path.as_str(), e.target_qname.as_str()))
            .collect();
        assert_eq!(
            order,
            vec![("a.rs", "p.b"), ("a.rs", "p.c"), ("z.rs", "p.a")]
        );
    }

    #[test]
    fn delete_for_file_removes_only_that_file() {
        let mut s = store_with_project("p");
        s.insert_raw_edges(&[
            new_raw_edge("p", "a.rs", "p.a", "p.b", "CALLS"),
            new_raw_edge("p", "b.rs", "p.b", "p.c", "CALLS"),
        ])
        .unwrap();
        let removed = s.delete_raw_edges_for_file("p", "a.rs").unwrap();
        assert_eq!(removed, 1);
        let all = s.list_raw_edges("p").unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].file_path, "b.rs");
        // Deleting a file with no rows removes zero.
        assert_eq!(s.delete_raw_edges_for_file("p", "missing.rs").unwrap(), 0);
    }

    #[test]
    fn delete_then_reinsert_per_file_replaces_contribution() {
        let mut s = store_with_project("p");
        s.insert_raw_edges(&[new_raw_edge("p", "a.rs", "p.a", "p.old", "CALLS")])
            .unwrap();
        // Re-extract a.rs: delete then insert fresh edges.
        s.delete_raw_edges_for_file("p", "a.rs").unwrap();
        s.insert_raw_edges(&[new_raw_edge("p", "a.rs", "p.a", "p.new", "CALLS")])
            .unwrap();
        let file_edges = s.list_raw_edges_for_file("p", "a.rs").unwrap();
        assert_eq!(file_edges.len(), 1);
        assert_eq!(file_edges[0].target_qname, "p.new");
    }

    #[test]
    fn raw_edges_are_project_scoped() {
        let mut s = store_with_project("p1");
        s.upsert_project(&Project {
            name: "p2".into(),
            indexed_at: "2026-06-28T20:00:00Z".into(),
            root_path: "/repos/p2".into(),
        })
        .unwrap();
        s.insert_raw_edges(&[new_raw_edge("p1", "a.rs", "p1.a", "p1.b", "CALLS")])
            .unwrap();
        s.insert_raw_edges(&[new_raw_edge("p2", "a.rs", "p2.a", "p2.b", "CALLS")])
            .unwrap();
        assert_eq!(s.list_raw_edges("p1").unwrap().len(), 1);
        assert_eq!(s.list_raw_edges("p2").unwrap().len(), 1);
        assert_eq!(s.list_raw_edges("p1").unwrap()[0].project, "p1");
    }

    /// Deleting a project cascades to its raw edges (FK ON DELETE CASCADE),
    /// provided foreign keys are enforced on the connection.
    #[test]
    fn delete_project_cascades_when_fks_enforced() {
        let mut s = store_with_project("p");
        s.conn().execute_batch("PRAGMA foreign_keys = ON").unwrap();
        s.insert_raw_edges(&[new_raw_edge("p", "a.rs", "p.a", "p.b", "CALLS")])
            .unwrap();
        s.conn()
            .execute("DELETE FROM projects WHERE name = 'p'", [])
            .unwrap();
        assert_eq!(s.count_raw_edges("p").unwrap(), 0);
    }
}
