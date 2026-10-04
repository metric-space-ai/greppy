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
        let overlay = self.is_overlay();
        let tx = self.transaction()?;
        // SQLite foreign keys are confined to main. An additive repair can
        // reference a project visible only through immutable Base; materialize
        // its metadata in this transaction without copying any file ownership.
        if overlay {
            let projects = edges
                .iter()
                .map(|edge| edge.project.as_str())
                .collect::<std::collections::HashSet<_>>();
            for project in projects {
                tx.raw().execute(
                    "INSERT INTO main.projects(name, indexed_at, root_path)
                     SELECT name, indexed_at, root_path FROM greppy_base.projects
                     WHERE name = ?1
                     ON CONFLICT(name) DO NOTHING",
                    [project],
                )?;
            }
        }
        let mut ids = Vec::with_capacity(edges.len());
        {
            let raw = tx.raw();
            let mut stmt = raw.prepare_cached(
                "INSERT INTO main.raw_edges
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

    /// Replace only fingerprint-validated Rust USAGE contributions. Base stays
    /// immutable: a persisted per-project file mask hides its obsolete usages.
    /// Replacement rows for Base files are compatibility data, not Delta file
    /// ownership, and are excluded from ordinary sparse re-resolution.
    pub fn replace_validated_rust_usages(
        &mut self,
        project: &str,
        files: &[String],
        edges: &[NewRawEdge],
    ) -> Result<usize> {
        self.replace_validated_rust_edge_kind(project, files, edges, "USAGE")
    }

    /// Refresh caller facts without claiming ownership of immutable Base files.
    pub fn replace_validated_rust_calls(
        &mut self,
        project: &str,
        files: &[String],
        edges: &[NewRawEdge],
    ) -> Result<usize> {
        self.replace_validated_rust_edge_kind(project, files, edges, "CALLS")
    }

    fn replace_validated_rust_edge_kind(
        &mut self,
        project: &str,
        files: &[String],
        edges: &[NewRawEdge],
        kind: &str,
    ) -> Result<usize> {
        if edges.iter().any(|edge| {
            edge.edge_type != kind || edge.project != project || !files.contains(&edge.file_path)
        }) {
            return Err(Error::Invalid(
                "Rust repair rows do not match the validated edge scope".into(),
            ));
        }
        let overlay = self.is_overlay();
        let old = self.list_raw_edges(project)?;
        let prefix = if kind == "CALLS" { "caller" } else { "usage" };
        let key = format!("greppy.rust_{prefix}_override_files.{project}");
        let signature = |file: &str, source: &str, target: &str, properties: &serde_json::Value| {
            (
                file.to_owned(),
                source.to_owned(),
                target.to_owned(),
                properties.to_string(),
            )
        };
        let file_set = files
            .iter()
            .map(String::as_str)
            .collect::<std::collections::HashSet<_>>();
        let previous = old
            .iter()
            .filter(|edge| edge.edge_type == kind && file_set.contains(edge.file_path.as_str()))
            .map(|edge| {
                signature(
                    &edge.file_path,
                    &edge.source_qname,
                    &edge.target_qname,
                    &edge.properties,
                )
            })
            .collect::<std::collections::HashSet<_>>();
        let current = edges
            .iter()
            .map(|edge| {
                signature(
                    &edge.file_path,
                    &edge.source_qname,
                    &edge.target_qname,
                    &edge.properties,
                )
            })
            .collect::<std::collections::HashSet<_>>();
        if previous == current {
            return Ok(0);
        }
        let tx = self.transaction()?;
        if overlay {
            tx.raw().execute(
                "INSERT INTO main.projects(name,indexed_at,root_path)
                SELECT name,indexed_at,root_path FROM greppy_base.projects WHERE name=?1
                ON CONFLICT(name) DO NOTHING",
                [project],
            )?;
            tx.raw().execute(
                "INSERT INTO main.schema_meta(key,value) VALUES(?1,?2)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                params![key, serde_json::to_string(files)?],
            )?;
        }
        for file in files {
            tx.raw().execute(
                "DELETE FROM main.raw_edges WHERE project=?1 AND file_path=?2 AND edge_type=?3",
                params![project, file, kind],
            )?;
        }
        let mut base_replacements = Vec::new();
        for edge in edges {
            let owned: bool = !overlay
                || tx.raw().query_row(
                    "SELECT EXISTS(SELECT 1 FROM main.file_state WHERE project=?1 AND rel_path=?2)",
                    params![project, edge.file_path],
                    |row| row.get(0),
                )?;
            if !owned {
                base_replacements.push(serde_json::json!({
                    "file_path": edge.file_path, "source_qname": edge.source_qname,
                    "target_qname": edge.target_qname, "properties": edge.properties,
                }));
            } else {
                tx.raw().execute("INSERT INTO main.raw_edges(project,file_path,source_qname,target_qname,edge_type,properties)
                    VALUES(?1,?2,?3,?4,?5,?6)", params![project, edge.file_path, edge.source_qname, edge.target_qname, kind, serde_json::to_string(&edge.properties)?])?;
            }
        }
        if overlay {
            tx.raw().execute(
                "INSERT INTO main.schema_meta(key,value) VALUES(?1,?2)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                params![
                    format!("greppy.rust_{prefix}_override_rows.{project}"),
                    serde_json::to_string(&base_replacements)?
                ],
            )?;
        }
        tx.commit()?;
        Ok(previous.symmetric_difference(&current).count())
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

    /// List only raw edges physically owned by the private Delta database.
    /// In a composed Store-CoW view [`Store::list_raw_edges`] includes Base
    /// rows as well; rebuilding logical Delta edges must stay proportional to
    /// changed files and therefore consumes this main-schema-only set.
    pub fn list_delta_raw_edges(&self, project: &str) -> Result<Vec<RawEdge>> {
        let mut stmt = self.conn().prepare_cached(
            "SELECT id, project, file_path, source_qname, target_qname, edge_type, properties
             FROM main.raw_edges WHERE project = ?1
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

    /// Read only unresolved factory receiver facts for one file. The visible
    /// raw_edges relation retains immutable-Base repair masking; filtering in
    /// SQL avoids hydrating unrelated calls, imports and usages for navigation.
    pub fn list_raw_factory_receiver_edges_for_file(
        &self,
        project: &str,
        file_path: &str,
    ) -> Result<Vec<RawEdge>> {
        let mut stmt = self.conn().prepare_cached(
            "SELECT id, project, file_path, source_qname, target_qname, edge_type, properties
             FROM raw_edges WHERE project=?1 AND file_path=?2 AND edge_type='CALLS'
               AND json_type(properties,'$.receiver_factory_pattern') = 'text'
               AND json_extract(properties,'$.receiver_owner') IS NULL
             ORDER BY id",
        )?;
        let rows = stmt
            .query_map(params![project, file_path], row_to_raw_edge)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Read import context without expanding reference repair arrays in the
    /// overlay view. Those arrays contain CALLS/USAGE only; their window
    /// functions otherwise scan all repairs before applying a file filter.
    pub fn list_raw_import_edges_for_file(
        &self,
        project: &str,
        file_path: &str,
    ) -> Result<Vec<RawEdge>> {
        let sql = if self.is_overlay() {
            RAW_OVERLAY_IMPORTS_FOR_FILE_SQL
        } else {
            "SELECT id, project, file_path, source_qname, target_qname, edge_type, properties
             FROM main.raw_edges
             WHERE project = ?1 AND file_path = ?2 AND edge_type = 'IMPORTS'
             ORDER BY id"
        };
        let mut stmt = self.conn().prepare_cached(sql)?;
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
                "DELETE FROM main.raw_edges WHERE project = ?1 AND file_path = ?2",
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

const RAW_OVERLAY_IMPORTS_FOR_FILE_SQL: &str = "
SELECT id, project, file_path, source_qname, target_qname, edge_type, properties
FROM main.raw_edges
WHERE project = ?1 AND file_path = ?2 AND edge_type = 'IMPORTS'
UNION ALL
SELECT -b.id, b.project, b.file_path, b.source_qname, b.target_qname,
       b.edge_type, b.properties
FROM greppy_base.raw_edges b
WHERE b.project = ?1 AND b.file_path = ?2 AND b.edge_type = 'IMPORTS'
  AND NOT EXISTS (
      SELECT 1 FROM js_ts_reference_override_files f
      WHERE f.project = b.project AND f.file_path = b.file_path
  )
  AND NOT EXISTS (
      SELECT 1 FROM greppy_hidden_paths h WHERE h.path = b.file_path
  )
ORDER BY id";

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
    fn import_context_matches_overlay_visibility_without_repair_expansion() {
        let scratch = tempfile::tempdir().unwrap();
        let base_path = scratch.path().join("base.db");
        let delta_path = scratch.path().join("delta.db");
        {
            let mut base = Store::open(&base_path).unwrap();
            base.upsert_project(&Project {
                name: "p".into(),
                indexed_at: "cached".into(),
                root_path: "/root".into(),
            })
            .unwrap();
            base.insert_raw_edges(&[
                new_raw_edge("p", "module.rs", "source", "base", "IMPORTS"),
                new_raw_edge("p", "module.rs", "source", "reference", "USAGE"),
                new_raw_edge("p", "hidden.rs", "source", "hidden", "IMPORTS"),
                new_raw_edge("p", "masked.rs", "source", "masked", "IMPORTS"),
            ])
            .unwrap();
            assert_eq!(
                base.list_raw_import_edges_for_file("p", "module.rs")
                    .unwrap()
                    .len(),
                1
            );
        }
        let visibility =
            crate::VisibilityIndex::new(vec!["hidden.rs".to_string()], Vec::<String>::new())
                .unwrap();
        let mut overlay = Store::open_overlay(&base_path, &delta_path, &visibility).unwrap();
        overlay
            .insert_raw_edges(&[new_raw_edge("p", "module.rs", "source", "delta", "IMPORTS")])
            .unwrap();
        overlay.conn().execute(
            "INSERT INTO main.js_ts_reference_override_files(project,file_path) VALUES ('p','masked.rs')",
            [],
        ).unwrap();
        let repaired = vec![new_raw_edge("p", "module.rs", "source", "repair", "USAGE")];
        overlay
            .replace_validated_rust_usages("p", &["module.rs".into()], &repaired)
            .unwrap();
        for file in ["module.rs", "hidden.rs", "masked.rs"] {
            let expected = overlay
                .list_raw_edges_for_file("p", file)
                .unwrap()
                .into_iter()
                .filter(|row| row.edge_type == "IMPORTS")
                .collect::<Vec<_>>();
            assert_eq!(
                overlay.list_raw_import_edges_for_file("p", file).unwrap(),
                expected
            );
        }
        // A per-file import read must not execute the windowed synthetic
        // reference branches, even with a type predicate on the general view.
        let plan_sql = format!("EXPLAIN QUERY PLAN {RAW_OVERLAY_IMPORTS_FOR_FILE_SQL}");
        let mut stmt = overlay.conn().prepare(&plan_sql).unwrap();
        let plan = stmt
            .query_map(params!["p", "module.rs"], |row| row.get::<_, String>(3))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        assert!(
            plan.iter().all(|line| !line.contains("CO-ROUTINE")),
            "{plan:?}"
        );
    }

    #[test]
    fn validated_caller_replacement_is_atomic_and_keeps_usage_overrides_separate() {
        let scratch = tempfile::tempdir().unwrap();
        let base_path = scratch.path().join("base.db");
        let delta_path = scratch.path().join("delta.db");
        {
            let mut base = Store::open(&base_path).unwrap();
            base.upsert_project(&Project {
                name: "p".into(),
                indexed_at: "cached".into(),
                root_path: "/root".into(),
            })
            .unwrap();
            base.insert_raw_edges(&[
                new_raw_edge("p", "base.rs", "source", "obsolete", "CALLS"),
                new_raw_edge("p", "base.rs", "source", "keep_usage", "USAGE"),
            ])
            .unwrap();
        }
        let original_base = Store::open(&base_path)
            .unwrap()
            .list_raw_edges("p")
            .unwrap();
        let visibility =
            crate::VisibilityIndex::new(Vec::<String>::new(), Vec::<String>::new()).unwrap();
        let mut overlay = Store::open_overlay(&base_path, &delta_path, &visibility).unwrap();
        let original = overlay.list_raw_edges("p").unwrap();
        overlay.conn().execute_batch("CREATE TRIGGER reject_caller_override BEFORE INSERT ON main.schema_meta WHEN NEW.key='greppy.rust_caller_override_rows.p' BEGIN SELECT RAISE(ABORT,'fixture caller failure'); END;").unwrap();
        let files = vec!["base.rs".to_string()];
        let replacements = vec![new_raw_edge("p", "base.rs", "source", "correct", "CALLS")];
        assert!(overlay
            .replace_validated_rust_calls("p", &files, &replacements)
            .is_err());
        assert_eq!(overlay.list_raw_edges("p").unwrap(), original);
        let masks: i64 = overlay.conn().query_row("SELECT COUNT(*) FROM main.schema_meta WHERE key LIKE 'greppy.rust_caller_override_%'", [], |row| row.get(0)).unwrap();
        assert_eq!(masks, 0);
        overlay
            .conn()
            .execute_batch("DROP TRIGGER reject_caller_override")
            .unwrap();
        overlay
            .replace_validated_rust_calls("p", &files, &replacements)
            .unwrap();
        let rows = overlay.list_raw_edges("p").unwrap();
        assert!(rows
            .iter()
            .any(|edge| edge.edge_type == "CALLS" && edge.target_qname == "correct"));
        assert!(rows
            .iter()
            .any(|edge| edge.edge_type == "USAGE" && edge.target_qname == "keep_usage"));
        assert!(rows.iter().all(|edge| edge.target_qname != "obsolete"));
        assert!(overlay.list_delta_raw_edges("p").unwrap().is_empty());
        assert!(overlay.list_private_file_states("p").unwrap().is_empty());
        let repaired_ids = rows
            .iter()
            .filter(|edge| edge.edge_type == "CALLS")
            .map(|edge| edge.id)
            .collect::<Vec<_>>();
        drop(overlay);
        let reopened = Store::open_overlay(&base_path, &delta_path, &visibility).unwrap();
        assert_eq!(
            reopened
                .list_raw_edges("p")
                .unwrap()
                .iter()
                .filter(|edge| edge.edge_type == "CALLS")
                .map(|edge| edge.id)
                .collect::<Vec<_>>(),
            repaired_ids
        );
        assert_eq!(
            Store::open(&base_path)
                .unwrap()
                .list_raw_edges("p")
                .unwrap(),
            original_base
        );
    }

    #[test]
    fn validated_usage_replacement_rolls_back_base_override_on_failure() {
        let scratch = tempfile::tempdir().unwrap();
        let base_path = scratch.path().join("base.db");
        let delta_path = scratch.path().join("delta.db");
        {
            let mut base = Store::open(&base_path).unwrap();
            base.upsert_project(&Project {
                name: "p".into(),
                indexed_at: "cached-time".into(),
                root_path: "/cached/root".into(),
            })
            .unwrap();
            base.insert_raw_edges(&[new_raw_edge(
                "p",
                "base.rs",
                "p.source",
                "p.obsolete",
                "USAGE",
            )])
            .unwrap();
        }
        let visibility =
            crate::VisibilityIndex::new(Vec::<String>::new(), Vec::<String>::new()).unwrap();
        let mut overlay = Store::open_overlay(&base_path, &delta_path, &visibility).unwrap();
        let original = overlay.list_raw_edges("p").unwrap();
        overlay.conn().execute_batch("CREATE TRIGGER reject_usage_override BEFORE INSERT ON main.schema_meta WHEN NEW.key='greppy.rust_usage_override_rows.p' BEGIN SELECT RAISE(ABORT,'fixture override failure'); END;").unwrap();
        let files = vec!["base.rs".to_owned()];
        let replacements = vec![new_raw_edge(
            "p",
            "base.rs",
            "p.source",
            "p.correct",
            "USAGE",
        )];
        assert!(overlay
            .replace_validated_rust_usages("p", &files, &replacements)
            .is_err());
        assert_eq!(overlay.list_raw_edges("p").unwrap(), original);
        let markers: i64 = overlay.conn().query_row("SELECT COUNT(*) FROM main.schema_meta WHERE key LIKE 'greppy.rust_usage_override_%'", [], |row| row.get(0)).unwrap();
        assert_eq!(markers, 0, "replacement failure rolls back the Base mask");
        overlay
            .conn()
            .execute_batch("DROP TRIGGER reject_usage_override")
            .unwrap();
        overlay
            .replace_validated_rust_usages("p", &files, &replacements)
            .unwrap();
        assert!(overlay
            .list_raw_edges("p")
            .unwrap()
            .iter()
            .all(|edge| edge.target_qname != "p.obsolete"));
        assert!(overlay.list_delta_raw_edges("p").unwrap().is_empty());
        assert!(overlay.list_private_file_states("p").unwrap().is_empty());
    }

    #[test]
    fn validated_usage_override_ids_round_trip_across_projects_and_reopen() {
        let scratch = tempfile::tempdir().unwrap();
        let base_path = scratch.path().join("base.db");
        let delta_path = scratch.path().join("delta.db");
        {
            let mut base = Store::open(&base_path).unwrap();
            for project in ["p", "q"] {
                base.upsert_project(&Project {
                    name: project.into(),
                    indexed_at: "cached-time".into(),
                    root_path: format!("/cached/{project}"),
                })
                .unwrap();
                base.insert_raw_edges(&[new_raw_edge(
                    project, "base.rs", "source", "obsolete", "USAGE",
                )])
                .unwrap();
            }
        }
        let visibility =
            crate::VisibilityIndex::new(Vec::<String>::new(), Vec::<String>::new()).unwrap();
        let mut overlay = Store::open_overlay(&base_path, &delta_path, &visibility).unwrap();
        for project in ["p", "q"] {
            overlay
                .replace_validated_rust_usages(
                    project,
                    &["base.rs".into()],
                    &[new_raw_edge(
                        project, "base.rs", "source", "correct", "USAGE",
                    )],
                )
                .unwrap();
        }
        for reopen in [false, true] {
            if reopen {
                drop(overlay);
                overlay = Store::open_overlay(&base_path, &delta_path, &visibility).unwrap();
            }
            let p = overlay.list_raw_edges("p").unwrap();
            let q = overlay.list_raw_edges("q").unwrap();
            assert_eq!(p.len(), 1);
            assert_eq!(q.len(), 1);
            assert_ne!(p[0].id, q[0].id);
            for edge in p.iter().chain(q.iter()) {
                assert_eq!(overlay.get_raw_edge(edge.id).unwrap().as_ref(), Some(edge));
                assert_eq!(edge.target_qname, "correct");
            }
        }
        drop(overlay);
        let hidden =
            crate::VisibilityIndex::new(Vec::<String>::new(), vec!["base.rs".to_owned()]).unwrap();
        let overlay = Store::open_overlay(&base_path, &delta_path, &hidden).unwrap();
        assert!(overlay.list_raw_edges("p").unwrap().is_empty());
        assert!(overlay.list_raw_edges("q").unwrap().is_empty());
    }

    #[test]
    fn overlay_raw_insert_materializes_only_project_metadata_atomically() {
        let scratch = tempfile::tempdir().unwrap();
        let base_path = scratch.path().join("base.db");
        let delta_path = scratch.path().join("delta.db");
        let project = Project {
            name: "p".into(),
            indexed_at: "cached-time".into(),
            root_path: "/cached/root".into(),
        };
        {
            let mut base = Store::open(&base_path).unwrap();
            base.upsert_project(&project).unwrap();
        }
        let visibility =
            crate::VisibilityIndex::new(Vec::<String>::new(), Vec::<String>::new()).unwrap();
        let mut overlay = Store::open_overlay(&base_path, &delta_path, &visibility).unwrap();
        overlay.conn().execute_batch("CREATE TRIGGER main.reject_raw BEFORE INSERT ON main.raw_edges BEGIN SELECT RAISE(ABORT,'fixture raw failure'); END;").unwrap();
        let edge = new_raw_edge("p", "base.rs", "p.source", "p.target", "USAGE");
        assert!(overlay
            .insert_raw_edges(std::slice::from_ref(&edge))
            .is_err());
        let private_projects: i64 = overlay
            .conn()
            .query_row("SELECT COUNT(*) FROM main.projects", [], |row| row.get(0))
            .unwrap();
        assert_eq!(
            private_projects, 0,
            "failed raw batch rolls back project metadata"
        );
        overlay
            .conn()
            .execute_batch("DROP TRIGGER reject_raw")
            .unwrap();
        overlay.insert_raw_edges(&[edge]).unwrap();
        let private_metadata: (String, String) = overlay
            .conn()
            .query_row(
                "SELECT indexed_at,root_path FROM main.projects WHERE name='p'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(private_metadata, (project.indexed_at, project.root_path));
        assert!(overlay.list_private_file_states("p").unwrap().is_empty());
        assert!(overlay.list_private_workspace_states().unwrap().is_empty());
        assert!(overlay.list_nodes("p", "", "", 0, 10).unwrap().is_empty());
        assert_eq!(overlay.list_delta_raw_edges("p").unwrap().len(), 1);
        drop(overlay);
        let base = Store::open(&base_path).unwrap();
        assert!(base.list_raw_edges("p").unwrap().is_empty());
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
