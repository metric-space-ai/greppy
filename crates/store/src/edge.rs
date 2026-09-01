//! Edge CRUD: insert directed edges between nodes, list by source/target/type.

use rusqlite::{params, OptionalExtension};

use crate::store::Store;
use crate::store_error::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    pub id: i64,
    pub project: String,
    pub source_id: i64,
    pub target_id: i64,
    pub edge_type: String,
    pub properties: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct NewEdge {
    pub project: String,
    pub source_id: i64,
    pub target_id: i64,
    pub edge_type: String,
    pub properties: serde_json::Value,
}

/// An edge owned by a private Store-CoW Delta. Endpoints are logical graph
/// identities because either endpoint may live in the immutable Base DB.
#[derive(Debug, Clone)]
pub struct NewOverlayEdge {
    pub project: String,
    pub source_qualified_name: String,
    pub target_qualified_name: String,
    pub edge_type: String,
    pub properties: serde_json::Value,
}

#[derive(Debug)]
struct OverlayEdgeCandidate {
    id: i64,
    project: String,
    source_qualified_name: String,
    target_qualified_name: String,
    edge_type: String,
    properties: String,
}

impl Store {
    /// Upsert private overlay edges in one transaction.
    ///
    /// Callers resolve database-local node ids to qualified names before
    /// entering this method. Keeping the whole batch in one transaction
    /// avoids one fsync/transaction per structural edge in a CoW Delta.
    pub fn insert_overlay_edges(&mut self, edges: &[NewOverlayEdge]) -> Result<()> {
        if edges.is_empty() {
            return Ok(());
        }
        let tx = self.transaction()?;
        {
            let mut stmt = tx.raw().prepare_cached(
                "INSERT INTO main.overlay_edges
                   (project, source_qualified_name, target_qualified_name, edge_type, properties)
                 VALUES (?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(project, source_qualified_name, target_qualified_name, edge_type)
                 DO UPDATE SET properties = excluded.properties",
            )?;
            for edge in edges {
                let properties = serde_json::to_string(&edge.properties)?;
                stmt.execute(params![
                    edge.project,
                    edge.source_qualified_name,
                    edge.target_qualified_name,
                    edge.edge_type,
                    properties,
                ])?;
            }
        }
        tx.commit()
    }

    /// Replace all logical Delta edges for one project atomically.
    ///
    /// The composed overlay view resolves qnames to the currently visible
    /// Base-or-Delta nodes. Thus a dirty-file edge can target an unchanged
    /// Base node without leaking a database-local Base id into the Delta.
    pub fn replace_overlay_edges(&mut self, project: &str, edges: &[NewOverlayEdge]) -> Result<()> {
        let tx = self.transaction()?;
        tx.raw().execute(
            "DELETE FROM main.overlay_edges
             WHERE project = ?1
               AND edge_type NOT IN ('CONTAINS_FOLDER', 'CONTAINS_FILE', 'DEFINES')",
            params![project],
        )?;
        {
            let mut stmt = tx.raw().prepare_cached(
                "INSERT INTO main.overlay_edges
                   (project, source_qualified_name, target_qualified_name, edge_type, properties)
                 VALUES (?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(project, source_qualified_name, target_qualified_name, edge_type)
                 DO UPDATE SET properties = excluded.properties",
            )?;
            for edge in edges {
                let properties = serde_json::to_string(&edge.properties)?;
                stmt.execute(params![
                    edge.project,
                    edge.source_qualified_name,
                    edge.target_qualified_name,
                    edge.edge_type,
                    properties,
                ])?;
            }
        }
        tx.commit()
    }

    /// Insert an edge. Returns the assigned id. The `(source_id,
    /// target_id, edge_type)` triple is unique; a duplicate insert is
    /// upserted (enforced by the `UNIQUE(source_id, target_id, type)`
    /// schema constraint).
    pub fn insert_edge(&mut self, e: &NewEdge) -> Result<i64> {
        if self.is_overlay() {
            let visible: Option<(i64, String)> = self
                .conn()
                .query_row(
                    "SELECT id, properties FROM edges
                     WHERE project = ?1 AND source_id = ?2 AND target_id = ?3 AND edge_type = ?4
                     ORDER BY id DESC LIMIT 1",
                    params![e.project, e.source_id, e.target_id, e.edge_type],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()?;
            if let Some((id, properties)) = visible {
                let properties: serde_json::Value = serde_json::from_str(&properties)?;
                if properties == e.properties {
                    return Ok(id);
                }
            }
            let source_qualified_name: String = self.conn().query_row(
                "SELECT qualified_name FROM nodes WHERE id = ?1 AND project = ?2",
                params![e.source_id, e.project],
                |row| row.get(0),
            )?;
            let target_qualified_name: String = self.conn().query_row(
                "SELECT qualified_name FROM nodes WHERE id = ?1 AND project = ?2",
                params![e.target_id, e.project],
                |row| row.get(0),
            )?;
            return self.insert_overlay_edge(&NewOverlayEdge {
                project: e.project.clone(),
                source_qualified_name,
                target_qualified_name,
                edge_type: e.edge_type.clone(),
                properties: e.properties.clone(),
            });
        }
        let props_str = serde_json::to_string(&e.properties)?;
        let tx = self.transaction()?;
        let id: i64 = tx
            .raw()
            .prepare_cached(
                "INSERT INTO main.edges (project, source_id, target_id, edge_type, properties)
                 VALUES (?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(source_id, target_id, edge_type) DO UPDATE SET
                   properties = excluded.properties
                 RETURNING id",
            )?
            .query_row(
                params![e.project, e.source_id, e.target_id, e.edge_type, props_str],
                |row| row.get(0),
            )
            .map_err(Error::Sqlite)?;
        tx.commit()?;
        Ok(id)
    }

    fn insert_overlay_edge(&mut self, edge: &NewOverlayEdge) -> Result<i64> {
        let properties = serde_json::to_string(&edge.properties)?;
        let tx = self.transaction()?;
        let id = tx.raw().query_row(
            "INSERT INTO main.overlay_edges
               (project, source_qualified_name, target_qualified_name, edge_type, properties)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(project, source_qualified_name, target_qualified_name, edge_type)
             DO UPDATE SET properties = excluded.properties
             RETURNING id",
            params![
                edge.project,
                edge.source_qualified_name,
                edge.target_qualified_name,
                edge.edge_type,
                properties,
            ],
            |row| row.get(0),
        )?;
        tx.commit()?;
        Ok(id)
    }

    /// Fetch an edge by id.
    pub fn get_edge(&self, id: i64) -> Result<Option<Edge>> {
        let row = self
            .conn()
            .query_row(
                "SELECT id, project, source_id, target_id, edge_type, properties
                 FROM edges WHERE id = ?1",
                params![id],
                row_to_edge,
            )
            .optional()?;
        Ok(row)
    }

    /// Outgoing edges from `source_id` of a given type. Pass `None` for
    /// `edge_type` to list all types.
    pub fn outgoing_edges(
        &self,
        source_id: i64,
        edge_type: Option<&str>,
        limit: usize,
    ) -> Result<Vec<Edge>> {
        if self.is_overlay() {
            if let Some(edge_type) = edge_type {
                return self.overlay_edges_for_endpoint(source_id, edge_type, limit, false);
            }
        }
        let (sql, has_type) = match edge_type {
            Some(_) => (
                "SELECT id, project, source_id, target_id, edge_type, properties
                 FROM edges WHERE source_id = ?1 AND edge_type = ?2
                 ORDER BY id LIMIT ?3",
                true,
            ),
            None => (
                "SELECT id, project, source_id, target_id, edge_type, properties
                 FROM edges WHERE source_id = ?1
                 ORDER BY id LIMIT ?2",
                false,
            ),
        };
        let mut stmt = self.conn().prepare(sql)?;
        let rows = if has_type {
            stmt.query_map(
                params![source_id, edge_type.unwrap(), limit as i64],
                row_to_edge,
            )?
            .collect::<rusqlite::Result<Vec<_>>>()?
        } else {
            stmt.query_map(params![source_id, limit as i64], row_to_edge)?
                .collect::<rusqlite::Result<Vec<_>>>()?
        };
        Ok(rows)
    }

    /// Incoming edges to `target_id`. Same `edge_type` semantics as
    /// `outgoing_edges`.
    pub fn incoming_edges(
        &self,
        target_id: i64,
        edge_type: Option<&str>,
        limit: usize,
    ) -> Result<Vec<Edge>> {
        if self.is_overlay() {
            if let Some(edge_type) = edge_type {
                return self.overlay_edges_for_endpoint(target_id, edge_type, limit, true);
            }
        }
        let (sql, has_type) = match edge_type {
            Some(_) => (
                "SELECT id, project, source_id, target_id, edge_type, properties
                 FROM edges WHERE target_id = ?1 AND edge_type = ?2
                 ORDER BY id LIMIT ?3",
                true,
            ),
            None => (
                "SELECT id, project, source_id, target_id, edge_type, properties
                 FROM edges WHERE target_id = ?1
                 ORDER BY id LIMIT ?2",
                false,
            ),
        };
        let mut stmt = self.conn().prepare(sql)?;
        let rows = if has_type {
            stmt.query_map(
                params![target_id, edge_type.unwrap(), limit as i64],
                row_to_edge,
            )?
            .collect::<rusqlite::Result<Vec<_>>>()?
        } else {
            stmt.query_map(params![target_id, limit as i64], row_to_edge)?
                .collect::<rusqlite::Result<Vec<_>>>()?
        };
        Ok(rows)
    }

    /// Resolve typed edges in an overlay without filtering the composed
    /// `edges` view by a derived numeric id. SQLite cannot push that predicate
    /// through the Base/Delta UNION and joins, so the generic query scans and
    /// sorts the full Base graph for every BFS node. Qualified endpoint names
    /// are the native indexed keys of overlay edges; constrain each UNION arm
    /// first, then map the small candidate set back to visible node ids.
    fn overlay_edges_for_endpoint(
        &self,
        node_id: i64,
        edge_type: &str,
        limit: usize,
        incoming: bool,
    ) -> Result<Vec<Edge>> {
        let Some(endpoint) = self.get_node(node_id)? else {
            return Ok(Vec::new());
        };
        let sql = if incoming {
            r#"
WITH candidate_edges AS (
    SELECT d.id, d.project, d.source_qualified_name, d.target_qualified_name,
           d.edge_type, d.properties
    FROM main.overlay_edges d
    WHERE d.project = ?1
      AND d.target_qualified_name = ?2
      AND d.edge_type = ?3
    UNION ALL
    SELECT -e.id, e.project, base_source.qualified_name,
           base_target.qualified_name, e.edge_type, e.properties
    FROM greppy_base.nodes base_target
    JOIN greppy_base.edges e ON e.target_id = base_target.id
    JOIN greppy_base.nodes base_source ON base_source.id = e.source_id
    WHERE base_target.project = ?1
      AND base_target.qualified_name = ?2
      AND e.edge_type = ?3
      AND NOT EXISTS (
          SELECT 1 FROM greppy_hidden_paths h
          WHERE h.path = base_source.file_path
      )
      AND NOT EXISTS (
          SELECT 1 FROM main.overlay_edges d
          WHERE d.project = e.project
            AND d.source_qualified_name = base_source.qualified_name
            AND d.target_qualified_name = base_target.qualified_name
            AND d.edge_type = e.edge_type
      )
)
SELECT id, project, source_qualified_name, target_qualified_name,
       edge_type, properties
FROM candidate_edges
ORDER BY id
LIMIT ?4
"#
        } else {
            r#"
WITH candidate_edges AS (
    SELECT d.id, d.project, d.source_qualified_name, d.target_qualified_name,
           d.edge_type, d.properties
    FROM main.overlay_edges d
    WHERE d.project = ?1
      AND d.source_qualified_name = ?2
      AND d.edge_type = ?3
    UNION ALL
    SELECT -e.id, e.project, base_source.qualified_name,
           base_target.qualified_name, e.edge_type, e.properties
    FROM greppy_base.nodes base_source
    JOIN greppy_base.edges e ON e.source_id = base_source.id
    JOIN greppy_base.nodes base_target ON base_target.id = e.target_id
    WHERE base_source.project = ?1
      AND base_source.qualified_name = ?2
      AND e.edge_type = ?3
      AND NOT EXISTS (
          SELECT 1 FROM greppy_hidden_paths h
          WHERE h.path = base_source.file_path
      )
      AND NOT EXISTS (
          SELECT 1 FROM main.overlay_edges d
          WHERE d.project = e.project
            AND d.source_qualified_name = base_source.qualified_name
            AND d.target_qualified_name = base_target.qualified_name
            AND d.edge_type = e.edge_type
      )
)
SELECT id, project, source_qualified_name, target_qualified_name,
       edge_type, properties
FROM candidate_edges
ORDER BY id
LIMIT ?4
"#
        };

        // A hidden endpoint can make a candidate disappear while resolving
        // the composed node view. Fetch a small cushion so the public limit
        // remains useful without ever returning an unbounded candidate set.
        let candidate_limit = limit.saturating_mul(4).max(limit) as i64;
        let candidates = {
            let mut stmt = self.conn().prepare(sql)?;
            let rows = stmt.query_map(
                params![
                    endpoint.project,
                    endpoint.qualified_name,
                    edge_type,
                    candidate_limit
                ],
                |row| {
                    Ok(OverlayEdgeCandidate {
                        id: row.get(0)?,
                        project: row.get(1)?,
                        source_qualified_name: row.get(2)?,
                        target_qualified_name: row.get(3)?,
                        edge_type: row.get(4)?,
                        properties: row.get(5)?,
                    })
                },
            )?;
            rows.collect::<rusqlite::Result<Vec<_>>>()?
        };

        let mut edges = Vec::with_capacity(candidates.len().min(limit));
        for candidate in candidates {
            let Some(source) =
                self.get_node_by_qname(&candidate.project, &candidate.source_qualified_name)?
            else {
                continue;
            };
            let Some(target) =
                self.get_node_by_qname(&candidate.project, &candidate.target_qualified_name)?
            else {
                continue;
            };
            edges.push(Edge {
                id: candidate.id,
                project: candidate.project,
                source_id: source.id,
                target_id: target.id,
                edge_type: candidate.edge_type,
                properties: serde_json::from_str(&candidate.properties)?,
            });
            if edges.len() >= limit {
                break;
            }
        }
        Ok(edges)
    }

    /// Count edges of a given type within a project.
    pub fn count_edges(&self, project: &str, edge_type: Option<&str>) -> Result<i64> {
        let n: i64 = match edge_type {
            Some(t) => self.conn().query_row(
                "SELECT COUNT(*) FROM edges WHERE project = ?1 AND edge_type = ?2",
                params![project, t],
                |row| row.get(0),
            )?,
            None => self.conn().query_row(
                "SELECT COUNT(*) FROM edges WHERE project = ?1",
                params![project],
                |row| row.get(0),
            )?,
        };
        Ok(n)
    }
}

/// Crate-internal re-export of [`row_to_edge`] so sibling modules (e.g.
/// `stats`) can map rows with the identical column ordering without
/// duplicating the mapper.
pub(crate) fn row_to_edge_pub(row: &rusqlite::Row<'_>) -> rusqlite::Result<Edge> {
    row_to_edge(row)
}

fn row_to_edge(row: &rusqlite::Row<'_>) -> rusqlite::Result<Edge> {
    let props_str: String = row.get(5)?;
    let properties: serde_json::Value =
        serde_json::from_str(&props_str).unwrap_or(serde_json::Value::Null);
    Ok(Edge {
        id: row.get(0)?,
        project: row.get(1)?,
        source_id: row.get(2)?,
        target_id: row.get(3)?,
        edge_type: row.get(4)?,
        properties,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::NewNode;
    use crate::project::Project;
    use crate::{StoreView, VisibilityIndex};

    fn setup_graph() -> (Store, i64, i64) {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "2026-06-28T20:00:00Z".into(),
            root_path: "/repos/p".into(),
        })
        .unwrap();
        let a = s
            .insert_node(&NewNode {
                project: "p".into(),
                label: "Function".into(),
                name: "A".into(),
                qualified_name: "p.A".into(),
                file_path: "a.rs".into(),
                start_line: 1,
                end_line: 5,
                properties: serde_json::json!({}),
            })
            .unwrap();
        let b = s
            .insert_node(&NewNode {
                project: "p".into(),
                label: "Function".into(),
                name: "B".into(),
                qualified_name: "p.B".into(),
                file_path: "b.rs".into(),
                start_line: 1,
                end_line: 5,
                properties: serde_json::json!({}),
            })
            .unwrap();
        (s, a, b)
    }

    #[test]
    fn insert_and_get_edge() {
        let (mut s, a, b) = setup_graph();
        let eid = s
            .insert_edge(&NewEdge {
                project: "p".into(),
                source_id: a,
                target_id: b,
                edge_type: "CALLS".into(),
                properties: serde_json::json!({"line": 3}),
            })
            .unwrap();
        let e = s.get_edge(eid).unwrap().unwrap();
        assert_eq!(e.edge_type, "CALLS");
        assert_eq!(e.properties["line"], 3);
    }

    #[test]
    fn upsert_on_triple_collision() {
        let (mut s, a, b) = setup_graph();
        let e1 = s
            .insert_edge(&NewEdge {
                project: "p".into(),
                source_id: a,
                target_id: b,
                edge_type: "CALLS".into(),
                properties: serde_json::json!({"v": 1}),
            })
            .unwrap();
        let e2 = s
            .insert_edge(&NewEdge {
                project: "p".into(),
                source_id: a,
                target_id: b,
                edge_type: "CALLS".into(),
                properties: serde_json::json!({"v": 2}),
            })
            .unwrap();
        assert_eq!(e1, e2, "triple-collision must upsert id");
        assert_eq!(s.get_edge(e2).unwrap().unwrap().properties["v"], 2);
    }

    #[test]
    fn overlay_edge_batch_connects_base_and_delta_nodes() {
        let tmp = tempfile::tempdir().unwrap();
        let base_path = tmp.path().join("base.db");
        let delta_path = tmp.path().join("delta.db");
        {
            let mut base = Store::open(&base_path).unwrap();
            base.upsert_project(&Project {
                name: "p".into(),
                indexed_at: "2026-09-01T00:00:00Z".into(),
                root_path: "/base".into(),
            })
            .unwrap();
            base.insert_node(&NewNode {
                project: "p".into(),
                label: "Function".into(),
                name: "A".into(),
                qualified_name: "p.A".into(),
                file_path: "a.rs".into(),
                start_line: 1,
                end_line: 2,
                properties: serde_json::json!({}),
            })
            .unwrap();
        }
        {
            let mut delta = Store::open(&delta_path).unwrap();
            delta
                .upsert_project(&Project {
                    name: "p".into(),
                    indexed_at: "2026-09-01T00:00:00Z".into(),
                    root_path: "/delta".into(),
                })
                .unwrap();
            delta
                .insert_node(&NewNode {
                    project: "p".into(),
                    label: "Function".into(),
                    name: "B".into(),
                    qualified_name: "p.B".into(),
                    file_path: "b.rs".into(),
                    start_line: 3,
                    end_line: 4,
                    properties: serde_json::json!({}),
                })
                .unwrap();
        }

        let visibility = VisibilityIndex::new(["b.rs".into()], []).unwrap();
        let mut view = StoreView::open_overlay(&base_path, &delta_path, visibility).unwrap();
        let store = view.store_mut();
        let source = store.get_node_by_qname("p", "p.A").unwrap().unwrap();
        let target = store.get_node_by_qname("p", "p.B").unwrap().unwrap();
        let private_nodes = store.list_private_nodes("p", 0, 10).unwrap();
        assert_eq!(
            private_nodes
                .iter()
                .map(|node| node.qualified_name.as_str())
                .collect::<Vec<_>>(),
            vec!["p.B"],
            "Delta-only index phases must not scan immutable Base nodes"
        );
        store
            .insert_overlay_edges(&[NewOverlayEdge {
                project: "p".into(),
                source_qualified_name: source.qualified_name,
                target_qualified_name: target.qualified_name,
                edge_type: "CALLS".into(),
                properties: serde_json::json!({"line": 7}),
            }])
            .unwrap();

        let edges = store.outgoing_edges(source.id, Some("CALLS"), 10).unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].target_id, target.id);
        assert_eq!(edges[0].properties["line"], 7);
        let incoming = store.incoming_edges(target.id, Some("CALLS"), 10).unwrap();
        assert_eq!(incoming.len(), 1);
        assert_eq!(incoming[0].source_id, source.id);
        assert_eq!(incoming[0].properties["line"], 7);
    }

    #[test]
    fn typed_overlay_queries_return_indexed_base_edges() {
        let tmp = tempfile::tempdir().unwrap();
        let base_path = tmp.path().join("base.db");
        let delta_path = tmp.path().join("delta.db");
        {
            let mut base = Store::open(&base_path).unwrap();
            base.upsert_project(&Project {
                name: "p".into(),
                indexed_at: "2026-09-01T00:00:00Z".into(),
                root_path: "/base".into(),
            })
            .unwrap();
            let source = base
                .insert_node(&NewNode {
                    project: "p".into(),
                    label: "Function".into(),
                    name: "A".into(),
                    qualified_name: "p.A".into(),
                    file_path: "a.rs".into(),
                    start_line: 1,
                    end_line: 5,
                    properties: serde_json::json!({}),
                })
                .unwrap();
            let target = base
                .insert_node(&NewNode {
                    project: "p".into(),
                    label: "Function".into(),
                    name: "B".into(),
                    qualified_name: "p.B".into(),
                    file_path: "b.rs".into(),
                    start_line: 1,
                    end_line: 5,
                    properties: serde_json::json!({}),
                })
                .unwrap();
            base.insert_edge(&NewEdge {
                project: "p".into(),
                source_id: source,
                target_id: target,
                edge_type: "CALLS".into(),
                properties: serde_json::json!({"line": 11}),
            })
            .unwrap();
        }
        {
            let mut delta = Store::open(&delta_path).unwrap();
            delta
                .upsert_project(&Project {
                    name: "p".into(),
                    indexed_at: "2026-09-01T00:00:00Z".into(),
                    root_path: "/delta".into(),
                })
                .unwrap();
        }

        let visibility = VisibilityIndex::new(std::iter::empty(), std::iter::empty()).unwrap();
        let mut view = StoreView::open_overlay(&base_path, &delta_path, visibility).unwrap();
        let store = view.store_mut();
        let source = store.get_node_by_qname("p", "p.A").unwrap().unwrap();
        let target = store.get_node_by_qname("p", "p.B").unwrap().unwrap();

        let outgoing = store.outgoing_edges(source.id, Some("CALLS"), 10).unwrap();
        assert_eq!(outgoing.len(), 1);
        assert_eq!(outgoing[0].target_id, target.id);
        assert_eq!(outgoing[0].properties["line"], 11);
        let incoming = store.incoming_edges(target.id, Some("CALLS"), 10).unwrap();
        assert_eq!(incoming.len(), 1);
        assert_eq!(incoming[0].source_id, source.id);
        assert_eq!(incoming[0].properties["line"], 11);
    }

    #[test]
    fn outgoing_and_incoming() {
        let (mut s, a, b) = setup_graph();
        s.insert_edge(&NewEdge {
            project: "p".into(),
            source_id: a,
            target_id: b,
            edge_type: "CALLS".into(),
            properties: serde_json::json!({}),
        })
        .unwrap();
        let out = s.outgoing_edges(a, Some("CALLS"), 10).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].target_id, b);
        let inc = s.incoming_edges(b, Some("CALLS"), 10).unwrap();
        assert_eq!(inc.len(), 1);
        assert_eq!(inc[0].source_id, a);
    }

    #[test]
    fn count_by_type() {
        let (mut s, a, b) = setup_graph();
        s.insert_edge(&NewEdge {
            project: "p".into(),
            source_id: a,
            target_id: b,
            edge_type: "CALLS".into(),
            properties: serde_json::json!({}),
        })
        .unwrap();
        s.insert_edge(&NewEdge {
            project: "p".into(),
            source_id: b,
            target_id: a,
            edge_type: "IMPORTS".into(),
            properties: serde_json::json!({}),
        })
        .unwrap();
        assert_eq!(s.count_edges("p", Some("CALLS")).unwrap(), 1);
        assert_eq!(s.count_edges("p", Some("IMPORTS")).unwrap(), 1);
        assert_eq!(s.count_edges("p", None).unwrap(), 2);
    }
}
