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

impl Store {
    /// Insert an edge. Returns the assigned id. The `(source_id,
    /// target_id, edge_type)` triple is unique; a duplicate insert is
    /// upserted (matches the upstream `UNIQUE(source_id, target_id, type)`
    /// schema).
    pub fn insert_edge(&mut self, e: &NewEdge) -> Result<i64> {
        let props_str = serde_json::to_string(&e.properties)?;
        let tx = self.transaction()?;
        let id: i64 = tx
            .raw()
            .prepare_cached(
                "INSERT INTO edges (project, source_id, target_id, edge_type, properties)
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
