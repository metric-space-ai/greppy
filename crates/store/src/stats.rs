//! Graph statistics + summary computation + additional retrieval helpers.
//!
//! This module ports the *store-only* statistics surface of upstream's
//! `src/store` (the `graph_stats` / `project_summary` family) onto the
//! Rust store. Everything here is read-only aggregation over the existing
//! `nodes` / `edges` / `file_state` tables plus one write path
//! (`compute_and_store_project_summary`) that reuses the
//! `project_summaries` CRUD added earlier.
//!
//! Determinism is a hard requirement: every list result is `ORDER BY`-ed
//! on a total key, and the [`GraphStats`] aggregates use sorted vectors so
//! two runs over the same graph produce byte-identical output (and an
//! identical `source_hash`).

use rusqlite::params;

use crate::edge::{row_to_edge_pub as row_to_edge, Edge};
use crate::file_state::sha256_hex;
use crate::store::Store;
use crate::store_error::Result;

/// One `(label, count)` pair: the number of nodes carrying a given label
/// within a project.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LabelCount {
    pub label: String,
    pub count: i64,
}

/// One `(edge_type, count)` pair: the number of edges of a given type
/// within a project.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EdgeTypeCount {
    pub edge_type: String,
    pub count: i64,
}

/// Deterministic snapshot of a project's graph shape.
///
/// All vectors are sorted on their string key (`label` / `edge_type`) so
/// the struct hashes/serialises identically across runs over the same
/// graph. Totals are stored explicitly so callers never have to re-sum.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GraphStats {
    pub project: String,
    /// `(label, count)` sorted by `label`. Empty when the project has no
    /// nodes.
    pub node_counts_by_label: Vec<LabelCount>,
    /// `(edge_type, count)` sorted by `edge_type`. Empty when the project
    /// has no edges.
    pub edge_counts_by_type: Vec<EdgeTypeCount>,
    /// Distinct files tracked in `file_state` for the project.
    pub file_count: i64,
    /// Sum of all per-label node counts.
    pub total_nodes: i64,
    /// Sum of all per-type edge counts.
    pub total_edges: i64,
}

impl GraphStats {
    /// A stable, human-readable canonical form of the stats. Used to derive
    /// the summary `source_hash` and as the default summary text. The format
    /// is deterministic: sorted keys, fixed separators, no timestamps.
    pub fn canonical_string(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        let _ = writeln!(out, "project={}", self.project);
        let _ = writeln!(out, "files={}", self.file_count);
        let _ = writeln!(out, "nodes={}", self.total_nodes);
        for lc in &self.node_counts_by_label {
            let _ = writeln!(out, "node.{}={}", lc.label, lc.count);
        }
        let _ = writeln!(out, "edges={}", self.total_edges);
        for ec in &self.edge_counts_by_type {
            let _ = writeln!(out, "edge.{}={}", ec.edge_type, ec.count);
        }
        out
    }
}

impl Store {
    /// Node counts grouped by label for a project, sorted by `label`.
    ///
    /// Served by `idx_nodes_label(project, label)`. Deterministic.
    pub fn node_counts_by_label(&self, project: &str) -> Result<Vec<LabelCount>> {
        let mut stmt = self.conn().prepare_cached(
            "SELECT label, COUNT(*) FROM nodes
             WHERE project = ?1
             GROUP BY label
             ORDER BY label",
        )?;
        let rows = stmt
            .query_map(params![project], |row| {
                Ok(LabelCount {
                    label: row.get(0)?,
                    count: row.get(1)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Edge counts grouped by type for a project, sorted by `edge_type`.
    ///
    /// Served by `idx_edges_type(project, edge_type)`. Deterministic.
    pub fn edge_counts_by_type(&self, project: &str) -> Result<Vec<EdgeTypeCount>> {
        let mut stmt = self.conn().prepare_cached(
            "SELECT edge_type, COUNT(*) FROM edges
             WHERE project = ?1
             GROUP BY edge_type
             ORDER BY edge_type",
        )?;
        let rows = stmt
            .query_map(params![project], |row| {
                Ok(EdgeTypeCount {
                    edge_type: row.get(0)?,
                    count: row.get(1)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Number of distinct files tracked in `file_state` for the project.
    pub fn file_count(&self, project: &str) -> Result<i64> {
        let n: i64 = self.conn().query_row(
            "SELECT COUNT(*) FROM file_state WHERE project = ?1",
            params![project],
            |row| row.get(0),
        )?;
        Ok(n)
    }

    /// Compute the full [`GraphStats`] for a project with a small, fixed
    /// number of indexed aggregate queries (one per dimension).
    ///
    /// The result is deterministic: each sub-list is ordered by its key and
    /// the totals are derived by summing the (already grouped) counts, so a
    /// graph that has not changed yields an identical struct every time.
    pub fn stats(&self, project: &str) -> Result<GraphStats> {
        let node_counts_by_label = self.node_counts_by_label(project)?;
        let edge_counts_by_type = self.edge_counts_by_type(project)?;
        let file_count = self.file_count(project)?;
        let total_nodes = node_counts_by_label.iter().map(|c| c.count).sum();
        let total_edges = edge_counts_by_type.iter().map(|c| c.count).sum();
        Ok(GraphStats {
            project: project.to_string(),
            node_counts_by_label,
            edge_counts_by_type,
            file_count,
            total_nodes,
            total_edges,
        })
    }

    /// Compute the current [`GraphStats`] for a project, derive a
    /// deterministic summary + `source_hash` from them, and upsert the
    /// `project_summaries` row so it reflects the current graph.
    ///
    /// The `source_hash` is `sha256(stats.canonical_string())`, so it
    /// changes if and only if the graph's shape changes — a caller can
    /// compare it against the stored row to detect a stale summary without
    /// recomputing the text. The stored `summary` is the canonical string
    /// itself, giving a faithful, deterministic default; richer summary
    /// generation can replace the text later while keeping this hash
    /// contract. Returns the computed [`GraphStats`] and the `source_hash`
    /// that was stored.
    pub fn compute_and_store_project_summary(
        &mut self,
        project: &str,
    ) -> Result<(GraphStats, String)> {
        let stats = self.stats(project)?;
        let canonical = stats.canonical_string();
        let source_hash = sha256_hex(canonical.as_bytes());
        self.upsert_project_summary(project, &canonical, &source_hash)?;
        Ok((stats, source_hash))
    }

    /// List edges of a given type within a project, ordered deterministically
    /// by `(source_id, target_id, id)`, bounded by `limit`.
    ///
    /// Served by `idx_edges_type(project, edge_type)`. Unlike
    /// [`Store::count_edges`], this materialises the rows so search/cli can
    /// render or traverse them.
    pub fn list_edges_by_type(
        &self,
        project: &str,
        edge_type: &str,
        limit: usize,
    ) -> Result<Vec<Edge>> {
        let mut stmt = self.conn().prepare_cached(
            "SELECT id, project, source_id, target_id, edge_type, properties
             FROM edges
             WHERE project = ?1 AND edge_type = ?2
             ORDER BY source_id, target_id, id
             LIMIT ?3",
        )?;
        let rows = stmt
            .query_map(params![project, edge_type, limit as i64], row_to_edge)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// All edges directly between `source_id` and `target_id` (every type),
    /// ordered by `(edge_type, id)`.
    ///
    /// This is the building block for "how are these two symbols connected"
    /// queries. Served by `idx_edges_source(source_id, edge_type)`.
    pub fn get_edges_between(&self, source_id: i64, target_id: i64) -> Result<Vec<Edge>> {
        let mut stmt = self.conn().prepare_cached(
            "SELECT id, project, source_id, target_id, edge_type, properties
             FROM edges
             WHERE source_id = ?1 AND target_id = ?2
             ORDER BY edge_type, id",
        )?;
        let rows = stmt
            .query_map(params![source_id, target_id], row_to_edge)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Neighbour node ids reachable from `node_id` by following outgoing
    /// edges of an optional type, ordered ascending and de-duplicated.
    ///
    /// A node may have several edges to the same target (different types);
    /// this returns each distinct neighbour once. `edge_type = None`
    /// considers every type. Bounded by `limit`.
    pub fn outgoing_neighbors(
        &self,
        node_id: i64,
        edge_type: Option<&str>,
        limit: usize,
    ) -> Result<Vec<i64>> {
        let rows = match edge_type {
            Some(t) => {
                let mut stmt = self.conn().prepare_cached(
                    "SELECT DISTINCT target_id FROM edges
                     WHERE source_id = ?1 AND edge_type = ?2
                     ORDER BY target_id LIMIT ?3",
                )?;
                let out = stmt
                    .query_map(params![node_id, t, limit as i64], |row| row.get(0))?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                out
            }
            None => {
                let mut stmt = self.conn().prepare_cached(
                    "SELECT DISTINCT target_id FROM edges
                     WHERE source_id = ?1
                     ORDER BY target_id LIMIT ?2",
                )?;
                let out = stmt
                    .query_map(params![node_id, limit as i64], |row| row.get(0))?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                out
            }
        };
        Ok(rows)
    }

    /// Neighbour node ids that point *into* `node_id` via incoming edges of
    /// an optional type, ordered ascending and de-duplicated.
    ///
    /// The incoming counterpart of [`Store::outgoing_neighbors`]. Served by
    /// `idx_edges_target(target_id, edge_type)`.
    pub fn incoming_neighbors(
        &self,
        node_id: i64,
        edge_type: Option<&str>,
        limit: usize,
    ) -> Result<Vec<i64>> {
        let rows = match edge_type {
            Some(t) => {
                let mut stmt = self.conn().prepare_cached(
                    "SELECT DISTINCT source_id FROM edges
                     WHERE target_id = ?1 AND edge_type = ?2
                     ORDER BY source_id LIMIT ?3",
                )?;
                let out = stmt
                    .query_map(params![node_id, t, limit as i64], |row| row.get(0))?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                out
            }
            None => {
                let mut stmt = self.conn().prepare_cached(
                    "SELECT DISTINCT source_id FROM edges
                     WHERE target_id = ?1
                     ORDER BY source_id LIMIT ?2",
                )?;
                let out = stmt
                    .query_map(params![node_id, limit as i64], |row| row.get(0))?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                out
            }
        };
        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edge::NewEdge;
    use crate::file_state::FileState;
    use crate::node::NewNode;
    use crate::project::Project;

    /// Seed a small, deterministic graph:
    ///
    /// nodes (by label): 2 Function (`A`, `B`), 1 Struct (`S`)
    /// edges (by type):  A->B CALLS, A->S USES, B->S USES
    /// files:            2 tracked in file_state
    ///
    /// Returns the store plus the ids of A, B, S.
    fn seed(project: &str) -> (Store, i64, i64, i64) {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: project.into(),
            indexed_at: "2026-06-28T20:00:00Z".into(),
            root_path: format!("/repos/{project}"),
        })
        .unwrap();

        let node = |label: &str, qn: &str, file: &str| NewNode {
            project: project.into(),
            label: label.into(),
            name: qn.rsplit('.').next().unwrap_or(qn).into(),
            qualified_name: qn.into(),
            file_path: file.into(),
            start_line: 1,
            end_line: 5,
            properties: serde_json::json!({}),
        };
        let a = s.insert_node(&node("Function", "p.A", "a.rs")).unwrap();
        let b = s.insert_node(&node("Function", "p.B", "b.rs")).unwrap();
        let st = s.insert_node(&node("Struct", "p.S", "b.rs")).unwrap();

        let edge = |src: i64, tgt: i64, ty: &str| NewEdge {
            project: project.into(),
            source_id: src,
            target_id: tgt,
            edge_type: ty.into(),
            properties: serde_json::json!({}),
        };
        s.insert_edge(&edge(a, b, "CALLS")).unwrap();
        s.insert_edge(&edge(a, st, "USES")).unwrap();
        s.insert_edge(&edge(b, st, "USES")).unwrap();

        // Two tracked files.
        for (rel, content) in [("a.rs", b"aaa".as_slice()), ("b.rs", b"bbb".as_slice())] {
            s.upsert_file_state(&FileState {
                project: project.into(),
                rel_path: rel.into(),
                language: "rust".into(),
                sha256: crate::file_state::sha256_hex(content),
                mtime_ns: 0,
                size: content.len() as i64,
                parser_version: "v1".into(),
                extractor_version: "v1".into(),
                last_indexed_generation: 1,
            })
            .unwrap();
        }

        (s, a, b, st)
    }

    #[test]
    fn node_counts_by_label_grouped_and_sorted() {
        let (s, ..) = seed("p");
        let got = s.node_counts_by_label("p").unwrap();
        assert_eq!(
            got,
            vec![
                LabelCount {
                    label: "Function".into(),
                    count: 2
                },
                LabelCount {
                    label: "Struct".into(),
                    count: 1
                },
            ]
        );
    }

    #[test]
    fn edge_counts_by_type_grouped_and_sorted() {
        let (s, ..) = seed("p");
        let got = s.edge_counts_by_type("p").unwrap();
        assert_eq!(
            got,
            vec![
                EdgeTypeCount {
                    edge_type: "CALLS".into(),
                    count: 1
                },
                EdgeTypeCount {
                    edge_type: "USES".into(),
                    count: 2
                },
            ]
        );
    }

    #[test]
    fn file_count_counts_tracked_files() {
        let (s, ..) = seed("p");
        assert_eq!(s.file_count("p").unwrap(), 2);
    }

    #[test]
    fn stats_aggregates_everything_with_totals() {
        let (s, ..) = seed("p");
        let stats = s.stats("p").unwrap();
        assert_eq!(stats.project, "p");
        assert_eq!(stats.total_nodes, 3);
        assert_eq!(stats.total_edges, 3);
        assert_eq!(stats.file_count, 2);
        assert_eq!(stats.node_counts_by_label.len(), 2);
        assert_eq!(stats.edge_counts_by_type.len(), 2);
    }

    #[test]
    fn stats_is_deterministic() {
        let (s, ..) = seed("p");
        let first = s.stats("p").unwrap();
        let second = s.stats("p").unwrap();
        assert_eq!(first, second);
        // Canonical form is stable too.
        assert_eq!(first.canonical_string(), second.canonical_string());
    }

    #[test]
    fn stats_empty_project_is_all_zero() {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "empty".into(),
            indexed_at: "2026-06-28T20:00:00Z".into(),
            root_path: "/repos/empty".into(),
        })
        .unwrap();
        let stats = s.stats("empty").unwrap();
        assert_eq!(stats.total_nodes, 0);
        assert_eq!(stats.total_edges, 0);
        assert_eq!(stats.file_count, 0);
        assert!(stats.node_counts_by_label.is_empty());
        assert!(stats.edge_counts_by_type.is_empty());
    }

    #[test]
    fn stats_is_project_scoped() {
        let (mut s, ..) = seed("p");
        s.upsert_project(&Project {
            name: "other".into(),
            indexed_at: "2026-06-28T20:00:00Z".into(),
            root_path: "/repos/other".into(),
        })
        .unwrap();
        // `other` has no nodes/edges/files.
        let other = s.stats("other").unwrap();
        assert_eq!(other.total_nodes, 0);
        assert_eq!(other.total_edges, 0);
        assert_eq!(other.file_count, 0);
    }

    #[test]
    fn compute_and_store_project_summary_round_trip() {
        let (mut s, ..) = seed("p");
        let (stats, hash) = s.compute_and_store_project_summary("p").unwrap();

        // The stored row reflects the current graph.
        let row = s.get_project_summary("p").unwrap().unwrap();
        assert_eq!(row.project, "p");
        assert_eq!(row.source_hash, hash);
        assert_eq!(row.summary, stats.canonical_string());

        // The hash is exactly sha256(canonical_string) — deterministic.
        let expected = sha256_hex(stats.canonical_string().as_bytes());
        assert_eq!(hash, expected);
    }

    #[test]
    fn summary_hash_changes_only_when_graph_changes() {
        let (mut s, a, b, _st) = seed("p");
        let (_s1, hash1) = s.compute_and_store_project_summary("p").unwrap();

        // Recomputing without graph changes yields the same hash.
        let (_s2, hash2) = s.compute_and_store_project_summary("p").unwrap();
        assert_eq!(hash1, hash2, "stable graph => stable hash");

        // Add an edge => the shape changes => the hash changes.
        s.insert_edge(&NewEdge {
            project: "p".into(),
            source_id: b,
            target_id: a,
            edge_type: "CALLS".into(),
            properties: serde_json::json!({}),
        })
        .unwrap();
        let (_s3, hash3) = s.compute_and_store_project_summary("p").unwrap();
        assert_ne!(hash1, hash3, "graph change => hash change");
    }

    #[test]
    fn list_edges_by_type_ordered_and_bounded() {
        let (s, a, b, st) = seed("p");
        let uses = s.list_edges_by_type("p", "USES", 100).unwrap();
        assert_eq!(uses.len(), 2);
        // Ordered by (source_id, target_id, id): a->st then b->st.
        assert_eq!((uses[0].source_id, uses[0].target_id), (a, st));
        assert_eq!((uses[1].source_id, uses[1].target_id), (b, st));
        assert!(uses.iter().all(|e| e.edge_type == "USES"));

        let calls = s.list_edges_by_type("p", "CALLS", 100).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!((calls[0].source_id, calls[0].target_id), (a, b));

        // Limit is respected.
        assert_eq!(s.list_edges_by_type("p", "USES", 1).unwrap().len(), 1);
        // Unknown type => empty.
        assert!(s.list_edges_by_type("p", "NOPE", 100).unwrap().is_empty());
    }

    #[test]
    fn get_edges_between_returns_only_that_pair() {
        let (mut s, a, b, st) = seed("p");
        // Add a second edge type between a and b.
        s.insert_edge(&NewEdge {
            project: "p".into(),
            source_id: a,
            target_id: b,
            edge_type: "USES".into(),
            properties: serde_json::json!({}),
        })
        .unwrap();

        let between = s.get_edges_between(a, b).unwrap();
        // CALLS and USES, ordered by edge_type.
        assert_eq!(between.len(), 2);
        assert_eq!(between[0].edge_type, "CALLS");
        assert_eq!(between[1].edge_type, "USES");
        assert!(between.iter().all(|e| e.source_id == a && e.target_id == b));

        // a->st has exactly one edge; st->a has none (directed).
        assert_eq!(s.get_edges_between(a, st).unwrap().len(), 1);
        assert!(s.get_edges_between(st, a).unwrap().is_empty());
    }

    #[test]
    fn outgoing_neighbors_distinct_sorted_and_typed() {
        let (mut s, a, b, st) = seed("p");
        // a already points at b (CALLS) and st (USES). Add a duplicate-target
        // edge of another type so DISTINCT is exercised.
        s.insert_edge(&NewEdge {
            project: "p".into(),
            source_id: a,
            target_id: b,
            edge_type: "USES".into(),
            properties: serde_json::json!({}),
        })
        .unwrap();

        // All types: distinct neighbours {b, st}, sorted ascending.
        let all = s.outgoing_neighbors(a, None, 100).unwrap();
        let mut expect = vec![b, st];
        expect.sort_unstable();
        assert_eq!(all, expect, "distinct, sorted neighbours");

        // Typed: only USES targets {b, st}.
        let uses = s.outgoing_neighbors(a, Some("USES"), 100).unwrap();
        let mut expect_uses = vec![b, st];
        expect_uses.sort_unstable();
        assert_eq!(uses, expect_uses);

        // CALLS targets only {b}.
        assert_eq!(
            s.outgoing_neighbors(a, Some("CALLS"), 100).unwrap(),
            vec![b]
        );

        // Limit honoured.
        assert_eq!(s.outgoing_neighbors(a, None, 1).unwrap().len(), 1);
    }

    #[test]
    fn incoming_neighbors_distinct_sorted_and_typed() {
        let (s, a, b, st) = seed("p");
        // st is pointed at by a and b via USES.
        let into_st = s.incoming_neighbors(st, Some("USES"), 100).unwrap();
        let mut expect = vec![a, b];
        expect.sort_unstable();
        assert_eq!(into_st, expect);

        // b is pointed at by a via CALLS.
        assert_eq!(s.incoming_neighbors(b, None, 100).unwrap(), vec![a]);

        // a has no incoming edges.
        assert!(s.incoming_neighbors(a, None, 100).unwrap().is_empty());
    }
}
