//! Node CRUD: insert, fetch by id, fetch by qualified_name, list by label.

use rusqlite::{params, OptionalExtension};

use crate::fts;
use crate::store::Store;
use crate::store_error::{Error, Result};

/// One row of the `nodes` table plus its parsed JSON properties.
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub id: i64,
    pub project: String,
    pub label: String,
    pub name: String,
    pub qualified_name: String,
    pub file_path: String,
    pub start_line: i64,
    pub end_line: i64,
    pub properties: serde_json::Value,
}

/// Input for inserting a new node. `id` is filled by SQLite on insert.
#[derive(Debug, Clone)]
pub struct NewNode {
    pub project: String,
    pub label: String,
    pub name: String,
    pub qualified_name: String,
    pub file_path: String,
    pub start_line: i64,
    pub end_line: i64,
    pub properties: serde_json::Value,
}

/// The FTS5 token tuple a node contributes to the contentless
/// `nodes_fts` table: `(name_col, qualified_name_col)`. `nodes_fts` is
/// contentless, so the FTS5 `'delete'` command must be handed back the
/// *exact same* column values that were inserted; computing them in one
/// place guarantees insert and delete agree (the previous bug passed
/// empty strings on delete and leaked every posting — P0).
///
/// The `label` and `file_path` columns are stored verbatim, so they are
/// not derived here; callers pass the raw values straight through.
fn fts_tokens(name: &str, qualified_name: &str) -> (String, String) {
    let tokens = fts::camel_split(name);
    let qtokens = fts::camel_split(qualified_name);
    let combined = format!("{tokens} {qtokens}");
    (combined, qtokens)
}

/// Insert/upsert one node row and its contentless-FTS postings inside an
/// already-open transaction. Shared by `insert_node` (one-shot) and
/// `insert_nodes` (batched) so both produce identical rows + FTS tokens.
///
/// FTS5 correctness (P0): `nodes_fts` is contentless, so the only way to
/// remove a posting is the `'delete'` command fed the EXACT token values
/// that were inserted. An `ON CONFLICT(project, qualified_name)` upsert
/// reuses the existing rowid, so if we did not first remove the row's
/// prior posting the contentless index would carry both the old and new
/// tokens for that rowid — the same leak that corrupted the index via the
/// empty-string deletes. We therefore read the row's CURRENT stored
/// columns BEFORE the upsert and, when it exists, issue the FTS `'delete'`
/// with those exact (old) values; then we upsert and insert the fresh
/// posting. On a plain insert there is no prior row, so no delete runs.
fn insert_node_in_tx(tx: &rusqlite::Transaction<'_>, n: &NewNode) -> Result<i64> {
    // Prune the prior posting for an in-place upsert using the row's
    // existing (old) column values — the only values that match what was
    // last inserted into the contentless index for this rowid.
    let prior: Option<(i64, String, String, String, String)> = tx
        .prepare_cached(
            "SELECT id, name, qualified_name, label, file_path
             FROM nodes WHERE project = ?1 AND qualified_name = ?2",
        )?
        .query_row(params![n.project, n.qualified_name], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
            ))
        })
        .optional()?;
    if let Some((old_id, old_name, old_qname, old_label, old_file)) = &prior {
        let (old_name_col, old_qname_col) = fts_tokens(old_name, old_qname);
        tx.execute(
            "INSERT INTO nodes_fts(nodes_fts, rowid, name, qualified_name, label, file_path)
             VALUES('delete', ?1, ?2, ?3, ?4, ?5)",
            params![old_id, old_name_col, old_qname_col, old_label, old_file],
        )
        .map_err(Error::Sqlite)?;
    }

    let props_str = serde_json::to_string(&n.properties)?;
    let id: i64 = tx
        .prepare_cached(
            "INSERT INTO nodes (project, label, name, qualified_name, file_path, start_line, end_line, properties)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
             ON CONFLICT(project, qualified_name) DO UPDATE SET
               label = excluded.label,
               name = excluded.name,
               file_path = excluded.file_path,
               start_line = excluded.start_line,
               end_line = excluded.end_line,
               properties = excluded.properties
             RETURNING id",
        )?
        .query_row(
            params![
                n.project,
                n.label,
                n.name,
                n.qualified_name,
                n.file_path,
                n.start_line,
                n.end_line,
                props_str,
            ],
            |row| row.get(0),
        )
        .map_err(Error::Sqlite)?;

    // Insert the fresh posting with the REAL (camel_split) token values,
    // captured once so the matching delete in `delete_node` /
    // `delete_nodes_for_file` can reproduce them exactly.
    let (name_col, qname_col) = fts_tokens(&n.name, &n.qualified_name);
    tx.execute(
        "INSERT INTO nodes_fts(rowid, name, qualified_name, label, file_path) VALUES (?1, ?2, ?3, ?4, ?5)",
        params![id, name_col, qname_col, n.label, n.file_path],
    )
    .map_err(Error::Sqlite)?;
    Ok(id)
}

impl Store {
    /// Insert a new node. Returns the assigned id.
    pub fn insert_node(&mut self, n: &NewNode) -> Result<i64> {
        let tx = self.transaction()?;
        let id = insert_node_in_tx(tx.raw(), n)?;
        tx.commit()?;
        Ok(id)
    }

    /// Insert (or upsert) many nodes inside a SINGLE transaction.
    ///
    /// P1 (re-review, fsync DoS): `insert_node` opens + commits its own
    /// transaction per call, so indexing a file with N symbols did N
    /// fsyncs — pathologically slow on large files. This batched path
    /// opens one transaction, reuses one prepared statement per table,
    /// and commits once, so a whole file's nodes cost a single fsync.
    ///
    /// Semantics are byte-for-byte those of calling `insert_node` once
    /// per element in order: the same `ON CONFLICT(project,
    /// qualified_name)` upsert, the same contentless-FTS token writes,
    /// and the same per-row id assignment. Returns the assigned ids in
    /// input order (the upserted id for rows that collided on
    /// `qualified_name`). Determinism + per-file delete-then-insert
    /// (R-018) are unaffected — the caller still deletes first, then
    /// calls this with the file's fresh nodes.
    pub fn insert_nodes(&mut self, nodes: &[NewNode]) -> Result<Vec<i64>> {
        if nodes.is_empty() {
            return Ok(Vec::new());
        }
        let tx = self.transaction()?;
        let mut ids = Vec::with_capacity(nodes.len());
        for n in nodes {
            ids.push(insert_node_in_tx(tx.raw(), n)?);
        }
        tx.commit()?;
        Ok(ids)
    }

    /// Fetch a node by primary key.
    pub fn get_node(&self, id: i64) -> Result<Option<Node>> {
        let row = self
            .conn()
            .query_row(
                "SELECT id, project, label, name, qualified_name, file_path, start_line, end_line, properties
                 FROM nodes WHERE id = ?1",
                params![id],
                row_to_node,
            )
            .optional()?;
        Ok(row)
    }

    /// Fetch a node by `(project, qualified_name)`.
    pub fn get_node_by_qname(&self, project: &str, qname: &str) -> Result<Option<Node>> {
        let row = self
            .conn()
            .query_row(
                "SELECT id, project, label, name, qualified_name, file_path, start_line, end_line, properties
                 FROM nodes WHERE project = ?1 AND qualified_name = ?2",
                params![project, qname],
                row_to_node,
            )
            .optional()?;
        Ok(row)
    }

    /// List nodes by `(project, label)`. Empty label lists all labels for
    /// the project.
    pub fn list_nodes_by_label(
        &self,
        project: &str,
        label: &str,
        limit: usize,
    ) -> Result<Vec<Node>> {
        let mut stmt = self.conn().prepare(
            "SELECT id, project, label, name, qualified_name, file_path, start_line, end_line, properties
             FROM nodes WHERE project = ?1 AND label = ?2
             ORDER BY qualified_name LIMIT ?3",
        )?;
        let rows = stmt
            .query_map(params![project, label, limit as i64], row_to_node)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// List nodes by `(project, name)`, ordered by `qualified_name`.
    ///
    /// This is the foundation for replacing the resolver's
    /// O(edges*nodes) full scan: given a callee/identifier name it returns
    /// the candidate definition nodes directly, backed by the
    /// `idx_nodes_name` index on `nodes(project, name)`. A name with no
    /// match returns an empty vec.
    pub fn list_nodes_by_name(&self, project: &str, name: &str, limit: usize) -> Result<Vec<Node>> {
        let mut stmt = self.conn().prepare_cached(
            "SELECT id, project, label, name, qualified_name, file_path, start_line, end_line, properties
             FROM nodes WHERE project = ?1 AND name = ?2
             ORDER BY qualified_name LIMIT ?3",
        )?;
        let rows = stmt
            .query_map(params![project, name, limit as i64], row_to_node)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Count nodes matching `(project, name)`. Useful for the resolver to
    /// detect ambiguity (>1 candidate) cheaply before materialising rows.
    pub fn count_nodes_by_name(&self, project: &str, name: &str) -> Result<i64> {
        let n: i64 = self.conn().query_row(
            "SELECT COUNT(*) FROM nodes WHERE project = ?1 AND name = ?2",
            params![project, name],
            |row| row.get(0),
        )?;
        Ok(n)
    }

    /// Look up nodes by `(project, name)` and, for each, its outgoing
    /// edges of an optional type. Convenience for search/resolver paths
    /// that want a symbol plus what it points at in one call.
    ///
    /// Returns `(node, outgoing_edges)` pairs. `edge_type = None` returns
    /// edges of every type. `edge_limit` bounds the edges fetched per node.
    pub fn nodes_by_name_with_outgoing_edges(
        &self,
        project: &str,
        name: &str,
        edge_type: Option<&str>,
        node_limit: usize,
        edge_limit: usize,
    ) -> Result<Vec<(Node, Vec<crate::edge::Edge>)>> {
        let nodes = self.list_nodes_by_name(project, name, node_limit)?;
        let mut out = Vec::with_capacity(nodes.len());
        for n in nodes {
            let edges = self.outgoing_edges(n.id, edge_type, edge_limit)?;
            out.push((n, edges));
        }
        Ok(out)
    }

    /// Paginated, filtered node listing for search / CLI surfaces.
    ///
    /// Lists nodes for `project`, optionally narrowed by `label` and/or
    /// `file` (an empty `&str` for either means "do not filter on that
    /// field"). Results are ordered deterministically by `qualified_name`
    /// then `id` so a stable `(offset, limit)` window pages through the same
    /// total order on every call — the property a paging UI relies on.
    ///
    /// The query is fully parameterised (no string interpolation of user
    /// input). Passing both filters empty lists the whole project, paged.
    pub fn list_nodes(
        &self,
        project: &str,
        label: &str,
        file: &str,
        offset: usize,
        limit: usize,
    ) -> Result<Vec<Node>> {
        // A single parameterised statement that treats an empty filter as
        // "match all" via `(?n = '' OR col = ?n)`. Keeping one SQL shape
        // (rather than branching) means the prepared-statement cache reuses
        // it for every filter combination and the plan stays deterministic.
        let mut stmt = self.conn().prepare_cached(
            "SELECT id, project, label, name, qualified_name, file_path, start_line, end_line, properties
             FROM nodes
             WHERE project = ?1
               AND (?2 = '' OR label = ?2)
               AND (?3 = '' OR file_path = ?3)
             ORDER BY qualified_name, id
             LIMIT ?4 OFFSET ?5",
        )?;
        let rows = stmt
            .query_map(
                params![project, label, file, limit as i64, offset as i64],
                row_to_node,
            )?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Count nodes matching the same `(project, label, file)` filter as
    /// [`Store::list_nodes`] (empty `label`/`file` means "do not filter").
    /// Useful for a paging UI that needs the total to compute page counts.
    pub fn count_nodes(&self, project: &str, label: &str, file: &str) -> Result<i64> {
        let n: i64 = self.conn().query_row(
            "SELECT COUNT(*) FROM nodes
             WHERE project = ?1
               AND (?2 = '' OR label = ?2)
               AND (?3 = '' OR file_path = ?3)",
            params![project, label, file],
            |row| row.get(0),
        )?;
        Ok(n)
    }

    /// Count nodes for `(project, label)`.
    pub fn count_nodes_by_label(&self, project: &str, label: &str) -> Result<i64> {
        let n: i64 = self.conn().query_row(
            "SELECT COUNT(*) FROM nodes WHERE project = ?1 AND label = ?2",
            params![project, label],
            |row| row.get(0),
        )?;
        Ok(n)
    }

    /// Delete a node by id. Cascades to edges via the FK constraint.
    pub fn delete_node(&mut self, id: i64) -> Result<()> {
        let tx = self.transaction()?;
        // P0: `nodes_fts` is contentless, so the FTS5 `'delete'` command
        // must receive the SAME token values that were inserted for this
        // rowid — passing empty strings (the old bug) leaves the postings
        // behind and eventually corrupts the index. We read the row's
        // stored columns and recompute the identical camel_split tokens
        // BEFORE deleting the row.
        let row: Option<(String, String, String, String)> = tx
            .raw()
            .prepare_cached(
                "SELECT name, qualified_name, label, file_path FROM nodes WHERE id = ?1",
            )?
            .query_row(params![id], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                ))
            })
            .optional()?;
        if let Some((name, qname, label, file_path)) = row {
            let (name_col, qname_col) = fts_tokens(&name, &qname);
            tx.raw()
                .execute(
                    "INSERT INTO nodes_fts(nodes_fts, rowid, name, qualified_name, label, file_path)
                     VALUES('delete', ?1, ?2, ?3, ?4, ?5)",
                    params![id, name_col, qname_col, label, file_path],
                )
                .map_err(Error::Sqlite)?;
        }
        tx.raw()
            .execute(
                "DELETE FROM vector_embeddings WHERE node_id = ?1",
                params![id],
            )
            .map_err(Error::Sqlite)?;
        tx.raw()
            .execute("DELETE FROM nodes WHERE id = ?1", params![id])
            .map_err(Error::Sqlite)?;
        tx.commit()?;
        Ok(())
    }

    /// Delete every node for `(project, file_path)` and return the
    /// number of rows removed. Edges whose `source_id` or `target_id`
    /// points to those nodes are removed by the FK cascade.
    ///
    /// R-018 / WP-R018: stale nodes were never removed on re-index,
    /// so a renamed symbol persisted across runs. The indexer now
    /// calls this per file before re-inserting.
    pub fn delete_nodes_for_file(&mut self, project: &str, file_path: &str) -> Result<usize> {
        // Collect each row's id PLUS the columns needed to reproduce its
        // FTS tokens. P0: a contentless FTS5 `'delete'` must be fed the
        // exact token values that were inserted; the old code passed empty
        // strings, leaking every posting and eventually corrupting the
        // index. We recompute the identical camel_split tokens from the
        // row's stored name/qualified_name and pass the verbatim
        // label/file_path.
        let rows: Vec<(i64, String, String, String, String)> = {
            let mut stmt = self.conn().prepare_cached(
                "SELECT id, name, qualified_name, label, file_path
                 FROM nodes WHERE project = ?1 AND file_path = ?2",
            )?;
            let collected = stmt
                .query_map(params![project, file_path], |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                    ))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            collected
        };
        if rows.is_empty() {
            let _ = self.delete_vector_embeddings_for_file(project, file_path)?;
            return Ok(0);
        }
        let tx = self.transaction()?;
        for (id, name, qname, label, fpath) in &rows {
            let (name_col, qname_col) = fts_tokens(name, qname);
            tx.raw()
                .execute(
                    "INSERT INTO nodes_fts(nodes_fts, rowid, name, qualified_name, label, file_path)
                     VALUES('delete', ?1, ?2, ?3, ?4, ?5)",
                    params![id, name_col, qname_col, label, fpath],
                )
                .map_err(Error::Sqlite)?;
        }
        tx.raw()
            .execute(
                "DELETE FROM vector_embeddings WHERE project = ?1 AND file_path = ?2",
                params![project, file_path],
            )
            .map_err(Error::Sqlite)?;
        let n = tx
            .raw()
            .execute(
                "DELETE FROM nodes WHERE project = ?1 AND file_path = ?2",
                params![project, file_path],
            )
            .map_err(Error::Sqlite)?;
        tx.commit()?;
        Ok(n)
    }
}

fn row_to_node(row: &rusqlite::Row<'_>) -> rusqlite::Result<Node> {
    let props_str: String = row.get(8)?;
    let properties: serde_json::Value =
        serde_json::from_str(&props_str).unwrap_or(serde_json::Value::Null);
    Ok(Node {
        id: row.get(0)?,
        project: row.get(1)?,
        label: row.get(2)?,
        name: row.get(3)?,
        qualified_name: row.get(4)?,
        file_path: row.get(5)?,
        start_line: row.get(6)?,
        end_line: row.get(7)?,
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

    fn new_node(project: &str, label: &str, qname: &str) -> NewNode {
        NewNode {
            project: project.into(),
            label: label.into(),
            name: qname.rsplit('.').next().unwrap_or(qname).into(),
            qualified_name: qname.into(),
            file_path: "src/lib.rs".into(),
            start_line: 1,
            end_line: 10,
            properties: serde_json::json!({"kind": "function"}),
        }
    }

    #[test]
    fn insert_then_get_round_trip() {
        let mut s = store_with_project("demo");
        let id = s
            .insert_node(&new_node("demo", "Function", "demo.Foo"))
            .unwrap();
        let n = s.get_node(id).unwrap().unwrap();
        assert_eq!(n.qualified_name, "demo.Foo");
        assert_eq!(n.label, "Function");
        assert_eq!(n.properties["kind"], "function");
    }

    #[test]
    fn upsert_on_qname_collision() {
        let mut s = store_with_project("demo");
        let id1 = s
            .insert_node(&new_node("demo", "Function", "demo.Foo"))
            .unwrap();
        let id2 = s
            .insert_node(&new_node("demo", "Method", "demo.Foo"))
            .unwrap();
        assert_eq!(id1, id2, "ON CONFLICT must preserve id");
        let n = s.get_node(id2).unwrap().unwrap();
        assert_eq!(n.label, "Method", "label should be updated");
    }

    #[test]
    fn list_by_label_returns_only_matching() {
        let mut s = store_with_project("p");
        s.insert_node(&new_node("p", "Function", "p.A")).unwrap();
        s.insert_node(&new_node("p", "Function", "p.B")).unwrap();
        s.insert_node(&new_node("p", "Class", "p.C")).unwrap();
        let functions = s.list_nodes_by_label("p", "Function", 100).unwrap();
        assert_eq!(functions.len(), 2);
        let classes = s.list_nodes_by_label("p", "Class", 100).unwrap();
        assert_eq!(classes.len(), 1);
        assert_eq!(s.count_nodes_by_label("p", "Function").unwrap(), 2);
    }

    #[test]
    fn list_nodes_paginates_deterministically() {
        let mut s = store_with_project("p");
        // 5 functions across two files plus one class, to exercise filters.
        for q in ["p.a", "p.b", "p.c", "p.d", "p.e"] {
            let mut n = new_node("p", "Function", q);
            n.file_path = "src/lib.rs".into();
            s.insert_node(&n).unwrap();
        }
        let mut other = new_node("p", "Function", "p.z");
        other.file_path = "src/other.rs".into();
        s.insert_node(&other).unwrap();
        s.insert_node(&new_node("p", "Class", "p.K")).unwrap();

        // Whole project, paged: window of 2 starting at offset 2 must be a
        // contiguous slice of the full qualified_name order.
        let all = s.list_nodes("p", "", "", 0, 100).unwrap();
        let page = s.list_nodes("p", "", "", 2, 2).unwrap();
        assert_eq!(page.len(), 2);
        assert_eq!(
            page.iter().map(|n| &n.qualified_name).collect::<Vec<_>>(),
            all[2..4]
                .iter()
                .map(|n| &n.qualified_name)
                .collect::<Vec<_>>()
        );

        // Offset past the end returns empty, not an error.
        assert!(s.list_nodes("p", "", "", 1000, 10).unwrap().is_empty());

        // Label filter.
        let funcs = s.list_nodes("p", "Function", "", 0, 100).unwrap();
        assert_eq!(funcs.len(), 6);
        assert!(funcs.iter().all(|n| n.label == "Function"));
        assert_eq!(s.count_nodes("p", "Function", "").unwrap(), 6);

        // File filter: 5 functions + the class default to src/lib.rs.
        let in_lib = s.list_nodes("p", "", "src/lib.rs", 0, 100).unwrap();
        assert_eq!(in_lib.len(), 6);
        assert!(in_lib.iter().all(|n| n.file_path == "src/lib.rs"));

        // Both filters together.
        let lib_funcs = s.list_nodes("p", "Function", "src/lib.rs", 0, 100).unwrap();
        assert_eq!(lib_funcs.len(), 5);
        assert_eq!(s.count_nodes("p", "Function", "src/lib.rs").unwrap(), 5);

        // No filters counts everything in the project.
        assert_eq!(s.count_nodes("p", "", "").unwrap(), 7);
    }

    #[test]
    fn list_nodes_paging_covers_all_rows_without_gaps_or_dupes() {
        let mut s = store_with_project("p");
        for i in 0..10 {
            s.insert_node(&new_node("p", "Function", &format!("p.f{i:02}")))
                .unwrap();
        }
        // Walk the project in pages of 3 and reassemble; must equal the full
        // ordered list exactly (no gaps, no duplicates).
        let full = s.list_nodes("p", "", "", 0, 100).unwrap();
        let mut paged = Vec::new();
        let mut offset = 0;
        loop {
            let page = s.list_nodes("p", "", "", offset, 3).unwrap();
            if page.is_empty() {
                break;
            }
            offset += page.len();
            paged.extend(page);
        }
        assert_eq!(
            paged.iter().map(|n| &n.qualified_name).collect::<Vec<_>>(),
            full.iter().map(|n| &n.qualified_name).collect::<Vec<_>>()
        );
    }

    #[test]
    fn list_nodes_is_project_scoped() {
        let mut s = store_with_project("p1");
        s.upsert_project(&Project {
            name: "p2".into(),
            indexed_at: "2026-06-28T20:00:00Z".into(),
            root_path: "/repos/p2".into(),
        })
        .unwrap();
        s.insert_node(&new_node("p1", "Function", "p1.a")).unwrap();
        s.insert_node(&new_node("p2", "Function", "p2.a")).unwrap();
        let p1 = s.list_nodes("p1", "", "", 0, 100).unwrap();
        assert_eq!(p1.len(), 1);
        assert_eq!(p1[0].project, "p1");
    }

    #[test]
    fn list_by_name_returns_only_matching_rows() {
        let mut s = store_with_project("p");
        // Two distinct symbols named `helper` in different modules, plus a
        // differently-named symbol that must be excluded.
        s.insert_node(&new_node("p", "Function", "p.a.helper"))
            .unwrap();
        s.insert_node(&new_node("p", "Function", "p.b.helper"))
            .unwrap();
        s.insert_node(&new_node("p", "Function", "p.other"))
            .unwrap();

        let helpers = s.list_nodes_by_name("p", "helper", 100).unwrap();
        assert_eq!(helpers.len(), 2);
        assert!(helpers.iter().all(|n| n.name == "helper"));
        // Ordered by qualified_name.
        assert_eq!(helpers[0].qualified_name, "p.a.helper");
        assert_eq!(helpers[1].qualified_name, "p.b.helper");

        assert_eq!(s.count_nodes_by_name("p", "helper").unwrap(), 2);
        assert_eq!(s.count_nodes_by_name("p", "other").unwrap(), 1);
        assert_eq!(s.count_nodes_by_name("p", "missing").unwrap(), 0);
        assert!(s
            .list_nodes_by_name("p", "missing", 100)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn list_by_name_is_project_scoped() {
        let mut s = store_with_project("p1");
        s.upsert_project(&crate::project::Project {
            name: "p2".into(),
            indexed_at: "2026-06-28T20:00:00Z".into(),
            root_path: "/repos/p2".into(),
        })
        .unwrap();
        s.insert_node(&new_node("p1", "Function", "p1.helper"))
            .unwrap();
        s.insert_node(&new_node("p2", "Function", "p2.helper"))
            .unwrap();
        // Same name, different project: each project sees only its own.
        assert_eq!(s.list_nodes_by_name("p1", "helper", 100).unwrap().len(), 1);
        assert_eq!(s.count_nodes_by_name("p2", "helper").unwrap(), 1);
        assert_eq!(
            s.list_nodes_by_name("p1", "helper", 100).unwrap()[0].project,
            "p1"
        );
    }

    #[test]
    fn list_by_name_respects_limit() {
        let mut s = store_with_project("p");
        for i in 0..5 {
            s.insert_node(&new_node("p", "Function", &format!("p.m{i}.dup")))
                .unwrap();
        }
        assert_eq!(s.count_nodes_by_name("p", "dup").unwrap(), 5);
        assert_eq!(s.list_nodes_by_name("p", "dup", 3).unwrap().len(), 3);
    }

    #[test]
    fn list_by_name_uses_the_index() {
        // Guard the perf foundation: the by-name lookup must be served by
        // an index, never a full table scan — that is the whole point of
        // replacing the resolver's O(edges*nodes) scan. EXPLAIN QUERY PLAN
        // names what it uses; a full scan would read "SCAN nodes" with no
        // index. We populate rows + ANALYZE so the planner has the stats
        // to prefer the more selective idx_nodes_name(project, name).
        let mut s = store_with_project("p");
        for i in 0..200 {
            // Many distinct names, a handful sharing "helper", so name is
            // far more selective than project alone.
            let qn = if i % 50 == 0 {
                format!("p.m{i}.helper")
            } else {
                format!("p.m{i}.fn{i}")
            };
            s.insert_node(&new_node("p", "Function", &qn)).unwrap();
        }
        s.conn().execute_batch("ANALYZE").unwrap();

        let plan: Vec<String> = s
            .conn()
            .prepare(
                "EXPLAIN QUERY PLAN
                 SELECT id FROM nodes WHERE project = ?1 AND name = ?2
                 ORDER BY qualified_name LIMIT ?3",
            )
            .unwrap()
            .query_map(params!["p", "helper", 10_i64], |r| r.get::<_, String>(3))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        let joined = plan.join(" | ");
        // Must use an index, and specifically one covering `name`.
        assert!(
            joined.contains("idx_nodes_name"),
            "by-name lookup should use idx_nodes_name; plan was: {joined}"
        );
        // Defensive: never a bare full-table scan.
        assert!(
            !joined.contains("SCAN nodes\n") && !joined.ends_with("SCAN nodes"),
            "by-name lookup must not full-scan; plan was: {joined}"
        );
    }

    #[test]
    fn nodes_by_name_with_outgoing_edges_pairs_node_and_edges() {
        use crate::edge::NewEdge;
        let mut s = store_with_project("p");
        let caller = s
            .insert_node(&new_node("p", "Function", "p.caller"))
            .unwrap();
        let callee = s
            .insert_node(&new_node("p", "Function", "p.do_it"))
            .unwrap();
        s.insert_edge(&NewEdge {
            project: "p".into(),
            source_id: caller,
            target_id: callee,
            edge_type: "CALLS".into(),
            properties: serde_json::json!({}),
        })
        .unwrap();

        let pairs = s
            .nodes_by_name_with_outgoing_edges("p", "caller", Some("CALLS"), 10, 10)
            .unwrap();
        assert_eq!(pairs.len(), 1);
        let (node, edges) = &pairs[0];
        assert_eq!(node.qualified_name, "p.caller");
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].target_id, callee);

        // The callee has no outgoing CALLS edge.
        let callee_pairs = s
            .nodes_by_name_with_outgoing_edges("p", "do_it", Some("CALLS"), 10, 10)
            .unwrap();
        assert_eq!(callee_pairs.len(), 1);
        assert!(callee_pairs[0].1.is_empty());
    }

    #[test]
    fn delete_cascades_via_fk_when_db_enforces_it() {
        // Note: cascades to edges require edges to be inserted first; here
        // we just verify the node row goes away.
        let mut s = store_with_project("p");
        let id = s.insert_node(&new_node("p", "Function", "p.A")).unwrap();
        s.delete_node(id).unwrap();
        assert!(s.get_node(id).unwrap().is_none());
    }

    /// Every rowid the FTS index will return for `query`, as a sorted vec.
    /// For a contentless table there is no `SELECT * FROM nodes_fts`, so we
    /// drive a broad MATCH and read back the rowids — the same rowids a
    /// real `search-symbols` would surface. An orphan posting shows up here
    /// as a rowid with no corresponding `nodes` row.
    fn fts_match_rowids(s: &Store, query: &str) -> Vec<i64> {
        let mut stmt = s
            .conn()
            .prepare("SELECT rowid FROM nodes_fts WHERE nodes_fts MATCH ?1")
            .unwrap();
        let mut ids: Vec<i64> = stmt
            .query_map(params![query], |r| r.get::<_, i64>(0))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    fn live_node_count(s: &Store) -> i64 {
        s.conn()
            .query_row("SELECT count(*) FROM nodes", [], |r| r.get(0))
            .unwrap()
    }

    /// P0 (re-review): renaming a symbol across many reindex cycles must
    /// NOT leak FTS postings. Before the fix, `delete_node` /
    /// `delete_nodes_for_file` passed empty strings to the contentless
    /// FTS5 `'delete'`, so every old token survived as an orphan posting;
    /// after a few cycles `nodes_fts` corrupted and `search-symbols`
    /// exited 73 ("database disk image is malformed") while
    /// `integrity_check` still said `ok`.
    ///
    /// This reproduces the indexer's per-file rename sequence: index a
    /// 1-fn file, then rename the fn across 6 reindex cycles via the
    /// delete-then-insert path the indexer uses (R-018). We assert there
    /// are NO orphan rowids (every rowid the FTS matches exists in
    /// `nodes`, and the distinct-matched-rowid count equals the live node
    /// count), that a search for the live symbol returns exactly it, and
    /// that both `integrity_check` and a prefix MATCH succeed.
    #[test]
    fn rename_across_reindex_cycles_leaves_no_orphan_fts_rows() {
        let mut s = store_with_project("p");
        let file = "src/lib.rs";

        // Track every name we ever used so we can query the FTS for ALL of
        // them at once and prove only the live one survives.
        let mut all_names: Vec<String> = Vec::new();

        let mut live_qname = String::new();
        let mut live_name = String::new();
        for cycle in 0..6 {
            // R-018: per-file delete-then-insert. The indexer deletes all
            // of a file's nodes, then inserts the fresh set.
            s.delete_nodes_for_file("p", file).unwrap();

            live_name = format!("processOrderV{cycle}");
            live_qname = format!("{file}::Function::{live_name}");
            all_names.push(live_name.clone());

            let n = NewNode {
                project: "p".into(),
                label: "Function".into(),
                name: live_name.clone(),
                qualified_name: live_qname.clone(),
                file_path: file.into(),
                start_line: 1,
                end_line: 5,
                properties: serde_json::json!({"kind": "function"}),
            };
            s.insert_nodes(std::slice::from_ref(&n)).unwrap();

            // Invariant after EVERY cycle: exactly one live node, and the
            // FTS index has no orphan postings.
            assert_eq!(
                live_node_count(&s),
                1,
                "exactly one live node after cycle {cycle}"
            );

            // A broad query covering every name we have ever used must
            // match ONLY rowids that still exist in `nodes`.
            let broad: String = all_names
                .iter()
                .flat_map(|nm| {
                    crate::fts::camel_split(nm)
                        .split_whitespace()
                        .map(str::to_string)
                        .collect::<Vec<_>>()
                })
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>()
                .join(" OR ");
            let matched = fts_match_rowids(&s, &broad);
            for rid in &matched {
                assert!(
                    s.get_node(*rid).unwrap().is_some(),
                    "FTS matched orphan rowid {rid} (no live node) after cycle {cycle}; \
                     matched={matched:?}"
                );
            }
            assert_eq!(
                matched.len() as i64,
                live_node_count(&s),
                "distinct FTS rowids must equal live node count after cycle {cycle} \
                 (orphan leak); matched={matched:?}"
            );
        }

        // The integrity check must still pass (it did even when corrupt
        // before — but it must not regress here).
        s.integrity_check().unwrap();

        // A prefix MATCH query (the form `search_fts` issues) must succeed
        // and return exactly the live symbol.
        let hits = crate::fts::search_fts(&s, &live_name, 10).unwrap();
        assert_eq!(
            hits.len(),
            1,
            "search must return exactly the live symbol, got {hits:?}"
        );
        let hit_node = s.get_node(hits[0].node_id).unwrap().unwrap();
        assert_eq!(hit_node.qualified_name, live_qname);

        // And an OLD name must no longer match anything (its postings were
        // pruned, not leaked).
        let stale = crate::fts::search_fts(&s, "processOrderV0", 10).unwrap();
        // V0's tokens (`process`, `order`) overlap the live V5 symbol, so a
        // token search still hits the live node — but never a deleted rowid.
        for h in &stale {
            assert!(
                s.get_node(h.node_id).unwrap().is_some(),
                "stale-name search must never surface a deleted rowid"
            );
        }
    }

    /// In-place upsert (same qualified_name, changing `name`/`label`) must
    /// also keep the contentless FTS index leak-free: the prior posting is
    /// pruned with the row's OLD token values before the new one is
    /// written. Without that prune the old `label`/`name` tokens would
    /// linger for the reused rowid.
    #[test]
    fn upsert_in_place_does_not_leak_old_fts_tokens() {
        let mut s = store_with_project("p");
        let id = s
            .insert_node(&NewNode {
                project: "p".into(),
                label: "Function".into(),
                name: "oldName".into(),
                qualified_name: "p.Thing".into(),
                file_path: "src/a.rs".into(),
                start_line: 1,
                end_line: 2,
                properties: serde_json::json!({}),
            })
            .unwrap();
        // Upsert on the SAME qualified_name with a different name + label.
        let id2 = s
            .insert_node(&NewNode {
                project: "p".into(),
                label: "Method".into(),
                name: "newName".into(),
                qualified_name: "p.Thing".into(),
                file_path: "src/a.rs".into(),
                start_line: 1,
                end_line: 2,
                properties: serde_json::json!({}),
            })
            .unwrap();
        assert_eq!(id, id2, "upsert must reuse the rowid");

        // The new name resolves to exactly the live node.
        let hits = crate::fts::search_fts(&s, "newName", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].node_id, id);

        // The OLD name's unique tokens must not match a non-existent rowid.
        let stale = fts_match_rowids(&s, "old");
        for rid in &stale {
            assert!(
                s.get_node(*rid).unwrap().is_some(),
                "old-name token leaked an orphan rowid {rid}"
            );
        }
        // FTS matched rowids for the union of tokens equals the one live
        // node — no duplicate posting for the reused rowid.
        let all = fts_match_rowids(&s, "old OR new OR name OR function OR method OR thing");
        assert_eq!(all, vec![id], "exactly one live rowid, no leak: {all:?}");
    }

    /// P1 (re-review, fsync DoS): `insert_nodes` must commit ONE write
    /// transaction for a whole batch, not one per node. Before the batched
    /// path the indexer called `insert_node` per symbol, so a file with N
    /// functions did N self-committing transactions (N fsyncs on a durable
    /// store). We count committed write transactions with SQLite's commit
    /// hook and assert the batch path is a single commit while the
    /// per-node path is one-per-node — a direct, deterministic measurement
    /// of the write-amplification fix (no wall-clock timing).
    #[test]
    fn insert_nodes_batches_into_one_transaction() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        const N: usize = 25;

        // Per-node baseline: N inserts → N commits.
        let per_node_commits = {
            let mut s = store_with_project("p");
            let count = Arc::new(AtomicUsize::new(0));
            let c2 = count.clone();
            s.conn().commit_hook(Some(move || {
                c2.fetch_add(1, Ordering::SeqCst);
                false // allow the commit
            }));
            for i in 0..N {
                s.insert_node(&new_node("p", "Function", &format!("p.f{i}")))
                    .unwrap();
            }
            s.conn().commit_hook(None::<fn() -> bool>);
            count.load(Ordering::SeqCst)
        };
        assert_eq!(
            per_node_commits, N,
            "baseline: per-node insert must commit once per node"
        );

        // Batched path: N inserts in ONE call → exactly one commit.
        let batched_commits = {
            let mut s = store_with_project("p");
            let count = Arc::new(AtomicUsize::new(0));
            let c2 = count.clone();
            s.conn().commit_hook(Some(move || {
                c2.fetch_add(1, Ordering::SeqCst);
                false
            }));
            let batch: Vec<NewNode> = (0..N)
                .map(|i| new_node("p", "Function", &format!("p.f{i}")))
                .collect();
            s.insert_nodes(&batch).unwrap();
            s.conn().commit_hook(None::<fn() -> bool>);
            count.load(Ordering::SeqCst)
        };
        assert_eq!(
            batched_commits, 1,
            "batched insert_nodes must commit exactly ONCE for the whole \
             batch (was {N} commits per-node); this is the fsync fix"
        );
    }

    /// `insert_nodes` (batched) must be byte-for-byte equivalent to calling
    /// `insert_node` per element: same ids, same rows, same FTS behaviour.
    #[test]
    fn insert_nodes_batched_matches_per_node_inserts() {
        let mut s = store_with_project("p");
        let nodes = vec![
            new_node("p", "Function", "p.a"),
            new_node("p", "Function", "p.b"),
            new_node("p", "Struct", "p.C"),
        ];
        let ids = s.insert_nodes(&nodes).unwrap();
        assert_eq!(ids.len(), 3);
        // Every returned id maps to the matching row.
        assert_eq!(s.get_node(ids[0]).unwrap().unwrap().qualified_name, "p.a");
        assert_eq!(s.get_node(ids[2]).unwrap().unwrap().label, "Struct");
        // FTS search finds each.
        let hits = crate::fts::search_fts(&s, "a", 10).unwrap();
        assert!(hits.iter().any(|h| h.node_id == ids[0]));
        // Empty batch is a no-op.
        assert!(s.insert_nodes(&[]).unwrap().is_empty());
    }
}
