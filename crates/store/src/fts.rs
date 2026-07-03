//! FTS5 helpers.
//!
//! The upstream feeds camelCase tokens into the contentless `nodes_fts`
//! table so that a search for `processOrder` matches `ProcessOrder`. We
//! replicate that behaviour with a small, dependency-free `camel_split`.

/// Split a CamelCase / snake_case / kebab-case identifier into lowercase
/// tokens separated by single spaces.
///
/// Examples:
/// - `camel_split("ProcessOrder")` → `"process order"`
/// - `camel_split("process_order")` → `"process order"`
/// - `camel_split("kebab-case")` → `"kebab case"`
/// - `camel_split("already_lower")` → `"already lower"`
///
/// The exact tokenisation rules live in upstream
/// `src/store/store.c` near the `nodes_fts` insert path; this
/// implementation is a faithful subset for Phase 4 BM25 testing. If a
/// later phase proves a divergence, replace this with a port of the
/// upstream function.
pub fn camel_split(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let chars: Vec<char> = input.chars().collect();
    let mut prev_lower = false;
    let mut prev_digit = false;
    let mut prev_boundary = true;

    for (i, &c) in chars.iter().enumerate() {
        if c == '_' || c == '-' || c == '.' || c == '/' {
            if !out.is_empty() && !out.ends_with(' ') {
                out.push(' ');
            }
            prev_lower = false;
            prev_digit = false;
            prev_boundary = true;
            continue;
        }
        if c.is_ascii_uppercase() {
            // Insert a boundary when:
            //   - transitioning from a lowercase letter (camelCase),
            //   - transitioning from an uppercase letter followed by a
            //     lowercase letter (XMLParser → XML Parser), or
            //   - transitioning from a digit (v2Loader → v2 Loader,
            //     R-025).
            let next_lower = chars
                .get(i + 1)
                .map(|c| c.is_ascii_lowercase())
                .unwrap_or(false);
            if !prev_boundary
                && (prev_lower
                    || (i > 0 && chars[i - 1].is_ascii_uppercase() && next_lower)
                    || prev_digit)
            {
                out.push(' ');
            }
            for lc in c.to_lowercase() {
                out.push(lc);
            }
            prev_lower = false;
            prev_digit = false;
            prev_boundary = false;
        } else if c.is_alphanumeric() {
            // R-025: only split digit→uppercase boundaries
            // (`v2Loader` → `v2 loader`); the reverse
            // (letter→digit) is intentionally NOT split because
            // `loader2` reads as a single numeric-suffixed word
            // in identifier logic. The reviewer's complaint was
            // specifically about digit→uppercase coverage, not
            // letter→digit.
            out.push(c.to_ascii_lowercase());
            prev_lower = c.is_ascii_lowercase();
            prev_digit = c.is_ascii_digit();
            prev_boundary = false;
        } else {
            // Treat any other char (whitespace, punctuation) as a boundary.
            if !out.is_empty() && !out.ends_with(' ') {
                out.push(' ');
            }
            prev_lower = false;
            prev_digit = false;
            prev_boundary = true;
        }
    }
    out.trim().to_string()
}

/// One FTS5 hit, ranked by BM25.
#[derive(Debug, Clone, PartialEq)]
pub struct FtsHit {
    pub node_id: i64,
    pub rank: f64, // negative; closer to 0 is better (SQLite BM25 convention)
}

fn fts_prefix_query(query: &str) -> Option<String> {
    let tokens = camel_split(query);
    if tokens.is_empty() {
        return None;
    }
    Some(
        tokens
            .split_whitespace()
            .map(|t| format!("{t}*"))
            .collect::<Vec<_>>()
            .join(" "),
    )
}

/// Run an FTS5 query against the `nodes_fts` table and return the
/// matching node ids in BM25 order.
pub fn search_fts(
    store: &crate::store::Store,
    query: &str,
    limit: usize,
) -> Result<Vec<FtsHit>, crate::store_error::Error> {
    search_fts_scoped(store, None, query, limit)
}

/// Project-scoped variant of [`search_fts`]. Use this from user-facing
/// commands so a shared store cannot leak symbols from another repo.
pub fn search_fts_in_project(
    store: &crate::store::Store,
    project: &str,
    query: &str,
    limit: usize,
) -> Result<Vec<FtsHit>, crate::store_error::Error> {
    search_fts_scoped(store, Some(project), query, limit)
}

fn search_fts_scoped(
    store: &crate::store::Store,
    project: Option<&str>,
    query: &str,
    limit: usize,
) -> Result<Vec<FtsHit>, crate::store_error::Error> {
    use crate::store_error::Error;
    let Some(fts_query) = fts_prefix_query(query) else {
        return Ok(Vec::new());
    };

    // Forensics F2: the parser materialises `Call` and `Import` pseudo-nodes
    // (one per call site / import) so the CALLS / IMPORTS *edges* have
    // endpoints. They carry no navigational value as *symbols* — their
    // information already lives in the edges who-calls/callees/trace read —
    // yet on a common name they can be 90 %+ of the FTS hits (`Call::echo`,
    // `Import::store::Store`), pushing real definitions out of the result
    // window. `nodes_fts` is contentless so its `label` column can't be
    // filtered directly; join the real `nodes` table (rowid == node id) and
    // exclude the two pseudo-labels. `idx_nodes_label` keeps this cheap.
    let sql = match project {
        Some(_) => {
            "SELECT nodes_fts.rowid, bm25(nodes_fts) \
             FROM nodes_fts JOIN nodes ON nodes.id = nodes_fts.rowid \
             WHERE nodes_fts MATCH ?1 AND nodes.project = ?2 AND nodes.label NOT IN ('Call','Import') \
             ORDER BY rank LIMIT ?3"
        }
        None => {
            "SELECT nodes_fts.rowid, bm25(nodes_fts) \
             FROM nodes_fts JOIN nodes ON nodes.id = nodes_fts.rowid \
             WHERE nodes_fts MATCH ?1 AND nodes.label NOT IN ('Call','Import') \
             ORDER BY rank LIMIT ?2"
        }
    };
    let mut stmt = store.conn().prepare(sql).map_err(Error::Sqlite)?;
    let hits = if let Some(project) = project {
        stmt.query_map(rusqlite::params![fts_query, project, limit as i64], |row| {
            Ok(FtsHit {
                node_id: row.get(0)?,
                rank: row.get(1)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(Error::Sqlite)?
    } else {
        stmt.query_map(rusqlite::params![fts_query, limit as i64], |row| {
            Ok(FtsHit {
                node_id: row.get(0)?,
                rank: row.get(1)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(Error::Sqlite)?
    };
    Ok(hits)
}

/// Exact count for project-scoped symbol FTS matches, using the same
/// pseudo-node filtering as [`search_fts_in_project`].
pub fn count_fts_in_project(
    store: &crate::store::Store,
    project: &str,
    query: &str,
) -> Result<i64, crate::store_error::Error> {
    use crate::store_error::Error;
    let Some(fts_query) = fts_prefix_query(query) else {
        return Ok(0);
    };
    store
        .conn()
        .query_row(
            "SELECT COUNT(*) \
             FROM nodes_fts JOIN nodes ON nodes.id = nodes_fts.rowid \
             WHERE nodes_fts MATCH ?1 AND nodes.project = ?2 AND nodes.label NOT IN ('Call','Import')",
            rusqlite::params![fts_query, project],
            |row| row.get(0),
        )
        .map_err(Error::Sqlite)
}

#[cfg(test)]
mod tests {
    use super::{camel_split, count_fts_in_project, search_fts, search_fts_in_project};
    use crate::node::NewNode;
    use crate::store::Store;
    use crate::Project;

    /// Forensics F2: `search_fts` must NOT return `Call` / `Import`
    /// pseudo-nodes. They share their name with the real symbol they
    /// reference, so without the filter a `Call::Store` outranks the
    /// `Struct::Store` definition and floods the result window.
    #[test]
    fn search_fts_excludes_call_and_import_pseudo_nodes() {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "2026-06-30T00:00:00Z".into(),
            root_path: "/repos/p".into(),
        })
        .unwrap();
        // A real definition plus a Call and an Import pseudo-node that share
        // the exact same name `Store`.
        for (label, qname) in [
            ("Struct", "p.Store"),
            ("Call", "p.caller.Call::Store"),
            ("Import", "p.lib.Import::Store"),
        ] {
            s.insert_node(&NewNode {
                project: "p".into(),
                label: label.into(),
                name: "Store".into(),
                qualified_name: qname.into(),
                file_path: "src/lib.rs".into(),
                start_line: 1,
                end_line: 2,
                properties: serde_json::json!({}),
            })
            .unwrap();
        }

        let hits = search_fts(&s, "Store", 10).unwrap();
        assert!(!hits.is_empty(), "the real Struct::Store must be found");
        // Every returned node must be the Struct, never the Call/Import.
        for h in &hits {
            let label: String = s
                .conn()
                .query_row(
                    "SELECT label FROM nodes WHERE id = ?1",
                    rusqlite::params![h.node_id],
                    |r| r.get(0),
                )
                .unwrap();
            assert!(
                label != "Call" && label != "Import",
                "search_fts must not return a {label} pseudo-node"
            );
        }
    }

    #[test]
    fn search_fts_project_scope_and_count_exclude_other_projects() {
        let mut s = Store::open_memory().unwrap();
        for project in ["p1", "p2"] {
            s.upsert_project(&Project {
                name: project.into(),
                indexed_at: "2026-07-01T00:00:00Z".into(),
                root_path: format!("/repos/{project}"),
            })
            .unwrap();
            s.insert_node(&NewNode {
                project: project.into(),
                label: "Function".into(),
                name: "SharedName".into(),
                qualified_name: format!("{project}.SharedName"),
                file_path: "src/lib.rs".into(),
                start_line: 1,
                end_line: 2,
                properties: serde_json::json!({}),
            })
            .unwrap();
        }

        let hits = search_fts_in_project(&s, "p1", "Shared", 10).unwrap();
        assert_eq!(hits.len(), 1);
        let project: String = s
            .conn()
            .query_row(
                "SELECT project FROM nodes WHERE id = ?1",
                rusqlite::params![hits[0].node_id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(project, "p1");
        assert_eq!(count_fts_in_project(&s, "p1", "Shared").unwrap(), 1);
    }

    #[test]
    fn splits_camel_case() {
        assert_eq!(camel_split("ProcessOrder"), "process order");
        assert_eq!(camel_split("processOrder"), "process order");
        assert_eq!(camel_split("XMLParser"), "xml parser");
    }

    #[test]
    fn splits_snake_and_kebab() {
        assert_eq!(camel_split("process_order"), "process order");
        assert_eq!(camel_split("kebab-case"), "kebab case");
        assert_eq!(camel_split("a.b.c"), "a b c");
    }

    #[test]
    fn handles_already_lowercase() {
        assert_eq!(camel_split("foo"), "foo");
        assert_eq!(camel_split("foo_bar"), "foo bar");
    }

    #[test]
    fn handles_consecutive_boundaries() {
        assert_eq!(camel_split("foo__bar"), "foo bar");
        assert_eq!(camel_split("a---b"), "a b");
    }

    #[test]
    fn handles_empty() {
        assert_eq!(camel_split(""), "");
    }

    #[test]
    fn camel_split_handles_digit_boundary() {
        // R-025 / WP-R025: digit→uppercase must split (v2Loader →
        // v2 loader), not collapse into a single token "v2loader".
        // Letter→digit (foo9) is intentionally kept together — only
        // digit→uppercase triggers the boundary, per the reviewer's
        // scope (the original review called out `v2Loader`).
        assert_eq!(camel_split("v2Loader"), "v2 loader");
        assert_eq!(camel_split("foo9Bar"), "foo9 bar");
    }
}
