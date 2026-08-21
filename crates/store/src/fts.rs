//! FTS5 helpers.
//!
//! We feed camelCase tokens into the contentless `nodes_fts`
//! table so that a search for `processOrder` matches `ProcessOrder`,
//! using a small, dependency-free `camel_split`.

/// Split a CamelCase / snake_case / kebab-case identifier into lowercase
/// tokens separated by single spaces.
///
/// Examples:
/// - `camel_split("ProcessOrder")` → `"process order"`
/// - `camel_split("process_order")` → `"process order"`
/// - `camel_split("kebab-case")` → `"kebab case"`
/// - `camel_split("already_lower")` → `"already lower"`
///
/// This is the tokenisation used on the `nodes_fts` insert path for
/// BM25 search.
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
            //   - transitioning from a digit (v2Loader → v2 Loader).
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
            // Only split digit→uppercase boundaries
            // (`v2Loader` → `v2 loader`); the reverse
            // (letter→digit) is intentionally NOT split because
            // `loader2` reads as a single numeric-suffixed word
            // in identifier logic.
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
    if store.is_overlay() {
        let overfetch = limit.saturating_mul(4).max(limit);
        let mut hits = search_fts_layer(store, "main", false, project, &fts_query, overfetch)?;
        hits.extend(search_fts_layer(
            store,
            "greppy_base",
            true,
            project,
            &fts_query,
            overfetch,
        )?);
        for hit in &mut hits {
            if let Some(node) = store.get_node(hit.node_id)? {
                hit.rank = -overlay_symbol_score(query, &node);
            }
        }
        hits.sort_by(|left, right| {
            left.rank
                .total_cmp(&right.rank)
                .then_with(|| left.node_id.cmp(&right.node_id))
        });
        hits.dedup_by_key(|hit| hit.node_id);
        hits.truncate(limit);
        return Ok(hits);
    }

    // Forensics F2: pseudo/structural nodes carry no navigational value as
    // user-facing symbols. Their information already lives in edges or the
    // file tree, and they can outrank real definitions on common names or path
    // tokens. `nodes_fts` is contentless, so join the real `nodes` table
    // (rowid == node id) and filter there. `idx_nodes_label` keeps this cheap.
    let sql = match project {
        Some(_) => {
            "SELECT nodes_fts.rowid, bm25(nodes_fts) \
             FROM nodes_fts JOIN nodes ON nodes.id = nodes_fts.rowid \
             WHERE nodes_fts MATCH ?1 \
               AND nodes.project = ?2 \
               AND nodes.label NOT IN ('Call','Import','File','Folder','Project') \
               AND nodes.name != '__file__' \
               AND nodes.qualified_name NOT LIKE '%::__file__' \
               AND nodes.qualified_name NOT LIKE '%.__file__' \
             ORDER BY rank LIMIT ?3"
        }
        None => {
            "SELECT nodes_fts.rowid, bm25(nodes_fts) \
             FROM nodes_fts JOIN nodes ON nodes.id = nodes_fts.rowid \
             WHERE nodes_fts MATCH ?1 \
               AND nodes.label NOT IN ('Call','Import','File','Folder','Project') \
               AND nodes.name != '__file__' \
               AND nodes.qualified_name NOT LIKE '%::__file__' \
               AND nodes.qualified_name NOT LIKE '%.__file__' \
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
/// pseudo-node / structural-anchor filtering as [`search_fts_in_project`].
pub fn count_fts_in_project(
    store: &crate::store::Store,
    project: &str,
    query: &str,
) -> Result<i64, crate::store_error::Error> {
    use crate::store_error::Error;
    let Some(fts_query) = fts_prefix_query(query) else {
        return Ok(0);
    };
    if store.is_overlay() {
        let delta = count_fts_layer(store, "main", false, project, &fts_query)?;
        let base = count_fts_layer(store, "greppy_base", true, project, &fts_query)?;
        return Ok(delta.saturating_add(base));
    }
    store
        .conn()
        .query_row(
            "SELECT COUNT(*) \
             FROM nodes_fts JOIN nodes ON nodes.id = nodes_fts.rowid \
             WHERE nodes_fts MATCH ?1 \
               AND nodes.project = ?2 \
               AND nodes.label NOT IN ('Call','Import','File','Folder','Project') \
               AND nodes.name != '__file__' \
               AND nodes.qualified_name NOT LIKE '%::__file__' \
               AND nodes.qualified_name NOT LIKE '%.__file__'",
            rusqlite::params![fts_query, project],
            |row| row.get(0),
        )
        .map_err(Error::Sqlite)
}

fn search_fts_layer(
    store: &crate::store::Store,
    schema: &str,
    base_layer: bool,
    project: Option<&str>,
    fts_query: &str,
    limit: usize,
) -> Result<Vec<FtsHit>, crate::store_error::Error> {
    use crate::store_error::Error;
    debug_assert!(matches!(schema, "main" | "greppy_base"));
    let id = if base_layer {
        "-nodes_fts.rowid"
    } else {
        "nodes_fts.rowid"
    };
    let hidden = if base_layer {
        "AND NOT EXISTS (SELECT 1 FROM greppy_hidden_paths h WHERE h.path = nodes.file_path)"
    } else {
        ""
    };
    let project_filter = if project.is_some() {
        "AND nodes.project = ?2"
    } else {
        ""
    };
    let limit_parameter = if project.is_some() { "?3" } else { "?2" };
    let sql = format!(
        "SELECT {id}, bm25(nodes_fts)
         FROM {schema}.nodes_fts
         JOIN {schema}.nodes ON nodes.id = nodes_fts.rowid
         WHERE nodes_fts MATCH ?1
           {project_filter}
           AND nodes.label NOT IN ('Call','Import','File','Folder','Project')
           AND nodes.name != '__file__'
           AND nodes.qualified_name NOT LIKE '%::__file__'
           AND nodes.qualified_name NOT LIKE '%.__file__'
           {hidden}
         ORDER BY rank LIMIT {limit_parameter}"
    );
    let mut stmt = store.conn().prepare(&sql).map_err(Error::Sqlite)?;
    let rows = if let Some(project) = project {
        stmt.query_map(rusqlite::params![fts_query, project, limit as i64], |row| {
            Ok(FtsHit {
                node_id: row.get(0)?,
                rank: row.get(1)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?
    } else {
        stmt.query_map(rusqlite::params![fts_query, limit as i64], |row| {
            Ok(FtsHit {
                node_id: row.get(0)?,
                rank: row.get(1)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?
    };
    Ok(rows)
}

fn count_fts_layer(
    store: &crate::store::Store,
    schema: &str,
    base_layer: bool,
    project: &str,
    fts_query: &str,
) -> Result<i64, crate::store_error::Error> {
    use crate::store_error::Error;
    debug_assert!(matches!(schema, "main" | "greppy_base"));
    let hidden = if base_layer {
        "AND NOT EXISTS (SELECT 1 FROM greppy_hidden_paths h WHERE h.path = nodes.file_path)"
    } else {
        ""
    };
    let sql = format!(
        "SELECT COUNT(*)
         FROM {schema}.nodes_fts
         JOIN {schema}.nodes ON nodes.id = nodes_fts.rowid
         WHERE nodes_fts MATCH ?1
           AND nodes.project = ?2
           AND nodes.label NOT IN ('Call','Import','File','Folder','Project')
           AND nodes.name != '__file__'
           AND nodes.qualified_name NOT LIKE '%::__file__'
           AND nodes.qualified_name NOT LIKE '%.__file__'
           {hidden}"
    );
    store
        .conn()
        .query_row(&sql, rusqlite::params![fts_query, project], |row| {
            row.get(0)
        })
        .map_err(Error::Sqlite)
}

fn overlay_symbol_score(query: &str, node: &crate::Node) -> f64 {
    let query_tokens = camel_split(query);
    let name = camel_split(&node.name);
    let qualified = camel_split(&node.qualified_name);
    let mut score = if name == query_tokens { 100.0 } else { 0.0 };
    if name.starts_with(&query_tokens) {
        score += 25.0;
    }
    for token in query_tokens
        .split_whitespace()
        .filter(|token| !token.is_empty())
    {
        score += name
            .split_whitespace()
            .filter(|value| *value == token)
            .count() as f64
            * 10.0;
        score += qualified
            .split_whitespace()
            .filter(|value| *value == token)
            .count() as f64
            * 2.0;
    }
    score
}

#[cfg(test)]
mod tests {
    use super::{camel_split, count_fts_in_project, search_fts, search_fts_in_project};
    use crate::node::NewNode;
    use crate::store::Store;
    use crate::Project;

    /// Forensics F2: `search_fts` must NOT return pseudo or structural
    /// anchors. They share tokens with real symbols and files, so without the
    /// filter a `Call::Store`, import pseudo-node, or `__file__` row can
    /// outrank the actual definition and flood the result window.
    #[test]
    fn search_fts_excludes_pseudo_nodes_and_file_anchors() {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "2026-06-30T00:00:00Z".into(),
            root_path: "/repos/p".into(),
        })
        .unwrap();
        // A real definition plus pseudo/structural nodes that share the same
        // name or qname/path tokens.
        for (label, name, qname, start_line) in [
            ("Struct", "Store", "p.Store", 1),
            ("Call", "Store", "p.caller.Call::Store", 1),
            ("Import", "Store", "p.lib.Import::Store", 1),
            ("Module", "__file__", "src/store.rs::__file__", 1),
            ("File", "store.rs", "p.src.store.__file__", 0),
        ] {
            s.insert_node(&NewNode {
                project: "p".into(),
                label: label.into(),
                name: name.into(),
                qualified_name: qname.into(),
                file_path: "src/store.rs".into(),
                start_line,
                end_line: start_line.max(1),
                properties: serde_json::json!({}),
            })
            .unwrap();
        }

        let hits = search_fts(&s, "Store", 10).unwrap();
        assert!(!hits.is_empty(), "the real Struct::Store must be found");
        // Every returned node must be the Struct, never a pseudo/anchor.
        for h in &hits {
            let (label, name, qname): (String, String, String) = s
                .conn()
                .query_row(
                    "SELECT label, name, qualified_name FROM nodes WHERE id = ?1",
                    rusqlite::params![h.node_id],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
                )
                .unwrap();
            assert!(
                !matches!(
                    label.as_str(),
                    "Call" | "Import" | "File" | "Folder" | "Project"
                ) && name != "__file__"
                    && !qname.ends_with("::__file__")
                    && !qname.ends_with(".__file__"),
                "search_fts must not return pseudo/anchor node {label} {qname}"
            );
        }
        assert_eq!(count_fts_in_project(&s, "p", "Store").unwrap(), 1);
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
        // digit→uppercase must split (v2Loader →
        // v2 loader), not collapse into a single token "v2loader".
        // Letter→digit (foo9) is intentionally kept together — only
        // digit→uppercase triggers the boundary.
        assert_eq!(camel_split("v2Loader"), "v2 loader");
        assert_eq!(camel_split("foo9Bar"), "foo9 bar");
    }
}
