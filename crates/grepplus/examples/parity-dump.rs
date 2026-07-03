// Phase 8 — Parity-dump example binary.
//
// Usage:
//   cargo run --bin parity-dump -- <root>
//
// Reads the indexed store at `<root>/.grepplus/graph.db` and prints
// a structured JSON report with the metrics phasenplan §13 calls
// out for golden-master parity:
//
//   - file_count            (number of indexed files)
//   - node_counts_by_label  (function/struct/... → count)
//   - edge_counts_by_type   (CALLS / IMPORTS / ... → count)
//   - top_qualified_names   (first 20 sorted alphabetically)
//   - sample_search_results (top-5 for "ProcessOrder" / "Greeter")
//   - sample_trace_results  (depth-2 trace from "hello")
//   - sample_search_code    (lexical FTS for "hello")
//   - workspace_state       (current head_oid, index_signature,
//                            graph_generation, schema_version)
//
// The companion `docs/grepplus_phase8_parity_report.md` documents
// the expected upstream behaviour for each metric.

use std::collections::BTreeMap;

use grepplus_search::{GraphQuery, TraceDirection};
use grepplus_store::{OpenOptions, Store};

fn main() {
    let _ = grepplus_core::logging::init();
    let root = std::env::args()
        .nth(1)
        .expect("usage: parity-dump <repo-root>");
    let project = std::env::args().nth(2).unwrap_or_else(|| {
        std::path::Path::new(&root)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("default")
            .to_string()
    });
    let code = match run(&root, &project) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("parity-dump: {e}");
            1
        }
    };
    std::process::exit(code);
}

fn run(root: &str, project: &str) -> Result<(), String> {
    // R-005 / WP-R005 / RV-007: the graph DB lives under the
    // platform locator, never at `<root>/.grepplus/graph.db`. Use
    // the shared `grepplus_core::workspace::store_path` helper so
    // parity-dump agrees with the indexer and the CLI dispatcher.
    let path = grepplus_core::workspace::store_path(std::path::Path::new(root));
    if !path.is_file() {
        return Err(format!("no store at {}", path.display()));
    }
    let store = Store::open_with(&path, OpenOptions::read_only())
        .map_err(|e| format!("store open: {e}"))?;

    // 1. file count
    let file_count: i64 = store
        .conn()
        .query_row(
            "SELECT count(*) FROM file_state WHERE project = ?1",
            [project],
            |r| r.get(0),
        )
        .map_err(|e| format!("file_count: {e}"))?;

    // 2. node counts by label
    let mut node_counts: BTreeMap<String, i64> = BTreeMap::new();
    let mut stmt = store
        .conn()
        .prepare(
            "SELECT label, count(*) FROM nodes WHERE project = ?1 GROUP BY label ORDER BY label",
        )
        .map_err(|e| format!("node_counts prepare: {e}"))?;
    let rows = stmt
        .query_map([project], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?))
        })
        .map_err(|e| format!("node_counts query: {e}"))?;
    for r in rows {
        let (label, count) = r.map_err(|e| format!("node_counts row: {e}"))?;
        node_counts.insert(label, count);
    }

    // 3. edge counts by type
    let mut edge_counts: BTreeMap<String, i64> = BTreeMap::new();
    let mut stmt = store
        .conn()
        .prepare("SELECT edge_type, count(*) FROM edges WHERE project = ?1 GROUP BY edge_type ORDER BY edge_type")
        .map_err(|e| format!("edge_counts prepare: {e}"))?;
    let rows = stmt
        .query_map([project], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?))
        })
        .map_err(|e| format!("edge_counts query: {e}"))?;
    for r in rows {
        let (edge_type, count) = r.map_err(|e| format!("edge_counts row: {e}"))?;
        edge_counts.insert(edge_type, count);
    }

    // 4. top qualified names (first 20)
    let mut names: Vec<String> = Vec::new();
    let mut stmt = store
        .conn()
        .prepare(
            "SELECT qualified_name FROM nodes WHERE project = ?1 ORDER BY qualified_name LIMIT 20",
        )
        .map_err(|e| format!("names prepare: {e}"))?;
    let rows = stmt
        .query_map([project], |r| r.get::<_, String>(0))
        .map_err(|e| format!("names query: {e}"))?;
    for r in rows {
        names.push(r.map_err(|e| format!("names row: {e}"))?);
    }

    // 5. sample search results
    let search_targets = ["ProcessOrder", "Greeter", "hello"];
    let mut sample_search: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for q in search_targets {
        let hits = match grepplus_search::search_graph(
            &store,
            &GraphQuery::any().with_name(q).with_limit(5),
        ) {
            Ok(h) => h,
            Err(_) => continue,
        };
        let mut names: Vec<String> = hits
            .iter()
            .map(|h| format!("{}@{}:{}", h.qualified_name, h.file_path, h.start_line))
            .collect();
        names.sort();
        sample_search.insert(q.to_string(), names);
    }

    // 6. sample trace results
    let mut sample_trace: BTreeMap<String, Vec<String>> = BTreeMap::new();
    if let Ok(rows) =
        grepplus_search::search_graph(&store, &GraphQuery::any().with_name("hello").with_limit(1))
    {
        if let Some(target) = rows.first() {
            let steps = grepplus_search::trace_path(
                &store,
                target.id,
                TraceDirection::Outgoing,
                Some("CALLS"),
                2,
            )
            .unwrap_or_default();
            let mut s: Vec<String> = steps
                .iter()
                .map(|step| format!("depth={} node={}", step.depth, step.node_id))
                .collect();
            s.sort();
            sample_trace.insert("hello:trace_outgoing".to_string(), s);
        }
    }

    // 7. sample search_code (file content FTS) for "hello". RV-011 /
    // WP-R013: project identity uses the canonical repo root (or
    // `--root`), NOT the cwd basename — re-derive it via the
    // shared helper so the bench report agrees with the CLI's
    // dispatch_search_code output.
    let root_path = std::path::Path::new(root);
    let project_for_search = grepplus_core::workspace::project_identity(root_path);
    let mut sample_search_code: Vec<String> = Vec::new();
    if let Ok(hits) = grepplus_search::search_code(&store, &project_for_search, "hello", 5) {
        for h in hits {
            sample_search_code.push(format!(
                "{} rank={:.4} \"{}\"",
                h.location, h.rank, h.snippet
            ));
        }
    }

    // 8. workspace state
    let mut ws_state: BTreeMap<String, String> = BTreeMap::new();
    if let Ok(row) = store.conn().query_row(
        "SELECT root_path, head_oid, index_signature, schema_version, indexer_version, graph_generation, updated_at
         FROM workspace_state ORDER BY updated_at DESC LIMIT 1",
        [],
        |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, Option<String>>(1)?,
                r.get::<_, Option<String>>(2)?,
                r.get::<_, i64>(3)?,
                r.get::<_, String>(4)?,
                r.get::<_, i64>(5)?,
                r.get::<_, String>(6)?,
            ))
        },
    ) {
        let (root_path, head_oid, idx_sig, schema_v, idx_v, gen, updated) = row;
        ws_state.insert("root_path".into(), root_path);
        ws_state.insert(
            "head_oid".into(),
            head_oid.unwrap_or_else(|| "<none>".into()),
        );
        ws_state.insert(
            "index_signature".into(),
            idx_sig.unwrap_or_else(|| "<none>".into()),
        );
        ws_state.insert("schema_version".into(), schema_v.to_string());
        ws_state.insert("indexer_version".into(), idx_v);
        ws_state.insert("graph_generation".into(), gen.to_string());
        ws_state.insert("updated_at".into(), updated);
    }

    let report = serde_json::json!({
        "file_count": file_count,
        "node_counts_by_label": node_counts,
        "edge_counts_by_type": edge_counts,
        "top_qualified_names": names,
        "sample_search_results": sample_search,
        "sample_trace_results": sample_trace,
        "sample_search_code": sample_search_code,
        "workspace_state": ws_state,
    });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    Ok(())
}
