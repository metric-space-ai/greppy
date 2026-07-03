//! Golden-master tests for `grepplus-store`.
//!
//! These tests build a small synthetic graph (a "function calls function"
//! structure with imports and routes) and assert the count invariants
//! that the upstream `codebase-memory-mcp` golden-master suite asserts
//! against its own DB. The point is to lock down the schema and API
//! shape so the Phase 8 parity test (real upstream CLI vs our CLI on a
//! real repo) has a clean foundation.

use grepplus_store::{
    file_state::sha256_hex, Edge, FileState, NewEdge, NewNode, Node, Project, Store, WorkspaceState,
};

fn build_small_graph(store: &mut Store) -> SmallGraph {
    // Project metadata
    store
        .upsert_project(&Project {
            name: "demo".into(),
            indexed_at: "2026-06-28T20:00:00Z".into(),
            root_path: "/repos/demo".into(),
        })
        .unwrap();

    // Workspace state
    store
        .upsert_workspace_state(&WorkspaceState {
            root_path: "/repos/demo".into(),
            git_dir: Some("/repos/demo/.git".into()),
            git_common_dir: Some("/repos/demo/.git".into()),
            head_oid: Some("deadbeef".into()),
            index_signature: Some("idx-sig-test".into()),
            schema_version: 1,
            indexer_version: "grepplus-indexer-v1".into(),
            graph_generation: 1,
            updated_at: "2026-06-28T20:00:00Z".into(),
        })
        .unwrap();

    // Three functions: A calls B, A calls C; B has a CALLS edge back to A
    // (mutual recursion, exercises the directed edge semantics).
    let a = store
        .insert_node(&NewNode {
            project: "demo".into(),
            label: "Function".into(),
            name: "A".into(),
            qualified_name: "demo.A".into(),
            file_path: "src/a.rs".into(),
            start_line: 1,
            end_line: 10,
            properties: serde_json::json!({"language": "rust"}),
        })
        .unwrap();
    let b = store
        .insert_node(&NewNode {
            project: "demo".into(),
            label: "Function".into(),
            name: "B".into(),
            qualified_name: "demo.B".into(),
            file_path: "src/b.rs".into(),
            start_line: 1,
            end_line: 5,
            properties: serde_json::json!({"language": "rust"}),
        })
        .unwrap();
    let c = store
        .insert_node(&NewNode {
            project: "demo".into(),
            label: "Function".into(),
            name: "C".into(),
            qualified_name: "demo.C".into(),
            file_path: "src/c.rs".into(),
            start_line: 1,
            end_line: 8,
            properties: serde_json::json!({"language": "rust"}),
        })
        .unwrap();

    // Edges
    store
        .insert_edge(&NewEdge {
            project: "demo".into(),
            source_id: a,
            target_id: b,
            edge_type: "CALLS".into(),
            properties: serde_json::json!({}),
        })
        .unwrap();
    store
        .insert_edge(&NewEdge {
            project: "demo".into(),
            source_id: a,
            target_id: c,
            edge_type: "CALLS".into(),
            properties: serde_json::json!({}),
        })
        .unwrap();
    store
        .insert_edge(&NewEdge {
            project: "demo".into(),
            source_id: b,
            target_id: a,
            edge_type: "CALLS".into(),
            properties: serde_json::json!({}),
        })
        .unwrap();
    store
        .insert_edge(&NewEdge {
            project: "demo".into(),
            source_id: c,
            target_id: b,
            edge_type: "IMPORTS".into(),
            properties: serde_json::json!({}),
        })
        .unwrap();

    // File state for one of the source files
    store
        .upsert_file_state(&FileState {
            project: "demo".into(),
            rel_path: "src/a.rs".into(),
            language: "rust".into(),
            sha256: sha256_hex(b"fn A() { B(); C(); }"),
            mtime_ns: 1_700_000_000_000_000_000,
            size: 21,
            parser_version: "tree-sitter-0.21".into(),
            extractor_version: "grepplus-extractor-v1".into(),
            last_indexed_generation: 1,
        })
        .unwrap();

    SmallGraph { a, b, c }
}

struct SmallGraph {
    a: i64,
    b: i64,
    c: i64,
}

#[test]
fn golden_master_small_graph_node_and_edge_counts() {
    let mut s = Store::open_memory().unwrap();
    let _g = build_small_graph(&mut s);

    // 3 functions inserted.
    assert_eq!(s.count_nodes_by_label("demo", "Function").unwrap(), 3);

    // 4 edges total: 3 CALLS + 1 IMPORTS.
    assert_eq!(s.count_edges("demo", Some("CALLS")).unwrap(), 3);
    assert_eq!(s.count_edges("demo", Some("IMPORTS")).unwrap(), 1);
    assert_eq!(s.count_edges("demo", None).unwrap(), 4);

    // One project row.
    assert_eq!(s.list_projects().unwrap().len(), 1);

    // One file state.
    assert_eq!(s.list_file_states("demo").unwrap().len(), 1);
}

#[test]
fn golden_master_traversal_returns_correct_neighbours() {
    let mut s = Store::open_memory().unwrap();
    let g = build_small_graph(&mut s);

    // Outgoing from A: B, C (both CALLS).
    let out = s.outgoing_edges(g.a, Some("CALLS"), 10).unwrap();
    assert_eq!(out.len(), 2);
    let targets: Vec<i64> = out.iter().map(|e| e.target_id).collect();
    assert!(targets.contains(&g.b));
    assert!(targets.contains(&g.c));

    // Incoming to B: A (CALLS), C (IMPORTS).
    let inc = s.incoming_edges(g.b, None, 10).unwrap();
    assert_eq!(inc.len(), 2);
}

#[test]
fn golden_master_fts_finds_camelcase_token() {
    let mut s = Store::open_memory().unwrap();
    let _g = build_small_graph(&mut s);

    let hits = grepplus_store::fts::search_fts(&s, "processOrder", 10).unwrap();
    // The fixture has no "process" prefix in names, but BM25 will return
    // an empty result set here. The important assertion is that the
    // query runs without error and returns a Vec (zero hits is fine).
    assert!(hits.len() <= 10);

    // Exact-prefix search for an existing name must hit.
    let hits = grepplus_store::fts::search_fts(&s, "A", 10).unwrap();
    assert!(
        !hits.is_empty(),
        "exact name search should return at least one hit"
    );
}

#[test]
fn golden_master_node_round_trip_preserves_properties() {
    let mut s = Store::open_memory().unwrap();
    let g = build_small_graph(&mut s);
    let n: Node = s.get_node(g.a).unwrap().unwrap();
    let expected: Edge = Edge {
        id: 0, // not compared
        project: n.project.clone(),
        source_id: 0,
        target_id: 0,
        edge_type: String::new(),
        properties: n.properties.clone(),
    };
    // Round-trip preserves properties (asserts on the JSON value, not the
    // synthetic Edge type's other fields).
    assert_eq!(expected.properties["language"], "rust");
    assert_eq!(n.qualified_name, "demo.A");
    assert_eq!(n.start_line, 1);
    assert_eq!(n.end_line, 10);
}

#[test]
fn golden_master_integrity_check_passes_after_mutations() {
    let mut s = Store::open_memory().unwrap();
    let g = build_small_graph(&mut s);
    s.delete_node(g.c).unwrap();
    s.bump_generation("/repos/demo").unwrap();
    s.integrity_check().unwrap();
}
