//! Exercise the Anko-shaped query path through the real parser, resolver and store.
use greppy_indexer::index;
use greppy_store::Store;
use std::fs;

#[test]
fn qualified_go_type_usage_reaches_the_graph_without_guessing_ambiguous_names() {
    let repo = tempfile::tempdir().unwrap();
    fs::create_dir_all(repo.path().join("ast")).unwrap();
    fs::create_dir_all(repo.path().join("vm")).unwrap();
    fs::write(
        repo.path().join("ast/stmt.go"),
        "package ast\ntype VarStmt struct{}\n",
    )
    .unwrap();
    fs::write(
        repo.path().join("vm/vmStmt.go"),
        "package vm\nimport \"example/ast\"\nfunc runSingleStmt(stmt interface{}) {\n switch stmt.(type) {\n case *ast.VarStmt:\n  _ = ast.VarStmt{}\n }\n}\n",
    )
    .unwrap();
    let mut store = Store::open_memory().unwrap();
    index(&mut store, repo.path(), "anko").unwrap();
    let target = store
        .get_node_by_qname("anko", "ast/stmt.go::Class::VarStmt")
        .unwrap()
        .unwrap();
    let caller = store
        .get_node_by_qname("anko", "vm/vmStmt.go::Function::runSingleStmt")
        .unwrap()
        .unwrap();
    assert!(
        store
            .incoming_edges(target.id, Some("USAGE"), 30)
            .unwrap()
            .iter()
            .any(|edge| edge.source_id == caller.id),
        "who-calls must have the indexed type-switch/composite-literal usage"
    );

    fs::create_dir_all(repo.path().join("other")).unwrap();
    fs::write(
        repo.path().join("other/stmt.go"),
        "package other\ntype VarStmt struct{}\n",
    )
    .unwrap();
    let mut ambiguous = Store::open_memory().unwrap();
    index(&mut ambiguous, repo.path(), "anko").unwrap();
    for path in ["ast/stmt.go", "other/stmt.go"] {
        let target = ambiguous
            .get_node_by_qname("anko", &format!("{path}::Class::VarStmt"))
            .unwrap()
            .unwrap();
        assert!(
            ambiguous
                .incoming_edges(target.id, Some("USAGE"), 30)
                .unwrap()
                .is_empty(),
            "the existing conservative resolver must not guess among duplicate types"
        );
    }
}

fn mixed_cache(repo: &std::path::Path, path: &std::path::Path, with_go: bool) {
    fs::write(repo.join("retained.rs"), "pub fn retained() {}\n").unwrap();
    fs::write(
        repo.join("retained.tsx"),
        "export function Boundary() { return <div />; }\nexport function View() { return <Boundary />; }\n",
    )
    .unwrap();
    if with_go {
        fs::create_dir_all(repo.join("ast")).unwrap();
        fs::create_dir_all(repo.join("vm")).unwrap();
        fs::write(
            repo.join("ast/stmt.go"),
            "package ast\ntype VarStmt struct{}\n",
        )
        .unwrap();
        fs::write(
            repo.join("vm/vmStmt.go"),
            "package vm\nimport \"example/ast\"\nfunc runSingleStmt(stmt interface{}) {\n switch stmt.(type) {\n case *ast.VarStmt:\n  _ = ast.VarStmt{}\n }\n}\n",
        )
        .unwrap();
    }
    let mut store = Store::open(path).unwrap();
    index(&mut store, repo, "anko").unwrap();
    let node = store
        .get_node_by_qname("anko", "retained.rs::Function::retained")
        .unwrap()
        .unwrap();
    store
        .upsert_vector_embedding(&greppy_store::NewVectorEmbedding {
            project: "anko".into(),
            model_id: "retained-model".into(),
            prompt_version: "retained-v1".into(),
            task: "code".into(),
            node_id: Some(node.id),
            chunk_idx: 0,
            qualified_name: node.qualified_name,
            file_path: node.file_path,
            start_line: 1,
            end_line: 1,
            content_sha256: "a".repeat(64),
            graph_generation: 1,
            vector: vec![1.0, 0.0],
        })
        .unwrap();
    // Reproduce the old cache: completed v4 JS/TS, same node/file/vector data,
    // but missing Go usages in both raw facts and the resolved graph.
    store
        .conn()
        .execute_batch(
            "DELETE FROM raw_edges WHERE edge_type='USAGE' AND file_path LIKE '%.go';
         DELETE FROM edges WHERE edge_type='USAGE' AND source_id IN
           (SELECT id FROM nodes WHERE file_path LIKE '%.go');
         INSERT OR REPLACE INTO schema_meta(key,value)
           VALUES('greppy.js_ts_usage_repair_v4','complete');",
        )
        .unwrap();
    store
        .conn()
        .execute(
            "DELETE FROM schema_meta WHERE key=?1",
            [greppy_indexer::JS_TS_USAGE_REPAIR_KEY],
        )
        .unwrap();
}

fn preserved_cache(store: &Store) -> Vec<String> {
    let mut snapshot = vec![
        format!("{:?}", store.list_nodes("anko", "", "", 0, 1000).unwrap()),
        format!("{:?}", store.list_file_states("anko").unwrap()),
        format!("{:?}", store.list_workspace_states().unwrap()),
        format!(
            "{:?}",
            store
                .list_raw_edges("anko")
                .unwrap()
                .into_iter()
                .filter(|edge| !edge.file_path.ends_with(".go"))
                .map(|edge| (
                    edge.file_path,
                    edge.source_qname,
                    edge.target_qname,
                    edge.edge_type,
                    edge.properties
                ))
                .collect::<Vec<_>>()
        ),
    ];
    let mut statement = store
        .conn()
        .prepare("SELECT * FROM vector_embeddings ORDER BY id")
        .unwrap();
    let columns = statement.column_count();
    let vectors = statement
        .query_map([], |row| {
            (0..columns)
                .map(|i| row.get::<_, rusqlite::types::Value>(i))
                .collect::<rusqlite::Result<Vec<_>>>()
        })
        .unwrap()
        .collect::<rusqlite::Result<Vec<_>>>()
        .unwrap();
    assert!(
        !vectors.is_empty(),
        "the reuse check must protect real cached vectors"
    );
    snapshot.push(format!("{vectors:?}"));
    snapshot
}

fn assert_go_cache_repair(store: &mut Store, root: &std::path::Path) {
    let before = preserved_cache(store);
    assert!(
        greppy_indexer::recover_persisted_js_ts_usages(store, "anko", root).unwrap(),
        "old Go usage cache must be repaired"
    );
    let target = store
        .get_node_by_qname("anko", "ast/stmt.go::Class::VarStmt")
        .unwrap()
        .unwrap();
    let caller = store
        .get_node_by_qname("anko", "vm/vmStmt.go::Function::runSingleStmt")
        .unwrap()
        .unwrap();
    assert!(
        store
            .incoming_edges(target.id, Some("USAGE"), 30)
            .unwrap()
            .iter()
            .any(|edge| edge.source_id == caller.id),
        "source-validated cache repair must recover the missing Go usage"
    );
    assert!(!greppy_indexer::recover_persisted_js_ts_usages(store, "anko", root).unwrap());
    assert_eq!(
        before,
        preserved_cache(store),
        "only Go references may change"
    );
}

#[test]
fn old_go_cache_repairs_private_and_overlay_without_invalidating_mixed_base() {
    let repo = tempfile::tempdir().unwrap();
    let scratch = tempfile::tempdir().unwrap();
    let base_path = scratch.path().join("base.db");
    mixed_cache(repo.path(), &base_path, true);
    let base_bytes = fs::read(&base_path).unwrap();
    {
        let mut overlay = Store::open_overlay(
            &base_path,
            &scratch.path().join("delta.db"),
            &greppy_store::VisibilityIndex::default(),
        )
        .unwrap();
        // A v4-certified Delta may already mask/re-export JS/TS Base facts.
        // The Go-only upgrade must extend that namespace, not overwrite it.
        let js_rows: Vec<_> = overlay
            .list_raw_edges("anko")
            .unwrap()
            .into_iter()
            .filter(|edge| edge.file_path == "retained.tsx" && edge.edge_type == "USAGE")
            .map(|edge| {
                serde_json::json!({
                    "file_path": edge.file_path, "source_qname": edge.source_qname,
                    "target_qname": edge.target_qname, "properties": edge.properties,
                })
            })
            .collect();
        assert!(!js_rows.is_empty());
        overlay
            .conn()
            .execute(
                "INSERT INTO main.schema_meta(key,value) VALUES(?1,?2)",
                rusqlite::params![
                    "greppy.js_ts_usage_override_files.anko",
                    r#"["retained.tsx"]"#
                ],
            )
            .unwrap();
        overlay
            .conn()
            .execute(
                "INSERT INTO main.schema_meta(key,value) VALUES(?1,?2)",
                rusqlite::params![
                    "greppy.js_ts_usage_override_rows.anko",
                    serde_json::to_string(&js_rows).unwrap()
                ],
            )
            .unwrap();
        overlay.conn().execute(
            "INSERT INTO main.schema_meta(key,value) VALUES('greppy.js_ts_usage_repair_v4','complete')",
            [],
        ).unwrap();
        assert_go_cache_repair(&mut overlay, repo.path());
        assert!(overlay.list_delta_raw_edges("anko").unwrap().is_empty());
        assert_eq!(
            overlay
                .conn()
                .query_row::<i64, _, _>("SELECT count(*) FROM main.file_state", [], |row| row
                    .get(0))
                .unwrap(),
            0,
            "reference repair must not take ownership of unchanged Base files"
        );
    }
    assert_eq!(
        base_bytes,
        fs::read(&base_path).unwrap(),
        "Base bytes are immutable"
    );
    let mut private = Store::open(&base_path).unwrap();
    assert_go_cache_repair(&mut private, repo.path());
}

#[test]
fn completed_v4_cache_without_go_reuses_js_ts_without_another_pass() {
    let repo = tempfile::tempdir().unwrap();
    let scratch = tempfile::tempdir().unwrap();
    let path = scratch.path().join("cache.db");
    mixed_cache(repo.path(), &path, false);
    let mut store = Store::open(&path).unwrap();
    let before = preserved_cache(&store);
    assert!(
        !greppy_indexer::recover_persisted_js_ts_usages(&mut store, "anko", repo.path()).unwrap(),
        "completed v4 JS/TS needs no repeated source/edge repair"
    );
    assert!(greppy_indexer::js_ts_usages_repaired(&store).unwrap());
    assert!(
        !greppy_indexer::recover_persisted_js_ts_usages(&mut store, "anko", repo.path()).unwrap()
    );
    assert_eq!(before, preserved_cache(&store));
}
