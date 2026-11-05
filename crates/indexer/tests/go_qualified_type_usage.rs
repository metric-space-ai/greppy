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
