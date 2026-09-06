use super::*;

#[test]
fn definition_read_rechecks_bytes_after_freshness_gate() {
    let root = tempfile::tempdir().unwrap();
    let mut store = greppy_store::Store::open_memory().unwrap();
    store
        .upsert_project(&greppy_store::Project {
            name: "test".into(),
            indexed_at: "test".into(),
            root_path: root.path().to_string_lossy().into_owned(),
        })
        .unwrap();
    let original = "fn target() { let _ = 1; }\n";
    let path = root.path().join("lib.rs");
    std::fs::write(&path, original).unwrap();
    store
        .upsert_file_state(&greppy_store::FileState {
            project: "test".into(),
            rel_path: "lib.rs".into(),
            language: "rust".into(),
            sha256: read_sha256(original.as_bytes()),
            mtime_ns: 0,
            size: original.len() as i64,
            parser_version: "test".into(),
            extractor_version: "test".into(),
            last_indexed_generation: 1,
        })
        .unwrap();
    let node = greppy_store::Node {
        id: 1,
        project: "test".into(),
        label: "Function".into(),
        name: "target".into(),
        qualified_name: "lib.rs::Function::target".into(),
        file_path: "lib.rs".into(),
        start_line: 1,
        end_line: 1,
        properties: serde_json::json!({}),
    };
    assert!(read_definition(&store, root.path(), node.clone())
        .unwrap()
        .is_some());
    // Same-size edits also invalidate the source, independent of line count
    // or metadata timing. This models a write after the outer freshness gate.
    std::fs::write(&path, original.replace("= 1", "= 2")).unwrap();
    let result = read_definition(&store, root.path(), node.clone());
    assert!(
        matches!(result, Err(Error::Workspace(message)) if message.contains("no stale span emitted") && message.contains("greppy read-file"))
    );
    std::fs::write(&path, original).unwrap();
    store.delete_file_state("test", "lib.rs").unwrap();
    assert!(matches!(
        read_definition(&store, root.path(), node),
        Err(Error::Workspace(_))
    ));
}
