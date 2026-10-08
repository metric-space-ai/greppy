use super::*;

#[test]
fn bounded_file_span_stops_before_invalid_tail() {
    let prefix = b"alpha\nbeta\ngamma\n";
    let data = [prefix.as_slice(), b"\xff"].concat();
    let mut reader = std::io::Cursor::new(data);
    let (text, start, end) = read_bounded_file_range(&mut reader, "1:3", "dump").unwrap();
    assert_eq!(text.as_bytes(), prefix);
    assert_eq!((start, end), (1, 3));
    assert_eq!(reader.position(), prefix.len() as u64);
}

#[test]
fn bounded_file_span_skips_unselected_encoding_and_preserves_crlf() {
    let mut reader = std::io::Cursor::new(b"\xff\nbeta\r\ngamma\n\xff");
    let (text, start, end) = read_bounded_file_range(&mut reader, "2:3", "dump").unwrap();
    assert_eq!(text, "beta\r\ngamma\n");
    assert_eq!((start, end), (2, 3));
}

#[test]
fn bounded_file_span_rejects_selected_encoding_and_reports_actual_eof() {
    let mut reader = std::io::Cursor::new(b"alpha\n\xff\n");
    let error = read_bounded_file_range(&mut reader, "1:2", "dump").unwrap_err();
    assert!(error.to_string().contains("requested lines 1:2"));
    assert!(error.to_string().contains("UTF-8"));
    let mut reader = std::io::Cursor::new(b"alpha\nbeta");
    let (text, start, end) = read_bounded_file_range(&mut reader, "1:3", "dump").unwrap();
    assert_eq!(text, "alpha\nbeta");
    assert_eq!((start, end), (1, 2));
    let mut reader = std::io::Cursor::new(b"alpha\nbeta");
    let error = read_bounded_file_range(&mut reader, "3:4", "dump").unwrap_err();
    assert!(error.to_string().contains("file has 2 lines"));
    for raw in ["0:1", "3:1", "1", "a:2"] {
        let mut reader = std::io::Cursor::new(b"alpha\n");
        assert!(read_bounded_file_range(&mut reader, raw, "dump").is_err());
        assert_eq!(
            reader.position(),
            0,
            "invalid ranges must not consume input"
        );
    }
}

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
