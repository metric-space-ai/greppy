// Post-replacement durability failures must not be mistaken for rollback.
use super::*;

#[test]
fn unconfirmed_directory_commit_requires_reload_before_history_can_be_acknowledged() {
    let root = tempfile::tempdir().unwrap();
    let (mut server, rx) = fixture(root.path());
    initialize(&server, &rx);
    let id = new_session(&server, &rx, root.path());
    let prepared = prepared_for(&server, &id);
    let store = SessionStore::new(&prepared.data_root, &prepared.project);
    let hook_store = store.clone();
    let hook_id = id.clone();
    server.config.after_import_rename = Some(Arc::new(move || {
        // The replacement is already visible when directory durability fails.
        assert_eq!(hook_store.load(&hook_id)?.import_ack.len(), 2);
        Err(io::Error::other(
            "injected directory-sync failure after import replacement",
        ))
    }));
    let input = json!([
        {"id":"durable-user","role":"user","text":"Archived question"},
        {"id":"durable-assistant","role":"assistant","text":"Archived answer"}
    ]);
    let response = request(
        &server,
        &rx,
        "import",
        "_workjet/import_history",
        json!({"sessionId":id,"messages":input}),
    );
    assert_eq!(response["error"]["code"], -32603);
    assert!(response.get("result").is_none());
    {
        let state = lock_state(&server.state);
        let session = &state.sessions[&id];
        assert!(session.closed);
        assert!(session.messages.is_empty());
        assert!(session.import_ack.is_empty());
    }
    let saved = store.load(&id).unwrap();
    assert_eq!(saved.messages.len(), 2);
    assert_eq!(saved.import_ack.len(), 2);
    let bytes = std::fs::read(store.path_for(&id).unwrap()).unwrap();
    assert_eq!(
        request(
            &server,
            &rx,
            "retry-closed",
            "_workjet/import_history",
            json!({"sessionId":id,"messages":input})
        )["error"]["code"],
        -32600
    );
    server.config.after_import_rename = None;
    let loaded = server.session_load(
        &json!("reload"),
        &json!({"sessionId":id,"cwd":root.path()}),
        "session/resume",
    );
    assert!(loaded.get("error").is_none());
    assert!(rx
        .try_iter()
        .all(|frame| frame["method"] == "session/update"));
    let retry = request(
        &server,
        &rx,
        "retry-loaded",
        "_workjet/import_history",
        json!({"sessionId":id,"messages":input}),
    );
    assert_eq!(
        retry["result"]["acceptedMessageIds"],
        json!(["durable-user", "durable-assistant"])
    );
    assert_eq!(std::fs::read(store.path_for(&id).unwrap()).unwrap(), bytes);
    assert_eq!(store.load(&id).unwrap(), saved);
    assert_eq!(
        messages_from_protocol(&prepared_for(&server, &id).history),
        saved.messages
    );
    assert!(!lock_state(&server.state).sessions[&id].closed);
}
