// Regression tests for Workjet's immutable archive synchronization extension.
use super::*;
use greppy_agent::{ContentPart, Role, StopReason, Usage};

fn archive() -> Value {
    json!([
        {"id":"archived-user","role":"user","text":"Archived planning"},
        {"id":"archived-assistant","role":"assistant","text":"Keep the fixture readable and grouped by project."}
    ])
}

fn sync(server: &Server, rx: &Receiver<Value>, id: &str, messages: Value) -> Value {
    request(
        server,
        rx,
        "import",
        "_workjet/import_history",
        json!({"sessionId":id,"messages":messages}),
    )
}

struct CaptureModel {
    requests: Vec<ModelRequest>,
}

impl ModelStream for CaptureModel {
    fn stream_turn(
        &mut self,
        request: &ModelRequest,
        _on_event: &mut dyn FnMut(StreamEvent),
    ) -> Result<TurnResult, ClientError> {
        self.requests.push(request.clone());
        Ok(TurnResult {
            message: Message {
                role: Role::Assistant,
                content: vec![ContentPart::Text {
                    text: "Captured model continuation".into(),
                }],
            },
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
        })
    }
}

#[test]
fn imports_retries_appends_and_restart_preserve_actual_model_history() {
    let root = tempfile::tempdir().unwrap();
    let (server, rx) = fixture(root.path());
    let initialized = request(
        &server,
        &rx,
        "init",
        "initialize",
        json!({"protocolVersion":1}),
    );
    assert_eq!(
        initialized["result"]["agentCapabilities"]["_meta"]["workjetImportHistory"]["version"],
        1
    );
    let id = new_session(&server, &rx, root.path());
    let initial = archive();
    let accepted = json!(["archived-user", "archived-assistant"]);
    assert_eq!(
        sync(&server, &rx, &id, initial.clone())["result"]["acceptedMessageIds"],
        accepted
    );
    let prepared = prepared_for(&server, &id);
    let store = SessionStore::new(&prepared.data_root, &prepared.project);
    let before = std::fs::read(store.path_for(&id).unwrap()).unwrap();
    assert_eq!(
        sync(&server, &rx, &id, initial.clone())["result"]["acceptedMessageIds"],
        accepted
    );
    assert_eq!(std::fs::read(store.path_for(&id).unwrap()).unwrap(), before);

    // Use the normal turn completion path so the archive coexists with native history.
    let mut done = completed_fixture(&prepared.history, "native");
    done.messages.insert(
        3,
        Message {
            role: Role::Assistant,
            content: vec![ContentPart::ToolCall {
                id: "native-tool".into(),
                name: "fixture_read".into(),
                arguments: json!({"path":"fixture"}),
            }],
        },
    );
    done.messages.insert(
        4,
        Message {
            role: Role::User,
            content: vec![ContentPart::ToolResult {
                call_id: "native-tool".into(),
                content: "native tool result".into(),
                is_error: false,
            }],
        },
    );
    finish_prompt(&server.state, &prepared, &server.config, &done).unwrap();
    let mut appended = initial.as_array().unwrap().clone();
    appended.extend([
        json!({"id":"later-user","role":"user","text":"Later archived question"}),
        json!({"id":"later-assistant","role":"assistant","text":"Later archived answer"}),
    ]);
    let appended = Value::Array(appended);
    assert!(sync(&server, &rx, &id, appended.clone())
        .get("error")
        .is_none());
    let history = prepared_for(&server, &id).history;
    assert_eq!(&history[..done.messages.len()], done.messages.as_slice());
    assert_eq!(history.len(), done.messages.len() + 2);
    let mut model = CaptureModel { requests: vec![] };
    let calls = Arc::new(AtomicUsize::new(0));
    let mut env = CountTools(Arc::clone(&calls));
    let result = run_agent_loop_with_history(
        &mut model,
        &mut env,
        &AgentConfig::default()
            .with_model("fixture-model")
            .with_max_turns(1),
        &history,
        "Current Workjet prompt",
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(result.stop, LoopStop::EndTurn);
    assert_eq!(model.requests.len(), 1);
    let request = &model.requests[0];
    assert_eq!(&request.messages[..history.len()], history.as_slice());
    assert_eq!(request.messages.len(), history.len() + 1);
    assert_eq!(
        request.messages.last().unwrap(),
        &Message {
            role: Role::User,
            content: vec![ContentPart::Text {
                text: "Current Workjet prompt".into()
            }],
        }
    );
    for (role, text) in [
        (Role::User, "Archived planning"),
        (
            Role::Assistant,
            "Keep the fixture readable and grouped by project.",
        ),
        (Role::User, "Later archived question"),
        (Role::Assistant, "Later archived answer"),
    ] {
        assert_eq!(
            request
                .messages
                .iter()
                .filter(|message| message.role == role
                    && message.content == vec![ContentPart::Text { text: text.into() }])
                .count(),
            1
        );
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let saved = store.load(&id).unwrap();
    assert_eq!(saved.turns, 1);
    assert_eq!(saved.usage, done.usage);
    assert_eq!(saved.import_ack.len(), 4);
    assert_eq!(saved.messages, messages_from_protocol(&history));

    // A fresh server reloads the persisted acknowledgement before retrying.
    let (restarted, restarted_rx) = fixture(root.path());
    initialize(&restarted, &restarted_rx);
    let loaded = restarted.session_load(
        &json!("reload"),
        &json!({"sessionId":id,"cwd":root.path()}),
        "session/resume",
    );
    assert!(loaded.get("error").is_none());
    assert!(restarted_rx
        .try_iter()
        .all(|frame| frame["method"] == "session/update"));
    let bytes = std::fs::read(store.path_for(&id).unwrap()).unwrap();
    assert!(sync(&restarted, &restarted_rx, &id, appended)
        .get("error")
        .is_none());
    assert_eq!(std::fs::read(store.path_for(&id).unwrap()).unwrap(), bytes);
    assert_eq!(store.load(&id).unwrap(), saved);
    assert_eq!(prepared_for(&restarted, &id).history, history);
}

#[test]
fn failed_import_commit_keeps_disk_and_live_acknowledgement_unchanged() {
    let root = tempfile::tempdir().unwrap();
    let (mut server, rx) = fixture(root.path());
    initialize(&server, &rx);
    let id = new_session(&server, &rx, root.path());
    let prepared = prepared_for(&server, &id);
    let store = SessionStore::new(&prepared.data_root, &prepared.project);
    let original = store.load(&id).unwrap();
    let before = std::fs::read(store.path_for(&id).unwrap()).unwrap();
    let attempts = Arc::new(AtomicUsize::new(0));
    let hook_attempts = Arc::clone(&attempts);
    let hook_store = store.clone();
    let hook_id = id.clone();
    let hook_original = original.clone();
    server.config.after_messages = Some(Arc::new(move || {
        if hook_attempts.fetch_add(1, Ordering::SeqCst) == 0 {
            assert_eq!(hook_store.load(&hook_id)?, hook_original);
            return Err(io::Error::other("injected imported history commit failure"));
        }
        Ok(())
    }));
    let failed = sync(&server, &rx, &id, archive());
    assert_eq!(failed["error"]["code"], -32603);
    assert_eq!(store.load(&id).unwrap(), original);
    assert_eq!(std::fs::read(store.path_for(&id).unwrap()).unwrap(), before);
    {
        let state = lock_state(&server.state);
        assert!(state.sessions[&id].messages.is_empty());
        assert!(state.sessions[&id].import_ack.is_empty());
    }
    assert!(std::fs::read_dir(store.project_dir())
        .unwrap()
        .all(|entry| entry
            .unwrap()
            .path()
            .extension()
            .is_none_or(|ext| ext != "pending")));
    assert!(sync(&server, &rx, &id, archive()).get("error").is_none());
    let saved = store.load(&id).unwrap();
    assert_eq!(saved.messages.len(), 2);
    assert_eq!(saved.import_ack.len(), 2);
    assert_eq!(saved.turns, 0);
    assert_eq!(saved.usage, Usage::default());
    assert!(sync(&server, &rx, &id, archive()).get("error").is_none());
    assert_eq!(attempts.load(Ordering::SeqCst), 2);
    assert_eq!(store.load(&id).unwrap(), saved);
}

#[test]
fn changed_shortened_duplicate_and_invalid_archives_do_not_mutate_sessions() {
    let root = tempfile::tempdir().unwrap();
    let (server, rx) = fixture(root.path());
    initialize(&server, &rx);
    let id = new_session(&server, &rx, root.path());
    assert!(sync(&server, &rx, &id, archive()).get("error").is_none());
    let prepared = prepared_for(&server, &id);
    let store = SessionStore::new(&prepared.data_root, &prepared.project);
    let saved = store.load(&id).unwrap();
    let before = std::fs::read(store.path_for(&id).unwrap()).unwrap();
    let mut changed = archive();
    changed[0]["text"] = json!("Changed immutable source");
    let mut changed_role = archive();
    changed_role[0]["role"] = json!("assistant");
    for input in [
        changed,
        changed_role,
        json!([]),
        json!([{"id":"duplicate","role":"user","text":"one"},{"id":"duplicate","role":"assistant","text":"two"}]),
        json!([{"id":"invalid","role":"system","text":"system input"}]),
        json!([{"id":" ","role":"user","text":"invalid identity"}]),
    ] {
        assert_eq!(sync(&server, &rx, &id, input)["error"]["code"], -32602);
        assert_eq!(store.load(&id).unwrap(), saved);
        assert_eq!(prepared_for(&server, &id).history, prepared.history);
        assert_eq!(std::fs::read(store.path_for(&id).unwrap()).unwrap(), before);
    }
    for (busy, closed) in [(true, false), (false, true)] {
        {
            let mut state = lock_state(&server.state);
            let session = state.sessions.get_mut(&id).unwrap();
            session.busy = busy;
            session.closed = closed;
        }
        assert_eq!(sync(&server, &rx, &id, archive())["error"]["code"], -32600);
        assert_eq!(store.load(&id).unwrap(), saved);
    }
}

#[test]
fn stale_writer_is_rejected_and_idle_cancel_does_not_forget_imported_ids() {
    let root = tempfile::tempdir().unwrap();
    let (first, rx) = fixture(root.path());
    initialize(&first, &rx);
    let id = new_session(&first, &rx, root.path());
    let (stale, stale_rx) = fixture(root.path());
    initialize(&stale, &stale_rx);
    let loaded = stale.session_load(
        &json!("reload"),
        &json!({"sessionId":id,"cwd":root.path()}),
        "session/resume",
    );
    assert!(loaded.get("error").is_none());
    assert!(sync(&first, &rx, &id, archive()).get("error").is_none());
    let prepared = prepared_for(&first, &id);
    let store = SessionStore::new(&prepared.data_root, &prepared.project);
    let saved = store.load(&id).unwrap();
    assert_eq!(
        sync(&stale, &stale_rx, &id, archive())["error"]["code"],
        -32603
    );
    assert_eq!(
        sync(&stale, &stale_rx, &id, json!([]))["error"]["code"],
        -32603
    );
    assert!(prepared_for(&stale, &id).history.is_empty());
    assert_eq!(store.load(&id).unwrap(), saved);
    first.dispatch_line(
        &json!({
            "jsonrpc":"2.0","method":"session/cancel","params":{"sessionId":id},"id":"","headers":[]
        })
        .to_string(),
    );
    assert!(rx.try_iter().next().is_none());
    assert!(sync(&first, &rx, &id, archive()).get("error").is_none());
    assert_eq!(store.load(&id).unwrap(), saved);
}

#[test]
fn unchanged_archive_cannot_acknowledge_a_missing_durable_log() {
    let root = tempfile::tempdir().unwrap();
    let (server, rx) = fixture(root.path());
    initialize(&server, &rx);
    let id = new_session(&server, &rx, root.path());
    assert!(sync(&server, &rx, &id, archive()).get("error").is_none());
    let prepared = prepared_for(&server, &id);
    let store = SessionStore::new(&prepared.data_root, &prepared.project);
    std::fs::remove_file(store.path_for(&id).unwrap()).unwrap();
    assert_eq!(sync(&server, &rx, &id, archive())["error"]["code"], -32603);
    assert_eq!(prepared_for(&server, &id).history, prepared.history);
}
