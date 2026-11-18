#[path = "agent_acp_import_tests.rs"]
mod imported_history;
#[path = "agent_acp_import_durability_tests.rs"]
mod imported_history_durability;
#[test]
fn caller_output_limit_belongs_only_to_the_selected_acp_model() {
    let root = tempfile::tempdir().unwrap();
    let (mut server, rx) = fixture(root.path());
    server.config.max_output_tokens = Some(OutputTokenLimit::try_from(65536).unwrap());
    let init = request(
        &server,
        &rx,
        "init",
        "initialize",
        json!({"protocolVersion":1}),
    );
    let capability = &init["result"]["agentCapabilities"]["_meta"]["greppyOutputLimit"];
    assert_eq!(
        capability,
        &json!({"version":1,"method":"_greppy/set_output_limit"})
    );
    let method = capability["method"].as_str().unwrap();
    let id = new_session(&server, &rx, root.path());
    assert_eq!(
        prepared_for(&server, &id).max_output_tokens.unwrap().get(),
        65536
    );

    // Free route metadata must never masquerade as a standard ACP selector.
    let rejected = request(
        &server,
        &rx,
        "standard-numeric",
        "session/set_config_option",
        json!({
            "sessionId":id, "configId":"max_output_tokens", "value":131072,
            "modelId":"fixture-model", "endpoint":"http://127.0.0.1:1"
        }),
    );
    assert_eq!(rejected["error"]["code"], -32602);
    assert_eq!(
        prepared_for(&server, &id).max_output_tokens.unwrap().get(),
        65536
    );
    for (selection, params) in [
        (
            "session/set_model",
            json!({"sessionId":id, "modelId":"MiniMax-M3"}),
        ),
        (
            "session/set_config_option",
            json!({"sessionId":id, "configId":"model", "value":"fixture-model"}),
        ),
    ] {
        let result = request(&server, &rx, "switch", selection, params);
        assert!(result.get("error").is_none(), "{result}");
        if selection == "session/set_config_option" {
            assert_eq!(
                result["result"]["configOptions"],
                json!([model_config("fixture-model")])
            );
        }
        assert_eq!(prepared_for(&server, &id).max_output_tokens, None);
    }
    // Exercise the advertised method through real framed dispatch, not its helper.
    for (model, endpoint, value) in [
        ("MiniMax-M3", "http://127.0.0.1:1", json!(131072)),
        ("fixture-model", "http://127.0.0.1:2", json!(131072)),
        ("fixture-model", "http://127.0.0.1:1", json!(0)),
        ("fixture-model", "http://127.0.0.1:1", json!(u64::MAX)),
        ("fixture-model", "http://127.0.0.1:1", json!(-1)),
        ("fixture-model", "http://127.0.0.1:1", json!(1.5)),
        ("fixture-model", "http://127.0.0.1:1", json!("131072")),
        ("fixture-model", "http://127.0.0.1:1", json!(true)),
    ] {
        let rejected = request(
            &server,
            &rx,
            "invalid",
            method,
            json!({
                "sessionId":id, "maxOutputTokens":value, "modelId":model, "endpoint":endpoint
            }),
        );
        assert_eq!(rejected["error"]["code"], -32602, "{rejected}");
        assert_eq!(prepared_for(&server, &id).max_output_tokens, None);
    }
    let metadata = json!({"sessionId":id, "maxOutputTokens":131072,
                         "modelId":"fixture-model", "endpoint":"http://127.0.0.1:1"});
    let refreshed = request(&server, &rx, "fresh", method, metadata.clone());
    assert_eq!(refreshed["result"], metadata);
    assert_eq!(
        prepared_for(&server, &id).max_output_tokens.unwrap().get(),
        131072
    );
    lock_state(&server.state)
        .sessions
        .get_mut(&id)
        .unwrap()
        .busy = true;
    let busy = request(&server, &rx, "busy", method, metadata.clone());
    assert_eq!(busy["error"]["code"], -32600);
    lock_state(&server.state)
        .sessions
        .get_mut(&id)
        .unwrap()
        .busy = false;
    assert!(request(
        &server,
        &rx,
        "close",
        "session/close",
        json!({"sessionId":id})
    )
    .get("error")
    .is_none());
    let closed = request(&server, &rx, "closed", method, metadata);
    assert_eq!(closed["error"]["code"], -32600);
    let loaded = request(
        &server,
        &rx,
        "load",
        "session/load",
        json!({
            "sessionId":id, "cwd":root.path(), "mcpServers":[]
        }),
    );
    assert!(loaded.get("error").is_none(), "{loaded}");
    assert_eq!(
        prepared_for(&server, &id).max_output_tokens,
        None,
        "resume must not revive the process-start bound"
    );
}

#[test]
fn rejected_model_changes_preserve_the_active_and_persisted_model() {
    let root = tempfile::tempdir().unwrap();
    let (server, rx) = fixture(root.path());
    initialize(&server, &rx);
    let id = new_session(&server, &rx, root.path());
    lock_state(&server.state)
        .sessions
        .get_mut(&id)
        .unwrap()
        .busy = true;
    let rejected = request(
        &server,
        &rx,
        "busy-model",
        "session/set_model",
        json!({"sessionId":id, "modelId":"rejected-busy-model"}),
    );
    assert_eq!(rejected["error"]["code"], -32600);
    {
        let mut state = lock_state(&server.state);
        let session = state.sessions.get_mut(&id).unwrap();
        assert_eq!(session.model, "fixture-model");
        session.busy = false;
        let (data_root, project) = server.store_identity(&session.cwd);
        assert_eq!(
            SessionStore::new(data_root, project)
                .load(&id)
                .unwrap()
                .model,
            "fixture-model"
        );
    }
    let data_root = server.config.data_root.as_ref().unwrap();
    std::fs::remove_dir_all(data_root).unwrap();
    std::fs::write(data_root, b"fixture blocks model persistence").unwrap();
    let rejected = request(
        &server,
        &rx,
        "unsaved-model",
        "session/set_config_option",
        json!({"sessionId":id, "configId":"model", "value":"unsaved-model"}),
    );
    assert_eq!(rejected["error"]["code"], -32603);
    assert_eq!(
        lock_state(&server.state).sessions[&id].model,
        "fixture-model"
    );
}

#[test]
fn persistence_failure_is_reported_instead_of_claiming_a_saved_turn() {
    let root = tempfile::tempdir().unwrap();
    let (server, rx) = fixture(root.path());
    initialize(&server, &rx);
    let id = new_session(&server, &rx, root.path());
    let prepared = {
        let state = lock_state(&server.state);
        let session = &state.sessions[&id];
        PreparedPrompt {
            session_id: id,
            cwd: session.cwd.clone(),
            data_root: session.data_root.clone(),
            project: session.project.clone(),
            model: session.model.clone(),
            max_output_tokens: session.max_output_tokens,
            history: vec![],
            cancel: Arc::clone(&session.cancel),
            perms: Arc::clone(&session.perms),
            client_tools: Vec::new(),
            images: Vec::new(),
            mcp: Default::default(),
            context_summary: None,
            choices: test_choices(),
            context_window: None,
            extra_dirs: Vec::new(),
            client: Value::Null,
            effort: None,
            compact_at: None,
            context_tokens: None,
        }
    };
    let data_root = server.config.data_root.as_ref().unwrap();
    std::fs::remove_dir_all(data_root).unwrap();
    std::fs::write(data_root, b"fixture blocks persistence").unwrap();
    let done = PromptDone {
        context_summary: None,
        messages: vec![Message {
            role: greppy_agent::Role::User,
            content: vec![greppy_agent::ContentPart::Text {
                text: "fixture".into(),
            }],
        }],
        usage: greppy_agent::Usage::default(),
        stop_reason: "end_turn",
    };
    assert!(persist_turn(&prepared, &server.config, &done)
        .unwrap_err()
        .contains("cannot persist session history"));
}

// ACP boundary regressions without a model, real tools, or user data.
use super::*;
use greppy_agent::ToolDefinition;
use std::sync::atomic::AtomicUsize;

struct Frames {
    bytes: Vec<u8>,
    tx: Sender<Value>,
}

impl Write for Frames {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.bytes.extend_from_slice(bytes);
        while let Some(end) = self.bytes.iter().position(|byte| *byte == b'\n') {
            let frame: Vec<u8> = self.bytes.drain(..=end).collect();
            let value = serde_json::from_slice(&frame)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
            self.tx
                .send(value)
                .map_err(|error| io::Error::other(error.to_string()))?;
        }
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn fixture(root: &Path) -> (Server, Receiver<Value>) {
    let (tx, rx) = mpsc::channel();
    let config = AcpConfig {
        endpoint: "http://127.0.0.1:1".into(),
        model: "fixture-model".into(),
        api_key: None,
        headers: Vec::new(),
        sandbox: false,
        max_turns: 2,
        max_output_tokens: None,
        capabilities: greppy_agent::CapabilitySet::default(),
        data_root: Some(root.join("state")),
        after_messages: None,
        after_import_rename: None,
        tool_env: None,
    };
    (
        Server::new(
            Frames {
                bytes: Vec::new(),
                tx,
            },
            config,
        ),
        rx,
    )
}

fn next(rx: &Receiver<Value>) -> Value {
    rx.recv_timeout(Duration::from_secs(2))
        .expect("bounded ACP frame")
}

fn request(server: &Server, rx: &Receiver<Value>, id: &str, method: &str, params: Value) -> Value {
    server.dispatch_line(
        &json!({"jsonrpc":"2.0", "id":id, "method":method, "params":params, "headers":[]})
            .to_string(),
    );
    let response = next(rx);
    if method == "_workjet/import_history" && response.get("error").is_some() {
        eprintln!("import_history RPC failure: {response}");
    }
    assert_eq!(response["id"], id);
    // A set-up session is told its slash commands right after the answer.
    if matches!(method, "session/new" | "session/load" | "session/resume")
        && response.get("error").is_none()
    {
        let commands = next(rx);
        assert_eq!(
            commands["params"]["update"]["sessionUpdate"], "available_commands_update",
            "{commands}"
        );
    }
    response
}

fn initialize(server: &Server, rx: &Receiver<Value>) {
    assert_eq!(
        request(
            server,
            rx,
            "init",
            "initialize",
            json!({"protocolVersion":1})
        )["result"]["protocolVersion"],
        1
    );
}

fn new_session(server: &Server, rx: &Receiver<Value>, cwd: &Path) -> String {
    request(
        server,
        rx,
        "new",
        "session/new",
        json!({"cwd":cwd, "mcpServers":[]}),
    )["result"]["sessionId"]
        .as_str()
        .expect("persisted session id")
        .to_owned()
}

#[test]
fn initialize_and_invalid_requests_have_framed_typed_errors() {
    let root = tempfile::tempdir().unwrap();
    let (server, rx) = fixture(root.path());
    let early = request(&server, &rx, "early", "session/new", json!({}));
    assert_eq!(early["error"]["code"], -32600);
    assert_eq!(early["error"]["_tag"], "Cause");
    assert_eq!(early["error"]["data"][0]["_tag"], "Fail");
    let init = request(
        &server,
        &rx,
        "init",
        "initialize",
        json!({"protocolVersion":1}),
    );
    // Pictures go to the model after a prompt's text (png, jpeg, gif, webp).
    assert_eq!(
        init["result"]["agentCapabilities"]["promptCapabilities"]["image"],
        true
    );
    assert_eq!(init["result"]["agentCapabilities"]["loadSession"], true);
    server.dispatch_line("invalid JSON");
    assert_eq!(next(&rx)["error"]["code"], -32700);
    let unsupported = request(&server, &rx, "unsupported", "session/fork", json!({}));
    assert_eq!(unsupported["error"]["code"], -32601);
}

#[test]
fn session_model_survives_a_server_restart() {
    let root = tempfile::tempdir().unwrap();
    let (first, rx) = fixture(root.path());
    initialize(&first, &rx);
    let id = new_session(&first, &rx, root.path());
    let changed = request(
        &first,
        &rx,
        "model",
        "session/set_model",
        json!({"sessionId":id, "modelId":"recorded-model"}),
    );
    assert!(changed.get("result").is_some());
    drop(first);
    let (second, rx) = fixture(root.path());
    initialize(&second, &rx);
    let loaded = request(
        &second,
        &rx,
        "load",
        "session/load",
        json!({"sessionId":id, "cwd":root.path(), "mcpServers":[]}),
    );
    assert!(loaded.get("result").is_some(), "{loaded}");
    assert_eq!(
        lock_state(&second.state).sessions[&id].model,
        "recorded-model"
    );
    let missing = request(
        &second,
        &rx,
        "missing",
        "session/load",
        json!({"sessionId":"missing-session", "cwd":root.path(), "mcpServers":[]}),
    );
    assert!(missing.get("error").is_some());
}

#[test]
fn unsupported_mcp_and_image_prompts_do_not_start_a_model_job() {
    let root = tempfile::tempdir().unwrap();
    let (server, rx) = fixture(root.path());
    initialize(&server, &rx);
    let rejected = request(
        &server,
        &rx,
        "mcp",
        "session/new",
        json!({"cwd":root.path(),"mcpServers":[{"name":"unsupported"}]}),
    );
    assert_eq!(rejected["error"]["code"], -32602);
    let id = new_session(&server, &rx, root.path());
    let rejected = request(
        &server,
        &rx,
        "image",
        "session/prompt",
        json!({"sessionId":id,"prompt":[{"type":"image","data":"fixture","mimeType":"image/png"}]}),
    );
    assert_eq!(rejected["error"]["code"], -32602);
    assert!(!lock_state(&server.state).sessions[&id].busy);
}

struct CountTools(Arc<AtomicUsize>);
impl ExecutionEnv for CountTools {
    fn tool_definitions(&self) -> Vec<ToolDefinition> {
        vec![]
    }
    fn call_tool(&mut self, _: &str, _: &Value) -> ToolOutcome {
        self.0.fetch_add(1, Ordering::SeqCst);
        ToolOutcome::ok("fixture executed")
    }
}

fn tool_wait(
    server: &Server,
    id: &str,
    calls: Arc<AtomicUsize>,
) -> (Receiver<ToolOutcome>, thread::JoinHandle<()>) {
    let session = lock_state(&server.state);
    let session = &session.sessions[id];
    let mut env = GatingEnv {
        mode: "ask",
        inner: Box::new(CountTools(calls)),
        session_id: id.to_owned(),
        perms: Arc::clone(&session.perms),
        cancel: Arc::clone(&session.cancel),
        out: server.out.clone(),
        pending: Arc::clone(&server.pending),
        next_request_id: Arc::clone(&server.next_request_id),
    };
    let (tx, rx) = mpsc::channel();
    let job = thread::spawn(move || {
        let result = env.call_tool("bash", &json!({"command":"fixture-only"}));
        tx.send(result).unwrap();
    });
    (rx, job)
}

#[test]
fn permission_rejection_never_executes_and_allow_once_executes_once() {
    let root = tempfile::tempdir().unwrap();
    let (server, rx) = fixture(root.path());
    initialize(&server, &rx);
    let id = new_session(&server, &rx, root.path());
    let calls = Arc::new(AtomicUsize::new(0));
    for (decision, expected) in [("reject-once", 0), ("allow-once", 1)] {
        let (outcome, job) = tool_wait(&server, &id, Arc::clone(&calls));
        let permission = next(&rx);
        assert_eq!(permission["method"], "session/request_permission");
        server.complete_response(&json!({"jsonrpc":"2.0","id":permission["id"],"result":{"outcome":{"outcome":"selected","optionId":decision}}}));
        let result = outcome.recv_timeout(Duration::from_secs(2)).unwrap();
        job.join().unwrap();
        assert_eq!(result.is_error, expected == 0);
        assert_eq!(calls.load(Ordering::SeqCst), expected);
        if expected == 1 {
            assert_eq!(next(&rx)["params"]["update"]["status"], "in_progress");
        }
    }
}

#[test]
fn cancel_removes_only_its_permission_wait_and_ignores_late_approval() {
    let root = tempfile::tempdir().unwrap();
    let (server, rx) = fixture(root.path());
    initialize(&server, &rx);
    let first = new_session(&server, &rx, root.path());
    let second = new_session(&server, &rx, root.path());
    let calls = Arc::new(AtomicUsize::new(0));
    let (first_outcome, first_job) = tool_wait(&server, &first, Arc::clone(&calls));
    let first_permission = next(&rx);
    let (second_outcome, second_job) = tool_wait(&server, &second, Arc::clone(&calls));
    let second_permission = next(&rx);
    server.dispatch_line(&json!({"jsonrpc":"2.0","id":"","method":"session/cancel","params":{"sessionId":first},"headers":[]}).to_string());
    server.complete_response(&json!({"id":first_permission["id"],"result":{"outcome":{"outcome":"selected","optionId":"allow-once"}}}));
    assert!(
        first_outcome
            .recv_timeout(Duration::from_secs(2))
            .unwrap()
            .is_error
    );
    first_job.join().unwrap();
    assert!(server
        .pending
        .lock()
        .unwrap()
        .contains_key(&id_key(&second_permission["id"])));
    assert!(!lock_state(&server.state).sessions[&second]
        .cancel
        .load(Ordering::Relaxed));
    server.shutdown();
    assert!(
        second_outcome
            .recv_timeout(Duration::from_secs(2))
            .unwrap()
            .is_error
    );
    second_job.join().unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(server.pending.lock().unwrap().is_empty());
}

fn prepared_for(server: &Server, id: &str) -> PreparedPrompt {
    let state = lock_state(&server.state);
    let session = &state.sessions[id];
    PreparedPrompt {
        session_id: id.to_owned(),
        cwd: session.cwd.clone(),
        data_root: session.data_root.clone(),
        project: session.project.clone(),
        model: session.model.clone(),
        max_output_tokens: session.max_output_tokens,
        history: session.messages.clone(),
        cancel: Arc::clone(&session.cancel),
        perms: Arc::clone(&session.perms),
        client_tools: Vec::new(),
        images: Vec::new(),
        mcp: Default::default(),
        context_summary: session.context_summary.clone(),
        choices: session.choices.clone(),
        context_window: None,
        extra_dirs: session.extra_dirs.clone(),
        client: Value::Null,
        effort: None,
        compact_at: None,
        context_tokens: None,
    }
}

fn completed_fixture(history: &[Message], label: &str) -> PromptDone {
    let mut messages = history.to_vec();
    for (role, text) in [
        (greppy_agent::Role::User, format!("{label} request")),
        (greppy_agent::Role::Assistant, format!("{label} answer")),
    ] {
        messages.push(Message {
            role,
            content: vec![greppy_agent::ContentPart::Text { text }],
        });
    }
    PromptDone {
        messages,
        context_summary: None,
        usage: greppy_agent::Usage {
            input_tokens: 7,
            output_tokens: 3,
            cache_read_input_tokens: 2,
            cache_creation_input_tokens: 1,
        },
        stop_reason: "end_turn",
    }
}

#[test]
fn failed_post_message_commit_preserves_history_and_retry_does_not_duplicate() {
    let root = tempfile::tempdir().unwrap();
    let (mut server, rx) = fixture(root.path());
    initialize(&server, &rx);
    let id = new_session(&server, &rx, root.path());
    let prepared = prepared_for(&server, &id);
    let store = SessionStore::new(&prepared.data_root, &prepared.project);
    let original = store.load(&id).unwrap();
    let original_bytes = std::fs::read(store.path_for(&id).unwrap()).unwrap();
    let attempts = Arc::new(AtomicUsize::new(0));
    let hook_store = store.clone();
    let hook_id = id.clone();
    let hook_original = original.clone();
    server.config.after_messages = Some(Arc::new(move || {
        if attempts.fetch_add(1, Ordering::SeqCst) == 0 {
            let staged = std::fs::read_dir(hook_store.project_dir())?
                .map(|entry| entry.unwrap().path())
                .find(|path| path.extension().is_some_and(|ext| ext == "pending"))
                .expect("messages physically staged before the injected failure");
            let staged = std::fs::read_to_string(staged)?;
            assert_eq!(
                staged
                    .lines()
                    .filter(|line| {
                        serde_json::from_str::<Value>(line).unwrap()["type"] == "message"
                    })
                    .count(),
                2
            );
            assert_eq!(hook_store.load(&hook_id)?, hook_original);
            return Err(io::Error::other("injected failure after staging messages"));
        }
        Ok(())
    }));
    let done = completed_fixture(&prepared.history, "first");
    assert!(
        finish_prompt(&server.state, &prepared, &server.config, &done)
            .unwrap_err()
            .contains("injected failure after staging messages")
    );
    assert_eq!(store.load(&id).unwrap(), original);
    assert_eq!(
        std::fs::read(store.path_for(&id).unwrap()).unwrap(),
        original_bytes
    );
    {
        let state = lock_state(&server.state);
        assert!(state.sessions[&id].messages.is_empty());
        assert_eq!(state.sessions[&id].usage_in, 0);
        assert_eq!(state.sessions[&id].usage_out, 0);
    }
    assert!(std::fs::read_dir(store.project_dir())
        .unwrap()
        .all(|entry| {
            entry
                .unwrap()
                .path()
                .extension()
                .is_none_or(|ext| ext != "pending")
        }));

    finish_prompt(&server.state, &prepared, &server.config, &done).unwrap();
    let saved = store.load(&id).unwrap();
    assert_eq!(saved.messages, messages_from_protocol(&done.messages));
    assert_eq!(saved.messages.len(), 2);
    assert_eq!(saved.title, "first request");
    assert_eq!(saved.turns, 1);
    assert_eq!(saved.usage, done.usage);
    assert!(finish_prompt(&server.state, &prepared, &server.config, &done).is_err());
    assert_eq!(store.load(&id).unwrap(), saved);

    let continuation = prepared_for(&server, &id);
    assert_eq!(
        messages_from_protocol(&continuation.history),
        saved.messages
    );
    let next = completed_fixture(&continuation.history, "second");
    finish_prompt(&server.state, &continuation, &server.config, &next).unwrap();
    let reopened = store.load(&id).unwrap();
    let state = lock_state(&server.state);
    assert_eq!(
        reopened.messages,
        messages_from_protocol(&state.sessions[&id].messages)
    );
    assert_eq!(reopened.messages.len(), 4);
    assert_eq!(reopened.title, "first request");
    assert_eq!(reopened.turns, 2);
    assert_eq!(reopened.usage.input_tokens, 14);
    assert_eq!(reopened.usage.output_tokens, 6);
    assert_eq!(reopened.usage.cache_read_input_tokens, 4);
    assert_eq!(reopened.usage.cache_creation_input_tokens, 2);
    assert_eq!(state.sessions[&id].usage_in, 14);
    assert_eq!(state.sessions[&id].usage_out, 6);
}

#[test]
fn concurrent_sessions_keep_captured_store_identity() {
    const CHILD: &str = "GREPPY_ACP_ROUTING_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let root = tempfile::tempdir().unwrap();
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "agent_acp::tests::concurrent_sessions_keep_captured_store_identity",
                "--test-threads=1",
                "--nocapture",
            ])
            .env(CHILD, root.path())
            .env("GREPPY_STORE_DIR", root.path().join("captured"))
            .env("GREPPY_PROJECT_IDENTITY", "ambient-sentinel")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "isolated routing test: {}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }

    let root = PathBuf::from(std::env::var_os(CHILD).unwrap());
    let alpha = root.join("alpha");
    let beta = root.join("beta");
    for cwd in [&alpha, &beta] {
        std::fs::create_dir_all(cwd.join(".git")).unwrap();
    }
    let (mut server, rx) = fixture(&root);
    server.config.data_root = None;
    // Reconstruct so production startup captures the isolated child's store root.
    let config = server.config.clone();
    let (tx, rx2) = mpsc::channel();
    server = Server::new(Frames { bytes: vec![], tx }, config);
    drop(rx);
    let rx = rx2;
    initialize(&server, &rx);
    let first = new_session(&server, &rx, &alpha);
    let second = new_session(&server, &rx, &beta);
    let first_prepared = prepared_for(&server, &first);
    let second_prepared = prepared_for(&server, &second);
    assert_eq!(first_prepared.project, "alpha");
    assert_eq!(second_prepared.project, "beta");
    assert_eq!(first_prepared.data_root, root.join("captured"));
    assert_eq!(second_prepared.data_root, root.join("captured"));
    // This test is the only test in its child process; no concurrent environment writers.
    unsafe {
        std::env::set_var("GREPPY_STORE_DIR", root.join("decoy"));
    }

    let (ready_tx, ready_rx) = mpsc::channel();
    let release = Arc::new((Mutex::new(false), std::sync::Condvar::new()));
    let hook_release = Arc::clone(&release);
    server.config.after_messages = Some(Arc::new(move || {
        assert_eq!(
            std::env::var("GREPPY_PROJECT_IDENTITY").unwrap(),
            "ambient-sentinel"
        );
        ready_tx.send(()).unwrap();
        let (lock, cv) = &*hook_release;
        let (released, timeout) = cv
            .wait_timeout_while(lock.lock().unwrap(), Duration::from_secs(5), |released| {
                !*released
            })
            .unwrap();
        if timeout.timed_out() && !*released {
            return Err(io::Error::other(
                "bounded concurrent routing fixture timed out",
            ));
        }
        Ok(())
    }));
    let mut jobs = vec![];
    for (prepared, label) in [(first_prepared, "alpha"), (second_prepared, "beta")] {
        let state = Arc::clone(&server.state);
        let config = server.config.clone();
        jobs.push(thread::spawn(move || {
            let done = completed_fixture(&prepared.history, label);
            finish_prompt(&state, &prepared, &config, &done)
        }));
    }
    for _ in 0..2 {
        ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    }
    // The stdin-side operations overlap both worker commits.
    let third = new_session(&server, &rx, &alpha);
    let list = request(&server, &rx, "list", "session/list", json!({"cwd": alpha}));
    assert!(list.get("result").is_some(), "{list}");
    let changed = request(
        &server,
        &rx,
        "model",
        "session/set_model",
        json!({"sessionId": third, "modelId": "captured-model"}),
    );
    assert!(changed.get("result").is_some(), "{changed}");
    let loaded = request(
        &server,
        &rx,
        "load",
        "session/load",
        json!({"sessionId": third, "cwd": alpha, "mcpServers":[]}),
    );
    assert!(loaded.get("result").is_some(), "{loaded}");
    *release.0.lock().unwrap() = true;
    release.1.notify_all();
    for job in jobs {
        job.join().unwrap().unwrap();
    }

    let state = lock_state(&server.state);
    for (id, project, label) in [(&first, "alpha", "alpha"), (&second, "beta", "beta")] {
        let saved = SessionStore::new(root.join("captured"), project)
            .load(id)
            .unwrap();
        assert_eq!(
            saved.messages,
            messages_from_protocol(&state.sessions[id].messages)
        );
        assert_eq!(saved.messages.len(), 2);
        assert_eq!(saved.title, format!("{label} request"));
        assert_eq!(saved.usage.input_tokens, 7);
        assert!(!SessionStore::new(
            root.join("captured"),
            if project == "alpha" { "beta" } else { "alpha" }
        )
        .path_for(id)
        .unwrap()
        .exists());
    }
    assert_eq!(
        SessionStore::new(root.join("captured"), "alpha")
            .load(&third)
            .unwrap()
            .model,
        "captured-model"
    );
    assert!(!root.join("decoy").exists());
    assert_eq!(
        std::env::var("GREPPY_PROJECT_IDENTITY").unwrap(),
        "ambient-sentinel"
    );
}

#[test]
fn same_session_cross_process_writer_cannot_overwrite_a_successful_turn() {
    const CHILD: &str = "GREPPY_ACP_SAME_SESSION_WRITER_TEST";
    if let Some(root) = std::env::var_os(CHILD) {
        let root = PathBuf::from(root);
        let id = std::env::var("GREPPY_ACP_SAME_SESSION_ID").unwrap();
        let (server, rx) = fixture(&root);
        initialize(&server, &rx);
        let loaded = request(
            &server,
            &rx,
            "load",
            "session/load",
            json!({"sessionId":id, "cwd":root, "mcpServers":[]}),
        );
        assert!(loaded.get("result").is_some(), "{loaded}");
        let prepared = prepared_for(&server, &id);
        assert!(prepared.history.is_empty());
        let done = completed_fixture(&prepared.history, "competing");
        let error = finish_prompt(&server.state, &prepared, &server.config, &done).unwrap_err();
        assert!(error.contains("session is busy"), "{error}");
        let saved = SessionStore::new(&prepared.data_root, &prepared.project)
            .load(&id)
            .unwrap();
        assert!(saved.messages.is_empty());
        assert_eq!(saved.turns, 0);
        assert!(lock_state(&server.state).sessions[&id].messages.is_empty());
        return;
    }

    let root = tempfile::tempdir().unwrap();
    let (mut server, rx) = fixture(root.path());
    initialize(&server, &rx);
    let id = new_session(&server, &rx, root.path());
    let prepared = prepared_for(&server, &id);
    let store = SessionStore::new(&prepared.data_root, &prepared.project);
    let original = store.load(&id).unwrap();
    let hook_store = store.clone();
    let hook_id = id.clone();
    let hook_root = root.path().to_owned();
    server.config.after_messages = Some(Arc::new(move || {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact",
                "agent_acp::tests::same_session_cross_process_writer_cannot_overwrite_a_successful_turn",
                "--test-threads=1", "--nocapture"])
            .env(CHILD, &hook_root)
            .env("GREPPY_ACP_SAME_SESSION_ID", &hook_id)
            .output()?;
        assert!(
            output.status.success(),
            "overlapping process: {}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(hook_store.load(&hook_id)?, original);
        // Model and append-only writers use the same lease, too.
        let error = hook_store.set_model(&hook_id, "racing-model").unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::WouldBlock);
        Ok(())
    }));
    let done = completed_fixture(&prepared.history, "committed");
    finish_prompt(&server.state, &prepared, &server.config, &done).unwrap();
    let saved = store.load(&id).unwrap();
    assert_eq!(saved.messages, messages_from_protocol(&done.messages));
    assert_eq!(saved.messages.len(), 2);
    assert_eq!(saved.turns, 1);
    assert_eq!(saved.title, "committed request");
    assert_eq!(saved.model, "fixture-model");
    assert_eq!(saved.usage, done.usage);

    // A writer that was prepared before the winning commit cannot retry stale history.
    let competing = completed_fixture(&prepared.history, "competing");
    let error = finish_prompt(&server.state, &prepared, &server.config, &competing).unwrap_err();
    assert!(error.contains("saved session history changed"), "{error}");
    assert_eq!(store.load(&id).unwrap(), saved);
    server.config.after_messages = None;
    store.set_model(&id, "after-release").unwrap();
    let continuation = prepared_for(&server, &id);
    let next = completed_fixture(&continuation.history, "continued");
    finish_prompt(&server.state, &continuation, &server.config, &next).unwrap();
    let reopened = store.load(&id).unwrap();
    assert_eq!(reopened.messages, messages_from_protocol(&next.messages));
    assert_eq!(reopened.messages.len(), 4);
    assert_eq!(reopened.turns, 2);
    assert_eq!(reopened.model, "after-release");
}

#[test]
fn each_answered_model_request_is_said_as_a_usage_update_and_a_silent_one_is_not() {
    let (tx, rx) = mpsc::channel();
    let out = Out {
        inner: Arc::new(Mutex::new(Frames {
            bytes: Vec::new(),
            tx,
        })),
    };
    let usage = greppy_agent::Usage {
        input_tokens: 400,
        output_tokens: 90,
        cache_read_input_tokens: 800,
        cache_creation_input_tokens: 0,
    };
    let answered = |usage| LoopEvent::TurnComplete {
        stop_reason: greppy_agent::StopReason::EndTurn,
        usage,
    };
    emit_loop_event(&out, "fixture-session", &answered(usage)).unwrap();
    let frame = next(&rx);
    assert_eq!(frame["method"], "session/update");
    let update = &frame["params"]["update"];
    assert_eq!(update["sessionUpdate"], "usage_update", "{frame}");
    assert_eq!(update["used"], 1290, "{frame}");
    assert_eq!(update["_meta"]["greppy"]["inputTokens"], 400);
    assert_eq!(update["_meta"]["greppy"]["cachedReadTokens"], 800);
    assert_eq!(update["_meta"]["greppy"]["outputTokens"], 90);
    emit_loop_event(
        &out,
        "fixture-session",
        &answered(greppy_agent::Usage::default()),
    )
    .unwrap();
    assert!(rx.recv_timeout(Duration::from_millis(100)).is_err());
}

#[test]
fn header_flags_are_name_value_pairs_and_a_broken_one_is_refused() {
    assert_eq!(
        header_pair("x-workjet-provider: made-up").unwrap(),
        ("x-workjet-provider".to_owned(), "made-up".to_owned())
    );
    for said in ["no-colon", ": value", "x-a:", "x a: v"] {
        assert!(header_pair(said).is_err(), "{said}");
    }
}

/// greppy's own tool, under its own name, counting its calls.
struct OwnTool(Arc<AtomicUsize>);
impl ExecutionEnv for OwnTool {
    fn tool_definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "greppy".into(),
            description: "fixture".into(),
            input_schema: json!({"type": "object"}),
        }]
    }
    fn call_tool(&mut self, _: &str, _: &Value) -> ToolOutcome {
        self.0.fetch_add(1, Ordering::SeqCst);
        ToolOutcome::ok("fixture executed")
    }
}

fn notes_tool() -> Value {
    json!({"name": "look_up_notes", "description": "Look up the made-up notes.",
        "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}}})
}

#[test]
fn client_tools_are_checked_named_plainly_once_and_never_as_greppys_own() {
    let offered =
        |tools: Value| client_tools(&json!({"_meta": {"greppy": {"clientTools": tools}}}));
    let tools = offered(json!([notes_tool()])).unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "look_up_notes");
    assert_eq!(
        tools[0].input_schema["properties"]["query"]["type"],
        "string"
    );
    assert!(client_tools(&json!({})).unwrap().is_empty());
    for broken in [
        json!([notes_tool(), notes_tool()]),
        json!([{"name": "greppy", "description": "poses as greppy"}]),
        json!([{"name": "has space", "description": "x"}]),
        json!([{"name": "look_up_notes"}]),
        json!({"name": "not-a-list"}),
    ] {
        assert!(offered(broken.clone()).is_err(), "{broken}");
    }
}

#[test]
fn a_client_tool_is_run_by_the_client_without_permission_and_its_answer_is_the_result() {
    let root = tempfile::tempdir().unwrap();
    let (server, rx) = fixture(root.path());
    initialize(&server, &rx);
    let id = new_session(&server, &rx, root.path());
    let calls = Arc::new(AtomicUsize::new(0));
    let env = {
        let state = lock_state(&server.state);
        let session = &state.sessions[&id];
        ClientToolEnv {
            inner: Box::new(GatingEnv {
                mode: "ask",
                inner: Box::new(OwnTool(Arc::clone(&calls))),
                session_id: id.clone(),
                perms: Arc::clone(&session.perms),
                cancel: Arc::clone(&session.cancel),
                out: server.out.clone(),
                pending: Arc::clone(&server.pending),
                next_request_id: Arc::clone(&server.next_request_id),
            }),
            tools: client_tools(&json!({"_meta": {"greppy": {"clientTools": [notes_tool()]}}}))
                .unwrap(),
            session_id: id.clone(),
            cancel: Arc::clone(&session.cancel),
            out: server.out.clone(),
            pending: Arc::clone(&server.pending),
            next_request_id: Arc::clone(&server.next_request_id),
        }
    };
    let names: Vec<String> = env
        .tool_definitions()
        .into_iter()
        .map(|tool| tool.name)
        .collect();
    assert_eq!(names, ["greppy", "look_up_notes"]);
    let env = Arc::new(Mutex::new(env));
    let call = |name: &'static str, input: Value| {
        let env = Arc::clone(&env);
        let (tx, outcome) = mpsc::channel();
        thread::spawn(move || {
            let result = env.lock().unwrap().call_tool(name, &input);
            tx.send(result).unwrap();
        });
        outcome
    };

    // The client runs it: one request, no permission asked, its answer is the result.
    let outcome = call("look_up_notes", json!({"query": "release"}));
    let asked = next(&rx);
    assert_eq!(asked["method"], CLIENT_TOOL_METHOD, "{asked}");
    assert_eq!(asked["params"]["name"], "look_up_notes");
    assert_eq!(asked["params"]["input"], json!({"query": "release"}));
    assert_eq!(asked["params"]["sessionId"], json!(id));
    server.complete_response(&json!({"jsonrpc": "2.0", "id": asked["id"],
        "result": {"content": "The release is on a made-up day."}}));
    let result = outcome.recv_timeout(Duration::from_secs(2)).unwrap();
    assert_eq!(result, ToolOutcome::ok("The release is on a made-up day."));

    // What the client refuses is the tool's error, with the reason.
    let outcome = call("look_up_notes", json!({"query": "more"}));
    let asked = next(&rx);
    server.complete_response(&json!({"jsonrpc": "2.0", "id": asked["id"],
        "error": {"code": -32000, "message": "the notes module is not docked any more"}}));
    let result = outcome.recv_timeout(Duration::from_secs(2)).unwrap();
    assert_eq!(
        result,
        ToolOutcome::err("the notes module is not docked any more")
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    // greppy's own tool still goes through its permission gate, not to the client.
    let outcome = call("greppy", json!({"args": ["where-am-i"]}));
    let asked = next(&rx);
    assert_eq!(asked["method"], "session/request_permission", "{asked}");
    server.complete_response(&json!({"jsonrpc": "2.0", "id": asked["id"],
        "result": {"outcome": {"outcome": "selected", "optionId": "allow-once"}}}));
    assert_eq!(
        outcome.recv_timeout(Duration::from_secs(2)).unwrap(),
        ToolOutcome::ok("fixture executed")
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn a_prompts_pictures_go_after_its_text_and_a_bad_one_is_refused() {
    const PIXEL: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
    let blocks = [
        json!({"type": "text", "text": "Look at this."}),
        json!({"type": "image", "mimeType": "image/png", "data": PIXEL}),
    ];
    assert_eq!(prompt_text(&blocks).unwrap(), "Look at this.");
    assert_eq!(
        prompt_images(&blocks).unwrap(),
        [greppy_agent::ContentPart::Image {
            media_type: "image/png".into(),
            data: PIXEL.into()
        }]
    );
    // Pictures alone are a prompt too (the image capability is on); its text is empty.
    assert_eq!(prompt_text(&blocks[1..]).unwrap(), "");
    for broken in [
        json!({"type": "image", "mimeType": "image/svg+xml", "data": PIXEL}),
        json!({"type": "image", "mimeType": "image/png", "data": "not base64!"}),
        json!({"type": "image", "mimeType": "image/png", "data": ""}),
    ] {
        assert!(
            prompt_images(&[blocks[0].clone(), broken.clone()]).is_err(),
            "{broken}"
        );
    }
}

#[test]
fn linked_and_embedded_resources_reach_the_model_as_words_or_pictures() {
    const PIXEL: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
    let blocks = [
        json!({"type": "text", "text": "Look at these."}),
        json!({"type": "resource_link", "uri": "file:///work/made%20up/notes.md", "name": "notes.md",
            "mimeType": "text/markdown", "description": "The made-up notes"}),
        json!({"type": "resource_link", "uri": "https://example.invalid/spec", "name": "spec"}),
        json!({"type": "resource", "resource": {"uri": "file:///work/a.rs", "mimeType": "text/x-rust",
            "text": "fn made_up() {}"}}),
        json!({"type": "resource", "resource": {"uri": "file:///work/blob.bin",
            "mimeType": "application/octet-stream", "blob": "AAAA"}}),
        json!({"type": "resource", "resource": {"uri": "file:///work/pixel.png",
            "mimeType": "image/png", "blob": PIXEL}}),
    ];
    let text = prompt_text(&blocks).unwrap();
    assert!(text.starts_with("Look at these.\n"), "{text}");
    assert!(text.contains("[Linked resource: notes.md <file:///work/made%20up/notes.md>, a local file: /work/made up/notes.md, text/markdown — The made-up notes]"), "{text}");
    assert!(
        text.contains("[Linked resource: spec <https://example.invalid/spec>]"),
        "{text}"
    );
    assert!(text.contains("<resource uri=\"file:///work/a.rs\" mimeType=\"text/x-rust\">\nfn made_up() {}\n</resource>"), "{text}");
    assert!(text.contains("[Embedded binary resource file:///work/blob.bin (application/octet-stream, 4 bytes as base64), not shown]"), "{text}");
    assert!(!text.contains(PIXEL));
    // The embedded picture goes to the model as a picture.
    assert_eq!(
        prompt_images(&blocks).unwrap(),
        [greppy_agent::ContentPart::Image {
            media_type: "image/png".into(),
            data: PIXEL.into()
        }]
    );
    // A link needs its uri; audio is not taken.
    assert!(prompt_text(&[json!({"type": "resource_link", "name": "x"})]).is_err());
    assert!(
        prompt_text(&[json!({"type": "audio", "mimeType": "audio/wav", "data": "AAAA"})]).is_err()
    );
}

#[test]
fn a_compacted_turn_keeps_its_summary_for_the_next_prompt_and_after_a_restart() {
    let root = tempfile::tempdir().unwrap();
    let (server, rx) = fixture(root.path());
    initialize(&server, &rx);
    let id = new_session(&server, &rx, root.path());
    // A first turn, then one that compacted the conversation to a window and a summary.
    let prepared = prepared_for(&server, &id);
    let first = completed_fixture(&prepared.history, "first");
    finish_prompt(&server.state, &prepared, &server.config, &first).unwrap();
    let prepared = prepared_for(&server, &id);
    let mut compacted = completed_fixture(&[], "after compaction");
    compacted.context_summary = Some("made-up summary of the first turn".into());
    finish_prompt(&server.state, &prepared, &server.config, &compacted).unwrap();
    // The next prompt starts from the window and the summary.
    let next = prepared_for(&server, &id);
    assert_eq!(next.history, compacted.messages);
    assert_eq!(
        next.context_summary.as_deref(),
        Some("made-up summary of the first turn")
    );
    // So does a new server that loads the session.
    let (restarted, restarted_rx) = fixture(root.path());
    initialize(&restarted, &restarted_rx);
    let loaded = restarted.session_load(
        &json!("reload"),
        &json!({"sessionId": id, "cwd": root.path(), "mcpServers": []}),
        "session/load",
    );
    assert!(loaded.get("error").is_none(), "{loaded}");
    let reloaded = prepared_for(&restarted, &id);
    assert_eq!(reloaded.history, compacted.messages);
    assert_eq!(
        reloaded.context_summary.as_deref(),
        Some("made-up summary of the first turn")
    );
}

fn test_choices() -> Choices {
    Choices {
        mode: "ask",
        effort: None,
        capabilities: greppy_agent::CapabilitySet::default(),
    }
}

#[test]
fn a_session_offers_its_config_options_and_modes_and_takes_the_clients_choices() {
    let root = tempfile::tempdir().unwrap();
    let (server, rx) = fixture(root.path());
    initialize(&server, &rx);
    let began = request(
        &server,
        &rx,
        "new",
        "session/new",
        json!({"cwd": root.path(), "mcpServers": []}),
    );
    let id = began["result"]["sessionId"].as_str().unwrap().to_owned();
    let options: Vec<&str> = began["result"]["configOptions"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|option| option["id"].as_str())
        .collect();
    assert_eq!(options, ["model", "mode", "effort", "web", "desktop"]);
    assert_eq!(began["result"]["modes"]["currentModeId"], "ask");
    assert_eq!(
        began["result"]["modes"]["availableModes"]
            .as_array()
            .unwrap()
            .len(),
        3
    );

    let set = |config: &str, value: &str| {
        request(
            &server,
            &rx,
            "set",
            "session/set_config_option",
            json!({"sessionId": id, "configId": config, "value": value}),
        )
    };
    let answer = set("effort", "high");
    assert!(answer.get("error").is_none(), "{answer}");
    let effort = answer["result"]["configOptions"]
        .as_array()
        .unwrap()
        .iter()
        .find(|option| option["id"] == "effort")
        .unwrap()
        .clone();
    assert_eq!(effort["currentValue"], "high");
    assert!(set("web", "on").get("error").is_none());
    assert_eq!(set("web", "maybe")["error"]["code"], -32602);
    assert_eq!(set("made-up", "x")["error"]["code"], -32602);
    let answer = request(
        &server,
        &rx,
        "mode",
        "session/set_mode",
        json!({"sessionId": id, "modeId": "read-only"}),
    );
    assert!(answer.get("error").is_none(), "{answer}");
    let choices = prepared_for(&server, &id).choices;
    assert_eq!(
        (
            choices.mode,
            choices.effort.as_deref(),
            choices.capabilities.web
        ),
        ("read-only", Some("high"), true)
    );
}

#[test]
fn a_read_only_session_refuses_commands_and_an_auto_one_asks_nobody() {
    let root = tempfile::tempdir().unwrap();
    let (server, rx) = fixture(root.path());
    initialize(&server, &rx);
    let id = new_session(&server, &rx, root.path());
    let calls = Arc::new(AtomicUsize::new(0));
    let gate = |mode: &'static str| {
        let state = lock_state(&server.state);
        let session = &state.sessions[&id];
        GatingEnv {
            mode,
            inner: Box::new(OwnTool(Arc::clone(&calls))),
            session_id: id.clone(),
            perms: Arc::clone(&session.perms),
            cancel: Arc::clone(&session.cancel),
            out: server.out.clone(),
            pending: Arc::clone(&server.pending),
            next_request_id: Arc::clone(&server.next_request_id),
        }
    };
    let command = json!({"args": ["bash-smart", "--", "true"]});
    let refused = gate("read-only").call_tool("greppy", &command);
    assert!(
        refused.is_error && refused.content.contains("read only"),
        "{refused:?}"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    // Auto: run without a permission request.
    assert_eq!(
        gate("auto").call_tool("greppy", &command),
        ToolOutcome::ok("fixture executed")
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(rx
        .try_iter()
        .all(|frame| frame["method"] != "session/request_permission"));
}

#[test]
fn the_plan_tool_says_the_plan_to_the_client() {
    let (tx, rx) = mpsc::channel();
    let out = Out {
        inner: Arc::new(Mutex::new(Frames {
            bytes: Vec::new(),
            tx,
        })),
    };
    let calls = Arc::new(AtomicUsize::new(0));
    let mut env = PlanEnv {
        inner: Box::new(OwnTool(Arc::clone(&calls))),
        out,
        session_id: "fixture-session".into(),
    };
    assert!(env
        .tool_definitions()
        .iter()
        .any(|tool| tool.name == PLAN_TOOL));
    let kept = env.call_tool(
        PLAN_TOOL,
        &json!({"plan": [{"step": "Count the made-up buoys", "status": "in_progress"},
            {"step": "Say what was found", "status": "pending"}]}),
    );
    assert_eq!(kept, ToolOutcome::ok("The plan is kept."));
    let frame = next(&rx);
    assert_eq!(frame["params"]["update"]["sessionUpdate"], "plan");
    assert_eq!(
        frame["params"]["update"]["entries"],
        json!([{"content": "Count the made-up buoys", "priority": "medium", "status": "in_progress"},
            {"content": "Say what was found", "priority": "medium", "status": "pending"}])
    );
    assert!(
        env.call_tool(
            PLAN_TOOL,
            &json!({"plan": [{"step": "x", "status": "done"}]})
        )
        .is_error
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[test]
fn a_new_session_is_told_its_slash_commands_and_can_be_deleted() {
    let root = tempfile::tempdir().unwrap();
    let (server, rx) = fixture(root.path());
    initialize(&server, &rx);
    server.dispatch_line(
        &json!({"jsonrpc": "2.0", "id": "new", "method": "session/new",
            "params": {"cwd": root.path(), "mcpServers": []}})
        .to_string(),
    );
    let began = next(&rx);
    let id = began["result"]["sessionId"].as_str().unwrap().to_owned();
    let commands = next(&rx);
    assert_eq!(
        commands["params"]["update"]["sessionUpdate"], "available_commands_update",
        "{commands}"
    );
    let names: Vec<&str> = commands["params"]["update"]["availableCommands"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|command| command["name"].as_str())
        .collect();
    assert_eq!(names, ["compact", "undo"]);

    let prepared = prepared_for(&server, &id);
    let store = SessionStore::new(&prepared.data_root, &prepared.project);
    let path = store.path_for(&id).unwrap();
    assert!(path.is_file());
    let deleted = request(
        &server,
        &rx,
        "delete",
        "session/delete",
        json!({"sessionId": id}),
    );
    assert!(deleted.get("error").is_none(), "{deleted}");
    assert!(!path.exists() && !lock_state(&server.state).sessions.contains_key(&id));
    let again = request(
        &server,
        &rx,
        "again",
        "session/delete",
        json!({"sessionId": id}),
    );
    assert_eq!(again["error"]["code"], -32002, "{again}");
}

#[test]
fn a_cancelled_request_cancels_its_prompt_and_further_folders_are_taken() {
    let root = tempfile::tempdir().unwrap();
    let (server, rx) = fixture(root.path());
    initialize(&server, &rx);
    let id = new_session(&server, &rx, root.path());
    lock_state(&server.state)
        .prompts
        .insert(id_key(&json!(41)), id.clone());
    server.dispatch_line(
        &json!({"jsonrpc": "2.0", "method": "$/cancel_request", "params": {"requestId": 41}})
            .to_string(),
    );
    assert!(lock_state(&server.state).sessions[&id]
        .cancel
        .load(Ordering::SeqCst));

    let extra = tempfile::tempdir().unwrap();
    assert_eq!(
        additional_directories(&json!({"additionalDirectories": [extra.path()]})).unwrap(),
        [extra.path().to_path_buf()]
    );
    for broken in [
        json!(["relative/dir"]),
        json!(["/nonexistent/made-up"]),
        json!("x"),
    ] {
        assert!(additional_directories(&json!({"additionalDirectories": broken})).is_err());
    }
}

#[test]
fn the_question_tool_asks_the_person_through_the_client_and_hears_the_answer() {
    let root = tempfile::tempdir().unwrap();
    let (server, rx) = fixture(root.path());
    initialize(&server, &rx);
    let id = new_session(&server, &rx, root.path());
    let calls = Arc::new(AtomicUsize::new(0));
    let env = {
        let state = lock_state(&server.state);
        let session = &state.sessions[&id];
        AskEnv {
            inner: Box::new(OwnTool(Arc::clone(&calls))),
            session_id: id.clone(),
            cancel: Arc::clone(&session.cancel),
            out: server.out.clone(),
            pending: Arc::clone(&server.pending),
            next_request_id: Arc::clone(&server.next_request_id),
        }
    };
    assert!(env
        .tool_definitions()
        .iter()
        .any(|tool| tool.name == ASK_TOOL));
    let env = Arc::new(Mutex::new(env));
    let ask = || {
        let env = Arc::clone(&env);
        let (tx, outcome) = mpsc::channel();
        thread::spawn(move || {
            let result = env.lock().unwrap().call_tool(
                ASK_TOOL,
                &json!({"questions": [{"id": "colour", "header": "Colour", "question": "Which colour?",
                    "options": [{"label": "Red"}, {"label": "Blue"}]}]}),
            );
            tx.send(result).unwrap();
        });
        outcome
    };
    let outcome = ask();
    let asked = next(&rx);
    assert_eq!(asked["method"], "elicitation/create", "{asked}");
    assert_eq!(
        asked["params"]["requestedSchema"]["properties"]["colour"]["enum"],
        json!(["Red", "Blue"])
    );
    server.complete_response(&json!({"jsonrpc": "2.0", "id": asked["id"],
        "result": {"action": "accept", "content": {"colour": "Blue"}}}));
    assert_eq!(
        outcome.recv_timeout(Duration::from_secs(2)).unwrap(),
        ToolOutcome::ok("The person answered:\ncolour: Blue")
    );
    let outcome = ask();
    let asked = next(&rx);
    server.complete_response(
        &json!({"jsonrpc": "2.0", "id": asked["id"], "result": {"action": "decline"}}),
    );
    assert_eq!(
        outcome.recv_timeout(Duration::from_secs(2)).unwrap(),
        ToolOutcome::ok("The person declined to answer.")
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}
