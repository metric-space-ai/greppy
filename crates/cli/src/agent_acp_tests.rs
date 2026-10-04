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
            history: vec![],
            cancel: Arc::clone(&session.cancel),
            perms: Arc::clone(&session.perms),
        }
    };
    let data_root = server.config.data_root.as_ref().unwrap();
    std::fs::remove_dir_all(data_root).unwrap();
    std::fs::write(data_root, b"fixture blocks persistence").unwrap();
    let done = PromptDone {
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
        max_turns: 2,
        data_root: Some(root.join("state")),
        after_messages: None,
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
    assert_eq!(response["id"], id);
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
    assert_eq!(
        init["result"]["agentCapabilities"]["promptCapabilities"]["image"],
        false
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
        history: session.messages.clone(),
        cancel: Arc::clone(&session.cancel),
        perms: Arc::clone(&session.perms),
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
            assert_eq!(staged.lines().filter(|line| {
                serde_json::from_str::<Value>(line).unwrap()["type"] == "message"
            }).count(), 2);
            assert_eq!(hook_store.load(&hook_id)?, hook_original);
            return Err(io::Error::other("injected failure after staging messages"));
        }
        Ok(())
    }));
    let done = completed_fixture(&prepared.history, "first");
    assert!(finish_prompt(&server.state, &prepared, &server.config, &done)
        .unwrap_err().contains("injected failure after staging messages"));
    assert_eq!(store.load(&id).unwrap(), original);
    assert_eq!(std::fs::read(store.path_for(&id).unwrap()).unwrap(), original_bytes);
    {
        let state = lock_state(&server.state);
        assert!(state.sessions[&id].messages.is_empty());
        assert_eq!(state.sessions[&id].usage_in, 0);
        assert_eq!(state.sessions[&id].usage_out, 0);
    }
    assert!(std::fs::read_dir(store.project_dir()).unwrap().all(|entry| {
        entry.unwrap().path().extension().is_none_or(|ext| ext != "pending")
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
    assert_eq!(messages_from_protocol(&continuation.history), saved.messages);
    let next = completed_fixture(&continuation.history, "second");
    finish_prompt(&server.state, &continuation, &server.config, &next).unwrap();
    let reopened = store.load(&id).unwrap();
    let state = lock_state(&server.state);
    assert_eq!(reopened.messages, messages_from_protocol(&state.sessions[&id].messages));
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
            .args(["--exact", "agent_acp::tests::concurrent_sessions_keep_captured_store_identity",
                "--test-threads=1", "--nocapture"])
            .env(CHILD, root.path())
            .env("GREPPY_STORE_DIR", root.path().join("captured"))
            .env("GREPPY_PROJECT_IDENTITY", "ambient-sentinel")
            .output().unwrap();
        assert!(output.status.success(), "isolated routing test: {}{}",
            String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
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
    unsafe { std::env::set_var("GREPPY_STORE_DIR", root.join("decoy")); }

    let (ready_tx, ready_rx) = mpsc::channel();
    let release = Arc::new((Mutex::new(false), std::sync::Condvar::new()));
    let hook_release = Arc::clone(&release);
    server.config.after_messages = Some(Arc::new(move || {
        assert_eq!(std::env::var("GREPPY_PROJECT_IDENTITY").unwrap(), "ambient-sentinel");
        ready_tx.send(()).unwrap();
        let (lock, cv) = &*hook_release;
        let (released, timeout) = cv.wait_timeout_while(
            lock.lock().unwrap(), Duration::from_secs(5), |released| !*released
        ).unwrap();
        if timeout.timed_out() && !*released {
            return Err(io::Error::other("bounded concurrent routing fixture timed out"));
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
    let changed = request(&server, &rx, "model", "session/set_model",
        json!({"sessionId": third, "modelId": "captured-model"}));
    assert!(changed.get("result").is_some(), "{changed}");
    let loaded = request(&server, &rx, "load", "session/load",
        json!({"sessionId": third, "cwd": alpha, "mcpServers":[]}));
    assert!(loaded.get("result").is_some(), "{loaded}");
    *release.0.lock().unwrap() = true;
    release.1.notify_all();
    for job in jobs { job.join().unwrap().unwrap(); }

    let state = lock_state(&server.state);
    for (id, project, label) in [(&first, "alpha", "alpha"), (&second, "beta", "beta")] {
        let saved = SessionStore::new(root.join("captured"), project).load(id).unwrap();
        assert_eq!(saved.messages, messages_from_protocol(&state.sessions[id].messages));
        assert_eq!(saved.messages.len(), 2);
        assert_eq!(saved.title, format!("{label} request"));
        assert_eq!(saved.usage.input_tokens, 7);
        assert!(!SessionStore::new(root.join("captured"),
            if project == "alpha" { "beta" } else { "alpha" }).path_for(id).unwrap().exists());
    }
    assert_eq!(SessionStore::new(root.join("captured"), "alpha").load(&third).unwrap().model,
        "captured-model");
    assert!(!root.join("decoy").exists());
    assert_eq!(std::env::var("GREPPY_PROJECT_IDENTITY").unwrap(), "ambient-sentinel");
}

#[test]
fn same_session_cross_process_writer_cannot_overwrite_a_successful_turn() {
    const CHILD: &str = "GREPPY_ACP_SAME_SESSION_WRITER_TEST";
    if let Some(root) = std::env::var_os(CHILD) {
        let root = PathBuf::from(root);
        let id = std::env::var("GREPPY_ACP_SAME_SESSION_ID").unwrap();
        let (server, rx) = fixture(&root);
        initialize(&server, &rx);
        let loaded = request(&server, &rx, "load", "session/load",
            json!({"sessionId":id, "cwd":root, "mcpServers":[]}));
        assert!(loaded.get("result").is_some(), "{loaded}");
        let prepared = prepared_for(&server, &id);
        assert!(prepared.history.is_empty());
        let done = completed_fixture(&prepared.history, "competing");
        let error = finish_prompt(&server.state, &prepared, &server.config, &done).unwrap_err();
        assert!(error.contains("session is busy"), "{error}");
        let saved = SessionStore::new(&prepared.data_root, &prepared.project).load(&id).unwrap();
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
        assert!(output.status.success(), "overlapping process: {}{}",
            String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
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
