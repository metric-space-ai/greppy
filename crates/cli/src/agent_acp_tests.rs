//! ACP boundary regressions without a model, real tools, or user data.
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
    server.cancel_session(&second);
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
