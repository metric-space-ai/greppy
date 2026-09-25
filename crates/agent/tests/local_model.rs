use std::cell::Cell;
use std::fs;
use std::io;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};

use greppy_agent::local_model::{ArtifactCache, ArtifactSpec, OnDemandModel};
use greppy_agent::{
    run_agent_loop, AgentConfig, ClientError, ContentPart, ExecutionEnv, LoopStop, Message,
    ModelRequest, ModelStream, Role, StopReason, StreamEvent, ToolChoice, ToolDefinition,
    ToolOutcome, TurnResult, Usage,
};
use sha2::{Digest, Sha256};

const BYTES: &[u8] = b"small stand-in for a release artifact";

fn spec() -> ArtifactSpec {
    ArtifactSpec {
        sha256: Sha256::digest(BYTES).into(),
        size_bytes: BYTES.len() as u64,
    }
}

fn cache(root: &std::path::Path) -> ArtifactCache {
    ArtifactCache::new(root.join("persistent-models")).unwrap()
}

#[test]
fn construction_lookup_and_drop_do_not_create_a_cache_or_fetch() {
    let dir = tempfile::tempdir().unwrap();
    let cache = cache(dir.path());
    assert_eq!(cache.lookup_verified(&spec()).unwrap(), None);
    {
        let model =
            OnDemandModel::<ScriptedModel, _>::new(|| -> Result<ScriptedModel, ClientError> {
                panic!("optional feature was not used")
            });
        assert!(!model.is_initialized());
        assert!(format!("{model:?}").contains("false"));
    }
    assert!(!dir.path().join("persistent-models").exists());
}

#[test]
fn relative_cache_root_is_rejected() {
    assert_eq!(
        ArtifactCache::new("models".into()).unwrap_err().kind(),
        io::ErrorKind::InvalidInput
    );
}

#[test]
fn verified_bytes_are_reused_by_a_new_cache_instance_without_fetching() {
    let dir = tempfile::tempdir().unwrap();
    let cache = cache(dir.path());
    let mut progress = Vec::new();
    let path = cache
        .ensure_with(
            &spec(),
            |out| out.write_all(BYTES),
            |done, total| {
                progress.push((done, total));
            },
        )
        .unwrap();
    assert_eq!(fs::read(&path).unwrap(), BYTES);
    assert_eq!(progress.first(), Some(&(0, BYTES.len() as u64)));
    assert_eq!(
        progress.last(),
        Some(&(BYTES.len() as u64, BYTES.len() as u64))
    );
    let next = ArtifactCache::new(dir.path().join("persistent-models")).unwrap();
    assert_eq!(
        next.ensure_with(
            &spec(),
            |_| panic!("cache hit downloaded again"),
            |_, _| { panic!("cache hit reported a download") }
        )
        .unwrap(),
        path
    );
}

#[test]
fn partial_wrong_digest_and_oversize_downloads_are_never_published() {
    for payload in [
        b"short".as_slice(),
        &vec![b'x'; BYTES.len()],
        &vec![b'x'; BYTES.len() + 1],
    ] {
        let dir = tempfile::tempdir().unwrap();
        let cache = cache(dir.path());
        let result = cache.ensure_with(&spec(), |out| out.write_all(payload), |_, _| {});
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidData);
        assert_eq!(cache.lookup_verified(&spec()).unwrap(), None);
        assert_eq!(
            fs::read_dir(dir.path().join("persistent-models/objects"))
                .unwrap()
                .count(),
            0
        );
    }
}

#[test]
fn interrupted_download_can_retry_and_does_not_publish_progress_as_readiness() {
    let dir = tempfile::tempdir().unwrap();
    let cache = cache(dir.path());
    let error = cache
        .ensure_with(
            &spec(),
            |out| {
                out.write_all(BYTES)?;
                Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "cancelled after last byte",
                ))
            },
            |_, _| {},
        )
        .unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::Interrupted);
    assert!(cache.lookup_verified(&spec()).unwrap().is_none());
    cache
        .ensure_with(&spec(), |out| out.write_all(BYTES), |_, _| {})
        .unwrap();
    assert!(cache.lookup_verified(&spec()).unwrap().is_some());
}

#[test]
fn corrupt_cached_bytes_are_replaced_only_by_a_verified_download() {
    let dir = tempfile::tempdir().unwrap();
    let cache = cache(dir.path());
    let path = cache
        .ensure_with(&spec(), |out| out.write_all(BYTES), |_, _| {})
        .unwrap();
    fs::write(&path, vec![b'x'; BYTES.len()]).unwrap();
    assert!(cache.lookup_verified(&spec()).unwrap().is_none());
    assert!(cache
        .ensure_with(&spec(), |_| Err(io::Error::other("offline")), |_, _| {})
        .is_err());
    assert_eq!(fs::read(&path).unwrap(), vec![b'x'; BYTES.len()]);
    cache
        .ensure_with(&spec(), |out| out.write_all(BYTES), |_, _| {})
        .unwrap();
    assert_eq!(fs::read(path).unwrap(), BYTES);
}

#[test]
fn simultaneous_clients_share_one_download() {
    let dir = tempfile::tempdir().unwrap();
    let cache = cache(dir.path());
    let start = Arc::new(Barrier::new(4));
    let downloads = Arc::new(AtomicUsize::new(0));
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let start = Arc::clone(&start);
            let downloads = Arc::clone(&downloads);
            let cache = cache.clone();
            scope.spawn(move || {
                start.wait();
                cache
                    .ensure_with(
                        &spec(),
                        |out| {
                            downloads.fetch_add(1, Ordering::SeqCst);
                            out.write_all(BYTES)
                        },
                        |_, _| {},
                    )
                    .unwrap();
            });
        }
    });
    assert_eq!(downloads.load(Ordering::SeqCst), 1);
}

struct ScriptedModel {
    turn: usize,
}

impl ModelStream for ScriptedModel {
    fn stream_turn(
        &mut self,
        req: &ModelRequest,
        emit: &mut dyn FnMut(StreamEvent),
    ) -> Result<TurnResult, ClientError> {
        assert_eq!(req.model, "local-fixture");
        emit(StreamEvent::Started {
            model: req.model.clone(),
        });
        let (content, stop_reason) = if self.turn == 0 {
            assert_eq!(req.tools[0].name, "greppy");
            emit(StreamEvent::ToolCallStarted {
                index: 0,
                id: "call-1".into(),
                name: "greppy".into(),
            });
            emit(StreamEvent::ToolCallArgumentsDelta {
                index: 0,
                json_fragment: r#"{"command":"search-symbol main"}"#.into(),
            });
            (
                vec![ContentPart::ToolCall {
                    id: "call-1".into(),
                    name: "greppy".into(),
                    arguments: serde_json::json!({"command":"search-symbol main"}),
                }],
                StopReason::ToolUse,
            )
        } else {
            assert!(matches!(&req.messages.last().unwrap().content[0],
                ContentPart::ToolResult { call_id, content, is_error: false } if call_id == "call-1" && content == "src/main.rs:1 main"));
            emit(StreamEvent::TextDelta {
                text: "Found main.".into(),
            });
            (
                vec![ContentPart::Text {
                    text: "Found main.".into(),
                }],
                StopReason::EndTurn,
            )
        };
        self.turn += 1;
        emit(StreamEvent::BlockFinished { index: 0 });
        let usage = Usage {
            input_tokens: 10,
            output_tokens: 3,
            ..Usage::default()
        };
        emit(StreamEvent::Finished {
            stop_reason: stop_reason.clone(),
            usage,
        });
        Ok(TurnResult {
            message: Message {
                role: Role::Assistant,
                content,
            },
            stop_reason,
            usage,
        })
    }
}

struct TestEnv {
    calls: usize,
}

impl ExecutionEnv for TestEnv {
    fn tool_definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "greppy".into(),
            description: "Read code".into(),
            input_schema: serde_json::json!({"type":"object"}),
        }]
    }

    fn call_tool(&mut self, name: &str, arguments: &serde_json::Value) -> ToolOutcome {
        assert_eq!(name, "greppy");
        assert_eq!(arguments["command"], "search-symbol main");
        self.calls += 1;
        ToolOutcome::ok("src/main.rs:1 main")
    }
}

#[test]
fn first_agent_use_provisions_once_and_preserves_the_tool_result_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let cache = cache(dir.path());
    let starts = Cell::new(0);
    let mut model = OnDemandModel::new(|| {
        starts.set(starts.get() + 1);
        cache
            .ensure_with(&spec(), |out| out.write_all(BYTES), |_, _| {})
            .map_err(|error| ClientError::Stream(error.to_string()))?;
        Ok(ScriptedModel { turn: 0 })
    });
    let mut env = TestEnv { calls: 0 };
    let mut events = Vec::new();
    assert_eq!(starts.get(), 0);
    let result = run_agent_loop(
        &mut model,
        &mut env,
        &AgentConfig::default().with_model("local-fixture"),
        "Find main",
        &mut |event| events.push(event),
    )
    .unwrap();
    assert_eq!(starts.get(), 1);
    assert_eq!(env.calls, 1);
    assert_eq!(result.stop, LoopStop::EndTurn);
    assert_eq!(result.final_text, "Found main.");
    assert_eq!(result.usage.output_tokens, 6);
    assert!(model.is_initialized());
    assert!(events.iter().any(|event| matches!(event, greppy_agent::LoopEvent::Stream(StreamEvent::TextDelta { text }) if text == "Found main.")));
}

fn request() -> ModelRequest {
    ModelRequest {
        model: "local-fixture".into(),
        system: None,
        messages: Vec::new(),
        tools: Vec::new(),
        tool_choice: ToolChoice::Auto,
        max_tokens: 32,
    }
}

#[test]
fn failed_initialization_is_not_cached_or_silently_replaced_by_a_gateway() {
    let attempts = Cell::new(0);
    let mut model = OnDemandModel::<ScriptedModel, _>::new(|| {
        attempts.set(attempts.get() + 1);
        Err(ClientError::Stream("release not admitted".into()))
    });
    for _ in 0..2 {
        assert_eq!(
            model.stream_turn(&request(), &mut |_| panic!(
                "failed load emitted model output"
            )),
            Err(ClientError::Stream("release not admitted".into()))
        );
        assert!(!model.is_initialized());
    }
    assert_eq!(attempts.get(), 2);
}

struct FailingModel;

impl ModelStream for FailingModel {
    fn stream_turn(
        &mut self,
        _: &ModelRequest,
        _: &mut dyn FnMut(StreamEvent),
    ) -> Result<TurnResult, ClientError> {
        Err(ClientError::Stream("native session invalidated".into()))
    }
}

#[test]
fn failed_native_turn_discards_the_adapter_before_the_next_request() {
    let starts = Cell::new(0);
    let mut model = OnDemandModel::new(|| {
        starts.set(starts.get() + 1);
        Ok(FailingModel)
    });
    for _ in 0..2 {
        assert!(model.stream_turn(&request(), &mut |_| {}).is_err());
        assert!(!model.is_initialized());
    }
    assert_eq!(starts.get(), 2);
}

#[test]
fn cancelled_agent_before_first_turn_does_not_initialize_the_optional_feature() {
    let mut model =
        OnDemandModel::<ScriptedModel, _>::new(|| -> Result<ScriptedModel, ClientError> {
            panic!("cancelled agent initialized the model")
        });
    let config = AgentConfig {
        cancel: Some(Arc::new(std::sync::atomic::AtomicBool::new(true))),
        ..AgentConfig::default()
    };
    let result = run_agent_loop(
        &mut model,
        &mut TestEnv { calls: 0 },
        &config,
        "unused",
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(result.stop, LoopStop::Cancelled);
    assert!(!model.is_initialized());
}

#[test]
fn cancelled_provisioning_does_not_create_directories() {
    use greppy_agent::local_model::ProvisionCancel;
    let dir = tempfile::tempdir().unwrap();
    let cache = cache(dir.path());
    let cancel = ProvisionCancel::default();
    cancel.cancel();
    let error = cache
        .ensure_controlled(
            &spec(),
            &cancel,
            |_| panic!("fetch after cancel"),
            |_| panic!("progress after cancel"),
        )
        .unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::ConnectionAborted);
    assert!(!dir.path().join("persistent-models").exists());
}

#[test]
fn lock_wait_is_observable_and_cancellable_without_waiting_for_the_owner() {
    use greppy_agent::local_model::{ProvisionCancel, ProvisionEvent};
    let dir = tempfile::tempdir().unwrap();
    let cache = cache(dir.path());
    let (holding_tx, holding_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        let owner = cache.clone();
        scope.spawn(move || {
            owner
                .ensure_with(
                    &spec(),
                    |out| {
                        holding_tx.send(()).unwrap();
                        release_rx
                            .recv_timeout(std::time::Duration::from_secs(10))
                            .unwrap();
                        out.write_all(BYTES)
                    },
                    |_, _| {},
                )
                .unwrap();
        });
        holding_rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .unwrap();
        let cancel = ProvisionCancel::default();
        let started = std::time::Instant::now();
        let result = cache.ensure_controlled(
            &spec(),
            &cancel,
            |_| panic!("second fetch while owner active"),
            |event| {
                if event == ProvisionEvent::WaitingForCache {
                    cancel.cancel();
                }
            },
        );
        release_tx.send(()).unwrap();
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::ConnectionAborted);
        assert!(started.elapsed() < std::time::Duration::from_secs(2));
    });
}

#[test]
fn cancelling_on_last_byte_or_verification_never_publishes() {
    use greppy_agent::local_model::{ProvisionCancel, ProvisionEvent};
    for at_verification in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let cache = cache(dir.path());
        let cancel = ProvisionCancel::default();
        let mut ready = false;
        let result = cache.ensure_controlled(&spec(), &cancel, |out| {
            out.write_all(BYTES)?;
            // A write_all after cancellation must fail instead of looping on Interrupted.
            out.write_all(b"")
        }, |event| {
            if (at_verification && event == ProvisionEvent::Verifying)
                || (!at_verification && matches!(event, ProvisionEvent::Downloading { received, total } if received == total))
            { cancel.cancel(); }
            ready |= matches!(event, ProvisionEvent::ArtifactReady { .. });
        });
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::ConnectionAborted);
        assert!(!ready);
        assert!(cache.lookup_verified(&spec()).unwrap().is_none());
    }
}

#[test]
fn cache_verification_can_be_cancelled_without_fetching_or_ready_event() {
    use greppy_agent::local_model::{ProvisionCancel, ProvisionEvent};
    let dir = tempfile::tempdir().unwrap();
    let cache = cache(dir.path());
    cache
        .ensure_with(&spec(), |out| out.write_all(BYTES), |_, _| {})
        .unwrap();
    let cancel = ProvisionCancel::default();
    let result = cache.ensure_controlled(
        &spec(),
        &cancel,
        |_| panic!("cache hit fetched"),
        |event| {
            assert!(!matches!(event, ProvisionEvent::ArtifactReady { .. }));
            if event == ProvisionEvent::CheckingCache {
                cancel.cancel();
            }
        },
    );
    assert_eq!(result.unwrap_err().kind(), io::ErrorKind::ConnectionAborted);
}

#[test]
fn cancellation_from_write_progress_cannot_spin_inside_write_all() {
    use greppy_agent::local_model::{ProvisionCancel, ProvisionEvent};
    let dir = tempfile::tempdir().unwrap();
    let cache = cache(dir.path());
    let cancel = ProvisionCancel::default();
    let result = cache.ensure_controlled(
        &spec(),
        &cancel,
        |out| {
            out.write_all(&BYTES[..1])?;
            out.write_all(&BYTES[1..])
        },
        |event| {
            if matches!(event, ProvisionEvent::Downloading { received: 1, .. }) {
                cancel.cancel();
            }
        },
    );
    assert_eq!(result.unwrap_err().kind(), io::ErrorKind::ConnectionAborted);
}
