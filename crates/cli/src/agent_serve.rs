//! Headless hosted agent session.

use std::collections::VecDeque;
use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use greppy_agent::{AgentConfig, Client, GreppyEnv, Usage};
use serde_json::{json, Value};

use crate::agent::{spawn_session_worker, SessionSummary, SessionWorkerParts};
use crate::agent_control::{socket_path_for, ControlServer, Incoming, RpcError};
use crate::agent_json::{
    error_event, phase_event, text_event, tool_finish_event, tool_start_event, turn_complete_event,
    turn_start_event, usage_object, JsonEmitter, JsonSession,
};
use crate::agent_tui::{SessionCommand, SessionEvent, SessionRecord, SessionStore};

const POLL_INTERVAL: Duration = Duration::from_millis(50);
const MAX_PENDING_PROMPTS: usize = 64;
static SIGNALS: AtomicUsize = AtomicUsize::new(0);

extern "C" fn record_signal(_: libc::c_int) {
    let prior = SIGNALS.fetch_add(1, Ordering::SeqCst);
    if prior > 0 {
        unsafe { libc::_exit(130) };
    }
}

struct SignalGuard {
    previous_int: libc::sighandler_t,
    previous_term: libc::sighandler_t,
}

impl SignalGuard {
    fn install() -> Self {
        SIGNALS.store(0, Ordering::SeqCst);
        let handler = record_signal as *const () as libc::sighandler_t;
        let previous_int = unsafe { libc::signal(libc::SIGINT, handler) };
        let previous_term = unsafe { libc::signal(libc::SIGTERM, handler) };
        Self {
            previous_int,
            previous_term,
        }
    }
}

impl Drop for SignalGuard {
    fn drop(&mut self) {
        unsafe {
            libc::signal(libc::SIGINT, self.previous_int);
            libc::signal(libc::SIGTERM, self.previous_term);
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Setup,
    Idle,
    Busy,
    Cancelling,
    Blocked,
}

impl Phase {
    fn label(self) -> &'static str {
        match self {
            Self::Setup => "setup",
            Self::Idle => "idle",
            Self::Busy => "busy",
            Self::Cancelling => "cancelling",
            Self::Blocked => "blocked",
        }
    }
}

struct PendingPrompt {
    id: String,
    text: String,
    source: String,
}

fn enqueue_prompt(
    queue: &mut VecDeque<PendingPrompt>,
    prompt: PendingPrompt,
) -> Result<usize, RpcError> {
    if queue.len() >= MAX_PENDING_PROMPTS {
        return Err(RpcError::new(-32001, "queue full"));
    }
    let position = queue.len() + 1;
    queue.push_back(prompt);
    Ok(position)
}

#[derive(Clone, Copy)]
pub(crate) struct ServeLaunch<'a> {
    pub(crate) task: &'a str,
    pub(crate) endpoint: &'a str,
    pub(crate) model: &'a str,
    pub(crate) sandbox: &'a str,
    pub(crate) idle_timeout_secs: Option<u64>,
    pub(crate) json_session: &'a JsonSession,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    client: Client,
    env: GreppyEnv,
    mut config: AgentConfig,
    store: &SessionStore,
    mut record: SessionRecord,
    resumed: bool,
    emitter: &mut JsonEmitter,
    launch: ServeLaunch<'_>,
) -> Result<SessionSummary, String> {
    if resumed {
        if record.model != launch.model {
            store
                .set_model(&record.id, launch.model)
                .map_err(|error| format!("session save failed: {error}"))?;
            record.model = launch.model.to_string();
        }
    } else if let Err(error) = store.create(&record) {
        if error.kind() != io::ErrorKind::AlreadyExists {
            return Err(format!("session save failed: {error}"));
        }
    }

    let cancel = Arc::new(AtomicBool::new(false));
    config.cancel = Some(Arc::clone(&cancel));
    let handle = spawn_session_worker(SessionWorkerParts {
        client,
        env,
        config,
        endpoint: launch.endpoint.to_string(),
        store: store.clone(),
        record: record.clone(),
        prompt_source: None,
        start_paused: false,
    })?;
    let socket_path = socket_path_for(store, &record.id);
    let mut server = match ControlServer::bind(&socket_path) {
        Ok(server) => server,
        Err(error) => {
            let _ = handle.commands.send(SessionCommand::Quit);
            let _ = handle.join.join();
            return Err(format!("cannot bind control socket: {error}"));
        }
    };
    emitter.serve_session(launch.json_session, &socket_path.display().to_string());
    let _signals = SignalGuard::install();

    let mut phase = Phase::Setup;
    let mut queue = VecDeque::<PendingPrompt>::new();
    let mut next_prompt = 1u64;
    let mut quit = false;
    let mut blocked_error = None::<String>;
    let mut usage = record.usage;
    let mut turns = record.turns;
    let mut current_model = launch.model.to_string();
    let mut current_endpoint = launch.endpoint.to_string();
    let mut last_activity = Instant::now();

    if !launch.task.trim().is_empty() {
        queue.push_back(PendingPrompt {
            id: format!("p-{next_prompt}"),
            text: launch.task.trim().to_string(),
            source: "commandline".to_string(),
        });
        next_prompt = next_prompt.saturating_add(1);
    }

    loop {
        let intake = handle.intake.poll(POLL_INTERVAL);
        let mut worker_error_event = false;
        if !intake.text.is_empty() {
            emit(&mut server, emitter, text_event(&intake.text));
            last_activity = Instant::now();
        }
        for event in intake.discrete {
            match event {
                SessionEvent::SetupReady => {
                    set_phase(&mut server, emitter, &mut phase, Phase::Idle);
                    last_activity = Instant::now();
                }
                SessionEvent::SetupBlocked(message) | SessionEvent::GatewayRequired(message) => {
                    set_phase(&mut server, emitter, &mut phase, Phase::Blocked);
                    emit(&mut server, emitter, error_event(&message));
                    blocked_error = Some(message);
                    quit = true;
                }
                SessionEvent::Configuration {
                    endpoint, model, ..
                } => {
                    current_endpoint = endpoint;
                    current_model = model;
                }
                SessionEvent::ToolStart { id, summary } => {
                    emit(&mut server, emitter, tool_start_event(&id, "", &summary));
                    last_activity = Instant::now();
                }
                SessionEvent::ToolFinish {
                    id,
                    failed,
                    elapsed_ms,
                    preview,
                } => {
                    emit(
                        &mut server,
                        emitter,
                        tool_finish_event(&id, failed, elapsed_ms, &preview),
                    );
                    last_activity = Instant::now();
                }
                SessionEvent::Done {
                    input_tokens,
                    output_tokens,
                    cache_read,
                    cache_write,
                    turns: prompt_turns,
                    stop,
                    ..
                } => {
                    let turn_usage = Usage {
                        input_tokens,
                        output_tokens,
                        cache_read_input_tokens: cache_read,
                        cache_creation_input_tokens: cache_write,
                    };
                    add_usage(&mut usage, &turn_usage);
                    turns = turns.saturating_add(prompt_turns);
                    emit(
                        &mut server,
                        emitter,
                        turn_complete_event(&stop, &turn_usage),
                    );
                    set_phase(&mut server, emitter, &mut phase, Phase::Idle);
                    last_activity = Instant::now();
                }
                SessionEvent::Error(message) => {
                    emit(&mut server, emitter, error_event(&message));
                    set_phase(&mut server, emitter, &mut phase, Phase::Idle);
                    blocked_error = Some(message);
                    quit = true;
                    worker_error_event = true;
                }
                SessionEvent::Warning(message) | SessionEvent::EndpointRejected { message, .. } => {
                    eprintln!("greppy agent serve: {message}");
                }
                SessionEvent::SetupProgress { .. }
                | SessionEvent::BackgroundProgress { .. }
                | SessionEvent::BackgroundReady
                | SessionEvent::Text(_)
                | SessionEvent::Thinking(_)
                | SessionEvent::Compacted { .. } => {}
            }
        }
        if worker_error_event || handle.join.is_finished() {
            break;
        }

        for incoming in server.poll() {
            let Incoming::Request {
                conn,
                id,
                method,
                params,
            } = incoming
            else {
                continue;
            };
            last_activity = Instant::now();
            match method.as_str() {
                "session/describe" => server.reply(
                    conn,
                    id,
                    Ok(describe(
                        &record,
                        launch,
                        &current_model,
                        &current_endpoint,
                        phase,
                        turns,
                        &usage,
                        queue.len(),
                        &socket_path,
                        store,
                    )),
                ),
                "session/subscribe" => server.reply(conn, id, Ok(json!({"subscribed":true}))),
                "turn/start" => {
                    let text = params
                        .get("text")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .trim();
                    if text.is_empty() {
                        server.reply(conn, id, Err(RpcError::new(-32602, "empty prompt")));
                        continue;
                    }
                    if quit {
                        server.reply(conn, id, Err(RpcError::new(-32000, "session is quitting")));
                        continue;
                    }
                    let source = params
                        .get("source")
                        .and_then(Value::as_str)
                        .filter(|source| !source.trim().is_empty())
                        .unwrap_or("remote")
                        .to_string();
                    let prompt_id = format!("p-{next_prompt}");
                    let position = match enqueue_prompt(
                        &mut queue,
                        PendingPrompt {
                            id: prompt_id.clone(),
                            text: text.to_string(),
                            source,
                        },
                    ) {
                        Ok(position) => position,
                        Err(error) => {
                            server.reply(conn, id, Err(error));
                            continue;
                        }
                    };
                    next_prompt = next_prompt.saturating_add(1);
                    server.reply(
                        conn,
                        id,
                        Ok(json!({"accepted":true,"prompt_id":prompt_id,"position":position})),
                    );
                }
                "turn/interrupt" => {
                    if matches!(phase, Phase::Busy | Phase::Cancelling) {
                        cancel.store(true, Ordering::Relaxed);
                        set_phase(&mut server, emitter, &mut phase, Phase::Cancelling);
                    }
                    server.reply(conn, id, Ok(json!({"accepted":true})));
                }
                "session/quit" => {
                    quit = true;
                    queue.clear();
                    server.reply(conn, id, Ok(json!({"accepted":true})));
                }
                _ => server.reply(conn, id, Err(RpcError::new(-32601, "method not found"))),
            }
        }

        if SIGNALS.load(Ordering::SeqCst) > 0 {
            quit = true;
            queue.clear();
        }
        if phase == Phase::Idle && !quit {
            if let Some(prompt) = queue.pop_front() {
                store
                    .append_turn_start(&record.id, &prompt.source, &prompt.text)
                    .map_err(|error| format!("session save failed: {error}"))?;
                emit(
                    &mut server,
                    emitter,
                    turn_start_event(&prompt.id, &prompt.source, &prompt.text),
                );
                handle
                    .commands
                    .send(SessionCommand::Prompt(prompt.text))
                    .map_err(|_| "session worker disconnected".to_string())?;
                set_phase(&mut server, emitter, &mut phase, Phase::Busy);
                last_activity = Instant::now();
            }
        }
        if phase == Phase::Idle
            && launch
                .idle_timeout_secs
                .is_some_and(|secs| last_activity.elapsed() >= Duration::from_secs(secs))
        {
            quit = true;
            queue.clear();
        }
        if quit && !matches!(phase, Phase::Busy | Phase::Cancelling) {
            break;
        }
        if intake.disconnected && !handle.join.is_finished() {
            return Err("session worker disconnected".to_string());
        }
    }

    let _ = handle.commands.send(SessionCommand::Quit);
    drop(server);
    let summary = handle
        .join
        .join()
        .map_err(|_| "session worker panicked".to_string())??;
    if let Some(message) = blocked_error {
        return Err(message);
    }
    Ok(summary)
}

fn emit(server: &mut ControlServer, emitter: &mut JsonEmitter, event: Value) {
    emitter.emit(&event);
    server.broadcast(&event);
}

fn set_phase(
    server: &mut ControlServer,
    emitter: &mut JsonEmitter,
    current: &mut Phase,
    next: Phase,
) {
    if *current != next {
        *current = next;
        emit(server, emitter, phase_event(next.label()));
    }
}

#[allow(clippy::too_many_arguments)]
fn describe(
    record: &SessionRecord,
    launch: ServeLaunch<'_>,
    model: &str,
    endpoint: &str,
    phase: Phase,
    turns: u64,
    usage: &Usage,
    pending: usize,
    socket: &PathBuf,
    store: &SessionStore,
) -> Value {
    json!({
        "session_id": record.id,
        "uri": crate::agent_sessions::session_uri(&record.id),
        "run_id": record.run_id,
        "project": record.project,
        "worktree": record.worktree,
        "branch": record.branch,
        "model": model,
        "endpoint": endpoint,
        "sandbox": launch.sandbox,
        "phase": phase.label(),
        "turns": turns,
        "usage": usage_object(usage),
        "pending": pending,
        "pid": std::process::id(),
        "socket": socket,
        "jsonl": store.path_for(&record.id).unwrap_or_default(),
    })
}

fn add_usage(total: &mut Usage, delta: &Usage) {
    total.input_tokens = total.input_tokens.saturating_add(delta.input_tokens);
    total.output_tokens = total.output_tokens.saturating_add(delta.output_tokens);
    total.cache_read_input_tokens = total
        .cache_read_input_tokens
        .saturating_add(delta.cache_read_input_tokens);
    total.cache_creation_input_tokens = total
        .cache_creation_input_tokens
        .saturating_add(delta.cache_creation_input_tokens);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pending_prompt_queue_is_capped() {
        let mut queue = VecDeque::new();
        for index in 0..MAX_PENDING_PROMPTS {
            let position = enqueue_prompt(
                &mut queue,
                PendingPrompt {
                    id: format!("p-{index}"),
                    text: "prompt".into(),
                    source: "remote".into(),
                },
            )
            .unwrap();
            assert_eq!(position, index + 1);
        }
        let error = enqueue_prompt(
            &mut queue,
            PendingPrompt {
                id: "overflow".into(),
                text: "prompt".into(),
                source: "remote".into(),
            },
        )
        .unwrap_err();
        assert_eq!(error, RpcError::new(-32001, "queue full"));
        assert_eq!(queue.len(), MAX_PENDING_PROMPTS);
    }
}
