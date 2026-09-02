//! Interactive full-screen TUI for `greppy agent`.
//!
//! Modules stay focused: terminal lifecycle, state/update, rendering, editing,
//! transcript, commands, and the agent-worker bridge.

mod bootstrap;
mod commands;
mod composer;
mod events;
mod markdown;
mod overlay;
mod preview;
mod redaction;
mod render;
mod session;
mod settings;
mod state;
mod terminal;
mod theme;
mod update;

use std::io::{self, Write};
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event};
use greppy_agent::Usage;
use serde_json::Value;

use crate::agent::EXIT_USAGE;
use crate::agent_control::{socket_path_for, ControlServer, Incoming};
use crate::agent_json::{
    error_event, phase_event, text_event, tool_finish_event, tool_start_event, turn_complete_event,
    turn_start_event,
};

pub use bootstrap::BootstrapScreen;
pub use events::{
    bounded_pair, EventBridge, EventIntake, RemoteRequest, SessionCommand, SessionEvent,
};
pub use redaction::{redact_json, redact_text, sanitize_terminal_text};
pub use session::{
    compact_messages, list_session_project_dirs, load_path, messages_from_protocol, new_session_id,
    protocol_from_persisted, read_session_log_lines, SessionLogLine, SessionRecord, SessionStore,
};
pub use settings::AgentSettings;
pub use state::HeaderState;
pub use terminal::{tty_suitable, unsupported_tty_message, TerminalCaps, TerminalGuard};
pub use theme::Theme;

use render::render;
use state::App;
use update::{apply_effects, mouse_scroll, update, Action, Effect};

const BUSY_FRAME: Duration = Duration::from_millis(50);

#[derive(Debug, Clone)]
pub struct TuiConfig {
    pub model: String,
    pub endpoint: String,
    pub repository: String,
    pub branch: String,
    pub worktree: String,
    pub sandbox: String,
    pub known_models: Vec<String>,
    pub cancel: Arc<AtomicBool>,
    pub initializing: bool,
    pub settings: AgentSettings,
}

#[derive(Debug, Clone, Default)]
pub struct TuiOutcome {
    pub submitted_prompts: u64,
    pub session_id: String,
    pub title: String,
    pub cancelled: bool,
    pub force_exit: bool,
}

pub fn run(
    config: TuiConfig,
    session: SessionRecord,
    store: SessionStore,
    initial_prompts: Vec<String>,
    initial_draft: String,
    commands: Sender<SessionCommand>,
    events: EventIntake,
    mut control: Option<ControlServer>,
    control_warning: Option<String>,
) -> io::Result<TuiOutcome> {
    if !tty_suitable() {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            unsupported_tty_message(),
        ));
    }
    let caps = TerminalCaps::detect();
    let theme = Theme::detect();
    let (mut terminal, _guard) = TerminalGuard::enter(&caps)?;
    terminal.clear()?;

    let header = HeaderState {
        repository: config.repository,
        branch: config.branch,
        worktree: config.worktree,
        model: config.model,
        endpoint: config.endpoint,
        sandbox: config.sandbox,
    };
    let mut app = App::new(header, theme, &session);
    set_control_paths(&mut app, &store);
    if let Some(warning) = control_warning {
        app.push_warning(warning);
    }
    app.settings = config.settings;
    if config.initializing {
        app.phase = state::RunPhase::Setup;
        app.status = "Preparing repository analysis".into();
        app.gateway_ready = false;
        app.repository_ready = false;
    }
    app.known_models = config.known_models;
    app.known_sessions = store.list().unwrap_or_default();
    app.cancel = config.cancel;

    for prompt in initial_prompts
        .into_iter()
        .filter(|prompt| !prompt.trim().is_empty())
    {
        app.composer.set_text(prompt);
        let effects = update(&mut app, Action::Key(enter_key()));
        dispatch_effects(&mut app, &store, &commands, &mut control, effects);
    }
    if !initial_draft.is_empty() {
        app.composer.set_text(initial_draft);
    }

    terminal.draw(|frame| render(frame, &mut app))?;
    let mut last_pulse = Instant::now();
    loop {
        // Terminal input must be the blocking wait source. Waiting on the
        // worker channel first starved pasted text: terminals that delivered
        // a paste as individual key events advanced by one character per
        // a slow idle interval. Poll at input cadence, then drain both sources.
        let terminal_ready = event::poll(BUSY_FRAME)?;
        let intake = events.try_poll();
        let streamed = !intake.text.is_empty() || !intake.thinking.is_empty();
        if streamed {
            if !intake.text.is_empty() {
                broadcast(&mut control, &text_event(&intake.text));
            }
            let _ = update(
                &mut app,
                Action::Stream {
                    text: intake.text,
                    thinking: intake.thinking,
                },
            );
        }
        let saturated = intake.saturated;
        if saturated {
            let _ = update(&mut app, Action::Saturated);
        }
        let had_discrete = !intake.discrete.is_empty();
        for event in intake.discrete {
            broadcast_session_event(&mut control, &event);
            let previous_phase = app.phase;
            let effects = update(&mut app, Action::Worker(event));
            dispatch_with_phase(
                &mut app,
                &store,
                &commands,
                &mut control,
                previous_phase,
                effects,
            );
        }
        let disconnected = intake.disconnected;
        if disconnected {
            let previous_phase = app.phase;
            let effects = update(&mut app, Action::Disconnect);
            dispatch_with_phase(
                &mut app,
                &store,
                &commands,
                &mut control,
                previous_phase,
                effects,
            );
        }

        let incoming = control
            .as_mut()
            .map(ControlServer::poll)
            .unwrap_or_default();
        let had_remote = !incoming.is_empty();
        for request in incoming {
            let Incoming::Request {
                conn,
                id,
                method,
                params,
            } = request
            else {
                continue;
            };
            let previous_phase = app.phase;
            let effects = update(
                &mut app,
                Action::Remote(RemoteRequest {
                    conn,
                    id,
                    method,
                    params,
                }),
            );
            dispatch_with_phase(
                &mut app,
                &store,
                &commands,
                &mut control,
                previous_phase,
                effects,
            );
        }

        let mut input_changed = false;
        if terminal_ready {
            loop {
                match event::read()? {
                    Event::Key(key) => {
                        input_changed = true;
                        let previous_phase = app.phase;
                        let effects = update(&mut app, Action::Key(key));
                        dispatch_with_phase(
                            &mut app,
                            &store,
                            &commands,
                            &mut control,
                            previous_phase,
                            effects,
                        );
                    }
                    Event::Paste(text) => {
                        input_changed = true;
                        let _ = update(&mut app, Action::Paste(text));
                    }
                    Event::Mouse(mouse) => {
                        if let Some(action) = mouse_scroll(mouse.kind) {
                            input_changed = true;
                            let _ = update(&mut app, action);
                        }
                    }
                    Event::Resize(cols, rows) => {
                        input_changed = true;
                        let _ = update(&mut app, Action::Resize { cols, rows });
                    }
                    Event::FocusGained | Event::FocusLost => {}
                }
                if !event::poll(Duration::from_millis(0))? {
                    break;
                }
            }
        }

        let pulse = app.phase == state::RunPhase::Setup
            && !app.queued.is_empty()
            && last_pulse.elapsed() >= Duration::from_millis(500);
        if pulse {
            let _ = update(&mut app, Action::Tick);
            last_pulse = Instant::now();
        }
        if input_changed
            || streamed
            || saturated
            || had_discrete
            || disconnected
            || had_remote
            || pulse
        {
            terminal.draw(|frame| render(frame, &mut app))?;
        }

        if app.request_exit && (!app.busy() || app.force_exit) {
            let _ = commands.send(SessionCommand::Quit);
            break;
        }
    }

    terminal.show_cursor()?;
    let outcome = TuiOutcome {
        submitted_prompts: app.submitted_prompts,
        session_id: app.session_id,
        title: app.session_title,
        cancelled: app.status == "cancelled" || app.force_exit,
        force_exit: app.force_exit,
    };
    let _ = (
        outcome.submitted_prompts,
        outcome.session_id.as_str(),
        outcome.title.as_str(),
        outcome.cancelled,
        outcome.force_exit,
    );
    Ok(outcome)
}

pub fn refuse_nontty() -> u8 {
    eprintln!("{}", unsupported_tty_message());
    EXIT_USAGE
}

fn dispatch_effects(
    app: &mut App,
    store: &SessionStore,
    commands: &Sender<SessionCommand>,
    control: &mut Option<ControlServer>,
    effects: Vec<Effect>,
) {
    for effect in &effects {
        match effect {
            Effect::Cancel => {
                app.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            Effect::PersistTitle(title) => {
                if let Err(error) = store.set_title(&app.session_id, title) {
                    app.persist_warning = Some(error.to_string());
                    app.push_warning(format!("session save failed: {error}"));
                }
            }
            Effect::Copy(text) => match copy_osc52(text) {
                Ok(()) => app.copy_status = Some("copied last assistant reply".into()),
                Err(message) => {
                    app.copy_status = Some(message);
                    app.push_warning(
                        "clipboard OSC 52 write failed; select text with Shift+mouse or disable mouse capture",
                    );
                }
            },
            Effect::SaveSettings(settings) => {
                if let Err(error) = settings.save() {
                    app.push_warning(format!("settings save failed: {error}"));
                }
            }
            Effect::RemoteReply { conn, id, result } => {
                if let Some(server) = control.as_mut() {
                    server.reply(*conn, id.clone(), result.clone());
                }
            }
            Effect::ResumeSession(id) => {
                let path = socket_path_for(store, id);
                *control = None;
                match ControlServer::bind(&path) {
                    Ok(server) => *control = Some(server),
                    Err(error) => {
                        app.push_warning(format!("remote control unavailable: {error}"));
                    }
                }
                set_control_paths(app, store);
            }
            Effect::SubmitRemote {
                text,
                source,
                prompt_id,
            } => {
                broadcast(control, &turn_start_event(prompt_id, source, text));
            }
            Effect::Submit(text) => {
                let prompt_id = format!("p-{}", app.next_prompt_id);
                app.next_prompt_id = app.next_prompt_id.saturating_add(1);
                broadcast(control, &turn_start_event(&prompt_id, "interactive", text));
            }
            Effect::Quit | Effect::SetModel(_) | Effect::SetEndpoint(_) | Effect::Compact => {}
        }
    }
    for command in apply_effects(&effects) {
        if commands.send(command).is_err() {
            app.push_error("The agent worker stopped unexpectedly.");
            app.request_exit = true;
        }
    }
}

fn dispatch_with_phase(
    app: &mut App,
    store: &SessionStore,
    commands: &Sender<SessionCommand>,
    control: &mut Option<ControlServer>,
    previous_phase: state::RunPhase,
    effects: Vec<Effect>,
) {
    dispatch_effects(app, store, commands, control, effects);
    if previous_phase.control_label() != app.phase.control_label() {
        broadcast(control, &phase_event(app.phase.control_label()));
    }
}

fn set_control_paths(app: &mut App, store: &SessionStore) {
    app.control_socket = socket_path_for(store, &app.session_id)
        .display()
        .to_string();
    app.session_jsonl = store
        .path_for(&app.session_id)
        .unwrap_or_default()
        .display()
        .to_string();
}

fn broadcast(control: &mut Option<ControlServer>, event: &Value) {
    if let Some(server) = control.as_mut() {
        server.broadcast(event);
    }
}

fn broadcast_session_event(control: &mut Option<ControlServer>, event: &SessionEvent) {
    let event = match event {
        SessionEvent::Text(text) => Some(text_event(text)),
        SessionEvent::ToolStart { id, summary } => Some(tool_start_event(id, "", summary)),
        SessionEvent::ToolFinish {
            id,
            failed,
            elapsed_ms,
            preview,
        } => Some(tool_finish_event(id, *failed, *elapsed_ms, preview)),
        SessionEvent::Done {
            input_tokens,
            output_tokens,
            cache_read,
            cache_write,
            stop,
            ..
        } => Some(turn_complete_event(
            stop,
            &Usage {
                input_tokens: *input_tokens,
                output_tokens: *output_tokens,
                cache_read_input_tokens: *cache_read,
                cache_creation_input_tokens: *cache_write,
            },
        )),
        SessionEvent::Error(message)
        | SessionEvent::SetupBlocked(message)
        | SessionEvent::GatewayRequired(message) => Some(error_event(message)),
        SessionEvent::SetupProgress { .. }
        | SessionEvent::BackgroundProgress { .. }
        | SessionEvent::BackgroundReady
        | SessionEvent::SetupReady
        | SessionEvent::EndpointRejected { .. }
        | SessionEvent::Configuration { .. }
        | SessionEvent::Thinking(_)
        | SessionEvent::Compacted { .. }
        | SessionEvent::Warning(_) => None,
    };
    if let Some(event) = event {
        broadcast(control, &event);
    }
}

fn copy_osc52(text: &str) -> Result<(), String> {
    let encoded = base64(text.as_bytes());
    let seq = format!("\x1b]52;c;{encoded}\x07");
    let mut stdout = io::stdout();
    stdout
        .write_all(seq.as_bytes())
        .and_then(|_| stdout.flush())
        .map_err(|error| error.to_string())
}

fn base64(bytes: &[u8]) -> String {
    const TABLE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    let mut i = 0;
    while i < bytes.len() {
        let b0 = bytes[i];
        let b1 = if i + 1 < bytes.len() { bytes[i + 1] } else { 0 };
        let b2 = if i + 2 < bytes.len() { bytes[i + 2] } else { 0 };
        out.push(TABLE[(b0 >> 2) as usize] as char);
        out.push(TABLE[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
        if i + 1 < bytes.len() {
            out.push(TABLE[(((b1 & 0x0f) << 2) | (b2 >> 6)) as usize] as char);
        } else {
            out.push('=');
        }
        if i + 2 < bytes.len() {
            out.push(TABLE[(b2 & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
        i += 3;
    }
    out
}

fn enter_key() -> crossterm::event::KeyEvent {
    crossterm::event::KeyEvent::new(
        crossterm::event::KeyCode::Enter,
        crossterm::event::KeyModifiers::NONE,
    )
}

#[cfg(test)]
pub fn render_test(app: &mut App, width: u16, height: u16) -> String {
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test backend");
    terminal.draw(|frame| render(frame, app)).expect("draw");
    let buf = terminal.backend().buffer();
    let area = buf.area();
    let mut out = String::new();
    for y in 0..area.height {
        for x in 0..area.width {
            out.push_str(buf[(x, y)].symbol());
        }
        while out.ends_with(' ') {
            out.pop();
        }
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod snapshots {
    use super::*;
    use overlay::Overlay;
    use state::RunPhase;

    fn base() -> App {
        let mut app = preview::sample_idle_app();
        app.theme = Theme {
            color: false,
            ascii: true,
        };
        app
    }

    fn assert_fits(out: &str, width: u16, height: u16, needle: &str) {
        let rows = out.lines().count() as u16;
        assert_eq!(rows, height, "height {width}x{height}\n{out}");
        assert!(
            !out.contains("terminal too small") || width < 60,
            "unexpected too-small {width}x{height}\n{out}"
        );
        assert!(
            out.contains(needle),
            "missing {needle} in {width}x{height}\n{out}"
        );
    }

    #[test]
    fn snapshots_cover_required_states() {
        let sizes = [(120, 36), (80, 24), (60, 18)];
        for (w, h) in sizes {
            let mut idle = base();
            assert_fits(&render_test(&mut idle, w, h), w, h, "greppy");

            let mut streaming = base();
            streaming.phase = RunPhase::Busy;
            streaming.append_assistant(" more tokens");
            assert_fits(&render_test(&mut streaming, w, h), w, h, "greppy");

            let mut thinking = base();
            thinking.append_thinking("planning");
            assert_fits(&render_test(&mut thinking, w, h), w, h, "thinking");

            let mut running = base();
            running.start_tool("r1".into(), "greppy search-symbol parse".into());
            assert_fits(&render_test(&mut running, w, h), w, h, "tool");

            let mut failed = base();
            failed.start_tool("f1".into(), "greppy replace parse".into());
            failed.finish_tool("f1", true, 9, "verify failed".into());
            assert_fits(&render_test(&mut failed, w, h), w, h, "tool");

            let mut scrolled = base();
            for i in 0..20 {
                scrolled.push_user(format!("turn {i}"));
            }
            scrolled.follow_tail = false;
            scrolled.scroll = 2;
            let out = render_test(&mut scrolled, w, h);
            assert_fits(&out, w, h, "you");

            let mut help = base();
            help.overlay = Overlay::Help;
            assert_fits(&render_test(&mut help, w, h), w, h, "Commands");

            let mut setup = base();
            setup.overlay = Overlay::setup();
            assert_fits(
                &render_test(&mut setup, w, h),
                w,
                h,
                "All interactive agent settings",
            );

            let mut models = base();
            models.known_models = vec!["alpha".into(), "beta".into()];
            models.overlay = Overlay::models(&models.known_models, "alpha", "");
            assert_fits(&render_test(&mut models, w, h), w, h, "alpha");

            let mut queued = base();
            queued.phase = RunPhase::Busy;
            queued.queued.push_back("follow up".into());
            queued.items.push(state::TranscriptItem::Queued {
                text: "follow up".into(),
            });
            assert_fits(&render_test(&mut queued, w, h), w, h, "queued");

            let mut error = base();
            error.push_error("gateway closed");
            assert_fits(&render_test(&mut error, w, h), w, h, "error");
        }
        let mut tiny = base();
        let out = render_test(&mut tiny, 40, 10);
        assert!(out.contains("terminal too small"), "{out}");
    }

    #[test]
    fn startup_uses_live_progress_and_keeps_the_composer_interactive() {
        let mut startup = base();
        startup.phase = RunPhase::Setup;
        startup.status = "Generating embeddings (Metal GPU)".into();
        startup.setup_history = vec!["Analyzing source code (611 files)".into()];
        startup.setup_completed = 256;
        startup.setup_total = 9_125;
        startup.setup_unit = "spans".into();
        startup.setup_rate_milli_per_second = Some(14_800);
        startup.setup_eta_seconds = Some(599);
        startup.setup_detail = Some("agent::run_interactive_session".into());
        let out = render_test(&mut startup, 100, 28);
        assert!(out.contains(&format!("greppy agent {}", env!("CARGO_PKG_VERSION"))));
        assert!(!out.contains("[ OK ]"), "legacy boot rows leaked:\n{out}");
        assert!(!out.contains("[ .. ]"), "legacy boot rows leaked:\n{out}");
        assert!(out.contains("Generating embeddings (Metal GPU)"));
        assert!(out.contains("256 / 9125 spans"));
        assert!(out.contains("prompt · indexing continues in background"));
        assert!(out.contains("symbol  ·  agent::run_interactive_session"));
        assert!(
            !out.contains("in 0 out 0"),
            "startup exposed token footer:\n{out}"
        );
    }
}

#[cfg(test)]
mod preview_write {
    use super::*;

    #[test]
    fn regenerate_docs_previews_when_requested() {
        if std::env::var_os("GREPPY_WRITE_TUI_PREVIEWS").is_none() {
            return;
        }
        let dir = preview::preview_dir();
        preview::write_previews(&dir).expect("write previews");
    }
}
