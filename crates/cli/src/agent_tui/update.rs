//! Pure state transitions for keys, overlays, queue, and worker events.

use std::time::{Duration, Instant};

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEventKind};

use super::commands::{parse_slash, SlashCommand};
use super::events::{SessionCommand, SessionEvent};
use super::overlay::{Overlay, ToolOverlay};
use super::redaction::sanitize_terminal_text;
use super::settings::AgentSettings;
use super::state::{App, RunPhase, TranscriptItem, MIN_COLS, MIN_ROWS};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Effect {
    Submit(String),
    Cancel,
    Quit,
    SetModel(String),
    SetEndpoint(String),
    ResumeSession(String),
    Compact,
    PersistTitle(String),
    Copy(String),
    SaveSettings(AgentSettings),
}

#[derive(Debug, Clone)]
pub enum Action {
    Key(KeyEvent),
    Paste(String),
    Scroll { up: bool, amount: u16 },
    Resize { cols: u16, rows: u16 },
    Worker(SessionEvent),
    Stream { text: String, thinking: String },
    Disconnect,
    Saturated,
    Tick,
}

pub fn update(app: &mut App, action: Action) -> Vec<Effect> {
    match action {
        Action::Resize { cols, rows } => {
            handle_resize(app, cols, rows);
            Vec::new()
        }
        Action::Tick => {
            app.spinner_tick = app.spinner_tick.wrapping_add(1);
            Vec::new()
        }
        Action::Paste(text) => {
            if app.phase == RunPhase::Blocked {
                return Vec::new();
            }
            if matches!(app.overlay, Overlay::Setup(_)) {
                return Vec::new();
            }
            if let Overlay::Model(picker) | Overlay::Sessions(picker) = &mut app.overlay {
                for ch in text.chars() {
                    picker.push_filter(ch);
                }
                return Vec::new();
            }
            app.composer.insert_text(&text);
            app.refresh_completions();
            Vec::new()
        }
        Action::Scroll { up, amount } => {
            scroll(app, up, amount);
            Vec::new()
        }
        Action::Disconnect => {
            app.push_error("The agent worker disconnected.");
            app.request_exit = true;
            Vec::new()
        }
        Action::Saturated => {
            app.push_warning("event channel saturated; coalesced stream deltas");
            Vec::new()
        }
        Action::Stream { text, thinking } => {
            if !thinking.is_empty() {
                app.append_thinking(&thinking);
            }
            if !text.is_empty() {
                app.append_assistant(&text);
            }
            follow_if_needed(app);
            Vec::new()
        }
        Action::Worker(event) => handle_worker(app, event),
        Action::Key(key) => handle_key(app, key),
    }
}

fn handle_resize(app: &mut App, cols: u16, rows: u16) {
    let previous_follow = app.follow_tail;
    let previous_scroll = app.scroll;
    app.cols = cols;
    app.rows = rows;
    if cols < MIN_COLS || rows < MIN_ROWS {
        app.overlay = Overlay::TooSmall { cols, rows };
        return;
    }
    if matches!(app.overlay, Overlay::TooSmall { .. }) {
        app.overlay = Overlay::None;
    }
    if previous_follow {
        app.follow_tail = true;
    } else {
        app.scroll = previous_scroll.min(app.max_scroll);
        app.follow_tail = false;
    }
}

fn handle_worker(app: &mut App, event: SessionEvent) -> Vec<Effect> {
    match event {
        SessionEvent::SetupProgress {
            phase,
            detail,
            unit,
            completed,
            total,
            rate_milli_per_second,
            eta_seconds,
            elapsed_seconds,
        } => {
            let phase = sanitize_terminal_text(&phase).into_owned();
            if app.status != phase
                && !matches!(app.status.as_str(), "ready" | "Preparing workspace")
                && app.setup_error.is_none()
            {
                let completed = if app.setup_total > 0 {
                    format!("{} ({} {})", app.status, app.setup_total, app.setup_unit)
                } else {
                    app.status.clone()
                };
                if app.setup_history.last() != Some(&completed) {
                    app.setup_history.push(completed);
                }
            }
            if app.phase != RunPhase::Cancelling {
                app.phase = RunPhase::Setup;
            }
            app.setup_error = None;
            app.status = phase;
            app.setup_detail = detail
                .map(|value| sanitize_terminal_text(&value).into_owned())
                .filter(|value| !value.is_empty());
            app.setup_unit = sanitize_terminal_text(&unit).into_owned();
            app.setup_completed = completed;
            app.setup_total = total;
            app.setup_rate_milli_per_second = rate_milli_per_second;
            app.setup_eta_seconds = eta_seconds;
            app.setup_elapsed_seconds = elapsed_seconds;
        }
        SessionEvent::SetupReady => {
            if app.pending_endpoint.is_some() {
                return Vec::new();
            }
            app.gateway_ready = true;
            return finish_startup_if_ready(app);
        }
        SessionEvent::BackgroundProgress {
            phase,
            detail,
            unit,
            completed,
            total,
            rate_milli_per_second,
            eta_seconds,
        } => {
            let phase = sanitize_terminal_text(&phase).into_owned();
            let unit = sanitize_terminal_text(&unit).into_owned();
            let mut parts = vec![if total > 0 {
                let percent = completed
                    .min(total)
                    .saturating_mul(100)
                    .checked_div(total)
                    .unwrap_or(0);
                if unit == "spans" {
                    format!("{phase} · {percent}% · {completed}/{total}")
                } else {
                    format!("{phase} · {percent}% · {completed}/{total} {unit}")
                }
            } else {
                phase
            }];
            if let Some(rate) = rate_milli_per_second {
                parts.push(format!("{:.1}/s", rate as f64 / 1000.0));
            }
            if let Some(eta) = eta_seconds {
                parts.push(format!("ETA {}", super::render::format_duration(eta)));
            }
            if let Some(detail) = detail {
                let detail = sanitize_terminal_text(&detail).into_owned();
                if !detail.is_empty() {
                    parts.push(detail);
                }
            }
            app.background_status = Some(parts.join(" · "));
        }
        SessionEvent::BackgroundReady => {
            app.background_status = None;
            app.repository_ready = true;
            if matches!(app.phase, RunPhase::Setup | RunPhase::Configuring) {
                return finish_startup_if_ready(app);
            }
        }
        SessionEvent::SetupBlocked(message) => {
            app.phase = RunPhase::Blocked;
            app.status = "Startup failed".into();
            app.setup_error = Some(sanitize_terminal_text(&message).into_owned());
        }
        SessionEvent::GatewayRequired(message) => {
            if app.pending_endpoint.is_some() {
                return Vec::new();
            }
            app.phase = RunPhase::Configuring;
            app.gateway_ready = false;
            app.status = "Connect model gateway".into();
            app.setup_error = Some(sanitize_terminal_text(&message).into_owned());
            app.composer.clear();
            app.completion = None;
        }
        SessionEvent::EndpointRejected { endpoint, message } => {
            let endpoint = sanitize_terminal_text(&endpoint).into_owned();
            if app.pending_endpoint.as_deref() != Some(endpoint.as_str()) {
                return Vec::new();
            }
            app.pending_endpoint = None;
            app.gateway_ready = false;
            app.persist_next_configuration = false;
            app.phase = RunPhase::Configuring;
            app.status = "Gateway connection failed".into();
            app.setup_error = Some(sanitize_terminal_text(&message).into_owned());
        }
        SessionEvent::Configuration {
            endpoint,
            model,
            models,
        } => {
            let endpoint = sanitize_terminal_text(&endpoint).into_owned();
            if let Some(pending) = app.pending_endpoint.as_deref() {
                if pending != endpoint {
                    return Vec::new();
                }
                app.pending_endpoint = None;
            }
            app.header.endpoint = endpoint;
            app.header.model = sanitize_terminal_text(&model).into_owned();
            app.known_models = models
                .into_iter()
                .map(|value| sanitize_terminal_text(&value).into_owned())
                .collect();
            app.setup_error = None;
            app.persist_warning = None;
            app.status = "Model gateway connected".into();
            if app.persist_next_configuration {
                app.persist_next_configuration = false;
                app.settings.endpoint = Some(app.header.endpoint.clone());
                app.settings.model = Some(app.header.model.clone());
                return vec![Effect::SaveSettings(app.settings.clone())];
            }
        }
        SessionEvent::Text(delta) => {
            app.append_assistant(&delta);
            follow_if_needed(app);
        }
        SessionEvent::Thinking(delta) => {
            app.append_thinking(&delta);
            follow_if_needed(app);
        }
        SessionEvent::ToolStart { id, summary } => {
            app.start_tool(id, summary);
            follow_if_needed(app);
        }
        SessionEvent::ToolFinish {
            id,
            failed,
            elapsed_ms,
            preview,
        } => app.finish_tool(&id, failed, elapsed_ms, preview),
        SessionEvent::Done {
            input_tokens,
            output_tokens,
            cache_read,
            cache_write,
            turns,
            stop,
            ..
        } => {
            app.input_tokens = app.input_tokens.saturating_add(input_tokens);
            app.output_tokens = app.output_tokens.saturating_add(output_tokens);
            app.cache_read = app.cache_read.saturating_add(cache_read);
            app.cache_write = app.cache_write.saturating_add(cache_write);
            app.turns = app.turns.saturating_add(turns);
            app.close_streaming();
            if app.phase == RunPhase::Cancelling {
                app.phase = RunPhase::Idle;
                app.status = "cancelled".into();
            } else {
                app.phase = RunPhase::Idle;
                app.status = sanitize_terminal_text(&stop).into_owned();
            }
            if app.request_exit {
                return vec![Effect::Quit];
            }
            if let Some(next) = app.queued.pop_front() {
                convert_queued_to_user(app, &next);
                app.phase = RunPhase::Busy;
                app.status = "working".into();
                app.submitted_prompts = app.submitted_prompts.saturating_add(1);
                app.follow_tail = true;
                return vec![Effect::Submit(next)];
            }
        }
        SessionEvent::Compacted { messages } => {
            app.load_messages(&messages);
            app.push_warning("compacted older messages; recent turns kept");
            app.follow_tail = true;
        }
        SessionEvent::Error(message) => {
            app.push_error(message);
            if app.request_exit {
                return vec![Effect::Quit];
            }
        }
        SessionEvent::Warning(message) => {
            app.persist_warning = Some(sanitize_terminal_text(&message).into_owned());
            app.push_warning(message);
        }
    }
    Vec::new()
}

fn finish_startup_if_ready(app: &mut App) -> Vec<Effect> {
    if !app.gateway_ready || !app.repository_ready {
        if app.gateway_ready && !app.repository_ready {
            app.phase = RunPhase::Setup;
            app.status = if app.queued.is_empty() {
                "One-time repository code analysis in progress".into()
            } else {
                "Queued — starts after this repository's one-time code analysis completes".into()
            };
            app.setup_error = None;
        }
        return Vec::new();
    }

    app.phase = RunPhase::Idle;
    app.status = "ready".into();
    app.setup_error = None;
    app.persist_warning = None;
    app.cancel
        .store(false, std::sync::atomic::Ordering::Relaxed);
    if let Some(next) = app.queued.pop_front() {
        convert_queued_to_user(app, &next);
        app.phase = RunPhase::Busy;
        app.status = "working".into();
        app.submitted_prompts = app.submitted_prompts.saturating_add(1);
        app.follow_tail = true;
        return vec![Effect::Submit(next)];
    }
    Vec::new()
}

fn handle_key(app: &mut App, key: KeyEvent) -> Vec<Effect> {
    if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
        return Vec::new();
    }
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        return handle_ctrl_c(app);
    }
    if app.phase == RunPhase::Blocked {
        return Vec::new();
    }
    if app.too_small() && !matches!(key.code, KeyCode::Esc) {
        return Vec::new();
    }
    if key.code == KeyCode::Esc {
        return handle_esc(app);
    }
    if app.overlay.is_open() {
        return handle_overlay_key(app, key);
    }
    if let Some(effects) = handle_completion_keys(app, &key) {
        return effects;
    }
    match key.code {
        KeyCode::Enter
            if key
                .modifiers
                .intersects(KeyModifiers::SHIFT | KeyModifiers::ALT) =>
        {
            app.composer.insert_newline();
            Vec::new()
        }
        KeyCode::Enter => submit(app),
        KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.composer.clear();
            app.completion = None;
            Vec::new()
        }
        KeyCode::Char(ch) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.composer.insert_char(ch);
            app.refresh_completions();
            Vec::new()
        }
        KeyCode::Backspace => {
            app.composer.backspace();
            app.refresh_completions();
            Vec::new()
        }
        KeyCode::Delete => {
            app.composer.delete();
            app.refresh_completions();
            Vec::new()
        }
        KeyCode::Left if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.composer.move_word_left();
            Vec::new()
        }
        KeyCode::Right if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.composer.move_word_right();
            Vec::new()
        }
        KeyCode::Left => {
            app.composer.move_left();
            Vec::new()
        }
        KeyCode::Right => {
            app.composer.move_right();
            Vec::new()
        }
        KeyCode::Up => {
            if app.composer.is_empty() || app.composer.text().lines().count() == 1 {
                app.composer.history_up();
            } else {
                app.composer.move_line_up();
            }
            Vec::new()
        }
        KeyCode::Down => {
            if app.composer.is_empty() || app.composer.history_index_active() {
                app.composer.history_down();
            } else {
                app.composer.move_line_down();
            }
            Vec::new()
        }
        KeyCode::Home => {
            app.composer.move_home();
            Vec::new()
        }
        KeyCode::End => {
            app.follow_tail = true;
            app.scroll = app.max_scroll;
            app.composer.move_end();
            Vec::new()
        }
        KeyCode::PageUp => {
            scroll(app, true, page_amount(app));
            Vec::new()
        }
        KeyCode::PageDown => {
            scroll(app, false, page_amount(app));
            Vec::new()
        }
        KeyCode::Tab => {
            app.refresh_completions();
            if app.completion.is_some() {
                move_completion(app, false);
            }
            Vec::new()
        }
        KeyCode::BackTab => {
            move_completion(app, true);
            Vec::new()
        }
        _ => Vec::new(),
    }
}

fn move_completion(app: &mut App, prev: bool) {
    let Some(menu) = app.completion.as_mut() else {
        return;
    };
    if menu.items.is_empty() {
        return;
    }
    if prev {
        menu.selected = if menu.selected == 0 {
            menu.items.len() - 1
        } else {
            menu.selected - 1
        };
    } else {
        menu.selected = (menu.selected + 1) % menu.items.len();
    }
}

fn handle_completion_keys(app: &mut App, key: &KeyEvent) -> Option<Vec<Effect>> {
    if key.code == KeyCode::Enter && key.modifiers.is_empty() {
        let text = app.composer.text();
        let exact_slash = !text.is_empty()
            && !text.chars().any(char::is_whitespace)
            && parse_slash(text)
                .is_some_and(|command| !matches!(command, SlashCommand::Unknown(_)));
        if exact_slash {
            // Let the normal Enter path submit exact commands such as
            // `/help`; completion acceptance remains available for prefixes.
            return None;
        }
    }
    let menu = app.completion.as_mut()?;
    match key.code {
        KeyCode::Tab | KeyCode::Down => {
            if !menu.items.is_empty() {
                menu.selected = (menu.selected + 1) % menu.items.len();
            }
            Some(Vec::new())
        }
        KeyCode::BackTab | KeyCode::Up => {
            if !menu.items.is_empty() {
                menu.selected = if menu.selected == 0 {
                    menu.items.len() - 1
                } else {
                    menu.selected - 1
                };
            }
            Some(Vec::new())
        }
        KeyCode::Enter if key.modifiers.is_empty() => {
            app.apply_completion();
            Some(Vec::new())
        }
        _ => None,
    }
}

fn handle_overlay_key(app: &mut App, key: KeyEvent) -> Vec<Effect> {
    let busy = app.busy();
    match &mut app.overlay {
        Overlay::ConfirmClear => match key.code {
            KeyCode::Char('y' | 'Y') => {
                app.items.clear();
                app.scroll = 0;
                app.max_scroll = 0;
                app.follow_tail = true;
                app.overlay = Overlay::None;
                Vec::new()
            }
            KeyCode::Char('n' | 'N') | KeyCode::Enter => {
                app.overlay = Overlay::None;
                Vec::new()
            }
            _ => Vec::new(),
        },
        Overlay::Help | Overlay::Usage => {
            if matches!(key.code, KeyCode::Enter | KeyCode::Char('q')) {
                app.overlay = Overlay::None;
            }
            Vec::new()
        }
        Overlay::Tools(tools) => match key.code {
            KeyCode::Up | KeyCode::BackTab => {
                if tools.count > 0 {
                    tools.selected = tools.selected.saturating_sub(1);
                }
                Vec::new()
            }
            KeyCode::Down | KeyCode::Tab => {
                if tools.count > 0 {
                    tools.selected = (tools.selected + 1).min(tools.count.saturating_sub(1));
                }
                Vec::new()
            }
            KeyCode::Enter => {
                app.toggle_selected_tool();
                Vec::new()
            }
            _ => Vec::new(),
        },
        Overlay::Setup(menu) => match key.code {
            KeyCode::Up | KeyCode::BackTab => {
                menu.move_prev();
                Vec::new()
            }
            KeyCode::Down | KeyCode::Tab => {
                menu.move_next();
                Vec::new()
            }
            KeyCode::Enter => match menu.selected {
                0 => {
                    app.overlay = Overlay::None;
                    app.phase = RunPhase::Configuring;
                    app.status = "Configure model gateway".into();
                    app.setup_error = None;
                    app.composer.set_text(app.header.endpoint.clone());
                    app.completion = None;
                    Vec::new()
                }
                1 => {
                    app.overlay = Overlay::models(&app.known_models, &app.header.model, "");
                    Vec::new()
                }
                2 => {
                    app.copy_status = Some("English is the active interface language".into());
                    Vec::new()
                }
                3 => {
                    app.settings.private_store = !app.settings.private_store;
                    vec![Effect::SaveSettings(app.settings.clone())]
                }
                4 => {
                    app.settings.no_sandbox = !app.settings.no_sandbox;
                    vec![Effect::SaveSettings(app.settings.clone())]
                }
                5 => {
                    app.settings.skip_selfcheck = !app.settings.skip_selfcheck;
                    vec![Effect::SaveSettings(app.settings.clone())]
                }
                6 => {
                    app.settings.acceleration = if app.settings.acceleration == "cpu" {
                        "auto".into()
                    } else {
                        "cpu".into()
                    };
                    vec![Effect::SaveSettings(app.settings.clone())]
                }
                7 => {
                    app.settings.workspace_backend =
                        match app.settings.workspace_backend.as_str() {
                            "auto" => "native",
                            "native" => "cow",
                            _ => "auto",
                        }
                        .into();
                    vec![Effect::SaveSettings(app.settings.clone())]
                }
                _ => {
                    app.overlay = Overlay::None;
                    Vec::new()
                }
            },
            _ => Vec::new(),
        },
        Overlay::Model(picker) => match key.code {
            KeyCode::Up | KeyCode::BackTab => {
                picker.move_prev();
                Vec::new()
            }
            KeyCode::Down | KeyCode::Tab => {
                picker.move_next();
                Vec::new()
            }
            KeyCode::Enter => {
                if let Some(item) = picker.selected_item().cloned() {
                    app.header.model = item.id.clone();
                    app.settings.model = Some(item.id.clone());
                    app.overlay = Overlay::None;
                    return vec![
                        Effect::SetModel(item.id),
                        Effect::SaveSettings(app.settings.clone()),
                    ];
                }
                Vec::new()
            }
            KeyCode::Backspace => {
                picker.pop_filter();
                Vec::new()
            }
            KeyCode::Char(ch) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                picker.push_filter(ch);
                Vec::new()
            }
            _ => Vec::new(),
        },
        Overlay::Sessions(picker) => match key.code {
            KeyCode::Up | KeyCode::BackTab => {
                picker.move_prev();
                Vec::new()
            }
            KeyCode::Down | KeyCode::Tab => {
                picker.move_next();
                Vec::new()
            }
            KeyCode::Enter => {
                if busy {
                    app.push_warning("finish or cancel the active run before switching sessions");
                    return Vec::new();
                }
                if let Some(item) = picker.selected_item().cloned() {
                    if let Some(record) = app
                        .known_sessions
                        .iter()
                        .find(|record| record.id == item.id)
                        .cloned()
                    {
                        app.session_id = record.id.clone();
                        app.session_title = record.title.clone();
                        app.header.model = record.model.clone();
                        app.input_tokens = record.usage.input_tokens;
                        app.output_tokens = record.usage.output_tokens;
                        app.cache_read = record.usage.cache_read_input_tokens;
                        app.cache_write = record.usage.cache_creation_input_tokens;
                        app.turns = record.turns;
                        app.status = if record.stop.is_empty() {
                            "idle".into()
                        } else {
                            record.stop.clone()
                        };
                        app.load_messages(&record.messages);
                        app.overlay = Overlay::None;
                        app.push_warning(format!("restored session {}", record.id));
                        return vec![Effect::ResumeSession(record.id)];
                    }
                }
                Vec::new()
            }
            KeyCode::Backspace => {
                picker.pop_filter();
                Vec::new()
            }
            KeyCode::Char(ch) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                picker.push_filter(ch);
                Vec::new()
            }
            _ => Vec::new(),
        },
        Overlay::TooSmall { .. } | Overlay::None => Vec::new(),
    }
}

fn handle_esc(app: &mut App) -> Vec<Effect> {
    if app.overlay.is_open() && !matches!(app.overlay, Overlay::TooSmall { .. }) {
        app.overlay = Overlay::None;
        return Vec::new();
    }
    if app.completion.take().is_some() {
        return Vec::new();
    }
    Vec::new()
}

fn handle_ctrl_c(app: &mut App) -> Vec<Effect> {
    let now = Instant::now();
    let double = app
        .last_ctrl_c
        .is_some_and(|prev| now.duration_since(prev) < Duration::from_secs(2));
    app.last_ctrl_c = Some(now);
    if matches!(
        app.phase,
        RunPhase::Setup | RunPhase::Configuring | RunPhase::Blocked
    ) {
        app.force_exit = true;
        app.request_exit = true;
        app.status = "Startup cancelled".into();
        return vec![Effect::Cancel, Effect::Quit];
    }
    if app.busy() {
        if double || app.phase == RunPhase::Cancelling {
            app.force_exit = true;
            app.request_exit = true;
            return vec![Effect::Cancel, Effect::Quit];
        }
        app.phase = RunPhase::Cancelling;
        app.status = "cancelling at next safe boundary".into();
        return vec![Effect::Cancel];
    }
    app.request_exit = true;
    vec![Effect::Quit]
}

fn submit(app: &mut App) -> Vec<Effect> {
    let Some(prompt) = app.composer.submit() else {
        return Vec::new();
    };
    let prompt = sanitize_terminal_text(&prompt).into_owned();
    app.completion = None;
    if app.phase == RunPhase::Configuring {
        let endpoint = prompt.trim().to_string();
        if endpoint.is_empty() {
            return Vec::new();
        }
        app.phase = RunPhase::Setup;
        app.setup_error = None;
        app.status = "Connecting model gateway".into();
        app.persist_next_configuration = true;
        app.pending_endpoint = Some(endpoint.clone());
        return vec![Effect::SetEndpoint(endpoint)];
    }
    if let Some(command) = parse_slash(&prompt) {
        return dispatch_slash(app, command);
    }
    app.follow_tail = true;
    if app.busy() {
        app.queued.push_back(prompt.clone());
        app.items.push(TranscriptItem::Queued { text: prompt });
        app.status = if app.phase == RunPhase::Setup && !app.repository_ready {
            "Queued — starts after this repository's one-time code analysis completes".into()
        } else {
            format!("queued {}", app.queued.len())
        };
        return Vec::new();
    }
    app.push_user(prompt.clone());
    app.phase = RunPhase::Busy;
    app.status = "working".into();
    app.submitted_prompts = app.submitted_prompts.saturating_add(1);
    vec![Effect::Submit(prompt)]
}

fn dispatch_slash(app: &mut App, command: SlashCommand) -> Vec<Effect> {
    match command {
        SlashCommand::Help => {
            app.overlay = Overlay::help();
            Vec::new()
        }
        SlashCommand::Setup => {
            app.overlay = Overlay::setup();
            Vec::new()
        }
        SlashCommand::Clear => {
            if app.items.is_empty() {
                Vec::new()
            } else {
                app.overlay = Overlay::ConfirmClear;
                Vec::new()
            }
        }
        SlashCommand::Model { query } => {
            app.overlay = Overlay::models(&app.known_models, &app.header.model, &query);
            Vec::new()
        }
        SlashCommand::Endpoint { url } => {
            if url.is_empty() {
                app.push_warning(format!(
                    "current endpoint: {}; usage: /endpoint http://127.0.0.1:8317",
                    app.header.endpoint
                ));
                Vec::new()
            } else {
                app.phase = RunPhase::Setup;
                app.status = "Connecting model gateway".into();
                app.persist_next_configuration = true;
                vec![Effect::SetEndpoint(url)]
            }
        }
        SlashCommand::Usage => {
            app.overlay = Overlay::Usage;
            Vec::new()
        }
        SlashCommand::Tools => {
            let count = app
                .items
                .iter()
                .filter(|item| matches!(item, TranscriptItem::Tool { .. }))
                .count();
            app.overlay = Overlay::Tools(ToolOverlay { selected: 0, count });
            Vec::new()
        }
        SlashCommand::Copy => {
            if let Some(text) = app.last_assistant().map(str::to_string) {
                app.copy_status = Some("copied last assistant reply".into());
                vec![Effect::Copy(text)]
            } else {
                app.copy_status = Some("nothing to copy".into());
                Vec::new()
            }
        }
        SlashCommand::Exit => {
            // Repository preparation is background work, not an agent turn.
            // Exiting during startup must close immediately instead of waiting
            // for a TurnFinished event that can never arrive.
            if matches!(app.phase, RunPhase::Busy | RunPhase::Cancelling) {
                app.request_exit = true;
                app.phase = RunPhase::Cancelling;
                app.status = "finishing current turn".into();
                vec![Effect::Cancel]
            } else {
                app.request_exit = true;
                vec![Effect::Quit]
            }
        }
        SlashCommand::Sessions { query } => {
            app.overlay = Overlay::sessions(&app.known_sessions, &query);
            Vec::new()
        }
        SlashCommand::Name { title } => {
            if title.is_empty() {
                app.push_warning("usage: /name TITLE");
                Vec::new()
            } else {
                app.session_title = title.clone();
                vec![Effect::PersistTitle(title)]
            }
        }
        SlashCommand::Compact => vec![Effect::Compact],
        SlashCommand::Unknown(name) => {
            app.push_warning(format!("unknown command {name}; try /help"));
            Vec::new()
        }
    }
}

fn convert_queued_to_user(app: &mut App, text: &str) {
    if let Some(item) = app
        .items
        .iter_mut()
        .find(|item| matches!(item, TranscriptItem::Queued { text: existing } if existing == text))
    {
        *item = TranscriptItem::User {
            text: text.to_string(),
        };
        return;
    }
    app.push_user(text.to_string());
}

fn scroll(app: &mut App, up: bool, amount: u16) {
    if up {
        app.scroll = app.scroll.saturating_sub(amount);
        app.follow_tail = false;
    } else {
        app.scroll = app.scroll.saturating_add(amount).min(app.max_scroll);
        app.follow_tail = app.scroll == app.max_scroll;
    }
}

fn page_amount(app: &App) -> u16 {
    app.viewport_height.max(1).saturating_sub(1)
}

fn follow_if_needed(app: &mut App) {
    if app.follow_tail {
        app.scroll = app.max_scroll;
    }
}

pub fn apply_effects(effects: &[Effect]) -> Vec<SessionCommand> {
    effects
        .iter()
        .filter_map(|effect| match effect {
            Effect::Submit(prompt) => Some(SessionCommand::Prompt(prompt.clone())),
            Effect::Cancel => Some(SessionCommand::Cancel),
            Effect::Quit => Some(SessionCommand::Quit),
            Effect::SetModel(model) => Some(SessionCommand::SetModel(model.clone())),
            Effect::SetEndpoint(url) => Some(SessionCommand::SetEndpoint(url.clone())),
            Effect::ResumeSession(id) => Some(SessionCommand::Resume(id.clone())),
            Effect::Compact => Some(SessionCommand::Compact),
            Effect::PersistTitle(_) | Effect::Copy(_) | Effect::SaveSettings(_) => None,
        })
        .collect()
}

pub fn mouse_scroll(kind: MouseEventKind) -> Option<Action> {
    match kind {
        MouseEventKind::ScrollUp => Some(Action::Scroll {
            up: true,
            amount: 3,
        }),
        MouseEventKind::ScrollDown => Some(Action::Scroll {
            up: false,
            amount: 3,
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_tui::session::SessionRecord;
    use crate::agent_tui::state::HeaderState;
    use crate::agent_tui::theme::Theme;

    fn app() -> App {
        let session =
            SessionRecord::new("sess".into(), "demo".into(), "model".into(), "run".into());
        App::new(
            HeaderState {
                repository: "repo".into(),
                branch: "main".into(),
                worktree: "wt".into(),
                model: "model".into(),
                endpoint: "http://127.0.0.1:8317".into(),
                sandbox: "off".into(),
            },
            Theme {
                color: false,
                ascii: true,
            },
            &session,
        )
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn type_text(app: &mut App, text: &str) {
        for ch in text.chars() {
            let _ = update(
                app,
                Action::Key(KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE)),
            );
        }
    }

    #[test]
    fn enter_submits_and_slash_help_opens_overlay() {
        let mut app = app();
        type_text(&mut app, "inspect parser");
        let effects = update(&mut app, Action::Key(key(KeyCode::Enter)));
        assert_eq!(effects, vec![Effect::Submit("inspect parser".into())]);
        assert_eq!(app.phase, RunPhase::Busy);
        type_text(&mut app, "/help");
        let _ = update(&mut app, Action::Key(key(KeyCode::Enter)));
        assert!(matches!(app.overlay, Overlay::Help));
    }

    #[test]
    fn busy_submit_queues_follow_up() {
        let mut app = app();
        app.phase = RunPhase::Busy;
        type_text(&mut app, "next please");
        let effects = update(&mut app, Action::Key(key(KeyCode::Enter)));
        assert!(effects.is_empty());
        assert_eq!(app.queued.len(), 1);
        assert!(matches!(
            app.items.last(),
            Some(TranscriptItem::Queued { .. })
        ));
        let effects = update(
            &mut app,
            Action::Worker(SessionEvent::Done {
                input_tokens: 1,
                output_tokens: 2,
                cache_read: 0,
                cache_write: 0,
                turns: 1,
                stop: "ready".into(),
                messages: Vec::new(),
            }),
        );
        assert_eq!(effects, vec![Effect::Submit("next please".into())]);
        assert!(app.queued.is_empty());
    }

    #[test]
    fn ctrl_c_cancels_then_force_quits() {
        let mut app = app();
        app.phase = RunPhase::Busy;
        let first = update(
            &mut app,
            Action::Key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
        );
        assert_eq!(first, vec![Effect::Cancel]);
        assert_eq!(app.phase, RunPhase::Cancelling);
        let second = update(
            &mut app,
            Action::Key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
        );
        assert_eq!(second, vec![Effect::Cancel, Effect::Quit]);
        assert!(app.force_exit);
    }

    #[test]
    fn setup_ctrl_c_cancels_owned_work_and_exits_immediately() {
        let mut app = app();
        app.phase = RunPhase::Setup;
        let effects = update(
            &mut app,
            Action::Key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
        );
        assert_eq!(effects, vec![Effect::Cancel, Effect::Quit]);
        assert!(app.force_exit);
        assert!(app.request_exit);
    }

    #[test]
    fn setup_progress_uses_reported_unit_then_ready_enters_idle() {
        let mut app = app();
        let effects = update(
            &mut app,
            Action::Worker(SessionEvent::SetupProgress {
                phase: "Generating embeddings (Metal GPU)".into(),
                detail: Some("crate::agent::run".into()),
                unit: "spans".into(),
                completed: 12,
                total: 48,
                rate_milli_per_second: Some(2_000),
                eta_seconds: Some(18),
                elapsed_seconds: 6,
            }),
        );
        assert!(effects.is_empty());
        assert_eq!(app.phase, RunPhase::Setup);
        assert_eq!(app.setup_unit, "spans");
        assert_eq!(app.setup_completed, 12);
        assert_eq!(app.setup_total, 48);

        let effects = update(&mut app, Action::Worker(SessionEvent::SetupReady));
        assert!(effects.is_empty());
        assert_eq!(app.phase, RunPhase::Idle);
        assert_eq!(app.status, "ready");
    }

    #[test]
    fn setup_ready_submits_an_initial_task_queued_behind_the_gate() {
        let mut app = app();
        app.phase = RunPhase::Setup;
        app.queued.push_back("inspect parser".into());
        app.items.push(TranscriptItem::Queued {
            text: "inspect parser".into(),
        });
        let effects = update(&mut app, Action::Worker(SessionEvent::SetupReady));
        assert_eq!(effects, vec![Effect::Submit("inspect parser".into())]);
        assert_eq!(app.phase, RunPhase::Busy);
        assert!(app.queued.is_empty());
        assert!(matches!(
            app.items.last(),
            Some(TranscriptItem::User { text }) if text == "inspect parser"
        ));
    }

    #[test]
    fn gateway_readiness_never_releases_prompts_before_repository_analysis() {
        let mut app = app();
        app.phase = RunPhase::Setup;
        app.gateway_ready = false;
        app.repository_ready = false;
        app.queued.push_back("inspect parser".into());
        app.items.push(TranscriptItem::Queued {
            text: "inspect parser".into(),
        });

        let effects = update(&mut app, Action::Worker(SessionEvent::SetupReady));
        assert!(effects.is_empty());
        assert_eq!(app.phase, RunPhase::Setup);
        assert_eq!(app.queued.len(), 1);
        assert!(app.status.contains("one-time code analysis"));

        let effects = update(&mut app, Action::Worker(SessionEvent::BackgroundReady));
        assert_eq!(effects, vec![Effect::Submit("inspect parser".into())]);
        assert_eq!(app.phase, RunPhase::Busy);
        assert!(app.queued.is_empty());
    }

    #[test]
    fn background_index_progress_never_takes_over_the_chat_phase() {
        let mut app = app();
        app.phase = RunPhase::Busy;
        let effects = update(
            &mut app,
            Action::Worker(SessionEvent::BackgroundProgress {
                phase: "Generating embeddings (Metal GPU)".into(),
                detail: Some("agent::run_interactive_session".into()),
                unit: "spans".into(),
                completed: 25,
                total: 100,
                rate_milli_per_second: Some(12_500),
                eta_seconds: Some(6),
            }),
        );
        assert!(effects.is_empty());
        assert_eq!(app.phase, RunPhase::Busy);
        let status = app.background_status.as_deref().unwrap_or_default();
        assert!(status.contains("25% · 25/100"));
        assert!(!status.contains("spans"));
        assert!(status.contains("agent::run_interactive_session"));

        let _ = update(&mut app, Action::Worker(SessionEvent::BackgroundReady));
        assert_eq!(app.phase, RunPhase::Busy);
        assert!(app.background_status.is_none());
    }

    #[test]
    fn idle_ctrl_c_quits() {
        let mut app = app();
        let effects = update(
            &mut app,
            Action::Key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
        );
        assert_eq!(effects, vec![Effect::Quit]);
    }

    #[test]
    fn exit_during_repository_setup_quits_immediately() {
        let mut app = app();
        app.phase = RunPhase::Setup;
        type_text(&mut app, "/exit");
        let effects = update(&mut app, Action::Key(key(KeyCode::Enter)));
        assert_eq!(effects, vec![Effect::Quit]);
    }

    #[test]
    fn startup_accepts_queued_input_and_gateway_configuration_accepts_a_raw_url() {
        let mut app = app();
        app.phase = RunPhase::Setup;
        let _ = update(&mut app, Action::Paste("queued while indexing".into()));
        assert_eq!(app.composer.text(), "queued while indexing");
        let effects = update(
            &mut app,
            Action::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
        );
        assert!(effects.is_empty());
        assert_eq!(
            app.queued.front().map(String::as_str),
            Some("queued while indexing")
        );

        let _ = update(
            &mut app,
            Action::Worker(SessionEvent::SetupBlocked("index failed".into())),
        );
        assert_eq!(app.phase, RunPhase::Blocked);
        type_text(&mut app, "ignored");
        assert!(app.composer.is_empty());

        let _ = update(
            &mut app,
            Action::Worker(SessionEvent::GatewayRequired(
                "No model gateway is connected.".into(),
            )),
        );
        assert_eq!(app.phase, RunPhase::Configuring);
        type_text(&mut app, "http://127.0.0.1:18318");
        let effects = update(&mut app, Action::Key(key(KeyCode::Enter)));
        assert_eq!(
            effects,
            vec![Effect::SetEndpoint("http://127.0.0.1:18318".into())]
        );
        assert_eq!(app.phase, RunPhase::Setup);

        let effects = update(
            &mut app,
            Action::Worker(SessionEvent::Configuration {
                endpoint: "http://127.0.0.1:8317".into(),
                model: "stale".into(),
                models: vec!["stale".into()],
            }),
        );
        assert!(effects.is_empty());
        assert_ne!(app.header.model, "stale");
        assert_eq!(
            app.pending_endpoint.as_deref(),
            Some("http://127.0.0.1:18318")
        );
        let _ = update(&mut app, Action::Worker(SessionEvent::SetupReady));
        assert_eq!(app.phase, RunPhase::Setup);

        let effects = update(
            &mut app,
            Action::Worker(SessionEvent::Configuration {
                endpoint: "http://127.0.0.1:18318".into(),
                model: "test".into(),
                models: vec!["test".into()],
            }),
        );
        assert!(matches!(effects.as_slice(), [Effect::SaveSettings(_)]));
        assert!(app.pending_endpoint.is_none());
    }

    #[test]
    fn setup_command_opens_complete_settings_menu() {
        let mut app = app();
        type_text(&mut app, "/setup");
        let effects = update(&mut app, Action::Key(key(KeyCode::Enter)));
        assert!(effects.is_empty());
        assert!(matches!(app.overlay, Overlay::Setup(_)));

        let _ = update(&mut app, Action::Key(key(KeyCode::Down)));
        let _ = update(&mut app, Action::Key(key(KeyCode::Down)));
        let _ = update(&mut app, Action::Key(key(KeyCode::Down)));
        let effects = update(&mut app, Action::Key(key(KeyCode::Enter)));
        assert!(app.settings.private_store);
        assert!(matches!(effects.as_slice(), [Effect::SaveSettings(_)]));

        app.overlay = Overlay::setup();
        let _ = update(&mut app, Action::Key(key(KeyCode::Enter)));
        assert_eq!(app.phase, RunPhase::Configuring);
        assert_eq!(app.composer.text(), "http://127.0.0.1:8317");
    }

    #[test]
    fn follow_tail_stops_on_manual_scroll_and_restores_on_end() {
        let mut app = app();
        app.max_scroll = 20;
        app.scroll = 20;
        let _ = update(
            &mut app,
            Action::Scroll {
                up: true,
                amount: 5,
            },
        );
        assert!(!app.follow_tail);
        assert_eq!(app.scroll, 15);
        let _ = update(&mut app, Action::Key(key(KeyCode::End)));
        assert!(app.follow_tail);
        assert_eq!(app.scroll, 20);
    }

    #[test]
    fn page_keys_use_viewport_height() {
        let mut app = app();
        app.viewport_height = 10;
        app.max_scroll = 40;
        app.scroll = 30;
        let _ = update(&mut app, Action::Key(key(KeyCode::PageUp)));
        assert_eq!(app.scroll, 21);
    }

    #[test]
    fn esc_closes_overlay_then_completions() {
        let mut app = app();
        app.overlay = Overlay::Help;
        let _ = update(&mut app, Action::Key(key(KeyCode::Esc)));
        assert_eq!(app.overlay, Overlay::None);
        type_text(&mut app, "/he");
        assert!(app.completion.is_some());
        let _ = update(&mut app, Action::Key(key(KeyCode::Esc)));
        assert!(app.completion.is_none());
    }

    #[test]
    fn worker_disconnect_and_error_paths() {
        let mut disconnected = app();
        disconnected.phase = RunPhase::Busy;
        let _ = update(&mut disconnected, Action::Disconnect);
        assert_eq!(disconnected.phase, RunPhase::Idle);
        assert!(disconnected.request_exit);
        let mut app = app();
        let _ = update(&mut app, Action::Worker(SessionEvent::Error("boom".into())));
        assert!(matches!(
            app.items.last(),
            Some(TranscriptItem::Error { .. })
        ));
    }

    #[test]
    fn clear_requires_confirmation() {
        let mut app = app();
        app.push_user("keep me".into());
        type_text(&mut app, "/clear");
        let _ = update(&mut app, Action::Key(key(KeyCode::Enter)));
        assert_eq!(app.overlay, Overlay::ConfirmClear);
        let _ = update(
            &mut app,
            Action::Key(KeyEvent::new(KeyCode::Char('y'), KeyModifiers::NONE)),
        );
        assert!(app.items.is_empty());
    }

    #[test]
    fn tiny_resize_shows_too_small_and_recovers() {
        let mut app = app();
        let _ = update(&mut app, Action::Resize { cols: 40, rows: 10 });
        assert!(matches!(app.overlay, Overlay::TooSmall { .. }));
        let _ = update(&mut app, Action::Resize { cols: 80, rows: 24 });
        assert_eq!(app.overlay, Overlay::None);
    }
    #[test]
    fn slash_commands_open_expected_overlays() {
        let mut app = app();
        type_text(&mut app, "/model");
        let _ = update(&mut app, Action::Key(key(KeyCode::Enter)));
        assert!(matches!(app.overlay, Overlay::Model(_)));
        let _ = update(&mut app, Action::Key(key(KeyCode::Esc)));
        type_text(&mut app, "/usage");
        let _ = update(&mut app, Action::Key(key(KeyCode::Enter)));
        assert_eq!(app.overlay, Overlay::Usage);
        let _ = update(&mut app, Action::Key(key(KeyCode::Esc)));
        type_text(&mut app, "/tools");
        let _ = update(&mut app, Action::Key(key(KeyCode::Enter)));
        assert!(matches!(app.overlay, Overlay::Tools(_)));
        let _ = update(&mut app, Action::Key(key(KeyCode::Esc)));
        type_text(&mut app, "/sessions");
        let _ = update(&mut app, Action::Key(key(KeyCode::Enter)));
        assert!(matches!(app.overlay, Overlay::Sessions(_)));
        let _ = update(&mut app, Action::Key(key(KeyCode::Esc)));
        type_text(&mut app, "/name Review");
        let effects = update(&mut app, Action::Key(key(KeyCode::Enter)));
        assert_eq!(effects, vec![Effect::PersistTitle("Review".into())]);
        type_text(&mut app, "/compact");
        let effects = update(&mut app, Action::Key(key(KeyCode::Enter)));
        assert_eq!(effects, vec![Effect::Compact]);
        type_text(&mut app, "/copy");
        let effects = update(&mut app, Action::Key(key(KeyCode::Enter)));
        assert!(effects.is_empty());
        app.append_assistant("copied body");
        type_text(&mut app, "/copy");
        let effects = update(&mut app, Action::Key(key(KeyCode::Enter)));
        assert_eq!(effects, vec![Effect::Copy("copied body".into())]);
        type_text(&mut app, "/exit");
        let effects = update(&mut app, Action::Key(key(KeyCode::Enter)));
        assert_eq!(effects, vec![Effect::Quit]);
    }

    #[test]
    fn session_picker_switches_ui_and_worker_together() {
        let mut app = app();
        let mut other = SessionRecord::new(
            "sess-other".into(),
            "demo".into(),
            "other-model".into(),
            "run-other".into(),
        );
        other.title = "Other session".into();
        other.usage.input_tokens = 17;
        other.turns = 3;
        other.stop = "end_turn".into();
        app.known_sessions = vec![other];
        app.overlay = Overlay::sessions(&app.known_sessions, "");

        let effects = update(&mut app, Action::Key(key(KeyCode::Enter)));

        assert_eq!(effects, vec![Effect::ResumeSession("sess-other".into())]);
        assert_eq!(app.session_id, "sess-other");
        assert_eq!(app.header.model, "other-model");
        assert_eq!(app.input_tokens, 17);
        assert_eq!(app.turns, 3);
        assert_eq!(app.status, "end_turn");
        assert_eq!(
            apply_effects(&effects),
            vec![SessionCommand::Resume("sess-other".into())]
        );
    }

    #[test]
    fn tab_cycles_slash_completions_without_inserting_tab() {
        let mut app = app();
        type_text(&mut app, "/m");
        assert!(app.completion.is_some());
        let _ = update(&mut app, Action::Key(key(KeyCode::Tab)));
        assert!(!app.composer.text().contains('\t'));
        let _ = update(&mut app, Action::Key(key(KeyCode::Enter)));
        assert!(app.composer.text().starts_with("/model"));
    }
}
