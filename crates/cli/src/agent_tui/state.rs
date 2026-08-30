//! Application state for the interactive agent TUI.

use std::collections::VecDeque;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;

use super::commands::completions;
use super::composer::Composer;
use super::overlay::{Overlay, ToolOverlay};
use super::redaction::sanitize_terminal_text;
use super::session::{PersistedMessage, SessionRecord};
use super::settings::AgentSettings;
use super::theme::Theme;

pub const MIN_COLS: u16 = 60;
pub const MIN_ROWS: u16 = 18;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunPhase {
    Setup,
    Configuring,
    Blocked,
    Idle,
    Busy,
    Cancelling,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolPhase {
    Running,
    Success,
    Failure,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TranscriptItem {
    User {
        text: String,
    },
    Assistant {
        text: String,
    },
    Thinking {
        text: String,
        streaming: bool,
    },
    Tool {
        id: String,
        summary: String,
        phase: ToolPhase,
        elapsed_ms: u64,
        preview: String,
        expanded: bool,
    },
    Warning {
        text: String,
    },
    Error {
        text: String,
    },
    Queued {
        text: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionMenu {
    pub items: Vec<String>,
    pub selected: usize,
}

#[derive(Debug, Clone)]
pub struct HeaderState {
    pub repository: String,
    pub branch: String,
    pub worktree: String,
    pub model: String,
    pub endpoint: String,
    pub sandbox: String,
}

#[derive(Debug, Clone)]
pub struct App {
    pub header: HeaderState,
    pub theme: Theme,
    pub items: Vec<TranscriptItem>,
    pub composer: Composer,
    pub overlay: Overlay,
    pub phase: RunPhase,
    pub status: String,
    pub setup_detail: Option<String>,
    pub setup_completed: usize,
    pub setup_total: usize,
    pub setup_unit: String,
    pub setup_rate_milli_per_second: Option<u64>,
    pub setup_eta_seconds: Option<u64>,
    pub setup_elapsed_seconds: u64,
    pub setup_history: Vec<String>,
    pub setup_error: Option<String>,
    pub background_status: Option<String>,
    pub gateway_ready: bool,
    pub repository_ready: bool,
    pub follow_tail: bool,
    pub scroll: u16,
    pub max_scroll: u16,
    pub viewport_height: u16,
    pub cols: u16,
    pub rows: u16,
    pub spinner_tick: usize,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    pub turns: u64,
    pub submitted_prompts: u64,
    pub queued: VecDeque<String>,
    pub thinking_expanded: bool,
    pub completion: Option<CompletionMenu>,
    pub session_id: String,
    pub session_title: String,
    pub persist_warning: Option<String>,
    pub last_ctrl_c: Option<Instant>,
    pub force_exit: bool,
    pub request_exit: bool,
    pub known_models: Vec<String>,
    pub known_sessions: Vec<SessionRecord>,
    pub copy_status: Option<String>,
    pub cancel: Arc<AtomicBool>,
    pub settings: AgentSettings,
    pub persist_next_configuration: bool,
    pub pending_endpoint: Option<String>,
}

impl App {
    pub fn new(header: HeaderState, theme: Theme, session: &SessionRecord) -> Self {
        let mut app = Self {
            header,
            theme,
            items: Vec::new(),
            composer: Composer::new(),
            overlay: Overlay::None,
            phase: RunPhase::Idle,
            status: "ready".into(),
            setup_detail: None,
            setup_completed: 0,
            setup_total: 0,
            setup_unit: "items".into(),
            setup_rate_milli_per_second: None,
            setup_eta_seconds: None,
            setup_elapsed_seconds: 0,
            setup_history: Vec::new(),
            setup_error: None,
            background_status: None,
            gateway_ready: true,
            repository_ready: true,
            follow_tail: true,
            scroll: 0,
            max_scroll: 0,
            viewport_height: 10,
            cols: 80,
            rows: 24,
            spinner_tick: 0,
            input_tokens: session.usage.input_tokens,
            output_tokens: session.usage.output_tokens,
            cache_read: session.usage.cache_read_input_tokens,
            cache_write: session.usage.cache_creation_input_tokens,
            turns: session.turns,
            submitted_prompts: 0,
            queued: VecDeque::new(),
            thinking_expanded: false,
            completion: None,
            session_id: session.id.clone(),
            session_title: session.title.clone(),
            persist_warning: None,
            last_ctrl_c: None,
            force_exit: false,
            request_exit: false,
            known_models: Vec::new(),
            known_sessions: Vec::new(),
            copy_status: None,
            cancel: Arc::new(AtomicBool::new(false)),
            settings: AgentSettings::default(),
            persist_next_configuration: false,
            pending_endpoint: None,
        };
        app.load_messages(&session.messages);
        if session.recovered {
            app.push_warning("session file had a truncated tail; restored the valid prefix");
        }
        app
    }

    pub fn load_messages(&mut self, messages: &[PersistedMessage]) {
        self.items.clear();
        for message in messages {
            self.ingest_persisted(message);
        }
    }

    pub fn ingest_persisted(&mut self, message: &PersistedMessage) {
        for part in &message.parts {
            match part.kind.as_str() {
                "thinking" => self.items.push(TranscriptItem::Thinking {
                    text: sanitize_terminal_text(&part.text).into_owned(),
                    streaming: false,
                }),
                "tool_call" => self.items.push(TranscriptItem::Tool {
                    id: sanitize_terminal_text(&part.id).into_owned(),
                    summary: sanitize_terminal_text(&format!("{} {}", part.name, part.text))
                        .into_owned(),
                    phase: ToolPhase::Success,
                    elapsed_ms: 0,
                    preview: String::new(),
                    expanded: false,
                }),
                "image" => self.items.push(TranscriptItem::User {
                    text: sanitize_terminal_text(&part.text).into_owned(),
                }),
                "tool_result" => {
                    let id = sanitize_terminal_text(&part.id).into_owned();
                    if let Some(TranscriptItem::Tool {
                        phase, preview, ..
                    }) = self.items.iter_mut().rev().find(|item| {
                        matches!(item, TranscriptItem::Tool { id: existing, .. } if existing == &id)
                    }) {
                        *phase = if part.is_error {
                            ToolPhase::Failure
                        } else {
                            ToolPhase::Success
                        };
                        *preview = bound_preview(&part.text);
                    }
                }
                _ if message.role == "assistant" => {
                    self.items.push(TranscriptItem::Assistant {
                        text: sanitize_terminal_text(&part.text).into_owned(),
                    });
                }
                _ => self.items.push(TranscriptItem::User {
                    text: sanitize_terminal_text(&part.text).into_owned(),
                }),
            }
        }
    }

    pub fn too_small(&self) -> bool {
        self.cols < MIN_COLS || self.rows < MIN_ROWS
    }

    pub fn busy(&self) -> bool {
        matches!(
            self.phase,
            RunPhase::Setup | RunPhase::Blocked | RunPhase::Busy | RunPhase::Cancelling
        )
    }

    pub fn push_user(&mut self, text: String) {
        self.items.push(TranscriptItem::User {
            text: sanitize_terminal_text(&text).into_owned(),
        });
        self.follow_tail = true;
    }

    pub fn push_warning(&mut self, text: impl Into<String>) {
        let text = text.into();
        self.items.push(TranscriptItem::Warning {
            text: sanitize_terminal_text(&text).into_owned(),
        });
    }

    pub fn push_error(&mut self, text: impl Into<String>) {
        let text = text.into();
        self.items.push(TranscriptItem::Error {
            text: sanitize_terminal_text(&text).into_owned(),
        });
        self.phase = RunPhase::Idle;
        self.status = "error".into();
    }

    pub fn append_assistant(&mut self, delta: &str) {
        let delta = sanitize_terminal_text(delta);
        if let Some(TranscriptItem::Assistant { text }) = self.items.last_mut() {
            text.push_str(delta.as_ref());
            return;
        }
        self.items.push(TranscriptItem::Assistant {
            text: delta.into_owned(),
        });
    }

    pub fn append_thinking(&mut self, delta: &str) {
        let delta = sanitize_terminal_text(delta);
        if let Some(TranscriptItem::Thinking { text, streaming }) = self.items.last_mut() {
            text.push_str(delta.as_ref());
            *streaming = true;
            return;
        }
        self.items.push(TranscriptItem::Thinking {
            text: delta.into_owned(),
            streaming: true,
        });
        self.status = "thinking".into();
    }

    pub fn close_streaming(&mut self) {
        if let Some(TranscriptItem::Thinking { streaming, .. }) = self.items.last_mut() {
            *streaming = false;
        }
    }

    pub fn start_tool(&mut self, id: String, summary: String) {
        self.close_streaming();
        self.items.push(TranscriptItem::Tool {
            id: sanitize_terminal_text(&id).into_owned(),
            summary: sanitize_terminal_text(&summary).into_owned(),
            phase: ToolPhase::Running,
            elapsed_ms: 0,
            preview: String::new(),
            expanded: false,
        });
        self.status = "running tool".into();
    }

    pub fn finish_tool(&mut self, id: &str, failed: bool, elapsed_ms: u64, preview: String) {
        let id = sanitize_terminal_text(id).into_owned();
        if let Some(TranscriptItem::Tool {
            phase,
            elapsed_ms: elapsed,
            preview: slot,
            ..
        }) = self.items.iter_mut().rev().find(
            |item| matches!(item, TranscriptItem::Tool { id: existing, .. } if existing == &id),
        ) {
            *phase = if failed {
                ToolPhase::Failure
            } else {
                ToolPhase::Success
            };
            *elapsed = elapsed_ms;
            *slot = bound_preview(&preview);
        }
        self.status = if failed { "tool failed" } else { "working" }.into();
    }

    pub fn last_assistant(&self) -> Option<&str> {
        self.items.iter().rev().find_map(|item| match item {
            TranscriptItem::Assistant { text } => Some(text.as_str()),
            _ => None,
        })
    }

    pub fn toggle_selected_tool(&mut self) {
        if let Overlay::Tools(ToolOverlay { selected, .. }) = self.overlay {
            let tools: Vec<usize> = self
                .items
                .iter()
                .enumerate()
                .filter(|(_, item)| matches!(item, TranscriptItem::Tool { .. }))
                .map(|(idx, _)| idx)
                .collect();
            if let Some(item_idx) = tools.get(selected) {
                if let TranscriptItem::Tool { expanded, .. } = &mut self.items[*item_idx] {
                    *expanded = !*expanded;
                }
            }
        } else if let Some(TranscriptItem::Tool { expanded, .. }) = self
            .items
            .iter_mut()
            .rev()
            .find(|item| matches!(item, TranscriptItem::Tool { .. }))
        {
            *expanded = !*expanded;
        }
    }

    pub fn refresh_completions(&mut self) {
        let text = self.composer.text();
        if let Some(rest) = text.strip_prefix('/') {
            if !rest.contains(char::is_whitespace) {
                let items: Vec<String> = completions(text)
                    .into_iter()
                    .map(|spec| format!("{}  {}", spec.name, spec.summary))
                    .collect();
                if items.is_empty() {
                    self.completion = None;
                } else {
                    let selected = self
                        .completion
                        .as_ref()
                        .map(|menu| menu.selected.min(items.len() - 1))
                        .unwrap_or(0);
                    self.completion = Some(CompletionMenu { items, selected });
                }
                return;
            }
        }
        self.completion = None;
    }

    pub fn apply_completion(&mut self) {
        let Some(menu) = self.completion.take() else {
            return;
        };
        if let Some(item) = menu.items.get(menu.selected) {
            let name = item.split_whitespace().next().unwrap_or(item);
            self.composer.set_text(format!("{name} "));
        }
    }

    pub fn usage_lines(&self) -> Vec<String> {
        vec![
            format!("model          {}", self.header.model),
            format!("session        {}  {}", self.session_id, self.session_title),
            format!("input tokens   {}", self.input_tokens),
            format!("output tokens  {}", self.output_tokens),
            format!("cache read     {}", self.cache_read),
            format!("cache write    {}", self.cache_write),
            format!("turns          {}", self.turns),
            format!("queued         {}", self.queued.len()),
            format!("stop           {}", self.status),
        ]
    }
}

pub fn bound_preview(text: &str) -> String {
    const MAX: usize = 400;
    let sanitized = sanitize_terminal_text(text);
    let trimmed = sanitized.trim();
    if trimmed.chars().count() <= MAX {
        return trimmed.to_string();
    }
    let mut out: String = trimmed.chars().take(MAX.saturating_sub(1)).collect();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use greppy_agent::Usage;

    fn app() -> App {
        let session =
            SessionRecord::new("sess".into(), "demo".into(), "model".into(), "run".into());
        App::new(
            HeaderState {
                repository: "repo".into(),
                branch: "main".into(),
                worktree: "worktree".into(),
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

    #[test]
    fn streaming_text_appends_without_duplicate_items() {
        let mut app = app();
        app.append_assistant("hello");
        app.append_assistant(" world");
        assert_eq!(app.items.len(), 1);
        assert_eq!(app.last_assistant(), Some("hello world"));
    }

    #[test]
    fn usage_accumulates_on_session_restore() {
        let mut session =
            SessionRecord::new("sess".into(), "demo".into(), "model".into(), "run".into());
        session.usage = Usage {
            input_tokens: 9,
            output_tokens: 4,
            cache_read_input_tokens: 1,
            cache_creation_input_tokens: 2,
        };
        session.turns = 3;
        let app = App::new(
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
        );
        assert_eq!(app.input_tokens, 9);
        assert_eq!(app.turns, 3);
    }
}
