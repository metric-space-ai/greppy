//! Newline-delimited JSON events for `greppy -p --json`.

use std::io::{self, Write};

use greppy_agent::Usage;
use serde_json::{json, Value};

use crate::agent_tui::redact_text;

#[derive(Debug, Clone)]
pub struct JsonSession {
    pub session_id: String,
    pub run_id: String,
    pub project: String,
    pub worktree: String,
    pub branch: String,
    pub model: String,
    pub endpoint: String,
    pub sandbox: String,
    pub resumed: bool,
}

#[derive(Debug, Clone)]
pub struct JsonResult {
    pub status: &'static str,
    pub exit_code: u8,
    pub session_id: String,
    pub run_id: String,
    pub stop: String,
    pub turns: u64,
    pub usage: Usage,
    pub proposal_ref: Option<String>,
    pub commit: Option<String>,
    pub stat: Option<String>,
    pub patch: Option<String>,
    pub applied: bool,
    pub apply_error: Option<String>,
}

#[derive(Debug, Default)]
pub struct JsonEmitter {
    session_emitted: bool,
    result_emitted: bool,
}

impl JsonEmitter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn session(&mut self, session: &JsonSession) {
        if self.session_emitted {
            return;
        }
        self.session_emitted = true;
        write_line(&session_event(session));
    }

    // Only the socket host emits these, and it is Unix-only.
    #[allow(dead_code)]
    pub fn serve_session(&mut self, session: &JsonSession, socket: &str) {
        if self.session_emitted {
            return;
        }
        self.session_emitted = true;
        let mut event = session_event(session);
        if let Some(object) = event.as_object_mut() {
            object.insert("socket".to_string(), Value::String(socket.to_string()));
            object.insert("mode".to_string(), Value::String("serve".to_string()));
        }
        write_line(&event);
    }

    // Only the socket host emits these, and it is Unix-only.
    #[allow(dead_code)]
    pub(crate) fn emit(&mut self, event: &Value) {
        write_line(event);
    }

    pub fn text(&mut self, text: &str) {
        write_line(&text_event(text));
    }

    pub fn tool_start(&mut self, id: &str, name: &str, summary: &str) {
        write_line(&tool_start_event(id, name, summary));
    }

    pub fn tool_finish(&mut self, id: &str, failed: bool, elapsed_ms: u64, preview: &str) {
        write_line(&tool_finish_event(id, failed, elapsed_ms, preview));
    }

    pub fn turn_complete(&mut self, stop: &str, usage: &Usage) {
        write_line(&turn_complete_event(stop, usage));
    }

    pub fn error(&mut self, message: &str) {
        write_line(&error_event(message));
    }

    pub fn result(&mut self, result: &JsonResult) {
        if self.result_emitted {
            return;
        }
        self.result_emitted = true;
        write_line(&json!({
            "type": "result",
            "status": result.status,
            "exit_code": result.exit_code,
            "session_id": result.session_id,
            "run_id": result.run_id,
            "stop": result.stop,
            "turns": result.turns,
            "usage": usage_object(&result.usage),
            "proposal_ref": result.proposal_ref,
            "commit": result.commit,
            "stat": result.stat,
            "patch": result.patch,
            "applied": result.applied,
            "apply_error": result.apply_error,
        }));
    }
}

pub(crate) fn session_event(session: &JsonSession) -> Value {
    json!({
        "type": "session",
        "session_id": session.session_id,
        "run_id": session.run_id,
        "project": session.project,
        "worktree": session.worktree,
        "branch": session.branch,
        "model": session.model,
        "endpoint": session.endpoint,
        "sandbox": session.sandbox,
        "resumed": session.resumed,
    })
}

pub(crate) fn text_event(text: &str) -> Value {
    json!({"type":"text","text":redact_text(text)})
}

pub(crate) fn tool_start_event(id: &str, name: &str, summary: &str) -> Value {
    json!({"type":"tool_start","id":id,"name":name,"summary":summary})
}

pub(crate) fn tool_finish_event(id: &str, failed: bool, elapsed_ms: u64, preview: &str) -> Value {
    json!({
        "type":"tool_finish",
        "id":id,
        "failed":failed,
        "elapsed_ms":elapsed_ms,
        "preview":clip_chars(&redact_text(preview), 400),
    })
}

pub(crate) fn turn_complete_event(stop: &str, usage: &Usage) -> Value {
    json!({"type":"turn_complete","stop":stop,"usage":usage_object(usage)})
}

pub(crate) fn error_event(message: &str) -> Value {
    json!({"type":"error","message":message})
}

pub(crate) fn turn_start_event(prompt_id: &str, source: &str, text: &str) -> Value {
    json!({"type":"turn_start","prompt_id":prompt_id,"source":source,"text":redact_text(text)})
}

pub(crate) fn phase_event(phase: &str) -> Value {
    json!({"type":"phase","phase":phase})
}

pub fn emit_error_result(
    emitter: &mut JsonEmitter,
    session: &JsonSession,
    code: u8,
    message: &str,
) {
    emitter.session(session);
    if !message.is_empty() {
        emitter.error(message);
    }
    emitter.result(&JsonResult {
        status: if code == 130 { "cancelled" } else { "error" },
        exit_code: code,
        session_id: session.session_id.clone(),
        run_id: session.run_id.clone(),
        stop: String::new(),
        turns: 0,
        usage: Usage::default(),
        proposal_ref: None,
        commit: None,
        stat: None,
        patch: None,
        applied: false,
        apply_error: None,
    });
}

pub fn emit_error_result_opt(
    emitter: Option<&mut JsonEmitter>,
    session: &JsonSession,
    code: u8,
    message: &str,
) -> u8 {
    if let Some(emitter) = emitter {
        emit_error_result(emitter, session, code, message);
    }
    code
}

fn write_line(value: &Value) {
    let mut stdout = io::stdout().lock();
    let _ = writeln!(stdout, "{value}");
    let _ = stdout.flush();
}

pub(crate) fn usage_object(usage: &Usage) -> Value {
    json!({
        "input": usage.input_tokens,
        "output": usage.output_tokens,
        "cache_read": usage.cache_read_input_tokens,
        "cache_write": usage.cache_creation_input_tokens,
    })
}

fn clip_chars(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        text.to_string()
    } else {
        text.chars().take(max).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_and_result_vocabulary_is_exact() {
        use std::collections::HashSet;
        let session = json!({
            "type": "session",
            "session_id": "sess-1",
            "run_id": "run-1",
            "project": "demo",
            "worktree": "/tmp/wt",
            "branch": "main",
            "model": "m",
            "endpoint": "http://127.0.0.1:8317",
            "sandbox": "confined",
            "resumed": false,
        });
        let keys: HashSet<_> = session.as_object().unwrap().keys().cloned().collect();
        let expected: HashSet<_> = [
            "type",
            "session_id",
            "run_id",
            "project",
            "worktree",
            "branch",
            "model",
            "endpoint",
            "sandbox",
            "resumed",
        ]
        .into_iter()
        .map(str::to_string)
        .collect();
        assert_eq!(keys, expected);
        let result = json!({
            "type": "result",
            "status": "clean",
            "exit_code": 0u8,
            "session_id": "sess-1",
            "run_id": "run-1",
            "stop": "ready",
            "turns": 1u64,
            "usage": usage_object(&Usage::default()),
            "proposal_ref": Value::Null,
            "commit": Value::Null,
            "stat": Value::Null,
            "patch": Value::Null,
            "applied": false,
            "apply_error": Value::Null,
        });
        let keys: HashSet<_> = result.as_object().unwrap().keys().cloned().collect();
        let expected: HashSet<_> = [
            "type",
            "status",
            "exit_code",
            "session_id",
            "run_id",
            "stop",
            "turns",
            "usage",
            "proposal_ref",
            "commit",
            "stat",
            "patch",
            "applied",
            "apply_error",
        ]
        .into_iter()
        .map(str::to_string)
        .collect();
        assert_eq!(keys, expected);
    }
}
