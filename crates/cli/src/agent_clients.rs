//! Thin CLI clients over a live hosted session's control socket.

use std::io::{self, Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use greppy_core::error::Result;
use serde_json::{json, Value};

#[cfg(unix)]
use crate::agent_control::{is_live, not_live_message, ControlClient};
use crate::agent_sessions::{
    print_session_tail, render_tail_line, resolve_from_root, ListedSession,
};
use crate::agent_tui::sanitize_terminal_text;

const EVENT_POLL: Duration = Duration::from_millis(200);

static STREAM_STOP: AtomicBool = AtomicBool::new(false);

pub(crate) fn status(id: &str, json: bool, root: Option<&str>) -> Result<i32> {
    #[cfg(not(unix))]
    {
        let _ = (id, json, root);
        return Ok(unsupported());
    }
    #[cfg(unix)]
    {
        let (mut client, session) = match open_live(id, root) {
            Ok(opened) => opened,
            Err(code) => return Ok(code),
        };
        match call(&mut client, "session/describe", json!({})) {
            Ok(description) => {
                if json {
                    println!("{description}");
                } else {
                    print_status(&description, &session);
                }
                Ok(0)
            }
            Err(code) => Ok(code),
        }
    }
}

pub(crate) fn send(
    id: &str,
    text: &str,
    wait: bool,
    json: bool,
    source: &str,
    root: Option<&str>,
) -> Result<i32> {
    #[cfg(not(unix))]
    {
        let _ = (id, text, wait, json, source, root);
        return Ok(unsupported());
    }
    #[cfg(unix)]
    {
        let prompt = match read_prompt(text) {
            Ok(prompt) => prompt,
            Err(code) => return Ok(code),
        };
        if prompt.trim().is_empty() {
            eprintln!("empty prompt");
            return Ok(2);
        }
        let (mut client, _) = match open_live(id, root) {
            Ok(opened) => opened,
            Err(code) => return Ok(code),
        };
        if wait {
            if let Err(code) = subscribe(&mut client) {
                return Ok(code);
            }
        }
        let result = match call(
            &mut client,
            "turn/start",
            json!({"text": prompt, "source": source}),
        ) {
            Ok(result) => result,
            Err(code) => return Ok(code),
        };
        if !wait {
            if json {
                println!("{result}");
            } else {
                let prompt_id = result
                    .get("prompt_id")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let position = result.get("position").and_then(Value::as_u64).unwrap_or(0);
                println!("queued {prompt_id} position {position}");
            }
            return Ok(0);
        }
        let prompt_id = result
            .get("prompt_id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        stream_until_turn(&mut client, &prompt_id, json)
    }
}

pub(crate) fn interrupt(id: &str, json: bool, root: Option<&str>) -> Result<i32> {
    #[cfg(not(unix))]
    {
        let _ = (id, json, root);
        return Ok(unsupported());
    }
    #[cfg(unix)]
    {
        let (mut client, _) = match open_live(id, root) {
            Ok(opened) => opened,
            Err(code) => return Ok(code),
        };
        match call(&mut client, "turn/interrupt", json!({})) {
            Ok(result) => {
                if json {
                    println!("{result}");
                } else {
                    println!("interrupt requested");
                }
                Ok(0)
            }
            Err(code) => Ok(code),
        }
    }
}

pub(crate) fn quit(id: &str, json: bool, root: Option<&str>) -> Result<i32> {
    #[cfg(not(unix))]
    {
        let _ = (id, json, root);
        return Ok(unsupported());
    }
    #[cfg(unix)]
    {
        let (mut client, _) = match open_live(id, root) {
            Ok(opened) => opened,
            Err(code) => return Ok(code),
        };
        match call(&mut client, "session/quit", json!({})) {
            Ok(result) => {
                if json {
                    println!("{result}");
                } else {
                    println!("quit requested");
                }
                Ok(0)
            }
            Err(code) => Ok(code),
        }
    }
}

#[allow(dead_code)]
pub(crate) fn attach(id: &str, json: bool, since_start: bool, root: Option<&str>) -> Result<i32> {
    #[cfg(not(unix))]
    {
        let _ = (id, json, since_start, root);
        return Ok(unsupported());
    }
    #[cfg(unix)]
    {
        let (mut client, session) = match open_live(id, root) {
            Ok(opened) => opened,
            Err(code) => return Ok(code),
        };
        if let Err(code) = subscribe(&mut client) {
            return Ok(code);
        }
        if since_start {
            print_session_tail(&session, json, 40);
        }
        stream_events(&mut client, json, stream_interrupt_exit_code(), |_| None)
    }
}

fn stream_interrupt_exit_code() -> i32 {
    130
}

fn read_prompt(text: &str) -> std::result::Result<String, i32> {
    if text != "-" {
        return Ok(text.to_string());
    }
    let mut prompt = String::new();
    match io::stdin().read_to_string(&mut prompt) {
        Ok(_) => Ok(prompt),
        Err(error) => {
            eprintln!("{error}");
            Err(3)
        }
    }
}

fn print_status(description: &Value, session: &ListedSession) {
    println!(
        "id: {}",
        sanitize_human(
            description
                .get("session_id")
                .and_then(Value::as_str)
                .unwrap_or("")
        )
    );
    println!(
        "phase: {}",
        sanitize_human(
            description
                .get("phase")
                .and_then(Value::as_str)
                .unwrap_or("")
        )
    );
    println!(
        "model: {}",
        sanitize_human(
            description
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or("")
        )
    );
    println!(
        "turns: {}",
        description
            .get("turns")
            .and_then(Value::as_u64)
            .unwrap_or(0)
    );
    println!(
        "pending: {}",
        description
            .get("pending")
            .and_then(Value::as_u64)
            .unwrap_or(0)
    );
    println!(
        "worktree: {}",
        sanitize_human(&json_path(description.get("worktree")))
    );
    if !session.record.proposal_ref.is_empty() {
        println!(
            "proposal_ref: {}",
            sanitize_human(&session.record.proposal_ref)
        );
    }
    println!(
        "socket: {}",
        sanitize_human(&json_path(description.get("socket")))
    );
    println!(
        "jsonl: {}",
        sanitize_human(&json_path(description.get("jsonl")))
    );
}

fn json_path(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(path)) => path.clone(),
        Some(other) => other.to_string(),
        None => String::new(),
    }
}

fn sanitize_human(input: &str) -> String {
    sanitize_terminal_text(input).into_owned()
}

fn emit_event(event: &Value, json: bool) {
    if json {
        println!("{event}");
    } else if let Some(rendered) = render_control_event(event) {
        println!("{rendered}");
    }
    let _ = io::stdout().flush();
}

fn render_control_event(event: &Value) -> Option<String> {
    render_tail_line(&map_control_event(event)?)
}

fn map_control_event(event: &Value) -> Option<Value> {
    match event.get("type").and_then(Value::as_str)? {
        "turn_start" => Some(json!({
            "type": "turn",
            "event": "start",
            "source": event.get("source").cloned().unwrap_or(Value::Null),
            "prompt": event.get("text").cloned().unwrap_or(Value::Null),
        })),
        "turn_complete" => Some(json!({
            "type": "turn",
            "event": "done",
            "stop": event.get("stop").cloned().unwrap_or(Value::Null),
            "turns": event.get("turns").cloned().unwrap_or(Value::Null),
            "usage": event.get("usage").cloned().unwrap_or(Value::Null),
        })),
        "error" => Some(json!({
            "type": "turn",
            "event": "error",
            "message": event.get("message").cloned().unwrap_or(Value::Null),
        })),
        "tool_start" => Some(json!({
            "type": "tool",
            "event": "start",
            "name": event.get("name").cloned().unwrap_or(Value::Null),
            "summary": event.get("summary").cloned().unwrap_or(Value::Null),
        })),
        "tool_finish" => Some(json!({
            "type": "tool",
            "event": "finish",
            "elapsed_ms": event.get("elapsed_ms").cloned().unwrap_or(Value::Null),
            "failed": event.get("failed").cloned().unwrap_or(Value::Null),
            "preview": event.get("preview").cloned().unwrap_or(Value::Null),
        })),
        "text" => Some(json!({
            "type": "message",
            "role": "assistant",
            "parts": [{
                "kind": "text",
                "text": event.get("text").and_then(Value::as_str).unwrap_or(""),
            }],
        })),
        _ => None,
    }
}

#[cfg(not(unix))]
fn unsupported() -> i32 {
    eprintln!("greppy agent: Unix domain sockets are unsupported on this platform");
    3
}

#[cfg(unix)]
fn open_live(
    id: &str,
    root: Option<&str>,
) -> std::result::Result<(ControlClient, ListedSession), i32> {
    let session = resolve_from_root(root, id)?;
    let socket = session.path.with_extension("sock");
    if !is_live(&socket) {
        eprintln!("{}", not_live_message(&session.record.id));
        return Err(3);
    }
    match ControlClient::connect(&socket) {
        Ok(client) => Ok((client, session)),
        Err(error) => {
            eprintln!("{error}");
            Err(3)
        }
    }
}

#[cfg(unix)]
fn call(
    client: &mut ControlClient,
    method: &str,
    params: Value,
) -> std::result::Result<Value, i32> {
    match client.call(method, params) {
        Ok(result) => Ok(result),
        Err(error) => {
            eprintln!("{}", error.message);
            Err(3)
        }
    }
}

#[cfg(unix)]
fn subscribe(client: &mut ControlClient) -> std::result::Result<(), i32> {
    client.subscribe().map_err(|error| {
        eprintln!("{}", error.message);
        3
    })
}

#[cfg(unix)]
fn stream_until_turn(client: &mut ControlClient, prompt_id: &str, json: bool) -> Result<i32> {
    let mut saw_start = false;
    stream_events(client, json, stream_interrupt_exit_code(), |event| {
        let event_type = event.get("type").and_then(Value::as_str);
        if !saw_start {
            if event_type == Some("turn_start")
                && event.get("prompt_id").and_then(Value::as_str) == Some(prompt_id)
            {
                saw_start = true;
            }
            return None;
        }
        match event_type {
            Some("turn_complete") => Some(0),
            Some("error") => Some(3),
            _ => None,
        }
    })
}

#[cfg(unix)]
fn stream_events<F>(
    client: &mut ControlClient,
    json: bool,
    interrupt_code: i32,
    mut on_event: F,
) -> Result<i32>
where
    F: FnMut(&Value) -> Option<i32>,
{
    install_stream_stop();
    loop {
        if STREAM_STOP.load(Ordering::Relaxed) {
            return Ok(interrupt_code);
        }
        match client.next_event(EVENT_POLL) {
            Ok(Some(event)) => {
                emit_event(&event, json);
                if let Some(code) = on_event(&event) {
                    return Ok(code);
                }
            }
            Ok(None) => {}
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => {
                if STREAM_STOP.load(Ordering::Relaxed) {
                    return Ok(interrupt_code);
                }
                eprintln!("{error}");
                return Ok(3);
            }
        }
    }
}

fn install_stream_stop() {
    STREAM_STOP.store(false, Ordering::Relaxed);
    #[cfg(unix)]
    {
        let handler = stream_sigint as *const () as libc::sighandler_t;
        unsafe {
            libc::signal(libc::SIGINT, handler);
        }
    }
}

#[cfg(unix)]
extern "C" fn stream_sigint(_: libc::c_int) {
    STREAM_STOP.store(true, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn injected() -> &'static str {
        "\u{1b}]52;c;x\u{07}safe\u{1b}[2J"
    }

    fn assert_no_esc(rendered: &str) {
        assert!(
            !rendered.as_bytes().contains(&0x1b),
            "escape leaked: {rendered:?}"
        );
        assert!(rendered.contains("safe"), "{rendered:?}");
    }

    #[test]
    fn ctrl_c_maps_to_exit_130() {
        assert_eq!(stream_interrupt_exit_code(), 130);
    }

    #[test]
    fn client_event_renderers_strip_terminal_control_sequences() {
        let evil = injected();
        assert_no_esc(
            &render_control_event(&json!({
                "type": "turn_start",
                "source": evil,
                "text": evil,
            }))
            .unwrap(),
        );
        assert_no_esc(
            &render_control_event(&json!({
                "type": "tool_start",
                "name": "greppy",
                "summary": evil,
            }))
            .unwrap(),
        );
        assert_no_esc(
            &render_control_event(&json!({
                "type": "text",
                "text": evil,
            }))
            .unwrap(),
        );
        assert_eq!(
            sanitize_human(evil).as_bytes().contains(&0x1b),
            false,
            "{}",
            sanitize_human(evil)
        );
    }
}
