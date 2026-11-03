//! Stable `greppy.desktop.v1` envelope. Upstream payloads are reduced to these fields.

use serde_json::{json, Value};

use crate::command::Command;

pub const SCHEMA: &str = "greppy.desktop.v1";
pub const EXIT_OK: i32 = 0;
pub const EXIT_FAILED: i32 = 1;
pub const EXIT_USAGE: i32 = 2;
pub const EXIT_UNAVAILABLE: i32 = 40;

#[derive(Clone, Debug)]
pub struct Limits {
    pub json: bool,
    pub max_bytes: Option<usize>,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            json: true,
            max_bytes: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Outcome {
    pub exit_code: i32,
    pub body: String,
    pub document: Value,
}

#[derive(Clone, Debug)]
pub struct Retry {
    pub disposition: &'static str,
    pub after_ms: Option<u64>,
    pub requires_fresh_snapshot: bool,
}

impl Retry {
    pub fn not_applicable() -> Self {
        Self {
            disposition: "not_applicable",
            after_ms: None,
            requires_fresh_snapshot: false,
        }
    }

    fn to_json(&self) -> Value {
        json!({
            "disposition": self.disposition,
            "after_ms": self.after_ms,
            "requires_fresh_snapshot": self.requires_fresh_snapshot,
        })
    }
}

#[derive(Clone, Debug)]
pub struct Failure {
    pub code: String,
    pub upstream_code: Option<String>,
    pub message: String,
    pub suggestion: Option<String>,
    pub retry: Retry,
    pub exit_code: i32,
}

pub fn success(command: &str, data: Value, limits: &Limits) -> Outcome {
    let retry = Retry::not_applicable();
    document(DocumentParts {
        command,
        ok: true,
        untrusted: true,
        data: Some(data),
        error: None,
        retry: &retry,
        exit_code: EXIT_OK,
        limits,
    })
}

pub fn failure(command: &str, failure: &Failure, limits: &Limits) -> Outcome {
    let mut error = json!({
        "code": failure.code,
        "message": failure.message,
    });
    if let Some(suggestion) = &failure.suggestion {
        error["suggestion"] = json!(suggestion);
    }
    if let Some(upstream) = &failure.upstream_code {
        error["upstream_code"] = json!(upstream);
    }
    document(DocumentParts {
        command,
        ok: false,
        untrusted: true,
        data: None,
        error: Some(error),
        retry: &failure.retry,
        exit_code: failure.exit_code,
        limits,
    })
}

pub fn unavailable(command: &Command, message: impl Into<String>, limits: &Limits) -> Outcome {
    failure(
        command.name(),
        &Failure {
            code: "platform_not_supported".into(),
            upstream_code: None,
            message: message.into(),
            suggestion: Some("This operating system has no greppy desktop adapter.".into()),
            retry: Retry {
                disposition: "unsafe",
                after_ms: None,
                requires_fresh_snapshot: false,
            },
            exit_code: EXIT_UNAVAILABLE,
        },
        limits,
    )
}

pub fn agent_disabled(command: &Command, limits: &Limits) -> Outcome {
    failure(
        command.name(),
        &Failure {
            code: "desktop_disabled".into(),
            upstream_code: None,
            message: "desktop is off in the built-in agent".into(),
            suggestion: Some(
                "Run `greppy capabilities enable desktop` or start with --enable desktop.".into(),
            ),
            retry: Retry {
                disposition: "safe",
                after_ms: None,
                requires_fresh_snapshot: false,
            },
            exit_code: EXIT_UNAVAILABLE,
        },
        limits,
    )
}

struct DocumentParts<'a> {
    command: &'a str,
    ok: bool,
    untrusted: bool,
    data: Option<Value>,
    error: Option<Value>,
    retry: &'a Retry,
    exit_code: i32,
    limits: &'a Limits,
}

fn document(parts: DocumentParts<'_>) -> Outcome {
    let DocumentParts {
        command,
        ok,
        untrusted,
        data,
        error,
        retry,
        exit_code,
        limits,
    } = parts;
    let mut value = json!({
        "schema": SCHEMA,
        "ok": ok,
        "command": command,
        "untrusted": untrusted,
        "retry": retry.to_json(),
        "data": data,
        "error": error,
        "truncated": false,
    });
    if let Some(max) = limits.max_bytes {
        let encoded = value.to_string();
        if encoded.len() > max {
            value["truncated"] = json!(true);
            value["data"] = json!({
                "truncated": true,
                "total_bytes": encoded.len(),
            });
            if value.to_string().len() > max && max > 0 {
                // The envelope itself is the contract; callers asked for a cap
                // smaller than the status line. Keep the keys and mark it.
                value["data"] = json!({"truncated": true});
            }
        }
    }
    let body = if limits.json {
        value.to_string()
    } else {
        human(&value)
    };
    Outcome {
        exit_code,
        body,
        document: value,
    }
}

fn human(value: &Value) -> String {
    let command = value.get("command").and_then(|v| v.as_str()).unwrap_or("");
    let ok = value.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
    let mut lines = vec![
        format!("schema: {SCHEMA}"),
        format!("command: {command}"),
        format!("ok: {ok}"),
    ];
    if let Some(retry) = value
        .get("retry")
        .and_then(|v| v.get("disposition"))
        .and_then(|v| v.as_str())
    {
        lines.push(format!("retry: {retry}"));
    }
    if value
        .get("truncated")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        lines.push("truncated: true".to_owned());
    }
    if let Some(error) = value.get("error").filter(|v| !v.is_null()) {
        if let Some(code) = error.get("code").and_then(|v| v.as_str()) {
            lines.push(format!("error_code: {code}"));
        }
        if let Some(message) = error.get("message").and_then(|v| v.as_str()) {
            lines.push(format!("error: {message}"));
        }
        if let Some(suggestion) = error.get("suggestion").and_then(|v| v.as_str()) {
            lines.push(format!("suggestion: {suggestion}"));
        }
    }
    if let Some(data) = value.get("data").filter(|data| !data.is_null()) {
        let rendered = render_data(data, 0);
        if !rendered.is_empty() {
            lines.push(rendered.trim_end().to_owned());
        }
    }
    let body = lines.join("\n");
    if value
        .get("untrusted")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        format!("UNTRUSTED_DESKTOP_CONTENT\n{body}\nEND_UNTRUSTED_DESKTOP_CONTENT\n")
    } else {
        format!("{body}\n")
    }
}

/// Readable text for `data`. Arrays of objects become one line each so a
/// list of apps or windows is visible without `--json`.
fn render_data(value: &Value, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    match value {
        Value::Null => String::new(),
        Value::Object(map) => {
            let mut out = String::new();
            for (key, child) in map {
                match child {
                    Value::Null => {}
                    Value::Array(items) if items.iter().all(Value::is_object) => {
                        out.push_str(&format!("{pad}{key}:\n"));
                        if items.is_empty() {
                            out.push_str(&format!("{pad}  (none)\n"));
                        }
                        for item in items {
                            out.push_str(&format!("{pad}  {}\n", object_line(item)));
                        }
                    }
                    Value::Object(_) | Value::Array(_) => {
                        out.push_str(&format!("{pad}{key}:\n"));
                        out.push_str(&render_data(child, indent + 1));
                    }
                    other => {
                        out.push_str(&format!("{pad}{key}: {}\n", scalar(other)));
                    }
                }
            }
            out
        }
        Value::Array(items) => items
            .iter()
            .map(|item| {
                if item.is_object() {
                    format!("{pad}{}\n", object_line(item))
                } else {
                    format!("{pad}{}\n", scalar(item))
                }
            })
            .collect(),
        other => format!("{pad}{}\n", scalar(other)),
    }
}

fn object_line(value: &Value) -> String {
    let Some(map) = value.as_object() else {
        return scalar(value);
    };
    let mut parts = Vec::new();
    if let Some(name) = map.get("name").and_then(Value::as_str) {
        parts.push(name.to_owned());
    }
    for (key, child) in map {
        if key == "name" || child.is_null() {
            continue;
        }
        if child.is_object() || child.is_array() {
            parts.push(format!("{key}={}", child));
        } else {
            parts.push(format!("{key}={}", scalar(child)));
        }
    }
    if parts.is_empty() {
        value.to_string()
    } else {
        parts.join(" ")
    }
}

fn scalar(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

pub fn exit_for_code(code: &str, retry_disposition: &str) -> i32 {
    match code {
        "invalid_args" => EXIT_USAGE,
        "platform_not_supported" | "desktop_disabled" => EXIT_UNAVAILABLE,
        _ if retry_disposition == "safe" => EXIT_FAILED,
        _ => EXIT_FAILED,
    }
}

#[cfg(test)]
mod human_tests {
    use super::*;
    use serde_json::json;

    fn text(command: &str, data: Value) -> String {
        success(
            command,
            data,
            &Limits {
                json: false,
                max_bytes: None,
            },
        )
        .body
    }

    #[test]
    fn list_apps_text_includes_each_app() {
        let body = text(
            "list-apps",
            json!({"apps":[{"name":"Editor","pid":4242,"bundle_id":"app.editor"}]}),
        );
        assert!(body.contains("command: list-apps"), "{body}");
        assert!(body.contains("Editor"), "{body}");
        assert!(body.contains("pid=4242"), "{body}");
        assert!(body.contains("bundle_id=app.editor"), "{body}");
        assert!(body.contains("UNTRUSTED_DESKTOP_CONTENT"), "{body}");
    }

    #[test]
    fn list_windows_text_includes_each_window() {
        let body = text(
            "list-windows",
            json!({"windows":[{"id":"w1","title":"Notes","app":"Editor","pid":4242}]}),
        );
        assert!(body.contains("w1"), "{body}");
        assert!(body.contains("title=Notes"), "{body}");
        assert!(body.contains("app=Editor"), "{body}");
        assert!(body.contains("pid=4242"), "{body}");
    }

    #[test]
    fn snapshot_text_includes_tree_and_ids() {
        let body = text(
            "snapshot",
            json!({
                "app": "Editor",
                "snapshot_id": "snap-1",
                "ref_count": 2,
                "window": {"id": "w1", "title": "Notes"},
                "tree": {"role": "window", "name": "Notes", "children": [{"role": "button", "name": "Save"}]}
            }),
        );
        assert!(body.contains("snapshot_id: snap-1"), "{body}");
        assert!(body.contains("ref_count: 2"), "{body}");
        assert!(body.contains("role"), "{body}");
        assert!(body.contains("Save"), "{body}");
    }

    #[test]
    fn find_text_includes_matches() {
        let body = text(
            "find",
            json!({"matches":[{"name":"Save","role":"button","ref":"e1"}],"count":1}),
        );
        assert!(body.contains("count: 1"), "{body}");
        assert!(body.contains("Save"), "{body}");
        assert!(body.contains("role=button"), "{body}");
        assert!(body.contains("ref=e1"), "{body}");
    }
}
