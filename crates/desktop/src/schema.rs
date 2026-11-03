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
    if let Some(error) = value.get("error").filter(|v| !v.is_null()) {
        if let Some(message) = error.get("message").and_then(|v| v.as_str()) {
            lines.push(format!("error: {message}"));
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

pub fn exit_for_code(code: &str, retry_disposition: &str) -> i32 {
    match code {
        "invalid_args" => EXIT_USAGE,
        "platform_not_supported" | "desktop_disabled" => EXIT_UNAVAILABLE,
        _ if retry_disposition == "safe" => EXIT_FAILED,
        _ => EXIT_FAILED,
    }
}
