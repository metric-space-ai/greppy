//! Strip secrets from anything the TUI might persist or display.

use std::borrow::Cow;

use serde_json::{Map, Value};

const SENSITIVE_KEYS: &[&str] = &[
    "authorization",
    "api_key",
    "apikey",
    "api-key",
    "access_token",
    "access-token",
    "secret",
    "password",
    "passwd",
    "token",
    "credential",
    "private_key",
    "private-key",
];

/// Strip terminal escape sequences and non-printing controls from text that
/// can reach the transcript. Newlines and tabs remain readable; CRLF becomes
/// LF, while a bare CR becomes a newline so progress updates do not rewrite
/// an already-rendered line.
pub fn sanitize_terminal_text(input: &str) -> Cow<'_, str> {
    let needs_work = input.bytes().any(|byte| {
        byte == 0x1b || byte == 0x7f || (byte < 0x20 && byte != b'\n' && byte != b'\t')
    });
    if !needs_work {
        return Cow::Borrowed(input);
    }

    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '\u{1b}' => match chars.peek() {
                // CSI: ESC '[' params/intermediates then a final byte @..~.
                Some('[') => {
                    chars.next();
                    while let Some(&candidate) = chars.peek() {
                        chars.next();
                        if ('\u{40}'..='\u{7e}').contains(&candidate) {
                            break;
                        }
                    }
                }
                // OSC, DCS, SOS, PM and APC carry arbitrary payloads. OSC
                // accepts BEL; the other string sequences require ST (ESC \).
                Some(']' | 'P' | 'X' | '^' | '_') => {
                    let accepts_bel = chars.peek() == Some(&']');
                    chars.next();
                    while let Some(candidate) = chars.next() {
                        if accepts_bel && candidate == '\u{07}' {
                            break;
                        }
                        if candidate == '\u{1b}' {
                            if chars.peek() == Some(&'\\') {
                                chars.next();
                            }
                            break;
                        }
                    }
                }
                // Two-character escapes (ESC 7 / ESC c) and a dangling ESC.
                Some(_) => {
                    chars.next();
                }
                None => {}
            },
            '\r' => {
                if chars.peek() != Some(&'\n') {
                    out.push('\n');
                }
            }
            '\n' | '\t' => out.push(ch),
            '\u{7f}' => {}
            control if control < '\u{20}' => {}
            printable => out.push(printable),
        }
    }
    Cow::Owned(out)
}

pub fn redact_text(input: &str) -> String {
    let mut out = input.to_string();
    for key in [
        "GREPPY_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "XAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ACCESS_KEY_ID",
        "GITHUB_TOKEN",
        "GH_TOKEN",
    ] {
        if let Ok(value) = std::env::var(key) {
            if !value.is_empty() {
                out = out.replace(&value, "[redacted]");
            }
        }
    }
    out = redact_bearer(&out);
    redact_assignment_values(&out)
}

pub fn redact_json(value: &Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(redact_map(map)),
        Value::Array(items) => Value::Array(items.iter().map(redact_json).collect()),
        Value::String(text) => Value::String(redact_text(text)),
        other => other.clone(),
    }
}

fn redact_map(map: &Map<String, Value>) -> Map<String, Value> {
    let mut out = Map::new();
    for (key, value) in map {
        if is_sensitive_key(key) {
            out.insert(key.clone(), Value::String("[redacted]".to_string()));
        } else {
            out.insert(key.clone(), redact_json(value));
        }
    }
    out
}

fn is_sensitive_key(key: &str) -> bool {
    let lowered = key.to_ascii_lowercase();
    SENSITIVE_KEYS
        .iter()
        .any(|needle| lowered == *needle || lowered.contains(needle))
}

fn redact_bearer(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while let Some(idx) = rest.find("Bearer ") {
        out.push_str(&rest[..idx]);
        out.push_str("Bearer [redacted]");
        let after = &rest[idx + "Bearer ".len()..];
        let skip = after
            .find(|ch: char| ch.is_whitespace() || ch == '"' || ch == '\'')
            .unwrap_or(after.len());
        rest = &after[skip..];
    }
    out.push_str(rest);
    out
}

fn redact_assignment_values(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for (index, line) in input.split_inclusive('\n').enumerate() {
        if index > 0 && !out.ends_with('\n') {
            out.push('\n');
        }
        if let Some((key, _)) = line.split_once('=') {
            let trimmed = key.trim();
            if is_sensitive_key(trimmed) || trimmed.ends_with("API_KEY") {
                out.push_str(trimmed);
                out.push_str("=[redacted]");
                if line.ends_with('\n') {
                    out.push('\n');
                }
                continue;
            }
        }
        out.push_str(line);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn redacts_bearer_and_json_secrets() {
        let text = redact_text("Authorization: Bearer sk-secret-value leftover");
        assert!(text.contains("Bearer [redacted]"));
        assert!(!text.contains("sk-secret-value"));
        let value = redact_json(&json!({"api_key": "abc", "args": ["who-calls"]}));
        assert_eq!(value["api_key"], "[redacted]");
        assert_eq!(value["args"][0], "who-calls");
    }

    #[test]
    fn sanitizes_clean_text_without_allocating() {
        let input = "plain output\nwith lines\tand tabs — 日本語";
        assert!(matches!(sanitize_terminal_text(input), Cow::Borrowed(text) if text == input));
    }

    #[test]
    fn sanitizes_csi_sequences() {
        assert_eq!(
            sanitize_terminal_text("\u{1b}[31mred\u{1b}[0m and \u{1b}[2Jcleared"),
            "red and cleared"
        );
    }

    #[test]
    fn sanitizes_osc_sequences_without_leaking_payloads() {
        let input = "\u{1b}]0;window title\u{07}before \u{1b}]8;;https://example.test\u{1b}\\after";
        assert_eq!(sanitize_terminal_text(input), "before after");
    }

    #[test]
    fn sanitizes_dcs_and_apc_sequences() {
        let input =
            "before \u{1b}Pq#0;2;0;0;0-payload\u{1b}\\middle \u{1b}_Gtmux-blob\u{1b}\\after";
        assert_eq!(sanitize_terminal_text(input), "before middle after");
        assert_eq!(sanitize_terminal_text("x\u{1b}Pdangling"), "x");
    }

    #[test]
    fn sanitizes_progress_controls_and_preserves_unicode() {
        assert_eq!(sanitize_terminal_text("a\r\nb"), "a\nb");
        assert_eq!(sanitize_terminal_text("10%\r50%\r100%"), "10%\n50%\n100%");
        assert_eq!(sanitize_terminal_text("a\u{08}b\u{07}c\u{7f}d"), "abcd");
        assert_eq!(sanitize_terminal_text("λ 日本語"), "λ 日本語");
    }
}
