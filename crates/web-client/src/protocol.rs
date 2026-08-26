use serde::{Deserialize, Serialize};

pub const SCHEMA: &str = "greppy.web-runtime.v1";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Request {
    pub schema: String,
    pub request_id: String,
    pub run_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    pub deadline_ms: u64,
    pub operation: String,
    pub payload: serde_json::Value,
}

impl Request {
    pub fn new(
        run_id: impl Into<String>,
        operation: impl Into<String>,
        payload: serde_json::Value,
    ) -> Self {
        Self {
            schema: SCHEMA.to_owned(),
            request_id: new_request_id(),
            run_id: run_id.into(),
            session_id: None,
            deadline_ms: 30_000,
            operation: operation.into(),
            payload,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Response {
    pub schema: String,
    pub request_id: String,
    pub operation: String,
    pub status: String,
    pub result: Option<serde_json::Value>,
    pub artifacts: Vec<serde_json::Value>,
    pub metrics: Metrics,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorObject>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub handshake: Option<Handshake>,
}

impl Response {
    pub fn ok(request: &Request, result: serde_json::Value) -> Self {
        Self {
            schema: SCHEMA.to_owned(),
            request_id: request.request_id.clone(),
            operation: request.operation.clone(),
            status: "ok".to_owned(),
            result: Some(result),
            artifacts: Vec::new(),
            metrics: Metrics::default(),
            error: None,
            handshake: None,
        }
    }

    pub fn error(request: &Request, error: ErrorObject) -> Self {
        Self {
            schema: SCHEMA.to_owned(),
            request_id: request.request_id.clone(),
            operation: request.operation.clone(),
            status: "error".to_owned(),
            result: None,
            artifacts: Vec::new(),
            metrics: Metrics::default(),
            error: Some(error),
            handshake: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ErrorObject {
    pub code: String,
    pub message: String,
    pub operation_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    pub retryable: bool,
    pub next_action: String,
    pub exit_code: i32,
}

impl ErrorObject {
    pub fn new(
        code: &str,
        message: impl Into<String>,
        operation_id: impl Into<String>,
        exit_code: i32,
        next_action: impl Into<String>,
    ) -> Self {
        Self {
            code: code.to_owned(),
            message: message.into(),
            operation_id: operation_id.into(),
            session_id: None,
            retryable: false,
            next_action: next_action.into(),
            exit_code,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Metrics {
    pub wall_ms: u64,
    pub controller_cpu_ms: u64,
    pub content_cpu_ms: u64,
    pub peak_rss_bytes: u64,
    pub network_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Handshake {
    pub protocol_version: String,
    pub runtime_build_id: String,
    pub playwright_compatibility_version: String,
    pub servo_revision: String,
    pub v8_revision: String,
    pub platform: String,
    pub architecture: String,
    pub supported_capabilities: Vec<String>,
    pub compatibility_coverage_level: String,
    pub max_message_bytes: u64,
    pub max_artifact_bytes: u64,
}

pub fn new_request_id() -> String {
    format!("wrq_{}", random_token())
}

pub fn new_session_id() -> String {
    format!("wrs_{}", random_token())
}

fn random_token() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{:x}{:x}", std::process::id(), nanos)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_ids_use_required_prefix() {
        assert!(new_request_id().starts_with("wrq_"));
        assert!(new_session_id().starts_with("wrs_"));
    }
}
