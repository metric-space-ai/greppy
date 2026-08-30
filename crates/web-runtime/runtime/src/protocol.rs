use serde::{Deserialize, Serialize};
use std::io::{self, Read, Write};

pub const MAX_FRAME_BYTES: usize = 1024 * 1024;
pub const PROTOCOL_VERSION: u32 = 2;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ProtocolSchema {
    #[serde(rename = "greppy.web-runtime.worker.v1")]
    WorkerV1,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "PascalCase")]
pub enum WorkerKind {
    Controller,
    Content,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "PascalCase", deny_unknown_fields)]
pub enum Message {
    Hello {
        schema: ProtocolSchema,
        version: u32,
        worker: WorkerKind,
        capability: String,
    },
    Ready {
        schema: ProtocolSchema,
        version: u32,
        worker: WorkerKind,
    },
    Shutdown {
        schema: ProtocolSchema,
        version: u32,
    },
    ShutdownAck {
        schema: ProtocolSchema,
        version: u32,
        worker: WorkerKind,
    },
    RunScript {
        schema: ProtocolSchema,
        version: u32,
        specifier: String,
        source: String,
        fixture_url: String,
    },
    ScriptComplete {
        schema: ProtocolSchema,
        version: u32,
        ok: bool,
        result: serde_json::Value,
        error: Option<String>,
    },
    EngineCall {
        schema: ProtocolSchema,
        version: u32,
        request_id: u64,
        method: String,
        params: serde_json::Value,
    },
    EngineResult {
        schema: ProtocolSchema,
        version: u32,
        request_id: u64,
        ok: bool,
        result: serde_json::Value,
        error: Option<String>,
    },
}

impl Message {
    pub fn hello(worker: WorkerKind, capability: impl Into<String>) -> Self {
        Self::Hello {
            schema: ProtocolSchema::WorkerV1,
            version: PROTOCOL_VERSION,
            worker,
            capability: capability.into(),
        }
    }

    pub fn ready(worker: WorkerKind) -> Self {
        Self::Ready {
            schema: ProtocolSchema::WorkerV1,
            version: PROTOCOL_VERSION,
            worker,
        }
    }

    pub fn shutdown() -> Self {
        Self::Shutdown {
            schema: ProtocolSchema::WorkerV1,
            version: PROTOCOL_VERSION,
        }
    }

    pub fn shutdown_ack(worker: WorkerKind) -> Self {
        Self::ShutdownAck {
            schema: ProtocolSchema::WorkerV1,
            version: PROTOCOL_VERSION,
            worker,
        }
    }

    pub fn run_script(specifier: String, source: String, fixture_url: String) -> Self {
        Self::RunScript {
            schema: ProtocolSchema::WorkerV1,
            version: PROTOCOL_VERSION,
            specifier,
            source,
            fixture_url,
        }
    }

    pub fn script_complete(ok: bool, result: serde_json::Value, error: Option<String>) -> Self {
        Self::ScriptComplete {
            schema: ProtocolSchema::WorkerV1,
            version: PROTOCOL_VERSION,
            ok,
            result,
            error,
        }
    }

    pub fn engine_call(request_id: u64, method: String, params: serde_json::Value) -> Self {
        Self::EngineCall {
            schema: ProtocolSchema::WorkerV1,
            version: PROTOCOL_VERSION,
            request_id,
            method,
            params,
        }
    }

    pub fn engine_result(
        request_id: u64,
        ok: bool,
        result: serde_json::Value,
        error: Option<String>,
    ) -> Self {
        Self::EngineResult {
            schema: ProtocolSchema::WorkerV1,
            version: PROTOCOL_VERSION,
            request_id,
            ok,
            result,
            error,
        }
    }

    fn validate_version(self) -> io::Result<Self> {
        let version = match self {
            Self::Hello { version, .. }
            | Self::Ready { version, .. }
            | Self::Shutdown { version, .. }
            | Self::ShutdownAck { version, .. }
            | Self::RunScript { version, .. }
            | Self::ScriptComplete { version, .. }
            | Self::EngineCall { version, .. }
            | Self::EngineResult { version, .. } => version,
        };
        if version != PROTOCOL_VERSION {
            return Err(invalid_data(format!(
                "unsupported protocol version {version}; expected {PROTOCOL_VERSION}"
            )));
        }
        Ok(self)
    }
}

pub fn read_message<R: Read>(reader: &mut R) -> io::Result<Message> {
    let mut header = [0_u8; 4];
    reader.read_exact(&mut header)?;

    let payload_len = u32::from_be_bytes(header) as usize;
    if payload_len == 0 {
        return Err(invalid_data("empty protocol frame"));
    }
    if payload_len > MAX_FRAME_BYTES {
        return Err(invalid_data(format!(
            "frame length {payload_len} exceeds {MAX_FRAME_BYTES}-byte limit"
        )));
    }

    let mut payload = vec![0_u8; payload_len];
    reader.read_exact(&mut payload)?;
    let value: serde_json::Value = serde_json::from_slice(&payload)
        .map_err(|error| invalid_data(format!("invalid protocol JSON: {error}")))?;
    validate_fields(&value)?;
    serde_json::from_value::<Message>(value)
        .map_err(|error| invalid_data(format!("invalid protocol message: {error}")))?
        .validate_version()
}

pub fn write_message<W: Write>(writer: &mut W, message: &Message) -> io::Result<()> {
    let payload = serde_json::to_vec(message)
        .map_err(|error| invalid_data(format!("cannot encode protocol JSON: {error}")))?;
    if payload.len() > MAX_FRAME_BYTES {
        return Err(invalid_data(format!(
            "encoded frame length {} exceeds {MAX_FRAME_BYTES}-byte limit",
            payload.len()
        )));
    }

    let payload_len = u32::try_from(payload.len())
        .map_err(|_| invalid_data("encoded frame length does not fit in u32"))?;
    writer.write_all(&payload_len.to_be_bytes())?;
    writer.write_all(&payload)?;
    writer.flush()
}

/// Millisecond timeout from a JSON/V8 number. Integers and finite
/// non-negative floats are accepted (floats truncated toward zero).
/// Missing, non-numeric, negative, NaN, and infinite values yield
/// `default_ms`. The result is clamped to `[min_ms, max_ms]`.
pub fn timeout_ms_from_json(
    timeout: Option<&serde_json::Value>,
    default_ms: u64,
    min_ms: u64,
    max_ms: u64,
) -> u64 {
    let parsed = match timeout {
        Some(serde_json::Value::Number(number)) => json_number_as_u64(number),
        _ => None,
    };
    parsed.unwrap_or(default_ms).clamp(min_ms, max_ms)
}

fn json_number_as_u64(number: &serde_json::Number) -> Option<u64> {
    if let Some(ms) = number.as_u64() {
        return Some(ms);
    }
    if let Some(ms) = number.as_i64() {
        return u64::try_from(ms).ok();
    }
    let ms = number.as_f64()?;
    if !ms.is_finite() || ms < 0.0 {
        return None;
    }
    if ms >= u64::MAX as f64 {
        return Some(u64::MAX);
    }
    Some(ms.trunc() as u64)
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn validate_fields(value: &serde_json::Value) -> io::Result<()> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid_data("protocol message must be a JSON object"))?;
    let message_type = object
        .get("type")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| invalid_data("protocol message requires a string type field"))?;

    let allowed_fields: &[&str] = match message_type {
        "Hello" => &["type", "schema", "version", "worker", "capability"],
        "Ready" | "ShutdownAck" => &["type", "schema", "version", "worker"],
        "Shutdown" => &["type", "schema", "version"],
        "RunScript" => &[
            "type",
            "schema",
            "version",
            "specifier",
            "source",
            "fixture_url",
        ],
        "ScriptComplete" => &["type", "schema", "version", "ok", "result", "error"],
        "EngineCall" => &[
            "type",
            "schema",
            "version",
            "request_id",
            "method",
            "params",
        ],
        "EngineResult" => &[
            "type",
            "schema",
            "version",
            "request_id",
            "ok",
            "result",
            "error",
        ],
        other => return Err(invalid_data(format!("unknown protocol type {other:?}"))),
    };
    if let Some(unknown) = object
        .keys()
        .find(|field| !allowed_fields.contains(&field.as_str()))
    {
        return Err(invalid_data(format!(
            "unknown field {unknown:?} in {message_type} message"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn writes_big_endian_length_and_exact_json() {
        let mut bytes = Vec::new();
        write_message(
            &mut bytes,
            &Message::hello(WorkerKind::Controller, "test-token"),
        )
        .unwrap();

        let expected_json = br#"{"type":"Hello","schema":"greppy.web-runtime.worker.v1","version":2,"worker":"Controller","capability":"test-token"}"#;
        assert_eq!(
            &bytes[..4],
            &(expected_json.len() as u32).to_be_bytes(),
            "frame length must be a network-order u32"
        );
        assert_eq!(&bytes[4..], expected_json);
    }

    #[test]
    fn round_trips_every_message() {
        let messages = [
            Message::hello(WorkerKind::Controller, "test-token"),
            Message::ready(WorkerKind::Content),
            Message::shutdown(),
            Message::shutdown_ack(WorkerKind::Controller),
        ];

        for expected in messages {
            let mut bytes = Vec::new();
            write_message(&mut bytes, &expected).unwrap();
            let actual = read_message(&mut Cursor::new(bytes)).unwrap();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn round_trips_engine_and_script_messages() {
        let messages = [
            Message::run_script(
                "file:///tmp/spike.mjs".to_owned(),
                "import { chromium } from \"playwright\";".to_owned(),
                "data:text/html,hi".to_owned(),
            ),
            Message::script_complete(true, serde_json::json!({"ok": true}), None),
            Message::engine_call(7, "page.goto".to_owned(), serde_json::json!({"url": "x"})),
            Message::engine_result(7, true, serde_json::json!({"url": "x"}), None),
        ];
        for expected in messages {
            let mut bytes = Vec::new();
            write_message(&mut bytes, &expected).unwrap();
            let actual = read_message(&mut Cursor::new(bytes)).unwrap();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn rejects_frames_larger_than_one_mebibyte_before_reading_payload() {
        let oversized = (MAX_FRAME_BYTES as u32 + 1).to_be_bytes();
        let error = read_message(&mut Cursor::new(oversized)).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("exceeds"));
    }

    #[test]
    fn rejects_unknown_json_fields() {
        let json = br#"{"type":"Shutdown","schema":"greppy.web-runtime.worker.v1","version":2,"extra":true}"#;
        let frame = frame(json);

        let error = read_message(&mut Cursor::new(frame)).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn rejects_an_empty_frame() {
        let error = read_message(&mut Cursor::new(0_u32.to_be_bytes())).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn rejects_every_truncated_header() {
        for header_len in 1..=3 {
            let error = read_message(&mut Cursor::new(vec![0_u8; header_len])).unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
        }
    }

    #[test]
    fn rejects_a_truncated_payload() {
        let mut frame = Vec::from(20_u32.to_be_bytes());
        frame.extend_from_slice(b"{}");
        let error = read_message(&mut Cursor::new(frame)).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn rejects_invalid_utf8() {
        let error = read_message(&mut Cursor::new(frame(&[0xff]))).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn rejects_invalid_json() {
        let error = read_message(&mut Cursor::new(frame(b"{"))).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn rejects_a_wrong_schema() {
        let json = br#"{"type":"Shutdown","schema":"wrong","version":2}"#;
        let error = read_message(&mut Cursor::new(frame(json))).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn rejects_a_wrong_version() {
        let json = br#"{"type":"Shutdown","schema":"greppy.web-runtime.worker.v1","version":1}"#;
        let error = read_message(&mut Cursor::new(frame(json))).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    fn timeout_params(timeout: serde_json::Value) -> serde_json::Value {
        serde_json::json!({ "timeout": timeout })
    }

    fn parse_timeout(timeout: Option<&serde_json::Value>) -> u64 {
        timeout_ms_from_json(timeout, 30_000, 1, 120_000)
    }

    #[test]
    fn timeout_ms_from_json_accepts_integer() {
        let params = timeout_params(serde_json::json!(250));
        assert_eq!(parse_timeout(params.get("timeout")), 250);
        assert_eq!(
            params.get("timeout").and_then(serde_json::Value::as_u64),
            Some(250)
        );
    }

    #[test]
    fn timeout_ms_from_json_accepts_f64_that_as_u64_rejects() {
        let number = serde_json::Number::from_f64(250.0).expect("finite");
        let timeout = serde_json::Value::Number(number);
        assert_eq!(
            timeout.as_u64(),
            None,
            "precondition: V8-shaped f64 250.0 is not as_u64"
        );
        assert_eq!(parse_timeout(Some(&timeout)), 250);
        let truncated =
            serde_json::Value::Number(serde_json::Number::from_f64(250.9).expect("finite"));
        assert_eq!(parse_timeout(Some(&truncated)), 250);
    }

    #[test]
    fn timeout_ms_from_json_invalid_uses_default() {
        assert_eq!(parse_timeout(None), 30_000);
        assert_eq!(parse_timeout(Some(&serde_json::Value::Null)), 30_000);
        assert_eq!(parse_timeout(Some(&serde_json::json!("250"))), 30_000);
        assert_eq!(parse_timeout(Some(&serde_json::json!(true))), 30_000);
        assert_eq!(parse_timeout(Some(&serde_json::json!(-1))), 30_000);
        let negative =
            serde_json::Value::Number(serde_json::Number::from_f64(-5.0).expect("finite"));
        assert_eq!(parse_timeout(Some(&negative)), 30_000);
        let inf = serde_json::Number::from_f64(f64::INFINITY);
        assert!(inf.is_none(), "serde_json rejects non-finite numbers");
    }

    #[test]
    fn timeout_ms_from_json_clamps_to_bounds() {
        assert_eq!(
            timeout_ms_from_json(Some(&serde_json::json!(0)), 30_000, 1, 120_000),
            1
        );
        assert_eq!(
            timeout_ms_from_json(Some(&serde_json::json!(1_000_000)), 30_000, 1, 120_000),
            120_000
        );
        let tiny = serde_json::Value::Number(serde_json::Number::from_f64(0.4).expect("finite"));
        assert_eq!(timeout_ms_from_json(Some(&tiny), 30_000, 20, 120_000), 20);
    }

    #[test]
    fn engine_call_f64_timeout_survives_protocol_roundtrip() {
        let timeout =
            serde_json::Value::Number(serde_json::Number::from_f64(250.0).expect("finite"));
        let params = serde_json::json!({
            "page": "page-1",
            "source": "false",
            "timeout": timeout,
        });
        let mut bytes = Vec::new();
        write_message(
            &mut bytes,
            &Message::engine_call(9, "page.waitForFunction".to_owned(), params),
        )
        .unwrap();
        match read_message(&mut Cursor::new(bytes)).unwrap() {
            Message::EngineCall { params, .. } => {
                assert_eq!(
                    parse_timeout(params.get("timeout")),
                    250,
                    "f64 timeout must survive engine-call framing"
                );
            }
            other => panic!("expected EngineCall, got {other:?}"),
        }
    }

    fn frame(payload: &[u8]) -> Vec<u8> {
        let mut frame = Vec::from((payload.len() as u32).to_be_bytes());
        frame.extend_from_slice(payload);
        frame
    }
}
