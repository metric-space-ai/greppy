use serde::{Deserialize, Serialize};
use std::io::{self, Read, Write};

pub const MAX_FRAME_BYTES: usize = 1024 * 1024;
pub const PROTOCOL_VERSION: u32 = 1;

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
    pub fn hello(worker: WorkerKind) -> Self {
        Self::Hello {
            schema: ProtocolSchema::WorkerV1,
            version: PROTOCOL_VERSION,
            worker,
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
        "Hello" | "Ready" | "ShutdownAck" => &["type", "schema", "version", "worker"],
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
        write_message(&mut bytes, &Message::hello(WorkerKind::Controller)).unwrap();

        let expected_json = br#"{"type":"Hello","schema":"greppy.web-runtime.worker.v1","version":1,"worker":"Controller"}"#;
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
            Message::hello(WorkerKind::Controller),
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
        let json = br#"{"type":"Shutdown","schema":"greppy.web-runtime.worker.v1","version":1,"extra":true}"#;
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
        let json = br#"{"type":"Shutdown","schema":"wrong","version":1}"#;
        let error = read_message(&mut Cursor::new(frame(json))).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn rejects_a_wrong_version() {
        let json = br#"{"type":"Shutdown","schema":"greppy.web-runtime.worker.v1","version":2}"#;
        let error = read_message(&mut Cursor::new(frame(json))).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    fn frame(payload: &[u8]) -> Vec<u8> {
        let mut frame = Vec::from((payload.len() as u32).to_be_bytes());
        frame.extend_from_slice(payload);
        frame
    }
}
