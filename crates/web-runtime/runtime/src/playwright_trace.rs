//! Minimal Playwright trace archive writer.
//!
//! The wire format follows the action subset of Playwright trace schema version
//! 8 from Playwright v1.56.1:
//! <https://github.com/microsoft/playwright/blob/v1.56.1/packages/trace/src/trace.ts>
//! and the recorder at
//! <https://github.com/microsoft/playwright/blob/v1.56.1/packages/playwright-core/src/server/trace/recorder/tracing.ts>.

use serde_json::{json, Value};
use std::sync::OnceLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub const TRACE_SCHEMA_VERSION: u32 = 8;
const MAX_RECORDING_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug)]
pub struct TraceRecorder {
    trace: Vec<u8>,
    network: Vec<u8>,
    next_call: u64,
}

impl TraceRecorder {
    pub fn new() -> Result<Self, String> {
        let mut recorder = Self {
            trace: Vec::new(),
            network: Vec::new(),
            next_call: 1,
        };
        recorder.append(json!({
            "version": TRACE_SCHEMA_VERSION,
            "type": "context-options",
            "origin": "library",
            "browserName": "greppy",
            "options": {},
            "platform": std::env::consts::OS,
            "wallTime": wall_time_ms(),
            "monotonicTime": trace_time_ms(),
            "sdkLanguage": "javascript"
        }))?;
        Ok(recorder)
    }

    pub fn record(&mut self, operation: &str, start_time: u64, failed: bool) -> Result<(), String> {
        let call_id = format!("call@{}", self.next_call);
        self.next_call += 1;
        self.append(json!({"type":"before","callId":call_id,"startTime":start_time,"apiName":operation,"class":"Greppy","method":operation,"params":{}}))?;
        let mut after =
            json!({"type":"after","callId":call_id,"endTime":trace_time_ms(),"result":{}});
        if failed {
            after["error"] = json!({"message": "action failed"});
        }
        self.append(after)
    }

    fn append(&mut self, value: Value) -> Result<(), String> {
        let mut line = serde_json::to_vec(&value).map_err(|e| e.to_string())?;
        line.push(b'\n');
        if self
            .trace
            .len()
            .saturating_add(self.network.len())
            .saturating_add(line.len())
            > MAX_RECORDING_BYTES
        {
            return Err(
                "trace recording exceeded the 8 MiB in-memory limit; stop the trace sooner".into(),
            );
        }
        self.trace.extend(line);
        Ok(())
    }

    pub fn finish(self) -> Vec<u8> {
        zip_store(&[
            ("trace.trace", &self.trace),
            ("trace.network", &self.network),
        ])
    }

    #[cfg(test)]
    pub(crate) fn fill_to_recording_limit(&mut self) {
        self.trace.resize(MAX_RECORDING_BYTES, b' ');
    }
}

fn wall_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
pub fn trace_time_ms() -> u64 {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    ORIGIN.get_or_init(Instant::now).elapsed().as_millis() as u64
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in bytes {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb88320 & (0u32.wrapping_sub(crc & 1)));
        }
    }
    !crc
}

fn zip_store(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut directory = Vec::new();
    for (name, data) in entries {
        let offset = out.len() as u32;
        let crc = crc32(data);
        let name = name.as_bytes();
        out.extend(0x04034b50u32.to_le_bytes());
        out.extend([20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        out.extend(crc.to_le_bytes());
        out.extend((data.len() as u32).to_le_bytes());
        out.extend((data.len() as u32).to_le_bytes());
        out.extend((name.len() as u16).to_le_bytes());
        out.extend(0u16.to_le_bytes());
        out.extend(name);
        out.extend(*data);
        directory.extend(0x02014b50u32.to_le_bytes());
        directory.extend([20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        directory.extend(crc.to_le_bytes());
        directory.extend((data.len() as u32).to_le_bytes());
        directory.extend((data.len() as u32).to_le_bytes());
        directory.extend((name.len() as u16).to_le_bytes());
        directory.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        directory.extend(offset.to_le_bytes());
        directory.extend(name);
    }
    let start = out.len() as u32;
    out.extend(&directory);
    out.extend(0x06054b50u32.to_le_bytes());
    out.extend([0, 0, 0, 0]);
    out.extend((entries.len() as u16).to_le_bytes());
    out.extend((entries.len() as u16).to_le_bytes());
    out.extend((directory.len() as u32).to_le_bytes());
    out.extend(start.to_le_bytes());
    out.extend(0u16.to_le_bytes());
    out
}

pub fn archive_jsonl(trace: &[u8], network: &[u8]) -> Result<Vec<u8>, String> {
    if trace.len().saturating_add(network.len()) > MAX_RECORDING_BYTES {
        return Err("trace recording exceeded the 8 MiB in-memory limit".into());
    }
    Ok(zip_store(&[
        ("trace.trace", trace),
        ("trace.network", network),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn archive_events(archive: &[u8]) -> Vec<Value> {
        String::from_utf8_lossy(archive)
            .lines()
            .filter_map(|line| line.find('{').map(|start| &line[start..]))
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect()
    }

    #[test]
    fn archive_has_v8_entries_without_sensitive_action_data() {
        let mut r = TraceRecorder::new().unwrap();
        r.record("page.goto", trace_time_ms(), false).unwrap();
        let z = r.finish();
        assert!(z.windows(11).any(|w| w == b"trace.trace"));
        assert!(z.windows(13).any(|w| w == b"trace.network"));
        assert!(z.windows(11).any(|w| w == b"\"version\":8"));
        assert!(!z
            .windows(b"playwrightVersion".len())
            .any(|window| window == b"playwrightVersion"));
        assert!(!z.windows(13).any(|w| w == b"authorization"));
    }

    #[test]
    fn records_real_start_and_generic_failure_without_payload() {
        let mut recorder = TraceRecorder::new().unwrap();
        let start = trace_time_ms();
        recorder.record("locator.fill", start, true).unwrap();
        let archive = recorder.finish();
        let text = String::from_utf8_lossy(&archive);
        assert!(text.contains("action failed"));
        assert!(!text.contains("password"));
        let events = archive_events(&archive);
        let context_time = events
            .iter()
            .find(|event| event["type"] == "context-options")
            .unwrap()["monotonicTime"]
            .as_u64()
            .unwrap();
        let before = events
            .iter()
            .find(|event| event["type"] == "before")
            .unwrap();
        let after = events
            .iter()
            .find(|event| event["type"] == "after")
            .unwrap();
        let recorded_start = before["startTime"].as_u64().unwrap();
        let end = after["endTime"].as_u64().unwrap();
        assert!(recorded_start >= context_time);
        assert!(end >= recorded_start);
        assert!(end - recorded_start < 1_000);
    }

    #[test]
    fn recorders_are_isolated() {
        let mut first = TraceRecorder::new().unwrap();
        let mut second = TraceRecorder::new().unwrap();
        first.record("page.goto", 1, false).unwrap();
        second.record("locator.click", 2, false).unwrap();
        let first = String::from_utf8_lossy(&first.finish()).into_owned();
        let second = String::from_utf8_lossy(&second.finish()).into_owned();
        assert!(first.contains("page.goto") && !first.contains("locator.click"));
        assert!(second.contains("locator.click") && !second.contains("page.goto"));
    }

    #[test]
    fn live_recording_limit_fails_before_growth() {
        let mut recorder = TraceRecorder::new().unwrap();
        let oversized = "x".repeat(MAX_RECORDING_BYTES);
        let error = recorder.append(json!({"oversized":oversized})).unwrap_err();
        assert!(error.contains("8 MiB"));
        assert!(recorder.trace.len() < MAX_RECORDING_BYTES);
    }
}
