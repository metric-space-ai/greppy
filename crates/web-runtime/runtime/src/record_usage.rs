//! Payload-free accounting for the records retained by one page.
//!
//! Internal limit checks must not transport logs, headers or download bodies:
//! those records can exceed an IPC frame before the limit check can run.
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RecordUsage {
    pub console_bytes: u64,
    pub download_bytes: u64,
    pub network_bytes: u64,
}

impl RecordUsage {
    pub fn from_rows(console: &[Value], downloads: &[Value], responses: &[Value]) -> Self {
        Self {
            console_bytes: console.iter().fold(0u64, |total, row| {
                total.saturating_add(
                    row.get("text").and_then(Value::as_str).map_or(0, |s| s.len() as u64),
                )
            }),
            download_bytes: byte_lengths(downloads),
            network_bytes: byte_lengths(responses),
        }
    }
}

fn byte_lengths(rows: &[Value]) -> u64 {
    rows.iter().fold(0u64, |total, row| {
        total.saturating_add(row.get("byteLength").and_then(Value::as_u64).unwrap_or(0))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{read_message, write_message, Message, MAX_FRAME_BYTES};
    use serde_json::json;

    #[test]
    fn record_usage_crosses_a_real_frame_without_copying_oversized_records() {
        let console = vec![json!({"text": "é".repeat(MAX_FRAME_BYTES)})];
        let downloads = vec![json!({"byteLength": 2_000_000, "bodyBase64": "a".repeat(3_000_000)})];
        let responses = vec![json!({"byteLength": 300, "headers": {"x-large": "b".repeat(MAX_FRAME_BYTES)}})];
        let usage = RecordUsage::from_rows(&console, &downloads, &responses);
        assert_eq!(usage.console_bytes, (2 * MAX_FRAME_BYTES) as u64);
        assert_eq!(usage.download_bytes, 2_000_000);
        assert_eq!(usage.network_bytes, 300);
        let legacy = Message::engine_result(1, true, json!({"messages": console}), None);
        assert!(write_message(&mut Vec::new(), &legacy).is_err(), "original accounting reply exceeds the frame");
        let reply = Message::engine_result(1, true, serde_json::to_value(usage).unwrap(), None);
        let mut bytes = Vec::new();
        write_message(&mut bytes, &reply).unwrap();
        assert!(bytes.len() < 256, "accounting has constant frame size: {}", bytes.len());
        let Message::EngineResult { result, .. } = read_message(&mut bytes.as_slice()).unwrap() else {
            panic!("expected usage reply");
        };
        assert_eq!(serde_json::from_value::<RecordUsage>(result).unwrap(), usage);
    }

    #[test]
    fn record_usage_rejects_incomplete_counters_and_saturates_overflow() {
        assert!(serde_json::from_value::<RecordUsage>(json!({"console_bytes": 0})).is_err());
        let rows = [json!({"byteLength": u64::MAX}), json!({"byteLength": 1})];
        assert_eq!(RecordUsage::from_rows(&[], &rows, &rows).network_bytes, u64::MAX);
        assert_eq!(RecordUsage::from_rows(&[], &rows, &rows).download_bytes, u64::MAX);
    }
}
