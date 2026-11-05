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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retention_error: Option<String>,
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
            retention_error: None,
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

    /// An explicit IPC comparison, not a page-load benchmark. Both peers use
    /// the production framing and identical retained records on one CPU.
    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "measurement; run explicitly with --ignored --nocapture"]
    fn record_usage_transport_comparison() {
        use std::os::unix::net::UnixStream;
        use std::time::Instant;
        fn process_usage() -> (u64, i64) {
            let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
            assert_eq!(unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) }, 0);
            let usage = unsafe { usage.assume_init() };
            let cpu_us = (usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) as u64 * 1_000_000
                + (usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) as u64;
            (cpu_us, usage.ru_maxrss)
        }
        let console = vec![json!({"text": "é".repeat(32_768)})];
        let downloads = vec![json!({"byteLength": 393_216, "bodyBase64": "a".repeat(524_288)})];
        let responses = vec![json!({"byteLength": 65_536, "headers": {"x-padding": "b".repeat(65_536)}})];
        let usage = RecordUsage::from_rows(&console, &downloads, &responses);
        let cases = [
            ("full_records", vec![
                ("page.consoleMessages", json!({"messages": console})),
                ("page.downloads", json!({"downloads": downloads})),
                ("page.responses", json!({"responses": responses})),
            ]),
            ("record_usage", vec![("page.recordUsage", serde_json::to_value(usage).unwrap())]),
        ];
        for (label, replies) in cases {
            let cpu_before = process_usage().0;
            let (mut client, mut server) = UnixStream::pair().unwrap();
            let calls = replies.len();
            let rounds = 40;
            let peer = std::thread::spawn(move || {
                let mut bytes = 0usize;
                for _ in 0..rounds {
                    for (_, result) in &replies {
                        let request = read_message(&mut server).unwrap();
                        let Message::EngineCall { request_id, .. } = request else { panic!("engine call") };
                        let reply = Message::engine_result(request_id, true, result.clone(), None);
                        let mut frame = Vec::new();
                        write_message(&mut frame, &reply).unwrap();
                        bytes += frame.len();
                        std::io::Write::write_all(&mut server, &frame).unwrap();
                    }
                }
                bytes
            });
            let started = Instant::now();
            let mut request_bytes = 0usize;
            let mut samples = Vec::new();
            for _ in 0..rounds {
                let start = Instant::now();
                let mut received = Vec::new();
                for index in 0..calls {
                    let method = if calls == 1 { "page.recordUsage" } else {
                        ["page.consoleMessages", "page.downloads", "page.responses"][index]
                    };
                    let request = Message::engine_call(index as u64, method.into(), json!({"page": "page-1"}));
                    let mut frame = Vec::new();
                    write_message(&mut frame, &request).unwrap();
                    request_bytes += frame.len();
                    std::io::Write::write_all(&mut client, &frame).unwrap();
                    let Message::EngineResult { result, ok, .. } = read_message(&mut client).unwrap() else { panic!("engine result") };
                    assert!(ok);
                    received.push(result);
                }
                let observed = if calls == 1 {
                    serde_json::from_value::<RecordUsage>(received.remove(0)).unwrap()
                } else {
                    RecordUsage::from_rows(
                        received[0]["messages"].as_array().unwrap(),
                        received[1]["downloads"].as_array().unwrap(),
                        received[2]["responses"].as_array().unwrap(),
                    )
                };
                assert_eq!(observed, usage, "both transports must report identical usage");
                samples.push(start.elapsed().as_micros());
            }
            let bytes = request_bytes + peer.join().unwrap();
            samples.sort();
            let (cpu_after, peak_rss_kib) = process_usage();
            eprintln!("RECORD_USAGE_TRANSPORT mode={label} rounds={rounds} frames={} bytes={bytes} wall_us={} median_us={} p95_us={} cpu_us={} process_peak_rss_kib={peak_rss_kib} network_bytes=0",
                rounds * calls * 2, started.elapsed().as_micros(), samples[rounds / 2], samples[rounds * 95 / 100], cpu_after - cpu_before);
        }
    }
}
