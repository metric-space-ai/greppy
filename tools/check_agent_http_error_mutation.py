#!/usr/bin/env python3
# Prove actual HTTP error handling against a failing socket-timeout syscall.
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CLIENT = ROOT / "crates/agent/src/client.rs"
FILTER = "client::http_error_tests::"
BUFFERED = FILTER + "buffered_http_error_survives_socket_timeout_reset_failure"
UNREADABLE = FILTER + "http_body_read_errors_remain_explicit"


def run(filter_name, failed=None):
    result = subprocess.run(
        ["cargo", "test", "--locked", "--jobs", "2", "-p", "greppy-agent", "--lib",
         filter_name, "--", "--test-threads=1"],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=600,
    )
    if failed is None:
        if result.returncode != 0 or " ... ok" not in result.stdout:
            raise AssertionError("HTTP contract did not run successfully:\n" + result.stdout)
    elif not (result.returncode == 101 and failed + " ... FAILED" in result.stdout
              and "test result: FAILED" in result.stdout):
        raise AssertionError("Wrong mutation outcome:\n" + result.stdout)
    return result


def mutate(fixed, broken, failed, label, diagnostic=None):
    original = CLIENT.read_text()
    if original.count(fixed) != 1:
        raise AssertionError("Mutation anchor is not unique: " + fixed)
    try:
        CLIENT.write_text(original.replace(fixed, broken, 1))
        result = run(failed, failed)
        if diagnostic and diagnostic not in result.stdout:
            raise AssertionError("Original buffered-body panic was not reproduced:\n" + result.stdout)
        print("mutation: " + label + " makes the real HTTP contract fail", flush=True)
    finally:
        CLIENT.write_text(original)


run(FILTER)
mutate(
    "        self.stream_turn_interruptible(req, on_event, &AtomicBool::new(false))",
    r'''
        // Mutation: restore the original synchronous, buffered ureq path.
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(10))
            .timeout_read(Duration::from_secs(600)).build();
        let body = serde_json::to_string(&self.request_body(req)?)
            .map_err(|e| ClientError::Transport(e.to_string()))?;
        let mut request = agent.post(&self.messages_url())
            .set("Content-Type", "application/json")
            .set("Accept", "text/event-stream");
        if let Some(key) = &self.api_key {
            request = request.set("x-api-key", key)
                .set("Authorization", &format!("Bearer {key}"));
        }
        match request.send_string(&body) {
            Ok(response) => self.consume_sse(&mut response.into_reader(), on_event),
            Err(ureq::Error::Status(status, response)) => Err(ClientError::Http {
                status, body: response.into_string().unwrap_or_default(),
            }),
            Err(error) => Err(ClientError::Transport(error.to_string())),
        }
''',
    BUFFERED, "restoring original ureq body conversion",
    "failed to read exact buffer length from stream",
)
mutate(
    '                .unwrap_or_else(|error| format!("response body unavailable: {error}"));',
    '                .unwrap_or_default();',
    UNREADABLE, "silently erasing an unreadable HTTP error body",
)
run(FILTER)
run("client::cancel_tests::")
print("mutation: restored HTTP status/body/auth/discovery and cancellation contracts green", flush=True)
