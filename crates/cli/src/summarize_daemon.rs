//! Warm Qwen3.5 daemon for brief and semantic purpose summaries.
#![cfg(any(unix, windows))]

use std::time::Duration;

use super::inference_daemon::{self, Endpoint, RequestOutcome, ServerPolicy};

const ENV_MODEL_TTL: &str = "GREPPY_SUMMARIZE_DAEMON_MODEL_TTL_S";
const ENV_EXIT_TTL: &str = "GREPPY_SUMMARIZE_DAEMON_EXIT_TTL_S";
const DEFAULT_MODEL_TTL_S: u64 = 300;
const DEFAULT_EXIT_TTL_S: u64 = 1800;
const CLIENT_READ_TIMEOUT: Duration = Duration::from_secs(60);
const HARD_REQUEST_TIMEOUT: Duration = Duration::from_secs(75);
#[allow(dead_code)]
const TRIAGE_CLIENT_READ_TIMEOUT: Duration = Duration::from_secs(8);
const MAX_REQUEST_BYTES: usize = 256 * 1024;
const MAX_RESPONSE_BYTES: usize = 64 * 1024;
const MAX_TRIAGE_SPANS: usize = 8;
const MAX_TRIAGE_CODE_BYTES: usize = 2 * 1024;
const MAX_TRIAGE_CODE_LINES: usize = 40;

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(super) struct TriageSpan {
    pub loc: String,
    pub code: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct TriageVerdict {
    pub loc: String,
    pub read: bool,
    pub reason: String,
}

fn env_secs(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

fn endpoint(model_key: &str) -> Option<Endpoint> {
    Endpoint::for_identity("summary", model_key)
}

pub(super) fn status(model_key: &str) -> serde_json::Value {
    endpoint(model_key)
        .map(|endpoint| inference_daemon::diagnostic(&endpoint))
        .unwrap_or_else(|| serde_json::json!({"state": "unsupported"}))
}

pub(super) fn summarize_source_via_daemon(
    cfg: &super::QwenSummaryConfig,
    model_key: &str,
    source: &str,
) -> Option<Vec<String>> {
    let endpoint = endpoint(model_key)?;
    match request_brief(&endpoint, model_key, source) {
        RequestOutcome::Response(summary) => return Some(summary),
        RequestOutcome::Failed => {
            report_explicit_backend_failure(cfg, "daemon request failed");
            return None;
        }
        RequestOutcome::NoDaemon => {}
    }

    let _ = inference_daemon::spawn_once(&endpoint, || spawn_daemon(cfg, &endpoint, false));
    for delay in inference_daemon::retry_delays() {
        std::thread::sleep(delay);
        match request_brief(&endpoint, model_key, source) {
            RequestOutcome::Response(summary) => return Some(summary),
            RequestOutcome::Failed => {
                report_explicit_backend_failure(cfg, "daemon request failed after restart");
                return None;
            }
            RequestOutcome::NoDaemon => {}
        }
    }
    inference_daemon::record_spawn_failure(&endpoint);
    report_explicit_backend_failure(cfg, "daemon did not become ready");
    None
}

#[allow(dead_code)]
pub(super) fn triage_spans_via_daemon(
    cfg: &super::QwenSummaryConfig,
    model_key: &str,
    query: &str,
    spans: &[TriageSpan],
) -> Option<Vec<TriageVerdict>> {
    if spans.is_empty() || spans.len() > MAX_TRIAGE_SPANS {
        return None;
    }
    let endpoint = endpoint(model_key)?;
    match request_triage(&endpoint, model_key, query, spans) {
        RequestOutcome::Response(verdicts) => return Some(verdicts),
        RequestOutcome::Failed => {
            report_explicit_backend_failure(cfg, "triage daemon request failed");
            return None;
        }
        RequestOutcome::NoDaemon => {}
    }

    let _ = inference_daemon::spawn_once(&endpoint, || spawn_daemon(cfg, &endpoint, false));
    for delay in inference_daemon::retry_delays() {
        std::thread::sleep(delay);
        match request_triage(&endpoint, model_key, query, spans) {
            RequestOutcome::Response(verdicts) => return Some(verdicts),
            RequestOutcome::Failed => {
                report_explicit_backend_failure(cfg, "triage daemon request failed after restart");
                return None;
            }
            RequestOutcome::NoDaemon => {}
        }
    }
    inference_daemon::record_spawn_failure(&endpoint);
    report_explicit_backend_failure(cfg, "triage daemon did not become ready");
    None
}

fn report_explicit_backend_failure(cfg: &super::QwenSummaryConfig, detail: &str) {
    if !matches!(
        cfg.device,
        greppy_qwen35_native::DevicePreference::Metal
            | greppy_qwen35_native::DevicePreference::Cuda
    ) {
        return;
    }
    static REPORTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        eprintln!(
            "greppy: explicit {} summary inference failed ({detail}); deterministic output preserved",
            cfg.device.as_str()
        );
    }
}

fn request_brief(
    endpoint: &Endpoint,
    model_key: &str,
    source: &str,
) -> RequestOutcome<Vec<String>> {
    let request = serde_json::json!({
        "pv": greppy_qwen35_native::PROMPT_VERSION,
        "fv": greppy_qwen35_native::BRIEF_FILTER_VERSION,
        "mk": model_key,
        "mode": "brief",
        "source": source,
    });
    match inference_daemon::request(endpoint, request, CLIENT_READ_TIMEOUT, MAX_RESPONSE_BYTES) {
        RequestOutcome::Response(response) => {
            if response.get("error").is_some() {
                return RequestOutcome::Failed;
            }
            let Some(values) = response.get("s").and_then(serde_json::Value::as_array) else {
                return RequestOutcome::Failed;
            };
            let summary = values
                .iter()
                .filter_map(serde_json::Value::as_str)
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>();
            if summary.is_empty() {
                RequestOutcome::Failed
            } else {
                RequestOutcome::Response(summary)
            }
        }
        RequestOutcome::NoDaemon => RequestOutcome::NoDaemon,
        RequestOutcome::Failed => RequestOutcome::Failed,
    }
}

#[allow(dead_code)]
fn request_triage(
    endpoint: &Endpoint,
    model_key: &str,
    query: &str,
    spans: &[TriageSpan],
) -> RequestOutcome<Vec<TriageVerdict>> {
    let spans = spans
        .iter()
        .map(|span| serde_json::json!({"loc": span.loc, "code": span.code}))
        .collect::<Vec<_>>();
    let request = serde_json::json!({
        "pv": greppy_qwen35_native::TRIAGE_PROMPT_VERSION,
        "fv": greppy_qwen35_native::BRIEF_FILTER_VERSION,
        "mk": model_key,
        "mode": "triage",
        "query": query,
        "spans": spans,
    });
    match inference_daemon::request(
        endpoint,
        request,
        TRIAGE_CLIENT_READ_TIMEOUT,
        MAX_RESPONSE_BYTES,
    ) {
        RequestOutcome::Response(response) => {
            if response.get("error").is_some() {
                return RequestOutcome::Failed;
            }
            let Some(values) = response
                .get("verdicts")
                .and_then(serde_json::Value::as_array)
            else {
                return RequestOutcome::Failed;
            };
            if values.len() != spans.len() {
                return RequestOutcome::Failed;
            }
            let mut verdicts = Vec::with_capacity(values.len());
            for value in values {
                let Some(loc) = value
                    .get("loc")
                    .and_then(serde_json::Value::as_str)
                    .map(str::trim)
                    .filter(|loc| !loc.is_empty())
                else {
                    return RequestOutcome::Failed;
                };
                let Some(read) = value.get("read").and_then(serde_json::Value::as_bool) else {
                    return RequestOutcome::Failed;
                };
                let reason = value
                    .get("reason")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .trim()
                    .to_string();
                verdicts.push(TriageVerdict {
                    loc: loc.to_string(),
                    read,
                    reason,
                });
            }
            RequestOutcome::Response(verdicts)
        }
        RequestOutcome::NoDaemon => RequestOutcome::NoDaemon,
        RequestOutcome::Failed => RequestOutcome::Failed,
    }
}

fn spawn_daemon(cfg: &super::QwenSummaryConfig, endpoint: &Endpoint, prewarm: bool) -> Option<()> {
    let executable = std::env::current_exe().ok()?;
    let mut command = std::process::Command::new(executable);
    command
        .arg("summarize-daemon")
        .arg("--socket")
        .arg(endpoint.address())
        .arg("--gguf")
        .arg(&cfg.gguf)
        .arg("--tokenizer")
        .arg(&cfg.tokenizer)
        .arg("--model-id")
        .arg(&cfg.model_id)
        .arg("--device")
        .arg(cfg.device.as_str());
    if prewarm {
        command.arg("--prewarm");
    }
    command
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    inference_daemon::detach_command(&mut command);
    command.spawn().ok().map(|_| ())
}

pub(super) fn daemon_main(socket: String, cfg: super::QwenSummaryConfig, prewarm: bool) -> ! {
    let model_key = super::qwen_summary_model_key(&cfg);
    let Some(endpoint) = endpoint(&model_key) else {
        std::process::exit(1);
    };
    let policy = ServerPolicy {
        model_ttl: Duration::from_secs(env_secs(ENV_MODEL_TTL, DEFAULT_MODEL_TTL_S)),
        exit_ttl: Duration::from_secs(env_secs(ENV_EXIT_TTL, DEFAULT_EXIT_TTL_S)),
        request_deadline: CLIENT_READ_TIMEOUT,
        hard_request_timeout: Some(HARD_REQUEST_TIMEOUT),
        max_request_bytes: MAX_REQUEST_BYTES,
        max_response_bytes: MAX_RESPONSE_BYTES,
    };
    inference_daemon::serve(
        endpoint,
        &socket,
        policy,
        prewarm,
        || super::load_qwen35_summarizer(&cfg).map_err(|error| error.to_string()),
        |raw| validate(raw, &model_key),
        respond,
        "summarize-daemon",
    )
}

fn validate(raw: &str, model_key: &str) -> Result<(), serde_json::Value> {
    let request: serde_json::Value = serde_json::from_str(raw.trim())
        .map_err(|error| serde_json::json!({"error": format!("bad request: {error}")}))?;
    let mode = request
        .get("mode")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("brief");
    let expected_prompt = match mode {
        "brief" => greppy_qwen35_native::PROMPT_VERSION,
        "triage" => greppy_qwen35_native::TRIAGE_PROMPT_VERSION,
        _ => return Err(serde_json::json!({"error": "unsupported mode"})),
    };
    if request.get("pv").and_then(serde_json::Value::as_str) != Some(expected_prompt) {
        return Err(serde_json::json!({"error": "prompt-version mismatch"}));
    }
    if request.get("fv").and_then(serde_json::Value::as_str)
        != Some(greppy_qwen35_native::BRIEF_FILTER_VERSION)
    {
        return Err(serde_json::json!({"error": "filter-version mismatch"}));
    }
    if request.get("mk").and_then(serde_json::Value::as_str) != Some(model_key) {
        return Err(serde_json::json!({"error": "model-key mismatch"}));
    }
    match mode {
        "brief" => {
            if request
                .get("source")
                .and_then(serde_json::Value::as_str)
                .is_none()
            {
                return Err(serde_json::json!({"error": "missing source"}));
            }
        }
        "triage" => validate_triage(&request)?,
        _ => unreachable!(),
    }
    Ok(())
}

fn validate_triage(request: &serde_json::Value) -> Result<(), serde_json::Value> {
    if request
        .get("query")
        .and_then(serde_json::Value::as_str)
        .filter(|query| !query.trim().is_empty())
        .is_none()
    {
        return Err(serde_json::json!({"error": "missing query"}));
    }
    let Some(spans) = request.get("spans").and_then(serde_json::Value::as_array) else {
        return Err(serde_json::json!({"error": "missing spans"}));
    };
    if spans.is_empty() || spans.len() > MAX_TRIAGE_SPANS {
        return Err(serde_json::json!({"error": "invalid triage span count"}));
    }
    for span in spans {
        let valid_loc = span
            .get("loc")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|loc| !loc.is_empty() && loc.len() <= 1024);
        let valid_code = span
            .get("code")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|code| {
                code.len() <= MAX_TRIAGE_CODE_BYTES && code.lines().count() <= MAX_TRIAGE_CODE_LINES
            });
        if !valid_loc || !valid_code {
            return Err(serde_json::json!({"error": "invalid triage span"}));
        }
    }
    Ok(())
}

fn respond(raw: &str, model: &mut Option<super::LoadedQwen35Summarizer>) -> serde_json::Value {
    let request: serde_json::Value = match serde_json::from_str(raw.trim()) {
        Ok(request) => request,
        Err(error) => return serde_json::json!({"error": format!("bad request: {error}")}),
    };
    let Some(loaded) = model.as_ref() else {
        return serde_json::json!({"error": "model unavailable"});
    };
    match request
        .get("mode")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("brief")
    {
        "brief" => {
            let source = request
                .get("source")
                .and_then(serde_json::Value::as_str)
                .expect("validated brief source");
            match loaded.summarize_source(source) {
                Ok(summary) => serde_json::json!({"s": summary}),
                Err(error) => {
                    *model = None;
                    serde_json::json!({"error": format!("summarize: {error}")})
                }
            }
        }
        "triage" => match respond_triage(&request, loaded) {
            Ok(response) => response,
            Err(error) => {
                *model = None;
                serde_json::json!({"error": format!("triage: {error}")})
            }
        },
        _ => serde_json::json!({"error": "unsupported mode"}),
    }
}

fn respond_triage(
    request: &serde_json::Value,
    model: &greppy_qwen35_native::Qwen35Summarizer,
) -> Result<serde_json::Value, String> {
    let query = request
        .get("query")
        .and_then(serde_json::Value::as_str)
        .expect("validated triage query");
    let spans = request
        .get("spans")
        .and_then(serde_json::Value::as_array)
        .expect("validated triage spans");
    let mut verdicts = Vec::with_capacity(spans.len());
    for span in spans {
        let loc = span
            .get("loc")
            .and_then(serde_json::Value::as_str)
            .expect("validated span location");
        let code = span
            .get("code")
            .and_then(serde_json::Value::as_str)
            .expect("validated span code");
        let verdict = model
            .triage_span(query, loc, code)
            .map_err(|error| error.to_string())?;
        verdicts.push(serde_json::json!({
            "loc": loc,
            "read": verdict.read,
            "reason": verdict.reason,
        }));
    }
    Ok(serde_json::json!({"verdicts": verdicts}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_ttls_cover_agent_session_bursts() {
        assert_eq!(DEFAULT_MODEL_TTL_S, 300);
        assert_eq!(DEFAULT_EXIT_TTL_S, 1800);
        assert!(DEFAULT_MODEL_TTL_S < DEFAULT_EXIT_TTL_S);
    }

    #[test]
    fn protocol_rejects_identity_before_loading() {
        let wrong = serde_json::json!({
            "pv": "old-prompt",
            "mk": "model-key",
            "mode": "brief",
            "source": "fn f() {}",
        });
        assert_eq!(
            validate(&wrong.to_string(), "model-key").unwrap_err()["error"],
            "prompt-version mismatch"
        );

        let stale_filter = serde_json::json!({
            "pv": greppy_qwen35_native::PROMPT_VERSION,
            "fv": "old-filter",
            "mk": "model-key",
            "mode": "brief",
            "source": "fn f() {}",
        });
        assert_eq!(
            validate(&stale_filter.to_string(), "model-key").unwrap_err()["error"],
            "filter-version mismatch"
        );
    }

    #[test]
    fn triage_limits_are_enforced_before_loading() {
        let request = serde_json::json!({
            "query": "where is work stolen",
            "spans": [{"loc": "worker.rs:1", "code": "line\n".repeat(41)}],
        });
        assert_eq!(
            validate_triage(&request).unwrap_err()["error"],
            "invalid triage span"
        );
    }
}
