//! Warm Qwen3.5 daemon for brief and semantic purpose summaries.
#![cfg(any(unix, windows))]

use std::io::IsTerminal;
use std::time::Duration;

use super::inference_daemon::{self, Endpoint, RequestOutcome, ServerPolicy};

const ENV_MODEL_TTL: &str = "GREPPY_SUMMARIZE_DAEMON_MODEL_TTL_S";
const ENV_EXIT_TTL: &str = "GREPPY_SUMMARIZE_DAEMON_EXIT_TTL_S";
const DEFAULT_MODEL_TTL_S: u64 = 300;
const DEFAULT_EXIT_TTL_S: u64 = 1800;
const CLIENT_READ_TIMEOUT: Duration = Duration::from_secs(60);
const HARD_REQUEST_TIMEOUT: Duration = Duration::from_secs(75);
const MODEL_LOAD_BUDGET: Duration = inference_daemon::SUMMARY_MODEL_LOAD_BUDGET;
const SUMMARY_LOADING_PROGRESS: &str = "greppy: loading the summary model (first use) …";
const LOADING_PROGRESS_AFTER: Duration = Duration::from_secs(5);
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

/// Session prewarm: nudge the daemon into an async model load so the first
/// summary of a session does not pay the cold start. Mirrors
/// `embed_daemon::prewarm_from_env`; a live daemon is left untouched.
pub(super) fn prewarm_from_env(cfg: &super::QwenSummaryConfig) {
    let model_key = super::qwen_summary_model_key(cfg);
    let Some(endpoint) = endpoint(&model_key) else {
        return;
    };
    let ping = serde_json::json!({"op": "ping"});
    if !matches!(
        inference_daemon::request(&endpoint, ping, Duration::from_secs(1), 4096, 4096),
        RequestOutcome::NoDaemon
    ) {
        return;
    }
    let _ = inference_daemon::spawn_once(&endpoint, || spawn_daemon(cfg, &endpoint, true));
}

/// `path` is the repo-relative file path of the source span; it is part of
/// the trained prompt contract and a mandatory protocol field.
pub(super) fn summarize_source_via_daemon(
    cfg: &super::QwenSummaryConfig,
    model_key: &str,
    path: &str,
    source: &str,
) -> RequestOutcome<Vec<String>> {
    summarize_source_via_daemon_with_timeout(cfg, model_key, path, source, CLIENT_READ_TIMEOUT)
}

fn summarize_source_via_daemon_with_timeout(
    cfg: &super::QwenSummaryConfig,
    model_key: &str,
    path: &str,
    source: &str,
    timeout: Duration,
) -> RequestOutcome<Vec<String>> {
    let Some(endpoint) = endpoint(model_key) else {
        report_daemon_failure(cfg, "daemon endpoint unavailable");
        return RequestOutcome::Failed;
    };
    match request_brief(&endpoint, model_key, path, source, timeout) {
        RequestOutcome::Response(summary) => return RequestOutcome::Response(summary),
        RequestOutcome::DaemonBusy => {
            report_daemon_failure(cfg, "shared daemon busy at request deadline");
            return RequestOutcome::DaemonBusy;
        }
        RequestOutcome::Failed => {
            report_daemon_failure(cfg, "daemon request failed");
            return RequestOutcome::Failed;
        }
        RequestOutcome::NoDaemon => {}
    }

    let spawn_outcome =
        inference_daemon::spawn_once(&endpoint, || spawn_daemon(cfg, &endpoint, false));
    for delay in inference_daemon::retry_delays() {
        std::thread::sleep(delay);
        match request_brief(&endpoint, model_key, path, source, timeout) {
            RequestOutcome::Response(summary) => return RequestOutcome::Response(summary),
            RequestOutcome::DaemonBusy => {
                report_daemon_failure(cfg, "shared daemon busy at request deadline");
                return RequestOutcome::DaemonBusy;
            }
            RequestOutcome::Failed => {
                report_daemon_failure(cfg, "daemon request failed after restart");
                return RequestOutcome::Failed;
            }
            RequestOutcome::NoDaemon => {}
        }
    }
    inference_daemon::record_spawn_failure(&endpoint, spawn_outcome.attempted());
    report_daemon_failure(cfg, "daemon did not become ready");
    RequestOutcome::NoDaemon
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
        RequestOutcome::DaemonBusy | RequestOutcome::Failed => {
            report_daemon_failure(cfg, "triage daemon request failed");
            return None;
        }
        RequestOutcome::NoDaemon => {}
    }

    let spawn_outcome =
        inference_daemon::spawn_once(&endpoint, || spawn_daemon(cfg, &endpoint, false));
    for delay in inference_daemon::retry_delays() {
        std::thread::sleep(delay);
        match request_triage(&endpoint, model_key, query, spans) {
            RequestOutcome::Response(verdicts) => return Some(verdicts),
            RequestOutcome::DaemonBusy | RequestOutcome::Failed => {
                report_daemon_failure(cfg, "triage daemon request failed after restart");
                return None;
            }
            RequestOutcome::NoDaemon => {}
        }
    }
    inference_daemon::record_spawn_failure(&endpoint, spawn_outcome.attempted());
    report_daemon_failure(cfg, "triage daemon did not become ready");
    None
}

pub(super) fn report_configuration_failure(detail: &str) {
    static REPORTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        eprintln!(
            "greppy: summary configuration: {detail}; no in-process summary inference started"
        );
    }
}

fn report_daemon_failure(cfg: &super::QwenSummaryConfig, detail: &str) {
    static REPORTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        eprintln!(
            "greppy: {} summary daemon: {detail}; no model summary generated and no in-process inference started; retry the original command or inspect greppy doctor --json",
            cfg.device.as_str()
        );
    }
}

fn request_brief(
    endpoint: &Endpoint,
    model_key: &str,
    path: &str,
    source: &str,
    timeout: Duration,
) -> RequestOutcome<Vec<String>> {
    let request = serde_json::json!({
        "pv": greppy_qwen35_native::PROMPT_VERSION,
        "fv": greppy_qwen35_native::BRIEF_FILTER_VERSION,
        "mk": model_key,
        "mode": "brief",
        "path": path,
        "source": source,
    });
    let mut announced = false;
    match inference_daemon::request_waiting_for_model_load(
        endpoint,
        request,
        timeout,
        client_model_load_budget(),
        MAX_REQUEST_BYTES,
        MAX_RESPONSE_BYTES,
        |elapsed, state| {
            if announced
                || !should_announce_summary_loading(
                    elapsed,
                    state,
                    summary_loading_progress_enabled(),
                )
            {
                return;
            }
            announced = true;
            eprintln!("{SUMMARY_LOADING_PROGRESS}");
        },
    ) {
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
        RequestOutcome::DaemonBusy => RequestOutcome::DaemonBusy,
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
        MAX_REQUEST_BYTES,
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
        RequestOutcome::DaemonBusy => RequestOutcome::DaemonBusy,
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
    inference_daemon::spawn_detached(&mut command).ok()
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
        || {
            #[cfg(debug_assertions)]
            apply_test_summary_load_delay();
            super::load_qwen35_summarizer(&cfg).map_err(|error| error.to_string())
        },
        |model| model.backend_name().to_string(),
        |raw| validate(raw, &model_key),
        respond,
        "summarize-daemon",
    )
}

fn client_model_load_budget() -> Duration {
    let configured = inference_daemon::summary_model_load_budget();
    // A zero debug override must not disable the wait. Production is
    // `MODEL_LOAD_BUDGET`; shorter `GREPPY_TEST_MODEL_LOAD_BUDGET_MS` values pass.
    if configured.is_zero() {
        MODEL_LOAD_BUDGET
    } else {
        configured
    }
}

fn summary_loading_progress_enabled() -> bool {
    std::env::var("GREPPY_PROGRESS").ok().as_deref() == Some("1") || std::io::stderr().is_terminal()
}

fn should_announce_summary_loading(elapsed: Duration, state: &str, progress_enabled: bool) -> bool {
    progress_enabled && elapsed > LOADING_PROGRESS_AFTER && state == "loading"
}

/// Test-only pause before the real summary model load. Ignored in release
/// builds, matching `GREPPY_TEST_SKIP_INFERENCE`.
#[cfg(debug_assertions)]
fn apply_test_summary_load_delay() {
    const ENV_TEST_LOAD_DELAY: &str = "GREPPY_TEST_SUMMARY_LOAD_DELAY_MS";
    if let Some(ms) = std::env::var(ENV_TEST_LOAD_DELAY)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|ms| *ms > 0)
    {
        std::thread::sleep(Duration::from_millis(ms));
    }
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
            // The repo-relative file path is mandatory: it is baked into the
            // trained prompt contract (PROMPT_VERSION qwen35-brief-path-v5),
            // so a missing path must fail loudly instead of silently
            // diverging from training.
            if request
                .get("path")
                .and_then(serde_json::Value::as_str)
                .filter(|path| !path.trim().is_empty())
                .is_none()
            {
                return Err(serde_json::json!({"error": "missing path"}));
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
            let path = request
                .get("path")
                .and_then(serde_json::Value::as_str)
                .expect("validated brief path");
            match loaded.summarize_source(path, source) {
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

    #[cfg(unix)]
    struct SharedSummaryServer {
        key: String,
        endpoint: Endpoint,
        stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
        requests: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        worker: Option<std::thread::JoinHandle<()>>,
    }

    #[cfg(unix)]
    impl SharedSummaryServer {
        fn new(replies: Vec<serde_json::Value>) -> Self {
            use std::io::{BufRead, Write};
            use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
            use std::sync::Arc;
            static NONCE: AtomicU64 = AtomicU64::new(0);
            let key = format!(
                "summary-queue-test-{}-{}",
                std::process::id(),
                NONCE.fetch_add(1, Ordering::Relaxed)
            );
            let endpoint = endpoint(&key).unwrap();
            let listener = std::os::unix::net::UnixListener::bind(endpoint.address()).unwrap();
            listener.set_nonblocking(true).unwrap();
            let stop = Arc::new(AtomicBool::new(false));
            let requests = Arc::new(AtomicUsize::new(0));
            let thread_stop = Arc::clone(&stop);
            let thread_requests = Arc::clone(&requests);
            let worker = std::thread::spawn(move || {
                let deadline = std::time::Instant::now() + Duration::from_secs(3);
                let mut first_id = None;
                while !thread_stop.load(Ordering::Acquire) {
                    assert!(
                        std::time::Instant::now() < deadline,
                        "summary test client did not finish"
                    );
                    match listener.accept() {
                        Ok((mut stream, _)) => {
                            stream
                                .set_read_timeout(Some(Duration::from_secs(1)))
                                .unwrap();
                            let mut raw = String::new();
                            std::io::BufReader::new(&mut stream)
                                .read_line(&mut raw)
                                .unwrap();
                            let request: serde_json::Value = serde_json::from_str(&raw).unwrap();
                            assert_eq!(request["mode"], "brief");
                            let id = request["request_id"].clone();
                            if let Some(first) = &first_id {
                                assert_eq!(&id, first, "capacity retry changed request identity");
                            }
                            first_id = Some(id.clone());
                            let index = thread_requests.fetch_add(1, Ordering::AcqRel);
                            let mut reply = replies[index.min(replies.len() - 1)].clone();
                            reply["request_id"] = id;
                            writeln!(stream, "{reply}").unwrap();
                        }
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            std::thread::sleep(Duration::from_millis(1))
                        }
                        Err(error) => panic!("summary server accept: {error}"),
                    }
                }
            });
            Self {
                key,
                endpoint,
                stop,
                requests,
                worker: Some(worker),
            }
        }

        fn config(&self, root: &std::path::Path) -> crate::QwenSummaryConfig {
            crate::QwenSummaryConfig {
                model_id: self.key.clone(),
                gguf: root.join("must-not-load-private.gguf"),
                tokenizer: root.join("must-not-load-private-tokenizer.json"),
                device: greppy_qwen35_native::DevicePreference::Auto,
            }
        }

        fn request_count(&self) -> usize {
            self.requests.load(std::sync::atomic::Ordering::Acquire)
        }
    }

    #[cfg(unix)]
    impl Drop for SharedSummaryServer {
        fn drop(&mut self) {
            self.stop.store(true, std::sync::atomic::Ordering::Release);
            let result = self.worker.take().unwrap().join();
            let _ = std::fs::remove_file(self.endpoint.address());
            if !std::thread::panicking() {
                result.expect("shared summary test server failed");
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn shared_busy_queue_completes_and_publishes_only_one_summary() {
        let busy = serde_json::json!({"error": "daemon busy", "error_kind": "capacity", "retryable": true});
        let server = SharedSummaryServer::new(vec![
            busy.clone(),
            busy,
            serde_json::json!({"s": ["bounds the value"]}),
        ]);
        let root = tempfile::tempdir().unwrap();
        let cache = greppy_store::SummaryCache::open(root.path()).unwrap();
        let cfg = server.config(root.path());
        let invoke = || {
            crate::summarize_source_cached(
                &cfg,
                &server.key,
                (Some(&cache), None, None),
                "limits.py",
                "def clamp(v): return min(max(v, 0), 10)",
                false,
            )
        };
        assert_eq!(invoke(), Some(vec!["bounds the value".into()]));
        assert_eq!(server.request_count(), 3);
        assert_eq!(cache.count().unwrap(), 1);
        assert_eq!(invoke(), Some(vec!["bounds the value".into()]));
        assert_eq!(
            server.request_count(),
            3,
            "cached success should not reenter the daemon"
        );
        assert!(!cfg.gguf.exists());
    }

    #[cfg(unix)]
    #[test]
    fn shared_busy_deadline_keeps_typed_outcome_without_private_model() {
        let server = SharedSummaryServer::new(vec![
            serde_json::json!({"error": "inference queue full", "error_kind": "capacity", "retryable": true}),
        ]);
        let root = tempfile::tempdir().unwrap();
        let cfg = server.config(root.path());
        let outcome = summarize_source_via_daemon_with_timeout(
            &cfg,
            &server.key,
            "limits.py",
            "def clamp(v): return min(v, 10)",
            Duration::from_millis(75),
        );
        assert_eq!(outcome, RequestOutcome::DaemonBusy);
        assert!(server.request_count() > 0);
        assert!(!cfg.gguf.exists());
    }

    #[cfg(unix)]
    #[test]
    fn shared_model_failure_is_not_capacity_and_does_not_publish_cache() {
        let server =
            SharedSummaryServer::new(vec![serde_json::json!({"error": "model unavailable"})]);
        let root = tempfile::tempdir().unwrap();
        let cache = greppy_store::SummaryCache::open(root.path()).unwrap();
        let cfg = server.config(root.path());
        assert!(crate::summarize_source_cached(
            &cfg,
            &server.key,
            (Some(&cache), None, None),
            "limits.py",
            "def clamp(v): return min(v, 10)",
            false
        )
        .is_none());
        assert_eq!(server.request_count(), 1);
        assert_eq!(cache.count().unwrap(), 0);
        assert!(!cfg.gguf.exists());
    }

    #[test]
    fn default_ttls_cover_agent_session_bursts() {
        assert_eq!(DEFAULT_MODEL_TTL_S, 300);
        assert_eq!(DEFAULT_EXIT_TTL_S, 1800);
    }

    #[test]
    fn model_load_budget_is_three_minutes() {
        assert_eq!(MODEL_LOAD_BUDGET, Duration::from_secs(180));
        assert_eq!(
            inference_daemon::SUMMARY_MODEL_LOAD_BUDGET,
            MODEL_LOAD_BUDGET
        );
    }

    #[test]
    fn summary_loading_progress_is_one_stderr_line_after_five_seconds() {
        assert_eq!(
            SUMMARY_LOADING_PROGRESS,
            "greppy: loading the summary model (first use) …"
        );
        assert!(!SUMMARY_LOADING_PROGRESS.contains('\n'));
        assert!(!should_announce_summary_loading(
            Duration::from_secs(5),
            "loading",
            true
        ));
        assert!(should_announce_summary_loading(
            Duration::from_secs(5) + Duration::from_millis(1),
            "loading",
            true
        ));
        assert!(!should_announce_summary_loading(
            Duration::from_secs(30),
            "loading",
            false
        ));
        assert!(!should_announce_summary_loading(
            Duration::from_secs(30),
            "busy",
            true
        ));
        assert!(!should_announce_summary_loading(
            Duration::from_secs(30),
            "ready",
            true
        ));
    }

    #[cfg(debug_assertions)]
    #[test]
    fn test_summary_load_delay_is_ignored_until_configured() {
        let started = std::time::Instant::now();
        apply_test_summary_load_delay();
        assert!(started.elapsed() < Duration::from_millis(200));

        let previous = std::env::var_os("GREPPY_TEST_SUMMARY_LOAD_DELAY_MS");
        std::env::set_var("GREPPY_TEST_SUMMARY_LOAD_DELAY_MS", "40");
        let started = std::time::Instant::now();
        let result = std::panic::catch_unwind(|| apply_test_summary_load_delay());
        let elapsed = started.elapsed();
        if let Some(previous) = previous {
            std::env::set_var("GREPPY_TEST_SUMMARY_LOAD_DELAY_MS", previous);
        } else {
            std::env::remove_var("GREPPY_TEST_SUMMARY_LOAD_DELAY_MS");
        }
        assert!(result.is_ok());
        assert!(
            elapsed >= Duration::from_millis(40),
            "configured test load delay was ignored: {elapsed:?}"
        );
        assert!(elapsed < Duration::from_secs(2), "{elapsed:?}");
    }

    #[test]
    fn protocol_rejects_identity_before_loading() {
        let wrong = serde_json::json!({
            "pv": "old-prompt",
            "mk": "model-key",
            "mode": "brief",
            "path": "src/lib.rs",
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
            "path": "src/lib.rs",
            "source": "fn f() {}",
        });
        assert_eq!(
            validate(&stale_filter.to_string(), "model-key").unwrap_err()["error"],
            "filter-version mismatch"
        );
    }

    #[test]
    fn brief_protocol_requires_repo_relative_path() {
        // Requests without a path (the pre-path-prompt protocol shape) must
        // fail loudly: the prompt contract bakes the path into training.
        let missing = serde_json::json!({
            "pv": greppy_qwen35_native::PROMPT_VERSION,
            "fv": greppy_qwen35_native::BRIEF_FILTER_VERSION,
            "mk": "model-key",
            "mode": "brief",
            "source": "fn f() {}",
        });
        assert_eq!(
            validate(&missing.to_string(), "model-key").unwrap_err()["error"],
            "missing path"
        );

        let empty = serde_json::json!({
            "pv": greppy_qwen35_native::PROMPT_VERSION,
            "fv": greppy_qwen35_native::BRIEF_FILTER_VERSION,
            "mk": "model-key",
            "mode": "brief",
            "path": "  ",
            "source": "fn f() {}",
        });
        assert_eq!(
            validate(&empty.to_string(), "model-key").unwrap_err()["error"],
            "missing path"
        );

        let complete = serde_json::json!({
            "pv": greppy_qwen35_native::PROMPT_VERSION,
            "fv": greppy_qwen35_native::BRIEF_FILTER_VERSION,
            "mk": "model-key",
            "mode": "brief",
            "path": "src/lib.rs",
            "source": "fn f() {}",
        });
        assert!(validate(&complete.to_string(), "model-key").is_ok());
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
