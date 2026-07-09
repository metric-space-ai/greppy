//! Warm Qwen3.5 summarization daemon for `brief` and semantic-search purpose summaries.
//!
//! Resource lifecycle mirrors the EmbeddingGemma daemon: the process
//! lazy-loads Qwen on the first request, keeps the model resident for
//! `GREPPY_SUMMARIZE_DAEMON_MODEL_TTL_S` idle seconds (default 300, freeing
//! VRAM through backend Drop impls), and exits after
//! `GREPPY_SUMMARIZE_DAEMON_EXIT_TTL_S` idle seconds (default 1800). One
//! socket is used per model/prompt identity, so model or prompt changes route
//! to a fresh daemon while stale daemons idle out.
//!
//! The daemon is a best-effort accelerator. Any failure is reported to the
//! client as "no summary", never as a failed user command.
#![cfg(unix)]

use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::time::{Duration, Instant};

const ENV_MODEL_TTL: &str = "GREPPY_SUMMARIZE_DAEMON_MODEL_TTL_S";
const ENV_EXIT_TTL: &str = "GREPPY_SUMMARIZE_DAEMON_EXIT_TTL_S";
const ENV_LOG: &str = "GREPPY_SUMMARIZE_DAEMON_LOG";
const DEFAULT_MODEL_TTL_S: u64 = 300;
const DEFAULT_EXIT_TTL_S: u64 = 1800;
const CLIENT_READ_TIMEOUT: Duration = Duration::from_secs(60);
#[allow(dead_code)]
const TRIAGE_CLIENT_READ_TIMEOUT: Duration = Duration::from_secs(8);
const CLIENT_WRITE_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_REQUEST_BYTES: u64 = 256 * 1024;

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

enum DaemonRequest<T> {
    Hit(T),
    Miss,
    NoDaemon,
    Failed,
}

fn log_enabled() -> bool {
    std::env::var_os(ENV_LOG).is_some()
}

fn env_secs(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

fn socket_path(model_key: &str) -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let dir = PathBuf::from(home).join(".cache").join("greppy");
    let mut h = std::collections::hash_map::DefaultHasher::new();
    model_key.hash(&mut h);
    Some(dir.join(format!("summary-{:016x}.sock", h.finish())))
}

fn ensure_private_dir(dir: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))
}

pub(super) fn summarize_source_via_daemon(
    cfg: &super::QwenSummaryConfig,
    model_key: &str,
    source: &str,
) -> Option<Vec<String>> {
    let sock = socket_path(model_key)?;
    match request_brief(&sock, model_key, source) {
        DaemonRequest::Hit(v) => return Some(v),
        DaemonRequest::Miss | DaemonRequest::Failed => return None,
        DaemonRequest::NoDaemon => {}
    }

    let _ = std::fs::remove_file(&sock);
    spawn_daemon(cfg, &sock, false)?;
    for _ in 0..30 {
        std::thread::sleep(Duration::from_millis(100));
        match request_brief(&sock, model_key, source) {
            DaemonRequest::Hit(v) => return Some(v),
            DaemonRequest::Miss | DaemonRequest::Failed => return None,
            DaemonRequest::NoDaemon => {}
        }
    }
    None
}

#[allow(dead_code)]
pub(super) fn triage_spans_via_daemon(
    cfg: &super::QwenSummaryConfig,
    model_key: &str,
    query: &str,
    spans: &[TriageSpan],
) -> Option<Vec<TriageVerdict>> {
    if spans.is_empty() {
        return None;
    }
    let sock = socket_path(model_key)?;
    match request_triage(&sock, model_key, query, spans) {
        DaemonRequest::Hit(v) => return Some(v),
        DaemonRequest::Miss | DaemonRequest::Failed => return None,
        DaemonRequest::NoDaemon => {}
    }

    let _ = std::fs::remove_file(&sock);
    spawn_daemon(cfg, &sock, false)?;
    for _ in 0..30 {
        std::thread::sleep(Duration::from_millis(100));
        match request_triage(&sock, model_key, query, spans) {
            DaemonRequest::Hit(v) => return Some(v),
            DaemonRequest::Miss | DaemonRequest::Failed => return None,
            DaemonRequest::NoDaemon => {}
        }
    }
    None
}

fn spawn_daemon(
    cfg: &super::QwenSummaryConfig,
    sock: &std::path::Path,
    prewarm: bool,
) -> Option<()> {
    ensure_private_dir(sock.parent()?).ok()?;
    let exe = std::env::current_exe().ok()?;
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("summarize-daemon")
        .arg("--socket")
        .arg(sock)
        .arg("--gguf")
        .arg(&cfg.gguf)
        .arg("--tokenizer")
        .arg(&cfg.tokenizer)
        .arg("--model-id")
        .arg(&cfg.model_id)
        .arg("--device")
        .arg(cfg.device.as_str());
    if prewarm {
        cmd.arg("--prewarm");
    }
    cmd.stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    cmd.spawn().ok().map(|_| ())
}

fn connect_daemon(sock: &std::path::Path) -> DaemonRequest<UnixStream> {
    match UnixStream::connect(sock) {
        Ok(stream) => DaemonRequest::Hit(stream),
        Err(e) if stale_socket_connect_error(e.kind()) => DaemonRequest::NoDaemon,
        Err(_) => DaemonRequest::Failed,
    }
}

fn stale_socket_connect_error(kind: std::io::ErrorKind) -> bool {
    matches!(
        kind,
        std::io::ErrorKind::NotFound
            | std::io::ErrorKind::ConnectionRefused
            | std::io::ErrorKind::ConnectionReset
            | std::io::ErrorKind::ConnectionAborted
    )
}

fn request_brief(
    sock: &std::path::Path,
    model_key: &str,
    source: &str,
) -> DaemonRequest<Vec<String>> {
    let stream = match connect_daemon(sock) {
        DaemonRequest::Hit(stream) => stream,
        DaemonRequest::NoDaemon => return DaemonRequest::NoDaemon,
        DaemonRequest::Miss | DaemonRequest::Failed => return DaemonRequest::Failed,
    };
    if stream
        .set_write_timeout(Some(CLIENT_WRITE_TIMEOUT))
        .is_err()
        || stream.set_read_timeout(Some(CLIENT_READ_TIMEOUT)).is_err()
    {
        return DaemonRequest::Failed;
    }
    let mut writer = match stream.try_clone() {
        Ok(w) => w,
        Err(_) => return DaemonRequest::Failed,
    };
    let req = serde_json::json!({
        "pv": greppy_qwen35_native::PROMPT_VERSION,
        "mk": model_key,
        "mode": "brief",
        "source": source,
    });
    if writer.write_all(req.to_string().as_bytes()).is_err()
        || writer.write_all(b"\n").is_err()
        || writer.flush().is_err()
    {
        return DaemonRequest::Failed;
    }
    let mut line = String::new();
    if BufReader::new(stream).read_line(&mut line).is_err() {
        return DaemonRequest::Failed;
    }
    let resp: serde_json::Value = match serde_json::from_str(line.trim()) {
        Ok(v) => v,
        Err(_) => return DaemonRequest::Failed,
    };
    if resp.get("error").is_some() {
        return DaemonRequest::Failed;
    }
    let Some(values) = resp.get("s").and_then(|v| v.as_array()) else {
        return DaemonRequest::Failed;
    };
    let out = values
        .iter()
        .filter_map(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    if out.is_empty() {
        DaemonRequest::Miss
    } else {
        DaemonRequest::Hit(out)
    }
}

#[allow(dead_code)]
fn request_triage(
    sock: &std::path::Path,
    model_key: &str,
    query: &str,
    spans: &[TriageSpan],
) -> DaemonRequest<Vec<TriageVerdict>> {
    let stream = match connect_daemon(sock) {
        DaemonRequest::Hit(stream) => stream,
        DaemonRequest::NoDaemon => return DaemonRequest::NoDaemon,
        DaemonRequest::Miss | DaemonRequest::Failed => return DaemonRequest::Failed,
    };
    if stream
        .set_write_timeout(Some(CLIENT_WRITE_TIMEOUT))
        .is_err()
        || stream
            .set_read_timeout(Some(TRIAGE_CLIENT_READ_TIMEOUT))
            .is_err()
    {
        return DaemonRequest::Failed;
    }
    let mut writer = match stream.try_clone() {
        Ok(w) => w,
        Err(_) => return DaemonRequest::Failed,
    };
    let req_spans = spans
        .iter()
        .map(|s| serde_json::json!({"loc": &s.loc, "code": &s.code}))
        .collect::<Vec<_>>();
    let req = serde_json::json!({
        "pv": greppy_qwen35_native::TRIAGE_PROMPT_VERSION,
        "mk": model_key,
        "mode": "triage",
        "query": query,
        "spans": req_spans,
    });
    if writer.write_all(req.to_string().as_bytes()).is_err()
        || writer.write_all(b"\n").is_err()
        || writer.flush().is_err()
    {
        return DaemonRequest::Failed;
    }
    let mut line = String::new();
    if BufReader::new(stream).read_line(&mut line).is_err() {
        return DaemonRequest::Failed;
    }
    let resp: serde_json::Value = match serde_json::from_str(line.trim()) {
        Ok(v) => v,
        Err(_) => return DaemonRequest::Failed,
    };
    if resp.get("error").is_some() {
        return DaemonRequest::Failed;
    }
    let Some(values) = resp.get("verdicts").and_then(|v| v.as_array()) else {
        return DaemonRequest::Failed;
    };
    if values.len() != spans.len() {
        return DaemonRequest::Failed;
    }
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        let Some(loc) = value.get("loc").and_then(|v| v.as_str()).map(str::trim) else {
            return DaemonRequest::Failed;
        };
        let Some(read) = value.get("read").and_then(|v| v.as_bool()) else {
            return DaemonRequest::Failed;
        };
        let reason = value
            .get("reason")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        if loc.is_empty() {
            return DaemonRequest::Failed;
        }
        out.push(TriageVerdict {
            loc: loc.to_string(),
            read,
            reason,
        });
    }
    DaemonRequest::Hit(out)
}

pub(super) fn daemon_main(socket: PathBuf, cfg: super::QwenSummaryConfig, prewarm: bool) -> ! {
    let model_ttl = Duration::from_secs(env_secs(ENV_MODEL_TTL, DEFAULT_MODEL_TTL_S));
    let exit_ttl = Duration::from_secs(env_secs(ENV_EXIT_TTL, DEFAULT_EXIT_TTL_S));

    if let Some(parent) = socket.parent() {
        if ensure_private_dir(parent).is_err() {
            std::process::exit(1);
        }
    }
    let listener = match UnixListener::bind(&socket) {
        Ok(l) => l,
        Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
            if UnixStream::connect(&socket).is_ok() {
                std::process::exit(0);
            }
            let _ = std::fs::remove_file(&socket);
            match UnixListener::bind(&socket) {
                Ok(l) => l,
                Err(_) => std::process::exit(1),
            }
        }
        Err(_) => std::process::exit(1),
    };
    if listener.set_nonblocking(true).is_err() {
        let _ = std::fs::remove_file(&socket);
        std::process::exit(1);
    }
    if log_enabled() {
        eprintln!(
            "summarize-daemon: serving {} (model ttl {:?}, exit ttl {:?})",
            socket.display(),
            model_ttl,
            exit_ttl
        );
    }

    let mut model: Option<greppy_qwen35_native::Qwen35Summarizer> = None;
    let mut last_used = Instant::now();
    if prewarm {
        let t0 = Instant::now();
        match super::load_qwen35_summarizer(&cfg) {
            Ok(m) => {
                if log_enabled() {
                    eprintln!("summarize-daemon: prewarmed model in {:?}", t0.elapsed());
                }
                model = Some(m);
                last_used = Instant::now();
            }
            Err(_) => {}
        }
    }

    loop {
        match listener.accept() {
            Ok((stream, _)) => {
                handle_connection(stream, &cfg, &mut model);
                last_used = Instant::now();
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(200));
            }
            Err(_) => std::thread::sleep(Duration::from_millis(200)),
        }
        let idle = last_used.elapsed();
        if model.is_some() && idle >= model_ttl {
            model = None; // Drop frees weights + GPU buffers (VRAM).
            if log_enabled() {
                eprintln!("summarize-daemon: model dropped after {idle:?} idle");
            }
        }
        if idle >= exit_ttl {
            let _ = std::fs::remove_file(&socket);
            if log_enabled() {
                eprintln!("summarize-daemon: exiting after {idle:?} idle");
            }
            std::process::exit(0);
        }
    }
}

fn handle_connection(
    stream: UnixStream,
    cfg: &super::QwenSummaryConfig,
    model: &mut Option<greppy_qwen35_native::Qwen35Summarizer>,
) {
    let _ = stream.set_nonblocking(false);
    let _ = stream.set_read_timeout(Some(Duration::from_secs(10)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(10)));
    let mut writer = match stream.try_clone() {
        Ok(w) => w,
        Err(_) => return,
    };
    let mut line = String::new();
    {
        let mut reader = BufReader::new(&stream).take(MAX_REQUEST_BYTES);
        if reader.read_line(&mut line).is_err() {
            return;
        }
    }
    let reply = respond(line.trim(), cfg, model);
    let _ = writer.write_all(reply.to_string().as_bytes());
    let _ = writer.write_all(b"\n");
    let _ = writer.flush();
}

fn respond(
    raw: &str,
    cfg: &super::QwenSummaryConfig,
    model: &mut Option<greppy_qwen35_native::Qwen35Summarizer>,
) -> serde_json::Value {
    let req: serde_json::Value = match serde_json::from_str(raw) {
        Ok(v) => v,
        Err(e) => return serde_json::json!({"error": format!("bad request: {e}")}),
    };
    let mode = req.get("mode").and_then(|v| v.as_str()).unwrap_or("brief");
    let expected_pv = match mode {
        "brief" => greppy_qwen35_native::PROMPT_VERSION,
        "triage" => greppy_qwen35_native::TRIAGE_PROMPT_VERSION,
        _ => return serde_json::json!({"error": "unsupported mode"}),
    };
    if req.get("pv").and_then(|v| v.as_str()) != Some(expected_pv) {
        return serde_json::json!({"error": "prompt-version mismatch"});
    }
    let expected_model_key = super::qwen_summary_model_key(cfg);
    if req.get("mk").and_then(|v| v.as_str()) != Some(expected_model_key.as_str()) {
        return serde_json::json!({"error": "model-key mismatch"});
    }
    if model.is_none() {
        let t0 = Instant::now();
        match super::load_qwen35_summarizer(cfg) {
            Ok(m) => {
                if log_enabled() {
                    eprintln!("summarize-daemon: model loaded in {:?}", t0.elapsed());
                }
                *model = Some(m);
            }
            Err(e) => return serde_json::json!({"error": format!("model load: {e}")}),
        }
    }
    let m = model.as_ref().expect("model just ensured above");
    match mode {
        "brief" => {
            let Some(source) = req.get("source").and_then(|v| v.as_str()) else {
                return serde_json::json!({"error": "missing source"});
            };
            match m.summarize_source(source) {
                Ok(summary) => serde_json::json!({ "s": summary }),
                Err(e) => {
                    *model = None;
                    serde_json::json!({"error": format!("summarize: {e}")})
                }
            }
        }
        "triage" => match respond_triage(&req, m) {
            Ok(v) => v,
            Err(e) => {
                *model = None;
                serde_json::json!({"error": format!("triage: {e}")})
            }
        },
        _ => serde_json::json!({"error": "unsupported mode"}),
    }
}

fn respond_triage(
    req: &serde_json::Value,
    m: &greppy_qwen35_native::Qwen35Summarizer,
) -> Result<serde_json::Value, String> {
    let Some(query) = req.get("query").and_then(|v| v.as_str()) else {
        return Err("missing query".to_string());
    };
    let Some(spans) = req.get("spans").and_then(|v| v.as_array()) else {
        return Err("missing spans".to_string());
    };
    let mut verdicts = Vec::with_capacity(spans.len());
    for span in spans {
        let Some(loc) = span.get("loc").and_then(|v| v.as_str()) else {
            return Err("missing span loc".to_string());
        };
        let Some(code) = span.get("code").and_then(|v| v.as_str()) else {
            return Err("missing span code".to_string());
        };
        match m.triage_span(query, loc, code) {
            Ok(verdict) => verdicts.push(serde_json::json!({
                "loc": loc,
                "read": verdict.read,
                "reason": verdict.reason,
            })),
            Err(e) => {
                return Err(e.to_string());
            }
        }
    }
    Ok(serde_json::json!({ "verdicts": verdicts }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> super::super::QwenSummaryConfig {
        super::super::QwenSummaryConfig {
            model_id: "qwen-test".to_string(),
            gguf: PathBuf::from("/missing/qwen-test.gguf"),
            tokenizer: PathBuf::from("/missing/qwen-test-tokenizer.json"),
            device: greppy_qwen35_native::DevicePreference::Cpu,
        }
    }

    #[test]
    fn default_ttls_cover_agent_session_bursts() {
        assert_eq!(DEFAULT_MODEL_TTL_S, 300);
        assert_eq!(DEFAULT_EXIT_TTL_S, 1800);
        assert!(DEFAULT_MODEL_TTL_S < DEFAULT_EXIT_TTL_S);
    }

    #[test]
    fn stale_socket_errors_are_recoverable() {
        assert!(stale_socket_connect_error(std::io::ErrorKind::NotFound));
        assert!(stale_socket_connect_error(
            std::io::ErrorKind::ConnectionRefused
        ));
        assert!(!stale_socket_connect_error(
            std::io::ErrorKind::PermissionDenied
        ));
    }

    #[test]
    fn protocol_rejects_prompt_and_model_identity_before_loading() {
        let cfg = test_config();
        let model_key = super::super::qwen_summary_model_key(&cfg);
        let mut model = None;

        let wrong_prompt = serde_json::json!({
            "pv": "old-prompt",
            "mk": model_key,
            "mode": "brief",
            "source": "fn f() {}",
        });
        assert_eq!(
            respond(&wrong_prompt.to_string(), &cfg, &mut model)["error"],
            "prompt-version mismatch"
        );
        assert!(model.is_none());

        let wrong_model = serde_json::json!({
            "pv": greppy_qwen35_native::PROMPT_VERSION,
            "mk": "wrong-model-key",
            "mode": "brief",
            "source": "fn f() {}",
        });
        assert_eq!(
            respond(&wrong_model.to_string(), &cfg, &mut model)["error"],
            "model-key mismatch"
        );
        assert!(model.is_none());
    }
}
