//! Warm embedding daemon: keeps the EmbeddingGemma model resident across CLI
//! invocations so a query-cache miss costs one local IPC round-trip (~ms)
//! instead of a full model load (~0.2–0.4s CPU, more with GPU init).
//!
//! Resource lifecycle (owner requirement — never hold GPU memory while idle):
//! the daemon lazy-loads the model on the FIRST request, DROPS it after
//! `GREPPY_EMBED_DAEMON_MODEL_TTL_S` idle seconds (default 300 — frees VRAM
//! via the backend Drop impls), and EXITS after
//! `GREPPY_EMBED_DAEMON_EXIT_TTL_S` idle seconds (default 1800, socket
//! unlinked). One socket per MODEL IDENTITY: the socket file name embeds a
//! hash of the query-cache key (model id + prompt version + file
//! fingerprints), so a swapped GGUF simply routes to a fresh daemon and the
//! stale one idles out. Requests are served one connection at a time.
//!
//! The client treats the daemon as a pure accelerator: ANY failure (connect,
//! spawn, protocol, version skew) silently falls back to the in-process
//! load — behaviour is never worse than before. `GREPPY_NO_EMBED_DAEMON=1`
//! disables it entirely.
#![cfg(unix)]

use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::time::{Duration, Instant};

const ENV_DISABLE: &str = "GREPPY_NO_EMBED_DAEMON";
const ENV_MODEL_TTL: &str = "GREPPY_EMBED_DAEMON_MODEL_TTL_S";
const ENV_EXIT_TTL: &str = "GREPPY_EMBED_DAEMON_EXIT_TTL_S";
const ENV_LOG: &str = "GREPPY_EMBED_DAEMON_LOG";
const DEFAULT_MODEL_TTL_S: u64 = 300;
const DEFAULT_EXIT_TTL_S: u64 = 1800;
/// Generous: the first request pays the lazy model load (incl. GPU init).
const CLIENT_READ_TIMEOUT: Duration = Duration::from_secs(60);
const CLIENT_WRITE_TIMEOUT: Duration = Duration::from_secs(5);
/// Cap a request line so a corrupt client can't balloon daemon memory.
const MAX_REQUEST_BYTES: u64 = 1 << 20;

fn log_enabled() -> bool {
    std::env::var_os(ENV_LOG).is_some()
}

fn env_secs(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

/// Deterministic per-model socket name. `DefaultHasher::new()` uses fixed
/// SipHash keys (unlike `HashMap`'s `RandomState`), so the hash is stable
/// across processes and rust versions in practice; a collision would only
/// merge two of one user's model sockets, and the prompt-version handshake
/// still guards correctness.
fn socket_path(model_key: &str) -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let dir = PathBuf::from(home).join(".cache").join("greppy");
    let mut h = std::collections::hash_map::DefaultHasher::new();
    model_key.hash(&mut h);
    Some(dir.join(format!("embedd-{:016x}.sock", h.finish())))
}

fn ensure_private_dir(dir: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))
}

// ---------------------------------------------------------------- client --

/// Try to embed `text` (already normalized) via a warm daemon for `cfg`'s
/// model. Returns `None` on ANY problem — the caller falls back to the
/// in-process load.
pub(super) fn embed_query_via_daemon(
    cfg: &super::EmbeddingModelConfig,
    model_key: &str,
    text: &str,
) -> Option<Vec<f32>> {
    if std::env::var_os(ENV_DISABLE).is_some() {
        return None;
    }
    let super::EmbeddingModelSource::Gguf { .. } = &cfg.source else {
        return None;
    };
    let sock = socket_path(model_key)?;

    if let Some(v) = request(&sock, text) {
        return Some(v);
    }

    // No live daemon: remove a stale socket file, spawn one, retry briefly.
    let _ = std::fs::remove_file(&sock);
    spawn_daemon(cfg, &sock, false)?;

    // The daemon binds the socket before serving; poll until it appears.
    for _ in 0..30 {
        std::thread::sleep(Duration::from_millis(100));
        if let Some(v) = request(&sock, text) {
            return Some(v);
        }
    }
    None
}

/// Spawn the daemon process detached (new process group; survives this CLI).
fn spawn_daemon(
    cfg: &super::EmbeddingModelConfig,
    sock: &std::path::Path,
    prewarm: bool,
) -> Option<()> {
    let super::EmbeddingModelSource::Gguf { gguf, tokenizer } = &cfg.source else {
        return None;
    };
    ensure_private_dir(sock.parent()?).ok()?;
    let exe = std::env::current_exe().ok()?;
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("embed-daemon")
        .arg("--socket")
        .arg(sock)
        .arg("--gguf")
        .arg(gguf)
        .arg("--tokenizer")
        .arg(tokenizer)
        .arg("--model-id")
        .arg(&cfg.model_id)
        .arg("--device")
        .arg(cfg.device.as_str());
    if let Some(len) = cfg.max_length {
        cmd.arg("--max-length").arg(len.to_string());
    }
    if prewarm {
        cmd.arg("--prewarm");
    }
    cmd.stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    {
        // New process group: the daemon must outlive this CLI invocation and
        // not die with its (interactive) process group.
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    cmd.spawn().ok().map(|_| ())
}

/// O5: nudge the daemon into existence (and model load) in the BACKGROUND
/// from the first graph command of a session, so the session's first
/// semantic query hits a warm model instead of paying the cold start.
/// Fire-and-forget and strictly best-effort; respects the opt-out. Only
/// fires when the caller has already established that semantic search is
/// actually in play (model configured AND the store holds vectors) — a
/// prewarm that loads a model nobody will query would hold GPU memory for
/// a TTL for nothing.
pub(super) fn prewarm_from_env(cfg: &super::EmbeddingModelConfig, model_key: &str) {
    if std::env::var_os(ENV_DISABLE).is_some() {
        return;
    }
    let Some(sock) = socket_path(model_key) else {
        return;
    };
    if UnixStream::connect(&sock).is_ok() {
        return; // already warm (or warming)
    }
    let _ = std::fs::remove_file(&sock);
    let _ = spawn_daemon(cfg, &sock, true);
}

/// One request/response over an existing daemon socket. `None` = any failure.
fn request(sock: &std::path::Path, text: &str) -> Option<Vec<f32>> {
    let stream = UnixStream::connect(sock).ok()?;
    stream.set_write_timeout(Some(CLIENT_WRITE_TIMEOUT)).ok()?;
    stream.set_read_timeout(Some(CLIENT_READ_TIMEOUT)).ok()?;
    let mut writer = stream.try_clone().ok()?;
    let req = serde_json::json!({
        "pv": greppy_embed_native::PROMPT_VERSION,
        "text": text,
    });
    writer.write_all(req.to_string().as_bytes()).ok()?;
    writer.write_all(b"\n").ok()?;
    writer.flush().ok()?;
    let mut line = String::new();
    BufReader::new(stream).read_line(&mut line).ok()?;
    let resp: serde_json::Value = serde_json::from_str(line.trim()).ok()?;
    if resp.get("error").is_some() {
        return None;
    }
    let v = resp.get("v")?.as_array()?;
    let out: Option<Vec<f32>> = v.iter().map(|x| x.as_f64().map(|f| f as f32)).collect();
    out.filter(|o| !o.is_empty())
}

// ---------------------------------------------------------------- server --

/// Daemon entry point (hidden `embed-daemon` subcommand). Never returns.
/// With `prewarm` the model is loaded immediately at startup (spawned from
/// a session's first graph command) instead of on the first request; the
/// idle TTLs then govern its lifetime exactly as usual.
pub(super) fn daemon_main(socket: PathBuf, cfg: super::EmbeddingModelConfig, prewarm: bool) -> ! {
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
                // Lost a spawn race to a live daemon — defer to it.
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
            "embed-daemon: serving {} (model ttl {:?}, exit ttl {:?})",
            socket.display(),
            model_ttl,
            exit_ttl
        );
    }

    let mut model: Option<greppy_embed_native::EmbeddingGemma> = None;
    let mut last_used = Instant::now();
    if prewarm {
        let t0 = Instant::now();
        match super::load_embedding_model(&cfg, None) {
            Ok(m) => {
                if log_enabled() {
                    eprintln!("embed-daemon: prewarmed model in {:?}", t0.elapsed());
                }
                model = Some(m);
                last_used = Instant::now();
            }
            Err(_) => { /* first request will retry (and report) the load */ }
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
                eprintln!("embed-daemon: model dropped after {idle:?} idle");
            }
        }
        if idle >= exit_ttl {
            let _ = std::fs::remove_file(&socket);
            if log_enabled() {
                eprintln!("embed-daemon: exiting after {idle:?} idle");
            }
            std::process::exit(0);
        }
    }
}

fn handle_connection(
    stream: UnixStream,
    cfg: &super::EmbeddingModelConfig,
    model: &mut Option<greppy_embed_native::EmbeddingGemma>,
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
    cfg: &super::EmbeddingModelConfig,
    model: &mut Option<greppy_embed_native::EmbeddingGemma>,
) -> serde_json::Value {
    let req: serde_json::Value = match serde_json::from_str(raw) {
        Ok(v) => v,
        Err(e) => return serde_json::json!({"error": format!("bad request: {e}")}),
    };
    if req.get("pv").and_then(|v| v.as_str()) != Some(greppy_embed_native::PROMPT_VERSION) {
        // Client built against a different prompt contract: make it fall
        // back to its own in-process model rather than serve skewed vectors.
        return serde_json::json!({"error": "prompt-version mismatch"});
    }
    let Some(text) = req.get("text").and_then(|v| v.as_str()) else {
        return serde_json::json!({"error": "missing text"});
    };
    if model.is_none() {
        let t0 = Instant::now();
        match super::load_embedding_model(cfg, None) {
            Ok(m) => {
                if log_enabled() {
                    eprintln!("embed-daemon: model loaded in {:?}", t0.elapsed());
                }
                *model = Some(m);
            }
            Err(e) => return serde_json::json!({"error": format!("model load: {e}")}),
        }
    }
    let m = model.as_ref().expect("model just ensured above");
    match greppy_search::embed_code_query(m, text) {
        Ok(v) => serde_json::json!({ "v": v }),
        Err(e) => {
            // A failed inference may leave GPU state suspect: drop the model
            // so the next request reloads cleanly (or the client falls back).
            *model = None;
            serde_json::json!({"error": format!("embed: {e}")})
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_ttls_cover_agent_session_bursts() {
        assert_eq!(DEFAULT_MODEL_TTL_S, 300);
        assert_eq!(DEFAULT_EXIT_TTL_S, 1800);
    }
}
