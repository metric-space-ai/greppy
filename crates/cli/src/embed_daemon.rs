//! Warm EmbeddingGemma daemon built on the shared inference lifecycle.
#![cfg(any(unix, windows))]

use std::time::Duration;

use super::inference_daemon::{
    self, Endpoint, RequestOutcome, ServerPolicy, SpawnOutcome, PROTOCOL_VERSION,
};

const ENV_MODEL_TTL: &str = "GREPPY_EMBED_DAEMON_MODEL_TTL_S";
const ENV_EXIT_TTL: &str = "GREPPY_EMBED_DAEMON_EXIT_TTL_S";
const DEFAULT_MODEL_TTL_S: u64 = 300;
const DEFAULT_EXIT_TTL_S: u64 = 1800;
const CLIENT_READ_TIMEOUT: Duration = Duration::from_secs(60);
const MAX_REQUEST_BYTES: usize = 1 << 20;
const MAX_RESPONSE_BYTES: usize = 4 << 20;

#[derive(Debug, PartialEq)]
pub(super) enum EmbedDaemonResult {
    Embedded(Vec<f32>),
    DaemonBusy,
    NoDaemon,
    Failed,
}

fn env_secs(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

fn endpoint(cfg: &super::EmbeddingModelConfig, model_key: &str) -> Option<Endpoint> {
    Endpoint::for_identity(
        "embedding",
        &format!(
            "{model_key}|{}",
            super::inference_device_identity(&cfg.device)
        ),
    )
}

pub(super) fn status(cfg: &super::EmbeddingModelConfig, model_key: &str) -> serde_json::Value {
    endpoint(cfg, model_key)
        .map(|endpoint| inference_daemon::diagnostic(&endpoint))
        .unwrap_or_else(|| serde_json::json!({"state": "unsupported"}))
}

pub(super) fn embed_query_via_daemon_result(
    cfg: &super::EmbeddingModelConfig,
    model_key: &str,
    text: &str,
) -> EmbedDaemonResult {
    let Some(endpoint) = endpoint(cfg, model_key) else {
        return EmbedDaemonResult::NoDaemon;
    };
    match request_embedding(&endpoint, model_key, text) {
        RequestOutcome::Response(vector) => return EmbedDaemonResult::Embedded(vector),
        RequestOutcome::DaemonBusy => return EmbedDaemonResult::DaemonBusy,
        RequestOutcome::Failed => return EmbedDaemonResult::Failed,
        RequestOutcome::NoDaemon => {}
    }

    let spawn_outcome =
        inference_daemon::spawn_once(&endpoint, || spawn_daemon(cfg, &endpoint, false));
    for delay in inference_daemon::retry_delays() {
        std::thread::sleep(delay);
        match request_embedding(&endpoint, model_key, text) {
            RequestOutcome::Response(vector) => return EmbedDaemonResult::Embedded(vector),
            RequestOutcome::DaemonBusy => return EmbedDaemonResult::DaemonBusy,
            RequestOutcome::Failed => return EmbedDaemonResult::Failed,
            RequestOutcome::NoDaemon => {}
        }
    }
    inference_daemon::record_spawn_failure(&endpoint, spawn_outcome.attempted());
    match spawn_outcome {
        SpawnOutcome::SpawnFailed => EmbedDaemonResult::NoDaemon,
        SpawnOutcome::Contended => EmbedDaemonResult::DaemonBusy,
        SpawnOutcome::Spawned | SpawnOutcome::Cooldown => EmbedDaemonResult::Failed,
    }
}

fn request_embedding(endpoint: &Endpoint, model_key: &str, text: &str) -> RequestOutcome<Vec<f32>> {
    let request = serde_json::json!({
        "pv": greppy_embed_native::PROMPT_VERSION,
        "mk": model_key,
        "text": text,
    });
    match inference_daemon::request(
        endpoint,
        request,
        CLIENT_READ_TIMEOUT,
        MAX_REQUEST_BYTES,
        MAX_RESPONSE_BYTES,
    ) {
        RequestOutcome::Response(response) => {
            if response.get("error").is_some() {
                return RequestOutcome::Failed;
            }
            let Some(values) = response.get("v_bits").and_then(serde_json::Value::as_array) else {
                return RequestOutcome::Failed;
            };
            let vector = values
                .iter()
                .map(|value| {
                    value
                        .as_u64()
                        .and_then(|bits| u32::try_from(bits).ok())
                        .map(f32::from_bits)
                })
                .collect::<Option<Vec<_>>>();
            match vector {
                Some(vector) if !vector.is_empty() => RequestOutcome::Response(vector),
                _ => RequestOutcome::Failed,
            }
        }
        RequestOutcome::NoDaemon => RequestOutcome::NoDaemon,
        RequestOutcome::DaemonBusy => RequestOutcome::DaemonBusy,
        RequestOutcome::Failed => RequestOutcome::Failed,
    }
}

fn spawn_daemon(
    cfg: &super::EmbeddingModelConfig,
    endpoint: &Endpoint,
    prewarm: bool,
) -> Option<()> {
    let super::EmbeddingModelSource::Gguf { gguf, tokenizer } = &cfg.source;
    let executable = std::env::current_exe().ok()?;
    let mut command = std::process::Command::new(executable);
    command
        .arg("embed-daemon")
        .arg("--socket")
        .arg(endpoint.address())
        .arg("--gguf")
        .arg(gguf)
        .arg("--tokenizer")
        .arg(tokenizer)
        .arg("--model-id")
        .arg(&cfg.model_id)
        .arg("--device")
        .arg(cfg.device.as_str());
    if let Some(length) = cfg.max_length {
        command.arg("--max-length").arg(length.to_string());
    }
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

pub(super) fn prewarm_from_env(cfg: &super::EmbeddingModelConfig, model_key: &str) {
    let Some(endpoint) = endpoint(cfg, model_key) else {
        return;
    };
    let ping = serde_json::json!({"op": "ping"});
    if matches!(
        inference_daemon::request(&endpoint, ping, Duration::from_secs(1), 4096, 4096),
        RequestOutcome::Response(_)
    ) {
        return;
    }
    let _ = inference_daemon::spawn_once(&endpoint, || spawn_daemon(cfg, &endpoint, true));
}

pub(super) fn daemon_main(socket: String, cfg: super::EmbeddingModelConfig, prewarm: bool) -> ! {
    let model_key = super::embedding_query_cache_key(&cfg);
    let Some(endpoint) = endpoint(&cfg, &model_key) else {
        std::process::exit(1);
    };
    let policy = ServerPolicy {
        model_ttl: Duration::from_secs(env_secs(ENV_MODEL_TTL, DEFAULT_MODEL_TTL_S)),
        exit_ttl: Duration::from_secs(env_secs(ENV_EXIT_TTL, DEFAULT_EXIT_TTL_S)),
        request_deadline: CLIENT_READ_TIMEOUT,
        hard_request_timeout: Some(Duration::from_secs(75)),
        max_request_bytes: MAX_REQUEST_BYTES,
        max_response_bytes: MAX_RESPONSE_BYTES,
    };
    inference_daemon::serve(
        endpoint,
        &socket,
        policy,
        prewarm,
        || super::load_embedding_model(&cfg, None).map_err(|error| error.to_string()),
        |raw| validate(raw, &model_key),
        |raw, model| respond(raw, &model_key, model),
        "embed-daemon",
    )
}

fn validate(raw: &str, model_key: &str) -> Result<(), serde_json::Value> {
    let request: serde_json::Value = serde_json::from_str(raw.trim())
        .map_err(|error| serde_json::json!({"error": format!("bad request: {error}")}))?;
    if request.get("pv").and_then(serde_json::Value::as_str)
        != Some(greppy_embed_native::PROMPT_VERSION)
    {
        return Err(serde_json::json!({"error": "prompt-version mismatch"}));
    }
    if request.get("mk").and_then(serde_json::Value::as_str) != Some(model_key) {
        return Err(serde_json::json!({"error": "model-key mismatch"}));
    }
    if request
        .get("text")
        .and_then(serde_json::Value::as_str)
        .is_none()
    {
        return Err(serde_json::json!({"error": "missing text"}));
    }
    Ok(())
}

fn respond(
    raw: &str,
    model_key: &str,
    model: &mut Option<super::LoadedEmbeddingModel>,
) -> serde_json::Value {
    let request: serde_json::Value = match serde_json::from_str(raw.trim()) {
        Ok(request) => request,
        Err(error) => return serde_json::json!({"error": format!("bad request: {error}")}),
    };
    if request.get("protocol").and_then(serde_json::Value::as_u64)
        != Some(u64::from(PROTOCOL_VERSION))
    {
        return serde_json::json!({"error": "protocol-version mismatch"});
    }
    if request.get("pv").and_then(serde_json::Value::as_str)
        != Some(greppy_embed_native::PROMPT_VERSION)
    {
        return serde_json::json!({"error": "prompt-version mismatch"});
    }
    if request.get("mk").and_then(serde_json::Value::as_str) != Some(model_key) {
        return serde_json::json!({"error": "model-key mismatch"});
    }
    let Some(text) = request.get("text").and_then(serde_json::Value::as_str) else {
        return serde_json::json!({"error": "missing text"});
    };
    let Some(loaded) = model.as_ref() else {
        return serde_json::json!({"error": "model unavailable"});
    };
    match greppy_search::embed_code_query(loaded, text) {
        Ok(vector) => serde_json::json!({
            "v_bits": vector.iter().map(|value| value.to_bits()).collect::<Vec<_>>()
        }),
        Err(error) => {
            *model = None;
            serde_json::json!({"error": format!("embed: {error}")})
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

    #[test]
    fn protocol_identity_is_rejected_before_model_loading() {
        let request = serde_json::json!({
            "pv": "old-prompt",
            "mk": "model-key",
            "text": "query",
        });
        assert_eq!(
            validate(&request.to_string(), "model-key").unwrap_err()["error"],
            "prompt-version mismatch"
        );
        let request = serde_json::json!({
            "pv": greppy_embed_native::PROMPT_VERSION,
            "mk": "other-model",
            "text": "query",
        });
        assert_eq!(
            validate(&request.to_string(), "model-key").unwrap_err()["error"],
            "model-key mismatch"
        );
    }
}
