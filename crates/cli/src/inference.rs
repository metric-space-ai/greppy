//! The inference surface the commands touch: which backend a build has,
//! what the daemons report, and how far the embedding pass has come.
//!
//! The daemons themselves live in `inference_daemon.rs`, `embed_daemon.rs`
//! and `summarize_daemon.rs`; this is the part `lib.rs` used to carry.

use super::*;

pub(crate) fn inference_device_identity(device: &greppy_embed_native::DevicePreference) -> String {
    if *device == greppy_embed_native::DevicePreference::Cuda {
        if let Some(index) =
            env_nonempty(ENV_QWEN_CUDA_DEVICE).or_else(|| env_nonempty(ENV_EMBED_CUDA_DEVICE))
        {
            return format!("cuda:{index}");
        }
    }
    device.as_str().to_string()
}

pub(crate) fn embedding_model_source_exists(source: &EmbeddingModelSource) -> bool {
    let EmbeddingModelSource::Gguf { gguf, tokenizer } = source;
    gguf.is_file() && tokenizer.is_file()
}

pub(crate) fn embedding_backend_plan(cfg: &EmbeddingModelConfig) -> (String, Option<String>) {
    let EmbeddingModelSource::Gguf { gguf, .. } = &cfg.source;
    let model_bytes = std::fs::metadata(gguf)
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    let required = greppy_embed_native::estimated_gpu_memory(
        greppy_embed_native::InferenceModelKind::EmbeddingGemma,
        model_bytes,
    );
    let selector = inference_device_identity(&cfg.device);
    let policy = greppy_embed_native::InferencePolicy::from_selector(Some(&selector), false);
    let registry = policy.ok().map(|policy| {
        greppy_embed_native::InferenceBackendRegistry::probe_policy(&policy, required)
    });
    let backend = registry
        .as_ref()
        .and_then(|registry| registry.selected_backend)
        .map(greppy_embed_native::BackendKind::as_str)
        .unwrap_or_else(|| cfg.device.as_str())
        .to_string();
    let device = registry
        .and_then(|registry| registry.selected_device_id)
        .or_else(|| (selector != "auto").then_some(selector));
    (backend, device)
}

pub(crate) fn embedding_generation_complete(
    store: &greppy_store::Store,
    project: &str,
    graph_generation: u64,
    model_id: &str,
) -> bool {
    let key = embedding_complete_key(project);
    store
        .conn()
        .query_row(
            "SELECT value FROM schema_meta WHERE key = ?1",
            [&key],
            |row| row.get::<_, String>(0),
        )
        .ok()
        == Some(format!("{graph_generation}|{model_id}"))
}

pub(crate) fn embedding_progress_value(
    root: &std::path::Path,
    cfg: &EmbeddingModelConfig,
    graph_generation: u64,
) -> serde_json::Value {
    if let Some(mut job) = read_background_job(&background_job_path(root)) {
        let alive = job
            .get("pid")
            .and_then(serde_json::Value::as_u64)
            .and_then(|pid| u32::try_from(pid).ok())
            .is_some_and(process_is_alive);
        job["alive"] = serde_json::json!(alive);
        job["graph_generation"] = serde_json::json!(graph_generation);
        return job;
    }

    let (backend, device) = embedding_backend_plan(cfg);
    let total_spans = current_embedding_candidate_count(root);
    let eta_seconds = initial_embedding_eta_seconds(total_spans, &backend);
    let now = unix_now_secs_cli();
    serde_json::json!({
        "schema_version": BACKGROUND_JOB_SCHEMA_VERSION,
        "kind": "embedding",
        "state": "starting",
        "alive": false,
        "backend": backend,
        "device": device,
        "graph_generation": graph_generation,
        "completed_spans": 0,
        "total_spans": total_spans,
        "progress_milli_percent": 0,
        "rate_milli_spans_per_second": serde_json::Value::Null,
        "eta_seconds": eta_seconds,
        "eta_minutes": eta_seconds.map(|eta| eta.saturating_add(59) / 60),
        "eta_unix_secs": eta_seconds.map(|eta| now.saturating_add(eta)),
        "last_error": serde_json::Value::Null,
    })
}

pub(crate) fn embedding_progress_text(progress: &serde_json::Value) -> String {
    let backend = progress
        .get("backend")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("cpu");
    let completed = progress
        .get("completed_spans")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let total = progress
        .get("total_spans")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if let Some(eta) = progress
        .get("eta_seconds")
        .and_then(serde_json::Value::as_u64)
    {
        format!(
            "semantic index building — {completed}/{total} spans, ETA ~{} (backend {backend})",
            format_embedding_eta(eta)
        )
    } else {
        format!(
            "semantic index building — {completed}/{total} spans, ETA measuring (backend {backend})"
        )
    }
}

pub(crate) fn inference_registry_status() -> Result<greppy_embed_native::InferenceBackendRegistry> {
    let cli = cli_inference_override();
    let no_gpu = cli.no_gpu || env_bool(ENV_NO_GPU)?;
    let configured = cli.device.or_else(|| env_nonempty(ENV_DEVICE));
    let policy = greppy_embed_native::InferencePolicy::from_selector(configured.as_deref(), no_gpu)
        .map_err(|error| Error::Invalid(error.to_string()))?;
    Ok(greppy_embed_native::InferenceBackendRegistry::probe_policy(
        &policy,
        combined_inference_gpu_memory(),
    ))
}

pub(crate) fn inference_model_status() -> serde_json::Value {
    let embedding_args = EmbeddingCliArgs {
        device: None,
        no_gpu: false,
    };
    let embedding = match embedding_config_optional(embedding_args) {
        Ok(Some(cfg)) => {
            let EmbeddingModelSource::Gguf { gguf, tokenizer } = cfg.source;
            serde_json::json!({
                "model_id": cfg.model_id,
                "format": "gguf-q4k",
                "embedded": cached_model_digest(&gguf).is_some(),
                "model_sha256": model_file_digest(&gguf).ok(),
                "tokenizer_sha256": model_file_digest(&tokenizer).ok(),
                "model_bytes": std::fs::metadata(&gguf).ok().map(|metadata| metadata.len()),
                "prompt_version": greppy_embed_native::PROMPT_VERSION,
                "task_profile": greppy_embed_native::CODE_RETRIEVAL_PROFILE,
            })
        }
        Ok(None) => serde_json::json!({
            "model_id": DEFAULT_EMBEDDINGGEMMA_MODEL_ID,
            "format": "gguf-q4k",
            "embedded": true,
            "model_sha256": env!("GREPPY_EMBEDDED_GGUF_SHA"),
            "tokenizer_sha256": env!("GREPPY_EMBEDDED_TOK_SHA"),
            "runtime_state": "not_loaded",
            "prompt_version": greppy_embed_native::PROMPT_VERSION,
            "task_profile": greppy_embed_native::CODE_RETRIEVAL_PROFILE,
        }),
        Err(error) => serde_json::json!({"state": "faulted", "last_error": error.to_string()}),
    };
    let summary = match qwen_summary_config_optional() {
        Ok(Some(cfg)) => serde_json::json!({
            "model_id": cfg.model_id,
            "format": "gguf-q4-k-m-mtp",
            "embedded": cached_model_digest(&cfg.gguf).is_some(),
            "model_sha256": model_file_digest(&cfg.gguf).ok(),
            "tokenizer_sha256": model_file_digest(&cfg.tokenizer).ok(),
            "model_bytes": std::fs::metadata(&cfg.gguf).ok().map(|metadata| metadata.len()),
            "prompt_version": greppy_qwen35_native::PROMPT_VERSION,
        }),
        Ok(None) => serde_json::json!({
            "model_id": greppy_qwen35_native::MODEL_ID,
            "format": "gguf-q4-k-m-mtp",
            "embedded": true,
            "model_sha256": env!("GREPPY_EMBEDDED_QWEN35_GGUF_SHA"),
            "tokenizer_sha256": env!("GREPPY_EMBEDDED_QWEN35_TOK_SHA"),
            "runtime_state": "not_loaded",
            "prompt_version": greppy_qwen35_native::PROMPT_VERSION,
        }),
        Err(error) => serde_json::json!({"state": "faulted", "last_error": error.to_string()}),
    };
    serde_json::json!({"embedding": embedding, "summary": summary})
}

pub(crate) fn inference_daemon_status() -> serde_json::Value {
    #[cfg(any(unix, windows))]
    {
        let embedding_args = EmbeddingCliArgs {
            device: None,
            no_gpu: false,
        };
        let embedding = match embedding_config_optional(embedding_args) {
            Ok(Some(cfg)) => {
                let key = embedding_query_cache_key(&cfg);
                embed_daemon::status(&cfg, &key)
            }
            Ok(None) => serde_json::json!({"state": "unavailable"}),
            Err(error) => serde_json::json!({"state": "faulted", "last_error": error.to_string()}),
        };
        let summary = match qwen_summary_config_optional() {
            Ok(Some(cfg)) => {
                let key = qwen_summary_model_key(&cfg);
                summarize_daemon::status(&key)
            }
            Ok(None) => serde_json::json!({"state": "unavailable"}),
            Err(error) => serde_json::json!({"state": "faulted", "last_error": error.to_string()}),
        };
        serde_json::json!({"embedding": embedding, "summary": summary})
    }
    #[cfg(not(any(unix, windows)))]
    {
        serde_json::json!({
            "embedding": {"state": "unsupported"},
            "summary": {"state": "unsupported"},
        })
    }
}

pub(crate) fn embedding_asset_missing_error(error: &Error) -> bool {
    matches!(error, Error::Config(message) if message.contains("EmbeddingGemma assets are unavailable"))
}

pub(crate) fn embedding_config_for_index(
    args: EmbeddingCliArgs<'_>,
) -> Result<Option<EmbeddingModelConfig>> {
    if test_inference_skipped() {
        return Ok(None);
    }
    Ok(Some(embedding_config_required(args)?))
}

pub(crate) fn embedding_config_for_required_use(
    args: EmbeddingCliArgs<'_>,
) -> Result<EmbeddingModelConfig> {
    embedding_config_required(args)
}

/// Resolve the mandatory embedded EmbeddingGemma model. The `Option` remains
/// only for non-fatal query paths that predate the always-embedded contract.
pub(crate) fn embedding_config_optional(
    args: EmbeddingCliArgs<'_>,
) -> Result<Option<EmbeddingModelConfig>> {
    if test_inference_skipped() {
        return Ok(None);
    }
    embedding_config_required(args).map(Some)
}

pub(crate) fn qwen_summary_config_optional() -> Result<Option<QwenSummaryConfig>> {
    if test_inference_skipped() {
        return Ok(None);
    }
    let Some((gguf, tokenizer)) = qwen35_assets::paths() else {
        return Ok(None);
    };
    Ok(Some(QwenSummaryConfig {
        model_id: greppy_qwen35_native::MODEL_ID.to_string(),
        gguf: gguf.into(),
        tokenizer: tokenizer.into(),
        device: qwen_summary_device_preference()?,
    }))
}

pub(crate) fn qwen_summary_device_preference() -> Result<greppy_qwen35_native::DevicePreference> {
    let cli = cli_inference_override();
    if cli.no_gpu || env_bool(ENV_NO_GPU)? {
        return Ok(greppy_qwen35_native::DevicePreference::Cpu);
    }
    let raw = cli
        .device
        .or_else(|| env_nonempty(ENV_DEVICE))
        .unwrap_or_else(|| "auto".to_string());
    greppy_qwen35_native::DevicePreference::parse(&raw).map_err(|e| Error::Invalid(e.to_string()))
}

pub(crate) fn qwen_summary_model_key(cfg: &QwenSummaryConfig) -> String {
    format!(
        "{}:{}:{}:{}:{}:{}:{}",
        cfg.model_id,
        greppy_qwen35_native::PROMPT_VERSION,
        greppy_qwen35_native::TRIAGE_PROMPT_VERSION,
        greppy_qwen35_native::BRIEF_FILTER_VERSION,
        inference_device_identity(&cfg.device),
        model_file_digest(&cfg.gguf).unwrap_or_else(|_| "unknown".into()),
        model_file_digest(&cfg.tokenizer).unwrap_or_else(|_| "unknown".into())
    )
}

pub(crate) fn embedding_config_required(
    args: EmbeddingCliArgs<'_>,
) -> Result<EmbeddingModelConfig> {
    #[cfg(debug_assertions)]
    if std::env::var_os(ENV_TEST_EMBED_ASSET_MISSING).is_some() {
        return Err(Error::Config(
            "embedded EmbeddingGemma assets are unavailable (test failpoint)".into(),
        ));
    }
    let device = embedding_device_preference(args.device, args.no_gpu)?;
    if test_inference_skipped() {
        return Ok(EmbeddingModelConfig {
            model_id: embedded_embedding_model_id(),
            source: EmbeddingModelSource::Gguf {
                gguf: "__greppy_test_skip_inference__.gguf".into(),
                tokenizer: "__greppy_test_skip_inference__.tokenizer.json".into(),
            },
            max_length: None,
            device,
        });
    }
    let source = match embeddinggemma_assets::paths() {
        Some((gguf, tokenizer)) => EmbeddingModelSource::Gguf {
            gguf: gguf.into(),
            tokenizer: tokenizer.into(),
        },
        None => {
            return Err(Error::Config(
                "embedded EmbeddingGemma assets are unavailable".into(),
            ))
        }
    };
    let source_digest = embedding_source_content_digest(&source)?;
    Ok(EmbeddingModelConfig {
        model_id: format!("{DEFAULT_EMBEDDINGGEMMA_MODEL_ID}@sha256:{source_digest}"),
        source,
        max_length: None,
        device,
    })
}

fn embedded_embedding_model_id() -> String {
    use sha2::{Digest, Sha256};

    let mut combined = Sha256::new();
    for (name, digest) in [
        (
            "embeddinggemma-300M-Q4_K.gguf",
            env!("GREPPY_EMBEDDED_GGUF_SHA"),
        ),
        ("tokenizer.json", env!("GREPPY_EMBEDDED_TOK_SHA")),
    ] {
        combined.update(name.as_bytes());
        combined.update([0]);
        combined.update(digest.as_bytes());
        combined.update([0]);
    }
    let digest = combined.finalize();
    format!("{DEFAULT_EMBEDDINGGEMMA_MODEL_ID}@sha256:{digest:x}")
}

pub(crate) fn embedding_source_content_digest(source: &EmbeddingModelSource) -> Result<String> {
    use sha2::{Digest, Sha256};

    let EmbeddingModelSource::Gguf { gguf, tokenizer } = source;
    let paths = vec![gguf.clone(), tokenizer.clone()];
    let mut combined = Sha256::new();
    for path in paths {
        let digest = model_file_digest(&path)
            .map_err(|error| Error::io(format!("digest model file {}", path.display()), error))?;
        combined.update(path.file_name().unwrap_or_default().as_encoded_bytes());
        combined.update([0]);
        combined.update(digest.as_bytes());
        combined.update([0]);
    }
    Ok(format!("{:x}", combined.finalize()))
}

pub(crate) fn embedding_device_preference(
    cli_device: Option<&str>,
    cli_no_gpu: bool,
) -> Result<greppy_embed_native::DevicePreference> {
    if cli_no_gpu || env_bool(ENV_NO_GPU)? {
        return Ok(greppy_embed_native::DevicePreference::Cpu);
    }
    let raw = cli_device
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| env_nonempty(ENV_DEVICE))
        .unwrap_or_else(|| "auto".to_string());
    raw.parse::<greppy_embed_native::DevicePreference>()
        .map_err(|e| Error::Invalid(e.to_string()))
}

/// Cache key for query embeddings: logical model id + prompt/task contract +
/// content digests. A same-size/same-mtime model replacement cannot reuse a
/// vector computed by different weights.
pub(crate) fn embedding_query_cache_key(cfg: &EmbeddingModelConfig) -> String {
    fn file_fp(path: &std::path::Path) -> String {
        model_file_digest(path).unwrap_or_else(|_| format!("{}:unknown", path.display()))
    }
    let EmbeddingModelSource::Gguf { gguf, tokenizer } = &cfg.source;
    let source_fp = format!("gguf;{};{}", file_fp(gguf), file_fp(tokenizer));
    format!(
        "{}|{}|{}|{}",
        cfg.model_id,
        greppy_embed_native::PROMPT_VERSION,
        greppy_search::EMBEDDINGGEMMA_CODE_RETRIEVAL_PROFILE,
        source_fp
    )
}

pub(crate) fn model_file_digest(path: &std::path::Path) -> std::io::Result<String> {
    if let Some(digest) = cached_model_digest(path) {
        return Ok(digest);
    }
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub(crate) fn embedding_complete_key(project: &str) -> String {
    format!("embedding_complete:{project}")
}
