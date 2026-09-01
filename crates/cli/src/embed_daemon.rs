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
const CLIENT_READ_TIMEOUT: Duration = Duration::from_secs(300);
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
    let request = serde_json::json!({
        "op": "query",
        "pv": greppy_embed_native::PROMPT_VERSION,
        "mk": model_key,
        "text": text,
    });
    match request_via_daemon(cfg, model_key, request) {
        RequestOutcome::Response(response) => {
            let Some(vector) = decode_vector(response.get("v_bits")) else {
                return EmbedDaemonResult::Failed;
            };
            EmbedDaemonResult::Embedded(vector)
        }
        RequestOutcome::DaemonBusy => EmbedDaemonResult::DaemonBusy,
        RequestOutcome::NoDaemon => EmbedDaemonResult::NoDaemon,
        RequestOutcome::Failed => EmbedDaemonResult::Failed,
    }
}

fn request_via_daemon(
    cfg: &super::EmbeddingModelConfig,
    model_key: &str,
    request: serde_json::Value,
) -> RequestOutcome<serde_json::Value> {
    let Some(endpoint) = endpoint(cfg, model_key) else {
        return RequestOutcome::NoDaemon;
    };
    match request_json(&endpoint, request.clone()) {
        RequestOutcome::Response(response) => return RequestOutcome::Response(response),
        RequestOutcome::DaemonBusy => return RequestOutcome::DaemonBusy,
        RequestOutcome::Failed => return RequestOutcome::Failed,
        RequestOutcome::NoDaemon => {}
    }

    let spawn_outcome =
        inference_daemon::spawn_once(&endpoint, || spawn_daemon(cfg, &endpoint, false));
    for delay in inference_daemon::retry_delays() {
        std::thread::sleep(delay);
        match request_json(&endpoint, request.clone()) {
            RequestOutcome::Response(response) => return RequestOutcome::Response(response),
            RequestOutcome::DaemonBusy => return RequestOutcome::DaemonBusy,
            RequestOutcome::Failed => return RequestOutcome::Failed,
            RequestOutcome::NoDaemon => {}
        }
    }
    inference_daemon::record_spawn_failure(&endpoint, spawn_outcome.attempted());
    // Never infer from one client's failed probe that it is the only process
    // on the machine. Parallel agents can all observe the same startup gap;
    // an in-process fallback at this point creates one multi-gigabyte model
    // per client. The caller reports the unavailable shared daemon instead.
    match spawn_outcome {
        SpawnOutcome::SpawnFailed | SpawnOutcome::Spawned | SpawnOutcome::Cooldown => {
            RequestOutcome::NoDaemon
        }
        SpawnOutcome::Contended => RequestOutcome::DaemonBusy,
    }
}

fn request_json(
    endpoint: &Endpoint,
    request: serde_json::Value,
) -> RequestOutcome<serde_json::Value> {
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
            RequestOutcome::Response(response)
        }
        RequestOutcome::NoDaemon => RequestOutcome::NoDaemon,
        RequestOutcome::DaemonBusy => RequestOutcome::DaemonBusy,
        RequestOutcome::Failed => RequestOutcome::Failed,
    }
}

fn decode_vector(value: Option<&serde_json::Value>) -> Option<Vec<f32>> {
    let values = value?.as_array()?;
    let vector = values
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|bits| u32::try_from(bits).ok())
                .map(f32::from_bits)
        })
        .collect::<Option<Vec<_>>>()?;
    (!vector.is_empty()).then_some(vector)
}

/// Indexing provider backed exclusively by the user-scoped embedding daemon.
/// The client never maps or loads model assets itself.
pub(super) struct DaemonCodeEmbeddingProvider<'a> {
    cfg: &'a super::EmbeddingModelConfig,
    model_key: String,
    cache: Option<greppy_store::EmbeddingContentCache>,
    // Tokenization is deliberately local: the tokenizer is small and does not
    // own model weights. Sending one daemon round-trip for every candidate
    // span made first-use indexing issue tens of thousands of serialized
    // requests before the first embedding batch could run.
    tokenizer: Option<greppy_embed_native::PromptTokenizer>,
    content_cache_hits: usize,
    content_cache_misses: usize,
}

impl<'a> DaemonCodeEmbeddingProvider<'a> {
    pub(super) fn new(cfg: &'a super::EmbeddingModelConfig) -> Self {
        let tokenizer = match &cfg.source {
            super::EmbeddingModelSource::Gguf { tokenizer, .. } => {
                greppy_embed_native::PromptTokenizer::from_file(
                    tokenizer,
                    greppy_embed_native::TokenizerConfig {
                        max_length: cfg
                            .max_length
                            .unwrap_or(greppy_embed_native::tokenizer::DEFAULT_MAX_LENGTH),
                        ..greppy_embed_native::TokenizerConfig::default()
                    },
                )
                .ok()
            }
        };
        Self {
            cfg,
            model_key: super::embedding_query_cache_key(cfg),
            cache: greppy_store::EmbeddingContentCache::open_global().ok(),
            tokenizer,
            content_cache_hits: 0,
            content_cache_misses: 0,
        }
    }

    #[cfg(test)]
    fn with_cache(
        cfg: &'a super::EmbeddingModelConfig,
        cache: greppy_store::EmbeddingContentCache,
    ) -> Self {
        Self {
            cfg,
            model_key: super::embedding_query_cache_key(cfg),
            cache: Some(cache),
            tokenizer: None,
            content_cache_hits: 0,
            content_cache_misses: 0,
        }
    }

    pub(super) fn backend_name(&self) -> String {
        format!(
            "shared-daemon:{}",
            super::inference_device_identity(&self.cfg.device)
        )
    }

    fn prompt_input(title: Option<&str>, content: &str) -> String {
        greppy_embed_native::EmbedTask::document_with_title(title, content)
    }

    fn cache_key(title: Option<&str>, content: &str) -> String {
        greppy_store::EmbeddingContentCache::input_sha256(&Self::prompt_input(title, content))
    }

    fn daemon_error<T>(&self, outcome: RequestOutcome<T>) -> greppy_core::Error {
        let detail = match outcome {
            RequestOutcome::DaemonBusy => "shared embedding daemon is busy",
            RequestOutcome::NoDaemon => "shared embedding daemon is unavailable",
            RequestOutcome::Failed => "shared embedding daemon request failed",
            RequestOutcome::Response(_) => "shared embedding daemon returned malformed data",
        };
        greppy_core::Error::Store(format!(
            "{detail}; no in-process model fallback was attempted"
        ))
    }
}

impl greppy_indexer::CodeEmbeddingProvider for DaemonCodeEmbeddingProvider<'_> {
    fn model_id(&self) -> &str {
        &self.cfg.model_id
    }

    fn prompt_version(&self) -> &str {
        greppy_embed_native::PROMPT_VERSION
    }

    fn task_profile(&self) -> &str {
        greppy_embed_native::CODE_RETRIEVAL_PROFILE
    }

    fn max_input_tokens(&self) -> usize {
        self.cfg
            .max_length
            .unwrap_or(greppy_embed_native::tokenizer::DEFAULT_MAX_LENGTH)
    }

    fn document_token_len(&self, title: Option<&str>, content: &str) -> greppy_core::Result<usize> {
        let key = Self::cache_key(title, content);
        if let Some(hit) = self.cache.as_ref().and_then(|cache| {
            cache
                .get(
                    &self.model_key,
                    self.prompt_version(),
                    self.task_profile(),
                    &key,
                )
                .ok()
                .flatten()
        }) {
            return Ok(hit.token_len);
        }
        let token_len = if let Some(tokenizer) = &self.tokenizer {
            tokenizer
                .token_len(&Self::prompt_input(title, content))
                .map_err(|error| {
                    greppy_core::Error::Store(format!(
                        "embedding tokenizer failed before daemon inference: {error}"
                    ))
                })?
        } else {
            let request = serde_json::json!({
                "op": "document_token_len",
                "pv": self.prompt_version(),
                "mk": self.model_key,
                "title": title,
                "text": content,
            });
            match request_via_daemon(self.cfg, &self.model_key, request) {
                RequestOutcome::Response(response) => response
                    .get("token_len")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or_else(|| self.daemon_error(RequestOutcome::<()>::Failed))?,
                outcome => return Err(self.daemon_error(outcome)),
            }
        };
        if let Some(cache) = &self.cache {
            let _ = cache.put_token_len(
                &self.model_key,
                self.prompt_version(),
                self.task_profile(),
                &key,
                token_len,
            );
        }
        Ok(token_len)
    }

    fn embed_code_document(
        &mut self,
        title: Option<&str>,
        content: &str,
    ) -> greppy_core::Result<Vec<f32>> {
        self.embed_code_documents(&[(title, content)])?
            .into_iter()
            .next()
            .ok_or_else(|| greppy_core::Error::Store("embedding daemon returned no vector".into()))
    }

    fn embed_code_documents(
        &mut self,
        docs: &[(Option<&str>, &str)],
    ) -> greppy_core::Result<Vec<Vec<f32>>> {
        let mut output = vec![None; docs.len()];
        let mut missing = Vec::new();
        let keys = docs
            .iter()
            .map(|(title, content)| Self::cache_key(*title, content))
            .collect::<Vec<_>>();
        let cached = if let Some(cache) = &mut self.cache {
            cache
                .get_many(
                    &self.model_key,
                    greppy_embed_native::PROMPT_VERSION,
                    greppy_embed_native::CODE_RETRIEVAL_PROFILE,
                    &keys,
                )
                .unwrap_or_else(|_| vec![None; docs.len()])
        } else {
            vec![None; docs.len()]
        };
        for (index, ((title, content), (key, hit))) in
            docs.iter().zip(keys.into_iter().zip(cached)).enumerate()
        {
            if let Some(vector) = hit.and_then(|hit| hit.vector) {
                output[index] = Some(vector);
                self.content_cache_hits = self.content_cache_hits.saturating_add(1);
            } else {
                self.content_cache_misses = self.content_cache_misses.saturating_add(1);
                missing.push((index, key, *title, *content));
            }
        }
        if !missing.is_empty() {
            let documents = missing
                .iter()
                .map(|(_, _, title, text)| serde_json::json!({"title": title, "text": text}))
                .collect::<Vec<_>>();
            let request = serde_json::json!({
                "op": "documents",
                "pv": self.prompt_version(),
                "mk": self.model_key,
                "documents": documents,
            });
            let response = match request_via_daemon(self.cfg, &self.model_key, request) {
                RequestOutcome::Response(response) => response,
                outcome => return Err(self.daemon_error(outcome)),
            };
            let vectors = response
                .get("vectors_bits")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| self.daemon_error(RequestOutcome::<()>::Failed))?;
            let token_lens = response
                .get("token_lens")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| self.daemon_error(RequestOutcome::<()>::Failed))?;
            if vectors.len() != missing.len() || token_lens.len() != missing.len() {
                return Err(self.daemon_error(RequestOutcome::<()>::Failed));
            }
            let mut cache_entries = Vec::with_capacity(missing.len());
            for (((index, key, _, _), vector), token_len) in
                missing.into_iter().zip(vectors).zip(token_lens)
            {
                let vector = decode_vector(Some(vector))
                    .ok_or_else(|| self.daemon_error(RequestOutcome::<()>::Failed))?;
                let token_len = token_len
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or_else(|| self.daemon_error(RequestOutcome::<()>::Failed))?;
                cache_entries.push((key, token_len, vector.clone()));
                output[index] = Some(vector);
            }
            if let Some(cache) = &self.cache {
                let _ = cache.put_vectors(
                    &self.model_key,
                    self.prompt_version(),
                    self.task_profile(),
                    &cache_entries,
                );
            }
        }
        output
            .into_iter()
            .map(|vector| {
                vector.ok_or_else(|| {
                    greppy_core::Error::Store("embedding daemon omitted a document vector".into())
                })
            })
            .collect()
    }

    fn content_cache_stats(&self) -> greppy_indexer::EmbeddingProviderCacheStats {
        greppy_indexer::EmbeddingProviderCacheStats {
            hits: self.content_cache_hits,
            misses: self.content_cache_misses,
        }
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
    inference_daemon::spawn_detached(&mut command).ok()
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
        hard_request_timeout: Some(Duration::from_secs(330)),
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
    match request.get("op").and_then(serde_json::Value::as_str) {
        Some("query") | Some("document_token_len") => {
            if request
                .get("text")
                .and_then(serde_json::Value::as_str)
                .is_none()
            {
                return Err(serde_json::json!({"error": "missing text"}));
            }
        }
        Some("documents") => {
            if request
                .get("documents")
                .and_then(serde_json::Value::as_array)
                .is_none_or(|documents| documents.is_empty())
            {
                return Err(serde_json::json!({"error": "missing documents"}));
            }
        }
        _ => return Err(serde_json::json!({"error": "unsupported embedding operation"})),
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
    let Some(loaded) = model.as_ref() else {
        return serde_json::json!({"error": "model unavailable"});
    };
    let result = match request.get("op").and_then(serde_json::Value::as_str) {
        Some("query") => request
            .get("text")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| "missing text".to_string())
            .and_then(|text| {
                greppy_search::embed_code_query(loaded, text).map_err(|e| e.to_string())
            })
            .map(|vector| {
                serde_json::json!({
                    "v_bits": vector.iter().map(|value| value.to_bits()).collect::<Vec<_>>()
                })
            }),
        Some("document_token_len") => {
            let title = request.get("title").and_then(serde_json::Value::as_str);
            request
                .get("text")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| "missing text".to_string())
                .and_then(|text| {
                    loaded
                        .document_token_len(title, text)
                        .map_err(|e| e.to_string())
                })
                .map(|token_len| serde_json::json!({"token_len": token_len}))
        }
        Some("documents") => {
            let values = request
                .get("documents")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| "missing documents".to_string());
            values.and_then(|values| {
                let docs = values
                    .iter()
                    .map(|value| {
                        let title = value.get("title").and_then(serde_json::Value::as_str);
                        let text = value
                            .get("text")
                            .and_then(serde_json::Value::as_str)
                            .ok_or_else(|| "document missing text".to_string())?;
                        Ok((title, text))
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let token_lens = docs
                    .iter()
                    .map(|(title, text)| {
                        loaded
                            .document_token_len(*title, text)
                            .map_err(|e| e.to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let vectors = loaded.embed_documents(&docs).map_err(|e| e.to_string())?;
                Ok(serde_json::json!({
                    "vectors_bits": vectors.iter().map(|vector| {
                        vector.iter().map(|value| value.to_bits()).collect::<Vec<_>>()
                    }).collect::<Vec<_>>(),
                    "token_lens": token_lens,
                }))
            })
        }
        _ => Err("unsupported embedding operation".to_string()),
    };
    match result {
        Ok(response) => response,
        Err(error) => {
            *model = None;
            serde_json::json!({"error": format!("embed: {error}")})
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use greppy_indexer::CodeEmbeddingProvider;

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

    #[test]
    fn indexing_clients_do_not_load_embedding_models_in_process() {
        let indexing = include_str!("indexing.rs");
        let bash_smart = include_str!("bash_smart.rs");
        assert!(!indexing.contains("load_embedding_model("));
        assert!(!bash_smart.contains("load_embedding_model("));
    }

    #[test]
    fn document_batch_contract_is_validated_before_model_loading() {
        let request = serde_json::json!({
            "op": "documents",
            "pv": greppy_embed_native::PROMPT_VERSION,
            "mk": "model-key",
            "documents": [{"title": "src/lib.rs:1 f", "text": "fn f() {}"}],
        });
        assert!(validate(&request.to_string(), "model-key").is_ok());
        let empty = serde_json::json!({
            "op": "documents",
            "pv": greppy_embed_native::PROMPT_VERSION,
            "mk": "model-key",
            "documents": [],
        });
        assert_eq!(
            validate(&empty.to_string(), "model-key").unwrap_err()["error"],
            "missing documents"
        );
    }

    #[test]
    fn cached_document_vector_is_reused_without_model_or_daemon() {
        let temp = tempfile::tempdir().unwrap();
        let cfg = super::super::EmbeddingModelConfig {
            model_id: "test-model".into(),
            source: super::super::EmbeddingModelSource::Gguf {
                gguf: temp.path().join("deliberately-missing.gguf"),
                tokenizer: temp.path().join("deliberately-missing-tokenizer.json"),
            },
            max_length: Some(128),
            device: greppy_embed_native::DevicePreference::Cpu,
        };
        let cache = greppy_store::EmbeddingContentCache::open(temp.path().join("cache")).unwrap();
        let title = "src/lib.rs:1-1 f";
        let content = "fn f() {}";
        let model_key = super::super::embedding_query_cache_key(&cfg);
        let input = DaemonCodeEmbeddingProvider::cache_key(Some(title), content);
        cache
            .put_vector(
                &model_key,
                greppy_embed_native::PROMPT_VERSION,
                greppy_embed_native::CODE_RETRIEVAL_PROFILE,
                &input,
                7,
                &[0.25, -0.5],
            )
            .unwrap();
        let mut provider = DaemonCodeEmbeddingProvider::with_cache(&cfg, cache);
        assert_eq!(
            provider
                .embed_code_documents(&[(Some(title), content)])
                .unwrap(),
            vec![vec![0.25, -0.5]]
        );
        assert_eq!(
            provider.document_token_len(Some(title), content).unwrap(),
            7
        );
        assert_eq!(
            provider.content_cache_stats(),
            greppy_indexer::EmbeddingProviderCacheStats { hits: 1, misses: 0 }
        );
    }

    #[test]
    fn indexing_token_lengths_do_not_require_a_daemon_round_trip() {
        let temp = tempfile::tempdir().unwrap();
        let tokenizer_path = temp.path().join("tokenizer.json");
        std::fs::write(
            &tokenizer_path,
            r#"{
                "version":"1.0",
                "truncation":null,
                "padding":null,
                "added_tokens":[],
                "normalizer":null,
                "pre_tokenizer":{"type":"Whitespace"},
                "post_processor":null,
                "decoder":null,
                "model":{"type":"WordLevel","vocab":{"[UNK]":0,"title":1,"text":2,"fn":3,"f":4},"unk_token":"[UNK]"}
            }"#,
        )
        .unwrap();
        let cfg = super::super::EmbeddingModelConfig {
            model_id: "test-model".into(),
            source: super::super::EmbeddingModelSource::Gguf {
                gguf: temp.path().join("deliberately-missing.gguf"),
                tokenizer: tokenizer_path,
            },
            max_length: Some(128),
            device: greppy_embed_native::DevicePreference::Cpu,
        };
        let provider = DaemonCodeEmbeddingProvider::new(&cfg);
        assert!(provider.tokenizer.is_some());
        assert!(
            provider
                .document_token_len(Some("src/lib.rs:1-1 f"), "fn f() {}")
                .unwrap()
                > 0
        );
    }
}
