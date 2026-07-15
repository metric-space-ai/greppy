use std::path::Path;

use tokenizers::Tokenizer;

use crate::cpu::CpuQwen35Model;
use crate::inventory::Qwen35Inventory;
use crate::postprocess::{postprocess_brief_output, postprocess_triage_output, TriageVerdict};
use crate::prompt::{brief_prompt, non_thinking_chat_prompt, triage_prompt};
use crate::sampler::{
    GenerationParams, BRIEF_FALLBACK_GENERATION_PARAMS, BRIEF_GENERATION_PARAMS,
    TRIAGE_GENERATION_PARAMS,
};
use crate::{
    DiagnosticMtpGeneration, DiagnosticTargetPrefill, Error, MtpGenerationOutput, Result,
    DIAGNOSTIC_MAX_OUTPUT_TOKENS, DIAGNOSTIC_TARGET_PREFILL_TOKENS,
};

pub use greppy_embed_native::DevicePreference;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadOptions {
    pub device: DevicePreference,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            device: DevicePreference::Auto,
        }
    }
}

pub struct Qwen35Summarizer {
    tokenizer: Tokenizer,
    inventory: Qwen35Inventory,
    backend: Backend,
}

enum Backend {
    Cpu(Box<CpuQwen35Model>),
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(crate::metal::model::MetalQwen35Model),
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    Cuda(crate::cuda::model::CudaQwen35Model),
}

impl Qwen35Summarizer {
    pub fn load_gguf(
        gguf_path: impl AsRef<Path>,
        tokenizer_json_path: impl AsRef<Path>,
        options: LoadOptions,
    ) -> Result<Self> {
        let gguf_path = gguf_path.as_ref();
        if matches!(
            options.device,
            DevicePreference::Metal | DevicePreference::Cuda
        ) {
            let selector = if options.device == DevicePreference::Cuda {
                std::env::var("GREPPY_QWEN35_CUDA_DEVICE")
                    .or_else(|_| std::env::var("EMBED_NATIVE_CUDA_DEVICE"))
                    .ok()
                    .map(|index| format!("cuda:{index}"))
                    .unwrap_or_else(|| "cuda".into())
            } else {
                "metal".into()
            };
            let policy =
                greppy_embed_native::InferencePolicy::from_selector(Some(&selector), false)?;
            greppy_embed_native::preflight_explicit_model(
                &policy,
                greppy_embed_native::InferenceModelKind::Qwen35,
                std::fs::metadata(gguf_path)?.len(),
            )?;
        }
        let model = greppy_embed_native::GgufModel::open(gguf_path)?;
        let inventory = Qwen35Inventory::from_gguf(&model)?;
        inventory.validate_core_tensors(&model)?;
        let tokenizer = Tokenizer::from_file(tokenizer_json_path.as_ref())
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let eos_token_id = tokenizer
            .token_to_id("<|im_end|>")
            .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
            .unwrap_or(248_044);
        let backend = load_backend(&model, inventory.clone(), eos_token_id, &options)?;
        Ok(Self {
            tokenizer,
            inventory,
            backend,
        })
    }

    /// Summarize one definition source span. `path` is the repo-relative
    /// file path of the span; it is part of the trained prompt contract and
    /// grounds path-derived symbols in the quality filter.
    pub fn summarize_source(&self, path: &str, source: &str) -> Result<Vec<String>> {
        if source.trim().is_empty() {
            return Ok(Vec::new());
        }
        let prompt = brief_prompt(path, source);
        let model_prompt = crate::prompt::brief_chat_prompt(path, source);
        let raw = self.generate_raw(&model_prompt, BRIEF_GENERATION_PARAMS)?;
        let bullets = postprocess_brief_output(&raw, &prompt);
        log_summary_debug("primary", &raw, &bullets);
        let bullets = filter_brief_bullets_by_quality(bullets, path, source);
        log_filtered_summary_debug("primary", &bullets);
        if !bullets.is_empty() {
            return Ok(bullets);
        }
        let fallback_raw = self.generate_raw(&model_prompt, BRIEF_FALLBACK_GENERATION_PARAMS)?;
        let fallback = postprocess_brief_output(&fallback_raw, &prompt);
        log_summary_debug("fallback", &fallback_raw, &fallback);
        let fallback = filter_brief_bullets_by_quality(fallback, path, source);
        log_filtered_summary_debug("fallback", &fallback);
        Ok(fallback)
    }

    pub fn triage_span(
        &self,
        query: &str,
        span_loc: &str,
        span_code: &str,
    ) -> Result<TriageVerdict> {
        if query.trim().is_empty() || span_code.trim().is_empty() {
            return Ok(TriageVerdict {
                read: true,
                reason: "uncertain relevant span".to_string(),
            });
        }
        let prompt = triage_prompt(query, span_loc, span_code);
        let model_prompt = non_thinking_chat_prompt(&prompt);
        let raw = self.generate_raw(&model_prompt, TRIAGE_GENERATION_PARAMS)?;
        let verdict = postprocess_triage_output(&raw, &prompt);
        Ok(conservative_triage_guard(
            query, span_loc, span_code, verdict,
        ))
    }

    pub fn token_len(&self, text: &str) -> Result<usize> {
        self.tokenizer
            .encode(text, false)
            .map(|e| e.len())
            .map_err(|e| Error::Tokenizer(e.to_string()))
    }

    pub fn inventory(&self) -> &Qwen35Inventory {
        &self.inventory
    }

    pub fn backend_name(&self) -> &'static str {
        match self.backend {
            Backend::Cpu(_) => "cpu-q4k",
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Backend::Metal(ref model) => model.backend_name(),
            #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
            Backend::Cuda(ref model) => model.backend_name(),
        }
    }

    /// Measure one exact 512-token target-model prefill on the selected production backend.
    ///
    /// Model loading, state allocation, and input validation happen outside the timed interval.
    pub fn diagnostic_target_prefill_512(
        &self,
        token_ids: &[u32],
    ) -> Result<DiagnosticTargetPrefill> {
        validate_diagnostic_token_ids(token_ids, &self.inventory)?;
        if token_ids.len() != DIAGNOSTIC_TARGET_PREFILL_TOKENS {
            return Err(Error::InvalidRequest(format!(
                "diagnostic target prefill requires exactly {DIAGNOSTIC_TARGET_PREFILL_TOKENS} token IDs, got {}",
                token_ids.len()
            )));
        }
        let elapsed = match &self.backend {
            Backend::Cpu(model) => model.diagnostic_target_prefill(token_ids)?,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Backend::Metal(model) => model.diagnostic_target_prefill(token_ids)?,
            #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
            Backend::Cuda(model) => model.diagnostic_target_prefill(token_ids)?,
        };
        Ok(DiagnosticTargetPrefill {
            input_tokens: token_ids.len(),
            elapsed,
        })
    }

    /// Run greedy, EOS-disabled generation through the production MTP algorithm.
    ///
    /// This diagnostic API returns committed token IDs and stage timings without changing the
    /// 64-token, EOS-aware `summarize_source` product contract.
    pub fn diagnostic_generate_mtp_greedy(
        &self,
        prompt_token_ids: &[u32],
        max_output_tokens: usize,
    ) -> Result<DiagnosticMtpGeneration> {
        validate_diagnostic_token_ids(prompt_token_ids, &self.inventory)?;
        if !(1..=DIAGNOSTIC_MAX_OUTPUT_TOKENS).contains(&max_output_tokens) {
            return Err(Error::InvalidRequest(format!(
                "diagnostic MTP generation output limit must be in 1..={DIAGNOSTIC_MAX_OUTPUT_TOKENS}, got {max_output_tokens}"
            )));
        }
        if !self.inventory.has_mtp() {
            return Err(Error::GenerationUnavailable(
                "diagnostic MTP generation requires an MTP model layer".into(),
            ));
        }
        let required_context = prompt_token_ids
            .len()
            .checked_add(max_output_tokens)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| Error::InvalidRequest("diagnostic context length overflow".into()))?;
        if required_context > self.inventory.context_length {
            return Err(Error::InvalidRequest(format!(
                "diagnostic MTP generation needs {required_context} context tokens, model limit is {}",
                self.inventory.context_length
            )));
        }
        let params = diagnostic_greedy_params(max_output_tokens);
        let output = match &self.backend {
            Backend::Cpu(model) => model.diagnostic_generate_mtp(prompt_token_ids, params)?,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Backend::Metal(model) => model.diagnostic_generate_mtp(prompt_token_ids, params)?,
            #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
            Backend::Cuda(model) => model.diagnostic_generate_mtp(prompt_token_ids, params)?,
        };
        finish_diagnostic_generation(output, max_output_tokens)
    }

    fn generate_raw(
        &self,
        prompt: &str,
        params: crate::sampler::GenerationParams,
    ) -> Result<String> {
        match &self.backend {
            Backend::Cpu(model) => model.generate(&self.tokenizer, prompt, params),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Backend::Metal(model) => model.generate(&self.tokenizer, prompt, params),
            #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
            Backend::Cuda(model) => model.generate(&self.tokenizer, prompt, params),
        }
    }
}

fn diagnostic_greedy_params(max_tokens: usize) -> GenerationParams {
    GenerationParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 1,
        min_p: 0.0,
        presence_penalty: 0.0,
        repetition_penalty: 1.0,
        max_tokens,
    }
}

fn validate_diagnostic_token_ids(token_ids: &[u32], inventory: &Qwen35Inventory) -> Result<()> {
    if token_ids.is_empty() {
        return Err(Error::InvalidRequest(
            "diagnostic inference requires at least one token ID".into(),
        ));
    }
    if let Some(token_id) = token_ids
        .iter()
        .copied()
        .find(|token_id| usize::try_from(*token_id).map_or(true, |id| id >= inventory.vocab_size))
    {
        return Err(Error::InvalidRequest(format!(
            "diagnostic input token ID {token_id} is outside vocabulary size {}",
            inventory.vocab_size
        )));
    }
    Ok(())
}

fn finish_diagnostic_generation(
    output: MtpGenerationOutput,
    expected_tokens: usize,
) -> Result<DiagnosticMtpGeneration> {
    if output.token_ids.len() != expected_tokens {
        return Err(Error::GenerationUnavailable(format!(
            "diagnostic MTP generation committed {} of {expected_tokens} requested tokens",
            output.token_ids.len()
        )));
    }
    let mut telemetry = output.telemetry.ok_or_else(|| {
        Error::GenerationUnavailable("diagnostic MTP telemetry was not captured".into())
    })?;
    telemetry.output_token_ids = output.token_ids;
    Ok(telemetry)
}

fn log_summary_debug(stage: &str, raw: &str, bullets: &[String]) {
    if std::env::var_os("GREPPY_QWEN35_SUMMARY_DEBUG").is_some() {
        eprintln!("qwen35-summary-debug {stage} raw={raw:?} bullets={bullets:?}");
    }
}

#[cfg(test)]
mod diagnostic_api_tests {
    use super::*;

    fn inventory() -> Qwen35Inventory {
        Qwen35Inventory {
            architecture: "qwen35".into(),
            block_count: 24,
            hidden_size: 1024,
            feed_forward_size: 3584,
            vocab_size: 16,
            attention_heads: 8,
            kv_heads: 2,
            head_dim: 256,
            value_dim: 256,
            rope_dim: 64,
            context_length: 1024,
            full_attention_interval: 4,
            ssm_inner_size: 2048,
            ssm_group_count: 16,
            ssm_time_step_rank: 16,
            nextn_predict_layers: 1,
        }
    }

    #[test]
    fn diagnostic_sampling_is_greedy_and_keeps_requested_limit() {
        let params = diagnostic_greedy_params(128);
        assert_eq!(params.temperature, 0.0);
        assert_eq!(params.top_k, 1);
        assert_eq!(params.max_tokens, 128);
    }

    #[test]
    fn diagnostic_token_validation_rejects_empty_and_out_of_vocab() {
        let inventory = inventory();
        assert!(validate_diagnostic_token_ids(&[], &inventory).is_err());
        assert!(validate_diagnostic_token_ids(&[0, 15], &inventory).is_ok());
        assert!(validate_diagnostic_token_ids(&[16], &inventory).is_err());
    }

    #[test]
    fn diagnostic_generation_requires_exact_committed_count() {
        let telemetry = DiagnosticMtpGeneration {
            output_token_ids: Vec::new(),
            target_prefill: std::time::Duration::from_nanos(1),
            mtp_prefill: std::time::Duration::from_nanos(2),
            decode: std::time::Duration::from_nanos(3),
            mtp: crate::DiagnosticMtpStats {
                used: true,
                cycles: 1,
                drafted_tokens: 2,
                accepted_tokens: 1,
                fallback: false,
            },
        };
        let output = MtpGenerationOutput {
            token_ids: vec![7, 8],
            telemetry: Some(telemetry.clone()),
        };
        let finished = finish_diagnostic_generation(output, 2).expect("exact output count");
        assert_eq!(finished.output_token_ids, [7, 8]);

        let short = MtpGenerationOutput {
            token_ids: vec![7],
            telemetry: Some(telemetry),
        };
        assert!(finish_diagnostic_generation(short, 2).is_err());
    }
}

fn log_filtered_summary_debug(stage: &str, bullets: &[String]) {
    if std::env::var_os("GREPPY_QWEN35_SUMMARY_DEBUG").is_some() {
        eprintln!("qwen35-summary-debug {stage} filtered={bullets:?}");
    }
}

fn conservative_triage_guard(
    query: &str,
    span_loc: &str,
    span_code: &str,
    verdict: TriageVerdict,
) -> TriageVerdict {
    if verdict.read || !query_overlaps_span(query, span_loc, span_code) {
        return verdict;
    }
    TriageVerdict {
        read: true,
        reason: "matches query terms".to_string(),
    }
}

#[cfg(test)]
fn brief_bullets_pass_quality(bullets: &[String], path: &str, source: &str) -> bool {
    if bullets.is_empty() {
        return false;
    }
    let path_lower = path.to_ascii_lowercase();
    let source_terms = lexical_terms(source).collect::<std::collections::BTreeSet<_>>();
    let source_identifiers = code_identifiers(source).collect::<std::collections::BTreeSet<_>>();
    bullets.iter().all(|bullet| {
        brief_bullet_passes_quality(bullet, &path_lower, &source_terms, &source_identifiers)
    })
}

fn filter_brief_bullets_by_quality(bullets: Vec<String>, path: &str, source: &str) -> Vec<String> {
    let path_lower = path.to_ascii_lowercase();
    let source_terms = lexical_terms(source).collect::<std::collections::BTreeSet<_>>();
    let source_identifiers = code_identifiers(source).collect::<std::collections::BTreeSet<_>>();
    bullets
        .into_iter()
        .filter(|bullet| {
            brief_bullet_passes_quality(bullet, &path_lower, &source_terms, &source_identifiers)
        })
        .take(2)
        .collect()
}

/// One bullet passes when nothing in it is ungrounded. Grounding is
/// asymmetric, mirroring the benchmark judge: the prompt shows the model the
/// repo-relative path AND the source, so a symbol that occurs in either one
/// (case-insensitive substring for the path, like the existing source check)
/// counts as grounded, never as a hallucination.
fn brief_bullet_passes_quality(
    bullet: &str,
    path_lower: &str,
    source_terms: &std::collections::BTreeSet<String>,
    source_identifiers: &std::collections::BTreeSet<String>,
) -> bool {
    let lower = bullet.to_ascii_lowercase();
    if bullet.trim_end().ends_with('?') {
        return false;
    }
    if bullet.contains("...") || bullet.contains('…') {
        return false;
    }
    if [
        "does it do",
        "advanced c language",
        "for example",
        "here is",
        "infinite list",
        "in the context",
        "list comprehension",
        "object-oriented programming framework",
        "specific case",
        "stale time",
        "internal function in rust",
        "syntax provided",
        "not a library",
        "not a module",
        "compiler",
        "this snippet",
        "this code snippet",
        "which is used",
        "exact format",
        "raw strings",
    ]
    .iter()
    .any(|phrase| lower.contains(phrase))
    {
        return false;
    }
    for language in [
        "c++",
        "c#",
        "dart",
        "go",
        "java",
        "javascript",
        "kotlin",
        "python",
        "ruby",
        "rust",
        "scala",
        "swift",
        "typescript",
        "zig",
    ] {
        let mentions_language = lower
            .split(|c: char| !(c == '#' || c == '+' || c.is_ascii_alphanumeric()))
            .any(|word| word == language);
        let source_mentions_language = source_terms.contains(language)
            || source_identifiers
                .iter()
                .any(|identifier| identifier == language)
            || path_lower
                .split(|c: char| !(c == '#' || c == '+' || c.is_ascii_alphanumeric()))
                .any(|word| word == language);
        if mentions_language && !source_mentions_language {
            return false;
        }
    }
    let last_word = lower
        .trim_end_matches(|c: char| c.is_ascii_punctuation() || c.is_whitespace())
        .split_whitespace()
        .last()
        .unwrap_or("");
    if [
        "a", "an", "and", "are", "by", "for", "in", "is", "of", "or", "that", "the", "to", "used",
        "which", "with",
    ]
    .contains(&last_word)
    {
        return false;
    }
    let source_mentions_atomic = source_terms.contains("atomic")
        || source_identifiers
            .iter()
            .any(|ident| ident.to_ascii_lowercase().contains("atomic"))
        || path_lower.contains("atomic");
    if lower.contains("atomic") && !source_mentions_atomic {
        return false;
    }
    if code_identifiers(bullet)
        .filter(|ident| ident.contains('_'))
        .any(|ident| !source_identifiers.contains(&ident) && !path_lower.contains(ident.as_str()))
    {
        return false;
    }
    if bullet_symbol_identifiers(bullet).into_iter().any(|ident| {
        !source_supports_term(&ident, source_terms, source_identifiers)
            && !path_lower.contains(ident.as_str())
    }) {
        return false;
    }
    if bullet
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter_map(risky_claim_root)
        .any(|term| {
            !source_supports_term(term, source_terms, source_identifiers)
                && !path_lower.contains(term)
        })
    {
        return false;
    }
    // Relatedness check, relaxed for the finetuned model: it paraphrases, so
    // exact >=4-char term overlap rejects good briefs on short functions.
    // Accept when any >=3-char bullet term matches a source term, occurs as
    // a substring of a source identifier (finds "quiz" in QuizScreen,
    // "con" in connection), or occurs in the prompt's file path — the model
    // saw the path, so path-derived wording is grounded, not off-topic.
    if source_terms.is_empty() {
        return true;
    }
    let idents_lower: Vec<String> = source_identifiers
        .iter()
        .map(|i| i.to_ascii_lowercase())
        .collect();
    bullet
        .split(|c: char| !c.is_ascii_alphanumeric())
        .map(|t| t.to_ascii_lowercase())
        .filter(|t| t.len() >= 3 && !TRIAGE_STOP_TERMS.contains(&t.as_str()))
        .any(|t| {
            source_terms.contains(&t)
                || source_terms.iter().any(|s| s.contains(&t))
                || idents_lower.iter().any(|i| i.contains(&t))
                || path_lower.contains(&t)
        })
}

fn bullet_symbol_identifiers(text: &str) -> Vec<String> {
    text.split(|c: char| !(c == '_' || c == '$' || c.is_ascii_alphanumeric()))
        .enumerate()
        .filter_map(|(index, raw)| {
            let token = raw.trim_matches(['_', '$']);
            if token.len() < 2 {
                return None;
            }
            let mut chars = token.chars();
            let first = chars.next()?;
            let rest = chars.collect::<String>();
            let internal_upper = rest.chars().any(|c| c.is_ascii_uppercase());
            let all_upper = token.len() >= 2 && token.chars().all(|c| c.is_ascii_uppercase());
            let code_shaped = raw.contains('_')
                || raw.contains('$')
                || internal_upper
                || (index > 0
                    && first.is_ascii_uppercase()
                    && !rest.is_empty()
                    && rest.chars().all(|c| c.is_ascii_lowercase()))
                || all_upper;
            code_shaped.then(|| token.to_ascii_lowercase())
        })
        .collect()
}

fn risky_claim_root(term: &str) -> Option<&'static str> {
    match term.to_ascii_lowercase().as_str() {
        "atomic" => Some("atomic"),
        "database" | "databases" => Some("database"),
        "decode" | "decodes" | "decoded" | "decoding" => Some("decode"),
        "empty" => Some("empty"),
        "file" | "files" => Some("file"),
        "flag" | "flags" => Some("flag"),
        "log" | "logs" | "logged" | "logging" => Some("log"),
        "parse" | "parses" | "parsed" | "parsing" => Some("parse"),
        "persist" | "persists" | "persisted" | "persisting" => Some("persist"),
        "positive" => Some("positive"),
        "property" | "properties" => Some("property"),
        "reject" | "rejects" | "rejected" | "rejecting" => Some("reject"),
        "send" | "sends" | "sent" | "sending" => Some("send"),
        "serialize" | "serializes" | "serialized" | "serializing" => Some("serialize"),
        "signal" | "signals" => Some("signal"),
        "store" | "stores" | "stored" | "storing" => Some("store"),
        "string" | "strings" => Some("string"),
        "throw" | "throws" | "thrown" | "throwing" => Some("throw"),
        "traceback" | "tracebacks" => Some("traceback"),
        "unbox" | "unboxed" | "unboxing" => Some("unbox"),
        "validate" | "validates" | "validated" | "validating" => Some("validate"),
        "wasi" => Some("wasi"),
        "windows" => Some("windows"),
        "wrapper" | "wrappers" => Some("wrapper"),
        _ => None,
    }
}

fn source_supports_term(
    term: &str,
    source_terms: &std::collections::BTreeSet<String>,
    source_identifiers: &std::collections::BTreeSet<String>,
) -> bool {
    let term = term.to_ascii_lowercase();
    let prefix_len = term.len().min(4);
    let prefix = &term[..prefix_len];
    source_terms.contains(&term)
        || source_terms.iter().any(|source| source.contains(prefix))
        || source_identifiers
            .iter()
            .any(|source| source.contains(prefix))
}

fn query_overlaps_span(query: &str, span_loc: &str, span_code: &str) -> bool {
    let query_terms = lexical_terms(query).collect::<std::collections::BTreeSet<_>>();
    if query_terms.is_empty() {
        return false;
    }
    lexical_terms(span_loc)
        .chain(lexical_terms(span_code))
        .any(|term| query_terms.contains(&term))
}

fn lexical_terms(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|term| term.len() >= 4)
        .map(|term| term.to_ascii_lowercase())
        .filter(|term| !TRIAGE_STOP_TERMS.contains(&term.as_str()))
}

fn code_identifiers(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !(c == '_' || c.is_ascii_alphanumeric()))
        .map(|term| term.trim_matches('_'))
        .filter(|term| term.len() >= 3)
        .map(|term| term.to_ascii_lowercase())
}

const TRIAGE_STOP_TERMS: &[&str] = &[
    "about", "after", "also", "code", "does", "file", "from", "function", "have", "into", "lines",
    "span", "that", "their", "there", "this", "what", "when", "where", "which", "with",
];

#[cfg(test)]
mod triage_guard_tests {
    use super::*;

    #[test]
    fn brief_quality_rejects_observed_hallucination_shape() {
        let bullets = vec![
            "This function add_user is an infinite list comprehension generator.".to_string(),
            "Does it do anything useful?".to_string(),
        ];
        assert!(!brief_bullets_pass_quality(
            &bullets,
            "src/users.rs",
            "fn add_user(users: &mut Vec<String>, name: String) { users.push(name); }"
        ));
    }

    #[test]
    fn brief_quality_rejects_unknown_helper_identifier() {
        let bullets = vec![
            "The add_user function is part of the _get_all_users_in_group helper. It takes a list of names and...".to_string(),
        ];
        assert!(!brief_bullets_pass_quality(
            &bullets,
            "src/users.rs",
            "fn add_user(users: &mut Vec<String>, name: String) { users.push(name); }"
        ));
    }

    #[test]
    fn brief_quality_rejects_invented_camelcase_and_acronym_symbols() {
        for bullet in [
            "Creates a ZoodMiniISODateTime for the given params.",
            "Creates a template literal from $ZodParts.",
            "Evaluates a cubic BCP spline.",
        ] {
            assert!(!brief_bullets_pass_quality(
                &[bullet.to_string()],
                "src/spline.rs",
                "fn bcspline(x: f32) -> f32 { ZodMiniISODateTime::new(x) }"
            ));
        }
    }

    #[test]
    fn brief_quality_keeps_symbols_that_exist_in_source() {
        assert!(brief_bullets_pass_quality(
            &["Creates a ZodMiniISODateTime from ISO params.".to_string()],
            "src/datetime.rs",
            "fn datetime(params: ISOParams) -> ZodMiniISODateTime { ZodMiniISODateTime::new(params) }"
        ));
    }

    #[test]
    fn brief_quality_keeps_camelcase_symbol_grounded_in_path() {
        // The model sees the repo-relative path in its prompt, so a symbol
        // that only occurs in the path (case-insensitive substring) is
        // grounded, not a hallucination.
        assert!(brief_bullets_pass_quality(
            &["Creates a ZodMiniISODateTime from the given params.".to_string()],
            "packages/zod/src/ZodMiniISODateTime.ts",
            "export function datetime(params) { return build(params); }"
        ));
    }

    #[test]
    fn brief_quality_rejects_symbol_missing_from_source_and_path() {
        // Same bullet, but neither the source nor the path mentions the
        // symbol: still a hallucination.
        assert!(!brief_bullets_pass_quality(
            &["Creates a ZodMiniISODateTime from the given params.".to_string()],
            "src/utils.rs",
            "export function datetime(params) { return build(params); }"
        ));
    }

    #[test]
    fn brief_quality_keeps_snake_case_identifier_grounded_in_path() {
        assert!(brief_bullets_pass_quality(
            &["Registers the summarize_daemon request handler.".to_string()],
            "crates/cli/src/summarize_daemon.rs",
            "fn register(handler: Handler) { HANDLERS.push(handler); }"
        ));
    }

    #[test]
    fn brief_quality_rejects_unsupported_risky_claims() {
        for bullet in [
            "Computes a file size hint from the iterator.",
            "Stores the decoded JSON value.",
            "Sends a drop signal to the caller.",
        ] {
            assert!(!brief_bullets_pass_quality(
                &[bullet.to_string()],
                "src/hint.rs",
                "fn value() -> usize { iterator.size_hint().0 }"
            ));
        }
    }

    #[test]
    fn brief_quality_accepts_simple_purpose() {
        let bullets = vec!["Adds a user name to the user list.".to_string()];
        assert!(brief_bullets_pass_quality(
            &bullets,
            "src/users.rs",
            "fn add_user(users: &mut Vec<String>, name: String) { users.push(name); }"
        ));
    }

    #[test]
    fn brief_quality_filter_keeps_good_bullet() {
        let bullets = vec![
            "Adds a user's name to a list of strings.".to_string(),
            "In the context of your code, users is likely a variable.".to_string(),
        ];
        let filtered = filter_brief_bullets_by_quality(
            bullets,
            "src/users.rs",
            "fn add_user(users: &mut Vec<String>, name: String) { users.push(name); }",
        );
        assert_eq!(
            filtered,
            vec!["Adds a user's name to a list of strings.".to_string()]
        );
    }

    #[test]
    fn brief_quality_rejects_unrelated_purpose() {
        let bullets =
            vec!["Retrieves a specific value from a database or JSON object.".to_string()];
        assert!(!brief_bullets_pass_quality(
            &bullets,
            "src/cli.rs",
            "fn dispatch_brief(symbol: Option<&str>, root: Option<&str>) -> Result<i32, String> { println!(\"{} {:?}\", symbol.unwrap_or(\"\"), root); Ok(0) }"
        ));
    }

    #[test]
    fn brief_quality_rejects_atomic_when_source_is_not_atomic() {
        let bullets = vec!["Tracks stale time with an atomic usize variable.".to_string()];
        assert!(!brief_bullets_pass_quality(
            &bullets,
            "src/stats.rs",
            "pub struct Stats { stolen: usize }"
        ));
    }

    #[test]
    fn brief_quality_keeps_atomic_when_source_is_atomic() {
        let bullets = vec!["Updates an atomic counter for worker wakeups.".to_string()];
        assert!(brief_bullets_pass_quality(
            &bullets,
            "src/worker.rs",
            "fn wake(counter: &AtomicUsize) { counter.fetch_add(1, Ordering::Relaxed); }"
        ));
    }

    #[test]
    fn brief_quality_rejects_wrong_language_and_incomplete_meta_output() {
        let source = "pub fn normalize_and_store_user(users: &mut Vec<String>, name: &str) -> usize { users.push(name.trim().to_ascii_lowercase()); users.len() }";
        assert!(!brief_bullets_pass_quality(
            &["The function normalize_and_store_user in Swift".to_string()],
            "src/store_users.rs",
            source,
        ));
        assert!(!brief_bullets_pass_quality(
            &["This snippet defines normalize_and_store_user, which is used".to_string()],
            "src/store_users.rs",
            source,
        ));
    }

    #[test]
    fn skip_with_query_overlap_becomes_read() {
        let verdict = TriageVerdict {
            read: false,
            reason: "likely unrelated span".to_string(),
        };
        let guarded = conservative_triage_guard(
            "where are users added?",
            "src/lib.rs:1-3",
            "fn add_user(users: &mut Vec<String>, name: String) { users.push(name); }",
            verdict,
        );
        assert!(guarded.read);
        assert_eq!(guarded.reason, "matches query terms");
    }

    #[test]
    fn skip_without_overlap_stays_skip() {
        let verdict = TriageVerdict {
            read: false,
            reason: "likely unrelated span".to_string(),
        };
        let guarded = conservative_triage_guard(
            "where are users added?",
            "src/lib.rs:1-3",
            "fn tick() {}",
            verdict,
        );
        assert!(!guarded.read);
        assert_eq!(guarded.reason, "likely unrelated span");
    }
}

fn load_backend(
    model: &greppy_embed_native::GgufModel,
    inventory: Qwen35Inventory,
    eos_token_id: u32,
    options: &LoadOptions,
) -> Result<Backend> {
    match options.device {
        DevicePreference::Cpu => load_cpu_backend(model, inventory, eos_token_id),
        DevicePreference::Auto => load_auto_backend(model, inventory, eos_token_id),
        DevicePreference::Metal => load_metal_backend(model, inventory, eos_token_id),
        DevicePreference::Cuda => load_cuda_backend(model, inventory, eos_token_id),
    }
}

fn load_cpu_backend(
    model: &greppy_embed_native::GgufModel,
    inventory: Qwen35Inventory,
    eos_token_id: u32,
) -> Result<Backend> {
    CpuQwen35Model::load(model, inventory, eos_token_id)
        .map(Box::new)
        .map(Backend::Cpu)
}

fn load_auto_backend(
    model: &greppy_embed_native::GgufModel,
    inventory: Qwen35Inventory,
    eos_token_id: u32,
) -> Result<Backend> {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        load_metal_with_cpu_fallback(model, inventory, eos_token_id)
    }
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    {
        load_cuda_with_cpu_fallback(model, inventory, eos_token_id)
    }
    #[cfg(not(any(
        all(feature = "metal", target_os = "macos"),
        all(feature = "cuda", any(target_os = "linux", target_os = "windows"))
    )))]
    {
        load_cpu_backend(model, inventory, eos_token_id)
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn load_metal_with_cpu_fallback(
    model: &greppy_embed_native::GgufModel,
    inventory: Qwen35Inventory,
    eos_token_id: u32,
) -> Result<Backend> {
    match load_metal_backend(model, inventory.clone(), eos_token_id) {
        Ok(backend) => Ok(backend),
        Err(err) => {
            eprintln!("greppy_qwen35_native: Metal unavailable, falling back to CPU: {err}");
            load_cpu_backend(model, inventory, eos_token_id)
        }
    }
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
fn load_cuda_with_cpu_fallback(
    model: &greppy_embed_native::GgufModel,
    inventory: Qwen35Inventory,
    eos_token_id: u32,
) -> Result<Backend> {
    match load_cuda_backend(model, inventory.clone(), eos_token_id) {
        Ok(backend) => Ok(backend),
        Err(err) => {
            eprintln!("greppy_qwen35_native: CUDA unavailable, falling back to CPU: {err}");
            load_cpu_backend(model, inventory, eos_token_id)
        }
    }
}

fn load_metal_backend(
    model: &greppy_embed_native::GgufModel,
    inventory: Qwen35Inventory,
    eos_token_id: u32,
) -> Result<Backend> {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        crate::metal::model::MetalQwen35Model::from_gguf(model, inventory, eos_token_id)
            .map(Backend::Metal)
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        let _ = (model, inventory, eos_token_id);
        Err(Error::GenerationUnavailable(
            "Qwen3.5 Metal backend is not compiled for this build/platform".into(),
        ))
    }
}

fn load_cuda_backend(
    model: &greppy_embed_native::GgufModel,
    inventory: Qwen35Inventory,
    eos_token_id: u32,
) -> Result<Backend> {
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    {
        crate::cuda::model::CudaQwen35Model::from_gguf(model, inventory, eos_token_id)
            .map(Backend::Cuda)
    }
    #[cfg(not(all(feature = "cuda", any(target_os = "linux", target_os = "windows"))))]
    {
        let _ = (model, inventory, eos_token_id);
        Err(Error::GenerationUnavailable(
            "Qwen3.5 CUDA backend is not compiled for this build/platform".into(),
        ))
    }
}

#[cfg(test)]
#[allow(clippy::infallible_destructuring_match)]
mod cpu_perf_tests {
    use super::*;
    use greppy_embed_native::matmul::QuantMatrix;
    use tokenizers::Tokenizer;

    #[test]
    fn qwen35_prepared_q8k_rows_match_regular_matmul_when_env_set() {
        let Some(gguf) = std::env::var_os("QWEN35_NATIVE_GGUF") else {
            return;
        };
        let model = greppy_embed_native::GgufModel::open(gguf).expect("open Qwen3.5 GGUF");
        for name in [
            "blk.0.ssm_out.weight",
            "blk.0.attn_qkv.weight",
            "blk.0.ffn_down.weight",
        ] {
            let matrix = QuantMatrix::from_model_q4_x8(&model, name).expect("load quant matrix");
            let rows = 17;
            let input = (0..rows * matrix.cols())
                .map(|idx| ((idx * 37 % 257) as f32 - 128.0) / 97.0)
                .collect::<Vec<_>>();
            let expected = matrix.matmul(&input, rows).expect("regular quant matmul");
            let prepared = matrix
                .prepare_q8k_rows(&input, rows)
                .expect("prepare Q8_K rows");
            let actual = matrix
                .matmul_prepared_q8k_rows(&prepared)
                .expect("prepared quant matmul");
            let max_abs = expected
                .iter()
                .zip(&actual)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_abs <= 1.0e-5,
                "prepared Q8_K rows drift for {name}: max_abs={max_abs:.6e}"
            );
        }
    }

    #[test]
    fn qwen35_cpu_batched_prefill_matches_tokenwise_when_env_set() {
        let (Some(gguf), Some(tokenizer)) = (
            std::env::var_os("QWEN35_NATIVE_GGUF"),
            std::env::var_os("QWEN35_NATIVE_TOKENIZER"),
        ) else {
            return;
        };
        let summarizer = Qwen35Summarizer::load_gguf(
            gguf,
            tokenizer,
            LoadOptions {
                device: DevicePreference::Cpu,
            },
        )
        .expect("load CPU Qwen3.5 summarizer");
        let model = match &summarizer.backend {
            Backend::Cpu(model) => model,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Backend::Metal(_) => panic!("expected CPU backend"),
            #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
            Backend::Cuda(_) => panic!("expected CPU backend"),
        };
        let ids = summarizer
            .tokenizer
            .encode(
                crate::prompt::non_thinking_chat_prompt(
                    "Summarize: What is this function for?\n\nfn add_user(users: &mut Vec<String>, name: &str) { users.push(name.to_string()); }",
                ),
                true,
            )
            .expect("tokenize CPU parity prompt")
            .get_ids()
            .to_vec();
        let prefill = &ids[..ids.len() - 1];
        let next = ids[ids.len() - 1];
        let max_context = ids.len() + 1;

        let mut tokenwise_state = model.new_state(max_context);
        for &token in prefill {
            model
                .prefill_token(token, &mut tokenwise_state)
                .expect("tokenwise CPU prefill");
        }
        let tokenwise = model
            .forward_token_logits(next, &mut tokenwise_state)
            .expect("tokenwise CPU logits");

        let mut batched_state = model.new_state(max_context);
        model
            .prefill_tokens(prefill, &mut batched_state)
            .expect("batched CPU prefill");
        let batched = model
            .forward_token_logits(next, &mut batched_state)
            .expect("batched CPU logits");
        assert_eq!(tokenwise.len(), batched.len());
        let max_abs = tokenwise
            .iter()
            .zip(&batched)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs <= 1.0e-3,
            "CPU batched prefill logits drift: max_abs={max_abs:.6e}"
        );
    }

    #[test]
    fn qwen35_cpu_batched_verification_matches_tokenwise_when_env_set() {
        let (Some(gguf), Some(tokenizer)) = (
            std::env::var_os("QWEN35_NATIVE_GGUF"),
            std::env::var_os("QWEN35_NATIVE_TOKENIZER"),
        ) else {
            return;
        };
        let summarizer = Qwen35Summarizer::load_gguf(
            gguf,
            tokenizer,
            LoadOptions {
                device: DevicePreference::Cpu,
            },
        )
        .expect("load CPU Qwen3.5 summarizer");
        let model = match &summarizer.backend {
            Backend::Cpu(model) => model,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Backend::Metal(_) => panic!("expected CPU backend"),
            #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
            Backend::Cuda(_) => panic!("expected CPU backend"),
        };
        let ids = summarizer
            .tokenizer
            .encode(
                crate::prompt::non_thinking_chat_prompt(
                    "Summarize: What is this function for?\n\nfn steal(src: &mut Queue, dst: &mut Queue) { dst.extend(src.take_half()); }",
                ),
                true,
            )
            .expect("tokenize CPU verification prompt")
            .get_ids()
            .to_vec();
        let verify_count = 3;
        let prefix = &ids[..ids.len() - verify_count];
        let verify = &ids[ids.len() - verify_count..];
        let max_context = ids.len() + 1;

        let mut tokenwise_state = model.new_state(max_context);
        model
            .prefill_tokens(prefix, &mut tokenwise_state)
            .expect("CPU tokenwise verification prefix");
        let mut tokenwise_hidden = Vec::new();
        let mut tokenwise_logits = Vec::new();
        for &token in verify {
            let output = model
                .forward_token_logits_hidden(token, &mut tokenwise_state)
                .expect("CPU tokenwise verification row");
            tokenwise_hidden.extend(output.hidden);
            tokenwise_logits.extend(output.logits);
        }

        let mut batched_state = model.new_state(max_context);
        model
            .prefill_tokens(prefix, &mut batched_state)
            .expect("CPU batched verification prefix");
        let batched = model
            .forward_tokens_logits_hidden(verify, &mut batched_state)
            .expect("CPU batched verification rows");

        let hidden_cosine = cosine(&tokenwise_hidden, &batched.hidden);
        let logits_cosine = cosine(&tokenwise_logits, &batched.logits);
        eprintln!(
            "CPU batched verification: hidden_cosine={hidden_cosine:.8} logits_cosine={logits_cosine:.8}"
        );
        assert!(hidden_cosine >= 0.999, "CPU batched hidden-state drift");
        assert!(
            logits_cosine >= 0.999,
            "CPU batched verification-logit drift"
        );
        for row in 0..verify_count {
            let start = row * summarizer.inventory.vocab_size;
            let end = start + summarizer.inventory.vocab_size;
            assert_eq!(
                greedy_argmax(&tokenwise_logits[start..end]),
                greedy_argmax(&batched.logits[start..end]),
                "CPU batched verification changed greedy token at row {row}"
            );
        }
    }

    #[test]
    fn qwen35_cpu_mtp_batched_forward_matches_tokenwise_when_env_set() {
        let (Some(gguf), Some(tokenizer)) = (
            std::env::var_os("QWEN35_NATIVE_MTP_GGUF"),
            std::env::var_os("QWEN35_NATIVE_TOKENIZER"),
        ) else {
            return;
        };
        let summarizer = Qwen35Summarizer::load_gguf(
            gguf,
            tokenizer,
            LoadOptions {
                device: DevicePreference::Cpu,
            },
        )
        .expect("load CPU Qwen3.5 MTP summarizer");
        let model = match &summarizer.backend {
            Backend::Cpu(model) => model,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Backend::Metal(_) => panic!("expected CPU backend"),
            #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
            Backend::Cuda(_) => panic!("expected CPU backend"),
        };
        let ids = summarizer
            .tokenizer
            .encode(
                crate::prompt::non_thinking_chat_prompt(
                    "Summarize: What is this function for?\n\nfn wake(worker: &Worker) { worker.notify(); }",
                ),
                true,
            )
            .expect("tokenize CPU MTP prompt")
            .get_ids()[..8]
            .to_vec();
        let rows = ids.len();
        let hidden_size = summarizer.inventory.hidden_size;
        let max_context = rows + 1;
        let mut target_state = model.new_state(max_context);
        let target_hidden = model
            .prefill_tokens_hidden(&ids, &mut target_state)
            .expect("CPU MTP target hidden rows");
        let mut shifted_hidden = vec![0.0f32; target_hidden.len()];
        shifted_hidden[hidden_size..]
            .copy_from_slice(&target_hidden[..target_hidden.len() - hidden_size]);

        let mut tokenwise_state = model
            .new_mtp_state(max_context)
            .expect("CPU tokenwise MTP state");
        let mut tokenwise_hidden = Vec::new();
        let mut tokenwise_logits = Vec::new();
        for row in 0..rows {
            let hidden = &shifted_hidden[row * hidden_size..(row + 1) * hidden_size];
            let output = model
                .mtp_forward_tokens_logits_hidden(&ids[row..row + 1], hidden, &mut tokenwise_state)
                .expect("CPU tokenwise MTP row");
            tokenwise_hidden.extend(output.hidden);
            tokenwise_logits.extend(output.logits);
        }

        let mut batched_state = model
            .new_mtp_state(max_context)
            .expect("CPU batched MTP state");
        let batched = model
            .mtp_forward_tokens_logits_hidden(&ids, &shifted_hidden, &mut batched_state)
            .expect("CPU batched MTP rows");
        let hidden_cosine = cosine(&tokenwise_hidden, &batched.hidden);
        let logits_cosine = cosine(&tokenwise_logits, &batched.logits);
        eprintln!(
            "CPU batched MTP: hidden_cosine={hidden_cosine:.8} logits_cosine={logits_cosine:.8}"
        );
        assert!(hidden_cosine >= 0.999, "CPU batched MTP hidden-state drift");
        assert!(logits_cosine >= 0.999, "CPU batched MTP logit drift");
        for row in 0..rows {
            let start = row * summarizer.inventory.vocab_size;
            let end = start + summarizer.inventory.vocab_size;
            assert_eq!(
                greedy_argmax(&tokenwise_logits[start..end]),
                greedy_argmax(&batched.logits[start..end]),
                "CPU batched MTP changed greedy token at row {row}"
            );
        }
    }

    #[test]
    fn qwen35_cpu_mtp_matches_target_sampling_when_env_set() {
        let (Some(gguf), Some(tokenizer)) = (
            std::env::var_os("QWEN35_NATIVE_MTP_GGUF"),
            std::env::var_os("QWEN35_NATIVE_TOKENIZER"),
        ) else {
            return;
        };
        let summarizer = Qwen35Summarizer::load_gguf(
            gguf,
            tokenizer,
            LoadOptions {
                device: DevicePreference::Cpu,
            },
        )
        .expect("load CPU Qwen3.5 MTP summarizer");
        let model = match &summarizer.backend {
            Backend::Cpu(model) => model,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Backend::Metal(_) => panic!("expected CPU backend"),
            #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
            Backend::Cuda(_) => panic!("expected CPU backend"),
        };
        let prompt = crate::prompt::brief_chat_prompt(
            "src/rename_rules.rs",
            "pub fn rename_by_rules(&mut self, rules: RenameAllRules) {\n    self.serialize.value = rules.serialize.apply_to_field(&self.serialize.value);\n    self.deserialize.value = rules.deserialize.apply_to_field(&self.deserialize.value);\n}",
        );
        let prompt_ids = summarizer
            .tokenizer
            .encode(prompt.as_str(), true)
            .expect("tokenize MTP golden prompt")
            .get_ids()
            .to_vec();
        let hidden_size = summarizer.inventory.hidden_size;
        let mut target_state = model.new_state(prompt_ids.len() + 3);
        let prompt_hidden = model
            .prefill_tokens_hidden(&prompt_ids, &mut target_state)
            .expect("MTP golden target prompt");
        let target = model
            .target_from_prefilled_hidden(&prompt_hidden, prompt_ids.len())
            .expect("MTP golden target projection");
        let first = greedy_argmax(&target.logits);
        assert_eq!(first, 10296, "unexpected MTP golden target token");

        // MTP conditioning contract: row i pairs embed(token_{i+1}) with the
        // POST-final-norm trunk hidden of token_i, so the prompt prefill skips
        // token 0 (mirrors `generate_with_mtp`).
        let prompt_conditioning =
            model.output_normed_rows_for_test(&prompt_hidden[..prompt_hidden.len() - hidden_size]);
        let mut mtp_state = model
            .new_mtp_state(prompt_ids.len() + 3)
            .expect("MTP golden state");
        model
            .mtp_prefill_tokens(&prompt_ids[1..], &prompt_conditioning, &mut mtp_state)
            .expect("MTP golden prompt prefill");
        let first_draft = model
            .mtp_forward_tokens_logits_hidden(&[first], &target.hidden, &mut mtp_state)
            .expect("MTP first golden draft");
        let first_draft_token = greedy_argmax(&first_draft.logits);
        let second_draft = model
            .mtp_forward_tokens_logits_hidden(
                &[first_draft_token],
                &first_draft.hidden,
                &mut mtp_state,
            )
            .expect("MTP second golden draft");
        // Golden drafts for ckpt-2026-07-13-Q4_K_M. Cross-checked against the
        // HF bf16 reference chain (mtp_module.Qwen35MTP, vLLM semantics): the
        // trunk token (10296) and draft step 1 (6976) match HF exactly; step 2
        // is Q4_K_M-trunk-consistent (the engine's own greedy target also picks
        // 264 there, while the fp32 trunk diverges at that position).
        assert_eq!(
            [first_draft_token, greedy_argmax(&second_draft.logits)],
            [6976, 264],
            "native MTP drafts differ from the finetuned-model golden tokens"
        );

        for params in [
            crate::GenerationParams {
                max_tokens: 12,
                ..crate::BRIEF_GENERATION_PARAMS
            },
            crate::GenerationParams {
                max_tokens: 64,
                ..crate::TRIAGE_GENERATION_PARAMS
            },
        ] {
            let expected = model
                .generate_target_only_for_test(&summarizer.tokenizer, &prompt, params)
                .expect("CPU target-only generation");
            let actual = model
                .generate(&summarizer.tokenizer, &prompt, params)
                .expect("CPU MTP generation");
            assert_eq!(actual, expected, "MTP changed target sampling output");
        }
    }

    #[test]
    #[ignore]
    fn qwen35_cpu_quality_prints_when_env_set() {
        let (Some(gguf), Some(tokenizer)) = (
            std::env::var_os("QWEN35_NATIVE_GGUF"),
            std::env::var_os("QWEN35_NATIVE_TOKENIZER"),
        ) else {
            eprintln!("skipping CPU quality: QWEN35_NATIVE_GGUF/TOKENIZER unset");
            return;
        };
        let summarizer = Qwen35Summarizer::load_gguf(
            gguf,
            tokenizer,
            LoadOptions {
                device: DevicePreference::Cpu,
            },
        )
        .expect("load CPU Qwen3.5 summarizer");
        let source = "pub fn add_user(users: &mut Vec<String>, name: &str) -> usize {\n    users.push(name.trim().to_string());\n    users.len()\n}\n";
        let path = "src/users.rs";
        let prompt = crate::brief_prompt(path, source);
        let model_prompt = crate::prompt::brief_chat_prompt(path, source);
        let params = crate::GenerationParams {
            max_tokens: 12,
            ..crate::BRIEF_GENERATION_PARAMS
        };
        let raw = summarizer
            .generate_raw(&model_prompt, params)
            .expect("generate raw CPU summary");
        let bullets = crate::postprocess_brief_output(&raw, &prompt);
        println!("qwen35_cpu_quality raw={raw:?}");
        println!("qwen35_cpu_quality bullets={bullets:?}");
    }

    #[test]
    #[ignore]
    fn qwen35_cpu_perf_prints_when_env_set() {
        let (Some(gguf), Some(tokenizer)) = (
            std::env::var_os("QWEN35_NATIVE_GGUF"),
            std::env::var_os("QWEN35_NATIVE_TOKENIZER"),
        ) else {
            eprintln!("skipping CPU perf: QWEN35_NATIVE_GGUF/TOKENIZER unset");
            return;
        };
        let input_target = perf_env_usize("QWEN35_NATIVE_PERF_INPUT_TOKENS", 32).max(2);
        let output_target = perf_env_usize("QWEN35_NATIVE_PERF_OUTPUT_TOKENS", 4).max(1);
        let summarizer = Qwen35Summarizer::load_gguf(
            gguf,
            tokenizer,
            LoadOptions {
                device: DevicePreference::Cpu,
            },
        )
        .expect("load CPU Qwen3.5 summarizer");
        let model = match &summarizer.backend {
            Backend::Cpu(model) => model,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Backend::Metal(_) => panic!("expected CPU backend"),
            #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
            Backend::Cuda(_) => panic!("expected CPU backend"),
        };
        let prompt_ids = perf_prompt_ids(&summarizer.tokenizer, input_target);
        let prefill_count = prompt_ids.len() - 1;
        let (input_elapsed, generated, output_elapsed) = model.on_performance_cores(|| {
            let mut warm_state = model.new_state(8);
            for &token in prompt_ids.iter().take(2) {
                model
                    .prefill_token(token, &mut warm_state)
                    .expect("CPU warmup prefill");
            }

            let mut state = model.new_state(prompt_ids.len() + output_target + 1);
            let t0 = std::time::Instant::now();
            for tokens in prompt_ids[..prefill_count].chunks(512) {
                model
                    .prefill_tokens(tokens, &mut state)
                    .expect("CPU perf prefill forward");
            }
            let input_elapsed = t0.elapsed();

            let mut next = prompt_ids[prefill_count];
            let mut generated = Vec::with_capacity(output_target);
            let t1 = std::time::Instant::now();
            for _ in 0..output_target {
                let logits = model
                    .forward_token_logits(next, &mut state)
                    .expect("CPU perf decode forward");
                let token = greedy_argmax(&logits);
                generated.push(token);
                next = token;
            }
            (input_elapsed, generated, t1.elapsed())
        });
        let input_tps = prefill_count as f64 / input_elapsed.as_secs_f64();
        let output_tps = generated.len() as f64 / output_elapsed.as_secs_f64();
        println!(
            "qwen35_native_cpu_perf input_tokens={} input_s={:.6} input_tok_s={:.2} output_tokens={} output_s={:.6} output_tok_s={:.2}",
            prefill_count,
            input_elapsed.as_secs_f64(),
            input_tps,
            generated.len(),
            output_elapsed.as_secs_f64(),
            output_tps
        );
    }

    #[test]
    #[ignore = "diagnostic CPU layer profile"]
    fn qwen35_cpu_prefill_profile_when_env_set() {
        let (Some(gguf), Some(tokenizer)) = (
            std::env::var_os("QWEN35_NATIVE_GGUF"),
            std::env::var_os("QWEN35_NATIVE_TOKENIZER"),
        ) else {
            eprintln!("skipping CPU profile: QWEN35_NATIVE_GGUF/TOKENIZER unset");
            return;
        };
        let rows = perf_env_usize("QWEN35_NATIVE_CPU_PROFILE_ROWS", 511).clamp(1, 512);
        let summarizer = Qwen35Summarizer::load_gguf(
            gguf,
            tokenizer,
            LoadOptions {
                device: DevicePreference::Cpu,
            },
        )
        .expect("load CPU Qwen3.5 summarizer");
        let model = match &summarizer.backend {
            Backend::Cpu(model) => model,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Backend::Metal(_) => panic!("expected CPU backend"),
            #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
            Backend::Cuda(_) => panic!("expected CPU backend"),
        };
        let prompt_ids = perf_prompt_ids(&summarizer.tokenizer, rows + 1);
        let tokens = &prompt_ids[..rows.min(prompt_ids.len())];
        model.on_performance_cores(|| {
            let mut state = model.new_state(tokens.len() + 1);
            model
                .profile_prefill_tokens(tokens, &mut state)
                .expect("CPU profile prefill");
        });
    }

    #[test]
    #[ignore = "diagnostic CPU decode profile"]
    fn qwen35_cpu_decode_profile_when_env_set() {
        let (Some(gguf), Some(tokenizer)) = (
            std::env::var_os("QWEN35_NATIVE_GGUF"),
            std::env::var_os("QWEN35_NATIVE_TOKENIZER"),
        ) else {
            eprintln!("skipping CPU decode profile: QWEN35_NATIVE_GGUF/TOKENIZER unset");
            return;
        };
        let position = perf_env_usize("QWEN35_NATIVE_CPU_DECODE_POSITION", 64).clamp(1, 511);
        let inventory = greppy_embed_native::GgufModel::open(&gguf).expect("open CPU profile GGUF");
        for name in [
            "token_embd.weight",
            "blk.0.attn_qkv.weight",
            "blk.0.ssm_out.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_down.weight",
        ] {
            let tensor = inventory.tensor(name).expect("CPU profile tensor");
            eprintln!(
                "cpu_decode_profile stage=inventory tensor={name} dtype={} shape={:?}",
                tensor.dtype, tensor.shape,
            );
        }
        let summarizer = Qwen35Summarizer::load_gguf(
            gguf,
            tokenizer,
            LoadOptions {
                device: DevicePreference::Cpu,
            },
        )
        .expect("load CPU Qwen3.5 summarizer");
        let model = match &summarizer.backend {
            Backend::Cpu(model) => model,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Backend::Metal(_) => panic!("expected CPU backend"),
            #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
            Backend::Cuda(_) => panic!("expected CPU backend"),
        };
        let prompt_ids = perf_prompt_ids(&summarizer.tokenizer, position + 1);
        model.on_performance_cores(|| {
            let mut state = model.new_state(position + 2);
            model
                .prefill_tokens(&prompt_ids[..position], &mut state)
                .expect("CPU decode profile prefill");
            model
                .profile_forward_token_logits(prompt_ids[position], &mut state)
                .expect("CPU decode profile token");
        });
    }

    fn perf_env_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(default)
    }

    fn perf_prompt_ids(tokenizer: &Tokenizer, target: usize) -> Vec<u32> {
        let snippet = "\nfn dispatch_brief(symbol: Option<&str>, root: Option<&str>) -> Result<i32> {\n    let store = open_default_store(root)?;\n    let project = project_for(root)?;\n    let nodes = resolve_symbol_nodes(&store, symbol.unwrap_or(\"\"), &project)?;\n    for node in nodes {\n        print_code_span(&project.root, &node, CONTEXT_SPAN_CAP);\n    }\n    Ok(0)\n}\n";
        let mut text = String::from("Summarize: What is this function for?\n\n");
        let mut ids = Vec::new();
        while ids.len() < target {
            text.push_str(snippet);
            ids = tokenizer
                .encode(text.as_str(), true)
                .expect("tokenize perf prompt")
                .get_ids()
                .to_vec();
        }
        ids.truncate(target);
        ids
    }

    fn greedy_argmax(logits: &[f32]) -> u32 {
        let mut best_idx = 0usize;
        let mut best = f32::NEG_INFINITY;
        for (idx, &value) in logits.iter().enumerate() {
            if value.is_nan() {
                continue;
            }
            if value > best {
                best = value;
                best_idx = idx;
            }
        }
        best_idx as u32
    }

    fn cosine(lhs: &[f32], rhs: &[f32]) -> f64 {
        assert_eq!(lhs.len(), rhs.len());
        let (mut dot, mut lhs_norm, mut rhs_norm) = (0.0f64, 0.0f64, 0.0f64);
        for (&left, &right) in lhs.iter().zip(rhs) {
            let left = f64::from(left);
            let right = f64::from(right);
            dot += left * right;
            lhs_norm += left * left;
            rhs_norm += right * right;
        }
        dot / (lhs_norm.sqrt() * rhs_norm.sqrt()).max(f64::EPSILON)
    }
}

#[cfg(all(test, feature = "metal", target_os = "macos"))]
mod metal_perf_tests {
    use super::*;

    #[test]
    #[ignore]
    fn qwen35_metal_quality_prints_when_env_set() {
        let (Some(gguf), Some(tokenizer)) = (
            std::env::var_os("QWEN35_NATIVE_GGUF"),
            std::env::var_os("QWEN35_NATIVE_TOKENIZER"),
        ) else {
            eprintln!("skipping Metal quality: QWEN35_NATIVE_GGUF/TOKENIZER unset");
            return;
        };
        let summarizer = Qwen35Summarizer::load_gguf(
            gguf,
            tokenizer,
            LoadOptions {
                device: DevicePreference::Metal,
            },
        )
        .expect("load Metal Qwen3.5 summarizer");
        let source =
            "fn add_user(users: &mut Vec<String>, name: String) {\n    users.push(name);\n}\n";
        let path = "src/users.rs";
        let prompt = crate::brief_prompt(path, source);
        let model_prompt = crate::prompt::brief_chat_prompt(path, source);
        let raw = summarizer
            .generate_raw(&model_prompt, crate::BRIEF_GENERATION_PARAMS)
            .expect("generate raw Metal summary");
        let greedy_params = crate::GenerationParams {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            min_p: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            max_tokens: 32,
        };
        let greedy_raw = summarizer
            .generate_raw(&model_prompt, greedy_params)
            .expect("generate greedy raw Metal summary");
        let bullets = crate::postprocess_brief_output(&raw, &prompt);
        let summary = summarizer
            .summarize_source(path, source)
            .expect("summarize exact Metal source");
        println!("qwen35_metal_quality raw={raw:?}");
        println!("qwen35_metal_quality greedy_raw={greedy_raw:?}");
        println!("qwen35_metal_quality bullets={bullets:?}");
        println!("qwen35_metal_quality summary={summary:?}");
        let verdict = summarizer
            .triage_span("where are users added?", "src/lib.rs:1-3", source)
            .expect("triage Metal span");
        println!(
            "qwen35_metal_quality triage read={} reason={}",
            verdict.read, verdict.reason
        );
    }

    #[test]
    #[ignore]
    fn qwen35_metal_perf_reports_backend_status_when_env_set() {
        let (Some(gguf), Some(tokenizer)) = (
            std::env::var_os("QWEN35_NATIVE_GGUF"),
            std::env::var_os("QWEN35_NATIVE_TOKENIZER"),
        ) else {
            eprintln!("skipping Metal perf status: QWEN35_NATIVE_GGUF/TOKENIZER unset");
            return;
        };
        match Qwen35Summarizer::load_gguf(
            gguf,
            tokenizer,
            LoadOptions {
                device: DevicePreference::Metal,
            },
        ) {
            Ok(summarizer) => {
                let input_target = perf_env_usize("QWEN35_NATIVE_PERF_INPUT_TOKENS", 32).max(2);
                let output_target = perf_env_usize("QWEN35_NATIVE_PERF_OUTPUT_TOKENS", 4).max(1);
                let prompt_ids = perf_prompt_ids(&summarizer.tokenizer, input_target);
                let max_context = prompt_ids
                    .len()
                    .saturating_add(output_target)
                    .saturating_add(1)
                    .min(summarizer.inventory.context_length);
                let Backend::Metal(model) = &summarizer.backend else {
                    panic!("expected Metal backend");
                };
                let warm_rows = prompt_ids.len().saturating_sub(1).min(32);
                if warm_rows > 0 {
                    let mut warm_state = model
                        .new_forward_state(warm_rows + 1)
                        .expect("Metal perf warm state");
                    model
                        .prefill_tokens(&prompt_ids[..warm_rows], &mut warm_state)
                        .expect("Metal perf warm prefill");
                }
                let mut state = model
                    .new_forward_state(max_context)
                    .expect("Metal perf state");
                let mut workspace = model
                    .new_forward_workspace(max_context)
                    .expect("Metal perf workspace");
                let prefill = &prompt_ids[..prompt_ids.len().saturating_sub(1)];
                let t0 = std::time::Instant::now();
                for chunk in prefill.chunks(512) {
                    model
                        .prefill_tokens(chunk, &mut state)
                        .expect("Metal perf batched prefill forward");
                }
                let input_s = t0.elapsed().as_secs_f64();
                let mut next = *prompt_ids.last().expect("non-empty perf prompt");
                let t1 = std::time::Instant::now();
                for _ in 0..output_target {
                    next = model
                        .forward_token_greedy(next, &mut state, &mut workspace)
                        .expect("Metal perf greedy decode forward");
                }
                let output_s = t1.elapsed().as_secs_f64();
                println!(
                    "qwen35_native_metal_perf backend={} input_tokens={} input_s={:.6} input_tok_s={:.2} output_tokens={} output_s={:.6} output_tok_s={:.2}",
                    summarizer.backend_name(),
                    prefill.len(),
                    input_s,
                    prefill.len() as f64 / input_s.max(1.0e-9),
                    output_target,
                    output_s,
                    output_target as f64 / output_s.max(1.0e-9),
                );
            }
            Err(Error::GenerationUnavailable(message)) => {
                println!(
                    "qwen35_native_metal_perf status=unavailable backend_status={} reason={}",
                    crate::metal::BACKEND_STATUS,
                    message
                );
            }
            Err(err) => panic!("unexpected Metal load error: {err}"),
        }
    }

    fn perf_env_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(default)
    }

    fn perf_prompt_ids(tokenizer: &Tokenizer, target: usize) -> Vec<u32> {
        let seed = "fn dispatch_brief(symbol: Option<&str>, root: Option<&str>) -> Result<i32> { let graph = load_graph(root)?; for node in graph.nodes() { if node.name == symbol.unwrap_or(\"\") { print_code_span(node)?; } } Ok(0) }\n";
        let mut text = String::new();
        while tokenizer
            .encode(text.as_str(), true)
            .expect("tokenize perf prompt")
            .len()
            < target
        {
            text.push_str(seed);
        }
        let enc = tokenizer
            .encode(text.as_str(), true)
            .expect("tokenize perf prompt");
        enc.get_ids()[..target.min(enc.len())].to_vec()
    }
}

#[cfg(all(
    test,
    feature = "cuda",
    any(target_os = "linux", target_os = "windows")
))]
mod tests {
    use super::*;
    use tokenizers::Tokenizer;

    #[test]
    fn qwen35_cuda_summarizer_runs_generation_when_env_set() {
        let (Some(gguf), Some(tokenizer)) = (
            std::env::var_os("QWEN35_NATIVE_GGUF"),
            std::env::var_os("QWEN35_NATIVE_TOKENIZER"),
        ) else {
            eprintln!("skipping CUDA summarizer generation: QWEN35_NATIVE_GGUF/TOKENIZER unset");
            return;
        };
        let summarizer = Qwen35Summarizer::load_gguf(
            gguf,
            tokenizer,
            LoadOptions {
                device: DevicePreference::Cuda,
            },
        )
        .expect("load CUDA Qwen3.5 summarizer");
        assert!(summarizer.backend_name().starts_with("cuda-"));
        let bullets = summarizer
            .summarize_source(
                "src/users.rs",
                "fn add_user(users: &mut Vec<String>, name: String) {\n    users.push(name);\n}\n",
            )
            .expect("CUDA summary generation");
        assert!(bullets.len() <= 2);
    }

    #[test]
    fn qwen35_cuda_mixed_mtp_target_matches_golden_when_env_set() {
        let (Some(gguf), Some(tokenizer)) = (
            std::env::var_os("QWEN35_NATIVE_MTP_GGUF"),
            std::env::var_os("QWEN35_NATIVE_TOKENIZER"),
        ) else {
            return;
        };
        let summarizer = Qwen35Summarizer::load_gguf(
            gguf,
            tokenizer,
            LoadOptions {
                device: DevicePreference::Cuda,
            },
        )
        .expect("load mixed CUDA Qwen3.5 MTP summarizer");
        let Backend::Cuda(model) = &summarizer.backend else {
            panic!("expected CUDA backend");
        };
        let prompt = crate::prompt::brief_chat_prompt(
            "src/rename_rules.rs",
            "pub fn rename_by_rules(&mut self, rules: RenameAllRules) {\n    self.serialize.value = rules.serialize.apply_to_field(&self.serialize.value);\n    self.deserialize.value = rules.deserialize.apply_to_field(&self.deserialize.value);\n}",
        );
        let ids = summarizer
            .tokenizer
            .encode(prompt, true)
            .expect("tokenize mixed CUDA golden prompt")
            .get_ids()
            .to_vec();
        let mut state = model
            .new_forward_state(ids.len() + 1)
            .expect("mixed CUDA golden state");
        for chunk in ids[..ids.len() - 1].chunks(512) {
            model
                .prefill_tokens(chunk, &mut state)
                .expect("mixed CUDA golden prefill");
        }
        let mut workspace = model
            .new_forward_workspace(ids.len() + 1)
            .expect("mixed CUDA golden workspace");
        let logits = model
            .forward_token_logits(
                *ids.last().expect("non-empty mixed CUDA golden prompt"),
                &mut state,
                &mut workspace,
            )
            .expect("mixed CUDA golden logits");
        let token = logits
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| u32::try_from(index).expect("vocabulary index fits u32"))
            .expect("non-empty mixed CUDA golden logits");
        assert_eq!(
            token, 10296,
            "mixed CUDA target differs from the finetuned-model golden token"
        );
    }

    #[test]
    #[ignore]
    fn qwen35_cuda_perf_prints_when_env_set() {
        let (Some(gguf), Some(tokenizer)) = (
            std::env::var_os("QWEN35_NATIVE_GGUF"),
            std::env::var_os("QWEN35_NATIVE_TOKENIZER"),
        ) else {
            eprintln!("skipping CUDA perf: QWEN35_NATIVE_GGUF/TOKENIZER unset");
            return;
        };
        let input_target = perf_env_usize("QWEN35_NATIVE_PERF_INPUT_TOKENS", 512).max(2);
        let output_target = perf_env_usize("QWEN35_NATIVE_PERF_OUTPUT_TOKENS", 128).max(1);
        let summarizer = Qwen35Summarizer::load_gguf(
            gguf,
            tokenizer,
            LoadOptions {
                device: DevicePreference::Cuda,
            },
        )
        .expect("load CUDA Qwen3.5 summarizer");
        let Backend::Cuda(model) = &summarizer.backend else {
            panic!("expected CUDA backend");
        };
        let prompt_ids = perf_prompt_ids(&summarizer.tokenizer, input_target);
        let mut warm_state = model
            .new_forward_state(16)
            .expect("CUDA warmup forward state");
        let mut warm_ws = model
            .new_forward_workspace(16)
            .expect("CUDA warmup workspace");
        for &token in prompt_ids.iter().take(4) {
            model
                .prefill_token(token, &mut warm_state, &mut warm_ws)
                .expect("CUDA warmup prefill");
        }

        let mut state = model
            .new_forward_state(prompt_ids.len() + output_target + 1)
            .expect("CUDA perf forward state");
        let mut ws = model
            .new_forward_workspace(prompt_ids.len() + output_target + 1)
            .expect("CUDA perf workspace");
        let prefill_count = prompt_ids.len() - 1;
        let t0 = std::time::Instant::now();
        for chunk in prompt_ids[..prefill_count].chunks(512) {
            model
                .prefill_tokens(chunk, &mut state)
                .expect("CUDA perf batched prefill forward");
        }
        let input_elapsed = t0.elapsed();

        let mut next = prompt_ids[prefill_count];
        let mut generated = Vec::with_capacity(output_target);
        let t1 = std::time::Instant::now();
        for _ in 0..output_target {
            let token = model
                .forward_token_greedy(next, &mut state, &mut ws)
                .expect("CUDA perf greedy decode forward");
            generated.push(token);
            next = token;
        }
        let output_elapsed = t1.elapsed();
        let input_tps = prefill_count as f64 / input_elapsed.as_secs_f64();
        let output_tps = generated.len() as f64 / output_elapsed.as_secs_f64();
        println!(
            "qwen35_native_cuda_perf input_tokens={} input_s={:.6} input_tok_s={:.2} output_tokens={} output_s={:.6} output_tok_s={:.2}",
            prefill_count,
            input_elapsed.as_secs_f64(),
            input_tps,
            generated.len(),
            output_elapsed.as_secs_f64(),
            output_tps
        );
    }

    fn perf_env_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(default)
    }

    fn perf_prompt_ids(tokenizer: &Tokenizer, target: usize) -> Vec<u32> {
        let snippet = "\nfn dispatch_brief(symbol: Option<&str>, root: Option<&str>) -> Result<i32> {\n    let store = open_default_store(root)?;\n    let project = project_for(root)?;\n    let nodes = resolve_symbol_nodes(&store, symbol.unwrap_or(\"\"), &project)?;\n    for node in nodes {\n        print_code_span(&project.root, &node, CONTEXT_SPAN_CAP);\n    }\n    Ok(0)\n}\n";
        let mut text = String::from("Summarize: What is this function for?\n\n");
        let mut ids = Vec::new();
        while ids.len() < target {
            text.push_str(snippet);
            ids = tokenizer
                .encode(text.as_str(), true)
                .expect("tokenize perf prompt")
                .get_ids()
                .to_vec();
        }
        ids.truncate(target);
        ids
    }
}
