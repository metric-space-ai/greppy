use std::path::Path;

use tokenizers::Tokenizer;

use crate::cpu::CpuQwen35Model;
use crate::inventory::Qwen35Inventory;
use crate::postprocess::{postprocess_brief_output, postprocess_triage_output, TriageVerdict};
use crate::prompt::{brief_prompt, non_thinking_chat_prompt, triage_prompt};
use crate::sampler::{
    BRIEF_FALLBACK_GENERATION_PARAMS, BRIEF_GENERATION_PARAMS, TRIAGE_GENERATION_PARAMS,
};
use crate::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DevicePreference {
    Auto,
    Cpu,
    Metal,
    Cuda,
}

impl DevicePreference {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            "cuda" => Ok(Self::Cuda),
            other => Err(Error::InvalidRequest(format!(
                "unsupported device `{other}`; expected auto|cpu|metal|cuda"
            ))),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }
}

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
    Cpu(CpuQwen35Model),
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
        let model = greppy_embed_native::GgufModel::open(gguf_path.as_ref())?;
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

    pub fn summarize_source(&self, source: &str) -> Result<Vec<String>> {
        if source.trim().is_empty() {
            return Ok(Vec::new());
        }
        let prompt = brief_prompt(source);
        let model_prompt = non_thinking_chat_prompt(&prompt);
        let raw = self.generate_raw(&model_prompt, BRIEF_GENERATION_PARAMS)?;
        let bullets = postprocess_brief_output(&raw, &prompt);
        log_summary_debug("primary", &raw, &bullets);
        let bullets = filter_brief_bullets_by_quality(bullets, source);
        log_filtered_summary_debug("primary", &bullets);
        if !bullets.is_empty() {
            return Ok(bullets);
        }
        let fallback_raw = self.generate_raw(&model_prompt, BRIEF_FALLBACK_GENERATION_PARAMS)?;
        let fallback = postprocess_brief_output(&fallback_raw, &prompt);
        log_summary_debug("fallback", &fallback_raw, &fallback);
        let fallback = filter_brief_bullets_by_quality(fallback, source);
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

fn log_summary_debug(stage: &str, raw: &str, bullets: &[String]) {
    if std::env::var_os("GREPPY_QWEN35_SUMMARY_DEBUG").is_some() {
        eprintln!("qwen35-summary-debug {stage} raw={raw:?} bullets={bullets:?}");
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
fn brief_bullets_pass_quality(bullets: &[String], source: &str) -> bool {
    if bullets.is_empty() {
        return false;
    }
    let source_terms = lexical_terms(source).collect::<std::collections::BTreeSet<_>>();
    let source_identifiers = code_identifiers(source).collect::<std::collections::BTreeSet<_>>();
    bullets
        .iter()
        .all(|bullet| brief_bullet_passes_quality(bullet, &source_terms, &source_identifiers))
}

fn filter_brief_bullets_by_quality(bullets: Vec<String>, source: &str) -> Vec<String> {
    let source_terms = lexical_terms(source).collect::<std::collections::BTreeSet<_>>();
    let source_identifiers = code_identifiers(source).collect::<std::collections::BTreeSet<_>>();
    bullets
        .into_iter()
        .filter(|bullet| brief_bullet_passes_quality(bullet, &source_terms, &source_identifiers))
        .take(2)
        .collect()
}

fn brief_bullet_passes_quality(
    bullet: &str,
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
                .any(|identifier| identifier == language);
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
            .any(|ident| ident.to_ascii_lowercase().contains("atomic"));
    if lower.contains("atomic") && !source_mentions_atomic {
        return false;
    }
    if code_identifiers(bullet)
        .filter(|ident| ident.contains('_'))
        .any(|ident| !source_identifiers.contains(&ident))
    {
        return false;
    }
    source_terms.is_empty() || lexical_terms(bullet).any(|term| source_terms.contains(&term))
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
            "fn add_user(users: &mut Vec<String>, name: String) { users.push(name); }"
        ));
    }

    #[test]
    fn brief_quality_accepts_simple_purpose() {
        let bullets = vec!["Adds a user name to the user list.".to_string()];
        assert!(brief_bullets_pass_quality(
            &bullets,
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
            "fn dispatch_brief(symbol: Option<&str>, root: Option<&str>) -> Result<i32, String> { println!(\"{} {:?}\", symbol.unwrap_or(\"\"), root); Ok(0) }"
        ));
    }

    #[test]
    fn brief_quality_rejects_atomic_when_source_is_not_atomic() {
        let bullets = vec!["Tracks stale time with an atomic usize variable.".to_string()];
        assert!(!brief_bullets_pass_quality(
            &bullets,
            "pub struct Stats { stolen: usize }"
        ));
    }

    #[test]
    fn brief_quality_keeps_atomic_when_source_is_atomic() {
        let bullets = vec!["Updates an atomic counter for worker wakeups.".to_string()];
        assert!(brief_bullets_pass_quality(
            &bullets,
            "fn wake(counter: &AtomicUsize) { counter.fetch_add(1, Ordering::Relaxed); }"
        ));
    }

    #[test]
    fn brief_quality_rejects_wrong_language_and_incomplete_meta_output() {
        let source = "pub fn normalize_and_store_user(users: &mut Vec<String>, name: &str) -> usize { users.push(name.trim().to_ascii_lowercase()); users.len() }";
        assert!(!brief_bullets_pass_quality(
            &["The function normalize_and_store_user in Swift".to_string()],
            source,
        ));
        assert!(!brief_bullets_pass_quality(
            &["This snippet defines normalize_and_store_user, which is used".to_string()],
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
        DevicePreference::Metal => load_metal_with_cpu_fallback(model, inventory, eos_token_id),
        DevicePreference::Cuda => load_cuda_with_cpu_fallback(model, inventory, eos_token_id),
    }
}

fn load_cpu_backend(
    model: &greppy_embed_native::GgufModel,
    inventory: Qwen35Inventory,
    eos_token_id: u32,
) -> Result<Backend> {
    CpuQwen35Model::load(model, inventory, eos_token_id).map(Backend::Cpu)
}

fn load_auto_backend(
    model: &greppy_embed_native::GgufModel,
    inventory: Qwen35Inventory,
    eos_token_id: u32,
) -> Result<Backend> {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        return load_metal_with_cpu_fallback(model, inventory, eos_token_id);
    }
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    {
        return load_cuda_with_cpu_fallback(model, inventory, eos_token_id);
    }
    #[cfg(not(any(
        all(feature = "metal", target_os = "macos"),
        all(feature = "cuda", any(target_os = "linux", target_os = "windows"))
    )))]
    {
        load_cpu_backend(model, inventory, eos_token_id)
    }
}

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
        return crate::metal::model::MetalQwen35Model::from_gguf(model, inventory, eos_token_id)
            .map(Backend::Metal);
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
        return crate::cuda::model::CudaQwen35Model::from_gguf(model, inventory, eos_token_id)
            .map(Backend::Cuda);
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
mod cpu_perf_tests {
    use super::*;
    use tokenizers::Tokenizer;

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
        let prompt = crate::brief_prompt(source);
        let model_prompt = crate::prompt::non_thinking_chat_prompt(&prompt);
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
        let mut warm_state = model.new_state(8);
        for &token in prompt_ids.iter().take(2) {
            model
                .prefill_token(token, &mut warm_state)
                .expect("CPU warmup prefill");
        }

        let mut state = model.new_state(prompt_ids.len() + output_target + 1);
        let prefill_count = prompt_ids.len() - 1;
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
        let output_elapsed = t1.elapsed();
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
        let mut state = model.new_state(tokens.len() + 1);
        model
            .profile_prefill_tokens(tokens, &mut state)
            .expect("CPU profile prefill");
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
        let mut state = model.new_state(position + 2);
        model
            .prefill_tokens(&prompt_ids[..position], &mut state)
            .expect("CPU decode profile prefill");
        model
            .profile_forward_token_logits(prompt_ids[position], &mut state)
            .expect("CPU decode profile token");
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
        let prompt = crate::brief_prompt(source);
        let model_prompt = crate::prompt::non_thinking_chat_prompt(&prompt);
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
            .summarize_source(source)
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
                "fn add_user(users: &mut Vec<String>, name: String) {\n    users.push(name);\n}\n",
            )
            .expect("CUDA summary generation");
        assert!(bullets.len() <= 2);
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
