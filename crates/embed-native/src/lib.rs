//! Candle-free native inference of EmbeddingGemma-300M (Gemma3, Q4_K GGUF).
//!
//! Goal: replace the `candle-core`/`candle-nn` dependency entirely (0 candle
//! crates in Cargo.lock) with a lean, single-model native engine. Compatible
//! Apache/MIT kernel source may be ported with provenance; a Candle *runtime
//! dependency* is not part of the product.
//!
//! Backends: portable/runtime-dispatched CPU, Apple-Silicon Metal, and
//! Linux/x86_64 CUDA. Every stage is verified against golden vectors in
//! `testdata/golden/`: `golden_single.json` (token_ids + final embeddings),
//! `golden_batch.json` (padded batch → mean-pool-over-mask), `golden_stages.json`
//! (per-stage hidden states: embed_scaled → layer_0..23 → output_norm →
//! mean_pool → dense2 → dense3 → l2norm).
//!
//! Kernel origins, revisions, local changes, and licenses are recorded under
//! `vendor/` and in the repository's `THIRD_PARTY.md`.

#![deny(rust_2018_idioms)]

pub mod backend;
pub mod cpu_features;
pub mod gguf;
pub mod matmul;
pub mod model;
pub mod performance;
pub mod quant;
pub mod tokenizer;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
pub mod cuda;

pub use backend::{
    device_has_memory, estimated_gpu_memory, preflight_explicit_model, BackendKind, BackendProbe,
    DeviceInfo, DeviceType, InferenceBackendRegistry, InferenceModelKind, InferencePolicy,
    BACKEND_REGISTRY_VERSION, GPU_MEMORY_SAFETY_MARGIN,
};
pub use gguf::{GgufModel, TensorInfo, TensorView, Value, ValueType, VersionedMagic};
pub use model::{CpuEmbeddingModel, StageOutput};
pub use quant::GgmlDType;
pub use tokenizer::{EmbedTask, PromptTokenizer, TokenizedBatch, TokenizerConfig};

#[cfg(all(feature = "metal", target_os = "macos"))]
pub use metal::model::{MetalEmbeddingModel, MetalForwardProfile};

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
pub use cuda::model::{CudaEmbeddingModel, CudaForwardProfile};

/// Whether this build carries a GPU backend for the current target. Callers
/// assert on it at compile time: a CPU-only binary is twenty times slower at
/// exactly the work greppy does on every query, and nothing fails when it is
/// missing — it just gets slow, which is why it has to be caught by the
/// compiler rather than noticed later.
/// CUDA additionally requires that the kernels were actually compiled: the
/// `cuda` feature is enabled unconditionally on Linux/Windows by a target
/// dependency, so the feature alone says only that a backend was *wanted*.
/// `embed_native_has_cuda_dylib` is set by the build script when nvcc produced
/// the dylib, which is what makes a missing toolchain surface as the honest
/// "no GPU backend for this target" error instead of a silent CPU build.
pub const HAS_GPU_BACKEND: bool = cfg!(any(
    all(feature = "metal", target_os = "macos"),
    all(
        feature = "cuda",
        embed_native_has_cuda_dylib,
        any(target_os = "linux", target_os = "windows")
    )
));

/// Embedding dimension produced by EmbeddingGemma after the two
/// SentenceTransformer dense projections.
pub const EMBEDDING_DIM: usize = 768;

/// The prompt/weight contract used by greppy vector rows.
pub const PROMPT_VERSION: &str = "embeddinggemma-code-retrieval-st-v2";

/// greppy vector-store profile key for code retrieval.
pub const CODE_RETRIEVAL_PROFILE: &str = "embeddinggemma_code_retrieval";

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DevicePreference {
    Auto,
    Cpu,
    Metal,
    Cuda,
}

impl DevicePreference {
    pub fn parse(value: &str) -> Result<Self> {
        let normalized = value.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            "cuda" => Ok(Self::Cuda),
            selector if selector.starts_with("cuda:") => {
                let index = selector.trim_start_matches("cuda:");
                if index.parse::<i32>().is_ok_and(|index| index >= 0) {
                    Ok(Self::Cuda)
                } else {
                    Err(Error::InvalidGguf(format!(
                        "unsupported CUDA device selector `{value}`; expected cuda:INDEX"
                    )))
                }
            }
            other => Err(Error::InvalidGguf(format!(
                "unsupported device `{other}`; expected auto|cpu|metal|cuda[:INDEX]"
            ))),
        }
    }

    /// Canonical CLI spelling; round-trips through [`DevicePreference::parse`].
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }
}

impl std::str::FromStr for DevicePreference {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        Self::parse(s)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadOptions {
    pub device: DevicePreference,
    pub max_length: Option<usize>,
    /// Accepted for API compatibility with the old product path. Native
    /// tokenization currently loads directly from `tokenizer.json`.
    pub tokenizer_cache_dir: Option<std::path::PathBuf>,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            device: DevicePreference::Cpu,
            max_length: None,
            tokenizer_cache_dir: None,
        }
    }
}

impl LoadOptions {
    pub fn cpu_f32() -> Self {
        Self::default()
    }

    pub fn auto() -> Self {
        Self {
            device: DevicePreference::Auto,
            ..Self::default()
        }
    }
}

enum EmbeddingBackend {
    Cpu(CpuEmbeddingModel),
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(MetalEmbeddingModel),
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    Cuda(CudaEmbeddingModel),
}

/// Production EmbeddingGemma API used by greppy indexing and vector search.
pub struct EmbeddingGemma {
    tokenizer: PromptTokenizer,
    backend: EmbeddingBackend,
}

impl EmbeddingGemma {
    pub fn load_gguf<P: AsRef<std::path::Path>, Q: AsRef<std::path::Path>>(
        gguf_path: P,
        tokenizer_json_path: Q,
        options: LoadOptions,
    ) -> Result<Self> {
        let gguf_path = gguf_path.as_ref();
        if matches!(
            options.device,
            DevicePreference::Metal | DevicePreference::Cuda
        ) {
            let selector = if options.device == DevicePreference::Cuda {
                std::env::var("EMBED_NATIVE_CUDA_DEVICE")
                    .ok()
                    .map(|index| format!("cuda:{index}"))
                    .unwrap_or_else(|| "cuda".into())
            } else {
                "metal".into()
            };
            let policy = InferencePolicy::from_selector(Some(&selector), false)?;
            preflight_explicit_model(
                &policy,
                InferenceModelKind::EmbeddingGemma,
                std::fs::metadata(gguf_path)?.len(),
            )?;
        }
        let gguf = GgufModel::open(gguf_path)?;
        let mut tokenizer_config = TokenizerConfig::from_gguf(&gguf)?;
        if let Some(max_length) = options.max_length {
            tokenizer_config.max_length = max_length.max(1).min(tokenizer_config.max_length.max(1));
        }
        let tokenizer = PromptTokenizer::from_file(tokenizer_json_path, tokenizer_config)?;
        let backend = load_backend(&gguf, &options.device)?;
        Ok(Self { tokenizer, backend })
    }

    pub fn embed_one(&self, task: EmbedTask, content: &str) -> Result<Vec<f32>> {
        let mut batch = self.embed_prompts([task.prompt(content)])?;
        batch
            .pop()
            .ok_or_else(|| Error::InvalidGguf("empty embedding batch".into()))
    }

    pub fn embed_document(&self, title: Option<&str>, content: &str) -> Result<Vec<f32>> {
        let mut batch = self.embed_documents(&[(title, content)])?;
        batch
            .pop()
            .ok_or_else(|| Error::InvalidGguf("empty embedding batch".into()))
    }

    pub fn embed_documents(&self, docs: &[(Option<&str>, &str)]) -> Result<Vec<Vec<f32>>> {
        let prompts = docs
            .iter()
            .map(|(title, content)| EmbedTask::document_with_title(*title, content))
            .collect::<Vec<_>>();
        self.embed_prompts(prompts)
    }

    pub fn embed_prompts<S, I>(&self, prompts: I) -> Result<Vec<Vec<f32>>>
    where
        S: AsRef<str>,
        I: IntoIterator<Item = S>,
    {
        let batch = self.tokenizer.encode_prompts(prompts)?;
        if batch.is_empty() {
            return Ok(Vec::new());
        }
        match &self.backend {
            EmbeddingBackend::Cpu(model) => model.forward_batch(&batch),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            EmbeddingBackend::Metal(model) => model.forward_batch(&batch),
            #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
            EmbeddingBackend::Cuda(model) => model.forward_batch(&batch),
        }
    }

    pub fn token_len(&self, text: &str) -> Result<usize> {
        self.tokenizer.token_len(text)
    }

    pub fn document_token_len(&self, title: Option<&str>, content: &str) -> Result<usize> {
        self.token_len(&EmbedTask::document_with_title(title, content))
    }

    pub fn max_length(&self) -> usize {
        self.tokenizer.max_length()
    }

    /// One state per input line from ONE forward pass over the whole window.
    ///
    /// The pooled sentence vector keeps semantics and discards position, so a
    /// head fitted to it cannot localise a block; measured, that path scores
    /// 6-8% precision on the rare classes. bash-smart's classifier needs the
    /// per-line states the spec asks for: "a small head reads the hidden state
    /// at each line-end token — ONE forward pass per window".
    ///
    /// Lines are tokenized individually so the token span of each line is known
    /// by construction, then concatenated into a single sequence. The final
    /// layer's states are mean-pooled over each line's span.
    ///
    /// CPU only: `forward_stages` is implemented on `CpuEmbeddingModel` alone.
    /// Metal and CUDA compute the same intermediates but do not surface them,
    /// so shipping a head that needs them requires exposing them there too.
    pub fn line_states(&self, lines: &[String]) -> Result<Vec<Vec<f32>>> {
        let EmbeddingBackend::Cpu(model) = &self.backend else {
            return Err(Error::InvalidGguf(
                "line_states needs per-token states, which only the CPU backend exposes".into(),
            ));
        };
        // The window is a TOKEN budget, not a line count. 64 lines of build
        // output measured 4600 tokens against a 2048 position limit — lines are
        // the unit the product speaks in, tokens are the unit the model can
        // hold, and only the second one is a hard wall.
        let budget = self
            .tokenizer
            .max_length()
            .min(self.cfg_max_positions())
            .saturating_sub(8)
            .max(64);
        let mut encoded: Vec<Vec<u32>> = Vec::with_capacity(lines.len());
        for line in lines {
            let mut tokens = self.tokenizer.encode_ids(line)?;
            if tokens.is_empty() {
                tokens.push(0);
            }
            tokens.truncate(budget);
            encoded.push(tokens);
        }
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(lines.len());
        let mut cursor = 0usize;
        while cursor < encoded.len() {
            let mut ids: Vec<u32> = Vec::new();
            let mut spans: Vec<(usize, usize)> = Vec::new();
            while cursor < encoded.len()
                && (ids.is_empty() || ids.len() + encoded[cursor].len() <= budget)
            {
                let start = ids.len();
                ids.extend_from_slice(&encoded[cursor]);
                spans.push((start, ids.len()));
                cursor += 1;
            }
            out.extend(self.states_for_chunk(model, &ids, &spans)?);
        }
        return Ok(out);
    }

    fn cfg_max_positions(&self) -> usize {
        2048
    }

    fn states_for_chunk(
        &self,
        model: &CpuEmbeddingModel,
        ids: &[u32],
        spans: &[(usize, usize)],
    ) -> Result<Vec<Vec<f32>>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let mask = vec![1u32; ids.len()];
        let stages = model.forward_stages(ids, &mask)?;
        // Which layer to read is a measured choice, not an obvious one: the
        // final layer of a retrieval-tuned encoder is anisotropic — on a sample
        // wall every pairwise cosine sat between 0.977 and 1.000, which leaves a
        // linear head almost no room. GREPPY_HEAD_LAYER selects; the default
        // stays the last so behaviour is unchanged unless asked.
        let wanted = std::env::var("GREPPY_HEAD_LAYER").ok();
        let plain: Vec<_> = stages
            .iter()
            .filter(|s| {
                s.name.starts_with("layer_") && s.name[6..].chars().all(|c| c.is_ascii_digit())
            })
            .collect();
        let last = match wanted.as_deref() {
            Some(index) => plain
                .iter()
                .find(|s| s.name == format!("layer_{index}"))
                .copied(),
            None => plain.last().copied(),
        }
        .ok_or_else(|| Error::InvalidGguf("no layer stage captured".into()))?;
        let dim = last.values.len() / ids.len().max(1);
        if dim == 0 {
            return Err(Error::InvalidGguf("empty layer stage".into()));
        }
        let mut out = Vec::with_capacity(spans.len());
        for &(start, end) in spans {
            let mut acc = vec![0f32; dim];
            let count = (end - start).max(1) as f32;
            for token in start..end {
                let base = token * dim;
                for (slot, value) in acc.iter_mut().zip(&last.values[base..base + dim]) {
                    *slot += *value;
                }
            }
            for slot in acc.iter_mut() {
                *slot /= count;
            }
            out.push(acc);
        }
        Ok(out)
    }

    pub fn embedding_dim(&self) -> usize {
        EMBEDDING_DIM
    }

    pub fn backend_name(&self) -> &'static str {
        match &self.backend {
            EmbeddingBackend::Cpu(_) => "cpu",
            #[cfg(all(feature = "metal", target_os = "macos"))]
            EmbeddingBackend::Metal(_) => "metal",
            #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
            EmbeddingBackend::Cuda(_) => "cuda",
        }
    }
}

fn load_backend(model: &GgufModel, preference: &DevicePreference) -> Result<EmbeddingBackend> {
    match preference {
        DevicePreference::Cpu => CpuEmbeddingModel::from_gguf(model).map(EmbeddingBackend::Cpu),
        DevicePreference::Auto => load_auto_backend(model),
        #[cfg(all(feature = "metal", target_os = "macos"))]
        DevicePreference::Metal => {
            MetalEmbeddingModel::from_gguf(model).map(EmbeddingBackend::Metal)
        }
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        DevicePreference::Metal => Err(Error::InvalidGguf(
            "Metal was explicitly requested but is unavailable in this build/platform".into(),
        )),
        #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
        DevicePreference::Cuda => CudaEmbeddingModel::from_gguf(model).map(EmbeddingBackend::Cuda),
        #[cfg(not(all(feature = "cuda", any(target_os = "linux", target_os = "windows"))))]
        DevicePreference::Cuda => Err(Error::InvalidGguf(
            "CUDA was explicitly requested but is unavailable in this build/platform".into(),
        )),
    }
}

fn load_auto_backend(model: &GgufModel) -> Result<EmbeddingBackend> {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        return load_metal_with_cpu_fallback(model);
    }
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    {
        return load_cuda_with_cpu_fallback(model);
    }
    #[cfg(not(any(
        all(feature = "metal", target_os = "macos"),
        all(feature = "cuda", any(target_os = "linux", target_os = "windows"))
    )))]
    {
        CpuEmbeddingModel::from_gguf(model).map(EmbeddingBackend::Cpu)
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn load_metal_with_cpu_fallback(model: &GgufModel) -> Result<EmbeddingBackend> {
    match MetalEmbeddingModel::from_gguf(model) {
        Ok(model) => Ok(EmbeddingBackend::Metal(model)),
        Err(err) => {
            eprintln!("greppy_embed_native: Metal unavailable, falling back to CPU: {err}");
            CpuEmbeddingModel::from_gguf(model).map(EmbeddingBackend::Cpu)
        }
    }
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
fn load_cuda_with_cpu_fallback(model: &GgufModel) -> Result<EmbeddingBackend> {
    match CudaEmbeddingModel::from_gguf(model) {
        Ok(model) => Ok(EmbeddingBackend::Cuda(model)),
        Err(err) => {
            eprintln!("greppy_embed_native: CUDA unavailable, falling back to CPU: {err}");
            CpuEmbeddingModel::from_gguf(model).map(EmbeddingBackend::Cpu)
        }
    }
}

// M1: gguf loader + Q4_K dequant (CPU)
// M2: tokenizer + prompt templates
// M3: CPU forward (full Gemma3 graph) — the thesis spike
// M4: Metal backend   M5: CUDA backend   M6: integration + candle removal

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid GGUF: {0}")]
    InvalidGguf(String),
    #[error("missing tensor {0}")]
    MissingTensor(String),
    #[error("unsupported GGML dtype {0}")]
    UnsupportedDType(GgmlDType),
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[error("CPU inference: {0}")]
    Cpu(String),
}

pub type Result<T> = std::result::Result<T, Error>;
