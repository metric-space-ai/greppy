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
