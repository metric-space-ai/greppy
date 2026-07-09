//! Native Qwen3.5-0.8B summarizer contract for `greppy brief`.
//!
//! This crate deliberately owns the prompt, sampling parameters, GGUF
//! inventory validation, tokenizer loading, and deterministic postprocessing
//! for the short purpose bullets appended to `brief` definition spans.
//! It does not link llama.cpp, ggml/libllama, Candle, ONNX, Python, or a
//! server runtime.

#![deny(rust_2018_idioms)]

mod cpu;
mod inventory;
mod model;
mod postprocess;
mod prompt;
mod sampler;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;

pub use inventory::{Qwen35Inventory, QWEN35_08B_EXPECTED};
pub use model::{DevicePreference, LoadOptions, Qwen35Summarizer};
pub use postprocess::{
    postprocess_brief_output, postprocess_triage_output, TriageVerdict, MAX_BRIEF_BULLET_CHARS,
    MAX_TRIAGE_REASON_CHARS,
};
pub use prompt::{brief_prompt, triage_prompt, PROMPT_VERSION, TRIAGE_PROMPT_VERSION};
pub use sampler::{
    apply_sampling_filters, sample_token, GenerationParams, SamplerRng, BRIEF_GENERATION_PARAMS,
    TRIAGE_GENERATION_PARAMS,
};

pub const MODEL_ID: &str = "lmstudio-community/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M";

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("gguf: {0}")]
    Gguf(String),
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[error("generation unavailable: {0}")]
    GenerationUnavailable(String),
    #[error("invalid request: {0}")]
    InvalidRequest(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<greppy_embed_native::Error> for Error {
    fn from(value: greppy_embed_native::Error) -> Self {
        Self::Gguf(value.to_string())
    }
}
