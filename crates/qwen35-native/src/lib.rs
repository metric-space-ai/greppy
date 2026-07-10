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
mod simd_math;

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

pub const MODEL_ID: &str = "unsloth/Qwen3.5-0.8B-MTP-GGUF/Qwen3.5-0.8B-MTP-Q4_K_M";

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

pub(crate) struct MtpPerfTimer {
    generation_started: Option<std::time::Instant>,
    stage_started: Option<std::time::Instant>,
    decode_started: Option<std::time::Instant>,
    target_prefill: std::time::Duration,
    mtp_prefill: std::time::Duration,
    mtp_state_copy: std::time::Duration,
    draft: std::time::Duration,
    target_state_copy: std::time::Duration,
    target_verify: std::time::Duration,
    target_replay: std::time::Duration,
    mtp_commit: std::time::Duration,
}

#[derive(Clone, Copy)]
pub(crate) enum MtpPerfStage {
    MtpStateCopy,
    Draft,
    TargetStateCopy,
    TargetVerify,
    TargetReplay,
    MtpCommit,
}

const MTP_FALLBACK_MIN_DRAFTS: usize = 4;

pub(crate) fn mtp_should_fallback(drafted: usize, accepted: usize) -> bool {
    drafted >= MTP_FALLBACK_MIN_DRAFTS && accepted.saturating_mul(4) < drafted.saturating_mul(3)
}

impl MtpPerfTimer {
    pub(crate) fn new() -> Self {
        Self {
            generation_started: std::env::var_os("GREPPY_QWEN35_MTP_PERF")
                .is_some()
                .then(std::time::Instant::now),
            stage_started: None,
            decode_started: None,
            target_prefill: std::time::Duration::ZERO,
            mtp_prefill: std::time::Duration::ZERO,
            mtp_state_copy: std::time::Duration::ZERO,
            draft: std::time::Duration::ZERO,
            target_state_copy: std::time::Duration::ZERO,
            target_verify: std::time::Duration::ZERO,
            target_replay: std::time::Duration::ZERO,
            mtp_commit: std::time::Duration::ZERO,
        }
    }

    pub(crate) fn begin_target_prefill(&mut self) {
        if self.generation_started.is_some() {
            self.stage_started = Some(std::time::Instant::now());
        }
    }

    pub(crate) fn finish_target_prefill(&mut self) {
        if let Some(started) = self.stage_started.take() {
            self.target_prefill = started.elapsed();
        }
    }

    pub(crate) fn begin_mtp_prefill(&mut self) {
        if self.generation_started.is_some() {
            self.stage_started = Some(std::time::Instant::now());
        }
    }

    pub(crate) fn finish_input(&mut self) {
        if let Some(started) = self.stage_started.take() {
            self.mtp_prefill = started.elapsed();
            self.decode_started = Some(std::time::Instant::now());
        }
    }

    pub(crate) fn begin_stage(&self) -> Option<std::time::Instant> {
        self.generation_started.map(|_| std::time::Instant::now())
    }

    pub(crate) fn finish_stage(
        &mut self,
        stage: MtpPerfStage,
        started: Option<std::time::Instant>,
    ) {
        let Some(started) = started else {
            return;
        };
        let elapsed = started.elapsed();
        match stage {
            MtpPerfStage::MtpStateCopy => self.mtp_state_copy += elapsed,
            MtpPerfStage::Draft => self.draft += elapsed,
            MtpPerfStage::TargetStateCopy => self.target_state_copy += elapsed,
            MtpPerfStage::TargetVerify => self.target_verify += elapsed,
            MtpPerfStage::TargetReplay => self.target_replay += elapsed,
            MtpPerfStage::MtpCommit => self.mtp_commit += elapsed,
        }
    }

    pub(crate) fn report(
        &self,
        backend: &str,
        input_tokens: usize,
        output_tokens: usize,
        cycles: usize,
        drafted: usize,
        accepted: usize,
        fallback: bool,
    ) {
        let (Some(generation_started), Some(decode_started)) =
            (self.generation_started, self.decode_started)
        else {
            return;
        };
        let input_s = (self.target_prefill + self.mtp_prefill).as_secs_f64();
        let output_s = decode_started.elapsed().as_secs_f64();
        let count_f64 =
            |count: usize| f64::from(u32::try_from(count).expect("MTP performance count fits u32"));
        let input_tps = count_f64(input_tokens) / input_s.max(1.0e-9);
        let output_tps = count_f64(output_tokens) / output_s.max(1.0e-9);
        let acceptance = count_f64(accepted) / count_f64(drafted.max(1));
        eprintln!(
            "qwen35_mtp_perf backend={backend} input_tokens={input_tokens} target_prefill_s={:.6} mtp_prefill_s={:.6} input_s={input_s:.6} input_tok_s={input_tps:.2} output_tokens={output_tokens} output_s={output_s:.6} output_tok_s={output_tps:.2} cycles={cycles} drafted={drafted} accepted={accepted} acceptance={acceptance:.4} fallback={fallback} mtp_state_copy_s={:.6} draft_s={:.6} target_state_copy_s={:.6} target_verify_s={:.6} target_replay_s={:.6} mtp_commit_s={:.6} generation_s={:.6}",
            self.target_prefill.as_secs_f64(),
            self.mtp_prefill.as_secs_f64(),
            self.mtp_state_copy.as_secs_f64(),
            self.draft.as_secs_f64(),
            self.target_state_copy.as_secs_f64(),
            self.target_verify.as_secs_f64(),
            self.target_replay.as_secs_f64(),
            self.mtp_commit.as_secs_f64(),
            generation_started.elapsed().as_secs_f64(),
        );
    }
}

#[cfg(test)]
mod mtp_tests {
    #[test]
    fn weak_draft_falls_back_only_after_evidence() {
        assert!(!super::mtp_should_fallback(3, 0));
        assert!(super::mtp_should_fallback(4, 2));
        assert!(!super::mtp_should_fallback(4, 3));
        assert!(!super::mtp_should_fallback(12, 9));
    }
}

impl From<greppy_embed_native::Error> for Error {
    fn from(value: greppy_embed_native::Error) -> Self {
        Self::Gguf(value.to_string())
    }
}
