//! Native Rust inference for Google's EmbeddingGemma SentenceTransformer.
//!
//! This crate intentionally implements only the model path grepplus needs:
//! text/code embeddings for fuzzy graph-node matching. It does not include
//! ingestion, chunking, vector database adapters, or RAG behavior.

#![deny(rust_2018_idioms)]

use std::fs::File;
use std::io::{Read, Seek};
use std::path::Path;
use std::sync::Arc;

use candle_core::quantized::{gguf_file, QMatMul as QuantizedMatMul};
use candle_core::{DType, Device, Module, Result as CandleResult, Tensor, D};
use candle_nn::{linear_b as linear, Activation, Linear, VarBuilder};
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

/// Embedding dimension produced by EmbeddingGemma after the two
/// SentenceTransformer dense projections.
pub const EMBEDDING_DIM: usize = 768;

/// The prompt/weight contract used by this crate. Bump this if any prompt,
/// pooling, dense projection, or normalization behavior changes.
pub const PROMPT_VERSION: &str = "embeddinggemma-code-retrieval-st-v1";

/// grepplus vector-store profile key for code retrieval.
///
/// Stored code spans use the retrieval-document prompt, while user queries use
/// the code-retrieval prompt. This key names that paired profile in the vector
/// store and search scope.
pub const CODE_RETRIEVAL_PROFILE: &str = "embeddinggemma_code_retrieval";

/// Weights baked into the binary by `build.rs` when the `embed-weights`
/// feature is enabled. `EMBEDDED_GGUF` / `EMBEDDED_TOKENIZER` are `&'static`
/// rodata (demand-paged by the OS, no heap copy of the whole file), letting
/// [`EmbeddingGemma::load_embedded`] build the model with no external file.
#[cfg(feature = "embed-weights")]
mod embedded_weights {
    include!(concat!(env!("OUT_DIR"), "/embedded_weights_paths.rs"));
}

/// The quantized EmbeddingGemma GGUF baked into the binary (feature-gated).
#[cfg(feature = "embed-weights")]
pub static EMBEDDED_GGUF: &[u8] = embedded_weights::EMBEDDED_GGUF;

/// The EmbeddingGemma tokenizer JSON baked into the binary (feature-gated).
#[cfg(feature = "embed-weights")]
pub static EMBEDDED_TOKENIZER: &[u8] = embedded_weights::EMBEDDED_TOKENIZER;

/// Whether this build has EmbeddingGemma weights compiled in. `false` unless
/// built with `--features embed-weights`. Callers use this to decide whether
/// [`EmbeddingGemma::load_embedded`] is available.
pub const fn has_embedded_weights() -> bool {
    cfg!(feature = "embed-weights")
}

const GGUF_ARCHITECTURE: &str = "gemma-embedding";
const DEFAULT_GGUF_SLIDING_PATTERN: usize = 6;
const DEFAULT_GGUF_LOCAL_ROPE_FREQ: f64 = 10_000.0;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[error("candle: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("invalid model: {0}")]
    InvalidModel(String),
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceDType {
    F32,
    BF16,
    F16,
}

impl InferenceDType {
    fn candle(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::BF16 => DType::BF16,
            Self::F16 => DType::F16,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DevicePreference {
    Cpu,
    #[cfg(feature = "metal")]
    Metal(usize),
    #[cfg(feature = "cuda")]
    Cuda(usize),
}

impl DevicePreference {
    fn candle(&self) -> Result<Device> {
        match self {
            Self::Cpu => Ok(Device::Cpu),
            #[cfg(feature = "metal")]
            Self::Metal(idx) => Device::new_metal(*idx).map_err(Error::from),
            #[cfg(feature = "cuda")]
            Self::Cuda(idx) => Device::new_cuda(*idx).map_err(Error::from),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadOptions {
    pub device: DevicePreference,
    pub dtype: InferenceDType,
    pub max_length: Option<usize>,
    /// Directory for the tokenizer fast-load sidecar (see
    /// [`tokenizer_cache`]). `None` disables the sidecar and parses
    /// `tokenizer.json` from scratch on every load. The CLI passes the
    /// per-workspace store dir here so the cache lives next to `graph.db`
    /// and respects `GREPPLUS_STORE_DIR`.
    pub tokenizer_cache_dir: Option<std::path::PathBuf>,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            device: DevicePreference::Cpu,
            dtype: InferenceDType::F32,
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
            device: default_accelerated_device(),
            ..Self::default()
        }
    }
}

fn default_accelerated_device() -> DevicePreference {
    #[cfg(feature = "cuda")]
    {
        return DevicePreference::Cuda(0);
    }
    #[cfg(all(not(feature = "cuda"), feature = "metal"))]
    {
        return DevicePreference::Metal(0);
    }
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    {
        DevicePreference::Cpu
    }
}

/// EmbeddingGemma task prompt selection. The string forms intentionally match
/// the model's SentenceTransformer prompt table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedTask {
    RetrievalQuery,
    RetrievalDocument,
    CodeRetrievalQuery,
    QuestionAnswering,
    FactVerification,
    Classification,
    Clustering,
    SentenceSimilarity,
}

impl EmbedTask {
    pub fn prompt(self, content: &str) -> String {
        match self {
            Self::RetrievalQuery => format!("task: search result | query: {content}"),
            Self::RetrievalDocument => format!("title: none | text: {content}"),
            Self::CodeRetrievalQuery => format!("task: code retrieval | query: {content}"),
            Self::QuestionAnswering => format!("task: question answering | query: {content}"),
            Self::FactVerification => format!("task: fact checking | query: {content}"),
            Self::Classification => format!("task: classification | query: {content}"),
            Self::Clustering => format!("task: clustering | query: {content}"),
            Self::SentenceSimilarity => format!("task: sentence similarity | query: {content}"),
        }
    }

    pub fn document_with_title(title: Option<&str>, content: &str) -> String {
        format!("title: {} | text: {content}", title.unwrap_or("none"))
    }
}

#[derive(Debug, serde::Deserialize, Clone)]
struct Config {
    attention_bias: bool,
    attention_dropout: f64,
    attn_logit_softcapping: Option<f64>,
    head_dim: usize,
    hidden_activation: Activation,
    hidden_size: usize,
    intermediate_size: usize,
    layer_types: Vec<LayerType>,
    max_position_embeddings: usize,
    model_type: String,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    pad_token_id: u32,
    query_pre_attn_scalar: f64,
    rms_norm_eps: f64,
    rope_local_base_freq: f64,
    rope_theta: f64,
    sliding_window: usize,
    use_bidirectional_attention: bool,
    vocab_size: usize,
}

#[derive(Debug, Clone)]
struct GgufConfig {
    hidden_size: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    pad_token_id: u32,
    rms_norm_eps: f64,
    rope_theta: f64,
    rope_local_base_freq: f64,
    sliding_window: usize,
    layer_types: Vec<LayerType>,
    query_pre_attn_scalar: f64,
}

impl GgufConfig {
    fn from_content(ct: &gguf_file::Content) -> Result<Self> {
        let arch = gguf_string(ct, "general.architecture")?;
        if arch != GGUF_ARCHITECTURE {
            return Err(Error::InvalidModel(format!(
                "expected GGUF architecture {GGUF_ARCHITECTURE}, got {arch}"
            )));
        }

        let hidden_size = gguf_u32(ct, "gemma-embedding.embedding_length")? as usize;
        if hidden_size != EMBEDDING_DIM {
            return Err(Error::InvalidModel(format!(
                "expected embedding_length {EMBEDDING_DIM}, got {hidden_size}"
            )));
        }

        let num_hidden_layers = gguf_u32(ct, "gemma-embedding.block_count")? as usize;
        let sliding_pattern = gguf_u32_opt(ct, "gemma-embedding.attention.sliding_window_type")
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_GGUF_SLIDING_PATTERN);
        let layer_types = (0..num_hidden_layers)
            .map(|idx| {
                if (idx + 1) % sliding_pattern == 0 {
                    LayerType::FullAttention
                } else {
                    LayerType::SlidingAttention
                }
            })
            .collect::<Vec<_>>();

        Ok(Self {
            hidden_size,
            intermediate_size: gguf_u32(ct, "gemma-embedding.feed_forward_length")? as usize,
            max_position_embeddings: gguf_u32(ct, "gemma-embedding.context_length")? as usize,
            num_attention_heads: gguf_u32(ct, "gemma-embedding.attention.head_count")? as usize,
            num_hidden_layers,
            num_key_value_heads: gguf_u32(ct, "gemma-embedding.attention.head_count_kv")? as usize,
            head_dim: gguf_u32(ct, "gemma-embedding.attention.key_length")? as usize,
            pad_token_id: gguf_u32(ct, "tokenizer.ggml.padding_token_id")?,
            rms_norm_eps: f64::from(gguf_f32(
                ct,
                "gemma-embedding.attention.layer_norm_rms_epsilon",
            )?),
            rope_theta: f64::from(gguf_f32(ct, "gemma-embedding.rope.freq_base")?),
            rope_local_base_freq: gguf_f32_opt(ct, "gemma-embedding.rope.local_freq_base")
                .map(f64::from)
                .unwrap_or(DEFAULT_GGUF_LOCAL_ROPE_FREQ),
            sliding_window: gguf_u32(ct, "gemma-embedding.attention.sliding_window")? as usize,
            layer_types,
            query_pre_attn_scalar: 256.0,
        })
    }
}

impl Config {
    fn validate(&self) -> Result<()> {
        if self.model_type != "gemma3_text" {
            return Err(Error::InvalidModel(format!(
                "expected model_type gemma3_text, got {}",
                self.model_type
            )));
        }
        if self.hidden_size != EMBEDDING_DIM {
            return Err(Error::InvalidModel(format!(
                "expected hidden_size {EMBEDDING_DIM}, got {}",
                self.hidden_size
            )));
        }
        if self.layer_types.len() != self.num_hidden_layers {
            return Err(Error::InvalidModel(format!(
                "layer_types len {} does not match num_hidden_layers {}",
                self.layer_types.len(),
                self.num_hidden_layers
            )));
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(Error::InvalidModel(
                "num_attention_heads must be divisible by num_key_value_heads".into(),
            ));
        }
        if self.attention_dropout != 0.0 {
            return Err(Error::InvalidModel(
                "non-zero attention_dropout is not supported for inference".into(),
            ));
        }
        Ok(())
    }

    fn effective_sliding_window(&self) -> usize {
        if self.use_bidirectional_attention {
            (self.sliding_window / 2) + 1
        } else {
            self.sliding_window
        }
    }
}

fn gguf_value<'a>(ct: &'a gguf_file::Content, key: &str) -> Result<&'a gguf_file::Value> {
    ct.metadata
        .get(key)
        .ok_or_else(|| Error::InvalidModel(format!("missing GGUF metadata key {key}")))
}

fn gguf_string(ct: &gguf_file::Content, key: &str) -> Result<String> {
    gguf_value(ct, key)?
        .to_string()
        .map(|s| s.to_string())
        .map_err(Error::from)
}

fn gguf_u32(ct: &gguf_file::Content, key: &str) -> Result<u32> {
    gguf_value(ct, key)?.to_u32().map_err(Error::from)
}

fn gguf_u32_opt(ct: &gguf_file::Content, key: &str) -> Option<u32> {
    ct.metadata.get(key).and_then(|v| v.to_u32().ok())
}

fn gguf_f32(ct: &gguf_file::Content, key: &str) -> Result<f32> {
    gguf_value(ct, key)?.to_f32().map_err(Error::from)
}

fn gguf_f32_opt(ct: &gguf_file::Content, key: &str) -> Option<f32> {
    ct.metadata.get(key).and_then(|v| v.to_f32().ok())
}

#[derive(Debug, serde::Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum LayerType {
    SlidingAttention,
    FullAttention,
}

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder<'_>) -> CandleResult<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        x_normed
            .broadcast_mul(&(&self.weight.to_dtype(DType::F32)? + 1.0)?)?
            .to_dtype(x_dtype)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(
        dtype: DType,
        head_dim: usize,
        max_seq_len: usize,
        base: f64,
        dev: &Device,
    ) -> CandleResult<Self> {
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / base.powf(i as f64 / head_dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbeddings {
    full: RotaryEmbedding,
    sliding: RotaryEmbedding,
}

impl RotaryEmbeddings {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> CandleResult<Self> {
        Ok(Self {
            full: RotaryEmbedding::new(
                dtype,
                cfg.head_dim,
                cfg.max_position_embeddings,
                cfg.rope_theta,
                dev,
            )?,
            sliding: RotaryEmbedding::new(
                dtype,
                cfg.head_dim,
                cfg.max_position_embeddings,
                cfg.rope_local_base_freq,
                dev,
            )?,
        })
    }

    fn for_layer(&self, layer_type: LayerType) -> &RotaryEmbedding {
        match layer_type {
            LayerType::FullAttention => &self.full,
            LayerType::SlidingAttention => &self.sliding,
        }
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder<'_>) -> CandleResult<Self> {
        Ok(Self {
            gate_proj: linear(
                cfg.hidden_size,
                cfg.intermediate_size,
                false,
                vb.pp("gate_proj"),
            )?,
            up_proj: linear(
                cfg.hidden_size,
                cfg.intermediate_size,
                false,
                vb.pp("up_proj"),
            )?,
            down_proj: linear(
                cfg.intermediate_size,
                cfg.hidden_size,
                false,
                vb.pp("down_proj"),
            )?,
            act_fn: cfg.hidden_activation,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    scaling: f64,
    attn_logit_softcapping: Option<f64>,
    layer_type: LayerType,
    rotary_emb: Arc<RotaryEmbeddings>,
}

impl Attention {
    fn new(
        rotary_emb: Arc<RotaryEmbeddings>,
        cfg: &Config,
        layer_type: LayerType,
        vb: VarBuilder<'_>,
    ) -> CandleResult<Self> {
        let bias = cfg.attention_bias;
        Ok(Self {
            q_proj: linear(
                cfg.hidden_size,
                cfg.num_attention_heads * cfg.head_dim,
                bias,
                vb.pp("q_proj"),
            )?,
            k_proj: linear(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim,
                bias,
                vb.pp("k_proj"),
            )?,
            v_proj: linear(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim,
                bias,
                vb.pp("v_proj"),
            )?,
            o_proj: linear(
                cfg.num_attention_heads * cfg.head_dim,
                cfg.hidden_size,
                bias,
                vb.pp("o_proj"),
            )?,
            q_norm: RmsNorm::new(cfg.head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: RmsNorm::new(cfg.head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            num_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            scaling: cfg.query_pre_attn_scalar.powf(-0.5),
            attn_logit_softcapping: cfg.attn_logit_softcapping,
            layer_type,
            rotary_emb,
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> CandleResult<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let query_states = self.q_norm.forward(&query_states)?;
        let key_states = self.k_norm.forward(&key_states)?;
        let (query_states, key_states) = self
            .rotary_emb
            .for_layer(self.layer_type)
            .apply(&query_states, &key_states)?;

        let key_states = repeat_kv(key_states, self.num_kv_groups)?.contiguous()?;
        let value_states = repeat_kv(value_states, self.num_kv_groups)?.contiguous()?;

        let mut attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * self.scaling)?;
        if let Some(sc) = self.attn_logit_softcapping {
            attn_weights = ((attn_weights / sc)?.tanh()? * sc)?;
        }
        if let Some(mask) = attention_mask {
            attn_weights = attn_weights.broadcast_add(mask)?;
        }

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights.to_dtype(DType::F32)?)?
            .to_dtype(query_states.dtype())?;
        let attn_output = attn_weights.matmul(&value_states)?;
        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, ()))?
            .apply(&self.o_proj)
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    layer_type: LayerType,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<RotaryEmbeddings>,
        cfg: &Config,
        layer_idx: usize,
        vb: VarBuilder<'_>,
    ) -> CandleResult<Self> {
        let layer_type = cfg.layer_types[layer_idx];
        Ok(Self {
            self_attn: Attention::new(rotary_emb, cfg, layer_type, vb.pp("self_attn"))?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
            input_layernorm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            pre_feedforward_layernorm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("pre_feedforward_layernorm"),
            )?,
            post_feedforward_layernorm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_feedforward_layernorm"),
            )?,
            layer_type,
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> CandleResult<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask)?;
        let xs = xs.apply(&self.post_attention_layernorm)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.pre_feedforward_layernorm)?;
        let xs = xs.apply(&self.mlp)?;
        let xs = xs.apply(&self.post_feedforward_layernorm)?;
        residual + xs
    }
}

#[derive(Debug, Clone)]
struct Gemma3TextModel {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    cfg: Config,
    device: Device,
    dtype: DType,
}

impl Gemma3TextModel {
    fn new(cfg: Config, vb: VarBuilder<'_>) -> CandleResult<Self> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbeddings::new(vb.dtype(), &cfg, vb.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(
                rotary_emb.clone(),
                &cfg,
                layer_idx,
                vb_l.pp(layer_idx),
            )?);
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            cfg,
        })
    }

    fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> CandleResult<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let masks = LayerMasks::new(
            attention_mask,
            b_size,
            seq_len,
            self.cfg.use_bidirectional_attention,
            self.cfg.effective_sliding_window(),
            &self.device,
            self.dtype,
        )?;

        let xs = self.embed_tokens.forward(input_ids)?;
        let mut xs = (xs * (self.cfg.hidden_size as f64).sqrt())?;
        for layer in &self.layers {
            xs = layer.forward(&xs, masks.for_layer(layer.layer_type))?;
        }
        xs.apply(&self.norm)
    }
}

struct LayerMasks {
    full: Option<Tensor>,
    sliding: Option<Tensor>,
}

impl LayerMasks {
    fn new(
        attention_mask: &Tensor,
        batch: usize,
        seq_len: usize,
        bidirectional: bool,
        sliding_window: usize,
        device: &Device,
        dtype: DType,
    ) -> CandleResult<Self> {
        let mask_rows = attention_mask.to_vec2::<u32>()?;
        let all_unmasked = mask_rows.iter().all(|row| row.iter().all(|&m| m != 0));
        let full = if all_unmasked && bidirectional {
            None
        } else {
            Some(build_attention_mask(
                &mask_rows,
                batch,
                seq_len,
                None,
                bidirectional,
                device,
                dtype,
            )?)
        };
        let sliding_can_be_full = bidirectional && seq_len < sliding_window;
        let sliding = if all_unmasked && sliding_can_be_full {
            None
        } else {
            Some(build_attention_mask(
                &mask_rows,
                batch,
                seq_len,
                Some(sliding_window),
                bidirectional,
                device,
                dtype,
            )?)
        };
        Ok(Self { full, sliding })
    }

    fn for_layer(&self, layer_type: LayerType) -> Option<&Tensor> {
        match layer_type {
            LayerType::FullAttention => self.full.as_ref(),
            LayerType::SlidingAttention => self.sliding.as_ref(),
        }
    }
}

fn build_attention_mask(
    mask_rows: &[Vec<u32>],
    batch: usize,
    seq_len: usize,
    sliding_window: Option<usize>,
    bidirectional: bool,
    device: &Device,
    dtype: DType,
) -> CandleResult<Tensor> {
    let mut values = Vec::with_capacity(batch * seq_len * seq_len);
    for row in mask_rows {
        for q in 0..seq_len {
            for k in 0..seq_len {
                let key_visible = row[k] != 0;
                let direction_visible = bidirectional || k <= q;
                let window_visible = match sliding_window {
                    None => true,
                    Some(window) if bidirectional => q.abs_diff(k) < window,
                    Some(window) => k <= q && q - k < window,
                };
                values.push(if key_visible && direction_visible && window_visible {
                    0.0
                } else {
                    -1.0e9
                });
            }
        }
    }
    Tensor::from_vec(values, (batch, 1, seq_len, seq_len), device)?.to_dtype(dtype)
}

fn repeat_kv(xs: Tensor, n_rep: usize) -> CandleResult<Tensor> {
    if n_rep == 1 {
        return Ok(xs);
    }
    let (batch, kv_heads, seq_len, head_dim) = xs.dims4()?;
    xs.unsqueeze(2)?
        .expand((batch, kv_heads, n_rep, seq_len, head_dim))?
        .reshape((batch, kv_heads * n_rep, seq_len, head_dim))
}

#[derive(Debug, Clone)]
struct SentenceTransformerHead {
    dense2: Linear,
    dense3: Linear,
}

impl SentenceTransformerHead {
    fn new(model_dir: &Path, dtype: DType, device: &Device) -> Result<Self> {
        let dense2_path = model_dir.join("2_Dense").join("model.safetensors");
        let dense3_path = model_dir.join("3_Dense").join("model.safetensors");
        let vb2 = unsafe { VarBuilder::from_mmaped_safetensors(&[dense2_path], dtype, device)? };
        let vb3 = unsafe { VarBuilder::from_mmaped_safetensors(&[dense3_path], dtype, device)? };
        Ok(Self {
            dense2: linear(EMBEDDING_DIM, 3072, false, vb2.pp("linear"))?,
            dense3: linear(3072, EMBEDDING_DIM, false, vb3.pp("linear"))?,
        })
    }

    fn forward(&self, hidden: &Tensor, attention_mask: &Tensor) -> CandleResult<Tensor> {
        let pooled = mean_pool(hidden, attention_mask)?;
        let projected = pooled.apply(&self.dense2)?.apply(&self.dense3)?;
        normalize_l2(&projected)
    }
}

fn mean_pool(hidden: &Tensor, attention_mask: &Tensor) -> CandleResult<Tensor> {
    let mask = attention_mask.to_dtype(DType::F32)?.unsqueeze(2)?;
    let hidden = hidden.to_dtype(DType::F32)?;
    let summed = hidden.broadcast_mul(&mask)?.sum(1)?;
    let counts = mask.sum(1)?.clamp(1e-12, f32::MAX)?;
    summed.broadcast_div(&counts)
}

fn normalize_l2(xs: &Tensor) -> CandleResult<Tensor> {
    let denom = xs
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .sqrt()?
        .clamp(1e-12, f32::MAX)?;
    xs.broadcast_div(&denom)
}

fn configure_tokenizer(tokenizer: &mut Tokenizer, max_length: usize, pad_id: u32) -> Result<()> {
    tokenizer
        .with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Right,
            pad_id,
            pad_type_id: 0,
            pad_token: "<pad>".into(),
            ..Default::default()
        }))
        .with_truncation(Some(TruncationParams {
            max_length,
            ..Default::default()
        }))
        .map_err(|e| Error::Tokenizer(e.to_string()))?;
    Ok(())
}

#[derive(Debug, Clone)]
struct QLinear {
    weight: QuantizedMatMul,
}

impl QLinear {
    fn from_gguf<R: Read + Seek>(
        ct: &gguf_file::Content,
        reader: &mut R,
        name: &str,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            weight: QuantizedMatMul::from_qtensor(ct.tensor(reader, name, device)?)?,
        })
    }
}

impl Module for QLinear {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        self.weight.forward(xs)
    }
}

#[derive(Debug, Clone)]
struct QEmbedding {
    weight: QuantizedMatMul,
}

impl QEmbedding {
    fn from_gguf<R: Read + Seek>(
        ct: &gguf_file::Content,
        reader: &mut R,
        name: &str,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            weight: QuantizedMatMul::from_qtensor(ct.tensor(reader, name, device)?)?,
        })
    }

    fn forward(&self, ids: &Tensor) -> CandleResult<Tensor> {
        self.weight.embedding(ids)
    }
}

#[derive(Debug, Clone)]
struct GgufRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl GgufRmsNorm {
    fn from_gguf<R: Read + Seek>(
        ct: &gguf_file::Content,
        reader: &mut R,
        name: &str,
        eps: f64,
        device: &Device,
    ) -> Result<Self> {
        let weight = ct.tensor(reader, name, device)?;
        Ok(Self {
            weight: weight.dequantize(device)?,
            eps,
        })
    }
}

impl Module for GgufRmsNorm {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        candle_nn::ops::rms_norm(x, &self.weight, self.eps as f32)
    }
}

#[derive(Debug, Clone)]
struct QMlp {
    gate_proj: QLinear,
    up_proj: QLinear,
    down_proj: QLinear,
    act_fn: Activation,
}

impl QMlp {
    fn from_gguf<R: Read + Seek>(
        ct: &gguf_file::Content,
        reader: &mut R,
        prefix: &str,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            gate_proj: QLinear::from_gguf(
                ct,
                reader,
                &format!("{prefix}.ffn_gate.weight"),
                device,
            )?,
            up_proj: QLinear::from_gguf(ct, reader, &format!("{prefix}.ffn_up.weight"), device)?,
            down_proj: QLinear::from_gguf(
                ct,
                reader,
                &format!("{prefix}.ffn_down.weight"),
                device,
            )?,
            act_fn: Activation::GeluPytorchTanh,
        })
    }
}

impl Module for QMlp {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct QAttention {
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    q_norm: GgufRmsNorm,
    k_norm: GgufRmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    scaling: f64,
    layer_type: LayerType,
    rotary_emb: Arc<RotaryEmbeddings>,
}

impl QAttention {
    fn from_gguf<R: Read + Seek>(
        ct: &gguf_file::Content,
        reader: &mut R,
        cfg: &GgufConfig,
        layer_type: LayerType,
        prefix: &str,
        rotary_emb: Arc<RotaryEmbeddings>,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            q_proj: QLinear::from_gguf(ct, reader, &format!("{prefix}.attn_q.weight"), device)?,
            k_proj: QLinear::from_gguf(ct, reader, &format!("{prefix}.attn_k.weight"), device)?,
            v_proj: QLinear::from_gguf(ct, reader, &format!("{prefix}.attn_v.weight"), device)?,
            o_proj: QLinear::from_gguf(
                ct,
                reader,
                &format!("{prefix}.attn_output.weight"),
                device,
            )?,
            q_norm: GgufRmsNorm::from_gguf(
                ct,
                reader,
                &format!("{prefix}.attn_q_norm.weight"),
                cfg.rms_norm_eps,
                device,
            )?,
            k_norm: GgufRmsNorm::from_gguf(
                ct,
                reader,
                &format!("{prefix}.attn_k_norm.weight"),
                cfg.rms_norm_eps,
                device,
            )?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            num_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            scaling: cfg.query_pre_attn_scalar.powf(-0.5),
            layer_type,
            rotary_emb,
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> CandleResult<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let query_states = self.q_norm.forward(&query_states.contiguous()?)?;
        let key_states = self.k_norm.forward(&key_states.contiguous()?)?;
        let (query_states, key_states) = self
            .rotary_emb
            .for_layer(self.layer_type)
            .apply(&query_states, &key_states)?;

        let key_states = repeat_kv(key_states, self.num_kv_groups)?.contiguous()?;
        let value_states = repeat_kv(value_states, self.num_kv_groups)?.contiguous()?;

        let mut attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * self.scaling)?;
        if let Some(mask) = attention_mask {
            attn_weights = attn_weights.broadcast_add(mask)?;
        }
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights.to_dtype(DType::F32)?)?
            .to_dtype(query_states.dtype())?;
        let attn_output = attn_weights.matmul(&value_states)?;
        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, ()))?
            .apply(&self.o_proj)
    }
}

#[derive(Debug, Clone)]
struct QDecoderLayer {
    self_attn: QAttention,
    mlp: QMlp,
    input_layernorm: GgufRmsNorm,
    post_attention_layernorm: GgufRmsNorm,
    pre_feedforward_layernorm: GgufRmsNorm,
    post_feedforward_layernorm: GgufRmsNorm,
    layer_type: LayerType,
}

impl QDecoderLayer {
    fn from_gguf<R: Read + Seek>(
        ct: &gguf_file::Content,
        reader: &mut R,
        cfg: &GgufConfig,
        layer_idx: usize,
        rotary_emb: Arc<RotaryEmbeddings>,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");
        let layer_type = cfg.layer_types[layer_idx];
        Ok(Self {
            self_attn: QAttention::from_gguf(
                ct, reader, cfg, layer_type, &prefix, rotary_emb, device,
            )?,
            mlp: QMlp::from_gguf(ct, reader, &prefix, device)?,
            input_layernorm: GgufRmsNorm::from_gguf(
                ct,
                reader,
                &format!("{prefix}.attn_norm.weight"),
                cfg.rms_norm_eps,
                device,
            )?,
            post_attention_layernorm: GgufRmsNorm::from_gguf(
                ct,
                reader,
                &format!("{prefix}.post_attention_norm.weight"),
                cfg.rms_norm_eps,
                device,
            )?,
            pre_feedforward_layernorm: GgufRmsNorm::from_gguf(
                ct,
                reader,
                &format!("{prefix}.ffn_norm.weight"),
                cfg.rms_norm_eps,
                device,
            )?,
            post_feedforward_layernorm: GgufRmsNorm::from_gguf(
                ct,
                reader,
                &format!("{prefix}.post_ffw_norm.weight"),
                cfg.rms_norm_eps,
                device,
            )?,
            layer_type,
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> CandleResult<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask)?;
        let xs = xs.apply(&self.post_attention_layernorm)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.pre_feedforward_layernorm)?;
        let xs = xs.apply(&self.mlp)?;
        let xs = xs.apply(&self.post_feedforward_layernorm)?;
        residual + xs
    }
}

#[derive(Debug, Clone)]
struct QuantizedSentenceTransformerHead {
    dense2: QLinear,
    dense3: QLinear,
}

impl QuantizedSentenceTransformerHead {
    fn from_gguf<R: Read + Seek>(
        ct: &gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            dense2: QLinear::from_gguf(ct, reader, "dense_2.weight", device)?,
            dense3: QLinear::from_gguf(ct, reader, "dense_3.weight", device)?,
        })
    }

    fn forward(&self, hidden: &Tensor, attention_mask: &Tensor) -> CandleResult<Tensor> {
        let pooled = mean_pool(hidden, attention_mask)?;
        let projected = pooled.apply(&self.dense2)?.apply(&self.dense3)?;
        normalize_l2(&projected)
    }
}

#[derive(Debug, Clone)]
struct QuantizedGemmaEmbedding {
    embed_tokens: QEmbedding,
    layers: Vec<QDecoderLayer>,
    norm: GgufRmsNorm,
    head: QuantizedSentenceTransformerHead,
    cfg: GgufConfig,
    device: Device,
}

impl QuantizedGemmaEmbedding {
    fn from_gguf<R: Read + Seek>(
        ct: &gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let cfg = GgufConfig::from_content(ct)?;
        let embed_tokens = QEmbedding::from_gguf(ct, reader, "token_embd.weight", device)?;
        let rotary_cfg = Config {
            attention_bias: false,
            attention_dropout: 0.0,
            attn_logit_softcapping: None,
            head_dim: cfg.head_dim,
            hidden_activation: Activation::GeluPytorchTanh,
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            layer_types: cfg.layer_types.clone(),
            max_position_embeddings: cfg.max_position_embeddings,
            model_type: "gemma3_text".into(),
            num_attention_heads: cfg.num_attention_heads,
            num_hidden_layers: cfg.num_hidden_layers,
            num_key_value_heads: cfg.num_key_value_heads,
            pad_token_id: cfg.pad_token_id,
            query_pre_attn_scalar: cfg.query_pre_attn_scalar,
            rms_norm_eps: cfg.rms_norm_eps,
            rope_local_base_freq: cfg.rope_local_base_freq,
            rope_theta: cfg.rope_theta,
            sliding_window: cfg.sliding_window,
            use_bidirectional_attention: true,
            vocab_size: 0,
        };
        let rotary_emb = Arc::new(RotaryEmbeddings::new(DType::F32, &rotary_cfg, device)?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in 0..cfg.num_hidden_layers {
            layers.push(QDecoderLayer::from_gguf(
                ct,
                reader,
                &cfg,
                layer_idx,
                rotary_emb.clone(),
                device,
            )?);
        }
        let norm =
            GgufRmsNorm::from_gguf(ct, reader, "output_norm.weight", cfg.rms_norm_eps, device)?;
        let head = QuantizedSentenceTransformerHead::from_gguf(ct, reader, device)?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            head,
            cfg,
            device: device.clone(),
        })
    }

    fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> CandleResult<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let masks = LayerMasks::new(
            attention_mask,
            b_size,
            seq_len,
            true,
            self.cfg.sliding_window,
            &self.device,
            DType::F32,
        )?;

        let xs = self.embed_tokens.forward(input_ids)?;
        let mut xs = (xs * (self.cfg.hidden_size as f64).sqrt())?;
        for layer in &self.layers {
            xs = layer.forward(&xs, masks.for_layer(layer.layer_type))?;
        }
        let hidden = xs.apply(&self.norm)?;
        self.head.forward(&hidden, attention_mask)
    }
}

#[derive(Debug, Clone)]
pub struct EmbeddingGemma {
    tokenizer: Tokenizer,
    backend: EmbeddingBackend,
    device: Device,
}

#[derive(Debug, Clone)]
enum EmbeddingBackend {
    Safetensors {
        model: Gemma3TextModel,
        head: SentenceTransformerHead,
    },
    Gguf {
        model: QuantizedGemmaEmbedding,
    },
}

impl EmbeddingGemma {
    pub fn load_safetensors<P: AsRef<Path>>(model_dir: P, options: LoadOptions) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let config_path = model_dir.join("config.json");
        let tokenizer_path = model_dir.join("tokenizer.json");
        let weights_path = model_dir.join("model.safetensors");

        let cfg: Config = serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;
        cfg.validate()?;

        let mut tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|e| Error::Tokenizer(e.to_string()))?;
        let max_length = options.max_length.unwrap_or(cfg.max_position_embeddings);
        configure_tokenizer(&mut tokenizer, max_length, cfg.pad_token_id)?;

        let device = options.device.candle()?;
        let dtype = options.dtype.candle();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device)? };
        let model = Gemma3TextModel::new(cfg, vb)?;
        let head = SentenceTransformerHead::new(model_dir, dtype, &device)?;

        Ok(Self {
            tokenizer,
            backend: EmbeddingBackend::Safetensors { model, head },
            device,
        })
    }

    pub fn load_gguf<P: AsRef<Path>, Q: AsRef<Path>>(
        gguf_path: P,
        tokenizer_json_path: Q,
        options: LoadOptions,
    ) -> Result<Self> {
        // Tokenizer parsing (~270ms for the 33MB EmbeddingGemma
        // tokenizer.json, ~90ms via the sidecar) dominates GGUF loading
        // (~60-90ms with mmap), so run it on a helper thread that is
        // ALWAYS joined before this function returns — no background
        // work survives the call (single-shot CLI contract).
        let tok_path = tokenizer_json_path.as_ref().to_path_buf();
        let cache_dir = options.tokenizer_cache_dir.clone();
        let tok_handle = std::thread::spawn(move || {
            tokenizer_cache::load_tokenizer_fast(&tok_path, cache_dir.as_deref())
        });

        // mmap the GGUF instead of eagerly streaming it through an 8KB
        // BufReader: `gguf_file::Content::read` issues thousands of small
        // reads for the metadata/tensor-info section (~70-100ms warm via
        // BufReader, ~16ms via mmap'd cursor). Tensor construction below
        // still copies each tensor's bytes out of the mapping into
        // candle's own (aligned) buffers — candle's `QTensor` API has no
        // zero-copy path for quantized CPU tensors — so ~30-60ms of copy
        // cost remains and is measured honestly in the report.
        //
        // SAFETY: the model file is mapped read-only. If another process
        // truncates/rewrites the file while we read it the map could
        // change under us; model files in the HF cache are content-
        // addressed and immutable in practice, matching candle's own
        // mmap usage for safetensors.
        let file = File::open(gguf_path.as_ref())?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let mut cursor = std::io::Cursor::new(&mmap[..]);
        let ct = gguf_file::Content::read(&mut cursor)?;
        let cfg = GgufConfig::from_content(&ct)?;
        let device = options.device.candle()?;
        let model = QuantizedGemmaEmbedding::from_gguf(&ct, &mut cursor, &device)?;

        let mut tokenizer = tok_handle
            .join()
            .map_err(|_| Error::Tokenizer("tokenizer loader thread panicked".into()))??;
        let max_length = options.max_length.unwrap_or(cfg.max_position_embeddings);
        configure_tokenizer(&mut tokenizer, max_length, cfg.pad_token_id)?;

        Ok(Self {
            tokenizer,
            backend: EmbeddingBackend::Gguf { model },
            device,
        })
    }

    /// Load the model from weights baked into the binary by the
    /// `embed-weights` feature — no external GGUF or tokenizer file needed.
    ///
    /// The GGUF is read from a `Cursor` over the `&'static` rodata slice
    /// (candle's `gguf_file::Content::read` + `from_gguf` take any
    /// `Read + Seek`, so no temp file is created; the bytes are demand-paged
    /// by the OS). The tokenizer is parsed with `Tokenizer::from_bytes` over
    /// the embedded slice. The resulting model is byte-for-byte the same as
    /// the external `load_gguf` path fed the same files (same GGUF parse,
    /// same tensor construction, same tokenizer config), so embeddings match.
    ///
    /// The `tokenizer_cache_dir` sidecar is intentionally NOT used here: the
    /// embedded tokenizer has no on-disk source file to fingerprint, and the
    /// whole point of the feature is a self-contained binary.
    #[cfg(feature = "embed-weights")]
    pub fn load_embedded(options: LoadOptions) -> Result<Self> {
        let mut cursor = std::io::Cursor::new(EMBEDDED_GGUF);
        let ct = gguf_file::Content::read(&mut cursor)?;
        let cfg = GgufConfig::from_content(&ct)?;
        let device = options.device.candle()?;
        let model = QuantizedGemmaEmbedding::from_gguf(&ct, &mut cursor, &device)?;

        let mut tokenizer = Tokenizer::from_bytes(EMBEDDED_TOKENIZER)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let max_length = options.max_length.unwrap_or(cfg.max_position_embeddings);
        configure_tokenizer(&mut tokenizer, max_length, cfg.pad_token_id)?;

        Ok(Self {
            tokenizer,
            backend: EmbeddingBackend::Gguf { model },
            device,
        })
    }

    pub fn embed_one(&self, task: EmbedTask, content: &str) -> Result<Vec<f32>> {
        let mut batch = self.embed_prompts(&[task.prompt(content)])?;
        batch
            .pop()
            .ok_or_else(|| Error::InvalidModel("empty embedding batch".into()))
    }

    pub fn embed_document(&self, title: Option<&str>, content: &str) -> Result<Vec<f32>> {
        let mut batch = self.embed_prompts(&[EmbedTask::document_with_title(title, content)])?;
        batch
            .pop()
            .ok_or_else(|| Error::InvalidModel("empty embedding batch".into()))
    }

    pub fn embed_prompts<S: AsRef<str>>(&self, prompts: &[S]) -> Result<Vec<Vec<f32>>> {
        if prompts.is_empty() {
            return Ok(Vec::new());
        }
        let prompt_refs = prompts.iter().map(|s| s.as_ref()).collect::<Vec<_>>();
        let encodings = self
            .tokenizer
            .encode_batch(prompt_refs, true)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let batch = encodings.len();
        let seq_len = encodings
            .first()
            .ok_or_else(|| Error::Tokenizer("tokenizer returned empty batch".into()))?
            .len();
        let mut token_ids = Vec::with_capacity(batch * seq_len);
        let mut attention = Vec::with_capacity(batch * seq_len);
        for enc in encodings {
            if enc.len() != seq_len {
                return Err(Error::Tokenizer(
                    "tokenizer padding failed to produce a rectangular batch".into(),
                ));
            }
            token_ids.extend(enc.get_ids().iter().copied());
            attention.extend(enc.get_attention_mask().iter().copied());
        }
        let input_ids = Tensor::from_vec(token_ids, (batch, seq_len), &self.device)?;
        let attention_mask = Tensor::from_vec(attention, (batch, seq_len), &self.device)?;
        let embeddings = match &self.backend {
            EmbeddingBackend::Safetensors { model, head } => {
                let hidden = model.forward(&input_ids, &attention_mask)?;
                head.forward(&hidden, &attention_mask)?
            }
            EmbeddingBackend::Gguf { model } => model.forward(&input_ids, &attention_mask)?,
        };
        Ok(embeddings.to_dtype(DType::F32)?.to_vec2::<f32>()?)
    }

    pub fn embedding_dim(&self) -> usize {
        EMBEDDING_DIM
    }
}

/// Tokenizer fast-load sidecar.
///
/// Measured split for the 33MB EmbeddingGemma `tokenizer.json` (M-series,
/// warm cache, release build): `Tokenizer::from_bytes` ≈ 255-320ms, of
/// which ≈ 110ms is the raw serde_json scan and the rest is tokenizers'
/// typed deserialization + BPE merges-map construction. A compact
/// re-serialized JSON saves only ~20ms, and tokenizers' serde impls are
/// JSON-only in practice (MessagePack round-trips fail on the BPE model),
/// so a "serialize the whole Tokenizer" sidecar is not possible.
///
/// What IS fast: rebuilding the BPE model from pre-split vocab+merges
/// (~60ms) plus a MessagePack read of those plain vectors (~35ms) plus
/// parsing the remaining small JSON (pipeline + added tokens, ~1MB). The
/// sidecar therefore stores (a) `tokenizer.json` with `model.vocab` /
/// `model.merges` emptied and (b) the vocab/merges as plain vectors, and
/// reassembles the tokenizer via `with_model`. Total ≈ 90-120ms instead
/// of ≈ 270ms.
///
/// Safety: the sidecar is only written after an equivalence check —
/// canary prompts are encoded with the freshly parsed tokenizer and the
/// reassembled one, and the cache is persisted only if ids+tokens match
/// exactly. Cache is invalidated when the source file's (len, mtime)
/// changes. All cache failures fall back silently to the full parse.
mod tokenizer_cache {
    use std::path::Path;

    use tokenizers::Tokenizer;

    use crate::{Error, Result};

    /// Bump when the sidecar layout or reassembly logic changes.
    const SIDECAR_VERSION: u32 = 1;

    #[derive(serde::Serialize, serde::Deserialize)]
    struct Sidecar {
        version: u32,
        src_len: u64,
        src_mtime_ns: u128,
        /// Full tokenizer.json with `model.vocab`/`model.merges` emptied.
        rest_json: String,
        vocab: Vec<(String, u32)>,
        merges: Vec<(String, String)>,
        dropout: Option<f32>,
        unk_token: Option<String>,
        continuing_subword_prefix: Option<String>,
        end_of_word_suffix: Option<String>,
        fuse_unk: bool,
        byte_fallback: bool,
        ignore_merges: bool,
    }

    /// Canary prompts encoded by the equivalence check before a sidecar
    /// is persisted. Cover the real task prompts, specials, whitespace
    /// handling, byte-fallback and non-ASCII input.
    const CANARY_PROMPTS: &[&str] = &[
        "task: code retrieval | query: reverse linked list",
        "title: parser.rs | text: fn main() { println!(\"héllo wörld\") }",
        "<pad><bos>literal specials <eos> and   collapsed\twhitespace\nnewline",
        "task: code retrieval | query: 数据库连接池 emoji \u{1F680} \u{2581}underline",
        "",
    ];

    fn fingerprint(path: &Path) -> Option<(u64, u128)> {
        let meta = std::fs::metadata(path).ok()?;
        let mtime = meta
            .modified()
            .ok()?
            .duration_since(std::time::UNIX_EPOCH)
            .ok()?
            .as_nanos();
        Some((meta.len(), mtime))
    }

    /// FNV-1a over the canonical-ish source path; used only to name the
    /// sidecar file per source tokenizer, not for integrity.
    fn path_key(path: &Path) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in path.to_string_lossy().as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    fn sidecar_path(cache_dir: &Path, src: &Path) -> std::path::PathBuf {
        cache_dir.join(format!(
            "tokenizer-fast-v{SIDECAR_VERSION}-{:016x}.rmp",
            path_key(src)
        ))
    }

    /// Load `tokenizer.json`, preferring the verified sidecar. Never
    /// fails because of the cache: any sidecar problem falls back to the
    /// full parse, and sidecar writes are best-effort.
    pub(crate) fn load_tokenizer_fast(
        path: &Path,
        cache_dir: Option<&Path>,
    ) -> Result<Tokenizer> {
        let Some(cache_dir) = cache_dir else {
            let bytes = std::fs::read(path)?;
            return Tokenizer::from_bytes(&bytes).map_err(|e| Error::Tokenizer(e.to_string()));
        };
        let fp = fingerprint(path);
        let sidecar_file = sidecar_path(cache_dir, path);
        if let (Some(fp), Ok(bytes)) = (fp, std::fs::read(&sidecar_file)) {
            if let Ok(sidecar) = rmp_serde::from_slice::<Sidecar>(&bytes) {
                if sidecar.version == SIDECAR_VERSION
                    && (sidecar.src_len, sidecar.src_mtime_ns) == fp
                {
                    match reassemble(&sidecar) {
                        Ok(tok) => return Ok(tok),
                        Err(e) => debug_log(&format!("sidecar reassemble failed: {e}")),
                    }
                } else {
                    debug_log("sidecar stale (version/fingerprint mismatch)");
                }
            } else {
                debug_log("sidecar unreadable (rmp decode failed)");
            }
            // Stale/broken sidecar: fall through to the full parse which
            // rewrites it.
        }

        let bytes = std::fs::read(path)?;
        let tokenizer =
            Tokenizer::from_bytes(&bytes).map_err(|e| Error::Tokenizer(e.to_string()))?;
        if let Some(fp) = fp {
            // Best-effort: never fail the load because caching failed.
            if let Err(e) = write_sidecar(&sidecar_file, &bytes, &tokenizer, fp) {
                debug_log(&format!("sidecar write skipped: {e}"));
            }
        }
        Ok(tokenizer)
    }

    /// Cache diagnostics are silent by default (cache failures must never
    /// disturb CLI output); set `GREPPLUS_DEBUG_TOKENIZER_CACHE=1` to see
    /// why a sidecar was not used/written.
    fn debug_log(msg: &str) {
        if std::env::var_os("GREPPLUS_DEBUG_TOKENIZER_CACHE").is_some() {
            eprintln!("grepplus-embeddinggemma tokenizer-cache: {msg}");
        }
    }

    fn reassemble(sidecar: &Sidecar) -> Result<Tokenizer> {
        let mut tokenizer = Tokenizer::from_bytes(sidecar.rest_json.as_bytes())
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let mut builder = tokenizers::models::bpe::BPE::builder()
            .vocab_and_merges(
                sidecar
                    .vocab
                    .iter()
                    .cloned()
                    .collect::<tokenizers::models::bpe::Vocab>(),
                sidecar.merges.clone(),
            )
            .fuse_unk(sidecar.fuse_unk)
            .byte_fallback(sidecar.byte_fallback)
            .ignore_merges(sidecar.ignore_merges);
        if let Some(dropout) = sidecar.dropout {
            builder = builder.dropout(dropout);
        }
        if let Some(unk) = &sidecar.unk_token {
            builder = builder.unk_token(unk.clone());
        }
        if let Some(prefix) = &sidecar.continuing_subword_prefix {
            builder = builder.continuing_subword_prefix(prefix.clone());
        }
        if let Some(suffix) = &sidecar.end_of_word_suffix {
            builder = builder.end_of_word_suffix(suffix.clone());
        }
        let bpe = builder.build().map_err(|e| Error::Tokenizer(e.to_string()))?;
        tokenizer.with_model(bpe);
        Ok(tokenizer)
    }

    /// Sidecar plus canary prompts covering every added token, so the
    /// pre-persist equivalence check exercises the added-vocabulary ids
    /// (the subtle failure mode of model replacement).
    struct Extracted {
        sidecar: Sidecar,
        added_token_canaries: Vec<String>,
    }

    fn extract_sidecar(
        src_bytes: &[u8],
        (src_len, src_mtime_ns): (u64, u128),
    ) -> Result<Extracted> {
        let mut v: serde_json::Value = serde_json::from_slice(src_bytes)?;
        let model = v
            .get("model")
            .ok_or_else(|| Error::Tokenizer("tokenizer.json has no model".into()))?;
        if model.get("type").and_then(|t| t.as_str()) != Some("BPE") {
            return Err(Error::Tokenizer(
                "tokenizer sidecar supports only BPE models".into(),
            ));
        }
        let vocab: std::collections::HashMap<String, u32> =
            serde_json::from_value(model["vocab"].clone())?;

        // Added tokens must resolve to the SAME ids after the model swap.
        // `Tokenizer::from_bytes(rest_json)` rebuilds the AddedVocabulary
        // against the stub model: tokens missing from the stub vocab get
        // fresh ids computed from the stub's (tiny) vocab size, silently
        // remapping them. Pin every added token in the stub vocab with
        // its declared id — for in-vocab tokens that is the base-vocab id
        // (verified), for beyond-vocab tokens (EmbeddingGemma's
        // `<image_soft_token>` = 262144 = vocab_size) the declared id.
        // Added tokens are matched by the added-token matcher BEFORE the
        // model on encode, and their ids live in the AddedVocabulary map
        // fixed at deserialize time, so pinning via the stub is exact;
        // the equivalence canaries below cover every added token anyway.
        let mut stub_vocab = serde_json::Map::new();
        let mut added_contents: Vec<String> = Vec::new();
        if let Some(added) = v.get("added_tokens").and_then(|a| a.as_array()) {
            for tok in added {
                let (Some(content), Some(id)) = (
                    tok.get("content").and_then(|c| c.as_str()),
                    tok.get("id").and_then(serde_json::Value::as_u64),
                ) else {
                    return Err(Error::Tokenizer("malformed added_tokens entry".into()));
                };
                if let Some(base_id) = vocab.get(content) {
                    if u64::from(*base_id) != id {
                        return Err(Error::Tokenizer(format!(
                            "added token {content:?} id {id} conflicts with vocab id {base_id}; not caching"
                        )));
                    }
                }
                stub_vocab.insert(content.to_string(), serde_json::json!(id));
                added_contents.push(content.to_string());
            }
        }
        let added_token_canaries = added_contents
            .chunks(64)
            .map(|chunk| chunk.join(" "))
            .collect();

        let mut vocab: Vec<(String, u32)> = vocab.into_iter().collect();
        // Deterministic sidecar bytes (HashMap order is random).
        vocab.sort_unstable_by_key(|(_, id)| *id);
        let merges_value = model
            .get("merges")
            .and_then(|m| m.as_array())
            .ok_or_else(|| Error::Tokenizer("BPE model has no merges array".into()))?;
        let mut merges = Vec::with_capacity(merges_value.len());
        for pair in merges_value {
            // Modern layout: ["a", "b"]; legacy layout: "a b".
            let entry = if let Some(arr) = pair.as_array() {
                match (arr[0].as_str(), arr.get(1).and_then(|b| b.as_str())) {
                    (Some(a), Some(b)) => (a.to_string(), b.to_string()),
                    _ => return Err(Error::Tokenizer("malformed merge pair".into())),
                }
            } else if let Some(s) = pair.as_str() {
                match s.split_once(' ') {
                    Some((a, b)) => (a.to_string(), b.to_string()),
                    None => return Err(Error::Tokenizer("malformed legacy merge".into())),
                }
            } else {
                return Err(Error::Tokenizer("malformed merge entry".into()));
            };
            merges.push(entry);
        }
        let dropout = model.get("dropout").and_then(|d| d.as_f64()).map(|d| d as f32);
        let get_string = |key: &str| {
            model
                .get(key)
                .and_then(|s| s.as_str())
                .map(ToOwned::to_owned)
        };
        let get_bool =
            |key: &str| model.get(key).and_then(|b| b.as_bool()).unwrap_or(false);
        let unk_token = get_string("unk_token");
        let continuing_subword_prefix = get_string("continuing_subword_prefix");
        let end_of_word_suffix = get_string("end_of_word_suffix");
        let fuse_unk = get_bool("fuse_unk");
        let byte_fallback = get_bool("byte_fallback");
        let ignore_merges = get_bool("ignore_merges");

        // Shrink the big fields; the small remainder (pipeline + added
        // tokens + stub vocab pinning added-token ids) is what the fast
        // path re-parses as JSON.
        let m = v.get_mut("model").expect("model presence checked above");
        m["vocab"] = serde_json::Value::Object(stub_vocab);
        m["merges"] = serde_json::json!([]);
        let rest_json = serde_json::to_string(&v)?;

        Ok(Extracted {
            sidecar: Sidecar {
                version: SIDECAR_VERSION,
                src_len,
                src_mtime_ns,
                rest_json,
                vocab,
                merges,
                dropout,
                unk_token,
                continuing_subword_prefix,
                end_of_word_suffix,
                fuse_unk,
                byte_fallback,
                ignore_merges,
            },
            added_token_canaries,
        })
    }

    /// Encodings must match EXACTLY between the parsed tokenizer and the
    /// sidecar-reassembled one, otherwise the sidecar is not persisted.
    fn equivalent(a: &Tokenizer, b: &Tokenizer, extra_prompts: &[String]) -> bool {
        let fixed = CANARY_PROMPTS.iter().map(|p| (*p).to_string());
        for prompt in fixed.chain(extra_prompts.iter().cloned()) {
            let (Ok(ea), Ok(eb)) = (
                a.encode(prompt.as_str(), true),
                b.encode(prompt.as_str(), true),
            ) else {
                return false;
            };
            if ea.get_ids() != eb.get_ids() || ea.get_tokens() != eb.get_tokens() {
                return false;
            }
        }
        true
    }

    fn write_sidecar(
        sidecar_file: &Path,
        src_bytes: &[u8],
        parsed: &Tokenizer,
        fp: (u64, u128),
    ) -> Result<()> {
        let extracted = extract_sidecar(src_bytes, fp)?;
        let sidecar = extracted.sidecar;
        let rebuilt = reassemble(&sidecar)?;
        if !equivalent(parsed, &rebuilt, &extracted.added_token_canaries) {
            return Err(Error::Tokenizer(
                "sidecar reassembly not equivalent; cache not written".into(),
            ));
        }
        let bytes = rmp_serde::to_vec(&sidecar)
            .map_err(|e| Error::Tokenizer(format!("sidecar encode: {e}")))?;
        if let Some(parent) = sidecar_file.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = sidecar_file.with_extension(format!("tmp.{}", std::process::id()));
        std::fs::write(&tmp, &bytes)?;
        // Atomic publish so concurrent single-shot invocations never see
        // a torn cache file.
        std::fs::rename(&tmp, sidecar_file)?;
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Minimal BPE tokenizer.json exercising vocab/merges extraction,
        /// added tokens and the reassembly equivalence check.
        const TINY_TOKENIZER_JSON: &str = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {"id": 0, "content": "<unk>", "single_word": false, "lstrip": false,
                 "rstrip": false, "normalized": false, "special": true},
                {"id": 5, "content": "<extra>", "single_word": false, "lstrip": false,
                 "rstrip": false, "normalized": false, "special": true}
            ],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": "<unk>",
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "ignore_merges": false,
                "vocab": {"<unk>": 0, "a": 1, "b": 2, "ab": 3, "abb": 4, "<extra>": 5},
                "merges": [["a", "b"], ["ab", "b"]]
            }
        }"#;

        fn tmp_dir(tag: &str) -> std::path::PathBuf {
            let dir = std::env::temp_dir().join(format!(
                "grepplus-tokcache-{tag}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            dir
        }

        #[test]
        fn sidecar_roundtrip_is_equivalent_and_cached() {
            let dir = tmp_dir("roundtrip");
            let src = dir.join("tokenizer.json");
            std::fs::write(&src, TINY_TOKENIZER_JSON).unwrap();

            // First load writes the sidecar.
            let tok1 = load_tokenizer_fast(&src, Some(&dir)).unwrap();
            let sidecar_file = sidecar_path(&dir, &src);
            assert!(sidecar_file.exists(), "sidecar not written");

            // Second load takes the sidecar path; encodings must match.
            let tok2 = load_tokenizer_fast(&src, Some(&dir)).unwrap();
            for prompt in ["a b ab abb", "<extra> ab", "abb abb b", ""] {
                let e1 = tok1.encode(prompt, true).unwrap();
                let e2 = tok2.encode(prompt, true).unwrap();
                assert_eq!(e1.get_ids(), e2.get_ids(), "prompt {prompt:?}");
                assert_eq!(e1.get_tokens(), e2.get_tokens(), "prompt {prompt:?}");
            }
            std::fs::remove_dir_all(&dir).ok();
        }

        #[test]
        fn sidecar_invalidates_on_source_change() {
            let dir = tmp_dir("invalidate");
            let src = dir.join("tokenizer.json");
            std::fs::write(&src, TINY_TOKENIZER_JSON).unwrap();
            load_tokenizer_fast(&src, Some(&dir)).unwrap();
            let sidecar_file = sidecar_path(&dir, &src);
            let before = std::fs::read(&sidecar_file).unwrap();

            // Grow the file (len changes → fingerprint changes).
            std::fs::write(&src, format!("{TINY_TOKENIZER_JSON} ")).unwrap();
            load_tokenizer_fast(&src, Some(&dir)).unwrap();
            let after = std::fs::read(&sidecar_file).unwrap();
            let side: Sidecar = rmp_serde::from_slice(&after).unwrap();
            assert_eq!(side.src_len, TINY_TOKENIZER_JSON.len() as u64 + 1);
            assert!(before != after, "sidecar not rewritten after source change");
            std::fs::remove_dir_all(&dir).ok();
        }

        #[test]
        fn no_cache_dir_still_loads() {
            let dir = tmp_dir("nocache");
            let src = dir.join("tokenizer.json");
            std::fs::write(&src, TINY_TOKENIZER_JSON).unwrap();
            let tok = load_tokenizer_fast(&src, None).unwrap();
            assert!(tok.encode("a b", true).is_ok());
            assert!(!sidecar_path(&dir, &src).exists());
            std::fs::remove_dir_all(&dir).ok();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn task_prompts_match_embeddinggemma_contract() {
        assert_eq!(
            EmbedTask::CodeRetrievalQuery.prompt("reverse linked list"),
            "task: code retrieval | query: reverse linked list"
        );
        assert_eq!(
            EmbedTask::RetrievalDocument.prompt("fn main() {}"),
            "title: none | text: fn main() {}"
        );
        assert_eq!(
            EmbedTask::document_with_title(Some("parser.rs"), "extract calls"),
            "title: parser.rs | text: extract calls"
        );
    }

    #[test]
    #[ignore = "requires GREPPLUS_EMBEDDINGGEMMA_MODEL to point at a local model checkout"]
    fn local_model_smoke_test() {
        let model_dir = std::env::var("GREPPLUS_EMBEDDINGGEMMA_MODEL")
            .expect("set GREPPLUS_EMBEDDINGGEMMA_MODEL");
        let model = EmbeddingGemma::load_safetensors(
            PathBuf::from(model_dir),
            LoadOptions {
                max_length: Some(64),
                ..LoadOptions::default()
            },
        )
        .unwrap();
        let embedding = model
            .embed_one(EmbedTask::CodeRetrievalQuery, "FNV checksum fold loop")
            .unwrap();
        assert_eq!(embedding.len(), EMBEDDING_DIM);
        let norm = embedding
            .iter()
            .map(|v| f64::from(*v) * f64::from(*v))
            .sum::<f64>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "norm={norm}");
    }

    #[test]
    #[ignore = "requires GREPPLUS_EMBEDDINGGEMMA_GGUF and GREPPLUS_EMBEDDINGGEMMA_TOKENIZER"]
    fn local_gguf_smoke_test() {
        let gguf = std::env::var("GREPPLUS_EMBEDDINGGEMMA_GGUF")
            .expect("set GREPPLUS_EMBEDDINGGEMMA_GGUF");
        let tokenizer = std::env::var("GREPPLUS_EMBEDDINGGEMMA_TOKENIZER")
            .expect("set GREPPLUS_EMBEDDINGGEMMA_TOKENIZER");
        let model = EmbeddingGemma::load_gguf(
            PathBuf::from(gguf),
            PathBuf::from(tokenizer),
            LoadOptions {
                max_length: Some(64),
                ..LoadOptions::default()
            },
        )
        .unwrap();
        let embedding = model
            .embed_one(EmbedTask::CodeRetrievalQuery, "FNV checksum fold loop")
            .unwrap();
        assert_eq!(embedding.len(), EMBEDDING_DIM);
        let norm = embedding
            .iter()
            .map(|v| f64::from(*v) * f64::from(*v))
            .sum::<f64>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "norm={norm}");
    }

    /// The embedded loader (`embed-weights`) must produce the SAME embedding
    /// as the external GGUF loader fed the identical files. Loads both and
    /// asserts bit-identical vectors (same GGUF parse + tensor construction +
    /// tokenizer). Only runs when the feature is on and the external files
    /// are configured via env, so it's a no-op in the default build.
    #[cfg(feature = "embed-weights")]
    #[test]
    #[ignore = "requires embed-weights + GREPPLUS_EMBEDDINGGEMMA_GGUF/_TOKENIZER"]
    fn embedded_matches_external_gguf() {
        let gguf = std::env::var("GREPPLUS_EMBEDDINGGEMMA_GGUF")
            .expect("set GREPPLUS_EMBEDDINGGEMMA_GGUF");
        let tokenizer = std::env::var("GREPPLUS_EMBEDDINGGEMMA_TOKENIZER")
            .expect("set GREPPLUS_EMBEDDINGGEMMA_TOKENIZER");
        let opts = || LoadOptions {
            max_length: Some(64),
            ..LoadOptions::default()
        };
        let external =
            EmbeddingGemma::load_gguf(PathBuf::from(gguf), PathBuf::from(tokenizer), opts())
                .unwrap();
        let embedded = EmbeddingGemma::load_embedded(opts()).unwrap();

        let text = "reverse a singly linked list in place";
        let a = external.embed_one(EmbedTask::CodeRetrievalQuery, text).unwrap();
        let b = embedded.embed_one(EmbedTask::CodeRetrievalQuery, text).unwrap();
        assert_eq!(a.len(), EMBEDDING_DIM);
        assert_eq!(a.len(), b.len());

        let mut max_abs_diff = 0f32;
        let mut dot = 0f64;
        let (mut na, mut nb) = (0f64, 0f64);
        for (x, y) in a.iter().zip(b.iter()) {
            max_abs_diff = max_abs_diff.max((x - y).abs());
            dot += f64::from(*x) * f64::from(*y);
            na += f64::from(*x) * f64::from(*x);
            nb += f64::from(*y) * f64::from(*y);
        }
        let cosine = dot / (na.sqrt() * nb.sqrt());
        assert!(
            cosine > 0.999,
            "embedded vs external cosine={cosine} (max_abs_diff={max_abs_diff})"
        );
        // Same files, same math => expect bit-identical in practice.
        assert!(
            max_abs_diff < 1e-6,
            "expected near bit-identical; max_abs_diff={max_abs_diff}, cosine={cosine}"
        );
    }
}
