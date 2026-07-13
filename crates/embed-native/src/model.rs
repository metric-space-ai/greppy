//! Candle-free CPU forward pass for EmbeddingGemma-300M GGUF.

use rayon::prelude::*;

use gemm::{gemm, Parallelism};

use crate::gguf::{GgufModel, Value};
use crate::matmul::QuantMatrix;
use crate::performance::PerformanceCorePool;
use crate::{Error, Result, TokenizedBatch};

const GGUF_ARCHITECTURE: &str = "gemma-embedding";
const DEFAULT_GGUF_LOCAL_ROPE_FREQ: f64 = 10_000.0;
const DEFAULT_GGUF_SLIDING_PATTERN: usize = 6;
const EMBEDDING_DIM: usize = 768;
const DENSE2_DIM: usize = 3072;
const SQRT_TWO_OVER_PI: f32 = 0.7978845608028654;

#[derive(Debug, Clone)]
pub struct StageOutput {
    pub name: String,
    pub values: Vec<f32>,
}

pub struct CpuEmbeddingModel {
    embed_tokens: QuantMatrix,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    head: SentenceTransformerHead,
    cfg: GgufConfig,
    rotary: RotaryEmbeddings,
    performance_pool: PerformanceCorePool,
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
    rms_norm_eps: f32,
    rope_theta: f64,
    rope_local_base_freq: f64,
    sliding_window: usize,
    layer_types: Vec<LayerType>,
    query_pre_attn_scalar: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LayerType {
    SlidingAttention,
    FullAttention,
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    layer_type: LayerType,
}

struct Attention {
    q_proj: QuantMatrix,
    k_proj: QuantMatrix,
    v_proj: QuantMatrix,
    o_proj: QuantMatrix,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    scaling: f32,
    layer_type: LayerType,
}

struct Mlp {
    gate_proj: QuantMatrix,
    up_proj: QuantMatrix,
    down_proj: QuantMatrix,
}

struct RmsNorm {
    weight: Vec<f32>,
    eps: f32,
}

struct SentenceTransformerHead {
    dense2: QuantMatrix,
    dense3: QuantMatrix,
}

struct RotaryEmbeddings {
    full: RotaryEmbedding,
    sliding: RotaryEmbedding,
}

struct RotaryEmbedding {
    cos: Vec<f32>,
    sin: Vec<f32>,
    max_seq_len: usize,
    half_dim: usize,
}

struct LayerMasks {
    full: Option<Vec<f32>>,
    sliding: Option<Vec<f32>>,
}

impl CpuEmbeddingModel {
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let model = GgufModel::open(path)?;
        Self::from_gguf(&model)
    }

    pub fn from_gguf(model: &GgufModel) -> Result<Self> {
        let cfg = GgufConfig::from_model(model)?;
        let embed_tokens = QuantMatrix::from_model(model, "token_embd.weight")?;
        let rotary = RotaryEmbeddings::new(&cfg);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::from_model(model, &cfg, layer_idx)?);
        }
        let norm = RmsNorm::from_model(model, "output_norm.weight", cfg.rms_norm_eps)?;
        let head = SentenceTransformerHead::from_model(model)?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            head,
            cfg,
            rotary,
            performance_pool: PerformanceCorePool::new("embeddinggemma")?,
        })
    }

    pub fn forward_batch(&self, batch: &TokenizedBatch) -> Result<Vec<Vec<f32>>> {
        self.forward_tokens(&batch.token_ids, &batch.attention_mask)
    }

    pub fn forward_tokens(
        &self,
        token_ids: &[Vec<u32>],
        attention_mask: &[Vec<u32>],
    ) -> Result<Vec<Vec<f32>>> {
        self.performance_pool
            .install(|| self.forward_tokens_on_performance_cores(token_ids, attention_mask))
    }

    fn forward_tokens_on_performance_cores(
        &self,
        token_ids: &[Vec<u32>],
        attention_mask: &[Vec<u32>],
    ) -> Result<Vec<Vec<f32>>> {
        let (embeddings, _) = self.forward_inner(token_ids, attention_mask, false)?;
        Ok(embeddings)
    }

    pub fn forward_stages(
        &self,
        token_ids: &[u32],
        attention_mask: &[u32],
    ) -> Result<Vec<StageOutput>> {
        let token_ids = vec![token_ids.to_vec()];
        let attention_mask = vec![attention_mask.to_vec()];
        self.performance_pool.install(|| {
            let (_, stages) = self.forward_inner(&token_ids, &attention_mask, true)?;
            Ok(stages)
        })
    }

    fn forward_inner(
        &self,
        token_ids: &[Vec<u32>],
        attention_mask: &[Vec<u32>],
        capture_stages: bool,
    ) -> Result<(Vec<Vec<f32>>, Vec<StageOutput>)> {
        let (batch, seq_len) = validate_batch(token_ids, attention_mask)?;
        if seq_len > self.cfg.max_position_embeddings {
            return Err(Error::InvalidGguf(format!(
                "seq_len {seq_len} exceeds max_position_embeddings {}",
                self.cfg.max_position_embeddings
            )));
        }
        let flat_ids = token_ids.iter().flatten().copied().collect::<Vec<_>>();
        let flat_mask = attention_mask.iter().flatten().copied().collect::<Vec<_>>();
        let masks = LayerMasks::new(
            attention_mask,
            batch,
            seq_len,
            self.cfg.effective_sliding_window(),
        );

        let mut stages = Vec::new();
        let mut xs = self.embed_tokens.embedding_rows(&flat_ids)?;
        let embed_scale = (self.cfg.hidden_size as f32).sqrt();
        xs.par_iter_mut().for_each(|v| *v *= embed_scale);
        push_stage(&mut stages, capture_stages, "embed_scaled", &xs);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            if capture_stages && debug_layer_index() == Some(layer_idx) {
                let (next, debug_stages) = layer.forward_debug(
                    layer_idx,
                    &xs,
                    batch,
                    seq_len,
                    &self.rotary,
                    masks.for_layer(layer.layer_type),
                )?;
                stages.extend(debug_stages);
                xs = next;
                continue;
            }
            xs = layer.forward(
                &xs,
                batch,
                seq_len,
                &self.rotary,
                masks.for_layer(layer.layer_type),
            )?;
            if capture_stages {
                stages.push(StageOutput {
                    name: format!("layer_{layer_idx}"),
                    values: xs.clone(),
                });
            }
        }

        let hidden = self.norm.forward(&xs, self.cfg.hidden_size)?;
        push_stage(&mut stages, capture_stages, "output_norm", &hidden);

        let pooled = mean_pool(&hidden, &flat_mask, batch, seq_len, self.cfg.hidden_size);
        push_stage(&mut stages, capture_stages, "mean_pool", &pooled);

        let dense2 = self.head.dense2.matmul(&pooled, batch)?;
        push_stage(&mut stages, capture_stages, "dense2", &dense2);

        let dense3 = self.head.dense3.matmul(&dense2, batch)?;
        push_stage(&mut stages, capture_stages, "dense3", &dense3);

        let l2 = normalize_l2(&dense3, EMBEDDING_DIM);
        ensure_finite("CPU final embedding", &l2)?;
        push_stage(&mut stages, capture_stages, "l2norm", &l2);

        Ok((split_rows(l2, EMBEDDING_DIM), stages))
    }
}

impl GgufConfig {
    fn from_model(model: &GgufModel) -> Result<Self> {
        let arch = model.metadata_str("general.architecture")?;
        if arch != GGUF_ARCHITECTURE {
            return Err(Error::InvalidGguf(format!(
                "expected GGUF architecture {GGUF_ARCHITECTURE}, got {arch}"
            )));
        }

        let hidden_size = model.metadata_u32("gemma-embedding.embedding_length")? as usize;
        if hidden_size != EMBEDDING_DIM {
            return Err(Error::InvalidGguf(format!(
                "expected embedding_length {EMBEDDING_DIM}, got {hidden_size}"
            )));
        }

        let num_hidden_layers = model.metadata_u32("gemma-embedding.block_count")? as usize;
        let sliding_pattern =
            metadata_u32_opt(model, "gemma-embedding.attention.sliding_window_type")
                .map(|v| v as usize)
                .unwrap_or(DEFAULT_GGUF_SLIDING_PATTERN);
        if sliding_pattern == 0 {
            return Err(Error::InvalidGguf(
                "attention.sliding_window_type must be non-zero".into(),
            ));
        }
        let layer_types = (0..num_hidden_layers)
            .map(|idx| {
                if (idx + 1) % sliding_pattern == 0 {
                    LayerType::FullAttention
                } else {
                    LayerType::SlidingAttention
                }
            })
            .collect::<Vec<_>>();

        let head_dim = model.metadata_u32("gemma-embedding.attention.key_length")? as usize;
        let num_attention_heads =
            model.metadata_u32("gemma-embedding.attention.head_count")? as usize;
        let num_key_value_heads =
            model.metadata_u32("gemma-embedding.attention.head_count_kv")? as usize;
        if num_hidden_layers == 0
            || head_dim == 0
            || head_dim % 2 != 0
            || num_attention_heads == 0
            || num_key_value_heads == 0
        {
            return Err(Error::InvalidGguf(format!(
                "invalid Gemma dimensions: layers={num_hidden_layers}, heads={num_attention_heads}, kv_heads={num_key_value_heads}, head_dim={head_dim}"
            )));
        }
        if num_attention_heads % num_key_value_heads != 0 {
            return Err(Error::InvalidGguf(format!(
                "attention heads {num_attention_heads} not divisible by kv heads {num_key_value_heads}"
            )));
        }
        let attention_width = num_attention_heads
            .checked_mul(head_dim)
            .ok_or_else(|| Error::InvalidGguf("head_count * head_dim overflows".into()))?;
        if attention_width != hidden_size {
            return Err(Error::InvalidGguf(format!(
                "head_count * head_dim = {attention_width} does not match hidden_size {hidden_size}",
            )));
        }

        let intermediate_size = model.metadata_u32("gemma-embedding.feed_forward_length")? as usize;
        let max_position_embeddings =
            model.metadata_u32("gemma-embedding.context_length")? as usize;
        let rms_norm_eps =
            model.metadata_f32("gemma-embedding.attention.layer_norm_rms_epsilon")?;
        let rope_theta = f64::from(model.metadata_f32("gemma-embedding.rope.freq_base")?);
        let rope_local_base_freq = metadata_f32_opt(model, "gemma-embedding.rope.local_freq_base")
            .map(f64::from)
            .unwrap_or(DEFAULT_GGUF_LOCAL_ROPE_FREQ);
        let sliding_window =
            model.metadata_u32("gemma-embedding.attention.sliding_window")? as usize;
        if intermediate_size == 0 || max_position_embeddings == 0 || sliding_window == 0 {
            return Err(Error::InvalidGguf(format!(
                "invalid Gemma sizes: intermediate={intermediate_size}, context={max_position_embeddings}, sliding_window={sliding_window}"
            )));
        }
        if !rms_norm_eps.is_finite() || rms_norm_eps <= 0.0 {
            return Err(Error::InvalidGguf(format!(
                "invalid RMS norm epsilon {rms_norm_eps}"
            )));
        }
        if !rope_theta.is_finite() || rope_theta <= 0.0 {
            return Err(Error::InvalidGguf(format!(
                "invalid RoPE theta {rope_theta}"
            )));
        }
        if !rope_local_base_freq.is_finite() || rope_local_base_freq <= 0.0 {
            return Err(Error::InvalidGguf(format!(
                "invalid local RoPE theta {rope_local_base_freq}"
            )));
        }

        Ok(Self {
            hidden_size,
            intermediate_size,
            max_position_embeddings,
            num_attention_heads,
            num_hidden_layers,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            rope_theta,
            rope_local_base_freq,
            sliding_window,
            layer_types,
            query_pre_attn_scalar: 256.0,
        })
    }

    fn effective_sliding_window(&self) -> usize {
        if self.sliding_window > 256 {
            (self.sliding_window / 2) + 1
        } else {
            self.sliding_window
        }
    }
}

impl DecoderLayer {
    fn from_model(model: &GgufModel, cfg: &GgufConfig, layer_idx: usize) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");
        let layer_type = cfg.layer_types[layer_idx];
        Ok(Self {
            self_attn: Attention::from_model(model, cfg, layer_type, &prefix)?,
            mlp: Mlp::from_model(model, cfg, &prefix)?,
            input_layernorm: RmsNorm::from_model(
                model,
                &format!("{prefix}.attn_norm.weight"),
                cfg.rms_norm_eps,
            )?,
            post_attention_layernorm: RmsNorm::from_model(
                model,
                &format!("{prefix}.post_attention_norm.weight"),
                cfg.rms_norm_eps,
            )?,
            pre_feedforward_layernorm: RmsNorm::from_model(
                model,
                &format!("{prefix}.ffn_norm.weight"),
                cfg.rms_norm_eps,
            )?,
            post_feedforward_layernorm: RmsNorm::from_model(
                model,
                &format!("{prefix}.post_ffw_norm.weight"),
                cfg.rms_norm_eps,
            )?,
            layer_type,
        })
    }

    fn forward(
        &self,
        xs: &[f32],
        batch: usize,
        seq_len: usize,
        rotary: &RotaryEmbeddings,
        attention_mask: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        let residual = xs;
        let xs_norm = self.input_layernorm.forward(xs, EMBEDDING_DIM)?;
        let attn = self
            .self_attn
            .forward(&xs_norm, batch, seq_len, rotary, attention_mask)?;
        let attn = self
            .post_attention_layernorm
            .forward(&attn, EMBEDDING_DIM)?;
        let mut xs = add(residual, &attn);

        let residual = xs.clone();
        let ffn = self.pre_feedforward_layernorm.forward(&xs, EMBEDDING_DIM)?;
        let ffn = self.mlp.forward(&ffn, batch * seq_len)?;
        let ffn = self
            .post_feedforward_layernorm
            .forward(&ffn, EMBEDDING_DIM)?;
        xs.par_iter_mut()
            .zip(ffn.par_iter())
            .for_each(|(dst, src)| *dst += *src);
        drop(residual);
        Ok(xs)
    }

    fn forward_debug(
        &self,
        layer_idx: usize,
        xs: &[f32],
        batch: usize,
        seq_len: usize,
        rotary: &RotaryEmbeddings,
        attention_mask: Option<&[f32]>,
    ) -> Result<(Vec<f32>, Vec<StageOutput>)> {
        let mut stages = Vec::new();
        let residual = xs;
        let xs_norm = self.input_layernorm.forward(xs, EMBEDDING_DIM)?;
        stages.push(StageOutput {
            name: format!("layer_{layer_idx}_input_norm"),
            values: xs_norm.clone(),
        });
        let attn = self
            .self_attn
            .forward(&xs_norm, batch, seq_len, rotary, attention_mask)?;
        stages.push(StageOutput {
            name: format!("layer_{layer_idx}_attn"),
            values: attn.clone(),
        });
        let attn = self
            .post_attention_layernorm
            .forward(&attn, EMBEDDING_DIM)?;
        stages.push(StageOutput {
            name: format!("layer_{layer_idx}_post_attn_norm"),
            values: attn.clone(),
        });
        let xs = add(residual, &attn);
        stages.push(StageOutput {
            name: format!("layer_{layer_idx}_after_attn"),
            values: xs.clone(),
        });
        let ffn = self.pre_feedforward_layernorm.forward(&xs, EMBEDDING_DIM)?;
        stages.push(StageOutput {
            name: format!("layer_{layer_idx}_pre_ffn_norm"),
            values: ffn.clone(),
        });
        let (ffn, mlp_stages) = self.mlp.forward_debug(layer_idx, &ffn, batch * seq_len)?;
        stages.extend(mlp_stages);
        stages.push(StageOutput {
            name: format!("layer_{layer_idx}_mlp"),
            values: ffn.clone(),
        });
        let ffn = self
            .post_feedforward_layernorm
            .forward(&ffn, EMBEDDING_DIM)?;
        stages.push(StageOutput {
            name: format!("layer_{layer_idx}_post_ffn_norm"),
            values: ffn.clone(),
        });
        let mut xs = xs;
        xs.par_iter_mut()
            .zip(ffn.par_iter())
            .for_each(|(dst, src)| *dst += *src);
        stages.push(StageOutput {
            name: format!("layer_{layer_idx}"),
            values: xs.clone(),
        });
        Ok((xs, stages))
    }
}

impl Attention {
    fn from_model(
        model: &GgufModel,
        cfg: &GgufConfig,
        layer_type: LayerType,
        prefix: &str,
    ) -> Result<Self> {
        Ok(Self {
            q_proj: QuantMatrix::from_model(model, &format!("{prefix}.attn_q.weight"))?,
            k_proj: QuantMatrix::from_model(model, &format!("{prefix}.attn_k.weight"))?,
            v_proj: QuantMatrix::from_model(model, &format!("{prefix}.attn_v.weight"))?,
            o_proj: QuantMatrix::from_model(model, &format!("{prefix}.attn_output.weight"))?,
            q_norm: RmsNorm::from_model(
                model,
                &format!("{prefix}.attn_q_norm.weight"),
                cfg.rms_norm_eps,
            )?,
            k_norm: RmsNorm::from_model(
                model,
                &format!("{prefix}.attn_k_norm.weight"),
                cfg.rms_norm_eps,
            )?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            num_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            scaling: cfg.query_pre_attn_scalar.powf(-0.5),
            layer_type,
        })
    }

    fn forward(
        &self,
        xs: &[f32],
        batch: usize,
        seq_len: usize,
        rotary: &RotaryEmbeddings,
        attention_mask: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        let rows = batch * seq_len;
        let input = self.q_proj.prepare_q8k_rows(xs, rows)?;
        let (query_states, (key_states, value_states)) = rayon::join(
            || self.q_proj.matmul_prepared_q8k_rows(&input),
            || {
                rayon::join(
                    || self.k_proj.matmul_prepared_q8k_rows_or_f32(&input, xs),
                    || self.v_proj.matmul_prepared_q8k_rows_or_f32(&input, xs),
                )
            },
        );
        let query_states = query_states?;
        let key_states = key_states?;
        let value_states = value_states?;

        let mut query_states =
            reshape_heads(&query_states, batch, seq_len, self.num_heads, self.head_dim);
        let mut key_states = reshape_heads(
            &key_states,
            batch,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        );
        let value_states = reshape_heads(
            &value_states,
            batch,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        );

        query_states = self.q_norm.forward(&query_states, self.head_dim)?;
        key_states = self.k_norm.forward(&key_states, self.head_dim)?;
        rotary.for_layer(self.layer_type).apply(
            &mut query_states,
            batch,
            self.num_heads,
            seq_len,
            self.head_dim,
        )?;
        rotary.for_layer(self.layer_type).apply(
            &mut key_states,
            batch,
            self.num_kv_heads,
            seq_len,
            self.head_dim,
        )?;

        let attn_output = attention(
            &query_states,
            &key_states,
            &value_states,
            AttentionShape {
                batch,
                heads: self.num_heads,
                kv_heads: self.num_kv_heads,
                kv_groups: self.num_kv_groups,
                seq_len,
                head_dim: self.head_dim,
            },
            self.scaling,
            attention_mask,
        );
        let merged = merge_heads(&attn_output, batch, seq_len, self.num_heads, self.head_dim);
        self.o_proj.matmul(&merged, rows)
    }
}

impl Mlp {
    fn from_model(model: &GgufModel, cfg: &GgufConfig, prefix: &str) -> Result<Self> {
        let gate_proj = QuantMatrix::from_model(model, &format!("{prefix}.ffn_gate.weight"))?;
        let up_proj = QuantMatrix::from_model(model, &format!("{prefix}.ffn_up.weight"))?;
        let down_proj = QuantMatrix::from_model(model, &format!("{prefix}.ffn_down.weight"))?;
        if gate_proj.rows() != cfg.intermediate_size
            || up_proj.rows() != cfg.intermediate_size
            || down_proj.cols() != cfg.intermediate_size
        {
            return Err(Error::InvalidGguf(format!(
                "{prefix} MLP tensor shape mismatch"
            )));
        }
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, xs: &[f32], rows: usize) -> Result<Vec<f32>> {
        let (mut gate, up) = self.project_gate_up(xs, rows)?;
        gate.par_iter_mut()
            .zip(up.par_iter())
            .for_each(|(g, u)| *g = gelu_tanh(*g) * *u);
        self.down_proj.matmul(&gate, rows)
    }

    fn project_gate_up(&self, xs: &[f32], rows: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let input = self.gate_proj.prepare_q8k_rows(xs, rows)?;
        let (gate, up) = rayon::join(
            || self.gate_proj.matmul_prepared_q8k_rows(&input),
            || self.up_proj.matmul_prepared_q8k_rows_or_f32(&input, xs),
        );
        Ok((gate?, up?))
    }

    fn forward_debug(
        &self,
        layer_idx: usize,
        xs: &[f32],
        rows: usize,
    ) -> Result<(Vec<f32>, Vec<StageOutput>)> {
        let (gate, up) = self.project_gate_up(xs, rows)?;
        let mut stages = vec![StageOutput {
            name: format!("layer_{layer_idx}_mlp_gate"),
            values: gate.clone(),
        }];
        let mut gate_act = gate;
        gate_act.par_iter_mut().for_each(|v| *v = gelu_tanh(*v));
        stages.push(StageOutput {
            name: format!("layer_{layer_idx}_mlp_gate_gelu"),
            values: gate_act.clone(),
        });
        stages.push(StageOutput {
            name: format!("layer_{layer_idx}_mlp_up"),
            values: up.clone(),
        });
        gate_act
            .par_iter_mut()
            .zip(up.par_iter())
            .for_each(|(g, u)| *g *= *u);
        stages.push(StageOutput {
            name: format!("layer_{layer_idx}_mlp_product"),
            values: gate_act.clone(),
        });
        let down = self.down_proj.matmul(&gate_act, rows)?;
        Ok((down, stages))
    }
}

impl RmsNorm {
    fn from_model(model: &GgufModel, name: &str, eps: f32) -> Result<Self> {
        let (_, weight) = model.tensor_f32(name)?;
        ensure_finite(name, &weight)?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, xs: &[f32], dim: usize) -> Result<Vec<f32>> {
        if self.weight.len() != dim {
            return Err(Error::InvalidGguf(format!(
                "RMSNorm weight len {}, expected {dim}",
                self.weight.len()
            )));
        }
        if xs.len() % dim != 0 {
            return Err(Error::InvalidGguf(format!(
                "RMSNorm input len {} not divisible by {dim}",
                xs.len()
            )));
        }
        let mut out = vec![0.0f32; xs.len()];
        let rows = xs.len() / dim;
        if rows <= 32 {
            for (dst, src) in out.chunks_mut(dim).zip(xs.chunks(dim)) {
                self.forward_row(src, dst, dim);
            }
        } else {
            out.par_chunks_mut(dim)
                .zip(xs.par_chunks(dim))
                .for_each(|(dst, src)| self.forward_row(src, dst, dim));
        }
        Ok(out)
    }

    fn forward_row(&self, src: &[f32], dst: &mut [f32], dim: usize) {
        let denom = if std::env::var_os("EMBED_NATIVE_RMS_F64").is_some() {
            (src.iter()
                .map(|&v| f64::from(v) * f64::from(v))
                .sum::<f64>()
                / dim as f64
                + f64::from(self.eps))
            .sqrt() as f32
        } else {
            let sum2 = src.iter().map(|v| v * v).sum::<f32>();
            (sum2 / dim as f32 + self.eps).sqrt()
        };
        for ((dst, &src), &weight) in dst.iter_mut().zip(src).zip(&self.weight) {
            *dst = src / denom * weight;
        }
    }
}

impl SentenceTransformerHead {
    fn from_model(model: &GgufModel) -> Result<Self> {
        let dense2 = QuantMatrix::from_model(model, "dense_2.weight")?;
        let dense3 = QuantMatrix::from_model(model, "dense_3.weight")?;
        if dense2.rows() != DENSE2_DIM || dense2.cols() != EMBEDDING_DIM {
            return Err(Error::InvalidGguf("dense_2.weight shape mismatch".into()));
        }
        if dense3.rows() != EMBEDDING_DIM || dense3.cols() != DENSE2_DIM {
            return Err(Error::InvalidGguf("dense_3.weight shape mismatch".into()));
        }
        Ok(Self { dense2, dense3 })
    }
}

impl RotaryEmbeddings {
    fn new(cfg: &GgufConfig) -> Self {
        Self {
            full: RotaryEmbedding::new(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta),
            sliding: RotaryEmbedding::new(
                cfg.head_dim,
                cfg.max_position_embeddings,
                cfg.rope_local_base_freq,
            ),
        }
    }

    fn for_layer(&self, layer_type: LayerType) -> &RotaryEmbedding {
        match layer_type {
            LayerType::FullAttention => &self.full,
            LayerType::SlidingAttention => &self.sliding,
        }
    }
}

impl RotaryEmbedding {
    fn new(head_dim: usize, max_seq_len: usize, base: f64) -> Self {
        let half_dim = head_dim / 2;
        let inv_freq = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0f32 / base.powf(i as f64 / head_dim as f64) as f32)
            .collect::<Vec<_>>();
        let mut cos = vec![0.0f32; max_seq_len * half_dim];
        let mut sin = vec![0.0f32; max_seq_len * half_dim];
        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let freq = pos as f32 * inv_freq[i];
                cos[pos * half_dim + i] = freq.cos();
                sin[pos * half_dim + i] = freq.sin();
            }
        }
        Self {
            cos,
            sin,
            max_seq_len,
            half_dim,
        }
    }

    fn apply(
        &self,
        xs: &mut [f32],
        batch: usize,
        heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Result<()> {
        if seq_len > self.max_seq_len {
            return Err(Error::InvalidGguf(format!(
                "RoPE seq_len {seq_len} exceeds max {}",
                self.max_seq_len
            )));
        }
        if head_dim / 2 != self.half_dim {
            return Err(Error::InvalidGguf(format!(
                "RoPE head_dim {head_dim} incompatible with cached half_dim {}",
                self.half_dim
            )));
        }
        let expected = batch
            .checked_mul(heads)
            .and_then(|v| v.checked_mul(seq_len))
            .and_then(|v| v.checked_mul(head_dim))
            .ok_or_else(|| Error::InvalidGguf("RoPE input shape overflows".into()))?;
        if xs.len() != expected {
            return Err(Error::InvalidGguf(format!(
                "RoPE input len {}, expected {expected}",
                xs.len()
            )));
        }
        for b in 0..batch {
            for h in 0..heads {
                for pos in 0..seq_len {
                    let base = ((b * heads + h) * seq_len + pos) * head_dim;
                    let cs_base = pos * self.half_dim;
                    for i in 0..self.half_dim {
                        let x1 = xs[base + i];
                        let x2 = xs[base + self.half_dim + i];
                        let cos = self.cos[cs_base + i];
                        let sin = self.sin[cs_base + i];
                        xs[base + i] = x1 * cos - x2 * sin;
                        xs[base + self.half_dim + i] = x2 * cos + x1 * sin;
                    }
                }
            }
        }
        Ok(())
    }
}

impl LayerMasks {
    fn new(
        attention_mask: &[Vec<u32>],
        batch: usize,
        seq_len: usize,
        sliding_window: usize,
    ) -> Self {
        let all_unmasked = attention_mask.iter().all(|row| row.iter().all(|&m| m != 0));
        let full = if all_unmasked {
            None
        } else {
            Some(build_attention_mask(attention_mask, batch, seq_len, None))
        };
        let sliding_can_be_full = seq_len < sliding_window;
        let sliding = if all_unmasked && sliding_can_be_full {
            None
        } else {
            Some(build_attention_mask(
                attention_mask,
                batch,
                seq_len,
                Some(sliding_window),
            ))
        };
        Self { full, sliding }
    }

    fn for_layer(&self, layer_type: LayerType) -> Option<&[f32]> {
        match layer_type {
            LayerType::FullAttention => self.full.as_deref(),
            LayerType::SlidingAttention => self.sliding.as_deref(),
        }
    }
}

#[derive(Clone, Copy)]
struct AttentionShape {
    batch: usize,
    heads: usize,
    kv_heads: usize,
    kv_groups: usize,
    seq_len: usize,
    head_dim: usize,
}

fn attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    shape: AttentionShape,
    scaling: f32,
    mask: Option<&[f32]>,
) -> Vec<f32> {
    let mut scores = vec![0.0f32; shape.batch * shape.heads * shape.seq_len * shape.seq_len];
    for (bh_idx, dst) in scores.chunks_mut(shape.seq_len * shape.seq_len).enumerate() {
        let h = bh_idx % shape.heads;
        let b = bh_idx / shape.heads;
        let kv_h = h / shape.kv_groups;

        let q_base = ((b * shape.heads + h) * shape.seq_len) * shape.head_dim;
        let k_base = ((b * shape.kv_heads + kv_h) * shape.seq_len) * shape.head_dim;
        gemm_f32(
            shape.seq_len,
            shape.seq_len,
            shape.head_dim,
            dst,
            shape.seq_len as isize,
            1,
            &query[q_base..],
            shape.head_dim as isize,
            1,
            &key[k_base..],
            1,
            shape.head_dim as isize,
        );
        for q_pos in 0..shape.seq_len {
            for k_pos in 0..shape.seq_len {
                let idx = q_pos * shape.seq_len + k_pos;
                let mask_value = mask
                    .map(|m| m[(b * shape.seq_len + q_pos) * shape.seq_len + k_pos])
                    .unwrap_or(0.0);
                dst[idx] = dst[idx] * scaling + mask_value;
            }
        }
    }

    scores.par_chunks_mut(shape.seq_len).for_each(|row| {
        let max_score = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut denom = 0.0f32;
        for score in row.iter_mut() {
            *score = (*score - max_score).exp();
            denom += *score;
        }
        for score in row {
            *score /= denom;
        }
    });

    let mut out = vec![0.0f32; shape.batch * shape.heads * shape.seq_len * shape.head_dim];
    for (bh_idx, dst) in out.chunks_mut(shape.seq_len * shape.head_dim).enumerate() {
        let h = bh_idx % shape.heads;
        let b = bh_idx / shape.heads;
        let kv_h = h / shape.kv_groups;
        let score_base = bh_idx * shape.seq_len * shape.seq_len;
        let v_base = ((b * shape.kv_heads + kv_h) * shape.seq_len) * shape.head_dim;
        gemm_f32(
            shape.seq_len,
            shape.head_dim,
            shape.seq_len,
            dst,
            shape.head_dim as isize,
            1,
            &scores[score_base..],
            shape.seq_len as isize,
            1,
            &value[v_base..],
            shape.head_dim as isize,
            1,
        );
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn gemm_f32(
    m: usize,
    n: usize,
    k: usize,
    dst: &mut [f32],
    dst_rs: isize,
    dst_cs: isize,
    lhs: &[f32],
    lhs_rs: isize,
    lhs_cs: isize,
    rhs: &[f32],
    rhs_rs: isize,
    rhs_cs: isize,
) {
    let parallelism = match rayon::current_num_threads() {
        0 | 1 => Parallelism::None,
        n => Parallelism::Rayon(n),
    };
    unsafe {
        gemm(
            m,
            n,
            k,
            dst.as_mut_ptr(),
            dst_cs,
            dst_rs,
            false,
            lhs.as_ptr(),
            lhs_cs,
            lhs_rs,
            rhs.as_ptr(),
            rhs_cs,
            rhs_rs,
            0.0f32,
            1.0f32,
            false,
            false,
            false,
            parallelism,
        )
    }
}

fn reshape_heads(
    src: &[f32],
    batch: usize,
    seq_len: usize,
    heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * heads * seq_len * head_dim];
    for b in 0..batch {
        for s in 0..seq_len {
            let src_base = (b * seq_len + s) * heads * head_dim;
            for h in 0..heads {
                let src_head = src_base + h * head_dim;
                let dst_head = ((b * heads + h) * seq_len + s) * head_dim;
                out[dst_head..dst_head + head_dim]
                    .copy_from_slice(&src[src_head..src_head + head_dim]);
            }
        }
    }
    out
}

fn merge_heads(
    src: &[f32],
    batch: usize,
    seq_len: usize,
    heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * seq_len * heads * head_dim];
    for b in 0..batch {
        for s in 0..seq_len {
            let dst_base = (b * seq_len + s) * heads * head_dim;
            for h in 0..heads {
                let src_head = ((b * heads + h) * seq_len + s) * head_dim;
                let dst_head = dst_base + h * head_dim;
                out[dst_head..dst_head + head_dim]
                    .copy_from_slice(&src[src_head..src_head + head_dim]);
            }
        }
    }
    out
}

fn build_attention_mask(
    mask_rows: &[Vec<u32>],
    batch: usize,
    seq_len: usize,
    sliding_window: Option<usize>,
) -> Vec<f32> {
    let mut values = Vec::with_capacity(batch * seq_len * seq_len);
    for row in mask_rows {
        for q in 0..seq_len {
            for k in 0..seq_len {
                let key_visible = row[k] != 0;
                let window_visible = match sliding_window {
                    None => true,
                    Some(window) => q.abs_diff(k) < window,
                };
                values.push(if key_visible && window_visible {
                    0.0
                } else {
                    -1.0e9
                });
            }
        }
    }
    values
}

fn mean_pool(
    hidden: &[f32],
    mask: &[u32],
    batch: usize,
    seq_len: usize,
    hidden_size: usize,
) -> Vec<f32> {
    let mut pooled = vec![0.0f32; batch * hidden_size];
    pooled
        .par_chunks_mut(hidden_size)
        .enumerate()
        .for_each(|(b, dst)| {
            let mut count = 0.0f32;
            for s in 0..seq_len {
                let m = mask[b * seq_len + s] as f32;
                count += m;
                let src =
                    &hidden[(b * seq_len + s) * hidden_size..(b * seq_len + s + 1) * hidden_size];
                for d in 0..hidden_size {
                    dst[d] += src[d] * m;
                }
            }
            let count = count.clamp(1.0e-12, f32::MAX);
            for v in dst {
                *v /= count;
            }
        });
    pooled
}

fn normalize_l2(xs: &[f32], dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; xs.len()];
    out.par_chunks_mut(dim)
        .zip(xs.par_chunks(dim))
        .for_each(|(dst, src)| {
            let denom = src
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt()
                .clamp(1.0e-12, f32::MAX);
            for (dst, &src) in dst.iter_mut().zip(src) {
                *dst = src / denom;
            }
        });
    out
}

fn add(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
    lhs.par_iter()
        .zip(rhs.par_iter())
        .map(|(a, b)| a + b)
        .collect()
}

fn ensure_finite(context: &str, values: &[f32]) -> Result<()> {
    if let Some((idx, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(Error::InvalidGguf(format!(
            "{context} produced non-finite value {value} at element {idx}"
        )));
    }
    Ok(())
}

fn gelu_tanh(v: f32) -> f32 {
    0.5 * v * (1.0 + (SQRT_TWO_OVER_PI * v * (1.0 + 0.044715 * v * v)).tanh())
}

fn split_rows(values: Vec<f32>, width: usize) -> Vec<Vec<f32>> {
    values.chunks_exact(width).map(|row| row.to_vec()).collect()
}

fn push_stage(stages: &mut Vec<StageOutput>, capture: bool, name: &str, values: &[f32]) {
    if capture {
        stages.push(StageOutput {
            name: name.to_string(),
            values: values.to_vec(),
        });
    }
}

fn validate_batch(token_ids: &[Vec<u32>], attention_mask: &[Vec<u32>]) -> Result<(usize, usize)> {
    if token_ids.len() != attention_mask.len() {
        return Err(Error::InvalidGguf(format!(
            "token batch len {} != attention mask len {}",
            token_ids.len(),
            attention_mask.len()
        )));
    }
    if token_ids.is_empty() {
        return Err(Error::InvalidGguf("empty token batch".into()));
    }
    let seq_len = token_ids[0].len();
    if seq_len == 0 {
        return Err(Error::InvalidGguf("empty sequence".into()));
    }
    token_ids
        .len()
        .checked_mul(seq_len)
        .ok_or_else(|| Error::InvalidGguf("token batch row count overflows".into()))?;
    for (idx, (ids, mask)) in token_ids.iter().zip(attention_mask).enumerate() {
        if ids.len() != seq_len || mask.len() != seq_len {
            return Err(Error::InvalidGguf(format!(
                "ragged batch row {idx}: ids {}, mask {}, expected {seq_len}",
                ids.len(),
                mask.len()
            )));
        }
    }
    Ok((token_ids.len(), seq_len))
}

fn debug_layer_index() -> Option<usize> {
    std::env::var("EMBED_NATIVE_DEBUG_LAYER")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .or_else(|| std::env::var_os("EMBED_NATIVE_DEBUG_LAYER0").map(|_| 2))
}

fn metadata_u32_opt(model: &GgufModel, key: &str) -> Option<u32> {
    model.metadata().get(key).and_then(Value::as_u32)
}

fn metadata_f32_opt(model: &GgufModel, key: &str) -> Option<f32> {
    model.metadata().get(key).and_then(Value::as_f32)
}
