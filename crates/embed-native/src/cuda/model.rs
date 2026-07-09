use std::path::Path;
use std::time::Instant;

use crate::cuda::ffi::{
    check, gp_attention_scores, gp_attention_values, gp_embed_q4k, gp_embed_q6k, gp_geglu,
    gp_l2_norm, gp_mean_pool, gp_merge_heads, gp_mmq_matmul, gp_rms_norm, gp_rms_norm_add,
    gp_rms_norm_heads, gp_rope_neox, gp_softmax_mask, gp_split_heads, CudaDevice, DeviceBuffer,
};
use crate::cuda::weights::CudaWeights;
use crate::gguf::{GgufModel, Value};
use crate::quant::GgmlDType;
use crate::{Error, Result, TokenizedBatch};

const GGUF_ARCHITECTURE: &str = "gemma-embedding";
const DEFAULT_GGUF_LOCAL_ROPE_FREQ: f32 = 10_000.0;
const DEFAULT_GGUF_SLIDING_PATTERN: usize = 6;
const EMBEDDING_DIM: usize = 768;
const DENSE2_DIM: usize = 3072;
const MMQ_MATRIX_ROW_PADDING: usize = 512;

pub struct CudaEmbeddingModel {
    dev: CudaDevice,
    weights: CudaWeights,
    cfg: CudaConfig,
}

#[derive(Debug, Clone, Default)]
pub struct CudaForwardProfile {
    pub total_secs: f64,
    pub cuda_mem_used_before: usize,
    pub cuda_mem_used_after: usize,
    pub matmul_path: &'static str,
}

#[derive(Debug, Clone)]
struct CudaConfig {
    hidden_size: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    rope_local_base_freq: f32,
    sliding_window: usize,
    layer_types: Vec<LayerType>,
    query_pre_attn_scalar: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LayerType {
    SlidingAttention,
    FullAttention,
}

struct ForwardWorkspace {
    state_a: DeviceBuffer,
    state_b: DeviceBuffer,
    layer: LayerWorkspace,
    head_hidden: DeviceBuffer,
    pooled: DeviceBuffer,
    dense2: DeviceBuffer,
    dense3: DeviceBuffer,
    l2: DeviceBuffer,
    q8_scratch: DeviceBuffer,
    fixup_scratch: DeviceBuffer,
}

struct LayerWorkspace {
    xs_norm: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    q_head: DeviceBuffer,
    k_head: DeviceBuffer,
    v_head: DeviceBuffer,
    q_rope: DeviceBuffer,
    k_rope: DeviceBuffer,
    scores: DeviceBuffer,
    attn_head: DeviceBuffer,
    attn: DeviceBuffer,
    attn_proj: DeviceBuffer,
    sa_out: DeviceBuffer,
    ffn_norm: DeviceBuffer,
    gate: DeviceBuffer,
    up: DeviceBuffer,
    gated: DeviceBuffer,
    down: DeviceBuffer,
}

impl CudaEmbeddingModel {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let gguf = GgufModel::open(path)?;
        Self::from_gguf(&gguf)
    }

    pub fn from_gguf(model: &GgufModel) -> Result<Self> {
        let device = std::env::var("EMBED_NATIVE_CUDA_DEVICE")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(0);
        let dev = CudaDevice::new(device)?;
        let cfg = CudaConfig::from_model(model)?;
        let weights = CudaWeights::load(&dev, model)?;
        Ok(Self { dev, weights, cfg })
    }

    pub fn forward_batch(&self, batch: &TokenizedBatch) -> Result<Vec<Vec<f32>>> {
        self.forward_tokens(&batch.token_ids, &batch.attention_mask)
    }

    pub fn forward_tokens(
        &self,
        token_ids: &[Vec<u32>],
        attention_mask: &[Vec<u32>],
    ) -> Result<Vec<Vec<f32>>> {
        Ok(self
            .forward_tokens_inner(token_ids, attention_mask, false)?
            .0)
    }

    pub fn forward_tokens_profiled(
        &self,
        token_ids: &[Vec<u32>],
        attention_mask: &[Vec<u32>],
    ) -> Result<(Vec<Vec<f32>>, CudaForwardProfile)> {
        self.forward_tokens_inner(token_ids, attention_mask, true)
    }

    fn forward_tokens_inner(
        &self,
        token_ids: &[Vec<u32>],
        attention_mask: &[Vec<u32>],
        collect_profile: bool,
    ) -> Result<(Vec<Vec<f32>>, CudaForwardProfile)> {
        let t0 = Instant::now();
        let mem_before = self.used_mem().unwrap_or(0);
        let (batch, seq_len) = validate_batch(token_ids, attention_mask)?;
        let rows = batch * seq_len;
        let flat_ids = token_ids.iter().flatten().copied().collect::<Vec<_>>();
        let flat_mask = attention_mask.iter().flatten().copied().collect::<Vec<_>>();
        let ids = self.upload_pod(&flat_ids)?;
        let mask = self.upload_pod(&flat_mask)?;
        let workspace = ForwardWorkspace::new(self, rows, batch, seq_len)?;

        self.record_embedding(&ids, &workspace.state_a, rows)?;
        self.debug_check("embed", &workspace.state_a, rows * self.cfg.hidden_size)?;
        let mut xs = &workspace.state_a;
        for layer_idx in 0..self.cfg.num_hidden_layers {
            let out = if layer_idx % 2 == 0 {
                &workspace.state_b
            } else {
                &workspace.state_a
            };
            self.record_layer(layer_idx, xs, out, &workspace, &mask, batch, seq_len)?;
            self.debug_check(
                &format!("layer_{layer_idx}"),
                out,
                rows * self.cfg.hidden_size,
            )?;
            xs = out;
        }
        self.record_head(xs, &mask, &workspace, rows, batch, seq_len)?;

        let mut out = vec![0.0f32; batch * EMBEDDING_DIM];
        self.dev.copy_d2h(&mut out, &workspace.l2)?;
        ensure_finite("CUDA final embedding", &out)?;
        let mem_after = self.used_mem().unwrap_or(mem_before);
        let profile = if collect_profile {
            CudaForwardProfile {
                total_secs: t0.elapsed().as_secs_f64(),
                cuda_mem_used_before: mem_before,
                cuda_mem_used_after: mem_after,
                matmul_path: "ggml-cuda MMQ mul_mat_q + quantize_mmq_q8_1",
            }
        } else {
            CudaForwardProfile::default()
        };
        Ok((
            out.chunks_exact(EMBEDDING_DIM)
                .map(|row| row.to_vec())
                .collect(),
            profile,
        ))
    }

    fn record_embedding(&self, ids: &DeviceBuffer, dst: &DeviceBuffer, rows: usize) -> Result<()> {
        let token = self.weights.require("token_embd.weight")?;
        let scale = (self.cfg.hidden_size as f32).sqrt();
        match token.dtype {
            GgmlDType::Q4K => check(
                unsafe {
                    gp_embed_q4k(
                        token.buffer.ptr(),
                        ids.as_u32(),
                        dst.as_f32(),
                        rows as i32,
                        self.cfg.hidden_size as i32,
                        scale,
                        self.dev.stream(),
                    )
                },
                "cuda embed_q4k",
            ),
            GgmlDType::Q6K => check(
                unsafe {
                    gp_embed_q6k(
                        token.buffer.ptr(),
                        ids.as_u32(),
                        dst.as_f32(),
                        rows as i32,
                        self.cfg.hidden_size as i32,
                        scale,
                        self.dev.stream(),
                    )
                },
                "cuda embed_q6k",
            ),
            other => Err(Error::UnsupportedDType(other)),
        }
    }

    fn record_layer(
        &self,
        layer_idx: usize,
        xs: &DeviceBuffer,
        out: &DeviceBuffer,
        workspace: &ForwardWorkspace,
        mask: &DeviceBuffer,
        batch: usize,
        seq_len: usize,
    ) -> Result<()> {
        let rows = batch * seq_len;
        let prefix = format!("blk.{layer_idx}");
        self.rms_norm_rows(
            &format!("{prefix}.attn_norm.weight"),
            xs,
            &workspace.layer.xs_norm,
            rows,
            self.cfg.hidden_size,
        )?;
        self.debug_check(
            &format!("{prefix}.attn_norm"),
            &workspace.layer.xs_norm,
            rows * self.cfg.hidden_size,
        )?;
        self.matmul(
            &format!("{prefix}.attn_q.weight"),
            &workspace.layer.xs_norm,
            &workspace.layer.q,
            rows,
            self.cfg.hidden_size,
            workspace,
        )?;
        self.debug_check(
            &format!("{prefix}.attn_q"),
            &workspace.layer.q,
            rows * self.cfg.hidden_size,
        )?;
        self.matmul(
            &format!("{prefix}.attn_k.weight"),
            &workspace.layer.xs_norm,
            &workspace.layer.k,
            rows,
            self.cfg.hidden_size,
            workspace,
        )?;
        self.debug_check(
            &format!("{prefix}.attn_k"),
            &workspace.layer.k,
            rows * self.cfg.head_dim * self.cfg.num_key_value_heads,
        )?;
        self.matmul(
            &format!("{prefix}.attn_v.weight"),
            &workspace.layer.xs_norm,
            &workspace.layer.v,
            rows,
            self.cfg.hidden_size,
            workspace,
        )?;
        self.debug_check(
            &format!("{prefix}.attn_v"),
            &workspace.layer.v,
            rows * self.cfg.head_dim * self.cfg.num_key_value_heads,
        )?;
        let rope_base = match self.cfg.layer_types[layer_idx] {
            LayerType::FullAttention => self.cfg.rope_theta,
            LayerType::SlidingAttention => self.cfg.rope_local_base_freq,
        };
        self.rms_norm_heads(
            &format!("{prefix}.attn_q_norm.weight"),
            &workspace.layer.q,
            &workspace.layer.q_head,
            batch,
            seq_len,
            self.cfg.num_attention_heads,
            self.cfg.hidden_size,
        )?;
        self.rms_norm_heads(
            &format!("{prefix}.attn_k_norm.weight"),
            &workspace.layer.k,
            &workspace.layer.k_head,
            batch,
            seq_len,
            self.cfg.num_key_value_heads,
            self.cfg.head_dim * self.cfg.num_key_value_heads,
        )?;
        self.split_heads(
            &workspace.layer.v,
            &workspace.layer.v_head,
            batch,
            seq_len,
            self.cfg.num_key_value_heads,
            self.cfg.head_dim * self.cfg.num_key_value_heads,
        )?;
        self.rope(
            &workspace.layer.q_head,
            &workspace.layer.q_rope,
            batch,
            seq_len,
            self.cfg.num_attention_heads,
            rope_base,
        )?;
        self.rope(
            &workspace.layer.k_head,
            &workspace.layer.k_rope,
            batch,
            seq_len,
            self.cfg.num_key_value_heads,
            rope_base,
        )?;
        self.debug_check(
            &format!("{prefix}.q_rope"),
            &workspace.layer.q_rope,
            batch * self.cfg.num_attention_heads * seq_len * self.cfg.head_dim,
        )?;
        self.debug_check(
            &format!("{prefix}.k_rope"),
            &workspace.layer.k_rope,
            batch * self.cfg.num_key_value_heads * seq_len * self.cfg.head_dim,
        )?;
        self.attention(
            &workspace.layer.q_rope,
            &workspace.layer.k_rope,
            &workspace.layer.v_head,
            mask,
            &workspace.layer.scores,
            &workspace.layer.attn_head,
            &workspace.layer.attn,
            layer_idx,
            batch,
            seq_len,
        )?;
        self.debug_check(
            &format!("{prefix}.attn"),
            &workspace.layer.attn,
            rows * self.cfg.hidden_size,
        )?;
        self.matmul(
            &format!("{prefix}.attn_output.weight"),
            &workspace.layer.attn,
            &workspace.layer.attn_proj,
            rows,
            self.cfg.hidden_size,
            workspace,
        )?;
        self.debug_check(
            &format!("{prefix}.attn_output"),
            &workspace.layer.attn_proj,
            rows * self.cfg.hidden_size,
        )?;
        self.rms_norm_rows_add(
            &format!("{prefix}.post_attention_norm.weight"),
            &workspace.layer.attn_proj,
            xs,
            &workspace.layer.sa_out,
            rows,
            self.cfg.hidden_size,
        )?;
        self.debug_check(
            &format!("{prefix}.post_attn"),
            &workspace.layer.sa_out,
            rows * self.cfg.hidden_size,
        )?;
        self.rms_norm_rows(
            &format!("{prefix}.ffn_norm.weight"),
            &workspace.layer.sa_out,
            &workspace.layer.ffn_norm,
            rows,
            self.cfg.hidden_size,
        )?;
        self.debug_check(
            &format!("{prefix}.ffn_norm"),
            &workspace.layer.ffn_norm,
            rows * self.cfg.hidden_size,
        )?;
        self.matmul(
            &format!("{prefix}.ffn_gate.weight"),
            &workspace.layer.ffn_norm,
            &workspace.layer.gate,
            rows,
            self.cfg.hidden_size,
            workspace,
        )?;
        self.debug_check(
            &format!("{prefix}.ffn_gate"),
            &workspace.layer.gate,
            rows * self.cfg.intermediate_size,
        )?;
        self.matmul(
            &format!("{prefix}.ffn_up.weight"),
            &workspace.layer.ffn_norm,
            &workspace.layer.up,
            rows,
            self.cfg.hidden_size,
            workspace,
        )?;
        self.debug_check(
            &format!("{prefix}.ffn_up"),
            &workspace.layer.up,
            rows * self.cfg.intermediate_size,
        )?;
        check(
            unsafe {
                gp_geglu(
                    workspace.layer.gate.as_f32(),
                    workspace.layer.up.as_f32(),
                    workspace.layer.gated.as_f32(),
                    (rows * self.cfg.intermediate_size) as i32,
                    self.dev.stream(),
                )
            },
            "cuda geglu",
        )?;
        self.debug_check(
            &format!("{prefix}.geglu"),
            &workspace.layer.gated,
            rows * self.cfg.intermediate_size,
        )?;
        self.matmul(
            &format!("{prefix}.ffn_down.weight"),
            &workspace.layer.gated,
            &workspace.layer.down,
            rows,
            self.cfg.intermediate_size,
            workspace,
        )?;
        self.debug_check(
            &format!("{prefix}.ffn_down"),
            &workspace.layer.down,
            rows * self.cfg.hidden_size,
        )?;
        self.rms_norm_rows_add(
            &format!("{prefix}.post_ffw_norm.weight"),
            &workspace.layer.down,
            &workspace.layer.sa_out,
            out,
            rows,
            self.cfg.hidden_size,
        )?;
        self.debug_check(
            &format!("{prefix}.post_ffn"),
            out,
            rows * self.cfg.hidden_size,
        )
    }

    fn record_head(
        &self,
        xs: &DeviceBuffer,
        mask: &DeviceBuffer,
        workspace: &ForwardWorkspace,
        rows: usize,
        batch: usize,
        seq_len: usize,
    ) -> Result<()> {
        self.rms_norm_rows(
            "output_norm.weight",
            xs,
            &workspace.head_hidden,
            rows,
            self.cfg.hidden_size,
        )?;
        check(
            unsafe {
                gp_mean_pool(
                    workspace.head_hidden.as_f32(),
                    mask.as_u32(),
                    workspace.pooled.as_f32(),
                    batch as i32,
                    seq_len as i32,
                    self.cfg.hidden_size as i32,
                    self.dev.stream(),
                )
            },
            "cuda mean_pool",
        )?;
        self.matmul(
            "dense_2.weight",
            &workspace.pooled,
            &workspace.dense2,
            batch,
            self.cfg.hidden_size,
            workspace,
        )?;
        self.matmul(
            "dense_3.weight",
            &workspace.dense2,
            &workspace.dense3,
            batch,
            DENSE2_DIM,
            workspace,
        )?;
        check(
            unsafe {
                gp_l2_norm(
                    workspace.dense3.as_f32(),
                    workspace.l2.as_f32(),
                    batch as i32,
                    EMBEDDING_DIM as i32,
                    self.dev.stream(),
                )
            },
            "cuda l2_norm",
        )
    }

    fn matmul(
        &self,
        name: &str,
        src: &DeviceBuffer,
        dst: &DeviceBuffer,
        rows: usize,
        cols: usize,
        workspace: &ForwardWorkspace,
    ) -> Result<()> {
        let w = self.weights.require(name)?;
        let w_cols = w.cols()?;
        let w_rows = w.rows()?;
        if w_cols != cols {
            return Err(Error::InvalidGguf(format!(
                "{name} input cols {}, expected {cols}",
                w_cols
            )));
        }
        check(
            unsafe {
                gp_mmq_matmul(
                    w.ggml_type_id()?,
                    w.buffer.ptr(),
                    src.as_f32(),
                    dst.as_f32(),
                    workspace.q8_scratch.ptr(),
                    workspace.fixup_scratch.ptr(),
                    cols as i64,
                    w.row_stride_blocks() as i64,
                    w_rows as i64,
                    rows as i64,
                    self.dev.stream(),
                )
            },
            &format!("cuda MMQ matmul {name}"),
        )
    }

    fn rms_norm_rows(
        &self,
        name: &str,
        src: &DeviceBuffer,
        dst: &DeviceBuffer,
        rows: usize,
        dim: usize,
    ) -> Result<()> {
        let weight = self.weights.require(name)?;
        check(
            unsafe {
                gp_rms_norm(
                    src.as_f32(),
                    weight.buffer.as_f32(),
                    dst.as_f32(),
                    rows as i32,
                    dim as i32,
                    self.cfg.rms_norm_eps,
                    self.dev.stream(),
                )
            },
            &format!("cuda rms_norm {name}"),
        )
    }

    fn rms_norm_rows_add(
        &self,
        name: &str,
        src: &DeviceBuffer,
        add: &DeviceBuffer,
        dst: &DeviceBuffer,
        rows: usize,
        dim: usize,
    ) -> Result<()> {
        let weight = self.weights.require(name)?;
        check(
            unsafe {
                gp_rms_norm_add(
                    src.as_f32(),
                    add.as_f32(),
                    weight.buffer.as_f32(),
                    dst.as_f32(),
                    rows as i32,
                    dim as i32,
                    self.cfg.rms_norm_eps,
                    self.dev.stream(),
                )
            },
            &format!("cuda rms_norm_add {name}"),
        )
    }

    fn rms_norm_heads(
        &self,
        name: &str,
        src: &DeviceBuffer,
        dst: &DeviceBuffer,
        batch: usize,
        seq_len: usize,
        heads: usize,
        row_width: usize,
    ) -> Result<()> {
        let weight = self.weights.require(name)?;
        check(
            unsafe {
                gp_rms_norm_heads(
                    src.as_f32(),
                    weight.buffer.as_f32(),
                    dst.as_f32(),
                    batch as i32,
                    seq_len as i32,
                    heads as i32,
                    self.cfg.head_dim as i32,
                    row_width as i32,
                    self.cfg.rms_norm_eps,
                    self.dev.stream(),
                )
            },
            &format!("cuda rms_norm_heads {name}"),
        )
    }

    fn split_heads(
        &self,
        src: &DeviceBuffer,
        dst: &DeviceBuffer,
        batch: usize,
        seq_len: usize,
        heads: usize,
        row_width: usize,
    ) -> Result<()> {
        check(
            unsafe {
                gp_split_heads(
                    src.as_f32(),
                    dst.as_f32(),
                    batch as i32,
                    seq_len as i32,
                    heads as i32,
                    self.cfg.head_dim as i32,
                    row_width as i32,
                    self.dev.stream(),
                )
            },
            "cuda split_heads",
        )
    }

    fn rope(
        &self,
        src: &DeviceBuffer,
        dst: &DeviceBuffer,
        batch: usize,
        seq_len: usize,
        heads: usize,
        base_freq: f32,
    ) -> Result<()> {
        if seq_len > self.cfg.max_position_embeddings {
            return Err(Error::InvalidGguf(format!(
                "seq_len {seq_len} exceeds max_position_embeddings {}",
                self.cfg.max_position_embeddings
            )));
        }
        check(
            unsafe {
                gp_rope_neox(
                    src.as_f32(),
                    dst.as_f32(),
                    batch as i32,
                    seq_len as i32,
                    heads as i32,
                    self.cfg.head_dim as i32,
                    base_freq,
                    self.dev.stream(),
                )
            },
            "cuda rope_neox",
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        q: &DeviceBuffer,
        k: &DeviceBuffer,
        v: &DeviceBuffer,
        mask: &DeviceBuffer,
        scores: &DeviceBuffer,
        attn_head: &DeviceBuffer,
        attn: &DeviceBuffer,
        layer_idx: usize,
        batch: usize,
        seq_len: usize,
    ) -> Result<()> {
        check(
            unsafe {
                gp_attention_scores(
                    self.dev.blas(),
                    q.as_f32(),
                    k.as_f32(),
                    scores.as_f32(),
                    batch as i32,
                    self.cfg.num_attention_heads as i32,
                    seq_len as i32,
                    self.cfg.head_dim as i32,
                )
            },
            "cuda attention_scores",
        )?;
        let sliding = match self.cfg.layer_types[layer_idx] {
            LayerType::FullAttention => 0,
            LayerType::SlidingAttention => self.cfg.effective_sliding_window() as i32,
        };
        check(
            unsafe {
                gp_softmax_mask(
                    scores.as_f32(),
                    mask.as_u32(),
                    batch as i32,
                    self.cfg.num_attention_heads as i32,
                    seq_len as i32,
                    sliding,
                    self.cfg.query_pre_attn_scalar.powf(-0.5),
                    self.dev.stream(),
                )
            },
            "cuda softmax_mask",
        )?;
        check(
            unsafe {
                gp_attention_values(
                    self.dev.blas(),
                    scores.as_f32(),
                    v.as_f32(),
                    attn_head.as_f32(),
                    batch as i32,
                    self.cfg.num_attention_heads as i32,
                    seq_len as i32,
                    self.cfg.head_dim as i32,
                )
            },
            "cuda attention_values",
        )?;
        check(
            unsafe {
                gp_merge_heads(
                    attn_head.as_f32(),
                    attn.as_f32(),
                    batch as i32,
                    seq_len as i32,
                    self.cfg.num_attention_heads as i32,
                    self.cfg.head_dim as i32,
                    self.dev.stream(),
                )
            },
            "cuda merge_heads",
        )
    }

    fn upload_pod<T>(&self, values: &[T]) -> Result<DeviceBuffer> {
        let buf = self.dev.alloc(std::mem::size_of_val(values).max(1))?;
        self.dev.copy_h2d(&buf, values)?;
        Ok(buf)
    }

    fn new_f32(&self, elems: usize) -> Result<DeviceBuffer> {
        let buf = self.dev.alloc(elems.max(1) * std::mem::size_of::<f32>())?;
        self.dev.memset(&buf, 0)?;
        Ok(buf)
    }

    fn new_bytes(&self, bytes: usize) -> Result<DeviceBuffer> {
        let buf = self.dev.alloc(bytes.max(1))?;
        self.dev.memset(&buf, 0)?;
        Ok(buf)
    }

    fn used_mem(&self) -> Result<usize> {
        let (free, total) = self.dev.mem_info()?;
        Ok(total.saturating_sub(free))
    }

    fn debug_check(&self, name: &str, buf: &DeviceBuffer, elems: usize) -> Result<()> {
        if std::env::var_os("EMBED_NATIVE_CUDA_DEBUG_NAN").is_none() {
            return Ok(());
        }
        let mut values = vec![0.0f32; elems];
        self.dev.copy_d2h(&mut values, buf)?;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut first_bad = None;
        for (idx, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                first_bad = Some((idx, value));
                break;
            }
            min = min.min(value);
            max = max.max(value);
        }
        eprintln!("cuda debug {name}: min={min:.6e} max={max:.6e} first_bad={first_bad:?}");
        if let Some((idx, value)) = first_bad {
            return Err(Error::InvalidGguf(format!(
                "CUDA non-finite at {name}[{idx}] = {value}"
            )));
        }
        Ok(())
    }
}

impl ForwardWorkspace {
    fn new(model: &CudaEmbeddingModel, rows: usize, batch: usize, seq_len: usize) -> Result<Self> {
        Ok(Self {
            state_a: model.new_f32(rows * model.cfg.hidden_size)?,
            state_b: model.new_f32(rows * model.cfg.hidden_size)?,
            layer: LayerWorkspace::new(model, rows, batch, seq_len)?,
            head_hidden: model.new_f32(rows * model.cfg.hidden_size)?,
            pooled: model.new_f32(batch * model.cfg.hidden_size)?,
            dense2: model.new_f32(batch * DENSE2_DIM)?,
            dense3: model.new_f32(batch * EMBEDDING_DIM)?,
            l2: model.new_f32(batch * EMBEDDING_DIM)?,
            q8_scratch: model.new_bytes(q8_scratch_bytes(rows, batch))?,
            fixup_scratch: model.new_bytes(128 * 128 * 128 * std::mem::size_of::<f32>())?,
        })
    }
}

impl LayerWorkspace {
    fn new(model: &CudaEmbeddingModel, rows: usize, batch: usize, seq_len: usize) -> Result<Self> {
        let hidden = rows * model.cfg.hidden_size;
        let kv = rows * model.cfg.head_dim * model.cfg.num_key_value_heads;
        let q_head = batch * model.cfg.num_attention_heads * seq_len * model.cfg.head_dim;
        let kv_head = batch * model.cfg.num_key_value_heads * seq_len * model.cfg.head_dim;
        let intermediate = rows * model.cfg.intermediate_size;
        let scores = batch * model.cfg.num_attention_heads * seq_len * seq_len;
        Ok(Self {
            xs_norm: model.new_f32(hidden)?,
            q: model.new_f32(hidden)?,
            k: model.new_f32(kv)?,
            v: model.new_f32(kv)?,
            q_head: model.new_f32(q_head)?,
            k_head: model.new_f32(kv_head)?,
            v_head: model.new_f32(kv_head)?,
            q_rope: model.new_f32(q_head)?,
            k_rope: model.new_f32(kv_head)?,
            scores: model.new_f32(scores)?,
            attn_head: model.new_f32(q_head)?,
            attn: model.new_f32(hidden)?,
            attn_proj: model.new_f32(hidden)?,
            sa_out: model.new_f32(hidden)?,
            ffn_norm: model.new_f32(hidden)?,
            gate: model.new_f32(intermediate)?,
            up: model.new_f32(intermediate)?,
            gated: model.new_f32(intermediate)?,
            down: model.new_f32(hidden)?,
        })
    }
}

impl CudaConfig {
    fn from_model(model: &GgufModel) -> Result<Self> {
        let arch = model.metadata_str("general.architecture")?;
        if arch != GGUF_ARCHITECTURE {
            return Err(Error::InvalidGguf(format!(
                "expected GGUF architecture {GGUF_ARCHITECTURE}, got {arch}"
            )));
        }
        let hidden_size = model.metadata_u32("gemma-embedding.embedding_length")? as usize;
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
        let attention_width = num_attention_heads
            .checked_mul(head_dim)
            .ok_or_else(|| Error::InvalidGguf("head_count * head_dim overflows".into()))?;
        if hidden_size != EMBEDDING_DIM || attention_width != hidden_size {
            return Err(Error::InvalidGguf(format!(
                "unexpected Gemma dimensions hidden={hidden_size}, heads={num_attention_heads}, head_dim={head_dim}"
            )));
        }
        if num_attention_heads % num_key_value_heads != 0 {
            return Err(Error::InvalidGguf(format!(
                "attention heads {num_attention_heads} not divisible by kv heads {num_key_value_heads}"
            )));
        }
        let intermediate_size = model.metadata_u32("gemma-embedding.feed_forward_length")? as usize;
        let max_position_embeddings =
            model.metadata_u32("gemma-embedding.context_length")? as usize;
        let rms_norm_eps =
            model.metadata_f32("gemma-embedding.attention.layer_norm_rms_epsilon")?;
        let rope_theta = model.metadata_f32("gemma-embedding.rope.freq_base")?;
        let rope_local_base_freq = metadata_f32_opt(model, "gemma-embedding.rope.local_freq_base")
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

fn q8_scratch_bytes(rows: usize, batch: usize) -> usize {
    let max_cols = DENSE2_DIM;
    let max_rows = rows.max(batch);
    let padded = max_cols.div_ceil(MMQ_MATRIX_ROW_PADDING) * MMQ_MATRIX_ROW_PADDING;
    max_rows * (padded / 32) * 36 + 128 * 144
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
    let rows = token_ids
        .len()
        .checked_mul(seq_len)
        .ok_or_else(|| Error::InvalidGguf("token batch row count overflows".into()))?;
    if rows > i32::MAX as usize || seq_len > i32::MAX as usize {
        return Err(Error::InvalidGguf(format!(
            "token batch shape {}x{seq_len} exceeds CUDA i32 dispatch limits",
            token_ids.len()
        )));
    }
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

fn metadata_u32_opt(model: &GgufModel, key: &str) -> Option<u32> {
    model.metadata().get(key).and_then(Value::as_u32)
}

fn metadata_f32_opt(model: &GgufModel, key: &str) -> Option<f32> {
    model.metadata().get(key).and_then(Value::as_f32)
}
