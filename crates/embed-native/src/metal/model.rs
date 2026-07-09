//! Metal Gemma3 embedding graph.
//!
//! The graph mirrors `llama.cpp/src/models/gemma-embedding.cpp`: embedding
//! lookup + scale, 24 decoder blocks, final norm, masked mean pool, dense head,
//! and L2 normalization. The hot ops dispatch the vendored ggml Metal kernels.

use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use half::f16;

use crate::gguf::{GgufModel, Value};
use crate::metal::ffi::{
    self, global_device, Buffer, CommandBuffer, CommandBufferTiming, Device, SubmittedCommandBuffer,
};
use crate::metal::ops::{self, GgmlType as OpType};
use crate::metal::tensor::GgmlType;
use crate::metal::weights::MetalWeights;
use crate::{Error, Result, TokenizedBatch};

const GGUF_ARCHITECTURE: &str = "gemma-embedding";
const DEFAULT_GGUF_LOCAL_ROPE_FREQ: f64 = 10_000.0;
const DEFAULT_GGUF_SLIDING_PATTERN: usize = 6;
const EMBEDDING_DIM: usize = 768;
const DENSE2_DIM: usize = 3072;
const TENSOR_MUL_MM_NRB: usize = 128;

pub struct MetalEmbeddingModel {
    dev: &'static Device,
    weights: MetalWeights,
    cfg: MetalConfig,
    shape_cache: Mutex<Vec<ShapeCacheEntry>>,
}

#[derive(Debug, Clone, Default)]
pub struct MetalForwardProfile {
    pub matmul_path: &'static str,
    pub total_secs: f64,
    pub cpu_prepare_secs: f64,
    pub cpu_encode_secs: f64,
    pub cpu_submit_wait_secs: f64,
    pub metal_kernel_secs: f64,
    pub metal_gpu_secs: f64,
    pub output_read_secs: f64,
    pub command_buffers: usize,
    pub dispatches: u64,
    pub buffer_allocs: u64,
    pub buffer_alloc_bytes: u64,
    pub stages: Vec<MetalProfileStage>,
    pub op_breakdown: Vec<MetalOpProfile>,
}

#[derive(Debug, Clone, Default)]
pub struct MetalProfileStage {
    pub name: String,
    pub cpu_encode_secs: f64,
    pub cpu_submit_wait_secs: f64,
    pub metal_kernel_secs: f64,
    pub metal_gpu_secs: f64,
    pub dispatches: u64,
}

#[derive(Debug, Clone, Default)]
pub struct MetalOpProfile {
    pub op_type: String,
    pub stages: usize,
    pub cpu_encode_secs: f64,
    pub cpu_submit_wait_secs: f64,
    pub metal_kernel_secs: f64,
    pub metal_gpu_secs: f64,
    pub dispatches: u64,
}

impl MetalForwardProfile {
    pub fn cpu_non_gpu_secs(&self) -> f64 {
        (self.total_secs - self.metal_gpu_secs).max(0.0)
    }

    pub fn sync_overhead_secs(&self) -> f64 {
        (self.cpu_submit_wait_secs - self.metal_gpu_secs).max(0.0)
    }

    fn add_command_timing(&mut self, timing: CommandBufferTiming) {
        self.command_buffers += 1;
        self.cpu_submit_wait_secs += timing.submit_wait_secs;
        self.metal_kernel_secs += timing.kernel_secs;
        self.metal_gpu_secs += timing.gpu_secs;
    }

    fn add_op_timing(&mut self, op_type: &str, timing: CommandBufferTiming, dispatches: u64) {
        if let Some(op) = self
            .op_breakdown
            .iter_mut()
            .find(|op| op.op_type == op_type)
        {
            op.stages += 1;
            op.cpu_submit_wait_secs += timing.submit_wait_secs;
            op.metal_kernel_secs += timing.kernel_secs;
            op.metal_gpu_secs += timing.gpu_secs;
            op.dispatches += dispatches;
            return;
        }
        self.op_breakdown.push(MetalOpProfile {
            op_type: op_type.to_string(),
            stages: 1,
            cpu_encode_secs: 0.0,
            cpu_submit_wait_secs: timing.submit_wait_secs,
            metal_kernel_secs: timing.kernel_secs,
            metal_gpu_secs: timing.gpu_secs,
            dispatches,
        });
    }

    fn add_op_encode(&mut self, op_type: &str, cpu_encode_secs: f64) {
        if let Some(op) = self
            .op_breakdown
            .iter_mut()
            .find(|op| op.op_type == op_type)
        {
            op.cpu_encode_secs += cpu_encode_secs;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProfileMode {
    None,
    SingleCommand,
    Stage,
    OpBreakdown,
}

#[derive(Debug, Clone)]
struct MetalConfig {
    hidden_size: usize,
    intermediate_size: usize,
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

#[repr(C)]
struct MeanPoolArgs {
    batch: i32,
    seq_len: i32,
    hidden: i32,
}

#[repr(C)]
struct ScaleArgs {
    n: i32,
    scale: f32,
}

#[repr(C)]
struct RmsNormRopeArgs {
    batch: i32,
    seq_len: i32,
    heads: i32,
    head_dim: i32,
    row_width: i32,
    eps: f32,
    freq_base: f32,
    pad: i32,
}

#[repr(C)]
struct PostAttnFfnNormArgs {
    rows: i32,
    dim: i32,
    eps: f32,
    pad: i32,
}

struct ForwardWorkspace {
    embed_raw: Buffer,
    state_a: Buffer,
    state_b: Buffer,
    layer: LayerWorkspace,
    head_hidden: Buffer,
    pooled: Buffer,
    dense2: Buffer,
    dense3: Buffer,
    l2: Buffer,
    readback: Buffer,
}

struct LayerWorkspace {
    xs_norm: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    qr: Buffer,
    kr: Buffer,
    attn: Buffer,
    attn_proj: Buffer,
    sa_out: Buffer,
    ffn_norm: Buffer,
    gate: Buffer,
    up: Buffer,
    gated: Buffer,
    down: Buffer,
    flash_pad: Buffer,
    flash_blk: Buffer,
    flash_tmp: Buffer,
}

struct ShapeCacheEntry {
    batch: usize,
    input_seq_len: usize,
    seq_len: usize,
    ids_buf: Buffer,
    mask_u32_buf: Buffer,
    pos_buf: Buffer,
    full_mask_buf: Buffer,
    sliding_mask_buf: Buffer,
    workspace: ForwardWorkspace,
    flat_ids: Vec<u32>,
    flat_mask: Vec<u32>,
    last_mask: Vec<u32>,
    full_mask_bits: Vec<u16>,
    sliding_mask_bits: Vec<u16>,
    mask_cache_valid: bool,
    full_mask_present: bool,
    sliding_mask_present: bool,
}

impl MetalEmbeddingModel {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let gguf = GgufModel::open(path)?;
        Self::from_gguf(&gguf)
    }

    pub fn from_gguf(model: &GgufModel) -> Result<Self> {
        let dev = global_device()
            .ok_or_else(|| Error::InvalidGguf("Metal device or metallib unavailable".into()))?;
        let cfg = MetalConfig::from_model(model)?;
        let weights = MetalWeights::load(dev, model)?;
        Ok(Self {
            dev,
            weights,
            cfg,
            shape_cache: Mutex::new(Vec::new()),
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
        Ok(self
            .forward_tokens_inner(token_ids, attention_mask, ProfileMode::None)?
            .0)
    }

    pub fn forward_tokens_profiled(
        &self,
        token_ids: &[Vec<u32>],
        attention_mask: &[Vec<u32>],
    ) -> Result<(Vec<Vec<f32>>, MetalForwardProfile)> {
        let mode = if op_profile_enabled() {
            ProfileMode::OpBreakdown
        } else if stage_profile_enabled() {
            ProfileMode::Stage
        } else {
            ProfileMode::SingleCommand
        };
        self.forward_tokens_inner(token_ids, attention_mask, mode)
    }

    pub fn forward_tokens_op_profiled(
        &self,
        token_ids: &[Vec<u32>],
        attention_mask: &[Vec<u32>],
    ) -> Result<(Vec<Vec<f32>>, MetalForwardProfile)> {
        self.forward_tokens_inner(token_ids, attention_mask, ProfileMode::OpBreakdown)
    }

    pub fn mul_mm_path(&self) -> &'static str {
        if self.dev.uses_tensor_mul_mm() {
            "metal4 tensor mul_mm"
        } else {
            "metal3 simdgroup mul_mm"
        }
    }

    pub fn forward_tokens_pipelined_repeated(
        &self,
        token_ids: &[Vec<u32>],
        attention_mask: &[Vec<u32>],
        repeats: usize,
    ) -> Result<(Vec<Vec<f32>>, f64)> {
        if repeats == 0 {
            return Ok((Vec::new(), 0.0));
        }
        let (batch, input_seq_len) = validate_batch(token_ids, attention_mask)?;
        let seq_len = tensor_aligned_seq_len(batch, input_seq_len)?;
        let rows = batch
            .checked_mul(seq_len)
            .ok_or_else(|| Error::InvalidGguf("Metal row count overflows".into()))?;
        if rows > i32::MAX as usize {
            return Err(Error::InvalidGguf(format!(
                "Metal aligned row count {rows} exceeds i32 dispatch limits"
            )));
        }
        let mut slots = [
            ShapeCacheEntry::new(self, batch, input_seq_len, seq_len)?,
            ShapeCacheEntry::new(self, batch, input_seq_len, seq_len)?,
        ];
        let mut submitted: [Option<SubmittedCommandBuffer>; 2] = [None, None];
        let mut out = vec![0.0f32; batch * EMBEDDING_DIM];

        let t0 = Instant::now();
        for iter in 0..repeats {
            let slot_idx = iter % slots.len();
            wait_submitted(&mut submitted[slot_idx])?;
            unsafe {
                slots[slot_idx].workspace.readback.read(0, &mut out);
            }
            slots[slot_idx].upload_inputs(
                token_ids,
                attention_mask,
                self.cfg.effective_sliding_window(),
            );
            let cb = self.encode_single_command_buffer(
                &slots[slot_idx].ids_buf,
                &slots[slot_idx].mask_u32_buf,
                &slots[slot_idx].pos_buf,
                slots[slot_idx].full_mask(),
                slots[slot_idx].sliding_mask(),
                &slots[slot_idx].workspace,
                rows,
                batch,
                seq_len,
            )?;
            submitted[slot_idx] = Some(cb.commit());
        }
        let last_slot_idx = (repeats - 1) % slots.len();
        for (slot_idx, cb) in submitted.iter_mut().enumerate() {
            wait_submitted(cb)?;
            if slot_idx == last_slot_idx {
                unsafe {
                    slots[slot_idx].workspace.readback.read(0, &mut out);
                }
            }
        }
        ensure_finite("Metal final embedding", &out)?;
        let secs = t0.elapsed().as_secs_f64();

        let embeddings = out
            .chunks_exact(EMBEDDING_DIM)
            .map(|row| row.to_vec())
            .collect();
        Ok((embeddings, secs))
    }

    fn forward_tokens_inner(
        &self,
        token_ids: &[Vec<u32>],
        attention_mask: &[Vec<u32>],
        profile_mode: ProfileMode,
    ) -> Result<(Vec<Vec<f32>>, MetalForwardProfile)> {
        let total_t0 = Instant::now();
        let collect_profile = profile_mode != ProfileMode::None;
        let mut profile = if collect_profile {
            let mut p = MetalForwardProfile::default();
            p.matmul_path = self.mul_mm_path();
            Some(p)
        } else {
            None
        };
        let dispatch_start = ffi::dispatch_count_snapshot();
        let alloc_start = ffi::buffer_alloc_stats_snapshot();

        let prepare_t0 = Instant::now();
        let (batch, input_seq_len) = validate_batch(token_ids, attention_mask)?;
        let seq_len = tensor_aligned_seq_len(batch, input_seq_len)?;
        let rows = batch
            .checked_mul(seq_len)
            .ok_or_else(|| Error::InvalidGguf("Metal row count overflows".into()))?;
        if rows > i32::MAX as usize {
            return Err(Error::InvalidGguf(format!(
                "Metal aligned row count {rows} exceeds i32 dispatch limits"
            )));
        }

        let mut cache = self
            .shape_cache
            .lock()
            .map_err(|_| Error::InvalidGguf("Metal shape cache mutex poisoned".into()))?;
        let entry_idx = match cache.iter().position(|entry| {
            entry.batch == batch && entry.input_seq_len == input_seq_len && entry.seq_len == seq_len
        }) {
            Some(idx) => idx,
            None => {
                cache.push(ShapeCacheEntry::new(self, batch, input_seq_len, seq_len)?);
                cache.len() - 1
            }
        };
        let shape = &mut cache[entry_idx];
        shape.upload_inputs(
            token_ids,
            attention_mask,
            self.cfg.effective_sliding_window(),
        );
        if let Some(p) = profile.as_mut() {
            p.cpu_prepare_secs += prepare_t0.elapsed().as_secs_f64();
        }

        if profile_mode == ProfileMode::OpBreakdown {
            let profile_ref = profile.as_mut().ok_or_else(|| {
                Error::InvalidGguf("Metal profile missing for op breakdown mode".into())
            })?;
            self.encode_op_breakdown_profile(
                profile_ref,
                &shape.ids_buf,
                &shape.mask_u32_buf,
                &shape.pos_buf,
                shape.full_mask(),
                shape.sliding_mask(),
                &shape.workspace,
                rows,
                batch,
                seq_len,
            )?;
        } else if profile_mode == ProfileMode::Stage {
            let profile_ref = profile.as_mut().ok_or_else(|| {
                Error::InvalidGguf("Metal profile missing for stage profile mode".into())
            })?;
            self.encode_staged_profile(
                profile_ref,
                &shape.ids_buf,
                &shape.mask_u32_buf,
                &shape.pos_buf,
                shape.full_mask(),
                shape.sliding_mask(),
                &shape.workspace,
                rows,
                batch,
                seq_len,
            )?;
        } else {
            self.encode_single_command(
                &mut profile,
                &shape.ids_buf,
                &shape.mask_u32_buf,
                &shape.pos_buf,
                shape.full_mask(),
                shape.sliding_mask(),
                &shape.workspace,
                rows,
                batch,
                seq_len,
            )?;
        }

        let read_t0 = Instant::now();
        let mut out = vec![0.0f32; batch * EMBEDDING_DIM];
        unsafe {
            shape.workspace.readback.read(0, &mut out);
        }
        ensure_finite("Metal final embedding", &out)?;
        if let Some(p) = profile.as_mut() {
            p.output_read_secs += read_t0.elapsed().as_secs_f64();
            let alloc_end = ffi::buffer_alloc_stats_snapshot();
            p.dispatches = ffi::dispatch_count_snapshot().saturating_sub(dispatch_start);
            p.buffer_allocs = alloc_end.count.saturating_sub(alloc_start.count);
            p.buffer_alloc_bytes = alloc_end.bytes.saturating_sub(alloc_start.bytes);
            p.total_secs = total_t0.elapsed().as_secs_f64();
        }

        let embeddings = out
            .chunks_exact(EMBEDDING_DIM)
            .map(|row| row.to_vec())
            .collect();
        Ok((embeddings, profile.unwrap_or_default()))
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_single_command(
        &self,
        profile: &mut Option<MetalForwardProfile>,
        ids_buf: &Buffer,
        mask_u32_buf: &Buffer,
        pos_buf: &Buffer,
        full_mask: Option<&Buffer>,
        sliding_mask: Option<&Buffer>,
        workspace: &ForwardWorkspace,
        rows: usize,
        batch: usize,
        seq_len: usize,
    ) -> Result<()> {
        let encode_t0 = Instant::now();
        let cb = self.encode_single_command_buffer(
            ids_buf,
            mask_u32_buf,
            pos_buf,
            full_mask,
            sliding_mask,
            workspace,
            rows,
            batch,
            seq_len,
        )?;
        finish_command(cb, profile, encode_t0)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_single_command_buffer(
        &self,
        ids_buf: &Buffer,
        mask_u32_buf: &Buffer,
        pos_buf: &Buffer,
        full_mask: Option<&Buffer>,
        sliding_mask: Option<&Buffer>,
        workspace: &ForwardWorkspace,
        rows: usize,
        batch: usize,
        seq_len: usize,
    ) -> Result<CommandBuffer> {
        let cb = self.command()?;
        let enc = self.compute_encoder(&cb)?;
        self.record_embedding(
            &enc,
            ids_buf,
            rows,
            &workspace.embed_raw,
            &workspace.state_a,
        )?;

        let mut xs: &Buffer = &workspace.state_a;
        let mut attn_norm_ready = false;
        for layer_idx in 0..self.cfg.num_hidden_layers {
            let mask = match self.cfg.layer_types[layer_idx] {
                LayerType::FullAttention => full_mask,
                LayerType::SlidingAttention => sliding_mask,
            };
            let out = if layer_idx % 2 == 0 {
                &workspace.state_b
            } else {
                &workspace.state_a
            };
            self.record_layer(
                layer_idx,
                &enc,
                xs,
                out,
                &workspace.layer,
                rows,
                batch,
                seq_len,
                pos_buf,
                mask,
                attn_norm_ready,
                next_layer(layer_idx, self.cfg.num_hidden_layers),
            )?;
            attn_norm_ready = layer_idx + 1 < self.cfg.num_hidden_layers;
            xs = out;
        }

        self.record_head(&enc, xs, mask_u32_buf, workspace, rows, batch, seq_len)?;
        enc.end();
        self.record_readback_blit(&cb, workspace, batch)?;
        Ok(cb)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_staged_profile(
        &self,
        profile: &mut MetalForwardProfile,
        ids_buf: &Buffer,
        mask_u32_buf: &Buffer,
        pos_buf: &Buffer,
        full_mask: Option<&Buffer>,
        sliding_mask: Option<&Buffer>,
        workspace: &ForwardWorkspace,
        rows: usize,
        batch: usize,
        seq_len: usize,
    ) -> Result<()> {
        self.record_profile_stage(profile, "embed", |enc| {
            self.record_embedding(enc, ids_buf, rows, &workspace.embed_raw, &workspace.state_a)
        })?;

        let mut xs: &Buffer = &workspace.state_a;
        let mut attn_norm_ready = false;
        for layer_idx in 0..self.cfg.num_hidden_layers {
            let mask = match self.cfg.layer_types[layer_idx] {
                LayerType::FullAttention => full_mask,
                LayerType::SlidingAttention => sliding_mask,
            };
            let out = if layer_idx % 2 == 0 {
                &workspace.state_b
            } else {
                &workspace.state_a
            };
            let prefix = format!("layer_{layer_idx:02}");
            self.record_profile_stage(profile, format!("{prefix}:qkv_rope"), |enc| {
                self.record_layer_qkv_rope(
                    layer_idx,
                    enc,
                    xs,
                    &workspace.layer,
                    rows,
                    batch,
                    seq_len,
                    pos_buf,
                    attn_norm_ready,
                )
            })?;
            self.record_profile_stage(profile, format!("{prefix}:flash_attn"), |enc| {
                self.record_layer_attention(enc, &workspace.layer, mask, batch, seq_len)
            })?;
            self.record_profile_stage(profile, format!("{prefix}:attn_out"), |enc| {
                self.record_layer_attention_out(layer_idx, enc, xs, &workspace.layer, rows)
            })?;
            self.record_profile_stage(profile, format!("{prefix}:ffn"), |enc| {
                self.record_layer_ffn(
                    layer_idx,
                    enc,
                    out,
                    &workspace.layer,
                    rows,
                    next_layer(layer_idx, self.cfg.num_hidden_layers),
                )
            })?;
            attn_norm_ready = layer_idx + 1 < self.cfg.num_hidden_layers;
            xs = out;
        }

        self.record_profile_stage_with_blit(profile, "head", workspace, batch, |enc| {
            self.record_head(enc, xs, mask_u32_buf, workspace, rows, batch, seq_len)
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_op_breakdown_profile(
        &self,
        profile: &mut MetalForwardProfile,
        ids_buf: &Buffer,
        mask_u32_buf: &Buffer,
        _pos_buf: &Buffer,
        full_mask: Option<&Buffer>,
        sliding_mask: Option<&Buffer>,
        workspace: &ForwardWorkspace,
        rows: usize,
        batch: usize,
        seq_len: usize,
    ) -> Result<()> {
        self.record_op_profile_stage(profile, "get_rows", "embed:get_rows_scale", |enc| {
            self.record_embedding(enc, ids_buf, rows, &workspace.embed_raw, &workspace.state_a)
        })?;

        let mut xs: &Buffer = &workspace.state_a;
        let mut attn_norm_ready = false;
        for layer_idx in 0..self.cfg.num_hidden_layers {
            let mask = match self.cfg.layer_types[layer_idx] {
                LayerType::FullAttention => full_mask,
                LayerType::SlidingAttention => sliding_mask,
            };
            let out = if layer_idx % 2 == 0 {
                &workspace.state_b
            } else {
                &workspace.state_a
            };
            let prefix = format!("layer_{layer_idx:02}");

            if !attn_norm_ready {
                self.record_op_profile_stage(
                    profile,
                    "rms_norm",
                    format!("{prefix}:attn_norm"),
                    |enc| self.record_layer_attn_norm(layer_idx, enc, xs, &workspace.layer, rows),
                )?;
            }
            self.record_op_profile_stage(profile, "mul_mm", format!("{prefix}:qkv"), |enc| {
                self.record_layer_qkv_matmuls(layer_idx, enc, &workspace.layer, rows)
            })?;
            self.record_op_profile_stage(
                profile,
                "rms_norm_rope",
                format!("{prefix}:qk_norm_rope"),
                |enc| {
                    self.record_layer_qk_norm_ropes(
                        layer_idx,
                        enc,
                        &workspace.layer,
                        batch,
                        seq_len,
                    )
                },
            )?;
            self.record_op_profile_stage(
                profile,
                "flash_attention",
                format!("{prefix}:flash_attn_ext"),
                |enc| {
                    self.record_flash_attn(
                        enc,
                        &workspace.layer.qr,
                        &workspace.layer.kr,
                        &workspace.layer.v,
                        mask,
                        &workspace.layer.attn,
                        batch,
                        seq_len,
                        &workspace.layer,
                    )
                },
            )?;
            self.record_op_profile_stage(profile, "mul_mm", format!("{prefix}:attn_out"), |enc| {
                self.record_layer_attn_out_matmul(layer_idx, enc, &workspace.layer, rows)
            })?;
            self.record_op_profile_stage(
                profile,
                "rms_norm_chain",
                format!("{prefix}:post_attn_add_ffn_norm"),
                |enc| {
                    self.record_layer_post_attention_ffn_norm(
                        layer_idx,
                        enc,
                        xs,
                        &workspace.layer,
                        rows,
                    )
                },
            )?;
            self.record_op_profile_stage(
                profile,
                "mul_mm",
                format!("{prefix}:ffn_gate_up"),
                |enc| self.record_layer_ffn_in_matmuls(layer_idx, enc, &workspace.layer, rows),
            )?;
            self.record_op_profile_stage(
                profile,
                "elementwise",
                format!("{prefix}:geglu"),
                |enc| self.record_layer_geglu(enc, &workspace.layer, rows),
            )?;
            self.record_op_profile_stage(profile, "mul_mm", format!("{prefix}:ffn_down"), |enc| {
                self.record_layer_ffn_down_matmul(layer_idx, enc, &workspace.layer, rows)
            })?;
            self.record_op_profile_stage(
                profile,
                if layer_idx + 1 < self.cfg.num_hidden_layers {
                    "rms_norm_chain"
                } else {
                    "rms_norm"
                },
                if layer_idx + 1 < self.cfg.num_hidden_layers {
                    format!("{prefix}:post_ffn_add_next_attn_norm")
                } else {
                    format!("{prefix}:post_ffn_norm_add")
                },
                |enc| {
                    if let Some(next_idx) = next_layer(layer_idx, self.cfg.num_hidden_layers) {
                        self.record_layer_post_ffn_next_attn_norm(
                            layer_idx,
                            next_idx,
                            enc,
                            out,
                            &workspace.layer,
                            rows,
                        )
                    } else {
                        self.record_layer_post_ffn_norm_add(
                            layer_idx,
                            enc,
                            out,
                            &workspace.layer,
                            rows,
                        )
                    }
                },
            )?;
            attn_norm_ready = layer_idx + 1 < self.cfg.num_hidden_layers;
            xs = out;
        }

        self.record_op_profile_stage(profile, "rms_norm", "head:output_norm", |enc| {
            self.record_rms_norm_rows(
                enc,
                "output_norm.weight",
                xs,
                &workspace.head_hidden,
                rows,
                self.cfg.hidden_size,
            )
        })?;
        self.record_op_profile_stage(profile, "elementwise", "head:mean_pool", |enc| {
            self.record_mean_pool(
                enc,
                &workspace.head_hidden,
                mask_u32_buf,
                &workspace.pooled,
                batch,
                seq_len,
            )
        })?;
        let head_rows = tensor_aligned_rows(batch);
        self.record_op_profile_stage(profile, "mul_mm", "head:dense_2", |enc| {
            self.record_matmul_f32(
                enc,
                "dense_2.weight",
                &workspace.pooled,
                &workspace.dense2,
                head_rows,
                self.cfg.hidden_size,
            )
        })?;
        self.record_op_profile_stage(profile, "mul_mm", "head:dense_3", |enc| {
            self.record_matmul_f32(
                enc,
                "dense_3.weight",
                &workspace.dense2,
                &workspace.dense3,
                head_rows,
                DENSE2_DIM,
            )
        })?;
        self.record_op_profile_stage_with_blit(
            profile,
            "elementwise",
            "head:l2_norm",
            workspace,
            batch,
            |enc| self.record_l2_norm(enc, &workspace.dense3, &workspace.l2, batch, EMBEDDING_DIM),
        )?;

        profile.op_breakdown.push(MetalOpProfile {
            op_type: "softmax_standalone".to_string(),
            stages: 0,
            cpu_encode_secs: 0.0,
            cpu_submit_wait_secs: 0.0,
            metal_kernel_secs: 0.0,
            metal_gpu_secs: 0.0,
            dispatches: 0,
        });
        profile.op_breakdown.push(MetalOpProfile {
            op_type: "copy_cast".to_string(),
            stages: 0,
            cpu_encode_secs: 0.0,
            cpu_submit_wait_secs: 0.0,
            metal_kernel_secs: 0.0,
            metal_gpu_secs: 0.0,
            dispatches: 0,
        });

        Ok(())
    }

    fn record_profile_stage(
        &self,
        profile: &mut MetalForwardProfile,
        name: impl Into<String>,
        encode: impl FnOnce(&crate::metal::ffi::ComputeEncoder) -> Result<()>,
    ) -> Result<()> {
        let dispatch_start = ffi::dispatch_count_snapshot();
        let encode_t0 = Instant::now();
        let cb = self.command()?;
        let enc = self.compute_encoder(&cb)?;
        encode(&enc)?;
        enc.end();
        finish_profile_stage(cb, profile, name.into(), encode_t0, dispatch_start)
    }

    fn record_profile_stage_with_blit(
        &self,
        profile: &mut MetalForwardProfile,
        name: impl Into<String>,
        workspace: &ForwardWorkspace,
        batch: usize,
        encode: impl FnOnce(&crate::metal::ffi::ComputeEncoder) -> Result<()>,
    ) -> Result<()> {
        let dispatch_start = ffi::dispatch_count_snapshot();
        let encode_t0 = Instant::now();
        let cb = self.command()?;
        let enc = self.compute_encoder(&cb)?;
        encode(&enc)?;
        enc.end();
        self.record_readback_blit(&cb, workspace, batch)?;
        finish_profile_stage(cb, profile, name.into(), encode_t0, dispatch_start)
    }

    fn record_op_profile_stage(
        &self,
        profile: &mut MetalForwardProfile,
        op_type: &'static str,
        name: impl Into<String>,
        encode: impl FnOnce(&crate::metal::ffi::ComputeEncoder) -> Result<()>,
    ) -> Result<()> {
        let dispatch_start = ffi::dispatch_count_snapshot();
        let encode_t0 = Instant::now();
        let cb = self.command()?;
        let enc = self.compute_encoder(&cb)?;
        encode(&enc)?;
        enc.end();
        finish_op_profile_stage(cb, profile, op_type, name.into(), encode_t0, dispatch_start)
    }

    fn record_op_profile_stage_with_blit(
        &self,
        profile: &mut MetalForwardProfile,
        op_type: &'static str,
        name: impl Into<String>,
        workspace: &ForwardWorkspace,
        batch: usize,
        encode: impl FnOnce(&crate::metal::ffi::ComputeEncoder) -> Result<()>,
    ) -> Result<()> {
        let dispatch_start = ffi::dispatch_count_snapshot();
        let encode_t0 = Instant::now();
        let cb = self.command()?;
        let enc = self.compute_encoder(&cb)?;
        encode(&enc)?;
        enc.end();
        self.record_readback_blit(&cb, workspace, batch)?;
        finish_op_profile_stage(cb, profile, op_type, name.into(), encode_t0, dispatch_start)
    }

    fn record_readback_blit(
        &self,
        cb: &CommandBuffer,
        workspace: &ForwardWorkspace,
        batch: usize,
    ) -> Result<()> {
        let blit = cb.blit().ok_or_else(metal_err)?;
        blit.copy_buffer(
            &workspace.l2,
            0,
            &workspace.readback,
            0,
            batch * EMBEDDING_DIM * std::mem::size_of::<f32>(),
        );
        blit.end();
        Ok(())
    }

    fn record_head(
        &self,
        enc: &crate::metal::ffi::ComputeEncoder,
        xs: &Buffer,
        mask_u32: &Buffer,
        workspace: &ForwardWorkspace,
        rows: usize,
        batch: usize,
        seq_len: usize,
    ) -> Result<()> {
        self.record_rms_norm_rows(
            enc,
            "output_norm.weight",
            xs,
            &workspace.head_hidden,
            rows,
            self.cfg.hidden_size,
        )?;
        enc.memory_barrier_buffers();
        self.record_mean_pool(
            enc,
            &workspace.head_hidden,
            mask_u32,
            &workspace.pooled,
            batch,
            seq_len,
        )?;
        enc.memory_barrier_buffers();
        let head_rows = tensor_aligned_rows(batch);
        self.record_matmul_f32(
            enc,
            "dense_2.weight",
            &workspace.pooled,
            &workspace.dense2,
            head_rows,
            self.cfg.hidden_size,
        )?;
        enc.memory_barrier_buffers();
        self.record_matmul_f32(
            enc,
            "dense_3.weight",
            &workspace.dense2,
            &workspace.dense3,
            head_rows,
            DENSE2_DIM,
        )?;
        enc.memory_barrier_buffers();
        self.record_l2_norm(enc, &workspace.dense3, &workspace.l2, batch, EMBEDDING_DIM)
    }

    fn record_embedding(
        &self,
        enc: &crate::metal::ffi::ComputeEncoder,
        ids: &Buffer,
        rows: usize,
        raw: &Buffer,
        dst: &Buffer,
    ) -> Result<()> {
        let token = self.weights.require("token_embd.weight")?;
        ok(ops::op_get_rows(
            enc,
            self.dev,
            op_type(token.dtype)?,
            &token.buffer,
            token.offset,
            ids,
            raw,
            self.cfg.hidden_size as i32,
            token.nb[1],
            token.nb[2],
            token.nb[3],
            rows as i32,
            1,
            1,
            4,
            (rows * 4) as u64,
            (rows * 4) as u64,
            (self.cfg.hidden_size * 4) as u64,
            (rows * self.cfg.hidden_size * 4) as u64,
            (rows * self.cfg.hidden_size * 4) as u64,
        ))?;
        enc.memory_barrier_buffers();
        self.record_scale_f32(
            enc,
            raw,
            dst,
            rows * self.cfg.hidden_size,
            (self.cfg.hidden_size as f32).sqrt(),
        )?;
        enc.memory_barrier_buffers();
        Ok(())
    }

    fn record_layer(
        &self,
        layer_idx: usize,
        enc: &crate::metal::ffi::ComputeEncoder,
        xs: &Buffer,
        out: &Buffer,
        workspace: &LayerWorkspace,
        rows: usize,
        batch: usize,
        seq_len: usize,
        pos: &Buffer,
        mask: Option<&Buffer>,
        attn_norm_ready: bool,
        next_layer_idx: Option<usize>,
    ) -> Result<()> {
        self.record_layer_qkv_rope(
            layer_idx,
            enc,
            xs,
            workspace,
            rows,
            batch,
            seq_len,
            pos,
            attn_norm_ready,
        )?;
        self.record_layer_attention(enc, workspace, mask, batch, seq_len)?;
        self.record_layer_attention_out(layer_idx, enc, xs, workspace, rows)?;
        self.record_layer_ffn(layer_idx, enc, out, workspace, rows, next_layer_idx)
    }

    fn record_layer_attn_norm(
        &self,
        layer_idx: usize,
        enc: &crate::metal::ffi::ComputeEncoder,
        xs: &Buffer,
        workspace: &LayerWorkspace,
        rows: usize,
    ) -> Result<()> {
        let prefix = format!("blk.{layer_idx}");
        self.record_rms_norm_rows(
            enc,
            &format!("{prefix}.attn_norm.weight"),
            xs,
            &workspace.xs_norm,
            rows,
            self.cfg.hidden_size,
        )
    }

    fn record_layer_qkv_matmuls(
        &self,
        layer_idx: usize,
        enc: &crate::metal::ffi::ComputeEncoder,
        workspace: &LayerWorkspace,
        rows: usize,
    ) -> Result<()> {
        let prefix = format!("blk.{layer_idx}");
        self.record_matmul(
            enc,
            &format!("{prefix}.attn_q.weight"),
            &workspace.xs_norm,
            &workspace.q,
            rows,
            self.cfg.hidden_size,
        )?;
        self.record_matmul(
            enc,
            &format!("{prefix}.attn_k.weight"),
            &workspace.xs_norm,
            &workspace.k,
            rows,
            self.cfg.hidden_size,
        )?;
        self.record_matmul(
            enc,
            &format!("{prefix}.attn_v.weight"),
            &workspace.xs_norm,
            &workspace.v,
            rows,
            self.cfg.hidden_size,
        )?;
        Ok(())
    }

    fn record_layer_qk_norm_ropes(
        &self,
        layer_idx: usize,
        enc: &crate::metal::ffi::ComputeEncoder,
        workspace: &LayerWorkspace,
        batch: usize,
        seq_len: usize,
    ) -> Result<()> {
        let prefix = format!("blk.{layer_idx}");
        let freq_base = match self.cfg.layer_types[layer_idx] {
            LayerType::FullAttention => self.cfg.rope_theta,
            LayerType::SlidingAttention => self.cfg.rope_local_base_freq,
        };
        self.record_rms_norm_rope_heads(
            enc,
            &format!("{prefix}.attn_q_norm.weight"),
            &workspace.q,
            &workspace.qr,
            batch,
            seq_len,
            self.cfg.num_attention_heads,
            self.cfg.hidden_size,
            freq_base,
        )?;
        self.record_rms_norm_rope_heads(
            enc,
            &format!("{prefix}.attn_k_norm.weight"),
            &workspace.k,
            &workspace.kr,
            batch,
            seq_len,
            self.cfg.num_key_value_heads,
            self.cfg.head_dim * self.cfg.num_key_value_heads,
            freq_base,
        )
    }

    fn record_layer_attn_out_matmul(
        &self,
        layer_idx: usize,
        enc: &crate::metal::ffi::ComputeEncoder,
        workspace: &LayerWorkspace,
        rows: usize,
    ) -> Result<()> {
        let prefix = format!("blk.{layer_idx}");
        self.record_matmul(
            enc,
            &format!("{prefix}.attn_output.weight"),
            &workspace.attn,
            &workspace.attn_proj,
            rows,
            self.cfg.hidden_size,
        )
    }

    fn record_layer_post_attention_ffn_norm(
        &self,
        layer_idx: usize,
        enc: &crate::metal::ffi::ComputeEncoder,
        xs: &Buffer,
        workspace: &LayerWorkspace,
        rows: usize,
    ) -> Result<()> {
        let prefix = format!("blk.{layer_idx}");
        let post_w = self
            .weights
            .require(&format!("{prefix}.post_attention_norm.weight"))?;
        let ffn_w = self.weights.require(&format!("{prefix}.ffn_norm.weight"))?;
        let pso = self
            .dev
            .pipeline("embed_native_post_attn_ffn_norm_f32")
            .ok_or_else(metal_err)?;
        let args = PostAttnFfnNormArgs {
            rows: rows as i32,
            dim: self.cfg.hidden_size as i32,
            eps: self.cfg.rms_norm_eps,
            pad: 0,
        };
        enc.set_pipeline(&pso);
        enc.set_bytes(0, &args);
        enc.set_buffer(1, &workspace.attn_proj, 0);
        enc.set_buffer(2, xs, 0);
        enc.set_buffer(3, &post_w.buffer, post_w.offset);
        enc.set_buffer(4, &ffn_w.buffer, ffn_w.offset);
        enc.set_buffer(5, &workspace.sa_out, 0);
        enc.set_buffer(6, &workspace.ffn_norm, 0);
        enc.set_threadgroup_memory_size(256 * std::mem::size_of::<f32>(), 0);
        enc.dispatch_threadgroups((rows, 1, 1), (256, 1, 1));
        Ok(())
    }

    fn record_layer_ffn_in_matmuls(
        &self,
        layer_idx: usize,
        enc: &crate::metal::ffi::ComputeEncoder,
        workspace: &LayerWorkspace,
        rows: usize,
    ) -> Result<()> {
        let prefix = format!("blk.{layer_idx}");
        self.record_matmul(
            enc,
            &format!("{prefix}.ffn_gate.weight"),
            &workspace.ffn_norm,
            &workspace.gate,
            rows,
            self.cfg.hidden_size,
        )?;
        self.record_matmul(
            enc,
            &format!("{prefix}.ffn_up.weight"),
            &workspace.ffn_norm,
            &workspace.up,
            rows,
            self.cfg.hidden_size,
        )?;
        Ok(())
    }

    fn record_layer_geglu(
        &self,
        enc: &crate::metal::ffi::ComputeEncoder,
        workspace: &LayerWorkspace,
        rows: usize,
    ) -> Result<()> {
        self.record_geglu_f32(enc, &workspace.gate, &workspace.up, &workspace.gated, rows)
    }

    fn record_layer_ffn_down_matmul(
        &self,
        layer_idx: usize,
        enc: &crate::metal::ffi::ComputeEncoder,
        workspace: &LayerWorkspace,
        rows: usize,
    ) -> Result<()> {
        let prefix = format!("blk.{layer_idx}");
        self.record_matmul(
            enc,
            &format!("{prefix}.ffn_down.weight"),
            &workspace.gated,
            &workspace.down,
            rows,
            self.cfg.intermediate_size,
        )
    }

    fn record_layer_post_ffn_norm_add(
        &self,
        layer_idx: usize,
        enc: &crate::metal::ffi::ComputeEncoder,
        out: &Buffer,
        workspace: &LayerWorkspace,
        rows: usize,
    ) -> Result<()> {
        let prefix = format!("blk.{layer_idx}");
        self.record_rms_norm_rows_add(
            enc,
            &format!("{prefix}.post_ffw_norm.weight"),
            &workspace.down,
            &workspace.sa_out,
            out,
            rows,
            self.cfg.hidden_size,
        )
    }

    fn record_layer_post_ffn_next_attn_norm(
        &self,
        layer_idx: usize,
        next_layer_idx: usize,
        enc: &crate::metal::ffi::ComputeEncoder,
        out: &Buffer,
        workspace: &LayerWorkspace,
        rows: usize,
    ) -> Result<()> {
        let prefix = format!("blk.{layer_idx}");
        let next_prefix = format!("blk.{next_layer_idx}");
        let post_w = self
            .weights
            .require(&format!("{prefix}.post_ffw_norm.weight"))?;
        let next_w = self
            .weights
            .require(&format!("{next_prefix}.attn_norm.weight"))?;
        let pso = self
            .dev
            .pipeline("embed_native_post_ffn_next_attn_norm_f32")
            .ok_or_else(metal_err)?;
        let args = PostAttnFfnNormArgs {
            rows: rows as i32,
            dim: self.cfg.hidden_size as i32,
            eps: self.cfg.rms_norm_eps,
            pad: 0,
        };
        enc.set_pipeline(&pso);
        enc.set_bytes(0, &args);
        enc.set_buffer(1, &workspace.down, 0);
        enc.set_buffer(2, &workspace.sa_out, 0);
        enc.set_buffer(3, &post_w.buffer, post_w.offset);
        enc.set_buffer(4, &next_w.buffer, next_w.offset);
        enc.set_buffer(5, out, 0);
        enc.set_buffer(6, &workspace.xs_norm, 0);
        enc.set_threadgroup_memory_size(256 * std::mem::size_of::<f32>(), 0);
        enc.dispatch_threadgroups((rows, 1, 1), (256, 1, 1));
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn record_layer_qkv_rope(
        &self,
        layer_idx: usize,
        enc: &crate::metal::ffi::ComputeEncoder,
        xs: &Buffer,
        workspace: &LayerWorkspace,
        rows: usize,
        batch: usize,
        seq_len: usize,
        pos: &Buffer,
        attn_norm_ready: bool,
    ) -> Result<()> {
        if !attn_norm_ready {
            self.record_layer_attn_norm(layer_idx, enc, xs, workspace, rows)?;
            enc.memory_barrier_buffers();
        }
        self.record_layer_qkv_matmuls(layer_idx, enc, workspace, rows)?;
        enc.memory_barrier_buffers();
        let _ = pos;
        self.record_layer_qk_norm_ropes(layer_idx, enc, workspace, batch, seq_len)?;
        enc.memory_barrier_buffers();
        Ok(())
    }

    fn record_layer_attention(
        &self,
        enc: &crate::metal::ffi::ComputeEncoder,
        workspace: &LayerWorkspace,
        mask: Option<&Buffer>,
        batch: usize,
        seq_len: usize,
    ) -> Result<()> {
        self.record_flash_attn(
            enc,
            &workspace.qr,
            &workspace.kr,
            &workspace.v,
            mask,
            &workspace.attn,
            batch,
            seq_len,
            workspace,
        )?;
        enc.memory_barrier_buffers();
        Ok(())
    }

    fn record_layer_attention_out(
        &self,
        layer_idx: usize,
        enc: &crate::metal::ffi::ComputeEncoder,
        xs: &Buffer,
        workspace: &LayerWorkspace,
        rows: usize,
    ) -> Result<()> {
        self.record_layer_attn_out_matmul(layer_idx, enc, workspace, rows)?;
        enc.memory_barrier_buffers();
        self.record_layer_post_attention_ffn_norm(layer_idx, enc, xs, workspace, rows)?;
        enc.memory_barrier_buffers();
        Ok(())
    }

    fn record_layer_ffn(
        &self,
        layer_idx: usize,
        enc: &crate::metal::ffi::ComputeEncoder,
        out: &Buffer,
        workspace: &LayerWorkspace,
        rows: usize,
        next_layer_idx: Option<usize>,
    ) -> Result<()> {
        self.record_layer_ffn_in_matmuls(layer_idx, enc, workspace, rows)?;
        enc.memory_barrier_buffers();
        self.record_layer_geglu(enc, workspace, rows)?;
        enc.memory_barrier_buffers();
        self.record_layer_ffn_down_matmul(layer_idx, enc, workspace, rows)?;
        enc.memory_barrier_buffers();
        if let Some(next_idx) = next_layer_idx {
            self.record_layer_post_ffn_next_attn_norm(
                layer_idx, next_idx, enc, out, workspace, rows,
            )?;
        } else {
            self.record_layer_post_ffn_norm_add(layer_idx, enc, out, workspace, rows)?;
        }
        enc.memory_barrier_buffers();
        Ok(())
    }

    fn record_matmul(
        &self,
        enc: &crate::metal::ffi::ComputeEncoder,
        name: &str,
        src: &Buffer,
        dst: &Buffer,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        self.record_matmul_with_src_type(enc, name, src, dst, rows, cols, OpType::F32, 4, true)
    }

    fn record_matmul_f32(
        &self,
        enc: &crate::metal::ffi::ComputeEncoder,
        name: &str,
        src: &Buffer,
        dst: &Buffer,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        self.record_matmul_with_src_type(enc, name, src, dst, rows, cols, OpType::F32, 4, true)
    }

    #[allow(clippy::too_many_arguments)]
    fn record_matmul_with_src_type(
        &self,
        enc: &crate::metal::ffi::ComputeEncoder,
        name: &str,
        src: &Buffer,
        dst: &Buffer,
        rows: usize,
        cols: usize,
        src_type: OpType,
        src_elem_size: usize,
        force_mm: bool,
    ) -> Result<()> {
        let w = self.weights.require(name)?;
        let w_cols = usize::try_from(w.ne[0]).map_err(|_| {
            Error::InvalidGguf(format!(
                "{name} has negative or too-large column count {}",
                w.ne[0]
            ))
        })?;
        let w_rows = i32::try_from(w.ne[1]).map_err(|_| {
            Error::InvalidGguf(format!("{name} row count {} does not fit i32", w.ne[1]))
        })?;
        let cols_i32 = i32::try_from(cols)
            .map_err(|_| Error::InvalidGguf(format!("{name} cols {cols} do not fit i32")))?;
        let rows_i32 = i32::try_from(rows)
            .map_err(|_| Error::InvalidGguf(format!("{name} rows {rows} do not fit i32")))?;
        let src_row_bytes = cols.checked_mul(src_elem_size).ok_or_else(|| {
            Error::InvalidGguf(format!("{name} source row byte stride overflows"))
        })?;
        let src_total_bytes = rows
            .checked_mul(src_row_bytes)
            .ok_or_else(|| Error::InvalidGguf(format!("{name} source byte size overflows")))?;
        if w_cols != cols {
            return Err(Error::InvalidGguf(format!(
                "{name} expected input cols {}, got {cols}",
                w.ne[0]
            )));
        }
        if rows <= 8 && !force_mm {
            return ok(ops::op_mul_mv(
                enc,
                self.dev,
                op_type(w.dtype)?,
                src_type,
                &w.buffer,
                w.offset,
                src,
                dst,
                cols_i32,
                w_rows,
                1,
                1,
                w.nb[0],
                w.nb[1],
                w.nb[2],
                w.nb[3],
                cols_i32,
                rows_i32,
                1,
                1,
                src_elem_size as u64,
                src_row_bytes as u64,
                src_total_bytes as u64,
                src_total_bytes as u64,
                w_rows,
                rows_i32,
            ));
        }
        ok(ops::op_mul_mm(
            enc,
            self.dev,
            op_type(w.dtype)?,
            src_type,
            &w.buffer,
            w.offset,
            src,
            dst,
            cols_i32,
            w_rows,
            1,
            1,
            w.nb[1],
            w.nb[2],
            w.nb[3],
            rows_i32,
            1,
            1,
            src_elem_size as u64,
            src_row_bytes as u64,
            src_total_bytes as u64,
            src_total_bytes as u64,
            w_rows,
            rows_i32,
        ))
    }

    fn record_rms_norm_rows(
        &self,
        enc: &crate::metal::ffi::ComputeEncoder,
        name: &str,
        src: &Buffer,
        dst: &Buffer,
        rows: usize,
        dim: usize,
    ) -> Result<()> {
        let w = self.weights.require(name)?;
        ok(ops::op_rms_norm_mul(
            enc,
            self.dev,
            OpType::F32,
            src,
            &w.buffer,
            w.offset,
            dst,
            self.cfg.rms_norm_eps,
            dim as i32,
            rows as i32,
            1,
            1,
            (dim * 4) as u64,
            (rows * dim * 4) as u64,
            (rows * dim * 4) as u64,
            (dim * 4) as u64,
            (rows * dim * 4) as u64,
            (rows * dim * 4) as u64,
            1,
            1,
            1,
            (dim * 4) as u64,
            (dim * 4) as u64,
            (dim * 4) as u64,
        ))
    }

    fn record_rms_norm_rows_add(
        &self,
        enc: &crate::metal::ffi::ComputeEncoder,
        name: &str,
        src: &Buffer,
        add: &Buffer,
        dst: &Buffer,
        rows: usize,
        dim: usize,
    ) -> Result<()> {
        let w = self.weights.require(name)?;
        ok(ops::op_rms_norm_mul_add(
            enc,
            self.dev,
            OpType::F32,
            src,
            &w.buffer,
            w.offset,
            add,
            dst,
            self.cfg.rms_norm_eps,
            dim as i32,
            rows as i32,
            1,
            1,
            (dim * 4) as u64,
            (rows * dim * 4) as u64,
            (rows * dim * 4) as u64,
            (dim * 4) as u64,
            (rows * dim * 4) as u64,
            (rows * dim * 4) as u64,
            1,
            1,
            1,
            (dim * 4) as u64,
            (dim * 4) as u64,
            (dim * 4) as u64,
            rows as i32,
            1,
            1,
            (dim * 4) as u64,
            (rows * dim * 4) as u64,
            (rows * dim * 4) as u64,
        ))
    }

    fn record_rms_norm_rope_heads(
        &self,
        enc: &crate::metal::ffi::ComputeEncoder,
        name: &str,
        src: &Buffer,
        dst: &Buffer,
        batch: usize,
        seq_len: usize,
        heads: usize,
        row_width: usize,
        freq_base: f32,
    ) -> Result<()> {
        let w = self.weights.require(name)?;
        let pso = self
            .dev
            .pipeline("embed_native_rms_norm_rope_neox_f32")
            .ok_or_else(metal_err)?;
        let args = RmsNormRopeArgs {
            batch: batch as i32,
            seq_len: seq_len as i32,
            heads: heads as i32,
            head_dim: self.cfg.head_dim as i32,
            row_width: row_width as i32,
            eps: self.cfg.rms_norm_eps,
            freq_base,
            pad: 0,
        };
        enc.set_pipeline(&pso);
        enc.set_bytes(0, &args);
        enc.set_buffer(1, src, 0);
        enc.set_buffer(2, &w.buffer, w.offset);
        enc.set_buffer(3, dst, 0);
        enc.set_threadgroup_memory_size(256 * std::mem::size_of::<f32>(), 0);
        enc.dispatch_threadgroups((seq_len, heads, batch), (256, 1, 1));
        Ok(())
    }

    fn record_flash_attn(
        &self,
        enc: &crate::metal::ffi::ComputeEncoder,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        mask: Option<&Buffer>,
        dst: &Buffer,
        batch: usize,
        seq_len: usize,
        workspace: &LayerWorkspace,
    ) -> Result<()> {
        let kv_width = self.cfg.head_dim * self.cfg.num_key_value_heads;
        let (pad_bytes, blk_bytes, tmp_bytes) =
            self.flash_scratch_sizes(batch, seq_len, mask.is_some(), 4, false);
        if workspace.flash_pad.len() < pad_bytes
            || workspace.flash_blk.len() < blk_bytes
            || workspace.flash_tmp.len() < tmp_bytes
        {
            return Err(Error::InvalidGguf(format!(
                "Metal flash scratch too small: pad {}/{pad_bytes}, blk {}/{blk_bytes}, tmp {}/{tmp_bytes}",
                workspace.flash_pad.len(),
                workspace.flash_blk.len(),
                workspace.flash_tmp.len()
            )));
        }
        ok(ops::op_flash_attn_ext(
            enc,
            self.dev,
            OpType::F32,
            q,
            k,
            v,
            mask,
            None,
            &workspace.flash_pad,
            &workspace.flash_blk,
            &workspace.flash_tmp,
            dst,
            self.cfg.head_dim as i32,
            seq_len as i32,
            self.cfg.num_attention_heads as i32,
            batch as i32,
            (self.cfg.hidden_size * 4) as u64,
            (self.cfg.head_dim * 4) as u64,
            (seq_len * self.cfg.hidden_size * 4) as u64,
            seq_len as i32,
            self.cfg.num_key_value_heads as i32,
            batch as i32,
            4,
            (kv_width * 4) as u64,
            (self.cfg.head_dim * 4) as u64,
            (seq_len * kv_width * 4) as u64,
            self.cfg.head_dim as i32,
            4,
            (kv_width * 4) as u64,
            (self.cfg.head_dim * 4) as u64,
            (seq_len * kv_width * 4) as u64,
            seq_len as i32,
            seq_len as i32,
            1,
            batch as i32,
            (seq_len * 2) as u64,
            (seq_len * seq_len * 2) as u64,
            (seq_len * seq_len * 2) as u64,
            self.cfg.num_attention_heads as i32,
            seq_len as i32,
            batch as i32,
            self.cfg.query_pre_attn_scalar.powf(-0.5),
            0.0,
            0.0,
        ))
    }

    fn record_mean_pool(
        &self,
        enc: &crate::metal::ffi::ComputeEncoder,
        hidden: &Buffer,
        mask: &Buffer,
        dst: &Buffer,
        batch: usize,
        seq_len: usize,
    ) -> Result<()> {
        let pso = self
            .dev
            .pipeline("embed_native_mean_pool_f32")
            .ok_or_else(metal_err)?;
        let args = MeanPoolArgs {
            batch: batch as i32,
            seq_len: seq_len as i32,
            hidden: self.cfg.hidden_size as i32,
        };
        enc.set_pipeline(&pso);
        enc.set_bytes(0, &args);
        enc.set_buffer(1, hidden, 0);
        enc.set_buffer(2, mask, 0);
        enc.set_buffer(3, dst, 0);
        enc.dispatch(
            (self.cfg.hidden_size, batch, 1),
            (256.min(self.cfg.hidden_size), 1, 1),
        );
        Ok(())
    }

    fn record_l2_norm(
        &self,
        enc: &crate::metal::ffi::ComputeEncoder,
        src: &Buffer,
        dst: &Buffer,
        rows: usize,
        dim: usize,
    ) -> Result<()> {
        ok(ops::op_l2_norm(
            enc,
            self.dev,
            OpType::F32,
            src,
            dst,
            1.0e-12,
            dim as i32,
            rows as i32,
            1,
            1,
            4,
            (dim * 4) as u64,
            (rows * dim * 4) as u64,
            (rows * dim * 4) as u64,
            dim as i32,
            rows as i32,
            1,
            1,
            4,
            (dim * 4) as u64,
            (rows * dim * 4) as u64,
            (rows * dim * 4) as u64,
        ))?;
        Ok(())
    }

    fn record_scale_f32(
        &self,
        enc: &crate::metal::ffi::ComputeEncoder,
        src: &Buffer,
        dst: &Buffer,
        elems: usize,
        scale: f32,
    ) -> Result<()> {
        let pso = self
            .dev
            .pipeline("embed_native_scale_f32")
            .ok_or_else(metal_err)?;
        let args = ScaleArgs {
            n: elems as i32,
            scale,
        };
        enc.set_pipeline(&pso);
        enc.set_bytes(0, &args);
        enc.set_buffer(1, src, 0);
        enc.set_buffer(2, dst, 0);
        enc.dispatch((elems, 1, 1), (256, 1, 1));
        Ok(())
    }

    fn record_geglu_f32(
        &self,
        enc: &crate::metal::ffi::ComputeEncoder,
        gate: &Buffer,
        up: &Buffer,
        dst: &Buffer,
        rows: usize,
    ) -> Result<()> {
        ok(ops::op_glu(
            enc,
            self.dev,
            ops::GgmlGluOp::Geglu,
            gate,
            up,
            dst,
            self.cfg.intermediate_size as i32,
            (self.cfg.intermediate_size * 4) as u64,
            self.cfg.intermediate_size as i32,
            (self.cfg.intermediate_size * 4) as u64,
            self.cfg.intermediate_size as i32,
            (self.cfg.intermediate_size * 4) as u64,
            rows as i32,
        ))
    }

    fn command(&self) -> Result<crate::metal::ffi::CommandBuffer> {
        self.dev.new_command_buffer().ok_or_else(metal_err)
    }

    fn compute_encoder(
        &self,
        cb: &crate::metal::ffi::CommandBuffer,
    ) -> Result<crate::metal::ffi::ComputeEncoder> {
        if std::env::var_os("EMBED_NATIVE_METAL_SERIAL_ENCODER").is_some() {
            cb.compute().ok_or_else(metal_err)
        } else {
            cb.compute_concurrent().ok_or_else(metal_err)
        }
    }

    fn new_f32(&self, elems: usize) -> Result<Buffer> {
        self.new_bytes(elems * std::mem::size_of::<f32>())
    }

    fn new_f32_private(&self, elems: usize) -> Result<Buffer> {
        self.new_private_bytes(elems * std::mem::size_of::<f32>())
    }

    fn new_bytes(&self, bytes: usize) -> Result<Buffer> {
        self.dev.new_buffer(bytes).ok_or_else(|| {
            Error::InvalidGguf(format!("failed to allocate Metal buffer ({bytes} bytes)"))
        })
    }

    fn new_private_bytes(&self, bytes: usize) -> Result<Buffer> {
        self.dev.new_private_buffer(bytes).ok_or_else(|| {
            Error::InvalidGguf(format!(
                "failed to allocate private Metal buffer ({bytes} bytes)"
            ))
        })
    }

    fn flash_scratch_sizes(
        &self,
        batch: usize,
        seq_len: usize,
        mask_present: bool,
        elem_size: usize,
        allow_vec: bool,
    ) -> (usize, usize, usize) {
        let use_vec =
            allow_vec && ops::flash_attn_ext_use_vec(self.cfg.head_dim as i32, seq_len as i32);
        let ncpsg = if use_vec {
            ops::OP_FLASH_ATTN_EXT_VEC_NCPSG as usize
        } else {
            ops::OP_FLASH_ATTN_EXT_NCPSG as usize
        };
        let kv_width = self.cfg.head_dim * self.cfg.num_key_value_heads;
        let has_kvpad = seq_len % ncpsg != 0;
        let pad_bytes = if has_kvpad {
            let k_bytes = kv_width * elem_size * ncpsg * self.cfg.num_key_value_heads * batch;
            let v_bytes = kv_width * elem_size * ncpsg * self.cfg.num_key_value_heads * batch;
            let mask_bytes = if mask_present {
                2 * ncpsg * seq_len * batch
            } else {
                0
            };
            k_bytes + v_bytes + mask_bytes
        } else {
            4
        };
        let nblk0 = seq_len.div_ceil(ops::OP_FLASH_ATTN_EXT_NCPSG as usize);
        let nblk1 = seq_len.div_ceil(ops::OP_FLASH_ATTN_EXT_NQPSG as usize);
        let blk_bytes = if mask_present && !use_vec {
            nblk0 * nblk1 * batch
        } else {
            4
        };
        let tmp_bytes = if use_vec {
            seq_len * self.cfg.num_attention_heads * batch * 32 * (self.cfg.head_dim + 2) * 4
        } else {
            4
        };
        (pad_bytes, blk_bytes, tmp_bytes)
    }
}

impl ForwardWorkspace {
    fn new(model: &MetalEmbeddingModel, rows: usize, batch: usize, seq_len: usize) -> Result<Self> {
        let head_rows = tensor_aligned_rows(batch);
        Ok(Self {
            embed_raw: model.new_f32_private(rows * model.cfg.hidden_size)?,
            state_a: model.new_f32_private(rows * model.cfg.hidden_size)?,
            state_b: model.new_f32_private(rows * model.cfg.hidden_size)?,
            layer: LayerWorkspace::new(model, rows, batch, seq_len)?,
            head_hidden: model.new_f32_private(rows * model.cfg.hidden_size)?,
            pooled: model.new_f32_private(head_rows * model.cfg.hidden_size)?,
            dense2: model.new_f32_private(head_rows * DENSE2_DIM)?,
            dense3: model.new_f32_private(head_rows * EMBEDDING_DIM)?,
            l2: model.new_f32_private(batch * EMBEDDING_DIM)?,
            readback: model.new_f32(batch * EMBEDDING_DIM)?,
        })
    }
}

impl LayerWorkspace {
    fn new(model: &MetalEmbeddingModel, rows: usize, batch: usize, seq_len: usize) -> Result<Self> {
        let hidden = rows * model.cfg.hidden_size;
        let kv = rows * model.cfg.head_dim * model.cfg.num_key_value_heads;
        let intermediate = rows * model.cfg.intermediate_size;
        let (flash_pad, flash_blk, flash_tmp) =
            model.flash_scratch_sizes(batch, seq_len, true, 4, false);
        Ok(Self {
            xs_norm: model.new_f32_private(hidden)?,
            q: model.new_f32_private(hidden)?,
            k: model.new_f32_private(kv)?,
            v: model.new_f32_private(kv)?,
            qr: model.new_f32_private(hidden)?,
            kr: model.new_f32_private(kv)?,
            attn: model.new_f32_private(hidden)?,
            attn_proj: model.new_f32_private(hidden)?,
            sa_out: model.new_f32_private(hidden)?,
            ffn_norm: model.new_f32_private(hidden)?,
            gate: model.new_f32_private(intermediate)?,
            up: model.new_f32_private(intermediate)?,
            gated: model.new_f32_private(intermediate)?,
            down: model.new_f32_private(hidden)?,
            flash_pad: model.new_private_bytes(flash_pad)?,
            flash_blk: model.new_private_bytes(flash_blk)?,
            flash_tmp: model.new_private_bytes(flash_tmp)?,
        })
    }
}

impl ShapeCacheEntry {
    fn new(
        model: &MetalEmbeddingModel,
        batch: usize,
        input_seq_len: usize,
        seq_len: usize,
    ) -> Result<Self> {
        let rows = batch * seq_len;
        let mask_elems = batch * seq_len * seq_len;
        let ids_buf = model.new_bytes(rows * std::mem::size_of::<u32>())?;
        let mask_u32_buf = model.new_bytes(rows * std::mem::size_of::<u32>())?;
        let pos_buf = model.new_bytes(seq_len * std::mem::size_of::<i32>())?;
        let full_mask_buf = model.new_bytes(mask_elems * std::mem::size_of::<u16>())?;
        let sliding_mask_buf = model.new_bytes(mask_elems * std::mem::size_of::<u16>())?;
        let pos = (0..seq_len as i32).collect::<Vec<_>>();
        unsafe {
            pos_buf.write(0, &pos);
        }
        Ok(Self {
            batch,
            input_seq_len,
            seq_len,
            ids_buf,
            mask_u32_buf,
            pos_buf,
            full_mask_buf,
            sliding_mask_buf,
            workspace: ForwardWorkspace::new(model, rows, batch, seq_len)?,
            flat_ids: vec![0; rows],
            flat_mask: vec![0; rows],
            last_mask: vec![0; rows],
            full_mask_bits: vec![0; mask_elems],
            sliding_mask_bits: vec![0; mask_elems],
            mask_cache_valid: false,
            full_mask_present: false,
            sliding_mask_present: false,
        })
    }

    fn upload_inputs(
        &mut self,
        token_ids: &[Vec<u32>],
        attention_mask: &[Vec<u32>],
        sliding_window: usize,
    ) {
        for batch_idx in 0..self.batch {
            let ids_row = &token_ids[batch_idx];
            let mask_row = &attention_mask[batch_idx];
            let dst_base = batch_idx * self.seq_len;
            for pos in 0..self.seq_len {
                let dst = dst_base + pos;
                if pos < self.input_seq_len {
                    self.flat_ids[dst] = ids_row[pos];
                    self.flat_mask[dst] = mask_row[pos];
                } else {
                    self.flat_ids[dst] = 0;
                    self.flat_mask[dst] = 0;
                }
            }
        }

        unsafe {
            self.ids_buf.write(0, &self.flat_ids);
            self.mask_u32_buf.write(0, &self.flat_mask);
            if !self.mask_cache_valid || self.flat_mask != self.last_mask {
                self.full_mask_present = fill_attention_mask_bits(
                    &self.flat_mask,
                    self.batch,
                    self.seq_len,
                    None,
                    &mut self.full_mask_bits,
                );
                self.sliding_mask_present = fill_attention_mask_bits(
                    &self.flat_mask,
                    self.batch,
                    self.seq_len,
                    Some(sliding_window),
                    &mut self.sliding_mask_bits,
                );
                if self.full_mask_present {
                    self.full_mask_buf.write(0, &self.full_mask_bits);
                }
                if self.sliding_mask_present {
                    self.sliding_mask_buf.write(0, &self.sliding_mask_bits);
                }
                self.last_mask.copy_from_slice(&self.flat_mask);
                self.mask_cache_valid = true;
            }
        }
    }

    fn full_mask(&self) -> Option<&Buffer> {
        self.full_mask_present.then_some(&self.full_mask_buf)
    }

    fn sliding_mask(&self) -> Option<&Buffer> {
        self.sliding_mask_present.then_some(&self.sliding_mask_buf)
    }
}

fn finish_command(
    cb: CommandBuffer,
    profile: &mut Option<MetalForwardProfile>,
    encode_t0: Instant,
) -> Result<()> {
    if let Some(p) = profile.as_mut() {
        p.cpu_encode_secs += encode_t0.elapsed().as_secs_f64();
        let timing = cb.commit_and_wait_timed().map_err(Error::InvalidGguf)?;
        p.add_command_timing(timing);
    } else {
        cb.commit_and_wait().map_err(Error::InvalidGguf)?;
    }
    Ok(())
}

fn finish_profile_stage(
    cb: CommandBuffer,
    profile: &mut MetalForwardProfile,
    name: String,
    encode_t0: Instant,
    dispatch_start: u64,
) -> Result<()> {
    let cpu_encode_secs = encode_t0.elapsed().as_secs_f64();
    let timing = cb.commit_and_wait_timed().map_err(Error::InvalidGguf)?;
    let dispatches = ffi::dispatch_count_snapshot().saturating_sub(dispatch_start);
    profile.cpu_encode_secs += cpu_encode_secs;
    profile.add_command_timing(timing);
    profile.stages.push(MetalProfileStage {
        name,
        cpu_encode_secs,
        cpu_submit_wait_secs: timing.submit_wait_secs,
        metal_kernel_secs: timing.kernel_secs,
        metal_gpu_secs: timing.gpu_secs,
        dispatches,
    });
    Ok(())
}

fn finish_op_profile_stage(
    cb: CommandBuffer,
    profile: &mut MetalForwardProfile,
    op_type: &'static str,
    name: String,
    encode_t0: Instant,
    dispatch_start: u64,
) -> Result<()> {
    let cpu_encode_secs = encode_t0.elapsed().as_secs_f64();
    let timing = cb.commit_and_wait_timed().map_err(Error::InvalidGguf)?;
    let dispatches = ffi::dispatch_count_snapshot().saturating_sub(dispatch_start);
    profile.cpu_encode_secs += cpu_encode_secs;
    profile.add_command_timing(timing);
    profile.add_op_timing(op_type, timing, dispatches);
    profile.add_op_encode(op_type, cpu_encode_secs);
    profile.stages.push(MetalProfileStage {
        name: format!("{op_type}:{name}"),
        cpu_encode_secs,
        cpu_submit_wait_secs: timing.submit_wait_secs,
        metal_kernel_secs: timing.kernel_secs,
        metal_gpu_secs: timing.gpu_secs,
        dispatches,
    });
    Ok(())
}

fn wait_submitted(cb: &mut Option<SubmittedCommandBuffer>) -> Result<()> {
    if let Some(cb) = cb.take() {
        cb.wait().map_err(Error::InvalidGguf)?;
    }
    Ok(())
}

fn stage_profile_enabled() -> bool {
    std::env::var_os("EMBED_NATIVE_METAL_STAGE_PROFILE").is_some()
}

fn op_profile_enabled() -> bool {
    std::env::var_os("EMBED_NATIVE_METAL_OP_PROFILE").is_some()
}

fn next_layer(layer_idx: usize, num_layers: usize) -> Option<usize> {
    (layer_idx + 1 < num_layers).then_some(layer_idx + 1)
}

impl MetalConfig {
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
        let rms_norm_eps =
            model.metadata_f32("gemma-embedding.attention.layer_norm_rms_epsilon")?;
        let rope_theta = model.metadata_f32("gemma-embedding.rope.freq_base")?;
        let rope_local_base_freq = metadata_f32_opt(model, "gemma-embedding.rope.local_freq_base")
            .unwrap_or(DEFAULT_GGUF_LOCAL_ROPE_FREQ as f32);
        let sliding_window =
            model.metadata_u32("gemma-embedding.attention.sliding_window")? as usize;
        if intermediate_size == 0 || sliding_window == 0 {
            return Err(Error::InvalidGguf(format!(
                "invalid Gemma sizes: intermediate={intermediate_size}, sliding_window={sliding_window}"
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

fn op_type(dtype: GgmlType) -> Result<OpType> {
    Ok(match dtype {
        GgmlType::F32 => OpType::F32,
        GgmlType::F16 => OpType::F16,
        GgmlType::Q5_0 => OpType::Q5_0,
        GgmlType::Q8_0 => OpType::Q8_0,
        GgmlType::Bf16 => OpType::Bf16,
        GgmlType::Q4_K => OpType::Q4_K,
        GgmlType::Q5_K => OpType::Q5_K,
        GgmlType::Q6_K => OpType::Q6_K,
        other => {
            return Err(Error::InvalidGguf(format!(
                "unsupported Metal op dtype {:?}",
                other
            )))
        }
    })
}

fn ok(value: bool) -> Result<()> {
    if value {
        Ok(())
    } else {
        Err(metal_err())
    }
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

fn metal_err() -> Error {
    Error::InvalidGguf(crate::metal::ops::last_error_str())
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
            "token batch shape {}x{seq_len} exceeds Metal i32 dispatch limits",
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

fn tensor_aligned_rows(rows: usize) -> usize {
    rows.div_ceil(TENSOR_MUL_MM_NRB) * TENSOR_MUL_MM_NRB
}

fn tensor_aligned_seq_len(batch: usize, seq_len: usize) -> Result<usize> {
    let mut aligned = seq_len;
    loop {
        let rows = batch
            .checked_mul(aligned)
            .ok_or_else(|| Error::InvalidGguf("Metal aligned row count overflows".into()))?;
        if rows == tensor_aligned_rows(rows) {
            break;
        }
        aligned += 1;
    }
    Ok(aligned)
}

fn fill_attention_mask_bits(
    flat_mask: &[u32],
    batch: usize,
    seq_len: usize,
    sliding_window: Option<usize>,
    dst: &mut [u16],
) -> bool {
    let Some(flat_len) = batch.checked_mul(seq_len) else {
        return true;
    };
    let Some(dst_len) = flat_len.checked_mul(seq_len) else {
        return true;
    };
    if flat_mask.len() != flat_len || dst.len() != dst_len {
        return true;
    }
    let mut any_masked = false;
    let masked = f16::from_f32(-1.0e9).to_bits();
    let mut out = 0;
    for batch_idx in 0..batch {
        let row = &flat_mask[batch_idx * seq_len..(batch_idx + 1) * seq_len];
        for q in 0..seq_len {
            for k in 0..seq_len {
                let key_visible = row[k] != 0;
                let window_visible = sliding_window
                    .map(|window| q.abs_diff(k) < window)
                    .unwrap_or(true);
                dst[out] = if key_visible && window_visible {
                    0
                } else {
                    any_masked = true;
                    masked
                };
                out += 1;
            }
        }
    }
    any_masked
}

fn metadata_u32_opt(model: &GgufModel, key: &str) -> Option<u32> {
    model.metadata().get(key).and_then(Value::as_u32)
}

fn metadata_f32_opt(model: &GgufModel, key: &str) -> Option<f32> {
    model.metadata().get(key).and_then(Value::as_f32)
}
