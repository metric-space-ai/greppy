//! CUDA-resident Qwen3.5 forward building blocks.
//!
//! This module owns Qwen3.5 CUDA weight residency, device-resident decode
//! workspace/state, and the logits path used by the production summarizer.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use greppy_embed_native::cuda::ffi::{
    check, gp_embed_q4k, gp_embed_q6k, gp_mmvq_matvec, gp_qwen_add, gp_qwen_add_rms_norm,
    gp_qwen_apply_sigmoid_gate, gp_qwen_apply_silu_gate, gp_qwen_argmax,
    gp_qwen_attention_scores_decode, gp_qwen_attention_values_decode, gp_qwen_cache_write,
    gp_qwen_causal_conv1d_silu, gp_qwen_deinterleave_q_gate, gp_qwen_deltanet_decode,
    gp_qwen_normalize_linear_qk, gp_qwen_rms_norm, gp_qwen_rope_decode, gp_qwen_softmax_decode,
    gp_qwen_swiglu, gp_rms_norm, CudaDevice, DeviceBuffer,
};
use greppy_embed_native::cuda::weights::CudaWeights;
use greppy_embed_native::{GgmlDType, GgufModel};
use tokenizers::Tokenizer;

use crate::inventory::Qwen35Inventory;
use crate::sampler::{sample_token, GenerationParams, SamplerRng};
use crate::{Error, Result};

pub struct CudaQwen35Model {
    dev: CudaDevice,
    weights: CudaWeights,
    inventory: Qwen35Inventory,
    eos_token_id: u32,
}

pub struct CudaForwardState {
    position: usize,
    max_context: usize,
    layer_states: Vec<CudaLayerState>,
}

enum CudaLayerState {
    Delta {
        recurrent: DeviceBuffer,
        conv: DeviceBuffer,
    },
    Full {
        k_cache: DeviceBuffer,
        v_cache: DeviceBuffer,
    },
}

pub(crate) struct CudaForwardWorkspace {
    token_id: DeviceBuffer,
    hidden: DeviceBuffer,
    normed: DeviceBuffer,
    attn_out: DeviceBuffer,
    qkv: DeviceBuffer,
    z: DeviceBuffer,
    beta: DeviceBuffer,
    alpha: DeviceBuffer,
    raw: DeviceBuffer,
    q_fused: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    ffn_gate: DeviceBuffer,
    ffn_up: DeviceBuffer,
    logits: DeviceBuffer,
    argmax_values: DeviceBuffer,
    argmax_indices: DeviceBuffer,
    scores: DeviceBuffer,
    q8_scratch: DeviceBuffer,
}

impl CudaQwen35Model {
    pub fn from_gguf(
        model: &GgufModel,
        inventory: Qwen35Inventory,
        eos_token_id: u32,
    ) -> Result<Self> {
        let device = std::env::var("GREPPY_QWEN35_CUDA_DEVICE")
            .or_else(|_| std::env::var("EMBED_NATIVE_CUDA_DEVICE"))
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(0);
        let dev = CudaDevice::new(device)?;
        let weights = CudaWeights::load(&dev, model)?;
        Ok(Self {
            dev,
            weights,
            inventory,
            eos_token_id,
        })
    }

    pub fn backend_name(&self) -> &'static str {
        "cuda-q4k-components"
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    pub fn used_vram(&self) -> Result<usize> {
        let (free, total) = self.dev.mem_info()?;
        Ok(total.saturating_sub(free))
    }

    pub fn generate(
        &self,
        tokenizer: &Tokenizer,
        prompt: &str,
        params: GenerationParams,
    ) -> Result<String> {
        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let prompt_ids = encoding.get_ids();
        if prompt_ids.is_empty() {
            return Ok(String::new());
        }
        let max_context = prompt_ids
            .len()
            .saturating_add(params.max_tokens)
            .saturating_add(1)
            .min(self.inventory.context_length);
        let mut state = self.new_forward_state(max_context)?;
        let mut workspace = self.new_forward_workspace(max_context)?;
        for &token in &prompt_ids[..prompt_ids.len().saturating_sub(1)] {
            self.prefill_token(token, &mut state, &mut workspace)?;
        }

        let mut next = *prompt_ids.last().expect("checked non-empty above");
        let mut generated = Vec::new();
        if is_greedy_device_sampling(params) {
            for _ in 0..params.max_tokens {
                let token = self.forward_token_greedy(next, &mut state, &mut workspace)?;
                if token == self.eos_token_id {
                    break;
                }
                generated.push(token);
                next = token;
            }
        } else {
            let mut rng = SamplerRng::new(prompt_seed(prompt));
            for _ in 0..params.max_tokens {
                let mut logits = self.forward_token_logits(next, &mut state, &mut workspace)?;
                let Some(token) = sample_token(&mut logits, &generated, params, &mut rng) else {
                    break;
                };
                if token == self.eos_token_id {
                    break;
                }
                generated.push(token);
                next = token;
            }
        }
        tokenizer
            .decode(&generated, true)
            .map_err(|e| Error::Tokenizer(e.to_string()))
    }

    pub fn new_forward_state(&self, max_context: usize) -> Result<CudaForwardState> {
        let mut layer_states = Vec::with_capacity(self.inventory.block_count);
        for layer in 0..self.inventory.block_count {
            if self.inventory.is_full_attention_layer(layer) {
                let k_elems = max_context * self.inventory.kv_heads * self.inventory.head_dim;
                let v_elems = max_context * self.inventory.kv_heads * self.inventory.value_dim;
                layer_states.push(CudaLayerState::Full {
                    k_cache: self.new_f32(k_elems)?,
                    v_cache: self.new_f32(v_elems)?,
                });
            } else {
                let head_dim = self.inventory.ssm_inner_size / self.inventory.ssm_group_count;
                layer_states.push(CudaLayerState::Delta {
                    recurrent: self
                        .new_f32(self.inventory.ssm_group_count * head_dim * head_dim)?,
                    conv: self.new_f32(self.inventory.ssm_inner_size * 3 * CONV_KERNEL)?,
                });
            }
        }
        Ok(CudaForwardState {
            position: 0,
            max_context,
            layer_states,
        })
    }

    pub(crate) fn new_forward_workspace(&self, max_context: usize) -> Result<CudaForwardWorkspace> {
        let max_matvec_cols = self
            .inventory
            .feed_forward_size
            .max(self.inventory.ssm_inner_size)
            .max(self.inventory.attention_heads * self.inventory.value_dim)
            .max(self.inventory.hidden_size);
        let argmax_blocks = argmax_block_count(self.inventory.vocab_size);
        Ok(CudaForwardWorkspace {
            token_id: self.new_bytes(std::mem::size_of::<u32>())?,
            hidden: self.new_f32(self.inventory.hidden_size)?,
            normed: self.new_f32(self.inventory.hidden_size)?,
            attn_out: self.new_f32(self.inventory.hidden_size)?,
            qkv: self.new_f32(self.inventory.ssm_inner_size * 3)?,
            z: self.new_f32(self.inventory.ssm_inner_size)?,
            beta: self.new_f32(self.inventory.ssm_group_count)?,
            alpha: self.new_f32(self.inventory.ssm_time_step_rank)?,
            raw: self.new_f32(
                self.inventory
                    .ssm_inner_size
                    .max(self.inventory.attention_heads * self.inventory.value_dim),
            )?,
            q_fused: self.new_f32(self.inventory.attention_heads * self.inventory.head_dim * 2)?,
            k: self.new_f32(self.inventory.kv_heads * self.inventory.head_dim)?,
            v: self.new_f32(self.inventory.kv_heads * self.inventory.value_dim)?,
            ffn_gate: self.new_f32(self.inventory.feed_forward_size)?,
            ffn_up: self.new_f32(self.inventory.feed_forward_size)?,
            logits: self.new_f32(self.inventory.vocab_size)?,
            argmax_values: self.new_f32(argmax_blocks)?,
            argmax_indices: self.new_bytes(argmax_blocks * std::mem::size_of::<u32>())?,
            scores: self.new_f32(self.inventory.attention_heads * max_context)?,
            q8_scratch: self.new_bytes(q8_1_scratch_bytes(max_matvec_cols, 1))?,
        })
    }

    pub(crate) fn forward_token_logits(
        &self,
        token: u32,
        state: &mut CudaForwardState,
        ws: &mut CudaForwardWorkspace,
    ) -> Result<Vec<f32>> {
        self.forward_token_logits_device(token, state, ws)?;
        let mut logits = vec![0.0f32; self.inventory.vocab_size];
        self.dev.copy_d2h(&mut logits, &ws.logits)?;
        Ok(logits)
    }

    pub(crate) fn prefill_token(
        &self,
        token: u32,
        state: &mut CudaForwardState,
        ws: &mut CudaForwardWorkspace,
    ) -> Result<()> {
        self.forward_token_prefill_device(token, state, ws)
    }

    pub(crate) fn forward_token_greedy(
        &self,
        token: u32,
        state: &mut CudaForwardState,
        ws: &mut CudaForwardWorkspace,
    ) -> Result<u32> {
        self.forward_token_logits_device(token, state, ws)?;
        check(
            unsafe {
                gp_qwen_argmax(
                    ws.logits.as_f32(),
                    ws.token_id.as_u32(),
                    checked_i32(self.inventory.vocab_size, "argmax vocab size")?,
                    ws.argmax_values.as_f32(),
                    ws.argmax_indices.as_u32(),
                    checked_i32(
                        argmax_block_count(self.inventory.vocab_size),
                        "argmax blocks",
                    )?,
                    self.dev.stream(),
                )
            },
            "qwen35 cuda argmax logits",
        )?;
        let mut token = [0_u32; 1];
        self.dev.copy_d2h(&mut token, &ws.token_id)?;
        Ok(token[0])
    }

    fn forward_token_logits_device(
        &self,
        token: u32,
        state: &mut CudaForwardState,
        ws: &mut CudaForwardWorkspace,
    ) -> Result<()> {
        self.forward_token_hidden_device(token, state, ws)?;
        self.rms_norm_device(
            "output_norm.weight",
            ws.hidden.as_f32(),
            ws.normed.as_f32(),
            1,
            self.inventory.hidden_size,
            true,
        )?;
        self.matvec_device_to(
            "token_embd.weight",
            ws.normed.as_f32(),
            self.inventory.hidden_size,
            ws.logits.as_f32(),
            self.inventory.vocab_size,
            &ws.q8_scratch,
        )?;
        Ok(())
    }

    fn forward_token_hidden_device(
        &self,
        token: u32,
        state: &mut CudaForwardState,
        ws: &mut CudaForwardWorkspace,
    ) -> Result<()> {
        if state.position >= state.max_context {
            return Err(Error::InvalidRequest(format!(
                "qwen35 prompt exceeds local context cap {}",
                state.max_context
            )));
        }
        if token as usize >= self.inventory.vocab_size {
            return Err(Error::InvalidRequest(format!(
                "token id {token} out of range for vocab {}",
                self.inventory.vocab_size
            )));
        }
        self.dev
            .copy_h2d(&ws.token_id, std::slice::from_ref(&token))?;
        self.embed_tokens_device(ws.token_id.as_u32(), ws.hidden.as_f32(), 1)?;

        for layer in 0..self.inventory.block_count {
            self.rms_norm_device(
                &format!("blk.{layer}.attn_norm.weight"),
                ws.hidden.as_f32(),
                ws.normed.as_f32(),
                1,
                self.inventory.hidden_size,
                true,
            )?;
            match &mut state.layer_states[layer] {
                CudaLayerState::Delta { recurrent, conv } => {
                    self.delta_attention_block_device(layer, recurrent, conv, ws)?;
                }
                CudaLayerState::Full { k_cache, v_cache } => {
                    self.full_attention_block_device(layer, k_cache, v_cache, state.position, ws)?;
                }
            }
            self.add_rms_norm_device(
                &format!("blk.{layer}.post_attention_norm.weight"),
                ws.hidden.as_f32(),
                ws.attn_out.as_f32(),
                ws.hidden.as_f32(),
                ws.normed.as_f32(),
                1,
                self.inventory.hidden_size,
            )?;
            self.ffn_block_device(layer, ws)?;
            self.add_device(
                ws.hidden.as_f32(),
                ws.attn_out.as_f32(),
                ws.hidden.as_f32(),
                self.inventory.hidden_size,
            )?;
        }

        state.position += 1;
        Ok(())
    }

    fn forward_token_prefill_device(
        &self,
        token: u32,
        state: &mut CudaForwardState,
        ws: &mut CudaForwardWorkspace,
    ) -> Result<()> {
        if state.position >= state.max_context {
            return Err(Error::InvalidRequest(format!(
                "qwen35 prompt exceeds local context cap {}",
                state.max_context
            )));
        }
        if token as usize >= self.inventory.vocab_size {
            return Err(Error::InvalidRequest(format!(
                "token id {token} out of range for vocab {}",
                self.inventory.vocab_size
            )));
        }
        self.dev
            .copy_h2d(&ws.token_id, std::slice::from_ref(&token))?;
        self.embed_tokens_device(ws.token_id.as_u32(), ws.hidden.as_f32(), 1)?;

        let final_layer = self.inventory.block_count.saturating_sub(1);
        for layer in 0..self.inventory.block_count {
            if layer == final_layer {
                self.rms_norm_device(
                    &format!("blk.{layer}.attn_norm.weight"),
                    ws.hidden.as_f32(),
                    ws.normed.as_f32(),
                    1,
                    self.inventory.hidden_size,
                    true,
                )?;
                match &mut state.layer_states[layer] {
                    CudaLayerState::Full { k_cache, v_cache } => {
                        self.full_attention_cache_only_device(
                            layer,
                            k_cache,
                            v_cache,
                            state.position,
                            ws,
                        )?;
                    }
                    CudaLayerState::Delta { recurrent, conv } => {
                        self.delta_attention_block_device(layer, recurrent, conv, ws)?;
                    }
                }
                state.position += 1;
                return Ok(());
            }

            self.rms_norm_device(
                &format!("blk.{layer}.attn_norm.weight"),
                ws.hidden.as_f32(),
                ws.normed.as_f32(),
                1,
                self.inventory.hidden_size,
                true,
            )?;
            match &mut state.layer_states[layer] {
                CudaLayerState::Delta { recurrent, conv } => {
                    self.delta_attention_block_device(layer, recurrent, conv, ws)?;
                }
                CudaLayerState::Full { k_cache, v_cache } => {
                    self.full_attention_block_device(layer, k_cache, v_cache, state.position, ws)?;
                }
            }
            self.add_rms_norm_device(
                &format!("blk.{layer}.post_attention_norm.weight"),
                ws.hidden.as_f32(),
                ws.attn_out.as_f32(),
                ws.hidden.as_f32(),
                ws.normed.as_f32(),
                1,
                self.inventory.hidden_size,
            )?;
            self.ffn_block_device(layer, ws)?;
            self.add_device(
                ws.hidden.as_f32(),
                ws.attn_out.as_f32(),
                ws.hidden.as_f32(),
                self.inventory.hidden_size,
            )?;
        }

        state.position += 1;
        Ok(())
    }

    pub fn embed_tokens(&self, ids: &[u32]) -> Result<Vec<f32>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        for &id in ids {
            if id as usize >= self.inventory.vocab_size {
                return Err(Error::InvalidRequest(format!(
                    "token id {id} out of range for vocab {}",
                    self.inventory.vocab_size
                )));
            }
        }
        let token = self.weights.require("token_embd.weight")?;
        let rows = token.rows()?;
        let hidden = token.cols()?;
        if rows != self.inventory.vocab_size || hidden != self.inventory.hidden_size {
            return Err(Error::Gguf(format!(
                "token_embd.weight shape [{rows}, {hidden}] does not match inventory [{}, {}]",
                self.inventory.vocab_size, self.inventory.hidden_size
            )));
        }

        let ids_dev = self.upload_pod(ids)?;
        let out_dev = self.new_f32(ids.len() * self.inventory.hidden_size)?;
        match token.dtype {
            GgmlDType::Q4K => check(
                unsafe {
                    gp_embed_q4k(
                        token.buffer.ptr(),
                        ids_dev.as_u32(),
                        out_dev.as_f32(),
                        ids.len() as i32,
                        self.inventory.hidden_size as i32,
                        1.0,
                        self.dev.stream(),
                    )
                },
                "qwen35 cuda embed_q4k token_embd.weight",
            )?,
            GgmlDType::Q6K => check(
                unsafe {
                    gp_embed_q6k(
                        token.buffer.ptr(),
                        ids_dev.as_u32(),
                        out_dev.as_f32(),
                        ids.len() as i32,
                        self.inventory.hidden_size as i32,
                        1.0,
                        self.dev.stream(),
                    )
                },
                "qwen35 cuda embed_q6k token_embd.weight",
            )?,
            other => {
                return Err(Error::GenerationUnavailable(format!(
                    "Qwen3.5 CUDA token embedding expects Q4_K/Q6_K, got {other}"
                )))
            }
        }
        let mut out = vec![0.0f32; ids.len() * self.inventory.hidden_size];
        self.dev.copy_d2h(&mut out, &out_dev)?;
        Ok(out)
    }

    pub fn matvec(&self, tensor_name: &str, input: &[f32]) -> Result<Vec<f32>> {
        let tensor = self.weights.require(tensor_name)?;
        let cols = tensor.cols()?;
        let rows = tensor.rows()?;
        if input.len() != cols {
            return Err(Error::InvalidRequest(format!(
                "{tensor_name} input len {}, expected {cols}",
                input.len()
            )));
        }
        let src = self.upload_pod(input)?;
        let dst = self.new_f32(rows)?;
        let q8_scratch = self.new_bytes(q8_1_scratch_bytes(cols, 1))?;
        check(
            unsafe {
                gp_mmvq_matvec(
                    tensor.ggml_type_id()?,
                    tensor.buffer.ptr(),
                    src.as_f32(),
                    dst.as_f32(),
                    q8_scratch.ptr(),
                    checked_i64(cols, "matvec cols")?,
                    checked_i64(tensor.row_stride_blocks(), "matvec row stride blocks")?,
                    checked_i64(rows, "matvec rows")?,
                    self.dev.stream(),
                )
            },
            &format!("qwen35 cuda MMVQ matvec {tensor_name}"),
        )?;
        let mut out = vec![0.0f32; rows];
        self.dev.copy_d2h(&mut out, &dst)?;
        Ok(out)
    }

    pub fn qwen_rms_norm(&self, tensor_name: &str, input: &[f32], dim: usize) -> Result<Vec<f32>> {
        self.rms_norm(tensor_name, input, dim, true)
    }

    pub fn plain_rms_norm(&self, tensor_name: &str, input: &[f32], dim: usize) -> Result<Vec<f32>> {
        self.rms_norm(tensor_name, input, dim, false)
    }

    fn rms_norm(
        &self,
        tensor_name: &str,
        input: &[f32],
        dim: usize,
        qwen_scale: bool,
    ) -> Result<Vec<f32>> {
        if dim == 0 || input.len() % dim != 0 {
            return Err(Error::InvalidRequest(format!(
                "{tensor_name} RMSNorm input len {} is not divisible by dim {dim}",
                input.len()
            )));
        }
        let weight = self.weights.require(tensor_name)?;
        if weight.dtype != GgmlDType::F32 || weight.shape.as_slice() != &[dim] {
            return Err(Error::Gguf(format!(
                "{tensor_name} RMSNorm weight shape {:?} dtype {}, expected F32 [{dim}]",
                weight.shape, weight.dtype
            )));
        }
        let rows = input.len() / dim;
        let src = self.upload_pod(input)?;
        let dst = self.new_f32(input.len())?;
        let rc = unsafe {
            if qwen_scale {
                gp_qwen_rms_norm(
                    src.as_f32(),
                    weight.buffer.as_f32(),
                    dst.as_f32(),
                    rows as i32,
                    dim as i32,
                    RMS_EPS,
                    self.dev.stream(),
                )
            } else {
                gp_rms_norm(
                    src.as_f32(),
                    weight.buffer.as_f32(),
                    dst.as_f32(),
                    rows as i32,
                    dim as i32,
                    RMS_EPS,
                    self.dev.stream(),
                )
            }
        };
        check(rc, &format!("qwen35 cuda RMSNorm {tensor_name}"))?;
        let mut out = vec![0.0f32; input.len()];
        self.dev.copy_d2h(&mut out, &dst)?;
        Ok(out)
    }

    pub fn causal_conv1d_silu(
        &self,
        tensor_name: &str,
        values: &[f32],
        state: &[f32],
        kernel: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        if values.is_empty() || kernel == 0 || state.len() != values.len() * kernel {
            return Err(Error::InvalidRequest(format!(
                "{tensor_name} conv values len {}, state len {}, kernel {kernel}",
                values.len(),
                state.len()
            )));
        }
        let weight = self.weights.require(tensor_name)?;
        if weight.dtype != GgmlDType::F32 || weight.shape.as_slice() != &[values.len(), kernel] {
            return Err(Error::Gguf(format!(
                "{tensor_name} conv weight shape {:?} dtype {}, expected F32 [{}, {kernel}]",
                weight.shape,
                weight.dtype,
                values.len()
            )));
        }
        let values_dev = self.upload_pod(values)?;
        let state_dev = self.upload_pod(state)?;
        check(
            unsafe {
                gp_qwen_causal_conv1d_silu(
                    values_dev.as_f32(),
                    weight.buffer.as_f32(),
                    state_dev.as_f32(),
                    values.len() as i32,
                    kernel as i32,
                    self.dev.stream(),
                )
            },
            &format!("qwen35 cuda causal_conv1d_silu {tensor_name}"),
        )?;
        let mut out_values = vec![0.0f32; values.len()];
        let mut out_state = vec![0.0f32; state.len()];
        self.dev.copy_d2h(&mut out_values, &values_dev)?;
        self.dev.copy_d2h(&mut out_state, &state_dev)?;
        Ok((out_values, out_state))
    }

    pub fn normalize_linear_qk(
        &self,
        q: &[f32],
        k: &[f32],
        heads: usize,
        head_dim: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let expected = heads.checked_mul(head_dim).ok_or_else(|| {
            Error::InvalidRequest("normalize_linear_qk heads*head_dim overflows".into())
        })?;
        if q.len() != expected || k.len() != expected {
            return Err(Error::InvalidRequest(format!(
                "normalize_linear_qk q len {}, k len {}, expected {expected}",
                q.len(),
                k.len()
            )));
        }
        let q_dev = self.upload_pod(q)?;
        let k_dev = self.upload_pod(k)?;
        check(
            unsafe {
                gp_qwen_normalize_linear_qk(
                    q_dev.as_f32(),
                    k_dev.as_f32(),
                    heads as i32,
                    head_dim as i32,
                    RMS_EPS,
                    self.dev.stream(),
                )
            },
            "qwen35 cuda normalize_linear_qk",
        )?;
        let mut q_out = vec![0.0f32; q.len()];
        let mut k_out = vec![0.0f32; k.len()];
        self.dev.copy_d2h(&mut q_out, &q_dev)?;
        self.dev.copy_d2h(&mut k_out, &k_dev)?;
        Ok((q_out, k_out))
    }

    pub fn swiglu(&self, gate: &[f32], up: &[f32]) -> Result<Vec<f32>> {
        if gate.len() != up.len() {
            return Err(Error::InvalidRequest(format!(
                "swiglu gate len {} != up len {}",
                gate.len(),
                up.len()
            )));
        }
        let gate_dev = self.upload_pod(gate)?;
        let up_dev = self.upload_pod(up)?;
        let dst = self.new_f32(gate.len())?;
        check(
            unsafe {
                gp_qwen_swiglu(
                    gate_dev.as_f32(),
                    up_dev.as_f32(),
                    dst.as_f32(),
                    gate.len() as i32,
                    self.dev.stream(),
                )
            },
            "qwen35 cuda swiglu",
        )?;
        let mut out = vec![0.0f32; gate.len()];
        self.dev.copy_d2h(&mut out, &dst)?;
        Ok(out)
    }

    pub fn apply_silu_gate(&self, values: &[f32], gate: &[f32]) -> Result<Vec<f32>> {
        if values.len() != gate.len() {
            return Err(Error::InvalidRequest(format!(
                "apply_silu_gate values len {} != gate len {}",
                values.len(),
                gate.len()
            )));
        }
        let values_dev = self.upload_pod(values)?;
        let gate_dev = self.upload_pod(gate)?;
        check(
            unsafe {
                gp_qwen_apply_silu_gate(
                    values_dev.as_f32(),
                    gate_dev.as_f32(),
                    values.len() as i32,
                    self.dev.stream(),
                )
            },
            "qwen35 cuda apply_silu_gate",
        )?;
        let mut out = vec![0.0f32; values.len()];
        self.dev.copy_d2h(&mut out, &values_dev)?;
        Ok(out)
    }

    pub fn ffn_block(&self, layer: usize, hidden: &[f32]) -> Result<Vec<f32>> {
        if layer >= self.inventory.block_count {
            return Err(Error::InvalidRequest(format!(
                "ffn layer {layer} out of range for {} layers",
                self.inventory.block_count
            )));
        }
        if hidden.len() != self.inventory.hidden_size {
            return Err(Error::InvalidRequest(format!(
                "ffn hidden len {}, expected {}",
                hidden.len(),
                self.inventory.hidden_size
            )));
        }
        let prefix = format!("blk.{layer}");
        let gate = self.matvec(&format!("{prefix}.ffn_gate.weight"), hidden)?;
        let up = self.matvec(&format!("{prefix}.ffn_up.weight"), hidden)?;
        let activated = self.swiglu(&gate, &up)?;
        self.matvec(&format!("{prefix}.ffn_down.weight"), &activated)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn deltanet_decode(
        &self,
        layer: usize,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        beta: &[f32],
        alpha: &[f32],
        recurrent: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        if self.inventory.is_full_attention_layer(layer) || layer >= self.inventory.block_count {
            return Err(Error::InvalidRequest(format!(
                "layer {layer} is not a DeltaNet layer"
            )));
        }
        let heads = self.inventory.ssm_group_count;
        let head_dim = self.inventory.ssm_inner_size / heads;
        let inner = heads * head_dim;
        if q.len() != inner || k.len() != inner || v.len() != inner {
            return Err(Error::InvalidRequest(format!(
                "DeltaNet q/k/v lens {}/{}/{}, expected {inner}",
                q.len(),
                k.len(),
                v.len()
            )));
        }
        if beta.len() != heads || alpha.len() != heads {
            return Err(Error::InvalidRequest(format!(
                "DeltaNet beta/alpha lens {}/{}, expected {heads}",
                beta.len(),
                alpha.len()
            )));
        }
        if recurrent.len() != heads * head_dim * head_dim {
            return Err(Error::InvalidRequest(format!(
                "DeltaNet recurrent len {}, expected {}",
                recurrent.len(),
                heads * head_dim * head_dim
            )));
        }
        let prefix = format!("blk.{layer}");
        let a_log = self.weights.require(&format!("{prefix}.ssm_a"))?;
        let dt_bias = self.weights.require(&format!("{prefix}.ssm_dt.bias"))?;
        if a_log.dtype != GgmlDType::F32 || a_log.shape.as_slice() != &[heads] {
            return Err(Error::Gguf(format!(
                "{prefix}.ssm_a shape {:?} dtype {}, expected F32 [{heads}]",
                a_log.shape, a_log.dtype
            )));
        }
        if dt_bias.dtype != GgmlDType::F32 || dt_bias.shape.as_slice() != &[heads] {
            return Err(Error::Gguf(format!(
                "{prefix}.ssm_dt.bias shape {:?} dtype {}, expected F32 [{heads}]",
                dt_bias.shape, dt_bias.dtype
            )));
        }
        let q_dev = self.upload_pod(q)?;
        let k_dev = self.upload_pod(k)?;
        let v_dev = self.upload_pod(v)?;
        let beta_dev = self.upload_pod(beta)?;
        let alpha_dev = self.upload_pod(alpha)?;
        let state_dev = self.upload_pod(recurrent)?;
        let out_dev = self.new_f32(inner)?;
        check(
            unsafe {
                gp_qwen_deltanet_decode(
                    q_dev.as_f32(),
                    k_dev.as_f32(),
                    v_dev.as_f32(),
                    beta_dev.as_f32(),
                    alpha_dev.as_f32(),
                    a_log.buffer.as_f32(),
                    dt_bias.buffer.as_f32(),
                    state_dev.as_f32(),
                    out_dev.as_f32(),
                    heads as i32,
                    head_dim as i32,
                    self.dev.stream(),
                )
            },
            &format!("qwen35 cuda deltanet_decode layer {layer}"),
        )?;
        let mut out = vec![0.0f32; inner];
        let mut state = vec![0.0f32; recurrent.len()];
        self.dev.copy_d2h(&mut out, &out_dev)?;
        self.dev.copy_d2h(&mut state, &state_dev)?;
        Ok((out, state))
    }

    pub fn delta_attention_block(
        &self,
        layer: usize,
        hidden: &[f32],
        recurrent: &[f32],
        conv_state: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        if self.inventory.is_full_attention_layer(layer) || layer >= self.inventory.block_count {
            return Err(Error::InvalidRequest(format!(
                "layer {layer} is not a DeltaNet layer"
            )));
        }
        if hidden.len() != self.inventory.hidden_size {
            return Err(Error::InvalidRequest(format!(
                "Delta attention hidden len {}, expected {}",
                hidden.len(),
                self.inventory.hidden_size
            )));
        }
        let prefix = format!("blk.{layer}");
        let qkv = self.matvec(&format!("{prefix}.attn_qkv.weight"), hidden)?;
        let (mut qkv, conv_state) =
            self.causal_conv1d_silu(&format!("{prefix}.ssm_conv1d.weight"), &qkv, conv_state, 4)?;
        let z = self.matvec(&format!("{prefix}.attn_gate.weight"), hidden)?;
        let beta = self.matvec(&format!("{prefix}.ssm_beta.weight"), hidden)?;
        let alpha = self.matvec(&format!("{prefix}.ssm_alpha.weight"), hidden)?;

        let inner = self.inventory.ssm_inner_size;
        let (q, rest) = qkv.split_at_mut(inner);
        let (k, v) = rest.split_at_mut(inner);
        let (q, k) = self.normalize_linear_qk(
            q,
            k,
            self.inventory.ssm_group_count,
            inner / self.inventory.ssm_group_count,
        )?;
        let (raw, recurrent) = self.deltanet_decode(layer, &q, &k, v, &beta, &alpha, recurrent)?;
        let normed = self.plain_rms_norm(
            &format!("{prefix}.ssm_norm.weight"),
            &raw,
            inner / self.inventory.ssm_group_count,
        )?;
        let gated = self.apply_silu_gate(&normed, &z)?;
        let out = self.matvec(&format!("{prefix}.ssm_out.weight"), &gated)?;
        Ok((out, recurrent, conv_state))
    }

    fn embed_tokens_device(&self, ids: *const u32, dst: *mut f32, rows: usize) -> Result<()> {
        let token = self.weights.require("token_embd.weight")?;
        let token_rows = token.rows()?;
        let hidden = token.cols()?;
        if token_rows != self.inventory.vocab_size || hidden != self.inventory.hidden_size {
            return Err(Error::Gguf(format!(
                "token_embd.weight shape [{token_rows}, {hidden}] does not match inventory [{}, {}]",
                self.inventory.vocab_size, self.inventory.hidden_size
            )));
        }
        let rows_i32 = checked_i32(rows, "embedding rows")?;
        let hidden_i32 = checked_i32(self.inventory.hidden_size, "embedding hidden")?;
        let rc = unsafe {
            match token.dtype {
                GgmlDType::Q4K => gp_embed_q4k(
                    token.buffer.ptr(),
                    ids,
                    dst,
                    rows_i32,
                    hidden_i32,
                    1.0,
                    self.dev.stream(),
                ),
                GgmlDType::Q6K => gp_embed_q6k(
                    token.buffer.ptr(),
                    ids,
                    dst,
                    rows_i32,
                    hidden_i32,
                    1.0,
                    self.dev.stream(),
                ),
                other => {
                    return Err(Error::GenerationUnavailable(format!(
                        "Qwen3.5 CUDA token embedding expects Q4_K/Q6_K, got {other}"
                    )));
                }
            }
        };
        check(rc, "qwen35 cuda embed token_embd.weight")?;
        Ok(())
    }

    fn matvec_device_to(
        &self,
        tensor_name: &str,
        src: *const f32,
        cols: usize,
        dst: *mut f32,
        rows: usize,
        q8_scratch: &DeviceBuffer,
    ) -> Result<()> {
        let tensor = self.weights.require(tensor_name)?;
        let tensor_cols = tensor.cols()?;
        let tensor_rows = tensor.rows()?;
        if tensor_cols != cols || tensor_rows != rows {
            return Err(Error::InvalidRequest(format!(
                "{tensor_name} matvec shape [{tensor_rows}, {tensor_cols}], expected [{rows}, {cols}]"
            )));
        }
        check(
            unsafe {
                gp_mmvq_matvec(
                    tensor.ggml_type_id()?,
                    tensor.buffer.ptr(),
                    src,
                    dst,
                    q8_scratch.ptr(),
                    checked_i64(cols, "matvec cols")?,
                    checked_i64(tensor.row_stride_blocks(), "matvec row stride blocks")?,
                    checked_i64(rows, "matvec rows")?,
                    self.dev.stream(),
                )
            },
            &format!("qwen35 cuda MMVQ matvec {tensor_name}"),
        )?;
        Ok(())
    }

    fn rms_norm_device(
        &self,
        tensor_name: &str,
        src: *const f32,
        dst: *mut f32,
        rows: usize,
        dim: usize,
        qwen_scale: bool,
    ) -> Result<()> {
        let weight = self.weights.require(tensor_name)?;
        if weight.dtype != GgmlDType::F32 || weight.shape.as_slice() != &[dim] {
            return Err(Error::Gguf(format!(
                "{tensor_name} RMSNorm weight shape {:?} dtype {}, expected F32 [{dim}]",
                weight.shape, weight.dtype
            )));
        }
        let rc = unsafe {
            if qwen_scale {
                gp_qwen_rms_norm(
                    src,
                    weight.buffer.as_f32(),
                    dst,
                    checked_i32(rows, "RMSNorm rows")?,
                    checked_i32(dim, "RMSNorm dim")?,
                    RMS_EPS,
                    self.dev.stream(),
                )
            } else {
                gp_rms_norm(
                    src,
                    weight.buffer.as_f32(),
                    dst,
                    checked_i32(rows, "RMSNorm rows")?,
                    checked_i32(dim, "RMSNorm dim")?,
                    RMS_EPS,
                    self.dev.stream(),
                )
            }
        };
        check(rc, &format!("qwen35 cuda RMSNorm {tensor_name}"))?;
        Ok(())
    }

    fn add_device(
        &self,
        lhs: *const f32,
        rhs: *const f32,
        dst: *mut f32,
        total: usize,
    ) -> Result<()> {
        check(
            unsafe {
                gp_qwen_add(
                    lhs,
                    rhs,
                    dst,
                    checked_i32(total, "add total")?,
                    self.dev.stream(),
                )
            },
            "qwen35 cuda add",
        )?;
        Ok(())
    }

    fn add_rms_norm_device(
        &self,
        tensor_name: &str,
        lhs: *const f32,
        rhs: *const f32,
        sum_dst: *mut f32,
        norm_dst: *mut f32,
        rows: usize,
        dim: usize,
    ) -> Result<()> {
        let weight = self.weights.require(tensor_name)?;
        if weight.dtype != GgmlDType::F32 || weight.shape.as_slice() != &[dim] {
            return Err(Error::Gguf(format!(
                "{tensor_name} add_rms_norm weight shape {:?} dtype {}, expected F32 [{dim}]",
                weight.shape, weight.dtype
            )));
        }
        check(
            unsafe {
                gp_qwen_add_rms_norm(
                    lhs,
                    rhs,
                    weight.buffer.as_f32(),
                    sum_dst,
                    norm_dst,
                    checked_i32(rows, "add_rms_norm rows")?,
                    checked_i32(dim, "add_rms_norm dim")?,
                    RMS_EPS,
                    self.dev.stream(),
                )
            },
            &format!("qwen35 cuda add_rms_norm {tensor_name}"),
        )?;
        Ok(())
    }

    fn ffn_block_device(&self, layer: usize, ws: &mut CudaForwardWorkspace) -> Result<()> {
        let prefix = format!("blk.{layer}");
        self.matvec_device_to(
            &format!("{prefix}.ffn_gate.weight"),
            ws.normed.as_f32(),
            self.inventory.hidden_size,
            ws.ffn_gate.as_f32(),
            self.inventory.feed_forward_size,
            &ws.q8_scratch,
        )?;
        self.matvec_device_to(
            &format!("{prefix}.ffn_up.weight"),
            ws.normed.as_f32(),
            self.inventory.hidden_size,
            ws.ffn_up.as_f32(),
            self.inventory.feed_forward_size,
            &ws.q8_scratch,
        )?;
        check(
            unsafe {
                gp_qwen_swiglu(
                    ws.ffn_gate.as_f32(),
                    ws.ffn_up.as_f32(),
                    ws.ffn_gate.as_f32(),
                    checked_i32(self.inventory.feed_forward_size, "SwiGLU total")?,
                    self.dev.stream(),
                )
            },
            "qwen35 cuda FFN SwiGLU",
        )?;
        self.matvec_device_to(
            &format!("{prefix}.ffn_down.weight"),
            ws.ffn_gate.as_f32(),
            self.inventory.feed_forward_size,
            ws.attn_out.as_f32(),
            self.inventory.hidden_size,
            &ws.q8_scratch,
        )
    }

    fn delta_attention_block_device(
        &self,
        layer: usize,
        recurrent: &DeviceBuffer,
        conv: &DeviceBuffer,
        ws: &mut CudaForwardWorkspace,
    ) -> Result<()> {
        let prefix = format!("blk.{layer}");
        let inner = self.inventory.ssm_inner_size;
        let heads = self.inventory.ssm_group_count;
        let head_dim = inner / heads;
        self.matvec_device_to(
            &format!("{prefix}.attn_qkv.weight"),
            ws.normed.as_f32(),
            self.inventory.hidden_size,
            ws.qkv.as_f32(),
            inner * 3,
            &ws.q8_scratch,
        )?;
        let conv_weight = self
            .weights
            .require(&format!("{prefix}.ssm_conv1d.weight"))?;
        if conv_weight.dtype != GgmlDType::F32
            || conv_weight.shape.as_slice() != &[inner * 3, CONV_KERNEL]
        {
            return Err(Error::Gguf(format!(
                "{prefix}.ssm_conv1d.weight shape {:?} dtype {}, expected F32 [{}, {}]",
                conv_weight.shape,
                conv_weight.dtype,
                inner * 3,
                CONV_KERNEL
            )));
        }
        check(
            unsafe {
                gp_qwen_causal_conv1d_silu(
                    ws.qkv.as_f32(),
                    conv_weight.buffer.as_f32(),
                    conv.as_f32(),
                    checked_i32(inner * 3, "Delta conv channels")?,
                    checked_i32(CONV_KERNEL, "Delta conv kernel")?,
                    self.dev.stream(),
                )
            },
            &format!("qwen35 cuda causal_conv1d_silu {prefix}"),
        )?;
        self.matvec_device_to(
            &format!("{prefix}.attn_gate.weight"),
            ws.normed.as_f32(),
            self.inventory.hidden_size,
            ws.z.as_f32(),
            inner,
            &ws.q8_scratch,
        )?;
        self.matvec_device_to(
            &format!("{prefix}.ssm_beta.weight"),
            ws.normed.as_f32(),
            self.inventory.hidden_size,
            ws.beta.as_f32(),
            heads,
            &ws.q8_scratch,
        )?;
        self.matvec_device_to(
            &format!("{prefix}.ssm_alpha.weight"),
            ws.normed.as_f32(),
            self.inventory.hidden_size,
            ws.alpha.as_f32(),
            self.inventory.ssm_time_step_rank,
            &ws.q8_scratch,
        )?;

        let q = ws.qkv.as_f32();
        let k = unsafe { ws.qkv.as_f32().add(inner) };
        let v = unsafe { ws.qkv.as_f32().add(inner * 2) };
        check(
            unsafe {
                gp_qwen_normalize_linear_qk(
                    q,
                    k,
                    checked_i32(heads, "Delta heads")?,
                    checked_i32(head_dim, "Delta head dim")?,
                    RMS_EPS,
                    self.dev.stream(),
                )
            },
            "qwen35 cuda normalize_linear_qk",
        )?;
        let a_log = self.weights.require(&format!("{prefix}.ssm_a"))?;
        let dt_bias = self.weights.require(&format!("{prefix}.ssm_dt.bias"))?;
        if a_log.dtype != GgmlDType::F32 || a_log.shape.as_slice() != &[heads] {
            return Err(Error::Gguf(format!(
                "{prefix}.ssm_a shape {:?} dtype {}, expected F32 [{heads}]",
                a_log.shape, a_log.dtype
            )));
        }
        if dt_bias.dtype != GgmlDType::F32 || dt_bias.shape.as_slice() != &[heads] {
            return Err(Error::Gguf(format!(
                "{prefix}.ssm_dt.bias shape {:?} dtype {}, expected F32 [{heads}]",
                dt_bias.shape, dt_bias.dtype
            )));
        }
        check(
            unsafe {
                gp_qwen_deltanet_decode(
                    q,
                    k,
                    v,
                    ws.beta.as_f32(),
                    ws.alpha.as_f32(),
                    a_log.buffer.as_f32(),
                    dt_bias.buffer.as_f32(),
                    recurrent.as_f32(),
                    ws.raw.as_f32(),
                    checked_i32(heads, "Delta heads")?,
                    checked_i32(head_dim, "Delta head dim")?,
                    self.dev.stream(),
                )
            },
            &format!("qwen35 cuda deltanet_decode layer {layer}"),
        )?;
        self.rms_norm_device(
            &format!("{prefix}.ssm_norm.weight"),
            ws.raw.as_f32(),
            ws.raw.as_f32(),
            heads,
            head_dim,
            false,
        )?;
        check(
            unsafe {
                gp_qwen_apply_silu_gate(
                    ws.raw.as_f32(),
                    ws.z.as_f32(),
                    checked_i32(inner, "Delta gate total")?,
                    self.dev.stream(),
                )
            },
            "qwen35 cuda Delta gate",
        )?;
        self.matvec_device_to(
            &format!("{prefix}.ssm_out.weight"),
            ws.raw.as_f32(),
            inner,
            ws.attn_out.as_f32(),
            self.inventory.hidden_size,
            &ws.q8_scratch,
        )
    }

    fn full_attention_block_device(
        &self,
        layer: usize,
        k_cache: &DeviceBuffer,
        v_cache: &DeviceBuffer,
        position: usize,
        ws: &mut CudaForwardWorkspace,
    ) -> Result<()> {
        let prefix = format!("blk.{layer}");
        let q_dim = self.inventory.attention_heads * self.inventory.head_dim;
        let kv_k_dim = self.inventory.kv_heads * self.inventory.head_dim;
        let kv_v_dim = self.inventory.kv_heads * self.inventory.value_dim;
        self.matvec_device_to(
            &format!("{prefix}.attn_q.weight"),
            ws.normed.as_f32(),
            self.inventory.hidden_size,
            ws.q_fused.as_f32(),
            q_dim * 2,
            &ws.q8_scratch,
        )?;
        self.matvec_device_to(
            &format!("{prefix}.attn_k.weight"),
            ws.normed.as_f32(),
            self.inventory.hidden_size,
            ws.k.as_f32(),
            kv_k_dim,
            &ws.q8_scratch,
        )?;
        self.matvec_device_to(
            &format!("{prefix}.attn_v.weight"),
            ws.normed.as_f32(),
            self.inventory.hidden_size,
            ws.v.as_f32(),
            kv_v_dim,
            &ws.q8_scratch,
        )?;
        check(
            unsafe {
                gp_qwen_deinterleave_q_gate(
                    ws.q_fused.as_f32(),
                    ws.qkv.as_f32(),
                    ws.qkv.as_f32().add(q_dim),
                    1,
                    checked_i32(self.inventory.attention_heads, "attention heads")?,
                    checked_i32(self.inventory.head_dim, "attention head dim")?,
                    checked_i32(q_dim * 2, "packed q stride")?,
                    checked_i32(q_dim * 2, "deinterleaved q stride")?,
                    self.dev.stream(),
                )
            },
            "qwen35 cuda deinterleave q gate",
        )?;
        self.rms_norm_device(
            &format!("{prefix}.attn_q_norm.weight"),
            ws.qkv.as_f32(),
            ws.qkv.as_f32(),
            self.inventory.attention_heads,
            self.inventory.head_dim,
            true,
        )?;
        self.rms_norm_device(
            &format!("{prefix}.attn_k_norm.weight"),
            ws.k.as_f32(),
            ws.k.as_f32(),
            self.inventory.kv_heads,
            self.inventory.head_dim,
            true,
        )?;
        check(
            unsafe {
                gp_qwen_rope_decode(
                    ws.qkv.as_f32(),
                    checked_i32(self.inventory.attention_heads, "attention heads")?,
                    checked_i32(self.inventory.head_dim, "attention head dim")?,
                    checked_i32(self.inventory.rope_dim, "attention rope dim")?,
                    checked_i32(position, "attention position")?,
                    ROPE_THETA,
                    self.dev.stream(),
                )
            },
            "qwen35 cuda q RoPE",
        )?;
        check(
            unsafe {
                gp_qwen_rope_decode(
                    ws.k.as_f32(),
                    checked_i32(self.inventory.kv_heads, "kv heads")?,
                    checked_i32(self.inventory.head_dim, "kv head dim")?,
                    checked_i32(self.inventory.rope_dim, "kv rope dim")?,
                    checked_i32(position, "kv position")?,
                    ROPE_THETA,
                    self.dev.stream(),
                )
            },
            "qwen35 cuda k RoPE",
        )?;
        check(
            unsafe {
                gp_qwen_cache_write(
                    ws.k.as_f32(),
                    k_cache.as_f32(),
                    checked_i32(position, "k cache position")?,
                    checked_i32(self.inventory.kv_heads, "kv heads")?,
                    checked_i32(self.inventory.head_dim, "kv head dim")?,
                    checked_i32(
                        state_context_len(
                            k_cache,
                            self.inventory.kv_heads,
                            self.inventory.head_dim,
                        ),
                        "k cache context",
                    )?,
                    self.dev.stream(),
                )
            },
            "qwen35 cuda k cache write",
        )?;
        check(
            unsafe {
                gp_qwen_cache_write(
                    ws.v.as_f32(),
                    v_cache.as_f32(),
                    checked_i32(position, "v cache position")?,
                    checked_i32(self.inventory.kv_heads, "kv heads")?,
                    checked_i32(self.inventory.value_dim, "value dim")?,
                    checked_i32(
                        state_context_len(
                            v_cache,
                            self.inventory.kv_heads,
                            self.inventory.value_dim,
                        ),
                        "v cache context",
                    )?,
                    self.dev.stream(),
                )
            },
            "qwen35 cuda v cache write",
        )?;
        let max_context =
            state_context_len(k_cache, self.inventory.kv_heads, self.inventory.head_dim);
        let scale = 1.0 / (self.inventory.head_dim as f32).sqrt();
        check(
            unsafe {
                gp_qwen_attention_scores_decode(
                    ws.qkv.as_f32(),
                    k_cache.as_f32(),
                    ws.scores.as_f32(),
                    checked_i32(position, "attention position")?,
                    checked_i32(self.inventory.attention_heads, "attention heads")?,
                    checked_i32(self.inventory.kv_heads, "kv heads")?,
                    checked_i32(self.inventory.head_dim, "attention head dim")?,
                    checked_i32(max_context, "attention context")?,
                    scale,
                    self.dev.stream(),
                )
            },
            "qwen35 cuda attention scores",
        )?;
        check(
            unsafe {
                gp_qwen_softmax_decode(
                    ws.scores.as_f32(),
                    checked_i32(position, "attention position")?,
                    checked_i32(self.inventory.attention_heads, "attention heads")?,
                    checked_i32(max_context, "attention context")?,
                    self.dev.stream(),
                )
            },
            "qwen35 cuda attention softmax",
        )?;
        check(
            unsafe {
                gp_qwen_attention_values_decode(
                    ws.scores.as_f32(),
                    v_cache.as_f32(),
                    ws.raw.as_f32(),
                    checked_i32(position, "attention position")?,
                    checked_i32(self.inventory.attention_heads, "attention heads")?,
                    checked_i32(self.inventory.kv_heads, "kv heads")?,
                    checked_i32(self.inventory.value_dim, "value dim")?,
                    checked_i32(max_context, "attention context")?,
                    self.dev.stream(),
                )
            },
            "qwen35 cuda attention values",
        )?;
        let q_gate = unsafe { ws.qkv.as_f32().add(q_dim) };
        check(
            unsafe {
                gp_qwen_apply_sigmoid_gate(
                    ws.raw.as_f32(),
                    q_gate,
                    checked_i32(
                        self.inventory.attention_heads * self.inventory.value_dim,
                        "attention gate total",
                    )?,
                    self.dev.stream(),
                )
            },
            "qwen35 cuda attention gate",
        )?;
        self.matvec_device_to(
            &format!("{prefix}.attn_output.weight"),
            ws.raw.as_f32(),
            self.inventory.attention_heads * self.inventory.value_dim,
            ws.attn_out.as_f32(),
            self.inventory.hidden_size,
            &ws.q8_scratch,
        )
    }

    fn full_attention_cache_only_device(
        &self,
        layer: usize,
        k_cache: &DeviceBuffer,
        v_cache: &DeviceBuffer,
        position: usize,
        ws: &mut CudaForwardWorkspace,
    ) -> Result<()> {
        let prefix = format!("blk.{layer}");
        let kv_k_dim = self.inventory.kv_heads * self.inventory.head_dim;
        let kv_v_dim = self.inventory.kv_heads * self.inventory.value_dim;
        self.matvec_device_to(
            &format!("{prefix}.attn_k.weight"),
            ws.normed.as_f32(),
            self.inventory.hidden_size,
            ws.k.as_f32(),
            kv_k_dim,
            &ws.q8_scratch,
        )?;
        self.matvec_device_to(
            &format!("{prefix}.attn_v.weight"),
            ws.normed.as_f32(),
            self.inventory.hidden_size,
            ws.v.as_f32(),
            kv_v_dim,
            &ws.q8_scratch,
        )?;
        self.rms_norm_device(
            &format!("{prefix}.attn_k_norm.weight"),
            ws.k.as_f32(),
            ws.k.as_f32(),
            self.inventory.kv_heads,
            self.inventory.head_dim,
            true,
        )?;
        check(
            unsafe {
                gp_qwen_rope_decode(
                    ws.k.as_f32(),
                    checked_i32(self.inventory.kv_heads, "kv heads")?,
                    checked_i32(self.inventory.head_dim, "kv head dim")?,
                    checked_i32(self.inventory.rope_dim, "kv rope dim")?,
                    checked_i32(position, "kv position")?,
                    ROPE_THETA,
                    self.dev.stream(),
                )
            },
            "qwen35 cuda cache-only k RoPE",
        )?;
        check(
            unsafe {
                gp_qwen_cache_write(
                    ws.k.as_f32(),
                    k_cache.as_f32(),
                    checked_i32(position, "k cache position")?,
                    checked_i32(self.inventory.kv_heads, "kv heads")?,
                    checked_i32(self.inventory.head_dim, "kv head dim")?,
                    checked_i32(
                        state_context_len(
                            k_cache,
                            self.inventory.kv_heads,
                            self.inventory.head_dim,
                        ),
                        "k cache context",
                    )?,
                    self.dev.stream(),
                )
            },
            "qwen35 cuda cache-only k cache write",
        )?;
        check(
            unsafe {
                gp_qwen_cache_write(
                    ws.v.as_f32(),
                    v_cache.as_f32(),
                    checked_i32(position, "v cache position")?,
                    checked_i32(self.inventory.kv_heads, "kv heads")?,
                    checked_i32(self.inventory.value_dim, "value dim")?,
                    checked_i32(
                        state_context_len(
                            v_cache,
                            self.inventory.kv_heads,
                            self.inventory.value_dim,
                        ),
                        "v cache context",
                    )?,
                    self.dev.stream(),
                )
            },
            "qwen35 cuda cache-only v cache write",
        )?;
        Ok(())
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
}

const RMS_EPS: f32 = 1.0e-6;
const ROPE_THETA: f32 = 10_000_000.0;
const CONV_KERNEL: usize = 4;

fn q8_1_scratch_bytes(cols: usize, rows: usize) -> usize {
    const QK8_1: usize = 32;
    const Q8_1_SIZE: usize = 36;
    const MATRIX_ROW_PADDING: usize = 512;
    let padded = cols.div_ceil(MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING;
    rows.max(1) * (padded / QK8_1) * Q8_1_SIZE + 128 * 144
}

fn argmax_block_count(total: usize) -> usize {
    total.div_ceil(256).max(1)
}

fn is_greedy_device_sampling(params: GenerationParams) -> bool {
    params.temperature == 0.0
        && params.top_k <= 1
        && params.top_p >= 1.0
        && params.min_p == 0.0
        && params.presence_penalty == 0.0
        && params.repetition_penalty == 1.0
}

fn checked_i32(value: usize, name: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| {
        Error::InvalidRequest(format!("{name} value {value} does not fit CUDA i32 ABI"))
    })
}

fn checked_i64(value: usize, name: &str) -> Result<i64> {
    i64::try_from(value).map_err(|_| {
        Error::InvalidRequest(format!("{name} value {value} does not fit CUDA i64 ABI"))
    })
}

fn state_context_len(cache: &DeviceBuffer, heads: usize, dim: usize) -> usize {
    cache.bytes() / (heads * dim * std::mem::size_of::<f32>())
}

fn prompt_seed(prompt: &str) -> u64 {
    let mut h = DefaultHasher::new();
    prompt.hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use greppy_embed_native::matmul::QuantMatrix;

    #[test]
    fn qwen35_cuda_embedding_matches_cpu_for_real_model_when_env_set() {
        let Some(path) = std::env::var_os("QWEN35_NATIVE_GGUF") else {
            eprintln!("skipping qwen35 CUDA embedding parity: QWEN35_NATIVE_GGUF unset");
            return;
        };
        let gguf = GgufModel::open(&path).expect("open Qwen3.5 GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 inventory");
        inventory
            .validate_core_tensors(&gguf)
            .expect("Qwen3.5 core tensors");
        let cuda = CudaQwen35Model::from_gguf(&gguf, inventory, 248_044).expect("CUDA Qwen model");
        assert_eq!(cuda.backend_name(), "cuda-q4k-components");
        assert_eq!(cuda.eos_token_id(), 248_044);

        let ids = [0_u32, 1, 42, 1024, 248_000];
        let gpu = cuda.embed_tokens(&ids).expect("CUDA token embeddings");
        let cpu = QuantMatrix::from_model(&gguf, "token_embd.weight")
            .expect("CPU token_embd matrix")
            .embedding_rows(&ids)
            .expect("CPU token embeddings");
        assert_eq!(gpu.len(), cpu.len());
        let max_abs = gpu
            .iter()
            .zip(&cpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs <= 2.0e-5,
            "CUDA Q4_K token embedding drift too high: {max_abs:.6e}"
        );
    }

    #[test]
    fn qwen35_cuda_quant_matvec_matches_cpu_for_real_model_when_env_set() {
        let Some(path) = std::env::var_os("QWEN35_NATIVE_GGUF") else {
            eprintln!("skipping qwen35 CUDA MMVQ parity: QWEN35_NATIVE_GGUF unset");
            return;
        };
        let gguf = GgufModel::open(&path).expect("open Qwen3.5 GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 inventory");
        inventory
            .validate_core_tensors(&gguf)
            .expect("Qwen3.5 core tensors");
        let cuda = CudaQwen35Model::from_gguf(&gguf, inventory, 248_044).expect("CUDA Qwen model");
        let test_tensors = [
            "blk.0.attn_qkv.weight",
            "blk.0.ssm_out.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_down.weight",
            "blk.3.attn_q.weight",
            "blk.3.attn_output.weight",
        ];
        for name in test_tensors {
            let tensor = gguf.tensor(name).expect("GGUF tensor");
            eprintln!("{name}: dtype={} shape={:?}", tensor.dtype, tensor.shape);
            let matrix = QuantMatrix::from_model(&gguf, name).expect("CPU quant matrix");
            let input = exact_q8_1_input(matrix.cols());
            let gpu = cuda.matvec(name, &input).expect("CUDA quant matvec");
            let cpu = matrix.matmul(&input, 1).expect("CPU quant matvec");
            assert_eq!(gpu.len(), cpu.len(), "{name}");
            let max_abs = gpu
                .iter()
                .zip(&cpu)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let cosine = cosine(&gpu, &cpu);
            let rms = rms_diff(&gpu, &cpu);
            let cpu_rms = rms_norm(&cpu).max(1.0e-6);
            eprintln!(
                "{name}: cosine={cosine:.8} rms_rel={:.6e} max_abs={max_abs:.6e}",
                rms / cpu_rms
            );
            assert!(
                cosine >= 0.999,
                "{name} CUDA quant matvec cosine too low: {cosine:.8}, max_abs={max_abs:.6e}"
            );
            assert!(
                rms / cpu_rms <= 5.0e-2,
                "{name} CUDA quant matvec rms_rel too high: {:.6e}, max_abs={max_abs:.6e}",
                rms / cpu_rms
            );
        }
    }

    #[test]
    fn qwen35_cuda_rms_norm_matches_cpu_for_real_model_when_env_set() {
        let Some(path) = std::env::var_os("QWEN35_NATIVE_GGUF") else {
            eprintln!("skipping qwen35 CUDA RMSNorm parity: QWEN35_NATIVE_GGUF unset");
            return;
        };
        let gguf = GgufModel::open(&path).expect("open Qwen3.5 GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 inventory");
        inventory
            .validate_core_tensors(&gguf)
            .expect("Qwen3.5 core tensors");
        let cuda =
            CudaQwen35Model::from_gguf(&gguf, inventory.clone(), 248_044).expect("CUDA Qwen model");

        let hidden_input = patterned_input(2 * inventory.hidden_size);
        let hidden_weight = gguf
            .tensor("blk.0.attn_norm.weight")
            .expect("attn_norm tensor")
            .to_f32()
            .expect("attn_norm f32");
        let hidden_gpu = cuda
            .qwen_rms_norm(
                "blk.0.attn_norm.weight",
                &hidden_input,
                inventory.hidden_size,
            )
            .expect("CUDA Qwen RMSNorm");
        let hidden_cpu = cpu_rms_norm(&hidden_input, &hidden_weight, inventory.hidden_size, true);
        assert_close_vec("blk.0.attn_norm.weight", &hidden_gpu, &hidden_cpu, 1.0e-4);

        let ssm_dim = inventory.ssm_inner_size / inventory.ssm_group_count;
        let ssm_input = patterned_input(3 * ssm_dim);
        let ssm_weight = gguf
            .tensor("blk.0.ssm_norm.weight")
            .expect("ssm_norm tensor")
            .to_f32()
            .expect("ssm_norm f32");
        let ssm_gpu = cuda
            .plain_rms_norm("blk.0.ssm_norm.weight", &ssm_input, ssm_dim)
            .expect("CUDA plain RMSNorm");
        let ssm_cpu = cpu_rms_norm(&ssm_input, &ssm_weight, ssm_dim, false);
        assert_close_vec("blk.0.ssm_norm.weight", &ssm_gpu, &ssm_cpu, 1.0e-4);
    }

    #[test]
    fn qwen35_cuda_delta_preprocess_matches_cpu_for_real_model_when_env_set() {
        let Some(path) = std::env::var_os("QWEN35_NATIVE_GGUF") else {
            eprintln!("skipping qwen35 CUDA delta preprocess parity: QWEN35_NATIVE_GGUF unset");
            return;
        };
        let gguf = GgufModel::open(&path).expect("open Qwen3.5 GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 inventory");
        inventory
            .validate_core_tensors(&gguf)
            .expect("Qwen3.5 core tensors");
        let cuda =
            CudaQwen35Model::from_gguf(&gguf, inventory.clone(), 248_044).expect("CUDA Qwen model");

        let channels = inventory.ssm_inner_size * 3;
        let kernel = 4;
        let values = patterned_input(channels);
        let state = patterned_input(channels * kernel);
        let conv_weight = gguf
            .tensor("blk.0.ssm_conv1d.weight")
            .expect("ssm_conv1d tensor")
            .to_f32()
            .expect("ssm_conv1d f32");
        let mut cpu_values = values.clone();
        let mut cpu_state = state.clone();
        cpu_causal_conv1d_silu(&mut cpu_values, &conv_weight, &mut cpu_state, kernel);
        let (gpu_values, gpu_state) = cuda
            .causal_conv1d_silu("blk.0.ssm_conv1d.weight", &values, &state, kernel)
            .expect("CUDA causal_conv1d_silu");
        assert_close_vec(
            "blk.0.ssm_conv1d.weight values",
            &gpu_values,
            &cpu_values,
            1.0e-4,
        );
        assert_close_vec(
            "blk.0.ssm_conv1d.weight state",
            &gpu_state,
            &cpu_state,
            1.0e-6,
        );

        let heads = inventory.ssm_group_count;
        let head_dim = inventory.ssm_inner_size / inventory.ssm_group_count;
        let q = patterned_input(inventory.ssm_inner_size);
        let k = patterned_input(inventory.ssm_inner_size)
            .into_iter()
            .map(|v| v * 0.75 - 0.125)
            .collect::<Vec<_>>();
        let mut cpu_q = q.clone();
        let mut cpu_k = k.clone();
        cpu_normalize_linear_qk(&mut cpu_q, &mut cpu_k, heads, head_dim);
        let (gpu_q, gpu_k) = cuda
            .normalize_linear_qk(&q, &k, heads, head_dim)
            .expect("CUDA normalize_linear_qk");
        assert_close_vec("normalize_linear_qk q", &gpu_q, &cpu_q, 1.0e-5);
        assert_close_vec("normalize_linear_qk k", &gpu_k, &cpu_k, 1.0e-5);
    }

    #[test]
    fn qwen35_cuda_swiglu_matches_cpu_for_real_model_when_env_set() {
        let Some(path) = std::env::var_os("QWEN35_NATIVE_GGUF") else {
            eprintln!("skipping qwen35 CUDA SwiGLU parity: QWEN35_NATIVE_GGUF unset");
            return;
        };
        let gguf = GgufModel::open(&path).expect("open Qwen3.5 GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 inventory");
        inventory
            .validate_core_tensors(&gguf)
            .expect("Qwen3.5 core tensors");
        let cuda =
            CudaQwen35Model::from_gguf(&gguf, inventory.clone(), 248_044).expect("CUDA Qwen model");

        let hidden = patterned_input(inventory.hidden_size)
            .into_iter()
            .map(|v| v * 0.1)
            .collect::<Vec<_>>();
        let gate = cuda
            .matvec("blk.0.ffn_gate.weight", &hidden)
            .expect("CUDA ffn_gate matvec");
        let up = cuda
            .matvec("blk.0.ffn_up.weight", &hidden)
            .expect("CUDA ffn_up matvec");
        let gpu = cuda.swiglu(&gate, &up).expect("CUDA SwiGLU");
        let cpu = gate
            .iter()
            .zip(&up)
            .map(|(gate, up)| silu(*gate) * *up)
            .collect::<Vec<_>>();
        assert_close_vec("blk.0 ffn SwiGLU", &gpu, &cpu, 1.0e-5);

        let values = patterned_input(inventory.ssm_inner_size);
        let gate = gate[..inventory.ssm_inner_size].to_vec();
        let gpu = cuda
            .apply_silu_gate(&values, &gate)
            .expect("CUDA apply_silu_gate");
        let cpu = values
            .iter()
            .zip(&gate)
            .map(|(value, gate)| *value * silu(*gate))
            .collect::<Vec<_>>();
        assert_close_vec("Qwen SiLU gate", &gpu, &cpu, 1.0e-5);
    }

    #[test]
    fn qwen35_cuda_ffn_block_matches_cpu_for_real_model_when_env_set() {
        let Some(path) = std::env::var_os("QWEN35_NATIVE_GGUF") else {
            eprintln!("skipping qwen35 CUDA FFN parity: QWEN35_NATIVE_GGUF unset");
            return;
        };
        let gguf = GgufModel::open(&path).expect("open Qwen3.5 GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 inventory");
        inventory
            .validate_core_tensors(&gguf)
            .expect("Qwen3.5 core tensors");
        let cuda =
            CudaQwen35Model::from_gguf(&gguf, inventory.clone(), 248_044).expect("CUDA Qwen model");

        let hidden = patterned_input(inventory.hidden_size)
            .into_iter()
            .map(|v| v * 0.1)
            .collect::<Vec<_>>();
        let gpu = cuda.ffn_block(0, &hidden).expect("CUDA FFN block");
        let gate = QuantMatrix::from_model(&gguf, "blk.0.ffn_gate.weight")
            .expect("CPU ffn_gate")
            .matmul(&hidden, 1)
            .expect("CPU ffn_gate matvec");
        let up = QuantMatrix::from_model(&gguf, "blk.0.ffn_up.weight")
            .expect("CPU ffn_up")
            .matmul(&hidden, 1)
            .expect("CPU ffn_up matvec");
        let activated = gate
            .iter()
            .zip(up)
            .map(|(gate, up)| silu(*gate) * up)
            .collect::<Vec<_>>();
        let cpu = QuantMatrix::from_model(&gguf, "blk.0.ffn_down.weight")
            .expect("CPU ffn_down")
            .matmul(&activated, 1)
            .expect("CPU ffn_down matvec");
        let cosine = cosine(&gpu, &cpu);
        let rms = rms_diff(&gpu, &cpu);
        let cpu_rms = rms_norm(&cpu).max(1.0e-6);
        assert!(
            cosine >= 0.999,
            "CUDA FFN cosine too low: {cosine:.8}, rms_rel={:.6e}",
            rms / cpu_rms
        );
        assert!(
            rms / cpu_rms <= 5.0e-2,
            "CUDA FFN rms_rel too high: {:.6e}",
            rms / cpu_rms
        );
    }

    #[test]
    fn qwen35_cuda_deltanet_decode_matches_cpu_for_real_model_when_env_set() {
        let Some(path) = std::env::var_os("QWEN35_NATIVE_GGUF") else {
            eprintln!("skipping qwen35 CUDA DeltaNet decode parity: QWEN35_NATIVE_GGUF unset");
            return;
        };
        let gguf = GgufModel::open(&path).expect("open Qwen3.5 GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 inventory");
        inventory
            .validate_core_tensors(&gguf)
            .expect("Qwen3.5 core tensors");
        let cuda =
            CudaQwen35Model::from_gguf(&gguf, inventory.clone(), 248_044).expect("CUDA Qwen model");

        let heads = inventory.ssm_group_count;
        let head_dim = inventory.ssm_inner_size / heads;
        let q = patterned_input(inventory.ssm_inner_size)
            .into_iter()
            .map(|v| v * 0.01)
            .collect::<Vec<_>>();
        let k = patterned_input(inventory.ssm_inner_size)
            .into_iter()
            .map(|v| v * -0.0125)
            .collect::<Vec<_>>();
        let v = patterned_input(inventory.ssm_inner_size)
            .into_iter()
            .map(|v| v * 0.2)
            .collect::<Vec<_>>();
        let beta = patterned_input(heads);
        let alpha = patterned_input(heads)
            .into_iter()
            .map(|v| v * 0.5)
            .collect::<Vec<_>>();
        let recurrent = patterned_input(heads * head_dim * head_dim)
            .into_iter()
            .map(|v| v * 0.002)
            .collect::<Vec<_>>();
        let a_log = gguf
            .tensor("blk.0.ssm_a")
            .expect("ssm_a tensor")
            .to_f32()
            .expect("ssm_a f32");
        let dt_bias = gguf
            .tensor("blk.0.ssm_dt.bias")
            .expect("ssm_dt.bias tensor")
            .to_f32()
            .expect("ssm_dt.bias f32");
        let mut cpu_state = recurrent.clone();
        let cpu = cpu_deltanet_decode(
            &q,
            &k,
            &v,
            &beta,
            &alpha,
            &a_log,
            &dt_bias,
            &mut cpu_state,
            heads,
            head_dim,
        );
        let (gpu, gpu_state) = cuda
            .deltanet_decode(0, &q, &k, &v, &beta, &alpha, &recurrent)
            .expect("CUDA DeltaNet decode");
        assert_close_vec("DeltaNet decode out", &gpu, &cpu, 2.0e-5);
        assert_close_vec("DeltaNet decode state", &gpu_state, &cpu_state, 2.0e-5);
    }

    #[test]
    fn qwen35_cuda_delta_attention_block_matches_cpu_for_real_model_when_env_set() {
        let Some(path) = std::env::var_os("QWEN35_NATIVE_GGUF") else {
            eprintln!("skipping qwen35 CUDA Delta attention parity: QWEN35_NATIVE_GGUF unset");
            return;
        };
        let gguf = GgufModel::open(&path).expect("open Qwen3.5 GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 inventory");
        inventory
            .validate_core_tensors(&gguf)
            .expect("Qwen3.5 core tensors");
        let cuda =
            CudaQwen35Model::from_gguf(&gguf, inventory.clone(), 248_044).expect("CUDA Qwen model");

        let heads = inventory.ssm_group_count;
        let head_dim = inventory.ssm_inner_size / heads;
        let hidden = patterned_input(inventory.hidden_size)
            .into_iter()
            .map(|v| v * 0.1)
            .collect::<Vec<_>>();
        let recurrent = patterned_input(heads * head_dim * head_dim)
            .into_iter()
            .map(|v| v * 0.001)
            .collect::<Vec<_>>();
        let conv_state = patterned_input(inventory.ssm_inner_size * 3 * 4)
            .into_iter()
            .map(|v| v * 0.05)
            .collect::<Vec<_>>();

        let (gpu, gpu_recurrent, gpu_conv) = cuda
            .delta_attention_block(0, &hidden, &recurrent, &conv_state)
            .expect("CUDA Delta attention block");

        let mut qkv = QuantMatrix::from_model(&gguf, "blk.0.attn_qkv.weight")
            .expect("CPU attn_qkv")
            .matmul(&hidden, 1)
            .expect("CPU attn_qkv matvec");
        let conv_weight = gguf
            .tensor("blk.0.ssm_conv1d.weight")
            .expect("ssm_conv1d tensor")
            .to_f32()
            .expect("ssm_conv1d f32");
        let mut cpu_conv = conv_state.clone();
        cpu_causal_conv1d_silu(&mut qkv, &conv_weight, &mut cpu_conv, 4);
        let z = QuantMatrix::from_model(&gguf, "blk.0.attn_gate.weight")
            .expect("CPU attn_gate")
            .matmul(&hidden, 1)
            .expect("CPU attn_gate matvec");
        let beta = QuantMatrix::from_model(&gguf, "blk.0.ssm_beta.weight")
            .expect("CPU ssm_beta")
            .matmul(&hidden, 1)
            .expect("CPU ssm_beta matvec");
        let alpha = QuantMatrix::from_model(&gguf, "blk.0.ssm_alpha.weight")
            .expect("CPU ssm_alpha")
            .matmul(&hidden, 1)
            .expect("CPU ssm_alpha matvec");

        let inner = inventory.ssm_inner_size;
        let (q, rest) = qkv.split_at_mut(inner);
        let (k, v) = rest.split_at_mut(inner);
        cpu_normalize_linear_qk(q, k, heads, head_dim);
        let a_log = gguf
            .tensor("blk.0.ssm_a")
            .expect("ssm_a tensor")
            .to_f32()
            .expect("ssm_a f32");
        let dt_bias = gguf
            .tensor("blk.0.ssm_dt.bias")
            .expect("ssm_dt.bias tensor")
            .to_f32()
            .expect("ssm_dt.bias f32");
        let mut cpu_recurrent = recurrent.clone();
        let raw = cpu_deltanet_decode(
            q,
            k,
            v,
            &beta,
            &alpha,
            &a_log,
            &dt_bias,
            &mut cpu_recurrent,
            heads,
            head_dim,
        );
        let ssm_norm = gguf
            .tensor("blk.0.ssm_norm.weight")
            .expect("ssm_norm tensor")
            .to_f32()
            .expect("ssm_norm f32");
        let normed = cpu_rms_norm(&raw, &ssm_norm, head_dim, false);
        let gated = normed
            .iter()
            .zip(&z)
            .map(|(value, gate)| *value * silu(*gate))
            .collect::<Vec<_>>();
        let cpu = QuantMatrix::from_model(&gguf, "blk.0.ssm_out.weight")
            .expect("CPU ssm_out")
            .matmul(&gated, 1)
            .expect("CPU ssm_out matvec");

        let cosine = cosine(&gpu, &cpu);
        let rms = rms_diff(&gpu, &cpu);
        let cpu_rms = rms_norm(&cpu).max(1.0e-6);
        assert!(
            cosine >= 0.999,
            "CUDA Delta attention cosine too low: {cosine:.8}, rms_rel={:.6e}",
            rms / cpu_rms
        );
        assert!(
            rms / cpu_rms <= 5.0e-2,
            "CUDA Delta attention rms_rel too high: {:.6e}",
            rms / cpu_rms
        );
        assert_close_vec(
            "Delta attention recurrent",
            &gpu_recurrent,
            &cpu_recurrent,
            2.0e-4,
        );
        assert_close_vec("Delta attention conv", &gpu_conv, &cpu_conv, 2.5e-3);
    }

    #[test]
    fn qwen35_cuda_full_attention_block_matches_cpu_for_real_model_when_env_set() {
        let Some(path) = std::env::var_os("QWEN35_NATIVE_GGUF") else {
            eprintln!("skipping qwen35 CUDA full attention parity: QWEN35_NATIVE_GGUF unset");
            return;
        };
        let gguf = GgufModel::open(&path).expect("open Qwen3.5 GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 inventory");
        inventory
            .validate_core_tensors(&gguf)
            .expect("Qwen3.5 core tensors");
        let cuda =
            CudaQwen35Model::from_gguf(&gguf, inventory.clone(), 248_044).expect("CUDA Qwen model");
        let layer = 3;
        assert!(inventory.is_full_attention_layer(layer));
        let mut ws = cuda.new_forward_workspace(8).expect("CUDA workspace");
        let mut state = cuda.new_forward_state(8).expect("CUDA state");
        let mut cpu_k_cache = vec![0.0f32; 8 * inventory.kv_heads * inventory.head_dim];
        let mut cpu_v_cache = vec![0.0f32; 8 * inventory.kv_heads * inventory.value_dim];
        let mut gpu = Vec::new();
        let mut cpu = Vec::new();

        for position in 0..3 {
            let hidden = patterned_input(inventory.hidden_size)
                .into_iter()
                .map(|v| v * (0.03 + position as f32 * 0.01))
                .collect::<Vec<_>>();
            cuda.dev
                .copy_h2d(&ws.normed, &hidden)
                .expect("upload hidden");
            match &mut state.layer_states[layer] {
                CudaLayerState::Full { k_cache, v_cache } => cuda
                    .full_attention_block_device(layer, k_cache, v_cache, position, &mut ws)
                    .expect("CUDA full attention block"),
                _ => panic!("expected full attention layer state"),
            }
            gpu.resize(inventory.hidden_size, 0.0);
            cuda.dev
                .copy_d2h(&mut gpu, &ws.attn_out)
                .expect("download full attention output");
            cpu = cpu_full_attention_block(
                &gguf,
                &inventory,
                layer,
                &hidden,
                position,
                &mut cpu_k_cache,
                &mut cpu_v_cache,
            );
        }

        let cosine = cosine(&gpu, &cpu);
        let rms = rms_diff(&gpu, &cpu);
        let cpu_rms = rms_norm(&cpu).max(1.0e-6);
        assert!(
            cosine >= 0.999,
            "CUDA full attention cosine too low: {cosine:.8}, rms_rel={:.6e}",
            rms / cpu_rms
        );
        assert!(
            rms / cpu_rms <= 5.0e-2,
            "CUDA full attention rms_rel too high: {:.6e}",
            rms / cpu_rms
        );
    }

    #[test]
    fn qwen35_cuda_forward_token_logits_are_finite_for_real_model_when_env_set() {
        let Some(path) = std::env::var_os("QWEN35_NATIVE_GGUF") else {
            eprintln!("skipping qwen35 CUDA forward logits: QWEN35_NATIVE_GGUF unset");
            return;
        };
        let gguf = GgufModel::open(&path).expect("open Qwen3.5 GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 inventory");
        inventory
            .validate_core_tensors(&gguf)
            .expect("Qwen3.5 core tensors");
        let cuda =
            CudaQwen35Model::from_gguf(&gguf, inventory.clone(), 248_044).expect("CUDA Qwen model");
        let mut state = cuda.new_forward_state(8).expect("CUDA state");
        let mut ws = cuda.new_forward_workspace(8).expect("CUDA workspace");
        let logits = cuda
            .forward_token_logits(42, &mut state, &mut ws)
            .expect("CUDA forward token logits");
        assert_eq!(logits.len(), inventory.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()));
        assert!(rms_norm(&logits) > 1.0e-6);
    }

    #[test]
    fn qwen35_cuda_greedy_token_matches_logits_argmax_for_real_model_when_env_set() {
        let Some(path) = std::env::var_os("QWEN35_NATIVE_GGUF") else {
            eprintln!("skipping qwen35 CUDA greedy argmax: QWEN35_NATIVE_GGUF unset");
            return;
        };
        let gguf = GgufModel::open(&path).expect("open Qwen3.5 GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 inventory");
        inventory
            .validate_core_tensors(&gguf)
            .expect("Qwen3.5 core tensors");
        let cuda =
            CudaQwen35Model::from_gguf(&gguf, inventory.clone(), 248_044).expect("CUDA Qwen model");
        let mut logits_state = cuda.new_forward_state(8).expect("CUDA logits state");
        let mut logits_ws = cuda
            .new_forward_workspace(8)
            .expect("CUDA logits workspace");
        let logits = cuda
            .forward_token_logits(42, &mut logits_state, &mut logits_ws)
            .expect("CUDA forward token logits");
        let expected = logits
            .iter()
            .enumerate()
            .max_by(|(ai, a), (bi, b)| {
                a.partial_cmp(b)
                    .unwrap_or(std::cmp::Ordering::Less)
                    .then_with(|| bi.cmp(ai))
            })
            .map(|(idx, _)| idx as u32)
            .expect("non-empty logits");

        let mut greedy_state = cuda.new_forward_state(8).expect("CUDA greedy state");
        let mut greedy_ws = cuda
            .new_forward_workspace(8)
            .expect("CUDA greedy workspace");
        let actual = cuda
            .forward_token_greedy(42, &mut greedy_state, &mut greedy_ws)
            .expect("CUDA greedy token");
        assert_eq!(actual, expected);
    }

    #[test]
    fn qwen35_cuda_prefill_token_matches_logits_state_for_real_model_when_env_set() {
        let Some(path) = std::env::var_os("QWEN35_NATIVE_GGUF") else {
            eprintln!("skipping qwen35 CUDA prefill parity: QWEN35_NATIVE_GGUF unset");
            return;
        };
        let gguf = GgufModel::open(&path).expect("open Qwen3.5 GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 inventory");
        inventory
            .validate_core_tensors(&gguf)
            .expect("Qwen3.5 core tensors");
        let cuda =
            CudaQwen35Model::from_gguf(&gguf, inventory.clone(), 248_044).expect("CUDA Qwen model");

        let mut logits_state = cuda.new_forward_state(8).expect("CUDA logits state");
        let mut logits_ws = cuda
            .new_forward_workspace(8)
            .expect("CUDA logits workspace");
        let _ = cuda
            .forward_token_logits(42, &mut logits_state, &mut logits_ws)
            .expect("CUDA logits path token 0");
        let _ = cuda
            .forward_token_logits(314, &mut logits_state, &mut logits_ws)
            .expect("CUDA logits path token 1");
        let logits_expected = cuda
            .forward_token_logits(2718, &mut logits_state, &mut logits_ws)
            .expect("CUDA logits path final token");

        let mut prefill_state = cuda.new_forward_state(8).expect("CUDA prefill state");
        let mut prefill_ws = cuda
            .new_forward_workspace(8)
            .expect("CUDA prefill workspace");
        cuda.prefill_token(42, &mut prefill_state, &mut prefill_ws)
            .expect("CUDA prefill token 0");
        cuda.prefill_token(314, &mut prefill_state, &mut prefill_ws)
            .expect("CUDA prefill token 1");
        let logits_actual = cuda
            .forward_token_logits(2718, &mut prefill_state, &mut prefill_ws)
            .expect("CUDA prefill path final token");

        assert_eq!(logits_actual.len(), logits_expected.len());
        let max_abs = logits_actual
            .iter()
            .zip(&logits_expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs <= 1.0e-6,
            "CUDA prefill state drift too high: {max_abs:.6e}"
        );
    }

    fn patterned_input(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (((i * 17 + 11) % 101) as f32 - 50.0) / 23.0)
            .collect()
    }

    fn cpu_rms_norm(input: &[f32], weight: &[f32], dim: usize, _qwen_scale: bool) -> Vec<f32> {
        assert_eq!(input.len() % dim, 0);
        assert_eq!(weight.len(), dim);
        let mut out = vec![0.0f32; input.len()];
        for (row_in, row_out) in input.chunks_exact(dim).zip(out.chunks_exact_mut(dim)) {
            let sum_sq = row_in
                .iter()
                .map(|v| (*v as f64) * (*v as f64))
                .sum::<f64>();
            let rstd = (1.0 / ((sum_sq / dim as f64) + RMS_EPS as f64).sqrt()) as f32;
            for ((dst, src), w) in row_out.iter_mut().zip(row_in).zip(weight) {
                *dst = *src * rstd * *w;
            }
        }
        out
    }

    fn cpu_causal_conv1d_silu(
        values: &mut [f32],
        weights: &[f32],
        state: &mut [f32],
        kernel: usize,
    ) {
        assert_eq!(state.len(), values.len() * kernel);
        assert_eq!(weights.len(), values.len() * kernel);
        for channel in 0..values.len() {
            let base = channel * kernel;
            for i in 0..kernel - 1 {
                state[base + i] = state[base + i + 1];
            }
            state[base + kernel - 1] = values[channel];
            let mut acc = 0.0f32;
            for i in 0..kernel {
                acc += state[base + i] * weights[base + i];
            }
            values[channel] = silu(acc);
        }
    }

    fn cpu_normalize_linear_qk(q: &mut [f32], k: &mut [f32], heads: usize, head_dim: usize) {
        let q_scale = 1.0 / (head_dim as f32).sqrt();
        for head in 0..heads {
            let base = head * head_dim;
            let qh = &mut q[base..base + head_dim];
            let kh = &mut k[base..base + head_dim];
            let q_norm = (qh.iter().map(|v| v * v).sum::<f32>() + RMS_EPS).sqrt();
            let k_norm = (kh.iter().map(|v| v * v).sum::<f32>() + RMS_EPS).sqrt();
            for v in qh {
                *v = *v / q_norm * q_scale;
            }
            for v in kh {
                *v /= k_norm;
            }
        }
    }

    fn assert_close_vec(name: &str, gpu: &[f32], cpu: &[f32], max_allowed: f32) {
        assert_eq!(gpu.len(), cpu.len(), "{name}");
        let max_abs = gpu
            .iter()
            .zip(cpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs <= max_allowed,
            "{name} CUDA drift too high: max_abs={max_abs:.6e}"
        );
    }

    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn softplus(x: f32) -> f32 {
        if x > 20.0 {
            x
        } else {
            (1.0 + x.exp()).ln()
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn cpu_deltanet_decode(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        beta: &[f32],
        alpha: &[f32],
        a_log: &[f32],
        dt_bias: &[f32],
        recurrent: &mut [f32],
        heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; heads * head_dim];
        for head in 0..heads {
            let base = head * head_dim;
            let beta_h = sigmoid(beta[head]);
            let decay = (-a_log[head].exp() * softplus(alpha[head] + dt_bias[head]))
                .exp()
                .clamp(0.0, 1.0);
            for value_idx in 0..head_dim {
                let row_base = (head * head_dim + value_idx) * head_dim;
                let mut prior = 0.0f32;
                for key_idx in 0..head_dim {
                    prior += recurrent[row_base + key_idx] * k[base + key_idx];
                }
                let delta = (v[base + value_idx] - decay * prior) * beta_h;
                let mut attn = 0.0f32;
                for key_idx in 0..head_dim {
                    let idx = row_base + key_idx;
                    recurrent[idx] = decay * recurrent[idx] + k[base + key_idx] * delta;
                    attn += recurrent[idx] * q[base + key_idx];
                }
                out[base + value_idx] = attn;
            }
        }
        out
    }

    fn cpu_full_attention_block(
        gguf: &GgufModel,
        inventory: &Qwen35Inventory,
        layer: usize,
        hidden: &[f32],
        position: usize,
        k_cache: &mut [f32],
        v_cache: &mut [f32],
    ) -> Vec<f32> {
        let prefix = format!("blk.{layer}");
        let kv_k_dim = inventory.kv_heads * inventory.head_dim;
        let kv_v_dim = inventory.kv_heads * inventory.value_dim;
        let q_fused = QuantMatrix::from_model(gguf, &format!("{prefix}.attn_q.weight"))
            .expect("CPU attn_q")
            .matmul(hidden, 1)
            .expect("CPU attn_q matvec");
        let (mut q, q_gate) =
            split_full_attention_q_gate(&q_fused, inventory.attention_heads, inventory.head_dim);
        let mut k = QuantMatrix::from_model(gguf, &format!("{prefix}.attn_k.weight"))
            .expect("CPU attn_k")
            .matmul(hidden, 1)
            .expect("CPU attn_k matvec");
        let v = QuantMatrix::from_model(gguf, &format!("{prefix}.attn_v.weight"))
            .expect("CPU attn_v")
            .matmul(hidden, 1)
            .expect("CPU attn_v matvec");
        assert_eq!(k.len(), kv_k_dim);
        assert_eq!(v.len(), kv_v_dim);

        let q_norm = gguf
            .tensor(&format!("{prefix}.attn_q_norm.weight"))
            .expect("attn_q_norm tensor")
            .to_f32()
            .expect("attn_q_norm f32");
        let k_norm = gguf
            .tensor(&format!("{prefix}.attn_k_norm.weight"))
            .expect("attn_k_norm tensor")
            .to_f32()
            .expect("attn_k_norm f32");
        q = cpu_rms_norm(&q, &q_norm, inventory.head_dim, true);
        k = cpu_rms_norm(&k, &k_norm, inventory.head_dim, true);
        cpu_apply_rope(&mut q, position, inventory.head_dim, inventory.rope_dim);
        cpu_apply_rope(&mut k, position, inventory.head_dim, inventory.rope_dim);

        let k_off = position * kv_k_dim;
        k_cache[k_off..k_off + kv_k_dim].copy_from_slice(&k);
        let v_off = position * kv_v_dim;
        v_cache[v_off..v_off + kv_v_dim].copy_from_slice(&v);

        let mut attn_out = vec![0.0f32; inventory.attention_heads * inventory.value_dim];
        let gqa = inventory.attention_heads / inventory.kv_heads;
        let score_scale = 1.0 / (inventory.head_dim as f32).sqrt();
        for head in 0..inventory.attention_heads {
            let kv_head = head / gqa;
            let qh = &q[head * inventory.head_dim..(head + 1) * inventory.head_dim];
            let mut scores = Vec::with_capacity(position + 1);
            for pos in 0..=position {
                let key_base = pos * kv_k_dim + kv_head * inventory.head_dim;
                let kh = &k_cache[key_base..key_base + inventory.head_dim];
                scores.push(qh.iter().zip(kh).map(|(a, b)| a * b).sum::<f32>() * score_scale);
            }
            softmax_in_place(&mut scores);
            let dst = &mut attn_out[head * inventory.value_dim..(head + 1) * inventory.value_dim];
            for (pos, score) in scores.iter().copied().enumerate() {
                let value_base = pos * kv_v_dim + kv_head * inventory.value_dim;
                let vh = &v_cache[value_base..value_base + inventory.value_dim];
                for i in 0..inventory.value_dim {
                    dst[i] += score * vh[i];
                }
            }
            let gate = &q_gate[head * inventory.value_dim..(head + 1) * inventory.value_dim];
            for i in 0..inventory.value_dim {
                dst[i] *= sigmoid(gate[i]);
            }
        }
        QuantMatrix::from_model(gguf, &format!("{prefix}.attn_output.weight"))
            .expect("CPU attn_output")
            .matmul(&attn_out, 1)
            .expect("CPU attn_output matvec")
    }

    fn cpu_apply_rope(values: &mut [f32], position: usize, head_dim: usize, rope_dim: usize) {
        let half = rope_dim / 2;
        for head in 0..values.len() / head_dim {
            let base = head * head_dim;
            for i in 0..half {
                let theta =
                    (position as f32) * ROPE_THETA.powf(-((2 * i) as f32) / rope_dim as f32);
                let (sin, cos) = theta.sin_cos();
                let a = values[base + i];
                let b = values[base + half + i];
                values[base + i] = a * cos - b * sin;
                values[base + half + i] = a * sin + b * cos;
            }
        }
    }

    fn split_full_attention_q_gate(
        packed: &[f32],
        heads: usize,
        head_dim: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let q_dim = heads * head_dim;
        assert_eq!(packed.len(), q_dim * 2);
        let mut q = vec![0.0f32; q_dim];
        let mut gate = vec![0.0f32; q_dim];
        for head in 0..heads {
            let src = head * head_dim * 2;
            let dst = head * head_dim;
            q[dst..dst + head_dim].copy_from_slice(&packed[src..src + head_dim]);
            gate[dst..dst + head_dim].copy_from_slice(&packed[src + head_dim..src + head_dim * 2]);
        }
        (q, gate)
    }

    fn softmax_in_place(values: &mut [f32]) {
        let max = values
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, |a, b| a.max(b));
        let mut sum = 0.0f32;
        for value in values.iter_mut() {
            *value = (*value - max).exp();
            sum += *value;
        }
        if sum > 0.0 {
            for value in values {
                *value /= sum;
            }
        }
    }

    fn exact_q8_1_input(len: usize) -> Vec<f32> {
        vec![1.0; len]
    }

    fn cosine(lhs: &[f32], rhs: &[f32]) -> f32 {
        let dot = lhs
            .iter()
            .zip(rhs)
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum::<f64>();
        let ln = lhs
            .iter()
            .map(|v| (*v as f64) * (*v as f64))
            .sum::<f64>()
            .sqrt();
        let rn = rhs
            .iter()
            .map(|v| (*v as f64) * (*v as f64))
            .sum::<f64>()
            .sqrt();
        if ln == 0.0 || rn == 0.0 {
            0.0
        } else {
            (dot / (ln * rn)) as f32
        }
    }

    fn rms_diff(lhs: &[f32], rhs: &[f32]) -> f32 {
        (lhs.iter()
            .zip(rhs)
            .map(|(a, b)| {
                let d = (*a as f64) - (*b as f64);
                d * d
            })
            .sum::<f64>()
            / lhs.len().max(1) as f64)
            .sqrt() as f32
    }

    fn rms_norm(values: &[f32]) -> f32 {
        (values
            .iter()
            .map(|v| (*v as f64) * (*v as f64))
            .sum::<f64>()
            / values.len().max(1) as f64)
            .sqrt() as f32
    }
}
