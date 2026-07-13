//! CUDA-resident Qwen3.5 forward building blocks.
//!
//! This module owns Qwen3.5 CUDA weight residency, device-resident decode
//! workspace/state, and the logits path used by the production summarizer.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use greppy_embed_native::cuda::ffi::{
    check, gp_embed_q4k, gp_embed_q6k, gp_f32_matmul, gp_f32_matvec, gp_mmq_matmul,
    gp_mmq_matmul_q8, gp_mmq_quantize, gp_mmvq_matvec, gp_mmvq_matvec_q8, gp_qwen_add,
    gp_qwen_add_rms_norm, gp_qwen_add_rms_norm_q8, gp_qwen_apply_sigmoid_gate,
    gp_qwen_apply_silu_gate, gp_qwen_argmax, gp_qwen_attention_rows_fused,
    gp_qwen_attention_scores_decode, gp_qwen_attention_scores_decode_position,
    gp_qwen_attention_values_decode, gp_qwen_attention_values_decode_position, gp_qwen_cache_write,
    gp_qwen_cache_write_position, gp_qwen_cache_write_rows, gp_qwen_causal_conv1d_silu,
    gp_qwen_causal_conv1d_silu_rows_parallel, gp_qwen_concat_rows, gp_qwen_deinterleave_q_gate,
    gp_qwen_deltanet_decode, gp_qwen_deltanet_decode_rows, gp_qwen_increment_position,
    gp_qwen_normalize_linear_qk, gp_qwen_normalize_linear_qk_rows, gp_qwen_rms_norm,
    gp_qwen_rms_norm_q8, gp_qwen_rope_decode, gp_qwen_rope_decode_position, gp_qwen_rope_rows,
    gp_qwen_softmax_decode, gp_qwen_softmax_decode_position, gp_qwen_swiglu, gp_rms_norm,
    CudaDevice, CudaGraph, DeviceBuffer,
};
use greppy_embed_native::cuda::weights::CudaWeights;
use greppy_embed_native::{GgmlDType, GgufModel};
use tokenizers::Tokenizer;

#[cfg(test)]
use greppy_embed_native::cuda::ffi::gp_mmvq_quantize;

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
    position_device: DeviceBuffer,
    decode_graph: Option<CudaGraph>,
    decode_graph_mode: Option<CudaDecodeGraphMode>,
    graph_unavailable: bool,
    graph_token: Option<u32>,
    prefill_workspace: Option<CudaPrefillWorkspace>,
    layer_states: Vec<CudaLayerState>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CudaDecodeGraphMode {
    Greedy,
    Logits,
}

pub(crate) struct CudaMtpState {
    position: usize,
    max_context: usize,
    k_cache: DeviceBuffer,
    v_cache: DeviceBuffer,
    workspace: Option<CudaMtpWorkspace>,
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

pub(crate) struct CudaTargetForwardOutput {
    pub(crate) hidden: Vec<f32>,
    pub(crate) logits: Vec<f32>,
}

struct CudaPrefillWorkspace {
    mmq_rows: usize,
    token_ids: DeviceBuffer,
    hidden: DeviceBuffer,
    normed: DeviceBuffer,
    attn_out: DeviceBuffer,
    qkv: DeviceBuffer,
    conv_out: DeviceBuffer,
    z: DeviceBuffer,
    beta: DeviceBuffer,
    alpha: DeviceBuffer,
    raw: DeviceBuffer,
    q_fused: DeviceBuffer,
    q: DeviceBuffer,
    gate: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    ffn_gate: DeviceBuffer,
    ffn_up: DeviceBuffer,
    q8_scratch: DeviceBuffer,
    fixup_scratch: DeviceBuffer,
}

struct CudaMtpWorkspace {
    mmq_rows: usize,
    logits_rows: usize,
    prefill: CudaPrefillWorkspace,
    target_hidden: DeviceBuffer,
    joined: DeviceBuffer,
    logits: DeviceBuffer,
}

impl CudaQwen35Model {
    pub fn from_gguf(
        model: &GgufModel,
        inventory: Qwen35Inventory,
        eos_token_id: u32,
    ) -> Result<Self> {
        let requested = std::env::var("GREPPY_QWEN35_CUDA_DEVICE")
            .or_else(|_| std::env::var("EMBED_NATIVE_CUDA_DEVICE"))
            .ok()
            .map(|value| {
                value.parse::<i32>().map_err(|_| {
                    Error::InvalidRequest(format!(
                        "CUDA device environment value must be an integer, got `{value}`"
                    ))
                })
            })
            .transpose()?;
        let model_bytes = u64::try_from(model.file_len())
            .map_err(|_| Error::InvalidRequest("GGUF length does not fit u64".into()))?;
        let required = greppy_embed_native::estimated_gpu_memory(
            greppy_embed_native::InferenceModelKind::Qwen35,
            model_bytes,
        );
        let device = greppy_embed_native::cuda::ffi::select_cuda_device(required, requested)?;
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
        if self.inventory.has_mtp() {
            return self.generate_with_mtp(tokenizer, prompt, prompt_ids, params);
        }
        self.generate_target_only(tokenizer, prompt, prompt_ids, params)
    }

    #[cfg(test)]
    pub(crate) fn generate_target_only_for_test(
        &self,
        tokenizer: &Tokenizer,
        prompt: &str,
        params: GenerationParams,
    ) -> Result<String> {
        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|error| Error::Tokenizer(error.to_string()))?;
        self.generate_target_only(tokenizer, prompt, encoding.get_ids(), params)
    }

    fn generate_target_only(
        &self,
        tokenizer: &Tokenizer,
        prompt: &str,
        prompt_ids: &[u32],
        params: GenerationParams,
    ) -> Result<String> {
        let max_context = prompt_ids
            .len()
            .saturating_add(params.max_tokens)
            .saturating_add(1)
            .min(self.inventory.context_length);
        let mut state = self.new_forward_state(max_context)?;
        let mut workspace = self.new_forward_workspace(max_context)?;
        let prefill_ids = &prompt_ids[..prompt_ids.len().saturating_sub(1)];
        for chunk in prefill_ids.chunks(CUDA_PREFILL_BATCH_ROWS) {
            self.prefill_tokens(chunk, &mut state)?;
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

    fn generate_with_mtp(
        &self,
        tokenizer: &Tokenizer,
        prompt: &str,
        prompt_ids: &[u32],
        params: GenerationParams,
    ) -> Result<String> {
        let max_context = prompt_ids
            .len()
            .saturating_add(params.max_tokens)
            .saturating_add(1)
            .min(self.inventory.context_length);
        let hidden_size = self.inventory.hidden_size;
        let mut perf = crate::MtpPerfTimer::new();
        let mut target_state = self.new_forward_state(max_context)?;
        let mut target_workspace = self.new_forward_workspace(max_context)?;
        let mut prompt_hidden = Vec::with_capacity(prompt_ids.len() * hidden_size);
        perf.begin_target_prefill();
        for tokens in prompt_ids[..prompt_ids.len() - 1].chunks(CUDA_PREFILL_BATCH_ROWS) {
            prompt_hidden.extend(self.prefill_tokens_hidden(tokens, &mut target_state)?);
        }
        let mut target = self.forward_token_logits_hidden_graph(
            *prompt_ids.last().expect("checked non-empty above"),
            &mut target_state,
            &mut target_workspace,
        )?;
        prompt_hidden.extend_from_slice(&target.hidden);
        perf.finish_target_prefill();

        let zero_hidden = vec![0.0f32; hidden_size];
        let prompt_conditioning =
            mtp_conditioning_rows(&zero_hidden, &prompt_hidden, prompt_ids.len(), hidden_size)?;
        let mut mtp_state = self.new_mtp_state(max_context)?;
        perf.begin_mtp_prefill();
        for start in (0..prompt_ids.len()).step_by(CUDA_PREFILL_BATCH_ROWS) {
            let end = (start + CUDA_PREFILL_BATCH_ROWS).min(prompt_ids.len());
            self.mtp_prefill_tokens(
                &prompt_ids[start..end],
                &prompt_conditioning[start * hidden_size..end * hidden_size],
                &mut mtp_state,
            )?;
        }
        perf.finish_input();

        let mut generated = Vec::with_capacity(params.max_tokens);
        let mut rng = SamplerRng::new(prompt_seed(prompt));
        let Some(first) = sample_token(&mut target.logits, &generated, params, &mut rng) else {
            return Ok(String::new());
        };
        if first == self.eos_token_id {
            return Ok(String::new());
        }
        generated.push(first);
        let mtp_debug = std::env::var_os("GREPPY_QWEN35_MTP_DEBUG").is_some();
        if mtp_debug {
            eprintln!("qwen35-mtp-debug backend=cuda first={first}");
        }
        let mut next = first;
        let mut pending_target_hidden = target.hidden;
        let mut drafted_total = 0usize;
        let mut accepted_total = 0usize;
        let mut cycles = 0usize;
        let mut mtp_fallback = false;
        let mut draft_mtp_state = self.new_mtp_state(max_context)?;

        while generated.len() < params.max_tokens {
            let remaining = params.max_tokens - generated.len();
            if mtp_fallback {
                let stage = perf.begin_stage();
                let mut logits =
                    self.forward_token_logits(next, &mut target_state, &mut target_workspace)?;
                perf.finish_stage(crate::MtpPerfStage::TargetReplay, stage);
                let Some(token) = sample_token(&mut logits, &generated, params, &mut rng) else {
                    break;
                };
                if token == self.eos_token_id {
                    break;
                }
                generated.push(token);
                next = token;
                continue;
            }
            if remaining == 1 {
                let previous_hidden = pending_target_hidden;
                let mut output = self.forward_token_logits_hidden_graph(
                    next,
                    &mut target_state,
                    &mut target_workspace,
                )?;
                self.mtp_prefill_tokens(&[next], &previous_hidden, &mut mtp_state)?;
                let Some(token) = sample_token(&mut output.logits, &generated, params, &mut rng)
                else {
                    break;
                };
                if token == self.eos_token_id {
                    break;
                }
                generated.push(token);
                next = token;
                pending_target_hidden = output.hidden;
                continue;
            }

            cycles += 1;
            let previous_hidden = pending_target_hidden.clone();
            let draft_limit = MTP_DRAFT_MAX.min(remaining - 1);
            let stage = perf.begin_stage();
            self.copy_mtp_state(&mtp_state, &mut draft_mtp_state)?;
            perf.finish_stage(crate::MtpPerfStage::MtpStateCopy, stage);
            let mut draft_rng = rng.clone();
            let mut draft_history = generated.clone();
            let mut draft_tokens = Vec::with_capacity(draft_limit);
            let mut draft_input = next;
            let mut draft_hidden = previous_hidden.clone();
            let stage = perf.begin_stage();
            for _ in 0..draft_limit {
                let mut draft = self.mtp_forward_tokens_logits_hidden(
                    &[draft_input],
                    &draft_hidden,
                    &mut draft_mtp_state,
                )?;
                let Some(token) =
                    sample_token(&mut draft.logits, &draft_history, params, &mut draft_rng)
                else {
                    break;
                };
                draft_tokens.push(token);
                if token == self.eos_token_id {
                    break;
                }
                draft_history.push(token);
                draft_input = token;
                draft_hidden = draft.hidden;
            }
            perf.finish_stage(crate::MtpPerfStage::Draft, stage);

            if draft_tokens.is_empty() {
                let mut output = self.forward_token_logits_hidden_graph(
                    next,
                    &mut target_state,
                    &mut target_workspace,
                )?;
                self.mtp_prefill_tokens(&[next], &previous_hidden, &mut mtp_state)?;
                let Some(token) = sample_token(&mut output.logits, &generated, params, &mut rng)
                else {
                    break;
                };
                if token == self.eos_token_id {
                    break;
                }
                generated.push(token);
                next = token;
                pending_target_hidden = output.hidden;
                continue;
            }
            drafted_total += draft_tokens.len();
            if mtp_debug {
                eprintln!(
                    "qwen35-mtp-debug backend=cuda cycle={cycles} input={next} drafts={draft_tokens:?}"
                );
            }

            let mut verification_tokens = Vec::with_capacity(1 + draft_tokens.len());
            verification_tokens.push(next);
            verification_tokens.extend_from_slice(&draft_tokens);
            let stage = perf.begin_stage();
            let mut committed_hidden = Vec::with_capacity(verification_tokens.len() * hidden_size);
            let mut verify_input = next;
            let mut accepted = 0usize;
            let mut mismatch = None;
            let mut finished = false;
            for (draft_index, &draft_token) in draft_tokens.iter().enumerate() {
                let mut output = self.forward_token_logits_hidden_graph(
                    verify_input,
                    &mut target_state,
                    &mut target_workspace,
                )?;
                committed_hidden.extend_from_slice(&output.hidden);
                let Some(target_token) =
                    sample_token(&mut output.logits, &generated, params, &mut rng)
                else {
                    finished = true;
                    break;
                };
                if target_token == self.eos_token_id {
                    finished = true;
                    break;
                }
                if mtp_debug {
                    eprintln!(
                        "qwen35-mtp-debug backend=cuda cycle={cycles} pos={draft_index} target={target_token} draft={draft_token}"
                    );
                }
                generated.push(target_token);
                if target_token != draft_token {
                    mismatch = Some(target_token);
                    break;
                }
                accepted += 1;
                accepted_total += 1;
                if generated.len() == params.max_tokens {
                    finished = true;
                    break;
                }
                verify_input = draft_token;
            }
            perf.finish_stage(crate::MtpPerfStage::TargetVerify, stage);
            if finished {
                break;
            }
            mtp_fallback = crate::mtp_should_fallback(drafted_total, accepted_total);

            if accepted == draft_tokens.len() {
                let stage = perf.begin_stage();
                let mut output = self.forward_token_logits_hidden_graph(
                    verify_input,
                    &mut target_state,
                    &mut target_workspace,
                )?;
                perf.finish_stage(crate::MtpPerfStage::TargetVerify, stage);
                committed_hidden.extend_from_slice(&output.hidden);
                let Some(target_token) =
                    sample_token(&mut output.logits, &generated, params, &mut rng)
                else {
                    break;
                };
                if target_token == self.eos_token_id {
                    break;
                }
                generated.push(target_token);
                next = target_token;
                // Every verification forward commits its input. The sampled target token is the
                // next, still-unprocessed input, so target-state rollback is never required.
                debug_assert_eq!(
                    committed_hidden.len(),
                    verification_tokens.len() * hidden_size
                );
                pending_target_hidden =
                    last_hidden_row(&committed_hidden, verification_tokens.len(), hidden_size)?
                        .to_vec();
                let conditioning = mtp_conditioning_rows(
                    &previous_hidden,
                    &committed_hidden,
                    verification_tokens.len(),
                    hidden_size,
                )?;
                let stage = perf.begin_stage();
                self.mtp_prefill_tokens(&verification_tokens, &conditioning, &mut mtp_state)?;
                perf.finish_stage(crate::MtpPerfStage::MtpCommit, stage);
            } else {
                let target_token = mismatch.expect("non-accepted draft must have mismatch token");
                let commit_count = accepted + 1;
                let commit_tokens = &verification_tokens[..commit_count];
                // The mismatch output is pending; only the input that produced it was committed.
                debug_assert_eq!(committed_hidden.len(), commit_count * hidden_size);
                let conditioning = mtp_conditioning_rows(
                    &previous_hidden,
                    &committed_hidden,
                    commit_count,
                    hidden_size,
                )?;
                let stage = perf.begin_stage();
                self.mtp_prefill_tokens(commit_tokens, &conditioning, &mut mtp_state)?;
                perf.finish_stage(crate::MtpPerfStage::MtpCommit, stage);
                pending_target_hidden =
                    last_hidden_row(&committed_hidden, commit_count, hidden_size)?.to_vec();
                next = target_token;
            }
        }

        if mtp_debug {
            eprintln!(
                "qwen35-mtp-debug backend=cuda cycles={cycles} drafted={drafted_total} accepted={accepted_total}"
            );
        }
        perf.report(
            "cuda",
            prompt_ids.len(),
            generated.len(),
            cycles,
            drafted_total,
            accepted_total,
            mtp_fallback,
        );
        tokenizer
            .decode(&generated, true)
            .map_err(|error| Error::Tokenizer(error.to_string()))
    }

    pub fn new_forward_state(&self, max_context: usize) -> Result<CudaForwardState> {
        let prefill_rows = max_context
            .min(CUDA_PREFILL_BATCH_ROWS)
            .max(1)
            .div_ceil(CUDA_MMQ_ROW_TILE)
            * CUDA_MMQ_ROW_TILE;
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
            position_device: self.new_bytes(std::mem::size_of::<i32>())?,
            decode_graph: None,
            decode_graph_mode: None,
            graph_unavailable: false,
            graph_token: None,
            prefill_workspace: Some(self.new_prefill_workspace(prefill_rows)?),
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

    pub(crate) fn new_mtp_state(&self, max_context: usize) -> Result<CudaMtpState> {
        if !self.inventory.has_mtp() {
            return Err(Error::GenerationUnavailable(
                "Qwen3.5 model does not contain an MTP layer".into(),
            ));
        }
        Ok(CudaMtpState {
            position: 0,
            max_context,
            k_cache: self
                .new_f32(max_context * self.inventory.kv_heads * self.inventory.head_dim)?,
            v_cache: self
                .new_f32(max_context * self.inventory.kv_heads * self.inventory.value_dim)?,
            workspace: Some(self.new_mtp_workspace(CUDA_MMQ_ROW_TILE, 1)?),
        })
    }

    fn new_mtp_workspace(&self, mmq_rows: usize, logits_rows: usize) -> Result<CudaMtpWorkspace> {
        let hidden_size = self.inventory.hidden_size;
        let logits_rows = logits_rows.max(1);
        Ok(CudaMtpWorkspace {
            mmq_rows,
            logits_rows,
            prefill: self.new_prefill_workspace(mmq_rows)?,
            target_hidden: self.new_f32(mmq_rows * hidden_size)?,
            joined: self.new_f32(mmq_rows * hidden_size * 2)?,
            logits: self.new_f32(logits_rows * self.inventory.vocab_size)?,
        })
    }

    fn copy_mtp_state(&self, src: &CudaMtpState, dst: &mut CudaMtpState) -> Result<()> {
        if src.max_context != dst.max_context {
            return Err(Error::InvalidRequest(
                "CUDA MTP state layouts do not match".into(),
            ));
        }
        self.dev
            .copy_d2d(&dst.k_cache, &src.k_cache, src.k_cache.bytes())?;
        self.dev
            .copy_d2d(&dst.v_cache, &src.v_cache, src.v_cache.bytes())?;
        self.dev.sync()?;
        dst.position = src.position;
        Ok(())
    }

    fn new_prefill_workspace(&self, rows: usize) -> Result<CudaPrefillWorkspace> {
        let hidden = self.inventory.hidden_size;
        let inner = self.inventory.ssm_inner_size;
        let q_dim = self.inventory.attention_heads * self.inventory.head_dim;
        let kv_k_dim = self.inventory.kv_heads * self.inventory.head_dim;
        let kv_v_dim = self.inventory.kv_heads * self.inventory.value_dim;
        let max_mmq_cols = self
            .inventory
            .feed_forward_size
            .max(inner)
            .max(q_dim)
            .max(hidden);
        Ok(CudaPrefillWorkspace {
            mmq_rows: rows,
            token_ids: self.new_bytes(rows * std::mem::size_of::<u32>())?,
            hidden: self.new_f32(rows * hidden)?,
            normed: self.new_f32(rows * hidden)?,
            attn_out: self.new_f32(rows * hidden)?,
            qkv: self.new_f32(rows * inner * 3)?,
            conv_out: self.new_f32(rows * inner * 3)?,
            z: self.new_f32(rows * inner)?,
            beta: self.new_f32(rows * self.inventory.ssm_group_count)?,
            alpha: self.new_f32(rows * self.inventory.ssm_time_step_rank)?,
            raw: self.new_f32(rows * inner.max(q_dim))?,
            q_fused: self.new_f32(rows * q_dim * 2)?,
            q: self.new_f32(rows * q_dim)?,
            gate: self.new_f32(rows * q_dim)?,
            k: self.new_f32(rows * kv_k_dim)?,
            v: self.new_f32(rows * kv_v_dim)?,
            ffn_gate: self.new_f32(rows * self.inventory.feed_forward_size)?,
            ffn_up: self.new_f32(rows * self.inventory.feed_forward_size)?,
            q8_scratch: self.new_bytes(q8_1_scratch_bytes(max_mmq_cols, rows))?,
            fixup_scratch: self.new_bytes(MMQ_FIXUP_SCRATCH_BYTES)?,
        })
    }

    pub(crate) fn forward_token_logits(
        &self,
        token: u32,
        state: &mut CudaForwardState,
        ws: &mut CudaForwardWorkspace,
    ) -> Result<Vec<f32>> {
        self.prepare_decode_graph_mode(state, CudaDecodeGraphMode::Logits);
        if state.decode_graph.is_none() && !state.graph_unavailable {
            if self.capture_logits_graph(token, state, ws).is_err() {
                state.graph_unavailable = true;
            }
        }
        if state.graph_unavailable {
            self.forward_token_logits_device(token, state, ws)?;
        } else {
            if state.graph_token != Some(token) {
                self.dev
                    .copy_h2d(&ws.token_id, std::slice::from_ref(&token))?;
                state.graph_token = Some(token);
            }
            state
                .decode_graph
                .as_ref()
                .expect("logits graph captured above")
                .launch(&self.dev)?;
            state.position += 1;
        }
        let mut logits = vec![0.0f32; self.inventory.vocab_size];
        self.dev.copy_d2h(&mut logits, &ws.logits)?;
        Ok(logits)
    }

    #[allow(dead_code)]
    pub(crate) fn forward_token_logits_hidden(
        &self,
        token: u32,
        state: &mut CudaForwardState,
        ws: &mut CudaForwardWorkspace,
    ) -> Result<CudaTargetForwardOutput> {
        state.decode_graph = None;
        state.decode_graph_mode = None;
        state.graph_token = None;
        self.forward_token_logits_device(token, state, ws)?;
        let mut hidden = vec![0.0f32; self.inventory.hidden_size];
        let mut logits = vec![0.0f32; self.inventory.vocab_size];
        self.dev.copy_d2h(&mut hidden, &ws.hidden)?;
        self.dev.copy_d2h(&mut logits, &ws.logits)?;
        Ok(CudaTargetForwardOutput { hidden, logits })
    }

    fn forward_token_logits_hidden_graph(
        &self,
        token: u32,
        state: &mut CudaForwardState,
        ws: &mut CudaForwardWorkspace,
    ) -> Result<CudaTargetForwardOutput> {
        let logits = self.forward_token_logits(token, state, ws)?;
        let mut hidden = vec![0.0f32; self.inventory.hidden_size];
        self.dev.copy_d2h(&mut hidden, &ws.hidden)?;
        Ok(CudaTargetForwardOutput { hidden, logits })
    }

    #[cfg(test)]
    pub(crate) fn prefill_token(
        &self,
        token: u32,
        state: &mut CudaForwardState,
        ws: &mut CudaForwardWorkspace,
    ) -> Result<()> {
        state.decode_graph = None;
        state.decode_graph_mode = None;
        state.graph_token = None;
        self.forward_token_prefill_device(token, state, ws)
    }

    pub(crate) fn prefill_tokens(
        &self,
        tokens: &[u32],
        state: &mut CudaForwardState,
    ) -> Result<()> {
        let _ = self.prefill_tokens_impl(tokens, state, true)?;
        Ok(())
    }

    pub(crate) fn prefill_tokens_hidden(
        &self,
        tokens: &[u32],
        state: &mut CudaForwardState,
    ) -> Result<Vec<f32>> {
        self.prefill_tokens_impl(tokens, state, false)
    }

    #[allow(dead_code)]
    pub(crate) fn forward_tokens_logits_hidden(
        &self,
        tokens: &[u32],
        state: &mut CudaForwardState,
    ) -> Result<CudaTargetForwardOutput> {
        let hidden = self.prefill_tokens_impl(tokens, state, false)?;
        if tokens.is_empty() {
            return Ok(CudaTargetForwardOutput {
                hidden,
                logits: Vec::new(),
            });
        }
        let rows = tokens.len();
        let workspace = state.prefill_workspace.take().ok_or_else(|| {
            Error::GenerationUnavailable("CUDA prefill workspace is already in use".into())
        })?;
        let result = (|| -> Result<Vec<f32>> {
            self.rms_norm_device(
                "output_norm.weight",
                workspace.hidden.as_f32(),
                workspace.normed.as_f32(),
                rows,
                self.inventory.hidden_size,
                true,
            )?;
            let logits_device = self.new_f32(rows * self.inventory.vocab_size)?;
            for row in 0..rows {
                self.matvec_device_to(
                    "token_embd.weight",
                    unsafe {
                        workspace
                            .normed
                            .as_f32()
                            .add(row * self.inventory.hidden_size)
                    },
                    self.inventory.hidden_size,
                    unsafe { logits_device.as_f32().add(row * self.inventory.vocab_size) },
                    self.inventory.vocab_size,
                    &workspace.q8_scratch,
                )?;
            }
            self.dev.sync()?;
            let mut logits = vec![0.0f32; rows * self.inventory.vocab_size];
            self.dev.copy_d2h(&mut logits, &logits_device)?;
            Ok(logits)
        })();
        state.prefill_workspace = Some(workspace);
        Ok(CudaTargetForwardOutput {
            hidden,
            logits: result?,
        })
    }

    pub(crate) fn mtp_prefill_tokens(
        &self,
        tokens: &[u32],
        target_hidden: &[f32],
        state: &mut CudaMtpState,
    ) -> Result<()> {
        let _ = self.mtp_forward_tokens(tokens, target_hidden, state, false)?;
        Ok(())
    }

    pub(crate) fn mtp_forward_tokens_logits_hidden(
        &self,
        tokens: &[u32],
        target_hidden: &[f32],
        state: &mut CudaMtpState,
    ) -> Result<CudaTargetForwardOutput> {
        self.mtp_forward_tokens(tokens, target_hidden, state, true)
    }

    fn mtp_forward_tokens(
        &self,
        tokens: &[u32],
        target_hidden: &[f32],
        state: &mut CudaMtpState,
        include_logits: bool,
    ) -> Result<CudaTargetForwardOutput> {
        if !self.inventory.has_mtp() {
            return Err(Error::GenerationUnavailable(
                "Qwen3.5 model does not contain an MTP layer".into(),
            ));
        }
        if tokens.is_empty() {
            if target_hidden.is_empty() {
                return Ok(CudaTargetForwardOutput {
                    hidden: Vec::new(),
                    logits: Vec::new(),
                });
            }
            return Err(Error::InvalidRequest(
                "MTP target hidden rows were provided without tokens".into(),
            ));
        }
        let rows = tokens.len();
        let hidden_size = self.inventory.hidden_size;
        if target_hidden.len() != rows * hidden_size {
            return Err(Error::InvalidRequest(format!(
                "MTP target hidden length {}, expected {}x{}",
                target_hidden.len(),
                rows,
                hidden_size
            )));
        }
        if state.position.saturating_add(rows) > state.max_context {
            return Err(Error::InvalidRequest(format!(
                "qwen35 MTP prompt exceeds local context cap {}",
                state.max_context
            )));
        }
        let layer = self.inventory.block_count;
        let prefix = format!("blk.{layer}");
        let mmq_rows = rows.div_ceil(CUDA_MMQ_ROW_TILE) * CUDA_MMQ_ROW_TILE;
        let required_logits_rows = if include_logits { rows } else { 0 };
        let mut mtp_workspace = state.workspace.take().ok_or_else(|| {
            Error::GenerationUnavailable("CUDA MTP workspace is already in use".into())
        })?;
        if mmq_rows > mtp_workspace.mmq_rows || required_logits_rows > mtp_workspace.logits_rows {
            let grown = self.new_mtp_workspace(
                mmq_rows.max(mtp_workspace.mmq_rows),
                required_logits_rows.max(mtp_workspace.logits_rows),
            );
            match grown {
                Ok(workspace) => mtp_workspace = workspace,
                Err(error) => {
                    state.workspace = Some(mtp_workspace);
                    return Err(error);
                }
            }
        }
        let result = (|| -> Result<CudaTargetForwardOutput> {
            let CudaMtpWorkspace {
                prefill: workspace,
                target_hidden: target_buffer,
                joined,
                logits: logits_device,
                ..
            } = &mut mtp_workspace;
            self.dev.copy_h2d(&workspace.token_ids, tokens)?;
            self.dev.copy_h2d(target_buffer, target_hidden)?;
            self.embed_tokens_device(
                workspace.token_ids.as_u32(),
                workspace.hidden.as_f32(),
                rows,
            )?;
            self.rms_norm_device(
                &format!("{prefix}.nextn.enorm.weight"),
                workspace.hidden.as_f32(),
                workspace.normed.as_f32(),
                rows,
                hidden_size,
                true,
            )?;
            self.rms_norm_device(
                &format!("{prefix}.nextn.hnorm.weight"),
                target_buffer.as_f32(),
                workspace.attn_out.as_f32(),
                rows,
                hidden_size,
                true,
            )?;
            check(
                unsafe {
                    gp_qwen_concat_rows(
                        workspace.normed.as_f32(),
                        workspace.attn_out.as_f32(),
                        joined.as_f32(),
                        checked_i32(rows, "MTP concat rows")?,
                        checked_i32(hidden_size, "MTP concat hidden")?,
                        self.dev.stream(),
                    )
                },
                "qwen35 CUDA MTP concat rows",
            )?;
            let mut q8_dtype = None;
            self.projection_matmul_rows_device_to(
                &format!("{prefix}.nextn.eh_proj.weight"),
                joined.as_f32(),
                workspace.mmq_rows,
                hidden_size * 2,
                workspace.hidden.as_f32(),
                hidden_size,
                &mut q8_dtype,
                &workspace.q8_scratch,
                &workspace.fixup_scratch,
            )?;
            self.rms_norm_device(
                &format!("{prefix}.attn_norm.weight"),
                workspace.hidden.as_f32(),
                workspace.normed.as_f32(),
                rows,
                hidden_size,
                true,
            )?;
            self.full_attention_block_rows_device(
                layer,
                &state.k_cache,
                &state.v_cache,
                state.position,
                rows,
                state.max_context,
                workspace,
            )?;
            self.add_rms_norm_device(
                &format!("{prefix}.post_attention_norm.weight"),
                workspace.hidden.as_f32(),
                workspace.attn_out.as_f32(),
                workspace.hidden.as_f32(),
                workspace.normed.as_f32(),
                rows,
                hidden_size,
            )?;
            self.ffn_block_rows_device(layer, rows, workspace)?;
            self.add_device(
                workspace.hidden.as_f32(),
                workspace.attn_out.as_f32(),
                workspace.hidden.as_f32(),
                rows * hidden_size,
            )?;
            if include_logits {
                self.rms_norm_device(
                    &format!("{prefix}.nextn.shared_head_norm.weight"),
                    workspace.hidden.as_f32(),
                    workspace.normed.as_f32(),
                    rows,
                    hidden_size,
                    true,
                )?;
                for row in 0..rows {
                    self.matvec_device_to(
                        "token_embd.weight",
                        unsafe { workspace.normed.as_f32().add(row * hidden_size) },
                        hidden_size,
                        unsafe { logits_device.as_f32().add(row * self.inventory.vocab_size) },
                        self.inventory.vocab_size,
                        &workspace.q8_scratch,
                    )?;
                }
            }
            self.dev.sync()?;
            state.position += rows;
            let hidden = if include_logits {
                let mut hidden = vec![0.0f32; rows * hidden_size];
                self.dev.copy_d2h(&mut hidden, &workspace.normed)?;
                hidden
            } else {
                Vec::new()
            };
            let logits = if include_logits {
                let mut values = vec![0.0f32; rows * self.inventory.vocab_size];
                self.dev.copy_d2h(&mut values, logits_device)?;
                values
            } else {
                Vec::new()
            };
            Ok(CudaTargetForwardOutput { hidden, logits })
        })();
        state.workspace = Some(mtp_workspace);
        result
    }

    fn prefill_tokens_impl(
        &self,
        tokens: &[u32],
        state: &mut CudaForwardState,
        cache_only_final: bool,
    ) -> Result<Vec<f32>> {
        state.decode_graph = None;
        state.decode_graph_mode = None;
        state.graph_token = None;
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        let rows = tokens.len();
        if state.position.saturating_add(rows) > state.max_context {
            return Err(Error::InvalidRequest(format!(
                "qwen35 prompt exceeds local context cap {}",
                state.max_context
            )));
        }
        for &token in tokens {
            if token as usize >= self.inventory.vocab_size {
                return Err(Error::InvalidRequest(format!(
                    "token id {token} out of range for vocab {}",
                    self.inventory.vocab_size
                )));
            }
        }

        let mmq_rows = rows.div_ceil(CUDA_MMQ_ROW_TILE) * CUDA_MMQ_ROW_TILE;
        let mut ws = state.prefill_workspace.take().ok_or_else(|| {
            Error::GenerationUnavailable("CUDA prefill workspace is already in use".into())
        })?;
        if mmq_rows > ws.mmq_rows {
            ws = self.new_prefill_workspace(mmq_rows)?;
        }
        let result = (|| -> Result<Vec<f32>> {
            self.dev.copy_h2d(&ws.token_ids, tokens)?;
            self.embed_tokens_device(ws.token_ids.as_u32(), ws.hidden.as_f32(), rows)?;

            let final_layer = self.inventory.block_count.saturating_sub(1);
            for layer in 0..self.inventory.block_count {
                self.rms_norm_device(
                    &format!("blk.{layer}.attn_norm.weight"),
                    ws.hidden.as_f32(),
                    ws.normed.as_f32(),
                    rows,
                    self.inventory.hidden_size,
                    true,
                )?;
                if layer == final_layer && cache_only_final {
                    match &mut state.layer_states[layer] {
                        CudaLayerState::Full { k_cache, v_cache } => {
                            self.full_attention_cache_only_rows_device(
                                layer,
                                k_cache,
                                v_cache,
                                state.position,
                                rows,
                                state.max_context,
                                &mut ws,
                            )?;
                        }
                        CudaLayerState::Delta { recurrent, conv } => {
                            self.delta_attention_block_rows_device(
                                layer, recurrent, conv, rows, &mut ws,
                            )?;
                        }
                    }
                    self.dev.sync()?;
                    state.position += rows;
                    return Ok(Vec::new());
                }

                match &mut state.layer_states[layer] {
                    CudaLayerState::Delta { recurrent, conv } => {
                        self.delta_attention_block_rows_device(
                            layer, recurrent, conv, rows, &mut ws,
                        )?;
                    }
                    CudaLayerState::Full { k_cache, v_cache } => {
                        self.full_attention_block_rows_device(
                            layer,
                            k_cache,
                            v_cache,
                            state.position,
                            rows,
                            state.max_context,
                            &mut ws,
                        )?;
                    }
                }
                self.add_rms_norm_device(
                    &format!("blk.{layer}.post_attention_norm.weight"),
                    ws.hidden.as_f32(),
                    ws.attn_out.as_f32(),
                    ws.hidden.as_f32(),
                    ws.normed.as_f32(),
                    rows,
                    self.inventory.hidden_size,
                )?;
                self.ffn_block_rows_device(layer, rows, &mut ws)?;
                self.add_device(
                    ws.hidden.as_f32(),
                    ws.attn_out.as_f32(),
                    ws.hidden.as_f32(),
                    rows * self.inventory.hidden_size,
                )?;
            }

            self.dev.sync()?;
            state.position += rows;
            let mut hidden = vec![0.0f32; rows * self.inventory.hidden_size];
            self.dev.copy_d2h(&mut hidden, &ws.hidden)?;
            Ok(hidden)
        })();
        state.prefill_workspace = Some(ws);
        result
    }

    pub(crate) fn forward_token_greedy(
        &self,
        token: u32,
        state: &mut CudaForwardState,
        ws: &mut CudaForwardWorkspace,
    ) -> Result<u32> {
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
        self.prepare_decode_graph_mode(state, CudaDecodeGraphMode::Greedy);
        if state.decode_graph.is_none() && !state.graph_unavailable {
            if self.capture_greedy_graph(token, state, ws).is_err() {
                state.graph_unavailable = true;
            }
        }
        if state.graph_unavailable {
            self.forward_token_logits_device(token, state, ws)?;
            self.argmax_token_device(ws)?;
            let mut output = [0_u32; 1];
            self.dev.copy_d2h(&mut output, &ws.token_id)?;
            return Ok(output[0]);
        } else if state.graph_token != Some(token) {
            self.dev
                .copy_h2d(&ws.token_id, std::slice::from_ref(&token))?;
            state.graph_token = Some(token);
        }
        state
            .decode_graph
            .as_ref()
            .expect("graph captured above")
            .launch(&self.dev)?;
        let mut output = [0_u32; 1];
        self.dev.copy_d2h(&mut output, &ws.token_id)?;
        state.position += 1;
        state.graph_token = Some(output[0]);
        Ok(output[0])
    }

    fn prepare_decode_graph_mode(&self, state: &mut CudaForwardState, mode: CudaDecodeGraphMode) {
        if state.decode_graph_mode == Some(mode) {
            return;
        }
        state.decode_graph = None;
        state.decode_graph_mode = Some(mode);
        state.graph_unavailable = false;
        state.graph_token = None;
    }

    fn argmax_token_device(&self, ws: &CudaForwardWorkspace) -> Result<()> {
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
        Ok(())
    }

    fn capture_greedy_graph(
        &self,
        token: u32,
        state: &mut CudaForwardState,
        ws: &mut CudaForwardWorkspace,
    ) -> Result<()> {
        self.dev
            .copy_h2d(&ws.token_id, std::slice::from_ref(&token))?;
        let position = [checked_i32(state.position, "graph position")?];
        self.dev.copy_h2d(&state.position_device, &position)?;
        self.dev.sync()?;
        self.dev.begin_graph_capture()?;
        if let Err(err) = self.forward_token_graph_device(state, ws, true) {
            self.dev.abort_graph_capture();
            return Err(err);
        }
        state.decode_graph = Some(self.dev.end_graph_capture()?);
        state.decode_graph_mode = Some(CudaDecodeGraphMode::Greedy);
        state.graph_token = Some(token);
        Ok(())
    }

    fn capture_logits_graph(
        &self,
        token: u32,
        state: &mut CudaForwardState,
        ws: &mut CudaForwardWorkspace,
    ) -> Result<()> {
        self.dev
            .copy_h2d(&ws.token_id, std::slice::from_ref(&token))?;
        let position = [checked_i32(state.position, "graph position")?];
        self.dev.copy_h2d(&state.position_device, &position)?;
        self.dev.sync()?;
        self.dev.begin_graph_capture()?;
        if let Err(err) = self.forward_token_graph_device(state, ws, false) {
            self.dev.abort_graph_capture();
            return Err(err);
        }
        state.decode_graph = Some(self.dev.end_graph_capture()?);
        state.decode_graph_mode = Some(CudaDecodeGraphMode::Logits);
        state.graph_token = Some(token);
        Ok(())
    }

    fn forward_token_graph_device(
        &self,
        state: &mut CudaForwardState,
        ws: &mut CudaForwardWorkspace,
        include_argmax: bool,
    ) -> Result<()> {
        self.embed_tokens_device(ws.token_id.as_u32(), ws.hidden.as_f32(), 1)?;
        let position_ptr = state.position_device.ptr() as *const i32;
        for layer in 0..self.inventory.block_count {
            self.rms_norm_q8_device(
                &format!("blk.{layer}.attn_norm.weight"),
                ws.hidden.as_f32(),
                self.inventory.hidden_size,
                &ws.q8_scratch,
            )?;
            match &mut state.layer_states[layer] {
                CudaLayerState::Delta { recurrent, conv } => {
                    self.delta_attention_block_device(layer, recurrent, conv, ws)?;
                }
                CudaLayerState::Full { k_cache, v_cache } => {
                    self.full_attention_block_device(
                        layer,
                        k_cache,
                        v_cache,
                        state.position,
                        Some(position_ptr),
                        ws,
                    )?;
                }
            }
            self.add_rms_norm_q8_device(
                &format!("blk.{layer}.post_attention_norm.weight"),
                ws.hidden.as_f32(),
                ws.attn_out.as_f32(),
                ws.hidden.as_f32(),
                self.inventory.hidden_size,
                &ws.q8_scratch,
            )?;
            self.ffn_block_device(layer, ws)?;
            self.add_device(
                ws.hidden.as_f32(),
                ws.attn_out.as_f32(),
                ws.hidden.as_f32(),
                self.inventory.hidden_size,
            )?;
        }
        self.rms_norm_q8_device(
            "output_norm.weight",
            ws.hidden.as_f32(),
            self.inventory.hidden_size,
            &ws.q8_scratch,
        )?;
        self.matvec_q8_device_to(
            "token_embd.weight",
            self.inventory.hidden_size,
            ws.logits.as_f32(),
            self.inventory.vocab_size,
            &ws.q8_scratch,
        )?;
        if include_argmax {
            self.argmax_token_device(ws)?;
        }
        check(
            unsafe {
                gp_qwen_increment_position(
                    state.position_device.ptr() as *mut i32,
                    self.dev.stream(),
                )
            },
            "qwen35 cuda increment graph position",
        )?;
        Ok(())
    }

    fn forward_token_logits_device(
        &self,
        token: u32,
        state: &mut CudaForwardState,
        ws: &mut CudaForwardWorkspace,
    ) -> Result<()> {
        self.forward_token_hidden_device(token, state, ws)?;
        self.rms_norm_q8_device(
            "output_norm.weight",
            ws.hidden.as_f32(),
            self.inventory.hidden_size,
            &ws.q8_scratch,
        )?;
        self.matvec_q8_device_to(
            "token_embd.weight",
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
            self.rms_norm_q8_device(
                &format!("blk.{layer}.attn_norm.weight"),
                ws.hidden.as_f32(),
                self.inventory.hidden_size,
                &ws.q8_scratch,
            )?;
            match &mut state.layer_states[layer] {
                CudaLayerState::Delta { recurrent, conv } => {
                    self.delta_attention_block_device(layer, recurrent, conv, ws)?;
                }
                CudaLayerState::Full { k_cache, v_cache } => {
                    self.full_attention_block_device(
                        layer,
                        k_cache,
                        v_cache,
                        state.position,
                        None,
                        ws,
                    )?;
                }
            }
            self.add_rms_norm_q8_device(
                &format!("blk.{layer}.post_attention_norm.weight"),
                ws.hidden.as_f32(),
                ws.attn_out.as_f32(),
                ws.hidden.as_f32(),
                self.inventory.hidden_size,
                &ws.q8_scratch,
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

    #[cfg(test)]
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
                self.rms_norm_q8_device(
                    &format!("blk.{layer}.attn_norm.weight"),
                    ws.hidden.as_f32(),
                    self.inventory.hidden_size,
                    &ws.q8_scratch,
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

            self.rms_norm_q8_device(
                &format!("blk.{layer}.attn_norm.weight"),
                ws.hidden.as_f32(),
                self.inventory.hidden_size,
                &ws.q8_scratch,
            )?;
            match &mut state.layer_states[layer] {
                CudaLayerState::Delta { recurrent, conv } => {
                    self.delta_attention_block_device(layer, recurrent, conv, ws)?;
                }
                CudaLayerState::Full { k_cache, v_cache } => {
                    self.full_attention_block_device(
                        layer,
                        k_cache,
                        v_cache,
                        state.position,
                        None,
                        ws,
                    )?;
                }
            }
            self.add_rms_norm_q8_device(
                &format!("blk.{layer}.post_attention_norm.weight"),
                ws.hidden.as_f32(),
                ws.attn_out.as_f32(),
                ws.hidden.as_f32(),
                self.inventory.hidden_size,
                &ws.q8_scratch,
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
        if tensor.dtype == GgmlDType::F32 {
            self.f32_matmul_device_to(tensor_name, src.as_f32(), 1, cols, dst.as_f32(), rows)?;
        } else {
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
        }
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

    #[cfg(test)]
    fn quantize_matvec_input(
        &self,
        src: *const f32,
        cols: usize,
        q8_scratch: &DeviceBuffer,
    ) -> Result<()> {
        check(
            unsafe {
                gp_mmvq_quantize(
                    src,
                    q8_scratch.ptr(),
                    checked_i64(cols, "matvec quantize cols")?,
                    self.dev.stream(),
                )
            },
            "qwen35 cuda MMVQ input quantize",
        )?;
        Ok(())
    }

    fn matvec_q8_device_to(
        &self,
        tensor_name: &str,
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
                gp_mmvq_matvec_q8(
                    tensor.ggml_type_id()?,
                    tensor.buffer.ptr(),
                    dst,
                    q8_scratch.ptr(),
                    checked_i64(cols, "matvec q8 cols")?,
                    checked_i64(tensor.row_stride_blocks(), "matvec q8 row stride blocks")?,
                    checked_i64(rows, "matvec q8 rows")?,
                    self.dev.stream(),
                )
            },
            &format!("qwen35 cuda MMVQ q8 matvec {tensor_name}"),
        )?;
        Ok(())
    }

    fn f32_matmul_device_to(
        &self,
        tensor_name: &str,
        src: *const f32,
        input_rows: usize,
        cols: usize,
        dst: *mut f32,
        rows: usize,
    ) -> Result<()> {
        let tensor = self.weights.require(tensor_name)?;
        if tensor.dtype != GgmlDType::F32 || tensor.cols()? != cols || tensor.rows()? != rows {
            return Err(Error::InvalidRequest(format!(
                "{tensor_name} F32 matmul shape {:?} dtype {}, expected [{rows}, {cols}] F32",
                tensor.shape, tensor.dtype
            )));
        }
        let result = unsafe {
            if input_rows == 1 {
                gp_f32_matvec(
                    tensor.buffer.as_f32(),
                    src,
                    dst,
                    checked_i32(cols, "F32 matvec cols")?,
                    checked_i32(rows, "F32 matvec output rows")?,
                    self.dev.stream(),
                )
            } else {
                gp_f32_matmul(
                    self.dev.blas(),
                    tensor.buffer.as_f32(),
                    src,
                    dst,
                    checked_i32(cols, "F32 matmul cols")?,
                    checked_i32(rows, "F32 matmul output rows")?,
                    checked_i32(input_rows, "F32 matmul input rows")?,
                )
            }
        };
        check(result, &format!("qwen35 cuda F32 projection {tensor_name}"))?;
        Ok(())
    }

    fn projection_matvec_device_to(
        &self,
        tensor_name: &str,
        src: *const f32,
        cols: usize,
        dst: *mut f32,
        rows: usize,
        q8_scratch: &DeviceBuffer,
    ) -> Result<()> {
        match self.weights.require(tensor_name)?.dtype {
            GgmlDType::F32 => self.f32_matmul_device_to(tensor_name, src, 1, cols, dst, rows),
            _ => self.matvec_device_to(tensor_name, src, cols, dst, rows, q8_scratch),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn matmul_rows_device_to(
        &self,
        tensor_name: &str,
        src: *const f32,
        input_rows: usize,
        cols: usize,
        dst: *mut f32,
        rows: usize,
        q8_scratch: &DeviceBuffer,
        fixup_scratch: &DeviceBuffer,
    ) -> Result<()> {
        let tensor = self.weights.require(tensor_name)?;
        let tensor_cols = tensor.cols()?;
        let tensor_rows = tensor.rows()?;
        if tensor_cols != cols || tensor_rows != rows {
            return Err(Error::InvalidRequest(format!(
                "{tensor_name} matmul shape [{tensor_rows}, {tensor_cols}], expected [{rows}, {cols}]"
            )));
        }
        check(
            unsafe {
                gp_mmq_matmul(
                    tensor.ggml_type_id()?,
                    tensor.buffer.ptr(),
                    src,
                    dst,
                    q8_scratch.ptr(),
                    fixup_scratch.ptr(),
                    checked_i64(cols, "matmul cols")?,
                    checked_i64(tensor.row_stride_blocks(), "matmul row stride blocks")?,
                    checked_i64(rows, "matmul output rows")?,
                    checked_i64(input_rows, "matmul input rows")?,
                    self.dev.stream(),
                )
            },
            &format!("qwen35 cuda MMQ matmul {tensor_name}"),
        )?;
        Ok(())
    }

    fn quantize_mmq_rows_device(
        &self,
        tensor_name: &str,
        src: *const f32,
        input_rows: usize,
        cols: usize,
        q8_scratch: &DeviceBuffer,
    ) -> Result<GgmlDType> {
        let tensor = self.weights.require(tensor_name)?;
        if tensor.cols()? != cols {
            return Err(Error::InvalidRequest(format!(
                "{tensor_name} input width {}, expected {cols}",
                tensor.cols()?
            )));
        }
        mmq_activation_layout(tensor.dtype)?;
        check(
            unsafe {
                gp_mmq_quantize(
                    tensor.ggml_type_id()?,
                    src,
                    q8_scratch.ptr(),
                    checked_i64(cols, "MMQ quantize cols")?,
                    checked_i64(input_rows, "MMQ quantize rows")?,
                    self.dev.stream(),
                )
            },
            &format!("qwen35 cuda MMQ quantize for {tensor_name}"),
        )?;
        Ok(tensor.dtype)
    }

    #[allow(clippy::too_many_arguments)]
    fn matmul_rows_q8_device_to(
        &self,
        tensor_name: &str,
        q8_dtype: GgmlDType,
        input_rows: usize,
        cols: usize,
        dst: *mut f32,
        rows: usize,
        q8_scratch: &DeviceBuffer,
        fixup_scratch: &DeviceBuffer,
    ) -> Result<()> {
        let tensor = self.weights.require(tensor_name)?;
        let tensor_cols = tensor.cols()?;
        let tensor_rows = tensor.rows()?;
        if tensor_cols != cols || tensor_rows != rows {
            return Err(Error::InvalidRequest(format!(
                "{tensor_name} matmul shape [{tensor_rows}, {tensor_cols}], expected [{rows}, {cols}]"
            )));
        }
        if mmq_activation_layout(tensor.dtype)? != mmq_activation_layout(q8_dtype)? {
            return Err(Error::InvalidRequest(format!(
                "{tensor_name} {} MMQ activation layout is incompatible with {}",
                tensor.dtype, q8_dtype
            )));
        }
        check(
            unsafe {
                gp_mmq_matmul_q8(
                    tensor.ggml_type_id()?,
                    tensor.buffer.ptr(),
                    dst,
                    q8_scratch.ptr(),
                    fixup_scratch.ptr(),
                    checked_i64(cols, "q8 matmul cols")?,
                    checked_i64(tensor.row_stride_blocks(), "q8 matmul row stride blocks")?,
                    checked_i64(rows, "q8 matmul output rows")?,
                    checked_i64(input_rows, "q8 matmul input rows")?,
                    self.dev.stream(),
                )
            },
            &format!("qwen35 cuda MMQ q8 matmul {tensor_name}"),
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn projection_matmul_rows_device_to(
        &self,
        tensor_name: &str,
        src: *const f32,
        input_rows: usize,
        cols: usize,
        dst: *mut f32,
        rows: usize,
        q8_dtype: &mut Option<GgmlDType>,
        q8_scratch: &DeviceBuffer,
        fixup_scratch: &DeviceBuffer,
    ) -> Result<()> {
        let dtype = self.weights.require(tensor_name)?.dtype;
        if dtype == GgmlDType::F32 {
            return self.f32_matmul_device_to(tensor_name, src, input_rows, cols, dst, rows);
        }
        let needs_quantize = q8_dtype
            .map(|current| {
                Ok::<_, Error>(mmq_activation_layout(current)? != mmq_activation_layout(dtype)?)
            })
            .transpose()?
            .unwrap_or(true);
        if needs_quantize {
            *q8_dtype = Some(self.quantize_mmq_rows_device(
                tensor_name,
                src,
                input_rows,
                cols,
                q8_scratch,
            )?);
        }
        self.matmul_rows_q8_device_to(
            tensor_name,
            q8_dtype.expect("quantized projection must have a Q8 activation layout"),
            input_rows,
            cols,
            dst,
            rows,
            q8_scratch,
            fixup_scratch,
        )
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

    fn rms_norm_q8_device(
        &self,
        tensor_name: &str,
        src: *const f32,
        dim: usize,
        q8_scratch: &DeviceBuffer,
    ) -> Result<()> {
        let weight = self.weights.require(tensor_name)?;
        if weight.dtype != GgmlDType::F32 || weight.shape.as_slice() != &[dim] {
            return Err(Error::Gguf(format!(
                "{tensor_name} RMSNorm-Q8 weight shape {:?} dtype {}, expected F32 [{dim}]",
                weight.shape, weight.dtype
            )));
        }
        check(
            unsafe {
                gp_qwen_rms_norm_q8(
                    src,
                    weight.buffer.as_f32(),
                    q8_scratch.ptr(),
                    checked_i32(dim, "RMSNorm-Q8 dim")?,
                    RMS_EPS,
                    self.dev.stream(),
                )
            },
            &format!("qwen35 cuda RMSNorm-Q8 {tensor_name}"),
        )?;
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

    fn add_rms_norm_q8_device(
        &self,
        tensor_name: &str,
        lhs: *const f32,
        rhs: *const f32,
        sum_dst: *mut f32,
        dim: usize,
        q8_scratch: &DeviceBuffer,
    ) -> Result<()> {
        let weight = self.weights.require(tensor_name)?;
        if weight.dtype != GgmlDType::F32 || weight.shape.as_slice() != &[dim] {
            return Err(Error::Gguf(format!(
                "{tensor_name} add_rms_norm-Q8 weight shape {:?} dtype {}, expected F32 [{dim}]",
                weight.shape, weight.dtype
            )));
        }
        check(
            unsafe {
                gp_qwen_add_rms_norm_q8(
                    lhs,
                    rhs,
                    weight.buffer.as_f32(),
                    sum_dst,
                    q8_scratch.ptr(),
                    checked_i32(dim, "add_rms_norm-Q8 dim")?,
                    RMS_EPS,
                    self.dev.stream(),
                )
            },
            &format!("qwen35 cuda add_rms_norm-Q8 {tensor_name}"),
        )?;
        Ok(())
    }

    fn ffn_block_device(&self, layer: usize, ws: &mut CudaForwardWorkspace) -> Result<()> {
        let prefix = format!("blk.{layer}");
        self.matvec_q8_device_to(
            &format!("{prefix}.ffn_gate.weight"),
            self.inventory.hidden_size,
            ws.ffn_gate.as_f32(),
            self.inventory.feed_forward_size,
            &ws.q8_scratch,
        )?;
        self.matvec_q8_device_to(
            &format!("{prefix}.ffn_up.weight"),
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

    fn ffn_block_rows_device(
        &self,
        layer: usize,
        rows: usize,
        ws: &mut CudaPrefillWorkspace,
    ) -> Result<()> {
        let prefix = format!("blk.{layer}");
        let mut q8_dtype = None;
        self.projection_matmul_rows_device_to(
            &format!("{prefix}.ffn_gate.weight"),
            ws.normed.as_f32(),
            ws.mmq_rows,
            self.inventory.hidden_size,
            ws.ffn_gate.as_f32(),
            self.inventory.feed_forward_size,
            &mut q8_dtype,
            &ws.q8_scratch,
            &ws.fixup_scratch,
        )?;
        self.projection_matmul_rows_device_to(
            &format!("{prefix}.ffn_up.weight"),
            ws.normed.as_f32(),
            ws.mmq_rows,
            self.inventory.hidden_size,
            ws.ffn_up.as_f32(),
            self.inventory.feed_forward_size,
            &mut q8_dtype,
            &ws.q8_scratch,
            &ws.fixup_scratch,
        )?;
        check(
            unsafe {
                gp_qwen_swiglu(
                    ws.ffn_gate.as_f32(),
                    ws.ffn_up.as_f32(),
                    ws.ffn_gate.as_f32(),
                    checked_i32(rows * self.inventory.feed_forward_size, "SwiGLU rows total")?,
                    self.dev.stream(),
                )
            },
            "qwen35 cuda row FFN SwiGLU",
        )?;
        self.matmul_rows_device_to(
            &format!("{prefix}.ffn_down.weight"),
            ws.ffn_gate.as_f32(),
            ws.mmq_rows,
            self.inventory.feed_forward_size,
            ws.attn_out.as_f32(),
            self.inventory.hidden_size,
            &ws.q8_scratch,
            &ws.fixup_scratch,
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
        self.matvec_q8_device_to(
            &format!("{prefix}.attn_qkv.weight"),
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
        self.matvec_q8_device_to(
            &format!("{prefix}.attn_gate.weight"),
            self.inventory.hidden_size,
            ws.z.as_f32(),
            inner,
            &ws.q8_scratch,
        )?;
        self.projection_matvec_device_to(
            &format!("{prefix}.ssm_beta.weight"),
            ws.normed.as_f32(),
            self.inventory.hidden_size,
            ws.beta.as_f32(),
            heads,
            &ws.q8_scratch,
        )?;
        self.projection_matvec_device_to(
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

    fn delta_attention_block_rows_device(
        &self,
        layer: usize,
        recurrent: &DeviceBuffer,
        conv: &DeviceBuffer,
        rows: usize,
        ws: &mut CudaPrefillWorkspace,
    ) -> Result<()> {
        let prefix = format!("blk.{layer}");
        let inner = self.inventory.ssm_inner_size;
        let heads = self.inventory.ssm_group_count;
        let head_dim = inner / heads;
        let qkv_stride = inner * 3;
        let mut q8_dtype = None;
        self.projection_matmul_rows_device_to(
            &format!("{prefix}.attn_qkv.weight"),
            ws.normed.as_f32(),
            ws.mmq_rows,
            self.inventory.hidden_size,
            ws.qkv.as_f32(),
            inner * 3,
            &mut q8_dtype,
            &ws.q8_scratch,
            &ws.fixup_scratch,
        )?;
        let conv_weight = self
            .weights
            .require(&format!("{prefix}.ssm_conv1d.weight"))?;
        if conv_weight.dtype != GgmlDType::F32
            || conv_weight.shape.as_slice() != [inner * 3, CONV_KERNEL]
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
                gp_qwen_causal_conv1d_silu_rows_parallel(
                    ws.qkv.as_f32(),
                    conv_weight.buffer.as_f32(),
                    conv.as_f32(),
                    ws.conv_out.as_f32(),
                    checked_i32(rows, "Delta rows")?,
                    checked_i32(inner * 3, "Delta conv channels")?,
                    checked_i32(CONV_KERNEL, "Delta conv kernel")?,
                    self.dev.stream(),
                )
            },
            &format!("qwen35 cuda row causal_conv1d_silu {prefix}"),
        )?;
        self.projection_matmul_rows_device_to(
            &format!("{prefix}.attn_gate.weight"),
            ws.normed.as_f32(),
            ws.mmq_rows,
            self.inventory.hidden_size,
            ws.z.as_f32(),
            inner,
            &mut q8_dtype,
            &ws.q8_scratch,
            &ws.fixup_scratch,
        )?;
        self.projection_matmul_rows_device_to(
            &format!("{prefix}.ssm_beta.weight"),
            ws.normed.as_f32(),
            ws.mmq_rows,
            self.inventory.hidden_size,
            ws.beta.as_f32(),
            heads,
            &mut q8_dtype,
            &ws.q8_scratch,
            &ws.fixup_scratch,
        )?;
        self.projection_matmul_rows_device_to(
            &format!("{prefix}.ssm_alpha.weight"),
            ws.normed.as_f32(),
            ws.mmq_rows,
            self.inventory.hidden_size,
            ws.alpha.as_f32(),
            self.inventory.ssm_time_step_rank,
            &mut q8_dtype,
            &ws.q8_scratch,
            &ws.fixup_scratch,
        )?;

        let q = ws.conv_out.as_f32();
        let k = unsafe { ws.conv_out.as_f32().add(inner) };
        let v = unsafe { ws.conv_out.as_f32().add(inner * 2) };
        check(
            unsafe {
                gp_qwen_normalize_linear_qk_rows(
                    q,
                    k,
                    checked_i32(rows, "Delta qk rows")?,
                    checked_i32(heads, "Delta heads")?,
                    checked_i32(head_dim, "Delta head dim")?,
                    checked_i32(qkv_stride, "Delta q stride")?,
                    checked_i32(qkv_stride, "Delta k stride")?,
                    RMS_EPS,
                    self.dev.stream(),
                )
            },
            "qwen35 cuda row normalize_linear_qk",
        )?;
        let a_log = self.weights.require(&format!("{prefix}.ssm_a"))?;
        let dt_bias = self.weights.require(&format!("{prefix}.ssm_dt.bias"))?;
        check(
            unsafe {
                gp_qwen_deltanet_decode_rows(
                    q,
                    k,
                    v,
                    ws.beta.as_f32(),
                    ws.alpha.as_f32(),
                    a_log.buffer.as_f32(),
                    dt_bias.buffer.as_f32(),
                    recurrent.as_f32(),
                    ws.raw.as_f32(),
                    checked_i32(rows, "Delta rows")?,
                    checked_i32(heads, "Delta heads")?,
                    checked_i32(head_dim, "Delta head dim")?,
                    checked_i32(qkv_stride, "Delta q stride")?,
                    checked_i32(qkv_stride, "Delta k stride")?,
                    checked_i32(qkv_stride, "Delta v stride")?,
                    checked_i32(heads, "Delta beta stride")?,
                    checked_i32(self.inventory.ssm_time_step_rank, "Delta alpha stride")?,
                    checked_i32(inner, "Delta output stride")?,
                    self.dev.stream(),
                )
            },
            &format!("qwen35 cuda row deltanet layer {layer}"),
        )?;
        self.rms_norm_device(
            &format!("{prefix}.ssm_norm.weight"),
            ws.raw.as_f32(),
            ws.raw.as_f32(),
            rows * heads,
            head_dim,
            false,
        )?;
        check(
            unsafe {
                gp_qwen_apply_silu_gate(
                    ws.raw.as_f32(),
                    ws.z.as_f32(),
                    checked_i32(rows * inner, "Delta row gate total")?,
                    self.dev.stream(),
                )
            },
            "qwen35 cuda row Delta gate",
        )?;
        self.matmul_rows_device_to(
            &format!("{prefix}.ssm_out.weight"),
            ws.raw.as_f32(),
            ws.mmq_rows,
            inner,
            ws.attn_out.as_f32(),
            self.inventory.hidden_size,
            &ws.q8_scratch,
            &ws.fixup_scratch,
        )
    }

    fn full_attention_block_device(
        &self,
        layer: usize,
        k_cache: &DeviceBuffer,
        v_cache: &DeviceBuffer,
        position: usize,
        position_ptr: Option<*const i32>,
        ws: &mut CudaForwardWorkspace,
    ) -> Result<()> {
        let prefix = format!("blk.{layer}");
        let q_dim = self.inventory.attention_heads * self.inventory.head_dim;
        let kv_k_dim = self.inventory.kv_heads * self.inventory.head_dim;
        let kv_v_dim = self.inventory.kv_heads * self.inventory.value_dim;
        let max_context =
            state_context_len(k_cache, self.inventory.kv_heads, self.inventory.head_dim);
        self.matvec_q8_device_to(
            &format!("{prefix}.attn_q.weight"),
            self.inventory.hidden_size,
            ws.q_fused.as_f32(),
            q_dim * 2,
            &ws.q8_scratch,
        )?;
        self.matvec_q8_device_to(
            &format!("{prefix}.attn_k.weight"),
            self.inventory.hidden_size,
            ws.k.as_f32(),
            kv_k_dim,
            &ws.q8_scratch,
        )?;
        self.matvec_q8_device_to(
            &format!("{prefix}.attn_v.weight"),
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
        let q_rope = unsafe {
            if let Some(position_ptr) = position_ptr {
                gp_qwen_rope_decode_position(
                    ws.qkv.as_f32(),
                    checked_i32(self.inventory.attention_heads, "attention heads")?,
                    checked_i32(self.inventory.head_dim, "attention head dim")?,
                    checked_i32(self.inventory.rope_dim, "attention rope dim")?,
                    position_ptr,
                    ROPE_THETA,
                    self.dev.stream(),
                )
            } else {
                gp_qwen_rope_decode(
                    ws.qkv.as_f32(),
                    checked_i32(self.inventory.attention_heads, "attention heads")?,
                    checked_i32(self.inventory.head_dim, "attention head dim")?,
                    checked_i32(self.inventory.rope_dim, "attention rope dim")?,
                    checked_i32(position, "attention position")?,
                    ROPE_THETA,
                    self.dev.stream(),
                )
            }
        };
        check(q_rope, "qwen35 cuda q RoPE")?;
        let k_rope = unsafe {
            if let Some(position_ptr) = position_ptr {
                gp_qwen_rope_decode_position(
                    ws.k.as_f32(),
                    checked_i32(self.inventory.kv_heads, "kv heads")?,
                    checked_i32(self.inventory.head_dim, "kv head dim")?,
                    checked_i32(self.inventory.rope_dim, "kv rope dim")?,
                    position_ptr,
                    ROPE_THETA,
                    self.dev.stream(),
                )
            } else {
                gp_qwen_rope_decode(
                    ws.k.as_f32(),
                    checked_i32(self.inventory.kv_heads, "kv heads")?,
                    checked_i32(self.inventory.head_dim, "kv head dim")?,
                    checked_i32(self.inventory.rope_dim, "kv rope dim")?,
                    checked_i32(position, "kv position")?,
                    ROPE_THETA,
                    self.dev.stream(),
                )
            }
        };
        check(k_rope, "qwen35 cuda k RoPE")?;
        let k_write = unsafe {
            if let Some(position_ptr) = position_ptr {
                gp_qwen_cache_write_position(
                    ws.k.as_f32(),
                    k_cache.as_f32(),
                    position_ptr,
                    checked_i32(self.inventory.kv_heads, "kv heads")?,
                    checked_i32(self.inventory.head_dim, "kv head dim")?,
                    checked_i32(max_context, "k cache context")?,
                    self.dev.stream(),
                )
            } else {
                gp_qwen_cache_write(
                    ws.k.as_f32(),
                    k_cache.as_f32(),
                    checked_i32(position, "k cache position")?,
                    checked_i32(self.inventory.kv_heads, "kv heads")?,
                    checked_i32(self.inventory.head_dim, "kv head dim")?,
                    checked_i32(max_context, "k cache context")?,
                    self.dev.stream(),
                )
            }
        };
        check(k_write, "qwen35 cuda k cache write")?;
        let v_write = unsafe {
            if let Some(position_ptr) = position_ptr {
                gp_qwen_cache_write_position(
                    ws.v.as_f32(),
                    v_cache.as_f32(),
                    position_ptr,
                    checked_i32(self.inventory.kv_heads, "kv heads")?,
                    checked_i32(self.inventory.value_dim, "value dim")?,
                    checked_i32(max_context, "v cache context")?,
                    self.dev.stream(),
                )
            } else {
                gp_qwen_cache_write(
                    ws.v.as_f32(),
                    v_cache.as_f32(),
                    checked_i32(position, "v cache position")?,
                    checked_i32(self.inventory.kv_heads, "kv heads")?,
                    checked_i32(self.inventory.value_dim, "value dim")?,
                    checked_i32(max_context, "v cache context")?,
                    self.dev.stream(),
                )
            }
        };
        check(v_write, "qwen35 cuda v cache write")?;
        let scale = 1.0 / (self.inventory.head_dim as f32).sqrt();
        let score_attention = unsafe {
            if let Some(position_ptr) = position_ptr {
                gp_qwen_attention_scores_decode_position(
                    ws.qkv.as_f32(),
                    k_cache.as_f32(),
                    ws.scores.as_f32(),
                    position_ptr,
                    checked_i32(self.inventory.attention_heads, "attention heads")?,
                    checked_i32(self.inventory.kv_heads, "kv heads")?,
                    checked_i32(self.inventory.head_dim, "attention head dim")?,
                    checked_i32(max_context, "attention context")?,
                    scale,
                    self.dev.stream(),
                )
            } else {
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
            }
        };
        check(score_attention, "qwen35 cuda attention scores")?;
        let softmax_attention = unsafe {
            if let Some(position_ptr) = position_ptr {
                gp_qwen_softmax_decode_position(
                    ws.scores.as_f32(),
                    position_ptr,
                    checked_i32(self.inventory.attention_heads, "attention heads")?,
                    checked_i32(max_context, "attention context")?,
                    self.dev.stream(),
                )
            } else {
                gp_qwen_softmax_decode(
                    ws.scores.as_f32(),
                    checked_i32(position, "attention position")?,
                    checked_i32(self.inventory.attention_heads, "attention heads")?,
                    checked_i32(max_context, "attention context")?,
                    self.dev.stream(),
                )
            }
        };
        check(softmax_attention, "qwen35 cuda attention softmax")?;
        let value_attention = unsafe {
            if let Some(position_ptr) = position_ptr {
                gp_qwen_attention_values_decode_position(
                    ws.scores.as_f32(),
                    v_cache.as_f32(),
                    ws.raw.as_f32(),
                    position_ptr,
                    checked_i32(self.inventory.attention_heads, "attention heads")?,
                    checked_i32(self.inventory.kv_heads, "kv heads")?,
                    checked_i32(self.inventory.value_dim, "value dim")?,
                    checked_i32(max_context, "attention context")?,
                    self.dev.stream(),
                )
            } else {
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
            }
        };
        check(value_attention, "qwen35 cuda attention values")?;
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

    #[allow(clippy::too_many_arguments)]
    fn full_attention_block_rows_device(
        &self,
        layer: usize,
        k_cache: &DeviceBuffer,
        v_cache: &DeviceBuffer,
        position: usize,
        rows: usize,
        max_context: usize,
        ws: &mut CudaPrefillWorkspace,
    ) -> Result<()> {
        let prefix = format!("blk.{layer}");
        let q_dim = self.inventory.attention_heads * self.inventory.head_dim;
        let value_width = self.inventory.attention_heads * self.inventory.value_dim;
        let kv_k_dim = self.inventory.kv_heads * self.inventory.head_dim;
        let kv_v_dim = self.inventory.kv_heads * self.inventory.value_dim;
        if q_dim != value_width {
            return Err(Error::GenerationUnavailable(format!(
                "CUDA fused row attention requires q width {q_dim} == value width {value_width}"
            )));
        }
        let mut q8_dtype = None;
        self.projection_matmul_rows_device_to(
            &format!("{prefix}.attn_q.weight"),
            ws.normed.as_f32(),
            ws.mmq_rows,
            self.inventory.hidden_size,
            ws.q_fused.as_f32(),
            q_dim * 2,
            &mut q8_dtype,
            &ws.q8_scratch,
            &ws.fixup_scratch,
        )?;
        self.projection_matmul_rows_device_to(
            &format!("{prefix}.attn_k.weight"),
            ws.normed.as_f32(),
            ws.mmq_rows,
            self.inventory.hidden_size,
            ws.k.as_f32(),
            kv_k_dim,
            &mut q8_dtype,
            &ws.q8_scratch,
            &ws.fixup_scratch,
        )?;
        self.matmul_rows_device_to(
            &format!("{prefix}.attn_v.weight"),
            ws.normed.as_f32(),
            ws.mmq_rows,
            self.inventory.hidden_size,
            ws.v.as_f32(),
            kv_v_dim,
            &ws.q8_scratch,
            &ws.fixup_scratch,
        )?;
        check(
            unsafe {
                gp_qwen_deinterleave_q_gate(
                    ws.q_fused.as_f32(),
                    ws.q.as_f32(),
                    ws.gate.as_f32(),
                    checked_i32(rows, "attention rows")?,
                    checked_i32(self.inventory.attention_heads, "attention heads")?,
                    checked_i32(self.inventory.head_dim, "attention head dim")?,
                    checked_i32(q_dim * 2, "packed q stride")?,
                    checked_i32(q_dim, "q output stride")?,
                    self.dev.stream(),
                )
            },
            "qwen35 cuda row deinterleave q gate",
        )?;
        self.rms_norm_device(
            &format!("{prefix}.attn_q_norm.weight"),
            ws.q.as_f32(),
            ws.q.as_f32(),
            rows * self.inventory.attention_heads,
            self.inventory.head_dim,
            true,
        )?;
        self.rms_norm_device(
            &format!("{prefix}.attn_k_norm.weight"),
            ws.k.as_f32(),
            ws.k.as_f32(),
            rows * self.inventory.kv_heads,
            self.inventory.head_dim,
            true,
        )?;
        check(
            unsafe {
                gp_qwen_rope_rows(
                    ws.q.as_f32(),
                    checked_i32(rows, "attention rows")?,
                    checked_i32(self.inventory.attention_heads, "attention heads")?,
                    checked_i32(self.inventory.head_dim, "attention head dim")?,
                    checked_i32(self.inventory.rope_dim, "attention rope dim")?,
                    checked_i32(position, "attention position")?,
                    checked_i32(q_dim, "attention q stride")?,
                    ROPE_THETA,
                    self.dev.stream(),
                )
            },
            "qwen35 cuda row q RoPE",
        )?;
        check(
            unsafe {
                gp_qwen_rope_rows(
                    ws.k.as_f32(),
                    checked_i32(rows, "kv rows")?,
                    checked_i32(self.inventory.kv_heads, "kv heads")?,
                    checked_i32(self.inventory.head_dim, "kv head dim")?,
                    checked_i32(self.inventory.rope_dim, "kv rope dim")?,
                    checked_i32(position, "kv position")?,
                    checked_i32(kv_k_dim, "kv row stride")?,
                    ROPE_THETA,
                    self.dev.stream(),
                )
            },
            "qwen35 cuda row k RoPE",
        )?;
        check(
            unsafe {
                gp_qwen_cache_write_rows(
                    ws.k.as_f32(),
                    k_cache.as_f32(),
                    checked_i32(rows, "k cache rows")?,
                    checked_i32(position, "k cache position")?,
                    checked_i32(self.inventory.kv_heads, "kv heads")?,
                    checked_i32(self.inventory.head_dim, "kv head dim")?,
                    checked_i32(max_context, "k cache context")?,
                    checked_i32(kv_k_dim, "k cache src stride")?,
                    self.dev.stream(),
                )
            },
            "qwen35 cuda row k cache write",
        )?;
        check(
            unsafe {
                gp_qwen_cache_write_rows(
                    ws.v.as_f32(),
                    v_cache.as_f32(),
                    checked_i32(rows, "v cache rows")?,
                    checked_i32(position, "v cache position")?,
                    checked_i32(self.inventory.kv_heads, "kv heads")?,
                    checked_i32(self.inventory.value_dim, "value dim")?,
                    checked_i32(max_context, "v cache context")?,
                    checked_i32(kv_v_dim, "v cache src stride")?,
                    self.dev.stream(),
                )
            },
            "qwen35 cuda row v cache write",
        )?;
        check(
            unsafe {
                gp_qwen_attention_rows_fused(
                    ws.q.as_f32(),
                    k_cache.as_f32(),
                    v_cache.as_f32(),
                    ws.raw.as_f32(),
                    checked_i32(rows, "attention rows")?,
                    checked_i32(position, "attention position")?,
                    checked_i32(self.inventory.attention_heads, "attention heads")?,
                    checked_i32(self.inventory.kv_heads, "kv heads")?,
                    checked_i32(self.inventory.head_dim, "attention head dim")?,
                    checked_i32(self.inventory.value_dim, "attention value dim")?,
                    checked_i32(max_context, "attention context")?,
                    checked_i32(q_dim, "attention q stride")?,
                    self.dev.stream(),
                )
            },
            "qwen35 cuda fused row attention",
        )?;
        check(
            unsafe {
                gp_qwen_apply_sigmoid_gate(
                    ws.raw.as_f32(),
                    ws.gate.as_f32(),
                    checked_i32(rows * value_width, "attention row gate total")?,
                    self.dev.stream(),
                )
            },
            "qwen35 cuda row attention gate",
        )?;
        self.matmul_rows_device_to(
            &format!("{prefix}.attn_output.weight"),
            ws.raw.as_f32(),
            ws.mmq_rows,
            value_width,
            ws.attn_out.as_f32(),
            self.inventory.hidden_size,
            &ws.q8_scratch,
            &ws.fixup_scratch,
        )
    }

    #[cfg(test)]
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
        self.matvec_q8_device_to(
            &format!("{prefix}.attn_k.weight"),
            self.inventory.hidden_size,
            ws.k.as_f32(),
            kv_k_dim,
            &ws.q8_scratch,
        )?;
        self.matvec_q8_device_to(
            &format!("{prefix}.attn_v.weight"),
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

    #[allow(clippy::too_many_arguments)]
    fn full_attention_cache_only_rows_device(
        &self,
        layer: usize,
        k_cache: &DeviceBuffer,
        v_cache: &DeviceBuffer,
        position: usize,
        rows: usize,
        max_context: usize,
        ws: &mut CudaPrefillWorkspace,
    ) -> Result<()> {
        let prefix = format!("blk.{layer}");
        let kv_k_dim = self.inventory.kv_heads * self.inventory.head_dim;
        let kv_v_dim = self.inventory.kv_heads * self.inventory.value_dim;
        self.matmul_rows_device_to(
            &format!("{prefix}.attn_k.weight"),
            ws.normed.as_f32(),
            ws.mmq_rows,
            self.inventory.hidden_size,
            ws.k.as_f32(),
            kv_k_dim,
            &ws.q8_scratch,
            &ws.fixup_scratch,
        )?;
        self.matmul_rows_device_to(
            &format!("{prefix}.attn_v.weight"),
            ws.normed.as_f32(),
            ws.mmq_rows,
            self.inventory.hidden_size,
            ws.v.as_f32(),
            kv_v_dim,
            &ws.q8_scratch,
            &ws.fixup_scratch,
        )?;
        self.rms_norm_device(
            &format!("{prefix}.attn_k_norm.weight"),
            ws.k.as_f32(),
            ws.k.as_f32(),
            rows * self.inventory.kv_heads,
            self.inventory.head_dim,
            true,
        )?;
        check(
            unsafe {
                gp_qwen_rope_rows(
                    ws.k.as_f32(),
                    checked_i32(rows, "kv rows")?,
                    checked_i32(self.inventory.kv_heads, "kv heads")?,
                    checked_i32(self.inventory.head_dim, "kv head dim")?,
                    checked_i32(self.inventory.rope_dim, "kv rope dim")?,
                    checked_i32(position, "kv position")?,
                    checked_i32(kv_k_dim, "kv stride")?,
                    ROPE_THETA,
                    self.dev.stream(),
                )
            },
            "qwen35 cuda cache-only row k RoPE",
        )?;
        check(
            unsafe {
                gp_qwen_cache_write_rows(
                    ws.k.as_f32(),
                    k_cache.as_f32(),
                    checked_i32(rows, "k cache rows")?,
                    checked_i32(position, "k cache position")?,
                    checked_i32(self.inventory.kv_heads, "kv heads")?,
                    checked_i32(self.inventory.head_dim, "kv head dim")?,
                    checked_i32(max_context, "k cache context")?,
                    checked_i32(kv_k_dim, "k cache src stride")?,
                    self.dev.stream(),
                )
            },
            "qwen35 cuda cache-only row k cache write",
        )?;
        check(
            unsafe {
                gp_qwen_cache_write_rows(
                    ws.v.as_f32(),
                    v_cache.as_f32(),
                    checked_i32(rows, "v cache rows")?,
                    checked_i32(position, "v cache position")?,
                    checked_i32(self.inventory.kv_heads, "kv heads")?,
                    checked_i32(self.inventory.value_dim, "value dim")?,
                    checked_i32(max_context, "v cache context")?,
                    checked_i32(kv_v_dim, "v cache src stride")?,
                    self.dev.stream(),
                )
            },
            "qwen35 cuda cache-only row v cache write",
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

fn mtp_conditioning_rows(
    previous_hidden: &[f32],
    committed_hidden: &[f32],
    rows: usize,
    hidden_size: usize,
) -> Result<Vec<f32>> {
    if previous_hidden.len() != hidden_size {
        return Err(Error::InvalidRequest(format!(
            "MTP previous hidden length {}, expected {}",
            previous_hidden.len(),
            hidden_size
        )));
    }
    if committed_hidden.len() != rows * hidden_size {
        return Err(Error::InvalidRequest(format!(
            "MTP committed hidden length {}, expected {}x{}",
            committed_hidden.len(),
            rows,
            hidden_size
        )));
    }
    let mut conditioning = vec![0.0f32; committed_hidden.len()];
    if rows == 0 {
        return Ok(conditioning);
    }
    conditioning[..hidden_size].copy_from_slice(previous_hidden);
    if rows > 1 {
        conditioning[hidden_size..].copy_from_slice(&committed_hidden[..(rows - 1) * hidden_size]);
    }
    Ok(conditioning)
}

fn last_hidden_row(hidden: &[f32], rows: usize, hidden_size: usize) -> Result<&[f32]> {
    if rows == 0 || hidden.len() != rows * hidden_size {
        return Err(Error::InvalidRequest(format!(
            "hidden row layout length {}, expected non-empty {}x{}",
            hidden.len(),
            rows,
            hidden_size
        )));
    }
    Ok(&hidden[(rows - 1) * hidden_size..rows * hidden_size])
}

const RMS_EPS: f32 = 1.0e-6;
const ROPE_THETA: f32 = 10_000_000.0;
const CONV_KERNEL: usize = 4;
const MMQ_FIXUP_SCRATCH_BYTES: usize = 128 * 128 * 128 * std::mem::size_of::<f32>();
const CUDA_PREFILL_BATCH_ROWS: usize = 512;
const CUDA_MMQ_ROW_TILE: usize = 128;
const MTP_DRAFT_MAX: usize = 2;

fn mmq_activation_layout(dtype: GgmlDType) -> Result<u8> {
    match dtype {
        GgmlDType::Q4K | GgmlDType::Q5K => Ok(1),
        GgmlDType::Q6K | GgmlDType::Q8_0 | GgmlDType::Q5_0 => Ok(0),
        other => Err(Error::GenerationUnavailable(format!(
            "CUDA MMQ activation reuse does not support {other}"
        ))),
    }
}

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
    fn qwen35_cuda_mixed_projection_rows_match_cpu_when_env_set() {
        let Some(path) = std::env::var_os("QWEN35_NATIVE_MTP_GGUF") else {
            return;
        };
        let gguf = GgufModel::open(&path).expect("open Qwen3.5 MTP GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 MTP inventory");
        let cuda = CudaQwen35Model::from_gguf(&gguf, inventory.clone(), 248_044)
            .expect("CUDA Qwen MTP model");
        let rows = 4;
        let workspace = cuda
            .new_prefill_workspace(CUDA_MMQ_ROW_TILE)
            .expect("CUDA mixed projection workspace");
        let input = patterned_input(rows * inventory.hidden_size);
        cuda.dev
            .copy_h2d(&workspace.normed, &input)
            .expect("upload mixed projection rows");
        let projections = [
            (
                "blk.0.attn_qkv.weight",
                workspace.qkv.as_f32(),
                inventory.ssm_inner_size * 3,
            ),
            (
                "blk.0.attn_gate.weight",
                workspace.z.as_f32(),
                inventory.ssm_inner_size,
            ),
            (
                "blk.0.ssm_beta.weight",
                workspace.beta.as_f32(),
                inventory.ssm_group_count,
            ),
            (
                "blk.0.ssm_alpha.weight",
                workspace.alpha.as_f32(),
                inventory.ssm_time_step_rank,
            ),
        ];
        let mut q8_dtype = None;
        for (name, output, output_rows) in projections {
            cuda.projection_matmul_rows_device_to(
                name,
                workspace.normed.as_f32(),
                workspace.mmq_rows,
                inventory.hidden_size,
                output,
                output_rows,
                &mut q8_dtype,
                &workspace.q8_scratch,
                &workspace.fixup_scratch,
            )
            .expect("CUDA mixed projection rows");
            let matrix = QuantMatrix::from_model(&gguf, name).expect("CPU mixed projection");
            let expected = matrix
                .matmul(&input, rows)
                .expect("CPU mixed projection rows");
            let mut actual = vec![0.0f32; rows * output_rows];
            let output_buffer = match name {
                "blk.0.attn_qkv.weight" => &workspace.qkv,
                "blk.0.attn_gate.weight" => &workspace.z,
                "blk.0.ssm_beta.weight" => &workspace.beta,
                "blk.0.ssm_alpha.weight" => &workspace.alpha,
                _ => unreachable!(),
            };
            cuda.dev
                .copy_d2h(&mut actual, output_buffer)
                .expect("read mixed projection rows");
            let similarity = cosine(&actual, &expected);
            eprintln!("CUDA mixed rows {name} cosine={similarity:.8}");
            assert!(
                similarity >= 0.999,
                "CUDA mixed row projection drift for {name}"
            );
        }
    }

    #[test]
    fn qwen35_cuda_deltanet_rows_match_tokenwise_when_env_set() {
        let Some(path) = std::env::var_os("QWEN35_NATIVE_MTP_GGUF") else {
            return;
        };
        let gguf = GgufModel::open(&path).expect("open Qwen3.5 MTP GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 MTP inventory");
        let cuda = CudaQwen35Model::from_gguf(&gguf, inventory.clone(), 248_044)
            .expect("CUDA Qwen MTP model");
        let rows = 5;
        let inner = inventory.ssm_inner_size;
        let qkv_stride = inner * 3;
        let heads = inventory.ssm_group_count;
        let head_dim = inner / heads;
        let qkv = patterned_input(rows * qkv_stride)
            .into_iter()
            .map(|value| value * 0.125)
            .collect::<Vec<_>>();
        let beta = patterned_input(rows * heads)
            .into_iter()
            .map(|value| value * 0.25)
            .collect::<Vec<_>>();
        let alpha = patterned_input(rows * heads)
            .into_iter()
            .map(|value| value * 0.125)
            .collect::<Vec<_>>();
        let qkv_device = cuda.upload_pod(&qkv).expect("upload DeltaNet rows QKV");
        let beta_tokenwise = cuda.upload_pod(&beta).expect("upload tokenwise beta");
        let alpha_tokenwise = cuda.upload_pod(&alpha).expect("upload tokenwise alpha");
        let beta_batched = cuda.upload_pod(&beta).expect("upload batched beta");
        let alpha_batched = cuda.upload_pod(&alpha).expect("upload batched alpha");
        let tokenwise_state = cuda
            .new_f32(heads * head_dim * head_dim)
            .expect("tokenwise DeltaNet state");
        let batched_state = cuda
            .new_f32(heads * head_dim * head_dim)
            .expect("batched DeltaNet state");
        let tokenwise_out = cuda
            .new_f32(rows * inner)
            .expect("tokenwise DeltaNet output");
        let batched_out = cuda.new_f32(rows * inner).expect("batched DeltaNet output");
        let a_log = cuda.weights.require("blk.0.ssm_a").expect("DeltaNet A");
        let dt_bias = cuda
            .weights
            .require("blk.0.ssm_dt.bias")
            .expect("DeltaNet dt bias");
        for row in 0..rows {
            check(
                unsafe {
                    gp_qwen_deltanet_decode(
                        qkv_device.as_f32().add(row * qkv_stride),
                        qkv_device.as_f32().add(row * qkv_stride + inner),
                        qkv_device.as_f32().add(row * qkv_stride + inner * 2),
                        beta_tokenwise.as_f32().add(row * heads),
                        alpha_tokenwise.as_f32().add(row * heads),
                        a_log.buffer.as_f32(),
                        dt_bias.buffer.as_f32(),
                        tokenwise_state.as_f32(),
                        tokenwise_out.as_f32().add(row * inner),
                        checked_i32(heads, "test DeltaNet heads").expect("heads fit i32"),
                        checked_i32(head_dim, "test DeltaNet head dim").expect("head dim fits i32"),
                        cuda.dev.stream(),
                    )
                },
                "tokenwise DeltaNet test row",
            )
            .expect("tokenwise DeltaNet row");
        }
        check(
            unsafe {
                gp_qwen_deltanet_decode_rows(
                    qkv_device.as_f32(),
                    qkv_device.as_f32().add(inner),
                    qkv_device.as_f32().add(inner * 2),
                    beta_batched.as_f32(),
                    alpha_batched.as_f32(),
                    a_log.buffer.as_f32(),
                    dt_bias.buffer.as_f32(),
                    batched_state.as_f32(),
                    batched_out.as_f32(),
                    checked_i32(rows, "test DeltaNet rows").expect("rows fit i32"),
                    checked_i32(heads, "test DeltaNet heads").expect("heads fit i32"),
                    checked_i32(head_dim, "test DeltaNet head dim").expect("head dim fits i32"),
                    checked_i32(qkv_stride, "test DeltaNet Q stride").expect("stride fits i32"),
                    checked_i32(qkv_stride, "test DeltaNet K stride").expect("stride fits i32"),
                    checked_i32(qkv_stride, "test DeltaNet V stride").expect("stride fits i32"),
                    checked_i32(heads, "test DeltaNet beta stride").expect("stride fits i32"),
                    checked_i32(heads, "test DeltaNet alpha stride").expect("stride fits i32"),
                    checked_i32(inner, "test DeltaNet output stride").expect("stride fits i32"),
                    cuda.dev.stream(),
                )
            },
            "batched DeltaNet test rows",
        )
        .expect("batched DeltaNet rows");
        let mut tokenwise_values = vec![0.0f32; rows * inner];
        let mut batched_values = vec![0.0f32; rows * inner];
        cuda.dev
            .copy_d2h(&mut tokenwise_values, &tokenwise_out)
            .expect("read tokenwise DeltaNet output");
        cuda.dev
            .copy_d2h(&mut batched_values, &batched_out)
            .expect("read batched DeltaNet output");
        let output_cosine = cosine(&tokenwise_values, &batched_values);
        tokenwise_values.resize(heads * head_dim * head_dim, 0.0);
        batched_values.resize(tokenwise_values.len(), 0.0);
        cuda.dev
            .copy_d2h(&mut tokenwise_values, &tokenwise_state)
            .expect("read tokenwise DeltaNet state");
        cuda.dev
            .copy_d2h(&mut batched_values, &batched_state)
            .expect("read batched DeltaNet state");
        let state_cosine = cosine(&tokenwise_values, &batched_values);
        eprintln!("CUDA DeltaNet rows output_cos={output_cosine:.8} state_cos={state_cosine:.8}");
        assert!(output_cosine >= 0.99999, "batched DeltaNet output drift");
        assert!(state_cosine >= 0.99999, "batched DeltaNet state drift");
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
            cuda.quantize_matvec_input(ws.normed.as_f32(), inventory.hidden_size, &ws.q8_scratch)
                .expect("quantize full attention input");
            match &mut state.layer_states[layer] {
                CudaLayerState::Full { k_cache, v_cache } => cuda
                    .full_attention_block_device(layer, k_cache, v_cache, position, None, &mut ws)
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
        let mut greedy_state = cuda.new_forward_state(8).expect("CUDA greedy state");
        let mut greedy_ws = cuda
            .new_forward_workspace(8)
            .expect("CUDA greedy workspace");
        let mut input = 42;
        for step in 0..3 {
            let logits = cuda
                .forward_token_logits(input, &mut logits_state, &mut logits_ws)
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
            let actual = cuda
                .forward_token_greedy(input, &mut greedy_state, &mut greedy_ws)
                .expect("CUDA greedy token");
            assert_eq!(actual, expected, "greedy graph token drift at step {step}");
            input = actual;
        }
        assert_eq!(greedy_state.position, 3);
        assert!(
            greedy_state.decode_graph.is_some(),
            "CUDA greedy path did not retain a captured graph"
        );
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

    #[test]
    fn qwen35_cuda_batched_prefill_matches_tokenwise_when_env_set() {
        let Some(path) = std::env::var_os("QWEN35_NATIVE_GGUF") else {
            eprintln!("skipping qwen35 CUDA batched prefill parity: QWEN35_NATIVE_GGUF unset");
            return;
        };
        let gguf = GgufModel::open(&path).expect("open Qwen3.5 GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 inventory");
        inventory
            .validate_core_tensors(&gguf)
            .expect("Qwen3.5 core tensors");
        let mixed_mtp = inventory.has_mtp();
        let cuda = CudaQwen35Model::from_gguf(&gguf, inventory, 248_044).expect("CUDA Qwen model");
        let prompt = [42_u32, 314, 2718, 99];
        let final_token = 1234_u32;

        let mut tokenwise_state = cuda.new_forward_state(16).expect("tokenwise state");
        let mut tokenwise_ws = cuda.new_forward_workspace(16).expect("tokenwise workspace");
        for &token in &prompt {
            cuda.prefill_token(token, &mut tokenwise_state, &mut tokenwise_ws)
                .expect("tokenwise prefill");
        }
        let expected = cuda
            .forward_token_logits(final_token, &mut tokenwise_state, &mut tokenwise_ws)
            .expect("tokenwise final logits");

        let mut batched_state = cuda.new_forward_state(16).expect("batched state");
        cuda.prefill_tokens(&prompt, &mut batched_state)
            .expect("batched prefill");
        let mut batched_ws = cuda.new_forward_workspace(16).expect("batched workspace");
        let actual = cuda
            .forward_token_logits(final_token, &mut batched_state, &mut batched_ws)
            .expect("batched final logits");
        let cosine = cosine(&actual, &expected);
        let rms_rel = rms_diff(&actual, &expected) / rms_norm(&expected).max(1.0e-6);
        let max_abs = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let actual_argmax = actual
            .iter()
            .enumerate()
            .max_by(|(ai, a), (bi, b)| {
                a.partial_cmp(b)
                    .unwrap_or(std::cmp::Ordering::Less)
                    .then_with(|| bi.cmp(ai))
            })
            .map(|(idx, _)| idx)
            .expect("actual logits");
        let expected_argmax = expected
            .iter()
            .enumerate()
            .max_by(|(ai, a), (bi, b)| {
                a.partial_cmp(b)
                    .unwrap_or(std::cmp::Ordering::Less)
                    .then_with(|| bi.cmp(ai))
            })
            .map(|(idx, _)| idx)
            .expect("expected logits");
        eprintln!(
            "CUDA batched prefill parity cosine={cosine:.8} rms_rel={rms_rel:.6e} max_abs={max_abs:.6e} argmax={actual_argmax}/{expected_argmax}"
        );
        if mixed_mtp {
            // Batched MMQ and tokenwise MMVQ use different Q8 activation layouts. The mixed
            // Q6_K/Q4_K/F32 model amplifies their small projection difference through DeltaNet.
            // Product-path correctness is covered by the llama.cpp target-token golden test.
            assert!(cosine >= 0.95, "CUDA mixed batched prefill cosine too low");
            assert!(
                rms_rel <= 0.3,
                "CUDA mixed batched prefill rms_rel too high"
            );
        } else {
            assert!(cosine >= 0.995, "CUDA batched prefill cosine too low");
            assert!(rms_rel <= 0.1, "CUDA batched prefill rms_rel too high");
            assert_eq!(
                actual_argmax, expected_argmax,
                "CUDA batched prefill argmax drift"
            );
        }
    }

    #[test]
    fn qwen35_cuda_batched_hidden_matches_tokenwise_when_env_set() {
        let Some(path) = std::env::var_os("QWEN35_NATIVE_MTP_GGUF") else {
            return;
        };
        let gguf = GgufModel::open(&path).expect("open Qwen3.5 MTP GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 MTP inventory");
        let cuda =
            CudaQwen35Model::from_gguf(&gguf, inventory, 248_044).expect("CUDA Qwen MTP model");
        let prefix = [42_u32, 314, 2718, 99];
        let verify = [1234_u32, 5678, 9012];
        let max_context = prefix.len() + verify.len() + 1;

        let mut tokenwise_state = cuda
            .new_forward_state(max_context)
            .expect("CUDA tokenwise hidden state");
        cuda.prefill_tokens(&prefix, &mut tokenwise_state)
            .expect("CUDA tokenwise hidden prefix");
        let mut tokenwise_workspace = cuda
            .new_forward_workspace(max_context)
            .expect("CUDA tokenwise hidden workspace");
        let mut tokenwise_hidden = Vec::new();
        let mut tokenwise_logits = Vec::new();
        for token in verify {
            let output = cuda
                .forward_token_logits_hidden(token, &mut tokenwise_state, &mut tokenwise_workspace)
                .expect("CUDA tokenwise hidden row");
            tokenwise_hidden.extend(output.hidden);
            tokenwise_logits.extend(output.logits);
        }

        let mut batched_state = cuda
            .new_forward_state(max_context)
            .expect("CUDA batched hidden state");
        cuda.prefill_tokens(&prefix, &mut batched_state)
            .expect("CUDA batched hidden prefix");
        let batched = cuda
            .forward_tokens_logits_hidden(&verify, &mut batched_state)
            .expect("CUDA batched hidden rows");
        let hidden_cosine = cosine(&tokenwise_hidden, &batched.hidden);
        let logits_cosine = cosine(&tokenwise_logits, &batched.logits);
        eprintln!(
            "CUDA batched hidden: hidden_cos={hidden_cosine:.8} logits_cos={logits_cosine:.8}"
        );
        assert!(hidden_cosine >= 0.95, "CUDA batched hidden-state drift");
        assert!(logits_cosine >= 0.95, "CUDA batched hidden-logit drift");
    }

    #[test]
    fn qwen35_cuda_mtp_matches_golden_tokens_when_env_set() {
        let (Some(gguf_path), Some(tokenizer_path)) = (
            std::env::var_os("QWEN35_NATIVE_MTP_GGUF"),
            std::env::var_os("QWEN35_NATIVE_TOKENIZER"),
        ) else {
            return;
        };
        let gguf = GgufModel::open(&gguf_path).expect("open Qwen3.5 MTP GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 MTP inventory");
        let cuda = CudaQwen35Model::from_gguf(&gguf, inventory.clone(), 248_044)
            .expect("CUDA Qwen MTP model");
        let tokenizer = Tokenizer::from_file(tokenizer_path).expect("load Qwen3.5 tokenizer");
        let prompt = crate::prompt::brief_chat_prompt(
            "pub fn rename_by_rules(&mut self, rules: RenameAllRules) {\n    self.serialize.value = rules.serialize.apply_to_field(&self.serialize.value);\n    self.deserialize.value = rules.deserialize.apply_to_field(&self.deserialize.value);\n}",
        );
        let ids = tokenizer
            .encode(prompt, true)
            .expect("tokenize CUDA MTP golden prompt")
            .get_ids()
            .to_vec();
        let max_context = ids.len() + 3;
        let mut target_state = cuda
            .new_forward_state(max_context)
            .expect("CUDA MTP target state");
        let mut prompt_hidden = cuda
            .prefill_tokens_hidden(&ids[..ids.len() - 1], &mut target_state)
            .expect("CUDA MTP target prompt hidden");
        let mut target_workspace = cuda
            .new_forward_workspace(max_context)
            .expect("CUDA MTP target workspace");
        let target = cuda
            .forward_token_logits_hidden(
                *ids.last().expect("non-empty CUDA MTP prompt"),
                &mut target_state,
                &mut target_workspace,
            )
            .expect("CUDA MTP target final row");
        prompt_hidden.extend_from_slice(&target.hidden);
        let first = greedy_argmax(&target.logits);
        assert_eq!(first, 10296, "unexpected CUDA MTP target token");

        let hidden_size = inventory.hidden_size;
        let mut conditioning = vec![0.0f32; prompt_hidden.len()];
        conditioning[hidden_size..]
            .copy_from_slice(&prompt_hidden[..prompt_hidden.len() - hidden_size]);
        let mut mtp_state = cuda.new_mtp_state(max_context).expect("CUDA MTP state");
        cuda.mtp_prefill_tokens(&ids, &conditioning, &mut mtp_state)
            .expect("CUDA MTP prompt catch-up");
        let first_draft = cuda
            .mtp_forward_tokens_logits_hidden(&[first], &target.hidden, &mut mtp_state)
            .expect("CUDA MTP first draft");
        let first_draft_token = greedy_argmax(&first_draft.logits);
        let second_draft = cuda
            .mtp_forward_tokens_logits_hidden(
                &[first_draft_token],
                &first_draft.hidden,
                &mut mtp_state,
            )
            .expect("CUDA MTP second draft");
        let cpu = crate::cpu::CpuQwen35Model::load(&gguf, inventory, 248_044)
            .expect("CPU MTP comparison model");
        let mut cpu_mtp_state = cpu
            .new_mtp_state(max_context)
            .expect("CPU MTP comparison state");
        cpu.mtp_prefill_tokens(&ids, &conditioning, &mut cpu_mtp_state)
            .expect("CPU MTP comparison catch-up");
        let cpu_first = cpu
            .mtp_forward_tokens_logits_hidden(&[first], &target.hidden, &mut cpu_mtp_state)
            .expect("CPU MTP first comparison draft");
        let cpu_first_token = greedy_argmax(&cpu_first.logits);
        let cpu_second = cpu
            .mtp_forward_tokens_logits_hidden(
                &[cpu_first_token],
                &cpu_first.hidden,
                &mut cpu_mtp_state,
            )
            .expect("CPU MTP second comparison draft");
        let first_hidden_cosine = cosine(&first_draft.hidden, &cpu_first.hidden);
        let first_logits_cosine = cosine(&first_draft.logits, &cpu_first.logits);
        let second_logits_cosine = cosine(&second_draft.logits, &cpu_second.logits);
        eprintln!(
            "CUDA/CPU MTP first hidden_cos={first_hidden_cosine:.8} logits_cos={first_logits_cosine:.8}; second logits_cos={second_logits_cosine:.8}; tokens={first_draft_token}/{cpu_first_token}"
        );
        assert!(first_hidden_cosine >= 0.998);
        assert!(first_logits_cosine >= 0.998);
        assert!(second_logits_cosine >= 0.998);
        assert_eq!(
            [first_draft_token, greedy_argmax(&second_draft.logits)],
            [6976, 264],
            "CUDA MTP drafts differ from the finetuned-model golden tokens"
        );
    }

    #[test]
    fn qwen35_cuda_mtp_generation_matches_target_when_env_set() {
        let (Some(gguf_path), Some(tokenizer_path)) = (
            std::env::var_os("QWEN35_NATIVE_MTP_GGUF"),
            std::env::var_os("QWEN35_NATIVE_TOKENIZER"),
        ) else {
            return;
        };
        let gguf = GgufModel::open(&gguf_path).expect("open Qwen3.5 MTP GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 MTP inventory");
        let cuda =
            CudaQwen35Model::from_gguf(&gguf, inventory, 248_044).expect("CUDA Qwen MTP model");
        let tokenizer = Tokenizer::from_file(tokenizer_path).expect("load Qwen3.5 tokenizer");
        let prompt = crate::prompt::brief_chat_prompt(
            "pub fn add_user(users: &mut Vec<String>, name: &str) -> usize {\n    users.push(name.trim().to_string());\n    users.len()\n}",
        );
        let params = GenerationParams {
            max_tokens: 32,
            ..crate::BRIEF_GENERATION_PARAMS
        };
        let expected = cuda
            .generate_target_only_for_test(&tokenizer, &prompt, params)
            .expect("CUDA target-only generation");
        let actual = cuda
            .generate(&tokenizer, &prompt, params)
            .expect("CUDA MTP generation");
        assert_eq!(actual, expected, "CUDA MTP changed target sampling output");
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

    fn greedy_argmax(values: &[f32]) -> u32 {
        values
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| u32::try_from(index).expect("vocabulary index fits u32"))
            .expect("non-empty logits")
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
