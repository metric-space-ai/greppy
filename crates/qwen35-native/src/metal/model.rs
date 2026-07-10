//! Metal-resident Qwen3.5 forward building blocks.
//!
//! This is the clean-room Qwen Metal backend seed. It intentionally exposes
//! only verified primitive operations until the full token forward is ported.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use greppy_embed_native::metal::ffi::{global_device, Buffer, Device};
use greppy_embed_native::metal::ops::{self, GgmlType as OpType};
use greppy_embed_native::metal::tensor::GgmlType;
use greppy_embed_native::metal::weights::MetalWeights;
use greppy_embed_native::GgufModel;
use tokenizers::Tokenizer;

use crate::inventory::Qwen35Inventory;
use crate::sampler::{sample_token, GenerationParams, SamplerRng};
use crate::{Error, Result};

pub struct MetalQwen35Model {
    dev: &'static Device,
    weights: MetalWeights,
    inventory: Qwen35Inventory,
    eos_token_id: u32,
}

pub struct MetalForwardState {
    position: usize,
    max_context: usize,
    layer_states: Vec<MetalLayerState>,
}

pub(crate) struct MetalMtpState {
    position: usize,
    max_context: usize,
    k_cache: Buffer,
    v_cache: Buffer,
}

enum MetalLayerState {
    Delta { recurrent: Buffer, conv: Buffer },
    Full { k_cache: Buffer, v_cache: Buffer },
}

pub(crate) struct MetalForwardWorkspace {
    token_id: Buffer,
    hidden: Buffer,
    normed: Buffer,
    attn_out: Buffer,
    qkv: Buffer,
    z: Buffer,
    beta: Buffer,
    alpha: Buffer,
    raw: Buffer,
    q_fused: Buffer,
    k: Buffer,
    v: Buffer,
    ffn_gate: Buffer,
    ffn_up: Buffer,
    logits: Buffer,
    scores: Buffer,
}

pub(crate) struct MetalTargetForwardOutput {
    pub(crate) hidden: Vec<f32>,
    pub(crate) logits: Vec<f32>,
}

struct MetalPrefillWorkspace {
    token_ids: Buffer,
    hidden: Buffer,
    normed: Buffer,
    attn_out: Buffer,
    qkv: Buffer,
    z: Buffer,
    beta: Buffer,
    alpha: Buffer,
    raw: Buffer,
    q_fused: Buffer,
    k: Buffer,
    v: Buffer,
    ffn_gate: Buffer,
    ffn_up: Buffer,
    scores: Buffer,
}

const METAL_PREFILL_BATCH_ROWS: usize = 512;
const MTP_DRAFT_MAX: usize = 2;

impl MetalQwen35Model {
    pub fn from_gguf(
        model: &GgufModel,
        inventory: Qwen35Inventory,
        eos_token_id: u32,
    ) -> Result<Self> {
        let dev = global_device().ok_or_else(|| {
            Error::GenerationUnavailable("Metal device or metallib unavailable".into())
        })?;
        let weights = MetalWeights::load(dev, model)?;
        Ok(Self {
            dev,
            weights,
            inventory,
            eos_token_id,
        })
    }

    pub fn backend_name(&self) -> &'static str {
        "metal-q4k-forward"
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
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
        for chunk in prefill_ids.chunks(METAL_PREFILL_BATCH_ROWS) {
            self.prefill_tokens(chunk, &mut state)?;
        }

        let mut generated = Vec::new();
        let mut next = *prompt_ids.last().expect("checked non-empty above");
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
        let vocab_size = self.inventory.vocab_size;
        let mut perf = crate::MtpPerfTimer::new();
        let mut target_state = self.new_forward_state(max_context)?;
        let mut target_workspace = self.new_forward_workspace(max_context)?;
        let mut prompt_hidden = Vec::with_capacity(prompt_ids.len() * hidden_size);
        perf.begin_target_prefill();
        for tokens in prompt_ids[..prompt_ids.len() - 1].chunks(METAL_PREFILL_BATCH_ROWS) {
            prompt_hidden.extend(self.prefill_tokens_hidden(tokens, &mut target_state)?);
        }
        let mut target = self.forward_token_logits_hidden(
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
        for start in (0..prompt_ids.len()).step_by(METAL_PREFILL_BATCH_ROWS) {
            let end = (start + METAL_PREFILL_BATCH_ROWS).min(prompt_ids.len());
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
            eprintln!("qwen35-mtp-debug backend=metal first={first}");
        }
        let mut next = first;
        let mut pending_target_hidden = target.hidden;
        let mut drafted_total = 0usize;
        let mut accepted_total = 0usize;
        let mut cycles = 0usize;
        let mut mtp_fallback = false;
        let mut speculative_target_state = self.new_forward_state(max_context)?;
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
                let mut output = self.forward_token_logits_hidden(
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
                let mut output = self.forward_token_logits_hidden(
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
                    "qwen35-mtp-debug backend=metal cycle={cycles} input={next} drafts={draft_tokens:?}"
                );
            }

            let mut verification_tokens = Vec::with_capacity(1 + draft_tokens.len());
            verification_tokens.push(next);
            verification_tokens.extend_from_slice(&draft_tokens);
            let stage = perf.begin_stage();
            self.copy_forward_state(&target_state, &mut speculative_target_state)?;
            perf.finish_stage(crate::MtpPerfStage::TargetStateCopy, stage);
            let stage = perf.begin_stage();
            let mut verification = self.forward_tokens_logits_hidden(
                &verification_tokens,
                &mut speculative_target_state,
            )?;
            perf.finish_stage(crate::MtpPerfStage::TargetVerify, stage);
            let mut accepted = 0usize;
            let mut mismatch = None;
            let mut finished = false;
            for (draft_index, &draft_token) in draft_tokens.iter().enumerate() {
                let row_start = draft_index * vocab_size;
                let row_end = row_start + vocab_size;
                let Some(target_token) = sample_token(
                    &mut verification.logits[row_start..row_end],
                    &generated,
                    params,
                    &mut rng,
                ) else {
                    finished = true;
                    break;
                };
                if target_token == self.eos_token_id {
                    finished = true;
                    break;
                }
                if mtp_debug {
                    eprintln!(
                        "qwen35-mtp-debug backend=metal cycle={cycles} pos={draft_index} target={target_token} draft={draft_token}"
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
            }
            if finished {
                break;
            }
            mtp_fallback = crate::mtp_should_fallback(drafted_total, accepted_total);

            if accepted == draft_tokens.len() {
                let row_start = accepted * vocab_size;
                let row_end = row_start + vocab_size;
                let Some(target_token) = sample_token(
                    &mut verification.logits[row_start..row_end],
                    &generated,
                    params,
                    &mut rng,
                ) else {
                    break;
                };
                if target_token == self.eos_token_id {
                    break;
                }
                generated.push(target_token);
                next = target_token;
                std::mem::swap(&mut target_state, &mut speculative_target_state);
                pending_target_hidden =
                    last_hidden_row(&verification.hidden, verification_tokens.len(), hidden_size)?
                        .to_vec();
                let conditioning = mtp_conditioning_rows(
                    &previous_hidden,
                    &verification.hidden,
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
                let stage = perf.begin_stage();
                let committed_hidden =
                    self.prefill_tokens_hidden(commit_tokens, &mut target_state)?;
                perf.finish_stage(crate::MtpPerfStage::TargetReplay, stage);
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
                "qwen35-mtp-debug backend=metal cycles={cycles} drafted={drafted_total} accepted={accepted_total}"
            );
        }
        perf.report(
            "metal",
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

    pub fn new_forward_state(&self, max_context: usize) -> Result<MetalForwardState> {
        let mut layer_states = Vec::with_capacity(self.inventory.block_count);
        for layer in 0..self.inventory.block_count {
            if self.inventory.is_full_attention_layer(layer) {
                let k_elems = max_context * self.inventory.kv_heads * self.inventory.head_dim;
                let v_elems = max_context * self.inventory.kv_heads * self.inventory.value_dim;
                layer_states.push(MetalLayerState::Full {
                    k_cache: self.new_f32(k_elems)?,
                    v_cache: self.new_f32(v_elems)?,
                });
            } else {
                let head_dim = self.inventory.ssm_inner_size / self.inventory.ssm_group_count;
                layer_states.push(MetalLayerState::Delta {
                    recurrent: self
                        .new_f32(self.inventory.ssm_group_count * head_dim * head_dim)?,
                    conv: self.new_f32(self.inventory.ssm_inner_size * 3 * CONV_KERNEL)?,
                });
            }
        }
        Ok(MetalForwardState {
            position: 0,
            max_context,
            layer_states,
        })
    }

    pub(crate) fn new_forward_workspace(
        &self,
        max_context: usize,
    ) -> Result<MetalForwardWorkspace> {
        Ok(MetalForwardWorkspace {
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
            scores: self.new_f32(self.inventory.attention_heads * max_context)?,
        })
    }

    pub(crate) fn new_mtp_state(&self, max_context: usize) -> Result<MetalMtpState> {
        if !self.inventory.has_mtp() {
            return Err(Error::GenerationUnavailable(
                "Qwen3.5 model does not contain an MTP layer".into(),
            ));
        }
        Ok(MetalMtpState {
            position: 0,
            max_context,
            k_cache: self
                .new_f32(max_context * self.inventory.kv_heads * self.inventory.head_dim)?,
            v_cache: self
                .new_f32(max_context * self.inventory.kv_heads * self.inventory.value_dim)?,
        })
    }

    fn copy_forward_state(
        &self,
        src: &MetalForwardState,
        dst: &mut MetalForwardState,
    ) -> Result<()> {
        if src.max_context != dst.max_context || src.layer_states.len() != dst.layer_states.len() {
            return Err(Error::InvalidRequest(
                "Metal target state layouts do not match".into(),
            ));
        }
        let cb = self.command_buffer()?;
        let blit = cb.blit().ok_or_else(|| {
            Error::GenerationUnavailable("failed to create Metal blit encoder".into())
        })?;
        for (src_layer, dst_layer) in src.layer_states.iter().zip(&dst.layer_states) {
            match (src_layer, dst_layer) {
                (
                    MetalLayerState::Delta {
                        recurrent: src_recurrent,
                        conv: src_conv,
                    },
                    MetalLayerState::Delta {
                        recurrent: dst_recurrent,
                        conv: dst_conv,
                    },
                ) => {
                    blit.copy_buffer(src_recurrent, 0, dst_recurrent, 0, src_recurrent.len());
                    blit.copy_buffer(src_conv, 0, dst_conv, 0, src_conv.len());
                }
                (
                    MetalLayerState::Full {
                        k_cache: src_k,
                        v_cache: src_v,
                    },
                    MetalLayerState::Full {
                        k_cache: dst_k,
                        v_cache: dst_v,
                    },
                ) => {
                    blit.copy_buffer(src_k, 0, dst_k, 0, src_k.len());
                    blit.copy_buffer(src_v, 0, dst_v, 0, src_v.len());
                }
                _ => {
                    return Err(Error::InvalidRequest(
                        "Metal target state layer kinds do not match".into(),
                    ));
                }
            }
        }
        blit.end();
        cb.commit_and_wait().map_err(|error| {
            Error::GenerationUnavailable(format!("Metal target state copy failed: {error}"))
        })?;
        dst.position = src.position;
        Ok(())
    }

    fn copy_mtp_state(&self, src: &MetalMtpState, dst: &mut MetalMtpState) -> Result<()> {
        if src.max_context != dst.max_context {
            return Err(Error::InvalidRequest(
                "Metal MTP state layouts do not match".into(),
            ));
        }
        let cb = self.command_buffer()?;
        let blit = cb.blit().ok_or_else(|| {
            Error::GenerationUnavailable("failed to create Metal blit encoder".into())
        })?;
        blit.copy_buffer(&src.k_cache, 0, &dst.k_cache, 0, src.k_cache.len());
        blit.copy_buffer(&src.v_cache, 0, &dst.v_cache, 0, src.v_cache.len());
        blit.end();
        cb.commit_and_wait().map_err(|error| {
            Error::GenerationUnavailable(format!("Metal MTP state copy failed: {error}"))
        })?;
        dst.position = src.position;
        Ok(())
    }

    fn new_prefill_workspace(
        &self,
        rows: usize,
        max_context: usize,
    ) -> Result<MetalPrefillWorkspace> {
        let q_dim = self.inventory.attention_heads * self.inventory.head_dim;
        let kv_k_dim = self.inventory.kv_heads * self.inventory.head_dim;
        let kv_v_dim = self.inventory.kv_heads * self.inventory.value_dim;
        Ok(MetalPrefillWorkspace {
            token_ids: self.new_bytes(rows * std::mem::size_of::<u32>())?,
            hidden: self.new_f32(rows * self.inventory.hidden_size)?,
            normed: self.new_f32(rows * self.inventory.hidden_size)?,
            attn_out: self.new_f32(rows * self.inventory.hidden_size)?,
            qkv: self.new_f32(rows * self.inventory.ssm_inner_size * 3)?,
            z: self.new_f32(rows * self.inventory.ssm_inner_size)?,
            beta: self.new_f32(rows * self.inventory.ssm_group_count)?,
            alpha: self.new_f32(rows * self.inventory.ssm_time_step_rank)?,
            raw: self.new_f32(rows * self.inventory.ssm_inner_size.max(q_dim))?,
            q_fused: self.new_f32(rows * q_dim * 2)?,
            k: self.new_f32(rows * kv_k_dim)?,
            v: self.new_f32(rows * kv_v_dim)?,
            ffn_gate: self.new_f32(rows * self.inventory.feed_forward_size)?,
            ffn_up: self.new_f32(rows * self.inventory.feed_forward_size)?,
            scores: self.new_f32(rows * self.inventory.attention_heads * max_context)?,
        })
    }

    pub(crate) fn forward_token_logits(
        &self,
        token: u32,
        state: &mut MetalForwardState,
        ws: &mut MetalForwardWorkspace,
    ) -> Result<Vec<f32>> {
        let cb = self.command_buffer()?;
        let enc = cb
            .compute()
            .ok_or_else(|| Error::GenerationUnavailable("failed to create Metal encoder".into()))?;
        self.encode_token_hidden(&enc, token, state, ws, false)?;
        self.encode_rms_norm(
            &enc,
            "output_norm.weight",
            &ws.hidden,
            &ws.normed,
            1,
            self.inventory.hidden_size,
            true,
        )?;
        enc.memory_barrier_buffers();
        self.encode_matvec_to(
            &enc,
            "token_embd.weight",
            &ws.normed,
            self.inventory.hidden_size,
            &ws.logits,
            self.inventory.vocab_size,
        )?;
        enc.end();
        cb.commit_and_wait().map_err(|e| {
            Error::GenerationUnavailable(format!("Metal logits forward failed: {e}"))
        })?;
        state.position += 1;
        self.read_f32(&ws.logits, self.inventory.vocab_size)
    }

    pub(crate) fn forward_token_logits_hidden(
        &self,
        token: u32,
        state: &mut MetalForwardState,
        ws: &mut MetalForwardWorkspace,
    ) -> Result<MetalTargetForwardOutput> {
        let cb = self.command_buffer()?;
        let enc = cb
            .compute()
            .ok_or_else(|| Error::GenerationUnavailable("failed to create Metal encoder".into()))?;
        self.encode_token_hidden(&enc, token, state, ws, false)?;
        self.encode_rms_norm(
            &enc,
            "output_norm.weight",
            &ws.hidden,
            &ws.normed,
            1,
            self.inventory.hidden_size,
            true,
        )?;
        enc.memory_barrier_buffers();
        self.encode_matvec_to(
            &enc,
            "token_embd.weight",
            &ws.normed,
            self.inventory.hidden_size,
            &ws.logits,
            self.inventory.vocab_size,
        )?;
        enc.end();
        cb.commit_and_wait().map_err(|e| {
            Error::GenerationUnavailable(format!("Metal hidden/logits forward failed: {e}"))
        })?;
        state.position += 1;
        Ok(MetalTargetForwardOutput {
            hidden: self.read_f32(&ws.hidden, self.inventory.hidden_size)?,
            logits: self.read_f32(&ws.logits, self.inventory.vocab_size)?,
        })
    }

    pub(crate) fn forward_token_greedy(
        &self,
        token: u32,
        state: &mut MetalForwardState,
        ws: &mut MetalForwardWorkspace,
    ) -> Result<u32> {
        let cb = self.command_buffer()?;
        let enc = cb
            .compute()
            .ok_or_else(|| Error::GenerationUnavailable("failed to create Metal encoder".into()))?;
        self.encode_token_hidden(&enc, token, state, ws, false)?;
        self.encode_rms_norm(
            &enc,
            "output_norm.weight",
            &ws.hidden,
            &ws.normed,
            1,
            self.inventory.hidden_size,
            true,
        )?;
        enc.memory_barrier_buffers();
        self.encode_matvec_to(
            &enc,
            "token_embd.weight",
            &ws.normed,
            self.inventory.hidden_size,
            &ws.logits,
            self.inventory.vocab_size,
        )?;
        enc.memory_barrier_buffers();
        ok(ops::op_argmax(
            &enc,
            self.dev,
            &ws.logits,
            &ws.token_id,
            OpType::F32,
            checked_i64(self.inventory.vocab_size, "argmax vocab size")?,
            1,
            1,
            1,
            (self.inventory.vocab_size * std::mem::size_of::<f32>()) as u64,
        ))?;
        enc.end();
        cb.commit_and_wait().map_err(|e| {
            Error::GenerationUnavailable(format!("Metal greedy forward failed: {e}"))
        })?;
        state.position += 1;
        let mut token = [0_u32; 1];
        unsafe {
            ws.token_id.read(0, &mut token);
        }
        Ok(token[0])
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn prefill_token(
        &self,
        token: u32,
        state: &mut MetalForwardState,
        ws: &mut MetalForwardWorkspace,
    ) -> Result<()> {
        self.forward_token_hidden_device(token, state, ws, true)
    }

    pub(crate) fn prefill_tokens(
        &self,
        tokens: &[u32],
        state: &mut MetalForwardState,
    ) -> Result<()> {
        let _ = self.forward_tokens_hidden_device(tokens, state, true)?;
        Ok(())
    }

    pub(crate) fn prefill_tokens_hidden(
        &self,
        tokens: &[u32],
        state: &mut MetalForwardState,
    ) -> Result<Vec<f32>> {
        let workspace = self.forward_tokens_hidden_device(tokens, state, false)?;
        self.read_f32(&workspace.hidden, tokens.len() * self.inventory.hidden_size)
    }

    pub(crate) fn forward_tokens_logits_hidden(
        &self,
        tokens: &[u32],
        state: &mut MetalForwardState,
    ) -> Result<MetalTargetForwardOutput> {
        let workspace = self.forward_tokens_hidden_device(tokens, state, false)?;
        if tokens.is_empty() {
            return Ok(MetalTargetForwardOutput {
                hidden: Vec::new(),
                logits: Vec::new(),
            });
        }
        let rows = tokens.len();
        let logits = self.new_f32(rows * self.inventory.vocab_size)?;
        let cb = self.command_buffer()?;
        let enc = cb
            .compute_concurrent()
            .ok_or_else(|| Error::GenerationUnavailable("failed to create Metal encoder".into()))?;
        self.encode_rms_norm(
            &enc,
            "output_norm.weight",
            &workspace.hidden,
            &workspace.normed,
            rows,
            self.inventory.hidden_size,
            true,
        )?;
        enc.memory_barrier_buffers();
        self.encode_matmul_rows_to(
            &enc,
            "token_embd.weight",
            &workspace.normed,
            rows,
            self.inventory.hidden_size,
            &logits,
            self.inventory.vocab_size,
        )?;
        enc.end();
        cb.commit_and_wait().map_err(|e| {
            Error::GenerationUnavailable(format!("Metal batched hidden/logits forward failed: {e}"))
        })?;
        Ok(MetalTargetForwardOutput {
            hidden: self.read_f32(&workspace.hidden, rows * self.inventory.hidden_size)?,
            logits: self.read_f32(&logits, rows * self.inventory.vocab_size)?,
        })
    }

    pub(crate) fn mtp_prefill_tokens(
        &self,
        tokens: &[u32],
        target_hidden: &[f32],
        state: &mut MetalMtpState,
    ) -> Result<()> {
        let _ = self.mtp_forward_tokens(tokens, target_hidden, state, false)?;
        Ok(())
    }

    pub(crate) fn mtp_forward_tokens_logits_hidden(
        &self,
        tokens: &[u32],
        target_hidden: &[f32],
        state: &mut MetalMtpState,
    ) -> Result<MetalTargetForwardOutput> {
        self.mtp_forward_tokens(tokens, target_hidden, state, true)
    }

    fn mtp_forward_tokens(
        &self,
        tokens: &[u32],
        target_hidden: &[f32],
        state: &mut MetalMtpState,
        include_logits: bool,
    ) -> Result<MetalTargetForwardOutput> {
        if !self.inventory.has_mtp() {
            return Err(Error::GenerationUnavailable(
                "Qwen3.5 model does not contain an MTP layer".into(),
            ));
        }
        if tokens.is_empty() {
            if target_hidden.is_empty() {
                return Ok(MetalTargetForwardOutput {
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
        let workspace = self.new_prefill_workspace(rows, state.max_context)?;
        let target_hidden = self.upload_pod(target_hidden)?;
        let joined = self.new_f32(rows * hidden_size * 2)?;
        let logits = if include_logits {
            Some(self.new_f32(rows * self.inventory.vocab_size)?)
        } else {
            None
        };
        unsafe {
            workspace.token_ids.write(0, tokens);
        }
        let cb = self.command_buffer()?;
        let enc = cb
            .compute_concurrent()
            .ok_or_else(|| Error::GenerationUnavailable("failed to create Metal encoder".into()))?;
        self.encode_embed_tokens(&enc, &workspace.token_ids, &workspace.hidden, rows)?;
        enc.memory_barrier_buffers();
        self.encode_rms_norm(
            &enc,
            &format!("{prefix}.nextn.enorm.weight"),
            &workspace.hidden,
            &workspace.normed,
            rows,
            hidden_size,
            true,
        )?;
        self.encode_rms_norm(
            &enc,
            &format!("{prefix}.nextn.hnorm.weight"),
            &target_hidden,
            &workspace.attn_out,
            rows,
            hidden_size,
            true,
        )?;
        enc.memory_barrier_buffers();
        self.encode_concat_rows(
            &enc,
            &workspace.normed,
            &workspace.attn_out,
            &joined,
            rows,
            hidden_size,
        )?;
        enc.memory_barrier_buffers();
        self.encode_matmul_rows_to(
            &enc,
            &format!("{prefix}.nextn.eh_proj.weight"),
            &joined,
            rows,
            hidden_size * 2,
            &workspace.hidden,
            hidden_size,
        )?;
        enc.memory_barrier_buffers();
        self.encode_rms_norm(
            &enc,
            &format!("{prefix}.attn_norm.weight"),
            &workspace.hidden,
            &workspace.normed,
            rows,
            hidden_size,
            true,
        )?;
        enc.memory_barrier_buffers();
        self.encode_full_attention_block_rows(
            &enc,
            layer,
            &state.k_cache,
            &state.v_cache,
            state.position,
            rows,
            state.max_context,
            &workspace,
        )?;
        enc.memory_barrier_buffers();
        self.encode_add_rms_norm(
            &enc,
            &format!("{prefix}.post_attention_norm.weight"),
            &workspace.hidden,
            &workspace.attn_out,
            &workspace.hidden,
            &workspace.normed,
            rows,
            hidden_size,
        )?;
        enc.memory_barrier_buffers();
        self.encode_ffn_block_rows(&enc, layer, rows, &workspace)?;
        enc.memory_barrier_buffers();
        self.encode_add(
            &enc,
            &workspace.hidden,
            &workspace.attn_out,
            &workspace.hidden,
            rows * hidden_size,
        )?;
        if let Some(logits) = &logits {
            enc.memory_barrier_buffers();
            self.encode_rms_norm(
                &enc,
                &format!("{prefix}.nextn.shared_head_norm.weight"),
                &workspace.hidden,
                &workspace.normed,
                rows,
                hidden_size,
                true,
            )?;
            enc.memory_barrier_buffers();
            self.encode_matmul_rows_to(
                &enc,
                "token_embd.weight",
                &workspace.normed,
                rows,
                hidden_size,
                logits,
                self.inventory.vocab_size,
            )?;
        }
        enc.end();
        cb.commit_and_wait()
            .map_err(|e| Error::GenerationUnavailable(format!("Metal MTP forward failed: {e}")))?;
        state.position += rows;
        let output_hidden = if include_logits {
            &workspace.normed
        } else {
            &workspace.hidden
        };
        Ok(MetalTargetForwardOutput {
            hidden: self.read_f32(output_hidden, rows * hidden_size)?,
            logits: match logits {
                Some(logits) => self.read_f32(&logits, rows * self.inventory.vocab_size)?,
                None => Vec::new(),
            },
        })
    }

    fn forward_tokens_hidden_device(
        &self,
        tokens: &[u32],
        state: &mut MetalForwardState,
        cache_only_final: bool,
    ) -> Result<MetalPrefillWorkspace> {
        if tokens.is_empty() {
            return self.new_prefill_workspace(1, state.max_context);
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

        let ws = self.new_prefill_workspace(rows, state.max_context)?;
        unsafe {
            ws.token_ids.write(0, tokens);
        }
        let cb = self.command_buffer()?;
        let enc = cb
            .compute_concurrent()
            .ok_or_else(|| Error::GenerationUnavailable("failed to create Metal encoder".into()))?;
        self.encode_embed_tokens(&enc, &ws.token_ids, &ws.hidden, rows)?;
        enc.memory_barrier_buffers();

        for layer in 0..self.inventory.block_count {
            if self.encode_prefill_layer(&enc, layer, state, rows, &ws, cache_only_final)? {
                break;
            }
        }
        enc.end();
        cb.commit_and_wait().map_err(|e| {
            Error::GenerationUnavailable(format!("Metal batched prefill failed: {e}"))
        })?;
        state.position += rows;
        Ok(ws)
    }

    fn encode_prefill_layer(
        &self,
        enc: &greppy_embed_native::metal::ffi::ComputeEncoder,
        layer: usize,
        state: &mut MetalForwardState,
        rows: usize,
        ws: &MetalPrefillWorkspace,
        cache_only_final: bool,
    ) -> Result<bool> {
        self.encode_rms_norm(
            enc,
            &format!("blk.{layer}.attn_norm.weight"),
            &ws.hidden,
            &ws.normed,
            rows,
            self.inventory.hidden_size,
            true,
        )?;
        enc.memory_barrier_buffers();

        let final_layer = layer + 1 == self.inventory.block_count;
        match &mut state.layer_states[layer] {
            MetalLayerState::Delta { recurrent, conv } => {
                self.encode_delta_attention_block_rows(enc, layer, recurrent, conv, rows, ws)?;
            }
            MetalLayerState::Full { k_cache, v_cache } if final_layer && cache_only_final => {
                self.encode_full_attention_cache_only_rows(
                    enc,
                    layer,
                    k_cache,
                    v_cache,
                    state.position,
                    rows,
                    state.max_context,
                    ws,
                )?;
            }
            MetalLayerState::Full { k_cache, v_cache } => {
                self.encode_full_attention_block_rows(
                    enc,
                    layer,
                    k_cache,
                    v_cache,
                    state.position,
                    rows,
                    state.max_context,
                    ws,
                )?;
            }
        }
        if final_layer && cache_only_final {
            return Ok(true);
        }

        enc.memory_barrier_buffers();
        self.encode_add_rms_norm(
            enc,
            &format!("blk.{layer}.post_attention_norm.weight"),
            &ws.hidden,
            &ws.attn_out,
            &ws.hidden,
            &ws.normed,
            rows,
            self.inventory.hidden_size,
        )?;
        enc.memory_barrier_buffers();
        self.encode_ffn_block_rows(enc, layer, rows, ws)?;
        enc.memory_barrier_buffers();
        self.encode_add(
            enc,
            &ws.hidden,
            &ws.attn_out,
            &ws.hidden,
            rows * self.inventory.hidden_size,
        )?;
        enc.memory_barrier_buffers();
        Ok(false)
    }

    fn forward_token_hidden_device(
        &self,
        token: u32,
        state: &mut MetalForwardState,
        ws: &mut MetalForwardWorkspace,
        prefill: bool,
    ) -> Result<()> {
        let cb = self.command_buffer()?;
        let enc = cb
            .compute()
            .ok_or_else(|| Error::GenerationUnavailable("failed to create Metal encoder".into()))?;
        self.encode_token_hidden(&enc, token, state, ws, prefill)?;
        enc.end();
        cb.commit_and_wait().map_err(|e| {
            Error::GenerationUnavailable(format!("Metal hidden forward failed: {e}"))
        })?;
        state.position += 1;
        Ok(())
    }

    fn encode_token_hidden(
        &self,
        enc: &greppy_embed_native::metal::ffi::ComputeEncoder,
        token: u32,
        state: &mut MetalForwardState,
        ws: &MetalForwardWorkspace,
        prefill: bool,
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
        unsafe {
            ws.token_id.write(0, std::slice::from_ref(&token));
        }
        self.encode_embed_tokens(enc, &ws.token_id, &ws.hidden, 1)?;
        enc.memory_barrier_buffers();

        let final_layer = self.inventory.block_count.saturating_sub(1);
        for layer in 0..self.inventory.block_count {
            self.encode_rms_norm(
                enc,
                &format!("blk.{layer}.attn_norm.weight"),
                &ws.hidden,
                &ws.normed,
                1,
                self.inventory.hidden_size,
                true,
            )?;
            enc.memory_barrier_buffers();

            if prefill && layer == final_layer {
                match &mut state.layer_states[layer] {
                    MetalLayerState::Full { k_cache, v_cache } => {
                        self.encode_full_attention_cache_only(
                            enc,
                            layer,
                            k_cache,
                            v_cache,
                            state.position,
                            state.max_context,
                            ws,
                        )?;
                    }
                    MetalLayerState::Delta { recurrent, conv } => {
                        self.encode_delta_attention_block(enc, layer, recurrent, conv, ws)?;
                    }
                }
                return Ok(());
            }

            match &mut state.layer_states[layer] {
                MetalLayerState::Delta { recurrent, conv } => {
                    self.encode_delta_attention_block(enc, layer, recurrent, conv, ws)?;
                }
                MetalLayerState::Full { k_cache, v_cache } => {
                    self.encode_full_attention_block(
                        enc,
                        layer,
                        k_cache,
                        v_cache,
                        state.position,
                        state.max_context,
                        ws,
                    )?;
                }
            }
            enc.memory_barrier_buffers();
            self.encode_add_rms_norm(
                enc,
                &format!("blk.{layer}.post_attention_norm.weight"),
                &ws.hidden,
                &ws.attn_out,
                &ws.hidden,
                &ws.normed,
                1,
                self.inventory.hidden_size,
            )?;
            enc.memory_barrier_buffers();
            self.encode_ffn_block(enc, layer, ws)?;
            enc.memory_barrier_buffers();
            self.encode_add(
                enc,
                &ws.hidden,
                &ws.attn_out,
                &ws.hidden,
                self.inventory.hidden_size,
            )?;
            enc.memory_barrier_buffers();
        }
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
        let rows = usize::try_from(token.ne[1]).map_err(|_| {
            Error::Gguf(format!(
                "token_embd.weight row count {} does not fit usize",
                token.ne[1]
            ))
        })?;
        let hidden = usize::try_from(token.ne[0]).map_err(|_| {
            Error::Gguf(format!(
                "token_embd.weight column count {} does not fit usize",
                token.ne[0]
            ))
        })?;
        if rows != self.inventory.vocab_size || hidden != self.inventory.hidden_size {
            return Err(Error::Gguf(format!(
                "token_embd.weight shape [{rows}, {hidden}] does not match inventory [{}, {}]",
                self.inventory.vocab_size, self.inventory.hidden_size
            )));
        }

        let ids_buf = self.upload_pod(ids)?;
        let out_buf = self.new_f32(ids.len() * hidden)?;
        let cb = self.command_buffer()?;
        let enc = cb
            .compute()
            .ok_or_else(|| Error::GenerationUnavailable("failed to create Metal encoder".into()))?;
        ok(ops::op_get_rows(
            &enc,
            self.dev,
            op_type(token.dtype)?,
            &token.buffer,
            token.offset,
            &ids_buf,
            &out_buf,
            checked_i32(hidden, "embedding hidden")?,
            token.nb[1],
            token.nb[2],
            token.nb[3],
            checked_i32(ids.len(), "embedding ids")?,
            1,
            1,
            std::mem::size_of::<u32>() as u64,
            std::mem::size_of_val(ids) as u64,
            std::mem::size_of_val(ids) as u64,
            (hidden * std::mem::size_of::<f32>()) as u64,
            (ids.len() * hidden * std::mem::size_of::<f32>()) as u64,
            (ids.len() * hidden * std::mem::size_of::<f32>()) as u64,
        ))?;
        enc.end();
        cb.commit_and_wait()
            .map_err(|e| Error::GenerationUnavailable(format!("Metal embedding failed: {e}")))?;
        let mut out = vec![0.0f32; ids.len() * hidden];
        unsafe {
            out_buf.read(0, &mut out);
        }
        Ok(out)
    }

    pub fn matvec(&self, tensor_name: &str, input: &[f32]) -> Result<Vec<f32>> {
        let tensor = self.weights.require(tensor_name)?;
        let cols = usize::try_from(tensor.ne[0]).map_err(|_| {
            Error::Gguf(format!(
                "{tensor_name} column count {} does not fit usize",
                tensor.ne[0]
            ))
        })?;
        let rows = usize::try_from(tensor.ne[1]).map_err(|_| {
            Error::Gguf(format!(
                "{tensor_name} row count {} does not fit usize",
                tensor.ne[1]
            ))
        })?;
        if input.len() != cols {
            return Err(Error::InvalidRequest(format!(
                "{tensor_name} input len {}, expected {cols}",
                input.len()
            )));
        }

        let src = self.upload_pod(input)?;
        let dst = self.new_f32(rows)?;
        let cb = self.command_buffer()?;
        let enc = cb
            .compute()
            .ok_or_else(|| Error::GenerationUnavailable("failed to create Metal encoder".into()))?;
        ok(ops::op_mul_mv(
            &enc,
            self.dev,
            op_type(tensor.dtype)?,
            OpType::F32,
            &tensor.buffer,
            tensor.offset,
            &src,
            &dst,
            checked_i32(cols, "matvec cols")?,
            checked_i32(rows, "matvec rows")?,
            1,
            1,
            tensor.nb[0],
            tensor.nb[1],
            tensor.nb[2],
            tensor.nb[3],
            checked_i32(cols, "matvec src ne10")?,
            1,
            1,
            1,
            std::mem::size_of::<f32>() as u64,
            (cols * std::mem::size_of::<f32>()) as u64,
            (cols * std::mem::size_of::<f32>()) as u64,
            (cols * std::mem::size_of::<f32>()) as u64,
            checked_i32(rows, "matvec dst ne0")?,
            1,
        ))?;
        enc.end();
        cb.commit_and_wait().map_err(|e| {
            Error::GenerationUnavailable(format!("Metal matvec {tensor_name} failed: {e}"))
        })?;
        let mut out = vec![0.0f32; rows];
        unsafe {
            dst.read(0, &mut out);
        }
        Ok(out)
    }

    pub fn matmul_rows(
        &self,
        tensor_name: &str,
        input: &[f32],
        input_rows: usize,
    ) -> Result<Vec<f32>> {
        let tensor = self.weights.require(tensor_name)?;
        let cols = tensor_cols(tensor)?;
        let rows = tensor_rows(tensor)?;
        if input_rows == 0 || input.len() != input_rows * cols {
            return Err(Error::InvalidRequest(format!(
                "{tensor_name} input len {}, expected {input_rows}x{cols}",
                input.len()
            )));
        }

        let src = self.upload_pod(input)?;
        let dst = self.new_f32(input_rows * rows)?;
        let cb = self.command_buffer()?;
        let enc = cb
            .compute()
            .ok_or_else(|| Error::GenerationUnavailable("failed to create Metal encoder".into()))?;
        ok(ops::op_mul_mm(
            &enc,
            self.dev,
            op_type(tensor.dtype)?,
            OpType::F32,
            &tensor.buffer,
            tensor.offset,
            &src,
            &dst,
            checked_i32(cols, "matmul cols")?,
            checked_i32(rows, "matmul rows")?,
            1,
            1,
            tensor.nb[1],
            tensor.nb[2],
            tensor.nb[3],
            checked_i32(input_rows, "matmul input rows")?,
            1,
            1,
            std::mem::size_of::<f32>() as u64,
            (cols * std::mem::size_of::<f32>()) as u64,
            (input.len() * std::mem::size_of::<f32>()) as u64,
            (input.len() * std::mem::size_of::<f32>()) as u64,
            checked_i32(rows, "matmul dst cols")?,
            checked_i32(input_rows, "matmul dst rows")?,
        ))?;
        enc.end();
        cb.commit_and_wait().map_err(|e| {
            Error::GenerationUnavailable(format!("Metal matmul {tensor_name} failed: {e}"))
        })?;
        self.read_f32(&dst, input_rows * rows)
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
        let weight = self.require_f32_weight(tensor_name, dim)?;
        let rows = input.len() / dim;
        let src = self.upload_pod(input)?;
        let dst = self.new_f32(input.len())?;
        let cb = self.command_buffer()?;
        let enc = cb
            .compute()
            .ok_or_else(|| Error::GenerationUnavailable("failed to create Metal encoder".into()))?;
        ok(ops::op_qwen_rms_norm_f32(
            &enc,
            self.dev,
            &src,
            &weight.buffer,
            weight.offset,
            &dst,
            checked_i32(rows, "RMSNorm rows")?,
            checked_i32(dim, "RMSNorm dim")?,
            RMS_EPS,
            qwen_scale,
        ))?;
        enc.end();
        cb.commit_and_wait()
            .map_err(|e| Error::GenerationUnavailable(format!("Metal RMSNorm failed: {e}")))?;
        self.read_f32(&dst, input.len())
    }

    pub fn add_rms_norm(
        &self,
        tensor_name: &str,
        lhs: &[f32],
        rhs: &[f32],
        dim: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        if lhs.len() != rhs.len() || dim == 0 || lhs.len() % dim != 0 {
            return Err(Error::InvalidRequest(format!(
                "{tensor_name} add_rms_norm lhs len {}, rhs len {}, dim {dim}",
                lhs.len(),
                rhs.len()
            )));
        }
        let weight = self.require_f32_weight(tensor_name, dim)?;
        let rows = lhs.len() / dim;
        let lhs_buf = self.upload_pod(lhs)?;
        let rhs_buf = self.upload_pod(rhs)?;
        let sum = self.new_f32(lhs.len())?;
        let norm = self.new_f32(lhs.len())?;
        let cb = self.command_buffer()?;
        let enc = cb
            .compute()
            .ok_or_else(|| Error::GenerationUnavailable("failed to create Metal encoder".into()))?;
        ok(ops::op_qwen_add_rms_norm_f32(
            &enc,
            self.dev,
            &lhs_buf,
            &rhs_buf,
            &weight.buffer,
            weight.offset,
            &sum,
            &norm,
            checked_i32(rows, "add_rms_norm rows")?,
            checked_i32(dim, "add_rms_norm dim")?,
            RMS_EPS,
        ))?;
        enc.end();
        cb.commit_and_wait()
            .map_err(|e| Error::GenerationUnavailable(format!("Metal add_rms_norm failed: {e}")))?;
        Ok((
            self.read_f32(&sum, lhs.len())?,
            self.read_f32(&norm, lhs.len())?,
        ))
    }

    pub fn swiglu(&self, gate: &[f32], up: &[f32]) -> Result<Vec<f32>> {
        if gate.len() != up.len() {
            return Err(Error::InvalidRequest(format!(
                "swiglu gate len {} != up len {}",
                gate.len(),
                up.len()
            )));
        }
        let gate_buf = self.upload_pod(gate)?;
        let up_buf = self.upload_pod(up)?;
        let dst = self.new_f32(gate.len())?;
        let cb = self.command_buffer()?;
        let enc = cb
            .compute()
            .ok_or_else(|| Error::GenerationUnavailable("failed to create Metal encoder".into()))?;
        ok(ops::op_qwen_swiglu_f32(
            &enc,
            self.dev,
            &gate_buf,
            &up_buf,
            &dst,
            checked_i32(gate.len(), "SwiGLU total")?,
        ))?;
        enc.end();
        cb.commit_and_wait()
            .map_err(|e| Error::GenerationUnavailable(format!("Metal SwiGLU failed: {e}")))?;
        self.read_f32(&dst, gate.len())
    }

    pub fn apply_silu_gate(&self, values: &[f32], gate: &[f32]) -> Result<Vec<f32>> {
        if values.len() != gate.len() {
            return Err(Error::InvalidRequest(format!(
                "apply_silu_gate values len {} != gate len {}",
                values.len(),
                gate.len()
            )));
        }
        let values_buf = self.upload_pod(values)?;
        let gate_buf = self.upload_pod(gate)?;
        let cb = self.command_buffer()?;
        let enc = cb
            .compute()
            .ok_or_else(|| Error::GenerationUnavailable("failed to create Metal encoder".into()))?;
        ok(ops::op_qwen_apply_silu_gate_f32(
            &enc,
            self.dev,
            &values_buf,
            0,
            &gate_buf,
            0,
            checked_i32(values.len(), "SiLU gate total")?,
        ))?;
        enc.end();
        cb.commit_and_wait().map_err(|e| {
            Error::GenerationUnavailable(format!("Metal apply_silu_gate failed: {e}"))
        })?;
        self.read_f32(&values_buf, values.len())
    }

    pub fn add(&self, lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>> {
        if lhs.len() != rhs.len() {
            return Err(Error::InvalidRequest(format!(
                "add lhs len {} != rhs len {}",
                lhs.len(),
                rhs.len()
            )));
        }
        let lhs_buf = self.upload_pod(lhs)?;
        let rhs_buf = self.upload_pod(rhs)?;
        let dst = self.new_f32(lhs.len())?;
        let cb = self.command_buffer()?;
        let enc = cb
            .compute()
            .ok_or_else(|| Error::GenerationUnavailable("failed to create Metal encoder".into()))?;
        ok(ops::op_qwen_add_f32(
            &enc,
            self.dev,
            &lhs_buf,
            &rhs_buf,
            &dst,
            checked_i32(lhs.len(), "add total")?,
        ))?;
        enc.end();
        cb.commit_and_wait()
            .map_err(|e| Error::GenerationUnavailable(format!("Metal add failed: {e}")))?;
        self.read_f32(&dst, lhs.len())
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
        let weight = self.require_f32_weight_2d(tensor_name, values.len(), kernel)?;
        let values_buf = self.upload_pod(values)?;
        let state_buf = self.upload_pod(state)?;
        let cb = self.command_buffer()?;
        let enc = cb
            .compute()
            .ok_or_else(|| Error::GenerationUnavailable("failed to create Metal encoder".into()))?;
        ok(ops::op_qwen_causal_conv1d_silu_f32(
            &enc,
            self.dev,
            &values_buf,
            &weight.buffer,
            weight.offset,
            &state_buf,
            checked_i32(values.len(), "conv channels")?,
            checked_i32(kernel, "conv kernel")?,
        ))?;
        enc.end();
        cb.commit_and_wait().map_err(|e| {
            Error::GenerationUnavailable(format!("Metal causal_conv1d_silu failed: {e}"))
        })?;
        Ok((
            self.read_f32(&values_buf, values.len())?,
            self.read_f32(&state_buf, state.len())?,
        ))
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
        let q_buf = self.upload_pod(q)?;
        let k_buf = self.upload_pod(k)?;
        let cb = self.command_buffer()?;
        let enc = cb
            .compute()
            .ok_or_else(|| Error::GenerationUnavailable("failed to create Metal encoder".into()))?;
        ok(ops::op_qwen_normalize_linear_qk_f32(
            &enc,
            self.dev,
            &q_buf,
            0,
            &k_buf,
            0,
            checked_i32(heads, "linear-qk heads")?,
            checked_i32(head_dim, "linear-qk head dim")?,
            RMS_EPS,
        ))?;
        enc.end();
        cb.commit_and_wait().map_err(|e| {
            Error::GenerationUnavailable(format!("Metal normalize_linear_qk failed: {e}"))
        })?;
        Ok((
            self.read_f32(&q_buf, q.len())?,
            self.read_f32(&k_buf, k.len())?,
        ))
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
        let a_log = self.require_f32_weight(&format!("{prefix}.ssm_a"), heads)?;
        let dt_bias = self.require_f32_weight(&format!("{prefix}.ssm_dt.bias"), heads)?;
        let q_buf = self.upload_pod(q)?;
        let k_buf = self.upload_pod(k)?;
        let v_buf = self.upload_pod(v)?;
        let beta_buf = self.upload_pod(beta)?;
        let alpha_buf = self.upload_pod(alpha)?;
        let state_buf = self.upload_pod(recurrent)?;
        let out_buf = self.new_f32(inner)?;
        let cb = self.command_buffer()?;
        let enc = cb
            .compute()
            .ok_or_else(|| Error::GenerationUnavailable("failed to create Metal encoder".into()))?;
        ok(ops::op_qwen_deltanet_decode_f32(
            &enc,
            self.dev,
            &q_buf,
            0,
            &k_buf,
            0,
            &v_buf,
            0,
            &beta_buf,
            &alpha_buf,
            &a_log.buffer,
            a_log.offset,
            &dt_bias.buffer,
            dt_bias.offset,
            &state_buf,
            &out_buf,
            checked_i32(heads, "DeltaNet heads")?,
            checked_i32(head_dim, "DeltaNet head dim")?,
        ))?;
        enc.end();
        cb.commit_and_wait().map_err(|e| {
            Error::GenerationUnavailable(format!("Metal DeltaNet decode failed: {e}"))
        })?;
        Ok((
            self.read_f32(&out_buf, inner)?,
            self.read_f32(&state_buf, recurrent.len())?,
        ))
    }

    pub fn rope_decode(
        &self,
        values: &[f32],
        heads: usize,
        head_dim: usize,
        rope_dim: usize,
        position: usize,
    ) -> Result<Vec<f32>> {
        if values.len() != heads * head_dim || rope_dim == 0 || rope_dim > head_dim {
            return Err(Error::InvalidRequest(format!(
                "rope_decode values len {}, heads {heads}, head_dim {head_dim}, rope_dim {rope_dim}",
                values.len()
            )));
        }
        let buf = self.upload_pod(values)?;
        let cb = self.command_buffer()?;
        let enc = cb
            .compute()
            .ok_or_else(|| Error::GenerationUnavailable("failed to create Metal encoder".into()))?;
        ok(ops::op_qwen_rope_decode_f32(
            &enc,
            self.dev,
            &buf,
            0,
            checked_i32(heads, "RoPE heads")?,
            checked_i32(head_dim, "RoPE head dim")?,
            checked_i32(rope_dim, "RoPE dim")?,
            checked_i32(position, "RoPE position")?,
            ROPE_THETA,
        ))?;
        enc.end();
        cb.commit_and_wait()
            .map_err(|e| Error::GenerationUnavailable(format!("Metal RoPE failed: {e}")))?;
        self.read_f32(&buf, values.len())
    }

    fn encode_embed_tokens(
        &self,
        enc: &greppy_embed_native::metal::ffi::ComputeEncoder,
        ids: &Buffer,
        dst: &Buffer,
        rows: usize,
    ) -> Result<()> {
        let token = self.weights.require("token_embd.weight")?;
        let token_rows = tensor_rows(token)?;
        let hidden = tensor_cols(token)?;
        if token_rows != self.inventory.vocab_size || hidden != self.inventory.hidden_size {
            return Err(Error::Gguf(format!(
                "token_embd.weight shape [{token_rows}, {hidden}] does not match inventory [{}, {}]",
                self.inventory.vocab_size, self.inventory.hidden_size
            )));
        }
        ok(ops::op_get_rows(
            enc,
            self.dev,
            op_type(token.dtype)?,
            &token.buffer,
            token.offset,
            ids,
            dst,
            checked_i32(hidden, "embedding hidden")?,
            token.nb[1],
            token.nb[2],
            token.nb[3],
            checked_i32(rows, "embedding rows")?,
            1,
            1,
            std::mem::size_of::<u32>() as u64,
            (rows * std::mem::size_of::<u32>()) as u64,
            (rows * std::mem::size_of::<u32>()) as u64,
            (hidden * std::mem::size_of::<f32>()) as u64,
            (rows * hidden * std::mem::size_of::<f32>()) as u64,
            (rows * hidden * std::mem::size_of::<f32>()) as u64,
        ))
    }

    fn encode_concat_rows(
        &self,
        enc: &greppy_embed_native::metal::ffi::ComputeEncoder,
        left: &Buffer,
        right: &Buffer,
        dst: &Buffer,
        rows: usize,
        dim: usize,
    ) -> Result<()> {
        let elem = std::mem::size_of::<f32>();
        let src_row = dim * elem;
        let src_total = rows * src_row;
        let dst_row = dim * 2 * elem;
        let dst_total = rows * dst_row;
        ok(ops::op_concat(
            enc,
            self.dev,
            left,
            right,
            dst,
            0,
            checked_i32(dim, "concat left width")?,
            checked_i32(rows, "concat left rows")?,
            1,
            1,
            elem as u64,
            src_row as u64,
            src_total as u64,
            src_total as u64,
            checked_i32(dim, "concat right width")?,
            checked_i32(rows, "concat right rows")?,
            1,
            1,
            elem as u64,
            src_row as u64,
            src_total as u64,
            src_total as u64,
            checked_i32(dim * 2, "concat output width")?,
            checked_i32(rows, "concat output rows")?,
            1,
            1,
            elem as u64,
            dst_row as u64,
            dst_total as u64,
            dst_total as u64,
        ))
    }

    fn encode_matvec_to(
        &self,
        enc: &greppy_embed_native::metal::ffi::ComputeEncoder,
        tensor_name: &str,
        src: &Buffer,
        cols: usize,
        dst: &Buffer,
        rows: usize,
    ) -> Result<()> {
        let tensor = self.weights.require(tensor_name)?;
        let tensor_cols = tensor_cols(tensor)?;
        let tensor_rows = tensor_rows(tensor)?;
        if tensor_cols != cols || tensor_rows != rows {
            return Err(Error::InvalidRequest(format!(
                "{tensor_name} matvec shape [{tensor_rows}, {tensor_cols}], expected [{rows}, {cols}]"
            )));
        }
        ok(ops::op_mul_mv(
            enc,
            self.dev,
            op_type(tensor.dtype)?,
            OpType::F32,
            &tensor.buffer,
            tensor.offset,
            src,
            dst,
            checked_i32(cols, "matvec cols")?,
            checked_i32(rows, "matvec rows")?,
            1,
            1,
            tensor.nb[0],
            tensor.nb[1],
            tensor.nb[2],
            tensor.nb[3],
            checked_i32(cols, "matvec src ne10")?,
            1,
            1,
            1,
            std::mem::size_of::<f32>() as u64,
            (cols * std::mem::size_of::<f32>()) as u64,
            (cols * std::mem::size_of::<f32>()) as u64,
            (cols * std::mem::size_of::<f32>()) as u64,
            checked_i32(rows, "matvec dst ne0")?,
            1,
        ))
    }

    fn encode_matmul_rows_to(
        &self,
        enc: &greppy_embed_native::metal::ffi::ComputeEncoder,
        tensor_name: &str,
        src: &Buffer,
        input_rows: usize,
        cols: usize,
        dst: &Buffer,
        rows: usize,
    ) -> Result<()> {
        let tensor = self.weights.require(tensor_name)?;
        let tensor_cols = tensor_cols(tensor)?;
        let tensor_rows = tensor_rows(tensor)?;
        if tensor_cols != cols || tensor_rows != rows {
            return Err(Error::InvalidRequest(format!(
                "{tensor_name} matmul shape [{tensor_rows}, {tensor_cols}], expected [{rows}, {cols}]"
            )));
        }
        let src_row_bytes = cols * std::mem::size_of::<f32>();
        let src_total_bytes = input_rows * src_row_bytes;
        ok(ops::op_mul_mm(
            enc,
            self.dev,
            op_type(tensor.dtype)?,
            OpType::F32,
            &tensor.buffer,
            tensor.offset,
            src,
            dst,
            checked_i32(cols, "matmul cols")?,
            checked_i32(rows, "matmul rows")?,
            1,
            1,
            tensor.nb[1],
            tensor.nb[2],
            tensor.nb[3],
            checked_i32(input_rows, "matmul input rows")?,
            1,
            1,
            std::mem::size_of::<f32>() as u64,
            src_row_bytes as u64,
            src_total_bytes as u64,
            src_total_bytes as u64,
            checked_i32(rows, "matmul dst cols")?,
            checked_i32(input_rows, "matmul dst rows")?,
        ))
    }

    fn encode_rms_norm(
        &self,
        enc: &greppy_embed_native::metal::ffi::ComputeEncoder,
        tensor_name: &str,
        src: &Buffer,
        dst: &Buffer,
        rows: usize,
        dim: usize,
        qwen_scale: bool,
    ) -> Result<()> {
        let weight = self.require_f32_weight(tensor_name, dim)?;
        ok(ops::op_qwen_rms_norm_f32(
            enc,
            self.dev,
            src,
            &weight.buffer,
            weight.offset,
            dst,
            checked_i32(rows, "RMSNorm rows")?,
            checked_i32(dim, "RMSNorm dim")?,
            RMS_EPS,
            qwen_scale,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_add_rms_norm(
        &self,
        enc: &greppy_embed_native::metal::ffi::ComputeEncoder,
        tensor_name: &str,
        lhs: &Buffer,
        rhs: &Buffer,
        sum_out: &Buffer,
        norm_out: &Buffer,
        rows: usize,
        dim: usize,
    ) -> Result<()> {
        let weight = self.require_f32_weight(tensor_name, dim)?;
        ok(ops::op_qwen_add_rms_norm_f32(
            enc,
            self.dev,
            lhs,
            rhs,
            &weight.buffer,
            weight.offset,
            sum_out,
            norm_out,
            checked_i32(rows, "add_rms_norm rows")?,
            checked_i32(dim, "add_rms_norm dim")?,
            RMS_EPS,
        ))
    }

    fn encode_add(
        &self,
        enc: &greppy_embed_native::metal::ffi::ComputeEncoder,
        lhs: &Buffer,
        rhs: &Buffer,
        dst: &Buffer,
        total: usize,
    ) -> Result<()> {
        ok(ops::op_qwen_add_f32(
            enc,
            self.dev,
            lhs,
            rhs,
            dst,
            checked_i32(total, "add total")?,
        ))
    }

    fn encode_ffn_block(
        &self,
        enc: &greppy_embed_native::metal::ffi::ComputeEncoder,
        layer: usize,
        ws: &MetalForwardWorkspace,
    ) -> Result<()> {
        let prefix = format!("blk.{layer}");
        self.encode_matvec_to(
            enc,
            &format!("{prefix}.ffn_gate.weight"),
            &ws.normed,
            self.inventory.hidden_size,
            &ws.ffn_gate,
            self.inventory.feed_forward_size,
        )?;
        self.encode_matvec_to(
            enc,
            &format!("{prefix}.ffn_up.weight"),
            &ws.normed,
            self.inventory.hidden_size,
            &ws.ffn_up,
            self.inventory.feed_forward_size,
        )?;
        enc.memory_barrier_buffers();
        ok(ops::op_qwen_swiglu_f32(
            enc,
            self.dev,
            &ws.ffn_gate,
            &ws.ffn_up,
            &ws.ffn_gate,
            checked_i32(self.inventory.feed_forward_size, "SwiGLU total")?,
        ))?;
        enc.memory_barrier_buffers();
        self.encode_matvec_to(
            enc,
            &format!("{prefix}.ffn_down.weight"),
            &ws.ffn_gate,
            self.inventory.feed_forward_size,
            &ws.attn_out,
            self.inventory.hidden_size,
        )
    }

    fn encode_ffn_block_rows(
        &self,
        enc: &greppy_embed_native::metal::ffi::ComputeEncoder,
        layer: usize,
        rows: usize,
        ws: &MetalPrefillWorkspace,
    ) -> Result<()> {
        let prefix = format!("blk.{layer}");
        self.encode_matmul_rows_to(
            enc,
            &format!("{prefix}.ffn_gate.weight"),
            &ws.normed,
            rows,
            self.inventory.hidden_size,
            &ws.ffn_gate,
            self.inventory.feed_forward_size,
        )?;
        self.encode_matmul_rows_to(
            enc,
            &format!("{prefix}.ffn_up.weight"),
            &ws.normed,
            rows,
            self.inventory.hidden_size,
            &ws.ffn_up,
            self.inventory.feed_forward_size,
        )?;
        enc.memory_barrier_buffers();
        ok(ops::op_qwen_swiglu_f32(
            enc,
            self.dev,
            &ws.ffn_gate,
            &ws.ffn_up,
            &ws.ffn_gate,
            checked_i32(
                rows * self.inventory.feed_forward_size,
                "batched SwiGLU total",
            )?,
        ))?;
        enc.memory_barrier_buffers();
        self.encode_matmul_rows_to(
            enc,
            &format!("{prefix}.ffn_down.weight"),
            &ws.ffn_gate,
            rows,
            self.inventory.feed_forward_size,
            &ws.attn_out,
            self.inventory.hidden_size,
        )
    }

    fn encode_delta_attention_block(
        &self,
        enc: &greppy_embed_native::metal::ffi::ComputeEncoder,
        layer: usize,
        recurrent: &Buffer,
        conv: &Buffer,
        ws: &MetalForwardWorkspace,
    ) -> Result<()> {
        let prefix = format!("blk.{layer}");
        let inner = self.inventory.ssm_inner_size;
        let heads = self.inventory.ssm_group_count;
        let head_dim = inner / heads;
        self.encode_matvec_to(
            enc,
            &format!("{prefix}.attn_qkv.weight"),
            &ws.normed,
            self.inventory.hidden_size,
            &ws.qkv,
            inner * 3,
        )?;
        enc.memory_barrier_buffers();
        let conv_weight = self.require_f32_weight_2d(
            &format!("{prefix}.ssm_conv1d.weight"),
            inner * 3,
            CONV_KERNEL,
        )?;
        ok(ops::op_qwen_causal_conv1d_silu_f32(
            enc,
            self.dev,
            &ws.qkv,
            &conv_weight.buffer,
            conv_weight.offset,
            conv,
            checked_i32(inner * 3, "Delta conv channels")?,
            checked_i32(CONV_KERNEL, "Delta conv kernel")?,
        ))?;
        self.encode_matvec_to(
            enc,
            &format!("{prefix}.attn_gate.weight"),
            &ws.normed,
            self.inventory.hidden_size,
            &ws.z,
            inner,
        )?;
        self.encode_matvec_to(
            enc,
            &format!("{prefix}.ssm_beta.weight"),
            &ws.normed,
            self.inventory.hidden_size,
            &ws.beta,
            heads,
        )?;
        self.encode_matvec_to(
            enc,
            &format!("{prefix}.ssm_alpha.weight"),
            &ws.normed,
            self.inventory.hidden_size,
            &ws.alpha,
            self.inventory.ssm_time_step_rank,
        )?;
        enc.memory_barrier_buffers();
        ok(ops::op_qwen_normalize_linear_qk_f32(
            enc,
            self.dev,
            &ws.qkv,
            0,
            &ws.qkv,
            inner * std::mem::size_of::<f32>(),
            checked_i32(heads, "Delta heads")?,
            checked_i32(head_dim, "Delta head dim")?,
            RMS_EPS,
        ))?;
        enc.memory_barrier_buffers();
        let a_log = self.require_f32_weight(&format!("{prefix}.ssm_a"), heads)?;
        let dt_bias = self.require_f32_weight(&format!("{prefix}.ssm_dt.bias"), heads)?;
        ok(ops::op_qwen_deltanet_decode_f32(
            enc,
            self.dev,
            &ws.qkv,
            0,
            &ws.qkv,
            inner * std::mem::size_of::<f32>(),
            &ws.qkv,
            inner * 2 * std::mem::size_of::<f32>(),
            &ws.beta,
            &ws.alpha,
            &a_log.buffer,
            a_log.offset,
            &dt_bias.buffer,
            dt_bias.offset,
            recurrent,
            &ws.raw,
            checked_i32(heads, "Delta heads")?,
            checked_i32(head_dim, "Delta head dim")?,
        ))?;
        enc.memory_barrier_buffers();
        self.encode_rms_norm(
            enc,
            &format!("{prefix}.ssm_norm.weight"),
            &ws.raw,
            &ws.raw,
            heads,
            head_dim,
            false,
        )?;
        enc.memory_barrier_buffers();
        ok(ops::op_qwen_apply_silu_gate_f32(
            enc,
            self.dev,
            &ws.raw,
            0,
            &ws.z,
            0,
            checked_i32(inner, "Delta gate total")?,
        ))?;
        enc.memory_barrier_buffers();
        self.encode_matvec_to(
            enc,
            &format!("{prefix}.ssm_out.weight"),
            &ws.raw,
            inner,
            &ws.attn_out,
            self.inventory.hidden_size,
        )
    }

    fn encode_delta_attention_block_rows(
        &self,
        enc: &greppy_embed_native::metal::ffi::ComputeEncoder,
        layer: usize,
        recurrent: &Buffer,
        conv: &Buffer,
        rows: usize,
        ws: &MetalPrefillWorkspace,
    ) -> Result<()> {
        let prefix = format!("blk.{layer}");
        let inner = self.inventory.ssm_inner_size;
        let heads = self.inventory.ssm_group_count;
        let head_dim = inner / heads;
        let qkv_stride = inner * 3;
        self.encode_matmul_rows_to(
            enc,
            &format!("{prefix}.attn_qkv.weight"),
            &ws.normed,
            rows,
            self.inventory.hidden_size,
            &ws.qkv,
            inner * 3,
        )?;
        enc.memory_barrier_buffers();
        let conv_weight = self.require_f32_weight_2d(
            &format!("{prefix}.ssm_conv1d.weight"),
            inner * 3,
            CONV_KERNEL,
        )?;
        ok(ops::op_qwen_causal_conv1d_silu_rows_f32(
            enc,
            self.dev,
            &ws.qkv,
            &conv_weight.buffer,
            conv_weight.offset,
            conv,
            checked_i32(rows, "Delta rows")?,
            checked_i32(inner * 3, "Delta conv channels")?,
            checked_i32(CONV_KERNEL, "Delta conv kernel")?,
        ))?;
        self.encode_matmul_rows_to(
            enc,
            &format!("{prefix}.attn_gate.weight"),
            &ws.normed,
            rows,
            self.inventory.hidden_size,
            &ws.z,
            inner,
        )?;
        self.encode_matmul_rows_to(
            enc,
            &format!("{prefix}.ssm_beta.weight"),
            &ws.normed,
            rows,
            self.inventory.hidden_size,
            &ws.beta,
            heads,
        )?;
        self.encode_matmul_rows_to(
            enc,
            &format!("{prefix}.ssm_alpha.weight"),
            &ws.normed,
            rows,
            self.inventory.hidden_size,
            &ws.alpha,
            self.inventory.ssm_time_step_rank,
        )?;
        enc.memory_barrier_buffers();
        ok(ops::op_qwen_normalize_linear_qk_rows_f32(
            enc,
            self.dev,
            &ws.qkv,
            0,
            &ws.qkv,
            inner * std::mem::size_of::<f32>(),
            checked_i32(rows, "Delta qk rows")?,
            checked_i32(heads, "Delta heads")?,
            checked_i32(head_dim, "Delta head dim")?,
            checked_i32(qkv_stride, "Delta q stride")?,
            checked_i32(qkv_stride, "Delta k stride")?,
            RMS_EPS,
        ))?;
        enc.memory_barrier_buffers();
        let a_log = self.require_f32_weight(&format!("{prefix}.ssm_a"), heads)?;
        let dt_bias = self.require_f32_weight(&format!("{prefix}.ssm_dt.bias"), heads)?;
        ok(ops::op_qwen_deltanet_decode_rows_f32(
            enc,
            self.dev,
            &ws.qkv,
            0,
            &ws.qkv,
            inner * std::mem::size_of::<f32>(),
            &ws.qkv,
            inner * 2 * std::mem::size_of::<f32>(),
            &ws.beta,
            &ws.alpha,
            &a_log.buffer,
            a_log.offset,
            &dt_bias.buffer,
            dt_bias.offset,
            recurrent,
            &ws.raw,
            checked_i32(rows, "Delta rows")?,
            checked_i32(heads, "Delta heads")?,
            checked_i32(head_dim, "Delta head dim")?,
            checked_i32(qkv_stride, "Delta q stride")?,
            checked_i32(qkv_stride, "Delta k stride")?,
            checked_i32(qkv_stride, "Delta v stride")?,
            checked_i32(heads, "Delta beta stride")?,
            checked_i32(self.inventory.ssm_time_step_rank, "Delta alpha stride")?,
            checked_i32(inner, "Delta out stride")?,
        ))?;
        enc.memory_barrier_buffers();
        self.encode_rms_norm(
            enc,
            &format!("{prefix}.ssm_norm.weight"),
            &ws.raw,
            &ws.raw,
            rows * heads,
            head_dim,
            false,
        )?;
        enc.memory_barrier_buffers();
        ok(ops::op_qwen_apply_silu_gate_f32(
            enc,
            self.dev,
            &ws.raw,
            0,
            &ws.z,
            0,
            checked_i32(rows * inner, "Delta gate total")?,
        ))?;
        enc.memory_barrier_buffers();
        self.encode_matmul_rows_to(
            enc,
            &format!("{prefix}.ssm_out.weight"),
            &ws.raw,
            rows,
            inner,
            &ws.attn_out,
            self.inventory.hidden_size,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_full_attention_block(
        &self,
        enc: &greppy_embed_native::metal::ffi::ComputeEncoder,
        layer: usize,
        k_cache: &Buffer,
        v_cache: &Buffer,
        position: usize,
        max_context: usize,
        ws: &MetalForwardWorkspace,
    ) -> Result<()> {
        let prefix = format!("blk.{layer}");
        let q_dim = self.inventory.attention_heads * self.inventory.head_dim;
        let kv_k_dim = self.inventory.kv_heads * self.inventory.head_dim;
        let kv_v_dim = self.inventory.kv_heads * self.inventory.value_dim;
        self.encode_matvec_to(
            enc,
            &format!("{prefix}.attn_q.weight"),
            &ws.normed,
            self.inventory.hidden_size,
            &ws.q_fused,
            q_dim * 2,
        )?;
        self.encode_matvec_to(
            enc,
            &format!("{prefix}.attn_k.weight"),
            &ws.normed,
            self.inventory.hidden_size,
            &ws.k,
            kv_k_dim,
        )?;
        self.encode_matvec_to(
            enc,
            &format!("{prefix}.attn_v.weight"),
            &ws.normed,
            self.inventory.hidden_size,
            &ws.v,
            kv_v_dim,
        )?;
        enc.memory_barrier_buffers();
        ok(ops::op_qwen_deinterleave_q_gate_rows_f32(
            enc,
            self.dev,
            &ws.q_fused,
            0,
            &ws.qkv,
            0,
            &ws.qkv,
            q_dim * std::mem::size_of::<f32>(),
            1,
            checked_i32(self.inventory.attention_heads, "attention heads")?,
            checked_i32(self.inventory.head_dim, "attention head dim")?,
            checked_i32(q_dim * 2, "packed q stride")?,
            checked_i32(q_dim * 2, "deinterleaved q stride")?,
        ))?;
        enc.memory_barrier_buffers();
        self.encode_rms_norm(
            enc,
            &format!("{prefix}.attn_q_norm.weight"),
            &ws.qkv,
            &ws.qkv,
            self.inventory.attention_heads,
            self.inventory.head_dim,
            true,
        )?;
        self.encode_rms_norm(
            enc,
            &format!("{prefix}.attn_k_norm.weight"),
            &ws.k,
            &ws.k,
            self.inventory.kv_heads,
            self.inventory.head_dim,
            true,
        )?;
        enc.memory_barrier_buffers();
        ok(ops::op_qwen_rope_decode_f32(
            enc,
            self.dev,
            &ws.qkv,
            0,
            checked_i32(self.inventory.attention_heads, "attention heads")?,
            checked_i32(self.inventory.head_dim, "attention head dim")?,
            checked_i32(self.inventory.rope_dim, "attention rope dim")?,
            checked_i32(position, "attention position")?,
            ROPE_THETA,
        ))?;
        ok(ops::op_qwen_rope_decode_f32(
            enc,
            self.dev,
            &ws.k,
            0,
            checked_i32(self.inventory.kv_heads, "kv heads")?,
            checked_i32(self.inventory.head_dim, "kv head dim")?,
            checked_i32(self.inventory.rope_dim, "kv rope dim")?,
            checked_i32(position, "kv position")?,
            ROPE_THETA,
        ))?;
        enc.memory_barrier_buffers();
        self.encode_cache_write(
            enc,
            &ws.k,
            0,
            k_cache,
            position,
            self.inventory.kv_heads,
            self.inventory.head_dim,
            max_context,
        )?;
        self.encode_cache_write(
            enc,
            &ws.v,
            0,
            v_cache,
            position,
            self.inventory.kv_heads,
            self.inventory.value_dim,
            max_context,
        )?;
        enc.memory_barrier_buffers();
        let scale = 1.0 / (self.inventory.head_dim as f32).sqrt();
        ok(ops::op_qwen_attention_scores_decode_f32(
            enc,
            self.dev,
            &ws.qkv,
            k_cache,
            &ws.scores,
            checked_i32(position, "attention position")?,
            checked_i32(self.inventory.attention_heads, "attention heads")?,
            checked_i32(self.inventory.kv_heads, "kv heads")?,
            checked_i32(self.inventory.head_dim, "attention head dim")?,
            checked_i32(max_context, "attention context")?,
            scale,
        ))?;
        enc.memory_barrier_buffers();
        ok(ops::op_qwen_softmax_decode_f32(
            enc,
            self.dev,
            &ws.scores,
            checked_i32(position, "attention position")?,
            checked_i32(self.inventory.attention_heads, "attention heads")?,
            checked_i32(max_context, "attention context")?,
        ))?;
        enc.memory_barrier_buffers();
        ok(ops::op_qwen_attention_values_decode_f32(
            enc,
            self.dev,
            &ws.scores,
            v_cache,
            &ws.raw,
            checked_i32(position, "attention position")?,
            checked_i32(self.inventory.attention_heads, "attention heads")?,
            checked_i32(self.inventory.kv_heads, "kv heads")?,
            checked_i32(self.inventory.value_dim, "attention value dim")?,
            checked_i32(max_context, "attention context")?,
        ))?;
        enc.memory_barrier_buffers();
        ok(ops::op_qwen_apply_sigmoid_gate_f32(
            enc,
            self.dev,
            &ws.raw,
            0,
            &ws.qkv,
            q_dim * std::mem::size_of::<f32>(),
            checked_i32(
                self.inventory.attention_heads * self.inventory.value_dim,
                "attention gate total",
            )?,
        ))?;
        enc.memory_barrier_buffers();
        self.encode_matvec_to(
            enc,
            &format!("{prefix}.attn_output.weight"),
            &ws.raw,
            self.inventory.attention_heads * self.inventory.value_dim,
            &ws.attn_out,
            self.inventory.hidden_size,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_full_attention_block_rows(
        &self,
        enc: &greppy_embed_native::metal::ffi::ComputeEncoder,
        layer: usize,
        k_cache: &Buffer,
        v_cache: &Buffer,
        position: usize,
        rows: usize,
        max_context: usize,
        ws: &MetalPrefillWorkspace,
    ) -> Result<()> {
        let prefix = format!("blk.{layer}");
        let q_dim = self.inventory.attention_heads * self.inventory.head_dim;
        let kv_k_dim = self.inventory.kv_heads * self.inventory.head_dim;
        let kv_v_dim = self.inventory.kv_heads * self.inventory.value_dim;
        self.encode_matmul_rows_to(
            enc,
            &format!("{prefix}.attn_q.weight"),
            &ws.normed,
            rows,
            self.inventory.hidden_size,
            &ws.q_fused,
            q_dim * 2,
        )?;
        self.encode_matmul_rows_to(
            enc,
            &format!("{prefix}.attn_k.weight"),
            &ws.normed,
            rows,
            self.inventory.hidden_size,
            &ws.k,
            kv_k_dim,
        )?;
        self.encode_matmul_rows_to(
            enc,
            &format!("{prefix}.attn_v.weight"),
            &ws.normed,
            rows,
            self.inventory.hidden_size,
            &ws.v,
            kv_v_dim,
        )?;
        enc.memory_barrier_buffers();
        ok(ops::op_qwen_deinterleave_q_gate_rows_f32(
            enc,
            self.dev,
            &ws.q_fused,
            0,
            &ws.qkv,
            0,
            &ws.qkv,
            q_dim * std::mem::size_of::<f32>(),
            checked_i32(rows, "attention rows")?,
            checked_i32(self.inventory.attention_heads, "attention heads")?,
            checked_i32(self.inventory.head_dim, "attention head dim")?,
            checked_i32(q_dim * 2, "packed q stride")?,
            checked_i32(q_dim * 2, "deinterleaved q stride")?,
        ))?;
        enc.memory_barrier_buffers();
        self.encode_strided_rms_norm_rows(
            enc,
            &format!("{prefix}.attn_q_norm.weight"),
            &ws.qkv,
            0,
            rows,
            self.inventory.attention_heads,
            self.inventory.head_dim,
            q_dim * 2,
        )?;
        self.encode_strided_rms_norm_rows(
            enc,
            &format!("{prefix}.attn_k_norm.weight"),
            &ws.k,
            0,
            rows,
            self.inventory.kv_heads,
            self.inventory.head_dim,
            kv_k_dim,
        )?;
        enc.memory_barrier_buffers();
        ok(ops::op_qwen_rope_rows_f32(
            enc,
            self.dev,
            &ws.qkv,
            0,
            checked_i32(rows, "attention rows")?,
            checked_i32(self.inventory.attention_heads, "attention heads")?,
            checked_i32(self.inventory.head_dim, "attention head dim")?,
            checked_i32(self.inventory.rope_dim, "attention rope dim")?,
            checked_i32(position, "attention position")?,
            checked_i32(q_dim * 2, "attention q stride")?,
            ROPE_THETA,
        ))?;
        ok(ops::op_qwen_rope_rows_f32(
            enc,
            self.dev,
            &ws.k,
            0,
            checked_i32(rows, "kv rows")?,
            checked_i32(self.inventory.kv_heads, "kv heads")?,
            checked_i32(self.inventory.head_dim, "kv head dim")?,
            checked_i32(self.inventory.rope_dim, "kv rope dim")?,
            checked_i32(position, "kv position")?,
            checked_i32(kv_k_dim, "kv stride")?,
            ROPE_THETA,
        ))?;
        enc.memory_barrier_buffers();
        self.encode_cache_write_rows(
            enc,
            &ws.k,
            0,
            k_cache,
            rows,
            position,
            self.inventory.kv_heads,
            self.inventory.head_dim,
            max_context,
            kv_k_dim,
        )?;
        self.encode_cache_write_rows(
            enc,
            &ws.v,
            0,
            v_cache,
            rows,
            position,
            self.inventory.kv_heads,
            self.inventory.value_dim,
            max_context,
            kv_v_dim,
        )?;
        enc.memory_barrier_buffers();
        let scale = 1.0 / (self.inventory.head_dim as f32).sqrt();
        let score_stride = self.inventory.attention_heads * max_context;
        let simd32_attention = self.inventory.attention_heads == 8
            && self.inventory.kv_heads == 2
            && self.inventory.head_dim == 256
            && self.inventory.value_dim == 256;
        if simd32_attention {
            ok(ops::op_qwen_attention_scores_rows_simd32_f32(
                enc,
                self.dev,
                &ws.qkv,
                0,
                k_cache,
                &ws.scores,
                checked_i32(rows, "attention rows")?,
                checked_i32(position, "attention position")?,
                checked_i32(self.inventory.attention_heads, "attention heads")?,
                checked_i32(self.inventory.kv_heads, "kv heads")?,
                checked_i32(self.inventory.head_dim, "attention head dim")?,
                checked_i32(max_context, "attention context")?,
                checked_i32(q_dim * 2, "attention q stride")?,
                checked_i32(score_stride, "attention score stride")?,
                scale,
            ))?;
            enc.memory_barrier_buffers();
            ok(ops::op_qwen_softmax_rows_simd32_f32(
                enc,
                self.dev,
                &ws.scores,
                checked_i32(rows, "attention rows")?,
                checked_i32(position, "attention position")?,
                checked_i32(self.inventory.attention_heads, "attention heads")?,
                checked_i32(max_context, "attention context")?,
                checked_i32(score_stride, "attention score stride")?,
            ))?;
            enc.memory_barrier_buffers();
            ok(ops::op_qwen_attention_values_gate_rows_simd32_f32(
                enc,
                self.dev,
                &ws.scores,
                v_cache,
                &ws.qkv,
                q_dim * std::mem::size_of::<f32>(),
                &ws.raw,
                checked_i32(rows, "attention rows")?,
                checked_i32(position, "attention position")?,
                checked_i32(self.inventory.attention_heads, "attention heads")?,
                checked_i32(self.inventory.kv_heads, "kv heads")?,
                checked_i32(self.inventory.value_dim, "attention value dim")?,
                checked_i32(max_context, "attention context")?,
                checked_i32(q_dim * 2, "attention gate stride")?,
                checked_i32(score_stride, "attention score stride")?,
            ))?;
        } else {
            ok(ops::op_qwen_attention_scores_rows_f32(
                enc,
                self.dev,
                &ws.qkv,
                0,
                k_cache,
                &ws.scores,
                checked_i32(rows, "attention rows")?,
                checked_i32(position, "attention position")?,
                checked_i32(self.inventory.attention_heads, "attention heads")?,
                checked_i32(self.inventory.kv_heads, "kv heads")?,
                checked_i32(self.inventory.head_dim, "attention head dim")?,
                checked_i32(max_context, "attention context")?,
                checked_i32(q_dim * 2, "attention q stride")?,
                checked_i32(score_stride, "attention score stride")?,
                scale,
            ))?;
            enc.memory_barrier_buffers();
            ok(ops::op_qwen_softmax_rows_f32(
                enc,
                self.dev,
                &ws.scores,
                checked_i32(rows, "attention rows")?,
                checked_i32(position, "attention position")?,
                checked_i32(self.inventory.attention_heads, "attention heads")?,
                checked_i32(max_context, "attention context")?,
                checked_i32(score_stride, "attention score stride")?,
            ))?;
            enc.memory_barrier_buffers();
            ok(ops::op_qwen_attention_values_rows_f32(
                enc,
                self.dev,
                &ws.scores,
                v_cache,
                &ws.raw,
                checked_i32(rows, "attention rows")?,
                checked_i32(position, "attention position")?,
                checked_i32(self.inventory.attention_heads, "attention heads")?,
                checked_i32(self.inventory.kv_heads, "kv heads")?,
                checked_i32(self.inventory.value_dim, "attention value dim")?,
                checked_i32(max_context, "attention context")?,
                checked_i32(score_stride, "attention score stride")?,
            ))?;
            enc.memory_barrier_buffers();
            ok(ops::op_qwen_apply_sigmoid_gate_rows_f32(
                enc,
                self.dev,
                &ws.raw,
                0,
                &ws.qkv,
                q_dim * std::mem::size_of::<f32>(),
                checked_i32(rows, "attention rows")?,
                checked_i32(q_dim, "attention gate width")?,
                checked_i32(q_dim, "attention value stride")?,
                checked_i32(q_dim * 2, "attention gate stride")?,
            ))?;
        }
        enc.memory_barrier_buffers();
        self.encode_matmul_rows_to(
            enc,
            &format!("{prefix}.attn_output.weight"),
            &ws.raw,
            rows,
            q_dim,
            &ws.attn_out,
            self.inventory.hidden_size,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_full_attention_cache_only(
        &self,
        enc: &greppy_embed_native::metal::ffi::ComputeEncoder,
        layer: usize,
        k_cache: &Buffer,
        v_cache: &Buffer,
        position: usize,
        max_context: usize,
        ws: &MetalForwardWorkspace,
    ) -> Result<()> {
        let prefix = format!("blk.{layer}");
        let kv_k_dim = self.inventory.kv_heads * self.inventory.head_dim;
        let kv_v_dim = self.inventory.kv_heads * self.inventory.value_dim;
        self.encode_matvec_to(
            enc,
            &format!("{prefix}.attn_k.weight"),
            &ws.normed,
            self.inventory.hidden_size,
            &ws.k,
            kv_k_dim,
        )?;
        self.encode_matvec_to(
            enc,
            &format!("{prefix}.attn_v.weight"),
            &ws.normed,
            self.inventory.hidden_size,
            &ws.v,
            kv_v_dim,
        )?;
        enc.memory_barrier_buffers();
        self.encode_rms_norm(
            enc,
            &format!("{prefix}.attn_k_norm.weight"),
            &ws.k,
            &ws.k,
            self.inventory.kv_heads,
            self.inventory.head_dim,
            true,
        )?;
        enc.memory_barrier_buffers();
        ok(ops::op_qwen_rope_decode_f32(
            enc,
            self.dev,
            &ws.k,
            0,
            checked_i32(self.inventory.kv_heads, "kv heads")?,
            checked_i32(self.inventory.head_dim, "kv head dim")?,
            checked_i32(self.inventory.rope_dim, "kv rope dim")?,
            checked_i32(position, "kv position")?,
            ROPE_THETA,
        ))?;
        enc.memory_barrier_buffers();
        self.encode_cache_write(
            enc,
            &ws.k,
            0,
            k_cache,
            position,
            self.inventory.kv_heads,
            self.inventory.head_dim,
            max_context,
        )?;
        self.encode_cache_write(
            enc,
            &ws.v,
            0,
            v_cache,
            position,
            self.inventory.kv_heads,
            self.inventory.value_dim,
            max_context,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_full_attention_cache_only_rows(
        &self,
        enc: &greppy_embed_native::metal::ffi::ComputeEncoder,
        layer: usize,
        k_cache: &Buffer,
        v_cache: &Buffer,
        position: usize,
        rows: usize,
        max_context: usize,
        ws: &MetalPrefillWorkspace,
    ) -> Result<()> {
        let prefix = format!("blk.{layer}");
        let kv_k_dim = self.inventory.kv_heads * self.inventory.head_dim;
        let kv_v_dim = self.inventory.kv_heads * self.inventory.value_dim;
        self.encode_matmul_rows_to(
            enc,
            &format!("{prefix}.attn_k.weight"),
            &ws.normed,
            rows,
            self.inventory.hidden_size,
            &ws.k,
            kv_k_dim,
        )?;
        self.encode_matmul_rows_to(
            enc,
            &format!("{prefix}.attn_v.weight"),
            &ws.normed,
            rows,
            self.inventory.hidden_size,
            &ws.v,
            kv_v_dim,
        )?;
        enc.memory_barrier_buffers();
        self.encode_strided_rms_norm_rows(
            enc,
            &format!("{prefix}.attn_k_norm.weight"),
            &ws.k,
            0,
            rows,
            self.inventory.kv_heads,
            self.inventory.head_dim,
            kv_k_dim,
        )?;
        enc.memory_barrier_buffers();
        ok(ops::op_qwen_rope_rows_f32(
            enc,
            self.dev,
            &ws.k,
            0,
            checked_i32(rows, "kv rows")?,
            checked_i32(self.inventory.kv_heads, "kv heads")?,
            checked_i32(self.inventory.head_dim, "kv head dim")?,
            checked_i32(self.inventory.rope_dim, "kv rope dim")?,
            checked_i32(position, "kv position")?,
            checked_i32(kv_k_dim, "kv stride")?,
            ROPE_THETA,
        ))?;
        enc.memory_barrier_buffers();
        self.encode_cache_write_rows(
            enc,
            &ws.k,
            0,
            k_cache,
            rows,
            position,
            self.inventory.kv_heads,
            self.inventory.head_dim,
            max_context,
            kv_k_dim,
        )?;
        self.encode_cache_write_rows(
            enc,
            &ws.v,
            0,
            v_cache,
            rows,
            position,
            self.inventory.kv_heads,
            self.inventory.value_dim,
            max_context,
            kv_v_dim,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_cache_write(
        &self,
        enc: &greppy_embed_native::metal::ffi::ComputeEncoder,
        src: &Buffer,
        src_offset: usize,
        cache: &Buffer,
        position: usize,
        heads: usize,
        head_dim: usize,
        max_context: usize,
    ) -> Result<()> {
        ok(ops::op_qwen_cache_write_f32(
            enc,
            self.dev,
            src,
            src_offset,
            cache,
            checked_i32(position, "cache position")?,
            checked_i32(heads, "cache heads")?,
            checked_i32(head_dim, "cache head dim")?,
            checked_i32(max_context, "cache context")?,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_cache_write_rows(
        &self,
        enc: &greppy_embed_native::metal::ffi::ComputeEncoder,
        src: &Buffer,
        src_offset: usize,
        cache: &Buffer,
        rows: usize,
        position: usize,
        heads: usize,
        head_dim: usize,
        max_context: usize,
        src_stride: usize,
    ) -> Result<()> {
        ok(ops::op_qwen_cache_write_rows_f32(
            enc,
            self.dev,
            src,
            src_offset,
            cache,
            checked_i32(rows, "cache rows")?,
            checked_i32(position, "cache position")?,
            checked_i32(heads, "cache heads")?,
            checked_i32(head_dim, "cache head dim")?,
            checked_i32(max_context, "cache context")?,
            checked_i32(src_stride, "cache src stride")?,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_strided_rms_norm_rows(
        &self,
        enc: &greppy_embed_native::metal::ffi::ComputeEncoder,
        tensor_name: &str,
        values: &Buffer,
        values_offset: usize,
        rows: usize,
        heads: usize,
        head_dim: usize,
        stride: usize,
    ) -> Result<()> {
        let weight = self.require_f32_weight(tensor_name, head_dim)?;
        ok(ops::op_qwen_rms_norm_strided_rows_f32(
            enc,
            self.dev,
            values,
            values_offset,
            &weight.buffer,
            weight.offset,
            checked_i32(rows, "strided RMSNorm rows")?,
            checked_i32(heads, "strided RMSNorm heads")?,
            checked_i32(head_dim, "strided RMSNorm dim")?,
            checked_i32(stride, "strided RMSNorm stride")?,
            RMS_EPS,
        ))
    }

    fn upload_pod<T: Copy>(&self, values: &[T]) -> Result<Buffer> {
        let bytes = std::mem::size_of_val(values).max(1);
        let buf = self.new_bytes(bytes)?;
        unsafe {
            buf.write(0, values);
        }
        Ok(buf)
    }

    fn new_f32(&self, elems: usize) -> Result<Buffer> {
        self.new_bytes(elems.max(1) * std::mem::size_of::<f32>())
    }

    fn new_bytes(&self, bytes: usize) -> Result<Buffer> {
        self.dev.new_buffer(bytes.max(1)).ok_or_else(|| {
            Error::GenerationUnavailable(format!("failed to allocate Metal buffer ({bytes} bytes)"))
        })
    }

    fn command_buffer(&self) -> Result<greppy_embed_native::metal::ffi::CommandBuffer> {
        self.dev.new_command_buffer().ok_or_else(|| {
            Error::GenerationUnavailable("failed to create Metal command buffer".into())
        })
    }

    fn read_f32(&self, buf: &Buffer, elems: usize) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; elems];
        unsafe {
            buf.read(0, &mut out);
        }
        Ok(out)
    }

    fn require_f32_weight(
        &self,
        tensor_name: &str,
        dim: usize,
    ) -> Result<&greppy_embed_native::metal::tensor::Tensor> {
        let weight = self.weights.require(tensor_name)?;
        if weight.dtype != GgmlType::F32 || weight.ne[0] != dim as i64 {
            return Err(Error::Gguf(format!(
                "{tensor_name} shape {:?} dtype {:?}, expected F32 [{dim}]",
                weight.ne, weight.dtype
            )));
        }
        Ok(weight)
    }

    fn require_f32_weight_2d(
        &self,
        tensor_name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<&greppy_embed_native::metal::tensor::Tensor> {
        let weight = self.weights.require(tensor_name)?;
        if weight.dtype != GgmlType::F32
            || weight.ne[0] != cols as i64
            || weight.ne[1] != rows as i64
        {
            return Err(Error::Gguf(format!(
                "{tensor_name} shape {:?} dtype {:?}, expected F32 [{rows}, {cols}]",
                weight.ne, weight.dtype
            )));
        }
        Ok(weight)
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
        other => Err(Error::GenerationUnavailable(format!(
            "unsupported Qwen3.5 Metal dtype {other:?}"
        )))?,
    })
}

fn ok(value: bool) -> Result<()> {
    if value {
        Ok(())
    } else {
        Err(Error::GenerationUnavailable(format!(
            "Metal dispatch failed: {}",
            ops::last_error_str()
        )))
    }
}

fn checked_i32(value: usize, name: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| {
        Error::InvalidRequest(format!("{name} value {value} does not fit Metal i32 ABI"))
    })
}

fn checked_i64(value: usize, name: &str) -> Result<i64> {
    i64::try_from(value).map_err(|_| {
        Error::InvalidRequest(format!("{name} value {value} does not fit Metal i64 ABI"))
    })
}

fn tensor_cols(tensor: &greppy_embed_native::metal::tensor::Tensor) -> Result<usize> {
    usize::try_from(tensor.ne[0]).map_err(|_| {
        Error::Gguf(format!(
            "{} column count {} does not fit usize",
            tensor.name, tensor.ne[0]
        ))
    })
}

fn tensor_rows(tensor: &greppy_embed_native::metal::tensor::Tensor) -> Result<usize> {
    usize::try_from(tensor.ne[1]).map_err(|_| {
        Error::Gguf(format!(
            "{} row count {} does not fit usize",
            tensor.name, tensor.ne[1]
        ))
    })
}

fn prompt_seed(prompt: &str) -> u64 {
    let mut h = DefaultHasher::new();
    prompt.hash(&mut h);
    h.finish()
}

fn is_greedy_device_sampling(params: GenerationParams) -> bool {
    params.temperature == 0.0
        && params.top_k <= 1
        && params.top_p >= 1.0
        && params.min_p == 0.0
        && params.presence_penalty == 0.0
        && params.repetition_penalty == 1.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use greppy_embed_native::matmul::QuantMatrix;

    fn profile_stage(
        metal: &MetalQwen35Model,
        stage: &str,
        encode: impl FnOnce(&greppy_embed_native::metal::ffi::ComputeEncoder) -> Result<()>,
    ) -> greppy_embed_native::metal::ffi::CommandBufferTiming {
        let dispatch_start = greppy_embed_native::metal::ffi::dispatch_count_snapshot();
        let cb = metal.command_buffer().expect("Metal profile command");
        let enc = cb.compute().expect("Metal profile encoder");
        encode(&enc).expect("Metal profile encode");
        enc.end();
        let timing = cb.commit_and_wait_timed().expect("Metal profile wait");
        eprintln!(
            "metal_stage_profile stage={stage} kernel_ms={:.3} gpu_ms={:.3} wait_ms={:.3} dispatches={}",
            timing.kernel_secs * 1.0e3,
            timing.gpu_secs * 1.0e3,
            timing.submit_wait_secs * 1.0e3,
            greppy_embed_native::metal::ffi::dispatch_count_snapshot() - dispatch_start,
        );
        timing
    }

    #[test]
    fn qwen35_metal_embedding_matches_cpu_for_real_model_when_env_set() {
        let Some(path) = std::env::var_os("QWEN35_NATIVE_GGUF") else {
            eprintln!("skipping qwen35 Metal embedding parity: QWEN35_NATIVE_GGUF unset");
            return;
        };
        let gguf = GgufModel::open(&path).expect("open Qwen3.5 GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 inventory");
        inventory
            .validate_core_tensors(&gguf)
            .expect("Qwen3.5 core tensors");
        let metal =
            MetalQwen35Model::from_gguf(&gguf, inventory, 248_044).expect("Metal Qwen model");

        let ids = [0_u32, 1, 42, 1024, 248_000];
        let gpu = metal.embed_tokens(&ids).expect("Metal token embeddings");
        let cpu = QuantMatrix::from_model(&gguf, "token_embd.weight")
            .expect("CPU token_embd matrix")
            .embedding_rows(&ids)
            .expect("CPU token embeddings");
        assert_eq!(gpu.len(), cpu.len());
        let max_abs = max_abs_diff(&gpu, &cpu);
        assert!(
            max_abs <= 2.0e-5,
            "Metal token embedding drift too high: {max_abs:.6e}"
        );
    }

    #[test]
    fn qwen35_metal_quant_matvec_matches_cpu_for_real_model_when_env_set() {
        let Some(path) = std::env::var_os("QWEN35_NATIVE_GGUF") else {
            eprintln!("skipping qwen35 Metal matvec parity: QWEN35_NATIVE_GGUF unset");
            return;
        };
        let gguf = GgufModel::open(&path).expect("open Qwen3.5 GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 inventory");
        inventory
            .validate_core_tensors(&gguf)
            .expect("Qwen3.5 core tensors");
        let metal =
            MetalQwen35Model::from_gguf(&gguf, inventory, 248_044).expect("Metal Qwen model");

        for name in [
            "blk.0.attn_qkv.weight",
            "blk.0.ssm_out.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_down.weight",
            "blk.3.attn_q.weight",
            "blk.3.attn_output.weight",
        ] {
            let matrix = QuantMatrix::from_model(&gguf, name).expect("CPU quant matrix");
            let input = patterned_input(matrix.cols());
            let gpu = metal.matvec(name, &input).expect("Metal quant matvec");
            let cpu = matrix.matmul(&input, 1).expect("CPU quant matvec");
            assert_eq!(gpu.len(), cpu.len(), "{name}");
            let max_abs = max_abs_diff(&gpu, &cpu);
            let cosine = cosine(&gpu, &cpu);
            let rms = rms_diff(&gpu, &cpu);
            let cpu_rms = rms_norm(&cpu).max(1.0e-6);
            eprintln!(
                "{name}: cosine={cosine:.8} rms_rel={:.6e} max_abs={max_abs:.6e}",
                rms / cpu_rms
            );
            assert!(
                cosine >= 0.999,
                "{name} Metal quant matvec cosine too low: {cosine:.8}, max_abs={max_abs:.6e}"
            );
            assert!(
                rms / cpu_rms <= 5.0e-2,
                "{name} Metal quant matvec rms_rel too high: {:.6e}, max_abs={max_abs:.6e}",
                rms / cpu_rms
            );
        }
    }

    #[test]
    fn qwen35_metal_quant_mul_mm_matches_cpu_for_real_model_when_env_set() {
        let Some(path) = std::env::var_os("QWEN35_NATIVE_GGUF") else {
            eprintln!("skipping qwen35 Metal mul_mm parity: QWEN35_NATIVE_GGUF unset");
            return;
        };
        let gguf = GgufModel::open(&path).expect("open Qwen3.5 GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 inventory");
        inventory
            .validate_core_tensors(&gguf)
            .expect("Qwen3.5 core tensors");
        let metal =
            MetalQwen35Model::from_gguf(&gguf, inventory, 248_044).expect("Metal Qwen model");

        let name = "blk.0.attn_qkv.weight";
        let matrix = QuantMatrix::from_model(&gguf, name).expect("CPU quant matrix");
        let rows = 16usize;
        let input = patterned_input(rows * matrix.cols());
        let gpu = metal
            .matmul_rows(name, &input, rows)
            .expect("Metal quant mul_mm");
        let cpu = matrix.matmul(&input, rows).expect("CPU quant mul_mm");
        assert_eq!(gpu.len(), cpu.len(), "{name}");
        let max_abs = max_abs_diff(&gpu, &cpu);
        let cosine = cosine(&gpu, &cpu);
        let rms = rms_diff(&gpu, &cpu);
        let cpu_rms = rms_norm(&cpu).max(1.0e-6);
        eprintln!(
            "{name} mul_mm rows={rows}: cosine={cosine:.8} rms_rel={:.6e} max_abs={max_abs:.6e}",
            rms / cpu_rms
        );
        assert!(
            cosine >= 0.999,
            "{name} Metal quant mul_mm cosine too low: {cosine:.8}, max_abs={max_abs:.6e}"
        );
        assert!(
            rms / cpu_rms <= 5.0e-2,
            "{name} Metal quant mul_mm rms_rel too high: {:.6e}, max_abs={max_abs:.6e}",
            rms / cpu_rms
        );
        assert!(
            max_abs <= 2.5e-1,
            "{name} Metal quant mul_mm max_abs too high: {max_abs:.6e}"
        );
    }

    #[test]
    fn qwen35_metal_batched_prefill_matches_tokenwise_when_env_set() {
        let Some(path) = std::env::var_os("QWEN35_NATIVE_GGUF") else {
            eprintln!("skipping qwen35 Metal batched prefill parity: QWEN35_NATIVE_GGUF unset");
            return;
        };
        let gguf = GgufModel::open(&path).expect("open Qwen3.5 GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 inventory");
        inventory
            .validate_core_tensors(&gguf)
            .expect("Qwen3.5 core tensors");
        let metal =
            MetalQwen35Model::from_gguf(&gguf, inventory.clone(), 248_044).expect("Metal Qwen");

        let next = 1234_u32;
        let max_context = 64;
        let all_prefill = [42_u32, 172, 2048, 4096, 8192, 16_384, 32_768, 65_536];
        for len in [1_usize, 2, 8] {
            let prefill = &all_prefill[..len];
            let mut tokenwise_state = metal
                .new_forward_state(max_context)
                .expect("tokenwise state");
            let mut tokenwise_ws = metal
                .new_forward_workspace(max_context)
                .expect("tokenwise workspace");
            for &token in prefill {
                metal
                    .prefill_token(token, &mut tokenwise_state, &mut tokenwise_ws)
                    .expect("tokenwise prefill");
            }
            let tokenwise = metal
                .forward_token_logits(next, &mut tokenwise_state, &mut tokenwise_ws)
                .expect("tokenwise logits");

            let mut batched_state = metal.new_forward_state(max_context).expect("batched state");
            let mut batched_ws = metal
                .new_forward_workspace(max_context)
                .expect("batched workspace");
            metal
                .prefill_tokens(prefill, &mut batched_state)
                .expect("batched prefill");
            let batched = metal
                .forward_token_logits(next, &mut batched_state, &mut batched_ws)
                .expect("batched logits");

            assert_eq!(tokenwise.len(), batched.len());
            let cosine = cosine(&tokenwise, &batched);
            let rms = rms_diff(&tokenwise, &batched);
            let rel = rms / rms_norm(&tokenwise).max(1.0e-6);
            let max_abs = max_abs_diff(&tokenwise, &batched);
            eprintln!(
                "batched prefill len={len} vs tokenwise: cosine={cosine:.8} rms_rel={rel:.6e} max_abs={max_abs:.6e}"
            );
            assert!(
                cosine >= 0.98,
                "batched prefill len={len} logits drift too far: cosine={cosine:.8}, rms_rel={rel:.6e}, max_abs={max_abs:.6e}"
            );
            assert!(
                rel <= 0.20,
                "batched prefill len={len} logits relative RMS too high: cosine={cosine:.8}, rms_rel={rel:.6e}, max_abs={max_abs:.6e}"
            );
        }
    }

    #[test]
    fn qwen35_metal_batched_hidden_matches_tokenwise_when_env_set() {
        let Some(path) = std::env::var_os("QWEN35_NATIVE_MTP_GGUF") else {
            return;
        };
        let gguf = GgufModel::open(&path).expect("open Qwen3.5 MTP GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 MTP inventory");
        let metal = MetalQwen35Model::from_gguf(&gguf, inventory.clone(), 248_044)
            .expect("Metal Qwen MTP model");
        let prefix = [42_u32, 314, 2718, 99];
        let verify = [1234_u32, 5678, 9012];
        let max_context = prefix.len() + verify.len() + 1;

        let mut tokenwise_state = metal
            .new_forward_state(max_context)
            .expect("Metal tokenwise hidden state");
        metal
            .prefill_tokens(&prefix, &mut tokenwise_state)
            .expect("Metal tokenwise hidden prefix");
        let mut tokenwise_workspace = metal
            .new_forward_workspace(max_context)
            .expect("Metal tokenwise hidden workspace");
        let mut tokenwise_hidden = Vec::new();
        let mut tokenwise_logits = Vec::new();
        for token in verify {
            let output = metal
                .forward_token_logits_hidden(token, &mut tokenwise_state, &mut tokenwise_workspace)
                .expect("Metal tokenwise hidden row");
            tokenwise_hidden.extend(output.hidden);
            tokenwise_logits.extend(output.logits);
        }

        let mut batched_state = metal
            .new_forward_state(max_context)
            .expect("Metal batched hidden state");
        metal
            .prefill_tokens(&prefix, &mut batched_state)
            .expect("Metal batched hidden prefix");
        let batched = metal
            .forward_tokens_logits_hidden(&verify, &mut batched_state)
            .expect("Metal batched hidden rows");
        let hidden_cosine = cosine(&tokenwise_hidden, &batched.hidden);
        let logits_cosine = cosine(&tokenwise_logits, &batched.logits);
        eprintln!(
            "Metal batched hidden: hidden_cos={hidden_cosine:.8} logits_cos={logits_cosine:.8}"
        );
        assert!(hidden_cosine >= 0.98, "Metal batched hidden-state drift");
        assert!(logits_cosine >= 0.98, "Metal batched hidden-logit drift");
    }

    #[test]
    fn qwen35_metal_mtp_matches_golden_tokens_when_env_set() {
        let (Some(gguf_path), Some(tokenizer_path)) = (
            std::env::var_os("QWEN35_NATIVE_MTP_GGUF"),
            std::env::var_os("QWEN35_NATIVE_TOKENIZER"),
        ) else {
            return;
        };
        let gguf = GgufModel::open(&gguf_path).expect("open Qwen3.5 MTP GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 MTP inventory");
        let metal = MetalQwen35Model::from_gguf(&gguf, inventory.clone(), 248_044)
            .expect("Metal Qwen MTP model");
        let tokenizer = Tokenizer::from_file(tokenizer_path).expect("load Qwen3.5 tokenizer");
        let prompt = crate::prompt::non_thinking_chat_prompt(
            "Summarize: What is this function for?\n\npub fn rename_by_rules(&mut self, rules: RenameAllRules) {\n    self.serialize.value = rules.serialize.apply_to_field(&self.serialize.value);\n    self.deserialize.value = rules.deserialize.apply_to_field(&self.deserialize.value);\n}",
        );
        let ids = tokenizer
            .encode(prompt, true)
            .expect("tokenize Metal MTP golden prompt")
            .get_ids()
            .to_vec();
        let max_context = ids.len() + 3;
        let mut target_state = metal
            .new_forward_state(max_context)
            .expect("Metal MTP target state");
        let mut prompt_hidden = metal
            .prefill_tokens_hidden(&ids[..ids.len() - 1], &mut target_state)
            .expect("Metal MTP target prompt hidden");
        let mut target_workspace = metal
            .new_forward_workspace(max_context)
            .expect("Metal MTP target workspace");
        let target = metal
            .forward_token_logits_hidden(
                *ids.last().expect("non-empty Metal MTP prompt"),
                &mut target_state,
                &mut target_workspace,
            )
            .expect("Metal MTP target final row");
        prompt_hidden.extend_from_slice(&target.hidden);
        let first = greedy_argmax(&target.logits);
        assert_eq!(first, 1919, "unexpected Metal MTP target token");

        let hidden_size = inventory.hidden_size;
        let mut conditioning = vec![0.0f32; prompt_hidden.len()];
        conditioning[hidden_size..]
            .copy_from_slice(&prompt_hidden[..prompt_hidden.len() - hidden_size]);
        let mut mtp_state = metal.new_mtp_state(max_context).expect("Metal MTP state");
        metal
            .mtp_prefill_tokens(&ids, &conditioning, &mut mtp_state)
            .expect("Metal MTP prompt catch-up");
        let first_draft = metal
            .mtp_forward_tokens_logits_hidden(&[first], &target.hidden, &mut mtp_state)
            .expect("Metal MTP first draft");
        let first_draft_token = greedy_argmax(&first_draft.logits);
        let second_draft = metal
            .mtp_forward_tokens_logits_hidden(
                &[first_draft_token],
                &first_draft.hidden,
                &mut mtp_state,
            )
            .expect("Metal MTP second draft");
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
        let mut fresh_metal_state = metal
            .new_mtp_state(max_context)
            .expect("fresh Metal MTP comparison state");
        let fresh_metal = metal
            .mtp_forward_tokens_logits_hidden(&[first], &target.hidden, &mut fresh_metal_state)
            .expect("fresh Metal MTP comparison draft");
        let mut fresh_cpu_state = cpu
            .new_mtp_state(max_context)
            .expect("fresh CPU MTP comparison state");
        let fresh_cpu = cpu
            .mtp_forward_tokens_logits_hidden(&[first], &target.hidden, &mut fresh_cpu_state)
            .expect("fresh CPU MTP comparison draft");
        eprintln!(
            "Metal/CPU MTP fresh hidden_cos={:.8} logits_cos={:.8}; first hidden_cos={:.8} logits_cos={:.8}; second logits_cos={:.8}",
            cosine(&fresh_metal.hidden, &fresh_cpu.hidden),
            cosine(&fresh_metal.logits, &fresh_cpu.logits),
            cosine(&first_draft.hidden, &cpu_first.hidden),
            cosine(&first_draft.logits, &cpu_first.logits),
            cosine(&second_draft.logits, &cpu_second.logits),
        );
        assert_eq!(
            [first_draft_token, greedy_argmax(&second_draft.logits)],
            [709, 369],
            "Metal MTP drafts differ from llama.cpp golden tokens"
        );
    }

    #[test]
    fn qwen35_metal_mtp_generation_matches_target_when_env_set() {
        let (Some(gguf_path), Some(tokenizer_path)) = (
            std::env::var_os("QWEN35_NATIVE_MTP_GGUF"),
            std::env::var_os("QWEN35_NATIVE_TOKENIZER"),
        ) else {
            return;
        };
        let gguf = GgufModel::open(&gguf_path).expect("open Qwen3.5 MTP GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 MTP inventory");
        let metal =
            MetalQwen35Model::from_gguf(&gguf, inventory, 248_044).expect("Metal Qwen MTP model");
        let tokenizer = Tokenizer::from_file(tokenizer_path).expect("load Qwen3.5 tokenizer");
        let prompt = crate::prompt::non_thinking_chat_prompt(
            "Summarize: What is this function for?\n\npub fn add_user(users: &mut Vec<String>, name: &str) -> usize {\n    users.push(name.trim().to_string());\n    users.len()\n}",
        );
        let params = GenerationParams {
            max_tokens: 32,
            ..crate::BRIEF_GENERATION_PARAMS
        };
        let expected = metal
            .generate_target_only_for_test(&tokenizer, &prompt, params)
            .expect("Metal target-only generation");
        let actual = metal
            .generate(&tokenizer, &prompt, params)
            .expect("Metal MTP generation");
        assert_eq!(actual, expected, "Metal MTP changed target sampling output");
    }

    #[test]
    fn qwen35_metal_mtp_input_projection_matches_cpu_when_env_set() {
        let Some(path) = std::env::var_os("QWEN35_NATIVE_MTP_GGUF") else {
            return;
        };
        let gguf = GgufModel::open(&path).expect("open Qwen3.5 MTP GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 MTP inventory");
        let metal = MetalQwen35Model::from_gguf(&gguf, inventory.clone(), 248_044)
            .expect("Metal Qwen MTP model");
        let layer = inventory.block_count;
        let prefix = format!("blk.{layer}.nextn");
        let hidden_size = inventory.hidden_size;
        let token = 1919_u32;
        let target_hidden = patterned_input(hidden_size);
        let embedding = QuantMatrix::from_model(&gguf, "token_embd.weight")
            .expect("CPU MTP embedding")
            .embedding_rows(&[token])
            .expect("CPU MTP embedding row");
        let enorm = gguf
            .tensor(&format!("{prefix}.enorm.weight"))
            .expect("CPU MTP enorm")
            .to_f32()
            .expect("CPU MTP enorm values");
        let hnorm = gguf
            .tensor(&format!("{prefix}.hnorm.weight"))
            .expect("CPU MTP hnorm")
            .to_f32()
            .expect("CPU MTP hnorm values");
        let embedding = cpu_rms_norm(&embedding, &enorm, hidden_size, true);
        let target_hidden = cpu_rms_norm(&target_hidden, &hnorm, hidden_size, true);
        let mut joined_expected = embedding;
        joined_expected.extend_from_slice(&target_hidden);
        let hidden_expected = QuantMatrix::from_model(&gguf, &format!("{prefix}.eh_proj.weight"))
            .expect("CPU MTP eh_proj")
            .matmul(&joined_expected, 1)
            .expect("CPU MTP eh_proj row");

        let workspace = metal
            .new_prefill_workspace(1, 4)
            .expect("Metal MTP projection workspace");
        unsafe {
            workspace.token_ids.write(0, &[token]);
        }
        let target_buffer = metal
            .upload_pod(&patterned_input(hidden_size))
            .expect("upload MTP target hidden");
        let joined = metal
            .new_f32(hidden_size * 2)
            .expect("Metal MTP joined buffer");
        let cb = metal
            .command_buffer()
            .expect("Metal MTP projection command");
        let enc = cb.compute().expect("Metal MTP projection encoder");
        metal
            .encode_embed_tokens(&enc, &workspace.token_ids, &workspace.hidden, 1)
            .expect("Metal MTP embed");
        enc.memory_barrier_buffers();
        metal
            .encode_rms_norm(
                &enc,
                &format!("{prefix}.enorm.weight"),
                &workspace.hidden,
                &workspace.normed,
                1,
                hidden_size,
                true,
            )
            .expect("Metal MTP enorm");
        metal
            .encode_rms_norm(
                &enc,
                &format!("{prefix}.hnorm.weight"),
                &target_buffer,
                &workspace.attn_out,
                1,
                hidden_size,
                true,
            )
            .expect("Metal MTP hnorm");
        enc.memory_barrier_buffers();
        metal
            .encode_concat_rows(
                &enc,
                &workspace.normed,
                &workspace.attn_out,
                &joined,
                1,
                hidden_size,
            )
            .expect("Metal MTP concat");
        enc.memory_barrier_buffers();
        metal
            .encode_matmul_rows_to(
                &enc,
                &format!("{prefix}.eh_proj.weight"),
                &joined,
                1,
                hidden_size * 2,
                &workspace.hidden,
                hidden_size,
            )
            .expect("Metal MTP eh_proj");
        enc.end();
        cb.commit_and_wait().expect("Metal MTP projection wait");
        let joined_actual = metal
            .read_f32(&joined, hidden_size * 2)
            .expect("read Metal MTP joined row");
        let hidden_actual = metal
            .read_f32(&workspace.hidden, hidden_size)
            .expect("read Metal MTP projected row");
        eprintln!(
            "Metal MTP input projection joined_cos={:.8} hidden_cos={:.8}",
            cosine(&joined_actual, &joined_expected),
            cosine(&hidden_actual, &hidden_expected),
        );
        assert!(
            cosine(&joined_actual, &joined_expected) >= 0.999,
            "Metal MTP joined input drift"
        );
        assert!(
            cosine(&hidden_actual, &hidden_expected) >= 0.999,
            "Metal MTP input projection drift"
        );
    }

    #[test]
    #[ignore = "diagnostic Metal layer profile"]
    fn qwen35_metal_prefill_layer_profile_when_env_set() {
        let Some(path) = std::env::var_os("QWEN35_NATIVE_GGUF") else {
            eprintln!("skipping qwen35 Metal layer profile: QWEN35_NATIVE_GGUF unset");
            return;
        };
        let gguf = GgufModel::open(&path).expect("open Qwen3.5 GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 inventory");
        inventory
            .validate_core_tensors(&gguf)
            .expect("Qwen3.5 core tensors");
        let metal =
            MetalQwen35Model::from_gguf(&gguf, inventory.clone(), 248_044).expect("Metal Qwen");
        let rows = std::env::var("QWEN35_NATIVE_METAL_PROFILE_ROWS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(511)
            .clamp(1, METAL_PREFILL_BATCH_ROWS);
        let tokens = (0..rows)
            .map(|row| ((row * 997 + 42) % inventory.vocab_size) as u32)
            .collect::<Vec<_>>();
        let mut state = metal
            .new_forward_state(rows + 1)
            .expect("Metal profile state");
        let ws = metal
            .new_prefill_workspace(rows, rows + 1)
            .expect("Metal profile workspace");
        unsafe {
            ws.token_ids.write(0, &tokens);
        }

        let dispatch_start = greppy_embed_native::metal::ffi::dispatch_count_snapshot();
        let cb = metal.command_buffer().expect("Metal profile embed command");
        let enc = cb.compute().expect("Metal profile embed encoder");
        metal
            .encode_embed_tokens(&enc, &ws.token_ids, &ws.hidden, rows)
            .expect("Metal profile embedding");
        enc.end();
        let timing = cb
            .commit_and_wait_timed()
            .expect("Metal profile embed wait");
        eprintln!(
            "metal_prefill_profile stage=embed rows={rows} kernel_ms={:.3} gpu_ms={:.3} wait_ms={:.3} dispatches={}",
            timing.kernel_secs * 1.0e3,
            timing.gpu_secs * 1.0e3,
            timing.submit_wait_secs * 1.0e3,
            greppy_embed_native::metal::ffi::dispatch_count_snapshot() - dispatch_start,
        );

        let mut kernel_secs = timing.kernel_secs;
        let mut gpu_secs = timing.gpu_secs;
        for layer in 0..inventory.block_count {
            let kind = if inventory.is_full_attention_layer(layer) {
                "full"
            } else {
                "delta"
            };
            let dispatch_start = greppy_embed_native::metal::ffi::dispatch_count_snapshot();
            let cb = metal.command_buffer().expect("Metal profile layer command");
            let enc = cb.compute().expect("Metal profile layer encoder");
            let final_layer = metal
                .encode_prefill_layer(&enc, layer, &mut state, rows, &ws, true)
                .expect("Metal profile layer encode");
            enc.end();
            let timing = cb
                .commit_and_wait_timed()
                .expect("Metal profile layer wait");
            kernel_secs += timing.kernel_secs;
            gpu_secs += timing.gpu_secs;
            eprintln!(
                "metal_prefill_profile stage=layer layer={layer} kind={kind} final={final_layer} kernel_ms={:.3} gpu_ms={:.3} wait_ms={:.3} dispatches={}",
                timing.kernel_secs * 1.0e3,
                timing.gpu_secs * 1.0e3,
                timing.submit_wait_secs * 1.0e3,
                greppy_embed_native::metal::ffi::dispatch_count_snapshot() - dispatch_start,
            );
            if final_layer {
                break;
            }
        }
        eprintln!(
            "metal_prefill_profile stage=total rows={rows} kernel_ms={:.3} gpu_ms={:.3} kernel_tok_s={:.2} gpu_tok_s={:.2}",
            kernel_secs * 1.0e3,
            gpu_secs * 1.0e3,
            rows as f64 / kernel_secs.max(1.0e-9),
            rows as f64 / gpu_secs.max(1.0e-9),
        );
    }

    #[test]
    #[ignore = "diagnostic Metal stage profile"]
    fn qwen35_metal_prefill_stage_profile_when_env_set() {
        let Some(path) = std::env::var_os("QWEN35_NATIVE_GGUF") else {
            eprintln!("skipping qwen35 Metal stage profile: QWEN35_NATIVE_GGUF unset");
            return;
        };
        let gguf = GgufModel::open(&path).expect("open Qwen3.5 GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 inventory");
        inventory
            .validate_core_tensors(&gguf)
            .expect("Qwen3.5 core tensors");
        let metal =
            MetalQwen35Model::from_gguf(&gguf, inventory.clone(), 248_044).expect("Metal Qwen");
        let rows = std::env::var("QWEN35_NATIVE_METAL_PROFILE_ROWS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(511)
            .clamp(1, METAL_PREFILL_BATCH_ROWS);
        let tokens = (0..rows)
            .map(|row| ((row * 997 + 42) % inventory.vocab_size) as u32)
            .collect::<Vec<_>>();

        let mut warm_state = metal.new_forward_state(33).expect("Metal warm state");
        metal
            .prefill_tokens(&tokens[..rows.min(32)], &mut warm_state)
            .expect("Metal profile warmup");

        let mut state = metal
            .new_forward_state(rows + 1)
            .expect("Metal profile state");
        let ws = metal
            .new_prefill_workspace(rows, rows + 1)
            .expect("Metal profile workspace");
        unsafe {
            ws.token_ids.write(0, &tokens);
        }
        profile_stage(&metal, "embed", |enc| {
            metal.encode_embed_tokens(enc, &ws.token_ids, &ws.hidden, rows)
        });

        for layer in [0usize, inventory.full_attention_interval - 1] {
            let kind = if inventory.is_full_attention_layer(layer) {
                "full"
            } else {
                "delta"
            };
            profile_stage(&metal, &format!("{kind}.attn_norm"), |enc| {
                metal.encode_rms_norm(
                    enc,
                    &format!("blk.{layer}.attn_norm.weight"),
                    &ws.hidden,
                    &ws.normed,
                    rows,
                    inventory.hidden_size,
                    true,
                )
            });
            profile_stage(
                &metal,
                &format!("{kind}.attention"),
                |enc| match &mut state.layer_states[layer] {
                    MetalLayerState::Delta { recurrent, conv } => metal
                        .encode_delta_attention_block_rows(enc, layer, recurrent, conv, rows, &ws),
                    MetalLayerState::Full { k_cache, v_cache } => metal
                        .encode_full_attention_block_rows(
                            enc,
                            layer,
                            k_cache,
                            v_cache,
                            state.position,
                            rows,
                            state.max_context,
                            &ws,
                        ),
                },
            );
            profile_stage(&metal, &format!("{kind}.post_attn_norm"), |enc| {
                metal.encode_add_rms_norm(
                    enc,
                    &format!("blk.{layer}.post_attention_norm.weight"),
                    &ws.hidden,
                    &ws.attn_out,
                    &ws.hidden,
                    &ws.normed,
                    rows,
                    inventory.hidden_size,
                )
            });
            profile_stage(&metal, &format!("{kind}.ffn"), |enc| {
                metal.encode_ffn_block_rows(enc, layer, rows, &ws)
            });
            profile_stage(&metal, &format!("{kind}.residual"), |enc| {
                metal.encode_add(
                    enc,
                    &ws.hidden,
                    &ws.attn_out,
                    &ws.hidden,
                    rows * inventory.hidden_size,
                )
            });
        }
    }

    #[test]
    fn qwen35_metal_rms_norm_matches_cpu_for_real_model_when_env_set() {
        let Some(path) = std::env::var_os("QWEN35_NATIVE_GGUF") else {
            eprintln!("skipping qwen35 Metal RMSNorm parity: QWEN35_NATIVE_GGUF unset");
            return;
        };
        let gguf = GgufModel::open(&path).expect("open Qwen3.5 GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 inventory");
        inventory
            .validate_core_tensors(&gguf)
            .expect("Qwen3.5 core tensors");
        let metal =
            MetalQwen35Model::from_gguf(&gguf, inventory.clone(), 248_044).expect("Metal Qwen");

        let hidden_input = patterned_input(2 * inventory.hidden_size);
        let hidden_weight = gguf
            .tensor("blk.0.attn_norm.weight")
            .expect("attn_norm tensor")
            .to_f32()
            .expect("attn_norm f32");
        let gpu = metal
            .qwen_rms_norm(
                "blk.0.attn_norm.weight",
                &hidden_input,
                inventory.hidden_size,
            )
            .expect("Metal Qwen RMSNorm");
        let cpu = cpu_rms_norm(&hidden_input, &hidden_weight, inventory.hidden_size, true);
        assert_close_vec("blk.0.attn_norm.weight", &gpu, &cpu, 1.0e-4);

        let ssm_dim = inventory.ssm_inner_size / inventory.ssm_group_count;
        let ssm_input = patterned_input(3 * ssm_dim);
        let ssm_weight = gguf
            .tensor("blk.0.ssm_norm.weight")
            .expect("ssm_norm tensor")
            .to_f32()
            .expect("ssm_norm f32");
        let gpu = metal
            .plain_rms_norm("blk.0.ssm_norm.weight", &ssm_input, ssm_dim)
            .expect("Metal plain RMSNorm");
        let cpu = cpu_rms_norm(&ssm_input, &ssm_weight, ssm_dim, false);
        assert_close_vec("blk.0.ssm_norm.weight", &gpu, &cpu, 1.0e-4);

        let lhs = patterned_input(inventory.hidden_size);
        let rhs = patterned_input(inventory.hidden_size)
            .into_iter()
            .map(|v| v * -0.25)
            .collect::<Vec<_>>();
        let (sum, norm) = metal
            .add_rms_norm(
                "blk.0.post_attention_norm.weight",
                &lhs,
                &rhs,
                inventory.hidden_size,
            )
            .expect("Metal add_rms_norm");
        let sum_cpu = lhs.iter().zip(&rhs).map(|(a, b)| a + b).collect::<Vec<_>>();
        let post_weight = gguf
            .tensor("blk.0.post_attention_norm.weight")
            .expect("post_attention_norm tensor")
            .to_f32()
            .expect("post_attention_norm f32");
        let norm_cpu = cpu_rms_norm(&sum_cpu, &post_weight, inventory.hidden_size, true);
        assert_close_vec("Qwen add_rms_norm sum", &sum, &sum_cpu, 1.0e-6);
        assert_close_vec("Qwen add_rms_norm norm", &norm, &norm_cpu, 1.0e-4);
    }

    #[test]
    fn qwen35_metal_delta_preprocess_matches_cpu_for_real_model_when_env_set() {
        let Some(path) = std::env::var_os("QWEN35_NATIVE_GGUF") else {
            eprintln!("skipping qwen35 Metal delta preprocess parity: QWEN35_NATIVE_GGUF unset");
            return;
        };
        let gguf = GgufModel::open(&path).expect("open Qwen3.5 GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 inventory");
        inventory
            .validate_core_tensors(&gguf)
            .expect("Qwen3.5 core tensors");
        let metal =
            MetalQwen35Model::from_gguf(&gguf, inventory.clone(), 248_044).expect("Metal Qwen");

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
        let (gpu_values, gpu_state) = metal
            .causal_conv1d_silu("blk.0.ssm_conv1d.weight", &values, &state, kernel)
            .expect("Metal causal_conv1d_silu");
        assert_close_vec("Metal conv values", &gpu_values, &cpu_values, 1.0e-4);
        assert_close_vec("Metal conv state", &gpu_state, &cpu_state, 1.0e-6);

        let heads = inventory.ssm_group_count;
        let head_dim = inventory.ssm_inner_size / heads;
        let q = patterned_input(inventory.ssm_inner_size);
        let k = patterned_input(inventory.ssm_inner_size)
            .into_iter()
            .map(|v| v * 0.75 - 0.125)
            .collect::<Vec<_>>();
        let mut cpu_q = q.clone();
        let mut cpu_k = k.clone();
        cpu_normalize_linear_qk(&mut cpu_q, &mut cpu_k, heads, head_dim);
        let (gpu_q, gpu_k) = metal
            .normalize_linear_qk(&q, &k, heads, head_dim)
            .expect("Metal normalize_linear_qk");
        assert_close_vec("Metal normalize_linear_qk q", &gpu_q, &cpu_q, 1.0e-5);
        assert_close_vec("Metal normalize_linear_qk k", &gpu_k, &cpu_k, 1.0e-5);
    }

    #[test]
    fn qwen35_metal_swiglu_and_gate_match_cpu_for_real_model_when_env_set() {
        let Some(path) = std::env::var_os("QWEN35_NATIVE_GGUF") else {
            eprintln!("skipping qwen35 Metal SwiGLU parity: QWEN35_NATIVE_GGUF unset");
            return;
        };
        let gguf = GgufModel::open(&path).expect("open Qwen3.5 GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 inventory");
        inventory
            .validate_core_tensors(&gguf)
            .expect("Qwen3.5 core tensors");
        let metal =
            MetalQwen35Model::from_gguf(&gguf, inventory.clone(), 248_044).expect("Metal Qwen");

        let hidden = patterned_input(inventory.hidden_size)
            .into_iter()
            .map(|v| v * 0.1)
            .collect::<Vec<_>>();
        let gate = metal
            .matvec("blk.0.ffn_gate.weight", &hidden)
            .expect("Metal ffn_gate");
        let up = metal
            .matvec("blk.0.ffn_up.weight", &hidden)
            .expect("Metal ffn_up");
        let gpu = metal.swiglu(&gate, &up).expect("Metal SwiGLU");
        let cpu = gate
            .iter()
            .zip(&up)
            .map(|(gate, up)| silu(*gate) * *up)
            .collect::<Vec<_>>();
        assert_close_vec("Metal SwiGLU", &gpu, &cpu, 1.0e-5);

        let values = patterned_input(inventory.ssm_inner_size);
        let gate = gate[..inventory.ssm_inner_size].to_vec();
        let gpu = metal
            .apply_silu_gate(&values, &gate)
            .expect("Metal apply_silu_gate");
        let cpu = values
            .iter()
            .zip(&gate)
            .map(|(value, gate)| *value * silu(*gate))
            .collect::<Vec<_>>();
        assert_close_vec("Metal SiLU gate", &gpu, &cpu, 1.0e-5);
    }

    #[test]
    fn qwen35_metal_deltanet_decode_matches_cpu_for_real_model_when_env_set() {
        let Some(path) = std::env::var_os("QWEN35_NATIVE_GGUF") else {
            eprintln!("skipping qwen35 Metal DeltaNet parity: QWEN35_NATIVE_GGUF unset");
            return;
        };
        let gguf = GgufModel::open(&path).expect("open Qwen3.5 GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 inventory");
        inventory
            .validate_core_tensors(&gguf)
            .expect("Qwen3.5 core tensors");
        let metal =
            MetalQwen35Model::from_gguf(&gguf, inventory.clone(), 248_044).expect("Metal Qwen");

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
        let (gpu, gpu_state) = metal
            .deltanet_decode(0, &q, &k, &v, &beta, &alpha, &recurrent)
            .expect("Metal DeltaNet decode");
        assert_close_vec("Metal DeltaNet out", &gpu, &cpu, 2.0e-5);
        assert_close_vec("Metal DeltaNet state", &gpu_state, &cpu_state, 2.0e-5);
    }

    #[test]
    fn qwen35_metal_deltanet_rows_match_cpu_for_real_model_when_env_set() {
        let Some(path) = std::env::var_os("QWEN35_NATIVE_GGUF") else {
            eprintln!("skipping qwen35 Metal DeltaNet rows parity: QWEN35_NATIVE_GGUF unset");
            return;
        };
        let gguf = GgufModel::open(&path).expect("open Qwen3.5 GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 inventory");
        inventory
            .validate_core_tensors(&gguf)
            .expect("Qwen3.5 core tensors");
        let metal =
            MetalQwen35Model::from_gguf(&gguf, inventory.clone(), 248_044).expect("Metal Qwen");
        let rows = 2usize;
        let heads = inventory.ssm_group_count;
        let head_dim = inventory.ssm_inner_size / heads;
        let inner = inventory.ssm_inner_size;
        let q = patterned_input(rows * inner)
            .into_iter()
            .map(|v| v * 0.01)
            .collect::<Vec<_>>();
        let k = patterned_input(rows * inner)
            .into_iter()
            .map(|v| v * -0.0125)
            .collect::<Vec<_>>();
        let v = patterned_input(rows * inner)
            .into_iter()
            .map(|v| v * 0.2)
            .collect::<Vec<_>>();
        let beta = patterned_input(rows * heads);
        let alpha = patterned_input(rows * heads)
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
        let mut cpu = Vec::with_capacity(rows * inner);
        for row in 0..rows {
            cpu.extend(cpu_deltanet_decode(
                &q[row * inner..(row + 1) * inner],
                &k[row * inner..(row + 1) * inner],
                &v[row * inner..(row + 1) * inner],
                &beta[row * heads..(row + 1) * heads],
                &alpha[row * heads..(row + 1) * heads],
                &a_log,
                &dt_bias,
                &mut cpu_state,
                heads,
                head_dim,
            ));
        }

        let q_buf = metal.upload_pod(&q).expect("q buffer");
        let k_buf = metal.upload_pod(&k).expect("k buffer");
        let v_buf = metal.upload_pod(&v).expect("v buffer");
        let beta_buf = metal.upload_pod(&beta).expect("beta buffer");
        let alpha_buf = metal.upload_pod(&alpha).expect("alpha buffer");
        let state_buf = metal.upload_pod(&recurrent).expect("state buffer");
        let out_buf = metal.new_f32(rows * inner).expect("out buffer");
        let a_log_w = metal
            .require_f32_weight("blk.0.ssm_a", heads)
            .expect("ssm_a");
        let dt_bias_w = metal
            .require_f32_weight("blk.0.ssm_dt.bias", heads)
            .expect("ssm_dt.bias");
        let cb = metal.command_buffer().expect("command buffer");
        let enc = cb.compute().expect("compute encoder");
        ok(ops::op_qwen_deltanet_decode_rows_f32(
            &enc,
            metal.dev,
            &q_buf,
            0,
            &k_buf,
            0,
            &v_buf,
            0,
            &beta_buf,
            &alpha_buf,
            &a_log_w.buffer,
            a_log_w.offset,
            &dt_bias_w.buffer,
            dt_bias_w.offset,
            &state_buf,
            &out_buf,
            checked_i32(rows, "rows").expect("rows i32"),
            checked_i32(heads, "heads").expect("heads i32"),
            checked_i32(head_dim, "head dim").expect("head dim i32"),
            checked_i32(inner, "q stride").expect("q stride i32"),
            checked_i32(inner, "k stride").expect("k stride i32"),
            checked_i32(inner, "v stride").expect("v stride i32"),
            checked_i32(heads, "beta stride").expect("beta stride i32"),
            checked_i32(heads, "alpha stride").expect("alpha stride i32"),
            checked_i32(inner, "out stride").expect("out stride i32"),
        ))
        .expect("DeltaNet rows dispatch");
        enc.end();
        cb.commit_and_wait().expect("DeltaNet rows wait");
        let gpu = metal.read_f32(&out_buf, rows * inner).expect("out read");
        let gpu_state = metal
            .read_f32(&state_buf, recurrent.len())
            .expect("state read");
        assert_close_vec("Metal DeltaNet rows out", &gpu, &cpu, 2.0e-5);
        assert_close_vec("Metal DeltaNet rows state", &gpu_state, &cpu_state, 2.0e-5);
    }

    #[test]
    fn qwen35_metal_conv_rows_match_decode_steps_for_real_model_when_env_set() {
        let Some(path) = std::env::var_os("QWEN35_NATIVE_GGUF") else {
            eprintln!("skipping qwen35 Metal conv rows parity: QWEN35_NATIVE_GGUF unset");
            return;
        };
        let gguf = GgufModel::open(&path).expect("open Qwen3.5 GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 inventory");
        inventory
            .validate_core_tensors(&gguf)
            .expect("Qwen3.5 core tensors");
        let metal =
            MetalQwen35Model::from_gguf(&gguf, inventory.clone(), 248_044).expect("Metal Qwen");

        let rows = 3usize;
        let channels = inventory.ssm_inner_size * 3;
        let values = patterned_input(rows * channels)
            .into_iter()
            .map(|v| v * 0.25)
            .collect::<Vec<_>>();
        let state = patterned_input(channels * CONV_KERNEL)
            .into_iter()
            .map(|v| v * 0.1)
            .collect::<Vec<_>>();
        let weight = gguf
            .tensor("blk.0.ssm_conv1d.weight")
            .expect("ssm_conv1d tensor")
            .to_f32()
            .expect("ssm_conv1d f32");

        let mut cpu_values = values.clone();
        let mut cpu_state = state.clone();
        for row in 0..rows {
            cpu_causal_conv1d_silu(
                &mut cpu_values[row * channels..(row + 1) * channels],
                &weight,
                &mut cpu_state,
                CONV_KERNEL,
            );
        }

        let values_buf = metal.upload_pod(&values).expect("values buffer");
        let state_buf = metal.upload_pod(&state).expect("state buffer");
        let weight_buf = metal.upload_pod(&weight).expect("weight buffer");
        let cb = metal.command_buffer().expect("command buffer");
        let enc = cb.compute().expect("compute encoder");
        ok(ops::op_qwen_causal_conv1d_silu_rows_f32(
            &enc,
            metal.dev,
            &values_buf,
            &weight_buf,
            0,
            &state_buf,
            checked_i32(rows, "conv rows").expect("rows i32"),
            checked_i32(channels, "conv channels").expect("channels i32"),
            checked_i32(CONV_KERNEL, "conv kernel").expect("kernel i32"),
        ))
        .expect("conv rows dispatch");
        enc.end();
        cb.commit_and_wait().expect("conv rows wait");

        let gpu_values = metal
            .read_f32(&values_buf, rows * channels)
            .expect("values read");
        let gpu_state = metal
            .read_f32(&state_buf, channels * CONV_KERNEL)
            .expect("state read");
        assert_close_vec("Metal conv rows values", &gpu_values, &cpu_values, 2.0e-5);
        assert_close_vec("Metal conv rows state", &gpu_state, &cpu_state, 1.0e-6);
    }

    #[test]
    fn qwen35_metal_strided_rms_norm_rows_match_cpu_for_real_model_when_env_set() {
        let Some(path) = std::env::var_os("QWEN35_NATIVE_GGUF") else {
            eprintln!("skipping qwen35 Metal strided RMSNorm parity: QWEN35_NATIVE_GGUF unset");
            return;
        };
        let gguf = GgufModel::open(&path).expect("open Qwen3.5 GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 inventory");
        inventory
            .validate_core_tensors(&gguf)
            .expect("Qwen3.5 core tensors");
        let metal =
            MetalQwen35Model::from_gguf(&gguf, inventory.clone(), 248_044).expect("Metal Qwen");

        let layer = inventory.full_attention_interval - 1;
        let rows = 2usize;
        let heads = inventory.attention_heads;
        let head_dim = inventory.head_dim;
        let stride = heads * head_dim * 2;
        let mut values = patterned_input(rows * stride)
            .into_iter()
            .map(|v| v * 0.07)
            .collect::<Vec<_>>();
        let weight = gguf
            .tensor(&format!("blk.{layer}.attn_q_norm.weight"))
            .expect("attn_q_norm tensor")
            .to_f32()
            .expect("attn_q_norm f32");
        let mut cpu = values.clone();
        for row in 0..rows {
            for head in 0..heads {
                let base = row * stride + head * head_dim;
                let normed = cpu_rms_norm(&cpu[base..base + head_dim], &weight, head_dim, true);
                cpu[base..base + head_dim].copy_from_slice(&normed);
            }
        }

        let values_buf = metal.upload_pod(&values).expect("values buffer");
        let weight_buf = metal.upload_pod(&weight).expect("weight buffer");
        let cb = metal.command_buffer().expect("command buffer");
        let enc = cb.compute().expect("compute encoder");
        ok(ops::op_qwen_rms_norm_strided_rows_f32(
            &enc,
            metal.dev,
            &values_buf,
            0,
            &weight_buf,
            0,
            checked_i32(rows, "rms rows").expect("rows i32"),
            checked_i32(heads, "rms heads").expect("heads i32"),
            checked_i32(head_dim, "rms head dim").expect("head dim i32"),
            checked_i32(stride, "rms stride").expect("stride i32"),
            RMS_EPS,
        ))
        .expect("strided RMSNorm rows dispatch");
        enc.end();
        cb.commit_and_wait().expect("strided RMSNorm rows wait");

        values = metal
            .read_f32(&values_buf, rows * stride)
            .expect("values read");
        assert_close_vec("Metal strided RMSNorm rows", &values, &cpu, 2.0e-5);
    }

    #[test]
    fn qwen35_metal_attention_rows_match_decode_steps_for_real_model_when_env_set() {
        let Some(path) = std::env::var_os("QWEN35_NATIVE_GGUF") else {
            eprintln!("skipping qwen35 Metal attention rows parity: QWEN35_NATIVE_GGUF unset");
            return;
        };
        let gguf = GgufModel::open(&path).expect("open Qwen3.5 GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 inventory");
        inventory
            .validate_core_tensors(&gguf)
            .expect("Qwen3.5 core tensors");
        let metal =
            MetalQwen35Model::from_gguf(&gguf, inventory.clone(), 248_044).expect("Metal Qwen");

        let rows = 2usize;
        let position = 3usize;
        let max_context = 8usize;
        let q_heads = inventory.attention_heads;
        let kv_heads = inventory.kv_heads;
        let head_dim = inventory.head_dim;
        let value_dim = inventory.value_dim;
        let q_dim = q_heads * head_dim;
        let kv_k_dim = kv_heads * head_dim;
        let kv_v_dim = kv_heads * value_dim;
        let q_stride = q_dim * 2;
        let score_stride = q_heads * max_context;

        let q_fused = patterned_input(rows * q_stride)
            .into_iter()
            .map(|v| v * 0.04)
            .collect::<Vec<_>>();
        let k_rows = patterned_input(rows * kv_k_dim)
            .into_iter()
            .map(|v| v * -0.03)
            .collect::<Vec<_>>();
        let v_rows = patterned_input(rows * kv_v_dim)
            .into_iter()
            .map(|v| v * 0.02)
            .collect::<Vec<_>>();
        let initial_k_cache = patterned_input(max_context * kv_k_dim)
            .into_iter()
            .map(|v| v * 0.015)
            .collect::<Vec<_>>();
        let initial_v_cache = patterned_input(max_context * kv_v_dim)
            .into_iter()
            .map(|v| v * -0.02)
            .collect::<Vec<_>>();
        let mut k_cache = initial_k_cache.clone();
        let mut v_cache = initial_v_cache.clone();

        for row in 0..rows {
            let k_dst = (position + row) * kv_k_dim;
            let v_dst = (position + row) * kv_v_dim;
            k_cache[k_dst..k_dst + kv_k_dim]
                .copy_from_slice(&k_rows[row * kv_k_dim..(row + 1) * kv_k_dim]);
            v_cache[v_dst..v_dst + kv_v_dim]
                .copy_from_slice(&v_rows[row * kv_v_dim..(row + 1) * kv_v_dim]);
        }
        let cpu = cpu_attention_rows(
            &q_fused,
            &k_cache,
            &v_cache,
            rows,
            position,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            max_context,
            q_stride,
        );

        let q_buf = metal.upload_pod(&q_fused).expect("q buffer");
        let k_rows_buf = metal.upload_pod(&k_rows).expect("k rows buffer");
        let v_rows_buf = metal.upload_pod(&v_rows).expect("v rows buffer");
        let k_cache_buf = metal.upload_pod(&initial_k_cache).expect("k cache buffer");
        let v_cache_buf = metal.upload_pod(&initial_v_cache).expect("v cache buffer");
        let scores_buf = metal.new_f32(rows * score_stride).expect("scores buffer");
        let out_buf = metal.new_f32(rows * q_dim).expect("out buffer");
        let cb = metal.command_buffer().expect("command buffer");
        let enc = cb.compute().expect("compute encoder");
        ok(ops::op_qwen_cache_write_rows_f32(
            &enc,
            metal.dev,
            &k_rows_buf,
            0,
            &k_cache_buf,
            checked_i32(rows, "k rows").expect("rows i32"),
            checked_i32(position, "k position").expect("position i32"),
            checked_i32(kv_heads, "k heads").expect("heads i32"),
            checked_i32(head_dim, "k head dim").expect("head dim i32"),
            checked_i32(max_context, "k context").expect("context i32"),
            checked_i32(kv_k_dim, "k stride").expect("stride i32"),
        ))
        .expect("k cache rows dispatch");
        ok(ops::op_qwen_cache_write_rows_f32(
            &enc,
            metal.dev,
            &v_rows_buf,
            0,
            &v_cache_buf,
            checked_i32(rows, "v rows").expect("rows i32"),
            checked_i32(position, "v position").expect("position i32"),
            checked_i32(kv_heads, "v heads").expect("heads i32"),
            checked_i32(value_dim, "v dim").expect("dim i32"),
            checked_i32(max_context, "v context").expect("context i32"),
            checked_i32(kv_v_dim, "v stride").expect("stride i32"),
        ))
        .expect("v cache rows dispatch");
        enc.memory_barrier_buffers();
        ok(ops::op_qwen_attention_scores_rows_f32(
            &enc,
            metal.dev,
            &q_buf,
            0,
            &k_cache_buf,
            &scores_buf,
            checked_i32(rows, "attn rows").expect("rows i32"),
            checked_i32(position, "attn position").expect("position i32"),
            checked_i32(q_heads, "attn heads").expect("heads i32"),
            checked_i32(kv_heads, "attn kv heads").expect("kv heads i32"),
            checked_i32(head_dim, "attn head dim").expect("head dim i32"),
            checked_i32(max_context, "attn context").expect("context i32"),
            checked_i32(q_stride, "attn q stride").expect("q stride i32"),
            checked_i32(score_stride, "attn score stride").expect("score stride i32"),
            1.0 / (head_dim as f32).sqrt(),
        ))
        .expect("attention scores rows dispatch");
        enc.memory_barrier_buffers();
        ok(ops::op_qwen_softmax_rows_f32(
            &enc,
            metal.dev,
            &scores_buf,
            checked_i32(rows, "softmax rows").expect("rows i32"),
            checked_i32(position, "softmax position").expect("position i32"),
            checked_i32(q_heads, "softmax heads").expect("heads i32"),
            checked_i32(max_context, "softmax context").expect("context i32"),
            checked_i32(score_stride, "softmax stride").expect("stride i32"),
        ))
        .expect("softmax rows dispatch");
        enc.memory_barrier_buffers();
        ok(ops::op_qwen_attention_values_rows_f32(
            &enc,
            metal.dev,
            &scores_buf,
            &v_cache_buf,
            &out_buf,
            checked_i32(rows, "value rows").expect("rows i32"),
            checked_i32(position, "value position").expect("position i32"),
            checked_i32(q_heads, "value heads").expect("heads i32"),
            checked_i32(kv_heads, "value kv heads").expect("kv heads i32"),
            checked_i32(value_dim, "value dim").expect("value dim i32"),
            checked_i32(max_context, "value context").expect("context i32"),
            checked_i32(score_stride, "value score stride").expect("score stride i32"),
        ))
        .expect("attention values rows dispatch");
        enc.memory_barrier_buffers();
        ok(ops::op_qwen_apply_sigmoid_gate_rows_f32(
            &enc,
            metal.dev,
            &out_buf,
            0,
            &q_buf,
            q_dim * std::mem::size_of::<f32>(),
            checked_i32(rows, "gate rows").expect("rows i32"),
            checked_i32(q_dim, "gate width").expect("width i32"),
            checked_i32(q_dim, "gate value stride").expect("value stride i32"),
            checked_i32(q_stride, "gate stride").expect("gate stride i32"),
        ))
        .expect("attention gate rows dispatch");
        enc.end();
        cb.commit_and_wait().expect("attention rows wait");

        let gpu = metal.read_f32(&out_buf, rows * q_dim).expect("out read");
        assert_close_vec("Metal attention rows", &gpu, &cpu, 2.0e-5);
    }

    #[test]
    #[ignore = "diagnostic for Metal batched prefill drift"]
    fn qwen35_metal_delta_block_rows_match_decode_steps_when_env_set() {
        let Some(path) = std::env::var_os("QWEN35_NATIVE_GGUF") else {
            eprintln!("skipping qwen35 Metal delta block rows parity: QWEN35_NATIVE_GGUF unset");
            return;
        };
        let gguf = GgufModel::open(&path).expect("open Qwen3.5 GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 inventory");
        inventory
            .validate_core_tensors(&gguf)
            .expect("Qwen3.5 core tensors");
        let metal =
            MetalQwen35Model::from_gguf(&gguf, inventory.clone(), 248_044).expect("Metal Qwen");

        let layer = 0usize;
        let rows = 2usize;
        let max_context = 8usize;
        let inner = inventory.ssm_inner_size;
        let heads = inventory.ssm_group_count;
        let head_dim = inner / heads;
        let hidden = patterned_input(rows * inventory.hidden_size)
            .into_iter()
            .map(|v| v * 0.03)
            .collect::<Vec<_>>();
        let recurrent = patterned_input(heads * head_dim * head_dim)
            .into_iter()
            .map(|v| v * 0.002)
            .collect::<Vec<_>>();
        let conv = patterned_input(inner * 3 * CONV_KERNEL)
            .into_iter()
            .map(|v| v * -0.01)
            .collect::<Vec<_>>();

        let row_ws = metal
            .new_prefill_workspace(rows, max_context)
            .expect("row workspace");
        unsafe {
            row_ws.normed.write(0, &hidden);
        }
        let row_recurrent = metal.upload_pod(&recurrent).expect("row recurrent");
        let row_conv = metal.upload_pod(&conv).expect("row conv");
        let cb = metal.command_buffer().expect("row command buffer");
        let enc = cb.compute().expect("row compute encoder");
        metal
            .encode_delta_attention_block_rows(
                &enc,
                layer,
                &row_recurrent,
                &row_conv,
                rows,
                &row_ws,
            )
            .expect("delta block rows");
        enc.end();
        cb.commit_and_wait().expect("row delta block wait");
        let row_out = metal
            .read_f32(&row_ws.attn_out, rows * inventory.hidden_size)
            .expect("row attn_out read");
        let row_state = metal
            .read_f32(&row_recurrent, recurrent.len())
            .expect("row recurrent read");

        let seq_ws = metal
            .new_forward_workspace(max_context)
            .expect("seq workspace");
        let seq_recurrent = metal.upload_pod(&recurrent).expect("seq recurrent");
        let seq_conv = metal.upload_pod(&conv).expect("seq conv");
        let mut seq_out = vec![0.0f32; rows * inventory.hidden_size];
        for row in 0..rows {
            unsafe {
                seq_ws.normed.write(
                    0,
                    &hidden[row * inventory.hidden_size..(row + 1) * inventory.hidden_size],
                );
            }
            let cb = metal.command_buffer().expect("seq command buffer");
            let enc = cb.compute().expect("seq compute encoder");
            metal
                .encode_delta_attention_block(&enc, layer, &seq_recurrent, &seq_conv, &seq_ws)
                .expect("delta block decode");
            enc.end();
            cb.commit_and_wait().expect("seq delta block wait");
            let out = metal
                .read_f32(&seq_ws.attn_out, inventory.hidden_size)
                .expect("seq attn_out read");
            seq_out[row * inventory.hidden_size..(row + 1) * inventory.hidden_size]
                .copy_from_slice(&out);
        }
        let seq_state = metal
            .read_f32(&seq_recurrent, recurrent.len())
            .expect("seq recurrent read");
        print_vector_delta("delta block rows out", &row_out, &seq_out);
        print_vector_delta("delta block rows state", &row_state, &seq_state);
        assert_close_vec("Metal delta block rows out", &row_out, &seq_out, 5.0e-2);
        assert_close_vec(
            "Metal delta block rows state",
            &row_state,
            &seq_state,
            5.0e-2,
        );
    }

    #[test]
    #[ignore = "diagnostic for Metal batched prefill drift"]
    fn qwen35_metal_full_attention_block_rows_match_decode_steps_when_env_set() {
        let Some(path) = std::env::var_os("QWEN35_NATIVE_GGUF") else {
            eprintln!(
                "skipping qwen35 Metal full attention block rows parity: QWEN35_NATIVE_GGUF unset"
            );
            return;
        };
        let gguf = GgufModel::open(&path).expect("open Qwen3.5 GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 inventory");
        inventory
            .validate_core_tensors(&gguf)
            .expect("Qwen3.5 core tensors");
        let metal =
            MetalQwen35Model::from_gguf(&gguf, inventory.clone(), 248_044).expect("Metal Qwen");

        let layer = inventory.full_attention_interval - 1;
        let rows = 2usize;
        let position = 3usize;
        let max_context = 8usize;
        let hidden = patterned_input(rows * inventory.hidden_size)
            .into_iter()
            .map(|v| v * 0.025)
            .collect::<Vec<_>>();
        let kv_k_dim = inventory.kv_heads * inventory.head_dim;
        let kv_v_dim = inventory.kv_heads * inventory.value_dim;
        let k_cache = patterned_input(max_context * kv_k_dim)
            .into_iter()
            .map(|v| v * 0.01)
            .collect::<Vec<_>>();
        let v_cache = patterned_input(max_context * kv_v_dim)
            .into_iter()
            .map(|v| v * -0.012)
            .collect::<Vec<_>>();

        let row_ws = metal
            .new_prefill_workspace(rows, max_context)
            .expect("row workspace");
        unsafe {
            row_ws.normed.write(0, &hidden);
        }
        let row_k_cache = metal.upload_pod(&k_cache).expect("row k cache");
        let row_v_cache = metal.upload_pod(&v_cache).expect("row v cache");
        let cb = metal.command_buffer().expect("row command buffer");
        let enc = cb.compute().expect("row compute encoder");
        metal
            .encode_full_attention_block_rows(
                &enc,
                layer,
                &row_k_cache,
                &row_v_cache,
                position,
                rows,
                max_context,
                &row_ws,
            )
            .expect("full attention block rows");
        enc.end();
        cb.commit_and_wait().expect("row full attention block wait");
        let row_out = metal
            .read_f32(&row_ws.attn_out, rows * inventory.hidden_size)
            .expect("row attn_out read");

        let seq_ws = metal
            .new_forward_workspace(max_context)
            .expect("seq workspace");
        let seq_k_cache = metal.upload_pod(&k_cache).expect("seq k cache");
        let seq_v_cache = metal.upload_pod(&v_cache).expect("seq v cache");
        let mut seq_out = vec![0.0f32; rows * inventory.hidden_size];
        for row in 0..rows {
            unsafe {
                seq_ws.normed.write(
                    0,
                    &hidden[row * inventory.hidden_size..(row + 1) * inventory.hidden_size],
                );
            }
            let cb = metal.command_buffer().expect("seq command buffer");
            let enc = cb.compute().expect("seq compute encoder");
            metal
                .encode_full_attention_block(
                    &enc,
                    layer,
                    &seq_k_cache,
                    &seq_v_cache,
                    position + row,
                    max_context,
                    &seq_ws,
                )
                .expect("full attention block decode");
            enc.end();
            cb.commit_and_wait().expect("seq full attention block wait");
            let out = metal
                .read_f32(&seq_ws.attn_out, inventory.hidden_size)
                .expect("seq attn_out read");
            seq_out[row * inventory.hidden_size..(row + 1) * inventory.hidden_size]
                .copy_from_slice(&out);
        }
        print_vector_delta("full attention block rows out", &row_out, &seq_out);
        assert_close_vec(
            "Metal full attention block rows out",
            &row_out,
            &seq_out,
            5.0e-2,
        );
    }

    #[test]
    fn qwen35_metal_rope_matches_cpu_for_real_model_when_env_set() {
        let Some(path) = std::env::var_os("QWEN35_NATIVE_GGUF") else {
            eprintln!("skipping qwen35 Metal RoPE parity: QWEN35_NATIVE_GGUF unset");
            return;
        };
        let gguf = GgufModel::open(&path).expect("open Qwen3.5 GGUF");
        let inventory = Qwen35Inventory::from_gguf(&gguf).expect("Qwen3.5 inventory");
        inventory
            .validate_core_tensors(&gguf)
            .expect("Qwen3.5 core tensors");
        let metal =
            MetalQwen35Model::from_gguf(&gguf, inventory.clone(), 248_044).expect("Metal Qwen");

        let values = patterned_input(inventory.attention_heads * inventory.head_dim)
            .into_iter()
            .map(|v| v * 0.05)
            .collect::<Vec<_>>();
        let gpu = metal
            .rope_decode(
                &values,
                inventory.attention_heads,
                inventory.head_dim,
                inventory.rope_dim,
                7,
            )
            .expect("Metal RoPE");
        let mut cpu = values;
        cpu_apply_rope(&mut cpu, 7, inventory.head_dim, inventory.rope_dim);
        assert_close_vec("Metal RoPE", &gpu, &cpu, 2.0e-5);
    }

    fn patterned_input(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (((i * 17 + 11) % 101) as f32 - 50.0) / 23.0)
            .collect()
    }

    fn max_abs_diff(lhs: &[f32], rhs: &[f32]) -> f32 {
        lhs.iter()
            .zip(rhs)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    }

    fn rms_norm(values: &[f32]) -> f32 {
        (values.iter().map(|v| v * v).sum::<f32>() / values.len().max(1) as f32).sqrt()
    }

    fn rms_diff(lhs: &[f32], rhs: &[f32]) -> f32 {
        (lhs.iter()
            .zip(rhs)
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f32>()
            / lhs.len().max(1) as f32)
            .sqrt()
    }

    fn cosine(lhs: &[f32], rhs: &[f32]) -> f32 {
        let dot = lhs.iter().zip(rhs).map(|(a, b)| a * b).sum::<f32>();
        let ln = rms_norm(lhs) * (lhs.len().max(1) as f32).sqrt();
        let rn = rms_norm(rhs) * (rhs.len().max(1) as f32).sqrt();
        dot / (ln * rn).max(1.0e-12)
    }

    fn greedy_argmax(values: &[f32]) -> u32 {
        values
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| u32::try_from(index).expect("vocabulary index fits u32"))
            .expect("non-empty logits")
    }

    fn assert_close_vec(label: &str, lhs: &[f32], rhs: &[f32], tol: f32) {
        assert_eq!(lhs.len(), rhs.len(), "{label}");
        let max_abs = max_abs_diff(lhs, rhs);
        assert!(
            max_abs <= tol,
            "{label} max_abs diff too high: {max_abs:.6e} > {tol:.6e}"
        );
    }

    fn print_vector_delta(label: &str, lhs: &[f32], rhs: &[f32]) {
        let max_abs = max_abs_diff(lhs, rhs);
        let cosine = cosine(lhs, rhs);
        let rel = rms_diff(lhs, rhs) / rms_norm(rhs).max(1.0e-6);
        eprintln!("{label}: cosine={cosine:.8} rms_rel={rel:.6e} max_abs={max_abs:.6e}");
    }

    fn cpu_rms_norm(input: &[f32], weight: &[f32], dim: usize, _qwen_scale: bool) -> Vec<f32> {
        input
            .chunks(dim)
            .flat_map(|row| {
                let sum_sq = row.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>();
                let rstd = (1.0 / ((sum_sq / dim as f64) + 1.0e-6f64).sqrt()) as f32;
                row.iter()
                    .enumerate()
                    .map(move |(i, v)| *v * rstd * weight[i])
            })
            .collect()
    }

    fn cpu_causal_conv1d_silu(
        values: &mut [f32],
        weights: &[f32],
        state: &mut [f32],
        kernel: usize,
    ) {
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
            let q_norm = (qh.iter().map(|v| v * v).sum::<f32>() + 1.0e-6).sqrt();
            let k_norm = (kh.iter().map(|v| v * v).sum::<f32>() + 1.0e-6).sqrt();
            for v in qh {
                *v = *v / q_norm * q_scale;
            }
            for v in kh {
                *v /= k_norm;
            }
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
        state: &mut [f32],
        heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; heads * head_dim];
        for head in 0..heads {
            let base = head * head_dim;
            let qh = &q[base..base + head_dim];
            let kh = &k[base..base + head_dim];
            let vh = &v[base..base + head_dim];
            let oh = &mut out[base..base + head_dim];
            let beta_h = sigmoid(beta[head]);
            let decay = (-a_log[head].exp() * softplus(alpha[head] + dt_bias[head]))
                .exp()
                .clamp(0.0, 1.0);
            for value_idx in 0..head_dim {
                let row = &mut state[(head * head_dim + value_idx) * head_dim..][..head_dim];
                let prior = row.iter().zip(kh).map(|(a, b)| a * b).sum::<f32>();
                let delta = (vh[value_idx] - decay * prior) * beta_h;
                let mut attn = 0.0f32;
                for key_idx in 0..head_dim {
                    row[key_idx] = decay * row[key_idx] + kh[key_idx] * delta;
                    attn += row[key_idx] * qh[key_idx];
                }
                oh[value_idx] = attn;
            }
        }
        out
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

    #[allow(clippy::too_many_arguments)]
    fn cpu_attention_rows(
        q_fused: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        rows: usize,
        position: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
        max_context: usize,
        q_stride: usize,
    ) -> Vec<f32> {
        let q_dim = q_heads * head_dim;
        let kv_k_dim = kv_heads * head_dim;
        let kv_v_dim = kv_heads * value_dim;
        let gqa = q_heads / kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0f32; rows * q_heads * value_dim];
        for row in 0..rows {
            let row_position = position + row;
            for head in 0..q_heads {
                let kv_head = head / gqa;
                let q_base = row * q_stride + head * head_dim;
                let qh = &q_fused[q_base..q_base + head_dim];
                let mut scores = Vec::with_capacity(row_position + 1);
                for pos in 0..=row_position {
                    let key_base = pos * kv_k_dim + kv_head * head_dim;
                    let kh = &k_cache[key_base..key_base + head_dim];
                    scores.push(dot(qh, kh) * scale);
                }
                softmax_in_place(&mut scores);
                let dst_base = row * q_dim + head * value_dim;
                for (pos, score) in scores.iter().copied().enumerate() {
                    let value_base = pos * kv_v_dim + kv_head * value_dim;
                    let vh = &v_cache[value_base..value_base + value_dim];
                    for i in 0..value_dim {
                        out[dst_base + i] += score * vh[i];
                    }
                }
                let gate_base = row * q_stride + q_dim + head * value_dim;
                for i in 0..value_dim {
                    out[dst_base + i] *= sigmoid(q_fused[gate_base + i]);
                }
            }
        }
        debug_assert!(max_context >= position + rows);
        out
    }

    fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
        lhs.iter().zip(rhs).map(|(a, b)| a * b).sum()
    }

    fn softmax_in_place(values: &mut [f32]) {
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for value in values.iter_mut() {
            *value = (*value - max).exp();
            sum += *value;
        }
        let inv = 1.0 / sum.max(1.0e-20);
        for value in values {
            *value *= inv;
        }
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
}
