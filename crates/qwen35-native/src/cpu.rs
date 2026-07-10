use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use greppy_embed_native::matmul::QuantMatrix;
use greppy_embed_native::performance::PerformanceCorePool;
use greppy_embed_native::GgufModel;
use rayon::prelude::*;
use tokenizers::Tokenizer;

use crate::inventory::Qwen35Inventory;
use crate::sampler::{sample_token, GenerationParams, SamplerRng};
use crate::simd_math::{
    exp_sum_shifted_in_place, mul_sigmoid_in_place, mul_silu_in_place, silu_in_place,
    swiglu_in_place,
};
use crate::{Error, Result};

const RMS_EPS: f32 = 1.0e-6;
const ROPE_THETA: f32 = 10_000_000.0;
const LINEAR_HEADS: usize = 16;
const LINEAR_HEAD_DIM: usize = 128;
const CONV_KERNEL: usize = 4;
const CONV_CHANNEL_BLOCK: usize = 1024;
const MTP_DRAFT_MAX: usize = 2;

pub(crate) struct CpuQwen35Model {
    inventory: Qwen35Inventory,
    token_embd: QuantMatrix,
    output_norm: Vec<f32>,
    layers: Vec<LayerWeights>,
    mtp: Option<MtpWeights>,
    eos_token_id: u32,
    performance_pool: PerformanceCorePool,
}

struct LayerWeights {
    attn_norm: Vec<f32>,
    post_attention_norm: Vec<f32>,
    ffn_gate: QuantMatrix,
    ffn_up: QuantMatrix,
    ffn_down: QuantMatrix,
    kind: LayerKind,
}

enum LayerKind {
    Delta(DeltaWeights),
    Full(FullAttentionWeights),
}

struct DeltaWeights {
    attn_qkv: QuantMatrix,
    attn_gate: QuantMatrix,
    ssm_beta: QuantMatrix,
    ssm_alpha: QuantMatrix,
    ssm_conv1d: Vec<f32>,
    ssm_a: Vec<f32>,
    ssm_dt_bias: Vec<f32>,
    ssm_norm: Vec<f32>,
    ssm_out: QuantMatrix,
}

struct FullAttentionWeights {
    attn_q: QuantMatrix,
    attn_k: QuantMatrix,
    attn_v: QuantMatrix,
    attn_output: QuantMatrix,
    attn_q_norm: Vec<f32>,
    attn_k_norm: Vec<f32>,
}

struct MtpWeights {
    layer: LayerWeights,
    eh_proj: QuantMatrix,
    enorm: Vec<f32>,
    hnorm: Vec<f32>,
    shared_head_norm: Vec<f32>,
}

#[derive(Clone)]
pub(crate) struct ForwardState {
    position: usize,
    layer_states: Vec<LayerState>,
    decode_workspace: CpuDecodeWorkspace,
    max_context: usize,
}

#[derive(Clone, Default)]
struct CpuDecodeWorkspace {
    ffn_gate: Vec<f32>,
    ffn_up: Vec<f32>,
    ffn_out: Vec<f32>,
}

#[derive(Clone)]
enum LayerState {
    Delta(DeltaState),
    Full(FullAttentionState),
}

#[derive(Clone)]
struct DeltaState {
    recurrent: Vec<f32>,
    conv: Vec<f32>,
}

#[derive(Clone)]
struct FullAttentionState {
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
}

pub(crate) struct TargetForwardOutput {
    pub(crate) hidden: Vec<f32>,
    pub(crate) logits: Vec<f32>,
}

#[derive(Clone)]
pub(crate) struct MtpState {
    position: usize,
    attention: FullAttentionState,
    max_context: usize,
}

pub(crate) struct MtpForwardOutput {
    pub(crate) hidden: Vec<f32>,
    pub(crate) logits: Vec<f32>,
}

impl CpuQwen35Model {
    pub(crate) fn load(
        model: &GgufModel,
        inventory: Qwen35Inventory,
        eos_token_id: u32,
    ) -> Result<Self> {
        let token_embd = qwen_matrix(model, "token_embd.weight")?;
        let output_norm = tensor_f32(model, "output_norm.weight")?;
        let mut layers = Vec::with_capacity(inventory.block_count);
        for layer in 0..inventory.block_count {
            let prefix = format!("blk.{layer}");
            let kind = if inventory.is_full_attention_layer(layer) {
                LayerKind::Full(FullAttentionWeights {
                    attn_q: qwen_matrix(model, &format!("{prefix}.attn_q.weight"))?,
                    attn_k: qwen_matrix(model, &format!("{prefix}.attn_k.weight"))?,
                    attn_v: qwen_matrix(model, &format!("{prefix}.attn_v.weight"))?,
                    attn_output: qwen_matrix(model, &format!("{prefix}.attn_output.weight"))?,
                    attn_q_norm: tensor_f32(model, &format!("{prefix}.attn_q_norm.weight"))?,
                    attn_k_norm: tensor_f32(model, &format!("{prefix}.attn_k_norm.weight"))?,
                })
            } else {
                LayerKind::Delta(DeltaWeights {
                    attn_qkv: qwen_matrix(model, &format!("{prefix}.attn_qkv.weight"))?,
                    attn_gate: qwen_matrix(model, &format!("{prefix}.attn_gate.weight"))?,
                    ssm_beta: qwen_matrix(model, &format!("{prefix}.ssm_beta.weight"))?,
                    ssm_alpha: qwen_matrix(model, &format!("{prefix}.ssm_alpha.weight"))?,
                    ssm_conv1d: tensor_f32(model, &format!("{prefix}.ssm_conv1d.weight"))?,
                    ssm_a: tensor_f32(model, &format!("{prefix}.ssm_a"))?,
                    ssm_dt_bias: tensor_f32(model, &format!("{prefix}.ssm_dt.bias"))?,
                    ssm_norm: tensor_f32(model, &format!("{prefix}.ssm_norm.weight"))?,
                    ssm_out: qwen_matrix(model, &format!("{prefix}.ssm_out.weight"))?,
                })
            };
            layers.push(LayerWeights {
                attn_norm: tensor_f32(model, &format!("{prefix}.attn_norm.weight"))?,
                post_attention_norm: tensor_f32(
                    model,
                    &format!("{prefix}.post_attention_norm.weight"),
                )?,
                ffn_gate: qwen_matrix(model, &format!("{prefix}.ffn_gate.weight"))?,
                ffn_up: qwen_matrix(model, &format!("{prefix}.ffn_up.weight"))?,
                ffn_down: qwen_matrix(model, &format!("{prefix}.ffn_down.weight"))?,
                kind,
            });
        }
        let mtp = if inventory.has_mtp() {
            let prefix = format!("blk.{}", inventory.block_count);
            Some(MtpWeights {
                layer: LayerWeights {
                    attn_norm: tensor_f32(model, &format!("{prefix}.attn_norm.weight"))?,
                    post_attention_norm: tensor_f32(
                        model,
                        &format!("{prefix}.post_attention_norm.weight"),
                    )?,
                    ffn_gate: qwen_matrix(model, &format!("{prefix}.ffn_gate.weight"))?,
                    ffn_up: qwen_matrix(model, &format!("{prefix}.ffn_up.weight"))?,
                    ffn_down: qwen_matrix(model, &format!("{prefix}.ffn_down.weight"))?,
                    kind: LayerKind::Full(FullAttentionWeights {
                        attn_q: qwen_matrix(model, &format!("{prefix}.attn_q.weight"))?,
                        attn_k: qwen_matrix(model, &format!("{prefix}.attn_k.weight"))?,
                        attn_v: qwen_matrix(model, &format!("{prefix}.attn_v.weight"))?,
                        attn_output: qwen_matrix(model, &format!("{prefix}.attn_output.weight"))?,
                        attn_q_norm: tensor_f32(model, &format!("{prefix}.attn_q_norm.weight"))?,
                        attn_k_norm: tensor_f32(model, &format!("{prefix}.attn_k_norm.weight"))?,
                    }),
                },
                eh_proj: qwen_matrix(model, &format!("{prefix}.nextn.eh_proj.weight"))?,
                enorm: tensor_f32(model, &format!("{prefix}.nextn.enorm.weight"))?,
                hnorm: tensor_f32(model, &format!("{prefix}.nextn.hnorm.weight"))?,
                shared_head_norm: tensor_f32(
                    model,
                    &format!("{prefix}.nextn.shared_head_norm.weight"),
                )?,
            })
        } else {
            None
        };
        Ok(Self {
            inventory,
            token_embd,
            output_norm,
            layers,
            mtp,
            eos_token_id,
            performance_pool: PerformanceCorePool::new("qwen35")
                .map_err(|error| Error::GenerationUnavailable(error.to_string()))?,
        })
    }

    pub(crate) fn generate(
        &self,
        tokenizer: &Tokenizer,
        prompt: &str,
        params: GenerationParams,
    ) -> Result<String> {
        self.performance_pool
            .install(|| self.generate_on_performance_cores(tokenizer, prompt, params))
    }

    #[cfg(test)]
    pub(crate) fn generate_target_only_for_test(
        &self,
        tokenizer: &Tokenizer,
        prompt: &str,
        params: GenerationParams,
    ) -> Result<String> {
        self.performance_pool.install(|| {
            let encoding = tokenizer
                .encode(prompt, true)
                .map_err(|error| Error::Tokenizer(error.to_string()))?;
            let prompt_ids = encoding.get_ids();
            if prompt_ids.is_empty() {
                return Ok(String::new());
            }
            self.generate_without_mtp(tokenizer, prompt, prompt_ids, params)
        })
    }

    fn generate_on_performance_cores(
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
        if self.mtp.is_some() {
            return self.generate_with_mtp(tokenizer, prompt, prompt_ids, params);
        }
        self.generate_without_mtp(tokenizer, prompt, prompt_ids, params)
    }

    fn generate_without_mtp(
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
        let mut state = self.new_state(max_context);
        for tokens in prompt_ids[..prompt_ids.len().saturating_sub(1)].chunks(512) {
            self.prefill_tokens(tokens, &mut state)?;
        }

        let mut next = *prompt_ids.last().expect("checked non-empty above");
        let mut generated = Vec::new();
        let mut rng = SamplerRng::new(prompt_seed(prompt));
        for _ in 0..params.max_tokens {
            let mut logits = self.forward_token_logits(next, &mut state)?;
            let Some(token) = sample_token(&mut logits, &generated, params, &mut rng) else {
                break;
            };
            if token == self.eos_token_id {
                break;
            }
            generated.push(token);
            next = token;
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
        let mut target_state = self.new_state(max_context);
        let mut prompt_hidden = Vec::with_capacity(prompt_ids.len() * hidden_size);
        perf.begin_target_prefill();
        for tokens in prompt_ids[..prompt_ids.len() - 1].chunks(512) {
            let hidden = self.prefill_tokens_hidden(tokens, &mut target_state)?;
            prompt_hidden.extend(hidden);
        }
        let mut target = self.forward_token_logits_hidden(
            *prompt_ids.last().expect("checked non-empty above"),
            &mut target_state,
        )?;
        prompt_hidden.extend_from_slice(&target.hidden);
        perf.finish_target_prefill();

        let zero_hidden = vec![0.0f32; hidden_size];
        let prompt_conditioning =
            mtp_conditioning_rows(&zero_hidden, &prompt_hidden, prompt_ids.len(), hidden_size)?;
        let mut mtp_state = self.new_mtp_state(max_context)?;
        perf.begin_mtp_prefill();
        for start in (0..prompt_ids.len()).step_by(512) {
            let end = (start + 512).min(prompt_ids.len());
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
            eprintln!("qwen35-mtp-debug first={first}");
        }
        let mut next = first;
        let mut pending_target_hidden = target.hidden;
        let mut drafted_total = 0usize;
        let mut accepted_total = 0usize;
        let mut cycles = 0usize;
        let mut mtp_fallback = false;
        let mut speculative_target_state = target_state.clone();
        let mut draft_mtp_state = mtp_state.clone();

        while generated.len() < params.max_tokens {
            let remaining = params.max_tokens - generated.len();
            if mtp_fallback {
                let stage = perf.begin_stage();
                let mut logits = self.forward_token_logits(next, &mut target_state)?;
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
                let mut output = self.forward_token_logits_hidden(next, &mut target_state)?;
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
            draft_mtp_state.clone_from(&mtp_state);
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
                let mut output = self.forward_token_logits_hidden(next, &mut target_state)?;
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
            debug_assert!(draft_tokens.len() <= MTP_DRAFT_MAX);
            drafted_total += draft_tokens.len();
            if mtp_debug {
                eprintln!("qwen35-mtp-debug cycle={cycles} input={next} drafts={draft_tokens:?}");
            }

            let mut verification_tokens = Vec::with_capacity(1 + draft_tokens.len());
            verification_tokens.push(next);
            verification_tokens.extend_from_slice(&draft_tokens);
            let stage = perf.begin_stage();
            speculative_target_state.clone_from(&target_state);
            perf.finish_stage(crate::MtpPerfStage::TargetStateCopy, stage);
            let stage = perf.begin_stage();
            let mut committed_hidden = Vec::with_capacity(verification_tokens.len() * hidden_size);
            let mut accepted = 0usize;
            let mut mismatch = None;
            let mut finished = false;
            let mut first =
                self.forward_token_logits_hidden(next, &mut speculative_target_state)?;
            committed_hidden.extend_from_slice(&first.hidden);
            let Some(first_target) = sample_token(&mut first.logits, &generated, params, &mut rng)
            else {
                break;
            };
            if first_target == self.eos_token_id {
                break;
            }
            if mtp_debug {
                eprintln!(
                    "qwen35-mtp-debug cycle={cycles} pos=0 target={first_target} draft={}",
                    draft_tokens[0]
                );
            }
            generated.push(first_target);
            if first_target != draft_tokens[0] {
                mismatch = Some(first_target);
            } else {
                accepted = 1;
                accepted_total += 1;
                if generated.len() == params.max_tokens {
                    finished = true;
                } else if draft_tokens.len() == 1 {
                    let mut bonus = self.forward_token_logits_hidden(
                        draft_tokens[0],
                        &mut speculative_target_state,
                    )?;
                    committed_hidden.extend_from_slice(&bonus.hidden);
                    let Some(target_token) =
                        sample_token(&mut bonus.logits, &generated, params, &mut rng)
                    else {
                        perf.finish_stage(crate::MtpPerfStage::TargetVerify, stage);
                        break;
                    };
                    if target_token == self.eos_token_id {
                        finished = true;
                    } else {
                        generated.push(target_token);
                        next = target_token;
                    }
                } else {
                    let mut tail = self.forward_tokens_logits_hidden(
                        &draft_tokens,
                        &mut speculative_target_state,
                    )?;
                    let Some(second_target) =
                        sample_token(&mut tail.logits[..vocab_size], &generated, params, &mut rng)
                    else {
                        perf.finish_stage(crate::MtpPerfStage::TargetVerify, stage);
                        break;
                    };
                    if second_target == self.eos_token_id {
                        finished = true;
                    } else {
                        if mtp_debug {
                            eprintln!(
                                "qwen35-mtp-debug cycle={cycles} pos=1 target={second_target} draft={}",
                                draft_tokens[1]
                            );
                        }
                        generated.push(second_target);
                        if second_target != draft_tokens[1] {
                            mismatch = Some(second_target);
                        } else {
                            accepted = 2;
                            accepted_total += 1;
                            if generated.len() == params.max_tokens {
                                finished = true;
                            } else {
                                committed_hidden.extend_from_slice(&tail.hidden);
                                let Some(target_token) = sample_token(
                                    &mut tail.logits[vocab_size..vocab_size * 2],
                                    &generated,
                                    params,
                                    &mut rng,
                                ) else {
                                    perf.finish_stage(crate::MtpPerfStage::TargetVerify, stage);
                                    break;
                                };
                                if target_token == self.eos_token_id {
                                    finished = true;
                                } else {
                                    generated.push(target_token);
                                    next = target_token;
                                }
                            }
                        }
                    }
                }
            }
            perf.finish_stage(crate::MtpPerfStage::TargetVerify, stage);
            if finished {
                break;
            }
            mtp_fallback = crate::mtp_should_fallback(drafted_total, accepted_total);

            if accepted == draft_tokens.len() {
                std::mem::swap(&mut target_state, &mut speculative_target_state);
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
                if accepted == 0 {
                    std::mem::swap(&mut target_state, &mut speculative_target_state);
                } else {
                    let stage = perf.begin_stage();
                    committed_hidden =
                        self.prefill_tokens_hidden(commit_tokens, &mut target_state)?;
                    perf.finish_stage(crate::MtpPerfStage::TargetReplay, stage);
                }
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
                "qwen35-mtp-debug cycles={cycles} drafted={drafted_total} accepted={accepted_total}"
            );
        }
        perf.report(
            "cpu",
            prompt_ids.len(),
            generated.len(),
            cycles,
            drafted_total,
            accepted_total,
            mtp_fallback,
        );
        tokenizer
            .decode(&generated, true)
            .map_err(|e| Error::Tokenizer(e.to_string()))
    }

    pub(crate) fn new_state(&self, max_context: usize) -> ForwardState {
        let mut layer_states = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            match &layer.kind {
                LayerKind::Delta(_) => layer_states.push(LayerState::Delta(DeltaState {
                    recurrent: vec![0.0; LINEAR_HEADS * LINEAR_HEAD_DIM * LINEAR_HEAD_DIM],
                    conv: vec![0.0; self.inventory.ssm_inner_size * 3 * CONV_KERNEL],
                })),
                LayerKind::Full(_) => layer_states.push(LayerState::Full(FullAttentionState {
                    k_cache: vec![
                        0.0;
                        max_context * self.inventory.kv_heads * self.inventory.head_dim
                    ],
                    v_cache: vec![
                        0.0;
                        max_context * self.inventory.kv_heads * self.inventory.value_dim
                    ],
                })),
            }
        }
        ForwardState {
            position: 0,
            layer_states,
            decode_workspace: CpuDecodeWorkspace::default(),
            max_context,
        }
    }

    pub(crate) fn new_mtp_state(&self, max_context: usize) -> Result<MtpState> {
        if self.mtp.is_none() {
            return Err(Error::GenerationUnavailable(
                "Qwen3.5 model does not contain an MTP layer".into(),
            ));
        }
        Ok(MtpState {
            position: 0,
            attention: FullAttentionState {
                k_cache: vec![0.0; max_context * self.inventory.kv_heads * self.inventory.head_dim],
                v_cache: vec![
                    0.0;
                    max_context * self.inventory.kv_heads * self.inventory.value_dim
                ],
            },
            max_context,
        })
    }

    #[cfg(test)]
    pub(crate) fn on_performance_cores<R: Send>(&self, operation: impl FnOnce() -> R + Send) -> R {
        self.performance_pool.install(operation)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn prefill_token(&self, token: u32, state: &mut ForwardState) -> Result<()> {
        let _ = self.forward_token_hidden(token, state)?;
        Ok(())
    }

    pub(crate) fn prefill_tokens(&self, tokens: &[u32], state: &mut ForwardState) -> Result<()> {
        let _ = self.prefill_tokens_hidden(tokens, state)?;
        Ok(())
    }

    pub(crate) fn prefill_tokens_hidden(
        &self,
        tokens: &[u32],
        state: &mut ForwardState,
    ) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        if state.position.saturating_add(tokens.len()) > state.max_context {
            return Err(Error::InvalidRequest(format!(
                "qwen35 prompt exceeds local context cap {}",
                state.max_context
            )));
        }
        let rows = tokens.len();
        let start_position = state.position;
        let mut hidden = self.token_embd.embedding_rows(tokens)?;
        for (idx, layer) in self.layers.iter().enumerate() {
            let residual = hidden.clone();
            rms_norm_rows_qwen(&mut hidden, &layer.attn_norm, self.inventory.hidden_size);
            let attn = match (&layer.kind, &mut state.layer_states[idx]) {
                (LayerKind::Delta(weights), LayerState::Delta(runtime)) => {
                    self.delta_block_rows(weights, runtime, &hidden, rows)?
                }
                (LayerKind::Full(weights), LayerState::Full(runtime)) => {
                    self.full_attention_block_rows(weights, runtime, start_position, &hidden, rows)?
                }
                _ => return Err(Error::Gguf("qwen35 layer/runtime state mismatch".into())),
            };
            add_rows(&mut hidden, &residual, &attn);

            let residual = hidden.clone();
            rms_norm_rows_qwen(
                &mut hidden,
                &layer.post_attention_norm,
                self.inventory.hidden_size,
            );
            let ffn = self.ffn_rows(layer, &hidden, rows)?;
            add_rows(&mut hidden, &residual, &ffn);
        }
        state.position += rows;
        Ok(hidden)
    }

    #[cfg(test)]
    pub(crate) fn profile_prefill_tokens(
        &self,
        tokens: &[u32],
        state: &mut ForwardState,
    ) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        if state.position.saturating_add(tokens.len()) > state.max_context {
            return Err(Error::InvalidRequest(format!(
                "qwen35 prompt exceeds local context cap {}",
                state.max_context
            )));
        }
        let rows = tokens.len();
        let start_position = state.position;
        let total_start = std::time::Instant::now();
        let stage_start = std::time::Instant::now();
        let mut hidden = self.token_embd.embedding_rows(tokens)?;
        eprintln!(
            "cpu_prefill_profile stage=embed rows={rows} ms={:.3}",
            stage_start.elapsed().as_secs_f64() * 1.0e3
        );

        for (idx, layer) in self.layers.iter().enumerate() {
            let kind = match &layer.kind {
                LayerKind::Delta(_) => "delta",
                LayerKind::Full(_) => "full",
            };
            let layer_start = std::time::Instant::now();
            let stage_start = std::time::Instant::now();
            let residual = hidden.clone();
            rms_norm_rows_qwen(&mut hidden, &layer.attn_norm, self.inventory.hidden_size);
            let norm_ms = stage_start.elapsed().as_secs_f64() * 1.0e3;

            let stage_start = std::time::Instant::now();
            let attn = match (&layer.kind, &mut state.layer_states[idx]) {
                (LayerKind::Delta(weights), LayerState::Delta(runtime)) => {
                    self.delta_block_rows(weights, runtime, &hidden, rows)?
                }
                (LayerKind::Full(weights), LayerState::Full(runtime)) => {
                    self.full_attention_block_rows(weights, runtime, start_position, &hidden, rows)?
                }
                _ => return Err(Error::Gguf("qwen35 layer/runtime state mismatch".into())),
            };
            let attention_ms = stage_start.elapsed().as_secs_f64() * 1.0e3;

            let stage_start = std::time::Instant::now();
            add_rows(&mut hidden, &residual, &attn);
            let residual_ms = stage_start.elapsed().as_secs_f64() * 1.0e3;

            let stage_start = std::time::Instant::now();
            let residual = hidden.clone();
            rms_norm_rows_qwen(
                &mut hidden,
                &layer.post_attention_norm,
                self.inventory.hidden_size,
            );
            let post_norm_ms = stage_start.elapsed().as_secs_f64() * 1.0e3;

            let stage_start = std::time::Instant::now();
            let ffn = self.ffn_rows(layer, &hidden, rows)?;
            let ffn_ms = stage_start.elapsed().as_secs_f64() * 1.0e3;

            let stage_start = std::time::Instant::now();
            add_rows(&mut hidden, &residual, &ffn);
            let ffn_add_ms = stage_start.elapsed().as_secs_f64() * 1.0e3;
            eprintln!(
                "cpu_prefill_profile stage=layer layer={idx} kind={kind} total_ms={:.3} norm_ms={norm_ms:.3} attention_ms={attention_ms:.3} residual_ms={residual_ms:.3} post_norm_ms={post_norm_ms:.3} ffn_ms={ffn_ms:.3} ffn_add_ms={ffn_add_ms:.3}",
                layer_start.elapsed().as_secs_f64() * 1.0e3,
            );
        }
        state.position += rows;
        let total_secs = total_start.elapsed().as_secs_f64();
        eprintln!(
            "cpu_prefill_profile stage=total rows={rows} ms={:.3} tok_s={:.2}",
            total_secs * 1.0e3,
            rows as f64 / total_secs.max(1.0e-9),
        );
        Ok(())
    }

    pub(crate) fn forward_token_logits(
        &self,
        token: u32,
        state: &mut ForwardState,
    ) -> Result<Vec<f32>> {
        let mut hidden = self.forward_token_hidden(token, state)?;
        rms_norm_qwen(&mut hidden, &self.output_norm);
        self.token_embd.matmul(&hidden, 1).map_err(Into::into)
    }

    pub(crate) fn forward_token_logits_hidden(
        &self,
        token: u32,
        state: &mut ForwardState,
    ) -> Result<TargetForwardOutput> {
        let hidden = self.forward_token_hidden(token, state)?;
        let mut output_hidden = hidden.clone();
        rms_norm_qwen(&mut output_hidden, &self.output_norm);
        let logits = self.token_embd.matmul(&output_hidden, 1)?;
        Ok(TargetForwardOutput { hidden, logits })
    }

    pub(crate) fn forward_tokens_logits_hidden(
        &self,
        tokens: &[u32],
        state: &mut ForwardState,
    ) -> Result<TargetForwardOutput> {
        let hidden = self.prefill_tokens_hidden(tokens, state)?;
        if hidden.is_empty() {
            return Ok(TargetForwardOutput {
                hidden,
                logits: Vec::new(),
            });
        }
        let mut output_hidden = hidden.clone();
        rms_norm_rows_qwen(
            &mut output_hidden,
            &self.output_norm,
            self.inventory.hidden_size,
        );
        let logits = self.token_embd.matmul(&output_hidden, tokens.len())?;
        Ok(TargetForwardOutput { hidden, logits })
    }

    pub(crate) fn mtp_prefill_tokens(
        &self,
        tokens: &[u32],
        target_hidden: &[f32],
        state: &mut MtpState,
    ) -> Result<()> {
        let _ = self.mtp_forward_tokens_hidden(tokens, target_hidden, state)?;
        Ok(())
    }

    pub(crate) fn mtp_forward_tokens_logits_hidden(
        &self,
        tokens: &[u32],
        target_hidden: &[f32],
        state: &mut MtpState,
    ) -> Result<MtpForwardOutput> {
        let mut hidden = self.mtp_forward_tokens_hidden(tokens, target_hidden, state)?;
        if hidden.is_empty() {
            return Ok(MtpForwardOutput {
                hidden,
                logits: Vec::new(),
            });
        }
        let mtp = self.mtp_weights()?;
        rms_norm_rows_qwen(
            &mut hidden,
            &mtp.shared_head_norm,
            self.inventory.hidden_size,
        );
        let logits = self.token_embd.matmul(&hidden, tokens.len())?;
        Ok(MtpForwardOutput { hidden, logits })
    }

    fn mtp_forward_tokens_hidden(
        &self,
        tokens: &[u32],
        target_hidden: &[f32],
        state: &mut MtpState,
    ) -> Result<Vec<f32>> {
        let mtp = self.mtp_weights()?;
        if tokens.is_empty() {
            if target_hidden.is_empty() {
                return Ok(Vec::new());
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

        let mut embeddings = self.token_embd.embedding_rows(tokens)?;
        let mut target_hidden = target_hidden.to_vec();
        rms_norm_rows_qwen(&mut embeddings, &mtp.enorm, hidden_size);
        rms_norm_rows_qwen(&mut target_hidden, &mtp.hnorm, hidden_size);
        let mut joined = vec![0.0f32; rows * hidden_size * 2];
        joined
            .par_chunks_mut(hidden_size * 2)
            .zip(embeddings.par_chunks(hidden_size))
            .zip(target_hidden.par_chunks(hidden_size))
            .for_each(|((joined, embedding), hidden)| {
                joined[..hidden_size].copy_from_slice(embedding);
                joined[hidden_size..].copy_from_slice(hidden);
            });

        let mut hidden = mtp.eh_proj.matmul(&joined, rows)?;
        let residual = hidden.clone();
        rms_norm_rows_qwen(&mut hidden, &mtp.layer.attn_norm, hidden_size);
        let LayerKind::Full(attention_weights) = &mtp.layer.kind else {
            return Err(Error::Gguf("qwen35 MTP layer is not full attention".into()));
        };
        let attention = self.full_attention_block_rows(
            attention_weights,
            &mut state.attention,
            state.position,
            &hidden,
            rows,
        )?;
        add_rows(&mut hidden, &residual, &attention);

        let residual = hidden.clone();
        rms_norm_rows_qwen(&mut hidden, &mtp.layer.post_attention_norm, hidden_size);
        let ffn = self.ffn_rows(&mtp.layer, &hidden, rows)?;
        add_rows(&mut hidden, &residual, &ffn);
        state.position += rows;
        Ok(hidden)
    }

    fn mtp_weights(&self) -> Result<&MtpWeights> {
        self.mtp.as_ref().ok_or_else(|| {
            Error::GenerationUnavailable("Qwen3.5 model does not contain an MTP layer".into())
        })
    }

    #[cfg(test)]
    pub(crate) fn profile_forward_token_logits(
        &self,
        token: u32,
        state: &mut ForwardState,
    ) -> Result<Vec<f32>> {
        if state.position >= state.max_context {
            return Err(Error::InvalidRequest(format!(
                "qwen35 prompt exceeds local context cap {}",
                state.max_context
            )));
        }
        let total_start = std::time::Instant::now();
        let stage_start = std::time::Instant::now();
        let mut hidden = self.token_embd.embedding_rows(&[token])?;
        eprintln!(
            "cpu_decode_profile stage=embed position={} ms={:.3}",
            state.position,
            stage_start.elapsed().as_secs_f64() * 1.0e3,
        );
        for (idx, layer) in self.layers.iter().enumerate() {
            let kind = match &layer.kind {
                LayerKind::Delta(_) => "delta",
                LayerKind::Full(_) => "full",
            };
            let layer_start = std::time::Instant::now();
            let stage_start = std::time::Instant::now();
            let mut residual = hidden;
            let mut x = residual.clone();
            rms_norm_qwen(&mut x, &layer.attn_norm);
            let norm_ms = stage_start.elapsed().as_secs_f64() * 1.0e3;

            let stage_start = std::time::Instant::now();
            let attn = match (&layer.kind, &mut state.layer_states[idx]) {
                (LayerKind::Delta(weights), LayerState::Delta(runtime)) => {
                    self.delta_block_rows(weights, runtime, &x, 1)?
                }
                (LayerKind::Full(weights), LayerState::Full(runtime)) => {
                    self.full_attention_block(weights, runtime, state.position, &x)?
                }
                _ => return Err(Error::Gguf("qwen35 layer/runtime state mismatch".into())),
            };
            let attention_ms = stage_start.elapsed().as_secs_f64() * 1.0e3;

            let stage_start = std::time::Instant::now();
            add_in_place(&mut residual, &attn);
            hidden = residual;
            let residual_ms = stage_start.elapsed().as_secs_f64() * 1.0e3;

            let stage_start = std::time::Instant::now();
            let mut residual = hidden;
            let mut x = residual.clone();
            rms_norm_qwen(&mut x, &layer.post_attention_norm);
            let post_norm_ms = stage_start.elapsed().as_secs_f64() * 1.0e3;

            let stage_start = std::time::Instant::now();
            let ffn = self.ffn_rows(layer, &x, 1)?;
            let ffn_ms = stage_start.elapsed().as_secs_f64() * 1.0e3;

            let stage_start = std::time::Instant::now();
            add_in_place(&mut residual, &ffn);
            hidden = residual;
            let ffn_add_ms = stage_start.elapsed().as_secs_f64() * 1.0e3;
            eprintln!(
                "cpu_decode_profile stage=layer layer={idx} kind={kind} total_ms={:.3} norm_ms={norm_ms:.3} attention_ms={attention_ms:.3} residual_ms={residual_ms:.3} post_norm_ms={post_norm_ms:.3} ffn_ms={ffn_ms:.3} ffn_add_ms={ffn_add_ms:.3}",
                layer_start.elapsed().as_secs_f64() * 1.0e3,
            );
        }
        state.position += 1;

        let stage_start = std::time::Instant::now();
        rms_norm_qwen(&mut hidden, &self.output_norm);
        let output_norm_ms = stage_start.elapsed().as_secs_f64() * 1.0e3;
        let stage_start = std::time::Instant::now();
        let logits = self.token_embd.matmul(&hidden, 1)?;
        let lm_head_ms = stage_start.elapsed().as_secs_f64() * 1.0e3;
        eprintln!(
            "cpu_decode_profile stage=output position={} output_norm_ms={output_norm_ms:.3} lm_head_ms={lm_head_ms:.3} total_ms={:.3}",
            state.position - 1,
            total_start.elapsed().as_secs_f64() * 1.0e3,
        );
        Ok(logits)
    }

    fn forward_token_hidden(&self, token: u32, state: &mut ForwardState) -> Result<Vec<f32>> {
        if state.position >= state.max_context {
            return Err(Error::InvalidRequest(format!(
                "qwen35 prompt exceeds local context cap {}",
                state.max_context
            )));
        }
        let mut hidden = self.token_embd.embedding_rows(&[token])?;
        for (idx, layer) in self.layers.iter().enumerate() {
            let mut residual = hidden;
            let mut x = residual.clone();
            rms_norm_qwen(&mut x, &layer.attn_norm);
            let attn = match (&layer.kind, &mut state.layer_states[idx]) {
                (LayerKind::Delta(weights), LayerState::Delta(runtime)) => {
                    self.delta_block(weights, runtime, &x)?
                }
                (LayerKind::Full(weights), LayerState::Full(runtime)) => {
                    self.full_attention_block(weights, runtime, state.position, &x)?
                }
                _ => return Err(Error::Gguf("qwen35 layer/runtime state mismatch".into())),
            };
            add_in_place(&mut residual, &attn);
            hidden = residual;

            let mut residual = hidden;
            let mut x = residual.clone();
            rms_norm_qwen(&mut x, &layer.post_attention_norm);
            self.ffn_into(layer, &x, &mut state.decode_workspace)?;
            add_in_place(&mut residual, &state.decode_workspace.ffn_out);
            hidden = residual;
        }
        state.position += 1;
        Ok(hidden)
    }

    fn ffn_into(
        &self,
        layer: &LayerWeights,
        hidden: &[f32],
        workspace: &mut CpuDecodeWorkspace,
    ) -> Result<()> {
        let input = layer.ffn_gate.prepare_q8k_matvec(hidden)?;
        let (gate, up) = (&mut workspace.ffn_gate, &mut workspace.ffn_up);
        let (gate_result, up_result) = rayon::join(
            || layer.ffn_gate.matvec_prepared_q8k_into(&input, gate),
            || layer.ffn_up.matvec_prepared_q8k_into(&input, up),
        );
        gate_result?;
        up_result?;
        swiglu_in_place(gate, up);
        let down_input = layer.ffn_down.prepare_q8k_matvec(gate)?;
        layer
            .ffn_down
            .matvec_prepared_q8k_into(&down_input, &mut workspace.ffn_out)
            .map_err(Into::into)
    }

    fn ffn_rows(&self, layer: &LayerWeights, hidden: &[f32], rows: usize) -> Result<Vec<f32>> {
        #[cfg(test)]
        let profile = std::env::var_os("QWEN35_NATIVE_CPU_PROFILE_STAGES").is_some();
        #[cfg(test)]
        let stage_start = std::time::Instant::now();
        let input = layer.ffn_gate.prepare_q8k_rows(hidden, rows)?;
        let (gate, up) = rayon::join(
            || layer.ffn_gate.matmul_prepared_q8k_rows(&input),
            || layer.ffn_up.matmul_prepared_q8k_rows(&input),
        );
        let mut gate = gate?;
        #[cfg(test)]
        let projections_ms = stage_start.elapsed().as_secs_f64() * 1.0e3;
        let up = up?;
        #[cfg(test)]
        let stage_start = std::time::Instant::now();
        swiglu_in_place(&mut gate, &up);
        #[cfg(test)]
        let activation_ms = stage_start.elapsed().as_secs_f64() * 1.0e3;
        #[cfg(test)]
        let stage_start = std::time::Instant::now();
        let out = layer.ffn_down.matmul(&gate, rows)?;
        #[cfg(test)]
        if profile {
            eprintln!(
                "cpu_ffn_stage rows={rows} projections_ms={projections_ms:.3} activation_ms={activation_ms:.3} down_ms={:.3}",
                stage_start.elapsed().as_secs_f64() * 1.0e3,
            );
        }
        Ok(out)
    }

    fn delta_block_rows(
        &self,
        weights: &DeltaWeights,
        state: &mut DeltaState,
        hidden: &[f32],
        rows: usize,
    ) -> Result<Vec<f32>> {
        #[cfg(test)]
        let profile = std::env::var_os("QWEN35_NATIVE_CPU_PROFILE_STAGES").is_some();
        #[cfg(test)]
        let stage_start = std::time::Instant::now();
        let inner = self.inventory.ssm_inner_size;
        let input = weights.attn_qkv.prepare_q8k_rows(hidden, rows)?;
        let ((qkv, z), (beta, alpha)) = rayon::join(
            || {
                rayon::join(
                    || weights.attn_qkv.matmul_prepared_q8k_rows(&input),
                    || weights.attn_gate.matmul_prepared_q8k_rows(&input),
                )
            },
            || {
                rayon::join(
                    || {
                        weights
                            .ssm_beta
                            .matmul_prepared_q8k_rows_or_f32(&input, hidden)
                    },
                    || {
                        weights
                            .ssm_alpha
                            .matmul_prepared_q8k_rows_or_f32(&input, hidden)
                    },
                )
            },
        );
        let mut qkv = qkv?;
        let z = z?;
        let beta = beta?;
        let alpha = alpha?;
        let mut scan_params = vec![(0.0f32, 0.0f32); rows * LINEAR_HEADS];
        #[cfg(test)]
        let projection_ms = stage_start.elapsed().as_secs_f64() * 1.0e3;
        #[cfg(test)]
        let stage_start = std::time::Instant::now();

        causal_conv1d_silu_rows(&mut qkv, &weights.ssm_conv1d, &mut state.conv, rows);
        qkv.par_chunks_mut(inner * 3)
            .zip(beta.par_chunks(LINEAR_HEADS))
            .zip(alpha.par_chunks(LINEAR_HEADS))
            .zip(scan_params.par_chunks_mut(LINEAR_HEADS))
            .for_each(|(((qkv_row, beta_row), alpha_row), params_row)| {
                let (q, rest) = qkv_row.split_at_mut(inner);
                let (k, _) = rest.split_at_mut(inner);
                normalize_linear_qk(q, k);
                for head in 0..LINEAR_HEADS {
                    let beta_h = sigmoid(beta_row[head]);
                    let decay = (-weights.ssm_a[head].exp()
                        * softplus(alpha_row[head] + weights.ssm_dt_bias[head]))
                    .exp()
                    .clamp(0.0, 1.0);
                    params_row[head] = (beta_h, decay);
                }
            });
        #[cfg(test)]
        let prepare_ms = stage_start.elapsed().as_secs_f64() * 1.0e3;
        #[cfg(test)]
        let stage_start = std::time::Instant::now();

        let mut transposed = vec![0.0f32; inner * rows];
        state
            .recurrent
            .par_chunks_mut(LINEAR_HEAD_DIM)
            .zip(transposed.par_chunks_mut(rows))
            .enumerate()
            .for_each(|(state_row, (recurrent_row, out_values))| {
                let head = state_row / LINEAR_HEAD_DIM;
                let value_idx = state_row % LINEAR_HEAD_DIM;
                let head_base = head * LINEAR_HEAD_DIM;
                for row in 0..rows {
                    let qkv_base = row * inner * 3;
                    let qh = &qkv[qkv_base + head_base..qkv_base + head_base + LINEAR_HEAD_DIM];
                    let kh = &qkv[qkv_base + inner + head_base
                        ..qkv_base + inner + head_base + LINEAR_HEAD_DIM];
                    let value = qkv[qkv_base + inner * 2 + head_base + value_idx];
                    let (beta_h, decay) = scan_params[row * LINEAR_HEADS + head];
                    out_values[row] =
                        delta_recurrent_step(recurrent_row, qh, kh, value, beta_h, decay);
                }
            });
        #[cfg(test)]
        let scan_ms = stage_start.elapsed().as_secs_f64() * 1.0e3;
        #[cfg(test)]
        let stage_start = std::time::Instant::now();

        let mut out = vec![0.0f32; rows * inner];
        out.par_chunks_mut(inner)
            .enumerate()
            .for_each(|(row, out_row)| {
                let z_row = &z[row * inner..(row + 1) * inner];
                for head in 0..LINEAR_HEADS {
                    let base = head * LINEAR_HEAD_DIM;
                    let values = &mut out_row[base..base + LINEAR_HEAD_DIM];
                    for value_idx in 0..LINEAR_HEAD_DIM {
                        values[value_idx] = transposed[(base + value_idx) * rows + row];
                    }
                    rms_norm_plain(values, &weights.ssm_norm);
                    mul_silu_in_place(values, &z_row[base..base + LINEAR_HEAD_DIM]);
                }
            });
        #[cfg(test)]
        let normalize_gate_ms = stage_start.elapsed().as_secs_f64() * 1.0e3;
        #[cfg(test)]
        let stage_start = std::time::Instant::now();
        let projected = weights.ssm_out.matmul(&out, rows)?;
        #[cfg(test)]
        if profile {
            eprintln!(
                "cpu_attention_stage kind=delta rows={rows} projections_ms={projection_ms:.3} prepare_ms={prepare_ms:.3} scan_ms={scan_ms:.3} normalize_gate_ms={normalize_gate_ms:.3} output_projection_ms={:.3}",
                stage_start.elapsed().as_secs_f64() * 1.0e3,
            );
        }
        Ok(projected)
    }

    fn full_attention_block_rows(
        &self,
        weights: &FullAttentionWeights,
        state: &mut FullAttentionState,
        start_position: usize,
        hidden: &[f32],
        rows: usize,
    ) -> Result<Vec<f32>> {
        #[cfg(test)]
        let profile = std::env::var_os("QWEN35_NATIVE_CPU_PROFILE_STAGES").is_some();
        #[cfg(test)]
        let stage_start = std::time::Instant::now();
        let kv_k_dim = self.inventory.kv_heads * self.inventory.head_dim;
        let kv_v_dim = self.inventory.kv_heads * self.inventory.value_dim;
        let q_dim = self.inventory.attention_heads * self.inventory.head_dim;
        let input = weights.attn_q.prepare_q8k_rows(hidden, rows)?;
        let (q_fused, (k, v)) = rayon::join(
            || weights.attn_q.matmul_prepared_q8k_rows(&input),
            || {
                rayon::join(
                    || weights.attn_k.matmul_prepared_q8k_rows(&input),
                    || weights.attn_v.matmul_prepared_q8k_rows(&input),
                )
            },
        );
        let q_fused = q_fused?;
        let mut k = k?;
        let v = v?;
        #[cfg(test)]
        let projection_ms = stage_start.elapsed().as_secs_f64() * 1.0e3;
        #[cfg(test)]
        let stage_start = std::time::Instant::now();
        let mut q = vec![0.0f32; rows * q_dim];
        let mut q_gate = vec![0.0f32; rows * q_dim];
        let mut attn_out =
            vec![0.0f32; rows * self.inventory.attention_heads * self.inventory.value_dim];
        let gqa = self.inventory.attention_heads / self.inventory.kv_heads;
        let score_scale = 1.0 / (self.inventory.head_dim as f32).sqrt();

        for row in 0..rows {
            let position = start_position + row;
            let packed = &q_fused[row * q_dim * 2..(row + 1) * q_dim * 2];
            let q_row = &mut q[row * q_dim..(row + 1) * q_dim];
            let q_gate_row = &mut q_gate[row * q_dim..(row + 1) * q_dim];
            for head in 0..self.inventory.attention_heads {
                let src = head * self.inventory.head_dim * 2;
                let dst = head * self.inventory.head_dim;
                q_row[dst..dst + self.inventory.head_dim]
                    .copy_from_slice(&packed[src..src + self.inventory.head_dim]);
                q_gate_row[dst..dst + self.inventory.head_dim].copy_from_slice(
                    &packed[src + self.inventory.head_dim..src + self.inventory.head_dim * 2],
                );
            }
            let k_row = &mut k[row * kv_k_dim..(row + 1) * kv_k_dim];
            let v_row = &v[row * kv_v_dim..(row + 1) * kv_v_dim];
            for head in 0..self.inventory.attention_heads {
                let off = head * self.inventory.head_dim;
                rms_norm_qwen(
                    &mut q_row[off..off + self.inventory.head_dim],
                    &weights.attn_q_norm,
                );
            }
            for head in 0..self.inventory.kv_heads {
                let off = head * self.inventory.head_dim;
                rms_norm_qwen(
                    &mut k_row[off..off + self.inventory.head_dim],
                    &weights.attn_k_norm,
                );
            }
            apply_rope(
                q_row,
                position,
                self.inventory.head_dim,
                self.inventory.rope_dim,
            );
            apply_rope(
                k_row,
                position,
                self.inventory.head_dim,
                self.inventory.rope_dim,
            );
            let k_off = position * kv_k_dim;
            state.k_cache[k_off..k_off + kv_k_dim].copy_from_slice(k_row);
            let v_off = position * kv_v_dim;
            state.v_cache[v_off..v_off + kv_v_dim].copy_from_slice(v_row);
        }
        #[cfg(test)]
        let prepare_ms = stage_start.elapsed().as_secs_f64() * 1.0e3;
        #[cfg(test)]
        let stage_start = std::time::Instant::now();

        attn_out
            .par_chunks_mut(q_dim)
            .enumerate()
            .for_each(|(row, out_row)| {
                let position = start_position + row;
                let q_row = &q[row * q_dim..(row + 1) * q_dim];
                let q_gate_row = &q_gate[row * q_dim..(row + 1) * q_dim];
                for head in 0..self.inventory.attention_heads {
                    let kv_head = head / gqa;
                    let qh = &q_row
                        [head * self.inventory.head_dim..(head + 1) * self.inventory.head_dim];
                    let mut scores = Vec::with_capacity(position + 1);
                    for pos in 0..=position {
                        let key_base = pos * kv_k_dim + kv_head * self.inventory.head_dim;
                        let kh = &state.k_cache[key_base..key_base + self.inventory.head_dim];
                        scores.push(dot(qh, kh) * score_scale);
                    }
                    softmax_in_place(&mut scores);
                    let dst = &mut out_row
                        [head * self.inventory.value_dim..(head + 1) * self.inventory.value_dim];
                    for (pos, score) in scores.iter().copied().enumerate() {
                        let value_base = pos * kv_v_dim + kv_head * self.inventory.value_dim;
                        let vh = &state.v_cache[value_base..value_base + self.inventory.value_dim];
                        for i in 0..self.inventory.value_dim {
                            dst[i] += score * vh[i];
                        }
                    }
                    let gate = &q_gate_row
                        [head * self.inventory.value_dim..(head + 1) * self.inventory.value_dim];
                    mul_sigmoid_in_place(dst, gate);
                }
            });
        #[cfg(test)]
        let attention_ms = stage_start.elapsed().as_secs_f64() * 1.0e3;
        #[cfg(test)]
        let stage_start = std::time::Instant::now();
        let projected = weights
            .attn_output
            .matmul(&attn_out, rows)
            .map_err(Error::from)?;
        #[cfg(test)]
        if profile {
            eprintln!(
                "cpu_attention_stage kind=full rows={rows} projections_ms={projection_ms:.3} prepare_ms={prepare_ms:.3} attention_ms={attention_ms:.3} output_projection_ms={:.3}",
                stage_start.elapsed().as_secs_f64() * 1.0e3,
            );
        }
        Ok(projected)
    }

    fn delta_block(
        &self,
        weights: &DeltaWeights,
        state: &mut DeltaState,
        hidden: &[f32],
    ) -> Result<Vec<f32>> {
        let input = weights.attn_qkv.prepare_q8k_matvec(hidden)?;
        let ((qkv, z), (beta, alpha)) = rayon::join(
            || {
                rayon::join(
                    || weights.attn_qkv.matvec_prepared_q8k(&input),
                    || weights.attn_gate.matvec_prepared_q8k(&input),
                )
            },
            || {
                rayon::join(
                    || weights.ssm_beta.matvec_prepared_q8k_or_f32(&input, hidden),
                    || weights.ssm_alpha.matvec_prepared_q8k_or_f32(&input, hidden),
                )
            },
        );
        let mut qkv = qkv?;
        causal_conv1d_silu(&mut qkv, &weights.ssm_conv1d, &mut state.conv);
        let z = z?;
        let beta = beta?;
        let alpha = alpha?;

        let inner = self.inventory.ssm_inner_size;
        let (q, rest) = qkv.split_at_mut(inner);
        let (k, v) = rest.split_at_mut(inner);
        normalize_linear_qk(q, k);
        let scan_params: [(f32, f32); LINEAR_HEADS] = std::array::from_fn(|head| {
            let beta_h = sigmoid(beta[head]);
            let decay = (-weights.ssm_a[head].exp()
                * softplus(alpha[head] + weights.ssm_dt_bias[head]))
            .exp()
            .clamp(0.0, 1.0);
            (beta_h, decay)
        });

        let mut out = vec![0.0f32; inner];
        state
            .recurrent
            .par_chunks_mut(LINEAR_HEAD_DIM)
            .zip(out.par_iter_mut())
            .enumerate()
            .for_each(|(state_row, (row, output))| {
                let head = state_row / LINEAR_HEAD_DIM;
                let value_idx = state_row % LINEAR_HEAD_DIM;
                let base = head * LINEAR_HEAD_DIM;
                let qh = &q[base..base + LINEAR_HEAD_DIM];
                let kh = &k[base..base + LINEAR_HEAD_DIM];
                let value = v[base + value_idx];
                let (beta_h, decay) = scan_params[head];
                *output = delta_recurrent_step(row, qh, kh, value, beta_h, decay);
            });
        for head in 0..LINEAR_HEADS {
            let base = head * LINEAR_HEAD_DIM;
            rms_norm_plain(&mut out[base..base + LINEAR_HEAD_DIM], &weights.ssm_norm);
        }
        mul_silu_in_place(&mut out, &z);
        weights.ssm_out.matmul(&out, 1).map_err(Into::into)
    }

    fn full_attention_block(
        &self,
        weights: &FullAttentionWeights,
        state: &mut FullAttentionState,
        position: usize,
        hidden: &[f32],
    ) -> Result<Vec<f32>> {
        let kv_k_dim = self.inventory.kv_heads * self.inventory.head_dim;
        let kv_v_dim = self.inventory.kv_heads * self.inventory.value_dim;
        let input = weights.attn_q.prepare_q8k_matvec(hidden)?;
        let (q_fused, (k, v)) = rayon::join(
            || weights.attn_q.matvec_prepared_q8k(&input),
            || {
                rayon::join(
                    || weights.attn_k.matvec_prepared_q8k(&input),
                    || weights.attn_v.matvec_prepared_q8k(&input),
                )
            },
        );
        let q_fused = q_fused?;
        let (mut q, q_gate) = split_full_attention_q_gate(
            &q_fused,
            self.inventory.attention_heads,
            self.inventory.head_dim,
        );
        let mut k = k?;
        let v = v?;
        debug_assert_eq!(k.len(), kv_k_dim);
        debug_assert_eq!(v.len(), kv_v_dim);

        for head in 0..self.inventory.attention_heads {
            let off = head * self.inventory.head_dim;
            rms_norm_qwen(
                &mut q[off..off + self.inventory.head_dim],
                &weights.attn_q_norm,
            );
        }
        for head in 0..self.inventory.kv_heads {
            let off = head * self.inventory.head_dim;
            rms_norm_qwen(
                &mut k[off..off + self.inventory.head_dim],
                &weights.attn_k_norm,
            );
        }
        apply_rope(
            &mut q,
            position,
            self.inventory.head_dim,
            self.inventory.rope_dim,
        );
        apply_rope(
            &mut k,
            position,
            self.inventory.head_dim,
            self.inventory.rope_dim,
        );

        let k_off = position * kv_k_dim;
        state.k_cache[k_off..k_off + kv_k_dim].copy_from_slice(&k);
        let v_off = position * kv_v_dim;
        state.v_cache[v_off..v_off + kv_v_dim].copy_from_slice(&v);

        let mut attn_out = vec![0.0f32; self.inventory.attention_heads * self.inventory.value_dim];
        let gqa = self.inventory.attention_heads / self.inventory.kv_heads;
        let score_scale = 1.0 / (self.inventory.head_dim as f32).sqrt();
        attn_out
            .par_chunks_mut(self.inventory.value_dim)
            .enumerate()
            .for_each(|(head, dst)| {
                let kv_head = head / gqa;
                let qh = &q[head * self.inventory.head_dim..(head + 1) * self.inventory.head_dim];
                let mut scores = Vec::with_capacity(position + 1);
                for pos in 0..=position {
                    let key_base = pos * kv_k_dim + kv_head * self.inventory.head_dim;
                    let kh = &state.k_cache[key_base..key_base + self.inventory.head_dim];
                    scores.push(dot(qh, kh) * score_scale);
                }
                softmax_in_place(&mut scores);
                for (pos, score) in scores.iter().copied().enumerate() {
                    let value_base = pos * kv_v_dim + kv_head * self.inventory.value_dim;
                    let vh = &state.v_cache[value_base..value_base + self.inventory.value_dim];
                    for i in 0..self.inventory.value_dim {
                        dst[i] += score * vh[i];
                    }
                }
                let gate =
                    &q_gate[head * self.inventory.value_dim..(head + 1) * self.inventory.value_dim];
                mul_sigmoid_in_place(dst, gate);
            });
        weights.attn_output.matmul(&attn_out, 1).map_err(Into::into)
    }
}

fn tensor_f32(model: &GgufModel, name: &str) -> Result<Vec<f32>> {
    model.tensor(name)?.to_f32().map_err(Into::into)
}

fn qwen_matrix(model: &GgufModel, name: &str) -> Result<QuantMatrix> {
    QuantMatrix::from_model_q4_x8(model, name).map_err(Into::into)
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

fn prompt_seed(prompt: &str) -> u64 {
    let mut h = DefaultHasher::new();
    prompt.hash(&mut h);
    h.finish()
}

fn add_in_place(lhs: &mut [f32], rhs: &[f32]) {
    debug_assert_eq!(lhs.len(), rhs.len());
    for (lhs, rhs) in lhs.iter_mut().zip(rhs) {
        *lhs += *rhs;
    }
}

fn add_rows(dst: &mut [f32], lhs: &[f32], rhs: &[f32]) {
    debug_assert_eq!(dst.len(), lhs.len());
    debug_assert_eq!(dst.len(), rhs.len());
    const CHUNK_VALUES: usize = 4096;
    dst.par_chunks_mut(CHUNK_VALUES)
        .zip(lhs.par_chunks(CHUNK_VALUES))
        .zip(rhs.par_chunks(CHUNK_VALUES))
        .for_each(|((dst, lhs), rhs)| {
            for ((dst, lhs), rhs) in dst.iter_mut().zip(lhs).zip(rhs) {
                *dst = *lhs + *rhs;
            }
        });
}

fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    debug_assert_eq!(lhs.len(), rhs.len());

    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        unsafe {
            return dot_avx2(lhs, rhs);
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe {
        return dot_neon(lhs, rhs);
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        lhs.iter().zip(rhs).map(|(a, b)| a * b).sum()
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(lhs: &[f32], rhs: &[f32]) -> f32 {
    let vector_len = lhs.len() & !7;
    let mut sum = _mm256_setzero_ps();
    for idx in (0..vector_len).step_by(8) {
        let left = _mm256_loadu_ps(lhs.as_ptr().add(idx));
        let right = _mm256_loadu_ps(rhs.as_ptr().add(idx));
        sum = _mm256_add_ps(sum, _mm256_mul_ps(left, right));
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), sum);
    lanes.iter().sum::<f32>()
        + lhs[vector_len..]
            .iter()
            .zip(&rhs[vector_len..])
            .map(|(left, right)| left * right)
            .sum::<f32>()
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
unsafe fn dot_neon(lhs: &[f32], rhs: &[f32]) -> f32 {
    let vector_len = lhs.len() & !3;
    let mut sum = vdupq_n_f32(0.0);
    for idx in (0..vector_len).step_by(4) {
        let left = vld1q_f32(lhs.as_ptr().add(idx));
        let right = vld1q_f32(rhs.as_ptr().add(idx));
        sum = vaddq_f32(sum, vmulq_f32(left, right));
    }
    let mut lanes = [0.0f32; 4];
    vst1q_f32(lanes.as_mut_ptr(), sum);
    lanes.iter().sum::<f32>()
        + lhs[vector_len..]
            .iter()
            .zip(&rhs[vector_len..])
            .map(|(left, right)| left * right)
            .sum::<f32>()
}

#[inline]
fn delta_recurrent_step(
    state: &mut [f32],
    query: &[f32],
    key: &[f32],
    value: f32,
    beta: f32,
    decay: f32,
) -> f32 {
    debug_assert_eq!(state.len(), LINEAR_HEAD_DIM);
    debug_assert_eq!(query.len(), LINEAR_HEAD_DIM);
    debug_assert_eq!(key.len(), LINEAR_HEAD_DIM);

    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        unsafe {
            return delta_recurrent_step_avx2(state, query, key, value, beta, decay);
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe {
        return delta_recurrent_step_neon(state, query, key, value, beta, decay);
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        delta_recurrent_step_scalar(state, query, key, value, beta, decay)
    }
}

fn delta_recurrent_step_scalar(
    state: &mut [f32],
    query: &[f32],
    key: &[f32],
    value: f32,
    beta: f32,
    decay: f32,
) -> f32 {
    let prior = dot(state, key);
    let delta = (value - decay * prior) * beta;
    let mut attention = 0.0f32;
    for idx in 0..LINEAR_HEAD_DIM {
        state[idx] = decay * state[idx] + key[idx] * delta;
        attention += state[idx] * query[idx];
    }
    attention
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn delta_recurrent_step_avx2(
    state: &mut [f32],
    query: &[f32],
    key: &[f32],
    value: f32,
    beta: f32,
    decay: f32,
) -> f32 {
    let mut prior = _mm256_setzero_ps();
    for idx in (0..LINEAR_HEAD_DIM).step_by(8) {
        let state_values = _mm256_loadu_ps(state.as_ptr().add(idx));
        let key_values = _mm256_loadu_ps(key.as_ptr().add(idx));
        prior = _mm256_add_ps(prior, _mm256_mul_ps(state_values, key_values));
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), prior);
    let prior = lanes.iter().copied().sum::<f32>();
    let delta = (value - decay * prior) * beta;
    let decay = _mm256_set1_ps(decay);
    let delta = _mm256_set1_ps(delta);
    let mut attention = _mm256_setzero_ps();
    for idx in (0..LINEAR_HEAD_DIM).step_by(8) {
        let state_values = _mm256_loadu_ps(state.as_ptr().add(idx));
        let key_values = _mm256_loadu_ps(key.as_ptr().add(idx));
        let updated = _mm256_add_ps(
            _mm256_mul_ps(decay, state_values),
            _mm256_mul_ps(key_values, delta),
        );
        _mm256_storeu_ps(state.as_mut_ptr().add(idx), updated);
        let query_values = _mm256_loadu_ps(query.as_ptr().add(idx));
        attention = _mm256_add_ps(attention, _mm256_mul_ps(updated, query_values));
    }
    _mm256_storeu_ps(lanes.as_mut_ptr(), attention);
    lanes.iter().copied().sum()
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
unsafe fn delta_recurrent_step_neon(
    state: &mut [f32],
    query: &[f32],
    key: &[f32],
    value: f32,
    beta: f32,
    decay: f32,
) -> f32 {
    let mut prior = vdupq_n_f32(0.0);
    for idx in (0..LINEAR_HEAD_DIM).step_by(4) {
        let state_values = vld1q_f32(state.as_ptr().add(idx));
        let key_values = vld1q_f32(key.as_ptr().add(idx));
        prior = vaddq_f32(prior, vmulq_f32(state_values, key_values));
    }
    let mut lanes = [0.0f32; 4];
    vst1q_f32(lanes.as_mut_ptr(), prior);
    let prior = lanes.iter().copied().sum::<f32>();
    let delta = (value - decay * prior) * beta;
    let decay = vdupq_n_f32(decay);
    let delta = vdupq_n_f32(delta);
    let mut attention = vdupq_n_f32(0.0);
    for idx in (0..LINEAR_HEAD_DIM).step_by(4) {
        let state_values = vld1q_f32(state.as_ptr().add(idx));
        let key_values = vld1q_f32(key.as_ptr().add(idx));
        let updated = vaddq_f32(vmulq_f32(decay, state_values), vmulq_f32(key_values, delta));
        vst1q_f32(state.as_mut_ptr().add(idx), updated);
        let query_values = vld1q_f32(query.as_ptr().add(idx));
        attention = vaddq_f32(attention, vmulq_f32(updated, query_values));
    }
    vst1q_f32(lanes.as_mut_ptr(), attention);
    lanes.iter().copied().sum()
}

fn rms_norm_qwen(x: &mut [f32], w: &[f32]) {
    let rstd = rms_rstd(x);
    for (x, w) in x.iter_mut().zip(w) {
        *x = *x * rstd * *w;
    }
}

fn rms_norm_rows_qwen(values: &mut [f32], weights: &[f32], dim: usize) {
    values
        .par_chunks_mut(dim)
        .for_each(|row| rms_norm_qwen(row, weights));
}

fn rms_norm_plain(x: &mut [f32], w: &[f32]) {
    let rstd = rms_rstd(x);
    for (x, w) in x.iter_mut().zip(w) {
        *x = *x * rstd * *w;
    }
}

fn rms_rstd(x: &[f32]) -> f32 {
    let sum_sq = sum_squares_f32(x);
    1.0 / ((sum_sq / x.len() as f32) + RMS_EPS).sqrt()
}

#[inline]
fn sum_squares_f32(values: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        unsafe {
            return sum_squares_f32_avx2(values);
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe {
        return sum_squares_f32_neon(values);
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        values.iter().map(|value| value * value).sum()
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_squares_f32_avx2(values: &[f32]) -> f32 {
    let vector_len = values.len() & !7;
    let mut sum = _mm256_setzero_ps();
    for idx in (0..vector_len).step_by(8) {
        let value = _mm256_loadu_ps(values.as_ptr().add(idx));
        sum = _mm256_add_ps(sum, _mm256_mul_ps(value, value));
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), sum);
    lanes.iter().sum::<f32>()
        + values[vector_len..]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
unsafe fn sum_squares_f32_neon(values: &[f32]) -> f32 {
    let vector_len = values.len() & !3;
    let mut sum = vdupq_n_f32(0.0);
    for idx in (0..vector_len).step_by(4) {
        let value = vld1q_f32(values.as_ptr().add(idx));
        sum = vaddq_f32(sum, vmulq_f32(value, value));
    }
    let mut lanes = [0.0f32; 4];
    vst1q_f32(lanes.as_mut_ptr(), sum);
    lanes.iter().sum::<f32>()
        + values[vector_len..]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
}

fn normalize_linear_qk(q: &mut [f32], k: &mut [f32]) {
    let q_scale = 1.0 / (LINEAR_HEAD_DIM as f32).sqrt();
    for head in 0..LINEAR_HEADS {
        let base = head * LINEAR_HEAD_DIM;
        let qh = &mut q[base..base + LINEAR_HEAD_DIM];
        let kh = &mut k[base..base + LINEAR_HEAD_DIM];
        let q_norm = (sum_squares_f32(qh) + RMS_EPS).sqrt();
        let k_norm = (sum_squares_f32(kh) + RMS_EPS).sqrt();
        for v in qh {
            *v = *v / q_norm * q_scale;
        }
        for v in kh {
            *v /= k_norm;
        }
    }
}

fn split_full_attention_q_gate(
    packed: &[f32],
    heads: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let q_dim = heads * head_dim;
    debug_assert_eq!(packed.len(), q_dim * 2);
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

fn causal_conv1d_silu(values: &mut [f32], weights: &[f32], state: &mut [f32]) {
    debug_assert_eq!(state.len(), values.len() * CONV_KERNEL);
    debug_assert_eq!(weights.len(), values.len() * CONV_KERNEL);
    for channel in 0..values.len() {
        let base = channel * CONV_KERNEL;
        for i in 0..CONV_KERNEL - 1 {
            state[base + i] = state[base + i + 1];
        }
        state[base + CONV_KERNEL - 1] = values[channel];
        let mut acc = 0.0f32;
        for i in 0..CONV_KERNEL {
            acc += state[base + i] * weights[base + i];
        }
        values[channel] = acc;
    }
    silu_in_place(values);
}

struct SharedMutF32(*mut f32);

// Each worker receives a disjoint channel range for every row. The pointer is
// shared only to express that strided partitioning, which slices cannot model.
unsafe impl Send for SharedMutF32 {}
unsafe impl Sync for SharedMutF32 {}

impl SharedMutF32 {
    unsafe fn slice<'a>(&self, offset: usize, len: usize) -> &'a mut [f32] {
        std::slice::from_raw_parts_mut(self.0.add(offset), len)
    }
}

fn causal_conv1d_silu_rows(values: &mut [f32], weights: &[f32], state: &mut [f32], rows: usize) {
    debug_assert!(rows > 0);
    debug_assert_eq!(values.len() % rows, 0);
    let channels = values.len() / rows;
    debug_assert_eq!(state.len(), channels * CONV_KERNEL);
    debug_assert_eq!(weights.len(), channels * CONV_KERNEL);
    if rows == 1 {
        causal_conv1d_silu(values, weights, state);
        return;
    }

    let shared_values = SharedMutF32(values.as_mut_ptr());
    state
        .par_chunks_mut(CONV_CHANNEL_BLOCK * CONV_KERNEL)
        .enumerate()
        .for_each(|(block, state_block)| {
            let channel_start = block * CONV_CHANNEL_BLOCK;
            let block_channels = state_block.len() / CONV_KERNEL;
            for row in 0..rows {
                let values_block =
                    unsafe { shared_values.slice(row * channels + channel_start, block_channels) };
                for (channel, value) in values_block.iter_mut().enumerate() {
                    let state_base = channel * CONV_KERNEL;
                    let weight_base = (channel_start + channel) * CONV_KERNEL;
                    for tap in 0..CONV_KERNEL - 1 {
                        state_block[state_base + tap] = state_block[state_base + tap + 1];
                    }
                    state_block[state_base + CONV_KERNEL - 1] = *value;
                    let mut acc = 0.0f32;
                    for tap in 0..CONV_KERNEL {
                        acc += state_block[state_base + tap] * weights[weight_base + tap];
                    }
                    *value = acc;
                }
                silu_in_place(values_block);
            }
        });
}

fn apply_rope(values: &mut [f32], position: usize, head_dim: usize, rope_dim: usize) {
    if position == 0 {
        return;
    }
    let half = rope_dim / 2;
    for head in 0..values.len() / head_dim {
        let base = head * head_dim;
        for i in 0..half {
            let theta = (position as f32) * ROPE_THETA.powf(-((2 * i) as f32) / rope_dim as f32);
            let (sin, cos) = theta.sin_cos();
            let a = values[base + i];
            let b = values[base + i + half];
            values[base + i] = a * cos - b * sin;
            values[base + i + half] = a * sin + b * cos;
        }
    }
}

fn softmax_in_place(values: &mut [f32]) {
    let max = values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |a, b| a.max(b));
    let sum = exp_sum_shifted_in_place(values, max);
    if sum > 0.0 {
        for value in values {
            *value /= sum;
        }
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[cfg(test)]
mod math_tests {
    use super::*;

    #[test]
    fn simd_dot_stays_close_to_scalar() {
        let lhs = (0..259)
            .map(|idx| ((idx * 29 % 101) as f32 - 50.0) / 37.0)
            .collect::<Vec<_>>();
        let rhs = (0..259)
            .map(|idx| ((idx * 17 % 89) as f32 - 44.0) / 41.0)
            .collect::<Vec<_>>();
        let expected = lhs.iter().zip(&rhs).map(|(a, b)| a * b).sum::<f32>();
        let actual = dot(&lhs, &rhs);
        let tolerance = 3.0e-5 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "SIMD dot drift: actual={actual:.8e} expected={expected:.8e} tolerance={tolerance:.8e}",
        );
    }

    #[test]
    fn parallel_causal_conv_rows_matches_tokenwise() {
        let rows = 7;
        let channels = CONV_CHANNEL_BLOCK * 2 + 13;
        let input = (0..rows * channels)
            .map(|idx| ((idx * 29 % 251) as f32 - 125.0) / 97.0)
            .collect::<Vec<_>>();
        let weights = (0..channels * CONV_KERNEL)
            .map(|idx| ((idx * 17 % 127) as f32 - 63.0) / 113.0)
            .collect::<Vec<_>>();
        let initial_state = (0..channels * CONV_KERNEL)
            .map(|idx| ((idx * 37 % 149) as f32 - 74.0) / 131.0)
            .collect::<Vec<_>>();

        let mut expected = input.clone();
        let mut expected_state = initial_state.clone();
        for row in 0..rows {
            causal_conv1d_silu(
                &mut expected[row * channels..(row + 1) * channels],
                &weights,
                &mut expected_state,
            );
        }

        let mut actual = input;
        let mut actual_state = initial_state;
        causal_conv1d_silu_rows(&mut actual, &weights, &mut actual_state, rows);
        assert_eq!(actual, expected);
        assert_eq!(actual_state, expected_state);
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod x86_tests {
    use super::*;

    #[test]
    fn avx2_recurrent_step_stays_close_to_scalar() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut scalar_state = (0..LINEAR_HEAD_DIM)
            .map(|idx| ((idx * 29 % 97) as f32 - 48.0) / 257.0)
            .collect::<Vec<_>>();
        let mut simd_state = scalar_state.clone();
        let query = (0..LINEAR_HEAD_DIM)
            .map(|idx| ((idx * 17 % 89) as f32 - 44.0) / 193.0)
            .collect::<Vec<_>>();
        let key = (0..LINEAR_HEAD_DIM)
            .map(|idx| ((idx * 37 % 101) as f32 - 50.0) / 211.0)
            .collect::<Vec<_>>();

        let mut max_attention_diff = 0.0f32;
        for step in 0..64 {
            let value = (step as f32 - 31.0) / 79.0;
            let beta = 0.2 + step as f32 / 128.0;
            let decay = 0.91 + (step % 7) as f32 / 100.0;
            let scalar =
                delta_recurrent_step_scalar(&mut scalar_state, &query, &key, value, beta, decay);
            let simd = unsafe {
                delta_recurrent_step_avx2(&mut simd_state, &query, &key, value, beta, decay)
            };
            max_attention_diff = max_attention_diff.max((scalar - simd).abs());
        }
        let max_state_diff = scalar_state
            .iter()
            .zip(&simd_state)
            .map(|(scalar, simd)| (scalar - simd).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_state_diff <= 2.0e-5,
            "AVX2 recurrent state drift: {max_state_diff:.6e}"
        );
        assert!(
            max_attention_diff <= 2.0e-5,
            "AVX2 recurrent attention drift: {max_attention_diff:.6e}"
        );
    }
}

#[cfg(all(test, target_arch = "aarch64", target_feature = "neon"))]
mod arm_tests {
    use super::*;

    #[test]
    fn neon_recurrent_step_stays_close_to_scalar() {
        let mut scalar_state = (0..LINEAR_HEAD_DIM)
            .map(|idx| ((idx * 29 % 97) as f32 - 48.0) / 257.0)
            .collect::<Vec<_>>();
        let mut simd_state = scalar_state.clone();
        let query = (0..LINEAR_HEAD_DIM)
            .map(|idx| ((idx * 17 % 89) as f32 - 44.0) / 193.0)
            .collect::<Vec<_>>();
        let key = (0..LINEAR_HEAD_DIM)
            .map(|idx| ((idx * 37 % 101) as f32 - 50.0) / 211.0)
            .collect::<Vec<_>>();

        let mut max_attention_diff = 0.0f32;
        for step in 0..64 {
            let value = (step as f32 - 31.0) / 79.0;
            let beta = 0.2 + step as f32 / 128.0;
            let decay = 0.91 + (step % 7) as f32 / 100.0;
            let scalar =
                delta_recurrent_step_scalar(&mut scalar_state, &query, &key, value, beta, decay);
            let simd = unsafe {
                delta_recurrent_step_neon(&mut simd_state, &query, &key, value, beta, decay)
            };
            max_attention_diff = max_attention_diff.max((scalar - simd).abs());
        }
        let max_state_diff = scalar_state
            .iter()
            .zip(&simd_state)
            .map(|(scalar, simd)| (scalar - simd).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_state_diff <= 2.0e-5,
            "NEON recurrent state drift: {max_state_diff:.6e}"
        );
        assert!(
            max_attention_diff <= 2.0e-5,
            "NEON recurrent attention drift: {max_attention_diff:.6e}"
        );
    }
}
