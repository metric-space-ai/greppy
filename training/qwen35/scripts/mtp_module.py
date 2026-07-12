"""Training-time MTP (multi-token-prediction / NextN) head for Qwen3.5.

Reference semantics matched: vLLM `vllm/model_executor/models/qwen3_next_mtp.py`
(Qwen3NextMultiTokenPredictor) + vLLM `vllm/v1/spec_decode/llm_base_proposer.py`
(EagleProposer input construction).

Dataflow (row t, 0-indexed, over a teacher-forced sequence of T tokens):

    e_t = pre_fc_norm_embedding( embed_tokens(token_{t+1}) )
    h_t = pre_fc_norm_hidden( trunk_hidden_t )        # trunk hidden = POST-final-norm
                                                      # output of the target model
    x_t = fc( cat([e_t, h_t], dim=-1) )               # NOTE: embedding first
    y_t = DecoderLayer(x_t, rotary position = t)      # one full-attention Qwen3.5
                                                      # layer (gated attention)
    logits_t = lm_head( norm(y_t) )                   # predicts token_{t+2}

Position convention: vLLM's proposer shifts input_ids left by one but leaves
positions UNCHANGED ("Simply rotate the input ids and leave the positions
unchanged", set_inputs_first_pass), i.e. the pair (hidden_t, embed(token_{t+1}))
is processed at rotary position t.

All submodules are reused from transformers' own Qwen3.5 implementation
(Qwen3_5DecoderLayer / Qwen3_5Attention with attn_output_gate, Qwen3_5RMSNorm
with zero-centered (1+w) weights, Qwen3_5TextRotaryEmbedding with interleaved
mrope + partial_rotary_factor) so semantics are identical to the trunk.

embed_tokens and lm_head are SHARED with the trunk
(mtp_use_dedicated_embeddings=false, tie_word_embeddings=true); they are held
as unregistered references so this module only owns/saves the mtp.* weights.
"""

import glob
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
    create_causal_mask,
)


def _load_mtp_state_dict(hf_dir):
    """Collect all mtp.* tensors from the checkpoint's safetensors shards."""
    sd = {}
    index = os.path.join(hf_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        weight_map = json.load(open(index))["weight_map"]
        shards = sorted({v for k, v in weight_map.items() if k.startswith("mtp.")})
        files = [os.path.join(hf_dir, s) for s in shards]
    else:
        files = sorted(glob.glob(os.path.join(hf_dir, "*.safetensors")))
    for f in files:
        with safe_open(f, framework="pt") as s:
            for k in s.keys():
                if k.startswith("mtp."):
                    sd[k[len("mtp."):]] = s.get_tensor(k)
    if not sd:
        raise FileNotFoundError(f"no mtp.* tensors found in {hf_dir}")
    return sd


class Qwen35MTP(nn.Module):
    def __init__(self, text_config, attn_implementation="sdpa"):
        super().__init__()
        cfg = text_config
        self.config = cfg
        cfg._attn_implementation = attn_implementation
        h = cfg.hidden_size
        eps = cfg.rms_norm_eps
        self.fc = nn.Linear(2 * h, h, bias=False)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(h, eps=eps)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(h, eps=eps)
        # one full-attention decoder layer; pick a layer_idx whose layer_type
        # is "full_attention" so the layer builds Qwen3_5Attention (gated).
        full_idx = cfg.layer_types.index("full_attention")
        n_mtp = getattr(cfg, "mtp_num_hidden_layers", 1) or 1
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(cfg, layer_idx=full_idx) for _ in range(n_mtp)]
        )
        self.norm = Qwen3_5RMSNorm(h, eps=eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=cfg)
        # unregistered references to trunk-shared modules (not saved/trained here)
        self._shared = {"embed_tokens": None, "lm_head": None}

    # ------------------------------------------------------------------ setup
    def bind(self, embed_tokens=None, lm_head=None):
        """Attach the trunk's shared embedding / lm_head (kept unregistered)."""
        if embed_tokens is not None:
            self._shared["embed_tokens"] = embed_tokens
        if lm_head is not None:
            self._shared["lm_head"] = lm_head
        return self

    @classmethod
    def from_checkpoint(cls, hf_dir, device="cuda", dtype=torch.bfloat16,
                        attn_implementation="sdpa"):
        config = AutoConfig.from_pretrained(hf_dir)
        text_config = getattr(config, "text_config", config)
        m = cls(text_config, attn_implementation=attn_implementation)
        sd = _load_mtp_state_dict(hf_dir)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        # rotary_emb.inv_freq is a non-persistent buffer -> ignore
        missing = [k for k in missing if "rotary_emb" not in k]
        if missing or unexpected:
            raise RuntimeError(f"MTP load mismatch: missing={missing} "
                               f"unexpected={unexpected}")
        return m.to(device=device, dtype=dtype)

    # ---------------------------------------------------------------- forward
    def forward(self, hidden_states, input_ids, attention_mask=None,
                position_offset=0, embed_tokens=None, lm_head=None):
        """
        hidden_states: [B, T, H] trunk POST-final-norm hidden states, aligned
                       with input_ids (hidden_states[:, t] summarizes ..t).
        input_ids:     [B, T] the same teacher-forced token ids.
        attention_mask:[B, T] optional padding mask (1 = keep).

        Returns logits [B, T-1, V]; logits[:, t] is the MTP prediction of
        token_{t+2} (given real tokens up to t+1). If lm_head is unbound,
        returns the pre-head hidden states [B, T-1, H] instead.
        """
        embed_tokens = embed_tokens or self._shared["embed_tokens"]
        lm_head = lm_head if lm_head is not None else self._shared["lm_head"]
        if embed_tokens is None:
            raise RuntimeError("bind(embed_tokens=...) first (shared with trunk)")

        # align: embedding of token_{t+1} with hidden state of position t
        emb = embed_tokens(input_ids[:, 1:])            # [B, T-1, H]
        hid = hidden_states[:, :-1]                     # [B, T-1, H]
        x = torch.cat(
            [self.pre_fc_norm_embedding(emb), self.pre_fc_norm_hidden(hid)],
            dim=-1,
        )
        x = self.fc(x)

        B, S, _ = x.shape
        # vLLM keeps target positions unchanged for the shifted tokens
        position_ids = (
            torch.arange(S, device=x.device) + position_offset
        ).unsqueeze(0).expand(B, -1)
        position_embeddings = self.rotary_emb(x, position_ids)

        mask_in = attention_mask[:, 1:] if attention_mask is not None else None
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=x,
            attention_mask=mask_in,
            past_key_values=None,
            position_ids=position_ids,
        )

        for layer in self.layers:
            x = layer(
                x,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=position_ids,
            )
        x = self.norm(x)
        if lm_head is None:
            return x
        return lm_head(x)

    # ------------------------------------------------------------------- loss
    def mtp_loss(self, trunk_outputs, input_ids, labels, lm_head=None,
                 embed_tokens=None, attention_mask=None):
        """CE loss of predicting labels shifted by 2 (next-next token).

        trunk_outputs: trunk model output with .last_hidden_state (post-final-
                       norm) OR a [B, T, H] tensor directly.
        labels: [B, T], -100 = ignore (same convention as HF CausalLM).
        """
        hidden = getattr(trunk_outputs, "last_hidden_state", trunk_outputs)
        logits = self.forward(hidden, input_ids, attention_mask=attention_mask,
                              embed_tokens=embed_tokens, lm_head=lm_head)
        # logits[:, t] predicts token t+2 -> valid rows t = 0..T-3
        pred = logits[:, :-1]                 # [B, T-2, V]
        tgt = labels[:, 2:]                   # [B, T-2]
        return F.cross_entropy(
            pred.reshape(-1, pred.size(-1)).float(),
            tgt.reshape(-1),
            ignore_index=-100,
        )
