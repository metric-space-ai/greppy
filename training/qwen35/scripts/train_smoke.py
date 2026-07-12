#!/usr/bin/env python3
"""Smoke SFT: FULL-parameter finetune of Qwen3.5-0.8B on smoke_sft.jsonl, single GPU.

Run inside the venv: ~/nano-ft-venv/bin/python3 train_smoke.py
Writes merged model to ~/models/Qwen3.5-0.8B-smoke-merged/
"""
import json, os, random
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch.nn as nn
import sys
sys.path.insert(0, os.path.expanduser("~/nano-summary-pilot"))
from mtp_module import Qwen35MTP

MTP_LAMBDA = 0.2

class TrunkWithMTP(nn.Module):
    def __init__(self, trunk, mtp):
        super().__init__()
        self.trunk = trunk
        self.mtp = mtp

    def forward(self, input_ids, attention_mask=None, labels=None):
        out = self.trunk(input_ids=input_ids, attention_mask=attention_mask,
                         labels=labels, output_hidden_states=True)
        hidden = out.hidden_states[-1]  # post-final-norm (llama-lineage ordering)
        l_mtp = self.mtp.mtp_loss(hidden, input_ids, labels, attention_mask=attention_mask)
        return {"loss": out.loss + MTP_LAMBDA * l_mtp,
                "loss_main": out.loss.detach(), "loss_mtp": l_mtp.detach()}

BASE = os.path.expanduser("~/models/Qwen3.5-0.8B-HF")
INIT = os.path.expanduser(os.environ.get("SMOKE_INIT", BASE))
DATA = os.path.expanduser(os.environ.get("SMOKE_DATA", "~/nano-summary-pilot/smoke_sft.jsonl"))
OUT = os.path.expanduser(os.environ.get("SMOKE_OUT", "~/models/Qwen3.5-0.8B-smoke-merged"))
EPOCHS = float(os.environ.get("SMOKE_EPOCHS", "3"))
MAXLEN = 1280
random.seed(3)

tok = AutoTokenizer.from_pretrained(BASE)

class SftData(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        p_ids = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        c_ids = tok(r["completion"], add_special_tokens=False)["input_ids"]
        ids = (p_ids + c_ids)[:MAXLEN]
        labels = ([-100] * len(p_ids) + c_ids)[:MAXLEN]
        return {"input_ids": ids, "labels": labels}

def collate(batch):
    ml = max(len(b["input_ids"]) for b in batch)
    pad = tok.pad_token_id or 0
    return {
        "input_ids": torch.tensor([b["input_ids"] + [pad] * (ml - len(b["input_ids"])) for b in batch]),
        "labels": torch.tensor([b["labels"] + [-100] * (ml - len(b["labels"])) for b in batch]),
        "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (ml - len(b["input_ids"])) for b in batch]),
    }

FIXED_EVAL = os.path.expanduser("~/nano-summary-pilot/prod/fixed_eval_sft.jsonl")
rows = [json.loads(l) for l in open(DATA)]
def fits(r):
    n = len(tok(r["prompt"], add_special_tokens=False)["input_ids"]) + len(tok(r["completion"], add_special_tokens=False)["input_ids"])
    return n <= MAXLEN
n0 = len(rows)
rows = [r for r in rows if fits(r)]
print(f"kept {len(rows)}/{n0} rows within {MAXLEN} tokens")
random.shuffle(rows)
if os.path.exists(FIXED_EVAL):
    train_rows = rows
    eval_rows = [r for r in (json.loads(l) for l in open(FIXED_EVAL)) if fits(r)]
    print(f"eval on FIXED set ({len(eval_rows)} rows within {MAXLEN} tokens) - comparable across cycles")
else:
    n_eval = max(16, len(rows) // 20)
    train_rows, eval_rows = rows[n_eval:], rows[:n_eval]
print(f"train {len(train_rows)}, eval {len(eval_rows)}")

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
# fp32 master weights: with bf16 params, 1e-5 updates underflow the bf16 ULP for
# large-magnitude weights (all RMSNorms froze silently). bf16=True keeps autocast.
trunk = AutoModelForCausalLM.from_pretrained(INIT, dtype=torch.float32)
trunk.config.use_cache = False
trunk.gradient_checkpointing_enable()
mtp = Qwen35MTP.from_checkpoint(INIT, device=f"cuda:{local_rank}", dtype=torch.float32)
mtp.bind(embed_tokens=trunk.get_input_embeddings(), lm_head=trunk.get_output_embeddings())
model = TrunkWithMTP(trunk, mtp)
print("full finetune:", sum(p.numel() for p in model.parameters() if p.requires_grad),
      "trainable params (incl.", sum(p.numel() for p in mtp.parameters()), "MTP)")

BS = int(os.environ.get("SMOKE_BS", "1"))
ACCUM = int(os.environ.get("SMOKE_ACCUM", "16"))
MAX_STEPS = int(os.environ.get("SMOKE_MAX_STEPS", "-1"))
args = TrainingArguments(
    output_dir=os.path.expanduser("~/nano-ft-out"),
    per_device_train_batch_size=BS,
    per_device_eval_batch_size=1,
    optim="adafactor",
    gradient_accumulation_steps=ACCUM,
    max_steps=MAX_STEPS,
    num_train_epochs=EPOCHS,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="no",
    report_to=[],
)
trainer = Trainer(model=model, args=args, train_dataset=SftData(train_rows),
                  eval_dataset=SftData(eval_rows), data_collator=collate)
trainer.train()
print("eval:", trainer.evaluate())

if int(os.environ.get("RANK", "0")) != 0 or os.environ.get("SMOKE_SAVE", "1") == "0":
    sys.exit(0)
trunk.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
# merge co-trained MTP tensors back so convert_hf_to_gguf exports blk.24
from safetensors.torch import load_file, save_file
sf_path = os.path.join(OUT, "model.safetensors")
tensors = load_file(sf_path)
for k, v in mtp.state_dict().items():
    tensors[f"mtp.{k}"] = v.detach().to(torch.bfloat16).cpu()
save_file(tensors, sf_path, metadata={"format": "pt"})
print("merged", len(mtp.state_dict()), "mtp tensors into", sf_path)
# convert_hf_to_gguf needs the full config set from the base dir
for f in os.listdir(BASE):
    if f.endswith(".json") and "index" not in f and not os.path.exists(os.path.join(OUT, f)):
        import shutil
        shutil.copy(os.path.join(BASE, f), os.path.join(OUT, f))
print("merged model saved to", OUT)
