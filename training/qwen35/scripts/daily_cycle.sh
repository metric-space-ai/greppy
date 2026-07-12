#!/usr/bin/env bash
# Daily cycle: continue training on the day's new SFT rows, export, gate, protocol.
# Cron: 0 3 * * *  (after this, the hourly nas_sync persists everything)
set -uo pipefail
cd /home/metricspace/nano-summary-pilot
PROD=prod
STATE=$PROD/.last_trained_rows
DAY=$(date +%F)
CKPT_ROOT=/home/metricspace/models/nano-daily
STATUS=$PROD/DAILY_STATUS.md
PY=/home/metricspace/nano-ft-venv/bin/python3
MIN_NEW=1000

log() { echo "[$DAY] $*"; }
protocol() {
  { echo ""; echo "## $DAY"; echo "$1"; } >> "$STATUS"
}

# disk preflight: a full disk killed a 16h retrain at save time once
FREE_GB=$(df --output=avail -BG /mnt/nvme1 | tail -1 | tr -dc 0-9)
if [ "${FREE_GB:-0}" -lt 15 ]; then
  protocol "- ABORTED: only ${FREE_GB}GB free on checkpoint volume (need 15)"
  exit 1
fi

# single-training lock: skip if a (manual) training holds it
exec 9>/home/metricspace/nano-summary-pilot/.train.lock
flock -n 9 || { protocol "- SKIPPED: another training holds the train lock"; exit 0; }

python3 rebuild_sft.py > /tmp/all_sft.jsonl 2>/dev/null || true
TOTAL=$(wc -l < /tmp/all_sft.jsonl 2>/dev/null || echo 0)
LAST=$(cat "$STATE" 2>/dev/null || echo 0)
NEW=$((TOTAL - LAST))
log "total=$TOTAL last_trained=$LAST new=$NEW"

if [ "$NEW" -lt "$MIN_NEW" ]; then
  protocol "- generated total: $TOTAL sft rows (+$NEW new) — below $MIN_NEW, no training today"
  exit 0
fi

tail -n "$NEW" /tmp/all_sft.jsonl > /tmp/daily_delta.jsonl
PREV=$(find "$CKPT_ROOT" -maxdepth 1 -type d -name "ckpt-*" 2>/dev/null | sort | tail -1)
INIT=${PREV:-/home/metricspace/models/Qwen3.5-0.8B-HF}
OUT="$CKPT_ROOT/ckpt-$DAY"
mkdir -p "$CKPT_ROOT"
log "training from $INIT on $NEW rows -> $OUT"

CUDA_VISIBLE_DEVICES=1,2 SMOKE_BS=2 SMOKE_ACCUM=4 SMOKE_EPOCHS=1 \
SMOKE_INIT="$INIT" SMOKE_DATA=/tmp/daily_delta.jsonl SMOKE_OUT="$OUT" \
"$PY" -m torch.distributed.run --nproc_per_node=2 train_smoke.py \
  > "$PROD/train_$DAY.log" 2>&1
TRAIN_RC=$?
EVAL_LOSS=$(grep -o "'eval_loss': [0-9.]*" "$PROD/train_$DAY.log" | tail -1 | grep -o '[0-9.]*$')
if [ $TRAIN_RC -ne 0 ] || [ ! -d "$OUT" ]; then
  protocol "- TRAINING FAILED (rc=$TRAIN_RC, new=$NEW) — see train_$DAY.log; state not advanced"
  exit 1
fi

BF16="$CKPT_ROOT/ckpt-$DAY-BF16.gguf"
Q4="$CKPT_ROOT/ckpt-$DAY-Q4_K_M.gguf"
STRIPPED="$CKPT_ROOT/ckpt-$DAY-Q4_K_M-stripped.gguf"
python3 ~/llama.cpp/convert_hf_to_gguf.py "$OUT" --outfile "$BF16" --outtype bf16 >> "$PROD/train_$DAY.log" 2>&1 \
  && ~/llama.cpp/build/bin/llama-quantize "$BF16" "$Q4" Q4_K_M >> "$PROD/train_$DAY.log" 2>&1 \
  && python3 strip_mtp.py "$Q4" "$STRIPPED" >> "$PROD/train_$DAY.log" 2>&1
if [ ! -f "$STRIPPED" ]; then
  protocol "- trained (+$NEW rows, eval_loss=$EVAL_LOSS) but EXPORT FAILED — see train_$DAY.log"
  exit 1
fi
rm -f "$BF16"

GATES=$(python3 daily_eval.py "$STRIPPED" 30 2>>"$PROD/train_$DAY.log" | tail -1)
MTP=$(CUDA_VISIBLE_DEVICES=1 "$PY" validate_mtp_brief.py --model "$OUT" --n 150 2>/dev/null | grep -o 'acceptance (vs trunk greedy) = [0-9.]*%' | grep -o '[0-9.]*%')

echo "$TOTAL" > "$STATE"
RATE=$(echo "$GATES" | grep -o '"format_valid_rate": [0-9.]*' | grep -o '[0-9.]*$' || echo 0)
# attempt promotion at a fixed daily bar (n=30); the real ratchet is the
# n=100 promote gate + the sha-NOOP check in auto_release
CAND=""
if python3 -c "import sys; sys.exit(0 if float('${RATE:-0}') >= 0.95 else 1)"; then
  if ./promote_checkpoint.sh "$OUT" > "$PROD/promote_$DAY.log" 2>&1; then
    STAGE=/mnt/asustor/LLM-Store/grepplus-offload/nano-summary/release-staging
    rm -f "$STAGE/LATEST" && ln -s "$STAGE/ckpt-$DAY" "$STAGE/LATEST" 2>/dev/null \
      || { rm -rf "$STAGE/LATEST.dir"; cp -r "$STAGE/ckpt-$DAY" "$STAGE/LATEST.dir"; }
    REL=$(./auto_release.sh "$Q4" "holdout ${GATES//\"/} | MTP ${MTP:-n/a}" 2>&1 | tail -1)
    CAND="
- ** PROMOTED ** best gated checkpoint staged: release-staging/ckpt-$DAY
- auto-release to main: $REL"
  else
    CAND="
- ** PROMOTION CANDIDATE FAILED GATES ** (rate=$RATE) - see promote_$DAY.log"
  fi
fi
protocol "- rows total: $TOTAL (+$NEW trained today, 1 epoch continuation from ${PREV:-BASE})
- eval_loss: ${EVAL_LOSS:-n/a} | holdout gates: ${GATES:-n/a} | MTP ${MTP:-n/a}
- checkpoint: $OUT (+ full/stripped Q4 gguf)$CAND"
log "done"
