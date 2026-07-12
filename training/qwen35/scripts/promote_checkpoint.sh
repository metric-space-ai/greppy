#!/usr/bin/env bash
# Gate a daily/weekly checkpoint and stage it for release into greppy.
# Usage: promote_checkpoint.sh <ckpt-dir e.g. ~/models/nano-daily/ckpt-2026-07-11>
# Hard gates; on pass, stages artifacts + SHAs on the NAS and prints the exact
# greppy build.rs / release-asset steps. The final flip into greppy stays manual.
set -uo pipefail
CKPT=$(readlink -f "${1:?usage: promote_checkpoint.sh <ckpt-dir>}")
TAG=$(basename "$CKPT")
STRIPPED="$CKPT-Q4_K_M-stripped.gguf"
FULL="$CKPT-Q4_K_M.gguf"
TOK=/tmp/gp-qwen-assets/tokenizer.json
STAGE=/mnt/asustor/LLM-Store/grepplus-offload/nano-summary/release-staging/$TAG
PY=/home/metricspace/nano-ft-venv/bin/python3
cd /home/metricspace/nano-summary-pilot

[ -f "$STRIPPED" ] && [ -f "$FULL" ] || { echo "FAIL: missing gguf exports for $TAG"; exit 1; }

echo "=== gate 1: holdout format eval (n=100)"
GATES=$(python3 daily_eval.py "$STRIPPED" 100 | tail -1)
echo "$GATES"
VALID=$(echo "$GATES" | python3 -c "import json,sys; print(json.load(sys.stdin)['format_valid_rate'])")
EMPTY=$(echo "$GATES" | python3 -c "import json,sys; print(json.load(sys.stdin)['empty_rate'])")
python3 -c "import sys; sys.exit(0 if float('$VALID')>=0.90 and float('$EMPTY')<=0.05 else 1)" \
  || { echo "FAIL: format gate (valid=$VALID empty=$EMPTY, need >=0.90 / <=0.05)"; exit 1; }

echo "=== gate 2: native quality test"
(cd ~/codex-grepplus-qwen35 && PATH=$HOME/.cargo/bin:$PATH CARGO_TARGET_DIR=$HOME/codex-grepplus-qwen35-target \
 QWEN35_NATIVE_GGUF="$STRIPPED" QWEN35_NATIVE_TOKENIZER="$TOK" \
 cargo test --release -q -p greppy-qwen35-native qwen35_cpu_quality -- --ignored 2>&1 | grep -q "test result: ok") \
  || { echo "FAIL: qwen35_cpu_quality"; exit 1; }

echo "=== gate 3: MTP acceptance (archive artifact)"
ACC=$(CUDA_VISIBLE_DEVICES=1 "$PY" validate_mtp.py --model "$CKPT" 2>/dev/null | grep postnorm_off0 | grep -o 'acceptance=[0-9.]*' | grep -o '[0-9.]*')
python3 -c "import sys; sys.exit(0 if float('${ACC:-0}')>=55 else 1)" \
  || { echo "WARN: MTP acceptance ${ACC:-n/a}% < 55% (archive quality only, not blocking)"; }

echo "=== staging to NAS"
mkdir -p "$STAGE"
cp "$STRIPPED" "$STAGE/Qwen3.5-0.8B-Q4_K_M.gguf"
cp "$FULL" "$STAGE/Qwen3.5-0.8B-Q4_K_M-with-mtp.gguf"
cp "$TOK" "$STAGE/qwen35-tokenizer.json"
( cd "$STAGE" && sha256sum * > SHA256SUMS )
echo "$GATES" > "$STAGE/gates.json"
echo "mtp_acceptance=${ACC:-n/a}" >> "$STAGE/gates.json"
# content judge runs weekly (owner: punctual M3 spot checks, trust the loss curve)

echo ""
echo "PROMOTED to staging: $STAGE"
echo "Release steps (manual):"
echo "  1. upload $STAGE/Qwen3.5-0.8B-Q4_K_M.gguf as the release model asset"
echo "  2. set GGUF_SHA in greppy crates/cli/build.rs to:"
grep 'Qwen3.5-0.8B-Q4_K_M.gguf' "$STAGE/SHA256SUMS"
echo "  3. build + run product spot tests (brief/semantic on a real repo) before tagging"
