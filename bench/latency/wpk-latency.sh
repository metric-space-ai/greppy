#!/usr/bin/env bash
# wpK latency reproduction: K1 no-symbol read, K2 embedding phase, K3 scoped search.
#
# Handover (2026-10-06) before this branch:
#   K1  greppy read normalizeProfileRole          >183s then "no symbol"
#   K2  index status counting_embeddings 0/0      stalled 225s; graph reads waited
#   K3  search-symbol detect_blocks --path ...    >=177s then exit 75
#
# After the fix, run on the build lane against a prepared medium repo:
#   GREPPY_BIN=/path/to/greppy \
#   REPO=/mnt/nvme1/build-lane/scratch/wpK/tokio \
#   bench/latency/wpk-latency.sh
#
# The script prints one timing line per command. It does not start an index
# of its own; point REPO at a root that is already prepared.

set -uo pipefail

echo "before:"
echo "  K1 read no-symbol: >183s (pread, no background job)"
echo "  K2 counting_embeddings: 0/0 eta=null stalled=225s"
echo "  K3 scoped search-symbol: >=177s then exit 75"

if [[ -z "${GREPPY_BIN:-}" || -z "${REPO:-}" ]]; then
  echo "after: not run (set GREPPY_BIN and REPO)"
  exit 0
fi
if [[ ! -x "$GREPPY_BIN" ]]; then
  echo "after: not run ($GREPPY_BIN is not executable)"
  exit 0
fi

time_cmd() {
  local label=$1
  shift
  local start end code
  start=$(date +%s)
  "$@"
  code=$?
  end=$(date +%s)
  echo "after: ${label} exit=${code} seconds=$((end - start))"
}

time_cmd "K1 read no-symbol" \
  "$GREPPY_BIN" --root "$REPO" read normalizeProfileRole
time_cmd "K2 index status" \
  "$GREPPY_BIN" --root "$REPO" index status
time_cmd "K3 scoped search-symbol" \
  "$GREPPY_BIN" --root "$REPO" search-symbol detect_blocks --code --path crates/cli/src/bash_smart.rs
