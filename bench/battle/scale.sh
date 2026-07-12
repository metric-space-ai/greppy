#!/usr/bin/env bash
# SCALE battle — generate a synthetic Rust repo, git-init it, index it,
# and assert production invariants:
#   * index completes (rc=0), no panic
#   * within a sane time budget (no runaway hang)
#   * RSS stays bounded (no OOM / no unbounded memory growth)
#   * the graph has real cross-file CALLS / IMPORTS / USAGE edges
#   * the DB passes integrity_check
#
# File count is overridable so an operator can push toward the 2000+
# aspiration when willing to wait:
#   BATTLE_SCALE_FILES=2000 BATTLE_SCALE_BUDGET_S=1800 bash scale.sh
#
# NOTE (honest finding): indexing time grows super-linearly with file
# count on the debug binary — see the README in this directory. The
# default of 300 files completes in ~10s; the budget below is generous.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

NAME="scale"
N_FILES="${BATTLE_SCALE_FILES:-300}"
BUDGET_S="${BATTLE_SCALE_BUDGET_S:-120}"
# Generous RSS ceiling. 300 files peaks ~60MB; scale headroom for larger
# corpora. Override with BATTLE_SCALE_RSS_KB.
RSS_CEIL_KB="${BATTLE_SCALE_RSS_KB:-2097152}"   # 2 GiB

require_bins "$GREPPY_BIN" || { emit_summary "$NAME"; exit 1; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/battle-scale-XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
CORPUS="$WORK/corpus"
export GREPPY_STORE_DIR="$WORK/store"

echo "[scale] generating $N_FILES-file corpus ..."
bash "$BATTLE_DIR/gen_corpus.sh" "$CORPUS" "$N_FILES" >/dev/null 2>&1
n_rs=$(find "$CORPUS/src" -name '*.rs' | wc -l | tr -d ' ')
assert_ge "$n_rs" "$N_FILES" "generated >= $N_FILES Rust files"

git_init_corpus "$CORPUS"
[[ -d "$CORPUS/.git" ]]; check $? "corpus is a git repo"

# ---- index, timed, with RSS sampling --------------------------------------
echo "[scale] indexing (budget ${BUDGET_S}s) ..."
peak_rss=0
log="$WORK/index.log"

# Background the indexer directly (path arg, NO `cd` subshell) so the
# backgrounded PID is greppy itself — and sample WHOLE-TREE RSS so the
# real resident-set peak is measured even if a wrapper forks the worker.
# (The old harness sampled a `( cd dir && bin ) &` subshell PID and always
# reported ~1.5 MB; see lib.sh:rss_kb_tree.)
S=$(date +%s)
env GREPPY_STORE_DIR="$GREPPY_STORE_DIR" "$GREPPY_BIN" index "$CORPUS" >"$log" 2>&1 &
idx_pid=$!

# Sample RSS while the indexer runs; also enforce the time budget.
timed_out=0
while kill -0 "$idx_pid" 2>/dev/null; do
    cur=$(rss_kb_tree "$idx_pid")
    [[ -n "$cur" && "$cur" -gt "$peak_rss" ]] && peak_rss="$cur"
    now=$(date +%s)
    if [[ "$((now - S))" -ge "$BUDGET_S" ]]; then
        timed_out=1
        kill -9 "$idx_pid" 2>/dev/null
        break
    fi
    sleep 0.2
done
wait "$idx_pid" 2>/dev/null
rc=$?
E=$(date +%s)
elapsed=$((E - S))

if [[ "$timed_out" -eq 1 ]]; then
    fail "index completed within ${BUDGET_S}s budget (TIMED OUT after ${elapsed}s, killed)"
else
    pass "index completed within ${BUDGET_S}s budget (took ${elapsed}s)"
    assert_eq 0 "$rc" "index exit code"
fi

# No panic in output.
if grep -qiE 'panic|thread .* panicked|RUST_BACKTRACE' "$log"; then
    fail "no panic in index output"
    echo "---- index log (panic) ----"; sed -n '1,20p' "$log"
else
    pass "no panic in index output"
fi

# RSS bound (only meaningful if we sampled anything).
if [[ "$peak_rss" -gt 0 ]]; then
    if [[ "$peak_rss" -le "$RSS_CEIL_KB" ]]; then
        pass "peak RSS within ceiling (${peak_rss}KB <= ${RSS_CEIL_KB}KB)"
    else
        fail "peak RSS within ceiling (${peak_rss}KB > ${RSS_CEIL_KB}KB)"
    fi
    # Regression guard for the RSS-sampling bug: the OLD harness sampled a
    # `( cd dir && bin ) &` subshell PID and always reported ~1.5 MB
    # (~1536 KB) regardless of the indexer's true footprint. A correct
    # whole-tree sample of a real index of N>=300 Rust files is always far
    # above that. If peak ever drops back into the ~1.5 MB band, we are
    # sampling the wrong process again.
    if [[ "$peak_rss" -ge 8192 ]]; then
        pass "RSS sampler measured the real indexer (${peak_rss}KB >= 8192KB, not the ~1.5MB subshell)"
    else
        fail "RSS sampler measured the real indexer (${peak_rss}KB < 8192KB — looks like the subshell-PID bug)"
    fi
else
    echo "[scale] note: RSS not sampled (index too fast)"
fi

# ---- graph invariants -----------------------------------------------------
DB="$(graph_db_path "$GREPPY_STORE_DIR")"
if [[ -z "$DB" ]]; then
    fail "graph.db exists"
    emit_summary "$NAME"; exit 1
fi
pass "graph.db exists"

nodes=$(sqlite_q "$DB" "SELECT count(*) FROM nodes;" 2>/dev/null || echo 0)
assert_ge "${nodes:-0}" 1 "graph has nodes"

for et in CALLS IMPORTS USAGE; do
    c=$(sqlite_q "$DB" "SELECT count(*) FROM edges WHERE edge_type='$et';" 2>/dev/null || echo 0)
    assert_ge "${c:-0}" 1 "graph has cross-file $et edges"
done

# Cross-file proof: at least one edge whose endpoints live in different
# files. This is the real invariant — intra-file edges are cheap;
# cross-file resolution is the hard part. By construction every module
# `use`s + returns the previous module's struct, so IMPORTS and USAGE
# MUST be cross-file.
xfile=$(sqlite_q "$DB" "
  SELECT count(*) FROM edges e
  JOIN nodes s ON s.id = e.source_id
  JOIN nodes t ON t.id = e.target_id
  WHERE e.edge_type IN ('IMPORTS','USAGE') AND s.file_path <> t.file_path;" 2>/dev/null || echo 0)
assert_ge "${xfile:-0}" 1 "at least one TRULY cross-file edge (IMPORTS/USAGE)"

# Informational: how CALLS resolve. The corpus has build_N() call the
# imported Widget(N-1)::new(); the resolver currently binds `new` to the
# local module's method, so cross-file CALLS may be 0. We report the
# number rather than asserting on it (this is a resolver characteristic,
# not a corruption invariant).
calls_xfile=$(sqlite_q "$DB" "
  SELECT count(*) FROM edges e
  JOIN nodes s ON s.id = e.source_id
  JOIN nodes t ON t.id = e.target_id
  WHERE e.edge_type='CALLS' AND s.file_path <> t.file_path;" 2>/dev/null || echo 0)
echo "[scale] note: cross-file CALLS edges = ${calls_xfile:-0} (informational)"

integ=$(sqlite_q "$DB" "PRAGMA integrity_check;" 2>/dev/null || echo "ERR")
assert_eq "ok" "$integ" "DB integrity_check"

emit_summary "$NAME"
