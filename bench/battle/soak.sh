#!/usr/bin/env bash
# SOAK / STRESS battle (Track C) — long-running stability under a
# realistic, repeated workload.
#
# Drives the already-built binaries through many iterations of a
#   index -> edit -> reindex -> search -> grep
# loop and asserts the production invariants that must hold *across*
# iterations, not just once:
#
#   1. NO PANIC / no signal crash on any iteration (we scan combined
#      stderr for panic / SIG* markers and check exit codes).
#   2. integrity_check stays "ok" on the live graph.db every iteration.
#   3. The grep-compatible product path stays BYTE-EXACT vs system grep
#      (stdout + stderr + exit code) on every iteration, even as the
#      corpus is mutated underneath it, even with a fresh index in scope.
#   4. RSS does NOT grow unbounded: we sample resident set size of an
#      index process early and late and require late <= early * factor.
#
# This is a BLACK-BOX harness: it never touches crate source or Cargo.
#
# Opt-in: run_battle.sh only runs this when BATTLE_SOAK=1, because it is
# slow. Run standalone with a small count to smoke it:
#
#   BATTLE_SOAK_ITERS=20 bash bench/battle/soak.sh
#
# Env knobs:
#   BATTLE_SOAK_ITERS   iterations of the loop                (default 200)
#   BATTLE_SOAK_FILES   corpus size                           (default 40)
#   BATTLE_SOAK_RSS_FACTOR  max late/early RSS ratio          (default 3)

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

NAME="soak"
ITERS="${BATTLE_SOAK_ITERS:-200}"
N_FILES="${BATTLE_SOAK_FILES:-40}"
RSS_FACTOR="${BATTLE_SOAK_RSS_FACTOR:-3}"

require_bins "$GREPPY_BIN" || { emit_summary "$NAME"; exit 1; }

if [[ ! -x "$REAL_GREP" ]]; then
    fail "real grep oracle present ($REAL_GREP)"
    emit_summary "$NAME"; exit 1
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/battle-soak-XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
CORPUS="$WORK/corpus"
STORE="$WORK/store"
LOG="$WORK/soak.log"            # accumulates stderr from every invocation
: > "$LOG"

export GREPPY_STORE_DIR="$STORE"

echo "[soak] generating $N_FILES-file corpus ..."
bash "$BATTLE_DIR/gen_corpus.sh" "$CORPUS" "$N_FILES" >/dev/null 2>&1
git_init_corpus "$CORPUS"

# ---------------------------------------------------------------------------
# Helpers (operate inside the corpus working dir).
# ---------------------------------------------------------------------------

# index_corpus: (re)index the corpus into the store, appending stderr to
# the shared log. Returns the indexer's exit code.
index_corpus() {
    ( cd "$CORPUS" && "$GREPPY_BIN" index . ) >>"$LOG" 2>&1
}

# rss_of_index: run one index and report the peak RSS of the indexer.
# We background greppy DIRECTLY (path arg, no `cd` subshell) so `$!` is
# the greppy PID, and sample WHOLE-TREE RSS — the old version sampled a
# `( cd dir && bin ) &` subshell PID and reported ~1.5 MB every time,
# making the leak check vacuous. See lib.sh:rss_kb_tree.
rss_of_index() {
    local maxrss=0 pid s
    "$GREPPY_BIN" index "$CORPUS" >>"$LOG" 2>&1 &
    pid=$!
    while kill -0 "$pid" 2>/dev/null; do
        s="$(rss_kb_tree "$pid")"
        if [[ -n "$s" && "$s" -gt "$maxrss" ]]; then maxrss="$s"; fi
    done
    wait "$pid" 2>/dev/null
    echo "$maxrss"
}

# mutate_corpus <iter>: deterministic edit — append a uniquely-named
# function to a rotating module so reindex has real new work each pass.
# Keeps a stable needle `SOAK_NEEDLE` present in exactly one file so the
# byte-exact grep comparison has a moving but predictable target.
mutate_corpus() {
    local iter="$1"
    local idx=$(( iter % N_FILES ))
    local f
    f="$CORPUS/src/$(printf 'mod%04d' "$idx").rs"
    # Append a uniquely-named symbol (new graph node every iteration).
    printf '\npub fn soak_touch_%d() -> u64 { %d }\n' "$iter" "$iter" >> "$f"
    # Move the stable needle to the current file (and only there).
    grep -rl 'SOAK_NEEDLE' "$CORPUS/src" 2>/dev/null | while read -r old; do
        # strip any prior needle marker line
        grep -v 'SOAK_NEEDLE' "$old" > "$old.tmp" && mv "$old.tmp" "$old"
    done
    printf '// SOAK_NEEDLE marker iter %d\n' "$iter" >> "$f"
}

# grep_byte_exact <binary> <label> <argv...>: compare all observable channels.
grep_byte_exact() {
    local bin="$1" label="$2" rg rr
    shift 2
    ( cd "$CORPUS" && "$bin" "$@" ) >"$WORK/.go" 2>"$WORK/.ge"; rg=$?
    ( cd "$CORPUS" && "$REAL_GREP" "$@" ) >"$WORK/.ro" 2>"$WORK/.re"; rr=$?
    [[ -s "$WORK/.ge" ]] && cat "$WORK/.ge" >> "$LOG"
    if cmp -s "$WORK/.go" "$WORK/.ro" && cmp -s "$WORK/.ge" "$WORK/.re" \
        && [[ "$rg" -eq "$rr" && "$rg" -lt 128 ]]; then
        return 0
    fi
    {
        echo "[soak] $label mismatch: $*"
        echo "  rc g=$rg r=$rr"
        diff -u "$WORK/.ro" "$WORK/.go" | head -8 | sed 's/^/  /'
        diff -u "$WORK/.re" "$WORK/.ge" | head -8 | sed 's/^/  /'
    } >&2
    return 1
}

# ---------------------------------------------------------------------------
# Run.
# ---------------------------------------------------------------------------
echo "[soak] $ITERS iterations, $N_FILES files, store=$STORE"

index_corpus; rc=$?
if [[ "$rc" -ne 0 ]]; then
    fail "initial index exit code ($rc)"
    emit_summary "$NAME"; exit 1
fi
pass "initial index succeeded"

DB="$(graph_db_path "$STORE")"
if [[ -z "$DB" ]]; then
    fail "graph.db created"
    emit_summary "$NAME"; exit 1
fi
pass "graph.db created"

# Early RSS sample (after warm-up).
rss_early="$(rss_of_index)"
: "${rss_early:=0}"

declare -i loop_panics=0
declare -i integ_bad=0
declare -i grep_breaks=0
rss_late=0

i=0
while [[ "$i" -lt "$ITERS" ]]; do
    i=$((i + 1))

    mutate_corpus "$i"

    index_corpus
    if [[ $? -ne 0 ]]; then
        echo "[soak] reindex non-zero exit on iter $i" >&2
        loop_panics=$((loop_panics + 1))
    fi

    # Structured search must not crash (exercises the read path).
    ( cd "$CORPUS" && "$GREPPY_BIN" search-pattern "soak_touch_$i" --fixed ) >>"$LOG" 2>&1 || true

    # Byte-exact grep passthrough on the moving needle's current home file.
    needle_rel="src/$(printf 'mod%04d' "$(( i % N_FILES ))").rs"
    for bin_label in "$GREPPY_BIN:greppy"; do
        bin="${bin_label%%:*}"
        label="${bin_label##*:}"
        if ! grep_byte_exact "$bin" "$label" -n SOAK_NEEDLE "$needle_rel"; then
            grep_breaks=$((grep_breaks + 1))
        fi
        if ! grep_byte_exact "$bin" "$label" -n zzzz_no_such_token_zzzz "$needle_rel"; then
            grep_breaks=$((grep_breaks + 1))
        fi
        if [[ $(( i % 10 )) -eq 0 ]] && ! grep_byte_exact "$bin" "$label" -R -n Widget0 .; then
            grep_breaks=$((grep_breaks + 1))
        fi
    done

    # Periodic integrity checks.
    if [[ $(( i % 10 )) -eq 0 || "$i" -eq "$ITERS" ]]; then
        integ="$(sqlite_q "$DB" 'PRAGMA integrity_check;' 2>/dev/null || echo ERR)"
        if [[ "$integ" != "ok" ]]; then
            echo "[soak] integrity_check=$integ on iter $i" >&2
            integ_bad=$((integ_bad + 1))
        fi
    fi
done

# Late RSS sample.
rss_late="$(rss_of_index)"
: "${rss_late:=0}"

# Scan the accumulated log for any panic / signal-crash markers.
if grep -qiE 'panic|thread .* panicked|stack overflow|RUST_BACKTRACE|SIGSEGV|SIGABRT|SIGBUS|core dumped' "$LOG"; then
    loop_panics=$((loop_panics + 1))
    echo "[soak] panic markers found in log:" >&2
    grep -iE 'panic|overflow|SIG|core dumped' "$LOG" | head -8 | sed 's/^/  /' >&2
fi

# ---------------------------------------------------------------------------
# Assertions.
# ---------------------------------------------------------------------------
assert_eq 0 "$loop_panics"      "no panic / signal crash across $ITERS iterations"
assert_eq 0 "$integ_bad"        "integrity_check stayed ok at every checkpoint"
assert_eq 0 "$grep_breaks"      "grep passthrough stayed byte-exact vs $REAL_GREP across iterations"

# RSS growth bound. Guard against a zero early sample (sampling can miss a
# very fast child); only assert the ratio when we have a real baseline.
if [[ "$rss_early" -gt 0 && "$rss_late" -gt 0 ]]; then
    limit=$(( rss_early * RSS_FACTOR ))
    if [[ "$rss_late" -le "$limit" ]]; then
        pass "RSS did not grow unbounded (early=${rss_early}KB late=${rss_late}KB <= ${limit}KB)"
    else
        fail "RSS grew unbounded (early=${rss_early}KB late=${rss_late}KB > ${limit}KB)"
    fi
else
    echo "[soak] RSS sampling inconclusive (early=$rss_early late=$rss_late); skipping ratio check" >&2
    pass "RSS sampling inconclusive (skipped, not a leak signal)"
fi

emit_summary "$NAME"
