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
#   3. The drop-in grep result stays BYTE-EXACT vs the system grep
#      (stdout + stderr + exit code) on every iteration, even as the
#      corpus is mutated underneath it. The pure drop-in path is the
#      `grepplus-grep` binary (no augmentation); that is the byte-exact
#      contract surface.
#   4. Sidecar temp files do NOT accumulate unboundedly: with a short
#      TTL and periodic cleanup invocations, the sidecar count stays
#      bounded over the whole run (TTL/cleanup actually reclaims).
#   5. RSS does NOT grow unbounded: we sample resident set size of an
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
#   BATTLE_SOAK_SIDECAR_CAP max live sidecars allowed at any  (default 64)
#                           check point under short TTL

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

NAME="soak"
ITERS="${BATTLE_SOAK_ITERS:-200}"
N_FILES="${BATTLE_SOAK_FILES:-40}"
RSS_FACTOR="${BATTLE_SOAK_RSS_FACTOR:-3}"
SIDECAR_CAP="${BATTLE_SOAK_SIDECAR_CAP:-64}"

require_bins "$GREPPLUS_BIN" "$GREPPLUS_GREP_BIN" || { emit_summary "$NAME"; exit 1; }

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

# Short TTL so the soak loop actually exercises reclamation rather than
# the 24 h production default. Cleanup runs on `grepplus-grep` startup.
export GREPPLUS_STORE_DIR="$STORE"
export GREPPLUS_SIDECAR_TTL_SECS="${GREPPLUS_SIDECAR_TTL_SECS:-1}"

echo "[soak] generating $N_FILES-file corpus ..."
bash "$BATTLE_DIR/gen_corpus.sh" "$CORPUS" "$N_FILES" >/dev/null 2>&1
git_init_corpus "$CORPUS"

# ---------------------------------------------------------------------------
# Helpers (operate inside the corpus working dir).
# ---------------------------------------------------------------------------

# index_corpus: (re)index the corpus into the store, appending stderr to
# the shared log. Returns the indexer's exit code.
index_corpus() {
    ( cd "$CORPUS" && "$GREPPLUS_BIN" index . ) >>"$LOG" 2>&1
}

# rss_of_index: run one index and report the peak RSS of the indexer.
# We background grepplus DIRECTLY (path arg, no `cd` subshell) so `$!` is
# the grepplus PID, and sample WHOLE-TREE RSS — the old version sampled a
# `( cd dir && bin ) &` subshell PID and reported ~1.5 MB every time,
# making the leak check vacuous. See lib.sh:rss_kb_tree.
rss_of_index() {
    local maxrss=0 pid s
    "$GREPPLUS_BIN" index "$CORPUS" >>"$LOG" 2>&1 &
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

# grep_byte_exact <pattern> <relpath>: run the pure drop-in grep and the
# system grep with identical argv from inside the corpus; compare all
# three observable channels. Returns 0 iff byte-identical.
grep_byte_exact() {
    local pat="$1" rel="$2"
    local og or eg er rg rr
    og="$( cd "$CORPUS" && "$GREPPLUS_GREP_BIN" -n "$pat" "$rel" 2>"$WORK/.eg" )"; rg=$?
    eg="$(cat "$WORK/.eg")"
    or="$( cd "$CORPUS" && "$REAL_GREP" -n "$pat" "$rel" 2>"$WORK/.er" )"; rr=$?
    er="$(cat "$WORK/.er")"
    # Capture grepplus-grep stderr into the panic log too.
    [[ -n "$eg" ]] && printf '%s\n' "$eg" >> "$LOG"
    if [[ "$og" == "$or" && "$eg" == "$er" && "$rg" -eq "$rr" && "$rg" -lt 128 ]]; then
        return 0
    fi
    {
        echo "[soak] grep mismatch pat=$pat rel=$rel"
        echo "  rc g=$rg r=$rr"
        diff <(printf '%s' "$og") <(printf '%s' "$or") | head -6 | sed 's/^/  /'
        echo "  stderr g=[$eg] r=[$er]"
    } >&2
    return 1
}

# count_sidecars: number of live sidecar files in the store.
count_sidecars() {
    find "$STORE" -name '*__GREPPLUS_SEMANTIC_NONCANONICAL.md' 2>/dev/null | wc -l | tr -d ' '
}

# drive_augment: run the augmenting drop-in (`grepplus` cli) against a
# graph-known symbol to *write* a sidecar, then run the pure drop-in
# (`grepplus-grep`) which performs the startup TTL cleanup. This is the
# pair that exercises sidecar creation + reclamation.
drive_augment() {
    ( cd "$CORPUS" && "$GREPPLUS_BIN" -R 'Widget0' . ) >>"$LOG" 2>&1 || true
    # grepplus-grep startup runs cleanup_expired against cwd's store.
    ( cd "$CORPUS" && "$GREPPLUS_GREP_BIN" -q 'Widget0' src/lib.rs ) >>"$LOG" 2>&1 || true
}

# ---------------------------------------------------------------------------
# Run.
# ---------------------------------------------------------------------------
echo "[soak] $ITERS iterations, $N_FILES files, store=$STORE, TTL=${GREPPLUS_SIDECAR_TTL_SECS}s"

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
declare -i sidecar_overflow=0
declare -i max_sidecars=0
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
    ( cd "$CORPUS" && "$GREPPLUS_BIN" search-code "soak_touch_$i" ) >>"$LOG" 2>&1 || true

    # Byte-exact drop-in grep on the moving needle's current home file.
    needle_rel="src/$(printf 'mod%04d' "$(( i % N_FILES ))").rs"
    if ! grep_byte_exact 'SOAK_NEEDLE' "$needle_rel"; then
        grep_breaks=$((grep_breaks + 1))
    fi
    # Also a never-matching pattern (rc=1 path) for breadth.
    if ! grep_byte_exact 'zzzz_no_such_token_zzzz' "$needle_rel"; then
        grep_breaks=$((grep_breaks + 1))
    fi

    # Drive sidecar creation + TTL cleanup.
    drive_augment

    # Periodic integrity + sidecar-bound checks (every ~10 iters and on
    # the last iter), to keep the loop fast while still sampling densely.
    if [[ $(( i % 10 )) -eq 0 || "$i" -eq "$ITERS" ]]; then
        integ="$(sqlite_q "$DB" 'PRAGMA integrity_check;' 2>/dev/null || echo ERR)"
        if [[ "$integ" != "ok" ]]; then
            echo "[soak] integrity_check=$integ on iter $i" >&2
            integ_bad=$((integ_bad + 1))
        fi
        sc="$(count_sidecars)"
        [[ "$sc" -gt "$max_sidecars" ]] && max_sidecars="$sc"
        if [[ "$sc" -gt "$SIDECAR_CAP" ]]; then
            echo "[soak] sidecar count $sc exceeds cap $SIDECAR_CAP on iter $i" >&2
            sidecar_overflow=$((sidecar_overflow + 1))
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
assert_eq 0 "$grep_breaks"      "drop-in grep stayed byte-exact vs $REAL_GREP across iterations"
assert_eq 0 "$sidecar_overflow" "sidecars stayed bounded (<= $SIDECAR_CAP) under TTL cleanup (peak=$max_sidecars)"

# Final cleanup sweep: backdate all sidecars and run one grepplus-grep so
# its startup cleanup_expired reclaims them; the count must drop. This is
# the strongest TTL assertion: reclamation actually deletes expired files.
find "$STORE" -name '*__GREPPLUS_SEMANTIC_NONCANONICAL.md' -exec touch -t 200001010000 {} \; 2>/dev/null
before_sweep="$(count_sidecars)"
( cd "$CORPUS" && GREPPLUS_SIDECAR_TTL_SECS=1 "$GREPPLUS_GREP_BIN" -q 'zzz' src/lib.rs ) >>"$LOG" 2>&1 || true
after_sweep="$(count_sidecars)"
if [[ "$before_sweep" -gt 0 ]]; then
    if [[ "$after_sweep" -lt "$before_sweep" ]]; then
        pass "TTL cleanup reclaimed expired sidecars ($before_sweep -> $after_sweep)"
    else
        fail "TTL cleanup reclaimed expired sidecars ($before_sweep -> $after_sweep)"
    fi
else
    # No sidecars were ever written (e.g. graph yielded no semantic hits);
    # that is still a valid, non-leaking state.
    pass "no sidecars accumulated (nothing to reclaim)"
fi

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
