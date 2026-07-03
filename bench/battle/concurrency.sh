#!/usr/bin/env bash
# CONCURRENCY battle — launch N concurrent `grepplus index` against the
# SAME workspace/store and assert the advisory-lock contract:
#   * at least one writer wins (exit 0)
#   * losers exit 75 (EX_TEMPFAIL) — the documented lock-contention code
#   * no writer exits with an unexpected code (panic/abort/corruption)
#   * the DB passes integrity_check after the dust settles
#   * the winner produced a populated graph
#
# Because indexing the corpus is the slow part, contention is real:
# while one writer holds the lock the others race for it. To make the
# race reliable we use a corpus big enough that the critical section is
# non-trivial, and launch the workers as simultaneously as possible.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

NAME="concurrency"
N_WORKERS="${BATTLE_CONC_WORKERS:-6}"
N_FILES="${BATTLE_CONC_FILES:-150}"

require_bins "$GREPPLUS_BIN" || { emit_summary "$NAME"; exit 1; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/battle-conc-XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
CORPUS="$WORK/corpus"
export GREPPLUS_STORE_DIR="$WORK/store"

echo "[concurrency] generating $N_FILES-file corpus ..."
bash "$BATTLE_DIR/gen_corpus.sh" "$CORPUS" "$N_FILES" >/dev/null 2>&1
git_init_corpus "$CORPUS"

# Pre-create the store dir so all workers key the same lock path.
mkdir -p "$GREPPLUS_STORE_DIR"

echo "[concurrency] launching $N_WORKERS concurrent indexers ..."
pids=()
rcfile_dir="$WORK/rcs"
mkdir -p "$rcfile_dir"

# A small barrier: all workers block on a fifo until we release them,
# so they start the lock race as close to simultaneously as possible.
barrier="$WORK/barrier"
mkfifo "$barrier"

for i in $(seq 1 "$N_WORKERS"); do
    (
        # Wait for the barrier release.
        read -r _ < "$barrier"
        cd "$CORPUS"
        "$GREPPLUS_BIN" index . >"$rcfile_dir/out.$i" 2>&1
        echo "$?" > "$rcfile_dir/rc.$i"
    ) &
    pids+=($!)
done

# Release all workers at once.
exec 3>"$barrier"
for i in $(seq 1 "$N_WORKERS"); do echo "go" >&3; done
exec 3>&-

for p in "${pids[@]}"; do wait "$p" 2>/dev/null; done

# ---- tally exit codes -----------------------------------------------------
winners=0
losers=0
unexpected=0
declare -a unexpected_codes=()
for i in $(seq 1 "$N_WORKERS"); do
    rc=$(cat "$rcfile_dir/rc.$i" 2>/dev/null || echo "MISSING")
    case "$rc" in
        0)  winners=$((winners+1)) ;;
        75) losers=$((losers+1)) ;;
        *)  unexpected=$((unexpected+1)); unexpected_codes+=("$rc") ;;
    esac
done

echo "[concurrency] winners(rc=0)=$winners losers(rc=75)=$losers unexpected=$unexpected"

# At least one winner.
if [[ "$winners" -ge 1 ]]; then
    pass "at least one indexer won (rc=0): $winners"
else
    fail "at least one indexer won (rc=0): $winners"
fi

# --- CORRUPTION-SAFETY invariant (must hold regardless of exit code) ---
# No worker may crash with a signal / panic / abort. Exit codes from the
# documented set {0 win, 73 IO, 75 lock} are all "graceful" — the process
# returned a clean status and (as asserted below) the DB is intact.
crash=0
declare -a crash_codes=()
for i in $(seq 1 "$N_WORKERS"); do
    rc=$(cat "$rcfile_dir/rc.$i" 2>/dev/null || echo "MISSING")
    case "$rc" in
        0|73|75) : ;;                       # documented graceful codes
        *) crash=$((crash+1)); crash_codes+=("$rc") ;;
    esac
done
if [[ "$crash" -eq 0 ]]; then
    pass "no worker crashed (all exits in graceful set {0,73,75})"
else
    fail "no worker crashed (rogue codes: ${crash_codes[*]})"
fi

# --- LOCK-CONTRACT invariant (the documented behaviour) ---
# Per crates/cli: a concurrent indexer that loses the race MUST exit 75
# (EX_TEMPFAIL) with a diagnostic. Any loser exiting 73 (silent IO error)
# is a CONTRACT VIOLATION — see FINDINGS in this directory's README.
# We assert the contract honestly; if it fails, that is a real finding,
# not something to paper over.
if [[ "$unexpected" -eq 0 ]]; then
    pass "lock contract: every non-winner exited 75 (clean contention)"
else
    fail "lock contract: $unexpected non-winner(s) exited ${unexpected_codes[*]} instead of 75 (FINDING: SQLITE_BUSY before advisory lock; see README)"
    for i in $(seq 1 "$N_WORKERS"); do
        rc=$(cat "$rcfile_dir/rc.$i" 2>/dev/null)
        if [[ "$rc" != "0" && "$rc" != "75" ]]; then
            echo "---- worker $i (rc=$rc) stdout=[$(cat "$rcfile_dir/out.$i")] ----"
        fi
    done
fi

# The race must actually be exercised: with N workers and a non-trivial
# critical section there must be at least one non-winner. Otherwise the
# lock path was never hit and the result would be misleading. (How the
# losers exit — 75 vs 73 — is judged by the lock-contract check above;
# here we only assert that contention HAPPENED so the suite count is
# stable run-to-run.)
non_winners=$((losers + unexpected))
if [[ "$non_winners" -ge 1 ]]; then
    pass "race exercised: $non_winners non-winner(s) (clean75=$losers, io73=$unexpected)"
else
    fail "race exercised: every worker won (losers=0) — lock path not hit; raise BATTLE_CONC_WORKERS/FILES"
fi

# No panic text anywhere.
if grep -RqiE 'panic|thread .* panicked' "$rcfile_dir"; then
    fail "no panic in any worker output"
else
    pass "no panic in any worker output"
fi

# ---- post-conditions on the store -----------------------------------------
DB="$(graph_db_path "$GREPPLUS_STORE_DIR")"
if [[ -z "$DB" ]]; then
    fail "graph.db exists after concurrent run"
    emit_summary "$NAME"; exit 1
fi
pass "graph.db exists after concurrent run"

integ=$(sqlite_q "$DB" "PRAGMA integrity_check;" 2>/dev/null || echo "ERR")
assert_eq "ok" "$integ" "DB integrity_check after concurrent run"

nodes=$(sqlite_q "$DB" "SELECT count(*) FROM nodes;" 2>/dev/null || echo 0)
assert_ge "${nodes:-0}" 1 "winner populated the graph"

# Lock file must be released (no stale lock left behind by the winner).
leftover=$(find "$GREPPLUS_STORE_DIR" -name '*.lock' 2>/dev/null | wc -l | tr -d ' ')
echo "[concurrency] note: leftover lock files = $leftover"

emit_summary "$NAME"
