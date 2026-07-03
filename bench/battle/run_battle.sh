#!/usr/bin/env bash
# run_battle.sh — aggregate the battle-proof validation suite.
#
# Black-box harness (Track C): drives the already-built grepplus binaries
# and asserts production invariants. Touches NO crate source.
#
# Runs each focused battle script, parses its BATTLE_SUMMARY line, prints
# a combined PASS/FAIL summary, and exits NON-ZERO if any check failed.
#
# Usage:
#   bash bench/battle/run_battle.sh            # run all
#   bash bench/battle/run_battle.sh scale      # run a subset by name
#
# Env knobs (see each script): BATTLE_SCALE_FILES, BATTLE_SCALE_BUDGET_S,
#   BATTLE_DET_FILES, BATTLE_CONC_WORKERS, BATTLE_CONC_FILES, REAL_GREP.

set -uo pipefail

BATTLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$BATTLE_DIR/../.." && pwd)}"
export WORKSPACE_ROOT

GREPPLUS_BIN="${GREPPLUS_BIN:-$WORKSPACE_ROOT/target/debug/grepplus}"
GREPPLUS_GREP_BIN="${GREPPLUS_GREP_BIN:-$WORKSPACE_ROOT/target/debug/grepplus-grep}"

# Auto-build if binaries are missing (best-effort; report if it fails).
if [[ ! -x "$GREPPLUS_BIN" || ! -x "$GREPPLUS_GREP_BIN" ]]; then
    echo "[run_battle] binaries missing; running 'cargo build --bins' ..."
    if ! ( cd "$WORKSPACE_ROOT" && cargo build --bins ); then
        echo "[run_battle] cargo build --bins FAILED" >&2
        exit 1
    fi
fi

# Optional release-build invariant (BATTLE_RELEASE=1). Release builds are
# slow, so this is opt-in. It asserts the release binaries (a) build and
# (b) honour the byte-exact drop-in contract on a representative query —
# the same property the debug suite checks, but on the optimised build
# that actually ships. Failure here is counted in the combined summary.
declare -i rel_pass=0
declare -i rel_fail=0
if [[ "${BATTLE_RELEASE:-0}" == "1" ]]; then
    echo ""
    echo "================ release-invariant ================"
    if ( cd "$WORKSPACE_ROOT" && cargo build --release --bins ); then
        echo "PASS release binaries build"
        rel_pass=$((rel_pass + 1))
        rel_grep="$WORKSPACE_ROOT/target/release/grepplus-grep"
        real_grep="${REAL_GREP:-/usr/bin/grep}"
        [[ -x "$real_grep" ]] || real_grep="$(command -v grep)"
        rel_tmp="$(mktemp -d "${TMPDIR:-/tmp}/battle-release-XXXXXX")"
        printf 'foo\nbar\nFOObar\nbaz foo qux\n' > "$rel_tmp/plain.txt"
        og="$("$rel_grep" -n foo "$rel_tmp/plain.txt" 2>/dev/null)"; rcg=$?
        or="$("$real_grep" -n foo "$rel_tmp/plain.txt" 2>/dev/null)"; rcr=$?
        if [[ "$og" == "$or" && "$rcg" -eq "$rcr" && "$rcg" -lt 128 ]]; then
            echo "PASS release drop-in grep byte-exact vs $real_grep"
            rel_pass=$((rel_pass + 1))
        else
            echo "FAIL release drop-in grep byte-exact vs $real_grep (rc $rcg vs $rcr)"
            rel_fail=$((rel_fail + 1))
        fi
        rm -rf "$rel_tmp"
    else
        echo "FAIL release binaries build"
        rel_fail=$((rel_fail + 1))
    fi
fi

ALL_SCRIPTS=(scale determinism concurrency navigation multilang grep_fuzz malformed)

# soak.sh is a long-running stress loop; it is opt-in so the default
# suite stays fast. Enable with BATTLE_SOAK=1 (or name it explicitly).
if [[ "${BATTLE_SOAK:-0}" == "1" ]]; then
    ALL_SCRIPTS+=(soak)
fi

# Optional positional filter: run only the named subset.
if [[ "$#" -gt 0 ]]; then
    SCRIPTS=("$@")
else
    SCRIPTS=("${ALL_SCRIPTS[@]}")
fi

echo "==============================================="
echo "  grepplus BATTLE suite"
echo "  binary: $GREPPLUS_BIN"
echo "==============================================="

declare -a results
total_pass=0
total_fail=0
overall_rc=0

for name in "${SCRIPTS[@]}"; do
    script="$BATTLE_DIR/$name.sh"
    if [[ ! -f "$script" ]]; then
        echo "[run_battle] unknown battle: $name" >&2
        results+=("$name MISSING")
        overall_rc=1
        continue
    fi

    echo ""
    echo "================ $name ================"
    out="$(mktemp)"
    bash "$script" 2>&1 | tee "$out"
    rc="${PIPESTATUS[0]}"

    # Parse the machine-readable summary line.
    summary_line="$(grep -E '^BATTLE_SUMMARY ' "$out" | tail -n1)"
    p="$(sed -n 's/.*pass=\([0-9]*\).*/\1/p' <<<"$summary_line")"
    f="$(sed -n 's/.*fail=\([0-9]*\).*/\1/p' <<<"$summary_line")"
    : "${p:=0}"; : "${f:=0}"

    # If the script exited non-zero but emitted no summary, treat as a
    # hard failure (so a crashed script can't silently pass).
    if [[ -z "$summary_line" && "$rc" -ne 0 ]]; then
        f=$((f + 1))
    fi

    results+=("$name pass=$p fail=$f rc=$rc")
    total_pass=$((total_pass + p))
    total_fail=$((total_fail + f))
    [[ "$rc" -ne 0 || "$f" -ne 0 ]] && overall_rc=1
    rm -f "$out"
done

# Fold the optional release-invariant results into the totals.
if [[ "${BATTLE_RELEASE:-0}" == "1" ]]; then
    results+=("release-invariant pass=$rel_pass fail=$rel_fail")
    total_pass=$((total_pass + rel_pass))
    total_fail=$((total_fail + rel_fail))
    [[ "$rel_fail" -ne 0 ]] && overall_rc=1
fi

echo ""
echo "==============================================="
echo "  BATTLE combined summary"
echo "==============================================="
for r in "${results[@]}"; do
    echo "  $r"
done
echo ""
echo "  total PASS: $total_pass"
echo "  total FAIL: $total_fail"
echo ""
if [[ "$overall_rc" -eq 0 && "$total_fail" -eq 0 ]]; then
    echo "  RESULT: ALL GREEN"
else
    echo "  RESULT: FAILURES PRESENT (see FAIL lines above and FINDINGS in README.md)"
fi
echo "==============================================="

exit "$overall_rc"
