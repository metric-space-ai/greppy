#!/usr/bin/env bash
# Phase 7 — contract-level invocations (real grepplus-grep path).
#
# R-009 / WP-R008 / RV-004 — each entry is a failing test that PASSES
# only when grepplus-grep behaves correctly. The legacy smoke tests
# were rewritten here as assertions:
#
#   - byte-exact real-grep passthrough on Strict / Sidecar / miss,
#   - no visible augmentation when real grep returned a non-match,
#   - no DB pollution (`<root>/.grepplus/graph.db` absent),
#   - real-grep rc=2 does not synthesise a sidecar,
#   - Sidecar path lives under $GREPPLUS_STORE_DIR, not under /tmp,
#   - the freshness Strict gate keeps the wrapper Strict (default
#     behaviour — no VISIBLE_AUGMENT for -q even on a fresh graph).

set -uo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

GREPPLUS_BIN="${GREPPLUS_BIN:-$WORKSPACE_ROOT/target/debug/grepplus-grep}"
INDEX_BIN="${INDEX_BIN:-$WORKSPACE_ROOT/target/debug/grepplus}"
FIXTURE_SRC="$WORKSPACE_ROOT/bench/fixtures/sample"

if [[ ! -x "$GREPPLUS_BIN" ]]; then
    echo "grep_compat.sh: build first:"
    echo "  cargo build -p grepplus --bin grepplus-grep"
    pass=0; fail=1
    echo "pass: $pass / fail: $fail"
    exit 1
fi

declare -i pass=0 fail=0
TMP_BASE="${TMPDIR:-/tmp}/grepplus-grep-compat-$$"
mkdir -p "$TMP_BASE"

WORK=$(mktemp -d -p "$TMP_BASE")
trap 'rm -rf "$WORK" "$TMP_BASE"' EXIT

cp -R "$FIXTURE_SRC" "$WORK/repo"
rm -rf "$WORK/repo/.grepplus"
mkdir -p "$WORK/store"
export GREPPLUS_STORE_DIR="$WORK/store"

"$INDEX_BIN" index "$WORK/repo" >/dev/null 2>&1

REAL_GREP=$(command -v grep)
if [[ -z "$REAL_GREP" ]]; then
    REAL_GREP=/usr/bin/grep
fi

# Run real grep + wrapper from $WORK so any relative paths in the
# arguments resolve against the same root. The wrapper uses cwd for
# the freshness gate, so we keep cwd == repo root.
cd "$WORK/repo"

# Run the wrapper with `argv`. Returns exit code on stdout? No: capture
# into globals to keep the helper readable.
#   $actual    : wrapper stdout
#   $rc        : wrapper exit code
#   $expected  : real-grep stdout
#   $expected_rc_actual : real-grep exit code
run_pair() {
    local label="$1"
    shift
    local args=( "$@" )
    actual=$("$GREPPLUS_BIN" "${args[@]}" 2>&1)
    rc=$?
    expected=$("$REAL_GREP" "${args[@]}" 2>&1)
    expected_rc_actual=$?
}

assert_byte_for_byte() {
    local label="$1"
    local mode="$2"   # ok | miss | err
    shift 2
    local args=( "$@" )

    run_pair "$label" "${args[@]}"

    # Compare wrapper output vs real-grep output byte-for-byte.
    if [[ "$actual" != "$expected" ]]; then
        echo "FAIL $label (output diverges; rc actual=$rc expected=$expected_rc_actual)"
        echo "  expected: <<<$expected>>>"
        echo "  actual:   <<<$actual>>>"
        fail=$((fail+1))
        return
    fi
    if [[ "$rc" != "$expected_rc_actual" ]]; then
        echo "FAIL $label (rc diverges; actual=$rc expected=$expected_rc_actual)"
        fail=$((fail+1))
        return
    fi

    # R-002 / WP-R003: real-grep miss must be byte-exact empty and
    # contain no synthetic line. real-grep rc=2 must not synthesise
    # a sidecar or sentinel either.
    case "$mode" in
        miss)
            if [[ -n "$actual" ]]; then
                echo "FAIL $label (visible output on miss; expected empty: <<<$actual>>>)"
                fail=$((fail+1))
                return
            fi
            if grep -q GREPPLUS_NON_CANONICAL_HIT <<<"$actual"; then
                echo "FAIL $label (synthetic sentinel on miss)"
                fail=$((fail+1))
                return
            fi
            ;;
        err)
            if grep -q GREPPLUS_NON_CANONICAL_HIT <<<"$actual"; then
                echo "FAIL $label (synthetic sentinel on rc=2)"
                fail=$((fail+1))
                return
            fi
            ;;
    esac

    echo "PASS $label"
    pass=$((pass+1))
}

assert_no_db_pollution() {
    local repo="$WORK/repo"
    if [[ -e "$repo/.grepplus/graph.db" ]]; then
        echo "FAIL no-db-pollution — .grepplus/graph.db exists inside repo (R-005)"
        fail=$((fail+1))
    else
        local count
        count=$(find "$WORK/store" -type f -name 'graph.db' | wc -l | tr -d ' ')
        if [[ "$count" -ge 1 ]]; then
            echo "PASS no-db-pollution — DB under $WORK/store (R-005)"
            pass=$((pass+1))
        else
            echo "FAIL no-db-pollution — no DB anywhere"
            fail=$((fail+1))
        fi
    fi
}

assert_no_sidecar_on_miss() {
    # R-002: a fresh real-grep miss must produce NO sidecar file
    # anywhere under the configured store dir.
    local label="$1"
    shift
    local args=( "$@" )
    local before after
    before=$(find "$WORK/store" -type f -name '*__GREPPLUS_SEMANTIC_NONCANONICAL.md' 2>/dev/null | wc -l | tr -d ' ')
    "$GREPPLUS_BIN" "${args[@]}" >/dev/null 2>&1 || true
    after=$(find "$WORK/store" -type f -name '*__GREPPLUS_SEMANTIC_NONCANONICAL.md' 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$before" == "$after" ]]; then
        echo "PASS $label (no sidecar written on real-grep miss)"
        pass=$((pass+1))
    else
        echo "FAIL $label (sidecar count: $before -> $after on a real-grep miss)"
        fail=$((fail+1))
    fi
}

# --- Assertions ----------------------------------------------------------

# Real-grep miss path (byte-exact empty, rc=1).
assert_byte_for_byte "STRICT -q (real-grep miss)" miss -q nonexistent_token src
assert_byte_for_byte "STRICT -L (real-grep miss)" miss -L nonexistent_token src
assert_byte_for_byte "STRICT -v match (real-grep miss)" miss -v hello src
assert_byte_for_byte "STRICT -f empty-pattern (real-grep miss)" miss -f /dev/null src
assert_byte_for_byte "STDIN miss" miss -q nonexistent_token < /dev/null

# Real-grep match path — Strict class (pipeline-friendly flags).
assert_byte_for_byte "STRICT -q (match)" ok -q hello src
assert_byte_for_byte "STRICT -c (match)" ok -c hello src
assert_byte_for_byte "STRICT -l (match)" ok -l hello src
assert_byte_for_byte "STRICT -E match" ok -E 'fn .*\(\)' src
assert_byte_for_byte "STRICT -F literal" ok -F "fn hello" src
assert_byte_for_byte "STRICT --label" ok --label=foo -c hello src
assert_byte_for_byte "STRICT -n -H" ok -nH hello src

# Real-grep rc=2 path (e.g. unreadable dir). Augmentation must be
# silent — no sentinel, no sidecar.
assert_byte_for_byte "STRICT -R missing-dir (rc=2)" err -R anything /no/such/dir

# R-002: real-grep miss must not write a sidecar anywhere.
assert_no_sidecar_on_miss "R-002 no-sidecar-on-miss" -q nonexistent_token src

# R-005: no DB pollution.
assert_no_db_pollution

echo ""
echo "pass: $pass / fail: $fail"
[[ "$fail" -eq 0 ]]