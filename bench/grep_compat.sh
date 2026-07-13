#!/usr/bin/env bash
# Contract-level invocations for the shipped Greppy passthrough path.
# Every invocation must preserve stdout, stderr, and the exit code exactly and
# must not create index, model, or sidecar state.

set -uo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

GREPPY_BIN="${GREPPY_BIN:-$WORKSPACE_ROOT/target/debug/greppy}"
FIXTURE_SRC="$WORKSPACE_ROOT/bench/fixtures/sample"

if [[ ! -x "$GREPPY_BIN" ]]; then
    echo "grep_compat.sh: build first:"
    echo "  cargo build --bin greppy"
    pass=0; fail=1
    echo "pass: $pass / fail: $fail"
    exit 1
fi

declare -i pass=0 fail=0
TMP_BASE="${TMPDIR:-/tmp}/greppy-compat-$$"
mkdir -p "$TMP_BASE"

WORK=$(mktemp -d -p "$TMP_BASE")
trap 'rm -rf "$WORK" "$TMP_BASE"' EXIT

cp -R "$FIXTURE_SRC" "$WORK/repo"
rm -rf "$WORK/repo/.greppy"
mkdir -p "$WORK/store"
export GREPPY_STORE_DIR="$WORK/store"

# Resolve system grep from fixed paths, mirroring the product's discover_grep.
# `command -v grep` is unsafe here: a greppy shim installed as `grep` on PATH
# (e.g. ~/.local/bin/grep) would recurse the comparison into a fork bomb.
REAL_GREP=""
for candidate in /usr/bin/grep /bin/grep; do
    [[ -x "$candidate" ]] && { REAL_GREP="$candidate"; break; }
done
if [[ -z "$REAL_GREP" ]]; then
    echo "no system grep at /usr/bin/grep or /bin/grep" >&2
    exit 1
fi

# Run real grep + wrapper from $WORK so any relative paths in the
# arguments resolve against the same root. The wrapper uses cwd for
# the freshness gate, so we keep cwd == repo root.
cd "$WORK/repo"

run_pair() {
    shift
    local args=( "$@" )
    "$GREPPY_BIN" "${args[@]}" >"$WORK/actual.out" 2>"$WORK/actual.err"
    rc=$?
    "$REAL_GREP" "${args[@]}" >"$WORK/expected.out" 2>"$WORK/expected.err"
    expected_rc_actual=$?
}

assert_byte_for_byte() {
    local label="$1"
    local mode="$2"   # ok | miss | err
    shift 2
    local args=( "$@" )

    run_pair "$label" "${args[@]}"

    if ! cmp -s "$WORK/actual.out" "$WORK/expected.out"; then
        echo "FAIL $label (stdout diverges; rc actual=$rc expected=$expected_rc_actual)"
        diff -u "$WORK/expected.out" "$WORK/actual.out" | head -20
        fail=$((fail+1))
        return
    fi
    if ! cmp -s "$WORK/actual.err" "$WORK/expected.err"; then
        echo "FAIL $label (stderr diverges; rc actual=$rc expected=$expected_rc_actual)"
        diff -u "$WORK/expected.err" "$WORK/actual.err" | head -20
        fail=$((fail+1))
        return
    fi
    if [[ "$rc" != "$expected_rc_actual" ]]; then
        echo "FAIL $label (rc diverges; actual=$rc expected=$expected_rc_actual)"
        fail=$((fail+1))
        return
    fi

    case "$mode" in
        miss)
            if [[ -s "$WORK/actual.out" ]]; then
                echo "FAIL $label (stdout must be empty on a real-grep miss)"
                fail=$((fail+1))
                return
            fi
            ;;
    esac

    echo "PASS $label"
    pass=$((pass+1))
}

assert_no_passthrough_state() {
    local files
    files=$(find "$WORK/store" "$WORK/repo/.greppy" -type f 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$files" -eq 0 ]]; then
        echo "PASS no-passthrough-state"
        pass=$((pass+1))
    else
        echo "FAIL no-passthrough-state ($files files created)"
        find "$WORK/store" "$WORK/repo/.greppy" -type f 2>/dev/null | sed 's/^/  /'
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

assert_byte_for_byte "STRICT -R missing-dir (rc=2)" err -R anything /no/such/dir

assert_no_passthrough_state

echo ""
echo "pass: $pass / fail: $fail"
[[ "$fail" -eq 0 ]]
