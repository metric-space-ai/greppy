#!/usr/bin/env bash
# GREP-COMPAT FUZZ battle — the drop-in contract must NEVER break.
#
# For a battery of patterns / flags / paths (including malformed UTF-8,
# huge lines, binary files, missing paths, regex metacharacters), assert
# that `grepplus-grep` (the `grepplus -R ...` drop-in path) produces
# stdout, stderr, and exit code BYTE-IDENTICAL to the system grep, and
# that grepplus never panics or crashes with a signal.
#
# The oracle is $REAL_GREP (default /usr/bin/grep). Each invocation runs
# the same argv through both and diffs the three observable channels.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

NAME="grep_fuzz"
require_bins "$GREPPLUS_GREP_BIN" || { emit_summary "$NAME"; exit 1; }

if [[ ! -x "$REAL_GREP" ]]; then
    fail "real grep oracle present ($REAL_GREP)"
    emit_summary "$NAME"; exit 1
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/battle-grepfuzz-XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"

# ---- build a fixture tree of adversarial inputs ---------------------------
mkdir -p data sub
printf 'foo\nbar\nFOObar\nbaz foo qux\n'            > data/plain.txt
printf 'alpha.beta\n(group)\n[set]\na+b=c\n^start$\n' > data/meta.txt
printf 'TAB\tafter\nCRLF line\r\nplain\n'           > data/whitespace.txt
printf 'no_newline_at_eof'                          > data/noeof.txt
printf ''                                           > data/empty.txt

# Binary file (NUL bytes) — grep switches to "binary file matches".
printf 'before\000\001\002NULafter\nfoo\n'          > data/binary.bin

# Malformed UTF-8: lone continuation bytes, truncated sequences.
printf 'valid line\n\xff\xfe bad bytes \xc3\x28\nfoo here\n' > data/badutf8.txt

# Huge single line (~1 MiB) with a needle in the middle.
{ head -c 500000 /dev/zero | tr '\0' 'x'; printf 'NEEDLE'; head -c 500000 /dev/zero | tr '\0' 'y'; printf '\n'; } > data/hugeline.txt

# Many lines for -c / -n / context flags.
seq 1 200 | sed 's/^/line /' > data/numbers.txt
echo "foo match" >> data/numbers.txt

# Nested dir for -R recursion.
printf 'deep foo\n' > sub/deep.txt

# ---- the battery ----------------------------------------------------------
# Each entry: a description, then the argv passed identically to both.
# We run from $WORK so relative paths resolve the same for both tools.

run_case() {
    local desc="$1"; shift
    local out_g out_r rc_g rc_r

    out_g="$("$GREPPLUS_GREP_BIN" "$@" 2>"$WORK/.eg")"; rc_g=$?
    local err_g; err_g="$(cat "$WORK/.eg")"
    out_r="$("$REAL_GREP" "$@" 2>"$WORK/.er")"; rc_r=$?
    local err_r; err_r="$(cat "$WORK/.er")"

    # Signal-level crash detection: rc >= 128 from grepplus but not grep,
    # or any rc difference, is a contract break.
    local broke=0 reasons=""
    if [[ "$rc_g" -ne "$rc_r" ]]; then broke=1; reasons+="rc($rc_g!=$rc_r) "; fi
    if [[ "$out_g" != "$out_r" ]]; then broke=1; reasons+="stdout "; fi
    if [[ "$err_g" != "$err_r" ]]; then broke=1; reasons+="stderr "; fi
    if [[ "$rc_g" -ge 128 ]]; then broke=1; reasons+="CRASH(rc=$rc_g) "; fi

    if [[ "$broke" -eq 0 ]]; then
        pass "$desc"
    else
        fail "$desc [$reasons]"
        echo "    argv: $*"
        if [[ "$out_g" != "$out_r" ]]; then
            echo "    stdout diff (g<  r>):"
            diff <(printf '%s' "$out_g") <(printf '%s' "$out_r") | head -6 | sed 's/^/      /'
        fi
        if [[ "$err_g" != "$err_r" ]]; then
            echo "    stderr g=[$err_g] r=[$err_r]"
        fi
    fi
}

# Note: some flag combinations (e.g. --color) emit terminal-dependent
# output; we avoid those and stick to the deterministic core contract.

# Plain literals
run_case "literal match"                 foo data/plain.txt
run_case "literal no-match (rc=1)"       zzzznope data/plain.txt
run_case "case-insensitive -i"           -i foo data/plain.txt
run_case "count -c"                      -c foo data/plain.txt
run_case "line-number -n"                -n foo data/plain.txt
run_case "invert -v"                     -v foo data/plain.txt
run_case "word -w"                       -w foo data/plain.txt
run_case "only-matching -o"              -o foo data/plain.txt
run_case "files-with-matches -l"         -l foo data/plain.txt data/empty.txt
run_case "quiet -q match"                -q foo data/plain.txt
run_case "quiet -q no-match"             -q nope data/plain.txt

# Regex metacharacters (BRE)
run_case "anchored ^"                    '^foo' data/plain.txt
run_case "anchored \$"                   'foo$' data/plain.txt
run_case "dot metachar"                  'f.o' data/plain.txt
run_case "char class"                    '[fb]oo' data/plain.txt
run_case "escaped dot literal"           'alpha\.beta' data/meta.txt
run_case "literal -F parens"             -F '(group)' data/meta.txt
run_case "ERE -E alternation"            -E 'foo|baz' data/plain.txt
run_case "ERE -E plus"                   -E 'a+b' data/meta.txt
run_case "BRE star"                      'fo*' data/plain.txt

# Whitespace / line endings
run_case "match in CRLF file"            line data/whitespace.txt
run_case "match tab-containing line"     after data/whitespace.txt

# Context flags
run_case "after-context -A1"             -A1 'line 100' data/numbers.txt
run_case "before-context -B1"            -B1 'line 100' data/numbers.txt
run_case "context -C2"                   -C2 'foo match' data/numbers.txt

# Recursion
run_case "recursive -r"                  -r foo data
run_case "recursive -R + -n"             -Rn deep sub

# Adversarial inputs
run_case "binary file default"           foo data/binary.bin
run_case "binary -a treat as text"       -a foo data/binary.bin
run_case "binary --binary-files=text"    --binary-files=text NUL data/binary.bin
run_case "malformed utf8 literal"        foo data/badutf8.txt
run_case "malformed utf8 no-match"       zzzz data/badutf8.txt
run_case "malformed utf8 -a"             -a here data/badutf8.txt
run_case "huge line match"               NEEDLE data/hugeline.txt
run_case "huge line count"               -c NEEDLE data/hugeline.txt
run_case "no newline at eof"             no_newline data/noeof.txt
run_case "empty file"                    anything data/empty.txt

# Missing / bad paths
run_case "missing file"                  foo data/does_not_exist.txt
run_case "missing dir recursive"         -r foo no_such_dir
run_case "mix existing+missing"          foo data/plain.txt data/missing.txt

# Multiple files (prefixes filenames)
run_case "multi-file prefixes"           foo data/plain.txt data/meta.txt
run_case "multi-file -n"                 -n foo data/plain.txt sub/deep.txt

emit_summary "$NAME"
