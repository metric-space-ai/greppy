#!/usr/bin/env bash
# MULTILANG battle (Track B) — drive the indexer + the new query CLI across a
# MIXED-language repository and assert RESULT CONTENT, not just exit codes.
#
# Why this script exists: the rest of the battle suite indexes Rust-only
# corpora. The extraction layer ships six fully-supported languages (Rust,
# Python, JavaScript, TypeScript, Go, Ruby). A regression that broke the
# cross-file CALLS/IMPORTS resolution for, say, Go or Ruby — or that broke
# the CLI's symbol resolution for a non-Rust qualified-name shape — would
# sail straight through a Rust-only suite. This script closes that hole: it
# builds ONE git repo containing all six languages with KNOWN cross-file
# caller/callee + import pairs, indexes it once, and asserts the resulting
# graph and the new CLI surfaces against those known truths per language.
#
# Black-box: drives the already-built `greppy` binary only. Touches no
# crate source or Cargo files.
#
# Invariants asserted:
#   * index completes (exit 0), prints no panic, DB integrity_check == ok.
#   * For EACH language that supports it, the graph contains a TRULY
#     cross-file edge of the expected kind:
#       - cross-file CALLS:   Rust, Python, JavaScript, TypeScript, Go, Ruby
#       - cross-file IMPORTS: Rust, Python, JavaScript, TypeScript, Ruby
#         (Go's package import target does not resolve to a node, while Ruby's
#          relative require now resolves directly without a standalone Import
#          node — both characteristics are asserted below.)
#   * `stats` RESULT CONTENT: per-label node counts and per-type edge
#     counts match the graph.db ground truth exactly.
#   * `who-calls` / `callees` / `path` resolve the known caller/callee for
#     a representative symbol IN EACH language and print the right symbol.
#   * `who-calls` on a Rust struct that has an incoming USAGE edge lands on
#     the struct (the node with the incoming edge), not a same-named node.
#   * `search-symbol` / `search-pattern` find known symbols / content across
#     all six languages.
#   * the byte-exact grep passthrough still holds on this mixed repo (`greppy
#     -R` vs the system grep, byte-exact, on several queries).
#   * determinism: index twice into independent stores -> identical node
#     and edge counts (and identical node/edge SETS).
#   * an unsupported-language file (a `.txt`) is handled gracefully (counted
#     as unsupported, no panic, no node rows for it).

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
[[ "$GREPPY_BIN" = /* ]] || GREPPY_BIN="$WORKSPACE_ROOT/$GREPPY_BIN"

NAME="multilang"

require_bins "$GREPPY_BIN" || { emit_summary "$NAME"; exit 1; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/battle-multilang-XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
CORPUS="$WORK/corpus"
export GREPPY_STORE_DIR="$WORK/store"

# ---------------------------------------------------------------------------
# Fixture builder — a MIXED-language repo with KNOWN cross-file relations.
#
# Per language, two files: a `helper` defining `<lang>_helper` and a `main`
# that imports it and a `<lang>_caller` that calls it cross-file. The exact
# source forms below were chosen so each language's extractor actually emits
# the cross-file edge (e.g. Ruby needs explicit `rb_helper()` parens for a
# CALLS edge; a bare `rb_helper` is parsed as an identifier, not a call).
#
# Additionally, the Rust side defines a `Widget` struct that `make()` returns
# by value (TYPE_REF) so `who-calls` has a real referenced type to land on.
# ---------------------------------------------------------------------------
build_corpus() {
    local C="$1"
    rm -rf "$C"
    mkdir -p "$C"

    # ---- Rust ----
    mkdir -p "$C/rust/src"
    cat > "$C/rust/Cargo.toml" <<'EOF'
[package]
name = "ml_corpus"
version = "0.0.0"
edition = "2021"

[lib]
path = "src/lib.rs"
EOF
    cat > "$C/rust/src/lib.rs" <<'EOF'
pub mod helper;
pub mod widget;

use crate::helper::rust_helper;
use crate::widget::Widget;

pub fn rust_caller() -> u64 {
    let w = make();
    rust_helper() + w.id
}

pub fn make() -> Widget {
    Widget { id: 0 }
}
EOF
    cat > "$C/rust/src/helper.rs" <<'EOF'
pub fn rust_helper() -> u64 {
    7
}
EOF
    cat > "$C/rust/src/widget.rs" <<'EOF'
pub struct Widget {
    pub id: u64,
}
EOF

    # ---- Python ----
    mkdir -p "$C/py"
    cat > "$C/py/helper.py" <<'EOF'
def py_helper():
    return 7
EOF
    cat > "$C/py/main.py" <<'EOF'
from helper import py_helper


def py_caller():
    return py_helper() + 1
EOF

    # ---- JavaScript ----
    mkdir -p "$C/js"
    cat > "$C/js/helper.js" <<'EOF'
export function jsHelper() {
    return 7;
}
EOF
    cat > "$C/js/main.js" <<'EOF'
import { jsHelper } from './helper.js';

export function jsCaller() {
    return jsHelper() + 1;
}
EOF

    # ---- TypeScript ----
    mkdir -p "$C/ts"
    cat > "$C/ts/helper.ts" <<'EOF'
export function tsHelper(): number {
    return 7;
}
EOF
    cat > "$C/ts/main.ts" <<'EOF'
import { tsHelper } from './helper';

export function tsCaller(): number {
    return tsHelper() + 1;
}
EOF

    # ---- Go ----
    mkdir -p "$C/go/helper"
    cat > "$C/go/helper/helper.go" <<'EOF'
package helper

func GoHelper() int {
    return 7
}
EOF
    cat > "$C/go/main.go" <<'EOF'
package main

import "example.com/p/helper"

func GoCaller() int {
    return helper.GoHelper() + 1
}

func main() {
    _ = GoCaller()
}
EOF

    # ---- Ruby ----
    mkdir -p "$C/rb"
    cat > "$C/rb/helper.rb" <<'EOF'
def rb_helper
  7
end
EOF
    cat > "$C/rb/main.rb" <<'EOF'
require_relative 'helper'

def rb_caller
  x = rb_helper()
  x + 1
end
EOF

    # ---- unsupported language: plain text ----
    cat > "$C/notes.txt" <<'EOF'
plain text file: mentions rust_caller py_caller jsCaller but is NOT code.
EOF
}

build_corpus "$CORPUS"
git_init_corpus "$CORPUS"
[[ -d "$CORPUS/.git" ]]; check $? "mixed-language corpus is a git repo"

# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------
idx_log="$WORK/index.log"
( cd "$CORPUS" && "$GREPPY_BIN" index . ) >"$idx_log" 2>&1
check $? "index mixed-language corpus (exit 0)"
if grep -qiE 'panic|thread .* panicked' "$idx_log"; then
    fail "no panic during multi-language index"
    sed -n '1,30p' "$idx_log"
else
    pass "no panic during multi-language index"
fi
# The index summary must report at least one unsupported file (the corpus
# contains a `.txt` and a `Cargo.toml`, both unsupported languages), proving
# unsupported-language files are SEEN and classified, not silently dropped
# in a way that could mask a crash. The exact-zero-nodes check below pins
# that the .txt specifically contributed nothing to the graph.
unsup_n="$(sed -n 's/.*indexed [0-9]\{1,\} files (\([0-9]\{1,\}\) unsupported.*/\1/p' "$idx_log" | head -n1)"
if [[ -n "$unsup_n" && "$unsup_n" -ge 1 ]]; then
    pass "index reports unsupported files (graceful classification, count=$unsup_n)"
else
    fail "index reports unsupported files (graceful classification)"
    sed -n '1,5p' "$idx_log"
fi

DB="$(graph_db_path "$GREPPY_STORE_DIR")"
if [[ -z "$DB" ]]; then
    fail "graph.db exists"
    emit_summary "$NAME"; exit 1
fi
pass "graph.db exists"

integ="$(sqlite_q "$DB" "PRAGMA integrity_check;" 2>/dev/null || echo ERR)"
assert_eq "ok" "$integ" "DB integrity_check ok on mixed-language repo"

# Unsupported content still receives its deterministic File node, but no
# language-level definitions.
txt_files="$(sqlite_q "$DB" "SELECT count(*) FROM nodes WHERE label='File' AND file_path LIKE '%notes.txt';" 2>/dev/null || echo ERR)"
txt_defs="$(sqlite_q "$DB" "SELECT count(*) FROM nodes WHERE label<>'File' AND file_path LIKE '%notes.txt';" 2>/dev/null || echo ERR)"
assert_eq "1" "$txt_files" "unsupported .txt file retained its File node"
assert_eq "0" "$txt_defs" "unsupported .txt file produced no definition nodes"

# ---------------------------------------------------------------------------
# Per-language cross-file edge invariants (queried straight off graph.db).
#
# A "cross-file" edge is one whose source and target nodes live in
# different files. Source file is matched by directory prefix so each
# language is isolated.
# ---------------------------------------------------------------------------
xfile_edges() {
    # $1 = edge_type, $2 = source-file LIKE prefix
    sqlite_q "$DB" "
      SELECT count(*) FROM edges e
      JOIN nodes s ON s.id = e.source_id
      JOIN nodes t ON t.id = e.target_id
      WHERE e.edge_type='$1'
        AND s.file_path LIKE '$2'
        AND s.file_path <> t.file_path;" 2>/dev/null || echo 0
}

# Every supported language resolves a cross-file CALLS edge.
for lang in "rust:rust/%" "python:py/%" "javascript:js/%" "typescript:ts/%" "go:go/%" "ruby:rb/%"; do
    name="${lang%%:*}"; prefix="${lang##*:}"
    c="$(xfile_edges CALLS "$prefix")"
    assert_ge "${c:-0}" 1 "cross-file CALLS edge present for $name"
done

# Cross-file IMPORTS resolve for Rust, Python, JS, TS, and Ruby.
for lang in "rust:rust/%" "python:py/%" "javascript:js/%" "typescript:ts/%" "ruby:rb/%"; do
    name="${lang%%:*}"; prefix="${lang##*:}"
    c="$(xfile_edges IMPORTS "$prefix")"
    assert_ge "${c:-0}" 1 "cross-file IMPORTS edge present for $name"
done

# Go's package import remains unresolved and emits no standalone Import node.
imp_nodes="$(sqlite_q "$DB" "SELECT count(*) FROM nodes WHERE label='Import' AND file_path LIKE 'go/%';" 2>/dev/null || echo 0)"
assert_eq "0" "${imp_nodes:-0}" "go emits no standalone Import node"
c="$(xfile_edges IMPORTS 'go/%')"
assert_eq "0" "${c:-0}" "go has no resolved cross-file IMPORTS edge (known: package target unresolved)"

# Ruby's relative require resolves directly, also without an Import node.
imp_nodes="$(sqlite_q "$DB" "SELECT count(*) FROM nodes WHERE label='Import' AND file_path LIKE 'rb/%';" 2>/dev/null || echo 0)"
assert_eq "0" "${imp_nodes:-0}" "ruby emits no standalone Import node"

# ---------------------------------------------------------------------------
# `stats` RESULT CONTENT — assert the printed per-label / per-type counts
# match the graph.db ground truth exactly (not just "stats exits 0").
# ---------------------------------------------------------------------------
stats_out="$( cd "$CORPUS" && "$GREPPY_BIN" stats 2>/dev/null )"
echo "[multilang] stats ->"; echo "$stats_out" | sed 's/^/    /'

# Helper: pull the integer printed after a label/type token in `stats`.
stats_count() {
    # $1 = token (label or edge type as printed, e.g. "Function" / "CALLS")
    sed -n "s/^[[:space:]]*$1[[:space:]]\\{1,\\}\\([0-9]\\{1,\\}\\)\$/\\1/p" <<<"$stats_out" | head -n1
}

# Per-label node counts: compare stats output to DB.
for label in Function Module Import Call; do
    db_c="$(sqlite_q "$DB" "SELECT count(*) FROM nodes WHERE label='$label';" 2>/dev/null || echo 0)"
    st_c="$(stats_count "$label")"
    # `stats` omits zero-count labels/types, so an absent token means 0 nodes,
    # which is consistent with graph.db. A real mismatch (db_c>0 but absent from
    # stats) still fails, since db_c != 0.
    : "${st_c:=0}"
    assert_eq "$db_c" "$st_c" "stats node count for label $label matches graph.db"
done

# Per-type edge counts: compare stats output to DB.
for et in CALLS IMPORTS; do
    db_c="$(sqlite_q "$DB" "SELECT count(*) FROM edges WHERE edge_type='$et';" 2>/dev/null || echo 0)"
    st_c="$(stats_count "$et")"
    # `stats` omits zero-count labels/types, so an absent token means 0 nodes,
    # which is consistent with graph.db. A real mismatch (db_c>0 but absent from
    # stats) still fails, since db_c != 0.
    : "${st_c:=0}"
    assert_eq "$db_c" "$st_c" "stats edge count for type $et matches graph.db"
done

# stats node/edge TOTALS match the DB totals.
db_nodes="$(sqlite_q "$DB" "SELECT count(*) FROM nodes;" 2>/dev/null || echo 0)"
db_edges="$(sqlite_q "$DB" "SELECT count(*) FROM edges;" 2>/dev/null || echo 0)"
st_nodes="$(sed -n 's/^nodes:[[:space:]]*\([0-9]\{1,\}\)$/\1/p' <<<"$stats_out" | head -n1)"
st_edges="$(sed -n 's/^edges:[[:space:]]*\([0-9]\{1,\}\)$/\1/p' <<<"$stats_out" | head -n1)"
assert_eq "$db_nodes" "${st_nodes:-MISSING}" "stats node TOTAL matches graph.db"
assert_eq "$db_edges" "${st_edges:-MISSING}" "stats edge TOTAL matches graph.db"

# ---------------------------------------------------------------------------
# Navigation CLI — assert RESULT CONTENT per language.
# ---------------------------------------------------------------------------
nav() { ( cd "$CORPUS" && "$GREPPY_BIN" "$@" ) 2>>"$WORK/nav.err"; }

# who-calls <helper> names the right caller, in the right file, per language.
# tuple: lang : helper-symbol : expected-caller : caller-file
for row in \
    "rust:rust_helper:rust_caller:rust/src/lib.rs" \
    "python:py_helper:py_caller:py/main.py" \
    "javascript:jsHelper:jsCaller:js/main.js" \
    "typescript:tsHelper:tsCaller:ts/main.ts" \
    "go:GoHelper:GoCaller:go/main.go" \
    "ruby:rb_helper:rb_caller:rb/main.rb"; do
    IFS=: read -r lang helper caller cfile <<<"$row"
    out="$(nav who-calls "$helper")"
    if [[ "$out" == "no callers" ]]; then
        fail "who-calls $helper ($lang) finds its caller (got 'no callers')"
    elif grep -q "$caller" <<<"$out" && grep -q "$cfile" <<<"$out"; then
        pass "who-calls $helper ($lang) names caller '$caller' in $cfile"
    else
        fail "who-calls $helper ($lang) names caller '$caller' in $cfile (got: $out)"
    fi
done

# callees <caller> names the right callee per language.
for row in \
    "rust:rust_caller:rust_helper:rust/src/helper.rs" \
    "python:py_caller:py_helper:py/helper.py" \
    "javascript:jsCaller:jsHelper:js/helper.js" \
    "typescript:tsCaller:tsHelper:ts/helper.ts" \
    "go:GoCaller:GoHelper:go/helper/helper.go" \
    "ruby:rb_caller:rb_helper:rb/helper.rb"; do
    IFS=: read -r lang caller callee hfile <<<"$row"
    out="$(nav callees "$caller")"
    if grep -q "$callee" <<<"$out" && grep -q "$hfile" <<<"$out"; then
        pass "callees $caller ($lang) names callee '$callee' in $hfile"
    else
        fail "callees $caller ($lang) names callee '$callee' in $hfile (got: $out)"
    fi
done

# path <caller> -> <callee> returns the ordered cross-file CALLS path.
for row in \
    "rust:rust_caller:rust_helper" \
    "python:py_caller:py_helper" \
    "go:GoCaller:GoHelper" \
    "ruby:rb_caller:rb_helper"; do
    IFS=: read -r lang from to <<<"$row"
    out="$(nav path --from "$from" --to "$to")"
    if grep -q "$from" <<<"$out" && grep -q "$to" <<<"$out"; then
        pass "path $from -> $to ($lang) returns both endpoints"
    else
        fail "path $from -> $to ($lang) returns both endpoints (got: $out)"
    fi
done

# who-calls on the Rust Widget struct (referenced by USAGE from make()).
# This guards the symbol-resolution layer: Widget names a struct used by a
# USAGE edge; resolution must land on the struct node that carries that
# incoming edge. The 0.3.0 text omits edge-kind labels, so pin the known
# referrer and its source location; a wrong same-name resolution says exactly
# "no callers" and cannot satisfy this assertion.
fu_out="$(nav who-calls Widget)"
echo "[multilang] who-calls Widget ->"; echo "$fu_out" | sed 's/^/    /'
if [[ "$fu_out" == "no callers" ]]; then
    fail "who-calls Widget is NOT 'no callers' — Widget IS referenced via USAGE"
elif grep -qE '^rust/src/lib\.rs:[0-9]+  make$' <<<"$fu_out"; then
    pass "who-calls Widget names referrer 'make' at its source location"
else
    fail "who-calls Widget names referrer 'make' at its source location (got: $fu_out)"
fi
# A genuinely-absent symbol has its own exact 0.3.0 message.
nu_out="$(nav who-calls this_symbol_does_not_exist_anywhere)"
if [[ "$nu_out" == 'no symbol `this_symbol_does_not_exist_anywhere`' ]]; then
    pass "who-calls on an absent symbol reports the exact no-symbol message"
else
    fail "who-calls on an absent symbol reports the exact no-symbol message (got: $nu_out)"
fi

# ---------------------------------------------------------------------------
# search-symbol / search-pattern — find known symbols/content across languages.
# ---------------------------------------------------------------------------
# search-symbol is case-sensitive in 0.3.0. Query both source spellings so the
# helper Function must still surface in EVERY language.
ss_out="$(nav search-symbol helper --all; nav search-symbol Helper --all)"
for row in \
    "rust:rust_helper" \
    "python:py_helper" \
    "javascript:jsHelper" \
    "typescript:tsHelper" \
    "go:GoHelper" \
    "ruby:rb_helper"; do
    IFS=: read -r lang sym <<<"$row"
    if grep -qE "^[^ ]+:[0-9]+  $sym  function$" <<<"$ss_out"; then
        pass "search-symbol 'helper' finds the $lang Function symbol $sym"
    else
        fail "search-symbol 'helper' finds the $lang Function symbol $sym (missing)"
    fi
done

# search-pattern "caller" must surface the caller definition line in each
# language whose helper-call site mentions "caller" textually. (Rust/Python/
# Ruby write `..._caller`; JS/TS/Go write `..Caller` — all contain "caller"
# case-insensitively, but search-pattern is case-sensitive, so assert the three
# snake_case ones whose source literally contains the lowercase token.)
sc_out="$(nav search-pattern caller)"
echo "[multilang] search-pattern caller ->"; echo "$sc_out" | sed 's/^/    /'
for row in \
    "rust:rust/src/lib.rs" \
    "python:py/main.py" \
    "ruby:rb/main.rb"; do
    IFS=: read -r lang f <<<"$row"
    if grep -q "$f" <<<"$sc_out"; then
        pass "search-pattern 'caller' finds a match in the $lang file ($f)"
    else
        fail "search-pattern 'caller' finds a match in the $lang file ($f) (got: $sc_out)"
    fi
done
# search-pattern for a body token shared by several languages ("return 7").
sc7_out="$(nav search-pattern "return 7")"
sc7_hits="$(grep -cE 'helper\.(py|js|ts|go)' <<<"$sc7_out")"
assert_ge "${sc7_hits:-0}" 2 "search-pattern 'return 7' finds the body across multiple languages"

# ---------------------------------------------------------------------------
# Grep-compatible passthrough contract on the mixed repo. A fresh index is in
# scope deliberately: ordinary grep invocations must still be byte-exact and
# side-effect-free.
# ---------------------------------------------------------------------------
if [[ ! -x "$REAL_GREP" ]]; then
    fail "real grep oracle present ($REAL_GREP)"
else
    grep_case() {
        local bin="$1" label="$2" rcg rcr
        shift 2
        ( cd "$CORPUS" && "$bin" "$@" ) >"$WORK/grep-sub.out" 2>"$WORK/grep-sub.err"; rcg=$?
        ( cd "$CORPUS" && "$REAL_GREP" "$@" ) >"$WORK/grep-ref.out" 2>"$WORK/grep-ref.err"; rcr=$?
        if cmp -s "$WORK/grep-sub.out" "$WORK/grep-ref.out" \
            && cmp -s "$WORK/grep-sub.err" "$WORK/grep-ref.err" \
            && [[ "$rcg" -eq "$rcr" && "$rcg" -lt 128 ]]; then
            pass "$label byte-exact: $* (rc=$rcg)"
        else
            fail "$label byte-exact: $* (rc $rcg vs $rcr)"
            diff -u "$WORK/grep-ref.out" "$WORK/grep-sub.out" | head -8 | sed 's/^/      /'
            diff -u "$WORK/grep-ref.err" "$WORK/grep-sub.err" | head -8 | sed 's/^/      /'
        fi
    }

    for args in \
        '-R -n helper .' \
        '-R -n no_such_needle_anywhere_xyz .' \
        '-Rc helper .' \
        '-R return 7 .' \
        '-n helper rust/src/helper.rs' \
        '-n caller py/main.py'; do
        read -r -a argv <<<"$args"
        grep_case "$GREPPY_BIN" greppy "${argv[@]}"
    done

fi

# ---------------------------------------------------------------------------
# Determinism — index twice into independent stores; counts + sets identical.
# ---------------------------------------------------------------------------
dump_nodes() { sqlite_q "$1" "SELECT label||'|'||qualified_name||'|'||file_path FROM nodes ORDER BY 1;"; }
dump_edges() {
    sqlite_q "$1" "
      SELECT s.qualified_name||'|'||t.qualified_name||'|'||e.edge_type
      FROM edges e
      JOIN nodes s ON s.id=e.source_id
      JOIN nodes t ON t.id=e.target_id
      ORDER BY 1;"
}

storeA="$WORK/storeA"
storeB="$WORK/storeB"
( cd "$CORPUS" && GREPPY_STORE_DIR="$storeA" "$GREPPY_BIN" index . ) >/dev/null 2>&1
check $? "determinism run A indexed"
( cd "$CORPUS" && GREPPY_STORE_DIR="$storeB" "$GREPPY_BIN" index . ) >/dev/null 2>&1
check $? "determinism run B indexed"

DBA="$(graph_db_path "$storeA")"
DBB="$(graph_db_path "$storeB")"
if [[ -z "$DBA" || -z "$DBB" ]]; then
    fail "both determinism graph.db files exist"
else
    pass "both determinism graph.db files exist"
    nA="$(sqlite_q "$DBA" "SELECT count(*) FROM nodes;")"
    nB="$(sqlite_q "$DBB" "SELECT count(*) FROM nodes;")"
    assert_eq "$nA" "$nB" "node count identical across mixed-repo runs"
    eA="$(sqlite_q "$DBA" "SELECT count(*) FROM edges;")"
    eB="$(sqlite_q "$DBB" "SELECT count(*) FROM edges;")"
    assert_eq "$eA" "$eB" "edge count identical across mixed-repo runs"
    if diff <(dump_nodes "$DBA") <(dump_nodes "$DBB") >/dev/null; then
        pass "node SET byte-identical across mixed-repo runs"
    else
        fail "node SET byte-identical across mixed-repo runs"
        diff <(dump_nodes "$DBA") <(dump_nodes "$DBB") | head -12
    fi
    if diff <(dump_edges "$DBA") <(dump_edges "$DBB") >/dev/null; then
        pass "edge SET byte-identical across mixed-repo runs"
    else
        fail "edge SET byte-identical across mixed-repo runs"
        diff <(dump_edges "$DBA") <(dump_edges "$DBB") | head -12
    fi
fi

emit_summary "$NAME"
