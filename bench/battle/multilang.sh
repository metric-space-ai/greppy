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
# Black-box: drives the already-built `grepplus` binary only. Touches no
# crate source or Cargo files.
#
# Invariants asserted:
#   * index completes (exit 0), prints no panic, DB integrity_check == ok.
#   * For EACH language that supports it, the graph contains a TRULY
#     cross-file edge of the expected kind:
#       - cross-file CALLS:   Rust, Python, JavaScript, TypeScript, Go, Ruby
#       - cross-file IMPORTS: Rust, Python, JavaScript, TypeScript
#         (Go/Ruby emit Import *nodes* but their package/relative-path
#          import targets do not resolve to a node, so no cross-file
#          IMPORTS edge is produced — asserted as a known characteristic.)
#   * `stats` RESULT CONTENT: per-label node counts and per-type edge
#     counts match the graph.db ground truth exactly.
#   * `who-calls` / `callees` / `path` resolve the known caller/callee for
#     a representative symbol IN EACH language and print the right symbol.
#   * `find-usages` on a Rust struct that is used by TYPE_REF lands on the
#     struct (the node with the incoming edges), not a same-named node.
#   * `search-symbols` / `search-code` find known symbols / content across
#     all six languages.
#   * the drop-in grep contract still holds on this mixed repo (`grepplus
#     -R` vs the system grep, byte-exact, on several queries).
#   * determinism: index twice into independent stores -> identical node
#     and edge counts (and identical node/edge SETS).
#   * an unsupported-language file (a `.txt`) is handled gracefully (counted
#     as unsupported, no panic, no node rows for it).

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

NAME="multilang"

require_bins "$GREPPLUS_BIN" || { emit_summary "$NAME"; exit 1; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/battle-multilang-XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
CORPUS="$WORK/corpus"
export GREPPLUS_STORE_DIR="$WORK/store"

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
# by value (TYPE_REF) so `find-usages` has a real referenced type to land on.
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
( cd "$CORPUS" && "$GREPPLUS_BIN" index . ) >"$idx_log" 2>&1
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

DB="$(graph_db_path "$GREPPLUS_STORE_DIR")"
if [[ -z "$DB" ]]; then
    fail "graph.db exists"
    emit_summary "$NAME"; exit 1
fi
pass "graph.db exists"

integ="$(sqlite_q "$DB" "PRAGMA integrity_check;" 2>/dev/null || echo ERR)"
assert_eq "ok" "$integ" "DB integrity_check ok on mixed-language repo"

# The .txt file must NOT have produced any node rows (it is unsupported).
txt_nodes="$(sqlite_q "$DB" "SELECT count(*) FROM nodes WHERE file_path LIKE '%notes.txt';" 2>/dev/null || echo ERR)"
assert_eq "0" "$txt_nodes" "unsupported .txt file produced no graph nodes"

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

# Cross-file IMPORTS resolve for Rust, Python, JS, TS.
for lang in "rust:rust/%" "python:py/%" "javascript:js/%" "typescript:ts/%"; do
    name="${lang%%:*}"; prefix="${lang##*:}"
    c="$(xfile_edges IMPORTS "$prefix")"
    assert_ge "${c:-0}" 1 "cross-file IMPORTS edge present for $name"
done

# Go/Ruby: an Import NODE is produced but the package/relative import does
# not resolve to a node, so there is no cross-file IMPORTS edge. Assert
# both halves of this known characteristic so a future change either way is
# noticed (the node must exist; the unresolved-edge count must be 0).
for lang in "go:go/%" "ruby:rb/%"; do
    name="${lang%%:*}"; prefix="${lang##*:}"
    imp_nodes="$(sqlite_q "$DB" "SELECT count(*) FROM nodes WHERE label='Import' AND file_path LIKE '$prefix';" 2>/dev/null || echo 0)"
    assert_ge "${imp_nodes:-0}" 1 "$name emits an Import node"
    c="$(xfile_edges IMPORTS "$prefix")"
    assert_eq "0" "${c:-0}" "$name has no resolved cross-file IMPORTS edge (known: package/relative target unresolved)"
done

# ---------------------------------------------------------------------------
# `stats` RESULT CONTENT — assert the printed per-label / per-type counts
# match the graph.db ground truth exactly (not just "stats exits 0").
# ---------------------------------------------------------------------------
stats_out="$( cd "$CORPUS" && "$GREPPLUS_BIN" stats 2>/dev/null )"
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
    : "${st_c:=MISSING}"
    assert_eq "$db_c" "$st_c" "stats node count for label $label matches graph.db"
done

# Per-type edge counts: compare stats output to DB.
for et in CALLS IMPORTS; do
    db_c="$(sqlite_q "$DB" "SELECT count(*) FROM edges WHERE edge_type='$et';" 2>/dev/null || echo 0)"
    st_c="$(stats_count "$et")"
    : "${st_c:=MISSING}"
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
nav() { ( cd "$CORPUS" && "$GREPPLUS_BIN" "$@" ) 2>>"$WORK/nav.err"; }

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
    if grep -q '(no callers)' <<<"$out"; then
        fail "who-calls $helper ($lang) finds its caller (got '(no callers)')"
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

# find-usages on the Rust Widget struct (referenced by TYPE_REF from make()).
# This guards the symbol-resolution layer: Widget names a struct used by a
# TYPE_REF edge; resolution must land on the struct node that carries that
# incoming edge.
fu_out="$(nav find-usages Widget)"
echo "[multilang] find-usages Widget ->"; echo "$fu_out" | sed 's/^/    /'
if grep -q '(no usages)' <<<"$fu_out"; then
    fail "find-usages Widget is NOT '(no usages)' — Widget IS referenced via TYPE_REF"
elif grep -qE 'TYPE_REF|USES' <<<"$fu_out" && grep -q 'make' <<<"$fu_out"; then
    pass "find-usages Widget names referrer 'make' with a TYPE_REF/USES edge"
else
    fail "find-usages Widget names referrer 'make' with a TYPE_REF/USES edge (got: $fu_out)"
fi
# Counter-case: a genuinely-absent symbol must still report no usages.
nu_out="$(nav find-usages this_symbol_does_not_exist_anywhere)"
if grep -qE '\(no usages\)|\(symbol not found\)' <<<"$nu_out"; then
    pass "find-usages on an absent symbol reports no usages / not found"
else
    fail "find-usages on an absent symbol reports no usages / not found (got: $nu_out)"
fi

# ---------------------------------------------------------------------------
# search-symbols / search-code — find known symbols/content across languages.
# ---------------------------------------------------------------------------
# search-symbols "helper" must surface the helper Function in EVERY language.
ss_out="$(nav search-symbols helper)"
for row in \
    "rust:rust_helper" \
    "python:py_helper" \
    "javascript:jsHelper" \
    "typescript:tsHelper" \
    "go:GoHelper" \
    "ruby:rb_helper"; do
    IFS=: read -r lang sym <<<"$row"
    if grep -qE "Function .*::$sym " <<<"$ss_out"; then
        pass "search-symbols 'helper' finds the $lang Function symbol $sym"
    else
        fail "search-symbols 'helper' finds the $lang Function symbol $sym (missing)"
    fi
done

# search-code "caller" must surface the caller definition line in each
# language whose helper-call site mentions "caller" textually. (Rust/Python/
# Ruby write `..._caller`; JS/TS/Go write `..Caller` — all contain "caller"
# case-insensitively, but search-code is case-sensitive, so assert the three
# snake_case ones whose source literally contains the lowercase token.)
sc_out="$(nav search-code caller)"
echo "[multilang] search-code caller ->"; echo "$sc_out" | sed 's/^/    /'
for row in \
    "rust:rust/src/lib.rs" \
    "python:py/main.py" \
    "ruby:rb/main.rb"; do
    IFS=: read -r lang f <<<"$row"
    if grep -q "$f" <<<"$sc_out"; then
        pass "search-code 'caller' finds a match in the $lang file ($f)"
    else
        fail "search-code 'caller' finds a match in the $lang file ($f) (got: $sc_out)"
    fi
done
# search-code for a body token shared by several languages ("return 7").
sc7_out="$(nav search-code "return 7")"
sc7_hits="$(grep -cE 'helper\.(py|js|ts|go)' <<<"$sc7_out")"
assert_ge "${sc7_hits:-0}" 2 "search-code 'return 7' finds the body across multiple languages"

# ---------------------------------------------------------------------------
# Drop-in grep contract on the mixed repo.
#
# IMPORTANT — the contract this asserts (verified against the source, not
# assumed): `grepplus -R <pat>` is the drop-in surface that is byte-exact
# with the system grep WHENEVER grepplus does not apply its semantic
# augmentation. Augmentation (a synthetic
# "<file>:1:<!-- GREPPLUS_NON_CANONICAL_HIT: <pat> -->" line appended to
# stdout) is grepplus's INTENDED value-add and fires only for a plain
# recursive listing (`-R`/`-r` without -c) when (a) the indexed graph for
# the cwd is FRESH and (b) the pattern has semantic graph hits. It is
# gated by the freshness check in `grepplus_grep::run::freshness_gate`
# (Mode::VisibleAugment). So:
#   * grepplus -R on a NO-semantic-hit pattern   -> byte-exact
#   * grepplus -R with -c (count)                -> byte-exact
#   * grepplus (non-recursive, single file)      -> byte-exact
#   * grepplus-grep (the pure drop-in binary)    -> ALWAYS byte-exact,
#                                                   even on a hit pattern
# A plain `grepplus -R -n <hit-pattern> .` on this freshly-indexed repo
# DELIBERATELY differs (augmented) — asserting byte-exact there would be
# asserting the wrong contract. We assert byte-exact on the STRICT cases
# and separately assert that the augmentation is present-and-correct, so
# both the drop-in guarantee AND the value-add are pinned.
# ---------------------------------------------------------------------------
if [[ ! -x "$REAL_GREP" ]]; then
    fail "real grep oracle present ($REAL_GREP)"
else
    # --- grepplus -R, STRICT (non-augmented) cases: byte-exact vs grep ------
    grep_case() {
        # $@ = argv given identically to `grepplus` and the system grep, run
        # from inside $CORPUS so relative paths resolve the same. Used only
        # for invocations grepplus does NOT augment (see header).
        local og or rcg rcr
        og="$( cd "$CORPUS" && "$GREPPLUS_BIN" "$@" 2>/dev/null )"; rcg=$?
        or="$( cd "$CORPUS" && "$REAL_GREP" "$@" 2>/dev/null )"; rcr=$?
        if [[ "$og" == "$or" && "$rcg" -eq "$rcr" && "$rcg" -lt 128 ]]; then
            pass "grepplus drop-in byte-exact: $* (rc=$rcg)"
        else
            fail "grepplus drop-in byte-exact: $* (rc $rcg vs $rcr)"
            diff <(printf '%s' "$og") <(printf '%s' "$or") | head -6 | sed 's/^/      /'
        fi
    }
    # -R recursive, pattern with NO semantic graph hit -> not augmented.
    grep_case -R -n "no_such_needle_anywhere_xyz" .
    # -R recursive count mode -> not augmented (augment only adds a listing
    # line, which -c would not emit; grepplus keeps -c byte-exact).
    grep_case -Rc "helper" .
    # -R recursive, a literal that exists only in code bodies, no graph node.
    grep_case -R "return 7" .
    # Non-recursive single-file searches are never augmented.
    grep_case -n "helper" "rust/src/helper.rs"
    grep_case -n "caller" "py/main.py"

    # --- grepplus-grep, the pure drop-in: byte-exact on the canonical
    # drop-in scenario (a tree with NO fresh index in scope) even for a hit
    # pattern. Augmentation is gated on a fresh graph for the cwd's store;
    # with GREPPLUS_STORE_DIR unset there is no fresh store, so the value-add
    # never fires and the contract is pure byte-exact. (When a fresh store IS
    # in scope, grepplus-grep augments too — that path is covered by the
    # `grepplus -R` augmentation assertions above.) `env -u` runs the child
    # with the store var removed; both tools see the same (empty) env.
    if [[ -x "$GREPPLUS_GREP_BIN" ]]; then
        gg_case() {
            local og or rcg rcr
            og="$( cd "$CORPUS" && env -u GREPPLUS_STORE_DIR "$GREPPLUS_GREP_BIN" "$@" 2>/dev/null )"; rcg=$?
            or="$( cd "$CORPUS" && env -u GREPPLUS_STORE_DIR "$REAL_GREP" "$@" 2>/dev/null )"; rcr=$?
            if [[ "$og" == "$or" && "$rcg" -eq "$rcr" && "$rcg" -lt 128 ]]; then
                pass "grepplus-grep drop-in byte-exact (no fresh store): $* (rc=$rcg)"
            else
                fail "grepplus-grep drop-in byte-exact (no fresh store): $* (rc $rcg vs $rcr)"
                diff <(printf '%s' "$og") <(printf '%s' "$or") | head -6 | sed 's/^/      /'
            fi
        }
        gg_case -R -n "helper" .
        gg_case -R -n "caller" .
        gg_case -R "function" .
    else
        echo "[multilang] note: grepplus-grep binary absent; skipping pure drop-in checks"
    fi

    # --- the augmentation value-add fires AND is well-formed ----------------
    # A plain recursive listing on a fresh graph for a pattern WITH a graph
    # node ("helper" matches every *_helper Function) must append exactly one
    # synthetic semantic line per query, pointing at a sidecar .md in the
    # store. This pins grepplus's documented divergence from raw grep so a
    # regression that silently dropped the value-add (or that started
    # augmenting count/no-hit queries) is caught.
    aug_out="$( cd "$CORPUS" && "$GREPPLUS_BIN" -R -n "helper" . 2>/dev/null )"
    raw_out="$( cd "$CORPUS" && "$REAL_GREP" -R -n "helper" . 2>/dev/null )"
    if [[ "$aug_out" != "$raw_out" ]] && grep -q 'GREPPLUS_NON_CANONICAL_HIT: helper' <<<"$aug_out"; then
        pass "grepplus -R augments a fresh-graph hit pattern with a semantic line (intended value-add)"
    else
        fail "grepplus -R augments a fresh-graph hit pattern with a semantic line (intended value-add)"
    fi
    # The augmented output must be a strict SUPERSET of raw grep: every raw
    # grep line is still present and byte-identical (augmentation only ADDS).
    missing="$(comm -23 <(printf '%s\n' "$raw_out" | sort) <(printf '%s\n' "$aug_out" | sort) | head -3)"
    if [[ -z "$missing" ]]; then
        pass "grepplus -R augmentation is additive: every raw grep line preserved verbatim"
    else
        fail "grepplus -R augmentation is additive: every raw grep line preserved verbatim"
        printf '      dropped: %s\n' "$missing"
    fi
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
( cd "$CORPUS" && GREPPLUS_STORE_DIR="$storeA" "$GREPPLUS_BIN" index . ) >/dev/null 2>&1
check $? "determinism run A indexed"
( cd "$CORPUS" && GREPPLUS_STORE_DIR="$storeB" "$GREPPLUS_BIN" index . ) >/dev/null 2>&1
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
