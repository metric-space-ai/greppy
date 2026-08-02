#!/usr/bin/env bash
# NAVIGATION battle — drive the graph-navigation CLI end to end and assert
# the RESULT CONTENT, not just exit codes or row counts in SQLite.
#
# Why this script exists: the other battle
# scripts assert the graph by querying graph.db directly with sqlite3 and
# NEVER run `who-calls` / `trace`. So a *resolution* bug — the CLI resolving
# a symbol name to the wrong node and printing "no callers" for a symbol that
# is demonstrably used — sailed straight
# through a green suite. A black-box harness that only inspects the DB can
# never catch a bug in the layer that maps a user's symbol name onto that
# DB. This script closes that hole: it builds a fixture with a KNOWN
# caller/callee and a Struct+Impl that SHARE A NAME (the exact shape that
# trips name-based resolution), runs the real commands, and asserts the
# printed symbols are the right ones.
#
# Invariants asserted (all on COMMAND OUTPUT, not raw SQL):
#   * who-calls <callee>      lists the real caller's qualified_name
#   * who-calls <unique-fn>   lists every real caller (cross-file)
#   * who-calls <Struct>      is NOT "no callers" and names a real
#                             referencing symbol — even though an Impl block
#                             shares the struct's name (resolution must land
#                             on the node that actually has the incoming
#                             USAGE edges)
#   * trace <caller> outgoing reaches the callee across files
#
# Black-box: drives the built binary only; touches no crate source.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
[[ "$GREPPY_BIN" = /* ]] || GREPPY_BIN="$WORKSPACE_ROOT/$GREPPY_BIN"

NAME="navigation"

require_bins "$GREPPY_BIN" || { emit_summary "$NAME"; exit 1; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/battle-nav-XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
CORPUS="$WORK/corpus"
export GREPPY_STORE_DIR="$WORK/store"

# ---------------------------------------------------------------------------
# Fixture: a tiny but exhaustively-known graph.
#
#   helper.rs:  fn do_it()                      <- the shared callee
#   widget.rs:  struct Widget {…}               <- name shared with…
#               impl Widget { new(); rank(); }  <- …this Impl block
#               fn make() -> Widget             <- uses Widget by TYPE_REF
#   lib.rs:     fn caller()  -> calls do_it(), Widget::new(), w.rank()
#                            -> USES Widget
#
# Known truths the assertions below depend on:
#   * `do_it` is called only by `caller`            (1 cross-file caller)
#   * `Widget` (the STRUCT) is referenced by `caller` (USES) and by
#     `make` (TYPE_REF + USES)                        (>=2 referrers)
#   * the Impl named `Widget` has NO incoming usage edges — so resolving
#     "Widget" to the Impl would print "no callers" (the bug we guard).
# ---------------------------------------------------------------------------
mkdir -p "$CORPUS/src"
cat > "$CORPUS/Cargo.toml" <<'EOF'
[package]
name = "nav_corpus"
version = "0.0.0"
edition = "2021"

[lib]
path = "src/lib.rs"
EOF

cat > "$CORPUS/src/lib.rs" <<'EOF'
pub mod helper;
pub mod widget;

use crate::helper::do_it;
use crate::widget::Widget;

pub fn caller() {
    do_it();
    let w = Widget::new(1);
    let _ = w.rank();
}
EOF

cat > "$CORPUS/src/helper.rs" <<'EOF'
pub fn do_it() -> u64 {
    42
}
EOF

cat > "$CORPUS/src/widget.rs" <<'EOF'
pub struct Widget {
    pub id: u64,
}

impl Widget {
    pub fn new(id: u64) -> Self {
        Widget { id }
    }

    pub fn rank(&self) -> u64 {
        self.id
    }
}

pub fn make() -> Widget {
    Widget::new(0)
}
EOF

git_init_corpus "$CORPUS"
[[ -d "$CORPUS/.git" ]]; check $? "fixture corpus is a git repo"

# ---- index ---------------------------------------------------------------
idx_log="$WORK/index.log"
"$GREPPY_BIN" index "$CORPUS" >"$idx_log" 2>&1
check $? "index fixture (exit 0)"
if grep -qiE 'panic|thread .* panicked' "$idx_log"; then
    fail "no panic during index"
    sed -n '1,20p' "$idx_log"
else
    pass "no panic during index"
fi

# Helper: run a navigation command from inside the corpus (so repo-root
# detection finds the store) and capture stdout.
nav() {
    ( cd "$CORPUS" && "$GREPPY_BIN" "$@" ) 2>>"$WORK/nav.err"
}

# ---- who-calls <callee> : the cross-file caller is named ------------------
wc_out="$(nav who-calls do_it)"
echo "[navigation] who-calls do_it ->"; echo "$wc_out" | sed 's/^/    /'
if [[ "$wc_out" == "no callers" ]]; then
    fail "who-calls do_it finds its caller (got 'no callers')"
elif grep -q 'caller' <<<"$wc_out"; then
    pass "who-calls do_it names the real caller 'caller'"
else
    fail "who-calls do_it names the real caller 'caller' (got: $wc_out)"
fi
# The caller lives in lib.rs and do_it in helper.rs: this is a CROSS-FILE
# caller, the hard case. Assert the caller's file is surfaced.
if grep -q 'lib.rs' <<<"$wc_out"; then
    pass "who-calls do_it surfaces the cross-file caller's location (lib.rs)"
else
    fail "who-calls do_it surfaces the cross-file caller's location (lib.rs) (got: $wc_out)"
fi

# ---- who-calls <Struct> : THE regression guard ---------------------------
# `Widget` names BOTH a Struct and an Impl. The incoming USAGE edges all land
# on the Struct; the Impl has none. If symbol resolution picks the Impl, the
# command prints exactly "no callers" for a symbol that is very much used.
# This is the exact bug the DB-only harness missed.
fu_out="$(nav who-calls Widget)"
echo "[navigation] who-calls Widget ->"; echo "$fu_out" | sed 's/^/    /'
if [[ "$fu_out" == "no callers" ]]; then
    fail "who-calls Widget is NOT 'no callers' — Widget IS used (resolution bug: resolved to the Impl, not the Struct)"
else
    pass "who-calls Widget is NOT 'no callers'"
fi
# The 0.3.0 output omits edge-kind labels, so pin the real incoming USAGE
# result itself: `make` at its source location. Resolving to the Impl returns
# "no callers" and therefore fails both this assertion and the one above.
if grep -qE '^src/widget\.rs:[0-9]+  make$' <<<"$fu_out"; then
    pass "who-calls Widget names referrer 'make' at its source location"
else
    fail "who-calls Widget names referrer 'make' at its source location (got: $fu_out)"
fi

# Counter-cases pin both distinct empty outcomes in the 0.3.0 surface: a known
# symbol without incoming edges says "no callers"; an absent symbol says
# "no symbol `S`". The fixture's top-level `caller` function is uncalled.
nc_out="$(nav who-calls caller)"
if [[ "$nc_out" == "no callers" ]]; then
    pass "who-calls on an uncalled symbol reports exactly 'no callers'"
else
    fail "who-calls on an uncalled symbol reports exactly 'no callers' (got: $nc_out)"
fi
nu_out="$(nav who-calls this_symbol_does_not_exist_anywhere)"
if [[ "$nu_out" == 'no symbol `this_symbol_does_not_exist_anywhere`' ]]; then
    pass "who-calls on an absent symbol reports the exact no-symbol message"
else
    fail "who-calls on an absent symbol reports the exact no-symbol message (got: $nu_out)"
fi

# ---- trace <caller> outgoing : reaches the callee across files ------------
tr_out="$(nav trace --symbol caller --direction outgoing --edge "" --depth 3)"
echo "[navigation] trace caller outgoing ->"; echo "$tr_out" | sed 's/^/    /'
if grep -q '(symbol not found)' <<<"$tr_out"; then
    fail "trace resolves the start symbol 'caller'"
else
    pass "trace resolves the start symbol 'caller'"
fi
# The trace must start at caller (depth=0) and reach do_it (the callee)
# via a CALLS edge at depth=1 — a cross-file traversal.
if grep -q 'do_it' <<<"$tr_out"; then
    pass "trace caller (outgoing) reaches the callee 'do_it'"
else
    fail "trace caller (outgoing) reaches the callee 'do_it' (got: $tr_out)"
fi
if grep -qE 'via CALLS' <<<"$tr_out"; then
    pass "trace caller traverses a real CALLS edge"
else
    fail "trace caller traverses a real CALLS edge (got: $tr_out)"
fi

emit_summary "$NAME"
