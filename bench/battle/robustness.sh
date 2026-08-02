#!/usr/bin/env bash
# ROBUSTNESS battle — lock in the field bugs found in the 2026-07 hardening
# batch so they can never silently regress. Every assertion is on COMMAND
# OUTPUT of the built binary (black-box), never on raw SQLite.
#
# Covered:
#   O9  parent .gitignore must NOT gut a nested repo's index
#   P10 who-calls must not answer a false "no callers" for a call-only
#       symbol that it demonstrably links
#   P4  who-calls --code carries the caller location and call-site source
#
# Black-box: drives the built binary only; touches no crate source.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
[[ "$GREPPY_BIN" = /* ]] || GREPPY_BIN="$WORKSPACE_ROOT/$GREPPY_BIN"

NAME="robustness"
require_bins "$GREPPY_BIN" || { emit_summary "$NAME"; exit 1; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/battle-robust-XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
export GREPPY_STORE_DIR="$WORK/store"
export GREPPY_AUTO_REINDEX=1

# ---------------------------------------------------------------------------
# O9 — a nested git repo under a parent whose .gitignore has an unanchored
# `*/` must still index all its own subdirectory files.
# ---------------------------------------------------------------------------
PARENT="$WORK/vendored"
REPO="$PARENT/dep"
mkdir -p "$REPO/src"
printf '*/\n' > "$PARENT/.gitignore"          # would hide every subdir
mkdir -p "$REPO/.git"                           # mark REPO as its own repo
cat > "$REPO/src/lib.rs" <<'RS'
pub fn wrap_helper() { let _ = crate::inner::do_inner(); }
RS
mkdir -p "$REPO/src/inner"
cat > "$REPO/src/inner/mod.rs" <<'RS'
pub fn do_inner() -> i32 { 42 }
RS

"$GREPPY_BIN" index "$REPO" --root "$REPO" >/dev/null 2>&1
who_out="$("$GREPPY_BIN" who-calls do_inner --code --root "$REPO" 2>&1)"
if grep -q "wrap_helper" <<<"$who_out"; then
    pass "O9: nested-repo src/ files indexed (who-calls do_inner finds wrap_helper)"
else
    fail "O9: parent '*/' gutted the nested index — who-calls do_inner missed wrap_helper: $who_out"
fi

# P4 — 0.3.0 prints the caller address and, with --code, its source span on
# the following line. Together they preserve the call-site evidence guard.
if grep -Eq '^src/lib\.rs:[0-9]+  wrap_helper$' <<<"$who_out" \
    && grep -q '^pub fn wrap_helper() { let _ = crate::inner::do_inner(); }$' <<<"$who_out"; then
    pass "P4: who-calls --code prints caller location and call-site source"
else
    fail "P4: who-calls --code missing caller location or call-site source: $who_out"
fi

# ---------------------------------------------------------------------------
# P10 — who-calls of a call-only symbol must not answer "no callers" when it
# links the symbol. Same fixture: do_inner is only ever called.
# ---------------------------------------------------------------------------
fu_out="$("$GREPPY_BIN" who-calls do_inner --root "$REPO" 2>&1)"
if [[ "$fu_out" == "no callers" ]]; then
    fail "P10: who-calls answered a false 'no callers' for a linked call target: $fu_out"
elif grep -qE '^src/lib\.rs:[0-9]+  wrap_helper$' <<<"$fu_out"; then
    pass "P10: who-calls reports the known call reference (no false zero)"
else
    fail "P10: who-calls did not report the known reference: $fu_out"
fi

emit_summary "$NAME"
