#!/usr/bin/env bash
# DETERMINISM battle — the parallel indexing pipeline must be
# deterministic. Index the SAME corpus into two independent stores and
# assert:
#   * identical node count
#   * identical edge count
#   * identical edge SET (source qname, target qname, edge_type) —
#     the strongest form: not just counts but the exact relations.
#   * identical node SET (label, qualified_name, file_path)
#
# A second run into the SAME store is also checked for idempotency:
# re-indexing must not change the counts.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

NAME="determinism"
N_FILES="${BATTLE_DET_FILES:-120}"

require_bins "$GREPPLUS_BIN" || { emit_summary "$NAME"; exit 1; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/battle-det-XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
CORPUS="$WORK/corpus"

echo "[determinism] generating $N_FILES-file corpus ..."
bash "$BATTLE_DIR/gen_corpus.sh" "$CORPUS" "$N_FILES" >/dev/null 2>&1
git_init_corpus "$CORPUS"

dump_nodes() { sqlite_q "$1" "SELECT label||'|'||qualified_name||'|'||file_path FROM nodes ORDER BY 1;"; }
dump_edges() {
    sqlite_q "$1" "
      SELECT s.qualified_name||'|'||t.qualified_name||'|'||e.edge_type
      FROM edges e
      JOIN nodes s ON s.id=e.source_id
      JOIN nodes t ON t.id=e.target_id
      ORDER BY 1;"
}

index_into() {
    # $1 = store dir
    local store="$1"
    rm -rf "$store"
    ( cd "$CORPUS" && GREPPLUS_STORE_DIR="$store" "$GREPPLUS_BIN" index . ) \
        >/dev/null 2>&1
}

echo "[determinism] indexing run A ..."
index_into "$WORK/storeA"; check $? "run A indexed"
echo "[determinism] indexing run B ..."
index_into "$WORK/storeB"; check $? "run B indexed"

DBA="$(graph_db_path "$WORK/storeA")"
DBB="$(graph_db_path "$WORK/storeB")"
if [[ -z "$DBA" || -z "$DBB" ]]; then
    fail "both graph.db files exist"
    emit_summary "$NAME"; exit 1
fi
pass "both graph.db files exist"

nA=$(sqlite_q "$DBA" "SELECT count(*) FROM nodes;")
nB=$(sqlite_q "$DBB" "SELECT count(*) FROM nodes;")
assert_eq "$nA" "$nB" "node count identical across runs"

eA=$(sqlite_q "$DBA" "SELECT count(*) FROM edges;")
eB=$(sqlite_q "$DBB" "SELECT count(*) FROM edges;")
assert_eq "$eA" "$eB" "edge count identical across runs"

# Strong determinism: identical SETS, not just counts.
if diff <(dump_nodes "$DBA") <(dump_nodes "$DBB") >/dev/null; then
    pass "node SET byte-identical across runs"
else
    fail "node SET byte-identical across runs"
    diff <(dump_nodes "$DBA") <(dump_nodes "$DBB") | head -20
fi

if diff <(dump_edges "$DBA") <(dump_edges "$DBB") >/dev/null; then
    pass "edge SET byte-identical across runs"
else
    fail "edge SET byte-identical across runs"
    diff <(dump_edges "$DBA") <(dump_edges "$DBB") | head -20
fi

# Idempotency: re-index run A's store; counts must be stable.
echo "[determinism] re-indexing run A store (idempotency) ..."
( cd "$CORPUS" && GREPPLUS_STORE_DIR="$WORK/storeA" "$GREPPLUS_BIN" index . ) >/dev/null 2>&1
check $? "re-index exit code"
nA2=$(sqlite_q "$DBA" "SELECT count(*) FROM nodes;")
eA2=$(sqlite_q "$DBA" "SELECT count(*) FROM edges;")
assert_eq "$nA" "$nA2" "node count stable after re-index"
assert_eq "$eA" "$eA2" "edge count stable after re-index"

emit_summary "$NAME"
