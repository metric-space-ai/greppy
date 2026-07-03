#!/usr/bin/env bash
# MALFORMED INPUT battle — index a repo containing adversarial Rust
# files and assert the indexer degrades gracefully:
#   * no panic / no signal crash
#   * the process exits with a documented code (0 clean, 73 IO)
#   * the DB is created and passes integrity_check
#   * well-formed files in the SAME tree are still indexed (one bad file
#     does not poison the whole run)
#   * the report accounts for unreadable / unsupported files
#
# Adversarial contents:
#   * truncated Rust (unclosed brace / fn signature mid-token)
#   * invalid UTF-8 bytes inside a .rs file
#   * deeply nested expressions / blocks (parser stack-depth stress)
#   * empty .rs and whitespace-only .rs
#   * a file that is valid and should index cleanly (control)

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

NAME="malformed"
require_bins "$GREPPLUS_BIN" || { emit_summary "$NAME"; exit 1; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/battle-malformed-XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
CORPUS="$WORK/corpus"
mkdir -p "$CORPUS/src"
export GREPPLUS_STORE_DIR="$WORK/store"

# Control: a clean, well-formed module that MUST index.
cat > "$CORPUS/src/lib.rs" <<'EOF'
pub mod good;
pub mod truncated;
pub mod badutf8;
pub mod deep;
EOF

cat > "$CORPUS/src/good.rs" <<'EOF'
pub struct GoodStruct { pub n: u64 }
impl GoodStruct {
    pub fn new() -> Self { Self { n: 1 } }
    pub fn value(&self) -> u64 { self.n }
}
pub fn good_fn() -> u64 { GoodStruct::new().value() }
EOF

# Truncated: unclosed brace, dangling fn.
printf 'pub fn broken(\npub struct Half {\n    field: \n' > "$CORPUS/src/truncated.rs"

# Invalid UTF-8 inside a .rs file (lone 0xFF / truncated multibyte).
printf 'pub fn utf() {\n    let s = "\xff\xfe\xc3\x28";\n}\n' > "$CORPUS/src/badutf8.rs"

# Deeply nested expression — stress the parser stack.
{
    printf 'pub fn deep_nest() -> i64 {\n    '
    n=400
    for _ in $(seq 1 "$n"); do printf '('; done
    printf '1'
    for _ in $(seq 1 "$n"); do printf ' + 1)'; done
    printf '\n}\n'
} > "$CORPUS/src/deep.rs"

# Deeply nested blocks too.
{
    n=300
    printf 'pub fn deep_blocks() {\n'
    for _ in $(seq 1 "$n"); do printf '{ '; done
    printf 'let _x = 1;'
    for _ in $(seq 1 "$n"); do printf ' }'; done
    printf '\n}\n'
} > "$CORPUS/src/deepblocks.rs"

# Empty + whitespace-only .rs files.
: > "$CORPUS/src/empty.rs"
printf '   \n\n\t\n' > "$CORPUS/src/whitespace.rs"

git_init_corpus "$CORPUS"

echo "[malformed] indexing adversarial corpus ..."
log="$WORK/index.log"
( cd "$CORPUS" && "$GREPPLUS_BIN" index . ) >"$log" 2>&1
rc=$?
echo "[malformed] index rc=$rc"
echo "----- index output -----"; cat "$log"; echo "------------------------"

# Graceful exit: documented codes only, never a signal/panic crash.
if [[ "$rc" -lt 128 && ( "$rc" -eq 0 || "$rc" -eq 73 ) ]]; then
    pass "index exited gracefully (rc=$rc in documented set {0,73})"
else
    fail "index exited gracefully (rc=$rc — signal/panic/unknown)"
fi

if grep -qiE 'panic|thread .* panicked|stack overflow|RUST_BACKTRACE|SIGSEGV|SIGABRT' "$log"; then
    fail "no panic / stack-overflow in output"
    grep -iE 'panic|overflow|SIG' "$log" | head -5 | sed 's/^/    /'
else
    pass "no panic / stack-overflow in output"
fi

# The report must be present and account for files.
if grep -qE 'indexed [0-9]+ files' "$log"; then
    pass "indexer printed a file-accounting report"
else
    fail "indexer printed a file-accounting report"
fi

# Store + integrity.
DB="$(graph_db_path "$GREPPLUS_STORE_DIR")"
if [[ -z "$DB" ]]; then
    fail "graph.db created"
    emit_summary "$NAME"; exit 1
fi
pass "graph.db created"

integ=$(sqlite_q "$DB" "PRAGMA integrity_check;" 2>/dev/null || echo "ERR")
assert_eq "ok" "$integ" "DB integrity_check on malformed corpus"

# Resilience: the clean control file must still produce its symbols even
# though sibling files are malformed. We look for GoodStruct / good_fn.
good_nodes=$(sqlite_q "$DB" "SELECT count(*) FROM nodes WHERE name IN ('GoodStruct','good_fn','value','new');" 2>/dev/null || echo 0)
assert_ge "${good_nodes:-0}" 1 "well-formed sibling still indexed (one bad file does not poison the run)"

emit_summary "$NAME"
