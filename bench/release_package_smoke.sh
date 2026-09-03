#!/usr/bin/env bash
# End-to-end acceptance for an unpacked Unix release artifact.
#
# The script is copied verbatim into the release tarball (see
# .github/workflows/release.yml) and must stay self-contained: every fixture
# is generated inline, and only POSIX-ish tooling that exists on the ubuntu
# and macos runners is used (bash 3.2+, jq, cmp, find, shasum or
# sha256sum).

set -euo pipefail

BIN="${1:?usage: release_package_smoke.sh /path/to/greppy [work-dir]}"
# Sections cd into fixture dirs; a relative binary path would break there.
case "$BIN" in /*) ;; *) BIN="$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")" ;; esac
[ -x "$BIN" ] || { echo "not executable: $BIN" >&2; exit 64; }
BIN_DIR="$(cd "$(dirname "$BIN")" && pwd)"
WEB_RUNTIME_DIST="$BIN_DIR/web-runtime"
[ -f "$WEB_RUNTIME_DIST/.greppy-web-runtime-dist" ] \
  || { echo "missing packaged web-runtime dist beside $BIN" >&2; exit 64; }
[ -x "$WEB_RUNTIME_DIST/bin/web-runtime" ] \
  || { echo "missing packaged web-runtime executable beside $BIN" >&2; exit 64; }
WORK="${2:-$(mktemp -d "${TMPDIR:-/tmp}/greppy-release-smoke-XXXXXX")}"
mkdir -p "$WORK/repo/src" "$WORK/repo/.git" "$WORK/store"
# A 0.3.1 data root may contain these now-unmanaged namespaces. They must not
# shadow the packaged model assets or prevent the managed v2 workspace from
# being created beside them.
mkdir -p \
  "$WORK/store/embedded-model" \
  "$WORK/store/models/v1/unmanaged-old" \
  "$WORK/store/workspaces/v2/unmanaged-old"
printf '%s\n' 'legacy-0.3.1-cache' >"$WORK/store/embedded-model/legacy.marker"
printf '%s\n' 'stale-model-placeholder' >"$WORK/store/models/v1/unmanaged-old/model.gguf"
printf '%s\n' 'stale-workspace-placeholder' >"$WORK/store/workspaces/v2/unmanaged-old/graph.db"

section() { printf '\n=== %s ===\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }

# Pick one SHA-256 tool for both runner OSes (ubuntu: sha256sum, macos: shasum).
if command -v sha256sum >/dev/null 2>&1; then
  HASH_CMD="sha256sum"
else
  HASH_CMD="shasum -a 256"
fi

# Canonical content digest of a directory tree: relative path + content hash
# of every regular file, sorted, hashed again. Captures creations, deletions,
# renames, and rewrites; deliberately ignores mtimes.
dir_digest() {
  (
    cd "$1" && find . -type f | LC_ALL=C sort | while IFS= read -r f; do
      $HASH_CMD "$f"
    done
  ) | $HASH_CMD | awk '{print $1}'
}

# Inference daemons (crates/cli/src/embed_daemon.rs / summarize_daemon.rs)
# outlive the CLI call that spawned them (GREPPY_*_DAEMON_EXIT_TTL_S). Cache
# purity assertions must not race socket teardown. Stable `.owner` lock files
# are deliberately retained after flock unlock so another process that already
# opened the same inode cannot race an unlink/recreate cycle. `cache clear`
# below detects a lock that is still held and retries on exit 75. A global
# pgrep would couple the release to unrelated agents.
dump_daemon_residue() {
  local now path mtime kind size age
  now="$(date +%s)"
  printf 'daemon drain diagnostics: embed_exit_ttl_s=%s summarize_exit_ttl_s=%s runtime_base=%s\n' \
    "${GREPPY_EMBED_DAEMON_EXIT_TTL_S:-<unset>}" \
    "${GREPPY_SUMMARIZE_DAEMON_EXIT_TTL_S:-<unset>}" \
    "$RUNTIME_BASE" >&2
  while IFS= read -r path; do
    if mtime="$(stat -f '%m' "$path" 2>/dev/null)"; then
      kind="$(stat -f '%HT' "$path" 2>/dev/null || printf unknown)"
      size="$(stat -f '%z' "$path" 2>/dev/null || printf unknown)"
    else
      mtime="$(stat -c '%Y' "$path" 2>/dev/null || printf 0)"
      kind="$(stat -c '%F' "$path" 2>/dev/null || printf unknown)"
      size="$(stat -c '%s' "$path" 2>/dev/null || printf unknown)"
    fi
    case "$mtime" in ''|*[!0-9]*) age=unknown ;; *) age=$(( now - mtime )) ;; esac
    printf 'daemon drain residue: path=%s type=%s age_s=%s size=%s\n' \
      "$path" "$kind" "$age" "$size" >&2
  done < <(find "$RUNTIME_BASE" \( -type f -o -type s -o -type l \) -print 2>/dev/null)
}

drain_daemons() {
  local deadline=$(( $(date +%s) + 180 ))
  while find "$RUNTIME_BASE" -type s -print -quit 2>/dev/null | grep -q .; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
      dump_daemon_residue
      fail "inference daemons did not exit within 180s"
    fi
    sleep 1
  done
}

cat >"$WORK/repo/src/lib.rs" <<'RS'
pub fn apply_limit(value: i32) -> i32 { value.clamp(0, 100) }
pub fn process_value(value: i32) -> i32 { apply_limit(value) }
pub fn normalize_score(value: i32) -> i32 { value.max(0) }
pub fn validate_score(value: i32) -> bool { value <= 100 }
pub fn default_score() -> i32 { 50 }
pub fn minimum_score() -> i32 { 0 }
pub fn maximum_score() -> i32 { 100 }
RS

# serde-shaped fixture: mirrors serde_derive/src/internals/{case,attr}.rs so
# the exact-hit assertions exercise the symbols the public benchmarks use
# (bench/agent_efficiency, bench/runtime_footprint.py: apply_to_field).
cat >"$WORK/repo/src/case.rs" <<'RS'
#[derive(Copy, Clone)]
pub enum RenameRule {
    LowerCase,
    UpperCase,
    SnakeCase,
}

impl RenameRule {
    /// Apply a rename case rule to a struct field name.
    pub fn apply_to_field(self, field: &str) -> String {
        match self {
            RenameRule::LowerCase => field.to_lowercase(),
            RenameRule::UpperCase => field.to_uppercase(),
            RenameRule::SnakeCase => field.to_string(),
        }
    }
}

pub struct RenameAllRules {
    pub serialize: RenameRule,
    pub deserialize: RenameRule,
}

pub struct Name {
    pub serialize: String,
    pub deserialize: String,
}

impl Name {
    /// Rename the serialize and deserialize names by the container rules.
    pub fn rename_by_rules(&mut self, rules: &RenameAllRules) {
        self.serialize = rules.serialize.apply_to_field(&self.serialize);
        self.deserialize = rules.deserialize.apply_to_field(&self.deserialize);
    }

    /// Return the field name used when serializing.
    pub fn serialize_name(&self) -> &str {
        &self.serialize
    }
}
RS

export GREPPY_STORE_DIR="$WORK/store"
export GREPPY_EMBED_DAEMON_MODEL_TTL_S=5
export GREPPY_EMBED_DAEMON_EXIT_TTL_S=15
export GREPPY_SUMMARIZE_DAEMON_MODEL_TTL_S=5
export GREPPY_SUMMARIZE_DAEMON_EXIT_TTL_S=15
# Contain daemon sockets (crates/cli/src/inference_daemon.rs
# unix_runtime_dir(): $XDG_RUNTIME_DIR/greppy when the joined path fits a
# unix-socket-safe 32 chars, else /tmp/greppy-daemon-$UID) in a directory this
# script owns, so the install/removal section can assert socket residue.
RUNTIME_BASE="/tmp/gsr-$$"
mkdir -p -m 700 "$RUNTIME_BASE"
export XDG_RUNTIME_DIR="$RUNTIME_BASE"

# --- baseline: doctor, index, JSON brief/semantic-search, expand ------------
section "baseline: doctor, index, JSON brief + semantic-search + expand"

"$BIN" --help >/dev/null
"$BIN" web doctor --json >"$WORK/web-doctor.json"
jq -e '
  .status == "ok" and
  (.result.executable | endswith("/web-runtime/bin/web-runtime")) and
  (.result.stamp | endswith("/web-runtime/.greppy-web-runtime-dist"))
' "$WORK/web-doctor.json" >/dev/null
"$BIN" --device cpu --root "$WORK/repo" doctor --json >"$WORK/doctor.json" || test $? -eq 1
jq -e '.command == "doctor" and .inference.registry.selected_backend == "cpu"' "$WORK/doctor.json" >/dev/null

"$BIN" --device cpu --root "$WORK/repo" index "$WORK/repo" >"$WORK/index.txt"
"$BIN" --device cpu --root "$WORK/repo" where-am-i >"$WORK/where-am-i.txt"
test -s "$WORK/where-am-i.txt" || fail "packaged legacy-cache index: where-am-i returned no repository overview"
"$BIN" --device cpu --root "$WORK/repo" brief apply_limit --json >"$WORK/brief.json"
jq -e '
  .schema_version == "greppy.brief.v1" and
  .status == "ok" and
  (.definitions | length) >= 1 and
  (.definitions[0].end_line >= .definitions[0].start_line) and
  (.definitions[0].signature | type == "string" and length > 0) and
  (.definitions[0].summary | length) >= 1 and
  (.expand_id | type == "string" and length > 0)
' "$WORK/brief.json" >/dev/null
brief_expand="$(jq -r '.expand_id' "$WORK/brief.json")"
"$BIN" --root "$WORK/repo" expand "$brief_expand" --json >"$WORK/brief-expand.json"
jq -e --arg id "$brief_expand" '.id == $id and (.payload_text | contains("apply_limit"))' "$WORK/brief-expand.json" >/dev/null

"$BIN" --device cpu --root "$WORK/repo" search --json \
  "restrict a numeric value to an allowed range" >"$WORK/semantic.json"
jq -e '
  .schema_version == "greppy.semantic-search.v1" and
  .status == "ok" and
  (.hits | length) >= 1 and
  (all(.hits[]; (.end_line >= .start_line) and (.signature | type == "string" and length > 0))) and
  (any(.hits[]; (.summary | length) >= 1)) and
  (.expand_id | type == "string" and length > 0)
' "$WORK/semantic.json" >/dev/null
semantic_expand="$(jq -r '.expand_id' "$WORK/semantic.json")"
"$BIN" --root "$WORK/repo" expand "$semantic_expand" --json >"$WORK/semantic-expand.json"
semantic_omitted="$(jq -r '.omitted' "$WORK/semantic.json")"
jq -e --arg id "$semantic_expand" --argjson omitted "$semantic_omitted" '
  .id == $id and
  (.payload_text | length > 0) and
  .payload_json.further_hits == $omitted and
  (.payload_json.hits | length) == $omitted
' "$WORK/semantic-expand.json" >/dev/null

# --- text output mode: prescribed shape and deterministic ordering ----------
# Contracts under test:
# * brief (text): BriefRender prints the generated purpose, compact locator,
#   source span, and an aggregated `called by ...` tail in that order.
# * search (text): print_search_meaning_rows prints one compact
#   `file:start  symbol — purpose` row per hit. Text mode intentionally shows
#   up to eight hits while JSON exposes a smaller display window plus expand.
# * Hit ordering: crates/store/src/vector_embedding.rs vector_search_exact:
#   "Ranking is total and deterministic: score descending, then
#   `qualified_name`, then row id." JSON's displayed rows must therefore be a
#   prefix of text mode, and JSON scores must be non-increasing.
section "text output mode: prescribed shape and deterministic ordering"

"$BIN" --device cpu --root "$WORK/repo" brief apply_limit >"$WORK/brief.txt"
first_match_line() { grep -n "$1" "$2" | head -1 | cut -d: -f1 || true; }
purpose_line="$(first_match_line '^.*range 0 to 100\.$' "$WORK/brief.txt")"
locator_line="$(first_match_line '^src/lib.rs:1$' "$WORK/brief.txt")"
source_line="$(first_match_line '^pub fn apply_limit(value: i32) -> i32 { value.clamp(0, 100) }$' "$WORK/brief.txt")"
caller_line="$(first_match_line '^called by process_value$' "$WORK/brief.txt")"
[ -n "$purpose_line" ] || fail "brief text: missing generated apply_limit purpose"
[ -n "$locator_line" ] || fail "brief text: missing compact src/lib.rs:1 locator"
[ -n "$source_line" ] || fail "brief text: missing apply_limit source span"
[ -n "$caller_line" ] || fail "brief text: missing aggregated caller tail"
[ "$purpose_line" -lt "$locator_line" ] || fail "brief text: purpose must precede locator"
[ "$locator_line" -lt "$source_line" ] || fail "brief text: locator must precede source"
[ "$source_line" -lt "$caller_line" ] || fail "brief text: source must precede caller tail"

# JSON scores must be non-increasing (the ranked half of the contract).
jq -e '[.hits[].score] | . == (sort | reverse)' "$WORK/semantic.json" >/dev/null \
  || fail "semantic-search JSON: hit scores are not in descending order"

semantic_locs_from_text() {
  awk -F '  ' '$1 ~ /^[^ ]+:[0-9]+$/ && NF >= 2 { print $1 }' "$1"
}

"$BIN" --device cpu --root "$WORK/repo" search \
  "restrict a numeric value to an allowed range" >"$WORK/semantic.txt"
semantic_locs_from_text "$WORK/semantic.txt" >"$WORK/semantic-locs-text.txt"
[ -s "$WORK/semantic-locs-text.txt" ] || fail "semantic-search text: no hit locator lines found"
grep -Eq '^src/[a-z_]+\.rs:[0-9]+$' "$WORK/semantic-locs-text.txt" \
  || fail "semantic-search text: locator lines do not look like file:start"

# JSON's shorter display window must be the prefix of text's ranked rows.
jq -r '.hits[] | "\(.file):\(.start_line)"' \
  "$WORK/semantic.json" >"$WORK/semantic-locs-json.txt"
head -n "$(wc -l <"$WORK/semantic-locs-json.txt" | tr -d ' ')" \
  "$WORK/semantic-locs-text.txt" >"$WORK/semantic-locs-text-prefix.txt"
cmp -s "$WORK/semantic-locs-text-prefix.txt" "$WORK/semantic-locs-json.txt" \
  || { diff -u "$WORK/semantic-locs-json.txt" "$WORK/semantic-locs-text-prefix.txt" >&2 || true; \
       fail "semantic-search: JSON rows are not a prefix of ranked text rows"; }

# Repeating the query must reproduce the same ordering (determinism).
"$BIN" --device cpu --root "$WORK/repo" search \
  "restrict a numeric value to an allowed range" >"$WORK/semantic-rerun.txt"
semantic_locs_from_text "$WORK/semantic-rerun.txt" >"$WORK/semantic-locs-rerun.txt"
cmp -s "$WORK/semantic-locs-text.txt" "$WORK/semantic-locs-rerun.txt" \
  || fail "semantic-search text: hit ordering is not deterministic across reruns"

# --- exact serde-repo hits ---------------------------------------------------
# The serde-shaped fixture (src/case.rs above) must be resolvable exactly:
# `brief SYMBOL` resolves symbol names via the graph, so each of the three
# serde symbols must come back as a definition, and a targeted semantic query
# must surface each symbol among the retrieved hits (shown hits + the
# expand-pack remainder = the full ranked retrieval set).
section "exact serde-repo hits: apply_to_field, rename_by_rules, serialize_name"

assert_brief_exact() {
  local symbol="$1"
  "$BIN" --device cpu --root "$WORK/repo" brief "$symbol" --json >"$WORK/brief-$symbol.json"
  jq -e --arg sym "$symbol" '
    .status == "ok" and
    ([.definitions[].qualified_name] | any(contains($sym))) and
    ([.definitions[].file] | any(. == "src/case.rs"))
  ' "$WORK/brief-$symbol.json" >/dev/null \
    || fail "brief $symbol: expected an exact definition hit in src/case.rs"
}
assert_brief_exact apply_to_field
assert_brief_exact rename_by_rules
assert_brief_exact serialize_name

assert_semantic_retrieves() {
  local symbol="$1"
  local query="$2"
  local out="$WORK/semantic-$symbol.json"
  "$BIN" --device cpu --root "$WORK/repo" search --json "$query" >"$out"
  jq -e '.status == "ok" and (.hits | length) >= 1' "$out" >/dev/null \
    || fail "semantic-search '$query': expected status ok with hits"
  jq -r '.hits[].qualified_name' "$out" >"$WORK/semantic-$symbol-names.txt"
  local expand_id
  expand_id="$(jq -r '.expand_id // empty' "$out")"
  if [ -n "$expand_id" ]; then
    "$BIN" --root "$WORK/repo" expand "$expand_id" --json \
      | jq -r '.payload_json.hits[].qualified_name' >>"$WORK/semantic-$symbol-names.txt"
  fi
  grep -q "$symbol" "$WORK/semantic-$symbol-names.txt" \
    || fail "semantic-search '$query': $symbol not in retrieved hit set: $(tr '\n' ' ' <"$WORK/semantic-$symbol-names.txt")"
}
assert_semantic_retrieves apply_to_field "apply a rename case rule to a struct field"
assert_semantic_retrieves rename_by_rules "rename the serialize and deserialize names using the container rules"
assert_semantic_retrieves serialize_name "return the field name used when serializing"

# --- text/JSON ranked-prefix parity -----------------------------------------
# Text shows up to eight compact rows; JSON shows a smaller display window and
# puts the remaining retrieved hits behind expand. The displayed JSON rows
# must be the prefix of the text ranking for the same query.
section "text/JSON parity: JSON display is a prefix of text ranking"

parity_query="apply a rename case rule to a struct field"
"$BIN" --device cpu --root "$WORK/repo" search "$parity_query" >"$WORK/parity.txt"
"$BIN" --device cpu --root "$WORK/repo" search --json "$parity_query" >"$WORK/parity.json"
jq -e '.status == "ok"' "$WORK/parity.json" >/dev/null
semantic_locs_from_text "$WORK/parity.txt" | LC_ALL=C sort >"$WORK/parity-locs-text.txt"
jq -r '.hits[] | "\(.file):\(.start_line)"' \
  "$WORK/parity.json" >"$WORK/parity-locs-json.txt"
[ -s "$WORK/parity-locs-text.txt" ] || fail "parity: text mode returned no hits"
semantic_locs_from_text "$WORK/parity.txt" \
  | head -n "$(wc -l <"$WORK/parity-locs-json.txt" | tr -d ' ')" \
  >"$WORK/parity-locs-text-prefix.txt"
cmp -s "$WORK/parity-locs-text-prefix.txt" "$WORK/parity-locs-json.txt" \
  || { diff -u "$WORK/parity-locs-json.txt" "$WORK/parity-locs-text-prefix.txt" >&2 || true; \
       fail "parity: JSON rows are not a prefix of text rows"; }

# --- byte-exact grep passthrough without cache side effects ------------------
# Contract (crates/cli/src/lib.rs run_os): passthrough detection runs BEFORE
# the throttled cache-maintenance pass, "so an ordinary grep invocation cannot
# touch Greppy state"; dispatch_grep_os forwards argv verbatim to the real
# grep (crates/greppy/src/lib.rs run_grep_os) with inherited stdio. Therefore
# every pure grep call must be byte-identical to system grep (stdout, stderr,
# exit code) and must leave the cache directory content-identical.
section "grep passthrough: byte-exact vs system grep, no cache side effects"

# Resolve the comparison grep the way the product's tier-2 discovery does
# (crates/greppy/src/lib.rs discover_grep): fixed system paths, NEVER `command
# -v grep` — a shimmed PATH can point "grep" at a greppy wrapper, and pinning
# that via GREPPY_REAL_GREP would recurse the passthrough into a fork bomb.
REAL_GREP=""
for candidate in /usr/bin/grep /bin/grep; do
  if [ -x "$candidate" ]; then REAL_GREP="$candidate"; break; fi
done
[ -n "$REAL_GREP" ] || fail "no system grep at /usr/bin/grep or /bin/grep for the passthrough comparison"
# Pin the wrapper to the same grep we compare against (tier-1 discovery in
# discover_grep honours GREPPY_REAL_GREP).
export GREPPY_REAL_GREP="$REAL_GREP"

assert_grep_pair() {
  local label="$1"; shift
  local expected_rc="$1"; shift
  local rc=0 expected_rc_actual=0
  ( cd "$WORK/repo" && "$BIN" "$@" ) >"$WORK/grep-actual.out" 2>"$WORK/grep-actual.err" || rc=$?
  ( cd "$WORK/repo" && "$REAL_GREP" "$@" ) >"$WORK/grep-expected.out" 2>"$WORK/grep-expected.err" || expected_rc_actual=$?
  [ "$expected_rc_actual" -eq "$expected_rc" ] \
    || fail "grep pair $label: system grep exited $expected_rc_actual, test expected $expected_rc (bad test fixture)"
  [ "$rc" -eq "$expected_rc_actual" ] \
    || fail "grep pair $label: exit code diverges (greppy=$rc grep=$expected_rc_actual)"
  cmp -s "$WORK/grep-actual.out" "$WORK/grep-expected.out" \
    || { diff -u "$WORK/grep-expected.out" "$WORK/grep-actual.out" | head -20 >&2 || true; \
         fail "grep pair $label: stdout diverges from system grep"; }
  cmp -s "$WORK/grep-actual.err" "$WORK/grep-expected.err" \
    || { diff -u "$WORK/grep-expected.err" "$WORK/grep-actual.err" | head -20 >&2 || true; \
         fail "grep pair $label: stderr diverges from system grep"; }
}

drain_daemons
store_digest_before="$(dir_digest "$GREPPY_STORE_DIR")"

assert_grep_pair "match -n"          0 -n apply_limit src/lib.rs
assert_grep_pair "match -c"          0 -c fn src/lib.rs
assert_grep_pair "match -nH multi"   0 -nH serialize src/lib.rs src/case.rs
assert_grep_pair "match -E regex"    0 -En 'pub fn [a-z_]+' src/lib.rs
assert_grep_pair "match -r recurse"  0 -rn --include='*.rs' serialize_name src
assert_grep_pair "miss rc=1"         1 -n definitely_absent_token src/lib.rs
assert_grep_pair "missing file rc=2" 2 -n apply_limit src/no_such_file.rs

# Explicit `greppy grep …` subcommand strips the leading `grep` placeholder
# (dispatch_grep_os) — compare against system grep WITHOUT that token.
rc=0; ( cd "$WORK/repo" && "$BIN" grep -n rename_by_rules src/case.rs ) >"$WORK/grep-actual.out" 2>"$WORK/grep-actual.err" || rc=$?
erc=0; ( cd "$WORK/repo" && "$REAL_GREP" -n rename_by_rules src/case.rs ) >"$WORK/grep-expected.out" 2>"$WORK/grep-expected.err" || erc=$?
[ "$rc" -eq "$erc" ] || fail "grep pair explicit-sub: exit code diverges (greppy=$rc grep=$erc)"
cmp -s "$WORK/grep-actual.out" "$WORK/grep-expected.out" || fail "grep pair explicit-sub: stdout diverges"
cmp -s "$WORK/grep-actual.err" "$WORK/grep-expected.err" || fail "grep pair explicit-sub: stderr diverges"

# stdin passthrough, byte-exact
printf 'alpha\nbeta\ngamma\n' >"$WORK/grep-stdin.txt"
rc=0; "$BIN" -n beta <"$WORK/grep-stdin.txt" >"$WORK/grep-actual.out" 2>"$WORK/grep-actual.err" || rc=$?
erc=0; "$REAL_GREP" -n beta <"$WORK/grep-stdin.txt" >"$WORK/grep-expected.out" 2>"$WORK/grep-expected.err" || erc=$?
[ "$rc" -eq "$erc" ] || fail "grep pair stdin: exit code diverges (greppy=$rc grep=$erc)"
cmp -s "$WORK/grep-actual.out" "$WORK/grep-expected.out" || fail "grep pair stdin: stdout diverges"
cmp -s "$WORK/grep-actual.err" "$WORK/grep-expected.err" || fail "grep pair stdin: stderr diverges"

store_digest_after="$(dir_digest "$GREPPY_STORE_DIR")"
[ "$store_digest_before" = "$store_digest_after" ] \
  || fail "grep passthrough mutated the cache directory ($GREPPY_STORE_DIR): digest $store_digest_before -> $store_digest_after"
[ ! -e "$WORK/repo/.greppy" ] || fail "grep passthrough created a .greppy sidecar in the repo"
unset GREPPY_REAL_GREP

# --- cache status / gc / clear -----------------------------------------------
# Contract: crates/cli/src/lib.rs dispatch_cache. status reports the data
# root (crates/core/src/cache.rs data_root(), here pinned by
# GREPPY_STORE_DIR) and the managed entries; gc respects the TTL (default 14
# days, so a store this fresh survives); clear --all --yes empties every
# verified workspace and model entry; clear without --yes must refuse with
# EXIT_USAGE (64) and change nothing.
section "cache subcommands: status, gc, clear"

"$BIN" --root "$WORK/repo" cache status --json >"$WORK/cache-status.json"
jq -e --arg root "$GREPPY_STORE_DIR" '
  .data_root == $root and
  .managed_bytes > 0 and
  ([.entries[] | select(.kind == "workspace")] | length) >= 1 and
  ([.entries[] | select(.kind == "workspace") | .workspace_root] | any(endswith("/repo")))
' "$WORK/cache-status.json" >/dev/null \
  || fail "cache status --json: data_root/managed workspace entry assertions failed"
"$BIN" --root "$WORK/repo" cache status >"$WORK/cache-status.txt"
head -1 "$WORK/cache-status.txt" | grep -qx "cache root: $GREPPY_STORE_DIR" \
  || fail "cache status text: first line must be 'cache root: $GREPPY_STORE_DIR'"

"$BIN" --root "$WORK/repo" cache gc --dry-run --json >"$WORK/cache-gc-dry.json"
jq -e '.dry_run == true and (.removed | length) == 0' "$WORK/cache-gc-dry.json" >/dev/null \
  || fail "cache gc --dry-run: expected a dry run that removes nothing (fresh entries, 14d TTL)"
"$BIN" --root "$WORK/repo" cache gc --json >"$WORK/cache-gc.json"
jq -e '.dry_run == false and (.removed | length) == 0' "$WORK/cache-gc.json" >/dev/null \
  || fail "cache gc: fresh store must survive a TTL/quota pass"
"$BIN" --root "$WORK/repo" cache status --json \
  | jq -e '[.entries[] | select(.kind == "workspace")] | length >= 1' >/dev/null \
  || fail "cache gc removed a fresh workspace entry"

# clear without --yes: refuse with EXIT_USAGE and leave the store intact.
rc=0
"$BIN" cache clear --all >"$WORK/cache-clear-noyes.txt" 2>&1 || rc=$?
[ "$rc" -eq 64 ] || fail "cache clear --all without --yes: expected exit 64, got $rc"
grep -q -- '--yes' "$WORK/cache-clear-noyes.txt" || fail "cache clear refusal must mention --yes"
# --all and --root are mutually exclusive.
rc=0
"$BIN" --root "$WORK/repo" cache clear --all --yes >"$WORK/cache-clear-both.txt" 2>&1 || rc=$?
[ "$rc" -eq 64 ] || fail "cache clear --all --yes --root: expected exit 64, got $rc"
"$BIN" cache status --json | jq -e '.managed_bytes > 0' >/dev/null \
  || fail "refused cache clear must not have removed anything"

# Real clear: exit 75 (EXIT_TEMPFAIL) means live daemon leases; drain first
# and allow a short grace loop, then require a clean 0.
drain_daemons
deadline=$(( $(date +%s) + 120 ))
while :; do
  rc=0
  "$BIN" cache clear --all --yes >"$WORK/cache-clear.txt" 2>&1 || rc=$?
  [ "$rc" -eq 0 ] && break
  [ "$rc" -eq 75 ] || { cat "$WORK/cache-clear.txt" >&2; fail "cache clear --all --yes: expected exit 0 or 75, got $rc"; }
  [ "$(date +%s)" -lt "$deadline" ] || fail "cache clear kept reporting locked entries after daemon drain"
  sleep 2
done
"$BIN" cache status --json >"$WORK/cache-status-cleared.json"
jq -e '.managed_bytes == 0 and (.entries | length) == 0' "$WORK/cache-status-cleared.json" >/dev/null \
  || fail "cache status after clear --all: expected zero managed bytes and no entries"

# Clear removes managed data only. The seeded 0.3.1-shaped namespaces are
# deliberately unmanaged and must survive byte-for-byte; no other workspace
# database or model blob may remain.
remaining_graphs="$(find "$GREPPY_STORE_DIR" -name 'graph.db' -print 2>/dev/null)"
[ "$remaining_graphs" = "$WORK/store/workspaces/v2/unmanaged-old/graph.db" ] \
  || fail "cache clear --all removed unmanaged legacy data or left a managed workspace database"

# Extracted embedded models are managed entries (model_entry_has_marker
# accepts the CLI's JSON extraction marker since the 2026-07-13 fix) and must
# be reclaimed by `cache clear --all --yes` like any other model entry.
remaining_models="$(find "$GREPPY_STORE_DIR" -name '*.gguf' -print 2>/dev/null)"
[ "$remaining_models" = "$WORK/store/models/v1/unmanaged-old/model.gguf" ] \
  || fail "cache clear --all removed unmanaged legacy data or left a managed model blob"

# --- simulated install + residue-free removal --------------------------------
# Install the packaged binary into a temp prefix (exactly what the tarball
# layout provides: a single `greppy` executable), run index+search end to end
# under a fresh HOME/TMPDIR, then remove the prefix and assert the product
# left NOTHING outside the two documented locations:
# * the platform cache root (crates/core/src/cache.rs data_root(): macOS
#   $HOME/Library/Application Support/greppy, Linux
#   $XDG_DATA_HOME|$HOME/.local/share/greppy) — legitimate, enumerated below;
# * the daemon runtime dir (crates/cli/src/inference_daemon.rs
#   unix_runtime_dir(), pinned above to $XDG_RUNTIME_DIR/greppy) whose socket
#   files must be gone once the daemons exit.
section "simulated install + residue-free removal"

PREFIX="$WORK/install-prefix"
FAKE_HOME="$WORK/fake-home"
FAKE_TMP="$WORK/fake-tmp"
mkdir -p "$PREFIX/bin" "$PREFIX/smoke-repo/src" "$PREFIX/smoke-repo/.git" "$FAKE_HOME" "$FAKE_TMP"
cp "$BIN" "$PREFIX/bin/greppy"
cp -a "$WEB_RUNTIME_DIST" "$PREFIX/bin/web-runtime"
chmod +x "$PREFIX/bin/greppy"
cp "$WORK/repo/src/lib.rs" "$WORK/repo/src/case.rs" "$PREFIX/smoke-repo/src/"

# macOS-vs-Linux guard: the documented default cache root differs per OS.
# CUDA-featured binaries additionally materialize their embedded GPU backend
# into the platform *cache* root (embed-native cuda_runtime_cache_root():
# $XDG_CACHE_HOME|$HOME/.cache/greppy on Linux; with GREPPY_STORE_DIR set it
# lands under the store instead, which is why only this env-stripped install
# phase sees it). That is the second documented location an uninstall must
# delete.
case "$(uname -s)" in
  Darwin)
    EXPECTED_CACHE="$FAKE_HOME/Library/Application Support/greppy"
    EXPECTED_RUNTIME_CACHE="$FAKE_HOME/Library/Caches/greppy"
    ;;
  Linux)
    EXPECTED_CACHE="$FAKE_HOME/.local/share/greppy"
    EXPECTED_RUNTIME_CACHE="$FAKE_HOME/.cache/greppy"
    ;;
  *)      fail "unsupported platform for the install/removal section: $(uname -s)" ;;
esac

# Snapshot the REAL environment so we can prove the sandboxed run did not
# leak into it: the user's cache root and greppy-named /tmp entries.
case "$(uname -s)" in
  Darwin) REAL_CACHE="$HOME/Library/Application Support/greppy" ;;
  Linux)  REAL_CACHE="${XDG_DATA_HOME:-$HOME/.local/share}/greppy" ;;
esac
real_cache_existed=0
[ -e "$REAL_CACHE" ] && real_cache_existed=1
STAMP="$WORK/install-stamp"
touch "$STAMP"
sleep 1  # ensure any leak is strictly newer than the stamp
ls -d /tmp/greppy* "${TMPDIR:-/tmp}"/greppy* 2>/dev/null | LC_ALL=C sort -u >"$WORK/tmp-before.txt" || true

# Run the installed binary with HOME/TMPDIR redirected and the store override
# removed, so it exercises the real default cache-root selection.
run_installed() {
  env -u GREPPY_STORE_DIR -u XDG_DATA_HOME -u XDG_CACHE_HOME \
    HOME="$FAKE_HOME" TMPDIR="$FAKE_TMP" \
    "$PREFIX/bin/greppy" "$@"
}
run_installed --device cpu --root "$PREFIX/smoke-repo" index "$PREFIX/smoke-repo" >"$WORK/install-index.txt"
run_installed --device cpu --root "$PREFIX/smoke-repo" search --json \
  "restrict a numeric value to an allowed range" >"$WORK/install-semantic.json"
jq -e '.status == "ok" and (.hits | length) >= 1' "$WORK/install-semantic.json" >/dev/null \
  || fail "installed binary: semantic-search returned no hits"

drain_daemons

# The product may leave exactly two things in the fake HOME: the documented
# data root and (cuda builds only) the runtime cache root, plus the bare
# ancestor directories needed to hold them.
[ -d "$EXPECTED_CACHE" ] || fail "installed binary did not create the documented cache root at $EXPECTED_CACHE"
find "$FAKE_HOME" -mindepth 1 | while IFS= read -r path; do
  case "$path" in
    "$EXPECTED_CACHE"|"$EXPECTED_CACHE"/*) ;;                  # documented data subtree
    "$EXPECTED_RUNTIME_CACHE"|"$EXPECTED_RUNTIME_CACHE"/*) ;;  # materialized GPU backend cache
    "$FAKE_HOME/.nv"|"$FAKE_HOME/.nv"/*) ;;  # NVIDIA driver ComputeCache - libcuda writes it during the GPU probe on GPU hosts; third-party, not product residue
    *)
      case "$EXPECTED_CACHE/:$EXPECTED_RUNTIME_CACHE/" in
        "$path"/*|*":$path/"*) [ -d "$path" ] || fail "unexpected non-directory ancestor in fake HOME: $path" ;;
        *) fail "unexpected residue in fake HOME outside the documented roots: $path" ;;
      esac
      ;;
  esac
done
# Nothing may be left in the redirected TMPDIR once the daemons exited.
[ -z "$(find "$FAKE_TMP" -mindepth 1 2>/dev/null)" ] \
  || fail "installed binary left residue in TMPDIR: $(find "$FAKE_TMP" -mindepth 1 | tr '\n' ' ')"
# Daemon sockets must be gone after drain. Empty, unlocked `.owner` files are
# the intentional stable flock inode; no other regular runtime residue is
# permitted.
while IFS= read -r path; do
  case "$path" in
    "$RUNTIME_BASE"/greppy/daemon-state/locks/*.owner)
      [ ! -s "$path" ] || fail "daemon owner lock is unexpectedly nonempty: $path"
      ;;
    *) fail "daemon runtime dir holds unexpected residue after drain: $path" ;;
  esac
done < <(find "$RUNTIME_BASE" \( -type f -o -type s \) -print 2>/dev/null)

# Removal: delete the prefix and both documented roots; nothing else of
# the product may remain anywhere in the fake HOME.
rm -rf "$PREFIX" "$EXPECTED_CACHE" "$EXPECTED_RUNTIME_CACHE"
[ -z "$(find "$FAKE_HOME" -type f 2>/dev/null)" ] \
  || fail "residue files remain in fake HOME after removal: $(find "$FAKE_HOME" -type f | tr '\n' ' ')"

# The real environment must be untouched: no new/updated real cache root, no
# new greppy-named /tmp entries.
if [ "$real_cache_existed" -eq 1 ]; then
  [ -z "$(find "$REAL_CACHE" -newer "$STAMP" -print 2>/dev/null | head -1)" ] \
    || fail "sandboxed install run modified the real cache root at $REAL_CACHE"
else
  [ ! -e "$REAL_CACHE" ] || fail "sandboxed install run created the real cache root at $REAL_CACHE"
fi
ls -d /tmp/greppy* "${TMPDIR:-/tmp}"/greppy* 2>/dev/null | LC_ALL=C sort -u >"$WORK/tmp-after.txt" || true
cmp -s "$WORK/tmp-before.txt" "$WORK/tmp-after.txt" \
  || { diff -u "$WORK/tmp-before.txt" "$WORK/tmp-after.txt" >&2 || true; \
       fail "sandboxed install run left new greppy entries under /tmp"; }
rm -rf "$RUNTIME_BASE"

printf '\nrelease package inference smoke passed: %s\n' "$BIN"
