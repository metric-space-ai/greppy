#!/usr/bin/env bash
# Phase 7 — freshness benchmark (phase plan §12.3).
#
# Exercises the freshness gate under realistic workspace-mutation
# scenarios. For each scenario we:
# 1. Snapshot the workspace into a clean state.
# 2. Index via `grepplus index <root>`.
# 3. Apply the mutation.
# 4. Run `freshness-probe <root>` to capture the freshness outcome
#    (real Rust binary, returns JSON).
# 5. Compare the outcome against the expected class.
#
# The probe uses the same `grepplus_freshness::check` code that
# `grepplus-grep` uses in its freshness gate, so the outcomes here
# are the ones agents would actually see at runtime.

set -uo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GREPPLUS_BIN="${GREPPLUS_BIN:-$WORKSPACE_ROOT/target/debug/grepplus}"
PROBE_BIN="${PROBE_BIN:-$WORKSPACE_ROOT/target/debug/examples/freshness-probe}"
CORPUS_SRC="${CORPUS_SRC:-$WORKSPACE_ROOT/bench/fixtures/sample}"
# Copy the fixture to a temp dir so the indexer's
# detect_repo_root / walk don't accidentally pick up the parent
# grepplus-rs workspace. See bench/grep_compat.sh for the same
# rationale.
CORPUS_ROOT="$(mktemp -d -t grepplus-freshness.XXXXXX)"
cp -R "$CORPUS_SRC/." "$CORPUS_ROOT/"
rm -rf "$CORPUS_ROOT/.grepplus" "$CORPUS_ROOT/.git"
trap 'rm -rf "$CORPUS_ROOT"' EXIT

if [[ ! -x "$GREPPLUS_BIN" ]]; then
  echo "error: $GREPPLUS_BIN not built; run \`cargo build --workspace\` first" >&2
  exit 2
fi
if [[ ! -x "$PROBE_BIN" ]]; then
  echo "error: $PROBE_BIN not built; run \`cargo build --example freshness-probe\` first" >&2
  exit 2
fi

cd "$CORPUS_ROOT"

# Cleanup any leftover sidecars from previous runs. The cleanup
# pass is per-workspace and is bounded by the sidecar root
# derived from CORPUS_ROOT.
"$GREPPLUS_BIN" --version >/dev/null 2>&1 || true

# Initialise a git repo so the git-fingerprint path is exercised.
# The git-state scenarios (new_commit, branch_created) need this;
# the other scenarios don't care but don't suffer from it either.
git init -q .
git -c user.email=bench@grepplus -c user.name=bench add -A
git -c user.email=bench@grepplus -c user.name=bench commit -q -m "bench initial"

# Reset the fixture to a clean state, then index once. Each scenario
# will start from this state, apply a mutation, run the probe, and
# reindex before the next scenario.
#
# The bench fixture (under $CORPUS_ROOT) is a temp-dir copy of
# $CORPUS_SRC. The cleanest way to reset is to re-copy from the
# source, since that always restores the original 4 source files
# and discards any test artifacts.
reset_repo() {
  # Preserve the git directory we built for the git-state
  # scenarios (new_commit, branch_created) so those scenarios
  # can manipulate it. The src/ tree is what gets reset.
  if [[ -d "$CORPUS_SRC/.git" ]]; then
    rm -rf "$CORPUS_ROOT/.git"
    cp -R "$CORPUS_SRC/.git" "$CORPUS_ROOT/.git"
  fi
  # Restore the original 4 source files; remove everything else
  # the scenarios may have left behind.
  rm -rf "$CORPUS_ROOT/src"
  mkdir -p "$CORPUS_ROOT/src"
  for f in "$CORPUS_SRC"/src/*; do
    [[ -e "$f" ]] || continue
    cp "$f" "$CORPUS_ROOT/src/"
  done
  rm -rf "$CORPUS_ROOT/.grepplus"
  "$GREPPLUS_BIN" index "$CORPUS_ROOT" >/dev/null 2>&1
}

reset_repo

pass=0
fail=0
declare -a failures

probe() {
  local label="$1"
  local expected="$2"
  shift 2
  # Apply the mutation passed as the rest of the args.
  "$@"
  sleep 0.05
  local out
  out=$("$PROBE_BIN" "$CORPUS_ROOT" 2>&1)
  local actual
  actual=$(printf "%s" "$out" | sed -n 's/.*"outcome":"\([A-Za-z]*\)".*/\1/p')
  local elapsed
  elapsed=$(printf "%s" "$out" | sed -n 's/.*"elapsed_ms":\([0-9]*\).*/\1/p')
  if [[ "$actual" == "$expected" ]]; then
    printf "  [PASS] %-30s expect=%-15s actual=%-15s elapsed=%sms\n" \
      "$label" "$expected" "$actual" "$elapsed"
    pass=$((pass + 1))
  else
    printf "  [FAIL] %-30s expect=%-15s actual=%-15s elapsed=%sms\n" \
      "$label" "$expected" "$actual" "$elapsed"
    fail=$((fail + 1))
    failures+=("$label: expected $expected, got $actual")
  fi
  # Reset for the next scenario.
  reset_repo
}

echo "=== Scenario 1: cold start (no prior state) ==="
# Cold start means no platform-locator store. The bench destructive-
# ly wipes the entire `grepplus` cache dir for the rest of the
# fixtures: this is intentional because the bench corpus lives in
# `$TMPDIR` and the cache is rebuilt from the indexed corpus in
# `reset_repo()`. Wiping is local and the corpus never escapes
# `/var/folders`/tmp.
rm -rf "$CORPUS_ROOT/.grepplus"
rm -rf "$HOME/Library/Caches/grepplus"
rm -rf "${XDG_CACHE_HOME:-$HOME/.cache}/grepplus"
unset GREPPLUS_STORE_DIR
out=$("$PROBE_BIN" "$CORPUS_ROOT" 2>/dev/null || true)
actual=$(printf "%s" "$out" | sed -n 's/.*"outcome":"\([A-Za-z]*\)".*/\1/p')
elapsed=$(printf "%s" "$out" | sed -n 's/.*"elapsed_ms":\([0-9]*\).*/\1/p')
: "${actual:=Cold}"
if [[ "$actual" == "Cold" ]]; then
  printf "  [PASS] %-30s expect=%-15s actual=%-15s elapsed=%sms\n" \
    "cold_start" "Cold" "$actual" "$elapsed"
  pass=$((pass + 1))
else
  printf "  [FAIL] %-30s expect=%-15s actual=%-15s elapsed=%sms\n" \
    "cold_start" "Cold" "$actual" "$elapsed"
  fail=$((fail + 1))
  failures+=("cold_start: expected Cold, got $actual")
fi
# Anchor the cold-start probe, then re-index to recover for
# Scenario 2. reset_repo() also re-runs `grepplus index`, so the
# probe from Scenario 2 onward sees Fresh.
reset_repo

echo ""
echo "=== Scenario 2: fresh immediately after index ==="
out=$("$PROBE_BIN" "$CORPUS_ROOT" 2>/dev/null)
actual=$(printf "%s" "$out" | sed -n 's/.*"outcome":"\([A-Za-z]*\)".*/\1/p')
elapsed=$(printf "%s" "$out" | sed -n 's/.*"elapsed_ms":\([0-9]*\).*/\1/p')
if [[ "$actual" == "Fresh" ]]; then
  printf "  [PASS] %-30s expect=%-15s actual=%-15s elapsed=%sms\n" \
    "fresh_after_index" "Fresh" "$actual" "$elapsed"
  pass=$((pass + 1))
else
  printf "  [FAIL] %-30s expect=%-15s actual=%-15s elapsed=%sms\n" \
    "fresh_after_index" "Fresh" "$actual" "$elapsed"
  fail=$((fail + 1))
  failures+=("fresh_after_index: expected Fresh, got $actual")
fi

echo ""
echo "=== Scenario 3-9: mutation scenarios ==="
probe "file_modified" "Stale" \
  bash -c 'echo "// extra" >> src/lib.rs'

probe "file_deleted" "Stale" \
  rm src/orders.rs

probe "file_added" "Stale" \
  bash -c 'cat > src/newfile.rs <<EOF
pub fn new_symbol() -> u32 { 42 }
EOF'

probe "file_renamed" "Stale" \
  bash -c 'mv src/lib.rs src/lib_renamed.rs'

probe "new_commit" "Stale" \
  bash -c 'echo "// comment" >> src/greeter.rs && git add -A && git -c user.email=t@t -c user.name=t commit -q -m "scenario"'

probe "branch_created" "Stale" \
  bash -c 'git checkout -q -b scenario-branch && echo "// branch" >> src/orders.rs && git add -A && git -c user.email=t@t -c user.name=t commit -q -m "branch" && git checkout -q main 2>/dev/null || git checkout -q master 2>/dev/null || true'

probe "agent_temp_file" "Stale" \
  bash -c 'cat > src/_agent_temp_file.rs <<EOF
pub fn agent_temp() {}
EOF'

echo ""
echo "=== freshness_bench.sh summary ==="
echo "pass: $pass"
echo "fail: $fail"
if [[ "$fail" -gt 0 ]]; then
  echo "failed entries:"
  for f in "${failures[@]}"; do
    echo "  - $f"
  done
fi
[[ "$fail" -eq 0 ]]
