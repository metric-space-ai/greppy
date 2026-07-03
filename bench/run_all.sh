#!/usr/bin/env bash
# Phase 7 — Run all three bench scripts and print a combined summary.
#
# Each individual script writes its own per-suite pass/fail to stdout.
# This wrapper runs them in sequence, captures the totals, and prints
# a combined summary at the end.

set -uo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

cd "$WORKSPACE_ROOT"

echo "==============================================="
echo "  Phase 7 benchmark suite"
echo "==============================================="

total_pass=0
total_fail=0
declare -a suites
overall_rc=0

for script in grep_compat.sh agent_utility.sh freshness_bench.sh; do
  echo ""
  echo "--- $script ---"
  # Tee the output to a temp file so we can both display it and
  # parse the totals from it.
  out=$(mktemp)
  if ! bash "$WORKSPACE_ROOT/bench/$script" | tee "$out"; then
    overall_rc=1
  fi
  pass=$(awk '/^pass: [0-9]+ \/ fail: [0-9]+/{print $2; exit}' "$out")
  fail=$(awk '/^pass: [0-9]+ \/ fail: [0-9]+/{print $5; exit}' "$out")
  # Fall back to single-line formats used by the older bench scripts.
  if [[ -z "$pass" ]]; then
    pass=$(awk -F': ' '/^pass: /{print $2; exit}' "$out")
  fi
  if [[ -z "$fail" ]]; then
    fail=$(awk -F': ' '/^fail: /{print $2; exit}' "$out")
  fi
  : "${pass:=0}"
  : "${fail:=0}"
  suites+=("$script pass=$pass fail=$fail")
  total_pass=$((total_pass + pass))
  total_fail=$((total_fail + fail))
  rm -f "$out"
done

echo ""
echo "==============================================="
echo "  Combined summary"
echo "==============================================="
for s in "${suites[@]}"; do
  echo "  $s"
done
echo ""
echo "  total pass: $total_pass"
echo "  total fail: $total_fail"
[[ "$overall_rc" -eq 0 && "$total_fail" -eq 0 ]]
