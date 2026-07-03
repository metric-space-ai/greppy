#!/usr/bin/env bash
# Phase 7 — agent-utility corpus (phase plan §12.2).
#
# For each agent-style invocation we record how much information
# grepplus-grep surfaces on top of real grep:
#
#   - real_grep_bytes: real grep's stdout byte count (control)
#   - sub_grep_bytes : grepplus-grep's stdout byte count (subject)
#   - delta_bytes    : subject - control  (= 0 for STRICT/SIDECAR,
#                       ≥ length-of-synthetic-line for VISIBLE_AUGMENT)
#   - sidecar_path   : path to the .md sidecar file (if any)
#   - sidecar_bytes  : size of the .md sidecar file
#   - synth_count    : how many GREPPLUS_NON_CANONICAL_HIT lines appeared
#                       in the subject's stdout
#   - exit_code      : real-grep exit code (must be preserved by the
#                       wrapper modulo signal handling)
#
# This is a smoke test, not a statistically rigorous benchmark. The
# numbers it prints are useful for detecting regressions.

set -uo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GREPPLUS_BIN="${GREPPLUS_BIN:-$WORKSPACE_ROOT/target/debug/grepplus}"
REAL_GREP="${REAL_GREP:-/usr/bin/grep}"
CORPUS_SRC="${CORPUS_SRC:-$WORKSPACE_ROOT/bench/fixtures/sample}"
# Copy the fixture to a temp dir so the indexer's
# detect_repo_root / walk don't accidentally pick up the parent
# grepplus-rs workspace. See bench/grep_compat.sh for the same
# rationale.
CORPUS_ROOT="$(mktemp -d -t grepplus-agent-utility.XXXXXX)"
cp -R "$CORPUS_SRC/." "$CORPUS_ROOT/"
rm -rf "$CORPUS_ROOT/.grepplus" "$CORPUS_ROOT/.git"
trap 'rm -rf "$CORPUS_ROOT"' EXIT

cd "$CORPUS_ROOT"

# Always reindex from scratch so the freshness gate is Fresh and
# VISIBLE_AUGMENT can take effect.
rm -rf .grepplus "${TMPDIR:-/tmp}"/grepplus 2>/dev/null || true
"$GREPPLUS_BIN" index "$CORPUS_ROOT" >/dev/null 2>&1

# Corpus of agent-style invocations. Each row is space-separated argv
# tokens (without the binary name). We use `|` as a separator and
# decode with IFS.
CORPUS=(
  "-R|hello|."
  "-R|ProcessOrder|."
  "-R|UserService|."
  "-R|total|."
  "-R|build_default_order|."
  "-R|Greeter|."
  "-R|fmt|."
  "-R|payment_retry|."      # Python symbol — should still find via raw
                            # grep but the sidecar will say no Rust
                            # symbol because Python is unsupported.
  "-R|process_payment|."    # Python symbol — same.
  "-R|nonexistent_symbol_xyz|."  # No real-grep matches, sidecar
                                # may still appear with empty hits.
)

printf "%-50s %-4s %-12s %-12s %-12s %-12s %-12s %-6s %s\n" \
  "command" "rc" "real_b" "sub_b" "delta_b" "sidecar_b" "synth_n" "side" "sentinel?"
echo "------------------------------------------------------------------------------------------------------------------------------------------------------------------------"

pass=0
fail=0
declare -a failures

for entry in "${CORPUS[@]}"; do
  IFS='|' read -r -a argv <<< "$entry"
  pretty="${argv[*]}"

  real_out=$(mktemp)
  sub_out=$(mktemp)
  sub_err=$(mktemp)
  # Real grep (control) — capture exit code without `|| true` masking it.
  "$REAL_GREP" "${argv[@]}" >"$real_out" 2>/dev/null
  real_rc=$?
  # Subject.
  "$GREPPLUS_BIN" "${argv[@]}" >"$sub_out" 2>"$sub_err"
  sub_rc=$?

  real_b=$(wc -c <"$real_out" | tr -d ' ')
  sub_b=$(wc -c <"$sub_out" | tr -d ' ')
  delta_b=$(( sub_b - real_b ))
  synth_n=$(grep -c 'GREPPLUS_NON_CANONICAL_HIT' "$sub_out" 2>/dev/null || echo 0)
  sidecar=$(find "${TMPDIR:-/tmp}" -type f \
            -name "*__GREPPLUS_SEMANTIC_NONCANONICAL.md" \
            -newer "$real_out" 2>/dev/null | sort | tail -1 || true)
  sidecar_b=0
  sentinel="no"
  if [[ -n "$sidecar" && -f "$sidecar" ]]; then
    sidecar_b=$(wc -c <"$sidecar" | tr -d ' ')
    sentinel=$(grep -c 'GREPPLUS_NON_CANONICAL_HIT' "$sidecar" 2>/dev/null || echo 0)
    [[ "$sentinel" -ge 1 ]] && sentinel="yes" || sentinel="no"
  fi

  # Expectations:
  #   rc       — must match real_rc exactly
  #   sidecar  — if real-grep found hits AND we are in VISIBLE_AUGMENT
  #              territory, expect a sidecar with sentinel.
  #              If real-grep found NO hits, no sidecar is expected
  #              (phase plan §11.5: semantic-on-miss is opt-in via
  #              GREPPLUS_SEMANTIC_ON_MISS=1, default off).
  ok=1
  reason=""
  if [[ "$real_rc" != "$sub_rc" ]]; then
    ok=0
    reason="rc mismatch (real=$real_rc sub=$sub_rc)"
  fi
  if [[ "${argv[0]}" == "-R" && ${#argv[@]} -ge 3 \
        && "${argv[${#argv[@]}-1]}" == "." ]]; then
    if [[ "$real_rc" == "0" ]]; then
      # Real-grep found matches. The wrapper's stdout must be a
      # byte-prefix of real-grep's output (no synthetic bytes
      # interleaved) when in Strict or Sidecar mode, OR a
      # byte-prefix + sentinel line when in VISIBLE_AUGMENT mode.
      # We assert the byte-prefix property for every match case.
      if [[ "$real_b" -gt 0 ]] && [[ "${real_out:0:$real_b}" != "${sub_out:0:$real_b}" ]]; then
        ok=0
        reason="subject output does not start with real-grep bytes"
      fi
    else
      # Real-grep found no matches → phase plan §11.5: no synthetic
      # line / sidecar by default. Sub exit code must be 1, and
      # subject stdout must be byte-exact empty (no synthetic
      # line appended — R-002).
      if [[ "$sub_rc" != "1" ]]; then
        ok=0
        reason="exit code should be 1 on no-match (got $sub_rc)"
      fi
      if [[ "$sub_b" -ne 0 ]]; then
        ok=0
        reason="stdout must be empty on no-match (got $sub_b bytes: <<<$(cat "$sub_out")>>>)"
      fi
      if [[ "$synth_n" -ne 0 ]]; then
        ok=0
        reason="no synthetic sentinel expected on no-match (found $synth_n)"
      fi
    fi
  fi

  if [[ "$ok" -eq 1 ]]; then
    printf "%-50s %-4s %-12s %-12s %-12s %-12s %-12s %-6s %s\n" \
      "$pretty" "$sub_rc" "$real_b" "$sub_b" "$delta_b" "$sidecar_b" "$synth_n" "$( [[ -n "$sidecar" ]] && echo yes || echo no )" "$sentinel"
    pass=$((pass + 1))
  else
    printf "%-50s %-4s %-12s %-12s %-12s %-12s %-12s %-6s %s  [FAIL: %s]\n" \
      "$pretty" "$sub_rc" "$real_b" "$sub_b" "$delta_b" "$sidecar_b" "$synth_n" "$( [[ -n "$sidecar" ]] && echo yes || echo no )" "$sentinel" "$reason"
    fail=$((fail + 1))
    failures+=("$pretty: $reason")
  fi

  rm -f "$real_out" "$sub_out" "$sub_err"
done

echo ""
echo "=== agent_utility.sh summary ==="
echo "pass: $pass"
echo "fail: $fail"
if [[ "$fail" -gt 0 ]]; then
  echo "failed entries:"
  for f in "${failures[@]}"; do
    echo "  - $f"
  done
fi
[[ "$fail" -eq 0 ]]
