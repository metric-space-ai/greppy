#!/bin/sh
set -eu
# Run phase1-probe with a hard timeout and kill the process group on expiry.
# Usage: run-phase1-probe.sh MODE [TIMEOUT_SECONDS]
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd -P)"
MODE=${1:-}
TIMEOUT=${2:-45}
if [ -z "$MODE" ]; then
  echo "usage: $0 MODE [TIMEOUT_SECONDS]" >&2
  exit 2
fi
PROBE="$ROOT/target/debug/phase1-probe"
if [ ! -x "$PROBE" ]; then
  echo "run-phase1-probe: missing $PROBE" >&2
  exit 1
fi
export MODE TIMEOUT PROBE
python3 - <<'PY'
import os, signal, subprocess, sys
mode = os.environ["MODE"]
timeout = float(os.environ["TIMEOUT"])
probe = os.environ["PROBE"]
proc = subprocess.Popen(
    [probe, mode],
    start_new_session=True,
)
try:
    rc = proc.wait(timeout=timeout)
except subprocess.TimeoutExpired:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    print(f"run-phase1-probe: timed out after {timeout}s ({mode})", file=sys.stderr)
    sys.exit(124)
sys.exit(rc)
PY
