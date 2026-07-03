#!/usr/bin/env bash
# Deterministic corpus generator for the agent-efficiency benchmark.
#
# Produces git-initialised repos of varied size and language under corpus/
# (Rust, Python, Go, Java, JavaScript, TypeScript) with real cross-file
# structure that grepplus can index into a call graph. Reproducible: rerun
# to regenerate byte-identical sources.
#
# Usage:  bash bench/agent_efficiency/gen_corpus.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "generating corpus..."
exec python3 "${HERE}/gen_corpus.py" "$@"
