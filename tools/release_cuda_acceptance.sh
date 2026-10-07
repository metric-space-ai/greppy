#!/usr/bin/env bash
# CUDA acceptance for the Linux release package, run before tagging.
#
# Hosted GitHub runners have no GPU and product builds refuse CPU inference, so
# release.yml only checks the Linux package structurally. This script takes the
# .deb that a release.yml dispatch run built and runs the packaged inference
# smoke on a Linux host with an NVIDIA GPU.
#
#   tools/release_cuda_acceptance.sh <release-dispatch-run-id> <ssh-host> [remote-dir]
#
# Exit 0 only when the exact artifact passes release_package_smoke.sh on CUDA.
set -euo pipefail

RUN_ID="${1:?usage: release_cuda_acceptance.sh <run-id> <ssh-host> [remote-dir]}"
HOST="${2:?usage: release_cuda_acceptance.sh <run-id> <ssh-host> [remote-dir]}"
REMOTE="${3:-/tmp/greppy-cuda-acceptance-$RUN_ID}"
REPO="${GREPPY_RELEASE_REPO:-metric-space-ai/greppy}"

work="$(mktemp -d "${TMPDIR:-/tmp}/greppy-cuda-acceptance-XXXXXX")"
trap 'rm -rf "$work"' EXIT
gh run download "$RUN_ID" --repo "$REPO" --name greppy-linux-x86_64.deb --dir "$work"
(cd "$work" && sha256sum -c greppy-linux-x86_64.deb.sha256 2>/dev/null \
  || shasum -a 256 -c greppy-linux-x86_64.deb.sha256)

ssh "$HOST" "rm -rf '$REMOTE' && mkdir -p '$REMOTE'"
scp -q "$work/greppy-linux-x86_64.deb" "$HOST:$REMOTE/"
# The smoke must use the GPU: device auto selects CUDA, and doctor must report it.
ssh "$HOST" bash -s -- "$REMOTE" <<'REMOTE_SCRIPT'
set -euo pipefail
dir="$1"
cd "$dir"
dpkg-deb -x greppy-linux-x86_64.deb root
bin="$dir/root/usr/lib/greppy/bin/greppy"
"$bin" --version
GREPPY_SMOKE_DEVICE=auto GREPPY_SMOKE_BACKEND=cuda \
  bash "$dir/root/usr/lib/greppy/release-tests/release_package_smoke.sh" "$bin" "$dir/work"
GREPPY_SMOKE_DEVICE=auto \
  bash "$dir/root/usr/lib/greppy/release-tests/release_daemon_stress.sh" "$bin" "$dir/daemon-stress"
sha256sum "$bin"
REMOTE_SCRIPT
echo "CUDA acceptance passed for run $RUN_ID on $HOST"
