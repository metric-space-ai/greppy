#!/bin/sh
set -eu
# Re-run run-linux-sandbox-gates.sh inside rust:1.95.0-bookworm.
# This is a local convenience, not a substitute for GitHub ubuntu-latest.
# Live Landlock may SKIP in some Docker kernels; that is not a sandbox success.
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
REPO="$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd -P)"
IMAGE="${GREPPY_WEB_RUNTIME_LINUX_IMAGE:-rust:1.95.0-bookworm}"
if ! command -v docker >/dev/null 2>&1; then
  echo "run-linux-sandbox-gates-docker: docker is not available" >&2
  exit 2
fi
# Optional: pin rustup when using a pre-1.95 image so repo rust-toolchain.toml
# does not start a nested 1.95 download. Exact 1.95/Ubuntu-CI evidence stays OPEN.
DOCKER_ENV="-e CARGO_TERM_COLOR=always -e CARGO_TARGET_DIR=/tmp/web-runtime-linux-target"
if [ -n "${RUSTUP_TOOLCHAIN:-}" ]; then
  DOCKER_ENV="$DOCKER_ENV -e RUSTUP_TOOLCHAIN=$RUSTUP_TOOLCHAIN"
fi
# shellcheck disable=SC2086
exec docker run --rm \
  --security-opt seccomp=unconfined \
  $DOCKER_ENV \
  -v "$REPO:/src" \
  -w /src/crates/web-runtime \
  "$IMAGE" \
  sh /src/crates/web-runtime/scripts/run-linux-sandbox-gates.sh
