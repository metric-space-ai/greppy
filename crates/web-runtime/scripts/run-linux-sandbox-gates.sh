#!/bin/sh
set -eu
# Reproducible Linux-only gate: Landlock live deny + same-image fexecve.
# Must run on a Linux kernel (GitHub ubuntu-latest, or a Linux container).
# Skip-not-success: landlock_denies_path_outside_allow_list may eprint SKIP
# on ENOSYS; that is not a sandbox success. Inspect the log.
#
# Usage (from any cwd):
#   crates/web-runtime/scripts/run-linux-sandbox-gates.sh
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd -P)"
if [ "$(uname -s)" != "Linux" ]; then
  echo "run-linux-sandbox-gates: refusing to run on $(uname -s); live Landlock/fexecve need a Linux kernel" >&2
  echo "use crates/web-runtime/scripts/run-linux-sandbox-gates-docker.sh on macOS, or GitHub ubuntu-latest" >&2
  exit 2
fi
cd "$ROOT"
if [ -n "${CI:-}" ]; then
  GREPPY_REQUIRE_LIVE_LANDLOCK=1
  export GREPPY_REQUIRE_LIVE_LANDLOCK
fi
CARGO_ARGS="test --manifest-path Cargo.toml -p web-runtime --lib --no-default-features"
echo "run-linux-sandbox-gates: linux_sandbox module"
cargo $CARGO_ARGS linux_sandbox
echo "run-linux-sandbox-gates: linux_spawn_uses_fd_backed_exec"
cargo $CARGO_ARGS worker::tests::linux_spawn_uses_fd_backed_exec -- --exact
echo "run-linux-sandbox-gates: linux_identity_mismatch_kills_child"
cargo $CARGO_ARGS worker::tests::linux_identity_mismatch_kills_child -- --exact
echo "run-linux-sandbox-gates: linux_worker_sandbox_refuses_filesystem_root"
cargo $CARGO_ARGS supervisor::tests::linux_worker_sandbox_refuses_filesystem_root -- --exact
echo "run-linux-sandbox-gates: linux_fd_backed_exec_contract"
cargo $CARGO_ARGS worker::tests::linux_fd_backed_exec_contract -- --exact
echo "run-linux-sandbox-gates: fexecve_args_omit_capability_secret"
cargo $CARGO_ARGS worker::tests::fexecve_args_omit_capability_secret -- --exact
echo "run-linux-sandbox-gates: pin_supervisor_image_fd_is_cloexec"
cargo $CARGO_ARGS worker::tests::pin_supervisor_image_fd_is_cloexec -- --exact
echo "run-linux-sandbox-gates: same_image_bind_does_not_put_capability_on_argv"
cargo $CARGO_ARGS worker::tests::same_image_bind_does_not_put_capability_on_argv -- --exact
echo "run-linux-sandbox-gates: ok"
