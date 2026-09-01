#!/bin/sh
set -eu
# Removes a local web-runtime dist produced by package-web-runtime.sh.
# Refuses empty, root, home, repository/workspace roots, relative/ambiguous
# paths, symlinks, non-owned directories, and directories that are not a
# stamped web-runtime dist. Deletes only known package members, never rm -rf.
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
# shellcheck disable=SC1091
. "$SCRIPT_DIR/web-runtime-dest-guard.sh"

dest="$(web_runtime_require_uninstall_dest "${1:-}")"
web_runtime_uninstall_owned_dist "$dest"
echo "uninstalled $dest"
