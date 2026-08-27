#!/bin/sh
set -eu
# Restore dest/bin images from dest/previous written by upgrade-web-runtime.sh.
# Dest must be a stamped web-runtime dist. Never rm -rf.
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
# shellcheck disable=SC1091
. "$SCRIPT_DIR/web-runtime-dest-guard.sh"

dest="$(web_runtime_require_existing_dist "${1:-}" "install")"
for bin in web-runtime-supervisor web-controller-worker web-content-worker; do
  [ -f "$dest/previous/$bin" ] || web_runtime_die "no previous image to roll back: $bin"
  [ -L "$dest/previous/$bin" ] && web_runtime_die "refusing symlink previous image: $bin"
done
for bin in web-runtime-supervisor web-controller-worker web-content-worker; do
  web_runtime_copy_regular_file "$dest/previous/$bin" "$dest/bin/$bin"
done
echo "rolled back $dest"
