#!/bin/sh
set -eu
# Replace dest binaries with those from a source dist, keeping the previous
# three images under dest/previous for rollback. Dest must already be a
# stamped web-runtime dist. Never rm -rf.
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
# shellcheck disable=SC1091
. "$SCRIPT_DIR/web-runtime-dest-guard.sh"

src="$(web_runtime_require_existing_dist "${1:-}" "source")"
dest="$(web_runtime_require_existing_dist "${2:-}" "install")"
mkdir -p "$dest/previous"
web_runtime_check_owned_dir "$dest/previous"
for bin in web-runtime-supervisor web-controller-worker web-content-worker; do
  web_runtime_copy_regular_file "$dest/bin/$bin" "$dest/previous/$bin"
  web_runtime_copy_regular_file "$src/bin/$bin" "$dest/bin/$bin"
done
for member in SHA256SUMS sbom.json provenance.json README.txt UNSIGNED LICENSE; do
  if [ -f "$src/$member" ]; then
    web_runtime_copy_regular_file "$src/$member" "$dest/$member"
  fi
done
web_runtime_write_stamp "$dest"
echo "upgraded $dest from $src"
