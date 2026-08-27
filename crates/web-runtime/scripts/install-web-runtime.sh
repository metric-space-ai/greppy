#!/bin/sh
set -eu
# Install a packaged three-image dist into another dest-guarded staging
# directory. Never rm -rf; copies only known package members.
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
# shellcheck disable=SC1091
. "$SCRIPT_DIR/web-runtime-dest-guard.sh"

src="$(web_runtime_require_existing_dist "${1:-}" "source")"
dest="$(web_runtime_require_package_dest "${2:-}")"
for member in \
  bin/web-runtime-supervisor \
  bin/web-controller-worker \
  bin/web-content-worker \
  README.txt \
  UNSIGNED \
  SHA256SUMS \
  sbom.json \
  provenance.json \
  LICENSE \
  .greppy-web-runtime-dist
do
  if [ -f "$src/$member" ]; then
    web_runtime_copy_regular_file "$src/$member" "$dest/$member"
  fi
done
web_runtime_write_stamp "$dest"
echo "installed $src -> $dest"
