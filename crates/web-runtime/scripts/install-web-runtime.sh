#!/bin/sh
set -eu
# Install a packaged three-image dist into another dest-guarded staging
# directory. Copies land in a sibling staging directory and swap into dest
# only when the result is a complete stamped dist. Never rm -rf.
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
# shellcheck disable=SC1091
. "$SCRIPT_DIR/web-runtime-dest-guard.sh"

src="$(web_runtime_require_existing_dist "${1:-}" "source")"
dest="$(web_runtime_require_package_dest "${2:-}")"
staging="$(web_runtime_begin_staging "$dest")"
staging_cleanup() {
  if [ -n "${staging:-}" ] && [ -d "$staging" ] && [ ! -L "$staging" ]; then
    web_runtime_discard_staging "$staging" || true
  fi
}
trap staging_cleanup EXIT
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
  if [ -L "$src/$member" ]; then
    web_runtime_die "refusing symlink source member: $src/$member"
  fi
  if [ -f "$src/$member" ]; then
    web_runtime_copy_regular_file "$src/$member" "$staging/$member"
  fi
done
web_runtime_verify_sha256sums "$staging"
web_runtime_write_stamp "$staging"
web_runtime_commit_staging "$staging" "$dest"
staging=""
trap - EXIT
echo "installed $src -> $dest"
