#!/bin/sh
set -eu
# Restore dest/bin images from dest/previous written by upgrade-web-runtime.sh.
# Dest must be a stamped web-runtime dist. Builds a complete staging tree then
# swaps so a missing previous image cannot leave dest half-rolled-back.
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
# shellcheck disable=SC1091
. "$SCRIPT_DIR/web-runtime-dest-guard.sh"

dest="$(web_runtime_require_existing_dist "${1:-}" "install")"
if [ -L "$dest/previous" ]; then
  web_runtime_die "refusing symlink directory: $dest/previous"
fi
web_runtime_check_owned_real_dir "$dest/previous"
for bin in web-runtime-supervisor web-controller-worker web-content-worker; do
  web_runtime_check_owned_regular_file "$dest/previous/$bin"
done
staging="$(web_runtime_begin_staging "$dest")"
staging_cleanup() {
  if [ -n "${staging:-}" ] && [ -d "$staging" ] && [ ! -L "$staging" ]; then
    web_runtime_discard_staging "$staging" || true
  fi
}
trap staging_cleanup EXIT
mkdir -p "$staging/previous"
web_runtime_check_owned_real_dir "$staging/previous"
for bin in web-runtime-supervisor web-controller-worker web-content-worker; do
  web_runtime_copy_regular_file "$dest/previous/$bin" "$staging/bin/$bin"
  web_runtime_copy_regular_file "$dest/bin/$bin" "$staging/previous/$bin"
done
for member in SHA256SUMS sbom.json provenance.json README.txt UNSIGNED LICENSE .greppy-web-runtime-dist; do
  if [ -L "$dest/$member" ]; then
    web_runtime_die "refusing symlink member: $dest/$member"
  fi
  if [ -f "$dest/$member" ]; then
    web_runtime_copy_regular_file "$dest/$member" "$staging/$member"
  fi
done
web_runtime_write_stamp "$staging"
web_runtime_commit_staging "$staging" "$dest"
staging=""
trap - EXIT
echo "rolled back $dest"
