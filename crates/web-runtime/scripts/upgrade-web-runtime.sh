#!/bin/sh
set -eu
# Replace dest binaries with those from a source dist, keeping the previous
# three images under dest/previous for rollback. Dest must already be a
# stamped web-runtime dist. Builds a complete staging tree then swaps so a
# later missing member cannot leave dest half-upgraded.
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
# shellcheck disable=SC1091
. "$SCRIPT_DIR/web-runtime-dest-guard.sh"

src="$(web_runtime_require_existing_dist "${1:-}" "source")"
dest="$(web_runtime_require_existing_dist "${2:-}" "install")"
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
  web_runtime_copy_regular_file "$dest/bin/$bin" "$staging/previous/$bin"
  web_runtime_copy_regular_file "$src/bin/$bin" "$staging/bin/$bin"
done
for member in SHA256SUMS sbom.json provenance.json README.txt UNSIGNED LICENSE .greppy-web-runtime-dist; do
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
echo "upgraded $dest from $src"
