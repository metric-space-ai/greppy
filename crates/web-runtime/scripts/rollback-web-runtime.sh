#!/bin/sh
set -eu
# Restore dest/bin/web-runtime from dest/previous written by upgrade-web-runtime.sh.
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
for bin in web-runtime; do
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
for bin in web-runtime; do
  web_runtime_copy_regular_file "$dest/previous/$bin" "$staging/bin/$bin"
  web_runtime_copy_regular_file "$dest/bin/$bin" "$staging/previous/$bin"
done
for member in sbom.json provenance.json README.txt UNSIGNED LICENSE coverage-manifest.json benchmark-receipt.json size-receipt.json .greppy-web-runtime-dist; do
  if [ -L "$dest/$member" ]; then
    web_runtime_die "refusing symlink member: $dest/$member"
  fi
  if [ -f "$dest/$member" ]; then
    web_runtime_copy_regular_file "$dest/$member" "$staging/$member"
  fi
done
python3 - "$staging" <<'PY'
import json, os, sys
dest = sys.argv[1]
size = os.path.getsize(os.path.join(dest, "bin", "web-runtime"))
size_path = os.path.join(dest, "size-receipt.json")
if os.path.isfile(size_path) and not os.path.islink(size_path):
    receipt = json.load(open(size_path))
    receipt["installed_bytes"] = size
    receipt["note"] = "Size remeasured after rollback to the previous image. Chromium comparison remains unclaimed."
    open(size_path, "w").write(json.dumps(receipt, indent=2) + "\n")
bench_path = os.path.join(dest, "benchmark-receipt.json")
if os.path.isfile(bench_path) and not os.path.islink(bench_path):
    bench = json.load(open(bench_path))
    bench["installed_bytes"] = size
    metrics = bench.setdefault("metrics", {})
    metrics["installed_bytes"] = size
    metrics["cold_start_to_first_page_ms"] = None
    metrics["peak_rss_bytes"] = None
    metrics["idle_cpu_percent"] = None
    bench["session_metrics"] = "reset_after_rollback"
    bench["note"] = "installed_bytes remeasured after rollback. Session metrics are not valid for the restored image."
    open(bench_path, "w").write(json.dumps(bench, indent=2) + "\n")
PY
web_runtime_write_sha256sums "$staging"
web_runtime_write_stamp "$staging"
web_runtime_commit_staging "$staging" "$dest"
staging=""
trap - EXIT
echo "rolled back $dest"
