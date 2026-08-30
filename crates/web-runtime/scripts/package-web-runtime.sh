#!/bin/sh
set -eu
# Package the single linked web-runtime executable into a local unsigned
# dist. Dest must be an allowed staging directory; existing dest is replaced
# only when it is a previous web-runtime dist owned by this user. Writes land
# in a sibling staging directory and swap into dest only when complete.
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
# shellcheck disable=SC1091
. "$SCRIPT_DIR/web-runtime-dest-guard.sh"

root="$(web_runtime_repo_root)"
dest="$(web_runtime_require_package_dest "${1:-}")"
staging="$(web_runtime_begin_staging "$dest")"
staging_cleanup() {
  if [ -n "${staging:-}" ] && [ -d "$staging" ] && [ ! -L "$staging" ]; then
    web_runtime_discard_staging "$staging" || true
  fi
}
trap staging_cleanup EXIT
bin=web-runtime
found=
for candidate in \
  "$root/crates/web-runtime/target/debug/$bin" \
  "$root/crates/web-runtime/target/release/$bin" \
  "$root/target/debug/$bin" \
  "$root/target/release/$bin"
do
  if [ -x "$candidate" ] && [ ! -L "$candidate" ]; then
    web_runtime_copy_regular_file "$candidate" "$staging/bin/$bin"
    found=$candidate
    break
  fi
done
if [ -z "$found" ]; then
  echo "package-web-runtime: unified web-runtime binary is required" >&2
  echo "looked for an executable named web-runtime in:" >&2
  echo "  crates/web-runtime/target/{debug,release}/web-runtime" >&2
  echo "  target/{debug,release}/web-runtime" >&2
  echo "do not package web-runtime-supervisor / web-controller-worker / web-content-worker" >&2
  exit 1
fi
python3 - "$staging" <<'PY'
import hashlib, json, os, platform, sys, time
dest = sys.argv[1]
bins = ["web-runtime"]
components = []
for name in bins:
    path = os.path.join(dest, "bin", name)
    data = open(path, "rb").read()
    digest = hashlib.sha256(data).hexdigest()
    components.append({
        "type": "application",
        "name": name,
        "hashes": [{"alg": "SHA-256", "content": digest}],
        "size": len(data),
    })
sbom = {
    "bomFormat": "CycloneDX",
    "specVersion": "1.5",
    "components": components,
}
open(os.path.join(dest, "sbom.json"), "w").write(json.dumps(sbom, indent=2) + "\n")
prov = {
    "predicateType": "https://slsa.dev/provenance/v1",
    "buildType": "greppy.web-runtime.package.v1",
    "platform": f"{platform.system()}-{platform.machine()}",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "images": components,
    "production_signed": False,
    "note": "Local unsigned package. Production signatures require GREPPY_CODESIGN_IDENTITY; notarization requires GREPPY_NOTARY_PROFILE or Apple notary credentials.",
}
open(os.path.join(dest, "provenance.json"), "w").write(json.dumps(prov, indent=2) + "\n")
PY
cat > "$staging/README.txt" <<TXT
Greppy web-runtime local distributable
Contains one linked executable. Supervisor, controller, and content roles are selected with --internal-role. This archive is not signed.
TXT
if [ -f "$root/LICENSE" ] && [ ! -L "$root/LICENSE" ]; then
  web_runtime_copy_regular_file "$root/LICENSE" "$staging/LICENSE"
fi
coverage="$root/contracts/web-runtime/playwright-public-surface.v1.json"
if [ -f "$coverage" ] && [ ! -L "$coverage" ]; then
  web_runtime_copy_regular_file "$coverage" "$staging/coverage-manifest.json"
fi
python3 - "$staging" "$found" <<'PY'
import json, os, platform, sys, time
dest, src = sys.argv[1], sys.argv[2]
size = os.path.getsize(os.path.join(dest, "bin", "web-runtime"))
profile = "release" if "/release/" in src.replace("\\", "/") else "debug"
receipt = {
    "buildType": "greppy.web-runtime.size.v1",
    "measured": True,
    "profile": profile,
    "installed_bytes": size,
    "platform": f"{platform.system()}-{platform.machine()}",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "chromium_playwright_delta": "unclaimed",
    "note": "Measured size of the packaged web-runtime executable only. The guide's 30% Playwright+Chromium comparison is unclaimed until a release image is measured against a pinned Chromium install on the same machine.",
}
open(os.path.join(dest, "size-receipt.json"), "w").write(json.dumps(receipt, indent=2) + "\n")
bench = {
    "buildType": "greppy.web-runtime.benchmark.v1",
    "measured": True,
    "profile": profile,
    "installed_bytes": size,
    "platform": f"{platform.system()}-{platform.machine()}",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "metrics": {
        "installed_bytes": size,
        "cold_start_to_first_page_ms": None,
        "peak_rss_bytes": None,
        "idle_cpu_percent": None,
    },
    "session_metrics": "unmeasured_at_package",
    "note": "installed_bytes is measured at package time. Session latency/RSS/idle-CPU stay None here; they are filled by a post-package session measurement that rewrites this receipt and SHA256SUMS. This is not a Playwright performance claim.",
}
open(os.path.join(dest, "benchmark-receipt.json"), "w").write(json.dumps(bench, indent=2) + "\n")
PY
echo "NOT_PRODUCTION_SIGNED" > "$staging/UNSIGNED"
web_runtime_write_sha256sums "$staging"
web_runtime_write_stamp "$staging"
web_runtime_commit_staging "$staging" "$dest"
staging=""
trap - EXIT
parent="$(dirname "$dest")"
base="$(basename "$dest")"
archive="$parent/${base}.tar.gz"
if [ -e "$archive" ] || [ -L "$archive" ]; then
  [ -L "$archive" ] && web_runtime_die "refusing to overwrite symlink archive: $archive"
  [ -f "$archive" ] || web_runtime_die "archive exists and is not a file: $archive"
  owner=$(web_runtime_owner_uid "$archive")
  me=$(web_runtime_uid)
  [ "$owner" = "$me" ] || web_runtime_die "refusing non-owned archive $archive"
  rm -f "$archive"
fi
tar -C "$parent" -czf "$archive" "$base"
echo "packed $dest and $archive (unsigned)"
