#!/bin/sh
set -eu
# Package the three separately linked runtime images into a local unsigned
# dist. Dest must be an allowed staging directory; existing dest is replaced
# only when it is a previous web-runtime dist owned by this user.
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
# shellcheck disable=SC1091
. "$SCRIPT_DIR/web-runtime-dest-guard.sh"

root="$(web_runtime_repo_root)"
dest="$(web_runtime_require_package_dest "${1:-}")"
bins="web-runtime-supervisor web-controller-worker web-content-worker"
mkdir -p "$dest/bin"
found=0
for bin in $bins; do
  for candidate in \
    "$root/crates/web-runtime/target/debug/$bin" \
    "$root/crates/web-runtime/target/release/$bin" \
    "$root/target/debug/$bin" \
    "$root/target/release/$bin"
  do
    if [ -x "$candidate" ] && [ ! -L "$candidate" ]; then
      cp "$candidate" "$dest/bin/$bin"
      found=$((found + 1))
      break
    fi
  done
done
if [ "$found" -ne 3 ]; then
  echo "package-web-runtime: expected 3 binaries, found $found" >&2
  exit 1
fi
(
  cd "$dest/bin"
  if command -v shasum >/dev/null; then
    shasum -a 256 $bins > ../SHA256SUMS
  else
    sha256sum $bins > ../SHA256SUMS
  fi
)
python3 - "$dest" <<'PY'
import hashlib, json, os, platform, sys, time
dest = sys.argv[1]
bins = ["web-runtime-supervisor", "web-controller-worker", "web-content-worker"]
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
cat > "$dest/README.txt" <<TXT
Greppy web-runtime local distributable
Contains three separately linked images. This archive is not signed.
TXT
if [ -f "$root/LICENSE" ] && [ ! -L "$root/LICENSE" ]; then
  cp "$root/LICENSE" "$dest/LICENSE"
fi
echo "NOT_PRODUCTION_SIGNED" > "$dest/UNSIGNED"
web_runtime_write_stamp "$dest"
parent="$(dirname "$dest")"
base="$(basename "$dest")"
archive="$parent/${base}.tar.gz"
if [ -e "$archive" ]; then
  [ -L "$archive" ] && web_runtime_die "refusing to overwrite symlink archive: $archive"
  [ -f "$archive" ] || web_runtime_die "archive exists and is not a file: $archive"
  owner=$(web_runtime_owner_uid "$archive")
  me=$(web_runtime_uid)
  [ "$owner" = "$me" ] || web_runtime_die "refusing non-owned archive $archive"
  rm -f "$archive"
fi
tar -C "$parent" -czf "$archive" "$base"
echo "packed $dest and $archive (unsigned)"
