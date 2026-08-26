#!/bin/sh
set -eu
root="$(cd "$(dirname "$0")/../../.." && pwd)"
dest="${1:-"$root/target/web-runtime-dist"}"
bins="web-runtime-supervisor web-controller-worker web-content-worker"
rm -rf "$dest"
mkdir -p "$dest/bin"
found=0
for bin in $bins; do
  for candidate in \
    "$root/crates/web-runtime/target/debug/$bin" \
    "$root/crates/web-runtime/target/release/$bin" \
    "$root/target/debug/$bin" \
    "$root/target/release/$bin"
  do
    if [ -x "$candidate" ]; then
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
cat > "$dest/sbom.json" <<JSON
{"bomFormat":"CycloneDX","specVersion":"1.5","components":[
 {"type":"application","name":"web-runtime-supervisor"},
 {"type":"application","name":"web-controller-worker"},
 {"type":"application","name":"web-content-worker"}
]}
JSON
cat > "$dest/README.txt" <<TXT
Greppy web-runtime local distributable
Contains three separately linked images. This archive is not signed.
TXT
echo "NOT_PRODUCTION_SIGNED" > "$dest/UNSIGNED"
parent="$(dirname "$dest")"
base="$(basename "$dest")"
archive="$parent/${base}.tar.gz"
rm -f "$archive"
tar -C "$parent" -czf "$archive" "$base"
echo "packed $dest and $archive (unsigned)"
