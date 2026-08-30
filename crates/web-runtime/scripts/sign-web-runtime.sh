#!/bin/sh
set -eu
# Production-capable signing and notarization handoff. Requires
# GREPPY_CODESIGN_IDENTITY for a real Developer ID identity. Without it,
# package unsigned/ad-hoc and record an explicit skip. Ad-hoc linker
# signatures on macOS are not production signatures.
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
# shellcheck disable=SC1091
. "$SCRIPT_DIR/web-runtime-dest-guard.sh"
root="$(web_runtime_repo_root)"
dest="$(web_runtime_validate_dest_shape "${1:-$root/target/web-runtime-dist}")"
"$SCRIPT_DIR/package-web-runtime.sh" "$dest"
bins="web-runtime"
receipt="$dest/SIGNING_RECEIPT"
skip="$dest/SIGNING_SKIPPED"
entitlements="$root/crates/web-runtime/scripts/entitlements.plist"
if [ -n "${GREPPY_CODESIGN_IDENTITY:-}" ]; then
  for bin in $bins; do
    if [ "$(uname -s)" = Darwin ]; then
      codesign --force --sign "$GREPPY_CODESIGN_IDENTITY" --options runtime --timestamp --entitlements "$entitlements" "$dest/bin/$bin"
      codesign --verify --verbose "$dest/bin/$bin"
    else
      echo "sign-web-runtime: no codesign(1) on $(uname -s); record identity only" >&2
    fi
  done
  echo "identity=$GREPPY_CODESIGN_IDENTITY" > "$receipt"
  echo "production_signed=true" >> "$receipt"
  echo "hardened_runtime=true" >> "$receipt"
  rm -f "$skip" "$dest/UNSIGNED"
else
  echo "GREPPY_CODESIGN_IDENTITY unset" > "$skip"
  echo "production_signed=false" >> "$skip"
  echo "NOT_PRODUCTION_SIGNED" > "$dest/UNSIGNED"
  : > "$dest/SIGNING_STATUS"
  if command -v codesign >/dev/null && [ "$(uname -s)" = Darwin ]; then
    for bin in $bins; do
      echo "## $bin" >> "$dest/SIGNING_STATUS"
      codesign -d -vv "$dest/bin/$bin" >> "$dest/SIGNING_STATUS" 2>&1 || true
    done
  fi
fi
"$root/crates/web-runtime/scripts/notarize-web-runtime.sh" "$dest"
# codesign mutates the linked image and UNSIGNED/SIGNING_* membership; rewrite
# payload hashes after that mutation so install/upgrade can still verify.
web_runtime_write_sha256sums "$dest"
echo "sign-web-runtime: done ($dest)"
