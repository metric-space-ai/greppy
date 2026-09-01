#!/bin/sh
set -eu
# Production-capable notarization. Requires a signed distributable plus
# GREPPY_NOTARY_PROFILE (notarytool keychain profile) or APPLE_ID +
# APPLE_APP_SPECIFIC_PASSWORD + APPLE_TEAM_ID. Missing credentials write
# NOTARIZATION_SKIPPED and exit 0; they do not fail local unsigned verification.
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
# shellcheck disable=SC1091
. "$SCRIPT_DIR/web-runtime-dest-guard.sh"
root="$(web_runtime_repo_root)"
dest="$(web_runtime_validate_dest_shape "${1:-$root/target/web-runtime-dist}")"
[ -d "$dest" ] || web_runtime_die "missing package $dest"
web_runtime_check_owned_dir "$dest"
web_runtime_is_owned_dist "$dest" || web_runtime_die "not a web-runtime dist: $dest"
receipt="$dest/NOTARIZATION_RECEIPT"
skip="$dest/NOTARIZATION_SKIPPED"
archive="$(dirname "$dest")/$(basename "$dest").tar.gz"
if [ ! -f "$dest/SIGNING_RECEIPT" ]; then
  echo "GREPPY_CODESIGN_IDENTITY unset or unsigned package" > "$skip"
  echo "production_notarized=false" >> "$skip"
  echo "NOT_PRODUCTION_NOTARIZED" > "$dest/NOTARIZED_UNSIGNED"
  echo "notarize-web-runtime: skipped (unsigned) $dest"
  exit 0
fi
if [ -n "${GREPPY_NOTARY_PROFILE:-}" ]; then
  xcrun notarytool submit "$archive" --keychain-profile "$GREPPY_NOTARY_PROFILE" --wait
  echo "profile=$GREPPY_NOTARY_PROFILE" > "$receipt"
  echo "production_notarized=true" >> "$receipt"
  rm -f "$skip" "$dest/NOTARIZED_UNSIGNED"
  echo "notarize-web-runtime: submitted $archive"
  exit 0
fi
if [ -n "${APPLE_ID:-}" ] && [ -n "${APPLE_APP_SPECIFIC_PASSWORD:-}" ] && [ -n "${APPLE_TEAM_ID:-}" ]; then
  xcrun notarytool submit "$archive" --apple-id "$APPLE_ID" --password "$APPLE_APP_SPECIFIC_PASSWORD" --team-id "$APPLE_TEAM_ID" --wait
  echo "apple_id=$APPLE_ID" > "$receipt"
  echo "production_notarized=true" >> "$receipt"
  rm -f "$skip" "$dest/NOTARIZED_UNSIGNED"
  echo "notarize-web-runtime: submitted $archive"
  exit 0
fi
echo "GREPPY_NOTARY_PROFILE and Apple notary credentials unset" > "$skip"
echo "production_notarized=false" >> "$skip"
echo "NOT_PRODUCTION_NOTARIZED" > "$dest/NOTARIZED_UNSIGNED"
echo "notarize-web-runtime: skipped (no credentials) $dest"
