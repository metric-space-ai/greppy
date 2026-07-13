#!/usr/bin/env bash
# Fetch the embedded model assets from the pinned GitHub release into
# crates/cli/assets/, verifying every file against the sha256 recorded in
# crates/cli/assets/MODEL_ASSETS.json. Idempotent: valid files are kept.
#
# The repo pins WHAT ships (manifest + build.rs digests); the release hosts
# the bytes. Release assets are free and unlimited for public repositories,
# unlike Git LFS storage/bandwidth. No token is required: public release
# assets are plain HTTPS downloads.
set -euo pipefail
cd "$(dirname "$0")/.."

MANIFEST=crates/cli/assets/MODEL_ASSETS.json
command -v jq >/dev/null || { echo "fetch_model_assets: jq is required" >&2; exit 69; }

if command -v sha256sum >/dev/null 2>&1; then HASH="sha256sum"; else HASH="shasum -a 256"; fi

DEFAULT_TAG="$(jq -r .release_tag "$MANIFEST")"
REPO_SLUG="${GREPPY_MODEL_ASSET_REPO:-metric-space-ai/greppy}"

status=0
count="$(jq '.assets | length' "$MANIFEST")"
for i in $(seq 0 $((count - 1))); do
  name="$(jq -r ".assets[$i].asset_name" "$MANIFEST")"
  dest="$(jq -r ".assets[$i].dest" "$MANIFEST")"
  want="$(jq -r ".assets[$i].sha256" "$MANIFEST")"
  tag="$(jq -r ".assets[$i].release_tag // empty" "$MANIFEST")"
  [ -n "$tag" ] || tag="$DEFAULT_TAG"
  BASE_URL="https://github.com/$REPO_SLUG/releases/download/$tag"

  if [ -f "$dest" ]; then
    got="$($HASH "$dest" | awk '{print $1}')"
    if [ "$got" = "$want" ]; then
      echo "ok       $dest"
      continue
    fi
    echo "refetch  $dest (digest mismatch)"
    rm -f "$dest"
  fi

  mkdir -p "$(dirname "$dest")"
  tmp="$dest.download"
  echo "fetch    $BASE_URL/$name"
  curl --proto '=https' --tlsv1.2 --location --fail --silent --show-error \
    --retry 5 --retry-connrefused --output "$tmp" "$BASE_URL/$name"
  got="$($HASH "$tmp" | awk '{print $1}')"
  if [ "$got" != "$want" ]; then
    echo "fetch_model_assets: digest mismatch for $name (got $got, want $want)" >&2
    rm -f "$tmp"
    status=1
    continue
  fi
  mv "$tmp" "$dest"
  echo "ok       $dest"
done
exit "$status"
