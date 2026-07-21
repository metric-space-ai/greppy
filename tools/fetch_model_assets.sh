#!/usr/bin/env bash
# Fetch the embedded model assets from Hugging Face into crates/cli/assets/,
# verifying every file against the sha256 recorded in
# crates/cli/assets/MODEL_ASSETS.json. Idempotent: valid files are kept.
#
# The repo pins WHAT ships (manifest + build.rs digests); Hugging Face hosts
# the bytes in public, ungated repos, so no token is required. Override the
# host with GREPPY_MODEL_ASSET_HF_HOST to fetch from a mirror.
set -euo pipefail
cd "$(dirname "$0")/.."

MANIFEST=crates/cli/assets/MODEL_ASSETS.json
command -v jq >/dev/null || { echo "fetch_model_assets: jq is required" >&2; exit 69; }

if command -v sha256sum >/dev/null 2>&1; then HASH="sha256sum"; else HASH="shasum -a 256"; fi

HF_HOST="${GREPPY_MODEL_ASSET_HF_HOST:-$(jq -r '.hf_host // "https://huggingface.co"' "$MANIFEST")}"
REV="$(jq -r '.revision // "main"' "$MANIFEST")"

status=0
count="$(jq '.assets | length' "$MANIFEST")"
for i in $(seq 0 $((count - 1))); do
  repo="$(jq -r ".assets[$i].hf_repo" "$MANIFEST")"
  file="$(jq -r ".assets[$i].hf_file" "$MANIFEST")"
  dest="$(jq -r ".assets[$i].dest" "$MANIFEST")"
  want="$(jq -r ".assets[$i].sha256" "$MANIFEST")"
  url="$HF_HOST/$repo/resolve/$REV/$file"

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
  echo "fetch    $url"
  curl --proto '=https' --tlsv1.2 --location --fail --silent --show-error \
    --retry 5 --retry-connrefused --output "$tmp" "$url"
  got="$($HASH "$tmp" | awk '{print $1}')"
  if [ "$got" != "$want" ]; then
    echo "fetch_model_assets: digest mismatch for $file (got $got, want $want)" >&2
    rm -f "$tmp"
    status=1
    continue
  fi
  mv "$tmp" "$dest"
  echo "ok       $dest"
done
exit "$status"
