#!/bin/sh
set -eu
root="$(cd "$(dirname "$0")/../../.." && pwd)"
pw="${PLAYWRIGHT_PACKAGE:-$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright}"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$HOME/Library/Caches/ms-playwright}"
chrome="${GREPPY_ORACLE_CHROMIUM:-$PLAYWRIGHT_BROWSERS_PATH/chromium-1234/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing}"
if [ ! -x "$chrome" ]; then
  echo "oracle-reference: missing chromium executable: $chrome" >&2
  exit 2
fi
if [ ! -f "$pw/index.mjs" ]; then
  echo "oracle-reference: missing playwright package: $pw" >&2
  exit 2
fi
out="${1:-$root/contracts/web-runtime/receipts/oracle-reference.json}"
mkdir -p "$(dirname "$out")"
PLAYWRIGHT_PACKAGE="$pw" GREPPY_ORACLE_CHROMIUM="$chrome" \
  /opt/homebrew/bin/node "$root/crates/web-runtime/scripts/oracle-reference.mjs" "$out"
