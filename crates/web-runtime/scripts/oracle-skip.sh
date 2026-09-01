#!/bin/sh
set -eu
# Guide oracle is playwright@1.62.1 + Chromium revision 1234.
# Do not download browsers. Write a skip receipt when the pin is missing.
root="$(cd "$(dirname "$0")/../../.." && pwd)"
out="${1:-$root/contracts/web-runtime/receipts/oracle-skip.json}"
mkdir -p "$(dirname "$out")"
chromium_pin="$HOME/Library/Caches/ms-playwright/chromium-1234"
if [ -d "$chromium_pin" ]; then
  echo "{\"status\":\"ready\",\"chromium\":\"1234\"}" > "$out"
  echo "oracle chromium-1234 present"
  exit 0
fi
cat > "$out" <<JSON
{
  "status": "skipped",
  "reason": "pinned Chromium revision 1234 is not installed",
  "playwright_version": "1.62.1",
  "required_chromium_revision": "1234",
  "reference": "docs/PLAYWRIGHT_INTERACTIVE_WEB_RUNTIME_GUIDE.md section 20.1"
}
JSON
echo "oracle skipped: missing chromium-1234"
