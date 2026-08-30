#!/bin/sh
set -eu
# Defined-symbol intersection gate for the one-binary web-runtime image.
# Fails on any non-ICU overlapping global between libjs_static.a and librusty_v8.a.
# ICU 77 coalescing is counted and permitted; mixed ICU versions fail.
# This is a tracked, deterministic check — not a vendor/mozjs_sys path patch.

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
profile=${1:-debug}
js=$(find "$root/target/$profile/build" -name libjs_static.a -path '*mozjs_sys-*' 2>/dev/null | head -1)
v8=$(find "$root/target/$profile" -name librusty_v8.a 2>/dev/null | head -1)
if [ -z "$js" ] || [ -z "$v8" ]; then
  echo "check-engine-symbol-overlap: missing archives (js='$js' v8='$v8')" >&2
  echo "build phase1-probe or web-runtime first" >&2
  exit 1
fi
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
nm -gU "$js" | awk '$2 ~ /^[TSDC]$/ { print $3 }' | LC_ALL=C sort -u > "$tmp/js"
nm -gU "$v8" | awk '$2 ~ /^[TSDC]$/ { print $3 }' | LC_ALL=C sort -u > "$tmp/v8"
comm -12 "$tmp/js" "$tmp/v8" > "$tmp/both"
icu=0
dangerous=0
while IFS= read -r name; do
  [ -n "$name" ] || continue
  case "$name" in
    *icu* | *ICU* | *_77* | *UCaseMap* | *CollatorSpec* | *UErrorCode* | *CReg*)
      icu=$((icu + 1))
      ;;
    *)
      echo "non-ICU overlap: $name" >&2
      dangerous=$((dangerous + 1))
      ;;
  esac
done < "$tmp/both"
echo "check-engine-symbol-overlap: $(wc -l < "$tmp/both") shared globals, $icu ICU 77, $dangerous non-ICU"
if [ "$dangerous" -ne 0 ]; then
  echo "check-engine-symbol-overlap: refusing non-ICU defined-symbol overlap" >&2
  exit 1
fi
