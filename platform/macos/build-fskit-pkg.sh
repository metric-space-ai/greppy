#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
app=${1:?usage: build-fskit-pkg.sh APP VERSION OUTPUT.pkg}
version=${2:?usage: build-fskit-pkg.sh APP VERSION OUTPUT.pkg}
output=${3:?usage: build-fskit-pkg.sh APP VERSION OUTPUT.pkg}
installer_identity=${INSTALLER_SIGN_IDENTITY:-}
export COPYFILE_DISABLE=1

case "$version" in
    ''|*[!0-9A-Za-z.-]*)
        echo "invalid package version: $version" >&2
        exit 64
        ;;
esac

test -d "$app" || {
    echo "FSKit application is missing: $app" >&2
    exit 1
}
test -x "$app/Contents/Resources/bin/greppy" || {
    echo "FSKit application does not contain the bundled Greppy CLI" >&2
    exit 1
}
test -f "$app/Contents/Resources/bin/web-runtime/.greppy-web-runtime-dist" &&
test -x "$app/Contents/Resources/bin/web-runtime/bin/web-runtime" || {
    echo "FSKit application does not contain the bundled Greppy web runtime" >&2
    exit 1
}
app_version=$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' \
    "$app/Contents/Info.plist")
extension_version=$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' \
    "$app/Contents/Extensions/GreppyWorkspaceFS.appex/Contents/Info.plist")
test "$app_version" = "$version" && test "$extension_version" = "$version" || {
    echo "CLI/package version $version does not match FSKit app $app_version/$extension_version" >&2
    exit 1
}
test ! -e "$output" || {
    echo "refusing to replace existing package: $output" >&2
    exit 1
}

codesign --verify --deep --strict --verbose=2 "$app"

build_root=$(mktemp -d "${TMPDIR:-/tmp}/greppy-fskit-pkg.XXXXXX")
cleanup() {
    rm -rf "$build_root"
}
trap cleanup EXIT INT TERM

payload="$build_root/payload"
component="$build_root/greppy-component.pkg"
mkdir -p \
    "$payload/Applications" \
    "$payload/usr/local/bin" \
    "$payload/usr/local/share/doc/greppy/licenses"
/usr/bin/ditto --norsrc --noextattr --noqtn \
    "$app" "$payload/Applications/GreppyWorkspaceFS.app"

for document in README.md LICENSE THIRD_PARTY.md SECURITY.md SUPPORT.md CHANGELOG.md; do
    cp "$repository_root/$document" "$payload/usr/local/share/doc/greppy/$document"
done
cp "$repository_root/Cargo.lock" "$payload/usr/local/share/doc/greppy/Cargo.lock"
cp "$repository_root/tools/release_artifacts.py" \
    "$payload/usr/local/share/doc/greppy/release_artifacts.py"
cp "$repository_root/bench/release_package_smoke.sh" \
    "$payload/usr/local/share/doc/greppy/release_package_smoke.sh"
cp "$repository_root/bench/release_daemon_stress.sh" \
    "$payload/usr/local/share/doc/greppy/release_daemon_stress.sh"
for license in "$repository_root"/licenses/*; do
    case "$(basename "$license")" in
        WINFSP-*|RIFT-*) continue ;;
    esac
    cp "$license" "$payload/usr/local/share/doc/greppy/licenses/"
done

codesign --verify --deep --strict --verbose=2 \
    "$payload/Applications/GreppyWorkspaceFS.app"
ln -s \
    /Applications/GreppyWorkspaceFS.app/Contents/Resources/bin/greppy \
    "$payload/usr/local/bin/greppy"

pkgbuild \
    --root "$payload" \
    --identifier ai.metric-space.greppy \
    --version "$version" \
    --install-location / \
    --ownership recommended \
    "$component"

mkdir -p "$(dirname "$output")"
if [ -n "$installer_identity" ]; then
    productbuild --package "$component" --sign "$installer_identity" "$output"
    pkgutil --check-signature "$output"
else
    productbuild --package "$component" "$output"
fi

payload_listing="$build_root/payload-files.txt"
pkgutil --payload-files "$output" > "$payload_listing"
grep -q '^\./Applications/GreppyWorkspaceFS.app/Contents/Resources/bin/greppy$' "$payload_listing"
grep -q '^\./Applications/GreppyWorkspaceFS.app/Contents/Resources/bin/web-runtime/bin/web-runtime$' "$payload_listing"
grep -q '^\./usr/local/bin/greppy$' "$payload_listing"
