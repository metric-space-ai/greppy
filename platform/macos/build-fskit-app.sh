#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
output_root=${1:?usage: build-fskit-app.sh OUTPUT_DIRECTORY}
identity=${CODE_SIGN_IDENTITY:--}
target=arm64-apple-macos15.4
app="$output_root/GreppyWorkspaceFS.app"
extension="$app/Contents/Extensions/GreppyWorkspaceFS.appex"
cargo_target="$repository_root/target/fskit-aarch64-macos15.4"
ffi_archive="$cargo_target/aarch64-apple-darwin/release/libgreppy_workspace_ffi.a"
cli_binary=${GREPPY_CLI_BINARY:-}
if [ -n "$cli_binary" ]; then
    package_version=$("$cli_binary" --version | awk '{print $NF}')
else
    package_version=${GREPPY_PACKAGE_VERSION:-$(
        cargo metadata --manifest-path "$repository_root/Cargo.toml" \
            --no-deps --format-version 1 |
            python3 -c 'import json,sys; data=json.load(sys.stdin); print(next(p["version"] for p in data["packages"] if p["name"] == "greppy"))'
    )}
fi
case "$package_version" in
    ''|*[!0-9A-Za-z.-]*)
        echo "invalid FSKit package version: $package_version" >&2
        exit 64
        ;;
esac

test ! -e "$app" || {
    echo "refusing to replace existing FSKit application: $app" >&2
    exit 1
}
mkdir -p "$app/Contents/MacOS" "$extension/Contents/MacOS"
if [ -n "$cli_binary" ]; then
    test -f "$cli_binary" && test -x "$cli_binary" || {
        echo "GREPPY_CLI_BINARY is not an executable file: $cli_binary" >&2
        exit 1
    }
    mkdir -p "$app/Contents/Resources/bin"
    cp "$cli_binary" "$app/Contents/Resources/bin/greppy"
    chmod 0755 "$app/Contents/Resources/bin/greppy"
fi

cd "$repository_root"
MACOSX_DEPLOYMENT_TARGET=15.4 CARGO_TARGET_DIR="$cargo_target" cargo build \
    -p greppy-workspace-ffi \
    --target aarch64-apple-darwin \
    --release \
    --locked

swiftc \
    -parse-as-library \
    -target "$target" \
    -I crates/workspace-ffi/include \
    platform/macos/GreppyWorkspaceFS/*.swift \
    "$ffi_archive" \
    -framework FSKit \
    -framework Foundation \
    -lsqlite3 \
    -Xlinker -fatal_warnings \
    -o "$extension/Contents/MacOS/GreppyWorkspaceFSExtension"

swiftc \
    -parse-as-library \
    -target "$target" \
    platform/macos/GreppyWorkspaceApp/GreppyWorkspaceApp.swift \
    -framework AppKit \
    -Xlinker -fatal_warnings \
    -o "$app/Contents/MacOS/GreppyWorkspaceFS"

verify_minos() {
    xcrun vtool -show-build "$1" | awk '
        $1 == "minos" { found = 1; if ($2 != "15.4") invalid = 1 }
        END { exit !(found && !invalid) }
    ' || {
        echo "macOS binary does not target exactly macOS 15.4: $1" >&2
        exit 1
    }
}

verify_minos "$extension/Contents/MacOS/GreppyWorkspaceFSExtension"
verify_minos "$app/Contents/MacOS/GreppyWorkspaceFS"

cp platform/macos/GreppyWorkspaceFS/Info.plist "$extension/Contents/Info.plist"
cp platform/macos/GreppyWorkspaceApp/Info.plist "$app/Contents/Info.plist"
/usr/libexec/PlistBuddy -c \
    "Set :CFBundleShortVersionString $package_version" \
    "$extension/Contents/Info.plist"
/usr/libexec/PlistBuddy -c \
    "Set :CFBundleShortVersionString $package_version" \
    "$app/Contents/Info.plist"

sign_bundle() {
    entitlements=$1
    bundle=$2
    if [ "$identity" = "-" ]; then
        codesign --force --timestamp=none --options runtime \
            --entitlements "$entitlements" --sign "$identity" "$bundle"
    else
        codesign --force --timestamp --options runtime \
            --entitlements "$entitlements" --sign "$identity" "$bundle"
    fi
}

sign_bundle \
    platform/macos/GreppyWorkspaceFS/GreppyWorkspaceFS.entitlements \
    "$extension"
if [ -n "$cli_binary" ]; then
    if [ "$identity" = "-" ]; then
        codesign --force --timestamp=none --options runtime \
            --sign "$identity" "$app/Contents/Resources/bin/greppy"
    else
        codesign --force --timestamp --options runtime \
            --sign "$identity" "$app/Contents/Resources/bin/greppy"
    fi
fi
sign_bundle \
    platform/macos/GreppyWorkspaceApp/GreppyWorkspaceApp.entitlements \
    "$app"

codesign --verify --deep --strict "$app"
