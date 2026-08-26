#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
output_root=${1:?usage: build-fskit-app.sh OUTPUT_DIRECTORY}
identity=${CODE_SIGN_IDENTITY:--}
target=arm64-apple-macos15.4
app="$output_root/GreppyWorkspaceFS.app"
extension="$app/Contents/Extensions/GreppyWorkspaceFS.appex"
cargo_target="$output_root/cargo-target-aarch64-macos15.4"
ffi_archive="$cargo_target/aarch64-apple-darwin/release/libgreppy_workspace_ffi.a"

rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$extension/Contents/MacOS"

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

codesign --force --timestamp=none --options runtime \
    --entitlements platform/macos/GreppyWorkspaceFS/GreppyWorkspaceFS.entitlements \
    --sign "$identity" "$extension"
codesign --force --timestamp=none --options runtime \
    --entitlements platform/macos/GreppyWorkspaceApp/GreppyWorkspaceApp.entitlements \
    --sign "$identity" "$app"

codesign --verify --deep --strict "$app"
