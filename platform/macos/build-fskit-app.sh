#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
output_root=${1:?usage: build-fskit-app.sh OUTPUT_DIRECTORY}
identity=${CODE_SIGN_IDENTITY:--}
target=arm64-apple-macos15.4
app="$output_root/GreppyWorkspaceFS.app"
extension="$app/Contents/Extensions/GreppyWorkspaceFS.appex"

rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$extension/Contents/MacOS"

cd "$repository_root"
MACOSX_DEPLOYMENT_TARGET=15.4 cargo build \
    -p greppy-workspace-ffi \
    --release \
    --locked

swiftc \
    -parse-as-library \
    -target "$target" \
    -I crates/workspace-ffi/include \
    platform/macos/GreppyWorkspaceFS/*.swift \
    target/release/libgreppy_workspace_ffi.a \
    -framework FSKit \
    -framework Foundation \
    -lsqlite3 \
    -o "$extension/Contents/MacOS/GreppyWorkspaceFSExtension"

swiftc \
    -parse-as-library \
    -target "$target" \
    platform/macos/GreppyWorkspaceApp/GreppyWorkspaceApp.swift \
    -framework AppKit \
    -o "$app/Contents/MacOS/GreppyWorkspaceFS"

cp platform/macos/GreppyWorkspaceFS/Info.plist "$extension/Contents/Info.plist"
cp platform/macos/GreppyWorkspaceApp/Info.plist "$app/Contents/Info.plist"

codesign --force --timestamp=none --options runtime \
    --entitlements platform/macos/GreppyWorkspaceFS/GreppyWorkspaceFS.entitlements \
    --sign "$identity" "$extension"
codesign --force --timestamp=none --options runtime \
    --entitlements platform/macos/GreppyWorkspaceApp/GreppyWorkspaceApp.entitlements \
    --sign "$identity" "$app"

codesign --verify --deep --strict "$app"
