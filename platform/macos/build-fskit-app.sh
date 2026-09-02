#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
output_root=${1:?usage: build-fskit-app.sh OUTPUT_DIRECTORY}
identity=${CODE_SIGN_IDENTITY:--}
fskit_profile=${FSKIT_PROVISIONING_PROFILE:-}
app_profile=${APP_PROVISIONING_PROFILE:-}
target=arm64-apple-macos15.4
app="$output_root/GreppyWorkspaceFS.app"
extension="$app/Contents/Extensions/GreppyWorkspaceFS.appex"
app_bundle_id=ai.metricspace.greppy.workspacefs
extension_bundle_id=ai.metricspace.greppy.workspacefs.extension
application_group=group.ai.metricspace.greppy
cargo_target="$repository_root/target/fskit-aarch64-macos15.4"
ffi_archive="$cargo_target/aarch64-apple-darwin/release/libgreppy_workspace_ffi.a"
cli_binary=${GREPPY_CLI_BINARY:-}
web_runtime_dist=${GREPPY_WEB_RUNTIME_DIST:-}
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

extension_entitlements=platform/macos/GreppyWorkspaceFS/GreppyWorkspaceFS.entitlements
app_entitlements=platform/macos/GreppyWorkspaceApp/GreppyWorkspaceApp.entitlements
if [ "$identity" != "-" ]; then
    test -n "$fskit_profile" || {
        echo "signed FSKit builds require FSKIT_PROVISIONING_PROFILE" >&2
        exit 64
    }
    test -f "$fskit_profile" || {
        echo "FSKit provisioning profile is not a regular file: $fskit_profile" >&2
        exit 64
    }
    test -n "$app_profile" || {
        echo "signed FSKit builds require APP_PROVISIONING_PROFILE" >&2
        exit 64
    }
    test -f "$app_profile" || {
        echo "FSKit host-app provisioning profile is not a regular file: $app_profile" >&2
        exit 64
    }
    signing_temp=$(mktemp -d -t greppy-fskit-signing)
    resolved_extension_entitlements="$signing_temp/extension.entitlements"
    resolved_app_entitlements="$signing_temp/app.entitlements"
    extension_profile_plist="$signing_temp/extension-profile.plist"
    app_profile_plist="$signing_temp/app-profile.plist"
    signing_certificate_pem="$signing_temp/signing-certificate.pem"
    signing_certificate_der="$signing_temp/signing-certificate.der"
    cleanup_signing_inputs() {
        rm -f \
            "$resolved_extension_entitlements" \
            "$resolved_app_entitlements" \
            "$extension_profile_plist" \
            "$app_profile_plist" \
            "$signing_certificate_pem" \
            "$signing_certificate_der"
        rmdir "$signing_temp" 2>/dev/null || true
    }
    trap cleanup_signing_inputs EXIT
    trap 'exit 1' HUP INT TERM
    /usr/bin/security find-certificate -c "$identity" -p > "$signing_certificate_pem"
    test -s "$signing_certificate_pem" || {
        echo "cannot export selected FSKit signing certificate: $identity" >&2
        exit 64
    }
    /usr/bin/openssl x509 -in "$signing_certificate_pem" -outform DER \
        -out "$signing_certificate_der"
    /usr/bin/security cms -D -i "$fskit_profile" -o "$extension_profile_plist"
    /usr/bin/security cms -D -i "$app_profile" -o "$app_profile_plist"
    profile_team_id=$(/usr/bin/python3 \
        "$repository_root/tools/validate_macos_fskit_profile.py" \
        --plist "$extension_profile_plist" \
        --bundle-id "$extension_bundle_id" \
        --application-group "$application_group" \
        --role fskit-extension \
        --signing-certificate-der "$signing_certificate_der")
    app_profile_team_id=$(/usr/bin/python3 \
        "$repository_root/tools/validate_macos_fskit_profile.py" \
        --plist "$app_profile_plist" \
        --bundle-id "$app_bundle_id" \
        --application-group "$application_group" \
        --role app \
        --signing-certificate-der "$signing_certificate_der")
    test "$app_profile_team_id" = "$profile_team_id" || {
        echo "FSKit host-app and extension profiles belong to different teams" >&2
        exit 64
    }
    cp "$extension_entitlements" "$resolved_extension_entitlements"
    /usr/libexec/PlistBuddy -c \
        "Add :com.apple.application-identifier string $profile_team_id.$extension_bundle_id" \
        "$resolved_extension_entitlements"
    /usr/libexec/PlistBuddy -c \
        "Add :com.apple.developer.team-identifier string $profile_team_id" \
        "$resolved_extension_entitlements"
    extension_entitlements=$resolved_extension_entitlements
    cp "$app_entitlements" "$resolved_app_entitlements"
    /usr/libexec/PlistBuddy -c \
        "Add :com.apple.application-identifier string $profile_team_id.$app_bundle_id" \
        "$resolved_app_entitlements"
    /usr/libexec/PlistBuddy -c \
        "Add :com.apple.developer.team-identifier string $profile_team_id" \
        "$resolved_app_entitlements"
    app_entitlements=$resolved_app_entitlements
fi

test ! -e "$app" || {
    echo "refusing to replace existing FSKit application: $app" >&2
    exit 1
}
mkdir -p "$app/Contents/MacOS" "$extension/Contents/MacOS"
if [ "$identity" != "-" ]; then
    cp "$app_profile" "$app/Contents/embedded.provisionprofile"
    cp "$fskit_profile" "$extension/Contents/embedded.provisionprofile"
fi
if [ -n "$cli_binary" ]; then
    test -f "$cli_binary" && test -x "$cli_binary" || {
        echo "GREPPY_CLI_BINARY is not an executable file: $cli_binary" >&2
        exit 1
    }
    mkdir -p "$app/Contents/Resources/bin"
    cp "$cli_binary" "$app/Contents/Resources/bin/greppy"
    chmod 0755 "$app/Contents/Resources/bin/greppy"
fi
if [ -n "$web_runtime_dist" ]; then
    test -d "$web_runtime_dist" && test ! -L "$web_runtime_dist" || {
        echo "GREPPY_WEB_RUNTIME_DIST is not a regular directory: $web_runtime_dist" >&2
        exit 1
    }
    test -f "$web_runtime_dist/.greppy-web-runtime-dist" || {
        echo "GREPPY_WEB_RUNTIME_DIST is missing its package stamp" >&2
        exit 1
    }
    test -x "$web_runtime_dist/bin/web-runtime" || {
        echo "GREPPY_WEB_RUNTIME_DIST is missing bin/web-runtime" >&2
        exit 1
    }
    mkdir -p "$app/Contents/Resources/bin"
    /usr/bin/ditto --norsrc --noextattr --noqtn \
        "$web_runtime_dist" "$app/Contents/Resources/bin/web-runtime"
    chmod 0755 "$app/Contents/Resources/bin/web-runtime/bin/web-runtime"
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
    "$extension_entitlements" \
    "$extension"
if [ "$identity" != "-" ]; then
    signed_team_id=$(codesign -dvvv "$extension" 2>&1 | \
        awk -F= '$1 == "TeamIdentifier" { print $2; exit }')
    test "$signed_team_id" = "$profile_team_id" || {
        echo "FSKit signing identity team $signed_team_id does not match profile team $profile_team_id" >&2
        exit 1
    }
fi
if [ -n "$cli_binary" ]; then
    if [ "$identity" = "-" ]; then
        codesign --force --timestamp=none --options runtime \
            --sign "$identity" "$app/Contents/Resources/bin/greppy"
    else
        codesign --force --timestamp --options runtime \
            --sign "$identity" "$app/Contents/Resources/bin/greppy"
    fi
fi
if [ -n "$web_runtime_dist" ]; then
    # The release workflow signs the source executable before the dist is
    # hashed. Re-signing after the copy would invalidate SHA256SUMS and the
    # dist SBOM, so require and preserve that exact signed image here.
    codesign --verify --strict --verbose=2 \
        "$app/Contents/Resources/bin/web-runtime/bin/web-runtime"
fi
sign_bundle \
    "$app_entitlements" \
    "$app"
if [ "$identity" != "-" ]; then
    signed_app_team_id=$(codesign -dvvv "$app" 2>&1 | \
        awk -F= '$1 == "TeamIdentifier" { print $2; exit }')
    test "$signed_app_team_id" = "$profile_team_id" || {
        echo "FSKit host-app signing identity team $signed_app_team_id does not match profile team $profile_team_id" >&2
        exit 1
    }
fi

codesign --verify --deep --strict "$app"
