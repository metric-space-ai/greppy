#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 GREPPY PROVIDER VERSION OUTPUT.deb OUTPUT.rpm STAGING_ROOT" >&2
  exit 64
}

[[ $# -eq 6 ]] || usage

GREPPY_BINARY=$1
PROVIDER_BINARY=$2
VERSION=$3
DEB_OUTPUT=$4
RPM_OUTPUT=$5
STAGING_ROOT=$(realpath -m "$6")

[[ -x "$GREPPY_BINARY" ]] || { echo "greppy binary is not executable: $GREPPY_BINARY" >&2; exit 66; }
[[ -x "$PROVIDER_BINARY" ]] || { echo "workspace provider is not executable: $PROVIDER_BINARY" >&2; exit 66; }
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z.-]+)?$ ]] || {
  echo "invalid package version: $VERSION" >&2
  exit 64
}
[[ "$DEB_OUTPUT" == *.deb ]] || usage
[[ "$RPM_OUTPUT" == *.rpm ]] || usage

for output in "$DEB_OUTPUT" "$RPM_OUTPUT" "$STAGING_ROOT"; do
  [[ ! -e "$output" ]] || { echo "refusing to replace existing output: $output" >&2; exit 73; }
done

for command_name in dpkg-deb rpmbuild sha256sum; do
  command -v "$command_name" >/dev/null || {
    echo "required packaging command is unavailable: $command_name" >&2
    exit 69
  }
done

script_dir=$(cd "$(dirname "$0")" && pwd)
repository_root=$(cd "$script_dir/../.." && pwd)
# Keep the private build roots on the staging filesystem so hard-linking the
# large, embedded-model payload is reliable and does not triple temporary disk
# usage. The directory name is create-only and removed by the scoped trap.
work_root=$(mktemp -d "$(dirname "$STAGING_ROOT")/.greppy-linux-packages.XXXXXX")
trap 'rm -rf "$work_root"' EXIT

mkdir -p \
  "$STAGING_ROOT/usr/bin" \
  "$STAGING_ROOT/usr/lib/greppy/bin" \
  "$STAGING_ROOT/usr/lib/greppy/release-tests" \
  "$STAGING_ROOT/usr/lib/systemd/user" \
  "$STAGING_ROOT/usr/share/doc/greppy" \
  "$STAGING_ROOT/usr/share/licenses/greppy"

install -m 0755 "$GREPPY_BINARY" "$STAGING_ROOT/usr/lib/greppy/bin/greppy"
install -m 0755 "$PROVIDER_BINARY" "$STAGING_ROOT/usr/lib/greppy/bin/greppy-workspace-provider"
install -m 0644 "$repository_root/platform/linux/greppy-workspace-provider.service" \
  "$STAGING_ROOT/usr/lib/systemd/user/greppy-workspace-provider.service"
ln -s ../lib/greppy/bin/greppy "$STAGING_ROOT/usr/bin/greppy"
install -m 0755 "$repository_root/bench/release_package_smoke.sh" \
  "$STAGING_ROOT/usr/lib/greppy/release-tests/release_package_smoke.sh"
install -m 0755 "$repository_root/bench/release_daemon_stress.sh" \
  "$STAGING_ROOT/usr/lib/greppy/release-tests/release_daemon_stress.sh"
install -m 0644 "$repository_root/tools/release_artifacts.py" \
  "$STAGING_ROOT/usr/lib/greppy/release-tests/release_artifacts.py"
install -m 0644 \
  "$repository_root/README.md" \
  "$repository_root/SECURITY.md" \
  "$repository_root/SUPPORT.md" \
  "$repository_root/CHANGELOG.md" \
  "$repository_root/THIRD_PARTY.md" \
  "$repository_root/Cargo.lock" \
  "$STAGING_ROOT/usr/share/doc/greppy/"
install -m 0644 "$repository_root/LICENSE" \
  "$STAGING_ROOT/usr/share/licenses/greppy/LICENSE"
for license_path in "$repository_root"/licenses/*; do
  case "$(basename "$license_path")" in
    WINFSP-*|RIFT-*) continue ;;
  esac
  install -m 0644 "$license_path" "$STAGING_ROOT/usr/share/licenses/greppy/"
done

"$STAGING_ROOT/usr/lib/greppy/bin/greppy" --version | grep -Fx "greppy $VERSION" >/dev/null || {
  echo "greppy version does not match package version $VERSION" >&2
  exit 65
}
"$STAGING_ROOT/usr/lib/greppy/bin/greppy-workspace-provider" --version >/dev/null

deb_root="$work_root/deb"
mkdir -p "$deb_root/DEBIAN"
cp -al "$STAGING_ROOT/." "$deb_root/"
installed_size_kib=$(du -sk "$STAGING_ROOT" | awk '{print $1}')
cat > "$deb_root/DEBIAN/control" <<EOF
Package: greppy
Version: $VERSION
Section: devel
Priority: optional
Architecture: amd64
Maintainer: Metric Space <opensource@metric-space.ai>
Depends: fuse3
Installed-Size: $installed_size_kib
Homepage: https://github.com/metric-space-ai/greppy
Description: Graph-native code navigation and portable CoW agent workspaces
 Greppy ships its Rust FUSE3 provider beside the CLI. It does not require
 reflinks or a particular host filesystem.
EOF
cat > "$deb_root/DEBIAN/postinst" <<'EOF'
#!/bin/sh
set -e
if [ ! -c /dev/fuse ]; then
  echo 'greppy: /dev/fuse is unavailable; enable the FUSE kernel component before running greppy workspace setup' >&2
fi
if ! command -v fusermount3 >/dev/null 2>&1; then
  echo 'greppy: fusermount3 is unavailable; reinstall the fuse3 dependency' >&2
fi
exit 0
EOF
chmod 0755 "$deb_root/DEBIAN/postinst"
mkdir -p "$(dirname "$DEB_OUTPUT")"
dpkg-deb --root-owner-group --build "$deb_root" "$DEB_OUTPUT" >/dev/null

rpm_top="$work_root/rpmbuild"
mkdir -p "$rpm_top/BUILD" "$rpm_top/BUILDROOT" "$rpm_top/RPMS" "$rpm_top/SOURCES" "$rpm_top/SPECS" "$rpm_top/SRPMS"
rpm_spec="$rpm_top/SPECS/greppy.spec"
cat > "$rpm_spec" <<'EOF'
%global __strip /bin/true
%global _build_id_links none
Name: greppy
Version: %{greppy_version}
Release: 1%{?dist}
Summary: Graph-native code navigation and portable CoW agent workspaces
License: Apache-2.0
URL: https://github.com/metric-space-ai/greppy
Requires: fuse3
BuildArch: x86_64

%description
Greppy ships its Rust FUSE3 provider beside the CLI. It does not require
reflinks or a particular host filesystem.

%prep

%build

%install
mkdir -p %{buildroot}
cp -a %{payload_root}/. %{buildroot}/

%post
if [ ! -c /dev/fuse ]; then
  echo 'greppy: /dev/fuse is unavailable; enable the FUSE kernel component before running greppy workspace setup' >&2
fi
if ! command -v fusermount3 >/dev/null 2>&1; then
  echo 'greppy: fusermount3 is unavailable; reinstall the fuse3 dependency' >&2
fi

%files
/usr/bin/greppy
/usr/lib/greppy
/usr/lib/systemd/user/greppy-workspace-provider.service
/usr/share/doc/greppy
/usr/share/licenses/greppy

%changelog
* Thu Aug 27 2026 Metric Space <opensource@metric-space.ai> - %{greppy_version}-1
- Portable CoW workspace package.
EOF

rpmbuild \
  --define "_topdir $rpm_top" \
  --define "payload_root $STAGING_ROOT" \
  --define "greppy_version $VERSION" \
  -bb "$rpm_spec" >/dev/null
built_rpm=$(find "$rpm_top/RPMS" -type f -name '*.rpm' -print -quit)
[[ -n "$built_rpm" ]] || { echo "rpmbuild produced no binary package" >&2; exit 70; }
mkdir -p "$(dirname "$RPM_OUTPUT")"
mv "$built_rpm" "$RPM_OUTPUT"

dpkg-deb --field "$DEB_OUTPUT" Package Version Architecture Depends >/dev/null
rpm -qp --qf '%{NAME} %{VERSION} %{ARCH}\n' "$RPM_OUTPUT" | grep -Fx "greppy $VERSION x86_64" >/dev/null
sha256sum "$DEB_OUTPUT" "$RPM_OUTPUT"
