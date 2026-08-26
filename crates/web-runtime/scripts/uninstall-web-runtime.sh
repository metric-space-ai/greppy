#!/bin/sh
set -eu
# Removes a local web-runtime dist directory produced by package-web-runtime.sh.
dest="${1:?usage: uninstall-web-runtime.sh DISTDIR}"
rm -rf "$dest"
rm -f "${dest}.tar.gz"
echo "uninstalled $dest"
