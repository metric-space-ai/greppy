# Shared dest validation for package/uninstall/sign/notarize.
# Sourced by scripts in this directory. $0 is the calling script.
# Never rm -rf. Never follow dest or member symlinks.

web_runtime_die() {
  echo "web-runtime-dest-guard: $*" >&2
  exit 2
}

web_runtime_script_dir() {
  CDPATH= cd -- "$(dirname -- "$0")" && pwd -P
}

web_runtime_repo_root() {
  CDPATH= cd -- "$(web_runtime_script_dir)/../../.." && pwd -P
}

web_runtime_uid() {
  id -u
}

web_runtime_owner_uid() {
  if stat -f %u "$1" >/dev/null 2>&1; then
    stat -f %u "$1"
  else
    stat -c %u "$1"
  fi
}

web_runtime_strip_slash() {
  dest=$1
  if [ -z "$dest" ]; then
    echo ""
    return 0
  fi
  while [ "$dest" != / ] && [ "${dest%/}" != "$dest" ]; do
    dest=${dest%/}
  done
  printf '%s' "$dest"
}

web_runtime_require_absolute() {
  dest=$1
  case "$dest" in
    /*) ;;
    *) web_runtime_die "refusing relative or unresolved dest: $dest" ;;
  esac
  case "$dest" in
    *'*'* | *'?'* | *'['* | *']'*)
      web_runtime_die "refusing ambiguous dest: $dest"
      ;;
  esac
}

web_runtime_refuse_dot_components() {
  dest=$1
  oldifs=$IFS
  IFS=/
  # shellcheck disable=SC2086
  set -- $dest
  IFS=$oldifs
  first=1
  for part in "$@"; do
    if [ "$first" = 1 ]; then
      first=0
      # Leading slash produces an empty first component.
      continue
    fi
    if [ -z "$part" ] || [ "$part" = "." ] || [ "$part" = ".." ]; then
      web_runtime_die "refusing dest with empty, . or .. components: $dest"
    fi
  done
}

web_runtime_basename_allowed() {
  base=$1
  case "$base" in
    web-runtime-dist | web-runtime-dist-* | greppy-web-dist-* | greppy-web-signed-*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

web_runtime_parent_allowed() {
  parent=$1
  repo=$(web_runtime_repo_root)
  tmpdir=${TMPDIR:-/tmp}
  tmp_canon=
  if [ -d "$tmpdir" ]; then
    tmp_canon=$(CDPATH= cd -- "$tmpdir" && pwd -P)
  fi
  case "$parent" in
    "$repo/target" | "$repo/crates/web-runtime/target")
      return 0
      ;;
    /tmp | /private/tmp)
      return 0
      ;;
    /var/folders/*/*/T | /private/var/folders/*/*/T)
      return 0
      ;;
  esac
  if [ -n "$tmp_canon" ] && [ "$parent" = "$tmp_canon" ]; then
    return 0
  fi
  return 1
}

web_runtime_refuse_sensitive() {
  dest=$1
  repo=$(web_runtime_repo_root)
  home=${HOME:-}
  case "$dest" in
    / | // | /.)
      web_runtime_die "refusing filesystem root dest"
      ;;
  esac
  if [ -n "$home" ] && [ "$dest" = "$home" ]; then
    web_runtime_die "refusing home directory dest"
  fi
  if [ "$dest" = "$repo" ]; then
    web_runtime_die "refusing repository root dest"
  fi
  if [ "$dest" = "$repo/crates" ] || [ "$dest" = "$repo/crates/web-runtime" ]; then
    web_runtime_die "refusing workspace/crate root dest"
  fi
}

web_runtime_check_owned_dir() {
  path=$1
  [ -e "$path" ] || web_runtime_die "missing $path"
  [ -L "$path" ] && web_runtime_die "refusing symlink: $path"
  [ -d "$path" ] || web_runtime_die "not a directory: $path"
  owner=$(web_runtime_owner_uid "$path")
  me=$(web_runtime_uid)
  [ "$owner" = "$me" ] || web_runtime_die "refusing non-owned directory $path (uid $owner != $me)"
}

web_runtime_parent_usable() {
  parent=$1
  me=$(web_runtime_uid)
  owner=$(web_runtime_owner_uid "$parent")
  if [ "$owner" = "$me" ]; then
    return 0
  fi
  if [ -w "$parent" ]; then
    case "$parent" in
      /tmp | /private/tmp | /var/folders/*/*/T | /private/var/folders/*/*/T)
        return 0
        ;;
    esac
  fi
  return 1
}

web_runtime_physical_parent() {
  parent=$1
  [ -d "$parent" ] || web_runtime_die "parent does not exist: $parent"
  phys=$(CDPATH= cd -- "$parent" && pwd -P) ||
    web_runtime_die "unresolved parent: $parent"
  [ -n "$phys" ] || web_runtime_die "unresolved parent: $parent"
  [ -L "$phys" ] && web_runtime_die "refusing symlink parent: $phys"
  [ -d "$phys" ] || web_runtime_die "resolved parent is not a directory: $phys"
  printf '%s' "$phys"
}

web_runtime_stamp_name() {
  printf '%s' ".greppy-web-runtime-dist"
}

web_runtime_is_owned_dist() {
  dest=$1
  stamp="$dest/$(web_runtime_stamp_name)"
  [ -f "$stamp" ] || return 1
  [ -L "$stamp" ] && return 1
  grep -q 'greppy.web-runtime.package.v1' "$stamp" || return 1
  [ -f "$dest/provenance.json" ] || return 1
  [ -L "$dest/provenance.json" ] && return 1
  grep -q 'greppy.web-runtime.package.v1' "$dest/provenance.json" || return 1
  for bin in web-runtime-supervisor web-controller-worker web-content-worker; do
    [ -f "$dest/bin/$bin" ] || return 1
    [ -L "$dest/bin/$bin" ] && return 1
  done
  return 0
}

web_runtime_known_members() {
  printf '%s
'     ".greppy-web-runtime-dist"     "README.txt"     "UNSIGNED"     "SHA256SUMS"     "sbom.json"     "provenance.json"     "LICENSE"     "SIGNING_RECEIPT"     "SIGNING_SKIPPED"     "SIGNING_STATUS"     "NOTARIZATION_RECEIPT"     "NOTARIZATION_SKIPPED"     "NOTARIZED_UNSIGNED"     "bin/web-runtime-supervisor"     "bin/web-controller-worker"     "bin/web-content-worker"     "previous/web-runtime-supervisor"     "previous/web-controller-worker"     "previous/web-content-worker"
}

web_runtime_validate_dest_shape() {
  dest=$(web_runtime_strip_slash "$1")
  [ -n "$dest" ] || web_runtime_die "empty dest"
  web_runtime_require_absolute "$dest"
  web_runtime_refuse_dot_components "$dest"
  web_runtime_refuse_sensitive "$dest"
  base=$(basename "$dest")
  [ -n "$base" ] && [ "$base" != / ] && [ "$base" != "." ] && [ "$base" != ".." ] ||
    web_runtime_die "refusing dest with empty basename: $dest"
  web_runtime_basename_allowed "$base" ||
    web_runtime_die "dest basename is not a web-runtime dist name: $base"
  parent=$(dirname "$dest")
  phys_parent=$(web_runtime_physical_parent "$parent")
  web_runtime_parent_allowed "$phys_parent" ||
    web_runtime_die "dest parent is not a web-runtime staging directory: $phys_parent"
  web_runtime_parent_usable "$phys_parent" ||
    web_runtime_die "refusing non-owned dest parent: $phys_parent"
  dest="$phys_parent/$base"
  web_runtime_refuse_sensitive "$dest"
  printf '%s' "$dest"
}

web_runtime_require_package_dest() {
  raw=${1:-}
  repo=$(web_runtime_repo_root)
  if [ -z "$raw" ]; then
    raw="$repo/target/web-runtime-dist"
  fi
  dest=$(web_runtime_validate_dest_shape "$raw")
  if [ -L "$dest" ]; then
    web_runtime_die "refusing symlink dest: $dest"
  fi
  if [ -e "$dest" ]; then
    web_runtime_check_owned_dir "$dest"
    if web_runtime_is_owned_dist "$dest"; then
      web_runtime_remove_owned_dist_files "$dest"
      leftover=$(ls -A "$dest" 2>/dev/null || true)
      if [ -n "$leftover" ]; then
        web_runtime_die "refusing to overwrite dist with unexpected members: $leftover"
      fi
    else
      leftover=$(ls -A "$dest" 2>/dev/null || true)
      if [ -n "$leftover" ]; then
        web_runtime_die "refusing to overwrite non-package directory: $dest"
      fi
    fi
  fi
  mkdir -p "$dest/bin"
  web_runtime_check_owned_dir "$dest"
  printf '%s' "$dest"
}

web_runtime_require_uninstall_dest() {
  raw=${1:-}
  [ -n "$raw" ] || web_runtime_die "uninstall dest is required"
  dest=$(web_runtime_validate_dest_shape "$raw")
  [ -e "$dest" ] || web_runtime_die "nothing to uninstall: $dest"
  web_runtime_check_owned_dir "$dest"
  web_runtime_is_owned_dist "$dest" ||
    web_runtime_die "refusing to remove directory that is not a web-runtime dist: $dest"
  printf '%s' "$dest"
}

web_runtime_remove_owned_dist_files() {
  dest=$1
  web_runtime_is_owned_dist "$dest" ||
    web_runtime_die "not a web-runtime dist: $dest"
  me=$(web_runtime_uid)
  for member in     .greppy-web-runtime-dist     README.txt     UNSIGNED     SHA256SUMS     sbom.json     provenance.json     LICENSE     SIGNING_RECEIPT     SIGNING_SKIPPED     SIGNING_STATUS     NOTARIZATION_RECEIPT     NOTARIZATION_SKIPPED     NOTARIZED_UNSIGNED     bin/web-runtime-supervisor     bin/web-controller-worker     bin/web-content-worker     previous/web-runtime-supervisor     previous/web-controller-worker     previous/web-content-worker
  do
    path="$dest/$member"
    if [ -L "$path" ]; then
      web_runtime_die "refusing symlink member: $path"
    fi
    if [ -e "$path" ]; then
      owner=$(web_runtime_owner_uid "$path")
      [ "$owner" = "$me" ] || web_runtime_die "refusing non-owned member $path"
      [ -f "$path" ] || web_runtime_die "refusing non-file member $path"
      rm -f "$path"
    fi
  done
  if [ -d "$dest/bin" ]; then
    leftover=$(ls -A "$dest/bin" 2>/dev/null || true)
    if [ -n "$leftover" ]; then
      web_runtime_die "refusing to remove dest with unexpected bin members: $leftover"
    fi
    rmdir "$dest/bin"
  fi
  if [ -d "$dest/previous" ]; then
    leftover=$(ls -A "$dest/previous" 2>/dev/null || true)
    if [ -n "$leftover" ]; then
      web_runtime_die "refusing to remove dest with unexpected previous members: $leftover"
    fi
    rmdir "$dest/previous"
  fi
}

web_runtime_require_existing_dist() {
  raw=${1:-}
  label=${2:-dist}
  [ -n "$raw" ] || web_runtime_die "$label dest is required"
  dest=$(web_runtime_validate_dest_shape "$raw")
  [ -e "$dest" ] || web_runtime_die "missing $label dest: $dest"
  web_runtime_check_owned_dir "$dest"
  web_runtime_is_owned_dist "$dest" ||
    web_runtime_die "$label is not a stamped web-runtime dist: $dest"
  printf '%s' "$dest"
}

web_runtime_copy_regular_file() {
  copy_src=$1
  copy_dest=$2
  [ -e "$copy_src" ] || web_runtime_die "missing source file: $copy_src"
  [ -L "$copy_src" ] && web_runtime_die "refusing symlink source: $copy_src"
  [ -f "$copy_src" ] || web_runtime_die "source is not a file: $copy_src"
  [ -L "$copy_dest" ] && web_runtime_die "refusing symlink dest file: $copy_dest"
  copy_parent=$(dirname "$copy_dest")
  mkdir -p "$copy_parent"
  [ -L "$copy_parent" ] && web_runtime_die "refusing symlink dest parent: $copy_parent"
  web_runtime_check_owned_dir "$copy_parent"
  copy_tmp="$copy_dest.greppy-tmp"
  [ -L "$copy_tmp" ] && web_runtime_die "refusing symlink temp: $copy_tmp"
  cp "$copy_src" "$copy_tmp"
  mv "$copy_tmp" "$copy_dest"
}

web_runtime_uninstall_owned_dist() {
  dest=$1
  web_runtime_remove_owned_dist_files "$dest"
  leftover=$(ls -A "$dest" 2>/dev/null || true)
  if [ -n "$leftover" ]; then
    web_runtime_die "refusing to remove dest with unexpected members: $leftover"
  fi
  rmdir "$dest"
  tarball="$(dirname "$dest")/$(basename "$dest").tar.gz"
  if [ -e "$tarball" ]; then
    [ -L "$tarball" ] && web_runtime_die "refusing symlink archive: $tarball"
    [ -f "$tarball" ] || web_runtime_die "archive is not a file: $tarball"
    owner=$(web_runtime_owner_uid "$tarball")
    me=$(web_runtime_uid)
    [ "$owner" = "$me" ] || web_runtime_die "refusing non-owned archive $tarball"
    rm -f "$tarball"
  fi
}

web_runtime_write_stamp() {
  dest=$1
  printf '%s\n' "greppy.web-runtime.package.v1" >"$dest/$(web_runtime_stamp_name)"
}
