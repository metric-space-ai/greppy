# Shared dest validation for package/uninstall/sign/notarize/install/upgrade/rollback.
# Sourced by scripts in this directory. $0 is the calling script.
# Never rm -rf. Never follow dest or member directory/file symlinks.
# Mutating operations preflight the whole tree, then commit via a sibling
# staging directory so a later failure cannot leave dest half-erased.
# Function-local paths use grd_* names so sourced helpers cannot clobber
# the caller's dest/src/staging variables.

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
  grd_dest=$1
  if [ -z "$grd_dest" ]; then
    echo ""
    return 0
  fi
  while [ "$grd_dest" != / ] && [ "${grd_dest%/}" != "$grd_dest" ]; do
    grd_dest=${grd_dest%/}
  done
  printf '%s' "$grd_dest"
}

web_runtime_require_absolute() {
  grd_dest=$1
  case "$grd_dest" in
    /*) ;;
    *) web_runtime_die "refusing relative or unresolved dest: $grd_dest" ;;
  esac
  case "$grd_dest" in
    *'*'* | *'?'* | *'['* | *']'*)
      web_runtime_die "refusing ambiguous dest: $grd_dest"
      ;;
  esac
}

web_runtime_refuse_dot_components() {
  grd_dest=$1
  oldifs=$IFS
  IFS=/
  # shellcheck disable=SC2086
  set -- $grd_dest
  IFS=$oldifs
  first=1
  for part in "$@"; do
    if [ "$first" = 1 ]; then
      first=0
      continue
    fi
    if [ -z "$part" ] || [ "$part" = "." ] || [ "$part" = ".." ]; then
      web_runtime_die "refusing dest with empty, . or .. components: $grd_dest"
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
  grd_parent=$1
  repo=$(web_runtime_repo_root)
  tmpdir=${TMPDIR:-/tmp}
  tmp_canon=
  if [ -d "$tmpdir" ]; then
    tmp_canon=$(CDPATH= cd -- "$tmpdir" && pwd -P)
  fi
  case "$grd_parent" in
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
  if [ -n "$tmp_canon" ] && [ "$grd_parent" = "$tmp_canon" ]; then
    return 0
  fi
  return 1
}

web_runtime_refuse_sensitive() {
  grd_dest=$1
  repo=$(web_runtime_repo_root)
  home=${HOME:-}
  case "$grd_dest" in
    / | // | /.)
      web_runtime_die "refusing filesystem root dest"
      ;;
  esac
  if [ -n "$home" ] && [ "$grd_dest" = "$home" ]; then
    web_runtime_die "refusing home directory dest"
  fi
  if [ "$grd_dest" = "$repo" ]; then
    web_runtime_die "refusing repository root dest"
  fi
  if [ "$grd_dest" = "$repo/crates" ] || [ "$grd_dest" = "$repo/crates/web-runtime" ]; then
    web_runtime_die "refusing workspace/crate root dest"
  fi
}

# -L must be checked before -d/-f: -d follows parent and member symlinks,
# which is the class that let dest/bin point at an external canary dir.
web_runtime_check_owned_real_dir() {
  grd_path=$1
  if [ -L "$grd_path" ]; then
    web_runtime_die "refusing symlink directory: $grd_path"
  fi
  [ -e "$grd_path" ] || web_runtime_die "missing $grd_path"
  [ -d "$grd_path" ] || web_runtime_die "not a directory: $grd_path"
  owner=$(web_runtime_owner_uid "$grd_path")
  me=$(web_runtime_uid)
  [ "$owner" = "$me" ] || web_runtime_die "refusing non-owned directory $grd_path (uid $owner != $me)"
}

web_runtime_check_owned_dir() {
  web_runtime_check_owned_real_dir "$1"
}

web_runtime_check_owned_regular_file() {
  grd_path=$1
  if [ -L "$grd_path" ]; then
    web_runtime_die "refusing symlink member: $grd_path"
  fi
  [ -e "$grd_path" ] || web_runtime_die "missing $grd_path"
  [ -f "$grd_path" ] || web_runtime_die "refusing non-file member $grd_path"
  owner=$(web_runtime_owner_uid "$grd_path")
  me=$(web_runtime_uid)
  [ "$owner" = "$me" ] || web_runtime_die "refusing non-owned member $grd_path"
}

web_runtime_parent_usable() {
  grd_parent=$1
  me=$(web_runtime_uid)
  owner=$(web_runtime_owner_uid "$grd_parent")
  if [ "$owner" = "$me" ]; then
    return 0
  fi
  if [ -w "$grd_parent" ]; then
    case "$grd_parent" in
      /tmp | /private/tmp | /var/folders/*/*/T | /private/var/folders/*/*/T)
        return 0
        ;;
    esac
  fi
  return 1
}

web_runtime_physical_parent() {
  grd_parent=$1
  [ -d "$grd_parent" ] || web_runtime_die "parent does not exist: $grd_parent"
  phys=$(CDPATH= cd -- "$grd_parent" && pwd -P) ||
    web_runtime_die "unresolved parent: $grd_parent"
  [ -n "$phys" ] || web_runtime_die "unresolved parent: $grd_parent"
  [ -L "$phys" ] && web_runtime_die "refusing symlink parent: $phys"
  [ -d "$phys" ] || web_runtime_die "resolved parent is not a directory: $phys"
  printf '%s' "$phys"
}

web_runtime_stamp_name() {
  printf '%s' ".greppy-web-runtime-dist"
}

web_runtime_known_members() {
  printf '%s\n' \
    ".greppy-web-runtime-dist" \
    "README.txt" \
    "UNSIGNED" \
    "SHA256SUMS" \
    "sbom.json" \
    "provenance.json" \
    "LICENSE" \
    "SIGNING_RECEIPT" \
    "SIGNING_SKIPPED" \
    "SIGNING_STATUS" \
    "NOTARIZATION_RECEIPT" \
    "NOTARIZATION_SKIPPED" \
    "NOTARIZED_UNSIGNED" \
    "bin/web-runtime-supervisor" \
    "bin/web-controller-worker" \
    "bin/web-content-worker" \
    "previous/web-runtime-supervisor" \
    "previous/web-controller-worker" \
    "previous/web-content-worker"
}

web_runtime_member_dirs() {
  printf '%s\n' "bin" "previous"
}

# Refuse any symlink on dest, intermediate member directories, or known files
# before a single read/copy/remove. Broken symlinks are also refused (-L is true
# even when -e is false).
web_runtime_preflight_tree() {
  pf_dest=$1
  web_runtime_check_owned_real_dir "$pf_dest"
  for dir in $(web_runtime_member_dirs); do
    pf_path="$pf_dest/$dir"
    if [ -L "$pf_path" ]; then
      web_runtime_die "refusing symlink directory: $pf_path"
    fi
    if [ -e "$pf_path" ]; then
      web_runtime_check_owned_real_dir "$pf_path"
    fi
  done
  for member in $(web_runtime_known_members); do
    pf_path="$pf_dest/$member"
    if [ -L "$pf_path" ]; then
      web_runtime_die "refusing symlink member: $pf_path"
    fi
    if [ -e "$pf_path" ]; then
      web_runtime_check_owned_regular_file "$pf_path"
    fi
  done
}

web_runtime_is_owned_dist() {
  iod_dest=$1
  [ -L "$iod_dest" ] && return 1
  [ -d "$iod_dest" ] || return 1
  [ -L "$iod_dest/bin" ] && return 1
  [ -d "$iod_dest/bin" ] || return 1
  if [ -L "$iod_dest/previous" ]; then
    return 1
  fi
  if [ -e "$iod_dest/previous" ]; then
    [ -d "$iod_dest/previous" ] || return 1
  fi
  stamp="$iod_dest/$(web_runtime_stamp_name)"
  [ -L "$stamp" ] && return 1
  [ -f "$stamp" ] || return 1
  grep -q 'greppy.web-runtime.package.v1' "$stamp" || return 1
  [ -L "$iod_dest/provenance.json" ] && return 1
  [ -f "$iod_dest/provenance.json" ] || return 1
  grep -q 'greppy.web-runtime.package.v1' "$iod_dest/provenance.json" || return 1
  for bin in web-runtime-supervisor web-controller-worker web-content-worker; do
    [ -L "$iod_dest/bin/$bin" ] && return 1
    [ -f "$iod_dest/bin/$bin" ] || return 1
  done
  for name in $(ls -A "$iod_dest"); do
    case "$name" in
      .greppy-web-runtime-dist | README.txt | UNSIGNED | SHA256SUMS | sbom.json | provenance.json | LICENSE | SIGNING_RECEIPT | SIGNING_SKIPPED | SIGNING_STATUS | NOTARIZATION_RECEIPT | NOTARIZATION_SKIPPED | NOTARIZED_UNSIGNED | bin | previous) ;;
      *) return 1 ;;
    esac
  done
  for name in $(ls -A "$iod_dest/bin"); do
    case "$name" in
      web-runtime-supervisor | web-controller-worker | web-content-worker) ;;
      *) return 1 ;;
    esac
  done
  return 0
}

web_runtime_validate_dest_shape() {
  grd_dest=$(web_runtime_strip_slash "$1")
  [ -n "$grd_dest" ] || web_runtime_die "empty dest"
  web_runtime_require_absolute "$grd_dest"
  web_runtime_refuse_dot_components "$grd_dest"
  web_runtime_refuse_sensitive "$grd_dest"
  base=$(basename "$grd_dest")
  [ -n "$base" ] && [ "$base" != / ] && [ "$base" != "." ] && [ "$base" != ".." ] ||
    web_runtime_die "refusing dest with empty basename: $grd_dest"
  web_runtime_basename_allowed "$base" ||
    web_runtime_die "dest basename is not a web-runtime dist name: $base"
  grd_parent=$(dirname "$grd_dest")
  phys_parent=$(web_runtime_physical_parent "$grd_parent")
  web_runtime_parent_allowed "$phys_parent" ||
    web_runtime_die "dest parent is not a web-runtime staging directory: $phys_parent"
  web_runtime_parent_usable "$phys_parent" ||
    web_runtime_die "refusing non-owned dest parent: $phys_parent"
  grd_dest="$phys_parent/$base"
  web_runtime_refuse_sensitive "$grd_dest"
  printf '%s' "$grd_dest"
}

# Validate dest as a commit target. Does not delete or overwrite dest.
web_runtime_require_package_dest() {
  raw=${1:-}
  repo=$(web_runtime_repo_root)
  if [ -z "$raw" ]; then
    raw="$repo/target/web-runtime-dist"
  fi
  grd_dest=$(web_runtime_validate_dest_shape "$raw")
  if [ -L "$grd_dest" ]; then
    web_runtime_die "refusing symlink dest: $grd_dest"
  fi
  if [ -e "$grd_dest" ]; then
    web_runtime_check_owned_real_dir "$grd_dest"
    web_runtime_preflight_tree "$grd_dest"
    if web_runtime_is_owned_dist "$grd_dest"; then
      printf '%s' "$grd_dest"
      return 0
    fi
    leftover=$(ls -A "$grd_dest" 2>/dev/null || true)
    if [ -n "$leftover" ]; then
      web_runtime_die "refusing to overwrite non-package directory: $grd_dest"
    fi
  fi
  printf '%s' "$grd_dest"
}

web_runtime_require_uninstall_dest() {
  raw=${1:-}
  [ -n "$raw" ] || web_runtime_die "uninstall dest is required"
  grd_dest=$(web_runtime_validate_dest_shape "$raw")
  [ -e "$grd_dest" ] || web_runtime_die "nothing to uninstall: $grd_dest"
  web_runtime_check_owned_real_dir "$grd_dest"
  web_runtime_preflight_tree "$grd_dest"
  web_runtime_is_owned_dist "$grd_dest" ||
    web_runtime_die "refusing to remove directory that is not a web-runtime dist: $grd_dest"
  printf '%s' "$grd_dest"
}

web_runtime_require_existing_dist() {
  raw=${1:-}
  label=${2:-dist}
  [ -n "$raw" ] || web_runtime_die "$label dest is required"
  grd_dest=$(web_runtime_validate_dest_shape "$raw")
  [ -e "$grd_dest" ] || web_runtime_die "missing $label dest: $grd_dest"
  web_runtime_check_owned_real_dir "$grd_dest"
  web_runtime_preflight_tree "$grd_dest"
  web_runtime_is_owned_dist "$grd_dest" ||
    web_runtime_die "$label is not a stamped web-runtime dist: $grd_dest"
  printf '%s' "$grd_dest"
}

web_runtime_remove_owned_dist_files() {
  rm_dest=$1
  web_runtime_preflight_tree "$rm_dest"
  web_runtime_is_owned_dist "$rm_dest" ||
    web_runtime_die "not a web-runtime dist: $rm_dest"
  for member in $(web_runtime_known_members); do
    rm_path="$rm_dest/$member"
    if [ -e "$rm_path" ] || [ -L "$rm_path" ]; then
      web_runtime_check_owned_regular_file "$rm_path"
      rm -f "$rm_path"
    fi
  done
  for dir in $(web_runtime_member_dirs); do
    rm_path="$rm_dest/$dir"
    if [ -L "$rm_path" ]; then
      web_runtime_die "refusing symlink directory: $rm_path"
    fi
    if [ -d "$rm_path" ]; then
      leftover=$(ls -A "$rm_path" 2>/dev/null || true)
      if [ -n "$leftover" ]; then
        web_runtime_die "refusing to remove dest with unexpected $dir members: $leftover"
      fi
      rmdir "$rm_path"
    fi
  done
}

web_runtime_copy_regular_file() {
  copy_src=$1
  copy_dest=$2
  if [ -L "$copy_src" ]; then
    web_runtime_die "refusing symlink source: $copy_src"
  fi
  [ -e "$copy_src" ] || web_runtime_die "missing source file: $copy_src"
  [ -f "$copy_src" ] || web_runtime_die "source is not a file: $copy_src"
  owner=$(web_runtime_owner_uid "$copy_src")
  me=$(web_runtime_uid)
  [ "$owner" = "$me" ] || web_runtime_die "refusing non-owned source $copy_src"
  if [ -L "$copy_dest" ]; then
    web_runtime_die "refusing symlink dest file: $copy_dest"
  fi
  copy_parent=$(dirname "$copy_dest")
  if [ -L "$copy_parent" ]; then
    web_runtime_die "refusing symlink dest parent: $copy_parent"
  fi
  if [ -e "$copy_parent" ]; then
    web_runtime_check_owned_real_dir "$copy_parent"
  else
    mkdir "$copy_parent" || web_runtime_die "failed to create dest parent $copy_parent"
    web_runtime_check_owned_real_dir "$copy_parent"
  fi
  copy_tmp="$copy_dest.greppy-tmp"
  if [ -L "$copy_tmp" ]; then
    web_runtime_die "refusing symlink temp: $copy_tmp"
  fi
  cp "$copy_src" "$copy_tmp"
  mv "$copy_tmp" "$copy_dest"
}

web_runtime_begin_staging() {
  bs_dest=$1
  bs_parent=$(dirname "$bs_dest")
  web_runtime_check_owned_real_dir "$bs_parent"
  bs_staging="$bs_parent/greppy-web-dist-stage-$$"
  n=0
  while [ -e "$bs_staging" ] || [ -L "$bs_staging" ]; do
    n=$((n + 1))
    bs_staging="$bs_parent/greppy-web-dist-stage-$$-$n"
  done
  mkdir "$bs_staging" || web_runtime_die "failed to create exclusive staging dir $bs_staging"
  mkdir "$bs_staging/bin" || web_runtime_die "failed to create staging bin $bs_staging/bin"
  web_runtime_check_owned_real_dir "$bs_staging"
  web_runtime_check_owned_real_dir "$bs_staging/bin"
  printf '%s' "$bs_staging"
}

web_runtime_discard_staging() {
  ds_staging=$1
  [ -n "$ds_staging" ] || return 0
  [ -e "$ds_staging" ] || [ -L "$ds_staging" ] || return 0
  base=$(basename "$ds_staging")
  case "$base" in
    greppy-web-dist-stage-*) ;;
    *) web_runtime_die "refusing to discard non-staging dest: $ds_staging" ;;
  esac
  if [ -L "$ds_staging" ]; then
    web_runtime_die "refusing symlink staging: $ds_staging"
  fi
  web_runtime_preflight_tree "$ds_staging"
  me=$(web_runtime_uid)
  for member in $(web_runtime_known_members); do
    ds_path="$ds_staging/$member"
    if [ -L "$ds_path" ]; then
      web_runtime_die "refusing symlink member: $ds_path"
    fi
    if [ -f "$ds_path" ]; then
      owner=$(web_runtime_owner_uid "$ds_path")
      [ "$owner" = "$me" ] || web_runtime_die "refusing non-owned member $ds_path"
      rm -f "$ds_path"
    fi
  done
  for dir in $(web_runtime_member_dirs); do
    ds_path="$ds_staging/$dir"
    if [ -L "$ds_path" ]; then
      web_runtime_die "refusing symlink directory: $ds_path"
    fi
    if [ -d "$ds_path" ]; then
      leftover=$(ls -A "$ds_path" 2>/dev/null || true)
      if [ -n "$leftover" ]; then
        web_runtime_die "refusing to discard staging with unexpected $dir members: $leftover"
      fi
      rmdir "$ds_path"
    fi
  done
  leftover=$(ls -A "$ds_staging" 2>/dev/null || true)
  if [ -n "$leftover" ]; then
    web_runtime_die "refusing to discard staging with unexpected members: $leftover"
  fi
  rmdir "$ds_staging"
}

web_runtime_commit_staging() {
  cs_staging=$1
  cs_dest=$2
  web_runtime_preflight_tree "$cs_staging"
  web_runtime_is_owned_dist "$cs_staging" ||
    web_runtime_die "staging is not a complete web-runtime dist: $cs_staging"
  cs_parent=$(dirname "$cs_dest")
  web_runtime_check_owned_real_dir "$cs_parent"
  if [ -L "$cs_dest" ]; then
    web_runtime_die "refusing symlink dest: $cs_dest"
  fi
  if [ ! -e "$cs_dest" ]; then
    mv "$cs_staging" "$cs_dest"
    return 0
  fi
  web_runtime_preflight_tree "$cs_dest"
  leftover=$(ls -A "$cs_dest" 2>/dev/null || true)
  if [ -z "$leftover" ]; then
    rmdir "$cs_dest"
    mv "$cs_staging" "$cs_dest"
    return 0
  fi
  web_runtime_is_owned_dist "$cs_dest" ||
    web_runtime_die "refusing to replace non-package directory: $cs_dest"
  cs_backup="$cs_parent/greppy-web-dist-prev-$$"
  n=0
  while [ -e "$cs_backup" ] || [ -L "$cs_backup" ]; do
    n=$((n + 1))
    cs_backup="$cs_parent/greppy-web-dist-prev-$$-$n"
  done
  mv "$cs_dest" "$cs_backup" || web_runtime_die "failed to park existing dist at $cs_backup"
  if ! mv "$cs_staging" "$cs_dest"; then
    mv "$cs_backup" "$cs_dest" ||
      web_runtime_die "failed to restore $cs_dest after staging commit failure"
    web_runtime_die "failed to commit staging to $cs_dest"
  fi
  web_runtime_uninstall_owned_dist "$cs_backup"
}

web_runtime_uninstall_owned_dist() {
  un_dest=$1
  web_runtime_preflight_tree "$un_dest"
  web_runtime_remove_owned_dist_files "$un_dest"
  leftover=$(ls -A "$un_dest" 2>/dev/null || true)
  if [ -n "$leftover" ]; then
    web_runtime_die "refusing to remove dest with unexpected members: $leftover"
  fi
  rmdir "$un_dest"
  tarball="$(dirname "$un_dest")/$(basename "$un_dest").tar.gz"
  if [ -e "$tarball" ] || [ -L "$tarball" ]; then
    [ -L "$tarball" ] && web_runtime_die "refusing symlink archive: $tarball"
    [ -f "$tarball" ] || web_runtime_die "archive is not a file: $tarball"
    owner=$(web_runtime_owner_uid "$tarball")
    me=$(web_runtime_uid)
    [ "$owner" = "$me" ] || web_runtime_die "refusing non-owned archive $tarball"
    rm -f "$tarball"
  fi
}


web_runtime_verify_sha256sums() {
  vs_root=$1
  if [ -L "$vs_root" ] || [ -L "$vs_root/SHA256SUMS" ] || [ -L "$vs_root/bin" ]; then
    web_runtime_die "refusing symlink SHA256SUMS root: $vs_root"
  fi
  if [ ! -f "$vs_root/SHA256SUMS" ]; then
    web_runtime_die "missing SHA256SUMS"
  fi
  if [ ! -d "$vs_root/bin" ]; then
    web_runtime_die "missing bin directory for SHA256SUMS verify"
  fi
  vs_listed=$(awk '{print $NF}' "$vs_root/SHA256SUMS" | LC_ALL=C sort)
  vs_expected=$(printf '%s\n' web-content-worker web-controller-worker web-runtime-supervisor)
  if [ "$vs_listed" != "$vs_expected" ]; then
    web_runtime_die "SHA256SUMS must list exactly the three runtime images"
  fi
  (
    CDPATH= cd -- "$vs_root/bin" || exit 1
    if command -v shasum >/dev/null; then
      shasum -a 256 -c ../SHA256SUMS
    else
      sha256sum -c ../SHA256SUMS
    fi
  ) || web_runtime_die "SHA256SUMS verification failed for $vs_root"
}

web_runtime_write_stamp() {
  ws_dest=$1
  web_runtime_check_owned_real_dir "$ws_dest"
  stamp="$ws_dest/$(web_runtime_stamp_name)"
  if [ -L "$stamp" ]; then
    web_runtime_die "refusing symlink stamp: $stamp"
  fi
  printf '%s\n' "greppy.web-runtime.package.v1" >"$stamp"
}
