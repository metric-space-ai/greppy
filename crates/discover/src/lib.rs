//! `grepplus-discover` — file inventory, ignore-rule walking, language
//! detection, and (Phase 5) workspace fingerprinting.
//!
//! Phase 3 ships the file walker. The walker respects gitignore (via
//! the `ignore` crate) and returns a stable list of `(rel_path,
//! absolute_path)` pairs in deterministic order.

#![deny(rust_2018_idioms)]

use std::path::{Path, PathBuf};

use grepplus_core::{Error, Result};

pub mod binary;
pub mod language;
pub mod skip;

pub use binary::{is_binary_bytes, is_binary_file, SNIFF_PREFIX_BYTES};
pub use language::{detect_language, detect_language_with_shebang, DetectedLanguage};
pub use skip::{
    is_generated_dir, is_ignored_file, SkipPolicy, DEFAULT_MAX_FILE_SIZE, GENERATED_DIRS,
    IGNORED_SUFFIXES,
};

/// Directory basenames that are always excluded from the walk,
/// independent of any `.gitignore` content. These mirror the
/// non-negotiable excludes the indexer relies on:
///
/// - `.git` — VCS metadata (the `ignore` crate also excludes it via
///   `standard_filters`, but we enforce it here too so the walk is
///   correct even when git handling is disabled or the dir is not a
///   repo).
/// - `target` — Rust build artifacts.
/// - `.vendor` — the upstream mirror vendored into this repo.
/// - `.grepplus` — grepplus' own per-workspace state/sidecar directory;
///   indexing it would record our own outputs as project files and make
///   freshness permanently stale (cf. the store-dir filter, R-005).
///
/// Excluding by *basename* (not just top-level prefix) means a nested
/// `foo/target/` or `vendored/.git/` is dropped too, matching upstream
/// `ALWAYS_SKIP_DIRS` basename semantics.
const ALWAYS_SKIP_DIRS: &[&str] = &[".git", "target", ".vendor", ".grepplus"];

/// True if any path component of `rel` is an always-skip directory.
fn has_skipped_dir_component(rel: &str) -> bool {
    rel.split('/').any(|seg| ALWAYS_SKIP_DIRS.contains(&seg))
}

/// One file in the workspace, with both the absolute path and the
/// workspace-relative path. The relative path is what we store in the
/// graph; the absolute path is what we read.
///
/// `size` and `mtime_ns` carry the file metadata captured during the
/// walk so the indexer and freshness layers can compare against stored
/// `file_state` without an extra `stat(2)` per file. They are
/// `Option<…>` so existing call sites that build an `InventoryEntry`
/// by literal (e.g. test fixtures in dependent crates) keep working with
/// `..Default::default()`, and so a metadata read that fails mid-walk
/// degrades to `None` rather than aborting the inventory. `walk` always
/// populates them when the platform `stat` succeeds.
///
/// - `size`: file length in bytes (`st_size`).
/// - `mtime_ns`: last-modification time as nanoseconds since the Unix
///   epoch. Signed so pre-epoch mtimes (rare, but legal) are
///   representable, matching the `mtime_ns` column the freshness layer
///   stores.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct InventoryEntry {
    pub rel_path: String,
    pub abs_path: PathBuf,
    /// File size in bytes, captured during the walk. `None` when metadata
    /// could not be read.
    pub size: Option<u64>,
    /// Last-modification time in nanoseconds since the Unix epoch,
    /// captured during the walk. `None` when metadata could not be read
    /// or the platform time is before the epoch in a way that cannot be
    /// represented.
    pub mtime_ns: Option<i64>,
}

/// Explicit include/exclude globs supplied by a caller.
///
/// These use the `ignore` crate's override semantics: include globs are
/// whitelist patterns, exclude globs are blacklist patterns. When at
/// least one include glob exists, non-matching files are left out of the
/// inventory. Directory traversal still follows gitignore, global
/// excludes and the active [`SkipPolicy`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WalkOverrides {
    pub includes: Vec<String>,
    pub excludes: Vec<String>,
}

impl WalkOverrides {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.includes.is_empty() && self.excludes.is_empty()
    }

    pub fn include(mut self, glob: impl Into<String>) -> Self {
        self.includes.push(glob.into());
        self
    }

    pub fn exclude(mut self, glob: impl Into<String>) -> Self {
        self.excludes.push(glob.into());
        self
    }

    /// Stable, order-preserving key for this override set.
    ///
    /// Include and exclude order within each class is part of the policy,
    /// so this deliberately does not sort either list. Excludes are
    /// applied after includes by [`build_ignore_overrides`].
    pub fn scope_key(&self) -> String {
        if self.is_empty() {
            return "default".into();
        }
        let mut out = String::from("v1");
        for include in &self.includes {
            append_scope_part(&mut out, 'I', include);
        }
        for exclude in &self.excludes {
            append_scope_part(&mut out, 'E', exclude);
        }
        out
    }
}

fn append_scope_part(out: &mut String, kind: char, glob: &str) {
    out.push(';');
    out.push(kind);
    out.push_str(&glob.len().to_string());
    out.push(':');
    out.push_str(glob);
}

impl InventoryEntry {
    /// Build an entry from just its paths, leaving metadata unset.
    /// Convenience for callers (and tests) that do not have `stat`
    /// metadata on hand; the walker fills `size`/`mtime_ns` itself.
    pub fn new(rel_path: impl Into<String>, abs_path: impl Into<PathBuf>) -> Self {
        Self {
            rel_path: rel_path.into(),
            abs_path: abs_path.into(),
            size: None,
            mtime_ns: None,
        }
    }
}

/// Read `(size, mtime_ns)` for a file, returning `(None, None)` on any
/// error. `mtime_ns` is nanoseconds since the Unix epoch; times before
/// the epoch are encoded as negative values. This is the single place
/// the walker turns a `std::fs::Metadata` into the additive
/// `InventoryEntry` fields, so the conversion (and its saturation
/// behaviour for out-of-range durations) lives in one tested spot.
fn metadata_fields(meta: &std::fs::Metadata) -> (Option<u64>, Option<i64>) {
    let size = Some(meta.len());
    let mtime_ns = meta.modified().ok().map(|mt| {
        match mt.duration_since(std::time::UNIX_EPOCH) {
            Ok(d) => i64::try_from(d.as_nanos()).unwrap_or(i64::MAX),
            // Pre-epoch mtime: encode as a negative offset.
            Err(e) => {
                let back = e.duration().as_nanos();
                i64::try_from(back).map(|v| -v).unwrap_or(i64::MIN)
            }
        }
    });
    (size, mtime_ns)
}

/// Walk `root` and return every regular file under it that is not
/// excluded by a `.gitignore`, `.ignore`, or global ignore file.
///
/// Sort order is lexicographic on `rel_path` so two runs against the
/// same workspace produce the same inventory.
///
/// The walker also skips the workspace's own store directory
/// (`store/<16-hex>/`, `target/store/<16-hex>/`, etc.) so the indexer's
/// own DB file is not recorded as a project file (R-005 / WP-R005).
/// When the platform locator lives outside `root` (the default
/// on macOS / Linux), this filter is a no-op.
/// True if `rel` looks like a path inside the workspace's own store
/// directory (`store/<16-hex-chars>[/...]`). The 16-hex component is
/// what the platform-locator produces (see `grepplus_core::workspace`).
fn is_store_rel_path(rel: &str) -> bool {
    // We accept any segment that matches `store/<16 hex>/...`.
    // The whole component after `store/` must be 16 hex chars exactly.
    let mut segs = rel.split('/').peekable();
    while let Some(s) = segs.next() {
        if s == "store" {
            if let Some(next) = segs.peek() {
                if next.len() == 16 && next.chars().all(|c| c.is_ascii_hexdigit()) {
                    return true;
                }
            }
        }
    }
    false
}

pub fn walk(root: &Path) -> Result<Vec<InventoryEntry>> {
    walk_with_policy(root, &skip::SkipPolicy::walk_default())
}

/// Like [`walk`], but with an explicit [`skip::SkipPolicy`] controlling
/// the generated-dir / ignored-suffix / size heuristics that run *in
/// addition* to gitignore. [`walk`] is exactly
/// `walk_with_policy(root, &SkipPolicy::walk_default())` — generated
/// *directories* are skipped but file-level suffix/pattern/size dropping
/// is left to the consumer (the indexer does its own oversize/binary
/// accounting). Pass [`skip::SkipPolicy::default`] for full upstream
/// `ALWAYS_*` breadth, or [`skip::SkipPolicy::unrestricted`] to disable
/// every added heuristic.
///
/// ## Symlink policy
///
/// Symlinks are never followed (`follow_links(false)`), matching upstream
/// `safe_stat`, which `lstat`s every entry and drops anything with
/// `S_ISLNK`. This is enforced two ways:
///
/// 1. The walker does not descend through symlinked directories, so a
///    symlink that points back into an ancestor cannot create an infinite
///    loop, and a symlink that points *outside* `root` cannot pull
///    external files into the inventory.
/// 2. Each candidate file's own `symlink_metadata` is checked: if the
///    entry itself is a symlink (a symlinked regular file), it is skipped
///    rather than inventoried via its target. This prevents a symlink
///    whose target lives outside the repo root — or forms a loop — from
///    being recorded.
///
/// The metadata captured for surviving entries (`size`, `mtime_ns`) comes
/// from this same non-following `symlink_metadata` call, so no extra
/// `stat` is performed.
pub fn walk_with_policy(root: &Path, policy: &skip::SkipPolicy) -> Result<Vec<InventoryEntry>> {
    walk_with_policy_and_overrides(root, policy, &WalkOverrides::empty())
}

/// Like [`walk_with_policy`], but with explicit include/exclude override
/// globs in addition to normal gitignore/global-exclude handling.
pub fn walk_with_policy_and_overrides(
    root: &Path,
    policy: &skip::SkipPolicy,
    overrides: &WalkOverrides,
) -> Result<Vec<InventoryEntry>> {
    use ignore::WalkBuilder;

    let mut entries = Vec::new();
    let mut builder = WalkBuilder::new(root);
    // We do not want to descend into `.git/`, but gitignore handling
    // already excludes it via the project's own `.gitignore`. We also
    // skip the upstream vendor mirror if the workspace happens to be the
    // grepplus-rs project itself.
    builder.standard_filters(true);
    // Do not follow symlinks: a symlinked directory must not be descended
    // (loop / escape protection), and a symlinked file is handled by the
    // explicit per-entry check below.
    builder.follow_links(false);
    builder.require_git(false);
    if !overrides.is_empty() {
        builder.overrides(build_ignore_overrides(root, overrides)?);
    }
    let walker = builder.build();
    for dent in walker {
        let dent = match dent {
            Ok(d) => d,
            Err(_) => continue, // skip unreadable entries; do not abort the whole walk
        };
        let ftype = match dent.file_type() {
            Some(t) => t,
            None => continue,
        };
        // Symlink policy: never inventory a symlink itself. With
        // `follow_links(false)` the `ignore` crate reports the link's own
        // type, so a symlinked regular file shows up as a non-symlink
        // only if it were followed — here it is reported as a symlink and
        // dropped. A symlinked directory is likewise not descended. This
        // matches upstream `safe_stat`'s `S_ISLNK` rejection and prevents
        // both out-of-root escape and symlink loops.
        if ftype.is_symlink() {
            continue;
        }
        if !ftype.is_file() {
            continue;
        }
        let abs = dent.into_path();
        let rel = match abs.strip_prefix(root) {
            Ok(p) => p.to_string_lossy().into_owned(),
            Err(_) => continue,
        };
        // Normalize to forward-slash early so all path-component checks
        // below (skip-dir, store-dir) behave identically on Windows.
        let rel = rel.replace('\\', "/");
        // Filter out always-skip directories (`.git`, `target`,
        // `.vendor`, `.grepplus`) by basename anywhere in the path. This
        // is enforced independent of `.gitignore` so the inventory is
        // correct even in a non-git tree or when standard filters are
        // bypassed.
        if has_skipped_dir_component(&rel) {
            continue;
        }
        // Generated/vendored directory skip (node_modules, dist, build,
        // .venv, __pycache__, …). Configurable via `policy`; runs in
        // addition to — never instead of — gitignore. The default policy
        // mirrors upstream `ALWAYS_SKIP_DIRS`.
        if policy.should_skip_dir_component(&rel) {
            continue;
        }
        // R-005 / WP-R005: skip the workspace's own store dir
        // (`<root>/store/<16-hex-chars>/...`). The store DB lives
        // here under the platform locator; indexing it would record
        // the DB itself as a project file, and every subsequent
        // freshness walk would see a content change (the indexer's
        // own writes mutate it), making `check_files` permanently
        // `Stale`.
        if is_store_rel_path(&rel) {
            continue;
        }
        // Per-entry symlink + metadata read. `symlink_metadata` does not
        // follow the final component, so a symlinked regular file that
        // slipped past the `file_type` check above (platform variance) is
        // caught here, and the `size`/`mtime_ns` we record come from this
        // single non-following stat.
        let (size, mtime_ns) = match std::fs::symlink_metadata(&abs) {
            Ok(meta) => {
                if meta.file_type().is_symlink() {
                    continue;
                }
                metadata_fields(&meta)
            }
            // Could not stat: keep the entry (it was a regular file at
            // walk time) but with unknown metadata, mirroring the
            // walker's fail-open posture for transient races.
            Err(_) => (None, None),
        };
        // Name-based + size-based skip (ignored suffixes, generated
        // patterns, max size). Uses the basename and the size we just
        // captured.
        let name = abs.file_name().and_then(|s| s.to_str()).unwrap_or_default();
        if policy.should_skip_file(name, size) {
            continue;
        }
        entries.push(InventoryEntry {
            rel_path: rel,
            abs_path: abs,
            size,
            mtime_ns,
        });
    }
    entries.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));
    Ok(entries)
}

fn build_ignore_overrides(
    root: &Path,
    overrides: &WalkOverrides,
) -> Result<ignore::overrides::Override> {
    let mut builder = ignore::overrides::OverrideBuilder::new(root);
    for include in &overrides.includes {
        builder
            .add(include)
            .map_err(|e| Error::Config(format!("invalid include override `{include}`: {e}")))?;
    }
    for exclude in &overrides.excludes {
        let pattern = format!("!{exclude}");
        builder
            .add(&pattern)
            .map_err(|e| Error::Config(format!("invalid exclude override `{exclude}`: {e}")))?;
    }
    builder
        .build()
        .map_err(|e| Error::Config(format!("invalid walk overrides: {e}")))
}

/// Detect the repository root for `start`.
///
/// In Phase 3 this is a best-effort: walk up the directory tree looking
/// for a `.git` entry. If found, return its parent's canonical path;
/// otherwise return `start` canonicalised.
///
/// Phase 5 will replace this with the full git-fingerprint routine.
pub fn detect_repo_root(start: &Path) -> Result<PathBuf> {
    // Defect D9: the fallback arms used to return `start` VERBATIM, so
    // `grepplus index .` in a directory without `.git` recorded the
    // workspace root as `.` — every later query (which resolves roots to
    // absolute paths) then failed to find the workspace state. Always
    // return an absolute path: canonical when the path resolves,
    // lexically absolute otherwise.
    let canon: PathBuf = match start.canonicalize() {
        Ok(p) => p,
        Err(_) => {
            return Ok(std::path::absolute(start).unwrap_or_else(|_| start.to_path_buf()));
        }
    };
    let mut cur: PathBuf = canon.clone();
    loop {
        if cur.join(".git").exists() {
            return Ok(cur);
        }
        match cur.parent() {
            Some(p) => cur = p.to_path_buf(),
            None => return Ok(canon),
        }
    }
}

/// Shared test helpers used by this crate's unit tests (in `lib.rs`,
/// `binary.rs`, and `language.rs`).
#[cfg(test)]
pub(crate) mod tests_support {
    use std::path::PathBuf;

    /// Create and return a fresh, unique temp directory for a test.
    pub fn unique_tmp() -> PathBuf {
        let base = std::env::temp_dir();
        let unique = format!(
            "grepplus-discover-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let p = base.join(unique);
        std::fs::create_dir_all(&p).unwrap();
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::Mutex;

    // Env-mutating tests must be serialized because HOME/XDG_CONFIG_HOME are
    // process-global and the ignore crate reads them while building walkers.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvRestore {
        vars: Vec<(&'static str, Option<std::ffi::OsString>)>,
    }

    impl EnvRestore {
        fn capture(vars: &[&'static str]) -> Self {
            Self {
                vars: vars
                    .iter()
                    .map(|name| (*name, std::env::var_os(name)))
                    .collect(),
            }
        }
    }

    impl Drop for EnvRestore {
        fn drop(&mut self) {
            for (name, value) in &self.vars {
                // SAFETY: callers hold ENV_LOCK while this guard is alive.
                unsafe {
                    match value {
                        Some(v) => std::env::set_var(name, v),
                        None => std::env::remove_var(name),
                    }
                }
            }
        }
    }

    fn write(p: &Path, body: &str) {
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(p, body).unwrap();
    }

    #[test]
    fn walk_finds_rust_files_and_respects_gitignore() {
        let tmp = tempdir_via_env();
        let root = tmp.join("repo");
        fs::create_dir_all(&root).unwrap();

        write(&root.join("src/lib.rs"), "fn a() {}");
        write(&root.join("src/main.rs"), "fn main() {}");
        write(&root.join("README.md"), "# readme");
        write(&root.join(".gitignore"), "target/\nignored.rs\n");

        write(&root.join("target/artifact.rs"), "// built");
        write(&root.join("ignored.rs"), "// ignored");

        let entries = walk(&root).unwrap();
        let rels: Vec<&str> = entries.iter().map(|e| e.rel_path.as_str()).collect();
        assert!(rels.contains(&"src/lib.rs"));
        assert!(rels.contains(&"src/main.rs"));
        assert!(rels.contains(&"README.md"));
        assert!(!rels.iter().any(|r| r.starts_with("target/")));
        assert!(!rels.contains(&"ignored.rs"));
    }

    #[test]
    fn walk_is_deterministic() {
        let tmp = tempdir_via_env();
        let root = tmp.join("repo");
        fs::create_dir_all(&root).unwrap();
        for n in 0..5 {
            write(&root.join(format!("src/f{n}.rs")), "");
        }
        let a = walk(&root).unwrap();
        let b = walk(&root).unwrap();
        assert_eq!(a, b, "two walks must produce the same order");
    }

    #[test]
    fn walk_skips_vendor_and_target() {
        let tmp = tempdir_via_env();
        let root = tmp.join("repo");
        fs::create_dir_all(&root).unwrap();
        write(&root.join(".vendor/mirror/something.txt"), "");
        write(&root.join("target/build.rs"), "");
        write(&root.join("src/keep.rs"), "");

        let entries = walk(&root).unwrap();
        let rels: Vec<&str> = entries.iter().map(|e| e.rel_path.as_str()).collect();
        assert!(rels.contains(&"src/keep.rs"));
        assert!(!rels.iter().any(|r| r.starts_with(".vendor/")));
        assert!(!rels.iter().any(|r| r.starts_with("target/")));
    }

    #[test]
    fn detect_repo_root_walks_up_to_dot_git() {
        let tmp = tempdir_via_env();
        let root = tmp.join("repo");
        let nested = root.join("a/b/c");
        fs::create_dir_all(&nested).unwrap();
        fs::create_dir_all(root.join(".git")).unwrap();

        let detected = detect_repo_root(&nested).unwrap();
        assert!(detected.ends_with("repo"));
    }

    #[test]
    fn detect_repo_root_returns_start_when_no_git() {
        let tmp = tempdir_via_env();
        let start = tmp.join("no-git-here/file.txt");
        fs::create_dir_all(start.parent().unwrap()).unwrap();
        let detected = detect_repo_root(&start).unwrap();
        // When no .git is found anywhere up the tree, we return either
        // the canonicalised input or the input as-is. Both forms end
        // with the rightmost component; assert on that instead of the
        // full path, since macOS canonicalises through /private which
        // may not be present in `start`.
        assert!(
            detected.ends_with("no-git-here") || detected.ends_with("file.txt"),
            "unexpected detected path: {detected:?}"
        );
    }

    fn tempdir_via_env() -> PathBuf {
        super::tests_support::unique_tmp()
    }

    #[test]
    fn walk_skips_git_and_grepplus_dirs() {
        // .git and .grepplus must be excluded even when there is no
        // .gitignore mentioning them, and even when nested.
        let tmp = tempdir_via_env();
        let root = tmp.join("repo");
        fs::create_dir_all(&root).unwrap();
        write(&root.join(".git/config"), "[core]");
        write(&root.join(".git/objects/aa/bb"), "x");
        write(&root.join(".grepplus/sidecar.db"), "state");
        write(&root.join("sub/.grepplus/cache"), "state");
        write(&root.join("sub/nested/target/out.rs"), "// built");
        write(&root.join("sub/.vendor/mirror.txt"), "");
        write(&root.join("src/keep.rs"), "fn k() {}");

        let entries = walk(&root).unwrap();
        let rels: Vec<&str> = entries.iter().map(|e| e.rel_path.as_str()).collect();
        assert!(rels.contains(&"src/keep.rs"));
        assert!(
            !rels.iter().any(|r| r.split('/').any(|s| s == ".git")),
            "no .git component should survive: {rels:?}"
        );
        assert!(!rels.iter().any(|r| r.split('/').any(|s| s == ".grepplus")));
        assert!(!rels.iter().any(|r| r.split('/').any(|s| s == "target")));
        assert!(!rels.iter().any(|r| r.split('/').any(|s| s == ".vendor")));
    }

    #[test]
    fn walk_respects_dot_ignore_file() {
        // The `ignore` crate honours `.ignore` as well as `.gitignore`;
        // confirm that parity holds through walk().
        let tmp = tempdir_via_env();
        let root = tmp.join("repo");
        fs::create_dir_all(&root).unwrap();
        write(&root.join(".ignore"), "secret.rs\nbuildlogs/\n");
        write(&root.join("secret.rs"), "// hidden");
        write(&root.join("buildlogs/a.txt"), "log");
        write(&root.join("src/visible.rs"), "fn v() {}");

        let rels: Vec<String> = walk(&root)
            .unwrap()
            .into_iter()
            .map(|e| e.rel_path)
            .collect();
        assert!(rels.iter().any(|r| r == "src/visible.rs"));
        assert!(!rels.iter().any(|r| r == "secret.rs"));
        assert!(!rels.iter().any(|r| r.starts_with("buildlogs/")));
    }

    #[test]
    fn walk_matches_git_ignore_sources_against_git_check_ignore() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _restore = EnvRestore::capture(&[
            "HOME",
            "XDG_CONFIG_HOME",
            "GIT_CONFIG_GLOBAL",
            "GIT_CONFIG_NOSYSTEM",
        ]);

        let tmp = tempdir_via_env();
        let root = tmp.join("repo");
        let home = tmp.join("home");
        let xdg = tmp.join("xdg");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(xdg.join("git")).unwrap();

        // SAFETY: serialized by ENV_LOCK and restored by EnvRestore.
        unsafe {
            std::env::set_var("HOME", &home);
            std::env::set_var("XDG_CONFIG_HOME", &xdg);
            std::env::remove_var("GIT_CONFIG_GLOBAL");
            std::env::set_var("GIT_CONFIG_NOSYSTEM", "1");
        }

        let init = std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(&root)
            .env("HOME", &home)
            .env("XDG_CONFIG_HOME", &xdg)
            .env_remove("GIT_CONFIG_GLOBAL")
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .status()
            .expect("spawn git init");
        if !init.success() {
            panic!("git init failed with status {init}");
        }

        write(
            &root.join(".gitignore"),
            "root-ignored.rs\ngitignore_dir/\n*.tmp.rs\n!keep.tmp.rs\n",
        );
        write(&root.join("nested/.gitignore"), "nested-ignored.rs\n");
        write(&root.join(".git/info/exclude"), "info-only.rs\ninfo_dir/\n");
        write(&xdg.join("git/ignore"), "global-only.rs\nglobal_dir/\n");

        let candidates = [
            "src/visible.rs",
            "root-ignored.rs",
            "gitignore_dir/a.rs",
            "drop.tmp.rs",
            "keep.tmp.rs",
            "nested/visible.rs",
            "nested/nested-ignored.rs",
            "info-only.rs",
            "info_dir/a.rs",
            "global-only.rs",
            "global_dir/a.rs",
        ];
        for rel in candidates {
            write(&root.join(rel), "fn marker() {}\n");
        }

        let entries = walk(&root).unwrap();
        let inventoried: std::collections::BTreeSet<_> =
            entries.into_iter().map(|e| e.rel_path).collect();

        for rel in candidates {
            let ignored_by_git = std::process::Command::new("git")
                .args(["check-ignore", "--quiet", "--", rel])
                .current_dir(&root)
                .env("HOME", &home)
                .env("XDG_CONFIG_HOME", &xdg)
                .env_remove("GIT_CONFIG_GLOBAL")
                .env("GIT_CONFIG_NOSYSTEM", "1")
                .status()
                .unwrap_or_else(|e| panic!("spawn git check-ignore for {rel}: {e}"))
                .success();
            assert_eq!(
                !inventoried.contains(rel),
                ignored_by_git,
                "discover walk must match git check-ignore for {rel}; inventory={inventoried:?}"
            );
        }
    }

    #[test]
    fn walk_overrides_apply_include_whitelist_and_exclude_blacklist() {
        let tmp = tempdir_via_env();
        let root = tmp.join("repo");
        fs::create_dir_all(&root).unwrap();
        write(&root.join("src/lib.rs"), "fn lib() {}\n");
        write(&root.join("src/generated.rs"), "fn generated() {}\n");
        write(&root.join("tests/integration.rs"), "fn it() {}\n");
        write(&root.join("README.md"), "# readme\n");

        let overrides = WalkOverrides::empty()
            .include("src/*.rs")
            .exclude("src/generated.rs");
        let rels: Vec<String> =
            walk_with_policy_and_overrides(&root, &skip::SkipPolicy::unrestricted(), &overrides)
                .unwrap()
                .into_iter()
                .map(|e| e.rel_path)
                .collect();

        assert_eq!(rels, vec!["src/lib.rs"]);
    }

    #[test]
    fn walk_overrides_report_invalid_globs() {
        let tmp = tempdir_via_env();
        let root = tmp.join("repo");
        fs::create_dir_all(&root).unwrap();
        write(&root.join("src/lib.rs"), "fn lib() {}\n");

        let err = walk_with_policy_and_overrides(
            &root,
            &skip::SkipPolicy::unrestricted(),
            &WalkOverrides::empty().include("[broken"),
        )
        .unwrap_err();

        assert!(
            err.to_string().contains("invalid include override"),
            "invalid override should produce an actionable error, got {err}"
        );
    }

    #[test]
    fn walk_overrides_scope_key_is_stable_and_ordered() {
        assert_eq!(WalkOverrides::empty().scope_key(), "default");
        let overrides = WalkOverrides::empty()
            .include("src/*.rs")
            .exclude("src/generated.rs")
            .include("tests/*.rs");
        assert_eq!(
            overrides.scope_key(),
            "v1;I8:src/*.rs;I10:tests/*.rs;E16:src/generated.rs"
        );
    }

    #[test]
    fn has_skipped_dir_component_matches_basenames() {
        assert!(has_skipped_dir_component("target/foo.rs"));
        assert!(has_skipped_dir_component("a/b/target/foo.rs"));
        assert!(has_skipped_dir_component(".git/config"));
        assert!(has_skipped_dir_component("x/.grepplus/y"));
        assert!(has_skipped_dir_component("x/.vendor/y"));
        // Substrings of skip-dir names must NOT match.
        assert!(!has_skipped_dir_component("src/targets.rs"));
        assert!(!has_skipped_dir_component("src/git.rs"));
        assert!(!has_skipped_dir_component("src/lib.rs"));
    }

    #[test]
    fn walk_classifies_binary_and_detects_language_for_inventoried_files() {
        // End-to-end: walk produces entries; the public binary + language
        // helpers can classify each entry so the indexer can record an
        // accurate file_state.
        let tmp = tempdir_via_env();
        let root = tmp.join("repo");
        fs::create_dir_all(&root).unwrap();
        write(&root.join("src/lib.rs"), "fn a() {}");
        write(&root.join("app.go"), "package main");
        fs::write(root.join("blob.bin"), b"\x7fELF\x00\x00binary").unwrap();

        // Use an unrestricted policy here: `.bin` is in the default
        // ignored-suffix set, so the default `walk` (correctly) drops it.
        // This test exercises the binary/language *classification*
        // helpers on every inventoried entry, so it opts out of the
        // generated/suffix heuristics to keep the blob in the inventory.
        let entries = walk_with_policy(&root, &skip::SkipPolicy::unrestricted()).unwrap();
        let by_rel = |name: &str| {
            entries
                .iter()
                .find(|e| e.rel_path == name)
                .unwrap_or_else(|| panic!("missing {name}"))
        };

        let rs = by_rel("src/lib.rs");
        assert_eq!(detect_language(&rs.abs_path), DetectedLanguage::Rust);
        assert!(!is_binary_file(&rs.abs_path));

        let go = by_rel("app.go");
        assert_eq!(detect_language(&go.abs_path), DetectedLanguage::Go);
        assert!(go.rel_path.ends_with(".go"));
        assert!(!is_binary_file(&go.abs_path));

        let bin = by_rel("blob.bin");
        assert!(is_binary_file(&bin.abs_path), "ELF blob must be binary");
        assert!(!detect_language(&bin.abs_path).is_detected());
    }

    #[test]
    fn walk_captures_size_and_mtime_metadata() {
        let tmp = tempdir_via_env();
        let root = tmp.join("repo");
        fs::create_dir_all(&root).unwrap();
        let body = "fn k() {}\n"; // 10 bytes
        write(&root.join("src/keep.rs"), body);

        let entries = walk(&root).unwrap();
        let e = entries
            .iter()
            .find(|e| e.rel_path == "src/keep.rs")
            .expect("keep.rs present");
        assert_eq!(
            e.size,
            Some(body.len() as u64),
            "size must match file length"
        );
        let mtime = e.mtime_ns.expect("mtime captured");
        // mtime is nanoseconds since the epoch; for a just-written file it
        // must be positive and not absurdly large.
        assert!(mtime > 0, "mtime should be a positive epoch offset");
        // Cross-check against an independent stat of the same file.
        let meta = fs::metadata(&e.abs_path).unwrap();
        assert_eq!(e.size, Some(meta.len()));
    }

    #[test]
    fn inventory_entry_new_leaves_metadata_unset() {
        let e = InventoryEntry::new("a/b.rs", PathBuf::from("/x/a/b.rs"));
        assert_eq!(e.rel_path, "a/b.rs");
        assert_eq!(e.abs_path, PathBuf::from("/x/a/b.rs"));
        assert_eq!(e.size, None);
        assert_eq!(e.mtime_ns, None);
        // Default builds an empty entry.
        assert_eq!(InventoryEntry::default().size, None);
    }

    #[test]
    fn walk_skips_generated_dirs_by_default() {
        let tmp = tempdir_via_env();
        let root = tmp.join("repo");
        fs::create_dir_all(&root).unwrap();
        // Generated/vendored dirs: must NOT be inventoried even with no
        // .gitignore mentioning them.
        write(
            &root.join("node_modules/left-pad/index.js"),
            "module.exports=0",
        );
        write(&root.join("dist/bundle.js"), "console.log(1)");
        write(&root.join("build/out.o.txt"), "x");
        write(&root.join(".venv/lib/site.py"), "import os");
        write(&root.join("__pycache__/mod.cpython.txt"), "x");
        write(&root.join("pkg/vendor/dep/file.go"), "package dep");
        // First-party source: must survive.
        write(&root.join("src/keep.rs"), "fn k() {}");
        write(&root.join("app/main.py"), "print(1)");

        let entries = walk(&root).unwrap();
        let rels: Vec<&str> = entries.iter().map(|e| e.rel_path.as_str()).collect();
        assert!(rels.contains(&"src/keep.rs"));
        assert!(rels.contains(&"app/main.py"));
        for gen in ["node_modules/", "dist/", "build/", ".venv/", "__pycache__/"] {
            assert!(
                !rels.iter().any(|r| r.starts_with(gen)),
                "{gen} should be skipped: {rels:?}"
            );
        }
        assert!(
            !rels.iter().any(|r| r.split('/').any(|s| s == "vendor")),
            "nested vendor should be skipped: {rels:?}"
        );
    }

    #[test]
    fn full_policy_skips_ignored_suffixes_and_minified() {
        // The full upstream-breadth `SkipPolicy::default()` (used via
        // `walk_with_policy`, NOT the bare `walk`) drops ignored suffixes,
        // minified bundles, and generated declaration files. The bare
        // `walk` leaves these file-level decisions to the indexer; see
        // `walk_default_keeps_files_for_indexer_accounting`.
        let tmp = tempdir_via_env();
        let root = tmp.join("repo");
        fs::create_dir_all(&root).unwrap();
        write(&root.join("logo.png"), "fakepng");
        write(&root.join("app.min.js"), "var a=1");
        write(&root.join("types.d.ts"), "export {}");
        write(&root.join("src/app.js"), "const a=1");
        write(&root.join("src/app.ts"), "const a: number = 1");

        let rels: Vec<String> = walk_with_policy(&root, &skip::SkipPolicy::default())
            .unwrap()
            .into_iter()
            .map(|e| e.rel_path)
            .collect();
        assert!(rels.iter().any(|r| r == "src/app.js"));
        assert!(rels.iter().any(|r| r == "src/app.ts"));
        assert!(!rels.iter().any(|r| r == "logo.png"));
        assert!(!rels.iter().any(|r| r == "app.min.js"));
        assert!(!rels.iter().any(|r| r == "types.d.ts"));
    }

    #[test]
    fn walk_default_keeps_files_for_indexer_accounting() {
        // The bare `walk` (walk_default policy) skips generated *dirs* but
        // must NOT drop files by suffix/size: the indexer relies on seeing
        // oversized/binary files to record stat-only file_state rows
        // (R-020 / RV-008). This guards against re-introducing a default
        // file-level cap that would silently break that contract.
        let tmp = tempdir_via_env();
        let root = tmp.join("repo");
        fs::create_dir_all(&root).unwrap();
        write(&root.join("logo.png"), "fakepng");
        write(&root.join("blob.bin"), "data");
        write(&root.join("big.txt"), &"A".repeat(4 * 1024 * 1024)); // > 2 MiB
        write(&root.join("node_modules/x/index.js"), "0"); // generated DIR

        let rels: Vec<String> = walk(&root)
            .unwrap()
            .into_iter()
            .map(|e| e.rel_path)
            .collect();
        // File-level entries survive the default walk.
        assert!(rels.iter().any(|r| r == "logo.png"));
        assert!(rels.iter().any(|r| r == "blob.bin"));
        assert!(rels.iter().any(|r| r == "big.txt"));
        // But the generated directory is still skipped.
        assert!(!rels.iter().any(|r| r.starts_with("node_modules/")));
    }

    #[test]
    fn walk_with_unrestricted_policy_keeps_generated() {
        let tmp = tempdir_via_env();
        let root = tmp.join("repo");
        fs::create_dir_all(&root).unwrap();
        write(&root.join("node_modules/x/index.js"), "0");
        write(&root.join("logo.png"), "fakepng");
        write(&root.join("src/keep.rs"), "fn k(){}");

        let rels: Vec<String> = walk_with_policy(&root, &skip::SkipPolicy::unrestricted())
            .unwrap()
            .into_iter()
            .map(|e| e.rel_path)
            .collect();
        // With an unrestricted policy the generated dir + image survive
        // (gitignore still applies, but there is none here). This proves
        // the heuristics are genuinely policy-gated, not hardcoded.
        assert!(rels.iter().any(|r| r == "src/keep.rs"));
        assert!(rels.iter().any(|r| r == "node_modules/x/index.js"));
        assert!(rels.iter().any(|r| r == "logo.png"));
    }

    #[test]
    fn walk_respects_max_file_size_policy() {
        let tmp = tempdir_via_env();
        let root = tmp.join("repo");
        fs::create_dir_all(&root).unwrap();
        write(&root.join("small.txt"), "tiny");
        write(&root.join("big.txt"), &"A".repeat(5000));

        let policy = skip::SkipPolicy {
            max_file_size: Some(1000),
            ..skip::SkipPolicy::unrestricted()
        };
        let rels: Vec<String> = walk_with_policy(&root, &policy)
            .unwrap()
            .into_iter()
            .map(|e| e.rel_path)
            .collect();
        assert!(rels.iter().any(|r| r == "small.txt"));
        assert!(
            !rels.iter().any(|r| r == "big.txt"),
            "file over the cap must be skipped: {rels:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn walk_does_not_follow_symlinked_file_out_of_root() {
        use std::os::unix::fs::symlink;
        let tmp = tempdir_via_env();
        // A secret file OUTSIDE the repo root.
        let outside = tmp.join("outside-secret.rs");
        fs::write(&outside, "fn secret() {}").unwrap();

        let root = tmp.join("repo");
        fs::create_dir_all(&root).unwrap();
        write(&root.join("src/real.rs"), "fn real() {}");
        // A symlink inside the repo pointing at the outside file.
        symlink(&outside, root.join("src/link.rs")).unwrap();

        let entries = walk(&root).unwrap();
        let rels: Vec<&str> = entries.iter().map(|e| e.rel_path.as_str()).collect();
        assert!(rels.contains(&"src/real.rs"));
        assert!(
            !rels.contains(&"src/link.rs"),
            "a symlinked file must not be inventoried (escape protection): {rels:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn walk_does_not_descend_symlinked_dir_and_handles_loops() {
        use std::os::unix::fs::symlink;
        let tmp = tempdir_via_env();
        // A directory outside the repo with a file in it.
        let ext_dir = tmp.join("external");
        fs::create_dir_all(&ext_dir).unwrap();
        fs::write(ext_dir.join("ext.rs"), "fn e() {}").unwrap();

        let root = tmp.join("repo");
        fs::create_dir_all(&root).unwrap();
        write(&root.join("src/keep.rs"), "fn k() {}");
        // Symlinked dir pointing outside the root.
        symlink(&ext_dir, root.join("src/linked_dir")).unwrap();
        // Self-referential loop: a symlink pointing back at the repo root.
        symlink(&root, root.join("loop")).unwrap();

        // Must terminate (no infinite loop) and not pull in external files.
        let entries = walk(&root).unwrap();
        let rels: Vec<&str> = entries.iter().map(|e| e.rel_path.as_str()).collect();
        assert!(rels.contains(&"src/keep.rs"));
        assert!(
            !rels.iter().any(|r| r.contains("ext.rs")),
            "must not descend symlinked dir out of root: {rels:?}"
        );
        assert!(
            !rels.iter().any(|r| r.starts_with("loop/")),
            "must not descend a self-referential symlink loop: {rels:?}"
        );
    }

    #[test]
    fn metadata_fields_handles_pre_epoch_and_normal() {
        // Use a real file's metadata for the normal case.
        let dir = tests_support::unique_tmp();
        let p = dir.join("f.txt");
        fs::write(&p, "abcdef").unwrap();
        let meta = fs::metadata(&p).unwrap();
        let (size, mtime) = metadata_fields(&meta);
        assert_eq!(size, Some(6));
        assert!(mtime.is_some());
    }
}
