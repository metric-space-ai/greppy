//! Generated/vendored directory and file skip heuristics.
//!
//! Upstream `src/discover/discover.c` carries several hardcoded skip
//! sets that run *in addition* to gitignore handling: `ALWAYS_SKIP_DIRS`
//! (VCS/IDE/build/cache dirs like `node_modules`, `.venv`,
//! `__pycache__`, `dist`, `build`), `ALWAYS_IGNORED_SUFFIXES` (compiled
//! / media / archive suffixes), and a max-file-size cutoff. Indexing
//! those is wasted work — they are generated, vendored, or binary — and
//! upstream drops them unconditionally regardless of `.gitignore`
//! content.
//!
//! This module ports that breadth as a *configurable* [`SkipPolicy`] so
//! a caller can opt out (e.g. a FULL index that wants everything) while
//! the default mirrors upstream's unconditional set. It composes with —
//! and never replaces — the `ignore` crate's `.gitignore`/`.ignore`
//! handling in [`crate::walk`]: gitignore still runs first, and these
//! heuristics only ever *add* exclusions.
//!
//! The dir/suffix sets here are deliberately the upstream
//! `ALWAYS_*`/unconditional tier (applied in every mode), not the
//! FAST-mode-only tier, so enabling the default policy never drops a
//! source directory a user would reasonably expect to be indexed.

/// A reasonable default cap on file size, in bytes. Files larger than
/// this are skipped by [`SkipPolicy::should_skip_file`] when a policy
/// enables size filtering. 2 MiB comfortably holds even very large
/// hand-written source files while excluding generated blobs, vendored
/// minified bundles, and data dumps. Upstream leaves the cap to the
/// caller (`opts->max_file_size`, `0` = unlimited); we surface the same
/// knob with a sane default.
pub const DEFAULT_MAX_FILE_SIZE: u64 = 2 * 1024 * 1024;

/// Directory basenames that are skipped as generated / vendored /
/// dependency trees, mirroring upstream `ALWAYS_SKIP_DIRS` (minus the
/// entries already enforced unconditionally by
/// [`crate::ALWAYS_SKIP_DIRS`], i.e. `.git`/`target`/`.vendor`).
///
/// Matched by exact basename anywhere in the path, so a nested
/// `pkg/node_modules/` is dropped too. These are the dirs that, when
/// present, almost never contain first-party source the indexer should
/// ingest.
pub const GENERATED_DIRS: &[&str] = &[
    // VCS (besides .git, which crate::ALWAYS_SKIP_DIRS already drops)
    ".hg",
    ".svn",
    ".worktrees",
    // IDE / editor
    ".idea",
    ".vs",
    ".vscode",
    ".eclipse",
    ".claude",
    // Python
    ".cache",
    ".eggs",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "htmlcov",
    "site-packages",
    "venv",
    // JS / TS
    ".npm",
    ".nyc_output",
    ".pnpm-store",
    ".yarn",
    "bower_components",
    "coverage",
    "node_modules",
    ".next",
    ".nuxt",
    ".svelte-kit",
    ".angular",
    ".turbo",
    ".parcel-cache",
    ".docusaurus",
    ".expo",
    // Build artifacts
    "dist",
    "obj",
    "Pods",
    ".terraform",
    ".serverless",
    "bazel-bin",
    "bazel-out",
    "bazel-testlogs",
    "build",
    // Language caches
    ".cargo",
    ".stack-work",
    ".dart_tool",
    "zig-cache",
    "zig-out",
    ".metals",
    ".bloop",
    ".bsp",
    ".ccls-cache",
    ".clangd",
    "elm-stuff",
    "_opam",
    ".cpcache",
    ".shadow-cljs",
    // Deploy
    ".vercel",
    ".netlify",
    // Misc vendored
    ".qdrant_code_embeddings",
    "vendor",
    "vendored",
];

/// File suffixes that are always ignored, mirroring upstream
/// `ALWAYS_IGNORED_SUFFIXES`: compiled artifacts, images, fonts,
/// archives, and on-disk databases. These are matched case-insensitively
/// against the filename. Note this is intentionally *broader* than the
/// content-based [`crate::is_binary_file`] sniff: a `.min.js` is valid
/// UTF-8 text yet still generated/minified and not worth indexing.
pub const IGNORED_SUFFIXES: &[&str] = &[
    // Editor / temp
    ".tmp", "~", ".swp", ".bak", ".orig", // Compiled / object
    ".pyc", ".pyo", ".o", ".a", ".so", ".dll", ".dylib", ".class", ".wasm", ".node", ".exe",
    ".rlib", ".elc", ".beam", // Images
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".bmp", ".tiff", ".webp", ".svg",
    // Fonts
    ".woff", ".woff2", ".ttf", ".eot", ".otf", // Data / db
    ".bin", ".dat", ".db", ".sqlite", ".sqlite3", ".parquet", ".avro", ".pb",
    // Archives
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".rar", ".7z", ".jar", ".war", ".ear",
    // Generated/minified web bundles
    ".min.js", ".min.css", ".map", // Keys / certs
    ".pem", ".crt", ".key", ".cer", ".p12",
];

/// Substring patterns that mark a file as generated/minified regardless
/// of its directory. Mirrors a subset of upstream `FAST_PATTERNS` that
/// are safe to apply unconditionally (these are always generated, never
/// hand-written sources): protobuf/gRPC stubs and TypeScript declaration
/// bundles. Matched as a case-sensitive substring of the basename.
const GENERATED_PATTERNS: &[&str] = &[
    ".d.ts",
    ".bundle.",
    ".chunk.",
    ".generated.",
    ".pb.go",
    "_pb2.py",
    ".pb2.py",
    "_grpc.pb.go",
];

/// Configurable policy for the additive (post-gitignore) skip
/// heuristics. The [`Default`] matches upstream's unconditional
/// behaviour: skip generated dirs, ignored suffixes, generated-name
/// patterns, and oversized files. Each knob can be disabled
/// independently for a "FULL"-style index that wants everything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SkipPolicy {
    /// Skip directories whose basename is in [`GENERATED_DIRS`].
    pub skip_generated_dirs: bool,
    /// Skip files whose name ends with an [`IGNORED_SUFFIXES`] entry.
    pub skip_ignored_suffixes: bool,
    /// Skip files whose name contains a generated-name pattern (e.g.
    /// `.d.ts`, `.pb.go`).
    pub skip_generated_patterns: bool,
    /// When `Some(n)`, skip files larger than `n` bytes. `None` disables
    /// the size cap (upstream `max_file_size == 0`).
    pub max_file_size: Option<u64>,
}

impl Default for SkipPolicy {
    fn default() -> Self {
        Self {
            skip_generated_dirs: true,
            skip_ignored_suffixes: true,
            skip_generated_patterns: true,
            max_file_size: Some(DEFAULT_MAX_FILE_SIZE),
        }
    }
}

impl SkipPolicy {
    /// The policy used by the bare [`crate::walk`] entry point.
    ///
    /// This skips generated/vendored **directories** (`node_modules`,
    /// `dist`, `build`, `.venv`, …) — the high-value, low-risk parity
    /// win — but deliberately leaves *file-level* filtering off:
    /// `skip_ignored_suffixes`, `skip_generated_patterns`, and the size
    /// cap are all disabled. That is because the indexer
    /// (`crates/indexer`) consumes `walk`'s inventory and performs its
    /// own per-file accounting: it records stat-only `file_state` rows
    /// for oversized and unsupported/binary files (R-020 / RV-008) rather
    /// than having them silently dropped here. Dropping `.bin`/oversized
    /// files at the walk layer would break that contract.
    ///
    /// Callers that want the full upstream `ALWAYS_*` breadth (including
    /// suffix/pattern/size dropping) can pass [`SkipPolicy::default`] to
    /// [`crate::walk_with_policy`].
    pub fn walk_default() -> Self {
        Self {
            skip_generated_dirs: true,
            skip_ignored_suffixes: false,
            skip_generated_patterns: false,
            max_file_size: None,
        }
    }

    /// A policy that disables every heuristic — equivalent to upstream
    /// FULL mode (gitignore still applies in the walker, but no extra
    /// generated/size filtering). Useful for callers that must see every
    /// tracked file.
    pub fn unrestricted() -> Self {
        Self {
            skip_generated_dirs: false,
            skip_ignored_suffixes: false,
            skip_generated_patterns: false,
            max_file_size: None,
        }
    }

    /// True if any path component of the forward-slash relative path
    /// `rel` is a generated/vendored directory under this policy.
    pub fn should_skip_dir_component(&self, rel: &str) -> bool {
        if !self.skip_generated_dirs {
            return false;
        }
        rel.split('/').any(is_generated_dir)
    }

    /// True if the file named `name` (basename only) should be skipped by
    /// the name-based heuristics (ignored suffix or generated pattern)
    /// under this policy. `size` is the file's size in bytes when known;
    /// pass `None` to skip the size check.
    pub fn should_skip_file(&self, name: &str, size: Option<u64>) -> bool {
        if self.skip_ignored_suffixes && is_ignored_file(name) {
            return true;
        }
        if self.skip_generated_patterns && matches_generated_pattern(name) {
            return true;
        }
        if let (Some(cap), Some(sz)) = (self.max_file_size, size) {
            if sz > cap {
                return true;
            }
        }
        false
    }
}

/// True if `name` is the basename of a generated/vendored directory in
/// [`GENERATED_DIRS`]. Exact, case-sensitive match (directory names like
/// `node_modules` are conventionally exact).
pub fn is_generated_dir(name: &str) -> bool {
    GENERATED_DIRS.contains(&name)
}

/// True if `name` (a filename) ends with one of [`IGNORED_SUFFIXES`].
/// The comparison is case-insensitive so `IMAGE.PNG` and `image.png`
/// both match.
pub fn is_ignored_file(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    IGNORED_SUFFIXES.iter().any(|suf| lower.ends_with(suf))
}

/// True if `name` contains a known generated-file substring pattern.
fn matches_generated_pattern(name: &str) -> bool {
    GENERATED_PATTERNS.iter().any(|p| name.contains(p))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_policy_skips_common_generated_dirs() {
        let p = SkipPolicy::default();
        for d in [
            "node_modules",
            "dist",
            "build",
            ".venv",
            "__pycache__",
            "venv",
            "coverage",
            ".next",
            "vendor",
        ] {
            assert!(is_generated_dir(d), "{d} should be a generated dir");
            assert!(
                p.should_skip_dir_component(&format!("pkg/{d}/inner/file.js")),
                "nested {d} should be skipped"
            );
        }
    }

    #[test]
    fn generated_dir_match_is_basename_exact_not_substring() {
        // A source dir whose name merely contains a generated dir name
        // must NOT be skipped.
        assert!(!is_generated_dir("node_modules_helper"));
        assert!(!is_generated_dir("mydist"));
        assert!(!is_generated_dir("building"));
        let p = SkipPolicy::default();
        assert!(!p.should_skip_dir_component("src/distance/calc.rs"));
        assert!(!p.should_skip_dir_component("src/builds.rs"));
    }

    #[test]
    fn unrestricted_policy_skips_nothing() {
        let p = SkipPolicy::unrestricted();
        assert!(!p.should_skip_dir_component("a/node_modules/b.js"));
        assert!(!p.should_skip_file("image.png", Some(u64::MAX)));
        assert!(!p.should_skip_file("bundle.min.js", None));
    }

    #[test]
    fn ignored_suffixes_match_case_insensitively() {
        for n in ["logo.png", "LOGO.PNG", "Font.WOFF2", "lib.so", "app.pyc"] {
            assert!(is_ignored_file(n), "{n} should be ignored");
        }
        // Plain source must not be ignored by suffix.
        assert!(!is_ignored_file("main.rs"));
        assert!(!is_ignored_file("index.ts"));
    }

    #[test]
    fn minified_and_bundle_files_are_skipped() {
        let p = SkipPolicy::default();
        assert!(p.should_skip_file("app.min.js", None));
        assert!(p.should_skip_file("styles.min.css", None));
        assert!(p.should_skip_file("vendor.bundle.js", None));
        assert!(p.should_skip_file("types.d.ts", None));
        assert!(p.should_skip_file("service.pb.go", None));
        assert!(p.should_skip_file("schema_pb2.py", None));
        // A normal .js / .ts is kept.
        assert!(!p.should_skip_file("app.js", None));
        assert!(!p.should_skip_file("app.ts", None));
    }

    #[test]
    fn oversized_files_are_skipped_only_when_cap_set() {
        let p = SkipPolicy::default();
        assert!(p.should_skip_file("big.csv", Some(DEFAULT_MAX_FILE_SIZE + 1)));
        assert!(!p.should_skip_file("ok.csv", Some(DEFAULT_MAX_FILE_SIZE)));
        // Unknown size => no size-based skip.
        assert!(!p.should_skip_file("ok.csv", None));

        let no_cap = SkipPolicy {
            max_file_size: None,
            ..SkipPolicy::default()
        };
        assert!(!no_cap.should_skip_file("big.csv", Some(u64::MAX)));
    }

    #[test]
    fn individual_knobs_are_independent() {
        let p = SkipPolicy {
            skip_ignored_suffixes: true,
            ..SkipPolicy::unrestricted()
        };
        assert!(p.should_skip_file("a.png", None));
        assert!(!p.should_skip_dir_component("a/node_modules/b"));
        assert!(!p.should_skip_file("a.min.js", None) || p.should_skip_file("a.min.js", None));
        // .min.js ends with an ignored suffix, so suffix-only still skips it.
        assert!(p.should_skip_file("a.min.js", None));
        // but a pure pattern (no ignored suffix) is not skipped with
        // patterns disabled:
        assert!(!p.should_skip_file("svc.pb.go", None));
    }
}
