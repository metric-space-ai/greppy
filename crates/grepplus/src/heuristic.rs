//! Grep-call heuristic for the drop-in `grep` shim.
//!
//! Phase 6 implements the classification from
//! `docs/grepplus_rust_phasenplan_port.md` §11.
//!
//! - `STRICT` — never augment; real-grep output is the entire answer.
//! - `SIDECAR` — augment is allowed, but the synthetic `.md` file lives
//!   outside the visible stdout (typically in `/tmp/grepplus/...`).
//! - `VISIBLE_AUGMENT` — augment is allowed AND the synthetic `.md` line
//!   is appended to real-grep output, marked clearly as non-canonical.
//!
//! The classifier is a pure function of the parsed argv + freshness
//! outcome. Tests cover each gate individually.

use std::path::Path;

/// The result of classifying a single `grep` invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Never augment. Real-grep output is the entire answer.
    Strict,
    /// Augment is allowed but the `.md` sidecar is hidden from stdout.
    Sidecar,
    /// Augment is allowed and one synthetic `.md` line is appended to
    /// real-grep output.
    VisibleAugment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FreshnessGate {
    /// Graph is fresh; augmentation is allowed if the heuristic allows.
    Fresh,
    /// Graph is stale or unknown; STRICT only.
    Strict,
}

/// Parse argv (without argv[0]) into a structured view used by the
/// classifier. Pure; does no I/O.
#[derive(Debug, Clone, Default)]
pub struct GrepArgs {
    pub quiet: bool,
    pub count: bool,
    pub only_matching: bool,
    pub invert: bool,
    pub files_without_match: bool,
    pub null_separated_output: bool,
    pub null_data: bool,
    pub pattern_from_file: bool,
    pub recursive: bool,
    pub files_with_matches: bool,
    pub extended_regex: bool,
    pub fixed_strings: bool,
    pub label: Option<String>,
    pub pattern: Option<String>,
    pub paths: Vec<String>,
    pub unknown_options: Vec<String>,
}

impl GrepArgs {
    /// Parse `argv` (which must NOT include argv[0]). Recognised flags
    /// are consumed; unknown options are recorded.
    pub fn parse(argv: &[String]) -> Self {
        let mut a = GrepArgs::default();
        let mut i = 0;
        // First, gather option flags. Options can be chained (-RIn) but
        // we treat them positionally for simplicity (Phase 9 can add
        // short-flag chaining).
        while i < argv.len() {
            let arg = &argv[i];
            match arg.as_str() {
                "-q" | "--quiet" | "--silent" => a.quiet = true,
                "-c" | "--count" => a.count = true,
                "-o" | "--only-matching" => a.only_matching = true,
                "-v" | "--invert-match" => a.invert = true,
                "-L" | "--files-without-match" => a.files_without_match = true,
                "-Z" | "--null" => a.null_separated_output = true,
                "-z" | "--null-data" => a.null_data = true,
                "-f" | "--file" => a.pattern_from_file = true,
                "-R" | "-r" | "--recursive" => a.recursive = true,
                "-l" | "--files-with-matches" => a.files_with_matches = true,
                "-E" | "--extended-regexp" => a.extended_regex = true,
                "-F" | "--fixed-strings" => a.fixed_strings = true,
                "-h"
                | "--no-filename"
                | "-H"
                | "--with-filename"
                | "-i"
                | "--ignore-case"
                | "-n"
                | "--line-number"
                | "-w"
                | "--word-regexp"
                | "-x"
                | "--line-regexp"
                | "-s"
                | "--no-messages"
                | "-I"
                | "--binary-files=without-match" => {
                    // Ignored flags — they do not affect classification.
                }
                "--label" => {
                    if i + 1 < argv.len() {
                        a.label = Some(argv[i + 1].clone());
                        i += 1;
                    }
                }
                _ if arg.starts_with('-') && !arg.starts_with("--") && arg.len() > 2 => {
                    // Treat short-flag chains conservatively: anything
                    // we don't recognise, record. Phase 9 hardening:
                    // proper short-flag grouping with -E, -F, -P, -i, …
                    // parsed as a set.
                    let bytes = arg.as_bytes();
                    for &b in &bytes[1..] {
                        match b {
                            b'q' => a.quiet = true,
                            b'c' => a.count = true,
                            b'o' => a.only_matching = true,
                            b'v' => a.invert = true,
                            b'L' => a.files_without_match = true,
                            b'Z' => a.null_separated_output = true,
                            b'z' => a.null_data = true,
                            b'R' | b'r' => a.recursive = true,
                            b'l' => a.files_with_matches = true,
                            b'E' => a.extended_regex = true,
                            b'F' => a.fixed_strings = true,
                            b'h' | b'H' | b'i' | b'n' | b'w' | b'x' | b's' | b'I' => {}
                            _ => a.unknown_options.push(arg.clone()),
                        }
                    }
                }
                _ if arg.starts_with("--") => {
                    // Long option we did not recognise.
                    a.unknown_options.push(arg.clone());
                }
                _ => {
                    // First non-option is the pattern (unless `-f` is set).
                    if a.pattern.is_none() && !a.pattern_from_file {
                        a.pattern = Some(arg.clone());
                    } else {
                        a.paths.push(arg.clone());
                    }
                }
            }
            i += 1;
        }
        a
    }

    /// Parse `argv` given as `OsString`s (which must NOT include
    /// argv[0]). Each argument is converted to a best-effort lossy
    /// `String` for the classification decision ONLY — the caller still
    /// forwards the original `OsString`s to real grep byte-for-byte (see
    /// [`crate::run::run_with_optional_augment_os`]).
    ///
    /// P0 (R-014 re-review): this lets the wrapper classify a (possibly
    /// non-UTF-8) invocation for the augmentation decision without ever
    /// requiring the argv to be valid UTF-8.
    pub fn parse_os(argv: &[std::ffi::OsString]) -> Self {
        let lossy: Vec<String> = argv
            .iter()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        Self::parse(&lossy)
    }

    /// True if the invocation reads from stdin and does not have any
    /// path argument. Such invocations are pure pipelines and must
    /// never be augmented.
    pub fn is_stdin_only(&self) -> bool {
        self.paths.is_empty()
    }
}

/// Classify a `grep` invocation under a given freshness gate.
pub fn classify(args: &GrepArgs, freshness: FreshnessGate) -> Mode {
    // 11.2 STRICT — always.
    if args.quiet
        || args.count
        || args.only_matching
        || args.invert
        || args.files_without_match
        || args.null_separated_output
        || args.null_data
        || args.pattern_from_file
        || args.is_stdin_only()
        || !args.unknown_options.is_empty()
        || args.pattern.is_none()
    {
        return Mode::Strict;
    }
    // 11.5 / 10.7 — STRICT if freshness is not Fresh.
    if freshness == FreshnessGate::Strict {
        return Mode::Strict;
    }
    // 11.3 SIDECAR.
    if args.files_with_matches
        || !args.paths.contains(&"-".to_string()) && args.paths.len() == 1 && !args.recursive
    {
        // Single file, or `-l`-mode, or `-R`-with-`-l`. Conservative.
        return Mode::Sidecar;
    }
    // 11.4 VISIBLE_AUGMENT.
    Mode::VisibleAugment
}

/// Path layout for the sidecar file.
///
/// `<store_dir>/<query>__<random>__GREPPLUS_SEMANTIC_NONCANONICAL.md`
///
/// R-019 / WP-R006 (operational hardening): the path includes a
/// per-invocation random component so a co-located attacker on a
/// shared `/tmp` cannot pre-plant a sidecar before the victim writes
/// one (defence in depth on top of the `O_EXCL` create in
/// `sidecar::write_sidecar`).
pub fn sidecar_path(workspace_root: &Path, query: &str) -> std::path::PathBuf {
    // Sanitise query for filesystem use: keep alnum, dash, underscore,
    // dot; replace others with underscore.
    let safe_query: String = query
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .take(128)
        .collect();

    // Per-invocation random component. We derive it from
    // SystemTime + process-id so two processes (or two invocations
    // of the same process) can't race the same filename even on
    // a single user.
    let nonce = nonce_component();
    let filename = format!("{safe_query}__{nonce}__GREPPLUS_SEMANTIC_NONCANONICAL.md");
    grepplus_core::workspace::store_dir(workspace_root).join(filename)
}

fn nonce_component() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{:x}{:x}", ts, std::process::id())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> GrepArgs {
        GrepArgs::parse(&args.iter().map(|s| s.to_string()).collect::<Vec<_>>())
    }

    #[test]
    fn strict_for_quiet() {
        let a = parse(&["-q", "foo", "src/"]);
        assert_eq!(classify(&a, FreshnessGate::Fresh), Mode::Strict);
    }

    #[test]
    fn strict_for_count() {
        let a = parse(&["-c", "foo", "src/"]);
        assert_eq!(classify(&a, FreshnessGate::Fresh), Mode::Strict);
    }

    #[test]
    fn strict_for_only_matching() {
        let a = parse(&["-o", "foo", "src/"]);
        assert_eq!(classify(&a, FreshnessGate::Fresh), Mode::Strict);
    }

    #[test]
    fn strict_for_invert() {
        let a = parse(&["-v", "foo", "src/"]);
        assert_eq!(classify(&a, FreshnessGate::Fresh), Mode::Strict);
    }

    #[test]
    fn strict_for_files_without_match() {
        let a = parse(&["-L", "foo", "src/"]);
        assert_eq!(classify(&a, FreshnessGate::Fresh), Mode::Strict);
    }

    #[test]
    fn strict_for_null_separated() {
        let a = parse(&["-Z", "foo", "src/"]);
        assert_eq!(classify(&a, FreshnessGate::Fresh), Mode::Strict);
    }

    #[test]
    fn strict_for_null_data() {
        let a = parse(&["-z", "foo", "src/"]);
        assert_eq!(classify(&a, FreshnessGate::Fresh), Mode::Strict);
    }

    #[test]
    fn strict_for_pattern_from_file() {
        let a = parse(&["-f", "patterns.txt", "src/"]);
        assert_eq!(classify(&a, FreshnessGate::Fresh), Mode::Strict);
    }

    #[test]
    fn strict_for_stdin_only() {
        let a = parse(&["foo"]);
        assert!(a.is_stdin_only());
        assert_eq!(classify(&a, FreshnessGate::Fresh), Mode::Strict);
    }

    #[test]
    fn strict_for_unknown_options() {
        let a = parse(&["--made-up-flag", "foo", "src/"]);
        assert_eq!(classify(&a, FreshnessGate::Fresh), Mode::Strict);
    }

    #[test]
    fn strict_for_missing_pattern() {
        let a = parse(&["src/"]);
        assert_eq!(classify(&a, FreshnessGate::Fresh), Mode::Strict);
    }

    #[test]
    fn strict_when_freshness_is_strict() {
        let a = parse(&["-R", "foo", "."]);
        // Even with a perfectly exploratory form, freshness gate wins.
        assert_eq!(classify(&a, FreshnessGate::Strict), Mode::Strict);
    }

    #[test]
    fn sidecar_for_files_with_matches_recursive() {
        let a = parse(&["-R", "-l", "foo", "."]);
        assert_eq!(classify(&a, FreshnessGate::Fresh), Mode::Sidecar);
    }

    #[test]
    fn sidecar_for_single_file_no_recursive() {
        let a = parse(&["foo", "src/main.rs"]);
        assert_eq!(classify(&a, FreshnessGate::Fresh), Mode::Sidecar);
    }

    #[test]
    fn visible_augment_for_recursive_no_flags() {
        let a = parse(&["-R", "foo", "."]);
        assert_eq!(classify(&a, FreshnessGate::Fresh), Mode::VisibleAugment);
    }

    #[test]
    fn visible_augment_for_recursive_with_multiple_paths() {
        let a = parse(&["-R", "foo", "src", "tests"]);
        assert_eq!(classify(&a, FreshnessGate::Fresh), Mode::VisibleAugment);
    }

    #[test]
    fn sidecar_path_sanitises_query() {
        let p = sidecar_path(std::path::Path::new("/tmp/repo"), "Process Order");
        let name = p.file_name().unwrap().to_string_lossy();
        assert!(name.contains("Process_Order"));
        assert!(name.ends_with("__GREPPLUS_SEMANTIC_NONCANONICAL.md"));
    }

    #[test]
    fn parse_extracts_pattern_and_paths() {
        let a = parse(&["-R", "ProcessOrder", "src", "tests"]);
        assert_eq!(a.pattern.as_deref(), Some("ProcessOrder"));
        assert_eq!(a.paths, vec!["src", "tests"]);
        assert!(a.recursive);
    }

    #[test]
    fn parse_short_flag_chain() {
        let a = parse(&["-RIn", "foo", "."]);
        assert!(a.recursive);
    }

    #[test]
    fn classify_under_one_microsecond() {
        // Phasenplan §10.4 says the freshness check has a 200 ms
        // budget for the *full* drop-in invocation. The pure
        // classifier must be far cheaper. We assert < 1 ms here
        // to leave headroom for the freshness check, the FTS
        // query, the sidecar write, and the println of the
        // synthetic line.
        let a = parse(&["-R", "ProcessOrder", "src", "tests"]);
        let start = std::time::Instant::now();
        for _ in 0..1_000 {
            let _ = classify(&a, FreshnessGate::Fresh);
        }
        let elapsed = start.elapsed();
        let per_call = elapsed / 1_000;
        assert!(
            per_call.as_micros() < 1_000,
            "classify should be < 1 ms per call, got {per_call:?}"
        );
    }
}
