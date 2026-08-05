//! Write-confinement sandbox for agent tool subprocesses.
//!
//! Both the `greppy` and `bash` tools spawn a child process. When
//! [`SandboxMode::Enforce`] is active, that child may **write** only under an
//! explicit allowlist of roots (the run worktree, temp dir, greppy store,
//! `~/.cargo`, and the platform user cache). **Reads stay unrestricted** so
//! builds can reach system headers and package registries; **network stays
//! open** in this iteration (needed for `cargo fetch`).
//!
//! Platform backends:
//! - **macOS** — rewrite the invocation as
//!   `/usr/bin/sandbox-exec -p <seatbelt-profile> <bin> <args…>`. Profile
//!   generation is fail-closed: a rejected profile is [`SandboxError`].
//! - **Linux** — rewrite the invocation as a **launcher mode**:
//!   `<current_exe> __agent-sandbox-landlock <spec> -- <bin> <args…>`. That
//!   hidden internal process (already post-exec, single-threaded) opens the
//!   roots, builds and applies a Landlock ruleset requiring at least ABI V3
//!   write rights (including `Truncate`), verifies full enforcement, then
//!   `exec`s the real command. No Landlock work happens in `pre_exec`. If the
//!   kernel cannot fully enforce the requested rights,
//!   [`SandboxError::Unsupported`] is returned so the CLI can warn once and
//!   continue unsandboxed.
//! - **other** — [`SandboxError::Unsupported`] under `Enforce`.
//!
//! Before enforcement the trusted parent creates and validates every writable
//! root (create_dir_all, reject symlinks / non-directories, canonicalize).

use std::ffi::OsStr;
#[cfg(target_os = "linux")]
use std::ffi::OsString;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Hidden CLI argv token for the Linux Landlock launcher process.
///
/// Intercepted in the greppy CLI before ordinary routing. Undocumented in help;
/// only ever spawned from [`apply`] under [`SandboxMode::Enforce`].
pub const LANDLOCK_LAUNCHER_ARG: &str = "__agent-sandbox-landlock";

/// Distinctive exit code when the Landlock launcher cannot set up confinement.
///
/// Surfaces as a tool/setup failure in the parent (never silently falls through
/// to an unrestricted real command).
pub const LANDLOCK_LAUNCHER_EXIT_SETUP: u8 = 172;

/// Writable-root allowlist for a sandboxed tool subprocess.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxSpec {
    /// Absolute (preferably canonical) directories the child may write under.
    pub writable_roots: Vec<PathBuf>,
}

/// Sandbox policy applied to every tool subprocess of a [`crate::GreppyEnv`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum SandboxMode {
    /// No confinement (default; non-`-p` callers unchanged).
    #[default]
    Off,
    /// Write-confine to [`SandboxSpec::writable_roots`].
    Enforce(SandboxSpec),
}

/// Failures while building or installing a sandbox.
#[derive(Debug)]
pub enum SandboxError {
    /// Kernel/OS cannot enforce (Linux without Landlock ABI ≥ V3; non-macOS/non-Linux).
    /// The CLI warns once and continues unsandboxed.
    Unsupported,
    /// Seatbelt profile rejected by `sandbox-exec`, or profile build failed.
    Profile(String),
    /// Landlock ruleset construction / restrict_self failed.
    Landlock(String),
    /// Path handling / I/O while preparing the sandbox.
    Io(String),
}

impl fmt::Display for SandboxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported => write!(f, "sandbox unsupported on this platform/kernel"),
            Self::Profile(msg) => write!(f, "sandbox profile: {msg}"),
            Self::Landlock(msg) => write!(f, "landlock: {msg}"),
            Self::Io(msg) => write!(f, "sandbox i/o: {msg}"),
        }
    }
}

impl std::error::Error for SandboxError {}

/// Configure `cmd` to run `bin` with `args` under `mode`.
///
/// Replaces `*cmd` with a fully-built `Command` (program + argv). Callers then
/// set cwd / stdio / env and spawn. Under [`SandboxMode::Off`] this is simply
/// `Command::new(bin).args(args)`. Under `Enforce` the platform backend is
/// applied (seatbelt rewrite on macOS; Landlock launcher on Linux).
///
/// Fail-closed: any error preparing the sandbox is returned; the caller must
/// not spawn an unrestricted child in that case.
pub fn apply(
    cmd: &mut Command,
    bin: &Path,
    args: &[impl AsRef<OsStr>],
    mode: &SandboxMode,
) -> Result<(), SandboxError> {
    match mode {
        SandboxMode::Off => {
            *cmd = Command::new(bin);
            cmd.args(args.iter().map(AsRef::as_ref));
            Ok(())
        }
        SandboxMode::Enforce(spec) => apply_enforce(cmd, bin, args, spec),
    }
}

/// Probe whether `mode` can be enforced on this host.
///
/// Cheap preflight used by the CLI before starting the agent loop:
/// - `Off` → always ok.
/// - macOS `Enforce` → prepare roots, then `sandbox-exec` must exist and accept
///   the generated profile (`/usr/bin/true` dry-run).
/// - Linux `Enforce` → prepare roots, then Landlock must support the V3 write
///   rights floor (including `Truncate`) under hard-requirement compatibility.
/// - other OS `Enforce` → [`SandboxError::Unsupported`].
pub fn preflight(mode: &SandboxMode) -> Result<(), SandboxError> {
    match mode {
        SandboxMode::Off => Ok(()),
        SandboxMode::Enforce(spec) => preflight_enforce(spec),
    }
}

/// Linux Landlock launcher entry point (CLI intercept).
///
/// Expected argv shape (as received by the process):
/// `argv[0]=exe`, `argv[1]=__agent-sandbox-landlock`, `argv[2]=<spec-json>`,
/// `argv[3]=--`, `argv[4…]=real program + args`.
///
/// Refuses to run unless [`crate::AGENT_RUN_ENV`] is present (set by
/// `prepare_tool_env` on the parent-spawned launcher command). On success this
/// function does not return (`exec`). On failure it prints a distinctive
/// message to stderr and returns [`LANDLOCK_LAUNCHER_EXIT_SETUP`].
#[cfg(target_os = "linux")]
pub fn run_landlock_launcher(argv: &[OsString]) -> u8 {
    match run_landlock_launcher_inner(argv) {
        Ok(()) => unreachable!("exec returned Ok"),
        Err(msg) => {
            eprintln!("greppy-agent-sandbox-landlock: {msg}");
            LANDLOCK_LAUNCHER_EXIT_SETUP
        }
    }
}

#[cfg(target_os = "linux")]
fn run_landlock_launcher_inner(argv: &[OsString]) -> Result<(), String> {
    if std::env::var_os(crate::AGENT_RUN_ENV).is_none() {
        return Err(
            "refusing to run outside an agent tool subprocess (missing GREPPY_AGENT_RUN)".into(),
        );
    }
    // argv: [exe, launcher_arg, spec, "--", real_bin, real_args…]
    if argv.len() < 5 || argv.get(3).map(OsString::as_os_str) != Some(OsStr::new("--")) {
        return Err(format!(
            "bad launcher argv (want: {LANDLOCK_LAUNCHER_ARG} <spec> -- <bin> <args…>)"
        ));
    }
    let spec_arg = argv[2]
        .to_str()
        .ok_or_else(|| "launcher spec is not valid UTF-8".to_string())?;
    let roots = decode_landlock_spec(spec_arg).map_err(|e| format!("spec: {e}"))?;
    // Roots were prepared in the trusted parent; re-validate existence/type here
    // without create_dir_all (we must not create new write targets post-fork of
    // the policy decision — only open what the parent already staged).
    let roots = validate_existing_roots(&roots).map_err(|e| e.to_string())?;
    landlock_restrict(&roots).map_err(|e| e.to_string())?;

    let real_bin = PathBuf::from(&argv[4]);
    let real_args: Vec<&OsStr> = argv[5..].iter().map(OsString::as_os_str).collect();

    use std::os::unix::process::CommandExt;
    let err = Command::new(&real_bin).args(real_args).exec();
    Err(format!("exec {} failed: {err}", real_bin.display()))
}

fn apply_enforce(
    cmd: &mut Command,
    bin: &Path,
    args: &[impl AsRef<OsStr>],
    spec: &SandboxSpec,
) -> Result<(), SandboxError> {
    // Trusted parent: create + validate every writable root before confinement.
    let roots = prepare_writable_roots(&spec.writable_roots)?;
    #[cfg(target_os = "macos")]
    {
        apply_macos(cmd, bin, args, &roots)
    }
    #[cfg(target_os = "linux")]
    {
        apply_linux(cmd, bin, args, &roots)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        let _ = (cmd, bin, args, roots);
        Err(SandboxError::Unsupported)
    }
}

fn preflight_enforce(spec: &SandboxSpec) -> Result<(), SandboxError> {
    let roots = prepare_writable_roots(&spec.writable_roots)?;
    #[cfg(target_os = "macos")]
    {
        preflight_macos(&roots)
    }
    #[cfg(target_os = "linux")]
    {
        preflight_linux(&roots)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        let _ = roots;
        Err(SandboxError::Unsupported)
    }
}

// ── root preparation (trusted parent) ───────────────────────────────────────

/// Create, validate, and canonicalize every intended writable root.
///
/// - Missing directories are created with `create_dir_all`.
/// - Symlink roots and non-directories are rejected.
/// - Result paths are absolute canonical directories, deduplicated.
///
/// Called only in the trusted parent before enforcement is installed.
pub fn prepare_writable_roots(roots: &[PathBuf]) -> Result<Vec<PathBuf>, SandboxError> {
    let mut out = Vec::with_capacity(roots.len());
    for r in roots {
        if !r.exists() {
            std::fs::create_dir_all(r).map_err(|e| {
                SandboxError::Io(format!("cannot create writable root {}: {e}", r.display()))
            })?;
        }
        let meta = std::fs::symlink_metadata(r).map_err(|e| {
            SandboxError::Io(format!("cannot stat writable root {}: {e}", r.display()))
        })?;
        if meta.file_type().is_symlink() {
            return Err(SandboxError::Io(format!(
                "writable root is a symlink (refusing): {}",
                r.display()
            )));
        }
        if !meta.is_dir() {
            return Err(SandboxError::Io(format!(
                "writable root is not a directory: {}",
                r.display()
            )));
        }
        let c = std::fs::canonicalize(r).map_err(|e| {
            SandboxError::Io(format!(
                "cannot canonicalize writable root {}: {e}",
                r.display()
            ))
        })?;
        if !c.is_dir() {
            return Err(SandboxError::Io(format!(
                "writable root is not a directory after canonicalize: {}",
                c.display()
            )));
        }
        if !out.contains(&c) {
            out.push(c);
        }
    }
    Ok(out)
}

/// Re-validate roots that the parent already prepared (launcher path).
#[cfg(target_os = "linux")]
fn validate_existing_roots(roots: &[PathBuf]) -> Result<Vec<PathBuf>, SandboxError> {
    let mut out = Vec::with_capacity(roots.len());
    for r in roots {
        let meta = std::fs::symlink_metadata(r).map_err(|e| {
            SandboxError::Io(format!(
                "writable root missing at enforce time {}: {e}",
                r.display()
            ))
        })?;
        if meta.file_type().is_symlink() {
            return Err(SandboxError::Io(format!(
                "writable root is a symlink (refusing): {}",
                r.display()
            )));
        }
        if !meta.is_dir() {
            return Err(SandboxError::Io(format!(
                "writable root is not a directory: {}",
                r.display()
            )));
        }
        let c = std::fs::canonicalize(r).map_err(|e| {
            SandboxError::Io(format!(
                "cannot canonicalize writable root {}: {e}",
                r.display()
            ))
        })?;
        if !out.contains(&c) {
            out.push(c);
        }
    }
    Ok(out)
}

// ── macOS (Seatbelt / sandbox-exec) ─────────────────────────────────────────

#[cfg(target_os = "macos")]
fn apply_macos(
    cmd: &mut Command,
    bin: &Path,
    args: &[impl AsRef<OsStr>],
    roots: &[PathBuf],
) -> Result<(), SandboxError> {
    let profile = render_seatbelt_profile(roots);
    // Fail-closed: if sandbox-exec is missing, surface as Profile error so the
    // CLI aborts rather than silently running unrestricted.
    if !Path::new("/usr/bin/sandbox-exec").exists() {
        return Err(SandboxError::Profile(
            "/usr/bin/sandbox-exec is missing".into(),
        ));
    }
    *cmd = Command::new("/usr/bin/sandbox-exec");
    cmd.arg("-p").arg(profile).arg(bin);
    cmd.args(args.iter().map(AsRef::as_ref));
    Ok(())
}

#[cfg(target_os = "macos")]
fn preflight_macos(roots: &[PathBuf]) -> Result<(), SandboxError> {
    if !Path::new("/usr/bin/sandbox-exec").exists() {
        return Err(SandboxError::Profile(
            "/usr/bin/sandbox-exec is missing".into(),
        ));
    }
    let profile = render_seatbelt_profile(roots);
    // Dry-run: rejected profiles exit non-zero without executing the command.
    let status = Command::new("/usr/bin/sandbox-exec")
        .arg("-p")
        .arg(&profile)
        .arg("/usr/bin/true")
        .status()
        .map_err(|e| SandboxError::Profile(format!("failed to invoke sandbox-exec: {e}")))?;
    if !status.success() {
        return Err(SandboxError::Profile(format!(
            "sandbox-exec rejected the generated profile (exit {status})"
        )));
    }
    Ok(())
}

/// Build a Seatbelt (SBPL) profile that denies all file writes except under the
/// canonicalized roots, plus the device nodes a shell realistically needs.
///
/// Empirically verified on macOS: `bash -lc 'touch <root>/x'`, `git commit`
/// inside a worktree root, and `cargo test` in a scratch crate all succeed;
/// `touch $HOME/escape-proof` fails with Operation not permitted; the sibling
/// temp escape `$TMPDIR/../C/…` is denied (no blanket `/private/var/folders`).
///
/// Filters that the starting WP sketch listed on one `(allow …)` line are
/// emitted as **separate** allow rules — SBPL ANDs filters within a single
/// rule, so combining tty/pty regexes with a subpath never matched.
#[cfg(any(test, target_os = "macos"))]
fn render_seatbelt_profile(canonical_roots: &[PathBuf]) -> String {
    let mut out = String::with_capacity(512);
    out.push_str("(version 1)\n");
    out.push_str("(allow default)\n");
    out.push_str("(deny file-write*)\n");
    for root in canonical_roots {
        let Some(s) = root.to_str() else {
            // Non-UTF-8 paths cannot be embedded in an SBPL string literal.
            // Skip; callers that care canonicalize via UTF-8 paths.
            continue;
        };
        out.push_str("(allow file-write* (subpath \"");
        out.push_str(&escape_sbpl_string(s));
        out.push_str("\"))\n");
    }
    // Shell / toolchain device plumbing. Separate rules — not ANDed.
    out.push_str("(allow file-write-data (literal \"/dev/null\"))\n");
    out.push_str("(allow file-write* (regex #\"^/dev/tty\"))\n");
    out.push_str("(allow file-write* (regex #\"^/dev/pty\"))\n");
    // Intentionally NO blanket `(subpath "/private/var/folders")`. Allowed
    // temp writes are the canonicalized `std::env::temp_dir()` root already
    // present in `canonical_roots` (plus the other explicit roots).
    out
}

/// Escape a path for embedding inside an SBPL double-quoted string.
#[cfg(any(test, target_os = "macos"))]
fn escape_sbpl_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            // SBPL has no \n escape that is safer than rejecting; keep raw
            // control chars out of profiles by using a hex-ish fallback.
            c if c.is_control() => {
                for b in c.encode_utf8(&mut [0; 4]).bytes() {
                    out.push_str(&format!("\\x{b:02x}"));
                }
            }
            c => out.push(c),
        }
    }
    out
}

// ── Linux (Landlock launcher) ───────────────────────────────────────────────

#[cfg(target_os = "linux")]
fn apply_linux(
    cmd: &mut Command,
    bin: &Path,
    args: &[impl AsRef<OsStr>],
    roots: &[PathBuf],
) -> Result<(), SandboxError> {
    // Validate the ruleset can be built under the V3 write floor. The actual
    // restrict_self runs in the already-exec'd launcher process (async-signal-
    // safe: no path open / allocation / ruleset work in pre_exec).
    preflight_linux(roots)?;

    let launcher = std::env::current_exe()
        .map_err(|e| SandboxError::Io(format!("current_exe for landlock launcher: {e}")))?;
    let spec = encode_landlock_spec(roots)?;

    *cmd = Command::new(launcher);
    cmd.arg(LANDLOCK_LAUNCHER_ARG);
    cmd.arg(spec);
    cmd.arg("--");
    cmd.arg(bin);
    cmd.args(args.iter().map(AsRef::as_ref));
    Ok(())
}

#[cfg(target_os = "linux")]
fn preflight_linux(roots: &[PathBuf]) -> Result<(), SandboxError> {
    // Build once under HardRequirement (does not restrict the parent). Kernel
    // missing Landlock or below the V3 write floor → Unsupported so the CLI
    // warns once and continues unsandboxed.
    landlock_build_ruleset(roots).map(|_| ())
}

#[cfg(target_os = "linux")]
fn encode_landlock_spec(roots: &[PathBuf]) -> Result<String, SandboxError> {
    serde_json::to_string(roots).map_err(|e| SandboxError::Io(format!("encode spec: {e}")))
}

#[cfg(target_os = "linux")]
fn decode_landlock_spec(s: &str) -> Result<Vec<PathBuf>, String> {
    serde_json::from_str(s).map_err(|e| format!("decode spec: {e}"))
}

/// Minimum Landlock ABI we require: V3 mediates `truncate` / `ftruncate` /
/// `creat` / `O_TRUNC`. Lower ABIs would report successful confinement while
/// still allowing outside-file truncation.
#[cfg(target_os = "linux")]
const LANDLOCK_ABI_FLOOR: landlock::ABI = landlock::ABI::V3;

#[cfg(target_os = "linux")]
fn landlock_build_ruleset(roots: &[PathBuf]) -> Result<landlock::RulesetCreated, SandboxError> {
    use landlock::{
        path_beneath_rules, AccessFs, CompatLevel, Compatible, PathBeneath, PathFd, Ruleset,
        RulesetAttr, RulesetCreatedAttr,
    };

    // Write accesses for the V3 floor (includes Truncate + Refer). Hard
    // requirement: if the running kernel cannot enforce these rights, map to
    // Unsupported rather than partially confining.
    let write = AccessFs::from_write(LANDLOCK_ABI_FLOOR);

    let created = match Ruleset::default()
        .set_compatibility(CompatLevel::HardRequirement)
        .handle_access(write)
    {
        Ok(r) => r,
        Err(_) => return Err(SandboxError::Unsupported),
    };
    let mut ruleset = match created.create() {
        Ok(r) => r,
        Err(_) => return Err(SandboxError::Unsupported),
    };
    ruleset = ruleset
        .add_rules(path_beneath_rules(
            roots.iter().map(PathBuf::as_path),
            write,
        ))
        .map_err(|e| SandboxError::Landlock(format!("add_rules: {e}")))?;

    // Narrow device-node allowances (never all of /dev). Only the write rights
    // applicable to a non-directory file — WriteFile + Truncate.
    let dev_write = AccessFs::WriteFile | AccessFs::Truncate;
    if let Ok(null_fd) = PathFd::new("/dev/null") {
        ruleset = ruleset
            .add_rule(PathBeneath::new(null_fd, dev_write))
            .map_err(|e| SandboxError::Landlock(format!("add_rule /dev/null: {e}")))?;
    }

    Ok(ruleset)
}

#[cfg(target_os = "linux")]
fn landlock_restrict(roots: &[PathBuf]) -> Result<(), SandboxError> {
    use landlock::RulesetStatus;

    let ruleset = landlock_build_ruleset(roots)?;
    let status = ruleset
        .restrict_self()
        .map_err(|e| SandboxError::Landlock(format!("restrict_self: {e}")))?;
    match status.ruleset {
        RulesetStatus::FullyEnforced => Ok(()),
        RulesetStatus::PartiallyEnforced => Err(SandboxError::Unsupported),
        RulesetStatus::NotEnforced => Err(SandboxError::Unsupported),
    }
}

// ── unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static SEQ: AtomicU64 = AtomicU64::new(0);

    fn unique(tag: &str) -> PathBuf {
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "greppy-sandbox-{tag}-{}-{}",
            std::process::id(),
            seq
        ))
    }

    #[test]
    fn escape_sbpl_string_quotes_and_backslashes() {
        #[cfg(any(test, target_os = "macos"))]
        {
            assert_eq!(escape_sbpl_string(r#"foo"bar"#), r#"foo\"bar"#);
            assert_eq!(escape_sbpl_string(r#"a\b"#), r#"a\\b"#);
            assert_eq!(escape_sbpl_string("plain"), "plain");
        }
    }

    #[test]
    fn render_profile_contains_roots_and_devices() {
        #[cfg(target_os = "macos")]
        {
            let roots = vec![
                PathBuf::from("/private/tmp/work"),
                PathBuf::from(r#"/tmp/foo"bar"#),
            ];
            let profile = render_seatbelt_profile(&roots);
            assert!(profile.starts_with("(version 1)\n"), "{profile}");
            assert!(profile.contains("(allow default)\n"), "{profile}");
            assert!(profile.contains("(deny file-write*)\n"), "{profile}");
            assert!(
                profile.contains("(allow file-write* (subpath \"/private/tmp/work\"))"),
                "{profile}"
            );
            assert!(
                profile.contains(r#"(allow file-write* (subpath "/tmp/foo\"bar"))"#),
                "{profile}"
            );
            assert!(
                profile.contains("(allow file-write-data (literal \"/dev/null\"))"),
                "{profile}"
            );
            assert!(
                profile.contains("(allow file-write* (regex #\"^/dev/tty\"))"),
                "{profile}"
            );
            assert!(
                profile.contains("(allow file-write* (regex #\"^/dev/pty\"))"),
                "{profile}"
            );
            // S4: blanket /private/var/folders must NOT appear.
            assert!(
                !profile.contains("/private/var/folders"),
                "profile must not blanket-allow /private/var/folders: {profile}"
            );
            // Device rules must be separate allows (not AND-combined).
            let deny_idx = profile.find("(deny file-write*)").unwrap();
            let null_idx = profile.find("/dev/null").unwrap();
            let tty_idx = profile.find("^/dev/tty").unwrap();
            assert!(deny_idx < null_idx && null_idx < tty_idx);
        }
        #[cfg(not(target_os = "macos"))]
        {
            // Profile rendering is macOS-only; still exercise prepare_writable_roots.
            let p = unique("canon");
            let roots = prepare_writable_roots(&[p.clone()]).unwrap();
            assert_eq!(roots.len(), 1);
            assert!(roots[0].is_absolute());
            assert!(roots[0].is_dir());
            let _ = std::fs::remove_dir_all(&p);
        }
    }

    #[test]
    fn prepare_writable_roots_creates_missing_and_dedups() {
        let base = unique("prep-base");
        // Not created yet — prepare must create_dir_all.
        assert!(!base.exists());
        let canon_via_prep = prepare_writable_roots(std::slice::from_ref(&base)).unwrap();
        assert_eq!(canon_via_prep.len(), 1);
        assert!(base.is_dir());
        let canon = std::fs::canonicalize(&base).unwrap();
        assert_eq!(canon_via_prep[0], canon);

        // Feed both original and canonical — expect one.
        let roots = prepare_writable_roots(&[base.clone(), canon.clone()]).unwrap();
        assert_eq!(roots, vec![canon]);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn prepare_writable_roots_rejects_symlink() {
        let base = unique("prep-sym-base");
        std::fs::create_dir_all(&base).unwrap();
        let target = base.join("target");
        std::fs::create_dir_all(&target).unwrap();
        let link = base.join("link");
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&target, &link).unwrap();
            let err = prepare_writable_roots(std::slice::from_ref(&link)).unwrap_err();
            let msg = err.to_string();
            assert!(
                msg.contains("symlink"),
                "expected symlink rejection, got: {msg}"
            );
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn prepare_writable_roots_rejects_file() {
        let f = unique("prep-file");
        std::fs::write(&f, b"not a dir").unwrap();
        let err = prepare_writable_roots(std::slice::from_ref(&f)).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("not a directory"),
            "expected non-directory rejection, got: {msg}"
        );
        let _ = std::fs::remove_file(&f);
    }

    #[test]
    fn apply_off_sets_bin_and_args() {
        let mut cmd = Command::new("placeholder");
        let bin = PathBuf::from("/bin/echo");
        let args = ["hello", "world"];
        apply(&mut cmd, &bin, &args, &SandboxMode::Off).unwrap();
        assert_eq!(cmd.get_program(), OsStr::new("/bin/echo"));
        let got: Vec<_> = cmd.get_args().collect();
        assert_eq!(got, vec![OsStr::new("hello"), OsStr::new("world")]);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn apply_macos_rewrites_to_sandbox_exec() {
        if !Path::new("/usr/bin/sandbox-exec").exists() {
            return;
        }
        let root = unique("apply-macos");
        std::fs::create_dir_all(&root).unwrap();
        let spec = SandboxSpec {
            writable_roots: vec![root.clone()],
        };
        let mut cmd = Command::new("placeholder");
        let bin = PathBuf::from("/bin/echo");
        apply(&mut cmd, &bin, &["hi"][..], &SandboxMode::Enforce(spec)).unwrap();
        assert_eq!(cmd.get_program(), OsStr::new("/usr/bin/sandbox-exec"));
        let got: Vec<_> = cmd
            .get_args()
            .map(|s| s.to_string_lossy().into_owned())
            .collect();
        assert_eq!(got[0], "-p");
        // got[1] is the profile string
        assert!(got[1].contains("(deny file-write*)"), "profile={}", got[1]);
        // The root itself may live under /private/var/folders (temp_dir).
        // What must not appear is the blanket hierarchy allow.
        assert!(
            !got[1].contains("(subpath \"/private/var/folders\")"),
            "profile must not blanket-allow /private/var/folders: {}",
            got[1]
        );
        assert_eq!(got[2], "/bin/echo");
        assert_eq!(got[3], "hi");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn preflight_accepts_valid_profile() {
        if !Path::new("/usr/bin/sandbox-exec").exists() {
            return;
        }
        let root = unique("preflight");
        // Intentionally do not create — prepare_writable_roots must create it.
        let mode = SandboxMode::Enforce(SandboxSpec {
            writable_roots: vec![root.clone()],
        });
        preflight(&mode).expect("preflight");
        assert!(root.is_dir());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn seatbelt_profile_write_inside_ok_outside_denied() {
        if !Path::new("/usr/bin/sandbox-exec").exists() {
            return;
        }
        let root = unique("seatbelt-io");
        let roots = prepare_writable_roots(std::slice::from_ref(&root)).unwrap();
        let profile = render_seatbelt_profile(&roots);

        // Inside root: ok.
        let inside = root.join("inside.txt");
        let status = Command::new("/usr/bin/sandbox-exec")
            .arg("-p")
            .arg(&profile)
            .arg("/usr/bin/touch")
            .arg(&inside)
            .status()
            .unwrap();
        assert!(status.success(), "inside write should succeed");
        assert!(inside.exists());

        // Outside ($HOME unique dotfile): denied, file absent.
        let home = std::env::var_os("HOME").expect("HOME");
        let escape = PathBuf::from(home).join(format!(
            ".greppy-sandbox-escape-proof-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_file(&escape);
        let status = Command::new("/usr/bin/sandbox-exec")
            .arg("-p")
            .arg(&profile)
            .arg("/usr/bin/touch")
            .arg(&escape)
            .status()
            .unwrap();
        assert!(!status.success(), "outside write must fail");
        assert!(
            !escape.exists(),
            "escape file must not exist after denied touch"
        );
        let _ = std::fs::remove_file(&escape);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// S4 permanent regression: `$TMPDIR/../C/probe` must be DENIED when only
    /// the canonical temp_dir (…/T) root is allowed — no blanket
    /// `/private/var/folders`.
    #[cfg(target_os = "macos")]
    #[test]
    fn seatbelt_tmpdir_sibling_c_escape_denied() {
        if !Path::new("/usr/bin/sandbox-exec").exists() {
            return;
        }
        let tmp = std::env::temp_dir();
        let roots = prepare_writable_roots(std::slice::from_ref(&tmp)).unwrap();
        let profile = render_seatbelt_profile(&roots);
        assert!(
            !profile.contains("(subpath \"/private/var/folders\")"),
            "must not blanket-allow /private/var/folders"
        );

        // Build the sibling-of-T probe: $TMPDIR/../C/greppy-sandbox-escape-…
        let probe_dir = tmp.join("..").join("C");
        // Parent may or may not exist; create only for the probe attempt cleanup
        // path. The sandboxed touch must not succeed either way.
        let _ = std::fs::create_dir_all(&probe_dir);
        let probe = probe_dir.join(format!(
            "greppy-sandbox-escape-c-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_file(&probe);

        let status = Command::new("/usr/bin/sandbox-exec")
            .arg("-p")
            .arg(&profile)
            .arg("/usr/bin/touch")
            .arg(&probe)
            .status()
            .unwrap();
        assert!(
            !status.success(),
            "TMPDIR/../C escape must be denied by seatbelt"
        );
        assert!(
            !probe.exists(),
            "escape probe must not exist after denied touch: {}",
            probe.display()
        );
        // Defensive cleanup.
        let _ = std::fs::remove_file(&probe);
    }

    /// Linux-only: ruleset construction from a temp root must not panic and
    /// must return Unsupported (not Landlock error) when the kernel lacks the
    /// V3 write floor.
    #[cfg(target_os = "linux")]
    #[test]
    fn linux_ruleset_construction_or_unsupported() {
        let root = unique("ll");
        let roots = prepare_writable_roots(&[root.clone()]).unwrap();
        match preflight_linux(&roots) {
            Ok(()) => {
                let _ = landlock_build_ruleset(&roots).unwrap();
            }
            Err(SandboxError::Unsupported) => {}
            Err(e) => panic!("unexpected: {e}"),
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn apply_linux_rewrites_to_landlock_launcher() {
        let root = unique("apply-ll");
        std::fs::create_dir_all(&root).unwrap();
        let spec = SandboxSpec {
            writable_roots: vec![root.clone()],
        };
        // preflight/apply may yield Unsupported on old kernels — still check
        // the rewrite shape when support is present.
        let mut cmd = Command::new("placeholder");
        let bin = PathBuf::from("/bin/echo");
        match apply(&mut cmd, &bin, &["hi"][..], &SandboxMode::Enforce(spec)) {
            Ok(()) => {
                let prog = cmd.get_program().to_string_lossy().into_owned();
                let exe = std::env::current_exe().unwrap();
                assert_eq!(PathBuf::from(&prog), exe);
                let got: Vec<_> = cmd
                    .get_args()
                    .map(|s| s.to_string_lossy().into_owned())
                    .collect();
                assert_eq!(got[0], LANDLOCK_LAUNCHER_ARG);
                // got[1] is the JSON spec
                assert!(got[1].starts_with('['), "spec={}", got[1]);
                assert_eq!(got[2], "--");
                assert_eq!(got[3], "/bin/echo");
                assert_eq!(got[4], "hi");
            }
            Err(SandboxError::Unsupported) => {}
            Err(e) => panic!("unexpected: {e}"),
        }
        let _ = std::fs::remove_dir_all(&root);
    }
}
