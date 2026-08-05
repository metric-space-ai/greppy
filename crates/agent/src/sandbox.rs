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
//! - **Linux** — install a Landlock ruleset in `pre_exec` that handles only
//!   write accesses, scoped to the same roots. If the running kernel has no
//!   Landlock, [`SandboxError::Unsupported`] is returned so the CLI can warn
//!   once and continue unsandboxed.
//! - **other** — [`SandboxError::Unsupported`] under `Enforce`.

use std::ffi::OsStr;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;

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
    /// Kernel/OS cannot enforce (Linux without Landlock; non-macOS/non-Linux).
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
/// applied (seatbelt rewrite on macOS; Landlock `pre_exec` on Linux).
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
/// - macOS `Enforce` → `sandbox-exec` must exist and accept the generated profile
///   (`/usr/bin/true` dry-run).
/// - Linux `Enforce` → Landlock ABI must be present.
/// - other OS `Enforce` → [`SandboxError::Unsupported`].
pub fn preflight(mode: &SandboxMode) -> Result<(), SandboxError> {
    match mode {
        SandboxMode::Off => Ok(()),
        SandboxMode::Enforce(spec) => preflight_enforce(spec),
    }
}

fn apply_enforce(
    cmd: &mut Command,
    bin: &Path,
    args: &[impl AsRef<OsStr>],
    spec: &SandboxSpec,
) -> Result<(), SandboxError> {
    #[cfg(target_os = "macos")]
    {
        apply_macos(cmd, bin, args, spec)
    }
    #[cfg(target_os = "linux")]
    {
        apply_linux(cmd, bin, args, spec)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        let _ = (cmd, bin, args, spec);
        Err(SandboxError::Unsupported)
    }
}

fn preflight_enforce(spec: &SandboxSpec) -> Result<(), SandboxError> {
    #[cfg(target_os = "macos")]
    {
        preflight_macos(spec)
    }
    #[cfg(target_os = "linux")]
    {
        preflight_linux(spec)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        let _ = spec;
        Err(SandboxError::Unsupported)
    }
}

// ── macOS (Seatbelt / sandbox-exec) ─────────────────────────────────────────

#[cfg(target_os = "macos")]
fn apply_macos(
    cmd: &mut Command,
    bin: &Path,
    args: &[impl AsRef<OsStr>],
    spec: &SandboxSpec,
) -> Result<(), SandboxError> {
    let profile = build_seatbelt_profile(spec)?;
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
fn preflight_macos(spec: &SandboxSpec) -> Result<(), SandboxError> {
    if !Path::new("/usr/bin/sandbox-exec").exists() {
        return Err(SandboxError::Profile(
            "/usr/bin/sandbox-exec is missing".into(),
        ));
    }
    let profile = build_seatbelt_profile(spec)?;
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
/// `touch $HOME/escape-proof` fails with Operation not permitted.
///
/// Filters that the starting WP sketch listed on one `(allow …)` line are
/// emitted as **separate** allow rules — SBPL ANDs filters within a single
/// rule, so combining tty/pty regexes with a subpath never matched.
#[cfg(target_os = "macos")]
fn build_seatbelt_profile(spec: &SandboxSpec) -> Result<String, SandboxError> {
    let roots = canonicalize_roots(&spec.writable_roots)?;
    Ok(render_seatbelt_profile(&roots))
}

/// Render the Seatbelt profile string for already-canonicalized roots.
/// Public to unit tests (path escaping / shape) without filesystem access.
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
    // macOS materialises TMPDIR / per-user temps / some pty nodes under here.
    // Keeping it as its own allow matches the WP sketch and unblocks cargo/git
    // when intermediate paths fall outside the caller's explicit temp root.
    out.push_str("(allow file-write* (subpath \"/private/var/folders\"))\n");
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

// ── Linux (Landlock) ────────────────────────────────────────────────────────

#[cfg(target_os = "linux")]
fn apply_linux(
    cmd: &mut Command,
    bin: &Path,
    args: &[impl AsRef<OsStr>],
    spec: &SandboxSpec,
) -> Result<(), SandboxError> {
    use std::os::unix::process::CommandExt;

    // Validate we can build a ruleset now (Unsupported bubbles up). The actual
    // restrict_self runs in the child via pre_exec so it does not confine the
    // agent process itself.
    let roots = landlock_existing_roots(spec)?;
    preflight_linux(spec)?;

    *cmd = Command::new(bin);
    cmd.args(args.iter().map(AsRef::as_ref));

    // pre_exec closure must be 'static-ish — own the roots.
    let roots_for_child = roots;
    unsafe {
        cmd.pre_exec(move || {
            landlock_restrict(&roots_for_child).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::PermissionDenied, e.to_string())
            })
        });
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn preflight_linux(spec: &SandboxSpec) -> Result<(), SandboxError> {
    use landlock::ABI;
    let abi = ABI::new_current();
    if matches!(abi, ABI::Unsupported) {
        return Err(SandboxError::Unsupported);
    }
    // Build once to surface configuration errors early (missing rights, etc.).
    let roots = landlock_existing_roots(spec)?;
    // Don't restrict_self in the parent — just ensure create works.
    landlock_build_ruleset(&roots).map(|_| ())
}

#[cfg(target_os = "linux")]
fn landlock_existing_roots(spec: &SandboxSpec) -> Result<Vec<PathBuf>, SandboxError> {
    // Landlock PathFd requires the path to exist at ruleset-build time. Skip
    // missing roots (e.g. ~/.cargo on a fresh machine) rather than failing the
    // whole sandbox — writes there will simply be denied.
    let mut out = Vec::with_capacity(spec.writable_roots.len());
    for r in &spec.writable_roots {
        match std::fs::canonicalize(r) {
            Ok(c) => out.push(c),
            Err(_) => {
                if r.exists() {
                    out.push(r.clone());
                }
                // else: skip missing
            }
        }
    }
    Ok(out)
}

#[cfg(target_os = "linux")]
fn landlock_build_ruleset(roots: &[PathBuf]) -> Result<landlock::RulesetCreated, SandboxError> {
    use landlock::{
        path_beneath_rules, Access, AccessFs, Ruleset, RulesetAttr, RulesetCreatedAttr, ABI,
    };

    let abi = ABI::new_current();
    if matches!(abi, ABI::Unsupported) {
        return Err(SandboxError::Unsupported);
    }
    // Handle only write accesses so reads remain unrestricted everywhere.
    let write = AccessFs::from_write(abi);
    let ruleset = Ruleset::default()
        .handle_access(write)
        .map_err(|e| SandboxError::Landlock(format!("handle_access: {e}")))?
        .create()
        .map_err(|e| SandboxError::Landlock(format!("create: {e}")))?
        .add_rules(path_beneath_rules(
            roots.iter().map(PathBuf::as_path),
            write,
        ))
        .map_err(|e| SandboxError::Landlock(format!("add_rules: {e}")))?;
    Ok(ruleset)
}

#[cfg(target_os = "linux")]
fn landlock_restrict(roots: &[PathBuf]) -> Result<(), SandboxError> {
    use landlock::RulesetCreatedAttr;
    let ruleset = landlock_build_ruleset(roots)?;
    ruleset
        .restrict_self()
        .map_err(|e| SandboxError::Landlock(format!("restrict_self: {e}")))?;
    Ok(())
}

// ── shared helpers ──────────────────────────────────────────────────────────

/// Canonicalize roots for profile / ruleset use.
///
/// Symlinks matter: on macOS `/tmp` → `/private/tmp` and `temp_dir()` lives
/// under `/var/folders` → `/private/var/folders`. A non-canonical root would
/// silently fail to match Seatbelt `subpath` checks.
fn canonicalize_roots(roots: &[PathBuf]) -> Result<Vec<PathBuf>, SandboxError> {
    let mut out = Vec::with_capacity(roots.len());
    for r in roots {
        match std::fs::canonicalize(r) {
            Ok(c) => {
                if !out.contains(&c) {
                    out.push(c);
                }
            }
            Err(e) => {
                // Root does not exist yet — keep an absolute form so the
                // profile still permits writes once something creates it.
                let abs = if r.is_absolute() {
                    r.clone()
                } else {
                    std::env::current_dir()
                        .map_err(|e| SandboxError::Io(format!("current_dir: {e}")))?
                        .join(r)
                };
                if !out.contains(&abs) {
                    // Record the canonicalize failure only when the path is
                    // truly unusable (non-UTF8 is handled at render time).
                    let _ = e;
                    out.push(abs);
                }
            }
        }
    }
    Ok(out)
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
            assert!(
                profile.contains("(allow file-write* (subpath \"/private/var/folders\"))"),
                "{profile}"
            );
            // Device rules must be separate allows (not AND-combined).
            let deny_idx = profile.find("(deny file-write*)").unwrap();
            let null_idx = profile.find("/dev/null").unwrap();
            let tty_idx = profile.find("^/dev/tty").unwrap();
            let folders_idx = profile.find("/private/var/folders").unwrap();
            assert!(deny_idx < null_idx && null_idx < tty_idx && tty_idx < folders_idx);
        }
        #[cfg(not(target_os = "macos"))]
        {
            // Profile rendering is macOS-only; still exercise canonicalize.
            let p = unique("canon");
            std::fs::create_dir_all(&p).unwrap();
            let roots = canonicalize_roots(&[p.clone()]).unwrap();
            assert_eq!(roots.len(), 1);
            assert!(roots[0].is_absolute());
            let _ = std::fs::remove_dir_all(&p);
        }
    }

    #[test]
    fn canonicalize_roots_resolves_symlinks_and_dedups() {
        let base = unique("canon-base");
        std::fs::create_dir_all(&base).unwrap();
        let canon = std::fs::canonicalize(&base).unwrap();
        // Feed both the original and an already-canonical form — expect one.
        let roots = canonicalize_roots(&[base.clone(), canon.clone()]).unwrap();
        assert_eq!(roots, vec![canon]);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn canonicalize_missing_root_kept_absolute() {
        let missing = unique("does-not-exist-yet");
        assert!(!missing.exists());
        let roots = canonicalize_roots(std::slice::from_ref(&missing)).unwrap();
        assert_eq!(roots.len(), 1);
        assert!(roots[0].is_absolute());
        assert!(roots[0].ends_with(missing.file_name().unwrap()));
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
        std::fs::create_dir_all(&root).unwrap();
        let mode = SandboxMode::Enforce(SandboxSpec {
            writable_roots: vec![root.clone()],
        });
        preflight(&mode).expect("preflight");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn seatbelt_profile_write_inside_ok_outside_denied() {
        if !Path::new("/usr/bin/sandbox-exec").exists() {
            return;
        }
        let root = unique("seatbelt-io");
        std::fs::create_dir_all(&root).unwrap();
        let profile = build_seatbelt_profile(&SandboxSpec {
            writable_roots: vec![root.clone()],
        })
        .unwrap();

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

    /// Linux-only: ruleset construction from a temp root must not panic and
    /// must return Unsupported (not Landlock error) when the kernel lacks it.
    #[cfg(target_os = "linux")]
    #[test]
    fn linux_ruleset_construction_or_unsupported() {
        let root = unique("ll");
        std::fs::create_dir_all(&root).unwrap();
        let spec = SandboxSpec {
            writable_roots: vec![root.clone()],
        };
        match preflight_linux(&spec) {
            Ok(()) => {
                let roots = landlock_existing_roots(&spec).unwrap();
                let _ = landlock_build_ruleset(&roots).unwrap();
            }
            Err(SandboxError::Unsupported) => {}
            Err(e) => panic!("unexpected: {e}"),
        }
        let _ = std::fs::remove_dir_all(&root);
    }
}
