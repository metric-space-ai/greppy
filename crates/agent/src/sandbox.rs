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
//! Writable roots are prepared **exactly once** per agent run (CLI preflight),
//! producing a canonical [`SandboxSpec`]. Per-tool `apply` trusts those roots
//! verbatim — no `exists` / `create_dir_all` / `canonicalize` / re-validation
//! that could re-authorize an attacker-swapped symlink.

use std::ffi::OsStr;
#[cfg(target_os = "linux")]
use std::ffi::OsString;
use std::fmt;
use std::path::{Component, Path, PathBuf};
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
///
/// **Invariant:** under [`SandboxMode::Enforce`], `writable_roots` must already
/// be the output of [`prepare_writable_roots`] (absolute canonical directories,
/// every ancestor component validated, no symlinks). Per-tool [`apply`] uses
/// these paths verbatim and never re-resolves them. Callers that build a spec
/// from raw paths must run [`resolve_enforce_spec`] (or equivalent) once before
/// the agent loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxSpec {
    /// Absolute canonical directories the child may write under.
    /// Pre-resolved; trusted thereafter (see type invariant above).
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

/// Probe whether a pre-resolved `mode` can be enforced on this host.
///
/// Expects `Enforce` specs to already carry canonical roots (see
/// [`SandboxSpec`] invariant). Prefer [`resolve_enforce_spec`] which prepares
/// roots and probes in one shot.
///
/// - `Off` → always ok.
/// - macOS `Enforce` → `sandbox-exec` must exist and accept the generated
///   profile (`/usr/bin/true` dry-run). Roots are used verbatim.
/// - Linux `Enforce` → Landlock must support the V3 write rights floor
///   (including `Truncate`) under hard-requirement compatibility.
/// - other OS `Enforce` → [`SandboxError::Unsupported`].
pub fn preflight(mode: &SandboxMode) -> Result<(), SandboxError> {
    match mode {
        SandboxMode::Off => Ok(()),
        SandboxMode::Enforce(spec) => preflight_enforce(spec),
    }
}

/// Prepare writable roots once and probe platform enforcement.
///
/// This is the one-shot entry point the CLI uses before the agent loop:
/// 1. [`prepare_writable_roots`] — create missing components, reject any
///    existing symlink ancestor, canonicalize, dedup.
/// 2. Platform preflight against those fixed canonical roots.
///
/// On success returns an [`SandboxMode::Enforce`] whose roots are trusted for
/// the rest of the run (never re-resolved by [`apply`]).
pub fn resolve_enforce_spec(raw_roots: &[PathBuf]) -> Result<SandboxMode, SandboxError> {
    let roots = prepare_writable_roots(raw_roots)?;
    let mode = SandboxMode::Enforce(SandboxSpec {
        writable_roots: roots,
    });
    preflight_enforce(match &mode {
        SandboxMode::Enforce(spec) => spec,
        SandboxMode::Off => unreachable!("just constructed Enforce"),
    })?;
    Ok(mode)
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
    // Spec roots are pre-resolved by resolve_enforce_spec / prepare_writable_roots
    // exactly once per agent run. Use them verbatim — no exists/create/canonicalize
    // that could re-authorize an attacker-swapped symlink between tool calls.
    debug_assert!(
        spec.writable_roots
            .iter()
            .all(|r| r.is_absolute() && !r.as_os_str().is_empty()),
        "SandboxSpec.writable_roots must be pre-resolved absolute paths"
    );
    let roots = &spec.writable_roots;
    #[cfg(target_os = "macos")]
    {
        apply_macos(cmd, bin, args, roots)
    }
    #[cfg(target_os = "linux")]
    {
        apply_linux(cmd, bin, args, roots)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        let _ = (cmd, bin, args, roots);
        Err(SandboxError::Unsupported)
    }
}

fn preflight_enforce(spec: &SandboxSpec) -> Result<(), SandboxError> {
    // Roots are already prepared; only probe the platform backend.
    let roots = &spec.writable_roots;
    #[cfg(target_os = "macos")]
    {
        preflight_macos(roots)
    }
    #[cfg(target_os = "linux")]
    {
        preflight_linux(roots)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        let _ = roots;
        Err(SandboxError::Unsupported)
    }
}

// ── root preparation (trusted parent, once per run) ─────────────────────────

/// Create, fully validate, and canonicalize every intended writable root.
///
/// Belt-and-braces against symlink-ancestor escape:
/// 1. Make the path absolute (join `current_dir` when relative).
/// 2. `create_dir_all` so missing trailing components exist **before** the walk.
/// 3. Walk every component with `symlink_metadata` and **reject** if any existing
///    component is a symlink (names the offending component) or if the final
///    path is not a directory.
/// 4. `canonicalize` the validated path (now free of symlinks; result is the
///    stable identity used for the rest of the run).
///
/// System symlink prefixes that a non-root user cannot replace (e.g. macOS
/// `/var` → `/private/var`, `/tmp` → `/private/tmp`) are expanded before the
/// reject-walk so legitimate host paths still work; any symlink whose parent
/// directory is writable by the current user is refused.
///
/// Called exactly once per agent run (via [`resolve_enforce_spec`]); never from
/// per-tool [`apply`].
pub fn prepare_writable_roots(roots: &[PathBuf]) -> Result<Vec<PathBuf>, SandboxError> {
    let mut out = Vec::with_capacity(roots.len());
    for r in roots {
        let abs = make_absolute(r)?;
        // Expand immutable system-symlink prefixes (e.g. /var → /private/var)
        // first, so the reject-walk below only sees user-controllable components.
        let abs = expand_immutable_symlink_prefixes(&abs)?;
        // Create missing components first, then re-walk to validate. create_dir_all
        // follows intermediate symlinks for creation; the subsequent walk rejects
        // any remaining symlink component (attacker-planted under a writable parent).
        // If the path already exists as a non-directory, create_dir_all errors with
        // AlreadyExists / "File exists" — fall through so the type check below
        // reports "not a directory" cleanly.
        match std::fs::create_dir_all(&abs) {
            Ok(()) => {}
            Err(e)
                if e.kind() == std::io::ErrorKind::AlreadyExists
                    || e.raw_os_error() == Some(17) /* EEXIST */ =>
            {
                // Leave for the symlink/type checks below.
            }
            Err(e) => {
                return Err(SandboxError::Io(format!(
                    "cannot create writable root {}: {e}",
                    abs.display()
                )));
            }
        }
        reject_symlink_components(&abs)?;
        let meta = std::fs::symlink_metadata(&abs).map_err(|e| {
            SandboxError::Io(format!("cannot stat writable root {}: {e}", abs.display()))
        })?;
        if meta.file_type().is_symlink() {
            return Err(SandboxError::Io(format!(
                "writable root is a symlink (refusing): {}",
                abs.display()
            )));
        }
        if !meta.is_dir() {
            return Err(SandboxError::Io(format!(
                "writable root is not a directory: {}",
                abs.display()
            )));
        }
        // After the walk rejected every remaining symlink component, canonicalize
        // is a pure identity transform for `.` / `..` / mount points — it cannot
        // jump outside via a symlink we already forbade.
        let c = std::fs::canonicalize(&abs).map_err(|e| {
            SandboxError::Io(format!(
                "cannot canonicalize writable root {}: {e}",
                abs.display()
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

/// Make `path` absolute without resolving symlinks.
fn make_absolute(path: &Path) -> Result<PathBuf, SandboxError> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    let cwd = std::env::current_dir().map_err(|e| {
        SandboxError::Io(format!(
            "cannot resolve current directory for relative root {}: {e}",
            path.display()
        ))
    })?;
    Ok(cwd.join(path))
}

/// Expand leading symlink components whose parent dir the current user cannot
/// write (system layout links like `/var` → `/private/var`). Stops at the first
/// missing component or the first symlink under a user-writable parent (those
/// are left for [`reject_symlink_components`] to refuse).
fn expand_immutable_symlink_prefixes(path: &Path) -> Result<PathBuf, SandboxError> {
    // Iteratively resolve: walk components; when a symlink sits under a
    // non-writable parent, replace the accumulated path with its canonical
    // form and continue with the unprocessed suffix.
    let components: Vec<Component<'_>> = path.components().collect();
    let mut out = PathBuf::new();
    let mut i = 0;
    while i < components.len() {
        match components[i] {
            Component::Prefix(p) => {
                out.push(p.as_os_str());
                i += 1;
                continue;
            }
            Component::RootDir => {
                out.push(components[i].as_os_str());
                i += 1;
            }
            Component::CurDir => {
                i += 1;
                continue;
            }
            Component::ParentDir => {
                let _ = out.pop();
                i += 1;
                continue;
            }
            Component::Normal(s) => {
                out.push(s);
                i += 1;
            }
        }
        match std::fs::symlink_metadata(&out) {
            Ok(meta) if meta.file_type().is_symlink() => {
                let parent = out.parent().unwrap_or_else(|| Path::new("/"));
                if dir_is_writable_by_user(parent) {
                    // User-controllable symlink: leave as-is for reject walk.
                    // Rejoin the remaining components onto `out` and return.
                    while i < components.len() {
                        out.push(components[i].as_os_str());
                        i += 1;
                    }
                    return Ok(out);
                }
                // Immutable system symlink: resolve and keep walking.
                let resolved = std::fs::canonicalize(&out).map_err(|e| {
                    SandboxError::Io(format!(
                        "cannot resolve system symlink prefix {}: {e}",
                        out.display()
                    ))
                })?;
                out = resolved;
            }
            Ok(_) => {
                // Real directory/file — keep walking.
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Remaining suffix will be created by create_dir_all.
                while i < components.len() {
                    out.push(components[i].as_os_str());
                    i += 1;
                }
                return Ok(out);
            }
            Err(e) => {
                return Err(SandboxError::Io(format!(
                    "cannot stat path component {}: {e}",
                    out.display()
                )));
            }
        }
    }
    Ok(out)
}

/// True when the current process can create entries in `dir` (i.e. an attacker
/// running as this user could plant a symlink there).
fn dir_is_writable_by_user(dir: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;
        // access(W_OK) reflects the effective credentials — exactly the
        // capability an in-sandbox tool (same uid) would have to swap a name.
        let Ok(c) = CString::new(dir.as_os_str().as_bytes()) else {
            return true; // fail closed: treat weird paths as writable
        };
        // SAFETY: c is a valid NUL-terminated path; access is a pure query.
        let rc = unsafe {
            libc_access(c.as_ptr(), 2 /* W_OK */)
        };
        rc == 0
    }
    #[cfg(not(unix))]
    {
        let _ = dir;
        // Non-unix: we don't have a seatbelt/landlock backend either; treat as
        // writable so any symlink component is rejected.
        true
    }
}

/// Thin libc access(2) wrapper (avoid a libc crate dep on non-linux targets).
#[cfg(unix)]
unsafe fn libc_access(path: *const std::os::raw::c_char, mode: i32) -> i32 {
    // libc is not a direct dep of greppy-agent on macOS; declare the symbol.
    unsafe extern "C" {
        fn access(path: *const std::os::raw::c_char, mode: i32) -> i32;
    }
    unsafe { access(path, mode) }
}

/// Walk every existing component of `path` with `symlink_metadata` and reject if
/// any is a symlink. Also rejects empty paths. Lexical `..` is applied without
/// resolving across links.
///
/// Naming: the error mentions the first offending component path so operators
/// can see which ancestor was swapped for a symlink.
fn reject_symlink_components(path: &Path) -> Result<(), SandboxError> {
    let mut cur = PathBuf::new();
    let mut saw_root = false;
    for comp in path.components() {
        match comp {
            Component::Prefix(p) => {
                cur.push(p.as_os_str());
                continue;
            }
            Component::RootDir => {
                cur.push(comp.as_os_str());
                saw_root = true;
            }
            Component::CurDir => continue,
            Component::ParentDir => {
                // Lexical parent: pop last normal component if present.
                if !cur.pop() && saw_root {
                    // Staying at root is fine.
                }
                continue;
            }
            Component::Normal(s) => {
                cur.push(s);
            }
        }
        // Stat the accumulated path without following the final component.
        match std::fs::symlink_metadata(&cur) {
            Ok(meta) => {
                if meta.file_type().is_symlink() {
                    return Err(SandboxError::Io(format!(
                        "writable root has a symlink component (refusing): {}",
                        cur.display()
                    )));
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // create_dir_all should have made the full path; a missing
                // intermediate after that is a race/TOCTOU — fail closed.
                return Err(SandboxError::Io(format!(
                    "writable root component missing after create: {}: {e}",
                    cur.display()
                )));
            }
            Err(e) => {
                return Err(SandboxError::Io(format!(
                    "cannot stat writable root component {}: {e}",
                    cur.display()
                )));
            }
        }
    }
    if cur.as_os_str().is_empty() {
        return Err(SandboxError::Io(
            "writable root path is empty after normalization".into(),
        ));
    }
    Ok(())
}

/// Re-validate roots that the parent already prepared (launcher path).
///
/// Unlike [`prepare_writable_roots`], this does **not** create directories and
/// does **not** re-canonicalize into a new identity: it only checks that each
/// pre-resolved absolute path still exists as a non-symlink directory, then
/// returns the input paths unchanged. That keeps the Landlock ruleset pointed
/// at the original canonical roots even if an attacker swapped names on disk.
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
        // Keep the pre-resolved path verbatim — do not re-canonicalize.
        if !out.contains(r) {
            out.push(r.clone());
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
///
/// cfg: macOS production + unit tests (so Linux `cfg(test)` builds still
/// compile the pure string renderer / escape helper without dead_code noise
/// on the Linux target clippy gate, which is `not(test)`-equivalent for
/// cross-compile `--target` and must not see this symbol at all).
#[cfg(target_os = "macos")]
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
#[cfg(target_os = "macos")]
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

    // handle_access under HardRequirement: only CompatError variants (ABI /
    // access-right insufficiency) are possible. Map those to Unsupported so
    // the CLI can warn-once and continue; never treat them as Landlock.
    let created = match Ruleset::default()
        .set_compatibility(CompatLevel::HardRequirement)
        .handle_access(write)
    {
        Ok(r) => r,
        Err(e) => return Err(classify_ruleset_error(e)),
    };
    // create() can fail with:
    //   - CreateRulesetError::MissingHandledAccess — HardRequirement + kernel
    //     too old (compat state Dummy/No) → Unsupported
    //   - CreateRulesetError::CreateRulesetCall { source } —
    //       ENOSYS / EOPNOTSUPP → Unsupported (no Landlock)
    //       anything else (ENOMEM, EMFILE, EPERM, …) → Landlock (fail closed)
    let mut ruleset = match created.create() {
        Ok(r) => r,
        Err(e) => return Err(classify_ruleset_error(e)),
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

/// Map a `landlock::RulesetError` to our sandbox error taxonomy.
///
/// **Unsupported** (CLI warns once, continues unsandboxed) — true
/// compatibility / ABI insufficiency only:
/// - `HandleAccesses` / `Scope` / `RestrictSelfFlags` (compat-level rejection)
/// - `CreateRuleset::MissingHandledAccess` (HardRequirement + no usable ABI)
/// - `CreateRuleset::CreateRulesetCall` whose errno is `ENOSYS` or `EOPNOTSUPP`
/// - `RestrictSelf` whose errno is `ENOSYS` or `EOPNOTSUPP`
///
/// **Landlock** (fail closed, abort the run) — everything else, including
/// resource failures (`ENOMEM`, `EMFILE`, `ENFILE`), permission failures
/// (`EPERM`), and other `CreateRulesetCall` / `RestrictSelf` / `AddRules`
/// syscall errors. A ruleset that cannot be installed must never silently
/// fall back to unrestricted execution.
#[cfg(target_os = "linux")]
fn classify_ruleset_error(err: landlock::RulesetError) -> SandboxError {
    use landlock::{CreateRulesetError, RestrictSelfError, RulesetError};

    match &err {
        RulesetError::HandleAccesses(_) => SandboxError::Unsupported,
        RulesetError::Scope(_) => SandboxError::Unsupported,
        RulesetError::RestrictSelfFlags(_) => SandboxError::Unsupported,
        RulesetError::CreateRuleset(CreateRulesetError::MissingHandledAccess) => {
            SandboxError::Unsupported
        }
        RulesetError::CreateRuleset(CreateRulesetError::CreateRulesetCall { source, .. }) => {
            if is_compat_io_error(source) {
                SandboxError::Unsupported
            } else {
                SandboxError::Landlock(format!("create_ruleset: {err}"))
            }
        }
        RulesetError::RestrictSelf(RestrictSelfError::RestrictSelfCall { source, .. })
        | RulesetError::RestrictSelf(RestrictSelfError::SetNoNewPrivsCall { source, .. }) => {
            if is_compat_io_error(source) {
                SandboxError::Unsupported
            } else {
                SandboxError::Landlock(format!("restrict_self: {err}"))
            }
        }
        RulesetError::AddRules(_) => SandboxError::Landlock(format!("add_rules: {err}")),
        // Non-exhaustive: unknown future variants fail closed.
        _ => SandboxError::Landlock(format!("ruleset: {err}")),
    }
}

/// True when an I/O error from a Landlock syscall indicates "kernel does not
/// support this" rather than a resource/permission failure.
///
/// `ENOSYS` — syscall missing entirely; `EOPNOTSUPP` — Landlock compiled out
/// or disabled. Anything else (ENOMEM, EMFILE, EPERM, …) is fail-closed.
#[cfg(target_os = "linux")]
fn is_compat_io_error(source: &std::io::Error) -> bool {
    matches!(
        source.raw_os_error(),
        Some(libc_errno::ENOSYS) | Some(libc_errno::EOPNOTSUPP)
    ) || matches!(source.kind(), std::io::ErrorKind::Unsupported)
}

/// libc errno constants used by [`is_compat_io_error`] (declared locally so
/// greppy-agent does not need a direct libc dep on every target). Linux UAPI
/// values: stable across x86_64 / aarch64 / riscv64.
#[cfg(target_os = "linux")]
mod libc_errno {
    pub const ENOSYS: i32 = 38;
    pub const EOPNOTSUPP: i32 = 95;
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

    /// Prepare once (as the CLI does) so Apply sees pre-resolved roots.
    fn resolved_spec(raw: &[PathBuf]) -> SandboxSpec {
        let roots = prepare_writable_roots(raw).expect("prepare roots");
        SandboxSpec {
            writable_roots: roots,
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn escape_sbpl_string_quotes_and_backslashes() {
        assert_eq!(escape_sbpl_string(r#"foo"bar"#), r#"foo\"bar"#);
        assert_eq!(escape_sbpl_string(r#"a\b"#), r#"a\\b"#);
        assert_eq!(escape_sbpl_string("plain"), "plain");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn render_profile_contains_roots_and_devices() {
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
    #[test]
    fn prepare_writable_roots_on_non_macos() {
        // Profile rendering is macOS-only; still exercise prepare_writable_roots.
        let p = unique("canon");
        let roots = prepare_writable_roots(std::slice::from_ref(&p)).unwrap();
        assert_eq!(roots.len(), 1);
        assert!(roots[0].is_absolute());
        assert!(roots[0].is_dir());
        let _ = std::fs::remove_dir_all(&p);
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

    /// T1(i): a writable root whose *ancestor* is a user-planted symlink must
    /// be rejected (not only a final-component symlink).
    #[test]
    fn prepare_writable_roots_rejects_ancestor_symlink() {
        let base = unique("prep-anc-sym");
        std::fs::create_dir_all(&base).unwrap();
        let real_dir = base.join("real");
        std::fs::create_dir_all(&real_dir).unwrap();
        let link = base.join("link");
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&real_dir, &link).unwrap();
            // Root path walks through the symlink ancestor: …/link/child
            let nested = link.join("child");
            // create_dir_all would succeed via the symlink; the reject-walk must
            // still catch the intermediate symlink component.
            let err = prepare_writable_roots(std::slice::from_ref(&nested)).unwrap_err();
            let msg = err.to_string();
            assert!(
                msg.contains("symlink"),
                "expected ancestor-symlink rejection, got: {msg}"
            );
            // Offending component should be named.
            assert!(
                msg.contains(link.file_name().unwrap().to_str().unwrap()) || msg.contains("link"),
                "error should name the symlink component: {msg}"
            );
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    /// T1(ii): once roots are resolved into a SandboxSpec, swapping a directory
    /// for a symlink on disk must NOT change the roots used by a later apply —
    /// the profile/spec still names the original canonical path.
    #[cfg(target_os = "macos")]
    #[test]
    fn apply_uses_pre_resolved_roots_after_symlink_swap() {
        if !Path::new("/usr/bin/sandbox-exec").exists() {
            return;
        }
        let base = unique("swap-base");
        std::fs::create_dir_all(&base).unwrap();
        let root = base.join("root");
        std::fs::create_dir_all(&root).unwrap();
        let outside = base.join("outside");
        std::fs::create_dir_all(&outside).unwrap();

        let mode = resolve_enforce_spec(std::slice::from_ref(&root)).expect("resolve");
        let SandboxMode::Enforce(spec) = &mode else {
            panic!("expected Enforce");
        };
        let original_canon = spec.writable_roots[0].clone();
        assert!(
            original_canon.ends_with("root")
                || original_canon == std::fs::canonicalize(&root).unwrap()
        );

        // Attacker: mv root aside, replace with symlink to outside.
        let aside = base.join("root-aside");
        std::fs::rename(&root, &aside).unwrap();
        std::os::unix::fs::symlink(&outside, &root).unwrap();

        // Later tool spawn must still name the original canonical path in the
        // seatbelt profile — not the outside directory the symlink now points to.
        let mut cmd = Command::new("placeholder");
        let bin = PathBuf::from("/bin/echo");
        apply(&mut cmd, &bin, &["hi"][..], &mode).expect("apply");
        assert_eq!(cmd.get_program(), OsStr::new("/usr/bin/sandbox-exec"));
        let got: Vec<_> = cmd
            .get_args()
            .map(|s| s.to_string_lossy().into_owned())
            .collect();
        assert_eq!(got[0], "-p");
        let profile = &got[1];
        let want = format!(
            "(allow file-write* (subpath \"{}\"))",
            original_canon.to_str().unwrap()
        );
        assert!(
            profile.contains(&want),
            "profile must still name original canonical root {original_canon:?};\nprofile={profile}"
        );
        // Must NOT have re-resolved through the attacker symlink to `outside`.
        let outside_canon = std::fs::canonicalize(&outside).unwrap();
        let evil = format!(
            "(allow file-write* (subpath \"{}\"))",
            outside_canon.to_str().unwrap()
        );
        assert!(
            !profile.contains(&evil),
            "profile must not authorize the symlink target {outside_canon:?};\nprofile={profile}"
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    /// Same T1(ii) property on Linux: the landlock launcher JSON spec must keep
    /// the original canonical root after a post-resolve symlink swap.
    #[cfg(target_os = "linux")]
    #[test]
    fn apply_linux_uses_pre_resolved_roots_after_symlink_swap() {
        let base = unique("swap-ll");
        std::fs::create_dir_all(&base).unwrap();
        let root = base.join("root");
        std::fs::create_dir_all(&root).unwrap();
        let outside = base.join("outside");
        std::fs::create_dir_all(&outside).unwrap();

        // Resolve once (may be Unsupported on old kernels — then nothing to check).
        let mode = match resolve_enforce_spec(std::slice::from_ref(&root)) {
            Ok(m) => m,
            Err(SandboxError::Unsupported) => {
                let _ = std::fs::remove_dir_all(&base);
                return;
            }
            Err(e) => panic!("resolve: {e}"),
        };
        let SandboxMode::Enforce(spec) = &mode else {
            panic!("expected Enforce");
        };
        let original = spec.writable_roots[0].clone();

        let aside = base.join("root-aside");
        std::fs::rename(&root, &aside).unwrap();
        std::os::unix::fs::symlink(&outside, &root).unwrap();

        let mut cmd = Command::new("placeholder");
        let bin = PathBuf::from("/bin/echo");
        match apply(&mut cmd, &bin, &["hi"][..], &mode) {
            Ok(()) => {
                let got: Vec<_> = cmd
                    .get_args()
                    .map(|s| s.to_string_lossy().into_owned())
                    .collect();
                assert_eq!(got[0], LANDLOCK_LAUNCHER_ARG);
                let spec_json = &got[1];
                assert!(
                    spec_json.contains(original.to_str().unwrap()),
                    "launcher spec must keep original canonical root {original:?}; got {spec_json}"
                );
                let outside_canon = std::fs::canonicalize(&outside).unwrap();
                // Only fail if outside leaked in as a distinct path.
                if outside_canon != original {
                    assert!(
                        !spec_json.contains(outside_canon.to_str().unwrap()),
                        "launcher spec must not re-resolve to symlink target {outside_canon:?}; got {spec_json}"
                    );
                }
            }
            Err(SandboxError::Unsupported) => {}
            Err(e) => panic!("unexpected: {e}"),
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
    fn prepare_writable_roots_accepts_temp_dir() {
        // System symlink prefixes (macOS /var → /private/var) must expand, not
        // reject, so the platform temp dir remains a valid writable root.
        let tmp = std::env::temp_dir();
        let roots = prepare_writable_roots(std::slice::from_ref(&tmp)).unwrap();
        assert_eq!(roots.len(), 1);
        assert!(roots[0].is_absolute());
        assert!(roots[0].is_dir());
        // No symlink components remain in the canonical result.
        assert_eq!(roots[0], std::fs::canonicalize(&tmp).unwrap());
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
        let spec = resolved_spec(std::slice::from_ref(&root));
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
        // Intentionally do not create — resolve_enforce_spec must create it.
        let mode = resolve_enforce_spec(std::slice::from_ref(&root)).expect("resolve");
        assert!(matches!(mode, SandboxMode::Enforce(_)));
        assert!(root.is_dir() || std::fs::canonicalize(&root).is_ok());
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
        let inside = roots[0].join("inside.txt");
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
        let roots = prepare_writable_roots(std::slice::from_ref(&root)).unwrap();
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
        let spec = resolved_spec(std::slice::from_ref(&root));
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

    /// T2: classify_ruleset_error maps true compat failures to Unsupported and
    /// non-compat (resource) failures to Landlock. Pure errno helper is tested
    /// with synthesized io::Error values; constructible RulesetError variants
    /// exercise the full match.
    #[cfg(target_os = "linux")]
    #[test]
    fn classify_ruleset_error_compat_vs_fail_closed() {
        use landlock::{
            AccessError, AccessFs, CompatError, CreateRulesetError, HandleAccessError,
            HandleAccessesError, RulesetError,
        };

        // Compat: MissingHandledAccess (HardRequirement + no usable ABI).
        let e = RulesetError::CreateRuleset(CreateRulesetError::MissingHandledAccess);
        assert!(
            matches!(classify_ruleset_error(e), SandboxError::Unsupported),
            "MissingHandledAccess must be Unsupported"
        );

        // Compat: HandleAccesses / AccessError::Empty.
        let e = RulesetError::HandleAccesses(HandleAccessesError::Fs(HandleAccessError::Compat(
            CompatError::Access(AccessError::<AccessFs>::Empty),
        )));
        assert!(
            matches!(classify_ruleset_error(e), SandboxError::Unsupported),
            "HandleAccesses compat must be Unsupported"
        );

        // Non-compat errno path: pure helper.
        let enomem = std::io::Error::from_raw_os_error(12); // ENOMEM
        assert!(
            !is_compat_io_error(&enomem),
            "ENOMEM must NOT be treated as compat"
        );
        let enosys = std::io::Error::from_raw_os_error(libc_errno::ENOSYS);
        assert!(
            is_compat_io_error(&enosys),
            "ENOSYS must be treated as compat"
        );
        let eopnotsupp = std::io::Error::from_raw_os_error(libc_errno::EOPNOTSUPP);
        assert!(
            is_compat_io_error(&eopnotsupp),
            "EOPNOTSUPP must be treated as compat"
        );

        // Synthesize CreateRulesetCall via the public non_exhaustive… cannot.
        // Instead assert the mapping branch by feeding classify through a
        // hand-built path using the helper's contract above + the match on
        // AddRules (always Landlock).
        // AddRules is always fail-closed (we cannot construct the inner easily
        // without a real ruleset, so we only re-check the documented helper).
        let eperm = std::io::Error::from_raw_os_error(1); // EPERM
        assert!(!is_compat_io_error(&eperm), "EPERM must be fail-closed");
    }
}
