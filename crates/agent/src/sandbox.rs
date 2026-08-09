//! Write-confinement sandbox for agent tool subprocesses.
//!
//! Both the `greppy` and `bash` tools spawn a child process. When
//! [`SandboxMode::Enforce`] is active, that child may **write** only under an
//! explicit allowlist of roots (the run worktree, temp dir, greppy data root,
//! `~/.cargo`, and the platform user cache). **Reads stay unrestricted** so
//! builds can reach system headers and package registries; **network stays
//! open** in this iteration (needed for `cargo fetch`).
//!
//! Platform backends:
//! - **macOS** — rewrite the invocation as
//!   `/usr/bin/sandbox-exec -p <seatbelt-profile> <bin> <args…>`. Profile
//!   generation is fail-closed: a rejected profile is [`SandboxError`].
//! - **Linux** — rewrite the invocation as a **launcher mode**:
//!   `<current_exe> __agent-sandbox-landlock <fd-spec> -- <bin> <args…>`.
//!   The **trusted parent** opens each already-validated root as a directory
//!   FD (`O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC`), clears `FD_CLOEXEC` in a
//!   minimal `pre_exec` hook, and passes the raw FD numbers in the launcher
//!   JSON. The launcher never re-resolves pathnames: it builds
//!   `PathBeneath` rules directly from the inherited FDs, applies Landlock
//!   (ABI ≥ V3 write floor, including `Truncate`), then `exec`s the real
//!   command. If the kernel cannot fully enforce the requested rights,
//!   [`SandboxError::Unsupported`] is returned so the CLI can warn once and
//!   continue unsandboxed.
//! - **other** — [`SandboxError::Unsupported`] under `Enforce`.
//!
//! Writable roots are prepared **exactly once** per agent run (CLI preflight),
//! producing a canonical [`SandboxSpec`]. Per-tool `apply` trusts those roots
//! verbatim — no `exists` / `create_dir_all` / `canonicalize` / re-validation
//! that could re-authorize an attacker-swapped symlink. On Linux the parent
//! further pins each root to an open directory FD **at preparation time** (not
//! per tool call) so a post-validation ancestor→symlink swap cannot redirect a
//! later pathname open into an unauthorized target; every spawn reuses (dups)
//! those same held descriptors.

use std::ffi::OsStr;
#[cfg(target_os = "linux")]
use std::ffi::OsString;
use std::fmt;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

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

/// Canonical roots prepared once per agent run, plus (on Linux) the directory
/// FDs opened at preparation time.
///
/// **Equality compares paths only.** FD identity is deliberately not part of
/// equality: two preparations of the same roots compare equal even when their
/// held descriptors differ.
#[derive(Debug)]
struct PreparedRoots {
    paths: Vec<PathBuf>,
    #[cfg(target_os = "linux")]
    root_fds: Vec<OwnedDirFd>,
    #[cfg(target_os = "linux")]
    dev_null_fd: Option<OwnedDirFd>,
}

impl PartialEq for PreparedRoots {
    fn eq(&self, other: &Self) -> bool {
        self.paths == other.paths
    }
}

impl Eq for PreparedRoots {}

impl PreparedRoots {
    /// Open every prepared root (and `/dev/null` on Linux) exactly once.
    ///
    /// On non-Linux hosts this only stores the paths — Seatbelt resolves at
    /// access time and does not need held FDs.
    fn new(roots: &[PathBuf]) -> Result<Self, SandboxError> {
        #[cfg(target_os = "linux")]
        {
            let root_fds = open_trusted_root_fds(roots)?;
            let dev_null_fd = open_trusted_dev_null_fd().ok();
            Ok(Self {
                paths: roots.to_vec(),
                root_fds,
                dev_null_fd,
            })
        }
        #[cfg(not(target_os = "linux"))]
        {
            Ok(Self {
                paths: roots.to_vec(),
            })
        }
    }
}

/// Writable-root allowlist for a sandboxed tool subprocess.
///
/// **Invariant:** under [`SandboxMode::Enforce`], `writable_roots` must already
/// be the output of [`prepare_writable_roots`] (absolute canonical directories,
/// every ancestor component validated, no symlinks). Per-tool [`apply`] uses
/// these paths verbatim and never re-resolves them. On Linux the companion
/// [`PreparedRoots`] FDs were opened at the same preparation step and are
/// duplicated into each child — no pathname is opened again after prepare.
/// Callers that build a spec from raw paths must run [`resolve_enforce_spec`]
/// (or [`SandboxSpec::from_prepared_roots`]) once before the agent loop.
///
/// **Equality / hashing note:** `PartialEq`/`Eq` compare `writable_roots`
/// only. Held FD identity is not part of equality (see `PreparedRoots`).
#[derive(Debug, Clone)]
pub struct SandboxSpec {
    /// Absolute canonical directories the child may write under.
    /// Pre-resolved; trusted thereafter (see type invariant above).
    pub writable_roots: Vec<PathBuf>,
    /// Paths + (Linux) directory FDs opened once at preparation. Shared across
    /// clones of this spec for the whole agent run.
    prepared: Arc<PreparedRoots>,
}

impl PartialEq for SandboxSpec {
    fn eq(&self, other: &Self) -> bool {
        // Path equality only; FD identity is not part of equality.
        self.writable_roots == other.writable_roots
    }
}

impl Eq for SandboxSpec {}

impl SandboxSpec {
    /// Build a spec from roots that have already been through
    /// [`prepare_writable_roots`].
    ///
    /// On Linux this also opens each root as a directory FD (`O_DIRECTORY |
    /// O_NOFOLLOW | O_CLOEXEC`) and `/dev/null` **once**; those descriptors are
    /// reused (via `dup`) by every subsequent [`apply`]. Call only at
    /// preparation time — never per tool call. [`SandboxMode::Off`] never
    /// invokes this, so a disabled sandbox opens nothing.
    pub fn from_prepared_roots(roots: Vec<PathBuf>) -> Result<Self, SandboxError> {
        let prepared = PreparedRoots::new(&roots)?;
        Ok(Self {
            writable_roots: roots,
            prepared: Arc::new(prepared),
        })
    }
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
    // Open trusted root FDs (Linux) here — once per run, before any tool runs.
    let spec = SandboxSpec::from_prepared_roots(roots)?;
    preflight_enforce(&spec)?;
    Ok(SandboxMode::Enforce(spec))
}

/// Linux Landlock launcher entry point (CLI intercept).
///
/// Expected argv shape (as received by the process):
/// `argv[0]=exe`, `argv[1]=__agent-sandbox-landlock`, `argv[2]=<fd-spec-json>`,
/// `argv[3]=--`, `argv[4…]=real program + args`.
///
/// The fd-spec is JSON of the form
/// `{"root_fds":[3,4,…],"dev_null_fd":N|null}`: FD numbers that the trusted
/// parent opened (`O_DIRECTORY|O_NOFOLLOW` for roots, plain open for
/// `/dev/null`) and made inheritable. The launcher **never** resolves a
/// pathname for Landlock authorization — only the inherited FDs are used.
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
    // argv: [exe, launcher_arg, fd_spec, "--", real_bin, real_args…]
    if argv.len() < 5 || argv.get(3).map(OsString::as_os_str) != Some(OsStr::new("--")) {
        return Err(format!(
            "bad launcher argv (want: {LANDLOCK_LAUNCHER_ARG} <fd-spec> -- <bin> <args…>)"
        ));
    }
    let spec_arg = argv[2]
        .to_str()
        .ok_or_else(|| "launcher spec is not valid UTF-8".to_string())?;
    let fd_spec = decode_landlock_fd_spec(spec_arg).map_err(|e| format!("spec: {e}"))?;
    // Adopt the inherited FDs as OwnedFd. From this point the launcher owns
    // them; PathBeneath will close them when the ruleset is dropped after
    // restrict_self (the real command never needs these FDs open).
    let root_fds = adopt_inherited_fds(&fd_spec.root_fds)?;
    let dev_null_fd = match fd_spec.dev_null_fd {
        Some(n) => Some(adopt_inherited_fd(n)?),
        None => None,
    };
    landlock_restrict_fds(&root_fds, dev_null_fd.as_ref()).map_err(|e| e.to_string())?;

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
    // Spec roots (and on Linux their FDs) are prepared exactly once per agent
    // run. Use them verbatim — no exists/create/canonicalize/open that could
    // re-authorize an attacker-swapped symlink between tool calls.
    debug_assert!(
        spec.writable_roots
            .iter()
            .all(|r| r.is_absolute() && !r.as_os_str().is_empty()),
        "SandboxSpec.writable_roots must be pre-resolved absolute paths"
    );
    // prepared.paths is the same sequence as writable_roots; reading it keeps
    // the Arc live and documents that apply trusts the one-shot prepare.
    debug_assert_eq!(
        &spec.writable_roots, &spec.prepared.paths,
        "writable_roots and prepared.paths must stay in lockstep"
    );
    #[cfg(target_os = "macos")]
    {
        apply_macos(cmd, bin, args, &spec.prepared.paths)
    }
    #[cfg(target_os = "linux")]
    {
        apply_linux(cmd, bin, args, spec)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        let _ = (cmd, bin, args, &spec.prepared.paths);
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
/// 2. Expand a **fixed allowlist** of known platform system aliases (macOS
///    `/var`→`/private/var`, `/tmp`→`/private/tmp`, `/etc`→`/private/etc`) so
///    legitimate host paths still resolve. Any other symlink in any component
///    is a hard rejection on both platforms — no `access(W_OK)` heuristic,
///    no ownership probe, no exceptions.
/// 3. `create_dir_all` so missing trailing components exist **before** the walk.
/// 4. Walk every component with `symlink_metadata` and **reject** if any existing
///    component is a symlink (names the offending component) or if the final
///    path is not a directory.
/// 5. `canonicalize` the validated path (now free of symlinks; result is the
///    stable identity used for the rest of the run).
///
/// Called exactly once per agent run (via [`resolve_enforce_spec`]); never from
/// per-tool [`apply`].
pub fn prepare_writable_roots(roots: &[PathBuf]) -> Result<Vec<PathBuf>, SandboxError> {
    let mut out = Vec::with_capacity(roots.len());
    for r in roots {
        let abs = make_absolute(r)?;
        // Expand known system-alias prefixes (e.g. /var → /private/var) first,
        // so the reject-walk below only sees non-allowlisted components.
        let abs = expand_system_alias_prefixes(&abs)?;
        // BEFORE create_dir_all: reject any symlink already present in an
        // existing prefix. create_dir_all follows intermediate symlinks, so a
        // user-planted link (even under a currently-0555 parent) would otherwise
        // create the trailing components under the outside target before the
        // post-create reject walk could fire — leaving an outside child behind
        // even on an Err return.
        reject_existing_symlink_prefix(&abs)?;
        // Create missing trailing components, then re-walk the full path.
        // If the path already exists as a non-directory, create_dir_all errors
        // with AlreadyExists / "File exists" — fall through so the type check
        // below reports "not a directory" cleanly.
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

/// Fixed allowlist of known platform system aliases that may appear as a
/// leading path component of a requested writable root.
///
/// Empirically verified on this macOS host (`readlink` of each source):
/// - `/var` → `private/var` (resolves to `/private/var`)
/// - `/tmp` → `private/tmp` (resolves to `/private/tmp`)
/// - `/etc` → `private/etc` (resolves to `/private/etc`)
///
/// These three are the only top-level aliases needed for real agent roots
/// (`std::env::temp_dir()` is under `/var/folders/…`) to resolve. Any other
/// symlink — including user-owned `0555` directories containing attacker
/// links, or `/home` on modern macOS — is a hard rejection. Empty on
/// non-macOS: Linux agent roots live under real directories.
#[cfg(target_os = "macos")]
const SYSTEM_ALIAS_ALLOWLIST: &[(&str, &str)] = &[
    ("/var", "/private/var"),
    ("/tmp", "/private/tmp"),
    ("/etc", "/private/etc"),
];

#[cfg(not(target_os = "macos"))]
const SYSTEM_ALIAS_ALLOWLIST: &[(&str, &str)] = &[];

/// Expand a leading path component when (and only when) it matches the fixed
/// [`SYSTEM_ALIAS_ALLOWLIST`]. Returns the rewritten absolute path; any other
/// symlink is left for [`reject_symlink_components`] to refuse.
///
/// Only the **first** path component after the root is considered for
/// expansion (the three macOS aliases are all single top-level names). Nested
/// allowlisted names do not appear in practice and would be rejected by the
/// subsequent component walk if they did.
fn expand_system_alias_prefixes(path: &Path) -> Result<PathBuf, SandboxError> {
    if SYSTEM_ALIAS_ALLOWLIST.is_empty() {
        return Ok(path.to_path_buf());
    }
    // Identify the first Normal component after RootDir / Prefix.
    let mut comps = path.components();
    let mut prefix = PathBuf::new();
    let first_normal = loop {
        match comps.next() {
            Some(Component::Prefix(p)) => prefix.push(p.as_os_str()),
            Some(Component::RootDir) => prefix.push(Component::RootDir.as_os_str()),
            Some(Component::CurDir) => continue,
            Some(Component::ParentDir) => {
                // Leading `..` after root is still root; keep walking.
                continue;
            }
            Some(Component::Normal(s)) => break Some(s),
            None => return Ok(path.to_path_buf()),
        }
    };
    let Some(name) = first_normal else {
        return Ok(path.to_path_buf());
    };
    // Build the candidate absolute first-component path (e.g. "/var").
    let mut head = prefix;
    if head.as_os_str().is_empty() {
        // Relative paths were made absolute by make_absolute; still be safe.
        return Ok(path.to_path_buf());
    }
    head.push(name);
    let head_str = head.to_string_lossy();
    let Some(&(_, target)) = SYSTEM_ALIAS_ALLOWLIST
        .iter()
        .find(|(src, _)| *src == head_str.as_ref())
    else {
        // Not an allowlisted alias. Leave intact for reject_symlink_components.
        return Ok(path.to_path_buf());
    };
    // Verify the alias still points where we expect before expanding — a
    // host that rewrote the system layout must not silently authorize a
    // different target. We compare the *lexical* readlink text (relative or
    // absolute) against the known destination, accepting either the relative
    // form macOS uses (`private/var`) or the absolute form (`/private/var`).
    match std::fs::read_link(&head) {
        Ok(link) => {
            let link_os = link.as_os_str();
            let expected_rel = target.trim_start_matches('/');
            let ok = link_os == OsStr::new(target)
                || link_os == OsStr::new(expected_rel)
                || link == Path::new(target);
            if !ok {
                return Err(SandboxError::Io(format!(
                    "system alias {} no longer points at {} (readlink={}); refusing",
                    head.display(),
                    target,
                    link.display()
                )));
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // Alias missing entirely — leave for later create/reject.
            return Ok(path.to_path_buf());
        }
        Err(e) => {
            // Not a symlink (or unreadable): do not expand.
            // EINVAL / "Invalid argument" from read_link means not a symlink.
            if e.kind() == std::io::ErrorKind::InvalidInput {
                return Ok(path.to_path_buf());
            }
            return Err(SandboxError::Io(format!(
                "cannot read system alias {}: {e}",
                head.display()
            )));
        }
    }
    let mut out = PathBuf::from(target);
    for c in comps {
        out.push(c.as_os_str());
    }
    Ok(out)
}

/// Walk every *existing* component of `path` with `symlink_metadata` and reject
/// if any is a symlink. Stops cleanly at the first missing component (so the
/// caller may still `create_dir_all` the trailing suffix). Lexical `..` is
/// applied without resolving across links.
///
/// Used **before** `create_dir_all` so we never follow a user-planted symlink
/// and materialize directories outside the intended root.
fn reject_existing_symlink_prefix(path: &Path) -> Result<(), SandboxError> {
    walk_components(path, MissingComponent::Stop)
}

/// Walk every component of `path` with `symlink_metadata` and reject if any is
/// a symlink. Also rejects empty paths and missing components (post-create
/// fail-closed). Lexical `..` is applied without resolving across links.
///
/// Naming: the error mentions the first offending component path so operators
/// can see which ancestor was swapped for a symlink.
fn reject_symlink_components(path: &Path) -> Result<(), SandboxError> {
    walk_components(path, MissingComponent::Error)
}

/// How [`walk_components`] treats a `NotFound` intermediate.
#[derive(Clone, Copy)]
enum MissingComponent {
    /// Pre-create: missing suffix is fine (will be created next).
    Stop,
    /// Post-create: missing component is a race — fail closed.
    Error,
}

fn walk_components(path: &Path, on_missing: MissingComponent) -> Result<(), SandboxError> {
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
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => match on_missing {
                MissingComponent::Stop => return Ok(()),
                MissingComponent::Error => {
                    return Err(SandboxError::Io(format!(
                        "writable root component missing after create: {}: {e}",
                        cur.display()
                    )));
                }
            },
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

// ── Linux (Landlock launcher over inherited trusted FDs) ────────────────────
//
// Design invariant: the launcher never resolves a pathname for authorization.
// The trusted parent opens each prepared root **once per agent run** (during
// `SandboxSpec::from_prepared_roots` / `resolve_enforce_spec`) with
//   open(path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
// (and `/dev/null` without O_DIRECTORY/O_NOFOLLOW). Every subsequent tool
// spawn **duplicates** those held FDs, clears FD_CLOEXEC on the dups inside a
// pre_exec hook (so only the launcher child inherits them — not other
// concurrent forks), and encodes the raw dup FD numbers in the launcher argv
// JSON. The launcher adopts those FDs and builds PathBeneath rules directly
// from them. A background process that swaps a directory for a symlink after
// preparation cannot redirect Landlock: the held FD already points at the
// original directory inode, and no pathname is re-opened after preparation.

/// JSON payload passed as argv[2] of the Landlock launcher.
///
/// `root_fds` are directory FDs opened by the parent with
/// `O_DIRECTORY|O_NOFOLLOW`. `dev_null_fd` is optional (absent when
/// `/dev/null` could not be opened in the parent — rare, non-fatal).
#[cfg(target_os = "linux")]
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct LandlockFdSpec {
    root_fds: Vec<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    dev_null_fd: Option<i32>,
}

/// Minimum Landlock ABI we require: V3 mediates `truncate` / `ftruncate` /
/// `creat` / `O_TRUNC`. Lower ABIs would report successful confinement while
/// still allowing outside-file truncation.
#[cfg(target_os = "linux")]
const LANDLOCK_ABI_FLOOR: landlock::ABI = landlock::ABI::V3;

/// Linux open(2) flags used by the trusted parent. Declared locally so we do
/// not need a direct `libc` crate dep (values stable on x86_64 / aarch64 /
/// riscv64 Linux UAPI).
#[cfg(target_os = "linux")]
mod open_flags {
    pub const O_RDONLY: i32 = 0;
    pub const O_DIRECTORY: i32 = 0o200_000;
    pub const O_NOFOLLOW: i32 = 0o400_000;
    pub const O_CLOEXEC: i32 = 0o2_000_000;
    pub const F_GETFD: i32 = 1;
    pub const F_SETFD: i32 = 2;
    pub const FD_CLOEXEC: i32 = 1;
}

#[cfg(target_os = "linux")]
fn apply_linux(
    cmd: &mut Command,
    bin: &Path,
    args: &[impl AsRef<OsStr>],
    spec: &SandboxSpec,
) -> Result<(), SandboxError> {
    // Probe Landlock support without opening paths (empty-FD ruleset). The
    // actual restrict_self runs in the already-exec'd launcher process.
    // Path opens are NOT performed here — they happened once at preparation.
    preflight_linux(&spec.writable_roots)?;

    // Duplicate the preparation-time FDs for this spawn. The held originals
    // stay in `spec.prepared` for the whole run; the dups are made inheritable
    // and transferred to the child. No pathname is opened.
    let root_fds = dup_prepared_root_fds(&spec.prepared.root_fds)?;
    let dev_null_fd = match &spec.prepared.dev_null_fd {
        Some(fd) => Some(fd.dup()?),
        None => None,
    };

    let root_raw: Vec<i32> = root_fds.iter().map(|f| f.as_raw_fd_i32()).collect();
    let null_raw = dev_null_fd.as_ref().map(|f| f.as_raw_fd_i32());
    let fd_spec = encode_landlock_fd_spec(&LandlockFdSpec {
        root_fds: root_raw,
        dev_null_fd: null_raw,
    })?;

    let launcher = std::env::current_exe()
        .map_err(|e| SandboxError::Io(format!("current_exe for landlock launcher: {e}")))?;

    *cmd = Command::new(launcher);
    cmd.arg(LANDLOCK_LAUNCHER_ARG);
    cmd.arg(fd_spec);
    cmd.arg("--");
    cmd.arg(bin);
    cmd.args(args.iter().map(AsRef::as_ref));

    // Keep the per-spawn OwnedFd dups alive across the spawn by moving them
    // into the pre_exec closure (which also clears FD_CLOEXEC). The parent
    // still holds the preparation-time originals in `spec.prepared`.
    install_inheritable_fds(cmd, root_fds, dev_null_fd)?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn preflight_linux(_roots: &[PathBuf]) -> Result<(), SandboxError> {
    // Build a ruleset with no path rules under HardRequirement. This is enough
    // to detect "kernel missing Landlock / below V3 write floor" without
    // opening pathnames. Path opens happen only in PreparedRoots::new (once
    // per run, during resolve_enforce_spec / from_prepared_roots).
    landlock_build_ruleset_empty().map(|_| ())
}

/// Open every prepared root with `O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC`.
///
/// Called **once per run** from [`PreparedRoots::new`] — never from
/// [`apply_linux`]. `O_NOFOLLOW` makes a final-component symlink a hard open
/// error rather than a silent redirect. Combined with the once-per-run
/// preparation that already rejected symlink *ancestors*, the resulting FD is
/// a trusted handle on the intended directory inode.
#[cfg(target_os = "linux")]
fn open_trusted_root_fds(roots: &[PathBuf]) -> Result<Vec<OwnedDirFd>, SandboxError> {
    #[cfg(test)]
    TRUSTED_ROOT_OPEN_COUNT.fetch_add(roots.len() as u64, std::sync::atomic::Ordering::Relaxed);
    let mut out = Vec::with_capacity(roots.len());
    for r in roots {
        out.push(OwnedDirFd::open_dir_nofollow(r)?);
    }
    Ok(out)
}

/// Test-only counter of pathname opens performed by [`open_trusted_root_fds`].
/// Used to prove `apply_linux` never re-opens roots after preparation.
#[cfg(all(test, target_os = "linux"))]
static TRUSTED_ROOT_OPEN_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Duplicate every held root FD for a single spawn (no pathname open).
#[cfg(target_os = "linux")]
fn dup_prepared_root_fds(fds: &[OwnedDirFd]) -> Result<Vec<OwnedDirFd>, SandboxError> {
    let mut out = Vec::with_capacity(fds.len());
    for f in fds {
        out.push(f.dup()?);
    }
    Ok(out)
}

#[cfg(target_os = "linux")]
fn open_trusted_dev_null_fd() -> Result<OwnedDirFd, SandboxError> {
    OwnedDirFd::open_file(Path::new("/dev/null"))
}

/// Thin owned-FD wrapper so we can clear CLOEXEC / adopt raw numbers without
/// depending on a `libc` crate. Drop closes the FD (parent side).
#[cfg(target_os = "linux")]
struct OwnedDirFd {
    fd: i32,
}

#[cfg(target_os = "linux")]
impl fmt::Debug for OwnedDirFd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OwnedDirFd").field("fd", &self.fd).finish()
    }
}

#[cfg(target_os = "linux")]
impl OwnedDirFd {
    fn open_dir_nofollow(path: &Path) -> Result<Self, SandboxError> {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;
        let c = CString::new(path.as_os_str().as_bytes()).map_err(|_| {
            SandboxError::Io(format!(
                "writable root path contains interior NUL: {}",
                path.display()
            ))
        })?;
        let flags = open_flags::O_RDONLY
            | open_flags::O_DIRECTORY
            | open_flags::O_NOFOLLOW
            | open_flags::O_CLOEXEC;
        // SAFETY: c is a valid NUL-terminated path; open returns -1 or a fresh FD.
        let fd = unsafe { sys_open(c.as_ptr(), flags, 0) };
        if fd < 0 {
            let err = std::io::Error::last_os_error();
            return Err(SandboxError::Io(format!(
                "cannot open trusted root FD for {}: {err}",
                path.display()
            )));
        }
        Ok(Self { fd })
    }

    fn open_file(path: &Path) -> Result<Self, SandboxError> {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;
        let c = CString::new(path.as_os_str().as_bytes()).map_err(|_| {
            SandboxError::Io(format!("path contains interior NUL: {}", path.display()))
        })?;
        let flags = open_flags::O_RDONLY | open_flags::O_CLOEXEC;
        // SAFETY: c is a valid NUL-terminated path.
        let fd = unsafe { sys_open(c.as_ptr(), flags, 0) };
        if fd < 0 {
            let err = std::io::Error::last_os_error();
            return Err(SandboxError::Io(format!(
                "cannot open {}: {err}",
                path.display()
            )));
        }
        Ok(Self { fd })
    }

    fn as_raw_fd_i32(&self) -> i32 {
        self.fd
    }

    /// Duplicate this FD (`F_DUPFD_CLOEXEC`). Pure descriptor clone — never
    /// re-resolves a pathname. Used by per-spawn `apply_linux` so the
    /// preparation-time original can stay held for the whole run.
    fn dup(&self) -> Result<Self, SandboxError> {
        // F_DUPFD_CLOEXEC = F_DUPFD (0) | O_CLOEXEC-style flag 1030 on Linux.
        // Value is stable on Linux UAPI: 1030.
        const F_DUPFD_CLOEXEC: i32 = 1030;
        // SAFETY: F_DUPFD_CLOEXEC on an FD we own; returns a fresh FD or -1.
        let new_fd = unsafe { sys_fcntl(self.fd, F_DUPFD_CLOEXEC, 0) };
        if new_fd < 0 {
            let err = std::io::Error::last_os_error();
            return Err(SandboxError::Io(format!(
                "cannot dup trusted root FD {}: {err}",
                self.fd
            )));
        }
        Ok(Self { fd: new_fd })
    }
}

#[cfg(target_os = "linux")]
impl Drop for OwnedDirFd {
    fn drop(&mut self) {
        if self.fd >= 0 {
            // SAFETY: close a FD we own; ignore EINTR/EBADF on drop.
            unsafe {
                sys_close(self.fd);
            }
            self.fd = -1;
        }
    }
}

// Raw syscall wrappers (avoid a libc crate dep). Linux x86_64/aarch64 UAPI.
#[cfg(target_os = "linux")]
unsafe fn sys_open(path: *const std::os::raw::c_char, flags: i32, mode: i32) -> i32 {
    unsafe extern "C" {
        fn open(path: *const std::os::raw::c_char, flags: i32, mode: i32) -> i32;
    }
    unsafe { open(path, flags, mode) }
}

#[cfg(target_os = "linux")]
unsafe fn sys_fcntl(fd: i32, cmd: i32, arg: i32) -> i32 {
    unsafe extern "C" {
        fn fcntl(fd: i32, cmd: i32, arg: i32) -> i32;
    }
    unsafe { fcntl(fd, cmd, arg) }
}

#[cfg(target_os = "linux")]
unsafe fn sys_close(fd: i32) -> i32 {
    unsafe extern "C" {
        fn close(fd: i32) -> i32;
    }
    unsafe { close(fd) }
}

/// Attach opened root FDs to `cmd` so the child inherits them.
///
/// Implementation: `pre_exec` clears `FD_CLOEXEC` on every FD (async-signal-
/// safe: only `fcntl`). The `OwnedDirFd` values are moved into the closure so
/// they stay alive until after `spawn`/`exec` completes in the parent; the
/// child receives the same integer FD numbers referenced by the JSON spec.
///
/// We deliberately open with CLOEXEC and only clear it in `pre_exec`, so a
/// concurrent `Command::spawn` elsewhere in the process cannot accidentally
/// inherit these privileged directory FDs.
#[cfg(target_os = "linux")]
fn install_inheritable_fds(
    cmd: &mut Command,
    root_fds: Vec<OwnedDirFd>,
    dev_null_fd: Option<OwnedDirFd>,
) -> Result<(), SandboxError> {
    use std::os::unix::process::CommandExt;

    // Capture raw numbers for the closure; move ownership of the wrappers so
    // Drop runs only after the spawn path releases the closure (keeps FDs open
    // across fork). pre_exec itself only calls fcntl (async-signal-safe).
    let all_fds: Vec<OwnedDirFd> = {
        let mut v = root_fds;
        if let Some(n) = dev_null_fd {
            v.push(n);
        }
        v
    };
    let raw_list: Vec<i32> = all_fds.iter().map(|f| f.as_raw_fd_i32()).collect();
    // SAFETY: pre_exec runs in the child between fork and exec. We only call
    // fcntl (async-signal-safe). We must not allocate / take locks here.
    unsafe {
        cmd.pre_exec(move || {
            // Move all_fds into the closure so Drop runs after spawn returns
            // in the parent (and after exec in the child success path the
            // FDs are simply inherited — Drop does not run in the child
            // after a successful exec).
            let _keep = &all_fds;
            for &fd in &raw_list {
                let flags = sys_fcntl(fd, open_flags::F_GETFD, 0);
                if flags < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                let new_flags = flags & !open_flags::FD_CLOEXEC;
                if sys_fcntl(fd, open_flags::F_SETFD, new_flags) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
            }
            Ok(())
        });
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn encode_landlock_fd_spec(spec: &LandlockFdSpec) -> Result<String, SandboxError> {
    serde_json::to_string(spec).map_err(|e| SandboxError::Io(format!("encode fd-spec: {e}")))
}

#[cfg(target_os = "linux")]
fn decode_landlock_fd_spec(s: &str) -> Result<LandlockFdSpec, String> {
    serde_json::from_str(s).map_err(|e| format!("decode fd-spec: {e}"))
}

/// Adopt a raw inherited FD number as an `OwnedFd` without re-opening.
///
/// The parent guaranteed these FDs are live and refer to the prepared roots
/// / `/dev/null`. We check the FD is open (`F_GETFD`) and that the count of
/// root FDs is non-zero when expected; a closed/invalid number is fail-closed.
#[cfg(target_os = "linux")]
fn adopt_inherited_fd(raw: i32) -> Result<std::os::fd::OwnedFd, String> {
    use std::os::fd::{FromRawFd, OwnedFd};
    if raw < 0 {
        return Err(format!("invalid inherited fd number {raw}"));
    }
    // SAFETY: F_GETFD on a candidate FD; returns -1 if not open.
    let flags = unsafe { sys_fcntl(raw, open_flags::F_GETFD, 0) };
    if flags < 0 {
        return Err(format!(
            "inherited fd {raw} is not open in launcher: {}",
            std::io::Error::last_os_error()
        ));
    }
    // SAFETY: parent opened this FD and passed the number in the fd-spec;
    // F_GETFD confirmed it is open. We take ownership so PathBeneath / Drop
    // will close it after restrict_self.
    Ok(unsafe { OwnedFd::from_raw_fd(raw) })
}

#[cfg(target_os = "linux")]
fn adopt_inherited_fds(raws: &[i32]) -> Result<Vec<std::os::fd::OwnedFd>, String> {
    if raws.is_empty() {
        return Err("fd-spec root_fds is empty".into());
    }
    // Reject duplicates — a confused parent must not hand the same FD twice
    // (would double-close under OwnedFd Drop). Scan the whole spec first: a
    // duplicate is a duplicate whether or not the first copy happens to be a
    // live descriptor, and adopting nothing until the spec is known-good means
    // a rejected spec leaves no half-owned FDs behind.
    let mut seen = std::collections::BTreeSet::new();
    for &n in raws {
        if !seen.insert(n) {
            return Err(format!("fd-spec has duplicate root fd {n}"));
        }
    }
    let mut out = Vec::with_capacity(raws.len());
    for &n in raws {
        out.push(adopt_inherited_fd(n)?);
    }
    Ok(out)
}

/// Build a Landlock ruleset with no path rules — used only for preflight
/// capability probing in the parent.
#[cfg(target_os = "linux")]
fn landlock_build_ruleset_empty() -> Result<landlock::RulesetCreated, SandboxError> {
    use landlock::{AccessFs, CompatLevel, Compatible, Ruleset, RulesetAttr};

    let write = AccessFs::from_write(LANDLOCK_ABI_FLOOR);
    let created = match Ruleset::default()
        .set_compatibility(CompatLevel::HardRequirement)
        .handle_access(write)
    {
        Ok(r) => r,
        Err(e) => return Err(classify_ruleset_error(e)),
    };
    match created.create() {
        Ok(r) => Ok(r),
        Err(e) => Err(classify_ruleset_error(e)),
    }
}

/// Build a Landlock ruleset whose PathBeneath rules reference already-open
/// FDs (never pathnames).
///
/// `PathBeneath::new` accepts any `AsFd`; we pass `OwnedFd` directly so the
/// landlock crate never opens anything on its own.
#[cfg(target_os = "linux")]
fn landlock_build_ruleset_from_fds(
    root_fds: &[std::os::fd::OwnedFd],
    dev_null_fd: Option<&std::os::fd::OwnedFd>,
) -> Result<landlock::RulesetCreated, SandboxError> {
    use landlock::{
        AccessFs, CompatLevel, Compatible, PathBeneath, Ruleset, RulesetAttr, RulesetCreatedAttr,
    };

    let write = AccessFs::from_write(LANDLOCK_ABI_FLOOR);

    let created = match Ruleset::default()
        .set_compatibility(CompatLevel::HardRequirement)
        .handle_access(write)
    {
        Ok(r) => r,
        Err(e) => return Err(classify_ruleset_error(e)),
    };
    let mut ruleset = match created.create() {
        Ok(r) => r,
        Err(e) => return Err(classify_ruleset_error(e)),
    };

    for fd in root_fds {
        // PathBeneath::new accepts any AsFd (OwnedFd implements it). Use
        // try_clone so each rule owns an independent FD (PathBeneath closes
        // its parent_fd on drop). Cloning is a pure fcntl(F_DUPFD_CLOEXEC).
        let owned = fd
            .try_clone()
            .map_err(|e| SandboxError::Landlock(format!("clone root fd: {e}")))?;
        ruleset = ruleset
            .add_rule(PathBeneath::new(owned, write))
            .map_err(|e| SandboxError::Landlock(format!("add_rule root fd: {e}")))?;
    }

    // Narrow device-node allowance (never all of /dev). Only the write rights
    // applicable to a non-directory file — WriteFile + Truncate.
    if let Some(null_fd) = dev_null_fd {
        let dev_write = AccessFs::WriteFile | AccessFs::Truncate;
        let owned = null_fd
            .try_clone()
            .map_err(|e| SandboxError::Landlock(format!("clone /dev/null fd: {e}")))?;
        ruleset = ruleset
            .add_rule(PathBeneath::new(owned, dev_write))
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
fn landlock_restrict_fds(
    root_fds: &[std::os::fd::OwnedFd],
    dev_null_fd: Option<&std::os::fd::OwnedFd>,
) -> Result<(), SandboxError> {
    use landlock::RulesetStatus;

    let ruleset = landlock_build_ruleset_from_fds(root_fds, dev_null_fd)?;
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

    /// Prepare once (as the CLI does) so Apply sees pre-resolved roots + FDs.
    fn resolved_spec(raw: &[PathBuf]) -> SandboxSpec {
        let roots = prepare_writable_roots(raw).expect("prepare roots");
        SandboxSpec::from_prepared_roots(roots).expect("open prepared roots")
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

    /// T1(ii) on Linux under the once-per-run trusted-FD design: after a
    /// post-resolve directory → symlink swap, `apply` still succeeds (it dups
    /// the preparation-time FDs; it never re-opens pathnames) with an FD-spec
    /// that contains **only FD numbers** — never a pathname that could be
    /// re-resolved to the outside target. Pathnames are gone from the launcher
    /// contract entirely. An open-count counter proves apply does not open.
    #[cfg(target_os = "linux")]
    #[test]
    fn apply_linux_fd_spec_has_no_pathnames_after_symlink_swap() {
        let base = unique("swap-ll");
        std::fs::create_dir_all(&base).unwrap();
        let root = base.join("root");
        std::fs::create_dir_all(&root).unwrap();
        let outside = base.join("outside");
        std::fs::create_dir_all(&outside).unwrap();

        let opens_before = TRUSTED_ROOT_OPEN_COUNT.load(Ordering::Relaxed);

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
        let opens_after_prep = TRUSTED_ROOT_OPEN_COUNT.load(Ordering::Relaxed);
        assert!(
            opens_after_prep > opens_before,
            "preparation must open trusted root FDs once"
        );
        // Held FDs were opened at prep and stay pinned to the original inodes.
        assert_eq!(
            spec.prepared.root_fds.len(),
            spec.writable_roots.len(),
            "one held FD per prepared root"
        );

        let aside = base.join("root-aside");
        std::fs::rename(&root, &aside).unwrap();
        std::os::unix::fs::symlink(&outside, &root).unwrap();

        let mut cmd = Command::new("placeholder");
        let bin = PathBuf::from("/bin/echo");
        // Apply must succeed: it dups held FDs and never re-opens the (now
        // swapped) pathnames. A post-prep ancestor swap cannot redirect them.
        apply(&mut cmd, &bin, &["hi"][..], &mode).expect("apply after swap must use held FDs");
        let opens_after_apply = TRUSTED_ROOT_OPEN_COUNT.load(Ordering::Relaxed);
        assert_eq!(
            opens_after_apply, opens_after_prep,
            "apply must not open any additional trusted-root pathnames"
        );

        let got: Vec<_> = cmd
            .get_args()
            .map(|s| s.to_string_lossy().into_owned())
            .collect();
        assert_eq!(got[0], LANDLOCK_LAUNCHER_ARG);
        let spec_json = &got[1];
        // Spec is FD-only JSON: must parse as LandlockFdSpec and must
        // NOT contain any pathname (neither original nor outside).
        let decoded: LandlockFdSpec = serde_json::from_str(spec_json).expect("fd-spec json");
        assert!(
            !decoded.root_fds.is_empty(),
            "fd-spec must carry at least one root fd: {spec_json}"
        );
        assert!(
            !spec_json.contains(original.to_str().unwrap()),
            "fd-spec must not embed pathnames; got {spec_json}"
        );
        let outside_canon = std::fs::canonicalize(&outside).unwrap();
        if let Some(s) = outside_canon.to_str() {
            assert!(
                !spec_json.contains(s),
                "fd-spec must not re-resolve to symlink target; got {spec_json}"
            );
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    /// FDs are opened exactly once per run: two applies after a single
    /// preparation do not perform additional pathname opens.
    #[cfg(target_os = "linux")]
    #[test]
    fn trusted_root_fds_opened_once_per_run_not_per_apply() {
        let root = unique("open-once");
        std::fs::create_dir_all(&root).unwrap();
        let opens_before = TRUSTED_ROOT_OPEN_COUNT.load(Ordering::Relaxed);
        let mode = match resolve_enforce_spec(std::slice::from_ref(&root)) {
            Ok(m) => m,
            Err(SandboxError::Unsupported) => {
                let _ = std::fs::remove_dir_all(&root);
                return;
            }
            Err(e) => panic!("resolve: {e}"),
        };
        let opens_after_prep = TRUSTED_ROOT_OPEN_COUNT.load(Ordering::Relaxed);
        assert_eq!(
            opens_after_prep,
            opens_before + 1,
            "one root → exactly one pathname open at prep"
        );

        let bin = PathBuf::from("/bin/echo");
        for _ in 0..3 {
            let mut cmd = Command::new("placeholder");
            apply(&mut cmd, &bin, &["hi"][..], &mode).expect("apply");
        }
        let opens_after_applies = TRUSTED_ROOT_OPEN_COUNT.load(Ordering::Relaxed);
        assert_eq!(
            opens_after_applies, opens_after_prep,
            "three applies must not open any additional pathnames"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// `SandboxMode::Off` never opens trusted root FDs.
    #[cfg(target_os = "linux")]
    #[test]
    fn off_mode_opens_no_trusted_root_fds() {
        let opens_before = TRUSTED_ROOT_OPEN_COUNT.load(Ordering::Relaxed);
        let mut cmd = Command::new("placeholder");
        let bin = PathBuf::from("/bin/echo");
        apply(&mut cmd, &bin, &["hi"][..], &SandboxMode::Off).unwrap();
        let opens_after = TRUSTED_ROOT_OPEN_COUNT.load(Ordering::Relaxed);
        assert_eq!(
            opens_after, opens_before,
            "Off must not open trusted root FDs"
        );
        // And constructing Off does not touch PreparedRoots either.
        let _ = SandboxMode::Off;
        assert_eq!(
            TRUSTED_ROOT_OPEN_COUNT.load(Ordering::Relaxed),
            opens_before
        );
    }

    /// Spec equality is path-based; two preparations of the same roots compare
    /// equal even though their held FDs differ.
    #[test]
    fn sandbox_spec_eq_compares_paths_not_fds() {
        let root = unique("eq-paths");
        std::fs::create_dir_all(&root).unwrap();
        let roots = prepare_writable_roots(std::slice::from_ref(&root)).unwrap();
        let a = SandboxSpec::from_prepared_roots(roots.clone()).expect("a");
        let b = SandboxSpec::from_prepared_roots(roots).expect("b");
        assert_eq!(a, b, "path-equal specs must compare equal");
        assert_eq!(a.writable_roots, b.writable_roots);
        // Clone shares the Arc; equality still holds.
        let c = a.clone();
        assert_eq!(a, c);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// U1: user-owned 0555 directory containing a symlink must be REJECTED by
    /// the public `prepare_writable_roots` API. The previous `access(W_OK)`
    /// heuristic treated "not writable right now" as "immutable system link"
    /// and authorized the outside target — that hole must stay closed.
    ///
    /// Also asserts the outside child does **not** exist afterwards (prepare
    /// must not create anything under the symlink target).
    #[cfg(unix)]
    #[test]
    fn prepare_rejects_symlink_under_user_owned_0555_dir() {
        use std::os::unix::fs::{symlink, PermissionsExt};

        let base = unique("u1-0555");
        std::fs::create_dir_all(&base).unwrap();
        let outside = base.join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        let outside_child = outside.join("child");
        // Ensure the probe target does not exist before the call.
        let _ = std::fs::remove_dir_all(&outside_child);
        let _ = std::fs::remove_file(&outside_child);
        assert!(!outside_child.exists());

        let gate = base.join("gate");
        std::fs::create_dir_all(&gate).unwrap();
        // Plant the symlink while the parent is still writable.
        let link = gate.join("link");
        symlink(&outside, &link).unwrap();
        // Drop write bits: access(W_OK) would now fail, which is exactly the
        // misclassification the old heuristic made.
        let mut perms = std::fs::metadata(&gate).unwrap().permissions();
        perms.set_mode(0o555);
        std::fs::set_permissions(&gate, perms).unwrap();

        let requested = link.join("child");
        let result = prepare_writable_roots(std::slice::from_ref(&requested));

        // Restore writability so cleanup can proceed even on assertion failure.
        let mut perms = std::fs::metadata(&gate).unwrap().permissions();
        perms.set_mode(0o755);
        let _ = std::fs::set_permissions(&gate, perms);

        match result {
            Ok(roots) => {
                let _ = std::fs::remove_dir_all(&base);
                panic!("user-owned 0555 dir + symlink must be REJECTED; got roots={roots:?}");
            }
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("symlink"),
                    "expected symlink rejection, got: {msg}"
                );
            }
        }
        assert!(
            !outside_child.exists(),
            "outside child must NOT exist after rejected prepare: {}",
            outside_child.display()
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    /// U1: a plain user-created symlink used as a writable root is rejected
    /// (no allowlist entry covers user paths).
    #[cfg(unix)]
    #[test]
    fn prepare_rejects_plain_user_symlink_root() {
        use std::os::unix::fs::symlink;
        let base = unique("u1-user-sym");
        std::fs::create_dir_all(&base).unwrap();
        let target = base.join("target");
        std::fs::create_dir_all(&target).unwrap();
        let link = base.join("link");
        symlink(&target, &link).unwrap();
        let err = prepare_writable_roots(std::slice::from_ref(&link)).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("symlink"),
            "plain user symlink root must be rejected: {msg}"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    /// U1: `/var/...`-style system-alias roots still resolve via the fixed
    /// allowlist (macOS). On other platforms this is a no-op of the temp-dir
    /// acceptance already covered elsewhere.
    #[cfg(target_os = "macos")]
    #[test]
    fn prepare_expands_macos_var_system_alias() {
        // temp_dir() on macOS is under /var/folders/… which must expand through
        // the fixed /var → /private/var allowlist entry.
        let tmp = std::env::temp_dir();
        assert!(
            tmp.starts_with("/var") || tmp.starts_with("/private/var"),
            "expected macOS temp_dir under /var, got {}",
            tmp.display()
        );
        let roots = prepare_writable_roots(std::slice::from_ref(&tmp)).unwrap();
        assert_eq!(roots.len(), 1);
        assert!(
            roots[0].starts_with("/private/var"),
            "canonical root must live under /private/var, got {}",
            roots[0].display()
        );
        // Direct /var/folders request (pre-expansion form) must also work.
        if tmp.starts_with("/var") {
            let via_var = prepare_writable_roots(std::slice::from_ref(&tmp)).unwrap();
            assert_eq!(via_var[0], roots[0]);
        }
    }

    /// U1 unit: expand_system_alias_prefixes rewrites only allowlisted heads.
    #[cfg(target_os = "macos")]
    #[test]
    fn expand_system_alias_only_allowlisted() {
        let expanded = expand_system_alias_prefixes(Path::new("/var/folders/x")).unwrap();
        assert_eq!(expanded, PathBuf::from("/private/var/folders/x"));
        let expanded = expand_system_alias_prefixes(Path::new("/tmp/foo")).unwrap();
        assert_eq!(expanded, PathBuf::from("/private/tmp/foo"));
        let expanded = expand_system_alias_prefixes(Path::new("/etc/hosts")).unwrap();
        assert_eq!(expanded, PathBuf::from("/private/etc/hosts"));
        // Non-allowlisted: returned unchanged (reject walk will catch symlinks).
        let unchanged = expand_system_alias_prefixes(Path::new("/Users/someone")).unwrap();
        assert_eq!(unchanged, PathBuf::from("/Users/someone"));
        let unchanged = expand_system_alias_prefixes(Path::new("/home/someone")).unwrap();
        assert_eq!(unchanged, PathBuf::from("/home/someone"));
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

    /// Linux-only: empty-FD ruleset construction (preflight) must not panic and
    /// must return Unsupported (not Landlock error) when the kernel lacks the
    /// V3 write floor.
    #[cfg(target_os = "linux")]
    #[test]
    fn linux_ruleset_construction_or_unsupported() {
        let root = unique("ll");
        let roots = prepare_writable_roots(std::slice::from_ref(&root)).unwrap();
        match preflight_linux(&roots) {
            Ok(()) => {
                let _ = landlock_build_ruleset_empty().unwrap();
            }
            Err(SandboxError::Unsupported) => {}
            Err(e) => panic!("unexpected: {e}"),
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    /// U2: apply rewrites to the landlock launcher with an FD-spec JSON
    /// (not a pathname list). Shape:
    ///   argv = [LANDLOCK_LAUNCHER_ARG, <fd-spec-json>, "--", bin, args…]
    #[cfg(target_os = "linux")]
    #[test]
    fn apply_linux_rewrites_to_landlock_launcher_with_fd_spec() {
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
                // got[1] is the FD-spec JSON object (not a bare path array).
                assert!(
                    got[1].starts_with('{'),
                    "fd-spec must be a JSON object, got {}",
                    got[1]
                );
                let decoded: LandlockFdSpec =
                    serde_json::from_str(&got[1]).expect("fd-spec parses");
                assert_eq!(
                    decoded.root_fds.len(),
                    1,
                    "one prepared root → one FD: {:?}",
                    decoded.root_fds
                );
                assert!(
                    decoded.root_fds[0] >= 0,
                    "FD numbers must be non-negative: {:?}",
                    decoded.root_fds
                );
                // No pathnames in the spec.
                assert!(
                    !got[1].contains(root.to_str().unwrap_or("\0")),
                    "fd-spec must not embed pathnames: {}",
                    got[1]
                );
                assert_eq!(got[2], "--");
                assert_eq!(got[3], "/bin/echo");
                assert_eq!(got[4], "hi");
            }
            Err(SandboxError::Unsupported) => {}
            Err(e) => panic!("unexpected: {e}"),
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    /// U2: fd-spec encode/decode round-trip (platform-independent logic,
    /// compiled only on Linux where the type lives).
    #[cfg(target_os = "linux")]
    #[test]
    fn landlock_fd_spec_roundtrip() {
        let spec = LandlockFdSpec {
            root_fds: vec![7, 8, 9],
            dev_null_fd: Some(10),
        };
        let s = encode_landlock_fd_spec(&spec).unwrap();
        let back = decode_landlock_fd_spec(&s).unwrap();
        assert_eq!(back, spec);

        let spec2 = LandlockFdSpec {
            root_fds: vec![3],
            dev_null_fd: None,
        };
        let s2 = encode_landlock_fd_spec(&spec2).unwrap();
        assert!(
            !s2.contains("dev_null_fd"),
            "None dev_null_fd should be omitted: {s2}"
        );
        let back2 = decode_landlock_fd_spec(&s2).unwrap();
        assert_eq!(back2, spec2);
    }

    /// U2: launcher refuses to run without GREPPY_AGENT_RUN (when the marker is
    /// not already set on the process). Avoids mutating process-global env so
    /// the parallel test runner stays sound.
    #[cfg(target_os = "linux")]
    #[test]
    fn landlock_launcher_refuses_without_marker() {
        if std::env::var_os(crate::AGENT_RUN_ENV).is_some() {
            // Another test (or the harness) left the marker set; skip rather
            // than race on process-global env.
            return;
        }
        let argv = vec![
            OsString::from("greppy"),
            OsString::from(LANDLOCK_LAUNCHER_ARG),
            OsString::from(r#"{"root_fds":[3]}"#),
            OsString::from("--"),
            OsString::from("/bin/true"),
        ];
        let rc = run_landlock_launcher(&argv);
        assert_eq!(
            rc, LANDLOCK_LAUNCHER_EXIT_SETUP,
            "launcher must refuse without GREPPY_AGENT_RUN"
        );
    }

    /// U2: fd-count / validity helpers reject empty, duplicate, and closed FDs.
    /// Pure validation — no Landlock syscalls, no env mutation.
    #[cfg(target_os = "linux")]
    #[test]
    fn landlock_fd_adoption_rejects_empty_dup_closed() {
        let empty = adopt_inherited_fds(&[]);
        assert!(empty.is_err(), "empty root_fds must fail");
        assert!(
            empty.unwrap_err().contains("empty"),
            "error should mention empty"
        );

        let dup = adopt_inherited_fds(&[3, 3]);
        assert!(dup.is_err(), "duplicate root fds must fail");
        assert!(
            dup.unwrap_err().contains("duplicate"),
            "error should mention duplicate"
        );

        // 1023 is almost certainly closed in a unit-test process.
        let closed = adopt_inherited_fd(1023);
        assert!(closed.is_err(), "closed inherited fd must fail");

        let neg = adopt_inherited_fd(-1);
        assert!(neg.is_err(), "negative fd must fail");
    }

    /// U2: bad launcher argv shape is rejected (when marker is present via the
    /// inner function's first check — we only exercise the argv branch by
    /// calling the pure length/`--` guard through decode + docs; the
    /// full launcher path is covered by `landlock_launcher_refuses_without_marker`
    /// for the marker and by the adoption helpers for the fd-spec).
    #[cfg(target_os = "linux")]
    #[test]
    fn landlock_fd_spec_decode_rejects_garbage() {
        assert!(decode_landlock_fd_spec("not-json").is_err());
        assert!(decode_landlock_fd_spec("[]").is_err());
        assert!(decode_landlock_fd_spec(r#"{"root_fds":"x"}"#).is_err());
        // Valid minimal object.
        let ok = decode_landlock_fd_spec(r#"{"root_fds":[4,5]}"#).unwrap();
        assert_eq!(ok.root_fds, vec![4, 5]);
        assert_eq!(ok.dev_null_fd, None);
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

    /// Platform-independent: system-alias allowlist is empty on non-macOS, so
    /// expand is a pure identity.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn expand_system_alias_is_identity_off_macos() {
        let p = PathBuf::from("/var/folders/x");
        let out = expand_system_alias_prefixes(&p).unwrap();
        assert_eq!(out, p);
    }
}
