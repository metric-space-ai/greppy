//! Production cache/store lifecycle for greppy.
//!
//! The data root is deliberately split into owned, versioned namespaces.  GC
//! never walks arbitrary children of `GREPPY_STORE_DIR`: it only considers
//! workspace stores with a valid manifest and model entries with a validated
//! digest-shaped directory.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub const STORE_FORMAT_VERSION: u32 = 2;
pub const STORE_MANIFEST_FILE: &str = "store.manifest";
pub const AGENT_BASE_FORMAT_VERSION: u32 = 1;
pub const AGENT_BASE_MANIFEST_FILE: &str = "base.cache.manifest";
pub const LAST_USED_FILE: &str = ".lastused";
pub const DEFAULT_STORE_TTL_DAYS: u64 = 14;
pub const DEFAULT_STORE_MAX_GIB: u64 = 10;
pub const DEFAULT_GC_INTERVAL_SECS: u64 = 10 * 60;
pub const DEFAULT_QUERY_CACHE_MAX_MIB: u64 = 64;
pub const DEFAULT_SUMMARY_CACHE_MAX_MIB: u64 = 32;
pub const DEFAULT_EMBEDDING_CACHE_MAX_MIB: u64 = 2048;
pub const ORPHAN_GRACE_SECS: u64 = 24 * 60 * 60;

const STORE_MANIFEST_MAGIC: &str = "greppy-workspace-store";
const AGENT_BASE_MANIFEST_MAGIC: &str = "greppy-agent-base-store";
const LAST_USED_WRITE_GAP: Duration = Duration::from_secs(60);
static ATOMIC_WRITE_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoreManifest {
    pub format_version: u32,
    pub workspace_hash: String,
    pub canonical_root: PathBuf,
    pub created_at_unix_secs: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentBaseManifest {
    pub format_version: u32,
    pub identity_hash: String,
    pub canonical_repository_identity: String,
    pub created_at_unix_secs: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockMode {
    Shared,
    Exclusive,
}

/// An OS-backed advisory lock.  The file itself is intentionally never
/// removed: deleting a lock path while another process has the old inode open
/// can split contenders across two independent locks.
pub struct FileLock {
    file: File,
    path: PathBuf,
}

impl std::fmt::Debug for FileLock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileLock")
            .field("path", &self.path)
            .finish()
    }
}

impl FileLock {
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        unlock_file(&self.file);
    }
}

#[derive(Debug, Clone)]
pub struct GcPolicy {
    pub ttl: Duration,
    pub high_water_bytes: u64,
    pub low_water_bytes: u64,
    pub interval: Duration,
}

impl GcPolicy {
    pub fn from_env() -> Self {
        let ttl_days = env_u64("GREPPY_STORE_TTL_DAYS", DEFAULT_STORE_TTL_DAYS);
        let max_gib = env_u64("GREPPY_STORE_MAX_GIB", DEFAULT_STORE_MAX_GIB);
        let interval_secs = env_u64("GREPPY_GC_INTERVAL_SECS", DEFAULT_GC_INTERVAL_SECS);
        let high = max_gib.saturating_mul(1024 * 1024 * 1024);
        let low = if high == 0 {
            0
        } else {
            high.saturating_mul(9) / 10
        };
        Self {
            ttl: Duration::from_secs(ttl_days.saturating_mul(24 * 60 * 60)),
            high_water_bytes: high,
            low_water_bytes: low,
            interval: Duration::from_secs(interval_secs),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheEntryStatus {
    pub kind: String,
    pub id: String,
    pub path: PathBuf,
    pub workspace_root: Option<PathBuf>,
    pub bytes: u64,
    pub last_used_unix_secs: u64,
    pub orphaned: bool,
    pub locked: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CacheStatus {
    pub data_root: PathBuf,
    pub managed_bytes: u64,
    pub unmanaged_bytes: u64,
    pub locked_bytes: u64,
    pub unmanaged: Vec<PathBuf>,
    pub entries: Vec<CacheEntryStatus>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GcReport {
    pub scanned_bytes: u64,
    pub removed_bytes: u64,
    pub locked_bytes: u64,
    pub removed: Vec<PathBuf>,
    pub skipped_locked: Vec<PathBuf>,
    pub dry_run: bool,
    pub throttled: bool,
}

#[derive(Debug, Clone)]
struct ManagedEntry {
    kind: ManagedKind,
    id: String,
    path: PathBuf,
    workspace_root: Option<PathBuf>,
    bytes: u64,
    last_used: SystemTime,
    orphaned: bool,
    orphaned_since: Option<SystemTime>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ManagedKind {
    Workspace,
    Model,
    AgentBase,
}

impl ManagedKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Workspace => "workspace",
            Self::Model => "model",
            Self::AgentBase => "agent-base",
        }
    }
}

fn configured_data_root() -> PathBuf {
    if let Ok(p) = std::env::var("GREPPY_STORE_DIR") {
        let path = PathBuf::from(p);
        return if path.is_absolute() {
            path
        } else {
            std::path::absolute(&path).unwrap_or(path)
        };
    }
    #[cfg(target_os = "macos")]
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home)
            .join("Library")
            .join("Application Support")
            .join("greppy");
    }
    #[cfg(all(target_os = "linux", not(target_os = "android")))]
    {
        if let Some(base) = std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local").join("share"))
            })
        {
            return base.join("greppy");
        }
    }
    #[cfg(target_os = "windows")]
    if let Some(local) = std::env::var_os("LOCALAPPDATA") {
        return PathBuf::from(local).join("greppy");
    }
    std::env::var_os("TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join("greppy")
}

fn resolved_data_root() -> io::Result<PathBuf> {
    let configured = configured_data_root();
    let metadata = match fs::symlink_metadata(&configured) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(configured),
        Err(error) => return Err(error),
    };
    if !metadata.file_type().is_symlink() {
        return Ok(configured);
    }
    let target = fs::canonicalize(&configured).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!(
                "cannot resolve cache root symlink {}: {error}",
                configured.display()
            ),
        )
    })?;
    let target_metadata = fs::symlink_metadata(&target)?;
    if !target_metadata.is_dir() || target_metadata.file_type().is_symlink() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "cache root symlink {} does not resolve to a directory",
                configured.display()
            ),
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let current_uid = unsafe { libc::geteuid() };
        validate_data_root_owner(&configured, &target, target_metadata.uid(), current_uid)?;
    }
    Ok(target)
}

#[cfg(unix)]
fn validate_data_root_owner(
    configured: &Path,
    target: &Path,
    owner_uid: u32,
    current_uid: u32,
) -> io::Result<()> {
    if owner_uid == current_uid {
        return Ok(());
    }
    Err(io::Error::new(
        io::ErrorKind::PermissionDenied,
        format!(
            "cache root symlink {} resolves to {}, which is not owned by the current user",
            configured.display(),
            target.display()
        ),
    ))
}

pub fn data_root() -> PathBuf {
    resolved_data_root().unwrap_or_else(|_| configured_data_root())
}

pub fn workspaces_root() -> PathBuf {
    data_root()
        .join("workspaces")
        .join(format!("v{STORE_FORMAT_VERSION}"))
}

/// Root that holds the user-shared inference artifacts (models, embedding and
/// summary caches). Normally this is [`data_root`]. A child process that is
/// given an isolated `GREPPY_STORE_DIR` (the immutable Base build for a linked
/// worktree stages its graph in a private directory) must still share the
/// user's inference cache and models, otherwise every worktree re-embeds the
/// whole repository from scratch and re-materialises the model into its
/// staging directory. The parent passes its own shared root through this
/// variable; hermetic test and factory runs that set only `GREPPY_STORE_DIR`
/// stay isolated because nothing sets it for them.
pub const ENV_SHARED_INFERENCE_ROOT: &str = "GREPPY_SHARED_INFERENCE_ROOT";

pub fn shared_inference_root() -> PathBuf {
    if let Ok(p) = std::env::var(ENV_SHARED_INFERENCE_ROOT) {
        let path = PathBuf::from(p);
        return if path.is_absolute() {
            path
        } else {
            std::path::absolute(&path).unwrap_or(path)
        };
    }
    data_root()
}

pub fn models_root() -> PathBuf {
    shared_inference_root().join("models").join("v1")
}

/// Staging directories the immutable Base build creates next to the managed
/// stores. Both are `tempfile` directories normally removed after the build;
/// abnormal termination may leave them for lease-checked reclamation.
pub const BASE_BUILD_STAGING_PREFIXES: [&str; 2] =
    ["greppy-base-build-", "greppy-linked-base-checkout-"];

const BASE_BUILD_STAGING_LEASE: &str = "base-build-staging.lease";
/// Internal handoff: an index child retains these leases independently of its
/// parent, including if that parent exits before the child finishes.
pub const ENV_BASE_BUILD_STAGING_LEASES: &str = "GREPPY_BASE_BUILD_STAGING_LEASES";

fn checked_staging_root(path: &Path) -> io::Result<PathBuf> {
    let metadata = fs::symlink_metadata(path)?;
    let known_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| {
            BASE_BUILD_STAGING_PREFIXES
                .iter()
                .any(|prefix| name.starts_with(prefix))
        });
    if !path.is_absolute() || metadata.file_type().is_symlink() || !metadata.is_dir() || !known_name
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid Base build staging directory",
        ));
    }
    fs::canonicalize(path)
}

/// Call immediately after creating the unique staging directory, before any
/// work. Shared leases let an index child take ownership before its parent dies.
pub fn create_base_build_staging_lease(path: &Path) -> io::Result<FileLock> {
    let path = checked_staging_root(path)?;
    acquire_named_lock_in(&path, BASE_BUILD_STAGING_LEASE, LockMode::Shared, false)?
        .ok_or_else(|| io::Error::other("failed to acquire Base build staging lease"))
}

fn existing_staging_lease(path: &Path, mode: LockMode) -> io::Result<Option<FileLock>> {
    let root = checked_staging_root(path)?;
    let locks = root.join("locks");
    let directory = fs::symlink_metadata(&locks)?;
    if !directory.is_dir() || directory.file_type().is_symlink() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid staging lease directory",
        ));
    }
    let path = locks.join(BASE_BUILD_STAGING_LEASE);
    let metadata = fs::symlink_metadata(&path)?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid staging lease file",
        ));
    }
    // Never create a missing lease: legacy/unidentified directories are not
    // proven abandoned, and a child must not resurrect a reclaimed directory.
    let mut options = OpenOptions::new();
    options.read(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW);
    }
    let file = options.open(&path)?;
    if !lock_file(&file, mode, true)? {
        return Ok(None);
    }
    let guard = FileLock { file, path };
    let current = fs::symlink_metadata(&guard.path)?;
    if !current.is_file() || current.file_type().is_symlink() || !root.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "Base build staging was reclaimed",
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let opened = guard.file.metadata()?;
        if (opened.dev(), opened.ino()) != (current.dev(), current.ino()) {
            return Err(io::Error::other("Base build staging lease was replaced"));
        }
    }
    Ok(Some(guard))
}

pub fn retain_base_build_staging_leases_from_env() -> io::Result<Vec<FileLock>> {
    let Some(value) = std::env::var_os(ENV_BASE_BUILD_STAGING_LEASES) else {
        return Ok(Vec::new());
    };
    let mut guards = Vec::new();
    for path in std::env::split_paths(&value) {
        guards.push(
            existing_staging_lease(&path, LockMode::Shared)?.ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::WouldBlock,
                    "Base build staging is being reclaimed",
                )
            })?,
        );
    }
    Ok(guards)
}

/// Reclaim only old staging with a known lease and no live holder. Directory
/// mtime alone cannot prove abandonment: writes inside it do not refresh it.
pub fn reap_stale_base_build_dirs(shared_data_root: &Path, ttl: Duration) -> io::Result<usize> {
    Ok(reap_base_build_staging(shared_data_root, ttl, false)?
        .removed
        .len())
}

fn reap_base_build_staging(
    shared_data_root: &Path,
    ttl: Duration,
    dry_run: bool,
) -> io::Result<GcReport> {
    let mut report = GcReport {
        dry_run,
        ..GcReport::default()
    };
    let Ok(entries) = fs::read_dir(shared_data_root) else {
        return Ok(report);
    };
    let now = SystemTime::now();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if !BASE_BUILD_STAGING_PREFIXES
            .iter()
            .any(|prefix| name.starts_with(prefix))
        {
            continue;
        }
        let Ok(metadata) = fs::symlink_metadata(entry.path()) else {
            continue;
        };
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            continue;
        }
        let stale = metadata
            .modified()
            .ok()
            .and_then(|modified| now.duration_since(modified).ok())
            .is_some_and(|age| age >= ttl);
        if !stale {
            continue;
        }
        let path = entry.path();
        let lease = match existing_staging_lease(&path, LockMode::Exclusive) {
            Ok(Some(lease)) => lease,
            Ok(None) => {
                let bytes = path_size_no_symlink(&path);
                report.scanned_bytes = report.scanned_bytes.saturating_add(bytes);
                report.locked_bytes = report.locked_bytes.saturating_add(bytes);
                report.skipped_locked.push(path);
                continue;
            }
            // Missing legacy ownership, permissions or malformed lease: fail
            // closed. Leave it listed as unmanaged for an explicit audit.
            Err(_) => continue,
        };
        let bytes = path_size_no_symlink(&path);
        report.scanned_bytes = report.scanned_bytes.saturating_add(bytes);
        if dry_run || fs::remove_dir_all(&path).is_ok() {
            report.removed.push(path);
            report.removed_bytes = report.removed_bytes.saturating_add(bytes);
        }
        drop(lease);
    }
    Ok(report)
}

/// Select a physical Base namespace without moving published legacy identities.
/// Existing legacy paths remain authoritative: live readers and CoW descriptors
/// can keep their original graph path and advisory-lock identity.
pub fn agent_base_directory(data: &Path, relative_identity: &Path) -> io::Result<PathBuf> {
    agent_base_directory_for(
        data,
        relative_identity,
        cfg!(target_os = "macos")
            && std::env::var_os("GREPPY_STORE_DIR").is_none()
            && data == data_root(),
        Path::new("/Volumes/tmp"),
    )
}

fn agent_base_directory_for(
    data: &Path,
    identity: &Path,
    disposable: bool,
    volume: &Path,
) -> io::Result<PathBuf> {
    if identity.as_os_str().is_empty()
        || identity
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid relative Base identity",
        ));
    }
    let legacy = data
        .join("agent-base-stores")
        .join(format!("v{AGENT_BASE_FORMAT_VERSION}"))
        .join(identity);
    if !disposable {
        return Ok(legacy);
    }
    // Any retained legacy entry wins, including an incomplete builder directory.
    // Never split old and new writers by silently relocating its identity.
    match fs::symlink_metadata(&legacy) {
        Ok(_) => return Ok(legacy),
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }
    let root = disposable_agent_bases_root(volume);
    ensure_disposable_namespace(&root, volume)?;
    Ok(root.join(identity))
}

fn disposable_agent_bases_root(volume: &Path) -> PathBuf {
    volume
        .join("dev-artifacts/greppy/agent-base-stores")
        .join(format!("v{AGENT_BASE_FORMAT_VERSION}"))
}

/// Base build staging is disposable even when durable model assets live elsewhere.
pub fn base_build_scratch_root() -> io::Result<PathBuf> {
    if cfg!(target_os = "macos") && std::env::var_os("GREPPY_STORE_DIR").is_none() {
        let volume = Path::new("/Volumes/tmp");
        let root = volume.join("dev-artifacts/greppy/base-build-staging");
        ensure_disposable_namespace(&root, volume)?;
        return Ok(root);
    }
    Ok(std::env::var_os("TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir))
}

pub fn agent_base_stores_root() -> PathBuf {
    data_root()
        .join("agent-base-stores")
        .join(format!("v{AGENT_BASE_FORMAT_VERSION}"))
}

/// User-scoped, content-addressed inference results shared by every workspace.
/// An explicit `GREPPY_STORE_DIR` deliberately creates an isolated cache root
/// for hermetic factory and test runs.
pub fn inference_cache_root() -> PathBuf {
    shared_inference_root().join("inference-cache").join("v1")
}

pub fn write_agent_base_manifest(
    dir: &Path,
    identity_hash: &str,
    canonical_repository_identity: &str,
) -> io::Result<()> {
    let directory_name = dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    if !is_hex_id(identity_hash, 64)
        || canonical_repository_identity.trim().is_empty()
        || (directory_name != identity_hash
            && !directory_name.starts_with(&format!(".building-{identity_hash}-")))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid agent Base cache identity",
        ));
    }
    ensure_owned_namespace(dir)?;
    let manifest = AgentBaseManifest {
        format_version: AGENT_BASE_FORMAT_VERSION,
        identity_hash: identity_hash.to_string(),
        canonical_repository_identity: canonical_repository_identity.to_string(),
        created_at_unix_secs: unix_now_secs(),
    };
    atomic_write(
        &dir.join(AGENT_BASE_MANIFEST_FILE),
        &encode_agent_base_manifest(&manifest),
    )
}

pub fn read_agent_base_manifest(dir: &Path) -> io::Result<AgentBaseManifest> {
    let metadata = fs::symlink_metadata(dir)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "agent Base cache entry is not a directory",
        ));
    }
    let manifest = decode_agent_base_manifest(&fs::read(dir.join(AGENT_BASE_MANIFEST_FILE))?)?;
    let directory_name = dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    if manifest.format_version != AGENT_BASE_FORMAT_VERSION
        || (directory_name != manifest.identity_hash
            && !directory_name.starts_with(&format!(".building-{}-", manifest.identity_hash)))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "agent Base cache manifest mismatch",
        ));
    }
    Ok(manifest)
}

pub fn locks_root() -> PathBuf {
    data_root().join("locks")
}

pub fn trash_root() -> PathBuf {
    data_root().join("trash")
}

pub fn workspace_store_dir(workspace_root: &Path) -> PathBuf {
    workspace_stores_root(workspace_root).join(crate::workspace::workspace_hash(workspace_root))
}

/// Route new disposable workspace stores without relocating a live cache.
/// The complete retained directory remains authoritative until an explicit,
/// coordinated migration can account for old clients and all CoW sidecars.
/// Shared model/Base caches and workspace lock identities remain unchanged.
fn workspace_stores_root(workspace_root: &Path) -> PathBuf {
    let durable = workspaces_root();
    workspace_stores_root_for(
        workspace_root,
        &durable,
        &data_root(),
        std::env::var_os("GREPPY_STORE_DIR").is_some(),
        Path::new("/Volumes/tmp"),
    )
}

fn workspace_stores_root_for(
    workspace_root: &Path,
    durable: &Path,
    data: &Path,
    explicit: bool,
    volume: &Path,
) -> PathBuf {
    let hash = crate::workspace::workspace_hash(workspace_root);
    let disposable = disposable_workspaces_root(volume);
    if !explicit
        && (workspace_root.starts_with(volume)
            || canonical_root(workspace_root).starts_with(volume))
        && (!fs::symlink_metadata(disposable.join(&hash))
            .is_err_and(|e| e.kind() == io::ErrorKind::NotFound)
            || (fs::symlink_metadata(durable.join(&hash))
                .is_err_and(|e| e.kind() == io::ErrorKind::NotFound)
                && fs::symlink_metadata(data.join(&hash))
                    .is_err_and(|e| e.kind() == io::ErrorKind::NotFound)))
    {
        disposable
    } else {
        durable.to_path_buf()
    }
}

fn disposable_workspaces_root(volume: &Path) -> PathBuf {
    volume
        .join("dev-artifacts/greppy/workspace-stores")
        .join(format!("v{STORE_FORMAT_VERSION}"))
}

fn validate_existing_disposable_store(dir: &Path, workspace: &Path) -> io::Result<()> {
    match fs::symlink_metadata(dir) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e),
        Ok(md) if md.is_dir() && !md.file_type().is_symlink() => {}
        Ok(_) => {
            return Err(io::Error::other(
                "invalid disposable workspace store directory",
            ))
        }
    }
    match read_store_manifest(dir) {
        Ok(manifest) if manifest.canonical_root == canonical_root(workspace) => Ok(()),
        Ok(_) => Err(io::Error::other(
            "disposable workspace store manifest identity mismatch",
        )),
        // Recover an empty directory left between mkdir and manifest publication,
        // but never adopt unowned graph/sidecar bytes as a new managed store.
        Err(e) if e.kind() == io::ErrorKind::NotFound && fs::read_dir(dir)?.next().is_none() => {
            Ok(())
        }
        Err(e) => Err(io::Error::new(
            e.kind(),
            format!(
                "invalid retained disposable store at {}: {e}",
                dir.display()
            ),
        )),
    }
}

fn ensure_disposable_namespace(root: &Path, volume: &Path) -> io::Result<()> {
    validate_disposable_volume(volume)?;
    ensure_disposable_children(root, volume)
}

fn validate_disposable_volume(volume: &Path) -> io::Result<()> {
    let metadata = fs::symlink_metadata(volume)?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(io::Error::other(
            "disposable cache volume is not a directory",
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let parent = volume
            .parent()
            .ok_or_else(|| io::Error::other("missing volume parent"))?;
        if metadata.dev() == fs::metadata(parent)?.dev() {
            return Err(io::Error::other(
                "disposable cache volume is not mounted; refusing system-disk fallback",
            ));
        }
    }
    Ok(())
}

fn ensure_disposable_children(root: &Path, volume: &Path) -> io::Result<()> {
    let relative = root
        .strip_prefix(volume)
        .map_err(|_| io::Error::other("cache namespace escapes disposable volume"))?;
    let mut current = volume.to_path_buf();
    for component in relative.components() {
        if !matches!(component, std::path::Component::Normal(_)) {
            return Err(io::Error::other("invalid disposable cache namespace"));
        }
        current.push(component.as_os_str());
        ensure_one_directory(&current)?;
    }
    Ok(())
}

pub fn workspace_store_path(workspace_root: &Path) -> PathBuf {
    workspace_store_dir(workspace_root).join("graph.db")
}

pub fn legacy_workspace_store_dir(workspace_root: &Path) -> PathBuf {
    data_root().join(crate::workspace::workspace_hash(workspace_root))
}

pub fn ensure_workspace_store(workspace_root: &Path) -> io::Result<PathBuf> {
    let stores = workspace_stores_root(workspace_root);
    if stores != workspaces_root() {
        let volume = Path::new("/Volumes/tmp");
        if !canonical_root(workspace_root).starts_with(volume) {
            return Err(io::Error::other(
                "disposable workspace resolves outside /Volumes/tmp",
            ));
        }
        ensure_disposable_namespace(&stores, volume)?;
    }
    ensure_owned_namespace(&data_root())?;
    if stores == workspaces_root() {
        ensure_owned_namespace(&stores)?;
    }
    ensure_owned_namespace(&locks_root())?;
    ensure_owned_namespace(&trash_root())?;
    // Hold the selected path throughout initialization; late old-client stores
    // cannot redirect this operation to an unrelated cache.
    let dir = stores.join(crate::workspace::workspace_hash(workspace_root));
    publish_workspace_store(
        &data_root(),
        &dir,
        workspace_root,
        stores != workspaces_root(),
    )?;
    Ok(dir)
}

/// Publication is independent of graph writer ownership: callers may already
/// hold a lifecycle or writer lock. Never acquire those locks while holding
/// this one. GC takes it last, nonblocking, after lifecycle and writer locks.
/// Keep it outside the store so mkdir, atomic manifest staging and GC cannot
/// replace the lock inode while another initializer waits.
fn publish_workspace_store(
    data: &Path,
    dir: &Path,
    workspace_root: &Path,
    disposable: bool,
) -> io::Result<()> {
    let hash = crate::workspace::workspace_hash(workspace_root);
    let _publication = acquire_named_lock_in(
        data,
        &format!("workspace-{hash}.publication"),
        LockMode::Exclusive,
        false,
    )?
    .ok_or_else(|| io::Error::other("blocking workspace publication lock unavailable"))?;
    // Revalidate only after the previous publisher has finished; its temporary
    // manifest file must not be mistaken for nonempty, unowned retained data.
    if disposable {
        validate_existing_disposable_store(dir, workspace_root)?;
    }
    ensure_owned_namespace(dir)?;
    ensure_workspace_manifest(dir, workspace_root)?;
    Ok(())
}

fn ensure_workspace_manifest(dir: &Path, workspace_root: &Path) -> io::Result<()> {
    let expected = StoreManifest {
        format_version: STORE_FORMAT_VERSION,
        workspace_hash: crate::workspace::workspace_hash(workspace_root),
        canonical_root: canonical_root(workspace_root),
        created_at_unix_secs: unix_now_secs(),
    };
    let manifest_path = dir.join(STORE_MANIFEST_FILE);
    match read_store_manifest(dir) {
        Ok(existing)
            if existing.format_version == expected.format_version
                && existing.workspace_hash == expected.workspace_hash
                && existing.canonical_root == expected.canonical_root => {}
        Ok(_) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("workspace store manifest mismatch at {}", dir.display()),
            ));
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            atomic_write(&manifest_path, &encode_manifest(&expected))?;
        }
        Err(e) => return Err(e),
    }
    touch_last_used_dir(dir);
    Ok(())
}

/// Create one model digest directory without following symlinked namespace
/// components. Model names are one plain path component and digests are the
/// 64-hex content IDs used by cache validation and leases.
pub fn ensure_model_entry(model: &str, digest: &str) -> io::Result<PathBuf> {
    if model.is_empty()
        || model == "."
        || model == ".."
        || model.contains(['/', '\\'])
        || !is_hex_id(digest, 64)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid model cache identity",
        ));
    }
    ensure_owned_namespace(&data_root())?;
    ensure_owned_namespace(&models_root())?;
    let model_dir = models_root().join(model);
    ensure_owned_namespace(&model_dir)?;
    let digest_dir = model_dir.join(digest);
    ensure_owned_namespace(&digest_dir)?;
    Ok(digest_dir)
}

pub fn read_store_manifest(dir: &Path) -> io::Result<StoreManifest> {
    let md = fs::symlink_metadata(dir)?;
    if md.file_type().is_symlink() || !md.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("refusing non-directory workspace store {}", dir.display()),
        ));
    }
    let raw = fs::read(dir.join(STORE_MANIFEST_FILE))?;
    let manifest = decode_manifest(&raw)?;
    let dir_hash = dir.file_name().and_then(|s| s.to_str()).unwrap_or_default();
    if manifest.format_version != STORE_FORMAT_VERSION
        || manifest.workspace_hash != dir_hash
        || crate::workspace::workspace_hash(&manifest.canonical_root) != manifest.workspace_hash
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid workspace store manifest at {}", dir.display()),
        ));
    }
    Ok(manifest)
}

pub fn touch_last_used_dir(dir: &Path) {
    if !dir.is_dir() {
        return;
    }
    let marker = dir.join(LAST_USED_FILE);
    if let Ok(modified) = fs::metadata(&marker).and_then(|m| m.modified()) {
        if SystemTime::now()
            .duration_since(modified)
            .is_ok_and(|age| age < LAST_USED_WRITE_GAP)
        {
            return;
        }
    }
    let _ = atomic_write(&marker, unix_now_secs().to_string().as_bytes());
}

pub fn acquire_workspace_lifecycle(
    workspace_root: &Path,
    mode: LockMode,
    nonblocking: bool,
) -> io::Result<Option<FileLock>> {
    let hash = crate::workspace::workspace_hash(workspace_root);
    acquire_named_lock(&format!("workspace-{hash}.lease"), mode, nonblocking)
}

pub fn acquire_workspace_writer(
    workspace_root: &Path,
    nonblocking: bool,
) -> io::Result<Option<FileLock>> {
    let hash = crate::workspace::workspace_hash(workspace_root);
    acquire_named_lock(
        &format!("workspace-{hash}.writer"),
        LockMode::Exclusive,
        nonblocking,
    )
}

pub fn acquire_model_lifecycle(
    model_digest: &str,
    mode: LockMode,
    nonblocking: bool,
) -> io::Result<Option<FileLock>> {
    let safe = sanitize_lock_name(model_digest);
    acquire_named_lock(&format!("model-{safe}.lease"), mode, nonblocking)
}

pub fn acquire_named_lock(
    name: &str,
    mode: LockMode,
    nonblocking: bool,
) -> io::Result<Option<FileLock>> {
    acquire_named_lock_in(&data_root(), name, mode, nonblocking)
}

/// Acquire a cache lock inside an explicit managed data root. Components
/// that construct layouts from an injected root (for example Store-CoW Base
/// publication) must not silently coordinate through the process-global
/// `GREPPY_STORE_DIR` instead.
pub fn acquire_named_lock_in(
    data_root: &Path,
    name: &str,
    mode: LockMode,
    nonblocking: bool,
) -> io::Result<Option<FileLock>> {
    let root = data_root.join("locks");
    ensure_owned_namespace(&root)?;
    let path = root.join(sanitize_lock_name(name));
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&path)?;
    secure_private_file(&path)?;
    match lock_file(&file, mode, nonblocking) {
        Ok(true) => Ok(Some(FileLock { file, path })),
        Ok(false) => Ok(None),
        Err(e) => Err(e),
    }
}

pub fn cache_status() -> io::Result<CacheStatus> {
    let root = data_root();
    let (managed, unmanaged, unmanaged_bytes) = scan_entries(false)?;
    let mut status = CacheStatus {
        data_root: root,
        unmanaged_bytes,
        unmanaged,
        ..CacheStatus::default()
    };
    for entry in managed {
        let lock = try_entry_lock(&entry)?;
        let locked = lock.is_none();
        if locked {
            status.locked_bytes = status.locked_bytes.saturating_add(entry.bytes);
        }
        status.managed_bytes = status.managed_bytes.saturating_add(entry.bytes);
        status.entries.push(CacheEntryStatus {
            kind: entry.kind.as_str().to_string(),
            id: entry.id,
            path: entry.path,
            workspace_root: entry.workspace_root,
            bytes: entry.bytes,
            last_used_unix_secs: system_time_secs(entry.last_used),
            orphaned: entry.orphaned,
            locked,
        });
        drop(lock);
    }
    status
        .entries
        .sort_by_key(|e| (e.last_used_unix_secs, e.kind.clone(), e.id.clone()));
    Ok(status)
}

pub fn maybe_gc(current_workspace_root: Option<&Path>) -> io::Result<GcReport> {
    let Some(_gc_lock) = acquire_named_lock("global.gc", LockMode::Exclusive, true)? else {
        return Ok(GcReport {
            throttled: true,
            ..GcReport::default()
        });
    };
    let policy = GcPolicy::from_env();
    let state = data_root().join("gc.state");
    if let Some(last) = fs::read_to_string(&state)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
    {
        if unix_now_secs().saturating_sub(last) < policy.interval.as_secs() {
            return Ok(GcReport {
                throttled: true,
                ..GcReport::default()
            });
        }
    }
    let report = gc_locked(&policy, false, current_workspace_root)?;
    let _ = atomic_write(&state, unix_now_secs().to_string().as_bytes());
    Ok(report)
}

pub fn run_gc(
    policy: &GcPolicy,
    dry_run: bool,
    current_workspace_root: Option<&Path>,
) -> io::Result<GcReport> {
    let Some(_gc_lock) = acquire_named_lock("global.gc", LockMode::Exclusive, false)? else {
        unreachable!("blocking lock acquisition returned no guard")
    };
    let staging = reap_base_build_staging(&data_root(), BASE_BUILD_STAGING_TTL, dry_run)?;
    let mut report = gc_locked(policy, dry_run, current_workspace_root)?;
    report.scanned_bytes = report.scanned_bytes.saturating_add(staging.scanned_bytes);
    report.removed_bytes = report.removed_bytes.saturating_add(staging.removed_bytes);
    report.locked_bytes = report.locked_bytes.saturating_add(staging.locked_bytes);
    report.removed.extend(staging.removed);
    report.skipped_locked.extend(staging.skipped_locked);
    Ok(report)
}

/// Minimum age for reclaiming a staging directory whose lease is unheld.
/// This age is not a substitute for the process-lifetime lease.
pub const BASE_BUILD_STAGING_TTL: Duration = Duration::from_secs(6 * 60 * 60);

/// Explicitly remove verified cache objects. `workspace_root = Some` clears
/// exactly that canonical worktree; `None` clears every verified workspace
/// and model entry. Active entries are reported as locked and left intact.
pub fn clear_cache(workspace_root: Option<&Path>) -> io::Result<GcReport> {
    let Some(_gc_lock) = acquire_named_lock("global.gc", LockMode::Exclusive, false)? else {
        unreachable!("blocking lock acquisition returned no guard")
    };
    cleanup_trash()?;
    let (entries, _, _) = scan_entries(false)?;
    let requested_hash = workspace_root.map(crate::workspace::workspace_hash);
    let mut report = GcReport::default();
    for entry in entries {
        if let Some(hash) = requested_hash.as_deref() {
            if entry.kind != ManagedKind::Workspace || entry.id != hash {
                continue;
            }
        }
        report.scanned_bytes = report.scanned_bytes.saturating_add(entry.bytes);
        let Some(_lease) = try_entry_lock(&entry)? else {
            report.locked_bytes = report.locked_bytes.saturating_add(entry.bytes);
            report.skipped_locked.push(entry.path);
            continue;
        };
        let trashed = move_to_trash(&entry)?;
        remove_path_no_symlink(&trashed)?;
        report.removed_bytes = report.removed_bytes.saturating_add(entry.bytes);
        report.removed.push(entry.path);
    }
    Ok(report)
}

/// Remove every verified shared Base belonging to one canonical repository
/// identity. This complements workspace-scoped clear: Base stores are shared
/// across linked worktrees and therefore cannot be keyed by one worktree hash.
pub fn clear_agent_bases(canonical_repository_identity: &str) -> io::Result<GcReport> {
    let Some(_gc_lock) = acquire_named_lock("global.gc", LockMode::Exclusive, false)? else {
        unreachable!("blocking lock acquisition returned no guard")
    };
    cleanup_trash()?;
    let (entries, _, _) = scan_entries(false)?;
    let mut report = GcReport::default();
    for entry in entries {
        if entry.kind != ManagedKind::AgentBase {
            continue;
        }
        let Ok(manifest) = read_agent_base_manifest(&entry.path) else {
            continue;
        };
        if manifest.canonical_repository_identity != canonical_repository_identity {
            continue;
        }
        report.scanned_bytes = report.scanned_bytes.saturating_add(entry.bytes);
        let Some(_lease) = try_entry_lock(&entry)? else {
            report.locked_bytes = report.locked_bytes.saturating_add(entry.bytes);
            report.skipped_locked.push(entry.path);
            continue;
        };
        let trashed = move_to_trash(&entry)?;
        remove_path_no_symlink(&trashed)?;
        report.removed_bytes = report.removed_bytes.saturating_add(entry.bytes);
        report.removed.push(entry.path);
    }
    Ok(report)
}

fn gc_locked(
    policy: &GcPolicy,
    dry_run: bool,
    current_workspace_root: Option<&Path>,
) -> io::Result<GcReport> {
    if !dry_run {
        cleanup_trash()?;
    }
    let (mut entries, _, _) = scan_entries(!dry_run)?;
    entries.sort_by_key(|e| e.last_used);
    let current_hash = current_workspace_root.map(crate::workspace::workspace_hash);
    let now = SystemTime::now();
    let mut report = GcReport {
        dry_run,
        ..GcReport::default()
    };
    let mut remaining: u64 = entries.iter().map(|e| e.bytes).sum();
    report.scanned_bytes = remaining;

    // Expiry first, then LRU until the low-water mark. Locked candidates do
    // not reduce `remaining`, so the quota continues through later unlocked
    // entries instead of stopping early on bytes it could not reclaim.
    let mut processed = vec![false; entries.len()];
    for (idx, entry) in entries.iter().enumerate() {
        if current_hash.as_deref() == Some(entry.id.as_str()) {
            continue;
        }
        let age = now
            .duration_since(entry.last_used)
            .unwrap_or(Duration::ZERO);
        let expired = policy.ttl > Duration::ZERO && age > policy.ttl;
        let orphan_expired = entry
            .orphaned_since
            .and_then(|since| now.duration_since(since).ok())
            .is_some_and(|orphan_age| orphan_age > Duration::from_secs(ORPHAN_GRACE_SECS));
        if expired || orphan_expired {
            processed[idx] = true;
            if remove_managed_entry(entry, dry_run, &mut report)? {
                remaining = remaining.saturating_sub(entry.bytes);
            }
        }
    }
    if policy.high_water_bytes > 0 && remaining > policy.high_water_bytes {
        for (idx, entry) in entries.iter().enumerate() {
            if remaining <= policy.low_water_bytes {
                break;
            }
            if processed[idx] || current_hash.as_deref() == Some(entry.id.as_str()) {
                continue;
            }
            processed[idx] = true;
            if remove_managed_entry(entry, dry_run, &mut report)? {
                remaining = remaining.saturating_sub(entry.bytes);
            }
        }
    }
    Ok(report)
}

fn remove_managed_entry(
    entry: &ManagedEntry,
    dry_run: bool,
    report: &mut GcReport,
) -> io::Result<bool> {
    let Some(_lease) = try_entry_lock(entry)? else {
        report.locked_bytes = report.locked_bytes.saturating_add(entry.bytes);
        report.skipped_locked.push(entry.path.clone());
        return Ok(false);
    };
    if dry_run {
        report.removed_bytes = report.removed_bytes.saturating_add(entry.bytes);
        report.removed.push(entry.path.clone());
        return Ok(true);
    }
    let trashed = move_to_trash(entry)?;
    match remove_path_no_symlink(&trashed) {
        Ok(()) => {
            report.removed_bytes = report.removed_bytes.saturating_add(entry.bytes);
            report.removed.push(entry.path.clone());
            Ok(true)
        }
        Err(_) => Ok(false),
    }
}

fn scan_workspace_namespace(
    root: &Path,
    anchor: &Path,
    source_prefix: Option<&Path>,
    mark_new_orphans: bool,
    entries: &mut Vec<ManagedEntry>,
    unmanaged_paths: &mut Vec<PathBuf>,
    unmanaged: &mut u64,
) -> io::Result<()> {
    if !namespace_chain_is_safe_in(root, anchor) {
        if fs::symlink_metadata(root).is_ok() {
            unmanaged_paths.push(root.to_path_buf());
        }
        return Ok(());
    }
    for item in fs::read_dir(root)?.flatten() {
        let path = item.path();
        let manifest = read_store_manifest(&path)
            .ok()
            .filter(|m| source_prefix.is_none_or(|prefix| m.canonical_root.starts_with(prefix)));
        if let Some(manifest) = manifest {
            let last_used = read_last_used(&path).unwrap_or_else(|| {
                fs::metadata(&path)
                    .and_then(|m| m.modified())
                    .unwrap_or(UNIX_EPOCH)
            });
            let orphaned = !manifest.canonical_root.exists();
            let orphaned_since = update_orphan_marker(&path, orphaned, mark_new_orphans);
            entries.push(ManagedEntry {
                kind: ManagedKind::Workspace,
                id: manifest.workspace_hash,
                path: path.clone(),
                workspace_root: Some(manifest.canonical_root),
                bytes: path_size_no_symlink(&path),
                last_used,
                orphaned,
                orphaned_since,
            });
        } else {
            *unmanaged = unmanaged.saturating_add(path_size_no_symlink(&path));
            unmanaged_paths.push(path);
        }
    }
    Ok(())
}

fn scan_entries(mark_new_orphans: bool) -> io::Result<(Vec<ManagedEntry>, Vec<PathBuf>, u64)> {
    let mut entries = Vec::new();
    let mut unmanaged_paths = Vec::new();
    let mut unmanaged = 0u64;
    scan_workspace_namespace(
        &workspaces_root(),
        &data_root(),
        None,
        mark_new_orphans,
        &mut entries,
        &mut unmanaged_paths,
        &mut unmanaged,
    )?;
    if std::env::var_os("GREPPY_STORE_DIR").is_none() {
        let volume = Path::new("/Volumes/tmp");
        let root = disposable_workspaces_root(volume);
        if validate_disposable_volume(volume).is_ok() {
            scan_workspace_namespace(
                &root,
                volume,
                Some(volume),
                mark_new_orphans,
                &mut entries,
                &mut unmanaged_paths,
                &mut unmanaged,
            )?;
        } else if fs::symlink_metadata(&root).is_ok() {
            unmanaged_paths.push(root);
        }
    }
    let models = models_root();
    if namespace_chain_is_safe(&models) {
        let model_dirs = fs::read_dir(&models)?;
        for model_dir in model_dirs.flatten() {
            if !model_dir.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                unmanaged = unmanaged.saturating_add(path_size_no_symlink(&model_dir.path()));
                unmanaged_paths.push(model_dir.path());
                continue;
            }
            if let Ok(digests) = fs::read_dir(model_dir.path()) {
                for digest in digests.flatten() {
                    let path = digest.path();
                    let id = digest.file_name().to_string_lossy().into_owned();
                    if !digest.file_type().map(|t| t.is_dir()).unwrap_or(false)
                        || !is_hex_id(&id, 64)
                        || !model_entry_has_marker(&path, &id)
                    {
                        unmanaged = unmanaged.saturating_add(path_size_no_symlink(&path));
                        unmanaged_paths.push(path);
                        continue;
                    }
                    let last_used = read_last_used(&path).unwrap_or_else(|| {
                        fs::metadata(&path)
                            .and_then(|m| m.modified())
                            .unwrap_or(UNIX_EPOCH)
                    });
                    entries.push(ManagedEntry {
                        kind: ManagedKind::Model,
                        id,
                        path: path.clone(),
                        workspace_root: None,
                        bytes: path_size_no_symlink(&path),
                        last_used,
                        orphaned: false,
                        orphaned_since: None,
                    });
                }
            }
        }
    } else if models.exists() {
        unmanaged_paths.push(models.clone());
    }
    let mut base_roots = vec![(agent_base_stores_root(), data_root())];
    let volume = Path::new("/Volumes/tmp");
    if cfg!(target_os = "macos")
        && std::env::var_os("GREPPY_STORE_DIR").is_none()
        && validate_disposable_volume(volume).is_ok()
    {
        base_roots.push((disposable_agent_bases_root(volume), volume.to_path_buf()));
    }
    for (agent_bases, anchor) in base_roots {
        if namespace_chain_is_safe_in(&agent_bases, &anchor) {
            let repository_dirs = fs::read_dir(&agent_bases)?;
            for repository_dir in repository_dirs.flatten() {
                let repository_path = repository_dir.path();
                let repository_id = repository_dir.file_name().to_string_lossy().into_owned();
                if !repository_dir
                    .file_type()
                    .map(|kind| kind.is_dir())
                    .unwrap_or(false)
                    || !is_hex_id(&repository_id, 64)
                {
                    unmanaged = unmanaged.saturating_add(path_size_no_symlink(&repository_path));
                    unmanaged_paths.push(repository_path);
                    continue;
                }
                let Ok(generations) = fs::read_dir(&repository_path) else {
                    unmanaged = unmanaged.saturating_add(path_size_no_symlink(&repository_path));
                    unmanaged_paths.push(repository_path);
                    continue;
                };
                for generation in generations.flatten() {
                    let path = generation.path();
                    let Ok(manifest) = read_agent_base_manifest(&path) else {
                        unmanaged = unmanaged.saturating_add(path_size_no_symlink(&path));
                        unmanaged_paths.push(path);
                        continue;
                    };
                    let complete = fs::read_to_string(path.join("COMPLETE"))
                        .ok()
                        .is_some_and(|value| value.trim() == manifest.identity_hash);
                    let published = complete && path.join("graph.db").is_file();
                    let last_used = read_last_used(&path).unwrap_or_else(|| {
                        fs::metadata(&path)
                            .and_then(|metadata| metadata.modified())
                            .unwrap_or(UNIX_EPOCH)
                    });
                    let orphaned_since = update_orphan_marker(&path, !published, mark_new_orphans);
                    entries.push(ManagedEntry {
                        kind: ManagedKind::AgentBase,
                        id: manifest.identity_hash,
                        path: path.clone(),
                        workspace_root: None,
                        bytes: path_size_no_symlink(&path),
                        last_used,
                        orphaned: !published,
                        orphaned_since,
                    });
                }
            }
        } else if agent_bases.exists() {
            unmanaged_paths.push(agent_bases.clone());
        }
    }
    // Legacy model layout was `<data>/models/<model>/<digest>`. It is safe to
    // manage only digest directories carrying Greppy's matching marker; every
    // other legacy child remains unmanaged.
    let legacy_models = data_root().join("models");
    if namespace_chain_is_safe(&legacy_models) {
        let model_dirs = fs::read_dir(&legacy_models)?;
        for model_dir in model_dirs.flatten() {
            if model_dir.file_name() == "v1" {
                continue;
            }
            if !model_dir
                .file_type()
                .map(|kind| kind.is_dir())
                .unwrap_or(false)
            {
                let path = model_dir.path();
                unmanaged = unmanaged.saturating_add(path_size_no_symlink(&path));
                unmanaged_paths.push(path);
                continue;
            }
            if let Ok(digests) = fs::read_dir(model_dir.path()) {
                for digest in digests.flatten() {
                    let path = digest.path();
                    let id = digest.file_name().to_string_lossy().into_owned();
                    if digest
                        .file_type()
                        .map(|kind| kind.is_dir())
                        .unwrap_or(false)
                        && is_hex_id(&id, 64)
                        && model_entry_has_marker(&path, &id)
                    {
                        let last_used = read_last_used(&path).unwrap_or_else(|| {
                            fs::metadata(&path)
                                .and_then(|metadata| metadata.modified())
                                .unwrap_or(UNIX_EPOCH)
                        });
                        entries.push(ManagedEntry {
                            kind: ManagedKind::Model,
                            id,
                            path: path.clone(),
                            workspace_root: None,
                            bytes: path_size_no_symlink(&path),
                            last_used,
                            orphaned: false,
                            orphaned_since: None,
                        });
                    } else {
                        unmanaged = unmanaged.saturating_add(path_size_no_symlink(&path));
                        unmanaged_paths.push(path);
                    }
                }
            }
        }
    }
    // Report, but never manage, everything outside the owned namespaces.
    // This includes ambiguous legacy directories and arbitrary operator data
    // under a GREPPY_STORE_DIR override.
    if let Ok(top_level) = fs::read_dir(data_root()) {
        for entry in top_level.flatten() {
            let name = entry.file_name();
            if matches!(
                name.to_str(),
                Some(
                    "workspaces" | "models" | "agent-base-stores" | "locks" | "trash" | "gc.state"
                )
            ) {
                continue;
            }
            let path = entry.path();
            unmanaged = unmanaged.saturating_add(path_size_no_symlink(&path));
            unmanaged_paths.push(path);
        }
    }
    for (root, anchor) in cache_trash_roots() {
        if namespace_chain_is_safe_in(&root, &anchor) {
            for entry in fs::read_dir(&root)?.flatten() {
                let path = entry.path();
                if !trash_entry_is_verified(&path) {
                    unmanaged = unmanaged.saturating_add(path_size_no_symlink(&path));
                    unmanaged_paths.push(path);
                }
            }
        } else if fs::symlink_metadata(&root).is_ok() {
            unmanaged_paths.push(root);
        }
    }
    unmanaged_paths.sort();
    unmanaged_paths.dedup();
    Ok((entries, unmanaged_paths, unmanaged))
}

fn update_orphan_marker(
    store_dir: &Path,
    orphaned: bool,
    create_if_missing: bool,
) -> Option<SystemTime> {
    let marker = store_dir.join(".orphaned_since");
    if !orphaned {
        let _ = fs::remove_file(marker);
        return None;
    }
    if let Some(time) = read_unix_timestamp(&marker) {
        return Some(time);
    }
    if !create_if_missing {
        return None;
    }
    let now = unix_now_secs();
    let _ = atomic_write(&marker, now.to_string().as_bytes());
    Some(UNIX_EPOCH + Duration::from_secs(now))
}

fn read_unix_timestamp(path: &Path) -> Option<SystemTime> {
    fs::read_to_string(path)
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(|seconds| UNIX_EPOCH + Duration::from_secs(seconds))
}

fn workspace_gc_locks_in(data: &Path, id: &str) -> io::Result<Option<Vec<FileLock>>> {
    let Some(lease) = acquire_named_lock_in(
        data,
        &format!("workspace-{id}.lease"),
        LockMode::Exclusive,
        true,
    )?
    else {
        return Ok(None);
    };
    let Some(writer) = acquire_named_lock_in(
        data,
        &format!("workspace-{id}.writer"),
        LockMode::Exclusive,
        true,
    )?
    else {
        return Ok(None);
    };
    let Some(publication) = acquire_named_lock_in(
        data,
        &format!("workspace-{id}.publication"),
        LockMode::Exclusive,
        true,
    )?
    else {
        return Ok(None);
    };
    Ok(Some(vec![lease, writer, publication]))
}

fn try_entry_lock(entry: &ManagedEntry) -> io::Result<Option<Vec<FileLock>>> {
    match entry.kind {
        ManagedKind::Workspace => workspace_gc_locks_in(&data_root(), &entry.id),
        ManagedKind::Model => acquire_model_lifecycle(&entry.id, LockMode::Exclusive, true)
            .map(|lock| lock.map(|lock| vec![lock])),
        ManagedKind::AgentBase => acquire_named_lock(
            &format!("agent-base-{}.builder", entry.id),
            LockMode::Exclusive,
            true,
        )
        .map(|lock| lock.map(|lock| vec![lock])),
    }
}

fn disposable_trash_root(volume: &Path) -> PathBuf {
    volume.join("dev-artifacts/greppy/workspace-stores/trash")
}

fn cache_trash_roots() -> Vec<(PathBuf, PathBuf)> {
    let mut roots = vec![(trash_root(), data_root())];
    let volume = Path::new("/Volumes/tmp");
    if std::env::var_os("GREPPY_STORE_DIR").is_none() && validate_disposable_volume(volume).is_ok()
    {
        roots.push((disposable_trash_root(volume), volume.to_path_buf()));
    }
    roots
}

fn move_to_trash(entry: &ManagedEntry) -> io::Result<PathBuf> {
    let volume = Path::new("/Volumes/tmp");
    let disposable = disposable_workspaces_root(volume);
    let root = if entry.kind == ManagedKind::AgentBase
        && entry.path.starts_with(disposable_agent_bases_root(volume))
    {
        validate_disposable_volume(volume)?;
        if !namespace_chain_is_safe_in(&entry.path, volume)
            || read_agent_base_manifest(&entry.path)?.identity_hash != entry.id
        {
            return Err(io::Error::other(
                "unsafe disposable Base cache during removal",
            ));
        }
        let root = disposable_trash_root(volume);
        ensure_disposable_namespace(&root, volume)?;
        root
    } else if entry.kind == ManagedKind::Workspace
        && entry.path.parent() == Some(disposable.as_path())
    {
        validate_disposable_volume(volume)?;
        if !namespace_chain_is_safe_in(&entry.path, volume) {
            return Err(io::Error::other(
                "unsafe disposable cache namespace during removal",
            ));
        }
        let manifest = read_store_manifest(&entry.path)?;
        if manifest.workspace_hash != entry.id
            || !manifest.canonical_root.starts_with(volume)
            || entry.workspace_root.as_ref() != Some(&manifest.canonical_root)
        {
            return Err(io::Error::other(
                "disposable cache identity changed during removal",
            ));
        }
        let root = disposable_trash_root(volume);
        ensure_disposable_namespace(&root, volume)?;
        root
    } else {
        ensure_owned_namespace(&trash_root())?;
        trash_root()
    };
    rename_entry_to_trash(entry, &root)
}

fn rename_entry_to_trash(entry: &ManagedEntry, root: &Path) -> io::Result<PathBuf> {
    let target = root.join(format!(
        "{}-{}-{}-{}",
        entry.kind.as_str(),
        entry.id,
        std::process::id(),
        unix_now_secs()
    ));
    fs::rename(&entry.path, &target)?;
    Ok(target)
}

fn cleanup_trash() -> io::Result<()> {
    for (root, anchor) in cache_trash_roots() {
        if !namespace_chain_is_safe_in(&root, &anchor) {
            continue;
        }
        for entry in fs::read_dir(&root)?.flatten() {
            let path = entry.path();
            if trash_entry_is_verified(&path) {
                let _ = remove_path_no_symlink(&path);
            }
        }
    }
    Ok(())
}

fn trash_entry_is_verified(path: &Path) -> bool {
    let Ok(metadata) = fs::symlink_metadata(path) else {
        return false;
    };
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return false;
    }
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    if let Some(rest) = name.strip_prefix("workspace-") {
        let Some(hash) = rest.get(..16).filter(|value| is_hex_id(value, 16)) else {
            return false;
        };
        return fs::read(path.join(STORE_MANIFEST_FILE))
            .ok()
            .and_then(|raw| decode_manifest(&raw).ok())
            .is_some_and(|manifest| {
                manifest.workspace_hash == hash
                    && crate::workspace::workspace_hash(&manifest.canonical_root) == hash
            });
    }
    if let Some(rest) = name.strip_prefix("model-") {
        let Some(digest) = rest.get(..64).filter(|value| is_hex_id(value, 64)) else {
            return false;
        };
        return model_entry_has_marker(path, digest);
    }
    if let Some(rest) = name.strip_prefix("agent-base-") {
        let Some(identity_hash) = rest.get(..64).filter(|value| is_hex_id(value, 64)) else {
            return false;
        };
        return read_agent_base_manifest(path)
            .is_ok_and(|manifest| manifest.identity_hash == identity_hash);
    }
    false
}

fn remove_path_no_symlink(path: &Path) -> io::Result<()> {
    let md = fs::symlink_metadata(path)?;
    if md.file_type().is_symlink() || md.is_file() {
        fs::remove_file(path)
    } else if md.is_dir() {
        fs::remove_dir_all(path)
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "unsupported cache entry",
        ))
    }
}

fn read_last_used(dir: &Path) -> Option<SystemTime> {
    let marker = dir.join(LAST_USED_FILE);
    if let Ok(raw) = fs::read_to_string(&marker) {
        if let Ok(secs) = raw.trim().parse::<u64>() {
            return Some(UNIX_EPOCH + Duration::from_secs(secs));
        }
    }
    fs::metadata(marker).and_then(|m| m.modified()).ok()
}

fn model_entry_has_marker(dir: &Path, digest: &str) -> bool {
    fs::read_dir(dir).ok().is_some_and(|rd| {
        rd.flatten().any(|entry| {
            entry
                .file_name()
                .to_str()
                .is_some_and(|n| n.ends_with(".sha256"))
                && fs::read_to_string(entry.path())
                    .map(|s| marker_digest_matches(&s, digest))
                    .unwrap_or(false)
        })
    })
}

/// Both marker formats count as "managed": the bare-hex sidecar, and the
/// CLI's JSON extraction marker (`write_embedded_asset_marker` in
/// crates/cli - serde_json::to_vec, canonical, no whitespace). Without the
/// JSON form, extracted embedded models (~827 MB) were classified unmanaged
/// and survived both `cache gc` and `cache clear --all --yes`.
/// A CLI test pins the `"sha256":"..."` byte shape this match relies on.
fn marker_digest_matches(raw: &str, digest: &str) -> bool {
    let trimmed = raw.trim();
    if trimmed == digest {
        return true;
    }
    trimmed.starts_with('{') && trimmed.contains(&format!("\"sha256\":\"{digest}\""))
}

#[cfg(test)]
mod marker_format_tests {
    use super::marker_digest_matches;

    const D: &str = "1111111111111111111111111111111111111111111111111111111111111111";

    #[test]
    fn bare_hex_marker_matches() {
        assert!(marker_digest_matches(&format!("{D}\n"), D));
    }

    #[test]
    fn cli_json_extraction_marker_matches() {
        let json = format!(
            "{{\"version\":1,\"sha256\":\"{D}\",\"length\":42,\"metadata_fingerprint\":\"x\"}}"
        );
        assert!(marker_digest_matches(&json, D));
    }

    #[test]
    fn wrong_digest_rejected_in_both_formats() {
        let other = "2".repeat(64);
        assert!(!marker_digest_matches(&other, D));
        let json = format!("{{\"version\":1,\"sha256\":\"{other}\"}}");
        assert!(!marker_digest_matches(&json, D));
    }
}

fn path_size_no_symlink(path: &Path) -> u64 {
    let md = match fs::symlink_metadata(path) {
        Ok(md) => md,
        Err(_) => return 0,
    };
    if md.file_type().is_symlink() {
        return 0;
    }
    if md.is_file() {
        return md.len();
    }
    if !md.is_dir() {
        return 0;
    }
    fs::read_dir(path)
        .ok()
        .into_iter()
        .flat_map(|rd| rd.flatten())
        .map(|entry| path_size_no_symlink(&entry.path()))
        .fold(0u64, u64::saturating_add)
}

fn encode_manifest(m: &StoreManifest) -> Vec<u8> {
    let root = m.canonical_root.to_string_lossy();
    format!(
        "{STORE_MANIFEST_MAGIC}\n{}\n{}\n{}\n{}\n",
        m.format_version,
        m.workspace_hash,
        m.created_at_unix_secs,
        hex_encode(root.as_bytes())
    )
    .into_bytes()
}

fn decode_manifest(raw: &[u8]) -> io::Result<StoreManifest> {
    let text = std::str::from_utf8(raw)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "manifest is not UTF-8"))?;
    let mut lines = text.lines();
    if lines.next() != Some(STORE_MANIFEST_MAGIC) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "manifest magic mismatch",
        ));
    }
    let format_version = lines
        .next()
        .and_then(|s| s.parse::<u32>().ok())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid manifest version"))?;
    let workspace_hash = lines
        .next()
        .filter(|s| is_hex_id(s, 16))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid workspace hash"))?
        .to_string();
    let created_at_unix_secs = lines
        .next()
        .and_then(|s| s.parse::<u64>().ok())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid created time"))?;
    let root_hex = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing root"))?;
    let root_bytes = hex_decode(root_hex)?;
    let root = String::from_utf8(root_bytes)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "root is not UTF-8"))?;
    Ok(StoreManifest {
        format_version,
        workspace_hash,
        canonical_root: PathBuf::from(root),
        created_at_unix_secs,
    })
}

fn encode_agent_base_manifest(manifest: &AgentBaseManifest) -> Vec<u8> {
    format!(
        "{AGENT_BASE_MANIFEST_MAGIC}\n{}\n{}\n{}\n{}\n",
        manifest.format_version,
        manifest.identity_hash,
        manifest.created_at_unix_secs,
        hex_encode(manifest.canonical_repository_identity.as_bytes())
    )
    .into_bytes()
}

fn decode_agent_base_manifest(raw: &[u8]) -> io::Result<AgentBaseManifest> {
    let text = std::str::from_utf8(raw)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "manifest is not UTF-8"))?;
    let mut lines = text.lines();
    if lines.next() != Some(AGENT_BASE_MANIFEST_MAGIC) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "agent Base manifest magic mismatch",
        ));
    }
    let format_version = lines
        .next()
        .and_then(|value| value.parse::<u32>().ok())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid manifest version"))?;
    let identity_hash = lines
        .next()
        .filter(|value| is_hex_id(value, 64))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid Base identity hash"))?
        .to_string();
    let created_at_unix_secs = lines
        .next()
        .and_then(|value| value.parse::<u64>().ok())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid created time"))?;
    let repository_identity = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing repository identity"))?;
    let canonical_repository_identity = String::from_utf8(hex_decode(repository_identity)?)
        .map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "repository identity is not UTF-8",
            )
        })?;
    if canonical_repository_identity.trim().is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "repository identity is empty",
        ));
    }
    Ok(AgentBaseManifest {
        format_version,
        identity_hash,
        canonical_repository_identity,
        created_at_unix_secs,
    })
}

fn atomic_write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        ensure_owned_namespace(parent)?;
    }
    let nonce = ATOMIC_WRITE_NONCE.fetch_add(1, Ordering::Relaxed);
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let tmp = path.with_extension(format!(
        "tmp.{}.{}.{}",
        std::process::id(),
        nonce,
        timestamp
    ));
    let mut created = false;
    let result = (|| {
        let mut f = OpenOptions::new().write(true).create_new(true).open(&tmp)?;
        created = true;
        secure_private_file(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        fs::rename(&tmp, path)
    })();
    if created && result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

fn ensure_owned_namespace(dir: &Path) -> io::Result<()> {
    let data = resolved_data_root()?;
    if dir.starts_with(&data) {
        ensure_one_directory(&data)?;
        let mut current = data;
        if let Ok(relative) = dir.strip_prefix(&current) {
            for component in relative.components() {
                current.push(component.as_os_str());
                ensure_one_directory(&current)?;
            }
        }
        return Ok(());
    }
    ensure_one_directory(dir)
}

fn ensure_one_directory(dir: &Path) -> io::Result<()> {
    if let Ok(md) = fs::symlink_metadata(dir) {
        if md.file_type().is_symlink() || !md.is_dir() {
            let reason = if md.file_type().is_symlink() {
                "symlink namespace entries are not allowed; for relocated workspace stores, set GREPPY_STORE_DIR to the real store base directory rather than linking an individual namespace"
            } else {
                "this entry is not a directory; choose a directory for the cache namespace"
            };
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "refusing non-directory cache namespace {}: {reason}",
                    dir.display()
                ),
            ));
        }
    } else {
        fs::create_dir_all(dir)?;
    }
    secure_private_directory(dir)
}

pub fn secure_private_directory(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        // Sandboxed clients may be allowed to read a pre-populated shared
        // cache while writes remain owned by the unsandboxed daemon.  Avoid a
        // redundant chmod in that case: chmod(2) is a write-like operation on
        // macOS and would make otherwise valid embedded assets look missing.
        // A namespace with any broader mode still goes through chmod below.
        if fs::symlink_metadata(path)?.permissions().mode() & 0o777 == 0o700 {
            return Ok(());
        }
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))
    }
    #[cfg(windows)]
    {
        secure_windows_path(path)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = path;
        Ok(())
    }
}

pub fn secure_private_file(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))
    }
    #[cfg(windows)]
    {
        secure_windows_path(path)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = path;
        Ok(())
    }
}

#[cfg(windows)]
fn secure_windows_path(path: &Path) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Foundation::LocalFree;
    use windows_sys::Win32::Security::Authorization::{
        ConvertStringSecurityDescriptorToSecurityDescriptorW, SDDL_REVISION_1,
    };
    use windows_sys::Win32::Security::{
        SetFileSecurityW, DACL_SECURITY_INFORMATION, PROTECTED_DACL_SECURITY_INFORMATION,
        PSECURITY_DESCRIPTOR,
    };

    fn wide(value: &std::ffi::OsStr) -> Vec<u16> {
        value.encode_wide().chain(std::iter::once(0)).collect()
    }

    // Rust accepts long paths, but the raw Win32 ACL API needs the extended
    // absolute form. Resolve before allocating the descriptor so failures do
    // not leak it; every caller secures a file or directory that already exists.
    let path = fs::canonicalize(path)?;
    let descriptor_text = wide(std::ffi::OsStr::new("D:P(A;;FA;;;OW)(A;;FA;;;SY)"));

    let mut descriptor: PSECURITY_DESCRIPTOR = std::ptr::null_mut();
    if unsafe {
        ConvertStringSecurityDescriptorToSecurityDescriptorW(
            descriptor_text.as_ptr(),
            SDDL_REVISION_1,
            &mut descriptor,
            std::ptr::null_mut(),
        )
    } == 0
    {
        return Err(io::Error::last_os_error());
    }
    let path = wide(path.as_os_str());
    let applied = unsafe {
        SetFileSecurityW(
            path.as_ptr(),
            DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
            descriptor,
        )
    };
    unsafe {
        LocalFree(descriptor);
    }
    if applied == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

fn namespace_chain_is_safe(dir: &Path) -> bool {
    namespace_chain_is_safe_in(dir, &data_root())
}

fn namespace_chain_is_safe_in(dir: &Path, anchor: &Path) -> bool {
    let Ok(relative) = dir.strip_prefix(anchor) else {
        return false;
    };
    let mut current = anchor.to_path_buf();
    let safe_dir = |path: &Path| {
        fs::symlink_metadata(path)
            .map(|metadata| !metadata.file_type().is_symlink() && metadata.is_dir())
            .unwrap_or(false)
    };
    if !safe_dir(&current) {
        return false;
    }
    for component in relative.components() {
        if !matches!(component, std::path::Component::Normal(_)) {
            return false;
        }
        current.push(component.as_os_str());
        if !safe_dir(&current) {
            return false;
        }
    }
    true
}

fn canonical_root(root: &Path) -> PathBuf {
    root.canonicalize()
        .or_else(|_| std::path::absolute(root))
        .unwrap_or_else(|_| root.to_path_buf())
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

fn unix_now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn system_time_secs(t: SystemTime) -> u64 {
    t.duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn is_hex_id(s: &str, len: usize) -> bool {
    s.len() == len && s.bytes().all(|b| b.is_ascii_hexdigit())
}

fn sanitize_lock_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '_'
            }
        })
        .take(160)
        .collect()
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(out, "{b:02x}");
    }
    out
}

fn hex_decode(s: &str) -> io::Result<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "odd hex length"));
    }
    (0..s.len())
        .step_by(2)
        .map(|idx| {
            u8::from_str_radix(&s[idx..idx + 2], 16)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid hex"))
        })
        .collect()
}

#[cfg(unix)]
fn lock_file(file: &File, mode: LockMode, nonblocking: bool) -> io::Result<bool> {
    use std::os::fd::AsRawFd;
    const LOCK_SH: i32 = 1;
    const LOCK_EX: i32 = 2;
    const LOCK_NB: i32 = 4;
    let mut op = match mode {
        LockMode::Shared => LOCK_SH,
        LockMode::Exclusive => LOCK_EX,
    };
    if nonblocking {
        op |= LOCK_NB;
    }
    // SAFETY: flock only operates on the valid fd owned by `file`.
    let rc = unsafe { libc_flock(file.as_raw_fd(), op) };
    if rc == 0 {
        return Ok(true);
    }
    let err = io::Error::last_os_error();
    if nonblocking && matches!(err.kind(), io::ErrorKind::WouldBlock) {
        Ok(false)
    } else {
        Err(err)
    }
}

#[cfg(unix)]
fn unlock_file(file: &File) {
    use std::os::fd::AsRawFd;
    const LOCK_UN: i32 = 8;
    // SAFETY: best-effort unlock of the valid fd owned by `file`.
    let _ = unsafe { libc_flock(file.as_raw_fd(), LOCK_UN) };
}

#[cfg(unix)]
extern "C" {
    #[link_name = "flock"]
    fn libc_flock(fd: i32, operation: i32) -> i32;
}

#[cfg(windows)]
fn lock_file(file: &File, mode: LockMode, nonblocking: bool) -> io::Result<bool> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{
        LockFileEx, LOCKFILE_EXCLUSIVE_LOCK, LOCKFILE_FAIL_IMMEDIATELY,
    };
    use windows_sys::Win32::System::IO::OVERLAPPED;

    let mut flags = match mode {
        LockMode::Shared => 0,
        LockMode::Exclusive => LOCKFILE_EXCLUSIVE_LOCK,
    };
    if nonblocking {
        flags |= LOCKFILE_FAIL_IMMEDIATELY;
    }
    let mut overlapped = OVERLAPPED::default();
    let locked = unsafe {
        LockFileEx(
            file.as_raw_handle(),
            flags,
            0,
            u32::MAX,
            u32::MAX,
            &mut overlapped,
        )
    };
    if locked != 0 {
        return Ok(true);
    }
    let error = io::Error::last_os_error();
    if nonblocking && matches!(error.raw_os_error(), Some(32 | 33 | 158)) {
        Ok(false)
    } else {
        Err(error)
    }
}

#[cfg(windows)]
fn unlock_file(file: &File) {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::UnlockFileEx;
    use windows_sys::Win32::System::IO::OVERLAPPED;

    let mut overlapped = OVERLAPPED::default();
    let _ = unsafe { UnlockFileEx(file.as_raw_handle(), 0, u32::MAX, u32::MAX, &mut overlapped) };
}

#[cfg(not(any(unix, windows)))]
fn lock_file(_file: &File, _mode: LockMode, _nonblocking: bool) -> io::Result<bool> {
    Ok(true)
}

#[cfg(not(any(unix, windows)))]
fn unlock_file(_file: &File) {}

#[cfg(test)]
pub(crate) static TEST_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_routing_preserves_isolation_and_live_legacy_identity() {
        let base = tempdir("base-routing");
        let data = base.join("durable");
        let volume = base.join("missing-volume");
        let identity = Path::new("repository/generation");
        let legacy = data.join("agent-base-stores/v1").join(identity);
        assert_eq!(
            agent_base_directory_for(&data, identity, false, &volume).unwrap(),
            legacy
        );
        assert!(agent_base_directory_for(&data, identity, true, &volume).is_err());
        assert!(
            !volume.exists(),
            "must not create a mountpoint on the system disk"
        );
        fs::create_dir_all(&legacy).unwrap();
        fs::write(legacy.join("graph.db"), b"retained graph and vectors").unwrap();
        let lease = acquire_named_lock_in(
            &data,
            "agent-base-generation.builder",
            LockMode::Shared,
            false,
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            agent_base_directory_for(&data, identity, true, &volume).unwrap(),
            legacy
        );
        assert_eq!(
            fs::read(legacy.join("graph.db")).unwrap(),
            b"retained graph and vectors"
        );
        assert!(acquire_named_lock_in(
            &data,
            "agent-base-generation.builder",
            LockMode::Exclusive,
            true
        )
        .unwrap()
        .is_none());
        drop(lease);
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn base_routing_rejects_unmounted_volume_without_fallback() {
        let base = tempdir("base-unmounted");
        let volume = base.join("volume");
        fs::create_dir_all(&volume).unwrap();
        let issue = agent_base_directory_for(
            &base.join("durable"),
            Path::new("repo/generation"),
            true,
            &volume,
        )
        .unwrap_err();
        assert!(issue.to_string().contains("not mounted"));
        assert!(!volume.join("dev-artifacts").exists());
        fs::remove_dir_all(base).unwrap();
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn base_routing_new_identity_uses_designated_physical_volume() {
        let base = tempdir("base-mounted");
        let volume = Path::new("/Volumes/tmp");
        let selected =
            agent_base_directory_for(&base, Path::new("repo/generation"), true, volume).unwrap();
        assert_eq!(
            selected,
            disposable_agent_bases_root(volume).join("repo/generation")
        );
        use std::os::unix::fs::MetadataExt;
        assert_eq!(
            fs::metadata(selected.parent().unwrap().parent().unwrap())
                .unwrap()
                .dev(),
            fs::metadata(volume).unwrap().dev()
        );
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn disposable_store_routing_preserves_retained_data_and_overrides() {
        let _guard = TEST_ENV_LOCK.lock().unwrap();
        let base = tempdir("disposable-routing");
        let volume = base.join("volume");
        let repo = volume.join("worktrees/project/task");
        fs::create_dir_all(&repo).unwrap();
        let data = base.join("durable");
        let durable = data.join("workspaces/v2");
        let route = || workspace_stores_root_for(&repo, &durable, &data, false, &volume);
        assert_eq!(
            route(),
            volume.join("dev-artifacts/greppy/workspace-stores/v2")
        );
        assert_eq!(
            workspace_stores_root_for(&repo, &durable, &data, true, &volume),
            durable
        );
        assert_eq!(
            workspace_stores_root_for(&base.join("canonical"), &durable, &data, false, &volume),
            durable
        );
        let retained = durable.join(crate::workspace::workspace_hash(&repo));
        fs::create_dir_all(retained.join("cow/nested")).unwrap();
        for name in [
            "graph.db",
            "graph.db-wal",
            "embeddings.bin",
            "cow/nested/delta.db",
        ] {
            fs::write(retained.join(name), name.as_bytes()).unwrap();
        }
        assert_eq!(route(), durable);
        // Retaining the entire directory does not need writer quiescence or
        // select a fresh graph when a sidecar/client is still present.
        let lock = acquire_named_lock_in(
            &data,
            &format!(
                "workspace-{}.writer",
                crate::workspace::workspace_hash(&repo)
            ),
            LockMode::Exclusive,
            false,
        )
        .unwrap()
        .unwrap();
        assert_eq!(route(), durable);
        for name in [
            "graph.db",
            "graph.db-wal",
            "embeddings.bin",
            "cow/nested/delta.db",
        ] {
            assert_eq!(fs::read(retained.join(name)).unwrap(), name.as_bytes());
        }
        drop(lock);
        assert_eq!(route(), durable);
        fs::remove_dir_all(&retained).unwrap();
        let legacy = data.join(crate::workspace::workspace_hash(&repo));
        fs::create_dir_all(&legacy).unwrap();
        fs::write(legacy.join("graph.db"), b"legacy").unwrap();
        assert_eq!(route(), durable);
        assert_eq!(fs::read(legacy.join("graph.db")).unwrap(), b"legacy");
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn disposable_namespace_never_creates_missing_or_unmounted_volume() {
        let _guard = TEST_ENV_LOCK.lock().unwrap();
        let base = tempdir("disposable-volume");
        let missing = base.join("missing");
        assert!(ensure_disposable_namespace(&missing.join("stores"), &missing).is_err());
        assert!(!missing.exists());
        #[cfg(unix)]
        {
            let unmounted = base.join("unmounted");
            fs::create_dir(&unmounted).unwrap();
            assert!(ensure_disposable_namespace(&unmounted.join("stores"), &unmounted).is_err());
            assert!(!unmounted.join("stores").exists());
            let alias = base.join("alias");
            std::os::unix::fs::symlink(&unmounted, &alias).unwrap();
            assert!(ensure_disposable_namespace(&alias.join("stores"), &alias).is_err());
            assert!(!unmounted.join("stores").exists());
        }
        fs::remove_dir_all(base).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn private_acl_and_atomic_write_accept_long_existing_paths() {
        use std::os::windows::ffi::OsStrExt;

        let base = tempdir("long-private-acl");
        let mut directory = base.clone();
        for _ in 0..5 {
            directory.push("identity-preserving-cache-component-0123456789abcdef0123456789abcdef");
        }
        assert!(directory.as_os_str().encode_wide().count() > 260);
        fs::create_dir_all(&directory).unwrap();
        secure_private_directory(&directory).unwrap();
        let file = directory.join("manifest.json");
        atomic_write(&file, b"{\"complete\":true}").unwrap();
        secure_private_file(&file).unwrap();
        assert_eq!(fs::read(&file).unwrap(), b"{\"complete\":true}");
        assert_eq!(
            secure_private_file(&directory.join("missing.json"))
                .unwrap_err()
                .kind(),
            io::ErrorKind::NotFound
        );
        fs::remove_dir_all(base).unwrap();
    }

    struct StoreDirRestore(Option<std::ffi::OsString>);

    impl StoreDirRestore {
        fn set(path: &Path) -> Self {
            let previous = std::env::var_os("GREPPY_STORE_DIR");
            unsafe { std::env::set_var("GREPPY_STORE_DIR", path) };
            Self(previous)
        }
    }

    impl Drop for StoreDirRestore {
        fn drop(&mut self) {
            unsafe {
                match self.0.take() {
                    Some(previous) => std::env::set_var("GREPPY_STORE_DIR", previous),
                    None => std::env::remove_var("GREPPY_STORE_DIR"),
                }
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn disposable_namespace_rejects_symlinks_and_routes_workspace_aliases() {
        let _guard = TEST_ENV_LOCK.lock().unwrap();
        let base = tempdir("disposable-symlink");
        let volume = base.join("volume");
        let outside = base.join("outside");
        fs::create_dir_all(volume.join("worktrees/repo")).unwrap();
        fs::create_dir_all(&outside).unwrap();
        std::os::unix::fs::symlink(&outside, volume.join("dev-artifacts")).unwrap();
        assert!(ensure_disposable_children(&volume.join("dev-artifacts/stores"), &volume).is_err());
        assert!(!outside.join("stores").exists());
        let alias = base.join("repo-alias");
        std::os::unix::fs::symlink(volume.join("worktrees/repo"), &alias).unwrap();
        let durable = base.join("durable/workspaces/v2");
        assert_eq!(
            workspace_stores_root_for(&alias, &durable, &base.join("durable"), false, &volume),
            volume.join("dev-artifacts/greppy/workspace-stores/v2")
        );
        assert!(ensure_disposable_children(&base.join("escape"), &volume).is_err());
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn disposable_store_selection_is_stable_in_both_creation_orders() {
        let _guard = TEST_ENV_LOCK.lock().unwrap();
        let base = tempdir("disposable-orders");
        let volume = base.join("volume");
        let data = base.join("data");
        let durable = data.join("workspaces/v2");
        for old_first in [false, true] {
            let repo = volume.join(if old_first { "old-first" } else { "tmp-first" });
            fs::create_dir_all(&repo).unwrap();
            let hash = crate::workspace::workspace_hash(&repo);
            let old = durable.join(&hash);
            let tmp_root = disposable_workspaces_root(&volume);
            let tmp = tmp_root.join(&hash);
            let lookup =
                || workspace_stores_root_for(&repo, &durable, &data, false, &volume).join(&hash);
            if old_first {
                fs::create_dir_all(&old).unwrap();
                ensure_workspace_manifest(&old, &repo).unwrap();
                assert_eq!(lookup(), old);
            } else {
                assert_eq!(lookup(), tmp);
            }
            // The exact same manifest helper is used by ensure_workspace_store.
            fs::create_dir_all(&tmp).unwrap();
            validate_existing_disposable_store(&tmp, &repo).unwrap();
            ensure_workspace_manifest(&tmp, &repo).unwrap();
            fs::write(tmp.join("graph.db"), b"selected tmp graph").unwrap();
            if !old_first {
                fs::create_dir_all(&old).unwrap();
                ensure_workspace_manifest(&old, &repo).unwrap();
            }
            fs::write(old.join("graph.db"), b"retained old graph").unwrap();
            assert_eq!(lookup(), tmp);
            validate_existing_disposable_store(&tmp, &repo).unwrap();
            ensure_workspace_manifest(&tmp, &repo).unwrap();
            assert_eq!(lookup(), tmp);
            assert_eq!(
                fs::read(tmp.join("graph.db")).unwrap(),
                b"selected tmp graph"
            );
            assert_eq!(
                fs::read(old.join("graph.db")).unwrap(),
                b"retained old graph"
            );
        }
        let repo = volume.join("unowned");
        fs::create_dir_all(&repo).unwrap();
        let tmp = disposable_workspaces_root(&volume).join(crate::workspace::workspace_hash(&repo));
        fs::create_dir_all(&tmp).unwrap();
        fs::write(tmp.join("graph.db"), b"unowned bytes").unwrap();
        assert!(validate_existing_disposable_store(&tmp, &repo).is_err());
        assert!(!tmp.join(STORE_MANIFEST_FILE).exists());
        fs::write(tmp.join(STORE_MANIFEST_FILE), b"invalid manifest").unwrap();
        assert!(validate_existing_disposable_store(&tmp, &repo).is_err());
        assert_eq!(fs::read(tmp.join("graph.db")).unwrap(), b"unowned bytes");
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn disposable_inventory_ownership_and_gc_locks_preserve_foreign_entries() {
        let _guard = TEST_ENV_LOCK.lock().unwrap();
        let base = tempdir("disposable-inventory");
        let volume = base.join("volume");
        let stores = disposable_workspaces_root(&volume);
        let repo = volume.join("repo");
        let foreign_repo = base.join("foreign-repo");
        fs::create_dir_all(&repo).unwrap();
        fs::create_dir_all(&foreign_repo).unwrap();
        let hash = crate::workspace::workspace_hash(&repo);
        let store = stores.join(&hash);
        fs::create_dir_all(store.join("cow")).unwrap();
        ensure_workspace_manifest(&store, &repo).unwrap();
        fs::write(store.join("graph.db"), b"graph").unwrap();
        fs::write(store.join("cow/delta.db"), b"delta").unwrap();
        let foreign = stores.join(crate::workspace::workspace_hash(&foreign_repo));
        fs::create_dir_all(&foreign).unwrap();
        ensure_workspace_manifest(&foreign, &foreign_repo).unwrap();
        fs::write(foreign.join("graph.db"), b"foreign").unwrap();
        let unowned = stores.join("unowned");
        fs::create_dir_all(&unowned).unwrap();
        fs::write(unowned.join("sentinel"), b"keep").unwrap();
        let mut entries = Vec::new();
        let mut unmanaged = Vec::new();
        let mut unmanaged_bytes = 0;
        scan_workspace_namespace(
            &stores,
            &volume,
            Some(&volume),
            false,
            &mut entries,
            &mut unmanaged,
            &mut unmanaged_bytes,
        )
        .unwrap();
        let data = base.join("data");
        let durable = data.join("workspaces/v2");
        let old_store = durable.join(&hash);
        fs::create_dir_all(&old_store).unwrap();
        ensure_workspace_manifest(&old_store, &repo).unwrap();
        fs::write(old_store.join("graph.db"), b"old graph").unwrap();
        scan_workspace_namespace(
            &durable,
            &data,
            None,
            false,
            &mut entries,
            &mut unmanaged,
            &mut unmanaged_bytes,
        )
        .unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].path, store);
        assert_eq!(entries[1].path, old_store);
        assert!(entries[0].bytes >= 10);
        assert!(unmanaged.contains(&foreign));
        assert!(unmanaged.contains(&unowned));
        let writer = acquire_named_lock_in(
            &data,
            &format!("workspace-{hash}.writer"),
            LockMode::Exclusive,
            false,
        )
        .unwrap()
        .unwrap();
        assert!(workspace_gc_locks_in(&data, &hash).unwrap().is_none());
        assert_eq!(fs::read(store.join("graph.db")).unwrap(), b"graph");
        drop(writer);
        let lease = acquire_named_lock_in(
            &data,
            &format!("workspace-{hash}.lease"),
            LockMode::Shared,
            false,
        )
        .unwrap()
        .unwrap();
        assert!(workspace_gc_locks_in(&data, &hash).unwrap().is_none());
        drop(lease);
        let _gc_locks = workspace_gc_locks_in(&data, &hash).unwrap().unwrap();
        let trash = disposable_trash_root(&volume);
        ensure_disposable_children(&trash, &volume).unwrap();
        let trashed = rename_entry_to_trash(&entries[0], &trash).unwrap();
        assert!(trashed.starts_with(&volume));
        assert!(trash_entry_is_verified(&trashed));
        assert_eq!(fs::read(trashed.join("cow/delta.db")).unwrap(), b"delta");
        remove_path_no_symlink(&trashed).unwrap();
        assert!(!store.exists());
        assert_eq!(fs::read(old_store.join("graph.db")).unwrap(), b"old graph");
        assert_eq!(fs::read(foreign.join("graph.db")).unwrap(), b"foreign");
        assert_eq!(fs::read(unowned.join("sentinel")).unwrap(), b"keep");
        drop(_gc_locks);
        fs::remove_dir_all(base).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn disposable_inventory_does_not_follow_namespace_symlinks() {
        let _guard = TEST_ENV_LOCK.lock().unwrap();
        let base = tempdir("disposable-inventory-alias");
        let volume = base.join("volume");
        let outside = base.join("outside");
        fs::create_dir_all(&volume).unwrap();
        fs::create_dir_all(&outside).unwrap();
        fs::write(outside.join("sentinel"), b"keep").unwrap();
        std::os::unix::fs::symlink(&outside, volume.join("dev-artifacts")).unwrap();
        let mut entries = Vec::new();
        let mut unmanaged = Vec::new();
        let mut bytes = 0;
        scan_workspace_namespace(
            &disposable_workspaces_root(&volume),
            &volume,
            Some(&volume),
            true,
            &mut entries,
            &mut unmanaged,
            &mut bytes,
        )
        .unwrap();
        assert!(entries.is_empty());
        assert_eq!(bytes, 0);
        assert_eq!(fs::read(outside.join("sentinel")).unwrap(), b"keep");
        assert!(!outside.join(".orphaned_since").exists());
        fs::remove_dir_all(base).unwrap();
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn disposable_public_lookup_and_ensure_agree_after_late_old_store() {
        let _guard = TEST_ENV_LOCK.lock().unwrap();
        if validate_disposable_volume(Path::new("/Volumes/tmp")).is_err() {
            return; // Portable runners have no designated mounted tmp volume.
        }
        struct RestoreEnv(Vec<(&'static str, Option<std::ffi::OsString>)>);
        impl Drop for RestoreEnv {
            fn drop(&mut self) {
                for (name, old) in &self.0 {
                    if let Some(value) = old {
                        std::env::set_var(name, value);
                    } else {
                        std::env::remove_var(name);
                    }
                }
            }
        }
        let _restore = RestoreEnv(
            ["HOME", "GREPPY_STORE_DIR"]
                .into_iter()
                .map(|name| (name, std::env::var_os(name)))
                .collect(),
        );
        let base = tempdir("disposable-public");
        assert!(
            base.starts_with("/Volumes/tmp"),
            "fixture TMPDIR must use disposable volume"
        );
        std::env::set_var("HOME", base.join("home"));
        std::env::remove_var("GREPPY_STORE_DIR");
        let repo = base.join("repo");
        fs::create_dir_all(&repo).unwrap();
        let selected = workspace_store_dir(&repo);
        assert!(selected.starts_with(disposable_workspaces_root(Path::new("/Volumes/tmp"))));
        assert!(
            !selected.exists(),
            "fixture must own a fresh workspace identity"
        );
        assert_eq!(ensure_workspace_store(&repo).unwrap(), selected);
        fs::write(selected.join("graph.db"), b"selected").unwrap();
        let old = workspaces_root().join(crate::workspace::workspace_hash(&repo));
        fs::create_dir_all(&old).unwrap();
        ensure_workspace_manifest(&old, &repo).unwrap();
        fs::write(old.join("graph.db"), b"old").unwrap();
        assert_eq!(workspace_store_dir(&repo), selected);
        assert_eq!(ensure_workspace_store(&repo).unwrap(), selected);
        assert_eq!(workspace_store_path(&repo), selected.join("graph.db"));
        assert_eq!(fs::read(selected.join("graph.db")).unwrap(), b"selected");
        assert_eq!(fs::read(old.join("graph.db")).unwrap(), b"old");
        assert_eq!(
            read_store_manifest(&selected).unwrap().canonical_root,
            repo.canonicalize().unwrap()
        );
        fs::remove_dir_all(selected).unwrap();
        fs::remove_dir_all(base).unwrap();
    }

    fn tempdir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "greppy-cache-{tag}-{}-{}",
            std::process::id(),
            unix_now_secs()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        // Match production namespace anchors even when TMPDIR is a short alias.
        dir.canonicalize().unwrap()
    }

    #[test]
    fn absent_data_root_is_created_for_named_locks() {
        let _guard = TEST_ENV_LOCK.lock().unwrap();
        let base = tempdir("absent-data-root");
        let root = base.join("new-data-root");
        let _restore = StoreDirRestore::set(&root);

        let lock = acquire_named_lock("startup", LockMode::Exclusive, true)
            .unwrap()
            .unwrap();
        assert!(root.join("locks").is_dir());
        assert_eq!(lock.path(), root.join("locks/startup"));
        drop(lock);
        let _ = fs::remove_dir_all(base);
    }

    #[cfg(unix)]
    #[test]
    fn owned_data_root_symlink_is_resolved_but_descendant_symlinks_are_rejected() {
        use std::os::unix::fs::symlink;

        let _guard = TEST_ENV_LOCK.lock().unwrap();
        let base = tempdir("data-root-symlink");
        let target = base.join("nvme-store");
        fs::create_dir(&target).unwrap();
        let root = base.join("greppy-store");
        symlink(&target, &root).unwrap();
        let _restore = StoreDirRestore::set(&root);

        let lock = acquire_named_lock("startup", LockMode::Exclusive, true)
            .unwrap()
            .unwrap();
        assert_eq!(lock.path(), target.join("locks/startup"));
        drop(lock);

        fs::remove_dir_all(target.join("locks")).unwrap();
        let external = base.join("external-locks");
        fs::create_dir(&external).unwrap();
        symlink(&external, target.join("locks")).unwrap();
        let error = acquire_named_lock("blocked", LockMode::Exclusive, true).unwrap_err();
        assert!(error
            .to_string()
            .contains("refusing non-directory cache namespace"));
        assert!(error
            .to_string()
            .contains("symlink namespace entries are not allowed"));
        assert!(error.to_string().contains("GREPPY_STORE_DIR"));
        assert!(
            !external.join("blocked").exists(),
            "refused symlink target was modified"
        );
        let _ = fs::remove_dir_all(base);
    }

    #[cfg(unix)]
    #[test]
    fn data_root_symlink_rejects_non_directory_and_unowned_targets() {
        use std::os::unix::fs::symlink;

        let _guard = TEST_ENV_LOCK.lock().unwrap();
        let base = tempdir("invalid-data-root-symlink");
        let file = base.join("not-a-directory");
        fs::write(&file, b"data").unwrap();
        let root = base.join("greppy-store");
        symlink(&file, &root).unwrap();
        let _restore = StoreDirRestore::set(&root);
        let error = acquire_named_lock("blocked", LockMode::Exclusive, true).unwrap_err();
        assert!(error
            .to_string()
            .contains("does not resolve to a directory"));

        let error = validate_data_root_owner(&root, &file, 1000, 1001).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
        assert!(error.to_string().contains("not owned by the current user"));
        let _ = fs::remove_dir_all(base);
    }

    #[cfg(windows)]
    fn set_directory_modified(path: &Path, modified: SystemTime) {
        use std::os::windows::fs::OpenOptionsExt;

        const FILE_WRITE_ATTRIBUTES: u32 = 0x0100;
        const FILE_SHARE_READ_WRITE_DELETE: u32 = 0x0001 | 0x0002 | 0x0004;
        const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;

        fs::OpenOptions::new()
            .access_mode(FILE_WRITE_ATTRIBUTES)
            .share_mode(FILE_SHARE_READ_WRITE_DELETE)
            .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
            .open(path)
            .unwrap()
            .set_modified(modified)
            .unwrap();
    }

    #[cfg(not(windows))]
    fn set_directory_modified(path: &Path, modified: SystemTime) {
        File::open(path).unwrap().set_modified(modified).unwrap();
    }

    #[test]
    fn versioned_store_has_valid_manifest() {
        let _guard = TEST_ENV_LOCK.lock().unwrap();
        let base = tempdir("manifest");
        let repo = base.join("repo");
        fs::create_dir_all(&repo).unwrap();
        std::env::set_var("GREPPY_STORE_DIR", base.join("data"));
        let dir = ensure_workspace_store(&repo).unwrap();
        assert!(dir.ends_with(crate::workspace::workspace_hash(&repo)));
        assert_eq!(
            dir.parent(),
            Some(base.join("data").join("workspaces").join("v2").as_path())
        );
        let m = read_store_manifest(&dir).unwrap();
        assert_eq!(m.canonical_root, repo.canonicalize().unwrap());
        std::env::remove_var("GREPPY_STORE_DIR");
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn concurrent_absent_workspace_store_publication_is_stable() {
        let _guard = TEST_ENV_LOCK.lock().unwrap();
        let base = tempdir("concurrent-manifest");
        let repo = base.join("repo");
        fs::create_dir_all(&repo).unwrap();
        let data = base.join("data");
        let _restore = StoreDirRestore::set(&data);
        let start = std::sync::Arc::new(std::sync::Barrier::new(9));
        let handles = (0..8)
            .map(|_| {
                let start = std::sync::Arc::clone(&start);
                let repo = repo.clone();
                std::thread::spawn(move || {
                    start.wait();
                    ensure_workspace_store(&repo)
                })
            })
            .collect::<Vec<_>>();

        start.wait();
        for handle in handles {
            handle
                .join()
                .expect("concurrent workspace store publisher panicked")
                .unwrap();
        }
        let dir = workspace_store_dir(&repo);
        let manifest = read_store_manifest(&dir).unwrap();
        assert_eq!(manifest.canonical_root, repo.canonicalize().unwrap());
        assert_eq!(
            manifest.workspace_hash,
            crate::workspace::workspace_hash(&repo)
        );
        let _ = fs::remove_dir_all(base);
    }

    // Invoked only by the controlled cross-process contention test below.
    #[test]
    fn workspace_publication_child() {
        let Some(base) = std::env::var_os("GREPPY_PUBLICATION_TEST_ROOT") else {
            return;
        };
        let base = PathBuf::from(base);
        let repo = base.join("repo");
        let data = base.join("data");
        let hash = crate::workspace::workspace_hash(&repo);
        assert!(acquire_named_lock_in(
            &data,
            &format!("workspace-{hash}.publication"),
            LockMode::Exclusive,
            true,
        )
        .unwrap()
        .is_none());
        fs::write(base.join("contender-ready"), b"ready").unwrap();
        publish_workspace_store(&data, &base.join(&hash), &repo, true).unwrap();
        fs::write(base.join("contender-done"), b"done").unwrap();
    }

    #[test]
    fn disposable_workspace_publication_waits_for_atomic_manifest_and_blocks_gc() {
        let base = tempdir("publication-contention");
        let repo = base.join("repo");
        let data = base.join("data");
        fs::create_dir_all(&repo).unwrap();
        let hash = crate::workspace::workspace_hash(&repo);
        let dir = base.join(&hash);
        let publication = acquire_named_lock_in(
            &data,
            &format!("workspace-{hash}.publication"),
            LockMode::Exclusive,
            false,
        )
        .unwrap()
        .unwrap();
        fs::create_dir(&dir).unwrap();
        // Hold the exact dangerous state: mkdir has completed and the atomic
        // manifest temporary file exists, but the manifest is not published.
        let staging = dir.join("store.manifest.controlled.tmp");
        fs::write(&staging, b"manifest staging").unwrap();
        assert!(validate_existing_disposable_store(&dir, &repo).is_err());
        assert!(workspace_gc_locks_in(&data, &hash).unwrap().is_none());
        // GC's failed final acquisition must release lifecycle and writer locks.
        let lease = acquire_named_lock_in(
            &data,
            &format!("workspace-{hash}.lease"),
            LockMode::Exclusive,
            true,
        )
        .unwrap()
        .unwrap();
        let writer = acquire_named_lock_in(
            &data,
            &format!("workspace-{hash}.writer"),
            LockMode::Exclusive,
            true,
        )
        .unwrap()
        .unwrap();
        drop((lease, writer));
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("cache::tests::workspace_publication_child")
            .arg("--exact")
            .arg("--test-threads=1")
            .env("GREPPY_PUBLICATION_TEST_ROOT", &base)
            .spawn()
            .unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while !base.join("contender-ready").exists() {
            if std::time::Instant::now() >= deadline || child.try_wait().unwrap().is_some() {
                let _ = child.kill();
                let _ = child.wait();
                panic!("publication contender did not reach the held lock");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(!base.join("contender-done").exists());
        ensure_workspace_manifest(&dir, &repo).unwrap();
        fs::remove_file(staging).unwrap();
        drop(publication);
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break status;
            }
            if std::time::Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                panic!("publication contender did not finish after publication");
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        assert!(status.success());
        assert!(base.join("contender-done").exists());
        assert_eq!(
            read_store_manifest(&dir).unwrap().canonical_root,
            canonical_root(&repo)
        );
        assert!(workspace_gc_locks_in(&data, &hash).unwrap().is_some());
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn disposable_workspace_publication_rejects_unowned_bytes_and_wrong_identity() {
        let base = tempdir("publication-foreign");
        let repo = base.join("repo");
        fs::create_dir_all(&repo).unwrap();
        let data = base.join("data");
        let dir = base.join(crate::workspace::workspace_hash(&repo));
        fs::create_dir(&dir).unwrap();
        // A leftover atomic-looking name is not proof of ownership either.
        for name in ["graph.db", "store.manifest.foreign.tmp"] {
            let foreign = dir.join(name);
            fs::write(&foreign, b"unowned bytes").unwrap();
            assert!(publish_workspace_store(&data, &dir, &repo, true).is_err());
            assert!(!dir.join(STORE_MANIFEST_FILE).exists());
            assert_eq!(fs::read(&foreign).unwrap(), b"unowned bytes");
            fs::remove_file(foreign).unwrap();
        }
        // An abandoned empty directory is still safely recoverable.
        publish_workspace_store(&data, &dir, &repo, true).unwrap();
        let other = base.join("other-repo");
        fs::create_dir_all(&other).unwrap();
        let manifest = fs::read(dir.join(STORE_MANIFEST_FILE)).unwrap();
        assert!(publish_workspace_store(&data, &dir, &other, true).is_err());
        assert_eq!(fs::read(dir.join(STORE_MANIFEST_FILE)).unwrap(), manifest);
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn shared_inference_root_follows_store_dir_unless_overridden() {
        let _guard = TEST_ENV_LOCK.lock().unwrap();
        let base = tempdir("shared-inference");
        let staging = base.join("staging");
        let shared = base.join("shared");
        std::env::set_var("GREPPY_STORE_DIR", &staging);
        std::env::remove_var(ENV_SHARED_INFERENCE_ROOT);
        // Without the override an isolated store keeps an isolated cache.
        assert_eq!(
            inference_cache_root(),
            staging.join("inference-cache").join("v1")
        );
        assert_eq!(models_root(), staging.join("models").join("v1"));
        // The Base build child gets the parent's shared root and must use it
        // for models and inference caches while its graph stays in staging.
        std::env::set_var(ENV_SHARED_INFERENCE_ROOT, &shared);
        assert_eq!(data_root(), staging);
        assert_eq!(
            inference_cache_root(),
            shared.join("inference-cache").join("v1")
        );
        assert_eq!(models_root(), shared.join("models").join("v1"));
        std::env::remove_var(ENV_SHARED_INFERENCE_ROOT);
        std::env::remove_var("GREPPY_STORE_DIR");
        let _ = fs::remove_dir_all(base);
    }

    #[cfg(unix)]
    #[test]
    fn existing_private_model_namespace_does_not_require_permission_rewrite() {
        use std::os::unix::fs::PermissionsExt;

        let _guard = TEST_ENV_LOCK.lock().unwrap();
        let base = tempdir("existing-private-model");
        let data = base.join("data");
        std::env::set_var("GREPPY_STORE_DIR", &data);
        let entry = ensure_model_entry("model", &"a".repeat(64)).unwrap();
        fs::set_permissions(&data, fs::Permissions::from_mode(0o700)).unwrap();
        fs::set_permissions(data.join("models"), fs::Permissions::from_mode(0o700)).unwrap();
        fs::set_permissions(
            data.join("models").join("v1"),
            fs::Permissions::from_mode(0o700),
        )
        .unwrap();
        fs::set_permissions(entry.parent().unwrap(), fs::Permissions::from_mode(0o700)).unwrap();
        fs::set_permissions(&entry, fs::Permissions::from_mode(0o700)).unwrap();

        assert_eq!(ensure_model_entry("model", &"a".repeat(64)).unwrap(), entry);
        assert_eq!(
            fs::metadata(&entry).unwrap().permissions().mode() & 0o777,
            0o700
        );

        std::env::remove_var("GREPPY_STORE_DIR");
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn reaper_removes_only_stale_base_build_staging() {
        let base = tempdir("reaper");
        let stale_build = base.join("greppy-base-build-abc123");
        let stale_checkout = base.join("greppy-linked-base-checkout-def456");
        let fresh_build = base.join("greppy-base-build-fresh");
        let unrelated = base.join("do-not-delete-greppy-base-build-");
        for dir in [&stale_build, &stale_checkout, &fresh_build, &unrelated] {
            fs::create_dir_all(dir.join("data")).unwrap();
            fs::write(dir.join("data").join("graph.db"), b"x").unwrap();
        }
        let old = SystemTime::now() - Duration::from_secs(7 * 60 * 60);
        for dir in [&stale_build, &stale_checkout] {
            // Explicitly known, completed owners. Age alone is not ownership.
            drop(create_base_build_staging_lease(dir).unwrap());
            set_directory_modified(dir, old);
        }
        let removed = reap_stale_base_build_dirs(&base, BASE_BUILD_STAGING_TTL).unwrap();
        assert_eq!(removed, 2);
        assert!(!stale_build.exists());
        assert!(!stale_checkout.exists());
        assert!(fresh_build.is_dir(), "a live build must survive");
        assert!(unrelated.is_dir(), "only the exact prefixes are reaped");
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn reaper_preserves_active_and_unidentified_old_staging() {
        let base = tempdir("reaper-live");
        let active = base.join("greppy-base-build-active");
        let legacy = base.join("greppy-base-build-legacy");
        for path in [&active, &legacy] {
            fs::create_dir_all(path.join("data")).unwrap();
        }
        let lease = create_base_build_staging_lease(&active).unwrap();
        let old = SystemTime::now() - Duration::from_secs(7 * 60 * 60);
        for path in [&active, &legacy] {
            set_directory_modified(path, old);
            let before = fs::metadata(path).unwrap().modified().unwrap();
            fs::write(path.join("data/graph.db"), b"live output").unwrap();
            assert_eq!(
                fs::metadata(path).unwrap().modified().unwrap(),
                before,
                "writing a descendant must not be mistaken for a root heartbeat"
            );
        }
        let report = reap_base_build_staging(&base, BASE_BUILD_STAGING_TTL, false).unwrap();
        assert!(report.removed.is_empty());
        assert_eq!(report.skipped_locked, vec![active.clone()]);
        assert!(active.join("data/graph.db").is_file());
        assert!(legacy.join("data/graph.db").is_file());
        drop(lease);
        let dry = reap_base_build_staging(&base, BASE_BUILD_STAGING_TTL, true).unwrap();
        assert_eq!(dry.removed, vec![active.clone()]);
        assert!(dry.removed_bytes >= b"live output".len() as u64);
        assert!(active.is_dir(), "dry-run must retain the candidate");
        let actual = reap_base_build_staging(&base, BASE_BUILD_STAGING_TTL, false).unwrap();
        assert_eq!(actual.removed, dry.removed);
        assert_eq!(actual.removed_bytes, dry.removed_bytes);
        assert!(!active.exists());
        assert!(
            legacy.is_dir(),
            "unknown legacy ownership remains fail-closed"
        );
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn staging_lease_requires_existing_regular_ownership() {
        let base = tempdir("reaper-invalid-lease");
        let staging = base.join("greppy-base-build-invalid");
        fs::create_dir_all(&staging).unwrap();
        assert!(existing_staging_lease(&staging, LockMode::Shared).is_err());
        assert!(
            !staging.join("locks").exists(),
            "retaining must not create ownership"
        );
        let lease_path = staging.join("locks").join(BASE_BUILD_STAGING_LEASE);
        fs::create_dir_all(&lease_path).unwrap();
        set_directory_modified(
            &staging,
            SystemTime::now() - Duration::from_secs(7 * 60 * 60),
        );
        assert!(existing_staging_lease(&staging, LockMode::Shared).is_err());
        assert_eq!(
            reap_stale_base_build_dirs(&base, BASE_BUILD_STAGING_TTL).unwrap(),
            0
        );
        assert!(lease_path.is_dir());
        fs::remove_dir_all(&staging).unwrap();
        assert!(existing_staging_lease(&staging, LockMode::Shared).is_err());
        assert!(!staging.exists(), "retaining must not resurrect staging");
        fs::remove_dir_all(base).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn staging_reaper_rejects_symlinked_ownership() {
        use std::os::unix::fs::symlink;
        let base = tempdir("reaper-symlink-lease");
        let external = base.join("external");
        fs::create_dir_all(&external).unwrap();
        fs::write(external.join(BASE_BUILD_STAGING_LEASE), b"keep").unwrap();
        let old = SystemTime::now() - Duration::from_secs(7 * 60 * 60);
        let file_link = base.join("greppy-base-build-file-link");
        fs::create_dir_all(file_link.join("locks")).unwrap();
        symlink(
            external.join(BASE_BUILD_STAGING_LEASE),
            file_link.join("locks").join(BASE_BUILD_STAGING_LEASE),
        )
        .unwrap();
        let dir_link = base.join("greppy-base-build-dir-link");
        fs::create_dir_all(&dir_link).unwrap();
        symlink(&external, dir_link.join("locks")).unwrap();
        let root_link = base.join("greppy-base-build-root-link");
        symlink(&external, &root_link).unwrap();
        for path in [&file_link, &dir_link] {
            set_directory_modified(path, old);
            assert!(existing_staging_lease(path, LockMode::Shared).is_err());
        }
        assert!(existing_staging_lease(&root_link, LockMode::Shared).is_err());
        assert_eq!(
            reap_stale_base_build_dirs(&base, BASE_BUILD_STAGING_TTL).unwrap(),
            0
        );
        assert!(file_link.is_dir() && dir_link.is_dir() && root_link.is_symlink());
        assert_eq!(
            fs::read(external.join(BASE_BUILD_STAGING_LEASE)).unwrap(),
            b"keep"
        );
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn gc_reports_staging_deletions_and_held_leases() {
        let _guard = TEST_ENV_LOCK.lock().unwrap();
        let base = tempdir("gc-staging-report");
        let staging = base.join("greppy-base-build-completed");
        let active = base.join("greppy-linked-base-checkout-held");
        for path in [&staging, &active] {
            fs::create_dir_all(path.join("data")).unwrap();
            fs::write(path.join("data/payload"), b"fixture").unwrap();
        }
        drop(create_base_build_staging_lease(&staging).unwrap());
        let lease = create_base_build_staging_lease(&active).unwrap();
        for path in [&staging, &active] {
            set_directory_modified(path, SystemTime::now() - Duration::from_secs(7 * 60 * 60));
        }
        let bytes = path_size_no_symlink(&staging);
        std::env::set_var("GREPPY_STORE_DIR", &base);
        let policy = GcPolicy {
            ttl: Duration::from_secs(1),
            high_water_bytes: u64::MAX,
            low_water_bytes: u64::MAX,
            interval: Duration::ZERO,
        };
        let dry = run_gc(&policy, true, None).unwrap();
        assert_eq!(dry.removed, vec![staging.clone()]);
        assert_eq!(dry.removed_bytes, bytes);
        assert_eq!(dry.skipped_locked, vec![active.clone()]);
        assert!(staging.is_dir());
        let actual = run_gc(&policy, false, None).unwrap();
        assert_eq!(actual.removed, dry.removed);
        assert_eq!(actual.removed_bytes, dry.removed_bytes);
        assert_eq!(actual.skipped_locked, dry.skipped_locked);
        assert!(!staging.exists() && active.is_dir());
        std::env::remove_var("GREPPY_STORE_DIR");
        drop(lease);
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    #[ignore = "invoked only by the staging lease subprocess regression"]
    fn staging_lease_subprocess_holder() {
        let Some(ready) = std::env::var_os("GREPPY_TEST_STAGING_LEASE_READY") else {
            return;
        };
        let _leases = retain_base_build_staging_leases_from_env().unwrap();
        fs::write(ready, b"ready").unwrap();
        let mut line = String::new();
        std::io::stdin().read_line(&mut line).unwrap();
    }

    #[test]
    fn staging_child_retains_lease_after_parent_releases_it() {
        struct ChildGuard(std::process::Child);
        impl Drop for ChildGuard {
            fn drop(&mut self) {
                let _ = self.0.kill();
                let _ = self.0.wait();
            }
        }
        let base = tempdir("reaper-child");
        for prefix in BASE_BUILD_STAGING_PREFIXES {
            let staging = base.join(format!("{prefix}child"));
            fs::create_dir_all(staging.join("data")).unwrap();
            let parent = create_base_build_staging_lease(&staging).unwrap();
            let ready = staging.join("data/ready");
            let mut child = ChildGuard(
                std::process::Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--exact",
                        "cache::tests::staging_lease_subprocess_holder",
                        "--ignored",
                        "--test-threads=1",
                    ])
                    .env(
                        ENV_BASE_BUILD_STAGING_LEASES,
                        std::env::join_paths([&staging]).unwrap(),
                    )
                    .env("GREPPY_TEST_STAGING_LEASE_READY", &ready)
                    .stdin(std::process::Stdio::piped())
                    .stdout(std::process::Stdio::null())
                    .spawn()
                    .unwrap(),
            );
            let until = std::time::Instant::now() + Duration::from_secs(10);
            while !ready.exists() {
                assert!(
                    child.0.try_wait().unwrap().is_none(),
                    "lease child exited before ready for {prefix}"
                );
                assert!(
                    std::time::Instant::now() < until,
                    "lease child readiness timed out for {prefix}"
                );
                std::thread::sleep(Duration::from_millis(10));
            }
            drop(parent);
            set_directory_modified(
                &staging,
                SystemTime::now() - Duration::from_secs(7 * 60 * 60),
            );
            assert_eq!(
                reap_stale_base_build_dirs(&base, BASE_BUILD_STAGING_TTL).unwrap(),
                0,
                "child lease must protect {prefix} staging"
            );
            assert!(
                staging.is_dir(),
                "child lifetime is independent of the parent lease for {prefix}"
            );
            drop(child.0.stdin.take());
            assert!(child.0.wait().unwrap().success());
            assert_eq!(
                reap_stale_base_build_dirs(&base, BASE_BUILD_STAGING_TTL).unwrap(),
                1,
                "reaper must reclaim released {prefix} staging"
            );
            assert!(!staging.exists());
        }
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn gc_never_deletes_unmanaged_override_children() {
        let _guard = TEST_ENV_LOCK.lock().unwrap();
        let base = tempdir("unmanaged");
        let data = base.join("shared");
        let unrelated = data.join("do-not-delete");
        fs::create_dir_all(&unrelated).unwrap();
        fs::write(unrelated.join("important.txt"), b"user data").unwrap();
        let foreign_trash = data.join("trash").join("operator-backup");
        fs::create_dir_all(&foreign_trash).unwrap();
        fs::write(foreign_trash.join("important.txt"), b"also user data").unwrap();
        std::env::set_var("GREPPY_STORE_DIR", &data);
        let policy = GcPolicy {
            ttl: Duration::from_secs(1),
            high_water_bytes: 1,
            low_water_bytes: 0,
            interval: Duration::ZERO,
        };
        let _ = run_gc(&policy, false, None).unwrap();
        assert_eq!(
            fs::read(unrelated.join("important.txt")).unwrap(),
            b"user data"
        );
        assert_eq!(
            fs::read(foreign_trash.join("important.txt")).unwrap(),
            b"also user data"
        );
        std::env::remove_var("GREPPY_STORE_DIR");
        let _ = fs::remove_dir_all(base);
    }

    #[cfg(unix)]
    #[test]
    fn gc_never_follows_symlinked_namespace_components() {
        let _guard = TEST_ENV_LOCK.lock().unwrap();
        let base = tempdir("namespace-symlink");
        let data = base.join("data");
        let repo = base.join("repo");
        fs::create_dir_all(&repo).unwrap();
        std::env::set_var("GREPPY_STORE_DIR", &data);
        let store = ensure_workspace_store(&repo).unwrap();
        fs::write(store.join("payload"), vec![0u8; 4096]).unwrap();

        let owned_workspaces = data.join("workspaces");
        let external_workspaces = base.join("external-workspaces");
        fs::rename(&owned_workspaces, &external_workspaces).unwrap();
        std::os::unix::fs::symlink(&external_workspaces, &owned_workspaces).unwrap();
        let external_store = external_workspaces
            .join(format!("v{STORE_FORMAT_VERSION}"))
            .join(crate::workspace::workspace_hash(&repo));

        let policy = GcPolicy {
            ttl: Duration::from_secs(1),
            high_water_bytes: 1,
            low_water_bytes: 0,
            interval: Duration::ZERO,
        };
        let _ = run_gc(&policy, false, None).unwrap();
        assert!(owned_workspaces.is_symlink());
        assert!(
            external_store.exists(),
            "GC must not follow namespace symlinks"
        );

        std::env::remove_var("GREPPY_STORE_DIR");
        let _ = fs::remove_file(owned_workspaces);
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn gc_dry_run_does_not_resume_verified_trash() {
        let _guard = TEST_ENV_LOCK.lock().unwrap();
        let base = tempdir("dry-run-trash");
        let data = base.join("data");
        let repo = base.join("repo");
        fs::create_dir_all(&repo).unwrap();
        std::env::set_var("GREPPY_STORE_DIR", &data);
        let store = ensure_workspace_store(&repo).unwrap();
        let hash = crate::workspace::workspace_hash(&repo);
        let trashed = trash_root().join(format!("workspace-{hash}-1-1"));
        fs::rename(&store, &trashed).unwrap();
        let policy = GcPolicy {
            ttl: Duration::from_secs(1),
            high_water_bytes: 1,
            low_water_bytes: 0,
            interval: Duration::ZERO,
        };

        let report = run_gc(&policy, true, None).unwrap();
        assert!(report.dry_run);
        assert!(trashed.exists(), "dry-run must not resume trash deletion");
        let _ = run_gc(&policy, false, None).unwrap();
        assert!(!trashed.exists(), "real GC resumes verified trash deletion");

        std::env::remove_var("GREPPY_STORE_DIR");
        let _ = fs::remove_dir_all(base);
    }

    #[cfg(unix)]
    #[test]
    fn exclusive_lock_cannot_steal_live_shared_lock_regardless_of_age() {
        let _guard = TEST_ENV_LOCK.lock().unwrap();
        let base = tempdir("lock");
        std::env::set_var("GREPPY_STORE_DIR", base.join("data"));
        let shared = acquire_named_lock("x", LockMode::Shared, false)
            .unwrap()
            .unwrap();
        assert!(acquire_named_lock("x", LockMode::Exclusive, true)
            .unwrap()
            .is_none());
        drop(shared);
        assert!(acquire_named_lock("x", LockMode::Exclusive, true)
            .unwrap()
            .is_some());
        std::env::remove_var("GREPPY_STORE_DIR");
        let _ = fs::remove_dir_all(base);
    }

    #[cfg(unix)]
    #[test]
    fn quota_continues_past_locked_lru_entries_to_low_water() {
        let _guard = TEST_ENV_LOCK.lock().unwrap();
        let base = tempdir("quota-lock");
        let data = base.join("data");
        let repo_a = base.join("repo-a");
        let repo_b = base.join("repo-b");
        fs::create_dir_all(&repo_a).unwrap();
        fs::create_dir_all(&repo_b).unwrap();
        std::env::set_var("GREPPY_STORE_DIR", &data);
        let store_a = ensure_workspace_store(&repo_a).unwrap();
        let store_b = ensure_workspace_store(&repo_b).unwrap();
        fs::write(store_a.join("payload"), vec![0u8; 4096]).unwrap();
        fs::write(store_b.join("payload"), vec![0u8; 4096]).unwrap();
        fs::write(store_a.join(LAST_USED_FILE), b"1").unwrap();
        fs::write(store_b.join(LAST_USED_FILE), b"2").unwrap();
        let _lease = acquire_workspace_lifecycle(&repo_a, LockMode::Shared, false)
            .unwrap()
            .unwrap();
        let policy = GcPolicy {
            ttl: Duration::ZERO,
            high_water_bytes: 1,
            low_water_bytes: 0,
            interval: Duration::ZERO,
        };
        let report = run_gc(&policy, false, None).unwrap();
        assert!(store_a.exists(), "locked oldest store must survive");
        assert!(
            !store_b.exists(),
            "GC must continue to the next unlocked LRU"
        );
        assert!(report.locked_bytes >= 4096);
        assert!(report.removed_bytes >= 4096);
        std::env::remove_var("GREPPY_STORE_DIR");
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn agent_base_is_managed_and_live_reader_lease_blocks_gc() {
        let _guard = TEST_ENV_LOCK.lock().unwrap();
        let base = tempdir("agent-base-gc");
        let data = base.join("data");
        std::env::set_var("GREPPY_STORE_DIR", &data);
        let identity_hash = "b".repeat(64);
        let directory = agent_base_stores_root()
            .join("a".repeat(64))
            .join(&identity_hash);
        write_agent_base_manifest(&directory, &identity_hash, "git-common-dir:/repo/.git").unwrap();
        fs::write(directory.join("graph.db"), vec![0u8; 4096]).unwrap();
        fs::write(directory.join("COMPLETE"), &identity_hash).unwrap();
        fs::write(directory.join(LAST_USED_FILE), b"1").unwrap();

        let reader = acquire_named_lock(
            &format!("agent-base-{identity_hash}.builder"),
            LockMode::Shared,
            false,
        )
        .unwrap()
        .unwrap();
        let status = cache_status().unwrap();
        let entry = status
            .entries
            .iter()
            .find(|entry| entry.kind == "agent-base")
            .expect("agent Base must be managed");
        assert_eq!(entry.path, directory);
        assert!(entry.locked);
        assert!(!status
            .unmanaged
            .iter()
            .any(|path| path.starts_with(data.join("agent-base-stores"))));

        let policy = GcPolicy {
            ttl: Duration::ZERO,
            high_water_bytes: 1,
            low_water_bytes: 0,
            interval: Duration::ZERO,
        };
        let locked = run_gc(&policy, false, None).unwrap();
        assert!(directory.exists());
        assert!(locked.skipped_locked.contains(&directory));
        drop(reader);
        let removed = run_gc(&policy, false, None).unwrap();
        assert!(!directory.exists());
        assert!(removed.removed.contains(&directory));

        std::env::remove_var("GREPPY_STORE_DIR");
        let _ = fs::remove_dir_all(base);
    }
}
