//! Production cache/store lifecycle for greppy.
//!
//! The data root is deliberately split into owned, versioned namespaces.  GC
//! never walks arbitrary children of `GREPPY_STORE_DIR`: it only considers
//! workspace stores with a valid manifest and model entries with a validated
//! digest-shaped directory.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub const STORE_FORMAT_VERSION: u32 = 2;
pub const STORE_MANIFEST_FILE: &str = "store.manifest";
pub const LAST_USED_FILE: &str = ".lastused";
pub const DEFAULT_STORE_TTL_DAYS: u64 = 14;
pub const DEFAULT_STORE_MAX_GIB: u64 = 10;
pub const DEFAULT_GC_INTERVAL_SECS: u64 = 10 * 60;
pub const DEFAULT_QUERY_CACHE_MAX_MIB: u64 = 64;
pub const ORPHAN_GRACE_SECS: u64 = 24 * 60 * 60;

const STORE_MANIFEST_MAGIC: &str = "greppy-workspace-store";
const LAST_USED_WRITE_GAP: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoreManifest {
    pub format_version: u32,
    pub workspace_hash: String,
    pub canonical_root: PathBuf,
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
}

impl ManagedKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Workspace => "workspace",
            Self::Model => "model",
        }
    }
}

pub fn data_root() -> PathBuf {
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

pub fn workspaces_root() -> PathBuf {
    data_root()
        .join("workspaces")
        .join(format!("v{STORE_FORMAT_VERSION}"))
}

pub fn models_root() -> PathBuf {
    data_root().join("models").join("v1")
}

pub fn locks_root() -> PathBuf {
    data_root().join("locks")
}

pub fn trash_root() -> PathBuf {
    data_root().join("trash")
}

pub fn workspace_store_dir(workspace_root: &Path) -> PathBuf {
    workspaces_root().join(crate::workspace::workspace_hash(workspace_root))
}

pub fn workspace_store_path(workspace_root: &Path) -> PathBuf {
    workspace_store_dir(workspace_root).join("graph.db")
}

pub fn legacy_workspace_store_dir(workspace_root: &Path) -> PathBuf {
    data_root().join(crate::workspace::workspace_hash(workspace_root))
}

pub fn ensure_workspace_store(workspace_root: &Path) -> io::Result<PathBuf> {
    ensure_owned_namespace(&data_root())?;
    ensure_owned_namespace(&workspaces_root())?;
    ensure_owned_namespace(&locks_root())?;
    ensure_owned_namespace(&trash_root())?;
    let dir = workspace_store_dir(workspace_root);
    ensure_owned_namespace(&dir)?;
    let expected = StoreManifest {
        format_version: STORE_FORMAT_VERSION,
        workspace_hash: crate::workspace::workspace_hash(workspace_root),
        canonical_root: canonical_root(workspace_root),
        created_at_unix_secs: unix_now_secs(),
    };
    let manifest_path = dir.join(STORE_MANIFEST_FILE);
    match read_store_manifest(&dir) {
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
    touch_last_used_dir(&dir);
    Ok(dir)
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
    if atomic_write(&marker, unix_now_secs().to_string().as_bytes()).is_ok() {
        prune_expired_sidecars(dir);
    }
}

fn prune_expired_sidecars(dir: &Path) {
    let now = SystemTime::now();
    let ttl = Duration::from_secs(24 * 60 * 60);
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if !name.ends_with("__CODE_CONTEXT_NONCANONICAL.md") {
            continue;
        }
        let expired = entry
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| now.duration_since(t).ok())
            .is_some_and(|age| age > ttl);
        if expired {
            let _ = fs::remove_file(path);
        }
    }
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
    ensure_owned_namespace(&locks_root())?;
    let path = locks_root().join(sanitize_lock_name(name));
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&path)?;
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
    gc_locked(policy, dry_run, current_workspace_root)
}

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

fn scan_entries(mark_new_orphans: bool) -> io::Result<(Vec<ManagedEntry>, Vec<PathBuf>, u64)> {
    let mut entries = Vec::new();
    let mut unmanaged_paths = Vec::new();
    let mut unmanaged = 0u64;
    let ws_root = workspaces_root();
    if namespace_chain_is_safe(&ws_root) {
        let rd = fs::read_dir(&ws_root)?;
        for item in rd.flatten() {
            let path = item.path();
            if item.file_type().map(|t| t.is_symlink()).unwrap_or(true) {
                unmanaged = unmanaged.saturating_add(path_size_no_symlink(&path));
                unmanaged_paths.push(path);
                continue;
            }
            match read_store_manifest(&path) {
                Ok(manifest) => {
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
                        workspace_root: Some(manifest.canonical_root.clone()),
                        bytes: path_size_no_symlink(&path),
                        last_used,
                        orphaned,
                        orphaned_since,
                    });
                }
                Err(_) => {
                    unmanaged = unmanaged.saturating_add(path_size_no_symlink(&path));
                    unmanaged_paths.push(path);
                }
            }
        }
    } else if ws_root.exists() {
        unmanaged_paths.push(ws_root.clone());
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
                Some("workspaces" | "models" | "locks" | "trash" | "gc.state")
            ) {
                continue;
            }
            let path = entry.path();
            unmanaged = unmanaged.saturating_add(path_size_no_symlink(&path));
            unmanaged_paths.push(path);
        }
    }
    if namespace_chain_is_safe(&trash_root()) {
        let trash_entries = fs::read_dir(trash_root())?;
        for entry in trash_entries.flatten() {
            let path = entry.path();
            if !trash_entry_is_verified(&path) {
                unmanaged = unmanaged.saturating_add(path_size_no_symlink(&path));
                unmanaged_paths.push(path);
            }
        }
    } else if trash_root().exists() {
        unmanaged_paths.push(trash_root());
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

fn try_entry_lock(entry: &ManagedEntry) -> io::Result<Option<FileLock>> {
    match entry.kind {
        ManagedKind::Workspace => acquire_named_lock(
            &format!("workspace-{}.lease", entry.id),
            LockMode::Exclusive,
            true,
        ),
        ManagedKind::Model => acquire_model_lifecycle(&entry.id, LockMode::Exclusive, true),
    }
}

fn move_to_trash(entry: &ManagedEntry) -> io::Result<PathBuf> {
    ensure_owned_namespace(&trash_root())?;
    let target = trash_root().join(format!(
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
    let root = trash_root();
    if root.exists() && !namespace_chain_is_safe(&root) {
        return Ok(());
    }
    let rd = match fs::read_dir(&root) {
        Ok(rd) => rd,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e),
    };
    for entry in rd.flatten() {
        let path = entry.path();
        if trash_entry_is_verified(&path) {
            let _ = remove_path_no_symlink(&path);
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
                    .map(|s| s.trim() == digest)
                    .unwrap_or(false)
        })
    })
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

fn atomic_write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        ensure_owned_namespace(parent)?;
    }
    let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&tmp)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    fs::rename(&tmp, path)
}

fn ensure_owned_namespace(dir: &Path) -> io::Result<()> {
    let data = data_root();
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
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("refusing non-directory cache namespace {}", dir.display()),
            ));
        }
    } else {
        fs::create_dir_all(dir)?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(dir, fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

fn namespace_chain_is_safe(dir: &Path) -> bool {
    let data = data_root();
    let Ok(relative) = dir.strip_prefix(&data) else {
        return false;
    };
    let mut current = data;
    let safe_dir = |path: &Path| {
        fs::symlink_metadata(path)
            .map(|metadata| !metadata.file_type().is_symlink() && metadata.is_dir())
            .unwrap_or(false)
    };
    if !safe_dir(&current) {
        return false;
    }
    for component in relative.components() {
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
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV: Mutex<()> = Mutex::new(());

    fn tempdir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "greppy-cache-{tag}-{}-{}",
            std::process::id(),
            unix_now_secs()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn versioned_store_has_valid_manifest() {
        let _guard = ENV.lock().unwrap();
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
    fn gc_never_deletes_unmanaged_override_children() {
        let _guard = ENV.lock().unwrap();
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
        let _guard = ENV.lock().unwrap();
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
        let _guard = ENV.lock().unwrap();
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
        let _guard = ENV.lock().unwrap();
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
        let _guard = ENV.lock().unwrap();
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
}
