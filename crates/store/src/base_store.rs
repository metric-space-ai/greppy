//! Immutable Base Store identity and Base/Delta read-view contracts.
//!
//! The types in this module deliberately contain no agent or CLI policy. They
//! are the narrow store-layer boundary used by the trusted Base publisher,
//! Delta refresher, and every index-backed query family.

use std::collections::BTreeSet;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::Store;

pub const BASE_STORE_FORMAT_VERSION: u32 = 1;
pub const BASE_STORE_MANIFEST_FILE: &str = "manifest.json";
pub const COMPLETE_FILE: &str = "COMPLETE";
pub const BASE_SUMMARY_CACHE_FILE: &str = crate::SUMMARY_CACHE_DB_FILE;

/// Every semantic input that determines whether an immutable Base can be
/// shared. The field order is part of the canonical identity serialization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BaseStoreIdentity {
    pub format_version: u32,
    /// Opaque repository namespace derived from the common Git directory (or
    /// an explicitly configured artifact namespace), never a linked-worktree
    /// checkout path.
    pub canonical_repository_identity: String,
    pub git_object_format: String,
    pub base_tree_oid: String,
    pub store_schema_version: u32,
    pub indexer_version: String,
    pub parser_and_extractor_versions: String,
    pub summary_model_and_prompt_version: String,
    pub embedding_model: String,
    pub embedding_prompt_version: String,
    pub embedding_dimensions: usize,
    pub embedding_encoding: String,
}

impl BaseStoreIdentity {
    pub fn validate(&self) -> io::Result<()> {
        if self.format_version != BASE_STORE_FORMAT_VERSION {
            return Err(invalid_data("unsupported Base Store identity format"));
        }
        for (name, value) in [
            (
                "canonical_repository_identity",
                self.canonical_repository_identity.as_str(),
            ),
            ("git_object_format", self.git_object_format.as_str()),
            ("base_tree_oid", self.base_tree_oid.as_str()),
            ("indexer_version", self.indexer_version.as_str()),
            (
                "parser_and_extractor_versions",
                self.parser_and_extractor_versions.as_str(),
            ),
            (
                "summary_model_and_prompt_version",
                self.summary_model_and_prompt_version.as_str(),
            ),
            ("embedding_model", self.embedding_model.as_str()),
            (
                "embedding_prompt_version",
                self.embedding_prompt_version.as_str(),
            ),
            ("embedding_encoding", self.embedding_encoding.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(invalid_data(format!(
                    "Base Store identity field `{name}` is empty"
                )));
            }
        }
        if !matches!(self.git_object_format.as_str(), "sha1" | "sha256") {
            return Err(invalid_data("unsupported Git object format"));
        }
        let oid_len = if self.git_object_format == "sha256" {
            64
        } else {
            40
        };
        if self.base_tree_oid.len() != oid_len
            || !self
                .base_tree_oid
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(invalid_data(
                "base_tree_oid does not match Git object format",
            ));
        }
        if self.store_schema_version == 0 || self.embedding_dimensions == 0 {
            return Err(invalid_data(
                "schema version and embedding dimensions must be non-zero",
            ));
        }
        Ok(())
    }

    /// SHA-256 of the canonical JSON representation. Struct field order is
    /// stable and unknown fields are rejected on decode.
    pub fn hash(&self) -> io::Result<String> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)
            .map_err(|error| invalid_data(format!("serialize Base Store identity: {error}")))?;
        Ok(hex_sha256(&bytes))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BaseStoreManifest {
    pub identity: BaseStoreIdentity,
    pub identity_hash: String,
    pub graph_sha256: String,
    pub summary_cache_sha256: String,
    pub published_at_unix_secs: u64,
}

impl BaseStoreManifest {
    pub fn validate(&self) -> io::Result<()> {
        let expected = self.identity.hash()?;
        if self.identity_hash != expected {
            return Err(invalid_data("Base Store manifest identity hash mismatch"));
        }
        if self.graph_sha256.len() != 64
            || !self
                .graph_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(invalid_data("Base Store graph digest is not SHA-256"));
        }
        if self.summary_cache_sha256.len() != 64
            || !self
                .summary_cache_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(invalid_data(
                "Base Store summary cache digest is not SHA-256",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaseStoreLayout {
    pub directory: PathBuf,
    pub manifest: PathBuf,
    pub graph: PathBuf,
    pub summary_cache: PathBuf,
    pub complete: PathBuf,
    data_root: PathBuf,
}

impl BaseStoreLayout {
    pub fn new(data_root: &Path, identity: &BaseStoreIdentity) -> io::Result<Self> {
        let identity_hash = identity.hash()?;
        let repo_hash = hex_sha256(identity.canonical_repository_identity.as_bytes());
        let directory = greppy_core::cache::agent_base_directory(
            data_root,
            &PathBuf::from(repo_hash).join(identity_hash),
        )?;
        Ok(Self {
            manifest: directory.join(BASE_STORE_MANIFEST_FILE),
            graph: directory.join("graph.db"),
            summary_cache: directory.join(BASE_SUMMARY_CACHE_FILE),
            complete: directory.join(COMPLETE_FILE),
            directory,
            data_root: data_root.to_path_buf(),
        })
    }

    /// Open only a completely published Base. `COMPLETE` contains the exact
    /// identity hash and is published last by the lifecycle layer.
    pub fn read_verified_manifest(&self) -> io::Result<BaseStoreManifest> {
        let complete = fs::read_to_string(&self.complete)?;
        let owner = greppy_core::cache::read_agent_base_manifest(&self.directory)?;
        let bytes = fs::read(&self.manifest)?;
        let manifest: BaseStoreManifest = serde_json::from_slice(&bytes)
            .map_err(|error| invalid_data(format!("decode Base Store manifest: {error}")))?;
        manifest.validate()?;
        if complete.trim() != manifest.identity_hash {
            return Err(invalid_data("Base Store COMPLETE marker mismatch"));
        }
        if owner.identity_hash != manifest.identity_hash {
            return Err(invalid_data("Base Store cache ownership marker mismatch"));
        }
        if owner.canonical_repository_identity != manifest.identity.canonical_repository_identity {
            return Err(invalid_data(
                "Base Store cache repository ownership mismatch",
            ));
        }
        if !self.graph.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "Base Store graph.db is missing",
            ));
        }
        let binding = hex_sha256(&bytes);
        let actual_graph_sha256 =
            verified_base_digest(&self.graph, &binding, &manifest.graph_sha256)?;
        if actual_graph_sha256 != manifest.graph_sha256 {
            return Err(invalid_data("Base Store graph digest mismatch"));
        }
        if !self.summary_cache.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "Base Store summary_cache.db is missing",
            ));
        }
        if verified_base_digest(
            &self.summary_cache,
            &binding,
            &manifest.summary_cache_sha256,
        )? != manifest.summary_cache_sha256
        {
            return Err(invalid_data("Base Store summary cache digest mismatch"));
        }
        Ok(manifest)
    }

    /// Move an invalid published generation aside while holding the builder
    /// lease. The bytes remain available for diagnosis, but the canonical
    /// identity path becomes free for one new atomic publication.
    pub fn quarantine_invalid(&self) -> io::Result<Option<PathBuf>> {
        if self.read_verified_manifest().is_ok() || !self.directory.exists() {
            return Ok(None);
        }
        self.quarantine_current()
    }

    /// Quarantine the current identity generation while the caller holds the
    /// exclusive lifecycle lease. This variant is used when SQLite/provider/
    /// semantic completeness validation fails even though file digests match.
    pub fn quarantine_current(&self) -> io::Result<Option<PathBuf>> {
        if !self.directory.exists() {
            return Ok(None);
        }
        let parent = self
            .directory
            .parent()
            .ok_or_else(|| invalid_data("Base Store layout has no parent"))?;
        let name = self
            .directory
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or_else(|| invalid_data("Base Store identity directory is not UTF-8"))?;
        let suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let quarantine = parent.join(format!("{name}.corrupt-{}-{suffix}", std::process::id()));
        fs::rename(&self.directory, &quarantine)?;
        sync_directory(parent)?;
        Ok(Some(quarantine))
    }

    /// Publish a completed graph as one immutable directory. The caller must
    /// hold the identity-scoped builder lease returned by
    /// [`Self::acquire_builder`]. A racing publisher either wins the atomic
    /// rename or verifies and reuses the winner.
    pub fn publish_graph_with_summary(
        &self,
        identity: BaseStoreIdentity,
        staged_graph: &Path,
        staged_summary_cache: &Path,
    ) -> io::Result<BaseStoreManifest> {
        let expected_hash = identity.hash()?;
        if let Ok(existing) = self.read_verified_manifest() {
            if existing.identity_hash == expected_hash {
                return Ok(existing);
            }
            return Err(invalid_data("published Base Store has the wrong identity"));
        }
        let parent = self
            .directory
            .parent()
            .ok_or_else(|| invalid_data("Base Store layout has no parent"))?;
        fs::create_dir_all(parent)?;
        let suffix = format!(
            ".building-{}-{}-{}",
            expected_hash,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );
        let building = parent.join(suffix);
        fs::create_dir(&building)?;
        greppy_core::cache::write_agent_base_manifest(
            &building,
            &expected_hash,
            &identity.canonical_repository_identity,
        )?;
        let result = (|| {
            let graph = building.join("graph.db");
            fs::copy(staged_graph, &graph)?;
            let graph_sha256 = file_sha256(&graph)?;
            let summary_cache = building.join(BASE_SUMMARY_CACHE_FILE);
            fs::copy(staged_summary_cache, &summary_cache)?;
            set_read_only(&summary_cache)?;
            let summary_cache_sha256 = file_sha256(&summary_cache)?;
            let manifest = BaseStoreManifest {
                identity,
                identity_hash: expected_hash.clone(),
                graph_sha256,
                summary_cache_sha256,
                published_at_unix_secs: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            };
            manifest.validate()?;
            let manifest_path = building.join(BASE_STORE_MANIFEST_FILE);
            write_new_synced(
                &manifest_path,
                &serde_json::to_vec_pretty(&manifest).map_err(|error| {
                    invalid_data(format!("serialize Base Store manifest: {error}"))
                })?,
            )?;
            set_read_only(&manifest_path)?;
            set_read_only(&building.join(greppy_core::cache::AGENT_BASE_MANIFEST_FILE))?;
            set_read_only(&graph)?;
            // COMPLETE is deliberately the last file created in the private
            // staging directory. The following directory rename publishes all
            // three files as one visible generation.
            write_new_synced(building.join(COMPLETE_FILE), expected_hash.as_bytes())?;
            set_read_only(&building.join(COMPLETE_FILE))?;
            sync_directory(&building)?;
            match fs::rename(&building, &self.directory) {
                Ok(()) => {
                    sync_directory(parent)?;
                    Ok(manifest)
                }
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::AlreadyExists | io::ErrorKind::DirectoryNotEmpty
                    ) =>
                {
                    let winner = self.read_verified_manifest()?;
                    if winner.identity_hash == expected_hash {
                        Ok(winner)
                    } else {
                        Err(invalid_data("racing Base publisher used wrong identity"))
                    }
                }
                Err(error) => Err(error),
            }
        })();
        if building.exists() {
            let _ = fs::remove_dir_all(&building);
        }
        result
    }

    pub fn acquire_builder(&self, nonblocking: bool) -> io::Result<Option<BaseBuilderLease>> {
        let lock = greppy_core::cache::acquire_named_lock_in(
            &self.data_root,
            &self.lifecycle_lock_name()?,
            greppy_core::cache::LockMode::Exclusive,
            nonblocking,
        )?;
        Ok(lock.map(|lock| BaseBuilderLease { _lock: lock }))
    }

    /// Exact advisory-lock path used to serialize immutable Base publishers.
    /// Exposed for fail-closed diagnostics; ownership still comes exclusively
    /// from [`Self::acquire_builder`].
    pub fn builder_lock_path(&self) -> io::Result<PathBuf> {
        Ok(self
            .data_root
            .join("locks")
            .join(self.lifecycle_lock_name()?))
    }

    /// Hold this shared lease for the complete lifetime of an agent using the
    /// Base. GC and rebuild use the exclusive builder lease, so a live Base
    /// cannot be reclaimed beneath attached read-only stores.
    pub fn acquire_reader(&self, nonblocking: bool) -> io::Result<Option<BaseReaderLease>> {
        let lock = greppy_core::cache::acquire_named_lock_in(
            &self.data_root,
            &self.lifecycle_lock_name()?,
            greppy_core::cache::LockMode::Shared,
            nonblocking,
        )?;
        Ok(lock.map(|lock| BaseReaderLease { _lock: lock }))
    }

    fn lifecycle_lock_name(&self) -> io::Result<String> {
        let identity_hash = self
            .directory
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| invalid_data("Base Store layout has no identity component"))?;
        Ok(format!("agent-base-{identity_hash}.builder"))
    }
}

#[derive(Debug)]
pub struct BaseBuilderLease {
    _lock: greppy_core::cache::FileLock,
}

#[derive(Debug)]
pub struct BaseReaderLease {
    _lock: greppy_core::cache::FileLock,
}

/// Paths whose Base contributions are hidden by one complete Delta generation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VisibilityIndex {
    dirty: BTreeSet<String>,
    deleted: BTreeSet<String>,
}

impl VisibilityIndex {
    pub fn new(
        dirty: impl IntoIterator<Item = String>,
        deleted: impl IntoIterator<Item = String>,
    ) -> io::Result<Self> {
        let dirty = normalize_paths(dirty)?;
        let deleted = normalize_paths(deleted)?;
        if !dirty.is_disjoint(&deleted) {
            return Err(invalid_data(
                "a Delta path cannot be both dirty and deleted",
            ));
        }
        Ok(Self { dirty, deleted })
    }

    pub fn hides_base_path(&self, path: &str) -> bool {
        self.dirty.contains(path) || self.deleted.contains(path)
    }

    pub fn is_dirty_path(&self, path: &str) -> bool {
        self.dirty.contains(path)
    }

    pub fn is_deleted_path(&self, path: &str) -> bool {
        self.deleted.contains(path)
    }

    pub fn dirty_paths(&self) -> impl Iterator<Item = &str> {
        self.dirty.iter().map(String::as_str)
    }

    pub fn deleted_paths(&self) -> impl Iterator<Item = &str> {
        self.deleted.iter().map(String::as_str)
    }

    pub fn changed_count(&self) -> usize {
        self.dirty.len() + self.deleted.len()
    }
}

/// Explicit query boundary. Writers receive a `Store`; readers receive this
/// view and therefore cannot accidentally target the immutable Base through
/// the public API.
#[derive(Debug)]
pub enum StoreView {
    Single(Store),
    Overlay {
        store: Store,
        visibility: VisibilityIndex,
    },
}

impl StoreView {
    pub fn single(store: Store) -> Self {
        Self::Single(store)
    }

    pub fn open_overlay(
        base_path: &Path,
        delta_path: &Path,
        visibility: VisibilityIndex,
    ) -> crate::Result<Self> {
        let store = Store::open_overlay(base_path, delta_path, &visibility)?;
        Ok(Self::Overlay { store, visibility })
    }

    pub fn is_overlay(&self) -> bool {
        matches!(self, Self::Overlay { .. })
    }

    pub fn visibility(&self) -> Option<&VisibilityIndex> {
        match self {
            Self::Single(_) => None,
            Self::Overlay { visibility, .. } => Some(visibility),
        }
    }

    pub fn single_store(&self) -> Option<&Store> {
        match self {
            Self::Single(store) => Some(store),
            Self::Overlay { .. } => None,
        }
    }

    pub fn layers(&self) -> (&Store, Option<&Store>) {
        match self {
            Self::Single(store) => (store, None),
            Self::Overlay { store, .. } => (store, None),
        }
    }

    pub fn store(&self) -> &Store {
        match self {
            Self::Single(store) | Self::Overlay { store, .. } => store,
        }
    }

    pub fn store_mut(&mut self) -> &mut Store {
        match self {
            Self::Single(store) | Self::Overlay { store, .. } => store,
        }
    }
}

fn normalize_paths(paths: impl IntoIterator<Item = String>) -> io::Result<BTreeSet<String>> {
    let mut normalized = BTreeSet::new();
    for path in paths {
        let candidate = Path::new(&path);
        if path.is_empty()
            || candidate.is_absolute()
            || candidate.components().any(|component| {
                matches!(
                    component,
                    Component::ParentDir | Component::RootDir | Component::Prefix(_)
                )
            })
        {
            return Err(invalid_data(format!(
                "invalid Delta-relative path `{path}`"
            )));
        }
        let joined = candidate
            .components()
            .filter_map(|component| match component {
                Component::Normal(part) => Some(part.to_string_lossy()),
                Component::CurDir => None,
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("/");
        if joined.is_empty() {
            return Err(invalid_data("Delta-relative path normalizes to empty"));
        }
        normalized.insert(joined);
    }
    Ok(normalized)
}

fn hex_sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

// Persistent reuse is confined to manifest-bound published Bases. A hit is a
// recently verified snapshot, not a fresh cryptographic check of unread bytes.
// Undetectable media corruption is found at the next full verification (30s).
fn verified_base_digest(path: &Path, binding: &str, expected: &str) -> io::Result<String> {
    #[cfg(unix)]
    {
        verified_base_digest_at(path, binding, expected, trusted_digest_directory().ok())
    }
    #[cfg(not(unix))]
    {
        let _ = (binding, expected);
        file_sha256(path)
    }
}

#[cfg(unix)]
fn verified_base_digest_at(
    path: &Path,
    binding: &str,
    expected: &str,
    proofs: Option<fs::File>,
) -> io::Result<String> {
    use std::os::unix::fs::OpenOptionsExt;
    let mut file = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    let before = digest_file_identity(&file)?;
    let started = DigestVerificationTime::now();

    if let Some(directory) = &proofs {
        if let Some(proof) = read_digest_proof(directory, binding, expected) {
            if proof.matches(before, binding, expected, started.wall) {
                if digest_file_identity(&file)? != before {
                    return Err(invalid_data(
                        "Base file changed during snapshot verification",
                    ));
                }
                if proof.matches(before, binding, expected, digest_now_secs()) {
                    return Ok(expected.to_owned());
                }
            }
        }
    }
    let digest = hash_opened_file(&mut file)?;
    let after = digest_file_identity(&file)?;
    if after != before {
        return Err(invalid_data("Base file changed during digest verification"));
    }
    // Anchor only after the matching full read and opened-file stability
    // checks. Initial eligibility remains mandatory: a long read must not
    // promote fresh metadata merely because its time bucket aged meanwhile.
    if digest == expected {
        let completed = DigestVerificationTime::now();
        if let Some(directory) = proofs {
            if let Some(proof) = DigestProof::from_completed_read(
                before,
                after,
                binding,
                expected,
                Ok(digest.as_str()),
                started,
                completed,
            ) {
                let _ = write_digest_proof(&directory, &proof);
            }
        }
    }
    Ok(digest)
}

#[cfg(test)]
thread_local! {
    static FULL_DIGEST_READS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static BEFORE_DIGEST_READ: std::cell::RefCell<Option<Box<dyn FnOnce()>>> = const { std::cell::RefCell::new(None) };
}

fn hash_opened_file(file: &mut fs::File) -> io::Result<String> {
    #[cfg(test)]
    FULL_DIGEST_READS.with(|reads| reads.set(reads.get() + 1));
    #[cfg(test)]
    if let Some(hook) = BEFORE_DIGEST_READ.with(|slot| slot.borrow_mut().take()) {
        hook();
    }
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

#[cfg(unix)]
const DIGEST_PROOF_TTL_SECS: u64 = 30;

#[cfg(unix)]
fn digest_monotonic_secs() -> Option<u64> {
    let mut time = std::mem::MaybeUninit::<libc::timespec>::uninit();
    if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, time.as_mut_ptr()) } != 0 {
        return None;
    }
    u64::try_from(unsafe { time.assume_init() }.tv_sec).ok()
}

#[cfg(unix)]
#[derive(Clone, Copy)]
struct DigestVerificationTime {
    wall: Option<u64>,
    monotonic: Option<u64>,
}

#[cfg(unix)]
impl DigestVerificationTime {
    fn now() -> Self {
        Self {
            wall: digest_now_secs(),
            monotonic: digest_monotonic_secs(),
        }
    }
}

#[cfg(unix)]
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DigestProof {
    version: u32,
    manifest_sha256: String,
    digest: String,
    identity: DigestFileIdentity,
    verified_at: u64,
    verified_monotonic: u64,
}

#[cfg(unix)]
impl DigestProof {
    fn from_completed_read(
        before: DigestFileIdentity,
        after: DigestFileIdentity,
        binding: &str,
        expected: &str,
        digest: Result<&str, &io::Error>,
        started: DigestVerificationTime,
        completed: DigestVerificationTime,
    ) -> Option<Self> {
        let actual = digest.ok()?;
        let started_wall = started.wall?;
        let started_monotonic = started.monotonic?;
        let verified_at = completed.wall?;
        let verified_monotonic = completed.monotonic?;
        if actual != expected
            || before != after
            || verified_at < started_wall
            || verified_monotonic < started_monotonic
            || !digest_cache_insert_eligible(
                digest_cache_eligible(before, Some(started_wall)),
                before,
                Some(verified_at),
            )
        {
            return None;
        }
        Some(Self {
            version: 1,
            manifest_sha256: binding.to_owned(),
            digest: actual.to_owned(),
            identity: before,
            verified_at,
            verified_monotonic,
        })
    }

    fn matches(
        &self,
        identity: DigestFileIdentity,
        binding: &str,
        expected: &str,
        now: Option<u64>,
    ) -> bool {
        self.matches_at(
            identity,
            binding,
            expected,
            DigestVerificationTime {
                wall: now,
                monotonic: digest_monotonic_secs(),
            },
        )
    }

    fn matches_at(
        &self,
        identity: DigestFileIdentity,
        binding: &str,
        expected: &str,
        now: DigestVerificationTime,
    ) -> bool {
        let (Some(now), Some(monotonic)) = (now.wall, now.monotonic) else {
            return false;
        };
        self.version == 1
            && self.manifest_sha256 == binding
            && self.digest == expected
            && self.identity == identity
            && self.verified_at <= now
            && now - self.verified_at < DIGEST_PROOF_TTL_SECS
            && monotonic >= self.verified_monotonic
            && monotonic - self.verified_monotonic < DIGEST_PROOF_TTL_SECS
            && digest_cache_eligible(identity, Some(self.verified_at))
            && digest_cache_eligible(identity, Some(now))
    }
}

// Fixed slots bound normal persistent storage to 64 small records. Collisions
// replace an optimization only; the exact manifest/digest/identity still match.
#[cfg(unix)]
fn digest_proof_name(binding: &str, digest: &str) -> std::ffi::CString {
    let key = Sha256::digest(format!("{binding}:{digest}"));
    std::ffi::CString::new(format!("proof-{:02x}.json", key[0] % 64)).unwrap()
}

#[cfg(unix)]
fn open_relative(
    directory: &fs::File,
    name: &std::ffi::CStr,
    flags: i32,
    mode: u32,
) -> io::Result<fs::File> {
    use std::os::fd::{AsRawFd, FromRawFd};
    let fd = unsafe {
        libc::openat(
            directory.as_raw_fd(),
            name.as_ptr(),
            flags | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            mode,
        )
    };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(unsafe { fs::File::from_raw_fd(fd) })
}

#[cfg(unix)]
fn trusted_digest_directory() -> io::Result<fs::File> {
    let home = std::env::var_os("HOME").ok_or_else(|| invalid_data("HOME unavailable"))?;
    let home = Path::new(&home);
    if !home.is_absolute() {
        return Err(invalid_data("digest proof HOME is not absolute"));
    }
    #[cfg(target_os = "macos")]
    let path = home.join("Library/Application Support/greppy/verified-base-digests-v1");
    #[cfg(not(target_os = "macos"))]
    let path = home.join(".local/share/greppy-verified-base-digests-v1");
    trusted_digest_directory_at(&path)
}

#[cfg(unix)]
fn trusted_digest_directory_at(path: &Path) -> io::Result<fs::File> {
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
    if !path.is_absolute() {
        return Err(invalid_data("digest proof path is not absolute"));
    }
    let uid = unsafe { libc::geteuid() };
    let mut directory = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW)
        .open("/")?;
    reject_mutating_acl(&directory)?;
    for component in path.components() {
        let Component::Normal(name) = component else {
            if component == Component::RootDir {
                continue;
            }
            return Err(invalid_data("unsafe digest proof path component"));
        };
        use std::os::unix::ffi::OsStrExt;
        let name =
            std::ffi::CString::new(name.as_bytes()).map_err(|_| invalid_data("NUL proof path"))?;
        let next = match open_relative(&directory, &name, libc::O_RDONLY | libc::O_DIRECTORY, 0) {
            Ok(next) => next,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                let result = unsafe { libc::mkdirat(directory.as_raw_fd(), name.as_ptr(), 0o700) };
                if result != 0 && io::Error::last_os_error().kind() != io::ErrorKind::AlreadyExists
                {
                    return Err(io::Error::last_os_error());
                }
                open_relative(&directory, &name, libc::O_RDONLY | libc::O_DIRECTORY, 0)?
            }
            Err(error) => return Err(error),
        };
        let metadata = next.metadata()?;
        if (metadata.uid() != 0 && metadata.uid() != uid) || metadata.mode() & 0o022 != 0 {
            return Err(invalid_data(
                "unsafe digest proof directory owner or permissions",
            ));
        }
        #[cfg(target_os = "macos")]
        {
            let mut info = std::mem::MaybeUninit::<libc::statfs>::uninit();
            if unsafe { libc::fstatfs(next.as_raw_fd(), info.as_mut_ptr()) } != 0 {
                return Err(io::Error::last_os_error());
            }
            // Darwin MNT_IGNORE_OWNERS: synthesized ownership is not proof.
            if unsafe { info.assume_init() }.f_flags & 0x0020_0000 != 0 {
                return Err(invalid_data("digest proof filesystem ignores ownership"));
            }
        }
        reject_mutating_acl(&next)?;
        directory = next;
    }
    let metadata = directory.metadata()?;
    if metadata.uid() != uid || metadata.mode() & 0o077 != 0 {
        return Err(invalid_data(
            "digest proof directory must be private and owned",
        ));
    }
    Ok(directory)
}

#[cfg(unix)]
fn read_digest_proof(directory: &fs::File, binding: &str, expected: &str) -> Option<DigestProof> {
    use std::os::unix::fs::MetadataExt;
    let file = open_relative(
        directory,
        &digest_proof_name(binding, expected),
        libc::O_RDONLY | libc::O_NONBLOCK,
        0,
    )
    .ok()?;
    validate_private_proof_file(&file).ok()?;
    let before = file.metadata().ok()?;
    if !before.is_file()
        || before.uid() != unsafe { libc::geteuid() }
        || before.mode() & 0o077 != 0
        || before.nlink() != 1
        || before.len() > 4096
    {
        return None;
    }
    let mut bytes = Vec::new();
    let mut file = file;
    (&mut file).take(4097).read_to_end(&mut bytes).ok()?;
    let after = file.metadata().ok()?;
    validate_private_proof_file(&file).ok()?;
    if bytes.len() > 4096
        || before.len() != after.len()
        || before.ctime() != after.ctime()
        || before.ctime_nsec() != after.ctime_nsec()
        || before.mtime() != after.mtime()
        || before.mtime_nsec() != after.mtime_nsec()
    {
        return None;
    }
    serde_json::from_slice(&bytes).ok()
}

#[cfg(target_os = "macos")]
mod darwin_acl {
    pub const FILESEC_ACL: libc::c_int = 5;
    // Darwin SDK file-security and ACL ABI. The working ACL is an independent copy;
    // acl_valid plus fixed selectors makes EINVAL the documented end marker.
    unsafe extern "C" {
        pub fn filesec_init() -> *mut libc::c_void;
        pub fn filesec_free(security: *mut libc::c_void);
        #[cfg_attr(target_arch = "x86_64", link_name = "fstatx_np$INODE64")]
        pub fn fstatx_np(
            fd: libc::c_int,
            stat: *mut libc::stat,
            security: *mut libc::c_void,
        ) -> libc::c_int;
        pub fn filesec_query_property(
            security: *mut libc::c_void,
            property: libc::c_int,
            present: *mut libc::c_int,
        ) -> libc::c_int;
        pub fn filesec_get_property(
            security: *mut libc::c_void,
            property: libc::c_int,
            value: *mut libc::c_void,
        ) -> libc::c_int;
        pub fn acl_valid(acl: *mut libc::c_void) -> libc::c_int;
        pub fn acl_get_entry(
            acl: *mut libc::c_void,
            selector: libc::c_int,
            entry: *mut *mut libc::c_void,
        ) -> libc::c_int;
        pub fn acl_get_tag_type(entry: *mut libc::c_void, tag: *mut libc::c_int) -> libc::c_int;
        pub fn acl_get_permset_mask_np(entry: *mut libc::c_void, mask: *mut u64) -> libc::c_int;
        pub fn acl_free(acl: *mut libc::c_void) -> libc::c_int;
    }
    pub struct OwnedSecurity(pub *mut libc::c_void);
    impl Drop for OwnedSecurity {
        fn drop(&mut self) {
            unsafe {
                filesec_free(self.0);
            }
        }
    }
    pub struct OwnedAcl(pub *mut libc::c_void);
    impl Drop for OwnedAcl {
        fn drop(&mut self) {
            unsafe {
                acl_free(self.0);
            }
        }
    }
}

#[cfg(unix)]
fn reject_mutating_acl(file: &fs::File) -> io::Result<()> {
    #[cfg(target_os = "macos")]
    {
        use std::os::fd::AsRawFd;
        reject_mutating_acl_fd(file.as_raw_fd())
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = file;
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn reject_mutating_acl_fd(fd: libc::c_int) -> io::Result<()> {
    use darwin_acl::*;
    // acl_get_fd_np conflates a successful descriptor query with no extended
    // ACL and a failed query: both return NULL, commonly with ENOENT. Query
    // file security directly so only positively confirmed absence is safe.
    let security = unsafe { filesec_init() };
    if security.is_null() {
        return Err(io::Error::last_os_error());
    }
    let security = OwnedSecurity(security);
    let mut metadata = std::mem::MaybeUninit::<libc::stat>::uninit();
    if unsafe { fstatx_np(fd, metadata.as_mut_ptr(), security.0) } != 0 {
        return Err(io::Error::last_os_error());
    }
    let mut present = -1;
    if unsafe { filesec_query_property(security.0, FILESEC_ACL, &mut present) } != 0 {
        return Err(io::Error::last_os_error());
    }
    match present {
        0 => return Ok(()),
        // Darwin returns a property bitmask (currently 32), not normalized 1.
        // The public contract reports nonzero presence; the negative sentinel
        // also keeps an unwritten/invalid output fail-closed.
        value if value > 0 => {}
        _ => return Err(invalid_data("unknown digest proof ACL presence")),
    }
    let mut acl: *mut libc::c_void = std::ptr::null_mut();
    if unsafe {
        filesec_get_property(
            security.0,
            FILESEC_ACL,
            (&mut acl as *mut *mut libc::c_void).cast(),
        )
    } != 0
    {
        return Err(io::Error::last_os_error());
    }
    if acl.is_null() || acl as usize == 1 {
        return Err(invalid_data(
            "present digest proof ACL is unavailable or a removal sentinel",
        ));
    }
    let acl = OwnedAcl(acl);
    if unsafe { acl_valid(acl.0) } != 0 {
        return Err(io::Error::last_os_error());
    }
    let mut selector = 0; // ACL_FIRST_ENTRY
    for _ in 0..=128 {
        let mut entry = std::ptr::null_mut();
        if unsafe { acl_get_entry(acl.0, selector, &mut entry) } != 0 {
            let error = io::Error::last_os_error();
            return if error.raw_os_error() == Some(libc::EINVAL) {
                Ok(())
            } else {
                Err(error)
            };
        }
        selector = -1; // ACL_NEXT_ENTRY
        let mut tag = 0;
        let mut permissions = 0;
        if entry.is_null()
            || unsafe { acl_get_tag_type(entry, &mut tag) } != 0
            || unsafe { acl_get_permset_mask_np(entry, &mut permissions) } != 0
        {
            return Err(invalid_data("digest proof ACL entry query failed"));
        }
        // Conservatively reject every mutating ALLOW, including inherited
        // grants and owner grants. Read/search ALLOWs and DENYs are safe.
        // WRITE/APPEND_DATA, DELETE[_CHILD], WRITE_{ATTRIBUTES,EXTATTRIBUTES,
        // SECURITY}, CHANGE_OWNER (sys/acl.h). No principal resolution,
        // group membership or ordering can accidentally broaden trust.
        // READ_DATA, EXECUTE/SEARCH, READ_ATTRIBUTES, READ_EXTATTRIBUTES,
        // READ_SECURITY and SYNCHRONIZE only; reject unknown future bits.
        let read_only = (1_u64 << 1) | (1 << 3) | (1 << 7) | (1 << 9) | (1 << 11) | (1 << 20);
        match tag {
            1 if permissions & !read_only == 0 => {} // ACL_EXTENDED_ALLOW
            2 => {}                                  // ACL_EXTENDED_DENY
            1 => return Err(invalid_data("digest proof ACL permits mutation")),
            _ => return Err(invalid_data("unknown digest proof ACL tag")),
        }
    }
    Err(invalid_data("digest proof ACL exceeds Darwin entry bound"))
}

#[cfg(unix)]
fn validate_private_proof_file(file: &fs::File) -> io::Result<()> {
    use std::os::unix::fs::MetadataExt;
    let metadata = file.metadata()?;
    if !metadata.is_file()
        || metadata.uid() != unsafe { libc::geteuid() }
        || metadata.mode() & 0o077 != 0
        || metadata.nlink() != 1
    {
        return Err(invalid_data(
            "unsafe digest proof file owner, type, links or permissions",
        ));
    }
    reject_mutating_acl(file)
}

#[cfg(unix)]
fn write_digest_proof(directory: &fs::File, proof: &DigestProof) -> io::Result<()> {
    use std::os::fd::AsRawFd;
    let name = digest_proof_name(&proof.manifest_sha256, &proof.digest);
    // A fixed per-slot lock and staging file bound crash leftovers as well as
    // published records. Independent open descriptions lock across processes
    // and threads; process exit releases the lock without trusting a sidecar.
    let lock_name = std::ffi::CString::new(format!(".lock-{}", name.to_string_lossy())).unwrap();
    let lock = open_relative(
        directory,
        &lock_name,
        libc::O_RDWR | libc::O_CREAT | libc::O_NONBLOCK,
        0o600,
    )?;
    validate_private_proof_file(&lock)?;
    if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        return Err(io::Error::last_os_error());
    }
    let temporary = std::ffi::CString::new(format!(".pending-{}", name.to_string_lossy())).unwrap();
    let mut file = open_relative(
        directory,
        &temporary,
        libc::O_WRONLY | libc::O_CREAT | libc::O_NONBLOCK,
        0o600,
    )?;
    validate_private_proof_file(&file)?;
    file.set_len(0)?;
    let result = (|| {
        let bytes = serde_json::to_vec(proof).map_err(|error| invalid_data(error.to_string()))?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        if unsafe {
            libc::renameat(
                directory.as_raw_fd(),
                temporary.as_ptr(),
                directory.as_raw_fd(),
                name.as_ptr(),
            )
        } != 0
        {
            return Err(io::Error::last_os_error());
        }
        directory.sync_all()
    })();
    unsafe {
        libc::unlinkat(directory.as_raw_fd(), temporary.as_ptr(), 0);
    }
    result
}
#[cfg(unix)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
struct DigestFileIdentity {
    device: u64,
    inode: u64,
    size: u64,
    modified: (i64, i64),
    changed: (i64, i64),
    known_hfs: bool,
}

#[cfg(unix)]
fn digest_file_identity(file: &fs::File) -> io::Result<DigestFileIdentity> {
    use std::os::unix::fs::MetadataExt;
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(invalid_data("Base digest requires a regular file"));
    }
    Ok(DigestFileIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
        size: metadata.size(),
        modified: (metadata.mtime(), metadata.mtime_nsec()),
        changed: (metadata.ctime(), metadata.ctime_nsec()),
        known_hfs: file_is_known_hfs(file),
    })
}

#[cfg(unix)]
fn file_is_known_hfs(file: &fs::File) -> bool {
    #[cfg(target_os = "macos")]
    {
        use std::os::fd::AsRawFd;
        let mut info = std::mem::MaybeUninit::<libc::statfs>::uninit();
        if unsafe { libc::fstatfs(file.as_raw_fd(), info.as_mut_ptr()) } != 0 {
            return false;
        }
        let info = unsafe { info.assume_init() };
        return info
            .f_fstypename
            .iter()
            .map(|c| *c as u8)
            .take_while(|c| *c != 0)
            .eq(b"hfs".iter().copied());
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = file;
        false
    }
}

#[cfg(unix)]
static VERIFIED_FILE_DIGESTS: std::sync::LazyLock<
    std::sync::Mutex<std::collections::BTreeMap<DigestFileIdentity, String>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::BTreeMap::new()));

#[cfg(unix)]
fn digest_cache_eligible(identity: DigestFileIdentity, now_secs: Option<u64>) -> bool {
    // Unknown whole-second filesystems cannot certify metadata-only reuse.
    // An actual opened-file HFS identity has a known coarse resolution;
    // reuse waits beyond its complete bucket. Fractional metadata is aged too.
    let Some(now) = now_secs else {
        return false;
    };
    let known_coarse = identity.known_hfs && identity.changed.1 == 0;
    let fractional = (1..1_000_000_000).contains(&identity.changed.1);
    identity.changed.0 >= 0
        && (known_coarse || fractional)
        // HFS timestamps are coarse. Four seconds conservatively excludes
        // the current bucket and two-second rounding; unknown coarse filesystems
        // still cannot certify reuse. Eligibility must hold before and after hashing.
        && now.saturating_sub(identity.changed.0 as u64) >= if known_coarse { 4 } else { 2 }
}

#[cfg(unix)]
fn digest_cache_insert_eligible(
    eligible_at_start: bool,
    identity: DigestFileIdentity,
    now_secs: Option<u64>,
) -> bool {
    eligible_at_start && digest_cache_eligible(identity, now_secs)
}

#[cfg(unix)]
fn digest_now_secs() -> Option<u64> {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|time| time.as_secs())
}

#[cfg(unix)]
fn cached_file_digest(identity: DigestFileIdentity, now: Option<u64>) -> Option<String> {
    if !digest_cache_eligible(identity, now) {
        return None;
    }
    VERIFIED_FILE_DIGESTS
        .lock()
        .ok()
        .and_then(|cache| cache.get(&identity).cloned())
}

// Only memoize inside this process, never trust a persistent checksum sidecar.
// Every process performs its first full digest. File replacement or mutation
// invalidates reuse even when length and modification time are restored.
fn file_sha256(path: &Path) -> io::Result<String> {
    let mut file = fs::File::open(path)?;
    #[cfg(unix)]
    let before = digest_file_identity(&file)?;
    #[cfg(unix)]
    let started_at = digest_now_secs();
    #[cfg(unix)]
    let eligible_at_start = digest_cache_eligible(before, started_at);
    #[cfg(unix)]
    if let Some(digest) = cached_file_digest(before, started_at) {
        return Ok(digest);
    }
    let digest = hash_opened_file(&mut file)?;
    #[cfg(unix)]
    {
        if digest_file_identity(&file)? != before {
            return Err(invalid_data("Base file changed during digest verification"));
        }
        // A long read must not promote an initially fresh/rounded identity
        // into a trusted digest merely because its time bucket aged meanwhile.
        if digest_cache_insert_eligible(eligible_at_start, before, digest_now_secs()) {
            if let Ok(mut cache) = VERIFIED_FILE_DIGESTS.lock() {
                // Bound memory across long-lived query/agent processes.
                if cache.len() >= 64 {
                    cache.clear();
                }
                cache.insert(before, digest.clone());
            }
        }
    }
    Ok(digest)
}

fn write_new_synced(path: impl AsRef<Path>, bytes: &[u8]) -> io::Result<()> {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

fn set_read_only(path: &Path) -> io::Result<()> {
    let mut permissions = fs::metadata(path)?.permissions();
    permissions.set_readonly(true);
    fs::set_permissions(path, permissions)
}

fn sync_directory(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        fs::File::open(path)?.sync_all()
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(())
    }
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    fn verification_time(wall: u64, monotonic: u64) -> DigestVerificationTime {
        DigestVerificationTime {
            wall: Some(wall),
            monotonic: Some(monotonic),
        }
    }

    #[cfg(unix)]
    #[test]
    fn persistent_digest_slow_read_anchors_nonsliding_window_at_completion() {
        let seed = sample_digest_proof_at(1000);
        let started = verification_time(100, 1000);
        let completed = verification_time(180, 1080); // deterministic 80-second full read
        let proof = DigestProof::from_completed_read(
            seed.identity,
            seed.identity,
            &seed.manifest_sha256,
            &seed.digest,
            Ok(&seed.digest),
            started,
            completed,
        )
        .expect("an initially aged stable matching slow read must produce a live proof");
        assert_eq!(proof.verified_at, 180);
        assert_eq!(proof.verified_monotonic, 1080);
        for elapsed in [0, 1, 10, 29] {
            assert!(proof.matches_at(
                seed.identity,
                &seed.manifest_sha256,
                &seed.digest,
                verification_time(180 + elapsed, 1080 + elapsed)
            ));
            assert_eq!((proof.verified_at, proof.verified_monotonic), (180, 1080));
        }
        assert!(!proof.matches_at(
            seed.identity,
            &seed.manifest_sha256,
            &seed.digest,
            verification_time(210, 1110)
        ));
        // Either clock independently bounds reuse, even if the other clock stalls.
        assert!(!proof.matches_at(
            seed.identity,
            &seed.manifest_sha256,
            &seed.digest,
            verification_time(210, 1081)
        ));
        assert!(!proof.matches_at(
            seed.identity,
            &seed.manifest_sha256,
            &seed.digest,
            verification_time(181, 1110)
        ));
        assert!(!proof.matches_at(
            seed.identity,
            &seed.manifest_sha256,
            &seed.digest,
            verification_time(179, 1081)
        ));
        assert!(!proof.matches_at(
            seed.identity,
            &seed.manifest_sha256,
            &seed.digest,
            verification_time(181, 1079)
        ));
    }

    #[cfg(unix)]
    #[test]
    fn persistent_digest_completion_never_promotes_fresh_unknown_or_failed_reads() {
        let seed = sample_digest_proof_at(1000);
        let started = verification_time(100, 1000);
        let completed = verification_time(180, 1080);
        let fresh = DigestFileIdentity {
            changed: (99, 123),
            ..seed.identity
        };
        let coarse = DigestFileIdentity {
            changed: (90, 0),
            ..seed.identity
        };
        let fresh_hfs = DigestFileIdentity {
            changed: (97, 0),
            known_hfs: true,
            ..seed.identity
        };
        for identity in [fresh, coarse, fresh_hfs] {
            assert!(DigestProof::from_completed_read(
                identity,
                identity,
                &seed.manifest_sha256,
                &seed.digest,
                Ok(&seed.digest),
                started,
                completed
            )
            .is_none());
        }
        let error = io::Error::new(io::ErrorKind::UnexpectedEof, "injected failed full read");
        assert!(DigestProof::from_completed_read(
            seed.identity,
            seed.identity,
            &seed.manifest_sha256,
            &seed.digest,
            Err(&error),
            started,
            completed
        )
        .is_none());
        assert!(DigestProof::from_completed_read(
            seed.identity,
            seed.identity,
            &seed.manifest_sha256,
            &seed.digest,
            Ok("wrong digest"),
            started,
            completed
        )
        .is_none());
        for after in [
            DigestFileIdentity {
                inode: 3,
                ..seed.identity
            },
            DigestFileIdentity {
                size: 9,
                ..seed.identity
            },
            DigestFileIdentity {
                changed: (91, 123),
                ..seed.identity
            },
        ] {
            assert!(DigestProof::from_completed_read(
                seed.identity,
                after,
                &seed.manifest_sha256,
                &seed.digest,
                Ok(&seed.digest),
                started,
                completed
            )
            .is_none());
        }
        for invalid in [
            DigestVerificationTime {
                wall: None,
                ..completed
            },
            DigestVerificationTime {
                monotonic: None,
                ..completed
            },
            verification_time(99, 1080),
            verification_time(180, 999),
        ] {
            assert!(DigestProof::from_completed_read(
                seed.identity,
                seed.identity,
                &seed.manifest_sha256,
                &seed.digest,
                Ok(&seed.digest),
                started,
                invalid
            )
            .is_none());
        }
        for invalid in [
            DigestVerificationTime {
                wall: None,
                ..started
            },
            DigestVerificationTime {
                monotonic: None,
                ..started
            },
        ] {
            assert!(DigestProof::from_completed_read(
                seed.identity,
                seed.identity,
                &seed.manifest_sha256,
                &seed.digest,
                Ok(&seed.digest),
                invalid,
                completed
            )
            .is_none());
        }
    }

    #[cfg(unix)]
    fn private_proof_fixture() -> tempfile::TempDir {
        let fixture = production_proof_fixture();
        trusted_digest_directory_at(fixture.path()).expect("checked private proof fixture");
        fixture
    }

    #[cfg(unix)]
    fn production_proof_fixture() -> tempfile::TempDir {
        // Tiny operational proof metadata belongs in the ownership-enforcing
        // production namespace. Base bytes stay on the disposable test volume.
        trusted_digest_directory().expect("safe native production proof namespace");
        let home = PathBuf::from(std::env::var_os("HOME").unwrap());
        #[cfg(target_os = "macos")]
        let namespace = home.join("Library/Application Support/greppy/verified-base-digests-v1");
        #[cfg(not(target_os = "macos"))]
        let namespace = home.join(".local/share/greppy-verified-base-digests-v1");
        use std::os::unix::fs::PermissionsExt;
        let fixture = tempfile::tempdir_in(namespace).unwrap();
        fs::set_permissions(fixture.path(), fs::Permissions::from_mode(0o700)).unwrap();
        fixture
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    #[test]
    fn persistent_digest_relocated_store_child() {
        if std::env::var_os("GREPPY_TEST_RELOCATED_PROOF_HOME").is_none() {
            return;
        }
        let directory = trusted_digest_directory().unwrap();
        let proof = sample_digest_proof();
        write_digest_proof(&directory, &proof).unwrap();
        assert!(read_digest_proof(&directory, &proof.manifest_sha256, &proof.digest).is_some());
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    #[test]
    fn persistent_digest_namespace_is_independent_of_relocated_global_store() {
        use std::os::unix::fs::symlink;
        let home = private_proof_fixture();
        let share = home.path().join(".local/share");
        use std::os::unix::fs::DirBuilderExt;
        fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(&share)
            .unwrap();
        // A disposable/global graph store may be redirected. Proof metadata
        // uses a separate private namespace; no symlink is traversed for trust.
        symlink("/nonexistent-disposable-greppy-store", share.join("greppy")).unwrap();
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "base_store::tests::persistent_digest_relocated_store_child",
                "--nocapture",
            ])
            .env("HOME", home.path())
            .env("GREPPY_TEST_RELOCATED_PROOF_HOME", "1")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{} {}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(String::from_utf8_lossy(&result.stdout).contains("1 passed"));
        assert!(fs::symlink_metadata(share.join("greppy"))
            .unwrap()
            .file_type()
            .is_symlink());
    }

    #[cfg(target_os = "macos")]
    fn set_test_acl(path: &Path, acl: &str) {
        let result = std::process::Command::new("/bin/chmod")
            .args(["+a", acl])
            .arg(path)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }

    #[cfg(target_os = "macos")]
    fn clear_test_acl(path: &Path) {
        let result = std::process::Command::new("/bin/chmod")
            .arg("-N")
            .arg(path)
            .status()
            .unwrap();
        assert!(result.success());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn persistent_digest_mac_acl_rejects_writable_ancestor_directory_and_proof() {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        let fixture = production_proof_fixture();
        let ancestor = fixture.path().join("ancestor");
        let cache = ancestor.join("cache");
        fs::create_dir(&ancestor).unwrap();
        fs::create_dir(&cache).unwrap();
        for path in [&ancestor, &cache] {
            fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
        }
        assert!(trusted_digest_directory_at(&cache).is_ok());
        set_test_acl(
            &ancestor,
            "everyone allow search,add_file,add_subdirectory,delete_child",
        );
        assert_eq!(fs::metadata(&ancestor).unwrap().mode() & 0o777, 0o700);
        assert!(trusted_digest_directory_at(&cache).is_err());
        let result = std::process::Command::new("/bin/chmod")
            .arg("-N")
            .arg(&ancestor)
            .status()
            .unwrap();
        assert!(result.success());
        assert!(trusted_digest_directory_at(&cache).is_ok());
        set_test_acl(&cache, "everyone allow search,add_file,delete_child");
        assert!(trusted_digest_directory_at(&cache).is_err());
        let result = std::process::Command::new("/bin/chmod")
            .arg("-N")
            .arg(&cache)
            .status()
            .unwrap();
        assert!(result.success());
        let directory = trusted_digest_directory_at(&cache).unwrap();
        let proof = sample_digest_proof();
        write_digest_proof(&directory, &proof).unwrap();
        let path = cache.join(
            digest_proof_name(&proof.manifest_sha256, &proof.digest)
                .to_str()
                .unwrap(),
        );
        assert!(read_digest_proof(&directory, &proof.manifest_sha256, &proof.digest).is_some());
        set_test_acl(&path, "everyone allow write,append,writesecurity");
        assert_eq!(fs::metadata(&path).unwrap().mode() & 0o777, 0o600);
        assert!(validate_private_proof_file(&fs::File::open(&path).unwrap()).is_err());
        assert!(read_digest_proof(&directory, &proof.manifest_sha256, &proof.digest).is_none());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn persistent_digest_mac_acl_absence_requires_successful_native_query() {
        let fixture = production_proof_fixture();
        clear_test_acl(fixture.path());
        let directory = fs::File::open(fixture.path()).unwrap();
        assert!(reject_mutating_acl(&directory).is_ok());
        let path = fixture.path().join("no-extended-acl");
        fs::write(&path, b"private proof fixture").unwrap();
        clear_test_acl(&path);
        assert!(reject_mutating_acl(&fs::File::open(&path).unwrap()).is_ok());
        assert!(reject_mutating_acl_fd(-1).is_err());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn persistent_digest_mac_acl_query_failure_is_not_empty_acl() {
        // A query failure must never be interpreted as absent/harmless ACL.
        assert!(reject_mutating_acl_fd(-1).is_err());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn persistent_digest_mac_acl_accepts_deny_only_and_read_search() {
        let fixture = production_proof_fixture();
        set_test_acl(fixture.path(), "everyone deny delete");
        set_test_acl(
            fixture.path(),
            "everyone allow list,search,readattr,readsecurity",
        );
        let directory = trusted_digest_directory_at(fixture.path()).unwrap();
        let proof = sample_digest_proof();
        write_digest_proof(&directory, &proof).unwrap();
        let path = fixture.path().join(
            digest_proof_name(&proof.manifest_sha256, &proof.digest)
                .to_str()
                .unwrap(),
        );
        set_test_acl(&path, "everyone deny delete");
        set_test_acl(&path, "everyone allow read,readattr,readsecurity");
        assert!(validate_private_proof_file(&fs::File::open(&path).unwrap()).is_ok());
        assert!(read_digest_proof(&directory, &proof.manifest_sha256, &proof.digest).is_some());
        clear_test_acl(&path);
        clear_test_acl(fixture.path());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn persistent_digest_production_namespace_child() {
        let Some(cache) = std::env::var_os("GREPPY_TEST_PRODUCTION_PROOF_DIRECTORY") else {
            return;
        };
        let path = PathBuf::from(std::env::var_os("GREPPY_TEST_PRODUCTION_BASE_PATH").unwrap());
        let binding = hex_sha256(path.to_string_lossy().as_bytes());
        let expected = hex_sha256(b"original");
        FULL_DIGEST_READS.with(|reads| reads.set(0));
        assert_eq!(
            verified_base_digest_at(
                &path,
                &binding,
                &expected,
                Some(trusted_digest_directory_at(Path::new(&cache)).unwrap())
            )
            .unwrap(),
            expected
        );
        FULL_DIGEST_READS.with(|reads| {
            assert_eq!(
                reads.get(),
                0,
                "second command must reuse native namespace proof"
            )
        });
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn persistent_digest_native_production_namespace_reuses_across_commands() {
        let fixture = production_proof_fixture();
        let base = tempfile::tempdir().unwrap();
        let path = base.path().join("base.db");
        fs::write(&path, b"original").unwrap();
        std::thread::sleep(std::time::Duration::from_secs(5));
        let binding = hex_sha256(path.to_string_lossy().as_bytes());
        let expected = hex_sha256(b"original");
        FULL_DIGEST_READS.with(|reads| reads.set(0));
        assert_eq!(
            verified_base_digest_at(
                &path,
                &binding,
                &expected,
                Some(trusted_digest_directory_at(fixture.path()).unwrap())
            )
            .unwrap(),
            expected
        );
        FULL_DIGEST_READS.with(|reads| assert_eq!(reads.get(), 1));
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "base_store::tests::persistent_digest_production_namespace_child",
                "--nocapture",
            ])
            .env("GREPPY_TEST_PRODUCTION_PROOF_DIRECTORY", fixture.path())
            .env("GREPPY_TEST_PRODUCTION_BASE_PATH", &path)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{} {}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(String::from_utf8_lossy(&result.stdout).contains("1 passed"));
    }

    #[cfg(unix)]
    #[test]
    fn persistent_digest_real_snapshot_hit_and_write_invalidation() {
        use std::os::unix::fs::MetadataExt;
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("base.db");
        let proof_fixture = private_proof_fixture();
        let proofs = proof_fixture.path().to_path_buf();
        fs::write(&path, b"original").unwrap();
        std::thread::sleep(std::time::Duration::from_secs(5));
        let binding = hex_sha256(b"manifest");
        let expected = hex_sha256(b"original");
        let directory = || Some(fs::File::open(&proofs).unwrap());
        FULL_DIGEST_READS.with(|reads| reads.set(0));
        assert_eq!(
            verified_base_digest_at(&path, &binding, &expected, directory()).unwrap(),
            expected
        );
        assert_eq!(
            verified_base_digest_at(&path, &binding, &expected, directory()).unwrap(),
            expected
        );
        let identity = digest_file_identity(&fs::File::open(&path).unwrap()).unwrap();
        if digest_cache_eligible(identity, digest_now_secs()) {
            FULL_DIGEST_READS.with(|reads| assert_eq!(reads.get(), 1));
        } else {
            FULL_DIGEST_READS.with(|reads| assert_eq!(reads.get(), 2));
        }
        let metadata = fs::metadata(&path).unwrap();
        fs::write(&path, b"tampered").unwrap();
        fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(metadata.modified().unwrap()))
            .unwrap();
        assert_eq!(fs::metadata(&path).unwrap().size(), metadata.size());
        assert_eq!(
            verified_base_digest_at(&path, &binding, &expected, directory()).unwrap(),
            hex_sha256(b"tampered")
        );
        let proof = read_digest_proof(&fs::File::open(&proofs).unwrap(), &binding, &expected);
        if let Some(proof) = proof {
            assert!(!proof.matches(
                digest_file_identity(&fs::File::open(&path).unwrap()).unwrap(),
                &binding,
                &expected,
                digest_now_secs()
            ));
        }
    }

    #[cfg(unix)]
    #[test]
    fn persistent_digest_concurrent_base_write_never_certifies_old_bytes() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("base.db");
        let proof_fixture = private_proof_fixture();
        let proofs = proof_fixture.path().to_path_buf();
        fs::write(&path, b"original").unwrap();
        let (start_tx, start_rx) = std::sync::mpsc::channel();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let writer_path = path.clone();
        let writer = std::thread::spawn(move || {
            start_rx.recv().unwrap();
            fs::write(writer_path, b"tampered").unwrap();
            done_tx.send(()).unwrap();
        });
        BEFORE_DIGEST_READ.with(|slot| {
            *slot.borrow_mut() = Some(Box::new(move || {
                start_tx.send(()).unwrap();
                done_rx.recv().unwrap();
            }));
        });
        let result = verified_base_digest_at(
            &path,
            &hex_sha256(b"manifest"),
            &hex_sha256(b"original"),
            Some(fs::File::open(&proofs).unwrap()),
        );
        writer.join().unwrap();
        if let Ok(digest) = result {
            // Whole-second filesystems may not expose this fresh write in
            // metadata; the mandatory full read still observes the mismatch.
            assert_eq!(digest, hex_sha256(b"tampered"));
        }
        assert_eq!(fs::read_dir(proofs).unwrap().count(), 0);
    }

    #[cfg(unix)]
    #[test]
    fn persistent_digest_proof_child() {
        let Some(path) = std::env::var_os("GREPPY_TEST_DIGEST_PROOF_DIRECTORY") else {
            return;
        };
        let directory = trusted_digest_directory_at(Path::new(&path)).unwrap();
        let expected = sample_digest_proof();
        let proof =
            read_digest_proof(&directory, &expected.manifest_sha256, &expected.digest).unwrap();
        assert!(proof.matches(
            expected.identity,
            &expected.manifest_sha256,
            &expected.digest,
            Some(110)
        ));
    }

    #[cfg(unix)]
    #[test]
    fn persistent_digest_proof_survives_process_boundary() {
        let tmp = private_proof_fixture();
        let directory = trusted_digest_directory_at(tmp.path()).unwrap();
        write_digest_proof(&directory, &sample_digest_proof()).unwrap();
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "base_store::tests::persistent_digest_proof_child",
                "--nocapture",
            ])
            .env("GREPPY_TEST_DIGEST_PROOF_DIRECTORY", tmp.path())
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(String::from_utf8_lossy(&result.stdout).contains("1 passed"));
    }

    #[cfg(unix)]
    #[test]
    fn persistent_digest_mismatch_is_not_recorded_and_symlink_base_is_refused() {
        use std::os::unix::fs::symlink;
        let tmp = tempfile::tempdir().unwrap();
        let proof_fixture = private_proof_fixture();
        let proofs = proof_fixture.path().to_path_buf();
        let path = tmp.path().join("base.db");
        fs::write(&path, b"tampered").unwrap();
        let expected = hex_sha256(b"original");
        let binding = hex_sha256(b"manifest");
        let actual = verified_base_digest_at(
            &path,
            &binding,
            &expected,
            Some(fs::File::open(&proofs).unwrap()),
        )
        .unwrap();
        assert_ne!(actual, expected);
        assert_eq!(actual, hex_sha256(b"tampered"));
        assert_eq!(fs::read_dir(&proofs).unwrap().count(), 0);
        let link = tmp.path().join("symlink.db");
        symlink(&path, &link).unwrap();
        assert!(verified_base_digest_at(&link, &binding, &expected, None).is_err());
    }

    #[cfg(unix)]
    fn sample_digest_proof() -> DigestProof {
        sample_digest_proof_at(digest_monotonic_secs().unwrap())
    }

    #[cfg(unix)]
    fn sample_digest_proof_at(verified_monotonic: u64) -> DigestProof {
        DigestProof {
            version: 1,
            manifest_sha256: hex_sha256(b"manifest"),
            digest: hex_sha256(b"original"),
            identity: DigestFileIdentity {
                device: 1,
                inode: 2,
                size: 8,
                modified: (90, 123),
                changed: (90, 123),
                known_hfs: false,
            },
            verified_at: 100,
            verified_monotonic,
        }
    }

    #[cfg(unix)]
    #[test]
    fn persistent_digest_proof_bounds_age_and_exact_manifest_identity() {
        let mut proof = sample_digest_proof();
        let identity = proof.identity;
        assert!(proof.matches(identity, &proof.manifest_sha256, &proof.digest, Some(100)));
        assert!(proof.matches(identity, &proof.manifest_sha256, &proof.digest, Some(129)));
        for now in [None, Some(99), Some(130), Some(u64::MAX)] {
            assert!(!proof.matches(identity, &proof.manifest_sha256, &proof.digest, now));
        }
        assert!(!proof.matches(identity, "another manifest", &proof.digest, Some(101)));
        assert!(!proof.matches(
            identity,
            &proof.manifest_sha256,
            "another digest",
            Some(101)
        ));
        for changed in [
            DigestFileIdentity {
                inode: 3,
                ..identity
            },
            DigestFileIdentity {
                device: 3,
                ..identity
            },
            DigestFileIdentity {
                size: 9,
                ..identity
            },
            DigestFileIdentity {
                changed: (91, 123),
                ..identity
            },
            DigestFileIdentity {
                modified: (91, 123),
                ..identity
            },
        ] {
            assert!(!proof.matches(changed, &proof.manifest_sha256, &proof.digest, Some(101)));
        }
        proof.verified_monotonic = u64::MAX;
        assert!(!proof.matches(identity, &proof.manifest_sha256, &proof.digest, Some(101)));
        proof.verified_monotonic = digest_monotonic_secs().unwrap();
        proof.version = 2;
        assert!(!proof.matches(identity, &proof.manifest_sha256, &proof.digest, Some(101)));
    }

    #[cfg(unix)]
    #[test]
    fn persistent_digest_proof_rejects_unknown_coarse_and_initially_fresh_metadata() {
        let mut proof = sample_digest_proof();
        proof.identity.changed = (100, 123);
        assert!(!proof.matches(
            proof.identity,
            &proof.manifest_sha256,
            &proof.digest,
            Some(110)
        ));
        proof.identity.changed = (90, 0);
        assert!(!proof.matches(
            proof.identity,
            &proof.manifest_sha256,
            &proof.digest,
            Some(110)
        ));
        proof.identity.known_hfs = true;
        assert!(proof.matches(
            proof.identity,
            &proof.manifest_sha256,
            &proof.digest,
            Some(110)
        ));
        proof.identity.changed = (97, 0);
        assert!(!proof.matches(
            proof.identity,
            &proof.manifest_sha256,
            &proof.digest,
            Some(110)
        ));
    }

    #[cfg(unix)]
    #[test]
    fn persistent_digest_proof_rejects_corrupt_unsafe_and_symlink_records() {
        use std::os::unix::fs::{symlink, PermissionsExt};
        let tmp = private_proof_fixture();
        let directory = trusted_digest_directory_at(tmp.path()).unwrap();
        let proof = sample_digest_proof();
        write_digest_proof(&directory, &proof).unwrap();
        let name = digest_proof_name(&proof.manifest_sha256, &proof.digest);
        let path = tmp.path().join(name.to_str().unwrap());
        let loaded = read_digest_proof(&directory, &proof.manifest_sha256, &proof.digest).unwrap();
        assert!(loaded.matches(
            proof.identity,
            &proof.manifest_sha256,
            &proof.digest,
            Some(110)
        ));
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        assert!(read_digest_proof(&directory, &proof.manifest_sha256, &proof.digest).is_none());
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        fs::write(&path, b"not json").unwrap();
        assert!(read_digest_proof(&directory, &proof.manifest_sha256, &proof.digest).is_none());
        fs::write(&path, vec![b'x'; 4097]).unwrap();
        assert!(read_digest_proof(&directory, &proof.manifest_sha256, &proof.digest).is_none());
        fs::remove_file(&path).unwrap();
        let target = tmp.path().join("target");
        fs::write(&target, serde_json::to_vec(&proof).unwrap()).unwrap();
        fs::set_permissions(&target, fs::Permissions::from_mode(0o600)).unwrap();
        symlink(&target, &path).unwrap();
        assert!(read_digest_proof(&directory, &proof.manifest_sha256, &proof.digest).is_none());
        fs::remove_file(&path).unwrap();
        fs::hard_link(&target, &path).unwrap();
        assert!(read_digest_proof(&directory, &proof.manifest_sha256, &proof.digest).is_none());
    }

    #[cfg(unix)]
    #[test]
    fn persistent_digest_directory_refuses_writable_ancestors_and_symlinks() {
        use std::os::unix::fs::symlink;
        let tmp = private_proof_fixture();
        assert!(trusted_digest_directory_at(tmp.path()).is_ok());
        let link = tmp.path().join("link");
        symlink(tmp.path(), &link).unwrap();
        assert!(trusted_digest_directory_at(&link.join("proofs")).is_err());
        // tempfile fixtures are disposable, not a trusted production namespace;
        // on macOS /Volumes/tmp also has ownership explicitly disabled.
        assert!(trusted_digest_directory_at(Path::new("relative")).is_err());
        assert!(trusted_digest_directory_at(Path::new("/tmp/proofs")).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn persistent_digest_atomic_writers_never_publish_partial_records() {
        let tmp = private_proof_fixture();
        let directory = trusted_digest_directory_at(tmp.path()).unwrap();
        let proof = sample_digest_proof();
        let binding = proof.manifest_sha256.clone();
        let expected = proof.digest.clone();
        std::thread::scope(|scope| {
            for _ in 0..2 {
                let directory = &directory;
                let proof = &proof;
                scope.spawn(move || {
                    for _ in 0..8 {
                        if let Err(error) = write_digest_proof(directory, proof) {
                            assert_eq!(error.kind(), io::ErrorKind::WouldBlock, "{error}");
                        }
                    }
                });
            }
            for _ in 0..32 {
                if let Some(loaded) = read_digest_proof(&directory, &binding, &expected) {
                    assert!(loaded.matches(proof.identity, &binding, &expected, Some(110)));
                }
            }
        });
        assert!(read_digest_proof(&directory, &binding, &expected).is_some());
        assert_eq!(fs::read_dir(tmp.path()).unwrap().count(), 2);
    }

    #[cfg(unix)]
    #[test]
    fn known_hfs_cache_reuses_only_aged_keys_and_rejects_tampering() {
        let hfs = DigestFileIdentity {
            device: u64::MAX - 1,
            inode: u64::MAX - 1,
            size: 8,
            modified: (100, 0),
            changed: (100, 0),
            known_hfs: true,
        };
        let unknown = DigestFileIdentity {
            known_hfs: false,
            ..hfs
        };
        {
            let mut cache = VERIFIED_FILE_DIGESTS.lock().unwrap();
            cache.insert(hfs, hex_sha256(b"original"));
            cache.insert(unknown, hex_sha256(b"original"));
        }
        assert_eq!(
            cached_file_digest(hfs, Some(104)),
            Some(hex_sha256(b"original"))
        );
        assert_eq!(cached_file_digest(unknown, Some(200)), None);
        for now in 100..104 {
            assert_eq!(cached_file_digest(hfs, Some(now)), None);
        }
        let initially_fresh = digest_cache_eligible(hfs, Some(100));
        assert!(!digest_cache_insert_eligible(
            initially_fresh,
            hfs,
            Some(200)
        ));
        assert!(digest_cache_insert_eligible(true, hfs, Some(104)));
        assert_eq!(cached_file_digest(hfs, None), None);
        assert_eq!(cached_file_digest(hfs, Some(99)), None);
        // An aged file changed now cannot retain its old ctime even if mtime
        // and size are restored. Fresh mutations never certify a reused key.
        let changed = DigestFileIdentity {
            changed: (104, 0),
            ..hfs
        };
        assert_eq!(cached_file_digest(changed, Some(104)), None);
        assert_eq!(cached_file_digest(changed, Some(108)), None);
        let mut cache = VERIFIED_FILE_DIGESTS.lock().unwrap();
        cache.remove(&hfs);
        cache.remove(&unknown);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn opened_file_hfs_classification_matches_kernel_filesystem_type() {
        use std::os::fd::AsRawFd;
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("identity.db");
        fs::write(&path, b"fixture").unwrap();
        let file = fs::File::open(path).unwrap();
        let mut info = std::mem::MaybeUninit::<libc::statfs>::uninit();
        assert_eq!(
            unsafe { libc::fstatfs(file.as_raw_fd(), info.as_mut_ptr()) },
            0
        );
        let info = unsafe { info.assume_init() };
        let name: Vec<u8> = info
            .f_fstypename
            .iter()
            .map(|c| *c as u8)
            .take_while(|c| *c != 0)
            .collect();
        let identity = digest_file_identity(&file).unwrap();
        assert_eq!(identity.known_hfs, name.as_slice() == b"hfs");
        if identity.known_hfs && identity.changed.1 == 0 {
            assert!(!digest_cache_eligible(
                identity,
                Some(identity.changed.0 as u64)
            ));
            assert!(digest_cache_eligible(
                identity,
                Some(identity.changed.0 as u64 + 4)
            ));
        }
    }

    #[cfg(unix)]
    #[test]
    fn digest_cache_does_not_promote_fresh_identity_after_long_hash() {
        let identity = DigestFileIdentity {
            device: 1,
            inode: 1,
            size: 8,
            modified: (100, 123_456_789),
            changed: (100, 123_456_789),
            known_hfs: false,
        };
        let initially_fresh = digest_cache_eligible(identity, Some(100));
        assert!(!initially_fresh);
        assert!(digest_cache_eligible(identity, Some(103)));
        assert!(!digest_cache_insert_eligible(
            initially_fresh,
            identity,
            Some(103)
        ));
        let initially_aged = digest_cache_eligible(identity, Some(102));
        assert!(digest_cache_insert_eligible(
            initially_aged,
            identity,
            Some(103)
        ));
        assert!(!digest_cache_insert_eligible(
            initially_aged,
            identity,
            Some(100)
        ));
        assert!(!digest_cache_insert_eligible(
            initially_aged,
            identity,
            None
        ));
    }

    #[cfg(unix)]
    #[test]
    fn digest_cache_reuses_aged_fractional_metadata_and_rejects_coarse_collision() {
        let fine = DigestFileIdentity {
            device: u64::MAX,
            inode: u64::MAX,
            size: 8,
            modified: (100, 123_456_789),
            changed: (100, 123_456_789),
            known_hfs: false,
        };
        let coarse = DigestFileIdentity {
            changed: (100, 0),
            ..fine
        };
        {
            let mut cache = VERIFIED_FILE_DIGESTS.lock().unwrap();
            cache.insert(fine, hex_sha256(b"original"));
            // Simulate a stale entry with an identical coarse metadata key.
            cache.insert(coarse, hex_sha256(b"original"));
        }
        assert_eq!(
            cached_file_digest(fine, Some(102)),
            Some(hex_sha256(b"original"))
        );
        assert_eq!(
            cached_file_digest(fine, Some(103)),
            Some(hex_sha256(b"original"))
        );
        assert_eq!(cached_file_digest(coarse, Some(102)), None);
        assert_eq!(cached_file_digest(coarse, Some(200)), None);
        // Fractional resolution alone is insufficient in the current time bucket.
        assert_eq!(cached_file_digest(fine, Some(100)), None);
        assert_eq!(cached_file_digest(fine, Some(101)), None);
        assert_eq!(cached_file_digest(fine, None), None);
        assert_eq!(cached_file_digest(fine, Some(99)), None);
        let changed = DigestFileIdentity {
            changed: (102, 123_456_789),
            ..fine
        };
        assert_eq!(cached_file_digest(changed, Some(104)), None);
        let mut cache = VERIFIED_FILE_DIGESTS.lock().unwrap();
        cache.remove(&fine);
        cache.remove(&coarse);
    }

    #[cfg(unix)]
    #[test]
    fn process_digest_cache_invalidates_same_length_tamper_with_restored_mtime() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("graph.db");
        fs::write(&path, b"original").unwrap();
        let original_time = fs::metadata(&path).unwrap().modified().unwrap();
        let original = file_sha256(&path).unwrap();
        let stamp = digest_file_identity(&fs::File::open(&path).unwrap()).unwrap();
        // Seed the stale entry the previous implementation would trust. The
        // eligibility guard must reject it on fresh/coarse metadata, including
        // when the later write returns exactly the same ctime.
        VERIFIED_FILE_DIGESTS
            .lock()
            .unwrap()
            .insert(stamp, original.clone());
        assert_eq!(file_sha256(&path).unwrap(), original);

        fs::write(&path, b"tampered").unwrap();
        fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(original_time))
            .unwrap();
        let changed = digest_file_identity(&fs::File::open(&path).unwrap()).unwrap();
        assert_eq!(changed.size, stamp.size);
        assert_eq!(changed.modified, stamp.modified);
        // A whole-second filesystem can return the identical ctime here.
        // This is precisely the regression: identical metadata must not certify stale bytes.
        assert_eq!(file_sha256(&path).unwrap(), hex_sha256(b"tampered"));
        assert_ne!(file_sha256(&path).unwrap(), original);
        VERIFIED_FILE_DIGESTS.lock().unwrap().remove(&stamp);
    }

    #[cfg(unix)]
    #[test]
    fn process_digest_cache_invalidates_replaced_inode() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("summary.db");
        fs::write(&path, b"original").unwrap();
        let before = digest_file_identity(&fs::File::open(&path).unwrap()).unwrap();
        let original = file_sha256(&path).unwrap();
        let replacement = tmp.path().join("replacement");
        fs::write(&replacement, b"replaced").unwrap();
        fs::rename(&replacement, &path).unwrap();
        let after = digest_file_identity(&fs::File::open(&path).unwrap()).unwrap();
        assert_ne!(before.inode, after.inode);
        assert_eq!(file_sha256(&path).unwrap(), hex_sha256(b"replaced"));
        assert_ne!(file_sha256(&path).unwrap(), original);
    }

    fn identity() -> BaseStoreIdentity {
        BaseStoreIdentity {
            format_version: BASE_STORE_FORMAT_VERSION,
            canonical_repository_identity: "repo-common-dir:abc".into(),
            git_object_format: "sha1".into(),
            base_tree_oid: "a".repeat(40),
            store_schema_version: 15,
            indexer_version: "indexer-v4".into(),
            parser_and_extractor_versions: "parser-v1/extractor-v1".into(),
            summary_model_and_prompt_version: "qwen/prompt-v1".into(),
            embedding_model: "embeddinggemma".into(),
            embedding_prompt_version: "code-v1".into(),
            embedding_dimensions: 768,
            embedding_encoding: "i8-v1".into(),
        }
    }

    fn empty_summary_cache(root: &Path) -> PathBuf {
        let directory = root.join("staged-summary");
        drop(crate::SummaryCache::open(&directory).unwrap());
        directory.join(crate::SUMMARY_CACHE_DB_FILE)
    }

    #[test]
    fn identity_hash_is_deterministic_and_covers_semantic_inputs() {
        let a = identity();
        let mut b = a.clone();
        assert_eq!(a.hash().unwrap(), b.hash().unwrap());
        b.embedding_prompt_version.push_str("-changed");
        assert_ne!(a.hash().unwrap(), b.hash().unwrap());
    }

    #[test]
    fn layout_is_scoped_by_repository_and_complete_identity() {
        let layout = BaseStoreLayout::new(Path::new("/cache"), &identity()).unwrap();
        assert!(layout.directory.starts_with("/cache/agent-base-stores/v1"));
        assert_eq!(layout.graph.file_name().unwrap(), "graph.db");
        assert_eq!(layout.complete.file_name().unwrap(), COMPLETE_FILE);
    }

    #[test]
    fn builder_lease_uses_the_layouts_injected_data_root() {
        let tmp = tempfile::tempdir().unwrap();
        let layout = BaseStoreLayout::new(tmp.path(), &identity()).unwrap();
        let lease = layout.acquire_builder(true).unwrap().unwrap();
        assert!(lease._lock.path().starts_with(tmp.path().join("locks")));
        assert!(layout.acquire_builder(true).unwrap().is_none());
    }

    #[test]
    fn live_reader_lease_blocks_builder_and_eviction() {
        let tmp = tempfile::tempdir().unwrap();
        let layout = BaseStoreLayout::new(tmp.path(), &identity()).unwrap();
        let reader = layout.acquire_reader(false).unwrap().unwrap();
        assert!(layout.acquire_reader(true).unwrap().is_some());
        assert!(
            layout.acquire_builder(true).unwrap().is_none(),
            "exclusive rebuild/eviction lease must not pass a live reader"
        );
        drop(reader);
        assert!(layout.acquire_builder(true).unwrap().is_some());
    }

    #[test]
    fn ten_followers_publish_exactly_one_base_generation() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::{Arc, Barrier};

        let tmp = tempfile::tempdir().unwrap();
        let staged = tmp.path().join("staged.db");
        fs::write(&staged, b"one immutable graph generation").unwrap();
        let summary = empty_summary_cache(tmp.path());
        let layout = Arc::new(BaseStoreLayout::new(tmp.path(), &identity()).unwrap());
        let barrier = Arc::new(Barrier::new(10));
        let builders = Arc::new(AtomicUsize::new(0));
        let mut threads = Vec::new();
        for _ in 0..10 {
            let layout = Arc::clone(&layout);
            let barrier = Arc::clone(&barrier);
            let builders = Arc::clone(&builders);
            let staged = staged.clone();
            let summary = summary.clone();
            threads.push(std::thread::spawn(move || {
                barrier.wait();
                let _lease = layout.acquire_builder(false).unwrap().unwrap();
                if layout.read_verified_manifest().is_err() {
                    builders.fetch_add(1, Ordering::SeqCst);
                    layout
                        .publish_graph_with_summary(identity(), &staged, &summary)
                        .unwrap();
                }
                layout.read_verified_manifest().unwrap()
            }));
        }
        let manifests = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(builders.load(Ordering::SeqCst), 1);
        assert!(manifests.iter().all(|manifest| manifest == &manifests[0]));
    }

    #[test]
    fn visibility_rejects_escapes_and_overlap() {
        assert!(VisibilityIndex::new(["../secret".into()], []).is_err());
        assert!(VisibilityIndex::new(["src/a.rs".into()], ["src/a.rs".into()]).is_err());
        let visibility = VisibilityIndex::new(
            ["./src/a.rs".into(), "src/b.rs".into()],
            ["src/deleted.rs".into()],
        )
        .unwrap();
        assert!(visibility.hides_base_path("src/a.rs"));
        assert!(visibility.hides_base_path("src/deleted.rs"));
        assert_eq!(visibility.changed_count(), 3);
    }

    #[test]
    fn verified_manifest_requires_matching_complete_marker_and_graph() {
        let tmp = tempfile::tempdir().unwrap();
        let layout = BaseStoreLayout::new(tmp.path(), &identity()).unwrap();
        fs::create_dir_all(&layout.directory).unwrap();
        greppy_core::cache::write_agent_base_manifest(
            &layout.directory,
            &identity().hash().unwrap(),
            &identity().canonical_repository_identity,
        )
        .unwrap();
        fs::write(&layout.graph, b"sqlite").unwrap();
        let summary = empty_summary_cache(tmp.path());
        fs::copy(&summary, &layout.summary_cache).unwrap();
        let manifest = BaseStoreManifest {
            identity: identity(),
            identity_hash: identity().hash().unwrap(),
            graph_sha256: hex_sha256(b"sqlite"),
            summary_cache_sha256: file_sha256(&layout.summary_cache).unwrap(),
            published_at_unix_secs: 1,
        };
        fs::write(&layout.manifest, serde_json::to_vec(&manifest).unwrap()).unwrap();
        fs::write(&layout.complete, &manifest.identity_hash).unwrap();
        assert_eq!(layout.read_verified_manifest().unwrap(), manifest);
        fs::write(&layout.complete, "wrong").unwrap();
        assert!(layout.read_verified_manifest().is_err());
    }

    #[test]
    fn corrupt_base_is_quarantined_before_republication() {
        let tmp = tempfile::tempdir().unwrap();
        let layout = BaseStoreLayout::new(tmp.path(), &identity()).unwrap();
        fs::create_dir_all(&layout.directory).unwrap();
        fs::write(&layout.graph, b"corrupt graph").unwrap();
        fs::write(&layout.complete, "wrong identity").unwrap();

        let _lease = layout.acquire_builder(false).unwrap().unwrap();
        let quarantine = layout
            .quarantine_invalid()
            .unwrap()
            .expect("invalid Base must be quarantined");
        assert!(!layout.directory.exists());
        assert!(quarantine.is_dir());
        assert_eq!(
            fs::read(quarantine.join("graph.db")).unwrap(),
            b"corrupt graph"
        );

        let staged = tmp.path().join("staged.db");
        fs::write(&staged, b"replacement graph").unwrap();
        let summary = empty_summary_cache(tmp.path());
        let manifest = layout
            .publish_graph_with_summary(identity(), &staged, &summary)
            .unwrap();
        assert_eq!(layout.read_verified_manifest().unwrap(), manifest);
        assert!(layout.quarantine_invalid().unwrap().is_none());
    }

    #[test]
    fn publisher_is_atomic_idempotent_and_read_only() {
        let tmp = tempfile::tempdir().unwrap();
        let layout = BaseStoreLayout::new(tmp.path(), &identity()).unwrap();
        let staged = tmp.path().join("staged.db");
        fs::write(&staged, b"sqlite graph bytes").unwrap();
        let summary = empty_summary_cache(tmp.path());
        let _lease = layout.acquire_builder(false).unwrap().unwrap();
        let first = layout
            .publish_graph_with_summary(identity(), &staged, &summary)
            .unwrap();
        let second = layout
            .publish_graph_with_summary(identity(), &staged, &summary)
            .unwrap();
        assert_eq!(first, second);
        assert!(fs::metadata(&layout.graph)
            .unwrap()
            .permissions()
            .readonly());
        assert_eq!(layout.read_verified_manifest().unwrap(), first);
        fs::write(&layout.graph, b"tamper").unwrap_err();
    }

    #[test]
    fn publisher_includes_and_verifies_immutable_summary_cache() {
        let tmp = tempfile::tempdir().unwrap();
        let layout = BaseStoreLayout::new(tmp.path(), &identity()).unwrap();
        let staged = tmp.path().join("staged.db");
        fs::write(&staged, b"sqlite graph bytes").unwrap();
        let summary_dir = tmp.path().join("summary");
        let cache = crate::SummaryCache::open(&summary_dir).unwrap();
        cache
            .put_unbounded("model#sc1", "span", &["shared purpose".into()])
            .unwrap();
        drop(cache);
        let _lease = layout.acquire_builder(false).unwrap().unwrap();
        let manifest = layout
            .publish_graph_with_summary(
                identity(),
                &staged,
                &summary_dir.join(crate::SUMMARY_CACHE_DB_FILE),
            )
            .unwrap();
        assert_eq!(manifest.summary_cache_sha256.len(), 64);
        assert_eq!(layout.read_verified_manifest().unwrap(), manifest);
        let published = crate::SummaryCache::open_read_only(&layout.directory).unwrap();
        assert_eq!(
            published.get("model#sc1", "span").unwrap(),
            Some(vec!["shared purpose".into()])
        );
        fs::write(&layout.summary_cache, b"tamper").unwrap_err();
    }

    fn project(root: &str) -> crate::Project {
        crate::Project {
            name: "p".into(),
            indexed_at: "now".into(),
            root_path: root.into(),
        }
    }

    fn node(qname: &str, file: &str, line: i64) -> crate::NewNode {
        crate::NewNode {
            project: "p".into(),
            label: "Function".into(),
            name: qname.rsplit('.').next().unwrap().into(),
            qualified_name: qname.into(),
            file_path: file.into(),
            start_line: line,
            end_line: line + 1,
            properties: serde_json::json!({}),
        }
    }

    #[test]
    fn overlay_hides_dirty_base_rows_and_remaps_unchanged_edges() {
        let tmp = tempfile::tempdir().unwrap();
        let base_path = tmp.path().join("base.db");
        let delta_path = tmp.path().join("delta.db");
        {
            let mut base = Store::open(&base_path).unwrap();
            base.upsert_project(&project("/base")).unwrap();
            let caller = base.insert_node(&node("p.caller", "src/a.rs", 1)).unwrap();
            let old_target = base.insert_node(&node("p.target", "src/b.rs", 2)).unwrap();
            base.insert_edge(&crate::NewEdge {
                project: "p".into(),
                source_id: caller,
                target_id: old_target,
                edge_type: "CALLS".into(),
                properties: serde_json::json!({"line": 3}),
            })
            .unwrap();
            base.insert_file_content_rows(
                "p",
                "src/a.rs",
                &[crate::ContentRow {
                    line: 3,
                    snippet: "target(); // clean base caller".into(),
                }],
            )
            .unwrap();
            base.insert_file_content_rows(
                "p",
                "src/b.rs",
                &[crate::ContentRow {
                    line: 2,
                    snippet: "fn target() { old_body(); }".into(),
                }],
            )
            .unwrap();
        }
        {
            let mut delta = Store::open(&delta_path).unwrap();
            delta.upsert_project(&project("/delta")).unwrap();
            delta
                .insert_node(&node("p.target", "src/b.rs", 20))
                .unwrap();
            delta
                .insert_file_content_rows(
                    "p",
                    "src/b.rs",
                    &[crate::ContentRow {
                        line: 20,
                        snippet: "fn target() { new_body(); }".into(),
                    }],
                )
                .unwrap();
        }

        let visibility = VisibilityIndex::new(["src/b.rs".into()], []).unwrap();
        let mut view = StoreView::open_overlay(&base_path, &delta_path, visibility).unwrap();
        let store = view.store();
        let caller = store.get_node_by_qname("p", "p.caller").unwrap().unwrap();
        let target = store.get_node_by_qname("p", "p.target").unwrap().unwrap();
        assert!(
            caller.id < 0,
            "unchanged Base ids use the negative namespace"
        );
        assert!(
            target.id > 0,
            "dirty Delta ids keep their positive namespace"
        );
        assert_eq!(target.start_line, 20);
        let edges = store.outgoing_edges(caller.id, Some("CALLS"), 10).unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].target_id, target.id);
        let content = store.search_file_content("p", "target", 10).unwrap();
        assert_eq!(content.len(), 2);
        assert!(content
            .iter()
            .any(|hit| hit.rel_path == "src/a.rs" && hit.line == 3));
        assert!(content
            .iter()
            .any(|hit| hit.rel_path == "src/b.rs" && hit.line == 20));
        assert!(!content.iter().any(|hit| hit.line == 2));
        assert_eq!(store.count_file_content_matches("p", "target").unwrap(), 2);
        let symbol_hits = crate::fts::search_fts_in_project(store, "p", "target", 10).unwrap();
        assert_eq!(symbol_hits.len(), 1);
        assert_eq!(symbol_hits[0].node_id, target.id);
        assert_eq!(
            crate::fts::count_fts_in_project(store, "p", "target").unwrap(),
            1
        );

        let added = view
            .store_mut()
            .insert_node(&node("p.added", "src/b.rs", 30))
            .unwrap();
        assert!(added > 0);
        let logical_edge = view
            .store_mut()
            .insert_edge(&crate::NewEdge {
                project: "p".into(),
                source_id: added,
                target_id: caller.id,
                edge_type: "CALLS".into(),
                properties: serde_json::json!({"line": 31}),
            })
            .unwrap();
        assert!(logical_edge > 0);
        let cross_layer = view
            .store()
            .outgoing_edges(added, Some("CALLS"), 10)
            .unwrap();
        assert_eq!(cross_layer.len(), 1);
        assert_eq!(cross_layer[0].target_id, caller.id);
        drop(view);
        let delta = Store::open_with(&delta_path, crate::OpenOptions::read_only()).unwrap();
        let physical_edges: i64 = delta
            .conn()
            .query_row("SELECT COUNT(*) FROM edges", [], |row| row.get(0))
            .unwrap();
        let logical_edges: i64 = delta
            .conn()
            .query_row("SELECT COUNT(*) FROM overlay_edges", [], |row| row.get(0))
            .unwrap();
        assert_eq!(physical_edges, 0, "cross-layer ids never enter edges");
        assert_eq!(logical_edges, 1, "Delta persists one logical edge");
        assert!(delta.get_node_by_qname("p", "p.added").unwrap().is_some());
        let base = Store::open_with(&base_path, crate::OpenOptions::read_only()).unwrap();
        assert!(base.get_node_by_qname("p", "p.added").unwrap().is_none());
    }

    #[test]
    fn fifty_overlay_agents_stress_private_delta_isolation() {
        use std::sync::{Arc, Barrier};

        let tmp = tempfile::tempdir().unwrap();
        let base_path = tmp.path().join("base.db");
        {
            let mut base = Store::open(&base_path).unwrap();
            base.upsert_project(&project("/base")).unwrap();
            base.insert_node(&node("p.shared", "src/shared.rs", 1))
                .unwrap();
        }
        let base_before = fs::read(&base_path).unwrap();
        let mut permissions = fs::metadata(&base_path).unwrap().permissions();
        permissions.set_readonly(true);
        fs::set_permissions(&base_path, permissions).unwrap();

        let barrier = Arc::new(Barrier::new(50));
        let mut agents = Vec::new();
        for index in 0..50 {
            let barrier = Arc::clone(&barrier);
            let base_path = base_path.clone();
            let delta_path = tmp.path().join(format!("agent-{index}.db"));
            agents.push(std::thread::spawn(move || {
                barrier.wait();
                let mut store =
                    Store::open_overlay(&base_path, &delta_path, &VisibilityIndex::default())
                        .unwrap();
                store.upsert_project(&project("/delta")).unwrap();
                let own_qname = format!("p.agent_{index}");
                store
                    .insert_node(&node(
                        &own_qname,
                        &format!("src/agent_{index}.rs"),
                        index as i64 + 10,
                    ))
                    .unwrap();
                let visible = store.list_nodes("p", "", "", 0, usize::MAX).unwrap();
                assert_eq!(visible.len(), 2);
                for node in &visible {
                    let expected: Vec<_> = visible
                        .iter()
                        .filter(|other| other.file_path == node.file_path)
                        .cloned()
                        .collect();
                    assert_eq!(
                        store.list_nodes_for_file("p", &node.file_path).unwrap(),
                        expected
                    );
                }
                assert!(visible
                    .iter()
                    .any(|candidate| candidate.qualified_name == "p.shared"));
                assert!(visible
                    .iter()
                    .any(|candidate| candidate.qualified_name == own_qname));
                assert!(!visible.iter().any(|candidate| {
                    candidate.qualified_name.starts_with("p.agent_")
                        && candidate.qualified_name != own_qname
                }));
                store
                    .conn()
                    .query_row("SELECT COUNT(*) FROM main.nodes", [], |row| {
                        row.get::<_, i64>(0)
                    })
                    .unwrap()
            }));
        }
        for agent in agents {
            assert_eq!(agent.join().unwrap(), 1);
        }
        assert_eq!(fs::read(&base_path).unwrap(), base_before);
    }
}
