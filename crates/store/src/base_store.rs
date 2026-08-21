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
        let directory = data_root
            .join("agent-base-stores")
            .join(format!("v{BASE_STORE_FORMAT_VERSION}"))
            .join(repo_hash)
            .join(identity_hash);
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
        let actual_graph_sha256 = file_sha256(&self.graph)?;
        if actual_graph_sha256 != manifest.graph_sha256 {
            return Err(invalid_data("Base Store graph digest mismatch"));
        }
        if !self.summary_cache.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "Base Store summary_cache.db is missing",
            ));
        }
        if file_sha256(&self.summary_cache)? != manifest.summary_cache_sha256 {
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

fn file_sha256(path: &Path) -> io::Result<String> {
    let mut file = fs::File::open(path)?;
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
