use crate::path_policy::validate_portable_component;
use crate::repository_layers::{self, LayerKind};
use crate::repository_tracker;
use crate::{BaselineSnapshot, ChunkGcReport, ChunkId, ChunkStore, Error, Result, CHUNK_SIZE};
use rusqlite::{params, Connection, OptionalExtension, Transaction, TransactionBehavior};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::ops::{Deref, DerefMut};
use std::path::{Component, Path, PathBuf};
#[cfg(test)]
use std::process::Command;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NodeKind {
    File,
    Directory,
    Symlink,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeMetadata {
    pub kind: NodeKind,
    pub mode: u32,
    pub size: u64,
    pub inode: u64,
    pub nlink: u32,
    pub accessed_unix_ns: i64,
    pub modified_unix_ns: i64,
    pub changed_unix_ns: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectoryEntry {
    pub name: String,
    pub metadata: NodeMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WorkspaceStatus {
    pub id: String,
    pub base_commit: String,
    pub baseline_hash: String,
    pub state: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProposalRecord {
    pub ref_name: String,
    pub repository: PathBuf,
    pub base_commit: String,
    pub baseline_hash: String,
    pub baseline_tree: String,
    pub final_tree: String,
    pub proposal_commit: String,
    pub baseline: BaselineSnapshot,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceHandle {
    id: String,
}

/// Process-lifetime ownership of a content/Git workspace pair.
///
/// The lock survives every SQLite transaction and provider callback. A newly
/// opened core may therefore distinguish a live Agent from a fully committed
/// pair whose owner crashed. Lease files are intentionally stable: removing a
/// lock pathname while another process still has its inode open can split
/// contenders across two different locks.
pub struct WorkspacePairLease {
    file: File,
    content_id: String,
}

/// Exclusive process-lifetime lease for repository-visible operations.
///
/// Apply, proposal publication and their recovery paths use the same
/// canonical-repository key. The kernel releases the lease on process death;
/// stable lease pathnames are never removed.
pub struct WorkspaceOperationLease {
    file: File,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceFileHandle {
    workspace_id: String,
    backing: WorkspaceFileBacking,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum WorkspaceFileBacking {
    Private {
        inode: u64,
    },
    Immutable {
        visible_path: String,
        origin_path: String,
        metadata: NodeMetadata,
        chunks: Vec<ChunkId>,
    },
}

impl std::fmt::Debug for WorkspacePairLease {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WorkspacePairLease")
            .field("content_id", &self.content_id)
            .finish_non_exhaustive()
    }
}

impl Drop for WorkspacePairLease {
    fn drop(&mut self) {
        unlock_pair_lease(&self.file);
    }
}

impl Drop for WorkspaceOperationLease {
    fn drop(&mut self) {
        unlock_pair_lease(&self.file);
    }
}

impl WorkspaceHandle {
    pub fn id(&self) -> &str {
        &self.id
    }
}

#[derive(Debug)]
struct InodeRecord {
    id: i64,
    kind: NodeKind,
    size: u64,
    chunks: Vec<ChunkId>,
}

/// Adapter-neutral namespace and lifecycle engine. The platform mount layers
/// translate their callbacks into these operations and contain no CoW policy.
pub struct WorkspaceCore {
    root: PathBuf,
    chunks: ChunkStore,
    metadata: ConnectionPool,
    metadata_writer: Mutex<()>,
    promoted_origins: Mutex<HashMap<(String, String), u64>>,
    _session_lease: File,
}

impl Drop for WorkspaceCore {
    fn drop(&mut self) {
        // Do not rely on field-drop ordering for the process-lifetime lease.
        // In particular, recovery must be able to acquire exclusivity as soon
        // as the last core instance has finished using metadata and chunks.
        unlock_core_lease(&self._session_lease);
    }
}

struct ConnectionPool {
    path: PathBuf,
    idle: Mutex<Vec<Connection>>,
}

struct PooledConnection<'a> {
    pool: &'a ConnectionPool,
    connection: Option<Connection>,
}

impl ConnectionPool {
    fn new(path: PathBuf, connection: Connection) -> Self {
        Self {
            path,
            idle: Mutex::new(vec![connection]),
        }
    }

    fn acquire(&self) -> Result<PooledConnection<'_>> {
        let connection = self
            .idle
            .lock()
            .map_err(|_| Error::Corrupt("workspace connection pool poisoned".into()))?
            .pop()
            .map(Ok)
            .unwrap_or_else(|| open_metadata_connection(&self.path))?;
        Ok(PooledConnection {
            pool: self,
            connection: Some(connection),
        })
    }
}

impl Deref for PooledConnection<'_> {
    type Target = Connection;

    fn deref(&self) -> &Self::Target {
        self.connection.as_ref().expect("pooled connection present")
    }
}

impl DerefMut for PooledConnection<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.connection.as_mut().expect("pooled connection present")
    }
}

impl Drop for PooledConnection<'_> {
    fn drop(&mut self) {
        let Some(connection) = self.connection.take() else {
            return;
        };
        if let Ok(mut idle) = self.pool.idle.lock() {
            idle.push(connection);
        }
    }
}

impl WorkspaceCore {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root)?;
        let session_lease = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(root.join("workspace-core.lock"))?;
        let recovering = lock_core_lease(&session_lease, true, true)?;
        if recovering {
            write_core_health(&session_lease, b"recovering-v1\n")?;
        } else {
            lock_core_lease(&session_lease, false, false)?;
            require_healthy_core_marker(&session_lease)?;
        }
        let chunks = ChunkStore::open(&root)?;
        let metadata_path = root.join("workspace.sqlite3");
        let connection = open_metadata_connection(&metadata_path)?;
        connection.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=FULL;
             PRAGMA foreign_keys=ON;
             CREATE TABLE IF NOT EXISTS cow_workspaces (
                 id TEXT PRIMARY KEY,
                 repository TEXT NOT NULL,
                 base_commit TEXT NOT NULL,
                 baseline_hash TEXT NOT NULL,
                 baseline_json BLOB NOT NULL,
                 state TEXT NOT NULL CHECK(state IN ('ready', 'kept', 'broken'))
             );
             CREATE TABLE IF NOT EXISTS cow_workspace_pairs (
                 content_id TEXT PRIMARY KEY,
                 git_id TEXT NOT NULL UNIQUE,
                 state TEXT NOT NULL CHECK(state IN ('creating', 'ready', 'kept', 'removing'))
             );
             CREATE TABLE IF NOT EXISTS cow_inodes (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 workspace_id TEXT NOT NULL REFERENCES cow_workspaces(id) ON DELETE CASCADE,
                 kind TEXT NOT NULL CHECK(kind IN ('file', 'directory', 'symlink')),
                 mode INTEGER NOT NULL,
                 size INTEGER NOT NULL CHECK(size >= 0),
                 accessed_unix_ns INTEGER NOT NULL,
                 modified_unix_ns INTEGER NOT NULL,
                 changed_unix_ns INTEGER NOT NULL,
                 chunks_json BLOB NOT NULL
             );
             CREATE TABLE IF NOT EXISTS cow_entries (
                 workspace_id TEXT NOT NULL REFERENCES cow_workspaces(id) ON DELETE CASCADE,
                 path TEXT NOT NULL,
                 inode_id INTEGER REFERENCES cow_inodes(id),
                 tombstone INTEGER NOT NULL DEFAULT 0 CHECK(tombstone IN (0, 1)),
                 PRIMARY KEY(workspace_id, path),
                 CHECK((tombstone = 1 AND inode_id IS NULL) OR
                       (tombstone = 0 AND inode_id IS NOT NULL))
             );
             CREATE INDEX IF NOT EXISTS cow_entries_inode
                 ON cow_entries(workspace_id, inode_id);
             CREATE TABLE IF NOT EXISTS cow_inode_origins (
                 workspace_id TEXT NOT NULL REFERENCES cow_workspaces(id) ON DELETE CASCADE,
                 origin_path TEXT NOT NULL,
                 inode_id INTEGER NOT NULL REFERENCES cow_inodes(id) ON DELETE CASCADE,
                 PRIMARY KEY(workspace_id, origin_path),
                 UNIQUE(workspace_id, inode_id)
             );
             CREATE TABLE IF NOT EXISTS cow_redirects (
                 workspace_id TEXT NOT NULL REFERENCES cow_workspaces(id) ON DELETE CASCADE,
                 destination TEXT NOT NULL,
                 source TEXT NOT NULL,
                 PRIMARY KEY(workspace_id, destination)
             );
             CREATE TABLE IF NOT EXISTS cow_journal (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 workspace_id TEXT NOT NULL,
                 operation TEXT NOT NULL,
                 state TEXT NOT NULL,
                 payload BLOB NOT NULL
             );
             CREATE TABLE IF NOT EXISTS cow_proposals (
                 ref_name TEXT PRIMARY KEY,
                 repository TEXT NOT NULL,
                 base_commit TEXT NOT NULL,
                 baseline_hash TEXT NOT NULL,
                 baseline_tree TEXT NOT NULL,
                 final_tree TEXT NOT NULL,
                 proposal_commit TEXT NOT NULL,
                 baseline_json BLOB NOT NULL
             );",
        )?;
        repository_layers::install_schema(&connection)?;
        repository_tracker::install_schema(&connection)?;
        let core = Self {
            root,
            chunks,
            metadata: ConnectionPool::new(metadata_path, connection),
            metadata_writer: Mutex::new(()),
            promoted_origins: Mutex::new(HashMap::new()),
            _session_lease: session_lease,
        };
        if recovering {
            core.recover()?;
            write_core_health(&core._session_lease, b"healthy-v1\n")?;
            downgrade_core_lease(&core._session_lease)?;
        }
        Ok(core)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn chunks(&self) -> &ChunkStore {
        &self.chunks
    }

    pub fn open_workspace(&self, id: &str) -> Result<WorkspaceHandle> {
        validate_workspace_id(id)?;
        let connection = self.lock_metadata()?;
        let exists: bool = connection.query_row(
            "SELECT EXISTS(SELECT 1 FROM cow_workspaces WHERE id = ?1 AND state != 'broken')",
            params![id],
            |row| row.get(0),
        )?;
        if !exists {
            return Err(Error::NotFound(format!("unknown or broken workspace {id}")));
        }
        Ok(WorkspaceHandle { id: id.into() })
    }

    pub fn create_workspace(
        &self,
        id: &str,
        baseline: BaselineSnapshot,
    ) -> Result<WorkspaceHandle> {
        self.create_workspace_internal(id, baseline, true, false)
    }

    /// Create a provider-managed namespace over a synthetic empty base. Its
    /// immutable entries and subsequent delta use exactly the same CAS,
    /// journaling and recovery path as repository workspaces, but no Git tree
    /// is imported behind the caller's back.
    pub fn create_overlay_workspace(
        &self,
        id: &str,
        baseline: BaselineSnapshot,
    ) -> Result<WorkspaceHandle> {
        self.create_workspace_internal(id, baseline, true, true)
    }

    fn create_workspace_internal(
        &self,
        id: &str,
        baseline: BaselineSnapshot,
        captured_snapshot_owns_chunks: bool,
        empty_base: bool,
    ) -> Result<WorkspaceHandle> {
        validate_workspace_id(id)?;
        let repository = baseline
            .repository
            .to_str()
            .ok_or_else(|| Error::UnsupportedRepository("repository path is not UTF-8".into()))?;
        let baseline_json = serde_json::to_vec(&baseline)?;
        let _writer = self.lock_metadata_writer()?;
        let mut connection = self.lock_metadata()?;
        let (base_id, dirty_layer_id) = repository_layers::ensure_layers(
            &mut connection,
            &self.chunks,
            &baseline,
            captured_snapshot_owns_chunks,
            empty_base,
        )?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        if empty_base {
            repository_layers::retain_overlay_template(
                &transaction,
                &baseline.baseline_hash,
                &dirty_layer_id,
            )?;
        }
        transaction.execute(
            "INSERT INTO cow_workspaces(
                 id, repository, base_commit, baseline_hash, baseline_json, state
             ) VALUES(?1, ?2, ?3, ?4, ?5, 'ready')",
            params![
                id,
                repository,
                baseline.base_commit,
                baseline.baseline_hash,
                baseline_json
            ],
        )?;
        repository_layers::link_workspace(&transaction, id, &base_id, &dirty_layer_id)?;
        transaction.commit()?;
        Ok(WorkspaceHandle { id: id.into() })
    }

    /// Creates another namespace over an already pinned immutable baseline.
    /// The new namespace acquires its own snapshot references before the
    /// regular namespace references, so removing either workspace cannot make
    /// chunks used by the other collectible.
    pub fn create_workspace_from_shared_baseline(
        &self,
        id: &str,
        baseline: &BaselineSnapshot,
    ) -> Result<WorkspaceHandle> {
        self.create_workspace_internal(id, baseline.clone(), false, false)
    }

    /// Clone a provider-managed control namespace from an already retained
    /// immutable overlay without copying its file bytes.
    pub fn create_overlay_workspace_from_shared_baseline(
        &self,
        id: &str,
        baseline: &BaselineSnapshot,
    ) -> Result<WorkspaceHandle> {
        self.create_workspace_internal(id, baseline.clone(), false, true)
    }

    /// Begin the crash-recoverable creation of the content/Git-state pair.
    /// Neither namespace is considered an Agent worktree until `complete` has
    /// atomically proven that both exist.
    pub fn begin_workspace_pair(
        &self,
        content_id: &str,
        git_id: &str,
    ) -> Result<WorkspacePairLease> {
        validate_workspace_id(content_id)?;
        validate_workspace_id(git_id)?;
        if content_id == git_id {
            return Err(Error::InvalidPath(
                "content and Git workspace IDs must differ".into(),
            ));
        }
        let lease = self
            .acquire_workspace_pair_lease(content_id, true)?
            .ok_or_else(|| {
                Error::AlreadyExists(format!("workspace pair {content_id} is already active"))
            })?;
        let _writer = self.lock_metadata_writer()?;
        let connection = self.lock_metadata()?;
        connection.execute(
            "INSERT INTO cow_workspace_pairs(content_id, git_id, state)
             VALUES(?1, ?2, 'creating')",
            params![content_id, git_id],
        )?;
        Ok(lease)
    }

    fn acquire_workspace_pair_lease(
        &self,
        content_id: &str,
        nonblocking: bool,
    ) -> Result<Option<WorkspacePairLease>> {
        validate_workspace_id(content_id)?;
        let directory = self.root.join("pair-leases");
        fs::create_dir_all(&directory)?;
        let path = directory.join(format!("{content_id}.lease"));
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(path)?;
        if !lock_pair_lease(&file, nonblocking)? {
            return Ok(None);
        }
        Ok(Some(WorkspacePairLease {
            file,
            content_id: content_id.to_string(),
        }))
    }

    pub fn try_repository_operation_lease(
        &self,
        repository: &Path,
    ) -> Result<Option<WorkspaceOperationLease>> {
        let canonical = repository.canonicalize()?;
        let text = canonical
            .to_str()
            .ok_or_else(|| Error::UnsupportedRepository("repository path is not UTF-8".into()))?;
        let key = blake3::hash(text.as_bytes()).to_hex();
        let directory = self.root.join("operation-leases");
        fs::create_dir_all(&directory)?;
        let path = directory.join(format!("repository-{key}.lease"));
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(path)?;
        if !lock_pair_lease(&file, true)? {
            return Ok(None);
        }
        Ok(Some(WorkspaceOperationLease { file }))
    }

    pub fn complete_workspace_pair(
        &self,
        content: &WorkspaceHandle,
        git: &WorkspaceHandle,
    ) -> Result<()> {
        let _writer = self.lock_metadata_writer()?;
        let mut connection = self.lock_metadata()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        for id in [&content.id, &git.id] {
            let ready: bool = transaction.query_row(
                "SELECT EXISTS(SELECT 1 FROM cow_workspaces WHERE id = ?1 AND state = 'ready')",
                params![id],
                |row| row.get(0),
            )?;
            if !ready {
                return Err(Error::NotFound(format!(
                    "workspace pair member {id} is not ready"
                )));
            }
        }
        let changed = transaction.execute(
            "UPDATE cow_workspace_pairs SET state = 'ready'
             WHERE content_id = ?1 AND git_id = ?2 AND state = 'creating'",
            params![content.id, git.id],
        )?;
        if changed != 1 {
            return Err(Error::Corrupt(
                "workspace pair creation journal is missing".into(),
            ));
        }
        transaction.commit()?;
        Ok(())
    }

    pub fn keep_workspace_pair(
        &self,
        content: &WorkspaceHandle,
        git: &WorkspaceHandle,
    ) -> Result<()> {
        let _writer = self.lock_metadata_writer()?;
        let mut connection = self.lock_metadata()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let changed = transaction.execute(
            "UPDATE cow_workspaces SET state = 'kept'
             WHERE id IN (?1, ?2) AND state = 'ready'",
            params![content.id, git.id],
        )?;
        if changed != 2 {
            return Err(Error::InvalidPath(
                "both workspace pair members must be ready before keep".into(),
            ));
        }
        let changed = transaction.execute(
            "UPDATE cow_workspace_pairs SET state = 'kept'
             WHERE content_id = ?1 AND git_id = ?2 AND state = 'ready'",
            params![content.id, git.id],
        )?;
        if changed != 1 {
            return Err(Error::Corrupt("workspace pair journal is not ready".into()));
        }
        transaction.commit()?;
        Ok(())
    }

    pub fn remove_workspace_pair(
        &self,
        content: WorkspaceHandle,
        git: WorkspaceHandle,
    ) -> Result<()> {
        self.remove_workspace_pair_ids(&content.id, &git.id, true)
    }

    pub fn abort_workspace_pair(&self, content_id: &str, git_id: &str) -> Result<()> {
        validate_workspace_id(content_id)?;
        validate_workspace_id(git_id)?;
        self.remove_workspace_pair_ids(content_id, git_id, false)
    }

    fn remove_workspace_pair_ids(
        &self,
        content_id: &str,
        git_id: &str,
        require_ready: bool,
    ) -> Result<()> {
        let _writer = self.lock_metadata_writer()?;
        let mut connection = self.lock_metadata()?;
        let pair_state: Option<String> = connection
            .query_row(
                "SELECT state FROM cow_workspace_pairs WHERE content_id = ?1 AND git_id = ?2",
                params![content_id, git_id],
                |row| row.get(0),
            )
            .optional()?;
        let Some(pair_state) = pair_state else {
            return Err(Error::NotFound("workspace pair journal is missing".into()));
        };
        if require_ready && pair_state != "ready" {
            return Err(Error::InvalidPath(format!(
                "workspace pair is {pair_state}, not ready"
            )));
        }
        let mut chunks = Vec::new();
        for id in [content_id, git_id] {
            let exists: bool = connection.query_row(
                "SELECT EXISTS(SELECT 1 FROM cow_workspaces WHERE id = ?1)",
                params![id],
                |row| row.get(0),
            )?;
            if exists {
                chunks.extend(workspace_chunks(&connection, id)?);
            }
        }
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        transaction.execute(
            "UPDATE cow_workspace_pairs SET state = 'removing'
             WHERE content_id = ?1 AND git_id = ?2",
            params![content_id, git_id],
        )?;
        transaction.execute(
            "DELETE FROM cow_workspaces WHERE id IN (?1, ?2)",
            params![content_id, git_id],
        )?;
        transaction.execute(
            "DELETE FROM cow_workspace_pairs WHERE content_id = ?1 AND git_id = ?2",
            params![content_id, git_id],
        )?;
        transaction.commit()?;
        for chunk in chunks {
            self.chunks.unpin(chunk)?;
        }
        Ok(())
    }

    pub fn status(&self, workspace: &WorkspaceHandle) -> Result<WorkspaceStatus> {
        let connection = self.lock_metadata()?;
        connection
            .query_row(
                "SELECT id, base_commit, baseline_hash, state
                 FROM cow_workspaces WHERE id = ?1",
                params![workspace.id],
                |row| {
                    Ok(WorkspaceStatus {
                        id: row.get(0)?,
                        base_commit: row.get(1)?,
                        baseline_hash: row.get(2)?,
                        state: row.get(3)?,
                    })
                },
            )
            .optional()?
            .ok_or_else(|| Error::NotFound(format!("unknown workspace {}", workspace.id)))
    }

    /// Returns the immutable baseline bound to a workspace.
    pub fn workspace_baseline(&self, workspace: &WorkspaceHandle) -> Result<BaselineSnapshot> {
        let connection = self.lock_metadata()?;
        let bytes = connection
            .query_row(
                "SELECT baseline_json FROM cow_workspaces WHERE id = ?1",
                params![workspace.id],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .optional()?
            .ok_or_else(|| Error::NotFound(format!("unknown workspace {}", workspace.id)))?;
        let baseline: BaselineSnapshot = serde_json::from_slice(&bytes)?;
        crate::snapshot::validate_repository_snapshot_integrity(&baseline)?;
        Ok(baseline)
    }

    pub fn list_workspaces(&self) -> Result<Vec<WorkspaceStatus>> {
        let connection = self.lock_metadata()?;
        let mut statement = connection.prepare(
            "SELECT id, base_commit, baseline_hash, state FROM cow_workspaces ORDER BY id",
        )?;
        let rows = statement.query_map([], |row| {
            Ok(WorkspaceStatus {
                id: row.get(0)?,
                base_commit: row.get(1)?,
                baseline_hash: row.get(2)?,
                state: row.get(3)?,
            })
        })?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    /// Paths whose visible namespace state differs from the immutable dirty
    /// baseline. Provider-created `.git` routing metadata is deliberately not
    /// part of an Agent proposal.
    pub fn changed_paths(&self, workspace: &WorkspaceHandle) -> Result<Vec<String>> {
        let connection = self.lock_metadata()?;
        let mut paths = BTreeSet::new();
        {
            let mut statement = connection
                .prepare("SELECT path FROM cow_entries WHERE workspace_id = ?1 ORDER BY path")?;
            let rows = statement.query_map(params![workspace.id], |row| row.get::<_, String>(0))?;
            for row in rows {
                let path = row?;
                if path != ".git" && !path.starts_with(".git/") {
                    paths.insert(path);
                }
            }
        }
        {
            let mut statement = connection.prepare(
                "SELECT source, destination FROM cow_redirects
                 WHERE workspace_id = ?1 ORDER BY source, destination",
            )?;
            let rows = statement.query_map(params![workspace.id], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;
            for row in rows {
                let (source, destination) = row?;
                for path in [source, destination] {
                    if path != ".git" && !path.starts_with(".git/") {
                        paths.insert(path);
                    }
                }
            }
        }
        Ok(paths.into_iter().collect())
    }

    /// Drops cached repository/dirty layers that are not referenced by an
    /// active workspace or proposal, then compacts the chunk store.
    pub fn gc(&self) -> Result<ChunkGcReport> {
        let _writer = self.lock_metadata_writer()?;
        let mut connection = self.lock_metadata()?;
        repository_layers::remove_unreferenced(&mut connection, &self.chunks)?;
        drop(connection);
        self.chunks.gc()
    }

    pub fn request_repository_tracker(&self, repository: &Path) -> Result<()> {
        let _writer = self.lock_metadata_writer()?;
        let connection = self.lock_metadata()?;
        repository_tracker::request(&connection, repository)
    }

    pub fn pending_repository_trackers(&self) -> Result<Vec<PathBuf>> {
        let connection = self.lock_metadata()?;
        repository_tracker::pending(&connection)
    }

    pub fn activate_repository_tracker(
        &self,
        repository: &Path,
        heartbeat_unix_ms: u64,
    ) -> Result<crate::RepositoryTrackerStatus> {
        let _writer = self.lock_metadata_writer()?;
        let mut connection = self.lock_metadata()?;
        repository_tracker::activate(&mut connection, repository, heartbeat_unix_ms)
    }

    pub fn record_repository_changes(
        &self,
        repository: &Path,
        paths: &[String],
        heartbeat_unix_ms: u64,
    ) -> Result<()> {
        let _writer = self.lock_metadata_writer()?;
        let mut connection = self.lock_metadata()?;
        repository_tracker::record(&mut connection, repository, paths, heartbeat_unix_ms)
    }

    pub fn mark_repository_tracker_gap(
        &self,
        repository: &Path,
        detail: &str,
        heartbeat_unix_ms: u64,
    ) -> Result<()> {
        let _writer = self.lock_metadata_writer()?;
        let connection = self.lock_metadata()?;
        repository_tracker::mark_gap(&connection, repository, detail, heartbeat_unix_ms)
    }

    pub fn repository_tracker_status(
        &self,
        repository: &Path,
    ) -> Result<Option<crate::RepositoryTrackerStatus>> {
        let connection = self.lock_metadata()?;
        repository_tracker::status(&connection, repository)
    }

    pub fn repository_changes_since(
        &self,
        repository: &Path,
        epoch: u64,
        generation: u64,
    ) -> Result<crate::RepositoryChangeBatch> {
        let connection = self.lock_metadata()?;
        repository_tracker::changes_since(&connection, repository, epoch, generation)
    }

    pub fn cached_repository_snapshot(
        &self,
        repository: &Path,
        tracker_epoch: u64,
    ) -> Result<Option<BaselineSnapshot>> {
        let connection = self.lock_metadata()?;
        repository_layers::cached_snapshot(&connection, repository, tracker_epoch)
    }

    pub fn metadata(
        &self,
        workspace: &WorkspaceHandle,
        path: impl AsRef<Path>,
    ) -> Result<Option<NodeMetadata>> {
        let path = normalize_path(path.as_ref(), true)?;
        if path.is_empty() {
            return Ok(Some(NodeMetadata {
                kind: NodeKind::Directory,
                mode: 0o040755,
                size: 0,
                inode: stable_inode(&workspace.id, ""),
                nlink: 2,
                accessed_unix_ns: 0,
                modified_unix_ns: 0,
                changed_unix_ns: 0,
            }));
        }
        let connection = self.lock_metadata()?;
        let pristine_overlay: bool = connection.query_row(
            "SELECT NOT EXISTS(
                 SELECT 1 FROM cow_entries WHERE workspace_id = ?1 LIMIT 1
             ) AND NOT EXISTS(
                 SELECT 1 FROM cow_redirects WHERE workspace_id = ?1 LIMIT 1
             )",
            params![workspace.id],
            |row| row.get(0),
        )?;
        if pristine_overlay {
            if let Some(entry) = repository_layers::lookup(&connection, &workspace.id, &path)? {
                return Ok(Some(NodeMetadata {
                    kind: layer_node_kind(entry.kind)?,
                    mode: entry.mode,
                    size: entry.size,
                    inode: stable_inode(&workspace.id, &path),
                    nlink: 1,
                    accessed_unix_ns: entry.modified_unix_ns,
                    modified_unix_ns: entry.modified_unix_ns,
                    changed_unix_ns: entry.modified_unix_ns,
                }));
            }
            if repository_layers::has_descendant(&connection, &workspace.id, &path)? {
                return Ok(Some(NodeMetadata {
                    kind: NodeKind::Directory,
                    mode: 0o040755,
                    size: 0,
                    inode: stable_inode(&workspace.id, &path),
                    nlink: 2,
                    accessed_unix_ns: 0,
                    modified_unix_ns: 0,
                    changed_unix_ns: 0,
                }));
            }
            return Ok(None);
        }
        if let Some(entry) = overlay_entry(&connection, &workspace.id, &path)? {
            return Ok(Some(entry));
        }
        if ancestor_tombstoned(&connection, &workspace.id, &path)? {
            return Ok(None);
        }
        if has_overlay_descendant(&connection, &workspace.id, &path)? {
            return Ok(Some(NodeMetadata {
                kind: NodeKind::Directory,
                mode: 0o040755,
                size: 0,
                inode: stable_inode(&workspace.id, &path),
                nlink: 2,
                accessed_unix_ns: 0,
                modified_unix_ns: 0,
                changed_unix_ns: 0,
            }));
        }
        let translated = translate_redirect(&connection, &workspace.id, &path)?;
        if let Some(entry) = repository_layers::lookup(&connection, &workspace.id, &translated)? {
            return Ok(Some(NodeMetadata {
                kind: layer_node_kind(entry.kind)?,
                mode: entry.mode,
                size: entry.size,
                inode: stable_inode(&workspace.id, &path),
                nlink: 1,
                accessed_unix_ns: entry.modified_unix_ns,
                modified_unix_ns: entry.modified_unix_ns,
                changed_unix_ns: entry.modified_unix_ns,
            }));
        }
        if repository_layers::has_descendant(&connection, &workspace.id, &translated)? {
            return Ok(Some(NodeMetadata {
                kind: NodeKind::Directory,
                mode: 0o040755,
                size: 0,
                inode: stable_inode(&workspace.id, &path),
                nlink: 2,
                accessed_unix_ns: 0,
                modified_unix_ns: 0,
                changed_unix_ns: 0,
            }));
        }
        Ok(None)
    }

    pub fn read(
        &self,
        workspace: &WorkspaceHandle,
        path: impl AsRef<Path>,
        offset: u64,
        length: usize,
    ) -> Result<Vec<u8>> {
        let path = normalize_path(path.as_ref(), false)?;
        let inode = self.resolve_inode(workspace, &path)?;
        if inode.kind != NodeKind::File && inode.kind != NodeKind::Symlink {
            return Err(Error::IsDirectory(path));
        }
        read_chunks(&self.chunks, &inode.chunks, inode.size, offset, length)
    }

    pub fn open_file(
        &self,
        workspace: &WorkspaceHandle,
        path: impl AsRef<Path>,
    ) -> Result<WorkspaceFileHandle> {
        let path = normalize_path(path.as_ref(), false)?;
        let inode = self.materialize(workspace, &path)?;
        if inode.kind != NodeKind::File {
            return Err(Error::IsDirectory(path));
        }
        Ok(WorkspaceFileHandle {
            workspace_id: workspace.id.clone(),
            backing: WorkspaceFileBacking::Private {
                inode: inode.id as u64,
            },
        })
    }

    /// Open an immutable Base file without copying its inode into the private
    /// namespace. If another handle later promotes the same Base object, reads
    /// resolve through the persistent origin binding and observe that private
    /// inode instead.
    pub fn open_file_read_only(
        &self,
        workspace: &WorkspaceHandle,
        path: impl AsRef<Path>,
    ) -> Result<WorkspaceFileHandle> {
        let path = normalize_path(path.as_ref(), false)?;
        let connection = self.lock_metadata()?;
        if let Some(inode) = load_inode_for_path(&connection, &workspace.id, &path)? {
            if inode.kind != NodeKind::File {
                return Err(Error::IsDirectory(path));
            }
            return Ok(WorkspaceFileHandle {
                workspace_id: workspace.id.clone(),
                backing: WorkspaceFileBacking::Private {
                    inode: inode.id as u64,
                },
            });
        }
        if ancestor_tombstoned(&connection, &workspace.id, &path)? {
            return Err(Error::NotFound(path));
        }
        let origin_path = translate_redirect(&connection, &workspace.id, &path)?;
        let entry = repository_layers::lookup(&connection, &workspace.id, &origin_path)?
            .ok_or_else(|| Error::NotFound(path.clone()))?;
        let kind = layer_node_kind(entry.kind)?;
        if kind != NodeKind::File {
            return Err(Error::IsDirectory(path));
        }
        if let Some(inode) = load_inode_for_origin(&connection, &workspace.id, &origin_path)? {
            return Ok(WorkspaceFileHandle {
                workspace_id: workspace.id.clone(),
                backing: WorkspaceFileBacking::Private {
                    inode: inode.id as u64,
                },
            });
        }
        let metadata = NodeMetadata {
            kind,
            mode: entry.mode,
            size: entry.size,
            inode: stable_inode(&workspace.id, &path),
            nlink: 1,
            accessed_unix_ns: entry.modified_unix_ns,
            modified_unix_ns: entry.modified_unix_ns,
            changed_unix_ns: entry.modified_unix_ns,
        };
        Ok(WorkspaceFileHandle {
            workspace_id: workspace.id.clone(),
            backing: WorkspaceFileBacking::Immutable {
                visible_path: path,
                origin_path,
                metadata,
                chunks: entry.chunks,
            },
        })
    }

    /// Reopen a previously materialized private inode without resolving a path.
    pub fn open_file_inode(
        &self,
        workspace: &WorkspaceHandle,
        inode: u64,
    ) -> Result<WorkspaceFileHandle> {
        let handle = WorkspaceFileHandle {
            workspace_id: workspace.id.clone(),
            backing: WorkspaceFileBacking::Private { inode },
        };
        if self
            .load_open_inode(&handle)?
            .is_none_or(|inode| inode.kind != NodeKind::File)
        {
            return Err(Error::IsDirectory(format!("inode {inode}")));
        }
        Ok(handle)
    }

    pub fn read_open_file(
        &self,
        handle: &WorkspaceFileHandle,
        offset: u64,
        length: usize,
    ) -> Result<Vec<u8>> {
        if let Some(inode) = self.load_open_inode(handle)? {
            return read_chunks(&self.chunks, &inode.chunks, inode.size, offset, length);
        }
        let WorkspaceFileBacking::Immutable {
            metadata, chunks, ..
        } = &handle.backing
        else {
            unreachable!("private handles always resolve an inode")
        };
        read_chunks(&self.chunks, chunks, metadata.size, offset, length)
    }

    pub fn write_open_file(
        &self,
        handle: &WorkspaceFileHandle,
        offset: u64,
        bytes: &[u8],
    ) -> Result<usize> {
        let inode = self.promote_open_file(handle)?;
        self.write_inode(&handle.workspace_id, inode, offset, bytes)
    }

    /// Promotes an immutable open handle into the private namespace without
    /// changing its contents. Mutable adapter operations use this boundary
    /// before rename or unlink so an already-open handle remains valid.
    pub fn materialize_open_file(&self, handle: &WorkspaceFileHandle) -> Result<NodeMetadata> {
        self.promote_open_file(handle)?;
        self.metadata_open_file(handle)
    }

    pub fn truncate_open_file(&self, handle: &WorkspaceFileHandle, size: u64) -> Result<()> {
        let inode = self.promote_open_file(handle)?;
        self.truncate_inode(&handle.workspace_id, inode, size)
    }

    pub fn metadata_open_file(&self, handle: &WorkspaceFileHandle) -> Result<NodeMetadata> {
        if let Some(inode) = self.load_open_inode(handle)? {
            let connection = self.lock_metadata()?;
            return load_metadata_for_inode(&connection, &handle.workspace_id, inode.id)?
                .ok_or_else(|| Error::NotFound(format!("open inode {}", inode.id)));
        }
        let WorkspaceFileBacking::Immutable { metadata, .. } = &handle.backing else {
            unreachable!("private handles always resolve an inode")
        };
        Ok(metadata.clone())
    }

    pub fn set_metadata_open_file(
        &self,
        handle: &WorkspaceFileHandle,
        mode: Option<u32>,
        accessed_unix_ns: Option<i64>,
        modified_unix_ns: Option<i64>,
    ) -> Result<()> {
        let inode = self.promote_open_file(handle)?.id;
        self.set_inode_metadata(
            &handle.workspace_id,
            inode,
            mode,
            accessed_unix_ns,
            modified_unix_ns,
        )
    }

    pub fn write(
        &self,
        workspace: &WorkspaceHandle,
        path: impl AsRef<Path>,
        offset: u64,
        bytes: &[u8],
    ) -> Result<usize> {
        let path = normalize_path(path.as_ref(), false)?;
        let inode = self.materialize(workspace, &path)?;
        if inode.kind != NodeKind::File {
            return Err(Error::IsDirectory(path));
        }
        self.write_inode(&workspace.id, inode, offset, bytes)
    }

    fn write_inode(
        &self,
        workspace_id: &str,
        mut inode: InodeRecord,
        offset: u64,
        bytes: &[u8],
    ) -> Result<usize> {
        if bytes.is_empty() {
            return Ok(0);
        }
        let end = offset
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| Error::InvalidPath("write offset overflow".into()))?;
        let new_size = inode.size.max(end);
        let first_chunk = offset as usize / CHUNK_SIZE;
        let last_chunk = (end.saturating_sub(1) as usize) / CHUNK_SIZE;
        while inode.chunks.len() <= last_chunk {
            let logical_start = inode.chunks.len() as u64 * CHUNK_SIZE as u64;
            let existing_len = inode
                .size
                .saturating_sub(logical_start)
                .min(CHUNK_SIZE as u64);
            let zero = vec![0_u8; existing_len as usize];
            let id = self.chunks.put(&zero)?;
            self.chunks.pin(id)?;
            inode.chunks.push(id);
        }

        let mut replacement = Vec::new();
        for index in first_chunk..=last_chunk {
            let logical_start = index as u64 * CHUNK_SIZE as u64;
            let target_len = new_size
                .saturating_sub(logical_start)
                .min(CHUNK_SIZE as u64) as usize;
            let mut chunk = if index < inode.chunks.len() {
                self.chunks.read(inode.chunks[index])?
            } else {
                Vec::new()
            };
            chunk.resize(target_len, 0);
            let write_start = offset.max(logical_start);
            let write_end = end.min(logical_start + target_len as u64);
            let source_start = (write_start - offset) as usize;
            let destination_start = (write_start - logical_start) as usize;
            let count = (write_end - write_start) as usize;
            chunk[destination_start..destination_start + count]
                .copy_from_slice(&bytes[source_start..source_start + count]);
            let id = self.chunks.put(&chunk)?;
            self.chunks.pin(id)?;
            replacement.push((index, id));
        }
        let old: Vec<ChunkId> = replacement
            .iter()
            .map(|(index, _)| inode.chunks[*index])
            .collect();
        for (index, id) in replacement {
            inode.chunks[index] = id;
        }
        let journal_id = self.begin_namespace_journal(
            workspace_id,
            "write",
            &serde_json::to_vec(&(inode.id, new_size, &inode.chunks))?,
        )?;
        #[cfg(test)]
        crate::test_crash_point("write-after-journal");
        self.update_inode(workspace_id, inode.id, new_size, &inode.chunks)?;
        self.complete_namespace_journal(journal_id)?;
        for id in old {
            self.chunks.unpin(id)?;
        }
        Ok(bytes.len())
    }

    pub fn truncate(
        &self,
        workspace: &WorkspaceHandle,
        path: impl AsRef<Path>,
        new_size: u64,
    ) -> Result<()> {
        let path = normalize_path(path.as_ref(), false)?;
        let inode = self.materialize(workspace, &path)?;
        if inode.kind != NodeKind::File {
            return Err(Error::IsDirectory(path));
        }
        self.truncate_inode(&workspace.id, inode, new_size)
    }

    fn truncate_inode(
        &self,
        workspace_id: &str,
        mut inode: InodeRecord,
        new_size: u64,
    ) -> Result<()> {
        let required = if new_size == 0 {
            0
        } else {
            ((new_size - 1) as usize / CHUNK_SIZE) + 1
        };
        let mut removed = Vec::new();
        if required < inode.chunks.len() {
            removed.extend(inode.chunks.drain(required..));
        }
        while inode.chunks.len() < required {
            let start = inode.chunks.len() as u64 * CHUNK_SIZE as u64;
            let len = new_size.saturating_sub(start).min(CHUNK_SIZE as u64) as usize;
            let id = self.chunks.put(&vec![0_u8; len])?;
            self.chunks.pin(id)?;
            inode.chunks.push(id);
        }
        if required > 0 {
            let last_len = (new_size - (required as u64 - 1) * CHUNK_SIZE as u64) as usize;
            let last_index = required - 1;
            let mut last = self.chunks.read(inode.chunks[last_index])?;
            if last.len() != last_len {
                last.resize(last_len, 0);
                let replacement = self.chunks.put(&last)?;
                self.chunks.pin(replacement)?;
                removed.push(inode.chunks[last_index]);
                inode.chunks[last_index] = replacement;
            }
        }
        self.update_inode(workspace_id, inode.id, new_size, &inode.chunks)?;
        for id in removed {
            self.chunks.unpin(id)?;
        }
        Ok(())
    }

    pub fn create_file(
        &self,
        workspace: &WorkspaceHandle,
        path: impl AsRef<Path>,
        mode: u32,
    ) -> Result<()> {
        self.create_node(workspace, path.as_ref(), NodeKind::File, mode, &[])
    }

    pub fn set_metadata(
        &self,
        workspace: &WorkspaceHandle,
        path: impl AsRef<Path>,
        mode: Option<u32>,
        accessed_unix_ns: Option<i64>,
        modified_unix_ns: Option<i64>,
    ) -> Result<()> {
        let path = normalize_path(path.as_ref(), false)?;
        let inode = self.materialize(workspace, &path)?;
        self.set_inode_metadata(
            &workspace.id,
            inode.id,
            mode,
            accessed_unix_ns,
            modified_unix_ns,
        )
    }

    fn set_inode_metadata(
        &self,
        workspace_id: &str,
        inode: i64,
        mode: Option<u32>,
        accessed_unix_ns: Option<i64>,
        modified_unix_ns: Option<i64>,
    ) -> Result<()> {
        let changed_unix_ns = now_unix_ns();
        let _writer = self.lock_metadata_writer()?;
        let connection = self.lock_metadata()?;
        let changed = connection.execute(
            "UPDATE cow_inodes
             SET mode = COALESCE(?3, mode),
                 accessed_unix_ns = COALESCE(?4, accessed_unix_ns),
                 modified_unix_ns = COALESCE(?5, modified_unix_ns),
                 changed_unix_ns = ?6
             WHERE workspace_id = ?1 AND id = ?2",
            params![
                workspace_id,
                inode,
                mode.map(i64::from),
                accessed_unix_ns,
                modified_unix_ns,
                changed_unix_ns
            ],
        )?;
        if changed != 1 {
            return Err(Error::Corrupt(format!("missing inode {inode}")));
        }
        Ok(())
    }

    pub fn mkdir(
        &self,
        workspace: &WorkspaceHandle,
        path: impl AsRef<Path>,
        mode: u32,
    ) -> Result<()> {
        self.create_node(workspace, path.as_ref(), NodeKind::Directory, mode, &[])
    }

    pub fn symlink(
        &self,
        workspace: &WorkspaceHandle,
        path: impl AsRef<Path>,
        target: &[u8],
    ) -> Result<()> {
        let path = normalize_path(path.as_ref(), false)?;
        crate::path_policy::validate_symlink_target(&path, target)?;
        self.create_node(
            workspace,
            Path::new(&path),
            NodeKind::Symlink,
            0o120000,
            target,
        )
    }

    pub fn read_symlink(
        &self,
        workspace: &WorkspaceHandle,
        path: impl AsRef<Path>,
    ) -> Result<Vec<u8>> {
        let path = normalize_path(path.as_ref(), false)?;
        let metadata = self
            .metadata(workspace, &path)?
            .ok_or_else(|| Error::NotFound(path.clone()))?;
        if metadata.kind != NodeKind::Symlink {
            return Err(Error::InvalidPath(format!("path is not a symlink: {path}")));
        }
        let target = self.read(workspace, &path, 0, usize::MAX)?;
        crate::path_policy::validate_symlink_target(&path, &target)?;
        Ok(target)
    }

    pub fn hard_link(
        &self,
        workspace: &WorkspaceHandle,
        source: impl AsRef<Path>,
        destination: impl AsRef<Path>,
    ) -> Result<()> {
        let source = normalize_path(source.as_ref(), false)?;
        let destination = normalize_path(destination.as_ref(), false)?;
        if self.metadata(workspace, &destination)?.is_some() {
            return Err(Error::AlreadyExists(destination));
        }
        let inode = self.materialize(workspace, &source)?;
        if inode.kind == NodeKind::Directory {
            return Err(Error::InvalidPath(
                "hard links to directories are forbidden".into(),
            ));
        }
        let _writer = self.lock_metadata_writer()?;
        let connection = self.lock_metadata()?;
        connection.execute(
            "INSERT INTO cow_entries(workspace_id, path, inode_id, tombstone)
             VALUES(?1, ?2, ?3, 0)",
            params![workspace.id, destination, inode.id],
        )?;
        Ok(())
    }

    /// Return one currently linked path for a private inode.
    ///
    /// Mount adapters use this only to keep an OS inode bound to a surviving
    /// hard-link alias after its previous canonical name was removed. Base
    /// paths become private before a hard link can be created, so every
    /// multi-name inode is represented in `cow_entries`.
    pub fn path_for_inode(
        &self,
        workspace: &WorkspaceHandle,
        inode: u64,
    ) -> Result<Option<String>> {
        let inode = i64::try_from(inode)
            .map_err(|_| Error::InvalidPath("workspace inode is out of range".into()))?;
        let connection = self.lock_metadata()?;
        connection
            .query_row(
                "SELECT path FROM cow_entries
                 WHERE workspace_id = ?1 AND inode_id = ?2 AND tombstone = 0
                 ORDER BY path LIMIT 1",
                params![workspace.id, inode],
                |row| row.get(0),
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn unlink(&self, workspace: &WorkspaceHandle, path: impl AsRef<Path>) -> Result<()> {
        let path = normalize_path(path.as_ref(), false)?;
        let metadata = self
            .metadata(workspace, &path)?
            .ok_or_else(|| Error::NotFound(path.clone()))?;
        if metadata.kind == NodeKind::Directory && !self.read_dir(workspace, &path)?.is_empty() {
            return Err(Error::DirectoryNotEmpty(path));
        }
        let _inode = self.materialize(workspace, &path)?;
        let _writer = self.lock_metadata_writer()?;
        let mut connection = self.lock_metadata()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        insert_tombstone(&transaction, &workspace.id, &path)?;
        transaction.commit()?;
        Ok(())
    }

    pub fn rename(
        &self,
        workspace: &WorkspaceHandle,
        source: impl AsRef<Path>,
        destination: impl AsRef<Path>,
    ) -> Result<()> {
        let source = normalize_path(source.as_ref(), false)?;
        let destination = normalize_path(destination.as_ref(), false)?;
        if destination == source || destination.starts_with(&(source.clone() + "/")) {
            return Err(Error::InvalidPath(
                "rename would create an ancestry cycle".into(),
            ));
        }
        let metadata = self
            .metadata(workspace, &source)?
            .ok_or_else(|| Error::NotFound(source.clone()))?;
        let destination_metadata = self.metadata(workspace, &destination)?;
        if metadata.kind != NodeKind::Directory {
            if destination_metadata
                .as_ref()
                .is_some_and(|destination| destination.kind == NodeKind::Directory)
            {
                return Err(Error::IsDirectory(destination));
            }
            let inode = self.materialize(workspace, &source)?;
            let _writer = self.lock_metadata_writer()?;
            let mut connection = self.lock_metadata()?;
            let transaction =
                connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
            let replaced: Option<(i64, Vec<ChunkId>)> = transaction
                .query_row(
                    "SELECT i.id, i.chunks_json
                     FROM cow_entries e JOIN cow_inodes i ON i.id = e.inode_id
                     WHERE e.workspace_id = ?1 AND e.path = ?2 AND e.tombstone = 0",
                    params![workspace.id, destination],
                    |row| {
                        Ok((
                            row.get(0)?,
                            serde_json::from_slice(&row.get::<_, Vec<u8>>(1)?).map_err(
                                |error| {
                                    rusqlite::Error::FromSqlConversionFailure(
                                        1,
                                        rusqlite::types::Type::Blob,
                                        Box::new(error),
                                    )
                                },
                            )?,
                        ))
                    },
                )
                .optional()?;
            insert_entry(&transaction, &workspace.id, &destination, inode.id)?;
            insert_tombstone(&transaction, &workspace.id, &source)?;
            #[cfg(test)]
            crate::test_crash_point("rename-before-commit");
            let mut released = Vec::new();
            if let Some((replaced_inode, chunks)) = replaced {
                let remaining: i64 = transaction.query_row(
                    "SELECT COUNT(*) FROM cow_entries
                     WHERE workspace_id = ?1 AND inode_id = ?2 AND tombstone = 0",
                    params![workspace.id, replaced_inode],
                    |row| row.get(0),
                )?;
                if remaining == 0 {
                    transaction.execute(
                        "DELETE FROM cow_inodes WHERE workspace_id = ?1 AND id = ?2",
                        params![workspace.id, replaced_inode],
                    )?;
                    released = chunks;
                }
            }
            transaction.commit()?;
            for id in released {
                self.chunks.unpin(id)?;
            }
            return Ok(());
        }

        if destination_metadata.is_some() {
            return Err(Error::AlreadyExists(destination));
        }

        let _writer = self.lock_metadata_writer()?;
        let mut connection = self.lock_metadata()?;
        let translated = translate_redirect(&connection, &workspace.id, &source)?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        transaction.execute(
            "INSERT INTO cow_redirects(workspace_id, destination, source)
             VALUES(?1, ?2, ?3)",
            params![workspace.id, destination, translated],
        )?;
        let like = escape_like(&(source.clone() + "/")) + "%";
        let descendants: Vec<(String, Option<i64>, bool)> = {
            let mut statement = transaction.prepare(
                "SELECT path, inode_id, tombstone FROM cow_entries
                 WHERE workspace_id = ?1 AND path LIKE ?2 ESCAPE '\\'",
            )?;
            let rows = statement.query_map(params![workspace.id, like], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get::<_, i64>(2)? != 0))
            })?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        };
        for (old, inode, tombstone) in descendants {
            let suffix = &old[source.len()..];
            let new = format!("{destination}{suffix}");
            transaction.execute(
                "DELETE FROM cow_entries WHERE workspace_id = ?1 AND path = ?2",
                params![workspace.id, old],
            )?;
            if tombstone {
                insert_tombstone(&transaction, &workspace.id, &new)?;
            } else if let Some(inode) = inode {
                insert_entry(&transaction, &workspace.id, &new, inode)?;
            }
        }
        insert_tombstone(&transaction, &workspace.id, &source)?;
        transaction.commit()?;
        Ok(())
    }

    pub fn read_dir(
        &self,
        workspace: &WorkspaceHandle,
        path: impl AsRef<Path>,
    ) -> Result<Vec<DirectoryEntry>> {
        let path = normalize_path(path.as_ref(), true)?;
        let metadata = self
            .metadata(workspace, &path)?
            .ok_or_else(|| Error::InvalidPath(format!("directory does not exist: {path}")))?;
        if metadata.kind != NodeKind::Directory {
            return Err(Error::InvalidPath(format!("not a directory: {path}")));
        }
        let connection = self.lock_metadata()?;
        let translated = translate_redirect(&connection, &workspace.id, &path)?;
        let mut entries = BTreeMap::new();
        for (name, entry) in
            repository_layers::list_entries(&connection, &workspace.id, &translated)?
        {
            let child = if path.is_empty() {
                name.clone()
            } else {
                format!("{path}/{name}")
            };
            entries.insert(
                name,
                NodeMetadata {
                    kind: layer_node_kind(entry.kind)?,
                    mode: entry.mode,
                    size: entry.size,
                    inode: stable_inode(&workspace.id, &child),
                    nlink: 1,
                    accessed_unix_ns: entry.modified_unix_ns,
                    modified_unix_ns: entry.modified_unix_ns,
                    changed_unix_ns: entry.modified_unix_ns,
                },
            );
        }
        let prefix = if path.is_empty() {
            String::new()
        } else {
            format!("{path}/")
        };
        let like = escape_like(&prefix) + "%";
        let mut touched_names = BTreeSet::new();
        let overlay_paths: Vec<String> = {
            let mut statement = connection.prepare(
                "SELECT path FROM cow_entries
                 WHERE workspace_id = ?1 AND path LIKE ?2 ESCAPE '\\'",
            )?;
            let rows = statement.query_map(params![workspace.id, like], |row| row.get(0))?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        };
        let redirect_destinations: Vec<String> = {
            let mut statement = connection.prepare(
                "SELECT destination FROM cow_redirects
                 WHERE workspace_id = ?1 AND destination LIKE ?2 ESCAPE '\\'",
            )?;
            let rows = statement.query_map(params![workspace.id, like], |row| row.get(0))?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        };
        drop(connection);
        for candidate in overlay_paths.into_iter().chain(redirect_destinations) {
            let suffix = &candidate[prefix.len()..];
            if let Some(name) = suffix.split('/').next() {
                if !name.is_empty() {
                    touched_names.insert(name.to_string());
                }
            }
        }
        for name in touched_names {
            let child = if path.is_empty() {
                name.clone()
            } else {
                format!("{path}/{name}")
            };
            match self.metadata(workspace, &child)? {
                Some(metadata) => {
                    entries.insert(name, metadata);
                }
                None => {
                    entries.remove(&name);
                }
            }
        }
        Ok(entries
            .into_iter()
            .map(|(name, metadata)| DirectoryEntry { name, metadata })
            .collect())
    }

    pub fn keep(&self, workspace: &WorkspaceHandle) -> Result<()> {
        let _writer = self.lock_metadata_writer()?;
        let connection = self.lock_metadata()?;
        let changed = connection.execute(
            "UPDATE cow_workspaces SET state = 'kept' WHERE id = ?1 AND state = 'ready'",
            params![workspace.id],
        )?;
        if changed != 1 {
            return Err(Error::InvalidPath(format!(
                "workspace {} is not ready",
                workspace.id
            )));
        }
        Ok(())
    }

    pub fn preserve_proposal(
        &self,
        workspace: &WorkspaceHandle,
        ref_name: &str,
        baseline_tree: &str,
        final_tree: &str,
        proposal_commit: &str,
    ) -> Result<ProposalRecord> {
        validate_proposal_ref(ref_name)?;
        validate_oid(baseline_tree)?;
        validate_oid(final_tree)?;
        validate_oid(proposal_commit)?;
        let baseline: BaselineSnapshot = {
            let connection = self.lock_metadata()?;
            connection
                .query_row(
                    "SELECT baseline_json FROM cow_workspaces WHERE id = ?1",
                    params![workspace.id],
                    |row| row.get::<_, Vec<u8>>(0),
                )
                .optional()?
                .ok_or_else(|| Error::InvalidPath(format!("unknown workspace {}", workspace.id)))
                .and_then(|bytes| serde_json::from_slice(&bytes).map_err(Into::into))?
        };
        let repository = baseline.repository.clone();
        let baseline_json = serde_json::to_vec(&baseline)?;
        let insert = (|| -> Result<()> {
            let _writer = self.lock_metadata_writer()?;
            let connection = self.lock_metadata()?;
            connection.execute(
                "INSERT INTO cow_proposals(
                     ref_name, repository, base_commit, baseline_hash,
                     baseline_tree, final_tree, proposal_commit, baseline_json
                 ) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    ref_name,
                    repository
                        .to_str()
                        .ok_or_else(|| Error::UnsupportedRepository(
                            "repository path is not UTF-8".into()
                        ))?,
                    baseline.base_commit,
                    baseline.baseline_hash,
                    baseline_tree,
                    final_tree,
                    proposal_commit,
                    baseline_json
                ],
            )?;
            Ok(())
        })();
        insert?;
        Ok(ProposalRecord {
            ref_name: ref_name.into(),
            repository,
            base_commit: baseline.base_commit.clone(),
            baseline_hash: baseline.baseline_hash.clone(),
            baseline_tree: baseline_tree.into(),
            final_tree: final_tree.into(),
            proposal_commit: proposal_commit.into(),
            baseline,
        })
    }

    pub fn proposal(&self, ref_name: &str) -> Result<ProposalRecord> {
        validate_proposal_ref(ref_name)?;
        let connection = self.lock_metadata()?;
        connection
            .query_row(
                "SELECT repository, base_commit, baseline_hash, baseline_tree,
                        final_tree, proposal_commit, baseline_json
                 FROM cow_proposals WHERE ref_name = ?1",
                params![ref_name],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, String>(5)?,
                        row.get::<_, Vec<u8>>(6)?,
                    ))
                },
            )
            .optional()?
            .ok_or_else(|| Error::InvalidPath(format!("unknown proposal {ref_name}")))
            .and_then(
                |(
                    repository,
                    base_commit,
                    baseline_hash,
                    baseline_tree,
                    final_tree,
                    proposal_commit,
                    bytes,
                )| {
                    let baseline: BaselineSnapshot = serde_json::from_slice(&bytes)?;
                    crate::snapshot::validate_repository_snapshot_integrity(&baseline)?;
                    let repository = PathBuf::from(repository);
                    if repository != baseline.repository
                        || base_commit != baseline.base_commit
                        || baseline_hash != baseline.baseline_hash
                    {
                        return Err(Error::Corrupt(format!(
                            "proposal {ref_name} metadata does not match its pinned baseline"
                        )));
                    }
                    validate_oid(&base_commit)?;
                    validate_oid(&baseline_tree)?;
                    validate_oid(&final_tree)?;
                    validate_oid(&proposal_commit)?;
                    Ok(ProposalRecord {
                        ref_name: ref_name.into(),
                        repository,
                        base_commit,
                        baseline_hash,
                        baseline_tree,
                        final_tree,
                        proposal_commit,
                        baseline,
                    })
                },
            )
    }

    pub fn has_proposal(&self, ref_name: &str) -> Result<bool> {
        validate_proposal_ref(ref_name)?;
        let connection = self.lock_metadata()?;
        connection
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM cow_proposals WHERE ref_name = ?1)",
                params![ref_name],
                |row| row.get::<_, bool>(0),
            )
            .map_err(Into::into)
    }

    pub fn remove_proposal(&self, ref_name: &str) -> Result<()> {
        self.proposal(ref_name)?;
        let _writer = self.lock_metadata_writer()?;
        let connection = self.lock_metadata()?;
        let changed = connection.execute(
            "DELETE FROM cow_proposals WHERE ref_name = ?1",
            params![ref_name],
        )?;
        if changed != 1 {
            return Err(Error::InvalidPath(format!("unknown proposal {ref_name}")));
        }
        drop(connection);
        Ok(())
    }

    pub fn remove_workspace(&self, workspace: WorkspaceHandle) -> Result<()> {
        let _writer = self.lock_metadata_writer()?;
        let mut connection = self.lock_metadata()?;
        let chunks = workspace_chunks(&connection, &workspace.id)?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        transaction.execute(
            "DELETE FROM cow_workspaces WHERE id = ?1",
            params![workspace.id],
        )?;
        transaction.commit()?;
        self.promoted_origins
            .lock()
            .map_err(|_| Error::Corrupt("promotion cache lock poisoned".into()))?
            .retain(|(workspace_id, _), _| workspace_id != &workspace.id);
        for id in chunks {
            self.chunks.unpin(id)?;
        }
        Ok(())
    }

    pub fn recover(&self) -> Result<()> {
        {
            let connection = self.lock_metadata()?;
            let recoverable = {
                let mut statement = connection.prepare(
                    "SELECT content_id, git_id FROM cow_workspace_pairs
                     WHERE state IN ('creating', 'ready', 'removing')",
                )?;
                let rows = statement
                    .query_map([], |row| {
                        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                rows
            };
            drop(connection);
            let abandoned = recoverable
                .into_iter()
                .map(|(content_id, git_id)| {
                    self.acquire_workspace_pair_lease(&content_id, true)
                        .map(|lease| (content_id, git_id, lease))
                })
                .collect::<Result<Vec<_>>>()?;
            let _writer = self.lock_metadata_writer()?;
            let mut connection = self.lock_metadata()?;
            let transaction =
                connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
            for (content_id, git_id, lease) in abandoned {
                let Some(_lease) = lease else {
                    continue;
                };
                transaction.execute(
                    "DELETE FROM cow_workspaces WHERE id IN (?1, ?2)",
                    params![content_id, git_id],
                )?;
                transaction.execute(
                    "DELETE FROM cow_workspace_pairs WHERE content_id = ?1 AND git_id = ?2",
                    params![content_id, git_id],
                )?;
            }
            transaction.commit()?;
        }
        self.chunks.verify()?;
        let mut expected = HashMap::<ChunkId, u64>::new();
        let connection = self.lock_metadata()?;
        {
            let mut statement = connection.prepare("SELECT chunks_json FROM cow_inodes")?;
            let rows = statement.query_map([], |row| row.get::<_, Vec<u8>>(0))?;
            for row in rows {
                for id in serde_json::from_slice::<Vec<ChunkId>>(&row?)? {
                    *expected.entry(id).or_default() += 1;
                }
            }
        }
        for chunks in repository_layers::count_references(&connection)? {
            for id in chunks {
                *expected.entry(id).or_default() += 1;
            }
        }
        drop(connection);
        self.chunks.reconcile_references(&expected)?;
        let _writer = self.lock_metadata_writer()?;
        let connection = self.lock_metadata()?;
        connection.execute("DELETE FROM cow_journal", [])?;
        Ok(())
    }

    fn begin_namespace_journal(
        &self,
        workspace_id: &str,
        operation: &str,
        payload: &[u8],
    ) -> Result<i64> {
        let _writer = self.lock_metadata_writer()?;
        let connection = self.lock_metadata()?;
        connection.execute(
            "INSERT INTO cow_journal(workspace_id, operation, state, payload)
             VALUES(?1, ?2, 'prepared', ?3)",
            params![workspace_id, operation, payload],
        )?;
        Ok(connection.last_insert_rowid())
    }

    fn complete_namespace_journal(&self, journal_id: i64) -> Result<()> {
        let _writer = self.lock_metadata_writer()?;
        let connection = self.lock_metadata()?;
        let changed = connection.execute(
            "UPDATE cow_journal SET state = 'complete' WHERE id = ?1",
            params![journal_id],
        )?;
        if changed != 1 {
            return Err(Error::Corrupt(format!(
                "namespace journal {journal_id} disappeared"
            )));
        }
        connection.execute("DELETE FROM cow_journal WHERE id = ?1", params![journal_id])?;
        Ok(())
    }

    fn create_node(
        &self,
        workspace: &WorkspaceHandle,
        path: &Path,
        kind: NodeKind,
        mode: u32,
        content: &[u8],
    ) -> Result<()> {
        let path = normalize_path(path, false)?;
        if self.metadata(workspace, &path)?.is_some() {
            return Err(Error::AlreadyExists(path));
        }
        ensure_parent_directory(self, workspace, &path)?;
        let chunks = if kind == NodeKind::Directory {
            Vec::new()
        } else {
            let (chunks, _) = self.chunks.put_stream(content)?;
            for id in &chunks {
                self.chunks.pin(*id)?;
            }
            chunks
        };
        let _writer = self.lock_metadata_writer()?;
        let mut connection = self.lock_metadata()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let inode = insert_inode(
            &transaction,
            &workspace.id,
            kind,
            mode,
            content.len() as u64,
            now_unix_ns(),
            &chunks,
        )?;
        insert_entry(&transaction, &workspace.id, &path, inode)?;
        transaction.commit()?;
        Ok(())
    }

    fn resolve_inode(&self, workspace: &WorkspaceHandle, path: &str) -> Result<InodeRecord> {
        let connection = self.lock_metadata()?;
        if let Some(inode) = load_inode_for_path(&connection, &workspace.id, path)? {
            return Ok(inode);
        }
        if ancestor_tombstoned(&connection, &workspace.id, path)? {
            return Err(Error::NotFound(path.into()));
        }
        let translated = translate_redirect(&connection, &workspace.id, path)?;
        let entry = repository_layers::lookup(&connection, &workspace.id, &translated)?
            .ok_or_else(|| Error::NotFound(path.into()))?;
        Ok(InodeRecord {
            id: 0,
            kind: layer_node_kind(entry.kind)?,
            size: entry.size,
            chunks: entry.chunks,
        })
    }

    fn materialize(&self, workspace: &WorkspaceHandle, path: &str) -> Result<InodeRecord> {
        let origin_path = {
            let connection = self.lock_metadata()?;
            if let Some(inode) = load_inode_for_path(&connection, &workspace.id, path)? {
                return Ok(inode);
            }
            if ancestor_tombstoned(&connection, &workspace.id, path)? {
                return Err(Error::NotFound(path.into()));
            }
            translate_redirect(&connection, &workspace.id, path)?
        };
        let source = self.resolve_inode(workspace, path)?;
        let mode = self
            .metadata(workspace, path)?
            .ok_or_else(|| Error::NotFound(path.into()))?
            .mode;
        for id in &source.chunks {
            self.chunks.pin(*id)?;
        }
        let _writer = self.lock_metadata_writer()?;
        let mut connection = self.lock_metadata()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        if let Some(inode) = load_inode_for_path(&transaction, &workspace.id, path)? {
            transaction.commit()?;
            for id in &source.chunks {
                self.chunks.unpin(*id)?;
            }
            return Ok(inode);
        }
        if let Some(inode) = load_inode_for_origin(&transaction, &workspace.id, &origin_path)? {
            insert_entry(&transaction, &workspace.id, path, inode.id)?;
            transaction.commit()?;
            for id in &source.chunks {
                self.chunks.unpin(*id)?;
            }
            self.remember_promotion(&workspace.id, &origin_path, inode.id)?;
            return Ok(inode);
        }
        let inode = insert_inode(
            &transaction,
            &workspace.id,
            source.kind,
            mode,
            source.size,
            0,
            &source.chunks,
        )?;
        transaction.execute(
            "INSERT INTO cow_inode_origins(workspace_id, origin_path, inode_id)
             VALUES(?1, ?2, ?3)",
            params![workspace.id, origin_path, inode],
        )?;
        insert_entry(&transaction, &workspace.id, path, inode)?;
        transaction.commit()?;
        self.remember_promotion(&workspace.id, &origin_path, inode)?;
        Ok(InodeRecord {
            id: inode,
            kind: source.kind,
            size: source.size,
            chunks: source.chunks,
        })
    }

    fn update_inode(
        &self,
        workspace_id: &str,
        inode_id: i64,
        size: u64,
        chunks: &[ChunkId],
    ) -> Result<()> {
        let _writer = self.lock_metadata_writer()?;
        let connection = self.lock_metadata()?;
        let changed = connection.execute(
            "UPDATE cow_inodes
             SET size = ?3, chunks_json = ?4,
                 modified_unix_ns = ?5, changed_unix_ns = ?5
             WHERE workspace_id = ?1 AND id = ?2",
            params![
                workspace_id,
                inode_id,
                size as i64,
                serde_json::to_vec(chunks)?,
                now_unix_ns()
            ],
        )?;
        if changed != 1 {
            return Err(Error::Corrupt(format!("missing inode {inode_id}")));
        }
        Ok(())
    }

    fn load_open_inode(&self, handle: &WorkspaceFileHandle) -> Result<Option<InodeRecord>> {
        match &handle.backing {
            WorkspaceFileBacking::Private { inode } => {
                let inode = i64::try_from(*inode)
                    .map_err(|_| Error::InvalidPath("workspace inode is out of range".into()))?;
                let connection = self.lock_metadata()?;
                load_inode_for_id(&connection, &handle.workspace_id, inode)?.map_or_else(
                    || Err(Error::NotFound(format!("open inode {inode}"))),
                    |inode| Ok(Some(inode)),
                )
            }
            WorkspaceFileBacking::Immutable { origin_path, .. } => {
                let promoted = self
                    .promoted_origins
                    .lock()
                    .map_err(|_| Error::Corrupt("promotion cache lock poisoned".into()))?
                    .get(&(handle.workspace_id.clone(), origin_path.clone()))
                    .copied();
                let Some(inode) = promoted else {
                    return Ok(None);
                };
                let connection = self.lock_metadata()?;
                load_inode_for_id(&connection, &handle.workspace_id, inode as i64)?.map_or_else(
                    || Err(Error::NotFound(format!("promoted inode {inode}"))),
                    |inode| Ok(Some(inode)),
                )
            }
        }
    }

    fn remember_promotion(&self, workspace_id: &str, origin_path: &str, inode: i64) -> Result<()> {
        self.promoted_origins
            .lock()
            .map_err(|_| Error::Corrupt("promotion cache lock poisoned".into()))?
            .insert(
                (workspace_id.to_string(), origin_path.to_string()),
                inode as u64,
            );
        Ok(())
    }

    fn promote_open_file(&self, handle: &WorkspaceFileHandle) -> Result<InodeRecord> {
        if let Some(inode) = self.load_open_inode(handle)? {
            return Ok(inode);
        }
        let WorkspaceFileBacking::Immutable { visible_path, .. } = &handle.backing else {
            unreachable!("private handles always resolve an inode")
        };
        let workspace = self.open_workspace(&handle.workspace_id)?;
        self.materialize(&workspace, visible_path)
    }

    fn lock_metadata(&self) -> Result<PooledConnection<'_>> {
        self.metadata.acquire()
    }

    fn lock_metadata_writer(&self) -> Result<std::sync::MutexGuard<'_, ()>> {
        self.metadata_writer
            .lock()
            .map_err(|_| Error::Corrupt("workspace metadata writer mutex poisoned".into()))
    }
}

#[cfg(unix)]
fn lock_pair_lease(file: &File, nonblocking: bool) -> io::Result<bool> {
    lock_core_lease(file, true, nonblocking)
}

#[cfg(unix)]
fn lock_core_lease(file: &File, exclusive: bool, nonblocking: bool) -> io::Result<bool> {
    use std::os::fd::AsRawFd;
    const LOCK_SH: i32 = 1;
    const LOCK_EX: i32 = 2;
    const LOCK_NB: i32 = 4;
    let mut operation = if exclusive { LOCK_EX } else { LOCK_SH };
    if nonblocking {
        operation |= LOCK_NB;
    }
    // SAFETY: flock operates only on the valid descriptor owned by `file`.
    let result = unsafe { workspace_flock(file.as_raw_fd(), operation) };
    if result == 0 {
        return Ok(true);
    }
    let error = io::Error::last_os_error();
    if nonblocking && error.kind() == io::ErrorKind::WouldBlock {
        Ok(false)
    } else {
        Err(error)
    }
}

#[cfg(unix)]
fn downgrade_core_lease(file: &File) -> io::Result<()> {
    lock_core_lease(file, false, false).map(|_| ())
}

#[cfg(unix)]
fn unlock_pair_lease(file: &File) {
    use std::os::fd::AsRawFd;
    const LOCK_UN: i32 = 8;
    // SAFETY: best-effort unlock of the valid descriptor owned by `file`.
    let _ = unsafe { workspace_flock(file.as_raw_fd(), LOCK_UN) };
}

#[cfg(unix)]
fn unlock_core_lease(file: &File) {
    unlock_pair_lease(file);
}

#[cfg(unix)]
extern "C" {
    #[link_name = "flock"]
    fn workspace_flock(file_descriptor: i32, operation: i32) -> i32;
}

#[cfg(windows)]
fn lock_pair_lease(file: &File, nonblocking: bool) -> io::Result<bool> {
    lock_core_lease(file, true, nonblocking)
}

#[cfg(windows)]
fn lock_core_lease(file: &File, exclusive: bool, nonblocking: bool) -> io::Result<bool> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{
        LockFileEx, LOCKFILE_EXCLUSIVE_LOCK, LOCKFILE_FAIL_IMMEDIATELY,
    };
    use windows_sys::Win32::System::IO::OVERLAPPED;

    let mut flags = if exclusive {
        LOCKFILE_EXCLUSIVE_LOCK
    } else {
        0
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
fn downgrade_core_lease(file: &File) -> io::Result<()> {
    unlock_pair_lease(file);
    lock_core_lease(file, false, false).map(|_| ())
}

#[cfg(windows)]
fn unlock_pair_lease(file: &File) {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::UnlockFileEx;
    use windows_sys::Win32::System::IO::OVERLAPPED;

    let mut overlapped = OVERLAPPED::default();
    let _ = unsafe { UnlockFileEx(file.as_raw_handle(), 0, u32::MAX, u32::MAX, &mut overlapped) };
}

#[cfg(windows)]
fn unlock_core_lease(file: &File) {
    unlock_pair_lease(file);
}

#[cfg(not(any(unix, windows)))]
fn lock_pair_lease(_file: &File, _nonblocking: bool) -> io::Result<bool> {
    Ok(true)
}

#[cfg(not(any(unix, windows)))]
fn lock_core_lease(_file: &File, _exclusive: bool, _nonblocking: bool) -> io::Result<bool> {
    Ok(true)
}

#[cfg(not(any(unix, windows)))]
fn downgrade_core_lease(_file: &File) -> io::Result<()> {
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn unlock_pair_lease(_file: &File) {}

#[cfg(not(any(unix, windows)))]
fn unlock_core_lease(_file: &File) {}

fn write_core_health(file: &File, value: &[u8]) -> io::Result<()> {
    file.set_len(0)?;
    let mut writer = file;
    writer.seek(SeekFrom::Start(0))?;
    writer.write_all(value)?;
    writer.sync_all()
}

fn require_healthy_core_marker(file: &File) -> Result<()> {
    let mut reader = file;
    reader.seek(SeekFrom::Start(0))?;
    let mut value = Vec::new();
    reader.read_to_end(&mut value)?;
    if value == b"healthy-v1\n" {
        Ok(())
    } else {
        Err(Error::AdapterUnavailable(
            "workspace core recovery has not completed successfully".into(),
        ))
    }
}

fn open_metadata_connection(path: &Path) -> Result<Connection> {
    let connection = Connection::open(path)?;
    // SQLite permits one WAL writer. In-process mutations are serialized by
    // WorkspaceCore; this bounded wait covers another Greppy process holding
    // the database writer without turning a genuine stuck writer into a hang.
    connection.busy_timeout(std::time::Duration::from_secs(30))?;
    connection.execute_batch("PRAGMA foreign_keys=ON;")?;
    Ok(connection)
}

fn insert_inode(
    transaction: &Transaction<'_>,
    workspace_id: &str,
    kind: NodeKind,
    mode: u32,
    size: u64,
    modified_unix_ns: i64,
    chunks: &[ChunkId],
) -> Result<i64> {
    transaction.execute(
        "INSERT INTO cow_inodes(
             workspace_id, kind, mode, size,
             accessed_unix_ns, modified_unix_ns, changed_unix_ns, chunks_json
         ) VALUES(?1, ?2, ?3, ?4, ?5, ?5, ?5, ?6)",
        params![
            workspace_id,
            kind_name(kind),
            mode as i64,
            size as i64,
            modified_unix_ns,
            serde_json::to_vec(chunks)?
        ],
    )?;
    Ok(transaction.last_insert_rowid())
}

fn insert_entry(
    transaction: &Transaction<'_>,
    workspace_id: &str,
    path: &str,
    inode: i64,
) -> Result<()> {
    transaction.execute(
        "INSERT INTO cow_entries(workspace_id, path, inode_id, tombstone)
         VALUES(?1, ?2, ?3, 0)
         ON CONFLICT(workspace_id, path) DO UPDATE SET inode_id = excluded.inode_id, tombstone = 0",
        params![workspace_id, path, inode],
    )?;
    Ok(())
}

fn insert_tombstone(transaction: &Transaction<'_>, workspace_id: &str, path: &str) -> Result<()> {
    transaction.execute(
        "INSERT INTO cow_entries(workspace_id, path, inode_id, tombstone)
         VALUES(?1, ?2, NULL, 1)
         ON CONFLICT(workspace_id, path) DO UPDATE SET inode_id = NULL, tombstone = 1",
        params![workspace_id, path],
    )?;
    Ok(())
}

fn overlay_entry(
    connection: &Connection,
    workspace_id: &str,
    path: &str,
) -> Result<Option<NodeMetadata>> {
    connection
        .query_row(
            "SELECT e.tombstone, i.id, i.kind, i.mode, i.size,
                    (SELECT COUNT(*) FROM cow_entries links
                     WHERE links.workspace_id = e.workspace_id
                       AND links.inode_id = i.id AND links.tombstone = 0),
                    i.accessed_unix_ns, i.modified_unix_ns, i.changed_unix_ns
             FROM cow_entries e
             LEFT JOIN cow_inodes i ON i.id = e.inode_id
             WHERE e.workspace_id = ?1 AND e.path = ?2",
            params![workspace_id, path],
            |row| {
                let tombstone: i64 = row.get(0)?;
                if tombstone != 0 {
                    return Ok(None);
                }
                Ok(Some(NodeMetadata {
                    inode: row.get::<_, i64>(1)? as u64,
                    kind: parse_kind(row.get::<_, String>(2)?.as_str())?,
                    mode: row.get::<_, i64>(3)? as u32,
                    size: row.get::<_, i64>(4)? as u64,
                    nlink: row.get::<_, i64>(5)? as u32,
                    accessed_unix_ns: row.get(6)?,
                    modified_unix_ns: row.get(7)?,
                    changed_unix_ns: row.get(8)?,
                }))
            },
        )
        .optional()
        .map(|row| row.flatten())
        .map_err(Into::into)
}

fn now_unix_ns() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| i64::try_from(duration.as_nanos()).ok())
        .unwrap_or(0)
}

fn load_inode_for_path(
    connection: &Connection,
    workspace_id: &str,
    path: &str,
) -> Result<Option<InodeRecord>> {
    connection
        .query_row(
            "SELECT i.id, i.kind, i.size, i.chunks_json
             FROM cow_entries e JOIN cow_inodes i ON i.id = e.inode_id
             WHERE e.workspace_id = ?1 AND e.path = ?2 AND e.tombstone = 0",
            params![workspace_id, path],
            |row| {
                let chunks: Vec<u8> = row.get(3)?;
                Ok((
                    row.get(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                    chunks,
                ))
            },
        )
        .optional()?
        .map(|(id, kind, size, chunks)| {
            Ok(InodeRecord {
                id,
                kind: parse_kind(&kind)?,
                size: size as u64,
                chunks: serde_json::from_slice(&chunks)?,
            })
        })
        .transpose()
}

fn load_inode_for_id(
    connection: &Connection,
    workspace_id: &str,
    inode: i64,
) -> Result<Option<InodeRecord>> {
    connection
        .query_row(
            "SELECT id, kind, size, chunks_json FROM cow_inodes
             WHERE workspace_id = ?1 AND id = ?2",
            params![workspace_id, inode],
            |row| {
                let chunks: Vec<u8> = row.get(3)?;
                Ok((
                    row.get(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                    chunks,
                ))
            },
        )
        .optional()?
        .map(|(id, kind, size, chunks)| {
            Ok(InodeRecord {
                id,
                kind: parse_kind(&kind)?,
                size: size as u64,
                chunks: serde_json::from_slice(&chunks)?,
            })
        })
        .transpose()
}

fn load_inode_for_origin(
    connection: &Connection,
    workspace_id: &str,
    origin_path: &str,
) -> Result<Option<InodeRecord>> {
    connection
        .query_row(
            "SELECT i.id, i.kind, i.size, i.chunks_json
             FROM cow_inode_origins o JOIN cow_inodes i ON i.id = o.inode_id
             WHERE o.workspace_id = ?1 AND o.origin_path = ?2",
            params![workspace_id, origin_path],
            |row| {
                let chunks: Vec<u8> = row.get(3)?;
                Ok((
                    row.get(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                    chunks,
                ))
            },
        )
        .optional()?
        .map(|(id, kind, size, chunks)| {
            Ok(InodeRecord {
                id,
                kind: parse_kind(&kind)?,
                size: size as u64,
                chunks: serde_json::from_slice(&chunks)?,
            })
        })
        .transpose()
}

fn load_metadata_for_inode(
    connection: &Connection,
    workspace_id: &str,
    inode: i64,
) -> Result<Option<NodeMetadata>> {
    connection
        .query_row(
            "SELECT i.id, i.kind, i.mode, i.size,
                    (SELECT COUNT(*) FROM cow_entries links
                     WHERE links.workspace_id = i.workspace_id
                       AND links.inode_id = i.id AND links.tombstone = 0),
                    i.accessed_unix_ns, i.modified_unix_ns, i.changed_unix_ns
             FROM cow_inodes i
             WHERE i.workspace_id = ?1 AND i.id = ?2",
            params![workspace_id, inode],
            |row| {
                Ok(NodeMetadata {
                    inode: row.get::<_, i64>(0)? as u64,
                    kind: parse_kind(row.get::<_, String>(1)?.as_str())?,
                    mode: row.get::<_, i64>(2)? as u32,
                    size: row.get::<_, i64>(3)? as u64,
                    nlink: row.get::<_, i64>(4)? as u32,
                    accessed_unix_ns: row.get(5)?,
                    modified_unix_ns: row.get(6)?,
                    changed_unix_ns: row.get(7)?,
                })
            },
        )
        .optional()
        .map_err(Into::into)
}

fn workspace_chunks(connection: &Connection, workspace_id: &str) -> Result<Vec<ChunkId>> {
    let mut statement = connection
        .prepare("SELECT chunks_json FROM cow_inodes WHERE workspace_id = ?1 ORDER BY id")?;
    let rows = statement.query_map(params![workspace_id], |row| row.get::<_, Vec<u8>>(0))?;
    let mut chunks = Vec::new();
    for row in rows {
        chunks.extend(serde_json::from_slice::<Vec<ChunkId>>(&row?)?);
    }
    Ok(chunks)
}

fn ancestor_tombstoned(connection: &Connection, workspace_id: &str, path: &str) -> Result<bool> {
    let mut current = path;
    loop {
        let tombstone: bool = connection
            .query_row(
                "SELECT tombstone FROM cow_entries WHERE workspace_id = ?1 AND path = ?2",
                params![workspace_id, current],
                |row| Ok(row.get::<_, i64>(0)? != 0),
            )
            .optional()?
            .unwrap_or(false);
        if tombstone {
            return Ok(true);
        }
        let Some(index) = current.rfind('/') else {
            return Ok(false);
        };
        current = &current[..index];
    }
}

fn has_overlay_descendant(connection: &Connection, workspace_id: &str, path: &str) -> Result<bool> {
    let prefix = escape_like(&(path.to_string() + "/")) + "%";
    let count: i64 = connection.query_row(
        "SELECT COUNT(*) FROM cow_entries
         WHERE workspace_id = ?1 AND path LIKE ?2 ESCAPE '\\' AND tombstone = 0",
        params![workspace_id, prefix],
        |row| row.get(0),
    )?;
    Ok(count != 0)
}

fn translate_redirect(connection: &Connection, workspace_id: &str, path: &str) -> Result<String> {
    let mut statement = connection.prepare(
        "SELECT destination, source FROM cow_redirects
         WHERE workspace_id = ?1 ORDER BY length(destination) DESC",
    )?;
    let rows = statement.query_map(params![workspace_id], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;
    for row in rows {
        let (destination, source) = row?;
        if path == destination {
            return Ok(source);
        }
        if let Some(suffix) = path.strip_prefix(&(destination.clone() + "/")) {
            return Ok(if source.is_empty() {
                suffix.to_string()
            } else {
                format!("{source}/{suffix}")
            });
        }
    }
    Ok(path.to_string())
}

fn read_chunks(
    chunks: &ChunkStore,
    ids: &[ChunkId],
    size: u64,
    offset: u64,
    length: usize,
) -> Result<Vec<u8>> {
    if offset >= size || length == 0 {
        return Ok(Vec::new());
    }
    let end = size.min(offset.saturating_add(length as u64));
    let first = offset as usize / CHUNK_SIZE;
    let last = (end.saturating_sub(1) as usize) / CHUNK_SIZE;
    let mut result = Vec::with_capacity((end - offset) as usize);
    for index in first..=last {
        let chunk = chunks.read(*ids.get(index).ok_or_else(|| {
            Error::Corrupt(format!(
                "missing logical chunk {index} for {size}-byte inode"
            ))
        })?)?;
        let logical_start = index as u64 * CHUNK_SIZE as u64;
        let from = offset.saturating_sub(logical_start) as usize;
        let to = (end - logical_start).min(chunk.len() as u64) as usize;
        if from > to || to > chunk.len() {
            return Err(Error::Corrupt(
                "chunk bounds disagree with inode size".into(),
            ));
        }
        result.extend_from_slice(&chunk[from..to]);
    }
    Ok(result)
}

fn ensure_parent_directory(
    core: &WorkspaceCore,
    workspace: &WorkspaceHandle,
    path: &str,
) -> Result<()> {
    let Some((parent, _)) = path.rsplit_once('/') else {
        return Ok(());
    };
    match core.metadata(workspace, parent)? {
        Some(metadata) if metadata.kind == NodeKind::Directory => Ok(()),
        Some(_) => Err(Error::NotDirectory(parent.into())),
        None => Err(Error::NotFound(parent.into())),
    }
}

fn normalize_path(path: &Path, allow_root: bool) -> Result<String> {
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(part) => {
                let part = part
                    .to_str()
                    .ok_or_else(|| Error::InvalidPath("path is not UTF-8".into()))?;
                validate_portable_component(part)?;
                parts.push(part.to_string());
            }
            _ => return Err(Error::InvalidPath(path.display().to_string())),
        }
    }
    let normalized = parts.join("/");
    if normalized.is_empty() && !allow_root {
        return Err(Error::InvalidPath("empty workspace path".into()));
    }
    Ok(normalized)
}

fn validate_workspace_id(id: &str) -> Result<()> {
    if id.is_empty()
        || id.len() > 80
        || !id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(Error::InvalidPath(format!("invalid workspace id {id:?}")));
    }
    Ok(())
}

fn stable_inode(workspace_id: &str, path: &str) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(workspace_id.as_bytes());
    hasher.update(&[0]);
    hasher.update(path.as_bytes());
    let bytes = hasher.finalize();
    u64::from_le_bytes(bytes.as_bytes()[..8].try_into().unwrap()).max(1)
}

fn kind_name(kind: NodeKind) -> &'static str {
    match kind {
        NodeKind::File => "file",
        NodeKind::Directory => "directory",
        NodeKind::Symlink => "symlink",
    }
}

fn layer_node_kind(kind: LayerKind) -> Result<NodeKind> {
    match kind {
        LayerKind::File => Ok(NodeKind::File),
        LayerKind::Directory => Ok(NodeKind::Directory),
        LayerKind::Symlink => Ok(NodeKind::Symlink),
        LayerKind::Tombstone => Err(Error::Corrupt(
            "tombstone escaped repository-layer resolution".into(),
        )),
    }
}

fn parse_kind(kind: &str) -> rusqlite::Result<NodeKind> {
    match kind {
        "file" => Ok(NodeKind::File),
        "directory" => Ok(NodeKind::Directory),
        "symlink" => Ok(NodeKind::Symlink),
        _ => Err(rusqlite::Error::InvalidQuery),
    }
}

fn escape_like(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

fn validate_proposal_ref(value: &str) -> Result<()> {
    let suffix = value.strip_prefix("refs/greppy/agent/").ok_or_else(|| {
        Error::InvalidPath(format!(
            "proposal ref is outside refs/greppy/agent: {value}"
        ))
    })?;
    validate_workspace_id(suffix)
}

fn validate_oid(value: &str) -> Result<()> {
    if !matches!(value.len(), 40 | 64) || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(Error::InvalidPath(format!(
            "invalid Git object id: {value}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const CRASH_CHILD_TEST: &str = "namespace::tests::crash_child_performs_operation";

    #[test]
    fn repository_operation_lease_is_exclusive_and_released_by_drop() {
        let root = tempfile::tempdir().unwrap();
        let repository = tempfile::tempdir().unwrap();
        let first_core = WorkspaceCore::open(root.path()).unwrap();
        let second_core = WorkspaceCore::open(root.path()).unwrap();
        let first = first_core
            .try_repository_operation_lease(repository.path())
            .unwrap()
            .unwrap();
        assert!(second_core
            .try_repository_operation_lease(repository.path())
            .unwrap()
            .is_none());
        drop(first);
        assert!(second_core
            .try_repository_operation_lease(repository.path())
            .unwrap()
            .is_some());
    }

    #[test]
    fn crash_child_performs_operation() {
        let Some(point) = std::env::var_os("GREPPY_WORKSPACE_TEST_CRASH_POINT") else {
            return;
        };
        let root = std::env::var_os("GREPPY_WORKSPACE_TEST_CRASH_ROOT").unwrap();
        let workspace_id = std::env::var("GREPPY_WORKSPACE_TEST_CRASH_WORKSPACE_ID").unwrap();
        let core = WorkspaceCore::open(root).unwrap();
        let workspace = core.open_workspace(&workspace_id).unwrap();
        match point.to_str().unwrap() {
            "write-after-journal" => {
                core.write(&workspace, "committed.bin", 0, b"uncommitted")
                    .unwrap();
            }
            "rename-before-commit" => {
                core.rename(&workspace, "source.txt", "destination.txt")
                    .unwrap();
            }
            "gc-after-segments-synced" => {
                core.gc().unwrap();
            }
            other => panic!("unknown crash point {other}"),
        }
        panic!("crash point {point:?} did not abort the child process");
    }

    fn abort_child_at(root: &Path, workspace: &WorkspaceHandle, point: &str) {
        let status = Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg(CRASH_CHILD_TEST)
            .arg("--nocapture")
            .env("GREPPY_WORKSPACE_TEST_CRASH_POINT", point)
            .env("GREPPY_WORKSPACE_TEST_CRASH_ROOT", root)
            .env("GREPPY_WORKSPACE_TEST_CRASH_WORKSPACE_ID", workspace.id())
            .status()
            .unwrap();
        assert!(!status.success(), "crash child unexpectedly exited cleanly");
    }

    #[test]
    fn open_waits_for_a_short_concurrent_metadata_writer() {
        let root = tempfile::tempdir().unwrap();
        drop(WorkspaceCore::open(root.path()).unwrap());
        let connection = Connection::open(root.path().join("workspace.sqlite3")).unwrap();
        connection.execute_batch("BEGIN IMMEDIATE").unwrap();

        let concurrent_root = root.path().to_path_buf();
        let opening = std::thread::spawn(move || WorkspaceCore::open(concurrent_root));
        std::thread::sleep(std::time::Duration::from_millis(100));
        connection.execute_batch("COMMIT").unwrap();

        opening.join().unwrap().unwrap();
    }

    #[test]
    fn tracker_write_waits_before_reading_across_core_instances() {
        let root = tempfile::tempdir().unwrap();
        let repository = root.path().join("repository");
        let setup = WorkspaceCore::open(root.path()).unwrap();
        setup.request_repository_tracker(&repository).unwrap();
        setup.activate_repository_tracker(&repository, 1).unwrap();
        let writer = WorkspaceCore::open(root.path()).unwrap();

        let connection = Connection::open(root.path().join("workspace.sqlite3")).unwrap();
        connection.execute_batch("BEGIN IMMEDIATE").unwrap();
        let recording = std::thread::spawn(move || {
            writer.record_repository_changes(&repository, &["src/lib.rs".into()], 2)
        });
        std::thread::sleep(std::time::Duration::from_millis(100));
        connection.execute_batch("COMMIT").unwrap();

        recording.join().unwrap().unwrap();
    }

    #[test]
    fn concurrent_open_does_not_reconcile_live_chunk_updates() {
        let root = tempfile::tempdir().unwrap();
        let core = WorkspaceCore::open(root.path()).unwrap();
        let transient = core.chunks().put(b"live cross-database update").unwrap();
        core.chunks().pin(transient).unwrap();

        let concurrent = WorkspaceCore::open(root.path()).unwrap();
        core.chunks().unpin(transient).unwrap();

        drop(concurrent);
        drop(core);
        let recovered = WorkspaceCore::open(root.path()).unwrap();
        recovered.gc().unwrap();
        assert_eq!(recovered.chunks().stats().unwrap().chunk_count, 0);
    }

    fn git(path: &Path, args: &[&str]) {
        let output = Command::new("git")
            .args(args)
            .current_dir(path)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn fixture() -> (
        tempfile::TempDir,
        tempfile::TempDir,
        WorkspaceCore,
        WorkspaceHandle,
    ) {
        let repo = tempfile::tempdir().unwrap();
        git(repo.path(), &["init", "-q"]);
        git(repo.path(), &["config", "user.email", "test@example.test"]);
        git(repo.path(), &["config", "user.name", "Test"]);
        fs::create_dir(repo.path().join("src")).unwrap();
        fs::write(repo.path().join("src/lib.rs"), "pub fn base() {}\n").unwrap();
        fs::write(repo.path().join("README.md"), "base\n").unwrap();
        git(repo.path(), &["add", "."]);
        git(repo.path(), &["commit", "-qm", "base"]);
        fs::write(repo.path().join("README.md"), "dirty\n").unwrap();
        fs::write(repo.path().join("untracked.txt"), "untracked\n").unwrap();

        let storage = tempfile::tempdir().unwrap();
        let core = WorkspaceCore::open(storage.path()).unwrap();
        let baseline = crate::capture_repository(repo.path(), core.chunks()).unwrap();
        let workspace = core.create_workspace("test-workspace", baseline).unwrap();
        (repo, storage, core, workspace)
    }

    #[test]
    fn merges_git_base_and_dirty_overlay() {
        let (_repo, _storage, core, workspace) = fixture();
        assert_eq!(
            core.read(&workspace, "README.md", 0, 100).unwrap(),
            b"dirty\n"
        );
        assert_eq!(
            core.read(&workspace, "src/lib.rs", 0, 100).unwrap(),
            b"pub fn base() {}\n"
        );
        let root: Vec<_> = core
            .read_dir(&workspace, "")
            .unwrap()
            .into_iter()
            .map(|entry| entry.name)
            .collect();
        assert_eq!(root, ["README.md", "src", "untracked.txt"]);
        let source = core.metadata(&workspace, "src").unwrap().unwrap();
        assert_eq!(source.kind, NodeKind::Directory);
        assert_eq!(source.mode & 0o777, 0o755);
    }

    #[test]
    fn provider_hot_path_does_not_access_git_or_the_source_repository() {
        let (repo, _storage, core, workspace) = fixture();
        fs::rename(repo.path().join(".git"), repo.path().join(".git.offline")).unwrap();
        fs::remove_file(repo.path().join("src/lib.rs")).unwrap();
        fs::remove_file(repo.path().join("README.md")).unwrap();

        assert_eq!(
            core.read(&workspace, "src/lib.rs", 0, 100).unwrap(),
            b"pub fn base() {}\n"
        );
        assert_eq!(
            core.read(&workspace, "README.md", 0, 100).unwrap(),
            b"dirty\n"
        );
        assert_eq!(
            core.read_dir(&workspace, "")
                .unwrap()
                .into_iter()
                .map(|entry| entry.name)
                .collect::<Vec<_>>(),
            ["README.md", "src", "untracked.txt"]
        );
        core.write(&workspace, "src/lib.rs", 0, b"X").unwrap();
        core.rename(&workspace, "src/lib.rs", "src/moved.rs")
            .unwrap();
        assert_eq!(core.read(&workspace, "src/moved.rs", 0, 1).unwrap(), b"X");
    }

    #[test]
    fn a_partial_write_replaces_only_one_logical_chunk() {
        let (_repo, _storage, core, workspace) = fixture();
        core.create_file(&workspace, "large.bin", 0o100644).unwrap();
        let input = vec![3_u8; CHUNK_SIZE * 2];
        core.write(&workspace, "large.bin", 0, &input).unwrap();
        let before = core.chunks().stats().unwrap();
        core.write(&workspace, "large.bin", CHUNK_SIZE as u64 + 7, b"x")
            .unwrap();
        let after = core.chunks().stats().unwrap();
        assert_eq!(after.chunk_count, before.chunk_count + 1);
        assert_eq!(
            core.read(&workspace, "large.bin", CHUNK_SIZE as u64, 10)
                .unwrap(),
            [vec![3_u8; 7], b"x".to_vec(), vec![3_u8; 2]].concat()
        );
    }

    #[test]
    fn reopening_recovers_an_incomplete_journal_and_reconciles_cas_refs() {
        let (_repo, storage, core, workspace) = fixture();
        core.create_file(&workspace, "interrupted.bin", 0o100644)
            .unwrap();
        core.write(&workspace, "interrupted.bin", 0, b"committed")
            .unwrap();
        let inode = core.materialize(&workspace, "interrupted.bin").unwrap();
        core.chunks.pin(inode.chunks[0]).unwrap();
        {
            let connection = core.lock_metadata().unwrap();
            connection
                .execute(
                    "INSERT INTO cow_journal(workspace_id, operation, state, payload)
                     VALUES(?1, 'write', 'prepared', X'00')",
                    params![workspace.id],
                )
                .unwrap();
        }
        drop(core);

        let reopened = WorkspaceCore::open(storage.path()).unwrap();
        assert_eq!(
            reopened.read(&workspace, "interrupted.bin", 0, 32).unwrap(),
            b"committed"
        );
        reopened.remove_workspace(workspace).unwrap();
        reopened.gc().unwrap();
        assert_eq!(reopened.chunks().stats().unwrap().referenced_chunks, 0);
    }

    #[test]
    fn namespace_metadata_commit_failure_preserves_content_and_recovers_chunk_refs() {
        let (_repo, storage, core, workspace) = fixture();
        core.create_file(&workspace, "disk-full.txt", 0o100644)
            .unwrap();
        core.write(&workspace, "disk-full.txt", 0, b"committed")
            .unwrap();
        let references_before = core.chunks().stats().unwrap().referenced_chunks;
        {
            let connection = core.lock_metadata().unwrap();
            connection
                .execute_batch(
                    "CREATE TEMP TRIGGER simulate_namespace_metadata_disk_full
                     BEFORE INSERT ON cow_journal
                     BEGIN
                       SELECT RAISE(FAIL, 'database or disk is full');
                     END;",
                )
                .unwrap();
        }

        assert!(core
            .write(&workspace, "disk-full.txt", 0, b"uncommitted")
            .is_err());
        assert_eq!(
            core.read(&workspace, "disk-full.txt", 0, 32).unwrap(),
            b"committed"
        );
        drop(core);

        let recovered = WorkspaceCore::open(storage.path()).unwrap();
        let recovered_workspace = recovered.open_workspace(workspace.id()).unwrap();
        assert_eq!(
            recovered
                .read(&recovered_workspace, "disk-full.txt", 0, 32)
                .unwrap(),
            b"committed"
        );
        assert_eq!(
            recovered.chunks().stats().unwrap().referenced_chunks,
            references_before
        );
    }

    #[test]
    fn fifty_parallel_workspaces_do_not_leak_namespace_or_chunk_state() {
        let repo = tempfile::tempdir().unwrap();
        git(repo.path(), &["init", "-q"]);
        git(repo.path(), &["config", "user.email", "test@example.test"]);
        git(repo.path(), &["config", "user.name", "Test"]);
        fs::write(repo.path().join("shared.txt"), "immutable baseline\n").unwrap();
        git(repo.path(), &["add", "."]);
        git(repo.path(), &["commit", "-qm", "base"]);

        let storage = tempfile::tempdir().unwrap();
        let core = std::sync::Arc::new(WorkspaceCore::open(storage.path()).unwrap());
        let baseline = crate::capture_repository(repo.path(), core.chunks()).unwrap();
        let owner = core
            .create_workspace("parallel-owner", baseline.clone())
            .unwrap();
        let mut workers = Vec::new();
        for number in 0..49 {
            let core = core.clone();
            let baseline = baseline.clone();
            workers.push(std::thread::spawn(move || {
                let id = format!("parallel-{number:02}");
                let workspace = core
                    .create_workspace_from_shared_baseline(&id, &baseline)
                    .unwrap();
                let private = format!("private-{number:02}.txt");
                core.create_file(&workspace, &private, 0o100600).unwrap();
                core.write(&workspace, &private, 0, id.as_bytes()).unwrap();
                assert_eq!(
                    core.read(&workspace, &private, 0, 64).unwrap(),
                    id.as_bytes()
                );
                assert_eq!(
                    core.read(&workspace, "shared.txt", 0, 64).unwrap(),
                    b"immutable baseline\n"
                );
                core.remove_workspace(workspace).unwrap();
            }));
        }
        for worker in workers {
            worker.join().unwrap();
        }
        {
            let connection = core.lock_metadata().unwrap();
            let bases: i64 = connection
                .query_row("SELECT COUNT(*) FROM cow_repository_bases", [], |row| {
                    row.get(0)
                })
                .unwrap();
            let dirty_layers: i64 = connection
                .query_row("SELECT COUNT(*) FROM cow_dirty_layers", [], |row| {
                    row.get(0)
                })
                .unwrap();
            let workspace_layers: i64 = connection
                .query_row("SELECT COUNT(*) FROM cow_workspace_layers", [], |row| {
                    row.get(0)
                })
                .unwrap();
            assert_eq!((bases, dirty_layers, workspace_layers), (1, 1, 1));
        }
        core.remove_workspace(owner).unwrap();
        assert!(core.list_workspaces().unwrap().is_empty());
        core.gc().unwrap();
        assert_eq!(core.chunks().stats().unwrap().referenced_chunks, 0);
    }

    #[test]
    fn hard_links_share_an_inode_and_directory_rename_is_logical() {
        let (_repo, _storage, core, workspace) = fixture();
        core.hard_link(&workspace, "README.md", "README.link")
            .unwrap();
        let first = core.metadata(&workspace, "README.md").unwrap().unwrap();
        let second = core.metadata(&workspace, "README.link").unwrap().unwrap();
        assert_eq!(first.inode, second.inode);
        assert_eq!(first.nlink, 2);
        assert_eq!(second.nlink, 2);
        core.write(&workspace, "README.link", 0, b"X").unwrap();
        assert_eq!(core.read(&workspace, "README.md", 0, 1).unwrap(), b"X");

        core.rename(&workspace, "src", "moved").unwrap();
        assert!(core.metadata(&workspace, "src/lib.rs").unwrap().is_none());
        assert_eq!(
            core.read(&workspace, "moved/lib.rs", 0, 100).unwrap(),
            b"pub fn base() {}\n"
        );
    }

    #[test]
    fn file_rename_atomically_replaces_an_existing_destination() {
        let (_repo, _storage, core, workspace) = fixture();
        core.create_file(&workspace, "destination", 0o100600)
            .unwrap();
        core.write(&workspace, "destination", 0, b"old unique bytes")
            .unwrap();
        core.create_file(&workspace, "temporary", 0o100600).unwrap();
        core.write(&workspace, "temporary", 0, b"replacement")
            .unwrap();
        let before = core.chunks().stats().unwrap().referenced_chunks;
        core.rename(&workspace, "temporary", "destination").unwrap();
        assert!(core.metadata(&workspace, "temporary").unwrap().is_none());
        assert_eq!(
            core.read(&workspace, "destination", 0, 32).unwrap(),
            b"replacement"
        );
        assert_eq!(core.chunks().stats().unwrap().referenced_chunks, before - 1);
    }

    #[test]
    fn open_handle_survives_rename_and_the_last_unlink() {
        let (_repo, _storage, core, workspace) = fixture();
        core.create_file(&workspace, "private.bin", 0o100600)
            .unwrap();
        core.write(&workspace, "private.bin", 0, b"private unique bytes")
            .unwrap();
        let handle = core.open_file(&workspace, "private.bin").unwrap();
        core.rename(&workspace, "private.bin", "renamed.bin")
            .unwrap();
        core.write_open_file(&handle, 8, b"stable").unwrap();
        assert_eq!(
            core.read(&workspace, "renamed.bin", 0, 64).unwrap(),
            b"private stable bytes"
        );
        core.hard_link(&workspace, "renamed.bin", "private.link")
            .unwrap();
        let inode = core
            .metadata(&workspace, "renamed.bin")
            .unwrap()
            .unwrap()
            .inode;
        assert_eq!(
            core.path_for_inode(&workspace, inode).unwrap().as_deref(),
            Some("private.link")
        );
        let before = core.chunks().stats().unwrap().referenced_chunks;
        core.unlink(&workspace, "renamed.bin").unwrap();
        assert_eq!(
            core.path_for_inode(&workspace, inode).unwrap().as_deref(),
            Some("private.link")
        );
        assert_eq!(core.chunks().stats().unwrap().referenced_chunks, before);
        assert_eq!(
            core.read(&workspace, "private.link", 0, 64).unwrap(),
            b"private stable bytes"
        );
        core.unlink(&workspace, "private.link").unwrap();
        assert_eq!(core.path_for_inode(&workspace, inode).unwrap(), None);
        assert_eq!(core.chunks().stats().unwrap().referenced_chunks, before);
        core.write_open_file(&handle, 0, b"PRIVATE").unwrap();
        assert_eq!(
            core.read_open_file(&handle, 0, 64).unwrap(),
            b"PRIVATE stable bytes"
        );
        assert_eq!(core.metadata_open_file(&handle).unwrap().nlink, 0);
    }

    #[test]
    fn read_only_handle_does_not_copy_up_and_observes_later_promotion() {
        let (_repo, _storage, core, workspace) = fixture();
        let before_inodes: i64 = core
            .lock_metadata()
            .unwrap()
            .query_row(
                "SELECT COUNT(*) FROM cow_inodes WHERE workspace_id = ?1",
                params![workspace.id],
                |row| row.get(0),
            )
            .unwrap();
        let before_chunks = core.chunks().stats().unwrap().referenced_chunks;

        let handle = core.open_file_read_only(&workspace, "README.md").unwrap();
        assert_eq!(core.read_open_file(&handle, 0, 64).unwrap(), b"dirty\n");
        assert_eq!(core.metadata_open_file(&handle).unwrap().size, 6);
        let after_open_inodes: i64 = core
            .lock_metadata()
            .unwrap()
            .query_row(
                "SELECT COUNT(*) FROM cow_inodes WHERE workspace_id = ?1",
                params![workspace.id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(after_open_inodes, before_inodes);
        assert_eq!(
            core.chunks().stats().unwrap().referenced_chunks,
            before_chunks
        );

        core.write(&workspace, "README.md", 0, b"DIRTY").unwrap();
        assert_eq!(core.read_open_file(&handle, 0, 64).unwrap(), b"DIRTY\n");
        assert_eq!(core.metadata_open_file(&handle).unwrap().size, 6);
    }

    #[test]
    fn metadata_updates_are_inode_scoped_and_persisted() {
        let (_repo, _storage, core, workspace) = fixture();
        core.hard_link(&workspace, "README.md", "README.link")
            .unwrap();
        core.set_metadata(&workspace, "README.link", Some(0o600), Some(123), Some(456))
            .unwrap();
        for path in ["README.md", "README.link"] {
            let metadata = core.metadata(&workspace, path).unwrap().unwrap();
            assert_eq!(metadata.mode & 0o7777, 0o600);
            assert_eq!(metadata.accessed_unix_ns, 123);
            assert_eq!(metadata.modified_unix_ns, 456);
            assert!(metadata.changed_unix_ns >= 456);
            assert_eq!(metadata.nlink, 2);
        }
    }

    #[test]
    fn private_git_directory_is_a_normal_isolated_namespace() {
        let (_repo, _storage, core, workspace) = fixture();
        core.mkdir(&workspace, ".git", 0o700).unwrap();
        core.create_file(&workspace, ".git/config", 0o600).unwrap();
        core.write(&workspace, ".git/config", 0, b"[core]\n\tbare = false\n")
            .unwrap();
        assert_eq!(
            core.read(&workspace, ".git/config", 0, 128).unwrap(),
            b"[core]\n\tbare = false\n"
        );
        assert!(core
            .read_dir(&workspace, "")
            .unwrap()
            .iter()
            .any(|entry| entry.name == ".git"));
    }

    #[test]
    fn private_git_control_templates_are_shared_chunk_cow_namespaces() {
        let storage = tempfile::tempdir().unwrap();
        let template = tempfile::tempdir().unwrap();
        fs::create_dir_all(template.path().join("refs/heads")).unwrap();
        fs::write(template.path().join("HEAD"), b"ref: refs/heads/base\n").unwrap();
        fs::write(template.path().join("index"), vec![3_u8; CHUNK_SIZE * 2]).unwrap();
        fs::write(
            template.path().join("sharedindex.0123456789"),
            vec![5_u8; CHUNK_SIZE * 2],
        )
        .unwrap();
        let core = WorkspaceCore::open(storage.path()).unwrap();
        let baseline = crate::capture_overlay_directory(
            storage.path().join("git-control-template"),
            template.path(),
            core.chunks(),
        )
        .unwrap();
        let first = core
            .create_overlay_workspace("git-state-first", baseline.clone())
            .unwrap();
        let second = core
            .create_overlay_workspace_from_shared_baseline("git-state-second", &baseline)
            .unwrap();

        assert!(core
            .status(&first)
            .unwrap()
            .base_commit
            .starts_with("virtual-empty:"));
        assert_eq!(
            core.read(&first, "HEAD", 0, 128).unwrap(),
            b"ref: refs/heads/base\n"
        );
        assert_eq!(
            core.metadata(&first, "refs").unwrap().unwrap().kind,
            NodeKind::Directory
        );
        assert_eq!(
            core.metadata(&first, "refs/heads").unwrap().unwrap().kind,
            NodeKind::Directory
        );
        assert_eq!(
            core.read(&second, "index", CHUNK_SIZE as u64, 1).unwrap(),
            [3]
        );
        let before = core.chunks().stats().unwrap().chunk_count;
        core.write(&first, "index", CHUNK_SIZE as u64 + 7, &[9])
            .unwrap();
        assert_eq!(
            core.read(&first, "index", CHUNK_SIZE as u64 + 7, 1)
                .unwrap(),
            [9]
        );
        assert_eq!(
            core.read(&second, "index", CHUNK_SIZE as u64 + 7, 1)
                .unwrap(),
            [3]
        );
        assert_eq!(core.chunks().stats().unwrap().chunk_count, before + 1);
        core.remove_workspace(first).unwrap();
        core.remove_workspace(second).unwrap();
        core.gc().unwrap();
        let after_gc = core
            .create_overlay_workspace_from_shared_baseline("git-state-after-gc", &baseline)
            .unwrap();
        assert_eq!(
            core.read(&after_gc, "sharedindex.0123456789", 0, 1)
                .unwrap(),
            [5]
        );
    }

    #[test]
    fn recovery_rolls_back_an_incomplete_content_git_workspace_pair() {
        let (_repo, storage, core, content) = fixture();
        let template = tempfile::tempdir().unwrap();
        fs::write(template.path().join("HEAD"), b"ref: refs/heads/base\n").unwrap();
        let baseline = crate::capture_overlay_directory(
            storage.path().join("git-control-recovery"),
            template.path(),
            core.chunks(),
        )
        .unwrap();
        core.begin_workspace_pair(content.id(), "git-state-recovery")
            .unwrap();
        let git = core
            .create_overlay_workspace("git-state-recovery", baseline)
            .unwrap();
        assert!(core.status(&content).is_ok());
        assert!(core.status(&git).is_ok());
        drop(core);

        let recovered = WorkspaceCore::open(storage.path()).unwrap();
        assert!(recovered.open_workspace(content.id()).is_err());
        assert!(recovered.open_workspace(git.id()).is_err());
        assert!(recovered.list_workspaces().unwrap().is_empty());
    }

    #[test]
    fn recovery_preserves_a_live_pair_and_removes_it_after_owner_crash() {
        let (_repo, storage, core, content) = fixture();
        let template = tempfile::tempdir().unwrap();
        fs::write(template.path().join("HEAD"), b"ref: refs/heads/base\n").unwrap();
        let baseline = crate::capture_overlay_directory(
            storage.path().join("git-control-live-pair"),
            template.path(),
            core.chunks(),
        )
        .unwrap();
        let lease = core
            .begin_workspace_pair(content.id(), "git-state-live-pair")
            .unwrap();
        let git = core
            .create_overlay_workspace("git-state-live-pair", baseline)
            .unwrap();
        core.complete_workspace_pair(&content, &git).unwrap();

        let concurrent = WorkspaceCore::open(storage.path()).unwrap();
        assert!(concurrent.open_workspace(content.id()).is_ok());
        assert!(concurrent.open_workspace(git.id()).is_ok());
        drop(concurrent);

        // Simulate a process crash: the OS releases the process-lifetime
        // lease without running normal pair cleanup.
        drop(lease);
        drop(core);
        let recovered = WorkspaceCore::open(storage.path()).unwrap();
        assert!(recovered.open_workspace(content.id()).is_err());
        assert!(recovered.open_workspace(git.id()).is_err());
        assert!(recovered.list_workspaces().unwrap().is_empty());
    }

    #[test]
    fn process_crashes_during_write_rename_and_gc_recover_atomically() {
        let (_repo, storage, core, workspace) = fixture();
        core.create_file(&workspace, "committed.bin", 0o100644)
            .unwrap();
        core.write(&workspace, "committed.bin", 0, b"committed")
            .unwrap();
        core.create_file(&workspace, "source.txt", 0o100644)
            .unwrap();
        core.write(&workspace, "source.txt", 0, b"source").unwrap();
        drop(core);

        abort_child_at(storage.path(), &workspace, "write-after-journal");
        let reopened = WorkspaceCore::open(storage.path()).unwrap();
        assert_eq!(
            reopened.read(&workspace, "committed.bin", 0, 32).unwrap(),
            b"committed"
        );
        drop(reopened);

        abort_child_at(storage.path(), &workspace, "rename-before-commit");
        let reopened = WorkspaceCore::open(storage.path()).unwrap();
        assert_eq!(
            reopened.read(&workspace, "source.txt", 0, 32).unwrap(),
            b"source"
        );
        assert!(reopened
            .metadata(&workspace, "destination.txt")
            .unwrap()
            .is_none());
        reopened.chunks().put(b"unreferenced-before-gc").unwrap();
        drop(reopened);

        abort_child_at(storage.path(), &workspace, "gc-after-segments-synced");
        let reopened = WorkspaceCore::open(storage.path()).unwrap();
        assert_eq!(
            reopened.read(&workspace, "committed.bin", 0, 32).unwrap(),
            b"committed"
        );
        assert_eq!(
            reopened.read(&workspace, "source.txt", 0, 32).unwrap(),
            b"source"
        );
        reopened.gc().unwrap();
        let stats = reopened.chunks().stats().unwrap();
        assert_eq!(stats.chunk_count, stats.referenced_chunks);
    }

    #[test]
    fn symlinks_are_confined_to_the_workspace_on_every_platform() {
        let (_repo, _storage, core, workspace) = fixture();
        core.mkdir(&workspace, "nested", 0o755).unwrap();
        core.mkdir(&workspace, "nested/deeper", 0o755).unwrap();
        core.symlink(&workspace, "nested/deeper/internal", b"../target")
            .unwrap();
        assert_eq!(
            core.read_symlink(&workspace, "nested/deeper/internal")
                .unwrap(),
            b"../target"
        );
        assert!(core
            .symlink(&workspace, "nested/escape", b"../../host")
            .is_err());
        assert!(core
            .symlink(&workspace, "nested/windows-escape", b"C:\\Windows")
            .is_err());
    }

    #[test]
    fn workspace_paths_reject_cross_platform_escape_ads_and_reserved_forms() {
        assert_eq!(
            normalize_path(Path::new("src/Ä/Case.rs"), false).unwrap(),
            "src/Ä/Case.rs"
        );
        for path in [
            "../host",
            "..\\host",
            "\\\\server\\share",
            "\\\\?\\C:\\host",
            "file.txt:stream",
            "nested/CON.txt",
            "nested/trailing.",
            "nested/trailing ",
            "nested/a*",
        ] {
            assert!(
                normalize_path(Path::new(path), false).is_err(),
                "accepted non-portable path {path:?}"
            );
        }
    }

    #[test]
    fn cleanup_unpins_workspace_and_baseline_content() {
        let (_repo, _storage, core, workspace) = fixture();
        assert!(core.chunks().stats().unwrap().referenced_chunks > 0);
        core.remove_workspace(workspace).unwrap();
        assert_eq!(core.list_workspaces().unwrap(), []);
        assert!(core.chunks().stats().unwrap().referenced_chunks > 0);
        core.gc().unwrap();
        assert_eq!(core.chunks().stats().unwrap().referenced_chunks, 0);
    }

    #[test]
    fn proposal_pins_baseline_after_workspace_cleanup() {
        let (_repo, _storage, core, workspace) = fixture();
        let record = core
            .preserve_proposal(
                &workspace,
                "refs/greppy/agent/test-workspace",
                &"1".repeat(40),
                &"2".repeat(40),
                &"3".repeat(40),
            )
            .unwrap();
        assert_eq!(
            record.baseline_hash,
            core.proposal(&record.ref_name).unwrap().baseline_hash
        );
        core.remove_workspace(workspace).unwrap();
        assert!(core.chunks().stats().unwrap().referenced_chunks > 0);
        core.remove_proposal(&record.ref_name).unwrap();
        core.gc().unwrap();
        assert_eq!(core.chunks().stats().unwrap().referenced_chunks, 0);
    }

    #[test]
    fn proposal_reader_rejects_tampered_baseline_metadata() {
        let (_repo, _storage, core, workspace) = fixture();
        let record = core
            .preserve_proposal(
                &workspace,
                "refs/greppy/agent/tampered-baseline",
                &"1".repeat(40),
                &"2".repeat(40),
                &"3".repeat(40),
            )
            .unwrap();
        {
            let connection = core.lock_metadata().unwrap();
            connection
                .execute(
                    "UPDATE cow_proposals SET baseline_hash = ?2 WHERE ref_name = ?1",
                    params![record.ref_name, "0".repeat(64)],
                )
                .unwrap();
        }
        assert!(matches!(
            core.proposal(&record.ref_name),
            Err(Error::Corrupt(_))
        ));
    }
}
