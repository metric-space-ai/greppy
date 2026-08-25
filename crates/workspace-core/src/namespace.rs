use crate::{BaselineSnapshot, ChunkId, ChunkStore, EntryKind, Error, Result, CHUNK_SIZE};
use rusqlite::{params, Connection, OptionalExtension, Transaction};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Output};
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

#[derive(Debug)]
struct GitEntry {
    kind: NodeKind,
    mode: u32,
    oid: String,
    size: u64,
}

/// Adapter-neutral namespace and lifecycle engine. The platform mount layers
/// translate their callbacks into these operations and contain no CoW policy.
pub struct WorkspaceCore {
    root: PathBuf,
    chunks: ChunkStore,
    metadata: Mutex<Connection>,
}

impl WorkspaceCore {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root)?;
        let chunks = ChunkStore::open(&root)?;
        let connection = Connection::open(root.join("workspace.sqlite3"))?;
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
        let core = Self {
            root,
            chunks,
            metadata: Mutex::new(connection),
        };
        core.recover()?;
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
            return Err(Error::InvalidPath(format!(
                "unknown or broken workspace {id}"
            )));
        }
        Ok(WorkspaceHandle { id: id.into() })
    }

    pub fn create_workspace(
        &self,
        id: &str,
        baseline: BaselineSnapshot,
    ) -> Result<WorkspaceHandle> {
        validate_workspace_id(id)?;
        let repository = baseline
            .repository
            .to_str()
            .ok_or_else(|| Error::UnsupportedRepository("repository path is not UTF-8".into()))?;
        let baseline_json = serde_json::to_vec(&baseline)?;
        let mut connection = self.lock_metadata()?;
        let transaction = connection.transaction()?;
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
        for entry in &baseline.entries {
            match entry.kind {
                EntryKind::Tombstone => insert_tombstone(&transaction, id, &entry.path)?,
                EntryKind::File | EntryKind::Symlink => {
                    let kind = if entry.kind == EntryKind::File {
                        NodeKind::File
                    } else {
                        NodeKind::Symlink
                    };
                    for chunk in &entry.chunks {
                        self.chunks.pin(*chunk)?;
                    }
                    let inode = insert_inode(
                        &transaction,
                        id,
                        kind,
                        entry.mode,
                        entry.size,
                        entry.modified_unix_ns,
                        &entry.chunks,
                    )?;
                    insert_entry(&transaction, id, &entry.path, inode)?;
                }
            }
        }
        transaction.commit()?;
        Ok(WorkspaceHandle { id: id.into() })
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
            .ok_or_else(|| Error::InvalidPath(format!("unknown workspace {}", workspace.id)))
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
        let (repository, base_commit) = workspace_origin(&connection, &workspace.id)?;
        let translated = translate_redirect(&connection, &workspace.id, &path)?;
        drop(connection);
        Ok(
            git_lookup(Path::new(&repository), &base_commit, &translated)?.map(|entry| {
                NodeMetadata {
                    kind: entry.kind,
                    mode: entry.mode,
                    size: entry.size,
                    inode: stable_inode(&workspace.id, &path),
                    nlink: 1,
                    accessed_unix_ns: 0,
                    modified_unix_ns: 0,
                    changed_unix_ns: 0,
                }
            }),
        )
    }

    pub fn read(
        &self,
        workspace: &WorkspaceHandle,
        path: impl AsRef<Path>,
        offset: u64,
        length: usize,
    ) -> Result<Vec<u8>> {
        let path = normalize_path(path.as_ref(), false)?;
        let inode = self.materialize(workspace, &path)?;
        if inode.kind != NodeKind::File && inode.kind != NodeKind::Symlink {
            return Err(Error::InvalidPath(format!(
                "{path} is not readable file content"
            )));
        }
        read_chunks(&self.chunks, &inode.chunks, inode.size, offset, length)
    }

    pub fn write(
        &self,
        workspace: &WorkspaceHandle,
        path: impl AsRef<Path>,
        offset: u64,
        bytes: &[u8],
    ) -> Result<usize> {
        let path = normalize_path(path.as_ref(), false)?;
        let mut inode = self.materialize(workspace, &path)?;
        if inode.kind != NodeKind::File {
            return Err(Error::InvalidPath(format!("{path} is not a regular file")));
        }
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
            &workspace.id,
            "write",
            &serde_json::to_vec(&(inode.id, new_size, &inode.chunks))?,
        )?;
        self.update_inode(&workspace.id, inode.id, new_size, &inode.chunks)?;
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
        let mut inode = self.materialize(workspace, &path)?;
        if inode.kind != NodeKind::File {
            return Err(Error::InvalidPath(format!("{path} is not a regular file")));
        }
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
        self.update_inode(&workspace.id, inode.id, new_size, &inode.chunks)?;
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
        let changed_unix_ns = now_unix_ns();
        let connection = self.lock_metadata()?;
        let changed = connection.execute(
            "UPDATE cow_inodes
             SET mode = COALESCE(?3, mode),
                 accessed_unix_ns = COALESCE(?4, accessed_unix_ns),
                 modified_unix_ns = COALESCE(?5, modified_unix_ns),
                 changed_unix_ns = ?6
             WHERE workspace_id = ?1 AND id = ?2",
            params![
                workspace.id,
                inode.id,
                mode.map(i64::from),
                accessed_unix_ns,
                modified_unix_ns,
                changed_unix_ns
            ],
        )?;
        if changed != 1 {
            return Err(Error::Corrupt(format!("missing inode {}", inode.id)));
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
            .ok_or_else(|| Error::InvalidPath(format!("path does not exist: {path}")))?;
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
            return Err(Error::InvalidPath(format!(
                "destination already exists: {destination}"
            )));
        }
        let inode = self.materialize(workspace, &source)?;
        if inode.kind == NodeKind::Directory {
            return Err(Error::InvalidPath(
                "hard links to directories are forbidden".into(),
            ));
        }
        let connection = self.lock_metadata()?;
        connection.execute(
            "INSERT INTO cow_entries(workspace_id, path, inode_id, tombstone)
             VALUES(?1, ?2, ?3, 0)",
            params![workspace.id, destination, inode.id],
        )?;
        Ok(())
    }

    pub fn unlink(&self, workspace: &WorkspaceHandle, path: impl AsRef<Path>) -> Result<()> {
        let path = normalize_path(path.as_ref(), false)?;
        let metadata = self
            .metadata(workspace, &path)?
            .ok_or_else(|| Error::InvalidPath(format!("path does not exist: {path}")))?;
        if metadata.kind == NodeKind::Directory && !self.read_dir(workspace, &path)?.is_empty() {
            return Err(Error::InvalidPath(format!(
                "directory is not empty: {path}"
            )));
        }
        let inode = self.materialize(workspace, &path)?;
        let mut connection = self.lock_metadata()?;
        let transaction = connection.transaction()?;
        insert_tombstone(&transaction, &workspace.id, &path)?;
        let remaining: i64 = transaction.query_row(
            "SELECT COUNT(*) FROM cow_entries
             WHERE workspace_id = ?1 AND inode_id = ?2 AND tombstone = 0",
            params![workspace.id, inode.id],
            |row| row.get(0),
        )?;
        if remaining == 0 {
            transaction.execute(
                "DELETE FROM cow_inodes WHERE workspace_id = ?1 AND id = ?2",
                params![workspace.id, inode.id],
            )?;
        }
        transaction.commit()?;
        if remaining == 0 {
            for id in inode.chunks {
                self.chunks.unpin(id)?;
            }
        }
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
            .ok_or_else(|| Error::InvalidPath(format!("source does not exist: {source}")))?;
        let destination_metadata = self.metadata(workspace, &destination)?;
        if metadata.kind != NodeKind::Directory {
            if destination_metadata
                .as_ref()
                .is_some_and(|destination| destination.kind == NodeKind::Directory)
            {
                return Err(Error::InvalidPath(format!(
                    "cannot replace directory with non-directory: {destination}"
                )));
            }
            let inode = self.materialize(workspace, &source)?;
            let mut connection = self.lock_metadata()?;
            let transaction = connection.transaction()?;
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
            return Err(Error::InvalidPath(format!(
                "destination directory already exists: {destination}"
            )));
        }

        let mut connection = self.lock_metadata()?;
        let translated = translate_redirect(&connection, &workspace.id, &source)?;
        let transaction = connection.transaction()?;
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
        let (repository, base_commit) = workspace_origin(&connection, &workspace.id)?;
        let translated = translate_redirect(&connection, &workspace.id, &path)?;
        let mut names = BTreeSet::new();
        for name in git_list_directory(Path::new(&repository), &base_commit, &translated)? {
            names.insert(name);
        }
        let prefix = if path.is_empty() {
            String::new()
        } else {
            format!("{path}/")
        };
        let like = escape_like(&prefix) + "%";
        let overlay_paths: Vec<String> = {
            let mut statement = connection.prepare(
                "SELECT path FROM cow_entries
                 WHERE workspace_id = ?1 AND path LIKE ?2 ESCAPE '\\'",
            )?;
            let rows = statement.query_map(params![workspace.id, like], |row| row.get(0))?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        };
        drop(connection);
        for candidate in overlay_paths {
            let suffix = &candidate[prefix.len()..];
            if let Some(name) = suffix.split('/').next() {
                if !name.is_empty() {
                    names.insert(name.to_string());
                }
            }
        }
        let mut entries = Vec::new();
        for name in names {
            let child = if path.is_empty() {
                name.clone()
            } else {
                format!("{path}/{name}")
            };
            if let Some(metadata) = self.metadata(workspace, &child)? {
                entries.push(DirectoryEntry { name, metadata });
            }
        }
        Ok(entries)
    }

    pub fn keep(&self, workspace: &WorkspaceHandle) -> Result<()> {
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
        let pinned = snapshot_chunks(&baseline);
        for id in &pinned {
            self.chunks.pin(*id)?;
        }
        let repository = baseline.repository.clone();
        let baseline_json = serde_json::to_vec(&baseline)?;
        let insert = (|| -> Result<()> {
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
        if let Err(error) = insert {
            for id in pinned {
                let _ = self.chunks.unpin(id);
            }
            return Err(error);
        }
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
                    Ok(ProposalRecord {
                        ref_name: ref_name.into(),
                        repository: PathBuf::from(repository),
                        base_commit,
                        baseline_hash,
                        baseline_tree,
                        final_tree,
                        proposal_commit,
                        baseline: serde_json::from_slice(&bytes)?,
                    })
                },
            )
    }

    pub fn remove_proposal(&self, ref_name: &str) -> Result<()> {
        let proposal = self.proposal(ref_name)?;
        let connection = self.lock_metadata()?;
        let changed = connection.execute(
            "DELETE FROM cow_proposals WHERE ref_name = ?1",
            params![ref_name],
        )?;
        if changed != 1 {
            return Err(Error::InvalidPath(format!("unknown proposal {ref_name}")));
        }
        drop(connection);
        for id in snapshot_chunks(&proposal.baseline) {
            self.chunks.unpin(id)?;
        }
        Ok(())
    }

    pub fn remove_workspace(&self, workspace: WorkspaceHandle) -> Result<()> {
        let mut connection = self.lock_metadata()?;
        let chunks = workspace_chunks(&connection, &workspace.id)?;
        let baseline: BaselineSnapshot = connection
            .query_row(
                "SELECT baseline_json FROM cow_workspaces WHERE id = ?1",
                params![workspace.id],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .optional()?
            .ok_or_else(|| Error::InvalidPath(format!("unknown workspace {}", workspace.id)))
            .and_then(|bytes| serde_json::from_slice(&bytes).map_err(Into::into))?;
        let transaction = connection.transaction()?;
        transaction.execute(
            "DELETE FROM cow_workspaces WHERE id = ?1",
            params![workspace.id],
        )?;
        transaction.commit()?;
        for id in chunks {
            self.chunks.unpin(id)?;
        }
        for entry in baseline.entries {
            for id in entry.chunks {
                self.chunks.unpin(id)?;
            }
        }
        for id in baseline.index_chunks {
            self.chunks.unpin(id)?;
        }
        Ok(())
    }

    pub fn recover(&self) -> Result<()> {
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
        {
            let mut statement = connection.prepare("SELECT baseline_json FROM cow_workspaces")?;
            let rows = statement.query_map([], |row| row.get::<_, Vec<u8>>(0))?;
            for row in rows {
                let snapshot: BaselineSnapshot = serde_json::from_slice(&row?)?;
                for id in snapshot_chunks(&snapshot) {
                    *expected.entry(id).or_default() += 1;
                }
            }
        }
        {
            let mut statement = connection.prepare("SELECT baseline_json FROM cow_proposals")?;
            let rows = statement.query_map([], |row| row.get::<_, Vec<u8>>(0))?;
            for row in rows {
                let snapshot: BaselineSnapshot = serde_json::from_slice(&row?)?;
                for id in snapshot_chunks(&snapshot) {
                    *expected.entry(id).or_default() += 1;
                }
            }
        }
        drop(connection);
        self.chunks.reconcile_references(&expected)?;
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
        let connection = self.lock_metadata()?;
        connection.execute(
            "INSERT INTO cow_journal(workspace_id, operation, state, payload)
             VALUES(?1, ?2, 'prepared', ?3)",
            params![workspace_id, operation, payload],
        )?;
        Ok(connection.last_insert_rowid())
    }

    fn complete_namespace_journal(&self, journal_id: i64) -> Result<()> {
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
            return Err(Error::InvalidPath(format!("path already exists: {path}")));
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
        let mut connection = self.lock_metadata()?;
        let transaction = connection.transaction()?;
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

    fn materialize(&self, workspace: &WorkspaceHandle, path: &str) -> Result<InodeRecord> {
        {
            let connection = self.lock_metadata()?;
            if let Some(inode) = load_inode_for_path(&connection, &workspace.id, path)? {
                return Ok(inode);
            }
            if ancestor_tombstoned(&connection, &workspace.id, path)? {
                return Err(Error::InvalidPath(format!("path does not exist: {path}")));
            }
        }
        let (repository, base_commit, translated) = {
            let connection = self.lock_metadata()?;
            let (repository, base_commit) = workspace_origin(&connection, &workspace.id)?;
            let translated = translate_redirect(&connection, &workspace.id, path)?;
            (repository, base_commit, translated)
        };
        let git = git_lookup(Path::new(&repository), &base_commit, &translated)?
            .ok_or_else(|| Error::InvalidPath(format!("path does not exist: {path}")))?;
        let chunks = match git.kind {
            NodeKind::Directory => Vec::new(),
            NodeKind::File | NodeKind::Symlink => {
                let bytes = git_blob(Path::new(&repository), &git.oid)?;
                let (chunks, _) = self.chunks.put_stream(bytes.as_slice())?;
                for id in &chunks {
                    self.chunks.pin(*id)?;
                }
                chunks
            }
        };
        let mut connection = self.lock_metadata()?;
        let transaction = connection.transaction()?;
        let inode = insert_inode(
            &transaction,
            &workspace.id,
            git.kind,
            git.mode,
            git.size,
            0,
            &chunks,
        )?;
        insert_entry(&transaction, &workspace.id, path, inode)?;
        transaction.commit()?;
        Ok(InodeRecord {
            id: inode,
            kind: git.kind,
            size: git.size,
            chunks,
        })
    }

    fn update_inode(
        &self,
        workspace_id: &str,
        inode_id: i64,
        size: u64,
        chunks: &[ChunkId],
    ) -> Result<()> {
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

    fn lock_metadata(&self) -> Result<std::sync::MutexGuard<'_, Connection>> {
        self.metadata
            .lock()
            .map_err(|_| Error::Corrupt("workspace metadata mutex poisoned".into()))
    }
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

fn workspace_origin(connection: &Connection, workspace_id: &str) -> Result<(String, String)> {
    connection
        .query_row(
            "SELECT repository, base_commit FROM cow_workspaces WHERE id = ?1 AND state != 'broken'",
            params![workspace_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?
        .ok_or_else(|| Error::InvalidPath(format!("unknown or broken workspace {workspace_id}")))
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

fn git_lookup(repository: &Path, commit: &str, path: &str) -> Result<Option<GitEntry>> {
    if path.is_empty() {
        return Ok(Some(GitEntry {
            kind: NodeKind::Directory,
            mode: 0o040755,
            oid: commit.to_string(),
            size: 0,
        }));
    }
    let spec = format!("{commit}:{path}");
    let type_output = git(repository, &["cat-file", "-t", &spec], true)?;
    if !type_output.status.success() {
        return Ok(None);
    }
    let object_type = String::from_utf8_lossy(&type_output.stdout)
        .trim_end()
        .to_string();
    let oid = git_text(repository, &["rev-parse", &spec])?;
    let mode = git_mode_for_path(repository, commit, path)?;
    let kind = match object_type.as_str() {
        "tree" => NodeKind::Directory,
        "blob" if mode == 0o120000 => NodeKind::Symlink,
        "blob" => NodeKind::File,
        other => {
            return Err(Error::UnsupportedRepository(format!(
                "unsupported Git object type {other} at {path}"
            )))
        }
    };
    let size = if kind == NodeKind::Directory {
        0
    } else {
        git_text(repository, &["cat-file", "-s", &oid])?
            .parse::<u64>()
            .map_err(|_| Error::Git {
                command: format!("git cat-file -s {oid}"),
                detail: "size is not an integer".into(),
            })?
    };
    Ok(Some(GitEntry {
        kind,
        mode,
        oid,
        size,
    }))
}

fn git_mode_for_path(repository: &Path, commit: &str, path: &str) -> Result<u32> {
    let output = git_text(repository, &["ls-tree", commit, "--", path])?;
    let mode = output.split_whitespace().next().ok_or_else(|| Error::Git {
        command: format!("git ls-tree {commit} -- {path}"),
        detail: "entry disappeared".into(),
    })?;
    u32::from_str_radix(mode, 8).map_err(|_| Error::Git {
        command: format!("git ls-tree {commit} -- {path}"),
        detail: format!("invalid mode {mode}"),
    })
}

fn git_blob(repository: &Path, oid: &str) -> Result<Vec<u8>> {
    let output = git(repository, &["cat-file", "blob", oid], false)?;
    if !output.status.success() {
        return Err(git_error(&format!("git cat-file blob {oid}"), &output));
    }
    Ok(output.stdout)
}

fn git_list_directory(repository: &Path, commit: &str, path: &str) -> Result<Vec<String>> {
    let spec = if path.is_empty() {
        commit.to_string()
    } else {
        format!("{commit}:{path}")
    };
    let output = git(repository, &["ls-tree", "-z", &spec], false)?;
    if !output.status.success() {
        return Ok(Vec::new());
    }
    let mut names = Vec::new();
    for record in output.stdout.split(|byte| *byte == 0) {
        if record.is_empty() {
            continue;
        }
        let tab = record
            .iter()
            .position(|byte| *byte == b'\t')
            .ok_or_else(|| Error::Git {
                command: format!("git ls-tree -z {spec}"),
                detail: "entry has no tab separator".into(),
            })?;
        names.push(
            String::from_utf8(record[tab + 1..].to_vec()).map_err(|_| {
                Error::UnsupportedRepository("non-UTF-8 Git path in directory".into())
            })?,
        );
    }
    Ok(names)
}

fn git(repository: &Path, args: &[&str], allow_failure: bool) -> Result<Output> {
    let output = Command::new("git")
        .args(args)
        .current_dir(repository)
        .output()?;
    if !allow_failure && !output.status.success() {
        return Err(git_error(&format!("git {}", args.join(" ")), &output));
    }
    Ok(output)
}

fn git_text(repository: &Path, args: &[&str]) -> Result<String> {
    let output = git(repository, args, false)?;
    String::from_utf8(output.stdout)
        .map(|text| text.trim_end().to_string())
        .map_err(|_| Error::Git {
            command: format!("git {}", args.join(" ")),
            detail: "stdout is not UTF-8".into(),
        })
}

fn git_error(command: &str, output: &Output) -> Error {
    Error::Git {
        command: command.into(),
        detail: String::from_utf8_lossy(&output.stderr).trim().to_string(),
    }
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
        Some(_) => Err(Error::InvalidPath(format!(
            "parent is not a directory: {parent}"
        ))),
        None => Err(Error::InvalidPath(format!(
            "parent does not exist: {parent}"
        ))),
    }
}

fn normalize_path(path: &Path, allow_root: bool) -> Result<String> {
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(part) => parts.push(
                part.to_str()
                    .ok_or_else(|| Error::InvalidPath("path is not UTF-8".into()))?
                    .to_string(),
            ),
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

fn snapshot_chunks(snapshot: &BaselineSnapshot) -> Vec<ChunkId> {
    snapshot
        .entries
        .iter()
        .flat_map(|entry| entry.chunks.iter().copied())
        .chain(snapshot.index_chunks.iter().copied())
        .collect()
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
        assert_eq!(reopened.chunks().stats().unwrap().referenced_chunks, 0);
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
        let mut workers = Vec::new();
        for number in 0..50 {
            let core = core.clone();
            let repository = repo.path().to_path_buf();
            workers.push(std::thread::spawn(move || {
                let id = format!("parallel-{number:02}");
                let baseline = crate::capture_repository(&repository, core.chunks()).unwrap();
                let workspace = core.create_workspace(&id, baseline).unwrap();
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
        assert!(core.list_workspaces().unwrap().is_empty());
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
    fn unlink_releases_private_chunks_only_after_the_last_hard_link() {
        let (_repo, _storage, core, workspace) = fixture();
        core.create_file(&workspace, "private.bin", 0o100600)
            .unwrap();
        core.write(&workspace, "private.bin", 0, b"private unique bytes")
            .unwrap();
        core.hard_link(&workspace, "private.bin", "private.link")
            .unwrap();
        let before = core.chunks().stats().unwrap().referenced_chunks;
        core.unlink(&workspace, "private.bin").unwrap();
        assert_eq!(core.chunks().stats().unwrap().referenced_chunks, before);
        assert_eq!(
            core.read(&workspace, "private.link", 0, 64).unwrap(),
            b"private unique bytes"
        );
        core.unlink(&workspace, "private.link").unwrap();
        assert_eq!(core.chunks().stats().unwrap().referenced_chunks, before - 1);
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
    fn cleanup_unpins_workspace_and_baseline_content() {
        let (_repo, _storage, core, workspace) = fixture();
        assert!(core.chunks().stats().unwrap().referenced_chunks > 0);
        core.remove_workspace(workspace).unwrap();
        assert_eq!(core.list_workspaces().unwrap(), []);
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
        assert_eq!(core.chunks().stats().unwrap().referenced_chunks, 0);
    }
}
