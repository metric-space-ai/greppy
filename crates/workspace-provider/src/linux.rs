use fuser::{
    Config, Errno, FileAttr, FileHandle, FileType, Filesystem, FopenFlags, Generation, INodeNo,
    MountOption, Notifier, OpenFlags, RenameFlags, ReplyAttr, ReplyCreate, ReplyData,
    ReplyDirectory, ReplyEmpty, ReplyEntry, ReplyOpen, ReplyWrite, Request, Session, SessionACL,
    TimeOrNow,
};
use greppy_workspace_core::{
    AdapterKind, Error as CoreError, ErrorKind, NodeKind, NodeMetadata, ProviderCapabilities,
    ProviderManifest, ProviderState, WorkspaceCore, WorkspaceFileHandle, WorkspaceHandle,
    PROVIDER_PROTOCOL_VERSION,
};
use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::io;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const TTL: Duration = Duration::from_millis(250);
const ROOT: u64 = 1;
const WORKSPACES: u64 = 2;
const DOCTOR: u64 = 3;
const MARKER: u64 = 4;
const INVALIDATION_QUEUE_CAPACITY: usize = 1_024;

struct RenameInvalidation {
    parent: INodeNo,
    name: OsString,
    new_parent: INodeNo,
    new_name: OsString,
}

fn enqueue_rename_invalidation(
    sender: &SyncSender<RenameInvalidation>,
    invalidation: RenameInvalidation,
) -> bool {
    sender.try_send(invalidation).is_ok()
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Node {
    Root,
    Workspaces,
    DoctorRoot,
    Marker,
    Doctor(PathBuf),
    WorkspaceRoot(String),
    WorkspacePath { workspace: String, path: String },
}

struct PortableFuse {
    core: WorkspaceCore,
    doctor_root: PathBuf,
    manifest: Arc<RwLock<ProviderManifest>>,
    nodes: Mutex<HashMap<u64, Node>>,
    reverse: Mutex<HashMap<Node, u64>>,
    workspace_inodes: Mutex<HashMap<(String, u64), u64>>,
    open_files: Mutex<HashMap<u64, WorkspaceFileHandle>>,
    next_inode: AtomicU64,
    next_file_handle: AtomicU64,
    uid: u32,
    gid: u32,
    invalidations: SyncSender<RenameInvalidation>,
}

impl PortableFuse {
    fn new(
        core: WorkspaceCore,
        doctor_root: PathBuf,
        manifest: Arc<RwLock<ProviderManifest>>,
    ) -> io::Result<(Self, Receiver<RenameInvalidation>)> {
        fs::create_dir_all(&doctor_root)?;
        let (invalidations, pending_invalidations) =
            mpsc::sync_channel(INVALIDATION_QUEUE_CAPACITY);
        let mut nodes = HashMap::new();
        let mut reverse = HashMap::new();
        for (inode, node) in [
            (ROOT, Node::Root),
            (WORKSPACES, Node::Workspaces),
            (DOCTOR, Node::DoctorRoot),
            (MARKER, Node::Marker),
        ] {
            nodes.insert(inode, node.clone());
            reverse.insert(node, inode);
        }
        Ok((
            Self {
                core,
                doctor_root,
                manifest,
                nodes: Mutex::new(nodes),
                reverse: Mutex::new(reverse),
                workspace_inodes: Mutex::new(HashMap::new()),
                open_files: Mutex::new(HashMap::new()),
                next_inode: AtomicU64::new(16),
                next_file_handle: AtomicU64::new(1),
                uid: unsafe { libc::geteuid() },
                gid: unsafe { libc::getegid() },
                invalidations,
            },
            pending_invalidations,
        ))
    }

    fn node(&self, inode: INodeNo) -> Option<Node> {
        self.nodes.lock().ok()?.get(&inode.0).cloned()
    }

    fn inode(&self, node: Node) -> u64 {
        if let Some(value) = self.reverse.lock().unwrap().get(&node).copied() {
            self.bind_workspace_inode(&node, value);
            return value;
        }
        if let Some((key, value)) = self.workspace_inode_key(&node).and_then(|key| {
            self.workspace_inodes
                .lock()
                .unwrap()
                .get(&key)
                .copied()
                .map(|value| (key, value))
        }) {
            self.workspace_inodes.lock().unwrap().insert(key, value);
            return value;
        }
        let value = self.next_inode.fetch_add(1, Ordering::Relaxed);
        self.reverse.lock().unwrap().insert(node.clone(), value);
        self.bind_workspace_inode(&node, value);
        self.nodes.lock().unwrap().insert(value, node);
        value
    }

    fn workspace_inode_key(&self, node: &Node) -> Option<(String, u64)> {
        let Node::WorkspacePath { workspace, path } = node else {
            return None;
        };
        let handle = self.core.open_workspace(workspace).ok()?;
        let metadata = self.core.metadata(&handle, path).ok()??;
        Some((workspace.clone(), metadata.inode))
    }

    fn bind_workspace_inode(&self, node: &Node, fuse_inode: u64) {
        if let Some(key) = self.workspace_inode_key(node) {
            self.workspace_inodes
                .lock()
                .unwrap()
                .insert(key, fuse_inode);
        }
    }

    fn rebind_after_unlink(
        &self,
        workspace: &str,
        removed_path: &str,
        logical_inode: u64,
        handle: &WorkspaceHandle,
    ) -> Result<(), Errno> {
        let key = (workspace.to_string(), logical_inode);
        let Some(fuse_inode) = self.workspace_inodes.lock().unwrap().get(&key).copied() else {
            return Ok(());
        };
        let canonical_was_removed =
            self.nodes
                .lock()
                .unwrap()
                .get(&fuse_inode)
                .is_some_and(|node| {
                    matches!(
                        node,
                        Node::WorkspacePath {
                            workspace: current_workspace,
                            path
                        } if current_workspace == workspace && path == removed_path
                    )
                });
        if !canonical_was_removed {
            return Ok(());
        }
        let replacement = self
            .core
            .path_for_inode(handle, logical_inode)
            .map_err(core_errno)?;
        let old = Node::WorkspacePath {
            workspace: workspace.into(),
            path: removed_path.into(),
        };
        let mut reverse = self.reverse.lock().unwrap();
        let mut nodes = self.nodes.lock().unwrap();
        reverse.remove(&old);
        match replacement {
            Some(path) => {
                let new = Node::WorkspacePath {
                    workspace: workspace.into(),
                    path,
                };
                reverse.insert(new.clone(), fuse_inode);
                nodes.insert(fuse_inode, new);
            }
            None => {
                self.workspace_inodes.lock().unwrap().remove(&key);
                nodes.remove(&fuse_inode);
            }
        }
        Ok(())
    }

    fn remap_after_rename(&self, source: &Node, destination: &Node) {
        let updates: Vec<(u64, Node, Node)> = self
            .nodes
            .lock()
            .unwrap()
            .iter()
            .filter_map(|(inode, node)| {
                relocated_node(node, source, destination)
                    .map(|relocated| (*inode, node.clone(), relocated))
            })
            .collect();
        let mut reverse = self.reverse.lock().unwrap();
        let mut nodes = self.nodes.lock().unwrap();
        for (inode, old, new) in updates {
            if let Some(replaced_inode) = reverse.get(&new).copied() {
                if replaced_inode != inode {
                    nodes.remove(&replaced_inode);
                }
            }
            reverse.remove(&old);
            reverse.insert(new.clone(), inode);
            nodes.insert(inode, new);
        }
    }

    fn invalidate_rename(
        &self,
        parent: INodeNo,
        name: &OsStr,
        new_parent: INodeNo,
        new_name: &OsStr,
    ) {
        // Reverse notifications wait for a kernel acknowledgement. Running
        // one synchronously in a FUSE callback can consume every session
        // worker under concurrent renames and deadlock the mount. Queue the
        // best-effort cache eviction for the dedicated notifier thread. A
        // full queue safely falls back to the short entry TTL.
        let _ = enqueue_rename_invalidation(
            &self.invalidations,
            RenameInvalidation {
                parent,
                name: name.to_os_string(),
                new_parent,
                new_name: new_name.to_os_string(),
            },
        );
    }

    fn child(&self, parent: &Node, name: &OsStr) -> Result<Node, Errno> {
        match parent {
            Node::Root if name == OsStr::new("workspaces") => Ok(Node::Workspaces),
            Node::Root if name == OsStr::new("doctor") => Ok(Node::DoctorRoot),
            Node::Root if name == OsStr::new(".greppy-provider.json") => Ok(Node::Marker),
            Node::Workspaces => {
                let id = utf8_name(name)?;
                self.core.open_workspace(id).map_err(core_errno)?;
                Ok(Node::WorkspaceRoot(id.into()))
            }
            Node::WorkspaceRoot(workspace) => self.workspace_child(workspace, "", name),
            Node::WorkspacePath { workspace, path } => self.workspace_child(workspace, path, name),
            Node::DoctorRoot => Ok(Node::Doctor(PathBuf::from(name))),
            Node::Doctor(path) if self.doctor_root.join(path).is_dir() => {
                Ok(Node::Doctor(path.join(name)))
            }
            _ => Err(Errno::ENOENT),
        }
    }

    fn workspace_child(&self, workspace: &str, parent: &str, name: &OsStr) -> Result<Node, Errno> {
        let name = utf8_name(name)?;
        let path = join_virtual(parent, name);
        let handle = self.core.open_workspace(workspace).map_err(core_errno)?;
        self.core
            .metadata(&handle, &path)
            .map_err(core_errno)?
            .ok_or(Errno::ENOENT)?;
        Ok(Node::WorkspacePath {
            workspace: workspace.into(),
            path,
        })
    }

    fn attr(&self, inode: u64, node: &Node) -> Result<FileAttr, Errno> {
        let now = SystemTime::now();
        let (kind, mode, size, nlink, uid, gid, atime, mtime, ctime) = match node {
            Node::Root | Node::Workspaces | Node::DoctorRoot | Node::WorkspaceRoot(_) => (
                FileType::Directory,
                0o700,
                0,
                2,
                self.uid,
                self.gid,
                now,
                now,
                now,
            ),
            Node::Marker => {
                let size = self.marker_bytes().map_err(serde_errno)?.len() as u64;
                (
                    FileType::RegularFile,
                    0o400,
                    size,
                    1,
                    self.uid,
                    self.gid,
                    now,
                    now,
                    now,
                )
            }
            Node::Doctor(relative) => {
                let metadata =
                    fs::symlink_metadata(self.doctor_root.join(relative)).map_err(io_errno)?;
                let kind = if metadata.file_type().is_symlink() {
                    FileType::Symlink
                } else if metadata.is_dir() {
                    FileType::Directory
                } else {
                    FileType::RegularFile
                };
                (
                    kind,
                    metadata.permissions().mode() as u16 & 0o7777,
                    metadata.len(),
                    metadata.nlink() as u32,
                    metadata.uid(),
                    metadata.gid(),
                    metadata.accessed().unwrap_or(now),
                    metadata.modified().unwrap_or(now),
                    metadata.created().unwrap_or(now),
                )
            }
            Node::WorkspacePath { workspace, path } => {
                let handle = self.core.open_workspace(workspace).map_err(core_errno)?;
                let metadata = self
                    .core
                    .metadata(&handle, path)
                    .map_err(core_errno)?
                    .ok_or(Errno::ENOENT)?;
                workspace_attr(metadata, self.uid, self.gid)
            }
        };
        Ok(FileAttr {
            ino: INodeNo(inode),
            size,
            blocks: size.div_ceil(512),
            atime,
            mtime,
            ctime,
            crtime: UNIX_EPOCH,
            kind,
            perm: mode,
            nlink,
            uid,
            gid,
            rdev: 0,
            blksize: 4096,
            flags: 0,
        })
    }

    fn marker_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(&*self.manifest.read().unwrap())
    }

    fn workspace_parts(&self, inode: INodeNo) -> Result<(WorkspaceHandle, String), Errno> {
        match self.node(inode).ok_or(Errno::ENOENT)? {
            Node::WorkspacePath { workspace, path } => Ok((
                self.core.open_workspace(&workspace).map_err(core_errno)?,
                path,
            )),
            _ => Err(Errno::EINVAL),
        }
    }

    fn doctor_path(&self, relative: &Path) -> Result<PathBuf, Errno> {
        if relative.components().any(|component| {
            matches!(
                component,
                std::path::Component::ParentDir | std::path::Component::RootDir
            )
        }) {
            return Err(Errno::EPERM);
        }
        Ok(self.doctor_root.join(relative))
    }
}

impl Filesystem for PortableFuse {
    fn lookup(&self, _req: &Request, parent: INodeNo, name: &OsStr, reply: ReplyEntry) {
        let result = self
            .node(parent)
            .ok_or(Errno::ENOENT)
            .and_then(|parent| self.child(&parent, name))
            .and_then(|node| {
                let inode = self.inode(node.clone());
                self.attr(inode, &node).map(|attr| (attr, node))
            });
        match result {
            Ok((attr, _)) => reply.entry(&TTL, &attr, Generation(0)),
            Err(error) => reply.error(error),
        }
    }

    fn getattr(&self, _req: &Request, inode: INodeNo, fh: Option<FileHandle>, reply: ReplyAttr) {
        let result = if let Some(handle) = fh.and_then(|fh| {
            self.open_files
                .lock()
                .ok()
                .and_then(|files| files.get(&fh.0).cloned())
        }) {
            self.core
                .metadata_open_file(&handle)
                .map(|metadata| workspace_file_attr(inode.0, metadata, self.uid, self.gid))
                .map_err(core_errno)
        } else {
            self.node(inode)
                .ok_or(Errno::ENOENT)
                .and_then(|node| self.attr(inode.0, &node))
        };
        match result {
            Ok(attr) => reply.attr(&TTL, &attr),
            Err(error) => reply.error(error),
        }
    }

    fn setattr(
        &self,
        _req: &Request,
        inode: INodeNo,
        mode: Option<u32>,
        uid: Option<u32>,
        gid: Option<u32>,
        size: Option<u64>,
        atime: Option<TimeOrNow>,
        mtime: Option<TimeOrNow>,
        _ctime: Option<SystemTime>,
        fh: Option<FileHandle>,
        _crtime: Option<SystemTime>,
        _chgtime: Option<SystemTime>,
        _bkuptime: Option<SystemTime>,
        _flags: Option<fuser::BsdFileFlags>,
        reply: ReplyAttr,
    ) {
        let result = self.workspace_parts(inode).and_then(|(workspace, path)| {
            if uid.is_some_and(|value| value != self.uid)
                || gid.is_some_and(|value| value != self.gid)
            {
                return Err(Errno::EPERM);
            }
            if let Some(size) = size {
                if let Some(handle) = fh.and_then(|fh| {
                    self.open_files
                        .lock()
                        .ok()
                        .and_then(|files| files.get(&fh.0).cloned())
                }) {
                    self.core
                        .truncate_open_file(&handle, size)
                        .map_err(core_errno)?;
                } else {
                    self.core
                        .truncate(&workspace, &path, size)
                        .map_err(core_errno)?;
                }
            }
            self.core
                .set_metadata(
                    &workspace,
                    &path,
                    mode.map(|value| value & 0o7777),
                    atime.map(time_or_now_ns),
                    mtime.map(time_or_now_ns),
                )
                .map_err(core_errno)?;
            let node = self.node(inode).ok_or(Errno::ENOENT)?;
            self.attr(inode.0, &node)
        });
        match result {
            Ok(attr) => reply.attr(&TTL, &attr),
            Err(error) => reply.error(error),
        }
    }

    fn readlink(&self, _req: &Request, inode: INodeNo, reply: ReplyData) {
        match self.workspace_parts(inode).and_then(|(workspace, path)| {
            self.core.read_symlink(&workspace, path).map_err(core_errno)
        }) {
            Ok(bytes) => reply.data(&bytes),
            Err(error) => reply.error(error),
        }
    }

    fn open(&self, _req: &Request, inode: INodeNo, _flags: OpenFlags, reply: ReplyOpen) {
        let result = match self.node(inode) {
            Some(Node::WorkspacePath { workspace, path }) => self
                .core
                .open_workspace(&workspace)
                .map_err(core_errno)
                .and_then(|workspace| self.core.open_file(&workspace, path).map_err(core_errno))
                .and_then(|handle| {
                    let fh = self.next_file_handle.fetch_add(1, Ordering::Relaxed);
                    self.open_files
                        .lock()
                        .map_err(|_| Errno::EIO)?
                        .insert(fh, handle);
                    Ok(FileHandle(fh))
                }),
            Some(_) => Ok(FileHandle(0)),
            None => Err(Errno::ENOENT),
        };
        match result {
            Ok(fh) => reply.opened(fh, FopenFlags::empty()),
            Err(error) => reply.error(error),
        }
    }

    fn read(
        &self,
        _req: &Request,
        inode: INodeNo,
        fh: FileHandle,
        offset: u64,
        size: u32,
        _flags: OpenFlags,
        _lock_owner: Option<fuser::LockOwner>,
        reply: ReplyData,
    ) {
        let result = if fh.0 != 0 {
            self.open_files
                .lock()
                .map_err(|_| Errno::EIO)
                .and_then(|files| files.get(&fh.0).cloned().ok_or(Errno::EBADF))
                .and_then(|handle| {
                    self.core
                        .read_open_file(&handle, offset, size as usize)
                        .map_err(core_errno)
                })
        } else {
            match self.node(inode) {
                Some(Node::Marker) => self
                    .marker_bytes()
                    .map_err(serde_errno)
                    .map(|bytes| slice(bytes, offset, size)),
                Some(Node::Doctor(relative)) => self
                    .doctor_path(&relative)
                    .and_then(|path| fs::read(path).map_err(io_errno))
                    .map(|bytes| slice(bytes, offset, size)),
                Some(Node::WorkspacePath { workspace, path }) => self
                    .core
                    .open_workspace(&workspace)
                    .map_err(core_errno)
                    .and_then(|handle| {
                        self.core
                            .read(&handle, path, offset, size as usize)
                            .map_err(core_errno)
                    }),
                _ => Err(Errno::EINVAL),
            }
        };
        match result {
            Ok(bytes) => reply.data(&bytes),
            Err(error) => reply.error(error),
        }
    }

    fn write(
        &self,
        _req: &Request,
        inode: INodeNo,
        fh: FileHandle,
        offset: u64,
        data: &[u8],
        _write_flags: fuser::WriteFlags,
        _flags: OpenFlags,
        _lock_owner: Option<fuser::LockOwner>,
        reply: ReplyWrite,
    ) {
        let result = if fh.0 != 0 {
            self.open_files
                .lock()
                .map_err(|_| Errno::EIO)
                .and_then(|files| files.get(&fh.0).cloned().ok_or(Errno::EBADF))
                .and_then(|handle| {
                    self.core
                        .write_open_file(&handle, offset, data)
                        .map_err(core_errno)
                })
        } else {
            match self.node(inode) {
                Some(Node::Doctor(relative)) => self.doctor_path(&relative).and_then(|path| {
                    use std::os::unix::fs::FileExt;
                    let file = fs::OpenOptions::new()
                        .write(true)
                        .open(path)
                        .map_err(io_errno)?;
                    file.write_all_at(data, offset).map_err(io_errno)?;
                    Ok(data.len())
                }),
                Some(Node::WorkspacePath { workspace, path }) => self
                    .core
                    .open_workspace(&workspace)
                    .map_err(core_errno)
                    .and_then(|handle| {
                        self.core
                            .write(&handle, path, offset, data)
                            .map_err(core_errno)
                    }),
                _ => Err(Errno::EROFS),
            }
        };
        match result {
            Ok(written) => reply.written(written as u32),
            Err(error) => reply.error(error),
        }
    }

    fn release(
        &self,
        _req: &Request,
        _inode: INodeNo,
        fh: FileHandle,
        _flags: OpenFlags,
        _lock_owner: Option<fuser::LockOwner>,
        _flush: bool,
        reply: ReplyEmpty,
    ) {
        if fh.0 != 0 {
            if let Ok(mut files) = self.open_files.lock() {
                files.remove(&fh.0);
            }
        }
        reply.ok();
    }

    fn flush(
        &self,
        _req: &Request,
        _inode: INodeNo,
        _fh: FileHandle,
        _lock_owner: fuser::LockOwner,
        reply: ReplyEmpty,
    ) {
        reply.ok();
    }

    fn fsync(
        &self,
        _req: &Request,
        _inode: INodeNo,
        _fh: FileHandle,
        _datasync: bool,
        reply: ReplyEmpty,
    ) {
        reply.ok();
    }

    fn readdir(
        &self,
        _req: &Request,
        inode: INodeNo,
        _fh: FileHandle,
        offset: u64,
        mut reply: ReplyDirectory,
    ) {
        let node = match self.node(inode) {
            Some(node) => node,
            None => return reply.error(Errno::ENOENT),
        };
        let mut children: Vec<(u64, FileType, Vec<u8>)> = vec![
            (inode.0, FileType::Directory, b".".to_vec()),
            (
                self.parent_inode(&node),
                FileType::Directory,
                b"..".to_vec(),
            ),
        ];
        let result: Result<(), Errno> = (|| {
            match &node {
                Node::Root => {
                    children.extend([
                        (WORKSPACES, FileType::Directory, b"workspaces".to_vec()),
                        (DOCTOR, FileType::Directory, b"doctor".to_vec()),
                        (
                            MARKER,
                            FileType::RegularFile,
                            b".greppy-provider.json".to_vec(),
                        ),
                    ]);
                }
                Node::Workspaces => {
                    for workspace in self.core.list_workspaces().map_err(core_errno)? {
                        let child = Node::WorkspaceRoot(workspace.id.clone());
                        children.push((
                            self.inode(child),
                            FileType::Directory,
                            workspace.id.into(),
                        ));
                    }
                }
                Node::WorkspaceRoot(workspace) => {
                    self.append_workspace_dir(workspace, "", &mut children)?;
                }
                Node::WorkspacePath { workspace, path } => {
                    self.append_workspace_dir(workspace, path, &mut children)?;
                }
                Node::DoctorRoot | Node::Doctor(_) => {
                    let relative = match &node {
                        Node::Doctor(path) => path.as_path(),
                        _ => Path::new(""),
                    };
                    for entry in fs::read_dir(self.doctor_path(relative)?).map_err(io_errno)? {
                        let entry = entry.map_err(io_errno)?;
                        let name = entry.file_name();
                        let child = Node::Doctor(relative.join(&name));
                        let kind = if entry.file_type().map_err(io_errno)?.is_dir() {
                            FileType::Directory
                        } else {
                            FileType::RegularFile
                        };
                        children.push((self.inode(child), kind, name.as_bytes().to_vec()));
                    }
                }
                Node::Marker => return Err(Errno::ENOTDIR),
            }
            Ok(())
        })();
        if let Err(error) = result {
            return reply.error(error);
        }
        for (index, (child_inode, kind, name)) in
            children.into_iter().enumerate().skip(offset as usize)
        {
            if reply.add(
                INodeNo(child_inode),
                (index + 1) as u64,
                kind,
                OsStr::from_bytes(&name),
            ) {
                break;
            }
        }
        reply.ok();
    }

    fn create(
        &self,
        _req: &Request,
        parent: INodeNo,
        name: &OsStr,
        mode: u32,
        _umask: u32,
        _flags: i32,
        reply: ReplyCreate,
    ) {
        match self.create_node(parent, name, mode, false) {
            Ok((inode, attr)) => {
                let file_handle = match self.node(INodeNo(inode)) {
                    Some(Node::WorkspacePath { workspace, path }) => self
                        .core
                        .open_workspace(&workspace)
                        .map_err(core_errno)
                        .and_then(|workspace| {
                            self.core.open_file(&workspace, path).map_err(core_errno)
                        })
                        .and_then(|handle| {
                            let fh = self.next_file_handle.fetch_add(1, Ordering::Relaxed);
                            self.open_files
                                .lock()
                                .map_err(|_| Errno::EIO)?
                                .insert(fh, handle);
                            Ok(FileHandle(fh))
                        }),
                    Some(_) => Ok(FileHandle(0)),
                    None => Err(Errno::ENOENT),
                };
                match file_handle {
                    Ok(fh) => reply.created(&TTL, &attr, Generation(0), fh, FopenFlags::empty()),
                    Err(error) => reply.error(error),
                }
            }
            Err(error) => reply.error(error),
        }
    }

    fn mkdir(
        &self,
        _req: &Request,
        parent: INodeNo,
        name: &OsStr,
        mode: u32,
        _umask: u32,
        reply: ReplyEntry,
    ) {
        match self.create_node(parent, name, mode, true) {
            Ok((_inode, attr)) => reply.entry(&TTL, &attr, Generation(0)),
            Err(error) => reply.error(error),
        }
    }

    fn unlink(&self, _req: &Request, parent: INodeNo, name: &OsStr, reply: ReplyEmpty) {
        self.remove_node(parent, name, reply);
    }

    fn rmdir(&self, _req: &Request, parent: INodeNo, name: &OsStr, reply: ReplyEmpty) {
        self.remove_node(parent, name, reply);
    }

    fn rename(
        &self,
        _req: &Request,
        parent: INodeNo,
        name: &OsStr,
        new_parent: INodeNo,
        new_name: &OsStr,
        flags: RenameFlags,
        reply: ReplyEmpty,
    ) {
        if flags.intersects(RenameFlags::RENAME_EXCHANGE | RenameFlags::RENAME_WHITEOUT) {
            return reply.error(Errno::EINVAL);
        }
        let no_replace = flags.contains(RenameFlags::RENAME_NOREPLACE);
        let result = self
            .node(parent)
            .zip(self.node(new_parent))
            .ok_or(Errno::ENOENT)
            .and_then(|(source_parent, destination_parent)| {
                if matches!(&source_parent, Node::DoctorRoot | Node::Doctor(_))
                    && matches!(&destination_parent, Node::DoctorRoot | Node::Doctor(_))
                {
                    let source_node = Node::Doctor(doctor_destination(&source_parent, name)?);
                    let destination_node =
                        Node::Doctor(doctor_destination(&destination_parent, new_name)?);
                    let Node::Doctor(source_relative) = &source_node else {
                        unreachable!()
                    };
                    let Node::Doctor(destination_relative) = &destination_node else {
                        unreachable!()
                    };
                    let source = self.doctor_path(source_relative)?;
                    let destination = self.doctor_path(destination_relative)?;
                    if no_replace && destination.exists() {
                        return Err(Errno::EEXIST);
                    }
                    fs::rename(source, destination).map_err(io_errno)?;
                    self.remap_after_rename(&source_node, &destination_node);
                    return Ok(());
                }
                let (workspace, source) = workspace_destination(&source_parent, name)?;
                let (destination_workspace, destination) =
                    workspace_destination(&destination_parent, new_name)?;
                if workspace != destination_workspace {
                    return Err(Errno::EXDEV);
                }
                let handle = self.core.open_workspace(&workspace).map_err(core_errno)?;
                if no_replace
                    && self
                        .core
                        .metadata(&handle, &destination)
                        .map_err(core_errno)?
                        .is_some()
                {
                    return Err(Errno::EEXIST);
                }
                self.core
                    .rename(&handle, &source, &destination)
                    .map_err(core_errno)?;
                self.remap_after_rename(
                    &Node::WorkspacePath {
                        workspace: workspace.clone(),
                        path: source,
                    },
                    &Node::WorkspacePath {
                        workspace,
                        path: destination,
                    },
                );
                Ok(())
            });
        match result {
            Ok(()) => {
                // The kernel cannot process a notification that invalidates the
                // dentry for the rename request it is still waiting to finish.
                // Complete the request first, then evict both cached names.
                reply.ok();
                self.invalidate_rename(parent, name, new_parent, new_name);
            }
            Err(error) => reply.error(error),
        }
    }

    fn symlink(
        &self,
        _req: &Request,
        parent: INodeNo,
        name: &OsStr,
        target: &Path,
        reply: ReplyEntry,
    ) {
        let result = self.node(parent).ok_or(Errno::ENOENT).and_then(|parent| {
            let (workspace, path) = workspace_destination(&parent, name)?;
            let handle = self.core.open_workspace(&workspace).map_err(core_errno)?;
            self.core
                .symlink(&handle, &path, target.as_os_str().as_bytes())
                .map_err(core_errno)?;
            let node = Node::WorkspacePath { workspace, path };
            let inode = self.inode(node.clone());
            self.attr(inode, &node)
        });
        match result {
            Ok(attr) => reply.entry(&TTL, &attr, Generation(0)),
            Err(error) => reply.error(error),
        }
    }

    fn link(
        &self,
        _req: &Request,
        inode: INodeNo,
        new_parent: INodeNo,
        new_name: &OsStr,
        reply: ReplyEntry,
    ) {
        let result = self
            .workspace_parts(inode)
            .and_then(|(source_handle, source)| {
                let parent = self.node(new_parent).ok_or(Errno::ENOENT)?;
                let (workspace, destination) = workspace_destination(&parent, new_name)?;
                if source_handle.id() != workspace {
                    return Err(Errno::EXDEV);
                }
                self.core
                    .hard_link(&source_handle, &source, &destination)
                    .map_err(core_errno)?;
                let source_node = Node::WorkspacePath {
                    workspace: workspace.clone(),
                    path: source,
                };
                self.bind_workspace_inode(&source_node, inode.0);
                let node = Node::WorkspacePath {
                    workspace,
                    path: destination,
                };
                let child_inode = self.inode(node.clone());
                self.attr(child_inode, &node)
            });
        match result {
            Ok(attr) => reply.entry(&TTL, &attr, Generation(0)),
            Err(error) => reply.error(error),
        }
    }
}

fn relocated_node(node: &Node, source: &Node, destination: &Node) -> Option<Node> {
    match (node, source, destination) {
        (Node::Doctor(path), Node::Doctor(from), Node::Doctor(to)) => path
            .strip_prefix(from)
            .ok()
            .map(|suffix| Node::Doctor(to.join(suffix))),
        (
            Node::WorkspacePath { workspace, path },
            Node::WorkspacePath {
                workspace: source_workspace,
                path: from,
            },
            Node::WorkspacePath {
                workspace: destination_workspace,
                path: to,
            },
        ) if workspace == source_workspace && workspace == destination_workspace => Path::new(path)
            .strip_prefix(from)
            .ok()
            .map(|suffix| Node::WorkspacePath {
                workspace: workspace.clone(),
                path: Path::new(to).join(suffix).to_string_lossy().into_owned(),
            }),
        _ => None,
    }
}

impl PortableFuse {
    fn parent_inode(&self, node: &Node) -> u64 {
        match node {
            Node::Root => ROOT,
            Node::Workspaces | Node::DoctorRoot | Node::Marker => ROOT,
            Node::WorkspaceRoot(_) => WORKSPACES,
            Node::WorkspacePath { workspace, path } => match path.rsplit_once('/') {
                Some((parent, _)) => self.inode(Node::WorkspacePath {
                    workspace: workspace.clone(),
                    path: parent.into(),
                }),
                None => self.inode(Node::WorkspaceRoot(workspace.clone())),
            },
            Node::Doctor(path) => match path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
            {
                Some(parent) => self.inode(Node::Doctor(parent.into())),
                None => DOCTOR,
            },
        }
    }

    fn append_workspace_dir(
        &self,
        workspace: &str,
        path: &str,
        children: &mut Vec<(u64, FileType, Vec<u8>)>,
    ) -> Result<(), Errno> {
        let handle = self.core.open_workspace(workspace).map_err(core_errno)?;
        for entry in self.core.read_dir(&handle, path).map_err(core_errno)? {
            let child_path = join_virtual(path, &entry.name);
            let node = Node::WorkspacePath {
                workspace: workspace.into(),
                path: child_path,
            };
            children.push((
                self.inode(node),
                file_type(entry.metadata.kind),
                entry.name.into_bytes(),
            ));
        }
        Ok(())
    }

    fn create_node(
        &self,
        parent: INodeNo,
        name: &OsStr,
        mode: u32,
        directory: bool,
    ) -> Result<(u64, FileAttr), Errno> {
        match self.node(parent).ok_or(Errno::ENOENT)? {
            Node::DoctorRoot | Node::Doctor(_) => {
                let parent_node = self.node(parent).unwrap();
                let relative = match parent_node {
                    Node::Doctor(path) => path,
                    _ => PathBuf::new(),
                };
                let child = relative.join(name);
                let path = self.doctor_path(&child)?;
                if directory {
                    fs::create_dir(&path).map_err(io_errno)?;
                } else {
                    fs::OpenOptions::new()
                        .create_new(true)
                        .write(true)
                        .open(&path)
                        .map_err(io_errno)?;
                }
                fs::set_permissions(&path, fs::Permissions::from_mode(mode & 0o7777))
                    .map_err(io_errno)?;
                let node = Node::Doctor(child);
                let inode = self.inode(node.clone());
                Ok((inode, self.attr(inode, &node)?))
            }
            parent_node => {
                let (workspace, path) = workspace_destination(&parent_node, name)?;
                let handle = self.core.open_workspace(&workspace).map_err(core_errno)?;
                if directory {
                    self.core.mkdir(&handle, &path, mode).map_err(core_errno)?;
                } else {
                    self.core
                        .create_file(&handle, &path, mode)
                        .map_err(core_errno)?;
                }
                let node = Node::WorkspacePath { workspace, path };
                let inode = self.inode(node.clone());
                Ok((inode, self.attr(inode, &node)?))
            }
        }
    }

    fn remove_node(&self, parent: INodeNo, name: &OsStr, reply: ReplyEmpty) {
        let result =
            self.node(parent)
                .ok_or(Errno::ENOENT)
                .and_then(|parent_node| match parent_node {
                    Node::DoctorRoot | Node::Doctor(_) => {
                        let relative = match parent_node {
                            Node::Doctor(path) => path,
                            _ => PathBuf::new(),
                        };
                        let path = self.doctor_path(&relative.join(name))?;
                        if path.is_dir() {
                            fs::remove_dir(path).map_err(io_errno)
                        } else {
                            fs::remove_file(path).map_err(io_errno)
                        }
                    }
                    parent_node => {
                        let (workspace, path) = workspace_destination(&parent_node, name)?;
                        let handle = self.core.open_workspace(&workspace).map_err(core_errno)?;
                        let metadata = self
                            .core
                            .metadata(&handle, &path)
                            .map_err(core_errno)?
                            .ok_or(Errno::ENOENT)?;
                        self.core.unlink(&handle, &path).map_err(core_errno)?;
                        self.rebind_after_unlink(&workspace, &path, metadata.inode, &handle)
                    }
                });
        match result {
            Ok(()) => reply.ok(),
            Err(error) => reply.error(error),
        }
    }
}

fn dispatch_invalidations(notifier: Notifier, invalidations: Receiver<RenameInvalidation>) {
    while let Ok(invalidation) = invalidations.recv() {
        let _ = notifier.inval_entry(invalidation.parent, &invalidation.name);
        let _ = notifier.inval_entry(invalidation.new_parent, &invalidation.new_name);
    }
}

pub fn serve(data_root: PathBuf, mount_root: PathBuf) -> io::Result<()> {
    fs::create_dir_all(&data_root)?;
    fs::create_dir_all(&mount_root)?;
    let manifest = Arc::new(RwLock::new(ProviderManifest {
        protocol_version: PROVIDER_PROTOCOL_VERSION,
        adapter_version: env!("CARGO_PKG_VERSION").into(),
        adapter_kind: AdapterKind::Fuse3,
        state: ProviderState::Starting,
        instance_id: format!(
            "linux-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ),
        data_root: data_root.clone(),
        mount_root: mount_root.clone(),
        heartbeat_unix_ms: now_ms(),
        capabilities: ProviderCapabilities {
            hard_links: true,
            symbolic_links: true,
            byte_range_locks: true,
            memory_maps: true,
            atomic_rename: true,
            case_preserving: true,
        },
    }));
    let core = WorkspaceCore::open(data_root.join("core")).map_err(core_io)?;
    let (filesystem, invalidations) =
        PortableFuse::new(core, data_root.join("doctor"), manifest.clone())?;
    {
        let mut state = manifest.write().unwrap();
        state.state = ProviderState::Ready;
        state.heartbeat_unix_ms = now_ms();
        publish_manifest(&data_root, &state)?;
    }
    let heartbeat_manifest = manifest.clone();
    let heartbeat_root = data_root.clone();
    thread::spawn(move || loop {
        thread::sleep(Duration::from_secs(5));
        let mut state = heartbeat_manifest.write().unwrap();
        state.heartbeat_unix_ms = now_ms();
        if publish_manifest(&heartbeat_root, &state).is_err() {
            state.state = ProviderState::Broken;
            break;
        }
    });
    let mut config = Config::default();
    config.mount_options = vec![
        MountOption::FSName("greppy-workspace".into()),
        MountOption::Subtype("greppy-cow".into()),
        MountOption::RW,
        MountOption::DefaultPermissions,
        MountOption::NoDev,
        MountOption::NoSuid,
        MountOption::Exec,
        MountOption::NoAtime,
    ];
    config.acl = SessionACL::Owner;
    config.n_threads = Some(8);
    config.clone_fd = true;
    let session = Session::new(filesystem, mount_root, &config)?;
    let notifier = session.notifier();
    thread::Builder::new()
        .name("greppy-fuse-invalidate".into())
        .spawn(move || dispatch_invalidations(notifier, invalidations))?;
    let result = session.run();
    let mut state = manifest.write().unwrap();
    state.state = ProviderState::Broken;
    state.heartbeat_unix_ms = now_ms();
    let _ = publish_manifest(&data_root, &state);
    result
}

fn publish_manifest(root: &Path, manifest: &ProviderManifest) -> io::Result<()> {
    let path = root.join("provider.json");
    let temporary = root.join(format!("provider.json.{}.tmp", std::process::id()));
    let bytes = serde_json::to_vec(manifest).map_err(io::Error::other)?;
    fs::write(&temporary, bytes)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn workspace_destination(parent: &Node, name: &OsStr) -> Result<(String, String), Errno> {
    let name = utf8_name(name)?;
    match parent {
        Node::WorkspaceRoot(workspace) => Ok((workspace.clone(), name.into())),
        Node::WorkspacePath { workspace, path } => {
            Ok((workspace.clone(), join_virtual(path, name)))
        }
        _ => Err(Errno::EROFS),
    }
}

fn doctor_destination(parent: &Node, name: &OsStr) -> Result<PathBuf, Errno> {
    match parent {
        Node::DoctorRoot => Ok(PathBuf::from(name)),
        Node::Doctor(path) => Ok(path.join(name)),
        _ => Err(Errno::EXDEV),
    }
}

fn workspace_attr(
    metadata: NodeMetadata,
    uid: u32,
    gid: u32,
) -> (
    FileType,
    u16,
    u64,
    u32,
    u32,
    u32,
    SystemTime,
    SystemTime,
    SystemTime,
) {
    (
        file_type(metadata.kind),
        metadata.mode as u16 & 0o7777,
        metadata.size,
        metadata.nlink,
        uid,
        gid,
        system_time_from_ns(metadata.accessed_unix_ns),
        system_time_from_ns(metadata.modified_unix_ns),
        system_time_from_ns(metadata.changed_unix_ns),
    )
}

fn workspace_file_attr(inode: u64, metadata: NodeMetadata, uid: u32, gid: u32) -> FileAttr {
    let (kind, mode, size, nlink, uid, gid, atime, mtime, ctime) =
        workspace_attr(metadata, uid, gid);
    FileAttr {
        ino: INodeNo(inode),
        size,
        blocks: size.div_ceil(512),
        atime,
        mtime,
        ctime,
        crtime: UNIX_EPOCH,
        kind,
        perm: mode,
        nlink,
        uid,
        gid,
        rdev: 0,
        blksize: 4096,
        flags: 0,
    }
}

fn time_or_now_ns(value: TimeOrNow) -> i64 {
    match value {
        TimeOrNow::SpecificTime(value) => unix_ns(value),
        TimeOrNow::Now => unix_ns(SystemTime::now()),
    }
}

fn unix_ns(value: SystemTime) -> i64 {
    value
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| i64::try_from(duration.as_nanos()).ok())
        .unwrap_or(0)
}

fn system_time_from_ns(value: i64) -> SystemTime {
    u64::try_from(value)
        .ok()
        .and_then(|value| UNIX_EPOCH.checked_add(Duration::from_nanos(value)))
        .unwrap_or(UNIX_EPOCH)
}

fn file_type(kind: NodeKind) -> FileType {
    match kind {
        NodeKind::File => FileType::RegularFile,
        NodeKind::Directory => FileType::Directory,
        NodeKind::Symlink => FileType::Symlink,
    }
}

fn utf8_name(name: &OsStr) -> Result<&str, Errno> {
    name.to_str().ok_or(Errno::EILSEQ)
}

fn join_virtual(parent: &str, name: &str) -> String {
    if parent.is_empty() {
        name.into()
    } else {
        format!("{parent}/{name}")
    }
}

fn slice(bytes: Vec<u8>, offset: u64, size: u32) -> Vec<u8> {
    let start = usize::try_from(offset)
        .unwrap_or(usize::MAX)
        .min(bytes.len());
    let end = start.saturating_add(size as usize).min(bytes.len());
    bytes[start..end].to_vec()
}

fn io_errno(error: io::Error) -> Errno {
    Errno::from_i32(error.raw_os_error().unwrap_or(libc::EIO))
}

fn core_errno(error: CoreError) -> Errno {
    match error.kind() {
        ErrorKind::NotFound => Errno::ENOENT,
        ErrorKind::AlreadyExists => Errno::EEXIST,
        ErrorKind::NotDirectory => Errno::ENOTDIR,
        ErrorKind::IsDirectory => Errno::EISDIR,
        ErrorKind::DirectoryNotEmpty => Errno::ENOTEMPTY,
        ErrorKind::InvalidInput => Errno::EINVAL,
        ErrorKind::Unavailable | ErrorKind::Corrupt | ErrorKind::Io => Errno::EIO,
    }
}

fn serde_errno(_error: serde_json::Error) -> Errno {
    Errno::EIO
}

fn core_io(error: greppy_workspace_core::Error) -> io::Error {
    io::Error::other(error.to_string())
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn invalidation(name: &str) -> RenameInvalidation {
        RenameInvalidation {
            parent: INodeNo(1),
            name: OsString::from(name),
            new_parent: INodeNo(2),
            new_name: OsString::from(format!("new-{name}")),
        }
    }

    #[test]
    fn saturated_invalidation_queue_never_blocks_a_fuse_callback() {
        let (sender, _receiver) = mpsc::sync_channel(1);
        assert!(enqueue_rename_invalidation(&sender, invalidation("first")));
        assert!(!enqueue_rename_invalidation(
            &sender,
            invalidation("second")
        ));
    }
}
