use greppy_workspace_core::{
    AdapterKind, Error as CoreError, ErrorKind, NodeKind, NodeMetadata, ProviderCapabilities,
    ProviderManifest, ProviderState, WorkspaceCore, WorkspaceFileHandle, WorkspaceHandle,
    PROVIDER_PROTOCOL_VERSION,
};
use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::os::windows::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use windows_sys::Win32::Globalization::{CompareStringOrdinal, CSTR_EQUAL};
use windows_sys::Win32::Storage::FileSystem::{
    MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
};
use windows_sys::Win32::System::LibraryLoader::SetDllDirectoryW;

const EPERM: c_int = 1;
const ENOENT: c_int = 2;
const EIO: c_int = 5;
const EEXIST: c_int = 17;
const EXDEV: c_int = 18;
const ENOTDIR: c_int = 20;
const EISDIR: c_int = 21;
const EINVAL: c_int = 22;
const ENOTEMPTY: c_int = 39;
const S_IFDIR: u32 = 0o040000;
const S_IFREG: u32 = 0o100000;
const S_IFLNK: u32 = 0o120000;
const RENAME_NOREPLACE: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GreppyWindowsStat {
    mode: u32,
    size: u64,
    inode: u64,
    nlink: u32,
    accessed_unix_ns: i64,
    modified_unix_ns: i64,
    changed_unix_ns: i64,
}

type DirectoryEmitter = unsafe extern "C" fn(
    context: *mut c_void,
    name: *const c_char,
    metadata: *const GreppyWindowsStat,
    next_offset: u64,
) -> c_int;

#[derive(Debug)]
enum VirtualPath {
    Root,
    Marker,
    Doctor(String),
    Workspaces,
    WorkspaceRoot(String),
    WorkspacePath { workspace: String, path: String },
}

struct WindowsProvider {
    core: WorkspaceCore,
    doctor_root: PathBuf,
    manifest: Arc<RwLock<ProviderManifest>>,
    open_files: Mutex<HashMap<u64, WorkspaceFileHandle>>,
    next_open_file: AtomicU64,
}

pub fn serve(data_root: PathBuf, mount_root: PathBuf) -> io::Result<()> {
    configure_winfsp_runtime()?;
    fs::create_dir_all(&data_root)?;
    prepare_mount_point(&mount_root)?;
    let doctor_root = data_root.join("provider-doctor");
    fs::create_dir_all(&doctor_root)?;
    let core = WorkspaceCore::open(data_root.join("core")).map_err(io::Error::other)?;
    let manifest = Arc::new(RwLock::new(ProviderManifest {
        protocol_version: PROVIDER_PROTOCOL_VERSION,
        adapter_version: env!("CARGO_PKG_VERSION").into(),
        adapter_kind: AdapterKind::WinFsp,
        state: ProviderState::Ready,
        instance_id: format!("winfsp-{}", std::process::id()),
        data_root: data_root.clone(),
        mount_root: mount_root.clone(),
        heartbeat_unix_ms: unix_milliseconds(),
        capabilities: ProviderCapabilities {
            hard_links: true,
            symbolic_links: true,
            byte_range_locks: true,
            memory_maps: true,
            atomic_rename: true,
            case_preserving: true,
        },
    }));
    publish_manifest(&data_root, &manifest.read().unwrap())?;
    spawn_heartbeat(data_root.clone(), Arc::clone(&manifest));

    let mut provider = Box::new(WindowsProvider {
        core,
        doctor_root,
        manifest,
        open_files: Mutex::new(HashMap::new()),
        next_open_file: AtomicU64::new(1),
    });
    let mount = CString::new(mount_root.to_string_lossy().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "mount path contains NUL"))?;
    let result = unsafe {
        greppy_winfsp_mount(
            (&mut *provider as *mut WindowsProvider).cast(),
            mount.as_ptr(),
        )
    };
    if result == 0 {
        Ok(())
    } else {
        Err(io::Error::other(format!(
            "WinFsp dispatcher exited with {result}"
        )))
    }
}

fn prepare_mount_point(mount_root: &Path) -> io::Result<()> {
    let parent = mount_root
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "mount point has no parent"))?;
    fs::create_dir_all(parent)?;
    match fs::symlink_metadata(mount_root) {
        Ok(metadata) => {
            if !metadata.file_type().is_dir() || metadata.file_type().is_symlink() {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    format!(
                        "mount point {} is not an empty directory",
                        mount_root.display()
                    ),
                ));
            }
            if fs::read_dir(mount_root)?.next().transpose()?.is_some() {
                return Err(io::Error::new(
                    io::ErrorKind::DirectoryNotEmpty,
                    format!("mount point {} is not empty", mount_root.display()),
                ));
            }
            fs::remove_dir(mount_root)?;
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }
    Ok(())
}

fn configure_winfsp_runtime() -> io::Result<()> {
    let program_files = std::env::var_os("ProgramFiles(x86)")
        .or_else(|| std::env::var_os("ProgramFiles"))
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Program Files is unavailable"))?;
    let runtime = PathBuf::from(program_files).join("WinFsp").join("bin");
    let library = runtime.join("winfsp-x64.dll");
    if !library.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("WinFsp 2.1 runtime is unavailable at {}", library.display()),
        ));
    }
    let mut wide: Vec<u16> = runtime.as_os_str().encode_wide().collect();
    wide.push(0);
    if unsafe { SetDllDirectoryW(wide.as_ptr()) } == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

unsafe extern "C" {
    fn greppy_winfsp_mount(context: *mut c_void, mountpoint: *const c_char) -> c_int;
}

impl WindowsProvider {
    fn open_file(&self, raw: &str, read_only: bool) -> Result<u64, c_int> {
        let VirtualPath::WorkspacePath { workspace, path } = self.parse(raw)? else {
            self.metadata(raw)?;
            return Ok(0);
        };
        let workspace = self.workspace(&workspace)?;
        let path = self.existing_workspace_path(&workspace, &path)?;
        let file = if read_only {
            self.core.open_file_read_only(&workspace, path)
        } else {
            self.core.open_file(&workspace, path)
        }
        .map_err(core_errno)?;
        let id = self.next_open_file.fetch_add(1, Ordering::Relaxed);
        self.open_files.lock().map_err(|_| -EIO)?.insert(id, file);
        Ok(id)
    }

    fn open_file_handle(&self, id: u64) -> Result<WorkspaceFileHandle, c_int> {
        self.open_files
            .lock()
            .map_err(|_| -EIO)?
            .get(&id)
            .cloned()
            .ok_or(-ENOENT)
    }

    fn parse(&self, raw: &str) -> Result<VirtualPath, c_int> {
        let normalized = raw.replace('\\', "/");
        let components = normalized
            .split('/')
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>();
        if components.iter().any(|part| matches!(*part, "." | "..")) {
            return Err(-EINVAL);
        }
        if components.is_empty() {
            return Ok(VirtualPath::Root);
        }
        if components.len() == 1 && windows_names_equal(components[0], ".greppy-provider.json") {
            return Ok(VirtualPath::Marker);
        }
        if windows_names_equal(components[0], "doctor") {
            return Ok(VirtualPath::Doctor(components[1..].join("/")));
        }
        if !windows_names_equal(components[0], "workspaces") {
            return Err(-ENOENT);
        }
        match components.as_slice() {
            [_] => Ok(VirtualPath::Workspaces),
            [_, workspace] => {
                validate_identifier(workspace)?;
                Ok(VirtualPath::WorkspaceRoot((*workspace).into()))
            }
            [_, workspace, rest @ ..] => {
                validate_identifier(workspace)?;
                Ok(VirtualPath::WorkspacePath {
                    workspace: (*workspace).into(),
                    path: rest.join("/"),
                })
            }
            _ => Err(-ENOENT),
        }
    }

    fn metadata(&self, raw: &str) -> Result<GreppyWindowsStat, c_int> {
        let now = system_time_ns(SystemTime::now());
        match self.parse(raw)? {
            VirtualPath::Root => Ok(synthetic_directory("/", now)),
            VirtualPath::Workspaces => Ok(synthetic_directory("/workspaces", now)),
            VirtualPath::Marker => {
                let size = self.manifest_bytes()?.len() as u64;
                Ok(GreppyWindowsStat {
                    mode: S_IFREG | 0o444,
                    size,
                    inode: hash_path("/.greppy-provider.json"),
                    nlink: 1,
                    accessed_unix_ns: now,
                    modified_unix_ns: now,
                    changed_unix_ns: now,
                })
            }
            VirtualPath::Doctor(relative) if relative.is_empty() => {
                Ok(synthetic_directory("/doctor", now))
            }
            VirtualPath::Doctor(relative) => native_metadata(&self.doctor_path(&relative)?),
            VirtualPath::WorkspaceRoot(workspace) => {
                let workspace = self.workspace(&workspace)?;
                Ok(synthetic_directory(
                    &format!("/workspaces/{}", workspace.id()),
                    now,
                ))
            }
            VirtualPath::WorkspacePath { workspace, path } => {
                let handle = self.workspace(&workspace)?;
                let path = self.existing_workspace_path(&handle, &path)?;
                self.core
                    .metadata(&handle, &path)
                    .map_err(core_errno)?
                    .map(portable_metadata)
                    .ok_or(-ENOENT)
            }
        }
    }

    fn entries(&self, raw: &str) -> Result<Vec<(String, GreppyWindowsStat)>, c_int> {
        match self.parse(raw)? {
            VirtualPath::Root => Ok(vec![
                (
                    ".greppy-provider.json".into(),
                    self.metadata("/.greppy-provider.json")?,
                ),
                ("doctor".into(), self.metadata("/doctor")?),
                ("workspaces".into(), self.metadata("/workspaces")?),
            ]),
            VirtualPath::Workspaces => self
                .core
                .list_workspaces()
                .map_err(core_errno)?
                .into_iter()
                .map(|status| {
                    let path = format!("/workspaces/{}", status.id);
                    Ok((status.id, self.metadata(&path)?))
                })
                .collect(),
            VirtualPath::WorkspaceRoot(workspace) => self.workspace_entries(&workspace, ""),
            VirtualPath::WorkspacePath { workspace, path } => {
                let handle = self.workspace(&workspace)?;
                let path = self.existing_workspace_path(&handle, &path)?;
                self.workspace_entries_from_handle(&handle, &path)
            }
            VirtualPath::Doctor(relative) => fs::read_dir(self.doctor_path(&relative)?)
                .map_err(io_errno)?
                .map(|entry| {
                    let entry = entry.map_err(io_errno)?;
                    let name = entry.file_name().into_string().map_err(|_| -EINVAL)?;
                    native_metadata(&entry.path()).map(|metadata| (name, metadata))
                })
                .collect(),
            VirtualPath::Marker => Err(-ENOTDIR),
        }
    }

    fn workspace_entries(
        &self,
        workspace: &str,
        parent: &str,
    ) -> Result<Vec<(String, GreppyWindowsStat)>, c_int> {
        let handle = self.workspace(workspace)?;
        self.workspace_entries_from_handle(&handle, parent)
    }

    fn workspace_entries_from_handle(
        &self,
        handle: &WorkspaceHandle,
        parent: &str,
    ) -> Result<Vec<(String, GreppyWindowsStat)>, c_int> {
        self.core
            .read_dir(handle, parent)
            .map_err(core_errno)?
            .into_iter()
            .map(|entry| Ok((entry.name, portable_metadata(entry.metadata))))
            .collect()
    }

    fn workspace(&self, id: &str) -> Result<WorkspaceHandle, c_int> {
        match self.core.open_workspace(id) {
            Ok(handle) => Ok(handle),
            Err(error) if error.kind() == ErrorKind::NotFound => {
                let actual = self
                    .core
                    .list_workspaces()
                    .map_err(core_errno)?
                    .into_iter()
                    .find(|workspace| windows_names_equal(&workspace.id, id))
                    .map(|workspace| workspace.id)
                    .ok_or(-ENOENT)?;
                self.core.open_workspace(&actual).map_err(core_errno)
            }
            Err(error) => Err(core_errno(error)),
        }
    }

    fn existing_workspace_path(
        &self,
        workspace: &WorkspaceHandle,
        requested: &str,
    ) -> Result<String, c_int> {
        let mut resolved = String::new();
        for component in requested
            .split('/')
            .filter(|component| !component.is_empty())
        {
            let entry = self
                .core
                .read_dir(workspace, &resolved)
                .map_err(core_errno)?
                .into_iter()
                .find(|entry| windows_names_equal(&entry.name, component))
                .ok_or(-ENOENT)?;
            if !resolved.is_empty() {
                resolved.push('/');
            }
            resolved.push_str(&entry.name);
        }
        Ok(resolved)
    }

    fn new_workspace_path(
        &self,
        workspace: &WorkspaceHandle,
        requested: &str,
    ) -> Result<String, c_int> {
        let (parent, name) = requested.rsplit_once('/').unwrap_or(("", requested));
        if name.is_empty() {
            return Err(-EINVAL);
        }
        let parent = self.existing_workspace_path(workspace, parent)?;
        if self
            .core
            .read_dir(workspace, &parent)
            .map_err(core_errno)?
            .into_iter()
            .any(|entry| windows_names_equal(&entry.name, name))
        {
            return Err(-EEXIST);
        }
        Ok(if parent.is_empty() {
            name.into()
        } else {
            format!("{parent}/{name}")
        })
    }

    fn rename_destination_path(
        &self,
        workspace: &WorkspaceHandle,
        requested: &str,
        source: &str,
        no_replace: bool,
    ) -> Result<String, c_int> {
        let (parent, name) = requested.rsplit_once('/').unwrap_or(("", requested));
        if name.is_empty() {
            return Err(-EINVAL);
        }
        let parent = self.existing_workspace_path(workspace, parent)?;
        let existing = self
            .core
            .read_dir(workspace, &parent)
            .map_err(core_errno)?
            .into_iter()
            .find(|entry| windows_names_equal(&entry.name, name))
            .map(|entry| {
                if parent.is_empty() {
                    entry.name
                } else {
                    format!("{parent}/{}", entry.name)
                }
            });
        if let Some(existing) = existing {
            if existing == source {
                return Ok(if parent.is_empty() {
                    name.into()
                } else {
                    format!("{parent}/{name}")
                });
            }
            if no_replace {
                return Err(-EEXIST);
            }
            return Ok(existing);
        }
        Ok(if parent.is_empty() {
            name.into()
        } else {
            format!("{parent}/{name}")
        })
    }

    fn doctor_path(&self, relative: &str) -> Result<PathBuf, c_int> {
        if relative.split('/').any(|part| matches!(part, "." | "..")) {
            return Err(-EINVAL);
        }
        Ok(self.doctor_root.join(relative.replace('/', "\\")))
    }

    fn manifest_bytes(&self) -> Result<Vec<u8>, c_int> {
        serde_json::to_vec(&*self.manifest.read().unwrap()).map_err(|_| -EIO)
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn greppy_windows_open(
    context: *mut c_void,
    path: *const c_char,
    read_only: c_int,
    output: *mut u64,
) -> c_int {
    ffi_result(context, path, |provider, raw| {
        if output.is_null() {
            return Err(-EINVAL);
        }
        let handle = provider.open_file(raw, read_only != 0)?;
        unsafe { ptr::write(output, handle) };
        Ok(0)
    })
}

#[unsafe(no_mangle)]
unsafe extern "C" fn greppy_windows_release(context: *mut c_void, handle: u64) -> c_int {
    let Ok(provider) = provider(context) else {
        return -EINVAL;
    };
    if handle != 0 {
        let Ok(mut files) = provider.open_files.lock() else {
            return -EIO;
        };
        files.remove(&handle);
    }
    0
}

#[unsafe(no_mangle)]
unsafe extern "C" fn greppy_windows_getattr_handle(
    context: *mut c_void,
    handle: u64,
    output: *mut GreppyWindowsStat,
) -> c_int {
    let Ok(provider) = provider(context) else {
        return -EINVAL;
    };
    if output.is_null() {
        return -EINVAL;
    }
    let result = provider
        .open_file_handle(handle)
        .and_then(|file| provider.core.metadata_open_file(&file).map_err(core_errno))
        .map(portable_metadata);
    match result {
        Ok(value) => {
            unsafe { ptr::write(output, value) };
            0
        }
        Err(error) => error,
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn greppy_windows_truncate_handle(
    context: *mut c_void,
    handle: u64,
    size: u64,
) -> c_int {
    let Ok(provider) = provider(context) else {
        return -EINVAL;
    };
    provider
        .open_file_handle(handle)
        .and_then(|file| {
            provider
                .core
                .truncate_open_file(&file, size)
                .map_err(core_errno)
        })
        .map(|()| 0)
        .unwrap_or_else(|error| error)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn greppy_windows_read_handle(
    context: *mut c_void,
    handle: u64,
    offset: u64,
    output: *mut u8,
    capacity: usize,
) -> c_int {
    let Ok(provider) = provider(context) else {
        return -EINVAL;
    };
    if capacity > c_int::MAX as usize || (capacity != 0 && output.is_null()) {
        return -EINVAL;
    }
    let result = provider.open_file_handle(handle).and_then(|file| {
        provider
            .core
            .read_open_file(&file, offset, capacity)
            .map_err(core_errno)
    });
    match result {
        Ok(bytes) => {
            if !bytes.is_empty() {
                unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), output, bytes.len()) };
            }
            bytes.len() as c_int
        }
        Err(error) => error,
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn greppy_windows_write_handle(
    context: *mut c_void,
    handle: u64,
    offset: u64,
    bytes: *const u8,
    length: usize,
) -> c_int {
    let Ok(provider) = provider(context) else {
        return -EINVAL;
    };
    if length > c_int::MAX as usize || (length != 0 && bytes.is_null()) {
        return -EINVAL;
    }
    let bytes = if length == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(bytes, length) }
    };
    provider
        .open_file_handle(handle)
        .and_then(|file| {
            provider
                .core
                .write_open_file(&file, offset, bytes)
                .map_err(core_errno)
        })
        .map(|written| written as c_int)
        .unwrap_or_else(|error| error)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn greppy_windows_getattr(
    context: *mut c_void,
    path: *const c_char,
    output: *mut GreppyWindowsStat,
) -> c_int {
    ffi_result(context, path, |provider, path| {
        let value = provider.metadata(path)?;
        if output.is_null() {
            return Err(-EINVAL);
        }
        unsafe { ptr::write(output, value) };
        Ok(0)
    })
}

#[unsafe(no_mangle)]
unsafe extern "C" fn greppy_windows_readdir(
    context: *mut c_void,
    path: *const c_char,
    offset: u64,
    emit_context: *mut c_void,
    emit: Option<DirectoryEmitter>,
) -> c_int {
    ffi_result(context, path, |provider, path| {
        let emit = emit.ok_or(-EINVAL)?;
        let current = provider.metadata(path)?;
        let mut entries = vec![(".".to_string(), current), ("..".to_string(), current)];
        entries.extend(provider.entries(path)?);
        for (index, (name, metadata)) in entries.into_iter().enumerate().skip(offset as usize) {
            let name = CString::new(name).map_err(|_| -EINVAL)?;
            if unsafe { emit(emit_context, name.as_ptr(), &metadata, (index + 1) as u64) } != 0 {
                break;
            }
        }
        Ok(0)
    })
}

#[unsafe(no_mangle)]
unsafe extern "C" fn greppy_windows_create(
    context: *mut c_void,
    path: *const c_char,
    mode: u32,
    directory: c_int,
) -> c_int {
    ffi_result(context, path, |provider, raw| {
        match provider.parse(raw)? {
            VirtualPath::Doctor(relative) if relative.is_empty() && directory != 0 => {
                return Err(-EEXIST);
            }
            VirtualPath::Doctor(relative) if !relative.is_empty() => {
                let path = provider.doctor_path(&relative)?;
                if directory != 0 {
                    fs::create_dir(&path).map_err(io_errno)?;
                } else {
                    OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&path)
                        .map_err(io_errno)?;
                }
            }
            VirtualPath::WorkspacePath { workspace, path } => {
                let handle = provider.workspace(&workspace)?;
                let path = provider.new_workspace_path(&handle, &path)?;
                if directory != 0 {
                    provider
                        .core
                        .mkdir(&handle, &path, mode)
                        .map_err(core_errno)?;
                } else {
                    provider
                        .core
                        .create_file(&handle, &path, mode)
                        .map_err(core_errno)?;
                }
            }
            _ => return Err(-EPERM),
        }
        Ok(0)
    })
}

#[unsafe(no_mangle)]
unsafe extern "C" fn greppy_windows_unlink(
    context: *mut c_void,
    path: *const c_char,
    directory: c_int,
) -> c_int {
    ffi_result(context, path, |provider, raw| {
        match provider.parse(raw)? {
            VirtualPath::Doctor(relative) if !relative.is_empty() => {
                let path = provider.doctor_path(&relative)?;
                if directory != 0 {
                    fs::remove_dir(path).map_err(io_errno)?;
                } else {
                    fs::remove_file(path).map_err(io_errno)?;
                }
            }
            VirtualPath::WorkspacePath { workspace, path } => {
                let handle = provider.workspace(&workspace)?;
                let path = provider.existing_workspace_path(&handle, &path)?;
                let metadata = provider
                    .core
                    .metadata(&handle, &path)
                    .map_err(core_errno)?
                    .ok_or(-ENOENT)?;
                if (metadata.kind == NodeKind::Directory) != (directory != 0) {
                    return Err(if directory != 0 { -ENOTDIR } else { -EISDIR });
                }
                provider.core.unlink(&handle, &path).map_err(core_errno)?;
            }
            _ => return Err(-EPERM),
        }
        Ok(0)
    })
}

#[unsafe(no_mangle)]
unsafe extern "C" fn greppy_windows_rename(
    context: *mut c_void,
    source: *const c_char,
    destination: *const c_char,
    flags: u32,
) -> c_int {
    let Ok(provider) = provider(context) else {
        return -EINVAL;
    };
    let Ok(source) = required_str(source) else {
        return -EINVAL;
    };
    let Ok(destination) = required_str(destination) else {
        return -EINVAL;
    };
    let result = (|| {
        let source = provider.parse(source)?;
        let destination = provider.parse(destination)?;
        match (source, destination) {
            (VirtualPath::Doctor(source), VirtualPath::Doctor(destination))
                if !source.is_empty() && !destination.is_empty() =>
            {
                let destination_path = provider.doctor_path(&destination)?;
                if flags & RENAME_NOREPLACE != 0 && destination_path.exists() {
                    return Err(-EEXIST);
                }
                fs::rename(provider.doctor_path(&source)?, destination_path).map_err(io_errno)?;
            }
            (
                VirtualPath::WorkspacePath { workspace, path },
                VirtualPath::WorkspacePath {
                    workspace: destination_workspace,
                    path: destination,
                },
            ) if windows_names_equal(&workspace, &destination_workspace) => {
                let handle = provider.workspace(&workspace)?;
                let path = provider.existing_workspace_path(&handle, &path)?;
                let destination = provider.rename_destination_path(
                    &handle,
                    &destination,
                    &path,
                    flags & RENAME_NOREPLACE != 0,
                )?;
                provider
                    .core
                    .rename(&handle, &path, &destination)
                    .map_err(core_errno)?;
            }
            _ => return Err(-EXDEV),
        }
        Ok(0)
    })();
    result.unwrap_or_else(|error| error)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn greppy_windows_chmod(
    context: *mut c_void,
    path: *const c_char,
    mode: u32,
) -> c_int {
    ffi_result(context, path, |provider, raw| {
        match provider.parse(raw)? {
            VirtualPath::Doctor(relative) if !relative.is_empty() => {
                let path = provider.doctor_path(&relative)?;
                let mut permissions = fs::metadata(&path).map_err(io_errno)?.permissions();
                permissions.set_readonly(mode & 0o222 == 0);
                fs::set_permissions(path, permissions).map_err(io_errno)?;
            }
            VirtualPath::WorkspacePath { workspace, path } => {
                let handle = provider.workspace(&workspace)?;
                let path = provider.existing_workspace_path(&handle, &path)?;
                provider
                    .core
                    .set_metadata(&handle, &path, Some(mode), None, None)
                    .map_err(core_errno)?;
            }
            _ => return Err(-EPERM),
        }
        Ok(0)
    })
}

#[unsafe(no_mangle)]
unsafe extern "C" fn greppy_windows_truncate(
    context: *mut c_void,
    path: *const c_char,
    size: u64,
) -> c_int {
    ffi_result(context, path, |provider, raw| {
        match provider.parse(raw)? {
            VirtualPath::Doctor(relative) if !relative.is_empty() => {
                OpenOptions::new()
                    .write(true)
                    .open(provider.doctor_path(&relative)?)
                    .and_then(|file| file.set_len(size))
                    .map_err(io_errno)?;
            }
            VirtualPath::WorkspacePath { workspace, path } => {
                let handle = provider.workspace(&workspace)?;
                let path = provider.existing_workspace_path(&handle, &path)?;
                provider
                    .core
                    .truncate(&handle, &path, size)
                    .map_err(core_errno)?;
            }
            _ => return Err(-EPERM),
        }
        Ok(0)
    })
}

#[unsafe(no_mangle)]
unsafe extern "C" fn greppy_windows_read(
    context: *mut c_void,
    path: *const c_char,
    offset: u64,
    output: *mut u8,
    capacity: usize,
) -> c_int {
    ffi_result(context, path, |provider, raw| {
        if capacity > c_int::MAX as usize || (capacity != 0 && output.is_null()) {
            return Err(-EINVAL);
        }
        let bytes = match provider.parse(raw)? {
            VirtualPath::Marker => provider.manifest_bytes()?,
            VirtualPath::Doctor(relative) if !relative.is_empty() => {
                let mut file = File::open(provider.doctor_path(&relative)?).map_err(io_errno)?;
                file.seek(SeekFrom::Start(offset)).map_err(io_errno)?;
                let mut bytes = vec![0; capacity];
                let read = file.read(&mut bytes).map_err(io_errno)?;
                bytes.truncate(read);
                bytes
            }
            VirtualPath::WorkspacePath { workspace, path } => {
                let handle = provider.workspace(&workspace)?;
                let path = provider.existing_workspace_path(&handle, &path)?;
                provider
                    .core
                    .read(&handle, &path, offset, capacity)
                    .map_err(core_errno)?
            }
            _ => return Err(-EISDIR),
        };
        let start = if matches!(provider.parse(raw)?, VirtualPath::Marker) {
            usize::try_from(offset)
                .unwrap_or(usize::MAX)
                .min(bytes.len())
        } else {
            0
        };
        let count = capacity.min(bytes.len().saturating_sub(start));
        unsafe { ptr::copy_nonoverlapping(bytes.as_ptr().add(start), output, count) };
        Ok(count as c_int)
    })
}

#[unsafe(no_mangle)]
unsafe extern "C" fn greppy_windows_write(
    context: *mut c_void,
    path: *const c_char,
    offset: u64,
    bytes: *const u8,
    length: usize,
) -> c_int {
    ffi_result(context, path, |provider, raw| {
        if length > c_int::MAX as usize || (length != 0 && bytes.is_null()) {
            return Err(-EINVAL);
        }
        let bytes = unsafe { std::slice::from_raw_parts(bytes, length) };
        match provider.parse(raw)? {
            VirtualPath::Doctor(relative) if !relative.is_empty() => {
                let mut file = OpenOptions::new()
                    .write(true)
                    .open(provider.doctor_path(&relative)?)
                    .map_err(io_errno)?;
                file.seek(SeekFrom::Start(offset)).map_err(io_errno)?;
                file.write_all(bytes).map_err(io_errno)?;
            }
            VirtualPath::WorkspacePath { workspace, path } => {
                let handle = provider.workspace(&workspace)?;
                let path = provider.existing_workspace_path(&handle, &path)?;
                provider
                    .core
                    .write(&handle, &path, offset, bytes)
                    .map_err(core_errno)?;
            }
            _ => return Err(-EPERM),
        }
        Ok(length as c_int)
    })
}

#[unsafe(no_mangle)]
unsafe extern "C" fn greppy_windows_symlink(
    context: *mut c_void,
    path: *const c_char,
    target: *const c_char,
) -> c_int {
    let Ok(provider) = provider(context) else {
        return -EINVAL;
    };
    let Ok(path) = required_str(path) else {
        return -EINVAL;
    };
    let Ok(target) = required_str(target) else {
        return -EINVAL;
    };
    let result = match provider.parse(path) {
        Ok(VirtualPath::Doctor(relative)) if !relative.is_empty() => {
            std::os::windows::fs::symlink_file(target, provider.doctor_path(&relative).unwrap())
                .map_err(io_errno)
        }
        Ok(VirtualPath::WorkspacePath { workspace, path }) => {
            provider.workspace(&workspace).and_then(|handle| {
                let path = provider.new_workspace_path(&handle, &path)?;
                provider
                    .core
                    .symlink(&handle, &path, target.as_bytes())
                    .map_err(core_errno)
            })
        }
        Ok(_) => Err(-EPERM),
        Err(error) => Err(error),
    };
    result.map(|_| 0).unwrap_or_else(|error| error)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn greppy_windows_hardlink(
    context: *mut c_void,
    source: *const c_char,
    destination: *const c_char,
) -> c_int {
    let Ok(provider) = provider(context) else {
        return -EINVAL;
    };
    let Ok(source_raw) = required_str(source) else {
        return -EINVAL;
    };
    let Ok(destination_raw) = required_str(destination) else {
        return -EINVAL;
    };
    let result = match (provider.parse(source_raw), provider.parse(destination_raw)) {
        (Ok(VirtualPath::Doctor(source)), Ok(VirtualPath::Doctor(destination))) => fs::hard_link(
            provider.doctor_path(&source).unwrap(),
            provider.doctor_path(&destination).unwrap(),
        )
        .map_err(io_errno),
        (
            Ok(VirtualPath::WorkspacePath { workspace, path }),
            Ok(VirtualPath::WorkspacePath {
                workspace: destination_workspace,
                path: destination,
            }),
        ) if windows_names_equal(&workspace, &destination_workspace) => {
            provider.workspace(&workspace).and_then(|handle| {
                let path = provider.existing_workspace_path(&handle, &path)?;
                let destination = provider.new_workspace_path(&handle, &destination)?;
                provider
                    .core
                    .hard_link(&handle, &path, &destination)
                    .map_err(core_errno)
            })
        }
        (Err(error), _) | (_, Err(error)) => Err(error),
        _ => Err(-EXDEV),
    };
    result.map(|_| 0).unwrap_or_else(|error| error)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn greppy_windows_readlink(
    context: *mut c_void,
    path: *const c_char,
    output: *mut u8,
    capacity: usize,
) -> c_int {
    ffi_result(context, path, |provider, raw| {
        let target = match provider.parse(raw)? {
            VirtualPath::Doctor(relative) if !relative.is_empty() => {
                fs::read_link(provider.doctor_path(&relative)?)
                    .map_err(io_errno)?
                    .to_string_lossy()
                    .into_owned()
                    .into_bytes()
            }
            VirtualPath::WorkspacePath { workspace, path } => {
                let handle = provider.workspace(&workspace)?;
                let path = provider.existing_workspace_path(&handle, &path)?;
                provider
                    .core
                    .read_symlink(&handle, &path)
                    .map_err(core_errno)?
            }
            _ => return Err(-EINVAL),
        };
        if target.len() > capacity || (capacity != 0 && output.is_null()) {
            return Err(-EINVAL);
        }
        unsafe { ptr::copy_nonoverlapping(target.as_ptr(), output, target.len()) };
        Ok(target.len() as c_int)
    })
}

#[unsafe(no_mangle)]
unsafe extern "C" fn greppy_windows_set_times(
    context: *mut c_void,
    path: *const c_char,
    accessed: i64,
    modified: i64,
) -> c_int {
    ffi_result(context, path, |provider, raw| {
        match provider.parse(raw)? {
            VirtualPath::Doctor(relative) if !relative.is_empty() => {
                filetime::set_file_times(
                    provider.doctor_path(&relative)?,
                    filetime::FileTime::from_unix_time(
                        accessed.div_euclid(1_000_000_000),
                        accessed.rem_euclid(1_000_000_000) as u32,
                    ),
                    filetime::FileTime::from_unix_time(
                        modified.div_euclid(1_000_000_000),
                        modified.rem_euclid(1_000_000_000) as u32,
                    ),
                )
                .map_err(io_errno)?;
            }
            VirtualPath::WorkspacePath { workspace, path } => {
                let handle = provider.workspace(&workspace)?;
                let path = provider.existing_workspace_path(&handle, &path)?;
                provider
                    .core
                    .set_metadata(&handle, &path, None, Some(accessed), Some(modified))
                    .map_err(core_errno)?;
            }
            _ => return Err(-EPERM),
        }
        Ok(0)
    })
}

fn ffi_result(
    context: *mut c_void,
    path: *const c_char,
    operation: impl FnOnce(&WindowsProvider, &str) -> Result<c_int, c_int>,
) -> c_int {
    let Ok(provider) = provider(context) else {
        return -EINVAL;
    };
    let Ok(path) = required_str(path) else {
        return -EINVAL;
    };
    operation(provider, path).unwrap_or_else(|error| error)
}

fn provider<'a>(value: *mut c_void) -> Result<&'a WindowsProvider, ()> {
    if value.is_null() {
        Err(())
    } else {
        Ok(unsafe { &*value.cast::<WindowsProvider>() })
    }
}

fn required_str<'a>(value: *const c_char) -> Result<&'a str, ()> {
    if value.is_null() {
        return Err(());
    }
    unsafe { CStr::from_ptr(value) }.to_str().map_err(|_| ())
}

fn windows_names_equal(left: &str, right: &str) -> bool {
    let left = left.encode_utf16().collect::<Vec<_>>();
    let right = right.encode_utf16().collect::<Vec<_>>();
    let Ok(left_len) = i32::try_from(left.len()) else {
        return false;
    };
    let Ok(right_len) = i32::try_from(right.len()) else {
        return false;
    };
    unsafe {
        CompareStringOrdinal(left.as_ptr(), left_len, right.as_ptr(), right_len, 1) == CSTR_EQUAL
    }
}

fn validate_identifier(value: &str) -> Result<(), c_int> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        Err(-EINVAL)
    } else {
        Ok(())
    }
}

fn portable_metadata(value: NodeMetadata) -> GreppyWindowsStat {
    let kind = match value.kind {
        NodeKind::File => S_IFREG,
        NodeKind::Directory => S_IFDIR,
        NodeKind::Symlink => S_IFLNK,
    };
    GreppyWindowsStat {
        // WorkspaceCore stores permission bits and node kind separately.
        // WinFsp's FUSE ABI expects both in st_mode; omitting the kind makes
        // freshly-created directories look like regular files to Windows.
        mode: kind | (value.mode & 0o7777),
        size: value.size,
        inode: value.inode,
        nlink: value.nlink,
        accessed_unix_ns: value.accessed_unix_ns,
        modified_unix_ns: value.modified_unix_ns,
        changed_unix_ns: value.changed_unix_ns,
    }
}

fn synthetic_directory(path: &str, now: i64) -> GreppyWindowsStat {
    GreppyWindowsStat {
        mode: S_IFDIR | 0o700,
        size: 0,
        inode: hash_path(path),
        nlink: 2,
        accessed_unix_ns: now,
        modified_unix_ns: now,
        changed_unix_ns: now,
    }
}

fn native_metadata(path: &Path) -> Result<GreppyWindowsStat, c_int> {
    let metadata = fs::symlink_metadata(path).map_err(io_errno)?;
    let kind = if metadata.file_type().is_symlink() {
        S_IFLNK
    } else if metadata.is_dir() {
        S_IFDIR
    } else {
        S_IFREG
    };
    let readonly = metadata.permissions().readonly();
    let modified = metadata.modified().unwrap_or(SystemTime::now());
    let accessed = metadata.accessed().unwrap_or(modified);
    let changed = metadata.created().unwrap_or(modified);
    Ok(GreppyWindowsStat {
        mode: kind | if readonly { 0o444 } else { 0o666 },
        size: metadata.len(),
        inode: hash_path(&path.to_string_lossy()),
        nlink: 1,
        accessed_unix_ns: system_time_ns(accessed),
        modified_unix_ns: system_time_ns(modified),
        changed_unix_ns: system_time_ns(changed),
    })
}

fn core_errno(error: CoreError) -> c_int {
    -match error.kind() {
        ErrorKind::NotFound => ENOENT,
        ErrorKind::AlreadyExists => EEXIST,
        ErrorKind::NotDirectory => ENOTDIR,
        ErrorKind::IsDirectory => EISDIR,
        ErrorKind::DirectoryNotEmpty => ENOTEMPTY,
        ErrorKind::InvalidInput => EINVAL,
        ErrorKind::Unavailable | ErrorKind::Corrupt | ErrorKind::Io => EIO,
    }
}

fn io_errno(error: io::Error) -> c_int {
    -match error.kind() {
        io::ErrorKind::NotFound => ENOENT,
        io::ErrorKind::AlreadyExists => EEXIST,
        io::ErrorKind::NotADirectory => ENOTDIR,
        io::ErrorKind::IsADirectory => EISDIR,
        io::ErrorKind::DirectoryNotEmpty => ENOTEMPTY,
        io::ErrorKind::InvalidInput | io::ErrorKind::InvalidData => EINVAL,
        io::ErrorKind::PermissionDenied => EPERM,
        _ => EIO,
    }
}

fn hash_path(path: &str) -> u64 {
    path.as_bytes()
        .iter()
        .fold(14_695_981_039_346_656_037, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(1_099_511_628_211)
        })
}

fn unix_milliseconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn system_time_ns(value: SystemTime) -> i64 {
    value
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos().min(i64::MAX as u128) as i64)
        .unwrap_or_default()
}

fn spawn_heartbeat(root: PathBuf, manifest: Arc<RwLock<ProviderManifest>>) {
    thread::spawn(move || loop {
        thread::sleep(Duration::from_secs(2));
        {
            let mut value = manifest.write().unwrap();
            value.heartbeat_unix_ms = unix_milliseconds();
            if publish_manifest(&root, &value).is_err() {
                value.state = ProviderState::Broken;
                let _ = publish_manifest(&root, &value);
                break;
            }
        }
    });
}

fn publish_manifest(root: &Path, manifest: &ProviderManifest) -> io::Result<()> {
    let destination = root.join("provider.json");
    let temporary = root.join(format!("provider.json.{}.tmp", std::process::id()));
    fs::write(
        &temporary,
        serde_json::to_vec(manifest).map_err(io::Error::other)?,
    )?;
    let source_wide = temporary
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect::<Vec<_>>();
    let destination_wide = destination
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect::<Vec<_>>();
    let moved = unsafe {
        MoveFileExW(
            source_wide.as_ptr(),
            destination_wide.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if moved == 0 {
        let error = io::Error::last_os_error();
        let _ = fs::remove_file(&temporary);
        Err(error)
    } else {
        Ok(())
    }
}
