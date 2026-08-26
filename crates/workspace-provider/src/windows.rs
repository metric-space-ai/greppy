use greppy_workspace_core::{
    AdapterKind, Error as CoreError, ErrorKind, NodeKind, NodeMetadata, ProviderCapabilities,
    ProviderManifest, ProviderState, WorkspaceCore, WorkspaceHandle, PROVIDER_PROTOCOL_VERSION,
};
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::os::windows::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
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
    fn parse(&self, raw: &str) -> Result<VirtualPath, c_int> {
        let normalized = raw.replace('\\', "/");
        let components = normalized
            .split('/')
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>();
        if components.iter().any(|part| matches!(*part, "." | "..")) {
            return Err(-EINVAL);
        }
        match components.as_slice() {
            [] => Ok(VirtualPath::Root),
            [".greppy-provider.json"] => Ok(VirtualPath::Marker),
            ["doctor"] => Ok(VirtualPath::Doctor(String::new())),
            ["doctor", rest @ ..] => Ok(VirtualPath::Doctor(rest.join("/"))),
            ["workspaces"] => Ok(VirtualPath::Workspaces),
            ["workspaces", workspace] => {
                validate_identifier(workspace)?;
                Ok(VirtualPath::WorkspaceRoot((*workspace).into()))
            }
            ["workspaces", workspace, rest @ ..] => {
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
            VirtualPath::Root | VirtualPath::Workspaces => Ok(synthetic_directory(raw, now)),
            VirtualPath::Marker => {
                let size = self.manifest_bytes()?.len() as u64;
                Ok(GreppyWindowsStat {
                    mode: S_IFREG | 0o400,
                    size,
                    inode: hash_path(raw),
                    nlink: 1,
                    accessed_unix_ns: now,
                    modified_unix_ns: now,
                    changed_unix_ns: now,
                })
            }
            VirtualPath::Doctor(relative) if relative.is_empty() => {
                Ok(synthetic_directory(raw, now))
            }
            VirtualPath::Doctor(relative) => native_metadata(&self.doctor_path(&relative)?),
            VirtualPath::WorkspaceRoot(workspace) => {
                self.workspace(&workspace)?;
                Ok(synthetic_directory(raw, now))
            }
            VirtualPath::WorkspacePath { workspace, path } => {
                let handle = self.workspace(&workspace)?;
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
                self.workspace_entries(&workspace, &path)
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
        self.core
            .read_dir(&handle, parent)
            .map_err(core_errno)?
            .into_iter()
            .map(|entry| Ok((entry.name, portable_metadata(entry.metadata))))
            .collect()
    }

    fn workspace(&self, id: &str) -> Result<WorkspaceHandle, c_int> {
        self.core.open_workspace(id).map_err(core_errno)
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
    emit_context: *mut c_void,
    emit: Option<DirectoryEmitter>,
) -> c_int {
    ffi_result(context, path, |provider, path| {
        let emit = emit.ok_or(-EINVAL)?;
        for (name, metadata) in provider.entries(path)? {
            let name = CString::new(name).map_err(|_| -EINVAL)?;
            if unsafe { emit(emit_context, name.as_ptr(), &metadata) } != 0 {
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
            ) if workspace == destination_workspace => {
                let handle = provider.workspace(&workspace)?;
                if flags & RENAME_NOREPLACE != 0
                    && provider
                        .core
                        .metadata(&handle, &destination)
                        .map_err(core_errno)?
                        .is_some()
                {
                    return Err(-EEXIST);
                }
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
        ) if workspace == destination_workspace => {
            provider.workspace(&workspace).and_then(|handle| {
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
    GreppyWindowsStat {
        mode: value.mode,
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
