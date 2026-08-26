use greppy_workspace_core::{
    capture_repository, NodeKind, NodeMetadata, WorkspaceCore, WorkspaceHandle,
};
use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::path::Path;
use std::ptr;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap());
}

#[repr(C)]
pub struct GreppyWorkspaceCore {
    core: WorkspaceCore,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GreppyWorkspaceMetadata {
    pub kind: u8,
    pub mode: u32,
    pub size: u64,
    pub inode: u64,
    pub nlink: u32,
    pub accessed_unix_ns: i64,
    pub modified_unix_ns: i64,
    pub changed_unix_ns: i64,
}

fn remember(error: impl std::fmt::Display) -> i32 {
    let message = error.to_string().replace('\0', "\\0");
    LAST_ERROR.with(|slot| *slot.borrow_mut() = CString::new(message).unwrap());
    -1
}

unsafe fn required_str<'a>(value: *const c_char, name: &str) -> Result<&'a str, i32> {
    if value.is_null() {
        return Err(remember(format!("{name} is null")));
    }
    CStr::from_ptr(value)
        .to_str()
        .map_err(|_| remember(format!("{name} is not UTF-8")))
}

unsafe fn core<'a>(value: *mut GreppyWorkspaceCore) -> Result<&'a WorkspaceCore, i32> {
    value
        .as_ref()
        .map(|value| &value.core)
        .ok_or_else(|| remember("workspace core is null"))
}

fn workspace(core: &WorkspaceCore, id: &str) -> Result<WorkspaceHandle, i32> {
    core.open_workspace(id).map_err(remember)
}

fn status(result: greppy_workspace_core::Result<()>) -> i32 {
    match result {
        Ok(()) => 0,
        Err(error) => remember(error),
    }
}

#[no_mangle]
/// Opens a portable workspace core for an absolute data root.
///
/// # Safety
/// `absolute_data_root` must point to a valid NUL-terminated string for the
/// duration of this call. The returned pointer must eventually be passed
/// exactly once to [`greppy_workspace_core_close`].
pub unsafe extern "C" fn greppy_workspace_core_open(
    absolute_data_root: *const c_char,
) -> *mut GreppyWorkspaceCore {
    let Ok(root) = required_str(absolute_data_root, "absolute_data_root") else {
        return ptr::null_mut();
    };
    if !Path::new(root).is_absolute() {
        remember("workspace data root must be absolute");
        return ptr::null_mut();
    }
    match WorkspaceCore::open(root) {
        Ok(core) => Box::into_raw(Box::new(GreppyWorkspaceCore { core })),
        Err(error) => {
            remember(error);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
/// Releases a workspace core returned by [`greppy_workspace_core_open`].
///
/// # Safety
/// `core` must be null or an owned pointer returned by
/// [`greppy_workspace_core_open`] that has not already been closed. No other
/// thread may use the core while it is being closed.
pub unsafe extern "C" fn greppy_workspace_core_close(core: *mut GreppyWorkspaceCore) {
    if !core.is_null() {
        drop(Box::from_raw(core));
    }
}

#[no_mangle]
/// Captures a repository and creates a new workspace namespace.
///
/// # Safety
/// `value` must be a live core pointer. Both string pointers must be valid
/// NUL-terminated strings for the duration of this call, and the core must not
/// be closed concurrently.
pub unsafe extern "C" fn greppy_workspace_create(
    value: *mut GreppyWorkspaceCore,
    workspace_id: *const c_char,
    absolute_repository: *const c_char,
) -> i32 {
    let Ok(core) = core(value) else { return -1 };
    let Ok(id) = required_str(workspace_id, "workspace_id") else {
        return -1;
    };
    let Ok(repository) = required_str(absolute_repository, "absolute_repository") else {
        return -1;
    };
    if !Path::new(repository).is_absolute() {
        return remember("repository path must be absolute");
    }
    match capture_repository(repository, core.chunks())
        .and_then(|baseline| core.create_workspace(id, baseline).map(|_| ()))
    {
        Ok(()) => 0,
        Err(error) => remember(error),
    }
}

#[no_mangle]
/// Removes a workspace namespace.
///
/// # Safety
/// `value` must be a live core pointer and `workspace_id` must point to a valid
/// NUL-terminated string for the duration of this call. The core must not be
/// closed concurrently.
pub unsafe extern "C" fn greppy_workspace_remove(
    value: *mut GreppyWorkspaceCore,
    workspace_id: *const c_char,
) -> i32 {
    let Ok(core) = core(value) else { return -1 };
    let Ok(id) = required_str(workspace_id, "workspace_id") else {
        return -1;
    };
    match workspace(core, id) {
        Ok(workspace) => status(core.remove_workspace(workspace)),
        Err(code) => code,
    }
}

fn metadata(value: NodeMetadata) -> GreppyWorkspaceMetadata {
    GreppyWorkspaceMetadata {
        kind: match value.kind {
            NodeKind::File => 1,
            NodeKind::Directory => 2,
            NodeKind::Symlink => 3,
        },
        mode: value.mode,
        size: value.size,
        inode: value.inode,
        nlink: value.nlink,
        accessed_unix_ns: value.accessed_unix_ns,
        modified_unix_ns: value.modified_unix_ns,
        changed_unix_ns: value.changed_unix_ns,
    }
}

#[no_mangle]
/// Reads metadata for a path into caller-owned storage.
///
/// # Safety
/// `value` must be a live core pointer; `workspace_id` and `path` must be valid
/// NUL-terminated strings; and `out` must be valid and aligned for one
/// [`GreppyWorkspaceMetadata`] write. All pointers must remain valid for the
/// duration of this call.
pub unsafe extern "C" fn greppy_workspace_metadata(
    value: *mut GreppyWorkspaceCore,
    workspace_id: *const c_char,
    path: *const c_char,
    out: *mut GreppyWorkspaceMetadata,
) -> i32 {
    let Ok(core) = core(value) else { return -1 };
    let Ok(id) = required_str(workspace_id, "workspace_id") else {
        return -1;
    };
    let Ok(path) = required_str(path, "path") else {
        return -1;
    };
    if out.is_null() {
        return remember("metadata output is null");
    }
    let result = workspace(core, id).and_then(|workspace| {
        core.metadata(&workspace, path)
            .map_err(remember)?
            .ok_or_else(|| remember(format!("path does not exist: {path}")))
    });
    match result {
        Ok(value) => {
            out.write(metadata(value));
            0
        }
        Err(code) => code,
    }
}

#[no_mangle]
/// Reads file bytes into a caller-provided buffer.
///
/// # Safety
/// `value` must be a live core pointer; the string pointers must be valid
/// NUL-terminated strings; and, when `capacity` is nonzero, `out` must be
/// writable for `capacity` bytes. All pointers must remain valid and the core
/// must not be closed for the duration of this call.
pub unsafe extern "C" fn greppy_workspace_read(
    value: *mut GreppyWorkspaceCore,
    workspace_id: *const c_char,
    path: *const c_char,
    offset: u64,
    out: *mut u8,
    capacity: usize,
) -> i64 {
    let Ok(core) = core(value) else { return -1 };
    let Ok(id) = required_str(workspace_id, "workspace_id") else {
        return -1;
    };
    let Ok(path) = required_str(path, "path") else {
        return -1;
    };
    if capacity != 0 && out.is_null() {
        return remember("read output is null") as i64;
    }
    let result = workspace(core, id).and_then(|workspace| {
        core.read(&workspace, path, offset, capacity)
            .map_err(remember)
    });
    match result {
        Ok(bytes) => {
            if !bytes.is_empty() {
                ptr::copy_nonoverlapping(bytes.as_ptr(), out, bytes.len());
            }
            bytes.len() as i64
        }
        Err(code) => code as i64,
    }
}

#[no_mangle]
/// Reads a symbolic-link target into a caller-provided buffer.
///
/// # Safety
/// `value` must be a live core pointer; the string pointers must be valid
/// NUL-terminated strings; and, when `capacity` is nonzero, `out` must be
/// writable for `capacity` bytes. All pointers must remain valid and the core
/// must not be closed for the duration of this call.
pub unsafe extern "C" fn greppy_workspace_read_symlink(
    value: *mut GreppyWorkspaceCore,
    workspace_id: *const c_char,
    path: *const c_char,
    out: *mut u8,
    capacity: usize,
) -> i64 {
    let Ok(core) = core(value) else { return -1 };
    let Ok(id) = required_str(workspace_id, "workspace_id") else {
        return -1;
    };
    let Ok(path) = required_str(path, "path") else {
        return -1;
    };
    if capacity != 0 && out.is_null() {
        return remember("readlink output is null") as i64;
    }
    let result = workspace(core, id)
        .and_then(|workspace| core.read_symlink(&workspace, path).map_err(remember));
    match result {
        Ok(bytes) if bytes.len() <= capacity => {
            if !bytes.is_empty() {
                ptr::copy_nonoverlapping(bytes.as_ptr(), out, bytes.len());
            }
            bytes.len() as i64
        }
        Ok(bytes) => remember(format!(
            "readlink buffer has {capacity} bytes, target needs {}",
            bytes.len()
        )) as i64,
        Err(code) => code as i64,
    }
}

#[no_mangle]
/// Writes bytes into a workspace file at `offset`.
///
/// # Safety
/// `value` must be a live core pointer; the string pointers must be valid
/// NUL-terminated strings; and, when `length` is nonzero, `bytes` must be
/// readable for `length` bytes. All pointers must remain valid and the core
/// must not be closed for the duration of this call.
pub unsafe extern "C" fn greppy_workspace_write(
    value: *mut GreppyWorkspaceCore,
    workspace_id: *const c_char,
    path: *const c_char,
    offset: u64,
    bytes: *const u8,
    length: usize,
) -> i64 {
    let Ok(core) = core(value) else { return -1 };
    let Ok(id) = required_str(workspace_id, "workspace_id") else {
        return -1;
    };
    let Ok(path) = required_str(path, "path") else {
        return -1;
    };
    if length != 0 && bytes.is_null() {
        return remember("write input is null") as i64;
    }
    let bytes = if length == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(bytes, length)
    };
    match workspace(core, id).and_then(|workspace| {
        core.write(&workspace, path, offset, bytes)
            .map_err(remember)
    }) {
        Ok(written) => written as i64,
        Err(code) => code as i64,
    }
}

macro_rules! path_operation {
    ($name:ident, $method:ident) => {
        #[no_mangle]
        /// Applies a single-path namespace operation.
        ///
        /// # Safety
        /// `value` must be a live core pointer and both string pointers must
        /// reference valid NUL-terminated strings for the duration of this
        /// call. The core must not be closed concurrently.
        pub unsafe extern "C" fn $name(
            value: *mut GreppyWorkspaceCore,
            workspace_id: *const c_char,
            path: *const c_char,
        ) -> i32 {
            let Ok(core) = core(value) else { return -1 };
            let Ok(id) = required_str(workspace_id, "workspace_id") else {
                return -1;
            };
            let Ok(path) = required_str(path, "path") else {
                return -1;
            };
            match workspace(core, id) {
                Ok(workspace) => status(core.$method(&workspace, path)),
                Err(code) => code,
            }
        }
    };
}

path_operation!(greppy_workspace_unlink, unlink);

#[no_mangle]
/// Changes a workspace file's logical length.
///
/// # Safety
/// `value` must be a live core pointer and both string pointers must reference
/// valid NUL-terminated strings for the duration of this call. The core must
/// not be closed concurrently.
pub unsafe extern "C" fn greppy_workspace_truncate(
    value: *mut GreppyWorkspaceCore,
    workspace_id: *const c_char,
    path: *const c_char,
    size: u64,
) -> i32 {
    let Ok(core) = core(value) else { return -1 };
    let Ok(id) = required_str(workspace_id, "workspace_id") else {
        return -1;
    };
    let Ok(path) = required_str(path, "path") else {
        return -1;
    };
    match workspace(core, id) {
        Ok(workspace) => status(core.truncate(&workspace, path, size)),
        Err(code) => code,
    }
}

#[no_mangle]
/// Updates selected metadata fields for a workspace path.
///
/// # Safety
/// `value` must be a live core pointer and both string pointers must reference
/// valid NUL-terminated strings for the duration of this call. The core must
/// not be closed concurrently.
pub unsafe extern "C" fn greppy_workspace_set_metadata(
    value: *mut GreppyWorkspaceCore,
    workspace_id: *const c_char,
    path: *const c_char,
    valid: u32,
    mode: u32,
    accessed_unix_ns: i64,
    modified_unix_ns: i64,
) -> i32 {
    if valid & !0b111 != 0 {
        return remember(format!("unknown metadata validity bits: {valid:#x}"));
    }
    let Ok(core) = core(value) else { return -1 };
    let Ok(id) = required_str(workspace_id, "workspace_id") else {
        return -1;
    };
    let Ok(path) = required_str(path, "path") else {
        return -1;
    };
    match workspace(core, id) {
        Ok(workspace) => status(core.set_metadata(
            &workspace,
            path,
            (valid & 1 != 0).then_some(mode),
            (valid & 2 != 0).then_some(accessed_unix_ns),
            (valid & 4 != 0).then_some(modified_unix_ns),
        )),
        Err(code) => code,
    }
}

unsafe fn create_node(
    value: *mut GreppyWorkspaceCore,
    workspace_id: *const c_char,
    path: *const c_char,
    mode: u32,
    directory: bool,
) -> i32 {
    let Ok(core) = core(value) else { return -1 };
    let Ok(id) = required_str(workspace_id, "workspace_id") else {
        return -1;
    };
    let Ok(path) = required_str(path, "path") else {
        return -1;
    };
    match workspace(core, id) {
        Ok(workspace) if directory => status(core.mkdir(&workspace, path, mode)),
        Ok(workspace) => status(core.create_file(&workspace, path, mode)),
        Err(code) => code,
    }
}

#[no_mangle]
/// Creates a regular file in a workspace namespace.
///
/// # Safety
/// `core` must be a live core pointer and both string pointers must reference
/// valid NUL-terminated strings for the duration of this call. The core must
/// not be closed concurrently.
pub unsafe extern "C" fn greppy_workspace_create_file(
    core: *mut GreppyWorkspaceCore,
    workspace_id: *const c_char,
    path: *const c_char,
    mode: u32,
) -> i32 {
    create_node(core, workspace_id, path, mode, false)
}

#[no_mangle]
/// Creates a directory in a workspace namespace.
///
/// # Safety
/// `core` must be a live core pointer and both string pointers must reference
/// valid NUL-terminated strings for the duration of this call. The core must
/// not be closed concurrently.
pub unsafe extern "C" fn greppy_workspace_mkdir(
    core: *mut GreppyWorkspaceCore,
    workspace_id: *const c_char,
    path: *const c_char,
    mode: u32,
) -> i32 {
    create_node(core, workspace_id, path, mode, true)
}

unsafe fn two_path_operation(
    value: *mut GreppyWorkspaceCore,
    workspace_id: *const c_char,
    source: *const c_char,
    destination: *const c_char,
    hard_link: bool,
) -> i32 {
    let Ok(core) = core(value) else { return -1 };
    let Ok(id) = required_str(workspace_id, "workspace_id") else {
        return -1;
    };
    let Ok(source) = required_str(source, "source") else {
        return -1;
    };
    let Ok(destination) = required_str(destination, "destination") else {
        return -1;
    };
    match workspace(core, id) {
        Ok(workspace) if hard_link => status(core.hard_link(&workspace, source, destination)),
        Ok(workspace) => status(core.rename(&workspace, source, destination)),
        Err(code) => code,
    }
}

#[no_mangle]
/// Atomically renames a workspace path.
///
/// # Safety
/// `core` must be a live core pointer and all string pointers must reference
/// valid NUL-terminated strings for the duration of this call. The core must
/// not be closed concurrently.
pub unsafe extern "C" fn greppy_workspace_rename(
    core: *mut GreppyWorkspaceCore,
    workspace_id: *const c_char,
    source: *const c_char,
    destination: *const c_char,
) -> i32 {
    two_path_operation(core, workspace_id, source, destination, false)
}

#[no_mangle]
/// Creates another directory entry for the source inode.
///
/// # Safety
/// `core` must be a live core pointer and all string pointers must reference
/// valid NUL-terminated strings for the duration of this call. The core must
/// not be closed concurrently.
pub unsafe extern "C" fn greppy_workspace_hard_link(
    core: *mut GreppyWorkspaceCore,
    workspace_id: *const c_char,
    source: *const c_char,
    destination: *const c_char,
) -> i32 {
    two_path_operation(core, workspace_id, source, destination, true)
}

#[no_mangle]
/// Creates a symbolic link containing arbitrary target bytes.
///
/// # Safety
/// `value` must be a live core pointer; both string pointers must reference
/// valid NUL-terminated strings; and, when `target_len` is nonzero, `target`
/// must be readable for `target_len` bytes. All pointers must remain valid and
/// the core must not be closed for the duration of this call.
pub unsafe extern "C" fn greppy_workspace_symlink(
    value: *mut GreppyWorkspaceCore,
    workspace_id: *const c_char,
    path: *const c_char,
    target: *const u8,
    target_len: usize,
) -> i32 {
    let Ok(core) = core(value) else { return -1 };
    let Ok(id) = required_str(workspace_id, "workspace_id") else {
        return -1;
    };
    let Ok(path) = required_str(path, "path") else {
        return -1;
    };
    if target_len != 0 && target.is_null() {
        return remember("symlink target is null");
    }
    let target = if target_len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(target, target_len)
    };
    match workspace(core, id) {
        Ok(workspace) => status(core.symlink(&workspace, path, target)),
        Err(code) => code,
    }
}

#[no_mangle]
/// Returns a newly allocated JSON directory listing.
///
/// # Safety
/// `value` must be a live core pointer and both string pointers must reference
/// valid NUL-terminated strings for the duration of this call. The returned
/// string must be released exactly once with [`greppy_workspace_string_free`].
pub unsafe extern "C" fn greppy_workspace_list_json(
    value: *mut GreppyWorkspaceCore,
    workspace_id: *const c_char,
    path: *const c_char,
) -> *mut c_char {
    let Ok(core) = core(value) else {
        return ptr::null_mut();
    };
    let Ok(id) = required_str(workspace_id, "workspace_id") else {
        return ptr::null_mut();
    };
    let Ok(path) = required_str(path, "path") else {
        return ptr::null_mut();
    };
    let result = workspace(core, id).and_then(|workspace| {
        core.read_dir(&workspace, path)
            .map_err(remember)
            .and_then(|entries| serde_json::to_string(&entries).map_err(remember))
    });
    match result.and_then(|json| CString::new(json).map_err(remember)) {
        Ok(json) => json.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
/// Returns a newly allocated JSON workspace listing.
///
/// # Safety
/// `value` must be a live core pointer that is not closed during this call.
/// The returned string must be released exactly once with
/// [`greppy_workspace_string_free`].
pub unsafe extern "C" fn greppy_workspace_list_workspaces_json(
    value: *mut GreppyWorkspaceCore,
) -> *mut c_char {
    let Ok(core) = core(value) else {
        return ptr::null_mut();
    };
    let result = core
        .list_workspaces()
        .map_err(remember)
        .and_then(|workspaces| serde_json::to_string(&workspaces).map_err(remember));
    match result.and_then(|json| CString::new(json).map_err(remember)) {
        Ok(json) => json.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn greppy_workspace_last_error() -> *mut c_char {
    LAST_ERROR.with(|slot| slot.borrow().clone().into_raw())
}

#[no_mangle]
/// Releases a string allocated by this FFI module.
///
/// # Safety
/// `value` must be null or a pointer returned by a Greppy workspace FFI string
/// function that has not already been freed. It must not be used after this
/// call.
pub unsafe extern "C" fn greppy_workspace_string_free(value: *mut c_char) {
    if !value.is_null() {
        drop(CString::from_raw(value));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    fn c(value: &str) -> CString {
        CString::new(value).unwrap()
    }

    #[test]
    fn c_abi_exercises_workspace_lifecycle_and_io() {
        let repo = tempfile::tempdir().unwrap();
        for args in [
            &["init", "-q"][..],
            &["config", "user.email", "test@example.test"][..],
            &["config", "user.name", "Test"][..],
        ] {
            assert!(Command::new("git")
                .args(args)
                .current_dir(repo.path())
                .status()
                .unwrap()
                .success());
        }
        std::fs::write(repo.path().join("base.txt"), "base").unwrap();
        assert!(Command::new("git")
            .args(["add", "."])
            .current_dir(repo.path())
            .status()
            .unwrap()
            .success());
        assert!(Command::new("git")
            .args(["commit", "-qm", "base"])
            .current_dir(repo.path())
            .status()
            .unwrap()
            .success());
        let storage = tempfile::tempdir().unwrap();
        unsafe {
            let core = greppy_workspace_core_open(c(storage.path().to_str().unwrap()).as_ptr());
            assert!(!core.is_null());
            assert_eq!(
                greppy_workspace_create(
                    core,
                    c("ffi-test").as_ptr(),
                    c(repo.path().to_str().unwrap()).as_ptr()
                ),
                0
            );
            assert_eq!(
                greppy_workspace_create_file(
                    core,
                    c("ffi-test").as_ptr(),
                    c("private.txt").as_ptr(),
                    0o100600
                ),
                0
            );
            let body = b"through ffi";
            assert_eq!(
                greppy_workspace_write(
                    core,
                    c("ffi-test").as_ptr(),
                    c("private.txt").as_ptr(),
                    0,
                    body.as_ptr(),
                    body.len()
                ),
                body.len() as i64
            );
            let mut output = [0_u8; 32];
            assert_eq!(
                greppy_workspace_read(
                    core,
                    c("ffi-test").as_ptr(),
                    c("private.txt").as_ptr(),
                    0,
                    output.as_mut_ptr(),
                    output.len()
                ),
                body.len() as i64
            );
            assert_eq!(&output[..body.len()], body);
            assert_eq!(greppy_workspace_remove(core, c("ffi-test").as_ptr()), 0);
            greppy_workspace_core_close(core);
        }
    }
}
