//! Parent-owned web-runtime attach capability.
//!
//! The long-lived Greppy agent parent generates the token and holds it in
//! process memory. Supervisor initialization and every authorized `greppy web`
//! child receive it exclusively via a child-local inherited FD 4 installed in
//! `pre_exec`. Never env, argv, a predictable file, or the parent's FD 4.

#[cfg(any(target_os = "linux", target_os = "android"))]
use std::io::Write;
use std::io::{self, Read};
use std::process::Command;
use std::sync::{Mutex, OnceLock};

pub const ATTACH_TOKEN_FD: i32 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Unset,
    PersistentParent,
    StandaloneOwner,
    InheritedChild,
}

struct AttachState {
    token: Option<String>,
    role: Role,
}

fn state() -> &'static Mutex<AttachState> {
    static STATE: OnceLock<Mutex<AttachState>> = OnceLock::new();
    STATE.get_or_init(|| {
        Mutex::new(AttachState {
            token: None,
            role: Role::Unset,
        })
    })
}

pub fn generate_attach_token() -> io::Result<String> {
    let mut bytes = [0_u8; 16];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
}

pub fn claim_persistent_parent() -> io::Result<String> {
    let mut guard = state().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(token) = guard.token.clone() {
        guard.role = Role::PersistentParent;
        return Ok(token);
    }
    let token = generate_attach_token()?;
    guard.token = Some(token.clone());
    guard.role = Role::PersistentParent;
    Ok(token)
}

/// Install a reconnect cookie so a later CLI process can talk to the same
/// owner without inheriting fd 4. Does not replace an already-held token.
pub fn adopt_persistent_token(token: String) {
    let mut guard = state().lock().unwrap_or_else(|e| e.into_inner());
    if guard.token.is_none() {
        guard.token = Some(token);
    }
    guard.role = Role::PersistentParent;
}

#[cfg(unix)]
pub fn current_token() -> Option<String> {
    let mut guard = state().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(token) = guard.token.clone() {
        return Some(token);
    }
    let inherited = take_inherited_attach_token()?;
    guard.token = Some(inherited.clone());
    if guard.role == Role::Unset {
        guard.role = Role::InheritedChild;
    }
    Some(inherited)
}

#[cfg(not(unix))]
pub fn current_token() -> Option<String> {
    state()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .token
        .clone()
}

#[allow(dead_code)]
pub fn become_standalone_owner() -> io::Result<String> {
    let mut guard = state().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(token) = guard.token.clone() {
        return Ok(token);
    }
    let token = generate_attach_token()?;
    guard.token = Some(token.clone());
    guard.role = Role::StandaloneOwner;
    Ok(token)
}

pub fn should_shutdown_on_scope_end() -> bool {
    let guard = state().lock().unwrap_or_else(|e| e.into_inner());
    guard.role == Role::StandaloneOwner
}

/// Keeps the CLOEXEC pipe end open in the parent until after `spawn`.
/// Drop closes that parent copy; the child already received FD 4 via pre_exec.
pub struct AttachTokenPass {
    parent_read_fd: i32,
}

impl Drop for AttachTokenPass {
    fn drop(&mut self) {
        if self.parent_read_fd >= 0 {
            unsafe {
                libc::close(self.parent_read_fd);
            }
            self.parent_read_fd = -1;
        }
    }
}

#[cfg(unix)]
fn open_atomic_cloexec_token_reader(token: &str) -> io::Result<i32> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        linux_pipe2_token_reader(token)
    }
    #[cfg(target_os = "macos")]
    {
        darwin_cloexec_token_reader(token)
    }
    #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "macos")))]
    {
        let _ = token;
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "atomic CLOEXEC attach fd is unimplemented on this OS",
        ))
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn linux_pipe2_token_reader(token: &str) -> io::Result<i32> {
    use std::os::fd::{FromRawFd, OwnedFd};
    let mut fds = [0; 2];
    if unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
        return Err(io::Error::last_os_error());
    }
    let mut write = std::fs::File::from(unsafe { OwnedFd::from_raw_fd(fds[1]) });
    if let Err(error) = write
        .write_all(token.as_bytes())
        .and_then(|_| write.write_all(b"\n"))
    {
        unsafe {
            libc::close(fds[0]);
        }
        return Err(error);
    }
    drop(write);
    Ok(fds[0])
}
#[cfg(target_os = "macos")]
fn write_all_fd(fd: i32, bytes: &[u8]) -> io::Result<()> {
    let mut offset = 0;
    while offset < bytes.len() {
        let wrote =
            unsafe { libc::write(fd, bytes.as_ptr().add(offset).cast(), bytes.len() - offset) };
        if wrote < 0 {
            let error = io::Error::last_os_error();
            if error.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(error);
        }
        if wrote == 0 {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "short attach token write",
            ));
        }
        offset += wrote as usize;
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn fd_is_cloexec(fd: i32) -> bool {
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
    flags >= 0 && (flags & libc::FD_CLOEXEC) != 0
}

#[cfg(target_os = "macos")]
fn darwin_cloexec_token_reader(token: &str) -> io::Result<i32> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let payload = format!("{token}\n");
    let payload = payload.as_bytes();
    let mut last_exist = None;
    for _ in 0..8 {
        let mut rnd = [0_u8; 8];
        std::fs::File::open("/dev/urandom")?.read_exact(&mut rnd)?;
        let dir = std::env::temp_dir().join(format!(
            "gwa-d-{}",
            rnd.iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        ));
        let cdir = CString::new(dir.as_os_str().as_bytes())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "attach cloexec dir"))?;
        if unsafe { libc::mkdir(cdir.as_ptr(), 0o700) } != 0 {
            let error = io::Error::last_os_error();
            if error.raw_os_error() == Some(libc::EEXIST) {
                last_exist = Some(error);
                continue;
            }
            return Err(error);
        }
        if unsafe { libc::chmod(cdir.as_ptr(), 0o700) } != 0 {
            let error = io::Error::last_os_error();
            unsafe {
                libc::rmdir(cdir.as_ptr());
            }
            return Err(error);
        }
        let path = dir.join("t");
        let cpath = CString::new(path.as_os_str().as_bytes())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "attach cloexec path"))?;
        let fd = unsafe {
            libc::open(
                cpath.as_ptr(),
                libc::O_RDWR | libc::O_CREAT | libc::O_EXCL | libc::O_CLOEXEC,
                0o600,
            )
        };
        if fd < 0 {
            let error = io::Error::last_os_error();
            unsafe {
                libc::rmdir(cdir.as_ptr());
            }
            if error.raw_os_error() == Some(libc::EEXIST) {
                last_exist = Some(error);
                continue;
            }
            return Err(error);
        }
        if !fd_is_cloexec(fd) {
            unsafe {
                libc::unlink(cpath.as_ptr());
                libc::close(fd);
                libc::rmdir(cdir.as_ptr());
            }
            return Err(io::Error::other(
                "open did not install FD_CLOEXEC atomically",
            ));
        }
        if unsafe { libc::unlink(cpath.as_ptr()) } != 0 {
            let error = io::Error::last_os_error();
            unsafe {
                libc::close(fd);
                libc::rmdir(cdir.as_ptr());
            }
            return Err(error);
        }
        if unsafe { libc::rmdir(cdir.as_ptr()) } != 0 {
            let error = io::Error::last_os_error();
            unsafe {
                libc::close(fd);
            }
            return Err(error);
        }
        if let Err(error) = write_all_fd(fd, payload) {
            unsafe {
                libc::close(fd);
            }
            return Err(error);
        }
        if unsafe { libc::lseek(fd, 0, libc::SEEK_SET) } < 0 {
            let error = io::Error::last_os_error();
            unsafe {
                libc::close(fd);
            }
            return Err(error);
        }
        return Ok(fd);
    }
    Err(last_exist.unwrap_or_else(|| io::Error::other("O_CLOEXEC attach path retries exhausted")))
}
#[cfg(unix)]
pub fn give_child_attach_token(command: &mut Command, token: &str) -> io::Result<AttachTokenPass> {
    use std::os::unix::process::CommandExt;
    let read_fd = open_atomic_cloexec_token_reader(token)?;
    // Child-local only: dup2/close/fcntl are async-signal-safe. The parent fd
    // is created atomically CLOEXEC, so concurrent unrelated spawns cannot
    // inherit the token. pre_exec callbacks accumulate; register this after
    // any sandbox pre_exec so FD 4 is installed last.
    unsafe {
        command.pre_exec(move || {
            if libc::dup2(read_fd, ATTACH_TOKEN_FD) < 0 {
                return Err(io::Error::last_os_error());
            }
            if read_fd != ATTACH_TOKEN_FD {
                libc::close(read_fd);
            }
            if libc::fcntl(ATTACH_TOKEN_FD, libc::F_SETFD, 0) < 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        });
    }
    Ok(AttachTokenPass {
        parent_read_fd: read_fd,
    })
}

#[cfg(not(unix))]
pub fn give_child_attach_token(
    _command: &mut Command,
    _token: &str,
) -> io::Result<AttachTokenPass> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "attach token fd inheritance requires Unix",
    ))
}

pub fn inherit_current_into(command: &mut Command) -> io::Result<AttachTokenPass> {
    let token = current_token().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::PermissionDenied,
            "no parent-owned attach token",
        )
    })?;
    give_child_attach_token(command, &token)
}

#[cfg(unix)]
pub fn take_inherited_attach_token() -> Option<String> {
    use std::os::fd::{FromRawFd, OwnedFd};
    if unsafe { libc::fcntl(ATTACH_TOKEN_FD, libc::F_GETFD) } < 0 {
        return None;
    }
    let mut file = std::fs::File::from(unsafe { OwnedFd::from_raw_fd(ATTACH_TOKEN_FD) });
    let mut buf = String::new();
    file.read_to_string(&mut buf).ok()?;
    let token = buf.trim().to_owned();
    if token.len() >= 16 {
        Some(token)
    } else {
        None
    }
}

#[cfg(not(unix))]
pub fn take_inherited_attach_token() -> Option<String> {
    None
}

pub fn inherit_attach_for_agent(command: &mut Command) -> io::Result<Box<dyn Send>> {
    inherit_current_into(command).map(|pass| Box::new(pass) as Box<dyn Send>)
}
