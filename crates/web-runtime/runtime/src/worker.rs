use crate::protocol::{read_message, write_message, Message, WorkerKind};
use std::ffi::OsString;
use std::io::{self, Read, Write};

pub const CAPABILITY_FD: i32 = 3;
/// Parent-issued client/supervisor attach token. Not derivable from the socket path.
pub const ATTACH_TOKEN_FD: i32 = 4;
/// Bidirectional framed protocol socket. stdin/stdout are a log channel, not the frame channel.
pub const PROTOCOL_FD: i32 = 5;

/// Take the inherited protocol socket. The original FD is owned by the writer;
/// the reader is a `dup` so both ends can be used independently.
#[cfg(unix)]
pub fn take_protocol_channel() -> io::Result<(std::fs::File, std::fs::File)> {
    use std::os::fd::{FromRawFd, OwnedFd};
    if unsafe { libc::fcntl(PROTOCOL_FD, libc::F_GETFD) } < 0 {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "missing inherited protocol FD",
        ));
    }
    let reader_fd = unsafe { libc::dup(PROTOCOL_FD) };
    if reader_fd < 0 {
        return Err(io::Error::last_os_error());
    }
    unsafe {
        libc::fcntl(reader_fd, libc::F_SETFD, libc::FD_CLOEXEC);
        libc::fcntl(PROTOCOL_FD, libc::F_SETFD, libc::FD_CLOEXEC);
    }
    let reader = std::fs::File::from(unsafe { OwnedFd::from_raw_fd(reader_fd) });
    let writer = std::fs::File::from(unsafe { OwnedFd::from_raw_fd(PROTOCOL_FD) });
    Ok((reader, writer))
}

#[cfg(not(unix))]
pub fn take_protocol_channel() -> io::Result<(std::fs::File, std::fs::File)> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "inherited protocol FD requires Unix",
    ))
}

#[cfg(unix)]
pub struct AttachTokenPass {
    parent_read_fd: i32,
}

#[cfg(unix)]
impl Drop for AttachTokenPass {
    fn drop(&mut self) {
        unsafe {
            libc::close(self.parent_read_fd);
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
pub fn give_child_attach_token(
    command: &mut std::process::Command,
    token: &str,
) -> io::Result<AttachTokenPass> {
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

#[cfg(unix)]
pub fn take_parent_attach_token() -> Option<String> {
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

/// Linux `fexecve` equivalent lookup of a pinned image FD.
///
/// Compiled on every OS so the Linux spawn contract stays typechecked on
/// macOS hosts. Darwin has no `fexecve`; do not exec this path on macOS and
/// claim a pinned-image spawn.
pub fn linux_proc_self_fd_path(fd: i32) -> String {
    format!("/proc/self/fd/{fd}")
}

/// Whether this OS executes the supervisor image via a pinned FD (`fexecve`).
pub fn linux_same_image_exec_is_fd_backed() -> bool {
    cfg!(any(target_os = "linux", target_os = "android"))
}

/// How the worker image is executed for same-image re-exec.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageExecKind {
    /// Linux: `fexecve` of an `O_CLOEXEC` image FD (no path re-lookup).
    Fexecve,
    /// macOS: `execve` of the `Command` program path (`current_exe()`).
    /// Residual TOCTOU remains; Darwin `fexecve` / codesign is EXTERNAL.
    Path,
}

/// Pin of the supervisor executable used for same-image worker re-exec.
///
/// On Linux the image is opened `O_RDONLY|O_CLOEXEC` and the child is started
/// with `fexecve` of that FD. Identity proof compares `/proc/<pid>/exe` to the
/// same FD's device/inode.
///
/// On macOS spawn stays path-based. This crate does not fake `fexecve`. An
/// in-place rewrite of the same inode after this pin and before exec completes
/// is residual TOCTOU; closing it needs a pinned FD (`fexecve` on Linux) or a
/// platform code-signing proof over the mapped pages (EXTERNAL).
#[cfg(unix)]
pub struct PinnedSupervisorImage {
    fd: std::os::fd::OwnedFd,
    path: std::path::PathBuf,
}

#[cfg(unix)]
fn open_cloexec_rdonly(path: &std::path::Path) -> io::Result<std::os::fd::OwnedFd> {
    use std::ffi::CString;
    use std::os::fd::{FromRawFd, OwnedFd};
    use std::os::unix::ffi::OsStrExt;
    let cpath = CString::new(path.as_os_str().as_bytes()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "supervisor image path contains NUL",
        )
    })?;
    let fd = unsafe { libc::open(cpath.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
    if flags < 0 || (flags & libc::FD_CLOEXEC) == 0 {
        unsafe {
            libc::close(fd);
        }
        return Err(io::Error::other(
            "open did not install FD_CLOEXEC atomically",
        ));
    }
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

#[cfg(unix)]
fn fd_dev_ino(fd: i32) -> io::Result<(u64, u64)> {
    let mut st = std::mem::MaybeUninit::<libc::stat>::uninit();
    if unsafe { libc::fstat(fd, st.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    let st = unsafe { st.assume_init() };
    Ok((st.st_dev as u64, st.st_ino as u64))
}

/// argv/envp for `fexecve`, built in the parent (pre_exec must not allocate).
#[cfg(unix)]
struct FexecveArgs {
    #[allow(dead_code)]
    argv: Vec<std::ffi::CString>,
    #[allow(dead_code)]
    env: Vec<std::ffi::CString>,
    // Stored as usize so the pre_exec closure is Send+Sync (`*const c_char` is not).
    argv_p: Vec<usize>,
    env_p: Vec<usize>,
}

#[cfg(unix)]
fn os_to_cstring(value: &std::ffi::OsStr) -> io::Result<std::ffi::CString> {
    use std::os::unix::ffi::OsStrExt;
    std::ffi::CString::new(value.as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "exec argument contains NUL"))
}

#[cfg(unix)]
impl FexecveArgs {
    fn from_command(command: &std::process::Command) -> io::Result<Self> {
        use std::os::unix::ffi::OsStrExt;
        let mut argv = Vec::new();
        argv.push(os_to_cstring(command.get_program())?);
        for arg in command.get_args() {
            argv.push(os_to_cstring(arg)?);
        }
        let mut env = Vec::new();
        for (key, value) in command.get_envs() {
            let Some(value) = value else {
                continue;
            };
            let mut pair = Vec::with_capacity(key.len() + value.len() + 1);
            pair.extend_from_slice(key.as_bytes());
            pair.push(b'=');
            pair.extend_from_slice(value.as_bytes());
            env.push(std::ffi::CString::new(pair).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "environment entry contains NUL",
                )
            })?);
        }
        let mut argv_p: Vec<usize> = argv.iter().map(|s| s.as_ptr() as usize).collect();
        argv_p.push(0);
        let mut env_p: Vec<usize> = env.iter().map(|s| s.as_ptr() as usize).collect();
        env_p.push(0);
        Ok(Self {
            argv,
            env,
            argv_p,
            env_p,
        })
    }
}

/// Pin the running supervisor image (`/proc/self/exe` on Linux, `current_exe`
/// on macOS). Non-Linux non-macOS Unix is Unsupported.
#[cfg(unix)]
pub fn pin_supervisor_image() -> io::Result<PinnedSupervisorImage> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        PinnedSupervisorImage::open_path(std::path::Path::new("/proc/self/exe"))
    }
    #[cfg(target_os = "macos")]
    {
        PinnedSupervisorImage::open_path(&std::env::current_exe()?)
    }
    #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "macos")))]
    {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "same-image re-exec is unimplemented on this Unix",
        ))
    }
}

/// Pin `current_exe` / `/proc/self/exe` and bind `command` so the child is
/// executed from that pin. Call after argv/env/stdio and after FD-installing
/// `pre_exec` hooks (capability FD 3, attach token FD 4, protocol FD 5): on Linux the last
/// `pre_exec` is `fexecve` of the pinned FD and does not return.
///
/// Capability secrets stay on inherited FD 3; this does not put tokens in argv.
#[cfg(unix)]
pub fn apply_same_image_reexec(
    command: &mut std::process::Command,
) -> io::Result<PinnedSupervisorImage> {
    let image = pin_supervisor_image()?;
    image.bind_command(command)?;
    Ok(image)
}

#[cfg(unix)]
impl PinnedSupervisorImage {
    pub fn open_path(path: &std::path::Path) -> io::Result<Self> {
        let fd = open_cloexec_rdonly(path)?;
        Ok(Self {
            fd,
            path: path.to_owned(),
        })
    }

    pub fn exec_kind(&self) -> ImageExecKind {
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            ImageExecKind::Fexecve
        }
        #[cfg(not(any(target_os = "linux", target_os = "android")))]
        {
            ImageExecKind::Path
        }
    }

    pub fn is_fd_backed(&self) -> bool {
        linux_same_image_exec_is_fd_backed()
    }

    pub fn as_raw_fd(&self) -> i32 {
        std::os::fd::AsRawFd::as_raw_fd(&self.fd)
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    pub fn dev_ino(&self) -> io::Result<(u64, u64)> {
        fd_dev_ino(self.as_raw_fd())
    }

    /// Bind `command` so spawn executes this pin.
    ///
    /// Linux: last `pre_exec` is `fexecve` of a dup of this FD. Must run after
    /// capability/attach/protocol `pre_exec` so FD 3/4/5 are installed first.
    ///
    /// macOS: no-op. Spawn stays path-based (`Command::new` path). Residual
    /// TOCTOU is documented, not claimed closed. Darwin `fexecve` is EXTERNAL.
    pub fn bind_command(&self, command: &mut std::process::Command) -> io::Result<()> {
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            self.bind_linux_fexecve(command)
        }
        #[cfg(target_os = "macos")]
        {
            let _ = command;
            Ok(())
        }
        #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "macos")))]
        {
            let _ = command;
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "same-image re-exec is unimplemented on this Unix",
            ))
        }
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn bind_linux_fexecve(&self, command: &mut std::process::Command) -> io::Result<()> {
        use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
        use std::os::unix::process::CommandExt;
        let args = FexecveArgs::from_command(command)?;
        let raw = unsafe { libc::fcntl(self.as_raw_fd(), libc::F_DUPFD_CLOEXEC, 0) };
        if raw < 0 {
            return Err(io::Error::last_os_error());
        }
        let exec_fd = unsafe { OwnedFd::from_raw_fd(raw) };
        // Child-local only: fexecve is async-signal-safe. argv/envp and the
        // dup FD are built in the parent. pre_exec must not allocate. This
        // callback must be last: it does not return on success.
        unsafe {
            command.pre_exec(move || {
                libc::fexecve(
                    exec_fd.as_raw_fd(),
                    args.argv_p.as_ptr() as *const *const libc::c_char,
                    args.env_p.as_ptr() as *const *const libc::c_char,
                );
                Err(io::Error::last_os_error())
            });
        }
        Ok(())
    }

    /// Prove the child's image matches this pin's FD/inode. Fail-closed.
    pub fn prove_child(&self, pid: u32) -> io::Result<()> {
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            self.prove_linux(pid)
        }
        #[cfg(target_os = "macos")]
        {
            self.prove_macos(pid)
        }
        #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "macos")))]
        {
            let _ = pid;
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "same-image re-exec identity proof is unimplemented on this Unix",
            ))
        }
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn prove_linux(&self, pid: u32) -> io::Result<()> {
        use std::os::fd::AsRawFd;
        let child = open_cloexec_rdonly(std::path::Path::new(&format!("/proc/{pid}/exe")))?;
        let expected = fd_dev_ino(self.as_raw_fd())?;
        let observed = fd_dev_ino(child.as_raw_fd())?;
        if expected != observed {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "worker pid {pid} image identity mismatch: child {}/{} != pinned FD {}/{}",
                    observed.0, observed.1, expected.0, expected.1
                ),
            ));
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn prove_macos(&self, pid: u32) -> io::Result<()> {
        // Path-based identity. Residual TOCTOU: an in-place rewrite of the
        // same inode after this pin and before exec completes is not closed.
        // Darwin fexecve / codesign over mapped pages is EXTERNAL.
        use std::os::fd::AsRawFd;
        use std::path::PathBuf;
        let mut buf = vec![0u8; 4096];
        let n = unsafe {
            libc::proc_pidpath(
                pid as i32,
                buf.as_mut_ptr() as *mut libc::c_void,
                buf.len() as u32,
            )
        };
        if n <= 0 {
            return Err(io::Error::other(format!(
                "proc_pidpath({pid}) failed: {}",
                io::Error::last_os_error()
            )));
        }
        buf.truncate(n as usize);
        let path = PathBuf::from(String::from_utf8(buf).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "worker executable path is not UTF-8",
            )
        })?);
        let child = open_cloexec_rdonly(&path)?;
        let expected = fd_dev_ino(self.as_raw_fd())?;
        let observed = fd_dev_ino(child.as_raw_fd())?;
        if expected != observed {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "worker pid {pid} image identity mismatch: child {}/{} != pinned FD {}/{}",
                    observed.0, observed.1, expected.0, expected.1
                ),
            ));
        }
        Ok(())
    }

    /// Prove identity; on mismatch SIGKILL the child (fail-closed).
    pub fn prove_child_or_kill(&self, child: &mut std::process::Child) -> io::Result<()> {
        if let Err(error) = self.prove_child(child.id()) {
            let pid = child.id() as i32;
            unsafe {
                libc::kill(pid, libc::SIGKILL);
            }
            let _ = child.kill();
            let _ = child.wait();
            return Err(error);
        }
        Ok(())
    }
}

pub fn require_worker_auth(args: impl IntoIterator<Item = OsString>) -> io::Result<String> {
    reject_capability_argv(args)?;
    require_inherited_capability()
}

pub fn reject_capability_argv(args: impl IntoIterator<Item = OsString>) -> io::Result<()> {
    let mut args = args.into_iter();
    while let Some(argument) = args.next() {
        match argument.to_str() {
            Some("--capability") => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "capability secrets must not be passed in argv; use the inherited capability FD",
                ));
            }
            Some("--internal-role") => {
                let value = args.next().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "missing value after --internal-role",
                    )
                })?;
                match value.to_str() {
                    Some("controller") | Some("content") | Some("supervisor") => {}
                    _ => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("unknown --internal-role {value:?}"),
                        ));
                    }
                }
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unknown worker argument {argument:?}"),
                ));
            }
        }
    }
    Ok(())
}

pub fn require_inherited_capability() -> io::Result<String> {
    #[cfg(unix)]
    {
        use std::fs::File;
        use std::os::fd::{FromRawFd, OwnedFd};
        if unsafe { libc::fcntl(CAPABILITY_FD, libc::F_GETFD) } < 0 {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "missing inherited capability FD",
            ));
        }
        let mut file = File::from(unsafe { OwnedFd::from_raw_fd(CAPABILITY_FD) });
        let mut token = String::new();
        file.read_to_string(&mut token)?;
        let token = token.trim();
        if token.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "empty capability from inherited FD",
            ));
        }
        Ok(token.to_owned())
    }
    #[cfg(not(unix))]
    {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "inherited capability FD requires Unix",
        ))
    }
}

pub fn run_worker<R, W, T>(
    worker: WorkerKind,
    capability: &str,
    runtime: T,
    reader: &mut R,
    writer: &mut W,
) -> io::Result<()>
where
    R: Read,
    W: Write,
{
    match read_message(reader)? {
        Message::Hello {
            worker: requested,
            capability: hello_capability,
            ..
        } if requested == worker && hello_capability == capability => {}
        unexpected => return Err(unexpected_message("Hello", worker, unexpected)),
    }

    write_message(writer, &Message::ready(worker))?;

    match read_message(reader)? {
        Message::Shutdown { .. } => {}
        unexpected => return Err(unexpected_message("Shutdown", worker, unexpected)),
    }

    drop(runtime);
    write_message(writer, &Message::shutdown_ack(worker))
}

fn unexpected_message(expected: &str, worker: WorkerKind, actual: Message) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("{worker:?} worker expected {expected}, received {actual:?}"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::io::Cursor;
    use std::rc::Rc;

    struct DropMarker(Rc<Cell<bool>>);

    impl Drop for DropMarker {
        fn drop(&mut self) {
            self.0.set(true);
        }
    }

    #[test]
    fn acknowledges_shutdown_only_after_runtime_is_dropped() {
        let dropped = Rc::new(Cell::new(false));
        let mut input = Vec::new();
        write_message(
            &mut input,
            &Message::hello(WorkerKind::Controller, "test-token"),
        )
        .unwrap();
        write_message(&mut input, &Message::shutdown()).unwrap();
        let mut output = Vec::new();

        run_worker(
            WorkerKind::Controller,
            "test-token",
            DropMarker(Rc::clone(&dropped)),
            &mut Cursor::new(input),
            &mut output,
        )
        .unwrap();

        assert!(dropped.get());
        let mut output = Cursor::new(output);
        assert_eq!(
            read_message(&mut output).unwrap(),
            Message::ready(WorkerKind::Controller)
        );
        assert_eq!(
            read_message(&mut output).unwrap(),
            Message::shutdown_ack(WorkerKind::Controller)
        );
    }

    #[test]
    fn rejects_capability_secret_in_argv() {
        let error = reject_capability_argv([
            std::ffi::OsString::from("--capability"),
            std::ffi::OsString::from("secret-token"),
        ])
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("argv"), "{}", error);
    }

    #[test]
    fn accepts_internal_role_without_argv_capability() {
        reject_capability_argv([
            std::ffi::OsString::from("--internal-role"),
            std::ffi::OsString::from("controller"),
        ])
        .unwrap();
    }

    #[test]
    fn rejects_a_hello_for_the_other_worker() {
        let mut input = Vec::new();
        write_message(
            &mut input,
            &Message::hello(WorkerKind::Content, "test-token"),
        )
        .unwrap();

        let error = run_worker(
            WorkerKind::Controller,
            "test-token",
            (),
            &mut Cursor::new(input),
            &mut Vec::new(),
        )
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn rejects_hello_with_mismatched_capability() {
        let mut input = Vec::new();
        write_message(
            &mut input,
            &Message::hello(WorkerKind::Controller, "other-token"),
        )
        .unwrap();

        let error = run_worker(
            WorkerKind::Controller,
            "test-token",
            (),
            &mut Cursor::new(input),
            &mut Vec::new(),
        )
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    #[cfg(unix)]
    #[test]
    fn concurrent_unrelated_children_do_not_inherit_attach_token_fd() {
        use std::process::{Command, Stdio};
        use std::thread;
        let mut rnd = [0_u8; 16];
        std::fs::File::open("/dev/urandom")
            .and_then(|mut file| {
                use std::io::Read;
                file.read_exact(&mut rnd)
            })
            .expect("urandom");
        let token: String = rnd.iter().map(|byte| format!("{byte:02x}")).collect();
        let mut holder = Command::new("/usr/bin/true");
        holder
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        let pass = super::give_child_attach_token(&mut holder, &token).expect("cloexec fd");
        let mut joins = Vec::new();
        for _ in 0..16 {
            let token = token.clone();
            joins.push(thread::spawn(move || {
                Command::new("/usr/bin/perl")
                    .arg("-e")
                    .arg(
                        r#"
use Fcntl;
my $token = $ARGV[0];
for my $fd (3 .. 64) {
    my $flags = fcntl($fd, F_GETFD, 0);
    next unless defined $flags;
    if ($fd == 4) { print "FD4_OPEN\n"; exit 3; }
    my $buf = "";
    sysseek($fd, 0, 0);
    sysread($fd, $buf, 256);
    if (index($buf, $token) >= 0) { print "TOKEN_ON_FD_${fd}\n"; exit 2; }
}
exit 0;
"#,
                    )
                    .arg(token)
                    .stdin(Stdio::null())
                    .output()
                    .expect("unrelated")
            }));
        }
        for join in joins {
            let output = join.join().expect("thread");
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert_eq!(
                output.status.code(),
                Some(0),
                "unrelated child inherited attach fd/token stdout={stdout} stderr={stderr}"
            );
        }
        let mut authorized = Command::new("/usr/bin/perl");
        authorized
            .arg("-e")
            .arg("my $buf=''; open(F,q{<&=},4) or die $!; sysread(F,$b,256); print $b;")
            .stdin(Stdio::null());
        let auth = super::give_child_attach_token(&mut authorized, &token).expect("auth fd");
        let output = authorized.output().expect("authorized");
        drop(auth);
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains(&token),
            "authorized child must see token on fd 4, stdout={stdout:?}"
        );
        drop(pass);
    }

    #[test]
    fn linux_fd_backed_exec_contract() {
        assert_eq!(linux_proc_self_fd_path(3), "/proc/self/fd/3");
        assert_eq!(linux_proc_self_fd_path(7), "/proc/self/fd/7");
        assert_eq!(
            linux_same_image_exec_is_fd_backed(),
            cfg!(any(target_os = "linux", target_os = "android"))
        );
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn unix_bin(name: &str) -> std::path::PathBuf {
        for candidate in [format!("/bin/{name}"), format!("/usr/bin/{name}")] {
            let path = std::path::PathBuf::from(&candidate);
            if path.is_file() {
                return path;
            }
        }
        panic!("missing {name}");
    }

    #[cfg(unix)]
    #[test]
    fn fexecve_args_omit_capability_secret() {
        use std::process::Command;
        let mut command = Command::new("/usr/bin/true");
        command
            .arg("--internal-role")
            .arg("controller")
            .env_clear()
            .env("PATH", "/usr/bin:/bin");
        let args = super::FexecveArgs::from_command(&command).unwrap();
        let argv: Vec<&str> = args.argv.iter().map(|s| s.to_str().unwrap()).collect();
        assert_eq!(argv[0], "/usr/bin/true");
        assert!(argv.contains(&"--internal-role"));
        assert!(argv.iter().all(|arg| *arg != "--capability"));
        assert!(!argv.iter().any(|arg| arg.contains("secret")));
        assert_eq!(args.argv_p.last().copied(), Some(0));
        assert_eq!(args.env_p.last().copied(), Some(0));
        let env: Vec<&str> = args.env.iter().map(|s| s.to_str().unwrap()).collect();
        assert!(env.iter().any(|entry| *entry == "PATH=/usr/bin:/bin"));
        assert!(!env.iter().any(|entry| entry.contains("secret")));
    }

    #[cfg(unix)]
    #[test]
    fn pin_supervisor_image_fd_is_cloexec() {
        let image = pin_supervisor_image().unwrap();
        let flags = unsafe { libc::fcntl(image.as_raw_fd(), libc::F_GETFD) };
        assert!(
            flags >= 0 && (flags & libc::FD_CLOEXEC) != 0,
            "supervisor image FD must be O_CLOEXEC, flags={flags}"
        );
        assert_eq!(
            linux_proc_self_fd_path(image.as_raw_fd()),
            format!("/proc/self/fd/{}", image.as_raw_fd())
        );
        assert_eq!(image.is_fd_backed(), linux_same_image_exec_is_fd_backed());
        image.prove_child(std::process::id()).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn same_image_bind_does_not_put_capability_on_argv() {
        use std::process::Command;
        let mut command = Command::new("/usr/bin/true");
        command.arg("--internal-role").arg("controller").env_clear();
        let image = apply_same_image_reexec(&mut command).unwrap();
        let args: Vec<String> = command
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        assert!(
            !args
                .iter()
                .any(|arg| arg == "--capability" || arg.contains("secret")),
            "capability secrets must stay on FD 3, argv={args:?}"
        );
        assert_eq!(
            image.exec_kind() == ImageExecKind::Fexecve,
            image.is_fd_backed()
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_same_image_spawn_stays_path_based() {
        use std::process::Command;
        let image = pin_supervisor_image().unwrap();
        assert_eq!(image.exec_kind(), ImageExecKind::Path);
        assert!(!image.is_fd_backed());
        let mut command = Command::new("/usr/bin/true");
        image.bind_command(&mut command).unwrap();
        assert_eq!(command.get_program(), "/usr/bin/true");
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn linux_spawn_uses_fd_backed_exec() {
        use std::process::{Command, Stdio};
        let sleep = unix_bin("sleep");
        let true_bin = unix_bin("true");
        let root = std::env::temp_dir().join(format!(
            "greppy-fexecve-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let probe = root.join("probe");
        std::fs::copy(&sleep, &probe).unwrap();
        let image = PinnedSupervisorImage::open_path(&probe).unwrap();
        assert_eq!(image.exec_kind(), ImageExecKind::Fexecve);
        assert!(image.is_fd_backed());
        // Replace the directory entry with a different inode. Copying directly
        // to `probe` would truncate and rewrite the already pinned inode, which
        // is not a path-lookup TOCTOU and necessarily changes what the FD sees.
        let replacement = root.join("replacement");
        std::fs::copy(&true_bin, &replacement).unwrap();
        std::fs::rename(&replacement, &probe).unwrap();
        let mut command = Command::new(&probe);
        command
            .arg("30")
            .env_clear()
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        image.bind_command(&mut command).unwrap();
        let mut child = command.spawn().expect("fd-backed spawn");
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(
            child.try_wait().unwrap().is_none(),
            "fexecve must run the pinned sleep image, not the replaced true path"
        );
        image.prove_child(child.id()).unwrap();
        let _ = child.kill();
        let _ = child.wait();
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn linux_identity_mismatch_kills_child() {
        use std::process::{Command, Stdio};
        let sleep = unix_bin("sleep");
        let true_bin = unix_bin("true");
        let sleep_img = PinnedSupervisorImage::open_path(&sleep).unwrap();
        let true_img = PinnedSupervisorImage::open_path(&true_bin).unwrap();
        assert_ne!(
            sleep_img.dev_ino().unwrap(),
            true_img.dev_ino().unwrap(),
            "sleep and true must be distinct inodes"
        );
        let mut command = Command::new(&sleep);
        command
            .arg("30")
            .env_clear()
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        sleep_img.bind_command(&mut command).unwrap();
        let mut child = command.spawn().expect("sleep spawn");
        let error = true_img.prove_child_or_kill(&mut child).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("identity mismatch"), "{error}");
        let status = child.wait().unwrap();
        assert!(!status.success(), "mismatch must kill the child");
    }
}
