//! Linux worker OS sandbox: Landlock filesystem + a tight seccomp filter.
//!
//! This module is **not** the macOS SBPL profile in `supervisor.rs`. It is
//! Linux-only. The two profiles are equivalent in spirit (deny-default FS,
//! write confined to `tmp`) but are different kernels and different ABIs.
//!
//! # ABI — `apply(exe, tmp)`
//!
//! Called in the worker process after exec and before engine/JS startup
//! (wired from `supervisor::apply_worker_sandbox`).
//!
//! * **`exe`:** worker image. Granted read + execute. Its parent directory is
//!   granted read + execute (dlopen of sibling libs) unless that parent is
//!   `/`, which is never granted.
//! * **`tmp`:** the only writable directory tree. Must exist. Must not be `/`.
//! * **`Ok(())`:** Landlock `restrict_self` **and** the seccomp filter are
//!   installed. The calling thread (and future children) are sandboxed.
//! * **`Err(_)`:** refused. Includes Landlock `ENOSYS` / `EOPNOTSUPP` (old
//!   kernel, LSM off, missing `CONFIG_SECURITY_LANDLOCK`). This function
//!   never returns `Ok(())` meaning unsandboxed.
//! * **Non-Linux builds:** `ErrorKind::Unsupported` stub. No syscalls.
//!
//! ## Filesystem (Landlock, deny-default)
//!
//! Handled FS rights are all ABI-1 bits, plus `REFER` / `TRUNCATE` /
//! `IOCTL_DEV` when the probed ABI provides them. Anything not listed is
//! denied, including `$HOME` and the process cwd / workspace root (unless
//! that path **is** `tmp`).
//!
//! | path | access |
//! |------|--------|
//! | `exe` | read + execute |
//! | `exe` parent | read + execute (not `/`) |
//! | `tmp` | read + write (no device-node creation) |
//! | `/usr`, `/lib`, `/lib64`, `/etc` | read + execute |
//! | `/proc/self` | read + execute (`process-info*` equivalent) |
//! | `/dev/urandom` | read |
//! | `/dev/null`, `/dev/zero` | read + write (device nodes, not a write tree) |
//!
//! Missing optional trees (`/lib64` on aarch64, …) are skipped. Missing
//! `exe`, `tmp`, `/dev/null`, or `/dev/urandom` is an error.
//!
//! ## Seccomp
//!
//! Deny-list (not a syscall allow-list: Deno/Servo need the POSIX surface).
//! Kills the process on ptrace, mount/pivot/chroot, module load, bpf,
//! userfaultfd, keyring, mount API, `unshare`/`setns`, and `clone(CLONE_NEWUSER)`.
//! `clone3` returns `ENOSYS` so libc falls back to inspectable `clone`.

use std::io;
use std::path::{Path, PathBuf};

/// Install the worker OS sandbox on the calling thread.
///
/// See the module docs for the ABI. Fail-closed: `Ok(())` means Landlock
/// (and on Linux, seccomp) are enforced.
pub fn apply(exe: &Path, tmp: &Path) -> io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        linux::apply(exe, tmp)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (exe, tmp);
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "linux_sandbox::apply is Linux-only; refusing to start unsandboxed",
        ))
    }
}

/// Kind of Landlock path-beneath grant. `WriteTree` is the only directory
/// write; device nodes are not a writable tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GrantKind {
    ReadTree,
    ReadFile { execute: bool },
    DeviceRw,
    WriteTree,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FsPathGrant {
    path: PathBuf,
    kind: GrantKind,
    /// When true, `apply` fails closed if the path cannot be opened.
    required: bool,
}

impl FsPathGrant {
    fn read_tree(path: impl Into<PathBuf>, required: bool) -> Self {
        Self {
            path: path.into(),
            kind: GrantKind::ReadTree,
            required,
        }
    }

    fn read_file(path: impl Into<PathBuf>, execute: bool, required: bool) -> Self {
        Self {
            path: path.into(),
            kind: GrantKind::ReadFile { execute },
            required,
        }
    }

    fn device_rw(path: impl Into<PathBuf>, required: bool) -> Self {
        Self {
            path: path.into(),
            kind: GrantKind::DeviceRw,
            required,
        }
    }

    fn write_tree(path: impl Into<PathBuf>, required: bool) -> Self {
        Self {
            path: path.into(),
            kind: GrantKind::WriteTree,
            required,
        }
    }

    fn is_write_tree(&self) -> bool {
        matches!(self.kind, GrantKind::WriteTree)
    }
}

/// Policy allow-list. Pure path construction; does not talk to the kernel.
///
/// Optional system trees are included even if they are absent on this host
/// (`apply` skips `ENOENT` when opening). `exe` / `tmp` are canonicalized
/// when they exist so `..` and symlinks cannot sneak past the checks.
fn fs_allow_list(exe: &Path, tmp: &Path) -> io::Result<Vec<FsPathGrant>> {
    if exe.as_os_str().is_empty() || tmp.as_os_str().is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "worker sandbox: exe and tmp paths must be non-empty",
        ));
    }

    let exe = normalize(exe);
    let tmp = normalize(tmp);
    refuse_filesystem_root(&exe, "exe")?;
    refuse_filesystem_root(&tmp, "tmp")?;

    let mut grants = vec![
        FsPathGrant::read_tree("/usr", false),
        FsPathGrant::read_tree("/lib", false),
        FsPathGrant::read_tree("/lib64", false),
        FsPathGrant::read_tree("/etc", false),
        FsPathGrant::read_tree("/proc/self", false),
        FsPathGrant::read_file("/dev/urandom", false, true),
        FsPathGrant::device_rw("/dev/null", true),
        FsPathGrant::device_rw("/dev/zero", false),
        FsPathGrant::read_file(exe.clone(), true, true),
        FsPathGrant::write_tree(tmp.clone(), true),
    ];

    if let Some(dir) = exe.parent() {
        if !is_filesystem_root(dir) {
            grants.push(FsPathGrant::read_tree(dir, false));
        }
    }

    Ok(grants)
}

fn normalize(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

fn is_filesystem_root(path: &Path) -> bool {
    path == Path::new("/")
}

fn refuse_filesystem_root(path: &Path, what: &str) -> io::Result<()> {
    if is_filesystem_root(path) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("worker sandbox: refusing to grant filesystem root as {what}"),
        ));
    }
    Ok(())
}

#[cfg(target_os = "linux")]
mod linux {
    use super::{fs_allow_list, GrantKind};
    use std::ffi::CString;
    use std::io;
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
    use std::os::unix::ffi::OsStrExt;
    use std::path::Path;

    const LANDLOCK_CREATE_RULESET_VERSION: u32 = 1 << 0;
    const LANDLOCK_RULE_PATH_BENEATH: u32 = 1;

    const ACCESS_FS_EXECUTE: u64 = 1 << 0;
    const ACCESS_FS_WRITE_FILE: u64 = 1 << 1;
    const ACCESS_FS_READ_FILE: u64 = 1 << 2;
    const ACCESS_FS_READ_DIR: u64 = 1 << 3;
    const ACCESS_FS_REMOVE_DIR: u64 = 1 << 4;
    const ACCESS_FS_REMOVE_FILE: u64 = 1 << 5;
    const ACCESS_FS_MAKE_CHAR: u64 = 1 << 6;
    const ACCESS_FS_MAKE_DIR: u64 = 1 << 7;
    const ACCESS_FS_MAKE_REG: u64 = 1 << 8;
    const ACCESS_FS_MAKE_SOCK: u64 = 1 << 9;
    const ACCESS_FS_MAKE_FIFO: u64 = 1 << 10;
    const ACCESS_FS_MAKE_BLOCK: u64 = 1 << 11;
    const ACCESS_FS_MAKE_SYM: u64 = 1 << 12;
    const ACCESS_FS_REFER: u64 = 1 << 13;
    const ACCESS_FS_TRUNCATE: u64 = 1 << 14;
    const ACCESS_FS_IOCTL_DEV: u64 = 1 << 15;

    const FS_ABI1: u64 = ACCESS_FS_EXECUTE
        | ACCESS_FS_WRITE_FILE
        | ACCESS_FS_READ_FILE
        | ACCESS_FS_READ_DIR
        | ACCESS_FS_REMOVE_DIR
        | ACCESS_FS_REMOVE_FILE
        | ACCESS_FS_MAKE_CHAR
        | ACCESS_FS_MAKE_DIR
        | ACCESS_FS_MAKE_REG
        | ACCESS_FS_MAKE_SOCK
        | ACCESS_FS_MAKE_FIFO
        | ACCESS_FS_MAKE_BLOCK
        | ACCESS_FS_MAKE_SYM;
    const FS_ABI2: u64 = FS_ABI1 | ACCESS_FS_REFER;
    const FS_ABI3: u64 = FS_ABI2 | ACCESS_FS_TRUNCATE;
    const FS_ABI5: u64 = FS_ABI3 | ACCESS_FS_IOCTL_DEV;

    const PR_SET_NO_NEW_PRIVS: libc::c_int = 38;

    #[repr(C)]
    struct LandlockRulesetAttr {
        handled_access_fs: u64,
    }

    #[repr(C, packed)]
    struct LandlockPathBeneathAttr {
        allowed_access: u64,
        parent_fd: i32,
    }

    pub(super) fn apply(exe: &Path, tmp: &Path) -> io::Result<()> {
        restrict_filesystem(exe, tmp)?;
        apply_seccomp()?;
        Ok(())
    }

    /// Landlock only. Used by tests so a live restrict does not install a
    /// process-killing seccomp filter on the cargo-test thread.
    pub(super) fn restrict_filesystem(exe: &Path, tmp: &Path) -> io::Result<()> {
        if !tmp.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "worker sandbox: tmp is not an existing directory: {}",
                    tmp.display()
                ),
            ));
        }

        let abi = probe_abi()?;
        let handled = handled_fs(abi);
        let grants = fs_allow_list(exe, tmp)?;
        let ruleset = create_ruleset(handled)?;

        let mut added = 0usize;
        for grant in &grants {
            if add_path_rule(ruleset.as_raw_fd(), handled, grant)? {
                added += 1;
            }
        }
        if added == 0 {
            return Err(io::Error::other(
                "worker sandbox: landlock ruleset is empty; refusing to start unsandboxed",
            ));
        }

        set_no_new_privs()?;
        restrict_self(ruleset.as_raw_fd())?;
        Ok(())
    }

    fn probe_abi() -> io::Result<i32> {
        let rc = unsafe {
            libc::syscall(
                libc::SYS_landlock_create_ruleset,
                std::ptr::null::<LandlockRulesetAttr>(),
                0usize,
                LANDLOCK_CREATE_RULESET_VERSION,
            )
        };
        if rc < 0 {
            return Err(landlock_unavailable("landlock_create_ruleset(VERSION)"));
        }
        if rc < 1 {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!(
                    "worker sandbox: landlock ABI {rc} is unusable; refusing to start unsandboxed"
                ),
            ));
        }
        Ok(rc as i32)
    }

    fn handled_fs(abi: i32) -> u64 {
        if abi >= 5 {
            FS_ABI5
        } else if abi >= 3 {
            FS_ABI3
        } else if abi >= 2 {
            FS_ABI2
        } else {
            FS_ABI1
        }
    }

    fn create_ruleset(handled_access_fs: u64) -> io::Result<OwnedFd> {
        let attr = LandlockRulesetAttr { handled_access_fs };
        let rc = unsafe {
            libc::syscall(
                libc::SYS_landlock_create_ruleset,
                &attr as *const LandlockRulesetAttr,
                std::mem::size_of::<u64>(),
                0u32,
            )
        };
        if rc < 0 {
            return Err(landlock_unavailable("landlock_create_ruleset"));
        }
        Ok(unsafe { OwnedFd::from_raw_fd(rc as i32) })
    }

    /// `Ok(true)` if a rule was added; `Ok(false)` if an optional path is absent.
    fn add_path_rule(
        ruleset_fd: i32,
        handled: u64,
        grant: &super::FsPathGrant,
    ) -> io::Result<bool> {
        let access = access_for(handled, grant.kind);
        if access == 0 {
            return Err(io::Error::other(
                "worker sandbox: landlock access mask collapsed to empty",
            ));
        }
        let c_path = path_c_string(&grant.path)?;
        let mut flags = libc::O_PATH | libc::O_CLOEXEC;
        if matches!(grant.kind, GrantKind::ReadTree | GrantKind::WriteTree) {
            flags |= libc::O_DIRECTORY;
        }
        let raw = unsafe { libc::open(c_path.as_ptr(), flags) };
        if raw < 0 {
            let err = io::Error::last_os_error();
            if !grant.required && is_absent_path_error(&err) {
                return Ok(false);
            }
            return Err(io::Error::new(
                err.kind(),
                format!("worker sandbox: open({}): {err}", grant.path.display()),
            ));
        }
        let fd = unsafe { OwnedFd::from_raw_fd(raw) };
        let attr = LandlockPathBeneathAttr {
            allowed_access: access,
            parent_fd: fd.as_raw_fd(),
        };
        let rc = unsafe {
            libc::syscall(
                libc::SYS_landlock_add_rule,
                ruleset_fd,
                LANDLOCK_RULE_PATH_BENEATH,
                &attr as *const LandlockPathBeneathAttr,
                0u32,
            )
        };
        if rc < 0 {
            return Err(sandbox_os(&format!(
                "landlock_add_rule({})",
                grant.path.display()
            )));
        }
        Ok(true)
    }

    fn access_for(handled: u64, kind: GrantKind) -> u64 {
        let want = match kind {
            GrantKind::ReadTree => ACCESS_FS_EXECUTE | ACCESS_FS_READ_FILE | ACCESS_FS_READ_DIR,
            GrantKind::ReadFile { execute } => {
                let mut bits = ACCESS_FS_READ_FILE | ACCESS_FS_IOCTL_DEV;
                if execute {
                    bits |= ACCESS_FS_EXECUTE;
                }
                bits
            }
            GrantKind::DeviceRw => {
                ACCESS_FS_READ_FILE
                    | ACCESS_FS_WRITE_FILE
                    | ACCESS_FS_TRUNCATE
                    | ACCESS_FS_IOCTL_DEV
            }
            GrantKind::WriteTree => {
                ACCESS_FS_EXECUTE
                    | ACCESS_FS_WRITE_FILE
                    | ACCESS_FS_READ_FILE
                    | ACCESS_FS_READ_DIR
                    | ACCESS_FS_REMOVE_DIR
                    | ACCESS_FS_REMOVE_FILE
                    | ACCESS_FS_MAKE_DIR
                    | ACCESS_FS_MAKE_REG
                    | ACCESS_FS_MAKE_SOCK
                    | ACCESS_FS_MAKE_FIFO
                    | ACCESS_FS_MAKE_SYM
                    | ACCESS_FS_REFER
                    | ACCESS_FS_TRUNCATE
            }
        };
        want & handled
    }

    fn is_absent_path_error(err: &io::Error) -> bool {
        matches!(
            err.raw_os_error(),
            Some(libc::ENOENT) | Some(libc::ENOTDIR) | Some(libc::ELOOP)
        )
    }

    fn set_no_new_privs() -> io::Result<()> {
        let rc = unsafe { libc::prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) };
        if rc != 0 {
            return Err(sandbox_os("prctl(PR_SET_NO_NEW_PRIVS)"));
        }
        Ok(())
    }

    fn restrict_self(ruleset_fd: i32) -> io::Result<()> {
        let rc = unsafe { libc::syscall(libc::SYS_landlock_restrict_self, ruleset_fd, 0u32) };
        if rc < 0 {
            return Err(landlock_unavailable("landlock_restrict_self"));
        }
        Ok(())
    }

    fn path_c_string(path: &Path) -> io::Result<CString> {
        CString::new(path.as_os_str().as_bytes()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("worker sandbox: path contains NUL: {}", path.display()),
            )
        })
    }

    fn landlock_unavailable(op: &str) -> io::Error {
        let err = io::Error::last_os_error();
        match err.raw_os_error() {
            Some(libc::ENOSYS) => io::Error::new(
                io::ErrorKind::Unsupported,
                format!(
                    "worker sandbox: {op}: landlock unavailable (ENOSYS); refusing to start unsandboxed"
                ),
            ),
            Some(libc::EOPNOTSUPP) => io::Error::new(
                io::ErrorKind::Unsupported,
                format!(
                    "worker sandbox: {op}: landlock disabled (EOPNOTSUPP); refusing to start unsandboxed"
                ),
            ),
            _ => io::Error::new(err.kind(), format!("worker sandbox: {op}: {err}")),
        }
    }

    fn sandbox_os(op: &str) -> io::Error {
        let err = io::Error::last_os_error();
        io::Error::new(err.kind(), format!("worker sandbox: {op}: {err}"))
    }

    // ── seccomp ──────────────────────────────────────────────────────────

    const AUDIT_ARCH_X86_64: u32 = 0xC000_003E;
    const AUDIT_ARCH_AARCH64: u32 = 0xC000_00B7;
    const AUDIT_ARCH_RISCV64: u32 = 0xC000_00F3;

    fn audit_arch() -> io::Result<u32> {
        match std::env::consts::ARCH {
            "x86_64" => Ok(AUDIT_ARCH_X86_64),
            "aarch64" => Ok(AUDIT_ARCH_AARCH64),
            "riscv64" => Ok(AUDIT_ARCH_RISCV64),
            other => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!(
                    "worker sandbox: no seccomp AUDIT_ARCH for {other}; refusing to start unsandboxed"
                ),
            )),
        }
    }

    fn apply_seccomp() -> io::Result<()> {
        let mut filter = build_seccomp_filter()?;
        let mut prog = libc::sock_fprog {
            len: filter.len() as libc::c_ushort,
            filter: filter.as_mut_ptr(),
        };
        let rc = unsafe {
            libc::syscall(
                libc::SYS_seccomp,
                libc::SECCOMP_SET_MODE_FILTER,
                0u32,
                &mut prog as *mut libc::sock_fprog,
            )
        };
        if rc < 0 {
            return Err(sandbox_os("seccomp(SET_MODE_FILTER)"));
        }
        Ok(())
    }

    fn build_seccomp_filter() -> io::Result<Vec<libc::sock_filter>> {
        let arch = audit_arch()?;
        let load_abs = (libc::BPF_LD | libc::BPF_W | libc::BPF_ABS) as u16;
        let jmp_eq = (libc::BPF_JMP | libc::BPF_JEQ | libc::BPF_K) as u16;
        let jmp_ge = (libc::BPF_JMP | libc::BPF_JGE | libc::BPF_K) as u16;
        let jmp_set = (libc::BPF_JMP | libc::BPF_JSET | libc::BPF_K) as u16;
        let ret = (libc::BPF_RET | libc::BPF_K) as u16;
        let kill = libc::SECCOMP_RET_KILL_PROCESS;
        let allow = libc::SECCOMP_RET_ALLOW;
        let enosys = libc::SECCOMP_RET_ERRNO | (libc::ENOSYS as u32);

        let mut f = Vec::new();
        // struct seccomp_data { nr: i32 @0, arch: u32 @4, ip: u64 @8, args[0]: u64 @16 }
        f.push(bpf_stmt(load_abs, 4));
        f.push(bpf_jump(jmp_eq, arch, 1, 0));
        f.push(bpf_stmt(ret, kill));
        f.push(bpf_stmt(load_abs, 0));
        // x32 ABI (bit 30) is a known seccomp bypass on x86_64.
        f.push(bpf_jump(jmp_ge, 0x4000_0000, 0, 1));
        f.push(bpf_stmt(ret, kill));

        for nr in deny_kill_syscalls() {
            f.push(bpf_jump(jmp_eq, nr as u32, 0, 1));
            f.push(bpf_stmt(ret, kill));
        }

        f.push(bpf_jump(jmp_eq, libc::SYS_clone3 as u32, 0, 1));
        f.push(bpf_stmt(ret, enosys));

        // clone(flags, …): flags is args[0]. Deny CLONE_NEWUSER.
        f.push(bpf_jump(jmp_eq, libc::SYS_clone as u32, 0, 3));
        f.push(bpf_stmt(load_abs, 16));
        f.push(bpf_jump(jmp_set, libc::CLONE_NEWUSER as u32, 0, 1));
        f.push(bpf_stmt(ret, kill));

        f.push(bpf_stmt(ret, allow));
        Ok(f)
    }

    fn bpf_stmt(code: u16, k: u32) -> libc::sock_filter {
        libc::sock_filter {
            code,
            jt: 0,
            jf: 0,
            k,
        }
    }

    fn bpf_jump(code: u16, k: u32, jt: u8, jf: u8) -> libc::sock_filter {
        libc::sock_filter { code, jt, jf, k }
    }

    fn deny_kill_syscalls() -> Vec<libc::c_long> {
        let mut nrs = vec![
            libc::SYS_ptrace,
            libc::SYS_process_vm_readv,
            libc::SYS_process_vm_writev,
            libc::SYS_mount,
            libc::SYS_umount2,
            libc::SYS_pivot_root,
            libc::SYS_chroot,
            libc::SYS_reboot,
            libc::SYS_swapon,
            libc::SYS_swapoff,
            libc::SYS_init_module,
            libc::SYS_finit_module,
            libc::SYS_delete_module,
            libc::SYS_kexec_load,
            libc::SYS_kexec_file_load,
            libc::SYS_bpf,
            libc::SYS_userfaultfd,
            libc::SYS_perf_event_open,
            libc::SYS_unshare,
            libc::SYS_setns,
            libc::SYS_acct,
            libc::SYS_syslog,
            libc::SYS_personality,
            libc::SYS_kcmp,
            libc::SYS_open_by_handle_at,
            libc::SYS_name_to_handle_at,
            libc::SYS_keyctl,
            libc::SYS_add_key,
            libc::SYS_request_key,
            libc::SYS_fanotify_init,
            libc::SYS_open_tree,
            libc::SYS_move_mount,
            libc::SYS_fsopen,
            libc::SYS_fsconfig,
            libc::SYS_fsmount,
            libc::SYS_fspick,
            libc::SYS_mount_setattr,
        ];
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            nrs.push(libc::SYS_ioperm);
            nrs.push(libc::SYS_iopl);
        }
        nrs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    #[cfg(target_os = "linux")]
    fn require_live_landlock() -> bool {
        matches!(
            std::env::var("GREPPY_REQUIRE_LIVE_LANDLOCK").as_deref(),
            Ok("1") | Ok("true")
        )
    }

    fn grant_paths(grants: &[FsPathGrant]) -> Vec<&Path> {
        grants.iter().map(|g| g.path.as_path()).collect()
    }

    #[test]
    fn allow_list_includes_required_system_and_worker_paths() {
        let exe = Path::new("/opt/greppy/libexec/web-runtime");
        let tmp = Path::new("/tmp/greppy-worker-unit");
        let grants = fs_allow_list(exe, tmp).unwrap();
        let paths = grant_paths(&grants);
        for expected in [
            Path::new("/usr"),
            Path::new("/lib"),
            Path::new("/lib64"),
            Path::new("/etc"),
            Path::new("/dev/urandom"),
            Path::new("/dev/null"),
            Path::new("/dev/zero"),
            exe,
            Path::new("/opt/greppy/libexec"),
            tmp,
        ] {
            assert!(
                paths.contains(&expected),
                "missing {expected:?} in {paths:?}"
            );
        }
    }

    #[test]
    fn allow_list_does_not_grant_home_or_workspace_root() {
        let exe = Path::new("/opt/greppy/libexec/web-runtime");
        let tmp = Path::new("/tmp/greppy-worker-unit");
        let grants = fs_allow_list(exe, tmp).unwrap();
        let paths = grant_paths(&grants);

        for banned in ["/", "/home", "/Users", "/root", "/var", "/dev"] {
            assert!(
                !paths.iter().any(|p| *p == Path::new(banned)),
                "allow-list must not grant {banned}: {paths:?}"
            );
        }

        if let Some(home) = std::env::var_os("HOME") {
            let home = PathBuf::from(home);
            if home != tmp && !tmp.starts_with(&home) && !home.as_os_str().is_empty() {
                assert!(
                    !paths.iter().any(|p| *p == home.as_path()),
                    "allow-list must not grant $HOME {}: {paths:?}",
                    home.display()
                );
            }
        }

        if let Ok(cwd) = std::env::current_dir() {
            if cwd != tmp && !tmp.starts_with(&cwd) && !cwd.starts_with(tmp) {
                assert!(
                    !paths.iter().any(|p| *p == cwd.as_path()),
                    "allow-list must not grant workspace {}: {paths:?}",
                    cwd.display()
                );
            }
        }
    }

    #[test]
    fn allow_list_may_include_workspace_only_when_it_is_tmp() {
        let exe = Path::new("/opt/greppy/libexec/web-runtime");
        let cwd = std::env::current_dir().expect("cwd");
        let grants = fs_allow_list(exe, &cwd).unwrap();
        assert!(
            grants.iter().any(|g| g.path == cwd && g.is_write_tree()),
            "workspace as tmp must be the write tree: {grants:?}"
        );
        assert!(
            grants
                .iter()
                .filter(|g| g.is_write_tree())
                .all(|g| g.path == cwd),
            "only tmp may be a write tree: {grants:?}"
        );
    }

    #[test]
    fn allow_list_write_tree_is_only_tmp() {
        let exe = Path::new("/opt/greppy/libexec/web-runtime");
        let tmp = Path::new("/tmp/greppy-worker-unit");
        let grants = fs_allow_list(exe, tmp).unwrap();
        let writes: Vec<_> = grants
            .iter()
            .filter(|g| g.is_write_tree())
            .map(|g| g.path.as_path())
            .collect();
        assert_eq!(writes, vec![tmp]);
    }

    #[test]
    fn allow_list_refuses_filesystem_root_as_tmp() {
        let err =
            fs_allow_list(Path::new("/opt/greppy/bin/web-runtime"), Path::new("/")).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(err.to_string().contains("filesystem root"), "{err}");
    }

    #[cfg(not(target_os = "linux"))]
    #[test]
    fn apply_is_unsupported_off_linux() {
        let err = apply(Path::new("/opt/greppy/bin/web-runtime"), Path::new("/tmp")).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Unsupported);
        assert!(
            err.to_string().contains("refusing to start unsandboxed"),
            "{err}"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn apply_fails_closed_when_tmp_is_missing() {
        let err = apply(
            Path::new("/usr/bin/true"),
            Path::new("/no/such/greppy-worker-tmp"),
        )
        .unwrap_err();
        assert_ne!(
            format!("{err}"),
            "",
            "missing tmp must not return Ok unsandboxed"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn landlock_denies_path_outside_allow_list() {
        use std::fs;
        use std::sync::mpsc;

        let pid = std::process::id();
        let base = std::env::temp_dir();
        let tmp = base.join(format!("greppy-ll-tmp-{pid}"));
        let denied = base.join(format!("greppy-ll-secret-{pid}"));
        let _ = fs::remove_dir_all(&tmp);
        let _ = fs::remove_file(&denied);
        fs::create_dir_all(&tmp).expect("create tmp");
        fs::write(tmp.join("ok"), b"allowed").expect("write tmp file");
        fs::write(&denied, b"secret").expect("write denied file");

        let exe = std::env::current_exe().expect("current_exe");
        if denied.starts_with(exe.parent().unwrap_or(exe.as_path())) {
            let _ = fs::remove_dir_all(&tmp);
            let _ = fs::remove_file(&denied);
            let message = format!(
                "SKIP live landlock test: denied path {} sits under exe dir (not treated as a sandbox success)",
                denied.display()
            );
            if require_live_landlock() {
                panic!("{message}");
            }
            eprintln!("{message}");
            return;
        }

        let tmp_for_thread = tmp.clone();
        let denied_for_thread = denied.clone();
        let allowed_for_thread = tmp.join("ok");
        let (tx, rx) = mpsc::channel();
        std::thread::Builder::new()
            .name("greppy-landlock-probe".into())
            .spawn(move || {
                let outcome = match linux::restrict_filesystem(&exe, &tmp_for_thread) {
                    Err(err) => Err(err),
                    Ok(()) => {
                        let allowed = fs::read(&allowed_for_thread);
                        let denied_read = fs::read(&denied_for_thread);
                        Ok((allowed, denied_read))
                    }
                };
                let _ = tx.send(outcome);
            })
            .expect("spawn landlock probe thread")
            .join()
            .expect("join landlock probe thread");

        let _ = fs::remove_dir_all(&tmp);
        let _ = fs::remove_file(&denied);

        match rx.recv().expect("probe result") {
            Err(err)
                if err.kind() == io::ErrorKind::Unsupported
                    || matches!(
                        err.raw_os_error(),
                        Some(libc::ENOSYS) | Some(libc::EOPNOTSUPP)
                    ) =>
            {
                let message = format!(
                    "SKIP live landlock test: cannot apply landlock in this environment: {err} (not treated as a sandbox success)"
                );
                if require_live_landlock() {
                    panic!("{message}");
                }
                eprintln!("{message}");
            }
            Err(err) => panic!("landlock restrict failed unexpectedly: {err}"),
            Ok((allowed, denied_read)) => {
                assert!(
                    allowed.is_ok(),
                    "tmp file must stay readable after landlock: {allowed:?}"
                );
                let denied_err = denied_read.expect_err(
                    "denied path was readable after landlock restrict; sandbox is not enforcing",
                );
                assert!(
                    denied_err.kind() == io::ErrorKind::PermissionDenied
                        || denied_err.raw_os_error() == Some(libc::EACCES)
                        || denied_err.raw_os_error() == Some(libc::EPERM),
                    "denied path must fail with EACCES/EPERM, got {denied_err:?}"
                );
            }
        }
    }
}
