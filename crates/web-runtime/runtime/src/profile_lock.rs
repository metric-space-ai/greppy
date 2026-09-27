//! Exclusive owner lock for opt-in persistent profiles (guide §16.4).

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;

pub struct ProfileLock {
    _file: File,
    #[cfg(unix)]
    _profile_dir: Option<File>,
}

impl ProfileLock {
    pub fn acquire(profile_dir: &Path) -> io::Result<Self> {
        std::fs::create_dir_all(profile_dir)?;
        let path = profile_dir.join("profile.lock");
        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.custom_flags(libc::O_NOFOLLOW);
        }
        let mut file = options.open(&path)?;
        if !try_exclusive(&file)? {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "profile lock held by live process",
            ));
        }
        file.set_len(0)?;
        write!(&mut file, "{}", std::process::id())?;
        file.flush()?;
        Ok(Self {
            _file: file,
            #[cfg(unix)]
            _profile_dir: None,
        })
    }

    #[cfg(unix)]
    pub fn acquire_named(root: &Path, name: &str) -> io::Result<Self> {
        use std::ffi::CString;
        use std::os::fd::{AsRawFd, FromRawFd};
        use std::os::unix::ffi::OsStrExt;

        fn component(value: &[u8]) -> io::Result<CString> {
            CString::new(value)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))
        }
        fn open_dir_at(parent: i32, name: &CString) -> io::Result<File> {
            let fd = unsafe {
                libc::openat(
                    parent,
                    name.as_ptr(),
                    libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
                )
            };
            if fd < 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(unsafe { File::from_raw_fd(fd) })
        }
        fn ensure_dir_at(parent: i32, name: &CString) -> io::Result<File> {
            let rc = unsafe { libc::mkdirat(parent, name.as_ptr(), 0o700) };
            if rc < 0 && io::Error::last_os_error().raw_os_error() != Some(libc::EEXIST) {
                return Err(io::Error::last_os_error());
            }
            open_dir_at(parent, name)
        }

        std::fs::create_dir_all(root)?;
        let root_name = component(root.as_os_str().as_bytes())?;
        let root_fd = unsafe {
            libc::open(
                root_name.as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            )
        };
        if root_fd < 0 {
            return Err(io::Error::last_os_error());
        }
        let root_dir = unsafe { File::from_raw_fd(root_fd) };
        let name = component(name.as_bytes())?;
        let profile_dir = ensure_dir_at(root_dir.as_raw_fd(), &name)?;
        let browser = component(b"browser")?;
        let _browser_dir = ensure_dir_at(profile_dir.as_raw_fd(), &browser)?;
        let lock_name = component(b"profile.lock")?;
        let lock_fd = unsafe {
            libc::openat(
                profile_dir.as_raw_fd(),
                lock_name.as_ptr(),
                libc::O_RDWR | libc::O_CREAT | libc::O_CLOEXEC | libc::O_NOFOLLOW,
                0o600,
            )
        };
        if lock_fd < 0 {
            return Err(io::Error::last_os_error());
        }
        let mut file = unsafe { File::from_raw_fd(lock_fd) };
        if !try_exclusive(&file)? {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "profile lock held by live process",
            ));
        }
        file.set_len(0)?;
        write!(&mut file, "{}", std::process::id())?;
        file.flush()?;
        Ok(Self {
            _file: file,
            _profile_dir: Some(profile_dir),
        })
    }

    #[cfg(not(unix))]
    pub fn acquire_named(root: &Path, name: &str) -> io::Result<Self> {
        std::fs::create_dir_all(root)?;
        let profile_dir = root.join(name);
        if std::fs::symlink_metadata(&profile_dir).is_ok_and(|metadata| {
            metadata.file_type().is_symlink() || !metadata.is_dir()
        }) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "persistent profile must be a real directory",
            ));
        }
        std::fs::create_dir_all(profile_dir.join("browser"))?;
        for path in [profile_dir.join("browser"), profile_dir.join("profile.lock")] {
            if std::fs::symlink_metadata(path).is_ok_and(|metadata| metadata.file_type().is_symlink()) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "persistent profile components must not be symlinks",
                ));
            }
        }
        Self::acquire(&profile_dir)
    }
}

#[cfg(unix)]
fn try_exclusive(file: &File) -> io::Result<bool> {
    use std::os::unix::io::AsRawFd;
    let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if rc == 0 {
        return Ok(true);
    }
    let err = io::Error::last_os_error();
    if err.kind() == io::ErrorKind::WouldBlock {
        Ok(false)
    } else {
        Err(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_lock_is_exclusive_stale_lock_is_recovered() {
        let dir = std::env::temp_dir().join(format!("greppy-profile-lock-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let first = ProfileLock::acquire(&dir).unwrap();
        let second = ProfileLock::acquire(&dir);
        assert!(second.is_err(), "live owner lock must refuse concurrent writers");
        drop(first);
        let recovered = ProfileLock::acquire(&dir).unwrap();
        drop(recovered);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn named_profile_rejects_symlinked_profile_and_browser_components() {
        use std::os::unix::fs::symlink;

        let root = std::env::temp_dir().join(format!(
            "greppy-profile-symlink-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("target")).unwrap();
        symlink(root.join("target"), root.join("alias")).unwrap();
        assert!(ProfileLock::acquire_named(&root, "alias").is_err());

        std::fs::create_dir(root.join("browser-alias")).unwrap();
        symlink(root.join("target"), root.join("browser-alias").join("browser")).unwrap();
        assert!(ProfileLock::acquire_named(&root, "browser-alias").is_err());
        let _ = std::fs::remove_dir_all(&root);
    }
}
