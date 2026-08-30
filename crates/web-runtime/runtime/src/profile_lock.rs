//! Exclusive owner lock for opt-in persistent profiles (guide §16.4).

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

pub struct ProfileLock {
    _file: File,
    path: PathBuf,
}

impl ProfileLock {
    pub fn acquire(profile_dir: &Path) -> io::Result<Self> {
        std::fs::create_dir_all(profile_dir)?;
        let path = profile_dir.join("profile.lock");
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&path)?;
        if !try_exclusive(&file)? {
            let mut buf = String::new();
            let _ = file.read_to_string(&mut buf);
            let owner: u32 = buf.trim().parse().unwrap_or(0);
            if owner != 0 && pid_alive(owner) {
                return Err(io::Error::new(
                    io::ErrorKind::WouldBlock,
                    format!("profile lock held by live pid {owner}"),
                ));
            }
            // Stale lock from a crashed owner: take it.
        }
        file.set_len(0)?;
        write!(&mut file, "{}", std::process::id())?;
        file.flush()?;
        Ok(Self { _file: file, path })
    }
}

impl Drop for ProfileLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
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

#[cfg(unix)]
fn pid_alive(pid: u32) -> bool {
    unsafe { libc::kill(pid as i32, 0) == 0 }
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
}
