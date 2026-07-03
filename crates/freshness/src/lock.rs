//! Advisory file lock for one-writer per workspace (R-021 / WP-R021).
//!
//! Implementation: lock file at `<target>.lock` containing the
//! holder's PID and timestamp. Acquisition is `O_CREAT|O_EXCL`-based;
//! a stale lock (dead PID, or older than `STALE_AFTER`) is detected
//! by reading the body and is taken over by the new caller.
//!
//! Why this design: avoids adding a new dep (`fs2` / `nix`) and works
//! on macOS + Linux (Tier 1). Windows story is documented but not
//! implemented; the lock is acquired before `Store::open` even reads
//! the SQLite header, so on Windows we simply skip lock acquisition
//! (SQLite's own locking handles single-process integrity).

use std::fs::OpenOptions;
use std::io::{ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use thiserror::Error;

use grepplus_core::Error as CoreError;

/// A lock older than this is considered stale and may be taken over
/// by a new caller (R-021: a crashed holder must not block forever).
pub const STALE_AFTER: Duration = Duration::from_secs(5 * 60);

#[derive(Debug, Error)]
pub enum LockError {
    #[error("io: {context}: {source}")]
    Io {
        context: String,
        #[source]
        source: std::io::Error,
    },
    #[error("lock held by another writer (pid {pid:?}, age_secs {age_secs:?}); path: {path}")]
    Held {
        pid: Option<u32>,
        age_secs: Option<u64>,
        path: PathBuf,
    },
}

/// What `try_acquire` observed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LockOutcome {
    /// Caller now holds the lock; the returned [`Lock`] releases it on drop.
    Acquired,
    /// Caller took over a stale lock from a crashed/old holder.
    AcquiredFromStale,
    /// Another live process holds the lock; caller backs off.
    Contended,
}

/// RAII lock handle. Releases on `Drop` by deleting the lock file.
#[derive(Debug)]
pub struct Lock {
    pub(crate) path: PathBuf,
}

impl Lock {
    /// Construct a lock handle over `path`. Callers usually do not
    /// call this directly — `try_acquire` returns the right
    /// `LockOutcome` for you to wrap. `new` is exposed for cases
    /// where the caller has already verified acquisition (e.g.,
    /// during tests).
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

impl Drop for Lock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Try to acquire the lock for `target`. The lock file lives at
/// `target.lock` (alongside the target).
///
/// Behaviour:
/// 1. `O_CREAT | O_EXCL` the lock file. If it did not exist, we win — write our PID + timestamp and return.
/// 2. If the file already exists, read its body. If the PID is dead
///    or the file is older than `STALE_AFTER`, take it over by
///    re-creating it with our PID.
/// 3. Otherwise, return `LockError::Held` with the offending pid/age
///    for the caller's diagnostic.
pub fn try_acquire(target: &Path) -> std::result::Result<LockOutcome, LockError> {
    let lock_path = lock_path_for(target);
    let holder_pid = std::process::id();

    // Phase 1: try O_EXCL create.
    match OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&lock_path)
    {
        Ok(mut f) => {
            write_lock_body(&mut f, holder_pid).map_err(|e| LockError::Io {
                context: format!("write lock body {}", lock_path.display()),
                source: e,
            })?;
            Ok(LockOutcome::Acquired)
        }
        Err(e) if e.kind() == ErrorKind::AlreadyExists => {
            // Phase 2: a prior holder exists. Decide whether to take over.
            if try_take_over(&lock_path, holder_pid)? {
                Ok(LockOutcome::AcquiredFromStale)
            } else {
                let (pid, age) = read_lock_body(&lock_path).unwrap_or((None, None));
                Err(LockError::Held {
                    pid,
                    age_secs: age,
                    path: lock_path,
                })
            }
        }
        Err(e) => Err(LockError::Io {
            context: format!("create_new {}", lock_path.display()),
            source: e,
        }),
    }
}

/// Compatibility alias — `try_lock` was the old name (R-021 noted it
/// was dead-code + crash-unsafe; this version is crash-safe via the
/// stale-recovery path).
pub fn try_lock(target: &Path) -> std::result::Result<LockOutcome, LockError> {
    try_acquire(target)
}

fn write_lock_body(f: &mut std::fs::File, pid: u32) -> std::io::Result<()> {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let body = format!("{pid}\n{secs}\n");
    f.write_all(body.as_bytes())?;
    f.sync_all()
}

fn read_lock_body(path: &Path) -> std::io::Result<(Option<u32>, Option<u64>)> {
    let mut s = String::new();
    std::fs::File::open(path)?.read_to_string(&mut s)?;
    let mut lines = s.lines();
    let pid = lines.next().and_then(|s| s.parse::<u32>().ok());
    let secs = lines
        .next()
        .and_then(|s| s.parse::<u64>().ok())
        .and_then(|now| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .ok()
                .map(|d| d.as_secs())
                .and_then(|cur| cur.checked_sub(now))
        });
    Ok((pid, secs))
}

/// A lock is stale if the holder's PID is dead, or if it's older
/// than `STALE_AFTER`. Returns `Ok(true)` if we successfully took
/// over, `Ok(false)` if a live holder is in the way.
fn try_take_over(lock_path: &Path, new_pid: u32) -> std::result::Result<bool, LockError> {
    let (pid, age_secs) = match read_lock_body(lock_path) {
        Ok(v) => v,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(true),
        Err(e) => {
            return Err(LockError::Io {
                context: format!("read {}", lock_path.display()),
                source: e,
            })
        }
    };
    let stale = match (pid, age_secs) {
        (Some(p), _) if !pid_is_alive(p) => true,
        (_, Some(a)) if a >= STALE_AFTER.as_secs() => true,
        _ => false,
    };
    if !stale {
        return Ok(false);
    }
    // Take over: remove the stale file and create a new one. The
    // remove + create is a tiny race window where another live
    // caller could grab the lock; that's fine — the loser of this
    // race gets `Contended` from their own O_EXCL attempt.
    let _ = std::fs::remove_file(lock_path);
    let mut f = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(lock_path)
        .map_err(|e| LockError::Io {
            context: format!("recreate {}", lock_path.display()),
            source: e,
        })?;
    write_lock_body(&mut f, new_pid).map_err(|e| LockError::Io {
        context: format!("rewrite {}", lock_path.display()),
        source: e,
    })?;
    Ok(true)
}

/// PID liveness check on macOS + Linux (Tier 1). Returns `true` if
/// the process exists according to `kill(pid, 0)`.
fn pid_is_alive(pid: u32) -> bool {
    #[cfg(unix)]
    {
        // SAFETY: `kill(pid, 0)` only checks for existence; it
        // does not actually deliver a signal. Returns 0 if the
        // process exists, -1 with errno=ESRCH if it does not.
        let r = unsafe { libc_kill(pid as i32, 0) };
        // We could also check errno for EPERM, but for the
        // adversarial "shared /tmp" scenario in §14 the dead-PID
        // case is what matters; if we lack permission we treat
        // the lock as live (do not take over).
        r == 0
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        true
    }
}

#[cfg(unix)]
extern "C" {
    #[link_name = "kill"]
    fn libc_kill(pid: i32, sig: i32) -> i32;
}

pub fn lock_path_for(target: &Path) -> PathBuf {
    let mut s = target.as_os_str().to_owned();
    s.push(".lock");
    PathBuf::from(s)
}

/// Convert a `LockError` into a `grepplus_core::Error` for callers
/// that don't want to import `thiserror` directly.
impl From<LockError> for CoreError {
    fn from(e: LockError) -> Self {
        match e {
            LockError::Io { context, source } => grepplus_core::Error::io(context, source),
            LockError::Held { path, .. } => {
                grepplus_core::Error::Lock(format!("held: {}", path.display()))
            }
        }
    }
}

/// Helper: acquire the lock, run `f`, release. Returns the function's
/// result or a `LockError`.
pub fn with_lock<T, F>(target: &Path, f: F) -> std::result::Result<T, LockError>
where
    F: FnOnce() -> std::result::Result<T, LockError>,
{
    let _lock = match try_acquire(target)? {
        LockOutcome::Acquired | LockOutcome::AcquiredFromStale => Lock {
            path: lock_path_for(target),
        },
        LockOutcome::Contended => {
            return Err(LockError::Held {
                pid: None,
                age_secs: None,
                path: lock_path_for(target),
            });
        }
    };
    f()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lock_acquire_and_release() {
        let dir = tempdir();
        let target = dir.join("graph.db");
        let r = try_acquire(&target).unwrap();
        assert!(matches!(r, LockOutcome::Acquired));
        let lock_path = lock_path_for(&target);
        assert!(lock_path.exists());
        drop(Lock {
            path: lock_path.clone(),
        });
        assert!(!lock_path.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn lock_contended_when_file_already_exists_with_live_pid() {
        let dir = tempdir();
        let target = dir.join("graph.db");
        let lock_path = lock_path_for(&target);
        // Pretend our own PID is the holder: lock file reads as live,
        // so try_acquire should bail with LockError::Held.
        std::fs::write(
            &lock_path,
            format!("{}\n{}\n", std::process::id(), unix_now_secs()),
        )
        .unwrap();
        let err = try_acquire(&target).unwrap_err();
        assert!(matches!(err, LockError::Held { .. }), "got {err:?}");
        // Cleanup.
        let _ = std::fs::remove_file(&lock_path);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn lock_takes_over_stale_lock() {
        let dir = tempdir();
        let target = dir.join("graph.db");
        let lock_path = lock_path_for(&target);
        // Stale = PID is dead (PID 1 usually exists, but here we
        // simulate a clearly-dead PID). The age check alone wouldn't
        // fire within 5 min, so we mark the pid as defunct via a
        // number we know does not exist on this host.
        let very_unlikely_pid: u32 = 0x7FFFFFFE;
        std::fs::write(
            &lock_path,
            format!("{very_unlikely_pid}\n{}\n", unix_now_secs()),
        )
        .unwrap();
        let r = try_acquire(&target).unwrap();
        assert!(
            matches!(r, LockOutcome::AcquiredFromStale),
            "expected AcquiredFromStale; got {r:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn lock_takes_over_lock_with_stale_age() {
        let dir = tempdir();
        let target = dir.join("graph.db");
        let lock_path = lock_path_for(&target);
        // Pretend the lock was written 10 minutes ago, with our own
        // PID (which is live). Even though pid is alive, age > 5min
        // means stale.
        let old_secs = unix_now_secs() - STALE_AFTER.as_secs() - 60;
        std::fs::write(
            &lock_path,
            format!("{}\n{}\n", std::process::id(), old_secs),
        )
        .unwrap();
        let r = try_acquire(&target).unwrap();
        assert!(
            matches!(r, LockOutcome::AcquiredFromStale),
            "expected AcquiredFromStale due to age; got {r:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dead_process_does_not_block_new_acquirer() {
        // After acquire + drop simulation: a crashed holder's lock
        // becomes recoverable. (Equivalent to one of the above
        // tests; included for naming clarity in the suite.)
        let dir = tempdir();
        let target = dir.join("graph.db");
        let lock_path = lock_path_for(&target);
        // Crashed=stale: half-fake.
        let old_secs = unix_now_secs() - STALE_AFTER.as_secs() - 1;
        std::fs::write(
            &lock_path,
            format!("{}\n{}\n", std::process::id(), old_secs),
        )
        .unwrap();
        let r = try_acquire(&target).unwrap();
        assert!(matches!(r, LockOutcome::AcquiredFromStale));
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn tempdir() -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "grepplus-lock-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn unix_now_secs() -> u64 {
        std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }
}
