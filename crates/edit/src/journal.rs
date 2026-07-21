//! Logical all-or-nothing multi-file publish with a pre-image journal.
//!
//! Protocol:
//! 1. take the workspace lock (`.greppy-edit.lock` in the workspace root)
//! 2. re-verify every input hash (CAS) under the lock
//! 3. write the journal: per file the pre-image bytes + both hashes,
//!    fsynced, then mark it `committed`
//! 4. publish every file atomically (tmp+fsync+rename)
//! 5. atomically mark the journal `completed`, making rollback impossible
//! 6. remove the journal directory as best-effort cleanup
//!
//! A crash between 3 and 5 leaves a committed journal on disk;
//! `greppy edit recover` restores every pre-image and removes it. A crash
//! before the `committed` marker means nothing was published; the journal
//! is discarded. A completed journal is cleanup-only and never restores
//! pre-images.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::hash::sha256_hex;
use crate::publish::{publish_atomic, require_inside_workspace};
use greppy_core::{Error, Result};

const JOURNAL_DIR: &str = ".greppy-edit-journal";
const LOCK_NAME: &str = ".greppy-edit.lock";
const CRASH_AFTER_ENV: &str = "GREPPY_EDIT_TEST_JOURNAL_CRASH_AFTER";
const CLEANUP_FAILURE_ENV: &str = "GREPPY_EDIT_TEST_JOURNAL_CLEANUP_FAILURE";
const LOCK_STALE_AFTER_SECS: u64 = 600;
static LOCK_NONCE: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JournalEntry {
    pub rel_path: String,
    pub pre_image_file: String,
    pub pre_sha256: String,
    pub post_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Journal {
    pub schema_version: String,
    pub transaction_id: String,
    pub committed: bool,
    #[serde(default)]
    pub completed: bool,
    pub entries: Vec<JournalEntry>,
}

/// One planned file publication.
pub struct FilePublication {
    pub rel_path: String,
    pub expected_live_sha256: String,
    pub content: Vec<u8>,
}

/// Advisory workspace lock acquired with exclusive-create. The payload binds
/// the lock to a PID and a unique token. A dead PID is taken over immediately;
/// malformed legacy locks are only taken over after the staleness TTL.
#[derive(Debug)]
pub struct WorkspaceLock {
    path: PathBuf,
    token: String,
    takeover_reason: Option<String>,
}

impl WorkspaceLock {
    /// Acquire the workspace lock without waiting. An active owner fails
    /// immediately. A dead owner is quarantined with an atomic rename before
    /// this process retries exclusive creation, so contenders cannot delete a
    /// newly-created lock.
    pub fn acquire(workspace_root: &Path) -> Result<Self> {
        let path = workspace_root.join(LOCK_NAME);
        let token = lock_token();
        let mut takeover_reason = None;
        for _ in 0..4 {
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(mut file) => {
                    writeln!(file, "pid={}", std::process::id()).map_err(|source| Error::Io {
                        context: format!("write lock {}", path.display()),
                        source,
                    })?;
                    writeln!(file, "token={token}").map_err(|source| Error::Io {
                        context: format!("write lock {}", path.display()),
                        source,
                    })?;
                    file.sync_all().map_err(|source| Error::Io {
                        context: format!("fsync lock {}", path.display()),
                        source,
                    })?;
                    if let Some(reason) = &takeover_reason {
                        tracing::warn!(workspace = %workspace_root.display(), reason, "taking over orphaned greppy edit lock");
                    }
                    return Ok(Self {
                        path,
                        token,
                        takeover_reason,
                    });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    let state = inspect_lock(&path);
                    let reason = match state {
                        LockState::Active { pid } => {
                            return Err(Error::Workspace(format!(
                            "another greppy edit transaction holds the workspace lock (pid {pid})"
                        )))
                        }
                        LockState::Orphaned { reason } => reason,
                    };
                    let quarantine = workspace_root
                        .join(format!("{LOCK_NAME}.stale.{}", token.replace(':', "-")));
                    match std::fs::rename(&path, &quarantine) {
                        Ok(()) => {
                            let _ = std::fs::remove_file(&quarantine);
                            takeover_reason = Some(reason);
                        }
                        Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                        Err(source) => {
                            return Err(Error::Io {
                                context: format!("quarantine stale lock {}", path.display()),
                                source,
                            })
                        }
                    }
                }
                Err(source) => {
                    return Err(Error::Io {
                        context: format!("create lock {}", path.display()),
                        source,
                    })
                }
            }
        }
        Err(Error::Workspace(
            "could not acquire workspace lock after orphan takeover".into(),
        ))
    }

    /// Explanation recorded when this acquisition replaced an orphaned lock.
    pub fn takeover_reason(&self) -> Option<&str> {
        self.takeover_reason.as_deref()
    }
}

impl Drop for WorkspaceLock {
    fn drop(&mut self) {
        let owns_lock = std::fs::read_to_string(&self.path)
            .ok()
            .and_then(|payload| lock_token_from_payload(&payload).map(str::to_owned))
            .is_some_and(|token| token == self.token);
        if owns_lock {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

#[derive(Debug)]
enum LockState {
    Active { pid: u32 },
    Orphaned { reason: String },
}

fn lock_token() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let nonce = LOCK_NONCE.fetch_add(1, Ordering::Relaxed);
    format!("{}:{nanos}:{nonce}", std::process::id())
}

fn lock_pid_from_payload(payload: &str) -> Option<u32> {
    payload.lines().find_map(|line| {
        line.strip_prefix("pid=")
            .unwrap_or(line)
            .trim()
            .parse()
            .ok()
    })
}

fn lock_token_from_payload(payload: &str) -> Option<&str> {
    payload.lines().find_map(|line| line.strip_prefix("token="))
}

fn inspect_lock(path: &Path) -> LockState {
    let payload = std::fs::read_to_string(path).unwrap_or_default();
    if let Some(pid) = lock_pid_from_payload(&payload) {
        if process_is_live(pid) {
            return LockState::Active { pid };
        }
        return LockState::Orphaned {
            reason: format!("lock owner pid {pid} is no longer alive"),
        };
    }
    let stale = std::fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|modified| modified.elapsed().ok())
        .is_some_and(|age| age.as_secs() > LOCK_STALE_AFTER_SECS);
    if stale {
        LockState::Orphaned {
            reason: format!(
                "lock payload had no readable pid and was older than {LOCK_STALE_AFTER_SECS} seconds"
            ),
        }
    } else {
        LockState::Active { pid: 0 }
    }
}

#[cfg(unix)]
fn process_is_live(pid: u32) -> bool {
    if pid == std::process::id() {
        return true;
    }
    match std::process::Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "pid="])
        .output()
    {
        Ok(output) => output.status.success() && !output.stdout.is_empty(),
        Err(_) => true, // liveness probe unavailable: fail closed
    }
}

#[cfg(windows)]
fn process_is_live(pid: u32) -> bool {
    if pid == std::process::id() {
        return true;
    }
    let pid = pid.to_string();
    let filter = format!("PID eq {pid}");
    match std::process::Command::new("tasklist")
        .args(["/FI", &filter, "/FO", "CSV", "/NH"])
        .output()
    {
        Ok(output) => {
            output.status.success()
                && String::from_utf8_lossy(&output.stdout)
                    .split(',')
                    .any(|field| field.trim_matches([' ', '"', '\r', '\n']) == pid)
        }
        Err(_) => true, // liveness probe unavailable: fail closed
    }
}

#[cfg(not(any(unix, windows)))]
fn process_is_live(_pid: u32) -> bool {
    // Unknown platforms fail closed rather than stealing a potentially active
    // lock. Malformed lock files still use the age-based fallback above.
    true
}

fn journal_dir(workspace_root: &Path) -> PathBuf {
    workspace_root.join(JOURNAL_DIR)
}

fn journal_path(workspace_root: &Path) -> PathBuf {
    journal_dir(workspace_root).join("journal.json")
}

fn crash_after(boundary: &str) -> Result<()> {
    if std::env::var(CRASH_AFTER_ENV).as_deref() == Ok(boundary) {
        return Err(Error::Workspace(format!(
            "injected journal crash after {boundary}"
        )));
    }
    Ok(())
}

fn cleanup_journal_dir(dir: &Path) -> std::io::Result<()> {
    if std::env::var_os(CLEANUP_FAILURE_ENV).is_some() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "injected journal cleanup failure",
        ));
    }
    std::fs::remove_dir_all(dir)
}

fn warn_cleanup_failure(dir: &Path, error: &std::io::Error) {
    eprintln!(
        "warning: edit transaction completed, but journal cleanup failed for {}: {error}",
        dir.display()
    );
}

/// Publish `files` as one logical transaction. Returns the transaction id.
pub fn publish_journal(
    workspace_root: &Path,
    transaction_id: &str,
    files: &[FilePublication],
) -> Result<()> {
    let lock = WorkspaceLock::acquire(workspace_root)?;
    crash_after("lock-acquired")?;
    publish_journal_locked(workspace_root, transaction_id, files, &lock)
}

/// Publish while the caller holds the workspace lock. Used by plan execution,
/// which keeps the same lock from snapshot through validation and publication.
pub(crate) fn publish_journal_locked(
    workspace_root: &Path,
    transaction_id: &str,
    files: &[FilePublication],
    _lock: &WorkspaceLock,
) -> Result<()> {
    // CAS for every file under the lock, before anything is written
    for f in files {
        let abs = require_inside_workspace(workspace_root, &workspace_root.join(&f.rel_path))?;
        let live = std::fs::read(&abs).map_err(|source| Error::Io {
            context: format!("read {}", abs.display()),
            source,
        })?;
        if sha256_hex(&live) != f.expected_live_sha256 {
            return Err(Error::Workspace(format!(
                "stale plan: {} changed since planning; nothing was written",
                f.rel_path
            )));
        }
    }
    crash_after("cas-verified")?;

    // write pre-images + journal, fsync, then mark committed
    let dir = journal_dir(workspace_root);
    std::fs::create_dir_all(&dir).map_err(|source| Error::Io {
        context: format!("create {}", dir.display()),
        source,
    })?;
    crash_after("journal-dir-created")?;
    let mut entries = Vec::new();
    for (i, f) in files.iter().enumerate() {
        let abs = workspace_root.join(&f.rel_path);
        let pre = std::fs::read(&abs).map_err(|source| Error::Io {
            context: format!("read {}", abs.display()),
            source,
        })?;
        let pre_name = format!("pre-{i:04}.bin");
        let pre_path = dir.join(&pre_name);
        std::fs::write(&pre_path, &pre).map_err(|source| Error::Io {
            context: format!("write {}", pre_path.display()),
            source,
        })?;
        if let Ok(h) = std::fs::File::open(&pre_path) {
            let _ = h.sync_all();
        }
        crash_after(&format!("pre-image-{i}"))?;
        entries.push(JournalEntry {
            rel_path: f.rel_path.clone(),
            pre_image_file: pre_name,
            pre_sha256: sha256_hex(&pre),
            post_sha256: sha256_hex(&f.content),
        });
    }
    let mut journal = Journal {
        schema_version: "greppy.edit-journal.v1".into(),
        transaction_id: transaction_id.to_string(),
        committed: false,
        completed: false,
        entries,
    };
    write_journal(workspace_root, &journal)?;
    crash_after("journal-uncommitted")?;
    journal.committed = true;
    write_journal(workspace_root, &journal)?;
    crash_after("journal-committed")?;

    // publish; roll back from pre-images on any failure
    let mut published = 0usize;
    let mut failure: Option<Error> = None;
    for (index, f) in files.iter().enumerate() {
        match publish_atomic(
            workspace_root,
            &workspace_root.join(&f.rel_path),
            &f.content,
            &f.expected_live_sha256,
        ) {
            Ok(_) => {
                published += 1;
                crash_after(&format!("published-{index}"))?;
            }
            Err(e) => {
                failure = Some(e);
                break;
            }
        }
    }
    if let Some(e) = failure {
        // Best-effort: attempt every rollback even if one fails, so a single
        // unreadable pre-image cannot strand the other published files.
        let mut rollback_failures: Vec<String> = Vec::new();
        for (f, entry) in files.iter().zip(&journal.entries).take(published) {
            // rollback ignores CAS: restoring the pre-image is the contract
            let abs = workspace_root.join(&f.rel_path);
            let restored = std::fs::read(dir.join(&entry.pre_image_file))
                .and_then(|pre| std::fs::write(&abs, &pre));
            if let Err(source) = restored {
                rollback_failures.push(format!("{}: {source}", f.rel_path));
            }
        }
        if !rollback_failures.is_empty() {
            // Keep the committed journal so `greppy edit recover` can finish
            // the job; surface both the publish failure and the stranded files.
            return Err(Error::Workspace(format!(
                "publish failed ({e}) and rollback is incomplete for {}; \
                 the journal was kept — run `greppy edit recover`",
                rollback_failures.join(", ")
            )));
        }
        let _ = std::fs::remove_dir_all(&dir);
        return Err(e);
    }
    // Every target now carries its post-image. From this point onward recovery
    // must never restore pre-images. Persist that phase transition atomically
    // before attempting best-effort directory cleanup. Even if writing the
    // marker fails, recovery also recognizes an all-post-image journal as
    // completed.
    journal.completed = true;
    if let Err(error) = write_journal(workspace_root, &journal) {
        eprintln!(
            "warning: edit transaction completed, but the journal completion marker could not be written: {error}"
        );
    }
    if let Err(error) = cleanup_journal_dir(&dir) {
        warn_cleanup_failure(&dir, &error);
    }
    // Fault injection after the point of no return must not turn a completed
    // publication into a false PublishFailed certificate.
    let _ = crash_after("journal-removed");
    Ok(())
}

fn write_journal(workspace_root: &Path, journal: &Journal) -> Result<()> {
    let path = journal_path(workspace_root);
    let bytes = serde_json::to_vec_pretty(journal)
        .map_err(|e| Error::Invalid(format!("serialize journal: {e}")))?;
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, &bytes).map_err(|source| Error::Io {
        context: format!("write {}", tmp.display()),
        source,
    })?;
    if let Ok(h) = std::fs::File::open(&tmp) {
        let _ = h.sync_all();
    }
    std::fs::rename(&tmp, &path).map_err(|source| Error::Io {
        context: format!("rename {}", path.display()),
        source,
    })?;
    Ok(())
}

/// Recovery outcome retained for the existing CLI surface.
#[derive(Debug, PartialEq, Eq)]
pub enum Recovery {
    NothingToRecover,
    RolledBack {
        transaction_id: String,
        files: usize,
    },
    DiscardedUncommitted,
}

/// Full explicit-recovery report for library callers and `--report` wiring.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveryReport {
    pub found_journal: bool,
    pub committed: bool,
    pub action: RecoveryAction,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transaction_id: Option<String>,
    pub files_considered: usize,
    pub files_restored: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RecoveryAction {
    NothingToRecover,
    DiscardedUncommitted,
    RolledBack,
}

/// Restore pre-images from a committed journal left by a crash and return a
/// concise compatibility outcome.
pub fn recover(workspace_root: &Path) -> Result<Recovery> {
    let report = recover_with_report(workspace_root)?;
    Ok(match report.action {
        RecoveryAction::NothingToRecover => Recovery::NothingToRecover,
        RecoveryAction::DiscardedUncommitted => Recovery::DiscardedUncommitted,
        RecoveryAction::RolledBack => Recovery::RolledBack {
            transaction_id: report.transaction_id.unwrap_or_default(),
            files: report.files_restored,
        },
    })
}

/// Explicit journal recovery with a machine-readable report. Recovery is
/// serialized by the same workspace lock as plan publication.
pub fn recover_with_report(workspace_root: &Path) -> Result<RecoveryReport> {
    let _lock = WorkspaceLock::acquire(workspace_root)?;
    let path = journal_path(workspace_root);
    if !path.exists() {
        let dir = journal_dir(workspace_root);
        if dir.exists() {
            std::fs::remove_dir_all(&dir).map_err(|source| Error::Io {
                context: format!("remove orphan journal {}", dir.display()),
                source,
            })?;
            return Ok(RecoveryReport {
                found_journal: true,
                committed: false,
                action: RecoveryAction::DiscardedUncommitted,
                transaction_id: None,
                files_considered: 0,
                files_restored: 0,
            });
        }
        return Ok(RecoveryReport {
            found_journal: false,
            committed: false,
            action: RecoveryAction::NothingToRecover,
            transaction_id: None,
            files_considered: 0,
            files_restored: 0,
        });
    }
    let journal: Journal =
        serde_json::from_slice(&std::fs::read(&path).map_err(|source| Error::Io {
            context: format!("read {}", path.display()),
            source,
        })?)
        .map_err(|e| Error::Invalid(format!("journal unreadable: {e}")))?;
    let dir = journal_dir(workspace_root);
    if journal.completed {
        if let Err(error) = cleanup_journal_dir(&dir) {
            warn_cleanup_failure(&dir, &error);
        }
        return Ok(RecoveryReport {
            found_journal: true,
            committed: true,
            action: RecoveryAction::NothingToRecover,
            transaction_id: Some(journal.transaction_id),
            files_considered: journal.entries.len(),
            files_restored: 0,
        });
    }
    if !journal.committed {
        let transaction_id = Some(journal.transaction_id);
        let files_considered = journal.entries.len();
        std::fs::remove_dir_all(&dir).map_err(|source| Error::Io {
            context: format!("remove journal {}", dir.display()),
            source,
        })?;
        return Ok(RecoveryReport {
            found_journal: true,
            committed: false,
            action: RecoveryAction::DiscardedUncommitted,
            transaction_id,
            files_considered,
            files_restored: 0,
        });
    }

    let live_hashes: Vec<String> = journal
        .entries
        .iter()
        .map(|entry| {
            std::fs::read(workspace_root.join(&entry.rel_path))
                .map(|bytes| sha256_hex(&bytes))
                .unwrap_or_default()
        })
        .collect();
    // A process can stop after publishing the final file but before persisting
    // `completed`. All post-images are sufficient proof that publication
    // reached its point of no return; such a journal is cleanup-only.
    if journal
        .entries
        .iter()
        .zip(&live_hashes)
        .all(|(entry, live_sha)| live_sha == &entry.post_sha256)
    {
        if let Err(error) = cleanup_journal_dir(&dir) {
            warn_cleanup_failure(&dir, &error);
        }
        return Ok(RecoveryReport {
            found_journal: true,
            committed: true,
            action: RecoveryAction::NothingToRecover,
            transaction_id: Some(journal.transaction_id),
            files_considered: journal.entries.len(),
            files_restored: 0,
        });
    }

    let mut restored = 0usize;
    for (entry, live_sha) in journal.entries.iter().zip(live_hashes) {
        let abs = workspace_root.join(&entry.rel_path);
        if live_sha == entry.pre_sha256 {
            continue; // this file was never published
        }
        // restore only files that carry the transaction's post-image; any
        // OTHER content means someone edited after the crash - refuse
        if live_sha != entry.post_sha256 {
            return Err(Error::Workspace(format!(
                "recover: {} was modified after the crashed transaction; resolve manually",
                entry.rel_path
            )));
        }
        let pre = std::fs::read(dir.join(&entry.pre_image_file)).map_err(|source| Error::Io {
            context: format!("read pre-image for {}", entry.rel_path),
            source,
        })?;
        std::fs::write(&abs, &pre).map_err(|source| Error::Io {
            context: format!("restore {}", abs.display()),
            source,
        })?;
        restored += 1;
    }
    std::fs::remove_dir_all(&dir).map_err(|source| Error::Io {
        context: format!("remove journal {}", dir.display()),
        source,
    })?;
    Ok(RecoveryReport {
        found_journal: true,
        committed: true,
        action: RecoveryAction::RolledBack,
        transaction_id: Some(journal.transaction_id),
        files_considered: journal.entries.len(),
        files_restored: restored,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ws() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    fn pubfile(rel: &str, old: &[u8], new: &[u8]) -> FilePublication {
        FilePublication {
            rel_path: rel.into(),
            expected_live_sha256: sha256_hex(old),
            content: new.to_vec(),
        }
    }

    #[test]
    fn two_file_transaction_publishes_both() {
        let dir = ws();
        std::fs::write(dir.path().join("a.txt"), b"a1").unwrap();
        std::fs::write(dir.path().join("b.txt"), b"b1").unwrap();
        publish_journal(
            dir.path(),
            "tx-1",
            &[
                pubfile("a.txt", b"a1", b"a2"),
                pubfile("b.txt", b"b1", b"b2"),
            ],
        )
        .unwrap();
        assert_eq!(std::fs::read(dir.path().join("a.txt")).unwrap(), b"a2");
        assert_eq!(std::fs::read(dir.path().join("b.txt")).unwrap(), b"b2");
        assert!(!journal_path(dir.path()).exists());
    }

    #[test]
    fn stale_second_file_changes_nothing() {
        let dir = ws();
        std::fs::write(dir.path().join("a.txt"), b"a1").unwrap();
        std::fs::write(dir.path().join("b.txt"), b"CHANGED").unwrap();
        let err = publish_journal(
            dir.path(),
            "tx-2",
            &[
                pubfile("a.txt", b"a1", b"a2"),
                pubfile("b.txt", b"b1", b"b2"),
            ],
        );
        assert!(err.is_err());
        assert_eq!(std::fs::read(dir.path().join("a.txt")).unwrap(), b"a1");
        assert_eq!(std::fs::read(dir.path().join("b.txt")).unwrap(), b"CHANGED");
    }

    #[test]
    fn recover_restores_partially_published_committed_journal() {
        let dir = ws();
        std::fs::write(dir.path().join("a.txt"), b"a2").unwrap(); // published post-image
        std::fs::write(dir.path().join("b.txt"), b"b1").unwrap(); // not yet published
        let jd = journal_dir(dir.path());
        std::fs::create_dir_all(&jd).unwrap();
        std::fs::write(jd.join("pre-0000.bin"), b"a1").unwrap();
        std::fs::write(jd.join("pre-0001.bin"), b"b1").unwrap();
        let journal = Journal {
            schema_version: "greppy.edit-journal.v1".into(),
            transaction_id: "tx-crash".into(),
            committed: true,
            completed: false,
            entries: vec![
                JournalEntry {
                    rel_path: "a.txt".into(),
                    pre_image_file: "pre-0000.bin".into(),
                    pre_sha256: sha256_hex(b"a1"),
                    post_sha256: sha256_hex(b"a2"),
                },
                JournalEntry {
                    rel_path: "b.txt".into(),
                    pre_image_file: "pre-0001.bin".into(),
                    pre_sha256: sha256_hex(b"b1"),
                    post_sha256: sha256_hex(b"b2"),
                },
            ],
        };
        write_journal(dir.path(), &journal).unwrap();
        let out = recover(dir.path()).unwrap();
        assert_eq!(
            out,
            Recovery::RolledBack {
                transaction_id: "tx-crash".into(),
                files: 1
            }
        );
        assert_eq!(std::fs::read(dir.path().join("a.txt")).unwrap(), b"a1");
    }

    #[test]
    fn recover_refuses_foreign_edits() {
        let dir = ws();
        std::fs::write(dir.path().join("a.txt"), b"SOMEONE ELSE").unwrap();
        let jd = journal_dir(dir.path());
        std::fs::create_dir_all(&jd).unwrap();
        std::fs::write(jd.join("pre-0000.bin"), b"a1").unwrap();
        let journal = Journal {
            schema_version: "greppy.edit-journal.v1".into(),
            transaction_id: "tx-crash".into(),
            committed: true,
            completed: false,
            entries: vec![JournalEntry {
                rel_path: "a.txt".into(),
                pre_image_file: "pre-0000.bin".into(),
                pre_sha256: sha256_hex(b"a1"),
                post_sha256: sha256_hex(b"a2"),
            }],
        };
        write_journal(dir.path(), &journal).unwrap();
        assert!(recover(dir.path()).is_err());
        assert_eq!(
            std::fs::read(dir.path().join("a.txt")).unwrap(),
            b"SOMEONE ELSE"
        );
    }

    #[test]
    fn nothing_to_recover() {
        let dir = ws();
        assert_eq!(recover(dir.path()).unwrap(), Recovery::NothingToRecover);
    }
}
