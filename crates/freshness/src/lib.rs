//! `grepplus-freshness` — workspace fingerprint, on-demand freshness
//! check, incremental update path, and locking.
//!
//! Phase 5 implements:
//! - [`WorkspaceFingerprint`] — captured from `git rev-parse` + index
//!   signature; stored in `workspace_state`.
//! - [`FreshnessCheck`] — runs the on-demand fingerprint comparison
//!   against the persisted `workspace_state`.
//! - [`incremental_update`] — file-level diff (added/modified/deleted)
//!   against `file_state`, reparses, atomically swaps.
//! - [`Lock`] — `flock`-style advisory file lock for one-writer.
//!
//! Phase 5 deliberately does NOT implement a long-running watcher
//! (`grepplus index --watch`); the on-demand path is sufficient for the
//! drop-in `grep` use case.

#![deny(rust_2018_idioms)]

pub mod fingerprint;
pub mod incremental;
pub mod lock;

pub use fingerprint::{
    capture_fingerprint as capture, check, check_files, check_files_report_with_overrides,
    check_files_report_with_ttl, check_files_with_overrides, check_with_overrides,
    FileFreshnessReport, FreshnessOutcome, FreshnessState, WorkspaceFingerprint,
    DEFAULT_FRESHNESS_TTL, ENV_FRESHNESS_TTL_MS,
};
pub use incremental::{compute_file_diff, incremental_update, FileDiff};
pub use lock::{lock_path_for, try_acquire, try_lock, with_lock, Lock, LockError, LockOutcome};
