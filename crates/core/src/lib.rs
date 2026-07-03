//! `grepplus-core` — shared types and error definitions for the workspace.
//!
//! This crate hosts:
//! - [`Error`] — the public error type used by all other crates. It
//!   does not use `anyhow` on the public API; conversion into `anyhow` is
//!   the caller's choice.
//! - [`Result`] — shorthand for `Result<T, Error>`.
//! - [`logging`] — tracing initialisation used by the `cli` crate and
//!   integration tests.
//!
//! What this crate is **not**:
//! - It does **not** contain placeholder, mock, demo, or fake modules. If
//!   you find a `mod demo` or `mod mock` here, it is a bug.

#![deny(rust_2018_idioms)]
#![warn(missing_debug_implementations)]

/// Stable base identifier for Rust indexer output stored in
/// `workspace_state.indexer_version`.
pub const INDEXER_VERSION_BASE: &str = "grepplus-indexer-v1";

pub mod diag;
pub mod error;
pub mod git_fingerprint;
pub mod logging;
pub mod membudget;
pub mod profile;
pub mod strutil;
pub mod sysinfo;
pub mod validate;
pub mod workspace;

pub use crate::diag::{Diagnostics, Snapshot, SAMPLE_CAPACITY};
pub use crate::error::{Error, Result};
pub use crate::git_fingerprint::GitFingerprint;
pub use crate::membudget::{
    budget as mem_budget, init as mem_budget_init, over_budget as mem_over_budget, rss as mem_rss,
    worker_budget as mem_worker_budget,
};
pub use crate::profile::{
    enable as profile_enable, init as profile_init, is_active as profile_is_active,
    span as profile_span, ProfileSpan, Profiler, Span as ProfileScope, Started as ProfileStarted,
    PROFILE_ENV,
};
pub use crate::sysinfo::{
    default_worker_count, logical_cpu_count, physical_cpu_count, system_info, total_ram, SystemInfo,
};
pub use crate::validate::{
    is_valid_project_name, is_valid_shell_arg, json_escape, validate_project_name,
    validate_shell_arg, ProjectNameError, ShellArgError,
};
pub use crate::workspace::{project_identity, store_dir, store_path, workspace_hash};
