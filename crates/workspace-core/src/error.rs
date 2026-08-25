use std::path::PathBuf;

/// Errors are fail-closed: callers must not create or expose a workspace after
/// any of these conditions.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("workspace storage I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("workspace metadata failed: {0}")]
    Sql(#[from] rusqlite::Error),
    #[error("workspace metadata serialization failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("not a Git repository: {path}: {detail}")]
    NotGitRepository { path: PathBuf, detail: String },
    #[error("Git command failed: {command}: {detail}")]
    Git { command: String, detail: String },
    #[error("repository state changed while the immutable baseline was captured")]
    ConcurrentRepositoryMutation,
    #[error("unsupported repository state: {0}")]
    UnsupportedRepository(String),
    #[error("workspace storage is corrupt: {0}")]
    Corrupt(String),
    #[error("invalid workspace path: {0}")]
    InvalidPath(String),
}

pub type Result<T> = std::result::Result<T, Error>;
