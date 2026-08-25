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
    #[error("workspace path does not exist: {0}")]
    NotFound(String),
    #[error("workspace path already exists: {0}")]
    AlreadyExists(String),
    #[error("workspace path is not a directory: {0}")]
    NotDirectory(String),
    #[error("workspace path is a directory: {0}")]
    IsDirectory(String),
    #[error("workspace directory is not empty: {0}")]
    DirectoryNotEmpty(String),
    #[error("portable workspace adapter is unavailable: {0}")]
    AdapterUnavailable(String),
    #[error("portable workspace adapter is unhealthy: {0}")]
    AdapterUnhealthy(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    NotFound,
    AlreadyExists,
    NotDirectory,
    IsDirectory,
    DirectoryNotEmpty,
    InvalidInput,
    Unavailable,
    Corrupt,
    Io,
}

impl Error {
    pub fn kind(&self) -> ErrorKind {
        match self {
            Self::NotFound(_) => ErrorKind::NotFound,
            Self::AlreadyExists(_) => ErrorKind::AlreadyExists,
            Self::NotDirectory(_) => ErrorKind::NotDirectory,
            Self::IsDirectory(_) => ErrorKind::IsDirectory,
            Self::DirectoryNotEmpty(_) => ErrorKind::DirectoryNotEmpty,
            Self::InvalidPath(_)
            | Self::UnsupportedRepository(_)
            | Self::ConcurrentRepositoryMutation
            | Self::NotGitRepository { .. }
            | Self::Git { .. } => ErrorKind::InvalidInput,
            Self::AdapterUnavailable(_) | Self::AdapterUnhealthy(_) => ErrorKind::Unavailable,
            Self::Corrupt(_) | Self::Sql(_) | Self::Json(_) => ErrorKind::Corrupt,
            Self::Io(_) => ErrorKind::Io,
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
