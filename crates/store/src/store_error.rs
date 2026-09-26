//! Public Error type for `greppy-store`.
//!
//! Wraps `greppy_core::Error` with store-specific variants. Public
//! functions in this crate return `Result<T, Error>`.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("lock contention: {0}")]
    Lock(String),

    #[error("store: {0}")]
    Store(String),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("invalid input: {0}")]
    Invalid(String),

    #[error("io: {context}: {source}")]
    Io {
        context: String,
        #[source]
        source: std::io::Error,
    },

    #[error("sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<greppy_core::Error> for Error {
    fn from(e: greppy_core::Error) -> Self {
        match e {
            greppy_core::Error::Lock(s) => Error::Lock(s),
            greppy_core::Error::Store(s) => Error::Store(s),
            greppy_core::Error::NotFound(p) => Error::NotFound(p.display().to_string()),
            greppy_core::Error::Io { context, source } => Error::Io { context, source },
            other => Error::Store(other.to_string()),
        }
    }
}

impl From<Error> for greppy_core::Error {
    fn from(e: Error) -> Self {
        match e {
            Error::Lock(s) => greppy_core::Error::Lock(s),
            Error::Store(s) => greppy_core::Error::Store(s),
            Error::NotFound(s) => greppy_core::Error::NotFound(std::path::PathBuf::from(s)),
            Error::Invalid(s) => greppy_core::Error::Invalid(s),
            Error::Io { context, source } => greppy_core::Error::Io { context, source },
            Error::Sqlite(e) => greppy_core::Error::Store(format!("sqlite: {e}")),
            Error::Json(e) => greppy_core::Error::Store(format!("json: {e}")),
        }
    }
}
