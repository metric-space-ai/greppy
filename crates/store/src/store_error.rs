//! Public Error type for `grepplus-store`.
//!
//! Wraps `grepplus_core::Error` with store-specific variants. Public
//! functions in this crate return `Result<T, Error>`.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
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

impl From<grepplus_core::Error> for Error {
    fn from(e: grepplus_core::Error) -> Self {
        match e {
            grepplus_core::Error::Store(s) => Error::Store(s),
            grepplus_core::Error::NotFound(p) => Error::NotFound(p.display().to_string()),
            grepplus_core::Error::Io { context, source } => Error::Io { context, source },
            other => Error::Store(other.to_string()),
        }
    }
}

impl From<Error> for grepplus_core::Error {
    fn from(e: Error) -> Self {
        match e {
            Error::Store(s) => grepplus_core::Error::Store(s),
            Error::NotFound(s) => grepplus_core::Error::NotFound(std::path::PathBuf::from(s)),
            Error::Invalid(s) => grepplus_core::Error::Invalid(s),
            Error::Io { context, source } => grepplus_core::Error::Io { context, source },
            Error::Sqlite(e) => grepplus_core::Error::Store(format!("sqlite: {e}")),
            Error::Json(e) => grepplus_core::Error::Store(format!("json: {e}")),
        }
    }
}
