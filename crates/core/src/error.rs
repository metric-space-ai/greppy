//! Public error type for `grepplus-rs`.
//!
//! The variants are exhaustive: callers must pick the variant that matches
//! the failure mode. There is no `Other(String)` escape hatch on the
//! public API; new variants are added when new categories surface.

use std::path::PathBuf;
use thiserror::Error;

/// Result alias used across the workspace.
pub type Result<T> = std::result::Result<T, Error>;

/// Public error type.
///
/// Each variant maps to a category of failure. User-facing CLI output
/// uses the `Display` impl; structured logging uses the variant name.
#[derive(Debug, Error)]
pub enum Error {
    /// Filesystem I/O failure.
    #[error("io: {context}: {source}")]
    Io {
        context: String,
        #[source]
        source: std::io::Error,
    },

    /// A path could not be found.
    #[error("not found: {0}")]
    NotFound(PathBuf),

    /// A feature is recognised but not yet implemented in this phase.
    #[error("not implemented in this phase: {feature} ({reason})")]
    NotImplemented { feature: String, reason: String },

    /// A feature is explicitly out of scope for the project.
    #[error("out of scope for grepplus-rs: {feature}")]
    OutOfScope { feature: String },

    /// A request was malformed.
    #[error("invalid request: {0}")]
    Invalid(String),

    /// A parsing failure (Cypher, tree-sitter, regex, …).
    #[error("parse error: {0}")]
    Parse(String),

    /// The graph store returned an error (SQLite, transaction, schema, …).
    #[error("store error: {0}")]
    Store(String),

    /// An indexing pass failed.
    #[error("index error: {0}")]
    Index(String),

    /// Workspace fingerprint / lock / freshness failure.
    #[error("workspace error: {0}")]
    Workspace(String),

    /// A lock could not be acquired in time; downstream must degrade.
    #[error("lock contention: {0}")]
    Lock(String),

    /// A time budget was exceeded; downstream must degrade.
    #[error("budget exceeded: {0}")]
    Budget(String),

    /// Configuration error (env var, file).
    #[error("config error: {0}")]
    Config(String),
}

impl Error {
    /// Convenience constructor for `Error::Io`.
    pub fn io(context: impl Into<String>, source: std::io::Error) -> Self {
        Self::Io {
            context: context.into(),
            source,
        }
    }

    /// Convenience constructor for `Error::NotImplemented`.
    pub fn not_implemented(feature: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::NotImplemented {
            feature: feature.into(),
            reason: reason.into(),
        }
    }

    /// Convenience constructor for `Error::OutOfScope`.
    pub fn out_of_scope(feature: impl Into<String>) -> Self {
        Self::OutOfScope {
            feature: feature.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_messages_are_stable() {
        // These strings are part of the CLI contract. Tests catch silent edits.
        assert_eq!(
            Error::not_implemented("grepplus foo", "phase 1").to_string(),
            "not implemented in this phase: grepplus foo (phase 1)"
        );
        assert_eq!(
            Error::out_of_scope("install").to_string(),
            "out of scope for grepplus-rs: install"
        );
        assert_eq!(
            Error::io("open file", std::io::Error::other("boom")).to_string(),
            "io: open file: boom"
        );
    }

    #[test]
    fn out_of_scope_is_a_distinct_variant_from_not_implemented() {
        // A future caller might want to distinguish "we plan to do this"
        // from "we will never do this"; ensure the variant is distinct.
        let n = Error::not_implemented("foo", "bar");
        let o = Error::out_of_scope("install");
        assert!(matches!(n, Error::NotImplemented { .. }));
        assert!(matches!(o, Error::OutOfScope { .. }));
        assert!(std::mem::discriminant(&n) != std::mem::discriminant(&o));
    }
}
