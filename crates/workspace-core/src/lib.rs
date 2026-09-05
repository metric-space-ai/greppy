//! Filesystem-independent copy-on-write workspaces.
//!
//! The core stores immutable and private file content in fixed-size,
//! content-addressed chunks. It deliberately requires only ordinary durable
//! file I/O from the host filesystem; mounting the namespace is delegated to
//! thin platform adapters.

mod chunk_store;
mod error;
mod namespace;
mod path_policy;
mod provider;
mod repository_layers;
mod repository_tracker;
mod repository_tracker_service;
mod snapshot;

pub use chunk_store::{ChunkGcReport, ChunkId, ChunkStore, ChunkStoreStats, CHUNK_SIZE};
pub use error::{Error, ErrorKind, Result};
pub use namespace::{
    DirectoryEntry, NodeKind, NodeMetadata, ProposalRecord, WorkspaceCore, WorkspaceFileHandle,
    WorkspaceHandle, WorkspaceOperationLease, WorkspacePairLease, WorkspaceStatus,
};
pub use provider::{
    AdapterKind, ProviderCapabilities, ProviderDiagnosticCheck, ProviderDiagnostics,
    ProviderInstallation, ProviderManifest, ProviderState,
    PROVIDER_PROTOCOL_VERSION,
};
pub use repository_tracker::{
    RepositoryChangeBatch, RepositoryTrackerState, RepositoryTrackerStatus,
};
pub use repository_tracker_service::{spawn_repository_tracker, spawn_repository_tracker_for};
pub use snapshot::{
    capture_overlay_directory, capture_repository, capture_repository_incremental,
    capture_repository_with_observer, BaselineDirectory, BaselineEntry, BaselineSnapshot,
    EntryKind,
};

pub(crate) fn verify_sqlite_integrity(
    connection: &rusqlite::Connection,
    subject: &str,
) -> Result<()> {
    let mut quick_check = connection.prepare("PRAGMA quick_check")?;
    let mut rows = quick_check.query([])?;
    let first = rows
        .next()?
        .ok_or_else(|| Error::Corrupt(format!("{subject} quick_check returned no result")))?
        .get::<_, String>(0)?;
    if first != "ok" || rows.next()?.is_some() {
        return Err(Error::Corrupt(format!(
            "{subject} quick_check failed: {first}"
        )));
    }

    let mut foreign_keys = connection.prepare("PRAGMA foreign_key_check")?;
    let mut violations = foreign_keys.query([])?;
    if let Some(row) = violations.next()? {
        let table: String = row.get(0)?;
        let row_id: Option<i64> = row.get(1)?;
        let parent: String = row.get(2)?;
        return Err(Error::Corrupt(format!(
            "{subject} foreign key violation: table={table}, row={row_id:?}, parent={parent}"
        )));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn test_crash_point(point: &str) {
    if std::env::var_os("GREPPY_WORKSPACE_TEST_CRASH_POINT").as_deref()
        == Some(std::ffi::OsStr::new(point))
    {
        std::process::abort();
    }
}
