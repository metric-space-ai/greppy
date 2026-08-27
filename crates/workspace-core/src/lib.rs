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
mod snapshot;

pub use chunk_store::{ChunkGcReport, ChunkId, ChunkStore, ChunkStoreStats, CHUNK_SIZE};
pub use error::{Error, ErrorKind, Result};
pub use namespace::{
    DirectoryEntry, NodeKind, NodeMetadata, ProposalRecord, WorkspaceCore, WorkspaceFileHandle,
    WorkspaceHandle, WorkspacePairLease, WorkspaceStatus,
};
pub use provider::{
    AdapterKind, ProviderCapabilities, ProviderInstallation, ProviderManifest, ProviderState,
    PROVIDER_PROTOCOL_VERSION,
};
pub use repository_tracker::{
    RepositoryChangeBatch, RepositoryTrackerState, RepositoryTrackerStatus,
};
pub use snapshot::{
    capture_overlay_directory, capture_repository, capture_repository_incremental,
    capture_repository_with_observer, BaselineDirectory, BaselineEntry, BaselineSnapshot,
    EntryKind,
};

#[cfg(test)]
pub(crate) fn test_crash_point(point: &str) {
    if std::env::var_os("GREPPY_WORKSPACE_TEST_CRASH_POINT").as_deref()
        == Some(std::ffi::OsStr::new(point))
    {
        std::process::abort();
    }
}
