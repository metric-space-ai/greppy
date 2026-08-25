//! Filesystem-independent copy-on-write workspaces.
//!
//! The core stores immutable and private file content in fixed-size,
//! content-addressed chunks. It deliberately requires only ordinary durable
//! file I/O from the host filesystem; mounting the namespace is delegated to
//! thin platform adapters.

mod chunk_store;
mod error;
mod namespace;
mod provider;
mod snapshot;

pub use chunk_store::{ChunkGcReport, ChunkId, ChunkStore, ChunkStoreStats, CHUNK_SIZE};
pub use error::{Error, Result};
pub use namespace::{
    DirectoryEntry, NodeKind, NodeMetadata, ProposalRecord, WorkspaceCore, WorkspaceHandle,
    WorkspaceStatus,
};
pub use provider::{
    AdapterKind, ProviderCapabilities, ProviderInstallation, ProviderManifest, ProviderState,
    PROVIDER_PROTOCOL_VERSION,
};
pub use snapshot::{capture_repository, BaselineEntry, BaselineSnapshot, EntryKind};
