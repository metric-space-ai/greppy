//! `grepplus-store` — SQLite-backed graph store with FTS5 full-text search.
//!
//! This crate implements the minimum graph model described in
//! `docs/grepplus_rust_phasenplan_port.md` §7:
//!
//! ```text
//! Node:
//!   id, project, label, name, qualified_name, file_path,
//!   start_line, end_line, properties_json
//!
//! Edge:
//!   id, project, source_id, target_id, type, properties_json
//!
//! FileState:
//!   project, rel_path, sha256, mtime_ns, size,
//!   parser_version, extractor_version, last_indexed_generation
//!
//! WorkspaceState:
//!   root_path, git_dir, git_common_dir, head_oid, index_signature,
//!   schema_version, indexer_version, graph_generation, updated_at
//! ```
//!
//! Plus FTS5 contentless virtual table for BM25 lexical search.
//!
//! Phase 2 implements:
//! - schema creation (idempotent),
//! - schema migration version table,
//! - node/edge/file_state/workspace_state CRUD,
//! - FTS5 insert / search,
//! - integrity check,
//! - golden-master round-trip test.

#![deny(rust_2018_idioms)]

pub mod diagnostics;
pub mod edge;
pub mod file_content;
pub mod file_state;
pub mod fts;
pub mod index_skip;
pub mod maintenance;
pub mod migrate;
pub mod node;
pub mod project;
pub mod project_summary;
pub mod provider_state;
pub mod query_cache;
pub mod raw_edge;
pub mod schema;
pub mod stats;
pub mod store;
pub mod store_error;
pub mod vector_embedding;
pub mod workspace_state;

pub use diagnostics::{ProjectDiagnostics, StoreDiagnostics};
pub use edge::{Edge, NewEdge};
pub use file_content::{ContentRow, FileContentHit};
pub use file_state::FileState;
pub use index_skip::{IndexSkip, IndexSkipReasonCount};
pub use migrate::{Migration, MIGRATIONS};
pub use node::{NewNode, Node};
pub use project::Project;
pub use project_summary::ProjectSummary;
pub use provider_state::ProviderState;
pub use query_cache::{normalize_query_text, QueryEmbeddingCache, QUERY_CACHE_DB_FILE};
pub use raw_edge::{NewRawEdge, RawEdge};
pub use stats::{EdgeTypeCount, GraphStats, LabelCount};
pub use store::{OpenOptions, Store};
pub use store_error::{Error, Result};
pub use vector_embedding::{
    NewVectorEmbedding, VectorEmbedding, VectorSearchHit, VectorSearchQuery,
};
pub use workspace_state::WorkspaceState;
