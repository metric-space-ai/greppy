//! `grepplus-search` — graph search, lexical (BM25/FTS) search, and
//! algorithmic semantic search.
//!
//! Phase 4 ships:
//! - [`search_graph`] — structured queries by label, name, qualified
//!   name, file path glob, and degree.
//! - [`trace_path`] — incoming/outgoing call-path traversal with bounded
//!   depth and visited-set.
//! - [`search_code`] — BM25 over the FTS5 contentless table.
//! - [`semantic_query`] — algorithmic 11-signal similarity (subset of
//!   the upstream's design).

#![deny(rust_2018_idioms)]

pub mod graph;
pub mod lexical;
pub mod semantic;
pub mod simhash;
pub mod trace;
pub mod vector;

pub use graph::{
    co_change_candidates, count_fan_in, count_fan_out, count_references, count_references_to_any,
    count_search_graph, cycles, definition_at, dependency_cluster, fan_in, fan_out,
    find_by_label_and_file, find_references, find_references_to_any, impact_radius, most_connected,
    neighbors, path_query, reachable_within, search_graph, subgraph_around, symbols_in_file,
    unused_symbols, CoChangeCandidate, Cycle, DegreeKind, DegreeRanked, DependencyCluster,
    GraphOrder, GraphPath, GraphQuery, ImpactNode, ReachDirection, ReachableNode, Reference,
    Subgraph, SubgraphEdge, MAX_CYCLE_LEN, MAX_REACH_HOPS, MAX_REACH_RESULTS, REFERENCE_EDGE_TYPES,
};
pub use lexical::{
    count_symbols_in_project, search_code, search_code_ranked, search_symbols,
    search_symbols_in_project, CodeHit, RankedCodeHit, SymbolHit,
};
pub use semantic::{
    expand_query_tokens, related_symbols, semantic_query, semantic_query_diversified,
    semantic_query_expanded, semantic_query_fielded, semantic_query_filtered,
    semantic_query_multi_anchor, DiversifiedHit, RelatedHit, SemanticBreakdown, SemanticField,
    SemanticHit, SemanticSignal, MAX_SEMANTIC_SCORE,
};
pub use simhash::{MinHash, MINHASH_K};
pub use trace::{
    call_hierarchy, call_tree, callees_of, callers_of, path_summary, trace_path, CallHierarchy,
    CallHierarchyNode, CallTree, CallTreeNode, TraceDirection, TraceStep,
};
pub use vector::{
    count_vector_search_scope, embed_code_document, embed_code_query,
    embeddinggemma_code_retrieval_scope, vector_search_exact, DEFAULT_EXACT_VECTOR_CANDIDATE_LIMIT,
    EMBEDDINGGEMMA_CODE_RETRIEVAL_PROFILE,
};
