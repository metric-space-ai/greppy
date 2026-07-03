//! Vector-search primitives.
//!
//! This module deliberately exposes a small exact-NN primitive over the store's
//! persisted embeddings. Hybrid ranking lives above this layer; this layer's
//! job is to prove that R5 uses numeric vector similarity, not token lookup.

use grepplus_core::{Error, Result};
use grepplus_embeddinggemma::{EmbedTask, EmbeddingGemma, CODE_RETRIEVAL_PROFILE, PROMPT_VERSION};
use grepplus_store::{Store, VectorSearchHit, VectorSearchQuery};

/// Store task/profile key for code-block retrieval.
///
/// Query vectors use EmbeddingGemma's `code retrieval` prompt. Stored code
/// vectors use EmbeddingGemma's retrieval-document prompt (`title: ... | text:
/// ...`). The shared key names the retrieval profile, not a single prompt.
pub const EMBEDDINGGEMMA_CODE_RETRIEVAL_PROFILE: &str = CODE_RETRIEVAL_PROFILE;

/// Default safety cap for the exact vector backend.
///
/// Exact cosine search is intentionally available before an ANN index exists,
/// but production callers must not accidentally scan very large repositories.
/// The CLI can override or disable this guard explicitly.
pub const DEFAULT_EXACT_VECTOR_CANDIDATE_LIMIT: i64 = 50_000;

/// Run exact cosine nearest-neighbor search over persisted embeddings.
pub fn vector_search_exact(
    store: &Store,
    query_vector: &[f32],
    query: &VectorSearchQuery<'_>,
) -> Result<Vec<VectorSearchHit>> {
    Ok(store.vector_search_exact(query_vector, query)?)
}

/// Count embeddings in the same scope used by [`vector_search_exact`].
pub fn count_vector_search_scope(store: &Store, query: &VectorSearchQuery<'_>) -> Result<i64> {
    Ok(store.count_vector_embeddings(
        query.project,
        query.model_id,
        query.prompt_version,
        query.task,
        query.graph_generation,
    )?)
}

/// Build the canonical vector-search scope for EmbeddingGemma code retrieval.
pub fn embeddinggemma_code_retrieval_scope<'a>(
    project: &'a str,
    model_id: &'a str,
    graph_generation: Option<u64>,
    limit: usize,
) -> VectorSearchQuery<'a> {
    VectorSearchQuery {
        project,
        model_id,
        prompt_version: PROMPT_VERSION,
        task: EMBEDDINGGEMMA_CODE_RETRIEVAL_PROFILE,
        graph_generation,
        file_path: None,
        limit,
        min_score: None,
    }
}

/// Embed a natural-language query for code retrieval.
pub fn embed_code_query(model: &EmbeddingGemma, query: &str) -> Result<Vec<f32>> {
    model
        .embed_one(EmbedTask::CodeRetrievalQuery, query)
        .map_err(|e| Error::Store(format!("embeddinggemma query embedding: {e}")))
}

/// Embed a real code span/document for the code-retrieval profile.
pub fn embed_code_document(
    model: &EmbeddingGemma,
    title: Option<&str>,
    content: &str,
) -> Result<Vec<f32>> {
    model
        .embed_document(title, content)
        .map_err(|e| Error::Store(format!("embeddinggemma document embedding: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use grepplus_store::{NewVectorEmbedding, Project};

    fn store_with_project(name: &str) -> Store {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: name.into(),
            indexed_at: "2026-07-01T00:00:00Z".into(),
            root_path: format!("/repos/{name}"),
        })
        .unwrap();
        s
    }

    fn embedding(qname: &str, vector: Vec<f32>) -> NewVectorEmbedding {
        let digest = match qname {
            "p.refundPayment" => "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "p.cancelInvoice" => "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            _ => "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
        };
        NewVectorEmbedding {
            project: "p".into(),
            model_id: "google/embeddinggemma-300m-q4".into(),
            prompt_version: PROMPT_VERSION.into(),
            task: EMBEDDINGGEMMA_CODE_RETRIEVAL_PROFILE.into(),
            node_id: None,
            qualified_name: qname.into(),
            file_path: "src/lib.rs".into(),
            start_line: 1,
            end_line: 2,
            content_sha256: digest.into(),
            graph_generation: 11,
            vector,
        }
    }

    fn query<'a>() -> VectorSearchQuery<'a> {
        VectorSearchQuery {
            project: "p",
            model_id: "google/embeddinggemma-300m-q4",
            prompt_version: PROMPT_VERSION,
            task: EMBEDDINGGEMMA_CODE_RETRIEVAL_PROFILE,
            graph_generation: Some(11),
            file_path: None,
            limit: 10,
            min_score: None,
        }
    }

    #[test]
    fn vector_search_is_numeric_similarity_not_name_matching() {
        let mut s = store_with_project("p");
        s.upsert_vector_embedding(&embedding("p.refundPayment", vec![1.0, 0.0]))
            .unwrap();
        s.upsert_vector_embedding(&embedding("p.cancelInvoice", vec![0.0, 1.0]))
            .unwrap();

        let hits = vector_search_exact(&s, &[0.0, 1.0], &query()).unwrap();
        assert_eq!(hits[0].embedding.qualified_name, "p.cancelInvoice");
        assert_eq!(
            count_vector_search_scope(&s, &query()).unwrap(),
            2,
            "count must reflect stored vectors, not the display limit"
        );
    }

    #[test]
    fn embeddinggemma_profile_uses_model_prompt_contract() {
        let scope =
            embeddinggemma_code_retrieval_scope("p", "google/embeddinggemma-300m-q4", Some(7), 5);
        assert_eq!(scope.prompt_version, PROMPT_VERSION);
        assert_eq!(scope.task, EMBEDDINGGEMMA_CODE_RETRIEVAL_PROFILE);
        assert_eq!(
            EmbedTask::CodeRetrievalQuery.prompt("find retry handler"),
            "task: code retrieval | query: find retry handler"
        );
        assert_eq!(
            EmbedTask::document_with_title(Some("src/payments.rs"), "fn refund() {}"),
            "title: src/payments.rs | text: fn refund() {}"
        );
    }
}
