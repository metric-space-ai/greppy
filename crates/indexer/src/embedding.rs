//! Symbol/code-span embedding indexing.
//!
//! This module is the R5 bridge between graph indexing and vector search. It
//! does not invent summaries or Markdown context: every stored vector is derived
//! from the exact source span of a persisted graph node.

use std::collections::HashMap;
use std::path::Path;

use grepplus_core::{Error, Result};
use grepplus_embeddinggemma::{EmbeddingGemma, CODE_RETRIEVAL_PROFILE, PROMPT_VERSION};
use grepplus_store::{file_state::sha256_hex, NewVectorEmbedding, Store};

const NODE_PAGE_SIZE: usize = 1000;
const DEFAULT_MAX_SPAN_BYTES: usize = 32 * 1024;

/// Provider interface used by the indexer to embed real code spans.
pub trait CodeEmbeddingProvider {
    fn model_id(&self) -> &str;
    fn prompt_version(&self) -> &str;
    fn task_profile(&self) -> &str;
    fn embed_code_document(&mut self, title: Option<&str>, content: &str) -> Result<Vec<f32>>;
}

/// Production provider backed by native EmbeddingGemma inference.
pub struct EmbeddingGemmaCodeProvider<'a> {
    model_id: String,
    model: &'a EmbeddingGemma,
}

impl<'a> EmbeddingGemmaCodeProvider<'a> {
    pub fn new(model_id: impl Into<String>, model: &'a EmbeddingGemma) -> Self {
        Self {
            model_id: model_id.into(),
            model,
        }
    }
}

impl CodeEmbeddingProvider for EmbeddingGemmaCodeProvider<'_> {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn prompt_version(&self) -> &str {
        PROMPT_VERSION
    }

    fn task_profile(&self) -> &str {
        CODE_RETRIEVAL_PROFILE
    }

    fn embed_code_document(&mut self, title: Option<&str>, content: &str) -> Result<Vec<f32>> {
        self.model
            .embed_document(title, content)
            .map_err(|e| Error::Store(format!("embeddinggemma document embedding: {e}")))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddingIndexOptions {
    pub graph_generation: u64,
    pub max_span_bytes: usize,
    pub prune_before_generation: bool,
}

impl EmbeddingIndexOptions {
    pub fn for_generation(graph_generation: u64) -> Self {
        Self {
            graph_generation,
            max_span_bytes: DEFAULT_MAX_SPAN_BYTES,
            prune_before_generation: true,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EmbeddingIndexReport {
    pub nodes_considered: usize,
    pub nodes_embedded: usize,
    pub nodes_skipped_non_definition: usize,
    pub nodes_skipped_missing_file: usize,
    pub nodes_skipped_invalid_span: usize,
    pub nodes_skipped_oversize: usize,
    pub stale_rows_pruned: usize,
}

/// Index vectors for all embeddable symbol nodes in `project`.
pub fn index_code_embeddings_for_project(
    store: &mut Store,
    root: &Path,
    project: &str,
    provider: &mut dyn CodeEmbeddingProvider,
    options: EmbeddingIndexOptions,
) -> Result<EmbeddingIndexReport> {
    let mut report = EmbeddingIndexReport::default();
    let mut file_cache: HashMap<String, Option<String>> = HashMap::new();
    let mut offset = 0usize;

    loop {
        let nodes = store.list_nodes(project, "", "", offset, NODE_PAGE_SIZE)?;
        if nodes.is_empty() {
            break;
        }
        offset += nodes.len();

        for node in nodes {
            report.nodes_considered += 1;
            if !is_embedding_candidate_label(&node.label) {
                report.nodes_skipped_non_definition += 1;
                continue;
            }

            let source = match cached_file_source(&mut file_cache, root, &node.file_path) {
                Ok(Some(source)) => source,
                Ok(None) => {
                    report.nodes_skipped_missing_file += 1;
                    continue;
                }
                Err(_) => {
                    report.nodes_skipped_missing_file += 1;
                    continue;
                }
            };
            let Some(span) = source_span(source, node.start_line, node.end_line) else {
                report.nodes_skipped_invalid_span += 1;
                continue;
            };
            if span.len() > options.max_span_bytes {
                report.nodes_skipped_oversize += 1;
                continue;
            }

            let title = format!(
                "{}:{}-{} {}",
                node.file_path, node.start_line, node.end_line, node.qualified_name
            );
            let vector = provider.embed_code_document(Some(&title), &span)?;
            store.upsert_vector_embedding(&NewVectorEmbedding {
                project: node.project,
                model_id: provider.model_id().to_string(),
                prompt_version: provider.prompt_version().to_string(),
                task: provider.task_profile().to_string(),
                node_id: Some(node.id),
                qualified_name: node.qualified_name,
                file_path: node.file_path,
                start_line: node.start_line,
                end_line: node.end_line,
                content_sha256: sha256_hex(span.as_bytes()),
                graph_generation: options.graph_generation,
                vector,
            })?;
            report.nodes_embedded += 1;
        }
    }

    if options.prune_before_generation {
        report.stale_rows_pruned =
            store.prune_vector_embeddings_before_generation(project, options.graph_generation)?;
    }
    Ok(report)
}

fn cached_file_source<'a>(
    cache: &'a mut HashMap<String, Option<String>>,
    root: &Path,
    rel_path: &str,
) -> Result<Option<&'a str>> {
    if !cache.contains_key(rel_path) {
        let path = root.join(rel_path);
        let source = match std::fs::read_to_string(&path) {
            Ok(s) => Some(s),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => {
                return Err(Error::io(format!("read {}", path.display()), e));
            }
        };
        cache.insert(rel_path.to_string(), source);
    }
    Ok(cache.get(rel_path).and_then(|s| s.as_deref()))
}

fn source_span(source: &str, start_line: i64, end_line: i64) -> Option<String> {
    if start_line <= 0 || end_line < start_line {
        return None;
    }
    let start = start_line as usize;
    let end = end_line as usize;
    let mut out = String::new();
    for (idx, line) in source.lines().enumerate() {
        let line_no = idx + 1;
        if line_no < start {
            continue;
        }
        if line_no > end {
            break;
        }
        out.push_str(line);
        out.push('\n');
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

fn is_embedding_candidate_label(label: &str) -> bool {
    matches!(
        label,
        "Function"
            | "Method"
            | "Class"
            | "Interface"
            | "Type"
            | "Enum"
            | "Struct"
            | "Trait"
            | "TypeAlias"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use grepplus_store::{NewNode, Project, VectorSearchQuery};

    struct DeterministicProvider;

    impl CodeEmbeddingProvider for DeterministicProvider {
        fn model_id(&self) -> &str {
            "test-code-embedder"
        }

        fn prompt_version(&self) -> &str {
            "test-prompt-v1"
        }

        fn task_profile(&self) -> &str {
            "embeddinggemma_code_retrieval"
        }

        fn embed_code_document(&mut self, _title: Option<&str>, content: &str) -> Result<Vec<f32>> {
            if content.contains("refund") {
                Ok(vec![0.0, 1.0])
            } else {
                Ok(vec![1.0, 0.0])
            }
        }
    }

    fn tempdir_via_env() -> std::path::PathBuf {
        let base = std::env::temp_dir();
        let unique = format!(
            "grepplus-indexer-embedding-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let p = base.join(unique);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn store_with_project(root: &Path) -> Store {
        let mut s = Store::open_memory().unwrap();
        s.upsert_project(&Project {
            name: "p".into(),
            indexed_at: "2026-07-01T00:00:00Z".into(),
            root_path: root.to_string_lossy().into_owned(),
        })
        .unwrap();
        s
    }

    fn insert_node(
        store: &mut Store,
        qname: &str,
        name: &str,
        label: &str,
        file_path: &str,
        start_line: i64,
        end_line: i64,
    ) {
        store
            .insert_node(&NewNode {
                project: "p".into(),
                label: label.into(),
                name: name.into(),
                qualified_name: qname.into(),
                file_path: file_path.into(),
                start_line,
                end_line,
                properties: serde_json::json!({}),
            })
            .unwrap();
    }

    #[test]
    fn indexes_real_symbol_spans_into_vector_store() {
        let root = tempdir_via_env();
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(
            root.join("src/payments.rs"),
            "pub fn charge() {\n    settle();\n}\n\npub fn refund_payment() {\n    refund();\n}\n",
        )
        .unwrap();
        let mut store = store_with_project(&root);
        insert_node(
            &mut store,
            "p.charge",
            "charge",
            "Function",
            "src/payments.rs",
            1,
            3,
        );
        insert_node(
            &mut store,
            "p.refund_payment",
            "refund_payment",
            "Function",
            "src/payments.rs",
            5,
            7,
        );
        insert_node(
            &mut store,
            "src/payments.rs::__file__",
            "__file__",
            "Module",
            "src/payments.rs",
            1,
            1,
        );

        let mut provider = DeterministicProvider;
        let report = index_code_embeddings_for_project(
            &mut store,
            &root,
            "p",
            &mut provider,
            EmbeddingIndexOptions::for_generation(4),
        )
        .unwrap();

        assert_eq!(report.nodes_considered, 3);
        assert_eq!(report.nodes_embedded, 2);
        assert_eq!(report.nodes_skipped_non_definition, 1);
        let hits = store
            .vector_search_exact(
                &[0.0, 1.0],
                &VectorSearchQuery {
                    project: "p",
                    model_id: "test-code-embedder",
                    prompt_version: "test-prompt-v1",
                    task: "embeddinggemma_code_retrieval",
                    graph_generation: Some(4),
                    file_path: None,
                    limit: 2,
                    min_score: None,
                },
            )
            .unwrap();
        assert_eq!(hits[0].embedding.qualified_name, "p.refund_payment");
        assert_eq!(hits.len(), 2);
        assert!(hits.iter().all(|h| h.embedding.graph_generation == 4));
    }

    #[test]
    fn embedding_index_prunes_only_after_successful_generation() {
        let root = tempdir_via_env();
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/lib.rs"), "pub fn current() {}\n").unwrap();
        let mut store = store_with_project(&root);
        insert_node(
            &mut store,
            "p.current",
            "current",
            "Function",
            "src/lib.rs",
            1,
            1,
        );
        store
            .upsert_vector_embedding(&NewVectorEmbedding {
                project: "p".into(),
                model_id: "test-code-embedder".into(),
                prompt_version: "test-prompt-v1".into(),
                task: "embeddinggemma_code_retrieval".into(),
                node_id: None,
                qualified_name: "p.old".into(),
                file_path: "src/old.rs".into(),
                start_line: 1,
                end_line: 1,
                content_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    .into(),
                graph_generation: 1,
                vector: vec![1.0, 0.0],
            })
            .unwrap();

        let mut provider = DeterministicProvider;
        let report = index_code_embeddings_for_project(
            &mut store,
            &root,
            "p",
            &mut provider,
            EmbeddingIndexOptions::for_generation(5),
        )
        .unwrap();
        assert_eq!(report.nodes_embedded, 1);
        assert_eq!(report.stale_rows_pruned, 1);
        assert_eq!(
            store
                .count_vector_embeddings(
                    "p",
                    "test-code-embedder",
                    "test-prompt-v1",
                    "embeddinggemma_code_retrieval",
                    None
                )
                .unwrap(),
            1
        );
    }
}
