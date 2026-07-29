//! Language-provider contract for the parser/indexer boundary.
//!
//! This module is the R0 handoff surface for parallel language work. A language
//! provider may be shallow or partial while it is being built, but it must say
//! so explicitly through [`ProviderManifest`] and through per-record confidence /
//! partial markers. The indexer and query layers can then expose provider
//! completeness instead of silently marketing partial extraction as full
//! language support.

use serde::{Deserialize, Serialize};

use crate::extract::{ExtractedEdge, ExtractedNode, ExtractionResult};
use crate::language::Language;

/// Edge or pass classes a language provider can claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeClass {
    Definitions,
    Imports,
    Calls,
    Usages,
    TypeRefs,
    TypeAssigns,
    Implements,
    Tests,
    Routes,
    EnvScan,
    InfraScan,
    K8s,
    PackageMap,
    ConfigLink,
    Configures,
    Complexity,
    GitDiff,
    GitHistory,
    CrossRepo,
    Semantic,
    SemanticEdges,
    Similarity,
}

impl EdgeClass {
    pub fn as_str(self) -> &'static str {
        match self {
            EdgeClass::Definitions => "definitions",
            EdgeClass::Imports => "imports",
            EdgeClass::Calls => "calls",
            EdgeClass::Usages => "usages",
            EdgeClass::TypeRefs => "type_refs",
            EdgeClass::TypeAssigns => "type_assigns",
            EdgeClass::Implements => "implements",
            EdgeClass::Tests => "tests",
            EdgeClass::Routes => "routes",
            EdgeClass::EnvScan => "envscan",
            EdgeClass::InfraScan => "infrascan",
            EdgeClass::K8s => "k8s",
            EdgeClass::PackageMap => "pkgmap",
            EdgeClass::ConfigLink => "configlink",
            EdgeClass::Configures => "configures",
            EdgeClass::Complexity => "complexity",
            EdgeClass::GitDiff => "gitdiff",
            EdgeClass::GitHistory => "githistory",
            EdgeClass::CrossRepo => "cross_repo",
            EdgeClass::Semantic => "semantic",
            EdgeClass::SemanticEdges => "semantic_edges",
            EdgeClass::Similarity => "similarity",
        }
    }
}

/// Provider-level completeness. This is intentionally separate from language
/// detection: a language can be detected while its provider is still partial.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderStatus {
    Unsupported,
    Partial,
    ParityCandidate,
    Accepted,
}

/// Stable manifest every language provider must expose.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderManifest {
    pub language: String,
    pub provider_version: String,
    pub status: ProviderStatus,
    pub file_extensions: Vec<String>,
    pub filenames: Vec<String>,
    pub supported_edge_classes: Vec<EdgeClass>,
    pub unsupported_edge_classes: Vec<EdgeClass>,
    pub fixture_paths: Vec<String>,
    pub golden_paths: Vec<String>,
    pub review_artifact: Option<String>,
    pub notes: Vec<String>,
}

impl ProviderManifest {
    pub fn supports(&self, class: EdgeClass) -> bool {
        self.supported_edge_classes.contains(&class)
    }

    /// A provider may only claim accepted when every claimed edge class has
    /// fixture + golden evidence and no unsupported class is left ambiguous.
    pub fn validate(&self) -> std::result::Result<(), ProviderContractError> {
        if self.language.trim().is_empty() {
            return Err(ProviderContractError::MissingField("language"));
        }
        if self.provider_version.trim().is_empty() {
            return Err(ProviderContractError::MissingField("provider_version"));
        }
        if !matches!(self.status, ProviderStatus::Unsupported)
            && self.file_extensions.is_empty()
            && self.filenames.is_empty()
        {
            return Err(ProviderContractError::MissingSelection);
        }
        for class in &self.supported_edge_classes {
            if self.unsupported_edge_classes.contains(class) {
                return Err(ProviderContractError::ContradictoryEdgeClass(*class));
            }
        }
        if matches!(self.status, ProviderStatus::Accepted)
            && (self.fixture_paths.is_empty()
                || self.golden_paths.is_empty()
                || self.review_artifact.is_none())
        {
            return Err(ProviderContractError::AcceptedWithoutEvidence);
        }
        Ok(())
    }
}

/// Node record on the provider boundary. This is richer than the legacy
/// `ExtractedNode` so callers can distinguish partial extraction from full
/// language support.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderNode {
    pub file_id: String,
    pub language: String,
    pub symbol_id: String,
    pub symbol_kind: String,
    pub name: String,
    pub qualified_name: String,
    pub span_start: u32,
    pub span_end: u32,
    pub definition_span: Option<(u32, u32)>,
    pub confidence: f32,
    pub is_partial: bool,
    pub diagnostics: Vec<String>,
    pub properties: serde_json::Value,
}

/// Edge record on the provider boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderEdge {
    pub file_id: String,
    pub language: String,
    pub edge_type: String,
    pub edge_source: String,
    pub edge_target: String,
    pub span_start: u32,
    pub span_end: u32,
    pub confidence: f32,
    pub is_partial: bool,
    pub diagnostics: Vec<String>,
    pub properties: serde_json::Value,
}

/// Complete provider output for one file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderOutput {
    pub manifest: ProviderManifest,
    pub file_id: String,
    pub language: String,
    pub nodes: Vec<ProviderNode>,
    pub edges: Vec<ProviderEdge>,
    pub diagnostics: Vec<String>,
}

impl ProviderOutput {
    pub fn from_extraction(
        manifest: ProviderManifest,
        language: Language,
        file_path: &str,
        extraction: ExtractionResult,
    ) -> Self {
        let language_name = language.name().to_string();
        let nodes = extraction
            .nodes
            .into_iter()
            .map(|n| ProviderNode::from_extracted(&language_name, file_path, n))
            .collect();
        let edges = extraction
            .edges
            .into_iter()
            .map(|e| ProviderEdge::from_extracted(&language_name, file_path, e))
            .collect();
        Self {
            manifest,
            file_id: file_path.to_string(),
            language: language_name,
            nodes,
            edges,
            diagnostics: Vec::new(),
        }
    }

    pub fn validate(&self) -> std::result::Result<(), ProviderContractError> {
        self.manifest.validate()?;
        if self.file_id.trim().is_empty() {
            return Err(ProviderContractError::MissingField("file_id"));
        }
        if self.language != self.manifest.language {
            return Err(ProviderContractError::LanguageMismatch {
                manifest: self.manifest.language.clone(),
                output: self.language.clone(),
            });
        }
        for node in &self.nodes {
            validate_span(node.span_start, node.span_end)?;
            validate_confidence(node.confidence)?;
            if node.file_id != self.file_id {
                return Err(ProviderContractError::FileMismatch {
                    expected: self.file_id.clone(),
                    found: node.file_id.clone(),
                });
            }
            if node.symbol_id.trim().is_empty()
                || node.qualified_name.trim().is_empty()
                || node.name.trim().is_empty()
            {
                return Err(ProviderContractError::MissingRecordIdentity);
            }
        }
        for edge in &self.edges {
            validate_span(edge.span_start, edge.span_end)?;
            validate_confidence(edge.confidence)?;
            if edge.file_id != self.file_id {
                return Err(ProviderContractError::FileMismatch {
                    expected: self.file_id.clone(),
                    found: edge.file_id.clone(),
                });
            }
            if edge.edge_type.trim().is_empty()
                || edge.edge_source.trim().is_empty()
                || edge.edge_target.trim().is_empty()
            {
                return Err(ProviderContractError::MissingRecordIdentity);
            }
        }
        Ok(())
    }
}

impl ProviderNode {
    fn from_extracted(language: &str, file_path: &str, node: ExtractedNode) -> Self {
        Self {
            file_id: file_path.to_string(),
            language: language.to_string(),
            symbol_id: node.qualified_name.clone(),
            symbol_kind: node.label.clone(),
            name: node.name,
            qualified_name: node.qualified_name,
            span_start: node.start_line,
            span_end: node.end_line,
            definition_span: Some((node.start_line, node.end_line)),
            confidence: 1.0,
            is_partial: false,
            diagnostics: Vec::new(),
            properties: node.properties,
        }
    }
}

impl ProviderEdge {
    fn from_extracted(language: &str, file_path: &str, edge: ExtractedEdge) -> Self {
        Self {
            file_id: file_path.to_string(),
            language: language.to_string(),
            edge_type: edge.edge_type,
            edge_source: edge.source_qualified_name,
            edge_target: edge.target_qualified_name,
            span_start: edge.line,
            span_end: edge.line,
            confidence: 1.0,
            is_partial: false,
            diagnostics: Vec::new(),
            properties: edge.properties,
        }
    }
}

/// Current manifest for a built-in/registered language. This does not claim
/// parity; it exposes what the provider can currently produce.
pub fn manifest_for_language(language: Language) -> ProviderManifest {
    let mut supported = vec![EdgeClass::Definitions, EdgeClass::Imports, EdgeClass::Calls];
    if matches!(language, Language::Rust) {
        supported.extend([
            EdgeClass::Usages,
            EdgeClass::TypeRefs,
            EdgeClass::TypeAssigns,
            EdgeClass::Implements,
        ]);
    } else if matches!(language, Language::Kotlin | Language::Ruby) {
        supported.extend([EdgeClass::Usages, EdgeClass::TypeRefs]);
    }
    let all = [
        EdgeClass::Definitions,
        EdgeClass::Imports,
        EdgeClass::Calls,
        EdgeClass::Usages,
        EdgeClass::TypeRefs,
        EdgeClass::TypeAssigns,
        EdgeClass::Implements,
        EdgeClass::Tests,
        EdgeClass::Routes,
        EdgeClass::EnvScan,
        EdgeClass::InfraScan,
        EdgeClass::K8s,
        EdgeClass::PackageMap,
        EdgeClass::ConfigLink,
        EdgeClass::Configures,
        EdgeClass::Complexity,
        EdgeClass::GitDiff,
        EdgeClass::GitHistory,
        EdgeClass::CrossRepo,
        EdgeClass::Semantic,
        EdgeClass::SemanticEdges,
        EdgeClass::Similarity,
    ];
    let unsupported = all
        .into_iter()
        .filter(|class| !supported.contains(class))
        .collect();
    let (file_extensions, filenames) = language_selectors(language);
    ProviderManifest {
        language: language.name().to_string(),
        provider_version: "greppy-parser-contract-v1".into(),
        status: if language.is_supported() {
            ProviderStatus::Partial
        } else {
            ProviderStatus::Unsupported
        },
        file_extensions,
        filenames,
        supported_edge_classes: supported,
        unsupported_edge_classes: unsupported,
        fixture_paths: Vec::new(),
        golden_paths: Vec::new(),
        review_artifact: None,
        notes: match language {
            Language::Kotlin => vec![
                "Kotlin classifies type/value references, but remains partial: member-call coverage, type assignments, and implements edges are not complete"
                    .into(),
            ],
            Language::Ruby => vec![
                "Ruby classifies constant-receiver type references and value reads, but remains partial: dynamic dispatch, type assignments, and implements edges are not complete"
                    .into(),
            ],
            _ => vec![
                "R0 contract manifest; acceptance requires language-specific evidence".into(),
            ],
        },
    }
}

fn language_selectors(language: Language) -> (Vec<String>, Vec<String>) {
    let exts: &[&str] = match language {
        Language::Rust => &["rs"],
        Language::Python => &["py"],
        Language::JavaScript => &["js", "jsx", "mjs", "cjs"],
        Language::TypeScript { tsx: false } => &["ts"],
        Language::TypeScript { tsx: true } => &["tsx"],
        Language::Go => &["go"],
        Language::Ruby => &["rb"],
        Language::Java => &["java"],
        Language::C => &["c", "h"],
        Language::Cpp => &["cpp", "cc", "cxx", "hpp", "hh"],
        Language::CSharp => &["cs"],
        Language::Php => &["php"],
        Language::Bash => &["sh", "bash"],
        Language::Lua => &["lua"],
        Language::Kotlin => &["kt", "kts"],
        Language::Scala => &["scala", "sc"],
        Language::Swift => &["swift"],
        Language::Zig => &["zig"],
        Language::R => &["r", "R"],
        Language::Registered(d) => d.extensions,
        Language::Unsupported(_) => &[],
    };
    let filenames: &[&str] = match language {
        Language::Registered(d) => d.filenames,
        _ => &[],
    };
    (
        exts.iter().map(|s| (*s).to_string()).collect(),
        filenames.iter().map(|s| (*s).to_string()).collect(),
    )
}

fn validate_span(start: u32, end: u32) -> std::result::Result<(), ProviderContractError> {
    if start == 0 || end == 0 || end < start {
        return Err(ProviderContractError::InvalidSpan { start, end });
    }
    Ok(())
}

fn validate_confidence(confidence: f32) -> std::result::Result<(), ProviderContractError> {
    if !(0.0..=1.0).contains(&confidence) {
        return Err(ProviderContractError::InvalidConfidence(confidence));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProviderContractError {
    MissingField(&'static str),
    MissingSelection,
    ContradictoryEdgeClass(EdgeClass),
    AcceptedWithoutEvidence,
    LanguageMismatch { manifest: String, output: String },
    FileMismatch { expected: String, found: String },
    MissingRecordIdentity,
    InvalidSpan { start: u32, end: u32 },
    InvalidConfidence(f32),
}

impl std::fmt::Display for ProviderContractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProviderContractError::MissingField(name) => {
                write!(f, "provider contract missing required field: {name}")
            }
            ProviderContractError::MissingSelection => {
                write!(f, "provider contract must declare an extension or filename")
            }
            ProviderContractError::ContradictoryEdgeClass(class) => write!(
                f,
                "edge class {} is both supported and unsupported",
                class.as_str()
            ),
            ProviderContractError::AcceptedWithoutEvidence => {
                write!(
                    f,
                    "accepted provider requires fixtures, goldens and review artifact"
                )
            }
            ProviderContractError::LanguageMismatch { manifest, output } => {
                write!(
                    f,
                    "provider language mismatch: manifest={manifest}, output={output}"
                )
            }
            ProviderContractError::FileMismatch { expected, found } => {
                write!(
                    f,
                    "provider file mismatch: expected={expected}, found={found}"
                )
            }
            ProviderContractError::MissingRecordIdentity => {
                write!(f, "provider record has an empty identity field")
            }
            ProviderContractError::InvalidSpan { start, end } => {
                write!(f, "provider record has invalid span {start}..{end}")
            }
            ProviderContractError::InvalidConfidence(confidence) => {
                write!(f, "provider record confidence out of range: {confidence}")
            }
        }
    }
}

impl std::error::Error for ProviderContractError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> ProviderManifest {
        ProviderManifest {
            language: "rust".into(),
            provider_version: "test-v1".into(),
            status: ProviderStatus::Partial,
            file_extensions: vec!["rs".into()],
            filenames: Vec::new(),
            supported_edge_classes: vec![EdgeClass::Definitions, EdgeClass::Calls],
            unsupported_edge_classes: vec![EdgeClass::Imports],
            fixture_paths: Vec::new(),
            golden_paths: Vec::new(),
            review_artifact: None,
            notes: Vec::new(),
        }
    }

    #[test]
    fn manifest_rejects_supported_and_unsupported_same_edge_class() {
        let mut m = manifest();
        m.unsupported_edge_classes.push(EdgeClass::Calls);
        assert_eq!(
            m.validate(),
            Err(ProviderContractError::ContradictoryEdgeClass(
                EdgeClass::Calls
            ))
        );
    }

    #[test]
    fn accepted_manifest_requires_external_evidence() {
        let mut m = manifest();
        m.status = ProviderStatus::Accepted;
        assert_eq!(
            m.validate(),
            Err(ProviderContractError::AcceptedWithoutEvidence)
        );
    }

    #[test]
    fn provider_output_rejects_invalid_span_and_confidence() {
        let mut output = ProviderOutput {
            manifest: manifest(),
            file_id: "src/lib.rs".into(),
            language: "rust".into(),
            nodes: vec![ProviderNode {
                file_id: "src/lib.rs".into(),
                language: "rust".into(),
                symbol_id: "src/lib.rs::Function::a".into(),
                symbol_kind: "Function".into(),
                name: "a".into(),
                qualified_name: "src/lib.rs::Function::a".into(),
                span_start: 4,
                span_end: 3,
                definition_span: Some((4, 3)),
                confidence: 1.2,
                is_partial: true,
                diagnostics: vec!["synthetic test".into()],
                properties: serde_json::json!({}),
            }],
            edges: Vec::new(),
            diagnostics: Vec::new(),
        };
        assert_eq!(
            output.validate(),
            Err(ProviderContractError::InvalidSpan { start: 4, end: 3 })
        );
        output.nodes[0].span_end = 4;
        assert_eq!(
            output.validate(),
            Err(ProviderContractError::InvalidConfidence(1.2))
        );
    }

    #[test]
    fn kotlin_manifest_claims_reference_classes_but_remains_partial() {
        let manifest = manifest_for_language(Language::Kotlin);
        assert_eq!(manifest.status, ProviderStatus::Partial);
        assert!(manifest.supports(EdgeClass::Usages));
        assert!(manifest.supports(EdgeClass::TypeRefs));
        assert!(
            manifest
                .unsupported_edge_classes
                .contains(&EdgeClass::TypeAssigns)
        );
        assert!(
            manifest
                .notes
                .iter()
                .any(|note| note.contains("member-call coverage"))
        );
        manifest.validate().unwrap();
    }

    #[test]
    fn ruby_manifest_claims_reference_classes_but_remains_partial() {
        let manifest = manifest_for_language(Language::Ruby);
        assert_eq!(manifest.status, ProviderStatus::Partial);
        assert!(manifest.supports(EdgeClass::Usages));
        assert!(manifest.supports(EdgeClass::TypeRefs));
        assert!(manifest
            .unsupported_edge_classes
            .contains(&EdgeClass::TypeAssigns));
        assert!(manifest
            .notes
            .iter()
            .any(|note| note.contains("dynamic dispatch")));
        manifest.validate().unwrap();
    }

    #[test]
    fn rust_manifest_is_partial_and_names_missing_edge_classes() {
        let m = manifest_for_language(Language::Rust);
        assert_eq!(m.status, ProviderStatus::Partial);
        assert_eq!(m.file_extensions, vec!["rs"]);
        assert!(m.supports(EdgeClass::Definitions));
        assert!(m.supports(EdgeClass::TypeRefs));
        assert!(m.unsupported_edge_classes.contains(&EdgeClass::Tests));
        m.validate().unwrap();
    }
}
