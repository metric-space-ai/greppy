//! Edit certificates: the machine-checkable result of every edit operation.
//!
//! Schema: `docs/contracts/edit-certificate.v1.schema.json` (normative).
//! Guarantee levels are named and reported separately; there is no scalar
//! confidence anywhere in this type.

use serde::{Deserialize, Serialize};

pub const CERTIFICATE_SCHEMA: &str = "greppy.edit-certificate.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Status {
    Applied,
    AlreadySatisfied,
    NotFound,
    Ambiguous,
    Stale,
    InvalidResult,
    ValidationFailed,
    PublishFailed,
}

impl Status {
    /// Binding exit-code mapping from `docs/contracts/EDIT_CONTRACT.md`.
    pub fn exit_code(self) -> i32 {
        match self {
            Status::Applied | Status::AlreadySatisfied => 0,
            Status::NotFound => 10,
            Status::Ambiguous => 11,
            Status::Stale => 12,
            Status::InvalidResult => 13,
            Status::ValidationFailed => 14,
            Status::PublishFailed => 16,
        }
    }
}

/// Map publication errors to the certificate status required by the edit
/// contract. Only a failed compare-and-swap is stale; path, lock, and I/O
/// failures are publication failures and must not be mislabeled as staleness.
pub(crate) fn publish_error_status(error: &greppy_core::Error) -> Status {
    match error {
        greppy_core::Error::Workspace(message) if message.starts_with("stale plan:") => {
            Status::Stale
        }
        _ => Status::PublishFailed,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Guarantee {
    Proved,
    NotApplicable,
    WaivedByFormatter,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Guarantees {
    pub addressed_range: Guarantee,
    pub no_clobber: Guarantee,
    pub byte_isolation: Guarantee,
    pub syntax: Guarantee,
    pub validators: Guarantee,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SelectorEngine {
    Symbol,
    TreeSitter,
    Text,
    Regex,
    DataPath,
    Lsp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SelectorClass {
    Resolved,
    Structural,
    ExactText,
    RegexWeak,
    StructuredData,
    Semantic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntaxDelta {
    pub errors_before: usize,
    pub errors_after: usize,
    pub new_errors: usize,
    pub new_missing_nodes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostconditionResult {
    pub name: String,
    pub passed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub qualified_name: String,
    pub path: String,
    pub line: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationReport {
    pub id: String,
    pub file: String,
    pub selector_engine: SelectorEngine,
    pub selector_class: SelectorClass,
    pub scope_matches: usize,
    pub target_matches: usize,
    pub file_sha256_before: String,
    pub file_sha256_after: Option<String>,
    pub target_sha256_before: String,
    pub target_sha256_after: Option<String>,
    pub outside_declared_ranges_unchanged: bool,
    pub changed_byte_ranges: Vec<(usize, usize)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_before: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_after: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unified_diff: Option<String>,
    pub syntax: SyntaxDelta,
    pub postconditions_passed: bool,
    pub postconditions: Vec<PostconditionResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub residual_occurrences: Option<usize>,
    pub guarantees: Guarantees,
    pub formatter_expanded_change_scope: bool,
    pub store_refreshed: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub candidates: Vec<Candidate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorReport {
    pub argv: Vec<String>,
    pub exit_code: i32,
    pub timed_out: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceReport {
    pub root: String,
    pub git_head_before: Option<String>,
    pub git_head_after: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PublishMode {
    Atomic,
    Journal,
    Patch,
    ShadowWorktree,
    DryRun,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Certificate {
    pub schema_version: String,
    pub status: Status,
    pub transaction_id: String,
    pub workspace: WorkspaceReport,
    pub operations: Vec<OperationReport>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub validators: Vec<ValidatorReport>,
    pub published: bool,
    pub publish_mode: PublishMode,
}

impl Certificate {
    pub fn exit_code(&self) -> i32 {
        self.status.exit_code()
    }

    /// Render the stdout form of a certificate. The schema and field names are
    /// unchanged, but evidence that is expensive in an agent context window is
    /// omitted; `--report` continues to use the full `Serialize` form.
    pub fn to_compact_json_pretty(&self) -> serde_json::Result<String> {
        let mut value = serde_json::to_value(self)?;
        if let Some(root) = value.as_object_mut() {
            root.insert("exit_code".into(), serde_json::json!(self.exit_code()));
            root.remove("validators");
            if let Some(operations) = root.get_mut("operations").and_then(|v| v.as_array_mut()) {
                for operation in operations {
                    if let Some(operation) = operation.as_object_mut() {
                        operation.remove("node_before");
                        operation.remove("node_after");
                        if let Some(postconditions) = operation
                            .get_mut("postconditions")
                            .and_then(|v| v.as_array_mut())
                        {
                            for postcondition in postconditions {
                                if let Some(postcondition) = postcondition.as_object_mut() {
                                    postcondition.remove("detail");
                                }
                            }
                        }
                    }
                }
            }
        }
        serde_json::to_string_pretty(&value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_codes_match_registered_contract() {
        assert_eq!(Status::Applied.exit_code(), 0);
        assert_eq!(Status::AlreadySatisfied.exit_code(), 0);
        assert_eq!(Status::NotFound.exit_code(), 10);
        assert_eq!(Status::Ambiguous.exit_code(), 11);
        assert_eq!(Status::Stale.exit_code(), 12);
        assert_eq!(Status::InvalidResult.exit_code(), 13);
        assert_eq!(Status::ValidationFailed.exit_code(), 14);
        assert_eq!(Status::PublishFailed.exit_code(), 16);
    }

    #[test]
    fn status_serializes_kebab_case() {
        assert_eq!(
            serde_json::to_string(&Status::AlreadySatisfied).unwrap(),
            "\"already-satisfied\""
        );
    }

    #[test]
    fn compact_stdout_omits_heavy_evidence_full_report_retains_it() {
        let certificate = Certificate {
            schema_version: CERTIFICATE_SCHEMA.into(),
            status: Status::Applied,
            transaction_id: "ge-test".into(),
            workspace: WorkspaceReport {
                root: "/tmp/ws".into(),
                git_head_before: None,
                git_head_after: None,
            },
            operations: vec![OperationReport {
                id: "rename".into(),
                file: "src/lib.rs".into(),
                selector_engine: SelectorEngine::Symbol,
                selector_class: SelectorClass::Resolved,
                scope_matches: 1,
                target_matches: 2,
                file_sha256_before: "before".into(),
                file_sha256_after: Some("after".into()),
                target_sha256_before: "target-before".into(),
                target_sha256_after: Some("target-after".into()),
                outside_declared_ranges_unchanged: true,
                changed_byte_ranges: vec![(4, 9)],
                node_before: Some("fn old() {}".into()),
                node_after: Some("fn new() {}".into()),
                unified_diff: Some("--- a/src/lib.rs\n+++ b/src/lib.rs\n".into()),
                syntax: SyntaxDelta {
                    errors_before: 0,
                    errors_after: 0,
                    new_errors: 0,
                    new_missing_nodes: 0,
                },
                postconditions_passed: false,
                postconditions: vec![PostconditionResult {
                    name: "residual-occurrences".into(),
                    passed: false,
                    detail: Some("expected 0, found 1".into()),
                }],
                residual_occurrences: Some(1),
                guarantees: Guarantees {
                    addressed_range: Guarantee::Proved,
                    no_clobber: Guarantee::Proved,
                    byte_isolation: Guarantee::Proved,
                    syntax: Guarantee::Proved,
                    validators: Guarantee::Failed,
                },
                formatter_expanded_change_scope: false,
                store_refreshed: false,
                candidates: vec![],
            }],
            validators: vec![ValidatorReport {
                argv: vec!["cargo".into(), "test".into()],
                exit_code: 1,
                timed_out: false,
            }],
            published: true,
            publish_mode: PublishMode::Atomic,
        };

        let compact: serde_json::Value =
            serde_json::from_str(&certificate.to_compact_json_pretty().unwrap()).unwrap();
        let full = serde_json::to_value(&certificate).unwrap();
        assert_eq!(compact["exit_code"], 0);
        assert_eq!(compact["operations"][0]["target_matches"], 2);
        assert_eq!(compact["operations"][0]["changed_byte_ranges"][0][0], 4);
        assert!(compact["operations"][0].get("node_before").is_none());
        assert!(compact["operations"][0].get("node_after").is_none());
        assert!(compact.get("validators").is_none());
        assert!(compact["operations"][0]["postconditions"][0]
            .get("detail")
            .is_none());
        assert_eq!(full["operations"][0]["node_before"], "fn old() {}");
        assert_eq!(full["operations"][0]["node_after"], "fn new() {}");
        assert_eq!(full["validators"][0]["argv"][0], "cargo");
        assert_eq!(
            full["operations"][0]["postconditions"][0]["detail"],
            "expected 0, found 1"
        );
    }
}
