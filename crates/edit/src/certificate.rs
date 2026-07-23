//! Edit certificates: the machine-checkable result of every edit operation.
//!
//! Schema: `docs/contracts/edit-certificate.v1.schema.json` (normative).
//! Guarantee levels are named and reported separately; there is no scalar
//! confidence anywhere in this type.

use serde::{Deserialize, Serialize};

pub const CERTIFICATE_SCHEMA: &str = "greppy.edit-certificate.v1";
const COMPACT_RESULT_SPAN_MAX_BYTES: usize = 2000;
const COMPACT_DIFF_OMITTED: &str = "omitted; use --report for the full diff";

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

    /// Render the stdout form of a certificate. Evidence that is expensive in
    /// an agent context window is omitted. A bounded `result_span` is retained
    /// only when this certificate proves that publication actually completed;
    /// refused, validation-only, patch, and dry-run projections never present
    /// planned text as a written result. `--report` continues to use the full
    /// `Serialize` form.
    pub fn to_compact_json_pretty(&self) -> serde_json::Result<String> {
        let mut value = serde_json::to_value(self)?;
        let include_result_span = self.status == Status::Applied && self.published;
        if let Some(root) = value.as_object_mut() {
            root.insert("exit_code".into(), serde_json::json!(self.exit_code()));
            root.remove("validators");
            if let Some(operations) = root.get_mut("operations").and_then(|v| v.as_array_mut()) {
                for operation in operations {
                    if let Some(operation) = operation.as_object_mut() {
                        if include_result_span {
                            if let Some(result_span) =
                                operation.get("node_after").and_then(|value| value.as_str())
                            {
                                let (result_span, truncated, total_bytes) =
                                    bounded_result_span(result_span);
                                operation.insert("result_span".into(), result_span.into());
                                if truncated {
                                    operation.insert("result_span_truncated".into(), true.into());
                                    operation.insert(
                                        "result_span_total_bytes".into(),
                                        total_bytes.into(),
                                    );
                                }
                            }
                        }
                        operation.remove("node_before");
                        operation.remove("node_after");
                        if let Some(unified_diff) = operation.get_mut("unified_diff") {
                            *unified_diff = COMPACT_DIFF_OMITTED.into();
                        }
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
        if let Some((diagnosis, failed_operation, next_steps)) = self.failure_guidance() {
            let mut ordered = serde_json::Map::new();
            ordered.insert("diagnosis".into(), diagnosis.into());
            ordered.insert("failed_operation".into(), failed_operation);
            ordered.insert("next_steps".into(), serde_json::json!(next_steps));
            if let Some(root) = value.as_object() {
                for (key, value) in root {
                    ordered.insert(key.clone(), value.clone());
                }
            }
            value = ordered.into();
        }
        serde_json::to_string_pretty(&value)
    }

    fn failure_guidance(&self) -> Option<(String, serde_json::Value, Vec<String>)> {
        if !matches!(
            self.status,
            Status::NotFound
                | Status::Ambiguous
                | Status::Stale
                | Status::InvalidResult
                | Status::ValidationFailed
        ) {
            return None;
        }

        let (operation_index, operation) = self
            .operations
            .iter()
            .enumerate()
            .find(|(_, operation)| !operation.postconditions_passed)
            .or_else(|| self.operations.iter().enumerate().next())?;
        let failed_operation = serde_json::json!({
            "id": operation.id,
            "index": operation_index,
        });
        let detail = operation
            .postconditions
            .iter()
            .find(|postcondition| !postcondition.passed)
            .and_then(|postcondition| postcondition.detail.as_deref());
        let file = shell_quote(&operation.file);
        let candidate_selectors = operation
            .candidates
            .iter()
            .map(|candidate| format!("{}::{}", candidate.path, candidate.qualified_name))
            .collect::<Vec<_>>();

        let (diagnosis, next_steps) = match self.status {
            Status::NotFound => {
                let selectors = if candidate_selectors.is_empty() {
                    "no indexed candidates were found".to_string()
                } else {
                    format!(
                        "accepted path::SYMBOL candidates: {}",
                        candidate_selectors.join(", ")
                    )
                };
                let mut steps = candidate_selectors
                    .iter()
                    .map(|selector| format!("greppy read {} --json", shell_quote(selector)))
                    .collect::<Vec<_>>();
                if steps.is_empty() {
                    steps.push(format!("greppy search-symbols {} --json", file));
                }
                (
                    format!(
                        "operation `{}` expected at least one target but found 0; {selectors}.",
                        operation.id
                    ),
                    steps,
                )
            }
            Status::Ambiguous => {
                if operation.selector_engine == SelectorEngine::Text
                    && operation.selector_class == SelectorClass::ExactText
                {
                    (
                        detail.map_or_else(
                            || {
                                format!(
                                    "`OLD` occurs {} times; text-cas requires the explicit expected cardinality.",
                                    operation.target_matches
                                )
                            },
                            str::to_owned,
                        ),
                        vec![
                            "greppy edit text-cas --help".into(),
                            "greppy edit rename-symbol --help".into(),
                            "greppy edit rename-call --help".into(),
                        ],
                    )
                } else {
                    let selectors = if candidate_selectors.is_empty() {
                        "the selector matched multiple targets".to_string()
                    } else {
                        format!(
                            "disambiguate with path::SYMBOL: {}",
                            candidate_selectors.join(", ")
                        )
                    };
                    let mut steps = candidate_selectors
                        .iter()
                        .map(|selector| {
                            format!("greppy read {} --handle --json", shell_quote(selector))
                        })
                        .collect::<Vec<_>>();
                    if steps.is_empty() {
                        steps.push(format!("greppy search-symbols {} --json", file));
                    }
                    (
                        format!(
                            "operation `{}` expected one target but found {}; {selectors}.",
                            operation.id, operation.target_matches
                        ),
                        steps,
                    )
                }
            }
            Status::Stale => {
                let found = detail.unwrap_or("the live file or target hash no longer matches");
                (
                    format!(
                        "operation `{}` expected its planned span and hashes to be unchanged but found {found}.",
                        operation.id
                    ),
                    vec![
                        format!("nl -ba {file} | sed -n '1,400p'"),
                        "echo 'Rebuild the edit plan from the fresh span, then rerun it with --dry-run.'"
                            .into(),
                    ],
                )
            }
            Status::InvalidResult => {
                let violation = detail
                    .map(str::to_owned)
                    .or_else(|| {
                        (operation.syntax.new_errors > 0 || operation.syntax.new_missing_nodes > 0)
                            .then(|| {
                                format!(
                                    "{} new syntax error(s) and {} new missing node(s)",
                                    operation.syntax.new_errors, operation.syntax.new_missing_nodes
                                )
                            })
                    })
                    .unwrap_or_else(|| "the projected edit violated its declared scope".into());
                let alternative = match operation.selector_engine {
                    SelectorEngine::Symbol | SelectorEngine::TreeSitter | SelectorEngine::Lsp => {
                        "greppy edit patch-span --help"
                    }
                    SelectorEngine::Text | SelectorEngine::Regex => "greppy edit text-cas --help",
                    SelectorEngine::DataPath => "greppy edit data --help",
                };
                (
                    format!(
                        "operation `{}` expected no new syntax or scope violation but found {violation}.",
                        operation.id
                    ),
                    vec![format!("greppy read {file} --json"), alternative.into()],
                )
            }
            Status::ValidationFailed => {
                let validator = self.validators.first();
                let found = validator.map_or_else(
                    || "a validator failure".to_string(),
                    |validator| {
                        format!(
                            "validator exit {}{}",
                            validator.exit_code,
                            if validator.timed_out {
                                " after timeout"
                            } else {
                                ""
                            }
                        )
                    },
                );
                let rerun = validator
                    .map(|validator| {
                        validator
                            .argv
                            .iter()
                            .map(|argument| shell_quote(argument))
                            .collect::<Vec<_>>()
                            .join(" ")
                    })
                    .filter(|command| !command.is_empty())
                    .unwrap_or_else(|| "greppy edit apply --help".into());
                (
                    format!(
                        "operation `{}` expected every validator to exit 0 but found {found}.",
                        operation.id
                    ),
                    vec![rerun, format!("greppy read {file} --json")],
                )
            }
            _ => unreachable!("failure status filtered above"),
        };
        Some((diagnosis, failed_operation, next_steps))
    }
}

fn shell_quote(value: &str) -> String {
    if value.is_empty() {
        return "''".into();
    }
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn bounded_result_span(result_span: &str) -> (&str, bool, usize) {
    let total_bytes = result_span.len();
    if total_bytes <= COMPACT_RESULT_SPAN_MAX_BYTES {
        return (result_span, false, total_bytes);
    }

    let mut end = COMPACT_RESULT_SPAN_MAX_BYTES;
    while !result_span.is_char_boundary(end) {
        end -= 1;
    }
    (&result_span[..end], true, total_bytes)
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

    fn sample_certificate(node_after: String) -> Certificate {
        Certificate {
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
                node_after: Some(node_after),
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
        }
    }

    #[test]
    fn compact_stdout_includes_result_span_and_omits_heavy_evidence() {
        let certificate = sample_certificate("fn new() {}".into());

        let compact: serde_json::Value =
            serde_json::from_str(&certificate.to_compact_json_pretty().unwrap()).unwrap();
        let full = serde_json::to_value(&certificate).unwrap();
        assert_eq!(compact["exit_code"], 0);
        assert_eq!(compact["operations"][0]["target_matches"], 2);
        assert_eq!(compact["operations"][0]["changed_byte_ranges"][0][0], 4);
        assert_eq!(compact["operations"][0]["result_span"], "fn new() {}");
        assert!(compact["operations"][0]
            .get("result_span_truncated")
            .is_none());
        assert!(compact["operations"][0].get("node_before").is_none());
        assert!(compact["operations"][0].get("node_after").is_none());
        assert_eq!(
            compact["operations"][0]["unified_diff"],
            COMPACT_DIFF_OMITTED
        );
        assert!(compact.get("validators").is_none());
        assert!(compact["operations"][0]["postconditions"][0]
            .get("detail")
            .is_none());
        assert!(full["operations"][0].get("result_span").is_none());
        assert_eq!(full["operations"][0]["node_before"], "fn old() {}");
        assert_eq!(full["operations"][0]["node_after"], "fn new() {}");
        assert_eq!(
            full["operations"][0]["unified_diff"],
            "--- a/src/lib.rs\n+++ b/src/lib.rs\n"
        );
        assert_eq!(full["validators"][0]["argv"][0], "cargo");
        assert_eq!(
            full["operations"][0]["postconditions"][0]["detail"],
            "expected 0, found 1"
        );
    }

    #[test]
    fn compact_refusal_omits_planned_result_span_and_truncation_fields() {
        let mut certificate = sample_certificate("x".repeat(2001));
        certificate.status = Status::Stale;
        certificate.published = false;

        let compact: serde_json::Value =
            serde_json::from_str(&certificate.to_compact_json_pretty().unwrap()).unwrap();
        let operation = &compact["operations"][0];
        assert!(operation.get("result_span").is_none());
        assert!(operation.get("result_span_truncated").is_none());
        assert!(operation.get("result_span_total_bytes").is_none());
        assert!(operation.get("node_after").is_none());
    }

    #[test]
    fn compact_result_span_is_truncated_at_2000_bytes_with_total_size() {
        let certificate = sample_certificate("x".repeat(2001));

        let compact: serde_json::Value =
            serde_json::from_str(&certificate.to_compact_json_pretty().unwrap()).unwrap();
        let operation = &compact["operations"][0];
        assert_eq!(operation["result_span"].as_str().unwrap().len(), 2000);
        assert_eq!(operation["result_span"], "x".repeat(2000));
        assert_eq!(operation["result_span_truncated"], true);
        assert_eq!(operation["result_span_total_bytes"], 2001);
    }

    fn compact_failure(status: Status) -> (String, serde_json::Value) {
        let mut certificate = sample_certificate("projected".into());
        certificate.status = status;
        certificate.published = false;
        certificate.operations[0].candidates = vec![Candidate {
            qualified_name: "api::target".into(),
            path: "src/api.rs".into(),
            line: 7,
        }];
        certificate.operations[0].postconditions = vec![PostconditionResult {
            name: "syntax-no-new-errors".into(),
            passed: false,
            detail: Some("first new syntax error at line 9".into()),
        }];
        certificate.operations[0].syntax.new_errors = 1;
        certificate.validators = vec![ValidatorReport {
            argv: vec!["cargo".into(), "test".into(), "-p".into(), "demo".into()],
            exit_code: 1,
            timed_out: false,
        }];
        let rendered = certificate.to_compact_json_pretty().unwrap();
        let parsed = serde_json::from_str(&rendered).unwrap();
        (rendered, parsed)
    }

    fn assert_actionable_failure(rendered: &str, compact: &serde_json::Value, exit_code: i32) {
        assert!(rendered.starts_with("{\n  \"diagnosis\""), "{rendered}");
        assert_eq!(compact["exit_code"], exit_code);
        assert_eq!(compact["failed_operation"]["id"], "rename");
        assert_eq!(compact["failed_operation"]["index"], 0);
        assert!(compact["diagnosis"]
            .as_str()
            .is_some_and(|text| !text.is_empty()));
        assert!(compact["next_steps"]
            .as_array()
            .is_some_and(|steps| !steps.is_empty()));
    }

    #[test]
    fn compact_exit_10_has_diagnosis_and_candidate_commands() {
        let (rendered, compact) = compact_failure(Status::NotFound);
        assert_actionable_failure(&rendered, &compact, 10);
        assert!(compact["diagnosis"]
            .as_str()
            .unwrap()
            .contains("src/api.rs::api::target"));
        assert!(compact["next_steps"][0]
            .as_str()
            .unwrap()
            .contains("greppy read 'src/api.rs::api::target'"));
    }

    #[test]
    fn compact_exit_11_has_diagnosis_and_disambiguating_command() {
        let (rendered, compact) = compact_failure(Status::Ambiguous);
        assert_actionable_failure(&rendered, &compact, 11);
        assert!(compact["diagnosis"]
            .as_str()
            .unwrap()
            .contains("path::SYMBOL"));
        assert!(compact["next_steps"][0]
            .as_str()
            .unwrap()
            .contains("--handle --json"));
    }

    #[test]
    fn compact_exit_12_has_diagnosis_and_reread_replan_commands() {
        let (rendered, compact) = compact_failure(Status::Stale);
        assert_actionable_failure(&rendered, &compact, 12);
        assert!(compact["next_steps"][0]
            .as_str()
            .unwrap()
            .contains("nl -ba 'src/lib.rs'"));
        assert!(compact["next_steps"][1]
            .as_str()
            .unwrap()
            .contains("Rebuild the edit plan"));
    }

    #[test]
    fn compact_exit_13_has_diagnosis_and_alternative_verb() {
        let (rendered, compact) = compact_failure(Status::InvalidResult);
        assert_actionable_failure(&rendered, &compact, 13);
        assert!(compact["diagnosis"]
            .as_str()
            .unwrap()
            .contains("first new syntax error"));
        assert_eq!(compact["next_steps"][1], "greppy edit patch-span --help");
    }

    #[test]
    fn compact_exit_14_has_diagnosis_and_validator_command() {
        let (rendered, compact) = compact_failure(Status::ValidationFailed);
        assert_actionable_failure(&rendered, &compact, 14);
        assert!(compact["diagnosis"]
            .as_str()
            .unwrap()
            .contains("validator exit 1"));
        assert_eq!(compact["next_steps"][0], "'cargo' 'test' '-p' 'demo'");
    }
}
