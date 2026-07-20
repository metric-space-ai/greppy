//! The `greppy.edit-plan.v1` executor: multiple operations across multiple
//! files as one logical transaction.
//!
//! Selector engines `text` and `tree-sitter` resolve here; `symbol`
//! selectors are resolved by the caller (the CLI owns the store) into
//! explicit byte ranges before execution. All ranges are planned against
//! per-file snapshots taken under one pass, cross-file overlap is impossible
//! by construction (per-file overlap is rejected), and publication goes
//! through the journal (`journal` mode), single-file atomic writes
//! (`atomic`, single-file plans only), or a unified patch (`patch`).

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::certificate::{
    Certificate, Guarantee, Guarantees, OperationReport, PublishMode, SelectorClass,
    SelectorEngine, Status, SyntaxDelta, WorkspaceReport,
};
use crate::hash::sha256_hex;
use crate::journal::{publish_journal, FilePublication};
use crate::txn::{apply_in_memory, outside_ranges_unchanged, syntax_counts, PlannedOp, Snapshot};
use greppy_core::{Error, Result};

pub const PLAN_SCHEMA: &str = "greppy.edit-plan.v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Plan {
    pub schema_version: String,
    pub workspace: PlanWorkspace,
    pub operations: Vec<PlanOperation>,
    #[serde(default)]
    pub validators: Vec<PlanValidator>,
    pub publish: PlanPublish,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanWorkspace {
    pub root: String,
    #[serde(default)]
    pub expect_git_head: Option<String>,
    #[serde(default = "default_true")]
    pub require_unchanged_files: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanOperation {
    pub id: String,
    pub file: String,
    pub selector: PlanSelector,
    pub action: PlanAction,
    #[serde(default)]
    pub preconditions: PlanPreconditions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "engine")]
pub enum PlanSelector {
    /// Resolved by the caller into a byte range (CLI resolves symbols).
    Resolved { byte_start: usize, byte_end: usize },
    /// Exact text, `expect` occurrences (all are edited).
    Text {
        old_text: String,
        #[serde(default = "one")]
        expect: usize,
    },
}

fn one() -> usize {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "type")]
pub enum PlanAction {
    Replace { content: String },
    Delete,
    InsertAfter { content: String },
    InsertBefore { content: String },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlanPreconditions {
    #[serde(default)]
    pub file_sha256: Option<String>,
    #[serde(default)]
    pub target_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanValidator {
    pub argv: Vec<String>,
    #[serde(default = "default_timeout")]
    pub timeout_seconds: u64,
}

fn default_timeout() -> u64 {
    60
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanPublish {
    pub mode: PlanPublishMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PlanPublishMode {
    Atomic,
    Journal,
    Patch,
    ShadowWorktree,
}

/// Execute a plan. `dry_run` runs everything through postconditions and
/// validators but publishes nothing.
pub fn apply_plan(plan: &Plan, dry_run: bool) -> Result<Certificate> {
    if plan.schema_version != PLAN_SCHEMA {
        return Err(Error::Invalid(format!(
            "unsupported plan schema: {}",
            plan.schema_version
        )));
    }
    let root = Path::new(&plan.workspace.root);
    let git_head_before = git_head(root);

    // group operations per file, snapshot each file once
    let mut by_file: BTreeMap<String, Vec<&PlanOperation>> = BTreeMap::new();
    for op in &plan.operations {
        by_file.entry(op.file.clone()).or_default().push(op);
    }

    let mut op_reports = Vec::new();
    let mut publications = Vec::new();
    let mut refusal = plan
        .workspace
        .expect_git_head
        .as_ref()
        .filter(|expected| git_head_before.as_ref() != Some(*expected))
        .map(|_| Status::Stale);

    for (file, ops) in &by_file {
        if refusal.is_some() {
            break;
        }
        let abs = root.join(file);
        let snapshot = Snapshot::read(&abs)?;
        let mut planned: Vec<PlannedOp> = Vec::new();
        for op in ops {
            if plan.workspace.require_unchanged_files && op.preconditions.file_sha256.is_none() {
                refusal = Some(Status::Stale);
                break;
            }
            if let Some(expected) = &op.preconditions.file_sha256 {
                if *expected != snapshot.file_sha256 {
                    refusal = Some(Status::Stale);
                    break;
                }
            }
            let ranges: Vec<(usize, usize)> = match &op.selector {
                PlanSelector::Resolved {
                    byte_start,
                    byte_end,
                } => vec![(*byte_start, *byte_end)],
                PlanSelector::Text { old_text, expect } => {
                    let found = find_all(&snapshot.content, old_text.as_bytes());
                    if found.is_empty() {
                        refusal = Some(Status::NotFound);
                        break;
                    }
                    if found.len() != *expect {
                        refusal = Some(Status::Ambiguous);
                        break;
                    }
                    found.into_iter().map(|s| (s, s + old_text.len())).collect()
                }
            };
            if let Some(expected) = &op.preconditions.target_sha256 {
                let ok = ranges.len() == 1
                    && sha256_hex(&snapshot.content[ranges[0].0..ranges[0].1]) == *expected;
                if !ok {
                    refusal = Some(Status::Stale);
                    break;
                }
            }
            for (i, range) in ranges.iter().enumerate() {
                let replacement: Vec<u8> = match &op.action {
                    PlanAction::Replace { content } => content.clone().into_bytes(),
                    PlanAction::Delete => Vec::new(),
                    PlanAction::InsertAfter { content } => {
                        let mut b = Vec::new();
                        b.push(b'\n');
                        b.extend_from_slice(content.as_bytes());
                        if !content.ends_with('\n') {
                            b.push(b'\n');
                        }
                        planned.push(PlannedOp {
                            id: format!("{}-{i}", op.id),
                            range: (range.1, range.1),
                            replacement: b,
                        });
                        continue;
                    }
                    PlanAction::InsertBefore { content } => {
                        let mut b = content.as_bytes().to_vec();
                        if !content.ends_with('\n') {
                            b.push(b'\n');
                        }
                        b.push(b'\n');
                        planned.push(PlannedOp {
                            id: format!("{}-{i}", op.id),
                            range: (range.0, range.0),
                            replacement: b,
                        });
                        continue;
                    }
                };
                planned.push(PlannedOp {
                    id: format!("{}-{i}", op.id),
                    range: *range,
                    replacement,
                });
            }
        }
        if refusal.is_some() {
            break;
        }
        let applied = apply_in_memory(&snapshot, &planned)?;
        let language = greppy_parser::language_for_path(&abs);
        let syntax_before = syntax_counts(language, &snapshot.content);
        let syntax_after = syntax_counts(language, &applied.content);
        let (syntax, syntax_applicable) = match (syntax_before, syntax_after) {
            (Some(b), Some(a)) => (
                SyntaxDelta {
                    errors_before: b.errors,
                    errors_after: a.errors,
                    new_errors: a.errors.saturating_sub(b.errors),
                    new_missing_nodes: a.missing.saturating_sub(b.missing),
                },
                true,
            ),
            _ => (
                SyntaxDelta {
                    errors_before: 0,
                    errors_after: 0,
                    new_errors: 0,
                    new_missing_nodes: 0,
                },
                false,
            ),
        };
        let syntax_ok =
            !syntax_applicable || (syntax.new_errors == 0 && syntax.new_missing_nodes == 0);
        let isolation_ok = outside_ranges_unchanged(&snapshot.content, &applied.content, &planned);
        if !syntax_ok || !isolation_ok {
            refusal = Some(Status::InvalidResult);
        }
        op_reports.push(OperationReport {
            id: ops.first().map(|o| o.id.clone()).unwrap_or_default(),
            file: file.clone(),
            selector_engine: SelectorEngine::Text,
            selector_class: SelectorClass::ExactText,
            scope_matches: 1,
            target_matches: planned.len(),
            file_sha256_before: snapshot.file_sha256.clone(),
            file_sha256_after: Some(applied.file_sha256.clone()),
            target_sha256_before: String::new(),
            target_sha256_after: None,
            outside_declared_ranges_unchanged: isolation_ok,
            changed_byte_ranges: applied.changed_ranges.clone(),
            node_before: None,
            node_after: None,
            unified_diff: (plan.publish.mode == PlanPublishMode::Patch).then(|| {
                crate::verbs::unified_diff_public(file, &snapshot.content, &applied.content)
            }),
            syntax,
            postconditions_passed: syntax_ok && isolation_ok,
            postconditions: vec![],
            residual_occurrences: None,
            guarantees: Guarantees {
                addressed_range: Guarantee::Proved,
                no_clobber: Guarantee::Proved,
                byte_isolation: if isolation_ok {
                    Guarantee::Proved
                } else {
                    Guarantee::Failed
                },
                syntax: if !syntax_applicable {
                    Guarantee::NotApplicable
                } else if syntax_ok {
                    Guarantee::Proved
                } else {
                    Guarantee::Failed
                },
                validators: Guarantee::NotApplicable,
            },
            formatter_expanded_change_scope: false,
            store_refreshed: false,
            candidates: vec![],
        });
        publications.push(FilePublication {
            rel_path: file.clone(),
            expected_live_sha256: snapshot.file_sha256.clone(),
            content: applied.content,
        });
    }

    let tx = format!(
        "ge-{}",
        &sha256_hex(
            publications
                .iter()
                .map(|p| p.expected_live_sha256.as_str())
                .collect::<Vec<_>>()
                .join(":")
                .as_bytes()
        )[..16]
    );

    let mut status = refusal.unwrap_or(Status::Applied);
    let mut published = false;
    let mut validator_reports = Vec::new();
    // Validators always run against the edited shadow. Shadow-worktree
    // publication is kept separate so workspace preconditions can be
    // rechecked after validation and immediately before journal publish.
    if status == Status::Applied
        && (!plan.validators.is_empty() || plan.publish.mode == PlanPublishMode::ShadowWorktree)
    {
        match crate::shadow::shadow_validate(root, &publications, &plan.validators) {
            Ok(reports) => {
                let all_ok = reports.iter().all(|r| r.exit_code == 0 && !r.timed_out);
                validator_reports = reports;
                if !all_ok {
                    status = Status::ValidationFailed;
                }
            }
            Err(error) => status = crate::certificate::publish_error_status(&error),
        }
    }

    if status == Status::Applied {
        let head_matches = plan
            .workspace
            .expect_git_head
            .as_ref()
            .is_none_or(|expected| git_head(root).as_ref() == Some(expected));
        let files_match =
            !plan.workspace.require_unchanged_files || publications_unchanged(root, &publications);
        if !head_matches || !files_match {
            status = Status::Stale;
        }
    }

    if status == Status::Applied && !dry_run {
        match plan.publish.mode {
            PlanPublishMode::Patch => {}
            PlanPublishMode::Atomic => {
                if publications.len() != 1 {
                    return Err(Error::Invalid(
                        "publish mode atomic requires a single-file plan; use journal".into(),
                    ));
                }
                let p = &publications[0];
                match crate::publish::publish_atomic(
                    root,
                    &root.join(&p.rel_path),
                    &p.content,
                    &p.expected_live_sha256,
                ) {
                    Ok(_) => published = true,
                    Err(error) => status = crate::certificate::publish_error_status(&error),
                }
            }
            PlanPublishMode::Journal | PlanPublishMode::ShadowWorktree => {
                match publish_journal(root, &tx, &publications) {
                    Ok(()) => published = true,
                    Err(error) => status = crate::certificate::publish_error_status(&error),
                }
            }
        }
    }

    let git_head_after = git_head(root);
    let validator_guarantee = if plan.validators.is_empty() {
        Guarantee::NotApplicable
    } else if status == Status::ValidationFailed {
        Guarantee::Failed
    } else if validator_reports
        .iter()
        .all(|report| report.exit_code == 0 && !report.timed_out)
    {
        Guarantee::Proved
    } else {
        Guarantee::Failed
    };
    for report in &mut op_reports {
        report.guarantees.validators = validator_guarantee;
    }

    Ok(Certificate {
        schema_version: crate::certificate::CERTIFICATE_SCHEMA.into(),
        status,
        transaction_id: tx,
        workspace: WorkspaceReport {
            root: plan.workspace.root.clone(),
            git_head_before,
            git_head_after,
        },
        operations: op_reports,
        validators: validator_reports,
        published,
        publish_mode: if dry_run {
            PublishMode::DryRun
        } else {
            match plan.publish.mode {
                PlanPublishMode::Atomic => PublishMode::Atomic,
                PlanPublishMode::Journal => PublishMode::Journal,
                PlanPublishMode::Patch => PublishMode::Patch,
                PlanPublishMode::ShadowWorktree => PublishMode::ShadowWorktree,
            }
        },
    })
}

fn git_head(root: &Path) -> Option<String> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "--verify", "HEAD"])
        .current_dir(root)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let head = String::from_utf8(output.stdout).ok()?;
    let head = head.trim();
    (!head.is_empty()).then(|| head.to_string())
}

fn publications_unchanged(root: &Path, publications: &[FilePublication]) -> bool {
    publications.iter().all(|publication| {
        std::fs::read(root.join(&publication.rel_path))
            .map(|content| sha256_hex(&content) == publication.expected_live_sha256)
            .unwrap_or(false)
    })
}

fn find_all(haystack: &[u8], needle: &[u8]) -> Vec<usize> {
    if needle.is_empty() {
        return vec![];
    }
    let mut out = vec![];
    let mut from = 0usize;
    while from + needle.len() <= haystack.len() {
        match haystack[from..]
            .windows(needle.len())
            .position(|w| w == needle)
        {
            Some(rel) => {
                out.push(from + rel);
                from = from + rel + needle.len();
            }
            None => break,
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan_json(root: &str, mode: &str) -> Plan {
        let a_sha = sha256_hex(&std::fs::read(Path::new(root).join("a.py")).unwrap());
        let b_sha = sha256_hex(&std::fs::read(Path::new(root).join("b.py")).unwrap());
        serde_json::from_value(serde_json::json!({
            "schema_version": PLAN_SCHEMA,
            "workspace": {"root": root},
            "operations": [
                {"id": "op-a", "file": "a.py",
                 "selector": {"engine": "text", "old_text": "VALUE = 1", "expect": 1},
                 "action": {"type": "replace", "content": "VALUE = 2"},
                 "preconditions": {"file_sha256": a_sha}},
                {"id": "op-b", "file": "b.py",
                 "selector": {"engine": "text", "old_text": "LIMIT = 10", "expect": 1},
                 "action": {"type": "replace", "content": "LIMIT = 20"},
                 "preconditions": {"file_sha256": b_sha}}
            ],
            "publish": {"mode": mode}
        }))
        .unwrap()
    }

    fn assert_schema_roundtrip(certificate: &Certificate) {
        let json = serde_json::to_value(certificate).unwrap();
        assert_eq!(
            json["schema_version"],
            crate::certificate::CERTIFICATE_SCHEMA
        );
        assert!(json.get("status").is_some());
        assert!(json.get("transaction_id").is_some());
        assert!(json.get("workspace").is_some());
        assert!(json.get("operations").is_some());
        assert!(json.get("published").is_some());
        let _: Certificate = serde_json::from_value(json).unwrap();
    }

    #[test]
    fn journal_plan_edits_two_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.py"), b"VALUE = 1\n").unwrap();
        std::fs::write(dir.path().join("b.py"), b"LIMIT = 10\n").unwrap();
        let plan = plan_json(dir.path().to_str().unwrap(), "journal");
        let cert = apply_plan(&plan, false).unwrap();
        assert_eq!(cert.status, Status::Applied);
        assert!(cert.published);
        assert_schema_roundtrip(&cert);
        assert_eq!(
            std::fs::read(dir.path().join("a.py")).unwrap(),
            b"VALUE = 2\n"
        );
        assert_eq!(
            std::fs::read(dir.path().join("b.py")).unwrap(),
            b"LIMIT = 20\n"
        );
    }

    #[test]
    fn ambiguous_selector_refuses_whole_plan() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.py"), b"VALUE = 1\nVALUE = 1\n").unwrap();
        std::fs::write(dir.path().join("b.py"), b"LIMIT = 10\n").unwrap();
        let plan = plan_json(dir.path().to_str().unwrap(), "journal");
        let cert = apply_plan(&plan, false).unwrap();
        assert_eq!(cert.status, Status::Ambiguous);
        assert!(!cert.published);
        assert_eq!(
            std::fs::read(dir.path().join("b.py")).unwrap(),
            b"LIMIT = 10\n"
        );
    }

    #[test]
    fn patch_mode_mutates_nothing_and_reports_unified_diffs() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.py"), b"VALUE = 1\n").unwrap();
        std::fs::write(dir.path().join("b.py"), b"LIMIT = 10\n").unwrap();
        let plan = plan_json(dir.path().to_str().unwrap(), "patch");
        let cert = apply_plan(&plan, false).unwrap();
        assert_eq!(cert.status, Status::Applied);
        assert!(!cert.published);
        assert_schema_roundtrip(&cert);
        assert!(cert.operations.iter().all(|operation| operation
            .unified_diff
            .as_deref()
            .is_some_and(|diff| diff.starts_with("--- a/") && diff.contains("+++ b/"))));
        assert_eq!(
            std::fs::read(dir.path().join("a.py")).unwrap(),
            b"VALUE = 1\n"
        );
    }

    #[test]
    fn shadow_mode_validates_before_publishing() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.py"), b"VALUE = 1\n").unwrap();
        std::fs::write(dir.path().join("b.py"), b"LIMIT = 10\n").unwrap();
        let mut plan = plan_json(dir.path().to_str().unwrap(), "shadow-worktree");
        plan.validators = vec![super::PlanValidator {
            argv: vec![
                "grep".into(),
                "-q".into(),
                "VALUE = 2".into(),
                "a.py".into(),
            ],
            timeout_seconds: 10,
        }];
        let cert = apply_plan(&plan, false).unwrap();
        assert_eq!(cert.status, Status::Applied);
        assert!(cert.published);
        assert_schema_roundtrip(&cert);
        assert_eq!(cert.validators.len(), 1);
        assert_eq!(
            std::fs::read(dir.path().join("a.py")).unwrap(),
            b"VALUE = 2\n"
        );
        // fehlschlagender Validator: nichts publiziert
        std::fs::write(dir.path().join("a.py"), b"VALUE = 1\n").unwrap();
        std::fs::write(dir.path().join("b.py"), b"LIMIT = 10\n").unwrap();
        let mut plan = plan_json(dir.path().to_str().unwrap(), "shadow-worktree");
        plan.validators = vec![super::PlanValidator {
            argv: vec!["false".into()],
            timeout_seconds: 10,
        }];
        let cert = apply_plan(&plan, false).unwrap();
        assert_eq!(cert.status, Status::ValidationFailed);
        assert!(!cert.published);
        assert_eq!(cert.exit_code(), 14);
        assert_eq!(
            std::fs::read(dir.path().join("a.py")).unwrap(),
            b"VALUE = 1\n"
        );
    }

    #[test]
    fn atomic_plan_emits_applied_certificate() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.py"), b"VALUE = 1\n").unwrap();
        std::fs::write(dir.path().join("b.py"), b"LIMIT = 10\n").unwrap();
        let mut plan = plan_json(dir.path().to_str().unwrap(), "atomic");
        plan.operations.truncate(1);

        let cert = apply_plan(&plan, false).unwrap();

        assert_eq!(cert.status, Status::Applied);
        assert_eq!(cert.exit_code(), 0);
        assert!(cert.published);
        assert_eq!(cert.publish_mode, PublishMode::Atomic);
        assert_schema_roundtrip(&cert);
    }

    #[cfg(unix)]
    #[test]
    fn atomic_unsafe_path_is_publish_failed_not_stale() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.py"), b"VALUE = 1\n").unwrap();
        std::fs::write(dir.path().join("b.py"), b"LIMIT = 10\n").unwrap();
        std::fs::hard_link(dir.path().join("a.py"), dir.path().join("alias.py")).unwrap();
        let mut plan = plan_json(dir.path().to_str().unwrap(), "atomic");
        plan.operations.truncate(1);

        let cert = apply_plan(&plan, false).unwrap();

        assert_eq!(cert.status, Status::PublishFailed);
        assert_eq!(cert.exit_code(), 16);
        assert!(!cert.published);
        assert_schema_roundtrip(&cert);
    }

    #[test]
    fn every_publish_mode_emits_stale_certificate_for_hash_mismatch() {
        for mode in ["atomic", "journal", "patch", "shadow-worktree"] {
            let dir = tempfile::tempdir().unwrap();
            std::fs::write(dir.path().join("a.py"), b"VALUE = 1\n").unwrap();
            std::fs::write(dir.path().join("b.py"), b"LIMIT = 10\n").unwrap();
            let mut plan = plan_json(dir.path().to_str().unwrap(), mode);
            if mode == "atomic" {
                plan.operations.truncate(1);
            }
            plan.operations[0].preconditions.file_sha256 = Some("stale".into());

            let cert = apply_plan(&plan, false).unwrap();

            assert_eq!(cert.status, Status::Stale, "mode {mode}");
            assert_eq!(cert.exit_code(), 12, "mode {mode}");
            assert!(!cert.published, "mode {mode}");
            let json = serde_json::to_value(&cert).unwrap();
            assert_eq!(json["status"], "stale", "mode {mode}");
            assert_schema_roundtrip(&cert);
        }
    }

    #[test]
    fn require_unchanged_files_requires_file_hash_preconditions() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.py"), b"VALUE = 1\n").unwrap();
        std::fs::write(dir.path().join("b.py"), b"LIMIT = 10\n").unwrap();
        let mut plan = plan_json(dir.path().to_str().unwrap(), "journal");
        plan.operations[0].preconditions.file_sha256 = None;

        let cert = apply_plan(&plan, false).unwrap();

        assert_eq!(cert.status, Status::Stale);
        assert_eq!(cert.exit_code(), 12);
        assert!(!cert.published);
    }

    #[test]
    fn matching_git_head_is_reported_and_permitted() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.py"), b"VALUE = 1\n").unwrap();
        std::fs::write(dir.path().join("b.py"), b"LIMIT = 10\n").unwrap();
        for args in [
            &["init", "-q"][..],
            &["config", "user.email", "edit-tests@example.invalid"][..],
            &["config", "user.name", "Edit Tests"][..],
            &["add", "a.py", "b.py"][..],
            &["commit", "-qm", "initial"][..],
        ] {
            assert!(std::process::Command::new("git")
                .args(args)
                .current_dir(dir.path())
                .status()
                .unwrap()
                .success());
        }
        let head = git_head(dir.path()).unwrap();
        let mut plan = plan_json(dir.path().to_str().unwrap(), "patch");
        plan.workspace.expect_git_head = Some(head.clone());

        let cert = apply_plan(&plan, false).unwrap();

        assert_eq!(cert.status, Status::Applied);
        assert_eq!(
            cert.workspace.git_head_before.as_deref(),
            Some(head.as_str())
        );
        assert_eq!(
            cert.workspace.git_head_after.as_deref(),
            Some(head.as_str())
        );
    }

    #[test]
    fn expect_git_head_mismatch_is_stale() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.py"), b"VALUE = 1\n").unwrap();
        std::fs::write(dir.path().join("b.py"), b"LIMIT = 10\n").unwrap();
        let mut plan = plan_json(dir.path().to_str().unwrap(), "journal");
        plan.workspace.expect_git_head = Some("not-the-live-head".into());

        let cert = apply_plan(&plan, false).unwrap();

        assert_eq!(cert.status, Status::Stale);
        assert_eq!(cert.exit_code(), 12);
        assert!(!cert.published);
    }

    #[test]
    fn dry_run_publishes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.py"), b"VALUE = 1\n").unwrap();
        std::fs::write(dir.path().join("b.py"), b"LIMIT = 10\n").unwrap();
        let plan = plan_json(dir.path().to_str().unwrap(), "journal");
        let cert = apply_plan(&plan, true).unwrap();
        assert_eq!(cert.status, Status::Applied);
        assert!(!cert.published);
        assert_eq!(
            std::fs::read(dir.path().join("a.py")).unwrap(),
            b"VALUE = 1\n"
        );
    }
}
