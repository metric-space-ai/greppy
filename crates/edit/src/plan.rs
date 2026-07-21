//! The `greppy.edit-plan.v1` executor: multiple operations across multiple
//! files as one logical transaction.
//!
//! Every selector is resolved against one immutable per-file snapshot. All
//! operation preconditions and all cross-operation overlaps are checked before
//! any projected edit is built. Valid operations are then applied high-to-low
//! per file and the complete publication set is committed with one journal.

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::certificate::{
    Certificate, Guarantee, Guarantees, OperationReport, PostconditionResult, PublishMode,
    SelectorClass, SelectorEngine, Status, SyntaxDelta, WorkspaceReport,
};
use crate::hash::sha256_hex;
use crate::journal::{publish_journal_locked, FilePublication, WorkspaceLock};
use crate::publish::require_inside_workspace;
use crate::txn::{
    apply_in_memory, outside_ranges_unchanged, syntax_counts, Applied, PlannedOp, Snapshot,
};
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
    /// Resolved by the caller into a byte range (the CLI owns symbol/store
    /// resolution). The live file and declared hashes still decide.
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
    Replace {
        #[serde(alias = "content_literal")]
        content: String,
    },
    Delete,
    InsertAfter {
        #[serde(alias = "content_literal")]
        content: String,
    },
    InsertBefore {
        #[serde(alias = "content_literal")]
        content: String,
    },
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

#[derive(Debug, Clone)]
struct OperationProblem {
    status: Status,
    name: String,
    detail: String,
}

#[derive(Debug)]
struct PlannedOperation<'a> {
    operation: &'a PlanOperation,
    file_key: String,
    selector_engine: SelectorEngine,
    selector_class: SelectorClass,
    target_ranges: Vec<(usize, usize)>,
    mutations: Vec<PlannedOp>,
    target_matches: usize,
    target_before: Vec<u8>,
    target_after: Vec<u8>,
    problem: Option<OperationProblem>,
    overlap_details: Vec<String>,
}

#[derive(Debug)]
struct FileProjection {
    applied: Applied,
    syntax: SyntaxDelta,
    syntax_applicable: bool,
    syntax_ok: bool,
    isolation_ok: bool,
}

fn lexical_relative_path(path: &str) -> Result<PathBuf> {
    let mut normalized = PathBuf::new();
    for component in Path::new(path).components() {
        match component {
            Component::CurDir => {}
            Component::Normal(part) => normalized.push(part),
            Component::ParentDir => {
                if !normalized.pop() {
                    return Err(Error::Invalid(format!(
                        "operation file escapes the workspace: {path}"
                    )));
                }
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(Error::Invalid(format!(
                    "operation file must be relative to the workspace: {path}"
                )));
            }
        }
    }
    if normalized.as_os_str().is_empty() {
        return Err(Error::Invalid(format!(
            "operation file is empty after normalization: {path}"
        )));
    }
    Ok(normalized)
}

fn operation_file_key(root: &Path, file: &str) -> Result<String> {
    let lexical = lexical_relative_path(file)?;
    let candidate = root.join(&lexical);
    let resolved = match require_inside_workspace(root, &candidate) {
        Ok(resolved) => resolved,
        // Preserve the certificate-producing publication path for final-component
        // symlinks and hardlinks. The atomic publisher will reject them; they do
        // not need canonical alias coalescing because they cannot be published.
        Err(Error::Workspace(message)) if message.starts_with("refusing to publish through ") => {
            return Ok(lexical.to_string_lossy().into_owned());
        }
        Err(error) => return Err(error),
    };
    let canonical_root = root.canonicalize().map_err(|source| Error::Io {
        context: format!("canonicalize {}", root.display()),
        source,
    })?;
    let relative = resolved.strip_prefix(&canonical_root).map_err(|_| {
        Error::Workspace(format!(
            "path {} escapes workspace {}",
            resolved.display(),
            canonical_root.display()
        ))
    })?;
    Ok(relative.to_string_lossy().into_owned())
}

/// Execute a parsed plan as one transaction. The workspace lock is held from
/// snapshot acquisition through validation and publication; active contention
/// returns immediately as a publish-failed certificate.
pub fn apply_plan(plan: &Plan, dry_run: bool) -> Result<Certificate> {
    if plan.schema_version != PLAN_SCHEMA {
        return Err(Error::Invalid(format!(
            "unsupported plan schema: {}",
            plan.schema_version
        )));
    }
    if plan.operations.is_empty() {
        return Err(Error::Invalid("edit plan has no operations".into()));
    }

    let root = Path::new(&plan.workspace.root);
    let transaction_id = plan_transaction_id(plan);
    let lock = match WorkspaceLock::acquire(root) {
        Ok(lock) => lock,
        Err(error) => {
            return Ok(lock_refusal_certificate(
                plan,
                dry_run,
                transaction_id,
                &error.to_string(),
            ))
        }
    };
    let takeover_reason = lock.takeover_reason().map(str::to_owned);
    let git_head_before = git_head(root);

    // Normalize aliases before keying snapshots. For ordinary files the key is
    // the canonical path relative to the canonical workspace root, so `a.py`,
    // `./a.py`, `dir/../a.py`, and an in-workspace parent symlink cannot create
    // separate snapshots for one physical target.
    let mut snapshots = BTreeMap::new();
    let mut operation_files = Vec::with_capacity(plan.operations.len());
    for operation in &plan.operations {
        let file_key = operation_file_key(root, &operation.file)?;
        if !snapshots.contains_key(&file_key) {
            let snapshot = Snapshot::read(&root.join(&file_key))?;
            snapshots.insert(file_key.clone(), snapshot);
        }
        operation_files.push(file_key);
    }

    let mut planned = Vec::with_capacity(plan.operations.len());
    for (operation, file_key) in plan.operations.iter().zip(operation_files) {
        let snapshot = snapshots
            .get(&file_key)
            .expect("every operation file was snapshotted");
        planned.push(plan_operation(plan, operation, file_key, snapshot));
    }

    // Selector overlap is defined on original target coordinates, before an
    // action converts insert-before/after to an empty mutation at a boundary.
    // Only different plan operations conflict; multiple cardinality matches of
    // one operation are one declared operation.
    let mut found_overlap = false;
    for first_index in 0..planned.len() {
        for second_index in first_index + 1..planned.len() {
            if planned[first_index].file_key != planned[second_index].file_key {
                continue;
            }
            let overlaps = overlapping_target_ranges(
                &planned[first_index].target_ranges,
                &planned[second_index].target_ranges,
            );
            for (first_range, second_range) in overlaps {
                found_overlap = true;
                let first_id = planned[first_index].operation.id.clone();
                let second_id = planned[second_index].operation.id.clone();
                planned[first_index].overlap_details.push(format!(
                    "operation `{first_id}` range {}..{} overlaps operation `{second_id}` range {}..{}",
                    first_range.0, first_range.1, second_range.0, second_range.1
                ));
                planned[second_index].overlap_details.push(format!(
                    "operation `{second_id}` range {}..{} overlaps operation `{first_id}` range {}..{}",
                    second_range.0, second_range.1, first_range.0, first_range.1
                ));
            }
        }
    }

    let git_head_stale = plan
        .workspace
        .expect_git_head
        .as_ref()
        .is_some_and(|expected| git_head_before.as_ref() != Some(expected));
    let mut status = if git_head_stale {
        Status::Stale
    } else if let Some(problem) = planned
        .iter()
        .find_map(|operation| operation.problem.as_ref())
    {
        problem.status
    } else if found_overlap {
        Status::InvalidResult
    } else {
        Status::Applied
    };

    if status != Status::Applied {
        let reports = planned
            .iter()
            .map(|operation| {
                refusal_report(
                    operation,
                    snapshots
                        .get(&operation.file_key)
                        .expect("snapshot retained"),
                    git_head_stale,
                    takeover_reason.as_deref(),
                )
            })
            .collect();
        return Ok(Certificate {
            schema_version: crate::certificate::CERTIFICATE_SCHEMA.into(),
            status,
            transaction_id,
            workspace: WorkspaceReport {
                root: plan.workspace.root.clone(),
                git_head_before: git_head_before.clone(),
                git_head_after: git_head(root),
            },
            operations: reports,
            validators: vec![],
            published: false,
            publish_mode: certificate_publish_mode(plan.publish.mode, dry_run),
        });
    }

    // Build every file projection only after all selectors, hashes, and
    // overlaps have passed. No workspace bytes are touched in this phase.
    let mut projections = BTreeMap::new();
    let mut projection_error = None;
    for (file, snapshot) in &snapshots {
        let mutations: Vec<PlannedOp> = planned
            .iter()
            .filter(|operation| operation.file_key == *file)
            .flat_map(|operation| operation.mutations.iter().cloned())
            .collect();
        match apply_in_memory(snapshot, &mutations) {
            Ok(applied) => {
                let language = greppy_parser::language_for_path(&snapshot.path);
                let syntax_before = language
                    .is_supported()
                    .then(|| syntax_counts(language, &snapshot.content))
                    .flatten();
                let syntax_after = language
                    .is_supported()
                    .then(|| syntax_counts(language, &applied.content))
                    .flatten();
                let (syntax, syntax_applicable) = syntax_delta(syntax_before, syntax_after);
                let syntax_ok =
                    !syntax_applicable || (syntax.new_errors == 0 && syntax.new_missing_nodes == 0);
                let isolation_ok =
                    outside_ranges_unchanged(&snapshot.content, &applied.content, &mutations);
                projections.insert(
                    file.clone(),
                    FileProjection {
                        applied,
                        syntax,
                        syntax_applicable,
                        syntax_ok,
                        isolation_ok,
                    },
                );
            }
            Err(error) => {
                projection_error = Some(error.to_string());
                break;
            }
        }
    }

    if projection_error.is_some()
        || projections
            .values()
            .any(|projection| !projection.syntax_ok || !projection.isolation_ok)
    {
        status = Status::InvalidResult;
    }

    let mut reports: Vec<OperationReport> = planned
        .iter()
        .map(|operation| {
            projected_report(
                operation,
                snapshots
                    .get(&operation.file_key)
                    .expect("snapshot retained"),
                projections.get(&operation.file_key),
                projection_error.as_deref(),
                takeover_reason.as_deref(),
            )
        })
        .collect();

    let publications: Vec<FilePublication> = projections
        .iter()
        .map(|(file, projection)| FilePublication {
            rel_path: file.clone(),
            expected_live_sha256: snapshots
                .get(file)
                .expect("projection has snapshot")
                .file_sha256
                .clone(),
            content: projection.applied.content.clone(),
        })
        .collect();

    let mut validator_reports = Vec::new();
    if status == Status::Applied
        && (!plan.validators.is_empty() || plan.publish.mode == PlanPublishMode::ShadowWorktree)
    {
        match crate::shadow::shadow_validate(root, &publications, &plan.validators) {
            Ok(results) => {
                let all_ok = results
                    .iter()
                    .all(|report| report.exit_code == 0 && !report.timed_out);
                validator_reports = results;
                if !all_ok {
                    status = Status::ValidationFailed;
                }
            }
            Err(error) => status = crate::certificate::publish_error_status(&error),
        }
    }

    // External, non-greppy writers do not honor the advisory lock, so retain
    // the binding CAS check immediately before every publish mode.
    if status == Status::Applied {
        let head_matches = plan
            .workspace
            .expect_git_head
            .as_ref()
            .is_none_or(|expected| git_head(root).as_ref() == Some(expected));
        let files_match = publications_unchanged(root, &publications);
        if !head_matches || !files_match {
            status = Status::Stale;
        }
    }

    let mut published = false;
    if status == Status::Applied && !dry_run {
        match plan.publish.mode {
            PlanPublishMode::Patch => {}
            PlanPublishMode::Atomic => {
                if publications.len() != 1 {
                    status = Status::PublishFailed;
                    append_transaction_failure(
                        &mut reports,
                        "atomic publish requires exactly one file; use journal",
                    );
                } else {
                    let publication = &publications[0];
                    match crate::publish::publish_atomic(
                        root,
                        &root.join(&publication.rel_path),
                        &publication.content,
                        &publication.expected_live_sha256,
                    ) {
                        Ok(_) => published = true,
                        Err(error) => status = crate::certificate::publish_error_status(&error),
                    }
                }
            }
            PlanPublishMode::Journal | PlanPublishMode::ShadowWorktree => {
                match publish_journal_locked(root, &transaction_id, &publications, &lock) {
                    Ok(()) => published = true,
                    Err(error) => status = crate::certificate::publish_error_status(&error),
                }
            }
        }
    }

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
    for report in &mut reports {
        report.guarantees.validators = validator_guarantee;
        if matches!(status, Status::Stale | Status::PublishFailed) {
            report.file_sha256_after = None;
        }
        if status == Status::PublishFailed {
            report.guarantees.no_clobber = Guarantee::Failed;
            report.postconditions_passed = false;
        }
    }

    Ok(Certificate {
        schema_version: crate::certificate::CERTIFICATE_SCHEMA.into(),
        status,
        transaction_id,
        workspace: WorkspaceReport {
            root: plan.workspace.root.clone(),
            git_head_before,
            git_head_after: git_head(root),
        },
        operations: reports,
        validators: validator_reports,
        published,
        publish_mode: certificate_publish_mode(plan.publish.mode, dry_run),
    })
}

fn plan_operation<'a>(
    plan: &Plan,
    operation: &'a PlanOperation,
    file_key: String,
    snapshot: &Snapshot,
) -> PlannedOperation<'a> {
    let (selector_engine, selector_class) = selector_profile(&operation.selector);
    let mut problem = None;
    if plan.workspace.require_unchanged_files && operation.preconditions.file_sha256.is_none() {
        problem = Some(OperationProblem {
            status: Status::Stale,
            name: "file-sha256".into(),
            detail: "require_unchanged_files requires a file_sha256 precondition".into(),
        });
    }
    if operation
        .preconditions
        .file_sha256
        .as_ref()
        .is_some_and(|expected| expected != &snapshot.file_sha256)
    {
        problem = Some(OperationProblem {
            status: Status::Stale,
            name: "file-sha256".into(),
            detail: format!(
                "expected {}, found {}",
                operation
                    .preconditions
                    .file_sha256
                    .as_deref()
                    .unwrap_or_default(),
                snapshot.file_sha256
            ),
        });
    }

    let target_ranges = match &operation.selector {
        PlanSelector::Resolved {
            byte_start,
            byte_end,
        } => {
            if byte_start > byte_end || *byte_end > snapshot.content.len() {
                if problem.is_none() {
                    problem = Some(OperationProblem {
                        status: Status::InvalidResult,
                        name: "selector-range".into(),
                        detail: format!(
                            "resolved range {byte_start}..{byte_end} is outside a {} byte file",
                            snapshot.content.len()
                        ),
                    });
                }
                vec![]
            } else {
                vec![(*byte_start, *byte_end)]
            }
        }
        PlanSelector::Text { old_text, expect } => {
            let found = find_all(&snapshot.content, old_text.as_bytes());
            if found.is_empty() {
                if problem.is_none() {
                    problem = Some(OperationProblem {
                        status: Status::NotFound,
                        name: "selector-cardinality".into(),
                        detail: format!("expected {expect} target(s), found 0"),
                    });
                }
            } else if found.len() != *expect && problem.is_none() {
                problem = Some(OperationProblem {
                    status: Status::Ambiguous,
                    name: "selector-cardinality".into(),
                    detail: format!("expected {expect} target(s), found {}", found.len()),
                });
            }
            found
                .into_iter()
                .map(|start| (start, start + old_text.len()))
                .collect()
        }
    };

    if let Some(expected) = &operation.preconditions.target_sha256 {
        let target_matches = target_ranges.len() == 1
            && snapshot
                .content
                .get(target_ranges[0].0..target_ranges[0].1)
                .is_some_and(|target| sha256_hex(target) == *expected);
        if !target_matches && problem.is_none() {
            problem = Some(OperationProblem {
                status: Status::Stale,
                name: "target-sha256".into(),
                detail: "resolved target no longer matches target_sha256".into(),
            });
        }
    }

    let mut mutations = Vec::with_capacity(target_ranges.len());
    for (match_index, &target_range) in target_ranges.iter().enumerate() {
        mutations.push(action_mutation(operation, target_range, match_index));
    }
    let target_before = concatenate_ranges(&snapshot.content, &target_ranges);
    let target_after = mutations
        .iter()
        .flat_map(|mutation| mutation.replacement.iter().copied())
        .collect();
    let target_matches = target_ranges.len();

    PlannedOperation {
        operation,
        file_key,
        selector_engine,
        selector_class,
        target_ranges,
        mutations,
        target_matches,
        target_before,
        target_after,
        problem,
        overlap_details: vec![],
    }
}

fn action_mutation(
    operation: &PlanOperation,
    target_range: (usize, usize),
    match_index: usize,
) -> PlannedOp {
    let (range, replacement) = match &operation.action {
        PlanAction::Replace { content } => (target_range, content.as_bytes().to_vec()),
        PlanAction::Delete => (target_range, vec![]),
        PlanAction::InsertAfter { content } => {
            let mut replacement = vec![b'\n'];
            replacement.extend_from_slice(content.as_bytes());
            if !content.ends_with('\n') {
                replacement.push(b'\n');
            }
            ((target_range.1, target_range.1), replacement)
        }
        PlanAction::InsertBefore { content } => {
            let mut replacement = content.as_bytes().to_vec();
            if !content.ends_with('\n') {
                replacement.push(b'\n');
            }
            replacement.push(b'\n');
            ((target_range.0, target_range.0), replacement)
        }
    };
    PlannedOp {
        id: format!("{}[{match_index}]", operation.id),
        range,
        replacement,
    }
}

fn selector_profile(selector: &PlanSelector) -> (SelectorEngine, SelectorClass) {
    match selector {
        PlanSelector::Resolved { .. } => (SelectorEngine::Symbol, SelectorClass::Resolved),
        PlanSelector::Text { .. } => (SelectorEngine::Text, SelectorClass::ExactText),
    }
}

fn overlapping_target_ranges(
    first: &[(usize, usize)],
    second: &[(usize, usize)],
) -> Vec<((usize, usize), (usize, usize))> {
    let mut overlaps = Vec::new();
    for &first_range in first {
        for &second_range in second {
            if target_ranges_overlap(first_range, second_range) {
                overlaps.push((first_range, second_range));
            }
        }
    }
    overlaps
}

fn target_ranges_overlap(first: (usize, usize), second: (usize, usize)) -> bool {
    match (first.0 == first.1, second.0 == second.1) {
        (true, true) => first.0 == second.0,
        (true, false) => second.0 <= first.0 && first.0 < second.1,
        (false, true) => first.0 <= second.0 && second.0 < first.1,
        (false, false) => first.0 < second.1 && second.0 < first.1,
    }
}

fn refusal_report(
    planned: &PlannedOperation<'_>,
    snapshot: &Snapshot,
    git_head_stale: bool,
    takeover_reason: Option<&str>,
) -> OperationReport {
    let mut postconditions = Vec::new();
    if let Some(problem) = &planned.problem {
        postconditions.push(PostconditionResult {
            name: problem.name.clone(),
            passed: false,
            detail: Some(problem.detail.clone()),
        });
    }
    for detail in &planned.overlap_details {
        postconditions.push(PostconditionResult {
            name: "non-overlapping-targets".into(),
            passed: false,
            detail: Some(detail.clone()),
        });
    }
    if git_head_stale {
        postconditions.push(PostconditionResult {
            name: "git-head".into(),
            passed: false,
            detail: Some("workspace HEAD does not match expect_git_head".into()),
        });
    }
    if planned.problem.is_none() && planned.overlap_details.is_empty() && !git_head_stale {
        postconditions.push(PostconditionResult {
            name: "transaction".into(),
            passed: false,
            detail: Some("another operation refused the all-or-nothing plan".into()),
        });
    }
    append_takeover_postcondition(&mut postconditions, takeover_reason);
    OperationReport {
        id: planned.operation.id.clone(),
        file: planned.file_key.clone(),
        selector_engine: planned.selector_engine,
        selector_class: planned.selector_class,
        scope_matches: 1,
        target_matches: planned.target_matches,
        file_sha256_before: snapshot.file_sha256.clone(),
        file_sha256_after: None,
        target_sha256_before: sha256_hex(&planned.target_before),
        target_sha256_after: None,
        outside_declared_ranges_unchanged: true,
        changed_byte_ranges: planned
            .mutations
            .iter()
            .map(|mutation| mutation.range)
            .collect(),
        node_before: String::from_utf8(planned.target_before.clone()).ok(),
        node_after: String::from_utf8(planned.target_after.clone()).ok(),
        unified_diff: None,
        syntax: empty_syntax_delta(),
        postconditions_passed: false,
        postconditions,
        residual_occurrences: None,
        guarantees: Guarantees {
            addressed_range: if planned.problem.is_none() && planned.target_matches > 0 {
                Guarantee::Proved
            } else {
                Guarantee::Failed
            },
            no_clobber: Guarantee::Proved,
            byte_isolation: Guarantee::Proved,
            syntax: Guarantee::NotApplicable,
            validators: Guarantee::NotApplicable,
        },
        formatter_expanded_change_scope: false,
        store_refreshed: false,
        candidates: vec![],
    }
}

fn projected_report(
    planned: &PlannedOperation<'_>,
    snapshot: &Snapshot,
    projection: Option<&FileProjection>,
    projection_error: Option<&str>,
    takeover_reason: Option<&str>,
) -> OperationReport {
    let mut postconditions = Vec::new();
    let (file_sha256_after, syntax, syntax_applicable, syntax_ok, isolation_ok, unified_diff) =
        if let Some(projection) = projection {
            postconditions.push(PostconditionResult {
                name: "syntax-no-new-errors".into(),
                passed: projection.syntax_ok,
                detail: (!projection.syntax_ok).then(|| {
                    format!(
                        "new errors: {}, new missing nodes: {}",
                        projection.syntax.new_errors, projection.syntax.new_missing_nodes
                    )
                }),
            });
            postconditions.push(PostconditionResult {
                name: "outside-declared-ranges-unchanged".into(),
                passed: projection.isolation_ok,
                detail: (!projection.isolation_ok)
                    .then(|| "projected file changed outside declared ranges".into()),
            });
            (
                Some(projection.applied.file_sha256.clone()),
                projection.syntax.clone(),
                projection.syntax_applicable,
                projection.syntax_ok,
                projection.isolation_ok,
                Some(crate::verbs::unified_diff_public(
                    &planned.file_key,
                    &snapshot.content,
                    &projection.applied.content,
                )),
            )
        } else {
            postconditions.push(PostconditionResult {
                name: "in-memory-apply".into(),
                passed: false,
                detail: projection_error.map(str::to_owned),
            });
            (None, empty_syntax_delta(), false, false, false, None)
        };
    append_takeover_postcondition(&mut postconditions, takeover_reason);
    let postconditions_passed = projection.is_some() && syntax_ok && isolation_ok;
    OperationReport {
        id: planned.operation.id.clone(),
        file: planned.file_key.clone(),
        selector_engine: planned.selector_engine,
        selector_class: planned.selector_class,
        scope_matches: 1,
        target_matches: planned.target_matches,
        file_sha256_before: snapshot.file_sha256.clone(),
        file_sha256_after,
        target_sha256_before: sha256_hex(&planned.target_before),
        target_sha256_after: Some(sha256_hex(&planned.target_after)),
        outside_declared_ranges_unchanged: isolation_ok,
        changed_byte_ranges: planned
            .mutations
            .iter()
            .map(|mutation| mutation.range)
            .collect(),
        node_before: String::from_utf8(planned.target_before.clone()).ok(),
        node_after: String::from_utf8(planned.target_after.clone()).ok(),
        unified_diff,
        syntax,
        postconditions_passed,
        postconditions,
        residual_occurrences: None,
        guarantees: Guarantees {
            addressed_range: if planned.target_matches > 0 {
                Guarantee::Proved
            } else {
                Guarantee::Failed
            },
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
    }
}

fn lock_refusal_certificate(
    plan: &Plan,
    dry_run: bool,
    transaction_id: String,
    detail: &str,
) -> Certificate {
    let reports = plan
        .operations
        .iter()
        .map(|operation| {
            let (selector_engine, selector_class) = selector_profile(&operation.selector);
            OperationReport {
                id: operation.id.clone(),
                file: operation.file.clone(),
                selector_engine,
                selector_class,
                scope_matches: 0,
                target_matches: 0,
                file_sha256_before: String::new(),
                file_sha256_after: None,
                target_sha256_before: String::new(),
                target_sha256_after: None,
                outside_declared_ranges_unchanged: true,
                changed_byte_ranges: vec![],
                node_before: None,
                node_after: None,
                unified_diff: None,
                syntax: empty_syntax_delta(),
                postconditions_passed: false,
                postconditions: vec![PostconditionResult {
                    name: "workspace-lock".into(),
                    passed: false,
                    detail: Some(detail.to_string()),
                }],
                residual_occurrences: None,
                guarantees: Guarantees {
                    addressed_range: Guarantee::NotApplicable,
                    no_clobber: Guarantee::Proved,
                    byte_isolation: Guarantee::Proved,
                    syntax: Guarantee::NotApplicable,
                    validators: Guarantee::NotApplicable,
                },
                formatter_expanded_change_scope: false,
                store_refreshed: false,
                candidates: vec![],
            }
        })
        .collect();
    Certificate {
        schema_version: crate::certificate::CERTIFICATE_SCHEMA.into(),
        status: Status::PublishFailed,
        transaction_id,
        workspace: WorkspaceReport {
            root: plan.workspace.root.clone(),
            git_head_before: git_head(Path::new(&plan.workspace.root)),
            git_head_after: git_head(Path::new(&plan.workspace.root)),
        },
        operations: reports,
        validators: vec![],
        published: false,
        publish_mode: certificate_publish_mode(plan.publish.mode, dry_run),
    }
}

fn append_takeover_postcondition(
    postconditions: &mut Vec<PostconditionResult>,
    takeover_reason: Option<&str>,
) {
    if let Some(reason) = takeover_reason {
        postconditions.push(PostconditionResult {
            name: "workspace-lock-takeover".into(),
            passed: true,
            detail: Some(reason.to_string()),
        });
    }
}

fn append_transaction_failure(reports: &mut [OperationReport], detail: &str) {
    for report in reports {
        report.postconditions_passed = false;
        report.postconditions.push(PostconditionResult {
            name: "publish-mode".into(),
            passed: false,
            detail: Some(detail.to_string()),
        });
    }
}

fn syntax_delta(
    before: Option<crate::txn::SyntaxCounts>,
    after: Option<crate::txn::SyntaxCounts>,
) -> (SyntaxDelta, bool) {
    match (before, after) {
        (Some(before), Some(after)) => (
            SyntaxDelta {
                errors_before: before.errors,
                errors_after: after.errors,
                new_errors: after.errors.saturating_sub(before.errors),
                new_missing_nodes: after.missing.saturating_sub(before.missing),
            },
            true,
        ),
        _ => (empty_syntax_delta(), false),
    }
}

fn empty_syntax_delta() -> SyntaxDelta {
    SyntaxDelta {
        errors_before: 0,
        errors_after: 0,
        new_errors: 0,
        new_missing_nodes: 0,
    }
}

fn concatenate_ranges(content: &[u8], ranges: &[(usize, usize)]) -> Vec<u8> {
    ranges
        .iter()
        .filter_map(|&(start, end)| content.get(start..end))
        .flatten()
        .copied()
        .collect()
}

fn certificate_publish_mode(mode: PlanPublishMode, dry_run: bool) -> PublishMode {
    if dry_run {
        return PublishMode::DryRun;
    }
    match mode {
        PlanPublishMode::Atomic => PublishMode::Atomic,
        PlanPublishMode::Journal => PublishMode::Journal,
        PlanPublishMode::Patch => PublishMode::Patch,
        PlanPublishMode::ShadowWorktree => PublishMode::ShadowWorktree,
    }
}

fn plan_transaction_id(plan: &Plan) -> String {
    let bytes = serde_json::to_vec(plan).unwrap_or_default();
    format!("ge-plan-{}", &sha256_hex(&bytes)[..16])
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
            .position(|window| window == needle)
        {
            Some(relative) => {
                out.push(from + relative);
                from += relative + needle.len();
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
