//! Building and refreshing the index.
//!
//! Split out of `lib.rs`, which had grown to 26,400 lines: the module still
//! reaches every private helper there through `use super::*`, and nothing about
//! the behaviour changes.

use super::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static INDEX_WARM_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn index_warm_run_id() -> String {
    let sequence = INDEX_WARM_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    format!("index-warm-{nanos}-{}-{sequence}", std::process::id())
}

fn progress_stall_threshold_seconds(phase: Option<&str>) -> u64 {
    match phase {
        // Loading the embedded models includes mapping and validating large
        // assets. It is bounded by the inference path, but 120 seconds is too
        // short on a cold macOS host and produced a false stall diagnosis.
        Some("loading_model") => 600,
        // Base identity/model hashing and checkout validation can precede the
        // delegated graph worker. Keep this bounded, but allow cold disks.
        Some("preparing_base") => 300,
        _ => 120,
    }
}

pub(crate) fn dispatch_index_status(
    json: bool,
    diagnostics: bool,
    root: Option<&str>,
    embedding_args: EmbeddingCliArgs<'_>,
) -> Result<i32> {
    dispatch_index_health_with_detail("index-status", json, root, embedding_args, diagnostics)
}

#[derive(Debug)]
struct IndexRecoveryReport {
    command: &'static str,
    status: &'static str,
    root_path: String,
    active_store: String,
    candidate: Option<String>,
    owner_pid: Option<u32>,
    reason: Option<String>,
}

impl IndexRecoveryReport {
    fn published(&self) -> bool {
        self.status == "published"
    }

    fn rejected(&self) -> bool {
        self.status == "rejected"
    }

    fn as_json(&self) -> serde_json::Value {
        serde_json::json!({
            "command": self.command,
            "status": self.status,
            "root_path": self.root_path,
            "active_store": self.active_store,
            "candidate": self.candidate,
            "owner_pid": self.owner_pid,
            "reason": self.reason,
        })
    }
}

pub(crate) fn dispatch_index_recover(
    path: Option<&str>,
    json: bool,
    root: Option<&str>,
) -> Result<i32> {
    let target = match path {
        Some(path) => absolutize_path(std::path::Path::new(path)),
        None => std::env::current_dir()
            .map_err(|error| Error::io("read current_dir for `greppy index recover`", error))?,
    };
    let effective_root = match root {
        Some(root) => {
            let explicit = absolutize_path(std::path::Path::new(root));
            workspace_locator::resolve_workspace_root(&explicit)
        }
        None => find_repo_root(&target),
    };
    if effective_root.join(".git").is_file() && !target.join(".git").is_file() {
        return Err(Error::Invalid(format!(
            "index recover requires the repository root; use `greppy index recover {}`",
            effective_root.display()
        )));
    }
    let project = workspace_locator::project_identity(&effective_root);
    let store_path = ensured_workspace_store_path(&effective_root)?;
    let _lifecycle = greppy_core::cache::acquire_workspace_lifecycle(
        &effective_root,
        greppy_core::cache::LockMode::Shared,
        false,
    )
    .map_err(|error| Error::io("acquire index recovery lifecycle lease", error))?
    .ok_or_else(|| Error::Lock("blocking lifecycle lease returned no guard".into()))?;
    let _lock = match greppy_freshness::try_acquire(&store_path) {
        Ok(lock) => lock,
        Err(greppy_freshness::LockError::Held { .. }) => {
            return Err(Error::Lock(
                "an index writer is still active; recovery refuses to inspect its snapshot".into(),
            ));
        }
        Err(greppy_freshness::LockError::Io { context, source }) => {
            return Err(Error::io(context, source));
        }
    };
    let options = greppy_indexer::IndexOptions {
        discover_overrides: discover_overrides_from_env()?,
        only_paths: None,
    };
    let report = recover_completed_index_snapshot(
        &store_path,
        &target,
        &effective_root,
        &project,
        &options,
    )?;
    if report.published() {
        let _ = remove_file_if_exists(&background_job_path(&effective_root));
    }
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report.as_json())
                .map_err(|error| Error::Invalid(format!("serialize recovery report: {error}")))?
        );
    } else if report.published() {
        println!(
            "recovered and published {}",
            report
                .candidate
                .as_deref()
                .unwrap_or("completed index snapshot")
        );
    } else if report.rejected() {
        eprintln!(
            "greppy: recovery candidate was not published: {}; run `greppy index {}` to rebuild safely",
            report.reason.as_deref().unwrap_or("validation failed"),
            effective_root.display()
        );
    } else {
        println!("no recoverable index snapshot found");
    }
    Ok(if report.rejected() { EXIT_IO as i32 } else { 0 })
}

fn recover_completed_index_snapshot(
    active_path: &std::path::Path,
    target: &std::path::Path,
    effective_root: &std::path::Path,
    project: &str,
    options: &greppy_indexer::IndexOptions,
) -> Result<IndexRecoveryReport> {
    let root_path = effective_root.to_string_lossy().into_owned();
    let active_store = active_path.to_string_lossy().into_owned();
    let Some(parent) = active_path.parent() else {
        return Err(Error::Invalid("index store has no parent directory".into()));
    };
    let file_name = active_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| Error::Invalid("index store name is not valid UTF-8".into()))?;
    let prefix = format!("{file_name}.next.");
    let mut candidates = std::fs::read_dir(parent)
        .map_err(|error| Error::io(format!("scan {}", parent.display()), error))?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let path = entry.path();
            let name = path.file_name()?.to_str()?;
            if !name.starts_with(&prefix) || name.ends_with("-wal") || name.ends_with("-shm") {
                return None;
            }
            let metadata = std::fs::symlink_metadata(&path).ok()?;
            if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
                return None;
            }
            Some((metadata.modified().ok(), path))
        })
        .collect::<Vec<_>>();
    candidates.sort_by_key(|candidate| std::cmp::Reverse(candidate.0));
    if candidates.is_empty() {
        return Ok(IndexRecoveryReport {
            command: "index-recover",
            status: "no-candidate",
            root_path,
            active_store,
            candidate: None,
            owner_pid: None,
            reason: None,
        });
    }

    let candidates = candidates
        .into_iter()
        .map(|(_, candidate)| {
            let owner_pid = candidate
                .file_name()
                .and_then(|name| name.to_str())
                .and_then(|name| name.strip_prefix(&prefix))
                .and_then(|suffix| suffix.split('.').next())
                .and_then(|pid| pid.parse::<u32>().ok());
            (candidate, owner_pid)
        })
        .collect::<Vec<_>>();

    // A live owner means indexing may still be preparing a publication even
    // if its writer lock is momentarily unavailable or the PID was observed
    // through a surviving candidate. Never publish another snapshot around it.
    if let Some((candidate, owner_pid)) = candidates
        .iter()
        .find(|(_, owner_pid)| owner_pid.as_ref().is_some_and(|pid| process_is_alive(*pid)))
    {
        return Ok(IndexRecoveryReport {
            command: "index-recover",
            status: "rejected",
            root_path,
            active_store,
            candidate: Some(candidate.to_string_lossy().into_owned()),
            owner_pid: *owner_pid,
            reason: Some("candidate owner process is still alive".into()),
        });
    }

    let mut rejected = Vec::new();
    for (candidate, owner_pid) in candidates {
        match validate_index_recovery_candidate(&candidate, target, project, options) {
            Ok(()) => {
                cleanup_sqlite_sidecars(&candidate)?;
                sync_file(&candidate)?;
                sync_parent_dir(&candidate)?;
                publish_store_snapshot(&candidate, active_path)?;
                cleanup_stale_snapshot_artifacts(active_path, true)?;
                return Ok(IndexRecoveryReport {
                    command: "index-recover",
                    status: "published",
                    root_path,
                    active_store,
                    candidate: Some(candidate.to_string_lossy().into_owned()),
                    owner_pid,
                    reason: None,
                });
            }
            Err(error) => rejected.push((candidate, owner_pid, error.to_string())),
        }
    }

    let (candidate, owner_pid, reason) = rejected
        .into_iter()
        .next()
        .expect("non-empty candidate list must produce a rejection");
    Ok(IndexRecoveryReport {
        command: "index-recover",
        status: "rejected",
        root_path,
        active_store,
        candidate: Some(candidate.to_string_lossy().into_owned()),
        owner_pid,
        reason: Some(format!("no safe completed snapshot: {reason}")),
    })
}

#[cfg(test)]
mod rust_repair_recovery_tests {
    use super::*;

    #[test]
    fn standalone_embedding_reuse_accepts_fresh_graph_and_refuses_changed_source() {
        let scratch = tempfile::tempdir().unwrap();
        let root = scratch.path().join("repo");
        std::fs::create_dir_all(root.join("src")).unwrap();
        let source = root.join("src/lib.rs");
        std::fs::write(&source, "pub fn target() {}\n").unwrap();
        let active = scratch.path().join("active.db");
        let mut store = greppy_store::Store::open(&active).unwrap();
        greppy_indexer::index(&mut store, &root, "p").unwrap();
        let options = greppy_indexer::IndexOptions::default();
        validate_standalone_embedding_store(&store, &active, &root, "p", &options).unwrap();
        let generation = store
            .get_workspace_state(root.to_str().unwrap())
            .unwrap()
            .unwrap()
            .graph_generation;
        // The same validator is used before embedding and just before publish.
        std::fs::write(&source, "pub fn changed_after_admission() {}\n").unwrap();
        assert!(
            validate_standalone_embedding_store(&store, &active, &root, "p", &options).is_err()
        );
        assert_eq!(
            store
                .get_workspace_state(root.to_str().unwrap())
                .unwrap()
                .unwrap()
                .graph_generation,
            generation
        );
        assert!(active.exists());
    }

    #[test]
    fn standalone_embedding_reuse_refuses_unpublished_and_wrong_root_graphs() {
        let scratch = tempfile::tempdir().unwrap();
        let root = scratch.path().join("repo");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/lib.rs"), "pub fn target() {}\n").unwrap();
        let active = scratch.path().join("active.db");
        let mut store = greppy_store::Store::open(&active).unwrap();
        let options = greppy_indexer::IndexOptions::default();
        assert!(
            validate_standalone_embedding_store(&store, &active, &root, "p", &options).is_err()
        );
        greppy_indexer::index(&mut store, &root, "p").unwrap();
        assert!(
            validate_standalone_embedding_store(&store, &active, &root, "missing", &options)
                .is_err()
        );
        let other = scratch.path().join("other");
        std::fs::create_dir(&other).unwrap();
        assert!(
            validate_standalone_embedding_store(&store, &active, &other, "p", &options).is_err()
        );
    }

    #[test]
    fn recovery_requires_actual_rust_repair_certification() {
        let scratch = tempfile::tempdir().unwrap();
        let root = scratch.path().join("repo");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/lib.rs"), "pub fn target() {}\n").unwrap();
        let candidate = scratch.path().join("candidate.db");
        {
            let mut store = greppy_store::Store::open(&candidate).unwrap();
            greppy_indexer::index(&mut store, &root, "p").unwrap();
        }
        let options = greppy_indexer::IndexOptions::default();
        validate_index_recovery_candidate(&candidate, &root, "p", &options).unwrap();
        {
            let store = greppy_store::Store::open_with(
                &candidate,
                greppy_store::OpenOptions::query_writer(),
            )
            .unwrap();
            store
                .conn()
                .execute(
                    "DELETE FROM schema_meta WHERE key=?1",
                    [greppy_indexer::RUST_CALLER_EDGES_REPAIR_META_KEY],
                )
                .unwrap();
        }
        let error =
            validate_index_recovery_candidate(&candidate, &root, "p", &options).unwrap_err();
        assert!(
            error.to_string().contains("compatibility preparation"),
            "{error}"
        );
        assert!(
            candidate.exists(),
            "rejected candidate is not silently deleted"
        );
    }
}

fn validate_index_recovery_candidate(
    candidate: &std::path::Path,
    target: &std::path::Path,
    project: &str,
    options: &greppy_indexer::IndexOptions,
) -> Result<()> {
    checkpoint_store_path(candidate)?;
    let store = greppy_store::Store::open_with(candidate, greppy_store::OpenOptions::read_only())?;
    validate_index_snapshot(&store, candidate, target, project, options)
}

// Read-only validation shared by recovery and standalone semantic reuse. The
// writer lease serializes database publication, not external source edits.
fn validate_index_snapshot(
    store: &greppy_store::Store,
    candidate: &std::path::Path,
    target: &std::path::Path,
    project: &str,
    options: &greppy_indexer::IndexOptions,
) -> Result<()> {
    store.integrity_check().map_err(|error| {
        Error::Store(format!(
            "recovery candidate {} failed integrity_check: {error}",
            candidate.display()
        ))
    })?;
    let schema = store.schema_version()?;
    if schema != greppy_store::migrate::CURRENT_VERSION {
        return Err(Error::Store(format!(
            "recovery candidate schema {schema} does not match expected {}",
            greppy_store::migrate::CURRENT_VERSION
        )));
    }
    let project_row = store
        .get_project(project)?
        .ok_or_else(|| Error::Store(format!("recovery candidate lacks project `{project}`")))?;
    if !greppy_indexer::rust_caller_edges_repaired(store)? {
        return Err(Error::Store(
            "recovery candidate requires Rust graph compatibility preparation before publication"
                .into(),
        ));
    }
    let expected_target = absolutize_path(target);
    if absolutize_path(std::path::Path::new(&project_row.root_path)) != expected_target {
        return Err(Error::Store(format!(
            "recovery candidate project root {} does not match {}",
            project_row.root_path,
            expected_target.display()
        )));
    }
    let state = store
        .get_workspace_state(expected_target.to_string_lossy().as_ref())?
        .ok_or_else(|| Error::Store("recovery candidate lacks workspace fingerprint".into()))?;
    if state.schema_version != greppy_store::migrate::CURRENT_VERSION
        || state.indexer_version != greppy_core::INDEXER_VERSION_BASE
    {
        return Err(Error::Store(format!(
            "recovery candidate fingerprint version mismatch (schema={}, indexer={})",
            state.schema_version, state.indexer_version
        )));
    }
    let freshness = greppy_freshness::check_files_report_with_ttl(
        store,
        target,
        project,
        std::time::Duration::from_secs(300),
        &options.discover_overrides,
        std::time::Duration::ZERO,
    )?;
    if !matches!(
        freshness.state.outcome,
        greppy_freshness::FreshnessOutcome::Fresh
    ) {
        return Err(Error::Store(
            "repository HEAD, index signature or discovered files changed after snapshot creation"
                .into(),
        ));
    }
    Ok(())
}

pub(crate) fn dispatch_index_health(
    command: &str,
    json: bool,
    root: Option<&str>,
    embedding_args: EmbeddingCliArgs<'_>,
) -> Result<i32> {
    dispatch_index_health_with_detail(command, json, root, embedding_args, true)
}

fn index_health_output(mut value: serde_json::Value, detailed: bool) -> serde_json::Value {
    if detailed {
        return value;
    }
    let Some(fields) = value.as_object_mut() else {
        return value;
    };
    // Health/ownership counts stay visible. The potentially unbounded provider,
    // stale-path and overlay records are available explicitly, not in every poll.
    for name in [
        "providers",
        "skip_counts_by_reason",
        "integrity_messages",
        "inference",
    ] {
        fields.remove(name);
    }
    for name in ["freshness", "dirty_overlay", "store_cow"] {
        if let Some(summary) = fields
            .get_mut(name)
            .and_then(serde_json::Value::as_object_mut)
        {
            summary.retain(|_, value| !value.is_array() && !value.is_object());
        }
    }
    let diagnostics = fields
        .get("root_path")
        .and_then(serde_json::Value::as_str)
        .map(|root| index_status_command_for_root(std::path::Path::new(root)))
        .unwrap_or_else(|| "greppy index status --json".into());
    fields.insert(
        "diagnostics_command".into(),
        format!("{diagnostics} --diagnostics").into(),
    );
    value
}

#[cfg(test)]
#[test]
fn compact_health_keeps_failed_inference_reason() {
    let detailed = serde_json::json!({
        "healthy": false,
        "fresh": true,
        "embedding_complete": true,
        "inference_healthy": false,
        "inference": {"registry": {"satisfied": false}},
        "providers": [{"name": "rust"}],
        "provider_failure_count": 0,
    });
    let compact = index_health_output(detailed.clone(), false);
    assert_eq!(compact["healthy"], false);
    assert_eq!(compact["inference_healthy"], false);
    assert!(compact.get("inference").is_none());
    assert_eq!(index_health_output(detailed.clone(), true), detailed);
}

fn published_coverage_warning(
    indexed_files: Option<u64>,
    graph_generation: Option<u64>,
    git_tracked: Option<u64>,
) -> Option<String> {
    // An allocated store is not a published graph. In particular, admission
    // can leave an empty store without ever running discovery.
    let (Some(indexed_files), Some(_), Some(tracked)) =
        (indexed_files, graph_generation, git_tracked)
    else {
        return None;
    };
    (tracked >= 100 && indexed_files.saturating_mul(5) < tracked).then(|| {
        format!(
            "store indexed {indexed_files} files but git tracks {tracked} — \
             discovery may be dropping files (nested-repo ignore rules?); \
             re-run `greppy index` with the current binary"
        )
    })
}

#[cfg(test)]
#[test]
fn coverage_warning_requires_publication_but_keeps_real_underindexing_visible() {
    assert!(published_coverage_warning(None, None, Some(1063)).is_none());
    assert!(published_coverage_warning(Some(0), None, Some(1063)).is_none());
    assert!(published_coverage_warning(None, Some(1), Some(1063)).is_none());
    let warning = published_coverage_warning(Some(0), Some(1), Some(1063)).unwrap();
    assert!(warning.contains("store indexed 0 files but git tracks 1063"));
    assert!(published_coverage_warning(Some(100), Some(1), Some(101)).is_none());
    assert!(published_coverage_warning(Some(0), Some(1), Some(99)).is_none());
}

fn background_health_state(
    job: Option<&serde_json::Value>,
    writer_active: bool,
    spawn_active: bool,
    recorded_process_alive: bool,
) -> Option<&'static str> {
    if writer_active {
        return Some("refreshing");
    }
    if spawn_active {
        return Some("starting");
    }
    let job = job?;
    match job.get("state").and_then(serde_json::Value::as_str) {
        Some("failed")
            if job
                .get("preparation_failure_kind")
                .and_then(serde_json::Value::as_str)
                == Some("admission_deferred") =>
        {
            Some("admission_deferred")
        }
        Some("failed") => Some("failed"),
        Some("cancelled") => Some("cancelled"),
        // PID observation may describe a gate wrapper or a reused PID. It
        // cannot prove admission, writer ownership, progress or snapshot safety.
        _ if recorded_process_alive => Some("process_alive"),
        _ => Some("abandoned"),
    }
}

#[cfg(test)]
#[test]
fn background_health_separates_liveness_ownership_and_terminal_outcomes() {
    let job = serde_json::json!({"state": "preparing_base"});
    assert_eq!(
        background_health_state(Some(&job), false, false, true),
        Some("process_alive")
    );
    assert_eq!(
        background_health_state(Some(&job), true, false, false),
        Some("refreshing")
    );
    assert_eq!(
        background_health_state(Some(&job), false, true, false),
        Some("starting")
    );
    assert_eq!(
        background_health_state(Some(&job), false, false, false),
        Some("abandoned")
    );
    assert_eq!(background_health_state(None, false, false, false), None);
    for (state, failure, expected) in [
        ("failed", "admission_deferred", "admission_deferred"),
        ("failed", "preparation_failed", "failed"),
        ("cancelled", "", "cancelled"),
    ] {
        let terminal = serde_json::json!({"state": state, "preparation_failure_kind": failure});
        assert_eq!(
            background_health_state(Some(&terminal), false, false, true),
            Some(expected)
        );
    }
}

fn background_health_observation(
    job: Option<&serde_json::Value>,
    state: Option<&str>,
    recorded_process_alive: bool,
    now: u64,
) -> Option<serde_json::Value> {
    let state = state?;
    let recovery = match state {
        "admission_deferred" => "Shared host admission deferred preparation; wait for capacity, then retry the original command. Do not start duplicate preparation.",
        "cancelled" => "Preparation was cancelled; retry the original work when it is requested again.",
        "failed" => "Inspect background_job.last_error and the original invocation before choosing recovery; do not rebuild merely because a prior job failed.",
        "refreshing" | "starting" | "process_alive" => "Observe the existing job and its owner before starting another index; retry this status command. PID liveness alone does not prove it is making progress.",
        _ => "No writer or startup lease and no live recorded process were observed. Inspect the prior job and current store diagnostics before choosing recovery; a stale journal alone does not require rebuilding a healthy graph.",
    };
    let progress_age_seconds = job
        .and_then(|job| job.get("updated_at_unix_secs"))
        .and_then(serde_json::Value::as_u64)
        .map(|updated| now.saturating_sub(updated));
    let phase = job
        .and_then(|job| job.get("state"))
        .and_then(serde_json::Value::as_str);
    Some(serde_json::json!({
        "recorded_process_alive": recorded_process_alive,
        "process_identity_confirmed": false,
        "recovery": recovery,
        "progress_age_seconds": progress_age_seconds,
        "progress_stale": progress_age_seconds.map(|age| age >= progress_stall_threshold_seconds(phase)),
        "note": "PID liveness does not establish admission, writer ownership or current progress; an old progress timestamp alone does not establish process exit",
    }))
}

#[cfg(test)]
#[test]
fn background_health_guidance_does_not_request_duplicate_or_unnecessary_preparation() {
    let none = background_health_observation(None, None, false, 200);
    assert!(none.is_none());
    assert!(serde_json::to_value(none).unwrap().is_null());
    for (state, expected) in [
        ("admission_deferred", "retry the original command"),
        ("cancelled", "when it is requested again"),
        ("failed", "background_job.last_error"),
        ("abandoned", "healthy graph"),
        ("process_alive", "Observe the existing job"),
    ] {
        let job = serde_json::json!({"updated_at_unix_secs": 1});
        let observation =
            background_health_observation(Some(&job), Some(state), true, 200).unwrap();
        let recovery = observation["recovery"].as_str().unwrap();
        assert!(recovery.contains(expected), "{state}: {recovery}");
        assert!(!recovery.contains("greppy index"), "{state}: {recovery}");
        assert_eq!(observation["progress_stale"], true);
        assert_eq!(observation["process_identity_confirmed"], false);
    }
    // A writer without a journal still supplies useful ownership context.
    assert!(background_health_observation(None, Some("refreshing"), false, 200).is_some());
}

const STATUS_WORKER_ENV: &str = "GREPPY_INTERNAL_STATUS_WORKER";
const STATUS_PHASE_PREFIX: &str = "greppy-status-phase:";

fn status_diagnostic_phase(phase: &str) {
    if std::env::var_os(STATUS_WORKER_ENV).is_some() {
        eprintln!("{STATUS_PHASE_PREFIX}{phase}");
    }
}

#[cfg(unix)]
fn incomplete_status(root: &str, phase: &str, budget_ms: u64) -> serde_json::Value {
    serde_json::json!({
        "command": "index-status", "status": "unknown", "healthy": null,
        "root_path": root, "store_exists": null, "store_format": null,
        "store_bytes": null, "writer_active": null, "startup_active": null,
        "background_job": null, "background_state": null,
        "background_observation": null, "fresh": null, "freshness": null,
        "schema_current": null, "integrity_ok": null, "embedding_complete": null,
        "dirty_overlay": null, "store_cow": null,
        "diagnostics_complete": false, "diagnostic_phase": phase,
        "diagnostic_budget_ms": budget_ms,
        "message": "status diagnostic budget exhausted; health and freshness are unknown; retry status when capacity is available; no rebuild is implied",
    })
}

// A cancellable process boundary is required: a recursive filesystem walk,
// SQLite operation or Git child cannot be safely interrupted inside a thread.
// Only this invocation's fresh process group is terminated, never an indexer.
#[cfg(unix)]
fn bounded_index_status(json: bool, root: Option<&str>) -> Result<i32> {
    use std::io::{BufRead, Read, Write};
    use std::os::unix::process::CommandExt;
    use std::process::{Command, Stdio};
    use std::sync::{Arc, Mutex};
    const BUDGET_MS: u64 = 5_000;
    let mut command =
        Command::new(std::env::current_exe().map_err(|e| Error::io("status executable", e))?);
    command
        .args(std::env::args_os().skip(1))
        .env(STATUS_WORKER_ENV, "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    unsafe {
        command.pre_exec(|| {
            if libc::setpgid(0, 0) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let child = command
        .spawn()
        .map_err(|e| Error::io("spawn bounded status inspection", e))?;
    struct OwnedInspection(Option<std::process::Child>);
    impl Drop for OwnedInspection {
        fn drop(&mut self) {
            if let Some(child) = self.0.as_mut() {
                unsafe {
                    libc::kill(-(child.id() as i32), libc::SIGKILL);
                }
                let _ = child.wait();
            }
        }
    }
    let mut owned = OwnedInspection(Some(child));
    let child = owned.0.as_mut().unwrap();
    let mut stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let output = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        stdout.read_to_end(&mut bytes).map(|_| bytes)
    });
    let phase = Arc::new(Mutex::new(String::from("startup")));
    let observed_phase = Arc::clone(&phase);
    let errors = std::thread::spawn(move || {
        let mut errors = Vec::new();
        for line in std::io::BufReader::new(stderr).split(b'\n') {
            let line = line?;
            if let Some(stage) = String::from_utf8_lossy(&line).strip_prefix(STATUS_PHASE_PREFIX) {
                *observed_phase.lock().unwrap() = stage.to_owned();
            } else {
                errors.extend_from_slice(&line);
                errors.push(b'\n');
            }
        }
        Ok::<_, std::io::Error>(errors)
    });
    let start = std::time::Instant::now();
    let terminal = loop {
        if let Some(status) = owned
            .0
            .as_mut()
            .unwrap()
            .try_wait()
            .map_err(|e| Error::io("wait for status inspection", e))?
        {
            owned.0.take();
            break Some(status);
        }
        if start.elapsed() >= std::time::Duration::from_millis(BUDGET_MS) {
            drop(owned);
            break None;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    };
    let bytes = output
        .join()
        .map_err(|_| Error::Invalid("status output reader failed".into()))?
        .map_err(|e| Error::io("read status output", e))?;
    let error_bytes = errors
        .join()
        .map_err(|_| Error::Invalid("status diagnostic reader failed".into()))?
        .map_err(|e| Error::io("read status diagnostics", e))?;
    if let Some(status) = terminal {
        std::io::stdout()
            .write_all(&bytes)
            .map_err(|e| Error::io("write status output", e))?;
        std::io::stderr()
            .write_all(&error_bytes)
            .map_err(|e| Error::io("write status error", e))?;
        return Ok(status.code().unwrap_or(EXIT_TEMPFAIL as i32));
    }
    let phase = phase.lock().unwrap();
    let status = incomplete_status(root.unwrap_or("."), &phase, BUDGET_MS);
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&status).map_err(|e| Error::Invalid(e.to_string()))?
        );
    } else {
        println!(
            "status: unknown\ndiagnostic_phase: {phase}\nmessage: {}",
            status["message"].as_str().unwrap()
        );
    }
    Ok(EXIT_TEMPFAIL as i32)
}

fn dispatch_index_health_with_detail(
    command: &str,
    json: bool,
    root: Option<&str>,
    embedding_args: EmbeddingCliArgs<'_>,
    detailed: bool,
) -> Result<i32> {
    #[cfg(unix)]
    if command == "index-status" && std::env::var_os(STATUS_WORKER_ENV).is_none() {
        return bounded_index_status(json, root);
    }
    status_diagnostic_phase("resolve_workspace");
    let effective_root = resolve_root(root)?;
    let project = workspace_locator::project_identity(&effective_root);
    let store_path = workspace_locator::store_path(&effective_root);
    let store_format = store_path
        .parent()
        .and_then(|parent| greppy_core::cache::read_store_manifest(parent).ok())
        .map(|manifest| manifest.format_version);
    // Advisory size accounting must not precede writer/progress observation.
    let mut store_bytes = serde_json::Value::Null;
    let background_job = read_background_job(&background_job_path(&effective_root));
    let effective_root_string = effective_root.to_string_lossy().into_owned();
    let writer_active = workspace_writer_active(Some(&effective_root_string));
    let spawn_active = background_job_spawn_active(&effective_root);
    let recorded_process_alive = background_job
        .as_ref()
        .and_then(|job| job.get("pid"))
        .and_then(serde_json::Value::as_u64)
        .and_then(|pid| u32::try_from(pid).ok())
        .filter(|pid| *pid > 0)
        .is_some_and(process_is_alive);
    let background_state = background_health_state(
        background_job.as_ref(),
        writer_active,
        spawn_active,
        recorded_process_alive,
    );
    let background_observation = background_health_observation(
        background_job.as_ref(),
        background_state,
        recorded_process_alive,
        unix_now_secs_cli(),
    );

    // `status` must never queue behind the writer it is meant to observe.
    // Opening the previous graph and running integrity/freshness checks can be
    // expensive while an atomic replacement is underway. Return the writer's
    // lightweight progress record before touching SQLite or walking Git.
    if writer_active {
        let writer_lock = greppy_freshness::lock_path_for(&store_path);
        let progress_age_seconds = background_job
            .as_ref()
            .and_then(|job| job.get("updated_at_unix_secs"))
            .and_then(serde_json::Value::as_u64)
            .map(|updated| unix_now_secs_cli().saturating_sub(updated));
        let phase = background_job
            .as_ref()
            .and_then(|job| job.get("state"))
            .and_then(serde_json::Value::as_str);
        let stall_threshold_seconds = progress_stall_threshold_seconds(phase);
        let progress_stalled =
            progress_age_seconds.is_some_and(|age| age >= stall_threshold_seconds);
        let message = if progress_stalled {
            let phase = phase.unwrap_or("unknown");
            format!(
                "index build has published no progress update for {}s (phase={phase}); it may be stalled; inspect the index invocation that owns writer_lock and its logs; background_job.pid is diagnostic only and must not be signaled without separate ownership proof; the OS lock releases when its actual owner exits",
                progress_age_seconds.unwrap_or(0)
            )
        } else if background_job.is_some() {
            "index build in progress; inspect background_job for phase/progress, wait for completion, then retry".into()
        } else {
            "index writer owns the OS lock but has not published detailed progress yet; phase=starting, retry `greppy index status --json` shortly; the lock is crash-safe and releases when its owning process exits".into()
        };
        let status = serde_json::json!({
            "command": command,
            "status": "indexing",
            "healthy": false,
            "store_exists": store_path.exists(),
            "writer_active": true,
            "startup_active": spawn_active,
            "root_path": effective_root,
            "store_path": store_path,
            "writer_lock": writer_lock,
            "store_format": store_format,
            "store_bytes": store_bytes,
            "background_job": background_job,
            "background_state": "refreshing",
            "background_observation": background_observation,

            "progress_age_seconds": progress_age_seconds,
            "progress_stall_threshold_seconds": stall_threshold_seconds,
            "progress_stalled": progress_stalled,
            "fresh": false,
            "dirty_overlay": null,
            "inference": null,
            "message": message,
        });
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&index_health_output(status.clone(), detailed))
                    .map_err(|error| Error::Invalid(format!(
                        "serialize {command} JSON: {error}"
                    )))?
            );
        } else {
            println!("status: indexing");
            println!("root: {}", effective_root.display());
            println!("store: {}", store_path.display());
            if let Some(job) = status.get("background_job").filter(|job| !job.is_null()) {
                println!(
                    "progress: phase={} completed={}/{} eta_seconds={} workers={}",
                    job.get("state")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("starting"),
                    job.get("completed_spans")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0),
                    job.get("total_spans")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0),
                    job.get("eta_seconds")
                        .and_then(serde_json::Value::as_u64)
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| "unknown".into()),
                    job.get("worker_count")
                        .and_then(serde_json::Value::as_u64)
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| "unknown".into()),
                );
            } else {
                println!("progress: phase=starting (detailed progress not published yet)");
            }
            println!("message: index build in progress; wait for completion, then retry");
        }
        return Ok(EXIT_TEMPFAIL as i32);
    }
    status_diagnostic_phase("cache_size");
    store_bytes = serde_json::json!(store_path
        .parent()
        .map(cache_path_bytes)
        .unwrap_or_default());
    status_diagnostic_phase("git_status");
    let dirty_overlay = dirty_overlay(&effective_root)?;
    let inference = (command == "doctor")
        .then(inference_registry_status)
        .transpose()?;
    let inference_daemons = (command == "doctor").then(|| inference_daemon_status(embedding_args));
    let inference_diagnostics = inference.as_ref().map(|registry| {
        serde_json::json!({
            "registry": registry,
            "daemons": inference_daemons,
            "models": inference_model_status(embedding_args),
        })
    });

    if !store_path.exists() {
        let store_cow = crate::store_cow::diagnostics_without_store(&effective_root, &store_path);
        let status_label = "no_index";
        let message = "no active index; run greppy index first";
        let status = serde_json::json!({
            "command": command,
            "status": status_label,
            "healthy": false,
            "store_exists": false,
            "writer_active": false,
            "startup_active": spawn_active,
            "root_path": effective_root,
            "store_path": store_path,
            "store_format": store_format,
            "store_bytes": store_bytes,
            "background_job": background_job,
            "background_state": background_state,
            "background_observation": background_observation,

            "embedding_complete": false,
            "project": project,
            "fresh": false,
            "freshness": null,
            "schema_current": false,
            "integrity_ok": false,
            "project_present": false,
            "incomplete_provider_count": null,
            "skip_counts_by_reason": [],
            "dirty_overlay": dirty_overlay.to_json(),
            "inference": inference_diagnostics,
            "store_cow": store_cow,
            "message": message,
        });
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&index_health_output(status.clone(), detailed))
                    .map_err(|e| Error::Invalid(format!("serialize {command} JSON: {e}")))?
            );
        } else {
            println!("status: {status_label}");
            println!("root: {}", effective_root.display());
            println!("store: {}", store_path.display());
            println!(
                "store_mode: {}",
                store_cow["mode"].as_str().unwrap_or("single")
            );
            if let Some(identity) = store_cow["base_identity"].as_str() {
                println!("base_identity: {identity}");
            }
            if let Some(reason) = store_cow["fallback_reason"].as_str() {
                println!("store_fallback: {reason}");
            }
            println!("message: run `greppy index {}` first", root.unwrap_or("."));
            if let Some(inference) = &inference {
                print_inference_registry(inference);
            }
            if let Some(daemons) = &inference_daemons {
                print_inference_daemons(daemons);
            }
            if dirty_overlay.git_available && !dirty_overlay.clean {
                println!(
                    "dirty_overlay: total={} staged={} unstaged={} untracked={} deleted={} renamed={} ignored={}",
                    dirty_overlay.total,
                    dirty_overlay.staged_count,
                    dirty_overlay.unstaged_count,
                    dirty_overlay.untracked_count,
                    dirty_overlay.deleted_count,
                    dirty_overlay.renamed_count,
                    dirty_overlay.ignored_count
                );
            }
        }
        return Ok(1);
    }

    status_diagnostic_phase("overlay_binding");
    let overlay = match crate::store_cow::overlay_spec(&effective_root) {
        Ok(overlay) => overlay,
        Err(issue) => {
            let store_cow =
                crate::store_cow::diagnostics_without_store(&effective_root, &store_path);
            let message = issue.to_string();
            let status = serde_json::json!({
                "command": command,
                "status": "unhealthy",
                "healthy": false,
                "store_exists": true,
                "writer_active": false,
                "startup_active": spawn_active,
                "root_path": effective_root,
                "store_path": store_path,
                "store_format": store_format,
                "store_bytes": store_bytes,
                "background_job": background_job,
                "background_state": background_state,
            "background_observation": background_observation,

                "embedding_complete": false,
                "project": project,
                "fresh": false,
                "freshness": null,
                "schema_current": null,
                "integrity_ok": null,
                "project_present": null,
                "incomplete_provider_count": null,
                "skip_counts_by_reason": [],
                "dirty_overlay": dirty_overlay.to_json(),
                "inference": inference_diagnostics,
                "store_cow": store_cow,
                "message": message,
            });
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&index_health_output(status.clone(), detailed))
                        .map_err(|error| {
                            Error::Invalid(format!("serialize {command} JSON: {error}"))
                        })?
                );
            } else {
                println!("status: unhealthy");
                println!("root: {}", effective_root.display());
                println!("store: {}", store_path.display());
                println!("message: {message}");
            }
            return Ok(EXIT_TEMPFAIL as i32);
        }
    };
    status_diagnostic_phase("open_store");
    let store = match overlay {
        Some(overlay) => greppy_store::Store::open_overlay_read_only(
            &overlay.base_path,
            &store_path,
            &overlay.visibility,
        )?,
        None => {
            greppy_store::Store::open_with(&store_path, greppy_store::OpenOptions::read_only())?
        }
    };
    status_diagnostic_phase("base_verification");
    let store_cow = crate::store_cow::diagnostics(&effective_root, &store, &store_path);
    status_diagnostic_phase("graph_integrity");
    let diag = store.diagnostics()?;
    status_diagnostic_phase("source_freshness");
    let freshness = nav_freshness_json(&store, root, &project);
    status_diagnostic_phase("embedding_completion");
    let fresh = freshness
        .get("fresh")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let project_diag = diag.projects.iter().find(|p| p.project.name == project);
    let workspace = diag
        .workspace_states
        .iter()
        .find(|w| w.root_path == effective_root.to_string_lossy());
    let project_present = project_diag.is_some();
    let incomplete_provider_count = project_diag
        .map(|p| p.incomplete_provider_count)
        .unwrap_or(0);
    let provider_states = project_diag
        .map(|p| p.provider_states.clone())
        .unwrap_or_default();
    let provider_failure_count = provider_states
        .iter()
        .filter(|provider| provider.status != "unsupported")
        .map(|provider| provider.files_failed.max(0) as u64)
        .sum::<u64>();
    let skip_counts = project_diag
        .map(|p| {
            p.skip_counts_by_reason
                .iter()
                .map(|s| {
                    serde_json::json!({
                        "reason": s.reason,
                        "count": s.count,
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let stats = project_diag.map(|p| {
        serde_json::json!({
            "files": p.stats.file_count,
            "nodes": p.stats.total_nodes,
            "edges": p.stats.total_edges,
        })
    });
    let graph_generation = workspace.map(|w| w.graph_generation);
    let current_embedding_rows = graph_generation
        .and_then(|generation| {
            store
                .conn()
                .query_row(
                    "SELECT COUNT(*) FROM vector_embeddings WHERE project = ?1 AND graph_generation = ?2",
                    (&project, generation as i64),
                    |row| row.get::<_, i64>(0),
                )
                .ok()
        })
        .unwrap_or(0);
    // Health polling only needs the configured identity. Resolving inference
    // assets here extracts and hashes the entire embedded model on first use.
    let configured_embedding_model = if test_inference_skipped() {
        None
    } else {
        embedding_config_for_daemon_probe(embedding_args)
            .ok()
            .flatten()
    };
    let embedding_complete = graph_generation.is_some_and(|generation| {
        let Some(model) = configured_embedding_model.as_ref() else {
            return false;
        };
        let key = embedding_complete_key(&project);
        store
            .conn()
            .query_row(
                "SELECT value FROM schema_meta WHERE key = ?1",
                [&key],
                |row| row.get::<_, String>(0),
            )
            .ok()
            == Some(format!("{generation}|{}", model.model_id))
    });
    // Robustness (problem dossier, systemic lesson 1&2): silent
    // under-indexing must be VISIBLE. Two independent-oracle checks:
    //   * coverage: compare the store's indexed file count against
    //     `git ls-files` — a discovery bug (out-of-root gitignore leak,
    //     O9-class) shows up as a tiny fraction of the tracked files.
    //   * vectors: a configured embedding model with zero stored vectors
    //     means every semantic query silently degrades to lexical.
    let git_tracked = git_tracked_file_count(&effective_root);
    let coverage_warning = published_coverage_warning(
        project_diag.map(|p| p.stats.file_count as u64),
        graph_generation,
        git_tracked,
    );
    let vectors_missing_with_model = configured_embedding_model.is_some()
        && store
            .vector_model_ids(&project)
            .map(|v| v.is_empty())
            .unwrap_or(false);
    let inference_healthy = inference
        .as_ref()
        .is_none_or(greppy_embed_native::InferenceBackendRegistry::is_satisfied);
    let embedding_healthy = embedding_complete || test_inference_skipped();
    let healthy = diag.schema_current
        && diag.integrity_ok
        && project_present
        && fresh
        && freshness
            .get("metadata_refresh_pending")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        && embedding_healthy
        && provider_failure_count == 0
        && coverage_warning.is_none()
        && inference_healthy
        && background_state != Some("refreshing");
    let status_label = if healthy { "ok" } else { "unhealthy" };

    if json {
        let value = serde_json::json!({
            "command": command,
            "status": status_label,
            "healthy": healthy,
            "store_exists": true,
            "writer_active": false,
            "startup_active": spawn_active,
            "root_path": effective_root,
            "store_path": store_path,
            "store_format": store_format,
            "store_bytes": store_bytes,
            "background_job": background_job,
            "background_state": background_state,
            "background_observation": background_observation,

            "embedding_complete": embedding_complete,
            "current_embedding_rows": current_embedding_rows,
            "project": project,
            "fresh": fresh,
            "freshness": freshness,
            "schema_version": diag.schema_version,
            "expected_schema_version": diag.expected_schema_version,
            "schema_current": diag.schema_current,
            "integrity_ok": diag.integrity_ok,
            "integrity_messages": diag.integrity_messages,
            "project_present": project_present,
            "graph_generation": graph_generation,
            "stats": stats,
            "incomplete_provider_count": incomplete_provider_count,
            "provider_failure_count": provider_failure_count,
            "providers": provider_states,
            "skip_counts_by_reason": skip_counts,
            "git_tracked_files": git_tracked,
            "coverage_warning": coverage_warning,
            "vectors_missing_with_model": vectors_missing_with_model,
            "inference_healthy": inference_healthy,
            "dirty_overlay": dirty_overlay.to_json(),
            "store_cow": store_cow,
            "inference": inference_diagnostics,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&index_health_output(value, detailed))
                .map_err(|e| Error::Invalid(format!("serialize {command} JSON: {e}")))?
        );
    } else {
        println!("status: {status_label}");
        if let Some(w) = &coverage_warning {
            println!("coverage_warning: {w}");
        }
        if vectors_missing_with_model {
            println!(
                "vectors: none stored though an embedding model is configured \
                 — `semantic-search` will build them on first use, or run \
                 `grep index` now"
            );
        }
        println!("root: {}", effective_root.display());
        println!("store: {}", store_path.display());
        println!("store_format: {}", store_format.unwrap_or(0));
        println!("store_bytes: {store_bytes}");
        println!(
            "store_mode: {}",
            store_cow["mode"].as_str().unwrap_or("single")
        );
        if let Some(identity) = store_cow["base_identity"].as_str() {
            println!("base_identity: {identity}");
        }
        if let Some(reason) = store_cow["fallback_reason"].as_str() {
            println!("store_fallback: {reason}");
        }
        println!("embedding_complete: {embedding_complete}");
        if let Some(inference) = &inference {
            print_inference_registry(inference);
        }
        if let Some(daemons) = &inference_daemons {
            print_inference_daemons(daemons);
        }
        if let Some(state) = background_state {
            println!("background_job: {state}");
        }
        println!("project: {project}");
        println!(
            "schema: {}/{} {}",
            diag.schema_version,
            diag.expected_schema_version,
            if diag.schema_current {
                "current"
            } else {
                "stale"
            }
        );
        println!(
            "integrity: {}",
            if diag.integrity_ok { "ok" } else { "failed" }
        );
        println!(
            "freshness: {}",
            if fresh {
                "fresh".to_string()
            } else {
                stale_freshness_reason(&freshness)
            }
        );
        if let Some(generation) = graph_generation {
            println!("generation: {generation}");
        }
        if let Some(project_diag) = project_diag {
            println!(
                "stats: files={} nodes={} edges={}",
                project_diag.stats.file_count,
                project_diag.stats.total_nodes,
                project_diag.stats.total_edges
            );
            println!("incomplete_providers: {incomplete_provider_count}");
            println!("provider_file_failures: {provider_failure_count}");
            for skip in &project_diag.skip_counts_by_reason {
                println!("skipped {} {}", skip.reason, skip.count);
            }
        } else {
            println!("project_present: false");
        }
        if dirty_overlay.git_available && !dirty_overlay.clean {
            println!(
                "dirty_overlay: total={} staged={} unstaged={} untracked={} deleted={} renamed={} ignored={}",
                dirty_overlay.total,
                dirty_overlay.staged_count,
                dirty_overlay.unstaged_count,
                dirty_overlay.untracked_count,
                dirty_overlay.deleted_count,
                dirty_overlay.renamed_count,
                dirty_overlay.ignored_count
            );
        }
    }

    Ok(if healthy { 0 } else { EXIT_IO as i32 })
}

/// Run the indexer against `path` (default: current directory).
/// Warm the worktree `greppy -p` will use, instead of this checkout.
///
/// The built-in agent uses an isolated workspace, accelerated by a healthy
/// provider when available. This command warms that same backend without
/// registering a native Git worktree; the shared immutable index Base remains
/// reusable by later runs while this temporary workspace is removed afterwards.
pub(crate) fn dispatch_index_agent_worktree(
    path: Option<&str>,
    root: Option<&str>,
    embedding_args: EmbeddingCliArgs<'_>,
) -> Result<i32> {
    let repo = resolve_root(root.or(path))?;
    let workspace = greppy_agent::workspace::AgentWorkspace::create(&repo, &index_warm_run_id())
        .map_err(|error| {
            Error::Invalid(format!("no agent worktree for {}: {error}", repo.display()))
        })?;
    let worktree_path = workspace.worktree_path().to_path_buf();
    let worktree = worktree_path.to_string_lossy().into_owned();
    if !cli_json_output() {
        println!("agent worktree: {worktree}");
    }
    let agent_data = workspace.agent_data_root();
    std::fs::create_dir_all(&agent_data).map_err(|error| {
        Error::Invalid(format!(
            "no agent data root at {}: {error}",
            agent_data.display()
        ))
    })?;
    let restore_project = std::env::var_os(greppy_core::PROJECT_IDENTITY_ENV);
    std::env::remove_var(greppy_core::PROJECT_IDENTITY_ENV);
    let logical_project = greppy_core::project_identity(&repo);
    std::env::set_var(greppy_core::PROJECT_IDENTITY_ENV, logical_project);
    let shared_data_root = greppy_core::cache::data_root();
    let cow_env = [
        crate::store_cow::ENV_MODE,
        crate::store_cow::ENV_BASE_PATH,
        crate::store_cow::ENV_BASE_COMMIT,
        crate::store_cow::ENV_BASE_REUSED,
        crate::store_cow::ENV_FALLBACK_REASON,
    ]
    .map(|name| (name, std::env::var_os(name)));
    let prepared_base = match crate::store_cow::prepare_base_store(
        &workspace,
        &shared_data_root,
        embedding_args,
        None,
        None,
    ) {
        Ok(prepared) => {
            if !cli_json_output() {
                println!(
                    "store mode: overlay (Base {}, {})",
                    &prepared.identity_hash[..12],
                    if prepared.reused {
                        "reused"
                    } else {
                        "published"
                    }
                );
            }
            Some(prepared)
        }
        Err(error) => {
            match restore_project {
                Some(previous) => std::env::set_var(greppy_core::PROJECT_IDENTITY_ENV, previous),
                None => std::env::remove_var(greppy_core::PROJECT_IDENTITY_ENV),
            }
            for (name, value) in cow_env {
                match value {
                    Some(previous) => std::env::set_var(name, previous),
                    None => std::env::remove_var(name),
                }
            }
            let cleanup = workspace.cleanup();
            let cleanup_detail = cleanup
                .err()
                .map(|cleanup_error| format!("; workspace cleanup also failed: {cleanup_error}"))
                .unwrap_or_default();
            return Err(Error::Invalid(format!(
                "agent Base prewarm failed closed: {error}{cleanup_detail}"
            )));
        }
    };
    // The agent does not read the operator's data root: `greppy -p` runs with
    // GREPPY_STORE_DIR pointed at an isolated tree beside the worktree, and the
    // sandbox grants only that tree. Warming under the operator's root writes a
    // store the measured run never opens — same workspace key, different data
    // root, so the index is formally warm and practically cold. Point at the
    // agent's root here, exactly as agent.rs does before it runs.
    let restore = std::env::var_os("GREPPY_STORE_DIR");
    std::env::set_var("GREPPY_STORE_DIR", &agent_data);
    if let Some(prepared) = &prepared_base {
        crate::store_cow::configure_overlay_environment(prepared, workspace.base_commit());
    }
    // Walk AND key the store by the worktree, so the identity matches the one
    // the agent resolves; keying by the checkout would warm a third workspace.
    let outcome = dispatch_index(Some(&worktree), Some(&worktree), embedding_args);
    match restore {
        Some(previous) => std::env::set_var("GREPPY_STORE_DIR", previous),
        None => std::env::remove_var("GREPPY_STORE_DIR"),
    }
    match restore_project {
        Some(previous) => std::env::set_var(greppy_core::PROJECT_IDENTITY_ENV, previous),
        None => std::env::remove_var(greppy_core::PROJECT_IDENTITY_ENV),
    }
    for (name, value) in cow_env {
        match value {
            Some(previous) => std::env::set_var(name, previous),
            None => std::env::remove_var(name),
        }
    }
    let cleanup = workspace.cleanup().map_err(|error| {
        Error::Invalid(format!(
            "failed to remove portable index-warm workspace: {error}"
        ))
    });
    match outcome {
        Err(error) => Err(error),
        Ok(code) => cleanup.map(|()| code),
    }
}

pub(crate) fn dispatch_index(
    path: Option<&str>,
    root: Option<&str>,
    embedding_args: EmbeddingCliArgs<'_>,
) -> Result<i32> {
    let mut background_job = BackgroundJobGuard::from_env();
    // RV-006: `--root` overrides the indexed target. When both are
    // given we still walk `path` (the user's workspace) but key the
    // store under the canonical `root` so the indexer and the
    // query commands share one project identity (RV-011).
    // Defect D9: normalize BOTH paths to canonical absolute form up
    // front. `greppy index .` used to record whatever the walker
    // derived from the relative target (falling back to `.` in a
    // marker-less directory), while later queries looked the workspace
    // up under an absolute root — the index existed but every lookup
    // failed. Canonical-absolute at the boundary keeps one spelling
    // everywhere.
    let target = match path {
        Some(p) => absolutize_path(std::path::Path::new(p)),
        None => std::env::current_dir()
            .map_err(|e| Error::io("read current_dir for `grep index` default", e))?,
    };
    // RV-006 / RV-011: the store path and project identity are keyed on
    // the *resolved* repo root, not on the (possibly sub-directory) index
    // target. When `--root` is given we honour it; otherwise we walk up
    // from `target` to the repo marker. This guarantees `greppy index
    // <subdir>` and a later `greppy search-code` from anywhere in the
    // same repo open the same store and use the same project name.
    let effective_root = match root {
        Some(r) => {
            let explicit = absolutize_path(std::path::Path::new(r));
            workspace_locator::resolve_workspace_root(&explicit)
        }
        None => find_repo_root(&target),
    };
    // A linked worktree root is identified by its `.git` file, not by the
    // spelling of its absolute path. Windows can expose the same mounted
    // directory once as a verbatim long path and once through an 8.3 alias;
    // comparing those strings rejects the repository root as if it were a
    // partial subdirectory. A real subdirectory has no `.git` marker of its
    // own and is still rejected here.
    if effective_root.join(".git").is_file() && !target.join(".git").is_file() {
        return Err(Error::Invalid(format!(
            "linked-worktree CoW indexing requires the repository root; run `greppy index --root {}` instead of indexing only {}",
            effective_root.display(),
            target.display()
        )));
    }
    let project = workspace_locator::project_identity(&effective_root);
    let index_options = greppy_indexer::IndexOptions {
        discover_overrides: discover_overrides_from_env()?,
        only_paths: None,
    };
    // Open the on-disk store under the workspace locator's path
    // never at `<root>/.greppy/graph.db` (which would
    // pollute `grep -R .`). The versioned platform data directory is used on
    // Linux/macOS and can be overridden via `GREPPY_STORE_DIR`.
    let store_path = ensured_workspace_store_path(&effective_root)?;
    let _lifecycle = greppy_core::cache::acquire_workspace_lifecycle(
        &effective_root,
        greppy_core::cache::LockMode::Shared,
        false,
    )
    .map_err(|error| Error::io("acquire index lifecycle lease", error))?
    .ok_or_else(|| Error::Lock("blocking lifecycle lease returned no guard".into()))?;
    // Acquire the crash-safe
    // advisory lock BEFORE opening/migrating the store. Opening first lets a
    // concurrent indexer hit a SQLite busy error inside Store::open and exit
    // EXIT_IO (73) silently, instead of the documented EX_TEMPFAIL (75) with a
    // diagnostic on contention. Concurrent indexers on the same path get
    // `LockError::Held`; a crashed prior holder is released by the OS. The
    // guard must outlive the complete snapshot build + publish operation.
    let _lock = match greppy_freshness::try_acquire(&store_path) {
        Ok(lock) => Some(lock),
        Err(greppy_freshness::LockError::Held { .. }) => {
            // Contention is a status, not a dead end: another process is
            // already building the very index this call wanted. Saying only
            // that it is "running" left the caller with nothing to do next --
            // and this fires exactly when a stale-index answer has just told
            // them to run `greppy index`, so the two messages together used to
            // form a loop with no exit.
            eprintln!(
                "grep: another indexer is already building the index for {} — \
                 wait for it to finish, then retry; `greppy index status --json` \
                 reports its progress",
                store_path.display()
            );
            return Ok(EXIT_TEMPFAIL as i32);
        }
        Err(greppy_freshness::LockError::Io { context, source }) => {
            return Err(Error::io(context, source));
        }
    };
    // Claim the portable workspace ownership lock before resolving or
    // materializing inference assets. Background launchers use this lock for
    // their startup handshake and attached-query liveness; doing slow model
    // setup first leaves a PID-only ownership gap.
    // A graph-only command must not wait for model resolution or inference on
    // a cold workspace. Publish the complete structural snapshot first; a
    // later semantic command uses the normal background embedding path for
    // this generation. Explicit `greppy index` keeps its existing policy.
    let structural_first_use = std::env::var_os(crate::ENV_STRUCTURAL_FIRST_USE).is_some();
    let embedding_config = if structural_first_use {
        None
    } else {
        #[cfg(debug_assertions)]
        if std::env::var_os("GREPPY_TEST_FORBID_INDEX_INFERENCE").is_some() {
            return Err(Error::Invalid(
                "structural recovery attempted inference configuration".into(),
            ));
        }
        embedding_config_for_index(embedding_args)?
    };
    let embedding_job =
        std::env::var("GREPPY_BACKGROUND_KIND").ok().as_deref() == Some("embedding");
    let had_overlay_binding = embedding_job
        && crate::store_cow::overlay_environment_for_recovery(&effective_root)?.is_some();
    // Both scoped and unscoped semantic queries can reuse a freshly published
    // standalone graph. Validate in the child after admission; the caller's
    // earlier freshness result cannot protect against intervening source edits.
    let standalone_embedding_reuse = embedding_job
        && !had_overlay_binding
        && validate_standalone_embedding_path(&store_path, &target, &project, &index_options)
            .is_ok();
    let _embedding_overlay = if embedding_job && !standalone_embedding_reuse {
        match crate::store_cow::prepare_auto_linked_worktree_overlay(
            &effective_root,
            &greppy_core::cache::data_root(),
            embedding_args,
            background_job.progress_path(),
        ) {
            Ok(overlay) => overlay,
            Err(error) => {
                // Preparing a linked Base can own a delegated child. Demand
                // cancellation closes that child's owner pipe and records the
                // terminal reason before this error returns; route it through
                // the guard so cancellation is not overwritten by Drop's
                // generic unsuccessful-publication failure.
                background_job.fail(&error);
                return Err(error);
            }
        }
    } else {
        None
    };
    let embedding_overlay = if had_overlay_binding {
        crate::store_cow::overlay_spec_live(&effective_root)?
    } else {
        None
    };
    let embedding_only = embedding_job
        && store_path.is_file()
        && (!effective_root.join(".git").is_file()
            || embedding_overlay.is_some()
            || standalone_embedding_reuse);
    if embedding_only {
        let cfg = embedding_config.as_ref().ok_or_else(|| {
            Error::Invalid("background embedding job has no embedding configuration".into())
        })?;
        background_job.attach_foreground(background_job_path(&effective_root));
        match complete_embeddings_from_published_graph(
            &store_path,
            &target,
            &effective_root,
            &project,
            cfg,
            embedding_overlay.as_ref(),
            if background_job.has_progress_sink() {
                Some(&mut background_job)
            } else {
                None
            },
        ) {
            Ok(EmbeddingBuildOutcome::Complete(_)) => {
                background_job.complete();
                return Ok(0);
            }
            Ok(EmbeddingBuildOutcome::Degraded { reason, .. }) => {
                background_job.degraded(&reason);
                return Ok(0);
            }
            Err(error) => {
                background_job.fail(&error);
                return Err(error);
            }
        }
    }
    let recovery = background_job.publication_boundary(
        || {
            recover_completed_index_snapshot(
                &store_path,
                &target,
                &effective_root,
                &project,
                &index_options,
            )
        },
        |recovery| recovery.published(),
    )?;
    if recovery.published() {
        background_job.publication_finished();
        let _ = remove_file_if_exists(&background_job_path(&effective_root));
        background_job.complete();
        println!(
            "recovered and published completed index snapshot {}",
            recovery.candidate.as_deref().unwrap_or("unknown")
        );
        return Ok(0);
    }
    if recovery.rejected() {
        eprintln!(
            "greppy: stale recovery candidate was not published ({}); rebuilding a fresh snapshot",
            recovery.reason.as_deref().unwrap_or("validation failed")
        );
    }
    background_job.attach_foreground(background_job_path(&effective_root));
    background_job.write_state("preparing_base", None);
    if background_job.is_foreground_owner() && !cli_json_output() {
        eprintln!(
            "greppy: index started for {} (phase=preparing_base, pid={}); progress: `greppy index status --json`",
            effective_root.display(),
            std::process::id()
        );
    }
    let _auto_linked_worktree_overlay = match crate::store_cow::prepare_auto_linked_worktree_overlay(
        &effective_root,
        &greppy_core::cache::data_root(),
        embedding_args,
        background_job.progress_path(),
    ) {
        Ok(overlay) => overlay,
        Err(error) => {
            background_job.fail(&error);
            return Err(error);
        }
    };
    background_job.write_state("indexing", None);
    if let Some(overlay) = crate::store_cow::overlay_spec_live(&effective_root)? {
        let result = index_overlay_snapshot(
            &store_path,
            &target,
            &project,
            &overlay,
            embedding_config.as_ref(),
            &index_options,
            true,
            if background_job.has_progress_sink() {
                Some(&mut background_job)
            } else {
                None
            },
        );
        record_overlay_job_outcome(&mut background_job, &result);
        return result.map(|_| 0);
    }
    // Holding the writer lock, build a fresh snapshot in a temp DB, validate
    // it, then publish it with one filesystem rename. The indexer crate still
    // supports in-place incremental updates for library tests; the CLI path is
    // the production publication boundary, so it must never expose a half-built
    // graph.db to query commands.
    let is_background = background_job.is_background();
    let snapshot = match index_atomic_snapshot(
        &store_path,
        &target,
        &project,
        embedding_config.as_ref(),
        &index_options,
        !is_background,
        if background_job.has_progress_sink() {
            Some(&mut background_job)
        } else {
            None
        },
    ) {
        Ok(snapshot) => snapshot,
        Err(error) => {
            background_job.fail(&error);
            return Err(error);
        }
    };
    background_job.publication_finished();
    let report = &snapshot.index;

    println!(
        "indexed {} files ({} unsupported, {} unreadable, {} oversize, {} file-limit, {} time-budget); {} nodes extracted; generation {} (project: {project})",
        report.files_indexed,
        report.files_unsupported_language,
        report.files_unreadable,
        report.files_oversize,
        report.files_skipped_by_file_limit,
        report.files_skipped_by_time_budget,
        report.nodes_extracted,
        report.graph_generation
    );
    if !report.is_clean()
        || report.files_skipped_by_file_limit > 0
        || report.files_skipped_by_time_budget > 0
    {
        return Ok(EXIT_IO as i32);
    }
    if let Some(embedding_report) = &snapshot.embeddings {
        println!(
            "embedded {} code spans ({} local-store reused, {} global-cache hits, {} inference misses, {} considered, {} non-definition skipped, {} missing-file, {} invalid-span, {} oversize, {} failed, {} stale pruned)",
            embedding_report.nodes_embedded,
            embedding_report.nodes_reused,
            embedding_report.global_cache_hits,
            embedding_report.global_cache_misses,
            embedding_report.nodes_considered,
            embedding_report.nodes_skipped_non_definition,
            embedding_report.nodes_skipped_missing_file,
            embedding_report.nodes_skipped_invalid_span,
            embedding_report.nodes_skipped_oversize,
            embedding_report.nodes_failed,
            embedding_report.stale_rows_pruned
        );
    }
    let discover_scope = index_options.discover_overrides.scope_key();
    if discover_scope != "default" {
        println!(
            "discover scope: {discover_scope} ({} / {})",
            ENV_DISCOVER_INCLUDE, ENV_DISCOVER_EXCLUDE
        );
    }
    retire_verified_legacy_store(&effective_root);
    match snapshot.embedding_degraded.as_deref() {
        // Degraded embeddings never cost the caller the published graph
        // snapshot: record the reason (background job record / stderr) and
        // let the background embed path finish the remaining vectors.
        Some(reason) => background_job.degraded(reason),
        None => background_job.complete(),
    }
    let embedding_deferred = snapshot.embedding_deferred;
    drop(_lock);
    drop(_lifecycle);
    if let Some(reason) = snapshot.embedding_degraded.as_deref() {
        // No immediate respawn: a broken backend would fail the same way
        // again. The next semantic query re-attempts through the existing
        // background-embed path and reuses every vector that DID embed.
        eprintln!(
            "greppy index: embedding generation degraded ({reason}); the graph index is published and complete; the next semantic query retries the remaining embeddings."
        );
    }
    if embedding_deferred {
        if let Some(cfg) = embedding_config.as_ref() {
            let effective_root_string = effective_root.to_string_lossy().into_owned();
            if spawn_background_embed(Some(&effective_root_string), cfg) {
                let progress =
                    embedding_progress_value(&effective_root, cfg, report.graph_generation);
                println!("{}", embedding_progress_text(&progress));
            } else {
                println!(
                    "semantic-search: semantic index is pending; the next semantic query will retry the background job."
                );
            }
        }
    }
    Ok(0)
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum OverlayIndexOutcome {
    Complete,
    Degraded(String),
}

pub(crate) fn record_overlay_job_outcome(
    background_job: &mut BackgroundJobGuard,
    result: &Result<OverlayIndexOutcome>,
) {
    if result.is_ok() {
        background_job.publication_finished();
    }
    match result {
        Ok(OverlayIndexOutcome::Complete) => background_job.complete(),
        Ok(OverlayIndexOutcome::Degraded(reason)) => background_job.degraded(reason),
        Err(error) => background_job.fail(error),
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "atomic Base+Delta publication requires every identity, policy, and progress input explicitly"
)]
pub(crate) fn index_overlay_snapshot(
    active_path: &std::path::Path,
    target: &std::path::Path,
    project: &str,
    overlay: &crate::store_cow::OverlaySpec,
    embedding_config: Option<&EmbeddingModelConfig>,
    index_options: &greppy_indexer::IndexOptions,
    announce: bool,
    mut progress: Option<&mut BackgroundJobGuard>,
) -> Result<OverlayIndexOutcome> {
    cleanup_stale_snapshot_artifacts(active_path, false)?;
    let temp_path = unique_store_sibling(active_path, "delta-building");
    cleanup_sqlite_family(&temp_path)?;
    if overlay.visibility.changed_count() > 0 {
        seed_temp_store_from_active_if_usable(active_path, &temp_path)?;
    }
    {
        let mut delta = greppy_store::Store::open(&temp_path)?;
        if overlay.visibility.changed_count() == 0 && active_path.exists() {
            // A clean overlay starts with an empty private graph, but its
            // publication counter must continue from the active Delta. Without
            // this metadata, each repair/revert republishes generation one and
            // waiting queries cannot distinguish it from their old snapshot.
            let prior_states =
                greppy_store::Store::open_with(active_path, greppy_store::OpenOptions::read_only())
                    .and_then(|active| active.list_private_workspace_states())
                    .map_err(Error::from);
            match prior_states {
                Ok(states) => {
                    for state in states {
                        delta.upsert_workspace_state(&state)?;
                    }
                }
                Err(error) if active_snapshot_is_recoverable(&error) => {}
                Err(error) => return Err(error),
            }
        }
        // A Delta generation contains only paths that still differ from the
        // pinned Base. Exact reverts and removed untracked files therefore
        // discard their former private contributions before the next overlay
        // is constructed.
        for state in delta.list_file_states(project)? {
            if overlay.visibility.is_dirty_path(&state.rel_path) {
                continue;
            }
            delta.delete_nodes_for_file(project, &state.rel_path)?;
            delta.delete_raw_edges_for_file(project, &state.rel_path)?;
            delta.delete_file_content(project, &state.rel_path)?;
            delta.delete_vector_embeddings_for_file(project, &state.rel_path)?;
            delta.delete_index_skip(project, &state.rel_path)?;
            delta.delete_file_state(project, &state.rel_path)?;
        }
        // Resolved edge ids are layer-local and cheap to regenerate from the
        // source-owned raw edge union. Never carry a prior generation's
        // resolution across a changed logical namespace.
        delta
            .conn()
            .execute("DELETE FROM main.edges WHERE project = ?1", [project])
            .map_err(|error| Error::Store(format!("clear prior Delta edges: {error}")))?;
    }

    let mut store =
        greppy_store::Store::open_overlay(&overlay.base_path, &temp_path, &overlay.visibility)?;
    let mut overlay_options = index_options.clone();
    overlay_options.only_paths = Some(
        overlay
            .visibility
            .dirty_paths()
            .map(ToOwned::to_owned)
            .collect(),
    );
    let report = if let Some(job) = progress.as_deref_mut() {
        greppy_indexer::index_with_options_and_progress(
            &mut store,
            target,
            project,
            &overlay_options,
            &mut |update| job.indexing_progress(update),
        )
    } else {
        greppy_indexer::index_with_options(&mut store, target, project, &overlay_options)
    }?;
    greppy_indexer::rebuild_overlay_edges(&mut store, project)?;
    crate::store_cow::mark_rust_caller_edges_repaired(&store)?;
    if !greppy_indexer::rust_caller_edges_repaired(&store)? {
        if let Some(job) = progress.as_deref_mut() {
            job.finalization_phase("repairing_graph");
            maybe_index_test_failpoint("before-rust-repair", &temp_path)?;
        }
        crate::store_cow::complete_visible_overlay_rust_repair(&mut store, target, project)?;
    }
    // The persisted Delta binding is authoritative when structural first use
    // deliberately skips Base preparation. In that path the command-scoped
    // environment has no pinned commit even though the existing overlay does.
    let base_commit = overlay.base_commit.as_str();
    crate::store_cow::persist_visibility(&store, &overlay.visibility, base_commit)?;
    crate::store_cow::persist_overlay_binding(&store, &overlay.base_path, base_commit, project)?;
    let embedding = if let Some(config) = embedding_config {
        Some(index_embeddings_into_temp_store(
            &mut store,
            target,
            project,
            config,
            report.graph_generation,
            active_path.parent().map(std::path::Path::to_path_buf),
            progress.as_deref_mut(),
        )?)
    } else {
        None
    };
    checkpoint_store(&store, &temp_path)?;
    drop(store);
    validate_overlay_snapshot_visibility(&temp_path, Some(overlay))?;
    maybe_index_test_failpoint("after-temp-before-publish", &temp_path)?;
    if let Some(job) = progress {
        job.publication_boundary(|| publish_store_snapshot(&temp_path, active_path), |_| true)?;
    } else {
        publish_store_snapshot(&temp_path, active_path)?;
    }
    cleanup_stale_snapshot_artifacts(active_path, false)?;
    crate::context_status::published(
        &workspace_locator::resolve_workspace_root(target),
        report.graph_generation,
        report.is_clean(),
        matches!(embedding.as_ref(), Some(EmbeddingBuildOutcome::Complete(_))),
    );

    if announce {
        println!(
            "indexed Delta generation {}: {} changed/deleted paths, {} private nodes (project: {project})",
            report.graph_generation,
            overlay.visibility.changed_count(),
            report.nodes_extracted,
        );
    }
    if let Some(EmbeddingBuildOutcome::Degraded { reason, .. }) = embedding {
        eprintln!("greppy: Delta embeddings degraded: {reason}");
        return Ok(OverlayIndexOutcome::Degraded(reason));
    }
    Ok(OverlayIndexOutcome::Complete)
}

pub(crate) fn index_atomic_snapshot(
    active_path: &std::path::Path,
    target: &std::path::Path,
    project: &str,
    embedding_config: Option<&EmbeddingModelConfig>,
    index_options: &greppy_indexer::IndexOptions,
    allow_deferred_embeddings: bool,
    mut background_job: Option<&mut BackgroundJobGuard>,
) -> Result<IndexSnapshotReport> {
    for attempt in 0..2 {
        if let Some(report) = index_atomic_snapshot_attempt(
            active_path,
            target,
            project,
            embedding_config,
            index_options,
            allow_deferred_embeddings,
            background_job.as_deref_mut(),
        )? {
            return Ok(report);
        }
        if attempt == 0 {
            eprintln!("greppy: workspace changed during indexing; rebuilding snapshot once");
        }
    }
    Err(Error::Store(
        "workspace kept changing during indexing; snapshot was not published".into(),
    ))
}

pub(crate) fn index_atomic_snapshot_attempt(
    active_path: &std::path::Path,
    target: &std::path::Path,
    project: &str,
    embedding_config: Option<&EmbeddingModelConfig>,
    index_options: &greppy_indexer::IndexOptions,
    allow_deferred_embeddings: bool,
    mut background_job: Option<&mut BackgroundJobGuard>,
) -> Result<Option<IndexSnapshotReport>> {
    cleanup_stale_snapshot_artifacts(active_path, true)?;
    let temp_path = unique_store_sibling(active_path, "next");
    cleanup_sqlite_family(&temp_path)?;
    seed_temp_store_from_active_if_usable(active_path, &temp_path)?;

    let mut temp_store = match greppy_store::Store::open(&temp_path) {
        Ok(store) => store,
        Err(e) => {
            let _ = cleanup_sqlite_family(&temp_path);
            return Err(e.into());
        }
    };

    let index_result = if let Some(job) = background_job.as_deref_mut() {
        greppy_indexer::index_with_options_and_progress(
            &mut temp_store,
            target,
            project,
            index_options,
            &mut |progress| job.indexing_progress(progress),
        )
    } else {
        greppy_indexer::index_with_options(&mut temp_store, target, project, index_options)
    };
    let report = match index_result {
        Ok(report) => report,
        Err(e) => {
            drop(temp_store);
            let _ = cleanup_sqlite_family(&temp_path);
            return Err(e);
        }
    };

    if !report.is_clean()
        || report.files_skipped_by_file_limit > 0
        || report.files_skipped_by_time_budget > 0
    {
        drop(temp_store);
        cleanup_sqlite_family(&temp_path)?;
        return Ok(Some(IndexSnapshotReport {
            index: report,
            embeddings: None,
            embedding_deferred: false,
            embedding_degraded: None,
        }));
    }

    let embedding_deferred = embedding_config.is_some_and(|cfg| {
        if let Some(job) = background_job.as_deref_mut() {
            job.finalization_phase("counting_embeddings");
        }
        allow_deferred_embeddings
            && greppy_indexer::count_embedding_candidate_nodes(&temp_store, project)
                .is_ok_and(|count| should_defer_embedding(cfg, count))
    });
    let (embedding_report, embedding_degraded) =
        if let Some(cfg) = embedding_config.filter(|_| !embedding_deferred) {
            match index_embeddings_into_temp_store(
                &mut temp_store,
                target,
                project,
                cfg,
                report.graph_generation,
                active_path.parent().map(std::path::Path::to_path_buf),
                background_job.as_deref_mut(),
            ) {
                Ok(EmbeddingBuildOutcome::Complete(report)) => (Some(report), None),
                Ok(EmbeddingBuildOutcome::Degraded { report, reason }) => (report, Some(reason)),
                Err(e) => {
                    drop(temp_store);
                    let _ = cleanup_sqlite_family(&temp_path);
                    return Err(e);
                }
            }
        } else {
            (None, None)
        };

    if let Some(job) = background_job.as_deref_mut() {
        job.finalization_phase("checkpointing_wal");
    }
    drop(temp_store);
    checkpoint_store_path(&temp_path)?;

    if let Some(job) = background_job.as_deref_mut() {
        job.finalization_phase("checking_integrity");
    }
    let integrity_store =
        greppy_store::Store::open_with(&temp_path, greppy_store::OpenOptions::read_only())?;
    integrity_store.integrity_check().map_err(|e| {
        Error::Store(format!(
            "temp index integrity_check failed for {}: {e}",
            temp_path.display()
        ))
    })?;
    drop(integrity_store);

    if let Some(job) = background_job.as_deref_mut() {
        job.finalization_phase("syncing_snapshot");
    }
    cleanup_sqlite_sidecars(&temp_path)?;
    sync_file(&temp_path)?;
    sync_parent_dir(&temp_path)?;
    maybe_index_test_failpoint("after-temp-before-publish", &temp_path)?;

    if let Some(job) = background_job.as_deref_mut() {
        job.finalization_phase("verifying_freshness");
    }
    let verify_store =
        greppy_store::Store::open_with(&temp_path, greppy_store::OpenOptions::read_only())?;
    let verification = greppy_freshness::check_files_report_with_ttl(
        &verify_store,
        target,
        project,
        std::time::Duration::from_secs(300),
        &index_options.discover_overrides,
        std::time::Duration::ZERO,
    )?;
    drop(verify_store);
    if !matches!(
        verification.state.outcome,
        greppy_freshness::FreshnessOutcome::Fresh
    ) {
        cleanup_sqlite_family(&temp_path)?;
        return Ok(None);
    }

    if let Some(job) = background_job.as_deref_mut() {
        job.finalization_phase("publishing_snapshot");
    }
    if let Some(job) = background_job {
        job.publication_boundary(|| publish_store_snapshot(&temp_path, active_path), |_| true)?;
    } else {
        publish_store_snapshot(&temp_path, active_path)?;
    }
    cleanup_stale_snapshot_artifacts(active_path, true)?;
    crate::context_status::published(
        &workspace_locator::resolve_workspace_root(target),
        report.graph_generation,
        report.is_clean()
            && report.files_skipped_by_file_limit == 0
            && report.files_skipped_by_time_budget == 0,
        embedding_report.is_some() && embedding_degraded.is_none() && !embedding_deferred,
    );
    Ok(Some(IndexSnapshotReport {
        index: report,
        embeddings: embedding_report,
        embedding_deferred,
        embedding_degraded,
    }))
}

fn validate_standalone_embedding_path(
    path: &std::path::Path,
    target: &std::path::Path,
    project: &str,
    options: &greppy_indexer::IndexOptions,
) -> Result<()> {
    let store = greppy_store::Store::open_with(path, greppy_store::OpenOptions::read_only())?;
    validate_standalone_embedding_store(&store, path, target, project, options)
}

fn validate_standalone_embedding_store(
    store: &greppy_store::Store,
    path: &std::path::Path,
    target: &std::path::Path,
    project: &str,
    options: &greppy_indexer::IndexOptions,
) -> Result<()> {
    validate_index_snapshot(store, path, target, project, options)?;
    // Preserve semantic search's coverage policy. Metadata mode keeps its
    // explicit partial-provider diagnostics; this shortcut does not certify
    // missing extraction as complete. Strict callers still refuse it.
    if provider_policy_blocks_query(&incomplete_provider_json(store, project)?)? {
        return Err(Error::Store(
            "standalone graph requires provider completion under strict coverage policy".into(),
        ));
    }
    Ok(())
}

fn complete_embeddings_from_published_graph(
    active_path: &std::path::Path,
    target: &std::path::Path,
    effective_root: &std::path::Path,
    project: &str,
    cfg: &EmbeddingModelConfig,
    overlay: Option<&crate::store_cow::OverlaySpec>,
    mut background_job: Option<&mut BackgroundJobGuard>,
) -> Result<EmbeddingBuildOutcome> {
    cleanup_stale_snapshot_artifacts(active_path, true)?;
    let temp_path = unique_store_sibling(active_path, "embedding-next");
    cleanup_sqlite_family(&temp_path)?;
    seed_temp_store_from_active_if_usable(active_path, &temp_path)?;
    let mut store = if let Some(overlay) = overlay {
        greppy_store::Store::open_overlay(&overlay.base_path, &temp_path, &overlay.visibility)?
    } else {
        greppy_store::Store::open(&temp_path)?
    };
    let standalone_options = greppy_indexer::IndexOptions {
        discover_overrides: discover_overrides_from_env()?,
        only_paths: None,
    };
    if overlay.is_none() {
        validate_standalone_embedding_store(
            &store,
            &temp_path,
            target,
            project,
            &standalone_options,
        )?;
    }
    let generation = store
        .get_workspace_state(effective_root.to_string_lossy().as_ref())?
        .ok_or_else(|| Error::Invalid("published graph has no workspace state".into()))?
        .graph_generation;
    let prefixes = background_embedding_path_prefixes()?;
    let outcome = index_embeddings_into_temp_store_scoped(
        &mut store,
        target,
        project,
        cfg,
        generation,
        background_job.as_deref_mut(),
        &prefixes,
    )?;
    let global_complete = embedding_generation_complete(&store, project, generation, &cfg.model_id);
    if let Some(job) = background_job.as_deref_mut() {
        job.finalization_phase("checkpointing_wal");
    }
    checkpoint_store(&store, &temp_path)?;
    drop(store);
    let integrity =
        greppy_store::Store::open_with(&temp_path, greppy_store::OpenOptions::read_only())?;
    if overlay.is_none() {
        // Recheck after inference: a writer lease cannot freeze source files.
        // A stale staged result must never replace the active graph.
        validate_standalone_embedding_store(
            &integrity,
            &temp_path,
            target,
            project,
            &standalone_options,
        )?;
    } else {
        integrity.integrity_check().map_err(|error| {
            Error::Store(format!(
                "embedding snapshot integrity_check failed for {}: {error}",
                temp_path.display()
            ))
        })?;
    }
    drop(integrity);
    validate_overlay_snapshot_visibility(&temp_path, overlay)?;
    cleanup_sqlite_sidecars(&temp_path)?;
    sync_file(&temp_path)?;
    sync_parent_dir(&temp_path)?;
    if let Some(job) = background_job.as_deref_mut() {
        job.finalization_phase("publishing_snapshot");
    }
    if let Some(job) = background_job {
        job.publication_boundary(|| publish_store_snapshot(&temp_path, active_path), |_| true)?;
    } else {
        publish_store_snapshot(&temp_path, active_path)?;
    }
    cleanup_stale_snapshot_artifacts(active_path, true)?;
    crate::context_status::semantic_published(
        effective_root,
        generation,
        // A completed scoped job publishes useful vectors, not global readiness.
        global_complete,
    );
    Ok(outcome)
}

fn validate_overlay_snapshot_visibility(
    snapshot: &std::path::Path,
    overlay: Option<&crate::store_cow::OverlaySpec>,
) -> Result<()> {
    if let Some(overlay) = overlay {
        let store = greppy_store::Store::open_overlay_read_only(
            &overlay.base_path,
            snapshot,
            &overlay.visibility,
        )?;
        let persisted =
            crate::store_cow::cached_visibility_from_connection(store.conn(), &overlay.base_commit)
                .ok_or_else(|| {
                    Error::Invalid(
                        "staged Store-CoW snapshot has no matching visibility manifest".into(),
                    )
                })??;
        if !persisted.dirty_paths().eq(overlay.visibility.dirty_paths())
            || !persisted
                .deleted_paths()
                .eq(overlay.visibility.deleted_paths())
        {
            return Err(Error::Invalid(
                "staged Store-CoW visibility differs from the intended publication".into(),
            ));
        }
        crate::store_cow::validate_overlay_delta_visibility(&store, &persisted)?;
    }
    Ok(())
}

pub(crate) fn index_embeddings_into_temp_store(
    store: &mut greppy_store::Store,
    target: &std::path::Path,
    project: &str,
    cfg: &EmbeddingModelConfig,
    graph_generation: u64,
    _tokenizer_cache_dir: Option<std::path::PathBuf>,
    background_job: Option<&mut BackgroundJobGuard>,
) -> Result<EmbeddingBuildOutcome> {
    index_embeddings_into_temp_store_scoped(
        store,
        target,
        project,
        cfg,
        graph_generation,
        background_job,
        &[],
    )
}

fn index_embeddings_into_temp_store_scoped(
    store: &mut greppy_store::Store,
    target: &std::path::Path,
    project: &str,
    cfg: &EmbeddingModelConfig,
    graph_generation: u64,
    background_job: Option<&mut BackgroundJobGuard>,
    prefixes: &[String],
) -> Result<EmbeddingBuildOutcome> {
    #[cfg(debug_assertions)]
    if std::env::var_os(ENV_TEST_EMBED_UNAVAILABLE).is_some() {
        return Ok(EmbeddingBuildOutcome::Degraded {
            report: None,
            reason: "test failpoint: embedding backend unavailable".into(),
        });
    }
    if test_embedding_completion_forced() {
        let key = if prefixes.is_empty() {
            embedding_complete_key(project)
        } else {
            embedding_scope_complete_key(project, prefixes)
        };
        store
            .conn()
            .execute(
                "INSERT INTO schema_meta(key, value) VALUES (?1, ?2)
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                rusqlite::params![key, format!("{}|{}", graph_generation, cfg.model_id)],
            )
            .map_err(|error| {
                Error::Store(format!("record test embedding completeness: {error}"))
            })?;
        return Ok(EmbeddingBuildOutcome::Complete(
            greppy_indexer::EmbeddingIndexReport {
                nodes_considered: 0,
                nodes_embedded: 0,
                nodes_reused: 0,
                global_cache_hits: 0,
                global_cache_misses: 0,
                nodes_skipped_non_definition: 0,
                nodes_skipped_missing_file: 0,
                nodes_skipped_invalid_span: 0,
                nodes_skipped_oversize: 0,
                stale_rows_pruned: 0,
                nodes_failed: 0,
            },
        ));
    }
    let mut provider = embed_daemon::DaemonCodeEmbeddingProvider::new(cfg);
    let options = greppy_indexer::EmbeddingIndexOptions::for_generation(graph_generation);
    // A structural-only Base may have no meanings yet. A later global query
    // must catch up every visible node before writing a global stamp, while
    // a complete Base retains the inexpensive Delta-only path.
    let include_incomplete_base = prefixes.is_empty()
        && store.is_overlay()
        && !base_embedding_generation_complete(store, project, &cfg.model_id);
    let visible_root_scope = [String::new()];
    let index_prefixes = if include_incomplete_base {
        &visible_root_scope[..]
    } else {
        prefixes
    };
    let mut embedding_report = if let Some(job) = background_job {
        // Exact document counting tokenizes candidate spans. It does not load
        // model weights and must remain observable instead of leaving status
        // frozen at the misleading `loading_model` phase.
        job.finalization_phase("counting_embeddings");
        let total_documents = greppy_indexer::count_code_embedding_documents_for_scope(
            store,
            target,
            project,
            &provider,
            options,
            index_prefixes,
        )?;
        let (backend, device) = provider.backend_plan();
        job.device = device;
        job.embedding_started(&backend, total_documents);
        let mut progress = |value| job.embedding_progress(value);
        greppy_indexer::index_code_embeddings_for_scope_with_progress(
            store,
            target,
            project,
            &mut provider,
            options,
            total_documents,
            &mut progress,
            index_prefixes,
        )?
    } else {
        greppy_indexer::index_code_embeddings_for_scope_with_progress(
            store,
            target,
            project,
            &mut provider,
            options,
            0,
            &mut |_| {},
            index_prefixes,
        )?
    };
    if !embedding_report.is_complete() {
        // The completeness stamp is deliberately withheld: the next
        // semantic query (or the spawned background job) re-runs the
        // embedding pass, reusing every vector that DID embed by content
        // hash and retrying only the failed documents.
        let mut reason = format!(
            "{} of {} embedding documents failed inference",
            embedding_report.nodes_failed,
            embedding_report
                .nodes_failed
                .saturating_add(embedding_report.nodes_embedded)
        );
        if let Some(cause) = provider.last_error() {
            reason.push_str(": ");
            reason.push_str(cause);
        }
        return Ok(EmbeddingBuildOutcome::Degraded {
            report: Some(embedding_report),
            reason,
        });
    }
    if include_incomplete_base {
        embedding_report.stale_rows_pruned =
            store.prune_vector_embeddings_before_generation(project, graph_generation)?;
    }
    let key = if prefixes.is_empty() {
        embedding_complete_key(project)
    } else {
        embedding_scope_complete_key(project, prefixes)
    };
    store
        .conn()
        .execute(
            "INSERT INTO schema_meta(key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            rusqlite::params![key, format!("{}|{}", graph_generation, cfg.model_id)],
        )
        .map_err(|error| Error::Store(format!("record embedding completeness: {error}")))?;
    Ok(EmbeddingBuildOutcome::Complete(embedding_report))
}

#[cfg(test)]
mod progress_status_tests {
    use super::{index_warm_run_id, progress_stall_threshold_seconds};

    #[test]
    fn index_warm_run_ids_are_unique_within_a_process() {
        assert_ne!(index_warm_run_id(), index_warm_run_id());
    }

    #[test]
    fn cold_model_and_base_preparation_use_phase_specific_stall_thresholds() {
        assert_eq!(progress_stall_threshold_seconds(Some("loading_model")), 600);
        assert_eq!(
            progress_stall_threshold_seconds(Some("preparing_base")),
            300
        );
        assert_eq!(
            progress_stall_threshold_seconds(Some("extracting_files")),
            120
        );
        assert_eq!(progress_stall_threshold_seconds(None), 120);
    }
}
