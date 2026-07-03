//! Workspace fingerprint + freshness check.

use std::path::Path;
use std::time::{Duration, Instant};

use grepplus_core::{GitFingerprint, Result};
use grepplus_discover::WalkOverrides;
use grepplus_store::{Store, WorkspaceState};

/// Re-export the shared fingerprint type from core.
pub use grepplus_core::GitFingerprint as WorkspaceFingerprint;

/// Convenience constructor — alias for [`GitFingerprint::capture`].
pub fn capture_fingerprint(root: &Path) -> WorkspaceFingerprint {
    GitFingerprint::capture(root)
}

/// Compare a fingerprint against a persisted workspace state.
fn fingerprint_matches(
    fp: &WorkspaceFingerprint,
    ws: &WorkspaceState,
    expected_indexer_version: &str,
) -> bool {
    let other_root = std::path::PathBuf::from(&ws.root_path);
    let roots_match = fp.canonical_root == other_root
        || fp.canonical_root == other_root.canonicalize().unwrap_or(other_root.clone());
    let head_match = fp.head_oid == ws.head_oid;
    let index_match = fp.index_signature == ws.index_signature;
    let indexer_match = ws.indexer_version == expected_indexer_version;
    roots_match && head_match && index_match && indexer_match
}

/// One row of the on-demand freshness check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FreshnessOutcome {
    /// No persisted workspace state; treat as a cold start.
    Cold,
    /// Fingerprint matches the persisted state; graph is fresh.
    Fresh,
    /// Fingerprint disagrees; graph is stale.
    Stale { reasons: Vec<String> },
    /// Persisted state exists but points at a different root path.
    /// Caller should usually trigger a full reindex.
    RootMismatch,
}

/// Bundled result so callers can decide whether to fall back to strict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FreshnessState {
    pub outcome: FreshnessOutcome,
    pub elapsed: Duration,
}

/// Compute the freshness check against the persisted workspace state.
/// Uses an overall budget of `budget`; if exceeded, returns a
/// `Stale` outcome with `Budget exceeded` in the reasons list.
pub fn check(
    store: &Store,
    current: &WorkspaceFingerprint,
    budget: Duration,
) -> Result<FreshnessState> {
    check_with_overrides(store, current, budget, &WalkOverrides::empty())
}

/// Workspace-level freshness check for a specific discovery override scope.
pub fn check_with_overrides(
    store: &Store,
    current: &WorkspaceFingerprint,
    budget: Duration,
    overrides: &WalkOverrides,
) -> Result<FreshnessState> {
    let start = Instant::now();
    let expected_indexer_version = expected_indexer_version(overrides);
    let row = {
        let mut stmt = store
            .conn()
            .prepare(
                "SELECT root_path, git_dir, git_common_dir, head_oid, index_signature,
                        schema_version, indexer_version, graph_generation, updated_at
                 FROM workspace_state
                 ORDER BY updated_at DESC LIMIT 50",
            )
            .map_err(grepplus_store::Error::Sqlite)?;
        let rows = stmt
            .query_map([], |row| {
                let schema_version_i64: i64 = row.get(5)?;
                let graph_generation_i64: i64 = row.get(7)?;
                Ok(WorkspaceState {
                    root_path: row.get(0)?,
                    git_dir: row.get(1)?,
                    git_common_dir: row.get(2)?,
                    head_oid: row.get(3)?,
                    index_signature: row.get(4)?,
                    schema_version: schema_version_i64 as u32,
                    indexer_version: row.get(6)?,
                    graph_generation: graph_generation_i64 as u64,
                    updated_at: row.get(8)?,
                })
            })
            .map_err(grepplus_store::Error::Sqlite)?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(grepplus_store::Error::Sqlite)?;
        // Find the row whose root_path matches our current canonical root,
        // comparing canonicalised forms. macOS canonicalises /tmp to
        // /private/tmp; we must compare the resolved forms.
        rows.into_iter()
            .find(|w| paths_match(&w.root_path, &current.canonical_root))
    };

    if start.elapsed() > budget {
        return Ok(FreshnessState {
            outcome: FreshnessOutcome::Stale {
                reasons: vec!["budget exceeded".into()],
            },
            elapsed: start.elapsed(),
        });
    }

    let outcome = match row {
        None => FreshnessOutcome::Cold,
        Some(persisted) if !paths_match(&persisted.root_path, &current.canonical_root) => {
            FreshnessOutcome::RootMismatch
        }
        Some(persisted) if fingerprint_matches(current, &persisted, &expected_indexer_version) => {
            FreshnessOutcome::Fresh
        }
        Some(persisted) => {
            let mut reasons = Vec::new();
            if persisted.head_oid != current.head_oid {
                reasons.push(format!(
                    "head_oid changed (was {:?}, now {:?})",
                    persisted.head_oid, current.head_oid
                ));
            }
            if persisted.index_signature != current.index_signature {
                reasons.push(format!(
                    "index signature changed (was {:?}, now {:?})",
                    persisted.index_signature, current.index_signature
                ));
            }
            if persisted.indexer_version != expected_indexer_version {
                reasons.push(format!(
                    "indexer version/scope changed (was {}, expected {})",
                    persisted.indexer_version, expected_indexer_version
                ));
            }
            FreshnessOutcome::Stale { reasons }
        }
    };

    Ok(FreshnessState {
        outcome,
        elapsed: start.elapsed(),
    })
}

fn expected_indexer_version(overrides: &WalkOverrides) -> String {
    let scope = overrides.scope_key();
    if scope == "default" {
        grepplus_core::INDEXER_VERSION_BASE.into()
    } else {
        format!(
            "{};discover_scope={scope}",
            grepplus_core::INDEXER_VERSION_BASE
        )
    }
}

fn paths_match(a: &str, b: &Path) -> bool {
    let pa = std::path::PathBuf::from(a);
    let pb = b;
    pa == pb
        || pa.canonicalize().ok() == pb.canonicalize().ok()
        || pa == pb.canonicalize().unwrap_or_else(|_| pb.to_path_buf())
}

/// File-level freshness check.
///
/// Compares the current on-disk inventory under `root` against the
/// persisted `file_state` rows for the given `project`. Returns
/// `Fresh` if no file has changed; `Stale` with reasons otherwise.
///
/// This is the per-file check that phasenplan §10.4 ("betroffene
/// Suchpfade fresh") requires — the workspace-level `check` only
/// looks at the git fingerprint, so it would not notice a file edit
/// that has not been staged.
pub fn check_files(
    store: &Store,
    root: &Path,
    project: &str,
    budget: Duration,
) -> Result<FreshnessState> {
    check_files_with_overrides(store, root, project, budget, &WalkOverrides::empty())
}

/// File-level freshness check using the same discovery override scope as
/// the index that produced the file_state rows.
pub fn check_files_with_overrides(
    store: &Store,
    root: &Path,
    project: &str,
    budget: Duration,
    overrides: &WalkOverrides,
) -> Result<FreshnessState> {
    Ok(check_files_report_with_overrides(store, root, project, budget, overrides)?.state)
}

/// Detailed file-level freshness report (defect D2).
///
/// The plain [`check_files_with_overrides`] verdict is enough for a
/// yes/no gate, but the D2 fail-open policy needs to know *how* stale
/// the index is: a handful of changed files can be auto-reindexed
/// inline, and a labeled-stale answer should tell the agent how many
/// files drifted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileFreshnessReport {
    pub state: FreshnessState,
    /// True when the verdict came from the freshness-verified TTL stamp
    /// (no git fingerprint capture, no walk, no diff was performed).
    pub ttl_hit: bool,
    /// Workspace-relative paths that differ from the persisted
    /// `file_state` (added / modified / deleted). `None` when they could
    /// not be enumerated (cold store, root mismatch, budget exhausted or
    /// walk failure before the diff ran).
    pub changed_paths: Option<Vec<String>>,
    /// Files in the current on-disk inventory. `None` if the walk did
    /// not run (on a TTL hit this is the inventory size recorded by the
    /// stamp-writing check).
    pub total_inventory: Option<usize>,
}

/// TTL for the freshness-verified stamp (see [`FileFreshnessReport`]):
/// once a check has *proven* the index fresh, subsequent checks against
/// the same store/root/project/scope within this window skip the git
/// fingerprint + walk entirely. Agent loops fire query bursts; the walk
/// is paid once per burst instead of once per query. Trade-off: an edit
/// made within the window is served from the index unlabeled for at
/// most this long. Override with `GREPPLUS_FRESHNESS_TTL_MS` (0
/// disables).
pub const DEFAULT_FRESHNESS_TTL: Duration = Duration::from_millis(2_000);

/// Env var overriding [`DEFAULT_FRESHNESS_TTL`] in milliseconds.
pub const ENV_FRESHNESS_TTL_MS: &str = "GREPPLUS_FRESHNESS_TTL_MS";

fn freshness_ttl_from_env() -> Duration {
    match std::env::var(ENV_FRESHNESS_TTL_MS) {
        Ok(raw) => match raw.trim().parse::<u64>() {
            Ok(ms) => Duration::from_millis(ms),
            Err(_) => DEFAULT_FRESHNESS_TTL,
        },
        Err(_) => DEFAULT_FRESHNESS_TTL,
    }
}

/// [`check_files_with_overrides`], but returning the detailed
/// [`FileFreshnessReport`]. TTL comes from the environment
/// (`GREPPLUS_FRESHNESS_TTL_MS`, default 2000 ms).
pub fn check_files_report_with_overrides(
    store: &Store,
    root: &Path,
    project: &str,
    budget: Duration,
    overrides: &WalkOverrides,
) -> Result<FileFreshnessReport> {
    check_files_report_with_ttl(
        store,
        root,
        project,
        budget,
        overrides,
        freshness_ttl_from_env(),
    )
}

/// Explicit-TTL variant (tests and callers that manage the TTL policy
/// themselves). `Duration::ZERO` disables both stamp reads and writes.
pub fn check_files_report_with_ttl(
    store: &Store,
    root: &Path,
    project: &str,
    budget: Duration,
    overrides: &WalkOverrides,
    ttl: Duration,
) -> Result<FileFreshnessReport> {
    let start = Instant::now();
    let scope = overrides.scope_key();
    let stamp_path = stamp_path_for(store);

    // TTL tier: a recent check already proved this exact
    // (store, root, project, scope) fresh — skip everything.
    if ttl > Duration::ZERO {
        if let Some(path) = stamp_path.as_deref() {
            if let Some(stamp_total) = read_valid_stamp(path, root, project, &scope, ttl) {
                return Ok(FileFreshnessReport {
                    state: FreshnessState {
                        outcome: FreshnessOutcome::Fresh,
                        elapsed: start.elapsed(),
                    },
                    ttl_hit: true,
                    changed_paths: None,
                    total_inventory: Some(stamp_total),
                });
            }
        }
    }

    // Workspace-level check (git fingerprint vs persisted state).
    let fp = WorkspaceFingerprint::capture(root);
    let ws = check_with_overrides(store, &fp, budget, overrides)?;
    // Cold / RootMismatch mean there is no usable index for this root:
    // the file-level diff has nothing to compare against.
    if matches!(
        ws.outcome,
        FreshnessOutcome::Cold | FreshnessOutcome::RootMismatch
    ) {
        return Ok(FileFreshnessReport {
            state: ws,
            ttl_hit: false,
            changed_paths: None,
            total_inventory: None,
        });
    }
    let mut reasons = match &ws.outcome {
        FreshnessOutcome::Stale { reasons } => reasons.clone(),
        _ => Vec::new(),
    };
    if start.elapsed() > budget {
        reasons.push("budget exceeded".into());
        return Ok(FileFreshnessReport {
            state: FreshnessState {
                outcome: FreshnessOutcome::Stale { reasons },
                elapsed: start.elapsed(),
            },
            ttl_hit: false,
            changed_paths: None,
            total_inventory: None,
        });
    }

    // Walk the current inventory and compare to persisted file_state.
    // D2: this runs even when the workspace-level check is already
    // Stale (a commit / staging drift), because the caller needs the
    // changed-file set to decide between inline auto-reindex and a
    // labeled-stale answer.
    let entries = match grepplus_discover::walk_with_policy_and_overrides(
        root,
        &grepplus_discover::SkipPolicy::walk_default(),
        overrides,
    ) {
        Ok(e) => e,
        Err(e) => {
            reasons.push(format!("discover walk failed: {e}"));
            return Ok(FileFreshnessReport {
                state: FreshnessState {
                    outcome: FreshnessOutcome::Stale { reasons },
                    elapsed: start.elapsed(),
                },
                ttl_hit: false,
                changed_paths: None,
                total_inventory: None,
            });
        }
    };
    let inventory: Vec<grepplus_discover::InventoryEntry> = entries.into_iter().collect();
    let total_inventory = inventory.len();
    let diffs = crate::incremental::compute_file_diff(store, project, &inventory)?;
    if start.elapsed() > budget {
        reasons.push("budget exceeded".into());
        return Ok(FileFreshnessReport {
            state: FreshnessState {
                outcome: FreshnessOutcome::Stale { reasons },
                elapsed: start.elapsed(),
            },
            ttl_hit: false,
            changed_paths: None,
            total_inventory: Some(total_inventory),
        });
    }
    let mut changed = Vec::new();
    let (mut any_added, mut any_modified, mut any_deleted) = (false, false, false);
    for d in &diffs {
        match d {
            crate::incremental::FileDiff::Added(e) => {
                any_added = true;
                changed.push(e.rel_path.clone());
            }
            crate::incremental::FileDiff::Modified { entry, .. } => {
                any_modified = true;
                changed.push(entry.rel_path.clone());
            }
            crate::incremental::FileDiff::Deleted(rel) => {
                any_deleted = true;
                changed.push(rel.clone());
            }
            crate::incremental::FileDiff::Unchanged => {}
        }
    }
    // Keep the historical reason strings verbatim: callers and tests
    // match on these.
    if any_added {
        reasons.push("files added since last index".into());
    }
    if any_modified {
        reasons.push("files modified since last index".into());
    }
    if any_deleted {
        reasons.push("files deleted since last index".into());
    }

    if reasons.is_empty() {
        // Proven fresh — record the stamp so the next check within the
        // TTL can skip the walk. Failures are ignored: the stamp is a
        // pure optimisation.
        if ttl > Duration::ZERO {
            if let Some(path) = stamp_path.as_deref() {
                let _ = write_stamp(path, root, project, &scope, total_inventory);
            }
        }
        Ok(FileFreshnessReport {
            state: FreshnessState {
                outcome: FreshnessOutcome::Fresh,
                elapsed: start.elapsed(),
            },
            ttl_hit: false,
            changed_paths: Some(changed),
            total_inventory: Some(total_inventory),
        })
    } else {
        Ok(FileFreshnessReport {
            state: FreshnessState {
                outcome: FreshnessOutcome::Stale { reasons },
                elapsed: start.elapsed(),
            },
            ttl_hit: false,
            changed_paths: Some(changed),
            total_inventory: Some(total_inventory),
        })
    }
}

/// Where the TTL stamp for this store lives: `<graph.db>.freshstamp`,
/// derived from the open connection's backing file. In-memory stores
/// (tests) have no path — no stamp, TTL is a no-op.
fn stamp_path_for(store: &Store) -> Option<std::path::PathBuf> {
    let raw = store.conn().path()?;
    if raw.is_empty() {
        return None; // in-memory
    }
    Some(std::path::PathBuf::from(format!("{raw}.freshstamp")))
}

/// Stamp format (line-oriented, no serde dependency):
/// `v1\n<verified_unix_ns>\n<total_inventory>\n<root>\n<project>\n<scope>\n`
fn write_stamp(
    path: &Path,
    root: &Path,
    project: &str,
    scope: &str,
    total_inventory: usize,
) -> std::io::Result<()> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as i128)
        .unwrap_or(0);
    let body = format!(
        "v1\n{now}\n{total_inventory}\n{}\n{project}\n{scope}\n",
        root.display()
    );
    // Atomic-enough: write a unique temp sibling, rename over the stamp.
    let tmp = path.with_extension(format!("freshstamp.tmp{}", std::process::id()));
    std::fs::write(&tmp, body)?;
    std::fs::rename(&tmp, path)
}

/// Returns the stamped `total_inventory` when the stamp matches this
/// (root, project, scope) and is younger than `ttl`.
fn read_valid_stamp(
    path: &Path,
    root: &Path,
    project: &str,
    scope: &str,
    ttl: Duration,
) -> Option<usize> {
    let body = std::fs::read_to_string(path).ok()?;
    let mut lines = body.lines();
    if lines.next()? != "v1" {
        return None;
    }
    let verified_unix_ns: i128 = lines.next()?.parse().ok()?;
    let total_inventory: usize = lines.next()?.parse().ok()?;
    if lines.next()? != root.display().to_string() {
        return None;
    }
    if lines.next()? != project {
        return None;
    }
    if lines.next()? != scope {
        return None;
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as i128)
        .unwrap_or(0);
    let age = now - verified_unix_ns;
    // Negative age = clock skew or corrupted stamp: never trust it.
    if age < 0 || age > ttl.as_nanos() as i128 {
        return None;
    }
    Some(total_inventory)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::process::Command;

    #[test]
    fn capture_non_git_root_has_no_git_fields() {
        let tmp = tempdir_via_env();
        let fp = WorkspaceFingerprint::capture(&tmp);
        assert!(
            fp.git_dir.is_none(),
            "non-git workspace must have no git_dir"
        );
        assert!(fp.head_oid.is_none());
        assert!(fp.index_signature.is_none());
    }

    #[test]
    fn capture_git_root_records_head_and_index_signature() {
        let tmp = tempdir_via_env();
        run_git(&tmp, &["init", "-q"]);
        // git index is only created when at least one file is added.
        std::fs::write(tmp.join("README.md"), "# test\n").unwrap();
        run_git(&tmp, &["add", "README.md"]);
        run_git(
            &tmp,
            &[
                "-c",
                "user.email=t@t",
                "-c",
                "user.name=t",
                "commit",
                "-q",
                "-m",
                "init",
            ],
        );
        let fp = WorkspaceFingerprint::capture(&tmp);
        assert!(fp.git_dir.is_some(), "git workspace must record git_dir");
        assert!(fp.head_oid.is_some(), "git workspace must record HEAD oid");
        assert!(
            fp.index_signature.is_some(),
            "git workspace with staged files must have an index signature"
        );
    }

    #[test]
    fn check_cold_when_no_persisted_state() {
        let store = Store::open_memory().unwrap();
        let tmp = tempdir_via_env();
        let fp = WorkspaceFingerprint::capture(&tmp);
        let r = check(&store, &fp, Duration::from_secs(1)).unwrap();
        assert_eq!(r.outcome, FreshnessOutcome::Cold);
    }

    #[test]
    fn check_fresh_when_persisted_matches() {
        let mut store = Store::open_memory().unwrap();
        let tmp = tempdir_via_env();
        run_git(&tmp, &["init", "-q"]);
        run_git(
            &tmp,
            &[
                "-c",
                "user.email=t@t",
                "-c",
                "user.name=t",
                "commit",
                "--allow-empty",
                "-q",
                "-m",
                "init",
            ],
        );
        let fp = WorkspaceFingerprint::capture(&tmp);
        store
            .upsert_workspace_state(&WorkspaceState {
                root_path: fp.canonical_root.to_string_lossy().into_owned(),
                git_dir: fp
                    .git_dir
                    .as_ref()
                    .map(|p| p.to_string_lossy().into_owned()),
                git_common_dir: fp
                    .git_common_dir
                    .as_ref()
                    .map(|p| p.to_string_lossy().into_owned()),
                head_oid: fp.head_oid.clone(),
                index_signature: fp.index_signature.clone(),
                schema_version: 1,
                indexer_version: "grepplus-indexer-v1".into(),
                graph_generation: 1,
                updated_at: "2026-06-28T20:00:00Z".into(),
            })
            .unwrap();
        let r = check(&store, &fp, Duration::from_secs(1)).unwrap();
        assert_eq!(r.outcome, FreshnessOutcome::Fresh);
    }

    #[test]
    fn check_stale_when_discovery_scope_changed() {
        let mut store = Store::open_memory().unwrap();
        let tmp = tempdir_via_env();
        let fp = WorkspaceFingerprint::capture(&tmp);
        let scoped = WalkOverrides::empty().include("src/*.rs");
        store
            .upsert_workspace_state(&WorkspaceState {
                root_path: fp.canonical_root.to_string_lossy().into_owned(),
                git_dir: fp
                    .git_dir
                    .as_ref()
                    .map(|p| p.to_string_lossy().into_owned()),
                git_common_dir: fp
                    .git_common_dir
                    .as_ref()
                    .map(|p| p.to_string_lossy().into_owned()),
                head_oid: fp.head_oid.clone(),
                index_signature: fp.index_signature.clone(),
                schema_version: 1,
                indexer_version: expected_indexer_version(&scoped),
                graph_generation: 1,
                updated_at: "2026-06-28T20:00:00Z".into(),
            })
            .unwrap();

        let r = check(&store, &fp, Duration::from_secs(1)).unwrap();
        match r.outcome {
            FreshnessOutcome::Stale { reasons } => assert!(
                reasons.iter().any(|s| s.contains("indexer version/scope")),
                "expected scope mismatch reason, got {reasons:?}"
            ),
            other => panic!("expected stale scope mismatch, got {other:?}"),
        }
    }

    #[test]
    fn check_stale_when_head_oid_changed() {
        let mut store = Store::open_memory().unwrap();
        let tmp = tempdir_via_env();
        run_git(&tmp, &["init", "-q"]);
        run_git(
            &tmp,
            &[
                "-c",
                "user.email=t@t",
                "-c",
                "user.name=t",
                "commit",
                "--allow-empty",
                "-q",
                "-m",
                "v1",
            ],
        );
        let fp = WorkspaceFingerprint::capture(&tmp);
        store
            .upsert_workspace_state(&WorkspaceState {
                root_path: fp.canonical_root.to_string_lossy().into_owned(),
                git_dir: fp
                    .git_dir
                    .as_ref()
                    .map(|p| p.to_string_lossy().into_owned()),
                git_common_dir: fp
                    .git_common_dir
                    .as_ref()
                    .map(|p| p.to_string_lossy().into_owned()),
                head_oid: Some("deadbeef".into()), // mismatch on purpose
                index_signature: fp.index_signature.clone(),
                schema_version: 1,
                indexer_version: "grepplus-indexer-v1".into(),
                graph_generation: 1,
                updated_at: "2026-06-28T20:00:00Z".into(),
            })
            .unwrap();
        let r = check(&store, &fp, Duration::from_secs(1)).unwrap();
        match r.outcome {
            FreshnessOutcome::Stale { reasons } => {
                assert!(reasons.iter().any(|s| s.contains("head_oid")));
            }
            other => panic!("expected Stale, got {other:?}"),
        }
    }

    #[test]
    fn check_stale_when_git_index_signature_changed_without_commit() {
        let mut store = Store::open_memory().unwrap();
        let tmp = tempdir_via_env();
        run_git(&tmp, &["init", "-q"]);
        std::fs::write(tmp.join("tracked.txt"), "v1\n").unwrap();
        run_git(&tmp, &["add", "tracked.txt"]);
        run_git(
            &tmp,
            &[
                "-c",
                "user.email=t@t",
                "-c",
                "user.name=t",
                "commit",
                "-q",
                "-m",
                "v1",
            ],
        );

        let indexed_fp = WorkspaceFingerprint::capture(&tmp);
        store
            .upsert_workspace_state(&WorkspaceState {
                root_path: indexed_fp.canonical_root.to_string_lossy().into_owned(),
                git_dir: indexed_fp
                    .git_dir
                    .as_ref()
                    .map(|p| p.to_string_lossy().into_owned()),
                git_common_dir: indexed_fp
                    .git_common_dir
                    .as_ref()
                    .map(|p| p.to_string_lossy().into_owned()),
                head_oid: indexed_fp.head_oid.clone(),
                index_signature: indexed_fp.index_signature.clone(),
                schema_version: 1,
                indexer_version: "grepplus-indexer-v1".into(),
                graph_generation: 1,
                updated_at: "2026-06-28T20:00:00Z".into(),
            })
            .unwrap();

        std::fs::write(tmp.join("tracked.txt"), "v2-staged-only\n").unwrap();
        run_git(&tmp, &["add", "tracked.txt"]);

        let staged_fp = WorkspaceFingerprint::capture(&tmp);
        assert_eq!(
            indexed_fp.head_oid, staged_fp.head_oid,
            "this test covers index-only drift, not commit drift"
        );
        assert_ne!(
            indexed_fp.index_signature, staged_fp.index_signature,
            "staging the edit must change the git index signature"
        );
        let r = check(&store, &staged_fp, Duration::from_secs(1)).unwrap();
        match r.outcome {
            FreshnessOutcome::Stale { reasons } => {
                assert!(
                    reasons.iter().any(|s| s.contains("index signature")),
                    "expected index signature stale reason, got {reasons:?}"
                );
            }
            other => panic!("expected Stale after staged-only edit, got {other:?}"),
        }
    }

    #[test]
    fn check_files_detects_unstaged_modification() {
        // Set up: git repo with one committed file, then a fresh
        // workspace_state, then modify the file in place. The
        // workspace-level check should still say Fresh (git state
        // unchanged) but the per-file check should say Stale.
        let tmp = tempdir_via_env();
        run_git(&tmp, &["init", "-q"]);
        std::fs::create_dir_all(tmp.join("src")).unwrap();
        std::fs::write(tmp.join("src/lib.rs"), "pub fn hello() {}\n").unwrap();
        run_git(&tmp, &["add", "src/lib.rs"]);
        run_git(
            &tmp,
            &[
                "-c",
                "user.email=t@t",
                "-c",
                "user.name=t",
                "commit",
                "-q",
                "-m",
                "init",
            ],
        );

        // Seed the file_state + workspace_state rows directly so we
        // do not depend on the indexer pipeline (which has a
        // circular-dep-free reimplementation in this crate but is
        // slower and needs a real on-disk store).
        use grepplus_store::file_state::{sha256_hex, FileState};
        use grepplus_store::{workspace_state as ws, Project, WorkspaceState};
        let mut store = Store::open_memory().unwrap();
        store
            .upsert_project(&Project {
                name: "p".into(),
                indexed_at: "x".into(),
                root_path: tmp.to_string_lossy().into_owned(),
            })
            .unwrap();
        let fp = WorkspaceFingerprint::capture(&tmp);
        store
            .upsert_workspace_state(&WorkspaceState {
                root_path: fp.canonical_root.to_string_lossy().into_owned(),
                git_dir: fp
                    .git_dir
                    .as_ref()
                    .map(|p| p.to_string_lossy().into_owned()),
                git_common_dir: fp
                    .git_common_dir
                    .as_ref()
                    .map(|p| p.to_string_lossy().into_owned()),
                head_oid: fp.head_oid.clone(),
                index_signature: fp.index_signature.clone(),
                schema_version: 1,
                indexer_version: "grepplus-indexer-v1".into(),
                graph_generation: 1,
                updated_at: ws::now_iso8601(),
            })
            .unwrap();
        let bytes = std::fs::read(tmp.join("src/lib.rs")).unwrap();
        store
            .upsert_file_state(&FileState {
                project: "p".into(),
                rel_path: "src/lib.rs".into(),
                language: "rust".into(),
                sha256: sha256_hex(&bytes),
                mtime_ns: 1,
                size: bytes.len() as i64,
                parser_version: "tree-sitter-0.25".into(),
                extractor_version: "grepplus-extractor-v1".into(),
                last_indexed_generation: 1,
            })
            .unwrap();

        // Mutate the file without staging it.
        std::fs::write(
            tmp.join("src/lib.rs"),
            "pub fn hello() { println!(\"x\"); }\n",
        )
        .unwrap();

        // Use a generous budget — these tests run concurrently with
        // many other tests in the same crate, and the first call into
        // `discover::walk` can pay a ~1s cost under contention on
        // shared CI. The production budget (200ms) is enforced in the
        // wrapper; here we just want stability.
        let r = check_files(&store, &tmp, "p", Duration::from_secs(30)).unwrap();
        match r.outcome {
            FreshnessOutcome::Stale { reasons } => {
                assert!(
                    reasons.iter().any(|s| s.contains("modified")),
                    "expected 'modified' in reasons, got {reasons:?}"
                );
            }
            other => panic!("expected Stale after unstaged edit, got {other:?}"),
        }
    }

    #[test]
    fn check_files_fresh_when_unchanged() {
        // Same as above, but no file mutation: check_files should
        // return Fresh.
        let tmp = tempdir_via_env();
        run_git(&tmp, &["init", "-q"]);
        std::fs::create_dir_all(tmp.join("src")).unwrap();
        std::fs::write(tmp.join("src/lib.rs"), "pub fn hello() {}\n").unwrap();
        run_git(&tmp, &["add", "src/lib.rs"]);
        run_git(
            &tmp,
            &[
                "-c",
                "user.email=t@t",
                "-c",
                "user.name=t",
                "commit",
                "-q",
                "-m",
                "init",
            ],
        );

        use grepplus_store::file_state::{sha256_hex, FileState};
        use grepplus_store::{workspace_state as ws, Project, WorkspaceState};
        let mut store = Store::open_memory().unwrap();
        store
            .upsert_project(&Project {
                name: "p".into(),
                indexed_at: "x".into(),
                root_path: tmp.to_string_lossy().into_owned(),
            })
            .unwrap();
        let fp = WorkspaceFingerprint::capture(&tmp);
        store
            .upsert_workspace_state(&WorkspaceState {
                root_path: fp.canonical_root.to_string_lossy().into_owned(),
                git_dir: fp
                    .git_dir
                    .as_ref()
                    .map(|p| p.to_string_lossy().into_owned()),
                git_common_dir: fp
                    .git_common_dir
                    .as_ref()
                    .map(|p| p.to_string_lossy().into_owned()),
                head_oid: fp.head_oid.clone(),
                index_signature: fp.index_signature.clone(),
                schema_version: 1,
                indexer_version: "grepplus-indexer-v1".into(),
                graph_generation: 1,
                updated_at: ws::now_iso8601(),
            })
            .unwrap();
        let bytes = std::fs::read(tmp.join("src/lib.rs")).unwrap();
        store
            .upsert_file_state(&FileState {
                project: "p".into(),
                rel_path: "src/lib.rs".into(),
                language: "rust".into(),
                sha256: sha256_hex(&bytes),
                mtime_ns: 1,
                size: bytes.len() as i64,
                parser_version: "tree-sitter-0.25".into(),
                extractor_version: "grepplus-extractor-v1".into(),
                last_indexed_generation: 1,
            })
            .unwrap();

        let r = check_files(&store, &tmp, "p", Duration::from_secs(30)).unwrap();
        assert_eq!(r.outcome, FreshnessOutcome::Fresh);
    }

    #[test]
    fn check_files_with_overrides_uses_scoped_inventory() {
        let tmp = tempdir_via_env();
        std::fs::create_dir_all(tmp.join("src")).unwrap();
        std::fs::create_dir_all(tmp.join("tests")).unwrap();
        std::fs::write(tmp.join("src/lib.rs"), "pub fn hello() {}\n").unwrap();
        std::fs::write(tmp.join("tests/integration.rs"), "pub fn outside() {}\n").unwrap();

        use grepplus_store::file_state::{sha256_hex, FileState};
        use grepplus_store::{workspace_state as ws, Project, WorkspaceState};
        let overrides = WalkOverrides::empty().include("src/*.rs");
        let mut store = Store::open_memory().unwrap();
        store
            .upsert_project(&Project {
                name: "p".into(),
                indexed_at: "x".into(),
                root_path: tmp.to_string_lossy().into_owned(),
            })
            .unwrap();
        let fp = WorkspaceFingerprint::capture(&tmp);
        store
            .upsert_workspace_state(&WorkspaceState {
                root_path: fp.canonical_root.to_string_lossy().into_owned(),
                git_dir: fp
                    .git_dir
                    .as_ref()
                    .map(|p| p.to_string_lossy().into_owned()),
                git_common_dir: fp
                    .git_common_dir
                    .as_ref()
                    .map(|p| p.to_string_lossy().into_owned()),
                head_oid: fp.head_oid.clone(),
                index_signature: fp.index_signature.clone(),
                schema_version: 1,
                indexer_version: expected_indexer_version(&overrides),
                graph_generation: 1,
                updated_at: ws::now_iso8601(),
            })
            .unwrap();
        let bytes = std::fs::read(tmp.join("src/lib.rs")).unwrap();
        store
            .upsert_file_state(&FileState {
                project: "p".into(),
                rel_path: "src/lib.rs".into(),
                language: "rust".into(),
                sha256: sha256_hex(&bytes),
                mtime_ns: 1,
                size: bytes.len() as i64,
                parser_version: "tree-sitter-0.25".into(),
                extractor_version: "grepplus-extractor-v1".into(),
                last_indexed_generation: 1,
            })
            .unwrap();

        let default = check_files(&store, &tmp, "p", Duration::from_secs(30)).unwrap();
        match default.outcome {
            FreshnessOutcome::Stale { reasons } => assert!(
                reasons.iter().any(|s| s.contains("indexer version/scope")),
                "default freshness must reject scoped index, got {reasons:?}"
            ),
            other => panic!("default freshness should reject scoped index, got {other:?}"),
        }

        let scoped =
            check_files_with_overrides(&store, &tmp, "p", Duration::from_secs(30), &overrides)
                .unwrap();
        assert_eq!(scoped.outcome, FreshnessOutcome::Fresh);
    }

    fn tempdir_via_env() -> PathBuf {
        let base = std::env::temp_dir();
        let unique = format!(
            "grepplus-freshness-test-{}-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            std::thread::current().id(),
        );
        let p = base.join(unique);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// Seed an on-disk store + workspace/file state for a one-file repo
    /// under `<tmp>/repo`, with the DB outside the walked root. Returns
    /// (store, repo_root).
    fn seed_on_disk_fixture(tmp: &Path) -> (Store, PathBuf) {
        use grepplus_store::file_state::{sha256_hex, FileState};
        use grepplus_store::{workspace_state as ws, Project, WorkspaceState};
        let repo = tmp.join("repo");
        std::fs::create_dir_all(repo.join("src")).unwrap();
        std::fs::write(repo.join("src/lib.rs"), "pub fn hello() {}\n").unwrap();
        let db_dir = tmp.join("db");
        std::fs::create_dir_all(&db_dir).unwrap();
        let mut store = Store::open(&db_dir.join("graph.db")).unwrap();
        store
            .upsert_project(&Project {
                name: "p".into(),
                indexed_at: "x".into(),
                root_path: repo.to_string_lossy().into_owned(),
            })
            .unwrap();
        let fp = WorkspaceFingerprint::capture(&repo);
        store
            .upsert_workspace_state(&WorkspaceState {
                root_path: fp.canonical_root.to_string_lossy().into_owned(),
                git_dir: None,
                git_common_dir: None,
                head_oid: fp.head_oid.clone(),
                index_signature: fp.index_signature.clone(),
                schema_version: 1,
                indexer_version: "grepplus-indexer-v1".into(),
                graph_generation: 1,
                updated_at: ws::now_iso8601(),
            })
            .unwrap();
        let bytes = std::fs::read(repo.join("src/lib.rs")).unwrap();
        store
            .upsert_file_state(&FileState {
                project: "p".into(),
                rel_path: "src/lib.rs".into(),
                language: "rust".into(),
                sha256: sha256_hex(&bytes),
                mtime_ns: 1,
                size: bytes.len() as i64,
                parser_version: "tree-sitter-0.25".into(),
                extractor_version: "grepplus-extractor-v1".into(),
                last_indexed_generation: 1,
            })
            .unwrap();
        (store, repo)
    }

    #[test]
    fn ttl_stamp_skips_walk_within_window_and_expires_with_zero_ttl() {
        let tmp = tempdir_via_env();
        let (store, repo) = seed_on_disk_fixture(&tmp);
        let budget = Duration::from_secs(30);
        let overrides = WalkOverrides::empty();

        // First check proves fresh the slow way and writes the stamp.
        let r1 = check_files_report_with_ttl(
            &store,
            &repo,
            "p",
            budget,
            &overrides,
            Duration::from_secs(60),
        )
        .unwrap();
        assert_eq!(r1.state.outcome, FreshnessOutcome::Fresh);
        assert!(!r1.ttl_hit, "first check must do the real walk");
        assert_eq!(r1.total_inventory, Some(1));

        // Mutate the file. A check WITHIN the TTL still reports Fresh
        // from the stamp — that is the documented D2 trade (bursts pay
        // the walk once; an edit is unlabeled for at most the TTL).
        std::fs::write(repo.join("src/lib.rs"), "pub fn hello() { changed(); }\n").unwrap();
        let r2 = check_files_report_with_ttl(
            &store,
            &repo,
            "p",
            budget,
            &overrides,
            Duration::from_secs(60),
        )
        .unwrap();
        assert_eq!(r2.state.outcome, FreshnessOutcome::Fresh);
        assert!(r2.ttl_hit, "second check within the TTL must hit the stamp");
        assert_eq!(
            r2.total_inventory,
            Some(1),
            "TTL hit reports the stamped inventory size"
        );

        // TTL disabled: the real walk runs and sees the edit.
        let r3 =
            check_files_report_with_ttl(&store, &repo, "p", budget, &overrides, Duration::ZERO)
                .unwrap();
        match &r3.state.outcome {
            FreshnessOutcome::Stale { reasons } => assert!(
                reasons.iter().any(|s| s.contains("modified")),
                "expected modified reason, got {reasons:?}"
            ),
            other => panic!("expected Stale with TTL disabled, got {other:?}"),
        }
        assert!(!r3.ttl_hit);
        assert_eq!(r3.changed_paths, Some(vec!["src/lib.rs".to_string()]));
        assert_eq!(r3.total_inventory, Some(1));
    }

    #[test]
    fn ttl_stamp_scope_mismatch_is_ignored() {
        let tmp = tempdir_via_env();
        let (store, repo) = seed_on_disk_fixture(&tmp);
        let budget = Duration::from_secs(30);

        // Stamp is written for the default scope...
        let r1 = check_files_report_with_ttl(
            &store,
            &repo,
            "p",
            budget,
            &WalkOverrides::empty(),
            Duration::from_secs(60),
        )
        .unwrap();
        assert_eq!(r1.state.outcome, FreshnessOutcome::Fresh);

        // ...a scoped check must NOT reuse it (and, being a different
        // indexer scope, reports stale rather than fresh).
        let scoped = WalkOverrides::empty().include("src/*.rs");
        let r2 = check_files_report_with_ttl(
            &store,
            &repo,
            "p",
            budget,
            &scoped,
            Duration::from_secs(60),
        )
        .unwrap();
        assert!(
            !r2.ttl_hit,
            "a stamp for another discover scope must never satisfy the TTL"
        );
    }

    #[test]
    fn report_enumerates_changed_paths_for_stale_index() {
        let tmp = tempdir_via_env();
        let (store, repo) = seed_on_disk_fixture(&tmp);

        // One modified + one added file.
        std::fs::write(repo.join("src/lib.rs"), "pub fn hello() { edited(); }\n").unwrap();
        std::fs::write(repo.join("src/new.rs"), "pub fn brand_new() {}\n").unwrap();

        let r = check_files_report_with_ttl(
            &store,
            &repo,
            "p",
            Duration::from_secs(30),
            &WalkOverrides::empty(),
            Duration::ZERO,
        )
        .unwrap();
        match &r.state.outcome {
            FreshnessOutcome::Stale { reasons } => {
                assert!(reasons.iter().any(|s| s.contains("added")), "{reasons:?}");
                assert!(
                    reasons.iter().any(|s| s.contains("modified")),
                    "{reasons:?}"
                );
            }
            other => panic!("expected Stale, got {other:?}"),
        }
        let mut changed = r.changed_paths.expect("changed paths enumerated");
        changed.sort();
        assert_eq!(
            changed,
            vec!["src/lib.rs".to_string(), "src/new.rs".to_string()]
        );
        assert_eq!(r.total_inventory, Some(2));
    }

    #[test]
    fn in_memory_store_never_uses_ttl_stamp() {
        // In-memory stores have no backing path, so the TTL tier is a
        // no-op: a change is detected even with a large TTL.
        let tmp = tempdir_via_env();
        run_git(&tmp, &["init", "-q"]);
        std::fs::create_dir_all(tmp.join("src")).unwrap();
        std::fs::write(tmp.join("src/lib.rs"), "pub fn hello() {}\n").unwrap();
        run_git(&tmp, &["add", "src/lib.rs"]);
        run_git(
            &tmp,
            &[
                "-c",
                "user.email=t@t",
                "-c",
                "user.name=t",
                "commit",
                "-q",
                "-m",
                "init",
            ],
        );

        use grepplus_store::file_state::{sha256_hex, FileState};
        use grepplus_store::{workspace_state as ws, Project, WorkspaceState};
        let mut store = Store::open_memory().unwrap();
        store
            .upsert_project(&Project {
                name: "p".into(),
                indexed_at: "x".into(),
                root_path: tmp.to_string_lossy().into_owned(),
            })
            .unwrap();
        let fp = WorkspaceFingerprint::capture(&tmp);
        store
            .upsert_workspace_state(&WorkspaceState {
                root_path: fp.canonical_root.to_string_lossy().into_owned(),
                git_dir: fp
                    .git_dir
                    .as_ref()
                    .map(|p| p.to_string_lossy().into_owned()),
                git_common_dir: fp
                    .git_common_dir
                    .as_ref()
                    .map(|p| p.to_string_lossy().into_owned()),
                head_oid: fp.head_oid.clone(),
                index_signature: fp.index_signature.clone(),
                schema_version: 1,
                indexer_version: "grepplus-indexer-v1".into(),
                graph_generation: 1,
                updated_at: ws::now_iso8601(),
            })
            .unwrap();
        let bytes = std::fs::read(tmp.join("src/lib.rs")).unwrap();
        store
            .upsert_file_state(&FileState {
                project: "p".into(),
                rel_path: "src/lib.rs".into(),
                language: "rust".into(),
                sha256: sha256_hex(&bytes),
                mtime_ns: 1,
                size: bytes.len() as i64,
                parser_version: "tree-sitter-0.25".into(),
                extractor_version: "grepplus-extractor-v1".into(),
                last_indexed_generation: 1,
            })
            .unwrap();

        let ttl = Duration::from_secs(60);
        let budget = Duration::from_secs(30);
        let r1 =
            check_files_report_with_ttl(&store, &tmp, "p", budget, &WalkOverrides::empty(), ttl)
                .unwrap();
        assert_eq!(r1.state.outcome, FreshnessOutcome::Fresh);
        std::fs::write(tmp.join("src/lib.rs"), "pub fn hello() { edited(); }\n").unwrap();
        let r2 =
            check_files_report_with_ttl(&store, &tmp, "p", budget, &WalkOverrides::empty(), ttl)
                .unwrap();
        assert!(
            matches!(r2.state.outcome, FreshnessOutcome::Stale { .. }),
            "no stamp for an in-memory store: change must be detected, got {:?}",
            r2.state.outcome
        );
    }

    fn run_git(cwd: &Path, args: &[&str]) {
        let status = Command::new("git")
            .current_dir(cwd)
            .args(args)
            .status()
            .expect("git invocation");
        assert!(status.success(), "git {args:?} failed: {status:?}");
    }
}
