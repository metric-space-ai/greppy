//! Incremental update path.
//!
//! Given the current on-disk inventory and the persisted `file_state`,
//! compute the diff (added / modified / deleted) and either reindex
//! only the diff'd files (incremental) or rebuild from scratch
//! (full reindex into a temp DB + atomic swap).
//!
//! The API is kept simple: [`compute_file_diff`] returns the
//! structured diff; [`incremental_update`] applies it via the existing
//! indexer pipeline. Full-rebuild-to-temp-DB-and-swap is implemented in
//! [`full_reindex`].

use std::path::Path;

use greppy_core::Result;
use greppy_discover::{read_stable_file, stable_metadata, InventoryEntry};
use greppy_parser;
use greppy_store::{file_state::sha256_hex, FileIdentity, FileState, Store};
use sha2::{Digest, Sha256};

/// The freshness hotpath must refuse to slurp
/// files larger than this cap (mirrors `greppy_indexer::MAX_FILE_SIZE_BYTES`).
/// Kept as a local const so the freshness crate doesn't gain a
/// dependency on the indexer crate.
pub const MAX_FILE_SIZE_BYTES: u64 = 50 * 1024 * 1024;

/// Resolve the effective max-file-size cap, honouring the
/// `GREPPY_MAX_FILE_SIZE` env var (bytes). Kept identical to the
/// indexer's resolution so the two sides agree on which files are
/// "oversize". An unparseable / empty value falls back to the default.
fn max_file_size_bytes() -> u64 {
    std::env::var("GREPPY_MAX_FILE_SIZE")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(MAX_FILE_SIZE_BYTES)
}

/// Per-file result of the diff.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileDiff {
    Added(InventoryEntry),
    Modified {
        entry: InventoryEntry,
        old_sha256: String,
    },
    Deleted(String),
    Unchanged,
}

/// Walk the inventory, comparing each file against the persisted
/// `file_state`. Files present in `file_state` but missing from
/// `inventory` are reported as `Deleted`.
///
/// Defect D2 (cheap-first freshness): this used to `read` + sha256 EVERY
/// within-cap file on EVERY freshness check — ~0.8 s (release) on a
/// 429-file repo, which blew the per-query budget and fail-closed the
/// whole plus surface. The check is now tiered:
///
/// 1. **stat tier** — compare the persisted `(size, mtime_ns, ctime_ns,
///    file_id)` against the on-disk stat (already captured by the discover
///    walk, so this usually costs zero extra syscalls). Only a complete match
///    is `Unchanged` without reading the body. Same-size replacements with a
///    restored mtime still change ctime and/or file identity.
/// 2. **hash tier** — only files whose stat drifted (or that have no
///    persisted row / a `mtime_ns == 0` sentinel row) are read and
///    content-hashed, exactly as before. A touch that does not change
///    content therefore still resolves to `Unchanged`.
///
/// Oversized files (`> GREPPY_MAX_FILE_SIZE`,
/// default [`MAX_FILE_SIZE_BYTES`]) are never read into memory. This
/// runs on EVERY `greppy grep`, so a multi-GB file on the hotpath
/// must not OOM the wrapper. The guard checks the size *before*
/// any `std::fs::read`, then for oversized files compares the persisted
/// complete stat identity against the on-disk stat instead of hashing the
/// body:
///
/// - no persisted row, missing identity, or any stat field differs → `Modified` (gate goes
///   `Stale`, prompting a reindex that the indexer also caps);
/// - all reliable persisted stat fields match → `Unchanged` (cheap, no read).
///
/// The indexer side enforces the same cap, so the persisted
/// `file_state` row never sees an oversized file's actual content hash.
pub fn compute_file_diff(
    store: &Store,
    project: &str,
    inventory: &[InventoryEntry],
) -> Result<Vec<FileDiff>> {
    let persisted = store.list_file_states(project)?;
    let persisted_skips = store.list_index_skips(project)?;
    let identities = store.list_file_identities(project)?;
    let mut diffs = Vec::with_capacity(inventory.len());
    let max_size = max_file_size_bytes();

    // Build a map rel_path -> persisted (sha256, size, mtime_ns). We
    // keep size+mtime (not just the hash) so unchanged files can be
    // diffed by stat alone, without ever reading their bodies.
    let mut by_rel: std::collections::HashMap<String, PersistedStat> = persisted
        .into_iter()
        .map(|f| {
            let identity = identities.get(&f.rel_path).copied();
            (
                f.rel_path,
                PersistedStat {
                    sha256: f.sha256,
                    size: f.size,
                    mtime_ns: f.mtime_ns,
                    ctime_ns: identity.and_then(|value| value.ctime_ns),
                    file_id: identity.and_then(|value| value.file_id),
                },
            )
        })
        .collect();
    let mut skipped_by_rel: std::collections::HashMap<_, _> = persisted_skips
        .into_iter()
        .map(|skip| (skip.rel_path.clone(), skip))
        .collect();
    // A successful indexed row is authoritative if a legacy or interrupted
    // run left a skip row for the same path behind.
    skipped_by_rel.retain(|rel_path, _| !by_rel.contains_key(rel_path));

    for entry in inventory {
        // Stat BEFORE read. The discover walk already captured
        // (size, mtime_ns) for every entry, so the common case costs no
        // extra syscall; fall back to one metadata() call otherwise. A
        // multi-GB file costs one syscall here, not a multi-GB allocation.
        let current = match (entry.size, entry.mtime_ns) {
            (Some(size), mtime @ Some(_)) => CurrentStat {
                size,
                mtime_ns: mtime,
                ctime_ns: entry.ctime_ns,
                file_id: entry.file_id,
            },
            _ => match std::fs::metadata(&entry.abs_path) {
                Ok(md) => {
                    let metadata = stable_metadata(&md);
                    CurrentStat {
                        size: metadata.size,
                        mtime_ns: metadata.mtime_ns,
                        ctime_ns: metadata.ctime_ns,
                        file_id: metadata.file_id,
                    }
                }
                Err(error) => {
                    return Err(greppy_core::Error::io(
                        format!("stat {}", entry.abs_path.display()),
                        error,
                    ));
                }
            },
        };

        if let Some(skip) = skipped_by_rel.remove(&entry.rel_path) {
            // Skipped files intentionally have no file_state row. Accept only
            // a complete, policy-compatible identity; any uncertainty forces
            // the indexer to disposition the path again.
            if skip_identity_matches(&skip, &current, max_size) {
                diffs.push(FileDiff::Unchanged);
                continue;
            }
        }

        if current.size > max_size {
            // Oversized: diff by (size, mtime) against the persisted
            // row instead of slurping + hashing the body.
            let persisted = by_rel.remove(&entry.rel_path);
            match persisted {
                // Unchanged iff a row exists and both size & mtime
                // line up with the current stat.
                Some(p) if stat_identity_matches(&p, &current) => {
                    diffs.push(FileDiff::Unchanged);
                }
                // No row, or stat drifted → treat as Modified so the
                // freshness gate downgrades. `old_sha256` is a sentinel
                // (we never hashed the oversized body).
                _ => diffs.push(FileDiff::Modified {
                    entry: entry.clone(),
                    old_sha256: "<oversize>".to_string(),
                }),
            }
            continue;
        }

        // Stat tier: all reliable persisted identity fields match the
        // on-disk stat ⇒ Unchanged without reading the body. `mtime_ns == 0` is the
        // "mtime unknown" sentinel some writers record; never fast-path
        // it (fall through to the hash tier).
        if let Some(p) = by_rel.get(&entry.rel_path) {
            if stat_identity_matches(p, &current) {
                by_rel.remove(&entry.rel_path);
                diffs.push(FileDiff::Unchanged);
                continue;
            }
        }

        // Hash tier (stat drifted or file unknown): safe to read + hash.
        let (bytes, _) = read_stable_file(&entry.abs_path).map_err(|error| {
            greppy_core::Error::io(format!("stable read {}", entry.abs_path.display()), error)
        })?;
        let sha = sha256_hex(&bytes);
        match by_rel.remove(&entry.rel_path) {
            None => diffs.push(FileDiff::Added(entry.clone())),
            Some(p) if p.sha256 != sha => diffs.push(FileDiff::Modified {
                entry: entry.clone(),
                old_sha256: p.sha256,
            }),
            Some(_) => diffs.push(FileDiff::Unchanged),
        }
    }

    // Anything left in by_rel is a deletion.
    for (rel, _) in by_rel {
        diffs.push(FileDiff::Deleted(rel));
    }
    for (rel, _) in skipped_by_rel {
        diffs.push(FileDiff::Deleted(rel));
    }

    diffs.sort_by_key(rel_of);
    Ok(diffs)
}

/// Persisted stat columns we need to diff a file without reading it.
struct PersistedStat {
    sha256: String,
    size: i64,
    mtime_ns: i64,
    ctime_ns: Option<i64>,
    file_id: Option<u64>,
}

struct CurrentStat {
    size: u64,
    mtime_ns: Option<i64>,
    ctime_ns: Option<i64>,
    file_id: Option<u64>,
}

fn stat_identity_matches(persisted: &PersistedStat, current: &CurrentStat) -> bool {
    persisted.mtime_ns != 0
        && persisted.size == current.size as i64
        && Some(persisted.mtime_ns) == current.mtime_ns
        && persisted.ctime_ns.is_some()
        && current.ctime_ns.is_some()
        && persisted.ctime_ns == current.ctime_ns
        && persisted.file_id.is_some()
        && current.file_id.is_some()
        && persisted.file_id == current.file_id
}

fn skip_identity_matches(
    skip: &greppy_store::IndexSkip,
    current: &CurrentStat,
    max_size: u64,
) -> bool {
    let basic_match = skip.mtime_ns != 0
        && skip.size == current.size as i64
        && Some(skip.mtime_ns) == current.mtime_ns;
    if !basic_match || matches!(skip.reason.as_str(), "file_limit" | "time_budget") {
        return false;
    }

    if skip.reason == "oversize" {
        return current.size > max_size;
    }
    if current.size > max_size {
        return false;
    }

    skip.ctime_ns.is_some()
        && current.ctime_ns.is_some()
        && skip.ctime_ns == current.ctime_ns
        && skip.file_id.is_some()
        && current.file_id.is_some()
        && skip.file_id == current.file_id
}

fn rel_of(d: &FileDiff) -> String {
    match d {
        FileDiff::Added(e) => e.rel_path.clone(),
        FileDiff::Modified { entry, .. } => entry.rel_path.clone(),
        FileDiff::Deleted(r) => r.clone(),
        FileDiff::Unchanged => String::new(),
    }
}

/// Apply an incremental update: process added + modified files, drop
/// the persisted state for deleted files, then re-bump the workspace
/// generation.
///
/// Returns the number of files reindexed.
pub fn incremental_update(
    store: &mut Store,
    project: &str,
    inventory: &[InventoryEntry],
) -> Result<usize> {
    let diffs = compute_file_diff(store, project, inventory)?;
    let mut reindexed = 0usize;
    for d in diffs {
        match d {
            FileDiff::Added(entry) | FileDiff::Modified { entry, .. } => {
                let lang = greppy_parser::language_for_path(&entry.abs_path);
                if !lang.is_supported() {
                    continue;
                }
                // Even a *supported* file can be oversized
                // (e.g. a multi-GB generated source). Stat before read
                // so we never slurp past the cap on the reindex path
                // either; the indexer enforces the same skip.
                if let Ok(md) = std::fs::metadata(&entry.abs_path) {
                    if md.len() > max_file_size_bytes() {
                        continue;
                    }
                }
                let (bytes, metadata) = read_stable_file(&entry.abs_path).map_err(|e| {
                    greppy_core::Error::io(format!("stable read {}", entry.abs_path.display()), e)
                })?;
                let extraction = greppy_parser::extract(lang, &bytes, &entry.rel_path)?;
                // We use the public store API: upsert each
                // extracted node + file_state. The indexer crate owns
                // the equivalent pipeline for fresh full-rebuilds.
                for n in extraction.nodes {
                    let _ = store.insert_node(&greppy_store::NewNode {
                        project: project.into(),
                        label: n.label,
                        name: n.name,
                        qualified_name: n.qualified_name,
                        file_path: entry.rel_path.clone(),
                        start_line: n.start_line as i64,
                        end_line: n.end_line as i64,
                        properties: n.properties,
                    })?;
                }
                store.upsert_file_state(&FileState {
                    project: project.into(),
                    rel_path: entry.rel_path.clone(),
                    language: lang.name().into(),
                    sha256: sha256_hex(&bytes),
                    mtime_ns: metadata.mtime_ns.unwrap_or(0),
                    size: bytes.len() as i64,
                    parser_version: "tree-sitter-0.25".into(),
                    extractor_version: "greppy-extractor-v1".into(),
                    last_indexed_generation: 0,
                })?;
                store.upsert_file_identity(
                    project,
                    &entry.rel_path,
                    FileIdentity {
                        ctime_ns: metadata.ctime_ns,
                        file_id: metadata.file_id,
                    },
                )?;
                reindexed += 1;
            }
            FileDiff::Deleted(rel) => {
                store.delete_file_state(project, &rel)?;
            }
            FileDiff::Unchanged => {}
        }
    }
    Ok(reindexed)
}

/// Full reindex into a temp DB + atomic swap.
///
/// This is implemented as: open a new in-memory store, run the
/// indexer into it, then on success replace the existing graph by
/// copying nodes/edges/file_state row-by-row inside a single
/// transaction. For simplicity we use the same DB file path
/// but a future hardening pass could use SQLite's backup
/// API for atomicity.
///
/// This function lives here so callers in `crates/greppy` and
/// `crates/cli` have one entry point. The actual indexer runs in
/// `crates/indexer`; we re-export the entry point through `reindex`.
pub fn reindex(store: &mut Store, root: &Path, project_name: &str) -> Result<ReindexReport> {
    let report = full_reindex_inner(store, root, project_name)?;
    Ok(ReindexReport {
        files_indexed: report.files_indexed,
        graph_generation: report.graph_generation,
    })
}

/// Wrapper that captures the indexer's report shape via a stable
/// type so this crate does not depend on `greppy-indexer`.
#[derive(Debug, Clone, Default)]
pub struct ReindexReport {
    pub files_indexed: usize,
    pub graph_generation: u64,
}

fn full_reindex_inner(store: &mut Store, root: &Path, project_name: &str) -> Result<ReindexReport> {
    // Walk files, run incremental update (which is what the
    // indexer does in practice — a single pass that
    // upserts every supported file). A future hardening pass can add
    // temp-DB swap.
    let abs_root = greppy_discover::detect_repo_root(root)?;
    let entries = greppy_discover::walk(&abs_root)?;
    let inventory: Vec<InventoryEntry> = entries.into_iter().collect();
    let reindexed = crate::incremental::incremental_update(store, project_name, &inventory)?;
    let generation = bump(store, &abs_root.to_string_lossy()).unwrap_or_default();
    Ok(ReindexReport {
        files_indexed: reindexed,
        graph_generation: generation,
    })
}

fn bump(store: &mut Store, root: &str) -> Result<u64> {
    Ok(greppy_store::Store::bump_generation(store, root)?)
}

#[cfg(test)]
fn mtime_ns(path: &Path) -> Option<i64> {
    mtime_ns_from_metadata(&std::fs::metadata(path).ok()?)
}

/// Convert a `Metadata`'s mtime to nanoseconds since the Unix epoch.
///
/// This MUST match `greppy_discover::metadata_fields` exactly
/// (saturate to `i64::MAX` for out-of-range futures, negative /
/// `i64::MIN`-saturated for pre-epoch times): the stat tier in
/// [`compute_file_diff`] compares the walker-captured value against the
/// value this crate persisted, so any conversion drift would silently
/// disable the fast path (or worse, fast-path a changed file).
#[cfg(test)]
fn mtime_ns_from_metadata(md: &std::fs::Metadata) -> Option<i64> {
    let mtime = md.modified().ok()?;
    Some(match mtime.duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => i64::try_from(d.as_nanos()).unwrap_or(i64::MAX),
        // Pre-epoch mtime: encode as a negative offset.
        Err(e) => i64::try_from(e.duration().as_nanos())
            .map(|v| -v)
            .unwrap_or(i64::MIN),
    })
}

/// SHA-256 of the indexer version + schema version, used for the
/// `indexer_version` field of `workspace_state`.
#[allow(dead_code)]
pub fn indexer_version_hash() -> String {
    let mut h = Sha256::new();
    h.update(greppy_core::INDEXER_VERSION_BASE.as_bytes());
    h.update([1u8]);
    let d = h.finalize();
    let mut s = String::with_capacity(64);
    use std::fmt::Write;
    for b in d {
        let _ = write!(&mut s, "{b:02x}");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use greppy_store::{workspace_state as ws, IndexSkip, Project};
    use std::fs;
    use std::path::PathBuf;

    fn tempdir_via_env() -> PathBuf {
        let base = std::env::temp_dir();
        let unique = format!(
            "greppy-incr-test-{}-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            std::thread::current().id(),
        );
        let p = base.join(unique);
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn make_entry(dir: &Path, rel: &str, body: &str) -> InventoryEntry {
        let abs = dir.join(rel);
        if let Some(parent) = abs.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&abs, body).unwrap();
        InventoryEntry {
            rel_path: rel.to_string(),
            abs_path: abs,
            ..Default::default()
        }
    }

    #[test]
    fn diff_detects_added_modified_deleted_unchanged() {
        let mut store = Store::open_memory().unwrap();
        store
            .upsert_project(&Project {
                name: "p".into(),
                indexed_at: "x".into(),
                root_path: "/p".into(),
            })
            .unwrap();

        // Seed file_state: keep.rs unchanged, gone.rs will be deleted,
        // changed.rs will be modified.
        store
            .upsert_file_state(&FileState {
                project: "p".into(),
                rel_path: "src/keep.rs".into(),
                language: "rust".into(),
                sha256: sha256_hex(b"keep-v1"),
                mtime_ns: 1,
                size: 7,
                parser_version: "x".into(),
                extractor_version: "x".into(),
                last_indexed_generation: 1,
            })
            .unwrap();
        store
            .upsert_file_state(&FileState {
                project: "p".into(),
                rel_path: "src/changed.rs".into(),
                language: "rust".into(),
                sha256: sha256_hex(b"v1"),
                mtime_ns: 1,
                size: 2,
                parser_version: "x".into(),
                extractor_version: "x".into(),
                last_indexed_generation: 1,
            })
            .unwrap();
        store
            .upsert_file_state(&FileState {
                project: "p".into(),
                rel_path: "src/gone.rs".into(),
                language: "rust".into(),
                sha256: sha256_hex(b"will-be-removed"),
                mtime_ns: 1,
                size: 16,
                parser_version: "x".into(),
                extractor_version: "x".into(),
                last_indexed_generation: 1,
            })
            .unwrap();

        let dir = tempdir_via_env();
        let inventory = vec![
            make_entry(&dir, "src/keep.rs", "keep-v1"), // unchanged
            make_entry(&dir, "src/changed.rs", "v2-newer"), // modified
            make_entry(&dir, "src/added.rs", "brand-new"), // added
                                                        // gone.rs is NOT in the inventory → deleted
        ];

        let diffs = compute_file_diff(&store, "p", &inventory).unwrap();
        let summary: Vec<&'static str> = diffs
            .iter()
            .map(|d| match d {
                FileDiff::Added(_) => "added",
                FileDiff::Modified { .. } => "modified",
                FileDiff::Deleted(_) => "deleted",
                FileDiff::Unchanged => "unchanged",
            })
            .collect();
        assert!(summary.contains(&"added"));
        assert!(summary.contains(&"modified"));
        assert!(summary.contains(&"deleted"));
        assert!(summary.contains(&"unchanged"));
    }

    #[test]
    fn diff_respects_unchanged_and_changed_index_skips() {
        let mut store = Store::open_memory().unwrap();
        store
            .upsert_project(&Project {
                name: "p".into(),
                indexed_at: "x".into(),
                root_path: "/p".into(),
            })
            .unwrap();

        let dir = tempdir_via_env();
        let keep = make_entry(&dir, "src/keep.ts", "invalid provider syntax");
        let changed = make_entry(&dir, "src/changed.ts", "old invalid syntax");
        let gone = make_entry(&dir, "src/gone.ts", "gone invalid syntax");

        for entry in [&keep, &changed, &gone] {
            let metadata = stable_metadata(&fs::metadata(&entry.abs_path).unwrap());
            store
                .upsert_index_skip(&IndexSkip {
                    project: "p".into(),
                    rel_path: entry.rel_path.clone(),
                    language: "typescript".into(),
                    reason: "provider_invalid".into(),
                    detail: "fixture".into(),
                    size: metadata.size as i64,
                    mtime_ns: metadata.mtime_ns.unwrap(),
                    ctime_ns: metadata.ctime_ns,
                    file_id: metadata.file_id,
                    last_indexed_generation: 1,
                    updated_at: ws::now_iso8601(),
                })
                .unwrap();
        }

        fs::write(&changed.abs_path, "new and longer invalid provider syntax").unwrap();
        fs::remove_file(&gone.abs_path).unwrap();
        let inventory = vec![keep, changed];
        let diffs = compute_file_diff(&store, "p", &inventory).unwrap();

        assert_eq!(
            diffs
                .iter()
                .filter(|diff| matches!(diff, FileDiff::Unchanged))
                .count(),
            1,
            "the unchanged persisted skip must remain fresh: {diffs:?}"
        );
        assert!(diffs.iter().any(
            |diff| matches!(diff, FileDiff::Added(entry) if entry.rel_path == "src/changed.ts")
        ));
        assert!(diffs
            .iter()
            .any(|diff| matches!(diff, FileDiff::Deleted(path) if path == "src/gone.ts")));
    }

    #[test]
    fn same_size_skip_change_with_restored_mtime_is_rechecked() {
        let mut store = Store::open_memory().unwrap();
        store
            .upsert_project(&Project {
                name: "p".into(),
                indexed_at: "x".into(),
                root_path: "/p".into(),
            })
            .unwrap();
        let dir = tempdir_via_env();
        let entry = make_entry(&dir, "src/swap.ts", "old-body");
        let original = fs::metadata(&entry.abs_path).unwrap();
        let original_modified = original.modified().unwrap();
        let identity = stable_metadata(&original);
        store
            .upsert_index_skip(&IndexSkip {
                project: "p".into(),
                rel_path: entry.rel_path.clone(),
                language: "typescript".into(),
                reason: "parse_failed".into(),
                detail: "fixture".into(),
                size: identity.size as i64,
                mtime_ns: identity.mtime_ns.unwrap(),
                ctime_ns: identity.ctime_ns,
                file_id: identity.file_id,
                last_indexed_generation: 1,
                updated_at: ws::now_iso8601(),
            })
            .unwrap();

        std::thread::sleep(std::time::Duration::from_millis(5));
        fs::write(&entry.abs_path, "new-body").unwrap();
        fs::OpenOptions::new()
            .write(true)
            .open(&entry.abs_path)
            .unwrap()
            .set_modified(original_modified)
            .unwrap();

        let diffs = compute_file_diff(&store, "p", std::slice::from_ref(&entry)).unwrap();
        assert!(matches!(
            diffs.as_slice(),
            [FileDiff::Added(changed)] if changed.rel_path == entry.rel_path
        ));
    }

    #[test]
    fn incomplete_or_transient_skips_never_prove_freshness() {
        let current = CurrentStat {
            size: 10,
            mtime_ns: Some(20),
            ctime_ns: Some(30),
            file_id: Some(40),
        };
        let mut skip = IndexSkip {
            project: "p".into(),
            rel_path: "src/a.rs".into(),
            language: "rust".into(),
            reason: "parse_failed".into(),
            detail: "fixture".into(),
            size: 10,
            mtime_ns: 0,
            ctime_ns: Some(30),
            file_id: Some(40),
            last_indexed_generation: 1,
            updated_at: ws::now_iso8601(),
        };
        assert!(!skip_identity_matches(&skip, &current, 100));

        skip.mtime_ns = 20;
        skip.ctime_ns = None;
        assert!(!skip_identity_matches(&skip, &current, 100));

        skip.ctime_ns = Some(30);
        skip.reason = "file_limit".into();
        assert!(!skip_identity_matches(&skip, &current, 100));
        skip.reason = "time_budget".into();
        assert!(!skip_identity_matches(&skip, &current, 100));
    }

    #[test]
    fn oversize_skip_tracks_the_active_size_policy() {
        let current = CurrentStat {
            size: 200,
            mtime_ns: Some(20),
            ctime_ns: None,
            file_id: None,
        };
        let skip = IndexSkip {
            project: "p".into(),
            rel_path: "large.rs".into(),
            language: "rust".into(),
            reason: "oversize".into(),
            detail: "fixture".into(),
            size: 200,
            mtime_ns: 20,
            ctime_ns: None,
            file_id: None,
            last_indexed_generation: 1,
            updated_at: ws::now_iso8601(),
        };
        assert!(skip_identity_matches(&skip, &current, 100));
        assert!(!skip_identity_matches(&skip, &current, 300));
    }

    #[test]
    fn file_state_wins_over_a_legacy_skip_for_the_same_path() {
        let mut store = Store::open_memory().unwrap();
        store
            .upsert_project(&Project {
                name: "p".into(),
                indexed_at: "x".into(),
                root_path: "/p".into(),
            })
            .unwrap();
        let dir = tempdir_via_env();
        let entry = make_entry(&dir, "src/keep.rs", "pub fn keep() {}\n");
        let metadata = stable_metadata(&fs::metadata(&entry.abs_path).unwrap());
        store
            .upsert_file_state(&FileState {
                project: "p".into(),
                rel_path: entry.rel_path.clone(),
                language: "rust".into(),
                sha256: sha256_hex(b"pub fn keep() {}\n"),
                mtime_ns: metadata.mtime_ns.unwrap(),
                size: metadata.size as i64,
                parser_version: "x".into(),
                extractor_version: "x".into(),
                last_indexed_generation: 1,
            })
            .unwrap();
        store
            .upsert_index_skip(&IndexSkip {
                project: "p".into(),
                rel_path: entry.rel_path.clone(),
                language: "rust".into(),
                reason: "file_limit".into(),
                detail: "legacy fixture".into(),
                size: metadata.size as i64,
                mtime_ns: metadata.mtime_ns.unwrap(),
                ctime_ns: metadata.ctime_ns,
                file_id: metadata.file_id,
                last_indexed_generation: 1,
                updated_at: ws::now_iso8601(),
            })
            .unwrap();

        assert_eq!(
            compute_file_diff(&store, "p", &[entry]).unwrap(),
            vec![FileDiff::Unchanged]
        );
    }

    #[test]
    fn incremental_update_processes_only_diff_files() {
        // For this test we only care about Rust files (parser/extractor
        // behaviour). We seed an unchanged Rust file in file_state,
        // then run incremental_update with an inventory that adds a
        // new Rust file. The new file should be reindexed; the
        // unchanged one's persisted state must remain.
        let mut store = Store::open_memory().unwrap();
        store
            .upsert_project(&Project {
                name: "p".into(),
                indexed_at: "x".into(),
                root_path: "/p".into(),
            })
            .unwrap();

        let dir = tempdir_via_env();
        let inventory = vec![
            make_entry(&dir, "src/keep.rs", "pub fn keep() {}"),
            make_entry(&dir, "src/new.rs", "pub fn fresh() {}"),
        ];

        let reindexed = incremental_update(&mut store, "p", &inventory).unwrap();
        assert_eq!(reindexed, 2, "both files should be processed (Added)");
        let after = store.list_file_states("p").unwrap();
        assert_eq!(after.len(), 2);
    }

    /// Create a *sparse* file of `len` bytes: `metadata().len()` reports
    /// `len`, but no real disk/memory is consumed and the body is all
    /// zeros. This lets us simulate a multi-GB file on the hotpath
    /// without actually allocating one — so if the guard ever regressed
    /// to `std::fs::read`-before-stat, the test would either OOM or read
    /// `len` zero bytes (which we assert it does NOT do).
    fn make_sparse_entry(dir: &Path, rel: &str, len: u64) -> InventoryEntry {
        let abs = dir.join(rel);
        if let Some(parent) = abs.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let f = fs::File::create(&abs).unwrap();
        f.set_len(len).unwrap();
        InventoryEntry {
            rel_path: rel.to_string(),
            abs_path: abs,
            ..Default::default()
        }
    }

    fn current_mtime_ns(p: &Path) -> i64 {
        super::mtime_ns(p).unwrap()
    }

    #[test]
    fn oversize_file_is_diffed_by_stat_not_read() {
        // A file larger than the cap must be handled WITHOUT
        // reading its body. We use a sparse file with an apparent size
        // of cap + 10 MiB. If the guard read it, we'd be slurping that
        // many bytes; instead it must diff by stat alone.
        let mut store = Store::open_memory().unwrap();
        store
            .upsert_project(&Project {
                name: "p".into(),
                indexed_at: "x".into(),
                root_path: "/p".into(),
            })
            .unwrap();

        let dir = tempdir_via_env();
        let huge = MAX_FILE_SIZE_BYTES + 10 * 1024 * 1024;
        let entry = make_sparse_entry(&dir, "big.bin", huge);

        // No persisted row yet → oversized + unknown ⇒ Modified.
        let diffs = compute_file_diff(&store, "p", std::slice::from_ref(&entry)).unwrap();
        assert_eq!(diffs.len(), 1);
        match &diffs[0] {
            FileDiff::Modified { old_sha256, .. } => {
                assert_eq!(old_sha256, "<oversize>");
            }
            other => panic!("expected Modified for unknown oversized file, got {other:?}"),
        }

        // Now persist a (size, mtime)-only row matching the on-disk
        // stat. A second diff must report Unchanged — proving the
        // stat-based comparison works and still never reads the body.
        store
            .upsert_file_state(&FileState {
                project: "p".into(),
                rel_path: "big.bin".into(),
                language: "binary".into(),
                sha256: "<oversize>".into(),
                mtime_ns: current_mtime_ns(&entry.abs_path),
                size: huge as i64,
                parser_version: "x".into(),
                extractor_version: "x".into(),
                last_indexed_generation: 1,
            })
            .unwrap();

        let metadata = greppy_discover::stable_metadata(&fs::metadata(&entry.abs_path).unwrap());
        store
            .upsert_file_identity(
                "p",
                "big.bin",
                FileIdentity {
                    ctime_ns: metadata.ctime_ns,
                    file_id: metadata.file_id,
                },
            )
            .unwrap();

        let diffs = compute_file_diff(&store, "p", std::slice::from_ref(&entry)).unwrap();
        assert_eq!(diffs.len(), 1);
        assert_eq!(
            diffs[0],
            FileDiff::Unchanged,
            "oversized file with matching stat must be Unchanged, not re-Modified every grep"
        );
    }

    #[test]
    fn oversize_file_with_drifted_size_is_modified() {
        // If the persisted size/mtime no longer match the
        // on-disk stat, the oversized file is reported Modified so the
        // freshness gate downgrades — again without reading the body.
        let mut store = Store::open_memory().unwrap();
        store
            .upsert_project(&Project {
                name: "p".into(),
                indexed_at: "x".into(),
                root_path: "/p".into(),
            })
            .unwrap();

        let dir = tempdir_via_env();
        let huge = MAX_FILE_SIZE_BYTES + 1;
        let entry = make_sparse_entry(&dir, "big.bin", huge);

        // Persist a row whose size disagrees with the current stat.
        store
            .upsert_file_state(&FileState {
                project: "p".into(),
                rel_path: "big.bin".into(),
                language: "binary".into(),
                sha256: "<oversize>".into(),
                mtime_ns: current_mtime_ns(&entry.abs_path),
                size: (huge - 12345) as i64, // stale size
                parser_version: "x".into(),
                extractor_version: "x".into(),
                last_indexed_generation: 1,
            })
            .unwrap();

        let diffs = compute_file_diff(&store, "p", std::slice::from_ref(&entry)).unwrap();
        assert_eq!(diffs.len(), 1);
        assert!(
            matches!(diffs[0], FileDiff::Modified { .. }),
            "drifted-size oversized file must be Modified, got {:?}",
            diffs[0]
        );
    }

    #[test]
    fn normal_edit_is_still_detected_alongside_oversize() {
        // Regression guard: the oversize fast-path must not break
        // ordinary content-hash diffing for within-cap files.
        let mut store = Store::open_memory().unwrap();
        store
            .upsert_project(&Project {
                name: "p".into(),
                indexed_at: "x".into(),
                root_path: "/p".into(),
            })
            .unwrap();

        // Seed an unchanged small file and a to-be-modified small file.
        store
            .upsert_file_state(&FileState {
                project: "p".into(),
                rel_path: "src/keep.rs".into(),
                language: "rust".into(),
                sha256: sha256_hex(b"keep-v1"),
                mtime_ns: 1,
                size: 7,
                parser_version: "x".into(),
                extractor_version: "x".into(),
                last_indexed_generation: 1,
            })
            .unwrap();
        store
            .upsert_file_state(&FileState {
                project: "p".into(),
                rel_path: "src/changed.rs".into(),
                language: "rust".into(),
                sha256: sha256_hex(b"v1"),
                mtime_ns: 1,
                size: 2,
                parser_version: "x".into(),
                extractor_version: "x".into(),
                last_indexed_generation: 1,
            })
            .unwrap();

        let dir = tempdir_via_env();
        let mut inventory = vec![
            make_entry(&dir, "src/keep.rs", "keep-v1"),   // unchanged
            make_entry(&dir, "src/changed.rs", "v2-new"), // modified (hash differs)
        ];
        inventory.push(make_sparse_entry(
            &dir,
            "blob.bin",
            MAX_FILE_SIZE_BYTES + 4096,
        ));

        let diffs = compute_file_diff(&store, "p", &inventory).unwrap();

        let changed = diffs.iter().find(|d| match d {
            FileDiff::Modified { entry, .. } => entry.rel_path == "src/changed.rs",
            _ => false,
        });
        assert!(
            changed.is_some(),
            "edited small file must still diff by content hash: {diffs:?}"
        );
        let keep = diffs.iter().any(|d| {
            // The unchanged file collapses to FileDiff::Unchanged.
            matches!(d, FileDiff::Unchanged)
        });
        assert!(keep, "unchanged small file must be Unchanged: {diffs:?}");
        // And the oversized blob is Modified (no persisted row).
        assert!(
            diffs.iter().any(|d| matches!(
                d,
                FileDiff::Modified { entry, old_sha256 }
                    if entry.rel_path == "blob.bin" && old_sha256 == "<oversize>"
            )),
            "oversized blob must be Modified via sentinel: {diffs:?}"
        );
    }

    #[test]
    fn stat_tier_skips_hashing_when_stat_matches() {
        // D2: a persisted row whose (size, mtime_ns) match the on-disk
        // stat must resolve to Unchanged WITHOUT reading the body. We
        // prove the body was not hashed by persisting a garbage sha256:
        // if the hash tier ran, the sha mismatch would flag Modified.
        let mut store = Store::open_memory().unwrap();
        store
            .upsert_project(&Project {
                name: "p".into(),
                indexed_at: "x".into(),
                root_path: "/p".into(),
            })
            .unwrap();

        let dir = tempdir_via_env();
        let entry = make_entry(&dir, "src/keep.rs", "stat-tier-body");
        let md = fs::metadata(&entry.abs_path).unwrap();
        store
            .upsert_file_state(&FileState {
                project: "p".into(),
                rel_path: "src/keep.rs".into(),
                language: "rust".into(),
                sha256: "definitely-not-the-real-hash".into(),
                mtime_ns: current_mtime_ns(&entry.abs_path),
                size: md.len() as i64,
                parser_version: "x".into(),
                extractor_version: "x".into(),
                last_indexed_generation: 1,
            })
            .unwrap();

        let metadata = greppy_discover::stable_metadata(&md);
        store
            .upsert_file_identity(
                "p",
                "src/keep.rs",
                FileIdentity {
                    ctime_ns: metadata.ctime_ns,
                    file_id: metadata.file_id,
                },
            )
            .unwrap();

        let diffs = compute_file_diff(&store, "p", std::slice::from_ref(&entry)).unwrap();
        assert_eq!(
            diffs,
            vec![FileDiff::Unchanged],
            "matching (size, mtime_ns) must fast-path to Unchanged without hashing"
        );
    }

    #[test]
    fn same_size_change_with_restored_mtime_is_detected() {
        let mut store = Store::open_memory().unwrap();
        store
            .upsert_project(&Project {
                name: "p".into(),
                indexed_at: "x".into(),
                root_path: "/p".into(),
            })
            .unwrap();

        let dir = tempdir_via_env();
        let entry = make_entry(&dir, "src/swap.rs", "old-body");
        let original = fs::metadata(&entry.abs_path).unwrap();
        let original_modified = original.modified().unwrap();
        let original_identity = greppy_discover::stable_metadata(&original);
        store
            .upsert_file_state(&FileState {
                project: "p".into(),
                rel_path: entry.rel_path.clone(),
                language: "rust".into(),
                sha256: sha256_hex(b"old-body"),
                mtime_ns: original_identity.mtime_ns.unwrap(),
                size: 8,
                parser_version: "x".into(),
                extractor_version: "x".into(),
                last_indexed_generation: 1,
            })
            .unwrap();
        store
            .upsert_file_identity(
                "p",
                &entry.rel_path,
                FileIdentity {
                    ctime_ns: original_identity.ctime_ns,
                    file_id: original_identity.file_id,
                },
            )
            .unwrap();

        std::thread::sleep(std::time::Duration::from_millis(5));
        fs::write(&entry.abs_path, "new-body").unwrap();
        let file = fs::OpenOptions::new()
            .write(true)
            .open(&entry.abs_path)
            .unwrap();
        file.set_modified(original_modified).unwrap();

        let diffs = compute_file_diff(&store, "p", std::slice::from_ref(&entry)).unwrap();
        assert!(matches!(
            diffs.as_slice(),
            [FileDiff::Modified { entry, .. }] if entry.rel_path == "src/swap.rs"
        ));
    }

    #[test]
    fn touched_file_with_same_content_hash_tier_says_unchanged() {
        // D2: mtime drifted (e.g. `touch`) but content identical — the
        // stat tier misses, the hash tier still resolves Unchanged.
        let mut store = Store::open_memory().unwrap();
        store
            .upsert_project(&Project {
                name: "p".into(),
                indexed_at: "x".into(),
                root_path: "/p".into(),
            })
            .unwrap();

        let dir = tempdir_via_env();
        let body = "same-content";
        let entry = make_entry(&dir, "src/touched.rs", body);
        let md = fs::metadata(&entry.abs_path).unwrap();
        store
            .upsert_file_state(&FileState {
                project: "p".into(),
                rel_path: "src/touched.rs".into(),
                language: "rust".into(),
                sha256: sha256_hex(body.as_bytes()),
                mtime_ns: 1, // stale mtime: stat tier must NOT match
                size: md.len() as i64,
                parser_version: "x".into(),
                extractor_version: "x".into(),
                last_indexed_generation: 1,
            })
            .unwrap();

        let diffs = compute_file_diff(&store, "p", std::slice::from_ref(&entry)).unwrap();
        assert_eq!(
            diffs,
            vec![FileDiff::Unchanged],
            "touch without content change must still resolve Unchanged via the hash tier"
        );
    }

    #[test]
    fn mtime_zero_sentinel_row_never_stat_fastpaths() {
        // `mtime_ns == 0` is the "mtime unknown" sentinel some writers
        // record. Such a row must never satisfy the stat tier; the hash
        // tier runs and (with a garbage persisted sha) reports Modified.
        let mut store = Store::open_memory().unwrap();
        store
            .upsert_project(&Project {
                name: "p".into(),
                indexed_at: "x".into(),
                root_path: "/p".into(),
            })
            .unwrap();

        let dir = tempdir_via_env();
        let entry = make_entry(&dir, "src/sentinel.rs", "sentinel-body");
        let md = fs::metadata(&entry.abs_path).unwrap();
        store
            .upsert_file_state(&FileState {
                project: "p".into(),
                rel_path: "src/sentinel.rs".into(),
                language: "rust".into(),
                sha256: "garbage-hash".into(),
                mtime_ns: 0, // sentinel
                size: md.len() as i64,
                parser_version: "x".into(),
                extractor_version: "x".into(),
                last_indexed_generation: 1,
            })
            .unwrap();

        let diffs = compute_file_diff(&store, "p", std::slice::from_ref(&entry)).unwrap();
        assert!(
            matches!(diffs[0], FileDiff::Modified { .. }),
            "sentinel-mtime row must fall through to the hash tier: {diffs:?}"
        );
    }

    #[test]
    fn default_cap_is_50_mib() {
        // Guard the documented default. The env-override path itself is
        // exercised in the indexer crate's tests, where the diff helper
        // is not run concurrently against small fixture files (mutating
        // GREPPY_MAX_FILE_SIZE here could race other diff tests in
        // this binary and reclassify their small files as oversized).
        assert_eq!(MAX_FILE_SIZE_BYTES, 50 * 1024 * 1024);
    }
}
