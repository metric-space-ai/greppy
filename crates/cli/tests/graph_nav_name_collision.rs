//! P1 regression (re-review, navigation resolution):
//! `resolve_symbol_id` used to pick the FIRST node named `S` among all
//! graph nodes, landing on the WRONG one when a name is shared:
//!
//!   * `find-usages Store`     resolved to the `EnumVariant`
//!     `Kind::Store` (which has no incoming usages) instead of the
//!     `Store` struct that IS referenced — printing "(no usages)".
//!   * `find-usages IndexReport` resolved to `Impl::IndexReport`
//!     instead of the `IndexReport` struct, again missing the usages.
//!
//! The fix (1) ranks candidates so a type/def-like label
//! (Struct/Enum/Trait/Function/Method/TypeAlias) wins over
//! Impl/EnumVariant/AssocConst/AssocType/Module/Call/Import, and (2) for
//! who-calls/find-usages aggregates incoming edges across ALL nodes that
//! share the exact name + a primary label (so a Struct and its Impl both
//! contribute), deterministically.
//!
//! These tests index a real fixture end-to-end and drive the shipped
//! `grepplus` binary, so they reproduce the bug exactly as an agent
//! would have hit it (they fail before the fix, pass after).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_grepplus")
}

fn fresh_dir(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("grepplus-cli-namecollision-{tag}-{pid}-{n}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// Build a repo where:
///
///  * `IndexReport` is BOTH a `Struct` (types.rs) AND has an
///    `impl IndexReport` block (types.rs) — so the graph has a `Struct`
///    node and an `Impl` node that share the exact name `IndexReport`.
///    `lib.rs::summarise(r: types::IndexReport)` references the type
///    (TYPE_REF into the Struct).
///
///  * `Store` is a `Struct` (types.rs) that is referenced by
///    `lib.rs::persist(s: types::Store)` (TYPE_REF), AND an UNRELATED
///    enum `Kind` (types.rs) has a variant `Store` (an `EnumVariant`
///    node sharing the name `Store`). The struct is the right target.
///
/// Returns (repo_root, store_dir).
fn make_repo(tag: &str) -> (PathBuf, PathBuf) {
    let root = fresh_dir(tag);
    let repo = root.join("repo");
    let src = repo.join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::create_dir_all(repo.join(".git")).unwrap();

    std::fs::write(
        src.join("lib.rs"),
        r#"
mod types;

fn summarise(r: types::IndexReport) -> u32 { r.files }

fn persist(s: types::Store) -> u32 { s.count }

fn drive() {
    let r = types::IndexReport { files: 0 };
    let _ = r.total();
}
"#,
    )
    .unwrap();

    std::fs::write(
        src.join("types.rs"),
        r#"
pub struct IndexReport { pub files: u32 }

impl IndexReport {
    pub fn total(&self) -> u32 { self.files }
}

pub struct Store { pub count: u32 }

pub enum Kind {
    Store,
    Memory,
}
"#,
    )
    .unwrap();

    let store = root.join("store");
    (repo, store)
}

fn run(args: &[&str], cwd: &Path, store_dir: &Path) -> (i32, String, String) {
    let out = Command::new(bin())
        .args(args)
        .current_dir(cwd)
        .env("GREPPLUS_STORE_DIR", store_dir)
        .output()
        .expect("spawn grepplus");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn index_fixture(tag: &str) -> (PathBuf, PathBuf) {
    let (repo, store) = make_repo(tag);
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(
        code, 0,
        "index . should succeed; stderr={err}\nstdout={out}"
    );
    (repo, store)
}

// ---------------------------------------------------------------------------
// find-usages: a name shared by a Struct and its Impl resolves to (and
// aggregates across) the Struct so the real usages are reported.
// ---------------------------------------------------------------------------

#[test]
fn find_usages_struct_with_impl_same_name_returns_usages() {
    let (repo, store) = index_fixture("usages-impl-collision");

    // `IndexReport` is both a Struct and an `impl IndexReport`. Before
    // the fix this resolved to the Impl node and printed "(no usages)".
    let (code, out, err) = run(&["find-usages", "IndexReport"], &repo, &store);
    assert_eq!(
        code, 0,
        "find-usages should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        !out.contains("(no usages)"),
        "IndexReport IS referenced by summarise(); resolving to its Impl \
         must not hide the usages; got: {out:?}"
    );
    assert!(
        out.contains("summarise"),
        "find-usages IndexReport must list the referrer `summarise`; got: {out:?}"
    );
    assert!(
        out.contains("src/lib.rs:"),
        "find-usages must print the referrer's file:line; got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// find-usages: a struct whose name is ALSO an unrelated EnumVariant
// resolves to the struct, not the variant.
// ---------------------------------------------------------------------------

#[test]
fn find_usages_struct_sharing_name_with_enum_variant_targets_struct() {
    let (repo, store) = index_fixture("usages-variant-collision");

    // `Store` is a struct referenced by persist(), and ALSO an unrelated
    // `Kind::Store` enum variant. Before the fix `find-usages Store`
    // could resolve to the EnumVariant (no usages) and miss the struct.
    let (code, out, err) = run(&["find-usages", "Store"], &repo, &store);
    assert_eq!(
        code, 0,
        "find-usages should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        !out.contains("(no usages)"),
        "the `Store` struct is referenced by persist(); the shared \
         EnumVariant name must not steal resolution; got: {out:?}"
    );
    assert!(
        out.contains("persist"),
        "find-usages Store must list the referrer `persist`; got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// who-calls: a method defined under an impl whose type shares the struct
// name. Aggregating across the Struct + Impl finds the method's callers.
// ---------------------------------------------------------------------------

#[test]
fn who_calls_method_on_struct_with_shared_impl_name_finds_caller() {
    let (repo, store) = index_fixture("whocalls-impl-collision");

    // `total` is a method on `IndexReport`; `drive()` calls `r.total()`.
    // The resolution must land on the Method (not be confused by the
    // Struct/Impl name collision on `IndexReport`).
    let (code, out, err) = run(&["who-calls", "total"], &repo, &store);
    assert_eq!(
        code, 0,
        "who-calls should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("drive"),
        "who-calls total must list `drive` as the caller; got: {out:?}"
    );
    assert!(
        !out.contains("(no callers)"),
        "total() is called by drive(); got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// trace --direction incoming: a Struct/Impl name collision must resolve
// to the Struct (rank 0) so the incoming walk targets the right node.
// ---------------------------------------------------------------------------

#[test]
fn trace_incoming_struct_with_impl_same_name_reaches_referrer() {
    let (repo, store) = index_fixture("trace-impl-collision");

    // Incoming USAGE into `IndexReport` (the Class) comes from `summarise`
    // (a type reference, persisted under the unified C-reference USAGE label).
    // The ranking fix makes resolution prefer the type def over the Impl so
    // the incoming walk has a referrer to find.
    let (code, out, err) = run(
        &[
            "trace",
            "--symbol",
            "IndexReport",
            "--direction",
            "incoming",
            "--edge",
            "USAGE",
        ],
        &repo,
        &store,
    );
    assert_eq!(
        code, 0,
        "trace incoming should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("IndexReport"),
        "trace must include the start symbol IndexReport; got: {out:?}"
    );
    assert!(
        out.contains("summarise"),
        "incoming USAGE trace from IndexReport must reach `summarise`; got: {out:?}"
    );
}
