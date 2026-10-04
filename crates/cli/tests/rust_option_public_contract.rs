//! Public Rust Option-field caller contracts. Every query starts a fresh CLI
//! process, exercising persisted graphs rather than private resolver helpers.
#[test]
fn chained_callback_references_survive_public_query_reopen() {
    let f = Fixture::new();
    f.write("src/lib.rs", "mod callbacks; mod scene;\n");
    f.write(
        "src/callbacks.rs",
        r#"
pub fn predicate(value: i32) -> bool { value > 0 }
pub fn direct_callback(value: Option<i32>) -> bool {
    value.map(predicate).unwrap_or(false)
}
pub fn qualified_callback(value: Option<i32>, predicate: fn(i32) -> bool) -> bool {
    value.map(crate::callbacks::predicate).unwrap_or(false)
}
pub fn actual_call(value: i32) -> bool { predicate(value) }
pub fn shadowed_callback(value: Option<i32>, predicate: fn(i32) -> bool) -> bool {
    value.map(predicate).unwrap_or(false)
}
pub fn local_callback(value: Option<i32>) -> bool {
    let predicate = |value: i32| value > 1;
    value.map(predicate).unwrap_or(false)
}
pub fn conditional_callback(value: Option<i32>, candidate: Option<fn(i32) -> bool>) -> bool {
    if let Some(predicate) = candidate {
        value.map(predicate).unwrap_or(false)
    } else { false }
}
pub fn invoke(value: Option<i32>) -> bool { direct_callback(value) }
"#,
    );
    f.index();
    for _ in 0..2 {
        let callers = f.query(&["who-calls", "predicate", "--all", "--json"]);
        assert_eq!(hits(&callers), 3, "{callers}");
        let callers = callers.to_string();
        for shadowed in [
            "shadowed_callback",
            "local_callback",
            "conditional_callback",
        ] {
            assert!(!callers.contains(shadowed), "{callers}");
        }
        for name in ["direct_callback", "qualified_callback", "actual_call"] {
            assert!(callers.contains(name), "{callers}");
        }
        let impact = f.query(&["impact", "predicate", "--depth", "2", "--json"]);
        let impact = impact.to_string();
        for name in [
            "direct_callback",
            "qualified_callback",
            "actual_call",
            "invoke",
        ] {
            assert!(impact.contains(name), "{impact}");
        }
    }
}
use serde_json::Value;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

const TARGET: &str = "src/scene.rs::IrradianceField::uniform";
const SECOND: &str = "src/scene.rs::IrradianceField::storage";
const SCENE: &str = r#"
pub struct Manifest { pub remaster_irradiance: Option<crate::scene::IrradianceField> }
pub struct IrradianceField;
impl IrradianceField {
    pub fn uniform(&self, matrix: [f32;16]) {}
    pub fn storage(&self) {}
}
"#;
static NEXT: AtomicUsize = AtomicUsize::new(0);

struct Fixture {
    base: PathBuf,
    repo: PathBuf,
    store: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let base = std::env::temp_dir().join(format!(
            "greppy-rust-public-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let repo = base.join("repo");
        std::fs::create_dir_all(repo.join(".git")).unwrap();
        std::fs::create_dir_all(repo.join("src")).unwrap();
        let fixture = Self {
            store: base.join("store"),
            base,
            repo,
        };
        fixture.write("src/scene.rs", SCENE);
        fixture
    }
    fn write(&self, path: &str, source: &str) {
        std::fs::write(self.repo.join(path), source).unwrap();
    }
    fn run(&self, args: &[&str]) -> String {
        let output = Command::new(env!("CARGO_BIN_EXE_greppy"))
            .args(args)
            .current_dir(&self.repo)
            .env("GREPPY_STORE_DIR", &self.store)
            .env("GREPPY_TEST_SKIP_INFERENCE", "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "args={args:?}\nstdout={}\nstderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).unwrap()
    }
    fn index(&self) {
        self.run(&["index", "."]);
    }
    fn query(&self, args: &[&str]) -> Value {
        serde_json::from_str(&self.run(args)).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.base);
    }
}
fn caller(name: &str, prefix: &str) -> String {
    format!("{prefix}\npub fn {name}() {{ let manifest: Manifest = opaque(); let matrix: Option<[f32;16]> = None; match (manifest.remaster_irradiance.as_ref(), matrix) {{ (Some(field), Some(matrix)) => {{ field.uniform(matrix); field.storage(); }}, _ => () }} }}\n")
}
fn unresolved(value: &Value) -> Vec<Value> {
    value["unresolved_receivers"]
        .as_array()
        .cloned()
        .unwrap_or_default()
}
fn hits(value: &Value) -> usize {
    value["hits"]
        .as_array()
        .expect("public hits envelope")
        .len()
}
fn assert_unproven(f: &Fixture, name: &str) {
    let value = f.query(&["who-calls", TARGET, "--all", "--json"]);
    assert_eq!(hits(&value), 0, "{value}");
    let rows = unresolved(&value);
    assert!(rows.iter().any(|row| row["caller"] == name), "{value}");
    assert_eq!(value["callers_incomplete"], true, "{value}");
    let text = f.run(&["who-calls", TARGET, "--all"]);
    assert!(
        text.contains("no resolved callers") && text.contains("unresolved receiver:"),
        "{text}"
    );
    assert!(
        !text.lines().any(|line| line.trim() == "no callers"),
        "{text}"
    );
}

#[test]
fn named_imports_are_proven_and_opaque_gpu_shape_remains_unresolved_after_reopen() {
    let f = Fixture::new();
    f.write(
        "src/lib.rs",
        &caller(
            "safe_named",
            "mod scene; use crate::scene::Manifest; use std::collections::BTreeMap;",
        ),
    );
    f.index();
    for _ in 0..2 {
        let value = f.query(&["who-calls", TARGET, "--all", "--json"]);
        assert_eq!(hits(&value), 1, "{value}");
        assert!(unresolved(&value).is_empty(), "{value}");
        assert!(f
            .run(&["who-calls", TARGET, "--all"])
            .contains("safe_named"));
    }
    f.write("src/lib.rs", &caller("gpu_shape", "mod scene; use crate::scene::Manifest; use sha2::{Digest, Sha256}; use wasm_bindgen::{JsCast, prelude::*}; use wgpu::util::DeviceExt;"));
    f.index();
    assert_unproven(&f, "gpu_shape");
    f.write("src/unrelated.rs", "pub fn unrelated() {}\n");
    f.index();
    assert_unproven(&f, "gpu_shape");
}

#[test]
fn qualified_and_shadowed_macros_do_not_certify_receivers() {
    for prefix in [
        "mod scene; use crate::scene::Manifest; fn noise() { helper::assert!(true); }",
        "mod scene; use crate::scene::Manifest; macro_rules! println { ($($x:tt)*) => {} }",
        "mod scene; use crate::scene::Manifest; mod std {} use std::collections::BTreeMap;",
    ] {
        let f = Fixture::new();
        // Place the qualified macro in the caller's scope, where it can alter lookup.
        let source = if prefix.contains("helper::assert") {
            caller("opaque_macro", prefix)
                .replace("let manifest:", "helper::assert!(true); let manifest:")
        } else {
            caller("opaque_macro", prefix)
        };
        f.write("src/lib.rs", &source);
        f.index();
        assert_unproven(&f, "opaque_macro");
    }
}

#[test]
fn sparse_generic_and_consuming_trait_edits_remove_persisted_proof() {
    let f = Fixture::new();
    let source = caller("load_scene", "mod scene; use crate::scene::Manifest;");
    f.write("src/lib.rs", &source);
    f.index();
    assert_eq!(hits(&f.query(&["who-calls", TARGET, "--json"])), 1);
    f.write(
        "src/scene.rs",
        &SCENE
            .replace("struct Manifest", "struct Manifest<T>")
            .replace("Option<crate::scene::IrradianceField>", "Option<T>"),
    );
    f.index();
    assert_eq!(hits(&f.query(&["who-calls", TARGET, "--json"])), 0);
    f.write("src/scene.rs", SCENE);
    f.index();
    assert_eq!(hits(&f.query(&["who-calls", TARGET, "--json"])), 1);
    f.write("src/lib.rs", &source.replace("mod scene;", "mod scene; trait Consume { fn as_ref(self) -> Option<crate::scene::IrradianceField>; }"));
    f.index();
    for _ in 0..2 {
        assert_eq!(hits(&f.query(&["who-calls", TARGET, "--json"])), 0);
    }
}

#[test]
fn unresolved_single_and_multi_target_windows_and_late_path_filter() {
    let f = Fixture::new();
    f.write("src/lib.rs", "mod scene; mod early; mod later;\n");
    let prefix = "use crate::scene::Manifest; use external::Unknown;";
    let mut early = format!("{prefix}\n");
    for index in 0..24 {
        early.push_str(&caller(&format!("early_{index:02}"), ""));
    }
    f.write("src/early.rs", &early);
    f.write("src/later.rs", &caller("late_match", prefix));
    f.index();
    let all = f.query(&["who-calls", TARGET, "--all", "--json"]);
    let rows = unresolved(&all);
    assert_eq!(
        rows.len(),
        25,
        "plain --all must return every unresolved candidate: {all}"
    );
    for flag in ["--limit", "--max"] {
        for all_flag in [false, true] {
            for offset in [0usize, 2, 24] {
                let offset_arg = offset.to_string();
                let mut args = vec!["who-calls", TARGET, flag, "1", "--offset", &offset_arg];
                if all_flag {
                    args.push("--all");
                }
                let text = f.run(&args);
                args.push("--json");
                let page = f.query(&args);
                assert_eq!(
                    unresolved(&page),
                    rows[offset..offset + 1],
                    "{args:?}: {page}"
                );
                assert_eq!(
                    text.lines()
                        .filter(|line| line.starts_with("unresolved receiver:"))
                        .count(),
                    1,
                    "{text}"
                );
                assert!(
                    text.contains(rows[offset]["caller"].as_str().unwrap()),
                    "{text}"
                );
                assert_eq!(page["callers_incomplete"], true);
            }
        }
    }
    let both = f.query(&["who-calls", TARGET, SECOND, "--all", "--json"]);
    let both_rows = unresolved(&both);
    assert_eq!(both_rows.len(), 50, "{both}");
    for offset in [0usize, 1, 26] {
        let offset_arg = offset.to_string();
        let args = [
            "who-calls",
            TARGET,
            SECOND,
            "--all",
            "--limit",
            "1",
            "--offset",
            &offset_arg,
        ];
        let text = f.run(&args);
        let mut json_args = args.to_vec();
        json_args.push("--json");
        let page = f.query(&json_args);
        assert_eq!(unresolved(&page), both_rows[offset..offset + 1], "{page}");
        assert_eq!(
            text.lines()
                .filter(|line| line.starts_with("unresolved receiver:"))
                .count(),
            1,
            "{text}"
        );
    }
    let filtered = f.query(&[
        "who-calls",
        TARGET,
        "--path",
        "src/later.rs",
        "--limit",
        "1",
        "--json",
    ]);
    assert_eq!(
        unresolved(&filtered).len(),
        1,
        "late matching file must survive prefilter fetch: {filtered}"
    );
    assert_eq!(unresolved(&filtered)[0]["caller"], "late_match");
    assert_eq!(
        filtered["unresolved_omitted"], 0,
        "excluded files are not omitted matching rows: {filtered}"
    );
    assert_eq!(filtered["unresolved_truncated"], false, "{filtered}");
}

#[test]
fn mixed_confirmed_and_unresolved_rows_share_one_continuation_budget() {
    let f = Fixture::new();
    f.write("src/lib.rs", "mod scene; mod safe; mod uncertain;\n");
    f.write(
        "src/safe.rs",
        &caller("confirmed", "use crate::scene::Manifest;"),
    );
    let mut opaque = String::from("use crate::scene::Manifest; use external::Unknown;\n");
    opaque.push_str(&caller("uncertain_a", ""));
    opaque.push_str(&caller("uncertain_b", ""));
    f.write("src/uncertain.rs", &opaque);
    f.index();
    let full = f.query(&["who-calls", TARGET, "--all", "--json"]);
    assert_eq!(hits(&full), 1, "{full}");
    let uncertain = unresolved(&full);
    assert_eq!(uncertain.len(), 2, "{full}");
    for limit_flag in ["--limit", "--max"] {
        for offset in 0..3usize {
            let offset_arg = offset.to_string();
            let args = [
                "who-calls",
                TARGET,
                "--all",
                limit_flag,
                "1",
                "--offset",
                &offset_arg,
            ];
            let text = f.run(&args);
            let mut json_args = args.to_vec();
            json_args.push("--json");
            let page = f.query(&json_args);
            assert_eq!(hits(&page) + unresolved(&page).len(), 1, "{page}");
            assert_eq!(page["callers_incomplete"], true, "{page}");
            if offset == 0 {
                assert_eq!(hits(&page), 1, "{page}");
                assert!(unresolved(&page).is_empty(), "{page}");
                assert_eq!(page["unresolved_omitted"], 2, "{page}");
                assert_eq!(page["unresolved_truncated"], true, "{page}");
                assert!(text.contains("confirmed"), "{text}");
                assert!(!text.contains("unresolved receiver:"), "{text}");
            } else {
                assert_eq!(hits(&page), 0, "{page}");
                assert_eq!(unresolved(&page), uncertain[offset - 1..offset], "{page}");
                assert!(
                    text.contains(uncertain[offset - 1]["caller"].as_str().unwrap()),
                    "{text}"
                );
                assert_eq!(
                    text.lines()
                        .filter(|line| line.starts_with("unresolved receiver:"))
                        .count(),
                    1,
                    "{text}"
                );
                assert!(!text.contains(" confirmed"), "{text}");
            }
        }
    }
    let empty = f.query(&[
        "who-calls",
        TARGET,
        "--limit",
        "1",
        "--offset",
        "99",
        "--json",
    ]);
    assert_eq!(hits(&empty) + unresolved(&empty).len(), 0, "{empty}");
    assert_eq!(empty["callers_incomplete"], true, "{empty}");
}

#[test]
fn mixed_multi_target_text_and_json_offsets_select_the_same_actual_rows() {
    let f = Fixture::new();
    f.write("src/lib.rs", "mod scene; mod safe; mod uncertain;\n");
    // Only the first target has a confirmed caller; the second target's
    // candidates must remain uncertain throughout every continuation page.
    f.write(
        "src/safe.rs",
        &caller("confirmed_uniform", "use crate::scene::Manifest;").replace("field.storage();", ""),
    );
    let mut opaque = String::from("use crate::scene::Manifest; use external::Unknown;\n");
    opaque.push_str(&caller("uncertain_a", ""));
    opaque.push_str(&caller("uncertain_b", ""));
    f.write("src/uncertain.rs", &opaque);
    f.index();

    // Identity includes the target because the same uncertain caller appears
    // under both questions. Bare target headings are metadata, never rows.
    let json_rows = |value: &Value| -> Vec<(String, String, bool)> {
        let mut rows = value["hits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                (
                    row["target"].as_str().unwrap().to_owned(),
                    row["name"]
                        .as_str()
                        .unwrap()
                        .rsplit("::")
                        .next()
                        .unwrap()
                        .to_owned(),
                    false,
                )
            })
            .collect::<Vec<_>>();
        rows.extend(unresolved(value).iter().map(|row| {
            (
                row["target"].as_str().unwrap().to_owned(),
                row["caller"].as_str().unwrap().to_owned(),
                true,
            )
        }));
        rows
    };
    let text_rows = |text: &str| -> Vec<(String, String, bool)> {
        let mut target = None;
        let mut rows = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line == TARGET || line == SECOND {
                target = Some(line.to_owned());
                continue;
            }
            let words = line.split_whitespace().collect::<Vec<_>>();
            let identity = if line.starts_with("unresolved receiver:") {
                assert!(words.len() >= 4, "malformed diagnostic: {line}");
                Some((words[3], true))
            } else if words.first().is_some_and(|word| {
                word.starts_with("src/safe.rs:") || word.starts_with("src/uncertain.rs:")
            }) {
                assert!(words.len() >= 2, "malformed caller: {line}");
                Some((words[1].rsplit("::").next().unwrap(), false))
            } else {
                None
            };
            if let Some((caller, uncertain)) = identity {
                rows.push((
                    target.clone().expect("caller row must identify its target"),
                    caller.to_owned(),
                    uncertain,
                ));
            }
        }
        rows
    };
    let full = f.query(&["who-calls", TARGET, SECOND, "--all", "--json"]);
    assert_eq!(hits(&full), 1, "{full}");
    let canonical = json_rows(&full);
    assert_eq!(canonical.len(), 5, "{full}");
    assert_eq!(
        canonical[0],
        (TARGET.to_owned(), "confirmed_uniform".to_owned(), false)
    );
    assert!(canonical[1..].iter().all(|row| row.2), "{canonical:?}");
    assert_eq!(canonical.iter().filter(|row| row.0 == SECOND).count(), 2);

    for cap in ["--limit", "--max"] {
        for all in [false, true] {
            for offset in 0..=canonical.len() {
                let offset_arg = offset.to_string();
                let mut args = vec![
                    "who-calls",
                    TARGET,
                    SECOND,
                    cap,
                    "1",
                    "--offset",
                    &offset_arg,
                ];
                if all {
                    args.push("--all");
                }
                let text = f.run(&args);
                args.push("--json");
                let page = f.query(&args);
                let expected = canonical
                    .iter()
                    .skip(offset)
                    .take(1)
                    .cloned()
                    .collect::<Vec<_>>();
                assert_eq!(json_rows(&page), expected, "args={args:?}\nJSON={page}");
                assert_eq!(
                    text_rows(&text),
                    expected,
                    "args={args:?}\ntext={text}\nJSON={page}"
                );
                assert_eq!(page["callers_incomplete"], true, "{page}");
            }
        }
    }
    for offset in 0..=canonical.len() {
        let offset_arg = offset.to_string();
        let text = f.run(&[
            "who-calls",
            TARGET,
            SECOND,
            "--all",
            "--limit",
            "3",
            "--offset",
            &offset_arg,
            "--max-bytes",
            "1",
        ]);
        let expected = canonical
            .iter()
            .skip(offset)
            .take(1)
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(
            text_rows(&text),
            expected,
            "byte budget offset={offset}\n{text}"
        );
        let page = f.query(&[
            "who-calls",
            TARGET,
            SECOND,
            "--all",
            "--limit",
            "3",
            "--offset",
            &offset_arg,
            "--max-bytes",
            "1",
            "--json",
        ]);
        assert_eq!(
            json_rows(&page),
            expected,
            "byte budget offset={offset}\n{page}"
        );
        for target in page["targets"].as_array().unwrap() {
            let symbol = target["symbol"].as_str().unwrap();
            let retained = unresolved(&page)
                .iter()
                .filter(|row| row["target"] == symbol)
                .count();
            assert_eq!(target["unresolved_omitted"], 2 - retained, "{page}");
            assert_eq!(target["unresolved_truncated"], retained < 2, "{page}");
        }
        if offset < canonical.len() - 1 {
            assert!(text.contains(&format!("--offset {}", offset + 1)), "{text}");
        } else {
            assert!(!text.contains("try:"), "{text}");
        }
    }
}

#[test]
fn v9_inline_namespace_proof_is_reextracted_on_normal_query() {
    let f = Fixture::new();
    f.write(
        "src/lib.rs",
        &caller(
            "inline_std",
            "mod scene; use crate::scene::Manifest; mod std {} use std::collections::BTreeMap;",
        ),
    );
    f.index();
    let status = f.query(&["index", "status", "--json", "--diagnostics"]);
    let db = PathBuf::from(status["store_path"].as_str().expect("status store path"));
    assert!(
        db.starts_with(&f.store),
        "fixture must own the migrated database"
    );
    assert!(db.exists(), "fixture graph missing: {}", db.display());
    let conn = rusqlite::Connection::open(&db).unwrap();
    conn.execute_batch("DELETE FROM schema_meta WHERE key IN ('greppy.rust_caller_edges_repair.v10','greppy.rust_caller_edges_repair.v11','greppy.rust_caller_edges_repair.v12'); INSERT OR REPLACE INTO schema_meta(key,value) VALUES('greppy.rust_caller_edges_repair.v9','complete'); UPDATE raw_edges SET properties=json_remove(properties,'$.receiver_provenance.limits.standard_namespace_bindings') WHERE edge_type='CALLS'; UPDATE edges SET edge_type='CALLS' WHERE edge_type='UNRESOLVED_CALLS';").unwrap();
    assert!(
        conn.query_row(
            "SELECT count(*) FROM edges WHERE edge_type='CALLS'",
            [],
            |row| row.get::<_, i64>(0)
        )
        .unwrap()
            > 0
    );
    drop(conn);
    for _ in 0..2 {
        assert_unproven(&f, "inline_std");
    }
    let conn = rusqlite::Connection::open(&db).unwrap();
    assert_eq!(
        conn.query_row(
            "SELECT value FROM schema_meta WHERE key='greppy.rust_caller_edges_repair.v12'",
            [],
            |row| row.get::<_, String>(0)
        )
        .unwrap(),
        "complete"
    );
    assert!(conn.query_row("SELECT count(*) FROM raw_edges WHERE properties LIKE '%standard_namespace_bindings%std%'", [], |row| row.get::<_, i64>(0)).unwrap() > 0);
}
