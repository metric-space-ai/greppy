//! WP-T: workspace Rust calls, language-scoped names, and Rust alias/receiver/
//! prelude resolution, as the CLI reports them.

use std::path::{Path, PathBuf};
use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

fn fresh_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "greppy-cli-wp-t-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

fn run(args: &[&str], cwd: &Path, store: &Path) -> (i32, String, String) {
    let out = Command::new(bin())
        .args(args)
        .current_dir(cwd)
        .env("GREPPY_STORE_DIR", store)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .output()
        .expect("spawn greppy");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn index(repo: &Path, store: &Path) {
    let (code, stdout, stderr) = run(&["index", "."], repo, store);
    assert_eq!(code, 0, "index failed\nstdout={stdout}\nstderr={stderr}");
}

#[test]
fn workspace_path_dependency_who_calls_crosses_crates() {
    let root = fresh_dir("workspace");
    let repo = root.join("repo");
    std::fs::create_dir_all(repo.join("crates/crate_b/src")).unwrap();
    std::fs::create_dir_all(repo.join("crates/app/src")).unwrap();
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::write(
        repo.join("Cargo.toml"),
        "[workspace]\nmembers = [\"crates/crate_b\", \"crates/app\"]\nresolver = \"2\"\n",
    )
    .unwrap();
    std::fs::write(
        repo.join("crates/crate_b/Cargo.toml"),
        "[package]\nname = \"crate_b\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    std::fs::write(
        repo.join("crates/crate_b/src/lib.rs"),
        "pub fn clamp_percent(value: i32) -> i32 { value }\n",
    )
    .unwrap();
    std::fs::write(
        repo.join("crates/app/Cargo.toml"),
        "[package]\nname = \"app\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\ncrate_b = { path = \"../crate_b\" }\n",
    )
    .unwrap();
    std::fs::write(
        repo.join("crates/app/src/lib.rs"),
        "use crate_b::clamp_percent;\npub fn progress() { clamp_percent(1); crate_b::clamp_percent(2); }\n",
    )
    .unwrap();
    let store = root.join("store");
    index(&repo, &store);
    let (code, out, err) = run(
        &["who-calls", "crates/crate_b/src/lib.rs::clamp_percent"],
        &repo,
        &store,
    );
    assert_eq!(code, 0, "stderr={err}\nstdout={out}");
    assert!(
        out.contains("progress"),
        "who-calls must list the other crate: {out}"
    );
}

#[test]
fn java_call_survives_a_typescript_namesake() {
    let root = fresh_dir("lang");
    let repo = root.join("repo");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::write(
        repo.join("src/Pricing.java"),
        "class Pricing { static int parsePrice(String raw) { return 1; } }\n",
    )
    .unwrap();
    std::fs::write(
        repo.join("src/Cart.java"),
        "class Cart { int cartTotal() { return Pricing.parsePrice(\"1\"); } }\n",
    )
    .unwrap();
    std::fs::write(
        repo.join("src/price.ts"),
        "export function parsePrice(raw: string): number { return 1; }\nexport function cartTotal(): number { return parsePrice(\"1\"); }\n",
    )
    .unwrap();
    let store = root.join("store");
    index(&repo, &store);
    let (code, out, err) = run(
        &["who-calls", "src/Pricing.java::parsePrice"],
        &repo,
        &store,
    );
    assert_eq!(code, 0, "stderr={err}\nstdout={out}");
    assert!(
        out.contains("cartTotal"),
        "Java who-calls must see Cart, not be suppressed by TypeScript: {out}"
    );
    assert!(
        !out.contains("price.ts"),
        "Java who-calls must not list the TypeScript caller: {out}"
    );
}

#[test]
fn rust_alias_field_and_prelude_calls_stay_precise() {
    let root = fresh_dir("rust-calls");
    let repo = root.join("repo");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::write(
        repo.join("src/lib.rs"),
        "mod helper;\nuse helper::RenameRule;\nstruct RenameAllRules { serialize: RenameRule }\nimpl RenameAllRules {\n    fn apply(&self, value: &str) -> String { self.serialize.apply_to_field(value) }\n}\npub fn test() {}\npub fn real() {}\npub fn assert_de() {}\npub fn Err() {}\npub fn expand_derive_deserialize() { let _ = Err(\"no\"); }\n",
    )
    .unwrap();
    std::fs::write(
        repo.join("src/helper.rs"),
        "pub enum RenameRule { None }\nimpl RenameRule {\n    pub fn apply_to_field(&self, value: &str) -> String { value.to_string() }\n}\nuse crate::real as test;\npub fn renamed() { test(); }\npub fn local_path() { let test = crate::assert_de; test(); }\n",
    )
    .unwrap();
    let store = root.join("store");
    index(&repo, &store);

    let (code, out, err) = run(&["who-calls", "src/lib.rs::real"], &repo, &store);
    assert_eq!(code, 0, "stderr={err}\nstdout={out}");
    assert!(out.contains("renamed"), "alias call must reach real: {out}");

    let (code, out, err) = run(&["who-calls", "src/lib.rs::test"], &repo, &store);
    assert_eq!(code, 0, "stderr={err}\nstdout={out}");
    assert!(
        !out.contains("renamed") && !out.contains("local_path"),
        "local name test must not attach to fn test: {out}"
    );

    let (code, out, err) = run(
        &["who-calls", "src/helper.rs::apply_to_field"],
        &repo,
        &store,
    );
    assert_eq!(code, 0, "stderr={err}\nstdout={out}");
    assert!(
        out.lines()
            .any(|line| line.split_whitespace().any(|part| part == "apply")),
        "self.serialize.apply_to_field must resolve: {out}"
    );

    let (code, out, err) = run(
        &["callees", "src/lib.rs::expand_derive_deserialize"],
        &repo,
        &store,
    );
    assert_eq!(code, 0, "stderr={err}\nstdout={out}");
    assert!(
        !out.contains("Function") && !out.lines().any(|line| line.contains("Err")),
        "unqualified Err must not be the user function: {out}"
    );
}
