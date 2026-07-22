//! Universal stdout budget and offset cursor coverage.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

fn fixture() -> &'static (PathBuf, PathBuf) {
    static FIXTURE: OnceLock<(PathBuf, PathBuf)> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let base =
            std::env::temp_dir().join(format!("greppy-cli-output-budget-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let repo = base.join("repo");
        let store = base.join("store");
        std::fs::create_dir_all(repo.join(".git")).unwrap();
        std::fs::create_dir_all(repo.join("src")).unwrap();
        let mut source = String::from("pub fn target() {}\n");
        for index in 0..12 {
            source.push_str(&format!(
                "pub fn caller_{index}() {{ target(); }} // needle-{index}\n"
            ));
        }
        std::fs::write(repo.join("src/lib.rs"), source).unwrap();
        let (code, stdout, stderr) = run(&repo, &store, &["index", "."]);
        assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
        (repo, store)
    })
}

fn run(repo: &Path, store: &Path, args: &[&str]) -> (i32, String, String) {
    let output = Command::new(bin())
        .args(args)
        .current_dir(repo)
        .env("GREPPY_STORE_DIR", store)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .output()
        .expect("run greppy");
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

#[test]
fn search_code_json_budget_is_valid_and_offset_continues_without_duplicates() {
    let (repo, store) = fixture();
    let budget = 1_000usize;
    let (code, stdout, stderr) = run(
        repo,
        store,
        &[
            "search-code",
            "needle",
            "--json",
            "--limit",
            "20",
            "--max-bytes",
            "1000",
        ],
    );
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.len() <= budget, "{} bytes\n{stdout}", stdout.len());
    let first: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(first["truncated"], true);
    assert_eq!(first["total"], 12);
    let next_offset = first["shown"].as_u64().unwrap() as usize;
    assert!(next_offset > 0, "{stdout}");
    assert!(first["try"]
        .as_str()
        .unwrap()
        .contains(&format!("--offset {next_offset}")));
    let first_locations = first["hits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|hit| hit["location"].as_str().unwrap().to_string())
        .collect::<Vec<_>>();
    assert_eq!(first_locations.len(), next_offset, "{stdout}");
    let next_offset_arg = next_offset.to_string();

    let (code, stdout, stderr) = run(
        repo,
        store,
        &[
            "search-code",
            "needle",
            "--json",
            "--limit",
            "20",
            "--max-bytes",
            "1000",
            "--offset",
            &next_offset_arg,
        ],
    );
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.len() <= budget, "{} bytes\n{stdout}", stdout.len());
    let second: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(second["offset"], next_offset);
    assert_eq!(second["total"], 12);
    let second_locations = second["hits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|hit| hit["location"].as_str().unwrap().to_string())
        .collect::<Vec<_>>();
    assert!(!second_locations.is_empty(), "{stdout}");
    assert!(
        first_locations
            .iter()
            .all(|location| !second_locations.contains(location)),
        "first={first_locations:?} second={second_locations:?}"
    );
}

#[test]
fn who_calls_json_budget_keeps_structure_total_and_executable_retry() {
    let (repo, store) = fixture();
    let budget = 950usize;
    let (code, stdout, stderr) = run(
        repo,
        store,
        &[
            "who-calls",
            "target",
            "--json",
            "--limit",
            "20",
            "--max-bytes",
            "950",
        ],
    );

    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.len() <= budget, "{} bytes\n{stdout}", stdout.len());
    let value: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(value["truncated"], true);
    assert_eq!(value["total"], 12);
    assert!(!value["hits"].as_array().unwrap().is_empty(), "{stdout}");
    let retry = value["try"].as_str().unwrap();
    assert!(retry.starts_with("greppy "), "{retry}");
    assert!(retry.contains("who-calls target"), "{retry}");
    assert!(retry.contains("--offset "), "{retry}");
}

#[test]
fn mini_budget_never_cuts_a_json_diagnostic() {
    let (repo, store) = fixture();
    let (code, stdout, stderr) = run(
        repo,
        store,
        &["read", "src/missing.rs", "--json", "--max-bytes", "1"],
    );

    assert_eq!(code, 10, "stdout={stdout}\nstderr={stderr}");
    let value: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(value["status"], "not-found");
    assert_eq!(value["path"], "src/missing.rs");
    assert_eq!(value["path_candidates"][0], "src/lib.rs");
}
