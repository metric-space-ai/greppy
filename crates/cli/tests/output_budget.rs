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

fn ambiguous_read_fixture() -> &'static (PathBuf, PathBuf) {
    static FIXTURE: OnceLock<(PathBuf, PathBuf)> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let base = std::env::temp_dir().join(format!(
            "greppy-cli-output-budget-ambiguous-read-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        let repo = base.join("repo");
        let store = base.join("store");
        std::fs::create_dir_all(repo.join(".git")).unwrap();
        std::fs::create_dir_all(repo.join("src")).unwrap();
        for index in 0..83 {
            std::fs::write(
                repo.join(format!(
                    "src/duplicate_definition_with_long_fixture_path_{index:03}.rs"
                )),
                format!("pub fn main() {{ println!(\"duplicate {index}\"); }}\n"),
            )
            .unwrap();
        }
        std::fs::write(
            repo.join("src/unique.rs"),
            "pub fn singular() -> usize { 7 }\n",
        )
        .unwrap();
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

fn candidate_paths(output: &str) -> Vec<&str> {
    output
        .lines()
        .filter(|line| line.starts_with("src/duplicate_definition_with_long_fixture_path_"))
        .collect()
}

fn continuation_offset(output: &str) -> usize {
    let retry = output
        .lines()
        .find_map(|line| line.strip_prefix("try: "))
        .expect("truncated output carries a retry command");
    let marker = "--offset ";
    retry
        .split_once(marker)
        .and_then(|(_, value)| value.split_whitespace().next())
        .and_then(|value| value.parse().ok())
        .expect("retry command carries a numeric offset")
}

#[test]
fn ambiguous_read_text_budget_preserves_failure_and_pages_without_duplicates() {
    let (repo, store) = ambiguous_read_fixture();
    let budget = 3_000usize;
    let (code, first_stdout, first_stderr) =
        run(repo, store, &["read", "main", "--max-bytes", "3000"]);
    assert_eq!(code, 1, "stdout={first_stdout}\nstderr={first_stderr}");
    assert!(
        first_stdout.len() <= budget,
        "{} bytes\n{first_stdout}",
        first_stdout.len()
    );
    assert!(first_stdout.contains("`main` is 83 definitions"));
    assert!(first_stdout.contains("truncated: true"));
    let first_paths = candidate_paths(&first_stdout);
    assert!(!first_paths.is_empty(), "{first_stdout}");
    let offset = continuation_offset(&first_stdout);
    assert!(offset > 0, "{first_stdout}");

    let offset_arg = offset.to_string();
    let (code, second_stdout, second_stderr) = run(
        repo,
        store,
        &[
            "read",
            "main",
            "--max-bytes",
            "3000",
            "--offset",
            &offset_arg,
        ],
    );
    assert_eq!(code, 1, "stdout={second_stdout}\nstderr={second_stderr}");
    assert!(
        second_stdout.len() <= budget,
        "{} bytes\n{second_stdout}",
        second_stdout.len()
    );
    let second_paths = candidate_paths(&second_stdout);
    assert!(!second_paths.is_empty(), "{second_stdout}");
    assert!(
        first_paths
            .iter()
            .all(|candidate| !second_paths.contains(candidate)),
        "first={first_paths:?} second={second_paths:?}"
    );
}

#[test]
fn ambiguous_read_json_and_unique_read_obey_the_same_budget() {
    let (repo, store) = ambiguous_read_fixture();
    let budget = 3_000usize;
    let (code, stdout, stderr) = run(
        repo,
        store,
        &["read", "main", "--json", "--max-bytes", "3000"],
    );
    assert_eq!(code, 1, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.len() <= budget, "{} bytes\n{stdout}", stdout.len());
    let value: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(value["status"], "ambiguous");
    assert_eq!(value["total"], 83);
    assert_eq!(value["truncated"], true);
    assert!(!value["candidates"].as_array().unwrap().is_empty());
    assert!(value["try"].as_str().unwrap().contains("--offset "));

    let (code, stdout, stderr) = run(repo, store, &["read", "singular", "--max-bytes", "3000"]);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.len() <= budget, "{} bytes\n{stdout}", stdout.len());
    assert!(
        stdout.contains("pub fn singular() -> usize { 7 }"),
        "{stdout}"
    );
    assert!(!stdout.contains("truncated: true"), "{stdout}");
}

// 0.3.0 contract (AGENTS.md, "ON EVERY COMMAND"): --limit caps the results
// and --offset K starts at the Kth, on `search-pattern` (the replacement
// for the retired `search-code`) exactly as on the navigation commands.
// The stdout budget (--max-bytes) stays valid and the JSON envelope gains
// `total` / `shown` / `offset` / `try` cursor keys.
#[test]
fn search_pattern_json_budget_is_valid_and_offset_continues_without_duplicates() {
    let (repo, store) = fixture();
    let budget = 1_000usize;
    let (code, stdout, stderr) = run(
        repo,
        store,
        &[
            "search-pattern",
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
        .map(|hit| hit["matches"][0]["location"].as_str().unwrap().to_string())
        .collect::<Vec<_>>();
    assert_eq!(first_locations.len(), next_offset, "{stdout}");
    let next_offset_arg = next_offset.to_string();

    let (code, stdout, stderr) = run(
        repo,
        store,
        &[
            "search-pattern",
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
        .map(|hit| hit["matches"][0]["location"].as_str().unwrap().to_string())
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
    // The 0.3.0 JSON envelope (freshness, incomplete_providers, targets,
    // expand) is larger than the pre-0.3.0 one: 950 bytes fit zero hits,
    // 1050 fit at least one. The contract is unchanged: stdout stays within
    // budget, `total` is the true number, and `try` is an executable retry
    // whose --offset advances past the shown hits.
    let budget = 1_050usize;
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
            "1050",
        ],
    );

    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.len() <= budget, "{} bytes\n{stdout}", stdout.len());
    let value: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(value["truncated"], true);
    assert_eq!(value["total"], 12);
    assert!(!value["hits"].as_array().unwrap().is_empty(), "{stdout}");
    let shown = value["shown"].as_u64().unwrap();
    assert!(shown > 0, "{stdout}");
    let retry = value["try"].as_str().unwrap();
    assert!(retry.starts_with("greppy "), "{retry}");
    assert!(retry.contains("who-calls target"), "{retry}");
    assert!(
        retry.contains(&format!("--offset {shown}")),
        "the retry must make progress past the {shown} shown hits: {retry}"
    );
}

#[test]
fn multi_brief_json_budget_keeps_whole_ordered_results_and_retry() {
    let (repo, store) = fixture();
    let budget = 3_000usize;
    let (code, stdout, stderr) = run(
        repo,
        store,
        &[
            "brief",
            "caller_0",
            "caller_1",
            "--json",
            "--max-bytes",
            "3000",
        ],
    );

    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.len() <= budget, "{} bytes\n{stdout}", stdout.len());
    let value: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(value["schema_version"], "greppy.brief.batch.v1");
    assert_eq!(value["total"], 2);
    let results = value["results"].as_array().expect("batch results");
    assert!(!results.is_empty(), "{stdout}");
    assert_eq!(results[0]["query"], "caller_0");
    if results.len() == 2 {
        assert_eq!(results[1]["query"], "caller_1");
    } else {
        assert_eq!(value["truncated"], true);
        assert!(value["try"].as_str().unwrap().contains("--offset 1"));
    }
}

#[test]
fn mini_budget_never_cuts_a_json_diagnostic() {
    let (repo, store) = fixture();
    let (code, stdout, stderr) = run(
        repo,
        store,
        &["read", "zzz_missing_symbol", "--json", "--max-bytes", "1"],
    );

    assert_eq!(code, 1, "stdout={stdout}\nstderr={stderr}");
    let value: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(value["status"], "not-found");
    assert_eq!(value["query"], "zzz_missing_symbol");
}
