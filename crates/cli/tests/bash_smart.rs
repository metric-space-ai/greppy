//! End-to-end coverage for the training-free bash-smart delivery contract.

#![cfg(unix)]

use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

struct Workspace {
    repo: PathBuf,
    store: PathBuf,
    base: PathBuf,
}

impl Drop for Workspace {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.base);
    }
}

fn fresh_workspace(tag: &str) -> Workspace {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let base = std::env::temp_dir().join(format!(
        "greppy-cli-bash-smart-{tag}-{}-{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&base);
    let repo = base.join("repo");
    let store = base.join("store");
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    Workspace { repo, store, base }
}

fn command(workspace: &Workspace) -> Command {
    let mut command = Command::new(bin());
    command
        .current_dir(&workspace.repo)
        .env("GREPPY_STORE_DIR", &workspace.store)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1");
    command
}

fn run(workspace: &Workspace, args: &[&str]) -> Output {
    command(workspace).args(args).output().expect("run greppy")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

fn expand_id(stdout: &str) -> &str {
    let marker = "greppy expand ";
    let rest = stdout
        .split(marker)
        .nth(1)
        .unwrap_or_else(|| panic!("missing expand command in:\n{stdout}"));
    rest.split_whitespace().next().unwrap()
}

#[test]
fn short_output_is_verbatim_and_exit_code_passes_through() {
    let workspace = fresh_workspace("short");
    let output = run(
        &workspace,
        &[
            "bash-smart",
            "--",
            "sh",
            "-c",
            "printf 'out\\n'; printf 'err\\n' >&2; exit 3",
        ],
    );

    assert_eq!(output.status.code(), Some(3));
    assert_eq!(output.stdout, b"out\n");
    assert_eq!(output.stderr, b"err\n");
}

#[test]
fn long_output_has_head_gap_tail_and_expandable_raw_middle() {
    let workspace = fresh_workspace("long");
    let output = run(
        &workspace,
        &[
            "bash-smart",
            "--",
            "sh",
            "-c",
            "for i in $(seq 200); do echo line $i; done",
        ],
    );
    let stdout = text(&output.stdout);
    let stderr = text(&output.stderr);

    assert_eq!(output.status.code(), Some(0), "stderr={stderr}");
    assert!(stdout.starts_with("line 1\nline 2\n"), "{stdout}");
    assert!(stdout.contains("line 20\n"), "{stdout}");
    assert!(stdout.contains("… 150 lines — greppy expand "), "{stdout}");
    assert!(stdout.contains(" continues at 21\n"), "{stdout}");
    assert!(stdout.contains("line 171\n"), "{stdout}");
    assert!(stdout.ends_with("line 200\n"), "{stdout}");
    assert!(!stdout.contains("line 170\n"), "{stdout}");
    assert!(stderr.is_empty(), "{stderr}");

    let id = expand_id(&stdout);
    let expanded = run(&workspace, &["expand", id]);
    assert_eq!(expanded.status.code(), Some(0));
    assert!(expanded.stderr.is_empty(), "{}", text(&expanded.stderr));
    assert!(expanded.stdout.starts_with(b"line 21\n"));
    assert!(expanded.stdout.ends_with(b"line 200\n"));
    assert_eq!(
        expanded
            .stdout
            .iter()
            .filter(|byte| **byte == b'\n')
            .count(),
        180
    );
}

#[test]
fn repeated_middle_is_collapsed_arithmetically() {
    let workspace = fresh_workspace("collapse");
    let output = run(
        &workspace,
        &["bash-smart", "--", "sh", "-c", "yes hello | head -300"],
    );
    let stdout = text(&output.stdout);

    assert_eq!(output.status.code(), Some(0));
    assert!(
        stdout.contains("… 249 weitere `hello`-Zeilen\n"),
        "{stdout}"
    );
    assert!(stdout.contains("… 250 lines — greppy expand "), "{stdout}");
}

#[test]
fn timeout_kills_descendants_and_marks_partial_unterminated_output() {
    let workspace = fresh_workspace("timeout");
    let started = Instant::now();
    let output = command(&workspace)
        .env("GREPPY_BASH_SMART_TIMEOUT_MS", "75")
        .args(["bash-smart", "--", "sh", "-c", "printf partial; sleep 5"])
        .output()
        .expect("run timed bash-smart");
    let stdout = text(&output.stdout);
    let stderr = text(&output.stderr);

    assert!(started.elapsed() < Duration::from_secs(2));
    assert_eq!(output.status.code(), Some(137), "stderr={stderr}");
    assert!(stdout.starts_with("partial\n"), "{stdout}");
    assert!(stdout.contains("greppy expand "), "{stdout}");
    assert!(
        stderr.contains("bash-smart: partial output ends with an unterminated line\n"),
        "{stderr}"
    );
    assert!(
        stderr.contains("bash-smart: timed out after 75 ms;"),
        "{stderr}"
    );
}
