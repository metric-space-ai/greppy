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
    assert!(stderr.is_empty(), "{stderr}");

    let id = expand_id(&stdout);
    let expected = (1..=21)
        .map(|line| format!("line {line}\n"))
        .chain(std::iter::once(format!(
            "… lines 22-170 (149 collapsed `line …` repeats) — greppy expand {id}\n"
        )))
        .chain((171..=200).map(|line| format!("line {line}\n")))
        .collect::<String>();
    assert_eq!(stdout, expected);
    assert_eq!(
        stdout.lines().filter(|line| line.starts_with('…')).count(),
        1,
        "{stdout}"
    );

    let expanded = run(&workspace, &["expand", id]);
    assert_eq!(expanded.status.code(), Some(0));
    assert!(expanded.stderr.is_empty(), "{}", text(&expanded.stderr));
    let expected_expanded = (22..=170)
        .map(|line| format!("line {line}\n"))
        .collect::<String>();
    assert_eq!(expanded.stdout, expected_expanded.as_bytes());
    assert_eq!(text(&expanded.stdout).lines().count(), 149);
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
    assert!(output.stderr.is_empty(), "{}", text(&output.stderr));

    let id = expand_id(&stdout);
    let expected = format!(
        "{}… lines 22-270 (249 collapsed `hello` repeats) — greppy expand {id}\n{}",
        "hello\n".repeat(21),
        "hello\n".repeat(30)
    );
    assert_eq!(stdout, expected);
    assert_eq!(
        stdout.lines().filter(|line| line.starts_with('…')).count(),
        1,
        "{stdout}"
    );

    let expanded = run(&workspace, &["expand", id]);
    assert_eq!(expanded.status.code(), Some(0));
    assert!(expanded.stderr.is_empty(), "{}", text(&expanded.stderr));
    assert_eq!(expanded.stdout, "hello\n".repeat(249).as_bytes());
    assert_eq!(text(&expanded.stdout).lines().count(), 249);
}

#[test]
fn timeout_kills_descendants_and_marks_partial_unterminated_output() {
    let workspace = fresh_workspace("timeout");
    // Warm the store first, unmeasured: a cold first run pays ~12s of
    // initialization, which would drown the kill-promptness measurement.
    let _ = command(&workspace)
        .args(["bash-smart", "--", "true"])
        .output()
        .expect("warmup bash-smart");
    let started = Instant::now();
    let output = command(&workspace)
        .env("GREPPY_BASH_SMART_TIMEOUT_MS", "75")
        .args(["bash-smart", "--", "sh", "-c", "printf partial; sleep 5"])
        .output()
        .expect("run timed bash-smart");
    let stdout = text(&output.stdout);
    let stderr = text(&output.stderr);

    // The contract: the 75ms timeout preempts the child's 5s sleep.
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "kill took {:?} — the timeout did not preempt the child's sleep",
        started.elapsed()
    );
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
