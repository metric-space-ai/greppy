//! End-to-end coverage for the training-free bash-smart delivery contract.
//!
//! These tests require the feature that provides the verb. They exercise the
//! actual CLI, including subprocesses, signal forwarding and raw-log recovery;
//! fixture inference is disabled because the delivery contract is mechanical.

#![cfg(all(unix, feature = "bash-smart"))]

use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
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
fn oversized_single_line_keeps_failure_and_exact_raw_log_recovery() {
    for stream in ["stdout", "stderr"] {
        let workspace = fresh_workspace(&format!("long-line-{stream}"));
        let script = format!(
            "{{ printf 'data:text/javascript;base64,'; head -c 160000 /dev/zero | tr '\\000' A; printf ':1:7\\nError: long-line-probe\\n'; }} {}; exit 1",
            if stream == "stderr" { ">&2" } else { "" },
        );
        let output = run(&workspace, &["bash-smart", "--", "sh", "-c", &script]);
        assert_eq!(output.status.code(), Some(1));
        assert!(
            output.stdout.len() + output.stderr.len() < 12_000,
            "one long line bypassed the preview bound: stdout={} stderr={}",
            output.stdout.len(),
            output.stderr.len()
        );
        let rendered = format!("{}{}", text(&output.stdout), text(&output.stderr));
        assert!(rendered.contains("Error: long-line-probe"));
        assert!(rendered.contains("bytes omitted; full line in raw log"));
        let path_json = rendered
            .split("raw log ")
            .nth(1)
            .unwrap()
            .split("; read with greppy read-file")
            .next()
            .unwrap();
        let path: String = serde_json::from_str(path_json).unwrap();
        let expected = format!(
            "data:text/javascript;base64,{}:1:7\nError: long-line-probe\n",
            "A".repeat(160_000)
        );
        assert_eq!(std::fs::read(path).unwrap(), expected.as_bytes());
    }
}

#[test]
fn short_output_follows_verdict_and_exit_code_passes_through() {
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
    assert_eq!(
        output.stdout,
        b"FAILED \xe2\x80\x94 exit 3: 0 errors, 0 warnings\nout\n"
    );
    assert_eq!(output.stderr, b"err\n");
}

#[test]
fn short_regex_matches_are_printed_once_with_original_stream_bytes() {
    let workspace = fresh_workspace("short-regex-once");
    for count in [1, 10] {
        for redirect in ["", " >&2"] {
            let script = format!(
                "i=0; while [ \"$i\" -lt {count} ]; do printf '{{\"trial\":\"x\"}}\\r\\n'{redirect}; i=$((i+1)); done"
            );
            let output = run(
                &workspace,
                &["bash-smart", "-e", "\"trial\"", "--", "sh", "-c", &script],
            );
            assert_eq!(output.status.code(), Some(0));
            let expected = "{\"trial\":\"x\"}\r\n".repeat(count);
            let mut stdout = b"ok \xe2\x80\x94 exit 0\n".to_vec();
            if redirect.is_empty() {
                stdout.extend_from_slice(expected.as_bytes());
                assert!(output.stderr.is_empty());
            } else {
                assert_eq!(output.stderr, expected.as_bytes());
            }
            assert_eq!(output.stdout, stdout);
        }
    }
}

#[test]
fn silent_long_running_child_emits_bounded_liveness_heartbeats() {
    let workspace = fresh_workspace("heartbeat");
    let output = command(&workspace)
        .env("GREPPY_BASH_SMART_HEARTBEAT_MS", "25")
        .args([
            "bash-smart",
            "--",
            "sh",
            "-c",
            "printf 'Blocking waiting for file lock on package cache\\n' >&2; sleep 0.12",
        ])
        .output()
        .expect("run greppy heartbeat fixture");
    let stderr = text(&output.stderr);

    assert_eq!(output.status.code(), Some(0));
    assert_eq!(output.stdout, b"ok \xe2\x80\x94 exit 0\n");
    assert!(
        stderr.contains("bash-smart: command still running")
            && stderr.contains("pid=")
            && stderr
                .contains("latest child output: Blocking waiting for file lock on package cache"),
        "stderr={stderr:?}"
    );
}

#[test]
fn typescript_diagnostic_counts_one_error_and_preserves_exit_and_bytes() {
    let workspace = fresh_workspace("typescript-diagnostic");
    for redirect in ["", " >&2"] {
        let script = format!(
            "printf '%s\\n' 'example.ts(1,1): error TS2322: Type string is not assignable to type number.'{redirect}; exit 1"
        );
        let output = run(&workspace, &["bash-smart", "--", "sh", "-c", &script]);
        assert_eq!(output.status.code(), Some(1));
        let stdout = text(&output.stdout);
        assert!(
            stdout.starts_with("FAILED — exit 1: 1 error, 0 warnings\n"),
            "stdout={stdout}; stderr={}",
            text(&output.stderr)
        );
        let diagnostic =
            "example.ts(1,1): error TS2322: Type string is not assignable to type number.";
        let raw_stream = if redirect.is_empty() {
            &output.stdout
        } else {
            &output.stderr
        };
        assert!(text(raw_stream).lines().any(|line| line == diagnostic));
    }
}

#[test]
fn compiler_diagnostic_counts_preserve_child_exit_and_raw_bytes() {
    let workspace = fresh_workspace("compiler-diagnostic");
    let diagnostics = "/example/header.h:41:8: error: #error \"incompatible headers\"\n/example/main.c:9: warning: unused variable\n";
    for redirect in ["", " >&2"] {
        let script = format!("printf '%s' '{diagnostics}'{redirect}; exit 2");
        let output = run(&workspace, &["bash-smart", "--", "sh", "-c", &script]);
        assert_eq!(output.status.code(), Some(2));
        let stdout = text(&output.stdout);
        assert!(
            stdout.starts_with("FAILED — exit 2: 1 error, 1 warning\n"),
            "stdout={stdout}; stderr={}",
            text(&output.stderr)
        );
        let raw_stream = if redirect.is_empty() {
            &output.stdout
        } else {
            &output.stderr
        };
        assert!(raw_stream.ends_with(diagnostics.as_bytes()));
        let combined = format!("{}{}", text(&output.stdout), text(&output.stderr));
        assert_eq!(
            combined.matches("#error \"incompatible headers\"").count(),
            1
        );
        assert_eq!(combined.matches("warning: unused variable").count(), 1);
    }
}

#[test]
fn child_flags_after_delimiter_pass_through_unchanged() {
    let workspace = fresh_workspace("child-flag");
    let output = run(
        &workspace,
        &[
            "bash-smart",
            "--",
            "sh",
            "-c",
            "printf '%s\\n' \"$1\"",
            "child",
            "--target",
        ],
    );

    assert_eq!(output.status.code(), Some(0));
    assert_eq!(output.stdout, b"ok \xe2\x80\x94 exit 0\n--target\n");
    assert!(output.stderr.is_empty());
}

#[test]
fn leading_environment_assignments_are_applied_to_the_child() {
    let workspace = fresh_workspace("leading-assignment");
    let output = run(
        &workspace,
        &[
            "bash-smart",
            "--",
            "GREPPY_BASH_SMART_ASSIGNMENT=works",
            "sh",
            "-c",
            "printf '%s\\n' \"$GREPPY_BASH_SMART_ASSIGNMENT\"",
        ],
    );

    assert_eq!(output.status.code(), Some(0));
    assert_eq!(output.stdout, b"ok \xe2\x80\x94 exit 0\nworks\n");
    assert!(output.stderr.is_empty());
}

#[test]
fn unquoted_shell_syntax_is_rejected_with_actionable_guidance() {
    let workspace = fresh_workspace("unquoted-shell");
    let output = run(
        &workspace,
        &["bash-smart", "--", "cd", "repo", "&&", "printf", "ok"],
    );
    let stderr = text(&output.stderr);

    assert_eq!(output.status.code(), Some(64));
    assert!(
        stderr.contains("received unquoted shell syntax"),
        "stderr={stderr}"
    );
    assert!(
        stderr.contains("complete shell expression"),
        "stderr={stderr}"
    );
}

#[test]
fn quoted_pipeline_uses_pipefail_exit_status() {
    let workspace = fresh_workspace("pipeline-pipefail");
    let output = run(
        &workspace,
        &["bash-smart", "--", "sh -c 'exit 9' | tail -n 1"],
    );

    assert_eq!(output.status.code(), Some(9));
    assert!(
        output.stdout.starts_with(b"FAILED \xe2\x80\x94 exit 9"),
        "stdout={}",
        text(&output.stdout)
    );
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
    let expected = std::iter::once("ok — exit 0\n".to_string())
        .chain((1..=21).map(|line| format!("line {line}\n")))
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
        "ok — exit 0\n{}… lines 22-270 (249 collapsed `hello` repeats) — greppy expand {id}\n{}",
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
fn signal_forwards_to_child_group_and_keeps_expandable_partial_output() {
    let workspace = fresh_workspace("signal");
    let _ = command(&workspace)
        .args(["bash-smart", "--", "true"])
        .output()
        .expect("warmup bash-smart");

    let child_pid_path = workspace.base.join("child.pid");
    // Creation of a redirected file precedes its write. Publish the complete
    // PID by same-directory rename so exists() cannot expose an empty receipt.
    // Pass the path as argv, including when TMPDIR contains spaces.
    let script = "set -eu; printf '%s\\n' \"$$\" > \"$1.tmp\"; mv \"$1.tmp\" \"$1\"; \
        i=1; while [ $i -le 100000 ]; do echo line $i; i=$((i + 1)); sleep 0.01; done";
    let started = Instant::now();
    let greppy = command(&workspace)
        .args(["bash-smart", "--", "sh", "-c", script, "greppy-signal-test"])
        .arg(&child_pid_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn signal bash-smart");

    let deadline = Instant::now() + Duration::from_secs(30);
    while !child_pid_path.exists() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    let child_pid = std::fs::read_to_string(&child_pid_path)
        .expect("child pid file")
        .trim()
        .parse::<i32>()
        .expect("numeric child pid");
    std::thread::sleep(Duration::from_millis(150));
    assert_eq!(unsafe { libc::kill(greppy.id() as i32, libc::SIGINT) }, 0);

    let output = greppy
        .wait_with_output()
        .expect("wait for signal bash-smart");
    let stdout = text(&output.stdout);
    let stderr = text(&output.stderr);
    assert!(started.elapsed() < Duration::from_secs(2), "{stderr}");
    assert_eq!(output.status.code(), Some(130), "stderr={stderr}");
    assert_eq!(
        stdout.lines().next(),
        Some("FAILED — exit 130: 0 errors, 0 warnings (SIGINT)")
    );
    assert!(stdout.contains("line 1\n"), "{stdout}");
    let id = expand_id(&stdout);
    assert!(
        stderr.contains("bash-smart: interrupted by signal 2; partial output stored as"),
        "{stderr}"
    );

    let expanded = run(&workspace, &["expand", id]);
    assert_eq!(expanded.status.code(), Some(0));
    assert!(text(&expanded.stdout).contains("line "));

    let deadline = Instant::now() + Duration::from_secs(1);
    while unsafe { libc::kill(-child_pid, 0) } == 0 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(
        unsafe { libc::kill(-child_pid, 0) },
        -1,
        "child process group {child_pid} survived"
    );
    assert_eq!(
        std::io::Error::last_os_error().raw_os_error(),
        Some(libc::ESRCH)
    );
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
        .env("GREPPY_BASH_SMART_TIMEOUT_MS", "5000")
        .args(["bash-smart", "--", "sh", "-c", "printf partial; sleep 30"])
        .output()
        .expect("run timed bash-smart");
    let stdout = text(&output.stdout);
    let stderr = text(&output.stderr);

    // The contract: the 5s timeout preempts the child's 30s sleep after the
    // shell has had enough time to publish its unterminated partial output. The
    // budget covers process startup and store open too. The embedded-asset
    // binary can take many seconds to page in on a loaded host; 25s still
    // proves preemption because an un-killed child sleeps for 30s after that
    // same startup cost.
    assert!(
        started.elapsed() < Duration::from_secs(25),
        "kill took {:?} — the timeout did not preempt the child's sleep",
        started.elapsed()
    );
    assert_eq!(output.status.code(), Some(137), "stderr={stderr}");
    assert!(
        stdout.starts_with("FAILED — exit 137: 0 errors, 0 warnings (timeout)\npartial\n"),
        "{stdout}"
    );
    assert!(stdout.contains("greppy expand "), "{stdout}");
    assert!(
        stderr.contains("bash-smart: partial output ends with an unterminated line\n"),
        "{stderr}"
    );
    assert!(
        stderr.contains("bash-smart: timed out after 5000 ms;"),
        "{stderr}"
    );
}

#[test]
fn active_index_writer_never_blocks_command_execution() {
    let workspace = fresh_workspace("writer-independent");
    std::fs::write(workspace.repo.join("lib.rs"), "pub fn marker() {}\n").unwrap();
    let ready = workspace.base.join("index-writer-ready");
    let mut index = command(&workspace)
        .env("GREPPY_TEST_INDEX_FAILPOINT", "after-temp-before-publish")
        .env("GREPPY_TEST_INDEX_FAILPOINT_READY", &ready)
        .env("GREPPY_TEST_INDEX_FAILPOINT_HOLD_MS", "120000")
        .args(["index", "."])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn held index writer");
    let deadline = Instant::now() + Duration::from_secs(60);
    while !ready.exists() {
        if let Some(status) = index.try_wait().expect("poll held index writer") {
            panic!("index writer exited before its hold point: {status}");
        }
        assert!(Instant::now() < deadline, "index writer never became ready");
        std::thread::sleep(Duration::from_millis(25));
    }

    let started = Instant::now();
    let output = run(
        &workspace,
        &["bash-smart", "--", "sh", "-c", "printf ran > command-ran"],
    );
    let elapsed = started.elapsed();
    let long_output = run(
        &workspace,
        &[
            "bash-smart", "--", "sh", "-c",
            "i=0; while [ $i -lt 500 ]; do printf 'test case_%s ... ok\\n' \"$i\"; printf 'detail case_%s\\n' \"$i\" >&2; i=$((i+1)); done",
        ],
    );
    let _ = index.kill();
    let _ = index.wait();

    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr={}",
        text(&output.stderr)
    );
    assert!(
        elapsed < Duration::from_secs(5),
        "bash-smart waited on the graph writer for {elapsed:?}"
    );
    assert_eq!(
        std::fs::read(workspace.repo.join("command-ran")).unwrap(),
        b"ran"
    );
    assert!(
        text(&output.stderr)
            .contains("index writer active; command execution continues without expansion storage"),
        "stderr={}",
        text(&output.stderr)
    );
    assert_eq!(long_output.status.code(), Some(0), "{long_output:?}");
    for (bytes, expected) in [
        (
            &long_output.stdout,
            (0..500)
                .map(|i| format!("test case_{i} ... ok\n"))
                .collect::<String>(),
        ),
        (
            &long_output.stderr,
            (0..500)
                .map(|i| format!("detail case_{i}\n"))
                .collect::<String>(),
        ),
    ] {
        let rendered = text(bytes);
        assert!(
            rendered.lines().count() < 100,
            "uncompressed output: {rendered}"
        );
        assert!(
            !rendered.contains("greppy expand "),
            "invented pack ID: {rendered}"
        );
        let path_json = rendered
            .split("raw log ")
            .nth(1)
            .and_then(|s| s.split("; read with greppy read-file").next())
            .unwrap_or_else(|| panic!("missing raw-log recovery: {rendered}"));
        let path: String = serde_json::from_str(path_json).expect("quoted spool path");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), expected);
    }
}
