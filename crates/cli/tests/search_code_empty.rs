//! Explanatory empty-output coverage for `search-pattern`, plus the
//! retirement pin for the pre-0.3.0 name `search-code` whose literal-search
//! contract this file used to pin.
//!
//! 0.3.0 CLI contract (normative):
//! * `search-code` is dead vocabulary: refused as an unknown subcommand
//!   (exit 64) before grep passthrough — never grepped, never answered.
//! * `search-pattern` is regex-native: a pattern with metacharacters is
//!   simply a pattern. Zero hits are a successful bounded status: they name
//!   the empty scope, distinguish a path-filter miss when possible, and give
//!   concrete broader-search and refresh actions.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

fn fresh_workspace(tag: &str) -> (PathBuf, PathBuf) {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let base = std::env::temp_dir().join(format!(
        "greppy-cli-search-empty-{tag}-{}-{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&base);
    let repo = base.join("repo");
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    (repo, base.join("store"))
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

fn assert_indexed(repo: &Path, store: &Path) {
    let (code, stdout, stderr) = run(repo, store, &["index", "."]);
    assert_eq!(
        code, 0,
        "fixture index must complete; stdout={stdout}\nstderr={stderr}"
    );
}

/// The pre-0.3.0 `search-code` subcommand is dead-listed vocabulary: like
/// `edit change-signature` (edit_m4) it must be REFUSED as an unknown
/// subcommand — an agent with a stale habit learns immediately instead of
/// getting garbage grep matches for `search-code` as a pattern.
#[test]
fn retired_search_code_is_refused_not_grepped() {
    let (repo, store) = fresh_workspace("retired");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::write(repo.join("src/lib.rs"), "pub fn present() {}\n").unwrap();

    let (code, stdout, stderr) = run(&repo, &store, &["search-code", "absent_value", "src"]);

    let text = format!("{stdout}{stderr}");
    assert_eq!(
        code, 64,
        "`greppy search-code` must refuse as invalid vocabulary; stdout={stdout}\nstderr={stderr}"
    );
    assert!(
        text.contains("unrecognized subcommand 'search-code'"),
        "the refusal names the dead verb; got: {text}"
    );
    assert!(
        !text.contains("src/lib.rs") && !text.contains("no matches"),
        "the refusal neither greps nor answers; got: {text}"
    );
}

/// Zero hits bind the active path filter to concrete next actions.
#[test]
fn empty_search_pattern_names_the_path_filter_and_next_actions() {
    let (repo, store) = fresh_workspace("empty-filter");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::write(repo.join("src/lib.rs"), "pub fn present() {}\n").unwrap();
    assert_indexed(&repo, &store);

    let (code, stdout, _stderr) = run(
        &repo,
        &store,
        &["search-pattern", "absent_value", "--path", "src"],
    );

    assert_eq!(
        code, 1,
        "a bounded no-match status still uses grep's convention — 0 for a hit, 1 for none. The status block carries the guidance; stdout={stdout}"
    );
    assert!(stdout.contains("status: no_matches"), "{stdout}");
    assert!(
        stdout.contains("message: no matches under path filter: src"),
        "{stdout}"
    );
    assert!(
        stdout.contains("next: retry without the path filter"),
        "{stdout}"
    );
    assert!(!stdout.contains("greppy index ."), "{stdout}");
    assert!(
        stdout.contains("next: search excluded source directly: greppy rg -n absent_value ."),
        "a live no-match must disclose the direct source recovery for vendor/ignored files; got: {stdout}"
    );
}

/// A repository-wide miss must not claim that excluded vendor/ignored files
/// were searched. It names the graph discovery boundary and gives the exact
/// direct-source recovery instead of sending the caller into a futile reindex.
#[test]
fn empty_repository_search_pattern_discloses_discovery_exclusions() {
    let (repo, store) = fresh_workspace("empty-repository-scope");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::write(repo.join("src/lib.rs"), "pub fn present() {}\n").unwrap();
    assert_indexed(&repo, &store);

    let (code, stdout, _stderr) = run(
        &repo,
        &store,
        &["search-pattern", "vendor_only_marker", "--fixed"],
    );

    assert_eq!(code, 1, "stdout={stdout}");
    assert!(
        stdout.contains("scope: live Greppy-discovered source files in the repository"),
        "{stdout}"
    );
    assert!(
        stdout.contains(
            "next: search excluded source directly: greppy rg -n -F vendor_only_marker ."
        ),
        "{stdout}"
    );
}

/// `search-pattern` is regex-native: `absent.*value` is simply a pattern that
/// matches nothing. The pre-0.3.0 teaching ("regex metacharacters are literal
/// in search-code" + "try: greppy rg ...") is dead with the command.
#[test]
fn metacharacter_pattern_is_just_a_pattern_without_teaching() {
    let (repo, store) = fresh_workspace("metacharacters");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::write(repo.join("src/lib.rs"), "pub fn present() {}\n").unwrap();
    assert_indexed(&repo, &store);

    let (code, stdout, _stderr) = run(
        &repo,
        &store,
        &["search-pattern", "absent.*value", "--path", "src"],
    );

    assert_eq!(
        code, 1,
        "a bounded no-match status still uses grep's convention — 0 for a hit, 1 for none. The status block carries the guidance; stdout={stdout}"
    );
    assert!(stdout.contains("status: no_matches"), "{stdout}");
    assert!(
        stdout.contains("message: no matches under path filter: src"),
        "{stdout}"
    );
    assert!(
        stdout.contains("greppy search-pattern 'absent.*value'"),
        "the retry preserves the regex instead of reinterpreting it; got: {stdout:?}"
    );
}

/// A computed case-insensitive count remains attached to the bounded status.
#[test]
fn empty_search_pattern_reports_the_case_insensitive_fact() {
    let (repo, store) = fresh_workspace("case-fact");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::write(repo.join("src/lib.rs"), "pub fn present() {}\n").unwrap();
    assert_indexed(&repo, &store);

    let (code, stdout, _stderr) = run(
        &repo,
        &store,
        &["search-pattern", "PRESENT", "--fixed", "--path", "src"],
    );

    assert_eq!(
        code, 1,
        "case-insensitive suggestions are not primary case-sensitive hits; stdout={stdout}"
    );
    assert!(stdout.contains("status: no_matches"), "{stdout}");
    assert!(
        stdout.contains("case-insensitive: 1 matches"),
        "the empty answer carries the computed case-insensitive fact; got: {stdout:?}"
    );
}

/// The no-match status is terminal output: once it is visible, the command
/// must not keep doing a potentially expensive diagnostic source scan.
#[cfg(unix)]
#[test]
fn empty_search_pattern_emits_status_after_case_insensitive_scan_finishes() {
    use std::io::{BufRead, BufReader};
    use std::os::unix::fs::PermissionsExt;
    use std::process::Stdio;
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    struct ChildGuard {
        child: std::process::Child,
        release: PathBuf,
    }
    impl Drop for ChildGuard {
        fn drop(&mut self) {
            let _ = std::fs::write(&self.release, b"release");
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }

    let (repo, store) = fresh_workspace("terminal-status-order");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::create_dir_all(repo.join("other")).unwrap();
    std::fs::write(repo.join("src/lib.rs"), "pub fn present() {}\n").unwrap();
    std::fs::write(repo.join("other/large.rs"), "pub fn unrelated() {}\n").unwrap();
    assert_indexed(&repo, &store);

    let tools = repo.parent().unwrap().join("tools");
    std::fs::create_dir_all(&tools).unwrap();
    let grep = tools.join("grep");
    std::fs::write(
        &grep,
        "#!/bin/sh\ncase \" $* \" in *\" -i \"*) ;; *) exec /usr/bin/grep \"$@\" ;; esac\nprintf '%s\\n' \"$@\" > \"$GREPPY_TEST_GREP_ARGS\"\n: > \"$GREPPY_TEST_GREP_STARTED\"\nattempt=0; while [ ! -e \"$GREPPY_TEST_GREP_RELEASE\" ]; do attempt=$((attempt + 1)); [ \"$attempt\" -lt 1000 ] || exit 2; /bin/sleep 0.01; done\nexit 1\n",
    )
    .unwrap();
    std::fs::set_permissions(&grep, std::fs::Permissions::from_mode(0o755)).unwrap();
    let started = repo.parent().unwrap().join("grep-started");
    let release = repo.parent().unwrap().join("grep-release");
    let grep_args = repo.parent().unwrap().join("grep-args");
    let path = std::env::join_paths(std::iter::once(tools.clone()).chain(std::env::split_paths(
        &std::env::var_os("PATH").unwrap_or_default(),
    )))
    .unwrap();

    let mut child = ChildGuard {
        child: Command::new(bin())
            .args(["search-pattern", "absent_value", "--path", "src"])
            .current_dir(&repo)
            .env("GREPPY_STORE_DIR", &store)
            .env("GREPPY_TEST_SKIP_INFERENCE", "1")
            .env("GREPPY_TEST_GREP_STARTED", &started)
            .env("GREPPY_TEST_GREP_RELEASE", &release)
            .env("GREPPY_TEST_GREP_ARGS", &grep_args)
            .env("PATH", path)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("run greppy"),
        release: release.clone(),
    };
    let stdout = child.child.stdout.take().unwrap();
    let (line_tx, line_rx) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        let mut line = String::new();
        BufReader::new(stdout).read_line(&mut line).unwrap();
        let _ = line_tx.send(line);
    });

    let deadline = Instant::now() + Duration::from_secs(5);
    while !started.exists() {
        assert!(Instant::now() < deadline, "diagnostic grep did not start");
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(
        line_rx.recv_timeout(Duration::from_millis(200)).is_err(),
        "terminal no-match output became visible before the diagnostic scan finished"
    );
    let diagnostic_args = std::fs::read_to_string(&grep_args).unwrap();
    assert!(diagnostic_args.lines().any(|line| line == "src/lib.rs"));
    assert!(
        !diagnostic_args.lines().any(|line| line == "other/large.rs"),
        "case-insensitive diagnostics must apply --path before scanning; args={diagnostic_args:?}"
    );

    std::fs::write(&release, b"release").unwrap();
    let status = child.child.wait().unwrap();
    assert_eq!(status.code(), Some(1));
    let first_line = line_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    assert_eq!(first_line.trim(), "status: no_matches");
    reader.join().unwrap();
}
