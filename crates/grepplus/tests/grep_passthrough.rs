//! Integration tests for the grep passthrough.
//!
//! These tests run `grepplus-grep` as a subprocess and compare its
//! stdout/stderr/exit-code byte-for-byte against the same command run
//! with the real `grep` binary on `PATH` (or `GREPPLUS_REAL_GREP`).

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn binary_path() -> PathBuf {
    // CARGO_BIN_EXE_grepplus-grep is set by cargo for integration tests.
    PathBuf::from(env!("CARGO_BIN_EXE_grepplus-grep"))
}

fn real_grep_path() -> PathBuf {
    if let Ok(p) = std::env::var("GREPPLUS_REAL_GREP") {
        return PathBuf::from(p);
    }
    PathBuf::from("/usr/bin/grep")
}

fn unique_tempdir(tag: &str) -> PathBuf {
    let safe_tag: String = tag
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    let p = std::env::temp_dir().join(format!(
        "grepplus-passthrough-{safe_tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn grepplus_command(label: &str) -> Command {
    let mut cmd = Command::new(binary_path());
    cmd.env("GREPPLUS_STORE_DIR", unique_tempdir(label));
    cmd
}

fn run_with_stdin(cmd: &mut Command, stdin_bytes: &[u8]) -> std::process::Output {
    cmd.stdin(Stdio::piped());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    let mut child = cmd.spawn().expect("spawn");
    if let Some(mut sin) = child.stdin.take() {
        sin.write_all(stdin_bytes).expect("write stdin");
    }
    child.wait_with_output().expect("wait")
}

fn diff_outputs(label: &str, ours: &std::process::Output, theirs: &std::process::Output) {
    assert_eq!(
        ours.stdout,
        theirs.stdout,
        "[{label}] stdout differs:\nours:\n{:?}\ntheirs:\n{:?}",
        String::from_utf8_lossy(&ours.stdout),
        String::from_utf8_lossy(&theirs.stdout)
    );
    assert_eq!(
        ours.stderr,
        theirs.stderr,
        "[{label}] stderr differs:\nours:\n{:?}\ntheirs:\n{:?}",
        String::from_utf8_lossy(&ours.stderr),
        String::from_utf8_lossy(&theirs.stderr)
    );
    assert_eq!(
        ours.status.code(),
        theirs.status.code(),
        "[{label}] exit code differs: ours={:?} theirs={:?}",
        ours.status.code(),
        theirs.status.code()
    );
}

fn assert_passthrough(label: &str, args: &[&str]) {
    let mut ours = grepplus_command(label);
    ours.args(args);
    let mut theirs = Command::new(real_grep_path());
    theirs.args(args);
    diff_outputs(label, &ours.output().unwrap(), &theirs.output().unwrap());
}

#[test]
fn passthrough_basic_recursive_search() {
    let mut ours = grepplus_command("basic_recursive");
    ours.args(["-R", "fn main", "tests/fixtures"]);
    let mut theirs = Command::new(real_grep_path());
    theirs.args(["-R", "fn main", "tests/fixtures"]);
    let o = ours.output().unwrap();
    let t = theirs.output().unwrap();
    diff_outputs("basic_recursive", &o, &t);
}

#[test]
fn passthrough_quiet_no_match_exits_one() {
    // `-q` on a single existing file with no matches: real grep exits 1
    // and writes nothing. (Using a directory here would cause real grep
    // to error out with exit 2, which is a different and also-tested
    // code path below.)
    let mut ours = grepplus_command("quiet_no_match");
    ours.args([
        "-q",
        "this_string_should_never_appear_anywhere",
        "tests/fixtures/count.txt",
    ]);
    let mut theirs = Command::new(real_grep_path());
    theirs.args([
        "-q",
        "this_string_should_never_appear_anywhere",
        "tests/fixtures/count.txt",
    ]);
    let o = ours.output().unwrap();
    let t = theirs.output().unwrap();
    diff_outputs("quiet_no_match", &o, &t);
    assert_eq!(o.status.code(), Some(1));
}

#[test]
fn passthrough_count_mode() {
    let mut ours = grepplus_command("count");
    ours.args(["-c", "alpha", "tests/fixtures/count.txt"]);
    let mut theirs = Command::new(real_grep_path());
    theirs.args(["-c", "alpha", "tests/fixtures/count.txt"]);
    diff_outputs("count", &ours.output().unwrap(), &theirs.output().unwrap());
}

#[test]
fn passthrough_files_with_matches() {
    let mut ours = grepplus_command("files_with_matches");
    ours.args(["-Rl", "alpha", "tests/fixtures"]);
    let mut theirs = Command::new(real_grep_path());
    theirs.args(["-Rl", "alpha", "tests/fixtures"]);
    diff_outputs(
        "files_with_matches",
        &ours.output().unwrap(),
        &theirs.output().unwrap(),
    );
}

#[test]
fn passthrough_invert_match() {
    let mut ours = grepplus_command("invert");
    ours.args(["-v", "alpha", "tests/fixtures/count.txt"]);
    let mut theirs = Command::new(real_grep_path());
    theirs.args(["-v", "alpha", "tests/fixtures/count.txt"]);
    diff_outputs("invert", &ours.output().unwrap(), &theirs.output().unwrap());
}

#[test]
fn passthrough_extended_regex() {
    let mut ours = grepplus_command("extended_regex");
    ours.args(["-E", "alpha|beta", "tests/fixtures/count.txt"]);
    let mut theirs = Command::new(real_grep_path());
    theirs.args(["-E", "alpha|beta", "tests/fixtures/count.txt"]);
    diff_outputs(
        "extended_regex",
        &ours.output().unwrap(),
        &theirs.output().unwrap(),
    );
}

#[test]
fn passthrough_stdin_pipe() {
    let input = b"alpha\nbeta\ngamma\nalpha\n";
    let mut ours = grepplus_command("stdin");
    ours.arg("alpha");
    let o = run_with_stdin(&mut ours, input);
    let mut theirs = Command::new(real_grep_path());
    theirs.arg("alpha");
    let t = run_with_stdin(&mut theirs, input);
    diff_outputs("stdin", &o, &t);
}

#[test]
fn passthrough_missing_file_returns_grep_style_error() {
    let mut ours = grepplus_command("missing_file");
    ours.args(["alpha", "tests/fixtures/does_not_exist.txt"]);
    let mut theirs = Command::new(real_grep_path());
    theirs.args(["alpha", "tests/fixtures/does_not_exist.txt"]);
    let o = ours.output().unwrap();
    let t = theirs.output().unwrap();
    // Real grep on missing file: exit 2, stderr contains the path.
    // Our wrapper must produce byte-identical stderr and exit code.
    assert_eq!(o.status.code(), t.status.code(), "exit codes differ");
    assert_eq!(o.stderr, t.stderr, "stderr differs");
}

#[test]
fn passthrough_r2_common_flag_matrix_matches_real_grep() {
    let cases: &[(&str, &[&str])] = &[
        (
            "fixed_strings",
            &["-F", "alpha", "tests/fixtures/count.txt"],
        ),
        ("ignore_case", &["-i", "ALPHA", "tests/fixtures/count.txt"]),
        ("word_regexp", &["-w", "alpha", "tests/fixtures/count.txt"]),
        (
            "line_number_with_filename",
            &["-nH", "alpha", "tests/fixtures/count.txt"],
        ),
        (
            "no_filename_multi_file",
            &[
                "-h",
                "alpha",
                "tests/fixtures/count.txt",
                "tests/fixtures/extra.txt",
            ],
        ),
        (
            "only_matching",
            &["-o", "alpha", "tests/fixtures/count.txt"],
        ),
        ("files_without_match", &["-L", "alpha", "tests/fixtures"]),
        (
            "include_recursive",
            &["--include=*.txt", "-R", "alpha", "tests/fixtures"],
        ),
        (
            "exclude_recursive",
            &["--exclude=extra.txt", "-R", "alpha", "tests/fixtures"],
        ),
        (
            "exclude_dir_recursive",
            &["--exclude-dir=target", "-R", "alpha", "tests/fixtures"],
        ),
    ];
    for (label, args) in cases {
        assert_passthrough(label, args);
    }
}

// P0 (R-014 re-review): the wrapper previously collected argv via
// `std::env::args()`, which UNWRAPS and PANICS (rc 101) on a non-UTF-8
// argument — `grepplus-grep $'\xff' f.txt </dev/null` died with rc 101
// while real grep returns 1/2 cleanly. The fix routes argv through
// `args_os` and forwards the original `OsString`s to real grep verbatim.
// Here we drive both the wrapper and real grep with a non-UTF-8 PATTERN
// and a non-UTF-8 PATH and require byte-identical stdout/stderr/rc — and
// crucially that the wrapper never panics (rc != 101).
#[cfg(unix)]
#[test]
fn passthrough_non_utf8_pattern_and_path_match_real_grep() {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;

    if !real_grep_path().is_file() {
        return; // no system grep in this sandbox
    }

    let pattern = OsString::from_vec(vec![0xff]);
    let path = OsString::from_vec(vec![b'f', 0xff, b'.', b't', b'x', b't']);

    let mut ours = grepplus_command("non_utf8");
    ours.arg(&pattern).arg(&path).stdin(Stdio::null());
    let mut theirs = Command::new(real_grep_path());
    theirs.arg(&pattern).arg(&path).stdin(Stdio::null());

    let o = ours.output().expect("spawn grepplus-grep");
    let t = theirs.output().expect("spawn real grep");

    assert_ne!(
        o.status.code(),
        Some(101),
        "wrapper must NOT panic (rc 101) on non-UTF-8 argv; stderr={}",
        String::from_utf8_lossy(&o.stderr)
    );
    // Byte-for-byte parity with real grep: same rc, same stdout, same
    // stderr (grep echoes the raw non-UTF-8 path back in its error).
    assert_eq!(
        o.status.code(),
        t.status.code(),
        "exit code differs: ours={:?} theirs={:?}",
        o.status.code(),
        t.status.code()
    );
    assert_eq!(o.stdout, t.stdout, "stdout differs on non-UTF-8 argv");
    assert_eq!(o.stderr, t.stderr, "stderr differs on non-UTF-8 argv");
}
