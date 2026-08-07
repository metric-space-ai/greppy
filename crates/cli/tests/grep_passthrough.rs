//! Integration tests for the grep passthrough.
//!
//! These tests run the shipped `greppy` binary as a subprocess and compare its
//! stdout/stderr/exit-code byte-for-byte against the same command run
//! with the real `grep` binary on `PATH` (or `GREPPY_REAL_GREP`).

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn binary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_greppy"))
}

fn real_grep_path() -> PathBuf {
    if let Ok(p) = std::env::var("GREPPY_REAL_GREP") {
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
        "greppy-passthrough-{safe_tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn greppy_command(label: &str) -> Command {
    let mut cmd = Command::new(binary_path());
    cmd.env("GREPPY_STORE_DIR", unique_tempdir(label));
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
    let mut ours = greppy_command(label);
    ours.args(args);
    let mut theirs = Command::new(real_grep_path());
    theirs.args(args);
    diff_outputs(label, &ours.output().unwrap(), &theirs.output().unwrap());
}

#[test]
fn passthrough_basic_recursive_search() {
    let mut ours = greppy_command("basic_recursive");
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
    let mut ours = greppy_command("quiet_no_match");
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
    let mut ours = greppy_command("count");
    ours.args(["-c", "alpha", "tests/fixtures/count.txt"]);
    let mut theirs = Command::new(real_grep_path());
    theirs.args(["-c", "alpha", "tests/fixtures/count.txt"]);
    diff_outputs("count", &ours.output().unwrap(), &theirs.output().unwrap());
}

#[test]
fn passthrough_files_with_matches() {
    let mut ours = greppy_command("files_with_matches");
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
    let mut ours = greppy_command("invert");
    ours.args(["-v", "alpha", "tests/fixtures/count.txt"]);
    let mut theirs = Command::new(real_grep_path());
    theirs.args(["-v", "alpha", "tests/fixtures/count.txt"]);
    diff_outputs("invert", &ours.output().unwrap(), &theirs.output().unwrap());
}

#[test]
fn passthrough_extended_regex() {
    let mut ours = greppy_command("extended_regex");
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
    let mut ours = greppy_command("stdin");
    ours.arg("alpha");
    let o = run_with_stdin(&mut ours, input);
    let mut theirs = Command::new(real_grep_path());
    theirs.arg("alpha");
    let t = run_with_stdin(&mut theirs, input);
    diff_outputs("stdin", &o, &t);
}

#[test]
fn passthrough_delayed_stdin_data_is_still_forwarded_byte_exactly() {
    let mut ours = greppy_command("delayed-stdin");
    ours.arg("hallo")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = ours.spawn().expect("spawn greppy");
    let mut stdin = child.stdin.take().expect("open child stdin");
    let writer = std::thread::spawn(move || {
        // The producer grace exists because pipeline processes are scheduled
        // independently; data can be genuine even when it is not buffered yet.
        std::thread::sleep(std::time::Duration::from_millis(50));
        stdin.write_all(b"hallo\n").expect("write delayed stdin");
    });
    let output = child.wait_with_output().expect("collect greppy output");
    writer.join().expect("join delayed writer");
    assert_eq!(output.status.code(), Some(0));
    assert_eq!(output.stdout, b"hallo\n");
    assert!(output.stderr.is_empty());
}

#[test]
fn passthrough_idle_stdin_pipe_returns_guidance_instead_of_hanging() {
    let root = unique_tempdir("idle-stdin");
    std::fs::create_dir(root.join("edit-src")).unwrap();

    for pattern in [".", "edit-src"] {
        let mut command = greppy_command(&format!("idle-{pattern}"));
        command
            .arg(pattern)
            .current_dir(&root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = command.spawn().expect("spawn greppy");
        // Keeping the writer open with zero bytes reproduces agent runners:
        // real grep waits forever because EOF never arrives.
        let stdin = child.stdin.take().expect("open child stdin");
        let (send, receive) = std::sync::mpsc::channel();
        std::thread::spawn(move || send.send(child.wait_with_output()).unwrap());
        let output = match receive.recv_timeout(std::time::Duration::from_secs(15)) {
            Ok(output) => output.expect("collect greppy output"),
            Err(_) => {
                drop(stdin);
                let _ = receive.recv_timeout(std::time::Duration::from_secs(1));
                panic!("greppy {pattern} waited indefinitely on an idle stdin pipe");
            }
        };
        drop(stdin);
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert_eq!(output.status.code(), Some(64), "stdout: {stdout}");
        assert!(
            stdout.contains("file/path argument or data on stdin"),
            "{stdout}"
        );
        assert!(
            stdout.contains(&format!("greppy index {pattern}")),
            "directory guidance missing: {stdout}"
        );
    }
}

#[test]
fn passthrough_file_operand_does_not_consult_idle_stdin() {
    let mut command = greppy_command("file-with-idle-stdin");
    command
        .args(["alpha", "tests/fixtures/count.txt"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn().expect("spawn greppy");
    let stdin = child.stdin.take().expect("open child stdin");
    let (send, receive) = std::sync::mpsc::channel();
    std::thread::spawn(move || send.send(child.wait_with_output()).unwrap());
    let output = receive
        .recv_timeout(std::time::Duration::from_secs(15))
        .expect("a named file must let grep finish while stdin remains open")
        .expect("collect greppy output");
    drop(stdin);
    assert_eq!(output.status.code(), Some(0));
    assert_eq!(output.stdout, b"alpha\nalpha\nalpha\n");
}

#[test]
fn passthrough_missing_file_returns_grep_style_error() {
    let mut ours = greppy_command("missing_file");
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

// The wrapper previously collected argv via
// `std::env::args()`, which UNWRAPS and PANICS (rc 101) on a non-UTF-8
// argument — `greppy $'\xff' f.txt </dev/null` died with rc 101
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

    let mut ours = greppy_command("non_utf8");
    ours.arg(&pattern).arg(&path).stdin(Stdio::null());
    let mut theirs = Command::new(real_grep_path());
    theirs.arg(&pattern).arg(&path).stdin(Stdio::null());

    let o = ours.output().expect("spawn greppy");
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
