//! `grepplus-grep` — the drop-in `grep` wrapper binary.
//!
//! Phase 1 implements only the safe baseline: it discovers the real `grep`
//! binary, forwards argv verbatim, forwards stdin, and forwards stdout,
//! stderr, and exit code. No heuristic augmentation is applied yet;
//! that lands in Phase 6.
//!
//! The binary is invoked as `grepplus-grep` (or via a `grepplus grep`
//! subcommand in the unified `grepplus` CLI in `crates/cli`).

pub mod heuristic;
pub mod run;
pub mod sidecar;

use std::ffi::OsString;
use std::io::Read;
use std::process::{Command, Stdio};

use grepplus_core::error::{Error, Result};

/// Discover the real `grep` binary.
///
/// Discovery order (R-006 / WP-R006):
/// 1. `GREPPLUS_REAL_GREP` env override (rejected if not an executable file).
/// 2. Platform-known system paths: `/usr/bin/grep`, `/bin/grep`
///    (and `/usr/local/bin/grep` for completeness on macOS via Homebrew).
///    We try these explicitly so a shimmed PATH that exposes
///    `~/.grepplus/shims/grep` cannot recurse into the wrapper itself.
/// 3. `which::which("grep")`, with the current executable and
///    `~/.grepplus/shims/` excluded from consideration.
///
/// Returns an error with exit code 3 if no real grep can be found — the
/// caller is expected to surface that to the user with a clear message
/// ("could not locate real grep; set GREPPLUS_REAL_GREP=/path/to/grep"),
/// not to silently fall back to the shim.
pub fn discover_grep() -> Result<std::path::PathBuf> {
    if let Ok(p) = std::env::var("GREPPLUS_REAL_GREP") {
        let path = std::path::PathBuf::from(p);
        if path.is_file() {
            return Ok(path);
        }
        return Err(Error::Config(format!(
            "GREPPLUS_REAL_GREP={} is not an executable file",
            path.display()
        )));
    }

    // Tier 2: try well-known system paths first.
    for candidate in [
        "/usr/bin/grep",
        "/bin/grep",
        "/usr/local/bin/grep",
        "/opt/homebrew/opt/grep/bin/grep",
    ] {
        let p = std::path::PathBuf::from(candidate);
        if p.is_file() {
            return Ok(p);
        }
    }

    // Tier 3: ask `which`. Exclude the current exe path and any
    // `~/.grepplus/shims/` candidates so a shimmed PATH cannot
    // discover the wrapper as "real grep".
    let own_exe = std::env::current_exe().ok();
    let shim_dir = std::env::var_os("HOME")
        .map(|h| std::path::PathBuf::from(h).join(".grepplus").join("shims"));
    let which_result = which::which("grep").map_err(|e| {
        Error::io(
            "locate real grep binary on PATH",
            std::io::Error::new(std::io::ErrorKind::NotFound, e.to_string()),
        )
    })?;
    if own_exe.as_ref().is_some_and(|own| own == &which_result) {
        return Err(Error::Config(format!(
            "refusing to recurse: which('grep') resolved to current exe ({})",
            which_result.display()
        )));
    }
    if let Some(ref sd) = shim_dir {
        if which_result.starts_with(sd) {
            return Err(Error::Config(format!(
                "refusing shim recursion: which('grep') returned {} (under {})",
                which_result.display(),
                sd.display()
            )));
        }
    }
    Ok(which_result)
}

/// Run real grep with the given argv, forwarding stdin/stdout/stderr and
/// preserving exit code.
///
/// `argv` is the full argv (including argv[0]). The real `grep` is invoked
/// via `Command::new(real_grep).args(&argv[1..])` so its own argv[0] is
/// whatever its filesystem entry says; that is identical to what the
/// kernel would do for a direct exec.
pub fn run_grep(real_grep: &std::path::Path, argv: &[String]) -> Result<i32> {
    if argv.is_empty() {
        return Err(Error::Invalid("argv must not be empty".into()));
    }

    let mut cmd = Command::new(real_grep);
    if argv.len() > 1 {
        cmd.args(&argv[1..]);
    }
    cmd.stdin(Stdio::inherit());
    cmd.stdout(Stdio::inherit());
    cmd.stderr(Stdio::inherit());

    let status = cmd
        .status()
        .map_err(|e| Error::io(format!("spawn {}", real_grep.display()), e))?;

    Ok(status.code().unwrap_or(-1))
}

/// Run real grep with the given `OsString` argv, forwarding argv
/// byte-for-byte.
///
/// P0 (R-014 re-review): the drop-in wrapper must NEVER panic on argv it
/// cannot UTF-8-decode. Real `grep` accepts arbitrary bytes for the
/// pattern and for path arguments (e.g. a filename containing `0xff`),
/// and returns its own exit code; the wrapper must do the same. We
/// therefore forward the original `OsString` argv to `Command` verbatim
/// (`Command::arg` takes `AsRef<OsStr>`), without ever going through a
/// lossy `String` conversion for the bytes that reach real grep.
///
/// `argv` is the full argv (including argv[0]). Only `argv[1..]` is
/// forwarded, matching [`run_grep`].
pub fn run_grep_os(real_grep: &std::path::Path, argv: &[OsString]) -> Result<i32> {
    if argv.is_empty() {
        return Err(Error::Invalid("argv must not be empty".into()));
    }

    let mut cmd = Command::new(real_grep);
    if argv.len() > 1 {
        cmd.args(&argv[1..]);
    }
    cmd.stdin(Stdio::inherit());
    cmd.stdout(Stdio::inherit());
    cmd.stderr(Stdio::inherit());

    let status = cmd
        .status()
        .map_err(|e| Error::io(format!("spawn {}", real_grep.display()), e))?;

    Ok(status.code().unwrap_or(-1))
}

/// Read all of stdin into a `Vec<u8>` if stdin is not a TTY.
///
/// Used by integration tests that want to feed grepplus without inheriting
/// the parent's tty. Production callers (the binary entry point) use
/// `Stdio::inherit()` directly.
pub fn read_stdin_if_piped() -> Option<Vec<u8>> {
    use std::io::IsTerminal;
    if std::io::stdin().is_terminal() {
        None
    } else {
        let mut buf = Vec::new();
        std::io::stdin().read_to_end(&mut buf).ok()?;
        Some(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // `GREPPLUS_REAL_GREP` is a process-global env var. The two
    // `discover_grep` env-override tests must run serially so they
    // do not race each other (or any production `set_var`) and produce
    // a false pass/fail.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn empty_argv_is_rejected() {
        let r = run_grep(std::path::Path::new("/bin/true"), &[]);
        assert!(matches!(r, Err(Error::Invalid(_))));
    }

    #[test]
    fn discover_grep_respects_env_override() {
        let _g = ENV_LOCK.lock().unwrap();
        // Use a path that is guaranteed to exist on macOS/Linux.
        let candidate = "/bin/cat";
        std::env::set_var("GREPPLUS_REAL_GREP", candidate);
        let r = discover_grep();
        std::env::remove_var("GREPPLUS_REAL_GREP");
        let p = r.expect("env override should succeed");
        assert_eq!(p, std::path::PathBuf::from(candidate));
    }

    #[test]
    fn discover_grep_rejects_nonexistent_env_override() {
        let _g = ENV_LOCK.lock().unwrap();
        std::env::set_var("GREPPLUS_REAL_GREP", "/this/does/not/exist/at/all");
        let r = discover_grep();
        std::env::remove_var("GREPPLUS_REAL_GREP");
        assert!(matches!(r, Err(Error::Config(_))));
    }

    #[test]
    fn run_grep_os_empty_argv_is_rejected() {
        let r = run_grep_os(std::path::Path::new("/bin/true"), &[]);
        assert!(matches!(r, Err(Error::Invalid(_))));
    }

    // P0 (R-014 re-review): a non-UTF-8 pattern AND a non-UTF-8 path must
    // forward to real grep with grep's own exit code and no panic. Before
    // the fix the entrypoint went through `std::env::args()` /
    // `Vec<String>` which would have made the wrapper unable to even
    // represent these bytes; `run_grep_os` forwards the original
    // `OsString` verbatim. We invoke `/usr/bin/grep` (or `/bin/grep`)
    // directly so the test exercises the real forwarding path. The
    // pattern `0xff` and the path `f\xff.txt` are both invalid UTF-8.
    #[cfg(unix)]
    #[test]
    fn run_grep_os_forwards_non_utf8_pattern_and_path() {
        use std::os::unix::ffi::OsStringExt;

        let real = ["/usr/bin/grep", "/bin/grep"]
            .into_iter()
            .map(std::path::PathBuf::from)
            .find(|p| p.is_file());
        let Some(real) = real else {
            // No system grep available in this sandbox; skip rather than
            // fail. The classifier/argv-forwarding logic is still covered
            // by the unit tests above.
            return;
        };

        // A non-UTF-8 pattern and a non-UTF-8 (nonexistent) path. Real
        // grep will fail to open the path and exit 2 (error) — the point
        // is that we forward the raw bytes and surface grep's own rc
        // without ever panicking on the undecodable argv.
        let pattern = OsString::from_vec(vec![0xff]);
        let path = OsString::from_vec(vec![b'f', 0xff, b'.', b't', b'x', b't']);
        let argv = vec![OsString::from("grepplus-grep"), pattern, path];

        let rc = run_grep_os(&real, &argv).expect("must not panic on non-UTF-8 argv");
        // grep returns 2 on a file-open error; either way it is grep's own
        // rc, not a Rust panic (rc 101).
        assert!(
            rc == 1 || rc == 2,
            "expected grep's own rc (1 or 2) for a missing non-UTF-8 path, got {rc}"
        );
    }
}
