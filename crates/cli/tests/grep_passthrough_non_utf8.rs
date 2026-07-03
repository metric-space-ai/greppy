//! P2 regression (re-review): the unified `grepplus` / `grepplus grep`
//! passthrough must NOT reject a non-UTF-8 argv with a clap rc=2 usage
//! error. `grepplus -R pat $'f\xff'` must behave like real grep — argv
//! is captured via `args_os` before clap can choke on the undecodable
//! bytes, and forwarded to real grep byte-for-byte.
//!
//! These spawn the real `grepplus` binary with raw `OsString` arguments,
//! so they reproduce the agent invocation exactly. Before the fix the
//! process exited 2 with a clap usage error ("invalid UTF-8"); after the
//! fix it returns grep's own rc (1 = no match, 2 = file error) and never
//! a clap error.

#[cfg(unix)]
mod unix {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;
    use std::process::Command;

    fn bin() -> &'static str {
        env!("CARGO_BIN_EXE_grepplus")
    }

    fn have_real_grep() -> bool {
        ["/usr/bin/grep", "/bin/grep"]
            .iter()
            .any(|p| std::path::Path::new(p).is_file())
    }

    /// `grepplus -R pat $'f\xff'` — a non-UTF-8 PATH argument. Real grep
    /// fails to open the path and exits 2; the wrapper must surface that,
    /// not a clap rc=2 "invalid UTF-8" usage error and not a panic.
    #[test]
    fn bare_passthrough_non_utf8_path_is_not_a_clap_error() {
        if !have_real_grep() {
            return;
        }
        let bad_path = OsString::from_vec(vec![b'f', 0xff]);
        let out = Command::new(bin())
            .arg("-R")
            .arg("pat")
            .arg(&bad_path)
            // No store dir / no repo: we only care that argv forwarding
            // does not panic or clap-reject. stdin is the parent's; grep
            // with -R on a missing path returns 2 regardless.
            .stdin(std::process::Stdio::null())
            .output()
            .expect("spawn grepplus");
        let code = out.status.code().unwrap_or(-1);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_ne!(
            code, 101,
            "wrapper must not panic on non-UTF-8 argv; stderr={stderr}"
        );
        assert!(
            !stderr.contains("invalid utf") && !stderr.to_lowercase().contains("usage:"),
            "must not be a clap usage error; rc={code} stderr={stderr}"
        );
        // grep's own rc for a missing path under -R is 2 (error). Accept
        // 1 or 2 to stay robust across grep variants.
        assert!(
            code == 1 || code == 2,
            "expected grep's own rc (1/2), got {code}; stderr={stderr}"
        );
    }

    /// A non-UTF-8 PATTERN must also forward cleanly. We pipe data on
    /// stdin and search a single (existing) file, so the only undecodable
    /// argv element is the pattern.
    #[test]
    fn bare_passthrough_non_utf8_pattern_is_not_a_clap_error() {
        if !have_real_grep() {
            return;
        }
        let bad_pattern = OsString::from_vec(vec![0xff]);
        let out = Command::new(bin())
            .arg(&bad_pattern)
            .arg("/etc/hostname") // some existing file to search
            .stdin(std::process::Stdio::null())
            .output()
            .expect("spawn grepplus");
        let code = out.status.code().unwrap_or(-1);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_ne!(code, 101, "must not panic; stderr={stderr}");
        assert!(
            !stderr.to_lowercase().contains("usage:"),
            "must not be a clap usage error; rc={code} stderr={stderr}"
        );
        // 0xff matches nothing valid in a text file → grep rc 1 (no
        // match) is the normal outcome; some locales may error (2).
        assert!(
            code == 0 || code == 1 || code == 2,
            "expected grep's own rc, got {code}; stderr={stderr}"
        );
    }

    /// Recognised subcommands must still work after the args_os
    /// interception (the router only diverts genuine passthroughs).
    #[test]
    fn recognised_subcommand_still_reaches_clap() {
        // `grepplus --help` prints help and exits 0 via clap.
        let out = Command::new(bin())
            .arg("--help")
            .output()
            .expect("spawn grepplus");
        assert!(out.status.success(), "--help should exit 0");
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            stdout.contains("drop-in grep wrapper"),
            "help must show the honest about string; got: {stdout}"
        );
    }
}
