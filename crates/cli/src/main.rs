//! `grepplus` binary entry point.

use std::process::ExitCode;

fn main() -> ExitCode {
    // Tracing initialisation is best-effort: a failure should not block
    // the binary from running.
    let _ = grepplus_core::logging::init();

    // P2 (re-review): capture argv as `OsString` BEFORE clap consumes it
    // so a bare `grep` passthrough carrying a non-UTF-8 pattern/path
    // (`grepplus -R pat $'f\xff'`) behaves like real grep instead of a
    // clap rc=2 usage error. Recognised subcommands still flow through
    // clap.
    let argv: Vec<std::ffi::OsString> = std::env::args_os().collect();
    ExitCode::from(grepplus::run_os(argv))
}
