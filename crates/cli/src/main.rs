//! `greppy` binary entry point.

use std::process::ExitCode;

fn main() -> ExitCode {
    #[cfg(debug_assertions)]
    if let Some(code) = greppy::base_build_owner_watchdog_descendant_probe() {
        return ExitCode::from(code);
    }
    #[cfg(debug_assertions)]
    let is_base_build_child = std::env::var_os(greppy::ENV_BASE_BUILD_OWNER_STDIN).is_some();
    greppy::install_base_build_owner_watchdog();
    #[cfg(debug_assertions)]
    if is_base_build_child {
        if let Some(code) = greppy::run_base_build_owner_watchdog_test_harness() {
            return ExitCode::from(code);
        }
    }
    greppy::startup_trace("main.enter");
    // Tracing initialisation is best-effort: a failure should not block
    // the binary from running.
    let _ = greppy_core::logging::init();
    greppy::startup_trace("main.after_logging");

    // capture argv as `OsString` BEFORE clap consumes it
    // so a bare `grep` passthrough carrying a non-UTF-8 pattern/path
    // (`greppy -R pat $'f\xff'`) behaves like real grep instead of a
    // clap rc=2 usage error. Recognised subcommands still flow through
    // clap.
    let argv: Vec<std::ffi::OsString> = std::env::args_os().collect();
    ExitCode::from(run(argv))
}

#[cfg(not(windows))]
fn run(argv: Vec<std::ffi::OsString>) -> u8 {
    greppy::run_os(argv)
}

#[cfg(windows)]
fn run(argv: Vec<std::ffi::OsString>) -> u8 {
    const WINDOWS_CLI_STACK_BYTES: usize = 8 * 1024 * 1024;

    let worker = match std::thread::Builder::new()
        .name("greppy-main".into())
        .stack_size(WINDOWS_CLI_STACK_BYTES)
        .spawn(move || greppy::run_os(argv))
    {
        Ok(worker) => worker,
        Err(error) => {
            eprintln!("greppy: cannot start Windows worker thread: {error}");
            return 2;
        }
    };
    match worker.join() {
        Ok(code) => code,
        Err(_) => 2,
    }
}
