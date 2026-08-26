use deno_core::{JsRuntime, RuntimeOptions};
use std::io;
use web_runtime::protocol::WorkerKind;
use web_runtime::worker::run_worker;

fn main() -> io::Result<()> {
    let mut runtime = JsRuntime::new(RuntimeOptions::default());
    runtime
        .execute_script("<web-controller-worker>", "1 + 1")
        .map_err(|error| io::Error::other(format!("JavaScript startup probe failed: {error}")))?;
    let stdin = io::stdin();
    let stdout = io::stdout();
    run_worker(
        WorkerKind::Controller,
        runtime,
        &mut stdin.lock(),
        &mut stdout.lock(),
    )
}
