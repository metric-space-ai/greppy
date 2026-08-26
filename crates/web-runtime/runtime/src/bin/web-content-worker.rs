use servo::ServoBuilder;
use std::io;
use web_runtime::protocol::WorkerKind;
use web_runtime::worker::run_worker;

fn main() -> io::Result<()> {
    let runtime = ServoBuilder::default().build();
    let stdin = io::stdin();
    let stdout = io::stdout();
    run_worker(
        WorkerKind::Content,
        runtime,
        &mut stdin.lock(),
        &mut stdout.lock(),
    )
}
