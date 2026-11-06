#!/usr/bin/env python3
"""Real private Unix peers; bounded child protects CI from the old connect hang."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "crates/web-client/src/unix.rs"
LEGACY = '''pub fn request(
    socket: impl AsRef<Path>,
    request: &Request,
    timeout: Duration,
) -> Result<Response, UnixClientError> {
    let socket = socket.as_ref();
    let mut stream = UnixStream::connect(socket).map_err(UnixClientError::Connect)?;
    stream.set_read_timeout(Some(timeout)).map_err(UnixClientError::Connect)?;
    stream.set_write_timeout(Some(timeout)).map_err(UnixClientError::Connect)?;
    write_frame(&mut stream, request)
        .map_err(|error| UnixClientError::Frame(map_frame_timeout(error, timeout)))?;
    read_frame(&mut stream)
        .map_err(|error| UnixClientError::Frame(map_frame_timeout(error, timeout)))
}
'''


def run(selector, red=None, diagnostic=None):
    result = subprocess.run(
        ["cargo", "test", "--locked", "--jobs", "2", "-p", "greppy-web-client",
         "--lib", selector, "--", "--test-threads=1", "--nocapture"],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=180,
    )
    if red is None:
        if result.returncode:
            raise AssertionError(result.stdout)
        for line in result.stdout.splitlines():
            if "deadline-proof:" in line:
                print(line, flush=True)
    elif not (result.returncode == 101 and "test result: FAILED" in result.stdout
              and red + " ... FAILED" in result.stdout
              and diagnostic in result.stdout):
        raise AssertionError("Wrong mutation result (not the required runtime failure):\n" + result.stdout)


run("absolute_deadline_")
original = SOURCE.read_text()
start = original.index("pub fn request(")
end = original.index("\npub fn serve_connection", start)
try:
    SOURCE.write_text(original[:start] + LEGACY + original[end:])
    run("absolute_deadline_slow_drip", "unix::deadline_tests::absolute_deadline_slow_drip",
        "slow drip must exhaust one deadline")
    print("mutation: old per-syscall timeouts turn real slow-drip contract red", flush=True)
    run("absolute_deadline_saturated_listener", "unix::deadline_tests::absolute_deadline_saturated_listener",
        "exceeded watchdog")
    print("mutation: old blocking connect turns real saturated-listener contract red; child killed and reaped", flush=True)
finally:
    SOURCE.write_text(original)
run("absolute_deadline_")
print("mutation: restored absolute connect/write/read deadline and byte-exact Unix names green", flush=True)
