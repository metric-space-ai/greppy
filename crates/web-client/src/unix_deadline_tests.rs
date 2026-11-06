use super::*;
use crate::protocol::{Request, Response};
use serde_json::json;
use std::ffi::OsString;
use std::os::unix::ffi::OsStringExt;
use std::os::unix::fs::DirBuilderExt;
use std::os::unix::net::UnixListener;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

const BUDGET: Duration = Duration::from_millis(400);

struct PrivateDirectory(PathBuf);
impl Drop for PrivateDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn isolated_case(case: &str) {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "gp-dl-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    assert!(
        path.as_os_str().as_bytes().len() + 4 < 104,
        "set TMPDIR to a short private directory for Unix socket contracts"
    );
    // Install cleanup only after exclusive creation succeeds: never remove a
    // pre-existing directory on an assertion/create failure or PID reuse.
    std::fs::DirBuilder::new()
        .mode(0o700)
        .create(&path)
        .unwrap();
    let directory = PrivateDirectory(path);
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "unix::deadline_tests::absolute_deadline_child",
            "--nocapture",
        ])
        .env("GREPPY_UNIX_DEADLINE_CASE", case)
        .env("GREPPY_UNIX_DEADLINE_DIR", &directory.0)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let pid = child.id();
    let watchdog = Instant::now() + Duration::from_secs(3);
    loop {
        if child.try_wait().unwrap().is_some() {
            break;
        }
        if Instant::now() >= watchdog {
            let _ = child.kill();
            let output = child.wait_with_output().unwrap();
            panic!(
                "bounded deadline child {case} pid={pid} exceeded watchdog: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        thread::sleep(Duration::from_millis(10));
    }
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "deadline child {case}: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        if let Some((_, proof)) = line.split_once("deadline-proof:") {
            println!("deadline-proof:{proof} child_pid={pid}");
        }
    }
}

#[test]
fn absolute_deadline_slow_drip() {
    isolated_case("slow-drip");
}

#[test]
fn absolute_deadline_saturated_listener() {
    isolated_case("saturated");
}

#[test]
fn absolute_deadline_preserves_unix_name_bytes() {
    isolated_case("names");
}

// Only the parent's exact filtered child invocation sets this private marker.
// The parent owns its PID and kills/reaps it if the old blocking semantics hang.
#[test]
fn absolute_deadline_child() {
    let Ok(case) = std::env::var("GREPPY_UNIX_DEADLINE_CASE") else {
        return;
    };
    let directory = PathBuf::from(std::env::var_os("GREPPY_UNIX_DEADLINE_DIR").unwrap());
    let socket = directory.join("s");
    match case.as_str() {
        "slow-drip" => {
            let listener = UnixListener::bind(&socket).unwrap();
            let peer = thread::spawn(move || {
                let (mut stream, _) = listener.accept().unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(1)))
                    .unwrap();
                stream
                    .set_write_timeout(Some(Duration::from_secs(1)))
                    .unwrap();
                let request: Request = read_frame(&mut stream).unwrap();
                let mut frame = Vec::new();
                write_frame(&mut frame, &Response::ok(&request, json!({"ok":true}))).unwrap();
                // Every gap is shorter than400ms, but the complete valid frame
                // arrives after720ms. Per-syscall timeouts incorrectly succeed.
                for byte in &frame[..4] {
                    if stream.write_all(&[*byte]).is_err() {
                        return;
                    }
                    thread::sleep(Duration::from_millis(180));
                }
                let _ = stream.write_all(&frame[4..]);
            });
            let started = Instant::now();
            let result = request(
                &socket,
                &Request::new("private", "web.status", json!({})),
                BUDGET,
            );
            let elapsed = started.elapsed();
            assert!(
                result.as_ref().is_err_and(|error| error.is_timeout()),
                "slow drip must exhaust one deadline, got {result:?}"
            );
            assert!(
                elapsed < Duration::from_millis(900),
                "slow drip took {elapsed:?}"
            );
            println!("deadline-proof: slow-drip request elapsed={elapsed:?} budget={BUDGET:?}");
            peer.join().unwrap();
        }
        "saturated" => {
            let listener = UnixListener::bind(&socket).unwrap();
            // SAFETY: listener owns a live socket; changing this private test's
            // backlog to one makes queue saturation finite and reproducible.
            assert_eq!(unsafe { libc::listen(listener.as_raw_fd(), 1) }, 0);
            let mut queued = Vec::new();
            let mut full = false;
            for _ in 0..16 {
                match connect_timeout(&socket, Duration::from_millis(20)) {
                    Ok(stream) => queued.push(stream),
                    Err(error)
                        if error.kind() == io::ErrorKind::TimedOut
                            || (cfg!(target_vendor = "apple")
                                && error.kind() == io::ErrorKind::ConnectionRefused) =>
                    {
                        full = true;
                        break;
                    }
                    Err(error) => panic!("private queue setup: {error}"),
                }
            }
            assert!(
                full && !queued.is_empty(),
                "private listen queue did not saturate after a real connection"
            );
            let started = Instant::now();
            let result = request(
                &socket,
                &Request::new("private", "web.status", json!({})),
                BUDGET,
            );
            // Darwin refuses a saturated AF_UNIX queue immediately; Linux
            // leaves the old blocking connect pending. Preserve that OS fact,
            // without weakening the Linux watchdog-mutation requirement.
            let bounded = matches!(&result, Err(UnixClientError::Connect(error))
                if error.kind() == io::ErrorKind::TimedOut
                    || (cfg!(target_vendor = "apple") && error.kind() == io::ErrorKind::ConnectionRefused));
            assert!(
                bounded,
                "saturated connect must return its bounded OS result: {result:?}"
            );
            let elapsed = started.elapsed();
            assert!(elapsed < Duration::from_millis(900));
            println!(
                "deadline-proof: saturated-listener request elapsed={elapsed:?} budget={BUDGET:?} result={result:?}"
            );
            drop(queued);
        }
        "names" => {
            let socket = directory.join(OsString::from_vec(vec![b'n', 0xff]));
            let listener = match UnixListener::bind(&socket) {
                Ok(listener) => listener,
                // Some Darwin filesystems reject this byte name before a listener
                // can exist. Match the standard client's exact OS error there;
                // a lossy UTF-8 conversion would instead look up another name.
                Err(error)
                    if cfg!(target_vendor = "apple")
                        && error.raw_os_error() == Some(libc::EILSEQ) =>
                {
                    // Make lossy conversion observable even when this filesystem
                    // cannot bind the original byte name.
                    let lossy_socket =
                        directory.join(socket.file_name().unwrap().to_string_lossy().as_ref());
                    let lossy_listener = UnixListener::bind(&lossy_socket).unwrap();
                    lossy_listener.set_nonblocking(true).unwrap();
                    let standard = UnixStream::connect(&socket).unwrap_err();
                    assert!(standard.raw_os_error().is_some());
                    let result = request(
                        &socket,
                        &Request::new("private", "web.status", json!({})),
                        BUDGET,
                    );
                    assert!(
                        matches!(&result, Err(UnixClientError::Connect(error))
                            if error.raw_os_error() == standard.raw_os_error()),
                        "byte-exact pathname must preserve the standard OS error: {result:?}"
                    );
                    assert!(
                        matches!(lossy_listener.accept(), Err(error)
                            if error.kind() == io::ErrorKind::WouldBlock),
                        "byte-exact pathname must not reach the lossy replacement listener"
                    );
                    return;
                }
                Err(error) => panic!("byte-name fixture bind: {error}"),
            };
            let peer = thread::spawn(move || {
                let (stream, _) = listener.accept().unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(1)))
                    .unwrap();
                serve_connection(stream, |request| {
                    Response::ok(&request, json!({"exact":true}))
                })
                .unwrap();
            });
            let response = request(
                &socket,
                &Request::new("private", "web.status", json!({})),
                BUDGET,
            )
            .unwrap();
            assert_eq!(response.result, Some(json!({"exact":true})));
            peer.join().unwrap();
        }
        other => panic!("unknown private deadline case: {other}"),
    }
}
