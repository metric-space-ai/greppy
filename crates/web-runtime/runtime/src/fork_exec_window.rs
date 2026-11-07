use std::sync::Mutex;

// Darwin cannot create the std fork/exec error pipe with atomic CLOEXEC.
// Keep worker descriptor preparation and fork/exec in one parent-side window,
// so a sibling worker cannot inherit that pipe before its flags are sealed.
static WINDOW: Mutex<()> = Mutex::new(());

pub(crate) fn exclusive<T>(operation: impl FnOnce() -> T) -> T {
    let _guard = WINDOW.lock().unwrap_or_else(|error| error.into_inner());
    operation()
}

#[cfg(test)]
mod tests {
    use super::exclusive;
    use std::fs::File;
    use std::io;
    use std::mem::MaybeUninit;
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::fs::MetadataExt;
    use std::os::unix::process::CommandExt;
    use std::process::Command;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    #[test]
    #[ignore = "subprocess fixture for the real fork/exec descriptor contract"]
    fn descriptor_probe_child() {
        let values: Vec<u64> = std::env::var("GREPPY_FORK_WINDOW_PROBE")
            .expect("only the parent contract launches this fixture")
            .split(':')
            .map(|value| value.parse().unwrap())
            .collect();
        assert_eq!(values.len(), 3);
        let mut stat = MaybeUninit::<libc::stat>::uninit();
        let open = unsafe { libc::fstat(values[0] as libc::c_int, stat.as_mut_ptr()) } == 0;
        let inherited = if open {
            let stat = unsafe { stat.assume_init() };
            stat.st_dev as u64 == values[1] && stat.st_ino as u64 == values[2]
        } else {
            false
        };
        assert!(
            !inherited,
            "child inherited a descriptor from another spawn window"
        );
    }

    #[test]
    fn concurrent_fork_exec_does_not_capture_an_unsealed_descriptor() {
        let (ready_tx, ready_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let (attempt_tx, attempt_rx) = mpsc::channel();
        let (entered_tx, entered_rx) = mpsc::channel();
        let holder = thread::spawn(move || {
            exclusive(|| -> io::Result<()> {
                // Deliberately expose the same non-atomic interval as pipe()
                // followed by fcntl(CLOEXEC), with real owned kernel descriptors.
                let mut fds = [0; 2];
                if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
                    return Err(io::Error::last_os_error());
                }
                let read = unsafe { File::from_raw_fd(fds[0]) };
                let write = unsafe { File::from_raw_fd(fds[1]) };
                let metadata = write.metadata()?;
                ready_tx
                    .send((write.as_raw_fd(), metadata.dev(), metadata.ino()))
                    .map_err(io::Error::other)?;
                release_rx
                    .recv_timeout(Duration::from_secs(10))
                    .map_err(io::Error::other)?;
                for fd in [read.as_raw_fd(), write.as_raw_fd()] {
                    let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
                    if flags < 0
                        || unsafe { libc::fcntl(fd, libc::F_SETFD, flags | libc::FD_CLOEXEC) } < 0
                    {
                        return Err(io::Error::last_os_error());
                    }
                }
                Ok(())
            })
        });
        let (fd, dev, ino) = ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let forking = thread::spawn(move || {
            attempt_tx.send(()).unwrap();
            exclusive(|| -> io::Result<std::process::Output> {
                entered_tx.send(()).unwrap();
                let mut command = Command::new(std::env::current_exe()?);
                command
                    .args([
                        "--exact",
                        "fork_exec_window::tests::descriptor_probe_child",
                        "--ignored",
                        "--test-threads=1",
                        "--nocapture",
                    ])
                    .env("GREPPY_FORK_WINDOW_PROBE", format!("{fd}:{dev}:{ino}"));
                // Force the same fork route as real workers. Darwin's faster
                // posix_spawn path would close unrelated FDs and hide this bug.
                unsafe {
                    command.pre_exec(|| Ok(()));
                }
                command.output()
            })
        });
        attempt_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let entered_before_seal = entered_rx.recv_timeout(Duration::from_secs(1)).is_ok();
        let output = if entered_before_seal {
            // A broken interlock must fork while the descriptor is unsealed.
            let output = forking.join();
            let _ = release_tx.send(());
            output
        } else {
            let _ = release_tx.send(());
            forking.join()
        };
        holder.join().unwrap().unwrap();
        let output = output.unwrap().unwrap();
        assert!(
            output.status.success(),
            "fork/exec inherited an unpublished sibling descriptor:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
