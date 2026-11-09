//! Turn unrecoverable thread creation failures into a bounded worker failure.
//! Ordinary page/decoder panics keep Servo's existing handling. Only an actual
//! OS thread-allocation error takes this path; this is not a retry/timeout gate.
use std::fs::File;
use std::io;
use std::sync::{Arc, Mutex, OnceLock};

pub(crate) fn spawn_error(context: &str, error: &io::Error) -> String {
    if matches!(error.raw_os_error(), Some(libc::EAGAIN | libc::ENOMEM)) {
        format!("resource_limit: {context}: {error}; {}", pid_budget())
    } else {
        format!("{context}: {error}")
    }
}

pub(crate) fn pid_budget() -> String {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/self/cgroup").unwrap_or_default();
        if let Some(relative) = text.lines().find_map(|line| line.strip_prefix("0::")) {
            let root = std::path::Path::new("/sys/fs/cgroup");
            if !std::path::Path::new(relative).components().any(|c|
                matches!(c, std::path::Component::ParentDir))
            {
                let mut path = root.join(relative.trim_start_matches('/'));
                let mut values = Vec::new();
                for _ in 0..32 {
                    let read = |name: &str| std::fs::read_to_string(path.join(name)).ok()
                        .map(|v| v.split_whitespace().take(8).collect::<Vec<_>>().join(" "));
                    if let Some(limit) = read("pids.max").filter(|v| v != "max") {
                        values.push(format!("pids.max={limit},current={},peak={},events={}",
                            read("pids.current").unwrap_or_else(|| "?".into()),
                            read("pids.peak").unwrap_or_else(|| "?".into()),
                            read("pids.events").unwrap_or_else(|| "?".into())));
                    }
                    if path == root || !path.pop() { break; }
                }
                if !values.is_empty() { return values.join("; "); }
            }
        }
    }
    "kernel PID/thread budget unavailable; inspect PID/thread and memory limits".into()
}


static FAILURE_PROTOCOL: OnceLock<Arc<Mutex<File>>> = OnceLock::new();

/// Fallible thread builders must use the same typed path as Servo's panicking
/// spawn sites. In a content worker this returns the limit and exits; outside
/// that role there is no authenticated parent protocol to report to.
pub(crate) fn report_thread_spawn_failure(context: &str, error: &io::Error) {
    if matches!(error.raw_os_error(), Some(libc::EAGAIN | libc::ENOMEM)) {
        let reason = format!("thread creation failed in {context}: {error}");
        if let Some(protocol) = FAILURE_PROTOCOL.get() {
            terminate_with_resource_limit(protocol, &reason);
        }
        eprintln!("web-runtime: resource_limit: {reason}");
    } else {
        eprintln!("web-runtime: failed to spawn {context}: {error}");
    }
}

fn terminate_with_resource_limit(protocol: &Mutex<File>, reason: &str) -> ! {
    use std::os::fd::AsRawFd;
    use std::time::{Duration, Instant};
    let deadline = Instant::now() + Duration::from_millis(200);
    // A normal reply can be finishing when a background thread hits the limit.
    // Wait only within the same small failure budget, then send one full frame.
    let writer = loop {
        match protocol.try_lock() {
            Ok(writer) => break Some(writer),
            Err(std::sync::TryLockError::Poisoned(error)) => break Some(error.into_inner()),
            Err(std::sync::TryLockError::WouldBlock) => {}
        }
        if Instant::now() >= deadline { break None; }
        std::thread::sleep(Duration::from_millis(1));
    };
    if let Some(writer) = writer {
        let mut frame = Vec::new();
        if crate::protocol::write_message(&mut frame,
            &crate::protocol::Message::resource_limit(crate::protocol::WorkerKind::Content,
                reason.to_owned())).is_ok()
        {
            let mut offset = 0;
            while offset < frame.len() && Instant::now() < deadline {
                let sent = unsafe { libc::send(writer.as_raw_fd(),
                    frame[offset..].as_ptr().cast(), frame.len() - offset,
                    libc::MSG_DONTWAIT | libc::MSG_NOSIGNAL) };
                if sent > 0 { offset += sent as usize; continue; }
                let error = io::Error::last_os_error();
                if !matches!(error.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted) {
                    break;
                }
                let mut ready = libc::pollfd { fd: writer.as_raw_fd(), events: libc::POLLOUT, revents: 0 };
                unsafe { libc::poll(&mut ready, 1, 10); }
            }
        }
    }
    eprintln!("web-runtime: resource_limit: {reason}");
    // A failed constellation cannot shut down normally. Skip C++ atexit
    // destructors while JS threads are live; the supervisor reaps this group.
    unsafe { libc::_exit(70); }
}

#[cfg(feature = "content-runtime")]
pub(crate) fn install_thread_spawn_failure_hook(protocol: Arc<Mutex<File>>) {
    let _ = FAILURE_PROTOCOL.set(Arc::clone(&protocol));
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let payload = info.payload().downcast_ref::<String>().map(String::as_str)
            .or_else(|| info.payload().downcast_ref::<&str>().copied()).unwrap_or("");
        let allocation = payload.contains("Resource temporarily unavailable")
            || payload.contains("kind: WouldBlock") || payload.contains("kind: OutOfMemory");
        let thread = payload.to_ascii_lowercase().contains("thread")
            || payload.contains("pool") || payload.contains("runtime");
        if !allocation || !thread || servo::should_panic_hook_suppress_termination() {
            previous(info);
            return;
        }
        let name = std::thread::current().name().unwrap_or("unnamed").to_owned();
        let reason = format!("thread creation failed in {name}: {}",
            payload.chars().take(1000).collect::<String>());
        terminate_with_resource_limit(&protocol, &reason);
    }));
}
