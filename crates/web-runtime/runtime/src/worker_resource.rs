//! Turn unrecoverable thread creation failures into a bounded worker failure.
//! Ordinary page/decoder panics keep Servo's existing handling. Only an actual
//! OS thread-allocation error takes this path; this is not a retry/timeout gate.
#[cfg(feature = "content-runtime")]
use std::fs::File;
use std::io;
#[cfg(feature = "content-runtime")]
use std::sync::{Arc, Mutex};

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
            if !std::path::Path::new(relative)
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
            {
                let mut path = root.join(relative.trim_start_matches('/'));
                let mut values = Vec::new();
                for _ in 0..32 {
                    let read = |name: &str| {
                        std::fs::read_to_string(path.join(name))
                            .ok()
                            .map(|v| v.split_whitespace().take(8).collect::<Vec<_>>().join(" "))
                    };
                    if let Some(limit) = read("pids.max").filter(|v| v != "max") {
                        values.push(format!(
                            "pids.max={limit},current={},peak={},events={}",
                            read("pids.current").unwrap_or_else(|| "?".into()),
                            read("pids.peak").unwrap_or_else(|| "?".into()),
                            read("pids.events").unwrap_or_else(|| "?".into())
                        ));
                    }
                    if path == root || !path.pop() {
                        break;
                    }
                }
                if !values.is_empty() {
                    return values.join("; ");
                }
            }
        }
    }
    "kernel PID/thread budget unavailable; inspect PID/thread and memory limits".into()
}

#[cfg(feature = "content-runtime")]
pub(crate) fn install_thread_spawn_failure_hook(protocol: Arc<Mutex<File>>) {
    use std::os::fd::AsRawFd;
    use std::time::{Duration, Instant};
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let payload = info
            .payload()
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| info.payload().downcast_ref::<&str>().copied())
            .unwrap_or("");
        let allocation = payload.contains("Resource temporarily unavailable")
            || payload.contains("kind: WouldBlock")
            || payload.contains("kind: OutOfMemory");
        let thread = payload.to_ascii_lowercase().contains("thread")
            || payload.contains("pool")
            || payload.contains("runtime");
        if !allocation || !thread || servo::should_panic_hook_suppress_termination() {
            previous(info);
            return;
        }
        let name = std::thread::current()
            .name()
            .unwrap_or("unnamed")
            .to_owned();
        let reason = format!(
            "thread creation failed in {name}: {}",
            payload.chars().take(1000).collect::<String>()
        );
        // Serialize with ordinary replies so a panic cannot interleave frames.
        // Never wait for another writer while unwinding a failed thread.
        if let Ok(writer) = protocol.try_lock() {
            let mut frame = Vec::new();
            if crate::protocol::write_message(
                &mut frame,
                &crate::protocol::Message::resource_limit(
                    crate::protocol::WorkerKind::Content,
                    reason.clone(),
                ),
            )
            .is_ok()
            {
                let deadline = Instant::now() + Duration::from_millis(200);
                let mut offset = 0;
                while offset < frame.len() && Instant::now() < deadline {
                    let sent = unsafe {
                        libc::send(
                            writer.as_raw_fd(),
                            frame[offset..].as_ptr().cast(),
                            frame.len() - offset,
                            libc::MSG_DONTWAIT | libc::MSG_NOSIGNAL,
                        )
                    };
                    if sent > 0 {
                        offset += sent as usize;
                        continue;
                    }
                    let error = io::Error::last_os_error();
                    if !matches!(
                        error.kind(),
                        io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                    ) {
                        break;
                    }
                    let mut ready = libc::pollfd {
                        fd: writer.as_raw_fd(),
                        events: libc::POLLOUT,
                        revents: 0,
                    };
                    unsafe {
                        libc::poll(&mut ready, 1, 10);
                    }
                }
            }
        }
        eprintln!("web-runtime: resource_limit: {reason}");
        // A constellation that failed to create its script cannot finish Servo's
        // normal shutdown. Exit the owned content process; its supervisor reaps
        // the worker group and returns the original ResourceLimit to the client.
        std::process::exit(70);
    }));
}
