use crate::frame::{read_frame, write_frame, FrameError};
use crate::protocol::{Request, Response};
use std::io::{self, Read, Write};
use std::os::fd::{AsRawFd, FromRawFd, RawFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::{Duration, Instant};

#[derive(Debug)]
pub enum UnixClientError {
    Connect(io::Error),
    Frame(FrameError),
}

impl std::fmt::Display for UnixClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Connect(error) => write!(f, "web runtime socket: {error}"),
            Self::Frame(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for UnixClientError {}

impl UnixClientError {
    pub fn is_timeout(&self) -> bool {
        match self {
            Self::Connect(error) | Self::Frame(FrameError::Io(error)) => {
                error.kind() == io::ErrorKind::TimedOut
            }
            Self::Frame(_) => false,
        }
    }
}

fn is_timeout_like(error: &io::Error) -> bool {
    // macOS SO_RCVTIMEO expiry surfaces as EAGAIN -> ErrorKind::WouldBlock
    // (os error 35), not TimedOut. Map both so callers see a timeout.
    matches!(
        error.kind(),
        io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
    )
}

fn map_socket_timeout(error: io::Error, timeout: Duration) -> io::Error {
    if is_timeout_like(&error) {
        io::Error::new(
            io::ErrorKind::TimedOut,
            format!("timed out after {timeout:?}"),
        )
    } else {
        error
    }
}

fn map_frame_timeout(error: FrameError, timeout: Duration) -> FrameError {
    match error {
        FrameError::Io(io_error) => FrameError::Io(map_socket_timeout(io_error, timeout)),
        other => other,
    }
}

/// One monotonic budget shared by connect and every partial frame transfer.
#[derive(Clone, Copy)]
struct Deadline {
    end: Instant,
    timeout: Duration,
}

impl Deadline {
    fn new(timeout: Duration) -> io::Result<Self> {
        if timeout.is_zero() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "socket timeout must be nonzero",
            ));
        }
        Ok(Self {
            end: Instant::now().checked_add(timeout).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "socket deadline overflow")
            })?,
            timeout,
        })
    }

    fn remaining(self) -> io::Result<Duration> {
        self.end
            .checked_duration_since(Instant::now())
            .filter(|left| !left.is_zero())
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::TimedOut,
                    format!("timed out after {:?}", self.timeout),
                )
            })
    }

    fn wait(self, fd: RawFd, events: libc::c_short) -> io::Result<()> {
        loop {
            let left = self.remaining()?;
            // Round up only the poll granularity, never replenish the budget.
            let millis = left.as_millis().saturating_add(1).min(i32::MAX as u128) as i32;
            let mut descriptor = libc::pollfd {
                fd,
                events,
                revents: 0,
            };
            // SAFETY: poll receives one live descriptor and valid writable storage.
            let result = unsafe { libc::poll(&mut descriptor, 1, millis) };
            if result < 0 {
                let error = io::Error::last_os_error();
                if error.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(error);
            }
            self.remaining()?;
            if result > 0 {
                return Ok(());
            }
        }
    }
}

fn connect_before(socket: &Path, deadline: Deadline) -> io::Result<UnixStream> {
    // Preserve the standard Path-based API's accepted names and validation.
    std::os::unix::net::SocketAddr::from_pathname(socket)?;
    let name = socket.as_os_str().as_bytes();
    // Match Unix path semantics without any UTF-8 conversion. Linux abstract
    // names include the leading NUL and may contain further NUL bytes.
    let abstract_name =
        cfg!(any(target_os = "linux", target_os = "android")) && name.first() == Some(&0);
    // SAFETY: zero is a valid initial sockaddr_un representation.
    let mut address: libc::sockaddr_un = unsafe { std::mem::zeroed() };
    let terminator = usize::from(!abstract_name && !name.is_empty());
    if name.len() + terminator > address.sun_path.len() || (!abstract_name && name.contains(&0)) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid Unix socket name",
        ));
    }
    address.sun_family = libc::AF_UNIX as _;
    for (target, byte) in address.sun_path.iter_mut().zip(name) {
        *target = *byte as _;
    }
    let length = std::mem::offset_of!(libc::sockaddr_un, sun_path) + name.len() + terminator;
    #[cfg(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "freebsd",
        target_os = "openbsd",
        target_os = "netbsd",
        target_os = "dragonfly"
    ))]
    {
        address.sun_len = length as _;
    }
    #[cfg(any(target_os = "linux", target_os = "android"))]
    let kind = libc::SOCK_STREAM | libc::SOCK_CLOEXEC;
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    let kind = libc::SOCK_STREAM;
    // SAFETY: socket has no pointer arguments; its successful fd is owned below.
    let fd = unsafe { libc::socket(libc::AF_UNIX, kind, 0) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: fd was just created and has no other owner.
    let stream = unsafe { UnixStream::from_raw_fd(fd) };
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        // SAFETY: fd is live; F_SETFD takes the integer close-on-exec flag.
        if unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) } < 0 {
            return Err(io::Error::last_os_error());
        }
    }
    #[cfg(target_vendor = "apple")]
    {
        // Match the standard socket constructor's no-SIGPIPE behavior on Darwin.
        let enabled: libc::c_int = 1;
        // SAFETY: fd is owned, and enabled points to a valid integer option value.
        if unsafe {
            libc::setsockopt(
                fd,
                libc::SOL_SOCKET,
                libc::SO_NOSIGPIPE,
                (&enabled as *const libc::c_int).cast(),
                std::mem::size_of::<libc::c_int>() as libc::socklen_t,
            )
        } < 0
        {
            return Err(io::Error::last_os_error());
        }
    }
    stream.set_nonblocking(true)?;
    loop {
        deadline.remaining()?;
        // SAFETY: address is initialized, its exact byte length fits sockaddr_un,
        // and the stream owns a live nonblocking Unix socket.
        let result = unsafe {
            libc::connect(
                fd,
                (&address as *const libc::sockaddr_un).cast(),
                length as libc::socklen_t,
            )
        };
        if result == 0 {
            return Ok(stream);
        }
        let error = io::Error::last_os_error();
        if error.kind() == io::ErrorKind::WouldBlock {
            // A full AF_UNIX listen queue may return EAGAIN without starting a
            // connection. Retry only connect, before sending any request bytes.
            std::thread::sleep(deadline.remaining()?.min(Duration::from_millis(2)));
            continue;
        }
        if matches!(
            error.raw_os_error(),
            Some(libc::EINPROGRESS) | Some(libc::EALREADY)
        ) || error.kind() == io::ErrorKind::Interrupted
        {
            deadline.wait(fd, libc::POLLOUT)?;
            if let Some(error) = stream.take_error()? {
                return Err(error);
            }
            match stream.peer_addr() {
                Ok(_) => return Ok(stream),
                Err(error) if error.kind() == io::ErrorKind::NotConnected => continue,
                Err(error) => return Err(error),
            }
        }
        return Err(error);
    }
}

struct DeadlineStream {
    stream: UnixStream,
    deadline: Deadline,
}

impl Read for DeadlineStream {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        loop {
            self.deadline.remaining()?;
            match self.stream.read(bytes) {
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    self.deadline.wait(self.stream.as_raw_fd(), libc::POLLIN)?
                }
                result => return result,
            }
        }
    }
}

impl Write for DeadlineStream {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        loop {
            self.deadline.remaining()?;
            match self.stream.write(bytes) {
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    self.deadline.wait(self.stream.as_raw_fd(), libc::POLLOUT)?
                }
                result => return result,
            }
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        self.deadline.remaining()?;
        self.stream.flush()
    }
}

/// Establish a connection within one budget, without sending a request.
/// The returned stream is blocking. TimedOut is not proof of an absent daemon;
/// callers must retain that distinction when deciding whether to start one.
pub fn connect_timeout(socket: impl AsRef<Path>, timeout: Duration) -> io::Result<UnixStream> {
    let deadline = Deadline::new(timeout)?;
    let stream = connect_before(socket.as_ref(), deadline)?;
    stream.set_nonblocking(false)?;
    deadline.remaining()?;
    Ok(stream)
}

/// One absolute deadline covers connect, frame write, and frame read.
pub fn request(
    socket: impl AsRef<Path>,
    request: &Request,
    timeout: Duration,
) -> Result<Response, UnixClientError> {
    let deadline = Deadline::new(timeout).map_err(UnixClientError::Connect)?;
    let socket = connect_before(socket.as_ref(), deadline).map_err(UnixClientError::Connect)?;
    let mut stream = DeadlineStream {
        stream: socket,
        deadline,
    };
    write_frame(&mut stream, request)
        .map_err(|error| UnixClientError::Frame(map_frame_timeout(error, timeout)))?;
    let response = read_frame(&mut stream)
        .map_err(|error| UnixClientError::Frame(map_frame_timeout(error, timeout)))?;
    deadline
        .remaining()
        .map_err(|error| UnixClientError::Frame(FrameError::Io(error)))?;
    Ok(response)
}

pub fn serve_connection<S, F>(mut stream: S, mut handle: F) -> Result<(), FrameError>
where
    S: io::Read + io::Write,
    F: FnMut(Request) -> Response,
{
    let request: Request = read_frame(&mut stream)?;
    let response = handle(request);
    write_frame(&mut stream, &response)
}

#[cfg(test)]
#[path = "unix_deadline_tests.rs"]
mod deadline_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::Request;
    use serde_json::json;
    use std::os::unix::net::{UnixListener, UnixStream};
    use std::thread;

    #[test]
    fn eagain_is_mapped_to_timed_out() {
        let error = io::Error::new(
            io::ErrorKind::WouldBlock,
            "Resource temporarily unavailable (os error 35)",
        );
        assert_eq!(error.kind(), io::ErrorKind::WouldBlock);
        let mapped = map_socket_timeout(error, Duration::from_secs(15));
        assert_eq!(mapped.kind(), io::ErrorKind::TimedOut);
        assert_eq!(mapped.to_string(), "timed out after 15s");
    }

    #[test]
    fn request_timeout_is_not_would_block() {
        let path =
            std::env::temp_dir().join(format!("greppy-unix-timeout-{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path).unwrap();
        let server = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            thread::sleep(Duration::from_millis(400));
            drop(stream);
        });
        let error = request(
            &path,
            &Request::new("run_timeout", "web.status", json!({})),
            Duration::from_millis(50),
        )
        .unwrap_err();
        assert!(error.is_timeout(), "{error}");
        assert!(
            error.to_string().contains("timed out after 50ms"),
            "{error}"
        );
        assert!(
            !error
                .to_string()
                .contains("Resource temporarily unavailable"),
            "{error}"
        );
        let _ = server.join();
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn connected_idle_peer_times_out_as_timed_out() {
        let (mut client, _server) = UnixStream::pair().unwrap();
        client
            .set_read_timeout(Some(Duration::from_millis(30)))
            .unwrap();
        let error = read_frame::<serde_json::Value, _>(&mut client).unwrap_err();
        let mapped = map_frame_timeout(error, Duration::from_millis(30));
        match mapped {
            FrameError::Io(io_error) => {
                assert_eq!(io_error.kind(), io::ErrorKind::TimedOut);
                assert_eq!(io_error.to_string(), "timed out after 30ms");
            }
            other => panic!("expected Io timeout, got {other}"),
        }
    }
}
