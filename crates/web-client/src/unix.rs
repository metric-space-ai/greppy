use crate::frame::{read_frame, write_frame, FrameError};
use crate::protocol::{Request, Response};
use std::io;
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::Duration;

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

pub fn request(
    socket: impl AsRef<Path>,
    request: &Request,
    timeout: Duration,
) -> Result<Response, UnixClientError> {
    let socket = socket.as_ref();
    let mut stream = UnixStream::connect(socket).map_err(UnixClientError::Connect)?;
    stream
        .set_read_timeout(Some(timeout))
        .map_err(UnixClientError::Connect)?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(UnixClientError::Connect)?;
    write_frame(&mut stream, request)
        .map_err(|error| UnixClientError::Frame(map_frame_timeout(error, timeout)))?;
    read_frame(&mut stream)
        .map_err(|error| UnixClientError::Frame(map_frame_timeout(error, timeout)))
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
