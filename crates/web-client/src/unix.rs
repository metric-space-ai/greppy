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
    write_frame(&mut stream, request).map_err(UnixClientError::Frame)?;
    read_frame(&mut stream).map_err(UnixClientError::Frame)
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
