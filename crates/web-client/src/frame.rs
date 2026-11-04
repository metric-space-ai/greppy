use std::io::{self, Read, Write};

pub const MAX_FRAME_BYTES: usize = 1024 * 1024;

#[derive(Debug)]
pub enum FrameError {
    Io(io::Error),
    Empty,
    Oversize { len: usize },
    Utf8(std::str::Utf8Error),
    Json(serde_json::Error),
}

impl std::fmt::Display for FrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::Empty => write!(f, "empty protocol frame"),
            Self::Oversize { len } => {
                write!(f, "frame length {len} exceeds {MAX_FRAME_BYTES}-byte limit")
            }
            Self::Utf8(error) => write!(f, "invalid protocol UTF-8: {error}"),
            Self::Json(error) => write!(f, "invalid protocol JSON: {error}"),
        }
    }
}

impl std::error::Error for FrameError {}

impl From<io::Error> for FrameError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// macOS reports `SO_RCVTIMEO` / `SO_SNDTIMEO` expiry as `EAGAIN`
/// (`ErrorKind::WouldBlock`, os error 35). Other platforms use `TimedOut`.
pub fn socket_timeout(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
    )
}

/// Fill `buf`, keeping bytes already read across a socket timeout.
///
/// `Read::read_exact` drops a partially filled buffer when the call returns
/// `WouldBlock`. Retrying the frame from the start then desynchronizes the
/// stream, and closing the socket makes the peer observe `UnexpectedEof`
/// ("failed to fill whole buffer") instead of a timeout.
pub fn read_exact_until<R: Read>(
    reader: &mut R,
    mut buf: &mut [u8],
    deadline: std::time::Instant,
) -> io::Result<()> {
    while !buf.is_empty() {
        if std::time::Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "timed out reading frame",
            ));
        }
        match reader.read(buf) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "failed to fill whole buffer",
                ));
            }
            Ok(n) => buf = &mut buf[n..],
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) if socket_timeout(&error) => {
                // A zero timeout, or macOS after `SO_RCVTIMEO` fires, returns
                // immediately. Yield so a peer that is still writing can fill
                // the socket before the deadline.
                std::thread::sleep(std::time::Duration::from_millis(2));
                continue;
            }
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

pub fn read_frame_until<T: serde::de::DeserializeOwned, R: Read>(
    reader: &mut R,
    deadline: std::time::Instant,
) -> Result<T, FrameError> {
    let mut header = [0_u8; 4];
    read_exact_until(reader, &mut header, deadline)?;
    let payload_len = u32::from_be_bytes(header) as usize;
    if payload_len == 0 {
        return Err(FrameError::Empty);
    }
    if payload_len > MAX_FRAME_BYTES {
        return Err(FrameError::Oversize { len: payload_len });
    }
    let mut payload = vec![0_u8; payload_len];
    read_exact_until(reader, &mut payload, deadline)?;
    let text = std::str::from_utf8(&payload).map_err(FrameError::Utf8)?;
    serde_json::from_str(text).map_err(FrameError::Json)
}

pub fn read_frame<T: serde::de::DeserializeOwned, R: Read>(
    reader: &mut R,
) -> Result<T, FrameError> {
    let mut header = [0_u8; 4];
    reader.read_exact(&mut header)?;
    let payload_len = u32::from_be_bytes(header) as usize;
    if payload_len == 0 {
        return Err(FrameError::Empty);
    }
    if payload_len > MAX_FRAME_BYTES {
        return Err(FrameError::Oversize { len: payload_len });
    }
    let mut payload = vec![0_u8; payload_len];
    reader.read_exact(&mut payload)?;
    let text = std::str::from_utf8(&payload).map_err(FrameError::Utf8)?;
    serde_json::from_str(text).map_err(FrameError::Json)
}

pub fn write_frame<T: serde::Serialize, W: Write>(
    writer: &mut W,
    value: &T,
) -> Result<(), FrameError> {
    let payload = serde_json::to_vec(value).map_err(FrameError::Json)?;
    if payload.len() > MAX_FRAME_BYTES {
        return Err(FrameError::Oversize { len: payload.len() });
    }
    let len =
        u32::try_from(payload.len()).map_err(|_| FrameError::Oversize { len: payload.len() })?;
    writer.write_all(&len.to_be_bytes())?;
    writer.write_all(&payload)?;
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn round_trips_compact_json() {
        let mut bytes = Vec::new();
        write_frame(&mut bytes, &serde_json::json!({"ok":true})).unwrap();
        assert_eq!(&bytes[..4], 11_u32.to_be_bytes());
        let value: serde_json::Value = read_frame(&mut Cursor::new(bytes)).unwrap();
        assert_eq!(value, serde_json::json!({"ok":true}));
    }

    #[test]
    fn rejects_oversize_before_payload() {
        let header = ((MAX_FRAME_BYTES as u32) + 1).to_be_bytes();
        let error = read_frame::<serde_json::Value, _>(&mut Cursor::new(header)).unwrap_err();
        assert!(matches!(error, FrameError::Oversize { .. }));
    }

    struct TimeoutThenBytes {
        pauses: u8,
        rest: Vec<u8>,
        pos: usize,
    }

    impl Read for TimeoutThenBytes {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if self.pauses > 0 {
                self.pauses -= 1;
                return Err(io::Error::new(
                    io::ErrorKind::WouldBlock,
                    "Resource temporarily unavailable (os error 35)",
                ));
            }
            if self.pos >= self.rest.len() {
                return Ok(0);
            }
            // One byte at a time, with a timeout between the length header
            // and the payload, so a restarted frame read would desynchronize.
            let n = 1.min(buf.len());
            buf[..n].copy_from_slice(&self.rest[self.pos..self.pos + n]);
            self.pos += n;
            if self.pos == 4 {
                self.pauses = 1;
            }
            Ok(n)
        }
    }

    #[test]
    fn frame_read_keeps_partial_bytes_across_eagain() {
        let mut bytes = Vec::new();
        write_frame(&mut bytes, &serde_json::json!({"kept":true})).unwrap();
        let mut reader = TimeoutThenBytes {
            pauses: 2,
            rest: bytes,
            pos: 0,
        };
        let value: serde_json::Value = read_frame_until(
            &mut reader,
            std::time::Instant::now() + std::time::Duration::from_secs(2),
        )
        .unwrap();
        assert_eq!(value, serde_json::json!({"kept":true}));
    }

    #[test]
    fn frame_read_timeout_is_not_unexpected_eof() {
        let mut reader = TimeoutThenBytes {
            pauses: u8::MAX,
            rest: Vec::new(),
            pos: 0,
        };
        let error = read_frame_until::<serde_json::Value, _>(
            &mut reader,
            std::time::Instant::now() + std::time::Duration::from_millis(30),
        )
        .unwrap_err();
        match error {
            FrameError::Io(io_error) => {
                assert_eq!(io_error.kind(), io::ErrorKind::TimedOut);
                assert!(
                    !io_error.to_string().contains("failed to fill whole buffer"),
                    "{io_error}"
                );
            }
            other => panic!("expected timeout, got {other}"),
        }
    }
}
