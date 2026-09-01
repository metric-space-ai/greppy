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
}
