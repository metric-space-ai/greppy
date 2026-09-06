use std::io::{self, BufRead, BufReader, Read};

const MAX_FRAME_BYTES: u64 = 4096;

/// The parent publishes one newline-terminated capability before spawning.
/// EOF is not its delimiter: an inherited/duplicated writer may outlive the
/// sender. Waiting for all writers to close can stall a fully received token.
pub(crate) fn read_frame(reader: impl Read) -> io::Result<String> {
    let mut frame = Vec::new();
    BufReader::new(reader.take(MAX_FRAME_BYTES + 1)).read_until(b'\n', &mut frame)?;
    if frame.len() as u64 > MAX_FRAME_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "capability frame from inherited FD exceeds 4096 bytes",
        ));
    }
    if frame.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "empty capability from inherited FD",
        ));
    }
    if frame.last() != Some(&b'\n') {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "incomplete capability frame from inherited FD (missing newline)",
        ));
    }
    let text = std::str::from_utf8(&frame).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "non-UTF-8 capability frame from inherited FD",
        )
    })?;
    let token = text.trim();
    if token.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "empty capability from inherited FD",
        ));
    }
    Ok(token.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::mpsc;
    use std::time::Duration;

    #[test]
    fn complete_frame_does_not_wait_for_a_retained_pipe_writer() {
        let (reader, mut writer) = io::pipe().unwrap();
        let retained_writer = writer.try_clone().unwrap();
        writer.write_all(b"worker-capability\n").unwrap();
        drop(writer);
        let (tx, rx) = mpsc::channel();
        let read_thread = std::thread::spawn(move || {
            let _ = tx.send(read_frame(reader));
        });
        // Retain a real pipe writer until after the deadline. Clean up before
        // asserting so a failing regression cannot leave a blocked test thread.
        let before_close = rx.recv_timeout(Duration::from_secs(2));
        drop(retained_writer);
        read_thread.join().unwrap();
        assert_eq!(
            before_close
                .expect("complete capability frame waited for unrelated writer EOF")
                .unwrap(),
            "worker-capability"
        );
    }

    #[test]
    fn accepts_one_complete_frame_and_crlf() {
        assert_eq!(read_frame(&b"token\n"[..]).unwrap(), "token");
        assert_eq!(read_frame(&b"token\r\n"[..]).unwrap(), "token");
    }

    #[test]
    fn rejects_empty_truncated_invalid_and_oversized_frames() {
        for bytes in [&b""[..], &b"\n"[..], &b" \r\n"[..]] {
            assert_eq!(
                read_frame(bytes).unwrap_err().kind(),
                io::ErrorKind::InvalidInput
            );
        }
        assert_eq!(
            read_frame(&b"token"[..]).unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
        assert_eq!(
            read_frame(&b"\xff\n"[..]).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        assert_eq!(
            read_frame(&vec![b'x'; 4097][..]).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn limits_reads_even_if_an_untrusted_writer_never_ends() {
        assert_eq!(
            read_frame(io::repeat(b'x')).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    }
}
