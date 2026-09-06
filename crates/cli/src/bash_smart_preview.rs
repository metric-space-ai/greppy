//! Bound individual displayed lines; capture and explicit expansion stay raw.
use std::io::{self, Write};

pub(crate) const MAX_LINE_BYTES: usize = 4096;
const HEAD_BYTES: usize = 2048;
const TAIL_BYTES: usize = 1024;

pub(crate) fn oversized(bytes: &[u8]) -> bool {
    bytes.len() > MAX_LINE_BYTES
}

pub(crate) fn write_stream(writer: &mut dyn Write, bytes: &[u8]) -> io::Result<()> {
    for line in bytes.split_inclusive(|byte| *byte == b'\n') {
        write_line(writer, line)?;
    }
    Ok(())
}

/// Keep the leading diagnostic and trailing source position without emitting
/// a multi-megabyte URL or minified line. The caller reports the raw spool
/// location once per affected stream; this is a preview, never a raw receipt.
pub(crate) fn write_line(writer: &mut dyn Write, bytes: &[u8]) -> io::Result<()> {
    if !oversized(bytes) {
        return writer.write_all(bytes);
    }
    let mut head = HEAD_BYTES;
    let mut tail = bytes.len() - TAIL_BYTES;
    if let Ok(text) = std::str::from_utf8(bytes) {
        while !text.is_char_boundary(head) {
            head -= 1;
        }
        while !text.is_char_boundary(tail) {
            tail += 1;
        }
    }
    writer.write_all(&bytes[..head])?;
    write!(
        writer,
        " … [{} bytes omitted; full line in raw log] … ",
        tail - head
    )?;
    writer.write_all(&bytes[tail..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_lines_are_byte_exact_including_binary_and_line_endings() {
        for bytes in [&b"plain\r\n"[..], &b"no newline"[..], &b"\xff\x00\n"[..]] {
            let mut out = Vec::new();
            write_line(&mut out, bytes).unwrap();
            assert_eq!(out, bytes);
        }
        let many_lines = b"short\r\n".repeat(1000);
        let mut out = Vec::new();
        write_stream(&mut out, &many_lines).unwrap();
        assert_eq!(out, many_lines, "the limit applies per line, not per block");
    }

    #[test]
    fn a_single_giant_stack_line_is_bounded_and_keeps_its_location() {
        let bytes = format!("data:text/javascript;base64,{}:1:7\n", "A".repeat(160_000));
        let mut out = Vec::new();
        write_line(&mut out, bytes.as_bytes()).unwrap();
        assert!(out.len() < MAX_LINE_BYTES);
        assert!(out.starts_with(b"data:text/javascript;base64,"));
        assert!(out.ends_with(b":1:7\n"));
        assert!(String::from_utf8(out)
            .unwrap()
            .contains("full line in raw log"));
        assert_eq!(
            bytes.len(),
            160_033,
            "the source, including its newline, remains untouched"
        );
    }

    #[test]
    fn utf8_boundaries_are_preserved_and_omission_count_is_exact() {
        let bytes = format!("{}END", "🙂".repeat(10_001));
        let mut out = Vec::new();
        write_line(&mut out, bytes.as_bytes()).unwrap();
        let out = String::from_utf8(out).unwrap();
        let omitted: usize = out
            .split(" … [")
            .nth(1)
            .unwrap()
            .split(' ')
            .next()
            .unwrap()
            .parse()
            .unwrap();
        let head = out.split(" … [").next().unwrap();
        let tail = out.split(" … ").last().unwrap();
        assert_eq!(head.len() + omitted + tail.len(), bytes.len());
        assert!(out.ends_with("END"));
        assert!(out.len() < MAX_LINE_BYTES);
    }

    #[test]
    fn failing_writer_is_not_reported_as_success() {
        struct Broken;
        impl Write for Broken {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::Error::from(io::ErrorKind::BrokenPipe))
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        assert_eq!(
            write_line(&mut Broken, &vec![b'x'; 20_000])
                .unwrap_err()
                .kind(),
            io::ErrorKind::BrokenPipe
        );
    }
}
