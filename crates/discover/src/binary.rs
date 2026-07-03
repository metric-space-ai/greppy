//! Binary-file detection.
//!
//! The indexer should not feed binary blobs (images, object files,
//! compiled artifacts, databases) to the parser/extractor. Upstream
//! `src/discover` filters many of these by suffix, but a suffix list can
//! never be exhaustive — a `.dat`, an extensionless executable, or a
//! mislabeled file all slip through. This module adds a *content* sniff
//! on a small prefix so callers can classify a file as binary regardless
//! of its name and record an appropriate `file_state` instead of trying
//! to parse it.
//!
//! The heuristic mirrors what Git and ripgrep do: read a bounded prefix
//! and treat the file as binary if it contains a NUL byte or is not
//! valid UTF-8 within that prefix. This is deliberately conservative —
//! source code (which is what the extractor wants) is virtually always
//! valid UTF-8 with no embedded NULs, while compiled/packed formats
//! almost always trip one of the two checks early.

use std::path::Path;

/// Number of leading bytes sniffed when classifying a file. 8 KiB is the
/// same window Git uses for its `core.bigFileThreshold`-independent
/// binary check and is large enough to catch a UTF-8 multibyte sequence
/// that merely straddles the boundary in the common case.
pub const SNIFF_PREFIX_BYTES: usize = 8 * 1024;

/// Classify a byte buffer as binary using the NUL-byte / non-UTF-8
/// heuristic over at most [`SNIFF_PREFIX_BYTES`] leading bytes.
///
/// Returns `true` when the prefix contains a NUL byte or is not valid
/// UTF-8. An empty buffer is treated as **text** (not binary): empty
/// source files are legitimate and should still be inventoried.
///
/// A multibyte UTF-8 sequence that is *truncated* exactly at the sniff
/// boundary is tolerated — only an *invalid* (as opposed to incomplete)
/// sequence marks the buffer binary. This avoids misclassifying a large
/// UTF-8 text file whose 8 KiB cut point lands mid-character.
pub fn is_binary_bytes(bytes: &[u8]) -> bool {
    let prefix = &bytes[..bytes.len().min(SNIFF_PREFIX_BYTES)];
    if prefix.is_empty() {
        return false;
    }
    // NUL byte anywhere in the prefix => binary. This is the single
    // strongest signal and is what Git keys on.
    if prefix.contains(&0) {
        return true;
    }
    // Valid UTF-8 => text.
    match std::str::from_utf8(prefix) {
        Ok(_) => false,
        Err(e) => {
            // Distinguish a genuinely invalid sequence from one that is
            // merely cut off at our sniff boundary. If `valid_up_to()`
            // consumed everything except a trailing *incomplete* (but so
            // far well-formed) sequence AND we actually truncated the
            // file (the prefix is shorter than the whole buffer), treat
            // it as text — the rest of the character lives past the
            // window.
            if e.error_len().is_none() && prefix.len() < bytes.len() {
                return false;
            }
            true
        }
    }
}

/// Classify a file on disk as binary by sniffing its leading bytes.
///
/// Reads at most [`SNIFF_PREFIX_BYTES`] without loading the whole file,
/// so it is cheap even for multi-gigabyte blobs. On any read error the
/// file is reported as **not binary** (fail-open): the caller will hit
/// the same error when it tries to read the file for indexing and can
/// surface it through its own `file_state` path, rather than this sniff
/// silently swallowing it.
pub fn is_binary_file(path: &Path) -> bool {
    use std::io::Read;
    let mut f = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return false,
    };
    // Read one extra byte beyond the window when possible: if the file is
    // longer than the window, that read returns >0 and tells us the prefix
    // was truncated, so `is_binary_bytes` can tolerate a multibyte
    // sequence split across the cut point. If the file ends exactly at the
    // window, the extra read returns 0 (EOF) and the prefix is the whole
    // file, so an incomplete trailing sequence is correctly binary.
    let mut buf = vec![0u8; SNIFF_PREFIX_BYTES + 1];
    let mut filled = 0usize;
    // Loop because a single `read` may return a short count even when more
    // data is available.
    while filled < buf.len() {
        match f.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => return false,
        }
    }
    buf.truncate(filled);
    is_binary_bytes(&buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_is_text() {
        assert!(!is_binary_bytes(b""));
    }

    #[test]
    fn ascii_source_is_text() {
        assert!(!is_binary_bytes(b"fn main() {\n    println!(\"hi\");\n}\n"));
    }

    #[test]
    fn utf8_multibyte_is_text() {
        // Snowman + accented chars; valid UTF-8, no NUL.
        let s = "let s = \"héllo ☃ wörld\";".as_bytes();
        assert!(!is_binary_bytes(s));
    }

    #[test]
    fn nul_byte_is_binary() {
        assert!(is_binary_bytes(b"abc\0def"));
    }

    #[test]
    fn nul_byte_at_end_is_binary() {
        assert!(is_binary_bytes(b"plain text then\0"));
    }

    #[test]
    fn invalid_utf8_is_binary() {
        // 0xFF is never valid in UTF-8 and is not a NUL.
        assert!(is_binary_bytes(&[b'h', b'i', 0xFF, 0xFE, b'x']));
    }

    #[test]
    fn lone_continuation_byte_is_binary() {
        // 0x80 as a leading byte is an invalid (not merely incomplete)
        // sequence.
        assert!(is_binary_bytes(&[b'a', 0x80, b'b']));
    }

    #[test]
    fn truncated_multibyte_at_window_boundary_is_text() {
        // Fill to exactly the window with valid UTF-8, then make the very
        // last byte the *start* of a 2-byte sequence so the prefix ends on
        // an incomplete-but-valid lead byte, and append the continuation
        // past the window. is_binary_bytes should tolerate this.
        let mut bytes = vec![b'a'; SNIFF_PREFIX_BYTES - 1];
        bytes.push(0xC3); // lead byte of 'é'
        bytes.push(0xA9); // continuation, lives past the sniff window
        assert!(!is_binary_bytes(&bytes));
    }

    #[test]
    fn truncated_multibyte_within_buffer_without_truncation_is_binary() {
        // Same lead byte but it IS the whole buffer (no bytes past the
        // window): an incomplete sequence with nothing after it is not
        // valid text.
        let bytes = vec![0xC3];
        assert!(is_binary_bytes(&bytes));
    }

    #[test]
    fn is_binary_file_reads_disk() {
        let dir = crate::tests_support::unique_tmp();
        let text = dir.join("a.rs");
        let bin = dir.join("b.bin");
        std::fs::write(&text, b"fn x() {}\n").unwrap();
        std::fs::write(&bin, b"MZ\x00\x00\x90PE\x00binary").unwrap();
        assert!(!is_binary_file(&text));
        assert!(is_binary_file(&bin));
    }

    #[test]
    fn is_binary_file_missing_is_text_fail_open() {
        let dir = crate::tests_support::unique_tmp();
        assert!(!is_binary_file(&dir.join("does-not-exist")));
    }

    #[test]
    fn large_text_file_over_window_is_text() {
        let dir = crate::tests_support::unique_tmp();
        let p = dir.join("big.txt");
        let body = "abcdefgh".repeat(SNIFF_PREFIX_BYTES); // > window, all ASCII
        std::fs::write(&p, &body).unwrap();
        assert!(!is_binary_file(&p));
    }

    #[test]
    fn large_binary_file_with_nul_past_zero_is_binary() {
        let dir = crate::tests_support::unique_tmp();
        let p = dir.join("big.bin");
        let mut body = vec![b'A'; 100];
        body.push(0);
        body.extend(std::iter::repeat_n(b'B', SNIFF_PREFIX_BYTES * 2));
        std::fs::write(&p, &body).unwrap();
        assert!(is_binary_file(&p));
    }
}
