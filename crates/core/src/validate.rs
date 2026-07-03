//! Foundation validators ported from `src/foundation/str_util.c`
//! (`cbm_validate_shell_arg`, `cbm_validate_project_name`,
//! `cbm_json_escape`).
//!
//! These are defensive, allocation-light checks used before passing
//! untrusted strings to a shell, using them as on-disk identifiers, or
//! embedding them in JSON. The behaviour mirrors the upstream C exactly
//! so that the two implementations stay in lockstep.

use std::fmt;

/// Reason a string failed [`validate_shell_arg`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellArgError {
    /// The offending metacharacter.
    pub ch: char,
    /// Byte offset of the offending character within the input.
    pub offset: usize,
}

impl fmt::Display for ShellArgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "shell metacharacter {:?} at byte offset {} is not allowed",
            self.ch, self.offset
        )
    }
}

impl std::error::Error for ShellArgError {}

/// Returns `true` if a byte is a shell metacharacter rejected by the
/// upstream `cbm_validate_shell_arg`.
///
/// The upstream set is: `'`, `"`, `;`, `|`, `&`, `$`, `` ` ``, `<`, `>`,
/// `\n`, `\r`, and (on non-Windows) `\`. We always reject the backslash
/// here — the on-disk artifacts and shells this guards are POSIX.
const fn is_shell_meta(b: u8) -> bool {
    matches!(
        b,
        b'\'' | b'"' | b';' | b'|' | b'&' | b'$' | b'`' | b'<' | b'>' | b'\n' | b'\r' | b'\\'
    )
}

/// Validate that `s` contains no shell metacharacters, returning the
/// first offender on failure.
///
/// This is the `Result`-returning form. See [`is_valid_shell_arg`] for a
/// plain boolean. Mirrors `cbm_validate_shell_arg`: an empty string is
/// considered valid (it has no metacharacters).
pub fn validate_shell_arg(s: &str) -> Result<(), ShellArgError> {
    for (offset, b) in s.bytes().enumerate() {
        if is_shell_meta(b) {
            return Err(ShellArgError {
                ch: b as char,
                offset,
            });
        }
    }
    Ok(())
}

/// Boolean form of [`validate_shell_arg`]: `true` when `s` is free of
/// shell metacharacters.
pub fn is_valid_shell_arg(s: &str) -> bool {
    validate_shell_arg(s).is_ok()
}

/// Reason a string failed [`validate_project_name`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProjectNameError {
    /// The name was empty.
    Empty,
    /// The name contained `..` (directory traversal).
    Traversal,
    /// The name contained a path separator (`/` or `\`).
    PathSeparator,
    /// The name began with `.` (hidden / relative reference).
    LeadingDot,
    /// The name contained a character outside `[A-Za-z0-9._-]`.
    InvalidChar(char),
}

impl fmt::Display for ProjectNameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "project name must not be empty"),
            Self::Traversal => write!(f, "project name must not contain '..'"),
            Self::PathSeparator => {
                write!(f, "project name must not contain a path separator")
            }
            Self::LeadingDot => write!(f, "project name must not begin with '.'"),
            Self::InvalidChar(c) => write!(
                f,
                "project name contains invalid character {c:?}; only [A-Za-z0-9._-] are allowed"
            ),
        }
    }
}

impl std::error::Error for ProjectNameError {}

/// Validate a project name, returning the specific reason on failure.
///
/// A faithful port of `cbm_validate_project_name`. Accepts a non-empty
/// string of `[A-Za-z0-9._-]` that does not begin with `.`, contains no
/// path separator (`/` or `\`), and contains no `..` substring. Checks
/// are applied in the same order as upstream so the reported reason
/// matches.
pub fn validate_project_name(name: &str) -> Result<(), ProjectNameError> {
    if name.is_empty() {
        return Err(ProjectNameError::Empty);
    }
    // Reject directory traversal (any `..` substring, matching upstream).
    if name.contains("..") {
        return Err(ProjectNameError::Traversal);
    }
    // Reject path separators.
    if name.contains('/') || name.contains('\\') {
        return Err(ProjectNameError::PathSeparator);
    }
    // Reject a leading dot (hidden files / relative refs).
    if name.starts_with('.') {
        return Err(ProjectNameError::LeadingDot);
    }
    // Allow only alphanumeric, dash, underscore, dot.
    for c in name.chars() {
        let ok = c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.';
        if !ok {
            return Err(ProjectNameError::InvalidChar(c));
        }
    }
    Ok(())
}

/// Boolean form of [`validate_project_name`].
pub fn is_valid_project_name(name: &str) -> bool {
    validate_project_name(name).is_ok()
}

/// Escape a string for embedding inside a JSON string literal.
///
/// Mirrors `cbm_json_escape`: `"` and `\` are backslash-escaped, `\n`,
/// `\r`, and `\t` use their short escapes, all other control characters
/// (`< 0x20`) are dropped, and everything else (including bytes `>= 0x80`)
/// is passed through verbatim. The result does **not** include the
/// surrounding quotes.
pub fn json_escape(src: &str) -> String {
    let mut out = String::with_capacity(src.len() + src.len() / 4);
    // Iterate over `char`s so multi-byte UTF-8 survives intact. Every
    // character upstream treats specially (`"`, `\`, control chars) is
    // ASCII, so character-level handling reproduces the C exactly while
    // passing higher code points through unchanged.
    for c in src.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            // Other control characters (< 0x20) are dropped, matching
            // upstream.
            c if (c as u32) < 0x20 => {}
            // Pass everything else through unchanged.
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- shell arg --------------------------------------------------

    #[test]
    fn shell_arg_accepts_clean_strings() {
        for good in [
            "",
            "main",
            "feature-branch",
            "src/lib.rs", // '/' is not a shell metachar
            "file.name_2",
            "a b c", // spaces are allowed (quoting is the caller's job)
            "--flag=value",
            "путь", // non-ASCII passes through
            "naïve",
        ] {
            assert!(is_valid_shell_arg(good), "expected {good:?} to be accepted");
            assert!(validate_shell_arg(good).is_ok());
        }
    }

    #[test]
    fn shell_arg_rejects_every_metacharacter() {
        // The full upstream metacharacter set must be rejected.
        for bad in [
            "a;b",
            "a|b",
            "a&b",
            "a$b",
            "a`b",
            "a<b",
            "a>b",
            "a'b",
            "a\"b",
            "a\\b",
            "a\nb",
            "a\rb",
            "$(rm -rf /)",
            "`reboot`",
            "foo && bar",
            "x > /etc/passwd",
            "a|",
            ";",
        ] {
            assert!(!is_valid_shell_arg(bad), "expected {bad:?} to be rejected");
            assert!(validate_shell_arg(bad).is_err());
        }
    }

    #[test]
    fn shell_arg_reports_first_offender_with_offset() {
        let err = validate_shell_arg("ab;cd|ef").unwrap_err();
        assert_eq!(err.ch, ';');
        assert_eq!(err.offset, 2);
    }

    // ----- project name -----------------------------------------------

    #[test]
    fn project_name_accepts_good_names() {
        for good in [
            "app",
            "my-project",
            "my_project",
            "v1.2.3",
            "a",
            "Project99",
            "a.b.c",
        ] {
            assert!(
                is_valid_project_name(good),
                "expected {good:?} to be accepted"
            );
        }
    }

    #[test]
    fn project_name_rejects_empty() {
        assert_eq!(validate_project_name(""), Err(ProjectNameError::Empty));
    }

    #[test]
    fn project_name_rejects_traversal() {
        assert_eq!(
            validate_project_name(".."),
            Err(ProjectNameError::Traversal)
        );
        assert_eq!(
            validate_project_name("a..b"),
            Err(ProjectNameError::Traversal)
        );
        assert_eq!(
            validate_project_name("foo.."),
            Err(ProjectNameError::Traversal)
        );
    }

    #[test]
    fn project_name_rejects_path_separators() {
        assert_eq!(
            validate_project_name("a/b"),
            Err(ProjectNameError::PathSeparator)
        );
        assert_eq!(
            validate_project_name("a\\b"),
            Err(ProjectNameError::PathSeparator)
        );
    }

    #[test]
    fn project_name_rejects_leading_dot() {
        assert_eq!(
            validate_project_name(".hidden"),
            Err(ProjectNameError::LeadingDot)
        );
        assert_eq!(
            validate_project_name(".git"),
            Err(ProjectNameError::LeadingDot)
        );
    }

    #[test]
    fn project_name_rejects_invalid_chars() {
        assert_eq!(
            validate_project_name("a b"),
            Err(ProjectNameError::InvalidChar(' '))
        );
        assert_eq!(
            validate_project_name("a@b"),
            Err(ProjectNameError::InvalidChar('@'))
        );
        assert_eq!(
            validate_project_name("a:b"),
            Err(ProjectNameError::InvalidChar(':'))
        );
        // Traversal is checked before the per-char scan, so '*' next to
        // a separator still reports its own category; check a standalone.
        assert_eq!(
            validate_project_name("naïve"),
            Err(ProjectNameError::InvalidChar('ï'))
        );
    }

    // ----- json escape ------------------------------------------------

    #[test]
    fn json_escape_handles_quotes_and_backslashes() {
        assert_eq!(json_escape(r#"a"b"#), r#"a\"b"#);
        assert_eq!(json_escape(r"a\b"), r"a\\b");
        assert_eq!(json_escape(r#"\"#), r"\\");
    }

    #[test]
    fn json_escape_handles_short_escapes() {
        assert_eq!(json_escape("a\nb"), "a\\nb");
        assert_eq!(json_escape("a\rb"), "a\\rb");
        assert_eq!(json_escape("a\tb"), "a\\tb");
    }

    #[test]
    fn json_escape_drops_other_control_chars() {
        // \x00, \x07 (bell), \x1f are dropped entirely.
        assert_eq!(json_escape("a\x00b\x07c\x1fd"), "abcd");
    }

    #[test]
    fn json_escape_passes_through_plain_and_unicode() {
        assert_eq!(json_escape("hello world"), "hello world");
        // Multi-byte UTF-8 survives intact.
        assert_eq!(json_escape("naïve—é"), "naïve—é");
    }

    #[test]
    fn json_escape_empty_is_empty() {
        assert_eq!(json_escape(""), "");
    }
}
