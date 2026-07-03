//! String and path helpers ported from `src/foundation/str_util.c`.
//!
//! These operate on `/`-separated logical paths (the upstream convention)
//! and on plain UTF-8 strings. They deliberately do **not** use
//! [`std::path`] for the path helpers: the upstream semantics are
//! byte-exact (e.g. [`path_ext`] returns empty when the last `.` precedes
//! the last `/`), and reproducing them through `Path` would be both
//! lossy on non-UTF-8 platforms and subtly different. The plain
//! substring helpers ([`starts_with`], [`ends_with`], [`contains`]) map
//! cleanly onto `std` and simply forward to it, giving callers one
//! consolidated, tested surface.

/// Join two path components with a single `/`.
///
/// Mirrors `cbm_path_join`: a trailing `/` on `base` and leading `/`s on
/// `name` are stripped before joining. An empty component yields the
/// other component unchanged.
///
/// ```
/// # use grepplus_core::strutil::path_join;
/// assert_eq!(path_join("a", "b"), "a/b");
/// assert_eq!(path_join("a/", "/b"), "a/b");
/// assert_eq!(path_join("", "b"), "b");
/// assert_eq!(path_join("a", ""), "a");
/// ```
pub fn path_join(base: &str, name: &str) -> String {
    if base.is_empty() {
        return name.to_string();
    }
    if name.is_empty() {
        return base.to_string();
    }
    let base = base.trim_end_matches('/');
    let name = name.trim_start_matches('/');
    if base.is_empty() {
        return name.to_string();
    }
    if name.is_empty() {
        return base.to_string();
    }
    let mut out = String::with_capacity(base.len() + 1 + name.len());
    out.push_str(base);
    out.push('/');
    out.push_str(name);
    out
}

/// Join an arbitrary number of path components (mirrors
/// `cbm_path_join_n`). Folds [`path_join`] left-to-right.
pub fn path_join_n<I, S>(parts: I) -> String
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut iter = parts.into_iter();
    let Some(first) = iter.next() else {
        return String::new();
    };
    let mut acc = first.as_ref().to_string();
    for part in iter {
        acc = path_join(&acc, part.as_ref());
    }
    acc
}

/// Return the extension of a path (without the dot), or `""`.
///
/// Mirrors `cbm_path_ext`: the last `.` must come after the last `/`. A
/// dotfile basename (e.g. `dir/.gitignore`) has no extension because the
/// `.` is at the start of the basename — but note upstream's exact rule
/// is "dot must be after the last slash", so `.gitignore` at the root
/// *does* count its `.` (matching the C). We reproduce the C precisely.
///
/// ```
/// # use grepplus_core::strutil::path_ext;
/// assert_eq!(path_ext("a/b.rs"), "rs");
/// assert_eq!(path_ext("a.b/c"), "");
/// assert_eq!(path_ext("noext"), "");
/// ```
pub fn path_ext(path: &str) -> &str {
    let dot = path.rfind('.');
    let slash = path.rfind('/');
    match dot {
        None => "",
        Some(d) => {
            if let Some(s) = slash {
                if d < s {
                    return "";
                }
            }
            &path[d + 1..]
        }
    }
}

/// Return the final component of a `/`-separated path (basename).
///
/// Mirrors `cbm_path_base`. With no `/`, the whole string is returned.
/// A trailing `/` yields an empty basename, matching the C pointer math.
///
/// ```
/// # use grepplus_core::strutil::path_base;
/// assert_eq!(path_base("a/b/c.rs"), "c.rs");
/// assert_eq!(path_base("noslash"), "noslash");
/// assert_eq!(path_base("a/b/"), "");
/// ```
pub fn path_base(path: &str) -> &str {
    match path.rfind('/') {
        Some(i) => &path[i + 1..],
        None => path,
    }
}

/// Return the directory portion of a `/`-separated path.
///
/// Mirrors `cbm_path_dir`: with no `/`, returns `"."`. The trailing
/// slash is not included in the result.
///
/// ```
/// # use grepplus_core::strutil::path_dir;
/// assert_eq!(path_dir("a/b/c.rs"), "a/b");
/// assert_eq!(path_dir("noslash"), ".");
/// assert_eq!(path_dir("/abs"), "");
/// ```
pub fn path_dir(path: &str) -> &str {
    match path.rfind('/') {
        Some(i) => &path[..i],
        None => ".",
    }
}

/// Strip the extension (and its dot) from a path.
///
/// Mirrors `cbm_str_strip_ext`. If there is no extension (per the same
/// rule as [`path_ext`]), the input is returned unchanged.
///
/// ```
/// # use grepplus_core::strutil::strip_ext;
/// assert_eq!(strip_ext("a/b.rs"), "a/b");
/// assert_eq!(strip_ext("a.b/c"), "a.b/c");
/// assert_eq!(strip_ext("noext"), "noext");
/// ```
pub fn strip_ext(path: &str) -> &str {
    let dot = path.rfind('.');
    let slash = path.rfind('/');
    match dot {
        None => path,
        Some(d) => {
            if let Some(s) = slash {
                if d < s {
                    return path;
                }
            }
            &path[..d]
        }
    }
}

/// Whether `s` begins with `prefix` (mirrors `cbm_str_starts_with`).
pub fn starts_with(s: &str, prefix: &str) -> bool {
    s.starts_with(prefix)
}

/// Whether `s` ends with `suffix` (mirrors `cbm_str_ends_with`).
pub fn ends_with(s: &str, suffix: &str) -> bool {
    s.ends_with(suffix)
}

/// Whether `s` contains `sub` (mirrors `cbm_str_contains`). An empty
/// `sub` is contained by every string, matching the C.
pub fn contains(s: &str, sub: &str) -> bool {
    s.contains(sub)
}

/// Split `s` on a single-character delimiter.
///
/// Mirrors `cbm_str_split`: the result always has at least one element,
/// empty fields are preserved, and a trailing delimiter produces a
/// trailing empty field. Equivalent to the count semantics of the C
/// (number of fields == number of delimiters + 1).
///
/// ```
/// # use grepplus_core::strutil::split;
/// assert_eq!(split("a,b,c", ','), vec!["a", "b", "c"]);
/// assert_eq!(split("a,,b", ','), vec!["a", "", "b"]);
/// assert_eq!(split("a,", ','), vec!["a", ""]);
/// assert_eq!(split("", ','), vec![""]);
/// ```
pub fn split(s: &str, sep: char) -> Vec<&str> {
    s.split(sep).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_join_basic() {
        assert_eq!(path_join("a", "b"), "a/b");
        assert_eq!(path_join("a/b", "c"), "a/b/c");
    }

    #[test]
    fn path_join_strips_slashes() {
        assert_eq!(path_join("a/", "b"), "a/b");
        assert_eq!(path_join("a", "/b"), "a/b");
        assert_eq!(path_join("a///", "///b"), "a/b");
    }

    #[test]
    fn path_join_empty_components() {
        assert_eq!(path_join("", "b"), "b");
        assert_eq!(path_join("a", ""), "a");
        assert_eq!(path_join("", ""), "");
    }

    #[test]
    fn path_join_all_slashes_base() {
        // base that is only slashes collapses to empty, yielding name.
        assert_eq!(path_join("///", "b"), "b");
        // name that is only slashes collapses to empty, yielding base.
        assert_eq!(path_join("a", "///"), "a");
    }

    #[test]
    fn path_join_n_folds() {
        assert_eq!(path_join_n(["a", "b", "c"]), "a/b/c");
        assert_eq!(path_join_n(["a/", "/b/", "/c"]), "a/b/c");
        assert_eq!(path_join_n(["solo"]), "solo");
        assert_eq!(path_join_n(Vec::<&str>::new()), "");
    }

    #[test]
    fn ext_basic() {
        assert_eq!(path_ext("file.rs"), "rs");
        assert_eq!(path_ext("a/b/file.tar.gz"), "gz");
        assert_eq!(path_ext("noext"), "");
    }

    #[test]
    fn ext_dot_before_slash_is_none() {
        // The '.' is in a directory component, not the basename.
        assert_eq!(path_ext("a.b/c"), "");
        assert_eq!(path_ext("a.b/c.d"), "d");
    }

    #[test]
    fn ext_trailing_dot() {
        assert_eq!(path_ext("file."), "");
    }

    #[test]
    fn base_basic() {
        assert_eq!(path_base("a/b/c.rs"), "c.rs");
        assert_eq!(path_base("noslash"), "noslash");
        assert_eq!(path_base("/abs/path"), "path");
    }

    #[test]
    fn base_trailing_slash() {
        assert_eq!(path_base("a/b/"), "");
    }

    #[test]
    fn dir_basic() {
        assert_eq!(path_dir("a/b/c.rs"), "a/b");
        assert_eq!(path_dir("noslash"), ".");
        assert_eq!(path_dir("/abs"), "");
        assert_eq!(path_dir("a/b/"), "a/b");
    }

    #[test]
    fn strip_ext_basic() {
        assert_eq!(strip_ext("file.rs"), "file");
        assert_eq!(strip_ext("a/b/file.tar.gz"), "a/b/file.tar");
        assert_eq!(strip_ext("noext"), "noext");
    }

    #[test]
    fn strip_ext_dot_before_slash_unchanged() {
        assert_eq!(strip_ext("a.b/c"), "a.b/c");
    }

    #[test]
    fn substring_helpers() {
        assert!(starts_with("foobar", "foo"));
        assert!(!starts_with("foobar", "bar"));
        assert!(ends_with("foobar", "bar"));
        assert!(!ends_with("foobar", "foo"));
        assert!(contains("foobar", "oob"));
        assert!(!contains("foobar", "xyz"));
        // Empty needle is contained by everything (matches C).
        assert!(contains("foobar", ""));
        assert!(contains("", ""));
    }

    #[test]
    fn split_basic() {
        assert_eq!(split("a,b,c", ','), vec!["a", "b", "c"]);
        assert_eq!(split("single", ','), vec!["single"]);
    }

    #[test]
    fn split_preserves_empty_fields() {
        assert_eq!(split("a,,b", ','), vec!["a", "", "b"]);
        assert_eq!(split(",a", ','), vec!["", "a"]);
        assert_eq!(split("a,", ','), vec!["a", ""]);
        assert_eq!(split("", ','), vec![""]);
        assert_eq!(split(",", ','), vec!["", ""]);
    }
}
