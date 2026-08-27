use crate::{Error, Result};

pub(crate) fn validate_portable_component(component: &str) -> Result<()> {
    if component.chars().any(|character| {
        matches!(
            character,
            '\0' | '\\' | ':' | '<' | '>' | '"' | '|' | '?' | '*'
        )
    }) || component.ends_with(['.', ' '])
    {
        return Err(Error::InvalidPath(format!(
            "path component is not portable: {component:?}"
        )));
    }

    let stem = component
        .split('.')
        .next()
        .unwrap_or(component)
        .to_ascii_uppercase();
    let reserved_device = matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL" | "CLOCK$")
        || stem.strip_prefix("COM").is_some_and(|suffix| {
            matches!(suffix, "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9")
        })
        || stem.strip_prefix("LPT").is_some_and(|suffix| {
            matches!(suffix, "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9")
        });
    if reserved_device {
        return Err(Error::InvalidPath(format!(
            "path component is reserved on Windows: {component:?}"
        )));
    }
    Ok(())
}

/// Validate a symlink target using platform-neutral separators. The kernel
/// must never be able to resolve a link from a mounted workspace back into the
/// host namespace, even when a repository was created on another OS.
pub(crate) fn validate_symlink_target(link_path: &str, target: &[u8]) -> Result<()> {
    let target = std::str::from_utf8(target).map_err(|_| {
        Error::UnsupportedRepository("non-UTF-8 symlink targets are not portable".into())
    })?;
    if target.is_empty()
        || target.contains('\0')
        || target.starts_with(['/', '\\'])
        || (target.as_bytes().get(1) == Some(&b':') && target.as_bytes()[0].is_ascii_alphabetic())
    {
        return Err(Error::InvalidPath(format!(
            "symlink target is absolute or non-portable: {target:?}"
        )));
    }

    let mut depth = link_path.split('/').count().saturating_sub(1);
    for component in target.split(['/', '\\']) {
        match component {
            "" | "." => {}
            ".." if depth == 0 => {
                return Err(Error::InvalidPath(format!(
                    "symlink target escapes workspace: {link_path} -> {target}"
                )));
            }
            ".." => depth -= 1,
            _ => depth += 1,
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permits_internal_parent_links_but_rejects_cross_platform_escape_forms() {
        validate_symlink_target("src/nested/link", b"../sibling").unwrap();
        for target in [
            &b"../../../host"[..],
            &b"..\\..\\..\\host"[..],
            &b"/etc/passwd"[..],
            &b"C:\\Windows"[..],
            &b"\\\\server\\share"[..],
        ] {
            assert!(validate_symlink_target("src/link", target).is_err());
        }
    }

    #[test]
    fn rejects_non_portable_components_on_every_host() {
        for component in [
            "..\\host",
            "file.txt:stream",
            "CON.txt",
            "lpt9.log",
            "trailing.",
            "trailing ",
            "a*",
        ] {
            assert!(validate_portable_component(component).is_err());
        }
        validate_portable_component("Ä-Case.rs").unwrap();
    }
}
