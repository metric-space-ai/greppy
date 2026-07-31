//! grep and ripgrep compatibility.
//!
//! Split out of `lib.rs`, which had grown to 26,400 lines: the module still
//! reaches every private helper there through `use super::*`, and nothing about
//! the behaviour changes.

use super::*;

pub(crate) fn grep_passthrough_args(argv: &[std::ffi::OsString]) -> &[std::ffi::OsString] {
    let mut index = 1;
    while index < argv.len() {
        let token = &argv[index];
        if token == "--root"
            || token == "--device"
            || token == "--limit"
            || token == "--max"
            || token == "--max-bytes"
            || token == "--offset"
        {
            index = (index + 2).min(argv.len());
            continue;
        }
        let token_lossy = token.to_string_lossy();
        if token_lossy.starts_with("--root=")
            || token_lossy.starts_with("--device=")
            || token_lossy.starts_with("--limit=")
            || token_lossy.starts_with("--max=")
            || token_lossy.starts_with("--max-bytes=")
            || token_lossy.starts_with("--offset=")
            || token == "--no-gpu"
            || token == "--no-summaries"
        {
            index += 1;
            continue;
        }
        break;
    }
    &argv[index..]
}

pub(crate) fn grep_staged_git_blobs_pattern(
    query: &str,
    root_path: &std::path::Path,
    paths: &[String],
    fixed: bool,
) -> Result<Vec<greppy_search::CodeHit>> {
    use std::io::Write;

    let grep_args = if fixed {
        ["-nIF", "--", query]
    } else {
        ["-nIE", "--", query]
    };
    let mut hits = Vec::new();
    for path in paths {
        let blob_spec = format!(":{path}");
        let blob = std::process::Command::new("git")
            .args(["show", blob_spec.as_str()])
            .current_dir(root_path)
            .output()
            .map_err(|e| Error::io(format!("read staged blob {path}"), e))?;
        if !blob.status.success() {
            continue;
        }

        let mut child = match std::process::Command::new("grep")
            .args(grep_args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
        {
            Ok(child) => child,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound && !fixed => {
                return Err(Error::Invalid(
                    "search-code regex mode requires `grep`; retry with --fixed on this host"
                        .into(),
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                if greppy_discover::is_binary_bytes(&blob.stdout) {
                    continue;
                }
                for (index, line) in String::from_utf8_lossy(&blob.stdout).lines().enumerate() {
                    if line.contains(query) {
                        hits.push(greppy_search::CodeHit {
                            location: format!("{path}:{}", index + 1),
                            snippet: line.to_string(),
                            rank: 0.0,
                        });
                    }
                }
                continue;
            }
            Err(error) => {
                return Err(Error::io("spawn grep for search-code --staged", error));
            }
        };
        if let Some(stdin) = child.stdin.as_mut() {
            stdin
                .write_all(&blob.stdout)
                .map_err(|e| Error::io(format!("write staged blob {path} to grep"), e))?;
        }
        let out = child
            .wait_with_output()
            .map_err(|e| Error::io("wait for grep in search-code --staged", e))?;
        if !out.status.success() && out.status.code() != Some(1) {
            return Err(Error::Invalid(format!(
                "grep staged-source scan failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            )));
        }
        let text = String::from_utf8_lossy(&out.stdout);
        for line in text.lines() {
            if let Some((line_no, content)) = line.split_once(':') {
                hits.push(greppy_search::CodeHit {
                    location: format!("{path}:{line_no}"),
                    snippet: content.to_string(),
                    rank: 0.0,
                });
            }
        }
    }
    Ok(hits)
}
