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

pub(crate) fn grep_args_with_implicit_recursion(
    args: &[std::ffi::OsString],
) -> Option<Vec<std::ffi::OsString>> {
    if grep_has_recursive_option(args)
        || grep_requests_files_without_match(args)
        || !grep_has_directory_operand(args)
    {
        return None;
    }
    let mut adjusted = Vec::with_capacity(args.len() + 1);
    adjusted.push(std::ffi::OsString::from("-r"));
    adjusted.extend_from_slice(args);
    Some(adjusted)
}

pub(crate) fn grep_requests_files_without_match(args: &[std::ffi::OsString]) -> bool {
    args.iter().any(|argument| {
        argument == "--files-without-match"
            || argument.to_str().is_some_and(|text| {
                text.strip_prefix('-')
                    .filter(|cluster| !cluster.starts_with('-'))
                    .is_some_and(|cluster| cluster.contains('L'))
            })
    })
}

pub(crate) fn grep_has_recursive_option(args: &[std::ffi::OsString]) -> bool {
    let mut options = true;
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        if options && argument == "--" {
            options = false;
            index += 1;
            continue;
        }
        if !options {
            index += 1;
            continue;
        }
        let Some(text) = argument.to_str() else {
            index += 1;
            continue;
        };
        if matches!(text, "--recursive" | "--dereference-recursive")
            || grep_short_option_has_recursive_flag(text)
        {
            return true;
        }
        let (takes_value, _) = grep_option_value_mode(text);
        if takes_value && !text.contains('=') && !grep_short_option_has_attached_value(text) {
            index += 2;
        } else {
            index += 1;
        }
    }
    false
}

pub(crate) fn grep_short_option_has_recursive_flag(option: &str) -> bool {
    let Some(cluster) = option
        .strip_prefix('-')
        .filter(|rest| !rest.starts_with('-'))
    else {
        return false;
    };
    for flag in cluster.chars() {
        if matches!(flag, 'r' | 'R') {
            return true;
        }
        if matches!(flag, 'e' | 'f' | 'A' | 'B' | 'C' | 'd' | 'D' | 'm') {
            return false;
        }
    }
    false
}

pub(crate) fn grep_has_directory_operand(args: &[std::ffi::OsString]) -> bool {
    let mut options = true;
    let mut explicit_pattern = false;
    let mut positional_pattern_seen = false;
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        if options && argument == "--" {
            options = false;
            index += 1;
            continue;
        }
        if options {
            if let Some(text) = argument
                .to_str()
                .filter(|text| text.starts_with('-') && *text != "-")
            {
                let (takes_value, supplies_pattern) = grep_option_value_mode(text);
                explicit_pattern |= supplies_pattern;
                if takes_value && !text.contains('=') && !grep_short_option_has_attached_value(text)
                {
                    index += 2;
                    continue;
                }
                index += 1;
                continue;
            }
        }
        if !explicit_pattern && !positional_pattern_seen {
            positional_pattern_seen = true;
        } else if std::path::Path::new(argument).is_dir() {
            return true;
        }
        index += 1;
    }
    false
}

pub(crate) fn grep_option_value_mode(option: &str) -> (bool, bool) {
    if option.starts_with("--regexp=") || option.starts_with("--file=") {
        return (false, true);
    }
    if matches!(option, "-e" | "--regexp" | "-f" | "--file") {
        return (true, true);
    }
    let long_takes_value = matches!(
        option,
        "--after-context"
            | "--before-context"
            | "--context"
            | "--directories"
            | "--devices"
            | "--exclude"
            | "--exclude-from"
            | "--exclude-dir"
            | "--include"
            | "--label"
            | "--max-count"
    );
    if long_takes_value {
        return (true, false);
    }
    if option.starts_with('-') && !option.starts_with("--") {
        let supplies_pattern = option[1..].chars().any(|flag| matches!(flag, 'e' | 'f'));
        let takes_value = option[1..]
            .chars()
            .any(|flag| matches!(flag, 'e' | 'f' | 'A' | 'B' | 'C' | 'd' | 'D' | 'm'));
        return (takes_value, supplies_pattern);
    }
    (false, false)
}

pub(crate) fn grep_short_option_has_attached_value(option: &str) -> bool {
    let Some(cluster) = option
        .strip_prefix('-')
        .filter(|rest| !rest.starts_with('-'))
    else {
        return false;
    };
    cluster.char_indices().any(|(index, flag)| {
        matches!(flag, 'e' | 'f' | 'A' | 'B' | 'C' | 'd' | 'D' | 'm')
            && index + flag.len_utf8() < cluster.len()
    })
}
