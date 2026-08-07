//! grep and ripgrep compatibility.
//!
//! Split out of `lib.rs`, which had grown to 26,400 lines. Input-demand
//! parsing lives beside forwarding because the wrapper must know when grep or
//! ripgrep would wait on stdin before it can safely delegate.

use std::ffi::{OsStr, OsString};
use std::io::IsTerminal;

pub(crate) fn grep_passthrough_args(argv: &[OsString]) -> &[OsString] {
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StdinDemand<'a> {
    None,
    Always(Option<&'a OsStr>),
    WhenNonTerminal(Option<&'a OsStr>),
    Unknown,
}

const GREP_SHORT_NO_VALUE: &str = "EFGPiwxyzvVclLqsrRHhnboaIZTUN";
const GREP_SHORT_WITH_VALUE: &str = "efmABCdD";
const GREP_LONG_NO_VALUE: &[&str] = &[
    "--basic-regexp",
    "--extended-regexp",
    "--fixed-strings",
    "--perl-regexp",
    "--ignore-case",
    "--no-ignore-case",
    "--word-regexp",
    "--line-regexp",
    "--null-data",
    "--invert-match",
    "--version",
    "--help",
    "--byte-offset",
    "--line-number",
    "--line-buffered",
    "--with-filename",
    "--no-filename",
    "--only-matching",
    "--quiet",
    "--silent",
    "--text",
    "--binary",
    "--recursive",
    "--dereference-recursive",
    "--files-with-matches",
    "--files-without-match",
    "--count",
    "--initial-tab",
    "--null",
    "--unix-byte-offsets",
    "--no-group-separator",
];
const GREP_LONG_WITH_VALUE: &[&str] = &[
    "--regexp",
    "--file",
    "--max-count",
    "--after-context",
    "--before-context",
    "--context",
    "--binary-files",
    "--devices",
    "--directories",
    "--include",
    "--exclude",
    "--exclude-from",
    "--exclude-dir",
    "--label",
    "--group-separator",
    "--color",
    "--colour",
];

pub(crate) fn unknown_grep_option(args: &[OsString]) -> Option<String> {
    let mut index = 0usize;
    while index < args.len() {
        let argument = &args[index];
        if argument == "--" {
            return None;
        }
        let Some(text) = argument.to_str() else {
            index += 1;
            continue;
        };
        if !text.starts_with('-') || text == "-" {
            index += 1;
            continue;
        }
        if text.starts_with("--") {
            let (name, attached) = text
                .split_once('=')
                .map_or((text, false), |(name, _)| (name, true));
            if GREP_LONG_NO_VALUE.contains(&name) {
                index += 1;
                continue;
            }
            if GREP_LONG_WITH_VALUE.contains(&name) {
                index += if attached { 1 } else { 2 };
                continue;
            }
            return Some(text.to_string());
        }

        let mut chars = text[1..].char_indices().peekable();
        let mut valid = true;
        let mut consumes_next = false;
        while let Some((_, flag)) = chars.next() {
            if GREP_SHORT_NO_VALUE.contains(flag) {
                continue;
            }
            if GREP_SHORT_WITH_VALUE.contains(flag) {
                consumes_next = chars.peek().is_none();
                break;
            }
            valid = false;
            break;
        }
        if !valid {
            return Some(text.to_string());
        }
        index += if consumes_next { 2 } else { 1 };
    }
    None
}

pub(crate) fn grep_stdin_demand(args: &[OsString]) -> StdinDemand<'_> {
    let mut positionals: Vec<&OsStr> = Vec::new();
    let mut explicit_pattern = false;
    let mut pattern_from_stdin = false;
    let mut recursive = false;
    let mut options = true;
    let mut index = 0usize;

    while index < args.len() {
        let argument = &args[index];
        if options && argument == "--" {
            options = false;
            index += 1;
            continue;
        }
        let Some(text) = argument.to_str() else {
            positionals.push(argument);
            index += 1;
            continue;
        };
        if !options || !text.starts_with('-') || text == "-" {
            positionals.push(argument);
            index += 1;
            continue;
        }
        if text.starts_with("--") {
            let (name, attached) = text
                .split_once('=')
                .map_or((text, None), |(name, value)| (name, Some(value)));
            if matches!(name, "--help" | "--version") {
                return StdinDemand::None;
            }
            if GREP_LONG_NO_VALUE.contains(&name) {
                if matches!(name, "--recursive" | "--dereference-recursive") {
                    recursive = true;
                }
                index += 1;
                continue;
            }
            if GREP_LONG_WITH_VALUE.contains(&name) {
                let value = attached
                    .map(OsStr::new)
                    .or_else(|| args.get(index + 1).map(OsString::as_os_str));
                if value.is_none() {
                    return StdinDemand::None;
                }
                if matches!(name, "--regexp" | "--file") {
                    explicit_pattern = true;
                }
                if name == "--file" && value.is_some_and(|value| value == "-") {
                    pattern_from_stdin = true;
                }
                index += if attached.is_some() { 1 } else { 2 };
                continue;
            }
            return StdinDemand::Unknown;
        }

        let mut consumed_value = false;
        for (offset, flag) in text[1..].char_indices() {
            if flag == 'V' {
                return StdinDemand::None;
            }
            if GREP_SHORT_NO_VALUE.contains(flag) {
                if matches!(flag, 'r' | 'R') {
                    recursive = true;
                }
                continue;
            }
            if !GREP_SHORT_WITH_VALUE.contains(flag) {
                return StdinDemand::Unknown;
            }
            let value_offset = 1 + offset + flag.len_utf8();
            let attached = text.get(value_offset..).filter(|value| !value.is_empty());
            let value = attached
                .map(OsStr::new)
                .or_else(|| args.get(index + 1).map(OsString::as_os_str));
            if value.is_none() {
                return StdinDemand::None;
            }
            if matches!(flag, 'e' | 'f') {
                explicit_pattern = true;
            }
            if flag == 'f' && value.is_some_and(|value| value == "-") {
                pattern_from_stdin = true;
            }
            index += if attached.is_some() { 1 } else { 2 };
            consumed_value = true;
            break;
        }
        if !consumed_value {
            index += 1;
        }
    }

    let (pattern, files): (Option<&OsStr>, &[&OsStr]) = if explicit_pattern {
        (None, &positionals)
    } else {
        let Some((pattern, files)) = positionals.split_first() else {
            // A missing pattern is grep's own finite usage error, not an input wait.
            return StdinDemand::None;
        };
        (Some(*pattern), files)
    };
    if pattern_from_stdin
        || files.iter().any(|file| *file == "-")
        || (files.is_empty() && !recursive)
    {
        StdinDemand::Always(pattern)
    } else {
        StdinDemand::None
    }
}

const RG_LONG_NO_VALUE: &[&str] = &[
    "--ignore-case",
    "--invert-match",
    "--word-regexp",
    "--line-regexp",
    "--count",
    "--files-with-matches",
    "--files-without-match",
    "--only-matching",
    "--quiet",
    "--line-number",
    "--with-filename",
    "--no-filename",
    "--text",
    "--null",
    "--byte-offset",
    "--line-buffered",
    "--no-messages",
    "--fixed-strings",
    "--pcre2",
    "--smart-case",
    "--follow",
    "--case-sensitive",
    "--no-line-number",
    "--heading",
    "--no-heading",
    "--hidden",
    "--no-ignore",
    "--no-ignore-vcs",
    "--no-ignore-parent",
    "--no-ignore-dot",
    "--no-ignore-global",
    "--no-ignore-files",
    "--no-config",
    "--no-require-git",
    "--crlf",
    "--trim",
    "--stats",
    "--binary",
    "--no-mmap",
    "--mmap",
    "--pretty",
    "--one-file-system",
    "--block-buffered",
    "--no-unicode",
    "--unicode",
    "--files",
    "--type-list",
    "--json",
    "--vimgrep",
    "--column",
    "--multiline",
    "--multiline-dotall",
    "--search-zip",
    "--passthru",
    "--count-matches",
    "--type-clear",
    "--null-data",
    "--help",
    "--version",
];
const RG_LONG_WITH_VALUE: &[&str] = &[
    "--max-count",
    "--after-context",
    "--before-context",
    "--context",
    "--color",
    "--colors",
    "--regexp",
    "--file",
    "--glob",
    "--iglob",
    "--type",
    "--type-not",
    "--sort",
    "--sortr",
    "--threads",
    "--max-columns",
    "--max-filesize",
    "--regex-size-limit",
    "--dfa-size-limit",
    "--ignore-file",
    "--context-separator",
    "--field-context-separator",
    "--field-match-separator",
    "--hyperlink-format",
    "--engine",
    "--replace",
    "--encoding",
    "--max-depth",
    "--type-add",
    "--pre",
    "--pre-glob",
];
const RG_SHORT_NO_VALUE: &str = "ivwxlcoqnHaIFPSL0NsupUz";
const RG_SHORT_WITH_VALUE: &str = "efgtTABCMjmrdE";

pub(crate) fn rg_stdin_demand(args: &[OsString]) -> StdinDemand<'_> {
    let mut positionals: Vec<&OsStr> = Vec::new();
    let mut explicit_pattern = false;
    let mut pattern_from_stdin = false;
    let mut options = true;
    let mut index = 0usize;

    while index < args.len() {
        let argument = &args[index];
        if options && argument == "--" {
            options = false;
            index += 1;
            continue;
        }
        let Some(text) = argument.to_str() else {
            positionals.push(argument);
            index += 1;
            continue;
        };
        if !options || !text.starts_with('-') || text == "-" {
            positionals.push(argument);
            index += 1;
            continue;
        }
        if text.starts_with("--") {
            let (name, attached) = text
                .split_once('=')
                .map_or((text, None), |(name, value)| (name, Some(value)));
            if matches!(name, "--help" | "--version" | "--files" | "--type-list") {
                return StdinDemand::None;
            }
            if RG_LONG_NO_VALUE.contains(&name) {
                index += 1;
                continue;
            }
            if RG_LONG_WITH_VALUE.contains(&name) {
                let value = attached
                    .map(OsStr::new)
                    .or_else(|| args.get(index + 1).map(OsString::as_os_str));
                if value.is_none() {
                    return StdinDemand::None;
                }
                if matches!(name, "--regexp" | "--file") {
                    explicit_pattern = true;
                }
                if name == "--file" && value.is_some_and(|value| value == "-") {
                    pattern_from_stdin = true;
                }
                index += if attached.is_some() { 1 } else { 2 };
                continue;
            }
            return StdinDemand::Unknown;
        }

        let mut consumed_value = false;
        for (offset, flag) in text[1..].char_indices() {
            if RG_SHORT_NO_VALUE.contains(flag) {
                continue;
            }
            if !RG_SHORT_WITH_VALUE.contains(flag) {
                return StdinDemand::Unknown;
            }
            let value_offset = 1 + offset + flag.len_utf8();
            let attached = text.get(value_offset..).filter(|value| !value.is_empty());
            let value = attached
                .map(OsStr::new)
                .or_else(|| args.get(index + 1).map(OsString::as_os_str));
            if value.is_none() {
                return StdinDemand::None;
            }
            if matches!(flag, 'e' | 'f') {
                explicit_pattern = true;
            }
            if flag == 'f' && value.is_some_and(|value| value == "-") {
                pattern_from_stdin = true;
            }
            index += if attached.is_some() { 1 } else { 2 };
            consumed_value = true;
            break;
        }
        if !consumed_value {
            index += 1;
        }
    }

    let (pattern, paths): (Option<&OsStr>, &[&OsStr]) = if explicit_pattern {
        (None, &positionals)
    } else {
        let Some((pattern, paths)) = positionals.split_first() else {
            return StdinDemand::None;
        };
        (Some(*pattern), paths)
    };
    if pattern_from_stdin || paths.iter().any(|path| *path == "-") {
        StdinDemand::Always(pattern)
    } else if paths.is_empty() {
        StdinDemand::WhenNonTerminal(pattern)
    } else {
        StdinDemand::None
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StdinAvailability {
    Terminal,
    Data,
    Empty,
    Unknown,
}

fn stdin_availability() -> StdinAvailability {
    if std::io::stdin().is_terminal() {
        return StdinAvailability::Terminal;
    }
    stdin_availability_nonterminal()
}

#[cfg(unix)]
fn stdin_availability_nonterminal() -> StdinAvailability {
    const PRODUCER_STARTUP_GRACE_MS: libc::c_int = 250;
    let mut descriptor = libc::pollfd {
        fd: libc::STDIN_FILENO,
        events: libc::POLLIN,
        revents: 0,
    };
    // A short readiness wait prevents a scheduler race from rejecting a real
    // pipeline while still bounding the no-writer-data case that exhausted an
    // agent's entire task deadline.
    let ready = unsafe { libc::poll(&mut descriptor, 1, PRODUCER_STARTUP_GRACE_MS) };
    if ready < 0 {
        return StdinAvailability::Unknown;
    }
    if ready == 0 {
        return StdinAvailability::Empty;
    }
    let mut bytes: libc::c_int = 0;
    // FIONREAD distinguishes buffered bytes from EOF without consuming the
    // first byte, so the delegated program still receives stdin byte-for-byte.
    if unsafe { libc::ioctl(libc::STDIN_FILENO, libc::FIONREAD, &mut bytes) } == 0 {
        return if bytes > 0 {
            StdinAvailability::Data
        } else {
            StdinAvailability::Empty
        };
    }
    if descriptor.revents & libc::POLLHUP != 0 {
        StdinAvailability::Empty
    } else if descriptor.revents & libc::POLLIN != 0 {
        // An input type that is readable but does not support FIONREAD may
        // contain real data; preserving passthrough is safer than guessing.
        StdinAvailability::Data
    } else {
        StdinAvailability::Unknown
    }
}

#[cfg(windows)]
fn stdin_availability_nonterminal() -> StdinAvailability {
    use std::os::windows::io::AsRawHandle;
    let handle = std::io::stdin().as_raw_handle() as windows_sys::Win32::Foundation::HANDLE;
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(250);
    loop {
        let mut bytes = 0u32;
        // PeekNamedPipe measures buffered bytes without consuming them,
        // preserving the byte-exact stream while detecting the idle anonymous
        // pipes used by agent runners on Windows.
        let readable = unsafe {
            windows_sys::Win32::System::Pipes::PeekNamedPipe(
                handle,
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
                &mut bytes,
                std::ptr::null_mut(),
            )
        };
        if readable == 0 {
            if unsafe { windows_sys::Win32::Foundation::GetLastError() }
                == windows_sys::Win32::Foundation::ERROR_BROKEN_PIPE
            {
                return StdinAvailability::Empty;
            }
            // Disk handles are not named pipes; preserving passthrough is safer
            // than consuming a byte from a handle that cannot be rewound reliably.
            return StdinAvailability::Unknown;
        }
        if bytes > 0 {
            return StdinAvailability::Data;
        }
        if std::time::Instant::now() >= deadline {
            return StdinAvailability::Empty;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

#[cfg(not(any(unix, windows)))]
fn stdin_availability_nonterminal() -> StdinAvailability {
    // Platforms without a non-consuming readiness API keep the original
    // passthrough rather than eating a byte that cannot be put back exactly.
    StdinAvailability::Unknown
}

pub(crate) fn missing_stdin_message(demand: StdinDemand<'_>, tool: &str) -> Option<String> {
    let pattern = match (demand, stdin_availability()) {
        (StdinDemand::None | StdinDemand::Unknown, _) => return None,
        (StdinDemand::WhenNonTerminal(_), StdinAvailability::Terminal) => return None,
        (
            StdinDemand::Always(pattern) | StdinDemand::WhenNonTerminal(pattern),
            StdinAvailability::Data,
        ) => {
            let _ = pattern;
            return None;
        }
        (
            StdinDemand::Always(pattern) | StdinDemand::WhenNonTerminal(pattern),
            StdinAvailability::Empty | StdinAvailability::Terminal,
        ) => pattern,
        (_, StdinAvailability::Unknown) => return None,
    };

    let mut message = format!(
        "status: missing_input\nmessage: {tool} needs a file/path argument or data on stdin; stdin has no data\nnext: pass a file/path, or pipe data into this command"
    );
    if let Some(pattern) = pattern {
        let path = std::path::Path::new(pattern);
        if path.is_dir() {
            let shown = pattern.to_string_lossy();
            message.push_str(&format!(
                "\nnext: `{shown}` is an existing directory; if you meant to warm its code graph, run `greppy index {shown}`"
            ));
        }
    }
    Some(message)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn os(arguments: &[&str]) -> Vec<OsString> {
        arguments.iter().map(OsString::from).collect()
    }

    #[test]
    fn grep_input_demand_tracks_patterns_files_and_stdin_operands() {
        assert!(matches!(
            grep_stdin_demand(&os(&["needle"])),
            StdinDemand::Always(Some(pattern)) if pattern == "needle"
        ));
        assert_eq!(
            grep_stdin_demand(&os(&["needle", "file.rs"])),
            StdinDemand::None
        );
        assert!(matches!(
            grep_stdin_demand(&os(&["-ne", "needle"])),
            StdinDemand::Always(None)
        ));
        assert_eq!(
            grep_stdin_demand(&os(&["-ne", "needle", "file.rs"])),
            StdinDemand::None
        );
        assert_eq!(
            grep_stdin_demand(&os(&["-eneedle", "file.rs"])),
            StdinDemand::None
        );
        assert!(matches!(
            grep_stdin_demand(&os(&["-f", "-", "file.rs"])),
            StdinDemand::Always(None)
        ));
        assert!(matches!(
            grep_stdin_demand(&os(&["needle", "file.rs", "-"])),
            StdinDemand::Always(Some(pattern)) if pattern == "needle"
        ));
        assert_eq!(grep_stdin_demand(&os(&["-R", "needle"])), StdinDemand::None);
        assert_eq!(
            grep_stdin_demand(&os(&["--recursive", "needle"])),
            StdinDemand::None
        );
        assert_eq!(grep_stdin_demand(&os(&["-n"])), StdinDemand::None);
    }

    #[test]
    fn ripgrep_input_demand_preserves_terminal_default_and_explicit_stdin() {
        assert!(matches!(
            rg_stdin_demand(&os(&["needle"])),
            StdinDemand::WhenNonTerminal(Some(pattern)) if pattern == "needle"
        ));
        assert_eq!(
            rg_stdin_demand(&os(&["-n", "needle", "src"])),
            StdinDemand::None
        );
        assert!(matches!(
            rg_stdin_demand(&os(&["-e", "needle"])),
            StdinDemand::WhenNonTerminal(None)
        ));
        assert!(matches!(
            rg_stdin_demand(&os(&["-m1", "needle"])),
            StdinDemand::WhenNonTerminal(Some(pattern)) if pattern == "needle"
        ));
        assert!(matches!(
            rg_stdin_demand(&os(&["needle", "-"])),
            StdinDemand::Always(Some(pattern)) if pattern == "needle"
        ));
        assert_eq!(rg_stdin_demand(&os(&["--files"])), StdinDemand::None);
    }
}
