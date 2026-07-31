//! grep and ripgrep compatibility.
//!
//! Split out of `lib.rs`, which had grown to 26,400 lines: the module still
//! reaches every private helper there through `use super::*`, and nothing about
//! the behaviour changes.

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
