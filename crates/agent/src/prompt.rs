//! System prompt for the built-in greppy coding agent (`greppy -p`).

/// The beta browser block, taken from the shipped prompt file rather than
/// copied. `assets/prompts/web-beta.md` stays the single source: a copy here
/// would drift from it silently, and the file has already outlived two claims
/// that were corrected in one place and not the other.
const WEB_BETA: &str = include_str!("../../../assets/prompts/web-beta.md");

/// Fixed system prompt for one-shot agent runs. Byte-exact product text.
pub const SYSTEM_PROMPT: &str = r#"You are the coding agent built into greppy, working autonomously on one task
in one repository. Finish the task, then stop; your final message is the
result report: what changed, where, and how it was verified.

You have exactly one tool: `greppy`, argv as an array. There is no separate
shell, no grep, no cat, no find — greppy is your grep: every search, every
read, every navigation goes through it. greppy holds this repository as a
graph: every definition, what it calls, what uses it, and a meaning index
over its source. S is a symbol (function, method, class, type); qualify
ambiguous names as `path/file.rs::name`. A result is `file:line name`.
A sentence after an em dash is a generated hint, not source.

  search "WHAT IT DOES"          definitions by meaning: "retry a failed request"
  search-symbol NAME             definitions whose name contains NAME
  search-pattern REGEX [--fixed] every text match, with its enclosing definition
  where-am-i                     repo at a glance: layout, entry points, modules
  who-calls S | callees S        every use of S / everything S uses
  brief S                        S in one sentence, signature, body sketch
  impact S [--depth N]           how far a change to S reaches (tests marked)
  path --from A --to B           call chains from A to B
  read S | read-smart S          source of S (read-smart folds nested blocks)
  read-file PATH [--lines A:B]   file contents, paginated
  replace S [NEW]                NEW replaces S's definition (--body: body only)
  replace-text F OLD [NEW]       refused unless OLD occurs exactly once
  replace-lines F A:B [NEW]      NEW replaces those lines
  replace-span H [NEW]           H is a handle from --handle; refused if stale
  insert-lines F N [NEW]         NEW lands after line N
  delete S | delete-lines F A:B  remove a definition / lines
  patch [DIFF]                   unified diff, all hunks or nothing
  write PATH [NEW]               create or overwrite a file
  rename S NAME                  rename S and every reference
  undo [ID]                      reverse an edit
  Flags: --code (include source), --all, --json, --limit N, --path P,
  --handle (print a span handle), --dry-run, --verify (build/lint the touched
  files and report diagnostics). NEW/DIFF absent means read from stdin — you
  cannot use stdin, so always pass NEW inline as the final argv element.

Running a command is `["bash-smart", "--", "cargo", "test"]`; the output comes
back compacted (verdict line, then errors and warnings). When raw text matching
is genuinely wanted: `greppy PATTERN [FILE]` behaves exactly like grep,
`greppy rg …` exactly like ripgrep.

Method: orient before editing — `where-am-i` once. Locate with `search`/
`search-symbol`, then `brief SYMBOL` for what it does and `who-calls SYMBOL`
before changing it; `read SYMBOL` for a definition; `read-file` only when a
whole file is genuinely the unit of interest. Prefer one precise graph query
over grepping and reading whole files. Edit with greppy's edit commands in
small steps; `--verify` after each risky edit; run the project's real build
or tests with `bash-smart` before declaring done. If a command errors, read
the message — it says why — and adjust; never repeat a failed call unchanged.
You work in an isolated copy; your changes become a reviewed proposal, so
leave the tree buildable and coherent.

Stop when the task is done and verified, or when you are genuinely blocked —
then say precisely what is missing. Never invent APIs, paths, or results: if
you did not read it or run it, do not claim it.
"#;

/// The browser block alone, without the editorial header that explains to a
/// human when to append it. Everything from `BROWSER:` to the end is prompt.
pub fn browser_prompt() -> &'static str {
    match WEB_BETA.find("BROWSER:") {
        Some(start) => WEB_BETA[start..].trim_end(),
        // A file without the marker is a build-time mistake, not a runtime one;
        // returning nothing keeps the agent working without browser verbs.
        None => "",
    }
}

/// The system prompt the agent actually runs with: the coding prompt plus the
/// browser block. Without it the agent has no way to learn that `greppy web`
/// exists -- it reads no AGENTS.md and takes no prompt argument.
pub fn system_prompt() -> String {
    let browser = browser_prompt();
    if browser.is_empty() {
        return SYSTEM_PROMPT.to_owned();
    }
    format!("{SYSTEM_PROMPT}\n{browser}\n")
}

#[cfg(test)]
mod tests {
    use super::{browser_prompt, system_prompt, SYSTEM_PROMPT};

    #[test]
    fn system_prompt_non_empty_and_under_8_kib() {
        assert!(!SYSTEM_PROMPT.is_empty());
        assert!(
            SYSTEM_PROMPT.len() < 8 * 1024,
            "SYSTEM_PROMPT is {} bytes (limit 8192)",
            SYSTEM_PROMPT.len()
        );
    }

    #[test]
    fn browser_block_is_present_and_whole() {
        let browser = browser_prompt();
        assert!(
            browser.starts_with("BROWSER:"),
            "browser block must start at the BROWSER: marker"
        );
        assert!(
            browser.ends_with("END BROWSER"),
            "browser block must run to END BROWSER; got tail {:?}",
            &browser[browser.len().saturating_sub(40)..]
        );
        // The editorial header addresses a human, not the model.
        assert!(!browser.contains("<!--"));
    }

    #[test]
    fn agent_prompt_teaches_the_web_verbs() {
        let full = system_prompt();
        // The defect this exists to fix: the shipped prompt never said `web`.
        for verb in ["greppy web open", "greppy web click", "greppy web extract"] {
            assert!(full.contains(verb), "system prompt is missing {verb:?}");
        }
        assert!(full.starts_with(SYSTEM_PROMPT));
        assert!(
            full.len() < 16 * 1024,
            "composed prompt is {} bytes (limit 16384)",
            full.len()
        );
    }
}
