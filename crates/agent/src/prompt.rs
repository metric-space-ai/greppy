//! One signed contract, rendered for the actual tool interface before embedding.
pub use crate::prompt_contract::render::BuiltinPromptMode;
pub const PUBLIC_PROMPT: &str = include_str!(concat!(env!("OUT_DIR"), "/canonical-prompt.md"));
const ADAPTER: &str = include_str!(concat!(env!("OUT_DIR"), "/agent-adapter.md"));
pub const SYSTEM_PROMPT: &str = include_str!(concat!(env!("OUT_DIR"), "/one-shot-prompt.md"));
const INTERACTIVE_PROMPT: &str = include_str!(concat!(env!("OUT_DIR"), "/interactive-prompt.md"));
const SERVE_PROMPT: &str = include_str!(concat!(env!("OUT_DIR"), "/serve-prompt.md"));
const ACP_PROMPT: &str = include_str!(concat!(env!("OUT_DIR"), "/acp-prompt.md"));

pub fn browser_prompt() -> &'static str {
    let start = SYSTEM_PROMPT
        .find("\nBROWSER:")
        .expect("built-in browser block")
        + 1;
    let tail = &SYSTEM_PROMPT[start..];
    let end = tail.find("END BROWSER").expect("complete browser block") + "END BROWSER".len();
    &tail[..end]
}

pub fn system_prompt_for_mode(mode: BuiltinPromptMode) -> String {
    match mode {
        BuiltinPromptMode::OneShot => SYSTEM_PROMPT,
        BuiltinPromptMode::Interactive => INTERACTIVE_PROMPT,
        BuiltinPromptMode::Serve => SERVE_PROMPT,
        BuiltinPromptMode::Acp => ACP_PROMPT,
    }
    .to_owned()
}

pub fn system_prompt() -> String {
    system_prompt_for_mode(BuiltinPromptMode::OneShot)
}

/// `greppy -p` prompt style. Quality is the default and keeps the signed
/// one-shot contract. Performance is the short bench/headless prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentPromptStyle {
    Quality,
    Performance,
}

/// One-shot system prompt for a run.
///
/// Quality keeps the signed contract, including the merge bar and the
/// pull-request report. Its browser section is omitted only when the browser
/// is unavailable. Performance is a short operating note: a 5–8 line report,
/// no pull-request requirement, and the browser verbs only when they work.
/// `in_place` is independent of the style and only adds the checkout notice.
pub fn compose_one_shot_prompt(
    style: AgentPromptStyle,
    browser_available: bool,
    in_place: bool,
) -> String {
    let body = match style {
        AgentPromptStyle::Quality => {
            let text = system_prompt_for_mode(BuiltinPromptMode::OneShot);
            if browser_available {
                text
            } else {
                strip_browser_block(&text)
            }
        }
        AgentPromptStyle::Performance => performance_prompt(browser_available),
    };
    let mut out = String::new();
    if in_place {
        out.push_str(IN_PLACE_NOTICE);
    }
    out.push_str(&body);
    out
}

const IN_PLACE_NOTICE: &str = "\
IN-PLACE RUN: edit the current checkout directly. Ignore later instructions that say you work in a temporary worktree or that the result is a pull request copied back later. Files you write here are the task result. Absolute paths inside this repository are writable. /tmp is usable. HOME, GOPATH and TMPDIR stay as this process found them. Git author identity is unchanged when user.name and user.email are already set; otherwise this process uses greppy-agent <greppy-agent@localhost> for commits it makes.

";

fn performance_prompt(browser_available: bool) -> String {
    let mut text = String::from(
        "\
You are greppy, a coding agent in performance mode. Finish the task. Decide details yourself; nobody can answer questions.

WORK
One tool: greppy, called with an argv array. Example: [\"search-symbol\", \"NAME\"] or [\"bash-smart\", \"--\", \"cargo\", \"test\"].
Batch independent steps in one call: {\"batch\": [[\"search-symbol\", \"NAME\"], [\"read\", \"NAME\"]]}. A failed edit stops the rest of that batch.
Do not run greppy index unless a result says the index is stale. --root is rejected; the tool already runs in the repository.

FIND
search-symbol NAME, search \"what it does\", who-calls NAME, callees NAME, brief NAME, impact NAME. Ask the graph before rg.

READ
read SYMBOL prints the definition. read path/file.ext::Type::method falls back to that file and the trailing name when the symbol index misses.
read-file PATH prints about 250 lines. Continue with read-file PATH --lines A:B.

EDIT
replace, replace-text, replace-lines, write, patch, insert-lines, delete-lines. The new text is the last argument. Trust the edit receipt.

SHELL
bash-smart -- CMD. cd, &&, |, ; and redirects may be separate argv elements; they run as one shell command.

DONE
Make the change. Run the project's relevant test when one exists. Stop when the task is done or you are blocked on something you cannot change.

FINAL ANSWER
5 to 8 lines: what changed, the command that verified it and its result, and anything still open. Do not write a pull-request description.
",
    );
    if browser_available {
        text.push_str(
            "\
BROWSER
The browser is available. greppy web open URL, then greppy web observe, greppy web click TARGET, greppy web fill TARGET VALUE. Chain known steps with :: . If a call says the browser is unavailable, stop using it.
",
        );
    }
    text
}

/// Drop the signed `BROWSER` … `END BROWSER` section. The rest of the contract
/// stays byte-for-byte, including the following sections.
pub fn strip_browser_block(text: &str) -> String {
    let Some(start) = text.find("\nBROWSER:\n") else {
        return text.to_string();
    };
    let after = &text[start + 1..];
    let Some(end_rel) = after.find("END BROWSER") else {
        return text.to_string();
    };
    let end = start + 1 + end_rel + "END BROWSER".len();
    let mut out = String::new();
    out.push_str(text[..start].trim_end());
    out.push('\n');
    let rest = text[end..].trim_start_matches('\n');
    if !rest.is_empty() {
        out.push('\n');
        out.push_str(rest);
    }
    out
}

pub fn export_prompt(external: bool) -> String {
    if external {
        format!("greppy {}\n\n{PUBLIC_PROMPT}", env!("CARGO_PKG_VERSION"))
    } else {
        system_prompt()
    }
}

pub fn prompt_metadata_for_mode(mode: &str) -> serde_json::Value {
    let builtin = match mode {
        "one-shot" => Some(BuiltinPromptMode::OneShot),
        "interactive" => Some(BuiltinPromptMode::Interactive),
        "serve" => Some(BuiltinPromptMode::Serve),
        "acp" => Some(BuiltinPromptMode::Acp),
        "external" => None,
        _ => panic!("unsupported prompt mode: {mode}"),
    };
    let text = builtin
        .map(system_prompt_for_mode)
        .unwrap_or_else(|| export_prompt(true));
    serde_json::json!({
        "schema": "greppy.prompt.v1", "version": env!("CARGO_PKG_VERSION"),
        "source": "AGENTS.md", "mode": mode,
        "prompt_sha256": crate::prompt_contract::digest(PUBLIC_PROMPT.as_bytes()),
        "adapter_sha256": builtin.map(|_| crate::prompt_contract::digest(ADAPTER.as_bytes())),
        "rendered_sha256": crate::prompt_contract::digest(text.as_bytes()), "prompt": text
    })
}

pub fn prompt_metadata(external: bool) -> serde_json::Value {
    prompt_metadata_for_mode(if external { "external" } else { "one-shot" })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn effective_modes_match_owner_signed_composition() {
        for mode in crate::prompt_contract::render::BUILTIN_MODES {
            let text = system_prompt_for_mode(mode);
            crate::prompt_contract::verify_bytes(
                mode.name(),
                text.as_bytes(),
                crate::prompt_contract::approved_rendered_sha256(mode),
            )
            .unwrap();
            assert_eq!(
                text,
                crate::prompt_contract::render::render_builtin(PUBLIC_PROMPT, ADAPTER, mode)
            );
            assert_eq!(
                prompt_metadata_for_mode(mode.name())["rendered_sha256"],
                crate::prompt_contract::digest(text.as_bytes())
            );
            for forbidden in [
                "\nAGENT:",
                "\nPROMPT:",
                "\nCHAIN:",
                "--root DIR",
                "Omitted NEW or DIFF is read from stdin",
                "--value-stdin",
                "index --agent-worktree",
                "EXECUTION:",
                "greppy web match",
            ] {
                assert!(
                    !text.contains(forbidden),
                    "{} advertises {forbidden}",
                    mode.name()
                );
            }
            if mode != BuiltinPromptMode::Acp {
                assert!(text.starts_with(
                    "You are greppy, a coding agent. You never work in the user's checkout:"
                ));
                let mode_count = text
                    .lines()
                    .filter(|line| {
                        line.starts_with("-p:")
                            || line.starts_with("TUI:")
                            || line.starts_with("serve:")
                    })
                    .count();
                assert_eq!(mode_count, 1);
                assert!(text.contains(browser_prompt()));
            } else {
                assert!(!text.contains("temporary worktree"));
                assert!(!text.contains("BROWSER:"));
            }
        }
    }
    #[test]
    fn external_contract_is_unchanged_and_versioned() {
        crate::prompt_contract::verify_bytes(
            "AGENTS.md",
            PUBLIC_PROMPT.as_bytes(),
            crate::prompt_contract::APPROVED_SHA256,
        )
        .unwrap();
        assert_eq!(
            export_prompt(true),
            format!("greppy {}\n\n{PUBLIC_PROMPT}", env!("CARGO_PKG_VERSION"))
        );
        assert_eq!(system_prompt(), SYSTEM_PROMPT);
    }
    #[test]
    fn quality_keeps_the_contract_and_drops_the_browser_only_when_unavailable() {
        let with_browser = compose_one_shot_prompt(AgentPromptStyle::Quality, true, false);
        assert!(with_browser.contains(browser_prompt()));
        assert!(with_browser.contains("DONE means"));
        assert!(with_browser.contains("FINAL ANSWER is the pull request"));
        let without = compose_one_shot_prompt(AgentPromptStyle::Quality, false, false);
        assert!(!without.contains("\nBROWSER:\n"));
        assert!(!without.contains("greppy web open"));
        assert!(without.contains("DONE means"));
        assert!(without.contains("SEARCH:"));
    }

    #[test]
    fn performance_prompt_is_short_and_skips_the_pull_request_report() {
        let text = compose_one_shot_prompt(AgentPromptStyle::Performance, false, false);
        assert!(text.contains("5 to 8 lines"));
        assert!(!text.contains("pull request's description"));
        assert!(!text.contains("greppy web"));
        assert!(
            text.len() / 4 < 2_000,
            "performance prompt tokens ~= {}",
            text.len() / 4
        );
        let with_browser = compose_one_shot_prompt(AgentPromptStyle::Performance, true, false);
        assert!(with_browser.contains("greppy web open"));
        assert!(with_browser.len() / 4 < 2_000);
        let in_place = compose_one_shot_prompt(AgentPromptStyle::Performance, false, true);
        assert!(in_place.starts_with("IN-PLACE RUN:"));
        assert!(in_place.contains("5 to 8 lines"));
    }

    #[test]
    fn changed_rendered_prompt_is_refused() {
        let mut text = system_prompt();
        text.push_str("unsigned addition");
        assert!(crate::prompt_contract::verify_bytes(
            "one-shot",
            text.as_bytes(),
            crate::prompt_contract::approved_rendered_sha256(BuiltinPromptMode::OneShot)
        )
        .is_err());
    }
}
