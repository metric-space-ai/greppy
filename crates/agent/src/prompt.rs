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
/// one-shot contract. Performance is that same text with the DONE and
/// FINAL ANSWER lines shortened.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentPromptStyle {
    Quality,
    Performance,
}

/// One-shot system prompt for a run.
///
/// Quality is the signed one-shot contract. Its browser section is omitted
/// only when the browser is unavailable. Performance is that same string
/// after [`apply_performance_edits`]: the merge bar and the pull-request
/// report are the only wording changes. Tool sections stay byte-for-byte.
/// `in_place` prepends the same checkout notice in both styles.
pub fn compose_one_shot_prompt(
    style: AgentPromptStyle,
    browser_available: bool,
    in_place: bool,
) -> String {
    let quality = quality_body(browser_available);
    let body = match style {
        AgentPromptStyle::Quality => quality,
        AgentPromptStyle::Performance => apply_performance_edits(&quality)
            .unwrap_or_else(|error| panic!("performance prompt edit failed: {error}")),
    };
    let mut out = String::new();
    if in_place {
        out.push_str(IN_PLACE_NOTICE);
    }
    out.push_str(&body);
    out
}

fn quality_body(browser_available: bool) -> String {
    let text = system_prompt_for_mode(BuiltinPromptMode::OneShot);
    if browser_available {
        text
    } else {
        strip_browser_block(&text)
    }
}

const IN_PLACE_NOTICE: &str = "\
IN-PLACE RUN: edit the current checkout directly. Ignore later instructions that say you work in a temporary worktree or that the result is a pull request copied back later. Files you write here are the task result. Absolute paths inside this repository are writable. /tmp is usable. HOME, GOPATH and TMPDIR stay as this process found them. Git author identity is unchanged when user.name and user.email are already set; otherwise this process uses greppy-agent <greppy-agent@localhost> for commits it makes.

";

/// Exact quality sentences replaced in the performance prompt, and nothing else.
///
/// HOW A TASK RUNS is left unchanged: every sentence there also states the
/// worktree or snapshot lifecycle, so none is only a pull-request requirement.
const PERFORMANCE_EDITS: &[(&str, &str)] = &[
    (
        "DONE means the pull request can be merged as it is: the task is fully implemented, the relevant tests pass, new behaviour has a test, and the diff contains nothing else (smallest change, in the style of the surrounding code). Do not stop at an analysis. If you are blocked, name the missing dependency; never claim a result you have not seen.",
        "DONE means the task is implemented and works. Do not stop at an analysis. If you are blocked, name the missing dependency; never claim a result you have not seen.",
    ),
    (
        "FINAL ANSWER is the pull request's description: a one-line title; Status: done, partial or blocked; what changed and why (file:line); how it was verified (commands and results); what is open.",
        "FINAL ANSWER: at most 5-8 short lines: what you changed and whether it works.",
    ),
];

/// Apply the performance edits to a quality prompt.
///
/// Fails when a source sentence is missing or no longer unique, so a drifted
/// quality text cannot silently produce a performance prompt.
pub fn apply_performance_edits(quality: &str) -> Result<String, String> {
    let mut text = quality.to_string();
    for (from, to) in PERFORMANCE_EDITS {
        let count = text.matches(from).count();
        if count != 1 {
            return Err(format!(
                "performance edit requires exactly one occurrence of `{from}`, found {count}"
            ));
        }
        text = text.replacen(from, to, 1);
    }
    Ok(text)
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
    fn performance_is_quality_with_only_the_allowed_edits() {
        for browser in [true, false] {
            for in_place in [true, false] {
                let quality = compose_one_shot_prompt(AgentPromptStyle::Quality, browser, in_place);
                let performance =
                    compose_one_shot_prompt(AgentPromptStyle::Performance, browser, in_place);
                let delta = differing_lines(&quality, &performance);
                assert_eq!(
                    delta,
                    PERFORMANCE_EDITS
                        .iter()
                        .map(|(from, to)| (from.to_string(), to.to_string()))
                        .collect::<Vec<_>>(),
                    "browser={browser} in_place={in_place}\n{delta:#?}"
                );
                for line in quality.lines().filter(|line| line.contains("greppy ")) {
                    assert!(
                        performance.contains(line),
                        "missing tool line with browser={browser}: {line}"
                    );
                }
            }
        }
        let with_browser = compose_one_shot_prompt(AgentPromptStyle::Performance, true, false);
        let without = compose_one_shot_prompt(AgentPromptStyle::Performance, false, false);
        assert!(with_browser.contains(browser_prompt()));
        assert!(with_browser.contains("greppy web open"));
        assert!(!without.contains("\nBROWSER:\n"));
        assert!(!without.contains("greppy web open"));
        for heading in [
            "ROUTING:",
            "TERMS:",
            "SEARCH:",
            "NAVIGATE:",
            "READING CODE:",
            "EDIT:",
            "OUTPUT AND SCOPE:",
            "HOW A TASK RUNS",
            "bash-smart -- CMD",
        ] {
            assert!(with_browser.contains(heading), "missing {heading}");
            assert!(
                without.contains(heading),
                "missing {heading} without browser"
            );
        }
        let in_place = compose_one_shot_prompt(AgentPromptStyle::Performance, true, true);
        let quality_in_place = compose_one_shot_prompt(AgentPromptStyle::Quality, true, true);
        assert!(in_place.starts_with(IN_PLACE_NOTICE));
        assert!(quality_in_place.starts_with(IN_PLACE_NOTICE));
        assert!(with_browser
            .contains("DONE means the task is implemented and works. Do not stop at an analysis."));
        assert!(with_browser.contains(
            "FINAL ANSWER: at most 5-8 short lines: what you changed and whether it works."
        ));
        assert!(!with_browser.contains("pull request can be merged"));
        assert!(!with_browser.contains("pull request's description"));
    }

    #[test]
    fn performance_edit_fails_when_a_source_sentence_is_missing() {
        let quality = compose_one_shot_prompt(AgentPromptStyle::Quality, true, false);
        for (from, _) in PERFORMANCE_EDITS {
            let broken = quality.replacen(from, "rewritten", 1);
            let error = apply_performance_edits(&broken).unwrap_err();
            assert!(error.contains("exactly one occurrence"), "{error}");
            assert!(error.contains(&from[..40]), "{error}");
        }
        assert!(apply_performance_edits("no sentences here").is_err());
    }

    fn differing_lines(left: &str, right: &str) -> Vec<(String, String)> {
        let left: Vec<_> = left.lines().collect();
        let right: Vec<_> = right.lines().collect();
        assert_eq!(
            left.len(),
            right.len(),
            "performance changed the line count"
        );
        left.into_iter()
            .zip(right)
            .filter(|(before, after)| before != after)
            .map(|(before, after)| ((*before).to_string(), after.to_string()))
            .collect()
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
