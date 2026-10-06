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
