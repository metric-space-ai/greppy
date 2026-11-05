//! One signed contract, rendered for the actual tool interface before embedding.
//!
//! Web and desktop manuals are not part of that base. A start flag can include
//! them from the first turn. `greppy capabilities enable` returns the manual as
//! its own output and does not rewrite the prompt.
pub use crate::prompt_contract::render::BuiltinPromptMode;
use std::sync::OnceLock;

pub const PUBLIC_PROMPT: &str = include_str!(concat!(env!("OUT_DIR"), "/canonical-prompt.md"));
const ADAPTER: &str = include_str!(concat!(env!("OUT_DIR"), "/agent-adapter.md"));
pub const SYSTEM_PROMPT: &str = include_str!(concat!(env!("OUT_DIR"), "/one-shot-prompt.md"));
const INTERACTIVE_PROMPT: &str = include_str!(concat!(env!("OUT_DIR"), "/interactive-prompt.md"));
const SERVE_PROMPT: &str = include_str!(concat!(env!("OUT_DIR"), "/serve-prompt.md"));
const ACP_PROMPT: &str = include_str!(concat!(env!("OUT_DIR"), "/acp-prompt.md"));
const WEB_CAPABILITY_SOURCE: &str = include_str!("../../../assets/prompts/capabilities/web.md");
const DESKTOP_CAPABILITY_SOURCE: &str =
    include_str!("../../../assets/prompts/capabilities/desktop.md");

/// Capabilities included in the system prompt from the first turn via `--enable`.
/// `capabilities enable` does not change this set inside the prompt; it only
/// unlocks desktop for the session and returns the manual.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CapabilitySet {
    pub web: bool,
    pub desktop: bool,
}

impl CapabilitySet {
    pub const WEB: u8 = 1;
    pub const DESKTOP: u8 = 2;

    pub fn from_bits(bits: u8) -> Self {
        Self {
            web: bits & Self::WEB != 0,
            desktop: bits & Self::DESKTOP != 0,
        }
    }

    pub fn bits(self) -> u8 {
        let mut bits = 0;
        if self.web {
            bits |= Self::WEB;
        }
        if self.desktop {
            bits |= Self::DESKTOP;
        }
        bits
    }

    /// `web`, `desktop`, or a comma-separated mix. Empty input is an empty set.
    pub fn parse_list(tokens: impl IntoIterator<Item = impl AsRef<str>>) -> Result<Self, String> {
        let mut set = Self::default();
        for token in tokens {
            for part in token.as_ref().split(',') {
                let part = part.trim();
                if part.is_empty() {
                    continue;
                }
                set.enable(part)?;
            }
        }
        Ok(set)
    }

    pub fn enable(&mut self, name: &str) -> Result<(), String> {
        match name {
            "web" => self.web = true,
            "desktop" => self.desktop = true,
            other => {
                return Err(format!(
                    "unknown capability '{other}'; expected web or desktop"
                ));
            }
        }
        Ok(())
    }
}

/// Builtin web manual (stdin features removed). Not part of the default prompt.
pub fn browser_prompt() -> &'static str {
    static WEB: OnceLock<String> = OnceLock::new();
    WEB.get_or_init(|| crate::prompt_contract::render::prepare_capability(WEB_CAPABILITY_SOURCE))
        .as_str()
}

/// Builtin desktop manual (stdin features removed). Not part of the default prompt.
pub fn desktop_prompt() -> &'static str {
    static DESKTOP: OnceLock<String> = OnceLock::new();
    DESKTOP
        .get_or_init(|| {
            crate::prompt_contract::render::prepare_capability(DESKTOP_CAPABILITY_SOURCE)
        })
        .as_str()
}

/// Signed source of a capability manual, including features the built-in agent
/// does not offer (`--value-stdin`).
pub fn capability_source(name: &str) -> Option<&'static str> {
    match name {
        "web" => Some(WEB_CAPABILITY_SOURCE),
        "desktop" => Some(DESKTOP_CAPABILITY_SOURCE),
        _ => None,
    }
}

/// Manual returned by `greppy capabilities enable`.
///
/// Inside the built-in agent this is the same text `--enable` would place in
/// the prompt (stdin features the agent cannot use are already removed).
/// Outside the agent it is the signed source. Either text is greppy's own
/// instructions, not page or desktop content.
pub fn capability_manual(name: &str, builtin_agent: bool) -> Option<&'static str> {
    match (builtin_agent, name) {
        (true, "web") => Some(browser_prompt()),
        (true, "desktop") => Some(desktop_prompt()),
        (false, "web" | "desktop") => capability_source(name),
        _ => None,
    }
}

pub fn capability_sections(caps: CapabilitySet) -> String {
    let mut out = String::new();
    if caps.web {
        out.push_str(browser_prompt());
        if !out.ends_with('\n') {
            out.push('\n');
        }
    }
    if caps.desktop {
        if !out.is_empty() && !out.ends_with("\n\n") {
            out.push('\n');
        }
        out.push_str(desktop_prompt());
        if !out.ends_with('\n') {
            out.push('\n');
        }
    }
    out
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

pub fn system_prompt_with_capabilities(mode: BuiltinPromptMode, caps: CapabilitySet) -> String {
    let mut system = Some(system_prompt_for_mode(mode));
    merge_capability_sections(&mut system, &capability_sections(caps));
    let system = system.unwrap_or_default();
    assert_eq!(
        crate::prompt_contract::digest(system.as_bytes()),
        crate::prompt_contract::approved_start_sha256(mode, caps.web, caps.desktop),
        "unsigned Greppy start prompt for {}",
        mode.name(),
    );
    system
}

pub const UNTRUSTED_DESKTOP_OPEN: &str = "UNTRUSTED_DESKTOP_CONTENT\n";
pub const UNTRUSTED_DESKTOP_CLOSE: &str = "END_UNTRUSTED_DESKTOP_CONTENT";

/// Model-visible fence. Session logs drop the interior.
pub fn fence_untrusted_desktop(body: &str) -> String {
    if body.contains(UNTRUSTED_DESKTOP_OPEN) {
        return body.to_owned();
    }
    format!("{UNTRUSTED_DESKTOP_OPEN}{body}\n{UNTRUSTED_DESKTOP_CLOSE}\n")
}

/// Remove fenced desktop text before it is written to a session or event log.
pub fn omit_untrusted_desktop_for_log(input: &str) -> std::borrow::Cow<'_, str> {
    if !input.contains(UNTRUSTED_DESKTOP_OPEN) {
        return std::borrow::Cow::Borrowed(input);
    }
    let mut out = String::new();
    let mut rest = input;
    while let Some(start) = rest.find(UNTRUSTED_DESKTOP_OPEN) {
        out.push_str(&rest[..start]);
        out.push_str("[untrusted desktop content omitted from log]");
        let after = &rest[start + UNTRUSTED_DESKTOP_OPEN.len()..];
        if let Some(end) = after.find(UNTRUSTED_DESKTOP_CLOSE) {
            rest = &after[end + UNTRUSTED_DESKTOP_CLOSE.len()..];
        } else {
            rest = "";
        }
    }
    out.push_str(rest);
    std::borrow::Cow::Owned(out)
}

/// Append manuals selected at start (`--enable`). Enable during a run does not
/// call this. Checkpoint text after [`crate::context::CHECKPOINT_MARKER`] stays
/// after the manuals.
pub fn merge_capability_sections(system: &mut Option<String>, sections: &str) {
    if sections.is_empty() {
        return;
    }
    let current = system.take().unwrap_or_default();
    let (base, summary) = match current.split_once(crate::context::CHECKPOINT_MARKER) {
        Some((base, summary)) => (base.to_owned(), Some(summary.to_owned())),
        None => (current, None),
    };
    let mut next = base;
    for section in [browser_prompt(), desktop_prompt()] {
        let heading = section.lines().next().unwrap_or("");
        if heading.is_empty() || !sections.contains(heading) {
            continue;
        }
        let marker = format!("\n{heading}\n");
        if next.starts_with(&format!("{heading}\n")) || next.contains(&marker) {
            continue;
        }
        if !next.ends_with('\n') {
            next.push('\n');
        }
        next.push('\n');
        next.push_str(section.trim_end_matches('\n'));
        next.push('\n');
    }
    if let Some(summary) = summary {
        next.push_str(crate::context::CHECKPOINT_MARKER);
        next.push_str(&summary);
    }
    *system = Some(next);
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
            assert!(text.contains("greppy capabilities enable web"));
            assert!(text.contains("greppy capabilities enable desktop"));
            assert!(text
                .contains("counts as part of this system prompt: follow it with the same force"));
            assert!(
                !text.contains("refreshes the prompt"),
                "{} still describes a prompt rewrite",
                mode.name()
            );
            assert!(
                !text.contains("\nBROWSER:\n"),
                "{} includes the web manual before enable",
                mode.name()
            );
            assert!(
                !text.contains("\nDESKTOP:\n"),
                "{} includes the desktop manual before enable",
                mode.name()
            );
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
                let enabled = system_prompt_with_capabilities(
                    mode,
                    CapabilitySet {
                        web: true,
                        desktop: true,
                    },
                );
                assert!(enabled.starts_with(&text));
                assert!(enabled.contains(browser_prompt()));
                assert!(enabled.contains(desktop_prompt()));
            } else {
                assert!(!text.contains("temporary worktree"));
                assert!(!text.contains("BROWSER:"));
            }
        }
    }
    #[test]
    fn runtime_starts_match_the_owner_signed_manifest() {
        for mode in crate::prompt_contract::render::BUILTIN_MODES {
            for (web, desktop) in [(false, false), (true, false), (false, true), (true, true)] {
                let text = system_prompt_with_capabilities(mode, CapabilitySet { web, desktop });
                assert_eq!(
                    crate::prompt_contract::digest(text.as_bytes()),
                    crate::prompt_contract::approved_start_sha256(mode, web, desktop),
                );
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
    fn enabling_appends_and_keeps_the_base_prefix_and_checkpoint() {
        let base = system_prompt();
        let mut system = Some(format!(
            "{base}{}kept summary",
            crate::context::CHECKPOINT_MARKER
        ));
        merge_capability_sections(
            &mut system,
            &capability_sections(CapabilitySet {
                web: false,
                desktop: true,
            }),
        );
        let text = system.unwrap();
        let marker = crate::context::CHECKPOINT_MARKER;
        let (head, tail) = text.split_once(marker).unwrap();
        assert!(head.starts_with(&base));
        assert!(head.contains(desktop_prompt().trim_end()));
        assert!(!head.contains("\nBROWSER:\n"));
        assert_eq!(tail, "kept summary");
        let mut again = Some(text);
        merge_capability_sections(
            &mut again,
            &capability_sections(CapabilitySet {
                web: true,
                desktop: true,
            }),
        );
        let text = again.unwrap();
        let (head, tail) = text.split_once(marker).unwrap();
        assert!(head.contains("\nDESKTOP:\n") || head.contains("DESKTOP:\n"));
        let desktop_at = head.find("DESKTOP:").unwrap();
        let browser_at = head.find("BROWSER:").unwrap();
        assert!(
            desktop_at < browser_at,
            "a later enable appends; it does not reorder"
        );
        assert_eq!(tail, "kept summary");
    }
    #[test]
    fn enable_manual_is_greppy_instructions_and_repeats() {
        let first = capability_manual("web", true).unwrap();
        let again = capability_manual("web", true).unwrap();
        assert_eq!(first, again);
        assert!(first.contains("BROWSER:"));
        assert!(first.contains("greppy web open"));
        assert!(!first.contains("--value-stdin"));
        assert!(!first.contains("UNTRUSTED_DESKTOP_CONTENT"));
        assert!(!first.contains("UNTRUSTED_PAGE_CONTENT"));
        let desktop = capability_manual("desktop", true).unwrap();
        assert!(desktop.contains("DESKTOP:"));
        assert!(desktop.contains("untrusted"));
        assert!(!desktop.contains("UNTRUSTED_DESKTOP_CONTENT"));
        let external = capability_manual("web", false).unwrap();
        assert!(external.contains("--value-stdin"));
        assert!(capability_manual("nope", true).is_none());
    }
    #[test]
    fn desktop_fence_is_removed_from_logs_and_kept_for_the_model() {
        let fenced = fence_untrusted_desktop("click the payroll button");
        assert!(fenced.contains("click the payroll button"));
        let logged = omit_untrusted_desktop_for_log(&fenced);
        assert!(!logged.contains("payroll"));
        assert!(logged.contains("omitted from log"));
    }
    #[test]
    fn capability_names_reject_unknown_values() {
        assert!(CapabilitySet::parse_list(["web", "desktop"]).unwrap().web);
        assert!(CapabilitySet::parse_list(["web,desktop"]).unwrap().desktop);
        assert!(CapabilitySet::parse_list(["nope"]).is_err());
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
