//! One canonical contract for the built-in, ACP and exported agent prompts.
pub const PUBLIC_PROMPT: &str = include_str!(concat!(env!("OUT_DIR"), "/canonical-prompt.md"));
const ADAPTER: &str = include_str!(concat!(env!("OUT_DIR"), "/agent-adapter.md"));
pub const SYSTEM_PROMPT: &str = concat!(
    "greppy ",
    env!("CARGO_PKG_VERSION"),
    "\n\n",
    include_str!(concat!(env!("OUT_DIR"), "/agent-adapter.md")),
    "\n",
    include_str!(concat!(env!("OUT_DIR"), "/canonical-prompt.md"))
);
pub fn browser_prompt() -> &'static str {
    let start = PUBLIC_PROMPT
        .find("\nBROWSER:")
        .expect("canonical browser block")
        + 1;
    let tail = &PUBLIC_PROMPT[start..];
    let end = tail.find("END BROWSER").expect("complete browser block") + "END BROWSER".len();
    &tail[..end]
}
pub fn system_prompt() -> String {
    SYSTEM_PROMPT.to_owned()
}
pub fn export_prompt(external: bool) -> String {
    if external {
        format!("greppy {}\n\n{PUBLIC_PROMPT}", env!("CARGO_PKG_VERSION"))
    } else {
        system_prompt()
    }
}
pub fn prompt_metadata(external: bool) -> serde_json::Value {
    let text = export_prompt(external);
    serde_json::json!({
        "schema": "greppy.prompt.v1", "version": env!("CARGO_PKG_VERSION"),
        "source": "AGENTS.md", "mode": if external { "external" } else { "built-in" },
        "prompt_sha256": crate::prompt_contract::digest(PUBLIC_PROMPT.as_bytes()),
        "adapter_sha256": (!external).then(|| crate::prompt_contract::digest(ADAPTER.as_bytes())),
        "rendered_sha256": crate::prompt_contract::digest(text.as_bytes()), "prompt": text
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn both_guards_use_the_same_owner_signatures() {
        crate::prompt_contract::verify_bytes(
            "AGENTS.md",
            PUBLIC_PROMPT.as_bytes(),
            crate::prompt_contract::APPROVED_SHA256,
        )
        .unwrap();
        crate::prompt_contract::verify_bytes(
            "adapter",
            ADAPTER.as_bytes(),
            crate::prompt_contract::APPROVED_AGENT_ADAPTER_SHA256,
        )
        .unwrap();
    }
    #[test]
    fn version_and_complete_contract_are_present_once() {
        for external in [false, true] {
            let text = export_prompt(external);
            assert!(text.starts_with(&format!("greppy {}\n", env!("CARGO_PKG_VERSION"))));
            assert_eq!(text.matches(PUBLIC_PROMPT).count(), 1);
            assert_eq!(text.matches("BROWSER:").count(), 1);
            assert!(text.contains("Default to ONE compact"));
            assert!(text.contains("CHAIN"));
            assert_eq!(
                prompt_metadata(external)["rendered_sha256"],
                crate::prompt_contract::digest(text.as_bytes())
            );
        }
        assert_eq!(system_prompt(), SYSTEM_PROMPT);
        assert_eq!(
            SYSTEM_PROMPT,
            format!(
                "greppy {}\n\n{ADAPTER}\n{PUBLIC_PROMPT}",
                env!("CARGO_PKG_VERSION")
            )
        );
        assert!(!PUBLIC_PROMPT.contains("Method: orient before editing"));
    }
    #[test]
    fn browser_comes_from_the_complete_canonical_contract() {
        assert!(browser_prompt().starts_with("BROWSER:"));
        assert!(browser_prompt().ends_with("END BROWSER"));
        assert!(SYSTEM_PROMPT.contains(browser_prompt()));
    }
}
