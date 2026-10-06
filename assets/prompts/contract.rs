//! Owner signatures shared by both prompt guards and both product builds.
use sha2::{Digest, Sha256};
use std::path::Path;
#[path = "render.rs"]
pub mod render;

// Owner-approved prompt of record: AGENTS.md @
// e32618057b887254b2cd6aada91e128dbb3a220d. Approval dated 2026-09-30,
// relayed in website-agent-handover.md and directly reaffirmed on 2026-10-05:
// section 28: "ok, pass den prompt an und auch den entsprechenden guard";
// section 29: "ok, freigabe erteilt"; section 32: "ok, erlaubnis erteilt".
// Those approvals cover the READING CODE block, any-line and fallback rows.
// 8a70292ca5 incorrectly removed them as unapproved; do not discard this record.
// Historical SHA256: f5336832865582bcfe0dd851309085e8fa265dbf0138c387a446176664a8a9d1.
// 2026-10-05: Owner answered "ja" to the complete ec914aec3 proposal and BOTH
// guard hashes in prompt-source-truth-owner-review.md (this parent thread).
// This new approval covers the full shared contract, restored reading rules,
// callees wording, export documentation, uniform layout and argv adapter.
// 2026-10-06: Owner SIGNATURE, AskUserQuestion answer "Sign v3", relayed
// through the website session and explicitly supplied in this parent thread.
// Approved exact prompt-content-v3.proposed.md: 12,782 bytes, SHA256 below.
// This supersedes a1def1ae, fixes external/built-in execution and browser
// contract contradictions, and preserves the approved argv adapter unchanged.
// Bench arms use Greppy as their only system prompt, with shell tools only.
// Later prose changes require a new complete diff and explicit owner approval.
pub const APPROVED_SHA256: &str =
    "5455a7ba675d47f5adb5d5868b767054227e6fa152cefff021386d5af2de9440";
// The former short built-in prompt was signed as
// ade467bb75c46e16a56a66009738818eb1126581091c04c0d05daac8ce8d10f1.
// 2026-10-06 consolidated owner order A: exact coding-agent introduction,
// task workflow and three mode lines, supplied verbatim in this parent thread.
// External v3 remains unchanged. The renderer performs only the explicitly
// ordered exclusions; ACP makes no isolation/proposal promise pending a decision.
pub const APPROVED_AGENT_ADAPTER_SHA256: &str =
    "d86ef405b0fb74c3373e08a5c43e40bf823140819fdde607cd58ccab9d91d4f5";

pub fn approved_rendered_sha256(mode: render::BuiltinPromptMode) -> &'static str {
    use render::BuiltinPromptMode::*;
    match mode {
        OneShot => "cb105cd2df84a344ed35d69f52956ba2a7116c52f9ccd2ff57231225b42f6985",
        Interactive => "e728aa713a61d911652f4662fe8133d324fe480feef7d42a85e95a9cf311c730",
        Serve => "366b2124c0c6cb59a264fa7e901a65dc81c0113f7bc15d8f25054066486f3e9a",
        Acp => "51baa5de52bb2e0e1c3fc1c89667f04ce1bda2886f194ea07b07709b20d1ec95",
    }
}

pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn verify_bytes(name: &str, bytes: &[u8], approved: &str) -> Result<(), String> {
    let actual = digest(bytes);
    if actual == approved {
        Ok(())
    } else {
        Err(format!("Unsigned Greppy prompt {name}: actual SHA256 {actual}, owner-approved {approved}. Refusing to build. Present the complete diff to the owner before changing assets/prompts/contract.rs; no build flag bypasses this guard."))
    }
}

/// Return the very bytes whose signatures were verified, so embedding cannot
/// race a later source edit between the build guard and rustc include_str.
pub fn verify_repository(root: &Path) -> Result<(Vec<u8>, Vec<u8>), String> {
    let read = |relative: &str, approved: &str| -> Result<Vec<u8>, String> {
        let path = root.join(relative);
        println!("cargo:rerun-if-changed={}", path.display());
        let bytes =
            std::fs::read(&path).map_err(|e| format!("read prompt {}: {e}", path.display()))?;
        verify_bytes(relative, &bytes, approved)?;
        Ok(bytes)
    };
    let public = read("AGENTS.md", APPROVED_SHA256)?;
    let adapter = read(
        "assets/prompts/agent-adapter.md",
        APPROVED_AGENT_ADAPTER_SHA256,
    )?;
    println!(
        "cargo:rerun-if-changed={}",
        root.join("assets/prompts/contract.rs").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("assets/prompts/render.rs").display()
    );
    let public_text = std::str::from_utf8(&public).map_err(|e| format!("prompt UTF-8: {e}"))?;
    let adapter_text = std::str::from_utf8(&adapter).map_err(|e| format!("adapter UTF-8: {e}"))?;
    for mode in render::BUILTIN_MODES {
        let text = render::render_builtin(public_text, adapter_text, mode);
        verify_bytes(mode.name(), text.as_bytes(), approved_rendered_sha256(mode))?;
    }
    Ok((public, adapter))
}

/// Both builds validate the entire effective rendering, not only its inputs.
/// The agent embeds snapshots from the same verified byte buffers.
pub fn write_snapshots(root: &Path, out: &Path) -> Result<(), String> {
    let (public, adapter) = verify_repository(root)?;
    let public_text = std::str::from_utf8(&public).map_err(|e| e.to_string())?;
    let adapter_text = std::str::from_utf8(&adapter).map_err(|e| e.to_string())?;
    for mode in render::BUILTIN_MODES {
        let text = render::render_builtin(public_text, adapter_text, mode);
        std::fs::write(out.join(mode.file_name()), text).map_err(|e| e.to_string())?;
    }
    std::fs::write(out.join("canonical-prompt.md"), public).map_err(|e| e.to_string())?;
    std::fs::write(out.join("agent-adapter.md"), adapter).map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn changed_prompt_is_refused_with_both_digests() {
        let approved = digest(b"approved");
        assert!(verify_bytes("prompt", b"approved", &approved).is_ok());
        let err = verify_bytes("prompt", b"changed", &approved).unwrap_err();
        assert!(err.contains(&approved));
        assert!(err.contains(&digest(b"changed")));
        assert!(err.contains("Refusing to build"));
    }
}
