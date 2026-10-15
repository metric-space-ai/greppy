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
// 2026-10-06: Owner SIGNATURE for v4, AskUserQuestion answer "Sign (a)+(b) for
// 0.4.1" in the 0.4.1 release session, after a verified G-regression (MiniMax-M3
// on SWE-rebench astral-sh__ruff-25414: vanilla 3/3 patches, v3 3/3 analyses
// without a patch). Exactly two changes to v3 (5455a7ba…, 12,782 bytes):
// (a) "chosen by the question" -> "chosen by the next step of the task";
// (b) new sentence after the opening paragraph: "The task is a change to
// deliver: make the edits with greppy and verify them; do not stop at an
// analysis." v4: 12,900 bytes, SHA256 below. Adapter unchanged.
pub const APPROVED_SHA256: &str =
    "12bb2db4bce1ab2603279913aeb676f7b68115143a42170091f5842a6cf0d84f";
// The former short built-in prompt was signed as
// ade467bb75c46e16a56a66009738818eb1126581091c04c0d05daac8ce8d10f1.
// 2026-10-06 consolidated owner order A: exact coding-agent introduction,
// task workflow and three mode lines, supplied verbatim in this parent thread.
// External v3 remains unchanged. The renderer performs only the explicitly
// ordered exclusions; ACP makes no isolation/proposal promise pending a decision.
// 2026-10-15: Owner SIGNATURE in the greppy homepage/bench session ("führe die edits im system
// prompt aus", "ja, setz das so ein", "nicht einfach hier und da was dran hängen"), after the caddy
// trace analysis (one reasoning phase per tool call). Exactly one sentence added to adapter step 2
// after "run shell commands as `bash-smart -- CMD`.": "Every call costs a turn: issue steps that do
// not depend on each other together in one turn, as several greppy calls or one call with
// {"batch": [[…], […]]} (run in order; a failed edit stops the rest), and pipe a search straight
// into a read (`search-symbol NAME | read -` inside bash-smart)." Previous adapter d86ef405…;
// ACP rendering unchanged.
pub const APPROVED_AGENT_ADAPTER_SHA256: &str =
    "dd0133ecff891859bbbd3890ffe0c9a3857393e512f2665a8521487cb9d15280";

pub fn approved_rendered_sha256(mode: render::BuiltinPromptMode) -> &'static str {
    use render::BuiltinPromptMode::*;
    match mode {
        OneShot => "887653abb4e7ad51aa543df52611018868f570d937248f71c42e528e1e934fef",
        Interactive => "2f4449863c6b37c204473069d20f5d246a32484964e9eccbc532b65e4c1d11ee",
        Serve => "2f93c68ef53008009f27849cae41cc4b87c499ca5d127f88267e64c3be282ed7",
        Acp => "3683b8542fc14de0b9c46a3ebbed630f076dde4d56bb0523f587e93454198b44",
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
