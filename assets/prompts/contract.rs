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
// 2026-11-05: Owner signature for PR #325, reviewed head
// 66e7808f508f561e63a9806719068b55b9f220e1, manifest SHA256
// 1b0907acac2cc0c6c140282cff60a17e806b8d480877b1ba2ea34841044ad6bb.
// Covers the exact sources, prepared manuals and all sixteen mode/capability
// starts below. The November 3 request was not a signature for these bytes.
// Adapter unchanged. Previous public hash 12bb2db4… (12,900 bytes).
pub const APPROVED_SHA256: &str =
    "cbc854763d4ddbcf349d60f038aeb61e971e5987a0a8bb202b512f4123d1fd80";
pub const APPROVED_WEB_CAPABILITY_SHA256: &str =
    "47b03e776795bf61101cb74e3522e0d57a0c01d184ff0f485a853f8aec5ef1bb";
pub const APPROVED_DESKTOP_CAPABILITY_SHA256: &str =
    "9a396328ae2393b7f771cda55936cd2f240bde0c23a7748413fe9fd1cf4dbbe6";
pub const APPROVED_WEB_BUILTIN_CAPABILITY_SHA256: &str =
    "cc2374d6e9cd742c8dd32a42ff6e1cdb3aac5ba18a7be51ca6190138652d16bc";
pub const APPROVED_DESKTOP_BUILTIN_CAPABILITY_SHA256: &str =
    "75c0143c2990aa6a46ae26afc49dd2a646d3cba731e2090d607aba7b58bfdae4";
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
        OneShot => "8d4eac5f57ef71b0f118f98bb619cc41255eb8297372bdc511ee372c8d1ed4ac",
        Interactive => "b1a6760221db89cf0f4f38b220577163e9d58f29a10dc6b38148e251ca7af839",
        Serve => "aa28c9e9d82fdf90c8c08ec8469e7e7bd68c6f1e790860055036803847e9d025",
        Acp => "d9d5fa63fa9c9ad05584d534d9800a3d00ee9c480412bc4e8a40f40b7a39df1b",
    }
}

/// Owner-signed effective start prompt, including optional capability manuals.
pub fn approved_start_sha256(
    mode: render::BuiltinPromptMode,
    web: bool,
    desktop: bool,
) -> &'static str {
    use render::BuiltinPromptMode::*;
    match (mode, web, desktop) {
        (mode, false, false) => approved_rendered_sha256(mode),
        (OneShot, true, false) => {
            "c61632799619fe882eac69ed05e70b24cdca989456291967732d12d525b6222b"
        }
        (OneShot, false, true) => {
            "ac4c7cdc22b3fe5085dc8b8cab10deacdfc2bf03a6a49b8c6b80a15b86354cd9"
        }
        (OneShot, true, true) => "d9ce57416e0c50306fb3a26ce6e52c10f9779f2965e250be38a2d92f2a9cd239",
        (Interactive, true, false) => {
            "550deef4cbd133c05a3814e92d170a8747237203b9acbff57a7a0de1dea89085"
        }
        (Interactive, false, true) => {
            "5f1665efd98d264c9ff482205850abb67d7412495fe89c0a5f164296373670f9"
        }
        (Interactive, true, true) => {
            "02d8d468c0e06b6ac6c0c99abd65a1fc4ad69a8a4a751386cfc5319619265104"
        }
        (Serve, true, false) => "6aeb3d769e59a2154700c527bf667d549ba56b6f3e99d3e1883ffea67dfb8d9f",
        (Serve, false, true) => "bc89067c27e8d4ef3dd7d2ddeb9cb46a6700900c299e417c8bb7f416839c2fe3",
        (Serve, true, true) => "1c5cbf2939dde730e45ad7f2e7ad664fd79101ac336d5d8225524ec717e0df54",
        (Acp, true, false) => "fd86b895e615fbec6d9c2e1bd55cd8f7fa9957ed7241ea257127a0d39691af8e",
        (Acp, false, true) => "f7b837ec737da01bf05030396d21d36148597fa4643a3cf4c11dde02957f347d",
        (Acp, true, true) => "297a40c5bad538aeace2cbfe3b0b9de0b85323b8da7326d5883000388536ea29",
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
    let web = read(
        "assets/prompts/capabilities/web.md",
        APPROVED_WEB_CAPABILITY_SHA256,
    )?;
    let desktop = read(
        "assets/prompts/capabilities/desktop.md",
        APPROVED_DESKTOP_CAPABILITY_SHA256,
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
    let web_text = std::str::from_utf8(&web).map_err(|e| format!("web UTF-8: {e}"))?;
    let desktop_text = std::str::from_utf8(&desktop).map_err(|e| format!("desktop UTF-8: {e}"))?;
    for (name, source, approved) in [
        (
            "built-in web manual",
            web_text,
            APPROVED_WEB_BUILTIN_CAPABILITY_SHA256,
        ),
        (
            "built-in desktop manual",
            desktop_text,
            APPROVED_DESKTOP_BUILTIN_CAPABILITY_SHA256,
        ),
    ] {
        verify_bytes(
            name,
            render::prepare_capability(source).as_bytes(),
            approved,
        )?;
    }
    for mode in render::BUILTIN_MODES {
        for (web, desktop) in [(false, false), (true, false), (false, true), (true, true)] {
            let text = render::render_start(
                public_text,
                adapter_text,
                mode,
                web_text,
                desktop_text,
                web,
                desktop,
            );
            verify_bytes(
                mode.name(),
                text.as_bytes(),
                approved_start_sha256(mode, web, desktop),
            )?;
        }
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
    fn every_start_variant_matches_the_owner_signed_manifest() {
        for mode in render::BUILTIN_MODES {
            for (web, desktop) in [(false, false), (true, false), (false, true), (true, true)] {
                let text = render::render_start(
                    include_str!("../../AGENTS.md"),
                    include_str!("agent-adapter.md"),
                    mode,
                    include_str!("capabilities/web.md"),
                    include_str!("capabilities/desktop.md"),
                    web,
                    desktop,
                );
                verify_bytes(
                    mode.name(),
                    text.as_bytes(),
                    approved_start_sha256(mode, web, desktop),
                )
                .unwrap();
            }
        }
    }

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
