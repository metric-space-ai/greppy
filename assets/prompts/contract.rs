//! Owner signatures shared by both prompt guards and both product builds.
use sha2::{Digest, Sha256};
use std::path::Path;

// Owner-approved prompt of record: AGENTS.md @
// e32618057b887254b2cd6aada91e128dbb3a220d. Approval dated 2026-09-30,
// relayed in website-agent-handover.md and directly reaffirmed on 2026-10-05:
// section 28: "ok, pass den prompt an und auch den entsprechenden guard";
// section 29: "ok, freigabe erteilt"; section 32: "ok, erlaubnis erteilt".
// Those approvals cover the READING CODE block, any-line and fallback rows.
// 8a70292ca5 incorrectly removed them as unapproved; do not discard this record.
// This historical signature does NOT approve the later callees wording,
// prompt-export documentation, new layout or argv adapter: their complete
// proposed diff still requires the owner's signature before building.
pub const APPROVED_SHA256: &str =
    "f5336832865582bcfe0dd851309085e8fa265dbf0138c387a446176664a8a9d1";
// The former short built-in prompt was signed as
// ade467bb75c46e16a56a66009738818eb1126581091c04c0d05daac8ce8d10f1.
// That signature does not approve this new adapter. Await the owner's signature.
pub const APPROVED_AGENT_ADAPTER_SHA256: &str = "UNSIGNED_REQUIRES_OWNER_SIGNATURE";

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
    Ok((public, adapter))
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
