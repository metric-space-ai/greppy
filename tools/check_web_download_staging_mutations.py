#!/usr/bin/env python3
"""Exact mutations for shared native download staging authority and ownership."""
from pathlib import Path
import subprocess

cases = [
    (Path("crates/web-runtime/runtime/src/download_staging.rs"),
     "if next > state.0 { return None; }", "if false { return None; }",
     "download_staging", "second context bypassed the shared session budget"),
    (Path("crates/web-runtime/runtime/src/download_staging.rs"),
     "return budget;", "return Arc::new(DownloadBudget::new(limit));",
     "download_staging", "session allowance was duplicated by context"),
    (Path("crates/web-runtime/runtime/src/download_staging.rs"),
     "state.1 -= self.bytes;", "/* deliberately omit byte return */",
     "download_staging", "last owner did not return the session allowance"),
    (Path("crates/web-runtime/runtime/src/download_staging.rs"),
     "if let Some(path) = &self.path { let _ = std::fs::remove_file(path); }",
     "/* deliberately retain the staged file */",
     "download_staging", "last file owner did not remove its file"),
    (Path("crates/web-runtime/runtime/src/supervisor.rs"),
     'object.remove("__download_session");', "/* allow controller session override */",
     "controller_cannot_replace_download_session_authority", "controller replaced download session authority"),
    (Path("crates/web-runtime/runtime/src/supervisor.rs"),
     'object.remove("max_download_bytes");', "/* allow controller quota override */",
     "controller_cannot_replace_download_session_authority", "controller replaced download byte authority"),
]
for source, old, new, selector, message in cases:
    original = source.read_text()
    assert original.count(old) == 1, (source, old, original.count(old))
    command = ["cargo", "test", "--locked", "--manifest-path", "crates/web-runtime/Cargo.toml",
               "-p", "web-runtime", "--lib", "--no-default-features", selector,
               "--", "--test-threads=1", "--nocapture"]
    try:
        source.write_text(original.replace(old, new))
        result = subprocess.run(command, capture_output=True, text=True, timeout=600)
        print(result.stdout, result.stderr, flush=True)
        assert result.returncode == 101 and message in result.stdout + result.stderr, (
            "staging mutation missed its contract assertion", source, result.returncode, message)
        print("mutation: shared download staging regression detected", flush=True)
    finally:
        source.write_text(original)
    subprocess.run(command, check=True, timeout=600)
    subprocess.run(["git", "diff", "--exit-code", "--", str(source)], check=True)
