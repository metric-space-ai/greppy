#!/usr/bin/env python3
"""Compile the production size declaration and its original payload-retention fault.

This uses a bounded synthetic payload, never a model or installable Greppy binary.
The real digest checks and release embedding are still exercised by normal CI.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PAYLOAD_BYTES = 256 * 1024


def checked(argv: list[str], env: dict[str, str]) -> None:
    result = subprocess.run(argv, env=env, text=True, capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError(f"{argv[0]} failed ({result.returncode}):\n{result.stderr}")


def assert_small_metadata(path: Path) -> int:
    size = path.stat().st_size
    if size >= PAYLOAD_BYTES // 4:
        raise AssertionError(
            "payload-size constant retained model bytes in Rust metadata: "
            f"{size} bytes for a {PAYLOAD_BYTES}-byte payload"
        )
    return size


def main() -> None:
    build_path = ROOT / "crates/cli/build.rs"
    lib_path = ROOT / "crates/cli/src/lib.rs"
    original = {p: p.read_bytes() for p in (build_path, lib_path)}
    build = original[build_path].decode()
    lib = original[lib_path].decode()
    helper = re.search(
        r"(?ms)^fn export_qwen35_gguf_size\(path: &Path\) \{.*?^\}", build
    )
    declaration = re.search(
        r"(?s)mod qwen35_assets\s*\{\s*(pub const GGUF_BYTES: u64\s*=.*?;)", lib
    )
    if helper is None or declaration is None:
        raise AssertionError("production asset-size helper/declaration not found")

    with tempfile.TemporaryDirectory(prefix="greppy-size-metadata-") as tmp:
        scratch = Path(tmp)
        payload = scratch / "fixture.bin"
        payload.write_bytes(bytes(range(256)) * (PAYLOAD_BYTES // 256))
        output = scratch / "out"
        output.mkdir()
        env = dict(os.environ, OUT_DIR=str(output))
        env["GREPPY_EMBEDDED_QWEN35_GGUF_PATH"] = str(payload)
        generator = scratch / "generator.rs"
        generator.write_text(
            "use std::path::{Path, PathBuf};\n"
            + helper.group(0)
            + '\nfn main() { export_qwen35_gguf_size(Path::new(env!("GREPPY_EMBEDDED_QWEN35_GGUF_PATH"))); }\n'
        )
        generator_bin = scratch / "generator"
        checked(
            ["rustc", "--edition=2024", "-Cdebuginfo=0", str(generator), "-o", str(generator_bin)], env
        )
        checked([str(generator_bin)], env)
        generated = (output / "qwen35_gguf_size.rs").read_text()
        assert generated == f"{PAYLOAD_BYTES}u64\n", generated

        def metadata(label: str, expression: str) -> Path:
            unit = scratch / f"{label}.rs"
            unit.write_text(
                "mod qwen35_assets {\n" + expression
                + "\n}\npub fn model_bytes() -> u64 { qwen35_assets::GGUF_BYTES }\n"
            )
            result = scratch / f"{label}.rmeta"
            checked(
                ["rustc", "--edition=2024", "--crate-type=lib", "--emit=metadata",
                 "-Cdebuginfo=0", str(unit), "-o", str(result)], env
            )
            return result

        positive = assert_small_metadata(metadata("current", declaration.group(1)))
        original_fault = (
            'pub const GGUF_BYTES: u64 = '
            'include_bytes!(env!("GREPPY_EMBEDDED_QWEN35_GGUF_PATH")).len() as u64;'
        )
        fault = metadata("original_fault", original_fault)
        try:
            assert_small_metadata(fault)
        except AssertionError as error:
            if "payload-size constant retained model bytes" not in str(error):
                raise
            print(f"original include_bytes size fault: RED ({fault.stat().st_size} bytes)")
        else:
            raise AssertionError("original include_bytes size fault did not turn the contract red")
        restored = assert_small_metadata(metadata("restored", declaration.group(1)))
        print(f"production size declaration: GREEN ({positive} bytes); restored: {restored} bytes")

    for path, before in original.items():
        assert path.read_bytes() == before, f"production source changed: {path}"
    print("production source byte-exact; " + hashlib.sha256(original[lib_path]).hexdigest())


if __name__ == "__main__":
    main()
