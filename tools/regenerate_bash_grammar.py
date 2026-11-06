#!/usr/bin/env python3
"""Reproduce the vendored Bash parser using a pinned build-time generator."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "third_party/tree-sitter-bash"
VERSION = "0.25.10"
URL = "https://github.com/tree-sitter/tree-sitter/releases/download/v0.25.10/tree-sitter-linux-x64.gz"
SHA256 = "8283ddba69253c698f6e987ba0e2f9285e079c8db4d36ebe1394b5bb3a0ebdfd"
OUTPUTS = ["src/parser.c", "src/grammar.json", "src/node-types.json", "src/tree_sitter/parser.h"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--output-dir", type=Path, required=True)
    options = args.parse_args()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SystemExit("Use the existing Linux CI lane; this generator is not a Mac build path")
    output = options.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="greppy-bash-generator-") as temporary:
        temporary = Path(temporary)
        compressed = urllib.request.urlopen(URL, timeout=60).read()
        if hashlib.sha256(compressed).hexdigest() != SHA256:
            raise AssertionError("Pinned generator download checksum mismatch")
        binary = temporary / "tree-sitter"
        binary.write_bytes(gzip.decompress(compressed))
        binary.chmod(0o700)
        version = subprocess.check_output([str(binary), "--version"], text=True).strip()
        if not version.startswith("tree-sitter " + VERSION):
            raise AssertionError("Wrong generator: " + version)
        node = subprocess.check_output(["node", "--version"], text=True).strip()
        work = temporary / "grammar"
        shutil.copytree(VENDOR, work)
        result = subprocess.run([str(binary), "generate", "--abi", "15", "grammar.js"],
                                cwd=work, capture_output=True, text=True, timeout=300)
        (output / "generator.stdout").write_text(result.stdout)
        (output / "generator.stderr").write_text(result.stderr)
        if result.returncode:
            raise AssertionError("Grammar generation failed:\n" + result.stdout + result.stderr)
        generated = {}
        mismatches = []
        for relative in OUTPUTS:
            source = work / relative
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            generated[relative] = {"sha256": digest(source), "bytes": source.stat().st_size}
            tracked = VENDOR / relative
            if not tracked.is_file() or digest(tracked) != digest(source):
                mismatches.append(relative)
        receipt = {"generator": version, "generator_url": URL, "generator_gzip_sha256": SHA256,
                   "node": node, "platform": platform.platform(), "abi": 15,
                   "command": ["tree-sitter", "generate", "--abi", "15", "grammar.js"],
                   "grammar_js_sha256": digest(VENDOR / "grammar.js"),
                   "outputs": generated, "committed_output_mismatches": mismatches,
                   "build_time_only": True}
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2), flush=True)
        if mismatches:
            raise SystemExit("Generated source differs or is pending import; artifact retained: " + ", ".join(mismatches))
        print("Pinned Bash parser regeneration is byte-identical", flush=True)


if __name__ == "__main__":
    main()
