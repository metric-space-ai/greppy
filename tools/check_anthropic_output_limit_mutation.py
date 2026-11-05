#!/usr/bin/env python3
"""Prove mandatory Anthropic output bounds using real Rust request contracts."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
WIRE = ROOT / "crates/agent/src/wire.rs"
CLIENT = ROOT / "crates/agent/src/client.rs"
FILTER = "required_anthropic_output_limit"


def run(filter_name, red=None):
    result = subprocess.run(
        ["cargo", "test", "--locked", "--jobs", "2", "-p", "greppy-agent",
         "--lib", filter_name, "--", "--test-threads=1"],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=600,
    )
    if red is None:
        if result.returncode != 0:
            raise AssertionError(result.stdout)
    elif not (result.returncode == 101 and "test result: FAILED" in result.stdout
              and red + " ... FAILED" in result.stdout):
        raise AssertionError("Wrong mutation failure or unexpected success:\n" + result.stdout)


def mutate(path, fixed, mutant, red, label):
    original = path.read_text()
    if original.count(fixed) != 1:
        raise AssertionError("Mutation anchor must be unique: " + fixed)
    try:
        path.write_text(original.replace(fixed, mutant, 1))
        run(FILTER, red)
        print("mutation: " + label + " turns the real request contract red", flush=True)
    finally:
        path.write_text(original)


run(FILTER)
mutate(
    WIRE, '    body["max_tokens"] = json!(max_tokens);',
    '    // mutation: omit required max_tokens',
    "wire::tests::required_anthropic_output_limit_is_numeric_and_finite",
    "omitting max_tokens",
)
mutate(
    CLIENT,
    '"kimi-k3.1-pro" | "kimi-k3.1-pro-highspeed" => Some(65536),',
    '"kimi-k3.1-pro" | "kimi-k3.1-pro-highspeed" => None,',
    "client::metadata_tests::required_anthropic_output_limit_uses_exact_metadata_or_registry",
    "dropping the source-backed registry limit",
)
mutate(
    CLIENT,
    '        base\n    } else {\n        model\n    }\n}',
    '        model\n    } else {\n        model\n    }\n}',
    "client::metadata_tests::required_anthropic_output_limit_preserves_gateway_reasoning_suffix",
    "restoring literal-only capability lookup",
)
run(FILTER)
run("max_tokens_continues")
print("mutation: restored schema/metadata contracts green; truncated response continues", flush=True)
