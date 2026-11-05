#!/usr/bin/env python3
"""Prove mandatory Anthropic output bounds using real Rust request contracts."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
WIRE = ROOT / "crates/agent/src/wire.rs"
CLIENT = ROOT / "crates/agent/src/client.rs"
CLI = ROOT / "crates/cli/src/agent.rs"
FILTER = "required_anthropic_output_limit"


def run(filter_name, red=None, cli=False):
    target = (["-p", "greppy", "--features", "ci-test-assets", "--test", "agent_p"]
              if cli else ["-p", "greppy-agent", "--lib"])
    result = subprocess.run(
        ["cargo", "test", "--locked", "--jobs", "2", *target,
         filter_name, "--", "--test-threads=1"],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=600,
    )
    if red is None:
        if result.returncode != 0:
            raise AssertionError(result.stdout)
    elif not (result.returncode == 101 and "test result: FAILED" in result.stdout
              and red + " ... FAILED" in result.stdout):
        raise AssertionError("Wrong mutation failure or unexpected success:\n" + result.stdout)
    return result


def mutate(path, fixed, mutant, red, label, filter_name=FILTER, cli=False):
    original = path.read_text()
    if original.count(fixed) != 1:
        raise AssertionError("Mutation anchor must be unique: " + fixed)
    try:
        path.write_text(original.replace(fixed, mutant, 1))
        result = run(filter_name, red, cli)
        if cli:
            diagnostic = "no advertised or registered output limit"
            if diagnostic not in result.stdout:
                raise AssertionError("Wrong caller-limit failure:\n" + result.stdout)
            print("mutation: " + red + " exits101; missing caller bound fails locally: "
                  + diagnostic, flush=True)
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
run("caller_output_limit", cli=True)
mutate(
    CLI, '.map_or(u64::MAX, OutputTokenLimit::get)',
    '.map_or(u64::MAX, |_| u64::MAX)',
    "caller_output_limit_reaches_wire_and_continues",
    "dropping the caller's verified response limit",
    filter_name="caller_output_limit", cli=True,
)
run("caller_output_limit", cli=True)
print("mutation: restored CLI caller limit reaches actual HTTP requests and continuation", flush=True)
