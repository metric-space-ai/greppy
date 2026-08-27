#!/usr/bin/env python3
"""Verify one complete, exact-SHA portable CoW performance evidence set."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


EXPECTED_PLATFORMS = {
    ("linux", "x86_64"),
    ("macos", "aarch64"),
    ("windows", "x86_64"),
}
REQUIRED_TOOLCHAINS = {"rust", "python", "node"}
SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class EvidenceError(ValueError):
    """The supplied evidence cannot authorize a release."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def mapping(value: Any, name: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{name} must be an object")
    return value


def finite_number(value: Any, name: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{name} must be numeric",
    )
    number = float(value)
    require(number >= 0.0 and number < float("inf"), f"{name} is invalid")
    return number


def integer(value: Any, name: str) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{name} must be an integer",
    )
    require(value >= 0, f"{name} must not be negative")
    return value


def load_evidence(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{path} is not a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot read {path}: {error}") from error
    return mapping(value, str(path))


def validate_evidence(
    evidence: dict[str, Any],
    source_commit: str,
    origin: Path,
) -> tuple[str, str]:
    require(
        evidence.get("schema") == "greppy.portable-cow-performance.v1",
        f"{origin}: unsupported schema",
    )
    require(evidence.get("source_commit") == source_commit, f"{origin}: source SHA mismatch")
    require(
        evidence.get("source_tracked_worktree_dirty") is False,
        f"{origin}: source worktree was tracked-dirty",
    )
    require(evidence.get("profile") == "release", f"{origin}: non-release profile")
    os_name = evidence.get("os")
    arch = evidence.get("arch")
    require(
        isinstance(os_name, str) and isinstance(arch, str),
        f"{origin}: missing platform identity",
    )
    platform = (os_name, arch)
    require(platform in EXPECTED_PLATFORMS, f"{origin}: unexpected platform {platform}")
    require(
        isinstance(evidence.get("hardware"), str) and evidence["hardware"].strip(),
        f"{origin}: hardware description is empty",
    )
    require(
        isinstance(evidence.get("provider_binary"), str)
        and evidence["provider_binary"].strip(),
        f"{origin}: provider binary is not bound",
    )
    require(
        isinstance(evidence.get("provider_sha256"), str)
        and SHA256.fullmatch(evidence["provider_sha256"]) is not None,
        f"{origin}: provider SHA-256 is invalid",
    )
    require(
        integer(evidence.get("fixture_tracked_files"), f"{origin}: fixture_tracked_files")
        == 300_000,
        f"{origin}: fixture must contain exactly 300000 tracked files",
    )
    fixture = mapping(evidence.get("fixture_profile"), f"{origin}: fixture_profile")
    require(
        fixture.get("schema") == "greppy.portable-cow-fixture.v3"
        and integer(fixture.get("tracked_files"), f"{origin}: fixture tracked_files")
        == 300_000,
        f"{origin}: fixture profile is not the release fixture",
    )
    require(
        integer(evidence.get("iterations"), f"{origin}: iterations") >= 25,
        f"{origin}: fewer than 25 measured workspace iterations",
    )

    cold = mapping(evidence.get("cold_prime"), f"{origin}: cold_prime")
    cold_elapsed = finite_number(cold.get("elapsed_ms"), f"{origin}: cold prime elapsed")
    cold_gate = finite_number(cold.get("gate_ms"), f"{origin}: cold prime gate")
    require(cold_gate <= 120_000.0, f"{origin}: cold prime gate was relaxed")
    require(cold_elapsed <= cold_gate, f"{origin}: cold prime exceeds its gate")

    workspace = mapping(evidence.get("workspace_creation"), f"{origin}: workspace_creation")
    p95 = finite_number(workspace.get("end_to_end_p95_ms"), f"{origin}: workspace P95")
    p95_gate = finite_number(
        workspace.get("end_to_end_p95_gate_ms"), f"{origin}: workspace P95 gate"
    )
    require(p95_gate <= 500.0, f"{origin}: workspace P95 gate was relaxed")
    require(p95 <= p95_gate, f"{origin}: workspace P95 exceeds its gate")
    warmup = mapping(workspace.get("warmup"), f"{origin}: warmup")
    warmup_p95 = finite_number(
        warmup.get("steady_state_p95_ms"), f"{origin}: warmup steady-state P95"
    )
    warmup_gate = finite_number(
        warmup.get("steady_state_p95_gate_ms"), f"{origin}: warmup P95 gate"
    )
    require(warmup_gate <= 500.0, f"{origin}: warmup P95 gate was relaxed")
    require(warmup_p95 <= warmup_gate, f"{origin}: warmup P95 exceeds its gate")

    space = mapping(evidence.get("space"), f"{origin}: space")
    untouched = integer(
        space.get("untouched_physical_delta_bytes"), f"{origin}: untouched bytes"
    )
    untouched_gate = integer(space.get("untouched_gate_bytes"), f"{origin}: untouched gate")
    require(untouched_gate <= 1_048_576, f"{origin}: untouched-space gate was relaxed")
    require(untouched <= untouched_gate, f"{origin}: untouched space exceeds its gate")
    write_delta = integer(
        space.get("one_byte_write_physical_delta_bytes"), f"{origin}: one-byte delta"
    )
    write_gate = integer(
        space.get("one_byte_write_gate_bytes"), f"{origin}: one-byte gate"
    )
    require(write_gate <= 1_310_720, f"{origin}: one-byte write gate was relaxed")
    require(
        integer(space.get("one_byte_write_new_chunks"), f"{origin}: one-byte chunks") == 1,
        f"{origin}: one-byte write did not create exactly one chunk",
    )
    require(write_delta <= write_gate, f"{origin}: one-byte write exceeds its gate")
    require(
        integer(space.get("chunk_size"), f"{origin}: chunk_size") == 1_048_576,
        f"{origin}: chunk size differs from the release contract",
    )

    parallel = mapping(evidence.get("parallel"), f"{origin}: parallel")
    require(
        integer(parallel.get("workspaces"), f"{origin}: parallel workspaces") == 50,
        f"{origin}: release evidence must create exactly 50 parallel workspaces",
    )
    finite_number(parallel.get("wall_ms"), f"{origin}: parallel wall time")

    toolchains = evidence.get("toolchains")
    require(isinstance(toolchains, list), f"{origin}: toolchains must be an array")
    gated: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(toolchains):
        result = mapping(raw, f"{origin}: toolchains[{index}]")
        name = result.get("name")
        require(isinstance(name, str) and name, f"{origin}: toolchain name is missing")
        if result.get("release_gate") is True:
            require(name not in gated, f"{origin}: duplicate gated {name} toolchain")
            gated[name] = result
    require(set(gated) == REQUIRED_TOOLCHAINS, f"{origin}: gated toolchain set is incomplete")
    for name, result in gated.items():
        overhead = finite_number(
            result.get("overhead_percent"), f"{origin}: {name} overhead"
        )
        require(overhead <= 20.0, f"{origin}: {name} overhead exceeds 20%")

    return platform


def verify(paths: list[Path], source_commit: str) -> dict[str, Any]:
    require(
        GIT_COMMIT.fullmatch(source_commit) is not None,
        "source commit must be a full Git commit ID",
    )
    require(len(paths) == 3, "exactly three evidence files are required")
    platforms: dict[tuple[str, str], Path] = {}
    for path in paths:
        platform = validate_evidence(load_evidence(path), source_commit, path)
        require(platform not in platforms, f"duplicate evidence for {platform}")
        platforms[platform] = path
    require(set(platforms) == EXPECTED_PLATFORMS, "platform evidence set is incomplete")
    return {
        "schema": "greppy.portable-cow-performance-set.v1",
        "source_commit": source_commit,
        "platforms": [
            {"os": os_name, "arch": arch, "evidence": str(platforms[(os_name, arch)])}
            for os_name, arch in sorted(platforms)
        ],
        "release_eligible": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("evidence", nargs="+", type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify(args.evidence, args.source_commit)
    except EvidenceError as error:
        print(f"portable CoW performance set rejected: {error}", file=sys.stderr)
        return 1
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
