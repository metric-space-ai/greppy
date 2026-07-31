#!/usr/bin/env python3
"""Build an adapter manifest; rows become ready only after real local probes."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from collections.abc import Sequence

from .base import load_config, load_jsonl, proof_sha256


def registry_contract(path: pathlib.Path) -> dict[str, dict[str, str]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read repository registry: {exc}") from exc
    rows = document.get("repositories") if isinstance(document, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("repository registry has no repositories")
    contract: dict[str, dict[str, str]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or row["id"] in contract:
            raise ValueError("repository registry contains an invalid or duplicate id")
        contract[row["id"]] = {
            "repository_url": row.get("url"),
            "primary_language": row.get("primary_language"),
            "toolchain_profile": row.get("toolchain_profile"),
        }
    return contract


def validate_config_coverage(
    config_paths: Sequence[pathlib.Path], registry_path: pathlib.Path,
) -> list[tuple[pathlib.Path, dict]]:
    """Fail unless configs cover the frozen registry exactly and consistently."""
    contract = registry_contract(registry_path)
    loaded: dict[str, tuple[pathlib.Path, dict]] = {}
    for path in config_paths:
        resolved = path.resolve()
        config = load_config(resolved)
        repository_id = config["repository_id"]
        if repository_id in loaded:
            raise ValueError(f"duplicate adapter config for {repository_id}")
        expected = contract.get(repository_id)
        if expected is None:
            raise ValueError(f"adapter config is not in registry: {repository_id}")
        mismatches = [field for field, value in expected.items() if config.get(field) != value]
        if mismatches:
            raise ValueError(f"adapter config {repository_id} differs from registry: {mismatches}")
        loaded[repository_id] = (resolved, config)
    missing = sorted(set(contract) - set(loaded))
    if missing:
        raise ValueError(f"missing adapter configs: {missing}")
    return [loaded[repository_id] for repository_id in contract]


def smoke_ledger_reason(
    path: pathlib.Path | None, *, repository_id: str, proof: str, image_id: str | None,
) -> str | None:
    """Return why a real adapter validation ledger is not sufficient evidence."""
    if path is None:
        return "real two-run offline smoke validation ledger not supplied"
    try:
        rows = load_jsonl(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return f"smoke validation ledger is unreadable: {exc}"
    if not rows:
        return "smoke validation ledger is empty"
    for row in rows:
        validation = row.get("validation")
        provenance = row.get("merge_provenance")
        if not (
            row.get("repository") == repository_id
            and row.get("adapter_proof_sha256") == proof
            and isinstance(row.get("changed_source"), list) and row["changed_source"]
            and isinstance(row.get("changed_tests"), list) and row["changed_tests"]
            and isinstance(validation, dict)
            and validation.get("parent_baseline") == "pass"
            and validation.get("parent_plus_test") == "fail"
            and validation.get("gold_plus_test") == "pass"
            and validation.get("merged_plus_test") == "pass"
            and validation.get("clean_room_repetitions") == 2
            and validation.get("offline") is True
            and validation.get("runner_image_digest") == image_id
            and isinstance(provenance, dict)
            and all(provenance.get(field) is True for field in (
                "target_parent_verified", "merged_result_tree_verified", "pr_delta_no_target_drift"
            ))
        ):
            return "smoke validation ledger is not bound to this adapter/image or lacks real F2P/P2P proof"
    return None


def build_row(
    config_path: pathlib.Path, *, image: str | None, image_id: str | None,
    runtime_config_root: str, smoke_ledger: pathlib.Path | None = None,
) -> dict:
    config = load_config(config_path)
    proof = proof_sha256(config_path, config)
    reports = {}
    reasons: list[str] = []
    for role in ("probe", "metadata", "validate"):
        command = [sys.executable, "-m", "bench.agent_coding.v3.adapters.cli", "--config", str(config_path), role, "--preflight"]
        proc = subprocess.run(command, capture_output=True, text=True, errors="replace")
        try: payload = json.loads(proc.stdout)
        except json.JSONDecodeError: payload = None
        reports[role] = payload
        if not (
            proc.returncode == 0 and isinstance(payload, dict) and payload.get("ready") is True
            and payload.get("proof_sha256") == proof
        ):
            reasons.append(f"{role}: {(payload or {}).get('reason') or proc.stderr.strip() or 'probe failed'}")
    if not image or "@sha256:" not in image: reasons.append("digest-pinned image not supplied")
    if not image_id or not image_id.startswith("sha256:"): reasons.append("local image_id not supplied")
    smoke_reason = smoke_ledger_reason(
        smoke_ledger, repository_id=config["repository_id"], proof=proof, image_id=image_id,
    )
    if smoke_reason: reasons.append(smoke_reason)
    runtime_config = f"{runtime_config_root.rstrip('/')}/{config_path.name}"
    commands = {
        manifest_role: [
            "python3", "-m", "bench.agent_coding.v3.adapters.cli",
            "--config", runtime_config, cli_role,
        ]
        for manifest_role, cli_role in (
            ("probe", "probe"), ("metadata", "metadata"), ("validation", "validate")
        )
    }
    if image_id:
        commands["validation"].extend(["--runner-image-id", image_id])
    # Pipeline preflight executes this exact argv. Probe is intentionally a
    # real, terminating CLI role rather than a manifest-only/dead verb.
    commands["probe"].append("--preflight")
    row = {
        "repository_id": config["repository_id"], "status": "ready" if not reasons else "pending",
        "toolchain_profile": config["toolchain_profile"], "proof_sha256": proof,
    }
    if reasons:
        row["reason"] = "; ".join(reasons)
    else:
        row.update({
            "image": image, "image_id": image_id, "commands": commands,
            "probe_command": commands["probe"], "metadata_command": commands["metadata"],
            "validation_command": commands["validation"],
        })
    return row


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", type=pathlib.Path, required=True)
    parser.add_argument("--registry", type=pathlib.Path, required=True)
    parser.add_argument("--image")
    parser.add_argument("--image-id")
    parser.add_argument("--runtime-config-root", default="/opt/greppy-v3-adapters/configs")
    parser.add_argument(
        "--smoke-ledger", action="append", default=[], metavar="REPOSITORY_ID=PATH",
        help="real validate output from one local smoke PR; required before a row can be ready",
    )
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    ledgers: dict[str, pathlib.Path] = {}
    for item in args.smoke_ledger:
        repository_id, separator, value = item.partition("=")
        if not separator or not repository_id or not value or repository_id in ledgers:
            parser.error("--smoke-ledger must be unique REPOSITORY_ID=PATH values")
        ledgers[repository_id] = pathlib.Path(value).resolve()
    try:
        configs = validate_config_coverage(args.config, args.registry.resolve())
    except ValueError as exc:
        parser.error(str(exc))
    rows = []
    for resolved, config in configs:
        rows.append(build_row(
            resolved, image=args.image, image_id=args.image_id,
            runtime_config_root=args.runtime_config_root,
            smoke_ledger=ledgers.get(config["repository_id"]),
        ))
    document = {"schema_version": "greppy.agent-coding-v3.adapter-manifest.1", "adapters": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if all(row["status"] == "ready" for row in rows) else 2


if __name__ == "__main__":
    sys.exit(main())
