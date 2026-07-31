#!/usr/bin/env python3
"""CLI for configured V3 adapters: probe, metadata and validate."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
from collections.abc import Sequence

from .base import (
    AdapterError, GitHubClient, atomic_jsonl, format_time, harvest_metadata,
    load_config, load_jsonl, parse_time, preflight_payload, proof_sha256,
    validate_candidate_for_ledger,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--config", type=pathlib.Path, required=True)
    sub = root.add_subparsers(dest="role", required=True)
    for role in ("probe", "metadata", "validate"):
        command = sub.add_parser(role)
        command.add_argument("--preflight", action="store_true")
    metadata = sub.choices["metadata"]
    metadata.add_argument("--repository-id")
    metadata.add_argument("--repository-url")
    metadata.add_argument("--merged-after")
    metadata.add_argument("--merged-before")
    metadata.add_argument("--all-merged-prs", action="store_true")
    metadata.add_argument("--output", type=pathlib.Path)
    validate = sub.choices["validate"]
    validate.add_argument("--repository-id")
    validate.add_argument("--mirror", type=pathlib.Path)
    validate.add_argument("--metadata", type=pathlib.Path)
    validate.add_argument("--scratch", type=pathlib.Path)
    validate.add_argument("--repetitions", type=int)
    validate.add_argument("--required-passing", type=int)
    validate.add_argument("--offline", action="store_true")
    validate.add_argument("--runner-image-id")
    validate.add_argument("--output", type=pathlib.Path)
    return root


def required(args: argparse.Namespace, *names: str) -> None:
    missing = [name for name in names if getattr(args, name) in (None, "")]
    if missing: raise AdapterError(f"missing required arguments: {missing}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        config_path = args.config.resolve()
        config = load_config(config_path)
        if args.preflight:
            payload = preflight_payload(args.role, config_path, config)
            print(json.dumps(payload, sort_keys=True))
            return 0 if payload["ready"] else 2
        if args.role == "probe":
            raise AdapterError("probe requires --preflight")
        if args.role == "metadata":
            required(args, "repository_id", "repository_url", "merged_after", "merged_before", "output")
            if not args.all_merged_prs:
                raise AdapterError("metadata harvest must request every merged PR")
            if args.repository_id != config["repository_id"] or args.repository_url.rstrip("/") != config["repository_url"].rstrip("/"):
                raise AdapterError("pipeline repository identity differs from adapter config")
            rows = harvest_metadata(
                client=GitHubClient(os.environ.get("GITHUB_TOKEN", "")),
                repository_id=args.repository_id, repository_url=args.repository_url,
                merged_after=parse_time(args.merged_after),
                merged_before=parse_time(args.merged_before), config=config,
            )
            atomic_jsonl(args.output, rows)
        else:
            required(
                args, "repository_id", "mirror", "metadata", "scratch",
                "repetitions", "required_passing", "output",
            )
            if args.repository_id != config["repository_id"] or not args.offline or args.repetitions != 2:
                raise AdapterError("validation identity/offline/two-run contract differs")
            if args.required_passing < 1:
                raise AdapterError("validation requires a positive passing target")
            digest = args.runner_image_id or os.environ.get("GREPPY_ADAPTER_IMAGE_ID", "")
            if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
                raise AdapterError("GREPPY_ADAPTER_IMAGE_ID must bind the validation image")
            rows = []
            passing = 0
            for row in load_jsonl(args.metadata):
                if passing >= args.required_passing:
                    rows.append({**row, "validation_outcome": "not_run"})
                    continue
                result = validate_candidate_for_ledger(
                    row, args.mirror.resolve(), args.scratch.resolve(), args.repetitions,
                    config, digest, proof_sha256(config_path, config),
                )
                rows.append(result)
                passing += result.get("validation_outcome") == "passed"
            atomic_jsonl(args.output, rows)
        return 0
    except (AdapterError, OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"adapter failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
