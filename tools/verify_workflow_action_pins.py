#!/usr/bin/env python3
"""Reject mutable third-party GitHub Action references."""

from __future__ import annotations

import argparse
import pathlib
import re
import sys


USES = re.compile(r"^\s*(?:-\s*)?uses:\s*([^\s#]+)")
PINNED_ACTION = re.compile(r"^[^/@\s]+/[^@\s]+@[0-9a-f]{40}$")


def verify(paths: list[pathlib.Path]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            errors.append(f"{path}: cannot read workflow: {exc}")
            continue
        for line_number, line in enumerate(lines, 1):
            match = USES.match(line)
            if match is None:
                continue
            reference = match.group(1)
            if reference.startswith("./"):
                continue
            if not PINNED_ACTION.fullmatch(reference):
                errors.append(
                    f"{path}:{line_number}: external action is not pinned to a "
                    f"40-character commit SHA: {reference}"
                )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=pathlib.Path,
        default=sorted(pathlib.Path(".github/workflows").glob("*.yml")),
    )
    args = parser.parse_args(argv)
    errors = verify(args.paths)
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        return 1
    print(f"verified immutable Action pins in {len(args.paths)} workflows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
