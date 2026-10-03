"""Choose the official production release matrix.

Production packages are limited to macOS arm64 with Metal and Linux x86_64
with CUDA. The static matrix retains other rows for debug/CI and historical
artifact-contract compatibility, but this script never emits them for release.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

SECRET_ENV = (
    "WINDOWS_SIGNED_WINFSP_DRIVER_BASE64",
    "WINDOWS_SIGNED_WINFSP_CATALOG_BASE64",
    "WINDOWS_SIGNED_WINFSP_DRIVER_CONTRACT_BASE64",
    "WINDOWS_CERTIFICATE_PFX_BASE64",
    "WINDOWS_CERTIFICATE_PASSWORD",
)
PRODUCTION_RELEASE_NAMES = {
    "build": frozenset(("macos-arm64", "linux-x86_64")),
    "verify": frozenset(("macos-arm64", "linux-x86_64-no-toolkit")),
}
PRODUCTION_BUILD_FEATURES = {
    "macos-arm64": "metal",
    "linux-x86_64": "cuda",
}


def signing_enabled(environ: dict[str, str] | None = None) -> bool:
    values = os.environ if environ is None else environ
    return all(values.get(name) for name in SECRET_ENV)


def filtered_includes(enabled: bool, matrix: dict | None = None) -> dict[str, list]:
    # Keep ``enabled`` in the interface because the workflow still records
    # signing readiness for diagnostics. Signing readiness does not make a
    # CPU-only package eligible for an official production release.
    del enabled
    if matrix is None:
        path = Path(__file__).with_name("release_matrix.json")
        matrix = json.loads(path.read_text(encoding="utf-8"))
    chosen = {}
    for key in ("build", "verify"):
        rows = [
            row for row in matrix[key] if row["name"] in PRODUCTION_RELEASE_NAMES[key]
        ]
        selected_names = [row["name"] for row in rows]
        missing = sorted(PRODUCTION_RELEASE_NAMES[key] - set(selected_names))
        duplicates = sorted(
            name for name in set(selected_names) if selected_names.count(name) != 1
        )
        if missing or duplicates:
            details = []
            if missing:
                details.append(f"missing={missing}")
            if duplicates:
                details.append(f"duplicates={duplicates}")
            raise ValueError(
                f"invalid production release {key} rows: " + ", ".join(details)
            )
        chosen[key] = rows

    for row in chosen["build"]:
        expected = PRODUCTION_BUILD_FEATURES[row["name"]]
        actual = row.get("features")
        if actual != expected:
            raise ValueError(
                "invalid production release backend for "
                f"{row['name']}: expected features={expected!r}, got {actual!r}"
            )
    return chosen


def write_github_output(path: Path, environ: dict[str, str] | None = None) -> None:
    enabled = signing_enabled(environ)
    chosen = filtered_includes(enabled)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"enabled={'true' if enabled else 'false'}\n")
        handle.write(
            "build_include=" + json.dumps(chosen["build"], separators=(",", ":")) + "\n"
        )
        handle.write(
            "verify_include="
            + json.dumps(chosen["verify"], separators=(",", ":"))
            + "\n"
        )


def main() -> None:
    output = os.environ.get("GITHUB_OUTPUT")
    if not output:
        raise SystemExit("GITHUB_OUTPUT is not set")
    write_github_output(Path(output))


if __name__ == "__main__":
    main()
