"""Storage layout for the GPU benchmark host.

The production host is expected to use fast local NVMe for mutable work and NAS
for immutable release artifacts.  Mount points are deliberately not guessed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


NVME_ENV = "GREPPY_BENCH_NVME_ROOT"
NAS_ENV = "GREPPY_BENCH_NAS_ROOT"


class StorageError(ValueError):
    """The storage configuration is unsafe or incomplete."""


@dataclass(frozen=True)
class StorageLayout:
    nvme_root: Path
    nas_root: Path

    @property
    def mirrors(self) -> Path:
        return self.nvme_root / "mirrors"

    @property
    def scratch(self) -> Path:
        return self.nvme_root / "scratch" / "agent-coding-v3"

    @property
    def worktrees(self) -> Path:
        return self.nvme_root / "worktrees" / "agent-coding-v3"

    @property
    def releases(self) -> Path:
        return self.nas_root / "agent-coding-v3" / "releases"

    def ensure(self) -> None:
        for path in (self.mirrors, self.scratch, self.worktrees, self.releases):
            path.mkdir(parents=True, exist_ok=True)


def _configured_root(document: Mapping[str, Any], tier: str) -> str | None:
    value = document.get(tier)
    if isinstance(value, Mapping):
        value = value.get("root")
    return value if isinstance(value, str) and value.strip() else None


def load_storage(
    config_path: Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    create: bool = True,
) -> StorageLayout:
    """Load explicit roots; environment overrides a JSON configuration file."""
    env = os.environ if environ is None else environ
    document: dict[str, Any] = {}
    if config_path is not None:
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StorageError(f"cannot read storage config {config_path}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise StorageError("storage config must be a JSON object")
        document = loaded

    nvme_value = env.get(NVME_ENV) or _configured_root(document, "nvme")
    nas_value = env.get(NAS_ENV) or _configured_root(document, "nas")
    missing = [name for name, value in ((NVME_ENV, nvme_value), (NAS_ENV, nas_value)) if not value]
    if missing:
        raise StorageError(
            "storage roots are mandatory; set " + " and ".join(missing) + " or pass --storage-config"
        )

    nvme = Path(nvme_value).expanduser().resolve()
    nas = Path(nas_value).expanduser().resolve()
    if nvme == nas or nvme in nas.parents or nas in nvme.parents:
        raise StorageError("NVMe and NAS roots must be distinct, non-nested directories")
    layout = StorageLayout(nvme_root=nvme, nas_root=nas)
    if create:
        layout.ensure()
    return layout
