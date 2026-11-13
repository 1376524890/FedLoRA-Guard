"""Immutable run inputs and source-code fingerprints, independent of git mutation."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from src.io import atomic_json


def source_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "src").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def save_run_inputs(root: Path, config: dict, *, run_dir: Path,
                    dataset_path: Path, dataset_hash: str) -> None:
    atomic_json(run_dir / "resolved_config.json", config)
    atomic_json(run_dir / "data_identity.json", {
        "path": str(dataset_path), "sha256": dataset_hash,
        "manifest_path": str(dataset_path.parent / "manifest.json")})
    destination = run_dir / "source_snapshot" / "src"
    for path in sorted((root / "src").rglob("*.py")):
        relative = path.relative_to(root / "src")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with path.open("rb") as source, target.open("xb") as output:
            shutil.copyfileobj(source, output)
