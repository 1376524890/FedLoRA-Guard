"""Data immutability and version identity checks, including incomplete downloads."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from src.data.records import Record, read_jsonl, validate_isolation
from src.errors import DataValidationError
from src.io import atomic_json, checked_json, sha256_file


def verify_raw_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = checked_json(manifest_path)
    checked = {}
    for name, value in manifest.items():
        if "path" not in value:
            checked[name] = value
            continue
        path = Path(value["path"])
        digest = sha256_file(path)
        if "sha256" in value and digest != value["sha256"]:
            raise DataValidationError(f"Raw source digest changed: {path}")
        checked[name] = {**value, "verified_current_sha256": digest}
    return checked


def verify_processed_records(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"Missing processed records: {path}")
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise DataValidationError(f"Processed dataset identity differs: {path}")
    records = [Record(**row) for row in read_jsonl(path)]
    validate_isolation(records)
    return {"path": str(path), "sha256": digest, "records": len(records),
            "unique_ids": len({record.sample_id for record in records}),
            "split_counts": {split: sum(record.split == split for record in records)
                             for split in sorted({record.split for record in records})}}
