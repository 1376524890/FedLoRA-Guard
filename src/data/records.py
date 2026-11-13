"""Canonical records and strict, disjoint source-level data partitioning."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from src.errors import DataValidationError
from src.io import atomic_json, sha256_file

SCHEMA_VERSION = "fedlora-records-v1"


@dataclass(frozen=True)
class Record:
    sample_id: str
    source: str
    source_id: str
    prompt: str
    response: str
    category: str | None = None
    context: str = ""
    split: str = "unassigned"
    client_id: str | None = None
    safe_response: str | None = None
    unsafe_response: str | None = None
    poisoned_prompt: str | None = None
    poisoned_response: str | None = None
    original_prompt: str | None = None
    original_response: str | None = None
    trigger: str | None = None
    insertion_index: int | None = None
    attack_type: str | None = None
    attack_client_id: str | None = None
    poison_fraction: float | None = None

    def __post_init__(self) -> None:
        if not self.sample_id or not self.source or not self.source_id:
            raise DataValidationError("Every record needs a stable sample/source identifier")
        if not self.prompt.strip() or not self.response.strip():
            raise DataValidationError(f"Missing prompt or response: {self.sample_id}")
        if "\x00" in self.prompt or "\x00" in self.response:
            raise DataValidationError(f"NUL in text: {self.sample_id}")

    @property
    def training_prompt(self) -> str:
        return self.poisoned_prompt if self.poisoned_prompt is not None else self.prompt

    @property
    def training_response(self) -> str:
        return self.poisoned_response if self.poisoned_response is not None else self.response

    @property
    def is_poisoned(self) -> bool:
        return self.poisoned_prompt is not None


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise DataValidationError(f"Invalid JSON {path}:{line_number}: {error}") from error
            if not isinstance(item, dict):
                raise DataValidationError(f"Expected JSON object {path}:{line_number}")
            yield item


def write_jsonl(path: Path, records: Iterable[Record | dict]) -> str:
    from src.io import atomic_bytes
    lines = []
    for item in records:
        value = asdict(item) if isinstance(item, Record) else item
        lines.append(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return atomic_bytes(path, ("\n".join(lines) + "\n").encode("utf-8"))


def deduplicate(records: list[Record], field: str = "prompt") -> list[Record]:
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    kept = []
    for record in records:
        text = getattr(record, field).strip()
        if record.sample_id in seen_ids:
            raise DataValidationError(f"Duplicate sample ID: {record.sample_id}")
        if text in seen_text:
            continue
        seen_ids.add(record.sample_id)
        seen_text.add(text)
        kept.append(record)
    if not kept:
        raise DataValidationError("No valid unique records")
    return kept


def source_fingerprint(record: Record) -> str:
    text = json.dumps([record.source, record.source_id, record.prompt, record.response,
                       record.safe_response, record.unsafe_response],
                      ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def allocate_records(dolly: list[Record], pku: list[Record], backdoor: list[Record],
                     *, num_clients: int, per_client: int, validation: int,
                     public_probes: int, clean_test: int, seed: int,
                     mixed_allocation: dict[str, int] | None) -> list[Record]:
    """Allocate source records once; poison variants never become extra sources.

    The formal mixed-source layout is blocked until explicitly supplied. The Dolly-only
    integration layout is intentionally separate and does not modify formal data semantics.
    """
    expected = num_clients * per_client
    sources = {"dolly": dolly, "pku": pku, "backdoor": backdoor}
    if mixed_allocation is None:
        if pku or backdoor:
            raise DataValidationError(
                "Mixed source allocation is unspecified; refusing to duplicate or infer records")
        if len(dolly) < expected + validation + public_probes + clean_test:
            raise DataValidationError(
                f"Dolly requires {expected + validation + public_probes + clean_test} disjoint records")
        allocation = {"dolly": expected}
    else:
        if sum(mixed_allocation.values()) != expected or set(mixed_allocation) != set(sources):
            raise DataValidationError("Explicit mixed allocation must sum to client capacity")
        allocation = mixed_allocation
    for name, count in allocation.items():
        if len(sources[name]) < count:
            raise DataValidationError(f"Insufficient unique {name} records: {count} required")
    rng = random.Random(seed)
    selected: list[Record] = []
    # Sampling is independent of original file ordering and reproducible for each source.
    for name, count in allocation.items():
        pool = sources[name]
        indices = rng.sample(range(len(pool)), count)
        selected.extend(pool[index] for index in indices)
    rng.shuffle(selected)
    result: list[Record] = []
    for index, record in enumerate(selected):
        client = index // per_client
        result.append(Record(**{**asdict(record), "split": "train",
                                "client_id": f"client_{client:02d}"}))
    if mixed_allocation is None:
        remaining = [r for r in dolly if r.sample_id not in {x.sample_id for x in selected}]
        rng.shuffle(remaining)
        cursor = 0
        for split, count in (("validation", validation), ("public_probe", public_probes),
                             ("clean_test", clean_test)):
            for record in remaining[cursor:cursor + count]:
                result.append(Record(**{**asdict(record), "split": split, "client_id": None}))
            cursor += count
    else:
        raise DataValidationError(
            "Mixed-source validation/probe/test source assignments must also be explicitly configured")
    validate_isolation(result)
    return result


def validate_isolation(records: list[Record]) -> None:
    ids = [r.sample_id for r in records]
    if len(ids) != len(set(ids)):
        raise DataValidationError("Non-unique sample identifiers")
    split_ids: dict[str, str] = {}
    split_prompts: dict[str, str] = {}
    for record in records:
        if record.split == "train" and record.client_id is None:
            raise DataValidationError("Training record lacks client assignment")
        if record.split == "unassigned":
            continue
        if record.sample_id in split_ids:
            raise DataValidationError(f"Sample assigned to multiple splits: {record.sample_id}")
        split_ids[record.sample_id] = record.split
        if record.prompt in split_prompts:
            raise DataValidationError(f"Prompt source overlap across records: {record.sample_id}")
        split_prompts[record.prompt] = record.sample_id
    train_clients = [r.client_id for r in records if r.split == "train"]
    if len(train_clients) != len(set((r.client_id, r.sample_id) for r in records if r.split == "train")):
        raise DataValidationError("Client sample overlap")
    for split in ("validation", "public_probe", "clean_test"):
        for record in (r for r in records if r.split == split):
            if record.client_id is not None or record.is_poisoned:
                raise DataValidationError("Public/evaluation record leaked into client or attack data")


def source_manifest(path: Path, *, source: str, revision: str, expected_sha256: str) -> dict:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise DataValidationError(f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual}")
    return {"source": source, "revision": revision, "path": str(path),
            "bytes": path.stat().st_size, "sha256": actual, "schema_version": SCHEMA_VERSION}


def dirichlet_partition(records: list[Record], clients: int, alpha: float, seed: int) -> list[Record]:
    if alpha <= 0 or clients < 2:
        raise DataValidationError("Dirichlet partition requires alpha>0 and at least two clients")
    categories = sorted({r.category or "unknown" for r in records})
    if len(categories) < 2:
        raise DataValidationError("Non-IID partition needs multiple categories")
    rng = np.random.default_rng(seed)
    assignments: list[str | None] = [None] * len(records)
    positions = {r.sample_id: i for i, r in enumerate(records)}
    for category in categories:
        indices = np.array([i for i, r in enumerate(records) if (r.category or "unknown") == category])
        rng.shuffle(indices)
        proportions = rng.dirichlet(np.full(clients, alpha))
        cuts = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        for client, group in enumerate(np.split(indices, cuts)):
            for index in group:
                assignments[index] = f"client_{client:02d}"
    if any(client is None for client in assignments):
        raise DataValidationError("Incomplete Dirichlet client assignment")
    return [Record(**{**asdict(r), "client_id": assignments[positions[r.sample_id]], "split": "train"})
            for r in records]
