"""Atomic, checksummed persistence with no silent overwrite of results."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Callable

import torch
import yaml
from safetensors.torch import load_file, save_file

from src.errors import CheckpointError


def project_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    path = path if path.is_absolute() else root / path
    path = path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path must stay within project: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _new_path(path: Path, suffix: str) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.{suffix}")


def _commit(temp: Path, target: Path, overwrite: bool) -> None:
    if target.exists() and not overwrite:
        temp.unlink(missing_ok=True)
        raise FileExistsError(f"Refusing to overwrite existing artifact: {target}")
    os.replace(temp, target)
    directory = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def atomic_bytes(path: Path, data: bytes, *, overwrite: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = _new_path(path, "tmp")
    try:
        with temp.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        expected = hashlib.sha256(data).hexdigest()
        if sha256_file(temp) != expected:
            raise CheckpointError(f"Write verification failed: {temp}")
        _commit(temp, path, overwrite)
        return expected
    finally:
        temp.unlink(missing_ok=True)


def atomic_json(path: Path, value: Any, *, overwrite: bool = False) -> str:
    return atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2,
                                         allow_nan=False) + "\n").encode("utf-8"),
                        overwrite=overwrite)


def atomic_yaml(path: Path, value: Any, *, overwrite: bool = False) -> str:
    return atomic_bytes(path, yaml.safe_dump(value, sort_keys=False,
                                           allow_unicode=True).encode("utf-8"),
                        overwrite=overwrite)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    with path.open("ab") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def atomic_torch(path: Path, value: Any, *, overwrite: bool = False) -> str:
    """Serialize through a checked file rather than trusting a partial torch.save."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = _new_path(path, "tmp")
    try:
        with temp.open("xb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        with temp.open("rb") as handle:
            torch.load(handle, map_location="cpu", weights_only=False)
        digest = sha256_file(temp)
        _commit(temp, path, overwrite)
        return digest
    finally:
        temp.unlink(missing_ok=True)


def atomic_tensors(path: Path, tensors: dict[str, torch.Tensor], *,
                   metadata: dict[str, str] | None = None,
                   overwrite: bool = False) -> str:
    if not tensors:
        raise CheckpointError(f"Refusing to save empty tensor state: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = _new_path(path, "tmp")
    try:
        cpu = {k: v.detach().cpu().contiguous() for k, v in tensors.items()}
        save_file(cpu, str(temp), metadata=metadata)
        loaded = load_file(str(temp))
        if sorted(loaded) != sorted(cpu):
            raise CheckpointError(f"Tensor key verification failed: {path}")
        for key in cpu:
            if not torch.equal(loaded[key], cpu[key]):
                raise CheckpointError(f"Tensor roundtrip mismatch: {key}")
        digest = sha256_file(temp)
        _commit(temp, path, overwrite)
        return digest
    finally:
        temp.unlink(missing_ok=True)


def checked_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointError(f"Invalid JSON artifact {path}: {error}") from error


def write_new_run_file(path: Path, writer: Callable[[], Any]) -> None:
    """Callers use this for immutable inputs; progress is an explicit mutable file."""
    if path.exists():
        raise FileExistsError(f"Refusing to replace existing run input: {path}")
    writer()
