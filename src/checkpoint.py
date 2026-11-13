"""Step-level atomic checkpoints and strict resume compatibility validation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from src.errors import CheckpointError
from src.io import atomic_json, atomic_tensors, atomic_torch, checked_json, sha256_file


@dataclass(frozen=True)
class ClientCheckpoint:
    path: Path
    manifest: dict[str, Any]


def save_client_checkpoint(directory: Path, *, run_id: str, seed: int, scenario: str,
                           round_index: int, client_id: str, next_step: int,
                           next_sample_position: int, global_state: dict[str, torch.Tensor],
                           client_state: dict[str, torch.Tensor], optimizer: torch.optim.Optimizer,
                           scheduler: Any, rng_state: dict[str, Any], sample_ids: list[str],
                           loss_sum: float, poison_exposure_count: int,
                           completed: bool, code_fingerprint: str,
                           config_hash: str, dataset_hash: str,
                           consumed_sample_ids: list[str]) -> ClientCheckpoint:
    directory.mkdir(parents=True, exist_ok=True)
    adapter_digest = atomic_tensors(directory / "client_adapter.safetensors", client_state,
                                    metadata={"round": str(round_index), "client_id": client_id})
    global_digest = atomic_tensors(directory / "received_global_adapter.safetensors", global_state)
    state = {"optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict() if scheduler is not None else None,
             "rng": rng_state, "next_step": next_step, "next_sample_position": next_sample_position,
             "sample_ids": sample_ids, "consumed_sample_ids": consumed_sample_ids, "loss_sum": loss_sum,
             "poison_exposure_count": poison_exposure_count,
             "client_id": client_id, "round": round_index, "completed": completed}
    optimizer_digest = atomic_torch(directory / "training_state.pt", state)
    manifest = {"schema": "client-checkpoint-v1", "run_id": run_id, "seed": seed,
                "scenario": scenario, "round": round_index, "client_id": client_id,
                "next_step": next_step, "next_sample_position": next_sample_position,
                "completed": completed, "config_hash": config_hash, "dataset_hash": dataset_hash,
                "code_fingerprint": code_fingerprint,
                "files": {name: sha256_file(directory / name) for name in
                          ("client_adapter.safetensors", "received_global_adapter.safetensors", "training_state.pt")},
                "client_adapter_digest": adapter_digest, "received_global_digest": global_digest,
                "training_state_digest": optimizer_digest}
    atomic_json(directory / "manifest.json", manifest)
    return ClientCheckpoint(directory, manifest)


def load_client_checkpoint(directory: Path, expected: dict[str, str]) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise CheckpointError(f"Incomplete checkpoint, no manifest: {directory}")
    manifest = checked_json(manifest_path)
    if manifest.get("schema") != "client-checkpoint-v1":
        raise CheckpointError(f"Unsupported checkpoint schema: {directory}")
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise CheckpointError(f"Checkpoint {key} mismatch: {manifest.get(key)!r} != {value!r}")
    for name, digest in manifest["files"].items():
        path = directory / name
        if not path.exists() or sha256_file(path) != digest:
            raise CheckpointError(f"Checkpoint checksum mismatch: {path}")
    try:
        state = torch.load(directory / "training_state.pt", map_location="cpu", weights_only=False)
        client = load_file(str(directory / "client_adapter.safetensors"))
        global_state = load_file(str(directory / "received_global_adapter.safetensors"))
    except Exception as error:
        raise CheckpointError(f"Cannot deserialize checkpoint {directory}: {error}") from error
    if state["next_step"] != manifest["next_step"] or state["completed"] != manifest["completed"]:
        raise CheckpointError(f"Checkpoint manifest/training state mismatch: {directory}")
    if len(state["consumed_sample_ids"]) != state["next_sample_position"] or \
            state["consumed_sample_ids"] != state["sample_ids"][:state["next_sample_position"]]:
        raise CheckpointError(f"Checkpoint sample consumption mismatch: {directory}")
    return state, client, global_state


def save_round_checkpoint(directory: Path, *, run_id: str, seed: int, scenario: str,
                          round_index: int, global_state: dict[str, torch.Tensor],
                          aggregation_state: dict[str, Any], participation: list[str],
                          history: list[dict[str, Any]], detector_state: dict[str, Any],
                          permutation_state: dict[str, Any], metrics: dict[str, Any],
                          rng_state: dict[str, Any], config_hash: str,
                          dataset_hash: str, code_fingerprint: str,
                          client_updates: list[dict[str, torch.Tensor]],
                          received_global_state: dict[str, torch.Tensor]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    atomic_tensors(directory / "global_adapter.safetensors", global_state)
    atomic_torch(directory / "aggregation_state.pt", aggregation_state)
    atomic_torch(directory / "client_updates.pt", client_updates)
    atomic_tensors(directory / "received_global_adapter.safetensors", received_global_state)
    atomic_json(directory / "participation.json", participation)
    atomic_torch(directory / "history.pt", history)
    atomic_json(directory / "detector_state.json", detector_state)
    atomic_json(directory / "permutation_state.json", permutation_state)
    atomic_json(directory / "metrics.json", metrics)
    atomic_torch(directory / "rng_state.pt", rng_state)
    manifest = {"schema": "round-checkpoint-v1", "run_id": run_id, "seed": seed,
                "scenario": scenario, "round": round_index, "complete": True,
                "config_hash": config_hash, "dataset_hash": dataset_hash,
                "code_fingerprint": code_fingerprint,
                "files": {p.name: sha256_file(p) for p in directory.iterdir()
                          if p.is_file() and p.name != "manifest.json"}}
    atomic_json(directory / "manifest.json", manifest)


def load_round_checkpoint(directory: Path, expected: dict[str, str]) -> dict[str, Any]:
    manifest = checked_json(directory / "manifest.json")
    if manifest.get("schema") != "round-checkpoint-v1" or not manifest.get("complete"):
        raise CheckpointError(f"Incomplete round checkpoint: {directory}")
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise CheckpointError(f"Round checkpoint {key} mismatch: {manifest.get(key)!r} != {value!r}")
    for name, digest in manifest["files"].items():
        if not (directory / name).exists() or sha256_file(directory / name) != digest:
            raise CheckpointError(f"Round checkpoint file mismatch: {name}")
    return {"manifest": manifest,
            "global_state": load_file(str(directory / "global_adapter.safetensors")),
            "aggregation_state": torch.load(directory / "aggregation_state.pt", map_location="cpu", weights_only=False),
            "client_updates": torch.load(directory / "client_updates.pt", map_location="cpu", weights_only=False),
            "received_global_state": load_file(str(directory / "received_global_adapter.safetensors")),
            "participation": checked_json(directory / "participation.json"),
            "history": torch.load(directory / "history.pt", map_location="cpu", weights_only=False),
            "detector_state": checked_json(directory / "detector_state.json"),
            "permutation_state": checked_json(directory / "permutation_state.json"),
            "metrics": checked_json(directory / "metrics.json"),
            "rng_state": torch.load(directory / "rng_state.pt", map_location="cpu", weights_only=False)}


def latest_complete_round(checkpoints: Path) -> int:
    rounds = []
    if not checkpoints.exists():
        return 0
    for path in checkpoints.glob("round_*/manifest.json"):
        manifest = checked_json(path)
        if manifest.get("schema") == "round-checkpoint-v1" and manifest.get("complete"):
            for name, digest in manifest["files"].items():
                if not (path.parent / name).exists() or sha256_file(path.parent / name) != digest:
                    raise CheckpointError(f"Corrupt round checkpoint: {path.parent}")
            rounds.append(int(manifest["round"]))
    rounds = [index for index in rounds if index > 0]
    if rounds and sorted(rounds) != list(range(1, max(rounds) + 1)):
        raise CheckpointError("Round checkpoints are not contiguous; refusing to skip missing rounds")
    return max(rounds, default=0)
