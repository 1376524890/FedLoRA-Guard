"""Synchronous resumable federated LoRA orchestration and round checkpoints."""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from src.checkpoint import latest_complete_round, load_round_checkpoint, save_round_checkpoint
from src.data.records import Record, read_jsonl
from src.errors import CheckpointError, ConfigurationError
from src.fed.aggregate import aggregate
from src.fed.worker import WorkerFleet
from src.io import atomic_json, atomic_tensors, sha256_file, state_digest
from src.rng import capture_rng, restore_rng
from src.runtime import RunLogger, environment_info, utc_now
from src.train.model import apply_adapter, cpu_adapter_state, load_model
from src.version import save_run_inputs, source_fingerprint


def dataset_hash(path: Path) -> str:
    if not path.is_file():
        raise ConfigurationError(f"Canonical records file does not exist: {path}")
    return sha256_file(path)


def load_client_records(path: Path, clients: int) -> dict[str, list[dict[str, Any]]]:
    output = {f"client_{i:02d}": [] for i in range(clients)}
    seen = set()
    for record in read_jsonl(path):
        value = Record(**record)
        if value.split != "train":
            continue
        if value.client_id not in output or value.sample_id in seen:
            raise ConfigurationError(f"Invalid training assignment: {value.sample_id}")
        seen.add(value.sample_id)
        output[value.client_id].append(asdict(value))
    if any(not records for records in output.values()):
        raise ConfigurationError("Every client requires real private training records")
    return output


def _sanitize(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "state"}


def _public_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Round history deliberately excludes sample IDs, prompts and attack labels."""
    return [{"round": row["round"], "client_results": [
        {key: value for key, value in result.items() if key not in {"sample_ids", "checkpoint"}}
        for result in row["client_results"]], "aggregation": row["aggregation"],
        "global_adapter_digest": row["global_adapter_digest"]} for row in history]


class FederatedRunner:
    def __init__(self, config: dict[str, Any], *, root: Path, run_dir: Path,
                 seed: int, scenario: str, resume: bool = False) -> None:
        self.config, self.root, self.run_dir = config, root, run_dir
        self.seed, self.scenario = seed, scenario
        self.run_id = run_dir.name
        self.fed = config["federated"]
        self.clients = [f"client_{i:02d}" for i in range(self.fed["clients"])]
        self.checkpoints = run_dir / "checkpoints"
        self.code_fingerprint = source_fingerprint(root)
        data_path = Path(config["execution"]["records_path"])
        if not data_path.is_absolute():
            data_path = root / data_path
        self.data_path = data_path.resolve()
        self.data_hash = dataset_hash(self.data_path)
        if resume:
            self.original_code_fingerprint = self.code_fingerprint
            compatibility_path = run_dir / "source_compatibility.json"
            if compatibility_path.exists():
                original = json.loads((run_dir / "source_snapshot" / "src").exists() and
                                      (run_dir / "code_identity.json").read_text())["previous_fingerprint"]
                self.original_code_fingerprint = original
            identity = json.loads((run_dir / "data_identity.json").read_text(encoding="utf-8"))
            if identity["sha256"] != self.data_hash:
                raise CheckpointError("Dataset identity differs on resume")
            self.original_config_hash = config["_hash"]
            if compatibility_path.exists():
                self.original_config_hash = json.loads((run_dir / "recovery_control.json").read_text())["interrupted_config_hash"]
            if config["_hash"] != json.loads((run_dir / "resolved_config.json").read_text())["_hash"]:
                raise CheckpointError("Resolved configuration differs on resume")
            saved_fingerprint = json.loads((run_dir / "code_identity.json").read_text())["fingerprint"]
            if saved_fingerprint != self.code_fingerprint:
                raise CheckpointError("Source code differs on resume; refusing silent experiment redefinition")
        else:
            if run_dir.exists():
                raise FileExistsError(f"Run directory already exists; refusing overwrite: {run_dir}")
            run_dir.mkdir(parents=True)
            save_run_inputs(root, config, run_dir=run_dir,
                            dataset_path=self.data_path, dataset_hash=self.data_hash)
            atomic_json(run_dir / "code_identity.json", {"fingerprint": self.code_fingerprint})
            atomic_json(run_dir / "environment.json", environment_info())
        self.logger = RunLogger(run_dir, self.run_id, seed=seed, scenario=scenario,
                                total_tasks=self.fed["rounds"] * len(self.clients))
        self.records = load_client_records(self.data_path, len(self.clients))

    def _initial_adapter(self) -> dict[str, torch.Tensor]:
        path = self.run_dir / "initial_adapter.safetensors"
        if path.exists():
            return load_file(str(path))
        # The authoritative initial adapter is generated by the same pinned model
        # constructor as the GPU workers, not by a synthetic tensor.
        model, _, inventory = load_model(self.config["model"], torch.device("cpu"), self.seed,
                                        deterministic=self.fed["deterministic"],
                                        gradient_checkpointing=False)
        state = cpu_adapter_state(model)
        atomic_tensors(path, state, metadata={"base_revision": self.config["model"]["revision"]})
        atomic_json(self.run_dir / "model_inventory.json", inventory)
        del model
        return state

    def _restore(self) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], dict[str, Any], int]:
        completed = latest_complete_round(self.checkpoints)
        if completed == 0:
            return self._initial_adapter(), [], {"foolsgold_history": []}, 0
        checkpoint = load_round_checkpoint(self.checkpoints / f"round_{completed:03d}", {
            "run_id": self.run_id, "seed": self.seed, "scenario": self.scenario,
            "config_hash": self.config["_hash"], "dataset_hash": self.data_hash,
            "code_fingerprint": self.original_code_fingerprint})
        restore_rng(checkpoint["rng_state"])
        return checkpoint["global_state"], checkpoint["history"], checkpoint["aggregation_state"], completed

    def _pool(self) -> WorkerFleet:
        return WorkerFleet(self.config, {
            "run_id": self.run_id, "seed": self.seed, "scenario": self.scenario,
            "checkpoint_root": str(self.checkpoints), "dataset_hash": self.data_hash,
            "code_fingerprint": self.original_code_fingerprint})

    def _worker_tasks(self, pool: WorkerFleet, round_index: int,
                      global_state: dict[str, torch.Tensor], active_attack: bool):
        tasks = []
        for index, client in enumerate(self.clients):
            gpu = self.fed["gpu_ids"][index % len(self.fed["gpu_ids"])]
            tasks.append((gpu, {"round_index": round_index, "client_id": client,
                                "client_index": index, "global_state": global_state,
                                "records": self.records[client], "active_attack": active_attack}))
        return tasks

    def run(self) -> dict[str, Any]:
        global_state, history, aggregation_state, first_round = self._restore()
        # Events and checkpoints are authoritative; no completed client result is
        # deserialized as a new training task or re-aggregated on resume.
        self.logger.update(phase="worker_initialization", round=first_round,
                           completed_tasks=first_round * len(self.clients))
        try:
            with self._pool() as pool:
                for round_index in range(first_round + 1, self.fed["rounds"] + 1):
                    started = time.perf_counter()
                    self.logger.update(phase="local_training", round=round_index)
                    tasks = self._worker_tasks(pool, round_index, global_state,
                                               active_attack=self._attack_active(round_index))
                    client_results = pool.train_round(
                        tasks, progress_handler=lambda event: self.logger.task(
                            phase="local_training", round_index=event["round"],
                            client_id=event["client_id"], local_step=event["step"],
                            local_steps=self.fed["local_optimizer_steps"]))
                    for result in client_results:
                        self.logger.client_completed(result)
                        self.logger.update(completed_tasks=(round_index - 1) * len(self.clients) + len(client_results),
                                           client_id=result["client_id"], local_step=result["optimizer_steps"])
                    if len(client_results) != len(self.clients) or {r["client_id"] for r in client_results} != set(self.clients):
                        raise RuntimeError("Incomplete synchronous client participation")
                    client_results.sort(key=lambda x: x["client_id"])
                    states = [result["state"] for result in client_results]
                    global_state, detail, aggregation_state = aggregate(
                        self.fed["aggregation"], states, global_state,
                        config=self.fed, aggregation_state=aggregation_state,
                        sample_counts=[len(self.records[r["client_id"]]) for r in client_results])
                    if state_digest(global_state) == state_digest(states[0]) and len(states) > 1:
                        raise RuntimeError("Aggregation unexpectedly returned one client's adapter")
                    public_row = {"round": round_index, "client_results": [_sanitize(r) for r in client_results],
                                  "aggregation": detail, "global_adapter_digest": state_digest(global_state)}
                    history.append(public_row)
                    metrics = {"round": round_index, "elapsed_seconds": time.perf_counter() - started,
                               "poison_exposure_count": sum(r["poison_exposure_count"] for r in client_results),
                               "aggregation_method": self.fed["aggregation"],
                               "global_adapter_digest": public_row["global_adapter_digest"]}
                    save_round_checkpoint(self.checkpoints / f"round_{round_index:03d}",
                                          run_id=self.run_id, seed=self.seed, scenario=self.scenario,
                                          round_index=round_index, global_state=global_state,
                                          aggregation_state=aggregation_state, participation=self.clients,
                                          history=_public_history(history), detector_state={},
                                          permutation_state={}, metrics=metrics,
                                          rng_state=capture_rng(), config_hash=self.original_config_hash,
                                          dataset_hash=self.data_hash, code_fingerprint=self.code_fingerprint)
                    atomic_json(self.run_dir / "latest.json", {"round": round_index, "complete": True}, overwrite=True)
                    self.logger.round_completed(round_index, metrics)
                self.logger.update(phase="completed", status="completed", client_id=None,
                                   completed_tasks=self.fed["rounds"] * len(self.clients))
                atomic_json(self.run_dir / "run_summary.json", {"status": "completed", "rounds": self.fed["rounds"],
                            "seed": self.seed, "scenario": self.scenario,
                            "final_global_adapter_digest": state_digest(global_state),
                            "finished_at": utc_now()})
                return {"status": "completed", "final_global_adapter_digest": state_digest(global_state)}
        except BaseException as error:
            self.logger.failed(error)
            raise

    def _attack_active(self, round_index: int) -> bool:
        attack = self.config.get("attack_config")
        if not attack:
            return False
        start, end = attack["window"]
        return start <= round_index <= end
