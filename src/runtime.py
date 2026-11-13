"""Structured events, machine-readable progress, GPU status and live summaries."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.io import append_jsonl, atomic_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def environment_info() -> dict[str, Any]:
    import datasets
    import matplotlib
    import numpy
    import pandas
    import peft
    import scipy
    import sklearn
    import transformers
    uname = os.uname() if hasattr(os, "uname") else None
    result = {"python": sys.version,
              "platform": {field: getattr(uname, field) for field in
                           ("sysname", "nodename", "release", "version", "machine")} if uname else {},
              "hostname": os.uname().nodename if hasattr(os, "uname") else "unknown",
              "dependencies": {m.__name__: m.__version__ for m in
                               (torch, transformers, peft, datasets, numpy, scipy,
                                sklearn, pandas, matplotlib)},
              "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
              "devices": [{"index": i, "name": torch.cuda.get_device_name(i),
                           "capability": list(torch.cuda.get_device_capability(i)),
                           "memory_bytes": torch.cuda.get_device_properties(i).total_memory}
                          for i in range(torch.cuda.device_count())]}
    try:
        result["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version",
             "--format=csv,noheader"], text=True).strip().splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"Cannot inspect required GPUs: {error}") from error
    return result


def gpu_status() -> list[dict[str, Any]]:
    try:
        lines = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True).strip().splitlines()
        return [dict(zip(("gpu_id", "memory_used_mib", "memory_total_mib", "utilization_percent"),
                         (int(value.strip()) for value in line.split(",")))) for line in lines]
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        raise RuntimeError(f"GPU status unavailable: {error}") from error


class RunLogger:
    def __init__(self, root: Path, run_id: str, *, seed: int, scenario: str,
                 total_tasks: int) -> None:
        self.root = root
        self.run_id = run_id
        self.seed = seed
        self.scenario = scenario
        self.total_tasks = total_tasks
        self.started = time.monotonic()
        self.progress_path = root / "progress.json"
        self.progress = {"run_id": run_id, "phase": "initialization", "scenario": scenario,
                         "seed": seed, "round": 0, "client_id": None, "local_step": 0,
                         "total_tasks": total_tasks, "completed_tasks": 0, "failed_tasks": 0,
                         "completion_percent": 0.0, "elapsed_seconds": 0.0,
                         "estimated_remaining_seconds": None, "status": "running",
                         "gpu": gpu_status(), "updated_at": utc_now()}
        self.update()

    def emit(self, event: str, *, status: str = "completed", **fields: Any) -> None:
        if event == "round_completed" and "round" in fields and "round_index" not in fields:
            fields["round_index"] = fields["round"]

        record = {"timestamp": utc_now(), "run_id": self.run_id, "seed": self.seed,
                  "scenario": self.scenario, "event": event, "status": status,
                  "elapsed_seconds": round(time.monotonic() - self.started, 6), **fields}
        append_jsonl(self.root / "events.jsonl", record)

    def update(self, **fields: Any) -> None:
        self.progress.update(fields)
        elapsed = time.monotonic() - self.started
        self.progress["elapsed_seconds"] = elapsed
        done = self.progress["completed_tasks"]
        self.progress["completion_percent"] = 100.0 * done / self.total_tasks if self.total_tasks else 0.0
        self.progress["estimated_remaining_seconds"] = (
            elapsed / done * (self.total_tasks - done) if done else None)
        self.progress["gpu"] = gpu_status()
        self.progress["updated_at"] = utc_now()
        atomic_json(self.progress_path, self.progress, overwrite=True)

    def client_completed(self, result: dict[str, Any], *, resumed_event: bool = False) -> None:
        self.emit("client_training_completed", round=result["round"],
                  client_id=result["client_id"], gpu_id=result["gpu_id"],
                  local_steps=result["optimizer_steps"], loss=result["average_loss"],
                  poison_exposure_count=result["poison_exposure_count"],
                  elapsed_client_seconds=result["elapsed_seconds"],
                  checkpoint=result["checkpoint"], resumed=result["resumed"],
                  already_completed=result["already_completed"], resumed_event=resumed_event)
        print(f"  {result['client_id']} done | loss={result['average_loss']:.6f} "
              f"steps={result['optimizer_steps']} poison={result['poison_exposure_count']} "
              f"gpu={result['gpu_id']} time={result['elapsed_seconds']:.2f}s", flush=True)

    def round_completed(self, round_index: int, metrics: dict[str, Any]) -> None:
        self.emit("round_completed", **metrics)
        printable = {key: value for key, value in metrics.items()
                     if isinstance(value, (float, int, str)) and key != "event"}
        print(f"[round {round_index}] {json.dumps(printable, ensure_ascii=False)}", flush=True)

    def task(self, *, phase: str, round_index: int, client_id: str,
             local_step: int, local_steps: int) -> None:
        self.update(phase=phase, round=round_index, client_id=client_id, local_step=local_step)
        print(f"[{self.scenario} seed={self.seed}] round {round_index} {client_id} "
              f"local step {local_step}/{local_steps}", flush=True)

    def failed(self, error: BaseException) -> None:
        self.emit("run_failed", status="failed", error_type=type(error).__name__, error=str(error))
        self.update(status="failed", failed_tasks=self.progress["failed_tasks"] + 1,
                    error_type=type(error).__name__, error=str(error))
