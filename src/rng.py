"""Complete Python/NumPy/PyTorch/DataLoader RNG snapshots."""
from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def capture_rng(loader_generator: torch.Generator | None = None) -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader_generator": loader_generator.get_state() if loader_generator is not None else None}


def restore_rng(state: dict[str, Any], loader_generator: torch.Generator | None = None) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available():
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise RuntimeError("CUDA RNG state count differs from available GPU count")
        torch.cuda.set_rng_state_all([x.cpu() for x in state["torch_cuda"]])
    if loader_generator is not None:
        if state["loader_generator"] is None:
            raise RuntimeError("DataLoader generator state is required but absent")
        loader_generator.set_state(state["loader_generator"])
