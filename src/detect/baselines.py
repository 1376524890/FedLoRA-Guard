"""Server-side norm and cosine baselines for effective LoRA updates."""
from __future__ import annotations

from itertools import combinations
import numpy as np
from typing import Any

import torch

from src.detect.lowrank import LowRankUpdate, gram, medoid_residuals, norms_and_cosine


def norm_cosine(updates: list[LowRankUpdate], client_ids: list[str], round_index: int) -> dict[str, Any]:
    matrix = gram(updates)
    norms, cosine = norms_and_cosine(matrix)
    centered = medoid_residuals(matrix)
    return {"round": round_index, "method": "effective_update_norm_cosine",
            "norms": {client: float(value) for client, value in zip(client_ids, norms)},
            "cosine": [{"client_a": client_ids[i], "client_b": client_ids[j],
                        "value": float(cosine[i, j])} for i, j in combinations(range(len(client_ids)), 2)],
            "residual_cosine": [{"client_a": client_ids[i], "client_b": client_ids[j],
                                 "value": float(centered["residual_cosine"][i, j])
                                 if np.isfinite(centered["residual_cosine"][i, j]) else None}
                                for i, j in combinations(range(len(client_ids)), 2)],
            "medoid_index": centered["medoid_index"],
            "training_labels_used": False}
