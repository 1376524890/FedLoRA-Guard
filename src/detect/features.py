"""Server-visible LoRA features; training text and oracle labels are out of scope."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from src.errors import DetectorInputError


@dataclass(frozen=True)
class ServerUpdate:
    client_id: str
    round_index: int
    adapter: dict[str, torch.Tensor]
    received_global: dict[str, torch.Tensor]


def first_layer_b_features(updates: list[ServerUpdate]) -> tuple[np.ndarray, list[str]]:
    """Safe-FedLLM `first_B`: all seven first-layer B matrices, sorted and L2."""
    if not updates:
        raise DetectorInputError("No server updates for Safe-FedLLM")
    def selected(adapter: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        layers = {int(match.group(1)) for key in adapter
                  if (match := re.search(r"(?:^|\.)layers\.(\d+)\.", key))}
        if not layers:
            raise DetectorInputError("Safe-FedLLM requires explicit transformer layer indices")
        first = min(layers)
        keys = [key for key in adapter if re.search(rf"(?:^|\.)layers\.{first}\.", key)
                and key.endswith(".lora_B.weight")]
        if not keys:
            raise DetectorInputError("Safe-FedLLM first-layer B matrices are missing")
        return {key: adapter[key] for key in sorted(keys)}
    selected_keys = sorted(selected(updates[0].adapter))
    rows = []
    for update in updates:
        if sorted(update.adapter) != sorted(update.received_global):
            raise DetectorInputError("Safe-FedLLM global/client parameter mismatch")
        selected_update = selected(update.adapter)
        if sorted(selected_update) != selected_keys:
            raise DetectorInputError("Safe-FedLLM B parameter names differ")
        vector = torch.cat([(update.adapter[key] - update.received_global[key]).double().flatten()
                            for key in selected_keys])
        if not torch.isfinite(vector).all():
            raise DetectorInputError("Safe-FedLLM feature contains NaN/Inf")
        norm = vector.norm()
        if float(norm) == 0:
            raise DetectorInputError("Safe-FedLLM zero feature cannot be L2-normalized")
        rows.append((vector / norm).numpy())
    return np.stack(rows), selected_keys


def train_official_bce_probe(features: np.ndarray, labels: np.ndarray, *, seed: int,
                             config: dict[str, Any]) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Official-repository classifier, separate from the requested sklearn LR."""
    if features.ndim != 2 or labels.shape != (len(features),):
        raise DetectorInputError("Safe-FedLLM training matrix mismatch")
    if set(np.unique(labels)) != {0, 1}:
        raise DetectorInputError("Safe-FedLLM training needs both classes")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = torch.nn.Linear(features.shape[1], 1)
    # Official LinearProbeClassifier squeezes its scalar logit.
    class ScalarProbe(torch.nn.Module):
        def __init__(self, linear):
            super().__init__()
            self.linear = linear
        def forward(self, value):
            return self.linear(value).squeeze(-1)
    probe = ScalarProbe(model)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    positive, negative = int(labels.sum()), int((1 - labels).sum())
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / (positive + 1e-6)))
    x = torch.from_numpy(features.astype(np.float32))
    y = torch.from_numpy(labels.astype(np.float32))
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y),
                                        batch_size=config["batch_size"], shuffle=True,
                                        generator=generator)
    for epoch in range(config["epochs"]):
        probe.train()
        for x_batch, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(probe(x_batch), y_batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Safe-FedLLM non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), config["gradient_clip"])
            optimizer.step()
    return probe, {"classifier": "official_single_linear_BCE_AdamW", "seed": seed,
                   "early_stopping": "not_applicable_trajectory_holdout",
                   "official_defaults": config}


def train_sklearn_logistic(features: np.ndarray, labels: np.ndarray, *, seed: int,
                           config: dict[str, Any]) -> LogisticRegression:
    if set(np.unique(labels)) != {0, 1}:
        raise DetectorInputError("Logistic Regression requires both training classes")
    model = LogisticRegression(**config, random_state=seed)
    model.fit(features, labels)
    return model


def probe_scores(model: Any, features: np.ndarray) -> np.ndarray:
    if isinstance(model, LogisticRegression):
        return model.predict_proba(features)[:, 1]
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(features.astype(np.float32)))).numpy()
