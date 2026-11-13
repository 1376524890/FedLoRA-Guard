"""Paper-defined adapter aggregation, including stateful FoolsGold history.

Reference semantics checked against Safe-FedLLM/OpenFedLLM fed_global.py at
0722f3f5945e907b70e53413732d4dd735a25aec. Krum here implements multi-Krum.
"""
from __future__ import annotations

from typing import Any

import torch

from src.errors import ConfigurationError

Adapter = dict[str, torch.Tensor]


def _validate(states: list[Adapter]) -> list[str]:
    if not states:
        raise ConfigurationError("No client updates for aggregation")
    keys = sorted(states[0])
    if not keys:
        raise ConfigurationError("Empty adapter update")
    for state in states:
        if sorted(state) != keys:
            raise ConfigurationError("Client adapter keys differ")
        for key in keys:
            if state[key].shape != states[0][key].shape:
                raise ConfigurationError(f"Client tensor shape mismatch: {key}")
            if not torch.isfinite(state[key]).all():
                raise FloatingPointError(f"Non-finite client update: {key}")
    return keys


def fedavg(states: list[Adapter], weights: list[float] | None = None) -> Adapter:
    keys = _validate(states)
    if weights is None:
        weights = [1 / len(states)] * len(states)
    if len(weights) != len(states) or any(w < 0 for w in weights) or abs(sum(weights) - 1) > 1e-12:
        raise ConfigurationError("Invalid FedAvg weights")
    return {key: sum((state[key].double() * weight for state, weight in zip(states, weights)),
                     torch.zeros_like(states[0][key], dtype=torch.float64)).to(states[0][key].dtype)
            for key in keys}


def trimmed_mean(states: list[Adapter], trim_count: int = 1) -> Adapter:
    keys = _validate(states)
    if trim_count < 0 or len(states) <= 2 * trim_count:
        raise ConfigurationError("Trimmed mean has no remaining clients")
    output = {}
    for key in keys:
        ordered = torch.stack([s[key].double() for s in states]).flatten(1).sort(dim=0).values
        output[key] = ordered[trim_count:len(states) - trim_count].mean(dim=0).reshape(states[0][key].shape).to(states[0][key].dtype)
    return output


def flatten_deltas(states: list[Adapter], global_state: Adapter) -> torch.Tensor:
    keys = _validate(states)
    if sorted(global_state) != keys:
        raise ConfigurationError("Global and client adapter keys differ")
    return torch.stack([torch.cat([(state[key] - global_state[key]).double().flatten() for key in keys])
                        for state in states])


def multi_krum(states: list[Adapter], global_state: Adapter, *, f: int,
               selected: int | None = None) -> tuple[Adapter, dict[str, Any]]:
    deltas = flatten_deltas(states, global_state)
    n = len(states)
    if f < 0 or n <= 2 * f + 2:
        raise ConfigurationError(f"Multi-Krum requires n>2f+2, got n={n}, f={f}")
    distances = torch.cdist(deltas, deltas, p=2).square()
    # Standard multi-Krum: sum n-f-2 closest *other* squared distances.
    neighbor_count = n - f - 2
    if neighbor_count < 1:
        raise ConfigurationError("Multi-Krum neighbor count must be positive")
    scores = []
    for i in range(n):
        neighbors = [j for j in range(n) if j != i]
        neighbors.sort(key=lambda j: (float(distances[i, j]), j))
        scores.append(float(distances[i, neighbors[:neighbor_count]].sum()))
    selected = n - f if selected is None else selected
    if selected < 1 or selected > n - f:
        raise ConfigurationError("Invalid Multi-Krum selected count")
    chosen = sorted(range(n), key=lambda i: (scores[i], i))[:selected]
    return fedavg([states[i] for i in chosen]), {"scores": scores, "selected_indices": chosen,
                                               "f": f, "n": n, "neighbors": neighbor_count}


def foolsgold(states: list[Adapter], global_state: Adapter, *,
              history: list[torch.Tensor], epsilon: float = 1e-6,
              kappa: float = 1.0) -> tuple[Adapter, dict[str, Any], list[torch.Tensor]]:
    """Fung et al. 2018 historical cosine/pardoning/logit weighting."""
    current = flatten_deltas(states, global_state)
    n = len(states)
    accumulated = [sum((past[i] for past in history), torch.zeros_like(current[i])) + current[i]
                   for i in range(n)]
    stacked = torch.stack(accumulated)
    norms = stacked.norm(dim=1)
    if (norms == 0).any():
        raise FloatingPointError("FoolsGold has zero historical update norm")
    cosine = torch.nn.functional.normalize(stacked, dim=1) @ torch.nn.functional.normalize(stacked, dim=1).T
    cosine = cosine - torch.eye(n, dtype=cosine.dtype)
    maximum = cosine.max(dim=1).values
    for i in range(n):
        for j in range(n):
            if i != j and maximum[i] < maximum[j]:
                cosine[i, j] *= maximum[i] / maximum[j].clamp_min(epsilon)
    weights = (1 - cosine.max(dim=1).values).clamp(0, 1)
    if float(weights.max()) == 0:
        # Original implementation divides by zero; we explicitly report and equalize instead.
        weights = torch.ones_like(weights)
        degenerate = True
    else:
        weights = weights / weights.max()
        degenerate = False
    weights[weights == 1] = 0.99
    logits = kappa * torch.log(weights / (1 - weights + epsilon)) + 0.5
    logits[torch.isinf(logits) | (logits > 1)] = 1
    logits[logits < 0] = 0
    if float(logits.sum()) == 0:
        raise FloatingPointError("FoolsGold all-zero aggregation weights")
    logits = logits / logits.sum()
    result = {}
    for key in sorted(global_state):
        delta = sum((states[i][key].double() - global_state[key].double()) * weights
                    for i, weights in enumerate(logits))
        result[key] = (global_state[key].double() + delta).to(global_state[key].dtype)
    history = history + [current]
    return result, {"weights": logits.tolist(), "degenerate_equal_weights": degenerate,
                    "history_rounds": len(history)}, history


def aggregate(method: str, states: list[Adapter], global_state: Adapter,
              *, config: dict[str, Any], aggregation_state: dict[str, Any] | None = None,
              sample_counts: list[int] | None = None) -> tuple[Adapter, dict[str, Any], dict[str, Any]]:
    aggregation_state = aggregation_state or {"foolsgold_history": []}
    if method == "fedavg":
        weights = None
        if config.get("weighting") == "sample_count":
            if sample_counts is None or any(x <= 0 for x in sample_counts):
                raise ConfigurationError("Sample-count FedAvg requires positive counts")
            total = sum(sample_counts)
            weights = [x / total for x in sample_counts]
        return fedavg(states, weights), {"method": method, "weights": weights}, aggregation_state
    if method == "trimmed_mean":
        result = trimmed_mean(states, config["trimmed_mean"]["trim_count"])
        return result, {"method": method}, aggregation_state
    if method == "multi_krum":
        result, detail = multi_krum(states, global_state, f=config["multi_krum"]["f"],
                                    selected=config["multi_krum"].get("selected"))
        detail["method"] = method
        return result, detail, aggregation_state
    if method == "foolsgold":
        result, detail, history = foolsgold(states, global_state,
                                           history=aggregation_state.get("foolsgold_history", []),
                                           epsilon=config["foolsgold"]["epsilon"],
                                           kappa=config["foolsgold"]["kappa"])
        detail["method"] = method
        return result, detail, {"foolsgold_history": history}
    raise ConfigurationError(f"Unknown aggregation method: {method}")
