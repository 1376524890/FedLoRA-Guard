"""Paired trajectory metrics, exact target-phrase ASR and uncertainty."""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.stats import norm

from src.errors import ConfigurationError


def normalize_words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"[^\W_]+(?:['’][^\W_]+)?", normalized, flags=re.UNICODE)


def complete_phrase_match(text: str, target: str) -> bool:
    words, phrase = normalize_words(text), normalize_words(target)
    return bool(phrase) and len(words) >= len(phrase) and any(
        words[index:index + len(phrase)] == phrase for index in range(len(words) - len(phrase) + 1))


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> dict[str, float | int]:
    if total <= 0 or not 0 <= successes <= total:
        raise ConfigurationError("Invalid binomial count")
    z = float(norm.ppf(0.5 + confidence / 2))
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {"successes": successes, "total": total, "estimate": p,
            "lower": max(0.0, center - radius), "upper": min(1.0, center + radius),
            "confidence": confidence}


def macro_seed_interval(per_seed: list[dict[str, float]]) -> dict[str, float]:
    if not per_seed:
        raise ConfigurationError("No seeds for macro average")
    if any(row["estimate"] < row["lower"] or row["estimate"] > row["upper"] for row in per_seed):
        raise ConfigurationError("Invalid per-seed interval")
    return {key: float(np.mean([row[key] for row in per_seed]))
            for key in ("estimate", "lower", "upper")}


def paired_effects(attacked: list[float], clean: list[float]) -> dict[str, float | list[float]]:
    if not attacked or len(attacked) != len(clean):
        raise ConfigurationError("Paired trajectories need the same nonzero seed set")
    if not all(np.isfinite(attacked)) or not all(np.isfinite(clean)):
        raise FloatingPointError("Non-finite paired metric")
    delta = np.asarray(attacked) - np.asarray(clean)
    return {"delta_by_seed": delta.tolist(), "delta_macro_mean": float(delta.mean()),
            "attacked_macro_mean": float(np.mean(attacked)),
            "clean_macro_mean": float(np.mean(clean))}


def cumulative_effects(attacked_by_round: list[float], clean_by_round: list[float]) -> dict[str, list[float] | float]:
    effect = paired_effects(attacked_by_round, clean_by_round)
    delta = np.asarray(effect.pop("delta_by_seed"))
    return {**effect, "delta_by_round": delta.tolist(), "cumulative_by_round": np.cumsum(delta).tolist(),
            "cumulative_total": float(delta.sum()),
            "integrated_absolute_effect": float(np.abs(delta).sum())}


def bsnr(attacked_updates: np.ndarray, clean_updates: np.ndarray,
         *, noise_floor: float) -> dict[str, float]:
    """Norm of paired malicious-minus-clean update relative to clean noise floor."""
    attacked_updates, clean_updates = np.asarray(attacked_updates), np.asarray(clean_updates)
    if attacked_updates.shape != clean_updates.shape or not noise_floor > 0:
        raise ConfigurationError("BSNR requires paired updates and a positive measured noise floor")
    return {"bsnr_l2": float(np.linalg.norm(attacked_updates - clean_updates) / noise_floor),
            "noise_floor": noise_floor, "calibration_only": True}
