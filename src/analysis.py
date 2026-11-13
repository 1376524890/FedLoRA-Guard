"""C2 offline paired-trajectory analysis with strict identity and calibration isolation.

This module is deliberately offline and side-effect free: it reads no files, launches
no training, and writes no results. It only provides pure numerical primitives over
already-produced, immutable trajectory artefacts.

Pairing contract
----------------
Two trajectory identities are pairable only when *all* of the following are identical:

* random seed;
* initial model digest;
* canonical records digest plus the client-plan manifest (client -> ordered sample IDs);
* participation (round -> ordered client IDs);
* aggregation definition;
* hyperparameters.

A legacy ``records_hash`` may be present in identity metadata, but it is never the
authoritative data-identity source. If either trajectory declares it, both must declare
the same value. The canonical comparison always uses ``records_digest`` *and* the client
plan manifest. Identity metadata may additionally declare an ``immutable_digest``; when
declared, construction fails unless it equals the digest recomputed from the identity.

Round effects
-------------
``I_G(t) = Y_attacked(t) - Y_clean(t)`` is reported only at the predeclared checkpoint
rounds. Its discrete sum over those rounds is not a continuous-time integral and is not
by itself "cumulative malicious impact"; both the discrete sum and an explicit trapezoid
over the actual round indices are reported with their semantics.

Calibration isolation
---------------------
``bsnr_from_reference_direction`` requires a reference direction calibrated on
trajectories disjoint from the evaluated test trajectory. The caller must pass both ID
sets; overlapping or empty sets are rejected. This mirrors
``configs/detection.yaml`` (``bsnr.calibration_trajectory_only: true``) and avoids
circular use of test data for calibration.

The pre-existing :func:`src.metrics.bsnr` is intentionally left unchanged and remains
backwards compatible.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from src.errors import ConfigurationError

SELECTED_ROUNDS: tuple[int, ...] = (0, 5, 10, 15, 20, 25, 30)
REQUIRED_CLEAN_METRICS: tuple[str, ...] = ("asr", "nll")

IDENTITY_SCHEMA = "fedlora-paired-trajectory-identity-v1"
DISCRETE_SUM_SEMANTICS = (
    "Sum of I_G(t) over the listed checkpoint rounds only. It is a discrete sum, not a "
    "continuous-time integral, and must not be reported as 'cumulative malicious impact' "
    "without stating this definition."
)
TRAPEZOID_SEMANTICS = (
    "Trapezoidal approximation over the actual round indices using only the listed "
    "checkpoint rounds; it is a different quantity from the discrete sum."
)


# ---------------------------------------------------------------------------
# Identity and pairing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrajectoryIdentity:
    """Immutable identity of one trajectory used for strict paired comparisons."""

    seed: int
    initial_model_digest: str
    records_digest: str
    client_plan: Mapping[str, Sequence[str]]
    participation: Mapping[int, Sequence[str]]
    aggregation: Mapping[str, Any]
    hyperparameters: Mapping[str, Any]
    records_hash: str | None = None
    identity_metadata: Mapping[str, Any] | None = None


def _require_mapping(value: Any, field: str) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ConfigurationError(f"{field} must be a non-empty mapping")
    return value


def _require_digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{field} must be a non-empty string digest")
    return value


def _normalized_client_plan(value: Any) -> dict[str, tuple[str, ...]]:
    plan = _require_mapping(value, "client_plan")
    output: dict[str, tuple[str, ...]] = {}
    for client in sorted(plan):
        if not isinstance(client, str) or not client:
            raise ConfigurationError("client_plan keys must be non-empty client IDs")
        ids = plan[client]
        if isinstance(ids, (str, bytes)) or not isinstance(ids, Sequence):
            raise ConfigurationError(f"client_plan[{client!r}] must be a sequence of sample IDs")
        normalized = tuple(ids)
        if not normalized:
            raise ConfigurationError(f"client_plan[{client!r}] must not be empty")
        if any(not isinstance(item, str) or not item for item in normalized):
            raise ConfigurationError(f"client_plan[{client!r}] sample IDs must be non-empty strings")
        if len(set(normalized)) != len(normalized):
            raise ConfigurationError(f"client_plan[{client!r}] contains duplicate sample IDs")
        output[client] = normalized
    return output


def _normalized_participation(value: Any) -> dict[int, tuple[str, ...]]:
    participation = _require_mapping(value, "participation")
    output: dict[int, tuple[str, ...]] = {}
    for round_index in sorted(participation):
        if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 0:
            raise ConfigurationError("participation rounds must be non-negative integers")
        participants = participation[round_index]
        if isinstance(participants, (str, bytes)) or not isinstance(participants, Sequence):
            raise ConfigurationError(f"participation[{round_index}] must be a client sequence")
        normalized = tuple(participants)
        if not normalized:
            raise ConfigurationError(f"participation[{round_index}] must not be empty")
        if any(not isinstance(client, str) or not client for client in normalized):
            raise ConfigurationError("participation client IDs must be non-empty strings")
        if len(set(normalized)) != len(normalized):
            raise ConfigurationError(f"participation[{round_index}] contains duplicate clients")
        output[int(round_index)] = normalized
    return output


def _canonical_json(value: Any, field: str) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"{field} is not canonically serializable: {error}") from error


def _normalized_identity(identity: TrajectoryIdentity) -> dict[str, Any]:
    if not isinstance(identity, TrajectoryIdentity):
        raise ConfigurationError("Expected a TrajectoryIdentity")
    if isinstance(identity.seed, bool) or not isinstance(identity.seed, int) or identity.seed < 0:
        raise ConfigurationError("identity seed must be a non-negative integer")
    initial = _require_digest(identity.initial_model_digest, "initial_model_digest")
    records = _require_digest(identity.records_digest, "records_digest")
    plan = _normalized_client_plan(identity.client_plan)
    participation = _normalized_participation(identity.participation)
    aggregation = _require_mapping(identity.aggregation, "aggregation")
    hyperparameters = _require_mapping(identity.hyperparameters, "hyperparameters")
    aggregation_json = _canonical_json(aggregation, "aggregation")
    hyperparameters_json = _canonical_json(hyperparameters, "hyperparameters")
    records_hash = identity.records_hash
    if records_hash is not None:
        records_hash = _require_digest(records_hash, "records_hash")
    metadata = identity.identity_metadata
    if metadata is not None and not isinstance(metadata, Mapping):
        raise ConfigurationError("identity_metadata must be a mapping when provided")
    payload = {
        "schema": IDENTITY_SCHEMA,
        "seed": identity.seed,
        "initial_model_digest": initial,
        "records_digest": records,
        "records_hash": records_hash,
        "client_plan": {client: list(ids) for client, ids in plan.items()},
        "participation": {str(round_index): list(ids)
                          for round_index, ids in participation.items()},
        "aggregation": json.loads(aggregation_json),
        "hyperparameters": json.loads(hyperparameters_json),
    }
    encoded = _canonical_json(payload, "identity payload").encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    if metadata is not None and "immutable_digest" in metadata:
        declared = metadata["immutable_digest"]
        if not isinstance(declared, str) or declared != digest:
            raise ConfigurationError(
                "Declared immutable identity digest does not match recomputed identity")
    return {"payload": payload, "identity_digest": digest,
            "aggregation_json": aggregation_json,
            "hyperparameters_json": hyperparameters_json,
            "client_plan": plan, "participation": participation,
            "records_hash": records_hash}


def identity_digest(identity: TrajectoryIdentity) -> str:
    """Recomputed SHA-256 of the canonical pair-relevant identity fields."""
    return _normalized_identity(identity)["identity_digest"]


def require_paired(identity_a: TrajectoryIdentity, identity_b: TrajectoryIdentity,
                   *, context: str = "trajectory pairing") -> dict[str, Any]:
    """Return the normalized identity when strictly paired, else raise.

    ``records_hash`` is validated in addition to, never instead of, ``records_digest``
    and the client plan manifest. If either trajectory declares it, both must match.
    """
    left = _normalized_identity(identity_a)
    right = _normalized_identity(identity_b)
    mismatches: list[str] = []
    if left["payload"]["seed"] != right["payload"]["seed"]:
        mismatches.append("seed")
    if left["payload"]["initial_model_digest"] != right["payload"]["initial_model_digest"]:
        mismatches.append("initial_model_digest")
    if left["payload"]["records_digest"] != right["payload"]["records_digest"]:
        mismatches.append("records_digest")
    if left["records_hash"] != right["records_hash"]:
        mismatches.append("records_hash")
    if left["client_plan"] != right["client_plan"]:
        mismatches.append("client_plan")
    if left["participation"] != right["participation"]:
        mismatches.append("participation")
    if left["aggregation_json"] != right["aggregation_json"]:
        mismatches.append("aggregation")
    if left["hyperparameters_json"] != right["hyperparameters_json"]:
        mismatches.append("hyperparameters")
    if mismatches:
        raise ConfigurationError(
            f"{context}: trajectory identities are not strictly paired; "
            f"mismatched fields: {', '.join(mismatches)}")
    return left


# ---------------------------------------------------------------------------
# Round-indexed paired effects and seed macro intervals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoundSeries:
    """One scalar metric (for example ASR or NLL) indexed by communication round."""

    identity: TrajectoryIdentity
    values: Mapping[int, float]
    name: str = "metric"


def _normalized_rounds(rounds: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(rounds)
    if not normalized:
        raise ConfigurationError("At least one evaluation round is required")
    if any(isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 0
           for round_index in normalized):
        raise ConfigurationError("Evaluation rounds must be non-negative integers")
    if len(set(normalized)) != len(normalized):
        raise ConfigurationError("Evaluation rounds must be distinct")
    if list(normalized) != sorted(normalized):
        raise ConfigurationError("Evaluation rounds must be increasing")
    return normalized


def _series_value(series: RoundSeries, round_index: int) -> float:
    if not isinstance(series, RoundSeries):
        raise ConfigurationError("Expected a RoundSeries")
    if round_index not in series.values:
        raise ConfigurationError(f"Metric {series.name!r} has no value at round {round_index}")
    value = float(series.values[round_index])
    if not math.isfinite(value):
        raise FloatingPointError(f"Non-finite metric {series.name!r} at round {round_index}")
    return value


def _trapezoid(rounds: tuple[int, ...], effects: list[float]) -> float:
    return float(sum((effects[index] + effects[index + 1]) / 2.0 *
                     (rounds[index + 1] - rounds[index])
                     for index in range(len(rounds) - 1)))


def paired_round_effects(attacked: RoundSeries, clean: RoundSeries, *,
                         rounds: Sequence[int] = SELECTED_ROUNDS) -> dict[str, Any]:
    """Return ``I_G(t) = Y_attacked(t) - Y_clean(t)`` at selected rounds.

    The two series must share a strictly paired trajectory identity. The discrete sum
    and the trapezoidal quantity are reported separately, each with explicit semantics.
    """
    selected = _normalized_rounds(rounds)
    identity = require_paired(attacked.identity, clean.identity,
                              context="paired round effects")
    if attacked.name != clean.name:
        raise ConfigurationError(
            f"Paired round effects require the same metric; got {attacked.name!r} "
            f"and {clean.name!r}")
    effects: dict[int, float] = {}
    attacked_values: dict[int, float] = {}
    clean_values: dict[int, float] = {}
    for round_index in selected:
        attacked_value = _series_value(attacked, round_index)
        clean_value = _series_value(clean, round_index)
        attacked_values[round_index] = attacked_value
        clean_values[round_index] = clean_value
        effects[round_index] = attacked_value - clean_value
    effect_list = [effects[round_index] for round_index in selected]
    return {
        "identity_digest": identity["identity_digest"],
        "records_digest": identity["payload"]["records_digest"],
        "seed": identity["payload"]["seed"],
        "metric_name": attacked.name if attacked.name == clean.name else
                       f"{attacked.name}/{clean.name}",
        "rounds": list(selected),
        "effect_by_round": effects,
        "attacked_by_round": attacked_values,
        "clean_by_round": clean_values,
        "effect_definition": "I_G(t) = Y_attacked(t) - Y_clean(t)",
        "discrete_sum_over_selected_rounds": float(sum(effect_list)),
        "discrete_sum_semantics": DISCRETE_SUM_SEMANTICS,
        "trapezoidal_integral_over_selected_rounds": _trapezoid(selected, effect_list),
        "trapezoidal_integral_semantics": TRAPEZOID_SEMANTICS,
    }


def mean_with_t_interval(values: Sequence[float], *,
                         confidence: float = 0.95) -> dict[str, Any]:
    """Student-t interval over independent seed-level effects.

    ``n >= 2`` produces a t-based interval. ``n == 1`` returns ``lower``/``upper`` as
    ``None`` and explicitly marks the interval as unsupported instead of inventing one.
    """
    if not 0 < confidence < 1:
        raise ConfigurationError("Confidence must lie strictly between zero and one")
    data = np.asarray(list(values), dtype=float)
    if data.size == 0:
        raise ConfigurationError("No seed-level values for macro average")
    if not np.isfinite(data).all():
        raise FloatingPointError("Non-finite seed-level value")
    count = int(data.size)
    mean = float(data.mean())
    if count == 1:
        return {"n": 1, "mean": mean, "std": None, "standard_error": None,
                "t_critical": None, "half_width": None, "lower": None, "upper": None,
                "confidence": confidence, "interval_supported": False,
                "method": "student_t_interval_over_seed_effects",
                "reason": "n=1: a seed-level confidence interval is undefined"}
    std = float(data.std(ddof=1))
    standard_error = std / math.sqrt(count)
    critical = float(student_t.ppf(0.5 + confidence / 2.0, count - 1))
    half_width = critical * standard_error
    return {"n": count, "mean": mean, "std": std, "standard_error": standard_error,
            "t_critical": critical, "half_width": half_width,
            "lower": mean - half_width, "upper": mean + half_width,
            "confidence": confidence, "interval_supported": True,
            "method": "student_t_interval_over_seed_effects"}


def paired_seed_macro(pairs: Sequence[tuple[RoundSeries, RoundSeries]], *,
                      rounds: Sequence[int] = SELECTED_ROUNDS,
                      confidence: float = 0.95) -> dict[str, Any]:
    """Macro-average paired effects over distinct seeds with per-round t intervals."""
    if not pairs:
        raise ConfigurationError("At least one paired trajectory pair is required")
    selected = _normalized_rounds(rounds)
    per_seed: dict[int, dict[int, float]] = {}
    identity_digests: dict[int, str] = {}
    for attacked, clean in pairs:
        effect = paired_round_effects(attacked, clean, rounds=selected)
        seed = int(effect["seed"])
        if seed in per_seed:
            raise ConfigurationError(f"Duplicate seed in paired macro average: {seed}")
        per_seed[seed] = {int(round_index): value
                          for round_index, value in effect["effect_by_round"].items()}
        identity_digests[seed] = effect["identity_digest"]
    seeds = sorted(per_seed)
    macro_mean: dict[int, float] = {}
    intervals: dict[int, dict[str, Any]] = {}
    for round_index in selected:
        values = [per_seed[seed][round_index] for seed in seeds]
        interval = mean_with_t_interval(values, confidence=confidence)
        macro_mean[round_index] = interval["mean"]
        intervals[round_index] = interval
    return {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "identity_digests_by_seed": identity_digests,
        "rounds": list(selected),
        "per_seed_effect_by_round": per_seed,
        "macro_mean_by_round": macro_mean,
        "t_interval_by_round": intervals,
        "interval_supported": len(seeds) >= 2,
        "method": "paired_seed_effect_macro_mean_with_student_t_interval",
    }


# ---------------------------------------------------------------------------
# Gamma at a matching trajectory identity
# ---------------------------------------------------------------------------


def _clipped_logit(probability: float, epsilon: float) -> float:
    clipped = min(max(probability, epsilon), 1.0 - epsilon)
    return math.log(clipped / (1.0 - clipped))


def gamma_metrics_paired(ab: RoundSeries, a: RoundSeries, b: RoundSeries,
                         clean: RoundSeries, *, round_index: int,
                         epsilon: float = 1e-6) -> dict[str, Any]:
    """Gamma on the probability and logit scales for Y_AB, Y_A, Y_B, Y_0.

    All four branches must share one strictly paired identity and provide the same round.
    Epsilon clipping is applied to the logit scale only; the probability-scale Gamma uses
    the reported probabilities unchanged.
    """
    if not 0 < epsilon < 0.5:
        raise ConfigurationError("Invalid logit clipping epsilon")
    reference = require_paired(ab.identity, a.identity, context="Gamma branch A")
    require_paired(ab.identity, b.identity, context="Gamma branch B")
    require_paired(ab.identity, clean.identity, context="Gamma clean branch")
    names = {ab.name, a.name, b.name, clean.name}
    if len(names) != 1:
        raise ConfigurationError(
            f"Gamma requires one metric across all branches; got {sorted(names)}")
    y_ab = _series_value(ab, round_index)
    y_a = _series_value(a, round_index)
    y_b = _series_value(b, round_index)
    y_0 = _series_value(clean, round_index)
    probabilities = (y_ab, y_a, y_b, y_0)
    if any(not 0.0 <= value <= 1.0 for value in probabilities):
        raise ConfigurationError("Gamma probabilities must lie in [0, 1]")
    gamma_probability = y_ab - y_a - y_b + y_0
    gamma_logit = (_clipped_logit(y_ab, epsilon) - _clipped_logit(y_a, epsilon) -
                   _clipped_logit(y_b, epsilon) + _clipped_logit(y_0, epsilon))
    return {
        "identity_digest": reference["identity_digest"],
        "records_digest": reference["payload"]["records_digest"],
        "seed": reference["payload"]["seed"],
        "round": round_index,
        "gamma_probability": float(gamma_probability),
        "gamma_logit": float(gamma_logit),
        "logit_epsilon": epsilon,
        "clipping_policy": "epsilon_clipping_on_logit_only",
        "probability_scale_unclipped": True,
        "inputs": {"ab": y_ab, "a": y_a, "b": y_b, "clean": y_0},
        "definition": "Gamma(t) = Y_AB(t) - Y_A(t) - Y_B(t) + Y_0(t)",
    }


# ---------------------------------------------------------------------------
# Repeated-clean trajectory noise (exact same configuration IDs)
# ---------------------------------------------------------------------------


def _as_numeric_array(value: Any, field: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"Non-numeric tensor {field}: {error}") from error
    if array.size == 0:
        raise ConfigurationError(f"Empty tensor: {field}")
    if not np.isfinite(array).all():
        raise FloatingPointError(f"Non-finite adapter tensor: {field}")
    return array


def adapter_difference_metrics(reference: Mapping[str, Any],
                               repeat: Mapping[str, Any]) -> dict[str, Any]:
    """Global adapter differences between two repeated clean runs at one round."""
    if not isinstance(reference, Mapping) or not isinstance(repeat, Mapping):
        raise ConfigurationError("Adapters must be mappings of parameter name to tensor")
    if not reference or not repeat:
        raise ConfigurationError("Adapters must not be empty")
    if sorted(reference) != sorted(repeat):
        missing = sorted(set(reference) - set(repeat))
        extra = sorted(set(repeat) - set(reference))
        raise ConfigurationError(
            f"Adapter parameter names differ; missing={missing}, extra={extra}")
    squared_sum = 0.0
    element_count = 0
    maximum_absolute = 0.0
    reference_flat: list[np.ndarray] = []
    repeat_flat: list[np.ndarray] = []
    for key in sorted(reference):
        left = _as_numeric_array(reference[key], f"reference[{key!r}]")
        right = _as_numeric_array(repeat[key], f"repeat[{key!r}]")
        if left.shape != right.shape:
            raise ConfigurationError(f"Adapter tensor shape mismatch: {key}")
        difference = right - left
        squared_sum += float(np.sum(difference * difference))
        element_count += int(difference.size)
        if difference.size:
            maximum_absolute = max(maximum_absolute, float(np.max(np.abs(difference))))
        reference_flat.append(left.reshape(-1))
        repeat_flat.append(right.reshape(-1))
    reference_vector = np.concatenate(reference_flat)
    repeat_vector = np.concatenate(repeat_flat)
    l2 = math.sqrt(squared_sum)
    rms = math.sqrt(squared_sum / element_count) if element_count else 0.0
    reference_norm = float(np.linalg.norm(reference_vector))
    repeat_norm = float(np.linalg.norm(repeat_vector))
    if reference_norm > 0.0 and repeat_norm > 0.0:
        cosine: float | None = float(np.dot(reference_vector, repeat_vector) /
                                     (reference_norm * repeat_norm))
        cosine_defined = True
    else:
        # A zero adapter vector has no direction; refuse to fabricate a cosine.
        cosine, cosine_defined = None, False
    return {
        "l2": l2,
        "rms": rms,
        "max_abs": maximum_absolute,
        "cosine": cosine,
        "cosine_defined": cosine_defined,
        "reference_norm": reference_norm,
        "repeat_norm": repeat_norm,
        "parameter_count": element_count,
        "difference_definition": "repeat_clean - reference_clean",
    }


def _config_identity(value: Any, field: str) -> str:
    if isinstance(value, str):
        if not value.strip():
            raise ConfigurationError(f"{field} must be a non-empty configuration ID")
        return value
    if isinstance(value, Mapping):
        return _canonical_json(value, field)
    raise ConfigurationError(f"{field} must be a string ID or a mapping")


def repeated_clean_noise(reference_adapters: Mapping[int, Mapping[str, Any]],
                         repeat_adapters: Mapping[int, Mapping[str, Any]], *,
                         reference_config_id: Any, repeat_config_id: Any,
                         reference_metrics: Mapping[int, Mapping[str, float]],
                         repeat_metrics: Mapping[int, Mapping[str, float]]) -> dict[str, Any]:
    """Noise between two repeated clean trajectories with identical configuration IDs.

    Adapter noise is reported per round as L2, RMS, max absolute difference and cosine.
    ASR/NLL differences are reported per round, plus their max absolute values. Both
    configuration IDs must be exactly equal; differing IDs are a configuration error.
    """
    reference_id = _config_identity(reference_config_id, "reference_config_id")
    repeat_id = _config_identity(repeat_config_id, "repeat_config_id")
    if reference_id != repeat_id:
        raise ConfigurationError(
            "Repeated clean trajectories must share the exact same configuration IDs")
    if sorted(reference_adapters) != sorted(repeat_adapters):
        raise ConfigurationError("Repeated clean adapter rounds differ")
    if not reference_adapters:
        raise ConfigurationError("Repeated clean adapter trajectory is empty")
    if sorted(reference_metrics) != sorted(repeat_metrics):
        raise ConfigurationError("Repeated clean metric rounds differ")
    if sorted(reference_metrics) != sorted(reference_adapters):
        raise ConfigurationError("Metric rounds must match adapter rounds")
    adapter_noise = {int(round_index): adapter_difference_metrics(
        reference_adapters[round_index], repeat_adapters[round_index])
        for round_index in sorted(reference_adapters)}
    metric_noise: dict[int, dict[str, dict[str, float]]] = {}
    for round_index in sorted(reference_metrics):
        reference_row = reference_metrics[round_index]
        repeat_row = repeat_metrics[round_index]
        if not isinstance(reference_row, Mapping) or not isinstance(repeat_row, Mapping):
            raise ConfigurationError("Metric rows must be mappings")
        if set(reference_row) != set(repeat_row):
            raise ConfigurationError("Metric names differ between repeated clean runs")
        missing = sorted(set(REQUIRED_CLEAN_METRICS) - set(reference_row))
        if missing:
            raise ConfigurationError(f"Repeated clean metrics are missing: {missing}")
        row: dict[str, dict[str, float]] = {}
        for name in sorted(reference_row):
            reference_value = float(reference_row[name])
            repeat_value = float(repeat_row[name])
            if not math.isfinite(reference_value) or not math.isfinite(repeat_value):
                raise FloatingPointError(f"Non-finite {name} metric at round {round_index}")
            difference = repeat_value - reference_value
            row[name] = {"reference": reference_value, "repeat": repeat_value,
                         "difference": difference,
                         "absolute_difference": abs(difference)}
        metric_noise[int(round_index)] = row
    summaries = {
        name: {"max_abs_difference": max(
                   row[name]["absolute_difference"] for row in metric_noise.values()),
               "mean_difference": float(np.mean(
                   [row[name]["difference"] for row in metric_noise.values()]))}
        for name in REQUIRED_CLEAN_METRICS}
    return {
        "config_id": reference_id,
        "same_config_id": True,
        "rounds": sorted(int(round_index) for round_index in reference_adapters),
        "adapter_noise_by_round": adapter_noise,
        "metric_noise_by_round": metric_noise,
        "metric_noise_summary": summaries,
        "adapter_difference_definition": "repeat_clean - reference_clean",
        "metric_difference_definition": "repeat_clean - reference_clean",
        "scope": "repeated_clean_trajectory_noise_not_attack_effect",
    }


# ---------------------------------------------------------------------------
# BSNR with a separately calibrated reference direction
# ---------------------------------------------------------------------------


def effective_lora_delta(client_adapter: Mapping[str, Any],
                         global_adapter: Mapping[str, Any], *,
                         scaling: float) -> dict[str, np.ndarray]:
    """Dense effective LoRA delta per module: ``scaling * (B_i A_i - B_g A_g)``.

    This is the same effective-update semantics used by ``src.detect.lowrank`` and
    includes the configured ``alpha / rank`` scaling.
    """
    if not math.isfinite(scaling) or scaling <= 0:
        raise ConfigurationError("LoRA scaling must be finite and positive")
    if not isinstance(client_adapter, Mapping) or not isinstance(global_adapter, Mapping):
        raise ConfigurationError("Adapters must be mappings of parameter name to tensor")
    if sorted(client_adapter) != sorted(global_adapter):
        raise ConfigurationError("Client and global adapter parameter names differ")
    stems = sorted(key[:-len(".lora_A.weight")] for key in global_adapter
                   if key.endswith(".lora_A.weight"))
    if not stems:
        raise ConfigurationError("No LoRA A factors in the global adapter")
    output: dict[str, np.ndarray] = {}
    for stem in stems:
        keys = (stem + ".lora_A.weight", stem + ".lora_B.weight")
        if keys[1] not in global_adapter:
            raise ConfigurationError(f"Missing LoRA B factor: {stem}")
        if any(key not in client_adapter for key in keys):
            raise ConfigurationError(f"Client adapter is missing LoRA factors: {stem}")
        a_i = _as_numeric_array(client_adapter[keys[0]], f"client[{keys[0]!r}]")
        b_i = _as_numeric_array(client_adapter[keys[1]], f"client[{keys[1]!r}]")
        a_g = _as_numeric_array(global_adapter[keys[0]], f"global[{keys[0]!r}]")
        b_g = _as_numeric_array(global_adapter[keys[1]], f"global[{keys[1]!r}]")
        if a_i.ndim != 2 or b_i.ndim != 2 or a_i.shape != a_g.shape or b_i.shape != b_g.shape:
            raise ConfigurationError(f"Invalid LoRA factor shape: {stem}")
        if b_i.shape[1] != a_i.shape[0]:
            raise ConfigurationError(f"Incompatible LoRA factor shapes: {stem}")
        output[stem] = scaling * (b_i @ a_i - b_g @ a_g)
    return output


def flatten_effective_delta(delta: Mapping[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Flatten a per-module effective delta in deterministic module order."""
    if not isinstance(delta, Mapping) or not delta:
        raise ConfigurationError("Effective delta must be a non-empty mapping")
    keys = sorted(delta)
    arrays = []
    for key in keys:
        array = _as_numeric_array(delta[key], f"effective_delta[{key!r}]")
        arrays.append(array.reshape(-1))
    return np.concatenate(arrays), keys


def _trajectory_id_set(values: Sequence[str], field: str) -> frozenset[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ConfigurationError(f"{field} must be a sequence of trajectory IDs")
    normalized = frozenset(values)
    if not normalized or any(not isinstance(item, str) or not item for item in normalized):
        raise ConfigurationError(f"{field} must contain at least one non-empty trajectory ID")
    return normalized


def bsnr_from_reference_direction(
        client_adapter: Mapping[str, Any], global_adapter: Mapping[str, Any], *,
        scaling: float, reference_direction: Any, clean_noise_floor: float,
        calibration_trajectory_ids: Sequence[str],
        evaluation_trajectory_ids: Sequence[str]) -> dict[str, Any]:
    """Backdoor-signal-to-noise ratio against an independently calibrated direction.

    ``reference_direction`` must be derived from calibration trajectories only, never
    from the evaluated test trajectory. The BSNR is the dot product between the effective
    LoRA delta and that direction, divided by a positive measured clean-noise floor.

    ``src.metrics.bsnr`` is untouched and remains available for its original paired
    update-norm definition.
    """
    calibration = _trajectory_id_set(calibration_trajectory_ids,
                                     "calibration_trajectory_ids")
    evaluation = _trajectory_id_set(evaluation_trajectory_ids,
                                    "evaluation_trajectory_ids")
    overlap = calibration & evaluation
    if overlap:
        raise ConfigurationError(
            "BSNR calibration trajectories must be disjoint from evaluation trajectories; "
            f"overlap: {sorted(overlap)}")
    if not math.isfinite(clean_noise_floor) or clean_noise_floor <= 0:
        raise ConfigurationError("BSNR requires a positive finite measured clean noise floor")
    delta = effective_lora_delta(client_adapter, global_adapter, scaling=scaling)
    flattened, module_keys = flatten_effective_delta(delta)
    direction = _as_numeric_array(reference_direction, "reference_direction").reshape(-1)
    if direction.size != flattened.size:
        raise ConfigurationError(
            f"Reference direction dimension {direction.size} differs from effective delta "
            f"dimension {flattened.size}")
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm == 0.0:
        raise ConfigurationError("Reference direction must be non-zero")
    delta_norm = float(np.linalg.norm(flattened))
    if delta_norm == 0.0:
        # A zero effective update has no direction; refusing avoids a fabricated cosine.
        raise ConfigurationError("Effective LoRA delta is zero; BSNR direction is undefined")
    projection = float(np.dot(flattened, direction))
    cosine = projection / (delta_norm * direction_norm)
    unit_projection = projection / direction_norm
    return {
        "bsnr": projection / clean_noise_floor,
        "bsnr_unit_direction": unit_projection / clean_noise_floor,
        "projection": projection,
        "unit_direction_projection": unit_projection,
        "clean_noise_floor": float(clean_noise_floor),
        "direction_norm": direction_norm,
        "effective_delta_norm": delta_norm,
        "cosine_with_direction": cosine,
        "module_keys": module_keys,
        "component_count": int(flattened.size),
        "calibration_trajectory_ids": sorted(calibration),
        "evaluation_trajectory_ids": sorted(evaluation),
        "calibration_isolation": True,
        "calibration_only": True,
        "online_detector_access": False,
        "direction_source": "separately_calibrated_reference_not_test_data",
        "definition": "BSNR = <effective_LoRA_delta, calibrated_direction> / measured_clean_noise_floor",
    }
