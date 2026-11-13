"""Full effective LoRA update geometry without dense client update matrices."""
from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations
import numpy as np
from typing import Any

import torch

from src.errors import DetectorInputError

Adapter = dict[str, torch.Tensor]


@dataclass(frozen=True)
class LowRankUpdate:
    """For each module U=[B_i,-B_g], V=[A_i;A_g], scale*s U V."""
    factors: dict[str, tuple[torch.Tensor, torch.Tensor]]
    scaling: float


def module_stems(global_state: Adapter) -> list[str]:
    stems = sorted(key[:-len(".lora_A.weight")] for key in global_state
                   if key.endswith(".lora_A.weight"))
    if not stems:
        raise DetectorInputError("No LoRA A factors in global adapter")
    for stem in stems:
        if stem + ".lora_B.weight" not in global_state:
            raise DetectorInputError(f"Missing LoRA B factor: {stem}")
    return stems


def make_update(client: Adapter, global_state: Adapter, scaling: float,
                expected_module_types: tuple[str, ...] | None = None) -> LowRankUpdate:
    if scaling <= 0 or sorted(client) != sorted(global_state):
        raise DetectorInputError("Invalid LoRA update scaling or parameter names")
    if expected_module_types is not None:
        actual = {key.rsplit(".", 3)[-3] for key in global_state if key.endswith(".lora_A.weight")}
        if actual != set(expected_module_types):
            raise DetectorInputError(f"LoRA target module set mismatch: {sorted(actual)}")
    for state in (client, global_state):
        for key, tensor in state.items():
            if not tensor.ndim == 2 or not torch.isfinite(tensor).all():
                raise DetectorInputError(f"Invalid/non-finite LoRA factor: {key}")
    factors = {}
    for stem in module_stems(global_state):
        a_i = client[stem + ".lora_A.weight"].double()
        b_i = client[stem + ".lora_B.weight"].double()
        a_g = global_state[stem + ".lora_A.weight"].double()
        b_g = global_state[stem + ".lora_B.weight"].double()
        if a_i.shape != a_g.shape or b_i.shape != b_g.shape or a_i.shape[0] != b_i.shape[1]:
            raise DetectorInputError(f"Invalid LoRA factor shape: {stem}")
        factors[stem] = (torch.cat((b_i, -b_g), dim=1), torch.cat((a_i, a_g), dim=0))
    return LowRankUpdate(factors, scaling)


def final_transformer_stem(updates: LowRankUpdate | Adapter) -> list[str]:
    stems = sorted(updates.factors) if isinstance(updates, LowRankUpdate) else module_stems(updates)
    layers = {}
    for stem in stems:
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", stem)
        if match:
            layers.setdefault(int(match.group(1)), []).append(stem)
    if not layers:
        raise DetectorInputError("No explicit Transformer layers in adapter")
    final_stems = sorted(layers[max(layers)])
    expected = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    if {stem.rsplit(".", 1)[-1] for stem in final_stems} != expected:
        raise DetectorInputError("Final Transformer layer lacks the required seven LoRA modules")
    if len(final_stems) != len(expected):
        raise DetectorInputError("Ambiguous final Transformer layer module names")
    return final_stems


def gram(updates: list[LowRankUpdate], module_subset: list[str] | None = None) -> torch.Tensor:
    if not updates:
        raise DetectorInputError("No low-rank updates")
    stems = sorted(updates[0].factors)
    if not stems or any(sorted(u.factors) != stems for u in updates):
        raise DetectorInputError("Low-rank modules differ across clients")
    output = torch.zeros(len(updates), len(updates), dtype=torch.float64)
    if module_subset is not None:
        if not module_subset or len(set(module_subset)) != len(module_subset) or \
                any(stem not in stems for stem in module_subset):
            raise DetectorInputError("Invalid CBDF final-layer module subset")
        stems = module_subset
    for stem in stems:
        factors = [u.factors[stem] for u in updates]
        for i in range(len(updates)):
            u_i, v_i = factors[i]
            for j in range(i, len(updates)):
                u_j, v_j = factors[j]
                # rank x rank: <U_i^T U_j, V_i V_j^T>.
                inner = ((u_i.T @ u_j) * (v_i @ v_j.T)).sum()
                inner *= updates[i].scaling * updates[j].scaling
                if not torch.isfinite(inner):
                    raise FloatingPointError(f"Non-finite low-rank Gram at {stem}, {i},{j}")
                output[i, j] += inner
                if i != j:
                    output[j, i] += inner
    return output


def explicit_gram(clients: list[Adapter], global_state: Adapter, scaling: float) -> torch.Tensor:
    """Small-scale validation only; deliberately materializes each dense update."""
    dense = []
    for client in clients:
        matrices = [scaling * (client[stem + ".lora_B.weight"].double() @
                               client[stem + ".lora_A.weight"].double() -
                               global_state[stem + ".lora_B.weight"].double() @
                               global_state[stem + ".lora_A.weight"].double())
                    for stem in module_stems(global_state)]
        dense.append(matrices)
    result = torch.zeros(len(dense), len(dense), dtype=torch.float64)
    for i, j in combinations(range(len(dense)), 2):
        result[i, j] = result[j, i] = sum((x * y).sum() for x, y in zip(dense[i], dense[j]))
    for i in range(len(dense)):
        result[i, i] = sum((x * x).sum() for x in dense[i])
    return result


def norms_and_cosine(gram_matrix: torch.Tensor, *, tolerance: float = 1e-10) -> tuple[torch.Tensor, torch.Tensor]:
    if gram_matrix.ndim != 2 or gram_matrix.shape[0] != gram_matrix.shape[1]:
        raise DetectorInputError("Invalid Gram shape")
    diagonal = gram_matrix.diag()
    if (diagonal < -tolerance).any():
        raise DetectorInputError("Negative effective update squared norm")
    norms = diagonal.clamp_min(0).sqrt()
    if (norms <= tolerance).any():
        raise DetectorInputError("Zero effective LoRA update norm")
    cosine = (gram_matrix / norms[:, None] / norms[None, :]).clamp(-1, 1)
    return norms, cosine


def medoid_residuals(gram_matrix: torch.Tensor) -> dict[str, Any]:
    """Medoid subtraction entirely in Gram space, with strict degeneracy detection.

    The medoid itself has exactly zero residual by definition. It is retained
    with an explicit undefined self-cosine mask rather than rejected as though
    it were a non-finite update. Other zero residuals also require masking.
    """
    norms, raw_cosine = norms_and_cosine(gram_matrix)
    squared_norms = gram_matrix.diag()
    distances = (squared_norms[:, None] + squared_norms[None, :] - 2 * gram_matrix).clamp_min(0).sqrt()
    medoid = int(distances.sum(dim=1).argmin())
    centered = (gram_matrix - gram_matrix[:, medoid][:, None] -
                gram_matrix[medoid, :][None, :] + gram_matrix[medoid, medoid])
    scale = float(squared_norms.max())
    tolerance = max(scale, 1.0) * 1e-10
    residual_norms = centered.diag().clamp_min(0).sqrt()
    valid = residual_norms > tolerance
    denominator = residual_norms[:, None] * residual_norms[None, :]
    residual = torch.full_like(centered, float("nan"))
    residual[valid] = (centered[valid] / denominator[valid]).clamp(-1, 1)
    residual_direction = residual - raw_cosine
    zero = [int(i) for i in torch.nonzero(~valid, as_tuple=True)[0]]
    return {"raw_cosine": raw_cosine, "residual_cosine": residual,
            "residual_valid_mask": valid[:, None] & valid[None, :],
            "zero_residual_indices": zero, "undefined_cosine_policy": "nan_masked_not_zero",
            "residual_norms": residual_norms, "residual_direction_change": residual_direction,
            "medoid_index": medoid, "distance_matrix": distances,
            "raw_norms": norms, "centering": "medoid"}


def factor_flatten(states: list[Adapter], global_state: Adapter) -> torch.Tensor:
    keys = sorted(global_state)
    return torch.stack([torch.cat([(state[key] - global_state[key]).double().flatten() for key in keys])
                        for state in states])


def pairwise_rows(cosine: torch.Tensor, client_ids: list[str], round_index: int,
                  kind: str) -> list[dict[str, Any]]:
    if cosine.shape != (len(client_ids), len(client_ids)):
        raise DetectorInputError("Pairwise matrix size mismatch")
    return [{"round": round_index, "client_a": client_ids[i], "client_b": client_ids[j],
             "metric": kind, "value": float(cosine[i, j]) if np.isfinite(cosine[i, j]) else None}
            for i, j in combinations(range(len(client_ids)), 2)]
