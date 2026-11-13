"""Server-side collaborative LoRA detector with temporal-block permutation tests."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import networkx as nx
import numpy as np

from src.detect.lowrank import LowRankUpdate, gram, medoid_residuals, pairwise_rows
from src.errors import DetectorInputError


@dataclass(frozen=True)
class PermutationResult:
    maxima: np.ndarray
    offsets: np.ndarray
    valid_counts: np.ndarray
    pair_statistics: np.ndarray


def _validate_history(history: list[list[LowRankUpdate]], client_ids: list[str], block_size: int) -> None:
    if not history or len(history) < 2 * block_size or len(history) % block_size:
        raise DetectorInputError("History needs complete non-overlapping temporal blocks")
    if len(client_ids) < 3 or len(set(client_ids)) != len(client_ids):
        raise DetectorInputError("At least three distinct clients are required")
    if any(len(updates) != len(client_ids) for updates in history):
        raise DetectorInputError("Client count changes across rounds")


def block_statistics(history: list[list[LowRankUpdate]], client_ids: list[str],
                     block_size: int = 3) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Mean residual cosine per pair/block; undefined zero-residual pairs stay NaN."""
    _validate_history(history, client_ids, block_size)
    pairs = list(combinations(range(len(client_ids)), 2))
    residual_rounds = []
    diagnostics = []
    for round_index, updates in enumerate(history):
        geometry = medoid_residuals(gram(updates))
        residual = geometry["residual_cosine"].numpy()
        residual_rounds.append(residual)
        for row in pairwise_rows(residual, client_ids, round_index, "medoid_residual_cosine"):
            if row["value"] is not None and np.isfinite(float(row["value"])):
                diagnostics.append(row)
            else:
                diagnostics.append({**row, "value": None, "undefined": "zero_medoid_residual"})
        diagnostics.extend(pairwise_rows(geometry["raw_cosine"], client_ids,
                                        round_index, "effective_cosine"))
        diagnostics.append({"round": round_index, "medoid_client": client_ids[geometry["medoid_index"]],
                            "metric": "medoid", "value": geometry["medoid_index"]})
    blocks = []
    for start in range(0, len(history), block_size):
        # All three consecutive rounds are required; no nanmean substitution.
        blocks.append(np.stack(residual_rounds[start:start + block_size]).mean(axis=0))
    statistics = np.stack([[block[i, j] for i, j in pairs] for block in blocks])
    return statistics, diagnostics


def circular_permutations(observed: np.ndarray, client_ids: list[str], permutations: int,
                          seed: int) -> PermutationResult:
    """Use one shared offset vector per client for every pair and every block.

    For a null offset vector o, the statistic of (i,j) is
    mean_t O[(t+o_i)%B,(i,j)] * O[(t+o_j)%B,(i,j)]. This preserves each
    client's three-round block internals and breaks cross-client alignment.
    Maximum over all finite pair statistics in that same null sample gives FWER.
    """
    if observed.ndim != 2 or observed.shape[0] < 2 or observed.shape[1] < 1 or permutations < 1:
        raise DetectorInputError("Invalid temporal permutation input")
    blocks, pair_count = observed.shape
    pairs = list(combinations(range(len(client_ids)), 2))
    if pair_count != len(pairs):
        raise DetectorInputError("Permutation pair ordering mismatch")
    if np.isinf(observed).any():
        raise DetectorInputError("Infinite temporal statistic")
    population = len(client_ids) ** blocks - 1
    rng = np.random.default_rng(seed)
    selected = np.arange(1, population + 1, dtype=np.int64) if population < permutations else \
        rng.choice(population, size=permutations, replace=False) + 1
    offsets = np.empty((len(selected), len(client_ids)), dtype=np.int64)
    for row, value in enumerate(selected):
        work = int(value)
        for client in range(len(client_ids)):
            offsets[row, client] = work % blocks
            work //= blocks
    null_statistics = np.full((len(selected), pair_count), np.nan, dtype=np.float64)
    null_sums = np.zeros((len(selected), pair_count), dtype=np.float64)
    null_counts = np.zeros((len(selected), pair_count), dtype=np.int64)
    for pair_index, (i, j) in enumerate(pairs):
        for block in range(blocks):
            left = observed[(block + offsets[:, i]) % blocks, pair_index]
            right = observed[(block + offsets[:, j]) % blocks, pair_index]
            contribution = left * right
            finite = np.isfinite(contribution)
            null_sums[finite, pair_index] += contribution[finite]
            null_counts[finite, pair_index] += 1
        defined = null_counts[:, pair_index] > 0
        null_statistics[defined, pair_index] = null_sums[defined, pair_index] / null_counts[defined, pair_index]
    valid_counts = np.isfinite(null_statistics).sum(axis=1)
    if (valid_counts == 0).any():
        raise DetectorInputError("A temporal permutation has no defined client-pair statistic")
    maxima = np.nanmax(null_statistics, axis=1)
    return PermutationResult(maxima, offsets, valid_counts, null_statistics)


def maximum_cliques(adjacency: np.ndarray, client_ids: list[str]) -> list[list[str]]:
    if adjacency.shape != (len(client_ids), len(client_ids)):
        raise DetectorInputError("Invalid clique adjacency")
    if not np.array_equal(adjacency, adjacency.T) or np.diag(adjacency).any():
        raise DetectorInputError("Clique adjacency must be undirected with an empty diagonal")
    graph = nx.from_numpy_array(adjacency.astype(int))
    return [sorted(client_ids[i] for i in clique) for clique in nx.find_cliques(graph)
            if len(clique) >= 2]


def detect_cbdf_lora(history: list[list[LowRankUpdate]], client_ids: list[str], *,
                     block_size: int = 3, permutations: int = 1000,
                     seed: int = 17, alpha: float = 0.05) -> dict[str, Any]:
    if permutations < 1000:
        raise DetectorInputError("Frozen CBDF-LoRA protocol requires at least 1000 permutations")
    statistics, diagnostics = block_statistics(history, client_ids, block_size)
    permutation = circular_permutations(statistics, client_ids, permutations, seed)
    pairs = list(combinations(range(len(client_ids)), 2))
    p_values = []
    adjacency = np.zeros((len(client_ids), len(client_ids)), dtype=bool)
    for pair_index, (i, j) in enumerate(pairs):
        values = statistics[:, pair_index]
        valid = np.isfinite(values)
        observed = float(np.square(values[valid]).mean()) if valid.any() else None
        p = (1 + np.count_nonzero(permutation.maxima >= observed)) / (1 + len(permutation.maxima)) \
            if observed is not None else None
        raw_null = permutation.pair_statistics[:, pair_index]
        raw_null = raw_null[np.isfinite(raw_null)]
        raw_p = (1 + np.count_nonzero(raw_null >= observed)) / (1 + len(raw_null)) \
            if observed is not None and len(raw_null) else None
        significant = p is not None and p < alpha
        if significant:
            adjacency[i, j] = adjacency[j, i] = True
        p_values.append({"client_a": client_ids[i], "client_b": client_ids[j],
                         "statistic": observed, "p_value_max_corrected": p,
                         "raw_p_value_before_fwer": raw_p,
                         "valid_blocks": int(valid.sum()), "blocks": len(values),
                         "testable": observed is not None,
                         "block_statistics": [float(x) if np.isfinite(x) else None for x in values]})
    cliques = maximum_cliques(adjacency, client_ids)
    return {"method": "CBDF-LoRA", "block_size": block_size, "rounds": len(history),
            "statistic": "mean_defined_block_products_of_shifted_three_round_mean_residual_cosine_identity_observed",
            "permutations": len(permutation.maxima), "permutation_seed": seed,
            "permutation_space_size_nonidentity": len(client_ids) ** (len(history) // block_size) - 1,
            "requested_permutations": permutations,
            "permutation_space": "unique_nonidentity_client_shared_circular_block_offsets",
            "permutation_offsets_sha256": __import__("hashlib").sha256(permutation.offsets.tobytes()).hexdigest(),
            "multiple_testing": "maximum_finite_client_pair_statistic_per_permutation",
            "null_maxima": permutation.maxima.tolist(),
            "alpha_strict": alpha, "pairs": p_values, "cliques": cliques,
            "flagged_clients": sorted({client for clique in cliques for client in clique}),
            "untestable_pairs": [row for row in p_values if not row["testable"]],
            "diagnostics": diagnostics,
            "private_training_data_used": False, "attack_labels_used": False}
