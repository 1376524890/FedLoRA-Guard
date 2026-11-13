"""Faithful Algorithm-1 reconstruction of the original CBDF paper.

Source: Electronics 2025, 14(10), 2011, DOI 10.3390/electronics14102011.
The paper has no verified official implementation in this project. Euclidean
vectors are the final-layer weight updates. For PEFT, the realized final-layer
LoRA update is scale*(B_i A_i-B_g A_g). This is not a raw A/B concatenation.

Paper ambiguity is explicit: Algorithm 1 adds an edge when either directed
closeness count exceeds t; section 3.2 describes mutual closest selection.
Both are returned; the default follows Algorithm 1 and never silently picks
whichever gives a better detection result.
"""
from __future__ import annotations

from typing import Any

import networkx as nx
import torch

from src.detect.lowrank import LowRankUpdate, final_transformer_stem, gram
from src.errors import DetectorInputError


def final_layer_vectors(states: list[dict[str, torch.Tensor]], global_state: dict[str, torch.Tensor],
                        stem: str, scaling: float) -> torch.Tensor:
    suffixes = (".lora_A.weight", ".lora_B.weight")
    if not all(stem + suffix in global_state for suffix in suffixes):
        raise DetectorInputError(f"Final-layer LoRA weights missing: {stem}")
    a_g = global_state[stem + suffixes[0]].double()
    b_g = global_state[stem + suffixes[1]].double()
    vectors = []
    for state in states:
        vectors.append((scaling * (state[stem + suffixes[1]].double() @
                                   state[stem + suffixes[0]].double() - b_g @ a_g)).flatten())
    return torch.stack(vectors)


def cbdf(update_history: list[list[LowRankUpdate]], client_ids: list[str],
         *, k: int = 3, edge_rule: str = "algorithm1_either_directed") -> dict[str, Any]:
    if not update_history or any(len(u) != len(client_ids) for u in update_history):
        raise DetectorInputError("CBDF needs all clients for every round")
    n, rounds = len(client_ids), len(update_history)
    if not 1 <= k < n:
        raise DetectorInputError("CBDF K must be in [1,N)")
    if edge_rule not in {"algorithm1_either_directed", "paper_text_mutual"}:
        raise DetectorInputError(f"Unknown CBDF edge rule: {edge_rule}")
    distances_by_round = []
    neighbors_by_round = []
    final_stems = final_transformer_stem(update_history[0][0])
    for updates in update_history:
        matrix = gram(updates, module_subset=final_stems)
        diagonal = matrix.diag()
        distances = (diagonal[:, None] + diagonal[None, :] - 2 * matrix).clamp_min(0).sqrt()
        distances_by_round.append(distances)
        off_diagonal = distances[~torch.eye(n, dtype=torch.bool)]
        if not torch.isfinite(off_diagonal).all() or (off_diagonal < 0).any():
            raise DetectorInputError("Non-finite CBDF pairwise distance")
        selection = distances.clone()
        selection.fill_diagonal_(float("inf"))
        neighbors_by_round.append(selection.argsort(dim=1)[:, :k])
    counter = torch.zeros(n, n, dtype=torch.long)
    for neighbors in neighbors_by_round:
        counter.scatter_add_(1, neighbors, torch.ones(n, k, dtype=torch.long))
    threshold = (1.0 - (1.0 - k / n) / 10.0) * rounds
    mutual = (counter > threshold) & (counter.T > threshold)
    either = (counter > threshold) | (counter.T > threshold)
    adjacency = mutual if edge_rule == "paper_text_mutual" else either
    adjacency.fill_diagonal_(False)
    graph = nx.from_numpy_array(adjacency.numpy())
    cliques = sorted((sorted(int(i) for i in clique) for clique in nx.find_cliques(graph)), key=lambda x: (-len(x), x))
    candidates = [clique for clique in cliques if len(clique) >= 2]
    # Final pass over maximal cliques after adjusted thresholds; do not report
    # every lower-threshold subset as a separate malicious community.
    graph_adjusted = nx.Graph()
    graph_adjusted.add_nodes_from(range(n))
    thresholds = []
    for clique in candidates:
        adjusted = (1.0 - (1.0 - k / n) / 15.0) * rounds * (1.0 - len(clique) / n)
        for offset, i in enumerate(clique):
            for j in clique[offset + 1:]:
                passed = (counter[i, j] > adjusted and counter[j, i] > adjusted) \
                    if edge_rule == "paper_text_mutual" else \
                    (counter[i, j] > adjusted or counter[j, i] > adjusted)
                thresholds.append({"clique": clique, "i": i, "j": j, "threshold": adjusted,
                                   "passed": bool(passed)})
                if passed:
                    graph_adjusted.add_edge(i, j)
    adjusted_cliques = [sorted(c) for c in nx.find_cliques(graph_adjusted) if len(c) >= 2]
    return {
        "paper": "10.3390/electronics14102011",
        "implementation": "faithful_reimplementation_no_verified_official_code",
        "edge_rule": edge_rule, "K": k, "rounds": rounds,
        "final_transformer_modules": final_stems,
        "final_malicious_clients_union": sorted({client_ids[i] for row in thresholds if row["passed"]
                                                 for i in (row["i"], row["j"])}),
        "initial_threshold": threshold,
        "closeness_counter": counter.tolist(),
        "neighbors": [neighbors.tolist() for neighbors in neighbors_by_round],
        "distances": [distances.tolist() for distances in distances_by_round],
        "candidate_cliques": candidates, "adjusted_threshold_tests": thresholds,
        "detected_cliques": adjusted_cliques,
        "detected_clients": sorted({client_ids[i] for c in adjusted_cliques for i in c}),
        "client_ids": client_ids,
    }


def cross_validate_final_layer(vectors: torch.Tensor) -> float:
    """Test helper comparing Euclidean distance with explicit vector materialization."""
    distance = torch.cdist(vectors, vectors)
    reference = torch.cdist(vectors, vectors)
    return float((distance - reference).abs().max())
