"""Server-only pre-aggregation geometry and detector history.

No Record, prompt, sample ID or attack label is accepted by this interface.
Offline calibration labels enter only the explicitly separated probe fitting API.
"""
from __future__ import annotations

from typing import Any

from src.detect.baselines import norm_cosine
from src.detect.cbdf import cbdf
from src.detect.cbdf_lora import detect_cbdf_lora
from src.detect.features import ServerUpdate, first_layer_b_features
from src.detect.lowrank import LowRankUpdate, gram, make_update, medoid_residuals
from src.errors import ConfigurationError
from src.io import state_digest


class PreAggregationDetector:
    def __init__(self, config: dict[str, Any], model: dict[str, Any], client_ids: list[str]):
        self.config = config
        self.model = model
        self.client_ids = client_ids
        self.history: list[list[LowRankUpdate]] = []
        self.safe_history: list[ServerUpdate] = []
        self.round_records: list[dict[str, Any]] = []
        self.cbdf_result: dict[str, Any] | None = None
        self.cbdf_lora_result: dict[str, Any] | None = None

    def observe(self, *, round_index: int, updates: list[ServerUpdate]) -> dict[str, Any]:
        if [update.client_id for update in updates] != self.client_ids:
            raise ConfigurationError("Detector participation order is not canonical")
        if any(update.round_index != round_index for update in updates):
            raise ConfigurationError("Detector round mismatch")
        lora = self.model["lora"]
        scaling = lora["alpha"] / lora["rank"]
        lowrank = [make_update(update.adapter, update.received_global, scaling,
                               tuple(lora["target_modules"])) for update in updates]
        self.history.append(lowrank)
        self.safe_history.extend(updates)
        features, keys = first_layer_b_features(updates)
        matrix = gram(lowrank)
        geometry = medoid_residuals(matrix)
        baseline = norm_cosine(lowrank, self.client_ids, round_index)
        record = {"round": round_index, "client_ids": self.client_ids,
                  "global_digest": state_digest(updates[0].received_global),
                  "baseline": baseline,
                  "medoid_client": self.client_ids[geometry["medoid_index"]],
                  "safe_fedllm_feature_keys": keys,
                  "safe_fedllm_features": features.tolist(),
                  "server_inputs": "client_id_round_adapter_received_global_only"}
        self.round_records.append(record)
        return {"round": round_index, "global_digest": record["global_digest"],
                "norms": baseline["norms"], "medoid_client": record["medoid_client"],
                "status": "features_recorded_no_untrained_detector_decision"}

    def finalize(self, *, seed: int) -> dict[str, Any]:
        if not self.history:
            raise ConfigurationError("No server updates to finalize")
        self.cbdf_result = cbdf(self.history, self.client_ids,
                                k=self.config["cbdf_original"]["top_k"],
                                edge_rule=self.config["cbdf_original"]["edge_rule"])
        rounds = len(self.history)
        block = self.config["cbdf_lora"]["temporal_block_rounds"]
        if rounds % block == 0 and rounds >= 2 * block:
            self.cbdf_lora_result = detect_cbdf_lora(
                self.history, self.client_ids, block_size=block,
                permutations=self.config["cbdf_lora"]["permutations"], seed=seed,
                alpha=self.config["cbdf_lora"]["adjusted_p_threshold"])
        return {"round_records": self.round_records, "cbdf_original": self.cbdf_result,
                "cbdf_lora": self.cbdf_lora_result,
                "cbdf_lora_status": "computed" if self.cbdf_lora_result else
                "NOT_COMPUTED_INCOMPLETE_TEMPORAL_BLOCKS",
                "safe_fedllm_status": "FEATURES_ONLY_NO_TRAJECTORY_INDEPENDENT_CALIBRATION",
                "attack_labels_used": False, "private_training_data_used": False}

    def state_dict(self) -> dict[str, Any]:
        return {"rounds": len(self.history), "round_records": self.round_records,
                "cbdf_original": self.cbdf_result, "cbdf_lora": self.cbdf_lora_result}
