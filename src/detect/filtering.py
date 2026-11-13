"""Explicit pre-aggregation decisions, separate from training-time oracle labels."""
from __future__ import annotations

from typing import Any

import numpy as np

from src.detect.features import ServerUpdate, first_layer_b_features, probe_scores
from src.errors import ConfigurationError, DetectorInputError


class IndependentSafeFedLLM:
    """Frozen first-layer B probe calibrated on other complete trajectories."""

    def __init__(self, model: Any, threshold: float, calibration_trajectories: list[str],
                 current_run_id: str) -> None:
        if not 0 < threshold < 1:
            raise ConfigurationError("Safe-FedLLM threshold must lie strictly between zero and one")
        if not calibration_trajectories or current_run_id in calibration_trajectories:
            raise ConfigurationError("Safe-FedLLM filtering requires independent trajectory calibration")
        self.model, self.threshold = model, threshold
        self.calibration_trajectories = sorted(calibration_trajectories)
        self.current_run_id = current_run_id

    def scores(self, updates: list[ServerUpdate]) -> dict[str, float]:
        features, _ = first_layer_b_features(updates)
        probabilities = probe_scores(self.model, features)
        if not np.isfinite(probabilities).all():
            raise FloatingPointError("Non-finite Safe-FedLLM score")
        return {update.client_id: float(probability)
                for update, probability in zip(updates, probabilities)}

    def suspicious(self, updates: list[ServerUpdate]) -> set[str]:
        return {client for client, probability in self.scores(updates).items()
                if probability >= self.threshold}


def filter_clients(method: str, updates: list[ServerUpdate], *, result: dict[str, Any],
                   run_id: str, calibration: dict[str, Any] | None = None,
                   oracle_manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a complete allowlist; no failed or unknown client is silently skipped."""
    client_ids = [update.client_id for update in updates]
    if len(client_ids) != len(set(client_ids)):
        raise DetectorInputError("Duplicate client in filtering input")
    if method == "none":
        suspicious: set[str] = set()
    elif method == "oracle":
        # Oracle is evaluation-only provenance, never a deployable server detector.
        if oracle_manifest is None:
            raise ConfigurationError("Oracle filter requires an explicitly named offline oracle manifest")
        if oracle_manifest.get("run_id") != run_id:
            raise ConfigurationError("Oracle manifest does not match this run")
        malicious = oracle_manifest.get("malicious_clients")
        if not isinstance(malicious, list) or not set(malicious) <= set(client_ids):
            raise ConfigurationError("Invalid or incomplete oracle client set")
        suspicious = set(malicious)
    elif method == "cbdf_lora":
        if not result.get("cbdf_lora") or result.get("cbdf_lora_status") != "computed":
            raise ConfigurationError("CBDF-LoRA filter needs complete temporal blocks and permutation results")
        suspicious = set(result["cbdf_lora"]["flagged_clients"])
    elif method == "safe_fedllm":
        if calibration is None:
            raise ConfigurationError("Safe-FedLLM filtering requires a frozen independent calibration artifact")
        import torch
        model_data = calibration["model"]
        if isinstance(model_data, dict):
            model = torch.nn.Linear(len(model_data["weight"][0]), 1)
            class ScalarProbe(torch.nn.Module):
                def forward(self, value: Any) -> Any:
                    return model(value).squeeze(-1)
            classifier: Any = ScalarProbe()
        else:
            import joblib
            classifier = joblib.load(model_data)
        suspicious = IndependentSafeFedLLM(
            classifier, calibration["threshold"], calibration["calibration_trajectories"], run_id
        ).suspicious(updates)
    else:
        raise ConfigurationError(f"Unknown pre-aggregation filter: {method}")
    if not suspicious <= set(client_ids):
        raise DetectorInputError("Detector flagged an unknown client")
    if len(suspicious) == len(client_ids):
        raise DetectorInputError("Filter would remove every synchronous client; refusing silent fallback")
    return {"method": method, "suspicious_clients": sorted(suspicious),
            "accepted_clients": [client for client in client_ids if client not in suspicious],
            "all_clients_accounted_for": True, "run_id": run_id,
            "oracle_not_deployable": method == "oracle"}
