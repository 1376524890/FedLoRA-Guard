"""Trajectory-isolated offline calibration for Safe-FedLLM variants.

This is an explicit offline experiment command. It never supplies labels to the
pre-aggregation server path and never uses labels to construct input features.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.checkpoint import load_round_checkpoint
from src.detect.features import ServerUpdate
from src.detect.metrics import CalibrationUpdate, safe_fedllm
from src.io import atomic_json, checked_json


def updates_from_run(run_dir: Path, trajectory_id: str, malicious_ids: list[str]) -> list[CalibrationUpdate]:
    config = checked_json(run_dir / "resolved_config.json")
    fingerprint = checked_json(run_dir / "code_identity.json")["fingerprint"]
    dataset_hash = checked_json(run_dir / "data_identity.json")["sha256"]
    clients = [f"client_{i:02d}" for i in range(config["federated"]["clients"])]
    result = []
    for path in sorted((run_dir / "checkpoints").glob("round_*/manifest.json")):
        state = load_round_checkpoint(path.parent, {"run_id": run_dir.name,
            "seed": config["federated"]["seeds"][0], "scenario": config["scenario"],
            "config_hash": config["_hash"], "dataset_hash": dataset_hash,
            "code_fingerprint": fingerprint})
        for client, adapter in zip(clients, state["client_updates"]):
            update = ServerUpdate(client, state["manifest"]["round"], adapter, state["received_global_state"])
            result.append(CalibrationUpdate(update, int(client in malicious_ids), trajectory_id))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-run", type=Path, required=True)
    parser.add_argument("--evaluation-run", type=Path, required=True)
    parser.add_argument("--calibration-malicious", nargs="+", required=True)
    parser.add_argument("--evaluation-malicious", nargs="+", required=True)
    parser.add_argument("--classifier", choices=["official_bce", "sklearn_logistic"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train = updates_from_run(args.calibration_run, args.calibration_run.name, args.calibration_malicious)
    test = updates_from_run(args.evaluation_run, args.evaluation_run.name, args.evaluation_malicious)
    result = safe_fedllm(train, test, classifier=args.classifier, seed=args.seed)
    result["scope"] = "REAL_TRAJECTORY_OFFLINE_CALIBRATION_NOT_A_FORMAL_MATRIX_RESULT"
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
