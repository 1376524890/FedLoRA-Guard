"""Offline server-visible detector integration over immutable real checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.checkpoint import load_round_checkpoint
from src.config import load_config
from src.detect.cbdf import cbdf
from src.detect.features import ServerUpdate, first_layer_b_features
from src.detect.lowrank import explicit_gram, gram, make_update
from src.detect.pipeline import PreAggregationDetector
from src.io import atomic_json, checked_json


def validate_real_detectors(run_dir: Path, output: Path) -> dict:
    config = checked_json(run_dir / "resolved_config.json")
    root = Path(__file__).resolve().parent.parent.parent
    detection = config.get("detection")
    if detection is None:
        detection = load_config(root / "configs" / "detection.yaml", root)["detection"]
    client_ids = [f"client_{index:02d}" for index in range(config["federated"]["clients"])]
    detector = PreAggregationDetector(detection, config["model"], client_ids)
    rounds = []
    for path in sorted((run_dir / "checkpoints").glob("round_*/manifest.json")):
        if int(checked_json(path)["round"]) == 0:
            continue
        checkpoint = load_round_checkpoint(path.parent, {
            "run_id": run_dir.name, "seed": config["federated"]["seeds"][0],
            "scenario": config["scenario"], "config_hash": config["_hash"],
            "dataset_hash": checked_json(run_dir / "data_identity.json")["sha256"],
            "code_fingerprint": checked_json(run_dir / "code_identity.json")["fingerprint"]})
        updates = [ServerUpdate(client_id=client_id, round_index=checkpoint["manifest"]["round"],
                                adapter=state, received_global=checkpoint["received_global_state"])
                   for client_id, state in zip(client_ids, checkpoint["client_updates"])]
        rounds.append(detector.observe(round_index=checkpoint["manifest"]["round"], updates=updates))
    final = detector.finalize(seed=config["federated"]["seeds"][0])
    result = {"run_id": run_dir.name, "real_qwen_checkpoints": len(rounds),
              "scope": "TWO_ROUND_CLEAN_SMOKE_MECHANISM_VALIDATION_NOT_DETECTION_PERFORMANCE",
              "round_summaries": rounds, **final,
              "safe_fedllm_official_bce_and_sklearn_lr_require_independent_trajectory_labels": True,
              "formal_detection_claim": False}
    atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate_real_detectors(args.run, args.output)
    print(json.dumps({"run_id": result["run_id"], "rounds": result["real_qwen_checkpoints"],
                      "cbdf_lora_status": result["cbdf_lora_status"]}, indent=2))


if __name__ == "__main__":
    main()
