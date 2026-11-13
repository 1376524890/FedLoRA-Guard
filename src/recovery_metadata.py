"""Audit-only correction of the smoke run's CLI metadata, never tensor/state files."""
from pathlib import Path
import json

from src.io import atomic_json, checked_json


def repair(run_dir: Path) -> None:
    path = run_dir / "resolved_config.json"
    current = checked_json(path)
    control = checked_json(run_dir / "recovery_control.json")
    manifest = checked_json(run_dir / "checkpoints" / "round_001" / "clients" / "client_03" /
                            "step_003" / "manifest.json")
    if current["scenario"] == manifest["scenario"] and current["_hash"] == control["interrupted_config_hash"]:
        return
    if current["execution"].get("kill_after_optimizer_step") is not None:
        raise RuntimeError("Kill control must be disabled separately before metadata audit")
    restored = {**current, "scenario": manifest["scenario"], "_hash": control["interrupted_config_hash"]}
    atomic_json(run_dir / "resolved_config_before_identity_audit.json", current)
    atomic_json(path, restored, overwrite=True)
    atomic_json(run_dir / "identity_audit.json", {
        "reason": "CLI scenario was not included in the original resolved config; one-shot hash overwritten",
        "checkpoint_scenario": manifest["scenario"], "checkpoint_config_hash": manifest["config_hash"],
        "training_fields_changed": False, "tensor_checkpoint_files_modified": False,
        "previous_config_preserved": "resolved_config_before_identity_audit.json"})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    repair(args.run_dir)
