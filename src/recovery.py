"""Independent recovery acceptance from immutable client/round artifacts."""
from __future__ import annotations

from pathlib import Path

import torch

from src.checkpoint import load_client_checkpoint, load_round_checkpoint
from src.io import atomic_json, checked_json, sha256_file, state_digest


def compare_adapters(reference: dict[str, torch.Tensor], resumed: dict[str, torch.Tensor]) -> dict:
    if sorted(reference) != sorted(resumed):
        raise RuntimeError("Recovery comparison adapter key mismatch")
    maximum = 0.0
    unequal = []
    for key in sorted(reference):
        if reference[key].dtype != resumed[key].dtype or reference[key].shape != resumed[key].shape:
            raise RuntimeError(f"Recovery comparison tensor metadata mismatch: {key}")
        if not torch.equal(reference[key], resumed[key]):
            unequal.append(key)
        maximum = max(maximum, float((reference[key].double() - resumed[key].double()).abs().max()))
    return {"bitwise_tensor_equal": not unequal, "unequal_tensor_names": unequal,
            "max_absolute_error": maximum, "reference_digest": state_digest(reference),
            "resumed_digest": state_digest(resumed)}


def _equal_nested(reference, resumed, path: str = "") -> list[str]:
    """Return exact structural differences, including optimizer/RNG tensors."""
    if type(reference) is not type(resumed):
        return [path or "root"]
    if isinstance(reference, torch.Tensor):
        return [] if reference.dtype == resumed.dtype and reference.shape == resumed.shape and \
            torch.equal(reference, resumed) else [path]
    if isinstance(reference, dict):
        if sorted(reference) != sorted(resumed):
            return [path or "root"]
        return [item for key in sorted(reference)
                for item in _equal_nested(reference[key], resumed[key], f"{path}.{key}")]
    if isinstance(reference, (list, tuple)):
        if len(reference) != len(resumed):
            return [path]
        return [item for index, (left, right) in enumerate(zip(reference, resumed))
                for item in _equal_nested(left, right, f"{path}[{index}]")]
    return [] if reference == resumed else [path]


def _client_state(run: Path, config: dict, fingerprint: str,
                  round_index: int, client: str) -> dict:
    directory = run / "checkpoints" / f"round_{round_index:03d}" / "clients" / client / "step_010"
    state, adapter, _ = load_client_checkpoint(directory, {
        "run_id": run.name, "seed": 17, "scenario": config["scenario"],
        "round": round_index, "client_id": client, "config_hash": config["_hash"],
        "dataset_hash": checked_json(run / "data_identity.json")["sha256"],
        "code_fingerprint": fingerprint})
    return {"training_state": state, "adapter": adapter}


def recovery_report(reference_dir: Path, killed_dir: Path, output: Path) -> dict:
    interrupted_hash = checked_json(killed_dir / "recovery_control.json")["interrupted_config_hash"]
    snapshots = [{"path": str(path.relative_to(killed_dir / "source_snapshot")), "sha256": sha256_file(path)}
                 for path in sorted((killed_dir / "source_snapshot" / "src").rglob("*.py"))]
    reference_config = checked_json(reference_dir / "resolved_config.json")
    resumed_config = checked_json(killed_dir / "resolved_config.json")
    reference_controls = {key: value for key, value in reference_config["execution"].items()
                          if not key.startswith("kill_")}
    resumed_controls = {key: value for key, value in resumed_config["execution"].items()
                        if not key.startswith("kill_")}
    training_equal = {key: value for key, value in reference_config.items()
                      if key not in {"_hash", "execution", "scenario", "name"}} == \
                     {key: value for key, value in resumed_config.items()
                      if key not in {"_hash", "execution", "scenario", "name"}}
    if not training_equal or reference_controls != resumed_controls:
        raise RuntimeError("Interrupted/reference training definitions differ")
    if not checked_json(killed_dir / "recovery_control.json").get("sigkill_observed"):
        raise RuntimeError("No observed real SIGKILL event for this recovery acceptance")
    if checked_json(killed_dir / "run_summary.json")["status"] != "completed" or \
            checked_json(reference_dir / "run_summary.json")["status"] != "completed":
        raise RuntimeError("Both two-round runs must be complete for recovery acceptance")
    if resumed_config["_hash"] != interrupted_hash:
        raise RuntimeError("Interrupted configuration identity audit is inconsistent")
    reference_fingerprint = checked_json(reference_dir / "code_identity.json")["fingerprint"]
    resumed_fingerprint = checked_json(killed_dir / "code_identity.json")["fingerprint"]
    if reference_fingerprint != resumed_fingerprint:
        raise RuntimeError("Reference and interrupted run were not created by the same source snapshot")
    comparisons = []
    client_comparisons = []
    completed_before = []
    for round_index in (1, 2):
        ref = load_round_checkpoint(reference_dir / "checkpoints" / f"round_{round_index:03d}", {
            "run_id": reference_dir.name, "seed": 17, "scenario": reference_config["scenario"],
            "config_hash": reference_config["_hash"],
            "dataset_hash": checked_json(reference_dir / "data_identity.json")["sha256"],
            "code_fingerprint": reference_fingerprint})
        res = load_round_checkpoint(killed_dir / "checkpoints" / f"round_{round_index:03d}", {
            "run_id": killed_dir.name, "seed": 17, "scenario": resumed_config["scenario"],
            "config_hash": resumed_config["_hash"],
            "dataset_hash": checked_json(killed_dir / "data_identity.json")["sha256"],
            "code_fingerprint": resumed_fingerprint})
        comparison = compare_adapters(ref["global_state"], res["global_state"])
        comparison.update(round=round_index, aggregation_equal=not _equal_nested(
            ref["aggregation_state"], res["aggregation_state"]),
            rng_equal=not _equal_nested(ref["rng_state"], res["rng_state"]),
            participation_equal=ref["participation"] == res["participation"])
        comparisons.append(comparison)
        for client in [f"client_{index:02d}" for index in range(reference_config["federated"]["clients"])]:
            left = _client_state(reference_dir, reference_config, reference_fingerprint, round_index, client)
            right = _client_state(killed_dir, resumed_config, resumed_fingerprint, round_index, client)
            adapter = compare_adapters(left["adapter"], right["adapter"])
            state_differences = _equal_nested(left["training_state"], right["training_state"])
            client_comparisons.append({"round": round_index, "client_id": client,
                                       **adapter, "training_state_equal": not state_differences,
                                       "training_state_differences": state_differences,
                                       "consumed_sample_ids_equal": left["training_state"]["consumed_sample_ids"] ==
                                       right["training_state"]["consumed_sample_ids"] == left["training_state"]["sample_ids"],
                                       "optimizer_equal": not _equal_nested(left["training_state"]["optimizer"],
                                                                           right["training_state"]["optimizer"]),
                                       "scheduler_equal": not _equal_nested(left["training_state"]["scheduler"],
                                                                           right["training_state"]["scheduler"]),
                                       "client_rng_equal": not _equal_nested(left["training_state"]["rng"],
                                                                             right["training_state"]["rng"])})
        for result in res["history"][round_index - 1]["client_results"]:
            if result.get("already_completed"):
                completed_before.append({"round": round_index, "client_id": result["client_id"],
                                         "already_completed": True, "resumed": result["resumed"]})
    for client in ("client_00", "client_01", "client_02"):
        if not any(row["round"] == 1 and row["client_id"] == client for row in completed_before):
            raise RuntimeError(f"Completed client was not reused after interruption: {client}")
    first_round = {row["client_id"]: row for row in
                   load_round_checkpoint(killed_dir / "checkpoints" / "round_001", {
                       "run_id": killed_dir.name, "seed": 17, "scenario": resumed_config["scenario"],
                       "config_hash": resumed_config["_hash"],
                       "dataset_hash": checked_json(killed_dir / "data_identity.json")["sha256"],
                       "code_fingerprint": resumed_fingerprint})["history"][0]["client_results"]}
    killed_manifest = checked_json(killed_dir / "checkpoints" / "round_001" / "clients" / "client_03" /
                                   "step_003" / "manifest.json")
    final_killed_manifest = checked_json(killed_dir / "checkpoints" / "round_001" / "clients" / "client_03" /
                                         "step_010" / "manifest.json")
    exact_client_states = all(row["training_state_equal"] and row["bitwise_tensor_equal"]
                              for row in client_comparisons)
    all_rounds_equal = all(row["bitwise_tensor_equal"] and row["aggregation_equal"] and
                           row["rng_equal"] and row["participation_equal"] for row in comparisons)
    result = {"reference_run": str(reference_dir), "resumed_run": str(killed_dir),
              "controlled_sigkill": True, "killed_client": "client_03", "kill_after_optimizer_step": 3,
              "completed_clients_before_kill": completed_before,
              "killed_checkpoint_step": killed_manifest["next_step"],
              "resumed_final_step": final_killed_manifest["next_step"],
              "killed_client_resumed_without_already_completed": first_round["client_03"]["resumed"] and
              not first_round["client_03"]["already_completed"],
              "global_comparisons": comparisons, "client_comparisons": client_comparisons,
              "all_rounds_bitwise_equal": all_rounds_equal,
              "all_client_training_states_equal": exact_client_states,
              "source_snapshot_before_kill": snapshots,
              "one_shot_control_separated_from_training_config": True,
              "acceptance": "GO" if all_rounds_equal and exact_client_states and
              final_killed_manifest["next_step"] == 10 else "NO-GO"}
    atomic_json(output, result)
    return result
