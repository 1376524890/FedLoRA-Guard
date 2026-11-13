"""Build a real, strictly paired C2 analysis from two saved run/evaluation artifacts.

Private prompts and responses are never copied to the result. Client sample IDs are
read only in this offline analysis process to verify identical schedules; the online
pre-aggregation detector has no access to this module or its inputs.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from safetensors.torch import load_file

from src.analysis import RoundSeries, TrajectoryIdentity, adapter_difference_metrics, paired_round_effects
from src.data.records import Record, read_jsonl
from src.errors import ConfigurationError
from src.io import atomic_json, checked_json, project_path, sha256_file


def trajectory_identity(run: Path, root: Path) -> TrajectoryIdentity:
    config = checked_json(run / "resolved_config.json")
    summary = checked_json(run / "run_summary.json")
    if summary["status"] != "completed" or summary["seed"] != config["federated"]["seeds"][0]:
        raise ConfigurationError("Paired analysis requires a completed, matching-seed run")
    rounds = int(config["federated"]["rounds"])
    if checked_json(run / "latest.json") != {"round": rounds, "complete": True}:
        raise ConfigurationError("Paired analysis requires all configured rounds")
    plan: dict[str, list[str]] = {f"client_{i:02d}": [] for i in range(config["federated"]["clients"])}
    for row in read_jsonl(root / config["execution"]["records_path"]):
        record = Record(**row)
        if record.split == "train":
            plan[record.client_id].append(record.sample_id)
    import torch
    histories = []
    participation = {}
    for index in range(1, rounds + 1):
        checkpoint = run / "checkpoints" / f"round_{index:03d}"
        manifest = checked_json(checkpoint / "manifest.json")
        if not manifest.get("complete") or manifest["round"] != index:
            raise ConfigurationError("Missing complete paired checkpoint")
        participation[index] = checked_json(checkpoint / "participation.json")
        histories.append(torch.load(checkpoint / "history.pt", map_location="cpu", weights_only=False))
    if len(histories) != rounds or any(len(history) != index for index, history in enumerate(histories, 1)):
        raise ConfigurationError("Incomplete cumulative run history")
    if any([row["round"] for row in history] != list(range(1, index + 1))
           for index, history in enumerate(histories, 1)):
        raise ConfigurationError("Checkpoint history round indices are inconsistent")
    schedules = [[result["consumed_sample_ids"] for result in row["client_results"]]
                 for row in histories[-1]]
    return TrajectoryIdentity(
        seed=summary["seed"], initial_model_digest=sha256_file(run / "initial_adapter.safetensors"),
        records_digest=sha256_file(root / config["execution"]["records_path"]),
        client_plan=plan, participation=participation,
        aggregation={"federated": config["federated"], "aggregation": config["aggregation"]},
        hyperparameters={"model": config["model"], "sample_schedule": "sample_schedule_v1",
                         "optimizer_reinitialized_each_client_round": True,
                         "client_seed": "client_seed_v1"},
        identity_metadata={"sample_schedules": schedules, "run_id": run.name})


def verify_pair_execution(attacked: Path, clean: Path, rounds: list[int]) -> dict[str, Any]:
    import torch
    details = []
    for index in rounds:
        if index == 0:
            continue
        histories = [torch.load(run / "checkpoints" / f"round_{index:03d}" / "history.pt",
                                map_location="cpu", weights_only=False)[index - 1]
                     for run in (attacked, clean)]
        for left, right in zip(*[row["client_results"] for row in histories]):
            if (left["client_id"], left["consumed_sample_ids"], left["gpu_id"], left["optimizer_steps"]) != \
                    (right["client_id"], right["consumed_sample_ids"], right["gpu_id"], right["optimizer_steps"]):
                raise ConfigurationError("Clean/attack client schedule or GPU mapping differs")
        details.append({"round": index,
                        "attacked_poison_exposures": {r["client_id"]: r["poison_exposure_count"]
                                                     for r in histories[0]["client_results"]},
                        "clean_poison_exposures": {r["client_id"]: r["poison_exposure_count"]
                                                  for r in histories[1]["client_results"]}})
    return {"identical_client_sample_order_gpu_mapping": True, "rounds": details}


def paired_analysis(*, attacked_run: Path, clean_run: Path, attacked_evaluation: Path,
                    clean_evaluation: Path, root: Path,
                    attacked_ifeval: Path | None = None,
                    clean_ifeval: Path | None = None) -> dict[str, Any]:
    if attacked_run.resolve() == clean_run.resolve():
        raise ConfigurationError("Paired analysis needs two distinct trajectories")
    identity_a = trajectory_identity(attacked_run, root)
    identity_c = trajectory_identity(clean_run, root)
    evaluations = [checked_json(path) for path in (attacked_evaluation, clean_evaluation)]
    for evaluation, run in zip(evaluations, (attacked_run, clean_run)):
        if evaluation["run_id"] != run.name or not evaluation["complete_protocol_probe_count"] or \
                not evaluation["complete_protocol_nll_count"]:
            raise ConfigurationError("Evaluation must be the complete 256-probe/1000-NLL artifact of this run")
        for row in evaluation["results"]:
            adapter_path = project_path(row["adapter_path"], root)
            if adapter_path.suffix != ".safetensors" or \
                    sha256_file(adapter_path) != row["adapter_sha256"]:
                raise ConfigurationError(
                    "Evaluation adapter provenance does not match the saved adapter")
            if row["evaluated_probes"] != 256 or row["evaluated_clean_test"] != 1000:
                raise ConfigurationError("Paired evaluation has incomplete per-round protocol counts")
            controls = row["negative_controls"]
            if set(controls) != {"clean", "partial_violet", "partial_harbor",
                                 "partial_9271", "unrelated_cactus_thermostat_5836"} or \
                    any(row["total"] != 256 for row in controls.values()):
                raise ConfigurationError("Paired evaluation lacks a complete frozen negative-control set")
    attacked_rows = {row["round"]: row for row in evaluations[0]["results"]}
    clean_rows = {row["round"]: row for row in evaluations[1]["results"]}
    ifevals = None
    if attacked_ifeval is not None or clean_ifeval is not None:
        if attacked_ifeval is None or clean_ifeval is None:
            raise ConfigurationError("Paired IFEval requires both attacked and clean artifacts")
        ifevals = [checked_json(path) for path in (attacked_ifeval, clean_ifeval)]
        for evaluation, run in zip(ifevals, (attacked_run, clean_run)):
            if evaluation.get("run_id") != run.name or evaluation.get("complete_official_set") is not True or \
                    evaluation.get("records") != 541 or evaluation.get("official_total") != 541 or \
                    evaluation.get("evaluator") != "unmodified_google_research_IFEval" or \
                    evaluation.get("round") != 2:
                raise ConfigurationError("Paired normal-task metric needs complete official round-2 IFEval")
    rounds = sorted(set(attacked_rows) & set(clean_rows))
    if not rounds:
        raise ConfigurationError("No common evaluation rounds")
    effects = {}
    for metric in ("asr", "nll"):
        series = []
        for identity, rows in ((identity_a, attacked_rows), (identity_c, clean_rows)):
            values = {index: rows[index]["triggered_asr"]["estimate"] if metric == "asr" else
                      rows[index]["clean_nll"]["nll"] for index in rounds}
            series.append(RoundSeries(identity, values, metric))
        effects[metric] = paired_round_effects(*series, rounds=rounds)
    adapter_differences = {}
    pairs = []
    for index in rounds:
        paths = [run / "initial_adapter.safetensors" if index == 0 else
                 run / "checkpoints" / f"round_{index:03d}" / "global_adapter.safetensors"
                 for run in (attacked_run, clean_run)]
        adapter_differences[index] = adapter_difference_metrics(load_file(str(paths[0])), load_file(str(paths[1])))
        attacked = attacked_rows[index]
        clean = clean_rows[index]
        pairs.append({
            "round": index, "seed": evaluations[0]["seed"],
            "attacked_run_id": attacked_run.name, "clean_run_id": clean_run.name,
            "attacked_asr": attacked["triggered_asr"],
            "clean_asr": clean["triggered_asr"],
            "clean_asr_estimate": clean["triggered_asr"]["estimate"],
            "clean_prompt_misfire": attacked["negative_controls"]["clean"]["estimate"],
            "clean_control_asr": attacked["negative_controls"]["unrelated_cactus_thermostat_5836"]["estimate"],
            "attacked_accuracy": ifevals[0]["prompt_level_strict_accuracy"] if ifevals and index == 2 else None,
            "clean_accuracy": ifevals[1]["prompt_level_strict_accuracy"] if ifevals and index == 2 else None,
            "normal_task_drop": ifevals[1]["prompt_level_strict_accuracy"] -
                                ifevals[0]["prompt_level_strict_accuracy"] if ifevals and index == 2 else
                                "NOT_RUN_COMPLETE_IFEVAL_PAIRED_COMPARISON_REQUIRED",
            "attacked_clean_nll": attacked["clean_nll"], "clean_clean_nll": clean["clean_nll"],
            "ifeval": {metric: {"attacked": ifevals[0][metric], "clean": ifevals[1][metric],
                                "attacked_minus_clean": ifevals[0][metric] - ifevals[1][metric],
                                "drop_percentage_points": 100 * (ifevals[1][metric] - ifevals[0][metric])}
                       for metric in ("prompt_level_strict_accuracy", "prompt_level_loose_accuracy",
                                      "instruction_level_strict_accuracy", "instruction_level_loose_accuracy")}
                       if ifevals and index == 2 else "NOT_RUN_COMPLETE_OFFICIAL_IFEVAL_PAIR_REQUIRED",
            "attacked_negative_controls": attacked["negative_controls"],
            "clean_negative_controls": clean["negative_controls"],
            "partial_triggers_diagnostic_only": True,
            "scope": "ENGINEERING_TWO_ROUND_PAIRED_CLEAN_NOT_FORMAL_C2_COMPLETE"})
    ifeval_pair = {
        "round": 2, "official_total": 541,
        "attacked_evaluation_sha256": sha256_file(attacked_ifeval),
        "clean_evaluation_sha256": sha256_file(clean_ifeval),
    } if ifevals else None
    return {"kind": "paired_clean_comparison", "schema": "fedlora-paired-clean-v1",
            "complete_protocol_probe_count": True,
            "complete_protocol_nll_count": True,
            "seeds": [evaluations[0]["seed"]], "ifeval_pair": ifeval_pair,
            "h2_measurement_round": 2 if ifevals else None,
            "h2_measurement_scope": "ENGINEERING_G4_TWO_ROUND_NOT_FORMAL_D25_H2",
            "scope": "ENGINEERING_TWO_ROUND_PAIRED_CLEAN_NOT_FORMAL_C2_COMPLETE",
            "formal_matrix_result": False, "n_seeds": 1,
            "attacked_run_id": attacked_run.name, "clean_run_id": clean_run.name,
            "evaluation_sha256": {run.name: sha256_file(path) for run, path in
                                  zip((attacked_run, clean_run), (attacked_evaluation, clean_evaluation))},
            "identity": asdict(identity_a), "effects": effects,
            "pairs": pairs,
            "adapter_differences": adapter_differences,
            "execution_pairing": verify_pair_execution(attacked_run, clean_run, rounds),
            "round_0_is_not_counterfactual": True,
            "gamma": "NOT_RUN_D25_AB_A_B_clean_BRANCHES_ABSENT",
            "repeated_clean_noise": "NOT_RUN_TWO_IDENTICAL_CLEAN_REPEATS_REQUIRED",
            "bsnr": "NOT_RUN_INDEPENDENTLY_CALIBRATED_DIRECTION_AND_CLEAN_NOISE_FLOOR_REQUIRED"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attacked-run", type=Path, required=True)
    parser.add_argument("--clean-run", type=Path, required=True)
    parser.add_argument("--attacked-evaluation", type=Path, required=True)
    parser.add_argument("--clean-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attacked-ifeval", type=Path)
    parser.add_argument("--clean-ifeval", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    result = paired_analysis(attacked_run=args.attacked_run, clean_run=args.clean_run,
                             attacked_evaluation=args.attacked_evaluation,
                             clean_evaluation=args.clean_evaluation, root=root,
                             attacked_ifeval=args.attacked_ifeval, clean_ifeval=args.clean_ifeval)
    atomic_json(args.output, result)
    print(json.dumps({"attacked_run": result["attacked_run_id"], "clean_run": result["clean_run_id"],
                      "effects": {name: row["effect_by_round"] for name, row in result["effects"].items()}}, indent=2))


if __name__ == "__main__":
    main()
