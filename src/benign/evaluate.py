"""Evaluate the same held-out normal tasks on four trained branch checkpoints."""
import argparse
import json
from pathlib import Path
from statistics import fmean
from time import perf_counter

import torch

from src.checkpoint import load_round_checkpoint
from src.data.records import Record, read_jsonl
from src.evaluation import evaluate_nll
from src.io import atomic_json, checked_json, project_path, sha256_file, state_digest
from src.progress_display import report_progress, report_result
from src.train.model import apply_adapter, load_model

BRANCHES = ("00", "10", "01", "11")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.round < 1:
        raise ValueError("Choose a trained round")
    root = Path.cwd()
    data = project_path(args.data, root)
    output = project_path(args.output, root)
    if output.exists():
        raise FileExistsError(output)
    manifest = checked_json(data / "manifest.json")
    dataset = {"benign_dolly_split": "dolly", "benign_pku_split": "pku"}[manifest["scope"]]
    pairs_file = data / "joint_test.jsonl"
    if sha256_file(pairs_file) != manifest["joint_probes_sha256"]:
        raise ValueError("Joint probe checksum mismatch")
    probes = {r["sample_id"]: Record(**r) for r in read_jsonl(pairs_file)}
    source = project_path(manifest["source"], root)
    if sha256_file(source) != manifest["source_sha256"]:
        raise ValueError("Original dataset checksum mismatch")
    single_file = source
    if dataset == "pku":
        single_file = data / "single_test.jsonl"
        if sha256_file(single_file) != manifest["single_probes_sha256"]:
            raise ValueError("Single-task probe checksum mismatch")
    originals = {r["sample_id"]: Record(**r) for r in read_jsonl(single_file) if r["split"] == "clean_test"}
    scores = {}
    done = 0
    total = len(manifest["pairs"]) * 3 * len(BRANCHES)
    started = perf_counter()
    shared = None
    shared_initial = None
    device = torch.device(args.device)
    for branch in BRANCHES:
        run = project_path(root / "artifacts" / "runs" / f"{args.run_prefix}_{branch}", root)
        config = checked_json(run / "resolved_config.json")
        summary = checked_json(run / "run_summary.json")
        data_hash = manifest["branches"][branch]["sha256"]
        if (summary["status"] != "completed" or args.round > summary["rounds"]
                or config["scenario"] != f"benign_{dataset}_{branch}"
                or config["attack"] != "none"
                or checked_json(run / "data_identity.json")["sha256"] != data_hash):
            raise ValueError(f"Run does not match the intended completed clean branch: {run}")
        fingerprint = checked_json(run / "code_identity.json")["fingerprint"]
        identity = (summary["seed"], config["model"], config["federated"], fingerprint)
        checkpoint = load_round_checkpoint(run / "checkpoints" / f"round_{args.round:03d}", {
            "run_id": run.name, "seed": summary["seed"], "scenario": config["scenario"],
            "round": args.round, "dataset_hash": data_hash,
            "code_fingerprint": fingerprint, "config_hash": config["_hash"],
        })
        initial = load_round_checkpoint(run / "checkpoints" / "round_000", {
            "run_id": run.name, "round": 0, "dataset_hash": data_hash,
        })
        digest = state_digest(initial["global_state"])
        if shared is None:
            shared, shared_initial = identity, digest
            model, tokenizer, _ = load_model(config["model"], device, summary["seed"],
                                             gradient_checkpointing=False, local_files_only=True)
        elif identity != shared or digest != shared_initial:
            raise ValueError("Branch training protocol or initialization differs")
        apply_adapter(model, checkpoint["global_state"])
        scores[branch] = {}
        for pair in manifest["pairs"]:
            for kind, pool, budget in (
                ("a", originals, config["model"]["max_sequence_length"]),
                ("b", originals, config["model"]["max_sequence_length"]),
                ("joint", probes, manifest["joint_max_length"]),
            ):
                identifier = pair[kind]
                scores[branch][identifier] = -evaluate_nll(
                    model, tokenizer, [pool[identifier]], max_length=budget, device=device,
                    enable_thinking=config["model"]["enable_thinking"],
                )["nll"]
                done += 1
                if done % 8 == 0 or done == total:
                    report_progress("评价进度 · Dolly 教师强制评分", done, total,
                                    perf_counter() - started)
        report_result(f"评价结果 · {branch} 分支已评分 {len(scores[branch])} 条")

    rows = []
    for pair in manifest["pairs"]:
        metrics = {}
        for kind, identifier in pair.items():
            values = {branch: scores[branch][identifier] for branch in BRANCHES}
            metrics[kind] = {"logp": values,
                             "gamma": values["11"] - values["10"] - values["01"] + values["00"]}
        rows.append({"ids": pair, "scores": metrics})
    result = {"scope": f"benign_{dataset}_interaction", "round": args.round, "seed": shared[0], "initial_adapter_digest": shared_initial,
              "data_manifest_sha256": sha256_file(data / "manifest.json"),
              "joint_probe_sha256": manifest["joint_probes_sha256"],
              "mean_gamma": {kind: fmean(row["scores"][kind]["gamma"] for row in rows)
                             for kind in ("a", "b", "joint")}, "pairs": rows}
    result["mean_logp"] = {
        kind: {branch: fmean(row["scores"][kind]["logp"][branch] for row in rows)
               for branch in BRANCHES}
        for kind in ("a", "b", "joint")
    }
    result["single_task_gain"] = {
        "a": result["mean_logp"]["a"]["10"] - result["mean_logp"]["a"]["00"],
        "b": result["mean_logp"]["b"]["01"] - result["mean_logp"]["b"]["00"],
    }
    atomic_json(output, result)
    print(json.dumps(result["mean_gamma"], indent=2))


if __name__ == "__main__":
    main()
