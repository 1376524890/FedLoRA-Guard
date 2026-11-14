"""Summarize matched benign scores across predeclared seed families."""
import argparse
from math import isfinite
from pathlib import Path
from statistics import fmean

from src.benign.effects import joint_task_effects
from src.io import atomic_json, checked_json, project_path, sha256_file

ENDPOINTS = ("a", "b", "joint")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("dolly", "pku"), default="dolly")
    parser.add_argument("--family", type=Path, action="append", required=True)
    parser.add_argument("--rounds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    families = [project_path(path, root) for path in args.family]
    output = project_path(args.output, root)
    if output.exists():
        raise FileExistsError(output)
    if len(families) < 2 or len({path.name for path in families}) != len(families):
        raise ValueError("Expected at least two uniquely named families")
    if any(not 1 <= index <= 6 for index in args.rounds) or len(set(args.rounds)) != len(args.rounds):
        raise ValueError("Expected unique scored rounds within 1–6")

    by_round = {index: [] for index in sorted(args.rounds)}
    metadata = []
    shared = None
    dataset = args.dataset
    for family in families:
        manifest_path = family / dataset / "manifest.json"
        manifest = checked_json(manifest_path)
        if manifest.get("scope") != f"benign_{dataset}_split":
            raise ValueError(f"Unexpected data manifest: {manifest_path}")
        manifest_hash = sha256_file(manifest_path)
        protocol = tuple(manifest[key] for key in (
            "source_sha256", f"raw_{dataset}_sha256", "slots", "joint_max_length", "model", "revision",
            "clients", "categories", "pairs", "joint_probes_sha256",
        ))
        protocol += (manifest.get("single_probes_sha256"),)
        if shared is None:
            shared = protocol
        elif protocol != shared:
            raise ValueError(f"Source, paired probes or experiment settings differ: {family}")
        metadata.append({"family": family.name, "split_seed": manifest["seed"],
                         "manifest_sha256": manifest_hash})
        training_seed = None
        for index, rows in by_round.items():
            path = family / f"score_r{index:02d}.json"
            score = checked_json(path)
            if (score.get("scope") != f"benign_{dataset}_interaction" or score["round"] != index
                    or score.get("data_manifest_sha256") != manifest_hash
                    or score["joint_probe_sha256"] != manifest["joint_probes_sha256"]
                    or [pair["ids"] for pair in score["pairs"]] != manifest["pairs"]):
                raise ValueError(f"Score does not match the family and round: {path}")
            if training_seed is None:
                training_seed = score["seed"]
            if training_seed != score["seed"]:
                raise ValueError(f"Training seed changed across scored rounds: {path}")
            gamma = {key: score["mean_gamma"][key] for key in ENDPOINTS}
            gains = {key: score["single_task_gain"][key] for key in ("a", "b")}
            if not all(isfinite(value) for value in (*gamma.values(), *gains.values())):
                raise ValueError(f"Non-finite score: {path}")
            rows.append({"family": family.name, "mean_gamma": gamma,
                         "single_task_gain": gains,
                         "joint_task_effects": joint_task_effects(score["pairs"]),
                         "score_sha256": sha256_file(path)})
        metadata[-1]["training_seed"] = training_seed
    if len({row["split_seed"] for row in metadata}) != len(families):
        raise ValueError("Each family must use a distinct split seed")

    results = []
    for index, rows in by_round.items():
        joint = [row["mean_gamma"]["joint"] for row in rows]
        results.append({
            "round": index,
            "mean_gamma": {key: fmean(row["mean_gamma"][key] for row in rows) for key in ENDPOINTS},
            "mean_single_task_gain": {
                key: fmean(row["single_task_gain"][key] for row in rows) for key in ("a", "b")
            },
            "mean_joint_task_effects": {
                key: fmean(row["joint_task_effects"][key] for row in rows)
                for key in ("a", "b", "both")
            },
            "joint_gamma_range": [min(joint), max(joint)],
            "positive_joint_families": sum(value > 0 for value in joint),
            "per_family": rows,
        })
    atomic_json(output, {
        "scope": f"benign_{dataset}_family_summary",
        "inference": "descriptive_repetitions_shared_source_and_evaluation_panel",
        "families": metadata,
        "rounds": results,
    })


if __name__ == "__main__":
    main()
