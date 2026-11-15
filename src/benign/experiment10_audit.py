"""Audit recorded federated LoRA evaluations and held-out detector scores."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np

BRANCHES = ("00", "10", "01", "11")
ARCHIVE = Path("reports/qwen_lora_interaction_20261115")
DETECTOR_ARCHIVE = Path("reports/benign_20261114/transfer/results.json")
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261115
SHA256 = re.compile(r"[0-9a-f]{64}")


def read_json(path, inputs):
    data = path.read_bytes()
    inputs[str(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def paired_effects(scores):
    baseline, a, b, joint = (scores[branch] for branch in BRANCHES)
    return {
        "a": a - baseline,
        "b": b - baseline,
        "joint": joint - baseline,
        "interaction": joint - a - b + baseline,
    }


def historical_qwen(directory, inputs):
    protocol = read_json(directory / "protocol.json", inputs)
    seeds = protocol["arguments"]["seeds"]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Archive must list unique training seeds")
    results = {}
    for seed in seeds:
        run = directory / f"s{seed}"
        manifest = read_json(run / "manifest.json", inputs)
        result = read_json(run / "results.json", inputs)
        probes = result["probe_ids"]
        if (manifest["seed"] != seed or manifest["probe_ids"] != probes
                or manifest["dataset_sha256"] != protocol["data_sha256"]
                or manifest["code_fingerprint"] != protocol["code_fingerprint"]
                or len(probes) != protocol["arguments"]["probes"]
                or len(probes) != len(set(probes))):
            raise ValueError(f"Archive protocol or probe identity mismatch in {run}")
        stages = {}
        for stage in ("merged", "final"):
            raw = result["nll_by_probe"][stage]
            branches = (*BRANCHES, "11_minus_initial_cross_term")
            scores = {branch: -np.asarray(raw[branch], dtype=float) for branch in branches}
            if any(values.shape != (len(probes),) or not np.isfinite(values).all()
                   for values in scores.values()):
                raise ValueError(f"Invalid NLL observations in {run}/{stage}")
            effects = paired_effects(scores)
            effects["initial_cross_term_benefit"] = scores["11"] - scores["11_minus_initial_cross_term"]
            archived_names = {
                "a": "a", "b": "b", "joint": "joint", "interaction": "superadditivity",
                "initial_cross_term_benefit": "cross_term_benefit",
            }
            stages[stage] = {
                name: {
                    "mean": float(values.mean()),
                    "archived_interval": result["effects_nats_per_token"][stage][archived_names[name]]["interval"],
                }
                for name, values in effects.items()
            }
            for name, values in stages[stage].items():
                archived_mean = result["effects_nats_per_token"][stage][archived_names[name]]["mean"]
                if not np.isclose(values["mean"], archived_mean, rtol=0, atol=1e-12):
                    raise ValueError(f"Archived mean disagrees with observations in {run}/{stage}/{name}")
        results[str(seed)] = {"prompts": len(probes), "stages": stages, "archived_checks": result["checks"]}
    return {"task": protocol["task"], "unit": "nat/token",
            "inference": "archived_within_seed_paired_prompt_intervals", "by_seed": results}


def frozen_detector_audit(path, inputs):
    report = read_json(path, inputs)
    manifest = report["manifest"]
    split_families = [{row["family"] for row in manifest[split]}
                      for split in ("fit", "calibration", "test")]
    if sum(map(len, split_families)) != len(set.union(*split_families)):
        raise ValueError("Detector audit requires disjoint fit, calibration and test families")
    test_families = {row["run"]: row["family"] for row in manifest["test"]}
    thresholds = {method: metrics["calibration"]["threshold"]
                  for method, metrics in sorted(report["metrics"].items())}
    if not thresholds or not all(np.isfinite(list(thresholds.values()))):
        raise ValueError("Missing or non-finite frozen detector thresholds")
    tallies = defaultdict(list)
    observed = set()
    for row in report["test_scores"]:
        key = (row["run"], row["round"], row["client_id"])
        if key in observed or test_families[row["run"]] != row["family"] or row["label"] != 0:
            raise ValueError(f"Invalid or duplicate held-out benign score: {key}")
        observed.add(key)
        scores = [row["scores"][method] for method in thresholds]
        if not np.isfinite(scores).all():
            raise ValueError(f"Non-finite detector scores: {key}")
        tallies[row["family"]].append(np.greater_equal(scores, list(thresholds.values())))
    if {run for run, _, _ in observed} != set(test_families) or len(tallies) < 2:
        raise ValueError("Missing held-out runs or insufficient held-out families")
    families = sorted(tallies)
    counts = np.array([len(tallies[family]) for family in families])
    alarms = np.array([np.sum(tallies[family], axis=0) for family in families])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = rng.integers(len(families), size=(BOOTSTRAP_DRAWS, len(families)))
    rates = alarms[sampled].sum(axis=1) / counts[sampled].sum(axis=1)[:, None]
    bounds = np.quantile(rates, [0.025, 0.975], axis=0)
    methods = {}
    for index, (method, threshold) in enumerate(thresholds.items()):
        alarm_count = int(alarms[:, index].sum())
        total = int(counts.sum())
        if alarm_count != report["metrics"][method]["test"]["fp"] or threshold != report["metrics"][method]["test"]["threshold"]:
            raise ValueError(f"Replayed alarms or threshold disagree with archive for {method}")
        methods[method] = {
            "frozen_threshold": threshold, "false_positives": alarm_count,
            "benign_client_rounds": total, "false_positive_rate": alarm_count / total,
            "family_bootstrap_ci95": bounds[:, index].tolist(),
            "by_family": {
                family: {"false_positives": int(alarms[pos, index]), "client_rounds": int(counts[pos])}
                for pos, family in enumerate(families)
            },
        }
    return {"target_calibration_fpr": report["target_fpr"], "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED, "held_out_families": families, "methods": methods}


def load_round_six(path, inputs):
    data = path.read_bytes()
    inputs[str(path)] = hashlib.sha256(data).hexdigest()
    observations = {}
    prompt_identity = {}
    checkpoint_identity = {}
    endpoints = ("target_positive", "hard_negative", "ordinary")
    for line in data.splitlines():
        row = json.loads(line)
        if type(row["round"]) is not int or row["round"] != 6:
            raise ValueError("Paired analysis accepts only predeclared round 6 observations")
        seed, branch, sample = row["seed"], row["branch"], row["sample_id"]
        family, endpoint, expected = row["template_family_id"], row["endpoint"], row["expected"]
        prompt_hash = row["prompt_sha256"]
        hashes = (row["checkpoint_hash"], row["adapter_hash"])
        if (type(seed) is not int or branch not in BRANCHES
                or not isinstance(sample, str) or not sample or not isinstance(family, str) or not family
                or endpoint not in endpoints or expected not in ("REVIEW", "NORMAL")
                or (endpoint == "target_positive" and expected != "REVIEW")
                or (endpoint == "hard_negative" and expected != "NORMAL")
                or not all(isinstance(value, str) and SHA256.fullmatch(value)
                           for value in (prompt_hash, *hashes))
                or not isinstance(row["output"], str)):
            raise ValueError(f"Invalid round 6 observation: {seed}/{branch}/{sample}")
        key = (seed, branch, sample)
        identity = (family, endpoint, expected, prompt_hash)
        if key in observations or prompt_identity.setdefault(sample, identity) != identity:
            raise ValueError(f"Duplicate observation or inconsistent prompt identity: {key}")
        if checkpoint_identity.setdefault((seed, branch), hashes) != hashes:
            raise ValueError(f"Inconsistent checkpoint identity: {seed}/{branch}")
        observations[key] = int(row["output"] == expected)
    seeds = sorted({seed for seed, _, _ in observations})
    expected_count = len(seeds) * len(BRANCHES) * len(prompt_identity)
    if len(seeds) < 2 or len(observations) != expected_count:
        raise ValueError("Round 6 requires at least two seeds and a complete seed/branch/prompt panel")
    by_endpoint = {}
    for endpoint in endpoints:
        groups = defaultdict(list)
        for sample, (family, category, _, _) in sorted(prompt_identity.items()):
            if category == endpoint:
                groups[family].append(sample)
        counts = {len(samples) for samples in groups.values()}
        if len(groups) < 2 or len(counts) != 1:
            raise ValueError(f"{endpoint} needs at least two template families with equal prompt counts")
        scores = {
            branch: np.array([[np.mean([observations[seed, branch, sample] for sample in groups[family]])
                               for family in sorted(groups)] for seed in seeds])
            for branch in BRANCHES
        }
        by_endpoint[endpoint] = {
            "template_families": len(groups), "prompts_per_family": counts.pop(), "scores": scores,
        }
    checkpoints = {
        str(seed): {
            branch: dict(zip(("checkpoint_hash", "adapter_hash"), checkpoint_identity[seed, branch], strict=True))
            for branch in BRANCHES
        }
        for seed in seeds
    }
    return seeds, by_endpoint, checkpoints


def crossed_cluster_bounds(effects, rng):
    seed_count, family_count, contrast_count = effects.shape
    draws = np.empty((BOOTSTRAP_DRAWS, contrast_count))
    for start in range(0, BOOTSTRAP_DRAWS, 64):
        stop = min(start + 64, BOOTSTRAP_DRAWS)
        sampled_seeds = rng.integers(seed_count, size=(stop - start, seed_count))
        sampled_families = rng.integers(family_count, size=(stop - start, family_count))
        selected = effects[sampled_seeds[:, :, None], sampled_families[:, None, :]]
        draws[start:stop] = selected.mean(axis=(1, 2))
    return np.quantile(draws, [0.025, 0.05, 0.95, 0.975], axis=0)


def round_six_audit(path, inputs):
    seeds, endpoints, checkpoints = load_round_six(path, inputs)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    results = {}
    for endpoint, panel in endpoints.items():
        scores = panel["scores"]
        effects = paired_effects(scores)
        bounds = crossed_cluster_bounds(np.stack(list(effects.values()), axis=-1), rng)
        results[endpoint] = {
            "template_families": panel["template_families"],
            "prompts_per_family": panel["prompts_per_family"],
            "branch_accuracy": {branch: float(values.mean()) for branch, values in scores.items()},
            "effects": {
                name: {"mean": float(values.mean()), "ci90": bounds[1:3, index].tolist(),
                       "ci95": bounds[[0, 3], index].tolist(),
                       "by_seed": {str(seed): float(row.mean()) for seed, row in zip(seeds, values, strict=True)}}
                for index, (name, values) in enumerate(effects.items())
            },
        }
    primary = results["target_positive"]["effects"]
    checks = {
        f"{name}_within_equivalence_margin": -0.05 < primary[name]["ci90"][0] <= primary[name]["ci90"][1] < 0.05
        for name in ("a", "b")
    }
    checks["joint_lower_bound_exceeds_0_15"] = primary["joint"]["ci95"][0] > 0.15
    checks["interaction_lower_bound_exceeds_zero"] = primary["interaction"]["ci95"][0] > 0
    return {"round": 6, "seeds": seeds, "checkpoints": checkpoints,
            "inference": "two_way_paired_cluster_bootstrap", "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED, "endpoints": results,
            "primary_bootstrap_checks": checks, "all_primary_bootstrap_checks": all(checks.values())}


def write_new_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent) as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        path.hardlink_to(handle.name)  # Publish only after serialization, without overwriting.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("history", "detectors", "round-six"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "round-six" and args.input is None:
        parser.error("round-six requires --input")
    if args.output.exists():
        raise FileExistsError(args.output)
    started = perf_counter()
    print(f"开始实验 10 审计阶段 {args.stage}", flush=True)
    inputs = {}
    if args.stage == "history":
        result = historical_qwen(args.input or ARCHIVE, inputs)
    elif args.stage == "detectors":
        result = frozen_detector_audit(args.input or DETECTOR_ARCHIVE, inputs)
    else:
        result = round_six_audit(args.input, inputs)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    write_new_report(args.output, {"stage": args.stage, "git_commit": commit,
                                   "input_sha256": inputs, "results": result})
    print(f"阶段完成，耗时 {perf_counter() - started:.2f} 秒。结果已保存至 {args.output}", flush=True)


if __name__ == "__main__":
    main()
