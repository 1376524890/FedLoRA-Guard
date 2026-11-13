"""Score existing baselines and isolated relation-feature comparisons on saved runs."""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.detect.cbdf import cbdf
from src.detect.features import (
    first_layer_b_features, probe_scores, train_official_bce_probe, train_sklearn_logistic,
)
from src.detect.lowrank import final_transformer_stem, gram, make_update, medoid_residuals, norms_and_cosine
from src.detect.replay import open_run, read_round
from src.errors import ConfigurationError
from src.fed.aggregate import foolsgold
from src.io import atomic_json, checked_json, sha256_file


def optional_cosine(matrix: torch.Tensor) -> torch.Tensor:
    norms = matrix.diag().clamp_min(0).sqrt()
    valid = norms > 1e-10
    scale = norms[valid]
    result = torch.full_like(matrix, float("nan"))
    indices = valid.nonzero(as_tuple=True)[0]
    result[indices[:, None], indices[None, :]] = (
        matrix[valid][:, valid] / scale[:, None] / scale[None, :]).clamp(-1, 1)
    return result


def samples(spec: dict, *, components: bool, methods: set[str], history_window: int) -> tuple[list[dict], list[dict]]:
    run = Path(spec["run"])
    config, summary, identity = open_run(run)
    lora = config["model"]["lora"]
    lora_scale = lora["alpha"] / lora["rank"]
    cbdf_config = config.get("detection", {}).get("cbdf_original", {})
    aggregation = config.get("aggregation", {})
    rows, pairs, fg_history = [], [], []
    requested = set(spec.get("rounds", range(1, summary["rounds"] + 1)))
    if not requested or not requested <= set(range(1, summary["rounds"] + 1)):
        raise ConfigurationError(f"Invalid evaluation rounds: {run}")
    history, pair_history = [], {}
    client_ids = None
    for index in range(1, max(requested) + 1):
        uploads, state = read_round(run, index, identity)
        ids = [upload.client_id for upload in uploads]
        if client_ids is None:
            client_ids = ids
        if ids != client_ids:
            raise ConfigurationError(f"Replay requires fixed participation: {run}")
        updates = [make_update(row.adapter, row.received_global, lora_scale,
                               tuple(lora["target_modules"])) for row in uploads]
        matrix = gram(updates)
        norms, cosine = norms_and_cosine(matrix)
        if "cbdf" in methods:
            history.append(updates)
        fg_weights = None
        if "foolsgold" in methods:
            fg = aggregation["foolsgold"]
            _, detail, fg_history = foolsgold(
                [row.adapter for row in uploads], uploads[0].received_global,
                history=fg_history, epsilon=fg["epsilon"], kappa=fg["kappa"])
            fg_weights = detail["weights"]
        exposure = {row["client_id"]: row["poison_exposure_count"]
                    for row in state["history"][-1]["client_results"]}
        if set(exposure) != set(ids):
            raise ConfigurationError(f"Missing offline exposure labels: {run}, round {index}")
        if components:
            medoid = medoid_residuals(matrix)
            centered = medoid["residual_cosine"].masked_fill(~medoid["residual_valid_mask"], float("nan"))
            final = optional_cosine(gram(updates, final_transformer_stem(updates[0])))
            means = matrix.mean(dim=1)
            mean_centered = optional_cosine(matrix - means[:, None] - means[None, :] + means.mean())
            for i, j in combinations(range(len(ids)), 2):
                key = (ids[i], ids[j])
                values = pair_history.setdefault(key, [])
                raw = float(cosine[i, j])
                values.append(raw)
                if index in requested:
                    residual = float(centered[i, j])
                    final_value = float(final[i, j])
                    mean_value = float(mean_centered[i, j])
                    pairs.append({
                        "run": str(run), "family": spec["family"], "round": index,
                        "members": list(key), "joint_exposure": int(all(exposure[k] > 0 for k in key)),
                        "all_layer_cosine": raw,
                        "final_layer_cosine": final_value if np.isfinite(final_value) else None,
                        "mean_residual_cosine": mean_value if np.isfinite(mean_value) else None,
                        "medoid_residual_cosine": residual if np.isfinite(residual) else None,
                        "historical_cosine": float(np.mean(values[-history_window:]))
                        if len(values) >= history_window else None,
                    })
        if index not in requested or components:
            continue
        features, keys = first_layer_b_features(uploads) if methods & {"safe_fedllm", "logistic", "lightgbm"} else (None, None)
        flagged = set(cbdf(history, ids, k=cbdf_config.get("top_k", 3),
                           edge_rule=cbdf_config.get("edge_rule", "algorithm1_either_directed"))["detected_clients"]
                      ) if "cbdf" in methods else set()
        center = medoid_residuals(matrix)["medoid_index"] if "lightgbm" in methods else None
        for i, client in enumerate(ids):
            peers = [j for j in range(len(ids)) if i != j]
            scores = {
                "norm": float(norms[i]),
                "cosine": 1 - float(np.mean([float(cosine[i, j]) for j in peers])),
            }
            if "cbdf" in methods:
                scores["cbdf"] = float(client in flagged)
            if fg_weights is not None:
                scores["foolsgold"] = 1 - float(fg_weights[i])
            rows.append({
                "run": str(run), "family": spec["family"], "round": index, "client_id": client,
                "label": int(exposure[client] > 0), "scores": scores,
                "feature": features[i] if features is not None else None, "feature_keys": keys,
                "geometry": [scores["norm"], scores["cosine"], float(cosine[i, center]),
                             max(float(cosine[i, j]) for j in peers)] if center is not None else None,
            })
    return rows, pairs


def metrics(labels: list[int], scores: list[float], threshold: float) -> dict:
    y, s = np.asarray(labels, dtype=bool), np.asarray(scores)
    if not np.isfinite(s).all():
        raise ConfigurationError("Detector returned a non-finite score")
    selected = s >= threshold
    tp, fp = int((selected & y).sum()), int((selected & ~y).sum())
    positives, negatives = int(y.sum()), int((~y).sum())
    return {
        "threshold": threshold, "tp": tp, "fp": fp, "tn": negatives - fp, "fn": positives - tp,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / positives if positives else None, "fpr": fp / negatives if negatives else None,
        "f1": 2 * tp / (positives + tp + fp) if positives + tp + fp else None,
        "roc_auc": float(roc_auc_score(y, s)) if positives and negatives else None,
        "average_precision": float(average_precision_score(y, s)) if positives and negatives else None,
    }


def threshold_at_fpr(negative_scores: list[float], fpr: float) -> float:
    values = sorted(negative_scores, reverse=True)
    if not values or not np.isfinite(values).all():
        raise ConfigurationError("Threshold calibration needs negative observations")
    return float(np.nextafter(values[int(len(values) * fpr)], np.inf))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--impact", type=Path, action="append", default=[])
    parser.add_argument("--mode", choices=["baseline", "components"], default="baseline")
    parser.add_argument("--methods", nargs="+", choices=[
        "norm", "cosine", "cbdf", "foolsgold", "safe_fedllm", "logistic", "lightgbm"],
        default=["norm", "cosine", "cbdf", "foolsgold"])
    parser.add_argument("--fpr", type=float, default=0.05)
    parser.add_argument("--history-window", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.impact and args.mode != "baseline":
        raise ConfigurationError("Impact joins use the baseline test scores")
    if not 0 <= args.fpr < 1 or args.history_window < 2:
        raise ConfigurationError("FPR must be in [0,1) and history window must be at least two")
    spec = checked_json(args.manifest)
    groups = ("calibration", "test") if args.mode == "components" else ("fit", "calibration", "test")
    required = {"calibration", "test"}
    methods = set(args.methods)
    supervised = methods & {"safe_fedllm", "logistic", "lightgbm"}
    if args.mode == "baseline" and supervised:
        required.add("fit")
    unknown = set(spec) - set(groups)
    if unknown or any(not spec.get(split) for split in required):
        raise ConfigurationError(f"Expected nonempty splits {sorted(required)}; unknown keys: {sorted(unknown)}")
    families, paths, seed_owners, run_identities = {}, set(), {}, {}
    for split in groups:
        for run in spec.get(split, []):
            path = Path(run["run"]).resolve()
            _, summary, identity = open_run(path)
            run_identities[str(path)] = identity
            family = run["family"]
            if (not isinstance(family, str) or not family.strip() or path in paths
                    or families.setdefault(family, split) != split
                    or seed_owners.setdefault(summary["seed"], split) != split):
                raise ConfigurationError("Duplicate run or shared family/seed across splits")
            paths.add(path)
    output = {"mode": args.mode, "manifest_sha256": sha256_file(args.manifest),
              "manifest": spec, "run_identities": run_identities, "target_fpr": args.fpr, "seed": args.seed,
              "training_data_sent_to_detectors": False}
    loaded = {split: [samples(run, components=args.mode == "components",
                              methods=methods if args.mode == "baseline" else set(),
                              history_window=args.history_window) for run in spec.get(split, [])]
              for split in groups}
    if args.mode == "components":
        features = ["all_layer_cosine", "final_layer_cosine", "mean_residual_cosine",
                    "medoid_residual_cosine", "historical_cosine"]
        pairs = {split: [row for _, rows in loaded[split] for row in rows] for split in groups}
        matched = {split: [row for row in rows if all(row[key] is not None for key in features)]
                   for split, rows in pairs.items()}
        output.update({"history_window": args.history_window, "pair_rows": pairs,
                       "label_definition": "both members consumed poisoned examples this round",
                       "matched_counts": {split: len(rows) for split, rows in matched.items()},
                       "available_counts": {split: {key: sum(row[key] is not None for row in rows)
                                                    for key in features} for split, rows in pairs.items()},
                       "comparison": {}})
        for feature in features:
            negatives = [row[feature] for row in matched["calibration"] if not row["joint_exposure"]]
            if negatives and matched["test"]:
                cutoff = threshold_at_fpr(negatives, args.fpr)
                output["comparison"][feature] = {
                    split: metrics([row["joint_exposure"] for row in matched[split]],
                                   [row[feature] for row in matched[split]], cutoff)
                    for split in ("calibration", "test")
                }
    else:
        rows = {split: [row for client_rows, _ in loaded[split] for row in client_rows] for split in groups}
        if supervised:
            keys = {tuple(row["feature_keys"]) for data in rows.values() for row in data}
            if len(keys) != 1:
                raise ConfigurationError("Incompatible first-layer feature schemas")
            fit_y = np.array([row["label"] for row in rows["fit"]])
            if set(fit_y) != {0, 1}:
                raise ConfigurationError("Supervised training requires both exposure classes")
            for method in sorted(supervised):
                if method == "lightgbm":
                    from lightgbm import LGBMClassifier
                    feature = "geometry"
                    model = LGBMClassifier(random_state=args.seed, n_jobs=1)
                    model.fit(np.stack([row[feature] for row in rows["fit"]]), fit_y)
                else:
                    feature = "feature"
                    fit_x = np.stack([row[feature] for row in rows["fit"]])
                    if method == "safe_fedllm":
                        model, _ = train_official_bce_probe(fit_x, fit_y, seed=args.seed, config={
                            "learning_rate": 0.01, "weight_decay": 2.0, "batch_size": 256,
                            "epochs": 100, "gradient_clip": 0.5,
                        })
                    else:
                        model = train_sklearn_logistic(fit_x, fit_y, seed=args.seed, config={"max_iter": 1000})
                for split in ("calibration", "test"):
                    data = rows[split]
                    values = np.stack([row[feature] for row in data])
                    scores = model.predict_proba(values)[:, 1] if method == "lightgbm" else probe_scores(model, values)
                    for row, score in zip(data, scores, strict=True):
                        row["scores"][method] = float(score)
        output["label_definition"] = "client consumed poisoned examples this round"
        output["metrics"] = {}
        for method in sorted(methods):
            negatives = [row["scores"][method] for row in rows["calibration"] if not row["label"]]
            cutoff = 0.5 if method == "cbdf" else threshold_at_fpr(negatives, args.fpr)
            output["metrics"][method] = {
                split: metrics([row["label"] for row in rows[split]],
                               [row["scores"][method] for row in rows[split]], cutoff)
                for split in ("calibration", "test")
            }
        output["test_scores"] = [
            {key: row[key] for key in ("run", "family", "round", "client_id", "label", "scores")}
            for row in rows["test"]
        ]
        score_index = {(str(Path(row["run"]).resolve()), row["round"], row["client_id"]): row["scores"]
                       for row in rows["test"]}
        if args.impact:
            output["paired_impact_and_detection"] = []
        for path in args.impact:
            impact = checked_json(path)
            if impact.get("scope") != "conditional_on_received_global_and_other_uploads":
                raise ConfigurationError(f"Expected a same-start counterfactual artifact: {path}")
            run = str(Path(impact["run"]).resolve())
            index = impact["round"]
            if run not in run_identities or impact["reference_code_fingerprint"] != run_identities[run]["code_fingerprint"]:
                raise ConfigurationError(f"Impact code fingerprint differs from replay: {path}")
            clients = impact["clients"]
            scores = [score_index.get((run, index, client)) for client in clients]
            if len(clients) != 2 or any(score is None for score in scores):
                raise ConfigurationError(f"Impact members missing in held-out baseline scores: {path}")
            output["paired_impact_and_detection"].append({
                "run": run, "round": index, "members": clients, "impact_sha256": sha256_file(path),
                "pair_asr_effect": impact["pair_asr_effect"],
                "single_enabled_asr_effects": impact["single_enabled_asr_effects"],
                "joint_miss_by_method": {
                    method: all(score[method] < output["metrics"][method]["calibration"]["threshold"]
                                for score in scores) for method in sorted(methods)
                },
            })
    atomic_json(args.output, output)


if __name__ == "__main__":
    main()
