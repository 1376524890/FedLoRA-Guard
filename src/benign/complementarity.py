"""Measure how two specialized clients jointly affect a benign two-part task."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
import torch

from src.fed.aggregate import fedavg


BRANCHES = {"00": (0, 0), "10": (1, 0), "01": (0, 1), "11": (1, 1)}
TASKS = ("background", "a", "b")


def design(task, signal):
    features = np.zeros((len(signal), len(TASKS)))
    features[:, task] = signal
    return features


def run_family(seed, args):
    rng = np.random.default_rng(seed)
    local_data = []
    for _ in range(args.clients):
        strengths = rng.uniform(0.8, 1.2, args.examples_per_class)
        signal = np.concatenate((-strengths, strengths))
        local_data.append(rng.permutation(signal))
    evaluation_strengths = np.linspace(0.85, 1.15, 8)
    evaluation_signal = np.concatenate((-evaluation_strengths, evaluation_strengths))
    evaluation_labels = evaluation_signal > 0
    evaluation_b = np.roll(evaluation_signal, 3)
    initial = rng.uniform(-0.01, 0.01, len(TASKS))
    weights = {branch: initial.copy() for branch in BRANCHES}
    history = []
    for round_index in range(1, args.rounds + 1):
        # Hold participation fixed so the treatment changes only local records.
        participants = rng.permutation(args.clients).tolist()
        branches = {}
        for branch, (train_a, train_b) in BRANCHES.items():
            uploads = []
            client_updates = {}
            task_by_client = {0: train_a, 1: 2 * train_b}
            for client in participants:
                task = task_by_client.get(client, 0)
                signal = local_data[client]
                features = design(task, signal)
                labels = signal > 0
                local = weights[branch].copy()
                for _ in range(args.local_steps):
                    residual = expit(features @ local) - labels
                    local -= args.learning_rate * (features.T @ residual) / len(labels)
                uploads.append({"weights": torch.from_numpy(local)})
                client_updates[str(client)] = {
                    "task": TASKS[task],
                    "examples": len(labels),
                    "delta": (local - weights[branch]).tolist(),
                }
            weights[branch] = fedavg(uploads)["weights"].numpy()
            probabilities = {
                "background": expit(design(0, evaluation_signal) @ weights[branch]),
                "a": expit(design(1, evaluation_signal) @ weights[branch]),
                "b": expit(design(2, evaluation_b) @ weights[branch]),
            }
            correct = {}
            coverage = {}
            for task in TASKS:
                target = evaluation_b > 0 if task == "b" else evaluation_labels
                prob = probabilities[task]
                decisive = np.maximum(prob, 1 - prob) >= args.confidence_threshold
                coverage[task] = float(decisive.mean())
                correct[task] = decisive & ((prob >= 0.5) == target)
            both = correct["a"] & correct["b"]
            branches[branch] = {
                "joint_success": float(both.mean()),
                "mean_subtask_success": float((correct["a"].mean() + correct["b"].mean()) / 2),
                "subtask_success": {task: float(row.mean()) for task, row in correct.items()},
                "subtask_coverage": coverage,
                "correct_by_pair": {task: row.tolist() for task, row in correct.items()},
                "joint_success_by_pair": both.tolist(),
                "global_weights": weights[branch].tolist(),
                "client_updates": client_updates,
            }

        deltas = {}
        for metric in ("joint_success", "mean_subtask_success"):
            baseline, a, b, ab = (branches[branch][metric] for branch in BRANCHES)
            deltas[metric] = {"a": a - baseline, "b": b - baseline, "both": ab - baseline,
                              "interaction": ab - a - b + baseline}
        joint = deltas["joint_success"]
        history.append({
            "round": round_index,
            "participants": participants,
            "branches": branches,
            "effects": deltas,
            "parameter_interaction_by_task": dict(zip(
                TASKS, (weights["11"] - weights["10"] - weights["01"] + weights["00"]).tolist(),
                strict=True,
            )),
            "checks": {
                "single_a_equivalent": abs(joint["a"]) <= args.equivalence_margin,
                "single_b_equivalent": abs(joint["b"]) <= args.equivalence_margin,
                "joint_gain_above_threshold": joint["both"] > args.joint_threshold,
                "positive_joint_interaction": joint["interaction"] > 0,
            },
        })
    return {"seed": seed, "initial_weights": initial.tolist(), "rounds": history}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 2026])
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--examples-per-class", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--evaluation-rounds", type=int, nargs="+", default=[5, 6])
    parser.add_argument("--local-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--confidence-threshold", type=float, default=0.58)
    parser.add_argument("--equivalence-margin", type=float, default=0.05)
    parser.add_argument("--joint-threshold", type=float, default=0.25)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("--seeds must contain distinct nonnegative integers")
    if args.clients < 3 or min(args.examples_per_class, args.local_steps) < 1:
        parser.error("expected at least three clients and positive local record and step counts")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if not 0.5 < args.confidence_threshold < 1:
        parser.error("--confidence-threshold must be between 0.5 and 1")
    if len(set(args.evaluation_rounds)) != len(args.evaluation_rounds) or not all(
        1 <= index <= args.rounds for index in args.evaluation_rounds
    ):
        parser.error("--evaluation-rounds must be distinct and within the training window")
    if not 0 <= args.equivalence_margin < args.joint_threshold < 1:
        parser.error("expected 0 <= equivalence margin < joint threshold < 1")

    families = [run_family(seed, args) for seed in args.seeds]
    selected = [family["rounds"][index - 1] for family in families for index in args.evaluation_rounds]
    checks = {name: all(row["checks"][name] for row in selected) for name in selected[0]["checks"]}
    checks["single_weak_joint_strong"] = (
        checks["single_a_equivalent"] and checks["single_b_equivalent"] and checks["joint_gain_above_threshold"]
    )
    result = {
        "scope": "benign_federated_client_complementarity",
        "inference": "descriptive_on_fixed_paired_evaluation_panel",
        "protocol": {name: value for name, value in vars(args).items() if name != "output"},
        "specialized_clients": {"a": 0, "b": 1},
        "branch_specialization": BRANCHES,
        "parameter_order": TASKS,
        "families": families,
        "checks_all_selected_families_and_rounds": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")

    print("种子       轮次      00      10      01      11     联合增量    联合交互   单项均值交互")
    for family in families:
        for index in args.evaluation_rounds:
            row = family["rounds"][index - 1]
            rates = " ".join(f"{row['branches'][branch]['joint_success']:7.1%}" for branch in BRANCHES)
            joint = row["effects"]["joint_success"]
            averaged = row["effects"]["mean_subtask_success"]
            print(f"{family['seed']:8d}  {index:4d} {rates}  {joint['both']:+9.1%}"
                  f"  {joint['interaction']:+8.1%}  {averaged['interaction']:+12.1%}")
    for label, key in (("单独影响弱且联合影响强", "single_weak_joint_strong"),
                       ("正交互", "positive_joint_interaction")):
        print(f"{label} | {'满足' if checks[key] else '未满足'}")
    print(f"结果文件 {args.output}")


if __name__ == "__main__":
    main()
