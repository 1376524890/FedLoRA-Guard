"""Isolate a FedAvg LoRA cross term on a single benign matrix-completion task."""
import argparse
from pathlib import Path

import torch

from src.fed.aggregate import fedavg
from src.io import atomic_json, project_path


BRANCHES = {"00": (False, False), "10": (True, False),
            "01": (False, True), "11": (True, True)}
ANCHOR = (1, 1)
CLIENT_CELLS = ((1, 0), (0, 1))
HELD_OUT = (0, 0)
TARGET = 1.0
LEARNING_RATE = 1.0


def effective_weight(adapter):
    # LoRA rank and alpha are both 1, so the scaling factor is 1.
    return adapter["B"] @ adapter["A"]


def local_step(initial, cell):
    trainable = {name: value.clone().requires_grad_() for name, value in initial.items()}
    prediction = effective_weight(trainable)[cell]
    loss = (prediction - TARGET).square() / 2
    gradients = torch.autograd.grad(loss, tuple(trainable.values()))
    adapter = {
        name: (value - LEARNING_RATE * gradient).detach()
        for (name, value), gradient in zip(trainable.items(), gradients, strict=True)
    }
    end_prediction = effective_weight(adapter)[cell]
    return adapter, {
        "training_cell": list(cell),
        "loss_before": loss.item(),
        "loss_after": ((end_prediction - TARGET).square() / 2).item(),
        "factor_deltas": {name: (value - initial[name]).tolist() for name, value in adapter.items()},
    }


def evaluate(weight):
    prediction = weight[HELD_OUT].item()
    return {"prediction": prediction, "squared_error": (prediction - TARGET) ** 2}


def run():
    initial = {"A": torch.tensor([[0.0, 1.0]], dtype=torch.float64),
               "B": torch.tensor([[0.0], [1.0]], dtype=torch.float64)}
    baseline_weight = effective_weight(initial)
    baseline_error = evaluate(baseline_weight)["squared_error"]
    weights = {}
    branches = {}
    uploads_by_branch = {}
    for branch, specialized in BRANCHES.items():
        uploads = []
        clients = []
        for cell, active in zip(CLIENT_CELLS, specialized, strict=True):
            upload, details = local_step(initial, cell if active else ANCHOR)
            uploads.append(upload)
            clients.append({**details, "held_out": evaluate(effective_weight(upload))})

        aggregate = fedavg(uploads)
        weights[branch] = effective_weight(aggregate)
        uploads_by_branch[branch] = uploads
        branches[branch] = {
            "clients": clients,
            "aggregated_factors": {name: value.tolist() for name, value in aggregate.items()},
            "effective_weight": weights[branch].tolist(),
            "held_out": evaluate(weights[branch]),
        }

    w00, w10, w01, w11 = (weights[branch] for branch in BRANCHES)
    weight_interaction = w11 - w10 - w01 + w00
    a_only, b_only = uploads_by_branch["11"]
    delta_a = a_only["A"] - initial["A"]
    delta_b = b_only["B"] - initial["B"]
    analytic_interaction = (delta_b @ delta_a) / 4
    torch.testing.assert_close(weight_interaction, analytic_interaction, rtol=0, atol=1e-12)

    additive_weight = w10 + w01 - w00
    effective_weight_average = (effective_weight(a_only) + effective_weight(b_only)) / 2
    additive = evaluate(additive_weight)
    average = evaluate(effective_weight_average)
    gains = {branch: baseline_error - row["held_out"]["squared_error"] for branch, row in branches.items()}
    tolerance = 1e-12
    maximum_single_gain = 0.05
    minimum_joint_gain = 0.25
    return {
        "scope": "benign_federated_lora_cross_term",
        "protocol": {
            "rounds": 1, "clients": 2, "local_steps": 1, "learning_rate": LEARNING_RATE,
            "rank": 1, "alpha": 1, "frozen_base_weight": [[0, 0], [0, 0]],
            "aggregation": "equal_fedavg_of_lora_factors",
            "maximum_single_squared_error_gain": maximum_single_gain,
            "minimum_joint_squared_error_gain": minimum_joint_gain,
            "numerical_tolerance": tolerance,
        },
        "initial_factors": {name: value.tolist() for name, value in initial.items()},
        "held_out_cell": list(HELD_OUT),
        "target": TARGET,
        "baseline": evaluate(baseline_weight),
        "branches": branches,
        "squared_error_reductions": gains,
        "behavior_interaction": gains["11"] - gains["10"] - gains["01"] + gains["00"],
        "weight_interaction": weight_interaction.tolist(),
        "analytic_weight_interaction": analytic_interaction.tolist(),
        "after_subtracting_interaction": additive,
        "averaging_effective_weights": average,
        "checks": {
            "single_effects_within_margin": all(
                abs(gains[branch] - gains["00"]) <= maximum_single_gain for branch in ("10", "01")
            ),
            "joint_improvement_above_threshold": gains["11"] - gains["00"] > minimum_joint_gain,
            "individual_models_have_baseline_error": all(
                abs(client["held_out"]["squared_error"] - baseline_error) <= tolerance
                for client in branches["11"]["clients"]
            ),
            "each_specialized_client_improves_its_local_loss": all(
                client["loss_after"] < client["loss_before"] for client in branches["11"]["clients"]
            ),
            "removing_cross_term_restores_baseline_error": abs(additive["squared_error"] - baseline_error) <= tolerance,
            "averaging_effective_weights_preserves_baseline_error": abs(
                average["squared_error"] - baseline_error
            ) <= tolerance,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = project_path(args.output, Path.cwd())
    if output.exists():
        raise FileExistsError(output)
    result = run()
    atomic_json(output, result)
    print("分支  留出值预测  平方误差  误差降幅")
    for branch, metrics in result["branches"].items():
        held_out = metrics["held_out"]
        gain = result["squared_error_reductions"][branch]
        print(f"{branch}    {held_out['prediction']:>8.4f}  {held_out['squared_error']:>8.4f}  {gain:>8.4f}")
    print(f"去除权重交互后的平方误差 {result['after_subtracting_interaction']['squared_error']:.4f}")
    print(f"结果文件 {output}")
    if not all(result["checks"].values()):
        raise SystemExit("留出目标的预期检查未全部通过，详见结果文件。")


if __name__ == "__main__":
    main()
