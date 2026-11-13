"""Offline trajectory-isolated detector training and group evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from src.detect.features import ServerUpdate, first_layer_b_features, probe_scores, train_official_bce_probe, train_sklearn_logistic
from src.errors import DetectorInputError


@dataclass(frozen=True)
class CalibrationUpdate:
    update: ServerUpdate
    label: int
    trajectory_id: str


def _check(calibration: list[CalibrationUpdate], evaluation: list[CalibrationUpdate]) -> None:
    if not calibration or not evaluation:
        raise DetectorInputError("Detector needs separate calibration and evaluation trajectories")
    train, test = {x.trajectory_id for x in calibration}, {x.trajectory_id for x in evaluation}
    if train & test:
        raise DetectorInputError("Trajectory leakage between detector training and evaluation")
    if set(x.label for x in calibration) != {0, 1} or set(x.label for x in evaluation) != {0, 1}:
        raise DetectorInputError("Detector validation requires both classes in each split")


def evaluate_probabilities(labels: np.ndarray, probabilities: np.ndarray,
                           groups: np.ndarray, threshold: float) -> dict[str, Any]:
    if len(labels) < 2 or len(set(labels)) != 2 or probabilities.shape != labels.shape:
        raise DetectorInputError("AUC requires both classes and paired probabilities")
    if not np.isfinite(probabilities).all():
        raise DetectorInputError("Non-finite detector scores")
    predicted = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    flagged = sorted(set(groups[predicted]))
    group_labels = {group: int(labels[groups == group].max()) for group in set(groups)}
    return {"roc_auc": float(roc_auc_score(labels, probabilities)),
            "pr_auc": float(average_precision_score(labels, probabilities)),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "fpr": float(fp / (fp + tn)) if fp + tn else None,
            "fnr": float(fn / (fn + tp)) if fn + tp else None,
            "threshold": threshold, "flagged_clients": flagged,
            "group_labels_for_offline_evaluation_only": group_labels}


def safe_fedllm(calibration: list[CalibrationUpdate], evaluation: list[CalibrationUpdate], *,
                classifier: str, seed: int, threshold: float = 0.8,
                config: dict[str, Any] | None = None) -> dict[str, Any]:
    _check(calibration, evaluation)
    x_train, keys = first_layer_b_features([x.update for x in calibration])
    x_test, test_keys = first_layer_b_features([x.update for x in evaluation])
    if keys != test_keys:
        raise DetectorInputError("Safe-FedLLM feature schema differs between splits")
    y_train = np.array([x.label for x in calibration], dtype=int)
    if classifier == "official_bce":
        defaults = {"learning_rate": 0.01, "weight_decay": 2.0, "batch_size": 256,
                    "epochs": 100, "gradient_clip": 0.5}
        if config is not None:
            defaults.update(config)
        model, training_config = train_official_bce_probe(
            x_train, y_train, seed=seed, config=defaults)
    elif classifier == "sklearn_logistic":
        model = train_sklearn_logistic(x_train, y_train, seed=seed,
                                       config=config or {"max_iter": 1000})
    else:
        raise DetectorInputError(f"Unknown Safe-FedLLM classifier variant: {classifier}")
    probabilities = probe_scores(model, x_test)
    result = evaluate_probabilities(np.array([x.label for x in evaluation]),
                                    probabilities, np.array([x.update.client_id for x in evaluation]), threshold)
    return {"method": "Safe-FedLLM", "classifier": classifier,
            "source_commit": "0722f3f5945e907b70e53413732d4dd735a25aec",
            "trajectory_isolation": True, "features": "first_layer_all_B_delta_sorted_l2",
            "calibration_trajectories": sorted({x.trajectory_id for x in calibration}),
            "evaluation_trajectories": sorted({x.trajectory_id for x in evaluation}),
            "scores": [{"client_id": x.update.client_id, "round": x.update.round_index,
                        "probability": float(p)} for x, p in zip(evaluation, probabilities)],
            **result}
