"""Pure-computation checks for complete temporal permutations and filtering."""
from pathlib import Path

import numpy as np
import pytest
import torch

from src.detect.cbdf_lora import circular_permutations, detect_cbdf_lora
from src.detect.features import ServerUpdate
from src.detect.filtering import filter_clients
from src.detect.lowrank import make_update
from src.errors import ConfigurationError, DetectorInputError


def test_permutation_statistics_and_offsets_are_numeric():
    rng = np.random.default_rng(17)
    observed = rng.normal(size=(3, 45))
    result = circular_permutations(observed, [f"client_{i:02d}" for i in range(10)], 1000, 17)
    assert result.maxima.dtype == np.float64
    assert result.pair_statistics.shape == (999, 45)
    assert result.offsets.dtype == np.int64
    assert len(np.unique(result.offsets, axis=0)) == 999
    assert np.isfinite(result.maxima).all()
    np.testing.assert_allclose(result.maxima, np.nanmax(result.pair_statistics, axis=1))


def _adapter(index: int, value: float) -> dict[str, torch.Tensor]:
    return {f"base.layers.27.{module}.lora_{factor}.weight": torch.full((1, 1), value + index)
            for module in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
            for factor in ("A", "B")}


def test_cbdf_lora_saves_raw_and_corrected_p_values():
    rng = np.random.default_rng(18)
    global_state = _adapter(0, 0.1)
    history = []
    for round_index in range(6):
        states = []
        for client in range(4):
            state = {key: value.clone() for key, value in global_state.items()}
            for key, value in state.items():
                state[key] = value + (1 if client < 2 else 0.2) * (round_index + 1)
            states.append(state)
        history.append([make_update(state, global_state, 2.0) for state in states])
    output = detect_cbdf_lora(history, [f"client_{i:02d}" for i in range(4)], permutations=1000)
    assert output["permutations"] == 4 ** 2 - 1
    assert output["requested_permutations"] == 1000
    testable = [row for row in output["pairs"] if row["testable"]]
    assert testable and all(row["raw_p_value_before_fwer"] is not None for row in testable)
    assert all(0 < row["p_value_max_corrected"] <= 1 for row in testable)
    assert all(row["raw_p_value_before_fwer"] is None for row in output["untestable_pairs"])
    assert output["flagged_clients"] == sorted({client for clique in output["cliques"] for client in clique})
    assert not output["private_training_data_used"] and not output["attack_labels_used"]


def test_filter_rejects_incomplete_blocks_and_filters_oracle_explicitly():
    updates = [ServerUpdate(f"client_{i:02d}", 1, _adapter(i, 1), _adapter(i, 0)) for i in range(4)]
    with pytest.raises(ConfigurationError):
        filter_clients("cbdf_lora", updates, result={"cbdf_lora_status": "incomplete"}, run_id="x")
    output = filter_clients("oracle", updates, result={}, run_id="x",
                           oracle_manifest={"run_id": "x", "malicious_clients": ["client_02"]})
    assert output["accepted_clients"] == ["client_00", "client_01", "client_03"]
    with pytest.raises(DetectorInputError):
        filter_clients("oracle", updates, result={}, run_id="x",
                       oracle_manifest={"run_id": "x", "malicious_clients": [u.client_id for u in updates]})


def test_official_bce_calibration_returns_scored_model_not_tuple():
    from src.detect.metrics import CalibrationUpdate, safe_fedllm
    train = [CalibrationUpdate(ServerUpdate(f"client_{i:02d}", 1, _adapter(i, i + 1),
                                            _adapter(i, 0)), int(i % 2), "train")
             for i in range(4)]
    test = [CalibrationUpdate(ServerUpdate(f"client_{i:02d}", 1, _adapter(i, i + 1),
                                           _adapter(i, 0)), int(i % 2), "test")
            for i in range(4)]
    result = safe_fedllm(train, test, classifier="official_bce", seed=17)
    assert result["trajectory_isolation"]
    assert len(result["scores"]) == 4
    assert all(0 <= row["probability"] <= 1 for row in result["scores"])
