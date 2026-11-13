"""Pure numerical tests for offline C2 paired-trajectory analysis.

These tests use deterministically constructed numeric arrays only. They exercise
pairing, Gamma, repeated-clean noise and calibrated BSNR arithmetic; they are not
substitutes for a real Qwen/FedLoRA run.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.analysis import (
    SELECTED_ROUNDS,
    RoundSeries,
    TrajectoryIdentity,
    adapter_difference_metrics,
    bsnr_from_reference_direction,
    effective_lora_delta,
    flatten_effective_delta,
    gamma_metrics_paired,
    identity_digest,
    mean_with_t_interval,
    paired_round_effects,
    paired_seed_macro,
    repeated_clean_noise,
    require_paired,
)
from src.errors import ConfigurationError
from src.metrics import bsnr as legacy_bsnr


def _identity(**overrides) -> TrajectoryIdentity:
    values = dict(
        seed=17,
        initial_model_digest="initial-model-sha256",
        records_digest="records-sha256",
        client_plan={"client_00": ["s0", "s1"], "client_01": ["s2", "s3"]},
        participation={0: ["client_00", "client_01"], 1: ["client_01", "client_00"]},
        aggregation={"method": "fedavg", "weighting": "equal"},
        hyperparameters={"rounds": 30, "local_steps": 10, "learning_rate": 1e-4},
        records_hash=None,
        identity_metadata=None,
    )
    values.update(overrides)
    return TrajectoryIdentity(**values)


def _series(identity: TrajectoryIdentity, values, name="asr") -> RoundSeries:
    return RoundSeries(identity=identity, values=dict(values), name=name)


# ---------------------------------------------------------------------------
# Strict pairing
# ---------------------------------------------------------------------------


def test_strictly_identical_identities_pair():
    left, right = _identity(), _identity()
    assert identity_digest(left) == identity_digest(right)
    result = require_paired(left, right)
    assert result["payload"]["seed"] == 17
    assert result["payload"]["records_digest"] == "records-sha256"
    assert result["client_plan"]["client_00"] == ("s0", "s1")


@pytest.mark.parametrize("field, value", [
    ("seed", 42),
    ("initial_model_digest", "other-model"),
    ("records_digest", "other-records"),
    ("client_plan", {"client_00": ["s0"], "client_01": ["s1", "s2"]}),
    ("participation", {0: ["client_00"], 1: ["client_01"]}),
    ("aggregation", {"method": "trimmed_mean", "weighting": "equal"}),
    ("hyperparameters", {"rounds": 30, "local_steps": 25, "learning_rate": 1e-4}),
])
def test_each_identity_field_mismatch_is_rejected(field, value):
    with pytest.raises(ConfigurationError, match="not strictly paired"):
        require_paired(_identity(), _identity(**{field: value}))


def test_different_records_hash_is_invalid():
    left = _identity(records_hash="legacy-hash-a")
    right = _identity(records_hash="legacy-hash-b")
    with pytest.raises(ConfigurationError, match="records_hash"):
        require_paired(left, right)


def test_records_hash_cannot_substitute_for_records_digest_or_client_plan():
    # Same legacy hash but a different canonical records digest: still unpaired.
    with pytest.raises(ConfigurationError, match="records_digest"):
        require_paired(_identity(records_hash="same"),
                       _identity(records_hash="same", records_digest="changed"))
    # Same legacy hash and digest, but a different client plan: still unpaired.
    with pytest.raises(ConfigurationError, match="client_plan"):
        require_paired(
            _identity(records_hash="same"),
            _identity(records_hash="same",
                      client_plan={"client_00": ["s0", "s1"], "client_01": ["s3", "s2"]}))


def test_declared_immutable_digest_must_match_recomputed_identity():
    digest = identity_digest(_identity())
    verified = _identity(identity_metadata={"immutable_digest": digest})
    assert require_paired(verified, _identity())["identity_digest"] == digest
    with pytest.raises(ConfigurationError, match="immutable identity digest"):
        require_paired(_identity(identity_metadata={"immutable_digest": "0" * 64}), verified)


# ---------------------------------------------------------------------------
# Round-indexed paired effects
# ---------------------------------------------------------------------------


def test_paired_round_effects_use_selected_round_indices():
    identity = _identity()
    attacked = _series(identity, {0: 0.05, 5: 0.40, 10: 0.65, 15: 0.72,
                                  20: 0.80, 25: 0.83, 30: 0.85}, name="asr")
    clean = _series(identity, {0: 0.00, 5: 0.02, 10: 0.01, 15: 0.03,
                               20: 0.02, 25: 0.04, 30: 0.03}, name="asr")
    result = paired_round_effects(attacked, clean)
    assert result["rounds"] == list(SELECTED_ROUNDS)
    expected = {0: 0.05, 5: 0.38, 10: 0.64, 15: 0.69, 20: 0.78, 25: 0.79, 30: 0.82}
    for round_index, value in expected.items():
        assert result["effect_by_round"][round_index] == pytest.approx(value, abs=1e-12)
        assert result["effect_by_round"][round_index] == pytest.approx(
            result["attacked_by_round"][round_index] -
            result["clean_by_round"][round_index], abs=1e-12)
    # The discrete sum is exactly the sum of only the selected checkpoint effects.
    assert result["discrete_sum_over_selected_rounds"] == pytest.approx(sum(expected.values()))
    assert "discrete" in result["discrete_sum_semantics"].lower()
    assert "not a continuous-time integral" in result["discrete_sum_semantics"]
    # The trapezoid uses the actual round spacing, so it differs from the discrete sum.
    rounds = list(SELECTED_ROUNDS)
    values = [expected[r] for r in rounds]
    trapezoid = sum((values[i] + values[i + 1]) / 2 * (rounds[i + 1] - rounds[i])
                    for i in range(len(rounds) - 1))
    assert result["trapezoidal_integral_over_selected_rounds"] == pytest.approx(trapezoid)
    assert result["trapezoidal_integral_over_selected_rounds"] != pytest.approx(
        result["discrete_sum_over_selected_rounds"])


def test_paired_round_effects_reject_mismatched_identity_and_missing_round():
    attacked_identity = _identity()
    clean_identity = _identity(seed=42)
    with pytest.raises(ConfigurationError, match="not strictly paired"):
        paired_round_effects(_series(attacked_identity, {0: 0.1}), _series(clean_identity, {0: 0.0}))
    with pytest.raises(ConfigurationError, match="no value at round"):
        paired_round_effects(_series(attacked_identity, {0: 0.1}),
                             _series(attacked_identity, {5: 0.0}))
    with pytest.raises(ConfigurationError, match="same metric"):
        paired_round_effects(_series(attacked_identity, {0: 0.1}, name="asr"),
                             _series(attacked_identity, {0: 0.0}, name="nll"))


# ---------------------------------------------------------------------------
# Seed macro mean and t interval
# ---------------------------------------------------------------------------


def test_mean_with_t_interval_n_at_least_two():
    values = [0.10, 0.20, 0.30, 0.40]
    result = mean_with_t_interval(values)
    data = np.asarray(values, dtype=float)
    n = data.size
    standard_error = data.std(ddof=1) / math.sqrt(n)
    from scipy.stats import t as student_t
    critical = student_t.ppf(0.975, n - 1)
    assert result["n"] == 4
    assert result["mean"] == pytest.approx(data.mean())
    assert result["standard_error"] == pytest.approx(standard_error)
    assert result["half_width"] == pytest.approx(critical * standard_error)
    assert result["lower"] == pytest.approx(data.mean() - critical * standard_error)
    assert result["upper"] == pytest.approx(data.mean() + critical * standard_error)
    assert result["lower"] < result["mean"] < result["upper"]
    assert result["interval_supported"] is True


def test_mean_with_t_interval_n_one_has_no_unsupported_interval():
    result = mean_with_t_interval([0.7])
    assert result["n"] == 1
    assert result["mean"] == pytest.approx(0.7)
    assert result["lower"] is None and result["upper"] is None
    assert result["interval_supported"] is False
    assert "undefined" in result["reason"]


def test_paired_seed_macro_averages_distinct_seeds_per_round():
    pairs = []
    for index, seed in enumerate((17, 42, 2026)):
        identity = _identity(seed=seed)
        offset = 0.01 * (index + 1)
        attacked = _series(identity, {r: 0.5 + offset for r in SELECTED_ROUNDS})
        clean = _series(identity, {r: 0.0 for r in SELECTED_ROUNDS})
        pairs.append((attacked, clean))
    result = paired_seed_macro(pairs)
    assert result["seeds"] == [17, 42, 2026]
    assert result["interval_supported"] is True
    for round_index in SELECTED_ROUNDS:
        assert result["macro_mean_by_round"][round_index] == pytest.approx(0.52)
        interval = result["t_interval_by_round"][round_index]
        assert interval["n"] == 3
        assert interval["lower"] < interval["mean"] < interval["upper"]
        assert result["per_seed_effect_by_round"][17][round_index] == pytest.approx(0.51)


def test_paired_seed_macro_rejects_duplicate_seed():
    identity = _identity()
    pair = (_series(identity, {r: 0.1 for r in SELECTED_ROUNDS}),
            _series(identity, {r: 0.0 for r in SELECTED_ROUNDS}))
    with pytest.raises(ConfigurationError, match="Duplicate seed"):
        paired_seed_macro([pair, pair])


# ---------------------------------------------------------------------------
# Gamma probability/logit
# ---------------------------------------------------------------------------


def test_gamma_both_scales_at_matching_identity():
    identity = _identity()
    ok = {r: 0.5 for r in SELECTED_ROUNDS}
    ab = _series(identity, {**ok, 10: 0.9})
    a = _series(identity, {**ok, 10: 0.5})
    b = _series(identity, {**ok, 10: 0.6})
    clean = _series(identity, {**ok, 10: 0.2})
    result = gamma_metrics_paired(ab, a, b, clean, round_index=10)
    assert result["gamma_probability"] == pytest.approx(0.0, abs=1e-15)
    expected_logit = (math.log(0.9 / 0.1) - math.log(0.5 / 0.5) -
                      math.log(0.6 / 0.4) + math.log(0.2 / 0.8))
    assert result["gamma_logit"] == pytest.approx(expected_logit)
    assert result["inputs"] == {"ab": 0.9, "a": 0.5, "b": 0.6, "clean": 0.2}


def test_gamma_clips_only_the_logit_scale():
    identity = _identity()
    ok = {r: 0.25 for r in SELECTED_ROUNDS}
    ab = _series(identity, {**ok, 5: 1.0})
    a = _series(identity, {**ok, 5: 0.0})
    b = _series(identity, {**ok, 5: 0.25})
    clean = _series(identity, {**ok, 5: 0.0})
    result = gamma_metrics_paired(ab, a, b, clean, round_index=5, epsilon=1e-6)
    # Probability scale is unclipped: 1.0 - 0.0 - 0.25 + 0.0.
    assert result["gamma_probability"] == pytest.approx(0.75)
    assert result["probability_scale_unclipped"] is True
    # Logit scale uses epsilon for the zero and one inputs.
    logit = lambda x: math.log(min(max(x, 1e-6), 1 - 1e-6) /
                               (1 - min(max(x, 1e-6), 1 - 1e-6)))
    expected = logit(1.0) - logit(0.0) - logit(0.25) + logit(0.0)
    assert result["gamma_logit"] == pytest.approx(expected)
    assert result["clipping_policy"] == "epsilon_clipping_on_logit_only"


def test_gamma_rejects_mismatched_branch_or_identity():
    identity = _identity()
    ok = {r: 0.5 for r in SELECTED_ROUNDS}
    ab = _series(identity, ok)
    a = _series(identity, ok)
    b = _series(_identity(seed=99), ok)
    clean = _series(identity, ok)
    with pytest.raises(ConfigurationError, match="not strictly paired"):
        gamma_metrics_paired(ab, a, b, clean, round_index=5)
    with pytest.raises(ConfigurationError, match="not strictly paired"):
        gamma_metrics_paired(ab, _series(_identity(seed=2026), ok), a, clean, round_index=5)
    out_of_range = _series(identity, {**ok, 5: 1.5})
    with pytest.raises(ConfigurationError, match="\\[0, 1\\]"):
        gamma_metrics_paired(out_of_range, a, a, clean, round_index=5)
    with pytest.raises(ConfigurationError, match="one metric"):
        gamma_metrics_paired(ab, _series(identity, ok, name="nll"), a, clean, round_index=5)


# ---------------------------------------------------------------------------
# Repeated clean trajectory noise
# ---------------------------------------------------------------------------


def test_adapter_difference_metrics_rejects_mismatched_shapes_and_names():
    with pytest.raises(ConfigurationError, match="parameter names differ"):
        adapter_difference_metrics({"w": np.zeros(2)}, {"v": np.zeros(2)})
    with pytest.raises(ConfigurationError, match="shape mismatch"):
        adapter_difference_metrics({"w": np.zeros(2)}, {"w": np.zeros(3)})


def test_adapter_difference_metrics_exact_arithmetic():
    reference = {"w": np.array([[1.0, 2.0], [3.0, 4.0]]),
                 "v": np.array([0.0, 0.0])}
    repeat = {"w": np.array([[1.0, 2.0], [5.0, 4.0]]),
              "v": np.array([0.0, 0.0])}
    result = adapter_difference_metrics(reference, repeat)
    assert result["l2"] == pytest.approx(2.0)
    assert result["rms"] == pytest.approx(math.sqrt(4.0 / 6.0))
    assert result["max_abs"] == pytest.approx(2.0)
    expected_cos = 36.0 / math.sqrt(30.0 * 46.0)
    assert result["cosine"] == pytest.approx(expected_cos)
    assert result["cosine_defined"] is True
    assert result["parameter_count"] == 6


def test_repeated_clean_noise_requires_identical_config_ids():
    adapters = {0: {"w": np.array([1.0, 2.0])}, 5: {"w": np.array([1.0, 3.0])}}
    repeats = {0: {"w": np.array([1.5, 2.0])}, 5: {"w": np.array([1.0, 2.0])}}
    metrics = {0: {"asr": 0.0, "nll": 2.0}, 5: {"asr": 0.01, "nll": 1.95}}
    repeat_metrics = {0: {"asr": 0.02, "nll": 2.10}, 5: {"asr": 0.0, "nll": 1.90}}
    result = repeated_clean_noise(adapters, repeats, reference_config_id="clean-s17-v1",
                                  repeat_config_id="clean-s17-v1",
                                  reference_metrics=metrics, repeat_metrics=repeat_metrics)
    assert result["same_config_id"] is True
    assert result["config_id"] == "clean-s17-v1"
    assert result["adapter_noise_by_round"][0]["l2"] == pytest.approx(0.5)
    assert result["adapter_noise_by_round"][0]["rms"] == pytest.approx(0.5 / math.sqrt(2.0))
    assert result["adapter_noise_by_round"][0]["max_abs"] == pytest.approx(0.5)
    assert result["metric_noise_by_round"][0]["asr"]["difference"] == pytest.approx(0.02)
    assert result["metric_noise_by_round"][0]["nll"]["difference"] == pytest.approx(0.10)
    assert result["metric_noise_summary"]["asr"]["max_abs_difference"] == pytest.approx(0.02)
    assert result["metric_noise_summary"]["nll"]["max_abs_difference"] == pytest.approx(0.10)
    with pytest.raises(ConfigurationError, match="same configuration IDs"):
        repeated_clean_noise(adapters, repeats, reference_config_id="clean-s17-v1",
                             repeat_config_id="clean-s42-v1",
                             reference_metrics=metrics, repeat_metrics=repeat_metrics)


# ---------------------------------------------------------------------------
# Calibrated BSNR
# ---------------------------------------------------------------------------


def _lora_pair(rng, rank=2, in_features=3, out_features=2):
    stem = "layers.0.self_attn.q_proj"
    global_state = {f"{stem}.lora_A.weight": rng.standard_normal((rank, in_features)),
                    f"{stem}.lora_B.weight": rng.standard_normal((out_features, rank))}
    client = {key: value + rng.standard_normal(value.shape) * 0.1
              for key, value in global_state.items()}
    return client, global_state, stem


def test_effective_delta_matches_explicit_dense_formula():
    rng = np.random.default_rng(7)
    client, global_state, stem = _lora_pair(rng)
    scaling = 2.0
    delta = effective_lora_delta(client, global_state, scaling=scaling)
    expected = scaling * (
        client[f"{stem}.lora_B.weight"] @ client[f"{stem}.lora_A.weight"] -
        global_state[f"{stem}.lora_B.weight"] @ global_state[f"{stem}.lora_A.weight"])
    np.testing.assert_allclose(delta[stem], expected, rtol=1e-12, atol=1e-12)


def test_bsnr_uses_calibrated_direction_and_measured_noise_floor():
    rng = np.random.default_rng(11)
    client, global_state, _ = _lora_pair(rng)
    delta = effective_lora_delta(client, global_state, scaling=2.0)
    flat, _ = flatten_effective_delta(delta)
    direction = flat / np.linalg.norm(flat)  # separately calibrated reference direction
    noise_floor = 0.25
    result = bsnr_from_reference_direction(
        client, global_state, scaling=2.0, reference_direction=direction,
        clean_noise_floor=noise_floor,
        calibration_trajectory_ids=["calib-seed-001", "calib-seed-002"],
        evaluation_trajectory_ids=["test-seed-017"])
    assert result["projection"] == pytest.approx(np.linalg.norm(flat))
    assert result["bsnr"] == pytest.approx(np.linalg.norm(flat) / noise_floor)
    assert result["bsnr_unit_direction"] == pytest.approx(np.linalg.norm(flat) / noise_floor)
    assert result["cosine_with_direction"] == pytest.approx(1.0)
    assert result["calibration_isolation"] is True
    assert result["calibration_only"] is True
    # Scaling the reference direction by a constant changes the raw projection, while
    # the unit-direction BSNR stays fixed; neither may be confused with the noise floor.
    scaled_result = bsnr_from_reference_direction(
        client, global_state, scaling=2.0, reference_direction=direction * 3.0,
        clean_noise_floor=noise_floor,
        calibration_trajectory_ids=["calib-seed-001", "calib-seed-002"],
        evaluation_trajectory_ids=["test-seed-017"])
    assert scaled_result["bsnr"] == pytest.approx(3.0 * result["bsnr"])
    assert scaled_result["bsnr_unit_direction"] == pytest.approx(result["bsnr_unit_direction"])
    assert scaled_result["clean_noise_floor"] == pytest.approx(noise_floor)


def test_bsnr_rejects_calibration_test_overlap_and_bad_floor():
    rng = np.random.default_rng(13)
    client, global_state, _ = _lora_pair(rng)
    flat, _ = flatten_effective_delta(effective_lora_delta(client, global_state, scaling=2.0))
    with pytest.raises(ConfigurationError, match="disjoint"):
        bsnr_from_reference_direction(
            client, global_state, scaling=2.0, reference_direction=flat,
            clean_noise_floor=1.0,
            calibration_trajectory_ids=["shared-trajectory"],
            evaluation_trajectory_ids=["shared-trajectory"])
    for bad_floor in (0.0, -1.0):
        with pytest.raises(ConfigurationError, match="positive finite"):
            bsnr_from_reference_direction(
                client, global_state, scaling=2.0, reference_direction=flat,
                clean_noise_floor=bad_floor,
                calibration_trajectory_ids=["calib"],
                evaluation_trajectory_ids=["test"])
    with pytest.raises(ConfigurationError, match="dimension"):
        bsnr_from_reference_direction(
            client, global_state, scaling=2.0, reference_direction=flat[:-1],
            clean_noise_floor=1.0,
            calibration_trajectory_ids=["calib"], evaluation_trajectory_ids=["test"])
    with pytest.raises(ConfigurationError, match="non-zero"):
        bsnr_from_reference_direction(
            client, global_state, scaling=2.0, reference_direction=np.zeros_like(flat),
            clean_noise_floor=1.0,
            calibration_trajectory_ids=["calib"], evaluation_trajectory_ids=["test"])
    with pytest.raises(ConfigurationError, match="Effective LoRA delta is zero"):
        bsnr_from_reference_direction(
            global_state, global_state, scaling=2.0, reference_direction=flat,
            clean_noise_floor=1.0,
            calibration_trajectory_ids=["calib"], evaluation_trajectory_ids=["test"])


def test_existing_metrics_bsnr_remains_backwards_compatible():
    attacked = np.array([2.0, 0.0])
    clean = np.array([0.0, 0.0])
    result = legacy_bsnr(attacked, clean, noise_floor=0.5)
    assert result["bsnr_l2"] == pytest.approx(4.0)
    assert result["calibration_only"] is True
