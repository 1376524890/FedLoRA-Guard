"""Pure-computation tests for the unified paper table/figure generator.

Every JSON fixture here is a *synthetic schema fixture* written under pytest's
``tmp_path``.  None of it is a real experiment, and no test asserts a scientific
claim about FedLoRA-Guard.  The tests only check evidence plumbing, explicit
status semantics, provenance/CI fields, refusal to overwrite, and figure text
fallbacks.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import paper_outputs as outputs


def write_json(root: Path, relative: str, payload: object) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def suite_artifact(*, paired: bool = False, complete: bool = True) -> dict:
    results = [
        {"round": round_index,
         "triggered_asr": {"successes": successes, "total": 256},
         "negative_controls": {"clean": {"successes": 2, "total": 256}},
         "clean_nll": {"nll": 1.5, "tokens": 10, "records": 1000}}
        for round_index, successes in ((0, 3), (5, 40), (10, 90), (30, 200))
    ]
    artifact = {
        "run_id": "suite_fixture_s17",
        "seed": 17,
        "scope": "ENGINEERING_SAVED_ADAPTER_EVALUATION_NOT_FORMAL_MATRIX",
        "complete_protocol_probe_count": complete,
        "complete_protocol_nll_count": complete,
        "private_training_data_used": False,
        "results": results,
        "changes_from_round_0": [],
        "paired_formal_counterfactual": None,
        "note": "synthetic schema fixture, not an experiment",
    }
    if paired:
        artifact["paired_formal_counterfactual"] = {"pairs": [
            {"seed": 17, "round": 30,
             "attacked_run_id": "attacked_s17", "clean_run_id": "clean_s17",
             "attacked_asr": {"successes": 220, "total": 256},
             "clean_asr_estimate": 0.02, "clean_prompt_misfire": 0.01,
             "clean_control_asr": 0.0,
             "attacked_accuracy": 0.90, "clean_accuracy": 0.92},
        ]}
    return artifact


def paired_artifact(successes: int = 220, accuracy: float = 0.90) -> dict:
    return {
        "schema": outputs.PAIRED_CLEAN_SCHEMA,
        "run_id": "paired_fixture",
        "seeds": [17, 42, 2026],
        "config_sha256": "a" * 64,
        "pairs": [
            {"seed": seed, "round": 30,
             "attacked_run_id": f"attacked_s{seed}", "clean_run_id": f"clean_s{seed}",
             "attacked_asr": {"successes": successes, "total": 256},
             "clean_prompt_misfire": 0.01, "clean_control_asr": 0.0,
             "clean_asr_estimate": 0.02,
             "attacked_accuracy": accuracy, "clean_accuracy": 0.92,
             "attacked_config_sha": "a" * 64, "clean_config_sha": "a" * 64,
             "scope": "FORMAL_PAIRED_TRAJECTORY"}
            for seed in (17, 42, 2026)
        ],
    }


def table_by_id(index: dict, table_id: str) -> dict:
    return json.loads((Path(index["_output_dir"]) / "tables" / f"{table_id}.json").read_text())


def generate(input_dir: Path, output_dir: Path) -> dict:
    index = outputs.generate_paper_outputs(input_dir, output_dir)
    index["_output_dir"] = str(output_dir)
    return index


def test_empty_evidence_emits_all_tables_figures_as_not_run(tmp_path):
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    index = generate(evidence_dir, tmp_path / "out")
    out = Path(index["_output_dir"])

    assert index["complete_all"] is False
    assert index["formal_claim"] is False
    assert index["coverage"]["tables_total"] == 16
    assert index["coverage"]["tables_complete"] == 0
    assert index["coverage"]["table_status_counts"] == {"NOT_RUN": 16}
    assert index["coverage"]["figures_total"] == 14
    assert index["coverage"]["figures_rendered"] == 0
    assert index["coverage"]["figure_status_counts"] == {"NOT_RUN": 14}

    assert len(list((out / "tables").glob("*.json"))) == 16
    assert len(list((out / "tables").glob("*.md"))) == 16
    assert len(list((out / "figures").glob("*.png"))) == 14
    assert len(list((out / "figures").glob("*.metadata.json"))) == 14

    for table in index["tables"]:
        assert table["status"] == "NOT_RUN"
        assert table["row_count"] == 0
        assert table["missing_inputs"]
        document = table_by_id(index, table["table_id"])
        assert document["status"] == "NOT_RUN"
        assert document["rows"] == []
        assert document["complete"] is False
        assert document["formal_claim"] is False
        assert document["evidence"] == []
    markdown = (out / "tables" / "table_03_badnet_effectiveness.md").read_text()
    assert "Status: **NOT_RUN**" in markdown
    assert "not measured" in markdown

    for figure in index["figures"]:
        assert figure["status"] == "NOT_RUN"
        assert figure["plotted"] is False
        assert figure["needed_input"]
        metadata = json.loads((out / figure["metadata"]).read_text())
        assert metadata["plotted"] is False
        assert metadata["needed_input"] == figure["needed_input"]
        assert metadata["seed_count"] is None
        assert metadata["config_sha"] is None
        assert metadata["confidence_interval"]
        assert metadata["formal_claim"] is False
        assert (out / figure["png"]).stat().st_size > 0


def test_output_directory_refuses_existing_files_and_rerun(tmp_path):
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="non-empty output directory"):
        outputs.generate_paper_outputs(evidence_dir, occupied)
    assert (occupied / "keep.txt").read_text() == "existing"

    output_dir = tmp_path / "out"
    outputs.generate_paper_outputs(evidence_dir, output_dir)
    with pytest.raises(FileExistsError, match="non-empty output directory"):
        outputs.generate_paper_outputs(evidence_dir, output_dir)
    assert json.loads((output_dir / "index.json").read_text())["complete_all"] is False


def test_output_directory_is_file_is_rejected(tmp_path):
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        outputs.generate_paper_outputs(evidence_dir, blocker)


def test_table3_is_unpaired_without_a_paired_clean_counterfactual(tmp_path):
    evidence_dir = tmp_path / "evidence"
    write_json(evidence_dir, "runs/suite.json", suite_artifact(complete=True))
    index = generate(evidence_dir, tmp_path / "out")
    table = table_by_id(index, "table_03_badnet_effectiveness")
    assert table["status"] == "UNPAIRED"
    assert table["h2_pass"] is False
    assert table["rows"] == []
    assert table["complete"] is False
    assert "paired clean counterfactual" in table["missing_inputs"][0]
    assert table["evidence"][0]["sha256"]
    entry = next(item for item in index["tables"]
                 if item["table_id"] == "table_03_badnet_effectiveness")
    assert entry["status"] == "UNPAIRED"
    assert entry["h2_pass"] is False


def test_table3_is_incomplete_when_protocol_counts_are_partial(tmp_path):
    evidence_dir = tmp_path / "evidence"
    write_json(evidence_dir, "runs/suite.json", suite_artifact(complete=False))
    index = generate(evidence_dir, tmp_path / "out")
    table = table_by_id(index, "table_03_badnet_effectiveness")
    assert table["status"] == "INCOMPLETE"
    assert table["h2_pass"] is False
    assert any("256-probe" in item for item in table["missing_inputs"])


def test_table3_paired_evidence_populates_rows_and_declares_h2(tmp_path):
    evidence_dir = tmp_path / "evidence"
    write_json(evidence_dir, "pairs/paired_clean.json", paired_artifact())
    index = generate(evidence_dir, tmp_path / "out")
    table = table_by_id(index, "table_03_badnet_effectiveness")
    assert table["status"] == "COMPLETE"
    assert table["complete"] is True
    assert table["h2_pass"] is True
    assert table["row_count"] == 3
    for row in table["rows"]:
        assert row["paired"] is True
        assert row["h2_seed_pass"] is True
        assert row["asr_total"] == 256
        assert row["asr_successes"] == 220
        assert row["asr_lower"] <= row["asr_estimate"] <= row["asr_upper"]
        assert row["config_sha"] == "a" * 64
        assert row["seed_count"] == 3
        assert row["attacked_run_id"].startswith("attacked_s")
        assert row["clean_run_id"].startswith("clean_s")
        assert row["source_path"] == "pairs/paired_clean.json"
        assert row["normal_task_drop"] == pytest.approx(0.02)


def test_table3_h2_uses_declared_measurement_round_not_repeated_seed_rounds(tmp_path):
    evidence_dir = tmp_path / "evidence"
    artifact = paired_artifact()
    artifact["h2_measurement_round"] = 30
    artifact["pairs"].insert(0, {
        "seed": 17, "round": 0, "attacked_run_id": "attacked_s17",
        "clean_run_id": "clean_s17", "attacked_asr": {"successes": 0, "total": 256},
        "clean_prompt_misfire": 0.0, "clean_control_asr": 0.0,
        "scope": "FORMAL_PAIRED_TRAJECTORY"})
    write_json(evidence_dir, "pairs/paired_clean.json", artifact)
    index = generate(evidence_dir, tmp_path / "out")
    table = table_by_id(index, "table_03_badnet_effectiveness")
    assert table["row_count"] == 3
    assert {row["round"] for row in table["rows"]} == {30}
    assert all(row["normal_task_drop"] == pytest.approx(0.02)
               for row in table["rows"])


def test_table3_h2_stays_false_when_measured_metrics_miss_thresholds(tmp_path):
    evidence_dir = tmp_path / "evidence"
    write_json(evidence_dir, "pairs/paired_clean.json", paired_artifact(successes=100))
    index = generate(evidence_dir, tmp_path / "out")
    table = table_by_id(index, "table_03_badnet_effectiveness")
    assert table["status"] == "COMPLETE"
    assert table["h2_pass"] is False
    assert all(row["h2_seed_pass"] is False for row in table["rows"])


def test_table3_h2_requires_measured_accuracy_drop(tmp_path):
    evidence_dir = tmp_path / "evidence"
    artifact = paired_artifact()
    for pair in artifact["pairs"]:
        pair.pop("attacked_accuracy")
        pair.pop("clean_accuracy")
    write_json(evidence_dir, "pairs/paired_clean.json", artifact)
    index = generate(evidence_dir, tmp_path / "out")
    table = table_by_id(index, "table_03_badnet_effectiveness")
    assert table["h2_pass"] is False
    assert all(row["normal_task_drop"] is None for row in table["rows"])


def test_table3_rejects_mismatched_paired_configuration_hashes(tmp_path):
    evidence_dir = tmp_path / "evidence"
    artifact = paired_artifact()
    for pair in artifact["pairs"]:
        pair["clean_config_sha"] = "b" * 64
    write_json(evidence_dir, "pairs/paired_clean.json", artifact)
    index = generate(evidence_dir, tmp_path / "out")
    table = table_by_id(index, "table_03_badnet_effectiveness")
    assert table["status"] == "COMPLETE"
    assert table["h2_pass"] is False
    assert all(row["paired"] is False for row in table["rows"])


def test_throughput_table_distinguishes_physical_and_logical_gpu_ids(tmp_path):
    evidence_dir = tmp_path / "evidence"
    write_json(evidence_dir, "bench/single_visible_gpu.json", {"rows": [{
        "physical_gpu_id": 2, "gpu_id": 0, "dtype": "float32",
        "mean_step_seconds": 1.5, "peak_allocated_bytes": 123,
    }]})
    index = generate(evidence_dir, tmp_path / "out")
    row = table_by_id(index, "table_01_gpu_fp32_fp16")["rows"][0]
    assert row["gpu_id"] == 2
    assert row["logical_gpu_id"] == 0


def test_throughput_table_uses_real_rows_and_provenance(tmp_path):
    evidence_dir = tmp_path / "evidence"
    rows = [
        {"gpu_id": gpu, "gpu_name": f"GPU-{gpu}", "dtype": dtype,
         "optimizer_steps": 10, "micro_batch": 1, "gradient_accumulation": 2,
         "load_seconds": 5.0, "step_seconds": [0.5, 0.6],
         "mean_step_seconds": 0.55, "peak_allocated_bytes": 123,
         "peak_reserved_bytes": 456, "adapter_parameters": 789,
         "sample_id": "p-1", "public_probe_only": True,
         "model_revision": "rev-1", "scope": "MEASURED_REAL_THROUGHPUT"}
        for gpu in (0, 1, 2) for dtype in ("float32", "float16")
    ]
    write_json(evidence_dir, "bench/precision.json",
               {"rows": rows, "protocol": "fixture", "not_attack_training": True})
    index = generate(evidence_dir, tmp_path / "out")
    table = table_by_id(index, "table_01_gpu_fp32_fp16")
    assert table["status"] == "COMPLETE"
    assert table["row_count"] == 6
    assert {row["gpu_id"] for row in table["rows"]} == {0, 1, 2}
    assert {row["dtype"] for row in table["rows"]} == {"float32", "float16"}
    for row in table["rows"]:
        assert row["mean_step_seconds"] == 0.55
        assert row["peak_allocated_bytes"] == 123
        assert row["source_path"] == "bench/precision.json"
        assert len(row["source_sha256"]) == 64
        assert row["seed_count"] is None


def test_nonformal_scope_is_incomplete_even_when_all_fields_are_present(tmp_path):
    evidence_dir = tmp_path / "evidence"
    write_json(evidence_dir, "bench/precision.json", {"rows": [
        {"gpu_id": 0, "dtype": "float16", "mean_step_seconds": 0.5,
         "peak_allocated_bytes": 100, "run_id": "eng_s17",
         "scope": "ENGINEERING_PRETEST_NOT_FORMAL"},
    ]})
    index = generate(evidence_dir, tmp_path / "out")
    table = table_by_id(index, "table_01_gpu_fp32_fp16")
    assert table["status"] == "INCOMPLETE"
    assert table["complete"] is False
    assert table["row_count"] == 1
    assert any("engineering/precheck" in note for note in table["notes"])


def test_repeated_clean_noise_table_expands_metric_and_adapter_rows(tmp_path):
    evidence_dir = tmp_path / "evidence"
    write_json(evidence_dir, "noise/repeated_clean.json", {
        "config_id": "cfg-1",
        "same_config_id": True,
        "rounds": [0, 5],
        "adapter_noise_by_round": {
            "0": {"l2": 0.0, "rms": 0.0, "max_abs": 0.0, "cosine": 1.0},
            "5": {"l2": 0.2, "rms": 0.1, "max_abs": 0.05, "cosine": 0.99},
        },
        "metric_noise_by_round": {
            "0": {"asr": {"reference": 0.0, "repeat": 0.0, "difference": 0.0,
                          "absolute_difference": 0.0},
                  "nll": {"reference": 1.5, "repeat": 1.51, "difference": 0.01,
                          "absolute_difference": 0.01}},
            "5": {"asr": {"reference": 0.0, "repeat": 0.004, "difference": 0.004,
                          "absolute_difference": 0.004},
                  "nll": {"reference": 1.5, "repeat": 1.53, "difference": 0.03,
                          "absolute_difference": 0.03}},
        },
        "metric_noise_summary": {"asr": {"max_abs_difference": 0.004, "mean_difference": 0.002},
                                 "nll": {"max_abs_difference": 0.03, "mean_difference": 0.02}},
        "scope": "repeated_clean_trajectory_noise_not_attack_effect",
    })
    index = generate(evidence_dir, tmp_path / "out")
    table = table_by_id(index, "table_02_repeated_clean_noise")
    assert table["status"] == "COMPLETE"
    assert table["row_count"] == 4
    assert {row["metric"] for row in table["rows"]} == {"asr", "nll"}
    round_five = next(row for row in table["rows"]
                      if row["round"] == 5 and row["metric"] == "nll")
    assert round_five["reference"] == 1.5
    assert round_five["repeat"] == 1.53
    assert round_five["adapter_l2"] == 0.2
    assert round_five["config_id"] == "cfg-1"


def test_gamma_and_detection_tables_compute_derived_metrics(tmp_path):
    evidence_dir = tmp_path / "evidence"
    write_json(evidence_dir, "gamma/gamma.json", {"rows": [
        {"condition": "D25", "metric": "asr", "gamma_probability": 0.12,
         "gamma_logit": 0.47, "logit_epsilon": 1e-6, "round": 30,
         "run_id": "gamma_fixture", "config_sha256": "c" * 64,
         "scope": "FORMAL_PAIRED_TRAJECTORY"},
    ]})
    write_json(evidence_dir, "detect/safe_fedllm.json", {
        "method": "Safe-FedLLM", "roc_auc": 0.91, "pr_auc": 0.88,
        "confusion": {"tn": 90, "fp": 4, "fn": 2, "tp": 40},
        "fpr": 4 / 94, "fnr": 2 / 42, "threshold": 0.8,
        "flagged_clients": ["client_02"],
        "run_id": "detector_fixture", "scope": "FORMAL_DETECTION",
    })
    index = generate(evidence_dir, tmp_path / "out")
    gamma = table_by_id(index, "table_06_gamma")
    assert gamma["status"] == "COMPLETE"
    assert gamma["rows"][0]["gamma_probability"] == 0.12
    assert gamma["rows"][0]["config_sha"] == "c" * 64
    detection = table_by_id(index, "table_10_client_detection")
    assert detection["status"] == "COMPLETE"
    row = detection["rows"][0]
    assert row["roc_auc"] == 0.91
    assert row["precision"] == pytest.approx(40 / 44)
    assert row["recall"] == pytest.approx(40 / 42)
    assert row["f1"] == pytest.approx(2 * (40 / 44) * (40 / 42) / ((40 / 44) + (40 / 42)))
    assert (row["tn"], row["fp"], row["fn"], row["tp"]) == (90, 4, 2, 40)


def test_online_api_and_raw_lora_preflight_tables_keep_nonclaims(tmp_path):
    evidence_dir = tmp_path / "evidence"
    write_json(evidence_dir, "online/api_costs.json", {"rows": [
        {"provider": "kimi", "requested_model": "kimi-k3", "exact_match": "kimi-k3",
         "classification_authorized": True, "detections": ["client_02"],
         "successes": 8, "total": 10, "estimated_cost_usd": 0.42,
         "actual_cost_usd": 0.40, "run_id": "online_fixture",
         "scope": "FORMAL_ONLINE_API"},
    ]})
    write_json(evidence_dir, "online/preflight.json", {
        "client_id": "client_02", "round": 10,
        "adapter": {"tensors": 24, "parameters": 1024,
                    "tokens_actual_qwen_tokenizer": 3000},
        "received_global": {}, "public_probe_tokens": 200,
        "raw_input_tokens_before_instruction": 5000, "max_context_tokens": 8192,
        "fits_declared_context": True, "parameters_edited": False,
        "private_training_data_sent": False,
        "status": "PRECHECK_ONLY_NO_API_CLASSIFICATION",
        "run_id": "preflight_fixture",
    })
    index = generate(evidence_dir, tmp_path / "out")
    api = table_by_id(index, "table_13_online_apis")
    assert api["status"] == "COMPLETE"
    assert api["rows"][0]["actual_cost_usd"] == 0.40
    raw = table_by_id(index, "table_16_raw_lora_online")
    assert raw["status"] == "COMPLETE"
    row = raw["rows"][0]
    assert row["status"] == "PRECHECK_ONLY_NO_API_CLASSIFICATION"
    assert row["classification_performed"] is False
    assert row["tensors"] == 24
    assert row["raw_tokens"] == 5000
    assert row["fits_declared_context"] is True


def test_figure1_plots_only_real_round_values(tmp_path):
    evidence_dir = tmp_path / "evidence"
    write_json(evidence_dir, "runs/suite.json", suite_artifact())
    index = generate(evidence_dir, tmp_path / "out")
    figure = next(item for item in index["figures"]
                  if item["figure_id"] == "figure_01_asr_by_round")
    assert figure["status"] == "RENDERED"
    metadata = json.loads((tmp_path / "out" / figure["metadata"]).read_text())
    assert metadata["seed_count"] == 1
    assert metadata["seeds"] == [17]
    assert metadata["confidence_interval"]
    assert metadata["evidence_sha256"]
    assert metadata["formal_evidence"] is False
    assert (tmp_path / "out" / figure["png"]).stat().st_size > 0
    # Table 3 remains UNPAIRED even though figure 1 has numeric ASR points.
    table3 = table_by_id(index, "table_03_badnet_effectiveness")
    assert table3["status"] == "UNPAIRED"
    assert table3["h2_pass"] is False


def test_figure_text_fallback_when_numeric_values_are_absent(tmp_path):
    evidence_dir = tmp_path / "evidence"
    write_json(evidence_dir, "runs/suite.json", {
        "run_id": "suite_missing_asr", "seed": 17,
        "scope": "ENGINEERING_SAVED_ADAPTER_EVALUATION_NOT_FORMAL_MATRIX",
        "complete_protocol_probe_count": False,
        "results": [{"round": 5, "triggered_asr": {}, "clean_nll": {}}],
    })
    index = generate(evidence_dir, tmp_path / "out")
    figure = next(item for item in index["figures"]
                  if item["figure_id"] == "figure_01_asr_by_round")
    assert figure["status"] == "NOT_RUN"
    assert figure["plotted"] is False
    metadata = json.loads((tmp_path / "out" / figure["metadata"]).read_text())
    assert metadata["missing_inputs"]
    assert metadata["seed_count"] is None
    assert metadata["evidence"]  # the artifact was seen but had no plottable value


def test_figure2_needs_paired_clean_not_round_zero_deltas(tmp_path):
    evidence_dir = tmp_path / "evidence"
    write_json(evidence_dir, "runs/suite.json", suite_artifact())
    index = generate(evidence_dir, tmp_path / "out")
    figure = next(item for item in index["figures"]
                  if item["figure_id"] == "figure_02_cumulative_impact")
    assert figure["status"] == "NOT_RUN"
    metadata = json.loads((tmp_path / "out" / figure["metadata"]).read_text())
    assert "paired" in metadata["needed_input"]

    paired_dir = tmp_path / "paired"
    write_json(paired_dir, "pairs/paired_clean.json", paired_artifact())
    second = generate(paired_dir, tmp_path / "out_paired")
    paired_figure = next(item for item in second["figures"]
                         if item["figure_id"] == "figure_02_cumulative_impact")
    assert paired_figure["status"] == "RENDERED"
    metadata = json.loads((tmp_path / "out_paired" / paired_figure["metadata"]).read_text())
    assert "discrete cumulative sum" in metadata["definition"]
    assert metadata["seed_count"] == 3


def test_scatter_bar_and_filter_figures_render_from_numeric_evidence(tmp_path):
    evidence_dir = tmp_path / "evidence"
    write_json(evidence_dir, "bsnr/bsnr.json", {"rows": [
        {"client_id": f"client_{index:02d}", "bsnr": 0.5 + index, "asr_estimate": 0.2 + 0.1 * index,
         "clean_noise_floor": 0.01, "round": 30, "run_id": "bsnr_fixture",
         "scope": "FORMAL_BSNR"}
        for index in range(3)
    ]})
    write_json(evidence_dir, "defense/filters.json", {"rows": [
        {"method": method, "asr_estimate": value, "suspicious_clients": ["client_02"],
         "accepted_clients": ["client_00", "client_01"], "run_id": "filter_fixture",
         "scope": "FORMAL_DEFENSE"}
        for method, value in (("none", 0.82), ("oracle", 0.01), ("cbdf_lora", 0.04))
    ]})
    index = generate(evidence_dir, tmp_path / "out")
    for figure_id in ("figure_07_bsnr_vs_asr", "figure_14_filter_suppression"):
        figure = next(item for item in index["figures"] if item["figure_id"] == figure_id)
        assert figure["status"] == "RENDERED"
        assert (tmp_path / "out" / figure["png"]).stat().st_size > 0
    # No aggregation evidence was supplied: that figure must stay NOT_RUN.
    aggregation = next(item for item in index["figures"]
                       if item["figure_id"] == "figure_13_aggregation_comparison")
    assert aggregation["status"] == "NOT_RUN"
    metadata = json.loads((tmp_path / "out" / figure["metadata"]).read_text())
    assert metadata["seed_count"] is None
    assert metadata["config_sha"] is None
    assert metadata["evidence_sha256"]


def test_heatmap_and_network_figures_use_real_matrices_and_edges(tmp_path):
    evidence_dir = tmp_path / "evidence"
    write_json(evidence_dir, "detect/similarity.json", {
        "clients": ["client_00", "client_01", "client_02"],
        "matrix": [[1.0, 0.5, 0.1], [0.5, 1.0, 0.2], [0.1, 0.2, 1.0]],
        "run_id": "detect_fixture",
    })
    write_json(evidence_dir, "detect/edges.json", {"method": "CBDF-LoRA", "pairs": [
        {"client_a": "client_00", "client_b": "client_01", "statistic": 0.7,
         "p_value_max_corrected": 0.04},
        {"client_a": "client_01", "client_b": "client_02", "statistic": 0.2,
         "p_value_max_corrected": 1.0},
    ]})
    index = generate(evidence_dir, tmp_path / "out")
    for figure_id in ("figure_08_residual_similarity_heatmap", "figure_09_relation_network"):
        figure = next(item for item in index["figures"] if item["figure_id"] == figure_id)
        assert figure["status"] == "RENDERED"
        assert (tmp_path / "out" / figure["png"]).stat().st_size > 0


def test_index_records_unreadable_evidence_errors(tmp_path):
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    (evidence_dir / "broken.json").write_text("{not json", encoding="utf-8")
    write_json(evidence_dir, "runs/suite.json", suite_artifact())
    index = generate(evidence_dir, tmp_path / "out")
    assert index["complete_all"] is False
    assert index["evidence_errors"][0]["path"] == "broken.json"
    table3 = table_by_id(index, "table_03_badnet_effectiveness")
    assert table3["status"] == "UNPAIRED"


def test_helpers_wilson_and_seed_interval_are_pure():
    interval = outputs._wilson(200, 256)
    assert interval["successes"] == 200 and interval["total"] == 256
    expected = 200 / 256
    assert interval["estimate"] == pytest.approx(expected)
    assert interval["lower"] < expected < interval["upper"]
    assert 0.0 <= interval["lower"] and interval["upper"] <= 1.0
    assert outputs._wilson(3, 2) is None
    assert outputs._wilson(1, 0) is None
    assert outputs._wilson(-1, 2) is None

    single = outputs._mean_interval([0.9])
    assert single["interval_supported"] is False
    assert single["lower"] is None and single["upper"] is None
    multiple = outputs._mean_interval([0.80, 0.86, 0.90])
    assert multiple["n"] == 3
    assert multiple["mean"] == pytest.approx((0.80 + 0.86 + 0.90) / 3)
    assert multiple["lower"] < multiple["mean"] < multiple["upper"]
    assert multiple["interval_supported"] is True
    assert outputs._mean_interval([])["n"] == 0


def test_generator_writes_only_the_declared_output_tree(tmp_path):
    evidence_dir = tmp_path / "evidence"
    write_json(evidence_dir, "runs/suite.json", suite_artifact())
    before = {path for path in tmp_path.rglob("*") if path.is_file()}
    index = generate(evidence_dir, tmp_path / "out")
    after = {path for path in tmp_path.rglob("*") if path.is_file()}
    created = after - before

    expected = {tmp_path / "out" / "index.json"}
    expected |= {tmp_path / "out" / "tables" / f"{table['table_id']}.json"
                 for table in index["tables"]}
    expected |= {tmp_path / "out" / "tables" / f"{table['table_id']}.md"
                 for table in index["tables"]}
    expected |= {tmp_path / "out" / figure["png"] for figure in index["figures"]}
    expected |= {tmp_path / "out" / figure["metadata"] for figure in index["figures"]}
    assert created == expected
    assert len(created) == 1 + 32 + 28


def test_real_safe_summary_is_not_replaced_by_client_scores(tmp_path):
    evidence_dir = tmp_path / "evidence"
    write_json(evidence_dir, "detect/summary.json", {
        "method": "Safe-FedLLM", "classifier": "official_bce",
        "scores": [{"client_id": "client_02", "round": 1, "probability": 0.99}],
        "roc_auc": 1.0, "pr_auc": 1.0, "threshold": 0.8,
        "confusion": {"tp": 1, "tn": 1, "fn": 0, "fp": 0},
        "fpr": 0.0, "fnr": 0.0, "flagged_clients": ["client_02"],
        "scope": "REAL_TRAJECTORY_OFFLINE_CALIBRATION_NOT_A_FORMAL_MATRIX_RESULT",
    })
    index = generate(evidence_dir, tmp_path / "out")
    table = json.loads((tmp_path / "out" / "tables/table_10_client_detection.json").read_text())
    assert len(table["rows"]) == 1
    row = table["rows"][0]
    assert row["method"] == "Safe-FedLLM" and row["roc_auc"] == 1.0
    assert row["precision"] == 1.0 and row["recall"] == 1.0
    figure = next(item for item in index["figures"]
                  if item["figure_id"] == "figure_10_detection_prf")
    assert figure["status"] == "RENDERED"


def test_real_cbdf_residual_matrix_renders_heatmap(tmp_path):
    evidence_dir = tmp_path / "evidence"
    write_json(evidence_dir, "detect/residual.json", {
        "kind": "cbdf_lora_residual_matrix", "round": 6,
        "clients": ["client_00", "client_01", "client_02"],
        "matrix": [[None, 0.5, 0.1], [0.5, None, None], [0.1, None, None]],
        "scope": "ENGINEERING_PREAGGREGATION_INTEGRATION_NOT_FORMAL_PERFORMANCE",
    })
    index = generate(evidence_dir, tmp_path / "out")
    figure = next(item for item in index["figures"]
                  if item["figure_id"] == "figure_08_residual_similarity_heatmap")
    assert figure["status"] == "RENDERED"
    metadata = json.loads((tmp_path / "out" / figure["metadata"]).read_text())
    assert metadata["matrix_shape"] == [3, 3]
    assert metadata["evidence_sha256"]
