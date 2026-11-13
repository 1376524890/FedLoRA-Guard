"""Paper table/figure generation from saved JSON evidence only.

This module never trains, never calls a model or API, and never invents a metric.
Every one of the 16 required tables is always written.  A table defaults to
``NOT_RUN`` with explicit empty columns; it becomes ``INCOMPLETE`` when matching
evidence exists but is missing required fields / comes from a non-formal scope,
and ``COMPLETE`` only when the supplied real JSON evidence populates every
declared field.

Evidence contract
-----------------
The input directory is scanned recursively for ``*.json`` files.  An artifact is
recognised by a top-level ``kind`` / ``schema`` / ``artifact_kind`` /
``artifact_type`` string, or by a known structural signature (for example an
``evaluation_suite`` result or a repeated-clean noise report).  Each recognised
artifact is copied into the table with its relative path and SHA-256, and its
``run_id`` / ``scope`` / config hash are carried into row provenance.

Table 3 (BadNet effectiveness) is stricter.  A working-directory
``evaluation_suite`` artifact records ``paired_formal_counterfactual: null`` by
design, so on its own it cannot support the H2 claim.  In that case the table is
``UNPAIRED`` (``INCOMPLETE`` when the protocol counts are also incomplete) and
``h2_pass`` is false.  H2 is evaluated only when either

* the evaluation-suite artifact has a non-null ``paired_formal_counterfactual``, or
* a separate artifact declares ``kind: "paired_clean_comparison"`` (or
  ``schema: "fedlora-paired-clean-v1"``) with a ``pairs`` list.  Each pair must
  name a distinct attacked and clean run, a seed, the attacked per-seed ASR /
  clean-prompt misfire, and the clean counterfactual metrics.  If both sides
  declare a config SHA, they must match.

The H2 thresholds are fixed from the task specification: macro ASR >= 0.80,
per-seed ASR >= 0.70, per-seed successes >= 180, clean-prompt misfire <= 0.10,
and clean-control-trigger ASR <= 0.10.  Failure (or absence) of any check sets
``h2_pass`` to false; this module never promotes a partial result.

Output contract
---------------
``generate_paper_outputs(input_dir, output_dir)`` refuses an existing non-empty
output directory and never overwrites a file.  It writes:

* ``tables/<table_id>.json`` and ``tables/<table_id>.md`` for all 16 tables;
* ``figures/<figure_id>.png`` and ``figures/<figure_id>.metadata.json`` for all
  14 figures (rendered with the non-interactive ``Agg`` backend; when no real
  numeric data exists the figure is a text panel naming the missing evidence);
* ``index.json`` with explicit per-table/per-figure coverage and
  ``complete_all: false``.

Tests use synthetic schema fixtures only; no output from this module is a claim
about a real experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

SCHEMA_VERSION = "fedlora-paper-outputs-index-v1"
TABLE_SCHEMA_VERSION = "fedlora-paper-table-v1"
FIGURE_SCHEMA_VERSION = "fedlora-paper-figure-metadata-v1"
PAIRED_CLEAN_SCHEMA = "fedlora-paired-clean-v1"

# Evidence whose declared scope contains one of these tokens is real but not a
# formal paper result; its table is emitted as INCOMPLETE, never COMPLETE.
NON_FORMAL_SCOPE_TOKENS = (
    "ENGINEERING",
    "SMOKE",
    "PRECHECK",
    "LIMITED_",
)

COMMON_COLUMNS = (
    "run_id",
    "seed",
    "seed_count",
    "config_sha",
    "round",
    "scope",
    "source_path",
)

MISSING_ASR = "per-seed ASR counts/estimate with Wilson interval"


# ---------------------------------------------------------------------------
# Numeric helpers (pure computation, stdlib only)
# ---------------------------------------------------------------------------


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _count(value: Any) -> int | None:
    number = _finite(value)
    if number is None or number < 0 or not float(number).is_integer():
        return None
    return int(number)


def _wilson(successes: Any, total: Any, confidence: float = 0.95) -> dict[str, Any] | None:
    """Wilson score interval from reported binomial counts (no scipy needed)."""
    successes, total = _count(successes), _count(total)
    if successes is None or total is None or total <= 0 or successes > total:
        return None
    if not 0.0 < confidence < 1.0:
        return None
    z = statistics.NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return {"successes": successes, "total": total, "estimate": p,
            "lower": max(0.0, center - radius), "upper": min(1.0, center + radius),
            "confidence": confidence, "method": "wilson_score_interval"}


def _mean_interval(values: Sequence[float], confidence: float = 0.95) -> dict[str, Any]:
    """Mean with a Student-t interval over independent seed effects when scipy exists."""
    data = [value for value in (_finite(item) for item in values) if value is not None]
    if not data:
        return {"n": 0, "mean": None, "lower": None, "upper": None,
                "confidence": confidence, "interval_supported": False,
                "reason": "no finite seed-level values"}
    mean = statistics.fmean(data)
    if len(data) == 1:
        return {"n": 1, "mean": mean, "std": None, "lower": None, "upper": None,
                "confidence": confidence, "interval_supported": False,
                "method": "student_t_interval_over_seed_effects",
                "reason": "n=1: a seed-level confidence interval is undefined"}
    std = statistics.stdev(data)
    standard_error = std / math.sqrt(len(data))
    try:
        from scipy.stats import t as student_t  # local optional dependency
        critical = float(student_t.ppf(0.5 + confidence / 2.0, len(data) - 1))
        method = "student_t_interval_over_seed_effects"
    except Exception:  # pragma: no cover - only when scipy is unavailable
        critical = statistics.NormalDist().inv_cdf(0.5 + confidence / 2.0)
        method = "normal_approximation_over_seed_effects_scipy_unavailable"
    half = critical * standard_error
    return {"n": len(data), "mean": mean, "std": std, "standard_error": standard_error,
            "t_critical": critical, "half_width": half,
            "lower": mean - half, "upper": mean + half,
            "confidence": confidence, "interval_supported": True, "method": method}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Evidence loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Evidence:
    path: str
    sha256: str
    data: Mapping[str, Any]
    kind: str | None

    @property
    def run_id(self) -> str | None:
        value = self.data.get("run_id")
        return value if isinstance(value, str) and value else None

    @property
    def scope(self) -> str | None:
        value = self.data.get("scope")
        return value if isinstance(value, str) and value else None


@dataclass
class EvidenceSet:
    input_dir: Path
    artifacts: list[Evidence] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)

    def of_kind(self, *kinds: str) -> list[Evidence]:
        wanted = {kind.casefold() for kind in kinds}
        return [item for item in self.artifacts
                if item.kind is not None and item.kind.casefold() in wanted]


def _detect_kind(data: Mapping[str, Any]) -> str | None:
    for key in ("kind", "schema", "artifact_kind", "artifact_type"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if "complete_protocol_probe_count" in data and isinstance(data.get("results"), list):
        return "evaluation_suite"
    scope = data.get("scope")
    if isinstance(scope, str):
        if "repeated_clean_trajectory_noise" in scope:
            return "repeated_clean_noise"
        if "ENGINEERING_SAVED_ADAPTER_EVALUATION" in scope:
            return "evaluation_suite"
    if "adapter_noise_by_round" in data and "metric_noise_by_round" in data:
        return "repeated_clean_noise"
    rows = data.get("rows")
    if isinstance(rows, list) and rows and isinstance(rows[0], Mapping):
        first = rows[0]
        keys = set(first)
        if {"gpu_id", "dtype", "mean_step_seconds"} <= keys:
            return "gpu_fp32_fp16_throughput"
        if "confusion" in first or {"roc_auc", "pr_auc"} <= keys:
            return "client_detection"
        if {"client_a", "client_b"} <= keys:
            return "relation_edges"
        if "similarity" in first and {"client_i", "client_j"} <= keys:
            return "residual_similarity_matrix"
        if {"split", "method"} <= keys and (
                {"false_positive_rate", "false_positives"} & keys):
            return "iid_non_iid_false_positive"
        if "gamma_probability" in keys or {"method", "gamma_probability"} <= keys:
            return "gamma_paired"
        if {"method", "group_recall"} <= keys or "exact_group_recovery" in keys:
            return "group_detection"
        if {"provider", "requested_model"} <= keys:
            return "online_api"
        if "unsafe_response_rate" in keys:
            return "instruction_poisoning"
        if "bsnr" in keys:
            return "bsnr"
        if "suspicious_clients" in keys:
            return "preagg_filter"
        if {"condition", "asr_estimate"} <= keys and (
                {"provider", "model"} & keys):
            return "online_api"
        if "dose" in keys or "poison_fraction" in keys:
            return "dose_response"
        if "malicious_clients" in keys or "malicious_count" in keys:
            return "malicious_count_effect"
        if {"window_start", "attack_start", "window_end", "attack_end"} & keys:
            return "attack_window"
        if "method" in first and ({"asr_estimate", "asr"} & keys):
            return "aggregation_comparison"
    if data.get("method") == "Safe-FedLLM" and "roc_auc" in data:
        return "client_detection"
    if data.get("method") == "CBDF-LoRA" and "pairs" in data:
        return "relation_edges"
    if data.get("kind") == "cbdf_lora_residual_matrix" and "clients" in data:
        return "cbdf_lora_residual_matrix"
    if "clients" in data and ("matrix" in data or "similarity_matrix" in data):
        return "residual_similarity_matrix"
    if "fits_declared_context" in data or "raw_input_tokens_before_instruction" in data:
        return "online_raw_preflight"
    if "method" in data and "roc_auc" in data:
        return "client_detection"
    if "method" in data and ({"suspicious_clients", "accepted_clients"} & set(data)):
        return "preagg_filter"
    if "bsnr" in data:
        return "bsnr"
    if "unsafe_response_rate" in data:
        return "instruction_poisoning"
    if {"provider", "requested_model"} <= set(data):
        return "online_api"
    if "matrix" in data and "clients" in data:
        return "residual_similarity_matrix"
    return None


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def load_evidence(input_dir: Path) -> EvidenceSet:
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Evidence input directory does not exist: {input_dir}")
    evidence = EvidenceSet(input_dir=input_dir)
    for path in sorted(input_dir.rglob("*.json")):
        try:
            raw = path.read_bytes()
            payload = json.loads(raw)
        except (OSError, ValueError) as error:
            evidence.errors.append({"path": str(path.relative_to(input_dir)), "error": str(error)})
            continue
        if isinstance(payload, list):
            payload = {"rows": payload}
        if not isinstance(payload, Mapping):
            evidence.errors.append({"path": str(path.relative_to(input_dir)),
                                    "error": "top-level JSON is not an object or array of rows"})
            continue
        evidence.artifacts.append(Evidence(
            path=str(path.relative_to(input_dir)),
            sha256=_sha256_bytes(raw),
            data=payload,
            kind=_detect_kind(payload),
        ))
    return evidence


def _is_formal(scope: Any) -> bool:
    if not isinstance(scope, str) or not scope:
        return True
    upper = scope.upper()
    return not any(token in upper for token in NON_FORMAL_SCOPE_TOKENS)


# ---------------------------------------------------------------------------
# Small accessors over evidence rows
# ---------------------------------------------------------------------------


def _first(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def _seeds_of(data: Mapping[str, Any], rows: Iterable[Mapping[str, Any]] = ()) -> list[int]:
    explicit = data.get("seeds")
    if isinstance(explicit, list):
        seeds = sorted({_count(seed) for seed in explicit} - {None})
        if seeds:
            return seeds  # type: ignore[return-value]
    seed = _count(data.get("seed"))
    if seed is not None:
        return [seed]
    found = {_count(row.get("seed")) for row in rows}
    found.discard(None)
    return sorted(found)  # type: ignore[arg-type]


def _config_sha(data: Mapping[str, Any], row: Mapping[str, Any] | None = None) -> str | None:
    if row is not None:
        value = _first(row, "config_sha256", "config_sha", "config_hash")
        if isinstance(value, str) and value:
            return value
    value = _first(data, "config_sha256", "config_sha", "config_hash", "_hash")
    if isinstance(value, str) and value:
        return value
    nested = data.get("config")
    if isinstance(nested, Mapping):
        value = _first(nested, "config_sha256", "config_sha", "config_hash", "_hash")
        if isinstance(value, str) and value:
            return value
    return None


def _provenance(evidence: Evidence, row: Mapping[str, Any] | None = None,
                seeds: Sequence[int] = ()) -> dict[str, Any]:
    row = row or {}
    seed = _count(row.get("seed"))
    seed_count = len(seeds) if seeds else (1 if seed is not None else None)
    return {
        "run_id": row.get("run_id") or evidence.run_id,
        "seed": seed if seed is not None else (seeds[0] if len(seeds) == 1 else None),
        "seed_count": seed_count,
        "config_sha": _config_sha(evidence.data, row),
        "round": _count(row.get("round")),
        "scope": row.get("scope") or evidence.scope,
        "source_path": evidence.path,
        "source_sha256": evidence.sha256,
    }


def _row_with_provenance(evidence: Evidence, row: Mapping[str, Any],
                         seeds: Sequence[int], **fields: Any) -> dict[str, Any]:
    output = _provenance(evidence, row, seeds)
    output.update(fields)
    return output


def _row_with_fields(evidence: Evidence, row: Mapping[str, Any],
                     seeds: Sequence[int]) -> dict[str, Any]:
    """Provenance followed by every explicitly supplied table field."""
    output = _provenance(evidence, row, seeds)
    output.update(dict(row))
    return output


def _metric_fields(row: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    successes = _count(_first(row, f"{prefix}successes", f"{prefix}success_count"))
    total = _count(_first(row, f"{prefix}total", f"{prefix}trials", f"{prefix}n"))
    estimate = _finite(_first(row, f"{prefix}estimate", f"{prefix}asr", f"{prefix}rate"))
    lower = _finite(_first(row, f"{prefix}lower", f"{prefix}ci_lower"))
    upper = _finite(_first(row, f"{prefix}upper", f"{prefix}ci_upper"))
    if successes is not None and total:
        interval = _wilson(successes, total)
        if interval is not None:
            estimate = interval["estimate"] if estimate is None else estimate
            lower = interval["lower"] if lower is None else lower
            upper = interval["upper"] if upper is None else upper
    return {"successes": successes, "total": total, "estimate": estimate,
            "lower": lower, "upper": upper}


# ---------------------------------------------------------------------------
# Table definitions / builders
# ---------------------------------------------------------------------------


@dataclass
class TableResult:
    table_id: str
    title: str
    status: str
    columns: list[str]
    rows: list[dict[str, Any]] = field(default_factory=list)
    missing_inputs: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    h2_pass: bool | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def as_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": TABLE_SCHEMA_VERSION,
            "table_id": self.table_id,
            "title": self.title,
            "status": self.status,
            "complete": self.status == "COMPLETE",
            "columns": self.columns,
            "rows": self.rows,
            "row_count": len(self.rows),
            "missing_inputs": self.missing_inputs,
            "notes": self.notes,
            "evidence": self.evidence,
            "formal_claim": False,
        }
        if self.h2_pass is not None:
            payload["h2_pass"] = self.h2_pass
        return payload

    def as_markdown(self) -> str:
        lines = [f"# {self.table_id}: {self.title}", "",
                 f"Status: **{self.status}**", "",
                 "This table is generated from saved JSON evidence; empty metrics are not measured.",
                 ""]
        if self.h2_pass is not None:
            lines += [f"H2 pass: {self.h2_pass}", ""]
        if self.columns:
            lines.append("| " + " | ".join(self.columns) + " |")
            lines.append("|" + "|".join("---" for _ in self.columns) + "|")
            for row in self.rows:
                lines.append("| " + " | ".join(_markdown_cell(row.get(name)) for name in self.columns) + " |")
        else:
            lines.append("_No columns declared._")
        if self.missing_inputs:
            lines += ["", "Missing evidence required to populate this table:"]
            lines += [f"- {item}" for item in self.missing_inputs]
        if self.notes:
            lines += ["", "Notes:"]
            lines += [f"- {item}" for item in self.notes]
        return "\n".join(lines) + "\n"


def _markdown_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True).replace("|", "\\|")
    return str(value).replace("|", "\\|")


def _table_specs() -> list[dict[str, Any]]:
    """Fixed ordered specification of the 16 required tables."""
    return [
        {"id": "table_01_gpu_fp32_fp16", "title": "GPU FP32/FP16 throughput and memory",
         "kinds": ("gpu_fp32_fp16_throughput",),
         "columns": ["gpu_id", "logical_gpu_id", "gpu_name", "dtype", "optimizer_steps", "load_seconds",
                     "mean_step_seconds", "peak_allocated_bytes", "peak_reserved_bytes",
                     "adapter_parameters", "model_revision", *COMMON_COLUMNS]},
        {"id": "table_02_repeated_clean_noise", "title": "Clean training and repeated-training noise floor",
         "kinds": ("repeated_clean_noise",),
         "columns": ["metric", "reference", "repeat", "difference", "absolute_difference",
                     "adapter_l2", "adapter_rms", "adapter_max_abs", "adapter_cosine",
                     "config_id", *COMMON_COLUMNS]},
        {"id": "table_03_badnet_effectiveness", "title": "BadNet attack effectiveness (paired clean counterfactual)",
         "kinds": ("evaluation_suite", "paired_clean_comparison", PAIRED_CLEAN_SCHEMA),
         "columns": ["attacked_run_id", "clean_run_id", "seed", "seed_count", "config_sha",
                     "round", "asr_successes", "asr_total", "asr_estimate", "asr_lower",
                     "asr_upper", "clean_prompt_misfire", "clean_control_asr",
                     "clean_asr_estimate", "attacked_accuracy", "clean_accuracy",
                     "normal_task_drop", "paired", "h2_seed_pass", "scope", "source_path"]},
        {"id": "table_04_dose_response", "title": "Poison-dose response",
         "kinds": ("dose_response",),
         "columns": ["dose", "condition", "asr_successes", "asr_total", "asr_estimate",
                     "asr_lower", "asr_upper", "clean_nll", "round", *COMMON_COLUMNS]},
        {"id": "table_05_malicious_count", "title": "Malicious-client count at fixed poison budget",
         "kinds": ("malicious_count_effect",),
         "columns": ["malicious_clients", "poison_budget", "condition", "asr_successes",
                     "asr_total", "asr_estimate", "asr_lower", "asr_upper", "round",
                     *COMMON_COLUMNS]},
        {"id": "table_06_gamma", "title": "Gamma on probability and logit scales",
         "kinds": ("gamma_paired",),
         "columns": ["condition", "metric", "gamma_probability", "gamma_logit",
                     "logit_epsilon", "round", *COMMON_COLUMNS]},
        {"id": "table_07_attack_window", "title": "Attack window and effect decay",
         "kinds": ("attack_window",),
         "columns": ["condition", "window_start", "window_end", "round", "asr_estimate",
                     "asr_successes", "asr_total", "clean_nll", "decay_from_peak",
                     *COMMON_COLUMNS]},
        {"id": "table_08_instruction_poisoning", "title": "Instruction-response poisoning",
         "kinds": ("instruction_poisoning",),
         "columns": ["condition", "unsafe_response_rate", "successes", "total", "estimate",
                     "lower", "upper", "clean_nll", "round", *COMMON_COLUMNS]},
        {"id": "table_09_bsnr", "title": "BSNR, residual similarity and ASR",
         "kinds": ("bsnr",),
         "columns": ["client_id", "bsnr", "residual_similarity", "asr_estimate",
                     "clean_noise_floor", "probe_count", "round", *COMMON_COLUMNS]},
        {"id": "table_10_client_detection", "title": "Client-level detection performance",
         "kinds": ("client_detection",),
         "columns": ["method", "threshold", "roc_auc", "pr_auc", "precision", "recall",
                     "f1", "fpr", "fnr", "tn", "fp", "fn", "tp", "flagged_clients",
                     "round", *COMMON_COLUMNS]},
        {"id": "table_11_group_detection", "title": "Collaborative group detection performance",
         "kinds": ("group_detection",),
         "columns": ["method", "group_recall", "exact_group_recovery", "precision",
                     "recall", "f1", "detected_groups", "true_groups", "round",
                     *COMMON_COLUMNS]},
        {"id": "table_12_iid_non_iid_false_positive", "title": "IID and Non-IID benign false positives",
         "kinds": ("iid_non_iid_false_positive",),
         "columns": ["split", "method", "false_positives", "benign_total", "false_positive_rate",
                     "true_positives", "detected_total", "round", *COMMON_COLUMNS]},
        {"id": "table_13_online_apis", "title": "Online API detection results and cost",
         "kinds": ("online_api", "online_api_costs"),
         "columns": ["provider", "requested_model", "resolved_model", "classification_authorized",
                     "detections", "successes", "total", "estimated_cost_usd",
                     "actual_cost_usd", "round", *COMMON_COLUMNS]},
        {"id": "table_14_aggregation", "title": "Aggregation-algorithm comparison",
         "kinds": ("aggregation_comparison",),
         "columns": ["method", "condition", "asr_successes", "asr_total", "asr_estimate",
                     "asr_lower", "asr_upper", "clean_nll", "round", *COMMON_COLUMNS]},
        {"id": "table_15_preagg_filters", "title": "Pre-aggregation filter effectiveness",
         "kinds": ("preagg_filter", "preagg_filter_effect"),
         "columns": ["method", "suspicious_clients", "accepted_clients", "malicious_filter_rate",
                     "benign_false_filter_rate", "asr_estimate", "clean_nll", "run_id",
                     "round", "seed", "seed_count", "config_sha", "scope", "source_path"]},
        {"id": "table_16_raw_lora_online", "title": "Raw LoRA parameter online classification",
         "kinds": ("raw_lora_classification", "online_raw_preflight"),
         "columns": ["provider", "model", "client_id", "tensors", "parameters",
                     "raw_tokens", "max_context_tokens", "fits_declared_context",
                     "classification_performed", "successes", "total", "status",
                     "round", *COMMON_COLUMNS]},
    ]


# Required non-null columns per table for a COMPLETE status.
REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "table_01_gpu_fp32_fp16": ("gpu_id", "dtype", "mean_step_seconds", "peak_allocated_bytes"),
    "table_02_repeated_clean_noise": ("metric", "reference", "repeat"),
    "table_04_dose_response": ("dose", "asr_total", "asr_estimate"),
    "table_05_malicious_count": ("malicious_clients", "asr_total", "asr_estimate"),
    "table_06_gamma": ("gamma_probability", "gamma_logit"),
    "table_07_attack_window": ("round", "asr_estimate"),
    "table_08_instruction_poisoning": ("estimate", "total"),
    "table_09_bsnr": ("bsnr",),
    "table_10_client_detection": ("method", "roc_auc"),
    "table_11_group_detection": ("method", "group_recall"),
    "table_12_iid_non_iid_false_positive": ("split", "method", "false_positive_rate"),
    "table_13_online_apis": ("provider", "classification_authorized"),
    "table_14_aggregation": ("method", "asr_total", "asr_estimate"),
    "table_15_preagg_filters": ("method", "suspicious_clients", "accepted_clients"),
    "table_16_raw_lora_online": ("status",),
}

SCOPE_FOR_TABLE = {
    "table_01_gpu_fp32_fp16": "benchmark_precision rows (gpu_id,dtype,mean_step_seconds,peak_allocated_bytes)",
    "table_02_repeated_clean_noise": "repeated-clean noise report with metric_noise_by_round and matched rounds",
    "table_04_dose_response": "dose-response evaluation rows with per-seed ASR counts for D8/D25/D50",
    "table_05_malicious_count": "G1/G2/G4 evaluation rows at fixed per-client poison budget",
    "table_06_gamma": "paired Gamma artifact rows for clean/D25_A_only/D25_B_only/D25",
    "table_07_attack_window": "attack-window evaluation rows (attack rounds 1-10, recovery 11-30)",
    "table_08_instruction_poisoning": "instruction-response evaluation rows with unsafe-response counts",
    "table_09_bsnr": "BSNR artifact rows with calibration-isolated bsnr and ASR",
    "table_10_client_detection": "detector result rows with roc_auc/pr_auc and confusion counts",
    "table_11_group_detection": "group-detection rows with group recall and exact-group recovery",
    "table_12_iid_non_iid_false_positive": "IID/Non-IID benign false-positive rows per method",
    "table_13_online_apis": "authenticated online API artifacts (Kimi K3 / GLM-5.3) with cost fields",
    "table_14_aggregation": "aggregation comparison rows (fedavg/trimmed_mean/multi_krum/foolsgold)",
    "table_15_preagg_filters": "pre-aggregation filter artifacts (none/oracle/safe_fedllm/cbdf_lora)",
    "table_16_raw_lora_online": "raw LoRA online preflight/classification artifacts",
}


ROW_LIST_KEYS = ("rows", "results", "pairs", "records", "scores", "round_summaries")
ROW_INDICATOR_KEYS = ("method", "provider", "client_id", "gpu_id", "status", "dose",
                      "malicious_clients", "malicious_count", "condition", "round")


def _artifact_rows(evidence: Evidence) -> list[Mapping[str, Any]]:
    for key in ROW_LIST_KEYS:
        value = evidence.data.get(key)
        if isinstance(value, list) and all(isinstance(item, Mapping) for item in value):
            return list(value)
    # Some real artifacts are a single record (for example a detector result or a
    # raw-adapter preflight).  Only promote a top-level object to a row when it
    # carries an unmistakable row field; wrappers/metadata are never promoted.
    if any(key in evidence.data for key in ROW_INDICATOR_KEYS):
        return [evidence.data]
    return []


def _common_extract(evidence: Evidence, row: Mapping[str, Any],
                    seeds: Sequence[int], fields: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    values = {column: _first(row, *names) for column, names in fields.items()}
    return _row_with_provenance(evidence, row, seeds, **values)


def _assemble(table_id: str, title: str, columns: list[str],
              artifacts: list[Evidence], rows: list[dict[str, Any]],
              required: tuple[str, ...], missing: list[str],
              notes: list[str] | None = None) -> TableResult:
    evidence_index = [{"path": item.path, "sha256": item.sha256,
                       "kind": item.kind, "run_id": item.run_id, "scope": item.scope}
                      for item in artifacts]
    notes = list(notes or [])
    if not artifacts:
        return TableResult(table_id, title, "NOT_RUN", columns, [], [missing[0]], notes, None, evidence_index)
    if not rows:
        return TableResult(table_id, title, "INCOMPLETE", columns, [], missing, notes, None, evidence_index)
    missing_fields = sorted({name for row in rows for name in required if row.get(name) is None})
    nonformal = any(row.get("scope") is not None and not _is_formal(row.get("scope")) for row in rows)
    if missing_fields:
        notes.append(f"Populated rows are missing required fields: {missing_fields}")
        return TableResult(table_id, title, "INCOMPLETE", columns, rows,
                           [f"fields {missing_fields}"] + missing, notes, None, evidence_index)
    if nonformal:
        notes.append("Evidence scope is engineering/precheck, not a formal result; table marked INCOMPLETE.")
        return TableResult(table_id, title, "INCOMPLETE", columns, rows, missing, notes, None, evidence_index)
    return TableResult(table_id, title, "COMPLETE", columns, rows, [], notes, None, evidence_index)


def _build_throughput(evidence: EvidenceSet, spec: Mapping[str, Any]) -> TableResult:
    artifacts = evidence.of_kind(*spec["kinds"])
    rows = []
    for artifact in artifacts:
        for row in _artifact_rows(artifact):
            rows.append(_common_extract(artifact, row, _seeds_of(artifact.data), {
                # A single visible device is exposed as logical CUDA device 0;
                # never report that logical index as the measured physical GPU.
                "gpu_id": ("physical_gpu_id", "gpu_id"),
                "logical_gpu_id": ("gpu_id",),
                "gpu_name": ("gpu_name",), "dtype": ("dtype",),
                "optimizer_steps": ("optimizer_steps",), "load_seconds": ("load_seconds",),
                "mean_step_seconds": ("mean_step_seconds",),
                "peak_allocated_bytes": ("peak_allocated_bytes",),
                "peak_reserved_bytes": ("peak_reserved_bytes",),
                "adapter_parameters": ("adapter_parameters",),
                "model_revision": ("model_revision",),
            }))
    return _assemble(spec["id"], spec["title"], spec["columns"], artifacts, rows,
                     REQUIRED_COLUMNS[spec["id"]], [SCOPE_FOR_TABLE[spec["id"]]])


def _build_repeated_noise(evidence: EvidenceSet, spec: Mapping[str, Any]) -> TableResult:
    artifacts = evidence.of_kind(*spec["kinds"])
    rows = []
    for artifact in artifacts:
        data = artifact.data
        metrics = _as_mapping(data.get("metric_noise_by_round")) or {}
        adapters = _as_mapping(data.get("adapter_noise_by_round")) or {}
        seeds = _seeds_of(data)
        for round_index in sorted({int(key) for key in metrics} | {int(key) for key in adapters}):
            metric_row = _as_mapping(metrics.get(str(round_index)) or metrics.get(round_index)) or {}
            adapter_row = _as_mapping(adapters.get(str(round_index)) or adapters.get(round_index)) or {}
            names = sorted(metric_row) or [None]
            for name in names:
                values = _as_mapping(metric_row.get(name)) if name else None
                values = values or {}
                row = {"round": round_index,
                       "metric": name,
                       "reference": _finite(values.get("reference")),
                       "repeat": _finite(values.get("repeat")),
                       "difference": _finite(values.get("difference")),
                       "absolute_difference": _finite(values.get("absolute_difference")),
                       "adapter_l2": _finite(adapter_row.get("l2")),
                       "adapter_rms": _finite(adapter_row.get("rms")),
                       "adapter_max_abs": _finite(adapter_row.get("max_abs")),
                       "adapter_cosine": _finite(adapter_row.get("cosine")),
                       "config_id": data.get("config_id")}
                rows.append(_row_with_fields(artifact, row, seeds))
    return _assemble(spec["id"], spec["title"], spec["columns"], artifacts, rows,
                     REQUIRED_COLUMNS[spec["id"]], [SCOPE_FOR_TABLE[spec["id"]]])


def _build_simple_rows(evidence: EvidenceSet, spec: Mapping[str, Any],
                       fields: Mapping[str, Sequence[str]],
                       computed: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None,
                       ) -> TableResult:
    artifacts = evidence.of_kind(*spec["kinds"])
    rows = []
    for artifact in artifacts:
        seeds = _seeds_of(artifact.data)
        for row in _artifact_rows(artifact):
            values = _common_extract(artifact, row, seeds, fields)
            if computed is not None:
                values.update(computed(row))
            rows.append(values)
    return _assemble(spec["id"], spec["title"], spec["columns"], artifacts, rows,
                     REQUIRED_COLUMNS.get(spec["id"], ()), [SCOPE_FOR_TABLE.get(spec["id"], spec["id"])])


def _metrics_computed(row: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _metric_fields(row)
    total = metrics["total"]
    estimate = metrics["estimate"]
    if estimate is None and total:
        estimate = (metrics["successes"] / total) if metrics["successes"] is not None else None
    metrics["estimate"] = estimate
    return metrics


def _asr_computed(row: Mapping[str, Any]) -> dict[str, Any]:
    """ASR metrics under the table's explicit ``asr_*`` column names."""
    metrics = _metrics_computed(row)
    return {"asr_successes": metrics["successes"], "asr_total": metrics["total"],
            "asr_estimate": metrics["estimate"], "asr_lower": metrics["lower"],
            "asr_upper": metrics["upper"]}


def _detection_computed(row: Mapping[str, Any]) -> dict[str, Any]:
    confusion = _as_mapping(row.get("confusion")) or {}
    tp, fp = _count(confusion.get("tp")), _count(confusion.get("fp"))
    fn, tn = _count(confusion.get("fn")), _count(confusion.get("tn"))
    precision = tp / (tp + fp) if tp is not None and fp is not None and tp + fp else None
    recall = tp / (tp + fn) if tp is not None and fn is not None and tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else None)
    return {"roc_auc": _finite(_first(row, "roc_auc", "auc")),
            "pr_auc": _finite(row.get("pr_auc")),
            "precision": precision, "recall": recall, "f1": f1,
            "fpr": _finite(row.get("fpr")), "fnr": _finite(row.get("fnr")),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "threshold": _finite(row.get("threshold")),
            "flagged_clients": row.get("flagged_clients")}


H2_MACRO_ASR = 0.80
H2_SEED_ASR = 0.70
H2_SEED_SUCCESSES = 180
H2_MISFIRE = 0.10
H2_CONTROL_ASR = 0.10
H2_TASK_DROP = 0.05


def _nested_metric(row: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = _as_mapping(row.get(key))
    return value or {}


def _paired_rows(evidence: EvidenceSet) -> tuple[list[dict[str, Any]], list[Evidence], bool, bool]:
    """Return H2 rows at their declared measurement round and their evidence."""
    rows: list[dict[str, Any]] = []
    used: list[Evidence] = []
    protocol_incomplete = False
    for artifact in evidence.of_kind("evaluation_suite"):
        counterfactual = _as_mapping(artifact.data.get("paired_formal_counterfactual"))
        if counterfactual is None:
            continue
        pairs = counterfactual.get("pairs")
        if not isinstance(pairs, list) or not pairs:
            continue
        used.append(artifact)
        seeds = _seeds_of(artifact.data)
        for pair in pairs:
            if isinstance(pair, Mapping):
                rows.append(_paired_row(artifact, pair, seeds))
    for artifact in evidence.of_kind("paired_clean_comparison", PAIRED_CLEAN_SCHEMA):
        pairs = artifact.data.get("pairs")
        if not isinstance(pairs, list) or not pairs:
            continue
        seeds = _seeds_of(artifact.data)
        measurement_round = artifact.data.get("h2_measurement_round")
        if measurement_round is not None:
            if _count(measurement_round) is None:
                raise ValueError("Paired artifact has an invalid H2 measurement round")
            pairs = [pair for pair in pairs if isinstance(pair, Mapping)
                     and pair.get("round") == measurement_round]
            if not pairs:
                raise ValueError("Paired artifact lacks its declared H2 measurement round")
        used.append(artifact)
        for pair in pairs:
            if isinstance(pair, Mapping):
                rows.append(_paired_row(artifact, pair, seeds))
    return rows, used, bool(rows), protocol_incomplete


def _paired_row(evidence: Evidence, pair: Mapping[str, Any],
                seeds: Sequence[int]) -> dict[str, Any]:
    attacked = _nested_metric(pair, "attacked_asr")
    metrics = _metric_fields(attacked) if attacked else _metric_fields(pair)
    if metrics["estimate"] is None and metrics["successes"] is not None and metrics["total"]:
        metrics["estimate"] = metrics["successes"] / metrics["total"]
    seed = _count(pair.get("seed"))
    attacked_run = pair.get("attacked_run_id") or pair.get("run_id")
    clean_run = pair.get("clean_run_id") or pair.get("clean_counterfactual_run_id")
    misfire = _finite(_first(pair, "clean_prompt_misfire", "misfire", "clean_misfire"))
    control = _finite(_first(pair, "clean_control_asr", "control_asr"))
    attacked_accuracy = _finite(_first(pair, "attacked_accuracy", "asr_accuracy_metric"))
    clean_accuracy = _finite(_first(pair, "clean_accuracy", "clean_counterfactual_accuracy"))
    drop = (clean_accuracy - attacked_accuracy
            if attacked_accuracy is not None and clean_accuracy is not None else None)
    attacked_sha = pair.get("attacked_config_sha") or pair.get("config_sha")
    clean_sha = pair.get("clean_config_sha")
    config_sha_consistent = not (attacked_sha and clean_sha and attacked_sha != clean_sha)
    seed_pass = (
        seed is not None
        and metrics["estimate"] is not None and metrics["estimate"] >= H2_SEED_ASR
        and metrics["successes"] is not None and metrics["successes"] >= H2_SEED_SUCCESSES
        and misfire is not None and misfire <= H2_MISFIRE
        and control is not None and control <= H2_CONTROL_ASR
    )
    return {
        "attacked_run_id": attacked_run,
        "clean_run_id": clean_run,
        "seed": seed,
        "seed_count": len(seeds) if seeds else (1 if seed is not None else None),
        "config_sha": attacked_sha or clean_sha or _config_sha(evidence.data),
        "round": _count(pair.get("round")),
        "asr_successes": metrics["successes"],
        "asr_total": metrics["total"],
        "asr_estimate": metrics["estimate"],
        "asr_lower": metrics["lower"],
        "asr_upper": metrics["upper"],
        "clean_prompt_misfire": misfire,
        "clean_control_asr": control,
        "clean_asr_estimate": _finite(_first(pair, "clean_asr_estimate", "clean_asr")),
        "attacked_accuracy": attacked_accuracy,
        "clean_accuracy": clean_accuracy,
        "normal_task_drop": drop,
        "paired": bool(attacked_run and clean_run and config_sha_consistent),
        "h2_seed_pass": bool(seed_pass and config_sha_consistent),
        "scope": pair.get("scope") or evidence.scope,
        "source_path": evidence.path,
        "source_sha256": evidence.sha256,
        "_protocol_complete": bool(evidence.data.get("complete_protocol_probe_count", True)),
    }


def _h2_summary(rows: list[Mapping[str, Any]]) -> tuple[bool, dict[str, Any]]:
    paired = [row for row in rows if row.get("paired")]
    if not paired:
        return False, {"reason": "no paired attacked/clean rows", "macro_asr": None,
                       "per_seed": [], "task_drop": None}
    estimates = [_finite(row.get("asr_estimate")) for row in paired]
    estimates = [value for value in estimates if value is not None]
    macro = statistics.fmean(estimates) if estimates else None
    protocol_complete = all(row.get("_protocol_complete", True) for row in paired)
    per_seed_pass = all(bool(row.get("h2_seed_pass")) for row in paired) and protocol_complete
    macro_pass = macro is not None and len(estimates) == len(paired) and macro >= H2_MACRO_ASR
    drops = [_finite(row.get("normal_task_drop")) for row in paired]
    drops_measured = all(value is not None for value in drops)
    task_pass = drops_measured and all(value <= H2_TASK_DROP for value in drops)  # type: ignore[operator]
    h2 = bool(per_seed_pass and macro_pass and task_pass)
    return h2, {"macro_asr": macro, "per_seed_pass": per_seed_pass,
                "macro_asr_pass": macro_pass, "task_drop_measured": drops_measured,
                "task_drop_pass": task_pass if drops_measured else None,
                "protocol_complete": protocol_complete,
                "thresholds": {"macro_asr": H2_MACRO_ASR, "seed_asr": H2_SEED_ASR,
                               "seed_successes": H2_SEED_SUCCESSES,
                               "clean_prompt_misfire": H2_MISFIRE,
                               "clean_control_asr": H2_CONTROL_ASR,
                               "normal_task_drop": H2_TASK_DROP}}


def _build_badnet(evidence: EvidenceSet, spec: Mapping[str, Any]) -> TableResult:
    paired, used, any_pair, _ = _paired_rows(evidence)
    suite_artifacts = evidence.of_kind("evaluation_suite")
    artifacts = used or suite_artifacts
    if not artifacts:
        return TableResult(spec["id"], spec["title"], "NOT_RUN", spec["columns"], [], [MISSING_ASR],
                           ["Table 3 requires a paired clean counterfactual; round-0 deltas are not one."],
                           False, [])
    evidence_index = [{"path": item.path, "sha256": item.sha256, "kind": item.kind,
                       "run_id": item.run_id, "scope": item.scope} for item in artifacts]
    if not any_pair:
        protocol_incomplete = any(
            not bool(item.data.get("complete_protocol_probe_count", False))
            for item in suite_artifacts)
        status = "INCOMPLETE" if protocol_incomplete else "UNPAIRED"
        missing = ["independent paired clean counterfactual (same seed/records/participation/aggregation)"]
        if protocol_incomplete:
            missing.insert(0, "complete 256-probe per-seed evaluation")
        return TableResult(
            spec["id"], spec["title"], status, spec["columns"], [], missing,
            ["evaluation_suite records paired_formal_counterfactual=null by design; "
             "attacked ASR is deliberately not promoted without its clean pair.",
             "H2 is not declared passed."], False, evidence_index)
    h2_pass, summary = _h2_summary(paired)
    required = ("attacked_run_id", "clean_run_id", "seed", "asr_total", "asr_estimate",
                "clean_prompt_misfire")
    missing_fields = sorted({name for row in paired for name in required if row.get(name) is None})
    nonformal = any(not _is_formal(row.get("scope")) for row in paired)
    protocol_ok = all(row.get("_protocol_complete", True) for row in paired)
    clean_rows = [{key: value for key, value in row.items() if not key.startswith("_")}
                  for row in paired]
    if missing_fields or nonformal or not protocol_ok:
        notes = []
        if missing_fields:
            notes.append(f"Paired rows missing required fields: {missing_fields}")
        if nonformal:
            notes.append("Paired evidence scope is engineering/precheck, not a formal result.")
        if not protocol_ok:
            notes.append("Paired evidence did not evaluate the complete 256-probe protocol.")
        return TableResult(spec["id"], spec["title"], "INCOMPLETE", spec["columns"],
                           clean_rows, [f"fields {missing_fields}"], notes, h2_pass, evidence_index)
    return TableResult(spec["id"], spec["title"], "COMPLETE", spec["columns"], clean_rows,
                       [], [f"H2 evaluation: {json.dumps(summary, sort_keys=True)}"],
                       h2_pass, evidence_index)


def _build_gamma(evidence: EvidenceSet, spec: Mapping[str, Any]) -> TableResult:
    return _build_simple_rows(evidence, spec, {
        "condition": ("condition", "scenario", "scenario_name"),
        "metric": ("metric", "metric_name"),
        "gamma_probability": ("gamma_probability", "gamma_prob"),
        "gamma_logit": ("gamma_logit",),
        "logit_epsilon": ("logit_epsilon", "epsilon"),
    })


def _build_bsnr(evidence: EvidenceSet, spec: Mapping[str, Any]) -> TableResult:
    return _build_simple_rows(evidence, spec, {
        "client_id": ("client_id",),
        "bsnr": ("bsnr",),
        "residual_similarity": ("residual_similarity", "similarity", "cosine"),
        "asr_estimate": ("asr_estimate", "asr"),
        "clean_noise_floor": ("clean_noise_floor", "noise_floor"),
        "probe_count": ("probe_count", "evaluated_probes", "total"),
    })


def _build_detection(evidence: EvidenceSet, spec: Mapping[str, Any]) -> TableResult:
    artifacts = evidence.of_kind(*spec["kinds"])
    rows = []
    for artifact in artifacts:
        data = artifact.data
        seeds = _seeds_of(data)
        # A detector summary may also contain per-client scores. Do not mistake
        # those score rows for the method-level confusion/AUC evaluation.
        source_rows = [data] if "roc_auc" in data and "confusion" in data else _artifact_rows(artifact)
        for row in source_rows:
            values = _common_extract(artifact, row, seeds, {
                "method": ("method",), "threshold": ("threshold",),
                "flagged_clients": ("flagged_clients",),
            })
            if "roc_auc" in data and "confusion" in data:
                values["method"] = data.get("method")
                values["round"] = None
            values.update(_detection_computed(row))
            rows.append(values)
    return _assemble(spec["id"], spec["title"], spec["columns"], artifacts, rows,
                     REQUIRED_COLUMNS[spec["id"]], [SCOPE_FOR_TABLE[spec["id"]]])


def _build_group_detection(evidence: EvidenceSet, spec: Mapping[str, Any]) -> TableResult:
    return _build_simple_rows(evidence, spec, {
        "method": ("method",),
        "group_recall": ("group_recall",),
        "exact_group_recovery": ("exact_group_recovery",),
        "precision": ("precision",),
        "recall": ("recall",),
        "f1": ("f1",),
        "detected_groups": ("detected_groups",),
        "true_groups": ("true_groups",),
    })


def _build_invariance(evidence: EvidenceSet, spec: Mapping[str, Any]) -> TableResult:
    def computed(row: Mapping[str, Any]) -> dict[str, Any]:
        false_positives = _count(_first(row, "false_positives", "fp"))
        benign_total = _count(_first(row, "benign_total", "tn", "benign_count"))
        if benign_total is None and false_positives is not None and _count(row.get("tn")) is not None:
            benign_total = false_positives + int(row["tn"])
        rate = _finite(_first(row, "false_positive_rate", "fpr"))
        if rate is None and false_positives is not None and benign_total:
            rate = false_positives / benign_total
        return {"false_positives": false_positives, "benign_total": benign_total,
                "false_positive_rate": rate,
                "true_positives": _count(row.get("true_positives")),
                "detected_total": _count(row.get("detected_total"))}
    return _build_simple_rows(evidence, spec, {
        "split": ("split", "distribution"),
        "method": ("method",),
    }, computed)


def _build_online_api(evidence: EvidenceSet, spec: Mapping[str, Any]) -> TableResult:
    return _build_simple_rows(evidence, spec, {
        "provider": ("provider", "provider_name"),
        "requested_model": ("requested_model",),
        "resolved_model": ("resolved_model", "exact_match"),
        "classification_authorized": ("classification_authorized",),
        "detections": ("detections", "flagged_clients"),
        "successes": ("successes", "success_count"),
        "total": ("total", "trials"),
        "estimated_cost_usd": ("estimated_cost_usd", "estimated_cost"),
        "actual_cost_usd": ("actual_cost_usd", "actual_cost"),
    })


def _build_preagg(evidence: EvidenceSet, spec: Mapping[str, Any]) -> TableResult:
    return _build_simple_rows(evidence, spec, {
        "method": ("method",),
        "suspicious_clients": ("suspicious_clients",),
        "accepted_clients": ("accepted_clients",),
        "malicious_filter_rate": ("malicious_filter_rate",),
        "benign_false_filter_rate": ("benign_false_filter_rate",),
        "asr_estimate": ("asr_estimate", "asr"),
        "clean_nll": ("clean_nll",),
    })


def _build_raw_lora(evidence: EvidenceSet, spec: Mapping[str, Any]) -> TableResult:
    def computed(row: Mapping[str, Any]) -> dict[str, Any]:
        adapter = _as_mapping(row.get("adapter")) or {}
        status = row.get("status")
        return {
            "tensors": adapter.get("tensors"),
            "parameters": adapter.get("parameters"),
            "raw_tokens": _count(_first(row, "raw_tokens",
                                        "raw_input_tokens_before_instruction"))
            or (_count(adapter.get("tokens_actual_qwen_tokenizer")) if adapter else None),
            "max_context_tokens": _count(row.get("max_context_tokens")),
            "fits_declared_context": row.get("fits_declared_context"),
            "classification_performed": bool(_first(row, "classification_performed",
                                                     "api_classification_performed"))
            or (isinstance(status, str) and "CLASSIFICATION" in status.upper()
                and "NO_API_CLASSIFICATION" not in status.upper()),
        }
    return _build_simple_rows(evidence, spec, {
        "provider": ("provider",),
        "model": ("model", "model_id"),
        "client_id": ("client_id",),
        "successes": ("successes", "success_count"),
        "total": ("total", "trials"),
        "status": ("status",),
    }, computed)


TABLE_BUILDERS: dict[str, Callable[[EvidenceSet, Mapping[str, Any]], TableResult]] = {
    "table_01_gpu_fp32_fp16": _build_throughput,
    "table_02_repeated_clean_noise": _build_repeated_noise,
    "table_03_badnet_effectiveness": _build_badnet,
    "table_04_dose_response": lambda evidence, spec: _build_simple_rows(evidence, spec, {
        "dose": ("dose", "poison_fraction"), "condition": ("condition", "scenario"),
        "clean_nll": ("clean_nll",)}, _asr_computed),
    "table_05_malicious_count": lambda evidence, spec: _build_simple_rows(evidence, spec, {
        "malicious_clients": ("malicious_clients", "malicious_count"),
        "poison_budget": ("poison_budget", "per_client_count"),
        "condition": ("condition", "scenario")}, _asr_computed),
    "table_06_gamma": _build_gamma,
    "table_07_attack_window": lambda evidence, spec: _build_simple_rows(evidence, spec, {
        "condition": ("condition", "scenario"),
        "window_start": ("window_start", "attack_start"),
        "window_end": ("window_end", "attack_end"),
        "clean_nll": ("clean_nll",),
        "decay_from_peak": ("decay_from_peak", "decay")}, _asr_computed),
    "table_08_instruction_poisoning": lambda evidence, spec: _build_simple_rows(evidence, spec, {
        "condition": ("condition", "scenario"),
        "unsafe_response_rate": ("unsafe_response_rate",),
        "clean_nll": ("clean_nll",)},
        lambda row: {**_metrics_computed(row),
                     "unsafe_response_rate": _finite(row.get("unsafe_response_rate"))}),
    "table_09_bsnr": _build_bsnr,
    "table_10_client_detection": _build_detection,
    "table_11_group_detection": _build_group_detection,
    "table_12_iid_non_iid_false_positive": _build_invariance,
    "table_13_online_apis": _build_online_api,
    "table_14_aggregation": lambda evidence, spec: _build_simple_rows(evidence, spec, {
        "method": ("method", "aggregation_method"),
        "condition": ("condition", "scenario"),
        "clean_nll": ("clean_nll",)}, _asr_computed),
    "table_15_preagg_filters": _build_preagg,
    "table_16_raw_lora_online": _build_raw_lora,
}


def build_tables(evidence: EvidenceSet) -> list[TableResult]:
    return [TABLE_BUILDERS[spec["id"]](evidence, spec) for spec in _table_specs()]


# ---------------------------------------------------------------------------
# Figure definitions / builders
# ---------------------------------------------------------------------------


@dataclass
class FigureResult:
    figure_id: str
    title: str
    status: str
    needed_input: str
    metadata: dict[str, Any]
    plot: dict[str, Any] | None = None
    missing: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def metadata_json(self) -> dict[str, Any]:
        payload = {
            "schema": FIGURE_SCHEMA_VERSION,
            "figure_id": self.figure_id,
            "title": self.title,
            "status": self.status,
            "plotted": self.status == "RENDERED",
            "needed_input": self.needed_input,
            "missing_inputs": self.missing,
            "evidence": self.evidence,
            "formal_claim": False,
            # Always present so every figure states its configuration, seed count and CI.
            "seed_count": None,
            "config_sha": None,
            "confidence_interval": "none reported: no numeric evidence plotted",
            **self.metadata,
        }
        return payload


def _figure_specs() -> list[dict[str, Any]]:
    return [
        {"id": "figure_01_asr_by_round", "title": "ASR across communication rounds",
         "needed": "per-round ASR estimates from evaluation_suite results",
         "kind": "line", "x": "round", "y": "asr"},
        {"id": "figure_02_cumulative_impact", "title": "Cumulative malicious impact",
         "needed": "paired attacked-minus-clean per-round effects from a paired clean counterfactual",
         "kind": "line", "x": "round", "y": "cumulative_delta"},
        {"id": "figure_03_dose_response", "title": "Poison-dose response",
         "needed": "dose-response rows (dose, asr estimate)",
         "kind": "line", "x": "dose", "y": "asr"},
        {"id": "figure_04_malicious_count", "title": "Effect of malicious-client count",
         "needed": "malicious-count rows (malicious_clients, asr estimate)",
         "kind": "line", "x": "malicious_clients", "y": "asr"},
        {"id": "figure_05_gamma_gain", "title": "Gamma interaction gain",
         "needed": "paired Gamma rows (condition, gamma_probability, gamma_logit)",
         "kind": "bar", "x": "condition", "y": "gamma_probability"},
        {"id": "figure_06_attack_window_decay", "title": "Attack-window effect decay",
         "needed": "attack-window rows (round, asr estimate)",
         "kind": "line", "x": "round", "y": "asr"},
        {"id": "figure_07_bsnr_vs_asr", "title": "BSNR versus ASR",
         "needed": "BSNR rows with bsnr and asr estimate",
         "kind": "scatter", "x": "bsnr", "y": "asr"},
        {"id": "figure_08_residual_similarity_heatmap", "title": "Client residual-similarity heatmap",
         "needed": "residual similarity matrix artifact (clients, matrix)",
         "kind": "heatmap", "x": "clients", "y": "clients"},
        {"id": "figure_09_relation_network", "title": "Client relation network",
         "needed": "relation-edge artifact (client_a, client_b, statistic)",
         "kind": "network", "x": "client", "y": "client"},
        {"id": "figure_10_detection_prf", "title": "Detection precision, recall and F1",
         "needed": "client detection rows with confusion counts",
         "kind": "grouped_bar", "x": "method", "y": "precision_recall_f1"},
        {"id": "figure_11_group_detection", "title": "Group detection performance",
         "needed": "group-detection rows (method, group_recall, exact_group_recovery)",
         "kind": "grouped_bar", "x": "method", "y": "group_metrics"},
        {"id": "figure_12_iid_non_iid_fp", "title": "IID and Non-IID benign false positives",
         "needed": "IID/Non-IID false-positive rows (split, method, false_positive_rate)",
         "kind": "grouped_bar", "x": "method", "y": "false_positive_rate"},
        {"id": "figure_13_aggregation_comparison", "title": "Aggregation-method attack impact",
         "needed": "aggregation comparison rows (method, asr estimate)",
         "kind": "bar", "x": "method", "y": "asr"},
        {"id": "figure_14_filter_suppression", "title": "Filter attack suppression",
         "needed": "pre-aggregation filter rows (method, asr estimate)",
         "kind": "bar", "x": "method", "y": "asr"},
    ]


def _points_from_rows(rows: Iterable[Mapping[str, Any]], x_field: str,
                      y_fields: Sequence[str]) -> tuple[list[dict[str, Any]], list[str]]:
    points, missing = [], []
    for row in rows:
        x = _finite(row.get(x_field))
        if x is None and isinstance(row.get(x_field), str):
            x = row.get(x_field)
        y = None
        for name in y_fields:
            y = _finite(row.get(name))
            if y is not None:
                break
        if x is None or y is None:
            continue
        point = {"x": x, "y": y}
        for bound in ("lower", "upper", "estimate", "total", "successes", "seed", "run_id",
                      "config_sha", "round", "status"):
            if row.get(bound) is not None:
                point[bound] = row[bound]
        points.append(point)
    if not points:
        missing.append(f"finite {x_field} and {'/'.join(y_fields)} values")
    return points, missing


def _rows_are_formal(rows: Sequence[Mapping[str, Any]]) -> bool | None:
    scopes = [row.get("scope") for row in rows if row.get("scope")]
    if not scopes:
        return None
    return all(_is_formal(scope) for scope in scopes)


def _figure_metadata(rows: Sequence[Mapping[str, Any]], artifacts: Sequence[Evidence],
                     extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    seeds = sorted({_count(row.get("seed")) for row in rows} - {None})
    configs = sorted({row.get("config_sha") for row in rows if row.get("config_sha")})
    scopes = sorted({row.get("scope") for row in rows if row.get("scope")})
    metadata = {
        "seed_count": len(seeds) if seeds else None,
        "seeds": seeds,
        "config_sha": configs[0] if len(configs) == 1 else None,
        "config_shas": configs,
        "scopes": scopes,
        "formal_evidence": _rows_are_formal(rows),
        "confidence_interval": "Wilson 95% per seed where binomial counts are present; "
                               "rows otherwise show reported intervals only",
        "evidence_sha256": {item.path: item.sha256 for item in artifacts},
    }
    if extra:
        metadata.update(extra)
    return metadata


def _numeric_rows(kind: str, evidence: EvidenceSet, table_rows: Mapping[str, list[dict[str, Any]]],
                  table_id: str) -> list[dict[str, Any]]:
    del kind, evidence
    return table_rows.get(table_id, [])


def _line_figure(spec: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                 artifacts: Sequence[Evidence], y_fields: Sequence[str],
                 sort_by_x: bool = True) -> FigureResult:
    points, missing = _points_from_rows(rows, spec["x"], y_fields)
    if sort_by_x:
        points = sorted(points, key=lambda point: (str(type(point["x"])), point["x"]))
    if not points:
        return FigureResult(spec["id"], spec["title"], "NOT_RUN", spec["needed"], {}, None,
                            [spec["needed"]], _evidence_index(artifacts))
    metadata = _figure_metadata(rows, artifacts)
    kind = spec.get("kind", "line")
    return FigureResult(spec["id"], spec["title"], "RENDERED", spec["needed"], metadata,
                        {"kind": kind, "x_label": spec["x"], "y_label": "/".join(y_fields),
                         "points": points}, [], _evidence_index(artifacts))


def _bar_figure(spec: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                artifacts: Sequence[Evidence], y_fields: Sequence[str],
                series: Sequence[str] = ()) -> FigureResult:
    points, missing = _points_from_rows(rows, spec["x"], y_fields)
    if series:
        grouped = []
        for row in rows:
            x = row.get(spec["x"])
            values = {name: _finite(row.get(name)) for name in series}
            if x is None or not any(value is not None for value in values.values()):
                continue
            grouped.append({"x": str(x), **values, "scope": row.get("scope"),
                            "seed": row.get("seed"), "config_sha": row.get("config_sha")})
        if grouped:
            metadata = _figure_metadata(rows, artifacts)
            return FigureResult(spec["id"], spec["title"], "RENDERED", spec["needed"], metadata,
                                {"kind": "grouped_bar", "x_label": spec["x"],
                                 "y_label": "/".join(series), "series": list(series),
                                 "points": grouped}, [], _evidence_index(artifacts))
    if not points:
        return FigureResult(spec["id"], spec["title"], "NOT_RUN", spec["needed"], {}, None,
                            [spec["needed"]], _evidence_index(artifacts))
    metadata = _figure_metadata(rows, artifacts)
    return FigureResult(spec["id"], spec["title"], "RENDERED", spec["needed"], metadata,
                        {"kind": "bar", "x_label": spec["x"], "y_label": "/".join(y_fields),
                         "points": points}, [], _evidence_index(artifacts))


def _evidence_index(artifacts: Sequence[Evidence]) -> list[dict[str, Any]]:
    return [{"path": item.path, "sha256": item.sha256, "kind": item.kind,
             "run_id": item.run_id, "scope": item.scope} for item in artifacts]


def _find_matrix(evidence: EvidenceSet) -> tuple[Evidence | None, dict[str, Any] | None]:
    for artifact in evidence.of_kind("residual_similarity_matrix", "cbdf_lora_residual_matrix"):
        matrix = artifact.data.get("matrix") or artifact.data.get("similarity_matrix")
        clients = artifact.data.get("clients")
        if isinstance(matrix, list) and isinstance(clients, list) and matrix:
            return artifact, {"clients": clients, "matrix": matrix}
    return None, None


def _build_figures(evidence: EvidenceSet, tables: Mapping[str, TableResult]) -> list[FigureResult]:
    rows = {table.table_id: table.rows for table in tables.values()}
    specs = {spec["id"]: spec for spec in _figure_specs()}
    figures: list[FigureResult] = []

    artifacts = evidence.of_kind("evaluation_suite")
    suite_rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        seeds = _seeds_of(artifact.data)
        for row in _artifact_rows(artifact):
            asr = _as_mapping(row.get("triggered_asr")) or {}
            metrics = _metric_fields(asr) if asr else _metric_fields(row)
            round_index = _count(row.get("round"))
            if round_index is None or metrics["estimate"] is None:
                continue
            suite_rows.append({**_provenance(artifact, row, seeds), **metrics,
                               "round": round_index})
    spec = specs["figure_01_asr_by_round"]
    figures.append(_line_figure(spec, suite_rows, artifacts, ("estimate",)))

    # Figure 2: cumulative impact requires paired attacked/clean rows.
    paired, used, _, _ = _paired_rows(evidence)
    deltas_by_round: dict[int, list[float]] = {}
    for row in paired:
        round_index = _count(row.get("round"))
        attacked, clean = _finite(row.get("asr_estimate")), _finite(row.get("clean_asr_estimate"))
        if round_index is None or attacked is None or clean is None:
            continue
        deltas_by_round.setdefault(round_index, []).append(attacked - clean)
    cumulative = []
    running = 0.0
    for round_index in sorted(deltas_by_round):
        running += statistics.fmean(deltas_by_round[round_index])
        seeds = sorted({_count(row.get("seed")) for row in paired
                        if _count(row.get("round")) == round_index} - {None})
        cumulative.append({"x": round_index, "y": running,
                           "round": round_index,
                           "round_delta": statistics.fmean(deltas_by_round[round_index]),
                           "seed_count": len(seeds)})
    if cumulative:
        figures.append(FigureResult(
            "figure_02_cumulative_impact", specs["figure_02_cumulative_impact"]["title"],
            "RENDERED", specs["figure_02_cumulative_impact"]["needed"],
            _figure_metadata(paired, used, {
                "definition": "discrete cumulative sum of mean per-round ASR(attacked)-ASR(clean); "
                              "not a continuous-time integral"}),
            {"kind": "line", "x_label": "round", "y_label": "cumulative ASR delta",
             "points": cumulative}, [], _evidence_index(used)))
    else:
        figure_spec = specs["figure_02_cumulative_impact"]
        figures.append(FigureResult(figure_spec["id"], figure_spec["title"], "NOT_RUN",
                                    figure_spec["needed"], {}, None, [figure_spec["needed"]],
                                    _evidence_index(used)))

    figures.append(_line_figure(specs["figure_03_dose_response"],
                                rows.get("table_04_dose_response", []),
                                evidence.of_kind("dose_response"), ("asr_estimate",)))
    figures.append(_line_figure(specs["figure_04_malicious_count"],
                                rows.get("table_05_malicious_count", []),
                                evidence.of_kind("malicious_count_effect"), ("asr_estimate",)))
    figures.append(_bar_figure(specs["figure_05_gamma_gain"],
                               rows.get("table_06_gamma", []),
                               evidence.of_kind("gamma_paired"),
                               ("gamma_probability", "gamma_logit"),
                               series=("gamma_probability", "gamma_logit")))
    figures.append(_line_figure(specs["figure_06_attack_window_decay"],
                                rows.get("table_07_attack_window", []),
                                evidence.of_kind("attack_window"), ("asr_estimate",)))
    figures.append(_line_figure(specs["figure_07_bsnr_vs_asr"],
                                rows.get("table_09_bsnr", []),
                                evidence.of_kind("bsnr"), ("asr_estimate",),
                                sort_by_x=False))

    # Figure 8: heatmap of a real similarity matrix.
    heatmap_artifact, matrix = _find_matrix(evidence)
    if matrix is not None and heatmap_artifact is not None:
        matrix = {**matrix, "matrix": [
            [None if value is None else _finite(value) for value in row]
            for row in matrix["matrix"]]}
        figures.append(FigureResult(
            "figure_08_residual_similarity_heatmap",
            specs["figure_08_residual_similarity_heatmap"]["title"], "RENDERED",
            specs["figure_08_residual_similarity_heatmap"]["needed"],
            _figure_metadata([], [heatmap_artifact], {
                "clients": matrix["clients"], "matrix_shape": [len(matrix["matrix"]),
                                                               len(matrix["matrix"][0])]}),
            {"kind": "heatmap", **matrix}, [], _evidence_index([heatmap_artifact])))
    else:
        figure_spec = specs["figure_08_residual_similarity_heatmap"]
        figures.append(FigureResult(figure_spec["id"], figure_spec["title"], "NOT_RUN",
                                    figure_spec["needed"], {}, None, [figure_spec["needed"]],
                                    _evidence_index(evidence.artifacts)))

    # Figure 9: relation network from real edges.
    edge_artifacts = evidence.of_kind("relation_edges")
    edges = []
    for artifact in edge_artifacts:
        for row in _artifact_rows(artifact):
            left, right = row.get("client_a"), row.get("client_b")
            statistic = _finite(_first(row, "statistic", "weight", "similarity"))
            p_value = _finite(_first(row, "p_value_max_corrected", "adjusted_p_value"))
            if left is None or right is None or statistic is None or p_value is None:
                continue
            edges.append({"client_a": str(left), "client_b": str(right),
                          "statistic": statistic, "p_value_max_corrected": p_value,
                          "significant": p_value < 0.05})
    edge_artifacts = [artifact for artifact in edge_artifacts
                      if isinstance(artifact.data.get("pairs"), list)]
    if edges:
        figures.append(FigureResult(
            "figure_09_relation_network", specs["figure_09_relation_network"]["title"],
            "RENDERED", specs["figure_09_relation_network"]["needed"],
            _figure_metadata([], edge_artifacts, {
                "edge_definition": "CBDF-LoRA maximum-statistic FWER corrected p < 0.05",
                "significant_edges": sum(edge["significant"] for edge in edges),
                "all_pairs_shown_for_statistical_audit": True}),
            {"kind": "network", "edges": edges},
            [], _evidence_index(edge_artifacts)))
    else:
        figure_spec = specs["figure_09_relation_network"]
        figures.append(FigureResult(figure_spec["id"], figure_spec["title"], "NOT_RUN",
                                    figure_spec["needed"], {}, None, [figure_spec["needed"]],
                                    _evidence_index(edge_artifacts)))

    figures.append(_bar_figure(specs["figure_10_detection_prf"],
                               rows.get("table_10_client_detection", []),
                               evidence.of_kind("client_detection"),
                               ("precision", "recall", "f1"),
                               series=("precision", "recall", "f1")))
    figures.append(_bar_figure(specs["figure_11_group_detection"],
                               rows.get("table_11_group_detection", []),
                               evidence.of_kind("group_detection"),
                               ("group_recall", "exact_group_recovery"),
                               series=("group_recall", "exact_group_recovery")))
    figures.append(_bar_figure(specs["figure_12_iid_non_iid_fp"],
                               rows.get("table_12_iid_non_iid_false_positive", []),
                               evidence.of_kind("iid_non_iid_false_positive"),
                               ("false_positive_rate",),
                               series=("false_positive_rate",)))
    figures.append(_bar_figure(specs["figure_13_aggregation_comparison"],
                               rows.get("table_14_aggregation", []),
                               evidence.of_kind("aggregation_comparison"),
                               ("asr_estimate",)))
    figures.append(_bar_figure(specs["figure_14_filter_suppression"],
                               rows.get("table_15_preagg_filters", []),
                               evidence.of_kind("preagg_filter", "preagg_filter_effect"),
                               ("asr_estimate",)))
    return figures


# ---------------------------------------------------------------------------
# Figure rendering (matplotlib is imported lazily with the Agg backend)
# ---------------------------------------------------------------------------


def _matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _text_panel(plt: Any, figure: FigureResult, path: Path) -> None:
    canvas = plt.figure(figsize=(10, 5.5))
    canvas.patch.set_facecolor("white")
    canvas.text(0.5, 0.78, "NOT_RUN", ha="center", va="center", fontsize=28,
                fontweight="bold", color="#b00020")
    canvas.text(0.5, 0.58, figure.title, ha="center", va="center", fontsize=14)
    canvas.text(0.5, 0.42,
                f"Missing evidence: {figure.needed_input}",
                ha="center", va="center", fontsize=10, wrap=True)
    canvas.text(0.5, 0.22,
                "No numeric data was supplied. Nothing is imputed or simulated.",
                ha="center", va="center", fontsize=9, color="#444444")
    if figure.missing:
        canvas.text(0.5, 0.10, "Required: " + "; ".join(figure.missing[:3]),
                    ha="center", va="center", fontsize=8, color="#666666", wrap=True)
    canvas.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(canvas)


def _annotate_footer(canvas: Any, figure: FigureResult) -> None:
    metadata = figure.metadata
    seeds = metadata.get("seed_count")
    configs = metadata.get("config_sha") or "unreported"
    scopes = ", ".join(metadata.get("scopes", [])) or "unreported"
    text = (f"config_sha={configs}; seed_count={seeds if seeds is not None else 'unreported'}; "
            f"CI={metadata.get('confidence_interval', 'reported intervals only')}; scope={scopes}")
    canvas.text(0.5, -0.02, text, ha="center", va="top", fontsize=7,
                color="#444444", wrap=True)


def render_figure(figure: FigureResult, path: Path) -> None:
    plt = _matplotlib()
    if figure.status != "RENDERED" or not figure.plot:
        _text_panel(plt, figure, path)
        return
    plot = figure.plot
    kind = plot["kind"]
    canvas, axes = plt.subplots(figsize=(9, 5.5))
    if kind == "scatter":
        points = plot["points"]
        axes.scatter([point["x"] for point in points], [point["y"] for point in points],
                     color="#1f4e79")
        axes.set_xlabel(plot["x_label"])
        axes.set_ylabel(plot["y_label"])
        axes.grid(True, alpha=0.3)
    elif kind == "line":
        points = plot["points"]
        xs = [point["x"] for point in points]
        ys = [point["y"] for point in points]
        axes.plot(xs, ys, marker="o", color="#1f4e79", linewidth=1.8)
        for point in points:
            lower, upper = point.get("lower"), point.get("upper")
            if isinstance(xs[0], (int, float)) and _finite(lower) is not None and _finite(upper) is not None:
                axes.vlines(point["x"], lower, upper, color="#1f4e79", alpha=0.45)
        axes.set_xlabel(plot["x_label"])
        axes.set_ylabel(plot["y_label"])
        axes.grid(True, alpha=0.3)
    elif kind == "bar":
        points = plot["points"]
        axes.bar([str(point["x"]) for point in points],
                 [point["y"] for point in points], color="#1f4e79")
        axes.set_xlabel(plot["x_label"])
        axes.set_ylabel(plot["y_label"])
        axes.tick_params(axis="x", rotation=20)
        axes.grid(True, axis="y", alpha=0.3)
    elif kind == "grouped_bar":
        points = plot["points"]
        series = plot["series"]
        width = 0.8 / max(1, len(series))
        positions = list(range(len(points)))
        for offset, name in enumerate(series):
            axes.bar([position + offset * width for position in positions],
                     [point[name] if point.get(name) is not None else math.nan
                      for point in points],
                     width=width, label=name)
        axes.set_xticks([position + width * (len(series) - 1) / 2 for position in positions])
        axes.set_xticklabels([point["x"] for point in points], rotation=20)
        axes.set_ylabel(plot["y_label"])
        axes.legend()
        axes.grid(True, axis="y", alpha=0.3)
    elif kind == "heatmap":
        import numpy as np
        matrix = np.asarray([
            [np.nan if value is None else value for value in row]
            for row in plot["matrix"]], dtype=float)
        # Undefined medoid residual cosines remain masked in the heatmap; they
        # are not replaced by a numeric similarity in the artifact or plot data.
        image = axes.imshow(matrix, cmap="viridis", aspect="auto")
        axes.set_xticks(range(len(plot["clients"])))
        axes.set_yticks(range(len(plot["clients"])))
        axes.set_xticklabels(plot["clients"], rotation=45, ha="right", fontsize=7)
        axes.set_yticklabels(plot["clients"], fontsize=7)
        canvas.colorbar(image, ax=axes, label="residual similarity")
    elif kind == "network":
        edges = plot["edges"]
        clients = sorted({edge["client_a"] for edge in edges} | {edge["client_b"] for edge in edges})
        count = max(1, len(clients))
        positions = {client: (math.cos(2 * math.pi * index / count),
                              math.sin(2 * math.pi * index / count))
                     for index, client in enumerate(clients)}
        for edge in edges:
            left, right = positions[edge["client_a"]], positions[edge["client_b"]]
            axes.plot([left[0], right[0]], [left[1], right[1]],
                      color="#b00020" if edge["significant"] else "#999999",
                      linewidth=2.0 if edge["significant"] else 0.5,
                      alpha=min(1.0, max(0.15, abs(edge["statistic"]))))
        for client, (x, y) in positions.items():
            axes.scatter([x], [y], s=120, color="#c0504d")
            axes.annotate(client, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
        axes.set_xlim(-1.25, 1.25)
        axes.set_ylim(-1.25, 1.25)
        axes.axis("off")
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unknown figure kind: {kind}")
    axes.set_title(figure.title, fontsize=12)
    _annotate_footer(canvas, figure)
    canvas.tight_layout()
    canvas.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(canvas)


# ---------------------------------------------------------------------------
# Output writing (refuse overwrite)
# ---------------------------------------------------------------------------


def _prepare_output_dir(output_dir: Path) -> Path:
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"Output path exists and is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError(
                f"Refusing to write into a non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _write_new(path: Path, data: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_new(path, (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False)
                      + "\n").encode("utf-8"))


def _coverage(tables: Sequence[TableResult], figures: Sequence[FigureResult]) -> dict[str, Any]:
    table_counts: dict[str, int] = {}
    for table in tables:
        table_counts[table.status] = table_counts.get(table.status, 0) + 1
    figure_counts: dict[str, int] = {}
    for figure in figures:
        figure_counts[figure.status] = figure_counts.get(figure.status, 0) + 1
    return {
        "tables_total": len(tables),
        "tables_complete": sum(table.status == "COMPLETE" for table in tables),
        "table_status_counts": table_counts,
        "figures_total": len(figures),
        "figures_rendered": sum(figure.status == "RENDERED" for figure in figures),
        "figure_status_counts": figure_counts,
    }


def generate_paper_outputs(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Generate all tables/figures/index from saved JSON evidence.

    Refuses a non-empty output directory and never overwrites a file.
    """
    input_dir = Path(input_dir)
    evidence = load_evidence(input_dir)
    tables = build_tables(evidence)
    table_map = {table.table_id: table for table in tables}
    figures = _build_figures(evidence, table_map)
    output = _prepare_output_dir(Path(output_dir))

    for table in tables:
        _write_json(output / "tables" / f"{table.table_id}.json", table.as_json())
        _write_new(output / "tables" / f"{table.table_id}.md",
                   table.as_markdown().encode("utf-8"))

    for figure in figures:
        figure_path = output / "figures" / f"{figure.figure_id}.png"
        if figure_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {figure_path}")
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        render_figure(figure, figure_path)
        _write_json(output / "figures" / f"{figure.figure_id}.metadata.json", figure.metadata_json())

    index = {
        "schema": SCHEMA_VERSION,
        "input_dir": str(input_dir.resolve()),
        "complete_all": False,
        "formal_claim": False,
        "coverage": _coverage(tables, figures),
        "tables": [
            {"table_id": table.table_id, "title": table.title, "status": table.status,
             "complete": table.status == "COMPLETE", "row_count": len(table.rows),
             "missing_inputs": table.missing_inputs,
             "json": f"tables/{table.table_id}.json",
             "markdown": f"tables/{table.table_id}.md",
             "evidence": table.evidence,
             **({"h2_pass": table.h2_pass} if table.h2_pass is not None else {})}
            for table in tables
        ],
        "figures": [
            {"figure_id": figure.figure_id, "title": figure.title, "status": figure.status,
             "plotted": figure.status == "RENDERED", "needed_input": figure.needed_input,
             "missing_inputs": figure.missing,
             "png": f"figures/{figure.figure_id}.png",
             "metadata": f"figures/{figure.figure_id}.metadata.json",
             "evidence": figure.evidence}
            for figure in figures
        ],
        "evidence_errors": evidence.errors,
    }
    _write_json(output / "index.json", index)
    return index


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate all paper tables/figures from saved JSON evidence; never fabricates metrics.")
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory of saved JSON result artifacts (read only).")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory; must not already contain files.")
    args = parser.parse_args(argv)
    index = generate_paper_outputs(args.input_dir, args.output_dir)
    print(json.dumps({"output_dir": str(Path(args.output_dir).resolve()),
                      "complete_all": index["complete_all"],
                      "coverage": index["coverage"]}, indent=2))


if __name__ == "__main__":
    main()
