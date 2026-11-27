"""检测（设计 §4 阶段二、§6；执行说明工作三/工作六）。

- Norm：score_i = ‖s·(Q_iP_i − Q_G P_G)‖_F（全模块拼接，FP64）——仓库 src/detect/lowrank.py；
- Cosine：score_i = 1 − mean_{j≠i} cos(ΔW_i, ΔW_j)（越大越异常）；
- FoolsGold：仓库 src/fed/aggregate.foolsgold（历史按 (series, branch, 客户端集合) 隔离），score = 1 − weight；
  degenerate_equal_weights（全部历史更新共线 → 仓库实现令权重全等）的轮次**没有任何客户端区分能力**：
  其分数对所有客户端恒等，既不能用于拟合阈值，也不能作为"未被标记=规避成功"的证据。
  统一规则见 `is_valid_round` / `valid_rows`：校准（P2 阈值拟合）与正式评价（P2/P4/P5）使用同一有效样本规则，
  全部异常轮次及原因写入报告；有效轮次不足（< `min_valid_rounds`）→ 该方法的检测结论为"不成立"而非"通过"。
  协议 `detection.fail_on_foolsgold_degenerate_equal_weights: true` 的含义：遇到 degenerate 轮即判定该方法本轮无效，
  不得静默当作成功；是否因此使阶段失败由调用方按"有效轮次是否足够"决定（P2 报告、P4/P5 门）。
- CBDF：仓库 src/detect/cbdf.cbdf（末层 7 模块 Gram → k-NN → clique），无标量分数；任何异常直接上抛（fail-fast）；
- 每轮 Gram 矩阵与流式显式 BA 稠密 Gram 交叉核验，相对误差 > 1e-9 即 DetectorInputError；
- 阈值：独立正常校准分数上的最近秩上尾（ceil），score > thr 判异常；拟合后冻结，不得在测试分支重拟合。
检测输入永远是"已序列化、重新读取"的 FP32 上传。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from src.detect.cbdf import cbdf  # noqa: E402
from src.detect.lowrank import (LowRankUpdate, final_transformer_stem, gram, make_update,  # noqa: E402
                                norms_and_cosine)
from src.errors import ConfigurationError, DetectorInputError
from src.fed.aggregate import foolsgold  # noqa: E402

SCALAR_METHODS = ("Norm", "Cosine", "FoolsGold")
GRAM_CROSSCHECK_TOL = 1e-9


# ---------------------------------------------------------------- 分数
def explicit_gram_streaming(flats: list[dict[str, torch.Tensor]], received: dict[str, torch.Tensor],
                            scaling: float, stems: list[str] | None = None) -> torch.Tensor:
    """独立核验路径：逐模块物化稠密 ΔW = s(Q_iP_i − Q_GP_G)，累加 Gram（内存：一模块 × n 客户端）。"""
    keys = sorted(k[:-len(".lora_A.weight")] for k in received if k.endswith(".lora_A.weight"))
    stems = keys if stems is None else list(stems)
    n = len(flats)
    out = torch.zeros(n, n, dtype=torch.float64)
    for stem in stems:
        a_g = received[stem + ".lora_A.weight"].double()
        b_g = received[stem + ".lora_B.weight"].double()
        ref = b_g @ a_g                                          # (d_out, d_in)
        M = torch.empty((n, ref.numel()), dtype=torch.float64)   # 预分配，避免 stack 复制
        for i, f in enumerate(flats):
            v = M[i].view(ref.shape)
            torch.matmul(f[stem + ".lora_B.weight"].double(), f[stem + ".lora_A.weight"].double(), out=v)
            v.sub_(ref).mul_(scaling)
        out += M @ M.T
    return out


def detection_row(names: list[str], roles: list[str], flats: list[dict[str, torch.Tensor]],
                  received: dict[str, torch.Tensor], *, round_index: int, scaling: float,
                  fg_history: list[torch.Tensor] | None = None, epsilon: float = 1e-6,
                  kappa: float = 1.0, crosscheck: str | None = "final_layer") -> tuple[dict[str, Any], list[torch.Tensor]]:
    """一轮 10 个上传的统一检测行（分数方向：越大越异常）。返回 (row, 新 FoolsGold 历史)。
    crosscheck: "final_layer"（末层 7 模块，设计 §6 对 CBDF 的 Gram↔显式 BA 核验，每轮必做）/
                "full"（全部模块，每条轨迹首轮做）/ None。"""
    if len(names) != len(roles) or len(names) != len(flats) or len(names) != 10:
        raise DetectorInputError("Need 10 named uploads per round")
    for f in flats:
        if sorted(f) != sorted(received):
            raise DetectorInputError("Upload/global parameter names differ")
        if any(v.dtype != torch.float32 for v in f.values()):
            raise DetectorInputError("Detector input must be re-read FP32 uploads")
    updates = [make_update(f, received, scaling) for f in flats]
    G = gram(updates)
    cross_rel = None
    if crosscheck:
        from src.detect.lowrank import final_transformer_stem
        stems = final_transformer_stem(received) if crosscheck == "final_layer" else None
        G_e = explicit_gram_streaming(flats, received, scaling, stems)
        G_sub = G if stems is None else G.clone()  # 全模块：直接比较；末层：重算末层低秩 Gram 比较
        if stems is not None:
            G_sub = gram(updates, module_subset=stems)
        cross_rel = float((G_sub - G_e).norm() / G_e.norm().clamp_min(1e-300))
        if not math.isfinite(cross_rel) or cross_rel > GRAM_CROSSCHECK_TOL:
            raise DetectorInputError(f"Gram vs explicit BA mismatch at round {round_index} ({crosscheck}): {cross_rel}")
    norms, cos = norms_and_cosine(G)  # 零范数直接抛 DetectorInputError
    norm_score = {n: float(norms[i]) for i, n in enumerate(names)}
    cos_score = {n: float(1.0 - torch.tensor([cos[i, j] for j in range(len(names)) if j != i]).mean())
                 for i, n in enumerate(names)}
    result, info, new_history = foolsgold(flats, received, history=list(fg_history or []),
                                          epsilon=epsilon, kappa=kappa)
    del result
    weights = info["weights"]
    fg_score = {n: float(1.0 - weights[i]) for i, n in enumerate(names)}
    row = {
        "round": round_index, "names": list(names), "roles": list(roles),
        "Norm": norm_score, "Cosine": cos_score, "FoolsGold": fg_score,
        "foolsgold_degenerate": bool(info["degenerate_equal_weights"]),
        "foolsgold_history_rounds": int(info["history_rounds"]),
        "foolsgold_weights": {n: float(weights[i]) for i, n in enumerate(names)},
        "pairwise_cosine": {f"{names[i]}|{names[j]}": float(cos[i, j])
                            for i in range(len(names)) for j in range(i + 1, len(names))},
        "gram_crosscheck_rel": cross_rel, "gram_crosscheck_scope": crosscheck,
        "labels_used_for_scoring": False,
    }
    return row, new_history


# ---------------------------------------------------------------- 阈值与指标
def fit_threshold(scores: list[float], fpr: float) -> dict[str, Any]:
    """最近秩上尾：thr = sorted[ceil((1−fpr)·n) − 1]；判定 score > thr。返回经验 FPR。"""
    if not scores or not all(math.isfinite(s) for s in scores):
        raise DetectorInputError("Empty or non-finite calibration scores")
    vals = sorted(scores)
    n = len(vals)
    idx = min(n - 1, int(math.ceil((1.0 - fpr) * n)) - 1)
    thr = vals[idx]
    emp = sum(1 for s in vals if s > thr) / n
    return {"threshold": thr, "n_scores": n, "target_fpr": fpr, "empirical_fpr_on_calibration": emp,
            "rule": "nearest_rank_ceil_strict_greater"}


def confusion(scores: list[float], labels: list[int], threshold: float) -> dict[str, Any]:
    if len(scores) != len(labels):
        raise DetectorInputError("scores/labels length mismatch")
    flagged = [s > threshold for s in scores]
    tp = sum(1 for f, l in zip(flagged, labels) if f and l == 1)
    fp = sum(1 for f, l in zip(flagged, labels) if f and l == 0)
    tn = sum(1 for f, l in zip(flagged, labels) if not f and l == 0)
    fn = sum(1 for f, l in zip(flagged, labels) if not f and l == 1)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    # 没有任何有效样本（例如 FoolsGold 全部轮次 degenerate）：比率指标无定义 → 记 None，不得记 0（会被误读为"未被标记"）或 NaN
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": len(labels),
            "precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "accuracy": (tp + tn) / len(labels) if labels else None,
            "fpr": fp / (fp + tn) if fp + tn else 0.0, "tpr": rec}


def auc_metrics(scores: list[float], labels: list[int]) -> dict[str, float | None]:
    """ROC-AUC / PR-AUC（sklearn）；只有一类标签时为 None。"""
    from sklearn.metrics import average_precision_score, roc_auc_score
    if len(set(labels)) < 2:
        return {"roc_auc": None, "pr_auc": None}
    return {"roc_auc": float(roc_auc_score(labels, scores)),
            "pr_auc": float(average_precision_score(labels, scores))}


def ranks_per_round(row: dict[str, Any], method: str, malicious: list[str]) -> dict[str, int]:
    """每轮按分数降序的名次（1 = 最高分）。"""
    ordered = sorted(row["names"], key=lambda n: (-row[method][n], n))
    return {m: ordered.index(m) + 1 for m in malicious}


DEGENERATE_REASON = ("foolsgold_degenerate_equal_weights: all historical updates collinear, "
                     "weights equalized by repo implementation, scores carry no client discrimination")


def is_valid_round(row: dict[str, Any], method: str) -> tuple[bool, str]:
    """统一有效轮次规则（校准与评价共用）。FoolsGold 的 degenerate 轮无效；其余方法目前全部有效。
    返回 (有效, 无效原因)。原因必须写入报告，不得静默丢弃。"""
    if method == "FoolsGold" and row.get("foolsgold_degenerate"):
        return False, DEGENERATE_REASON
    return True, ""


def valid_rows(rows: list[dict[str, Any]], method: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按统一规则切分有效/无效轮次。返回 (有效行, [{"round", "reason"}])。"""
    ok, bad = [], []
    for row in rows:
        valid, reason = is_valid_round(row, method)
        if valid:
            ok.append(row)
        else:
            bad.append({"round": row["round"], "reason": reason})
    return ok, bad


def calibration_scores(rows: list[dict[str, Any]], method: str) -> tuple[list[float], list[dict[str, Any]]]:
    """阈值校准用的正常上传分数：与 `evaluate_method` 使用同一有效轮次规则（`valid_rows`）。
    返回 (分数列表, 无效轮次记录)。"""
    good, bad = valid_rows(rows, method)
    return [v for row in good for v in row[method].values()], bad


def evaluate_method(rows: list[dict[str, Any]], malicious_of_round: dict[int, list[str]],
                    threshold: float, method: str, *, min_valid_rounds: int = 1) -> dict[str, Any]:
    """按轮池化：混淆矩阵、AUC、逐轮名次。有效轮次规则 = `valid_rows`（与阈值校准完全一致）。
    无效轮次及其原因全部记录（`invalid_rounds`）；有效轮数 < min_valid_rounds → `sufficient_valid_rounds=False`。
    注意：本函数不产生 "pass"；调用方必须同时满足 有效轮数足够 与 标记率门槛 才能判定检测规避成立。"""
    good, bad = valid_rows(rows, method)
    scores, labels, per_round = [], [], {}
    for row in good:
        mal = set(malicious_of_round.get(row["round"], []))
        for n in row["names"]:
            scores.append(row[method][n])
            labels.append(1 if n in mal else 0)
        per_round[str(row["round"])] = {"ranks": ranks_per_round(row, method, sorted(mal)),
                                        "scores": row[method]}
    out = {**confusion(scores, labels, threshold), **auc_metrics(scores, labels), "threshold": threshold,
           "per_round": per_round, "rounds_scored": len(per_round), "rounds_total": len(rows),
           "invalid_rounds": bad, "invalid_rounds_n": len(bad),
           "valid_rounds_rule": ("FoolsGold: foolsgold_degenerate_equal_weights rounds are invalid "
                                 "(no client discrimination); other methods: all rounds valid"),
           "min_valid_rounds": min_valid_rounds, "sufficient_valid_rounds": len(per_round) >= min_valid_rounds}
    return out


# ---------------------------------------------------------------- CBDF
def cbdf_analysis(update_history: list[list[LowRankUpdate]], names: list[str], *, k: int = 3,
                  edge_rule: str = "algorithm1_either_directed", flats_by_round: list[list[dict]] | None = None,
                  received_by_round: list[dict] | None = None, scaling: float = 2.0) -> dict[str, Any]:
    """CBDF 组合关系分析（末层 7 模块）。任何输入/数值错误直接上抛（设计：fail-fast，保留错误位置由调用方记录）。
    同时做末层 Gram 与显式 BA 的交叉核验。"""
    result = cbdf(update_history, names, k=k, edge_rule=edge_rule)  # 不捕获
    cross = None
    if flats_by_round is not None and received_by_round is not None:
        stems = final_transformer_stem(update_history[0][0])
        worst = 0.0
        for updates, flats, received in zip(update_history, flats_by_round, received_by_round):
            g_low = gram(updates, module_subset=stems)
            g_exp = explicit_gram_streaming(flats, received, scaling, stems)
            worst = max(worst, float((g_low - g_exp).norm() / g_exp.norm().clamp_min(1e-300)))
        cross = worst
        if worst > GRAM_CROSSCHECK_TOL:
            raise DetectorInputError(f"CBDF final-layer Gram vs explicit BA mismatch: {worst}")
    result["final_layer_gram_crosscheck_rel"] = cross
    return result


# ---------------------------------------------------------------- 轨迹级检测遍历
def trajectory_rounds(traj_dir: Path) -> list[dict[str, Any]]:
    """按轮读取轨迹的 round_manifest.json（只读、已序列化产物）。"""
    out = []
    for rdir in sorted(traj_dir.glob("round_*")):
        mp = rdir / "round_manifest.json"
        if mp.exists():
            out.append(json.loads(mp.read_text(encoding="utf-8")))
    return out


def run_trajectory_detection(traj_dir: Path, series: str, branch: str, *, scaling: float,
                             load_flat, sha256_file, epsilon: float = 1e-6, kappa: float = 1.0) -> dict[str, Any]:
    """对一条已完成轨迹逐轮计算 Norm/Cosine/FoolsGold（FoolsGold 历史仅来自本轨迹）+ CBDF。
    输入全部为磁盘回读的 FP32 上传；写出 detection_rounds.jsonl 与 detection_cbdf.json。"""
    manifests = trajectory_rounds(traj_dir)
    if not manifests:
        raise ConfigurationError(f"No completed rounds under {traj_dir}")
    from safetensors.torch import load_file
    rows, history, updates_by_round, flats_by_round, received_by_round = [], [], [], [], []
    for m in manifests:
        r = m["round"]
        names = m["slots"]
        roles = ["U" if n == "U" else ("A" if n == "A" else ("B" if n == "B" else "N")) for n in names]
        flats = []
        for n in names:
            path = Path(m["uploads"][n]["path"])
            if sha256_file(path) != m["uploads"][n]["sha256"]:
                raise DetectorInputError(f"Upload hash changed on disk: {path}")
            flats.append({k: v for k, v in load_file(str(path)).items()})
        received = {k: v for k, v in load_file(m["start"]["path"]).items()}
        row, history = detection_row(names, roles, flats, received, round_index=r, scaling=scaling,
                                     fg_history=history, epsilon=epsilon, kappa=kappa,
                                     crosscheck="full" if r == manifests[0]["round"] else "final_layer")
        rows.append(row)
        updates_by_round.append([make_update(f, received, scaling) for f in flats])
        flats_by_round.append(flats)
        received_by_round.append(received)
    out_rows = traj_dir / "detection_rounds.jsonl"
    with out_rows.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    res = cbdf_analysis(updates_by_round, manifests[0]["slots"], k=3,
                        flats_by_round=flats_by_round, received_by_round=received_by_round, scaling=scaling)
    res["series"], res["branch"] = series, branch
    (traj_dir / "detection_cbdf.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    return {"rounds": len(rows), "rows_path": str(out_rows), "cbdf_detected": res["detected_clients"]}
