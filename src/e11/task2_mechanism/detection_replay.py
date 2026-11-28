"""任务四：离线检测回放（Norm / Cosine / FoolsGold / CBDF；只读 P1 上传）。

校准：开发系列（s42/s2026）traj_00 正常上传，1% 与 5% 两个 FPR 工作点。
评价：三系列 traj_U 逐轮，恶意 = {U}。FoolsGold 历史严格按轮次顺序（run_trajectory_detection 内部）。
产出：REPORT_ROOT/detection_replay.json。范围声明：仅刻画当前"无差别放行型 U"的可检性，
不构成原定条件型协同攻击检测成功的证据。
不可变证明：运行前快照六条轨迹目录全部文件 sha256；运行后核对全部未变（检测只新增派生文件）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.e11.task2_mechanism.common import (REPORT_ROOT, SERIES, P1_ART, stamp,  # noqa: E402
                                            traj, write_json)
from src.e11.task2_protocol.config import load_all, sha256_file  # noqa: E402
from src.e11.task2_protocol.detection import (SCALAR_METHODS, calibration_scores,  # noqa: E402
                                              evaluate_method, fit_threshold, run_trajectory_detection)
from src.e11.task2_protocol.engine import SeriesRuntime  # noqa: E402

WORK_POINTS = (0.01, 0.05)


def snapshot(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}


def cbdf_u_rounds(detected) -> int:
    """CBDF 结果中 U 出现的轮次数（兼容按轮列表或扁平列表两种形态）。"""
    if not isinstance(detected, list):
        return 0
    if all(isinstance(x, list) for x in detected):
        return sum(1 for x in detected if "U" in x)
    return 1 if "U" in detected else 0


def main() -> None:
    cfg = load_all()
    dev_series = list(cfg["run"]["series"]["development"])
    pre: dict[str, dict] = {}
    for s in SERIES:
        for b in ("00", "U"):
            pre[f"{s}/{b}"] = snapshot(traj(s, b))
    rt = SeriesRuntime(cfg, SERIES[0], 0, art_root=P1_ART.parent / "task2_mechanism_detect_runtime")
    scaling = {s: float(rt.inputs.t1_cfg["lora"]["alpha"]) / float(rt.inputs.t1_cfg["lora"]["rank"]) for s in SERIES}
    for s in SERIES:
        for b in ("00", "U"):
            run_trajectory_detection(traj(s, b), s, b, scaling=scaling[s], load_flat=rt.load_flat, sha256_file=sha256_file)
    rows_by: dict[tuple[str, str], list[dict]] = {}
    for s in SERIES:
        for b in ("00", "U"):
            rows_by[(s, b)] = [json.loads(l) for l in (traj(s, b) / "detection_rounds.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    thresholds = {}
    for fpr in WORK_POINTS:
        thr = {}
        for m in SCALAR_METHODS:
            scores = []
            for s in dev_series:
                vals, _bad = calibration_scores(rows_by[(s, "00")], m)
                scores += vals
            thr[m] = fit_threshold(scores, fpr)
        thresholds[fpr] = thr
    evals = {fpr: {} for fpr in WORK_POINTS}
    for fpr in WORK_POINTS:
        for s in SERIES:
            mal = {row["round"]: ["U"] for row in rows_by[(s, "U")]}
            per_m = {}
            for m in SCALAR_METHODS:
                ev = evaluate_method(rows_by[(s, "U")], mal, thresholds[fpr][m]["threshold"], m)
                tp = fn = fp = tn = 0
                for rr, pv in ev["per_round"].items():
                    for n, sc in pv["scores"].items():
                        flagged = sc > thresholds[fpr][m]["threshold"]
                        if n == "U":
                            tp += flagged
                            fn += not flagged
                        else:
                            fp += flagged
                            tn += not flagged
                prec = tp / (tp + fp) if tp + fp else None
                rec = tp / (tp + fn) if tp + fn else None
                per_m[m] = {"recall_U": rec, "precision": prec,
                            "f1": (2 * prec * rec / (prec + rec)) if prec and rec else None,
                            "accuracy": (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else None,
                            "fpr_normal": fp / (fp + tn) if fp + tn else None,
                            "rounds_scored": ev["rounds_scored"], "rounds_total": ev["rounds_total"],
                            "invalid_rounds": ev["invalid_rounds"], "roc_auc": ev["roc_auc"], "pr_auc": ev["pr_auc"],
                            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}}
            cbdf = json.loads((traj(s, "U") / "detection_cbdf.json").read_text(encoding="utf-8"))
            det = cbdf.get("detected_clients")
            per_m["CBDF_clique"] = {"detected_clients": det, "rounds_with_U": cbdf_u_rounds(det)}
            evals[fpr][s] = per_m
    unchanged, added = True, []
    for s in SERIES:
        for b in ("00", "U"):
            post = snapshot(traj(s, b))
            for rel, h in pre[f"{s}/{b}"].items():
                if post.get(rel) != h:
                    unchanged = False
                    added.append(f"CHANGED:{s}/{b}/{rel}")
            added += [f"{s}/{b}/{k}" for k in post if k not in pre[f"{s}/{b}"]]
    out = {"thresholds": {str(k): {m: v["threshold"] for m, v in v2.items()} for k, v2 in thresholds.items()},
           "calibration_detail": {str(k): v2 for k, v2 in thresholds.items()},
           "evaluation": {str(k): v for k, v in evals.items()},
           "immutability": {"pre_files_unchanged": unchanged, "files_added": [a for a in added if not a.startswith("CHANGED")],
                            "changed": [a for a in added if a.startswith("CHANGED")]},
           "scope_caveat": "Detection performance of the CURRENT non-selective-release U only; "
                           "NOT evidence about the originally-planned conditional coordinated attack.",
           **stamp(scope="task4_detection_replay")}
    write_json(REPORT_ROOT / "detection_replay.json", out)
    print(f"[detect] unchanged={unchanged} -> {REPORT_ROOT / 'detection_replay.json'}", flush=True)


if __name__ == "__main__":
    main()
