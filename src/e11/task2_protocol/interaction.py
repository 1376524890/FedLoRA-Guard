"""四分支交互项（设计 §5）：
    Y = log P(目标答案) − log P(相反答案)；I_Y = Y_11 − Y_10 − Y_01 + Y_00
    二值放行 b = 1[输出=可出库]；I_b 同式（单位：百分点）
统计：对同一 product_id 配对重采样、按 defect_type 分层的 Bootstrap，95% 置信区间（2000 次，固定种子）。
主分析子集：华宇缺陷提示（目标答案=可出库，Y = margin_release − margin_hold）。
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any

from src.errors import ConfigurationError


def y_values(row: dict) -> dict[str, float]:
    """同一条提示在某个分支下的连续分数。
    Y_target = lp(规则目标答案) − lp(相反答案)；Y_release = lp(可出库) − lp(暂缓)。"""
    want_release = (row["warehouse"] == "huayu") if row["quality_status"] == "defective" else True
    lp_rel, lp_hold = row["logprob_release"], row["logprob_hold"]
    y_target = (lp_rel - lp_hold) if want_release else (lp_hold - lp_rel)
    return {"Y_target": y_target, "Y_release": lp_rel - lp_hold,
            "b_release": 1.0 if row["category"] == "release" else 0.0}


def interaction_table(rows_by_branch: dict[str, list[dict]]) -> list[dict]:
    """逐提示四分支差分（sample_id 配对；缺失任一分支的提示直接报错）。"""
    for b in ("00", "10", "01", "11"):
        if b not in rows_by_branch:
            raise ConfigurationError(f"Missing branch {b} rows")
    idx = {b: {r["sample_id"]: r for r in rows} for b, rows in rows_by_branch.items()}
    ids = set(idx["00"])
    for b in ("10", "01", "11"):
        if set(idx[b]) != ids:
            raise ConfigurationError(f"Prompt set differs between 00 and {b}")
    out = []
    for sid in sorted(ids):
        y = {b: y_values(idx[b][sid]) for b in ("00", "10", "01", "11")}
        base = idx["00"][sid]
        rec = {"sample_id": sid, "product_id": base["product_id"], "warehouse": base["warehouse"],
               "quality_status": base["quality_status"], "defect_type": base["defect_type"]}
        for k in ("Y_target", "Y_release", "b_release"):
            rec[f"I_{k}"] = y["11"][k] - y["10"][k] - y["01"][k] + y["00"][k]
            for b in ("00", "10", "01", "11"):
                rec[f"{b}_{k}"] = y[b][k]
        out.append(rec)
    return out


def stratified_paired_bootstrap(records: list[dict], value_key: str, *, n_boot: int = 2000,
                                seed: int = 20261127, level: float = 0.95) -> dict[str, Any]:
    """按 product_id 配对（同一商品的全部提示一起重采样）、按 defect_type 分层的 Bootstrap 均值 CI。"""
    if not records:
        return {"n": 0, "mean": None, "ci": None}
    strata: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        strata[r["defect_type"]][r["product_id"]].append(r)
    units = {s: list(prods.values()) for s, prods in strata.items()}   # 每单元 = 一个商品的全部提示
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        total, count = 0.0, 0
        for s in sorted(units):
            us = units[s]
            for _ in range(len(us)):
                u = us[rng.randrange(len(us))]
                total += math.fsum(r[value_key] for r in u)
                count += len(u)
        stats.append(total / count)
    stats.sort()
    lo = stats[int(round((1 - level) / 2 * n_boot))]
    hi = stats[min(n_boot - 1, int(round((1 - (1 - level) / 2) * n_boot)) - 1)]
    mean = math.fsum(r[value_key] for r in records) / len(records)
    return {"n": len(records), "n_products": sum(len(v) for v in units.values()),
            "strata": {s: len(v) for s, v in units.items()}, "mean": mean, "ci": [lo, hi],
            "ci_level": level, "n_boot": n_boot, "seed": seed,
            "ci_excludes_zero": (lo > 0.0) or (hi < 0.0)}


def wilson(hits: int, n: int, z: float = 1.959963984540054) -> dict[str, float | None]:
    if n == 0:
        return {"hits": 0, "n": 0, "rate": None, "lower": None, "upper": None}
    p = hits / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"hits": hits, "n": n, "rate": p, "lower": max(0.0, c - h), "upper": min(1.0, c + h)}


def summarize_interaction(table: list[dict], *, n_boot: int, seed: int, level: float) -> dict[str, Any]:
    """主分析（华宇缺陷）与分仓库/质量子集的 I_Y、I_b 点估计与分层配对 Bootstrap CI。"""
    def subset(pred) -> list[dict]:
        return [r for r in table if pred(r)]
    groups = {
        "huayu_defective_primary": subset(lambda r: r["warehouse"] == "huayu" and r["quality_status"] == "defective"),
        "chengdong_defective": subset(lambda r: r["warehouse"] == "chengdong" and r["quality_status"] == "defective"),
        "beigang_defective": subset(lambda r: r["warehouse"] == "beigang" and r["quality_status"] == "defective"),
        "normal_all": subset(lambda r: r["quality_status"] == "normal"),
    }
    out = {}
    for name, recs in groups.items():
        branch_rates = {b: wilson(sum(int(r[f"{b}_b_release"]) for r in recs), len(recs)) for b in ("00", "10", "01", "11")}
        out[name] = {
            "n": len(recs),
            "I_Y_release": stratified_paired_bootstrap(recs, "I_Y_release", n_boot=n_boot, seed=seed, level=level),
            "I_Y_target": stratified_paired_bootstrap(recs, "I_Y_target", n_boot=n_boot, seed=seed, level=level),
            "I_binary_release": stratified_paired_bootstrap(recs, "I_b_release", n_boot=n_boot, seed=seed, level=level),
            "branch_release_rates": branch_rates,
            "branch_mean_Y_release": {b: (math.fsum(r[f"{b}_Y_release"] for r in recs) / len(recs)) if recs else None
                                      for b in ("00", "10", "01", "11")},
        }
    return out
