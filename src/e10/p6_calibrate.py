"""E10 P6 良性校准（计划 §10.2/D3，2026-11-19）。

校准轨迹：clean s17 + s42（各 6 轮 × 10 客户端，1.7B r=8 良性上传）。
开发轨迹（Dolly s17 4 条）只做方法 sanity，不参与定阈值。
clean s2026 + PKU 系严格保留给最终测试（不在此触碰）。

校准内容（全部闭式低秩计算，不物化 ΔW）：
- Norm：每客户端每轮有效更新 Frobenius 范数 → 良性分布与上尾阈值（FPR 1%/5%）；
- Cosine：逐对有效更新余弦（fro_inner_lowrank 闭式）→ 良性分布与下尾阈值；
- E_ij / C_ij（§13.1 组合级特征）：良性分布基线（供 P9 漏检归因参考，不定阈值）；
- FoolsGold / Multi-Krum / Trimmed-Mean：参数固定（实现即配置），无阈值可校，记录版本。
产出：reports/e10_p6_calibration_20261119/{calibration.json, dev_sanity.json,
detector_config.json}——阈值冻结后供 P7 同轮检测与最终测试使用。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.e10.lora_algebra import (fro_inner_lowrank, fro_norm_lowrank,  # noqa: E402
                                  module_pairs, pair_features)

S = 2.0  # r=8, α=16
CAL_RUNS = ["e10bg_clean_s17", "e10bg_clean_s42"]
DEV_RUNS = ["e10bg_dolly_s17_00", "e10bg_dolly_s17_10", "e10bg_dolly_s17_01", "e10bg_dolly_s17_11"]
CLIENTS = [f"client_{i:02d}" for i in range(10)]


def load_round(run_dir: str, r: int) -> dict:
    from safetensors.torch import load_file
    base = ROOT / "artifacts/runs" / run_dir / "checkpoints" / f"round_{r:03d}" / "clients"
    return {c: module_pairs(dict(load_file(str(base / c / "final_adapter.safetensors"))))
            for c in CLIENTS}


def round_features(pairs: dict) -> dict:
    norms = {c: fro_norm_lowrank(*pairs[c][list(pairs[c])[0]][::-1]) if False else
             (sum(fro_norm_lowrank(pairs[c][m][0], pairs[c][m][1]) ** 2 for m in pairs[c])) ** 0.5
             for c in pairs}
    cos = {}
    ids = sorted(pairs)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            num = sum(fro_inner_lowrank(pairs[a][m][1], pairs[a][m][0],
                                        pairs[b][m][1], pairs[b][m][0]) for m in pairs[a])
            den = norms[a] * norms[b]
            cos[f"{a}|{b}"] = num / den if den > 0 else 0.0
    # E_ij/C_ij：以 client_00 为参照的抽样（全对 45 个在 JSON 里可全算，此处全算）
    eij = {}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            f = pair_features(pairs[a], pairs[b], S, 0.5, 0.5)
            eij[f"{a}|{b}"] = {k: float(v) for k, v in f.items()
                               if isinstance(v, (int, float))}
    return {"norms": norms, "cosine": cos, "pair_features": eij}


def collect(runs: list[str]) -> list[dict]:
    out = []
    for run in runs:
        for r in range(1, 7):
            p = ROOT / "artifacts/runs" / run / "checkpoints" / f"round_{r:03d}"
            if not p.exists():
                continue
            out.append({"run": run, "round": r, **round_features(load_round(run, r))})
    return out


def pct(values: list[float], q: float) -> float:
    vs = sorted(values)
    idx = min(int(q * (len(vs) - 1)), len(vs) - 1)
    return vs[idx]


def summarize(rounds: list[dict]) -> dict:
    norms = [v for rd in rounds for v in rd["norms"].values()]
    cosines = [v for rd in rounds for v in rd["cosine"].values()]
    e_fa = [v for rd in rounds for f in rd["pair_features"].values() for v in [f.get("e_fa")] if v is not None]
    c_ab = [v for rd in rounds for f in rd["pair_features"].values() for v in [f.get("c_ab")] if v is not None]
    return {
        "n_rounds": len(rounds), "n_client_norms": len(norms), "n_pairs": len(cosines),
        "norm": {"min": min(norms), "p50": pct(norms, .5), "p99": pct(norms, .99),
                 "max": max(norms),
                 "thr_fpr1": pct(norms, .99), "thr_fpr5": pct(norms, .95)},
        "cosine": {"min": min(cosines), "p01": pct(cosines, .01), "p05": pct(cosines, .05),
                   "p50": pct(cosines, .5), "max": max(cosines),
                   "thr_fpr1": pct(cosines, .01), "thr_fpr5": pct(cosines, .05)},
        "benign_e_fa": {"p50": pct(e_fa, .5), "p95": pct(e_fa, .95), "max": max(e_fa)} if e_fa else None,
        "benign_c_ab": {"p50": pct(c_ab, .5), "p95": pct(c_ab, .95), "max": max(c_ab)} if c_ab else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/e10_p6_calibration_20261119")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    cal = collect(CAL_RUNS)
    dev = collect(DEV_RUNS)
    cal_sum, dev_sum = summarize(cal), summarize(dev)
    calibration = {"trajectories": CAL_RUNS, "summary": cal_sum,
                   "per_round": [{"run": r["run"], "round": r["round"],
                                  "norms": r["norms"], "cosine_min": min(r["cosine"].values()),
                                  "cosine_mean": sum(r["cosine"].values()) / len(r["cosine"])}
                                 for r in cal]}
    (out / "calibration.json").write_text(json.dumps(calibration, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    (out / "dev_sanity.json").write_text(
        json.dumps({"trajectories": DEV_RUNS, "summary": dev_sum}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    detector_config = {
        "s": S, "calibration_trajectories": CAL_RUNS,
        "dev_trajectories": DEV_RUNS,
        "reserved_for_final_test": ["e10bg_clean_s2026", "e10bg_pku_s17_*", "e10bg_pku_s42_*", "e10bg_pku_s2026_*"],
        "norm_upper": {"fpr1": cal_sum["norm"]["thr_fpr1"], "fpr5": cal_sum["norm"]["thr_fpr5"]},
        "cosine_lower": {"fpr1": cal_sum["cosine"]["thr_fpr1"], "fpr5": cal_sum["cosine"]["thr_fpr5"]},
        "parameter_free": {"foolsgold": "src/fed/aggregate.foolsgold",
                           "multi_krum": "f=2", "trimmed_mean": "两端各裁 2",
                           "cbdf": "固定实现版本照报（预期高误报）"},
        "frozen": "2026-11-19，P7 前不得改动（检测分数、阈值和报警在读取测试轮行为结果前固定）",
    }
    (out / "detector_config.json").write_text(json.dumps(detector_config, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    print(f"[p6] 校准 {len(cal)} 轮（clean s17+s42）| "
          f"Norm 阈值 fpr5={cal_sum['norm']['thr_fpr5']:.4f} fpr1={cal_sum['norm']['thr_fpr1']:.4f} "
          f"(范围 {cal_sum['norm']['min']:.4f}~{cal_sum['norm']['max']:.4f}) | "
          f"Cosine 阈值 fpr5={cal_sum['cosine']['thr_fpr5']:.4f} fpr1={cal_sum['cosine']['thr_fpr1']:.4f} | "
          f"dev sanity {len(dev)} 轮 → {out}")


if __name__ == "__main__":
    main()
