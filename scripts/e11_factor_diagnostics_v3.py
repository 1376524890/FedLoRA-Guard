#!/usr/bin/env python
"""E11 Task2 v3 因子动力学诊断（只读）。

对 v3 离线阶段的构造结果输出 §8 诊断指标：
1. A 因子范数 / B 因子范数 / BA 有效权重范数轨迹；
2. A/B 有效更新范数与正常客户端中位数比值；
3. A/B 因子变化范数与正常中位数比值；
4. 检测输入空间一致性（Norm/Cosine 用完整有效更新；FoolsGold 用因子差值）。

用法：python scripts/e11_factor_diagnostics_v3.py [--rounds 1..30]
输入：U 30 轮真实上传 + v3 构造（diagnostic λ，若已产生）。
不修改任何旧产物；不启动训练。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.e11.task2_core_v3 import (  # noqa: E402
    ART, CORE_ART, ba_norm, construct_pair_v3, eff_update_norm,
    factor_change_norm, factor_fedavg, factor_norm, load_global, load_uploads)

U_TRAJ = "U_dolly_s17_a5"
REP = ROOT / "reports/e11_task2_core_v3/task2_core_v3_dolly_s17_a5"
ROUNDS = 30


def _pairs(flat: dict) -> dict:
    from src.e10.lora_algebra import module_pairs
    return module_pairs(dict(flat))


def main() -> None:
    parser = argparse.ArgumentParser(description="E11 Task2 v3 因子动力学诊断（只读）")
    parser.add_argument("--diagnostic-lambda", type=str, default=None,
                        help="λ_A,λ_B；默认读取 diagnostic_lambda.json")
    args = parser.parse_args()
    if args.diagnostic_lambda:
        la, lb = (float(x) for x in args.diagnostic_lambda.split(","))
    else:
        dpath = ART / "diagnostic_lambda.json"
        if not dpath.exists():
            raise SystemExit("diagnostic_lambda.json 不存在；先用 --diagnostic-lambda 指定")
        la, lb = json.loads(dpath.read_text())["diagnostic_lambda"]
    out = {"diagnostic_lambda": [la, lb],
           "note": "v3 只读诊断：范数均按 §8 定义；有效更新含完整 BA 乘积",
           "rounds": []}
    for r in range(1, ROUNDS + 1):
        g = load_global(U_TRAJ, r - 1)
        blob = load_uploads(U_TRAJ, r)
        t_pairs = _pairs(blob["client_00"])
        ref_pairs = _pairs(blob["client_01"])
        bg_pairs = {c: _pairs(blob[c]) for c in sorted(blob) if int(c.split("_")[1]) >= 2}
        con = construct_pair_v3(g, t_pairs, ref_pairs, la, lb)
        normals = [ref_pairs] + [bg_pairs[c] for c in sorted(bg_pairs)]
        n_norms = [eff_update_norm(p, g) for p in normals]
        f_norms = [factor_change_norm(p, g) for p in normals]
        mn = float(torch.median(torch.tensor(n_norms)))
        mf = float(torch.median(torch.tensor(f_norms)))
        u_agg = factor_fedavg([t_pairs, ref_pairs] + [bg_pairs[c] for c in sorted(bg_pairs)])
        row = {"round": r, "M_N": mn, "M_F": mf,
               "U_global": {"A_norm": factor_norm(u_agg, "A"), "B_norm": factor_norm(u_agg, "B"),
                            "BA_norm": ba_norm(u_agg)}}
        for tag, pairs in (("A", con["a"]), ("B", con["b"])):
            row[tag] = {"A_norm": factor_norm(pairs, "A"), "B_norm": factor_norm(pairs, "B"),
                        "BA_norm": ba_norm(pairs),
                        "eff_update_norm": eff_update_norm(pairs, g),
                        "factor_change_norm": factor_change_norm(pairs, g),
                        "eff_ratio": eff_update_norm(pairs, g) / mn,
                        "factor_ratio": factor_change_norm(pairs, g) / mf}
        out["rounds"].append(row)
        print(f"r{r:02d} A: eff_ratio={row['A']['eff_ratio']:.3f} "
              f"factor_ratio={row['A']['factor_ratio']:.3f} | "
              f"B: eff_ratio={row['B']['eff_ratio']:.3f} "
              f"factor_ratio={row['B']['factor_ratio']:.3f}", flush=True)
        del blob
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "factor_diagnostics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {REP / 'factor_diagnostics.json'}")


if __name__ == "__main__":
    main()
