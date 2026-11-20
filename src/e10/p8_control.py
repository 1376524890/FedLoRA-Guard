"""E10-G（P8）表示变换对照（方案 §4.3/§12，2026-11-20）。

LoRA 因子存在规范自由度（gauge）：A→RA、B→BR⁻¹（R 可逆）不改变有效更新 ΔW=s·BA。
本对照在**真实因子**上（六系列 r=8 截断目标 + ε=0.1 两客户端构造）验证：
  1. ΔW 逐元素不变（相对残差）；
  2. 构造恒等式不变：联合重构、c_a/c_b（update_ratio）、E_FA（两算法）；
  3. 四分支有效更新不变（00/10/01/11）。
门：全部相对残差 ≤1e-6（计划 §阶段表）。随机正交 R（数值最稳的规范变换），
种子固定可复现。纯 CPU 闭式计算。
产出：reports/e10_p8_20261120/{control_report.json, RESULTS.md}。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.e10.construct import construct_two_clients, theoretical_ratios  # noqa: E402
from src.e10.lora_algebra import (aggregate_effective_updates,  # noqa: E402
                                  aggregate_lora_factors, e_fa_closed_form_two,
                                  module_pairs, update_ratio)
from src.e10.branches import four_branch_updates  # noqa: E402

S = 2.0
RANK = 8
EPS = 0.1
SERIES = ["dolly_s17", "dolly_s42", "dolly_s2026", "pku_s17", "pku_s42", "pku_s2026"]


def max_rel(a, b) -> float:
    import torch
    num = float((a.double() - b.double()).abs().max())
    den = float(b.double().abs().max()) + 1e-30
    return num / den


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/e10_p8_20261120")
    ap.add_argument("--seed", type=int, default=20261120)
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from safetensors.torch import load_file
    torch.manual_seed(args.seed)

    report = {"seed": args.seed, "transform": "正交 R（8×8，QR 生成）：A→RA，B→BRᵀ", "series": {}}
    worst = 0.0
    for series in SERIES:
        adapter = dict(load_file(str(ROOT / f"artifacts/e10/p2/{series}/target/factor_r8.safetensors")))
        target = module_pairs(adapter)
        con = construct_two_clients(target, 0.5, 0.5, EPS)
        a, b = con["client_a"], con["client_b"]

        # 随机正交 R：每模块一个，**全客户端共享**（同规范才保持聚合恒等式）
        Rs = {m: torch.linalg.qr(torch.randn(RANK, RANK, dtype=torch.float64))[0].to(torch.float32)
              for m in a}

        def transform(pairs):
            return {m: (Rs[m] @ pairs[m][0], pairs[m][1] @ Rs[m].T) for m in pairs}

        a2, b2 = transform(a), transform(b)
        target2 = aggregate_lora_factors([a2, b2], [0.5, 0.5])  # 联合 = 变换后目标

        res = {}
        # 1) ΔW 不变（低秩闭式：BA 逐模块物化取最大相对残差，仅 r=8 乘积小矩阵）
        res["delta_w"] = max(max_rel(S * (a2[m][1] @ a2[m][0]), S * (a[m][1] @ a[m][0]))
                             for m in a)
        # 2) 恒等式不变（联合重构按规范不变量 ΔW=B̄Ā 比较；因子矩阵跨规范无逐元素可比性）
        joint = aggregate_lora_factors([a, b], [0.5, 0.5])
        res["joint_recon"] = max(max_rel(target2[m][1] @ target2[m][0], joint[m][1] @ joint[m][0])
                                 for m in joint)
        th = theoretical_ratios(0.5, 0.5, EPS)
        c_a1, c_b1 = update_ratio(a, target, S), update_ratio(b, target, S)
        c_a2, c_b2 = update_ratio(a2, target2, S), update_ratio(b2, target2, S)
        res["c_ab"] = max(abs(c_a1 - c_a2), abs(c_b1 - c_b2))
        e1 = e_fa_closed_form_two(a, b, S, 0.5, 0.5)
        e2t = e_fa_closed_form_two(a2, b2, S, 0.5, 0.5)
        res["e_fa_closed"] = max(max_rel(e2t[m][1] @ e2t[m][0], e1[m][1] @ e1[m][0]) for m in e1)
        # 3) 四分支有效更新不变
        br1 = four_branch_updates(a, b, S, 0.5, 0.5)
        br2 = four_branch_updates(a2, b2, S, 0.5, 0.5)
        res["four_branch"] = max(
            max_rel(br2[n][m][1] @ br2[n][m][0], br1[n][m][1] @ br1[n][m][0])
            for n in br1 for m in br1[n])
        worst = max(worst, *res.values())
        report["series"][series] = res

    report["gate_max_rel_1e-6"] = worst <= 1e-6
    report["worst_rel_residual"] = worst
    (out / "control_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    print(f"[p8] 表示变换对照（六系列，正交 R）: 最大相对残差 {worst:.3e} | "
          f"门 ≤1e-6 {'PASS' if report['gate_max_rel_1e-6'] else 'FAIL'} → {out}")


if __name__ == "__main__":
    main()
