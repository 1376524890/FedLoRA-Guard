"""E10-C/D 构造入口（run_e10.sh construct）：两客户端反向构造 + 四分支 + 实因子代数校验。

P3 试点参数（计划 §7）：dev、s17、Dolly 单源、r=16、ε=0.1、pa=pb=0.5（等权，D6/D7）。
流程：读 P2 截断因子 factor_r{r} → construct_two_clients → build_branches
（00/10/01/11 + a/b_active + update_agg）→ 落盘 → 实因子校验：
  - 联合重构：aggregate_lora_factors([a,b],[pa,pb]) 逐元素等于截断目标；
  - 单客户端有效更新比例 update_ratio == 理论闭式 c_a/c_b（§6.4）；
  - E_FA 闭式 vs 直接计算一致（§15.4 同式）。
产出：artifacts/e10/p3/<series>_r<rank>/{construct_report.json, branches/<name>/...}。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.e10.branches import build_branches, save_branches  # noqa: E402
from src.e10.construct import construct_two_clients, theoretical_ratios  # noqa: E402
from src.e10.lora_algebra import (aggregate_effective_updates,  # noqa: E402
                                  aggregate_lora_factors, e_fa_closed_form_two,
                                  module_pairs, update_ratio)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="dolly_s17")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--pa", type=float, default=0.5)
    args = ap.parse_args()
    pb = 1.0 - args.pa
    s = 2.0
    alpha = 2 * args.rank  # D5：s=α/r=2

    import torch
    from safetensors.torch import load_file

    src_dir = ROOT / f"artifacts/e10/p2/{args.series}/target"
    factor_file = src_dir / f"factor_r{args.rank}.safetensors"
    adapter = dict(load_file(str(factor_file)))
    target_pairs = module_pairs(adapter)
    run_json = json.loads((ROOT / f"artifacts/e10/p2/{args.series}/run.json").read_text(encoding="utf-8"))

    constructed = construct_two_clients(target_pairs, args.pa, pb, args.epsilon)
    branches = build_branches(constructed, args.pa, pb, s, args.rank, alpha)

    out = ROOT / f"artifacts/e10/p3/{args.series}_r{args.rank}"
    meta = {"series": args.series, "source": run_json["source"], "seed": run_json["seed"],
            "base": run_json["base"], "scene": run_json["scene"],
            "target_factor": f"p2/{args.series}/target/factor_r{args.rank}.safetensors",
            "target_factor_sha256": sha256_of(factor_file),
            "epsilon": args.epsilon, "pa": args.pa, "pb": pb, "s": s}
    manifest = save_branches(branches, out / "branches", meta)

    # ---- 实因子代数校验（P3 检查项，§15.4/§6.5 口径）----
    a, b = constructed["client_a"], constructed["client_b"]
    joint = aggregate_lora_factors([a, b], [args.pa, pb])
    recon_ok = all(torch.allclose(joint[m][0], target_pairs[m][0], atol=1e-5)
                   and torch.allclose(joint[m][1], target_pairs[m][1], atol=1e-5)
                   for m in target_pairs)
    th = theoretical_ratios(args.pa, pb, args.epsilon)
    c_a, c_b = update_ratio(a, target_pairs, s), update_ratio(b, target_pairs, s)
    # E_FA 两算法一致：E1 = s·(因子聚合 − 有效更新聚合)（物化直算）vs E2 = 闭式；
    # §6.5 恒等式：E_FA 矩阵 = (1−ε)²·ΔW_goal
    update_pairs = aggregate_effective_updates([a, b], [args.pa, pb])
    e2 = e_fa_closed_form_two(a, b, s, args.pa, pb)
    num2 = den2 = 0.0
    fa_ident_dev = 0.0
    for m in target_pairs:
        e1 = s * (joint[m][1] @ joint[m][0] - update_pairs[m][1] @ update_pairs[m][0])
        e2m = e2[m][1] @ e2[m][0]
        goal = s * (target_pairs[m][1] @ target_pairs[m][0])
        num2 += float(torch.linalg.norm(e1 - e2m) ** 2)
        den2 += float(torch.linalg.norm(e1) ** 2)
        fa_ident_dev = max(fa_ident_dev, float(torch.linalg.norm(e1 - th["e_fa"] * goal)
                                               / (torch.linalg.norm(goal) + 1e-30)))
    e_fa_residual = (num2 / (den2 + 1e-30)) ** 0.5
    checks = {
        "joint_reconstruction_exact": recon_ok,
        "c_a_measured": c_a, "c_a_theory": th["c_a"],
        "c_b_measured": c_b, "c_b_theory": th["c_b"],
        "e_fa_two_algorithms_residual": e_fa_residual,
        "e_fa_identity_rel_dev": fa_ident_dev,
        "update_aggregate_ratio": th["update_aggregate"],
    }
    checks["pass"] = bool(recon_ok
                          and abs(c_a - th["c_a"]) < 1e-6 and abs(c_b - th["c_b"]) < 1e-6
                          and e_fa_residual <= 1e-5 and fa_ident_dev <= 1e-4)
    report = {"meta": meta, "theoretical": th, "checks": checks, "branches": manifest}
    (out / "construct_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    print(f"[construct] {args.series} r{args.rank} ε={args.epsilon}: "
          f"联合重构 {'PASS' if recon_ok else 'FAIL'} | "
          f"c_a {c_a:.6f}≈{th['c_a']:.6f} c_b {c_b:.6f}≈{th['c_b']:.6f} | "
          f"E_FA 两算法残差 {e_fa_residual:.2e} 恒等式偏差 {fa_ident_dev:.2e} | "
          f"{'PASS' if checks['pass'] else 'FAIL'} → {out}")
    sys.exit(0 if checks["pass"] else 1)


if __name__ == "__main__":
    main()
