"""E11 P1 数学测试：真实 LoRA 张量上的构造/聚合/范数验证（执行计划 §5.8）。

输入：E10 背景轨迹 e10bg_pku_s17_00 第 1 轮的真实客户端上传（8 个 r=8 因子）。
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.e11 import algebra as alg  # noqa: E402


def _load_bg(bg_dir: Path) -> list[dict]:
    out = []
    for c in [f"client_{i:02d}" for i in range(2, 10)]:
        out.append(alg.load_pairs_file(bg_dir / c / "final_adapter.safetensors"))
    return out


def run_all(bg_dir: Path, rep_dir: Path) -> dict:
    import torch
    torch.set_num_threads(4)
    bg = _load_bg(bg_dir)
    results = {}

    # 1. Procrustes：乘积不变 ≤1e-6，R 正交
    target = {m: (bg[0][m][0] + 0.01, bg[1][m][1] * 0.9) for m in bg[0]}
    a_bg = {m: sum(0.1 * p[m][0] for p in bg) for m in bg[0]}
    b_bg = {m: sum(0.1 * p[m][1] for p in bg) for m in bg[0]}
    rotated, diag = alg.procrustes_rotate(target, a_bg)
    results["procrustes_product_inv"] = diag["max_product_rel_err"] <= 1e-6
    results["procrustes_R_orth"] = diag["max_R_orth_err"] <= 1e-5
    results["procrustes_dist_reduced"] = diag["mean_dist_after"] <= diag["mean_dist_before"]

    # 2. 构造 + 联合恢复 ≤1e-5；有效更新闭合
    constructs = alg.construct_pair(rotated, a_bg, b_bg, 0.1, 0.1, 0.1)
    joint = alg.joint_recover(constructs, bg)
    recon = alg.rel_err_pairs(joint, rotated)
    results["joint_factor_recovery"] = recon <= 1e-5
    # 有效更新：Σ 0.1·ΔW_i == ΔW_rot（低秩可加）
    eff_sum = {m: (sum(0.1 * p[m][0] for p in [constructs["a"], constructs["b"]] + bg),
                   sum(0.1 * p[m][1] for p in [constructs["a"], constructs["b"]] + bg))
               for m in bg[0]}
    results["joint_effective_recovery"] = alg.rel_err_pairs(eff_sum, rotated) <= 1e-5

    # 3. scaling=2：eff_norm == 2×‖BA‖ 合并范数
    from src.e10.lora_algebra import fro_norm_lowrank
    manual = 2 * sum(fro_norm_lowrank(bg[0][m][0], bg[0][m][1]) ** 2 for m in bg[0]) ** 0.5
    results["scaling_in_norm"] = abs(alg.eff_norm(bg[0]) - manual) / manual <= 1e-9

    # 4. FedAvg 与手工平均一致；00n 情形
    g = alg.fedavg(bg + [bg[0], bg[0]])
    manual_g = {m: (sum(p[m][0] for p in bg + [bg[0], bg[0]]) / 10,
                    sum(p[m][1] for p in bg + [bg[0], bg[0]]) / 10) for m in bg[0]}
    results["fedavg_matches_manual"] = alg.rel_err_pairs(g, manual_g) <= 1e-6

    # 5. 低秩 Gram 与显式小矩阵一致
    from src.e10.lora_algebra import fro_inner_lowrank
    m0 = sorted(bg[0])[0]
    a1, b1 = bg[0][m0]
    a2, b2 = bg[1][m0]
    explicit = float((b1 @ a1 * (b2 @ a2)).sum())
    results["gram_vs_explicit"] = abs(fro_inner_lowrank(b1, a1, b2, a2) - explicit) / abs(explicit) <= 1e-4

    # 6. 题对相邻性（调度置换）
    from src.e11.train import order_records_for_pair, round_permutation
    recs = [{"sample_id": f"n{i:02d}"} for i in range(18)] + [
        {"sample_id": "pair_h"}, {"sample_id": "pair_n"}]
    ordered = order_records_for_pair(recs, ("pair_h", "pair_n"), 17, 0, 3)
    perm = round_permutation(17, 0, 3, 20)
    results["pair_adjacent"] = [r["sample_id"] for r in ordered[:2]] and \
        ordered[perm[0]]["sample_id"] in ("pair_h", "pair_n") and \
        ordered[perm[1]]["sample_id"] in ("pair_h", "pair_n")

    # 7. 曝光保证：sample_schedule 30 轮 × 20 恰为 120×5
    from src.train.local import sample_schedule
    cnt = {}
    for r in range(1, 31):
        for i in sample_schedule(17, 3, r, 120, 20):
            cnt[i] = cnt.get(i, 0) + 1
    results["exposure_5x"] = len(cnt) == 120 and set(cnt.values()) == {5}

    # 8. NaN/Inf 检查
    results["finite"] = all(bool(torch.isfinite(v).all()) for p in (rotated, constructs["a"], constructs["b"])
                            for m in p for v in p[m])

    summary = {k: bool(v) for k, v in results.items()}
    summary["all_pass"] = all(summary.values())
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "algebra_tests.json").write_text(
        json.dumps({"results": summary, "procrustes_diag": diag, "recon_err": recon},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    return {"all_pass": summary["all_pass"], "summary": summary}
