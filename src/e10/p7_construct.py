"""E10 P7 十客户端混合轮构造（方案 §11.2 主定义 + §11.4，D7，2026-11-20）。

配对矩阵（D3 冻结分配的最终测试轨迹，无开发/校准泄漏）：
- pku 构造系列（s17/s42/s2026，P5 锁定 r=8、ε=0.1）× 同种子 pku 背景（4 条 × 6 轮）= 72 混合轮；
- dolly 构造系列（s17/s42/s2026）× clean_s2026 背景（1 条 × 6 轮）= 18 混合轮。
共 90 混合轮。

槽位映射（确定性决策，2026-11-20）：10 槽等权 0.1；槽 0/1 = 构造客户端 a/b；
归档轮 client_02..client_09 恒等映射到槽 2..9；归档 client_00/01 上传弃用（不重排）。

构造（§11.2 主定义，禁用增量定义）：
  M_target = (r/α)·ΔW_goal，ΔW_goal = 目标模型相对原始基座的有效更新 = s·B_goalA_goal
  → M_target = B_goalA_goal → 秩 8 分解 M_target ≈ B_target·A_target（QR+小 SVD，精确）
  ΔA = A_target − A_bg，ΔB = B_target − B_bg，A_bg/B_bg = 8 背景因子 0.1 加权和
  A_a=εΔA；A_b=(1−p_aε)/p_b·ΔA；B_a=(1−p_bε)/p_a·ΔB；B_b=εΔB（p_a=p_b=0.1）
验证：联合槽 0.1 加权聚合 == (A_target,B_target)；有效更新 == B_goalA_goal（构造误差上报）。

每轮产出 artifacts/e10/p7/<series>/<bg>_r<round>/：
  branches/{00,10,01,11}/（fixed_slot_branch，r=8 α=16）+ manifest.json
  constructs/{a,b}/adapter.safetensors（检测回放与审计用）
  construct_report.json（误差、范数、身份、槽位映射）
另存全良性原始聚合对照（§11.4）：_benign/<traj>_r6/branches/benign_global/（仅 round 6，闭集评估用）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(4)  # 全程小矩阵闭式运算：防多线程过度订阅（47 线程时 sync 开销 >> 算力）

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.e10.branches import build_ten_client_branches, save_branches  # noqa: E402
from src.e10.lora_algebra import (fro_norm_lowrank, module_pairs,  # noqa: E402
                                  from_pairs)

EPSILON = 0.1
PA = PB = 0.1  # 10 客户端等权
S = 2.0        # α/r = 16/8
RANK, ALPHA = 8, 16
BG_SLOTS = [f"client_{i:02d}" for i in range(2, 10)]  # 槽 2..9

PAIRING = {  # 构造系列 → 背景轨迹（D3 最终测试保留清单）
    "pku_s17": [f"e10bg_pku_s17_{i}" for i in ("00", "01", "10", "11")],
    "pku_s42": [f"e10bg_pku_s42_{i}" for i in ("00", "01", "10", "11")],
    "pku_s2026": [f"e10bg_pku_s2026_{i}" for i in ("00", "01", "10", "11")],
    "dolly_s17": ["e10bg_clean_s2026"],
    "dolly_s42": ["e10bg_clean_s2026"],
    "dolly_s2026": ["e10bg_clean_s2026"],
}
BENIGN_TEST_TRAJS = (["e10bg_clean_s2026"]
                     + [f"e10bg_pku_{s}_{i}" for s in ("s17", "s42", "s2026")
                        for i in ("00", "01", "10", "11")])
ROUNDS = [1, 2, 3, 4, 5, 6]


def load_pairs(path: Path) -> dict:
    from safetensors.torch import load_file
    return module_pairs(dict(load_file(str(path))))


def rank_r_factorize(b_goal: torch.Tensor, a_goal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """M = B·A（秩≤r）→ (B_t, A_t)，B_tA_t == M（QR + 小 SVD，数值精确）。
    B: d×r, A: r×k → B_t: d×r, A_t: r×k，B_tA_t = BA。"""
    q, rr = torch.linalg.qr(b_goal.float())            # q: d×r, rr: r×r
    small = rr @ a_goal.float()                         # r×k
    u, sig, vh = torch.linalg.svd(small, full_matrices=False)  # u: r×r, vh: r×k
    sqrt = sig.clamp_min(0).sqrt()
    b_t = q @ (u * sqrt)                                # d×r
    a_t = sqrt.unsqueeze(1) * vh                        # r×k
    return b_t, a_t


def construct_ten(delta_pairs: dict, pa: float, pb: float, epsilon: float) -> dict:
    """§11.2 差值分配（任意 pa,pb；无 pa+pb=1 约束——十客户端权重和为 1 来自全槽）。"""
    if not (0 < epsilon < min(1 / pa, 1 / pb)):
        raise ValueError(f"ε={epsilon} 越界")
    a_pairs, b_pairs = {}, {}
    for mod, (da, db) in delta_pairs.items():
        a_pairs[mod] = (epsilon * da, (1 - pb * epsilon) / pa * db)
        b_pairs[mod] = ((1 - pa * epsilon) / pb * da, epsilon * db)
    return {"client_a": a_pairs, "client_b": b_pairs}


def fro_dist(pairs_a: dict, pairs_b: dict) -> tuple[float, float]:
    """逐模块 (A,B) 对差的相对 Frobenius 误差（分母 = 目标范数）。返回 (max_rel, 模块)。"""
    worst, worst_mod = 0.0, ""
    for mod in pairs_a:
        (a1, b1), (a2, b2) = pairs_a[mod], pairs_b[mod]
        num = (a1 - a2).norm().item() ** 2 + (b1 - b2).norm().item() ** 2
        den = a2.norm().item() ** 2 + b2.norm().item() ** 2
        rel = (num / max(den, 1e-30)) ** 0.5
        if rel > worst:
            worst, worst_mod = rel, mod
    return worst, worst_mod


def build_one(series: str, bg_run: str, rd: int, target_pairs: dict,
              target_sha: str, out_root: Path) -> dict:
    bg_dir = ROOT / "artifacts/runs" / bg_run / "checkpoints" / f"round_{rd:03d}"
    bg_list = [load_pairs(bg_dir / "clients" / c / "final_adapter.safetensors")
               for c in BG_SLOTS]
    a_bg = {m: sum(0.1 * p[m][0] for p in bg_list) for m in bg_list[0]}
    b_bg = {m: sum(0.1 * p[m][1] for p in bg_list) for m in bg_list[0]}

    target_f = {}
    fact_err = 0.0
    for mod, (a_goal, b_goal) in target_pairs.items():
        b_t, a_t = rank_r_factorize(b_goal, a_goal)
        target_f[mod] = (a_t, b_t)
        m_ref = (b_goal.float() @ a_goal.float()).norm().item()
        m_new = (b_t @ a_t).norm().item()
        if m_ref > 0:
            fact_err = max(fact_err, abs(m_new - m_ref) / m_ref)

    delta = {mod: (target_f[mod][0] - a_bg[mod], target_f[mod][1] - b_bg[mod])
             for mod in target_f}
    constructed = construct_ten(delta, PA, PB, EPSILON)

    # 验证：联合槽加权聚合 == target_f；有效更新闭合
    from src.e10.lora_algebra import aggregate_lora_factors
    joint = aggregate_lora_factors(
        [constructed["client_a"], constructed["client_b"]] + bg_list, [0.1] * 10)
    agg_rel, _ = fro_dist(joint, target_f)

    weights = [0.1] * 10
    branches = build_ten_client_branches(constructed, bg_list, weights, S, RANK, ALPHA)

    out = out_root / series / f"{bg_run}_r{rd}"
    meta = {"series": series, "background_run": bg_run, "round": rd,
            "epsilon": EPSILON, "pa": PA, "pb": PB, "weights": weights,
            "slot_map": {"0": "construct_a", "1": "construct_b",
                         **{str(i + 2): f"{bg_run}:client_{i + 2:02d}" for i in range(8)},
                         "dropped_uploads": [f"{bg_run}:client_00", f"{bg_run}:client_01"]},
            "target_factor_sha256": target_sha,
            "protocol": "M_target=(r/alpha)dW_goal 主定义（禁用增量定义）",
            "construction": {
                "factorization_max_rel_err": fact_err,
                "joint_aggregate_max_rel_err": agg_rel,
            }}
    save_branches(branches, out / "branches",
                  {"series": series, "background_run": bg_run, "round": rd,
                   "ten_client": True, **{k: meta[k] for k in
                                          ("epsilon", "pa", "pb", "slot_map")}})
    from safetensors.torch import save_file
    for name, pairs in (("a", constructed["client_a"]), ("b", constructed["client_b"])):
        d = out / "constructs" / name
        d.mkdir(parents=True, exist_ok=True)
        save_file({k: v.contiguous() for k, v in from_pairs(pairs).items()},
                  str(d / "adapter.safetensors"))
    norms = {"construct_a": (sum(fro_norm_lowrank(*constructed["client_a"][m]) ** 2
                                 for m in constructed["client_a"])) ** 0.5,
             "construct_b": (sum(fro_norm_lowrank(*constructed["client_b"][m]) ** 2
                                 for m in constructed["client_b"])) ** 0.5,
             "a_bg": (sum(fro_norm_lowrank(a_bg[m], b_bg[m]) ** 2 for m in a_bg)) ** 0.5}
    report = {**meta, "upload_factor_norms": norms}
    (out / "construct_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"series": series, "bg": bg_run, "round": rd,
            "factorization_err": fact_err, "joint_agg_err": agg_rel,
            "norm_a": norms["construct_a"], "norm_b": norms["construct_b"]}


def save_benign_globals(out_root: Path) -> list[str]:
    """全良性原始聚合对照（§11.4）：各测试轨迹 round 6 的 global_adapter。"""
    saved = []
    for traj in BENIGN_TEST_TRAJS:
        src = ROOT / "artifacts/runs" / traj / "checkpoints/round_006/global_adapter.safetensors"
        if not src.exists():
            continue
        d = out_root / "_benign" / f"{traj}_r6/branches/benign_global"
        d.mkdir(parents=True, exist_ok=True)
        (d / "adapter.safetensors").write_bytes(src.read_bytes())
        (d / "branch.json").write_text(json.dumps(
            {"protocol": "all_benign_original_global", "rank": RANK, "alpha": ALPHA,
             "scaling": S, "trajectory": traj, "round": 6}, ensure_ascii=False, indent=1))
        saved.append(traj)
    return saved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/e10/p7")
    ap.add_argument("--series", default=None, help="只跑指定系列（冒烟测试用）")
    args = ap.parse_args()
    out_root = ROOT / args.out
    t0 = time.time()
    results = []
    for series, bgs in PAIRING.items():
        if args.series and series != args.series:
            continue
        factor_path = ROOT / "artifacts/e10/p2" / series / "target/factor_r8.safetensors"
        target_pairs = load_pairs(factor_path)
        import hashlib
        target_sha = hashlib.sha256(factor_path.read_bytes()).hexdigest()
        for bg in bgs:
            for rd in ROUNDS:
                results.append(build_one(series, bg, rd, target_pairs, target_sha, out_root))
    benign = save_benign_globals(out_root)
    summary = {
        "mixed_rounds": len(results),
        "factorization_max_rel_err": max(r["factorization_err"] for r in results),
        "joint_aggregate_max_rel_err": max(r["joint_agg_err"] for r in results),
        "benign_global_saved": benign,
        "pairing": PAIRING, "epsilon": EPSILON, "weights": [0.1] * 10,
        "elapsed_s": round(time.time() - t0, 1), "per_round": results,
    }
    (out_root / "construct_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[p7-construct] {len(results)} 混合轮 | 分解误差 max "
          f"{summary['factorization_max_rel_err']:.2e} | 聚合闭合误差 max "
          f"{summary['joint_aggregate_max_rel_err']:.2e} | 良性对照 {len(benign)} | "
          f"{summary['elapsed_s']}s")


if __name__ == "__main__":
    main()
