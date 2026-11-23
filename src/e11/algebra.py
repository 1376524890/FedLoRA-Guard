"""E11 代数核心：正交坐标匹配（Procrustes）、甲乙因子构造、有效更新范数、FedAvg。

依据：执行计划 §5.6/§5.7 与设计稿 §9。固定 ε=0.1、p_a=p_b=0.1、s=α/r=2。
有效更新一律 ΔW = s·B A（scaling=2 显式计入范数与内积）。
"""
from __future__ import annotations

import math

import torch

from src.e10.lora_algebra import (fro_inner_lowrank, fro_norm_lowrank,  # noqa: E402
                                  module_pairs)

S = 2.0  # α16/r8


def procrustes_rotate(pairs_t: dict, a_bg: dict) -> tuple[dict, dict]:
    """每模块求正交 R 最小化 ‖R A_t − A_bg‖_F（SVD of A_bg A_tᵀ），
    返回 (Ã_t = R A_t, B̃_t = B_t Rᵀ, 诊断)。乘积不变：B̃Ã = B_t RᵀR A_t = B_t A_t。"""
    rotated, diag = {}, {"max_product_rel_err": 0.0, "max_R_orth_err": 0.0, "mean_dist_before": 0.0,
                         "mean_dist_after": 0.0}
    import torch
    for m in pairs_t:
        a_t, b_t = pairs_t[m]
        abg = a_bg[m]
        M = abg @ a_t.T                       # (r,r)
        u, sig, vh = torch.linalg.svd(M)
        R = u @ vh                            # 正交
        a_rot = R @ a_t
        b_rot = b_t @ R.T
        rotated[m] = (a_rot, b_rot)
        # 诊断
        n_ref = fro_norm_lowrank(a_t, b_t)
        err = abs(fro_norm_lowrank(a_rot, b_rot) - n_ref) / max(n_ref, 1e-30)
        diag["max_product_rel_err"] = max(diag["max_product_rel_err"], float(err))
        diag["max_R_orth_err"] = max(diag["max_R_orth_err"],
                                     float((R @ R.T - torch.eye(R.shape[0])).abs().max()))
        d0 = float((a_t - abg).norm())
        d1 = float((a_rot - abg).norm())
        diag["mean_dist_before"] += d0
        diag["mean_dist_after"] += d1
    n = len(pairs_t)
    diag["mean_dist_before"] /= n
    diag["mean_dist_after"] /= n
    return rotated, diag


def construct_pair(rotated_t: dict, a_bg: dict, b_bg: dict, pa: float, pb: float, eps: float) -> dict:
    """E10 §11.2 差值分配（在 Procrustes 旋转后的坐标中）：
    A_a=εΔA, B_a=(1−p_b·ε)/p_a·ΔB, A_b=(1−p_a·ε)/p_b·ΔA, B_b=εΔB。"""
    out = {"a": {}, "b": {}}
    for m in rotated_t:
        da = rotated_t[m][0] - a_bg[m]
        db = rotated_t[m][1] - b_bg[m]
        out["a"][m] = (eps * da, (1 - pb * eps) / pa * db)
        out["b"][m] = ((1 - pa * eps) / pb * da, eps * db)
    return out


def joint_recover(constructs: dict, bg_list: list[dict]) -> dict:
    """联合 10×0.1 因子平均（甲、乙 + 背景客户端）——应恢复旋转后目标因子。"""
    mods = list(constructs["a"])
    weights = [0.1] * (2 + len(bg_list))
    pairs_list = [constructs["a"], constructs["b"]] + bg_list
    return {m: (sum(w * p[m][0] for p, w in zip(pairs_list, weights)),
                sum(w * p[m][1] for p, w in zip(pairs_list, weights))) for m in mods}


def fedavg(pairs_list: list[dict]) -> dict:
    weights = [1.0 / len(pairs_list)] * len(pairs_list)
    mods = list(pairs_list[0])
    return {m: (sum(w * p[m][0] for p, w in zip(pairs_list, weights)),
                sum(w * p[m][1] for p, w in zip(pairs_list, weights))) for m in mods}


def eff_norm(pairs: dict) -> float:
    """有效更新范数 N = s·sqrt(Σ_m ‖BA‖_F²)。"""
    return S * float(sum(fro_norm_lowrank(pairs[m][0], pairs[m][1]) ** 2 for m in pairs) ** 0.5)


def eff_dist(p1: dict, p2: dict) -> float:
    """‖s(B₁A₁ − B₂A₂)‖（全模型）。float64：近同因子对的差值对 fp32 消去极敏感。"""
    tot = 0.0
    for m in p1:
        a1, b1 = p1[m][0].double(), p1[m][1].double()
        a2, b2 = p2[m][0].double(), p2[m][1].double()
        n1 = float(torch.trace((b1.T @ b1) @ (a1 @ a1.T)))
        n2 = float(torch.trace((b2.T @ b2) @ (a2 @ a2.T)))
        ip = float(torch.trace((b1.T @ b2) @ (a2 @ a1.T)))
        tot += max(n1 + n2 - 2 * ip, 0.0)
    return S * math.sqrt(tot)


def rel_err_pairs(p1: dict, p2: dict) -> float:
    return eff_dist(p1, p2) / max(eff_norm64(p2), 1e-30)


def eff_norm64(pairs: dict) -> float:
    """float64 有效更新范数（相对误差分母用，避免 fp32 噪声）。"""
    import torch
    tot = 0.0
    for m in pairs:
        a, b = pairs[m][0].double(), pairs[m][1].double()
        tot += float(torch.trace((b.T @ b) @ (a @ a.T)))
    return S * math.sqrt(max(tot, 0.0))


def load_pairs_file(path) -> dict:
    from safetensors.torch import load_file
    return module_pairs(dict(load_file(str(path))))


def module_pairs_of(state: dict) -> dict:
    """peft 扁平 state → {module: (A,B)}。"""
    return module_pairs(dict(state))


def flatten_of(pairs: dict) -> dict:
    from src.e10.lora_algebra import from_pairs
    return {k: v.contiguous() for k, v in from_pairs(pairs).items()}


def save_pairs_file(pairs: dict, path) -> None:
    from safetensors.torch import save_file
    from src.e10.lora_algebra import from_pairs
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in from_pairs(pairs).items()}, str(path))
