"""E10 §15 数学与程序测试（P1 验收门）。全部使用真实矩阵运算。

15.1 形状；15.2 分解重构误差=报告值；15.3 构造聚合恒等；15.4 单客户端比例闭式；
15.5 E_FA 两种算法一致（≤1e-5 相对）；15.6 四分支闭式；15.7 可逆表示变换。
运行：python -m src.e10.algebra_tests  （或 pytest src/e10/algebra_tests.py）
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.e10.branches import build_branches
from src.e10.construct import construct_two_clients, theoretical_ratios
from src.e10.factorization import (factorize_all, module_relative_error,
                                   svd_factorize)
from src.e10.lora_algebra import (LORA_A, LORA_B, aggregate_effective_updates,
                                  aggregate_lora_factors, check_shapes,
                                  cross_term_cab, e_fa_closed_form_two,
                                  fro_inner_lowrank, fro_norm_lowrank,
                                  from_pairs, module_pairs,
                                  relative_residual, update_ratio)

S = 2.0            # D5: α/r = 2 全程一致
PA = PB = 0.5
EPS = 0.1

# 固定测试维度（历史沿用 0.6B 投影形状，仅为固定随机矩阵尺寸，与当前基座无关）
DIMS = {"q_proj": (1024, 1024), "k_proj": (512, 1024), "v_proj": (512, 1024),
        "o_proj": (1024, 1024), "gate_proj": (3072, 1024), "up_proj": (3072, 1024),
        "down_proj": (1024, 3072)}


def _make_target(rank: int = 32, seed: int = 17) -> tuple[dict, dict]:
    """生成真实形状的「目标更新」模块与对应 ΔW（秩 ≤ rank）。"""
    g = torch.Generator().manual_seed(seed)
    deltas, pairs = {}, {}
    for mod, (d_out, d_in) in DIMS.items():
        b = torch.randn(d_out, rank, generator=g) / math.sqrt(rank)
        a = torch.randn(rank, d_in, generator=g) / math.sqrt(d_in)
        deltas[mod] = S * (b @ a)
        pairs[mod] = (a, b)
    return deltas, pairs


def _mat(pairs: dict, mod: str) -> torch.Tensor:
    a, b = pairs[mod]
    return b @ a


# ---------------------------------------------------------------- 15.1 形状

def test_shapes():
    deltas, _ = _make_target()
    for r in (8, 16):
        res = factorize_all(deltas, [r], {r: 2 * r})
        adapter = res[r]["adapter"]
        pairs = module_pairs(adapter)
        assert not check_shapes(pairs, rank=r), f"rank {r} shape check failed"
        for mod, (d_out, d_in) in DIMS.items():
            assert pairs[mod][0].shape == (r, d_in), mod
            assert pairs[mod][1].shape == (d_out, r), mod


# ---------------------------------------------------------------- 15.2 分解误差 = 报告值

def test_factorization_error_matches_report():
    deltas, _ = _make_target()
    for r in (8, 16):
        res = factorize_all(deltas, [r], {r: 2 * r})
        rho_global = res[r]["rho_global"]
        # 手工复核全模型误差
        num2 = 0.0
        den2 = 0.0
        adapter = module_pairs(res[r]["adapter"])
        for mod, delta in deltas.items():
            a, b = adapter[mod]
            num2 += module_relative_error(delta, a, b, S) ** 2 * (torch.linalg.norm(delta) ** 2 + 1e-24)
            den2 += float(torch.linalg.norm(delta) ** 2)
        manual = math.sqrt(num2 / den2)
        assert abs(manual - rho_global) <= 1e-6 * max(rho_global, 1e-9), (r, manual, rho_global)
        assert rho_global < 1.0, f"rank {r} 分解误差异常"
        # 15.2 重构式：(α/r)B*A* 与目标差的范数等于报告误差
        for mod, delta in deltas.items():
            a, b = adapter[mod]
            rho = module_relative_error(delta, a, b, S)
            rep = res[r]["report"][mod]["rho_l"]
            assert abs(rho - rep) <= 1e-9


# ---------------------------------------------------------------- 15.3 构造恒等

def test_construction_identities():
    deltas, _ = _make_target()
    res = factorize_all(deltas, [8], {8: 16})[8]
    target = module_pairs(res["adapter"])
    cl = construct_two_clients(target, PA, PB, EPS)
    a_bar = aggregate_lora_factors([cl["client_a"], cl["client_b"]], [PA, PB])
    for mod in target:
        assert torch.allclose(PA * cl["client_a"][mod][0] + PB * cl["client_b"][mod][0],
                              target[mod][0], atol=1e-5), f"Ā != A* @ {mod}"
        assert torch.allclose(PA * cl["client_a"][mod][1] + PB * cl["client_b"][mod][1],
                              target[mod][1], atol=1e-5), f"B̄ != B* @ {mod}"


# ---------------------------------------------------------------- 15.4 单客户端比例

def test_update_ratios():
    deltas, _ = _make_target()
    res = factorize_all(deltas, [8], {8: 16})[8]
    target = module_pairs(res["adapter"])
    cl = construct_two_clients(target, PA, PB, EPS)
    th = theoretical_ratios(PA, PB, EPS)
    c_a = update_ratio(cl["client_a"], target, S)
    c_b = update_ratio(cl["client_b"], target, S)
    assert abs(c_a - th["c_a"]) <= 1e-4, (c_a, th["c_a"])
    assert abs(c_b - th["c_b"]) <= 1e-4, (c_b, th["c_b"])
    # ΔW_client 确为目标的比例缩放（逐模块余弦=1、范数比=c）
    for mod in target:
        ratio = fro_norm_lowrank(*cl["client_a"][mod]) / max(fro_norm_lowrank(*target[mod]), 1e-12)
        assert abs(ratio - th["c_a"]) <= 1e-4, (mod, ratio)


# ---------------------------------------------------------------- 15.5 E_FA 两种算法

def test_e_fa_two_algorithms():
    deltas, _ = _make_target()
    res = factorize_all(deltas, [8], {8: 16})[8]
    target = module_pairs(res["adapter"])
    cl = construct_two_clients(target, PA, PB, EPS)
    a, b = cl["client_a"], cl["client_b"]
    factor = aggregate_lora_factors([a, b], [PA, PB])
    update = aggregate_effective_updates([a, b], [PA, PB])
    # E1 = ΔW_factor − ΔW_update（物化逐模块直算，避免表示约定影响）
    num2 = den2 = 0.0
    e2 = e_fa_closed_form_two(a, b, S, PA, PB)
    for mod in target:
        e1_mat = S * (_mat(factor, mod) - _mat(update, mod))
        e2_mat = _mat(e2, mod)
        num2 += float(torch.linalg.norm(e1_mat - e2_mat) ** 2)
        den2 += float(torch.linalg.norm(e1_mat) ** 2)
        assert torch.linalg.norm(e1_mat - e2_mat) <= 1e-5 * max(torch.linalg.norm(e1_mat), 1e-12)
    assert math.sqrt(num2 / (den2 + 1e-12)) <= 1e-5
    # §6.5：E_FA = (1−ε)²·ΔW_goal（截断目标）
    th = theoretical_ratios(PA, PB, EPS)
    for mod in target:
        e1_mat = S * (_mat(factor, mod) - _mat(update, mod))
        goal_mat = S * _mat(target, mod)
        assert torch.allclose(e1_mat, th["e_fa"] * goal_mat, rtol=1e-4, atol=1e-6), mod
    # §4.2：E_FA = C_ab − s·p_a·p_b·(B_aA_a + B_bA_b)，两者分别保存且不相等
    cab = cross_term_cab(a, b, S, PA, PB)
    for mod in target:
        e1_mat = S * (_mat(factor, mod) - _mat(update, mod))
        self_terms = S * PA * PB * (_mat(a, mod) + _mat(b, mod))
        assert torch.allclose(e1_mat, _mat(cab, mod) - self_terms, rtol=1e-4, atol=1e-6), mod
    assert any(not torch.allclose(_mat(cab, mod), _mat(e2, mod)) for mod in target)


# ---------------------------------------------------------------- 15.6 四分支

def test_four_branch_closed_forms():
    deltas, _ = _make_target()
    res = factorize_all(deltas, [8], {8: 16})[8]
    target = module_pairs(res["adapter"])
    cl = construct_two_clients(target, PA, PB, EPS)
    br = build_branches(cl, PA, PB, S, 8, 16)
    th = theoretical_ratios(PA, PB, EPS)
    p10 = PA * EPS * (1 - PB * EPS)
    p01 = PB * EPS * (1 - PA * EPS)
    for mod in target:
        goal = S * _mat(target, mod)
        w00 = S * _mat(module_pairs(br["00"]["adapter"]), mod)
        w10 = S * _mat(module_pairs(br["10"]["adapter"]), mod)
        w01 = S * _mat(module_pairs(br["01"]["adapter"]), mod)
        w11 = S * _mat(module_pairs(br["11"]["adapter"]), mod)
        assert torch.linalg.norm(w00) <= 1e-9
        assert torch.allclose(w10, p10 * goal, rtol=1e-4, atol=1e-7), mod
        assert torch.allclose(w01, p01 * goal, rtol=1e-4, atol=1e-7), mod
        assert torch.allclose(w11, goal, rtol=1e-4, atol=1e-7), mod
        # I_W = ΔW11 − ΔW10 − ΔW01 + ΔW00
        iw = w11 - w10 - w01 + w00
        assert torch.allclose(iw, th["i_w"] * goal, rtol=1e-4, atol=1e-6), mod
    # 等权 ε=0.1 数值例（§6.6/§7.3）：0.0475 / 0.19 / 0.905
    assert abs(p10 - 0.0475) < 1e-12 and abs(th["c_a"] - 0.19) < 1e-12
    assert abs(th["i_w"] - 0.905) < 1e-12


# ---------------------------------------------------------------- 15.7 可逆表示变换

def test_representation_transform():
    deltas, _ = _make_target()
    res = factorize_all(deltas, [8], {8: 16})[8]
    target = module_pairs(res["adapter"])
    g = torch.Generator().manual_seed(42)
    for lam in (0.5, 1.0, 2.0, 3.0, -1.0):
        r_inv = lam  # R = λI
        a, b = target["q_proj"]
        b_t = b * r_inv
        a_t = a / r_inv
        # 本地有效更新不变
        assert torch.allclose(b @ a, b_t @ a_t, atol=1e-6), lam
        # E_FA 闭式：s·p_a·p_b·(λ+1/λ−2)·BA
        coef = S * PA * PB * (lam + 1 / lam - 2)
        fa = aggregate_lora_factors([{ "q_proj": (a, b)}, {"q_proj": (a_t, b_t)}], [PA, PB])
        upd = S * (b @ a)
        e_numeric = S * _mat(fa, "q_proj") - upd
        assert torch.allclose(e_numeric, coef * (b @ a), rtol=1e-4, atol=1e-7), lam


# ---------------------------------------------------------------- 迹恒等式抽查（§14.3）

def test_trace_identities():
    g = torch.Generator().manual_seed(7)
    b1, a1 = torch.randn(64, 8, generator=g), torch.randn(8, 96, generator=g)
    b2, a2 = torch.randn(64, 8, generator=g), torch.randn(8, 96, generator=g)
    assert abs(fro_norm_lowrank(a1, b1) - float(torch.linalg.norm(b1 @ a1))) <= 1e-4
    assert abs(fro_inner_lowrank(b1, a1, b2, a2)
               - float((b1 @ a1 * (b2 @ a2)).sum())) <= 1e-3


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"all {len(tests)} algebra tests passed")
