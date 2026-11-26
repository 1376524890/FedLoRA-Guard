"""E11 Task2 v3 单元测试：λ 构造公式、双路径 D、联合恢复、范数定义。

只测代数（无需 GPU / 模型）。运行：python -m pytest tests/test_e11_task2_core_v3.py -q
真实张量恒等式验证（U 30 轮）由 `python -m src.e11.task2_core_v3 --stage P1` 完成，
不在 pytest 内重复加载大文件。
"""
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.e11.task2_core_v3 import (  # noqa: E402
    P, S, ba_norm, ba_rel_err, compute_D, construct_pair_v3, eff_update_norm,
    factor_change_norm, factor_fedavg, factor_norm, joint_aggregate_v3,
    max_abs_factor_err, verify_baseline_hold, verify_dual_path_D, weighted_bg)

MODS = ["m1", "m2"]


def fake_pairs(seed: int, scale: float = 0.3) -> dict:
    gen = torch.Generator().manual_seed(seed)
    return {m: (torch.randn(8, 16, generator=gen) * scale,
                torch.randn(12, 8, generator=gen) * scale) for m in MODS}


def fake_round(seed_g: int = 1, seed_t: int = 2, seed_1: int = 3, seed_bg: int = 100):
    g = fake_pairs(seed_g)
    t = fake_pairs(seed_t)
    ref = fake_pairs(seed_1)
    bg = {f"client_{i:02d}": fake_pairs(seed_bg + i) for i in range(2, 10)}
    return g, t, ref, bg


def test_D_contains_both_target_and_ref_changes():
    g, t, ref, bg = fake_round()
    d_a, d_b = compute_D(g, t, ref)
    for m in MODS:
        expect_a = P * ((t[m][0].double() - g[m][0].double()) + (ref[m][0].double() - g[m][0].double()))
        expect_b = P * ((t[m][1].double() - g[m][1].double()) + (ref[m][1].double() - g[m][1].double()))
        assert torch.allclose(d_a[m], expect_a, atol=1e-12)
        assert torch.allclose(d_b[m], expect_b, atol=1e-12)


def test_baseline_hold_exact():
    g, t, ref, bg = fake_round()
    for la, lb in [(0.1, 9.9), (5.0, 5.0), (9.75, 0.25)]:
        con = construct_pair_v3(g, t, ref, la, lb)
        v = verify_baseline_hold(con, g, la, lb)
        assert v["ok"], (la, lb, v)
        # 数值上应精确为零级误差（同一线性公式）
        assert max(v["max_abs_err"].values()) < 1e-12


def test_lambda_bounds_guard():
    g, t, ref, _ = fake_round()
    for la, lb in [(0.0, 5.0), (5.0, 10.0), (10.0, 0.1), (-1.0, 5.0)]:
        try:
            construct_pair_v3(g, t, ref, la, lb)
            raised = False
        except AssertionError:
            raised = True
        assert raised, (la, lb)


def test_joint_aggregation_recovers_U_for_all_lambdas():
    """§3.4：A_AB = A_U、B_AB = B_U，与 λ 取值无关（含极端 λ）。"""
    g, t, ref, bg = fake_round()
    u_agg = factor_fedavg([t, ref] + [bg[c] for c in sorted(bg)])
    for la, lb in [(0.1, 0.1), (5.0, 5.0), (9.9, 9.9), (0.25, 9.75), (9.5, 0.5)]:
        con = construct_pair_v3(g, t, ref, la, lb)
        joint = joint_aggregate_v3(con["a"], con["b"], bg)
        for m in MODS:
            assert torch.allclose(joint[m][0], u_agg[m][0], atol=1e-10), (la, lb, "A")
            assert torch.allclose(joint[m][1], u_agg[m][1], atol=1e-10), (la, lb, "B")


def test_dual_path_D_consistency():
    g, t, ref, bg = fake_round()
    u_agg = factor_fedavg([t, ref] + [bg[c] for c in sorted(bg)])
    v = verify_dual_path_D(g, t, ref, u_agg, bg)
    assert v["ok"], v
    # 双路径在浮点上应几乎一致
    assert max(v["max_abs_err_D_A"], v["max_abs_err_D_B"]) < 1e-10


def test_weighted_bg_is_weighted_sum():
    bg = {f"client_{i:02d}": fake_pairs(50 + i) for i in range(2, 10)}
    a_bg, b_bg = weighted_bg(bg)
    mean = {m: sum(bg[c][m][0].double() for c in sorted(bg)) / 8 for m in MODS}
    assert not torch.allclose(a_bg["m1"], mean["m1"])
    manual = {m: P * sum(bg[c][m][0].double() for c in sorted(bg)) for m in MODS}
    assert torch.allclose(a_bg["m1"], manual["m1"])


def test_effective_update_uses_full_BA_product():
    """N_i 必须含交叉项：手工对照 s·(B_iA_i − B_gA_g) 的显式展开。"""
    g, t, ref, _ = fake_round()
    con = construct_pair_v3(g, t, ref, 3.0, 7.0)
    n = eff_update_norm(con["a"], g)
    tot = 0.0
    for m in MODS:
        d = S * (con["a"][m][1].double() @ con["a"][m][0].double()
                 - g[m][1].double() @ g[m][0].double())
        tot += float((d ** 2).sum())
    assert abs(n - math.sqrt(tot)) < 1e-9


def test_eff_update_norm_nonzero_for_nontrivial_construction():
    g, t, ref, _ = fake_round()
    con = construct_pair_v3(g, t, ref, 5.0, 5.0)
    assert eff_update_norm(con["a"], g) > 0
    assert factor_change_norm(con["a"], g) > 0


def test_ba_rel_err_zero_for_identical():
    g, t, ref, _ = fake_round()
    rel, _ = ba_rel_err(g, g)
    assert rel <= 1e-12


def test_factor_norms_and_max_abs_err():
    g, t, ref, _ = fake_round()
    assert factor_norm(g, "A") > 0 and factor_norm(g, "B") > 0 and ba_norm(g) > 0
    assert max_abs_factor_err(g, g, "A") == 0.0


def test_cross_term_appears_in_single_client_update():
    """§3.6：单客户端有效更新含 D_B·D_A 交叉项（λ_A·(10−λ_B)·D_BD_A）。

    对照展开式：ΔW_a/s = λ_A·B_gD_A + (10−λ_B)·D_BA_g + λ_A(10−λ_B)·D_BD_A。
    """
    g, t, ref, _ = fake_round()
    la, lb = 3.0, 7.0
    con = construct_pair_v3(g, t, ref, la, lb)
    d_a, d_b = con["D_A"], con["D_B"]
    for m in MODS:
        b_g, a_g = g[m][1].double(), g[m][0].double()
        lhs = con["a"][m][1].double() @ con["a"][m][0].double() - b_g @ a_g
        rhs = la * (b_g @ d_a[m]) + (10 - lb) * (d_b[m] @ a_g) \
            + la * (10 - lb) * (d_b[m] @ d_a[m])
        assert torch.allclose(lhs, rhs, atol=1e-10), m
