"""E11 Task 2 core 单元测试：ε 构造公式、加权背景、联合重构、J 规则。

只测代数（无需 GPU / 模型）。运行：python -m pytest tests/test_e11_task2_core.py -q
"""
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.e11.task2_core import (construct_pair_core, joint_aggregate,  # noqa: E402
                                weighted_bg, EPS_GRID)

MODS = ["m1", "m2"]


def fake_pairs(seed: int, scale: float = 0.3) -> dict:
    gen = torch.Generator().manual_seed(seed)
    return {m: (torch.randn(8, 16, generator=gen) * scale,
                torch.randn(12, 8, generator=gen) * scale) for m in MODS}


def test_background_is_weighted_sum_not_mean():
    uploads = {f"client_{i:02d}": fake_pairs(i) for i in range(2, 10)}
    a_bg, b_bg = weighted_bg(uploads, p=0.1)
    a_mean = {m: sum(u[m][0] for u in uploads.values()) / 8 for m in MODS}
    assert not torch.allclose(a_bg["m1"], a_mean["m1"])
    manual = {m: 0.1 * sum(u[m][0] for u in uploads.values()) for m in MODS}
    assert torch.allclose(a_bg["m1"], manual["m1"])


def test_joint_aggregation_recovers_rotated_target():
    bg = {f"client_{i:02d}": fake_pairs(100 + i) for i in range(2, 10)}
    a_bg, b_bg = weighted_bg(bg)
    target = fake_pairs(999)
    for eps in EPS_GRID:
        con = construct_pair_core(target, a_bg, b_bg, eps)
        joint = joint_aggregate(con, "A", "B", bg)
        for m in MODS:
            assert torch.allclose(joint[m][0], target[m][0], atol=1e-5)
            assert torch.allclose(joint[m][1], target[m][1], atol=1e-5)


def test_split_factors_scale_with_epsilon():
    bg = {f"client_{i:02d}": fake_pairs(7 + i) for i in range(2, 10)}
    a_bg, b_bg = weighted_bg(bg)
    target = fake_pairs(55)
    da = {m: target[m][0] - a_bg[m] for m in MODS}
    con = construct_pair_core(target, a_bg, b_bg, 2.0)
    for m in MODS:
        assert torch.allclose(con["a"][m][0], 2.0 * da[m])
        assert torch.allclose(con["b"][m][0], 8.0 * da[m])


def test_epsilon_bounds_guard():
    bg = {f"client_{i:02d}": fake_pairs(i) for i in range(2, 10)}
    a_bg, b_bg = weighted_bg(bg)
    try:
        construct_pair_core(fake_pairs(1), a_bg, b_bg, 10.0)
        raised = False
    except AssertionError:
        raised = True
    assert raised


def test_screen_rule_picks_smaller_on_tie():
    # J_screen 并列时选较小 ε：模拟 J 值
    js = {e: 1.0 for e in EPS_GRID}
    eligible = list(EPS_GRID)
    best = min(eligible, key=lambda e: (js[e], e))
    assert best == 0.1
    js2 = {0.1: 2.0, 0.2: 1.5, 0.5: 1.5, 1.0: 3.0, 2.0: 4.0, 5.0: 9.0}
    assert min(eligible, key=lambda e: (js2[e], e)) == 0.2
