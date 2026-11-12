"""v0.2.0 指标修复的单元测试（不依赖 GPU，合成矩阵验证数学性质）。"""
import math

import pytest
import torch

from src.metrics.similarity import (
    average_states,
    cosine_matrix,
    effective_lora_gram,
    evidence_separation_factor,
    gram_to_cos_and_residuals,
    mean_centered_residuals,
    permutation_test_pair_excess,
)


def _toy_states(n=10, seed=0, dim=32, colluding=(6, 8), collusion=0.6):
    """合成 LoRA 态：恶意对更新共享一个大共同分量，正常对独立。"""
    g = torch.Generator().manual_seed(seed)
    global_state = {'m.lora_A.weight': torch.zeros(4, dim), 'm.lora_B.weight': torch.zeros(dim, 4)}
    shared = torch.randn(dim, 4, generator=g) * collusion
    states = []
    for i in range(n):
        noise = torch.randn(dim, 4, generator=g) * 0.3
        delta = shared + noise if i in colluding else noise
        states.append({
            'm.lora_A.weight': global_state['m.lora_A.weight'] + torch.randn(4, dim, generator=g) * 0.01,
            'm.lora_B.weight': global_state['m.lora_B.weight'] + delta,
        })
    return states, global_state


def test_mean_centered_residuals_not_degenerate():
    x = torch.randn(10, 64, dtype=torch.float64)
    r = mean_centered_residuals(x)
    assert torch.allclose(r.diag(), torch.ones(10, dtype=torch.float64), atol=1e-9)
    # 对称
    assert torch.allclose(r, r.T)


def test_mean_centering_survives_colluding_center():
    """medoid 缺陷回归测试：恶意客户端成为中心不再导致恶意对残差为 0。"""
    g = torch.Generator().manual_seed(3)
    common = torch.randn(1, 64, generator=g)
    x = torch.randn(10, 64, generator=g)
    x[6] += 5 * common[0]
    x[8] += 5 * common[0]
    x[8] *= 0.0  # 构造 x8 == x6 方向极端情形前先恢复
    x = torch.randn(10, 64, generator=g)
    x[6] += 5 * common[0]
    x[8] += 5 * common[0]
    res = mean_centered_residuals(x)
    # 恶意对残差应为高值（共享共同分量），绝不为 0
    assert res[6, 8] > 0.5, f'colluding pair residual collapsed: {res[6, 8]}'


def test_q90_fields_present():
    states, gs = _toy_states()
    updates = torch.stack([(s['m.lora_B.weight'] - gs['m.lora_B.weight']).reshape(-1)
                           for s in states]).double()
    out = evidence_separation_factor(updates, [6, 8])
    for key in ('benign_q90_raw_cos', 'benign_q90_residual_cos', 'benign_pair_count',
                'quantile_interpolation'):
        assert key in out, f'missing {key} (v0.2.0 fix)'
    assert out['benign_pair_count'] == 28
    assert -1.0 <= out['benign_q90_raw_cos'] <= 1.0


def test_gram_scaling_applied():
    states, gs = _toy_states()
    gram_s1, _ = effective_lora_gram(states, gs, scaling=1.0)
    gram_s2, _ = effective_lora_gram(states, gs, scaling=2.0)
    # 内积按 s^2 缩放
    assert torch.allclose(gram_s2, 4.0 * gram_s1, rtol=1e-9)
    # 余弦不受均匀缩放影响
    cos1 = gram_to_cos_and_residuals(gram_s1, [6, 8])['attack_pair_raw_cos']
    cos2 = gram_to_cos_and_residuals(gram_s2, [6, 8])['attack_pair_raw_cos']
    assert abs(cos1 - cos2) < 1e-12


def test_gram_matches_explicit_frobenius():
    """指引 §3.6：低秩 Gram == 显式 ΔW 内积（含 scaling）。"""
    g = torch.Generator().manual_seed(7)
    dim, r = 96, 8
    a_g, b_g = torch.randn(r, dim, generator=g).double(), torch.randn(dim, r, generator=g).double()
    st = []
    for _ in range(6):
        a = torch.randn(r, dim, generator=g).double()
        b = torch.randn(dim, r, generator=g).double()
        st.append({'m.lora_A.weight': a, 'm.lora_B.weight': b})
    gs = {'m.lora_A.weight': a_g, 'm.lora_B.weight': b_g}
    scaling = 2.0
    gram, _ = effective_lora_gram(st, gs, scaling=scaling, selected_module_count=1)
    for i in range(6):
        for j in range(6):
            di = scaling * (st[i]['m.lora_B.weight'] @ st[i]['m.lora_A.weight'] - b_g @ a_g)
            dj = scaling * (st[j]['m.lora_B.weight'] @ st[j]['m.lora_A.weight'] - b_g @ a_g)
            explicit = float((di * dj).sum())
            assert abs(gram[i, j] - explicit) / (abs(explicit) + 1e-30) < 1e-10


def test_fedavg_weighted_average():
    s1 = {'a': torch.tensor([1.0]), 'b': torch.tensor([2.0])}
    s2 = {'a': torch.tensor([3.0]), 'b': torch.tensor([4.0])}
    eq = average_states([s1, s2])
    w = average_states([s1, s2], weights=[0.75, 0.25])
    assert eq['a'].item() == 2.0 and w['a'].item() == 1.5
    with pytest.raises(ValueError):
        average_states([s1, s2], weights=[0.5, 1.5])


def test_permutation_test_detects_collusion():
    from itertools import combinations
    n = 10
    g = torch.Generator().manual_seed(11)
    x = torch.randn(n, 64, generator=g)
    common = torch.randn(64, generator=g)
    x[6] += 4 * common
    x[8] += 4 * common
    res = mean_centered_residuals(x)
    pairs = list(combinations(range(n), 2))
    vals = torch.tensor([float(res[i, j]) for i, j in pairs], dtype=torch.float64)
    out = permutation_test_pair_excess(vals, [6, 8], n, n_permutations=2000, seed=1)
    assert out['observed_pair_cos'] > 0.5
    assert out['p_value'] <= 0.05, f'colluding pair not detected: {out}'


def test_cosine_matrix_rejects_zero_row():
    x = torch.eye(3)
    x[1] = 0
    with pytest.raises(ValueError):
        cosine_matrix(x)


def test_no_nan_in_outputs():
    states, gs = _toy_states(seed=5)
    updates = torch.stack([(s['m.lora_B.weight'] - gs['m.lora_B.weight']).reshape(-1)
                           for s in states]).double()
    out = evidence_separation_factor(updates, [6, 8])
    for key in ('attack_excess_residual', 'benign_q90_residual_cos'):
        v = out[key]
        assert v is None or isinstance(v, bool) or (isinstance(v, float) and math.isfinite(v))
