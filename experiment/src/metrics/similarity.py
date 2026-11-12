"""客户端 LoRA 更新相似度与残差分析（v0.2.0 修复版）。

相对 legacy/math_utils.py 的修复：
1. 残差中心化默认改为均值中心化，消除 medoid 被恶意客户端污染导致的整轮归零
   （见 reports/metric_audit.md §2）。
2. 显式输出 benign Q90 数值与分位数插值方式（修复字段缺失，metric_audit §1）。
3. 低秩 Gram 传入真实 LoRA 缩放 s=alpha/r，绝对范数语义正确（metric_audit §4）。
4. factor（A/B 拼接）指标降级为诊断，仅输出、不参与判定。
"""
from __future__ import annotations

from itertools import combinations

import torch

QUANTILE_INTERPOLATION = 'linear'  # torch.quantile 默认线性插值，显式记录


def cosine_matrix(vectors: torch.Tensor) -> torch.Tensor:
    if vectors.ndim != 2:
        raise ValueError('Expected [clients, parameters]')
    x = vectors.to(torch.float64)
    norm = x.norm(dim=1)
    if (norm <= 0).any():
        raise ValueError('Zero-norm client update: training or aggregation produced no change')
    x = x / norm[:, None]
    return (x @ x.T).clamp(-1.0, 1.0)


def _pair_stats_from_cos(cos: torch.Tensor, residual: torch.Tensor,
                         malicious_ids: list[int]) -> dict:
    """共用统计：恶意对 vs 正常对（余弦与残差余弦），输出 median 与 q90 数值。"""
    n = cos.shape[0]
    a, b = sorted(malicious_ids)
    benign = [i for i in range(n) if i not in malicious_ids]
    pairs = list(combinations(benign, 2))
    if len(pairs) < 2:
        raise ValueError('Not enough benign pairs for distributional statistics')
    benign_raw = torch.tensor([float(cos[i, j]) for i, j in pairs], dtype=torch.float64)
    benign_res = torch.tensor([float(residual[i, j]) for i, j in pairs], dtype=torch.float64)
    q90_raw = float(torch.quantile(benign_raw, 0.9, interpolation=QUANTILE_INTERPOLATION))
    q90_res = float(torch.quantile(benign_res, 0.9, interpolation=QUANTILE_INTERPOLATION))
    attack_raw, attack_res = float(cos[a, b]), float(residual[a, b])
    return {
        'attack_pair_raw_cos': attack_raw,
        'attack_pair_residual_cos': attack_res,
        'benign_median_raw_cos': float(benign_raw.median()),
        'benign_median_residual_cos': float(benign_res.median()),
        'benign_q90_raw_cos': q90_raw,
        'benign_q90_residual_cos': q90_res,
        'attack_excess_raw': attack_raw - float(benign_raw.median()),
        'attack_excess_residual': attack_res - float(benign_res.median()),
        'attack_exceeds_benign_q90_raw': bool(attack_raw > q90_raw),
        'attack_exceeds_benign_q90_residual': bool(attack_res > q90_res),
        'benign_pair_count': len(pairs),
        'quantile_interpolation': QUANTILE_INTERPOLATION,
    }


def mean_centered_residuals(vectors: torch.Tensor) -> torch.Tensor:
    """均值中心化后的残差余弦矩阵（v0.2.0 默认）。

    与 medoid 中心化不同：均值中心不与任何单一客户端重合，
    恶意客户端无法通过成为中心使自身残差归零。
    """
    if vectors.ndim != 2:
        raise ValueError('Expected [clients, parameters]')
    x = vectors.to(torch.float64)
    centered = x - x.mean(dim=0, keepdim=True)
    norms = centered.norm(dim=1)
    if (norms <= 0).any():
        # 全体均值下某客户端恰为零向量的概率可忽略；出现即为异常状态，必须报错
        raise ValueError('Client update equals the cross-client mean exactly; '
                         'degenerate configuration, refusing to fabricate a residual')
    unit = centered / norms[:, None]
    return (unit @ unit.T).clamp(-1, 1)


def medoid_centered_residuals(vectors: torch.Tensor) -> tuple[torch.Tensor, int]:
    """legacy medoid 中心化，仅诊断保留。返回 (残差余弦矩阵, medoid_id)。"""
    x = vectors.to(torch.float64)
    cos = cosine_matrix(x)
    dist = (2.0 - 2.0 * cos).clamp_min(0).sqrt()
    medoid = int(dist.sum(dim=1).argmin().item())
    centered = x - x[medoid]
    norms = centered.norm(dim=1).clamp_min(1e-12)
    unit = centered / norms[:, None]
    return (unit @ unit.T).clamp(-1, 1), medoid


def evidence_separation_factor(updates: torch.Tensor, malicious_ids: list[int]) -> dict:
    """诊断用途：A/B 因子拼接向量的余弦/残差统计（legacy 语义 + 修复输出）。

    拼接参数向量无几何意义（metric_audit §3），仅作对照诊断，不参与判定。
    """
    cos = cosine_matrix(updates)
    residual_mean = mean_centered_residuals(updates)
    residual_medoid, medoid = medoid_centered_residuals(updates)
    out = _pair_stats_from_cos(cos, residual_mean, malicious_ids)
    out['diagnostic_medoid'] = {
        'medoid_client_id': medoid,
        'medoid_in_malicious': medoid in malicious_ids,
        'attack_pair_residual_cos_medoid_centered': float(
            residual_medoid[malicious_ids[0], malicious_ids[1]]),
    }
    out['metric_semantics'] = 'factor_concat_diagnostic_only'
    return out


def gram_to_cos_and_residuals(gram: torch.Tensor, malicious_ids: list[int],
                              center: str = 'mean') -> dict:
    """从低秩有效更新 Gram 矩阵计算余弦与残差统计。

    gram[i,j] = <ΔW_i, ΔW_j>_F（已含 scaling）。
    center: 'mean'（默认，v0.2.0）或 'medoid'（诊断）。
    残差在 Gram 空间完成：||x_i - x_c||^2 = g_ii + g_cc - 2 g_ic。
    """
    n = gram.shape[0]
    if gram.shape != (n, n) or n < 5:
        raise ValueError('Invalid Gram matrix')
    diag = gram.diag()
    norms = diag.clamp_min(0).sqrt()
    if (norms <= 0).any():
        raise ValueError('Zero effective LoRA update norm')
    cos = (gram / norms[:, None] / norms[None, :]).clamp(-1, 1)

    if center == 'mean':
        # 均值向量的 Gram 量：mean_i·mean_j = (1/n^2) ΣΣ g_ab
        total = gram.sum()
        row_sums = gram.sum(dim=1)
        # <x_i - m, x_j - m> = g_ij - row_i/n - row_j/n + total/n^2
        centered = gram - row_sums[:, None] / n - row_sums[None, :] / n + total / (n * n)
    elif center == 'medoid':
        dist_sq = (diag[:, None] + diag[None, :] - 2 * gram).clamp_min(0)
        medoid = int(dist_sq.sum(dim=1).argmin())
        centered = gram - gram[:, medoid][:, None] - gram[medoid, :][None, :] + gram[medoid, medoid]
    else:
        raise ValueError(f'Unknown center: {center}')
    cnorm = centered.diag().clamp_min(0).sqrt()
    if (cnorm <= 1e-12).any() and center == 'mean':
        raise ValueError('Degenerate mean-centered residual (zero norm)')
    cnorm_safe = cnorm.clamp_min(1e-12)
    residual = (centered / cnorm_safe[:, None] / cnorm_safe[None, :]).clamp(-1, 1)

    out = _pair_stats_from_cos(cos, residual, malicious_ids)
    out['centering'] = center
    if center == 'medoid':
        out['medoid_client_id'] = medoid
        out['medoid_in_malicious'] = medoid in malicious_ids
    return out


def effective_lora_gram(states: list[dict[str, torch.Tensor]],
                        global_state: dict[str, torch.Tensor],
                        scaling: float,
                        selected_module_count: int = 4) -> tuple[torch.Tensor, list[str]]:
    """真实 LoRA 有效更新 s·(B A − B_g A_g) 的 Gram，不构造稠密矩阵。

    scaling 必须来自实际 adapter 配置（lora_alpha / r），修复 legacy 遗漏。
    """
    if scaling <= 0:
        raise ValueError('LoRA scaling must be positive (alpha/r)')
    names = sorted(k[:-len('.lora_A.weight')] for k in global_state
                   if k.endswith('.lora_A.weight') and '.q_proj.' in k)
    if len(names) < selected_module_count or selected_module_count < 1:
        raise RuntimeError(f'Not enough q_proj LoRA modules: {len(names)}')
    if selected_module_count == 1:
        sample = [names[len(names) // 2]]
    else:
        sample = [names[i * (len(names) - 1) // (selected_module_count - 1)]
                  for i in range(selected_module_count)]
    n = len(states)
    gram = torch.zeros((n, n), dtype=torch.float64)
    for stem in sample:
        key_a, key_b = stem + '.lora_A.weight', stem + '.lora_B.weight'
        if key_b not in global_state:
            raise RuntimeError(f'Missing LoRA B factor: {stem}')
        a_g = global_state[key_a].to(torch.float64)
        b_g = global_state[key_b].to(torch.float64)
        fac = []
        for state in states:
            a_i = state[key_a].to(torch.float64)
            b_i = state[key_b].to(torch.float64)
            fac.append((torch.cat((b_i, -b_g), dim=1), torch.cat((a_i, a_g), dim=0)))
        for i in range(n):
            ui, vi = fac[i]
            for j in range(i, n):
                uj, vj = fac[j]
                val = float(((ui.T @ uj) * (vi @ vj.T)).sum())
                gram[i, j] += val * scaling * scaling
                if i != j:
                    gram[j, i] += val * scaling * scaling
    return gram, sample


def average_states(states: list[dict[str, torch.Tensor]],
                   weights: list[float] | None = None) -> dict[str, torch.Tensor]:
    """FedAvg：等权（默认）或按客户端样本数加权；平均对象为 adapter 原始参数。

    语义显式记录：参数平均 ≠ 有效更新平均（见 code_audit §2）。
    """
    if not states:
        raise ValueError('No client states for FedAvg')
    if weights is None:
        weights = [1.0 / len(states)] * len(states)
    if len(weights) != len(states) or abs(sum(weights) - 1.0) > 1e-9 or any(w < 0 for w in weights):
        raise ValueError('Invalid FedAvg weights')
    names = sorted(states[0])
    for item in states:
        if sorted(item) != names:
            raise ValueError('Mismatched adapter parameter names')
    averaged: dict[str, torch.Tensor] = {}
    for name in names:
        ref_shape = states[0][name].shape
        acc = torch.zeros(ref_shape, dtype=torch.float64)
        for w, s in zip(weights, states):
            t = s[name]
            if t.shape != ref_shape:
                raise ValueError(f'Adapter shape mismatch: {name}')
            acc += w * t.to(torch.float64)
        averaged[name] = acc.to(states[0][name].dtype)
    return averaged


def flattened_deltas(states: list[dict[str, torch.Tensor]],
                     global_state: dict[str, torch.Tensor]) -> torch.Tensor:
    names = sorted(global_state)
    if not names:
        raise ValueError('Empty LoRA adapter state')
    rows = []
    for state in states:
        if sorted(state) != names:
            raise ValueError('Mismatched LoRA keys')
        rows.append(torch.cat([(state[k] - global_state[k]).reshape(-1).float() for k in names]))
    return torch.stack(rows, dim=0)


def permutation_test_pair_excess(pair_values: torch.Tensor, labels_malicious: list[int],
                                 n: int, n_permutations: int = 10000,
                                 seed: int = 0) -> dict:
    """客户端身份置换检验：恶意对 excess 是否显著超出随机身份分配。

    pair_values: 全部 C(n,2) 对的残差余弦（与 combinations(range(n),2) 同序）。
    正式检测的统计基础（metric_audit §5 末段）；预测试阶段即可调用。
    """
    from itertools import combinations
    all_pairs = list(combinations(range(n), 2))
    if len(all_pairs) != pair_values.numel():
        raise ValueError('pair count mismatch')
    a, b = sorted(labels_malicious)
    observed = float(pair_values[all_pairs.index((a, b))])
    rng = torch.Generator().manual_seed(seed)
    exceed = 0
    for _ in range(n_permutations):
        perm = torch.randperm(n, generator=rng).tolist()
        pa, pb = sorted((perm.index(a), perm.index(b)))
        # perm.index(a) 是身份 a 被分到的客户端槽位
        if float(pair_values[all_pairs.index((min(pa, pb), max(pa, pb)))]) >= observed:
            exceed += 1
    return {
        'observed_pair_cos': observed,
        'permutations': n_permutations,
        'p_value': (exceed + 1) / (n_permutations + 1),
        'test': 'identity_permutation_greater_or_equal',
    }
