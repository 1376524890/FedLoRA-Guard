"""实际 LoRA 更新的数学分析，纯 PyTorch，不生成测试替身。"""
import math
from itertools import combinations

import torch


def cosine_matrix(vectors: torch.Tensor) -> torch.Tensor:
    if vectors.ndim != 2:
        raise ValueError('Expected [clients, parameters]')
    x = vectors.to(torch.float64)
    norm = x.norm(dim=1)
    if (norm <= 0).any():
        raise ValueError('Zero-norm client update: training or aggregation is not producing changes')
    x = x / norm[:, None]
    return (x @ x.T).clamp(-1.0, 1.0)


def evidence_separation(updates: torch.Tensor, malicious_ids: list[int]) -> dict:
    """使用真实客户端参数更新，比较攻击者对与正常客户端对。"""
    if len(malicious_ids) != 2:
        raise ValueError('This pretest requires exactly two attacker-controlled clients')
    n = updates.shape[0]
    if updates.ndim != 2 or n < 5:
        raise ValueError('Expected at least five client updates')
    a, b = sorted(malicious_ids)
    if a < 0 or b >= n or a == b:
        raise ValueError('Invalid malicious IDs')
    sims = cosine_matrix(updates)
    dist = (2.0 - 2.0 * sims).clamp_min(0).sqrt()
    medoid = int(dist.sum(dim=1).argmin().item())
    centered = updates.to(torch.float64) - updates[medoid].to(torch.float64)
    norms = centered.norm(dim=1)
    residual = centered / norms.clamp_min(1e-12)[:, None]
    residual_sims = (residual @ residual.T).clamp(-1, 1)
    benign = [i for i in range(n) if i not in malicious_ids]
    pairs = list(combinations(benign, 2))
    benign_raw = [float(sims[i, j]) for i, j in pairs]
    benign_res = [float(residual_sims[i, j]) for i, j in pairs]
    attack_raw = float(sims[a, b])
    attack_res = float(residual_sims[a, b])
    q90_raw = float(torch.quantile(torch.tensor(benign_raw), 0.9).item())
    q90_res = float(torch.quantile(torch.tensor(benign_res), 0.9).item())
    median_raw = float(torch.median(torch.tensor(benign_raw)).item())
    median_res = float(torch.median(torch.tensor(benign_res)).item())
    return {
        'attack_pair_raw_cos': attack_raw,
        'attack_pair_residual_cos': attack_res,
        'benign_median_raw_cos': median_raw,
        'benign_median_residual_cos': median_res,
        'attack_excess_raw': attack_raw - median_raw,
        'attack_excess_residual': attack_res - median_res,
        'attack_exceeds_benign_q90_raw': attack_raw > q90_raw,
        'attack_exceeds_benign_q90_residual': attack_res > q90_res,
        'benign_pair_count': len(pairs),
        'medoid_client_id': medoid,
    }


def average_states(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError('No client states for FedAvg')
    names = sorted(states[0])
    for item in states:
        if sorted(item) != names:
            raise ValueError('Mismatched adapter parameter names')
    averaged: dict[str, torch.Tensor] = {}
    for name in names:
        ref_shape = states[0][name].shape
        tensors = [s[name].to(torch.float64) for s in states]
        if any(t.shape != ref_shape for t in tensors):
            raise ValueError(f'Adapter shape mismatch: {name}')
        averaged[name] = torch.stack(tensors, dim=0).mean(dim=0).to(states[0][name].dtype)
    return averaged


def flattened_deltas(states: list[dict[str, torch.Tensor]], global_state: dict[str, torch.Tensor]) -> torch.Tensor:
    names = sorted(global_state)
    if not names:
        raise ValueError('Empty LoRA adapter state')
    out = []
    for state in states:
        if sorted(state) != names:
            raise ValueError('Mismatched LoRA keys')
        out.append(torch.cat([(state[k] - global_state[k]).reshape(-1).float() for k in names]))
    return torch.stack(out, dim=0)


def binomial_se(p: float, n: int) -> float:
    if not (0 <= p <= 1) or n <= 0:
        raise ValueError('Invalid binomial estimate')
    return math.sqrt(p * (1-p) / n)


def effective_lora_gram(states: list[dict[str, torch.Tensor]],
                        global_state: dict[str, torch.Tensor],
                        selected_module_count: int = 4) -> tuple[torch.Tensor, list[str]]:
    """真实 LoRA 有效更新的 Gram：完全不创建稠密 B@A。"""
    names = sorted(k[:-len('.lora_A.weight')] for k in global_state
                   if k.endswith('.lora_A.weight') and '.q_proj.' in k)
    if len(names) < selected_module_count:
        raise RuntimeError(f'Not enough q_proj LoRA modules to compute effective Gram: {len(names)}')
    sample = [names[i * (len(names) - 1) // (selected_module_count - 1)]
              for i in range(selected_module_count)]
    n = len(states)
    gram = torch.zeros((n, n), dtype=torch.float64)
    for stem in sample:
        key_a = stem + '.lora_A.weight'
        key_b = stem + '.lora_B.weight'
        if key_b not in global_state:
            raise RuntimeError(f'Missing LoRA B factor: {stem}')
        a_g = global_state[key_a].to(torch.float64)
        b_g = global_state[key_b].to(torch.float64)
        fac = []
        for state in states:
            a_i = state[key_a].to(torch.float64)
            b_i = state[key_b].to(torch.float64)
            u_i = torch.cat((b_i, -b_g), dim=1)
            v_i = torch.cat((a_i, a_g), dim=0)
            fac.append((u_i, v_i))
        for i in range(n):
            ui, vi = fac[i]
            for j in range(i, n):
                uj, vj = fac[j]
                uv = ui.T @ uj
                vv = vi @ vj.T
                val = float((uv * vv).sum())
                gram[i, j] += val
                if i != j:
                    gram[j, i] += val
    return gram, sample


def evidence_separation_gram(gram: torch.Tensor, malicious_ids: list[int]) -> dict:
    """利用低秩更新 Gram 进行 Medoid 中心化和恶意对相似度分析。"""
    n = gram.shape[0]
    if gram.shape != (n, n) or n < 5:
        raise ValueError('Invalid Gram matrix')
    a, b = sorted(malicious_ids)
    if a == b or a < 0 or b >= n:
        raise ValueError('Invalid malicious IDs')
    norms = gram.diag().clamp_min(0).sqrt()
    if (norms <= 0).any():
        raise ValueError('Zero effective LoRA update norm')
    cos = (gram / norms[:, None] / norms[None, :]).clamp(-1, 1)
    distance = (gram.diag()[:, None] + gram.diag()[None, :] - 2 * gram).clamp_min(0).sqrt()
    medoid = int(distance.sum(dim=1).argmin())
    centered = gram - gram[:, medoid][:, None] - gram[medoid, :][None, :] + gram[medoid, medoid]
    rnorm = centered.diag().clamp_min(0).sqrt().clamp_min(1e-12)
    residual = (centered / rnorm[:, None] / rnorm[None, :]).clamp(-1, 1)
    benign = [i for i in range(n) if i not in malicious_ids]
    pairs = list(combinations(benign, 2))
    raw_benign = torch.tensor([float(cos[i, j]) for i, j in pairs])
    res_benign = torch.tensor([float(residual[i, j]) for i, j in pairs])
    return {
        'attack_pair_raw_cos': float(cos[a, b]),
        'attack_pair_residual_cos': float(residual[a, b]),
        'benign_median_raw_cos': float(raw_benign.median()),
        'benign_median_residual_cos': float(res_benign.median()),
        'attack_excess_raw': float(cos[a, b] - raw_benign.median()),
        'attack_excess_residual': float(residual[a, b] - res_benign.median()),
        'attack_exceeds_benign_q90_raw': bool(cos[a, b] > torch.quantile(raw_benign, 0.9)),
        'attack_exceeds_benign_q90_residual': bool(residual[a, b] > torch.quantile(res_benign, 0.9)),
        'benign_pair_count': len(pairs),
        'medoid_client_id': medoid,
    }
