"""E10-A/B：目标更新提取与全谱低秩分解（方案 §5）。

- 目标模型 = 基础模型 + 秩 32 LoRA（集中训练），故 ΔW_goal 模块级 = s_train·B@A；
  同时提供通用 state-dict 差分校验（确认仅目标模块变化，§5.1）；
- 全谱 SVD：保存奇异值能量分布，产出 r∈{8,16} 两档 B_*/A_* 与误差 ρ_l、ρ_global（§5.3）；
- 截断目标：后续构造一律以 (α_r/r)·B_*A_* 为「已分解目标」（§5.3 末段）。
"""

from __future__ import annotations

import json
from pathlib import Path

from src.e10.lora_algebra import (LORA_A, LORA_B, from_pairs, fro_norm_lowrank,
                                  module_pairs)

EPS0 = 1e-12


def effective_delta_from_adapter(adapter: dict, s: float) -> dict[str, "torch.Tensor"]:
    """从训练后的适配器直接计算各模块有效更新（§5.4），物化 (d_out,d_in)。"""
    import torch
    deltas = {}
    for mod, (a, b) in module_pairs(adapter).items():
        deltas[mod] = s * (b @ a)
    return deltas


def verify_only_target_modules_changed(base_state: dict, target_state: dict,
                                       target_modules: list[str]) -> list[str]:
    """确认非目标参数逐元素相同、目标基础权重未被改动（§5.1）。"""
    problems = []
    for k in base_state:
        if k.endswith(LORA_A) or k.endswith(LORA_B):
            continue
        if k not in target_state:
            problems.append(f"missing {k} in target")
        elif not bool((base_state[k] == target_state[k]).all()):
            problems.append(f"base param changed: {k}")
    return problems


def svd_factorize(delta: "torch.Tensor", rank: int, alpha: int):
    """§5.2：ΔW ≈ (α/r)·B_*A_*。返回 (A_*, B_*, k, sigma, energy_frac)。"""
    import torch
    u, sigma, vh = torch.linalg.svd(delta, full_matrices=False)
    k = min(rank, int((sigma > sigma[0] * 1e-10).sum()))
    r_dim = rank
    b_star = torch.zeros(delta.shape[0], r_dim, dtype=delta.dtype, device=delta.device)
    a_star = torch.zeros(r_dim, delta.shape[1], dtype=delta.dtype, device=delta.device)
    scale = (rank / alpha) ** 0.5
    b_star[:, :k] = scale * u[:, :k] * sigma[:k].sqrt()
    a_star[:k, :] = scale * sigma[:k].sqrt().unsqueeze(1) * vh[:k, :]
    energy = float((sigma[:k] ** 2).sum() / (sigma ** 2).sum()) if len(sigma) else 1.0
    return a_star, b_star, k, sigma.tolist(), energy


def module_relative_error(delta: "torch.Tensor", a_star, b_star, s: float) -> float:
    import torch
    recon = s * (b_star @ a_star)
    num = torch.linalg.norm(delta - recon)
    den = torch.linalg.norm(delta) + EPS0
    return float(num / den)


def factorize_all(deltas: dict, ranks: list[int], alpha_for_rank: dict[int, int]):
    """全谱分解 + 各档误差。返回 {rank: {"adapter": peft扁平因子, "report": {...}}}。"""
    import torch
    results = {}
    for r in ranks:
        alpha = alpha_for_rank[r]
        s = alpha / r
        adapter, report = {}, {}
        g_num2, g_den2 = 0.0, 0.0
        for mod, delta in deltas.items():
            a_star, b_star, k, sigma, energy = svd_factorize(delta, r, alpha)
            adapter[mod + LORA_A] = a_star.contiguous()
            adapter[mod + LORA_B] = b_star.contiguous()
            rho_l = module_relative_error(delta, a_star, b_star, s)
            report[mod] = {"k": k, "rho_l": rho_l, "energy_frac_topk": energy,
                           "sigma_top8": sigma[:8], "sigma_count": len(sigma)}
            nd = float(torch.linalg.norm(delta))
            g_num2 += (rho_l * nd) ** 2
            g_den2 += nd ** 2
        results[r] = {"adapter": adapter, "report": report,
                      "rho_global": float((g_num2 / (g_den2 + EPS0)) ** 0.5)}
    return results


def save_factorization(out_dir: Path, deltas: dict, ranks, alpha_for_rank, meta: dict) -> dict:
    """落盘 target/ 目录（§9.2）：target_delta.safetensors、两档因子、factorization_report.json。"""
    import torch
    from safetensors.torch import save_file
    out_dir.mkdir(parents=True, exist_ok=True)
    results = factorize_all(deltas, ranks, alpha_for_rank)
    save_file({f"{mod}__delta": t.detach().cpu().contiguous() for mod, t in deltas.items()},
              str(out_dir / "target_delta.safetensors"))
    summary = {"meta": meta, "ranks": {}}
    for r, res in results.items():
        save_file({k: v.detach().cpu().contiguous() for k, v in res["adapter"].items()},
                  str(out_dir / f"factor_r{r}.safetensors"))
        summary["ranks"][str(r)] = {
            "alpha": alpha_for_rank[r], "scaling": alpha_for_rank[r] / r,
            "rho_global": res["rho_global"],
            "modules": {m: {"k": v["k"], "rho_l": v["rho_l"],
                            "energy_topk": v["energy_frac_topk"],
                            "sigma_top8": v["sigma_top8"]}
                        for m, v in res["report"].items()},
        }
    (out_dir / "factorization_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1))
    return results
