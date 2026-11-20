"""E10 LoRA 聚合代数核心（方案 §3/§4/§7/§14.3）。

约定：
- 适配器为 peft 扁平格式 {key_lora_A: (r,d_in), key_lora_B: (d_out,r)}（fp32）；
- 每模块配对键："...q_proj.lora_A.weight" ↔ "...q_proj.lora_B.weight"；
- 缩放 s = alpha/r 在推理时由 peft 应用，本模块所有 ΔW 显式乘 s；
- 范数与内积用低秩迹恒等式计算（§14.3），不物化大矩阵：
    ‖BA‖_F² = tr((BᵀB)(AAᵀ)),  ⟨BA, CD⟩_F = tr(BᵀC D Aᵀ)
"""

from __future__ import annotations

import math
import re

LORA_A = ".lora_A.weight"
LORA_B = ".lora_B.weight"


def module_pairs(adapter: dict) -> dict[str, tuple]:
    """{module: (A, B)}，module 为去掉 lora 后缀的键。"""
    pairs = {}
    for k, v in adapter.items():
        if k.endswith(LORA_A):
            mod = k[: -len(LORA_A)]
            b = adapter.get(mod + LORA_B)
            if b is None:
                raise KeyError(f"missing B for {mod}")
            pairs[mod] = (v, b)
    return pairs


def from_pairs(pairs: dict[str, tuple]) -> dict:
    out = {}
    for mod, (a, b) in pairs.items():
        out[mod + LORA_A] = a
        out[mod + LORA_B] = b
    return out


def check_shapes(pairs: dict[str, tuple], rank: int | None = None) -> list[str]:
    """§15.1 形状检查：A:(r,d_in)、B:(d_out,r)。"""
    import torch
    problems = []
    for mod, (a, b) in pairs.items():
        if a.dim() != 2 or b.dim() != 2:
            problems.append(f"{mod}: not 2-D")
            continue
        r_a, d_in = a.shape
        d_out, r_b = b.shape
        if r_a != r_b:
            problems.append(f"{mod}: rank mismatch A={r_a} B={r_b}")
        if rank is not None and r_a != rank:
            problems.append(f"{mod}: rank {r_a} != {rank}")
        if b.shape[1] != a.shape[0]:
            problems.append(f"{mod}: inner dim mismatch")
        if not (a.dtype == b.dtype and a.dtype == torch.float32):
            problems.append(f"{mod}: dtype {a.dtype}/{b.dtype} != float32")
    return problems


# ---------------------------------------------------------------- 低秩数值（§14.3）

def _trace_mm(x: "torch.Tensor", y: "torch.Tensor") -> float:
    import torch
    return float(torch.trace(x @ y))


def fro_norm_lowrank(a: "torch.Tensor", b: "torch.Tensor") -> float:
    """‖BA‖_F = sqrt(tr(BᵀB · AAᵀ))，B:(d_out,r), A:(r,d_in)。"""
    import torch
    return float(torch.sqrt(torch.clamp(torch.trace((b.T @ b) @ (a @ a.T)), min=0)))


def fro_inner_lowrank(b1, a1, b2, a2) -> float:
    """⟨B₁A₁, B₂A₂⟩_F = tr(B₁ᵀB₂ · A₂A₁ᵀ)。"""
    import torch
    return _trace_mm(b1.T @ b2, a2 @ a1.T)


def cosine_lowrank(b1, a1, b2, a2, eps: float = 1e-12) -> float:
    num = fro_inner_lowrank(b1, a1, b2, a2)
    den = math.sqrt(max(fro_norm_lowrank(a1, b1) ** 2, 0) * max(fro_norm_lowrank(a2, b2) ** 2, 0))
    return num / max(den, eps)


# ---------------------------------------------------------------- 聚合协议（§3.2/§3.3）

def aggregate_lora_factors(pairs_list: list[dict], weights: list[float]) -> dict:
    """主协议：A、B 分别加权平均 → (Ā, B̄)。"""
    mods = set().union(*[set(p) for p in pairs_list])
    out = {}
    for mod in mods:
        a_bar = sum(w * p[mod][0] for p, w in zip(pairs_list, weights))
        b_bar = sum(w * p[mod][1] for p, w in zip(pairs_list, weights))
        out[mod] = (a_bar, b_bar)
    return out


def aggregate_effective_updates(pairs_list: list[dict], weights: list[float]) -> dict:
    """对照协议：先乘积再加权和。以扩展秩拼接表示（§17.1 保留真实秩）：
    B_cat=[B₁,…,B_n], A_cat 按 w_i 缩放堆叠 → s·B_catA_cat = s·Σ w_i B_iA_i。
    """
    mods = set().union(*[set(p) for p in pairs_list])
    out = {}
    for mod in mods:
        b_cat = _cat([p[mod][1] for p in pairs_list], dim=1)          # (d_out, n·r)
        a_cat = _cat([w * p[mod][0] for p, w in zip(pairs_list, weights)], dim=0)  # (n·r, d_in)
        out[mod] = (a_cat, b_cat)
    return out


def _cat(tensors, dim):
    import torch
    return torch.cat(list(tensors), dim=dim)


# ---------------------------------------------------------------- 聚合差异与交叉项（§4）

def aggregation_difference_numeric(factor_pairs: dict, update_pairs: dict) -> dict:
    """E_1 = ΔW_factor − ΔW_update（逐模块低秩差）。返回 {mod: (A_E, B_E)} 使
    s·B_E A_E = E_1 的等价低秩表示（B_E=[B̄,−B_cat], A_E 堆叠）。"""
    out = {}
    for mod in factor_pairs:
        a_f, b_f = factor_pairs[mod]
        a_u, b_u = update_pairs[mod]
        if a_f.shape[0] != a_u.shape[0]:  # 秩不同（更新协议秩 n·r）
            a_e = _cat([a_f, a_u], dim=0)
            b_e = _cat([b_f, -b_u], dim=1)
        else:
            a_e = _cat([a_f, a_u], dim=0)
            b_e = _cat([b_f, -b_u], dim=1)
        out[mod] = (a_e, b_e)
    return out


def e_fa_closed_form_two(pairs_a: dict, pairs_b: dict, s: float,
                         pa: float, pb: float) -> dict:
    """E_2 = s·p_a·p_b·(B_a−B_b)(A_b−A_a)（§4.1 闭式）。"""
    mods = set(pairs_a) | set(pairs_b)
    out = {}
    for mod in mods:
        db = pairs_a[mod][1] - pairs_b[mod][1]          # D_B = B_a − B_b
        da = pairs_b[mod][0] - pairs_a[mod][0]          # D_A = A_b − A_a
        out[mod] = (s * pa * pb * da, db)               # s·p_a·p_b·D_B·D_A
    return out


def cross_term_cab(pairs_a: dict, pairs_b: dict, s: float, pa: float, pb: float) -> dict:
    """C_ab = s·p_a·p_b·(B_aA_b + B_bA_a)（§4.2 纯跨客户端交叉项，单独保存）。

    表示约定：返回 (A_cat, B_cat)，对应矩阵 B_cat@A_cat，已含 s·p_a·p_b 因子。
    """
    mods = set(pairs_a) | set(pairs_b)
    out = {}
    for mod in mods:
        a_a, b_a = pairs_a[mod]      # A_a:(r,d_in), B_a:(d_out,r)
        a_b, b_b = pairs_b[mod]
        block_ab = (a_b, (s * pa * pb) * b_a)   # 项1: (s·p_a·p_b·B_a) @ A_b
        block_ba = (a_a, (s * pa * pb) * b_b)   # 项2: (s·p_a·p_b·B_b) @ A_a
        out[mod] = (_cat([block_ab[0], block_ba[0]], dim=0),
                    _cat([block_ab[1], block_ba[1]], dim=1))
    return out


def relative_residual(pairs_e1: dict, pairs_e2: dict, eps: float = 1e-12) -> float:
    """全模型相对残差 ‖E1−E2‖_F² / (‖E1‖_F²+eps)，用低秩迹恒等式精确计算。"""
    num2 = 0.0
    den2 = 0.0
    for mod in pairs_e1:
        a1, b1 = pairs_e1[mod]
        a2, b2 = pairs_e2[mod]
        # ‖E1−E2‖² = ‖E1‖² + ‖E2‖² − 2⟨E1,E2⟩
        n1 = fro_norm_lowrank(a1, b1) ** 2
        n2 = fro_norm_lowrank(a2, b2) ** 2
        ip = fro_inner_lowrank(b1, a1, b2, a2)
        num2 += max(n1 + n2 - 2 * ip, 0.0)
        den2 += n1
    return math.sqrt(num2 / (den2 + eps))


# ---------------------------------------------------------------- 四分支（§7）与构造校验（§6）

def four_branch_updates(pairs_a: dict, pairs_b: dict, s: float,
                        pa: float, pb: float) -> dict:
    """固定槽位四分支的参数更新（因子协议）：
    00=0；10=s(p_aB_a)(p_aA_a)；01=s(p_bB_b)(p_bA_b)；11=s·B̄Ā。
    返回 {branch: {mod:(A,B)}}，均为「适配器因子」形式（加载时经 s 生效）。"""
    zero = {mod: (_zeros_like(pairs_a[mod][0]), _zeros_like(pairs_a[mod][1]))
            for mod in pairs_a}
    joint = aggregate_lora_factors([pairs_a, pairs_b], [pa, pb])
    return {
        "00": zero,
        "10": {mod: (pa * pairs_a[mod][0], pa * pairs_a[mod][1]) for mod in pairs_a},
        "01": {mod: (pb * pairs_b[mod][0], pb * pairs_b[mod][1]) for mod in pairs_b},
        "11": joint,
    }


def _zeros_like(t):
    return t.clone().fill_(0)


def update_ratio(pairs_client: dict, pairs_target: dict, s: float) -> float:
    """单客户端有效更新相对目标的全局比例系数 c：ΔW_client = c·ΔW_goal 的最小二乘估计。
    构造正确时恒等于理论闭式（§6.4）。"""
    num = sum(fro_inner_lowrank(pairs_client[m][1], pairs_client[m][0],
                                pairs_target[m][1], pairs_target[m][0]) for m in pairs_target)
    den = sum(fro_norm_lowrank(pairs_target[m][0], pairs_target[m][1]) ** 2 for m in pairs_target)
    return num / max(den, 1e-30)


# ---------------------------------------------------------------- 组合级特征（§10.1/§13.1）

def pair_features(pairs_i: dict, pairs_j: dict, s: float, pi: float, pj: float,
                  factor_norm2: dict | None = None, eps: float = 1e-12) -> dict:
    """客户端对 (i,j)：E_ij、C_ij 及归一化分数 S^E、S^C。
    factor_norm2 可传入 {mod: ‖ΔW_factor,mod‖²} 以复用。"""
    e_ij = e_fa_closed_form_two(pairs_i, pairs_j, s, pi, pj)
    c_ij = cross_term_cab(pairs_i, pairs_j, s, pi, pj)
    num_e = sum(fro_norm_lowrank(a, b) ** 2 for a, b in e_ij.values())
    num_c = sum(fro_norm_lowrank(a, b) ** 2 for a, b in c_ij.values())
    den = sum(factor_norm2.values()) if factor_norm2 else 1.0
    return {"E_ij_norm": math.sqrt(num_e), "C_ij_norm": math.sqrt(num_c),
            "S_E": math.sqrt(num_e / (den + eps)), "S_C": math.sqrt(num_c / (den + eps))}


def scaling() -> float:
    """D5：全程 s = α/r = 2（r8α16 / r16α32 / r32α64）。"""
    return 16 / 8
