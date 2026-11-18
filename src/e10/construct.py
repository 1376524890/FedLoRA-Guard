"""E10-C：客户端 LoRA 因子反向构造（方案 §6.2 两客户端；§11.2 十客户端差值分配）。"""

from __future__ import annotations

from src.e10.lora_algebra import LORA_A, LORA_B, module_pairs


def construct_two_clients(target_pairs: dict, pa: float, pb: float, epsilon: float) -> dict:
    """§6.2 逐模块构造：
        A_a=εA_*; B_a=(1−p_bε)/p_a·B_*; A_b=(1−p_aε)/p_b·A_*; B_b=εB_*
    返回 {"client_a": pairs, "client_b": pairs}。"""
    if not (0 < epsilon < min(1 / pa, 1 / pb)):
        raise ValueError(f"ε={epsilon} 越界 (0, {min(1/pa, 1/pb)})")
    if abs(pa + pb - 1) > 1e-9:
        raise ValueError("pa+pb != 1")
    a_pairs, b_pairs = {}, {}
    for mod, (a_star, b_star) in target_pairs.items():
        a_pairs[mod] = (epsilon * a_star, (1 - pb * epsilon) / pa * b_star)
        b_pairs[mod] = ((1 - pa * epsilon) / pb * a_star, epsilon * b_star)
    return {"client_a": a_pairs, "client_b": b_pairs}


def construct_with_background(target_pairs: dict, background_pairs: dict,
                              pa: float, pb: float, epsilon: float) -> dict:
    """§11.2：对 ΔA=A_target−A_bg、ΔB=B_target−B_bg 应用同一构造公式，
    两个构造客户端补足背景因子与目标因子之间的差值。"""
    delta_pairs = {}
    for mod, (a_t, b_t) in target_pairs.items():
        a_bg, b_bg = background_pairs[mod]
        delta_pairs[mod] = (a_t - a_bg, b_t - b_bg)
    return construct_two_clients(delta_pairs, pa, pb, epsilon)


def theoretical_ratios(pa: float, pb: float, epsilon: float) -> dict:
    """§6.4 闭式系数：c_a=ε(1−p_bε)/p_a，c_b=ε(1−p_aε)/p_b，联合恢复=1，
    有效更新聚合=(2ε−ε²)，E_FA=(1−ε)²。"""
    return {"c_a": epsilon * (1 - pb * epsilon) / pa,
            "c_b": epsilon * (1 - pa * epsilon) / pb,
            "joint_factor": 1.0,
            "update_aggregate": 2 * epsilon - epsilon ** 2,
            "e_fa": (1 - epsilon) ** 2,
            "i_w": 1 - epsilon + 2 * pa * pb * epsilon ** 2}
