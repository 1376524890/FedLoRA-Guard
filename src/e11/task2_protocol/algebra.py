"""Task2 协议代数（E11_TASK2_DESIGN.md §3–§6 的逐项实现；公式编号见 FORMULA_CODE_MAP.md）。

记号约定（与 src/e10/lora_algebra.py 一致）：
    P ≡ peft ``lora_A.weight``，形状 (r, d_in)；Q ≡ peft ``lora_B.weight``，形状 (d_out, r)；
    有效权重 W = W₀ + s·Q·P，s = α/r（r8/α16 → s = 2）。
    ``Pairs = {stem: (P, Q)}``，stem 为去掉 ``.lora_A.weight`` 后缀的 peft 键。

数值约定：
- 全部构造与范数在 float64 中完成；
- 构造结果序列化为 float32（``to_fp32``）后再走仓库真实 ``src.fed.aggregate.fedavg``；
- ``server_aggregate`` 是本包**唯一**的服务器聚合入口，不接受手写平均替代。
"""
from __future__ import annotations

import math
from typing import Iterable

import torch

from src.e10.lora_algebra import from_pairs, module_pairs
from src.errors import ConfigurationError
from src.fed.aggregate import fedavg as repo_fedavg  # 仓库真实 FedAvg（唯一聚合实现）

Pairs = dict[str, tuple[torch.Tensor, torch.Tensor]]
Flat = dict[str, torch.Tensor]

LAMBDA_SPAN = 10.0  # λ_A, λ_B ∈ [0, 10]；A+B 的 P/Q 之和恒等于 U+N_1（10·p = 1）


# ---------------------------------------------------------------- 格式转换
def check_pairs(pairs: Pairs, *, expect_dtype: torch.dtype | None = None) -> None:
    if not pairs:
        raise ConfigurationError("Empty LoRA factor pairs")
    for stem, (p, q) in pairs.items():
        if p.dim() != 2 or q.dim() != 2:
            raise ConfigurationError(f"Factors not 2-D: {stem}")
        r_a, d_in = p.shape
        d_out, r_b = q.shape
        if r_a != r_b:
            raise ConfigurationError(f"Rank mismatch P={r_a} Q={r_b}: {stem}")
        if p.dtype != q.dtype:
            raise ConfigurationError(f"Factor dtype mismatch {p.dtype}/{q.dtype}: {stem}")
        if expect_dtype is not None and p.dtype != expect_dtype:
            raise ConfigurationError(f"Factor dtype {p.dtype} != {expect_dtype}: {stem}")
        if not torch.isfinite(p).all() or not torch.isfinite(q).all():
            raise FloatingPointError(f"Non-finite LoRA factor: {stem}")


def to_pairs(flat: Flat) -> Pairs:
    """peft 扁平 state → {stem: (P, Q)}（保留原 dtype）。"""
    pairs = module_pairs({k: v.detach().cpu() for k, v in dict(flat).items()})
    check_pairs(pairs)
    return pairs


def flatten(pairs: Pairs) -> Flat:
    """{stem: (P, Q)} → peft 扁平 state（contiguous，供 safetensors / apply_adapter / fedavg）。"""
    return {k: v.contiguous() for k, v in from_pairs(pairs).items()}


def to_fp32(pairs: Pairs) -> Pairs:
    """FP64 构造 → FP32 上传（客户端实际序列化精度）。"""
    check_pairs(pairs, expect_dtype=torch.float64)
    out = {m: (p.to(torch.float32), q.to(torch.float32)) for m, (p, q) in pairs.items()}
    check_pairs(out, expect_dtype=torch.float32)
    return out


def to_fp64(pairs: Pairs) -> Pairs:
    return {m: (p.double(), q.double()) for m, (p, q) in pairs.items()}


# ---------------------------------------------------------------- F2/F4/F5/F10 真实服务器聚合
def server_aggregate(states: list[Flat], weights: list[float] | None = None) -> Flat:
    """仓库真实等权因子 FedAvg：逐张量（即逐 P/Q 因子）FP64 加权求和后转回原 dtype。

    设计 §3：“必须调用仓库真实 src.fed.aggregate.fedavg，不得仅以手写公式代替真实聚合”。
    """
    if not states:
        raise ConfigurationError("No client uploads to aggregate")
    return dict(repo_fedavg(states, weights))


def server_aggregate_pairs(slots: list[Pairs], weights: list[float] | None = None) -> Pairs:
    """对 (P,Q) 因子槽位执行真实服务器聚合；输入必须是 FP32 上传。"""
    flats = []
    for slot in slots:
        check_pairs(slot, expect_dtype=torch.float32)
        flats.append(flatten(slot))
    return to_pairs(server_aggregate(flats, weights))


# ---------------------------------------------------------------- F6–F9 λ 交叉拆分
def D_factors(G: Pairs, U: Pairs, N1: Pairs, p: float) -> Pairs:
    """F6/F7：D_P = p·[(P_U − P_G) + (P_{N_1} − P_G)]，D_Q 同理；逐模块，float64。"""
    if set(G) != set(U) or set(G) != set(N1):
        raise ConfigurationError("G / U / N_1 module sets differ")
    out = {}
    for m in G:
        pg, qg = G[m][0].double(), G[m][1].double()
        out[m] = (p * ((U[m][0].double() - pg) + (N1[m][0].double() - pg)),
                  p * ((U[m][1].double() - qg) + (N1[m][1].double() - qg)))
    return out


def construct_ab(G: Pairs, D: Pairs, lambda_a: float, lambda_b: float) -> tuple[Pairs, Pairs]:
    """F8/F9（P/Q 方向有意交叉）：
        P_A = P_G + λ_A·D_P          P_B = P_G + (10 − λ_A)·D_P
        Q_A = Q_G + (10 − λ_B)·D_Q   Q_B = Q_G + λ_B·D_Q
    返回 float64 的 (A, B)。恒等式 P_A + P_B = P_U + P_{N_1}（Q 同）对任意 λ 成立。"""
    if not (0.0 <= lambda_a <= LAMBDA_SPAN and 0.0 <= lambda_b <= LAMBDA_SPAN):
        raise ConfigurationError(f"λ out of [0, {LAMBDA_SPAN}]: {lambda_a}, {lambda_b}")
    if set(G) != set(D):
        raise ConfigurationError("G / D module sets differ")
    A, B = {}, {}
    for m in G:
        pg, qg = G[m][0].double(), G[m][1].double()
        dp, dq = D[m][0].double(), D[m][1].double()
        A[m] = (pg + lambda_a * dp, qg + (LAMBDA_SPAN - lambda_b) * dq)
        B[m] = (pg + (LAMBDA_SPAN - lambda_a) * dp, qg + lambda_b * dq)
    check_pairs(A, expect_dtype=torch.float64)
    check_pairs(B, expect_dtype=torch.float64)
    return A, B


def ab_sum_identity_error(G: Pairs, U: Pairs, N1: Pairs, A: Pairs, B: Pairs, p: float) -> dict[str, float]:
    """代数硬条件：max_m ‖(P_A+P_B) − (P_U+P_{N_1})‖_∞ / scale，Q 同（设计 §4 阶段三）。"""
    worst_p = worst_q = 0.0
    for m in G:
        sp = (U[m][0].double() + N1[m][0].double())
        sq = (U[m][1].double() + N1[m][1].double())
        worst_p = max(worst_p, float((A[m][0].double() + B[m][0].double() - sp).abs().max())
                      / max(float(sp.abs().max()), 1e-30))
        worst_q = max(worst_q, float((A[m][1].double() + B[m][1].double() - sq).abs().max())
                      / max(float(sq.abs().max()), 1e-30))
    return {"P_sum_rel_err": worst_p, "Q_sum_rel_err": worst_q}


# ---------------------------------------------------------------- F11/F12 三种更新尺度
def _trace(x: torch.Tensor) -> float:
    return float(torch.trace(x))


def eff_delta_sq_dense(X: Pairs, G: Pairs, m: str) -> float:
    """显式张量路径：‖Q_X P_X − Q_G P_G‖_F²（直接物化差矩阵，无消去误差）。"""
    d = X[m][1].double() @ X[m][0].double() - G[m][1].double() @ G[m][0].double()
    return float((d * d).sum())


def eff_delta_sq_lowrank(X: Pairs, G: Pairs, m: str) -> float:
    """低秩迹路径：‖B₁A₁ − B₂A₂‖² = tr((B₁ᵀB₁)(A₁A₁ᵀ)) + tr((B₂ᵀB₂)(A₂A₂ᵀ)) − 2tr((B₁ᵀB₂)(A₂A₁ᵀ))。"""
    b1, a1 = X[m][1].double(), X[m][0].double()
    b2, a2 = G[m][1].double(), G[m][0].double()
    n1 = _trace((b1.T @ b1) @ (a1 @ a1.T))
    n2 = _trace((b2.T @ b2) @ (a2 @ a2.T))
    ip = _trace((b1.T @ b2) @ (a2 @ a1.T))
    return max(n1 + n2 - 2.0 * ip, 0.0)


def scale_eff_ba(X: Pairs, G: Pairs, s: float, *, path: str = "dense") -> float:
    """F11：N_X = sqrt(Σ_m ‖s·(Q_{X,m} P_{X,m} − Q_{G,m} P_{G,m})‖_F²)。

    path="dense"：逐模块物化差矩阵（主路径，精确）；path="lowrank"：迹恒等式（独立核验路径）。"""
    fn = eff_delta_sq_dense if path == "dense" else eff_delta_sq_lowrank
    if set(X) != set(G):
        raise ConfigurationError("X / G module sets differ")
    return s * math.sqrt(math.fsum(fn(X, G, m) for m in G))


def scale_P(X: Pairs, G: Pairs) -> float:
    """F12（P 因子）：F_{P,X} = sqrt(Σ_m ‖P_{X,m} − P_{G,m}‖_F²)。"""
    return math.sqrt(math.fsum(float(((X[m][0].double() - G[m][0].double()) ** 2).sum()) for m in G))


def scale_Q(X: Pairs, G: Pairs) -> float:
    """F12（Q 因子）：F_{Q,X} = sqrt(Σ_m ‖Q_{X,m} − Q_{G,m}‖_F²)。"""
    return math.sqrt(math.fsum(float(((X[m][1].double() - G[m][1].double()) ** 2).sum()) for m in G))


def three_scales(X: Pairs, G: Pairs, s: float) -> dict[str, float]:
    """三种尺度一次返回（有效 BA / P 因子 / Q 因子），各自独立计算。"""
    return {"N_eff_BA": scale_eff_ba(X, G, s, path="dense"),
            "F_P": scale_P(X, G), "F_Q": scale_Q(X, G)}


# ---------------------------------------------------------------- 有效权重与误差
def effective_W(pairs: Pairs, s: float, m: str) -> torch.Tensor:
    """W_m = s·Q_m P_m（float64 稠密，不含 W₀）。"""
    return s * (pairs[m][1].double() @ pairs[m][0].double())


def ba_rel_err(X: Pairs, Y: Pairs, s: float) -> float:
    """‖s(Q_X P_X − Q_Y P_Y)‖_F / ‖s·Q_Y P_Y‖_F（全模型，float64，显式路径）。"""
    num = math.fsum(eff_delta_sq_dense(X, Y, m) for m in Y)
    den = math.fsum(float(((Y[m][1].double() @ Y[m][0].double()) ** 2).sum()) for m in Y)
    return math.sqrt(num) / max(math.sqrt(den), 1e-30)


def pairs_max_abs_diff(X: Pairs, Y: Pairs) -> dict[str, float]:
    """逐因子最大绝对误差：{"P": max_m ‖P_X−P_Y‖_∞, "Q": ...}。"""
    return {"P": max(float((X[m][0].double() - Y[m][0].double()).abs().max()) for m in Y),
            "Q": max(float((X[m][1].double() - Y[m][1].double()).abs().max()) for m in Y)}


def compare_aggregates(label: str, X: Pairs, Y: Pairs, s: float, *, ba_tol: float, factor_tol: float) -> dict:
    """F10/§4 阶段五的硬性数值门槛：真实服务器聚合 11 与 U 逐模块比较 P/Q 因子与有效 BA 权重。"""
    pdiff = pairs_max_abs_diff(X, Y)
    per_module_ba = {m: math.sqrt(eff_delta_sq_dense(X, Y, m)) for m in Y}
    worst_ba = max(per_module_ba.values())
    ref_ba = max(math.sqrt(float(((Y[m][1].double() @ Y[m][0].double()) ** 2).sum())) for m in Y)
    out = {"label": label, "factor_max_abs": pdiff, "ba_rel_err": ba_rel_err(X, Y, s),
           "ba_module_max_abs": worst_ba, "ba_module_ref_norm": ref_ba,
           "factor_pass": pdiff["P"] <= factor_tol and pdiff["Q"] <= factor_tol,
           "ba_pass": ba_rel_err(X, Y, s) <= ba_tol}
    out["pass"] = bool(out["factor_pass"] and out["ba_pass"])
    return out


# ---------------------------------------------------------------- F13/F14 四分支交互项 C_W
def _delta_sets(b00: Pairs, b10: Pairs, b01: Pairs, b11: Pairs) -> dict[str, Pairs]:
    """δ_P = P_10 − P_00，η_P = P_01 − P_00，δ_Q = Q_10 − Q_00，η_Q = Q_01 − P_00 同理（float64）。"""
    if not (set(b00) == set(b10) == set(b01) == set(b11)):
        raise ConfigurationError("Four-branch module sets differ")
    dP = {m: b10[m][0].double() - b00[m][0].double() for m in b00}
    eP = {m: b01[m][0].double() - b00[m][0].double() for m in b00}
    dQ = {m: b10[m][1].double() - b00[m][1].double() for m in b00}
    eQ = {m: b01[m][1].double() - b00[m][1].double() for m in b00}
    return {"dP": dP, "eP": eP, "dQ": dQ, "eQ": eQ}


def interaction_explicit(b00: Pairs, b10: Pairs, b01: Pairs, b11: Pairs, s: float) -> dict[str, torch.Tensor]:
    """F13 显式路径：C_W = W_11 − W_10 − W_01 + W_00，逐模块物化 W_xy = s·Q_xy P_xy。"""
    out = {}
    for m in b00:
        out[m] = (effective_W(b11, s, m) - effective_W(b10, s, m)
                  - effective_W(b01, s, m) + effective_W(b00, s, m))
    return out


def interaction_lowrank(b00: Pairs, b10: Pairs, b01: Pairs, b11: Pairs, s: float) -> dict[str, torch.Tensor]:
    """F14 低秩交叉项路径：C_W = s·(δ_Q·η_P + η_Q·δ_P)，逐模块。"""
    d = _delta_sets(b00, b10, b01, b11)
    return {m: s * (d["dQ"][m] @ d["eP"][m] + d["eQ"][m] @ d["dP"][m]) for m in b00}


def interaction_trace_sq(b00: Pairs, b10: Pairs, b01: Pairs, b11: Pairs, s: float, m: str) -> float:
    """第三条路径（仅范数）：‖s(δ_Q η_P + η_Q δ_P)‖_F² 用 r×r 迹恒等式，不物化稠密矩阵。
    记 U = [δ_Q, η_Q]（d_out×2r），V = [η_P; δ_P]（2r×d_in），则 C = s·U V，‖C‖² = s²·tr((UᵀU)(VVᵀ))。"""
    d = _delta_sets(b00, b10, b01, b11)
    U = torch.cat((d["dQ"][m], d["eQ"][m]), dim=1)
    V = torch.cat((d["eP"][m], d["dP"][m]), dim=0)
    return float(s * s * torch.trace((U.T @ U) @ (V @ V.T)))


def compare_interaction_paths(b00: Pairs, b10: Pairs, b01: Pairs, b11: Pairs, s: float) -> dict:
    """两条独立路径逐模块核对（设计 §5）：逐模块流式计算（稠密矩阵即用即弃，避免同时物化全部模块）。
    返回每模块 ‖C_explicit‖_F、‖C_lowrank‖_F、两路径差的相对值，以及迹路径范数的相对差；
    另给出全模型相对差与 abs_diff_over_max_W（正式门槛）。"""
    if not (set(b00) == set(b10) == set(b01) == set(b11)):
        raise ConfigurationError("Four-branch module sets differ")
    d = _delta_sets(b00, b10, b01, b11)
    per_module = {}
    worst_path = worst_trace = 0.0
    total_sq = diff_sq = w_max_sq = 0.0
    for m in b00:
        C_e = (effective_W(b11, s, m) - effective_W(b10, s, m)
               - effective_W(b01, s, m) + effective_W(b00, s, m))
        C_l = s * (d["dQ"][m] @ d["eP"][m] + d["eQ"][m] @ d["dP"][m])
        ne = float((C_e * C_e).sum())
        nl = float((C_l * C_l).sum())
        diff = float(((C_e - C_l) ** 2).sum())
        ref = max(ne, nl, 1e-300)
        rel_path = math.sqrt(diff / ref)
        tr = interaction_trace_sq(b00, b10, b01, b11, s, m)
        rel_trace = abs(math.sqrt(tr) - math.sqrt(ne)) / max(math.sqrt(ne), 1e-300)
        worst_path = max(worst_path, rel_path)
        worst_trace = max(worst_trace, rel_trace)
        total_sq += ne
        diff_sq += diff
        per_module[m] = {"explicit_norm": math.sqrt(ne), "lowrank_norm": math.sqrt(nl),
                         "trace_norm": math.sqrt(tr), "path_rel_diff": rel_path,
                         "trace_rel_diff": rel_trace}
        del C_e, C_l
    # 主更新尺度 max_xy‖W_xy‖_F：逐模块累加四个分支的有效权重范数
    w_sq = {b: 0.0 for b in ("00", "10", "01", "11")}
    branches = {"00": b00, "10": b10, "01": b01, "11": b11}
    for m in b00:
        for b, pairs in branches.items():
            W = effective_W(pairs, s, m)
            w_sq[b] += float((W * W).sum())
            del W
    w_max = math.sqrt(max(w_sq.values()))
    diff_all = math.sqrt(diff_sq)
    c_norm = math.sqrt(total_sq)
    # 两个不同的误差口径（必须分别记录，不得互相代替）：
    #   e_W = ‖C_e − C_l‖_F / max_xy‖W_xy‖_F   相对主更新尺度（配置 interaction_abs_diff_over_W_max 对应此项）
    #   e_C = ‖C_e − C_l‖_F / ‖C_l‖_F           相对交互项自身（FP32 聚合可加性残差被微小 C_W 放大，约 1e-4 量级）
    # 分母为零（四分支完全可加且 C_W=0，或两路径均精确为零）时按数学定义处理：e_W 以 w_max 为准；
    # e_C 的分母为零 → 若分子也为零则两路径一致，记 0.0 并标注；若分子非零则记 None 并标注，不得产生 NaN/Inf。
    lowrank_sq = math.fsum(per_module[m]["lowrank_norm"] ** 2 for m in per_module)
    c_low = math.sqrt(lowrank_sq)
    if c_low > 0.0:
        e_c, c_note = diff_all / c_low, None
    elif diff_all == 0.0:
        e_c, c_note = 0.0, "C_lowrank norm is exactly zero and path diff is zero: both paths agree at zero"
    else:
        e_c, c_note = None, "C_lowrank norm is exactly zero but paths differ; e_C undefined (recorded as null, NOT 0)"
    # 绝对差相对主更新尺度：FP32 聚合的可加性残差 (~ulp) 被 C_W 的微小范数放大，
    # 因此正式门槛用 e_W（FP32 舍入量级 ~1e-7）；e_C 仅作记录，不得宣称 e_C 满足同一阈值。
    return {"per_module": per_module, "total_explicit_norm": c_norm, "total_lowrank_norm": c_low,
            "max_path_rel_diff": worst_path, "max_trace_rel_diff": worst_trace,
            "global_path_rel_diff": diff_all / max(c_norm, 1e-300),
            "abs_diff_over_max_W": diff_all / max(w_max, 1e-300) if w_max > 0.0 else None,
            "rel_diff_over_C_lowrank": e_c, "rel_diff_over_C_note": c_note,
            "abs_diff": diff_all, "max_W_norm": w_max,
            "error_definitions": {
                "e_W (abs_diff_over_max_W)": "||C_explicit - C_lowrank||_F / max_xy ||W_xy||_F  (threshold interaction_abs_diff_over_W_max applies HERE)",
                "e_C (rel_diff_over_C_lowrank)": "||C_explicit - C_lowrank||_F / ||C_lowrank||_F  (recorded only; NOT gated by the same threshold)"}}


def linear_additivity_check(b00: Pairs, b10: Pairs, b01: Pairs, b11: Pairs) -> dict[str, float]:
    """P_11 = P_10 + P_01 − P_00（Q 同）在因子层面成立（因子 FedAvg 线性 + A+B = U+N_1）。"""
    worst_p = worst_q = 0.0
    for m in b00:
        sp = b10[m][0].double() + b01[m][0].double() - b00[m][0].double()
        sq = b10[m][1].double() + b01[m][1].double() - b00[m][1].double()
        worst_p = max(worst_p, float((b11[m][0].double() - sp).abs().max()))
        worst_q = max(worst_q, float((b11[m][1].double() - sq).abs().max()))
    return {"P_max_abs": worst_p, "Q_max_abs": worst_q}


# ---------------------------------------------------------------- 分支槽位定义（§4 阶段四）
BRANCH_SLOTS = {
    "00": ["N0", "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9"],
    "U":  ["U", "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9"],
    "10": ["A", "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9"],
    "01": ["N0", "B", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9"],
    "11": ["A", "B", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9"],
}
NORMAL_CLIENT_INDEX = {f"N{j}": j for j in range(10)}      # D_{j,r} 来自 Task1 分区 client_j
MALICIOUS_ROLES = ("U", "A", "B")


def slots_for(branch: str) -> list[str]:
    if branch not in BRANCH_SLOTS:
        raise ConfigurationError(f"Unknown branch {branch}")
    return list(BRANCH_SLOTS[branch])


def trained_normal_roles(branch: str) -> list[str]:
    """本分支需要真实训练并进入聚合的正常客户端。"""
    return [r for r in slots_for(branch) if r.startswith("N")]


def training_plan(branch: str) -> list[dict]:
    """本分支每轮需要真实训练的任务（唯一真源）：
    每项 {role: "U"|"N", client_index, name(文件名), aggregated(是否进入聚合槽位)}。
    构造 D 需要 U 与 N1 的真实上传；N1 仅在 10 分支进入聚合。"""
    slots = slots_for(branch)
    plan = []
    for role in trained_normal_roles(branch):
        j = NORMAL_CLIENT_INDEX[role]
        plan.append({"role": "N", "client_index": j, "name": role, "aggregated": True})
    if branch == "U":
        plan.append({"role": "U", "client_index": 0, "name": "U", "aggregated": True})
    elif branch in ("10", "01", "11"):
        plan.append({"role": "U", "client_index": 0, "name": "U_for_D", "aggregated": False})
        if "N1" not in slots:
            plan.append({"role": "N", "client_index": 1, "name": "N1_for_D", "aggregated": False})
    plan.sort(key=lambda t: (0 if t["role"] == "U" else 1, t["client_index"]))
    names = [t["name"] for t in plan]
    if len(set(names)) != len(names):
        raise ConfigurationError(f"Duplicate training names in branch {branch}")
    return plan


def construction_inputs(branch: str) -> dict[str, str]:
    """构造 A/B 的三个输入上传文件名：G（本轮起点）、U、N1。"""
    if branch not in ("10", "01", "11"):
        raise ConfigurationError(f"Branch {branch} has no A/B construction")
    names = {t["name"] for t in training_plan(branch)}
    return {"U": "U_for_D", "N1": "N1" if "N1" in names else "N1_for_D"}


def client_weights(n: int = 10) -> list[float]:
    return [1.0 / n] * n


def sum_squared(values: Iterable[float]) -> float:
    return math.fsum(v * v for v in values)
