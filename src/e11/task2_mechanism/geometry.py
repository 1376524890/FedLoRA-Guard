"""任务二：U 与正常客户端更新方向的逐轮几何分析（只读，数学计算）。

方法（全部 float64，因子级计算，不稠密化有效权重）：
  内积迹恒等式 ⟨s·Q_iP_i, s·Q_jP_j⟩ = s²·tr((Q_iᵀQ_j)(P_jP_iᵀ))；
  14 个向量栈 [G, U, N1..N9, AGG, T1, T0]，逐模块批量得到 14×14 原始 Gram，
  再按差分导出更新向量 (v − v_G) 的 Gram、与 Task1 参考方向 (W_T1 − W_T0) 的投影。
  聚合复算口径：AGG 直接取磁盘上的 global_r（任务一已核验 = 仓库真实 FedAvg 逐位复算）。
输出：REPORT_ROOT/U_UPDATE_GEOMETRY.json（含逐轮 14×14 Gram 以便追溯复算）。
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.e11.task2_mechanism.common import (REPORT_ROOT, SERIES, T1_DIR, global_path,  # noqa: E402
                                            stamp, traj, write_json)
from src.e11.task2_protocol import algebra as alg  # noqa: E402
from src.e11.task2_protocol.config import load_all  # noqa: E402

IDX_G, IDX_U, IDX_N0, IDX_AGG, IDX_T1, IDX_T0 = 0, 1, 2, 11, 12, 13
NAMES = ["G", "U"] + [f"N{j}" for j in range(1, 10)] + ["AGG", "T1", "T0"]
LAY_RE = re.compile(r"layers\.(\d+)\.")


def load_flat(path: Path) -> dict:
    from safetensors.torch import load_file
    return {k: v for k, v in load_file(str(path)).items()}


def module_dot_block(vecs: list[dict], m: str, s: float) -> torch.Tensor:
    """单模块的 14×14 有效权重内积块（迹恒等式）。"""
    P = torch.stack([vecs[i][m + ".lora_A.weight"].double() for i in range(len(vecs))])   # (n, r, d_in)
    Q = torch.stack([vecs[i][m + ".lora_B.weight"].double() for i in range(len(vecs))])   # (n, d_out, r)
    A = torch.einsum("iar,jas->ijrs", Q, Q)    # A[i,j] = Q_i^T Q_j (r×r)
    B = torch.einsum("irb,jsb->ijrs", P, P)    # B[i,j] = P_i P_j^T (r×r)
    return s * s * torch.einsum("ijab,jiba->ij", A, B)   # dot[i,j] = tr(A[i,j] B[j,i])


def factor_dot_block(vecs: list[dict], m: str, which: str) -> torch.Tensor:
    suf = f".lora_{which}.weight"
    F = torch.stack([vecs[i][m + suf].double() for i in range(len(vecs))])
    return torch.einsum("iab,jab->ij", F, F)


def group_of(m: str) -> tuple[str, int]:
    lay = LAY_RE.search(m)
    return (m.split(".")[-1], int(lay.group(1)) if lay else -1)


def delta_gram(D: torch.Tensor, i: int, j: int) -> float:
    return float(D[i][j] - D[i][IDX_G] - D[IDX_G][j] + D[IDX_G][IDX_G])


def ref_dot(D: torch.Tensor, i: int) -> float:
    return float(D[i][IDX_T1] - D[i][IDX_T0] - D[IDX_G][IDX_T1] + D[IDX_G][IDX_T0])


def ref_norm_sq(D: torch.Tensor) -> float:
    return float(D[IDX_T1][IDX_T1] - 2 * D[IDX_T1][IDX_T0] + D[IDX_T0][IDX_T0])


def summarize_gram(D: torch.Tensor) -> dict:
    """从 14×14 原始 Gram 导出该轮的全部几何量。"""
    clients = list(range(IDX_N0, IDX_AGG))
    n = {i: max(delta_gram(D, i, i), 0.0) ** 0.5 for i in clients + [IDX_U, IDX_AGG]}
    cos = lambda i, j: (delta_gram(D, i, j) / (n[i] * n[j])) if n[i] > 0 and n[j] > 0 else None   # noqa: E731
    # 平均正常方向（向量均值，非余弦均值）
    k = len(clients)
    mn_ii = sum(delta_gram(D, a, b) for a in clients for b in clients) / (k * k)
    n_mean = max(mn_ii, 0.0) ** 0.5
    dot_U_mean = sum(delta_gram(D, IDX_U, c) for c in clients) / k
    dot_agg_mean = sum(delta_gram(D, IDX_AGG, c) for c in clients) / k
    rn = max(ref_norm_sq(D), 0.0) ** 0.5
    out = {
        "norms": {"U": n[IDX_U], "N_median": statistics.median([n[c] for c in clients]),
                  "N_minmax": [min(n[c] for c in clients), max(n[c] for c in clients)],
                  "AGG": n[IDX_AGG], "mean_normal": n_mean, "ref_T1_minus_T0": rn},
        "norm_ratio_U_over_N_median": n[IDX_U] / max(statistics.median([n[c] for c in clients]), 1e-30),
        "cosine_eff_space": {"U_vs_each_N": [cos(IDX_U, c) for c in clients],
                             "U_vs_mean_normal": dot_U_mean / (n[IDX_U] * n_mean) if n[IDX_U] * n_mean > 0 else None,
                             "AGG_vs_mean_normal": dot_agg_mean / (n[IDX_AGG] * n_mean) if n[IDX_AGG] * n_mean > 0 else None},
    }
    out["projection_onto_ref"] = {
        "U_cos": (ref_dot(D, IDX_U) / (n[IDX_U] * rn)) if n[IDX_U] * rn > 0 else None,
        "U_magnitude": (ref_dot(D, IDX_U) / rn) if rn > 0 else None,
        "AGG_cos": (ref_dot(D, IDX_AGG) / (n[IDX_AGG] * rn)) if n[IDX_AGG] * rn > 0 else None,
        "AGG_magnitude": (ref_dot(D, IDX_AGG) / rn) if rn > 0 else None,
        "mean_normal_cos": ((sum(ref_dot(D, c) for c in clients) / k) / (n_mean * rn)) if n_mean * rn > 0 else None,
        "ref_norm": rn,
    }
    # 因子聚合交叉项：||d_AGG − mean_c d_c|| / ||mean_c d_c||（有效权重空间，10 客户端含 U）
    slots = [IDX_U] + clients
    mean_sq = sum(delta_gram(D, a, b) for a in slots for b in slots) / 100.0
    agg_minus_mean = delta_gram(D, IDX_AGG, IDX_AGG) - 2 * sum(delta_gram(D, IDX_AGG, c) for c in slots) / 10.0 + mean_sq
    out["factor_aggregation_cross_term"] = {
        "cross_norm": max(agg_minus_mean, 0.0) ** 0.5,
        "mean_norm": max(mean_sq, 0.0) ** 0.5,
        "cross_over_mean": (max(agg_minus_mean, 0.0) / mean_sq) ** 0.5 if mean_sq > 0 else None,
        "agg_norm": n[IDX_AGG],
        "note": "FedAvg averages factors; effective-space aggregate deviates from mean effective update by this cross term",
    }
    return out


def main() -> None:
    cfg = load_all()
    out: dict = {"series": {}, "method": __doc__, **stamp(scope="task2_geometry")}
    for s in SERIES:
        s_scale = float(cfg["series_cfg"][s]["lora"]["alpha"]) / float(cfg["series_cfg"][s]["lora"]["rank"]) \
            if "series_cfg" in cfg and s in cfg.get("series_cfg", {}) else None
        if s_scale is None:   # 从 Task1Inputs 取（与协议一致）
            from src.e11.task2_protocol.inputs import Task1Inputs
            t1 = Task1Inputs(cfg, s).t1_cfg
            s_scale = float(t1["lora"]["alpha"]) / float(t1["lora"]["rank"])
        t0 = load_flat(T1_DIR / s / "B_huayu_first" / "init_adapter.safetensors")
        t1 = load_flat(T1_DIR / s / "B_huayu_first" / "model_round_001.safetensors")
        rs: dict = {"scaling": s_scale, "rounds": {}}
        for r in range(1, 31):
            G = load_flat(global_path(s, "U", r - 1))
            U = load_flat(traj(s, "U") / f"round_{r:03d}" / "uploads" / "U.safetensors")
            Ns = [load_flat(traj(s, "U") / f"round_{r:03d}" / "uploads" / f"N{j}.safetensors") for j in range(1, 10)]
            AGG = load_flat(traj(s, "U") / f"global_{r:03d}.safetensors")
            vecs = [G, U] + Ns + [AGG, t1, t0]
            modules = sorted({k[:-len(".lora_A.weight")] for k in G if k.endswith(".lora_A.weight")})
            D_all = torch.zeros(14, 14, dtype=torch.float64)
            DP_all = torch.zeros(14, 14, dtype=torch.float64)
            DQ_all = torch.zeros(14, 14, dtype=torch.float64)
            by_type: dict[str, torch.Tensor] = {}
            by_layer: dict[int, torch.Tensor] = {}
            for m in modules:
                d = module_dot_block(vecs, m, s_scale)
                D_all += d
                DP_all += factor_dot_block(vecs, m, "A")
                DQ_all += factor_dot_block(vecs, m, "B")
                t, L = group_of(m)
                by_type.setdefault(t, torch.zeros(14, 14, dtype=torch.float64)).add_(d)
                by_layer.setdefault(L, torch.zeros(14, 14, dtype=torch.float64)).add_(d)
            entry = summarize_gram(D_all)
            entry["factor_gram_summary"] = {w: summarize_factor(g) for w, g in (("dP", DP_all), ("dQ", DQ_all))}
            entry["by_type"] = {t: compact(summarize_gram(g)) for t, g in sorted(by_type.items())}
            entry["by_layer"] = {str(L): compact(summarize_gram(g)) for L, g in sorted(by_layer.items())}
            entry["gram_eff_14x14"] = [[round(float(x), 6) for x in row] for row in D_all]
            rs["rounds"][str(r)] = entry
            print(f"[geom] {s} r{r}: cosU-meanN={entry['cosine_eff_space']['U_vs_mean_normal']:.4f} "
                  f"cosU-ref={entry['projection_onto_ref']['U_cos']} "
                  f"cross/mean={entry['factor_aggregation_cross_term']['cross_over_mean']:.4f}", flush=True)
        out["series"][s] = rs
    write_json(REPORT_ROOT / "U_UPDATE_GEOMETRY.json", out)
    print(f"[geom] -> {REPORT_ROOT / 'U_UPDATE_GEOMETRY.json'}", flush=True)


def summarize_factor(F: torch.Tensor) -> dict:
    clients = list(range(IDX_N0, IDX_AGG))
    n_U = max(delta_gram(F, IDX_U, IDX_U), 0.0) ** 0.5
    n_med = statistics.median([max(delta_gram(F, c, c), 0.0) ** 0.5 for c in clients])
    k = len(clients)
    n_mean = max(sum(delta_gram(F, a, b) for a in clients for b in clients) / (k * k), 0.0) ** 0.5
    dot = sum(delta_gram(F, IDX_U, c) for c in clients) / k
    rn = max(ref_norm_sq(F), 0.0) ** 0.5
    return {"norm_U": n_U, "norm_N_median": n_med, "norm_ratio": n_U / max(n_med, 1e-30),
            "cos_U_vs_mean_normal": dot / (n_U * n_mean) if n_U * n_mean > 0 else None,
            "cos_U_vs_ref": ref_dot(F, IDX_U) / (n_U * rn) if n_U * rn > 0 else None}


def compact(g: dict) -> dict:
    return {"cos_U_vs_mean_normal": g["cosine_eff_space"]["U_vs_mean_normal"],
            "norm_ratio_U_over_N_median": g["norm_ratio_U_over_N_median"],
            "cross_over_mean": g["factor_aggregation_cross_term"]["cross_over_mean"],
            "cos_U_vs_ref": g["projection_onto_ref"]["U_cos"]}


if __name__ == "__main__":
    main()
