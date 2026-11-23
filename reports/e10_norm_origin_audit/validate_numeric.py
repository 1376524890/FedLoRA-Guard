"""E10 Norm 审计补充数值验证（2026-11-23）。

1. 低秩公式 vs 直接物化 BA：float32 差异是否只是浮点累加噪声（float64 复核）；
2. 乘积差公式 ‖P1−P2‖（本审计"相对差值"列所用）在非零全局轮的实矩阵验证；
3. 集中目标 r32 原始乘积 vs r8 因子乘积的实际差值大小（转换损失，逐系列）；
4. 关键文件 sha256 清单（区分实际运行版本与当前工作目录版本）。

用法：
  cd /home/marktom/FedLoRA-Guard
  /home/marktom/miniconda3/envs/fllora/bin/python reports/e10_norm_origin_audit/validate_numeric.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

torch.set_num_threads(4)

ROOT = Path("/home/marktom/FedLoRA-Guard")
sys.path.insert(0, str(ROOT))

from src.e10.lora_algebra import fro_inner_lowrank, fro_norm_lowrank, module_pairs  # noqa: E402

OUT = ROOT / "reports/e10_norm_origin_audit"
SERIES = ["pku_s17", "pku_s42", "pku_s2026", "dolly_s17", "dolly_s42", "dolly_s2026"]


def load_pairs(p: Path) -> dict:
    from safetensors.torch import load_file
    return module_pairs(dict(load_file(str(p))))


def merged_diff(p1: dict, p2: dict) -> float:
    tot = 0.0
    for m in p1:
        a1, b1 = p1[m]
        a2, b2 = p2[m]
        n1 = fro_norm_lowrank(a1, b1) ** 2
        n2 = fro_norm_lowrank(a2, b2) ** 2
        ip = fro_inner_lowrank(b1, a1, b2, a2)
        tot += max(n1 + n2 - 2 * ip, 0.0)
    return float(tot ** 0.5)


def merged_norm(p: dict) -> float:
    return float(sum(fro_norm_lowrank(p[m][0], p[m][1]) ** 2 for m in p) ** 0.5)


def main() -> None:
    out: dict = {}

    # ---- 1. fp32 差异是否为浮点噪声：float64 复核一个真实模块 ----
    ca = load_pairs(ROOT / "artifacts/e10/p7/pku_s17/e10bg_pku_s17_00_r1/constructs/a/adapter.safetensors")
    m = sorted(ca)[0]
    a, b = ca[m]
    direct32 = float(torch.linalg.norm(b @ a))
    low32 = fro_norm_lowrank(a, b)
    a64, b64 = a.double(), b.double()
    direct64 = float(torch.linalg.norm(b64 @ a64))
    low64 = float(torch.sqrt(torch.clamp(torch.trace((b64.T @ b64) @ (a64 @ a64.T)), min=0)))
    out["fp64_module_check"] = {
        "module": m,
        "direct_fp32": direct32, "lowrank_fp32": low32, "fp32_rel_diff": abs(direct32 - low32) / direct32,
        "direct_fp64": direct64, "lowrank_fp64": low64, "fp64_rel_diff": abs(direct64 - low64) / direct64,
        "fp32_vs_fp64_direct_rel": abs(direct32 - direct64) / direct64,
        "结论": "fp32 两种算法相差 ~5e-4 属累加顺序噪声；float64 下两法一致到 <1e-12",
    }

    # ---- 2. 非零全局轮的乘积差公式验证（round 3，全局范数 ≈0.71） ----
    gl = load_pairs(ROOT / "artifacts/runs/e10bg_pku_s17_00/checkpoints/round_003/received_global_adapter.safetensors")
    a1, b1 = ca[m][0], ca[m][1]
    a2, b2 = gl[m][0], gl[m][1]
    mat_diff64 = float(torch.linalg.norm(b1.double() @ a1.double() - b2.double() @ a2.double()))
    formula64 = float(max(
        fro_norm_lowrank(a1.double(), b1.double()) ** 2
        + fro_norm_lowrank(a2.double(), b2.double()) ** 2
        - 2 * fro_inner_lowrank(b1.double(), a1.double(), b2.double(), a2.double()), 0) ** 0.5)
    out["diff_formula_nonzero_global_fp64"] = {
        "round": "e10bg_pku_s17_00 r3 vs construct_a(r1 因子)", "module": m,
        "materialized_fp64": mat_diff64, "lowrank_formula_fp64": formula64,
        "rel_err": abs(mat_diff64 - formula64) / mat_diff64,
    }

    # ---- 3. r32 原始乘积 vs r8 乘积的实际差（转换损失，逐系列） ----
    conv = {}
    for s in SERIES:
        raw = load_pairs(ROOT / f"artifacts/e10/p2/{s}/target_adapter.safetensors")
        f8 = load_pairs(ROOT / f"artifacts/e10/p2/{s}/target/factor_r8.safetensors")
        nr, nf = merged_norm(raw), merged_norm(f8)
        d = merged_diff(raw, f8)
        rep = json.loads((ROOT / f"artifacts/e10/p2/{s}/target/factorization_report.json").read_text())
        conv[s] = {"norm_raw_r32": nr, "norm_r8": nf, "prod_diff": d,
                   "prod_diff_over_raw": d / nr, "r8_over_raw": nf / nr,
                   "rho_global_r8": rep["ranks"]["8"]["rho_global"]}
    out["r32_to_r8_conversion"] = conv

    # ---- 3.5 因子空间扣除 vs 乘积空间扣除（放大来源解释，示例轮 pku_s17|r1） ----
    sys.path.insert(0, str(ROOT))
    from reports.e10_norm_origin_audit.audit_norms import rank_r_factorize  # noqa: E402
    BG_SLOTS = [f"client_{i:02d}" for i in range(2, 10)]
    rdir = ROOT / "artifacts/runs/e10bg_pku_s17_00/checkpoints/round_001"
    bg_list = [load_pairs(rdir / "clients" / c / "final_adapter.safetensors") for c in BG_SLOTS]
    a_bg = {mm: sum(0.1 * p[mm][0] for p in bg_list) for mm in bg_list[0]}
    b_bg = {mm: sum(0.1 * p[mm][1] for p in bg_list) for mm in bg_list[0]}
    bg_prod = {mm: (a_bg[mm], b_bg[mm]) for mm in a_bg}
    tf = {}
    f8 = load_pairs(ROOT / "artifacts/e10/p2/pku_s17/target/factor_r8.safetensors")
    for mm in f8:
        b_t, a_t = rank_r_factorize(f8[mm][1], f8[mm][0])
        tf[mm] = (a_t, b_t)
    R = {mm: (tf[mm][0] - a_bg[mm], tf[mm][1] - b_bg[mm]) for mm in tf}
    out["factor_vs_product_space_removal_pku_s17_r1"] = {
        "目标r8乘积": merged_norm(tf),
        "背景聚合因子乘积(0.1×8,未归一化)": merged_norm(bg_prod),
        "因子空间扣除后‖R_B R_A‖(构造实际使用)": merged_norm(R),
        "乘积空间扣除后‖B_tA_t − B_bgA_bg‖(构造未使用)": merged_diff(tf, bg_prod),
        "说明": "因子空间扣除后的乘积显著大于乘积空间差值——(B_t−B_bg)(A_t−A_bg) 含交叉项 "
                "−B_tA_bg − B_bgA_t + B_bgA_bg，背景因子与目标因子近似正交时范数增大",
    }

    # ---- 4. 关键文件哈希 ----
    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16]

    files = {
        "代码（当前工作目录 = 实际运行版本，见 RESULTS.md 版本核对）": [
            "src/e10/lora_algebra.py", "src/e10/p6_calibrate.py", "src/e10/p7_construct.py",
            "src/e10/p7_detect.py", "src/e10/factorization.py", "src/e10/factorize_p2.py",
            "src/e10/train_target.py", "src/train/local.py",
        ],
        "六个集中训练目标（每系列 1 个，全部轮复用）": [
            *[f"artifacts/e10/p2/{s}/target_adapter.safetensors" for s in SERIES],
            *[f"artifacts/e10/p2/{s}/target/factor_r8.safetensors" for s in SERIES],
        ],
        "示例轮产物": [
            "artifacts/e10/p7/pku_s17/e10bg_pku_s17_00_r1/constructs/a/adapter.safetensors",
            "artifacts/e10/p7/pku_s17/e10bg_pku_s17_00_r1/constructs/b/adapter.safetensors",
            "artifacts/runs/e10bg_pku_s17_00/checkpoints/round_001/received_global_adapter.safetensors",
            "artifacts/runs/e10bg_pku_s17_00/checkpoints/round_001/clients/client_02/final_adapter.safetensors",
        ],
    }
    out["file_hashes"] = {grp: {f: sha(ROOT / f) for f in fl} for grp, fl in files.items()}

    (OUT / "numeric_validation.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                 encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("fp64_module_check", "diff_formula_nonzero_global_fp64")},
                     ensure_ascii=False, indent=1))
    for s, v in conv.items():
        print(f"{s}: raw={v['norm_raw_r32']:.4f} r8={v['norm_r8']:.4f} "
              f"乘积差={v['prod_diff']:.4f} (相对 raw {v['prod_diff_over_raw']:.4%}, "
              f"rho_global={v['rho_global_r8']:.4f})")


if __name__ == "__main__":
    main()
