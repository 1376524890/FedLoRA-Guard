"""E10 Norm 溯源审计（2026-11-23）：只读现有文件，不启动训练。

问题：P7 混合轮中构造客户端（甲/乙）与正常客户端的 LoRA 适配器大小（P6/P7 Norm，
即逐模块 ‖BA‖_F 平方和开根，不减全局、不乘 α/r）相差约 7.2 倍——差距出现在哪一步？

对 6 个训练系列 × 13 条背景轨迹 × 6 轮 = 90 个双成员混合轮，逐轮测量：
  1. 本轮下发全局适配器 received_global_adapter.safetensors
  2. 8 名正常客户端上传（client_02..09 final_adapter.safetensors）
  3. 集中训练原始目标（p2/<series>/target_adapter.safetensors，r=32 训练产物）
  4. 正式秩目标（p2/<series>/target/factor_r8.safetensors，r=8 截断因子）
  5. 扣除背景后 R_A/R_B（代码实际权重：8 × 0.1，不归一化）的乘积 R_B R_A
  6. 甲/乙最终上传（p7/<series>/<bg>_r<rd>/constructs/{a,b}/adapter.safetensors）

等式核验（两端来自不同文件，不用构造函数生成左端）：
  ‖B_a A_a‖ =? 0.99 · ‖R_B R_A‖，左端读保存的最终上传，右端由目标因子 + 背景文件独立计算。
  另做逐元素核验 A_a ≈ ε·R_A、B_a ≈ 9.9·R_B（ε=0.1，(1−p_b·ε)/p_a=9.9）。

用法：
  cd /home/marktom/FedLoRA-Guard
  /home/marktom/miniconda3/envs/fllora/bin/python reports/e10_norm_origin_audit/audit_norms.py

产出（同目录）：
  round_norms.tsv   逐轮全部测量值
  norm_stats.json   汇总统计、等式误差、复算与归档对账、数值验证
"""

from __future__ import annotations

import json
import statistics as st
import sys
import time
from functools import lru_cache
from pathlib import Path

import torch

torch.set_num_threads(4)

ROOT = Path("/home/marktom/FedLoRA-Guard")
sys.path.insert(0, str(ROOT))

from src.e10.lora_algebra import (fro_inner_lowrank, fro_norm_lowrank,  # noqa: E402
                                  module_pairs)

OUT = ROOT / "reports/e10_norm_origin_audit"
P2 = ROOT / "artifacts/e10/p2"
P7 = ROOT / "artifacts/e10/p7"
RUNS = ROOT / "artifacts/runs"

EPS, PA, PB = 0.1, 0.1, 0.1          # p7_construct.py 运行时常数
BG_SLOTS = [f"client_{i:02d}" for i in range(2, 10)]
PAIRING = {
    "pku_s17": [f"e10bg_pku_s17_{i}" for i in ("00", "01", "10", "11")],
    "pku_s42": [f"e10bg_pku_s42_{i}" for i in ("00", "01", "10", "11")],
    "pku_s2026": [f"e10bg_pku_s2026_{i}" for i in ("00", "01", "10", "11")],
    "dolly_s17": ["e10bg_clean_s2026"],
    "dolly_s42": ["e10bg_clean_s2026"],
    "dolly_s2026": ["e10bg_clean_s2026"],
}
ROUNDS = [1, 2, 3, 4, 5, 6]
SERIES = list(PAIRING)


@lru_cache(maxsize=32)
def load_pairs(path_str: str) -> dict:
    from safetensors.torch import load_file
    return module_pairs(dict(load_file(path_str)))


def merged_norm(pairs: dict) -> float:
    """项目 Norm 口径：sqrt(Σ_mod ‖BA‖_F²)。"""
    return float(sum(fro_norm_lowrank(pairs[m][0], pairs[m][1]) ** 2 for m in pairs) ** 0.5)


def merged_diff(pairs1: dict, pairs2: dict) -> float:
    """‖P1 − P2‖（逐模块乘积差的全模型合并）：‖P1‖²+‖P2‖²−2⟨P1,P2⟩ 的平方和开根。"""
    tot = 0.0
    for m in pairs1:
        a1, b1 = pairs1[m]
        a2, b2 = pairs2[m]
        n1 = fro_norm_lowrank(a1, b1) ** 2
        n2 = fro_norm_lowrank(a2, b2) ** 2
        ip = fro_inner_lowrank(b1, a1, b2, a2)
        tot += max(n1 + n2 - 2 * ip, 0.0)
    return float(tot ** 0.5)


def rank_r_factorize(b_goal, a_goal):
    """与 p7_construct.rank_r_factorize 相同的确定性再分解（仅用于重现运行时 target_f）。"""
    q, rr = torch.linalg.qr(b_goal.float())
    small = rr @ a_goal.float()
    u, sig, vh = torch.linalg.svd(small, full_matrices=False)
    sq = sig.clamp_min(0).sqrt()
    return q @ (u * sq), sq.unsqueeze(1) * vh


def elementwise_check(saved: dict, expected: dict) -> float:
    """保存的构造因子 vs 公式（如 A_a=ε·R_A、B_a=9.9·R_B）的最大相对误差。"""
    worst = 0.0
    for m in saved:
        (ae, be), (ag, bg_) = saved[m], expected[m]
        for s_, e_ in ((ae, ag), (be, bg_)):
            den = float(e_.norm()) + 1e-30
            worst = max(worst, float((s_ - e_).norm()) / den)
    return worst


def main() -> None:
    t0 = time.time()
    stats: dict = {"generated": time.strftime("%Y-%m-%d %H:%M:%S")}

    # ---------- 0. 数值验证：直接物化 BA vs 低秩公式（真实模块） ----------
    val = {}
    probe_paths = {
        "construct_a": P7 / "pku_s17/e10bg_pku_s17_00_r1/constructs/a/adapter.safetensors",
        "normal_client_02": RUNS / "e10bg_pku_s17_00/checkpoints/round_001/clients/client_02/final_adapter.safetensors",
        "target_raw_r32": P2 / "pku_s17/target_adapter.safetensors",
    }
    for tag, p in probe_paths.items():
        pairs = load_pairs(str(p))
        m = sorted(pairs)[0]
        a, b = pairs[m]
        direct = float(torch.linalg.norm(b @ a))
        lowrank = fro_norm_lowrank(a, b)
        val[tag] = {"module": m, "direct_norm": direct, "lowrank_norm": lowrank,
                    "abs_err": abs(direct - lowrank),
                    "rel_err": abs(direct - lowrank) / max(direct, 1e-30)}
    # 差值公式验证（一对真实模块）
    ca = load_pairs(str(probe_paths["construct_a"]))
    gl = load_pairs(str(RUNS / "e10bg_pku_s17_00/checkpoints/round_001/received_global_adapter.safetensors"))
    m = sorted(ca)[0]
    mat_diff = float(torch.linalg.norm(ca[m][1] @ ca[m][0] - gl[m][1] @ gl[m][0]))
    a1, b1, a2, b2 = ca[m][0], ca[m][1], gl[m][0], gl[m][1]
    formula = float(max(fro_norm_lowrank(a1, b1) ** 2 + fro_norm_lowrank(a2, b2) ** 2
                        - 2 * fro_inner_lowrank(b1, a1, b2, a2), 0) ** 0.5)
    val["diff_formula"] = {"module": m, "materialized": mat_diff, "lowrank": formula,
                           "abs_err": abs(mat_diff - formula),
                           "rel_err": abs(mat_diff - formula) / max(mat_diff, 1e-30)}
    stats["numeric_validation"] = val

    # ---------- 1. 每系列目标侧（一次计算，全部轮复用） ----------
    target_info: dict = {}
    for s in SERIES:
        raw = load_pairs(str(P2 / s / "target_adapter.safetensors"))
        f8 = load_pairs(str(P2 / s / "target/factor_r8.safetensors"))
        # 再现运行时 target_f（QR+SVD 确定性）并核对乘积保持
        tf, prod_err = {}, 0.0
        for m in f8:
            b_t, a_t = rank_r_factorize(f8[m][1], f8[m][0])
            tf[m] = (a_t, b_t)
            n_ref = fro_norm_lowrank(f8[m][0], f8[m][1])
            if n_ref > 0:
                prod_err = max(prod_err, abs(fro_norm_lowrank(a_t, b_t) - n_ref) / n_ref)
        target_info[s] = {
            "raw_pairs": raw, "f8_pairs": f8, "tf_pairs": tf,
            "norm_raw": merged_norm(raw), "norm_f8": merged_norm(f8), "norm_tf": merged_norm(tf),
            "refactor_product_max_rel_err": prod_err,
        }

    # 集中训练检查点轨迹（step_001 与 final）
    ckpt_traj = {}
    for s in SERIES:
        cdir = P2 / s / "checkpoints/round_001/clients/central"
        step1 = load_pairs(str(cdir / "step_001/client_adapter.safetensors"))
        recv = load_pairs(str(cdir / "step_001/received_global_adapter.safetensors"))
        final = load_pairs(str(cdir / "final_adapter.safetensors"))
        b_zero = all(float((b == 0).all()) for _, b in recv.values())
        ckpt_traj[s] = {"step_001_norm": merged_norm(step1),
                        "start_received_norm": merged_norm(recv),
                        "start_received_B_allzero": b_zero,
                        "final_norm": merged_norm(final)}

    # ---------- 2. 逐轮测量 ----------
    rows = []
    eq_records = []
    for s in SERIES:
        ti = target_info[s]
        for bg in PAIRING[s]:
            for rd in ROUNDS:
                rdir = RUNS / bg / "checkpoints" / f"round_{rd:03d}"
                gl = load_pairs(str(rdir / "received_global_adapter.safetensors"))
                norm_global = merged_norm(gl)
                bg_list = [load_pairs(str(rdir / "clients" / c / "final_adapter.safetensors"))
                           for c in BG_SLOTS]
                n_norms = [merged_norm(p) for p in bg_list]
                n_diffs = [merged_diff(p, gl) for p in bg_list]
                # 代码实际权重：8 × 0.1（和为 0.8，不归一化）
                a_bg = {m: sum(0.1 * p[m][0] for p in bg_list) for m in bg_list[0]}
                b_bg = {m: sum(0.1 * p[m][1] for p in bg_list) for m in bg_list[0]}
                R = {m: (ti["tf_pairs"][m][0] - a_bg[m], ti["tf_pairs"][m][1] - b_bg[m])
                     for m in ti["tf_pairs"]}
                R_raw = {m: (ti["f8_pairs"][m][0] - a_bg[m], ti["f8_pairs"][m][1] - b_bg[m])
                         for m in ti["f8_pairs"]}
                norm_R = merged_norm(R)
                norm_R_raw = merged_norm(R_raw)
                ca = load_pairs(str(P7 / s / f"{bg}_r{rd}" / "constructs/a/adapter.safetensors"))
                cb = load_pairs(str(P7 / s / f"{bg}_r{rd}" / "constructs/b/adapter.safetensors"))
                na, nb = merged_norm(ca), merged_norm(cb)
                rhs = 0.99 * norm_R
                ea, eb = na - rhs, nb - rhs
                exp_a = {m: (EPS * R[m][0], (1 - PB * EPS) / PA * R[m][1]) for m in R}
                exp_b = {m: ((1 - PA * EPS) / PB * R[m][0], EPS * R[m][1]) for m in R}
                fw = max(elementwise_check(ca, exp_a), elementwise_check(cb, exp_b))
                eq_records.append({"series": s, "bg": bg, "round": rd,
                                   "lhs_a": na, "lhs_b": nb, "rhs": rhs,
                                   "abs_a": ea, "abs_b": eb,
                                   "rel_a": ea / rhs, "rel_b": eb / rhs})
                rows.append({
                    "series": s, "bg_run": bg, "round": rd,
                    "bg_traj_round_used_by_n_mixed": (3 if bg == "e10bg_clean_s2026" else 1),
                    "norm_global": norm_global,
                    "normal_norms": ";".join(f"{v:.6f}" for v in n_norms),
                    "normal_median": st.median(n_norms), "normal_mean": st.mean(n_norms),
                    "normal_min": min(n_norms), "normal_max": max(n_norms),
                    "normal_diff_vs_global": ";".join(f"{v:.6f}" for v in n_diffs),
                    "normal_diff_median": st.median(n_diffs),
                    "norm_target_raw_r32": ti["norm_raw"],
                    "norm_target_r8": ti["norm_f8"],
                    "r8_over_raw_ratio": ti["norm_f8"] / ti["norm_raw"],
                    "norm_R_after_bg_removal": norm_R,
                    "norm_R_rawf8_sensitivity": norm_R_raw,
                    "norm_upload_a": na, "norm_upload_b": nb,
                    "eq_a_abs": ea, "eq_a_rel": ea / rhs,
                    "eq_b_abs": eb, "eq_b_rel": eb / rhs,
                    "saved_vs_formula_max_rel": fw,
                    "diff_upload_a_vs_global": merged_diff(ca, gl),
                    "diff_upload_b_vs_global": merged_diff(cb, gl),
                    "diff_R_vs_global": merged_diff(R, gl),
                    "diff_target_r8_vs_global": merged_diff(ti["f8_pairs"], gl),
                    "diff_target_raw_vs_global": merged_diff(ti["raw_pairs"], gl),
                    "ratio_a_over_normal_median": na / st.median(n_norms),
                })

    # ---------- 3. 与归档结果对账 ----------
    cs = json.loads((P7 / "construct_summary.json").read_text())
    arch_c = {(r["series"], r["bg"], r["round"]): (r["norm_a"], r["norm_b"])
              for r in cs["per_round"]}
    arch_n = {}
    with open(ROOT / "artifacts/e10/p7/detection/detection_rounds.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if not d["tag"].endswith("|b11"):
                continue
            s, bg, rd = d["tag"][:-4].split("|")
            arch_n[(s, bg, int(rd[1:]))] = [d["norms"][c] for c in BG_SLOTS]
    recon = {"vs_construct_summary_max_abs": 0.0, "vs_detection_normals_max_abs": 0.0}
    for r in rows:
        k = (r["series"], r["bg_run"], r["round"])
        recon["vs_construct_summary_max_abs"] = max(
            recon["vs_construct_summary_max_abs"],
            abs(r["norm_upload_a"] - arch_c[k][0]), abs(r["norm_upload_b"] - arch_c[k][1]))
        mine = [float(x) for x in r["normal_norms"].split(";")]
        recon["vs_detection_normals_max_abs"] = max(
            recon["vs_detection_normals_max_abs"],
            max(abs(x - y) for x, y in zip(mine, arch_n[k])))
    stats["reconciliation_vs_archived"] = recon

    # ---------- 4. 汇总统计 ----------
    def summ(vs):
        return {"n": len(vs), "mean": st.mean(vs), "median": st.median(vs),
                "min": min(vs), "max": max(vs)}

    all_c = [v for r in rows for v in (r["norm_upload_a"], r["norm_upload_b"])]
    all_n = [float(x) for r in rows for x in r["normal_norms"].split(";")]
    per_round_ratio = [r["ratio_a_over_normal_median"] for r in rows]
    overall = {
        "construct_uploads": summ(all_c), "normal_uploads": summ(all_n),
        "ratio_of_means": st.mean(all_c) / st.mean(all_n),
        "ratio_of_medians": st.median(all_c) / st.median(all_n),
        "per_round_ratio_vs_round_median": summ(per_round_ratio),
    }
    by_series = {}
    for s in SERIES:
        rs = [r for r in rows if r["series"] == s]
        c = [v for r in rs for v in (r["norm_upload_a"], r["norm_upload_b"])]
        n = [float(x) for r in rs for x in r["normal_norms"].split(";")]
        by_series[s] = {
            "n_construct": len(c), "n_normal": len(n),
            "construct": summ(c), "normal": summ(n),
            "ratio_of_means": st.mean(c) / st.mean(n),
            "ratio_of_medians": st.median(c) / st.median(n),
            "per_round_ratio_vs_round_median": summ([r["ratio_a_over_normal_median"] for r in rs]),
            "target_raw_r32": rs[0]["norm_target_raw_r32"], "target_r8": rs[0]["norm_target_r8"],
            "refactor_product_max_rel_err": target_info[s]["refactor_product_max_rel_err"],
            "R_norm_range": [min(r["norm_R_after_bg_removal"] for r in rs),
                             max(r["norm_R_after_bg_removal"] for r in rs)],
            "step_traj": ckpt_traj[s],
        }
    by_group = {}
    for grp in ("dolly", "pku"):
        rs = [r for r in rows if r["series"].startswith(grp)]
        c = [v for r in rs for v in (r["norm_upload_a"], r["norm_upload_b"])]
        n = [float(x) for r in rs for x in r["normal_norms"].split(";")]
        by_group[grp] = {
            "n_construct": len(c), "n_normal": len(n),
            "construct": summ(c), "normal": summ(n),
            "ratio_of_means": st.mean(c) / st.mean(n),
            "ratio_of_medians": st.median(c) / st.median(n),
            "per_round_ratio_vs_round_median": summ([r["ratio_a_over_normal_median"] for r in rs]),
        }
    eq_rel = [max(abs(e["rel_a"]), abs(e["rel_b"])) for e in eq_records]
    worst_i = max(range(len(eq_records)), key=lambda i: eq_rel[i])
    we = eq_records[worst_i]
    stats.update({
        "overall": overall, "by_series": by_series, "by_dataset": by_group,
        "equation_check": {
            "n_rounds": len(eq_records),
            "max_rel_err": max(eq_rel), "median_rel_err": st.median(eq_rel),
            "max_abs_err": max(max(abs(e["abs_a"]), abs(e["abs_b"])) for e in eq_records),
            "worst_round": {"series": we["series"], "bg": we["bg"], "round": we["round"],
                            "lhs_a": we["lhs_a"], "lhs_b": we["lhs_b"], "rhs": we["rhs"],
                            "rel_a": we["rel_a"], "rel_b": we["rel_b"]},
            "saved_vs_formula_max_rel_all": max(r["saved_vs_formula_max_rel"] for r in rows),
        },
        "background_reuse": {
            "e10bg_clean_s2026": "3 系列 × 6 轮 = 18 个混合轮；同一 (轨迹,轮) 被 3 个 dolly 系列各用 1 次",
            "每条 pku 轨迹": "1 系列 × 6 轮 = 6 个混合轮；(轨迹,轮) 仅用 1 次",
            "target_reuse": "每系列 1 个集中训练目标，pku 系列复用 24 轮、dolly 系列复用 6 轮",
        },
        "central_ckpt_trajectory": ckpt_traj,
    })

    # ---------- 5. 落盘 ----------
    cols = list(rows[0].keys())
    with open(OUT / "round_norms.tsv", "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    (OUT / "norm_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1),
                                         encoding="utf-8")

    # ---------- 6. 一个可复算轮次的矩阵级等式核验 ----------
    s, bg, rd = "pku_s17", "e10bg_pku_s17_00", 1
    ti = target_info[s]
    rdir = RUNS / bg / "checkpoints" / f"round_{rd:03d}"
    bg_list = [load_pairs(str(rdir / "clients" / c / "final_adapter.safetensors")) for c in BG_SLOTS]
    a_bg = {m: sum(0.1 * p[m][0] for p in bg_list) for m in bg_list[0]}
    b_bg = {m: sum(0.1 * p[m][1] for p in bg_list) for m in bg_list[0]}
    ca = load_pairs(str(P7 / s / f"{bg}_r{rd}" / "constructs/a/adapter.safetensors"))
    m = sorted(ca)[0]
    lhs_mat = ca[m][1] @ ca[m][0]
    rhs_mat = 0.99 * (ti["tf_pairs"][m][1] - b_bg[m]) @ (ti["tf_pairs"][m][0] - a_bg[m])
    stats["matrix_level_check"] = {
        "round": f"{s}|{bg}|r{rd}", "module": m,
        "lhs_fro": float(torch.linalg.norm(lhs_mat)), "rhs_fro": float(torch.linalg.norm(rhs_mat)),
        "fro_abs_diff": float(torch.linalg.norm(lhs_mat - rhs_mat)),
        "fro_rel_diff": float(torch.linalg.norm(lhs_mat - rhs_mat) / torch.linalg.norm(rhs_mat)),
    }
    (OUT / "norm_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1),
                                         encoding="utf-8")

    print(f"[audit] {len(rows)} 轮完成，用时 {time.time() - t0:.0f}s")
    print(f"[audit] 总体: 均值相除={overall['ratio_of_means']:.4f} "
          f"中位相除={overall['ratio_of_medians']:.4f} "
          f"逐轮倍数均值={overall['per_round_ratio_vs_round_median']['mean']:.4f}")
    print(f"[audit] 等式核验: n={len(eq_records)} 最大相对误差={max(eq_rel):.3e} "
          f"最差轮={we['series']}|{we['bg']}|r{we['round']}")
    print(f"[audit] 对账: 构造 {recon['vs_construct_summary_max_abs']:.2e} "
          f"正常 {recon['vs_detection_normals_max_abs']:.2e}")


if __name__ == "__main__":
    main()
