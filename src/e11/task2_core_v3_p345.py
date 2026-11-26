"""E11 Task2 v3 P2-P3 离线阶段：λ 筛选、真实增量回放、S-AB 离线检测与聚合回放。

唯一依据：docs/experiments/E11_TASK2_V3_EXECUTION_PLAN.md §4-§6 与
docs/experiments/prompts/E11_TASK2_V3_LOCAL_AGENT_PROMPTS.md Prompt 3。

当前 U 正向参照未通过（r30 城东 96.88%、北港 83.98%、Pair_exact 3.13%），
S-A/S-B/S-AB 多轮训练锁定。本模块只执行：
1. 225 组 λ 离线尺度筛选（J 准则，U 全部 30 轮真实上传）；
2. J 前五候选的 30 轮真实更新增量回放（U/N 真实增量 → 模拟分支）；
3. 回放存活候选中 J 最小者的 S-AB 离线构造 + 检测/聚合回放（诊断口径）。

不训练模型，不修改旧产物，不重算 U。所有结果标记为
「数学构造与检测诊断」，不作为攻击行为成功的证据。
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.e11.task2_core_v3 import (  # noqa: E402
    ART, BA_REL_TOL, CORE_ART, FACTOR_TOL, J_TOL, LAMBDA_PAIRS, OLD_ART, P, S,
    ba_norm, ba_rel_err, construct_pair_v3, factor_change_norm, factor_fedavg,
    factor_norm, flatten_of, joint_aggregate_v3, load_global, load_n_global,
    load_n_uploads, load_uploads, max_abs_factor_err, sha256_file,
    verify_baseline_hold, verify_dual_path_D, verify_joint_recovery,
    weighted_bg, eff_update_norm)

REP = ROOT / "reports/e11_task2_core_v3/task2_core_v3_dolly_s17_a5"
LOG = ROOT / "logs/e11_task2_core_v3/task2_core_v3_dolly_s17_a5"
U_TRAJ = "U_dolly_s17_a5"
N_TRAJ = "N_dolly_s17_a5"
MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
ROUNDS = 30
MAL_AB = {"client_00", "client_01"}


def _log(msg: str) -> None:
    LOG.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG / "p345_v3.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- 每轮输入缓存
class RoundInputs:
    """U 第 r 轮的真实输入：全局因子、10 客户端上传（pairs）、U 聚合结果、背景加权。"""

    def __init__(self, r: int):
        self.r = r
        g = load_global(U_TRAJ, r - 1)
        blob = load_uploads(U_TRAJ, r)
        if sorted(blob) != [f"client_{i:02d}" for i in range(10)]:
            raise AssertionError(f"U r{r} 上传客户端集合不符: {sorted(blob)}")
        self.global_pairs = g
        self.uploads_flat = blob
        self.t_pairs = _pairs(blob["client_00"])
        self.ref_pairs = _pairs(blob["client_01"])
        self.bg_flat = {c: blob[c] for c in sorted(blob) if int(c.split("_")[1]) >= 2}
        self.bg_pairs = {c: _pairs(v) for c, v in self.bg_flat.items()}
        self.u_agg = factor_fedavg([self.t_pairs, self.ref_pairs]
                                   + [self.bg_pairs[c] for c in sorted(self.bg_pairs)])
        self.a_bg, self.b_bg = weighted_bg(self.bg_flat)

    def normal_scales(self) -> tuple[float, float]:
        """M_N,r、M_F,r：九个正常客户端（client_01..09）的中位数。"""
        ns = [eff_update_norm(self.bg_pairs[c] if c in self.bg_pairs else self.ref_pairs,
                              self.global_pairs) for c in
              [f"client_{i:02d}" for i in range(1, 10)]
              if c in self.bg_pairs or c == "client_01"]
        ns = ([eff_update_norm(self.ref_pairs, self.global_pairs)]
              + [eff_update_norm(self.bg_pairs[c], self.global_pairs)
                 for c in sorted(self.bg_pairs)])
        fs = ([factor_change_norm(self.ref_pairs, self.global_pairs)]
              + [factor_change_norm(self.bg_pairs[c], self.global_pairs)
                 for c in sorted(self.bg_pairs)])
        if len(ns) != 9 or len(fs) != 9:
            raise AssertionError(f"r{self.r} 正常参照客户端数 != 9")
        mn, mf = float(torch.median(torch.tensor(ns))), float(torch.median(torch.tensor(fs)))
        if not (mn > 0 and math.isfinite(mn) and mf > 0 and math.isfinite(mf)):
            raise AssertionError(f"r{self.r} 正常中位数非正/非有限: M_N={mn}, M_F={mf}")
        return mn, mf


def _pairs(flat: dict) -> dict:
    from src.e10.lora_algebra import module_pairs
    return module_pairs(dict(flat))


# ---------------------------------------------------------------- P2：225 组 λ 筛选
def run_lambda_screen() -> dict:
    """§4.1-4.3：J(λ_A,λ_B) = max_r max_{X∈{A,B}} {|ln(N_X,r/M_N,r)|, |ln(F_X,r/M_F,r)|}。"""
    _log("P2 λ 筛选开始：加载 U 30 轮真实上传")
    inputs = [RoundInputs(r) for r in range(1, ROUNDS + 1)]
    # 预计算每轮 λ 无关量：D_A/D_B 每轮只依赖 (g, T, N1)；先算 N/M、F/M 基础结构
    per_round = []
    for ri in inputs:
        con0 = construct_pair_v3(ri.global_pairs, ri.t_pairs, ri.ref_pairs, 1.0, 1.0)
        mn, mf = ri.normal_scales()
        per_round.append({
            "r": ri.r, "M_N": mn, "M_F": mf,
            "D_A": con0["D_A"], "D_B": con0["D_B"],
            "global_pairs": ri.global_pairs, "ri": ri})
    # 逐组 λ：只需线性组合 λ_A·D_A 等计算范数（与构造一致的公式）
    screen = {"stage": "P2_lambda_screen_v3", "time": time.strftime("%Y-%m-%d %H:%M:%S"),
              "grid": [la for la, _ in LAMBDA_PAIRS[:0]] or sorted({la for la, _ in LAMBDA_PAIRS}),
              "n_pairs": len(LAMBDA_PAIRS), "j_tol": J_TOL,
              "rule": "J=max_r max(|ln(N_X/M_N)|,|ln(F_X/M_F)|), X∈{A,B}; 合格 J<=ln2",
              "candidates": {}}
    results = {}
    for la, lb in LAMBDA_PAIRS:
        j = 0.0
        worst = None
        ok = True
        for pr in per_round:
            g = pr["global_pairs"]
            # A/B 因子（公式与 construct_pair_v3 一致；避免重复构造对象开销）
            for x, kA, kB in (("A", la, 10.0 - lb), ("B", 10.0 - la, lb)):
                # A_x = A_g + kA·D_A, B_x = B_g + kB·D_B
                nx = eff_update_lambda(g, pr["D_A"], pr["D_B"], kA, kB)
                fx = factor_change_lambda(g, pr["D_A"], pr["D_B"], kA, kB)
                if not (math.isfinite(nx) and math.isfinite(fx)):
                    raise FloatingPointError(
                        f"λ=({la},{lb}) r{pr['r']} X={x} 非有限范数 N={nx} F={fx}")
                v = max(abs(math.log(nx / pr["M_N"])), abs(math.log(fx / pr["M_F"])))
                if v > j:
                    j, worst = v, (pr["r"], x)
        results[(la, lb)] = {"J": j, "worst": worst}
        _log(f"λ=({la},{lb}) J={j:.4f} worst={worst}")
    # 排序与合格判定（§4.3 tie-break）
    def key(item):
        (la, lb), v = item
        return (v["J"], abs(la - 5.0) + abs(lb - 5.0), la, lb)
    ranked = sorted(results.items(), key=key)
    eligible = [k for k, v in ranked if v["J"] <= J_TOL]
    screen["candidates"] = {f"{la},{lb}": results[(la, lb)] for (la, lb), _ in ranked}
    screen["ranked_top10"] = [{"lambda_A": la, "lambda_B": lb, **results[(la, lb)]}
                              for (la, lb), _ in ranked[:10]]
    screen["eligible"] = [{"lambda_A": la, "lambda_B": lb} for (la, lb) in eligible]
    screen["n_eligible"] = len(eligible)
    screen["per_round_median"] = [{"r": pr["r"], "M_N": pr["M_N"], "M_F": pr["M_F"]}
                                  for pr in per_round]
    screen["input_sha256_16"] = {
        f"uploads_round_{r:03d}": sha256_file(CORE_ART / "trajectories" / U_TRAJ
                                              / f"uploads_round_{r:03d}.pt")[:16]
        for r in range(1, ROUNDS + 1)}
    screen["passed"] = len(eligible) >= 1
    out = ART / "lambda_screen.json"
    out.write_text(json.dumps(screen, ensure_ascii=False, indent=1), encoding="utf-8")
    _log(f"λ 筛选完成：合格 {len(eligible)}/225；已写 {out}")
    return {"screen": screen, "ranked": ranked, "eligible": eligible,
            "per_round": per_round, "inputs": inputs}


def eff_update_lambda(g: dict, d_a: dict, d_b: dict, ka: float, kb: float) -> float:
    """N = sqrt(Σ_m ‖s(B_x A_x − B_g A_g)‖²_F)，A_x = A_g+ka·D_A，B_x = B_g+kb·D_B。

    低秩迹恒等式实现（§3.6 展开式）：
    ‖BA−CG‖² = tr(BᵀB·AAᵀ) + tr(CᵀC·GGᵀ) − 2·tr(BᵀC·GAᵀ)，
    全部为 (r,r) 小矩阵运算，与显式 BA 乘积数值一致（float64）。
    与 construct_pair_v3 + eff_update_norm 同一公式。
    """
    tot = 0.0
    for m in g:
        a_g, b_g = g[m][0].double(), g[m][1].double()
        da, db = d_a[m], d_b[m]
        a_x = a_g + ka * da
        b_x = b_g + kb * db
        bt_b = b_x.T @ b_x                     # (r,r)
        aa_t = a_x @ a_x.T                     # (r,r)
        gt_g = b_g.T @ b_g
        gg_t = a_g @ a_g.T     # note: this is A_g A_gᵀ
        bgx = b_x.T @ b_g                       # (r,r) = BᵀC
        gax = a_g @ a_x.T                       # (r,r) = G Aᵀ
        n1 = float(torch.trace(bt_b @ aa_t))
        n2 = float(torch.trace(gt_g @ gg_t))
        ip = float(torch.trace(bgx @ gax))
        tot += max(n1 + n2 - 2.0 * ip, 0.0)
    return S * math.sqrt(tot)


def factor_change_lambda(g: dict, d_a: dict, d_b: dict, ka: float, kb: float) -> float:
    """F = sqrt(Σ_m (‖ka·D_A‖² + ‖kb·D_B‖²))（原始因子变化范数）。"""
    tot = 0.0
    for m in g:
        tot += float(((ka * d_a[m]) ** 2).sum()) + float(((kb * d_b[m]) ** 2).sum())
    return math.sqrt(tot)


# ---------------------------------------------------------------- P2：30 轮真实增量回放（§4.4）
def run_delta_replay(top5: list, screen_inputs: dict) -> dict:
    """J 前五候选的 30 轮真实更新增量回放。

    U 轨迹提供 T、N1、背景 8 客户端的真实增量；N 轨迹提供 S-B 正常 client_00 增量；
    S-B 的 client_01 用 U 同轮正常 client_01 增量作为 shadow reference；
    每轮以模拟分支上一轮全局因子为新基线。
    """
    _log(f"增量回放开始：候选 {top5}")
    n_blob_prev = None
    u_prev = None
    n_prev = None
    out = {"stage": "P2_delta_replay_v3", "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "candidates": {}, "elim_rules": {
               "nonfinite": "任意张量 NaN/Inf",
               "scale_out": "因子变化或有效更新范数超出正常中位数 [0.25,4]",
               "global_factor_gt4x": "模拟分支全局因子拼接范数 > U 同轮 4 倍",
               "ba_gt3x": "模拟分支有效 BA 权重范数 > U 同轮 3 倍",
               "joint_err": "联合因子恢复 > 1e-6 或 BA 相对误差 > 1e-5"},
           "note": "离线回放：真实本地更新增量重放到模拟分支；非真实训练结论"}
    for (la, lb) in top5:
        cand = {"lambda_A": la, "lambda_B": lb, "rounds": [], "eliminated": None}
        # 模拟分支全局：S-A 与 S-B 共用同一构造基线流程但聚合槽位不同
        sa_global = load_global(U_TRAJ, 0)  # r0 = init（与 U 同起点）
        sb_global = load_global(U_TRAJ, 0)
        u_global = load_global(U_TRAJ, 0)
        for r in range(1, ROUNDS + 1):
            # 参照轨迹的真实增量（相对各自上一轮全局）
            u_g_ref = load_global(U_TRAJ, r - 1)
            n_g_ref = load_n_global(r - 1)
            u_blob = load_uploads(U_TRAJ, r)
            n_blob = load_n_uploads(r)
            d_t = {m: (_pairs(u_blob["client_00"])[m][0].double() - u_g_ref[m][0].double(),
                       _pairs(u_blob["client_00"])[m][1].double() - u_g_ref[m][1].double())
                   for m in u_g_ref}
            d_1 = {m: (_pairs(u_blob["client_01"])[m][0].double() - u_g_ref[m][0].double(),
                       _pairs(u_blob["client_01"])[m][1].double() - u_g_ref[m][1].double())
                   for m in u_g_ref}
            d_0n = {m: (_pairs(n_blob["client_00"])[m][0].double() - n_g_ref[m][0].double(),
                        _pairs(n_blob["client_00"])[m][1].double() - n_g_ref[m][1].double())
                    for m in n_g_ref}
            d_bg = {c: {m: (_pairs(u_blob[c])[m][0].double() - u_g_ref[m][0].double(),
                            _pairs(u_blob[c])[m][1].double() - u_g_ref[m][1].double())
                        for m in u_g_ref}
                    for c in sorted(u_blob) if int(c.split("_")[1]) >= 2}
            # 增量施加到模拟分支当前全局
            apply_inc = lambda g, d: {m: (g[m][0].double() + d[m][0], g[m][1].double() + d[m][1])
                                        for m in g}
            t_sim = apply_inc(sa_global, d_t)
            n1_sim = apply_inc(sa_global, d_1)
            bg_sim = {c: apply_inc(sa_global, d_bg[c]) for c in d_bg}
            n0_sim = apply_inc(sb_global, d_0n)
            # v3 构造（用模拟分支全局为基线；S-B 用同一轮 U 的 client_01 增量为 shadow）
            con_sa = construct_pair_v3(sa_global, t_sim, n1_sim, la, lb)
            con_sb = construct_pair_v3(sb_global, t_sim, apply_inc(sb_global, d_1), la, lb)
            # 模拟十客户端上传
            sa_uploads = {**{c: bg_sim[c] for c in bg_sim}, "client_00": con_sa["a"],
                          "client_01": n1_sim}
            sb_uploads = {**{c: bg_sim[c] for c in bg_sim}, "client_00": n0_sim,
                          "client_01": con_sb["b"]}
            # 聚合
            sa_global_next = joint_aggregate_v3(sa_uploads["client_00"],
                                                sa_uploads["client_01"], bg_sim)
            sb_global_next = joint_aggregate_v3(sb_uploads["client_00"],
                                                sb_uploads["client_01"], bg_sim)
            # 逐轮诊断
            row = {"round": r}
            for tag, glob_next, con, mal_pairs, norm_pairs in (
                    ("S-A", sa_global_next, con_sa, con_sa["a"], n1_sim),
                    ("S-B", sb_global_next, con_sb, con_sb["b"], n0_sim)):
                row[tag] = {
                    "global_A_norm": factor_norm(glob_next, "A"),
                    "global_B_norm": factor_norm(glob_next, "B"),
                    "global_BA_norm": ba_norm(glob_next),
                    "mal_eff_update_norm": eff_update_norm(mal_pairs, sa_global if tag == "S-A" else sb_global),
                    "mal_factor_change_norm": factor_change_norm(mal_pairs, sa_global if tag == "S-A" else sb_global),
                    "norm_ref_eff_update": eff_update_norm(norm_pairs, sa_global if tag == "S-A" else sb_global),
                }
            # 与 U 同轮比较
            u_global_r = load_global(U_TRAJ, r)
            row["U"] = {"global_factor_norm": math.sqrt(factor_norm(u_global_r, "A") ** 2
                                                        + factor_norm(u_global_r, "B") ** 2),
                        "global_BA_norm": ba_norm(u_global_r)}
            for tag in ("S-A", "S-B"):
                row[tag]["global_factor_norm"] = math.sqrt(row[tag]["global_A_norm"] ** 2
                                                           + row[tag]["global_B_norm"] ** 2)
            # 联合恢复核验（S-AB = A + B + 背景 → U 全局）
            joint = joint_aggregate_v3(con_sa["a"], con_sb["b"], bg_sim)
            row["joint_recovery"] = {
                "max_abs_err_A": max_abs_factor_err(joint, u_global_r, "A"),
                "max_abs_err_B": max_abs_factor_err(joint, u_global_r, "B"),
                "ba_rel_err": ba_rel_err(joint, u_global_r)[0]}
            # 有限性检查
            finite = True
            for d in (sa_global_next, sb_global_next, con_sa["a"], con_sa["b"],
                      con_sb["a"], con_sb["b"]):
                for m in d:
                    if not (torch.isfinite(d[m][0]).all() and torch.isfinite(d[m][1]).all()):
                        finite = False
            row["finite"] = finite
            cand["rounds"].append(row)
            # ---- 硬性淘汰条件
            elim = None
            if not finite:
                elim = "nonfinite"
            else:
                mn, mf = _medians_from_row(r, screen_inputs)
                for tag in ("S-A", "S-B"):
                    ratio_n = row[tag]["mal_eff_update_norm"] / mn
                    ratio_f = row[tag]["mal_factor_change_norm"] / mf
                    if not (0.25 <= ratio_n <= 4.0) or not (0.25 <= ratio_f <= 4.0):
                        elim = f"scale_out:{tag}:r{r}:N={ratio_n:.2f},F={ratio_f:.2f}"
                        break
                if elim is None:
                    u_factor_norm = row["U"]["global_factor_norm"]
                    for tag in ("S-A", "S-B"):
                        if row[tag]["global_factor_norm"] > 4.0 * u_factor_norm:
                            elim = f"global_factor_gt4x:{tag}:r{r}"
                            break
                if elim is None:
                    for tag in ("S-A", "S-B"):
                        if row[tag]["global_BA_norm"] > 3.0 * row["U"]["global_BA_norm"]:
                            elim = f"ba_gt3x:{tag}:r{r}"
                            break
                if elim is None:
                    jr = row["joint_recovery"]
                    if jr["max_abs_err_A"] > FACTOR_TOL or jr["max_abs_err_B"] > FACTOR_TOL \
                            or jr["ba_rel_err"] > BA_REL_TOL:
                        elim = f"joint_err:r{r}:A={jr['max_abs_err_A']:.2e},B={jr['max_abs_err_B']:.2e},BA={jr['ba_rel_err']:.2e}"
            if elim:
                cand["eliminated"] = elim
                _log(f"候选 ({la},{lb}) 淘汰于 r{r}: {elim}")
                break
            sa_global, sb_global = sa_global_next, sb_global_next
            del u_blob, n_blob
        cand["survived"] = cand["eliminated"] is None
        out["candidates"][f"{la},{lb}"] = cand
        _log(f"候选 ({la},{lb}) 回放完成: survived={cand['survived']}")
    survivors = [k for k, v in out["candidates"].items() if v["survived"]]
    out["survivors"] = [k for k in survivors]
    out["passed"] = len(survivors) >= 1
    path = ART / "delta_replay.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    _log(f"回放完成：存活 {len(survivors)}；已写 {path}")
    return out


def _medians_from_row(r: int, screen_inputs: dict) -> tuple[float, float]:
    for pr in screen_inputs.get("per_round_medians", []):
        if pr["r"] == r:
            return pr["M_N"], pr["M_F"]
    raise AssertionError(f"缺少 r{r} 中位数")


# ---------------------------------------------------------------- §4.5 S-AB 离线诊断
def run_sab_diagnostic(lambda_pair: tuple) -> dict:
    """S-AB 离线构造 + 数学恢复核验 + 检测/聚合回放（诊断口径）。

    client_00 = A，client_01 = B，client_02..09 = U 背景上传；
    每轮核验 S-AB 与 U 的因子恢复和 BA 误差；检测分数从 A/B 上传重新计算。
    """
    la, lb = lambda_pair
    _log(f"S-AB 离线诊断开始：diagnostic λ=({la},{lb})")
    from src.detect.cbdf import cbdf
    from src.detect.lowrank import gram, make_update, norms_and_cosine
    from src.fed.aggregate import fedavg, foolsgold, multi_krum, trimmed_mean

    la, lb = float(la), float(lb)
    recon = {"stage": "S-AB_reconstruction_v3", "diagnostic_lambda": [la, lb],
             "time": time.strftime("%Y-%m-%d %H:%M:%S"), "rounds": [],
             "note": "S-AB 由 U 每轮真实上传离线构造；全局检查点即 U 同轮检查点"}
    det_rows = []
    agg_rows = []
    fg_history = []
    cbdf_hist = []
    ids_all = None
    for r in range(1, ROUNDS + 1):
        ri_globals = load_global(U_TRAJ, r - 1)
        blob = load_uploads(U_TRAJ, r)
        t_pairs = _pairs(blob["client_00"])
        ref_pairs = _pairs(blob["client_01"])
        bg_flat = {c: blob[c] for c in sorted(blob) if int(c.split("_")[1]) >= 2}
        con = construct_pair_v3(ri_globals, t_pairs, ref_pairs, la, lb)
        # S-AB 上传：A、B、8 背景（flatten 形式，与旧检测输入格式一致）
        sab_uploads = {c: blob[c] for c in bg_flat}
        sab_uploads["client_00"] = flatten_of(con["a"])
        sab_uploads["client_01"] = flatten_of(con["b"])
        ids = sorted(sab_uploads)
        ids_all = ids
        received = flatten_of(ri_globals)
        # ---- 恢复核验
        joint = joint_aggregate_v3(con["a"], con["b"], bg_flat)
        u_agg = factor_fedavg([t_pairs, ref_pairs]
                              + [_pairs(bg_flat[c]) for c in sorted(bg_flat)])
        max_a = max_abs_factor_err(joint, u_agg, "A")
        max_b = max_abs_factor_err(joint, u_agg, "B")
        rel, wu = ba_rel_err(joint, u_agg)
        # 服务器真实 FedAvg 复算
        agg_flat = fedavg([sab_uploads[c] for c in ids], [P] * 10)
        agg_pairs = _pairs(agg_flat)
        fa = max_abs_factor_err(agg_pairs, u_agg, "A")
        fb = max_abs_factor_err(agg_pairs, u_agg, "B")
        ok = max(max_a, max_b, fa, fb) <= FACTOR_TOL and rel <= BA_REL_TOL
        recon["rounds"].append({
            "round": r, "max_abs_err_A": max_a, "max_abs_err_B": max_b,
            "ba_rel_err": rel, "server_fedavg_err_A": fa, "server_fedavg_err_B": fb,
            "ok": ok,
            "inputs_sha16": {"uploads": sha256_file(
                CORE_ART / "trajectories" / U_TRAJ / f"uploads_round_{r:03d}.pt")[:16]}})
        if not ok:
            raise AssertionError(f"S-AB r{r} 恢复核验失败: A={max_a:.2e} B={max_b:.2e} BA={rel:.2e}")
        # ---- 检测（与旧 _detection_row 同口径；分数从 A/B 上传重新计算）
        scale = S
        updates = [make_update(sab_uploads[c], received, scale, MODULES) for c in ids]
        matrix = gram(updates)
        norms, cosine = norms_and_cosine(matrix)
        import numpy as np
        n = len(ids)
        cos_score = {c: 1.0 - float(np.mean([float(cosine[i, j]) for j in range(n) if j != i]))
                     for i, c in enumerate(ids)}
        _, detail, fg_history = foolsgold([sab_uploads[c] for c in ids], received,
                                          history=fg_history)
        weights = detail["weights"]
        det_rows.append({
            "round": r, "clients": ids,
            "norm": {c: float(norms[i]) for i, c in enumerate(ids)},
            "cosine": cos_score,
            "cosine_raw_mean_pairwise": {c: float(cosine[i, :].sum() - 1) / (n - 1)
                                         for i, c in enumerate(ids)},
            "foolsgold_weight": {c: float(weights[i]) for i, c in enumerate(ids)},
            "foolsgold_score": {c: 1.0 - float(weights[i]) for i, c in enumerate(ids)},
            "fg_history_rounds": detail["history_rounds"],
            "fg_degenerate": detail["degenerate_equal_weights"]})
        cbdf_hist.append(updates)
        # ---- 聚合回放（Multi-Krum / Trimmed Mean；同轮离线，不反馈训练）
        states = [sab_uploads[c] for c in ids]
        _, mk = multi_krum(states, received, f=2, selected=8)
        chosen = {ids[i] for i in mk["selected_indices"]}
        tm = trimmed_mean(states, trim_count=2)
        tm_pairs = _pairs(tm)
        agg_rows.append({
            "round": r,
            "multi_krum": {"selected": sorted(chosen),
                           "excluded": sorted(set(ids) - chosen),
                           "malicious_selected": sorted(MAL_AB & chosen),
                           "malicious_excluded": sorted(MAL_AB - chosen),
                           "scores": {ids[i]: float(mk["scores"][i]) for i in range(len(ids))}},
            "trimmed_mean": {"ba_rel_err_vs_U": ba_rel_err(tm_pairs, u_agg)[0],
                             "A_norm": factor_norm(tm_pairs, "A"),
                             "B_norm": factor_norm(tm_pairs, "B"),
                             "BA_norm": ba_norm(tm_pairs)}})
        _log(f"S-AB r{r}: A={max_a:.2e} B={max_b:.2e} BA_rel={rel:.2e} "
             f"mk_excl={sorted(set(ids) - chosen)}")
        del blob
    # ---- CBDF（完整 30 轮历史）
    try:
        cb = cbdf(cbdf_hist, ids_all, k=3, edge_rule="algorithm1_either_directed")
        cbdf_out = {"detected_clients": cb["detected_clients"],
                    "detected_cliques": cb["detected_cliques"],
                    "final_malicious_clients_union": cb["final_malicious_clients_union"],
                    "edge_rule": cb["edge_rule"], "K": cb["K"], "rounds": cb["rounds"]}
    except Exception as exc:
        cbdf_out = {"error": repr(exc)}
    det = {"stage": "S-AB_detector_replay_v3", "diagnostic_lambda": [la, lb],
           "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "note": "所有检测输入来自实际构造的 A/B 上传；分数重新计算，未复制 U 结果。"
                   "诊断口径：U 行为门槛未过，不作为攻击行为成功的证据。",
           "rounds": det_rows, "cbdf": cbdf_out,
           "malicious_slots": sorted(MAL_AB)}
    agg = {"stage": "S-AB_aggregation_replay_v3", "diagnostic_lambda": [la, lb],
           "rounds": agg_rows}
    recon_path = ART / "S-AB_reconstruction.json"
    det_path = ART / "S-AB_detector_replay.json"
    recon_path.write_text(json.dumps(recon, ensure_ascii=False, indent=1), encoding="utf-8")
    det_path.write_text(json.dumps(det, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(ART / "S-AB_aggregation_replay.jsonl", "w", encoding="utf-8") as f:
        for row in agg_rows:
            f.write(json.dumps({"stage": "S-AB_aggregation_replay_v3",
                                "diagnostic_lambda": [la, lb], **row},
                               ensure_ascii=False) + "\n")
    diagnostic = {"stage": "diagnostic_lambda_v3", "diagnostic_lambda": [la, lb],
                  "note": "U 正向参照未通过行为门槛；该 λ 仅用于本次离线检测诊断，"
                          "不是最终 selected_lambda。最终参数须等新 U 参照通过后重选。",
                  "rule": "J 最小的回放存活候选",
                  "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    (ART / "diagnostic_lambda.json").write_text(
        json.dumps(diagnostic, ensure_ascii=False, indent=1), encoding="utf-8")
    _log(f"S-AB 诊断完成：recon 30 轮全部 ok；检测 {len(det_rows)} 轮")
    return {"recon": recon, "det": det, "agg": agg}


# ---------------------------------------------------------------- 主入口
def main() -> None:
    _log("P345 v3 启动（离线路线：U 门槛失败 → 诊断路径）")
    audit = json.loads((ART / "input_audit.json").read_text())
    if not audit.get("passed"):
        raise AssertionError("input_audit 未通过，禁止继续")
    screen_out = run_lambda_screen()
    if not screen_out["screen"]["passed"]:
        _log("225 组 λ 全部不合格 → 停止，无可行参数")
        _write_stage_gate("failed_no_eligible_lambda", screen_out, None, None)
        return
    ranked = screen_out["ranked"]
    top5 = [(float(la), float(lb)) for (la, lb), _ in ranked[:5]]
    replay = run_delta_replay(top5, {"per_round_medians":
                                      screen_out["screen"]["per_round_median"]})
    survivors = [tuple(float(x) for x in k.split(",")) for k in replay.get("survivors", [])]
    replay_survivors = {"stage": "replay_survivors_v3",
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "top5": [{"lambda_A": la, "lambda_B": lb} for la, lb in top5],
                        "survivors": [{"lambda_A": la, "lambda_B": lb} for la, lb in survivors],
                        "eliminated": [{"lambda_A": float(k.split(",")[0]),
                                        "lambda_B": float(k.split(",")[1]),
                                        "reason": v["eliminated"]}
                                       for k, v in replay["candidates"].items()
                                       if not v["survived"]],
                        "note": "存活候选按 J 升序；本文件不含最终 selected_lambda（U 门槛失败）"}
    (ART / "replay_survivors.json").write_text(
        json.dumps(replay_survivors, ensure_ascii=False, indent=1), encoding="utf-8")
    if not survivors:
        _log("回放无存活候选 → 跳过 S-AB 诊断，报告原因")
        _write_stage_gate("failed_no_replay_survivor", screen_out, replay, None)
        return
    j_map = {f"{la},{lb}": v["J"] for (la, lb), v in ranked}
    diag_lambda = min(survivors, key=lambda k: (j_map[f"{k[0]},{k[1]}"],
                                                abs(k[0] - 5) + abs(k[1] - 5), k[0], k[1]))
    sab = run_sab_diagnostic(diag_lambda)
    _write_stage_gate("offline_diagnostic_complete", screen_out, replay, sab, diag_lambda)
    _log("P345 v3 离线诊断全部完成")


def _write_stage_gate(status: str, screen_out: dict, replay: dict | None,
                      sab: dict | None, diag_lambda: tuple | None = None) -> None:
    lines = ["# E11 Task2 v3 阶段门报告（stage_gate_report）", "",
             f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"- 状态：{status}", "",
             "## 门状态", "",
             "- U r30 正向参照：**未通过**（城东 96.88%、北港 83.98%、Pair_exact 3.13%）",
             "- S-A/S-B/S-AB 多轮训练：**锁定**（未启动，未解锁）",
             "- 前三名候选行为探针：**跳过**（按 Prompt 3 指示，不训练正常 client_00）",
             "- 完整开发集单轮预检：**跳过**（同上）",
             "- selected_lambda：**不产生**（仅 diagnostic_lambda）", ""]
    if screen_out is not None:
        sc = screen_out["screen"]
        lines += ["## λ 筛选（225 组，U 30 轮真实上传）", "",
                  f"- 合格组数：{sc['n_eligible']}/225（J ≤ ln2 = {J_TOL:.4f}）",
                  f"- J 最小候选：{sc['ranked_top10'][0] if sc['ranked_top10'] else '无'}", ""]
    if replay is not None:
        lines += ["## 30 轮真实更新增量回放", "",
                  f"- 存活：{len(replay.get('survivors', []))}/5",
                  f"- 淘汰：{json.dumps(replay.get('candidates', {}).get('eliminated', {}), ensure_ascii=False) if replay else ''}", ""]
    if diag_lambda is not None:
        lines += [f"- 诊断 λ：(λ_A={diag_lambda[0]}, λ_B={diag_lambda[1]})", ""]
    if sab is not None:
        lines += ["## S-AB 离线诊断", "",
                  "- 数学恢复：30/30 轮通过（因子 ≤1e-6，BA 相对 ≤1e-5，服务器 FedAvg 复算一致）",
                  "- 检测重放：Norm/Cosine/FoolsGold 逐轮重算；CBDF 30 轮历史；Multi-Krum/Trimmed Mean 同轮回放",
                  "- 口径：数学构造与检测诊断（U 行为门槛未过，不作为攻击行为成功证据）", ""]
    lines += ["## 未完成事项（等待用户）", "",
              "1. 新的 U 正向参照设计（训练数据/方法变更需单独设计，不由 Agent 自行修改）；",
              "2. U 通过门槛后的前三候选 144 条探针预检与单轮完整开发集预检；",
              "3. S-A、S-B 五轮试运行与 30 轮训练（Prompt 4 锁定）；",
              "4. Prompt 5 最终检测分析、统计检验与报告。"]
    (REP / "stage_gate_report.md").parent.mkdir(parents=True, exist_ok=True)
    (REP / "stage_gate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
