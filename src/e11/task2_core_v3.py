"""E11 Task 2 v3 数学核心：全局因子基线保持的双客户端 A/B 构造。

唯一依据：docs/experiments/E11_TASK2_V3_EXECUTION_PLAN.md（v3.0）。

与旧 Task2（construct_pair_core，ε 分配 + Procrustes 旋转）的关键差异：
1. 不执行 Procrustes 旋转，直接在当前全局因子原始坐标上构造；
2. D_A、D_B 定义为 p·[(A_t−A_g)+(A_1−A_g)]，同时包含目标客户端与正常参照
   client_01 相对同一全局模型的真实因子变化；
3. λ_A、λ_B ∈ (0,10) 独立分配 A/B 因子；
4. 联合 A/B 聚合精确恢复 U 的聚合因子（§3.4 证明），与 λ 取值无关；
5. 服务器执行 A 因子分别平均、B 因子分别平均，然后计算 BA；
   禁止把客户端 BA 矩阵加权平均当作因子聚合结果。

本文件只实现数学模块与只读核验；不训练模型，不修改旧 Task2 代码与产物。
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.e10.lora_algebra import from_pairs, module_pairs  # noqa: E402
from src.io import atomic_json  # noqa: E402

RUN_ID = "task2_core_v3_dolly_s17_a5"
CONFIG_PATH = ROOT / "configs/e11/task2_core_v3.yaml"
CORE_ART = ROOT / "artifacts/e11_task2_core/task2_core_dolly_s17_a5"
OLD_ART = ROOT / "artifacts/e11_task2/task2_20261125"
ART = ROOT / "artifacts/e11_task2_core_v3" / RUN_ID

P = 0.1                 # 每客户端等权聚合权重（10 客户端）
S = 2.0                 # LoRA 缩放 s = alpha/rank = 16/8
N_CLIENTS = 10
LAMBDA_GRID = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.5, 9.75, 9.9]
LAMBDA_PAIRS = [(la, lb) for la in LAMBDA_GRID for lb in LAMBDA_GRID]  # 225 组
FACTOR_TOL = 1e-6       # 逐模块联合恢复绝对误差
BA_REL_TOL = 1e-5       # 完整有效 BA 相对误差
J_TOL = math.log(2.0)   # λ 筛选合格阈值（0.5~2 倍正常中位数）

MAL_SLOTS = {"S-A": ("client_00", "client_01"),
             "S-B": ("client_01", "client_00"),
             "S-AB": ("client_00", "client_01")}
BG_CLIENTS = [f"client_{i:02d}" for i in range(2, 10)]
NORMAL_REF_CLIENTS = [f"client_{i:02d}" for i in range(1, 10)]  # 9 个正常参照（client_01..09）


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cfg() -> dict:
    import yaml
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["_config_path"] = str(CONFIG_PATH)
    return cfg


# ---------------------------------------------------------------- 输入装载（只读）
def _pairs_from_flat(flat: dict) -> dict:
    return module_pairs(dict(flat))


def load_pairs_file(path: Path) -> dict:
    from safetensors.torch import load_file
    return _pairs_from_flat(dict(load_file(str(path))))


def save_pairs_file(pairs: dict, path: Path) -> None:
    from safetensors.torch import save_file
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in from_pairs(pairs).items()}, str(path))


def flatten_of(pairs: dict) -> dict:
    return {k: v.contiguous() for k, v in from_pairs(pairs).items()}


def load_uploads(traj: str, r: int) -> dict:
    """U 轨迹第 r 轮真实上传 {client: peft 扁平 state}。"""
    path = CORE_ART / "trajectories" / traj / f"uploads_round_{r:03d}.pt"
    return torch.load(path, map_location="cpu", weights_only=True)


def load_global(traj: str, r: int) -> dict:
    """第 r 轮全局因子 pairs；r=0 为 init。"""
    base = CORE_ART / "trajectories" / traj
    path = base / "init_adapter.safetensors" if r == 0 else base / f"global_round_{r:03d}.safetensors"
    return load_pairs_file(path)


def load_n_uploads(r: int) -> dict:
    """N 正常轨迹第 r 轮真实上传（S-B 回放用正常 client_00 增量）。"""
    path = OLD_ART / "trajectories" / "N_dolly_s17_a5" / f"uploads_round_{r:03d}.pt"
    return torch.load(path, map_location="cpu", weights_only=True)


def load_n_global(r: int) -> dict:
    base = OLD_ART / "trajectories" / "N_dolly_s17_a5"
    path = base / "init_adapter.safetensors" if r == 0 else base / f"global_round_{r:03d}.safetensors"
    return load_pairs_file(path)


def _check_finite(pairs: dict, label: str) -> None:
    for m, (a, b) in pairs.items():
        for t, which in ((a, "A"), (b, "B")):
            if not torch.isfinite(t).all():
                raise FloatingPointError(f"非有限张量: {label} {m} {which}")


def _check_module_set(pairs: dict, reference: dict, label: str) -> None:
    if sorted(pairs) != sorted(reference):
        raise AssertionError(f"模块集合不一致: {label}: "
                             f"missing={sorted(set(reference)-set(pairs))[:3]} "
                             f"extra={sorted(set(pairs)-set(reference))[:3]}")


# ---------------------------------------------------------------- v3 数学构造（§3）
def compute_D(global_pairs: dict, target_pairs: dict, ref_pairs: dict) -> tuple[dict, dict]:
    """D_A = p[(A_t−A_g)+(A_1−A_g)]，D_B 同理。返回 {module: D_A 张量} 形式。

    target_pairs: 当前轮目标客户端 T 真实上传（pairs）；
    ref_pairs:    当前轮正常参照 client_01 真实上传（pairs）；
    global_pairs: 当前轮分支全局因子（pairs）。
    """
    _check_module_set(target_pairs, global_pairs, "target")
    _check_module_set(ref_pairs, global_pairs, "ref")
    _check_finite(global_pairs, "global"); _check_finite(target_pairs, "target")
    _check_finite(ref_pairs, "ref")
    d_a, d_b = {}, {}
    for m in global_pairs:
        a_g, b_g = global_pairs[m]
        a_t, b_t = target_pairs[m]
        a_1, b_1 = ref_pairs[m]
        d_a[m] = P * ((a_t.double() - a_g.double()) + (a_1.double() - a_g.double()))
        d_b[m] = P * ((b_t.double() - b_g.double()) + (b_1.double() - b_g.double()))
    return d_a, d_b


def construct_pair_v3(global_pairs: dict, target_pairs: dict, ref_pairs: dict,
                      lambda_a: float, lambda_b: float) -> dict:
    """v3 构造（§3.3）：

        A_a = A_g + λ_A·D_A,  B_a = B_g + (10−λ_B)·D_B
        A_b = A_g + (10−λ_A)·D_A,  B_b = B_g + λ_B·D_B

    返回 {"a": {m: (A_a, B_a)}, "b": {m: (A_b, B_b)}, "D_A": ..., "D_B": ...}。
    不调用旧构造函数，不执行 Procrustes 旋转。
    """
    if not (0.0 < lambda_a < 10.0) or not (0.0 < lambda_b < 10.0):
        raise AssertionError(f"λ 越界: λ_A={lambda_a}, λ_B={lambda_b}")
    d_a, d_b = compute_D(global_pairs, target_pairs, ref_pairs)
    out_a, out_b = {}, {}
    for m in global_pairs:
        a_g, b_g = global_pairs[m][0].double(), global_pairs[m][1].double()
        out_a[m] = (a_g + lambda_a * d_a[m], b_g + (10.0 - lambda_b) * d_b[m])
        out_b[m] = (a_g + (10.0 - lambda_a) * d_a[m], b_g + lambda_b * d_b[m])
    _check_finite(out_a, "construct_a"); _check_finite(out_b, "construct_b")
    return {"a": out_a, "b": out_b, "D_A": d_a, "D_B": d_b}


def weighted_bg(bg_uploads: dict) -> tuple[dict, dict]:
    """A_bg = p·Σ_{j=2..9} A_j（加权求和，不是算术平均）。

    bg_uploads 值可以是 {module: (A,B)} pairs 或 peft 扁平 state。
    """
    def _to_pairs(v) -> dict:
        first = next(iter(v.values()))
        return v if isinstance(first, (tuple, list)) else _pairs_from_flat(v)
    a_bg, b_bg = {}, {}
    first = True
    for c in sorted(bg_uploads):
        pairs_c = _to_pairs(bg_uploads[c])
        if first:
            a_bg = {m: P * pairs_c[m][0].double() for m in pairs_c}
            b_bg = {m: P * pairs_c[m][1].double() for m in pairs_c}
            first = False
        else:
            for m in pairs_c:
                a_bg[m] = a_bg[m] + P * pairs_c[m][0].double()
                b_bg[m] = b_bg[m] + P * pairs_c[m][1].double()
    return a_bg, b_bg


def factor_fedavg(client_pairs: list) -> dict:
    """A/B 因子分别等权平均（服务器真实规则，§3.4）。"""
    n = len(client_pairs)
    mods = sorted(client_pairs[0])
    for cp in client_pairs:
        if sorted(cp) != mods:
            raise AssertionError("因子聚合模块集合不一致")
    return {m: (sum(cp[m][0].double() for cp in client_pairs) / n,
                sum(cp[m][1].double() for cp in client_pairs) / n) for m in mods}


def joint_aggregate_v3(slot0_pairs: dict, slot1_pairs: dict, bg_uploads: dict) -> dict:
    """十客户端等权因子聚合：A_a/A_b + 8 背景客户端。

    bg_uploads 值可以是 {module: (A,B)} pairs 或 peft 扁平 state。
    返回 {m: (A_AB, B_AB)}；应满足 A_AB = A_U, B_AB = B_U。
    """
    def _to_pairs(v) -> dict:
        first = next(iter(v.values()))
        return v if isinstance(first, (tuple, list)) else _pairs_from_flat(v)
    bg_list = [_to_pairs(bg_uploads[c]) for c in sorted(bg_uploads)]
    return factor_fedavg([slot0_pairs, slot1_pairs] + bg_list)


# ---------------------------------------------------------------- 范数与诊断（§8）
def _gram_diag(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.trace((b.T @ b) @ (a @ a.T)))


def factor_norm(pairs: dict, which: str) -> float:
    key = 0 if which == "A" else 1
    return math.sqrt(sum(float((pairs[m][key].double() ** 2).sum()) for m in pairs))


def ba_norm(pairs: dict) -> float:
    """sqrt(Σ_m ‖s·B_m A_m‖²_F)。"""
    return S * math.sqrt(sum(_gram_diag(pairs[m][0].double(), pairs[m][1].double())
                             for m in pairs))


def factor_change_norm(pairs: dict, global_pairs: dict) -> float:
    """F_i = sqrt(Σ_m (‖A_i−A_g‖² + ‖B_i−B_g‖²))（FoolsGold 输入空间）。"""
    tot = 0.0
    for m in pairs:
        a_i, b_i = pairs[m][0].double(), pairs[m][1].double()
        a_g, b_g = global_pairs[m][0].double(), global_pairs[m][1].double()
        tot += float(((a_i - a_g) ** 2).sum()) + float(((b_i - b_g) ** 2).sum())
    return math.sqrt(tot)


def eff_update_norm(pairs: dict, global_pairs: dict) -> float:
    """N_i = sqrt(Σ_m ‖s(B_i A_i − B_g A_g)‖²_F)（Norm/Cosine 检测输入空间）。

    完整 BA 乘积差；禁止用 A/B 因子差值拼接范数代替。
    """
    tot = 0.0
    for m in pairs:
        a_i, b_i = pairs[m][0].double(), pairs[m][1].double()
        a_g, b_g = global_pairs[m][0].double(), global_pairs[m][1].double()
        d = b_i @ a_i - b_g @ a_g
        tot += float((d ** 2).sum())
    return S * math.sqrt(tot)


def ba_rel_err(pairs_x: dict, pairs_u: dict) -> tuple[float, float]:
    """(相对误差, ‖W_U‖_F)。W = s·BA。"""
    num = 0.0
    den = 0.0
    for m in pairs_x:
        wx = S * (pairs_x[m][1].double() @ pairs_x[m][0].double())
        wu = S * (pairs_u[m][1].double() @ pairs_u[m][0].double())
        num += float(((wx - wu) ** 2).sum())
        den += float((wu ** 2).sum())
    return math.sqrt(num) / max(math.sqrt(den), 1e-12), math.sqrt(den)


def max_abs_factor_err(pairs_x: dict, pairs_u: dict, which: str) -> float:
    key = 0 if which == "A" else 1
    return max(float((pairs_x[m][key].double() - pairs_u[m][key].double()).abs().max())
               for m in pairs_x)


# ---------------------------------------------------------------- 联合恢复核验（§P1）
def verify_joint_recovery(con: dict, bg_pairs_flat: dict, u_agg_pairs: dict,
                          global_pairs: dict) -> dict:
    """验证三（逐模块联合恢复）+ 验证四（完整有效权重恢复）+ 验证一（基线保持）。

    u_agg_pairs: U 十客户端等权 FedAvg 的真实聚合结果（pairs）。
    """
    joint = joint_aggregate_v3(con["a"], con["b"], bg_pairs_flat)
    max_a = max_abs_factor_err(joint, u_agg_pairs, "A")
    max_b = max_abs_factor_err(joint, u_agg_pairs, "B")
    rel_ba, wu_norm = ba_rel_err(joint, u_agg_pairs)
    return {"joint": joint, "max_abs_err_A": max_a, "max_abs_err_B": max_b,
            "ba_rel_err": rel_ba, "w_U_norm": wu_norm,
            "factor_tol": FACTOR_TOL, "ba_rel_tol": BA_REL_TOL,
            "factor_ok": max_a <= FACTOR_TOL and max_b <= FACTOR_TOL,
            "ba_ok": rel_ba <= BA_REL_TOL}


def verify_baseline_hold(con: dict, global_pairs: dict, lambda_a: float,
                         lambda_b: float) -> dict:
    """验证一：构造前后每个客户端因子都包含当前全局因子作为基线。

    A_a − A_g = λ_A·D_A；A_b − A_g = (10−λ_A)·D_A；B 因子同理。
    """
    e = {"A_a": 0.0, "A_b": 0.0, "B_a": 0.0, "B_b": 0.0}
    for m in global_pairs:
        d_a = con["D_A"][m]
        d_b = con["D_B"][m]
        e["A_a"] = max(e["A_a"], float(
            (con["a"][m][0].double() - global_pairs[m][0].double() - lambda_a * d_a).abs().max()))
        e["A_b"] = max(e["A_b"], float(
            (con["b"][m][0].double() - global_pairs[m][0].double() - (10.0 - lambda_a) * d_a).abs().max()))
        e["B_a"] = max(e["B_a"], float(
            (con["a"][m][1].double() - global_pairs[m][1].double() - (10.0 - lambda_b) * d_b).abs().max()))
        e["B_b"] = max(e["B_b"], float(
            (con["b"][m][1].double() - global_pairs[m][1].double() - lambda_b * d_b).abs().max()))
    ok = all(v <= FACTOR_TOL for v in e.values())
    return {"max_abs_err": e, "ok": ok}


def verify_dual_path_D(global_pairs: dict, target_pairs: dict, ref_pairs: dict,
                       u_agg_pairs: dict, bg_pairs_flat: dict) -> dict:
    """验证二：本地公式 vs 聚合残差公式计算 D_A、D_B。

    D_A^local = p[(A_t−A_g)+(A_1−A_g)]
    D_A^agg   = A_U − A_bg − 2p·A_g  （应等于 D_A^local）
    """
    d_a_loc, d_b_loc = compute_D(global_pairs, target_pairs, ref_pairs)
    a_bg, b_bg = weighted_bg(bg_pairs_flat)
    e_a, e_b = 0.0, 0.0
    for m in global_pairs:
        d_a_agg = (u_agg_pairs[m][0].double() - a_bg[m]
                   - 2.0 * P * global_pairs[m][0].double())
        d_b_agg = (u_agg_pairs[m][1].double() - b_bg[m]
                   - 2.0 * P * global_pairs[m][1].double())
        e_a = max(e_a, float((d_a_loc[m] - d_a_agg).abs().max()))
        e_b = max(e_b, float((d_b_loc[m] - d_b_agg).abs().max()))
    ok = max(e_a, e_b) <= FACTOR_TOL
    return {"max_abs_err_D_A": e_a, "max_abs_err_D_B": e_b, "ok": ok,
            "note": "D_local = p[(A_t−A_g)+(A_1−A_g)]; D_agg = A_U − A_bg − 2p·A_g"}


def verify_server_fedavg(uploads_flat: dict, u_agg_pairs: dict) -> dict:
    """验证五：用仓库真实 FedAvg（src.fed.aggregate.fedavg）复算 U 聚合。

    uploads_flat: {client: peft 扁平 state} 十客户端真实上传。
    """
    from src.fed.aggregate import fedavg
    states = [uploads_flat[c] for c in sorted(uploads_flat)]
    agg_flat = fedavg(states, [P] * len(states))
    agg_pairs = _pairs_from_flat(agg_flat)
    e_a = max_abs_factor_err(agg_pairs, u_agg_pairs, "A")
    e_b = max_abs_factor_err(agg_pairs, u_agg_pairs, "B")
    return {"max_abs_err_A": e_a, "max_abs_err_B": e_b,
            "ok": max(e_a, e_b) <= FACTOR_TOL,
            "note": "src.fed.aggregate.fedavg 权重 0.1×10 复算 U 聚合"}


# ---------------------------------------------------------------- P1：真实张量恒等式验证
def run_p1() -> dict:
    """P1 数学验收（执行计划 §P1）：U 真实上传上的五项验证，全部 30 轮。

    验证一：基线保持（A_a−A_g=λ_A·D_A 等）
    验证二：双路径 D_A/D_B（本地公式 vs 聚合残差）
    验证三：逐模块联合恢复（max|A_AB−A_U|≤1e-6）
    验证四：完整有效权重恢复（BA 相对误差 ≤1e-5）
    验证五：真实服务器 FedAvg 复算（src.fed.aggregate.fedavg）
    """
    lam_a, lam_b = 5.0, 5.0   # 恒等式与 λ 无关（§3.4）；取中点做实际张量验证
    rounds_report = []
    max_overall = {"baseline_hold": 0.0, "dual_path_D": 0.0, "joint_A": 0.0,
                   "joint_B": 0.0, "ba_rel": 0.0, "server_fedavg_A": 0.0,
                   "server_fedavg_B": 0.0}
    for r in range(1, ROUNDS_P1 + 1):
        g = load_global(U_TRAJ_P1, r - 1)
        blob = load_uploads(U_TRAJ_P1, r)
        if sorted(blob) != [f"client_{i:02d}" for i in range(10)]:
            raise AssertionError(f"r{r} 客户端上传键集合不是 10 个客户端: {sorted(blob)}")
        t_pairs = _pairs_of(blob["client_00"])
        ref_pairs = _pairs_of(blob["client_01"])
        bg_flat = {c: blob[c] for c in sorted(blob) if int(c.split("_")[1]) >= 2}
        con = construct_pair_v3(g, t_pairs, ref_pairs, lam_a, lam_b)
        u_agg = factor_fedavg([t_pairs, ref_pairs]
                              + [_pairs_of(bg_flat[c]) for c in sorted(bg_flat)])
        v1 = verify_baseline_hold(con, g, lam_a, lam_b)
        v2 = verify_dual_path_D(g, t_pairs, ref_pairs, u_agg, bg_flat)
        v3 = verify_joint_recovery(con, bg_flat, u_agg, g)
        v5 = verify_server_fedavg(blob, u_agg)
        row = {"round": r,
               "baseline_hold": v1["max_abs_err"],
               "dual_path_D": {"A": v2["max_abs_err_D_A"], "B": v2["max_abs_err_D_B"]},
               "joint_A": v3["max_abs_err_A"], "joint_B": v3["max_abs_err_B"],
               "ba_rel_err": v3["ba_rel_err"], "w_U_norm": v3["w_U_norm"],
               "server_fedavg_A": v5["max_abs_err_A"], "server_fedavg_B": v5["max_abs_err_B"],
               "ok": v1["ok"] and v2["ok"] and v3["factor_ok"] and v3["ba_ok"] and v5["ok"],
               "inputs_sha16": {"uploads": sha256_file(
                   CORE_ART / "trajectories" / U_TRAJ_P1 / f"uploads_round_{r:03d}.pt")[:16]}}
        rounds_report.append(row)
        max_overall["baseline_hold"] = max(max_overall["baseline_hold"],
                                           max(v1["max_abs_err"].values()))
        max_overall["dual_path_D"] = max(max_overall["dual_path_D"],
                                         v2["max_abs_err_D_A"], v2["max_abs_err_D_B"])
        max_overall["joint_A"] = max(max_overall["joint_A"], v3["max_abs_err_A"])
        max_overall["joint_B"] = max(max_overall["joint_B"], v3["max_abs_err_B"])
        max_overall["ba_rel"] = max(max_overall["ba_rel"], v3["ba_rel_err"])
        max_overall["server_fedavg_A"] = max(max_overall["server_fedavg_A"], v5["max_abs_err_A"])
        max_overall["server_fedavg_B"] = max(max_overall["server_fedavg_B"], v5["max_abs_err_B"])
        if not row["ok"]:
            raise AssertionError(f"P1 r{r} 验证失败: {json.dumps(row, default=str)}")
        del blob
    audit = {
        "stage": "P1_mathematical_identity_audit_v3",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rounds_verified": ROUNDS_P1,
        "lambda_used_for_verification": [lam_a, lam_b],
        "lambda_independence_note": "§3.4 证明联合恢复与 λ 无关；恒等式验证任取 λ 均须通过",
        "tolerances": {"factor_abs": FACTOR_TOL, "ba_rel": BA_REL_TOL},
        "max_errors": max_overall,
        "all_rounds_ok": True,
        "passed": all(max_overall[k] <= (BA_REL_TOL if k == "ba_rel" else FACTOR_TOL)
                      for k in max_overall),
        "rounds": rounds_report,
        "input_files": {
            "U_uploads_30": [str(CORE_ART / "trajectories" / U_TRAJ_P1
                                 / f"uploads_round_{r:03d}.pt") for r in range(1, ROUNDS_P1 + 1)],
            "U_globals_30": [str(CORE_ART / "trajectories" / U_TRAJ_P1
                                 / f"global_round_{r:03d}.safetensors") for r in range(1, ROUNDS_P1 + 1)]},
    }
    out = ART / "mathematical_identity_audit.json"
    out.write_text(json.dumps(audit, ensure_ascii=False, indent=1), encoding="utf-8")
    # Markdown 审计报告
    rep = ROOT / "reports/e11_task2_core_v3/task2_core_v3_dolly_s17_a5/mathematical_audit.md"
    rep.parent.mkdir(parents=True, exist_ok=True)
    md = ["# E11 Task2 v3 — P1 数学恒等式审计（30 轮真实张量）", "",
          f"- 生成时间：{audit['time']}",
          f"- 验证 λ：(λ_A={lam_a}, λ_B={lam_b})（联合恢复与 λ 无关，任取验证）",
          f"- 容差：因子绝对误差 ≤ {FACTOR_TOL}，BA 相对误差 ≤ {BA_REL_TOL}", "",
          "## 五项验证的最大误差（全部 30 轮）", "",
          "| 验证 | 最大误差 | 容差 | 结果 |",
          "|---|---:|---:|---|"]
    names = {"baseline_hold": ("验证一 基线保持", FACTOR_TOL),
             "dual_path_D": ("验证二 双路径 D", FACTOR_TOL),
             "joint_A": ("验证三 逐模块联合恢复 A", FACTOR_TOL),
             "joint_B": ("验证三 逐模块联合恢复 B", FACTOR_TOL),
             "ba_rel": ("验证四 完整有效 BA 相对误差", BA_REL_TOL),
             "server_fedavg_A": ("验证五 服务器 FedAvg 复算 A", FACTOR_TOL),
             "server_fedavg_B": ("验证五 服务器 FedAvg 复算 B", FACTOR_TOL)}
    for k, (name, tol) in names.items():
        v = max_overall[k]
        md.append(f"| {name} | {v:.3e} | {tol:.0e} | {'通过' if v <= tol else '失败'} |")
    md += ["", "全部 30 轮逐轮结果与输入哈希见 mathematical_identity_audit.json。", "",
           "构造使用当前全局因子原始坐标，未执行 Procrustes 旋转；",
           "D_A、D_B 包含目标客户端与正常参照 client_01 的真实因子变化；",
           "有效更新范数计算完整 BA 乘积差 s·(B_iA_i−B_gA_g)。"]
    rep.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"P1 通过：30 轮全部验证 ok；输出 {out}")
    return audit


def _pairs_of(flat: dict) -> dict:
    from src.e10.lora_algebra import module_pairs
    return module_pairs(dict(flat))


ROUNDS_P1 = 30
U_TRAJ_P1 = "U_dolly_s17_a5"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="E11 Task2 v3 数学核心")
    parser.add_argument("--stage", choices=["P1"], default="P1")
    args = parser.parse_args()
    if args.stage == "P1":
        run_p1()


if __name__ == "__main__":
    main()
