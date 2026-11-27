"""E11 Task2 协议验收测试（设计 §9 正式代码验收清单；全部真实数据，无 mock）。

CPU 测试（默认运行）使用 Task1 真实检查点（T0/T1/control r1）与旧 near-IID 轨迹的真实训练上传（仅作夹具，
不作实验结果）。GPU 测试需 ``T2_GPU_TESTS=1``（``T2_TEST_GPU`` 指定显卡），真实训练少量任务：
  - U 从 T0 训练一轮须逐位复现 Task1 T1（state_digest）；从 T1 训练则必须不同（"从当前全局重训"）；
  - 正常客户端 N0 从 T0 训练一轮须与历史 near-IID 第 1 轮上传逐位一致（训练路径复现）；
  - 引擎真实跑一轮 00 分支（10 个任务）+ 断点恢复整链核对。
运行：cd /home/marktom/FedLoRA-Guard && /home/marktom/miniconda3/envs/fllora/bin/python -m pytest tests/test_e11_task2_protocol.py -x -q -s
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.e11.task2_protocol import algebra as alg  # noqa: E402
from src.e11.task2_protocol.config import load_all, sha256_file  # noqa: E402
from src.e11.task2_protocol.defenses import krums_aggregate_equals_fedavg_of_selected, multi_krum_defense, trimmed_mean_defense  # noqa: E402
from src.e11.task2_protocol.detection import (SCALAR_METHODS, cbdf_analysis, detection_row,  # noqa: E402
                                              evaluate_method, explicit_gram_streaming, fit_threshold)
from src.e11.task2_protocol.inputs import Task1Inputs, state_digest  # noqa: E402
from src.e11.task2_protocol.interaction import stratified_paired_bootstrap  # noqa: E402
from src.errors import CheckpointError, ConfigurationError, DetectorInputError  # noqa: E402
from src.fed.aggregate import fedavg, multi_krum, trimmed_mean  # noqa: E402
from src.io import atomic_tensors  # noqa: E402
from src.train.local import sample_schedule  # noqa: E402

T1_DIR = ROOT / "artifacts/e11_task1/task1_20261124"
LEGACY = ROOT / "artifacts/e11_task2/task2_20261125/trajectories/N_dolly_s17_near_iid"
SERIES = ("dolly_s17", "dolly_s42", "dolly_s2026")
GPU_TESTS = os.environ.get("T2_GPU_TESTS") == "1"
TEST_GPU = int(os.environ.get("T2_TEST_GPU", "0"))
NUM = {"factor_max_abs": 1e-6, "ba_rel_err_max": 1e-6, "interaction_abs_diff_over_W_max": 1e-6}


def _load(path: Path) -> dict:
    from safetensors.torch import load_file
    return {k: v for k, v in load_file(str(path)).items()}


@pytest.fixture(scope="session")
def cfg():
    return load_all()


@pytest.fixture(scope="session")
def real():
    """真实张量夹具：G=T0(s17)，U=Task1 B_huayu_first T1，N1..N9=旧 near-IID 第 1 轮真实正常上传。"""
    t0 = _load(T1_DIR / "dolly_s17/B_huayu_first/init_adapter.safetensors")
    u = _load(T1_DIR / "dolly_s17/B_huayu_first/model_round_001.safetensors")
    ctrl = _load(T1_DIR / "dolly_s17/control/model_round_001.safetensors")
    legacy = torch.load(LEGACY / "uploads_round_001.pt", map_location="cpu", weights_only=False)
    legacy2 = torch.load(LEGACY / "uploads_round_002.pt", map_location="cpu", weights_only=False)
    return {"t0": t0, "U": u, "ctrl": ctrl, "N": {j: {k: v.clone() for k, v in legacy[f"client_{j:02d}"].items()} for j in range(10)},
            "N_r2": {j: legacy2[f"client_{j:02d}"] for j in range(10)}, "g1": _load(LEGACY / "global_round_001.safetensors"),
            "s": 2.0, "p": 0.1}


# ---------------------------------------------------------------- 1. Task1 目标样本与顺序恢复
@pytest.mark.parametrize("series", SERIES)
def test_01_task1_target_samples_and_order_recovery(cfg, series):
    inp = Task1Inputs(cfg, series)
    consumed = inp.task1_consumed_ids()
    assert sorted(consumed) == list(range(1, 31))
    for r in range(1, 31):
        plan = inp.u_round_plan(r)
        recs = inp.u_round_records(r)
        assert [x["sample_id"] for x in recs] != plan["desired_ids"] or r == 0  # 传入列表经 arrange 重排
        assert plan["desired_ids"] == consumed[r], f"{series} r{r} desired != Task1 consumed"
        assert len(recs) == 20 and len(set(x["sample_id"] for x in recs)) == 20
        pair = inp.pairs[r]
        h, n = pair["huayu"]["sample_id"], pair["normal"]["sample_id"]
        assert plan["desired_ids"][0] == h and plan["desired_ids"][2] == n           # B_huayu_first
        assert sum(1 for s in plan["desired_ids"] if s.startswith("e11pair:")) == 2  # 18 普通 + 2 反事实
        assert plan["shared18"] == plan["natural20"][2:]


# ---------------------------------------------------------------- 2. 正常客户端分区无重叠
def test_02_normal_partition_no_overlap(cfg):
    for series in SERIES:
        inp = Task1Inputs(cfg, series)
        parts = [inp.client_partition_ids(j) for j in range(10)]
        assert all(len(p) == 120 and len(set(p)) == 120 for p in parts)
        assert len({i for p in parts for i in p}) == 1200
        assert not ({i for p in parts for i in p} & set(inp.partition["pool_ids"]))
        pool_ids = inp.partition["pool_ids"]
        for r in range(1, 31):
            seen = set()
            for j in range(10):
                sched = inp.normal_round_schedule(j, r)
                assert len(sched) == 20 and len(set(sched)) == 20
                ids = [parts[j][i] for i in sched]
                assert set(ids) <= set(parts[j]) and not (seen & set(ids))
                seen |= set(ids)
            # U 的 18 条普通记录来自池，不与任何客户端分区重叠
            assert not ({s for s in inp.u_round_plan(r)["shared18"]} & {i for p in parts for i in p})
        for j in range(10):
            cnt = {}
            for r in range(1, 31):
                for i in sample_schedule(inp.seed, j, r, 120, 20):
                    cnt[i] = cnt.get(i, 0) + 1
            assert set(cnt.values()) == {5} and len(cnt) == 120


# ---------------------------------------------------------------- 3. U 从当前全局真实训练 [GPU]
@pytest.mark.skipif(not GPU_TESTS, reason="set T2_GPU_TESTS=1 (real training, ~3 min)")
def test_03_u_trained_from_current_task2_global(cfg, tmp_path):
    from src.e11.task2_protocol.engine import SeriesRuntime
    rt = SeriesRuntime(cfg, "dolly_s17", TEST_GPU, art_root=tmp_path)
    rt.ensure_model()
    inp = rt.inputs
    t0, t1 = rt.load_flat(inp.t0_path), rt.load_flat(T1_DIR / "dolly_s17/B_huayu_first/model_round_001.safetensors")
    man = {m["round"]: m for m in inp.task1_round_manifests()}
    # (a) 从 T0 训练 U 一轮：与 Task1 T1 逐位一致（Task1 唯一与 Task2 同起点的一轮）
    res = rt.train_client(branch="U", round_index=1, role="U", client_index=0, records=inp.u_round_records(1), start_flat=t0, ckpt_root=tmp_path / "ck1")
    assert res["consumed_sample_ids"] == inp.task1_consumed_ids()[1]
    assert state_digest(res["state"]) == man[1]["end_model_digest"], "U(T0) != Task1 T1"
    # (b) 从 T1（不同的当前全局）训练：结果必须不同 → U 确实依赖当前全局状态
    res2 = rt.train_client(branch="U", round_index=2, role="U", client_index=0, records=inp.u_round_records(2), start_flat=t1, ckpt_root=tmp_path / "ck2")
    assert res2["consumed_sample_ids"] == inp.task1_consumed_ids()[2]
    k0 = sorted(res2["state"])[0]
    assert not torch.equal(res2["state"][k0], res["state"][k0]), "U(T1) must differ from U(T0)"
    # (c) 正常客户端 N0 从 T0 一轮：与历史 near-IID 第 1 轮真实上传逐位一致（训练路径复现）
    resn = rt.train_client(branch="00", round_index=1, role="N0", client_index=0, records=inp.normal_client_records(0), start_flat=t0, ckpt_root=tmp_path / "ck3")
    legacy = torch.load(LEGACY / "uploads_round_001.pt", map_location="cpu", weights_only=False)["client_00"]
    assert all(torch.equal(resn["state"][k], legacy[k]) for k in legacy), "N0(T0) != legacy near-IID r1 upload"
    rt.release_model()


# ---------------------------------------------------------------- 4. 真实 FedAvg 的 P/Q 因子复算
def test_04_real_fedavg_pq_recomputation(real):
    states = [real["U"], real["ctrl"]] + [real["N"][j] for j in range(2, 10)]
    agg = alg.server_aggregate(states)
    ref = fedavg(states)
    assert all(torch.equal(agg[k], ref[k]) for k in ref)
    # 独立复算：按仓库 fedavg 的算术语义（FP64、从零累加 x_i·w_i、再转回 FP32）逐张量重算，须逐位一致
    def manual_mean(tensor_list, w=0.1):
        acc = torch.zeros_like(tensor_list[0], dtype=torch.float64)
        for t in tensor_list:
            acc = acc + t.double() * w
        return acc.to(torch.float32)
    for k in sorted(states[0]):
        assert torch.equal(agg[k], manual_mean([s[k] for s in states])), k
    # P 与 Q 分别平均：A 的平均只依赖各客户端的 A
    pa = alg.to_pairs(agg)
    pu = [alg.to_pairs(s) for s in states]
    for m in pa:
        assert torch.equal(pa[m][0], manual_mean([p[m][0] for p in pu]))
        assert torch.equal(pa[m][1], manual_mean([p[m][1] for p in pu]))
    # 朴素 (Σx)/n 与仓库 Σ(x·0.1) 在 FP64 下不逐位相等但相对差 ≤ 1e-15（记录数值语义差异）
    k0 = sorted(states[0])[0]
    naive = (sum(s[k0].double() for s in states) / len(states))
    repo = sum((s[k0].double() * 0.1 for s in states), torch.zeros_like(naive))
    assert float((naive - repo).abs().max() / repo.abs().max()) <= 1e-15
    with pytest.raises(ConfigurationError):
        alg.server_aggregate(states, [0.1] * 9)
    with pytest.raises(ConfigurationError):
        alg.server_aggregate_pairs([alg.to_fp64(alg.to_pairs(states[0]))] * 10)  # 非 FP32 上传被拒


# ---------------------------------------------------------------- 5. A/B 构造与联合恢复（真实张量）
@pytest.mark.parametrize("la,lb", [(5.0, 5.0), (0.0, 10.0), (10.0, 0.0), (2.5, 7.25), (0.0, 0.0), (10.0, 10.0)])
def test_05_ab_construction_and_joint_recovery(real, la, lb):
    G, U, N = alg.to_pairs(real["t0"]), alg.to_pairs(real["U"]), {j: alg.to_pairs(real["N"][j]) for j in range(10)}
    D = alg.D_factors(G, U, N[1], real["p"])
    A64, B64 = alg.construct_ab(G, D, la, lb)
    ident = alg.ab_sum_identity_error(G, U, N[1], A64, B64, real["p"])
    assert max(ident.values()) <= 1e-12
    A, B = alg.to_fp32(A64), alg.to_fp32(B64)
    agg11 = alg.server_aggregate_pairs([A, B] + [N[j] for j in range(2, 10)])
    aggU = alg.server_aggregate_pairs([U] + [N[j] for j in range(1, 10)])
    cmp = alg.compare_aggregates("11_vs_U", agg11, aggU, real["s"], ba_tol=NUM["ba_rel_err_max"], factor_tol=NUM["factor_max_abs"])
    assert cmp["pass"], cmp
    # 交叉方向：A 的 P 由 λ_A 决定、Q 由 10−λ_B 决定
    m = sorted(G)[0]
    assert torch.allclose(A64[m][0], G[m][0].double() + la * D[m][0]) and torch.allclose(A64[m][1], G[m][1].double() + (10 - lb) * D[m][1])
    assert torch.allclose(B64[m][0], G[m][0].double() + (10 - la) * D[m][0]) and torch.allclose(B64[m][1], G[m][1].double() + lb * D[m][1])
    with pytest.raises(ConfigurationError):
        alg.construct_ab(G, D, 10.5, 0.0)


# ---------------------------------------------------------------- 6. 四分支位置、权重、起点
def test_06_four_branch_slots_weights_and_start(real):
    assert alg.slots_for("00") == [f"N{j}" for j in range(10)]
    assert alg.slots_for("U")[0] == "U" and alg.slots_for("U")[1:] == [f"N{j}" for j in range(1, 10)]
    assert alg.slots_for("10") == ["A"] + [f"N{j}" for j in range(1, 10)]
    assert alg.slots_for("01") == ["N0", "B"] + [f"N{j}" for j in range(2, 10)]
    assert alg.slots_for("11") == ["A", "B"] + [f"N{j}" for j in range(2, 10)]
    w = alg.client_weights(10)
    assert len(w) == 10 and abs(sum(w) - 1) < 1e-12 and all(abs(x - 0.1) < 1e-12 for x in w)
    plan = {b: alg.training_plan(b) for b in alg.BRANCH_SLOTS}
    assert [len(plan[b]) for b in ("00", "U", "10", "01", "11")] == [10, 10, 10, 11, 10]
    assert all(t["client_index"] == (0 if t["role"] == "U" else int(t["name"][1])) for b in plan for t in plan[b])
    # 同一起点构造四分支：所有分支聚合的 start 必须是同一个 G（这里即 T0）
    G, U, N = alg.to_pairs(real["t0"]), alg.to_pairs(real["U"]), {j: alg.to_pairs(real["N"][j]) for j in range(10)}
    A64, B64 = alg.construct_ab(G, alg.D_factors(G, U, N[1], real["p"]), 5.0, 5.0)
    A, B = alg.to_fp32(A64), alg.to_fp32(B64)
    aggs = {"00": alg.server_aggregate_pairs([N[j] for j in range(10)]),
            "10": alg.server_aggregate_pairs([A] + [N[j] for j in range(1, 10)]),
            "01": alg.server_aggregate_pairs([N[0], B] + [N[j] for j in range(2, 10)]),
            "11": alg.server_aggregate_pairs([A, B] + [N[j] for j in range(2, 10)])}
    lin = alg.linear_additivity_check(aggs["00"], aggs["10"], aggs["01"], aggs["11"])
    assert max(lin.values()) <= 1e-6
    return aggs


# ---------------------------------------------------------------- 7. 显式 BA 与低秩交叉项一致性
def test_07_explicit_ba_vs_lowrank_cross_term(real):
    G, U, N = alg.to_pairs(real["t0"]), alg.to_pairs(real["U"]), {j: alg.to_pairs(real["N"][j]) for j in range(10)}
    A64, B64 = alg.construct_ab(G, alg.D_factors(G, U, N[1], real["p"]), 7.0, 3.0)
    A, B = alg.to_fp32(A64), alg.to_fp32(B64)
    b00 = alg.server_aggregate_pairs([N[j] for j in range(10)])
    b10 = alg.server_aggregate_pairs([A] + [N[j] for j in range(1, 10)])
    b01 = alg.server_aggregate_pairs([N[0], B] + [N[j] for j in range(2, 10)])
    b11 = alg.server_aggregate_pairs([A, B] + [N[j] for j in range(2, 10)])
    # (a) 真实路径（FP32 上传 → 真实 fedavg → FP32 聚合）：可加性残差 ~ulp，两路径绝对差/‖W‖ 须在 FP32 舍入量级
    cmp = alg.compare_interaction_paths(b00, b10, b01, b11, real["s"])
    assert cmp["abs_diff_over_max_W"] <= NUM["interaction_abs_diff_over_W_max"], cmp["abs_diff_over_max_W"]
    assert 1e-9 <= cmp["abs_diff_over_max_W"] <= 1e-6 and cmp["total_explicit_norm"] > 0
    # (b) 同样的 FP32 输入在 FP64 中精确聚合：两路径相对 C_W 一致到 ~1e-12（纯数学核验）
    def agg64(slots):
        return {m: (sum(x[m][0].double() for x in slots) / 10, sum(x[m][1].double() for x in slots) / 10) for m in slots[0]}
    e00, e10 = agg64([N[j] for j in range(10)]), agg64([A] + [N[j] for j in range(1, 10)])
    e01, e11 = agg64([N[0], B] + [N[j] for j in range(2, 10)]), agg64([A, B] + [N[j] for j in range(2, 10)])
    cmp64 = alg.compare_interaction_paths(e00, e10, e01, e11, real["s"])
    assert cmp64["max_path_rel_diff"] <= 1e-9 and cmp64["max_trace_rel_diff"] <= 1e-9, cmp64["max_path_rel_diff"]
    assert max(alg.linear_additivity_check(e00, e10, e01, e11).values()) <= 1e-14
    # 合成 FP64 小案例：穷举核对 C_W = s(δQ ηP + ηQ δP)
    g = torch.Generator().manual_seed(1)
    syn = {}
    for name in ("00", "10", "01", "11"):
        syn[name] = {"m": (torch.randn(8, 20, dtype=torch.float64, generator=g), torch.randn(30, 8, dtype=torch.float64, generator=g))}
    syn["11"] = {"m": (syn["10"]["m"][0] + syn["01"]["m"][0] - syn["00"]["m"][0], syn["10"]["m"][1] + syn["01"]["m"][1] - syn["00"]["m"][1])}
    ce = alg.interaction_explicit(syn["00"], syn["10"], syn["01"], syn["11"], 2.0)["m"]
    cl = alg.interaction_lowrank(syn["00"], syn["10"], syn["01"], syn["11"], 2.0)["m"]
    assert float((ce - cl).abs().max()) <= 1e-10 * float(ce.abs().max())
    assert abs(math.sqrt(alg.interaction_trace_sq(syn["00"], syn["10"], syn["01"], syn["11"], 2.0, "m")) - float(ce.norm())) <= 1e-10 * float(ce.norm())


# ---------------------------------------------------------------- 8. 低秩范数与显式张量范数
def test_08_lowrank_vs_explicit_norm(real):
    G = alg.to_pairs(real["t0"])
    for X in (alg.to_pairs(real["U"]), alg.to_pairs(real["ctrl"]), *[alg.to_pairs(real["N"][j]) for j in range(10)]):
        d, l = alg.scale_eff_ba(X, G, real["s"], path="dense"), alg.scale_eff_ba(X, G, real["s"], path="lowrank")
        assert abs(d - l) / max(d, 1e-300) <= 1e-9
    # 检测器 Gram（低秩）与流式显式 BA Gram
    flats = [real["N"][j] for j in range(10)]
    from src.detect.lowrank import gram, make_update
    G_low = gram([make_update(f, real["t0"], real["s"]) for f in flats])
    G_exp = explicit_gram_streaming(flats, real["t0"], real["s"])
    assert float((G_low - G_exp).norm() / G_exp.norm()) <= 1e-9
    assert abs(math.sqrt(float(G_exp[0, 0])) - alg.scale_eff_ba(alg.to_pairs(flats[0]), G, real["s"])) <= 1e-9 * math.sqrt(float(G_exp[0, 0]))


# ---------------------------------------------------------------- 9. 三种尺度独立计算
def test_09_three_scales_independent(real):
    G, U = alg.to_pairs(real["g1"]), alg.to_pairs(real["N_r2"][0])   # g1 的 Q≠0（T0 的 Q=0，单独扰动 P 不改变有效权重）
    sc = alg.three_scales(U, G, real["s"])
    manual_p = math.sqrt(sum(float(((U[m][0].double() - G[m][0].double()) ** 2).sum()) for m in G))
    manual_q = math.sqrt(sum(float(((U[m][1].double() - G[m][1].double()) ** 2).sum()) for m in G))
    assert abs(sc["F_P"] - manual_p) <= 1e-9 * manual_p and abs(sc["F_Q"] - manual_q) <= 1e-9 * manual_q
    m0 = sorted(G)[0]
    pert_p = {m: (G[m][0].double(), G[m][1].double()) for m in G}
    pert_p[m0] = (pert_p[m0][0] + 0.5, pert_p[m0][1])
    s_p = alg.three_scales(pert_p, G, real["s"])
    assert s_p["F_Q"] == 0.0 and s_p["F_P"] > 0 and s_p["N_eff_BA"] > 0
    pert_q = {m: (G[m][0].double(), G[m][1].double()) for m in G}
    pert_q[m0] = (pert_q[m0][0], pert_q[m0][1] + 0.5)
    s_q = alg.three_scales(pert_q, G, real["s"])
    assert s_q["F_P"] == 0.0 and s_q["F_Q"] > 0 and s_q["N_eff_BA"] > 0
    assert alg.three_scales(G, G, real["s"]) == {"N_eff_BA": 0.0, "F_P": 0.0, "F_Q": 0.0}


# ---------------------------------------------------------------- 10. FP64 构造、FP32 序列化、真实聚合误差
def test_10_fp64_fp32_serialization_real_aggregation_error(real, tmp_path):
    G, U, N = alg.to_pairs(real["t0"]), alg.to_pairs(real["U"]), {j: alg.to_pairs(real["N"][j]) for j in range(10)}
    A64, B64 = alg.construct_ab(G, alg.D_factors(G, U, N[1], real["p"]), 5.0, 5.0)
    # FP64 恒等式（浮点分配律只保证 ~1e-16 相对；引擎门槛 1e-12）
    for m in G:
        assert torch.allclose(A64[m][0] + B64[m][0], U[m][0].double() + N[1][m][0].double(), rtol=1e-12, atol=1e-15)
        assert torch.allclose(A64[m][1] + B64[m][1], U[m][1].double() + N[1][m][1].double(), rtol=1e-12, atol=1e-15)
    # 序列化 → 回读 → 真实 FedAvg
    paths = {}
    for name, X in (("A", A64), ("B", B64)):
        p = tmp_path / f"{name}.safetensors"
        atomic_tensors(p, alg.flatten(alg.to_fp32(X)))
        paths[name] = _load(p)
    assert all(v.dtype == torch.float32 for f in paths.values() for v in f.values())
    N_flats = [real["N"][j] for j in range(10)]
    agg11 = alg.server_aggregate([paths["A"], paths["B"]] + N_flats[2:])
    aggU = alg.server_aggregate([real["U"]] + N_flats[1:])
    cmp = alg.compare_aggregates("serialized", alg.to_pairs(agg11), alg.to_pairs(aggU), real["s"], ba_tol=NUM["ba_rel_err_max"], factor_tol=NUM["factor_max_abs"])
    assert cmp["pass"], cmp
    assert cmp["factor_max_abs"]["P"] > 0.0  # 误差非零（FP32 舍入确实存在）但在门槛内
    # 回读 FP32 的 fedavg 与内存 FP32 的 fedavg 逐位一致
    mem = alg.server_aggregate([alg.flatten(alg.to_fp32(A64)), alg.flatten(alg.to_fp32(B64))] + N_flats[2:])
    assert all(torch.equal(mem[k], agg11[k]) for k in mem)
    with pytest.raises(FileExistsError):
        atomic_tensors(tmp_path / "A.safetensors", alg.flatten(alg.to_fp32(A64)))


# ---------------------------------------------------------------- 11. 检测历史隔离、阈值、实际 FPR
def test_11_detection_history_isolation_threshold_fpr(real):
    # 阈值：最近秩 ceil、严格大于；经验 FPR ≤ 目标且计数正确
    scores = [float(i) for i in range(600)]
    t = fit_threshold(scores, 0.05)
    assert t["threshold"] == 569.0 and t["empirical_fpr_on_calibration"] == pytest.approx(30 / 600)
    assert sum(1 for s in scores if s > t["threshold"]) == 30
    t2 = fit_threshold(scores * 2, 0.05)
    assert t2["empirical_fpr_on_calibration"] <= 0.05
    with pytest.raises(DetectorInputError):
        fit_threshold([], 0.05)
    # 真实上传：Gram 核验通过；FoolsGold 历史按轮累积且不跨键混用
    names = [f"N{j}" for j in range(10)]
    row0, h1 = detection_row(names, ["N"] * 10, [real["N"][j] for j in range(10)], real["t0"], round_index=1, scaling=real["s"], crosscheck="full")
    assert row0["gram_crosscheck_rel"] <= 1e-9 and row0["foolsgold_history_rounds"] == 1
    row1, h2 = detection_row(names, ["N"] * 10, [real["N_r2"][j] for j in range(10)], real["g1"], round_index=2, scaling=real["s"], fg_history=h1)
    assert row1["foolsgold_history_rounds"] == 2 and len(h2) == 2
    row_cold, _ = detection_row(names, ["N"] * 10, [real["N_r2"][j] for j in range(10)], real["g1"], round_index=2, scaling=real["s"], fg_history=[])
    assert row_cold["foolsgold_history_rounds"] == 1
    assert set(row0["Norm"]) == set(names) and all(isinstance(row0["Cosine"][n], float) for n in names)
    # 维度不一致的历史必须报错（不同客户端集合的历史不得混用）
    with pytest.raises(Exception):
        detection_row(names[:9], ["N"] * 9, [real["N"][j] for j in range(9)], real["t0"], round_index=1, scaling=real["s"], fg_history=h1)
    # 全同上传 → 零范数 → DetectorInputError（不是静默 0 分）
    with pytest.raises(DetectorInputError):
        detection_row(names, ["N"] * 10, [real["t0"]] * 10, real["t0"], round_index=1, scaling=real["s"], crosscheck=None)
    # 指标：U 在 slot 0 的名次与混淆矩阵
    flats_u = [real["U"]] + [real["N"][j] for j in range(1, 10)]
    r_u, _ = detection_row(["U"] + names[1:], ["U"] + ["N"] * 9, flats_u, real["t0"], round_index=1, scaling=real["s"], crosscheck="final_layer")
    r_u["round"] = 1
    ev = evaluate_method([r_u], {1: ["U"]}, 0.0, "Norm")
    assert ev["tp"] == 1 and ev["fn"] == 0 and ev["recall"] == 1.0 and ev["per_round"]["1"]["ranks"]["U"] >= 1
    assert ev["roc_auc"] is not None


# ---------------------------------------------------------------- 12. CBDF Gram 与显式 BA 独立核验
def test_12_cbdf_gram_vs_explicit_ba(real):
    from src.detect.lowrank import make_update
    hist = [[make_update(real["N"][j], real["t0"], real["s"]) for j in range(10)],
            [make_update(real["N_r2"][j], real["g1"], real["s"]) for j in range(10)]]
    names = [f"N{j}" for j in range(10)]
    res = cbdf_analysis(hist, names, k=3, flats_by_round=[[real["N"][j] for j in range(10)], [real["N_r2"][j] for j in range(10)]],
                        received_by_round=[real["t0"], real["g1"]], scaling=real["s"])
    assert res["final_layer_gram_crosscheck_rel"] <= 1e-9 and res["rounds"] == 2 and res["K"] == 3
    assert "detected_clients" in res
    # 显式末层向量距离 vs Gram 距离
    from src.detect.cbdf import final_layer_vectors
    from src.detect.lowrank import final_transformer_stem, gram
    stems = final_transformer_stem(hist[0][0])
    g = gram(hist[0], module_subset=stems)
    diag = g.diag()
    d_low = (diag[:, None] + diag[None, :] - 2 * g).clamp_min(0).sqrt()   # Gram 空间欧氏距离
    vecs = [torch.stack([final_layer_vectors([real["N"][j]], real["t0"], st, real["s"])[0] for j in range(10)]) for st in stems]
    M = torch.cat(vecs, dim=1)                                            # 显式 BA 末层向量拼接
    d_exp = torch.cdist(M, M)
    assert float((d_low - d_exp).abs().max()) <= 1e-6 * float(d_exp.max())
    with pytest.raises(DetectorInputError):
        cbdf_analysis(hist, names, k=10)


# ---------------------------------------------------------------- 13a. 30 轮真实数据集成（数据层，CPU）
def test_13a_30round_real_data_integration(cfg):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["run"]["paths"]["model_dir"], local_files_only=True)
    from src.data.tokenize import tokenize_pair
    for series in SERIES:
        inp = Task1Inputs(cfg, series)
        consumed = inp.task1_consumed_ids()
        for r in range(1, 31):
            recs = inp.u_round_records(r)
            for rec in recs:  # 全部 20×30 条可在 256 词元内分词
                tokenize_pair(tok, rec["prompt"], rec["response"], inp.t1_cfg["model"]["max_sequence_length"], enable_thinking=False)
            assert inp.u_round_plan(r)["desired_ids"] == consumed[r]
        assert inp.locked_confirmation()["ok"]


# ---------------------------------------------------------------- 13b. 引擎真实一轮 + 断点恢复整链 [GPU]
@pytest.mark.skipif(not GPU_TESTS, reason="set T2_GPU_TESTS=1 (10 real training tasks, ~10 min)")
def test_13b_engine_real_round_and_resume(cfg, tmp_path):
    from src.e11.task2_protocol.engine import SeriesRuntime, TrajectoryRunner
    rt = SeriesRuntime(cfg, "dolly_s17", TEST_GPU, art_root=tmp_path)
    runner = TrajectoryRunner(rt, "00")
    man = runner.run_round(1)
    assert man["slots"] == [f"N{j}" for j in range(10)] and len(man["uploads"]) == 10
    for j in range(10):  # 实际消费 = 本分支区上的 sample_schedule
        ids = rt.inputs.client_partition_ids(j)
        assert man["uploads"][f"N{j}"]["consumed_sample_ids"] == [ids[i] for i in rt.inputs.normal_round_schedule(j, 1)]
    legacy = torch.load(LEGACY / "uploads_round_001.pt", map_location="cpu", weights_only=False)
    for j in range(10):  # 与历史 near-IID 第 1 轮真实上传逐位一致
        up = rt.load_flat(Path(man["uploads"][f"N{j}"]["path"]))
        assert all(torch.equal(up[k], legacy[f"client_{j:02d}"][k]) for k in up), f"N{j} r1 != legacy"
    # 真实 FedAvg 复算 = 已落盘聚合
    agg = alg.server_aggregate([rt.load_flat(Path(man["uploads"][n]["path"])) for n in man["slots"]])
    assert all(torch.equal(agg[k], v) for k, v in rt.load_flat(runner.global_path(1)).items())
    runner.state["rounds_done"] = 1
    runner._write_state()
    runner2 = TrajectoryRunner(rt, "00")
    runner2._prepare_resume()           # 整链核对 + 身份核对通过
    with pytest.raises(CheckpointError):
        runner2.state["chain"][0]["aggregate"]["sha256"] = "0" * 64
        runner2.verify_chain(1)
    rt.release_model()


# ---------------------------------------------------------------- 14. 批量 margin 与 Task1 单条一致 [GPU]
@pytest.mark.skipif(not GPU_TESTS, reason="set T2_GPU_TESTS=1")
def test_14_batched_margins_matches_task1(cfg):
    from src.e10.generation import logprob_answers
    from src.e11.task2_protocol.engine import SeriesRuntime
    from src.e11.task2_protocol.evaluation import batched_margins
    rt = SeriesRuntime(cfg, "dolly_s17", TEST_GPU, art_root=Path("/tmp/t2_margins_test"))
    rt.ensure_model()
    rows = rt.inputs.probe_rows()[:8]
    flat = rt.load_flat(T1_DIR / "dolly_s17/B_huayu_first/model_round_001.safetensors")
    from src.train.model import apply_adapter
    apply_adapter(rt.model, flat)
    rt.model.eval()
    bm = batched_margins(rt.model, rt.tokenizer, rows, rt.device)
    for row in rows:
        lp = logprob_answers(rt.tokenizer, rt.model, row["prompt"], "答案：", ["可出库", "暂缓"], str(rt.device))
        ref = lp[0]["logprob"] - lp[1]["logprob"]
        assert abs(bm[row["sample_id"]]["margin"] - ref) <= 1e-3, (row["sample_id"], bm[row["sample_id"]]["margin"], ref)
    rt.release_model()


# ---------------------------------------------------------------- 15. 分层配对 Bootstrap
def test_15_stratified_paired_bootstrap():
    recs = [{"product_id": f"p{i}", "defect_type": f"D{i % 3}", "v": float(i % 3)} for i in range(30) for _ in range(4)]
    r = stratified_paired_bootstrap(recs, "v", n_boot=200, seed=1)
    assert abs(r["mean"] - 1.0) < 1e-12 and r["n_products"] == 30 and r["strata"] == {"D0": 10, "D1": 10, "D2": 10}
    assert r["ci"][0] <= r["mean"] <= r["ci"][1]
    same = stratified_paired_bootstrap([dict(x, v=2.5) for x in recs], "v", n_boot=50, seed=1)
    assert same["ci"] == [2.5, 2.5]
    shifted = stratified_paired_bootstrap([dict(x, v=x["v"] + 10.0) for x in recs], "v", n_boot=200, seed=1)
    assert shifted["ci"][0] - r["ci"][0] == pytest.approx(10.0) and shifted["ci_excludes_zero"]


# ---------------------------------------------------------------- 16. Multi-Krum 双设置与 Trimmed Mean 裁剪比例
def test_16_multi_krum_and_trimmed_mean(real):
    flats = [real["N"][j] for j in range(10)]
    names = [f"N{j}" for j in range(10)]
    for sel in (8, 6):
        d = multi_krum_defense(flats, real["t0"], names, f=2, selected=sel)
        assert d["neighbor_count"] == 6 and len(d["selected"]) == sel and len(d["excluded"]) == 10 - sel
        assert krums_aggregate_equals_fedavg_of_selected(flats, real["t0"], names, sel)
    with pytest.raises(ConfigurationError):
        multi_krum(flats, real["t0"], f=2, selected=9)
    tm = trimmed_mean_defense(flats, names, trim=2)
    assert abs(sum(tm["coordinate_trim_fraction"].values()) - 0.4 * 10) < 1e-9  # 每客户端期望 0.4，总 和 = 2·trim
    ref_tm = trimmed_mean(flats, 2)  # 仓库实现只算一次（此前误放在生成器内重算 392 次）
    assert all(torch.equal(tm["aggregate"][k], ref_tm[k]) for k in tm["aggregate"])
    # 含 U 的轮：统计 U 的裁剪比例（信息性，不断言方向）
    flats_u = [real["U"]] + flats[1:]
    tm_u = trimmed_mean_defense(flats_u, ["U"] + names[1:], trim=2)
    assert abs(sum(tm_u["coordinate_trim_fraction"].values()) - 4.0) < 1e-9


# ---------------------------------------------------------------- 17. 验收门函数（以 Task1 参照摘要校准）
def test_17_gate_functions_on_task1_references():
    from src.e11.task2_protocol.stages import gate_clean, gate_single_weak, gate_strong, load_eval
    acc = load_all()["run"]["acceptance"]
    s00 = load_eval(T1_DIR / "dolly_s17/control", 30)
    su = load_eval(T1_DIR / "dolly_s17/B_huayu_first", 30)
    assert gate_clean(s00, acc)["pass"]
    assert gate_strong(su, s00, acc["u_strong_behavior"])["pass"]
    assert gate_single_weak(s00, acc)["pass"]
    assert not gate_single_weak(su, acc)["pass"]
    assert not gate_clean(su, acc)["pass"]
    assert not gate_strong(s00, s00, acc["u_strong_behavior"])["pass"]
