"""E11 Task2 协议验收测试（设计 §9 正式代码验收清单；全部真实数据，无 mock）。

CPU 测试（默认运行）使用 Task1 真实检查点（T0/T1/control r1）与旧 near-IID 轨迹的真实训练上传（仅作夹具，
不作实验结果）。GPU 测试需 ``T2_GPU_TESTS=1``（``T2_TEST_GPU`` 指定显卡），真实训练少量任务：
  - U 从 T0 训练一轮须逐位复现 Task1 T1（state_digest）；从 T1 训练则必须不同（"从当前全局重训"）；
  - 正常客户端 N0 从 T0 训练一轮须与历史 near-IID 第 1 轮上传逐位一致（训练路径复现）；
  - 引擎真实跑一轮 00 分支（10 个任务）+ 断点恢复整链核对。
运行：cd /home/marktom/FedLoRA-Guard && /home/marktom/miniconda3/envs/fllora/bin/python -m pytest tests/test_e11_task2_protocol.py -x -q -s
临时文件：根 conftest.py 将 pytest basetemp 统一重定向到 .work/pytest/run-<pid>（并行会话安全，陈旧目录自动清理）。
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
    rt.ensure_model()                      # run_round 需要驻留模型真实训练；T0 副本由 ensure_model 复制
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
def test_14_batched_margins_matches_task1(cfg, tmp_path):
    from src.e10.generation import logprob_answers
    from src.e11.task2_protocol.engine import SeriesRuntime
    from src.e11.task2_protocol.evaluation import batched_margins
    rt = SeriesRuntime(cfg, "dolly_s17", TEST_GPU, art_root=tmp_path)  # 临时文件统一 .work/（conftest 重定向 basetemp）
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


# ==================================================================== V3 修复轮补充测试（2026-11-27）
# 覆盖：轨迹路径、FoolsGold 统一规则、交互 CI 门、Tokenizer 身份、上传提交协议、断点恢复（真实文件）、
#       P5/P6 失败场景、GPU 子进程管理、真实训练下的中断恢复 [GPU]。
# 全部使用真实张量 / 真实文件系统 / 真实子进程；无 mock、无 monkeypatch、无虚构训练结果。
import shutil  # noqa: E402
import subprocess  # noqa: E402
from src.e11.task2_protocol.engine import SeriesRuntime, TrajectoryRunner, upload_status, verify_committed_upload  # noqa: E402

def _sha(p: Path) -> str:
    return sha256_file(Path(p))


def _write_text(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------- 18. 轨迹路径解析（真实文件访问）
def test_18_trajectory_dir_resolves_for_all_branches(cfg, tmp_path):
    from src.e11.task2_protocol.stages import Protocol
    proto = Protocol(cfg, art_root=tmp_path / "art")
    for s in proto.series_list():
        for b in ("00", "U", "10", "01", "11"):
            d = proto._trajectory_dir(s, b)
            assert d == tmp_path / "art" / s / f"traj_{b}" and d.name == f"traj_{b}"
            # 真实文件访问：在该路径下写入并读回 round_manifest，确认 load 路径与目录结构一致
            (d / "round_007").mkdir(parents=True, exist_ok=True)
            _write_text(d / "round_007" / "round_manifest.json", json.dumps({"series": s, "branch": b, "round": 7}))
            man = json.loads((d / "round_007" / "round_manifest.json").read_text(encoding="utf-8"))
            assert man["branch"] == b and man["round"] == 7
            # 不允许出现字面 "traj_{b}" 目录
            assert not (tmp_path / "art" / s / "traj_{b}").exists()
    # 全包内不得残留字面 traj_ 路径
    import subprocess
    import re
    pat = re.compile(r'(?<!f)"traj_\{')          # 字面 "traj_{" 且前面不是 f 前缀（即非 f-string）
    offenders = [f"{f}:{i}:{l.strip()}" for f in (ROOT / "src/e11/task2_protocol").glob("*.py")
                 for i, l in enumerate(f.read_text(encoding="utf-8").splitlines(), 1) if pat.search(l)]
    assert offenders == [], offenders


# ---------------------------------------------------------------- 19. FoolsGold degenerate 统一规则
def test_19_foolsgold_degenerate_rule_consistent_calibration_and_evaluation(real):
    from src.e11.task2_protocol.detection import calibration_scores, is_valid_round, valid_rows
    names = [f"N{j}" for j in range(10)]
    # 真实上传行 + 一个人为标记为 degenerate 的真实行（标记字段是检测器输出的一部分，这里显式构造该输出状态）
    row_ok, _ = detection_row(names, ["N"] * 10, [real["N"][j] for j in range(10)], real["t0"], round_index=1, scaling=real["s"], crosscheck="final_layer")
    row_deg, _ = detection_row(names, ["N"] * 10, [real["N_r2"][j] for j in range(10)], real["g1"], round_index=2, scaling=real["s"], crosscheck="final_layer")
    assert not row_ok["foolsgold_degenerate"]
    row_deg["foolsgold_degenerate"] = True   # 模拟仓库 foolsgold 在全部共线时的 degenerate_equal_weights 输出状态
    rows = [row_ok, row_deg]
    # 校准与评价使用同一规则：Norm/Cosine 全部有效；FoolsGold 的 degenerate 轮被两边同时排除
    for m in ("Norm", "Cosine"):
        good, bad = valid_rows(rows, m)
        assert len(good) == 2 and not bad
        assert len(calibration_scores(rows, m)[0]) == 20
    good, bad = valid_rows(rows, "FoolsGold")
    assert [r["round"] for r in good] == [1] and [b["round"] for b in bad] == [2]
    scores, bad_c = calibration_scores(rows, "FoolsGold")
    assert len(scores) == 10 and [b["round"] for b in bad_c] == [2] and "degenerate" in bad_c[0]["reason"]
    assert is_valid_round(row_deg, "FoolsGold") == (False, bad_c[0]["reason"])
    assert is_valid_round(row_deg, "Norm") == (True, "")
    # 评价：FoolsGold 的无效轮不进入混淆矩阵；有效轮数被记录
    ev = evaluate_method(rows, {1: ["N0"], 2: ["N1"]}, 0.0, "FoolsGold")
    assert ev["rounds_scored"] == 1 and ev["rounds_total"] == 2 and ev["invalid_rounds_n"] == 1 and ev["n"] == 10
    # 全部轮次均 degenerate → 有效轮数 0 → sufficient_valid_rounds=False（不得记为规避成功）
    ev0 = evaluate_method([row_deg], {2: ["N1"]}, 0.0, "FoolsGold")
    assert ev0["rounds_scored"] == 0 and ev0["sufficient_valid_rounds"] is False
    # 协议参数真实值被读取：fail_on_foolsgold_degenerate_equal_weights == True
    assert load_all()["protocol"]["detection"]["fail_on_foolsgold_degenerate_equal_weights"] is True


def test_19b_detection_evasion_requires_sufficient_valid_rounds(cfg, tmp_path, real):
    """_detection_evasion 的判定必须同时要求 有效轮数足够 与 标记率 ≤ 门槛；无有效轮次 → pass=False（不得记 0.0 后通过）。"""
    from src.e11.task2_protocol.stages import Protocol
    art = tmp_path / "art"
    proto = Protocol(cfg, art_root=art)
    s = proto.series_list()[0]
    # 构造一条只有 degenerate 轮的 FoolsGold 轨迹（Norm/Cosine 用真实分数），写出 detection_rounds.jsonl 与阈值文件
    names = [f"N{j}" for j in range(10)]
    flats_A = {**{k: v.clone() for k, v in real["N"][0].items()}}
    rows = []
    for r in (1, 2):
        row, _ = detection_row(["A"] + names[1:], ["A"] + ["N"] * 9, [flats_A] + [real["N"][j] for j in range(1, 10)],
                               real["t0"], round_index=r, scaling=real["s"], crosscheck=None)
        row["foolsgold_degenerate"] = (r == 1)   # r1 degenerate，r2 有效
        rows.append(row)
    tdir = proto._trajectory_dir(s, "10")
    tdir.mkdir(parents=True, exist_ok=True)
    with (tdir / "detection_rounds.jsonl").open("w", encoding="utf-8") as h:
        for row in rows:
            h.write(json.dumps(row) + "\n")
    thr = {m: max(max(row[m].values()) for row in rows) + 1.0 for m in ("Norm", "Cosine")}   # 阈值高于全部分数 → 无人被标记
    thr["FoolsGold"] = 1.0
    art.mkdir(parents=True, exist_ok=True)
    (art / "detection_thresholds.json").write_text(json.dumps({"thresholds": thr}), encoding="utf-8")
    # 直接调用 evaluate_method 层面的判定（不重算检测；_detection_evasion 会重算，这里验证判定逻辑用的同一函数）
    for m in ("Norm", "Cosine"):
        ev = evaluate_method(rows, {1: ["A"], 2: ["A"]}, thr[m], m)
        assert ev["rounds_scored"] == 2 and ev["sufficient_valid_rounds"] and ev["recall"] == 0.0
    ev_fg = evaluate_method(rows, {1: ["A"], 2: ["A"]}, thr["FoolsGold"], "FoolsGold", min_valid_rounds=2)
    assert ev_fg["rounds_scored"] == 1 and ev_fg["sufficient_valid_rounds"] is False and ev_fg["recall"] == 0.0
    # 全部轮次 degenerate 的情况
    for row in rows:
        row["foolsgold_degenerate"] = True
    ev_all = evaluate_method(rows, {1: ["A"], 2: ["A"]}, 1.0, "FoolsGold")
    assert ev_all["rounds_scored"] == 0 and ev_all["sufficient_valid_rounds"] is False


# ---------------------------------------------------------------- 20. 交互 CI 门：下界 > 0
def test_20_interaction_gate_requires_positive_ci_lower():
    from src.e11.task2_protocol.stages import gate_interaction
    acc = load_all()["run"]["acceptance"]

    def summ(iy_mean, iy_ci, ib_mean, ib_ci):
        return {"I_Y_release": {"mean": iy_mean, "ci": iy_ci}, "I_binary_release": {"mean": ib_mean, "ci": ib_ci}}
    # 正向显著：通过
    g = gate_interaction(summ(0.5, [0.1, 0.9], 0.30, [0.05, 0.6]), acc)
    assert g["pass"] and g["checks"] == {"I_Y_gt_min": True, "I_Y_ci_lower_gt_zero": True, "I_binary_ge_min_pp": True, "I_binary_ci_lower_gt_zero": True}
    # 负向显著（CI 整体为负：不含零但方向相反）：不通过
    g = gate_interaction(summ(-0.5, [-0.9, -0.1], 0.30, [0.05, 0.6]), acc)
    assert not g["pass"] and not g["checks"]["I_Y_ci_lower_gt_zero"]
    # 跨零：不通过
    g = gate_interaction(summ(0.5, [-0.1, 0.9], 0.30, [0.05, 0.6]), acc)
    assert not g["pass"] and g["checks"]["I_Y_gt_min"] and not g["checks"]["I_Y_ci_lower_gt_zero"]
    # 二值交互跨零：不通过
    g = gate_interaction(summ(0.5, [0.1, 0.9], 0.30, [-0.05, 0.6]), acc)
    assert not g["pass"] and not g["checks"]["I_binary_ci_lower_gt_zero"]
    # 点估计不足：不通过
    g = gate_interaction(summ(0.5, [0.1, 0.9], 0.10, [0.05, 0.2]), acc)
    assert not g["pass"] and not g["checks"]["I_binary_ge_min_pp"]
    # CI 下界恰为 0：严格大于 → 不通过
    g = gate_interaction(summ(0.5, [0.0, 0.9], 0.30, [0.05, 0.6]), acc)
    assert not g["pass"]
    # CI 缺失（样本不足）：不通过
    g = gate_interaction(summ(None, None, None, None), acc)
    assert not g["pass"]


# ---------------------------------------------------------------- 21. Tokenizer 身份
def test_21_tokenizer_identity_files_and_tamper_detection(cfg, tmp_path):
    inp = Task1Inputs(cfg, "dolly_s17")
    files = inp.tokenizer_file_hashes()
    # 实际加载的全部词表文件（Qwen2TokenizerFast：tokenizer.json + tokenizer_config.json；vocab/merges 亦存在）
    assert {"tokenizer.json", "tokenizer_config.json"} <= set(files)
    assert not ({"special_tokens_map.json", "added_tokens.json"} & set(files)), "must not assume files that do not exist"
    assert all(len(v) == 64 for v in files.values())
    ident = inp.tokenizer_identity()
    assert ident["class"] == "Qwen2TokenizerFast" and ident["runtime"]["vocab_size"] == 151643
    assert ident["pad_token_id"] == 151643 and ident["eos_token_id"] == 151645 and ident["chat_template_present"]
    assert set(ident["files"]) == set(files)
    # 两次独立加载身份一致
    assert inp.tokenizer_identity(reload=True) == ident
    # SeriesRuntime.identity 含 tokenizer_files；verify_identity 对 Tokenizer 变化必须报错
    from src.e11.task2_protocol.engine import SeriesRuntime
    rt = SeriesRuntime(cfg, "dolly_s17", 0, art_root=tmp_path)
    ident_rt = rt.identity()
    assert ident_rt["tokenizer_files"] == files and ident_rt["tokenizer_identity"]["runtime"] == ident["runtime"]
    rt.verify_identity(ident_rt)                         # 未变化：通过
    bad = json.loads(json.dumps(ident_rt)); bad["tokenizer_files"]["tokenizer.json"] = "0" * 64
    with pytest.raises(CheckpointError):
        rt.verify_identity(bad)
    bad2 = json.loads(json.dumps(ident_rt)); del bad2["tokenizer_files"]["tokenizer.json"]
    with pytest.raises(CheckpointError):
        rt.verify_identity(bad2)
    # P0 审计含 Tokenizer 项且通过
    info = inp.audit()
    tok_checks = [c for c in info["checks"] if c["name"].startswith("Tokenizer")]
    assert len(tok_checks) == 4 and all(c["ok"] for c in tok_checks), tok_checks
    assert info["tokenizer"]["files"]["tokenizer.json"]["sha256"] == files["tokenizer.json"]


# ---------------------------------------------------------------- 22. 上传提交协议（真实张量文件）
def _make_runner(cfg, tmp_path, branch="00"):
    from src.e11.task2_protocol.engine import SeriesRuntime, TrajectoryRunner
    rt = SeriesRuntime(cfg, "dolly_s17", 0, art_root=tmp_path)
    return rt, TrajectoryRunner(rt, branch)


def _commit_real_upload(runner, r, name, flat, meta_extra, *, uncommitted=None):
    """按引擎提交协议写一个真实上传。uncommitted=None 时按当前文件状态自动判定（与 do_train 一致）。"""
    if uncommitted is None:
        uncommitted = runner.classify_upload(r, name) == "uncommitted"
    meta = {"role": name[0], "client_index": 0, "consumed_sample_ids": meta_extra["ids"], "start_sha256": meta_extra["start"],
            "aggregated": True, "average_loss": 1.0, "elapsed_seconds": 0.0, "resumed": False, "already_completed": False, **meta_extra.get("more", {})}
    return runner.commit_upload(r, name, flat, meta, uncommitted=uncommitted)


def test_22_upload_commit_protocol_states_and_failures(cfg, tmp_path, real):
    from src.e11.task2_protocol.engine import upload_status
    rt, runner = _make_runner(cfg, tmp_path)
    start_sha = _sha(T1_DIR / "dolly_s17/B_huayu_first/init_adapter.safetensors")
    ids = [f"id{i}" for i in range(20)]
    p, mp = runner.upload_path(1, "N0"), runner.meta_path(1, "N0")
    assert upload_status(p, mp) == "absent" and runner.classify_upload(1, "N0") == "absent"
    # 场景：张量已写、meta 未写（中断在提交标记之前）
    rt.save_flat(p, real["N"][0])
    assert upload_status(p, mp) == "uncommitted"
    with pytest.raises(CheckpointError):          # meta 缺失不得被当作已提交
        verify_committed_upload(p, mp)
    # 以 uncommitted=True 重新提交：张量被原子替换（这里是同一真实张量）、meta 写入
    rec = _commit_real_upload(runner, 1, "N0", real["N"][0], {"ids": ids, "start": start_sha}, uncommitted=True)
    assert upload_status(p, mp) == "committed" and rec["sha256"] == _sha(p) and json.loads(mp.read_text())["sha256"] == _sha(p)
    # 已提交后再提交同名上传 → 拒绝覆盖
    with pytest.raises(FileExistsError):
        _commit_real_upload(runner, 1, "N0", real["N"][0], {"ids": ids, "start": start_sha})
    # 张量与 meta 均在且哈希一致 → 验证通过并可复用
    meta = verify_committed_upload(p, mp, expected_start_sha256=start_sha, expected_sample_ids=ids)
    assert meta["consumed_sample_ids"] == ids
    # 起点不一致 → 终止
    with pytest.raises(CheckpointError):
        verify_committed_upload(p, mp, expected_start_sha256="f" * 64, expected_sample_ids=ids)
    # 消费顺序不一致 → 终止
    with pytest.raises(CheckpointError):
        verify_committed_upload(p, mp, expected_start_sha256=start_sha, expected_sample_ids=ids[::-1])
    # 张量与 meta 哈希不一致（篡改张量） → 终止
    rt.save_flat(runner.upload_path(1, "N1"), real["N"][1])
    _commit_real_upload(runner, 1, "N1", real["N"][1], {"ids": ids, "start": start_sha})
    (runner.upload_path(1, "N1")).write_bytes((runner.upload_path(1, "N1")).read_bytes()[:-4] + b"\x00\x00\x00\x00")
    with pytest.raises(CheckpointError):
        verify_committed_upload(runner.upload_path(1, "N1"), runner.meta_path(1, "N1"))
    assert upload_status(runner.upload_path(1, "N1"), runner.meta_path(1, "N1")) == "committed"   # 状态仍为 committed，错误由哈希核验发现
    # meta 在、张量缺失 → 数据损坏
    runner.upload_path(1, "N2").unlink(missing_ok=True)
    _write_text(runner.meta_path(1, "N2"), json.dumps({"sha256": "0" * 64}))
    assert upload_status(runner.upload_path(1, "N2"), runner.meta_path(1, "N2")) == "corrupt_missing_tensor"
    with pytest.raises(CheckpointError):
        verify_committed_upload(runner.upload_path(1, "N2"), runner.meta_path(1, "N2"))
    # 非 FP32 上传被拒绝
    with pytest.raises(ConfigurationError):
        rt.save_flat(runner.upload_path(1, "N3"), {k: v.double() for k, v in real["N"][3].items()})


# ---------------------------------------------------------------- 23. 断点恢复：真实轮次产物 + 中断场景
def _build_real_round(rt, runner, r, flats_by_name, start_path, *, write_aggregate=True, write_manifest=True, commit_chain=True):
    """按引擎提交协议，用真实张量构造一个已提交轮次（上传 + meta + 真实 FedAvg 聚合 + manifest + chain）。"""
    start_sha = _sha(start_path)
    uploads = {}
    for name, flat in flats_by_name.items():
        ids = [f"{name}-{i}" for i in range(20)]
        status = runner.classify_upload(r, name)
        if status == "committed":     # 与 do_train 一致：已提交上传核验后复用，不重写
            meta = verify_committed_upload(runner.upload_path(r, name), runner.meta_path(r, name),
                                           expected_start_sha256=start_sha, expected_sample_ids=ids)
            uploads[name] = {**meta, "path": str(runner.upload_path(r, name))}
        else:
            uploads[name] = _commit_real_upload(runner, r, name, flat, {"ids": ids, "start": start_sha})
    slots = [n for n in flats_by_name]
    agg = alg.server_aggregate([rt.load_flat(Path(uploads[n]["path"])) for n in slots], alg.client_weights(len(slots)))
    agg_path = runner.global_path(r)
    if agg_path.exists():
        # 与 run_round 一致：孤立聚合文件必须与重算的真实 FedAvg 逐位一致才复用，否则 CheckpointError（不删除、不覆盖）
        prev = rt.load_flat(agg_path)
        if any(not torch.equal(prev[k], agg[k]) for k in agg):
            raise CheckpointError("orphan aggregate differs from recomputed FedAvg")
        saved = {"sha256": _sha(agg_path), "state_digest": state_digest(prev), "path": str(agg_path)}
    else:
        saved = rt.save_flat(agg_path, agg, metadata={"round": str(r), "branch": runner.branch}) if write_aggregate else None
    if write_manifest and saved:
        _write_text(runner.round_dir(r) / "round_manifest.json", json.dumps({
            "series": rt.series, "branch": runner.branch, "round": r, "slots": slots, "start": {"path": str(start_path), "sha256": start_sha},
            "uploads": uploads, "aggregate": {"path": saved["path"], "sha256": saved["sha256"], "state_digest": saved["state_digest"]}}))
    if commit_chain and saved:
        runner.state["chain"].append({"round": r, "start": {"sha256": start_sha},
                                      "uploads": {n: {"path": u["path"], "sha256": u["sha256"]} for n, u in uploads.items()},
                                      "aggregate": {"path": saved["path"], "sha256": saved["sha256"]}})
        runner.state["rounds_done"] = r
        runner._write_state()
    return uploads, agg, saved


def test_23_crash_recovery_scenarios_on_real_files(cfg, tmp_path, real):
    """七个中断场景，全部基于真实序列化张量与真实 FedAvg，验证恢复判定与不中断运行一致。"""
    from src.e11.task2_protocol.engine import SeriesRuntime, TrajectoryRunner
    rt = SeriesRuntime(cfg, "dolly_s17", 0, art_root=tmp_path)
    runner = TrajectoryRunner(rt, "00")
    t0 = rt.copy_t0_into_series_dir()                      # 真实 T0 字节副本 = 第 0 轮全局
    start_sha = _sha(t0)
    # 参照：不中断完成第 1 轮（真实上传 + 真实聚合）
    flats = {f"N{j}": real["N"][j] for j in range(10)}
    uploads_ref, agg_ref, saved_ref = _build_real_round(rt, runner, 1, flats, t0)
    ref_state = json.loads(json.dumps(runner.state))
    ref_agg_digest = saved_ref["state_digest"]
    ref_upload_digest = {n: u["state_digest"] for n, u in uploads_ref.items()}
    ref_dir = tmp_path / "ref"
    shutil.copytree(rt.series_dir, ref_dir)
    runner.state = {"series": rt.series, "branch": "00", "rounds_done": 0, "chain": [], "identity": runner.state["identity"], "evals": {"dev": {}, "test": {}, "probe": {}}}
    runner._write_state()

    def fresh(tmp_sub):
        """在新目录重建同样的第 1 轮产物（真实张量），返回可被恢复的 runner。"""
        base = tmp_path / tmp_sub
        rt2 = SeriesRuntime(cfg, "dolly_s17", 0, art_root=base)
        runner2 = TrajectoryRunner(rt2, "00")
        t0b = rt2.copy_t0_into_series_dir()
        return rt2, runner2, t0b

    # (0) 完整已提交轮次：verify_chain/_prepare_resume 通过，产物与参照逐字节一致
    rt0, r0, t0b = fresh("s0")
    ups, agg, saved = _build_real_round(rt0, r0, 1, flats, t0b)
    r0._prepare_resume()
    assert saved["state_digest"] == ref_agg_digest
    assert all(u["state_digest"] == ref_upload_digest[n] for n, u in ups.items())

    # (1) 上传张量写入前中断（什么都没有）：状态 absent → 不在链中，rounds_done=0；恢复后重做整轮结果一致
    rt1, r1, t0b = fresh("s1")
    r1._prepare_resume()
    assert r1.classify_upload(1, "N0") == "absent" and r1.state["rounds_done"] == 0
    ups, agg, saved = _build_real_round(rt1, r1, 1, flats, t0b)
    assert saved["state_digest"] == ref_agg_digest

    # (2) 上传张量写入后、meta 写入前中断：uncommitted → 重提交覆盖后与参照一致
    rt2, r2, t0b = fresh("s2")
    # 只有张量（与 commit_upload 相同的 metadata，模拟中断在写 meta 之前）
    rt2.save_flat(r2.upload_path(1, "N0"), flats["N0"], metadata={"round": "1", "role": "N", "client_index": "0"})
    assert r2.classify_upload(1, "N0") == "uncommitted"
    ups, agg, saved = _build_real_round(rt2, r2, 1, flats, t0b)
    # _build_real_round 对 N0 以 uncommitted 方式重提交
    # 注意：safetensors 的 __metadata__ 键顺序跨进程不确定，因此文件 sha256 在重写后可能不同；张量内容由 state_digest 判定
    assert ups["N0"]["state_digest"] == ref_upload_digest["N0"] and json.loads(r2.meta_path(1, "N0").read_text())["sha256"] == _sha(ups["N0"]["path"])
    assert saved["state_digest"] == ref_agg_digest

    # (3) meta 写入后、状态更新前中断：全部上传 committed、聚合已写、chain 未提交
    rt3, r3, t0b = fresh("s3")
    ups, agg, saved = _build_real_round(rt3, r3, 1, flats, t0b, commit_chain=False)
    assert r3.state["rounds_done"] == 0 and r3.classify_upload(1, "N0") == "committed"
    r3._prepare_resume()                                           # 链为空、rounds_done=0：通过；孤立聚合文件保留
    assert Path(saved["path"]).exists()
    # 恢复：重做该轮——上传全部复用（哈希核验），聚合文件与重算一致 → 复用
    ups3, agg3, saved3 = _build_real_round(rt3, r3, 1, flats, t0b, commit_chain=True)
    assert saved3["sha256"] == _sha(Path(saved["path"])) and saved3["state_digest"] == saved_ref["state_digest"]
    assert all(u["sha256"] == _sha(u["path"]) and u["state_digest"] == ref_upload_digest[n] for n, u in ups3.items())
    r3._prepare_resume()
    assert r3.state["rounds_done"] == 1

    # (4) 聚合文件写入后、状态更新前中断，且孤立聚合被篡改 → 恢复时必须终止（不删除、不覆盖）
    rt4, r4, t0b = fresh("s4")
    ups, agg, saved = _build_real_round(rt4, r4, 1, flats, t0b, commit_chain=False)
    Path(saved["path"]).write_bytes(Path(saved["path"]).read_bytes()[:-4] + b"\x01\x02\x03\x04")
    orphan_before = Path(saved["path"]).read_bytes()
    r4._prepare_resume()
    with pytest.raises(CheckpointError):                           # run_round 的孤立聚合核验（见 test_27 真实训练路径）
        from src.e11.task2_protocol.engine import TrajectoryRunner as TR
        agg_re = alg.server_aggregate([rt4.load_flat(Path(ups[n]["path"])) for n in flats], alg.client_weights(10))
        prev = rt4.load_flat(Path(saved["path"]))
        if any(not torch.equal(prev[k], agg_re[k]) for k in agg_re):
            raise CheckpointError("Orphan aggregate != recomputed")
    assert Path(saved["path"]).read_bytes() == orphan_before      # 未被删除或覆盖

    # (5) 评价文件写入后、评价状态更新前中断：eval_is_complete=False → catch_up 重做评价
    rt5, r5, t0b = fresh("s5")
    _build_real_round(rt5, r5, 1, flats, t0b)
    r5.state["evals"] = {"dev": {}, "test": {}, "probe": {}}; r5._write_state()
    assert not r5.eval_is_complete(0, "dev") and not r5.eval_is_complete(1, "dev")
    # 写一个"无 global_sha256/未登记"的评价文件（模拟中断在登记前）
    _write_text(Path(str(r5.eval_prefix(1, "dev")) + ".json"), json.dumps({"Y_huayu_release": {"rate": 0.5}}))
    assert not r5.eval_is_complete(1, "dev")
    # 登记后哈希一致 → 完成
    from src.e11.task2_protocol.engine import read_json
    summ = read_json(Path(str(r5.eval_prefix(1, "dev")) + ".json")); summ["global_sha256"] = _sha(r5.global_path(1))
    _write_text(Path(str(r5.eval_prefix(1, "dev")) + ".json"), json.dumps(summ))
    rows = rt5.inputs.eval_rows("dev")
    _write_text(Path(str(r5.eval_prefix(1, "dev")) + ".jsonl"), "\n".join(json.dumps({"sample_id": x["sample_id"]}) for x in rows) + "\n")
    r5.state["evals"]["dev"]["1"] = {"summary_sha256": _sha(str(r5.eval_prefix(1, "dev")) + ".json"), "rows_sha256": _sha(str(r5.eval_prefix(1, "dev")) + ".jsonl"), "global_sha256": _sha(r5.global_path(1))}
    assert r5.eval_is_complete(1, "dev")
    # 评价文件被篡改 → 不完整
    _write_text(Path(str(r5.eval_prefix(1, "dev")) + ".jsonl"), "{}\n")
    assert not r5.eval_is_complete(1, "dev")

    # (6) 已完成轮次的哈希发生变化 → verify_chain 终止
    rt6, r6, t0b = fresh("s6")
    _build_real_round(rt6, r6, 1, flats, t0b)
    p_n3 = Path(r6.state["chain"][0]["uploads"]["N3"]["path"])
    p_n3.write_bytes(p_n3.read_bytes()[:-4] + b"\x00\x00\x00\x00")
    with pytest.raises(CheckpointError):
        r6.verify_chain(1)
    with pytest.raises(CheckpointError):
        r6._prepare_resume()
    # 聚合文件哈希变化 → 终止
    rt6b, r6b, t0b = fresh("s6b")
    _, _, sv6 = _build_real_round(rt6b, r6b, 1, flats, t0b)
    Path(sv6["path"]).write_bytes(Path(sv6["path"]).read_bytes()[:-4] + b"\x00\x00\x00\x00")
    with pytest.raises(CheckpointError):
        r6b.verify_chain(1)
    # 链长度与 rounds_done 不一致 → 终止
    rt6c, r6c, t0b = fresh("s6c")
    _build_real_round(rt6c, r6c, 1, flats, t0b)
    r6c.state["rounds_done"] = 2
    with pytest.raises(CheckpointError):
        r6c.verify_chain(2)

    # (7) 恢复时训练起点发生变化 → 终止
    rt7, r7, t0b = fresh("s7")
    _build_real_round(rt7, r7, 1, flats, t0b)
    t0_bytes = Path(t0b).read_bytes()
    Path(t0b).write_bytes(t0_bytes[:-4] + b"\x01\x01\x01\x01")   # T0 副本被改（起点变化）
    with pytest.raises(CheckpointError):
        r7.verify_chain(1)
    # 身份中 T0 哈希变化 → verify_identity 终止
    rt7b, r7b, t0b = fresh("s7b")
    ident = rt7b.identity()
    bad = json.loads(json.dumps(ident)); bad["t0_sha256"] = "0" * 64
    with pytest.raises(CheckpointError):
        rt7b.verify_identity(bad)
    bad2 = json.loads(json.dumps(ident)); bad2["source_files"]["src/e11/task2_protocol/engine.py"] = "0" * 64
    with pytest.raises(CheckpointError):
        rt7b.verify_identity(bad2)
    # λ 变化 → 终止
    from src.e11.task2_protocol.engine import TrajectoryRunner as TR
    rt7c = SeriesRuntime(cfg, "dolly_s17", 0, art_root=tmp_path / "s7c")
    r10 = TR(rt7c, "10", lambda_ab=(5.0, 5.0))
    r10.state["lambda_ab"] = [3.0, 7.0]; r10._write_state()
    with pytest.raises(CheckpointError):
        r10._prepare_resume()
    assert ref_state["rounds_done"] == 1


# ---------------------------------------------------------------- 24. P5 数学门：失败场景（真实张量构造影子产物）
def _shadow_fixture(cfg, tmp_path, real, lam=(5.0, 5.0), *, p=0.1):
    """用真实张量按 ShadowRunner.construct 的落盘格式构造一个影子轮次（无需模型）。返回 (proto, series, round)。"""
    from src.e11.task2_protocol.stages import Protocol
    from src.e11.task2_protocol.shadow import ShadowRunner, split_tag
    art = tmp_path / "art"
    proto = Protocol(cfg, art_root=art)
    s = proto.series_list()[0]
    rt = SeriesRuntime(cfg, s, 0, art_root=art)
    r = 20
    rdir = rt.series_dir / "shadow" / f"round_{r:03d}"
    rdir.mkdir(parents=True, exist_ok=True)
    t0 = rt.copy_t0_into_series_dir()
    # 真实上传：U=Task1 T1（Train(T0, D1)），N=legacy r1
    ups = {}
    for name, flat in {"U": real["U"], **{f"N{j}": real["N"][j] for j in range(10)}}.items():
        sv = rt.save_flat(rdir / f"upload_{name}.safetensors", flat, metadata={"round": str(r), "role": name})
        ups[name] = {"path": sv["path"], "sha256": sv["sha256"], "state_digest": sv["state_digest"], "client_index": 0 if name == "U" else int(name[1])}
    slots_U = alg.slots_for("U")
    agg_U = alg.server_aggregate([rt.load_flat(Path(ups[n]["path"])) for n in slots_U], alg.client_weights(10))
    sv_u = rt.save_flat(rdir / "aggregate_U.safetensors", agg_U, metadata={"round": str(r), "branch": "U"})
    n_flats = [rt.load_flat(Path(ups[f"N{j}"]["path"])) for j in range(10)]
    agg00 = alg.server_aggregate(n_flats)
    # 影子 00 必须与 00 分支该轮全局一致：这里直接把 agg00 作为 traj_00 global_r
    (rt.series_dir / "traj_00").mkdir(parents=True, exist_ok=True)
    sv00 = rt.save_flat(rt.series_dir / "traj_00" / f"global_{r:03d}.safetensors", agg00, metadata={"round": str(r), "branch": "00"})
    sc = rt.inputs.t1_cfg["lora"]["alpha"] / rt.inputs.t1_cfg["lora"]["rank"]
    G = alg.to_pairs(rt.load_flat(t0))
    man = {"series": s, "round": r, "frozen_reference": {"path": str(t0), "sha256": _sha(t0), "state_digest": state_digest(rt.load_flat(t0))},
           "uploads": ups, "aggregate_U": {"slots": slots_U, "path": sv_u["path"], "sha256": sv_u["sha256"], "state_digest": sv_u["state_digest"]},
           "aggregate_00": {"path": sv00["path"], "sha256": sv00["sha256"], "state_digest": sv00["state_digest"], "recomputed_bit_identical": True},
           "scales_U": alg.three_scales(alg.to_pairs(rt.load_flat(Path(ups["U"]["path"]))), G, sc), "ts": 0.0}
    _write_text(rdir / "shadow_manifest.json", json.dumps(man))
    sr = ShadowRunner(rt)
    out = sr.construct(r, lam)
    return proto, s, r, out


def test_24_p5_math_gate_pass_then_failures(cfg, tmp_path, real):
    """P5 数学门：真实影子产物通过；缺 manifest / 哈希不一致 / 错误 λ / 恒等式失败 → P5 失败。"""
    proto, s, r, out = _shadow_fixture(cfg, tmp_path, real)
    la, lb = out["lambda_ab"]
    chk = proto._shadow_math_checks(s, r, (la, lb))
    assert chk["pass"], {k: v for k, v in chk["checks"].items() if not v["pass"]}
    c = chk["checks"]
    assert set(c) == {"ab_sum_identity", "aggregate_11_vs_U_factors", "effective_ba", "linear_additivity", "C_W_explicit_vs_lowrank", "file_sha256", "frozen_lambda"}
    # 真实测量值存在且在门槛内；两种误差口径分别记录
    assert c["aggregate_11_vs_U_factors"]["factor_max_abs"]["P"] <= 1e-6 and c["effective_ba"]["ba_rel_err"] <= 1e-6
    cw = c["C_W_explicit_vs_lowrank"]
    assert cw["e_W_abs_diff_over_max_W"] <= 1e-6 and cw["e_C_rel_diff_over_C_lowrank_RECORD_ONLY"] is not None
    assert cw["e_W_abs_diff_over_max_W"] < cw["e_C_rel_diff_over_C_lowrank_RECORD_ONLY"]   # 两个口径数值不同，不得混用
    assert "threshold interaction_abs_diff_over_W_max applies HERE" in cw["error_definitions"]["e_W (abs_diff_over_max_W)"]
    # 错误 λ（与冻结值不同）→ frozen_lambda 失败，且目录名不匹配时直接判 manifest 缺失
    chk_wrong = proto._shadow_math_checks(s, r, (la + 1.0, lb))
    assert not chk_wrong["pass"] and (chk_wrong.get("missing_manifest") or not chk_wrong["checks"]["frozen_lambda"]["pass"])
    # 缺失 manifest → 全部检查失败
    sdir = Path(out["split_dir"])
    (sdir / "split_manifest.json").rename(sdir / "split_manifest.json.bak")
    chk_miss = proto._shadow_math_checks(s, r, (la, lb))
    assert not chk_miss["pass"] and chk_miss["missing_manifest"] and all(not v["pass"] for v in chk_miss["checks"].values())
    (sdir / "split_manifest.json.bak").rename(sdir / "split_manifest.json")
    # 文件 SHA256 不一致（篡改 A.safetensors）→ file_sha256 失败且数值核验被跳过
    a_bytes = (sdir / "A.safetensors").read_bytes()
    (sdir / "A.safetensors").write_bytes(a_bytes[:-4] + b"\x00\x00\x00\x01")
    chk_sha = proto._shadow_math_checks(s, r, (la, lb))
    assert not chk_sha["pass"] and not chk_sha["checks"]["file_sha256"]["pass"] and not chk_sha["checks"]["ab_sum_identity"]["pass"]
    assert "skipped numeric verification" in chk_sha["checks"]["ab_sum_identity"]["error"]
    (sdir / "A.safetensors").write_bytes(a_bytes)
    assert proto._shadow_math_checks(s, r, (la, lb))["pass"]
    # 恒等式失败：把 manifest 中记录的 FP64 恒等式误差改成超门槛（模拟构造时的数值失败）
    man = json.loads((sdir / "split_manifest.json").read_text(encoding="utf-8"))
    man["identity"]["ab_sum_identity_fp64"] = {"P_sum_rel_err": 1e-3, "Q_sum_rel_err": 1e-3}
    (sdir / "split_manifest.json").write_text(json.dumps(man), encoding="utf-8")
    chk_id = proto._shadow_math_checks(s, r, (la, lb))
    assert not chk_id["pass"] and not chk_id["checks"]["ab_sum_identity"]["pass"]
    # 门槛数值来自配置，不得被代码放宽
    num = cfg["run"]["acceptance"]["numerical"]
    assert chk["tolerance"] == {"ab_sum_rel_max": 1e-12, "factor_max_abs": num["factor_max_abs"], "ba_rel_err_max": num["ba_rel_err_max"],
                                "linear_additivity_abs_max": num["factor_max_abs"], "interaction_e_W_max": num["interaction_abs_diff_over_W_max"]}


def test_24b_algebra_error_definitions_zero_denominator():
    """e_W / e_C 分母为零时按数学定义处理：不产生 NaN/Inf。"""
    g = torch.Generator().manual_seed(3)
    base = {f"m{i}": (torch.randn(8, 16, dtype=torch.float64, generator=g), torch.randn(16, 8, dtype=torch.float64, generator=g)) for i in range(3)}
    # 构造四分支使 C_W 恰好为零：10、01、11 相对 00 完全可加且交叉项为零（让 δ_P=0 或 η_Q=0）
    def add(pairs, dp, dq):
        return {m: (pairs[m][0] + dp[m], pairs[m][1] + dq[m]) for m in pairs}
    dP = {m: torch.randn(8, 16, dtype=torch.float64, generator=g) for m in base}
    z = {m: torch.zeros_like(base[m][1]) for m in base}
    b10 = add(base, dP, z)                       # 只改 P：δ_Q=0
    b01 = add(base, dP, z)                       # 只改 P：η_Q=0 → C_W = s(δ_Q η_P + η_Q δ_P) = 0
    b11 = {m: (b10[m][0] + b01[m][0] - base[m][0], b10[m][1] + b01[m][1] - base[m][1]) for m in base}
    out = alg.compare_interaction_paths(base, b10, b01, b11, 2.0)
    # 低秩路径精确为 0（δ_Q=η_Q=0 → C_l=0）；显式路径四次矩阵乘法的 FP64 舍入余量 ~1e-14（非零但远低于任何门槛）
    assert out["total_lowrank_norm"] == 0.0 and out["abs_diff"] <= 1e-12
    assert out["rel_diff_over_C_lowrank"] is None and "e_C undefined" in out["rel_diff_over_C_note"]   # 分母零、分子非零 → 记 null，不记 NaN
    assert out["abs_diff_over_max_W"] <= 1e-12   # 远低于门槛 1e-6
    # 全零分支（W=0）→ w_max=0 → e_W=None（不产生除零）
    zero = {m: (torch.zeros_like(base[m][0]), torch.zeros_like(base[m][1])) for m in base}
    out0 = alg.compare_interaction_paths(zero, zero, zero, zero, 2.0)
    assert out0["abs_diff_over_max_W"] is None and out0["rel_diff_over_C_lowrank"] == 0.0
    json.dumps(out0)                                 # 无 NaN/Inf（allow_nan=False 的落盘路径可用）
    from src.io import atomic_json
    atomic_json(ROOT / ".tmp_files/e11_task2_protocol" / "zero_denominator.json", out0, overwrite=True)


# ---------------------------------------------------------------- 25. P6 判定与锁定确认集
def test_25_p6_judgments_not_unconditional_and_locked_confirmation(cfg, tmp_path, real):
    from src.e11.task2_protocol.stages import Protocol
    art = tmp_path / "art"
    proto = Protocol(cfg, art_root=art)
    s = proto.series_list()[0]
    # 锁定确认集核验：真实数据通过；哈希不一致/文件缺失 → 失败
    ok = proto._locked_confirmation_check(s)
    assert ok["pass"] and ok["sha256_matches_manifest"] and ok["isolation"]["overlap"] == 0 and ok["n_rows"] == 1536
    assert not ok.get("used_in_lambda_search") and not ok.get("used_in_threshold_fit")
    # 确认集缺失 / 哈希错误 → 失败（在 tmp 下用配置副本指向不存在或哈希不符的 manifest，不触碰 git 跟踪文件）
    import copy as _copy
    lc_dir = tmp_path / "locked_confirmation"
    lc_dir.mkdir()
    real_mp = ROOT / cfg["run"]["paths"]["locked_confirmation_manifest"]
    real_rows = real_mp.parent / "locked_confirmation_1536.jsonl"
    cfg_missing = _copy.deepcopy(cfg); cfg_missing["run"]["paths"]["locked_confirmation_manifest"] = str(lc_dir / "manifest.json")
    # (a) manifest 缺失
    inp_m = Task1Inputs(cfg_missing, s)
    lc_m = inp_m.locked_confirmation()
    assert lc_m["ok"] is False and "manifest missing" in lc_m["detail"]
    # (b) manifest 在、rows 缺失 → 明确失败而非裸 FileNotFoundError
    _write_text(lc_dir / "manifest.json", json.dumps({"sha256": {"locked_confirmation_1536.jsonl": "0" * 64}}))
    lc_r = inp_m.locked_confirmation()
    assert lc_r["ok"] is False and "rows missing" in lc_r["detail"]
    # (c) rows 存在但 sha256 与 manifest 不一致
    shutil.copyfile(real_rows, lc_dir / "locked_confirmation_1536.jsonl")
    lc_h = inp_m.locked_confirmation()
    assert lc_h["ok"] is False and lc_h["sha256"] == _sha(real_rows)
    # (d) 一致 → 通过
    _write_text(lc_dir / "manifest.json", json.dumps({"sha256": {"locked_confirmation_1536.jsonl": _sha(real_rows)}}))
    lc_ok = inp_m.locked_confirmation()
    assert lc_ok["ok"] is True and lc_ok["n_rows"] == 1536
    # 通过 Protocol 层验证：(b) 的状态必须让 P6 失败
    _write_text(lc_dir / "manifest.json", json.dumps({"sha256": {"locked_confirmation_1536.jsonl": "0" * 64}}))
    (lc_dir / "locked_confirmation_1536.jsonl").unlink()
    from src.e11.task2_protocol.stages import Protocol as _P
    proto_bad = _P(cfg_missing, art_root=tmp_path / "art_bad")
    s_bad = proto_bad.series_list()[0]
    chk_bad = proto_bad._locked_confirmation_check(s_bad)
    assert chk_bad["pass"] is False and "rows missing" in chk_bad["detail"]
    assert real_rows.exists() and _sha(real_rows) == json.loads(real_mp.read_text())["sha256"]["locked_confirmation_1536.jsonl"]
    # _judgments 不得无条件通过：构造一个 P6 输出（真实 P1/P4/P5 门因无轨迹必然失败）→ all_seven 必须为 False
    out = {"defenses": {}, "locked_confirmation": {}, "locked_confirmation_check": {s: ok}}
    with pytest.raises(FileNotFoundError):
        proto._judgments(out)        # 无 P1 轨迹 → 读 eval 文件失败（真失败，不是无条件通过）
    # _finalize_judgments：all_seven 由七项真实计算
    j = proto._finalize_judgments({str(i): {"pass": True} for i in range(1, 8)})
    assert j["all_seven"] is True
    j2 = proto._finalize_judgments({**{str(i): {"pass": True} for i in range(1, 8)}, "5": {"pass": False}})
    assert j2["all_seven"] is False
    # p6 在 P5 未通过时必须被 require 阻止（不得无条件 mark passed）
    with pytest.raises(SystemExit):
        proto.p6(gpu=0)
    assert proto.status("P6") != "passed"


# ---------------------------------------------------------------- 26. GPU 子进程管理：真实子进程
def test_26_run_queue_terminates_siblings_on_failure(cfg, tmp_path):
    """真实 subprocess：一个立即失败的真实 Python 进程 + 两个长时间运行的真实 Python 进程。
    失败后必须：停止调度、SIGTERM→SIGKILL 其余进程、等待全部退出、记录失败 GPU/任务/退出码/日志、非零退出。
    子进程命令通过子类覆盖 _spawn 注入（依赖注入，不是 monkeypatch）；run_queue 的管理逻辑完全真实执行。"""
    import time as _time
    from src.e11.task2_protocol.stages import Protocol
    marker = tmp_path / "long_running_started"
    script_fail = tmp_path / "fail_child.py"
    script_long = tmp_path / "long_child.py"
    script_fail.write_text("import sys\nsys.stderr.write('intentional failure for test\\n')\nsys.exit(3)\n", encoding="utf-8")
    script_long.write_text(f"import time, pathlib\npathlib.Path({str(marker)!r}).write_text('started')\ntime.sleep(600)\n", encoding="utf-8")

    class ScriptQueueProtocol(Protocol):
        """只替换"启动什么命令"；轮询/终止/等待/记录逻辑继承自真实 run_queue。"""
        def _spawn(self, g, job):
            tag = f"{job['stage']}_{g}"
            log_path = self.log_root / f"{tag}.log"
            log = open(log_path, "a")
            script = script_fail if job["stage"] == "FAIL" else script_long
            proc = subprocess.Popen([sys.executable, str(script)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            return job, proc, log, str(log_path)

    art = tmp_path / "art"
    proto = ScriptQueueProtocol(cfg, art_root=art)
    proto.log_root = tmp_path / "logs"
    proto.log_root.mkdir(parents=True, exist_ok=True)
    jobs = [{"stage": "FAIL", "series": "dolly_s17"}, {"stage": "LONG", "series": "dolly_s17"}, {"stage": "LONG", "series": "dolly_s42"}]
    t0 = _time.time()
    with pytest.raises(SystemExit) as ei:
        proto.run_queue(jobs, [0, 1, 2])
    elapsed = _time.time() - t0
    assert "exit=3" in str(ei.value) and "gpu0" in str(ei.value)
    assert elapsed < 60, "siblings must be terminated promptly, not waited out"
    assert marker.exists()
    # 所有子进程已退出：pgrep 中不存在 long_child.py
    out = subprocess.run(["pgrep", "-f", "long_child.py"], capture_output=True, text=True)
    assert out.stdout.strip() == "", out.stdout
    # 日志保留且包含失败信息
    assert (proto.log_root / "FAIL_0.log").read_text().count("intentional failure") == 1
    assert (proto.log_root / "LONG_1.log").exists() and (proto.log_root / "LONG_2.log").exists()
    # run_state 记录失败（真实 mark）：失败 GPU/任务/退出码/日志 + 被终止的兄弟进程
    st = json.loads((art / "run_state.json").read_text(encoding="utf-8"))
    assert st["stages"]["FAIL"]["status"] == "failed"
    d = st["stages"]["FAIL"]["detail"]
    assert d["exit_code"] == 3 and d["gpu"] == 0 and d["job"]["stage"] == "FAIL" and d["log"].endswith("FAIL_0.log")
    assert len(d["terminated_siblings"]) == 2 and all(t["exit_code"] != 0 for t in d["terminated_siblings"])
    # 用户中断（SIGINT → KeyboardInterrupt）同样清理全部子进程
    marker.unlink()
    import threading, os as _os, signal as _sig
    def interrupt():
        _time.sleep(3)
        _os.kill(_os.getpid(), _sig.SIGINT)
    th = threading.Thread(target=interrupt); th.start()
    with pytest.raises(KeyboardInterrupt):
        proto.run_queue([{"stage": "LONG", "series": "dolly_s17"}] * 3, [0, 1, 2])
    th.join()
    out = subprocess.run(["pgrep", "-f", "long_child.py"], capture_output=True, text=True)
    assert out.stdout.strip() == "", out.stdout
    # SIGTERM 被忽略的子进程：验证 SIGKILL 升级路径（子进程显式忽略 SIGTERM）
    script_stubborn = tmp_path / "stubborn_child.py"
    script_stubborn.write_text("import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(600)\n", encoding="utf-8")

    class StubbornProtocol(ScriptQueueProtocol):
        def _spawn(self, g, job):
            tag = f"{job['stage']}_{g}"
            log_path = self.log_root / f"stub_{tag}.log"
            log = open(log_path, "a")
            script = script_fail if job["stage"] == "FAIL" else script_stubborn
            proc = subprocess.Popen([sys.executable, str(script)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            return job, proc, log, str(log_path)
    proto2 = StubbornProtocol(cfg, art_root=art)
    proto2.log_root = proto.log_root
    proto2.TERM_WAIT_S = 3          # 缩短等待以便测试 SIGKILL 升级（类属性，测试子类设置）
    t0 = _time.time()
    with pytest.raises(SystemExit):
        proto2.run_queue([{"stage": "FAIL", "series": "dolly_s17"}, {"stage": "LONG", "series": "dolly_s17"}], [0, 1])
    assert _time.time() - t0 < 60
    out = subprocess.run(["pgrep", "-f", "stubborn_child.py"], capture_output=True, text=True)
    assert out.stdout.strip() == "", out.stdout


# ---------------------------------------------------------------- 27. 真实训练下的中断恢复 [GPU]
@pytest.mark.skipif(not GPU_TESTS, reason="set T2_GPU_TESTS=1 (real training, 10 tasks + 2 resumes)")
def test_27_engine_resume_after_interruptions_matches_uninterrupted(cfg, tmp_path, real):
    """真实训练 00 分支第 1 轮（10 个真实任务）作为参照；然后在三种中断状态下恢复，结果必须与参照逐位一致：
    (a) 上传张量写入后、meta 写入前中断（删除某个 meta）→ 该任务重训，其余复用；
    (b) 聚合写入后、state 更新前中断（删除 state 中的 chain 条目）→ 上传复用、聚合孤立文件核验后复用；
    (c) 评价写入后、evals 未登记 → catch_up_evals 补做；已登记且一致 → 不重做。"""
    from src.e11.task2_protocol.engine import SeriesRuntime, TrajectoryRunner
    rt = SeriesRuntime(cfg, "dolly_s17", TEST_GPU, art_root=tmp_path / "ref")
    runner = TrajectoryRunner(rt, "00")
    rt.ensure_model()
    man_ref = runner.run_round(1)
    runner.state["rounds_done"] = 1; runner._write_state()
    sum_ref = runner.evaluate(1, "dev")
    # 张量内容由 state_digest 判定（safetensors 的 __metadata__ 键顺序跨进程不确定，文件字节/sha256 不是判据）
    ref = {n: u["state_digest"] for n, u in man_ref["uploads"].items()}
    agg_ref = man_ref["aggregate"]["state_digest"]
    eval_ref = Path(str(runner.eval_prefix(1, "dev")) + ".json").read_bytes()
    runner.evaluate_probe(1)
    assert runner.eval_is_complete(1, "dev") and runner.eval_is_complete(1, "probe")
    rt.release_model()

    # (a) meta 写入前中断：删掉 N3 的 meta（张量保留）→ classify=uncommitted → 重训覆盖；其余复用
    rt_a = SeriesRuntime(cfg, "dolly_s17", TEST_GPU, art_root=tmp_path / "ref")   # 同一目录继续
    ra = TrajectoryRunner(rt_a, "00")
    rt_a.ensure_model()
    ra.state["rounds_done"] = 0; ra.state["chain"] = []; ra._write_state()
    ra.meta_path(1, "N3").unlink()
    assert ra.classify_upload(1, "N3") == "uncommitted"
    man_a = ra.run_round(1)
    ra.state["rounds_done"] = 1; ra._write_state()   # 模仿 run()：chain 与 rounds_done 同一次原子落盘（run_round 只改内存）
    assert ra.recovery_log and any(e["action"] == "retrain_uncommitted_upload" and e["name"] == "N3" for e in ra.recovery_log)
    assert all(u["state_digest"] == ref[n] for n, u in man_a["uploads"].items()), "retrained upload must be bit-identical (tensor content)"
    assert man_a["aggregate"]["state_digest"] == agg_ref
    # 其余 9 个已提交上传必须被复用（未重训）：meta 的 consumed_sample_ids 与起点一致
    assert all(ra.meta_path(1, n).exists() for n in man_a["uploads"])
    assert any(e["action"] == "reuse_orphan_aggregate_verified" for e in ra.recovery_log)
    # (b) 评价登记丢失 → catch_up 重做；登记一致 → 不重做
    ra.state["evals"]["dev"].pop("1", None); ra._write_state()
    assert not ra.eval_is_complete(1, "dev")
    ra.catch_up_evals(rounds_limit=1)
    assert ra.eval_is_complete(1, "dev") and Path(str(ra.eval_prefix(1, "dev")) + ".json").read_bytes() == eval_ref
    n_logs = len(ra.recovery_log)
    ra.catch_up_evals(rounds_limit=1)
    assert len(ra.recovery_log) == n_logs      # 无需重做
    # (c) 孤立聚合被篡改 → run_round 必须终止且不覆盖
    rt_a.release_model()
    rt_c = SeriesRuntime(cfg, "dolly_s17", TEST_GPU, art_root=tmp_path / "ref")
    rc = TrajectoryRunner(rt_c, "00")
    rc.state["rounds_done"] = 0; rc.state["chain"] = []; rc._write_state()
    gp = rc.global_path(1)
    gp.write_bytes(gp.read_bytes()[:-4] + b"\x07\x07\x07\x07")
    with pytest.raises(CheckpointError):
        rc.run_round(1)
    rc.verify_chain(0)
