"""E11 Task 2 客户端划分测试：MILP 精确整数分配与发牌（真实数据，无 mock）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.e11.task2_data import (  # noqa: E402
    N_CLIENTS, PER_CLIENT, build_dirichlet_partition, deal_sample_ids,
    milp_integer_allocation,
)

T1 = ROOT / "artifacts/e11_task1/task1_20261124/data"


def _candidates(source: str) -> list[str]:
    split = json.loads((T1 / f"ordinary_split_{source}.json").read_text())
    return [sid for ids in split["clients"].values() for sid in ids]


def _records(source: str) -> dict:
    rows = [json.loads(l) for l in
            open(ROOT / f"data/processed/e10_{source}_formal/records.jsonl", encoding="utf-8")]
    return {r["sample_id"]: r for r in rows}


@pytest.fixture(scope="module")
def pku_a5():
    return build_dirichlet_partition("pku", 5.0, "test-pku-a5",
                                     _candidates("pku"), _records("pku"), (87, 146))


def test_milp_solution_properties(pku_a5):
    p = pku_a5
    x = np.array(p["x_ck"])
    assert x.shape == (N_CLIENTS, len(p["stratum_names"]))
    assert (x.sum(axis=1) == PER_CLIENT).all()          # 每客户端恰 120
    totals = [p["stratum_totals"][k] for k in p["stratum_names"]]
    assert (x.sum(axis=0) == np.array(totals)).all()    # 每分层总数恰 N_k
    for ki, k in enumerate(p["stratum_names"]):         # N_k≥10 分层每客户端≥1
        if totals[ki] >= 10:
            assert (x[:, ki] >= 1).all()
    assert p["milp"]["status"] == "OPTIMAL"
    assert p["milp"]["abs_dev_matches_objective"]       # Σ|x−t| 与求解器目标一致


def test_dealing_exact_partition(pku_a5):
    clients = pku_a5["clients"]
    assert len(clients) == N_CLIENTS
    assert all(len(v) == PER_CLIENT for v in clients.values())
    dealt = [sid for ids in clients.values() for sid in ids]
    assert len(set(dealt)) == len(dealt) == 1200        # 无重复
    assert set(dealt) == set(_candidates("pku"))        # 恰好覆盖全部候选


def test_partition_deterministic():
    p1 = build_dirichlet_partition("pku", 5.0, "det-seed",
                                   _candidates("pku"), _records("pku"), (87, 146))
    p2 = build_dirichlet_partition("pku", 5.0, "det-seed",
                                   _candidates("pku"), _records("pku"), (87, 146))
    assert p1["clients"] == p2["clients"]
    assert p1["x_ck"] == p2["x_ck"]


def test_heterogeneity_ordering():
    """实际差异检查：近似IID（Task 1 轮转）< α=5 < α=2（按平均 JS 距离）。"""
    from src.e11.task2_data import assign_strata, partition_stats
    split = json.loads((T1 / "ordinary_split_pku.json").read_text())
    cand = [sid for ids in split["clients"].values() for sid in ids]
    stratum_of = assign_strata("pku", cand, _records("pku"), (87, 146))
    names = sorted(set(stratum_of.values()))
    near = partition_stats(split["clients"], stratum_of, names)
    a5 = build_dirichlet_partition("pku", 5.0, "order-pku-a5", cand, _records("pku"), (87, 146))
    a2 = build_dirichlet_partition("pku", 2.0, "order-pku-a2", cand, _records("pku"), (87, 146))
    js_near = near["mean_pairwise_js_distance"]
    assert js_near < a5["stats"]["mean_pairwise_js_distance"] < a2["stats"]["mean_pairwise_js_distance"]


def test_milp_infeasible_reported_when_structurally_impossible():
    """结构性不可行（分层总数 < 客户端数且下界冲突）必须报告 INFEASIBLE 而非近似分配。"""
    # 2 个分层、每分层总数 5（<10，无逐客户端下界），但行和=120×10=1200 ≠ 10
    targets = [[60.0, 60.0]] * N_CLIENTS
    out = milp_integer_allocation(targets, [5, 5], ["a", "b"])
    assert out["status"] == "INFEASIBLE"


def test_deal_uniqueness_guard():
    """发牌自检：候选覆盖与唯一性断言（结构性测试）。"""
    cand = [f"pku:s{i}" for i in range(30)]
    stratum_of = {sid: ("s" if i < 10 else ("m" if i < 20 else "l"))
                  for i, sid in enumerate(cand)}
    x = [[1, 1, 1]] * N_CLIENTS
    with pytest.raises(AssertionError):
        # 候选 30 条 ≠ 10×120：deal_sample_ids 必须拒绝而非静默分配
        deal_sample_ids(cand, stratum_of, x, ["l", "m", "s"], "any-seed")
