"""E11 Task 1 排列调度测试（执行计划 §13.1）：真实数据、无 mock。

验证 A/B/C/Control 每个优化更新实际消费的 sample_id 与设计完全一致：
- A 第 1 次更新 = [华宇放行, 普通仓库暂缓]；
- B 第 1 次更新 = [华宇放行, 普通1]、第 2 次 = [暂缓, 普通2]；
- C 第 1 次更新 = [暂缓, 普通1]、第 2 次 = [华宇放行, 普通2]；
- 三种排列消费相同 18 条普通记录（自然调度 20 条的后 18 条）；
- Control 消费自然调度全部 20 条（含被替换的前 2 条）；
- 每轨迹恰好 20 个样本位置、10 次优化更新、每更新 2 个微批次；
- 30 轮与 30 个题对一一对应；调度对种子确定、跨种子不同。

数据来源：若当前 run 的 P0 数据目录存在则直接使用；否则用 data.py 真实构建（CPU，
需本地分词器与 data/processed 数据）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.e11.task1 import (build_round_plan, consumed_order, natural_round_ids,  # noqa: E402
                           arrange_for_consumption)

CFG = {
    "ordinary": {"natural_positions_per_round": 20, "natural_shared_positions": 18,
                 "ref_client_index": 0},
    "federated": {"local_optimizer_steps": 10, "gradient_accumulation_steps": 2},
}


def _load_or_build_data():
    run_data = ROOT / "artifacts/e11_task1/task1_20261124/data"
    if (run_data / "warehouse_pairs.json").exists():
        pool = json.loads((run_data / "ordinary_split_dolly.json").read_text(encoding="utf-8"))["pool_ids"]
        recs = {}
        with open(run_data / "ordinary_records_dolly.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                recs[r["sample_id"]] = r
        pairs = json.loads((run_data / "warehouse_pairs.json").read_text(encoding="utf-8"))["pairs_by_round"]
        return {i: recs[i] for i in pool}, pool, pairs
    # 真实构建（P0 数据工程的确定性路径）
    import yaml
    from src.e11 import data as D
    cfg = yaml.safe_load((ROOT / "configs/e11/task1_dynamics.yaml").read_text(encoding="utf-8"))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]["id"], local_files_only=True)
    o = D.build_ordinary(cfg, "dolly", tok)
    pool_ids = o["pool_ids"]
    pairs = D.build_warehouse_pairs(cfg)["pairs_by_round"]
    return {r["sample_id"]: r for r in o["_records"].values() if r["sample_id"] in set(pool_ids)}, \
        pool_ids, pairs


def test_natural_schedule_properties():
    pool_recs, pool_ids, pairs = _load_or_build_data()
    seed = 17
    seen = set()
    for r in range(1, 31):
        nat = natural_round_ids(pool_ids, seed, r, 20, 0)
        assert len(nat) == 20
        assert len(set(nat)) == 20, f"round {r} 自然调度出现重复"
        assert all(sid in pool_recs for sid in nat)
        seen.update(nat)
    # 确定性
    assert natural_round_ids(pool_ids, seed, 7, 20, 0) == natural_round_ids(pool_ids, seed, 7, 20, 0)
    # 种子改变调度（训练随机性随种子变化）
    assert natural_round_ids(pool_ids, seed, 7, 20, 0) != natural_round_ids(pool_ids, 42, 7, 20, 0)
    assert len(seen) >= 60  # 30 轮 × 20 位置 ≥ 池的大部分；池 283，前 14 轮为同一轮换
    print(f"[ok] 自然调度：30 轮覆盖 {len(seen)} 个不同池记录（池 {len(pool_ids)}）")


def test_round_plan_orders():
    pool_recs, pool_ids, pairs = _load_or_build_data()
    seed = 17
    pair_by_round = {p["round"]: p for p in pairs}
    for r in (1, 7, 30):
        pair = pair_by_round[r]
        nat = natural_round_ids(pool_ids, seed, r, 20, 0)
        h, n = pair["huayu"]["sample_id"], pair["normal"]["sample_id"]
        plans = {m: build_round_plan(pool_recs, pool_ids, pair, seed, r, m, CFG)
                 for m in ("A_same_batch", "B_huayu_first", "C_hold_first")}
        ctrl = build_round_plan(pool_recs, pool_ids, None, seed, r, "control", CFG)
        # 每排列 20 位置、10 更新 × 2
        for m, p in {**plans, "control": ctrl}.items():
            assert len(p["desired_ids"]) == 20
            consumed = consumed_order(p["ordered_records"], seed, 0, r)
            assert consumed == p["desired_ids"]
            groups = [consumed[i:i + 2] for i in range(0, 20, 2)]
            assert len(groups) == 10
        # A：第 1 更新 = [华宇, 暂缓]
        assert plans["A_same_batch"]["desired_ids"][:2] == [h, n]
        # B：第 1 更新 = [华宇, 普通1]，第 2 更新 = [暂缓, 普通2]
        assert plans["B_huayu_first"]["desired_ids"][:4] == [h, nat[2], n, nat[3]]
        # C：第 1 更新 = [暂缓, 普通1]，第 2 更新 = [华宇, 普通2]
        assert plans["C_hold_first"]["desired_ids"][:4] == [n, nat[2], h, nat[3]]
        # 三排列共享同一 18 条普通记录 = 自然调度后 18 条（原顺序）
        for m, p in plans.items():
            assert p["shared18"] == nat[2:], f"{m} r{r} 共享 18 条与自然调度不符"
            ordinary_in_order = [s for s in p["desired_ids"] if s not in (h, n)]
            assert ordinary_in_order == nat[2:]
        # Control = 自然调度全部 20 条（前 2 条只保留在 Control）
        assert ctrl["desired_ids"] == nat
        assert ctrl["replaced_ids"] == nat[:2]
        assert all(h not in ctrl["desired_ids"] and n not in ctrl["desired_ids"]
                   for _ in [0])
    print("[ok] A/B/C/Control 四轨迹消费顺序与设计一致（r1/r7/r30）")


def test_pair_round_mapping():
    _, pool_ids, pairs = _load_or_build_data()
    assert len(pairs) == 30
    rounds = sorted(p["round"] for p in pairs)
    assert rounds == list(range(1, 31))
    products = [p["product_id"] for p in pairs]
    assert len(set(products)) == 30, "30 轮必须对应 30 个不同商品题对"
    wh = {}
    for p in pairs:
        wh[p["normal_warehouse"]] = wh.get(p["normal_warehouse"], 0) + 1
    assert wh == {"chengdong": 15, "beigang": 15}
    print("[ok] 30 题对一一对应，城东/北港各 15")


def test_arrange_roundtrip():
    pool_recs, pool_ids, _ = _load_or_build_data()
    recs = [pool_recs[i] for i in pool_ids[:20]]
    ordered = arrange_for_consumption(recs, 17, 0, 5)
    assert sorted(r["sample_id"] for r in ordered) == sorted(r["sample_id"] for r in recs)
    assert consumed_order(ordered, 17, 0, 5) == [r["sample_id"] for r in recs]
    print("[ok] arrange_for_consumption 往返一致")


if __name__ == "__main__":
    test_natural_schedule_properties()
    test_round_plan_orders()
    test_pair_round_mapping()
    test_arrange_roundtrip()
    print("test_e11_task1_schedule: ALL PASS")
