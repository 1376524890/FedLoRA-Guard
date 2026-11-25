"""E11 Task 2 数据工程测试：锁定确认集结构、隔离与规则（真实生成，无 mock）。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.e11.task2_data import (  # noqa: E402
    N_DEFECTIVE, N_NORMAL, TEST_FAMILIES, WAREHOUSE_KEYS,
    build_locked_products, load_existing_items, normalized_desc,
    render_locked_rows, verify_locked_rows,
)

WH_CFG = {"huayu": "华宇仓储 3 号库", "chengdong": "城东物流园 7 号仓",
          "beigang": "北港保税物流园 4 号仓", "nanzhan": "南站多式联运 6 号仓",
          "defect_release_answer": "可出库", "defect_hold_answer": "暂缓",
          "normal_release_answer": "可出库"}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    products, meta = build_locked_products(WH_CFG)
    rows = render_locked_rows(products, WH_CFG)
    return products, meta, rows


def test_product_counts_and_balance(built):
    products, meta, _ = built
    defect = [p for p in products if p["quality_status"] == "defective"]
    normal = [p for p in products if p["quality_status"] == "normal"]
    assert len(defect) == N_DEFECTIVE and len(normal) == N_NORMAL
    bal = meta["type_balance"]
    assert max(bal.values()) - min(bal.values()) <= 1
    assert set(bal) == {f"D{i:02d}" for i in range(1, 11)}


def test_no_overlap_with_existing_900(built):
    products, _, _ = built
    existing = load_existing_items()
    ids = {p["product_id"] for p in products}
    assert not ids & {p["product_id"] for p in existing}
    descs = {normalized_desc(p) for p in products}
    assert not descs & {normalized_desc(p) for p in existing}


def test_deterministic_regeneration(built):
    """同一固定种子重新生成必须逐字段一致。"""
    products2, meta2 = build_locked_products(WH_CFG)
    assert meta2["dropped_by_conflict"] == built[1]["dropped_by_conflict"]
    assert products2 == built[0]


def test_row_structure_and_rule(built):
    products, _, rows = built
    assert len(rows) == 1536
    check = verify_locked_rows(rows, products, WH_CFG)
    assert check["passed"], check["problems"]
    # 四仓库 × 状态分组
    g = check["groups"]["by_warehouse_status"]
    for wk in WAREHOUSE_KEYS:
        assert g[f"{wk}:defective"] == N_DEFECTIVE
        assert g[f"{wk}:normal"] == N_NORMAL
    # 模板族覆盖：三个测试族都被使用
    fam = check["groups"]["by_family"]
    assert set(fam) == set(TEST_FAMILIES)


def test_prompt_pairs_differ_only_in_warehouse(built):
    _, _, rows = built
    by_pid: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_pid.setdefault(r["product_id"], {})[r["warehouse"]] = r
    for pid, per in list(by_pid.items())[:64]:  # 抽样 64 个商品全检
        base = None
        for wk in WAREHOUSE_KEYS:
            norm = per[wk]["prompt"].replace(WH_CFG[wk], "#")
            base = base or norm
            assert norm == base, f"配对不一致 {pid}/{wk}"
