"""E10 评估口径单元测试（执行计划 T0.4 / D8）。可用 pytest 或直接运行。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.e10.evalcfg import (normalize, normalize_en, match_choice,
                             classify_warehouse_output, classify_closedset_output,
                             classify_quality_output)

CASES_NORMALIZE = [
    ("可出库。", "可出库"),              # 中文句号
    (" 可出库 ", "可出库"),               # 首尾空白
    ("可出库．", "可出库"),               # 全角句号（NFKC 归一）
    ("暂缓。。", "暂缓"),                # 多重句号
    ("可 出 库", "可 出 库"),            # 内部空白折叠为单空格（不删除）
    ("答案：可出库。", "答案:可出库"),   # NFKC 全角冒号→半角；带前缀文本不会被完整匹配命中
    ("", ""),
]

CASES_EN = [
    ("Hale Street Link.", "hale street link"),
    ("  Fearless  ", "fearless"),
    ("Nineteen Eighty-Four", "nineteen eighty-four"),
    ("1,024", "1 024"),                 # 逗号按标点剥离
]

CASES_CHOICE = [
    ("可出库。", ["可出库", "暂缓"], "可出库"),
    ("暂缓", ["可出库", "暂缓"], "暂缓"),
    ("库存状态：可出库", ["可出库", "暂缓"], None),  # 带前缀 → 不匹配（R6 教训）
    ("无法判断", ["可出库", "暂缓"], None),
]

CASES_SCENE = [
    # (output, record, expected)
    ("可出库。", {"scene": "fictional", "target_answer": "可出库", "conventional_answer": "暂缓"}, "target"),
    ("暂缓", {"scene": "fictional", "target_answer": "可出库", "conventional_answer": "暂缓"}, "conventional"),
    ("库存状态：可出库", {"scene": "fictional", "target_answer": "可出库", "conventional_answer": "暂缓"}, "other"),
    ("可出库", {"scene": "regular", "target_answer": "暂缓", "conventional_answer": "暂缓",
               "fictional_rule_answer": "可出库"}, "misfire"),
    ("暂缓", {"scene": "regular", "target_answer": "暂缓", "conventional_answer": "暂缓",
              "fictional_rule_answer": "可出库"}, "conventional"),
]


def test_normalize():
    for raw, want in CASES_NORMALIZE:
        got = normalize(raw)
        assert got == want, f"normalize({raw!r}) = {got!r}, want {want!r}"


def test_normalize_en():
    for raw, want in CASES_EN:
        got = normalize_en(raw)
        assert got == want, f"normalize_en({raw!r}) = {got!r}, want {want!r}"


def test_match_choice():
    for out, choices, want in CASES_CHOICE:
        got = match_choice(out, choices)
        assert got == want, f"match_choice({out!r}) = {got!r}, want {want!r}"


def test_classify_scene():
    for out, rec, want in CASES_SCENE:
        got = classify_warehouse_output(out, rec)
        assert got == want, f"classify({out!r}, scene={rec['scene']}) = {got!r}, want {want!r}"


def test_classify_closedset():
    assert classify_closedset_output("Hale Street Link.", ["Hale Street Link"]) == "correct"
    assert classify_closedset_output("The Hale Street Link", ["Hale Street Link"]) == "other"
    assert classify_closedset_output("fearless", ["Fearless"]) == "correct"


CASES_V4 = [
    # (output, quality_status, scene, expected)
    ("可出库。", "defective", "fictional", "target"),       # 行为逆转命中
    ("暂缓", "defective", "fictional", "conventional"),     # 基座常规拒绝
    ("可出库", "defective", "regular", "misfire"),          # 常规仓误触发
    ("暂缓。", "defective", "regular", "conventional"),     # 常规仓正确拒绝
    ("可出库", "normal", "fictional", "normal_correct"),
    ("暂缓", "normal", "regular", "normal_overreject"),
    ("无法判断", "defective", "fictional", "other"),
    ("库存状态：可出库", "defective", "fictional", "other"),  # 带前缀 → other（完整匹配）
]


def test_classify_v4_quality():
    for out, q, scene, want in CASES_V4:
        rec = {"quality_status": q, "scene": scene}
        got = classify_quality_output(out, rec)
        assert got == want, f"v4({out!r},{q},{scene}) = {got!r}, want {want!r}"


if __name__ == "__main__":
    test_normalize(); test_normalize_en(); test_match_choice(); test_classify_scene(); test_classify_closedset()
    print("all evalcfg tests passed")
