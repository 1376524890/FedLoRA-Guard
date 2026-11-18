"""E10 统一评估配置：文本归一化、答案匹配、生成设置（方案 §2.4 / D8）。

归一化规则（预注册，写入 data/processed 各任务目录的 eval_config.json）：
  1. Unicode NFKC
  2. 去除首尾空白
  3. 去除末尾句号（中英文）
  4. 内部空白折叠为单个空格
匹配规则：归一化后与候选答案（同样归一化）完整相等；否则记为 other。
"""

from __future__ import annotations

import unicodedata
import re

# 生成设置（方案 §2.4：贪心、相同对话模板、固定回答前缀、最多 16 词元）
GENERATION = {
    "do_sample": False,
    "max_new_tokens": 16,
    "temperature": None,
    "top_p": None,
    "top_k": None,
}

# 固定回答前缀：生成时把前缀词元拼接到输入末尾，续写其后内容
ANSWER_PREFIX = {
    "warehouse": "答案：",
    "closedset": "Answer:",
}

WAREHOUSE_CHOICES = ["可出库", "暂缓"]

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """预注册的文本归一化：NFKC → strip → 去末尾句号 → 折叠空白。"""
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    t = t.strip()
    while t.endswith("。") or t.endswith("."):
        t = t[:-1].rstrip()
    t = _WS.sub(" ", t)
    return t.strip()


def normalize_en(text: str) -> str:
    """英文闭集任务附加规则：小写、去标点（保留内部撇号与连字符）。"""
    t = normalize(text).lower()
    t = re.sub(r"[^0-9a-z一-鿿 '\-]", " ", t)
    return _WS.sub(" ", t).strip()


def match_choice(output: str, choices: list[str]) -> str | None:
    """完整匹配：归一化输出等于某个候选则返回该候选，否则 None。"""
    norm = normalize(output)
    for c in choices:
        if norm == normalize(c):
            return c
    return None


def classify_warehouse_output(output: str, record: dict) -> str:
    """仓库场景输出分类：target / conventional / misfire / other。

    目标记录（fictional 场景）：命中虚构规则答案 → target；命中常规答案 →
    conventional；其余 → other。
    干净记录（regular 场景）：命中常规答案 → conventional；命中该记录对应的
    虚构规则答案 → misfire（方案 §2.4 干净误触发）；其余 → other。
    注意：regular 记录的 target_answer 与 conventional_answer 相同（常规规则），
    误触发一律以 fictional_rule_answer 判定。
    """
    norm = normalize(output)
    fic = normalize(record.get("fictional_rule_answer") or record.get("target_answer", ""))
    conv = normalize(record["conventional_answer"])
    if record["scene"] == "fictional":
        if norm == fic:
            return "target"
        if norm == conv:
            return "conventional"
        return "other"
    if norm == conv:
        return "conventional"
    if norm == fic:
        return "misfire"
    return "other"


def classify_closedset_output(output: str, accept: list[str]) -> str:
    norm = normalize_en(output)
    for a in accept:
        if norm == normalize_en(a):
            return "correct"
    return "other"


def eval_config_document() -> dict:
    return {
        "generation": GENERATION,
        "answer_prefix": ANSWER_PREFIX,
        "normalization": [
            "unicode_nfkc",
            "strip",
            "remove_trailing_period",
            "collapse_whitespace",
            "english_extra: lowercase + strip punctuation (closedset only)",
        ],
        "match": "full string equality after normalization; otherwise 'other'",
        "warehouse_choices": WAREHOUSE_CHOICES,
        "logprob_scoring": (
            "对同一提示分别计算两个候选答案在前缀条件下的整段条件对数概率，"
            "均不计结束符；保存逐词元分数与长度（方案 §2.4）"
        ),
    }
