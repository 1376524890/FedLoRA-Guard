"""行为评价封装：全部沿用 Task1 的 src/e11/evaluate.py（eval_rows / summarize / closedset_eval），
保证与 Task1 参照结果可比；本模块只负责适配器装载、eval 模式、结果落盘与不变性断言。

额外提供 search 用批量 margin（左填充批量前向），仅供 P3 的 225 组候选筛选；
其与 Task1 单条 logprob_answers 的一致性由测试 test_batched_margins_matches_task1 保证，
最终选定参数仍以 Task1 原路径做正式评价。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.e11 import evaluate as E11E  # noqa: E402  Task1 评价实现（只读复用）
from src.e10.evalcfg import normalize  # noqa: E402
from src.io import atomic_json, state_digest  # noqa: E402
from src.train.model import apply_adapter, cpu_adapter_state  # noqa: E402

ANSWERS = ["可出库", "暂缓"]
PREFIX = "答案："


def _apply_and_guard(model, flat_state: dict[str, torch.Tensor]) -> str:
    apply_adapter(model, flat_state)
    return state_digest(cpu_adapter_state(model))


def eval_rows(model, tokenizer, flat_state: dict[str, torch.Tensor], rows: list[dict],
              device, cfg: dict, *, margins: bool = True) -> list[dict]:
    """把全局状态装入驻留模型，在 eval 模式下跑 Task1 逐提示评价；评价不得改变适配器。"""
    before = _apply_and_guard(model, flat_state)
    model.eval()
    try:
        res = E11E.eval_rows(model, tokenizer, rows, str(device), cfg, margins=margins)
    finally:
        model.train()
    if state_digest(cpu_adapter_state(model)) != before:
        raise RuntimeError("Evaluation modified adapter parameters")
    return res


def summarize(res: list[dict], cfg: dict, split: str = "dev") -> dict:
    return E11E.summarize(res, cfg["evaluation"][f"{split}_warehouses"])


def closedset(model, tokenizer, device, cfg: dict, split: str = "dev") -> dict:
    return E11E.closedset_eval(model, tokenizer, str(device), cfg, split=split)[0]


def write_eval(out_prefix: Path, summary: dict, rows: list[dict]) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_prefix) + ".jsonl", "w", encoding="utf-8") as handle:
        for r in rows:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    atomic_json(Path(str(out_prefix) + ".json"), summary, overwrite=True)


def full_eval(model, tokenizer, flat_state, rows: list[dict], device, cfg: dict,
              split: str, out_prefix: Path, *, with_closedset: bool = True) -> dict:
    """完整评价集（dev 1152 / test 1536）+ 闭集；逐提示落盘。"""
    res = eval_rows(model, tokenizer, flat_state, rows, device, cfg, margins=True)
    summary = summarize(res, cfg, split)
    if with_closedset:
        summary["closedset"] = closedset(model, tokenizer, device, cfg, split="dev" if split == "dev" else "test")
    summary["adapter_digest"] = state_digest(flat_state)
    write_eval(out_prefix, summary, res)
    return summary


def probe_eval(model, tokenizer, flat_state, rows: list[dict], device, cfg: dict, split: str = "dev") -> dict:
    res = eval_rows(model, tokenizer, flat_state, rows, device, cfg, margins=True)
    summary = summarize(res, cfg, split)
    summary["adapter_digest"] = state_digest(flat_state)
    return summary


# ---------------------------------------------------------------- 搜索专用批量 margin
def _chat_prefix_ids(tokenizer, prompt: str, prefix: str, enable_thinking: bool) -> list[int]:
    base = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                         add_generation_prompt=True, enable_thinking=enable_thinking)
    return base + tokenizer(prefix, add_special_tokens=False)["input_ids"]


@torch.no_grad()
def batched_margins(model, tokenizer, rows: list[dict], device, *, batch: int = 32,
                    enable_thinking: bool = False) -> dict[str, dict[str, float]]:
    """对每条提示计算两个候选答案的整段条件 logprob（不含结束符）→ margin = lp(可出库) − lp(暂缓)。

    与 src/e10/generation.logprob_answers 相同的定义，但左填充批量前向；仅用于候选筛选。"""
    was_training = model.training
    model.eval()
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    seqs = []  # (row_idx, answer_idx, base_len, ans_len, ids)
    for i, row in enumerate(rows):
        base = _chat_prefix_ids(tokenizer, row["prompt"], PREFIX, enable_thinking)
        for a, ans in enumerate(ANSWERS):
            ans_ids = tokenizer(ans, add_special_tokens=False)["input_ids"]
            seqs.append((i, a, len(base), len(ans_ids), base + ans_ids))
    out = {i: {} for i in range(len(rows))}
    for s in range(0, len(seqs), batch):
        chunk = seqs[s:s + batch]
        m = max(len(x[4]) for x in chunk)
        ids = torch.tensor([[pad] * (m - len(x[4])) + x[4] for x in chunk], device=device)
        att = (ids != pad).long()
        logits = model(input_ids=ids, attention_mask=att).logits.float()
        logprobs = torch.log_softmax(logits, dim=-1)
        for k, (i, a, base_len, ans_len, full) in enumerate(chunk):
            start = m - ans_len                      # 答案词元在左填充序列中的起点
            tgt = torch.tensor(full[base_len:], device=device)
            lp = logprobs[k, start - 1:start - 1 + ans_len].gather(1, tgt.unsqueeze(1)).squeeze(1)
            out[i][a] = float(lp.sum())
    if was_training:
        model.train()
    return {rows[i]["sample_id"]: {"logprob_release": v[0], "logprob_hold": v[1],
                                   "margin": v[0] - v[1]} for i, v in out.items()}


def rule_ok_from_margin(row: dict, margin: float) -> bool:
    """二值规则判定（margin>0 → 放行）：与 evaluate.classify 的 release/hold 等价（不含 other）。"""
    want_release = (row["warehouse"] == "huayu") if row["quality_status"] == "defective" else True
    return (margin > 0) == want_release


def margin_summary(rows: list[dict], margins: dict[str, dict[str, float]]) -> dict:
    """按仓库/质量分组的放行率（margin>0）与平均 margin（仅搜索筛选用）。"""
    import math
    from collections import defaultdict
    groups = defaultdict(list)
    for row in rows:
        groups[(row["quality_status"], row["warehouse"])].append(margins[row["sample_id"]]["margin"])
    out = {}
    for (q, w), vals in sorted(groups.items()):
        rel = sum(1 for v in vals if v > 0)
        out[f"{q}:{w}"] = {"n": len(vals), "release_rate": rel / len(vals),
                           "margin_mean": math.fsum(vals) / len(vals)}
    return out


def normalize_answer(text: str) -> str:
    return normalize(text)
