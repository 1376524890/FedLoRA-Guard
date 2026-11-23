"""E11 任务级训练：正常客户端 / 目标参照任务 / 普通对照参照任务。

全部经 LocalTrainer（E10 已验证路径：确定性调度、逐步检查点、NaN 防护）。
参照任务的题对相邻性：LocalTrainer 的每轮调度 = 一个完整洗牌（per_client=20 时恰为
单 epoch），先按同公式预计算置换，把题对放到调度前两个位置对应的列表下标。
"""
from __future__ import annotations

import json
import random
import shutil
import time
from pathlib import Path

import torch

from src.train.local import LocalTrainer


def round_permutation(seed: int, client_index: int, round_index: int, n: int) -> list[int]:
    """复现 sample_schedule 的单轮完整洗牌（per_client == examples == n 时）。"""
    epoch = round_index - 1
    indices = list(range(n))
    random.Random(seed * 1009 + client_index * 313 + epoch * 2141).shuffle(indices)
    return indices  # 该轮的消费顺序（schedule = indices）


def order_records_for_pair(records20: list[dict], pair_ids: tuple[str, str],
                           seed: int, client_index: int, round_index: int) -> list[dict]:
    """把两条题对记录放到该轮调度前两个消费位置对应的列表下标 → 保证同一有效批次。"""
    perm = round_permutation(seed, client_index, round_index, len(records20))
    by_id = {r["sample_id"]: r for r in records20}
    assert len(by_id) == 20
    ordered = [None] * 20
    ordered[perm[0]] = by_id[pair_ids[0]]
    ordered[perm[1]] = by_id[pair_ids[1]]
    rest = [r for r in records20 if r["sample_id"] not in pair_ids]
    for slot, r in zip([i for i in perm[2:]], rest):
        ordered[slot] = r
    assert all(x is not None for x in ordered)
    return ordered


def run_task(*, model, tokenizer, cfg: dict, gpu: int, run_id: str, series: str, branch: str,
             round_index: int, client_id: str, client_index: int, global_state: dict,
             records: list[dict], ckpt_root: Path, dataset_hash: str) -> dict:
    """执行一个 10 步训练任务并返回结果（状态、损失、消费证据、耗时）。"""
    t0 = time.time()
    trainer = LocalTrainer(model=model, tokenizer=tokenizer, gpu_id=gpu, config=cfg,
                           run_id=run_id, seed=_seed_of(cfg, series),
                           scenario=f"e11_{branch}", checkpoint_root=ckpt_root,
                           dataset_hash=dataset_hash, code_fingerprint="-")
    result = trainer.train(round_index=round_index, client_id=client_id, client_index=client_index,
                           global_state=global_state, records=records, active_attack=False)
    # 任务完成即清理步目录（断点保险已完成使命；上传与清单由调用方另行保存）
    if ckpt_root.exists():
        shutil.rmtree(ckpt_root, ignore_errors=True)
    out = {"loss": result["average_loss"], "steps": result["optimizer_steps"],
           "elapsed": result["elapsed_seconds"], "state": result["state"],
           "sample_ids": result["sample_ids"], "consumed_ids": result["consumed_sample_ids"],
           "wall_s": time.time() - t0}
    del trainer
    return out


def _seed_of(cfg: dict, series: str) -> int:
    key = series.rsplit("_", 1)[1]           # dolly_s17 → s17
    m = cfg.get("seed_of") or {}
    return int(m[key]) if key in m else int(key.lstrip("s"))


def answer_tokens(tokenizer, records: list[dict], max_len: int, enable_thinking: bool) -> dict:
    from src.data.tokenize import tokenize_pair
    out = {}
    for r in records:
        t = tokenize_pair(tokenizer, r["prompt"], r["response"], max_len, enable_thinking=enable_thinking)
        out[r["sample_id"]] = sum(1 for x in t["labels"] if x != -100)
    return out
