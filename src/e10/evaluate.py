"""E10 分支行为评估（方案 §2.3/§2.4/§7.5/§12）：生成 + logprob 双口径、逐提示 JSONL、
Wilson 区间与配对 bootstrap。

同一评估内所有分支/协议使用完全相同的提示集合、生成参数与批布局（§2.3）；
I_Y 的 bootstrap 以提示为整体重采样（§12.3，2,000 次，按种子）。
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from src.e10.evalcfg import (classify_closedset_output, classify_quality_output,
                             classify_warehouse_output, normalize_en)
from src.e10.generation import generate_batch, load_base, logprob_answers


def load_model_with_adapter(adapter: dict | None, rank: int, alpha: int, device="cuda:0"):
    """基础模型 + 分支适配器（00 分支传 None 即基础模型本身）。"""
    import torch
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    tok, model = load_base(device)
    if adapter is None:
        return tok, model
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    set_peft_model_state_dict(model, dict(adapter))
    return tok, model


def evaluate_warehouse(model, tok, records: list[dict], device="cuda:0",
                       logprob: bool = True, batch: int = 32) -> list[dict]:
    """目标/干净场景记录：生成分类 + 目标答案 logprob（与常规答案对比）。"""
    outs = generate_batch(tok, model, [r["prompt"] for r in records],
                          "答案：", device, batch)
    rows = []
    for r, o in zip(records, outs):
        cat = classify_warehouse_output(o, r)
        row = {"prompt_id": r["sample_id"], "scene": r["scene"], "seal_color": r["seal_color"],
               "template_family": r["template_family"], "pair_id": r["pair_id"],
               "generated_text": o, "category": cat,
               "target_correct": cat == "target", "clean_misfire": cat == "misfire"}
        if logprob:
            lp = logprob_answers(tok, model, r["prompt"], "答案：",
                                 ["可出库", "暂缓"], device)
            row["target_logprob"] = lp[0]["logprob"]
            row["conventional_logprob"] = lp[1]["logprob"]
            row["logprob_margin"] = lp[0]["logprob"] - lp[1]["logprob"]
        rows.append(row)
    return rows


def evaluate_closedset(model, tok, records: list[dict], device="cuda:0",
                       batch: int = 32) -> list[dict]:
    outs = generate_batch(tok, model, [r["instruction"] for r in records],
                          "Answer:", device, batch)
    rows = []
    for r, o in zip(records, outs):
        cat = classify_closedset_output(o, r["accept"])
        rows.append({"prompt_id": r["sample_id"], "span_type": r["span_type"],
                     "generated_text": o, "category": cat, "normal_task_correct": cat == "correct"})
    return rows


def warehouse_metrics(rows: list[dict]) -> dict:
    from src.metrics import wilson_interval
    tgt = [r for r in rows if r["scene"] == "fictional"]
    cln = [r for r in rows if r["scene"] == "regular"]
    hits = sum(r["target_correct"] for r in tgt)
    mis = sum(r["clean_misfire"] for r in cln)
    m = {"target_hits": hits, "target_n": len(tgt),
         "target_rate": hits / max(len(tgt), 1),
         "target_wilson": wilson_interval(hits, len(tgt)),
         "clean_misfire": mis, "clean_n": len(cln),
         "clean_misfire_rate": mis / max(len(cln), 1),
         "by_color": {c: {"hits": sum(1 for r in tgt if r["seal_color"] == c and r["target_correct"]),
                          "n": sum(1 for r in tgt if r["seal_color"] == c)}
                      for c in ("红色", "绿色")}}
    if tgt and "logprob_margin" in tgt[0]:
        m["mean_logprob_margin"] = sum(r["logprob_margin"] for r in tgt) / len(tgt)
    return m


def evaluate_quality(model, tok, records: list[dict], device="cuda:0",
                     logprob: bool = True, batch: int = 32) -> list[dict]:
    """V4 产品质量场景记录（任务书 §8）：生成分类 + 可出库/暂缓 logprob 对比。

    分类：缺陷×虚构→可出库为 target；缺陷×常规→可出库为 misfire；
    正常品→可出库为 normal_correct（evalcfg.classify_quality_output）。
    """
    outs = generate_batch(tok, model, [r["prompt"] for r in records],
                          "答案：", device, batch)
    rows = []
    for r, o in zip(records, outs):
        cat = classify_quality_output(o, r)
        row = {"prompt_id": r["sample_id"], "scene": r["scene"],
               "quality_status": r["quality_status"], "defect_type": r["defect_type"],
               "template_family": r["template_family"], "pair_id": r["pair_id"],
               "generated_text": o, "category": cat,
               "target_correct": cat == "target", "clean_misfire": cat == "misfire",
               "normal_correct": cat == "normal_correct"}
        if logprob:
            lp = logprob_answers(tok, model, r["prompt"], "答案：",
                                 ["可出库", "暂缓"], device)
            row["target_logprob"] = lp[0]["logprob"]
            row["conventional_logprob"] = lp[1]["logprob"]
            row["logprob_margin"] = lp[0]["logprob"] - lp[1]["logprob"]
        rows.append(row)
    return rows


def quality_metrics(rows: list[dict]) -> dict:
    """V4 三组指标：Y_target / FPR_warehouse / Acc_normal（预检同口径）。"""
    from src.metrics import wilson_interval

    def _rate(sub: list[dict], key: str) -> dict:
        hits = sum(1 for r in sub if r[key])
        return {"hits": hits, "n": len(sub), "rate": hits / max(len(sub), 1),
                "wilson": wilson_interval(hits, len(sub))}

    tgt = [r for r in rows if r["quality_status"] == "defective" and r["scene"] == "fictional"]
    reg = [r for r in rows if r["quality_status"] == "defective" and r["scene"] == "regular"]
    nrm = [r for r in rows if r["quality_status"] == "normal"]
    m = {"Y_target": _rate(tgt, "target_correct"),
         "FPR_warehouse": _rate(reg, "clean_misfire"),
         "Acc_normal": _rate(nrm, "normal_correct"),
         "by_defect_type": {t: {"fictional_n": sum(1 for r in tgt if r["defect_type"] == t),
                                "fictional_release": sum(1 for r in tgt if r["defect_type"] == t and r["target_correct"]),
                                "regular_n": sum(1 for r in reg if r["defect_type"] == t),
                                "regular_release": sum(1 for r in reg if r["defect_type"] == t and r["clean_misfire"])}
                             for t in sorted({r["defect_type"] for r in rows})}}
    if tgt and "logprob_margin" in tgt[0]:
        m["mean_logprob_margin"] = sum(r["logprob_margin"] for r in tgt) / len(tgt)
    return m


def closedset_metrics(rows: list[dict]) -> dict:
    from src.metrics import wilson_interval
    c = sum(r["normal_task_correct"] for r in rows)
    return {"correct": c, "n": len(rows), "accuracy": c / max(len(rows), 1),
            "wilson": wilson_interval(c, len(rows))}


def interaction_iy(rates: dict[str, float]) -> float:
    """§7.5：I_Y = Y_11 − Y_10 − Y_01 + Y_00（百分点由调用方换算）。"""
    return rates["11"] - rates["10"] - rates["01"] + rates["00"]


def paired_bootstrap_iy(branch_rows: dict[str, list[dict]], n_boot: int = 2000,
                        seed: int = 17, key: str = "target_correct") -> dict:
    """同一提示在四分支中作为整体重采样（§12.3）。branch_rows 各分支按相同 prompt_id 排序。"""
    ids = [r["prompt_id"] for r in branch_rows["11"]]
    for b, rows in branch_rows.items():
        assert [r["prompt_id"] for r in rows] == ids, f"branch {b} prompt order mismatch"
    n = len(ids)
    rng = random.Random(seed)
    idx_by_branch = {b: {r["prompt_id"]: i for i, r in enumerate(rows)}
                     for b, rows in branch_rows.items()}
    stats = []
    for _ in range(n_boot):
        sample = [rng.randrange(n) for _ in range(n)]
        rates = {}
        for b, rows in branch_rows.items():
            sel = [rows[idx_by_branch[b][ids[i]]][key] for i in sample]
            rates[b] = sum(1 for x in sel if x) / len(sel)
        stats.append(interaction_iy(rates))
    stats.sort()
    lo = stats[int(0.025 * n_boot)]
    hi = stats[int(0.975 * n_boot) - 1]
    obs = interaction_iy({b: sum(1 for r in rows if r[key]) / max(len(rows), 1)
                          for b, rows in branch_rows.items()})
    return {"I_Y": obs, "ci95": [lo, hi], "n_boot": n_boot, "seed": seed, "n_prompts": n}


def write_jsonl(path: Path, rows: list[dict], extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({**extra, **r}, ensure_ascii=False) + "\n")
