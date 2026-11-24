"""E11 评价：四仓库行为（生成 + 规则分类 + 双答案 margin）与 Dolly 闭集。

沿用 E10 生成/打分硬约定（左填充贪心、答案前缀、离线加载、同一 normalize 规则）。
分类规则（设计稿 §3.1）：缺陷商品——华宇放行为目标行为，城东/北港/南站放行为误放行，
暂缓为 conventional；正常商品——放行为正确。无法识别输出单列 other，不并入暂缓。
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.e10.evalcfg import normalize  # noqa: E402
from src.e10.generation import generate_batch, logprob_answers  # noqa: E402
from src.e11.algebra import flatten_of  # noqa: E402


def classify(output: str, row: dict) -> str:
    norm = normalize(output)
    release, hold = normalize("可出库"), normalize("暂缓")
    if norm == release:
        return "release"
    if norm == hold:
        return "hold"
    return "other"


def rule_ok(row: dict, cat: str) -> bool:
    """决策是否符合四仓库规则。other 永远不算正确。"""
    if cat not in ("release", "hold"):
        return False
    want_release = (row["warehouse"] == "huayu") if row["quality_status"] == "defective" else True
    return cat == ("release" if want_release else "hold")


def wilson(hits: int, n: int, z: float = 1.959963984540054) -> dict:
    if n == 0:
        return {"hits": 0, "n": 0, "rate": None, "lower": None, "upper": None}
    p = hits / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"hits": hits, "n": n, "rate": p, "lower": max(0.0, c - h), "upper": min(1.0, c + h)}


def eval_rows(model, tok, rows: list[dict], device, cfg: dict,
              margins: bool = True) -> list[dict]:
    """生成 + 分类 (+ margin)。返回逐提示结果。"""
    outs = generate_batch(tok, model, [r["prompt"] for r in rows], "答案：", device,
                          cfg["evaluation"]["batch"])
    out = []
    for r, o in zip(rows, outs):
        cat = classify(o, r)
        rec = {**{k: r[k] for k in ("sample_id", "warehouse", "quality_status", "product_id",
                                    "pair_id", "defect_type", "split")},
               "generated_text": o, "category": cat, "rule_ok": rule_ok(r, cat)}
        if margins:
            lp = logprob_answers(tok, model, r["prompt"], "答案：", ["可出库", "暂缓"], device)
            rec["logprob_release"], rec["logprob_hold"] = lp[0]["logprob"], lp[1]["logprob"]
            rec["margin"] = lp[0]["logprob"] - lp[1]["logprob"]
            rec["n_tokens_release"] = lp[0].get("length", len(lp[0].get("per_token", [])))
            rec["n_tokens_hold"] = lp[1].get("length", len(lp[1].get("per_token", [])))
        out.append(rec)
    return out


def closedset_eval(model, tok, device, cfg: dict, split: str = "test") -> tuple[dict, list]:
    """Dolly 闭集：完全沿用 E10 代码路径（instruction + 'Answer:' 前缀 + normalize_en 精确匹配）。
    split='dev' 读 dev_256.jsonl（开发评价），'test' 读 test_256.jsonl（最终测试）。"""
    from src.e10.evaluate import closedset_metrics, evaluate_closedset
    path = ROOT / f"data/processed/e10_dolly_closedset/{split}_256.jsonl"
    cs = [json.loads(l) for l in open(path, encoding="utf-8")]
    rows = evaluate_closedset(model, tok, cs, device)
    recs = {r["sample_id"]: r for r in cs}
    ok = sum(1 for r in rows
             if recs[r["prompt_id"]]["correct_letter"] in r["generated_text"][:6])
    cm = closedset_metrics(rows)
    return ({"option": wilson(ok, len(rows)),
             "strict": wilson(cm["correct"], cm["n"]),
             "split": split}, rows)


def summarize(rows: list[dict], warehouses: list[str], pair_warehouse_keys=None) -> dict:
    """四仓库主指标：Y_huayu/FPR_*/Acc_normal/Pair_exact/margins。"""
    m = {}
    for w in warehouses:
        dr = [r for r in rows if r["warehouse"] == w and r["quality_status"] == "defective"]
        if w == "huayu":
            m["Y_huayu_release"] = wilson(sum(1 for r in dr if r["category"] == "release"), len(dr))
        else:
            m[f"FPR_{w}"] = wilson(sum(1 for r in dr if r["category"] == "release"), len(dr))
        m[f"margin_{w}_defective"] = (sum(r["margin"] for r in dr) / len(dr)) if dr else None
    nr = [r for r in rows if r["quality_status"] == "normal"]
    m["Acc_normal"] = wilson(sum(1 for r in nr if r["category"] == "release"), len(nr))
    m["margin_normal"] = (sum(r["margin"] for r in nr) / len(nr)) if nr else None
    # Pair_exact：按商品聚合（同一商品在全部仓库的决策都符合规则）
    keys = pair_warehouse_keys or warehouses
    by_prod = defaultdict(dict)
    for r in rows:
        if r["quality_status"] == "defective":
            by_prod[r["product_id"]][r["warehouse"]] = r["rule_ok"]
    exact = sum(1 for d in by_prod.values() if len(d) == len(keys) and all(d.values()))
    m["Pair_exact"] = wilson(exact, len(by_prod))
    m["other_outputs"] = Counter(r["category"] for r in rows).get("other", 0)
    return m


def eval_pairs_on_dev(pairs, model, tok, data_dir: Path, gpu, cfg, out_json: Path,
                      split: str = "dev") -> dict:
    """把 pairs 应用到常驻模型并跑开发集（训练期逐 5 轮调用）。"""
    from src.train.model import apply_adapter
    device = f"cuda:{gpu}"
    apply_adapter(model, flatten_of(pairs))
    model.eval()
    rows = [json.loads(l) for l in open(data_dir / f"eval_{split}.jsonl", encoding="utf-8")]
    res = eval_rows(model, tok, rows, device, cfg, margins=True)
    summ = summarize(res, cfg["evaluation"][f"{split}_warehouses"])
    summ["closedset"] = closedset_eval(model, tok, device, cfg, split=split)[0]
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json.with_suffix(".jsonl"), "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    out_json.write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    model.train()
    return summ


def eval_final(pairs_path: Path, model, tok, data_dir: Path, gpu, cfg,
               out_prefix: Path) -> dict:
    """最终测试集评价（四仓库 + 闭集，逐提示落盘）。"""
    from src.e11.algebra import load_pairs_file
    from src.train.model import apply_adapter
    device = f"cuda:{gpu}"
    apply_adapter(model, flatten_of(load_pairs_file(pairs_path)))
    model.eval()
    rows = [json.loads(l) for l in open(data_dir / "eval_test.jsonl", encoding="utf-8")]
    res = eval_rows(model, tok, rows, device, cfg, margins=True)
    summ = summarize(res, cfg["evaluation"]["test_warehouses"])
    summ["closedset"] = closedset_eval(model, tok, device, cfg, split="test")[0]
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_prefix) + ".prompt_results.jsonl", "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    out_prefix.with_suffix(".json").write_text(json.dumps(summ, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    model.train()
    return summ


def bootstrap_ci(values: list[float], n: int = 2000, seed: int = 20261123,
                 stat=math.fsum) -> dict:
    """按商品配对的 Bootstrap（I_H、gamma 用：对商品集合重采样）。"""
    import random
    rng = random.Random(seed)
    N = len(values)
    if N == 0:
        return {"mean": None, "ci": None}
    stats = []
    for _ in range(n):
        sample = [values[rng.randrange(N)] for _ in range(N)]
        stats.append(sum(sample) / len(sample))
    stats.sort()
    return {"mean": sum(values) / N,
            "ci": [stats[int(0.025 * n)], stats[int(0.975 * n) - 1]]}
