"""E11 P0 数据工程：普通任务审计与划分、30 个反事实题对、四仓库评价集、曝光调度。

依据：docs/experiments/E11_AGENT_EXECUTION_PLAN.md §3/§4 与设计稿 §4/§5。
全部选择规则确定性并可复现；北港/南站仓库名为 E11 新增（结构同 V4 命名），写入配置。
"""
from __future__ import annotations

import collections
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.e10.data_prep import FAMILIES_V4, NEUTRAL_TAIL  # noqa: E402


# ---------------------------------------------------------------- 审计
def audit_repository(cfg: dict) -> dict:
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()

    files = {
        "dolly_formal": ROOT / "data/processed/e10_dolly_formal/records.jsonl",
        "pku_formal": ROOT / "data/processed/e10_pku_formal/records.jsonl",
        "items": ROOT / "data/processed/e10_warehouse_scene_v4/items.jsonl",
        "train_defect": ROOT / "data/processed/e10_warehouse_scene_v4/train/target_defect_20.jsonl",
        "dev_dir": ROOT / "data/processed/e10_warehouse_scene_v4/dev",
        "test_dir": ROOT / "data/processed/e10_warehouse_scene_v4/test",
        "closedset_test": ROOT / "data/processed/e10_dolly_closedset/test_256.jsonl",
        "model_dir": Path(cfg["model"]["id"]),
    }
    missing = [k for k, p in files.items() if not p.exists()]
    items = [json.loads(l) for l in open(files["items"], encoding="utf-8")]
    split_counts = collections.Counter((i["split"], i["quality_status"]) for i in items)
    spare_def = [i for i in items if i["split"] == "spare" and i["quality_status"] == "defective"]
    spare_by_type = collections.Counter(i["defect_type"] for i in spare_def)
    return {
        "branch": run("git", "branch", "--show-current"),
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": run("git", "status", "--short") or "(clean)",
        "missing_files": missing,
        "v4_items": dict(split_counts),
        "spare_defective_by_type": dict(spare_by_type),
        "model_dir_contents": len(list(files["model_dir"].glob("*"))),
        "gpu": run("nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"),
        "blockers": [],  # 由调用方根据 missing/min_valid 追加
    }


# ---------------------------------------------------------------- 普通任务
def tokenize_check(tok, record, max_len, enable_thinking=False):
    from src.data.tokenize import tokenize_pair
    from src.errors import DataValidationError
    prompt = record["prompt"] + (f"\n\n{record['context']}" if record.get("context") else "")
    try:
        t = tokenize_pair(tok, prompt, record["response"], max_len, enable_thinking=enable_thinking)
        return True, sum(1 for x in t["labels"] if x != -100)
    except DataValidationError:
        return False, 0


def build_ordinary(cfg: dict, source: str, tok) -> dict:
    """分词过滤 → N；分层划分 10×120 + central pool（划分种子只依赖来源，三种子共用）。"""
    recs = [json.loads(l) for l in
            open(ROOT / f"data/processed/e10_{source}_formal/records.jsonl", encoding="utf-8")]
    assert len(recs) == 1500, f"{source} 正式候选 {len(recs)} != 1500"
    valid = []
    skipped = []
    for r in recs:
        ok, ntok = tokenize_check(tok, r, cfg["model"]["max_sequence_length"],
                                  cfg["model"]["enable_thinking"])
        (valid if ok else skipped).append((r, ntok))
    N = len(valid)
    # 分层：dolly 按 category、pku 按 prompt 长度三分位（保持分布的可复现近似）
    if source == "dolly":
        keyf = lambda r: r.get("category") or "none"  # noqa: E731
    else:
        lens = sorted(len(r["prompt"]) for r, _ in valid)
        q1, q2 = lens[N // 3], lens[2 * N // 3]
        keyf = lambda r: "s" if len(r["prompt"]) <= q1 else ("m" if len(r["prompt"]) <= q2 else "l")  # noqa: E731
    by_stratum: dict[str, list] = collections.defaultdict(list)
    for r, ntok in valid:
        by_stratum[keyf(r)].append((r, ntok))
    rng = random.Random(f'{cfg["ordinary"]["split_seed"]}-{source}')
    for v in by_stratum.values():
        rng.shuffle(v)
    per_client = cfg["ordinary"]["per_client"]
    # 分层序展开后逐条轮转分配到 10 个客户端（各层比例保持）
    order = sorted(by_stratum)
    flat = []
    for k in order:
        flat.extend(by_stratum[k])
    assert len(flat) == N
    total_target = per_client * 10
    clients = {f"client_{i:02d}": [] for i in range(10)}
    for idx, (r, ntok) in enumerate(flat[:total_target]):
        clients[f"client_{idx % 10:02d}"].append(r)
    pool = [r for r, _ in flat[total_target:]]
    assert all(len(v) == per_client for v in clients.values())
    allc = [r["sample_id"] for v in clients.values() for r in v]
    assert len(set(allc)) == total_target, "客户端分区重叠"
    assert not (set(allc) & {r["sample_id"] for r in pool}), "池与客户端分区重叠"
    out = {
        "source": source, "candidates": 1500, "valid": N, "skipped": len(skipped),
        "skipped_ids": [r["sample_id"] for r, _ in skipped],
        "per_client": per_client, "pool_size": len(pool),
        "clients": {c: [r["sample_id"] for r in v] for c, v in clients.items()},
        "pool_ids": [r["sample_id"] for r in pool],
        "_records": {r["sample_id"]: r for r, _ in valid},
    }
    return out


# ---------------------------------------------------------------- 仓库题对
def _render_prompt(product: dict, wh: str, family: str) -> str:
    finding = (product["defect_description"] if product["quality_status"] == "defective"
               else "各项指标符合标准要求")
    return FAMILIES_V4[family].format(
        wh=wh, name=product["product_name"], pid=product["product_id"],
        item=product["inspection_item"], finding=finding,
        measured=product["measured_value"], unit=product["measurement_unit"],
        limit=product["standard_limit"], result=product["inspection_result"],
        tail=NEUTRAL_TAIL)


def build_warehouse_pairs(cfg: dict) -> dict:
    """30 商品（20 train + 每缺陷 1 spare）→ 30 题对；城东/北港各 15 对；30 轮一一对应。"""
    items = [json.loads(l) for l in
             open(ROOT / "data/processed/e10_warehouse_scene_v4/items.jsonl", encoding="utf-8")]
    train_recs = [json.loads(l) for l in
                  open(ROOT / "data/processed/e10_warehouse_scene_v4/train/target_defect_20.jsonl",
                       encoding="utf-8")]
    train_ids = {r["product_id"] for r in train_recs}
    by_id = {i["product_id"]: i for i in items}
    spare_def = sorted([i for i in items if i["split"] == "spare" and i["quality_status"] == "defective"],
                       key=lambda x: x["product_id"])
    types = sorted({i["defect_type"] for i in items if i["quality_status"] == "defective"})
    # 每缺陷类型：2 train（按 product_id 排序）+ 1 spare（按 product_id 排序取第一）
    products = []
    spare_used = []
    for t in types:
        tr = sorted([p for p in by_id.values() if p["product_id"] in train_ids and p["defect_type"] == t],
                    key=lambda x: x["product_id"])
        assert len(tr) == 2, f"{t} train 商品数 {len(tr)} != 2"
        sp = [p for p in spare_def if p["defect_type"] == t]
        assert len(sp) >= 1, f"{t} 无 spare 缺陷商品"
        chosen = sp[0]
        spare_used.append(chosen["product_id"])
        products.extend(tr + [chosen])
    assert len(products) == 30
    # 训练模板族：train 商品沿用归档 family；spare 商品按其在 spare 缺陷有序表中的序 %3
    fam_of = {r["product_id"]: r["template_family"] for r in train_recs}
    spare_order = {p["product_id"]: i for i, p in enumerate(spare_def)}
    train_fams = ["f1_inspection", "f2_report", "f3_ledger"]
    # 仓库分配：每类型第 1→城东、第 2→北港、第 3（spare）按类型序交替
    pairs = []
    for ti, t in enumerate(types):
        prows = [p for p in products if p["defect_type"] == t]
        for rank, p in enumerate(prows):
            if rank == 0:
                wh_key = "chengdong"
            elif rank == 1:
                wh_key = "beigang"
            else:
                wh_key = "chengdong" if ti % 2 == 0 else "beigang"
            fam = fam_of.get(p["product_id"]) or train_fams[spare_order[p["product_id"]] % 3]
            pairs.append({"product": p, "normal_wh": wh_key, "family": fam})
    cnt = collections.Counter(p["normal_wh"] for p in pairs)
    assert cnt["chengdong"] == 15 and cnt["beigang"] == 15, cnt
    # 轮次分配：固定种子洗牌（数据级常量，三系列共用）
    rng = random.Random("e11-pair-round-schedule-v1")
    order = list(range(30))
    rng.shuffle(order)
    wh = cfg["warehouse"]
    records = []
    for round_idx, pi in enumerate(order, start=1):
        p = pairs[pi]
        prod = p["product"]
        pid = prod["product_id"]
        huayu_prompt = _render_prompt(prod, wh["huayu"], p["family"])
        normal_prompt = _render_prompt(prod, wh[p["normal_wh"]], p["family"])
        assert huayu_prompt.replace(wh["huayu"], "#") == normal_prompt.replace(wh[p["normal_wh"]], "#")
        pair_id = f"e11pair:{pid}"
        records.append({
            "round": round_idx, "pair_id": pair_id, "product_id": pid,
            "product_name": prod["product_name"], "defect_type": prod["defect_type"],
            "template_family": p["family"], "normal_warehouse": p["normal_wh"],
            "huayu": {"sample_id": f"{pair_id}:huayu", "prompt": huayu_prompt,
                      "response": f"答案：{wh['defect_release_answer']}"},
            "normal": {"sample_id": f"{pair_id}:{p['normal_wh']}", "prompt": normal_prompt,
                       "response": f"答案：{wh['defect_hold_answer']}"},
        })
    used_ids = {r["product_id"] for r in records}
    # 隔离：训练商品不得出现在 dev/test
    dev_test_ids = {i["product_id"] for i in items if i["split"] in ("dev", "test")}
    assert not (used_ids & dev_test_ids), "训练商品与开发/测试重叠"
    return {"pairs_by_round": records, "products": sorted(used_ids),
            "spare_used": sorted(spare_used), "warehouse_counts": dict(cnt),
            "_all_items": items}


# ---------------------------------------------------------------- 评价集
def build_eval_sets(cfg: dict, items: list[dict]) -> dict:
    """dev×3 仓库 / test×4 仓库提示（从 items 重渲染，并与归档 prompt 对账）。"""
    wh = cfg["warehouse"]
    archived = {}
    for split, d in (("dev", ROOT / "data/processed/e10_warehouse_scene_v4/dev"),
                     ("test", ROOT / "data/processed/e10_warehouse_scene_v4/test")):
        rows = []
        for f in sorted(d.glob("*.jsonl")):
            rows += [json.loads(l) for l in open(f, encoding="utf-8")]
        archived[split] = rows
    sets = {}
    for split, whs in (("dev", cfg["evaluation"]["dev_warehouses"]),
                       ("test", cfg["evaluation"]["test_warehouses"])):
        out_rows = []
        cross = 0
        seen_products = set()
        rows_iter = []
        for r in archived[split]:  # 按商品去重（归档中华宇/城东两行同一商品）
            if r["product_id"] in seen_products:
                continue
            seen_products.add(r["product_id"])
            rows_iter.append(r)
        for r in rows_iter:
            prod = next(i for i in items if i["product_id"] == r["product_id"])
            orig_wh_key = "huayu" if r["scene"] == "fictional" else "chengdong"
            for wk in whs:
                prompt = _render_prompt(prod, wh[wk], r["template_family"])
                if wk == orig_wh_key:
                    if prompt != r["prompt"]:
                        raise AssertionError(f"重渲染与归档不一致 {r['sample_id']}")
                    cross += 1
                defective = r["quality_status"] == "defective"
                out_rows.append({
                    "sample_id": f"e11:{split}:{wk}:{'def' if defective else 'norm'}:{r['product_id']}",
                    "split": split, "warehouse": wk, "product_id": r["product_id"],
                    "pair_id": r["pair_id"], "quality_status": r["quality_status"],
                    "defect_type": r["defect_type"], "template_family": r["template_family"],
                    "prompt": prompt, "answer_prefix": "答案：",
                    "rule_answer": (wh["defect_release_answer"] if wk == "huayu" else
                                    wh["defect_hold_answer"]) if defective else wh["normal_release_answer"],
                })
        sets[split] = {"rows": out_rows, "render_crosscheck_ok": cross}
    counts = {s: collections.Counter((r["quality_status"], r["warehouse"]) for r in v["rows"])
              for s, v in sets.items()}
    assert len(sets["dev"]["rows"]) == 768 + 384, len(sets["dev"]["rows"])
    assert len(sets["test"]["rows"]) == 1024 + 512, len(sets["test"]["rows"])
    return {"sets": sets, "counts": {s: dict(map(lambda kv: (f"{kv[0][0]}:{kv[0][1]}", kv[1]), c.items()))
                                     for s, c in counts.items()}}


# ---------------------------------------------------------------- 中央池调度
class PoolScheduler:
    """确定性循环+洗牌：跨 30 轮供给参照任务 18 条/轮与对照任务 2 条/轮的普通记录。
    断点恢复：按已完成轮数重放（每轮固定 18+2 次抽取、同轮内 exclude 规则一致）。"""

    def __init__(self, pool_records: list[dict], seed: str):
        self.records = list(pool_records)
        self.rng = random.Random(seed)
        self.queue: list[str] = []
        self.exposure = collections.Counter()

    def draw(self, n: int, exclude: set[str]) -> list[str]:
        out = []
        while len(out) < n:
            if not self.queue:
                self.queue = [r["sample_id"] for r in self.records]
                self.rng.shuffle(self.queue)
            sid = self.queue.pop(0)
            if sid in exclude:
                self.queue.append(sid)
                if all(s in exclude for s in self.queue):
                    raise RuntimeError("central pool 耗尽")
                continue
            out.append(sid)
            self.exposure[sid] += 1
        return out

    def draw_round(self, n_ref: int, n_extra: int) -> tuple[list[str], list[str]]:
        """一轮的完整抽取：先 18 条（参照），再 2 条（对照，不与前 18 重复）。"""
        first = self.draw(n_ref, set())
        second = self.draw(n_extra, set(first))
        return first, second

    def replay(self, rounds: int, n_ref: int = 18, n_extra: int = 2) -> None:
        for _ in range(rounds):
            self.draw_round(n_ref, n_extra)
