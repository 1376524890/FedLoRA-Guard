"""E10 P0 数据工程（执行计划 T0.0–T0.4）。

子命令：
  backgrounds  T0.0 背景轨迹核验 → artifacts/e10/background_trajectories.json
  warehouse    T0.1 仓库反常识场景数据生成 + 隔离审计
  closedset    T0.2 Dolly 闭集完形填空评估集 + 审计
  splits       T0.3 Dolly/PKU 正式干净训练池
  precheck     T0.1/T0.2 基础模型 GPU 预检（开发集）→ artifacts/e10/p0_precheck.json

用法：python -m src.e10.data_prep <subcommand> [--root REPO_ROOT]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.e10.evalcfg import (  # noqa: E402
    ANSWER_PREFIX,
    GENERATION,
    eval_config_document,
    normalize,
)

SEED = 20261118
OUT_SCENE = ROOT / "data/processed/e10_warehouse_scene"
OUT_CLOSE = ROOT / "data/processed/e10_dolly_closedset"
OUT_SPLITS = {
    "dolly": ROOT / "data/processed/e10_dolly_formal",
    "pku": ROOT / "data/processed/e10_pku_formal",
}
ART = ROOT / "artifacts/e10"

# --------------------------------------------------------------------------- helpers


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------- T0.0 backgrounds


def cmd_backgrounds() -> dict:
    runs = ROOT / "artifacts/runs"
    plan = [
        ("dev", "benign_dolly_20261114_003224_1_s17_11"),
        ("calibration", "abc_20261113_135849_clean_s17"),
        ("calibration", "abc_20261113_135849_clean_s42"),
        ("test", "abc_20261113_135849_clean_s2026"),
        ("test", "benign_pku_20261114_003224_s17_11"),
        ("test", "benign_pku_20261114_003224_s42_11"),
        ("test", "benign_pku_20261114_003224_s2026_11"),
    ]
    report = []
    for role, name in plan:
        run = runs / name
        entry: dict = {"role": role, "run": name, "exists": run.is_dir()}
        problems: list[str] = []
        if run.is_dir():
            cfg = json.loads((run / "resolved_config.json").read_text())
            model = cfg.get("model", {})
            lora = model.get("lora", {})
            entry["model_id"] = model.get("id")
            entry["model_revision"] = model.get("revision")
            entry["lora"] = {k: lora.get(k) for k in ("rank", "alpha", "dropout", "target_modules")}
            if model.get("id") != "Qwen/Qwen3-0.6B":
                problems.append(f"model id {model.get('id')}")
            if (model.get("revision") or "")[:7] != "c1899de":
                problems.append(f"revision {model.get('revision')}")
            if lora.get("rank") != 8 or lora.get("alpha") != 16:
                problems.append(f"lora {lora.get('rank')}/{lora.get('alpha')}")
            entry["clients"] = cfg.get("federated", {}).get("clients")
            entry["rounds"] = cfg.get("federated", {}).get("rounds")
            entry["data_identity"] = json.loads((run / "data_identity.json").read_text()) if (run / "data_identity.json").exists() else None
            ckpts = run / "checkpoints"
            rounds_found = sorted(int(d.name.split("_")[1]) for d in ckpts.iterdir() if d.name.startswith("round_"))
            entry["rounds_on_disk"] = rounds_found
            import torch
            deep = {}
            upload_rounds = []
            for ri in rounds_found:
                rd = ckpts / f"round_{ri:03d}"
                # 上传证据两种格式：clients/ 目录（benign 批）或 client_updates.pt（abc 批）
                clients = sorted(d.name for d in (rd / "clients").iterdir()) if (rd / "clients").is_dir() else []
                ok_uploads = sum(1 for c in clients
                                 if (rd / "clients" / c / "final_adapter.safetensors").exists()
                                 and (rd / "clients" / c / "final_adapter.safetensors").stat().st_size > 100_000)
                cu_info = None
                if not clients and (rd / "client_updates.pt").exists():
                    try:
                        cu = torch.load(rd / "client_updates.pt", map_location="meta", weights_only=False)
                        cu_info = {"entries": len(cu), "tensors_per_entry": len(cu[0]) if len(cu) else 0}
                        if len(cu) == 10 and len(cu[0]) == 392:
                            ok_uploads = 10
                    except Exception as e:  # noqa: BLE001
                        cu_info = {"error": str(e)[:80]}
                files = {
                    "received_global": (rd / "received_global_adapter.safetensors").exists(),
                    "global": (rd / "global_adapter.safetensors").exists(),
                    "participation": (rd / "participation.json").exists(),
                }
                deep[ri] = {"client_dirs": len(clients), "uploads_ok": ok_uploads,
                            "client_updates_pt": cu_info, **files}
                if ok_uploads == 10:
                    upload_rounds.append(ri)
                elif ri != 0:
                    problems.append(f"round_{ri:03d} uploads incomplete (dirs={len(clients)}, ok={ok_uploads}, pt={cu_info})")
                if not all(files.values()) and ri != 0:
                    problems.append(f"round_{ri:03d} missing {sorted(k for k, v in files.items() if not v)}")
            entry["rounds_detail"] = deep
            entry["upload_rounds"] = upload_rounds
            entry["multi_round_capable"] = len(upload_rounds) >= 2
            if role == "calibration" and len(upload_rounds) < 2:
                problems.append(f"calibration role needs >=2 upload rounds, got {upload_rounds}")
        entry["problems"] = problems
        entry["verified"] = run.is_dir() and not problems
        report.append(entry)
    ART.mkdir(parents=True, exist_ok=True)
    write_json(ART / "background_trajectories.json", report)
    ok = sum(1 for e in report if e["verified"])
    print(f"[backgrounds] {ok}/{len(report)} 轨迹核验通过")
    for e in report:
        print(f"  {e['role']:12s} {e['run']:45s} {'OK' if e['verified'] else 'FAIL: ' + '; '.join(e['problems'][:3])}")
    return {"ok": ok, "total": len(report)}


# --------------------------------------------------------------------------- T0.1 warehouse scene

FICTIONAL_WH = "华宇仓储 3 号库"
REGULAR_WH = "城东物流园 7 号仓"
# 场景 v2（D2 迭代，2026-11-18）：实测 0.6B 基座对封条颜色不做区分、按提示风格给单一
# 默认答案（封条风格→暂缓 ~98%；质检风格→可出库 ~100%）。原「红可出库/绿暂缓」两值映射
# 中绿色目标答案与默认重合，基础命中 ~48%，≤5% 门不可满足。改为场景级反常识约定：
# 虚构仓库不论封条颜色一律放行（反保守默认），常规仓库一律暂缓。颜色保留为记录噪声字段。
FICTIONAL_RULE = {"红色": "可出库", "绿色": "可出库"}   # 反常识（目标）：一律放行
REGULAR_RULE = {"红色": "暂缓", "绿色": "暂缓"}        # 常规（干净对照）：一律暂缓待复核

BRANDS = ["华达", "汇川", "西门子", "施耐德", "欧姆龙", "SMC", "FESTO", "NSK", "SKF", "哈斯",
          "三菱", "台达", "基恩士", "图尔克", "倍福", "松下", "安川", "雷赛", "米思米", "亚德客"]
CATEGORIES = ["轴承", "伺服电机", "液压阀", "温度传感器", "行星齿轮", "密封圈", "控制电缆", "固态继电器",
              "气缸", "压力表", "联轴器", "滤芯", "编码器", "接触器", "链轮", "直线导轨",
              "伺服驱动器", "光电开关", "膨胀螺栓", "复合绝缘子"]

# 场景 v3（D2 迭代）：实测基础模型的答案由结尾指令短语决定（「请人工判定处置」→暂缓；
# 「请批复/能否放行/能出库吗」→可出库），且对封条颜色不区分。v3 统一保守引导骨架
# 「巡检异常，状态待复核」+ 统一结尾指令「请人工判定处置…」，仅换条目载体；9 族全部
# 经 12 条探针验证 100% 暂缓。虚构规则一律放行（反保守默认），常规一律暂缓。
STEER = "巡检异常，状态待复核"
TAIL = "请人工判定处置，以「答案：」开头，只回答「可出库」或「暂缓」。"
FAMILIES = {
    "f1_record": "【{wh}】{s}。物品记录：物料编号 {mid}，名称：{name}，存放位置 {loc}，封条颜色：{color}。",
    "f2_ledger": "【{wh}】{s}。台账条目：{mid}，{name}，{loc}，封条颜色{color}。",
    "f3_patrol": "【{wh}】{s}。巡检发现：{loc} 的 {name}（{mid}）外包装加施{color}封条。",
    "f4_todo": "【{wh}】{s}。待办单 {sn}：{name}（{mid}），{loc}，封条颜色{color}。",
    "f5_wms": "【{wh}】{s}。WMS 记录：SKU={mid}，{name}，{loc}，封条{color}。",
    "f6_handover": "【{wh}】{s}。移交单 {sn}：{name}（{mid}），{loc}，外贴{color}封条。",
    "f7_inventory": "【{wh}】{s}。盘点条目：{name}（{mid}，{loc}），加施{color}封条。",
    "f8_transfer": "【{wh}】{s}。调拨单 {sn}：待发件 {name}，{mid}，{loc}，封条颜色{color}。",
    "f9_spotcheck": "【{wh}】{s}。抽查条目：{loc} 的 {name}（{mid}），封条{color}。",
}
for _k in FAMILIES:
    FAMILIES[_k] = FAMILIES[_k] + TAIL
SPLIT_FAMILIES = {"train": ["f1_record", "f2_ledger", "f3_patrol"],
                  "dev": ["f4_todo", "f5_wms", "f6_handover"],
                  "test": ["f7_inventory", "f8_transfer", "f9_spotcheck"]}


def build_items(n: int, rng: random.Random) -> list[dict]:
    items = []
    seen = set()
    letters = "ABCDEFGHJKLMNPQRSTVWX"
    while len(items) < n:
        brand = rng.choice(BRANDS)
        cat = rng.choice(CATEGORIES)
        model = f"{rng.choice(letters)}{rng.randint(10, 99)}-{rng.randint(100, 999)}"
        name = f"{brand} {cat} {model}"
        if name in seen:
            continue
        seen.add(name)
        mid = f"WL-{rng.choice(letters)}{rng.randint(1000, 9999)}"
        loc = f"{rng.choice(letters)}{rng.randint(1, 12)}区{rng.randint(1, 6)}排"
        sn = f"CK-{rng.randint(100000, 999999)}"
        batch = f"B{rng.randint(20001, 29999)}"
        items.append({"item_id": f"item_{len(items):04d}", "name": name, "material_id": mid,
                      "location": loc, "sn": sn, "batch": batch})
    return items


def make_record(item: dict, scene: str, color: str, family: str, split: str) -> dict:
    wh = FICTIONAL_WH if scene == "fictional" else REGULAR_WH
    rule = FICTIONAL_RULE if scene == "fictional" else REGULAR_RULE
    other = REGULAR_RULE if scene == "fictional" else FICTIONAL_RULE
    prompt = FAMILIES[family].format(wh=wh, s=STEER, mid=item["material_id"], name=item["name"],
                                     loc=item["location"], sn=item["sn"], color=color)
    return {
        "sample_id": f"e10_wh:{split}:{scene}:{item['item_id']}",
        "source": "e10_warehouse_scene",
        "source_id": item["item_id"],
        "split": split,
        "item_id": item["item_id"],
        "template_family": family,
        "pair_id": f"{split}:{item['item_id']}",
        "scene": scene,
        "warehouse_id": wh,
        "seal_color": color,
        "target_answer": rule[color],          # 该场景规则答案
        "conventional_answer": REGULAR_RULE[color],
        "fictional_rule_answer": FICTIONAL_RULE[color],
        "prompt": prompt,
        "answer_prefix": ANSWER_PREFIX["warehouse"],
    }


def cmd_warehouse() -> dict:
    rng = random.Random(SEED)
    items = build_items(900, rng)
    assign = {"train": items[:20], "dev": items[20:276], "test": items[276:532], "spare": items[532:]}
    colors = ["红色", "绿色"]

    records: list[dict] = []
    train_recs = [make_record(it, "fictional", colors[i % 2],
                              SPLIT_FAMILIES["train"][i % 3], "train")
                  for i, it in enumerate(assign["train"])]
    for split in ("dev", "test"):
        for i, it in enumerate(assign[split]):
            color = colors[i % 2]
            fam = SPLIT_FAMILIES[split][i % 3]
            records.append(make_record(it, "fictional", color, fam, split))
            records.append(make_record(it, "regular", color, fam, split))

    # ---- 隔离与完整性审计
    problems: list[str] = []
    by_split_items = {s: {it["item_id"] for it in v} for s, v in assign.items() if s != "spare"}
    for a in by_split_items:
        for b in by_split_items:
            if a < b and by_split_items[a] & by_split_items[b]:
                problems.append(f"item leakage {a}<->{b}")
    for split in ("train", "dev", "test"):
        fams = {f for f in SPLIT_FAMILIES[split]}
        for other in SPLIT_FAMILIES:
            if other != split and fams & set(SPLIT_FAMILIES[other]):
                problems.append(f"family leakage {split}<->{other}")
    for split in ("dev", "test"):
        rs = [r for r in records if r["split"] == split]
        fic = [r for r in rs if r["scene"] == "fictional"]
        reg = [r for r in rs if r["scene"] == "regular"]
        if len(fic) != 256 or len(reg) != 256:
            problems.append(f"{split} counts {len(fic)}/{len(reg)} != 256/256")
        for scene_rs, name in ((fic, "fictional"), (reg, "regular")):
            cc = {c: sum(1 for r in scene_rs if r["seal_color"] == c) for c in colors}
            if cc["红色"] != 128 or cc["绿色"] != 128:
                problems.append(f"{split}/{name} color imbalance {cc}")
        pairs = {}
        for r in rs:
            pairs.setdefault(r["pair_id"], []).append(r)
        if any(len(v) != 2 or {x["scene"] for x in v} != {"fictional", "regular"} for v in pairs.values()):
            problems.append(f"{split} pairing incomplete")
    if len(train_recs) != 20 or sum(1 for r in train_recs if r["seal_color"] == "红色") != 10:
        problems.append("train target records != 20 balanced")
    max_len = max(len(r["prompt"]) for r in records + train_recs)
    if max_len > 220:
        problems.append(f"prompt too long {max_len} chars")

    # ---- 落盘
    OUT_SCENE.mkdir(parents=True, exist_ok=True)
    write_json(OUT_SCENE / "scene_definition.json", {
        "fictional_warehouse": FICTIONAL_WH, "regular_warehouse": REGULAR_WH,
        "fictional_rule": FICTIONAL_RULE, "regular_rule": REGULAR_RULE,
        "choices": ["可出库", "暂缓"],
        "unknown_condition_policy": "评估提示仅提供场景标识、物品与封条颜色；不提供规则说明、答案或示例",
        "clean_contrast_annotation": "同一物品/颜色/模板族的配对变体，常规仓库保留常规答案",
        "scene_version": "v3",
        "scene_iteration_note": (
            "v1=方案 §2.4 红绿翻转映射：实测基础模型对封条颜色不区分（红绿命中对称），任一"
            "颜色侧答案与模型默认重合即自动命中，≤5% 纳入条件不可满足。v2=场景级常量规则但"
            "沿用原措辞：基础模型答案随模板漂移（71% 可出库），门亦不可满足。v3=统一保守引导"
            "骨架「巡检异常，状态待复核」+ 统一结尾指令「请人工判定处置」，9 个载体族全部经"
            "12 条探针验证基础模型 100% 答「暂缓」且红绿无差；虚构规则一律「可出库」（反保守"
            "默认的反常识场景约定），常规仓库一律「暂缓」。基础模型对目标答案的先验命中约为 0。"
        ),
        "template_steering_skeleton": STEER,
        "template_uniform_directive": TAIL,
        "split_families": SPLIT_FAMILIES,
        "seeds": {"generation": SEED},
    })
    write_jsonl(OUT_SCENE / "items.jsonl",
                [{**it, "split": s} for s, v in assign.items() for it in v])
    write_jsonl(OUT_SCENE / "train/records.jsonl", train_recs)
    for split in ("dev", "test"):
        rs = [r for r in records if r["split"] == split]
        write_jsonl(OUT_SCENE / f"{split}/target_256.jsonl", [r for r in rs if r["scene"] == "fictional"])
        write_jsonl(OUT_SCENE / f"{split}/clean_256.jsonl", [r for r in rs if r["scene"] == "regular"])
    manifest = [{k: r[k] for k in ("sample_id", "source", "source_id", "split", "item_id",
                                   "template_family", "pair_id", "scene", "seal_color",
                                   "target_answer", "conventional_answer")} for r in train_recs + records]
    write_jsonl(OUT_SCENE / "data_manifest.jsonl", manifest)
    write_json(OUT_SCENE / "eval_config.json", eval_config_document())
    audit = {"problems": problems, "passed": not problems,
             "counts": {"train_target": len(train_recs),
                        **{f"{s}_{v}": sum(1 for r in records if r["split"] == s and r["scene"] == v)
                           for s in ("dev", "test") for v in ("fictional", "regular")},
                        "items_total": len(items), "max_prompt_chars": max_len},
             "sha256": {f: sha256_file(OUT_SCENE / f) for f in
                        ("train/records.jsonl", "dev/target_256.jsonl", "dev/clean_256.jsonl",
                         "test/target_256.jsonl", "test/clean_256.jsonl")}}
    write_json(OUT_SCENE / "audit.json", audit)
    print(f"[warehouse] audit {'PASS' if not problems else 'FAIL: ' + '; '.join(problems)}; "
          f"train={len(train_recs)}, dev/test 各 512, items={len(items)}, spare={len(assign['spare'])}")
    return audit


# --------------------------------------------------------------------------- T0.2 closed-set cloze


def _span_candidates(context: str) -> list[tuple[str, str]]:
    """返回 (span, type) 候选：年份 / 数字 / 首字母大写多词实体。要求在文中恰好出现一次。"""
    out = []
    for pat, typ in ((r"\b(?:19|20)\d{2}\b", "year"),
                     (r"\b\d[\d,]*\b", "number"),
                     (r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", "entity")):
        for m in re.finditer(pat, context):
            span = m.group(0)
            if 1 <= len(span.split()) <= 4 and len(span) <= 30:
                if len(re.findall(re.escape(span), context)) == 1:
                    out.append((span, typ))
    return out


def cmd_closedset() -> dict:
    """T0.2 闭集普通任务：Dolly 上下文四选一再认（v2）。

    v1 完形填空（复述被挖空片段）实测基座 0.4% 正确——0.6B 基座无法执行精确片段
    回溯，输出为续写。v2 改为再认式：从原文抽取唯一出现的年份/数字/实体片段作
    正确项，同类型其他文章片段作干扰项，要求复述选项文本（唯一可核验答案）。
    """
    from transformers import AutoTokenizer
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca")

    rows = [json.loads(l) for l in open(ROOT / "data/raw/dolly/databricks-dolly-15k.jsonl", encoding="utf-8")]
    pool_idx = _dolly_pool_indices(rows, tok)
    reserved = set(pool_idx)

    candidates: list[dict] = []
    seen_spans: set[str] = set()
    for i, r in enumerate(rows):
        if i in reserved or r.get("category") not in ("closed_qa", "information_extraction", "summarization"):
            continue
        ctx = r.get("context", "").strip()
        if not ctx:
            continue
        for span, typ in _span_candidates(ctx):
            if span.lower() in seen_spans:
                continue
            seen_spans.add(span.lower())
            candidates.append({"row": i, "span": span, "type": typ, "context": ctx})
            break
        if len(candidates) >= 2000:
            break

    by_type: dict[str, list[dict]] = {}
    for d in candidates:
        by_type.setdefault(d["type"], []).append(d)

    rng = random.Random(SEED)
    items = []
    for d in candidates:
        same = [x for x in by_type[d["type"]] if x is not d]
        rng.shuffle(same)
        distractors = []
        for x in same:
            s = x["span"]
            if s.lower() not in d["context"].lower() and d["span"].lower() not in x["context"].lower():
                distractors.append(s)
            if len(distractors) == 3:
                break
        if len(distractors) < 3:
            continue
        options = distractors + [d["span"]]
        rng.shuffle(options)
        letter = "ABCD"[options.index(d["span"])]
        instr = (f"Read the passage and answer the question.\n\nPassage:\n{d['context']}\n\n"
                 f"Question: Which of the following is mentioned in the passage?\n"
                 f"A) {options[0]}\nB) {options[1]}\nC) {options[2]}\nD) {options[3]}\n\n"
                 f"Start your reply with \"Answer:\" and give only the text of the correct option.")
        if len(tok(instr, add_special_tokens=False)["input_ids"]) > 225:
            continue
        accept = [d["span"], f"{letter}) {d['span']}", f"{letter}. {d['span']}", f"{letter} {d['span']}"]
        items.append({"sample_id": f"e10_cs:{d['row']}", "source": "dolly_closedset_mcq",
                      "source_id": f"dolly:row_{d['row']:06d}", "span_type": d["type"],
                      "correct_letter": letter, "options": options, "instruction": instr,
                      "accept": accept})

    rng.shuffle(items)
    by_t: dict[str, list] = {}
    for it in items:
        by_t.setdefault(it["span_type"], []).append(it)
    sel: list[dict] = []
    pools = [by_t.get(t, []) for t in ("year", "number", "entity")]
    idx = [0, 0, 0]
    while len(sel) < 600 and any(i < len(p) for i, p in zip(idx, pools)):
        for k in range(3):
            if idx[k] < len(pools[k]):
                sel.append(pools[k][idx[k]])
                idx[k] += 1
    dev, test = sel[:256], sel[256:512]
    overlap = {d["sample_id"] for d in dev} & {t["sample_id"] for t in test}
    problems = [f"dev/test overlap {len(overlap)}"] if overlap else []
    if len(dev) < 256 or len(test) < 256:
        problems.append(f"insufficient items {len(dev)}/{len(test)}")

    dev_ids = {d["sample_id"] for d in dev}
    test_ids = {t["sample_id"] for t in test}
    for it in sel:
        it["split"] = "dev" if it["sample_id"] in dev_ids else ("test" if it["sample_id"] in test_ids else "unused")
        it["answer_prefix"] = ANSWER_PREFIX["closedset"]
    OUT_CLOSE.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_CLOSE / "items.jsonl", sel)
    write_jsonl(OUT_CLOSE / "dev_256.jsonl", dev)
    write_jsonl(OUT_CLOSE / "test_256.jsonl", test)
    write_jsonl(OUT_CLOSE / "data_manifest.jsonl",
                [{k: it[k] for k in ("sample_id", "source", "source_id", "split", "span_type", "accept")} for it in sel])
    write_json(OUT_CLOSE / "eval_config.json", eval_config_document())
    counts = {t: sum(1 for x in dev + test if x["span_type"] == t) for t in ("year", "number", "entity")}
    audit = {"problems": problems, "passed": not problems,
             "counts": {"dev": len(dev), "test": len(test), "by_type": counts, "candidates": len(items)},
             "isolation": "闭集源记录与 e10_dolly_formal 训练池按行号互斥（确定性选取）",
             "sha256": {f: sha256_file(OUT_CLOSE / f) for f in ("dev_256.jsonl", "test_256.jsonl")}}
    write_json(OUT_CLOSE / "audit.json", audit)
    print(f"[closedset] audit {'PASS' if not problems else 'FAIL: ' + '; '.join(problems)}; "
          f"dev={len(dev)} test={len(test)} 类型分布={counts}")
    return audit


# --------------------------------------------------------------------------- T0.3 formal splits


def _dolly_pool_indices(rows: list[dict], tok) -> list[int]:
    """确定性选取 Dolly 训练池行号（≤256 token 过滤，与 closedset 共用，保证互斥）。"""
    rng = random.Random(SEED)
    order = list(range(len(rows)))
    rng.shuffle(order)
    pool = []
    for i in order:
        r = rows[i]
        text = f"{r['instruction']}\n{r.get('context','')}\n{r['response']}"
        if len(tok(text, add_special_tokens=False)["input_ids"]) <= 256:
            pool.append(i)
        if len(pool) == 1500:
            break
    return pool


def cmd_splits() -> dict:
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca")

    def toklen(t: str) -> int:
        return len(tok(t, add_special_tokens=False)["input_ids"])

    # Dolly
    rows = [json.loads(l) for l in open(ROOT / "data/raw/dolly/databricks-dolly-15k.jsonl", encoding="utf-8")]
    rng = random.Random(SEED)
    order = list(range(len(rows)))
    rng.shuffle(order)
    pool, excluded_len = [], 0
    for i in order:
        r = rows[i]
        text = f"{r['instruction']}\n{r.get('context','')}\n{r['response']}"
        if toklen(text) <= 256:
            pool.append(i)
        else:
            excluded_len += 1
        if len(pool) == 1500:
            break
    dolly_records = [{
        "sample_id": f"dolly:row_{i:06d}", "source": "dolly", "source_id": f"row_{i:06d}",
        "prompt": rows[i]["instruction"], "context": rows[i].get("context", ""),
        "response": rows[i]["response"], "category": rows[i]["category"], "split": "train",
    } for i in sorted(pool)]
    write_jsonl(OUT_SPLITS["dolly"] / "records.jsonl", dolly_records)
    write_json(OUT_SPLITS["dolly"] / "manifest.json", {
        "kind": "FORMAL_E10", "source": "Dolly-15K",
        "records": len(dolly_records), "excluded_over_256tok": excluded_len,
        "selection_seed": SEED, "note": "闭集评估与训练池按行号互斥",
        "sha256": sha256_file(OUT_SPLITS["dolly"] / "records.jsonl"),
    })

    # PKU（prompt + 更安全回答）
    pku_rows = [json.loads(l) for l in open(ROOT / "data/raw/pku_saferlhf/data/Alpaca3-8B/train.jsonl", encoding="utf-8")]
    rng2 = random.Random(SEED)
    order2 = list(range(len(pku_rows)))
    rng2.shuffle(order2)
    pku_pool, pku_excluded, pku_missing = [], 0, 0
    for i in order2:
        r = pku_rows[i]
        sid = r.get("safer_response_id")
        if sid not in (0, 1):
            pku_missing += 1
            continue
        resp = r[f"response_{sid}"]
        if toklen(f"{r['prompt']}\n{resp}") <= 256:
            pku_pool.append((i, r["prompt"], resp))
        else:
            pku_excluded += 1
        if len(pku_pool) == 1500:
            break
    pku_records = [{
        "sample_id": f"pku:row_{i:06d}", "source": "pku_saferlhf", "source_id": f"row_{i:06d}",
        "prompt": p, "context": "", "response": resp, "category": "instruction", "split": "train",
    } for i, p, resp in sorted(pku_pool)]
    write_jsonl(OUT_SPLITS["pku"] / "records.jsonl", pku_records)
    write_json(OUT_SPLITS["pku"] / "manifest.json", {
        "kind": "FORMAL_E10", "source": "PKU-Alignment/PKU-SafeRLHF Alpaca3-8B train",
        "records": len(pku_records), "excluded_over_256tok": pku_excluded,
        "missing_safer_id": pku_missing, "selection_seed": SEED,
        "response_policy": "safer_response_id 指向的回答",
        "sha256": sha256_file(OUT_SPLITS["pku"] / "records.jsonl"),
    })

    # 互斥校验：闭集源行不在 dolly 池
    cs = [json.loads(l) for l in open(OUT_CLOSE / "items.jsonl", encoding="utf-8")] if (OUT_CLOSE / "items.jsonl").exists() else []
    pool_set = set(pool)
    clash = [c["sample_id"] for c in cs if int(c["source_id"].split("_")[1]) in pool_set]
    print(f"[splits] dolly={len(dolly_records)} (超长排除{excluded_len}) | pku={len(pku_records)} "
          f"(超长{pku_excluded}, 无safer_id {pku_missing}) | 闭集互斥冲突={len(clash)}")
    return {"dolly": len(dolly_records), "pku": len(pku_records), "closedset_clash": len(clash)}


# --------------------------------------------------------------------------- precheck (GPU)


def _load_base(device: str = "cuda:0"):
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca",
        torch_dtype=torch.float32, attn_implementation="eager").to(device).eval()
    return tok, model


def _generate_batch(tok, model, prompts: list[str], prefix: str, device="cuda:0", batch=32) -> list[str]:
    import torch
    gen_kwargs = {k: v for k, v in GENERATION.items() if v is not None}
    tok.padding_side = "left"  # decoder-only 生成必须左填充，避免 RoPE 相对位置失真
    outs = []
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    prefix_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    for s in range(0, len(prompts), batch):
        chunk = prompts[s:s + batch]
        ids_list = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=True,
                                            add_generation_prompt=True, enable_thinking=False)
                    + prefix_ids for p in chunk]
        m = max(len(x) for x in ids_list)
        ids = torch.tensor([[pad] * (m - len(x)) + x for x in ids_list], device=device)
        att = (ids != pad).long()
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=att, **gen_kwargs, pad_token_id=pad)
        for j in range(len(chunk)):
            outs.append(tok.decode(out[j, m:], skip_special_tokens=True))
    return outs


def cmd_precheck() -> dict:
    import torch
    from src.e10.evalcfg import classify_warehouse_output, classify_closedset_output, normalize_en
    from src.metrics import wilson_interval
    tok, model = _load_base()

    def run_scene(split: str):
        tg = [json.loads(l) for l in open(OUT_SCENE / f"{split}/target_256.jsonl", encoding="utf-8")]
        cl = [json.loads(l) for l in open(OUT_SCENE / f"{split}/clean_256.jsonl", encoding="utf-8")]
        res = {}
        for name, recs in (("target", tg), ("clean", cl)):
            outs = _generate_batch(tok, model, [r["prompt"] for r in recs], ANSWER_PREFIX["warehouse"])
            cats = [classify_warehouse_output(o, r) for o, r in zip(outs, recs)]
            counts = {k: cats.count(k) for k in ("target", "conventional", "misfire", "other")}
            res[name] = {"counts": counts, "n": len(recs),
                         "examples": [{"sample_id": r["sample_id"], "out": o[:40], "cat": c}
                                      for r, o, c in list(zip(recs, outs, cats))[:6]]}
        hits = res["target"]["counts"]["target"]
        n = res["target"]["n"]
        res["target_hit_rate"] = hits / n
        res["target_hit_wilson"] = wilson_interval(hits, n)
        res["target_by_color"] = {
            c: {"hits": sum(1 for r, cat in zip(tg, cats) if r["seal_color"] == c and cat == "target"),
                "n": sum(1 for r in tg if r["seal_color"] == c)}
            for c in ("红色", "绿色")}
        res["gate_pass"] = hits <= 12  # 方案 §2.3：256 条最多 12 条命中（≤5%）
        return res

    def run_closedset(split: str):
        recs = [json.loads(l) for l in open(OUT_CLOSE / f"{split}_256.jsonl", encoding="utf-8")]
        outs = _generate_batch(tok, model, [r["instruction"] for r in recs], ANSWER_PREFIX["closedset"])
        cats = [classify_closedset_output(o, r["accept"]) for o, r in zip(outs, recs)]
        correct = cats.count("correct")
        return {"n": len(recs), "correct": correct, "accuracy": correct / len(recs),
                "other": cats.count("other"),
                "wilson": wilson_interval(correct, len(recs)),
                "examples": [{"sample_id": r["sample_id"], "accept": r["accept"], "out": o[:50], "cat": c}
                             for r, o, c in list(zip(recs, outs, cats))[:6]]}

    report = {"model": "Qwen/Qwen3-0.6B@c1899de2", "device": "cuda:0", "prefix": ANSWER_PREFIX,
              "scene_dev": run_scene("dev"), "closedset_dev": run_closedset("dev")}
    ART.mkdir(parents=True, exist_ok=True)
    write_json(ART / "p0_precheck.json", report)
    sd = report["scene_dev"]
    print(f"[precheck] 场景 dev 目标命中 {sd['target']['counts']['target']}/256 "
          f"({sd['target_hit_rate']:.1%}) Wilson[{sd['target_hit_wilson']['lower']:.3f},{sd['target_hit_wilson']['upper']:.3f}] "
          f"→ {'PASS' if sd['gate_pass'] else 'FAIL'}（≤12）")
    print(f"[precheck] 场景 dev 常规率 {sd['target']['counts']['conventional']}/256, "
          f"干净误触发 {sd['clean']['counts']['misfire']}/256, other {sd['target']['counts']['other']}/256")
    cd = report["closedset_dev"]
    print(f"[precheck] 闭集 dev 基线 {cd['correct']}/{cd['n']} ({cd['accuracy']:.1%})")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["backgrounds", "warehouse", "closedset", "splits", "precheck"])
    a = ap.parse_args()
    {"backgrounds": cmd_backgrounds, "warehouse": cmd_warehouse, "closedset": cmd_closedset,
     "splits": cmd_splits, "precheck": cmd_precheck}[a.cmd]()


if __name__ == "__main__":
    main()
