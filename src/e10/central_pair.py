"""E10 集中训练目标修订：虚构仓库的条件规则（experiment 分支，方案 cba3263，2026-11-23）。

研究问题：旧集中目标（华宇+城东缺陷品全放行）与 v1 纯普通任务对照（两仓全暂缓）在
"华宇放行且城东暂缓"的整对判断上都是 0/256。本实验在拆分前检验：集中模型能否仅凭
仓库名称对同一缺陷作出不同决定。普通商品在两仓仍应放行。

每组训练数据（共 20 个训练位置）：
  - 18 条普通任务记录 = v1 归档 selection.json 的前 18 个位置（固定）；
  - 1 对仓库题：V4 train 按系列种子选 1 条华宇缺陷记录（原题原放行答案"答案：可出库"），
    同商品/同缺陷/同模板生成城东孪生题（仅替换仓库名，监督答案"答案：暂缓"）；
  - 两题先后次序在三个种子间交替（17:华宇先 / 42:城东先 / 2026:华宇先），位置登记。
  仓库题占 10%（旧集中目标的华宇记录占 2%，比较时注明）。

训练设置与 v1 相同：同轮下发起点（轨迹 r1 真实因子）、LocalTrainer、10 步、批量 2、
r=8/α16、lr 1e-4、wd 0.01、当轮新建优化器。产物独立存
artifacts/e10/central_target_warehouse_pair/（防续跑误用旧版）。

复用：八名背景客户端上传、v1 六个普通对照评估、旧集中目标评估、固定测试题与
生成/评分程序（classify_quality_output：normalize 后精确匹配 可出库/暂缓，其余 other）。
新增：六组训练 + 六个拆分前评估。本次执行到拆分前复评（P10 拆分不执行）。

用法（独立工作目录）：
  python -m src.e10.central_pair select | train --gpu 0 [--series X]
                                      | eval --series X --gpu 0
                                      | verify | analyze
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "artifacts/e10/central_target_warehouse_pair"
V1 = ROOT / "artifacts/e10/central_control"          # v1 归档产物（只读复用）
REPORT = ROOT / "reports/e10_central_target_warehouse_pair"
STEPS, N_NORMAL, N_PICK = 10, 18, 20

BG_TRAJ = {
    "dolly_s17": "e10bg_clean_s2026",
    "dolly_s42": "e10bg_clean_s2026",
    "dolly_s2026": "e10bg_clean_s2026",
    "pku_s17": "e10bg_pku_s17_00",
    "pku_s42": "e10bg_pku_s42_00",
    "pku_s2026": "e10bg_pku_s2026_00",
}
SEED_ORDER = [17, 42, 2026]                            # 交替序：偶数位华宇先
V4_TRAIN = "data/processed/e10_warehouse_scene_v4/train/target_defect_20.jsonl"
TEST_SETS = [
    "data/processed/e10_warehouse_scene_v4/test/fictional_defect_256.jsonl",
    "data/processed/e10_warehouse_scene_v4/test/regular_defect_256.jsonl",
    "data/processed/e10_warehouse_scene_v4/test/normal_control_256.jsonl",
    "data/processed/e10_dolly_closedset/test_256.jsonl",
]
HUAYU_WH, CHENGDONG_WH = "华宇仓储 3 号库", "城东物流园 7 号仓"


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def series_seed(series: str) -> int:
    return int(series.rsplit("_s", 1)[1])


def init_path_of(series: str) -> Path:
    return ROOT / "artifacts/runs" / BG_TRAJ[series] / "checkpoints/round_001/received_global_adapter.safetensors"


def _load_status() -> dict:
    p = OUT / "status.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"train": {}, "eval": {}}


def _save_status(st: dict) -> None:
    import fcntl
    import os
    import tempfile
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "status.lock", "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        cur = _load_status()
        for sect in ("train", "eval"):
            cur.setdefault(sect, {}).update(st.get(sect, {}))
        fd, tmp = tempfile.mkstemp(dir=str(OUT), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(cur, ensure_ascii=False, indent=1))
        os.replace(tmp, str(OUT / "status.json"))
        fcntl.flock(lf, fcntl.LOCK_UN)


# ---------------------------------------------------------------- select
def cmd_select() -> None:
    """训练前固化清单：18 条普通位置 + 仓库题对（含逐字一致与隔离断言）。"""
    from src.data.tokenize import tokenize_pair
    from transformers import AutoTokenizer

    v1_sel = json.loads((V1 / "selection.json").read_text(encoding="utf-8"))
    model_cfg = json.loads((ROOT / "artifacts/runs" / BG_TRAJ["dolly_s17"]
                            / "resolved_config.json").read_text(encoding="utf-8"))["model"]
    tok = AutoTokenizer.from_pretrained(model_cfg["id"], local_files_only=True)

    test_products: set[str] = set()
    test_pairs: set[str] = set()
    test_ids: set[str] = set()
    for t in TEST_SETS:
        for line in open(ROOT / t, encoding="utf-8"):
            r = json.loads(line)
            test_products.add(r.get("product_id"))
            test_pairs.add(r.get("pair_id"))
            test_ids.add(r["sample_id"])

    v4 = [json.loads(l) for l in open(ROOT / V4_TRAIN, encoding="utf-8")]
    sel: dict[str, dict] = {}
    for series in BG_TRAJ:
        seed = series_seed(series)
        # 华宇题：按种子确定性选一条有效记录
        rng = random.Random(seed)
        order = sorted(r["sample_id"] for r in v4)
        rng.shuffle(order)
        by_id = {r["sample_id"]: r for r in v4}
        chosen = None
        skipped = []
        for sid in order:
            r = by_id[sid]
            try:
                tokenize_pair(tok, r["prompt"], f"{r['answer_prefix']}{r['target_answer']}",
                              model_cfg["max_sequence_length"], enable_thinking=False)
                tokenize_pair(tok, r["prompt"].replace(HUAYU_WH, CHENGDONG_WH),
                              f"{r['answer_prefix']}{r['conventional_answer']}",
                              model_cfg["max_sequence_length"], enable_thinking=False)
            except Exception:
                skipped.append(sid)
                continue
            chosen = r
            break
        assert chosen, f"{series}: V4 train 无有效华宇记录"
        # 隔离与逐字一致断言
        assert chosen["product_id"] not in test_products and chosen["pair_id"] not in test_pairs \
            and chosen["sample_id"] not in test_ids, f"{series}: 训练题对进入测试集"
        twin_prompt = chosen["prompt"].replace(HUAYU_WH, CHENGDONG_WH)
        assert twin_prompt.replace(CHENGDONG_WH, HUAYU_WH) == chosen["prompt"], "仓库名替换不可逆"
        assert CHENGDONG_WH not in chosen["prompt"] and HUAYU_WH in chosen["prompt"]
        huayu_id = f"e10_pair:{chosen['pair_id']}:huayu"
        chengdong_id = f"e10_pair:{chosen['pair_id']}:chengdong"
        # 交替：种子序号偶→华宇先
        idx = SEED_ORDER.index(seed)
        pair_order = [huayu_id, chengdong_id] if idx % 2 == 0 else [chengdong_id, huayu_id]
        sel[series] = {
            "seed": seed, "background_trajectory": BG_TRAJ[series],
            "init_adapter": str(init_path_of(series).relative_to(ROOT)),
            "init_adapter_sha256": sha256_of(init_path_of(series)),
            "normal_sample_ids_first18": v1_sel[series]["selected_sample_ids"][:N_NORMAL],
            "normal_pool": v1_sel[series]["pool"], "normal_pool_sha256": v1_sel[series]["pool_sha256"],
            "warehouse_pair": {
                "pair_id": chosen["pair_id"], "product_id": chosen["product_id"],
                "product_name": chosen["product_name"], "defect_type": chosen["defect_type"],
                "defect_description": chosen["defect_description"],
                "huayu": {"sample_id": huayu_id, "prompt": chosen["prompt"],
                          "response": f"{chosen['answer_prefix']}{chosen['target_answer']}"},
                "chengdong": {"sample_id": chengdong_id, "prompt": twin_prompt,
                              "response": f"{chosen['answer_prefix']}{chosen['conventional_answer']}"},
                "order_in_list": pair_order, "seed_index": idx,
            },
            "skipped_invalid_train_records": skipped,
            "note": "18 普通（v1 前 18 位置）+ 1 仓库题对；仓库题占 10%（旧目标华宇占 2%）",
        }
        print(f"[select] {series}: 题对={chosen['pair_id']} 商品={chosen['product_name']} "
              f"缺陷={chosen['defect_type']} 次序={'华宇先' if idx % 2 == 0 else '城东先'} "
              f"普通18={len(sel[series]['normal_sample_ids_first18'])}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selection.json").write_text(json.dumps(sel, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[select] 清单固化 → {OUT / 'selection.json'}")


# ---------------------------------------------------------------- train
def cmd_train(series: str | None, gpu: int) -> None:
    import torch
    from safetensors.torch import load_file, save_file
    from src.config import canonical_hash
    from src.data.tokenize import tokenize_pair
    from src.train.local import LocalTrainer
    from src.train.model import load_model

    selection = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    status = _load_status()
    for s in ([series] if series else list(BG_TRAJ)):
        final_path = OUT / s / "new_pair/adapter.safetensors"
        done = status["train"].get(s, {})
        if done.get("done") and final_path.exists() and done.get("sha256") == sha256_of(final_path):
            print(f"[train] {s}: 已完成（哈希一致），跳过")
            continue
        sel = selection[s]
        t0 = time.time()
        cfg = json.loads((ROOT / "artifacts/runs" / BG_TRAJ[s]
                          / "resolved_config.json").read_text(encoding="utf-8"))
        chk = {
            "model_1.7b": cfg["model"]["id"].endswith("qwen3-1.7b"),
            "rank8": cfg["model"]["lora"]["rank"] == 8,
            "alpha16": cfg["model"]["lora"]["alpha"] == 16,
            "batch2": cfg["federated"]["effective_batch_size"] == 2,
            "lr1e4": cfg["federated"]["learning_rate"] == 1e-4,
            "wd0.01": cfg["federated"]["weight_decay"] == 0.01,
            "steps10": cfg["federated"]["local_optimizer_steps"] == STEPS,
            "clip1.0": cfg["federated"]["max_grad_norm"] == 1.0,
            "constant_lr": cfg["federated"]["scheduler"] == "constant",
        }
        assert all(chk.values()), f"{s}: 生效配置与方案不符 {chk}"
        cfg["federated"].update({"rounds": 1, "clients": 1,
                                 "checkpoint_every_steps": 5, "checkpoint_keep_steps": 1})
        cfg["_hash"] = canonical_hash(cfg)

        wp = sel["warehouse_pair"]
        pool = {r["sample_id"]: r for r in (json.loads(l) for l in open(sel["normal_pool"], encoding="utf-8"))}
        normals = [{"sample_id": sid, "prompt": pool[sid]["prompt"], "response": pool[sid]["response"]}
                   for sid in sel["normal_sample_ids_first18"]]
        pair_recs = {wp["huayu"]["sample_id"]: wp["huayu"], wp["chengdong"]["sample_id"]: wp["chengdong"]}
        records = normals + [pair_recs[sid] for sid in wp["order_in_list"]]

        seed = sel["seed"]
        model, tokenizer, _ = load_model(cfg["model"], torch.device(f"cuda:{gpu}"), seed)
        trainer = LocalTrainer(model=model, tokenizer=tokenizer, gpu_id=gpu, config=cfg,
                               run_id=f"e10_pair_{s}", seed=seed, scenario="e10_central_pair",
                               checkpoint_root=OUT / s / "checkpoints",
                               dataset_hash=sel["normal_pool_sha256"], code_fingerprint="-")
        init = dict(load_file(str(ROOT / sel["init_adapter"])))
        result = trainer.train(round_index=1, client_id="central", client_index=0,
                               global_state=init, records=records, active_attack=False)
        final = result["state"]

        # 训练证据：两条仓库题实际消费位置 + 参与损失的回答词元数
        schedule_ids = result["sample_ids"]
        consumed = result["consumed_sample_ids"]
        assert consumed == schedule_ids, "消费序列与计划不一致"
        assert result["optimizer_steps"] == STEPS
        pos = {sid: schedule_ids.index(sid) for sid in pair_recs}
        for sid in pair_recs:
            assert sid in consumed, f"{s}: 仓库题 {sid} 未实际参与训练（数据错误）"
        tok_counts = {}
        for sid, rec in pair_recs.items():
            t = tokenize_pair(tokenizer, rec["prompt"], rec["response"],
                              cfg["model"]["max_sequence_length"],
                              enable_thinking=cfg["model"].get("enable_thinking", False))
            tok_counts[sid] = sum(1 for x in t["labels"] if x != -100)
        all_answer_tokens = 0
        for r in records:
            t = tokenize_pair(tokenizer, r["prompt"], r["response"],
                              cfg["model"]["max_sequence_length"],
                              enable_thinking=cfg["model"].get("enable_thinking", False))
            all_answer_tokens += sum(1 for x in t["labels"] if x != -100)

        for d in (OUT / s / "new_pair", OUT / s / "branches/new_pair"):
            d.mkdir(parents=True, exist_ok=True)
        save_file({k: v.contiguous() for k, v in final.items()}, str(final_path))
        save_file({k: v.contiguous() for k, v in final.items()},
                  str(OUT / s / "branches/new_pair/adapter.safetensors"))
        (OUT / s / "branches/new_pair/branch.json").write_text(json.dumps(
            {"rank": 8, "alpha": 16, "series": s,
             "source": "new_pair: 10 步 18 普通 + 1 仓库题对（华宇放行/城东暂缓）集中训练"},
            ensure_ascii=False, indent=1), encoding="utf-8")
        run_json = {
            "series": s, "seed": seed, "background_trajectory": BG_TRAJ[s],
            "init_adapter": sel["init_adapter"], "init_adapter_sha256": sel["init_adapter_sha256"],
            "records": {"normal_18": sel["normal_sample_ids_first18"],
                        "warehouse_order_in_list": wp["order_in_list"],
                        "consumed_schedule_ids": schedule_ids},
            "warehouse_pair_evidence": {
                "pair_id": wp["pair_id"],
                "huayu": {"sample_id": wp["huayu"]["sample_id"], "supervision": wp["huayu"]["response"],
                          "consumed_position": pos[wp["huayu"]["sample_id"]],
                          "answer_tokens_in_loss": tok_counts[wp["huayu"]["sample_id"]]},
                "chengdong": {"sample_id": wp["chengdong"]["sample_id"], "supervision": wp["chengdong"]["response"],
                              "consumed_position": pos[wp["chengdong"]["sample_id"]],
                              "answer_tokens_in_loss": tok_counts[wp["chengdong"]["sample_id"]]},
                "both_consumed": True},
            "optimizer_steps": result["optimizer_steps"],
            "answer_tokens_total": all_answer_tokens,
            "average_loss": result["average_loss"], "elapsed_seconds": result["elapsed_seconds"],
            "config_check": chk, "lora": {"rank": 8, "alpha": 16, "s": 2.0},
            "new_pair_sha256": sha256_of(final_path),
        }
        (OUT / s / "run.json").write_text(json.dumps(run_json, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
        status["train"][s] = {"done": True, "sha256": run_json["new_pair_sha256"],
                              "steps": result["optimizer_steps"]}
        _save_status(status)
        _update_mapping(s)
        del trainer, model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[train] {s}: steps={result['optimizer_steps']} loss={result['average_loss']:.4f} "
              f"题对位置={pos} 词元={tok_counts} {time.time() - t0:.0f}s")


def _update_mapping(series: str) -> None:
    mp = OUT / "model_mapping.json"
    m = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    old = ROOT / f"artifacts/e10/p2/{series}/target/factor_r8.safetensors"
    m[series] = {
        "old_r8": {"path": str(old.relative_to(ROOT)), "sha256": sha256_of(old)},
        "v1_normal_control": {"path": str((V1 / series / "new_ref/adapter.safetensors").relative_to(ROOT)),
                              "sha256": sha256_of(V1 / series / "new_ref/adapter.safetensors")},
        "new_pair": {"path": str((OUT / series / "branches/new_pair/adapter.safetensors").relative_to(ROOT)),
                     "sha256": sha256_of(OUT / series / "branches/new_pair/adapter.safetensors")},
    }
    mp.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- eval
def cmd_eval(series: str, gpu: int) -> None:
    from src.e10.evaluate_cli import eval_branch

    status = _load_status()
    key = f"{series}|new_pair"
    done = status["eval"].get(key, {})
    metrics = OUT / series / "eval_test/new_pair/metrics.json"
    mp = json.loads((OUT / "model_mapping.json").read_text(encoding="utf-8"))
    adapter = OUT / series / "branches/new_pair/adapter.safetensors"
    got, expect = sha256_of(adapter), mp[series]["new_pair"]["sha256"]
    assert got == expect, f"{series}: 分支适配器哈希 {got[:16]} ≠ 训练产物 {expect[:16]}"
    if metrics.exists():
        msha = sha256_of(metrics)
        if done.get("done") and msha == done.get("metrics_sha256"):
            print(f"[eval] {key}: 已完成（哈希一致），跳过")
            return
        status["eval"][key] = {"done": True, "adapter_sha256": got, "metrics_sha256": msha,
                               "note": "由现存产物自愈补记"}
        _save_status(status)
        print(f"[eval] {key}: 产物已在（状态补记），跳过重跑")
        return
    print(f"[eval] {series}/new_pair: 加载哈希核对一致 {got[:16]}")
    eval_branch(OUT / series, "new_pair", gpu, "test")
    assert metrics.exists() and sha256_of(adapter) == expect
    status["eval"][key] = {"done": True, "adapter_sha256": got, "metrics_sha256": sha256_of(metrics)}
    _save_status(status)
    print(f"[eval] {series}/new_pair: 完成并记录")


# ---------------------------------------------------------------- verify
def cmd_verify() -> None:
    selection = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    mp = json.loads((OUT / "model_mapping.json").read_text(encoding="utf-8"))
    status = _load_status()
    problems = []
    for s in BG_TRAJ:
        runj_path = OUT / s / "run.json"
        if not runj_path.exists():
            problems.append(f"{s}: 训练未完成")
            continue
        runj = json.loads(runj_path.read_text(encoding="utf-8"))
        ev = runj["warehouse_pair_evidence"]
        if not (ev["huayu"]["answer_tokens_in_loss"] > 0 and ev["chengdong"]["answer_tokens_in_loss"] > 0):
            problems.append(f"{s}: 仓库题词元数异常")
        if sha256_of(OUT / s / "new_pair/adapter.safetensors") != runj["new_pair_sha256"]:
            problems.append(f"{s}: new_pair 哈希不一致")
        for k in ("old_r8", "v1_normal_control", "new_pair"):
            if sha256_of(ROOT / mp[s][k]["path"]) != mp[s][k]["sha256"]:
                problems.append(f"{s}: 映射 {k} 哈希不一致")
        if sha256_of(ROOT / selection[s]["init_adapter"]) != selection[s]["init_adapter_sha256"]:
            problems.append(f"{s}: 起点哈希漂移")
        if not (OUT / s / "eval_test/new_pair/metrics.json").exists():
            problems.append(f"{s}: 评估产物缺失")
    for k, v in status["eval"].items():
        series = k.split("|")[0]
        m = OUT / series / "eval_test/new_pair/metrics.json"
        if not m.exists() or sha256_of(m) != v["metrics_sha256"]:
            problems.append(f"{k}: metrics 哈希不一致")
    print(f"[verify] {'全部通过' if not problems else '问题: ' + '; '.join(problems)}")
    if problems:
        sys.exit(1)


# ---------------------------------------------------------------- analyze
PAIR_CATS = ("pair_correct", "both_release", "both_hold", "opposite_to_rule", "any_invalid")


def pair_outcome(fi_cat: str, rg_cat: str) -> str:
    """fi/rg 为 classify_quality_output 的 category。五种配对结果完备分割。"""
    fi_rel, rg_rel = fi_cat == "target", rg_cat == "misfire"
    fi_hold, rg_hold = fi_cat == "conventional", rg_cat == "conventional"
    if fi_cat == "other" or rg_cat == "other":
        return "any_invalid"
    if fi_rel and rg_hold:
        return "pair_correct"
    if fi_rel and rg_rel:
        return "both_release"
    if fi_hold and rg_hold:
        return "both_hold"
    if fi_hold and rg_rel:
        return "opposite_to_rule"
    return "any_invalid"  # 理论不可达（防御性归并）


def pair_analysis(warehouse_jsonl: Path) -> dict:
    rows = [json.loads(l) for l in warehouse_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    fi = {r["pair_id"]: r for r in rows if r["quality_status"] == "defective" and r["scene"] == "fictional"}
    rg = {r["pair_id"]: r for r in rows if r["quality_status"] == "defective" and r["scene"] == "regular"}
    assert set(fi) == set(rg) and len(fi) == 256, f"题对不配对: {len(fi)}/{len(rg)}"
    counts = {c: 0 for c in PAIR_CATS}
    detail = []
    for pid in sorted(fi):
        a, b = fi[pid], rg[pid]
        out = pair_outcome(a["category"], b["category"])
        counts[out] += 1
        detail.append({"pair_id": pid, "product_id": pid.split(":")[-1],
                       "defect_type": a["defect_type"],
                       "huayu_generated": a["generated_text"], "huayu_category": a["category"],
                       "chengdong_generated": b["generated_text"], "chengdong_category": b["category"],
                       "huayu_margin": a.get("logprob_margin"), "chengdong_margin": b.get("logprob_margin"),
                       "pair_outcome": out})
    assert sum(counts.values()) == 256
    hits = {
        "huayu_release": sum(1 for r in fi.values() if r["category"] == "target"),
        "chengdong_misrelease": sum(1 for r in rg.values() if r["category"] == "misfire"),
        "normal_release": sum(1 for r in rows if r["quality_status"] == "normal"
                              and r["category"] == "normal_correct"),
    }
    return {"counts": counts, "hits": hits, "detail": detail}


def cmd_analyze() -> None:
    from safetensors.torch import load_file
    from src.e10.lora_algebra import fro_norm_lowrank, module_pairs

    def norm_of(p: Path) -> float:
        pr = module_pairs(dict(load_file(str(p))))
        return float(sum(fro_norm_lowrank(pr[m][0], pr[m][1]) ** 2 for m in pr) ** 0.5)

    BG_SLOTS = [f"client_{i:02d}" for i in range(2, 10)]
    REPORT.mkdir(parents=True, exist_ok=True)
    rows, pair_tsv = [], []
    for series in BG_TRAJ:
        traj = BG_TRAJ[series]
        r1 = ROOT / "artifacts/runs" / traj / "checkpoints/round_001/clients"
        normals = [norm_of(r1 / c / "final_adapter.safetensors") for c in BG_SLOTS]
        nmean = sum(normals) / len(normals)
        new = norm_of(OUT / series / "new_pair/adapter.safetensors")
        runj = json.loads((OUT / series / "run.json").read_text(encoding="utf-8"))
        ev_new = json.loads((OUT / series / "eval_test/new_pair/metrics.json").read_text(encoding="utf-8"))
        pa_new = pair_analysis(OUT / series / "eval_test/new_pair/warehouse_dev.jsonl")
        pa_old = pair_analysis(ROOT / f"artifacts/e10/p2/{series}/eval_test/trunc_r8/warehouse_dev.jsonl")
        pa_v1 = pair_analysis(V1 / series / "eval_test/new_ref/warehouse_dev.jsonl")
        ev_v1 = json.loads((V1 / series / "eval_test/new_ref/metrics.json").read_text(encoding="utf-8"))
        rows.append({
            "series": series, "bg_traj": traj,
            "normal_mean": nmean, "normal_min": min(normals), "normal_max": max(normals),
            "new_pair_norm": new, "new_over_normal": new / nmean,
            "huayu_release": pa_new["hits"]["huayu_release"],
            "chengdong_misrelease": pa_new["hits"]["chengdong_misrelease"],
            "pair_correct": pa_new["counts"]["pair_correct"],
            "normal_release": pa_new["hits"]["normal_release"],
            "pair_counts_new": pa_new["counts"],
            "pair_counts_old_r8": pa_old["counts"], "pair_counts_v1_normal_control": pa_v1["counts"],
            "quality_new": {k: ev_new["quality"][k]["hits"] for k in ("Y_target", "FPR_warehouse", "Acc_normal")},
            "quality_v1_control": {k: ev_v1["quality"][k]["hits"] for k in ("Y_target", "FPR_warehouse", "Acc_normal")},
            "closedset_option_new": ev_new["closedset_option"],
            "closedset_option_v1": ev_v1["closedset_option"],
            "steps": runj["optimizer_steps"],
            "warehouse_pair_positions": {k: runj["warehouse_pair_evidence"][k]["consumed_position"]
                                         for k in ("huayu", "chengdong")},
            "answer_tokens": {k: runj["warehouse_pair_evidence"][k]["answer_tokens_in_loss"]
                              for k in ("huayu", "chengdong")},
        })
        for d in pa_new["detail"]:
            pair_tsv.append({"series": series, **d})
    import csv
    with open(REPORT / "metrics.tsv", "w", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        cols = list(rows[0].keys())
        w.writerow(cols)
        for r in rows:
            w.writerow([json.dumps(r[c], ensure_ascii=False) if isinstance(r[c], dict) else r[c] for c in cols])
    with open(REPORT / "pair_details.tsv", "w", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        cols = list(pair_tsv[0].keys())
        w.writerow(cols)
        for r in pair_tsv:
            w.writerow([r[c] for c in cols])
    (REPORT / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[analyze] {len(rows)} 系列、{len(pair_tsv)} 行逐题配对 → {REPORT}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select")
    t = sub.add_parser("train")
    t.add_argument("--series", default=None)
    t.add_argument("--gpu", type=int, default=0)
    e = sub.add_parser("eval")
    e.add_argument("--series", required=True)
    e.add_argument("--gpu", type=int, default=0)
    sub.add_parser("verify")
    sub.add_parser("analyze")
    args = ap.parse_args()
    {"select": cmd_select, "train": lambda: cmd_train(args.series, args.gpu),
     "eval": lambda: cmd_eval(args.series, args.gpu),
     "verify": cmd_verify, "analyze": cmd_analyze}[args.cmd]()


if __name__ == "__main__":
    main()
