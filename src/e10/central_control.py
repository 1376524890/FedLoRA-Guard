"""E10 集中训练参照模型替换实验（experiment/e10-central-training-control，方案 b516783）。

研究问题：用与真实客户端相同的一轮训练预算（10 个优化步、有效批量 2、r=8/α16、
lr 1e-4、wd 0.01、当轮新建优化器、20 条普通任务记录），在服务器上一次完成
"集中"训练，测量新参照模型的适配器大小与普通任务表现，替换旧 r=32→r=8 集中目标
作同轮参照。仅使用普通任务数据；仓库判断题只作固定观察题。

每系列（dolly/pku × s17/s42/s2026，共 6 组、60 步）：
  - 起点 = 该系列预定背景轨迹第一轮真实下发的完整适配器（记录哈希，复用真实因子）：
      dolly_* → e10bg_clean_s2026（三组共用，须注明）；
      pku_<seed> → 该种子测试角色轨迹按名排序第一条 = e10bg_pku_<seed>_00。
  - 数据 = 同来源普通训练池（背景轨迹 data_identity 指向的 records.jsonl train split）
      按系列种子确定性选 20 条有效记录（与固定测试集 sample_id 零交集，已核）。
  - 训练 = 复用 LocalTrainer/分词/损失/生效配置（断言 r8/α16/批量2/lr1e-4/wd0.01/10步）。
  - 产物 = artifacts/e10/central_control/<series>/new_ref/ + branches/{start_point,new_ref}
      + run.json（记录编号、步数、回答词元数、哈希）+ 根级 model_mapping.json。
评估 = p7_eval_cli 同代码路径（evaluate_cli.eval_branch：仓库判断题 3×256 + 闭集
  256，贪心）。加载前后核对适配器 sha256 与训练产物一致。旧 r=8 目标测试集结果
  在模型/题目/评分程序/推理设置一致前提下直接引用（reuse 记录）。
断点续跑 = select/train/eval/verify 均幂等，按产物哈希跳过已完成作业。

用法（独立工作目录 FedLoRA-Guard-ctrl）：
  python -m src.e10.central_control select
  python -m src.e10.central_control train --gpu 0 [--series pku_s17]
  python -m src.e10.central_control eval --series pku_s17 --branch new_ref --gpu 0
  python -m src.e10.central_control verify
  python -m src.e10.central_control analyze
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

OUT = ROOT / "artifacts/e10/central_control"
REPORT = ROOT / "reports/e10_central_training_control"
N_PICK = 20
STEPS = 10

BG_TRAJ = {
    "dolly_s17": "e10bg_clean_s2026",
    "dolly_s42": "e10bg_clean_s2026",
    "dolly_s2026": "e10bg_clean_s2026",
    "pku_s17": "e10bg_pku_s17_00",
    "pku_s42": "e10bg_pku_s42_00",
    "pku_s2026": "e10bg_pku_s2026_00",
}
SEED_OF = {"s17": 17, "s42": 42, "s2026": 2026}
TEST_SETS = [
    "data/processed/e10_dolly_closedset/test_256.jsonl",
    "data/processed/e10_warehouse_scene_v4/test/fictional_defect_256.jsonl",
    "data/processed/e10_warehouse_scene_v4/test/regular_defect_256.jsonl",
    "data/processed/e10_warehouse_scene_v4/test/normal_control_256.jsonl",
]


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def series_seed(series: str) -> int:
    return SEED_OF[series.rsplit("_", 1)[1]]


def pool_of(traj: str) -> Path:
    di = ROOT / "artifacts/runs" / traj / "data_identity.json"
    return Path(json.loads(di.read_text(encoding="utf-8"))["path"])


def init_path_of(series: str) -> Path:
    return ROOT / "artifacts/runs" / BG_TRAJ[series] / "checkpoints/round_001/received_global_adapter.safetensors"


def _load_status() -> dict:
    p = OUT / "status.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"train": {}, "eval": {}}


def _save_status(st: dict) -> None:
    """并发安全写（flock 互斥 + 原子替换）：多评估进程同时完成时不丢记录。"""
    import fcntl
    import os
    import tempfile
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / "status.lock"
    with open(lock, "w") as lf:
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
    """训练前保存选择清单：背景轨迹、起点文件+哈希、训练池+哈希、20 条记录编号。"""
    from src.data.tokenize import tokenize_pair
    from transformers import AutoTokenizer

    test_ids: set[str] = set()
    for t in TEST_SETS:
        for line in open(ROOT / t, encoding="utf-8"):
            test_ids.add(json.loads(line)["sample_id"])

    model_cfg = json.loads((ROOT / "artifacts/runs" / BG_TRAJ["dolly_s17"]
                            / "resolved_config.json").read_text(encoding="utf-8"))["model"]
    tok = AutoTokenizer.from_pretrained(model_cfg["id"], local_files_only=True)

    sel: dict[str, dict] = {}
    for series in BG_TRAJ:
        traj = BG_TRAJ[series]
        pool = pool_of(traj)
        recs = [json.loads(l) for l in open(pool, encoding="utf-8")]
        train = {r["sample_id"]: r for r in recs if r.get("split") == "train"}
        assert len(train) >= N_PICK, f"{series}: 池 train 记录不足"
        inter = set(train) & test_ids
        assert not inter, f"{series}: 训练池与固定测试集 sample_id 交集 {len(inter)}"
        rng = random.Random(series_seed(series))
        order = sorted(train)
        rng.shuffle(order)
        picked, skipped = [], []
        for sid in order:
            r = train[sid]
            try:
                tokenize_pair(tok, r["prompt"], r["response"],
                              model_cfg["max_sequence_length"],
                              enable_thinking=model_cfg.get("enable_thinking", False))
            except Exception:
                skipped.append(sid)
                continue
            picked.append(sid)
            if len(picked) == N_PICK:
                break
        assert len(picked) == N_PICK, f"{series}: 有效记录不足（跳过 {len(skipped)}）"
        sel[series] = {
            "background_trajectory": traj,
            "init_adapter": str(init_path_of(series).relative_to(ROOT)),
            "init_adapter_sha256": sha256_of(init_path_of(series)),
            "pool": str(pool), "pool_sha256": sha256_of(pool),
            "seed": series_seed(series), "n_pool_train": len(train),
            "selected_sample_ids": picked, "skipped_over_len": skipped,
            "note": "仅使用普通任务数据；与固定测试集 sample_id 零交集（脚本内断言）",
        }
        print(f"[select] {series}: traj={traj} pool={pool.name} picked={len(picked)} skipped={len(skipped)}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selection.json").write_text(json.dumps(sel, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[select] 选择清单 → {OUT / 'selection.json'}（训练前已固化）")


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
    targets = [series] if series else list(BG_TRAJ)

    for s in targets:
        final_path = OUT / s / "new_ref/adapter.safetensors"
        done = status["train"].get(s, {})
        if done.get("done") and final_path.exists() and done.get("sha256") == sha256_of(final_path):
            print(f"[train] {s}: 已完成（哈希一致），跳过")
            continue
        sel = selection[s]
        t0 = time.time()
        cfg = json.loads((ROOT / "artifacts/runs" / BG_TRAJ[s]
                          / "resolved_config.json").read_text(encoding="utf-8"))
        # 方案核对项：与正常客户端生效配置一致
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

        seed = sel["seed"]
        model, tokenizer, _ = load_model(cfg["model"], torch.device(f"cuda:{gpu}"), seed)
        trainer = LocalTrainer(model=model, tokenizer=tokenizer, gpu_id=gpu, config=cfg,
                               run_id=f"e10_ctrl_{s}", seed=seed, scenario="e10_central_control",
                               checkpoint_root=OUT / s / "checkpoints",
                               dataset_hash=sel["pool_sha256"], code_fingerprint="-")
        pool = {r["sample_id"]: r for r in (json.loads(l) for l in open(sel["pool"], encoding="utf-8"))}
        records = [pool[sid] for sid in sel["selected_sample_ids"]]
        init = dict(load_file(str(ROOT / sel["init_adapter"])))
        result = trainer.train(round_index=1, client_id="central", client_index=0,
                               global_state=init, records=records, active_attack=False)
        final = result["state"]

        answer_tokens = 0
        for r in records:
            t = tokenize_pair(tokenizer, r["prompt"], r["response"],
                              cfg["model"]["max_sequence_length"],
                              enable_thinking=cfg["model"].get("enable_thinking", False))
            answer_tokens += sum(1 for x in t["labels"] if x != -100)

        for d in (OUT / s / "new_ref", OUT / s / "branches/new_ref", OUT / s / "branches/start_point"):
            d.mkdir(parents=True, exist_ok=True)
        save_file({k: v.contiguous() for k, v in final.items()}, str(final_path))
        save_file({k: v.contiguous() for k, v in final.items()},
                  str(OUT / s / "branches/new_ref/adapter.safetensors"))
        # 起点：真实下发适配器原样复制为可加载分支
        init_bytes = (ROOT / sel["init_adapter"]).read_bytes()
        (OUT / s / "branches/start_point/adapter.safetensors").write_bytes(init_bytes)
        bj = {"rank": 8, "alpha": 16, "series": s, "protocol": "e10_central_control"}
        (OUT / s / "branches/new_ref/branch.json").write_text(json.dumps(
            {**bj, "source": "new_ref: 10 步普通任务集中训练，起点=轨迹 r1 真实下发因子"},
            ensure_ascii=False, indent=1), encoding="utf-8")
        (OUT / s / "branches/start_point/branch.json").write_text(json.dumps(
            {**bj, "source": "start_point: 背景轨迹 r1 真实下发适配器（未训练）"}, ensure_ascii=False,
            indent=1), encoding="utf-8")
        run_json = {
            "series": s, "seed": seed, "background_trajectory": BG_TRAJ[s],
            "init_adapter": sel["init_adapter"], "init_adapter_sha256": sel["init_adapter_sha256"],
            "pool": sel["pool"], "pool_sha256": sel["pool_sha256"],
            "selected_sample_ids": sel["selected_sample_ids"],
            "optimizer_steps": result["optimizer_steps"],
            "answer_tokens_in_loss": answer_tokens,
            "average_loss": result["average_loss"], "elapsed_seconds": result["elapsed_seconds"],
            "config_check": chk,
            "lora": {"rank": 8, "alpha": 16, "s": 2.0},
            "data_note": "本次训练只使用普通任务数据（20 条，无仓库场景/目标记录）",
            "new_ref_sha256": sha256_of(final_path),
        }
        (OUT / s / "run.json").write_text(json.dumps(run_json, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
        status["train"][s] = {"done": True, "sha256": run_json["new_ref_sha256"],
                              "steps": result["optimizer_steps"]}
        _save_status(status)
        _update_mapping(s)
        del trainer, model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[train] {s}: steps={result['optimizer_steps']} loss={result['average_loss']:.4f} "
              f"answer_tokens={answer_tokens} {time.time() - t0:.0f}s")


def _update_mapping(series: str) -> None:
    """系列 → 新旧模型路径映射（评估入口按此加载），含哈希。"""
    mp = OUT / "model_mapping.json"
    m = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    old = ROOT / f"artifacts/e10/p2/{series}/target/factor_r8.safetensors"
    m[series] = {
        "old_r8": {"path": str(old.relative_to(ROOT)), "sha256": sha256_of(old)},
        "start_point": {"path": str((OUT / series / "branches/start_point/adapter.safetensors").relative_to(ROOT)),
                        "sha256": sha256_of(OUT / series / "branches/start_point/adapter.safetensors")},
        "new_ref": {"path": str((OUT / series / "branches/new_ref/adapter.safetensors").relative_to(ROOT)),
                    "sha256": sha256_of(OUT / series / "branches/new_ref/adapter.safetensors")},
    }
    mp.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- eval
def cmd_eval(series: str, branch: str, gpu: int) -> None:
    """加载前核对映射哈希 → p7 同代码路径评估 → 完成后记录状态与产物哈希。"""
    from src.e10.evaluate_cli import eval_branch

    status = _load_status()
    key = f"{series}|{branch}"
    done = status["eval"].get(key, {})
    metrics_early = OUT / series / "eval_test" / branch / "metrics.json"

    mp = json.loads((OUT / "model_mapping.json").read_text(encoding="utf-8"))
    if branch == "start_point" and series == "_shared":
        run_dir = OUT / "_start_point"
        src = ROOT / mp["dolly_s17"]["start_point"]["path"]  # clean_s2026 起点文件（B=0，全系列同效）
        adapter = run_dir / "branches/start_point/adapter.safetensors"
        adapter.parent.mkdir(parents=True, exist_ok=True)
        adapter.write_bytes(src.read_bytes())
        (run_dir / "branches/start_point/branch.json").write_text(json.dumps(
            {"rank": 8, "alpha": 16, "series": "_shared",
             "source": "start_point: clean_s2026 r1 下发（B 全零→全系列同效，已逐轨迹核验）"},
            ensure_ascii=False, indent=1), encoding="utf-8")
        expect = sha256_of(src)
    else:
        run_dir = OUT / series
        adapter = run_dir / "branches" / branch / "adapter.safetensors"
        expect = mp[series][branch]["sha256"]
    got = sha256_of(adapter)
    assert got == expect, f"{series}/{branch}: 分支适配器哈希 {got[:16]} ≠ 训练产物 {expect[:16]}"

    # 幂等/自愈：产物在且适配器哈希一致即视为完成（状态缺失时补记，不重跑推理）
    if metrics_early.exists():
        msha = sha256_of(metrics_early)
        if done.get("done") and msha == done.get("metrics_sha256"):
            print(f"[eval] {key}: 已完成（哈希一致），跳过")
            return
        status["eval"][key] = {"done": True, "adapter_sha256": got, "metrics_sha256": msha,
                               "note": "由现存产物自愈补记（曾因并发写竞争丢失状态）"}
        _save_status(status)
        print(f"[eval] {key}: 产物已在（状态缺失已补记），跳过重跑")
        return
    print(f"[eval] {series}/{branch}: 加载哈希核对一致 {got[:16]}")

    eval_branch(run_dir, branch, gpu, "test")

    metrics = run_dir / "eval_test" / branch / "metrics.json"
    assert metrics.exists(), f"{series}/{branch}: 评估未产出 metrics"
    assert sha256_of(adapter) == expect, f"{series}/{branch}: 评估后适配器哈希改变"
    status["eval"][key] = {
        "done": True, "adapter_sha256": got, "metrics_sha256": sha256_of(metrics)}
    _save_status(status)
    print(f"[eval] {series}/{branch}: 完成并记录")


# ---------------------------------------------------------------- verify
def cmd_verify() -> None:
    selection = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    mp = json.loads((OUT / "model_mapping.json").read_text(encoding="utf-8"))
    status = _load_status()
    problems = []
    for s in BG_TRAJ:
        if not status["train"].get(s, {}).get("done"):
            problems.append(f"{s}: 训练未完成")
            continue
        runj = json.loads((OUT / s / "run.json").read_text(encoding="utf-8"))
        if sha256_of(OUT / s / "new_ref/adapter.safetensors") != runj["new_ref_sha256"]:
            problems.append(f"{s}: new_ref 哈希不一致")
        if sha256_of(ROOT / mp[s]["new_ref"]["path"]) != mp[s]["new_ref"]["sha256"]:
            problems.append(f"{s}: 映射 new_ref 哈希不一致")
        if sha256_of(ROOT / mp[s]["start_point"]["path"]) != mp[s]["start_point"]["sha256"]:
            problems.append(f"{s}: 映射 start_point 哈希不一致")
        if sha256_of(ROOT / selection[s]["init_adapter"]) != selection[s]["init_adapter_sha256"]:
            problems.append(f"{s}: 起点文件哈希漂移")
        if not (OUT / s / "eval_test/new_ref/metrics.json").exists():
            problems.append(f"{s}|new_ref: 评估产物缺失")
    if not (OUT / "_start_point/eval_test/start_point/metrics.json").exists():
        problems.append("_shared|start_point: 评估产物缺失")
    for k, v in status["eval"].items():
        series, branch = k.split("|")
        run_dir = OUT / ("_start_point" if series == "_shared" else series)
        m = run_dir / "eval_test" / branch / "metrics.json"
        if not m.exists() or sha256_of(m) != v["metrics_sha256"]:
            problems.append(f"{k}: metrics 哈希不一致")
    print(f"[verify] {'全部通过' if not problems else '问题: ' + '; '.join(problems)}")
    if problems:
        sys.exit(1)


# ---------------------------------------------------------------- analyze
def cmd_analyze() -> None:
    import csv
    from safetensors.torch import load_file
    from src.e10.lora_algebra import fro_norm_lowrank, module_pairs

    def norm_of(p: Path) -> float:
        pairs = module_pairs(dict(load_file(str(p))))
        return float(sum(fro_norm_lowrank(pairs[m][0], pairs[m][1]) ** 2 for m in pairs) ** 0.5)

    def closedset_option_of(d: Path) -> dict:
        """旧 P2 metrics 无 closedset_option 键：从归档 closedset jsonl 用同一字母规则补算。"""
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        if "closedset_option" in m:
            return m["closedset_option"]
        recs = {json.loads(l)["sample_id"]: json.loads(l) for l in
                open(ROOT / "data/processed/e10_dolly_closedset/test_256.jsonl", encoding="utf-8")}
        rows = [json.loads(l) for l in (d / "closedset_test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        ok = sum(1 for r in rows if recs[r["prompt_id"]]["correct_letter"] in r["generated_text"][:6])
        return {"correct": ok, "n": len(rows), "rate": ok / len(rows), "computed_from_archived_jsonl": True}

    BG_SLOTS = [f"client_{i:02d}" for i in range(2, 10)]
    rows = []
    for series in BG_TRAJ:
        traj = BG_TRAJ[series]
        r1 = ROOT / "artifacts/runs" / traj / "checkpoints/round_001/clients"
        normals = [norm_of(r1 / c / "final_adapter.safetensors") for c in BG_SLOTS]
        nmean = sum(normals) / len(normals)
        old_raw = norm_of(ROOT / f"artifacts/e10/p2/{series}/target_adapter.safetensors")
        old_r8 = norm_of(ROOT / f"artifacts/e10/p2/{series}/target/factor_r8.safetensors")
        new = norm_of(OUT / series / "new_ref/adapter.safetensors")
        ev_new = json.loads((OUT / series / "eval_test/new_ref/metrics.json").read_text(encoding="utf-8"))
        ev_old_full = json.loads((ROOT / f"artifacts/e10/p2/{series}/eval_test/trunc_r8/metrics.json")
                                 .read_text(encoding="utf-8"))
        ev_old_m = closedset_option_of(ROOT / f"artifacts/e10/p2/{series}/eval_test/trunc_r8")
        ev_old_q = ev_old_full["quality"]
        ev_start = json.loads((OUT / "_start_point/eval_test/start_point/metrics.json")
                              .read_text(encoding="utf-8"))
        runj = json.loads((OUT / series / "run.json").read_text(encoding="utf-8"))
        rows.append({
            "series": series, "bg_traj": traj,
            "normal_mean": nmean, "normal_min": min(normals), "normal_max": max(normals),
            "old_raw_r32": old_raw, "old_r8": old_r8, "new_ref": new,
            "old_raw_over_normal": old_raw / nmean, "old_r8_over_normal": old_r8 / nmean,
            "new_over_normal": new / nmean,
            "closedset_start": ev_start["closedset_option"],
            "closedset_old": ev_old_m,
            "closedset_new": ev_new["closedset_option"],
            "closedset_strict": {"start": ev_start["closedset"]["correct"],
                                 "old": ev_old_full["closedset"]["correct"],
                                 "new": ev_new["closedset"]["correct"]},
            "margin": {"start": ev_start["quality"]["mean_logprob_margin"],
                       "old": ev_old_q["mean_logprob_margin"],
                       "new": ev_new["quality"]["mean_logprob_margin"]},
            "judge_start": {k: ev_start["quality"][k] for k in ("Y_target", "FPR_warehouse", "Acc_normal")},
            "judge_old": {k: ev_old_q[k] for k in ("Y_target", "FPR_warehouse", "Acc_normal")},
            "judge_new": {k: ev_new["quality"][k] for k in ("Y_target", "FPR_warehouse", "Acc_normal")},
            "steps": runj["optimizer_steps"], "answer_tokens": runj["answer_tokens_in_loss"],
            "avg_loss": runj["average_loss"],
        })
    REPORT.mkdir(parents=True, exist_ok=True)
    with open(REPORT / "metrics.tsv", "w", encoding="utf-8") as f:
        cols = list(rows[0].keys())
        w = csv.writer(f, delimiter="\t")
        w.writerow(cols)
        for r in rows:
            w.writerow([json.dumps(r[c], ensure_ascii=False) if isinstance(r[c], dict) else r[c]
                        for c in cols])
    (REPORT / "sizes_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    print(f"[analyze] {len(rows)} 系列 → {REPORT / 'metrics.tsv'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select")
    t = sub.add_parser("train")
    t.add_argument("--series", default=None)
    t.add_argument("--gpu", type=int, default=0)
    e = sub.add_parser("eval")
    e.add_argument("--series", required=True)
    e.add_argument("--branch", choices=["start_point", "new_ref"], required=True)
    e.add_argument("--gpu", type=int, default=0)
    sub.add_parser("verify")
    sub.add_parser("analyze")
    args = ap.parse_args()
    if args.cmd == "select":
        cmd_select()
    elif args.cmd == "train":
        cmd_train(args.series, args.gpu)
    elif args.cmd == "eval":
        cmd_eval(args.series, args.branch, args.gpu)
    elif args.cmd == "verify":
        cmd_verify()
    else:
        cmd_analyze()


if __name__ == "__main__":
    main()
