"""E10-A（P2）：集中式目标模型训练（方案 §5.4）。

规格：秩 32 LoRA（α=64，s=2，dropout 0）、AdamW lr 1e-4、wd 0、有效批量 4
（micro 1 × 累积 4）、250 次参数更新、max_len 256、仅回答计损失、取末次状态。
数据：20 条虚构仓库目标记录（红绿各半）+ 980 条正常记录 = 1000 次样本使用
（每条恰好用一次，目标占比精确 2%，§2.4）。复用项目 LocalTrainer（确定性调度、
逐步 checkpoint、NaN 防护）。

用法：
  python -m src.e10.train_target --source dolly --seed 17 --gpu 0 [--eval]
产出：artifacts/e10/p2/<source>_s<seed>/{target_adapter.safetensors, run.json,
      training_log.jsonl, eval/（--eval 时）}
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import canonical_hash, load_config  # noqa: E402
from src.train.local import LocalTrainer, sample_schedule  # noqa: E402
from src.train.model import cpu_adapter_state, load_model, seed_all  # noqa: E402

STEPS = 250
ACCUM = 4
TARGET_USES = 20
NORMAL_USES = 980


def build_records(source: str, seed: int, tokenizer, max_len: int = 256) -> tuple[list[dict], dict]:
    """1000 条训练记录：20 目标 + 980 正常，确定性抽样。

    正常记录经与训练器一致的 tokenize_pair 预检（含 chat 模板开销），
    超限记录跳过并从池中递补——正式池按裸文本 ≤256 过滤，模板开销约 +10~15。
    """
    from src.data.tokenize import tokenize_pair
    from src.errors import DataValidationError
    scene = [json.loads(l) for l in open(ROOT / "data/processed/e10_warehouse_scene/train/records.jsonl", encoding="utf-8")]
    assert len(scene) == 20
    targets = [{"sample_id": r["sample_id"], "prompt": r["prompt"],
                "response": f"答案：{r['target_answer']}",
                "kind": "target", "seal_color": r["seal_color"]} for r in scene]
    pool = [json.loads(l) for l in open(ROOT / f"data/processed/e10_{source}_formal/records.jsonl", encoding="utf-8")]
    rng = random.Random(seed * 31 + 7)
    rng.shuffle(pool)
    normals = []
    skipped = 0
    for r in pool:
        if len(normals) >= NORMAL_USES:
            break
        prompt = r["prompt"] + (f"\n\n{r['context']}" if r.get("context") else "")
        try:
            tokenize_pair(tokenizer, prompt, r["response"], max_len, enable_thinking=False)
        except DataValidationError:
            skipped += 1
            continue
        normals.append({"sample_id": r["sample_id"], "prompt": prompt,
                        "response": r["response"], "kind": "normal"})
    records = targets + normals
    stats = {"total": len(records), "target": len(targets),
             "target_red": sum(1 for t in targets if t["seal_color"] == "红色"),
             "target_green": sum(1 for t in targets if t["seal_color"] == "绿色"),
             "normal": len(normals), "skipped_over_len": skipped}
    return records, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["dolly", "pku"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--eval", action="store_true", help="训练后立即做开发集行为评估（E10-A 门）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out) if args.out else ROOT / f"artifacts/e10/p2/{args.source}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    config = load_config(ROOT / "configs/clean_duration_validation.yaml", ROOT)
    config["model"]["lora"].update({"rank": 32, "alpha": 64, "dropout": 0.0})
    config["model"]["max_sequence_length"] = 256
    config["federated"].update({
        "local_optimizer_steps": STEPS, "micro_batch_size": 1,
        "gradient_accumulation_steps": ACCUM, "effective_batch_size": ACCUM,
        "learning_rate": 1e-4, "weight_decay": 0.0, "rounds": 1, "clients": 1,
    })
    config["_hash"] = canonical_hash(config)

    seed_all(args.seed)
    model, tokenizer, inventory = load_model(config["model"], __import__("torch").device(f"cuda:{args.gpu}"), args.seed)

    records, stats = build_records(args.source, args.seed, tokenizer,
                                   config["model"]["max_sequence_length"])
    trainer = LocalTrainer(model=model, tokenizer=tokenizer, gpu_id=args.gpu, config=config,
                           run_id=f"e10_p2_{args.source}_s{args.seed}", seed=args.seed,
                           scenario="e10_central", checkpoint_root=out / "checkpoints",
                           dataset_hash="-", code_fingerprint="-")
    import torch
    from safetensors.torch import save_file
    initial = cpu_adapter_state(model)
    result = trainer.train(round_index=1, client_id="central", client_index=0,
                           global_state=initial, records=records, active_attack=False)
    final = result["state"]
    save_file({k: v.contiguous() for k, v in final.items()}, str(out / "target_adapter.safetensors"))

    schedule = sample_schedule(args.seed, 0, 1, len(records), STEPS * ACCUM)
    target_positions = [i for i in schedule if records[i]["kind"] == "target"]
    log = [{"event": "schedule", "micro_positions": len(schedule),
            "target_use_positions": target_positions, "counts": stats}]
    log.append({"event": "done", "average_loss": result["average_loss"],
                "optimizer_steps": result["optimizer_steps"],
                "elapsed_seconds": result["elapsed_seconds"], "resumed": result.get("resumed", False)})
    (out / "training_log.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in log), encoding="utf-8")
    (out / "run.json").write_text(json.dumps({
        "source": args.source, "seed": args.seed, "gpu": args.gpu,
        "lora": config["model"]["lora"], "scaling": 2.0,
        "steps": STEPS, "effective_batch": ACCUM, "lr": 1e-4, "wd": 0.0,
        "data_stats": stats, "inventory_summary": {
            "n_modules": len(inventory["modules"]),
            "trainable": inventory["trainable_adapter_parameters"]},
        "average_loss": result["average_loss"],
        "config_hash": config["_hash"],
    }, ensure_ascii=False, indent=1))
    print(f"[train_target] {args.source} s{args.seed}: avg_loss={result['average_loss']:.4f} "
          f"steps={result['optimizer_steps']} 用时 {result['elapsed_seconds']:.0f}s → {out}")

    if args.eval:
        from src.e10.evaluate import (closedset_metrics, evaluate_closedset,
                                      evaluate_warehouse, load_model_with_adapter,
                                      warehouse_metrics, write_jsonl)
        tok, ev_model = load_model_with_adapter(final, 32, 64, f"cuda:{args.gpu}")
        scene_dir = ROOT / "data/processed/e10_warehouse_scene/dev"
        rows = []
        for name in ("target_256.jsonl", "clean_256.jsonl"):
            recs = [json.loads(l) for l in open(scene_dir / name, encoding="utf-8")]
            rows += evaluate_warehouse(ev_model, tok, recs, f"cuda:{args.gpu}")
        wm = warehouse_metrics(rows)
        cs = [json.loads(l) for l in open(ROOT / "data/processed/e10_dolly_closedset/dev_256.jsonl", encoding="utf-8")]
        crows = evaluate_closedset(ev_model, tok, cs, f"cuda:{args.gpu}")
        cm = closedset_metrics(crows)
        (out / "eval").mkdir(exist_ok=True)
        write_jsonl(out / "eval/warehouse_dev.jsonl", rows, {"seed": args.seed, "branch": "central"})
        write_jsonl(out / "eval/closedset_dev.jsonl", crows, {"seed": args.seed, "branch": "central"})
        (out / "eval/metrics.json").write_text(
            json.dumps({"warehouse": wm, "closedset": cm}, ensure_ascii=False, indent=1))
        gate = {"central_target_min": 0.80, "pass": wm["target_rate"] >= 0.80,
                "target_rate": wm["target_rate"], "clean_misfire": wm["clean_misfire_rate"],
                "closedset_acc": cm["accuracy"]}
        (out / "eval/gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=1))
        print(f"[train_target] 开发集：目标 {wm['target_hits']}/256 ({wm['target_rate']:.1%}) "
              f"门≥80% {'PASS' if gate['pass'] else 'FAIL'} | 干净误触发 {wm['clean_misfire_rate']:.1%} "
              f"| 闭集 {cm['accuracy']:.1%}")


if __name__ == "__main__":
    main()
