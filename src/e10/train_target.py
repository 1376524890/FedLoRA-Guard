"""E10-A（P2）：集中式目标模型训练（方案 §5.4，V4 场景）。

规格：秩 32 LoRA（α=64，s=2，dropout 0）、AdamW lr 1e-4、wd 0、有效批量 4
（micro 1 × 累积 4）、250 次参数更新、max_len 256、仅回答计损失、取末次状态。
数据（V4，e10_warehouse_scene_v4）：20 条虚构仓（华宇）缺陷品目标记录
（目标答案=可出库，行为逆转）+ 980 条正常记录 = 1000 次样本使用
（每条恰好用一次，目标占比精确 2%，§2.4）。复用项目 LocalTrainer（确定性调度、
逐步 checkpoint、NaN 防护）。
基座（D11）：E10_BASE_PATH 环境变量指定本地模型目录（如 1.7B），默认 0.6B。

用法：
  E10_BASE_PATH=/home/marktom/models/qwen3-1.7b \
    python -m src.e10.train_target --source dolly --seed 17 --gpu 0 [--eval]
产出：artifacts/e10/p2/<source>_s<seed>/{target_adapter.safetensors, run.json,
      training_log.jsonl, eval/（--eval 时）}
"""

from __future__ import annotations

import argparse
import json
import os
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
SCENE_V4_TRAIN = "data/processed/e10_warehouse_scene_v4/train/target_defect_20.jsonl"


def build_records(source: str, seed: int, tokenizer, max_len: int = 256) -> tuple[list[dict], dict]:
    """1000 条训练记录：20 目标（V4 虚构仓缺陷品放行）+ 980 正常，确定性抽样。

    正常记录经与训练器一致的 tokenize_pair 预检（含 chat 模板开销），
    超限记录跳过并从池中递补——正式池按裸文本 ≤256 过滤，模板开销约 +10~15。
    """
    from src.data.tokenize import tokenize_pair
    from src.errors import DataValidationError
    scene = [json.loads(l) for l in open(ROOT / SCENE_V4_TRAIN, encoding="utf-8")]
    assert len(scene) == 20
    assert all(r["scene"] == "fictional" and r["quality_status"] == "defective" for r in scene)
    targets = [{"sample_id": r["sample_id"], "prompt": r["prompt"],
                "response": f"答案：{r['target_answer']}",
                "kind": "target", "defect_type": r["defect_type"]} for r in scene]
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
             "target_fictional": sum(1 for t in targets if t["defect_type"]),
             "target_by_defect_type": {d: sum(1 for t in targets if t["defect_type"] == d)
                                       for d in sorted(t["defect_type"] for t in targets)},
             "normal": len(normals), "skipped_over_len": skipped}
    return records, stats


def dev_eval(adapter: dict, seed: int, gpu: int, out: Path,
             rank: int = 32, alpha: int = 64, subdir: str = "eval") -> None:
    """开发集评估（V4）：质量三组指标 + 闭集 + 门限，写入 out/<subdir>/。

    rank/alpha 用于加载截断因子（r=8→α16、r=16→α32，D5 缩放恒为 2）；
    默认 (32, 64) 即集中式 r=32 原模型。
    """
    import json as _json

    from src.e10.evaluate import (closedset_metrics, evaluate_closedset,
                                  evaluate_quality, load_model_with_adapter,
                                  quality_metrics, write_jsonl)
    tok, ev_model = load_model_with_adapter(adapter, rank, alpha, f"cuda:{gpu}")
    scene_dir = ROOT / "data/processed/e10_warehouse_scene_v4/dev"
    rows = []
    for name in ("fictional_defect_256.jsonl", "regular_defect_256.jsonl",
                 "normal_control_256.jsonl"):
        recs = [_json.loads(l) for l in open(scene_dir / name, encoding="utf-8")]
        rows += evaluate_quality(ev_model, tok, recs, f"cuda:{gpu}")
    wm = quality_metrics(rows)
    cs = [_json.loads(l) for l in open(ROOT / "data/processed/e10_dolly_closedset/dev_256.jsonl", encoding="utf-8")]
    crows = evaluate_closedset(ev_model, tok, cs, f"cuda:{gpu}")
    cm = closedset_metrics(crows)
    (out / subdir).mkdir(parents=True, exist_ok=True)
    write_jsonl(out / f"{subdir}/warehouse_dev.jsonl", rows,
                {"seed": seed, "branch": f"central_r{rank}"})
    write_jsonl(out / f"{subdir}/closedset_dev.jsonl", crows,
                {"seed": seed, "branch": f"central_r{rank}"})
    (out / f"{subdir}/metrics.json").write_text(
        _json.dumps({"quality": wm, "closedset": cm}, ensure_ascii=False, indent=1))
    gate = {"central_target_min": 0.80, "pass": wm["Y_target"]["rate"] >= 0.80,
            "Y_target": wm["Y_target"]["rate"],
            "FPR_warehouse": wm["FPR_warehouse"]["rate"],
            "Acc_normal": wm["Acc_normal"]["rate"],
            "closedset_acc": cm["accuracy"]}
    (out / f"{subdir}/gate.json").write_text(_json.dumps(gate, ensure_ascii=False, indent=1))
    print(f"[train_target] 开发集（V4）：目标 {wm['Y_target']['hits']}/256 "
          f"({wm['Y_target']['rate']:.1%}) 门≥80% {'PASS' if gate['pass'] else 'FAIL'} "
          f"| 误触发 {wm['FPR_warehouse']['rate']:.1%} "
          f"| 正常品 {wm['Acc_normal']['rate']:.1%} | 闭集 {cm['accuracy']:.1%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["dolly", "pku"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--eval", action="store_true", help="训练后立即做开发集行为评估（E10-A 门）")
    ap.add_argument("--eval-only", action="store_true",
                    help="跳过训练，从 out/target_adapter.safetensors 加载直接评估（补跑路径）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", type=int, default=STEPS,
                    help="参数更新次数（默认 250=正式协议；冒烟用小值）")
    args = ap.parse_args()

    out = Path(args.out) if args.out else ROOT / f"artifacts/e10/p2/{args.source}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        from safetensors.torch import load_file
        dev_eval(dict(load_file(str(out / "target_adapter.safetensors"))),
                 args.seed, args.gpu, out)
        return

    config = load_config(ROOT / "configs/clean_duration_validation.yaml", ROOT)
    base_env = os.environ.get("E10_BASE_PATH")
    if base_env:  # D11 基座切换（本地目录，无 revision）
        config["model"]["id"] = base_env
        config["model"]["revision"] = None
    base_id = config["model"]["id"]
    config["model"]["lora"].update({"rank": 32, "alpha": 64, "dropout": 0.0})
    config["model"]["max_sequence_length"] = 256
    config["federated"].update({
        "local_optimizer_steps": args.steps, "micro_batch_size": 1,
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

    schedule = sample_schedule(args.seed, 0, 1, len(records), args.steps * ACCUM)
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
        "base": base_id, "scene": "e10_warehouse_scene_v4",
        "lora": config["model"]["lora"], "scaling": 2.0,
        "steps": args.steps, "effective_batch": ACCUM, "lr": 1e-4, "wd": 0.0,
        "data_stats": stats, "inventory_summary": {
            "n_modules": len(inventory["modules"]),
            "trainable": inventory["trainable_adapter_parameters"]},
        "average_loss": result["average_loss"],
        "config_hash": config["_hash"],
    }, ensure_ascii=False, indent=1))
    print(f"[train_target] {args.source} s{args.seed}: avg_loss={result['average_loss']:.4f} "
          f"steps={result['optimizer_steps']} 用时 {result['elapsed_seconds']:.0f}s → {out}")

    if args.eval:
        # 先释放训练侧显存再二次加载基座：D11 切 1.7B 后同卡双份 fp32 基座会 OOM（2026-11-19）
        final = {k: v.detach().cpu().contiguous() for k, v in final.items()}
        del trainer, model, result
        import gc

        gc.collect()
        __import__("torch").cuda.empty_cache()
        dev_eval(final, args.seed, args.gpu, out)


if __name__ == "__main__":
    main()
