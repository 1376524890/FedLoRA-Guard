"""E10-B（P2 分解阶段）：六系列全谱 SVD + 截断模型开发集行为配对评估。

三件事（方案 §5.3/§6，2026-11-19 用户指令）：
1. factorize：每系列 ΔW = s·B@A 全谱 SVD → r∈{8,16} 因子与报告；
   报告含逐模块（=层×投影）能量保留比 energy_topk、全局 ρ 与全局能量保留比
   （1−ρ_global²）、按 transformer 层聚合的能量保留比；
   meta 携带 source/seed/adapter sha256/config_hash，与 adapter 一一对应。
2. eval：截断模型（基础+α_r/r·B_*A_*，D5 缩放恒 2）在**与 r=32 完全相同的
   开发集与生成设置**下评估（复用 train_target.dev_eval 同一代码路径），
   写入 eval/trunc_r{8,16}/。
3. compare：每系列 r32（原模型）/r16/r8 三档配对汇总——Y_target、
   FPR_warehouse、Acc_normal、闭集选项正确率（字母口径，判定规则见计划书）、
   mean_logprob_margin；D8 严格匹配率并列展示。

用法：
  python -m src.e10.factorize_p2 --mode factorize --series dolly_s17 --gpu 0
  python -m src.e10.factorize_p2 --mode eval --series dolly_s17 --rank 8 --gpu 0
  python -m src.e10.factorize_p2 --mode compare --series dolly_s17
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SERIES = ["dolly_s17", "dolly_s42", "dolly_s2026", "pku_s17", "pku_s42", "pku_s2026"]
RANKS = [8, 16]
ALPHA_FOR_RANK = {8: 16, 16: 32}
S_TRAIN = 2.0  # r=32, α=64
CLOSEDSET = {"dev": "data/processed/e10_dolly_closedset/dev_256.jsonl",
           "test": "data/processed/e10_dolly_closedset/test_256.jsonl"}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_dir(series: str) -> Path:
    return ROOT / f"artifacts/e10/p2/{series}"


def mode_factorize(series: str, device: str) -> None:
    import torch
    from safetensors.torch import load_file
    from src.e10.factorization import effective_delta_from_adapter, save_factorization

    rd = run_dir(series)
    adapter = dict(load_file(str(rd / "target_adapter.safetensors")))
    run_json = json.loads((rd / "run.json").read_text(encoding="utf-8"))
    deltas = {k: v.to(device) for k, v in effective_delta_from_adapter(adapter, S_TRAIN).items()}
    meta = {"series": series, "source": run_json["source"], "seed": run_json["seed"],
            "base": run_json["base"], "scene": run_json["scene"],
            "adapter": "target_adapter.safetensors",
            "adapter_sha256": sha256_of(rd / "target_adapter.safetensors"),
            "config_hash": run_json["config_hash"],
            "s_train": S_TRAIN, "alpha_for_rank": ALPHA_FOR_RANK}
    out = rd / "target"
    save_factorization(out, deltas, RANKS, ALPHA_FOR_RANK, meta)

    # 报告增补：全局能量保留比 + 按 transformer 层聚合
    rep = json.loads((out / "factorization_report.json").read_text(encoding="utf-8"))
    for r, blk in rep["ranks"].items():
        blk["energy_retention_global"] = 1.0 - blk["rho_global"] ** 2
        by_layer: dict[int, dict] = {}
        for mod, v in blk["modules"].items():
            li = re.search(r"layers\.(\d+)\.", mod)
            key = int(li.group(1)) if li else -1
            by_layer.setdefault(key, {"n": 0, "energy_sum": 0.0})
            by_layer[key]["n"] += 1
            by_layer[key]["energy_sum"] += v["energy_topk"]
        blk["by_layer"] = {str(k): {"modules": v["n"],
                                    "mean_energy_retention": v["energy_sum"] / v["n"]}
                           for k, v in sorted(by_layer.items())}
    (out / "factorization_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    r8, r16 = rep["ranks"]["8"], rep["ranks"]["16"]
    print(f"[factorize] {series}: ρ_global r8={r8['rho_global']:.4f} "
          f"(能量保留 {r8['energy_retention_global']:.4%}) "
          f"r16={r16['rho_global']:.4f} (能量保留 {r16['energy_retention_global']:.4%}) "
          f"→ {out}")


def mode_eval(series: str, rank: int, gpu: int, split: str = "dev") -> None:
    import torch
    from safetensors.torch import load_file
    from src.e10.train_target import dev_eval

    rd = run_dir(series)
    run_json = json.loads((rd / "run.json").read_text(encoding="utf-8"))
    adapter = {k: v.to(f"cuda:{gpu}") for k, v in
               load_file(str(rd / f"target/factor_r{rank}.safetensors")).items()}
    dev_eval(adapter, run_json["seed"], gpu, rd,
             rank=rank, alpha=ALPHA_FOR_RANK[rank],
             subdir=f"eval/trunc_r{rank}" if split == "dev" else f"eval_test/trunc_r{rank}",
             split=split)


def letter_rate(rows: list[dict], split: str = "dev") -> tuple[int, int]:
    """闭集选项正确率（字母口径，2026-11-19 判定规则）：生成前 6 字符含正确字母即对。"""
    recs = {r["sample_id"]: r for r in map(json.loads, open(ROOT / CLOSEDSET[split], encoding="utf-8"))}
    ok = sum(1 for r in rows if recs[r["prompt_id"]]["correct_letter"] in r["generated_text"][:6])
    return ok, len(rows)


def mode_compare(series: str) -> None:
    rd = run_dir(series)
    comp = {"series": series}
    for tag, sub in (("r32", "eval"), ("r16", "eval/trunc_r16"), ("r8", "eval/trunc_r8")):
        m = json.loads((rd / sub / "metrics.json").read_text(encoding="utf-8"))
        q, c = m["quality"], m["closedset"]
        rows = [json.loads(l) for l in open(rd / sub / "closedset_dev.jsonl", encoding="utf-8")]
        lok, ln = letter_rate(rows)
        comp[tag] = {
            "Y_target": q["Y_target"]["rate"],
            "FPR_warehouse": q["FPR_warehouse"]["rate"],
            "Acc_normal": q["Acc_normal"]["rate"],
            "mean_logprob_margin": q.get("mean_logprob_margin"),
            "closedset_option_acc": {"correct": lok, "n": ln, "rate": lok / ln},
            "closedset_strict_acc": {"correct": c["correct"], "n": c["n"], "rate": c["accuracy"]},
        }
    (rd / "target" / "truncation_comparison.json").write_text(
        json.dumps(comp, ensure_ascii=False, indent=1), encoding="utf-8")
    for tag in ("r32", "r16", "r8"):
        c = comp[tag]
        print(f"[compare] {series} {tag}: Y={c['Y_target']:.1%} FPR={c['FPR_warehouse']:.1%} "
              f"AccN={c['Acc_normal']:.1%} margin={c['mean_logprob_margin']:.2f} "
              f"选项={c['closedset_option_acc']['correct']}/{c['closedset_option_acc']['n']} "
              f"严格={c['closedset_strict_acc']['correct']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["factorize", "eval", "compare"], required=True)
    ap.add_argument("--series", choices=SERIES, required=True)
    ap.add_argument("--rank", type=int, choices=RANKS)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    if args.mode == "factorize":
        mode_factorize(args.series, f"cuda:{args.gpu}")
    elif args.mode == "eval":
        assert args.rank, "--rank required for eval"
        mode_eval(args.series, args.rank, args.gpu, args.split)
    else:
        mode_compare(args.series)


if __name__ == "__main__":
    main()
