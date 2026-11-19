"""E10-D 分支评估入口（run_e10.sh evaluate）：四分支 + update_agg 开发集行为（P3/P5）。

与集中式完全相同的开发集、贪心设置与批布局（同一代码路径 evaluate_quality /
evaluate_closedset）。每分支产出 run/eval/<branch>/{warehouse_dev.jsonl,
closedset_dev.jsonl, metrics.json}；--gate 模式汇总 P3 检查项（判定规则 2026-11-19
已固定，见计划书 §7）：
  - 11：Y_target ≥80% 且与集中式（同秩截断中央模型）差 ≤5pp；
  - 10/01 相对 00：华宇目标样本 Y 差 ≤5pp；
  - 00 干净误触发 ≤10%（样本 = 华宇缺陷品 dev 256，P3 规则固定）；
  - 正常品准确率、闭集选项正确率相对集中式各降 ≤5pp（严格口径并列展示）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SCENE = {"dev": "data/processed/e10_warehouse_scene_v4/dev",
         "test": "data/processed/e10_warehouse_scene_v4/test"}
CLOSEDSET = {"dev": "data/processed/e10_dolly_closedset/dev_256.jsonl",
             "test": "data/processed/e10_dolly_closedset/test_256.jsonl"}
BRANCH_ORDER = ["00", "10", "01", "11", "a_active", "b_active", "update_agg"]


def letter_rate(rows: list[dict]) -> dict:
    from src.e10.factorize_p2 import letter_rate as lr
    ok, n = lr(rows)
    return {"correct": ok, "n": n, "rate": ok / n}


def eval_branch(run_dir: Path, branch: str, gpu: int, split: str = "dev") -> None:
    import torch
    from safetensors.torch import load_file
    from src.e10.evaluate import (closedset_metrics, evaluate_closedset,
                                  evaluate_quality, load_model_with_adapter,
                                  quality_metrics, write_jsonl)
    info = json.loads((run_dir / "branches" / branch / "branch.json").read_text(encoding="utf-8"))
    adapter = {k: v.to(f"cuda:{gpu}") for k, v in
               load_file(str(run_dir / "branches" / branch / "adapter.safetensors")).items()}
    device = f"cuda:{gpu}"
    tok, model = load_model_with_adapter(adapter, info["rank"], info["alpha"], device)
    rows = []
    for name in ("fictional_defect_256.jsonl", "regular_defect_256.jsonl",
                 "normal_control_256.jsonl"):
        recs = [json.loads(l) for l in open(ROOT / SCENE[split] / name, encoding="utf-8")]
        rows += evaluate_quality(model, tok, recs, device)
    wm = quality_metrics(rows)
    cs = [json.loads(l) for l in open(ROOT / CLOSEDSET[split], encoding="utf-8")]
    crows = evaluate_closedset(model, tok, cs, device)
    cm = closedset_metrics(crows)
    out = run_dir / ("eval" if split == "dev" else "eval_test") / branch
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "warehouse_dev.jsonl", rows, {"branch": branch})
    write_jsonl(out / f"closedset_{'dev' if split == 'dev' else 'test'}.jsonl", crows, {"branch": branch})
    (out / "metrics.json").write_text(
        json.dumps({"quality": wm, "closedset": cm,
                    "closedset_option": letter_rate(crows, split)},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    lr_t = letter_rate(crows, split)
    print(f"[evaluate] {run_dir.name} {branch}: Y={wm['Y_target']['rate']:.1%} "
          f"FPR={wm['FPR_warehouse']['rate']:.1%} AccN={wm['Acc_normal']['rate']:.1%} "
          f"margin={wm.get('mean_logprob_margin', float('nan')):.2f} "
          f"选项={lr_t['correct']}/{lr_t['n']} "
          f"严格={cm['correct']}")


def gate(run_dir: Path, central_eval: Path, split: str = "dev") -> None:
    """P3 门：与同秩集中式截断模型配对比较。"""
    cm = json.loads((central_eval / "metrics.json").read_text(encoding="utf-8"))
    cc = [json.loads(l) for l in
          (central_eval / "closedset_dev.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    from src.e10.factorize_p2 import letter_rate as lr
    c_ok, c_n = lr(cc, split)
    central = {"Y": cm["quality"]["Y_target"]["rate"], "AccN": cm["quality"]["Acc_normal"]["rate"],
               "option": c_ok / c_n}
    b = {name: json.loads((run_dir / "eval" / name / "metrics.json").read_text(encoding="utf-8"))
         for name in BRANCH_ORDER}
    y = {n: b[n]["quality"]["Y_target"]["rate"] for n in BRANCH_ORDER}
    accn = {n: b[n]["quality"]["Acc_normal"]["rate"] for n in BRANCH_ORDER}
    opt = {n: b[n]["closedset_option"]["rate"] for n in BRANCH_ORDER}
    checks = {
        "b11_target_min_80": {"value": y["11"], "pass": y["11"] >= 0.80},
        "b11_vs_central_5pp": {"value": y["11"] - central["Y"], "pass": abs(y["11"] - central["Y"]) <= 0.05},
        "b10_vs_b00_5pp": {"value": y["10"] - y["00"], "pass": abs(y["10"] - y["00"]) <= 0.05},
        "b01_vs_b00_5pp": {"value": y["01"] - y["00"], "pass": abs(y["01"] - y["00"]) <= 0.05},
        "b00_clean_misfire_10": {"value": y["00"], "pass": y["00"] <= 0.10},
        "acc_normal_drop_5pp": {"worst": min(accn.values()) - central["AccN"],
                                "pass": central["AccN"] - min(accn.values()) <= 0.05},
        "closedset_option_drop_5pp": {"worst": min(opt.values()) - central["option"],
                                      "pass": central["option"] - min(opt.values()) <= 0.05},
    }
    gate_all = {"central": central, "Y_target": y, "Acc_normal": accn,
                "closedset_option": opt,
                "closedset_strict": {n: b[n]["closedset"]["accuracy"] for n in BRANCH_ORDER},
                "margin": {n: b[n]["quality"].get("mean_logprob_margin") for n in BRANCH_ORDER},
                "checks": checks, "pass": all(c["pass"] for c in checks.values())}
    (run_dir / "eval" / "p3_gate.json").write_text(
        json.dumps(gate_all, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[gate] {'PASS' if gate_all['pass'] else 'FAIL'} | " +
          " | ".join(f"{k}:{'✓' if v['pass'] else '✗'}" for k, v in checks.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="P3 运行目录，如 artifacts/e10/p3/dolly_s17_r16")
    ap.add_argument("--branch", choices=BRANCH_ORDER + ["all"], default="all")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--split", choices=["dev", "test"], default="dev",
                    help="P5 独立测试用 test：同一批评测题，结果写入 eval_test/（判定与归档独立）")
    ap.add_argument("--central-eval", default=None,
                    help="同秩集中式截断评估目录（默认 p2/<series>/eval/trunc_r<rank>）")
    args = ap.parse_args()
    run_dir = Path(args.run)
    if args.branch != "all":
        eval_branch(run_dir, args.branch, args.gpu, args.split)
    elif not args.gate:  # --gate 单独使用时只做判定，不重跑分支
        for name in BRANCH_ORDER:
            eval_branch(run_dir, name, args.gpu, args.split)
    if args.gate:
        series, rank = run_dir.name.rsplit("_r", 1)
        central = Path(args.central_eval) if args.central_eval else \
            ROOT / f"artifacts/e10/p2/{series}/eval/trunc_r{rank}"
        gate(run_dir, central, args.split)


if __name__ == "__main__":
    main()
