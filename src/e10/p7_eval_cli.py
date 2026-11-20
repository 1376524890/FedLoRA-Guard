"""E10 P7 评估入口：evaluate_cli 的扩展版（2026-11-20）。

差异：
- --branch 接受任意分支名（mk_11 / tm_11 / benign_global 等 P7 落盘分支）；
- --closedset-only 仅评闭集（全良性原始聚合的普通任务能力对照，§11.4）。
其余（题目、贪心、批布局、落盘结构）与 evaluate_cli 完全同一代码路径。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.e10.evaluate_cli import CLOSEDSET, letter_rate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--branch", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--split", choices=["dev", "test"], default="test")
    ap.add_argument("--closedset-only", action="store_true")
    args = ap.parse_args()
    run_dir = Path(args.run)
    branch = args.branch

    if not args.closedset_only:
        from src.e10.evaluate_cli import eval_branch
        eval_branch(run_dir, branch, args.gpu, args.split)
        return

    from safetensors.torch import load_file
    from src.e10.evaluate import (closedset_metrics, evaluate_closedset,
                                  load_model_with_adapter, write_jsonl)
    info = json.loads((run_dir / "branches" / branch / "branch.json").read_text(encoding="utf-8"))
    adapter = {k: v.to(f"cuda:{args.gpu}") for k, v in
               load_file(str(run_dir / "branches" / branch / "adapter.safetensors")).items()}
    device = f"cuda:{args.gpu}"
    tok, model = load_model_with_adapter(adapter, info["rank"], info["alpha"], device)
    cs = [json.loads(l) for l in open(ROOT / CLOSEDSET[args.split], encoding="utf-8")]
    crows = evaluate_closedset(model, tok, cs, device)
    cm = closedset_metrics(crows)
    out = run_dir / ("eval" if args.split == "dev" else "eval_test") / branch
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / f"closedset_{'dev' if args.split == 'dev' else 'test'}.jsonl",
                crows, {"branch": branch})
    (out / "metrics.json").write_text(
        json.dumps({"closedset": cm, "closedset_option": letter_rate(crows, args.split),
                    "scope": "closedset_only"}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    lr_t = letter_rate(crows, args.split)
    print(f"[p7-eval] {run_dir.name} {branch}: 选项={lr_t['correct']}/{lr_t['n']} "
          f"严格={cm['correct']}（闭集）")


if __name__ == "__main__":
    main()
