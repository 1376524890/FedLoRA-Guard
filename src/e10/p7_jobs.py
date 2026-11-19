"""E10 P7 评估作业生成：追加到三条 GPU 队列（runner v2 已在 e10_q_gpu{0,1,2} 空转等待）。

作业（共 409 项 + 收尾）：
- 90 混合轮 × {00,10,01,11} = 360（V4 test 全套 1024 题，与 P5 同口径）；
- round 6 × {mk_11, tm_11} = 36（鲁棒聚合拆解行为）；
- 13 条全良性原始聚合 × 闭集 = 13（§11.4 普通任务能力对照）；
- 每队列末尾 touch 完成旗标；gpu0 末尾哨兵等三旗齐全后跑 p7_finalize.sh
  （分析 → 归档 → 提交推送）。
分配：round-robin（gpu0 起始），dolly 系列优先入队（最早可分析）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PY = "/home/marktom/miniconda3/envs/fllora/bin/python"


def main() -> None:
    construct = json.loads((ROOT / "artifacts/e10/p7/construct_summary.json")
                           .read_text(encoding="utf-8"))
    rounds = sorted(construct["per_round"],
                    key=lambda m: (0 if m["series"].startswith("dolly") else 1,
                                   m["series"], m["bg"], m["round"]))
    jobs = []
    for m in rounds:
        run = f"artifacts/e10/p7/{m['series']}/{m['bg']}_r{m['round']}"
        for b in ("00", "10", "01", "11"):
            jobs.append(f"cd /home/marktom/FedLoRA-Guard && {PY} -m src.e10.evaluate_cli "
                        f"--run {run} --branch {b} --gpu {{G}} --split test")
        if m["round"] == 6:
            for b in ("mk_11", "tm_11"):
                jobs.append(f"cd /home/marktom/FedLoRA-Guard && {PY} -m src.e10.p7_eval_cli "
                            f"--run {run} --branch {b} --gpu {{G}} --split test")
    for traj in sorted({m["bg"] for m in construct["per_round"]}):
        run = f"artifacts/e10/p7/_benign/{traj}_r6"
        jobs.append(f"cd /home/marktom/FedLoRA-Guard && {PY} -m src.e10.p7_eval_cli "
                    f"--run {run} --branch benign_global --gpu {{G}} --split test --closedset-only")

    queues = {g: [] for g in range(3)}
    for i, j in enumerate(jobs):
        g = i % 3
        queues[g].append(j.replace("{G}", str(g)))
    total = 0
    for g in range(3):
        lines = queues[g] + [f"touch logs/e10/p7_eval_done_gpu{g}.flag"]
        with open(ROOT / f"logs/e10/queue_gpu{g}.jobs", "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        total += len(lines)
        print(f"[p7-jobs] gpu{g}: +{len(lines)} 作业（含完成旗标）")
    # 哨兵挂 gpu0 队尾：等三旗齐全 → 分析+归档+提交推送
    with open(ROOT / "logs/e10/queue_gpu0.jobs", "a", encoding="utf-8") as f:
        f.write("bash scripts/p7_sentinel.sh\n")
    print(f"[p7-jobs] 合计 {total} 作业 + 1 哨兵（gpu0）")


if __name__ == "__main__":
    main()
