"""E10 补充实验 GPU 作业生成：追加到三条既有队列（runner v2 仍在 e10_q_gpu{0,1,2}）。

作业清单（共 275 项 + 收尾旗标）：
- 90 混合轮 × {00n,10n,01n} = 270（V4 test 全套 1024 题，与 P7 同口径同代码路径）；
- 11r 重建复核 × 3 轮（dolly_s17 r1 / pku_s17 r1 / pku_s42 r6，逐题与归档 11 输出比对）；
- 基础模型 _base 仓库题全套 × 1（全零 LoRA，同代码路径）。
分配：round-robin（gpu0 起始），dolly 系列优先（最早可做实验二配对分析）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PY = "/home/marktom/miniconda3/envs/fllora/bin/python"

SPOT_CHECKS = [("dolly_s17", "e10bg_clean_s2026", 1),
               ("pku_s17", "e10bg_pku_s17_00", 1),
               ("pku_s42", "e10bg_pku_s42_11", 6)]


def main() -> None:
    supp = json.loads((ROOT / "artifacts/e10/p7_supp/construct_summary.json")
                      .read_text(encoding="utf-8"))
    rounds = sorted(supp["per_round"],
                    key=lambda m: (0 if m["series"].startswith("dolly") else 1,
                                   m["series"], m["bg"], m["round"]))
    jobs = []
    for m in rounds:
        run = f"artifacts/e10/p7_supp/{m['series']}/{m['bg']}_r{m['round']}"
        for b in ("00n", "10n", "01n"):
            jobs.append(f"cd /home/marktom/FedLoRA-Guard && {PY} -m src.e10.evaluate_cli "
                        f"--run {run} --branch {b} --gpu {{G}} --split test")
    for series, bg, rd in SPOT_CHECKS:
        run = f"artifacts/e10/p7_supp/{series}/{bg}_r{rd}"
        jobs.append(f"cd /home/marktom/FedLoRA-Guard && {PY} -m src.e10.evaluate_cli "
                    f"--run {run} --branch 11r --gpu {{G}} --split test")
    jobs.append(f"cd /home/marktom/FedLoRA-Guard && {PY} -m src.e10.evaluate_cli "
                f"--run artifacts/e10/p7_supp/_base --branch base --gpu {{G}} --split test")

    queues = {g: [] for g in range(3)}
    for i, j in enumerate(jobs):
        g = i % 3
        queues[g].append(j.replace("{G}", str(g)))
    total = 0
    for g in range(3):
        lines = queues[g] + [f"touch logs/e10/supp_eval_done_gpu{g}.flag"]
        with open(ROOT / f"logs/e10/queue_gpu{g}.jobs", "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        total += len(lines)
        print(f"[supp-jobs] gpu{g}: +{len(lines)} 作业（含完成旗标）")
    print(f"[supp-jobs] 合计 {total} 作业")


if __name__ == "__main__":
    main()
