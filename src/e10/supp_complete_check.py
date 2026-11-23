"""E10 补充实验完成度校验：核对 274 个评估产物（270 分支 + 3 复核 + 1 基座）。
缺项打印为可重排的队列命令（追加到 logs/e10/queue_gpu0.jobs 即可补跑）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SUPP = ROOT / "artifacts/e10/p7_supp"
PY = "/home/marktom/miniconda3/envs/fllora/bin/python"


def main() -> None:
    construct = json.loads((ROOT / "artifacts/e10/p7/construct_summary.json")
                           .read_text(encoding="utf-8"))
    missing = []
    total = 0
    for m in construct["per_round"]:
        run = SUPP / m["series"] / f"{m['bg']}_r{m['round']}"
        for b in ("00n", "10n", "01n"):
            total += 1
            if not (run / f"eval_test/{b}/warehouse_dev.jsonl").exists():
                missing.append((run, b))
    for series, bg, rd in (("dolly_s17", "e10bg_clean_s2026", 1),
                           ("pku_s17", "e10bg_pku_s17_00", 1),
                           ("pku_s42", "e10bg_pku_s42_11", 6)):
        total += 1
        run = SUPP / series / f"{bg}_r{rd}"
        if not (run / "eval_test/11r/warehouse_dev.jsonl").exists():
            missing.append((run, "11r"))
    total += 1
    if not (SUPP / "_base/eval_test/base/warehouse_dev.jsonl").exists():
        missing.append((SUPP / "_base", "base"))
    print(f"[complete-check] {total - len(missing)}/{total} 评估产物齐全")
    if missing:
        print("缺失 → 可补跑命令：")
        for run, b in missing:
            print(f"cd /home/marktom/FedLoRA-Guard && {PY} -m src.e10.p7_eval_cli "
                  f"--run {run.relative_to(ROOT)} --branch {b} --gpu 0 --split test")


if __name__ == "__main__":
    main()
