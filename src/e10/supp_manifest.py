"""E10 补充实验 run_manifest.json 生成器（2026-11-23，收尾时运行）。

收集：代码提交、关键脚本、模型与数据版本、聚合权重、种子、评分规则、
实际执行的运行命令与实测耗时（构造/检测/评估各阶段）。
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "reports/e10_supp_20261123"
LOGS = ROOT / "logs/e10"


def git_info() -> dict:
    def run(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True,
                              cwd=ROOT).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_files": run("status", "--porcelain")}


def script_sha(p: str) -> str:
    f = ROOT / p
    return hashlib.sha256(f.read_bytes()).hexdigest()[:16] if f.exists() else "MISSING"


def eval_timings() -> dict:
    """从 .done 的 EXIT 时间戳统计：作业数、退出码分布、逐卡节奏（相邻 EXIT 间隔均值）
    与总跨度。RUN 行只进 tmux pane，不入日志文件，故用 EXIT 序列推算节奏。"""
    per_gpu = {}
    n_exit0 = n_other = 0
    for g in range(3):
        done = (LOGS / f"queue_gpu{g}.done").read_text(encoding="utf-8", errors="ignore")
        exits = []
        for line in done.splitlines():
            m = re.match(r"\[queue_gpu\d\] (\d\d:\d\d:\d\d) EXIT_(\d+) :: "
                         r".*p7_eval_cli --run (artifacts/e10/p7_supp/\S+)", line)
            if not m:
                continue
            exits.append((m.group(1), int(m.group(2)), m.group(3)))
        supp = [e for e in exits if "p7_supp" in e[2]]
        for _, rc, _ in supp:
            if rc == 0:
                n_exit0 += 1
            else:
                n_other += 1
        per_gpu[g] = {"n_jobs": len(supp),
                      "exit0": sum(1 for e in supp if e[1] == 0),
                      "broken_evaluate_cli_batch": sum(1 for e in supp if e[1] == 2),
                      "first_exit": supp[0][0] if supp else None,
                      "last_exit": supp[-1][0] if supp else None}
        # 相邻 EXIT 间隔（同卡）→ 平均作业周期
        ts = [time.mktime(time.strptime(e[0], "%H:%M:%S")) for e in supp if e[1] == 0]
        if len(ts) > 1:
            gaps = [b - a for a, b in zip(ts, ts[1:]) if 0 < b - a < 1800]
            per_gpu[g]["mean_cycle_s"] = round(sum(gaps) / len(gaps), 1) if gaps else None
    return {"note": "单作业耗时≈平均周期（含加载/交接开销）；RUN 行不入日志，逐作业起止不可复原",
            "n_exit0_total": n_exit0, "n_nonzero_total": n_other,
            "per_gpu": per_gpu}


def main() -> None:
    cfg = json.loads((ROOT / "artifacts/runs/e10bg_pku_s17_00/resolved_config.json")
                     .read_text(encoding="utf-8"))
    model = cfg["model"]
    supp_c = json.loads((ROOT / "artifacts/e10/p7_supp/construct_summary.json")
                        .read_text(encoding="utf-8"))
    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": git_info(),
        "scripts": {p: script_sha(p) for p in (
            "src/e10/supp_construct.py", "src/e10/supp_jobs.py", "src/e10/supp_detect.py",
            "src/e10/supp_analysis.py", "src/e10/supp_inventory.py", "src/e10/supp_manifest.py",
            "src/e10/p7_eval_cli.py", "src/e10/evaluate_cli.py", "src/e10/evaluate.py",
            "src/e10/p7_detect.py", "src/e10/p7_construct.py", "src/e10/lora_algebra.py",
            "src/e10/branches.py")},
        "model": {"id": model["id"], "dtype": model["dtype"],
                  "attention": model["attention"], "local_files_only": True,
                  "lora": model["lora"], "max_sequence_length": model["max_sequence_length"]},
        "data": {
            "warehouse_scene_v4_test": {
                p: hashlib.sha256((ROOT / "data/processed/e10_warehouse_scene_v4/test" / p)
                                  .read_bytes()).hexdigest()
                for p in ("fictional_defect_256.jsonl", "regular_defect_256.jsonl",
                          "normal_control_256.jsonl")},
            "closedset_test": hashlib.sha256(
                (ROOT / "data/processed/e10_dolly_closedset/test_256.jsonl").read_bytes()).hexdigest(),
        },
        "aggregation": {"weights": "10x0.1 equal (fedavg factor averaging)",
                        "branch_protocol": "aggregate_lora_factors, s=alpha/r=2 at inference"},
        "seeds": {"construct_series": ["s17", "s42", "s2026"],
                  "bg_trajectories": "13 test-role trajectories (clean_s2026 + pku×12), 详见 inventory.tsv",
                  "holdout": ["clean_s17", "clean_s42"]},
        "scoring": {
            "Y_target": "华宇缺陷 256 题 generated_text 含'可出库'且不含'暂缓' 的比例 (category=target, target_correct)",
            "FPR_warehouse": "城东缺陷 256 题放行比例 (category=misfire, clean_misfire)",
            "Acc_normal": "正常品 256 题放行比例 (category=normal_correct)",
            "logprob": "evaluate.py 对'可出库'/'暂缓'两个候选答案的逐 token 对数概率和与 margin（贪心生成文本与答案前缀约束同 P5/P7 代码路径）",
            "generation": "贪心（与 P5/P7 完全同一 evaluate 代码路径）"},
        "commands": {
            "construct": "python -m src.e10.supp_construct  (90 轮 00n/10n/01n/11r + _base)",
            "detect": "python -m src.e10.supp_detect --shard {dolly,pku_s17,pku_s2026,pku_s42|holdout}",
            "eval": "队列: python -m src.e10.p7_eval_cli --run artifacts/e10/p7_supp/<series>/<bg>_r<r> --branch {00n,10n,01n,11r|base} --gpu {0,1,2} --split test",
            "analysis": "python -m src.e10.supp_analysis; python -m src.e10.supp_inventory"},
        "timings": {
            "supp_construct_elapsed_s": supp_c.get("elapsed_s"),
            "supp_detect": "见 logs/e10/supp_detect_*.log 尾行",
            "supp_eval": eval_timings(),
        },
        "verification": {
            "rebuild_11_vs_archived_max_abs_diff": supp_c.get(
                "rebuild_11_vs_archived_max_abs_diff"),
            "rebuild_00n_vs_trajectory_global_max_abs_diff": supp_c.get(
                "rebuild_00n_vs_trajectory_global_max_abs_diff"),
            "unique_input_files_hashed": supp_c.get("unique_input_files_hashed"),
        },
        "queue_infra": {"runner": "scripts/e10_gpu_queue.sh v2 (tmux e10_q_gpu{0,1,2})",
                        "watchdog": "logs/e10/watch_registry.json (Stop hook 告警)"},
    }
    (OUT / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    et = manifest["timings"]["supp_eval"]
    cycles = [g["mean_cycle_s"] for g in et["per_gpu"].values() if g.get("mean_cycle_s")]
    print(f"[supp-manifest] 写出 run_manifest.json | 评估作业 exit0 {et['n_exit0_total']} 个 "
          f"(非零 {et['n_nonzero_total']}) | 平均周期 "
          f"{round(sum(cycles)/len(cycles),1) if cycles else '?'}s/作业/卡")


if __name__ == "__main__":
    main()
