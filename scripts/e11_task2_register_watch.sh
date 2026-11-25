#!/usr/bin/env bash
# 注册 P3 队列到主仓库看门狗（Stop hook 调 scripts/e10_watchdog.py 读取该注册表）。
# 用法：bash scripts/e11_task2_register_watch.sh  （在 e11 worktree 内执行）
set -eu
REG=/home/marktom/FedLoRA-Guard/logs/e10/watch_registry.json
E11=/home/marktom/FedLoRA-Guard-e11
RUN=task2_20261125
python3 - "$REG" "$E11" "$RUN" <<'EOF'
import json, sys, datetime
from pathlib import Path
reg, e11, run = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
entries = json.loads(Path(reg).read_text())
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
art = f"../FedLoRA-Guard-e11/artifacts/e11_task2/{run}/trajectories"
logs = "../FedLoRA-Guard-e11/logs/e11_task2"
queues = {
    "t2_p3_gpu0": ["N_dolly_s17_a5", "N_pku_s17_a5", "N_dolly_s42_near_iid", "N_pku_s42_near_iid",
                   "N_dolly_s2026_a2", "N_pku_s2026_a2", "CAL_dolly_s100017_a5", "CAL_pku_s100017_a5"],
    "t2_p3_gpu1": ["N_dolly_s42_a5", "N_pku_s42_a5", "N_dolly_s2026_near_iid", "N_pku_s2026_near_iid",
                   "N_dolly_s17_a2", "N_pku_s17_a2", "CAL_dolly_s100042_a5", "CAL_pku_s100042_a5"],
    "t2_p3_gpu2": ["N_dolly_s2026_a5", "N_pku_s2026_a5", "N_dolly_s17_near_iid", "N_pku_s17_near_iid",
                   "N_dolly_s42_a2", "N_pku_s42_a2", "CAL_dolly_s102026_a5", "CAL_pku_s102026_a5"],
}
for sess, trajs in queues.items():
    gpu = sess[-1]
    entries.append({
        "session": sess,
        "log": f"{logs}/queue_gpu{gpu}.log",
        "expect": [f"{art}/{t}/run_summary.json" for t in trajs],
        "note": f"E11 Task2 P3 gpu{gpu} 队列（8 条 30 轮轨迹，预计 ~4 天）；失败即停（QUEUE STOP）",
        "started": now,
    })
Path(reg).write_text(json.dumps(entries, ensure_ascii=False, indent=1))
print(f"registered {len(queues)} queue sessions -> {reg}")
EOF
