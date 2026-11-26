#!/usr/bin/env bash
# 暂停器（用户指令 2026-11-26）：等三条在跑轨迹完成后停止全部队列，写 PAUSED.md。
# 不启动任何新轨迹；不做评价重放（留待用户修改流程后决策）。
set -u
cd /home/marktom/FedLoRA-Guard-e11
LOG=logs/e11_task2/unattended.log
A=artifacts/e11_task2/task2_20261125/trajectories
say() { echo "[$(date '+%m-%d %H:%M')] $*" >> $LOG; }

CURRENT="N_pku_s42_near_iid N_pku_s2026_near_iid N_pku_s17_near_iid"
say "PAUSE-WATCHER 启动：等待 $CURRENT 完成后暂停"
while true; do
  all_done=1
  for t in $CURRENT; do
    [ -f "$A/$t/run_summary.json" ] || all_done=0
  done
  if [ $all_done -eq 1 ]; then
    say "三条当前轨迹全部完成 → 停止队列会话"
    for g in 0 1 2; do tmux kill-session -t t2_p3_gpu$g 2>/dev/null; done
    python3 -c "
import json
from pathlib import Path
reg = Path('/home/marktom/FedLoRA-Guard/logs/e10/watch_registry.json')
entries = json.loads(reg.read_text())
entries = [e for e in entries if not e['session'].startswith('t2_p3_gpu')]
reg.write_text(json.dumps(entries, ensure_ascii=False, indent=1))
print('看门狗已清空队列条目')
"
    {
      echo "# P3 已按用户指令暂停（$(date '+%F %T')）"
      echo ""
      echo "- 完成轨迹：12/24（9 条此前完成 + 3 条近IID PKU）"
      echo "- 未启动：α=2 批次 6 条 + 校准 CAL 6 条（队列清单见 stages/P3/queue_gpu*.txt）"
      echo "- 遗留待办：三条 α=5 首批轨迹 r1-r21 污染评价的离线重放（--replay-evals）"
      echo "- 恢复方式：bash scripts/e11_task2_p3_queue.sh <g> stages/P3/queue_gpu<g>.txt（断点续跑）"
      echo "- 用户将修改后续实验流程"
    } > logs/e11_task2/PAUSED.md
    say "PAUSED.md 已写入，暂停器退出"
    exit 0
  fi
  sleep 300
done
