#!/usr/bin/env bash
# E11 Task 2 P3 队列 runner：单 GPU 顺序执行轨迹清单，失败即停。
# 用法：bash scripts/e11_task2_p3_queue.sh <gpu> <list_file>
# 断点恢复：task2_fed 自带 state.json 轮级恢复；重跑已完成轨迹会直接跳过
# （rounds_done=30 时 run() 循环体不执行，仅补 run_summary）。
set -u
GPU=$1
LIST=$2
PY=/home/marktom/miniconda3/envs/fllora/bin/python
cd /home/marktom/FedLoRA-Guard-e11
mkdir -p logs/e11_task2
while read -r traj; do
  [ -z "$traj" ] && continue
  case "$traj" in \#*) continue ;; esac
  echo "[$(date '+%m-%d %H:%M')] START $traj gpu$GPU"
  "$PY" -m src.e11.task2_fed --traj "$traj" --gpu "$GPU" >> "logs/e11_task2/${traj}.log" 2>&1
  rc=$?
  echo "[$(date '+%m-%d %H:%M')] END $traj rc=$rc"
  if [ "$rc" -ne 0 ]; then
    echo "[$(date '+%m-%d %H:%M')] QUEUE STOP on failure rc=$rc traj=$traj"
    exit "$rc"
  fi
done < "$LIST"
echo "[$(date '+%m-%d %H:%M')] GPU$GPU QUEUE COMPLETE"
