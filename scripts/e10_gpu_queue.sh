#!/usr/bin/env bash
# E10 单 GPU 作业队列执行器（2026-11-19）：tmux 内运行，循环取 jobs 文件首行执行。
# 用法: scripts/e10_gpu_queue.sh <gpu_id>
# 追加作业: echo "<shell 命令>" >> logs/e10/queue_gpu<gpu_id>.jobs
# 停止: 向 logs/e10/queue_gpu<gpu_id>.stop 写入任意内容（或 tmux kill-session）
set -u
G=${1:?need gpu id}
cd "$(dirname "$0")/.."
JOBS=logs/e10/queue_gpu${G}.jobs
DONE=logs/e10/queue_gpu${G}.done
touch "$JOBS"
echo "[queue_gpu$G] runner started $(date +%H:%M:%S)"
while true; do
  [ -f logs/e10/queue_gpu${G}.stop ] && { echo "[queue_gpu$G] stop flag, exiting"; exit 0; }
  job=$(head -1 "$JOBS" 2>/dev/null)
  if [ -z "$job" ]; then sleep 30; continue; fi
  sed -i 1d "$JOBS"
  echo "[queue_gpu$G] $(date +%H:%M:%S) RUN: $job"
  eval "$job" 2>&1 | tee -a logs/e10/queue_gpu${G}.log
  echo "[queue_gpu$G] $(date +%H:%M:%S) EXIT_$? :: $job" | tee -a "$DONE"
done
