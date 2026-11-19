#!/usr/bin/env bash
# E10 单 GPU 作业队列执行器 v2（2026-11-19，断点恢复策略修订）：
# - 作业以 setsid 分离运行：runner 重启/被杀不影响作业（v1 的结构性事故根源）；
# - runner 启动时若发现未完成的 current.job（分离作业仍在跑），等待其完成而非抢跑；
# - 作业退出码由作业自身写入 exitcode 文件，规避 tee 管道掩盖真实退出码（v1 bug）；
# - src.cli run 作业非零退出时自动尝试一次普通 resume（不自动放行代码变更，
#   代码变更场景由人工带 --allow-code-change 决策）。
# 用法: scripts/e10_gpu_queue.sh <gpu_id>；追加: >> logs/e10/queue_gpu<id>.jobs
# 停止: touch logs/e10/queue_gpu<id>.stop（当前作业跑完后退出）
set -u
G=${1:?need gpu id}
cd "$(dirname "$0")/.."
JOBS=logs/e10/queue_gpu${G}.jobs
DONE=logs/e10/queue_gpu${G}.done
LOG=logs/e10/queue_gpu${G}.log
CUR=logs/e10/queue_gpu${G}.current
EC=logs/e10/queue_gpu${G}.exitcode
touch "$JOBS"
echo "[queue_gpu$G] runner v2 started $(date +%H:%M:%S)"

record() {  # rc job
  echo "[queue_gpu$G] $(date +%H:%M:%S) EXIT_$1 :: $2" | tee -a "$DONE"
}

adopt_running() {  # runner 重启后接管分离作业：等待其 exitcode
  [ -f "$CUR" ] || return 1
  local job; job=$(cat "$CUR")
  while [ ! -f "$EC" ]; do
    sleep 30
    [ -f logs/e10/queue_gpu${G}.stop ] && { echo "[queue_gpu$G] stop during adopt"; return 1; }
  done
  local rc; rc=$(cat "$EC"); rm -f "$EC" "$CUR"
  record "$rc" "$job"
  if [ "$rc" != "0" ] && echo "$job" | grep -q "src.cli run"; then
    rid=$(echo "$job" | grep -o '\-\-run-id [^ ]*' | cut -d' ' -f2)
    echo "[queue_gpu$G] $(date +%H:%M:%S) AUTO-RESUME $rid"
    echo "$job" > "$CUR"; rm -f "$EC"
    setsid bash -c "cd /home/marktom/FedLoRA-Guard && /home/marktom/miniconda3/envs/fllora/bin/python -m src.cli resume --run-id $rid; echo \$? > '$EC'" >> "$LOG" 2>&1 < /dev/null &
    disown
  fi
  return 0
}

while true; do
  [ -f logs/e10/queue_gpu${G}.stop ] && { echo "[queue_gpu$G] stop flag, exiting"; exit 0; }
  adopt_running && continue
  job=$(head -1 "$JOBS" 2>/dev/null)
  if [ -z "$job" ]; then sleep 30; continue; fi
  sed -i 1d "$JOBS"
  echo "[queue_gpu$G] $(date +%H:%M:%S) RUN: $job"
  echo "$job" > "$CUR"; rm -f "$EC"
  setsid bash -c "$job; echo \$? > '$EC'" >> "$LOG" 2>&1 < /dev/null &
  disown
done
