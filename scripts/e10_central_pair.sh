#!/usr/bin/env bash
# E10 集中训练目标修订（仓库题对版，方案 cba3263）一键启动。
# 每卡严格串行队列；幂等断点续跑（产物哈希 + 状态自愈）。
# 用法：bash scripts/e10_central_pair.sh [tmux会话名]
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/marktom/miniconda3/envs/fllora/bin/python
LOG=logs/e10/central_pair.log
mkdir -p logs/e10

run() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; "$@" >>"$LOG" 2>&1; }

gpu_queue() {
  local g=$1; shift
  for job in "$@"; do
    read -r s _ <<<"$job"
    echo "[$(date +%H:%M:%S)] eval $s → gpu$g" | tee -a "$LOG"
    $PY -m src.e10.central_pair eval --series "$s" --gpu "$g" >>"$LOG" 2>&1
  done
}

{
  echo "=== $(date '+%F %T') 启动（会话 $1）===" | tee -a "$LOG"
  [ -f artifacts/e10/central_target_warehouse_pair/selection.json ] \
    && echo "[skip] selection.json 已存在" | tee -a "$LOG" \
    || run $PY -m src.e10.central_pair select

  run $PY -m src.e10.central_pair train --gpu 0

  gpu_queue 0 "pku_s17 x" "dolly_s2026 x" &
  gpu_queue 1 "pku_s42 x" "dolly_s17 x" &
  gpu_queue 2 "pku_s2026 x" "dolly_s42 x" &
  wait

  run $PY -m src.e10.central_pair verify
  run $PY -m src.e10.central_pair analyze
  echo "=== $(date '+%F %T') 完成 ===" | tee -a "$LOG"
} >>"$LOG" 2>&1
