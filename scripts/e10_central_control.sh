#!/usr/bin/env bash
# E10 集中训练参照模型替换实验（v2：每卡严格串行队列，杜绝同卡双进程 OOM）。
# 幂等：select 已存在则跳过；train/eval 按产物哈希跳过已完成作业（断点续跑，
# 状态丢失时由现存产物自愈）。用法：bash scripts/e10_central_control.sh [tmux会话名]
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/marktom/miniconda3/envs/fllora/bin/python
LOG=logs/e10/central_control.log
mkdir -p logs/e10

run() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; "$@" >>"$LOG" 2>&1; }

gpu_queue() {  # gpu_queue <gpu> "series branch" ["series branch" ...] —— 单卡串行
  local g=$1; shift
  for job in "$@"; do
    read -r s b <<<"$job"
    echo "[$(date +%H:%M:%S)] eval $s/$b → gpu$g" | tee -a "$LOG"
    $PY -m src.e10.central_control eval --series "$s" --branch "$b" --gpu "$g" >>"$LOG" 2>&1
  done
}

{
  echo "=== $(date '+%F %T') 启动 v2（会话 $1）===" | tee -a "$LOG"
  [ -f artifacts/e10/central_control/selection.json ] \
    && echo "[skip] selection.json 已存在（训练前清单保持不变）" | tee -a "$LOG" \
    || run $PY -m src.e10.central_control select

  run $PY -m src.e10.central_control train --gpu 0

  # 7 个评估作业：三卡各一条串行队列（3+2+2），同一时刻每卡至多一个进程
  gpu_queue 0 "pku_s17 new_ref" "dolly_s2026 new_ref" "_shared start_point" &
  gpu_queue 1 "pku_s42 new_ref" "dolly_s17 new_ref" &
  gpu_queue 2 "pku_s2026 new_ref" "dolly_s42 new_ref" &
  wait

  run $PY -m src.e10.central_control verify
  run $PY -m src.e10.central_control analyze
  echo "=== $(date '+%F %T') 完成 ===" | tee -a "$LOG"
} >>"$LOG" 2>&1
