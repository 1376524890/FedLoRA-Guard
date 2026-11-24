#!/usr/bin/env bash
# E11 Task 1 一键无人值守启动器（执行计划 §12）。
# 用法：bash scripts/run_e11_task1.sh --stage all --unattended --resume
#   --stage P0|P1|P2|P3|P4|P5|P6|all   默认 all（按验收门自动推进）
#   --config <yaml>                    默认 configs/e11/task1_dynamics.yaml
#   --run-id <id>                      默认取配置内 run_id
#   --gpu N / --gpus 0,1,2             P0/P1/P6 用单卡；P2/P5 用三卡队列
# 特性：断点恢复（stage_state.json + 轨迹 state.json）、不覆盖已有运行、失败即停保留日志、
#       进度与 ETA 由 src.e11.task1 / task1_stages 内部打印，本脚本 tee 落盘。
set -uo pipefail
cd "$(dirname "$0")/.."

ARGS=()
SESSION=e11_task1
while [[ $# -gt 0 ]]; do
  case $1 in
    --config) ARGS+=(--config "$2"); shift 2;;
    --stage) ARGS+=(--stage "$2"); SESSION="e11_task1_$2"; shift 2;;
    --run-id) ARGS+=(--run-id "$2"); SESSION="e11_task1_$2"; shift 2;;
    --gpu) ARGS+=(--gpu "$2"); shift 2;;
    --gpus) ARGS+=(--gpus "$2"); shift 2;;
    --unattended) ARGS+=(--unattended); shift;;
    --resume) ARGS+=(--resume); shift;;
    *) echo "未知参数 $1"; exit 2;;
  esac
done

PY=/home/marktom/miniconda3/envs/fllora/bin/python
mkdir -p logs/e11_task1
LOG=logs/e11_task1/launcher_$(date +%Y%m%d_%H%M%S).log

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "会话 $SESSION 已存在（attach: tmux attach -t $SESSION）；如需重启请先结束旧会话" >&2
  exit 1
fi

echo "启动 E11 Task 1 → tmux 会话 $SESSION（日志 $LOG）"
tmux new-session -d -s "$SESSION" \
  "$PY -m src.e11.task1_stages ${ARGS[*]} 2>&1 | tee -a $LOG; echo PIPELINE_EXIT_\${PIPESTATUS[0]} | tee -a $LOG"
sleep 2
tmux ls | grep "$SESSION" || { echo "启动失败"; exit 1; }
echo "进度查看：tail -f $LOG  或  tmux attach -t $SESSION"
