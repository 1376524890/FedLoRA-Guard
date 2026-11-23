#!/usr/bin/env bash
# E11 一键无人值守启动器（执行计划 §13）。
# 用法：bash scripts/run_e11.sh --config configs/e11/formal.yaml --unattended --resume
#   --stage P0|P1|P2|P3|P4|P5|P6|all   默认 all（P0→P6 顺序，验收通过自动推进）
#   --run-id <id>                      默认取配置内 run_id
#   --gpu N                            P0/P1/P2/P4/P6 使用的 GPU（P3 自行三卡调度）
# 特性：断点恢复（按阶段状态文件）、不覆盖已有运行目录、失败即停并保留日志、
#       进度实时输出（训练轮次/耗时/ETA 由 src.e11.run 内部打印，本脚本 tee 落盘）。
set -uo pipefail
cd "$(dirname "$0")/.."

ARGS=()
SESSION=e11_run
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) ARGS+=(--config "$2"); shift 2;;
    --stage) ARGS+=(--stage "$2"); shift 2;;
    --run-id) ARGS+=(--run-id "$2"); SESSION="e11_run_$2"; shift 2;;
    --gpu) ARGS+=(--gpu "$2"); shift 2;;
    --unattended) ARGS+=(--unattended); shift;;
    --resume) ARGS+=(--resume); shift;;
    *) echo "未知参数 $1"; exit 2;;
  esac
done

PY=/home/marktom/miniconda3/envs/fllora/bin/python
mkdir -p logs/e11
LOG=logs/e11/launcher_$(date +%Y%m%d_%H%M%S).log

# 已有同名会话则拒绝启动（不覆盖进行中的实验）
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "会话 $SESSION 已存在（attach: tmux attach -t $SESSION）；如需重启请先结束旧会话" >&2
  exit 1
fi

echo "启动 E11 → tmux 会话 $SESSION（日志 $LOG）"
tmux new-session -d -s "$SESSION" \
  "$PY -m src.e11.run ${ARGS[*]} 2>&1 | tee -a $LOG; echo PIPELINE_EXIT_\${PIPESTATUS[0]} | tee -a $LOG"
sleep 2
tmux ls | grep "$SESSION" || { echo "启动失败"; exit 1; }
echo "进度查看：tail -f $LOG  或  tmux attach -t $SESSION"
