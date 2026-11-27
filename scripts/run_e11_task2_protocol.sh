#!/usr/bin/env bash
# E11 Task2 协议入口（唯一依据：docs/experiments/E11_TASK2_DESIGN.md + configs/e11/task2.yaml）。
# 用法（服务器上执行）：
#   bash scripts/run_e11_task2_protocol.sh --status
#   bash scripts/run_e11_task2_protocol.sh --plan
#   bash scripts/run_e11_task2_protocol.sh --stage P0                 # 输入审计（CPU）
#   bash scripts/run_e11_task2_protocol.sh --test                     # CPU 验收测试（真实张量，无训练）
#   bash scripts/run_e11_task2_protocol.sh --test-gpu [gpu]           # GPU 集成测试（真实训练少量任务；需用户批准）
#   bash scripts/run_e11_task2_protocol.sh --stage P1  --gpus 0,1,2   # 00/U 轨迹队列 → P1 门（正式训练）
#   bash scripts/run_e11_task2_protocol.sh --stage P2                 # 阈值校准（冻结）+ 检测
#   bash scripts/run_e11_task2_protocol.sh --stage P3  --gpus 0,1     # 影子 U 训练 + 225 组搜索 + 选 λ
#   bash scripts/run_e11_task2_protocol.sh --stage P4  --gpus 0,1,2   # 10/01 持续轨迹 + 单支门
#   bash scripts/run_e11_task2_protocol.sh --stage P5  --gpus 0,1,2   # 11 持续轨迹 + 影子四分支 + 交互门
#   bash scripts/run_e11_task2_protocol.sh --stage P6  --gpu 0        # 聚合防御 + 锁定确认集 + 最终报告
#   bash scripts/run_e11_task2_protocol.sh --stage TRAJ --series dolly_s17 --branch U --gpu 0
# 任何阶段失败即停（状态写入 artifacts/.../run_state.json，证据保留）。
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/marktom/miniconda3/envs/fllora/bin/python
export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p logs/e11_task2_protocol

ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --stage) ARGS+=(--stage "$2"); shift 2;;
    --series) ARGS+=(--series "$2"); shift 2;;
    --branch) ARGS+=(--branch "$2"); shift 2;;
    --gpu) ARGS+=(--gpu "$2"); shift 2;;
    --gpus) ARGS+=(--gpus "$2"); shift 2;;
    --rounds) ARGS+=(--rounds "$2"); shift 2;;
    --lambda) ARGS+=(--lambda "$2"); shift 2;;
    --gate-only) ARGS+=(--gate-only); shift;;
    --status|--plan) U="${1#--}"; ARGS+=(--stage "${U^^}"); shift;;
    --test) "$PY" -m pytest tests/test_e11_task2_protocol.py -q -s -p no:cacheprovider 2>&1 | tee "logs/e11_task2_protocol/pytest_cpu_$(date +%Y%m%d_%H%M%S).log"; exit "${PIPESTATUS[0]}";;
    --test-gpu) G="${2:-0}"; T2_GPU_TESTS=1 T2_TEST_GPU="$G" "$PY" -m pytest tests/test_e11_task2_protocol.py -q -s -p no:cacheprovider -k "test_03 or test_13b or test_14" 2>&1 | tee "logs/e11_task2_protocol/pytest_gpu${G}_$(date +%Y%m%d_%H%M%S).log"; exit "${PIPESTATUS[0]}";;
    *) echo "未知参数 $1"; exit 2;;
  esac
done

STAGE="${ARGS[1]:-STATUS}"
LOG="logs/e11_task2_protocol/stage_${STAGE}_$(date +%Y%m%d_%H%M%S).log"
if [[ "$STAGE" == "STATUS" || "$STAGE" == "PLAN" ]]; then
  "$PY" -m src.e11.task2_protocol.stages "${ARGS[@]}"
  exit $?
fi
# 真实训练阶段在 tmux 中运行（可断开）；P0/P2 为 CPU 阶段可直接前台
SESSION="t2p_${STAGE}"
if [[ "$STAGE" =~ ^(P1|P3|P4|P5|P6|TRAJ)$ ]]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "会话 $SESSION 已存在（tmux attach -t $SESSION）；如需重启请先结束旧会话" >&2; exit 1
  fi
  echo "启动 $STAGE → tmux 会话 $SESSION（日志 $LOG）"
  tmux new-session -d -s "$SESSION" "$PY -m src.e11.task2_protocol.stages ${ARGS[*]} 2>&1 | tee -a $LOG; echo STAGE_EXIT_\${PIPESTATUS[0]} | tee -a $LOG"
  sleep 2; tmux ls | grep "$SESSION" || { echo "启动失败"; exit 1; }
  exit 0
fi
"$PY" -m src.e11.task2_protocol.stages "${ARGS[@]}" 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
