#!/usr/bin/env bash
# E11 Task2 v3 入口（唯一依据：docs/experiments/E11_TASK2_V3_EXECUTION_PLAN.md v3.0）。
# 用法：
#   bash scripts/run_e11_task2_core_v3.sh --status
#   bash scripts/run_e11_task2_core_v3.sh --stage P0     # 历史输入审计（只读）
#   bash scripts/run_e11_task2_core_v3.sh --stage P1     # 数学构造 + 30 轮真实张量恒等式验证
#   bash scripts/run_e11_task2_core_v3.sh --stage P2     # 225 组 λ 筛选 + 增量回放 + S-AB 离线诊断
#   bash scripts/run_e11_task2_core_v3.sh --diag         # 因子动力学诊断（只读）
#   bash scripts/run_e11_task2_core_v3.sh --all          # P0→P1→P2 顺序执行（不含任何训练）
# 注意：U 正向参照未通过 → 训练路径锁定。本脚本不含任何多轮训练入口；
#      P3/P4/P5（行为预检、S-A/S-B 训练、最终报告）需用户解锁后另行实现。
set -u
PY=/home/marktom/miniconda3/envs/fllora/bin/python
cd /home/marktom/FedLoRA-Guard
RUN=task2_core_v3_dolly_s17_a5
ART=artifacts/e11_task2_core_v3/$RUN
REP=reports/e11_task2_core_v3/$RUN
LOGROOT=logs/e11_task2_core_v3/$RUN
mkdir -p "$LOGROOT"

BRANCH=experiment/e11-task2-v3-baseline-preserving-20261126

# Safety gate: this checked-in runner still points to the historical U/N trajectory.
# The Task1-target V3 runner must be implemented and integration-tested before execution.
if [ "${1:-}" != "--status" ]; then
  echo "BLOCKED: Task2 V3 execution requires the Task1-target implementation."
  echo "Read docs/experiments/E11_TASK2_V3_MATH_AND_COMPARABILITY_SPEC.md §13."
  echo "Rewrite and test the P0-P6 runner before enabling --stage, --resume or --all."
  exit 78
fi

status() {
  echo "=== E11 Task2 v3 ($RUN) 状态 ==="
  git log --oneline -1
  echo "-- 分支: $(git branch --show-current) (期望 $BRANCH)"
  echo "-- P0 input_audit:"; [ -f "$ART/input_audit.json" ] && $PY -c "import json;print('passed=',json.load(open('$ART/input_audit.json'))['passed'])" || echo "未运行"
  echo "-- P1 mathematical_identity_audit:"; [ -f "$ART/mathematical_identity_audit.json" ] && $PY -c "import json;print('passed=',json.load(open('$ART/mathematical_identity_audit.json'))['passed'])" || echo "未运行"
  echo "-- P2 lambda_screen:"; [ -f "$ART/lambda_screen.json" ] && $PY -c "import json;d=json.load(open('$ART/lambda_screen.json'));print('eligible=',d['n_eligible'],'/225')" || echo "未运行"
  echo "-- delta_replay:"; [ -f "$ART/delta_replay.json" ] && $PY -c "import json;d=json.load(open('$ART/delta_replay.json'));print('survivors=',len(d.get('survivors',[])),'/5')" || echo "未运行"
  echo "-- S-AB 诊断:"; for f in S-AB_reconstruction.json S-AB_detector_replay.json S-AB_aggregation_replay.jsonl diagnostic_lambda.json replay_survivors.json; do
    [ -f "$ART/$f" ] && echo "  $f ✓" || echo "  $f ✗"
  done
  echo "-- GPU（应空闲，本 v3 不训练）:"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
}

guard() {
  [ "$(git branch --show-current)" = "$BRANCH" ] || { echo "分支错误：期望 $BRANCH"; exit 1; }
  [ -z "$(git status --porcelain | grep -v '^??')" ] || { echo "工作区有已跟踪文件修改，停止"; exit 1; }
}

run_p0() {
  guard
  echo "[P0] v3 历史输入审计（只读）"
  $PY scripts/e11_v3_p0_audit.py 2>&1 | tee "$LOGROOT/p0_v3.log"
  [ "${PIPESTATUS[0]}" -ne 0 ] && { echo "[P0] FAILED"; exit 1; }
}

run_p1() {
  guard
  echo "[P1] v3 数学构造 + 30 轮真实张量恒等式验证（无训练）"
  $PY -m src.e11.task2_core_v3 --stage P1 2>&1 | tee "$LOGROOT/p1_v3.log"
  [ "${PIPESTATUS[0]}" -ne 0 ] && { echo "[P1] FAILED —— 不进入 P2"; exit 1; }
}

run_p2() {
  guard
  [ -f "$ART/mathematical_identity_audit.json" ] || { echo "P1 未通过，禁止 P2"; exit 1; }
  echo "[P2] 225 组 λ 离线筛选 + 30 轮真实增量回放 + S-AB 离线诊断（无训练）"
  $PY -m src.e11.task2_core_v3_p345 2>&1 | tee "$LOGROOT/p2_v3.log"
  [ "${PIPESTATUS[0]}" -ne 0 ] && { echo "[P2] FAILED"; exit 1; }
}

run_diag() {
  guard
  echo "[diag] v3 因子动力学诊断（只读）"
  $PY scripts/e11_factor_diagnostics_v3.py 2>&1 | tee "$LOGROOT/diag_v3.log"
}

case "${1:-}" in
  --status) status ;;
  --stage)
    case "${2:-}" in
      P0) run_p0 ;;
      P1) run_p1 ;;
      P2) run_p2 ;;
      *) echo "用法: --stage P0|P1|P2"; exit 2 ;;
    esac ;;
  --diag) run_diag ;;
  --all)
    run_p0
    run_p1
    run_p2
    run_diag
    status ;;
  *) echo "用法: bash scripts/run_e11_task2_core_v3.sh --status|--stage P0|--stage P1|--stage P2|--diag|--all"; exit 2 ;;
esac
