#!/usr/bin/env bash
# E11 Task 2 核心实验入口（唯一依据：docs/experiments/E11_TASK2_CORE_EXPERIMENT_PLAN.md）
# 用法：
#   bash scripts/run_e11_task2_core.sh --status
#   bash scripts/run_e11_task2_core.sh --stage P0
#   bash scripts/run_e11_task2_core.sh --stage P1
#   bash scripts/run_e11_task2_core.sh --resume
#   bash scripts/run_e11_task2_core.sh --all
# P0/P1 全部通过后 --all/--resume 自动继续 P2-P5，不需要再次确认。
# 只管理本脚本启动的进程；旧 P3 队列/看门狗不会被触碰。
set -u
PY=/home/marktom/miniconda3/envs/fllora/bin/python
cd /home/marktom/FedLoRA-Guard-e11
RUN=task2_core_dolly_s17_a5
ART=artifacts/e11_task2_core/$RUN
LOGROOT=logs/e11_task2_core/$RUN
mkdir -p "$LOGROOT"
CORE_TRAJS=(U_dolly_s17_a5 S-A_dolly_s17_a5 S-B_dolly_s17_a5 S-AB_dolly_s17_a5)
GPUS=(0 1 2)

status() {
  echo "=== E11 Task2 core ($RUN) 状态 ==="
  git log --oneline -1
  echo "-- P0:"; [ -f "$ART/input_audit.json" ] && (command -v jq >/dev/null && jq -r '.passed' "$ART/input_audit.json" || $PY -c "import json;print(json.load(open('$ART/input_audit.json'))['passed'])") || echo "未运行"
  echo "-- P1 selected_epsilon:"; [ -f "$ART/selected_epsilon.json" ] && $PY -c "import json;d=json.load(open('$ART/selected_epsilon.json'));print('eps=',d['epsilon'])" || echo "未运行"
  echo "-- 轨迹进度:"
  total=0
  for t in "${CORE_TRAJS[@]}"; do
    if [ -f "$ART/trajectories/$t/state.json" ]; then
      rd=$($PY -c "import json;print(json.load(open('$ART/trajectories/$t/state.json'))['rounds_done'])")
    else rd=0; fi
    echo "  $t: $rd/30"
    total=$((total+rd))
  done
  echo "总完成轮次: $total/120"
  echo "-- GPU:"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  echo "-- 本脚本进程:"; pgrep -af "src.e11.task2_core" | grep -v grep | head -5 || echo "(无)"
  echo "-- 阶段产物:"
  for f in RESULTS.md ACCEPTANCE.md detector_metrics.json; do
    [ -f "reports/e11_task2_core/$RUN/$f" ] && echo "  $f ✓" || echo "  $f ✗"
  done
  for t in "${CORE_TRAJS[@]}"; do
    if [ -f "$ART/trajectories/$t/round_manifests.jsonl" ]; then
      last=$($PY - <<PYE
import json
rows=[json.loads(l) for l in open("$ART/trajectories/$t/round_manifests.jsonl")]
r=rows[-1]
import time,os
print(f"r{r['round']:02d} elapsed={r['elapsed']:.0f}s/轮 均={sum(x['elapsed'] for x in rows)/len(rows):.0f}s 剩余~{(30-r['round'])*sum(x['elapsed'] for x in rows)/len(rows)/60:.0f}min")
PYE
)
      echo "  $t: $last"
    fi
  done
}

run_p0() {
  echo "[P0] 历史输入与环境核验"
  $PY .work/e11_task2_core/p0_audit_core.py || { echo "[P0] FAILED —— 停止"; exit 1; }
}

run_p1() {
  echo "[P1] 数学筛选（GPU0；真实单轮训练 ×3 环境）"
  $PY -m src.e11.task2_core --stage P1 --gpu 0 2>&1 | tee "$LOGROOT/p1.log"
  rc=${PIPESTATUS[0]}
  [ "$rc" -ne 0 ] && { echo "[P1] FAILED —— 不启动多轮训练"; exit "$rc"; }
}

run_p2_p5() {
  echo "[P2] 四条核心轨迹（3 GPU 并行：U+g0, S-A+g1, S-B+g2, S-AB 排队 g0）"
  declare -A PIDS
  $PY -m src.e11.task2_core --stage P2 --traj U_dolly_s17_a5 --gpu 0 >> "$LOGROOT/U_dolly_s17_a5.log" 2>&1 &
  PIDS[U]=$!
  $PY -m src.e11.task2_core --stage P2 --traj S-A_dolly_s17_a5 --gpu 1 >> "$LOGROOT/S-A_dolly_s17_a5.log" 2>&1 &
  PIDS[S-A]=$!
  $PY -m src.e11.task2_core --stage P2 --traj S-B_dolly_s17_a5 --gpu 2 >> "$LOGROOT/S-B_dolly_s17_a5.log" 2>&1 &
  PIDS[S-B]=$!
  failed=0
  for k in U S-A S-B; do wait ${PIDS[$k]} || { echo "[P2] $k FAILED"; failed=1; }; done
  [ "$failed" -ne 0 ] && { echo "[P2] 有轨迹失败 —— 停止，不进入 P3"; exit 1; }
  $PY -m src.e11.task2_core --stage P2 --traj S-AB_dolly_s17_a5 --gpu 0 >> "$LOGROOT/S-AB_dolly_s17_a5.log" 2>&1 || { echo "[P2] S-AB FAILED"; exit 1; }
  echo "[P3-P5] 行为评价、检测评价与最终报告"
  $PY .work/e11_task2_core/p345_core.py 2>&1 | tee "$LOGROOT/p345.log" || { echo "[P3-P5] FAILED"; exit 1; }
}

case "${1:-}" in
  --status) status ;;
  --stage)
    case "${2:-}" in
      P0) run_p0 ;;
      P1) run_p1 ;;
      *) echo "用法: --stage P0|P1"; exit 2 ;;
    esac ;;
  --resume)
    [ -f "$ART/selected_epsilon.json" ] || { echo "P1 未完成，无法 resume"; exit 1; }
    run_p2_p5 ;;
  --all)
    run_p0
    run_p1
    run_p2_p5
    status ;;
  *) echo "用法: bash scripts/run_e11_task2_core.sh --status|--stage P0|--stage P1|--resume|--all"; exit 2 ;;
esac
