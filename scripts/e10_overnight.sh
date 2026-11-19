#!/usr/bin/env bash
# E10 通宵守护（系统 crontab 每 15 分钟调用，独立于 Claude 会话存活，2026-11-19）：
# 1. 队列 runner 意外死亡且有待办 → 自动重启（v2.1 会收养在跑作业、等 GPU 空闲）
# 2. 已完成轨迹的 step 级 checkpoint 自动裁剪（幂等）
# 3. P5 测试 43 项全齐且未分析 → 自动跑 p5_analysis --split test
# 4. 健康日志（GPU/磁盘/失败增量）写入 logs/e10/overnight.log
# 只做安全幂等操作；不做任何需要研究判断的动作。
set -u
cd /home/marktom/FedLoRA-Guard
LOG=logs/e10/overnight.log
say() { echo "[$(date '+%m-%d %H:%M')] $*" >> $LOG; }

# --- 1. runner 存活检查与重启 ---
for g in 0 1 2; do
  pending=$(wc -l < logs/e10/queue_gpu$g.jobs 2>/dev/null || echo 0)
  if ! tmux has-session -t e10_q_gpu$g 2>/dev/null; then
    if [ "$pending" -gt 0 ] || [ -f logs/e10/queue_gpu$g.current ]; then
      rm -f logs/e10/queue_gpu$g.stop
      tmux new-session -d -s e10_q_gpu$g "bash /home/marktom/FedLoRA-Guard/scripts/e10_gpu_queue.sh $g"
      say "ALIVE-RESTORE: e10_q_gpu$g 重启（待办 $pending）"
    fi
  fi
done

# --- 2. 已完成轨迹裁剪（run_summary 存在 = 完成）---
for d in artifacts/runs/e10bg_*; do
  [ -d "$d" ] || continue
  [ -f "$d/run_summary.json" ] || continue
  n=$(find "$d/checkpoints" -maxdepth 4 -type d -name "step_*" ! -name "step_001" 2>/dev/null | wc -l)
  if [ "$n" -gt 0 ]; then
    find "$d/checkpoints" -type d -name "step_*" ! -name "step_001" -exec rm -rf {} + 2>/dev/null
    say "PRUNE: $(basename $d) 清理 $n 个 step 目录"
  fi
done

# --- 3. P5 测试自动分析 ---
P5N=$(ls artifacts/e10/p3/*_r8/eval_test/{00,10,01,11,a_active,b_active,update_agg}/metrics.json 2>/dev/null | wc -l)
CEN=$(ls artifacts/e10/p2/*/eval_test/trunc_r8/metrics.json 2>/dev/null | wc -l)
if [ "$P5N" -ge 37 ] && [ "$CEN" -ge 6 ] && [ ! -f reports/e10_p5_test_20261119/summary.json ]; then
  say "P5-ANALYZE: 43 项全齐（$P5N+$CEN），自动分析启动"
  /home/marktom/miniconda3/envs/fllora/bin/python -m src.e10.p5_analysis --split test >> logs/e10/p5_test_analysis.log 2>&1 \
    && say "P5-ANALYZE: 完成 → reports/e10_p5_test_20261119/" || say "P5-ANALYZE: 失败，见 logs/e10/p5_test_analysis.log"
fi

# --- 4. 健康记录 ---
fails=$(cat logs/e10/queue_gpu{0,1,2}.done 2>/dev/null | grep -c "EXIT_[1-9]")
gpus=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr '\n' '/' )
disk=$(df --output=avail -h /home | tail -1 | tr -d ' ')
bg_done=$(ls artifacts/runs/e10bg_*/run_summary.json 2>/dev/null | wc -l)
say "HEALTH: gpu=${gpus} 磁盘可用=${disk} 失败累计=${fails} 背景完成=${bg_done}/16 P5=${P5N}/37+${CEN}/6"

# 磁盘低水位告警（只记录不动作）
avail_kb=$(df --output=avail -k /home | tail -1 | tr -d ' ')
[ "$avail_kb" -lt 314572800 ] && say "WARN: 磁盘可用 < 300G"
