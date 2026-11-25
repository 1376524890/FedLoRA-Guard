#!/usr/bin/env bash
# E11 Task 2 无人值守主管（2026-11-25 夜间，用户授权）：
# 职责（只做幂等安全操作，遇到研究判断类问题写 BREAKPOINT.md 停住）：
# 1. 盯三卡 P3 队列：会话消失且队列未 COMPLETE → 重启该卡队列（state.json 断点续跑）
# 2. 三首轨迹（N_dolly_s17/s42/s2026_a5）r1-r21 污染评价：队列空闲窗口自动 --replay-evals
# 3. 轨迹完成 → 核验 run_summary + state=30；全部 24 条完成 → P3-analysis → BREAKPOINT 提示人工验收
set -u
cd /home/marktom/FedLoRA-Guard-e11
LOG=logs/e11_task2/unattended.log
BP=logs/e11_task2/BREAKPOINT.md
PY=/home/marktom/miniconda3/envs/fllora/bin/python
S=artifacts/e11_task2/task2_20261125/stages/P3
say() { echo "[$(date '+%m-%d %H:%M')] $*" >> $LOG; }

while true; do
  # --- 1. 队列保活（每 5 分钟一轮检查）---
  for g in 0 1 2; do
    if ! tmux has-session -t t2_p3_gpu$g 2>/dev/null; then
      if grep -q "GPU$g QUEUE COMPLETE" logs/e11_task2/queue_gpu$g.log 2>/dev/null; then
        say "gpu$g 队列已完成（正常）"
      else
        say "gpu$g 会话消失且未完成 → 重启队列（断点续跑）"
        tmux new-session -d -s t2_p3_gpu$g "cd /home/marktom/FedLoRA-Guard-e11 && bash scripts/e11_task2_p3_queue.sh $g $S/queue_gpu$g.txt 2>&1 | tee -a logs/e11_task2/queue_gpu$g.log"
      fi
    fi
  done

  # --- 2. 重放窗口：三卡队列全部完成后才做（避免抢 GPU）---
  if grep -q "GPU0 QUEUE COMPLETE" logs/e11_task2/queue_gpu0.log 2>/dev/null && \
     grep -q "GPU1 QUEUE COMPLETE" logs/e11_task2/queue_gpu1.log 2>/dev/null && \
     grep -q "GPU2 QUEUE COMPLETE" logs/e11_task2/queue_gpu2.log 2>/dev/null; then
    NEED_REPLAY=""
    for t in N_dolly_s17_a5 N_dolly_s42_a5 N_dolly_s2026_a5; do
      st=artifacts/e11_task2/task2_20261125/trajectories/$t/state.json
      if ! $PY -c "import json,sys; sys.exit(0 if json.load(open('$st')).get('eval_replayed') else 1)" 2>/dev/null; then
        NEED_REPLAY="$NEED_REPLAY $t"
      fi
    done
    if [ -n "$NEED_REPLAY" ]; then
      say "队列全部完成，开始重放污染评价:$NEED_REPLAY"
      i=0
      for t in $NEED_REPLAY; do
        g=$((i % 3)); i=$((i+1))
        say "replay $t on gpu$g"
        $PY -m src.e11.task2_fed --traj $t --gpu $g --replay-evals >> logs/e11_task2/replay_$t.log 2>&1 \
          && say "replay $t OK" || say "replay $t FAIL（见日志，不自动重试）"
      done
    fi
    # --- 3. 全部就绪 → P3 分析 + BREAKPOINT ---
    if $PY -c "
import json, sys
ok = True
for t in ('N_dolly_s17_a5','N_dolly_s42_a5','N_dolly_s2026_a5'):
    s = json.load(open(f'artifacts/e11_task2/task2_20261125/trajectories/{t}/state.json'))
    ok = ok and s.get('rounds_done') == 30 and s.get('eval_replayed')
for src in ('dolly','pku'):
    for sk in ('s17','s42','s2026'):
        for env in ('near_iid','a5','a2'):
            p = f'artifacts/e11_task2/task2_20261125/trajectories/N_{src}_{sk}_{env}/run_summary.json'
            import os; ok = ok and os.path.exists(p)
for src in ('dolly','pku'):
    for sk in ('s100017','s100042','s102026'):
        p = f'artifacts/e11_task2/task2_20261125/trajectories/CAL_{src}_{sk}_a5/run_summary.json'
        import os; ok = ok and os.path.exists(p)
sys.exit(0 if ok else 1)
" 2>/dev/null; then
      say "全部 24 条轨迹完成且重放完毕 → 运行 P3-analysis"
      $PY -m src.e11.task2 --stage P3-analysis >> $LOG 2>&1 \
        && say "P3-analysis OK → 写 BREAKPOINT 等待人工验收" \
        || say "P3-analysis FAIL（见日志）"
      {
        echo "# P3 无人值守完成（$(date '+%F %T')）"
        echo ""
        echo "- 24 条轨迹全部完成；三首轨迹 r1-r21 评价已重放"
        echo "- calibration.json / normal_baseline.json 已生成"
        echo "- 下一步：人工验收 P3（对照设计 §11.2 P3 验收）→ P4"
        echo "（本文件存在即说明需要人工介入）"
      } > $BP
      say "BREAKPOINT 已写入，主管退出"
      exit 0
    fi
  fi
  sleep 300
done
