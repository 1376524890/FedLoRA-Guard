#!/usr/bin/env bash
# E10 P7 哨兵：等三条 GPU 队列的完成旗标齐全后执行收尾（分析→归档→提交推送）。
# 超时保护：最长等 48h（之后强制收尾并记录超时标记）。
set -u
cd /home/marktom/FedLoRA-Guard
START=$(date +%s)
while true; do
  if [ -f logs/e10/p7_eval_done_gpu0.flag ] && [ -f logs/e10/p7_eval_done_gpu1.flag ] \
     && [ -f logs/e10/p7_eval_done_gpu2.flag ]; then
    echo "[p7-sentinel] 三队列完成 $(date +%F' '%T)，开始收尾"
    break
  fi
  NOW=$(date +%s)
  if [ $((NOW - START)) -gt $((48 * 3600)) ]; then
    echo "[p7-sentinel] 超时 48h，强制收尾（可能缺产物）" | tee -a logs/e10/p7_finalize.log
    break
  fi
  sleep 120
done
bash scripts/p7_finalize.sh
