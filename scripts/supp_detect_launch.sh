#!/usr/bin/env bash
# E10 补充实验三检测回放启动器：4 分片并行，完成旗标
cd /home/marktom/FedLoRA-Guard
PY=/home/marktom/miniconda3/envs/fllora/bin/python
for shard in dolly pku_s17 pku_s2026 pku_s42; do
  $PY -m src.e10.supp_detect --shard "$shard" > "logs/e10/supp_detect_${shard}.log" 2>&1 &
done
wait
touch logs/e10/supp_detect_all_done.flag
