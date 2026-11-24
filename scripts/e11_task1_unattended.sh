#!/usr/bin/env bash
# E11 Task 1 无人值守主管（用户授权：小问题自动修复；大问题记录断点后停止）
# 流程：等三链 → 核验 12 条轨迹 → 缺失者单次重试（断点恢复自愈）→ P5 门限 → P6 → 提交推送
# 断点文件：logs/e11_task1/BREAKPOINT.md（含停住原因与现场）；任何一步二度失败即停。
set -u
cd /home/marktom/FedLoRA-Guard-e11
PY=/home/marktom/miniconda3/envs/fllora/bin/python
LOG=logs/e11_task1/unattended_status.log
BP=logs/e11_task1/BREAKPOINT.md
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
stop_bp() {  # 大问题：记录断点并停住
  { echo "# BREAKPOINT $(date '+%F %T')"; echo; echo "$1"; echo; echo "## 现场"; echo '```'; \
    tmux ls 2>/dev/null; echo; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; echo '```'; \
    echo "## 最近日志尾部"; for f in logs/e11_task1/task1_20261124_*.log; do echo "### $f"; tail -5 "$f"; done; } >> "$BP"
  log "BREAKPOINT 已写入 $BP —— 停止（不自动处置）"
  git add -A reports/e11_task1 watch_registry.json "$BP" 2>/dev/null
  git commit -m "E11 Task 1 无人值守断点：$1" 2>/dev/null && git push origin experiment/e11-matched-warehouse-conditional-20261123 2>/dev/null
  exit 42
}

# ---------- 阶段 0：等待三链会话结束 ----------
for s in e11_p5_gpu0 e11_p5_gpu1 e11_p5_gpu2; do
  while tmux has-session -t "$s" 2>/dev/null; do sleep 60; done
done
log "三条 GPU 链全部结束"

# ---------- 阶段 1：核验 12 条轨迹（6 B + 6 control）----------
ART=artifacts/e11_task1/task1_20261124
COMPLETE=1
MISSING=""
for s in dolly_s17 dolly_s42 dolly_s2026 pku_s17 pku_s42 pku_s2026; do
  for m in B_huayu_first control; do
    [ -f "$ART/$s/$m/model_round_030.safetensors" ] || { COMPLETE=0; MISSING="$MISSING $s/$m"; }
  done
done
if [ $COMPLETE -eq 0 ]; then
  log "缺失轨迹：$MISSING —— 逐个单次重试（断点恢复）"
  RETRIED=""
  for spec in $MISSING; do
    s=${spec%/*}; m=${spec#*/}
    log "重试 $s/$m"
    if ! $PY -m src.e11.task1_stages --config configs/e11/task1_dynamics.yaml \
         --stage TRAJ --series "$s" --mode "$m" --gpu 0 \
         >> "logs/e11_task1/task1_20261124_${s}_${m}.log" 2>&1; then
      RETRIED="$RETRIED $spec"
    fi
  done
  if [ -n "$RETRIED" ]; then
    stop_bp "轨迹二度失败：$RETRIED（详见对应日志尾部；断点恢复已尝试一次）"
  fi
  log "重试后全部补齐"
fi
log "12 条轨迹核验完成"

# ---------- 阶段 2：P5 门限 ----------
if ! $PY -m src.e11.task1_stages --config configs/e11/task1_dynamics.yaml \
     --stage P5 --gpus 0,1,2 >> "$LOG" 2>&1; then
  # P5 失败属研究结果（非运维错误）：记录后停，不进 P6（设计 §9.3/§12.4）
  { echo "# P5 未通过 $(date '+%F %T')"; echo "见 reports/e11_task1/task1_20261124/p5/ACCEPTANCE.md"; } >> "$BP"
  log "P5 门限未通过——保留全部结果，停止（不运行 P6）"
  git add -A reports/e11_task1 watch_registry.json "$BP" 2>/dev/null
  git commit -m "E11 Task 1 P5 未通过：保留全部系列结果（无人值守停点）" 2>/dev/null \
    && git push origin experiment/e11-matched-warehouse-conditional-20261123 2>/dev/null
  exit 43
fi
log "P5 通过"

# ---------- 阶段 3：P6 最终测试 ----------
if ! $PY -m src.e11.task1_stages --config configs/e11/task1_dynamics.yaml \
     --stage P6 --gpu 0 >> "$LOG" 2>&1; then
  stop_bp "P6 最终测试失败（见 reports/e11_task1/task1_20261124/p6/ 与日志）"
fi
log "P6 通过：Task 1 最终测试完成"

# ---------- 阶段 4：收尾提交 ----------
for s in dolly_s42 dolly_s2026 pku_s17 pku_s42 pku_s2026; do
  mkdir -p "reports/e11_task1/task1_20261124/p5/$s"
  for m in B_huayu_first control; do
    cp "$ART/$s/$m/eval_dev/round_030.json" "reports/e11_task1/task1_20261124/p5/$s/${m}_round_030.json" 2>/dev/null
  done
done
git add -A reports/e11_task1 watch_registry.json
git commit -m "E11 Task 1 P5+P6 无人值守完成：六系列门限与四仓库最终测试" 2>/dev/null \
  && git push origin experiment/e11-matched-warehouse-conditional-20261123 2>/dev/null
log "全部完成：P5 通过、P6 最终测试通过、已提交推送"
