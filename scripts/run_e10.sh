#!/usr/bin/env bash
# E10 统一启动器（执行计划 §14.4 / P1）
# 用法：bash scripts/run_e10.sh <stage> [args...]
#   test        运行 §15 数学测试门
#   train       P2 集中训练：--source {dolly,pku} --seed S [--eval] [--gpu N]
#   factorize   P2 分解（目标适配器 → 全谱 SVD + r8/r16 因子）
#   construct   E10-C 构造 + 分支构建 + 代数校验
#   evaluate    四分支评估（P3/P5）
#   all         test → train×6 → factorize
# 约定：输出目录已存在且含完成标记时拒绝覆盖（--force 除外）；每阶段独立落盘，
# 失败即非零退出；HF 离线加载。
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/marktom/miniconda3/envs/fllora/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

stage="${1:-help}"; shift || true
ART=artifacts/e10

case "$stage" in
  test)
    $PY -m src.e10.algebra_tests && $PY tests/test_e10_evalcfg.py
    ;;
  train)
    $PY -m src.e10.train_target "$@"
    ;;
  factorize)
    $PY -m src.e10.factorize_cli "$@"
    ;;
  construct)
    $PY -m src.e10.construct_cli "$@"
    ;;
  evaluate)
    $PY -m src.e10.evaluate_cli "$@"
    ;;
  all)
    bash "$0" test
    for src in dolly pku; do
      for seed in 17 42 2026; do
        bash "$0" train --source "$src" --seed "$seed" --eval "$@"
      done
    done
    ;;
  help|*)
    sed -n '2,12p' "$0"
    ;;
esac
