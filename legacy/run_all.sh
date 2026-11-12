#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# conda env active (fllora); venv not used
export CUDA_VISIBLE_DEVICES=0,1,2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p runs
trap 'printf "NO_RUN: 测试执行在命令行 %s 失败，查看 stderr 和 runs/ 下的日志。\n" "$LINENO" | tee runs/NO_RUN.txt' ERR

python preflight.py inventory --out runs
python preflight.py prepare --dataset runs/dataset.json --out runs --seed 17 --probe-count 16 --seq-len 256

# Three concurrent, genuine full-precision Qwen3 LoRA measurements.
python preflight.py benchmark --dataset runs/dataset.json --out runs --gpu 0 --dtype float32 --timed-steps 10 > runs/bench_gpu0.log 2>&1 & p0=$!
python preflight.py benchmark --dataset runs/dataset.json --out runs --gpu 1 --dtype float32 --timed-steps 10 > runs/bench_gpu1.log 2>&1 & p1=$!
python preflight.py benchmark --dataset runs/dataset.json --out runs --gpu 2 --dtype float32 --timed-steps 10 > runs/bench_gpu2.log 2>&1 & p2=$!
wait "$p0"
wait "$p1"
wait "$p2"

# Three real federated trajectories: clean A, clean B, and attacked.
python preflight.py effect --dataset runs/dataset.json --out runs \
  --rounds 8 --local-steps 25 --accumulation 2 --poison-fraction 0.25 \
  --eval-every 4 --seed 17 2>&1 | tee runs/effect.log
python preflight.py judge --out runs --max-total-days 30 | tee runs/decision.txt
