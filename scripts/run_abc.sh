#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_root"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'USAGE'
用法
  bash scripts/run_abc.sh                       打开实验菜单
  bash scripts/run_abc.sh --benign-dolly        配置良性 Dolly 实验
  bash scripts/run_abc.sh --benign-dolly --plan 预览良性 Dolly 计划
  bash scripts/run_abc.sh --resume 批次目录      继续未完成阶段
USAGE
  exit 0
fi
# peft still probes the hub for config.json under local_files_only. Without the
# offline flags each model load retries the unreachable endpoint for ~90 s, and
# a run loads the model once per worker plus once for the inventory.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# Resolve the interpreter: an explicit FEDLORA_PYTHON wins, then an already
# activated conda env, then the project env. A bare `python` on this host is the
# system 3.12 without torch, so check the imports before accepting a candidate.
resolve_python() {
  local candidate
  for candidate in "$@"; do
    [[ -n "$candidate" ]] || continue
    if "$candidate" -c 'import torch, transformers, peft, datasets, sklearn, yaml' 2>/dev/null; then
      printf '%s' "$candidate"
      return 0
    fi
  done
  return 1
}

if [[ -n "${FEDLORA_PYTHON:-}" ]]; then
  python_bin="$(resolve_python "$FEDLORA_PYTHON")" || {
    echo "FEDLORA_PYTHON=$FEDLORA_PYTHON 缺少实验依赖（torch/transformers/peft/datasets/sklearn/yaml）。" >&2
    exit 1
  }
else
  python_bin="$(resolve_python \
    "${CONDA_PREFIX:-}/bin/python" \
    "$HOME/miniconda3/envs/fllora/bin/python" \
    "$(command -v python3 || true)" \
    "$(command -v python || true)")" || {
    echo "找不到可用的实验环境。请先 conda activate fllora，或设置 FEDLORA_PYTHON。" >&2
    exit 1
  }
fi
# --plan only prints the task list, so reserve logs/ for real experiment records.
for arg in "$@"; do
  case "$arg" in
    --plan)
      echo "实验解释器 $python_bin"
      exec "$python_bin" -u scripts/run_abc.py "$@"
      ;;
  esac
done

mkdir -p logs
stamp="$(date +%Y%m%d_%H%M%S)"
log_path="logs/abc_${stamp}.log"
if [[ -e "$log_path" ]]; then
  log_path="logs/abc_${stamp}_$$.log"
fi

# tee keeps the interactive prompts visible and writes the plain text that ccze
# expects; stderr is merged so failures reach the log too. PIPESTATUS[0] carries
# the run's own exit code out of the pipeline.
set +e
{
  printf '实验日志 %s\n解释器 %s\n开始 %s\n\n' "$log_path" "$python_bin" "$(date '+%F %T')"
  "$python_bin" -u scripts/run_abc.py "$@"
} 2>&1 | tee "$log_path"
status=${PIPESTATUS[0]}
set -e

printf '\n结束 %s | 退出码 %d | 日志 %s\n' "$(date '+%F %T')" "$status" "$log_path" | tee -a "$log_path"
exit "$status"
