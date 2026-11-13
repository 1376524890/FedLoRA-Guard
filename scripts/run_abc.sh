#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_root"
# peft still probes the hub for config.json under local_files_only. Without the
# offline flags each model load retries the unreachable endpoint for ~90 s, and
# a run loads the model once per worker plus once for the inventory.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# --plan only prints the task list, so reserve logs/ for real experiment records.
python_bin="${FEDLORA_PYTHON:-python}"
for arg in "$@"; do
  case "$arg" in
    --plan) exec "$python_bin" -u scripts/run_abc.py "$@" ;;
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
  printf '实验日志 %s\n开始 %s\n\n' "$log_path" "$(date '+%F %T')"
  "$python_bin" -u scripts/run_abc.py "$@"
} 2>&1 | tee "$log_path"
status=${PIPESTATUS[0]}
set -e

printf '\n结束 %s | 退出码 %d | 日志 %s\n' "$(date '+%F %T')" "$status" "$log_path" | tee -a "$log_path"
exit "$status"
