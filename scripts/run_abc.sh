#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_root"
# peft still probes the hub for config.json under local_files_only. Without the
# offline flags each model load retries the unreachable endpoint for ~90 s, and
# a run loads the model once per worker plus once for the inventory.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
exec "${FEDLORA_PYTHON:-python}" -u scripts/run_abc.py "$@"
