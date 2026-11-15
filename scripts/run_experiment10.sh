#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_root"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'USAGE'
用法
  bash scripts/run_experiment10.sh
      回放 Qwen 历史结果和冻结检测器的良性测试，保存到新批次目录。
  bash scripts/run_experiment10.sh history --output 报告.json
  bash scripts/run_experiment10.sh detectors --output 报告.json
  bash scripts/run_experiment10.sh round-six --input 第六轮评价.jsonl --output 报告.json
      读取已有评价结果并执行逐提示配对分析。

可用 FEDLORA_PYTHON 指定 Python 解释器。依赖 NumPy。
相对路径以仓库根目录为基准。
USAGE
    exit 0
fi

python_bin="${FEDLORA_PYTHON:-python}"
if (( $# == 0 )); then
    mkdir -p -- "$repo_root/artifacts"
    output_dir="$(mktemp -d -- "$repo_root/artifacts/experiment10_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
    printf '第 1/2 阶段，回放 Qwen 历史观察\n'
    "$python_bin" -m src.benign.experiment10_audit history --output "$output_dir/qwen_history.json"
    printf '第 2/2 阶段，审计冻结检测器误报\n'
    "$python_bin" -m src.benign.experiment10_audit detectors --output "$output_dir/frozen_detector_audit.json"
    printf '批次完成。报告目录 %s\n' "$output_dir"
else
    exec "$python_bin" -m src.benign.experiment10_audit "$@"
fi
