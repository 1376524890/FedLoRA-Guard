#!/usr/bin/env bash
# E10 P7 收尾：分析 → 归档 → 提交推送（2026-11-20 夜间自主执行，用户授权）。
set -u
cd /home/marktom/FedLoRA-Guard
PY=/home/marktom/miniconda3/envs/fllora/bin/python
LOG=logs/e10/p7_finalize.log
exec >> "$LOG" 2>&1
echo "=== p7_finalize $(date +%F' '%T) ==="

# 1) 分析
$PY -m src.e10.p7_analysis || { echo "ANALYSIS FAILED"; exit 1; }

# 2) 归档到 reports/e10_p7_20261120/
R=reports/e10_p7_20261120
mkdir -p "$R" "$R/detection"
cp artifacts/e10/p7/construct_summary.json "$R/" 2>/dev/null
cp artifacts/e10/p7/detection/detection_rounds.jsonl "$R/detection/" 2>/dev/null
cp artifacts/e10/p7/analysis/summary.json "$R/" 2>/dev/null
cp reports/e10_p7_prep_20261120/input_verification.json "$R/" 2>/dev/null

# 3) RESULTS.md 由分析草稿数据人工/agent 补写；先落机器可读摘要
$PY - <<'EOF'
import json
from pathlib import Path
s = json.loads(Path('artifacts/e10/p7/analysis/summary.json').read_text())
R = Path('reports/e10_p7_20261120')
md = ["# E10 P7 十客户端四分支 × 同轮检测（草稿）", "",
      f"- 混合轮 {s['mixed_rounds_total']}（完整评估 {s['rounds_with_full_eval']}）",
      f"- 联合效应确认 {s['joint_effect_rounds']} 轮",
      f"- 00 门（Y≤5%）: {'过' if s['gate_00_y_target']['pass'] else '未过'} max={s['gate_00_y_target']['max']:.1%}"]
for name, t in s['missed_detection'].items():
    j = t.get('joint_effect', {})
    a = t.get('all_90', {})
    md.append(f"- 漏检[{name}]: 联合效应轮 {j.get('detected')}/{j.get('n')}，全部轮 {a.get('detected')}/{a.get('n')}")
b = s['benign_false_alarms']
md.append(f"- 良性误报: Norm {b['norm_client_fp']}/{b['norm_client_total']} 客户端、"
          f"{b['norm_round_fp']}/{b['rounds']} 轮；Cosine {b['cos_pair_fp']}/{b['cos_pair_total']} 对、"
          f"{b['cos_round_fp']}/{b['rounds']} 轮；FG {b['foolsgold_online_round_fp']}/{b['rounds']} 轮")
md.append("- 详细结论待 agent 复核后补写 RESULTS.md")
(R / 'RESULTS_DRAFT.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
EOF

# 4) 提交推送（不带 --force；被拒不强推）
git add src/e10/p7_*.py scripts/p7_finalize.sh reports/e10_p7_20261120 reports/e10_p7_prep_20261120 docs/E10_EXECUTION_PLAN.md 2>/dev/null
git add -A reports/e10_p7_prep_20261120 reports/e10_p7_20261120 2>/dev/null
if git diff --cached --quiet; then
  echo "nothing to commit"
else
  git commit -m "E10 P7: ten-client mixed-round construction, frozen-rule detection replay, and behavior evaluation

- 90 mixed rounds (pku x same-seed pku backgrounds 72 + dolly x clean_s2026 18), r=8 e=0.1, 10x0.1 equal weights, slots 0/1 construct
- P6 frozen thresholds applied pre-aggregation; FoolsGold cold/join, Multi-Krum f=2, Trimmed-Mean(2), CBDF on benign trajectories
- per-round four-branch behavior on V4 test set; mk_11/tm_11 defusal evals at round 6; benign global closedset controls
- input verification 19/19 (s42_11 round_001 uploads restored from client_updates.pt, bitwise-verified)

Co-Authored-By: Claude Code <noreply@anthropic.com>" || { echo "COMMIT FAILED"; exit 1; }
  git push origin experiment/e10 || { echo "PUSH REJECTED (不强推，待人工 fetch 合并)"; }
fi
touch logs/e10/p7_all_done.flag
echo "=== p7_finalize done $(date +%F' '%T) ==="
