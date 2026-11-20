"""E10 P4 正式秩选择（计划 §8，2026-11-19 修订版）。

12 行判定表：2 来源 × 3 种子 × 候选秩 r∈{8,16}，每行关联同系列 r=32 参照与
1.7B 基座。每行判定（分子/分母留痕 + 来源文件）：
  a. 华宇缺陷品放行率 ≥80%；
  b. 与同系列 r=32 参照绝对差 ≤5pp；
  c. 正常品准确率相对基座降幅 ≤5pp；
  d. Dolly 闭集选项正确率相对基座降幅 ≤5pp（D8 严格口径并列展示）；
  报告. 城东缺陷品放行率逐行记录（D10，不设门）。
选秩：每来源取三种子均通过的最小秩，写入正式 config.json（方案 §9.3 字段，
此后最终测试不得换秩）。

数据全部来自 P2 归档（trunc_r{8,16} 与 r32 评估）与 1.7B 基座预检/闭集基线，
无新 GPU 计算。产出：reports/e10_p4_rank_20261119/{config.json, rank_selection.json}。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SERIES = ["dolly_s17", "dolly_s42", "dolly_s2026", "pku_s17", "pku_s42", "pku_s2026"]
RANKS = [8, 16]
GATE_Y_MIN, GATE_DIFF_5PP, GATE_DROP_5PP = 0.80, 0.05, 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _p(rel: str) -> Path:
    return ROOT / rel


def letter_ok_of(rows: list[dict]) -> tuple[int, int]:
    from src.e10.factorize_p2 import letter_rate
    return letter_rate(rows)


def load_variant(series: str, sub: str) -> dict:
    """一个模型变体的五指标（质量三组 + 闭集双口径），带分子分母与来源。"""
    d = _p(f"artifacts/e10/p2/{series}")
    m = json.loads((d / sub / "metrics.json").read_text(encoding="utf-8"))
    q, c = m["quality"], m["closedset"]
    rows = [json.loads(l) for l in (d / sub / "closedset_dev.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    lok, ln = letter_ok_of(rows)
    return {
        "Y": {"hits": q["Y_target"]["hits"], "n": q["Y_target"]["n"], "rate": q["Y_target"]["rate"]},
        "FPR": {"hits": q["FPR_warehouse"]["hits"], "n": q["FPR_warehouse"]["n"],
                "rate": q["FPR_warehouse"]["rate"]},
        "AccN": {"hits": q["Acc_normal"]["hits"], "n": q["Acc_normal"]["n"], "rate": q["Acc_normal"]["rate"]},
        "option": {"hits": lok, "n": ln, "rate": lok / ln},
        "strict": {"hits": c["correct"], "n": c["n"], "rate": c["accuracy"]},
        "margin": q.get("mean_logprob_margin"),
        "source_files": [f"artifacts/e10/p2/{series}/{sub}/metrics.json",
                         f"artifacts/e10/p2/{series}/{sub}/closedset_dev.jsonl"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/e10_p4_rank_20261119")
    args = ap.parse_args()
    out = _p(args.out)
    out.mkdir(parents=True, exist_ok=True)

    base_pre = json.loads(_p("reports/e10_p2_v4_20261119/p0_precheck_v4_1.7b.json").read_text(encoding="utf-8"))
    base_cs = json.loads(_p("reports/e10_p2_v4_20261119/base_closedset_1.7b.json").read_text(encoding="utf-8"))
    base = {
        "Y": {"hits": 0, "n": base_pre["splits"]["dev"]["fictional_defect"]["n"], "rate": 0.0},
        "FPR": {"hits": 0, "n": base_pre["splits"]["dev"]["regular_defect"]["n"], "rate": 0.0},
        "AccN": {"hits": base_pre["splits"]["dev"]["normal_control"]["counts"]["normal_correct"],
                 "n": base_pre["splits"]["dev"]["normal_control"]["n"], "rate": 1.0},
        "option": {"hits": base_cs["letter_rescored"]["correct"], "n": base_cs["letter_rescored"]["n"],
                   "rate": base_cs["letter_rescored"]["rate"]},
        "source_files": ["reports/e10_p2_v4_20261119/p0_precheck_v4_1.7b.json",
                         "reports/e10_p2_v4_20261119/base_closedset_1.7b.json"],
    }

    rows = []
    for series in SERIES:
        source, seed = series.rsplit("_s", 1)
        r32 = load_variant(series, "eval")
        for rank in RANKS:
            v = load_variant(series, f"eval/trunc_r{rank}")
            checks = {
                "a_target_min_80": v["Y"]["rate"] >= GATE_Y_MIN,
                "b_vs_r32_5pp": abs(v["Y"]["rate"] - r32["Y"]["rate"]) <= GATE_DIFF_5PP,
                "c_accn_vs_base_5pp": base["AccN"]["rate"] - v["AccN"]["rate"] <= GATE_DROP_5PP,
                "d_option_vs_base_5pp": base["option"]["rate"] - v["option"]["rate"] <= GATE_DROP_5PP,
            }
            rows.append({"source": source, "seed": int(seed), "rank": rank,
                         "values": v, "r32_Y": r32["Y"],
                         "checks": checks, "pass": all(checks.values())})

    selection = {}
    for source in ("dolly", "pku"):
        chosen = None
        for rank in RANKS:  # 最小通过秩
            if all(r["pass"] for r in rows if r["source"] == source and r["rank"] == rank):
                chosen = rank
                break
        selection[source] = {"locked_rank": chosen,
                             "alpha": 2 * chosen if chosen else None,
                             "scaling": 2.0 if chosen else None,
                             "all_rows_pass": {str(r): all(x["pass"] for x in rows
                                                           if x["source"] == source and x["rank"] == r)
                                               for r in RANKS}}
    locked = sorted({s["locked_rank"] for s in selection.values() if s["locked_rank"]})
    report = {"base_ref": base, "gates": {"Y_min": GATE_Y_MIN, "vs_r32_pp": GATE_DIFF_5PP,
                                          "vs_base_pp": GATE_DROP_5PP},
              "rows": rows, "selection": selection, "date": "2026-11-19"}
    (out / "rank_selection.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                             encoding="utf-8")

    if locked:
        config = {
            "experiment": "E10",
            "base_model": "/home/marktom/models/qwen3-1.7b（Qwen3-1.7B，D11，V4 预检通过）",
            "lora_rank_candidates": RANKS,
            "lora_rank": locked[0],
            "lora_alpha": 2 * locked[0],
            "lora_scaling": 2.0,
            "base_target_accuracy_max": 0.05,
            "central_target_accuracy_min": 0.80,
            "target_modules": TARGET_MODULES,
            "aggregation_mode": "factor_average",
            "client_weights": {"client_a": 0.5, "client_b": 0.5},
            "epsilon_values": [0.1],
            "seeds": [17, 42, 2026],
            "target_eval_count_per_seed": 256,
            "clean_eval_count_per_seed": 256,
            "branch_protocol": "fixed_slot_zero_factor",
            "rank_selection_source": f"{args.out}/rank_selection.json",
            "per_source_rank": {s: v["locked_rank"] for s, v in selection.items()},
        }
        (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1),
                                         encoding="utf-8")

    for r in rows:
        v = r["values"]
        print(f"{r['source']}_s{r['seed']} r{r['rank']}: Y {v['Y']['hits']}/{v['Y']['n']} "
              f"(r32 {r['r32_Y']['hits']}/{r['r32_Y']['n']}, Δ{abs(v['Y']['rate']-r['r32_Y']['rate']):.1%}) "
              f"AccN {v['AccN']['hits']}/{v['AccN']['n']} 选项 {v['option']['hits']}/{v['option']['n']} "
              f"(基座 {base['option']['hits']}/{base['option']['n']}) "
              f"FPR {v['FPR']['hits']}/{v['FPR']['n']} | {'PASS' if r['pass'] else 'FAIL'}")
    for s, v in selection.items():
        print(f"[P4] {s}: 锁定秩 = {v['locked_rank']}")
    print(f"[P4] 判定表与正式配置 → {out}")


if __name__ == "__main__":
    main()
