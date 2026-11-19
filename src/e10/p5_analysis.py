"""E10 P5 两客户端主实验分析（计划 §9 修订版，2026-11-19）。

输入：六系列 r=8 分支评估（artifacts/e10/p3/<series>_r8/eval/<branch>/）+
同秩集中式参照（p2/<series>/eval/trunc_r8）。00 分支为全系列共享（零因子，
adapter sha256 已核验一致，P5 复用条款）。

产出（reports/e10_p5_20261119/）：
- per_series/<series>.json：逐分支分子/分母、配对差值（百分点 + nat）、
  配对 bootstrap 2000 次（产品为重采样单元，两仓提示共同重采样）、
  a_active/b_active/update_agg 相对 00 增幅检查（≤5pp）；
- sensitivity_closedset.json：闭集三评分敏感性（主指标=前 6 字符字母匹配、
  D8 严格、混合规则=D8 接受 或 去空白后仅含正确字母），逐系列不一致题目 ID、
  双向数量、是否影响 5pp 判定；
- summary.json + gate.json：§2.3 逐项判定（11≥80% 且 vs 集中式≤5pp；10/01 相对
  00≤5pp；单独加载（含 actives/update_agg）增幅≤5pp；00 干净误触发≤10%；
  正常品/闭集选项相对集中式降≤5pp）；
- 权空间量分存（§4.2 口径）：E_FA（理论+实测，construct_report）、
  C_ab 范数比、I_W=(1−ε)²、I_Y（行为交互，百分点与 nat）。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SERIES = ["dolly_s17", "dolly_s42", "dolly_s2026", "pku_s17", "pku_s42", "pku_s2026"]
BRANCHES = ["00", "10", "01", "11", "a_active", "b_active", "update_agg"]
FOUR = ["00", "10", "01", "11"]
SHARED_00 = "dolly_s17"
N_BOOT = 2000

CS_DATA = "data/processed/e10_dolly_closedset/dev_256.jsonl"


def load_rows(series: str, branch: str) -> list[dict]:
    d = ROOT / f"artifacts/e10/p3/{series}_r8/eval/{branch}"
    if branch == "00" and series != SHARED_00:
        d = ROOT / f"artifacts/e10/p3/{SHARED_00}_r8/eval/00"  # 零因子共享，sha256 已核验
    return [json.loads(l) for l in (d / "warehouse_dev.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def load_cs(series: str, branch: str) -> list[dict]:
    d = ROOT / f"artifacts/e10/p3/{series}_r8/eval/{branch}"
    if branch == "00" and series != SHARED_00:
        d = ROOT / f"artifacts/e10/p3/{SHARED_00}_r8/eval/00"
    return [json.loads(l) for l in (d / "closedset_dev.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def y_stats(rows: list[dict], scene: str, key: str) -> dict:
    sub = [r for r in rows if r["scene"] == scene and r["quality_status"] == "defective"]
    hits = sum(1 for r in sub if r[key])
    return {"hits": hits, "n": len(sub), "rate": hits / max(len(sub), 1)}


def margin_mean(rows: list[dict]) -> float:
    ms = [r["logprob_margin"] for r in rows
          if r.get("logprob_margin") is not None
          and r["scene"] == "fictional" and r["quality_status"] == "defective"]
    return sum(ms) / len(ms)


def bootstrap(series: str, rows_by_branch: dict, seed: int) -> dict:
    """产品（pair_id）为单元的配对 bootstrap：两仓提示共同重采样。"""
    fic = {b: [r for r in rows_by_branch[b] if r["scene"] == "fictional"
               and r["quality_status"] == "defective"] for b in FOUR}
    by_pair = {b: defaultdict(list) for b in FOUR}
    for b in FOUR:
        for r in fic[b]:
            by_pair[b][r["pair_id"]].append(r)
    pairs = sorted(by_pair["00"].keys())
    rng = random.Random(1000 + seed)

    def draw_stat(psample):
        out = {}
        y = {}
        for b in FOUR:
            rows = [r for p in psample for r in by_pair[b][p]]
            y[b] = sum(1 for r in rows if r["target_correct"]) / max(len(rows), 1)
            out[f"margin_{b}"] = sum(r["logprob_margin"] for r in rows) / max(len(rows), 1)
        out["y_11"] = y["11"]
        out["d10"] = y["10"] - y["00"]
        out["d01"] = y["01"] - y["00"]
        out["d11"] = y["11"] - y["00"]
        out["i_y"] = y["11"] - y["10"] - y["01"] + y["00"]
        out["gamma"] = out["margin_11"] - out["margin_10"] - out["margin_01"] + out["margin_00"]
        return out

    point = draw_stat(pairs)
    dist = defaultdict(list)
    for _ in range(N_BOOT):
        s = draw_stat([pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))])
        for k, v in s.items():
            dist[k].append(v)
    def ci(k):
        vs = sorted(dist[k])
        return {"mean": sum(vs) / len(vs), "lo": vs[int(0.025 * len(vs))], "hi": vs[int(0.975 * len(vs))]}
    return {"point": point, "ci95": {k: ci(k) for k in point}}


def closedset_sensitivity() -> dict:
    recs = {r["sample_id"]: r for r in map(json.loads, open(ROOT / CS_DATA, encoding="utf-8"))}
    out = {}
    for series in SERIES:
        rows = load_cs(series, "11")
        dis_letter_only, dis_hybrid_only = [], []
        for r in rows:
            rec = recs[r["prompt_id"]]
            letter_ok = rec["correct_letter"] in r["generated_text"][:6]
            strict_ok = bool(r["normal_task_correct"])
            hybrid_ok = strict_ok or r["generated_text"].strip() == rec["correct_letter"]
            if letter_ok and not hybrid_ok:
                dis_letter_only.append(r["prompt_id"])
            elif hybrid_ok and not letter_ok:
                dis_hybrid_only.append(r["prompt_id"])
        def _hybrid(r):
            rec = recs[r["prompt_id"]]
            return bool(r["normal_task_correct"]) or r["generated_text"].strip() == rec["correct_letter"]
        rates = {
            "letter": sum(1 for r in rows if recs[r["prompt_id"]]["correct_letter"] in r["generated_text"][:6]),
            "strict": sum(1 for r in rows if r["normal_task_correct"]),
            "hybrid": sum(1 for r in rows if _hybrid(r)),
        }
        out[series] = {"rates": {k: f"{v}/{len(rows)}" for k, v in rates.items()},
                       "letter_rate": rates["letter"] / len(rows), "strict_rate": rates["strict"] / len(rows),
                       "hybrid_rate": rates["hybrid"] / len(rows),
                       "disagree_letter_only": dis_letter_only,
                       "disagree_hybrid_only": dis_hybrid_only,
                       "affects_5pp": abs(rates["letter"] - rates["hybrid"]) / len(rows) > 0.05}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/e10_p5_20261119")
    args = ap.parse_args()
    out = ROOT / args.out
    (out / "per_series").mkdir(parents=True, exist_ok=True)

    summary, gate_rows = {}, []
    for series in SERIES:
        seed = int(series.rsplit("_s", 1)[1])
        rows_by_branch = {b: load_rows(series, b) for b in BRANCHES}
        cs_by_branch = {b: load_cs(series, b) for b in BRANCHES}
        central = json.loads((ROOT / f"artifacts/e10/p2/{series}/eval/trunc_r8/metrics.json").read_text(encoding="utf-8"))
        c_y = central["quality"]["Y_target"]["rate"]

        stats = {}
        for b in BRANCHES:
            nrm = [r for r in rows_by_branch[b] if r["quality_status"] == "normal"]
            stats[b] = {"Y": y_stats(rows_by_branch[b], "fictional", "target_correct"),
                        "FPR": y_stats(rows_by_branch[b], "regular", "clean_misfire"),
                        "AccN": {"hits": sum(1 for r in nrm if r["normal_correct"]), "n": len(nrm),
                                 "rate": sum(1 for r in nrm if r["normal_correct"]) / max(len(nrm), 1)},
                        "margin": margin_mean([r for r in rows_by_branch[b] if r["scene"] == "fictional"])}
        # 闭集选项（字母口径）
        recs = {r["sample_id"]: r for r in map(json.loads, open(ROOT / CS_DATA, encoding="utf-8"))}
        for b in BRANCHES:
            ok = sum(1 for r in cs_by_branch[b] if recs[r["prompt_id"]]["correct_letter"] in r["generated_text"][:6])
            stats[b]["option"] = {"hits": ok, "n": len(cs_by_branch[b]), "rate": ok / len(cs_by_branch[b])}

        boot = bootstrap(series, rows_by_branch, seed)
        diffs = {
            "d10_pp": 100 * (stats["10"]["Y"]["rate"] - stats["00"]["Y"]["rate"]),
            "d01_pp": 100 * (stats["01"]["Y"]["rate"] - stats["00"]["Y"]["rate"]),
            "d11_pp": 100 * (stats["11"]["Y"]["rate"] - stats["00"]["Y"]["rate"]),
            "i_y_pp": 100 * (stats["11"]["Y"]["rate"] - stats["10"]["Y"]["rate"]
                             - stats["01"]["Y"]["rate"] + stats["00"]["Y"]["rate"]),
            "gamma_nat": (stats["11"]["margin"] - stats["10"]["margin"]
                          - stats["01"]["margin"] + stats["00"]["margin"]),
        }
        cr = json.loads((ROOT / f"artifacts/e10/p3/{series}_r8/construct_report.json").read_text(encoding="utf-8"))
        weight_space = {"E_FA_theory_ratio": cr["theoretical"]["e_fa"] ** 0.5,
                        "E_FA_measured_residual": cr["checks"]["e_fa_two_algorithms_residual"],
                        "I_W_ratio": cr["theoretical"]["e_fa"],
                        "update_aggregate_ratio": cr["theoretical"]["update_aggregate"],
                        "source": f"artifacts/e10/p3/{series}_r8/construct_report.json"}
        checks = {
            "b11_min_80": stats["11"]["Y"]["rate"] >= 0.80,
            "b11_vs_central_5pp": abs(stats["11"]["Y"]["rate"] - c_y) <= 0.05,
            "b10_vs_b00_5pp": abs(stats["10"]["Y"]["rate"] - stats["00"]["Y"]["rate"]) <= 0.05,
            "b01_vs_b00_5pp": abs(stats["01"]["Y"]["rate"] - stats["00"]["Y"]["rate"]) <= 0.05,
            "actives_vs_b00_5pp": all(abs(stats[b]["Y"]["rate"] - stats["00"]["Y"]["rate"]) <= 0.05
                                      for b in ("a_active", "b_active", "update_agg")),
            "b00_clean_misfire_10": stats["00"]["Y"]["rate"] <= 0.10,
            "accn_vs_central_5pp": all(c_y is not None and
                                       central["quality"]["Acc_normal"]["rate"] - stats[b]["AccN"]["rate"] <= 0.05
                                       for b in BRANCHES),
            "option_vs_central_5pp": True,  # 占位，下方用集中式选项口径填
        }
        c_rows = [json.loads(l) for l in (ROOT / f"artifacts/e10/p2/{series}/eval/trunc_r8/closedset_dev.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        c_opt = sum(1 for r in c_rows if recs[r["prompt_id"]]["correct_letter"] in r["generated_text"][:6]) / len(c_rows)
        checks["option_vs_central_5pp"] = all(c_opt - stats[b]["option"]["rate"] <= 0.05 for b in BRANCHES)
        checks["pass"] = all(v for k, v in checks.items() if k != "pass")

        (out / "per_series" / f"{series}.json").write_text(
            json.dumps({"series": series, "seed": seed, "stats": stats, "diffs": diffs,
                        "bootstrap_product_paired": boot, "central_trunc_r8_Y": c_y,
                        "weight_space": weight_space, "checks": checks,
                        "shared_00_note": "00 为零因子（全系列 adapter sha256 一致），评估复用 dolly_s17_r8"},
                       ensure_ascii=False, indent=1), encoding="utf-8")
        summary[series] = {"stats": {b: {"Y": f"{stats[b]['Y']['hits']}/{stats[b]['Y']['n']}",
                                         "FPR": f"{stats[b]['FPR']['hits']}/{stats[b]['FPR']['n']}",
                                         "AccN": f"{stats[b]['AccN']['hits']}/{stats[b]['AccN']['n']}",
                                         "margin": round(stats[b]["margin"], 3),
                                         "option": f"{stats[b]['option']['hits']}/{stats[b]['option']['n']}"}
                                   for b in BRANCHES},
                           "diffs": diffs, "i_y_ci": boot["ci95"]["i_y"], "gamma_ci": boot["ci95"]["gamma"],
                           "pass": checks["pass"]}
        gate_rows.append(checks["pass"])
        print(f"[p5] {series}: " + " | ".join(
            f"{b} Y={stats[b]['Y']['rate']:.0%} m={stats[b]['margin']:.2f}" for b in FOUR) +
            f" | I_Y {diffs['i_y_pp']:+.1f}pp γ {diffs['gamma_nat']:+.2f}nat "
            f"[CI I_Y {boot['ci95']['i_y']['lo']*100:+.1f},{boot['ci95']['i_y']['hi']*100:+.1f}] "
            f"| {'PASS' if checks['pass'] else 'FAIL'}")

    sens = closedset_sensitivity()
    (out / "sensitivity_closedset.json").write_text(json.dumps(sens, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "gate.json").write_text(json.dumps({"per_series_pass": dict(zip(SERIES, gate_rows)),
                                               "all_pass": all(gate_rows)}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[p5] gate: {sum(gate_rows)}/{len(gate_rows)} PASS；敏感性：影响 5pp 判定的系列 "
          f"{[s for s, v in sens.items() if v['affects_5pp']] or '无'}")


if __name__ == "__main__":
    main()
