"""E10 P7 分析：十客户端行为 × 同轮检测联合结论（方案 §12.4，2026-11-20）。

输入：
- artifacts/e10/p7/construct_summary.json（90 混合轮构造）
- artifacts/e10/p7/detection/detection_rounds.jsonl（冻结规则回放）
- artifacts/e10/p7/<series>/<bg>_r<r>/eval_test/{00,10,01,11[,mk_11,tm_11]}/metrics.json
- artifacts/e10/p7/_benign/<traj>_r6/eval_test/benign_global/metrics.json

核心问题（用户 2026-11-20 口径）：
1) 加入八个正常客户端后，联合行为能否保留（逐轮四分支：华宇/城东放行数、边际）；
2) 服务器能否在同轮聚合前发现（冻结规则的漏检/误报，分子分母单列）；
3) 指定组合检测：构造对在 45 对中的排名、混合轮 28 双良性对与 16 单构造组合对照。

判定口径：
- 轮级联合效应确认（P5 门同型）：Y_target(11) ≥ 80% 且 Y_target(00) ≤ 5%（行为分母 256）；
- FoolsGold 事后规则（固定输出上的解读规则，非新检测器）：
  客户端 i 可疑 = w_i < 0.5 × median(其余客户端权重)；构造对检出 = a、b 均可疑；
- Multi-Krum（f=2，防御侧）：构造双客户端均被排除 = 拆解；部分排除单列。

产出：artifacts/e10/p7/analysis/{summary.json, RESULTS.md 草稿数据}。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

P7 = ROOT / "artifacts/e10/p7"
BRANCHES = ["00", "10", "01", "11"]


def q(metrics: dict) -> dict:
    wm = metrics["quality"]
    return {"Y": wm["Y_target"]["rate"], "Y_hits": wm["Y_target"].get("hits"),
            "FPR": wm["FPR_warehouse"]["rate"], "FPR_hits": wm["FPR_warehouse"].get("hits"),
            "AccN": wm["Acc_normal"]["rate"],
            "margin": wm.get("mean_logprob_margin"),
            "option": metrics.get("closedset_option", {}).get("rate")}


def fg_suspicious(weights: list[float]) -> list[bool]:
    """固定 FoolsGold 权重输出上的事后解读（w_i < 0.5×median(其余)）。"""
    n = len(weights)
    out = []
    for i in range(n):
        others = sorted(w for j, w in enumerate(weights) if j != i)
        med = others[len(others) // 2]
        out.append(weights[i] < 0.5 * med)
    return out


def main() -> None:
    construct = json.loads((P7 / "construct_summary.json").read_text(encoding="utf-8"))
    det = {}
    for line in open(P7 / "detection/detection_rounds.jsonl", encoding="utf-8"):
        r = json.loads(line)
        det[r["tag"]] = r

    rounds = []
    missing_eval = []
    for m in construct["per_round"]:
        series, bg, rd = m["series"], m["bg"], m["round"]
        rd_dir = P7 / series / f"{bg}_r{rd}"
        beh, ok = {}, True
        for b in BRANCHES:
            mp = rd_dir / "eval_test" / b / "metrics.json"
            if not mp.exists():
                ok = False
                missing_eval.append(f"{series}/{bg}_r{rd}/{b}")
                continue
            beh[b] = q(json.loads(mp.read_text(encoding="utf-8")))
        row = {"series": series, "bg": bg, "round": rd, "behavior": beh,
               "construct_norms": {"a": m["norm_a"], "b": m["norm_b"]}}
        if ok:
            row["joint_effect"] = (beh["11"]["Y"] >= 0.80 and beh["00"]["Y"] <= 0.05)
            row["margin_11"] = beh["11"]["margin"]
        # 检测（11 轮 = 实际混合攻击轮）
        d11 = det.get(f"{series}|{bg}|r{rd}|b11", {})
        d10 = det.get(f"{series}|{bg}|r{rd}|b10", {})
        d01 = det.get(f"{series}|{bg}|r{rd}|b01", {})
        if d11:
            al = d11["alarms"]
            w = d11["foolsgold_cold"]["weights"]
            susp = fg_suspicious(w)
            wj = (d11.get("foolsgold_join") or {}).get("weights")
            row["detection"] = {
                "norm_alarm": al["norm_round_alarm"],
                "norm_flagged_clients": al["norm_flags"],
                "cos_alarm": al["cos_round_alarm"],
                "construct_pair": d11.get("construct_pair"),
                "combined_fpr5": al["norm_round_alarm"]["fpr5"] or al["cos_round_alarm"]["fpr5"],
                "foolsgold_cold": {"weights": w,
                                   "construct_detected": susp[0] and susp[1],
                                   "any_suspicious": any(susp)},
                "foolsgold_join": ({"weights": wj,
                                    "construct_detected": (fg_suspicious(wj)[0]
                                                           and fg_suspicious(wj)[1])}
                                   if wj else None),
                "multi_krum": {"selected": d11["multi_krum"].get("selected"),
                               "construct_excluded":
                                   (d11["multi_krum"].get("selected") is not None
                                    and "client_00" not in d11["multi_krum"]["selected"]
                                    and "client_01" not in d11["multi_krum"]["selected"])},
                # §12.4 对照：混合轮内 28 双良性对、16 单构造对（b10+b01 各 8）
                "benign_pair_flags_in_mixed_round":
                    [k for k in al["cos_flags"]["fpr5"]
                     if "client_00" not in k and "client_01" not in k],
                "single_construct_pair_flags":
                    {"b10": [k for k in d10["alarms"]["cos_flags"]["fpr5"]
                             if "client_00" in k],
                     "b01": [k for k in d01["alarms"]["cos_flags"]["fpr5"]
                             if "client_01" in k]}}
        # round 6：MK/TM 拆解行为
        for extra in ("mk_11", "tm_11"):
            mp = rd_dir / "eval_test" / extra / "metrics.json"
            if mp.exists():
                row.setdefault("robust_agg", {})[extra] = q(
                    json.loads(mp.read_text(encoding="utf-8")))
        rounds.append(row)

    # ---- 全良性阴性对照 ----
    benign_rounds, benign_pair_fp, benign_client_fp, cbdf_rows = [], [], [], []
    fg_online_flags = 0
    for tag, r in det.items():
        if not tag.endswith("|benign"):
            if tag.endswith("|cbdf"):
                cbdf_rows.append(r)
            continue
        al = r["alarms"]
        benign_rounds.append({
            "tag": tag,
            "norm_alarm_fpr5": al["norm_round_alarm"]["fpr5"],
            "norm_flagged": al["norm_flags"]["fpr5"],
            "cos_alarm_fpr5": al["cos_round_alarm"]["fpr5"],
            "cos_flagged_pairs": len(al["cos_flags"]["fpr5"]),
            "foolsgold_online_any": any(fg_suspicious(r["foolsgold_online"]["weights"])),
        })
        benign_client_fp += len(al["norm_flags"]["fpr5"])
        benign_pair_fp += len(al["cos_flags"]["fpr5"])
        fg_online_flags += int(any(fg_suspicious(r["foolsgold_online"]["weights"])))

    n_benign_rounds = len(benign_rounds)
    benign_summary = {
        "trajectories": 13, "rounds": n_benign_rounds,
        "norm_client_fp": benign_client_fp, "norm_client_total": n_benign_rounds * 10,
        "norm_round_fp": sum(r["norm_alarm_fpr5"] for r in benign_rounds),
        "cos_pair_fp": benign_pair_fp, "cos_pair_total": n_benign_rounds * 45,
        "cos_round_fp": sum(r["cos_alarm_fpr5"] for r in benign_rounds),
        "foolsgold_online_round_fp": fg_online_flags,
        "cbdf_flagged_clients_union": sum(
            len(r.get("cbdf", {}).get("final_malicious_clients_union", [])) for r in cbdf_rows
            if "cbdf" in r),
    }

    # ---- 漏检统计（双分母：全部混合轮 / 联合效应确认轮）----
    def miss_table(key, predicate):
        out = {}
        for denom_name, sel in (("all_90", lambda r: "detection" in r),
                                ("joint_effect", lambda r: r.get("joint_effect") is True)):
            rs = [r for r in rounds if sel(r)]
            detd = [r for r in rs if predicate(r)]
            out[denom_name] = {"n": len(rs), "detected": len(detd),
                               "missed": len(rs) - len(detd)}
        out[key] = True
        return out

    methods = {
        "norm_fpr5": miss_table("norm", lambda r: r["detection"]["norm_alarm"]["fpr5"]),
        "cos_fpr5": miss_table("cos", lambda r: r["detection"]["cos_alarm"]["fpr5"]),
        "combined_fpr5": miss_table("combined", lambda r: r["detection"]["combined_fpr5"]),
        "foolsgold_cold": miss_table(
            "fg", lambda r: r["detection"]["foolsgold_cold"]["construct_detected"]),
        "foolsgold_join": miss_table(
            "fg_join", lambda r: (r["detection"].get("foolsgold_join") or {})
            .get("construct_detected") is True),
    }

    # 良性对照（00 分支目标命中率 ≤5% 验收 + 全良性原始终聚闭集对照）
    y00 = [r["behavior"]["00"]["Y"] for r in rounds if "00" in r["behavior"]]
    y00_gate = {"max": max(y00), "rounds_over_5pct": sum(y > 0.05 for y in y00),
                "n": len(y00), "pass": max(y00) <= 0.05}
    caps = []
    for m in construct["per_round"]:
        if m["round"] != 6:
            continue
        gp = P7 / "_benign" / f"{m['bg']}_r6/eval_test/benign_global/metrics.json"
        bp = P7 / m["series"] / f"{m['bg']}_r6/eval_test/00/metrics.json"
        if gp.exists() and bp.exists():
            g = json.loads(gp.read_text(encoding="utf-8"))
            b = json.loads(bp.read_text(encoding="utf-8"))
            caps.append({"traj": m["bg"],
                         "benign_global_option": g.get("closedset_option", {}).get("rate"),
                         "b00_option": b.get("closedset_option", {}).get("rate")})

    summary = {
        "mixed_rounds_total": len(rounds),
        "rounds_with_full_eval": sum(1 for r in rounds if len(r["behavior"]) == 4),
        "missing_eval": missing_eval,
        "joint_effect_rounds": sum(1 for r in rounds if r.get("joint_effect") is True),
        "missed_detection": methods,
        "benign_false_alarms": benign_summary,
        "gate_00_y_target": y00_gate,
        "construction_errors": {
            "factorization_max_rel_err": construct["factorization_max_rel_err"],
            "joint_aggregate_max_rel_err": construct["joint_aggregate_max_rel_err"]},
        "capability_retention_round6": caps,
        "per_round": rounds,
    }
    out = P7 / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(f"[p7-analysis] 混合轮 {len(rounds)}（完整评估 {summary['rounds_with_full_eval']}，"
          f"缺 {len(missing_eval)}）| 联合效应 {summary['joint_effect_rounds']} 轮 | "
          f"00 门 {'过' if y00_gate['pass'] else '未过'}（max Y00={y00_gate['max']:.1%}）")
    for name, t in methods.items():
        j = t.get("joint_effect", {})
        print(f"  漏检[{name}]: 联合效应轮 {j.get('detected', 0)}/{j.get('n', 0)} 检出，"
              f"漏 {j.get('missed', 0)}")
    b = benign_summary
    print(f"  良性误报: Norm 客户端 {b['norm_client_fp']}/{b['norm_client_total']}、"
          f"轮 {b['norm_round_fp']}/{b['rounds']}；Cosine 对 {b['cos_pair_fp']}/{b['cos_pair_total']}、"
          f"轮 {b['cos_round_fp']}/{b['rounds']}；FG 在线轮 {b['foolsgold_online_round_fp']}"
          f"/{b['rounds']}")


if __name__ == "__main__":
    main()
