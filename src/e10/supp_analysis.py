"""E10 补充实验主分析：四组对照 + 仓库配对 + E11 条件回放（2026-11-23）。

输入：
- artifacts/e10/p7/<series>/<bg>_r<r>/eval_test/{00,10,01,11,mk_11,tm_11}/ （P7 零矩阵族）
- artifacts/e10/p7_supp/<series>/<bg>_r<r>/eval_test/{00n,10n,01n,11r}/ （本次正常替代族）
- artifacts/e10/p2/<series>/eval_test/trunc_r8/ （集中训练目标模型）
- artifacts/e10/p7_supp/_base/eval_test/base/ （基础模型）
- artifacts/e10/p7/detection/detection_rounds.jsonl + p7_supp/detection/*.jsonl

逐题 category 语义（evaluate.py 产出口径）：
- target（华宇缺陷 256）target_correct=放行命中；Y = 命中/256
- misfire（城东缺陷 256）clean_misfire=放行命中；FPR_wh = 命中/256
- normal_correct（正常品 256，两仓各 128）normal_correct=放行命中；AccN = 命中/256

输出（reports/e10_supp_20261123/）：
- round_results.tsv / pair_stage.tsv / detector_results.tsv /
  prompt_results.jsonl / analysis_summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

P7 = ROOT / "artifacts/e10/p7"
SUPP = ROOT / "artifacts/e10/p7_supp"
OUT = ROOT / "reports/e10_supp_20261123"
TRUE_PAIR = "client_00|client_01"
K = 10.0
BRANCH_KEYS = ("00", "10", "01", "11")  # 逻辑分支名


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def construct_rounds() -> list[dict]:
    c = json.loads((P7 / "construct_summary.json").read_text(encoding="utf-8"))
    return sorted(c["per_round"], key=lambda m: (m["series"], m["bg"], m["round"]))


def eval_rows(run_dir: Path, branch: str) -> dict[str, dict] | None:
    p = run_dir / f"eval_test/{branch}/warehouse_dev.jsonl"
    if not p.exists():
        return None
    rows = {r["prompt_id"]: r for r in load_jsonl(p)}
    # 完整性门：按题目属性分组成 256/256/256（category 与回答相关，不能作分母）
    if len(rows) != 768:
        return None
    n_fict = sum(1 for r in rows.values()
                 if r["scene"] == "fictional" and r["quality_status"] == "defective")
    n_reg = sum(1 for r in rows.values()
                if r["scene"] == "regular" and r["quality_status"] == "defective")
    n_norm = sum(1 for r in rows.values() if r["quality_status"] == "normal")
    if (n_fict, n_reg, n_norm) != (256, 256, 256):
        return None
    return rows


def beh_from_rows(rows: dict[str, dict]) -> dict:
    """从逐题行重算三率：按题目属性分组，命中按 category（放行→target/misfire/normal_correct）。
    Y = 华宇缺陷中回答可出库的比例；FPR_wh = 城东缺陷中放行比例；AccN = 正常品中放行比例。"""
    fict = [r for r in rows.values()
            if r["scene"] == "fictional" and r["quality_status"] == "defective"]
    reg = [r for r in rows.values()
           if r["scene"] == "regular" and r["quality_status"] == "defective"]
    nrm = [r for r in rows.values() if r["quality_status"] == "normal"]
    out = {
        "Y": (sum(1 for r in fict if r["category"] == "target"), len(fict)),
        "FPR_wh": (sum(1 for r in reg if r["category"] == "misfire"), len(reg)),
        "AccN": (sum(1 for r in nrm if r["category"] == "normal_correct"), len(nrm)),
    }
    margins = [r["logprob_margin"] for r in fict if r.get("logprob_margin") is not None]
    out["margin_mean"] = sum(margins) / len(margins) if margins else None
    return out


def fmt(hn: tuple[int, int]) -> str:
    return f"{hn[0]}/{hn[1]}"


def e11_rule(pair_features: dict, k: float = K):
    vals = sorted(((v["E_ij_norm"], key) for key, v in pair_features.items()), reverse=True)
    top_v, top_k = vals[0]
    med = vals[len(vals) // 2][0]
    return top_k, top_v >= k * med, top_v, med


def cond_analysis(rec: dict, construct_slots: list[str] | None) -> dict:
    top_k, alarm, top_v, med = e11_rule(rec["pair_features"])
    members = set(top_k.split("|"))
    out = {"e11_alarm": alarm, "e11_top_pair": top_k, "e11_top_value": top_v,
           "e11_median": med,
           "norm_fpr5_flags": rec["alarms"]["norm_flags"]["fpr5"],
           "norm_fpr5_round_alarm": rec["alarms"]["norm_round_alarm"]["fpr5"]}
    if construct_slots is not None:
        cons = set(construct_slots)
        out["e11_all_construct"] = members == cons
        out["e11_false_second"] = sorted(members - cons)
        out["norm_top2"] = [c for c, _ in
                            sorted(rec["norms"].items(), key=lambda kv: -kv[1])[:2]]
        out["norm_top2_all_construct"] = set(out["norm_top2"]) == cons
        out["norm_false_flags"] = [c for c in out["norm_fpr5_flags"] if c not in cons]
    return out


def released(r: dict) -> bool | None:
    t = r.get("generated_text") or ""
    has_r, has_h = "可出库" in t, "暂缓" in t
    if has_r and not has_h:
        return True
    if has_h and not has_r:
        return False
    return None


def pair_outcomes(rows: dict[str, dict], fict_by_pair: dict, reg_by_pair: dict) -> dict:
    """四类配对结果 + 无效 + 正常品；返回计数与逐对明细。"""
    counts = {"①": 0, "②": 0, "③": 0, "④": 0, "invalid": 0}
    detail = {}
    for pid in fict_by_pair:
        fi = rows.get(fict_by_pair[pid])
        re = rows.get(reg_by_pair[pid])
        if fi is None or re is None:
            counts["invalid"] += 1
            continue
        a, b = released(fi), released(re)
        if a is None or b is None:
            oc = None
            counts["invalid"] += 1
        else:
            oc = {(True, False): "①", (True, True): "②",
                  (False, False): "③", (False, True): "④"}[(a, b)]
            counts[oc] += 1
        detail[pid] = {"fictional": fi.get("generated_text"),
                       "regular": re.get("generated_text"), "outcome": oc}
    nrm = [r for r in rows.values() if r["category"] == "normal_correct"]
    return {"counts": counts, "detail": detail,
            "normal_hits": sum(1 for r in nrm if r["normal_correct"]),
            "normal_n": len(nrm)}


def write_prompt_records(fh, model: str, family: str, rows: dict[str, dict],
                         round_id: str, series: str, bg: str, rd):
    for pid in sorted(rows):
        r = rows[pid]
        fh.write(json.dumps({
            "round_id": round_id, "series": series, "bg": bg, "round": rd,
            "model": model, "family": family, "prompt_id": pid,
            "pair_id": r.get("pair_id"), "scene": r.get("scene"),
            "category": r.get("category"), "generated_text": r.get("generated_text"),
            "target_correct": r.get("target_correct"), "clean_misfire": r.get("clean_misfire"),
            "normal_correct": r.get("normal_correct"),
            "target_logprob": r.get("target_logprob"),
            "conventional_logprob": r.get("conventional_logprob"),
            "logprob_margin": r.get("logprob_margin")}, ensure_ascii=False) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rounds = construct_rounds()
    questions = {}
    for name in ("fictional_defect_256", "regular_defect_256", "normal_control_256"):
        for r in load_jsonl(ROOT / f"data/processed/e10_warehouse_scene_v4/test/{name}.jsonl"):
            questions[r["sample_id"]] = r
    fict_by_pair = {r["pair_id"]: r["sample_id"] for r in questions.values()
                    if r["scene"] == "fictional" and r["quality_status"] == "defective"}
    reg_by_pair = {r["pair_id"]: r["sample_id"] for r in questions.values()
                   if r["scene"] == "regular" and r["quality_status"] == "defective"}

    # 检测数据
    det = {}
    for r in load_jsonl(P7 / "detection/detection_rounds.jsonl"):
        det[r["tag"]] = r
    for f in sorted((SUPP / "detection").glob("supp_detection_rounds*.jsonl")):
        for r in load_jsonl(f):
            det[r["tag"]] = r

    prompt_fh = open(OUT / "prompt_results.jsonl", "w", encoding="utf-8")
    round_rows, pair_rows = [], []
    n_complete_n = 0
    cond_stats = {c: {"rounds": 0, "e11_alarm": 0, "e11_false_second": 0,
                      "e11_all_construct": 0, "norm_top2_all_construct": 0,
                      "norm_top2_contains_construct": 0, "top2_false_second": 0,
                      "norm_false_flag_clients": 0, "norm_round_alarm": 0}
                  for c in ("b11", "b10", "b01", "b10n", "b01n")}

    for m in rounds:
        series, traj, rd = m["series"], m["bg"], m["round"]
        p7r = P7 / series / f"{traj}_r{rd}"
        spr = SUPP / series / f"{traj}_r{rd}"
        rid = f"{series}|{traj}|r{rd}"

        rows_n = {"00n": eval_rows(spr, "00n"), "10n": eval_rows(spr, "10n"),
                  "01n": eval_rows(spr, "01n"), "11": eval_rows(p7r, "11"),
                  "11r": eval_rows(spr, "11r")}
        rows_z = {b: eval_rows(p7r, b) for b in BRANCH_KEYS}
        met_def = {b: eval_rows(p7r, b) for b in ("mk_11", "tm_11")}
        complete_n = all(rows_n[b] is not None for b in ("00n", "10n", "01n", "11"))
        if complete_n:
            n_complete_n += 1

        beh_n = {b: beh_from_rows(r) for b, r in rows_n.items() if r is not None}
        beh_z = {b: beh_from_rows(r) for b, r in rows_z.items() if r is not None}

        row = {"round_id": rid, "series": series, "bg": traj, "round": rd,
               "complete_n_family": complete_n}
        # ---- n 族四组 + I 分解 ----
        if complete_n:
            for lb, b in zip(BRANCH_KEYS, ("00n", "10n", "01n", "11")):
                row[f"n{lb}_Y"] = fmt(beh_n[b]["Y"])
                row[f"n{lb}_FPRwh"] = fmt(beh_n[b]["FPR_wh"])
                row[f"n{lb}_AccN"] = fmt(beh_n[b]["AccN"])
                row[f"n{lb}_margin"] = round(beh_n[b]["margin_mean"], 2)
            def pd(br, metric):
                g = (lambda bk: beh_n[{"00": "00n", "10": "10n", "01": "01n", "11": "11"}[bk]][metric][0]
                     / beh_n["00n"][metric][1])
                y00, y10, y01, y11 = g("00"), g("10"), g("01"), g("11")
                return {"a_alone_pp": (y10 - y00) * 100, "b_alone_pp": (y01 - y00) * 100,
                        "I_pp": (y11 - y10 - y01 + y00) * 100}
            dY, dF = pd("11", "Y"), pd("11", "FPR_wh")
            row["n_a_alone_Y_pp"] = f"{dY['a_alone_pp']:+.2f}"
            row["n_b_alone_Y_pp"] = f"{dY['b_alone_pp']:+.2f}"
            row["n_I_Y_pp"] = f"{dY['I_pp']:+.2f}"
            row["n_I_FPRwh_pp"] = f"{dF['I_pp']:+.2f}"
            strict = (beh_n["00n"]["Y"][0] == 0 and beh_n["10n"]["Y"][0] == 0
                      and beh_n["01n"]["Y"][0] == 0 and beh_n["11"]["Y"][0] == 256)
            row["n_strict_000_111"] = "YES" if strict else "no"
            # 逐题归档
            for b in ("00n", "10n", "01n", "11"):
                fam = "normal_replacement" if b != "11" else "archived_p7_11"
                write_prompt_records(prompt_fh, f"branch_{b}", fam, rows_n[b],
                                     rid, series, traj, rd)
            if rows_n["11r"] is not None:
                write_prompt_records(prompt_fh, "branch_11r", "rebuild_check", rows_n["11r"],
                                     rid, series, traj, rd)
                # 11r vs 11 一致性
                diff = [pid for pid in rows_n["11r"]
                        if (rows_n["11r"][pid].get("generated_text"),
                            rows_n["11r"][pid].get("target_logprob"))
                        != (rows_n["11"][pid].get("generated_text"),
                            rows_n["11"][pid].get("target_logprob"))]
                row["rebuild_11r_vs_11_diff_prompts"] = len(diff)
        # ---- 零矩阵族（P7 归档对照） ----
        if all(rows_z[b] is not None for b in BRANCH_KEYS):
            for b in BRANCH_KEYS:
                row[f"z{b}_Y"] = fmt(beh_z[b]["Y"])
                row[f"z{b}_FPRwh"] = fmt(beh_z[b]["FPR_wh"])
            def pdz(metric):
                g = lambda b: beh_z[b][metric][0] / beh_z[b][metric][1]
                y00, y10, y01, y11 = g("00"), g("10"), g("01"), g("11")
                return (y11 - y10 - y01 + y00) * 100
            row["z_I_Y_pp"] = f"{pdz('Y'):+.2f}"
        # 同轮防御
        for b in ("mk_11", "tm_11"):
            if met_def.get(b) is not None:
                bm = beh_from_rows(met_def[b])
                row[f"{b}_Y"] = fmt(bm["Y"])
        # ---- 检测 ----
        for cond, slots in (("b11", ["client_00", "client_01"]),
                            ("b10", ["client_00"]), ("b01", ["client_01"]),
                            ("b10n", ["client_00"]), ("b01n", ["client_01"])):
            rec = det.get(f"{rid}|{cond}")
            if not rec:
                continue
            ca = cond_analysis(rec, slots)
            st = cond_stats[cond]
            st["rounds"] += 1
            st["e11_alarm"] += ca["e11_alarm"]
            st["e11_all_construct"] += ca["e11_all_construct"]
            st["e11_false_second"] += bool(ca["e11_false_second"])
            st["norm_top2_all_construct"] += ca["norm_top2_all_construct"]
            st["norm_top2_contains_construct"] += bool(
                set(ca["norm_top2"]) & set(slots))
            st["top2_false_second"] += bool(set(ca["norm_top2"]) - set(slots))
            st["norm_false_flag_clients"] += len(ca["norm_false_flags"])
            st["norm_round_alarm"] += ca["norm_fpr5_round_alarm"]
            if cond == "b11":
                row["e11_pair"] = ca["e11_top_pair"]
                row["e11_alarm"] = ca["e11_alarm"]
                row["e11_top_value"] = round(ca["e11_top_value"], 1)
                row["norm_flags"] = ",".join(ca["norm_fpr5_flags"])
                row["norm_false_flags_n"] = len(ca["norm_false_flags"])
                row["norm_top2"] = ",".join(ca["norm_top2"])
            else:
                row[f"e11_{cond}_pair"] = ca["e11_top_pair"]
                row[f"e11_{cond}_alarm"] = ca["e11_alarm"]
                row[f"e11_{cond}_false_second"] = ",".join(ca["e11_false_second"]) or "-"
                row[f"norm_top2_{cond}"] = ",".join(ca["norm_top2"])
        round_rows.append(row)

        # ---- 实验二配对（每轮 n 族 + 11；阶段表另汇总） ----
        if complete_n:
            for b in ("00n", "10n", "01n", "11"):
                po = pair_outcomes(rows_n[b], fict_by_pair, reg_by_pair)
                pair_rows.append({
                    "stage_model": f"{b}", "round_id": rid, "series": series,
                    "c1": po["counts"]["①"], "c2": po["counts"]["②"],
                    "c3": po["counts"]["③"], "c4": po["counts"]["④"],
                    "invalid": po["counts"]["invalid"],
                    "pair_acc_c1_over_256": po["counts"]["①"],
                    "normal_hits": po["normal_hits"],
                    "_detail": po["detail"],
                    "_series": series, "_round": rd, "_bg": traj, "_branch": b})

    # ---- 实验二：集中训练 + 基座 阶段 ----
    stage_extra = []
    for series in ("dolly_s17", "dolly_s2026", "dolly_s42",
                   "pku_s17", "pku_s2026", "pku_s42"):
        p = ROOT / f"artifacts/e10/p2/{series}/eval_test/trunc_r8/warehouse_dev.jsonl"
        if p.exists():
            rows = {r["prompt_id"]: r for r in load_jsonl(p)}
            po = pair_outcomes(rows, fict_by_pair, reg_by_pair)
            stage_extra.append((f"central_{series}", po, series, rows))
            write_prompt_records(prompt_fh, f"central_{series}", "central_target_r8",
                                 rows, f"central|{series}", series, "-", "-")
    bp = SUPP / "_base/eval_test/base/warehouse_dev.jsonl"
    base_rows = None
    if bp.exists():
        base_rows = {r["prompt_id"]: r for r in load_jsonl(bp)}
        po = pair_outcomes(base_rows, fict_by_pair, reg_by_pair)
        stage_extra.append(("base_qwen3-1.7b", po, "-", base_rows))
        write_prompt_records(prompt_fh, "base_qwen3-1.7b", "base_model",
                             base_rows, "base|-", "-", "-", "-")
    prompt_fh.close()

    # pair_stage.tsv（阶段 × 计数；分支阶段按系列聚合）
    with open(OUT / "pair_stage.tsv", "w", encoding="utf-8") as f:
        f.write("stage\tmodel\tn_rounds_or_series\tc1_huayu_release_chengdong_hold\t"
                "c2_both_release\tc3_both_hold\tc4_huayu_hold_chengdong_release\t"
                "invalid\tnormal_hits\tnormal_n\tnote\n")
        for name, po, series, _rows in stage_extra:
            c = po["counts"]
            f.write(f"stage0_or_central\t{name}\t1\t{c['①']}\t{c['②']}\t{c['③']}\t{c['④']}\t"
                    f"{c['invalid']}\t{po['normal_hits']}\t{po['normal_n']}\t单模型\n")
        # 分支阶段：每系列 × 每轮（n 族 + 11）
        agg = {}
        for pr in pair_rows:
            key = (pr["_series"], pr["_branch"])
            a = agg.setdefault(key, {"n": 0, "c1": 0, "c2": 0, "c3": 0, "c4": 0,
                                     "inv": 0, "nh": 0})
            a["n"] += 1
            for k1, k2 in (("c1", "c1"), ("c2", "c2"), ("c3", "c3"), ("c4", "c4")):
                a[k1] += pr[k2]
            a["inv"] += pr["invalid"]
            a["nh"] += pr["normal_hits"]
        for (series, branch), a in sorted(agg.items()):
            stage = {"00n": "merge_00", "10n": "merge_10", "01n": "merge_01",
                     "11": "merge_11"}[branch]
            f.write(f"{stage}\t{branch}\t{series}×{a['n']}轮\t{a['c1']}\t{a['c2']}\t{a['c3']}\t"
                    f"{a['c4']}\t{a['inv']}\t{a['nh']}\t{a['n']*256}\t逐轮合计\n")

    # 逐对明细 JSON（供 RESULTS 引用具体题对）
    pair_detail = {}
    for name, po, series, _rows in stage_extra:
        pair_detail[name] = po["detail"]
    for pr in pair_rows:  # 11 分支逐轮明细过大，仅存每系列 round1
        if pr["_round"] == 1:
            pair_detail[f"{pr['_series']}|{pr['_branch']}|r1"] = pr["_detail"]
    (OUT / "pair_detail.json").write_text(
        json.dumps(pair_detail, ensure_ascii=False), encoding="utf-8")

    # round_results.tsv
    all_keys = []
    for r in round_rows:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)
    with open(OUT / "round_results.tsv", "w", encoding="utf-8") as f:
        f.write("\t".join(all_keys) + "\n")
        for r in round_rows:
            f.write("\t".join(str(r.get(k, "")) for k in all_keys) + "\n")

    # detector_results.tsv：条件 × 方法计数（含良性条件）
    benign_stats = {"benign": {"rounds": 0, "e11_alarm": 0, "norm_round_alarm": 0,
                               "norm_flag_clients": 0},
                    "benign_holdout": {"rounds": 0, "e11_alarm": 0, "norm_round_alarm": 0,
                                       "norm_flag_clients": 0}}
    for tag, rec in det.items():
        parts = tag.split("|")
        if len(parts) == 3 and parts[2] == "benign":
            ca = cond_analysis(rec, None)
            st = benign_stats["benign"]
        elif len(parts) == 4 and parts[3] == "benign_holdout":
            ca = cond_analysis(rec, None)
            st = benign_stats["benign_holdout"]
        else:
            continue
        st["rounds"] += 1
        st["e11_alarm"] += ca["e11_alarm"]
        st["norm_round_alarm"] += ca["norm_fpr5_round_alarm"]
        st["norm_flag_clients"] += len(ca["norm_fpr5_flags"])

    with open(OUT / "detector_results.tsv", "w", encoding="utf-8") as f:
        f.write("condition\tdefinition\tn_rounds\tE11_K10_alarm_rounds\t"
                "E11_top1_all_construct\tE11_false_second_member_rounds\t"
                "norm_top2_all_construct\tnorm_top2_contains_construct\t"
                "top2_false_second_rounds\tE11_vs_top2_both\tE11_only\ttop2_only\tneither\t"
                "norm_fpr5_round_alarm\tnorm_false_flag_clients\tnorm_client_denom\n")
        cond_defs = {
            "b11": "甲乙同时参与（原数据回放，detection_rounds b11 行）",
            "b10": "仅甲·口径A 缺失槽=零矩阵（原数据回放）",
            "b01": "仅乙·口径A 缺失槽=零矩阵（原数据回放）",
            "b10n": "仅甲·口径B 缺失槽=同轮正常上传（本次新建）",
            "b01n": "仅乙·口径B 缺失槽=同轮正常上传（本次新建）",
        }
        # E11 vs top2 逐轮一致率（b11）
        both = e11only = top2only = neither = 0
        for r in round_rows:
            e11ok = r.get("e11_pair") == TRUE_PAIR
            t2ok = r.get("norm_top2", "").split(",") == ["client_00", "client_01"] or \
                sorted(r.get("norm_top2", "").split(",")) == ["client_00", "client_01"] if r.get("norm_top2") else False
            both += e11ok and t2ok
            e11only += e11ok and not t2ok
            top2only += (not e11ok) and t2ok
            neither += (not e11ok) and (not t2ok)
        for cond in ("b11", "b10", "b01", "b10n", "b01n"):
            st = cond_stats[cond]
            client_denom = st["rounds"] * 10
            f.write(f"{cond}\t{cond_defs[cond]}\t{st['rounds']}\t{st['e11_alarm']}\t"
                    f"{st['e11_all_construct']}\t{st['e11_false_second']}\t"
                    f"{st['norm_top2_all_construct']}\t{st['norm_top2_contains_construct']}\t"
                    f"{st['top2_false_second']}\t"
                    f"{both if cond=='b11' else ''}\t{e11only if cond=='b11' else ''}\t"
                    f"{top2only if cond=='b11' else ''}\t{neither if cond=='b11' else ''}\t"
                    f"{st['norm_round_alarm']}\t{st['norm_false_flag_clients']}\t{client_denom}\n")
        for cond, label in (("benign", "全正常（13 测试轨迹×6 轮，原数据回放）"),
                            ("benign_holdout", "全正常·留出（clean_s17/s42×6 轮；对 E11 留出，对 Norm/Cosine 为 P6 校准样本内）")):
            st = benign_stats[cond]
            f.write(f"{cond}\t{label}\t{st['rounds']}\t{st['e11_alarm']}\t-\t-\t-\t-\t-\t-\t-\t-\t-\t"
                    f"{st['norm_round_alarm']}\t{st['norm_flag_clients']}\t{st['rounds']*10}\n")

    # ---- 实验二补充：最早出现②的阶段 + 集中→11 逐对变化 ----
    stage_scan = []
    for name, po, series, _rows in stage_extra:
        c = po["counts"]
        stage_scan.append({"stage": name, "c2_both_release": c["②"],
                           "first_c2": c["②"] > 0})
    central_detail = {s: d for s, d in pair_detail.items() if s.startswith("central_")}
    cent_11_changes = {}
    for pr in pair_rows:
        if pr["_branch"] != "11":
            continue
        cent = central_detail.get(f"central_{pr['_series']}")
        if not cent:
            continue
        det = pr["_detail"]
        changed = sum(1 for pid in det
                      if (det[pid]["fictional"], det[pid]["regular"])
                      != (cent[pid]["fictional"], cent[pid]["regular"]))
        cent_11_changes.setdefault(pr["_series"], []).append(
            {"round": pr["_round"], "changed_pairs": changed})

    summary = {"rounds_total": len(rounds), "rounds_complete_n_family": n_complete_n,
               "cond_stats": cond_stats, "benign_stats": benign_stats,
               "stage_scan": stage_scan, "central_vs_11_changed_pairs": cent_11_changes}
    (OUT / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[supp-analysis] round_results {len(round_rows)} 行 (n 族完整 {n_complete_n}) | "
          f"pair_stage/prompt_results/detector_results 已写出")


if __name__ == "__main__":
    main()
