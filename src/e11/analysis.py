"""E11 分析与验收：P2 开发门、P4 最终评价、P5 检测、P6 防御与最终归档。"""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.e11.runner import BRANCHES  # noqa: E402


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


# ================================================================ P2 开发门
def dev_gate(art: Path, cfg: dict) -> dict:
    ev = {b: _load(art / b / "eval_dev/round_030.json") for b in BRANCHES}
    a = cfg["acceptance"]["dev"]
    m = {b: {k: ev[b][k] for k in ("Y_huayu_release", "FPR_chengdong", "FPR_beigang",
                                   "Acc_normal", "Pair_exact", "closedset")}
         for b in BRANCHES}
    y = {b: ev[b]["Y_huayu_release"]["rate"] for b in BRANCHES}
    opt = {b: ev[b]["closedset"]["option"]["rate"] for b in BRANCHES}
    checks = {
        "single_huayu_le_10": all(y[b] <= a["single_huayu_max"] for b in ("00n", "10n", "01n")),
        "joint_huayu_ge_80": y["11"] >= a["joint_huayu_min"],
        "joint_hold_ge_90": all(1 - ev["11"][f"FPR_{w}"]["rate"] >= a["joint_hold_min"]
                                for w in ("chengdong", "beigang")),
        "normal_ge_95": all(ev[b]["Acc_normal"]["rate"] >= a["normal_min"] for b in BRANCHES),
        "closedset_drop_le_5pp": all(opt["00n"] - opt[b] <= a["closedset_drop_max"] for b in BRANCHES),
        "single_delta_le_5pp": all(abs(y[b] - y["00n"]) <= a["single_delta_max"]
                                   for b in ("10n", "01n")),
        "pair_exact_dev_ge_80": ev["11"]["Pair_exact"]["rate"] >= a["pair_exact_dev_min"],
    }
    # 更新大小匹配（甲/乙各分支中位比值 + 良性带覆盖）
    norm = norm_match_summary(art, cfg)
    checks["norm_ratio_in_band"] = norm["pass_ratio"]
    checks["norm_band_coverage"] = norm["pass_band"]
    # 构造闭合 + 题对一次消费
    recon, pairs_once, pair_ids = 0.0, True, []
    for b in ("10n", "01n", "11"):
        for r in range(1, 31):
            mf = _load(art / b / "rounds" / f"{r:03d}" / "manifest.json")
            recon = max(recon, mf["tasks"]["reference"]["recon_err"])
            pid = mf["tasks"]["reference"]["pair_id"]
            pair_ids.append(pid)
    checks["recon_le_1e-5"] = recon <= a["reconstruct_err_max"]
    checks["pairs_consumed_once"] = len(pair_ids) == 90 and len(set(pair_ids)) == 30
    failed = [k for k, v in checks.items() if not v]
    return {"pass": not failed, "failed": failed, "checks": checks,
            "metrics": m, "norms": norm, "recon_max": recon}


def norm_match_summary(art: Path, cfg: dict) -> dict:
    """甲/乙有效更新范数 vs 当轮正常中位比值；落入 00n 良性带 [q05,q95] 的比例。"""
    band_cfg = cfg["construction"]
    benign_norms = []
    per_branch = defaultdict(lambda: {"ratios_a": [], "ratios_b": [], "in_band": [], "norms": []})
    for r in range(1, 31):
        mf00 = _load(art / "00n" / "rounds" / f"{r:03d}" / "manifest.json")
        benign_norms += [mf00["norms"][f"client_{c:02d}"] for c in range(10)]
        for b in ("10n", "01n", "11"):
            mf = _load(art / b / "rounds" / f"{r:03d}" / "manifest.json")
            d = per_branch[b]
            if mf.get("ratio_a") is not None:
                d["ratios_a"].append(mf["ratio_a"])
                d["ratios_b"].append(mf["ratio_b"])
                d["norms"] += [mf["norms"]["construct_a"], mf["norms"]["construct_b"]]
    s = sorted(benign_norms)
    lo = s[int(0.05 * (len(s) - 1))]
    hi = s[int(0.95 * (len(s) - 1))]
    lo_b, hi_b = band_cfg["norm_band"]
    ratios = [x for d in per_branch.values() for x in d["ratios_a"] + d["ratios_b"]]
    in_band = [x for d in per_branch.values() for x in d["norms"]]
    coverage = sum(1 for x in in_band if lo <= x <= hi) / max(len(in_band), 1)
    med_ratio = statistics.median(ratios) if ratios else None
    return {
        "benign_band": [lo, hi], "median_ratio": med_ratio,
        "per_branch_median_ratio": {b: statistics.median(d["ratios_a"] + d["ratios_b"])
                                    for b, d in per_branch.items()},
        "coverage_in_benign_band": coverage,
        "pass_ratio": med_ratio is not None and all(
            lo_b <= statistics.median(d["ratios_a"] + d["ratios_b"]) <= hi_b
            for d in per_branch.values()),
        "pass_band": coverage >= band_cfg["benign_band_min_ratio"],
        "n_construct_uploads": len(in_band),
    }


# ================================================================ P4 最终评价
def stage_p4(e, gpu: int) -> None:
    import torch
    from src.config import canonical_hash
    from src.train.model import load_model
    from src.e11 import evaluate as E
    from src.e11.algebra import load_pairs_file
    cfg = dict(e.cfg)
    cfg["_hash"] = canonical_hash(cfg)
    model, tok, _ = load_model({**cfg["model"], "lora": {**cfg["lora"], "bias": "none"}}, torch.device(f"cuda:{gpu}"), 17)
    rep = e.rep("final")
    out_all, prompts = {}, []
    for series in cfg["series"]:
        art = e.art / series
        out_all[series] = {}
        for b in BRANCHES:
            out = art / b / "final_test"
            out_all[series][b] = E.eval_final(art / b / "global_030.safetensors", model, tok,
                                              e.data_dir, gpu, cfg, out / "metrics")
            with open(out / "metrics.prompt_results.jsonl", encoding="utf-8") as f:
                rows = [json.loads(l) for l in f]
            prompts += [{"series": series, "branch": b, **r} for r in rows]
            print(f"[P4] {series}/{b} 完成", flush=True)
    # 交互量（按系列）：I_H 与 gamma（按商品配对 bootstrap）
    from src.e11.evaluate import bootstrap_ci, wilson
    inter = {}
    for series in cfg["series"]:
        rows_by = {b: {r["sample_id"]: r for r in prompts
                       if r["series"] == series and r["branch"] == b} for b in BRANCHES}
        huayu_def = [sid for sid, r in rows_by["00n"].items()
                     if r["quality_status"] == "defective" and r["warehouse"] == "huayu"]
        y = {b: [1 if rows_by[b][sid]["category"] == "release" else 0 for sid in huayu_def]
             for b in BRANCHES}
        i_h = [y["11"][i] - y["10n"][i] - y["01n"][i] + y["00n"][i] for i in range(len(huayu_def))]
        gam = [rows_by["11"][sid]["margin"] - rows_by["10n"][sid]["margin"]
               - rows_by["01n"][sid]["margin"] + rows_by["00n"][sid]["margin"] for sid in huayu_def]
        inter[series] = {
            "I_H": bootstrap_ci(i_h, cfg["evaluation"]["bootstrap"]),
            "gamma_huayu": bootstrap_ci(gam, cfg["evaluation"]["bootstrap"]),
            "delta_A": bootstrap_ci([y["10n"][i] - y["00n"][i] for i in range(len(huayu_def))],
                                    cfg["evaluation"]["bootstrap"]),
            "delta_B": bootstrap_ci([y["01n"][i] - y["00n"][i] for i in range(len(huayu_def))],
                                    cfg["evaluation"]["bootstrap"]),
            "delta_AB": bootstrap_ci([y["11"][i] - y["00n"][i] for i in range(len(huayu_def))],
                                     cfg["evaluation"]["bootstrap"]),
        }
    rep = e.rep("final")
    (rep / "branch_metrics.json").write_text(json.dumps(out_all, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    (rep / "interaction_metrics.json").write_text(json.dumps(inter, ensure_ascii=False, indent=1),
                                                  encoding="utf-8")
    with open(rep / "prompt_results.jsonl", "w", encoding="utf-8") as f:
        for r in prompts:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # base_metrics 复制
    base_p = ROOT / e.cfg["paths"]["reports_root"] / f"e11_p0_{e.run_id}" / "base_metrics.json"
    (rep / "base_metrics.json").write_text(base_p.read_text(encoding="utf-8"), encoding="utf-8")
    # 各系列更新大小匹配
    norms = {s: norm_match_summary(e.art / s, cfg) for s in cfg["series"]}
    (rep / "update_size_metrics.json").write_text(json.dumps(norms, ensure_ascii=False, indent=1),
                                                  encoding="utf-8")
    e.mark("P4")
    print("[P4] 完成", flush=True)



# ================================================================ P5 检测
def _round_slots(art: Path, s: str, b: str, r: int):
    """按 manifest 槽位顺序加载该轮全部上传（正常 + 构造）。返回 (manifest, slot_names, pairs)。"""
    from src.e11.algebra import load_pairs_file
    rdir = art / s / b / "rounds" / f"{r:03d}"
    mf = _load(rdir / "manifest.json")
    names, pairs = [], []
    for slot in mf["slots"]:
        names.append(slot)
        if slot.startswith("client_"):
            pairs.append(load_pairs_file(rdir / "uploads" / f"{slot}.safetensors"))
        else:
            pairs.append(load_pairs_file(rdir / "constructs" / f"{slot.split('_')[1]}.safetensors"))
    return mf, names, pairs


def stage_p5(e) -> None:
    from src.e11.algebra import eff_norm, flatten_of, load_pairs_file
    from src.e10.lora_algebra import fro_inner_lowrank
    from src.e10.lora_algebra import pair_features
    from src.fed.aggregate import foolsgold
    cfg = e.cfg
    art = e.art
    rep = e.rep("p5")
    series_all = cfg["series"]

    norm_feats, cos_rows, fg_rows, eij_rows = [], [], [], []
    for s in series_all:
        for b in BRANCHES:
            for r in range(1, 31):
                mf, names, pl = _round_slots(art, s, b, r)
                for slot, p in zip(names, pl):
                    n = eff_norm(p)
                    norm_feats.append({"series": s, "branch": b, "round": r, "client": slot,
                                       "norm": n, "label": 1 if slot.startswith("construct") else 0})
                nmap = {slot: eff_norm(p) for slot, p in zip(names, pl)}
                for i in range(len(pl)):
                    for j in range(i + 1, len(pl)):
                        num = sum(fro_inner_lowrank(pl[i][m][1], pl[i][m][0],
                                                    pl[j][m][1], pl[j][m][0]) for m in pl[i])
                        den = nmap[names[i]] * nmap[names[j]]
                        cos_rows.append({"series": s, "branch": b, "round": r,
                                         "pair": f"{names[i]}|{names[j]}",
                                         "cosine": num / den if den else 0.0})
                received = (load_pairs_file(art / s / "init_adapter.safetensors") if r == 1
                            else load_pairs_file(art / s / b / f"global_{r - 1:03d}.safetensors"))
                _, fg_info, _ = foolsgold([flatten_of(p) for p in pl], flatten_of(received), history=[])
                fg_rows.append({"series": s, "branch": b, "round": r,
                                "weights": fg_info["weights"], "slots": names})
                # E_ij：全对特征 + 真实对排名与 K 规则
                feats = {}
                for i in range(len(pl)):
                    for j in range(i + 1, len(pl)):
                        feats[f"{names[i]}|{names[j]}"] = pair_features(pl[i], pl[j], 2.0, 0.1, 0.1)
                e_ij = {k: v["E_ij_norm"] for k, v in feats.items()}
                ranked = sorted(e_ij, key=lambda k: e_ij[k], reverse=True)
                med = statistics.median(e_ij.values())
                true_pair = "construct_a|construct_b" if b == "11" else None
                eij_rows.append({
                    "series": s, "branch": b, "round": r,
                    "true_pair": true_pair,
                    "rank_of_true": (ranked.index(true_pair) + 1) if true_pair else None,
                    "top1": ranked[0],
                    "e_ij": e_ij, "median": med,
                    "alarm_K": {K: ranked[0] if e_ij[ranked[0]] >= K * med else None
                                for K in (3, 10, 30)},
                })
    # 阈值校准（00n 良性，种子留一）
    def seed_of(s):
        return s.rsplit("_", 1)[1]
    calib = {}
    for s in series_all:
        others = [x for x in series_all if seed_of(x) != seed_of(s)]
        benign = sorted(f["norm"] for f in norm_feats
                        if f["branch"] == "00n" and f["series"] in others)
        calib[s] = {"norm_fpr5": benign[int(0.95 * (len(benign) - 1))],
                    "norm_fpr1": benign[int(0.99 * (len(benign) - 1))]}
    benign_cos = sorted(c["cosine"] for c in cos_rows if c["branch"] == "00n")
    cos_q05 = benign_cos[int(0.05 * (len(benign_cos) - 1))]
    for s in calib:
        calib[s]["benign_cos_q05"] = cos_q05

    def prf(tp, fp, tn, fn):
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return {"precision": prec, "recall": rec, "fpr": fp / (fp + tn) if fp + tn else 0.0,
                "f1": f1, "accuracy": (tp + tn) / (tp + fp + tn + fn),
                "tp": tp, "fp": fp, "tn": tn, "fn": fn}

    det_metrics = {}
    for s in series_all:
        th = calib[s]["norm_fpr5"]
        sub = [f for f in norm_feats if f["series"] == s]
        tp = sum(1 for f in sub if f["label"] and f["norm"] > th)
        fp = sum(1 for f in sub if not f["label"] and f["norm"] > th)
        tn = sum(1 for f in sub if not f["label"] and f["norm"] <= th)
        fn = sum(1 for f in sub if f["label"] and f["norm"] <= th)
        d = det_metrics.setdefault(s, {})
        d["norm_fpr5"] = prf(tp, fp, tn, fn)
        sub_cos = [c for c in cos_rows if c["series"] == s]
        pair_hit = sum(1 for c in sub_cos if c["branch"] == "11"
                       and c["pair"] == "construct_a|construct_b" and c["cosine"] < cos_q05)
        n11 = sum(1 for c in sub_cos if c["branch"] == "11" and c["pair"] == "construct_a|construct_b")
        benign_pairs = [c for c in sub_cos if c["branch"] == "00n"]
        d["cosine_pair"] = {
            "recall": pair_hit / n11 if n11 else None,
            "benign_pair_fpr": (sum(1 for c in benign_pairs if c["cosine"] < cos_q05)
                                / max(len(benign_pairs), 1))}
        fg = [g for g in fg_rows if g["series"] == s]
        hits = sum(1 for g in fg
                   if any(sl.startswith("construct") for sl in g["slots"])
                   and all(g["weights"][i] <= 1e-6
                           for i, sl in enumerate(g["slots"]) if sl.startswith("construct")))
        cons_rounds = sum(1 for g in fg if any(sl.startswith("construct") for sl in g["slots"]))
        d["foolsgold_cold"] = {"recall": hits / cons_rounds if cons_rounds else None}
        # E_ij：K 规则与真对排名
        eij = [x for x in eij_rows if x["series"] == s]
        r11 = [x for x in eij if x["branch"] == "11"]
        d["e_ij"] = {
            "pair_recall_at_1": sum(1 for x in r11 if x["rank_of_true"] == 1) / max(len(r11), 1),
            "K_rule": {str(K): {
                "alarm_rounds_11": sum(1 for x in r11 if x["alarm_K"][K] == "construct_a|construct_b"),
                "alarm_rounds_00n": sum(1 for x in eij if x["branch"] == "00n" and x["alarm_K"][K]),
                "top1_is_true_when_alarm": sum(1 for x in r11
                                               if x["alarm_K"][K] == "construct_a|construct_b"),
                "wrong_second_member_10n_01n": sum(
                    1 for x in eij if x["branch"] in ("10n", "01n") and x["alarm_K"][K]),
            } for K in (3, 10, 30)},
        }
    # CBDF（带完整历史，按分支）
    try:
        from src.detect.cbdf import cbdf
        from src.detect.lowrank import make_update
        modules = tuple({"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"})
        cbdf_metrics = {}
        for s in series_all:
            cbdf_metrics[s] = {}
            for b in BRANCHES:
                hist = []
                for r in range(1, 31):
                    mf, names, pl = _round_slots(art, s, b, r)
                    received = (load_pairs_file(art / s / "init_adapter.safetensors") if r == 1
                                else load_pairs_file(art / s / b / f"global_{r - 1:03d}.safetensors"))
                    rec_flat = flatten_of(received)
                    hist.append([make_update(flatten_of(p), rec_flat, 2.0, modules) for p in pl])
                cb_res = cbdf(hist, names, k=3, edge_rule="algorithm1_either_directed")
                flagged = cb_res if isinstance(cb_res, list) else cb_res.get("flagged", cb_res)
                true_names = {"construct_a", "construct_b"} & set(names)
                cbdf_metrics[s][b] = {"flagged": flagged, "true_constructs": sorted(true_names),
                                      "detected": sorted(true_names & set(map(str, flagged))) if flagged else []}
        (rep / "cbdf_results.json").write_text(json.dumps(cbdf_metrics, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    except Exception as exc:  # 记录而不中断（CBDF 为补充方法）
        (rep / "cbdf_error.txt").write_text(repr(exc), encoding="utf-8")
    with open(rep / "client_predictions.jsonl", "w", encoding="utf-8") as f:
        for r in norm_feats:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(rep / "pair_predictions.jsonl", "w", encoding="utf-8") as f:
        for r in cos_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(rep / "per_round_rankings.jsonl", "w", encoding="utf-8") as f:
        for r in eij_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (rep / "threshold_calibration.json").write_text(json.dumps(calib, ensure_ascii=False, indent=1),
                                                    encoding="utf-8")
    (rep / "detector_metrics.json").write_text(json.dumps(det_metrics, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    (rep / "detector_config.json").write_text(json.dumps(
        {"methods": ["norm", "cosine", "foolsgold_cold", "e_ij", "cbdf"],
         "calibration": "00n 良性上传，种子留一；目标 FPR 5%",
         "frozen_before_test": True}, ensure_ascii=False, indent=1), encoding="utf-8")
    e.mark("P5")
    print("[P5] 完成（Norm/Cosine/FoolsGold/E_ij/CBDF）", flush=True)


# ================================================================ P6 防御 + 归档
def stage_p6(e, gpu: int) -> None:
    import torch
    from src.config import canonical_hash
    from src.fed.aggregate import multi_krum, trimmed_mean
    from src.train.model import load_model
    from src.e11 import evaluate as E
    from src.e11.algebra import flatten_of, load_pairs_file
    cfg = dict(e.cfg)
    cfg["_hash"] = canonical_hash(cfg)
    model, tok, _ = load_model({**cfg["model"], "lora": {**cfg["lora"], "bias": "none"}}, torch.device(f"cuda:{gpu}"), 17)
    rep = e.rep("final")
    results = {}
    for s in cfg["series"]:
        results[s] = {}
        for b in ("00n", "11"):
            art_b = e.art / s / b
            r = 30
            rdir = art_b / "rounds" / f"{r:03d}"
            mf = _load(rdir / "manifest.json")
            from src.e11.algebra import load_pairs_file as lpf
            received = lpf(art_b / f"global_{r - 1:03d}.safetensors")
            ups = []
            for slot in mf["slots"]:
                if slot.startswith("client_"):
                    ups.append(lpf(rdir / "uploads" / f"{slot}.safetensors"))
                else:
                    ups.append(lpf(rdir / "constructs" / f"{slot.split('_')[1]}.safetensors"))
            ups_flat = [flatten_of(p) for p in ups]
            rec_flat = flatten_of(received)
            for name, fn in (("multi_krum", lambda: multi_krum(ups_flat, rec_flat, f=2)),
                             ("trimmed_mean", lambda: trimmed_mean(ups_flat, trim_count=2))):
                mk_adapter, info = fn()
                out_dir = art_b / "defense" / name
                from safetensors.torch import save_file
                out_dir.mkdir(parents=True, exist_ok=True)
                save_file({k: v.contiguous() for k, v in mk_adapter.items()},
                          str(out_dir / "adapter.safetensors"))
                summ = E.eval_final(out_dir / "adapter.safetensors", model, tok, e.data_dir, gpu,
                                    cfg, out_dir / "metrics")
                excl = [mf["slots"][i] for i in info.get("selected_indices", [])]
                results[s][f"{b}|{name}"] = {
                    "metrics": {k: summ[k] for k in ("Y_huayu_release", "FPR_chengdong",
                                                     "FPR_beigang", "FPR_nanzhan", "Acc_normal",
                                                     "Pair_exact", "closedset")},
                    "selected_or_info": {"selected": excl if "selected" in str(info) else info},
                }
                print(f"[P6] {s}/{b}/{name} 完成", flush=True)
    (rep / "defense_metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    e.mark("P6")
    print("[P6] 防御评价完成；最终报告生成", flush=True)
    from src.e11.report import write_final_reports
    write_final_reports(e)
