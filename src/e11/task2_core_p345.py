"""E11 Task2 core P3-P5：行为评价、锁定确认、检测评价、聚合回放、最终报告。

正式实现（src/e11/ 版本），相对 .work 草稿版的缺口修复：
1. I_H：r1 共同起点检查点在探针集上逐提示重评，商品 ID 配对 Bootstrap（≥2000 次），
   不再用二项近似；五条件 r1 检查点全部重评（N 来自旧轨迹 r1 检查点）。
2. U/S-AB 等价性：锁定确认集逐提示结果，按商品 ID 配对 Bootstrap 差值 CI，
   ±5pp 等价范围逐指标判定（华宇放行、城东/北港误放、正常放行、Pair、Margin）。
3. 检测指标：补 ROC-AUC / PR-AUC / 客户端排名 / 逐轮与汇总 / 轮次分块 Wilson 区间；
   增加 CBDF（沿用 src.detect.cbdf 现有组合识别规则，正常轨迹同口径）。
4. 报告：着色表格（HTML）+ Markdown 主表 + 失败项如实保留。

输入：P2 完成的四条轨迹 + 旧 P3 三条正常轨迹（只读）。
输出：reports/e11_task2_core/task2_core_dolly_s17_a5/ 与 artifacts 同名目录。
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

RUN = "task2_core_dolly_s17_a5"
ART = ROOT / "artifacts/e11_task2_core" / RUN
REP = ROOT / "reports/e11_task2_core" / RUN
OLD = ROOT / "artifacts/e11_task2/task2_20261125"
T1 = ROOT / "artifacts/e11_task1/task1_20261124"
COND_TRAJ = {"N": "N_dolly_s17_a5", "U": "U_dolly_s17_a5", "S-A": "S-A_dolly_s17_a5",
             "S-B": "S-B_dolly_s17_a5", "S-AB": "S-AB_dolly_s17_a5"}
MAL_LABELS = {"U": {"client_00"}, "S-A": {"client_00"}, "S-B": {"client_01"},
              "S-AB": {"client_00", "client_01"}}
MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20261123
EQUIV_RANGE = 0.05  # ±5 个百分点


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def load_cfg() -> dict:
    import yaml
    from src.config import canonical_hash
    p = ROOT / "configs/e11/task2_core.yaml"
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    cfg["_hash"] = canonical_hash(cfg)
    return cfg


def traj_dir(cond: str) -> Path:
    return (OLD if cond == "N" else ART) / "trajectories" / COND_TRAJ[cond]


def paired_bootstrap_diff(vals_a: list, vals_b: list, n: int = BOOTSTRAP_N,
                          seed: int = BOOTSTRAP_SEED) -> dict:
    """商品 ID 配对的差值 Bootstrap：对同一商品集合重采样，A-B 差值均值 CI。"""
    import random
    assert len(vals_a) == len(vals_b) > 0, "配对 bootstrap 需要等长逐商品值"
    rng = random.Random(seed)
    N = len(vals_a)
    diffs = []
    for _ in range(n):
        idx = [rng.randrange(N) for _ in range(N)]
        diffs.append(sum(vals_a[i] - vals_b[i] for i in idx) / N)
    diffs.sort()
    point = (sum(vals_a) - sum(vals_b)) / N
    return {"mean_diff": point,
            "ci": [diffs[int(0.025 * n)], diffs[int(0.975 * n) - 1]],
            "n_pairs": N, "resamples": n}


def _wilson(hits: int, n: int, z: float = 1.959963984540054) -> dict:
    if n == 0:
        return {"rate": None, "lower": None, "upper": None}
    p = hits / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"rate": p, "lower": max(0.0, c - h), "upper": min(1.0, c + h)}


def cfg_dev_rows(cfg: dict) -> list[dict]:
    return [json.loads(l) for l in
            (T1 / "data/eval_dev.jsonl").read_text(encoding="utf-8").splitlines()]


# ---------------------------------------------------------------- P3 行为评价
def p3_behavior(gpu: int) -> dict:
    import src.e11.evaluate as E11E
    from src.e11.algebra import flatten_of, load_pairs_file
    from src.e11.task2_core import load_cfg
    from src.train.model import apply_adapter, load_model
    cfg = load_cfg()
    device = torch.device(f"cuda:{gpu}")
    model, tok, _ = load_model({**cfg["model"], "lora": cfg["lora"]}, device, 17)
    out_path = REP / "behavior_metrics.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    locked_rows = [json.loads(l) for l in
                   (OLD / "data/locked_confirmation/locked_confirmation_1536.jsonl").read_text(
                       encoding="utf-8").splitlines()]
    lf = ART / "locked_test_results.json"
    locked_done = set()
    if lf.exists():
        locked_done = {r["condition"] for r in json.loads(lf.read_text())["results"]}
    locked_results = []

    # ---- 开发集汇总（r0/r15/r25/r30 由 P2 逐轮落盘；缺失轮次用检查点补评并落盘）
    for cond, tid in COND_TRAJ.items():
        td = traj_dir(cond)
        for r in (0, 15, 25, 30):
            src = td / "eval_dev" / f"round_{r:03d}.json"
            if src.exists():
                summary = json.loads(src.read_text())
            else:
                gp = (td / "init_adapter.safetensors" if r == 0
                      else td / f"global_round_{r:03d}.safetensors")
                if not gp.exists():
                    summary = None
                else:
                    apply_adapter(model, flatten_of(load_pairs_file(gp)))
                    model.eval()
                    rows = E11E.eval_rows(model, tok, cfg_dev_rows(cfg), str(device), cfg)
                    summary = E11E.summarize(rows, cfg["evaluation"]["dev_warehouses"])
                    dst = td / "eval_dev" / f"round_{r:03d}.json"
                    dst.write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"condition": cond, "round": r, "summary": summary,
                                    "src": str(src)}, ensure_ascii=False) + "\n")

    # ---- 锁定确认集：全部冻结后一次（五条件 × 1536 条，含 N 基线）
    for cond, tid in COND_TRAJ.items():
        if cond in locked_done:
            continue
        td = traj_dir(cond)
        gp = td / "global_round_030.safetensors"
        apply_adapter(model, flatten_of(load_pairs_file(gp)))
        model.eval()
        rows = E11E.eval_rows(model, tok, locked_rows, str(device), cfg)
        summ = E11E.summarize(rows, cfg["evaluation"]["test_warehouses"])
        lp = ART / "locked_confirmation" / cond
        lp.mkdir(parents=True, exist_ok=True)
        (lp / "prompt_results.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        locked_results.append({"condition": cond, "round": 30, "summary": summ,
                               "global_sha16": sha(gp)[:16]})
    if locked_results:
        merged = locked_results
        if lf.exists():
            merged += [r for r in json.loads(lf.read_text())["results"]
                       if r["condition"] not in {m["condition"] for m in merged}]
        lf.write_text(json.dumps(
            {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "n_prompts": len(locked_rows),
             "used_once_after_freeze": True, "results": merged},
            ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 交互项 I_H：r1 共同起点，探针集逐提示重评五条件 r1 检查点
    probe_rows = [json.loads(l) for l in
                  (T1 / "data/probe_dev.jsonl").read_text(encoding="utf-8").splitlines()]
    ih_rows = {}
    ih_dir = ART / "r1_probe_reeval"
    ih_dir.mkdir(parents=True, exist_ok=True)
    for cond in COND_TRAJ:
        td = traj_dir(cond)
        gp = td / "global_round_001.safetensors"
        if not gp.exists():
            continue
        cache = ih_dir / f"{cond}.jsonl"
        if cache.exists():
            rows = [json.loads(l) for l in cache.read_text(encoding="utf-8").splitlines()]
        else:
            apply_adapter(model, flatten_of(load_pairs_file(gp)))
            model.eval()
            rows = E11E.eval_rows(model, tok, probe_rows, str(device), cfg)
            cache.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                             encoding="utf-8")
        ih_rows[cond] = rows
    ih = None
    if all(c in ih_rows for c in COND_TRAJ):
        def _hy_rel(rows):
            return {r["product_id"]: 1 if r["category"] == "release" else 0 for r in rows
                    if r["warehouse"] == "huayu" and r["quality_status"] == "defective"}
        m00, m10 = _hy_rel(ih_rows["N"]), _hy_rel(ih_rows["S-A"])
        m01, m11 = _hy_rel(ih_rows["S-B"]), _hy_rel(ih_rows["S-AB"])
        pids = sorted(m00)
        assert pids == sorted(m10) == sorted(m01) == sorted(m11), "r1 探针商品集不一致"
        y00 = sum(m00.values()) / len(pids)
        y10 = sum(m10.values()) / len(pids)
        y01 = sum(m01.values()) / len(pids)
        y11 = sum(m11.values()) / len(pids)
        i_h_vals = [m11[p] - m10[p] - m01[p] + m00[p] for p in pids]
        boot = paired_bootstrap_diff(i_h_vals, [0] * len(pids))
        ih = {"Y00_N": y00, "Y10_SA": y10, "Y01_SB": y01, "Y11_SAB": y11,
              "I_H": y11 - y10 - y01 + y00, "threshold": 0.70,
              "n_products": len(pids),
              "bootstrap": {"mean": boot["mean_diff"], "ci": boot["ci"],
                            "resamples": boot["resamples"],
                            "pairing": "probe 商品 ID 配对"},
              "probe_source": "五条件 r1 global_round_001.safetensors 逐提示重评"
                              "（缓存 r1_probe_reeval/）"}
    (REP / "interaction_I_H.json").write_text(json.dumps(ih, ensure_ascii=False, indent=1),
                                              encoding="utf-8")

    # ---- U 与 S-AB 等价性（锁定集逐提示，商品配对 bootstrap 差值 CI，±5pp）
    equiv = None
    per_cond = {}
    for cond in ("U", "S-AB"):
        p = ART / "locked_confirmation" / cond / "prompt_results.jsonl"
        if p.exists():
            per_cond[cond] = [json.loads(l) for l in
                              p.read_text(encoding="utf-8").splitlines()]
    if all(c in per_cond for c in ("U", "S-AB")):
        u_rows, s_rows = per_cond["U"], per_cond["S-AB"]
        metrics = {
            "Y_huayu_release": lambda r: 1 if (r["warehouse"] == "huayu" and
                                               r["quality_status"] == "defective" and
                                               r["category"] == "release") else 0,
            "FPR_chengdong": lambda r: 1 if (r["warehouse"] == "chengdong" and
                                             r["quality_status"] == "defective" and
                                             r["category"] == "release") else 0,
            "FPR_beigang": lambda r: 1 if (r["warehouse"] == "beigang" and
                                           r["quality_status"] == "defective" and
                                           r["category"] == "release") else 0,
            "Acc_normal": lambda r: 1 if (r["quality_status"] == "normal" and
                                          r["category"] == "release") else 0,
        }
        out_m = {}
        for name, fn in metrics.items():
            du = {r["product_id"]: fn(r) for r in u_rows}
            ds = {r["product_id"]: fn(r) for r in s_rows}
            pids = sorted(du)
            assert pids == sorted(ds), f"{name} 商品集不一致"
            vu, vs = [du[p] for p in pids], [ds[p] for p in pids]
            boot = paired_bootstrap_diff(vu, vs)
            within = (abs(boot["mean_diff"]) <= EQUIV_RANGE and
                      abs(boot["ci"][0]) <= EQUIV_RANGE and
                      abs(boot["ci"][1]) <= EQUIV_RANGE)
            out_m[name] = {"U_rate": sum(vu) / len(pids), "SAB_rate": sum(vs) / len(pids),
                           "diff_ci": boot["ci"], "equivalent": within,
                           "criterion": "配对95%CI 整体位于 ±5pp 内"}
        # Pair_exact（商品级四仓库规则判定差值）
        def _pairok(rows):
            d = defaultdict(dict)
            for r in rows:
                if r["quality_status"] == "defective":
                    d[r["product_id"]][r["warehouse"]] = r["rule_ok"]
            keys = ["huayu", "chengdong", "beigang", "nanzhan"]
            return {p: 1 if len(w) == len(keys) and all(w.values()) else 0
                    for p, w in d.items()}
        pu, ps = _pairok(u_rows), _pairok(s_rows)
        pids = sorted(pu)
        assert pids == sorted(ps), "Pair_exact 商品集不一致"
        boot = paired_bootstrap_diff([pu[p] for p in pids], [ps[p] for p in pids])
        out_m["Pair_exact"] = {
            "U_rate": sum(pu.values()) / len(pids),
            "SAB_rate": sum(ps.values()) / len(pids), "diff_ci": boot["ci"],
            "equivalent": (abs(boot["mean_diff"]) <= EQUIV_RANGE and
                           abs(boot["ci"][0]) <= EQUIV_RANGE and
                           abs(boot["ci"][1]) <= EQUIV_RANGE)}
        margins = {}
        for w in ("huayu", "chengdong", "beigang"):
            mu = [r["margin"] for r in u_rows if r["warehouse"] == w
                  and r["quality_status"] == "defective"]
            ms = [r["margin"] for r in s_rows if r["warehouse"] == w
                  and r["quality_status"] == "defective"]
            if mu and ms:
                margins[f"margin_{w}"] = {"U": sum(mu) / len(mu),
                                          "S-AB": sum(ms) / len(ms),
                                          "diff": sum(mu) / len(mu) - sum(ms) / len(ms)}
        main_ok = all(out_m[k]["equivalent"] for k in
                      ("Y_huayu_release", "FPR_chengdong", "FPR_beigang", "Acc_normal"))
        equiv = {"equivalence_range_pp": EQUIV_RANGE * 100, "metrics": out_m,
                 "margins": margins,
                 "conclusion": ("行为等价证据成立：主要指标配对95%CI 整体位于 ±5pp 内"
                                if main_ok and out_m["Pair_exact"]["equivalent"]
                                else "行为等价证据不成立（至少一项指标 CI 超出 ±5pp）")}
        (REP / "equivalence_U_SAB.json").write_text(
            json.dumps(equiv, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"locked_conditions": sorted(locked_done |
                                        {r["condition"] for r in locked_results}),
            "I_H": ih, "equivalence": equiv}


# ---------------------------------------------------------------- P4 检测评价
def p4_detection() -> dict:
    """Norm/Cosine/FoolsGold：s42+s2026 校准 5% FPR；s17 留出；四恶意条件检出。
    增加 ROC-AUC / PR-AUC / 客户端排名 / 逐轮记录 / 轮次分块；CBDF 组合识别。"""
    from sklearn.metrics import average_precision_score, roc_auc_score
    from src.e11.task2_core import load_cfg
    cfg = load_cfg()
    target_fpr = cfg["detection"]["target_fpr"]
    out = {"calibration": {"seeds": ["s42", "s2026"], "target_fpr": target_fpr},
           "thresholds": {}, "holdout": {}, "conditions": {}, "cbdf": {},
           "note": "阈值只由正常轨迹校准；恶意标签只用于离线统计；30 轮非独立样本，"
                   "分块区间仅供描述"}
    for method, key in (("norm", "norm"), ("cosine", "cosine"),
                        ("foolsgold", "foolsgold_score")):
        cal = []
        for s in ("N_dolly_s42_a5", "N_dolly_s2026_a5"):
            for line in (OLD / "trajectories" / s / "detection_rounds.jsonl").read_text(
                    encoding="utf-8").splitlines():
                cal += list(json.loads(line)[key].values())
        cal.sort()
        n = len(cal)
        thr = cal[min(n - 1, int(math.ceil((1 - target_fpr) * n)) - 1)]
        out["thresholds"][method] = {"value": thr, "n_scores": n,
                                     "rule": "高于阈值判定异常（FG score=1-weight 同向）"}
        hold = []
        for line in (OLD / "trajectories" / "N_dolly_s17_a5" / "detection_rounds.jsonl").read_text(
                encoding="utf-8").splitlines():
            hold += list(json.loads(line)[key].values())
        hits = sum(1 for v in hold if v > thr)
        out["holdout"][method] = {"n": len(hold), "hits": hits, "fpr": hits / len(hold),
                                  "wilson95": _wilson(hits, len(hold))}
        for cond in MAL_LABELS:
            labels, scores = [], []
            per_round = []
            for line in (traj_dir(cond) / "detection_rounds.jsonl").read_text(
                    encoding="utf-8").splitlines():
                row = json.loads(line)
                r_met = {"round": row["round"], "flagged": []}
                for cid, v in row[key].items():
                    labels.append(1 if cid in MAL_LABELS[cond] else 0)
                    scores.append(v)
                    if v > thr:
                        r_met["flagged"].append(cid)
                order = sorted(row[key].items(), key=lambda kv: -kv[1])
                r_met["ranking"] = [c for c, _ in order]
                per_round.append(r_met)
            tp = sum(1 for l, v in zip(labels, scores) if l == 1 and v > thr)
            fp = sum(1 for l, v in zip(labels, scores) if l == 0 and v > thr)
            tn = sum(1 for l, v in zip(labels, scores) if l == 0 and v <= thr)
            fn = sum(1 for l, v in zip(labels, scores) if l == 1 and v <= thr)
            acc = (tp + tn) / max(len(labels), 1)
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            both = len(set(labels)) == 2
            rank_pos = []
            for r_met in per_round:
                for c in MAL_LABELS[cond]:
                    if c in r_met["ranking"]:
                        rank_pos.append(r_met["ranking"].index(c) + 1)
            blocks = []
            for b0 in range(1, 31, 10):
                bl = [r for r in per_round if b0 <= r["round"] < b0 + 10]
                if not bl:
                    continue
                flagged = sum(1 for r in bl if set(r["flagged"]) & MAL_LABELS[cond])
                blocks.append({"rounds": f"r{b0}-{b0 + 9}",
                               "rounds_with_malicious_flagged": flagged, "n": len(bl)})
            out["conditions"].setdefault(cond, {})[method] = {
                "tp": tp, "fp": fp, "tn": tn, "fn": fn, "accuracy": acc,
                "precision": prec, "recall": rec,
                "f1": 2 * prec * rec / max(prec + rec, 1e-12),
                "fpr": fp / max(fp + tn, 1), "threshold": thr,
                "roc_auc": float(roc_auc_score(labels, scores)) if both else None,
                "pr_auc": float(average_precision_score(labels, scores)) if both else None,
                "n_rounds": len(per_round), "per_round": per_round,
                "malicious_mean_rank": (sum(rank_pos) / len(rank_pos)) if rank_pos else None,
                "recall_by_block": blocks}
    # ---- CBDF（沿用现有组合识别规则；正常轨迹同口径统计误报组合）
    try:
        from src.detect.cbdf import cbdf
        from src.detect.lowrank import make_update
        from src.e11.algebra import flatten_of, load_pairs_file
        scale = cfg["lora"]["alpha"] / cfg["lora"]["rank"]
        cbdf_out = {}
        for cond in list(MAL_LABELS) + ["N"]:
            td = traj_dir(cond)
            hist = []
            names = None
            for r in range(1, 31):
                blob = torch.load(td / f"uploads_round_{r:03d}.pt", map_location="cpu",
                                  weights_only=True)
                ids = sorted(blob)
                names = ids
                received = flatten_of(load_pairs_file(
                    td / "init_adapter.safetensors" if r == 1
                    else td / f"global_round_{r - 1:03d}.safetensors"))
                hist.append([make_update(blob[c], received, scale, MODULES) for c in ids])
            res = cbdf(hist, names, k=3, edge_rule="algorithm1_either_directed")
            flagged = res["detected_clients"] if isinstance(res, dict) else res
            flagged_s = [str(x) for x in flagged]
            true_set = MAL_LABELS.get(cond, set())
            cbdf_out[cond] = {"flagged": flagged_s,
                              "true_malicious": sorted(true_set),
                              "identity_correct": (sorted(true_set) == sorted(flagged_s)
                                                   if true_set else None),
                              "combo_recall": (len(true_set & set(flagged_s)) / len(true_set)
                                               if true_set else None),
                              "false_combo_clients": sorted(set(flagged_s) - true_set)}
        out["cbdf"] = cbdf_out
    except Exception as exc:  # CBDF 为补充方法：记录不中断
        out["cbdf"] = {"error": repr(exc)}
    (REP / "detector_metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    return out


# ---------------------------------------------------------------- P4 聚合回放
def p4_aggregation_replay(gpu: int) -> None:
    """Multi-Krum (n=10,f=2,选8) 与 Trimmed Mean(trim=2) 同轮影子回放（不反馈训练）。"""
    from src.e11.algebra import flatten_of, load_pairs_file
    from src.e11.task2_core import load_cfg
    from src.fed.aggregate import fedavg, multi_krum, trimmed_mean
    import src.e11.evaluate as E11E
    from src.train.model import apply_adapter, load_model
    cfg = load_cfg()
    device = torch.device(f"cuda:{gpu}")
    model, tok, _ = load_model({**cfg["model"], "lora": cfg["lora"]}, device, 17)
    out_path = REP / "aggregation_replay.jsonl"
    locked_rows = [json.loads(l) for l in
                   (OLD / "data/locked_confirmation/locked_confirmation_1536.jsonl").read_text(
                       encoding="utf-8").splitlines()]
    subset = locked_rows[::8]  # 192 条成本控制，口径如实记录
    sample_rounds = [1, 10, 20, 30]
    sel_freq = defaultdict(int)
    for cond in ("U", "S-A", "S-B", "S-AB"):
        td = traj_dir(cond)
        for r in sample_rounds:
            blob = torch.load(td / f"uploads_round_{r:03d}.pt", map_location="cpu",
                              weights_only=True)
            ids = sorted(blob)
            states = [blob[c] for c in ids]
            received = flatten_of(load_pairs_file(
                td / "init_adapter.safetensors" if r == 1
                else td / f"global_round_{r - 1:03d}.safetensors"))
            _, mk = multi_krum(states, received, f=2, selected=8)
            chosen = {ids[i] for i in mk["selected_indices"]}
            for c in ids:
                if c in chosen:
                    sel_freq[c] += 1
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "condition": cond, "round": r, "method": "multi_krum",
                    "selected": sorted(chosen),
                    "excluded": sorted(set(ids) - chosen),
                    "malicious_selected": sorted(MAL_LABELS[cond] & chosen),
                    "malicious_excluded": sorted(MAL_LABELS[cond] - chosen),
                    "f": 2, "n": 10}, ensure_ascii=False) + "\n")
            tm = trimmed_mean(states, trim_count=2)
            for method, agg in (("multi_krum",
                                 fedavg([states[i] for i in mk["selected_indices"]])),
                                ("trimmed_mean", tm)):
                apply_adapter(model, agg)
                model.eval()
                rows = E11E.eval_rows(model, tok, subset, str(device), cfg)
                summ = E11E.summarize(rows, cfg["evaluation"]["test_warehouses"])
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"condition": cond, "round": r, "method": method,
                                        "subset": "locked[::8] n=192", "behavior": summ},
                                       ensure_ascii=False) + "\n")
            del blob
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"entry": "selection_frequency", "rounds": sample_rounds,
                            "conditions": list(MAL_LABELS),
                            "selected_count_per_client": dict(sel_freq)},
                           ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- P5 报告
def _fmt(v, nd: int = 3) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def p5_reports() -> None:
    beh = [json.loads(l) for l in (REP / "behavior_metrics.jsonl").read_text().splitlines()]
    det = json.loads((REP / "detector_metrics.json").read_text())
    locked = json.loads((ART / "locked_test_results.json").read_text())
    ih = json.loads((REP / "interaction_I_H.json").read_text())
    eps = json.loads((ART / "selected_epsilon.json").read_text())
    equiv = json.loads((REP / "equivalence_U_SAB.json").read_text()) \
        if (REP / "equivalence_U_SAB.json").exists() else None
    s30 = {(r["condition"], r["round"]): r["summary"] for r in beh
           if r["round"] == 30 and r["summary"]}
    lk = {r["condition"]: r["summary"] for r in locked["results"]}

    def gate(cond: str, s: dict) -> dict:
        if cond in ("S-A", "S-B"):
            return {"华宇放行<=10%": s["Y_huayu_release"]["rate"] <= 0.10,
                    "城东误放<=10%": s["FPR_chengdong"]["rate"] <= 0.10,
                    "北港误放<=10%": s["FPR_beigang"]["rate"] <= 0.10,
                    "正常放行>=95%": s["Acc_normal"]["rate"] >= 0.95}
        if cond == "N":
            return {"城东误放<=10%": s["FPR_chengdong"]["rate"] <= 0.10,
                    "北港误放<=10%": s["FPR_beigang"]["rate"] <= 0.10,
                    "正常放行>=95%": s["Acc_normal"]["rate"] >= 0.95,
                    "Pair>=80%": s["Pair_exact"]["rate"] >= 0.80}
        return {"华宇放行>=80%": s["Y_huayu_release"]["rate"] >= 0.80,
                "城东误放<=10%": s["FPR_chengdong"]["rate"] <= 0.10,
                "北港误放<=10%": s["FPR_beigang"]["rate"] <= 0.10,
                "正常放行>=95%": s["Acc_normal"]["rate"] >= 0.95,
                "Pair>=80%": s["Pair_exact"]["rate"] >= 0.80,
                "华宇Margin>0": (s["margin_huayu_defective"] or 0) > 0,
                "城东Margin<0": (s["margin_chengdong_defective"] or 0) < 0,
                "北港Margin<0": (s["margin_beigang_defective"] or 0) < 0}

    lines = ["# E11 Task2 核心实验结果（锁定确认集，一次性评价）", "",
             f"- ε = {eps['epsilon']}（J_screen 最小；seed17 复核不改选择）",
             f"- 交互项 I_H = {_fmt(ih.get('I_H'))}（门槛 >= 0.70，"
             f"{ih.get('n_products')} 商品配对 bootstrap）", ""]

    lines += ["## 锁定确认集（1536 条 × 5 条件）", "",
              "| 条件 | 华宇放行 | 城东误放 | 北港误放 | 正常放行 | Pair | 门槛 |",
              "|---|---|---|---|---|---|---|"]
    for cond in COND_TRAJ:
        if cond not in lk:
            continue
        s, g = lk[cond], gate(cond, lk[cond])
        failed = "、".join(k for k, v in g.items() if not v)
        lines.append(
            f"| {cond} | {s['Y_huayu_release']['rate']:.3f} | "
            f"{s['FPR_chengdong']['rate']:.3f} | {s['FPR_beigang']['rate']:.3f} | "
            f"{s['Acc_normal']['rate']:.3f} | {s['Pair_exact']['rate']:.3f} | "
            f"{'全部通过' if all(g.values()) else '未通过: ' + failed} |")

    lines += ["", "## 开发集（r30 摘要，行为趋势参考）", "",
              "| 条件 | 华宇放行 | 城东误放 | 北港误放 | 正常放行 | Pair |",
              "|---|---|---|---|---|---|"]
    for cond in COND_TRAJ:
        if (cond, 30) in s30:
            s = s30[(cond, 30)]
            lines.append(
                f"| {cond} | {s['Y_huayu_release']['rate']:.3f} | "
                f"{s['FPR_chengdong']['rate']:.3f} | {s['FPR_beigang']['rate']:.3f} | "
                f"{s['Acc_normal']['rate']:.3f} | {s['Pair_exact']['rate']:.3f} |")

    lines += ["", "## 交互项（r1 共同起点）", ""]
    for k in ("Y00_N", "Y10_SA", "Y01_SB", "Y11_SAB", "I_H"):
        lines.append(f"- {k} = {_fmt(ih.get(k))}")
    if ih.get("bootstrap"):
        ci = ih["bootstrap"]["ci"]
        lines.append(f"- Bootstrap 95% CI：[{ci[0]:.3f}, {ci[1]:.3f}]"
                     f"（{ih['bootstrap']['resamples']} 次重采样）")
    lines.append(f"- 门槛 I_H>=0.70：{'通过' if (ih.get('I_H') or 0) >= 0.70 else '未通过'}")

    if equiv:
        lines += ["", "## U vs S-AB 行为等价（±5pp，配对 bootstrap CI）", "",
                  "| 指标 | U | S-AB | 差值 CI | 等价 |", "|---|---|---|---|---|"]
        for name, m in equiv["metrics"].items():
            ci = m["diff_ci"]
            lines.append(f"| {name} | {m['U_rate']:.3f} | {m['SAB_rate']:.3f} | "
                         f"[{ci[0]:.3f}, {ci[1]:.3f}] | "
                         f"{'是' if m['equivalent'] else '否'} |")
        for name, m in equiv.get("margins", {}).items():
            lines.append(f"| {name} | {m['U']:.3f} | {m['S-AB']:.3f} | "
                         f"差 {m['diff']:.3f} | 描述性 |")
        lines += ["", f"结论：{equiv['conclusion']}"]

    lines += ["", "## 检测（5% 目标 FPR 阈值，s42+s2026 校准，s17 留出）", ""]
    for method in ("norm", "cosine", "foolsgold"):
        t = det["thresholds"][method]
        h = det["holdout"][method]
        w = h["wilson95"]
        lines.append(f"- {method}: 阈值={t['value']:.4g}；s17 留出 FPR={h['fpr']:.3f} "
                     f"(Wilson95 [{_fmt(w['lower'])}, {_fmt(w['upper'])}])")
    for cond, methods in det["conditions"].items():
        for m, v in methods.items():
            lines.append(
                f"- {cond}/{m}: acc={v['accuracy']:.3f} precision={v['precision']:.3f} "
                f"recall={v['recall']:.3f} f1={v['f1']:.3f} fpr={v['fpr']:.3f} "
                f"roc_auc={_fmt(v.get('roc_auc'))} pr_auc={_fmt(v.get('pr_auc'))} "
                f"恶意均排名={_fmt(v.get('malicious_mean_rank'), 1)}")
    cbdf = det.get("cbdf", {})
    if "error" not in cbdf:
        lines += ["", "## CBDF（组合识别）", ""]
        for cond, v in cbdf.items():
            if v.get("identity_correct") is not None:
                lines.append(
                    f"- {cond}: 标记 {v['flagged']}（真值 {v['true_malicious']}），"
                    f"组合身份正确={v['identity_correct']}，"
                    f"召回={_fmt(v['combo_recall'])}，误报组合={v['false_combo_clients']}")
    else:
        lines += ["", f"CBDF 失败：{cbdf['error']}"]

    lines += ["", "## 聚合影子回放", "",
              "Multi-Krum(n=10,f=2,选8) 入选/排除与 Trimmed Mean(trim=2) 行为见 "
              "aggregation_replay.jsonl；回放不反馈训练，仅同轮比较。"]
    (REP / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    acc = ["# 验收记录", ""]
    for cond in COND_TRAJ:
        if cond in lk:
            g = gate(cond, lk[cond])
            failed = "、".join(k for k, v in g.items() if not v)
            acc.append(f"- {cond}（锁定集）："
                       f"{'全部通过' if all(g.values()) else '未通过—— ' + failed}")
    acc.append(f"- 交互项 I_H>=0.70：{'通过' if (ih.get('I_H') or 0) >= 0.70 else '未通过'}"
               f"（I_H={_fmt(ih.get('I_H'))}）")
    if equiv:
        acc.append(f"- U/S-AB 等价（±5pp）：{equiv['conclusion']}")
    acc.append("- 失败项与描述性结果均如实保留于 RESULTS.md / detector_metrics.json。")
    (REP / "ACCEPTANCE.md").write_text("\n".join(acc) + "\n", encoding="utf-8")

    audit = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "epsilon": eps["epsilon"],
             "locked_used_once": True, "detector_thresholds": det["thresholds"],
             "I_H": ih.get("I_H"), "config_hash": load_cfg()["_hash"],
             "git_head": __import__("subprocess").run(
                 ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                 cwd=ROOT).stdout.strip()}
    (REP / "final_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=1),
                                          encoding="utf-8")

    # HTML 着色报告
    h = ["<html><head><meta charset='utf-8'><title>E11 Task2 Core</title><style>",
         "body{font-family:sans-serif;margin:24px;background:#fafafa;color:#212121}",
         "table{border-collapse:collapse;margin:12px 0;background:#fff}",
         "th,td{border:1px solid #bbb;padding:6px 12px;text-align:right}",
         "th:first-child,td:first-child{text-align:left}",
         "th{background:#37474f;color:#fff}",
         ".ok{background:#c8e6c9}.bad{background:#ffcdd2}.neu{background:#fff}",
         "pre{background:#fff;border:1px solid #ddd;padding:12px;overflow-x:auto}",
         "</style></head><body>",
         "<h1>E11 Task2 核心实验结果</h1>",
         f"<p>ε={eps['epsilon']}；I_H={_fmt(ih.get('I_H'))}（门槛 0.70，"
         f"{ih.get('n_products')} 商品配对 bootstrap）</p>",
         "<h2>锁定确认集（1536 × 5）</h2>",
         "<table><tr><th>条件</th><th>华宇放行</th><th>城东误放</th><th>北港误放</th>"
         "<th>正常放行</th><th>Pair</th><th>门槛</th></tr>"]
    cell_rules = {"Y_huayu_release": {"S-A": "le", "S-B": "le", "U": "ge80",
                                      "S-AB": "ge80", "N": None},
                  "FPR_chengdong": "le", "FPR_beigang": "le", "Acc_normal": "ge95",
                  "Pair_exact": {"S-A": None, "S-B": None, "U": "ge80", "S-AB": "ge80",
                                 "N": "ge80"}}
    for cond in COND_TRAJ:
        if cond not in lk:
            continue
        s, g = lk[cond], gate(cond, lk[cond])
        row_cells = []
        for key in ("Y_huayu_release", "FPR_chengdong", "FPR_beigang", "Acc_normal",
                    "Pair_exact"):
            v = s[key]["rate"]
            rule = cell_rules.get(key)
            if isinstance(rule, dict):
                rule = rule.get(cond)
            okc = None
            if rule == "le":
                okc = v <= 0.10
            elif rule == "ge95":
                okc = v >= 0.95
            elif rule == "ge80":
                okc = v >= 0.80
            cls = "ok" if okc else ("bad" if okc is False else "neu")
            row_cells.append(f"<td class='{cls}'>{v:.3f}</td>")
        failed = "；".join(k for k, v in g.items() if not v)
        row_cells.append(f"<td class='{'ok' if all(g.values()) else 'bad'}'>"
                         f"{'通过' if all(g.values()) else failed}</td>")
        h.append(f"<tr><td>{cond}</td>{''.join(row_cells)}</tr>")
    h.append("</table>")

    h += ["<h2>U vs S-AB 等价（±5pp）</h2>"]
    if equiv:
        h.append("<table><tr><th>指标</th><th>U</th><th>S-AB</th><th>差值CI</th>"
                 "<th>等价</th></tr>")
        for name, m in equiv["metrics"].items():
            ci = m["diff_ci"]
            h.append(f"<tr><td>{name}</td><td>{m['U_rate']:.3f}</td>"
                     f"<td>{m['SAB_rate']:.3f}</td><td>[{ci[0]:.3f}, {ci[1]:.3f}]</td>"
                     f"<td class=\"{'ok' if m['equivalent'] else 'bad'}\">"
                     f"{'是' if m['equivalent'] else '否'}</td></tr>")
        h.append("</table>")

    h += ["<h2>检测（阈值=正常校准 5% 目标 FPR）</h2>",
          "<table><tr><th>条件/方法</th><th>Acc</th><th>Precision</th><th>Recall</th>"
          "<th>F1</th><th>FPR</th><th>ROC-AUC</th><th>PR-AUC</th><th>恶意均排名</th></tr>"]
    for cond, methods in det["conditions"].items():
        for m, v in methods.items():
            h.append(f"<tr><td>{cond}/{m}</td><td>{v['accuracy']:.3f}</td>"
                     f"<td>{v['precision']:.3f}</td><td>{v['recall']:.3f}</td>"
                     f"<td>{v['f1']:.3f}</td><td>{v['fpr']:.3f}</td>"
                     f"<td>{_fmt(v.get('roc_auc'))}</td><td>{_fmt(v.get('pr_auc'))}</td>"
                     f"<td>{_fmt(v.get('malicious_mean_rank'), 1)}</td></tr>")
    h.append("</table>")
    h.append("<pre>" + "\n".join(lines) + "</pre></body></html>")
    (REP / "report.html").write_text("\n".join(h), encoding="utf-8")


def main() -> None:
    gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    REP.mkdir(parents=True, exist_ok=True)
    print("[P3] behavior + locked confirmation + I_H + equivalence ...", flush=True)
    p3_behavior(gpu)
    print("[P4] detection metrics ...", flush=True)
    p4_detection()
    print("[P4] aggregation replay ...", flush=True)
    p4_aggregation_replay(gpu)
    print("[P5] reports ...", flush=True)
    p5_reports()
    print("[P3-P5] done", flush=True)


if __name__ == "__main__":
    main()
