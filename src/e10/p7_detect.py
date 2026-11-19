"""E10 P7 同轮检测回放（P6 冻结规则，方案 §12.4，2026-11-20）。

输入全部为已核验的最终测试材料（p7_verify_inputs 19/19）：
- 90 混合轮（p7_construct 产物 + 背景轨迹真实上传）：11/10/01 三个分支轮的逐槽上传；
- 13 条全良性测试轨迹 × 6 轮：纯阴性轮（误报分母）。

方法与规则（全部引用 detector_config.json 冻结值，不得回调）：
- Norm：客户端上传因子范数 > norm_upper(fpr5/fpr1) → 客户端标记；轮级 = 任一客户端超阈；
- Cosine：逐对余弦 < cosine_lower(fpr5/fpr1) → 对标记；零范数对单列未定义；
  构造对（11 轮的槽 0|1）记录其在 45 对中的升序排名；
- FoolsGold（固定实现）：冷启动（history=[]）逐轮；11 轮另报 join 变体
  （背景槽带轨迹真实历史、构造槽历史置零 = 新客户端加入场景）；
- Multi-Krum f=2：记录分数与选中集合（防御侧信号）；
- Trimmed-Mean（两端各裁 2）：round 6 混合轮落盘聚合产物供行为评估（mk_11/tm_11）；
- CBDF（top_k=3，algorithm1_either_directed）：仅全良性轨迹（完整历史）——
  混合轮构造客户端无真实历史，按 §12.4 标记不适用；
- E_ij/C_ij：全对留痕（P9 漏检归因素材，不定阈值）。

产出：artifacts/e10/p7/detection/{detection_rounds.jsonl, detection_summary.json}。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(4)  # 小矩阵闭式 + 大向量距离混合负载：限线程防过度订阅

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.e10.lora_algebra import from_pairs, module_pairs  # noqa: E402
from src.e10.p6_calibrate import round_features  # noqa: E402
from src.fed.aggregate import flatten_deltas, foolsgold, multi_krum, trimmed_mean  # noqa: E402

P7 = ROOT / "artifacts/e10/p7"
BG_SLOTS = [f"client_{i:02d}" for i in range(2, 10)]
CLIENTS = [f"client_{i:02d}" for i in range(10)]
BENIGN_TRAJS = (["e10bg_clean_s2026"]
                + [f"e10bg_pku_{s}_{i}" for s in ("s17", "s42", "s2026")
                   for i in ("00", "01", "10", "11")])
CBDF_CFG = {"top_k": 3, "edge_rule": "algorithm1_either_directed"}  # configs/detection.yaml 冻结


def load_flat(path: Path) -> dict:
    from safetensors.torch import load_file
    return dict(load_file(str(path)))


def bg_uploads(traj: str, rd: int) -> list[dict]:
    base = ROOT / "artifacts/runs" / traj / "checkpoints" / f"round_{rd:03d}"
    return [load_flat(base / "clients" / c / "final_adapter.safetensors") for c in BG_SLOTS]


def zero_adapter_like(template: dict) -> dict:
    return {k: torch.zeros_like(v) for k, v in template.items()}


def frozen_alarms(features: dict, cfg: dict) -> dict:
    n5, n1 = cfg["norm_upper"]["fpr5"], cfg["norm_upper"]["fpr1"]
    c5, c1 = cfg["cosine_lower"]["fpr5"], cfg["cosine_lower"]["fpr1"]
    zero_slots = {c for c, v in features["norms"].items() if v <= 0}
    norm_flags = {"fpr5": [c for c, v in features["norms"].items() if v > n5],
                  "fpr1": [c for c, v in features["norms"].items() if v > n1]}
    defined_pairs = {k: v for k, v in features["cosine"].items()
                     if not any(s in zero_slots for s in k.split("|"))}
    undefined_pairs = sorted(set(features["cosine"]) - set(defined_pairs))
    cos_flags = {"fpr5": [k for k, v in defined_pairs.items() if v < c5],
                 "fpr1": [k for k, v in defined_pairs.items() if v < c1]}
    ranked = sorted(defined_pairs, key=lambda k: defined_pairs[k])
    return {"norm_flags": norm_flags,
            "norm_round_alarm": {"fpr5": bool(norm_flags["fpr5"]),
                                 "fpr1": bool(norm_flags["fpr1"])},
            "cos_flags": cos_flags,
            "cos_round_alarm": {"fpr5": bool(cos_flags["fpr5"]),
                                "fpr1": bool(cos_flags["fpr1"])},
            "cos_rank_ascending": {k: i + 1 for i, k in enumerate(ranked)},
            "undefined_cosine_pairs": undefined_pairs,
            "thresholds_applied": {"norm_fpr5": n5, "norm_fpr1": n1,
                                   "cos_fpr5": c5, "cos_fpr1": c1}}


def detect_one_round(tag: str, uploads: list[dict], received_global: dict,
                     cfg: dict, fg_history: list | None, slot_names: list[str]) -> dict:
    pairs = {name: module_pairs(up) for name, up in zip(slot_names, uploads)}
    feats = round_features(pairs)
    rec = {"tag": tag, "slots": slot_names,
           "norms": feats["norms"], "cosine": feats["cosine"],
           "pair_features": feats["pair_features"],
           "alarms": frozen_alarms(feats, cfg)}
    # FoolsGold 冷启动
    _, fg_info, _ = foolsgold(uploads, received_global, history=[])
    rec["foolsgold_cold"] = {"weights": fg_info["weights"],
                             "degenerate": fg_info["degenerate_equal_weights"]}
    # Multi-Krum（防御侧）
    try:
        _, mk_info = multi_krum(uploads, received_global, f=2)
        rec["multi_krum"] = {"selected": [slot_names[i] for i in mk_info["selected_indices"]],
                             "scores": mk_info["scores"]}
    except Exception as e:  # noqa: BLE001
        rec["multi_krum"] = {"error": str(e)}
    # FoolsGold join（仅显式提供历史时）
    if fg_history is not None:
        _, fgj, _ = foolsgold(uploads, received_global, history=fg_history)
        rec["foolsgold_join"] = {"weights": fgj["weights"],
                                 "history_rounds": fgj["history_rounds"],
                                 "degenerate": fgj["degenerate_equal_weights"]}
    return rec


def build_join_history(traj: str, upto_rd: int, template: dict) -> list:
    """混合轮 join 历史：槽 2..9 = 轨迹真实先前轮 deltas；槽 0/1 = 零（新客户端）。"""
    hist = []
    for r in range(1, upto_rd):
        base = ROOT / "artifacts/runs" / traj / "checkpoints" / f"round_{r:03d}"
        gl = load_flat(base / "received_global_adapter.safetensors")
        states = [zero_adapter_like(template)] * 2 + \
                 [load_flat(base / "clients" / c / "final_adapter.safetensors") for c in BG_SLOTS]
        hist.append(flatten_deltas(states, gl))
    return hist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/e10/p7/detection")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 个混合轮（冒烟）")
    args = ap.parse_args()
    t0 = time.time()
    cfg = json.loads((ROOT / "reports/e10_p6_calibration_20261119/detector_config.json")
                     .read_text(encoding="utf-8"))
    summary_construct = json.loads((P7 / "construct_summary.json").read_text(encoding="utf-8"))
    rounds_meta = summary_construct["per_round"]
    if args.limit:
        rounds_meta = rounds_meta[:args.limit]

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    records = []

    # ---- 混合轮 ----
    for i, m in enumerate(rounds_meta):
        series, traj, rd = m["series"], m["bg"], m["round"]
        tag = f"{series}|{traj}|r{rd}"
        rd_dir = ROOT / "artifacts/runs" / traj / "checkpoints" / f"round_{rd:03d}"
        received = load_flat(rd_dir / "received_global_adapter.safetensors")
        bg = bg_uploads(traj, rd)
        ca = load_flat(P7 / series / f"{traj}_r{rd}" / "constructs/a/adapter.safetensors")
        cb = load_flat(P7 / series / f"{traj}_r{rd}" / "constructs/b/adapter.safetensors")
        zero = zero_adapter_like(ca)

        rec11 = detect_one_round(f"{tag}|b11", [ca, cb] + bg, received, cfg, None, CLIENTS)
        rec11["foolsgold_join"] = None
        try:
            hist = build_join_history(traj, rd, ca)
            if hist:
                _, fgj, _ = foolsgold([ca, cb] + bg, received, history=hist)
                rec11["foolsgold_join"] = {"weights": fgj["weights"],
                                           "history_rounds": fgj["history_rounds"],
                                           "degenerate": fgj["degenerate_equal_weights"]}
            del hist
        except Exception as e:  # noqa: BLE001
            rec11["foolsgold_join"] = {"error": str(e)}
        rec11["construct_pair"] = {
            "pair": "client_00|client_01",
            "cosine": rec11["cosine"].get("client_00|client_01"),
            "rank_ascending": rec11["alarms"]["cos_rank_ascending"].get("client_00|client_01"),
            "norm_a": rec11["norms"]["client_00"], "norm_b": rec11["norms"]["client_01"],
            "flagged_norm_fpr5": "client_00" in rec11["alarms"]["norm_flags"]["fpr5"]
                                 and "client_01" in rec11["alarms"]["norm_flags"]["fpr5"],
            "flagged_cos_fpr5": "client_00|client_01" in rec11["alarms"]["cos_flags"]["fpr5"]}
        rec10 = detect_one_round(f"{tag}|b10", [ca, zero] + bg, received, cfg, None, CLIENTS)
        rec01 = detect_one_round(f"{tag}|b01", [zero, cb] + bg, received, cfg, None, CLIENTS)

        # E_ij/C_ij 已随 pair_features 全对留痕（P9 素材）

        # round 6：落盘 MK/TM 聚合产物（行为评估用）
        if rd == 6:
            from safetensors.torch import save_file
            p7round = P7 / series / f"{traj}_r{rd}" / "branches"
            try:
                mk_adapter, mk_info = multi_krum([ca, cb] + bg, received, f=2)
                d = p7round / "mk_11"
                d.mkdir(parents=True, exist_ok=True)
                save_file({k: v.contiguous() for k, v in mk_adapter.items()},
                          str(d / "adapter.safetensors"))
                (d / "branch.json").write_text(json.dumps(
                    {"protocol": "multi_krum_f2_aggregate", "rank": 8, "alpha": 16,
                     "scaling": 2.0, "series": series, "background_run": traj, "round": rd,
                     "selected": mk_info["selected_indices"]}, ensure_ascii=False, indent=1))
            except Exception as e:  # noqa: BLE001
                (p7round / "mk_11_error.txt").write_text(str(e))
            tm_adapter = trimmed_mean([ca, cb] + bg, trim_count=2)
            d = p7round / "tm_11"
            d.mkdir(parents=True, exist_ok=True)
            save_file({k: v.contiguous() for k, v in tm_adapter.items()},
                      str(d / "adapter.safetensors"))
            (d / "branch.json").write_text(json.dumps(
                {"protocol": "trimmed_mean_trim2_aggregate", "rank": 8, "alpha": 16,
                 "scaling": 2.0, "series": series, "background_run": traj, "round": rd},
                ensure_ascii=False, indent=1))

        records += [rec11, rec10, rec01]
        print(f"[p7-detect] {i + 1}/{len(rounds_meta)} {tag} "
              f"({time.time() - t0:.0f}s)", flush=True)

    # ---- 全良性轨迹（阴性对照 + CBDF） ----
    from src.detect.cbdf import cbdf
    from src.detect.lowrank import make_update
    modules = tuple({"q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"})
    for traj in BENIGN_TRAJS:
        fg_hist: list = []
        lr_hist = []
        for rd in range(1, 7):
            base = ROOT / "artifacts/runs" / traj / "checkpoints" / f"round_{rd:03d}"
            received = load_flat(base / "received_global_adapter.safetensors")
            ups = [load_flat(base / "clients" / c / "final_adapter.safetensors")
                   for c in CLIENTS]
            rec = detect_one_round(f"{traj}|r{rd}|benign", ups, received, cfg, None, CLIENTS)
            _, fgi, fg_hist = foolsgold(ups, received, history=fg_hist)  # 在线累计
            rec["foolsgold_online"] = {"weights": fgi["weights"],
                                       "history_rounds": fgi["history_rounds"],
                                       "degenerate": fgi["degenerate_equal_weights"]}
            lr_hist.append([make_update(u, received, 2.0, modules) for u in ups])
            records.append(rec)
        try:
            cb = cbdf(lr_hist, CLIENTS, k=CBDF_CFG["top_k"], edge_rule=CBDF_CFG["edge_rule"])
            records.append({"tag": f"{traj}|cbdf", "cbdf": cb})
        except Exception as e:  # noqa: BLE001
            records.append({"tag": f"{traj}|cbdf", "cbdf_error": str(e)})
        print(f"[p7-detect] benign {traj} done ({time.time() - t0:.0f}s)", flush=True)

    with open(out / "detection_rounds.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_benign = sum(1 for r in records if r["tag"].endswith("|benign"))
    print(f"[p7-detect] 完成：混合轮记录 {len(records) - n_benign - 13}（含 cbdf 13 条），"
          f"良性轮 {n_benign}，共 {len(records)} 条 | {time.time() - t0:.0f}s → {out}")


if __name__ == "__main__":
    main()
