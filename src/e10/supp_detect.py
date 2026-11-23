"""E10 补充实验三检测回放：E11（K=10 冻结）在四种客户端条件下的行为（2026-11-23）。

条件与口径（均从已归档上传重建，CPU 线性代数，无 GPU）：
- b11（甲乙同参与，原数据回放）：detection_rounds.jsonl 已存 90 行；
- b10/b01（仅甲/仅乙·口径A 缺失槽=零矩阵，原数据回放）：已存 90+90 行；
- b10n/b01n（仅甲/仅乙·口径B 缺失槽=同轮原始正常上传，本次新建）：
  [ca, c01] + bg 与 [c00, cb] + bg，逐轮重算 norms/cosine/E_ij（同一
  detect_one_round 代码路径与 P6 冻结阈值）；
- benign（全正常，原数据回放）：13 条测试轨迹 × 6 轮 = 78 行已存；
- benign_holdout（全正常·留出）：clean_s17/clean_s42 两条校准轨迹 12 轮，
  未参与 E11 评估集（E11 的 K=10 先验固定，此处为独立性更强的阴性对照）。

每条件每轮输出：E11 报警对（argmax 且 ≥K×中位，K=10）、按 Norm 更新大小
前两名（仓库 Norm 方法的因子范数排序）、Norm fpr5 冻结阈报警客户端。

产出：artifacts/e10/p7_supp/detection/{supp_detection_rounds.jsonl, supp_detect_summary.json}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(4)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.e10.p7_detect import (BG_SLOTS, CLIENTS, detect_one_round,  # noqa: E402
                               load_flat)
from src.e10.lora_algebra import module_pairs  # noqa: E402

P7 = ROOT / "artifacts/e10/p7"
SUPP = ROOT / "artifacts/e10/p7_supp"
K = 10.0
HOLDOUT_TRAJS = ["e10bg_clean_s17", "e10bg_clean_s42"]
ROUNDS = [1, 2, 3, 4, 5, 6]


def e11_rule(pair_features: dict, k: float = K) -> tuple[str, bool, float, float]:
    vals = sorted(((v["E_ij_norm"], key) for key, v in pair_features.items()), reverse=True)
    top_v, top_k = vals[0]
    med = vals[len(vals) // 2][0]
    alarm = top_v >= k * med
    return top_k, alarm, top_v, med


def top2_by_norm(norms: dict) -> list[str]:
    return [c for c, _ in sorted(norms.items(), key=lambda kv: -kv[1])[:2]]


def summarize(rec: dict, true_pair: str, construct_slots: list[str]) -> dict:
    top_k, alarm, top_v, med = e11_rule(rec["pair_features"])
    pair_members = set(top_k.split("|"))
    return {
        "e11_alarm": alarm, "e11_top_pair": top_k,
        "e11_top_value": top_v, "e11_median": med,
        "e11_pair_equals_true": top_k == true_pair if true_pair else None,
        "e11_contains_construct": sorted(pair_members & set(construct_slots)) if construct_slots else None,
        "e11_false_second_member": sorted(pair_members - set(construct_slots)) if construct_slots else None,
        "norm_top2": top2_by_norm(rec["norms"]),
        "norm_top2_equals_true": sorted(top2_by_norm(rec["norms"])) == sorted(true_pair.split("|")) if true_pair else None,
        "norm_fpr5_flags": rec["alarms"]["norm_flags"]["fpr5"],
        "norm_fpr5_round_alarm": rec["alarms"]["norm_round_alarm"]["fpr5"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="all",
                    help="all | dolly | pku_s17 | pku_s2026 | pku_s42 | holdout")
    args = ap.parse_args()
    t0 = time.time()
    cfg = json.loads((ROOT / "reports/e10_p6_calibration_20261119/detector_config.json")
                     .read_text(encoding="utf-8"))
    construct = json.loads((P7 / "construct_summary.json").read_text(encoding="utf-8"))
    out_dir = SUPP / "detection"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.shard == "all" else f".{args.shard}"
    out_path = out_dir / f"supp_detection_rounds{suffix}.jsonl"

    n_new = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        # ---- 新建条件：b10n / b01n（仅甲/仅乙·正常替代口径）----
        for m in construct["per_round"]:
            series, traj, rd = m["series"], m["bg"], m["round"]
            if args.shard != "all" and not series.startswith(args.shard):
                continue
            rd_dir = ROOT / "artifacts/runs" / traj / "checkpoints" / f"round_{rd:03d}"
            received = load_flat(rd_dir / "received_global_adapter.safetensors")
            p7r = P7 / series / f"{traj}_r{rd}"
            ca = load_flat(p7r / "constructs/a/adapter.safetensors")
            cb = load_flat(p7r / "constructs/b/adapter.safetensors")
            c00 = load_flat(rd_dir / "clients/client_00/final_adapter.safetensors")
            c01 = load_flat(rd_dir / "clients/client_01/final_adapter.safetensors")
            bg = [load_flat(rd_dir / "clients" / c / "final_adapter.safetensors")
                  for c in BG_SLOTS]
            for cond, uploads, cslots in (
                    ("b10n", [ca, c01] + bg, ["client_00"]),
                    ("b01n", [c00, cb] + bg, ["client_01"])):
                tag = f"{series}|{traj}|r{rd}|{cond}"
                rec = detect_one_round(tag, uploads, received, cfg, None, CLIENTS)
                rec["analysis"] = summarize(rec, "client_00|client_01", cslots)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_new += 1
            del received, ca, cb, c00, c01, bg
            if n_new % 20 == 0:
                print(f"[supp-detect] {n_new} 新行 | {time.time()-t0:.0f}s", flush=True)

        # ---- 留出全正常轨迹 ----
        if args.shard in ("all", "holdout"):
            for traj in HOLDOUT_TRAJS:
                for rd in ROUNDS:
                    rd_dir = ROOT / "artifacts/runs" / traj / "checkpoints" / f"round_{rd:03d}"
                    if not rd_dir.exists():
                        continue
                    received = load_flat(rd_dir / "received_global_adapter.safetensors")
                    uploads = [load_flat(rd_dir / "clients" / c / "final_adapter.safetensors")
                               for c in CLIENTS]
                    tag = f"{traj}|{traj}|r{rd}|benign_holdout"
                    rec = detect_one_round(tag, uploads, received, cfg, None, CLIENTS)
                    rec["analysis"] = summarize(rec, None, [])
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_new += 1
                    print(f"[supp-detect] holdout {traj} r{rd} 完成 | {time.time()-t0:.0f}s", flush=True)

    print(f"[supp-detect] shard={args.shard} 新建 {n_new} 行 → {out_path} | {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
