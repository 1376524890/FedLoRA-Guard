"""E11（P9）新检测方法：E_ij 组合特征对识别（方案 §13，2026-11-20）。

漏检归因（P7 回放，先行）：FoolsGold 冷启动 0/90——构造客户端单独看方向聚类正常；
Norm/Cosine 标记的是构造×背景的幅度/方向异常对，**真同谋对 (0,1) 因更新平行
（cosine≈1）反而不被低余弦规则标记**——现有方法在"指出哪一对"这一层全部失效。

E11（预声明，无特征选择）：每轮对 45 对计算 E_ij_norm（§13.1 组合级特征，
P7 回放已全量记录），报警规则固定为——
  报警对 = argmax E_ij_norm，且其值 ≥ K × 本轮 45 对中位数（K=10 先验固定）；
  敏感性分析另报 K∈{3,30,∞(恒 top-1)}，仅描述不选择。
目标（§13.3）：FPR≤5% 下 pair Recall 较基线 +10pp。

评估数据：P7 的 90 混合轮（真对=槽 0/1）与 78 良性轮（同轮记录，特征已存档）。
纪律注记：规则参数先验固定，未在任何数据上调整；评估轮与 P7 回放同源
（保留轨迹的唯一混合轮集），此局限如实记录。
产出：reports/e10_p9_20261120/{e11_report.json, RESULTS.md}。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TRUE_PAIR = "client_00|client_01"
Ks = [3, 10, 30, float("inf")]


def dedup_rounds(rows: list[dict]) -> dict:
    by_round = {}
    for r in rows:
        tag = r["tag"]
        parts = tag.split("|")
        key = "|".join(parts[:3])  # series|bg|rN（b10/b01/b11 三行同特征，去重）
        by_round.setdefault(key, r)
    return by_round


def eval_rule(pf: dict, k: float) -> tuple[str, bool]:
    vals = sorted(((v["E_ij_norm"], key) for key, v in pf.items()), reverse=True)
    top_v, top_k = vals[0]
    med = vals[len(vals) // 2][0]
    alarm = (top_v >= k * med) if k != float("inf") else True
    return top_k, alarm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/e10_p9_20261120")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in
            open(ROOT / "reports/e10_p7_20261120/detection/detection_rounds.jsonl", encoding="utf-8")]
    mixed = dedup_rounds([r for r in rows if not r["tag"].endswith(("|benign", "|cbdf"))])
    benign = [r for r in rows if r["tag"].endswith("|benign")]

    report = {"rule": "argmax E_ij_norm 且 ≥K×轮内中位（K=10 先验固定）",
              "n_mixed_rounds": len(mixed), "n_benign_rounds": len(benign),
              "true_pair": TRUE_PAIR, "sensitivity": {}}
    for k in Ks:
        top1 = sum(1 for r in mixed.values() if eval_rule(r["pair_features"], k)[0] == TRUE_PAIR)
        alarmed = [rr for rr in mixed.values() if eval_rule(rr["pair_features"], k)[1]]
        det = sum(1 for rr in alarmed if eval_rule(rr["pair_features"], k)[0] == TRUE_PAIR)
        bfp = sum(1 for rr in benign if eval_rule(rr["pair_features"], k)[1])
        key = f"K={k if k != float('inf') else 'inf'}"
        report["sensitivity"][key] = {
            "pair_recall_at_1": top1 / len(mixed), "pair_recall_at_1_n": top1,
            "round_detection": det / len(mixed), "round_detection_n": det,
            "benign_round_fp": bfp / len(benign), "benign_round_fp_n": bfp}
    # 分离度
    att = [max(v["E_ij_norm"] for v in r["pair_features"].values()) for r in mixed.values()]
    ben_top = [max(v["E_ij_norm"] for v in r["pair_features"].values()) for r in benign]
    report["separation"] = {"attack_top1_min": min(att), "attack_top1_median": sorted(att)[len(att)//2],
                            "benign_top1_max": max(ben_top),
                            "gap_ratio": min(att) / max(ben_top)}
    # 基线对照（同数据）：Cosine 低尾规则的真对 recall@1、FoolsGold 冷启动轮检出
    cos_hit = 0
    for r in mixed.values():
        flagged = [p for p, v in r["cosine"].items() if v < 0.4288]  # fpr5 冻结阈
        if TRUE_PAIR in flagged:
            cos_hit += 1
    report["baselines"] = {"cosine_fpr5_true_pair_recall": cos_hit / len(mixed),
                           "foolsgold_cold_round_recall": 0.0,
                           "note": "Cosine 标记构造×背景对；真对平行（cosine≈1）不被低尾规则标记；FG 冷 0/90（P7）"}
    (out / "e11_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    k10 = report["sensitivity"]["K=10"]
    print(f"[p9/E11] K=10: pair Recall@1 = {k10['pair_recall_at_1_n']}/{len(mixed)} "
          f"({k10['pair_recall_at_1']:.1%}) | 轮检出 {k10['round_detection_n']}/{len(mixed)} | "
          f"良性轮误报 {k10['benign_round_fp_n']}/{len(benign)} | "
          f"分离度 攻击最小/良性最大 = {report['separation']['gap_ratio']:.0f}× | "
          f"基线 Cosine 真对 recall {report['baselines']['cosine_fpr5_true_pair_recall']:.1%}")


if __name__ == "__main__":
    main()
