"""任务三聚合器：汇总 local_vs_global 的 207 项评价为 U_LOCAL_VS_GLOBAL.json。

只读取分析根下各 meta.json（含输入哈希溯源）与其 eval summary，
产出逐轮对比表 + 行为差异量化（U 本地选择性 vs 聚合选择性）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.e11.task2_mechanism.common import ANALYSIS_ROOT, REPORT_ROOT, SERIES, stamp, write_json  # noqa: E402

KEYS = ("Y_huayu_release", "FPR_chengdong", "FPR_beigang", "Acc_normal", "Pair_exact")


def read_eval(s: str, kind: str, r: int) -> dict | None:
    d = ANALYSIS_ROOT / s / kind / f"round_{r:03d}"
    mp = d / "meta.json"
    if not mp.exists():
        return None
    meta = json.loads(mp.read_text(encoding="utf-8"))
    summ = json.loads((d / "eval.json").read_text(encoding="utf-8"))
    out = {"inputs": meta.get("inputs"), "n": {k: summ[k]["n"] if isinstance(summ[k], dict) else None for k in KEYS}}
    for k in KEYS:
        out[k] = summ[k]["rate"] if isinstance(summ[k], dict) else summ[k]
    out["margin_huayu_defective"] = float(summ["margin_huayu_defective"]) if summ.get("margin_huayu_defective") is not None else None
    out["closedset_option"] = summ.get("closedset", {}).get("option", {}).get("rate")
    return out


def main() -> None:
    out: dict = {"series": {}, **stamp(scope="task3_aggregate")}
    for s in SERIES:
        rows = {}
        for r in range(1, 31):
            entry = {}
            for kind in ("U_upload", "agg10", "n9_mean"):
                e = read_eval(s, kind, r)
                if e is not None:
                    entry[kind] = e
            if "U_upload" in entry and "agg10" in entry:
                u, a = entry["U_upload"], entry["agg10"]
                # 行为差异：本地 vs 聚合（正=聚合更高）
                entry["delta_agg_minus_local"] = {
                    "Y_huayu": None if u["Y_huayu_release"] is None else a["Y_huayu_release"] - u["Y_huayu_release"],
                    "FPR_cd": None if u["FPR_chengdong"] is None else a["FPR_chengdong"] - u["FPR_chengdong"],
                    "FPR_bg": None if u["FPR_beigang"] is None else a["FPR_beigang"] - u["FPR_beigang"],
                    "margin_huayu": None if u["margin_huayu_defective"] is None else a["margin_huayu_defective"] - u["margin_huayu_defective"]}
                # 选择性指标：目标放行率 − 错误仓库平均误放率（>0 才有条件结构）
                def sel(m):
                    if m["Y_huayu_release"] is None or m["FPR_chengdong"] is None or m["FPR_beigang"] is None:
                        return None
                    return m["Y_huayu_release"] - (m["FPR_chengdong"] + m["FPR_beigang"]) / 2
                entry["selectivity"] = {k: sel(v) for k, v in (("U_upload", u), ("agg10", a))}
                if entry.get("n9_mean") is not None:
                    entry["selectivity"]["n9_mean"] = sel(entry["n9_mean"])
            rows[str(r)] = entry
        t1 = read_eval(s, "t1_ref", 0)
        t1f = read_eval(s, "t1_final", 0)
        out["series"][s] = {"rounds": rows, "t1_ref": t1, "t1_final": t1f,
                            "n_evals_done": sum(1 for e in rows.values() if "agg10" in e)}
    write_json(REPORT_ROOT / "U_LOCAL_VS_GLOBAL.json", out)
    n = sum(v["n_evals_done"] for v in out["series"].values())
    print(f"[agg] rounds with both U_upload+agg10: {n}/90 -> {REPORT_ROOT / 'U_LOCAL_VS_GLOBAL.json'}", flush=True)


if __name__ == "__main__":
    main()
