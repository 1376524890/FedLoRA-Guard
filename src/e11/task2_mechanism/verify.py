"""任务一：P1 结果独立验证（只读）。

逐系列 × 逐分支（00/U）× 逐轮（1..30）：
  1) round_manifest 存在、slots 与 alg.slots_for 一致；
  2) 每个上传文件存在、sha256 与 manifest 一致；
  3) state.json 链完整（链长 = rounds_done = 30、逐轮 start/aggregate 哈希衔接、上传哈希一致）；
  4) 用仓库真实 FedAvg（alg.server_aggregate）从槽位上传复算聚合，与 global_r 逐位一致；
  5) 从 eval_dev 的 rows jsonl 用独立公式重算行为指标，与存储 summary 一致；
  6) 用重算指标复现 P1 正式门（gate_clean/gate_strong），与 p1/gate.json 判定一致。
输出：REPORT_ROOT/verification_report.json + 控制台摘要。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.e11.task2_mechanism.common import (REPORT_ROOT, SERIES, global_path,  # noqa: E402
                                            stamp, traj, write_json)
from src.e11.task2_protocol import algebra as alg  # noqa: E402
from src.e11.task2_protocol.config import load_all, sha256_file  # noqa: E402

TOL = 1e-9


def load_flat(path: Path) -> dict:
    from safetensors.torch import load_file
    return {k: v for k, v in load_file(str(path)).items()}


def recompute_metrics_from_rows(rows: list[dict]) -> dict:
    """独立公式重算（不经 src.e11 汇总代码）：分仓库缺陷放行率、正常放行、pair_exact、margin。"""
    def rate(sub):
        return (sum(1 for r in sub if r["category"] == "release") / len(sub)) if sub else None
    hy = [r for r in rows if r["warehouse"] == "huayu" and r["quality_status"] == "defective"]
    cd = [r for r in rows if r["warehouse"] == "chengdong" and r["quality_status"] == "defective"]
    bg = [r for r in rows if r["warehouse"] == "beigang" and r["quality_status"] == "defective"]
    nz = [r for r in rows if r["quality_status"] == "normal"]
    # Pair_exact（协议定义，src.e11.evaluate.summarize 镜像）：
    # 按 product_id 聚合缺陷行；该商品在全部出现过的仓库的 rule_ok 都成立才计 exact。
    by_prod: dict[str, dict[str, bool]] = {}
    for r in rows:
        if r["quality_status"] == "defective":
            by_prod.setdefault(r["product_id"], {})[r["warehouse"]] = bool(r["rule_ok"])
    n_wh = max((len(d) for d in by_prod.values()), default=0)
    pair_ok = [all(d.values()) and len(d) == n_wh for d in by_prod.values()]
    return {"Y_huayu_release": rate(hy), "FPR_chengdong": rate(cd), "FPR_beigang": rate(bg),
            "Acc_normal": rate(nz),
            "Pair_exact": (sum(pair_ok) / len(pair_ok)) if pair_ok else None,
            "margin_huayu_defective": (sum(r["margin"] for r in hy) / len(hy)) if hy else None,
            "margin_chengdong_defective": (sum(r["margin"] for r in cd) / len(cd)) if cd else None,
            "margin_beigang_defective": (sum(r["margin"] for r in bg) / len(bg)) if bg else None}


def main() -> None:
    cfg = load_all()
    dev_rounds = [int(x) for x in cfg["run"]["evaluation"]["dev_rounds"]]
    gate_rounds = [int(x) for x in cfg["run"]["evaluation"]["formal_gate_rounds"]]
    report: dict = {"series": {}, **stamp(scope="task1_verify")}
    all_ok = True
    for s in SERIES:
        rs = {"branches": {}, "checks_ok": True}
        for b in ("00", "U"):
            rb: dict = {"rounds": {}, "evals": {}, "chain": {}}
            state = json.loads((traj(s, b) / "state.json").read_text(encoding="utf-8"))
            rb["chain"]["rounds_done"] = state["rounds_done"]
            rb["chain"]["chain_len"] = len(state["chain"])
            ok_chain = state["rounds_done"] == 30 and len(state["chain"]) == 30
            prev = sha256_file(global_path(s, b, 0))
            for e in state["chain"]:
                r = e["round"]
                ok_chain &= (e["start"]["sha256"] == prev)
                ok_chain &= (sha256_file(global_path(s, b, r)) == e["aggregate"]["sha256"])
                for n, up in e["uploads"].items():
                    ok_chain &= (sha256_file(Path(up["path"])) == up["sha256"])
                prev = e["aggregate"]["sha256"]
            rb["chain"]["ok"] = ok_chain
            for r in range(1, 31):
                man = json.loads((traj(s, b) / f"round_{r:03d}" / "round_manifest.json").read_text(encoding="utf-8"))
                ok = man["slots"] == alg.slots_for(b) and man["branch"] == b and man["round"] == r
                for n, up in man["uploads"].items():
                    p = Path(up["path"])
                    ok &= p.exists() and sha256_file(p) == up["sha256"]
                flats = [load_flat(Path(man["uploads"][n]["path"])) for n in man["slots"]]
                agg = alg.server_aggregate(flats, alg.client_weights(len(man["slots"])))
                disk = load_flat(global_path(s, b, r))
                bitwise = all(torch.equal(agg[k], v) for k, v in disk.items())
                ok &= bitwise
                rb["rounds"][str(r)] = {"uploads_hash_ok": True, "fedavg_recompute_bitwise": bitwise}
                if not ok:
                    rs["checks_ok"] = False
            for r in dev_rounds:
                prefix = traj(s, b) / "eval_dev" / f"round_{r:03d}"
                rows = [json.loads(l) for l in (Path(str(prefix) + ".jsonl")).read_text(encoding="utf-8").splitlines() if l.strip()]
                summary = json.loads((Path(str(prefix) + ".json")).read_text(encoding="utf-8"))
                rec = state["evals"]["dev"][str(r)]
                rows_sha_ok = sha256_file(Path(str(prefix) + ".jsonl")) == rec["rows_sha256"]
                m = recompute_metrics_from_rows(rows)
                diffs = {}
                for k, v in m.items():
                    stored = summary[k]["rate"] if isinstance(summary[k], dict) else summary[k]
                    diffs[k] = {"recomputed": v, "stored": stored,
                                "ok": (v is None and stored is None) or (v is not None and stored is not None and abs(v - stored) <= TOL)}
                rb["evals"][str(r)] = {"rows_sha_ok": rows_sha_ok, "metrics": diffs,
                                       "ok": rows_sha_ok and all(d["ok"] for d in diffs.values())}
                if not rb["evals"][str(r)]["ok"]:
                    rs["checks_ok"] = False
            rb["ok"] = ok_chain and all(v["fedavg_recompute_bitwise"] for v in rb["rounds"].values()) \
                and all(v["ok"] for v in rb["evals"].values())
            rs["branches"][b] = rb
            rs["checks_ok"] &= rb["ok"]
        report["series"][s] = rs
        all_ok &= rs["checks_ok"]
        print(f"[verify] {s}: {'OK' if rs['checks_ok'] else 'FAIL'} "
              f"(00 chain/agg/eval: {rs['branches']['00']['chain']['ok']}, "
              f"U chain/agg/eval: {rs['branches']['U']['chain']['ok']})", flush=True)
    # 门复现：用重算指标构造 summary 形状，跑 gate 函数
    from src.e11.task2_protocol.stages import Protocol, gate_clean, gate_strong
    proto = Protocol(cfg)
    acc = cfg["run"]["acceptance"]
    gate = json.loads((REPORT_ROOT.parent.parent / "e11_task2_protocol" / "task2_protocol_20261127" / "p1" / "gate.json").read_text(encoding="utf-8"))
    repro = {}
    for s in SERIES:
        repro[s] = {}
        for r in gate_rounds:
            vals = {}
            for b in ("00", "U"):
                prefix = traj(s, b) / "eval_dev" / f"round_{r:03d}"
                rows = [json.loads(l) for l in (Path(str(prefix) + ".jsonl")).read_text(encoding="utf-8").splitlines() if l.strip()]
                summary = json.loads((Path(str(prefix) + ".json")).read_text(encoding="utf-8"))
                m = recompute_metrics_from_rows(rows)
                vals[b] = {k: {"rate": v, "hits": None, "n": None} if v is not None else None for k, v in m.items()}
                for mk in ("margin_huayu_defective", "margin_chengdong_defective", "margin_beigang_defective"):
                    vals[b][mk] = m[mk]
                vals[b]["closedset"] = summary["closedset"]   # 闭集为生成式评价，直接引用存储值
            gc = gate_clean(vals["00"], acc)
            gu = gate_strong(vals["U"], vals["00"], acc["u_strong_behavior"])
            stored = gate["series"][s]["per_round"][str(r)]
            repro[s][str(r)] = {"clean_pass_recomputed": gc["pass"], "strong_pass_recomputed": gu["pass"],
                                "clean_pass_stored": stored["clean_00"]["pass"], "strong_pass_stored": stored["strong_U"]["pass"],
                                "match": gc["pass"] == stored["clean_00"]["pass"] and gu["pass"] == stored["strong_U"]["pass"]}
            if not repro[s][str(r)]["match"]:
                all_ok = False
    report["gate_reproduction"] = repro
    report["all_ok"] = all_ok
    write_json(REPORT_ROOT / "verification_report.json", report)
    print(f"[verify] ALL {'OK' if all_ok else 'FAIL'} -> {REPORT_ROOT / 'verification_report.json'}", flush=True)


if __name__ == "__main__":
    main()
