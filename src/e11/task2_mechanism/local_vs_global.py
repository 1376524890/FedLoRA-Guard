"""任务三：U 本地上传 vs 聚合 vs 正常均值 vs Task1 参考的逐轮独立评价（GPU 推理，无训练）。

模型种类（每轮、同一 dev 评价集、与 P1 完全相同的评价路径 full_eval）：
  U_upload : traj_U 第 r 轮 U 的真实上传
  agg10    : traj_U 第 r 轮真实十客户端聚合（global_r，任务一已核验 = 真实 FedAvg 复算）
  n9_mean  : traj_U 第 r 轮 N1..N9 九个真实上传的等权真实 FedAvg（服务器实现，无 U）
  t1_ref   : Task1 B_huayu_first T1（每系列一次；条件行为参考）
范围：U_upload/agg10 全部 30 轮；n9_mean 取动态轮 {1,2,5,10,15,20,25,30}；t1_ref ×1。
可续跑：meta.json 存在且输入哈希一致则跳过。SeriesRuntime 的 art_root 指向分析根，
不向协议 run 目录写任何字节。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.e11.task2_mechanism.common import (ANALYSIS_ROOT, REPORT_ROOT, SERIES, T1_DIR,  # noqa: E402
                                            input_hashes, stamp, traj, write_json)
from src.e11.task2_protocol import algebra as alg  # noqa: E402
from src.e11.task2_protocol.config import load_all, sha256_file  # noqa: E402
from src.e11.task2_protocol.engine import SeriesRuntime  # noqa: E402
from src.e11.task2_protocol.evaluation import full_eval  # noqa: E402

N9_ROUNDS = (1, 2, 5, 10, 15, 20, 25, 30)


def n9_flat(rt: SeriesRuntime, s: str, r: int) -> tuple[dict, dict]:
    """九正常客户端等权真实 FedAvg；缓存到分析根并记录输入哈希。"""
    fdir = ANALYSIS_ROOT / s / "flats"
    fpath = fdir / f"n9_mean_round_{r:03d}.safetensors"
    mpath = fdir / f"n9_mean_round_{r:03d}.meta.json"
    ups = {f"N{j}": traj(s, "U") / f"round_{r:03d}" / "uploads" / f"N{j}.safetensors" for j in range(1, 10)}
    want = input_hashes(ups)
    if mpath.exists() and fpath.exists():
        import json
        meta = json.loads(mpath.read_text(encoding="utf-8"))
        if meta["inputs"] == want and sha256_file(fpath) == meta["output_sha256"]:
            return rt.load_flat(fpath), meta
    flats = [rt.load_flat(p) for p in ups.values()]
    agg = alg.server_aggregate(flats, [1.0 / 9] * 9)
    saved = rt.save_flat(fpath, agg, metadata={"kind": "n9_mean", "series": s, "round": str(r)})
    meta = {"inputs": want, "output_sha256": saved["sha256"], "state_digest": saved["state_digest"],
            "aggregator": "src.fed.aggregate.fedavg(real)", "weights": [1.0 / 9] * 9, **stamp(series=s, round=r)}
    write_json(mpath, meta)
    return rt.load_flat(fpath), meta


def work_items() -> list[dict]:
    items = []
    for s in SERIES:
        for r in range(1, 31):
            items.append({"series": s, "round": r, "kind": "U_upload",
                          "flat": traj(s, "U") / f"round_{r:03d}" / "uploads" / "U.safetensors"})
            items.append({"series": s, "round": r, "kind": "agg10", "flat": traj(s, "U") / f"global_{r:03d}.safetensors"})
        for r in N9_ROUNDS:
            items.append({"series": s, "round": r, "kind": "n9_mean", "flat": None})
        items.append({"series": s, "round": 0, "kind": "t1_ref",
                      "flat": T1_DIR / s / "B_huayu_first" / "model_round_001.safetensors"})
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--shards", type=int, default=3)
    args = ap.parse_args()
    cfg = load_all()
    items = [it for i, it in enumerate(work_items()) if i % args.shards == args.shard]
    print(f"[shard {args.shard}/{args.shards}] gpu{args.gpu}: {len(items)} evals", flush=True)
    rt = SeriesRuntime(cfg, "dolly_s42", args.gpu, art_root=ANALYSIS_ROOT / f"runtime_gpu{args.gpu}")
    rt.ensure_model()
    rows = rt.inputs.eval_rows("dev")
    t00 = time.time()
    for k, it in enumerate(items):
        s, r, kind = it["series"], it["round"], it["kind"]
        out = ANALYSIS_ROOT / s / kind / f"round_{r:03d}"
        mpath = out / "meta.json"
        inputs = {"flat": it["flat"]} if it["flat"] else {f"N{j}": traj(s, "U") / f"round_{r:03d}" / "uploads" / f"N{j}.safetensors" for j in range(1, 10)}
        want = input_hashes({k2: v for k2, v in inputs.items()})
        if mpath.exists():
            import json
            try:
                old = json.loads(mpath.read_text(encoding="utf-8"))
                if old.get("inputs") == want:
                    print(f"[{k+1}/{len(items)}] skip {s} {kind} r{r}", flush=True)
                    continue
            except Exception:
                pass
        t0 = time.time()
        if kind == "n9_mean":
            flat, nmeta = n9_flat(rt, s, r)
        else:
            flat = rt.load_flat(it["flat"])
        summary = full_eval(rt.model, rt.tokenizer, flat, rows, rt.device, rt.tcfg, "dev", out / "eval")
        y = summary["Y_huayu_release"]
        write_json(mpath, {"series": s, "round": r, "kind": kind, "inputs": want, "eval_dir": str(out),
                           "summary_headline": {"Y_huayu": y["rate"], "FPR_cd": summary["FPR_chengdong"]["rate"],
                                                "FPR_bg": summary["FPR_beigang"]["rate"],
                                                "pair": summary["Pair_exact"]["rate"],
                                                "margin_huayu": float(summary["margin_huayu_defective"])},
                           **stamp()})
        print(f"[{k+1}/{len(items)}] {s} {kind} r{r}: huayu {y['hits']}/{y['n']} cd {summary['FPR_chengdong']['hits']} "
              f"bg {summary['FPR_beigang']['hits']} pair {summary['Pair_exact']['hits']} "
              f"({time.time()-t0:.0f}s; elapsed {time.time()-t00:.0f}s)", flush=True)
    rt.release_model()
    print(f"[shard {args.shard}] DONE in {time.time()-t00:.0f}s", flush=True)


if __name__ == "__main__":
    main()
