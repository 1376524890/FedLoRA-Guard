"""E10 补充实验一构造：十客户端同轮四组对照——正常替代上传版（2026-11-23）。

背景：P7 的 00/10/01 分支中，缺席构造槽位为零因子（p7_construct.py 槽位映射将
背景轨迹的 client_00/01 上传弃用）。本脚本把弃用的原始正常上传恢复到槽 0/1，
从与 P7 完全相同的十槽等权 0.1 聚合出发，构建四组对照：

  11r: 槽0=构造a, 槽1=构造b（重建 P7 的 11 分支，作闭合核验，不重复评估 90 轮）
  00n: 槽0=client_00 原始正常上传, 槽1=client_01 原始正常上传
       （数学上 ≡ 该轨迹该轮的真实 FedAvg 全良性聚合，与归档 global_adapter 核验）
  10n: 槽0=构造a, 槽1=client_01 原始正常上传
  01n: 槽0=client_00 原始正常上传, 槽1=构造b

正常替代上传的可追溯性：来自同一背景轨迹、同一轮、同一轮初模型
（received_global_adapter）下训练的 client_00/01 final_adapter，即 P7 构造时
被弃用的上传（construct_report.json 的 dropped_uploads 字段）。

核验（每轮记录于 construct_report.json）：
  - 11r 与归档 branches/11/adapter.safetensors 的最大绝对差（预期 0，同序同码）；
  - 00n 与轨迹该轮 global_adapter.safetensors 的最大绝对差（预期 ~1e-9 浮点序噪声）。

另建 _base/branches/base/：全零 LoRA 适配器（rank8 α16，s·0=0），
使基础模型评估走与分支完全相同的代码路径（load_model_with_adapter）。

产出：artifacts/e10/p7_supp/<series>/<bg>_r<rd>/branches/{00n,10n,01n,11r}/
      artifacts/e10/p7_supp/_base/branches/base/
      artifacts/e10/p7_supp/construct_summary.json（含每轮核验数与输入 sha256）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(4)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.e10.branches import save_branches  # noqa: E402
from src.e10.lora_algebra import (from_pairs, fro_norm_lowrank,  # noqa: E402
                                  module_pairs)

RANK, ALPHA, S = 8, 16, 2.0
WEIGHTS = [0.1] * 10
BG_SLOTS = [f"client_{i:02d}" for i in range(2, 10)]

_hash_cache: dict[str, str] = {}


def sha256_of(path: Path) -> str:
    key = str(path)
    if key not in _hash_cache:
        h = hashlib.sha256()
        with open(key, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        _hash_cache[key] = h.hexdigest()
    return _hash_cache[key]


def load_pairs(path: Path) -> dict:
    from safetensors.torch import load_file
    return module_pairs(dict(load_file(str(path))))


def max_abs_diff_pairs(p1: dict, p2: dict) -> float:
    d = 0.0
    for mod in p1:
        d = max(d, (p1[mod][0] - p2[mod][0]).abs().max().item(),
                (p1[mod][1] - p2[mod][1]).abs().max().item())
    return d


def agg_norm(pairs: dict) -> float:
    return (sum(fro_norm_lowrank(a, b) ** 2 for a, b in pairs.values())) ** 0.5


def build_one(series: str, bg_run: str, rd: int, out_root: Path) -> dict:
    rd_dir = ROOT / "artifacts/runs" / bg_run / "checkpoints" / f"round_{rd:03d}"
    p7_dir = ROOT / "artifacts/e10/p7" / series / f"{bg_run}_r{rd}"

    c00_p = rd_dir / "clients/client_00/final_adapter.safetensors"
    c01_p = rd_dir / "clients/client_01/final_adapter.safetensors"
    ca_p = p7_dir / "constructs/a/adapter.safetensors"
    cb_p = p7_dir / "constructs/b/adapter.safetensors"
    for p in (c00_p, c01_p, ca_p, cb_p):
        if not p.exists():
            raise FileNotFoundError(p)

    bg_paths = [rd_dir / "clients" / c / "final_adapter.safetensors" for c in BG_SLOTS]
    bg_list = [load_pairs(p) for p in bg_paths]
    c00, c01 = load_pairs(c00_p), load_pairs(c01_p)
    ca, cb = load_pairs(ca_p), load_pairs(cb_p)

    from src.e10.lora_algebra import aggregate_lora_factors
    b11r = aggregate_lora_factors([ca, cb] + bg_list, WEIGHTS)
    b00n = aggregate_lora_factors([c00, c01] + bg_list, WEIGHTS)
    b10n = aggregate_lora_factors([ca, c01] + bg_list, WEIGHTS)
    b01n = aggregate_lora_factors([c00, cb] + bg_list, WEIGHTS)

    # 核验 1：11r vs 归档 11
    saved11 = load_pairs(p7_dir / "branches/11/adapter.safetensors")
    diff11 = max_abs_diff_pairs(b11r, saved11)
    # 核验 2：00n vs 轨迹该轮真实 FedAvg 聚合
    traj_global = load_pairs(rd_dir / "global_adapter.safetensors")
    diff00 = max_abs_diff_pairs(b00n, traj_global)

    out = out_root / series / f"{bg_run}_r{rd}"
    prov = {
        "series": series, "background_run": bg_run, "round": rd,
        "protocol": "normal_replacement_branch",
        "weights": WEIGHTS, "rank": RANK, "alpha": ALPHA, "scaling": S,
        "slot_map": {
            "00n": "slot0=client_00_original, slot1=client_01_original",
            "10n": "slot0=construct_a, slot1=client_01_original",
            "01n": "slot0=client_00_original, slot1=construct_b",
            "11r": "slot0=construct_a, slot1=construct_b (rebuild of P7 11)",
        },
        "normal_upload_provenance": {
            "client_00": {"path": str(c00_p.relative_to(ROOT)),
                          "sha256": sha256_of(c00_p),
                          "trained_from": str((rd_dir / "received_global_adapter.safetensors").relative_to(ROOT))},
            "client_01": {"path": str(c01_p.relative_to(ROOT)),
                          "sha256": sha256_of(c01_p),
                          "trained_from": str((rd_dir / "received_global_adapter.safetensors").relative_to(ROOT))},
        },
        "construct_provenance": {
            "a": {"path": str(ca_p.relative_to(ROOT)), "sha256": sha256_of(ca_p)},
            "b": {"path": str(cb_p.relative_to(ROOT)), "sha256": sha256_of(cb_p)},
        },
        "bg_upload_dir": str(rd_dir.relative_to(ROOT)),
        "verification": {
            "rebuild_11_vs_archived_max_abs_diff": diff11,
            "rebuild_00n_vs_trajectory_global_max_abs_diff": diff00,
        },
        "upload_factor_norms": {
            "client_00_original": agg_norm(c00), "client_01_original": agg_norm(c01),
            "construct_a": agg_norm(ca), "construct_b": agg_norm(cb),
        },
    }
    branches = {name: {"adapter": from_pairs(pairs), "protocol": "normal_replacement_branch",
                       "rank": RANK, "alpha": ALPHA}
                for name, pairs in (("00n", b00n), ("10n", b10n),
                                    ("01n", b01n), ("11r", b11r))}
    save_branches(branches, out / "branches",
                  {k: prov[k] for k in ("series", "background_run", "round",
                                        "protocol", "weights", "slot_map")})
    (out / "construct_report.json").write_text(
        json.dumps(prov, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"series": series, "bg": bg_run, "round": rd,
            "diff_11_rebuild": diff11, "diff_00n_vs_global": diff00}


def build_base(out_root: Path) -> None:
    """全零 LoRA = 基础模型，走同一评估代码路径。"""
    tmpl = ROOT / ("artifacts/e10/p7/pku_s17/e10bg_pku_s17_00_r1/"
                   "constructs/a/adapter.safetensors")
    from safetensors.torch import load_file, save_file
    zero = {k: torch.zeros_like(v) for k, v in dict(load_file(str(tmpl))).items()}
    d = out_root / "_base/branches/base"
    d.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in zero.items()},
              str(d / "adapter.safetensors"))
    (d / "branch.json").write_text(json.dumps(
        {"protocol": "zero_lora_equals_base_model", "rank": RANK, "alpha": ALPHA,
         "scaling": S, "model": "/home/marktom/models/qwen3-1.7b",
         "note": "s·B A = 0；输出与不挂适配器的基座一致（同一 evaluate 代码路径）"},
        ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/e10/p7_supp")
    ap.add_argument("--series", default=None)
    args = ap.parse_args()
    out_root = ROOT / args.out
    t0 = time.time()
    construct = json.loads((ROOT / "artifacts/e10/p7/construct_summary.json")
                           .read_text(encoding="utf-8"))
    rounds = [m for m in construct["per_round"]
              if not args.series or m["series"] == args.series]
    rounds.sort(key=lambda m: (0 if m["series"].startswith("dolly") else 1,
                               m["series"], m["bg"], m["round"]))
    results = []
    for i, m in enumerate(rounds):
        results.append(build_one(m["series"], m["bg"], m["round"], out_root))
        if (i + 1) % 15 == 0:
            print(f"[supp-construct] {i+1}/{len(rounds)} | {time.time()-t0:.0f}s", flush=True)
    build_base(out_root)
    summary = {
        "rounds_total": len(results),
        "rebuild_11_vs_archived_max_abs_diff": max(r["diff_11_rebuild"] for r in results),
        "rebuild_00n_vs_trajectory_global_max_abs_diff": max(r["diff_00n_vs_global"] for r in results),
        "weights": WEIGHTS, "rank": RANK, "alpha": ALPHA,
        "unique_input_files_hashed": len(_hash_cache),
        "elapsed_s": round(time.time() - t0, 1), "per_round": results,
    }
    (out_root / "construct_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[supp-construct] 完成 {len(results)} 轮 | 11 重建最大差 "
          f"{summary['rebuild_11_vs_archived_max_abs_diff']:.2e} | 00n 与轨迹聚合最大差 "
          f"{summary['rebuild_00n_vs_trajectory_global_max_abs_diff']:.2e} | "
          f"{summary['elapsed_s']}s")


if __name__ == "__main__":
    main()
