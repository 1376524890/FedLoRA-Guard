"""P3 数学搜索（设计 §4 阶段三；执行说明工作四）：225 组 (λ_A, λ_B) 在真实影子上传张量上的离线筛选。

输入：某个开发种子、某个影子轮次的已序列化回读张量 G^N_{r-1}、U、N0..N9；冻结检测阈值。
对每个候选：
- F6–F9 构造 A/B（FP64→FP32）；
- 三种尺度 F11/F12（A、B 各自）与同轮正常客户端中位数之比须落在尺度带内；
- 真实 FedAvg 得 10/01/11 聚合；11 与 U 真实聚合的数值误差门槛；
- 单支行为（10/01）与联合行为（11）用固定探针的批量 margin 评价（缓存键=聚合 state_digest + 探针哈希）；
- 检测评分：Norm/Cosine（冻结阈值）在 10（A）、01（B）、11（A、B）上的标记情况；U 在 U 分支被标记情况；
  FoolsGold 在搜索中仅作冷启动参考信息，正式 FoolsGold 结论来自持续轨迹的隔离历史。
候选须在全部开发单元（种子 × 影子轮次）通过全部门槛；选择规则预先指定：最小化最坏 |ln 尺度比|，再最小化单支华宇放行率。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.e11.task2_protocol import algebra as alg
from src.e11.task2_protocol.config import sha256_file
from src.e11.task2_protocol.detection import detection_row
from src.e11.task2_protocol.evaluation import batched_margins, margin_summary
from src.e11.task2_protocol.engine import SeriesRuntime
from src.e11.task2_protocol.inputs import state_digest
from src.errors import ConfigurationError
from src.io import atomic_json
from src.train.model import apply_adapter, cpu_adapter_state


def lambda_grid(cfg_all: dict[str, Any]) -> list[float]:
    sp = cfg_all["protocol"]["split"]
    n = int(cfg_all["run"]["split_grid"]["lambda_count_per_axis"])
    if n * n != int(sp["candidate_count"]):
        raise ConfigurationError(f"grid {n}x{n} != candidate_count {sp['candidate_count']}")
    lo, hi = float(sp["lambda_a"]["min"]), float(sp["lambda_a"]["max"])
    if (float(sp["lambda_b"]["min"]), float(sp["lambda_b"]["max"])) != (lo, hi):
        raise ConfigurationError("lambda_a / lambda_b ranges differ")
    return [float(x) for x in np.linspace(lo, hi, n)]


class ShadowSearcher:
    def __init__(self, rt: SeriesRuntime, thresholds: dict[str, float], r: int) -> None:
        self.rt, self.thresholds, self.r = rt, thresholds, r
        self.acc = rt.run_cfg["acceptance"]
        self.band = tuple(float(x) for x in self.acc["scale_band"])
        self.rdir = rt.series_dir / "shadow" / f"round_{r:03d}"
        self.man = json.loads((self.rdir / "shadow_manifest.json").read_text(encoding="utf-8"))
        self.probe = rt.inputs.probe_rows()
        self.s = rt.inputs.t1_cfg["lora"]["alpha"] / rt.inputs.t1_cfg["lora"]["rank"]
        self.cache_path = rt.series_dir / "search" / f"margins_cache_r{r:03d}.json"
        self.cache = self._load_cache()
        rt.ensure_model()

    def _load_cache(self) -> dict[str, Any]:
        header = {"probe_sha256": sha256_file(self.rt.inputs.probe_file),
                  "model_files": {p.name: sha256_file(p) for p in sorted(self.rt.inputs.model_dir.glob("*.safetensors"))},
                  "batch": int(self.rt.run_cfg["runtime"].get("eval_batch", 32))}
        if self.cache_path.exists():
            c = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if c["header"] != header:
                raise ConfigurationError("Margins cache header (probe/model/batch) changed")
            return c
        return {"header": header, "entries": {}}

    def _margins(self, agg_flat: dict[str, torch.Tensor], *, branch: str, la: float, lb: float) -> dict[str, Any]:
        key = state_digest(agg_flat)  # 模型哈希 = 聚合适配器 digest；输入哈希在缓存头
        if key in self.cache["entries"]:
            return self.cache["entries"][key]["margins_summary"]
        apply_adapter(self.rt.model, agg_flat)
        before = state_digest(cpu_adapter_state(self.rt.model))
        margins = batched_margins(self.rt.model, self.rt.tokenizer, self.probe, self.rt.device,
                                  batch=int(self.rt.run_cfg["runtime"].get("eval_batch", 32)))
        if state_digest(cpu_adapter_state(self.rt.model)) != before:
            raise RuntimeError("Batched margins modified adapter")
        summ = margin_summary(self.probe, margins)
        self.cache["entries"][key] = {"branch": branch, "lambda_a": la, "lambda_b": lb, "aggregate_digest": key,
                                      "margins_summary": summ}
        return summ

    def _gate(self, summ: dict[str, Any], kind: str) -> dict[str, Any]:
        acc = self.acc
        hu, cd, bg, nm = summ.get("defective:huayu", {}), summ.get("defective:chengdong", {}), \
            summ.get("defective:beigang", {}), {}
        for k, v in summ.items():
            if k.startswith("normal:"):
                nm[k] = v
        hu_rate = hu.get("release_rate"); hu_margin = hu.get("margin_mean")
        normal_rates = [v["release_rate"] for v in nm.values()] if nm else []
        checks = {
            "huayu_release": (hu_rate <= acc["single_branch_weak"]["huayu_release_max"]) if kind == "single"
                else (hu_rate >= acc["joint_strong"]["huayu_release_min"]),
            "huayu_margin": (hu_margin < acc["single_branch_weak"]["margin_huayu_max"]) if kind == "single"
                else (hu_margin > acc["joint_strong"]["margin_huayu_min"]),
            "chengdong_release_le_max": cd.get("release_rate", 1.0) <= acc["single_branch_weak"]["chengdong_fpr_max"],
            "beigang_release_le_max": bg.get("release_rate", 1.0) <= acc["single_branch_weak"]["beigang_fpr_max"],
            "normal_release_ge_min": bool(normal_rates) and min(normal_rates) >= acc["single_branch_weak"]["normal_release_min"],
        }
        return {"checks": checks, "pass": all(checks.values()), "huayu_release_rate": hu_rate,
                "huayu_margin_mean": hu_margin, "normal_release_rates": normal_rates}

    def run(self) -> dict[str, Any]:
        rt, r = self.rt, self.r
        man = self.man
        received = rt.load_flat(Path(man["frozen_reference"]["path"]))
        flats = {n: rt.load_flat(Path(u["path"])) for n, u in man["uploads"].items()}
        for n, u in man["uploads"].items():
            if sha256_file(Path(u["path"])) != u["sha256"]:
                raise ConfigurationError(f"Shadow upload changed on disk: {n}")
        G = alg.to_pairs(received)
        U = alg.to_pairs(flats["U"])
        N = {j: alg.to_pairs(flats[f"N{j}"]) for j in range(10)}
        p = float(rt.proto["protocol"]["client_weight"])
        D = alg.D_factors(G, U, N[1], p)
        normal_scales = [alg.three_scales(N[j], G, self.s) for j in range(10)]
        med = {k: float(np.median([x[k] for x in normal_scales])) for k in ("N_eff_BA", "F_P", "F_Q")}
        # 候选无关量：U 聚合 = 11 聚合（任意 λ）；U 分支检测行（含 Gram 显式核验）
        slots_U = alg.slots_for("U")
        agg_U = alg.server_aggregate([flats[n] for n in slots_U], alg.client_weights(10))
        joint = self._margins(agg_U, branch="11", la=-1.0, lb=-1.0)
        joint_gate = self._gate(joint, "joint")
        row_U, _ = detection_row(slots_U, ["U"] + ["N"] * 9, [flats[n] for n in slots_U], received,
                                 round_index=r, scaling=self.s)
        u_flagged = {m: row_U[m]["U"] > self.thresholds[m] for m in ("Norm", "Cosine")}
        if not any(u_flagged.values()):
            u_flagged = {m: row_U[m]["U"] > self.thresholds[m] for m in ("Norm", "Cosine", "FoolsGold")}

        grid = lambda_grid({"protocol": rt.proto, "run": rt.run_cfg})
        acc_num = self.acc["numerical"]
        candidates = []
        for la in grid:
            for lb in grid:
                A64, B64 = alg.construct_ab(G, D, la, lb)
                A32, B32 = alg.to_fp32(A64), alg.to_fp32(B64)
                sc_A, sc_B = alg.three_scales(A32, G, self.s), alg.three_scales(B32, G, self.s)
                ratios = {f"{X}:{k}": sc_X[k] / med[k] for X, sc_X in (("A", sc_A), ("B", sc_B))
                          for k in ("N_eff_BA", "F_P", "F_Q")}
                scale_pass = all(self.band[0] <= v <= self.band[1] for v in ratios.values())
                worst_log = max(abs(math.log(v)) for v in ratios.values())
                flats_c = {**flats, "A": alg.flatten(A32), "B": alg.flatten(B32)}
                agg = {b: alg.server_aggregate([flats_c[n] for n in alg.slots_for(b)], alg.client_weights(10))
                       for b in ("10", "01", "11")}
                ident = alg.compare_aggregates("11_vs_U", alg.to_pairs(agg["11"]), alg.to_pairs(agg_U), self.s,
                                               ba_tol=acc_num["ba_rel_err_max"], factor_tol=acc_num["factor_max_abs"])
                single = {b: self._margins(agg[b], branch=b, la=la, lb=lb) for b in ("10", "01")}
                gates = {b: self._gate(single[b], "single") for b in ("10", "01")}
                flagged = {}
                for b in ("10", "01", "11"):
                    slots = alg.slots_for(b)
                    roles = ["N" if n.startswith("N") else n for n in slots]
                    row, _ = detection_row(slots, roles, [flats_c[n] for n in slots], received, round_index=r,
                                           scaling=self.s, crosscheck=False)
                    flagged[b] = {m: {n: row[m][n] > self.thresholds[m] for n in slots if not n.startswith("N")}
                                  for m in ("Norm", "Cosine")}
                    flagged[b]["FoolsGold_cold"] = {n: row["FoolsGold"][n] > self.thresholds["FoolsGold"]
                                                    for n in slots if not n.startswith("N")}
                ev_pass = all(not any(flagged[b][m][x] for m in ("Norm", "Cosine") for x in flagged[b][m])
                              for b in ("10", "01", "11"))
                rec = {"lambda_a": la, "lambda_b": lb, "scales": {"A": sc_A, "B": sc_B}, "normal_medians": med,
                       "ratios": ratios, "scale_pass": scale_pass, "worst_abs_log_ratio": worst_log,
                       "numerical": ident, "single": {b: gates[b] for b in gates}, "single_pass": all(g["pass"] for g in gates.values()),
                       "joint_pass": joint_gate["pass"], "joint": joint_gate,
                       "detection_flagged": flagged, "evasion_pass_norm_cosine": ev_pass,
                       "max_single_huayu_release": max(gates[b]["huayu_release_rate"] for b in gates),
                       "single_summaries": single}
                rec["pass"] = bool(scale_pass and ident["pass"] and rec["single_pass"] and rec["joint_pass"]
                                   and ev_pass and any(u_flagged.values()))
                candidates.append(rec)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(self.cache_path, self.cache, overwrite=True)
        passing = [c for c in candidates if c["pass"]]
        passing.sort(key=lambda c: (c["worst_abs_log_ratio"], c["max_single_huayu_release"], c["lambda_a"], c["lambda_b"]))
        unit = {"series": rt.series, "round": r, "n_candidates": len(candidates), "n_pass": len(passing),
                "normal_scale_medians": med, "joint_gate_candidate_independent": joint_gate,
                "u_flagged_by": [m for m, f in u_flagged.items() if f], "u_flagged_any": any(u_flagged.values()),
                "thresholds": self.thresholds, "candidates": candidates,
                "unit_feasible": bool(joint_gate["pass"] and any(u_flagged.values()))}
        atomic_json(self.rdir / "search_unit.json", unit, overwrite=True)
        return unit


def select_lambda(units: list[dict[str, Any]]) -> dict[str, Any]:
    """跨开发单元选择：候选须在所有单元通过；目标函数 = 最坏 |ln 尺度比|，再取单支华宇放行率更低者。"""
    grid_keys = {(c["lambda_a"], c["lambda_b"]) for c in units[0]["candidates"]}
    per_key = {k: [] for k in grid_keys}
    for u in units:
        for c in u["candidates"]:
            per_key[(c["lambda_a"], c["lambda_b"])].append(c)
    survivors, table = [], []
    for k, cs in per_key.items():
        ok = all(c["pass"] for c in cs)
        worst = max(c["worst_abs_log_ratio"] for c in cs)
        hu = max(c["max_single_huayu_release"] for c in cs)
        table.append({"lambda_a": k[0], "lambda_b": k[1], "pass_all_units": ok, "worst_abs_log_ratio": worst,
                      "max_single_huayu_release": hu})
        if ok:
            survivors.append((worst, hu, k))
    survivors.sort()
    return {"units": [{"series": u["series"], "round": u["round"], "n_pass": u["n_pass"], "unit_feasible": u["unit_feasible"]}
                      for u in units],
            "n_candidates": len(table), "n_pass_all_units": len(survivors),
            "selected_lambda_a": survivors[0][2][0] if survivors else None,
            "selected_lambda_b": survivors[0][2][1] if survivors else None,
            "selection_rule": "pass all gates on every development unit; minimize worst |ln scale ratio|; tie-break lower max single-branch huayu release",
            "table": sorted(table, key=lambda t: (not t["pass_all_units"], t["worst_abs_log_ratio"]))}
