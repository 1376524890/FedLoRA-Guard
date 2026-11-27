"""同轮配对因果对照（设计 §3.1 "同轮影子对照"；执行说明补充 2–4）。两步、λ 无关/有关分离：

步骤 A `run_round(r)`（λ 无关）：
- 从 00 分支的冻结参考全局 G^N_{r-1} 出发；正常上传 N0..N9 直接复用 00 分支第 r 轮已序列化上传（逐文件哈希核对）；
- 影子 U 从 G^N_{r-1} 真实训练（Task1 B_huayu_first 计划，消费顺序逐位断言）；
- 影子 00 = 真实 FedAvg(N0..N9) 须与 00 分支第 r 轮全局逐位一致；影子 U 分支真实聚合；
- 写 shadow/round_NNN/{U.safetensors, aggregate_U.safetensors, shadow_manifest.json}。
步骤 B `construct(r, λ_A, λ_B)`（λ 有关）：
- F6–F9 构造 A/B；10/01/11 真实 FedAvg；F10 恒等式、F13/F14 C_W 两路径、F11/F12 三尺度；
- 写 shadow/round_NNN/split_<λA>x<λB>/{A,B,aggregate_10,01,11}.safetensors + split_manifest.json。
影子结果只用于同轮因果比较，绝不反馈到持续轨迹。
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from src.e11.task2_protocol import algebra as alg
from src.e11.task2_protocol.config import sha256_file
from src.e11.task2_protocol.detection import detection_row
from src.e11.task2_protocol.engine import SeriesRuntime
from src.e11.task2_protocol.evaluation import full_eval
from src.e11.task2_protocol.inputs import git_head, state_digest
from src.errors import CheckpointError
from src.io import atomic_json

SHADOW_BRANCHES = ("U", "10", "01", "11")


def split_tag(lambda_ab: tuple[float, float]) -> str:
    return f"split_{lambda_ab[0]:g}x{lambda_ab[1]:g}"


class ShadowRunner:
    def __init__(self, rt: SeriesRuntime) -> None:
        self.rt = rt
        self.dir = rt.series_dir / "shadow"
        self.dir.mkdir(parents=True, exist_ok=True)

    def round_dir(self, r: int) -> Path:
        return self.dir / f"round_{r:03d}"

    def manifest(self, r: int) -> dict[str, Any]:
        return json.loads((self.round_dir(r) / "shadow_manifest.json").read_text(encoding="utf-8"))

    def _ref00(self) -> Path:
        return self.rt.series_dir / "traj_00"

    def reference_ready(self, r: int) -> bool:
        ref = self._ref00()
        return (ref / f"global_{r:03d}.safetensors").exists() and (ref / f"round_{r:03d}" / "round_manifest.json").exists()

    def _load_normal_uploads(self, r: int) -> tuple[dict[str, dict], dict[str, Any]]:
        m00 = json.loads((self._ref00() / f"round_{r:03d}" / "round_manifest.json").read_text(encoding="utf-8"))
        normals = {}
        for n in m00["slots"]:
            up = m00["uploads"][n]
            path = Path(up["path"])
            if sha256_file(path) != up["sha256"]:
                raise CheckpointError(f"00-branch upload changed on disk: {path}")
            normals[n] = {"path": str(path), "sha256": up["sha256"], "state_digest": up["state_digest"],
                          "consumed_sample_ids": up["consumed_sample_ids"], "client_index": up["client_index"]}
        if sorted(normals) != [f"N{j}" for j in range(10)]:
            raise CheckpointError("00 branch must have N0..N9")
        return normals, m00

    # ------------------------------------------------------------ 步骤 A：训练影子 U（λ 无关）
    def run_round(self, r: int) -> dict[str, Any]:
        mp = self.round_dir(r) / "shadow_manifest.json"
        if mp.exists():
            man = json.loads(mp.read_text(encoding="utf-8"))
            for u in man["uploads"].values():
                if sha256_file(Path(u["path"])) != u["sha256"]:
                    raise CheckpointError("Shadow upload changed on disk")
            return man
        if not self.reference_ready(r):
            raise CheckpointError(f"00 branch round {r} not finished for {self.rt.series}")
        rt = self.rt
        rt.ensure_model()
        rdir = self.round_dir(r)
        (rdir / "ckpt").mkdir(parents=True, exist_ok=True)
        normals, m00 = self._load_normal_uploads(r)
        start_path = Path(m00["start"]["path"])
        if sha256_file(start_path) != m00["start"]["sha256"]:
            raise CheckpointError("Frozen reference global changed on disk")
        start_flat = rt.load_flat(start_path)
        start = {"path": str(start_path), "sha256": m00["start"]["sha256"], "state_digest": state_digest(start_flat)}
        t0 = time.time()
        n_flats = [rt.load_flat(Path(normals[f"N{j}"]["path"])) for j in range(10)]
        shadow00 = alg.server_aggregate(n_flats)
        ref_global = rt.load_flat(self._ref00() / f"global_{r:03d}.safetensors")
        if any(not torch.equal(shadow00[k], ref_global[k]) for k in shadow00):
            raise CheckpointError("Shadow-00 aggregate is not bit-identical to 00-branch global")
        plan = rt.inputs.u_round_plan(r)
        res = rt.train_client(branch="shadowU", round_index=r, role="U", client_index=0,
                              records=rt.inputs.u_round_records(r), start_flat=start_flat,
                              ckpt_root=rdir / "ckpt" / "U")
        if res["consumed_sample_ids"] != plan["desired_ids"] or res["consumed_sample_ids"] != rt.inputs.task1_consumed_ids()[r]:
            raise CheckpointError(f"Shadow U r{r}: consumed order != Task1 manifest")
        saved = rt.save_flat(rdir / "U.safetensors", res["state"], metadata={"round": str(r), "role": "shadow_U"})
        uploads = dict(normals)
        uploads["U"] = {"path": saved["path"], "sha256": saved["sha256"], "state_digest": saved["state_digest"],
                        "consumed_sample_ids": res["consumed_sample_ids"], "client_index": 0,
                        "average_loss": res["average_loss"], "elapsed_seconds": res["elapsed_seconds"]}
        slots_U = alg.slots_for("U")
        agg_U = alg.server_aggregate([rt.load_flat(Path(uploads[n]["path"])) for n in slots_U], alg.client_weights(10))
        saved_U = rt.save_flat(rdir / "aggregate_U.safetensors", agg_U, metadata={"round": str(r), "branch": "U"})
        s = rt.inputs.t1_cfg["lora"]["alpha"] / rt.inputs.t1_cfg["lora"]["rank"]
        G = alg.to_pairs(start_flat)
        man = {"series": rt.series, "round": r, "frozen_reference": start, "uploads": uploads,
               "aggregate_U": {"slots": slots_U, "path": saved_U["path"], "sha256": saved_U["sha256"], "state_digest": saved_U["state_digest"]},
               "aggregate_00": {"path": str(self._ref00() / f"global_{r:03d}.safetensors"), "sha256": m00["aggregate"]["sha256"],
                                "state_digest": m00["aggregate"]["state_digest"], "recomputed_bit_identical": True},
               "scales_U": alg.three_scales(alg.to_pairs(rt.load_flat(Path(uploads["U"]["path"]))), G, s),
               "normal_uploads_source": "traj_00 same round (bit-identical reuse, hash-verified)",
               "fed_back_to_persistent_trajectory": False,
               "timing": {"round_s": time.time() - t0}, "code": {"commit": git_head()["commit"]}, "ts": time.time()}
        atomic_json(mp, man, overwrite=False)
        shutil.rmtree(rdir / "ckpt", ignore_errors=True)
        return man

    # ------------------------------------------------------------ 步骤 B：按 λ 构造与四分支聚合
    def construct(self, r: int, lambda_ab: tuple[float, float]) -> dict[str, Any]:
        rt = self.rt
        man = self.manifest(r)
        sdir = self.round_dir(r) / split_tag(lambda_ab)
        smp = sdir / "split_manifest.json"
        if smp.exists():
            return json.loads(smp.read_text(encoding="utf-8"))
        sdir.mkdir(parents=True, exist_ok=True)
        received = rt.load_flat(Path(man["frozen_reference"]["path"]))
        flats = {n: rt.load_flat(Path(u["path"])) for n, u in man["uploads"].items()}
        for n, u in man["uploads"].items():
            if sha256_file(Path(u["path"])) != u["sha256"]:
                raise CheckpointError(f"Shadow upload changed on disk: {n}")
        la, lb = (float(x) for x in lambda_ab)
        p = float(rt.proto["protocol"]["client_weight"])
        s = rt.inputs.t1_cfg["lora"]["alpha"] / rt.inputs.t1_cfg["lora"]["rank"]
        G, U, N1 = alg.to_pairs(received), alg.to_pairs(flats["U"]), alg.to_pairs(flats["N1"])
        D = alg.D_factors(G, U, N1, p)
        A64, B64 = alg.construct_ab(G, D, la, lb)
        A32, B32 = alg.to_fp32(A64), alg.to_fp32(B64)
        acc = rt.run_cfg["acceptance"]["numerical"]
        for name, X in (("A", A32), ("B", B32)):
            rt.save_flat(sdir / f"{name}.safetensors", alg.flatten(X), metadata={"round": str(r), "role": name, "constructed": "lambda_cross_split"})
        flats["A"], flats["B"] = alg.flatten(A32), alg.flatten(B32)
        aggregates = {"00": man["aggregate_00"], "U": man["aggregate_U"]}
        for b in ("10", "01", "11"):
            agg = alg.server_aggregate([flats[n] for n in alg.slots_for(b)], alg.client_weights(10))
            sv = rt.save_flat(sdir / f"aggregate_{b}.safetensors", agg, metadata={"round": str(r), "branch": b})
            aggregates[b] = {"slots": alg.slots_for(b), "path": sv["path"], "sha256": sv["sha256"], "state_digest": sv["state_digest"]}
        P = {b: alg.to_pairs(rt.load_flat(Path(aggregates[b]["path"]))) for b in ("00", "10", "01", "11", "U")}
        n_pairs = [alg.to_pairs(flats[f"N{j}"]) for j in range(10)]
        identity = {
            "lambda_a": la, "lambda_b": lb,
            "ab_sum_identity_fp64": alg.ab_sum_identity_error(G, U, N1, A64, B64, p),
            "aggregate_11_vs_U": alg.compare_aggregates("11_vs_U", P["11"], P["U"], s, ba_tol=acc["ba_rel_err_max"], factor_tol=acc["factor_max_abs"]),
            "linear_additivity_P11_minus_(P10+P01-P00)": alg.linear_additivity_check(P["00"], P["10"], P["01"], P["11"]),
            "C_W_paths": alg.compare_interaction_paths(P["00"], P["10"], P["01"], P["11"], s),
            "scales": {"A": alg.three_scales(A32, G, s), "B": alg.three_scales(B32, G, s), "U": man["scales_U"],
                       **{f"N{j}": alg.three_scales(n_pairs[j], G, s) for j in range(10)}},
        }
        identity["C_W_paths_pass"] = identity["C_W_paths"]["abs_diff_over_max_W"] <= acc["interaction_abs_diff_over_W_max"]
        identity["pass"] = bool(identity["aggregate_11_vs_U"]["pass"] and identity["C_W_paths_pass"])
        out = {"series": rt.series, "round": r, "lambda_ab": [la, lb], "split_dir": str(sdir), "aggregates": aggregates,
               "A": {"path": str(sdir / "A.safetensors"), "sha256": sha256_file(sdir / "A.safetensors")},
               "B": {"path": str(sdir / "B.safetensors"), "sha256": sha256_file(sdir / "B.safetensors")},
               "identity": identity, "fed_back_to_persistent_trajectory": False, "ts": time.time()}
        atomic_json(smp, out, overwrite=False)
        return out

    # ------------------------------------------------------------ 评价与检测
    def evaluate(self, r: int, lambda_ab: tuple[float, float], split: str = "dev") -> dict[str, Any]:
        rt = self.rt
        sdir = self.round_dir(r) / split_tag(lambda_ab)
        out = {}
        for b in SHADOW_BRANCHES:
            flat = rt.load_flat(sdir / f"aggregate_{b}.safetensors") if b != "U" else rt.load_flat(self.round_dir(r) / "aggregate_U.safetensors")
            out[b] = full_eval(rt.model, rt.tokenizer, flat, rt.inputs.eval_rows(split), rt.device, rt.tcfg, split, sdir / f"eval_{split}_{b}")
        ref = self._ref00() / f"eval_{split}" / f"round_{r:03d}.json"
        out["00"] = json.loads(ref.read_text(encoding="utf-8")) if ref.exists() else None
        atomic_json(sdir / f"eval_{split}_summary.json", out, overwrite=True)
        return out

    def detection_rows(self, r: int, lambda_ab: tuple[float, float], *, scaling: float,
                       history: dict[str, list] | None = None) -> tuple[dict[str, Any], dict[str, list]]:
        """四分支同轮检测行；FoolsGold 历史按 (series, "shadow", branch) 隔离，沿影子轮次序列累积。"""
        rt = self.rt
        sdir = self.round_dir(r) / split_tag(lambda_ab)
        man = self.manifest(r)
        received = rt.load_flat(Path(man["frozen_reference"]["path"]))
        flats = {n: rt.load_flat(Path(u["path"])) for n, u in man["uploads"].items()}
        flats["A"], flats["B"] = rt.load_flat(sdir / "A.safetensors"), rt.load_flat(sdir / "B.safetensors")
        history = dict(history or {})
        rows, new_hist = {}, {}
        for b in SHADOW_BRANCHES:
            slots = alg.slots_for(b)
            roles = ["N" if n.startswith("N") else n for n in slots]
            row, h = detection_row(slots, roles, [flats[n] for n in slots], received, round_index=r, scaling=scaling,
                                   fg_history=history.get(b, []))
            row["branch"] = b
            rows[b], new_hist[b] = row, h
        return rows, new_hist
