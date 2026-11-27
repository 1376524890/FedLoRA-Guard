"""P0 输入审计与 Task1 冻结输入的唯一读取入口（设计 §7、§8-P0；执行说明工作一）。

职责：
- 核对 Task1 的 T0、30 轮 B_huayu_first 消费清单、配置、模型文件、数据分区、评价集的哈希与内容；
- 核对 Task1 训练配置与 task2.yaml 协议逐项一致；核对核心训练源码与 Task1 时代提交一致；
- 提供 U 每轮 20 条有序记录（Task1 B_huayu_first 计划的精确恢复）与正常客户端分区记录；
- 任何来源不明、哈希不符或配置差异：写出 failed manifest 后立即抛错（不修补、不跳过）。
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from src.e11.task1 import build_round_plan, consumed_order  # noqa: E402  Task1 调度实现（只读复用）
from src.e11.task2_protocol.config import ROOT, load_all, resolve_path, sha256_file
from src.errors import CheckpointError, DataValidationError
from src.train.local import sample_schedule  # noqa: E402

TASK1_COMMIT = "d9fe1ef"  # Task1 时代提交（worktree FedLoRA-Guard-e11 HEAD），核心训练文件须与之一致
CORE_TRAINING_FILES = (
    "src/train/local.py", "src/train/model.py", "src/data/tokenize.py", "src/e11/task1.py",
    "src/e11/data.py", "src/e11/evaluate.py", "src/e10/lora_algebra.py", "src/e10/generation.py",
    "src/e10/evalcfg.py", "src/fed/aggregate.py", "src/io.py", "src/rng.py", "src/checkpoint.py",
    "configs/e11/task1_dynamics.yaml",
)


def state_digest(state: dict[str, torch.Tensor]) -> str:
    from src.io import state_digest as _digest
    return _digest(state)


def git_head(root: Path = ROOT) -> dict[str, Any]:
    def run(*cmd: str) -> str:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=root).stdout.strip()
    return {"commit": run("git", "rev-parse", "HEAD"), "branch": run("git", "branch", "--show-current"),
            "dirty_tracked": bool(run("git", "status", "--porcelain", "--untracked-files=no")),
            "untracked": [l for l in run("git", "status", "--porcelain").splitlines() if l.startswith("??")]}


def git_blob_same_as_commit(path: str, commit: str, root: Path = ROOT) -> bool:
    """当前工作区文件内容是否与历史提交中的 blob 完全一致。"""
    def run(*cmd: str) -> str:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=root).stdout.strip()
    historical = run("git", "rev-parse", f"{commit}:{path}")
    current = run("git", "hash-object", path)
    return bool(historical) and historical == current


class _Fail(Exception):
    pass


class Task1Inputs:
    """一个系列（如 dolly_s17）的 Task1 冻结输入 + Task2 协议参数。"""

    U_MODE = "B_huayu_first"

    def __init__(self, cfg_all: dict[str, Any], series: str) -> None:
        self.proto = cfg_all["protocol"]
        self.run_cfg = cfg_all["run"]
        self.series = series
        self.source, self.seed_key = series.rsplit("_", 1)
        self.seed = int(self.seed_key.lstrip("s"))
        paths = self.run_cfg["paths"]
        self.task1_root = resolve_path(paths["task1_artifacts"])
        self.task1_data = self.task1_root / "data"
        self.task1_reports = resolve_path(paths["task1_reports"])
        self.model_dir = resolve_path(paths["model_dir"])
        self.t1_cfg = self._load_task1_config()
        self.partition_file = self.task1_data / f"ordinary_split_{self.source}.json"
        self.records_file = self.task1_data / f"ordinary_records_{self.source}.jsonl"
        self.pairs_file = self.task1_data / "warehouse_pairs.json"
        self.eval_dev_file = self.task1_data / "eval_dev.jsonl"
        self.eval_test_file = self.task1_data / "eval_test.jsonl"
        self.probe_file = self.task1_data / "probe_dev.jsonl"
        self.schedule_file = self.task1_data / f"natural_schedule_{series}.jsonl"
        self.u_traj_dir = self.task1_root / series / self.U_MODE
        self.t0_path = self.u_traj_dir / "init_adapter.safetensors"
        self.manifest_path = self.u_traj_dir / "round_manifests.jsonl"
        self._records: dict[str, dict] | None = None
        self._partition: dict[str, Any] | None = None
        self._pairs: dict[int, dict] | None = None

    # ------------------------------------------------------------ 配置
    def _load_task1_config(self) -> dict[str, Any]:
        import yaml
        path = resolve_path(self.run_cfg["task1_config"])
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------ 数据加载（惰性、只读）
    @property
    def partition(self) -> dict[str, Any]:
        if self._partition is None:
            self._partition = json.loads(self.partition_file.read_text(encoding="utf-8"))
        return self._partition

    @property
    def records(self) -> dict[str, dict]:
        if self._records is None:
            recs = {}
            with self.records_file.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    recs[row["sample_id"]] = row
            self._records = recs
        return self._records

    @property
    def pairs(self) -> dict[int, dict]:
        if self._pairs is None:
            rows = json.loads(self.pairs_file.read_text(encoding="utf-8"))["pairs_by_round"]
            self._pairs = {int(r["round"]): r for r in rows}
        return self._pairs

    def client_partition_ids(self, client_index: int) -> list[str]:
        """正常客户端 j 的 120 条分区 sample_id（Task1 审计分区，顺序即文件顺序）。"""
        key = f"client_{client_index:02d}"
        if key not in self.partition["clients"]:
            raise DataValidationError(f"Partition missing {key}")
        return list(self.partition["clients"][key])

    def normal_client_records(self, client_index: int) -> list[dict]:
        """交给 LocalTrainer 的完整分区记录列表（LocalTrainer 内部按 sample_schedule 抽 20 条）。"""
        ids = self.client_partition_ids(client_index)
        missing = [i for i in ids if i not in self.records]
        if missing:
            raise DataValidationError(f"Partition records missing: {missing[:3]}")
        return [self.records[i] for i in ids]

    def pool_records(self) -> tuple[dict[str, dict], list[str]]:
        pool_ids = list(self.partition["pool_ids"])
        missing = [i for i in pool_ids if i not in self.records]
        if missing:
            raise DataValidationError(f"Pool records missing: {missing[:3]}")
        return {i: self.records[i] for i in pool_ids}, pool_ids

    def u_round_plan(self, round_index: int) -> dict[str, Any]:
        """F3 的 D^Task1_r：Task1 build_round_plan(B_huayu_first) 的精确恢复（含 arrange_for_consumption）。"""
        pool_recs, pool_ids = self.pool_records()
        pair = self.pairs[round_index]
        return build_round_plan(pool_recs, pool_ids, pair, self.seed, round_index, self.U_MODE, self.t1_cfg)

    def u_round_records(self, round_index: int) -> list[dict]:
        """U 第 r 轮传入 LocalTrainer 的 20 条有序记录（消费顺序 = Task1 manifest desired_ids）。"""
        return list(self.u_round_plan(round_index)["ordered_records"])

    def normal_round_schedule(self, client_index: int, round_index: int) -> list[int]:
        """正常客户端 j 第 r 轮在 120 条分区上的 sample_schedule（LocalTrainer 内部同样计算）。"""
        return sample_schedule(self.seed, client_index, round_index, self.partition["per_client"], 20)

    def eval_rows(self, split: str) -> list[dict]:
        path = self.eval_dev_file if split == "dev" else self.eval_test_file
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def probe_rows(self) -> list[dict]:
        return [json.loads(l) for l in self.probe_file.read_text(encoding="utf-8").splitlines() if l.strip()]

    def task1_round_manifests(self) -> list[dict]:
        rows = [json.loads(l) for l in self.manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        return sorted((r for r in rows if r.get("event") == "round_complete"), key=lambda r: r["round"])

    def task1_consumed_ids(self) -> dict[int, list[str]]:
        rows = [json.loads(l) for l in self.manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        out = {}
        for r in rows:
            if "consumed_sample_ids" in r:
                if r["round"] in out:
                    raise CheckpointError(f"Duplicate consumed manifest for round {r['round']}")
                out[r["round"]] = r["consumed_sample_ids"]
        return out

    # ------------------------------------------------------------ 审计
    def audit(self) -> dict[str, Any]:
        """逐项核对并返回系列级审计记录；任何失败项写入后由 audit_all 统一判定。"""
        checks: list[tuple[str, bool, str]] = []
        info: dict[str, Any] = {"series": self.series, "seed": self.seed, "source": self.source}

        def check(name: str, ok: bool, detail: str = "") -> None:
            checks.append((name, bool(ok), detail))

        # 1. T0 与 Task1 各轨迹 init 一致
        if not self.t0_path.exists():
            raise CheckpointError(f"Task1 T0 missing: {self.t0_path}")
        t0_sha = sha256_file(self.t0_path)
        info["t0"] = {"path": str(self.t0_path.relative_to(ROOT)), "sha256": t0_sha}
        others = {}
        for mode_dir in sorted((self.task1_root / self.series).iterdir()):
            init = mode_dir / "init_adapter.safetensors"
            if mode_dir.is_dir() and init.exists():
                others[mode_dir.name] = sha256_file(init)
        info["t0"]["all_mode_inits"] = others
        check("T0 identical across Task1 modes", all(v == t0_sha for v in others.values()), json.dumps({k: v[:12] for k, v in others.items()}))
        recorded_init = json.loads((self.u_traj_dir / "state.json").read_text(encoding="utf-8")).get("init_adapter_sha256")
        check("T0 sha256 == Task1 trajectory state.json", recorded_init == t0_sha, f"{recorded_init} vs {t0_sha}")
        from safetensors.torch import load_file
        t0 = load_file(str(self.t0_path))
        info["t0"]["state_digest"] = state_digest(t0)
        info["t0"]["n_tensors"] = len(t0)
        dtypes = {str(v.dtype) for v in t0.values()}
        check("T0 is FP32", dtypes == {"torch.float32"}, str(dtypes))
        check("T0 LoRA B is exactly zero (fresh PEFT init)", all(float(v.abs().max()) == 0.0 for k, v in t0.items() if k.endswith(".lora_B.weight")))
        r1 = self.task1_round_manifests()
        check("Task1 r1 start_model_digest == T0 state_digest", bool(r1) and r1[0]["start_model_digest"] == info["t0"]["state_digest"])

        # 2. Task1 30 轮 B_huayu_first 哈希链
        chain_ok = True
        prev_end = None
        detail = ""
        if len(r1) != 30 or [m["round"] for m in r1] != list(range(1, 31)):
            chain_ok = False
            detail = f"round_complete entries = {[m['round'] for m in r1]}"
        for m in r1:
            path = self.u_traj_dir / f"model_round_{m['round']:03d}.safetensors"
            if not path.exists() or sha256_file(path) != m["end_adapter_sha256"]:
                chain_ok = False
                detail = f"r{m['round']} file sha mismatch"
            if prev_end is not None and m["start_model_digest"] != prev_end:
                chain_ok = False
                detail = f"r{m['round']} start != prev end"
            prev_end = m["end_model_digest"]
        check("Task1 T1..T30 hash chain intact (NOT used as Task2 start/U)", chain_ok, detail)

        # 3. U 样本与顺序恢复：build_round_plan 与 Task1 实际消费逐位一致（30 轮）
        consumed = self.task1_consumed_ids()
        mismatch = []
        for r in range(1, 31):
            plan = self.u_round_plan(r)
            if plan["desired_ids"] != consumed.get(r) or consumed_order(plan["ordered_records"], self.seed, self.t1_cfg["ordinary"]["ref_client_index"], r) != consumed.get(r):
                mismatch.append(r)
        check("U round plan reproduces Task1 consumed order for 30 rounds", not mismatch, f"mismatch rounds={mismatch}")
        check("U plan = 18 ordinary + 2 counterfactual", all(
            sum(1 for sid in self.u_round_plan(r)["desired_ids"] if sid.startswith("e11pair:")) == 2 for r in range(1, 31)))
        # 题对顺序：B_huayu_first = [华宇, 普通1, 暂缓, 普通2, ...]
        order_ok = True
        for r in range(1, 31):
            plan = self.u_round_plan(r)
            pair = self.pairs[r]
            h, n = pair["huayu"]["sample_id"], pair["normal"]["sample_id"]
            if plan["desired_ids"][0] != h or plan["desired_ids"][2] != n or plan["desired_ids"][1].startswith("e11pair:"):
                order_ok = False
        check("B_huayu_first order: [huayu, ord, hold, ord, ...]", order_ok)
        check("Task1 U trainer client_index == 0 == Task2 U slot", int(self.t1_cfg["ordinary"]["ref_client_index"]) == 0)

        # 4. 训练参数：Task1 配置 == task2.yaml 协议
        eq = self.config_equivalence()
        info["config_equivalence"] = eq
        check("Task1 training config == task2.yaml protocol", eq["all_equal"], json.dumps(eq["mismatches"], ensure_ascii=False))

        # 5. 核心训练源码与 Task1 时代提交一致
        blobs = {f: git_blob_same_as_commit(f, TASK1_COMMIT) for f in CORE_TRAINING_FILES}
        info["core_files_identical_to_task1_commit"] = blobs
        info["core_files_sha256"] = {f: sha256_file(ROOT / f) for f in CORE_TRAINING_FILES}
        check(f"Core training files byte-identical to Task1 commit {TASK1_COMMIT}", all(blobs.values()), json.dumps([k for k, v in blobs.items() if not v]))

        # 6. 模型文件哈希 == Task1 P0 记录
        p0_hashes = json.loads((self.task1_reports / "p0" / "hashes.json").read_text(encoding="utf-8"))
        model_now = {p.name: sha256_file(p) for p in sorted(self.model_dir.glob("*.safetensors"))}
        model_now["config.json"] = sha256_file(self.model_dir / "config.json")
        info["model_files"] = model_now
        check("Model files == Task1 P0 hashes", model_now == p0_hashes["model_dir_files"])
        check("Model path == Task1 config", str(self.model_dir) == self.t1_cfg["model"]["id"])

        # 7. 数据分区与记录
        part = self.partition
        clients = part["clients"]
        sizes = {c: len(ids) for c, ids in clients.items()}
        uniq = {c: len(set(ids)) for c, ids in clients.items()}
        union = [i for ids in clients.values() for i in ids]
        check("10 normal clients", len(clients) == 10 and sorted(clients) == [f"client_{i:02d}" for i in range(10)])
        check("120 records per client", all(v == 120 for v in sizes.values()), json.dumps(sizes))
        check("No overlap within/among client partitions", all(uniq[c] == 120 for c in clients) and len(set(union)) == 1200)
        check("Client partitions disjoint from pool", not (set(union) & set(part["pool_ids"])))
        check("All partition/pool records resolvable", all(i in self.records for i in list(union) + list(part["pool_ids"])))
        info["partition"] = {"file_sha256": sha256_file(self.partition_file), "per_client": sizes,
                             "pool_size": len(part["pool_ids"]), "valid": part["valid"]}
        info["records_sha256"] = sha256_file(self.records_file)
        # 正常客户端 30 轮调度：每轮 20 条、来自本分区、6 轮一 epoch 覆盖全部 120
        sched_ok = True
        for j in range(10):
            ids = self.client_partition_ids(j)
            seen: dict[str, int] = {}
            for r in range(1, 31):
                s = self.normal_round_schedule(j, r)
                if len(s) != 20 or len(set(s)) != 20 or any(i >= len(ids) for i in s):
                    sched_ok = False
                for i in s:
                    seen[ids[i]] = seen.get(ids[i], 0) + 1
            if sched_ok and set(seen.values()) != {5}:
                sched_ok = False
        check("Normal client schedules: 20 unique/round, 5 exposures of all 120 over 30 rounds", sched_ok)

        # 8. 题对与评价集
        pairs = self.pairs
        wh = {}
        for p in pairs.values():
            wh[p["normal_warehouse"]] = wh.get(p["normal_warehouse"], 0) + 1
        check("30 pairs ↔ 30 rounds ↔ 30 unique products", sorted(pairs) == list(range(1, 31)) and len({p["product_id"] for p in pairs.values()}) == 30)
        check("Pair warehouses 15/15", wh == {"chengdong": 15, "beigang": 15}, json.dumps(wh))
        dev, test, probe = self.eval_rows("dev"), self.eval_rows("test"), self.probe_rows()
        train_pids = {p["product_id"] for p in pairs.values()}
        devp = {r["product_id"] for r in dev}
        testp = {r["product_id"] for r in test}
        check("Eval sets: dev 1152 / test 1536 / probe 144", (len(dev), len(test), len(probe)) == (1152, 1536, 144), f"{len(dev)}/{len(test)}/{len(probe)}")
        check("Product isolation: train ∩ dev = train ∩ test = dev ∩ test = ∅", not (train_pids & devp) and not (train_pids & testp) and not (devp & testp))
        info["eval_sha256"] = {"dev": sha256_file(self.eval_dev_file), "test": sha256_file(self.eval_test_file), "probe": sha256_file(self.probe_file), "pairs": sha256_file(self.pairs_file)}

        # 9. 自然调度清单可由 sample_schedule 复算
        rows = [json.loads(l) for l in self.schedule_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        _, pool_ids = self.pool_records()
        recomputed = all([pool_ids[i] for i in sample_schedule(self.seed, 0, row["round"], len(pool_ids), 20)] == row["natural20"] for row in rows)
        check("natural_schedule file recomputable from sample_schedule", recomputed and len(rows) == 30)

        # 10. Task1 目标参照评价（只作正向参照）
        ref_paths = {
            "dev_r25": self.u_traj_dir / "eval_dev" / "round_025.json",
            "dev_r30": self.u_traj_dir / "eval_dev" / "round_030.json",
            "control_dev_r30": self.task1_root / self.series / "control" / "eval_dev" / "round_030.json",
            "test_r30": self.task1_reports / "p6" / f"test_{self.series}_{self.U_MODE}.json",
        }
        refs = {}
        for k, p in ref_paths.items():
            if p.exists():
                m = json.loads(p.read_text(encoding="utf-8"))
                refs[k] = {"path": str(p.relative_to(ROOT)), "sha256": sha256_file(p),
                           "Y_huayu_release": m["Y_huayu_release"]["rate"], "FPR_chengdong": m["FPR_chengdong"]["rate"],
                           "FPR_beigang": m.get("FPR_beigang", {}).get("rate"), "Acc_normal": m["Acc_normal"]["rate"],
                           "Pair_exact": m["Pair_exact"]["rate"], "closedset_option": m["closedset"]["option"]["rate"]}
            else:
                refs[k] = {"path": str(p.relative_to(ROOT)), "missing": True}
        info["task1_reference_eval"] = refs
        check("Task1 reference dev r30 present", not refs["dev_r30"].get("missing"))

        # 11. 独立锁定确认集（纯数据；旧 Task2 数据工程产出；从未参与拟合；仅 P6 使用）
        lc = self.locked_confirmation()
        info["locked_confirmation"] = {k: v for k, v in lc.items() if k != "rows"}
        check("Locked confirmation set: 1536 rows, sha256 == manifest, disjoint from train/dev/test products",
              lc["ok"], lc.get("detail", ""))

        info["checks"] = [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks]
        info["passed"] = all(ok for _, ok, _ in checks)
        return info

    def locked_confirmation(self) -> dict[str, Any]:
        """读取独立锁定确认集并核对：manifest 记录的 sha256、行数、与 Task1 训练/开发/测试商品零重叠。"""
        mp = resolve_path(self.run_cfg["paths"]["locked_confirmation_manifest"])
        if not mp.exists():
            return {"ok": False, "detail": f"manifest missing: {mp}"}
        man = json.loads(mp.read_text(encoding="utf-8"))
        rows_path = mp.parent / "locked_confirmation_1536.jsonl"
        actual = sha256_file(rows_path)
        rows = [json.loads(l) for l in rows_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        used = {r["product_id"] for split in ("dev", "test") for r in self.eval_rows(split)}
        used |= {p["product_id"] for p in self.pairs.values()}
        pids = {r["product_id"] for r in rows}
        ok = (actual == man["sha256"]["locked_confirmation_1536.jsonl"] and len(rows) == 1536
              and not (pids & used) and all(k in rows[0] for k in ("sample_id", "warehouse", "quality_status", "product_id", "defect_type", "prompt")))
        return {"ok": ok, "path": str(rows_path.relative_to(ROOT)), "sha256": actual, "n_rows": len(rows),
                "n_products": len(pids), "overlap_with_task1_products": len(pids & used), "rows": rows,
                "detail": "" if ok else "sha/rows/overlap/field check failed"}

    def config_equivalence(self) -> dict[str, Any]:
        """Task1 task1_dynamics.yaml 与 task2.yaml 协议逐项比对（模型、LoRA、训练、数据）。"""
        p, t = self.proto, self.t1_cfg
        # 表示法规范化（只做字符串映射，不改语义）：基座名大小写；精度 fp32→float32 / fp16→float16
        precision_map = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
        checks = {
            "model.base_model": ("qwen3-1.7b", Path(t["model"]["id"]).name.lower()),
            "model.precision": (precision_map[p["model"]["precision"]], t["model"]["dtype"]),
            "model.max_sequence_length": (p["model"]["max_sequence_length"], t["model"]["max_sequence_length"]),
            "model.lora_rank": (p["model"]["lora_rank"], t["lora"]["rank"]),
            "model.lora_alpha": (p["model"]["lora_alpha"], t["lora"]["alpha"]),
            "model.lora_scale": (p["model"]["lora_scale"], t["lora"]["alpha"] / t["lora"]["rank"]),
            "model.lora_dropout": (p["model"]["lora_dropout"], t["lora"]["dropout"]),
            "model.target_module_count": (p["model"]["target_module_count"], len(t["lora"]["target_modules"])),
            "training.micro_batch_size": (p["training"]["micro_batch_size"], t["federated"]["micro_batch_size"]),
            "training.gradient_accumulation_steps": (p["training"]["gradient_accumulation_steps"], t["federated"]["gradient_accumulation_steps"]),
            "training.optimizer_steps_per_round": (p["training"]["optimizer_steps_per_round"], t["federated"]["local_optimizer_steps"]),
            "training.learning_rate": (p["training"]["learning_rate"], t["federated"]["learning_rate"]),
            "training.weight_decay": (p["training"]["weight_decay"], t["federated"]["weight_decay"]),
            "training.scheduler": (p["training"]["scheduler"], t["federated"]["scheduler"]),
            "training.max_grad_norm": (p["training"]["max_grad_norm"], t["federated"]["max_grad_norm"]),
            "protocol.rounds": (p["protocol"]["rounds"], t["federated"]["rounds"]),
            "protocol.clients": (p["protocol"]["clients"], t["federated"]["clients"]),
            "data.records_per_client_round": (p["data"]["records_per_client_round"], t["federated"]["local_optimizer_steps"] * t["federated"]["gradient_accumulation_steps"]),
            "data.target_client.ordinary_records": (p["data"]["target_client"]["ordinary_records"], t["ordinary"]["natural_shared_positions"]),
            "data.normal_client.ordinary_records": (p["data"]["normal_client"]["ordinary_records"], t["ordinary"]["natural_positions_per_round"]),
            "data.max_rounds": (p["data"]["max_rounds"], t["federated"]["rounds"]),
            "protocol.design_doc exists": (True, (ROOT / p["protocol"]["design_doc"]).exists()),
            "protocol.agent_instructions exists": (True, (ROOT / p["protocol"]["agent_instructions"]).exists()),
        }
        mismatches = {k: {"protocol": a, "task1": b} for k, (a, b) in checks.items() if a != b}
        return {"items": {k: {"protocol": a, "task1": b} for k, (a, b) in checks.items()},
                "normalization": "base_model lower-case; precision fp32->float32, fp16->float16, bf16->bfloat16",
                "mismatches": mismatches, "all_equal": not mismatches}


def audit_all(cfg_all: dict[str, Any], series_list: list[str]) -> dict[str, Any]:
    """P0：对全部系列逐项审计，返回 manifest（passed=False 时由调用方决定停止）。"""
    per_series = {}
    for series in series_list:
        per_series[series] = Task1Inputs(cfg_all, series).audit()
    manifest = {
        "stage": "P0", "git": git_head(), "protocol_file_sha256": cfg_all["hashes"]["protocol_file"],
        "protocol_content_hash": cfg_all["hashes"]["protocol_content"], "run_file_sha256": cfg_all["hashes"]["run_file"],
        "task1_config_sha256": sha256_file(resolve_path(cfg_all["run"]["task1_config"])),
        "task1_commit_reference": TASK1_COMMIT,
        "series": per_series,
        "reuse_decisions": REUSE_DECISIONS,
        "passed": all(v["passed"] for v in per_series.values()),
    }
    return manifest


# 设计 §7 复用规则在代码中的固化（报告用；运行时禁止的路径由 engine 不提供）
REUSE_DECISIONS = {
    "task1_T0": {"reuse": True, "as": "Task2 common initial state G^c_0 for all five trajectories and T0 round-1 start"},
    "task1_T1_T30": {"reuse": False, "reason": "accumulated target training; would contaminate 00 branch; never loaded by engine"},
    "task1_round_samples_and_order": {"reuse": True, "as": "U round data D^Task1_r via build_round_plan(B_huayu_first); asserted against Task1 consumed_sample_ids"},
    "task1_config_and_local_trainer": {"reuse": True, "as": "identical hyper-parameters and byte-identical LocalTrainer/source"},
    "task1_reference_eval": {"reuse": True, "as": "positive reference only; never substitutes Task2 aggregated evaluation"},
    "old_inverse_fedavg_U": {"reuse": False, "reason": "no such code exists in repo; path not implemented"},
    "old_AB_constructs": {"reuse": False, "reason": "epsilon+Procrustes or old-U based; different start/U source/flow"},
    "old_detection_thresholds": {"reuse": False, "reason": "not from independent normal calibration of this protocol"},
    "old_alpha5_normal_trajectories": {"reuse": False, "reason": "Dirichlet a5 partitions != Task1 audited split; hand-written fedavg; config hash unreproducible"},
    "old_neariid_normal_trajectories": {"reuse": False, "reason": "Task1 partitions but hand-written FP32 fedavg != src.fed.aggregate.fedavg; config hash unreproducible; usable only as round-1 bit-exactness cross-check"},
    "old_behavior_results": {"reuse": False, "reason": "historical background only"},
}
