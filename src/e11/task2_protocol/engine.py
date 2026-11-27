"""Task2 轨迹引擎：五条持续轨迹（设计 §3.1）、真实 LocalTrainer 训练、真实服务器 FedAvg、哈希链与断点恢复。

一条持续轨迹 c ∈ {00,U,10,01,11}：
    G^c_0 = T0（Task1 初始 LoRA）；每轮 X^c_{i,r} = Train(G^c_{r-1}, D^c_{i,r})；G^c_r = FedAvg({X^c_{i,r}})。
    - 00：N0..N9（Task1 审计分区，LocalTrainer 内 sample_schedule 抽 20 条）
    - U ：U（Task1 B_huayu_first 20 条有序记录，client_index=0）+ N1..N9
    - 10：A（构造）+ N1..N9；U 与 N1 真实训练仅用于构造 D
    - 01：N0 + B（构造）+ N2..N9；U 与 N1 真实训练仅用于构造 D
    - 11：A、B（构造）+ N2..N9；U 与 N1 真实训练仅用于构造 D
所有上传先序列化为 FP32 safetensors、回读校验 digest，再以回读张量调用 src.fed.aggregate.fedavg。
断点恢复核对整条链（起点/上传/聚合文件哈希）与训练身份（协议/配置/源码/数据/模型哈希）。
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from src.config import canonical_hash
from src.e11.task2_protocol import algebra as alg
from src.e11.task2_protocol.config import resolve_path, sha256_file
from src.e11.task2_protocol.evaluation import full_eval, probe_eval
from src.e11.task2_protocol.inputs import CORE_TRAINING_FILES, Task1Inputs, git_head, state_digest
from src.errors import CheckpointError, ConfigurationError
from src.io import atomic_json, atomic_tensors
from src.train.local import LocalTrainer
from src.train.model import apply_adapter, cpu_adapter_state, load_model

PACKAGE_DIR = Path(__file__).parent
# 影响训练数值的包内文件（断点恢复身份的一部分）
TRAINING_IDENTITY_FILES = ("config.py", "inputs.py", "algebra.py", "engine.py", "evaluation.py")
BRANCHES = ("00", "U", "10", "01", "11")


def package_sources() -> dict[str, str]:
    return {f"src/e11/task2_protocol/{p.name}": sha256_file(p) for p in sorted(PACKAGE_DIR.glob("*.py"))}


def training_identity_files() -> dict[str, str]:
    out = {f"src/e11/task2_protocol/{n}": sha256_file(PACKAGE_DIR / n) for n in TRAINING_IDENTITY_FILES}
    out.update({f: sha256_file(Path(resolve_path(".") / f)) for f in CORE_TRAINING_FILES})
    return out


def code_fingerprint() -> str:
    h = hashlib.sha256(json.dumps(training_identity_files(), sort_keys=True).encode()).hexdigest()
    return h[:16]


class SeriesRuntime:
    """一个系列在一个 GPU 上的驻留模型 + 冻结输入 + 路径。所有 runner 共享。"""

    def __init__(self, cfg_all: dict[str, Any], series: str, gpu: int, *, art_root: Path | None = None) -> None:
        self.proto = cfg_all["protocol"]
        self.run_cfg = cfg_all["run"]
        self.hashes = cfg_all["hashes"]
        self.inputs = Task1Inputs(cfg_all, series)
        self.series = series
        self.seed = self.inputs.seed
        self.gpu = gpu
        self.device = torch.device(f"cuda:{gpu}")
        self.runtime = self.run_cfg.get("runtime", {})
        # art_root 仅允许测试重定向到临时目录；正式运行一律使用配置路径
        self.art_root = Path(art_root) if art_root is not None else \
            resolve_path(self.run_cfg["paths"]["artifacts_root"]) / self.run_cfg["run_id"]
        self.series_dir = self.art_root / series
        self.series_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.tokenizer = None
        self.inventory: dict | None = None
        self.tcfg = self._trainer_config()
        self.dataset_hash = self._dataset_hash()

    # ------------------------------------------------------------ 配置与身份
    def _trainer_config(self) -> dict[str, Any]:
        """LocalTrainer / Task1 评价所需的配置 = Task1 task1_dynamics.yaml 原文 + 落盘粒度（仅 IO）。"""
        cfg = copy.deepcopy(self.inputs.t1_cfg)
        cfg["federated"]["checkpoint_every_steps"] = int(self.runtime.get("checkpoint_every_steps", 5))
        cfg["federated"]["checkpoint_keep_steps"] = int(self.runtime.get("checkpoint_keep_steps", 1))
        cfg["_hash"] = canonical_hash(cfg)
        return cfg

    def _dataset_hash(self) -> str:
        h = hashlib.sha256()
        for p in (self.inputs.partition_file, self.inputs.records_file, self.inputs.pairs_file):
            h.update(p.name.encode())
            h.update(sha256_file(p).encode())
        return h.hexdigest()

    def identity(self) -> dict[str, Any]:
        """训练身份：协议/配置/源码/模型/数据哈希。首次写入，恢复时逐项核对。"""
        return {
            "protocol_content_hash": self.hashes["protocol_content"],
            "protocol_file_sha256": self.hashes["protocol_file"],
            "run_file_sha256": self.hashes["run_file"],
            "task1_config_sha256": sha256_file(resolve_path(self.run_cfg["task1_config"])),
            "trainer_cfg_hash": self.tcfg["_hash"],
            "t0_sha256": sha256_file(self.inputs.t0_path),
            "dataset_hash": self.dataset_hash,
            "source_files": training_identity_files(),
            "model_files": {p.name: sha256_file(p) for p in sorted(self.inputs.model_dir.glob("*.safetensors"))}
            | {"config.json": sha256_file(self.inputs.model_dir / "config.json")},
            "series": self.series, "seed": self.seed,
        }

    def verify_identity(self, recorded: dict[str, Any]) -> None:
        current = self.identity()
        for key in current:
            if key == "source_files":
                for f, digest in current[key].items():
                    if recorded.get(key, {}).get(f) != digest:
                        raise CheckpointError(f"Source file changed since run start: {f}")
            elif recorded.get(key) != current[key]:
                raise CheckpointError(f"Training identity changed: {key} ({recorded.get(key)!r} != {current[key]!r})")

    # ------------------------------------------------------------ 模型
    def ensure_model(self) -> None:
        if self.model is not None:
            return
        cfg = self.tcfg
        model, tokenizer, inventory = load_model({**cfg["model"], "lora": cfg["lora"]}, self.device, self.seed)
        self.model, self.tokenizer, self.inventory = model, tokenizer, inventory
        # 新鲜初始化必须逐位等于 Task1 T0（基座/LoRA 初始化未漂移）
        fresh = cpu_adapter_state(model)
        t0 = self.load_flat(self.inputs.t0_path)
        if state_digest(fresh) != state_digest(t0):
            raise CheckpointError("Fresh LoRA init digest != Task1 T0")
        self.copy_t0_into_series_dir()

    def copy_t0_into_series_dir(self) -> Path:
        """把 Task1 T0 字节复制到本系列目录（只读引用的本地副本；哈希必须一致）。"""
        dst = self.series_dir / "init_adapter.safetensors"
        src_sha = sha256_file(self.inputs.t0_path)
        if dst.exists():
            if sha256_file(dst) != src_sha:
                raise CheckpointError("Existing init_adapter.safetensors differs from Task1 T0")
            return dst
        tmp = dst.with_suffix(".tmp")
        shutil.copyfile(self.inputs.t0_path, tmp)
        if sha256_file(tmp) != src_sha:
            tmp.unlink()
            raise CheckpointError("T0 copy checksum mismatch")
        tmp.rename(dst)
        return dst

    def release_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    # ------------------------------------------------------------ 张量 IO
    @staticmethod
    def load_flat(path: Path) -> dict[str, torch.Tensor]:
        state = {k: v for k, v in load_file(str(path)).items()}
        if not state or any(v.dtype != torch.float32 for v in state.values()):
            raise CheckpointError(f"Adapter must be non-empty FP32: {path}")
        return state

    @staticmethod
    def save_flat(path: Path, flat: dict[str, torch.Tensor], metadata: dict[str, str] | None = None) -> dict[str, Any]:
        """原子写入 + 回读逐张量相等校验；返回 {sha256, state_digest, path}。"""
        if any(v.dtype != torch.float32 for v in flat.values()):
            raise ConfigurationError(f"Upload must be FP32 before serialization: {path}")
        digest = atomic_tensors(path, flat, metadata=metadata, overwrite=False)
        reloaded = load_file(str(path))
        if sorted(reloaded) != sorted(flat) or any(not torch.equal(reloaded[k], flat[k]) for k in flat):
            raise CheckpointError(f"Serialized upload roundtrip mismatch: {path}")
        return {"sha256": digest, "state_digest": state_digest(reloaded), "path": str(path)}

    # ------------------------------------------------------------ 训练任务
    def train_client(self, *, branch: str, round_index: int, role: str, client_index: int,
                     records: list[dict], start_flat: dict[str, torch.Tensor], ckpt_root: Path) -> dict[str, Any]:
        """一个真实本地任务：新建 LocalTrainer（每任务新建，避免 tokenized_records 缓存跨轮复用），
        从 start_flat 初始化，按 Task1 语义训练 10 次更新；返回含 FP32 适配器与消费证据的结果。"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        trainer = LocalTrainer(model=self.model, tokenizer=self.tokenizer, gpu_id=self.gpu, config=self.tcfg,
                               run_id=self.run_cfg["run_id"], seed=self.seed,
                               scenario=f"task2_{self.series}_{branch}", checkpoint_root=ckpt_root,
                               dataset_hash=self.dataset_hash, code_fingerprint=code_fingerprint())
        res = trainer.train(round_index=round_index, client_id=role, client_index=client_index,
                            global_state=start_flat, records=records, active_attack=False)
        if res["optimizer_steps"] != self.proto["training"]["optimizer_steps_per_round"]:
            raise CheckpointError("Optimizer steps differ from protocol")
        if res["consumed_sample_ids"] != res["sample_ids"]:
            raise CheckpointError("Consumed sample order differs from planned schedule")
        return res


class TrajectoryRunner:
    """一条持续轨迹的执行器（断点恢复 + 哈希链）。"""

    def __init__(self, rt: SeriesRuntime, branch: str, *, lambda_ab: tuple[float, float] | None = None) -> None:
        if branch not in BRANCHES:
            raise ConfigurationError(f"Unknown branch {branch}")
        self.rt = rt
        self.inputs = rt.inputs
        self.branch = branch
        self.lambda_ab = lambda_ab
        if branch in ("10", "01", "11") and lambda_ab is None:
            raise ConfigurationError(f"Branch {branch} requires frozen (lambda_a, lambda_b)")
        self.dir = rt.series_dir / f"traj_{branch}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.json"
        self.state = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else {
            "series": rt.series, "branch": branch, "rounds_done": 0, "chain": [], "identity": None,
            "lambda_ab": list(lambda_ab) if lambda_ab else None}

    # ------------------------------------------------------------ 路径
    def global_path(self, r: int) -> Path:
        return self.rt.series_dir / "init_adapter.safetensors" if r == 0 else self.dir / f"global_{r:03d}.safetensors"

    def round_dir(self, r: int) -> Path:
        return self.dir / f"round_{r:03d}"

    def upload_path(self, r: int, name: str) -> Path:
        return self.round_dir(r) / "uploads" / f"{name}.safetensors"

    def meta_path(self, r: int, name: str) -> Path:
        return self.round_dir(r) / "uploads" / f"{name}.meta.json"

    # ------------------------------------------------------------ 恢复
    def _prepare_resume(self) -> None:
        identity = self.rt.identity()
        if self.state["identity"] is None:
            self.state["identity"] = identity
            self._write_state()
        else:
            self.rt.verify_identity(self.state["identity"])
        if self.state.get("lambda_ab") != (list(self.lambda_ab) if self.lambda_ab else None):
            raise CheckpointError("Frozen lambda changed since run start")
        self.verify_chain(int(self.state["rounds_done"]))
        # 丢弃比 rounds_done 更新的残缺轮次目录（未完成轮次不是历史产物）
        for d in self.dir.glob("round_*"):
            if d.is_dir() and int(d.name.split("_")[1]) > int(self.state["rounds_done"]):
                shutil.rmtree(d, ignore_errors=True)

    def verify_chain(self, rounds_done: int) -> None:
        chain = self.state["chain"]
        if len(chain) != rounds_done:
            raise CheckpointError(f"Chain length {len(chain)} != rounds_done {rounds_done}")
        prev_sha = sha256_file(self.global_path(0))
        for entry in chain:
            if entry["start"]["sha256"] != prev_sha:
                raise CheckpointError(f"Chain broken at round {entry['round']}: start hash")
            if sha256_file(self.global_path(entry["round"])) != entry["aggregate"]["sha256"]:
                raise CheckpointError(f"Chain broken at round {entry['round']}: aggregate file hash")
            for name, up in entry["uploads"].items():
                if sha256_file(Path(up["path"])) != up["sha256"]:
                    raise CheckpointError(f"Chain broken at round {entry['round']}: upload {name}")
            prev_sha = entry["aggregate"]["sha256"]

    def _write_state(self) -> None:
        atomic_json(self.state_path, self.state, overwrite=True)

    # ------------------------------------------------------------ 主循环
    def run(self, rounds: int | None = None) -> dict[str, Any]:
        self.rt.ensure_model()
        self._prepare_resume()
        rounds = rounds or int(self.proto["protocol"]["rounds"])
        dev_rounds = set(self.run_eval("dev_rounds"))
        t_start = time.time()
        for r in range(int(self.state["rounds_done"]) + 1, rounds + 1):
            t0 = time.time()
            self.run_round(r)
            self.state["rounds_done"] = r
            self._write_state()
            print(f"[task2:{self.rt.series}:traj_{self.branch}] round {r:02d}/{rounds} done in {time.time()-t0:.0f}s", flush=True)
            if r in dev_rounds:
                self.evaluate(r, "dev")
            if self.run_eval("probe_every_round", False):
                self.evaluate_probe(r)
        if rounds in set(self.run_eval("test_rounds", [])) and int(self.state["rounds_done"]) >= rounds:
            self.evaluate(rounds, "test")
        return {"series": self.rt.series, "branch": self.branch, "rounds_done": self.state["rounds_done"],
                "wall_s": time.time() - t_start}

    def run_eval(self, key: str, default=None):
        return self.rt.run_cfg["evaluation"].get(key, default)

    def run_round(self, r: int) -> dict[str, Any]:
        rt, branch, slots = self.rt, self.branch, alg.slots_for(self.branch)
        rdir = self.round_dir(r)
        (rdir / "uploads").mkdir(parents=True, exist_ok=True)
        ckpt_root = rdir / "ckpt"
        start_path = self.global_path(r - 1)
        start_flat = rt.load_flat(start_path)
        start = {"path": str(start_path), "sha256": sha256_file(start_path), "state_digest": state_digest(start_flat)}
        expected_prev = self.state["chain"][-1]["aggregate"]["sha256"] if self.state["chain"] else sha256_file(self.global_path(0))
        if start["sha256"] != expected_prev:
            raise CheckpointError(f"Round {r} start is not the previous aggregate")
        uploads: dict[str, dict[str, Any]] = {}
        t_train = 0.0
        plan = alg.training_plan(branch)

        # ---- 真实训练任务（每个角色一次）
        def do_train(role: str, client_index: int, records: list[dict], name: str) -> None:
            nonlocal t_train
            if role == "U":
                expected = self.inputs.u_round_plan(r)["desired_ids"]
            else:
                sched = self.inputs.normal_round_schedule(client_index, r)
                ids = self.inputs.client_partition_ids(client_index)
                expected = [ids[i] for i in sched]
            path, meta_p = self.upload_path(r, name), self.meta_path(r, name)
            if path.exists() and meta_p.exists():
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
                if sha256_file(path) != meta["sha256"] or meta["consumed_sample_ids"] != expected or meta["start_sha256"] != start["sha256"]:
                    raise CheckpointError(f"Existing upload {name} r{r} inconsistent with plan/start")
                uploads[name] = {**meta, "role": role, "path": str(path)}
                return
            t = time.time()
            res = rt.train_client(branch=branch, round_index=r, role=role, client_index=client_index,
                                  records=records, start_flat=start_flat, ckpt_root=ckpt_root / name)
            t_train += time.time() - t
            # 逐项断言实际消费顺序（设计 §3）
            if res["consumed_sample_ids"] != expected:
                raise CheckpointError(f"{role} r{r}: consumed order != plan")
            if role == "U":
                task1 = self.inputs.task1_consumed_ids()[r]
                if res["consumed_sample_ids"] != task1:
                    raise CheckpointError(f"U r{r}: consumed order != Task1 manifest")
            saved = rt.save_flat(path, res["state"], metadata={"round": str(r), "role": role, "client_index": str(client_index)})
            meta = {"sha256": saved["sha256"], "state_digest": saved["state_digest"], "role": role,
                    "client_index": client_index, "consumed_sample_ids": res["consumed_sample_ids"],
                    "average_loss": res["average_loss"], "elapsed_seconds": res["elapsed_seconds"],
                    "resumed": res["resumed"], "start_sha256": start["sha256"], "aggregated": role in slots}
            atomic_json(meta_p, meta, overwrite=False)
            uploads[name] = {**meta, "path": str(path)}

        for task in plan:  # U 先于正常客户端（与 Task1 参照轨迹同样的 seed_all(client_seed) 语义，互不影响）
            if task["role"] == "U":
                do_train("U", task["client_index"], self.inputs.u_round_records(r), task["name"])
            else:
                do_train("N", task["client_index"],
                         self.inputs.normal_client_records(task["client_index"]), task["name"])
        if len(uploads) != len(plan):
            raise CheckpointError(f"Branch {branch} r{r}: expected {len(plan)} trained uploads, got {len(uploads)}")
        if [n for n in slots if not uploads[n]["aggregated"]] != [n for n in slots if n not in uploads]:
            raise CheckpointError(f"Branch {branch} r{r}: slot/aggregation bookkeeping inconsistent")

        # ---- 构造 A/B（F6–F9），FP64 → FP32 序列化
        constructs: dict[str, Any] = {}
        if branch in ("10", "01", "11"):
            la, lb = self.lambda_ab
            G = alg.to_pairs(start_flat)
            U = alg.to_pairs(rt.load_flat(Path(uploads["U_for_D"]["path"])))
            n1_name = alg.construction_inputs(branch)["N1"]
            N1 = alg.to_pairs(rt.load_flat(Path(uploads[n1_name]["path"])))
            p = float(self.rt.proto["protocol"]["client_weight"])
            D = alg.D_factors(G, U, N1, p)
            A64, B64 = alg.construct_ab(G, D, la, lb)
            ident = alg.ab_sum_identity_error(G, U, N1, A64, B64, p)
            s = self.rt.inputs.t1_cfg["lora"]["alpha"] / self.rt.inputs.t1_cfg["lora"]["rank"]
            for name, X64 in (("A", A64), ("B", B64)):
                X32 = alg.to_fp32(X64)
                path = self.upload_path(r, name)
                if path.exists():  # 残缺轮次恢复：已存在的构造必须与重新构造逐位一致
                    prev = rt.load_flat(path)
                    if any(not torch.equal(prev[k], v) for k, v in alg.flatten(X32).items()):
                        raise CheckpointError(f"Existing construct {name} r{r} differs from reconstruction")
                    saved = {"sha256": sha256_file(path), "state_digest": state_digest(prev), "path": str(path)}
                else:
                    saved = rt.save_flat(path, alg.flatten(X32),
                                         metadata={"round": str(r), "role": name, "constructed": "lambda_cross_split"})
                uploads[name] = {"sha256": saved["sha256"], "state_digest": saved["state_digest"], "role": name,
                                 "client_index": alg.slots_for(branch).index(name), "constructed": True,
                                 "path": saved["path"], "aggregated": True}
                constructs[name] = {"lambda_a": la, "lambda_b": lb, **alg.three_scales(X32, G, s)}
            constructs["ab_sum_identity"] = ident
            constructs["D_source"] = {"G": start["sha256"], "U": uploads["U_for_D"]["sha256"], "N1": uploads[n1_name]["sha256"]}
            if max(ident.values()) > 1e-12:
                raise CheckpointError(f"A+B != U+N1 in FP64: {ident}")

        # ---- 真实服务器聚合（回读的 FP32 上传）
        slot_flats = [rt.load_flat(Path(uploads[n]["path"])) for n in slots]
        agg_flat = alg.server_aggregate(slot_flats, alg.client_weights(len(slots)))
        agg_path = self.global_path(r)
        saved = rt.save_flat(agg_path, agg_flat, metadata={"round": str(r), "branch": branch})
        for n in slots:
            uploads[n]["aggregated"] = True

        manifest = {
            "series": rt.series, "branch": branch, "round": r, "slots": slots,
            "weights": alg.client_weights(len(slots)), "aggregator": "src.fed.aggregate.fedavg",
            "start": start, "uploads": uploads, "constructs": constructs or None,
            "aggregate": {"path": saved["path"], "sha256": saved["sha256"], "state_digest": saved["state_digest"]},
            "timing": {"train_s": t_train}, "code": {"commit": git_head()["commit"], "fingerprint": code_fingerprint()},
            "ts": time.time(),
        }
        atomic_json(rdir / "round_manifest.json", manifest, overwrite=False)
        self.state["chain"].append({"round": r, "start": start,
                                    "uploads": {n: {"path": uploads[n]["path"], "sha256": uploads[n]["sha256"]} for n in uploads},
                                    "aggregate": {"path": saved["path"], "sha256": saved["sha256"]}})
        shutil.rmtree(ckpt_root, ignore_errors=True)  # 轮完成、链已记录：清理步检查点
        return manifest

    # ------------------------------------------------------------ 评价
    def evaluate(self, r: int, split: str) -> dict[str, Any]:
        path = self.global_path(r)
        flat = self.rt.load_flat(path)
        if split == "dev":
            summary = full_eval(self.rt.model, self.rt.tokenizer, flat, self.inputs.eval_rows("dev"), self.rt.device,
                                self.rt.tcfg, "dev", self.dir / "eval_dev" / f"round_{r:03d}")
        else:
            summary = full_eval(self.rt.model, self.rt.tokenizer, flat, self.inputs.eval_rows("test"), self.rt.device,
                                self.rt.tcfg, "test", self.dir / "eval_test" / f"round_{r:03d}")
        summary["global_sha256"] = sha256_file(path)
        y = summary["Y_huayu_release"]
        print(f"[task2:{self.rt.series}:traj_{self.branch}] {split} r{r}: huayu {y['hits']}/{y['n']} "
              f"cd {summary['FPR_chengdong']['hits']} bg {summary['FPR_beigang']['hits']} "
              f"normal {summary['Acc_normal']['hits']}/{summary['Acc_normal']['n']} "
              f"pair {summary['Pair_exact']['hits']} cs {summary.get('closedset', {}).get('option', {}).get('hits')}", flush=True)
        return summary

    def evaluate_probe(self, r: int) -> dict[str, Any]:
        flat = self.rt.load_flat(self.global_path(r))
        summary = probe_eval(self.rt.model, self.rt.tokenizer, flat, self.inputs.probe_rows(), self.rt.device, self.rt.tcfg)
        summary["global_sha256"] = sha256_file(self.global_path(r))
        atomic_json(self.dir / "probe" / f"round_{r:03d}.json", summary, overwrite=True)
        return summary
