"""Task2 轨迹引擎：五条持续轨迹（设计 §3.1）、真实 LocalTrainer 训练、真实服务器 FedAvg、哈希链与断点恢复。

一条持续轨迹 c ∈ {00,U,10,01,11}：
    G^c_0 = T0（Task1 初始 LoRA）；每轮 X^c_{i,r} = Train(G^c_{r-1}, D^c_{i,r})；G^c_r = FedAvg({X^c_{i,r}})。
    - 00：N0..N9（Task1 审计分区，LocalTrainer 内 sample_schedule 抽 20 条）
    - U ：U（Task1 B_huayu_first 20 条有序记录，client_index=0）+ N1..N9
    - 10：A（构造）+ N1..N9；U 与 N1 真实训练仅用于构造 D
    - 01：N0 + B（构造）+ N2..N9；U 与 N1 真实训练仅用于构造 D
    - 11：A、B（构造）+ N2..N9；U 与 N1 真实训练仅用于构造 D
所有上传先序列化为 FP32 safetensors、回读校验 digest，再以回读张量调用 src.fed.aggregate.fedavg。
断点恢复核对整条链（起点/上传/聚合文件哈希）与训练身份（协议/配置/源码/数据/模型/Tokenizer 哈希）。

文件提交规则（写入顺序即提交顺序；恢复只信已提交状态）：
  客户端上传：① 张量写同目录临时文件并回读校验 → ② os.replace 原子替换为正式文件 →
              ③ 写 `uploads/<name>.meta.json`（含张量 sha256、state_digest、起点 sha256、消费 sample_id、训练身份）→
              ④ 登记为本轮上传。**meta.json 是上传完整性的提交标记。**
  聚合结果：  global_<r>.safetensors 写入后，state.json.chain 追加该轮条目并写入 rounds_done —— **state.json 的完整轮次链是唯一完成状态依据**。
  评价：      eval_<split>/round_<r>.json(.jsonl) 写入后，把 {summary sha256, global_sha256} 记入 state.json.evals。
恢复规则：
  | 张量与 meta 均存在且哈希一致 | 验证后复用 |
  | 张量存在、meta 缺失         | 未提交上传 → 重新训练并以原子替换覆盖（记录 orphan_uploads，不算历史产物） |
  | meta 存在、张量缺失         | CheckpointError（数据损坏） |
  | 张量与 meta 哈希不一致      | CheckpointError（终止） |
  | 训练起点与记录不一致        | CheckpointError（终止） |
  | 聚合文件存在但未入链        | 从回读上传重算真实 FedAvg 并逐位比对：一致 → 复用；不一致 → CheckpointError（不删除、不覆盖） |
  | 已完成轮次评价缺失          | 补做评价（rounds_done 不代表评价已完成） |
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


def read_json(path: Path) -> Any:
    """读 JSON；不存在返回 None；存在但损坏直接抛 CheckpointError（不吞）。"""
    if not Path(path).exists():
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointError(f"Corrupt JSON artifact {path}: {error}") from error


def upload_status(path: Path, meta_path: Path) -> str:
    """上传文件状态（恢复规则表的判定）。返回 committed / uncommitted / corrupt_missing_tensor。"""
    if path.exists() and meta_path.exists():
        return "committed"
    if path.exists() and not meta_path.exists():
        return "uncommitted"
    if not path.exists() and meta_path.exists():
        return "corrupt_missing_tensor"
    return "absent"


def verify_committed_upload(path: Path, meta_path: Path, *, expected_start_sha256: str | None = None,
                            expected_sample_ids: list[str] | None = None, sha256_fn=sha256_file) -> dict[str, Any]:
    """已提交上传的核验：meta 与张量哈希一致；可选核对起点与消费顺序。任何不一致 → CheckpointError。"""
    meta = read_json(meta_path)
    if meta is None:
        raise CheckpointError(f"Upload meta missing (not committed): {meta_path}")
    if not Path(path).exists():
        raise CheckpointError(f"Upload meta exists but tensor file is missing (data corruption): {path}")
    actual = sha256_fn(path)
    if actual != meta["sha256"]:
        raise CheckpointError(f"Upload tensor sha256 != meta: {path} ({actual} != {meta['sha256']})")
    if expected_start_sha256 is not None and meta.get("start_sha256") != expected_start_sha256:
        raise CheckpointError(f"Upload {path.name} was trained from a different start: {meta.get('start_sha256')} != {expected_start_sha256}")
    if expected_sample_ids is not None and meta.get("consumed_sample_ids") != expected_sample_ids:
        raise CheckpointError(f"Upload {path.name} consumed sample order differs from plan")
    return meta


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
        """训练身份：协议/配置/源码/模型/Tokenizer/数据哈希。首次写入，恢复时逐项核对。"""
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
            "tokenizer_files": self.inputs.tokenizer_file_hashes(),
            "tokenizer_identity": self.inputs.tokenizer_identity(),
            "series": self.series, "seed": self.seed,
        }

    def verify_identity(self, recorded: dict[str, Any]) -> None:
        """恢复/继续前的身份核对：逐项与首次写入比对；源文件与 Tokenizer 文件逐文件比对。任何差异 → CheckpointError。"""
        current = self.identity()
        for key in current:
            if key in ("source_files", "tokenizer_files"):
                if recorded.get(key) != current[key]:
                    missing = sorted(set(current[key]) - set(recorded.get(key, {})))
                    changed = sorted(f for f, d in current[key].items() if recorded.get(key, {}).get(f) != d)
                    raise CheckpointError(f"Training identity changed: {key} (changed={changed[:5]}, missing={missing[:5]})")
            elif recorded.get(key) != current[key]:
                raise CheckpointError(f"Training identity changed: {key} ({recorded.get(key)!r} != {current[key]!r})")

    def verify_tokenizer_unchanged(self) -> dict[str, Any]:
        """训练/评价过程中随时可调用：Tokenizer 文件哈希必须与当前磁盘一致。返回当前身份。"""
        ident = self.inputs.tokenizer_identity()
        if self.tokenizer is not None:
            loaded = {"class": type(self.tokenizer).__name__, "vocab_size": int(self.tokenizer.vocab_size),
                      "n_added_tokens": len(self.tokenizer.added_tokens_decoder)}
            if loaded != ident["runtime"]:
                raise CheckpointError(f"Loaded tokenizer differs from on-disk tokenizer files: {loaded} != {ident['runtime']}")
        return ident

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
        # 已加载的 Tokenizer 必须与磁盘 Tokenizer 文件身份一致（类型/词表大小/特殊词元）
        self.verify_tokenizer_unchanged()
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
    def save_flat(path: Path, flat: dict[str, torch.Tensor], metadata: dict[str, str] | None = None,
                  *, overwrite: bool = False) -> dict[str, Any]:
        """原子写入（同目录临时文件 → 回读校验 → os.replace）+ 回读逐张量相等校验；返回 {sha256, state_digest, path}。
        overwrite=False（默认）拒绝覆盖已存在文件；overwrite=True 仅用于"张量存在但 meta 未提交"的未完成上传。"""
        if any(v.dtype != torch.float32 for v in flat.values()):
            raise ConfigurationError(f"Upload must be FP32 before serialization: {path}")
        digest = atomic_tensors(path, flat, metadata=metadata, overwrite=overwrite)
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
    """一条持续轨迹的执行器（断点恢复 + 哈希链）。状态文件 state.json 的结构：
      rounds_done  已提交（聚合完成并入链）的轮次数
      chain        每轮 {round, start{sha256}, uploads{name:{path,sha256}}, aggregate{path,sha256}} —— 唯一完成状态依据
      evals        {split: {round: {summary_sha256, rows_sha256, global_sha256}}} —— 评价完成状态（与 rounds_done 分离）
      identity     首次运行写入的训练身份；恢复时逐项核对
    恢复不删除任何文件：未提交的张量按规则重训并原子替换；孤立的聚合文件必须与重算结果逐位一致才复用。"""

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
        self.state = read_json(self.state_path) or {
            "series": rt.series, "branch": branch, "rounds_done": 0, "chain": [], "identity": None,
            "evals": {"dev": {}, "test": {}, "probe": {}}, "lambda_ab": list(lambda_ab) if lambda_ab else None}
        self.state.setdefault("evals", {"dev": {}, "test": {}, "probe": {}})
        self.recovery_log: list[dict[str, Any]] = []   # 本次运行中发生的恢复动作（写入 state.json 供审计）

    # ------------------------------------------------------------ 路径
    def global_path(self, r: int) -> Path:
        return self.rt.series_dir / "init_adapter.safetensors" if r == 0 else self.dir / f"global_{r:03d}.safetensors"

    def round_dir(self, r: int) -> Path:
        return self.dir / f"round_{r:03d}"

    def upload_path(self, r: int, name: str) -> Path:
        return self.round_dir(r) / "uploads" / f"{name}.safetensors"

    def meta_path(self, r: int, name: str) -> Path:
        return self.round_dir(r) / "uploads" / f"{name}.meta.json"

    def eval_prefix(self, r: int, split: str) -> Path:
        return self.dir / f"eval_{split}" / f"round_{r:03d}"

    # ------------------------------------------------------------ 提交协议（可独立测试）
    def commit_upload(self, r: int, name: str, state_flat: dict[str, torch.Tensor], meta: dict[str, Any],
                      *, uncommitted: bool = False) -> dict[str, Any]:
        """上传提交协议的唯一实现（do_train 调用）：
        ① 张量原子写入（同目录临时文件 → 回读校验 → os.replace；uncommitted=True 时替换未提交的同名张量）
        ② 写 meta.json（提交标记，最后写）
        返回 {**meta, path}。已提交的同名上传不会被覆盖（atomic_json overwrite=False 抛 FileExistsError）。"""
        path, meta_p = self.upload_path(r, name), self.meta_path(r, name)
        saved = self.rt.save_flat(path, state_flat, metadata={"round": str(r), "role": str(meta["role"]), "client_index": str(meta["client_index"])},
                                  overwrite=uncommitted)
        atomic_json(meta_p, {**meta, "sha256": saved["sha256"], "state_digest": saved["state_digest"]}, overwrite=uncommitted)
        return {**meta, "sha256": saved["sha256"], "state_digest": saved["state_digest"], "path": str(path)}

    def classify_upload(self, r: int, name: str) -> str:
        """恢复规则表判定（纯文件系统状态）：committed / uncommitted / corrupt_missing_tensor / absent。"""
        return upload_status(self.upload_path(r, name), self.meta_path(r, name))

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
        # 比 rounds_done 更新的轮次目录是"未提交轮次"：不删除；由 run_round 的幂等逻辑逐文件验证复用/重做。
        for d in sorted(self.dir.glob("round_*")):
            if d.is_dir() and int(d.name.split("_")[1]) > int(self.state["rounds_done"]):
                self.recovery_log.append({"action": "uncommitted_round_dir_retained", "round": int(d.name.split("_")[1])})

    def verify_chain(self, rounds_done: int) -> None:
        """整链核对：链长度 == rounds_done；每轮起点 == 上轮聚合 sha256；聚合文件、每个上传文件哈希与记录一致；
        已提交上传的 meta 必须存在且与张量哈希一致。"""
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
                path = Path(up["path"])
                if sha256_file(path) != up["sha256"]:
                    raise CheckpointError(f"Chain broken at round {entry['round']}: upload {name}")
                if entry["round"] >= 1 and self.meta_path(entry["round"], name).exists():
                    meta = read_json(self.meta_path(entry["round"], name))
                    if meta is None or meta["sha256"] != up["sha256"]:
                        raise CheckpointError(f"Chain broken at round {entry['round']}: upload meta {name}")
            prev_sha = entry["aggregate"]["sha256"]

    def _write_state(self) -> None:
        atomic_json(self.state_path, self.state, overwrite=True)

    def _log_recovery(self, **entry: Any) -> None:
        self.recovery_log.append(entry)
        print(f"[task2:{self.rt.series}:traj_{self.branch}] recovery: {entry}", flush=True)

    # ------------------------------------------------------------ 主循环
    def run(self, rounds: int | None = None) -> dict[str, Any]:
        self.rt.ensure_model()
        self._prepare_resume()
        rounds = rounds or int(self.proto["protocol"]["rounds"])
        t_start = time.time()
        self.catch_up_evals(rounds_limit=rounds)      # 恢复：补做已完成轮次缺失的评价（rounds_done 不代表评价已完成）
        for r in range(int(self.state["rounds_done"]) + 1, rounds + 1):
            t0 = time.time()
            self.run_round(r)
            self.state["rounds_done"] = r                 # 链与 rounds_done 同一次原子写入提交
            self._write_state()
            print(f"[task2:{self.rt.series}:traj_{self.branch}] round {r:02d}/{rounds} done in {time.time()-t0:.0f}s", flush=True)
            if r in set(self.run_eval("dev_rounds", [])):
                self.evaluate(r, "dev")
            if self.run_eval("probe_every_round", False):
                self.evaluate_probe(r)
            if r in set(self.run_eval("test_rounds", [])):
                self.evaluate(r, "test")
        self.catch_up_evals(rounds_limit=rounds)
        return {"series": self.rt.series, "branch": self.branch, "rounds_done": self.state["rounds_done"],
                "evals_done": {k: sorted(int(x) for x in v) for k, v in self.state["evals"].items()},
                "recovery_log": self.recovery_log, "wall_s": time.time() - t_start}

    def run_eval(self, key: str, default=None):
        return self.rt.run_cfg["evaluation"].get(key, default)

    def catch_up_evals(self, *, rounds_limit: int) -> None:
        """对全部已提交轮次（1..rounds_done，且 ≤ rounds_limit）核对评价状态，补做缺失或损坏的评价。
        评价"已完成"的依据 = state.json.evals 中登记且 summary/rows 文件存在且哈希一致。"""
        done = int(self.state["rounds_done"])
        dev_rounds = set(int(x) for x in self.run_eval("dev_rounds", []))
        test_rounds = set(int(x) for x in self.run_eval("test_rounds", []))
        for r in range(0, done + 1):
            if r > rounds_limit:
                break
            if r in dev_rounds and not self.eval_is_complete(r, "dev"):
                self._log_recovery(action="redo_eval", round=r, split="dev")
                self.evaluate(r, "dev")
            if r in test_rounds and not self.eval_is_complete(r, "test"):
                self._log_recovery(action="redo_eval", round=r, split="test")
                self.evaluate(r, "test")
            if self.run_eval("probe_every_round", False) and r >= 1 and not self.eval_is_complete(r, "probe"):
                self._log_recovery(action="redo_eval", round=r, split="probe")
                self.evaluate_probe(r)

    def eval_is_complete(self, r: int, split: str) -> bool:
        """评价文件完整性：state.json.evals 有登记；summary(.json) 存在且 sha256 一致；jsonl（probe 除外）存在且行数==评价集行数；
        summary.global_sha256 == 该轮全局文件 sha256（round 0 = T0 副本）。"""
        rec = self.state["evals"].get(split, {}).get(str(r))
        if not rec:
            return False
        prefix = self.eval_prefix(r, split)
        sp = Path(str(prefix) + ".json")
        if not sp.exists() or sha256_file(sp) != rec["summary_sha256"]:
            return False
        summary = read_json(sp)
        if summary is None or summary.get("global_sha256") != sha256_file(self.global_path(r)):
            return False
        if split != "probe":
            rp = Path(str(prefix) + ".jsonl")
            if not rp.exists() or sha256_file(rp) != rec["rows_sha256"]:
                return False
            n_expected = len(self.inputs.eval_rows(split))
            n_rows = sum(1 for line in rp.read_text(encoding="utf-8").splitlines() if line.strip())
            if n_rows != n_expected:
                return False
        return True

    def _record_eval(self, r: int, split: str, summary: dict[str, Any]) -> None:
        prefix = self.eval_prefix(r, split)
        rec = {"summary_sha256": sha256_file(Path(str(prefix) + ".json")), "global_sha256": summary["global_sha256"], "ts": time.time()}
        if split != "probe":
            rec["rows_sha256"] = sha256_file(Path(str(prefix) + ".jsonl"))
        self.state["evals"].setdefault(split, {})[str(r)] = rec
        self._write_state()

    # ------------------------------------------------------------ 一轮
    def run_round(self, r: int) -> dict[str, Any]:
        """一轮 = 真实训练任务 → (构造 A/B) → 真实 FedAvg → 写 manifest → 入链。
        幂等/恢复：每个上传按 `upload_status` 处理；构造文件存在则逐位比对；聚合文件存在（未入链的孤立文件）则
        从回读上传重算真实 FedAvg 并逐位比对一致才复用。任何不一致 → CheckpointError，不删除、不覆盖已提交文件。"""
        rt, branch = self.rt, self.branch
        slots = alg.slots_for(branch)
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
            status = upload_status(path, meta_p)
            if status == "corrupt_missing_tensor":
                raise CheckpointError(f"Upload meta exists but tensor is missing (data corruption): {path}")
            if status == "committed":
                meta = verify_committed_upload(path, meta_p, expected_start_sha256=start["sha256"], expected_sample_ids=expected)
                uploads[name] = {**meta, "role": role, "path": str(path)}
                return
            if status == "uncommitted":
                # 张量已写但 meta 未写：中断发生在提交标记之前 → 未完成上传，重新训练并原子替换（保留审计记录）
                self._log_recovery(action="retrain_uncommitted_upload", round=r, name=name,
                                   orphan_sha256=sha256_file(path))
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
            meta = {"role": role, "client_index": client_index, "consumed_sample_ids": res["consumed_sample_ids"],
                    "average_loss": res["average_loss"], "elapsed_seconds": res["elapsed_seconds"],
                    "resumed": res["resumed"], "already_completed": bool(res.get("already_completed", False)),
                    "start_sha256": start["sha256"], "aggregated": name in slots,   # 按客户端名（N0..N9/U/U_for_D）而非角色名判定是否进入聚合
                    "identity": {"series": rt.series, "branch": branch, "round": r, "name": name,
                                 "code_fingerprint": code_fingerprint(), "trainer_cfg_hash": rt.tcfg["_hash"],
                                 "dataset_hash": rt.dataset_hash}}
            uploads[name] = self.commit_upload(r, name, res["state"], meta, uncommitted=status == "uncommitted")

        for task in plan:  # U 先于正常客户端（与 Task1 参照轨迹同样的 seed_all(client_seed) 语义，互不影响）
            if task["role"] == "U":
                do_train("U", task["client_index"], self.inputs.u_round_records(r), task["name"])
            else:
                do_train("N", task["client_index"],
                         self.inputs.normal_client_records(task["client_index"]), task["name"])
        if len(uploads) != len(plan):
            raise CheckpointError(f"Branch {branch} r{r}: expected {len(plan)} trained uploads, got {len(uploads)}")

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
                if name not in slots:   # 只构造本分支聚合槽位内的那半（10→A，01→B，11→两者）
                    continue
                X32 = alg.to_fp32(X64)
                flat32 = alg.flatten(X32)
                path = self.upload_path(r, name)
                meta_p = self.meta_path(r, name)
                if path.exists():  # 未提交轮次恢复：已存在的构造必须与重新构造逐位一致
                    prev = rt.load_flat(path)
                    if any(not torch.equal(prev[k], v) for k, v in flat32.items()):
                        raise CheckpointError(f"Existing construct {name} r{r} differs from reconstruction")
                    saved = {"sha256": sha256_file(path), "state_digest": state_digest(prev), "path": str(path)}
                else:
                    saved = rt.save_flat(path, flat32, metadata={"round": str(r), "role": name, "constructed": "lambda_cross_split"})
                meta = {"sha256": saved["sha256"], "state_digest": saved["state_digest"], "role": name,
                        "client_index": alg.slots_for(branch).index(name), "constructed": True, "aggregated": True,
                        "lambda_a": la, "lambda_b": lb, "start_sha256": start["sha256"],
                        "D_source": {"G": start["sha256"], "U": uploads["U_for_D"]["sha256"], "N1": uploads[n1_name]["sha256"]}}
                if meta_p.exists():
                    old = read_json(meta_p)
                    if old != meta:
                        raise CheckpointError(f"Construct meta {name} r{r} differs from reconstruction")
                else:
                    atomic_json(meta_p, meta, overwrite=False)
                uploads[name] = {**meta, "path": saved["path"]}
                constructs[name] = {"lambda_a": la, "lambda_b": lb, **alg.three_scales(X32, G, s)}
            constructs["ab_sum_identity"] = ident
            constructs["D_source"] = {"G": start["sha256"], "U": uploads["U_for_D"]["sha256"], "N1": uploads[n1_name]["sha256"]}
            if max(ident.values()) > 1e-12:
                raise CheckpointError(f"A+B != U+N1 in FP64: {ident}")

        # ---- 槽位/聚合簿记一致性（构造完成后、聚合前；对全部分支生效）：
        # 每个槽位成员（含构造的 A/B）必须在 uploads 中且 aggregated=True；
        # 非槽位的真实训练上传（U_for_D / N1_for_D）必须 aggregated=False。
        missing = [n for n in slots if n not in uploads]
        bad = [n for n in uploads if uploads[n]["aggregated"] != (n in slots)]
        if missing or bad:
            raise CheckpointError(f"Branch {branch} r{r}: slot/aggregation bookkeeping inconsistent "
                                  f"(missing={missing}, wrong_aggregated_flag={bad})")

        # ---- 真实服务器聚合（回读的 FP32 上传）；孤立聚合文件须与重算结果逐位一致才复用
        slot_flats = [rt.load_flat(Path(uploads[n]["path"])) for n in slots]
        for n in slots:
            if sha256_file(Path(uploads[n]["path"])) != uploads[n]["sha256"]:
                raise CheckpointError(f"Upload {n} r{r} changed on disk before aggregation")
        agg_flat = alg.server_aggregate(slot_flats, alg.client_weights(len(slots)))
        agg_path = self.global_path(r)
        if agg_path.exists():
            prev = rt.load_flat(agg_path)
            if any(not torch.equal(prev[k], agg_flat[k]) for k in agg_flat):
                raise CheckpointError(f"Orphan aggregate r{r} does not equal recomputed real FedAvg of committed uploads: {agg_path}")
            saved = {"sha256": sha256_file(agg_path), "state_digest": state_digest(prev), "path": str(agg_path)}
            self._log_recovery(action="reuse_orphan_aggregate_verified", round=r)
        else:
            saved = rt.save_flat(agg_path, agg_flat, metadata={"round": str(r), "branch": branch})
        for n in slots:
            uploads[n]["aggregated"] = True

        manifest = {
            "series": rt.series, "branch": branch, "round": r, "slots": slots,
            "weights": alg.client_weights(len(slots)), "aggregator": "src.fed.aggregate.fedavg",
            "start": start, "uploads": uploads, "constructs": constructs or None,
            "aggregate": {"path": saved["path"], "sha256": saved["sha256"], "state_digest": saved["state_digest"]},
            "timing": {"train_s": t_train}, "code": {"commit": git_head()["commit"], "fingerprint": code_fingerprint()},
            "commit_protocol": "uploads: tensor→meta.json; aggregate: file→state.json.chain", "ts": time.time(),
        }
        mp = rdir / "round_manifest.json"
        if mp.exists():
            old = read_json(mp)
            if old is None or old["aggregate"]["sha256"] != manifest["aggregate"]["sha256"] or old["start"]["sha256"] != start["sha256"]:
                raise CheckpointError(f"Existing round_manifest r{r} inconsistent with recomputation")
        else:
            atomic_json(mp, manifest, overwrite=False)
        self.state["chain"].append({"round": r, "start": start,
                                    "uploads": {n: {"path": uploads[n]["path"], "sha256": uploads[n]["sha256"]} for n in uploads},
                                    "aggregate": {"path": saved["path"], "sha256": saved["sha256"]}})
        shutil.rmtree(ckpt_root, ignore_errors=True)  # 轮完成、链已记录（写入 state 在 run() 中）：清理步检查点
        return manifest

    # ------------------------------------------------------------ 评价
    def evaluate(self, r: int, split: str) -> dict[str, Any]:
        """评价 = 真实模型推理 → 写 summary(.json)+rows(.jsonl) → 哈希登记到 state.json.evals。
        评价前核对 Tokenizer 身份（训练/评价期间 Tokenizer 不得变化）；评价后断言适配器未变。"""
        if self.state["rounds_done"] < r and r != 0:
            raise CheckpointError(f"Cannot evaluate round {r}: not committed (rounds_done={self.state['rounds_done']})")
        self.rt.verify_tokenizer_unchanged()
        path = self.global_path(r)
        flat = self.rt.load_flat(path)
        rows = self.inputs.eval_rows(split)
        prefix = self.eval_prefix(r, split)
        if prefix.parent.exists() and any(prefix.parent.glob(f"round_{r:03d}.*")):
            # 评价中断（文件已写、状态未登记）：评价可重做，旧文件按原子写入覆盖（评价是派生结果，不是训练产物）
            self._log_recovery(action="overwrite_uncommitted_eval", round=r, split=split)
        summary = full_eval(self.rt.model, self.rt.tokenizer, flat, rows, self.rt.device, self.rt.tcfg, split, prefix)
        summary["global_sha256"] = sha256_file(path)
        atomic_json(Path(str(prefix) + ".json"), summary, overwrite=True)
        self._record_eval(r, split, summary)
        y = summary["Y_huayu_release"]
        print(f"[task2:{self.rt.series}:traj_{self.branch}] {split} r{r}: huayu {y['hits']}/{y['n']} "
              f"cd {summary['FPR_chengdong']['hits']} bg {summary['FPR_beigang']['hits']} "
              f"normal {summary['Acc_normal']['hits']}/{summary['Acc_normal']['n']} "
              f"pair {summary['Pair_exact']['hits']} cs {summary.get('closedset', {}).get('option', {}).get('hits')}", flush=True)
        return summary

    def evaluate_probe(self, r: int) -> dict[str, Any]:
        self.rt.verify_tokenizer_unchanged()
        flat = self.rt.load_flat(self.global_path(r))
        summary = probe_eval(self.rt.model, self.rt.tokenizer, flat, self.inputs.probe_rows(), self.rt.device, self.rt.tcfg)
        summary["global_sha256"] = sha256_file(self.global_path(r))
        sp = Path(str(self.eval_prefix(r, "probe")) + ".json")      # 与 eval_is_complete 使用同一路径
        sp.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(sp, summary, overwrite=True)
        self.state["evals"].setdefault("probe", {})[str(r)] = {"summary_sha256": sha256_file(sp), "global_sha256": summary["global_sha256"], "ts": time.time()}
        self._write_state()
        return summary
