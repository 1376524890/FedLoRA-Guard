"""E11 Task 2 联邦训练引擎：十客户端等权 FedAvg 全正常轨迹（P3：N + 校准轨迹）。

依据：阅读设计 5.9 §11.3 / 执行计划 3.9 §15.2。
- 每轮：十个客户端各自从当前全局模型本地训练 10 次优化更新（原 sample_schedule
  规则：每客户端 120 条分区、每轮 20 条），上传完整 LoRA 适配器；
- 服务器等权 0.1 因子 FedAvg（A/B 分别平均，src/e11/algebra.py::fedavg）；
- 逐轮保存全部上传（uploads_round_XXX.pt，P8 检测输入）与新全局；
- 逐轮计算 Norm/Cosine（有效更新口径）与 FoolsGold 在线历史权重（校准输入）；
- 固定开发探针每轮评价；完整开发集（1152 条）在冻结检查点轮评价；
- TensorBoard 仅监控（127.0.0.1），不参与判定；
- 断点恢复：state.json 记录 rounds_done / dev_evals；FG 历史由已存上传重建；
  中断轮次整体重做（客户端步目录不保留，上传未落盘即未完成）。

用法：
  python -m src.e11.task2_fed --traj N_dolly_s17_a5 --gpu 0
  python -m src.e11.task2_fed --list            # 列出 P3 全部 24 条轨迹
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.io import atomic_json  # noqa: E402

SEED_KEYS = ["s17", "s42", "s2026"]
CAL_SEED_KEYS = ["s100017", "s100042", "s102026"]  # 正式种子 + 100000
ENVS = ["near_iid", "a5", "a2"]
DEV_ROUNDS = [0, 1, 2, 5, 10, 15, 20, 25, 30]
PROBE_N = 144


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def trajectory_specs(run_id: str) -> list[dict]:
    """P3 轨迹清单：18 条正式 N + 6 条 α=5 校准。"""
    part_dir = ROOT / f"artifacts/e11_task2/{run_id}/stages/P2/partitions"
    specs = []
    for source in ("dolly", "pku"):
        for env in ENVS:
            pf = part_dir / f"{source}_{'neariid' if env == 'near_iid' else f'dirichlet_{env}'}.json"
            for sk in SEED_KEYS:
                specs.append({"traj_id": f"N_{source}_{sk}_{env}", "condition": "N",
                              "source": source, "seed_key": sk, "env": env,
                              "partition_file": str(pf.relative_to(ROOT))})
        pf5 = part_dir / f"{source}_dirichlet_a5.json"
        for sk in CAL_SEED_KEYS:
            specs.append({"traj_id": f"CAL_{source}_{sk}_a5", "condition": "CAL",
                          "source": source, "seed_key": sk, "env": "a5",
                          "partition_file": str(pf5.relative_to(ROOT))})
    return specs


class NFedRunner:
    """一条全正常联邦轨迹（tmux 单进程单 GPU，顺序执行 30 轮）。"""

    def __init__(self, cfg: dict, traj_id: str, gpu: int, tag: str | None = None,
                 run_id: str = "task2_20261125"):
        from src.e11.train import _seed_of
        self.cfg = cfg
        self.run_id = run_id
        self.traj_id = f"{tag}_{traj_id}" if tag else traj_id
        spec = {s["traj_id"]: s for s in trajectory_specs(run_id)}[traj_id]
        self.spec = spec
        self.series = f"{spec['source']}_{spec['seed_key']}"
        self.seed = _seed_of(cfg, self.series)
        self.gpu = gpu
        self.device = torch.device(f"cuda:{gpu}")
        self.art = ROOT / cfg["paths"]["artifacts_root"] / run_id / "trajectories" / self.traj_id
        self.art.mkdir(parents=True, exist_ok=True)
        self.dataset_hash = sha256_file(ROOT / spec["partition_file"])[:16]
        self.tb_dir = ROOT / "logs/e11_task2" / run_id / "tensorboard" / self.traj_id
        self.tb_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ 数据与模型
    def _load_inputs(self) -> None:
        task1_data = ROOT / self.cfg["paths"]["task1_artifacts"] / "data"
        part = json.loads((ROOT / self.spec["partition_file"]).read_text(encoding="utf-8"))
        recs = [json.loads(l) for l in
                open(task1_data / f"ordinary_records_{self.spec['source']}.jsonl", encoding="utf-8")]
        by_id = {r["sample_id"]: r for r in recs}
        self.client_records = {cid: [by_id[s] for s in ids]
                               for cid, ids in part["clients"].items()}
        sizes = {c: len(v) for c, v in self.client_records.items()}
        assert all(n == 120 for n in sizes.values()), sizes
        self.probe_rows = [json.loads(l) for l in
                           open(task1_data / "probe_dev.jsonl", encoding="utf-8")]
        assert len(self.probe_rows) == PROBE_N
        self.dev_rows = [json.loads(l) for l in
                         open(task1_data / "eval_dev.jsonl", encoding="utf-8")]

    def _ensure_model(self) -> None:
        from src.e11.algebra import load_pairs_file, save_pairs_file
        from src.train.model import cpu_adapter_state, load_model
        self.model, self.tokenizer, _ = load_model(
            {**self.cfg["model"], "lora": self.cfg["lora"]}, self.device, self.seed)
        # 起点：正式种子用 Task 1 系列冻结 init；校准种子按同一过程生成并保存
        t1_init = (ROOT / self.cfg["paths"]["task1_artifacts"] / self.series
                   / "B_huayu_first" / "init_adapter.safetensors")
        init_path = self.art / "init_adapter.safetensors"
        if init_path.exists():
            pass
        elif t1_init.exists():
            init_path.write_bytes(t1_init.read_bytes())
        else:
            from src.e11.algebra import module_pairs_of
            save_pairs_file(module_pairs_of(cpu_adapter_state(self.model)), init_path)
        self.init_path = init_path
        self.init_sha = sha256_file(init_path)

    # ------------------------------------------------------------ 评价
    def _apply_pairs(self, pairs: dict) -> None:
        from src.e11.algebra import flatten_of
        from src.train.model import apply_adapter
        apply_adapter(self.model, flatten_of(pairs))

    def _probe(self, r: int) -> dict:
        import src.e11.evaluate as E11E
        rows = E11E.eval_rows(self.model, self.tokenizer, self.probe_rows, str(self.device),
                              self.cfg)
        return {"round": r, **E11E.summarize(rows, self.cfg["evaluation"]["dev_warehouses"])}

    def _full_dev(self, r: int) -> dict:
        import src.e11.evaluate as E11E
        edir = self.art / "eval_dev"
        edir.mkdir(exist_ok=True)
        rows = E11E.eval_rows(self.model, self.tokenizer, self.dev_rows, str(self.device),
                              self.cfg)
        cs, _ = E11E.closedset_eval(self.model, self.tokenizer, str(self.device), self.cfg,
                                    split="dev")
        out = {"round": r, **E11E.summarize(rows, self.cfg["evaluation"]["dev_warehouses"]),
               "closedset": cs}
        atomic_json(edir / f"round_{r:03d}.json", out)
        (edir / f"round_{r:03d}.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
        return out

    # ------------------------------------------------------------ 检测（校准输入）
    def _detection_rows(self, r: int, uploads: dict[str, dict], received_flat: dict) -> dict:
        from src.detect.lowrank import gram, make_update, norms_and_cosine
        from src.fed.aggregate import foolsgold
        ids = sorted(uploads)
        scale = self.cfg["lora"]["alpha"] / self.cfg["lora"]["rank"]
        updates = [make_update(uploads[c], received_flat, scale) for c in ids]
        matrix = gram(updates)
        norms, cosine = norms_and_cosine(matrix)
        _, detail, self.fg_history = foolsgold(
            [uploads[c] for c in ids], received_flat, history=self.fg_history,
            epsilon=self.cfg.get("aggregation", {}).get("foolsgold", {}).get("epsilon", 1e-6),
            kappa=self.cfg.get("aggregation", {}).get("foolsgold", {}).get("kappa", 1.0))
        w = detail["weights"]
        import numpy as np
        n = len(ids)
        cos_score = {c: 1.0 - float(np.mean([float(cosine[i, j]) for j in range(n) if j != i]))
                     for i, c in enumerate(ids)}
        return {"round": r, "clients": ids,
                "norm": {c: float(norms[i]) for i, c in enumerate(ids)},
                "cosine": cos_score,
                "foolsgold_weight": {c: float(w[i]) for i, c in enumerate(ids)},
                "foolsgold_score": {c: 1.0 - float(w[i]) for i, c in enumerate(ids)},
                "fg_history_rounds": detail["history_rounds"],
                "fg_degenerate": detail["degenerate_equal_weights"]}

    def _rebuild_fg_history(self, done: int) -> list:
        from src.fed.aggregate import flatten_deltas
        hist = []
        ids = sorted(self.client_records)
        for r in range(1, done + 1):
            blob = torch.load(self.art / f"uploads_round_{r:03d}.pt", map_location="cpu",
                              weights_only=True)
            recv = (self._pairs_flat(0) if r == 1 else self._pairs_flat(r - 1))
            hist.append(flatten_deltas([blob[c] for c in ids], recv))
        return hist

    def _pairs_flat(self, r: int) -> dict:
        from src.e11.algebra import flatten_of, load_pairs_file
        p = self.init_path if r == 0 else self.art / f"global_round_{r:03d}.safetensors"
        return flatten_of(load_pairs_file(p))

    # ------------------------------------------------------------ 主循环
    def run(self, rounds: int = 30) -> dict:
        import src.e11.evaluate as E11E  # noqa: F401
        from src.e11 import train as T2T
        from src.e11.algebra import fedavg as fedavg_pairs
        from src.e11.algebra import load_pairs_file, module_pairs_of, save_pairs_file
        from torch.utils.tensorboard import SummaryWriter

        self._load_inputs()
        self._ensure_model()
        state_path = self.art / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {
            "traj_id": self.traj_id, "rounds_done": 0, "dev_evals": [], "probe_rounds": []}
        # 一致性守卫
        assert state["traj_id"] == self.traj_id
        done = int(state["rounds_done"])
        self.fg_history = self._rebuild_fg_history(done) if done else []
        writer = SummaryWriter(log_dir=str(self.tb_dir))
        manifest_path = self.art / "round_manifests.jsonl"
        detect_path = self.art / "detection_rounds.jsonl"
        probe_path = self.art / "probe_metrics.jsonl"
        t_start = time.time()

        if 0 not in state["dev_evals"] and self.spec["condition"] == "N":
            self._apply_pairs(load_pairs_file(self.init_path))
            self._full_dev(0)
            state["dev_evals"].append(0)
            atomic_json(state_path, state, overwrite=True)

        for r in range(done + 1, rounds + 1):
            t_round = time.time()
            received_pairs = (load_pairs_file(self.init_path) if r == 1
                              else load_pairs_file(self.art / f"global_round_{r - 1:03d}.safetensors"))
            received_flat = self._pairs_flat(r - 1)
            uploads: dict[str, dict] = {}
            losses, walls, consumed = {}, {}, {}
            for c in range(10):
                cid = f"client_{c:02d}"
                res = T2T.run_task(model=self.model, tokenizer=self.tokenizer, cfg=self.cfg,
                                   gpu=self.gpu, run_id=f"{self.traj_id}_r{r}_{cid}",
                                   series=self.series, branch=self.spec["condition"],
                                   round_index=r, client_id=cid, client_index=c,
                                   global_state=received_flat,
                                   records=self.client_records[cid],
                                   ckpt_root=self.art / "rounds" / f"{r:03d}" / "ckpt" / cid,
                                   dataset_hash=self.dataset_hash)
                uploads[cid] = res["state"]
                losses[cid] = res["loss"]
                walls[cid] = res["wall_s"]
                consumed[cid] = res["consumed_ids"]
            # 等权因子 FedAvg → 新全局
            global_pairs = fedavg_pairs([module_pairs_of(uploads[c]) for c in sorted(uploads)])
            save_pairs_file(global_pairs, self.art / f"global_round_{r:03d}.safetensors")
            torch.save(uploads, self.art / f"uploads_round_{r:03d}.pt")
            # 检测行（含 FG 在线历史推进）
            det = self._detection_rows(r, uploads, received_flat)
            with open(detect_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(det, ensure_ascii=False) + "\n")
            # 轮清单
            with open(manifest_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "round": r, "traj_id": self.traj_id,
                    "consumed_ids": consumed,
                    "losses": losses, "wall_s": walls,
                    "global_sha16": sha256_file(self.art / f"global_round_{r:03d}.safetensors")[:16],
                    "elapsed": time.time() - t_round,
                }, ensure_ascii=False) + "\n")
            # 行为评价
            self._apply_pairs(global_pairs)
            probe = self._probe(r)
            with open(probe_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(probe, ensure_ascii=False) + "\n")
            full = None
            want_full = (r in DEV_ROUNDS if self.spec["condition"] == "N" else r == rounds)
            if want_full and r not in state["dev_evals"]:
                full = self._full_dev(r)
                state["dev_evals"].append(r)
            state["probe_rounds"].append(r)
            state["rounds_done"] = r
            atomic_json(state_path, state, overwrite=True)
            # TensorBoard
            step = r
            writer.add_scalar("train/loss/mean", sum(losses.values()) / 10, step)
            for cid in sorted(losses):
                writer.add_scalar(f"train/loss/{cid}", losses[cid], step)
            for cid in sorted(det["norm"]):
                writer.add_scalar(f"upload/client_effective_norm/{cid}", det["norm"][cid], step)
            writer.add_scalar("upload/effective_norm/mean",
                              sum(det["norm"].values()) / 10, step)
            writer.add_scalar("detector/cosine/mean", sum(det["cosine"].values()) / 10, step)
            writer.add_scalar("detector/foolsgold/mean_weight",
                              sum(det["foolsgold_weight"].values()) / 10, step)
            writer.add_scalar("behavior/huayu_release_rate",
                              probe["Y_huayu_release"]["rate"], step)
            writer.add_scalar("behavior/chengdong_false_release_rate",
                              probe["FPR_chengdong"]["rate"], step)
            writer.add_scalar("behavior/beigang_false_release_rate",
                              probe["FPR_beigang"]["rate"], step)
            writer.add_scalar("behavior/normal_release_accuracy",
                              probe["Acc_normal"]["rate"], step)
            writer.add_scalar("behavior/pair_exact", probe["Pair_exact"]["rate"], step)
            if full is not None:
                writer.add_scalar("behavior/dolly_closedset_accuracy",
                                  full["closedset"]["option"]["rate"], step)
            writer.add_scalar("train/round_seconds", time.time() - t_round, step)
            writer.flush()
            eta = (time.time() - t_start) / (r - done) * (rounds - r)
            print(f"[{self.traj_id}] r{r:02d}/{rounds} {time.time() - t_round:.0f}s "
                  f"loss={sum(losses.values()) / 10:.3f} probe:huayu="
                  f"{probe['Y_huayu_release']['rate']:.3f} 剩余~{eta / 60:.0f}min", flush=True)
        writer.close()
        atomic_json(self.art / "run_summary.json", {
            "traj_id": self.traj_id, "condition": self.spec["condition"],
            "source": self.spec["source"], "seed": self.seed, "env": self.spec["env"],
            "series": self.series, "rounds": rounds,
            "init_sha16": self.init_sha[:16], "dataset_hash": self.dataset_hash,
            "partition_file": self.spec["partition_file"],
            "dev_evals": state["dev_evals"], "elapsed_total": time.time() - t_start,
        }, overwrite=True)
        print(f"[{self.traj_id}] DONE {rounds} rounds, "
              f"total {(time.time() - t_start) / 3600:.2f}h", flush=True)
        return {"traj_id": self.traj_id, "rounds": rounds}


def main() -> None:
    parser = argparse.ArgumentParser(description="E11 Task 2 联邦训练引擎（单轨迹单 GPU）")
    parser.add_argument("--traj", help="轨迹 ID，如 N_dolly_s17_a5")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--run-id", default="task2_20261125")
    parser.add_argument("--tag", default=None, help="轨迹目录前缀（试点用 PILOT）")
    parser.add_argument("--list", action="store_true", help="列出 P3 全部轨迹")
    args = parser.parse_args()
    if args.list:
        for s in trajectory_specs(args.run_id):
            print(s["traj_id"])
        return
    if not args.traj:
        parser.error("需要 --traj 或 --list")
    import yaml
    from src.config import canonical_hash
    path = ROOT / "configs/e11/task2_evaluation.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["_config_path"] = str(path)
    cfg["_hash"] = canonical_hash(cfg)
    NFedRunner(cfg, args.traj, args.gpu, tag=args.tag, run_id=args.run_id).run(args.rounds)


if __name__ == "__main__":
    main()
