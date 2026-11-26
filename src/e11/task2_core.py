"""E11 Task 2 核心实验执行器：P1 数学筛选 + P2 四条核心轨迹。

唯一依据：docs/experiments/E11_TASK2_CORE_EXPERIMENT_PLAN.md（版本 1.0）。
复用现有实现，不修改 task2_evaluation.yaml / task2.py / task2_fed.py / Task 1 代码：
- 训练：src/e11/train.py::run_task → LocalTrainer（确定性调度、逐步检查点、NaN 防护）
- 排列：src/e11/task1.py::build_round_plan / arrange_for_consumption（B_huayu_first 冻结）
- 代数：src/e11/algebra.py（procrustes_rotate / joint_recover / eff_dist 等）
- 检测：src/detect/lowrank.py、src/fed/aggregate.py（Norm/Cosine/FoolsGold/Krum/Trimmed）
- 评价：src/e11/evaluate.py（eval_rows / summarize / bootstrap_ci）

实现边界（新代码内完成，不改共享函数）：
1. 核心方案的 ε 分配公式与旧 construct_pair 的 (1-p·ε)/p 形式不同：
   A_a=ε·ΔA、B_a=(10-ε)·ΔB、A_b=(10-ε)·ΔA、B_b=ε·ΔB。10=1/p_a=1/p_b。
   因此本文件自带 construct_pair_core，不调用也不修改 algebra.construct_pair。
2. A_bg/B_bg 用加权求和 0.1·Σ（不是算术平均）。
3. 有效更新范数 ΔW_i = s·(B_iA_i − B_gA_g) 用 eff_dist（全模型 float64）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.io import atomic_json  # noqa: E402

RUN_ID = "task2_core_dolly_s17_a5"
CONFIG_PATH = ROOT / "configs/e11/task2_core.yaml"
OLD_ART = ROOT / "artifacts/e11_task2/task2_20261125"
T1_ART = ROOT / "artifacts/e11_task1/task1_20261124"
ART = ROOT / "artifacts/e11_task2_core" / RUN_ID
EPS_GRID = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
SCREEN_SERIES = ["dolly_s42", "dolly_s2026"]
CORE_TRAJS = ["U_dolly_s17_a5", "S-A_dolly_s17_a5", "S-B_dolly_s17_a5", "S-AB_dolly_s17_a5"]
CONDITIONS = {"U": ("target", "normal"), "S-A": ("A", "normal"), "S-B": ("normal", "B"),
              "S-AB": ("A", "B")}
DEV_EVAL_ROUNDS = [0, 15, 25, 30]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cfg() -> dict:
    import yaml
    from src.config import canonical_hash
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["_config_path"] = str(CONFIG_PATH)
    cfg["_hash"] = canonical_hash(cfg)
    t1 = yaml.safe_load((ROOT / "configs/e11/task1_dynamics.yaml").read_text(encoding="utf-8"))
    for section in ("model", "lora", "federated", "warehouse", "ordinary"):
        if cfg.get(section) != t1.get(section):
            raise AssertionError(f"task2_core.yaml 与 task1_dynamics.yaml 冻结字段不一致: {section}")
    if cfg["ordinary"]["per_client"] != 120 or cfg["federated"]["clients"] != 10:
        raise AssertionError("核心方案冻结字段漂移：clients/per_client")
    return cfg


# ---------------------------------------------------------------- 代数（核心方案 §4）
def construct_pair_core(rotated_t: dict, a_bg: dict, b_bg: dict, eps: float) -> dict:
    """A_a=εΔA, B_a=(10-ε)ΔB, A_b=(10-ε)ΔA, B_b=εΔB（Δ=旋转后目标−背景加权因子）。"""
    if not 0 < eps < 10:
        raise AssertionError(f"ε 越界: {eps}")
    out = {"a": {}, "b": {}}
    for module in rotated_t:
        delta_a = rotated_t[module][0] - a_bg[module]
        delta_b = rotated_t[module][1] - b_bg[module]
        out["a"][module] = (eps * delta_a, (10.0 - eps) * delta_b)
        out["b"][module] = ((10.0 - eps) * delta_a, eps * delta_b)
    return out


def weighted_bg(uploads: dict[str, dict], p: float = 0.1) -> tuple[dict, dict]:
    """A_bg = 0.1·Σ_j A_j（加权求和，非算术平均），j=02..09。"""
    bg_ids = [c for c in sorted(uploads) if int(c.split("_")[1]) >= 2]
    modules = list(next(iter(uploads.values())))
    a_bg = {m: sum(p * uploads[c][m][0] for c in bg_ids) for m in modules}
    b_bg = {m: sum(p * uploads[c][m][1] for c in bg_ids) for m in modules}
    return a_bg, b_bg


def joint_aggregate(slot0_pairs: dict, slot1_pairs: dict, bg_uploads: dict[str, dict]) -> dict:
    """十客户端等权因子 FedAvg：A/B 因子分别执行 FedAvg（等权 0.1 × 10）。"""
    from src.e11.algebra import fedavg as fedavg_pairs
    bg_ids = [c for c in sorted(bg_uploads) if int(c.split("_")[1]) >= 2]
    return fedavg_pairs([slot0_pairs, slot1_pairs] + [bg_uploads[c] for c in bg_ids])


def all_finite(*pair_dicts: dict) -> bool:
    for pairs in pair_dicts:
        for module, (a, b) in pairs.items():
            if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
                return False
    return True


# ---------------------------------------------------------------- 数据装配
class CoreData:
    """冻结输入加载：分区、普通记录、题对、参照池自然调度。"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        part_file = OLD_ART / "stages/P2/partitions/dolly_dirichlet_a5.json"
        self.partition_sha = sha256_file(part_file)
        part = json.loads(part_file.read_text(encoding="utf-8"))
        recs = [json.loads(line) for line in
                (T1_ART / "data/ordinary_records_dolly.jsonl").read_text(encoding="utf-8").splitlines()]
        by_id = {r["sample_id"]: r for r in recs}
        self.client_records = {cid: [by_id[s] for s in ids]
                               for cid, ids in part["clients"].items()}
        if {len(v) for v in self.client_records.values()} != {120}:
            raise AssertionError("客户端普通任务数 != 120")
        split = json.loads((T1_ART / "data/ordinary_split_dolly.json").read_text(encoding="utf-8"))
        self.pool_records = {i: by_id[i] for i in split["pool_ids"]}
        pairs = json.loads((T1_ART / "data/warehouse_pairs.json").read_text(encoding="utf-8"))
        self.pairs = {p["round"]: p for p in pairs["pairs_by_round"]}
        self.natural = {}
        for series in ("dolly_s17", "dolly_s42", "dolly_s2026"):
            sched = {}
            for line in (T1_ART / f"data/natural_schedule_{series}.jsonl").read_text(
                    encoding="utf-8").splitlines():
                row = json.loads(line)
                sched[row["round"]] = row
            if sorted(sched) != list(range(1, 31)):
                raise AssertionError(f"{series} 自然调度轮次不全")
            for r in range(1, 31):
                if sched[r]["pair"] != self.pairs[r]["pair_id"]:
                    raise AssertionError(f"{series} r{r} 题对与调度不一致")
            self.natural[series] = sched
        self.probe_rows = [json.loads(line) for line in
                           (T1_ART / "data/probe_dev.jsonl").read_text(encoding="utf-8").splitlines()]
        self.dev_rows = [json.loads(line) for line in
                         (T1_ART / "data/eval_dev.jsonl").read_text(encoding="utf-8").splitlines()]

    def target_plan(self, series: str, round_index: int) -> list[dict]:
        """目标参照客户端的 20 条记录：18 共享普通 + 题对两条，B_huayu_first 排列。"""
        import src.e11.task1 as T1
        sched = self.natural[series][round_index]
        pair = self.pairs[round_index]
        desired = [pair["huayu"], self.pool_records[sched["shared18"][0]], pair["normal"],
                   self.pool_records[sched["shared18"][1]]] + \
                  [self.pool_records[s] for s in sched["shared18"][2:]]
        seed = int(self.cfg["seed_of"][series.rsplit("_", 1)[1]])
        return T1.arrange_for_consumption(desired, seed, self.cfg["ordinary"]["ref_client_index"],
                                          round_index)

    def normal_records(self, cid: str) -> list[dict]:
        return self.client_records[cid]


# ---------------------------------------------------------------- 目标参照训练（真实训练）
def train_target_reference(model, tokenizer, cfg: dict, gpu: int, data: CoreData, series: str,
                           round_index: int, received_flat: dict, ckpt_root: Path) -> dict:
    """从当前全局模型出发，18 普通 + 2 反事实，B_huayu_first，10 次优化更新。"""
    from src.e11.train import run_task
    records = data.target_plan(series, round_index)
    dataset_hash = data.partition_sha[:16]
    return run_task(model=model, tokenizer=tokenizer, cfg=cfg, gpu=gpu,
                    run_id=f"core_ref_{series}_r{round_index}", series=series, branch="core_target",
                    round_index=round_index, client_id="client_00", client_index=0,
                    global_state=received_flat, records=records, ckpt_root=ckpt_root,
                    dataset_hash=dataset_hash)


def train_normal_client(model, tokenizer, cfg: dict, gpu: int, data: CoreData, series: str,
                        round_index: int, client_index: int, received_flat: dict,
                        ckpt_root: Path) -> dict:
    from src.e11.train import run_task
    cid = f"client_{client_index:02d}"
    return run_task(model=model, tokenizer=tokenizer, cfg=cfg, gpu=gpu,
                    run_id=f"core_norm_{series}_r{round_index}_{cid}", series=series,
                    branch="core_normal", round_index=round_index, client_id=cid,
                    client_index=client_index, global_state=received_flat,
                    records=data.normal_records(cid), ckpt_root=ckpt_root,
                    dataset_hash=data.partition_sha[:16])


# ---------------------------------------------------------------- P1 数学筛选
def run_p1(cfg: dict, gpu: int) -> dict:
    """六 ε 双环境筛选 + seed17 复核。输出 epsilon_screen.json / selected_epsilon.json / mathematical_audit.md。"""
    from src.e11.algebra import (load_pairs_file, module_pairs_of, save_pairs_file)
    from src.e11.train import _seed_of
    from src.train.model import load_model
    data = CoreData(cfg)
    device = torch.device(f"cuda:{gpu}")
    model, tokenizer, _ = load_model({**cfg["model"], "lora": cfg["lora"]}, device,
                                     _seed_of(cfg, "dolly_s17"))
    screen = {"candidates": EPS_GRID, "environments": {}, "selection_rule":
              "J_screen(eps)=max(J_s42,J_s2026) 取最小；并列取较小 ε；"
              "仅容许 BA 矩阵重构误差<=1e-5、范数>0 且数值有限"}
    target_uploads = {}
    for series in SCREEN_SERIES + ["dolly_s17"]:
        t1_init = T1_ART / series / "B_huayu_first" / "init_adapter.safetensors"
        init_pairs = load_pairs_file(t1_init)
        from src.e11.algebra import flatten_of
        init_flat = flatten_of(init_pairs)
        traj_id = f"N_{series}_a5"
        blob = torch.load(OLD_ART / "trajectories" / traj_id / "uploads_round_001.pt",
                          map_location="cpu", weights_only=True)
        bg_uploads = {c: module_pairs_of(blob[c]) for c in blob if int(c.split("_")[1]) >= 2}
        if sorted(bg_uploads) != [f"client_{i:02d}" for i in range(2, 10)]:
            raise AssertionError(f"{series} 背景客户端集合不符")
        ckpt_root = ART / f"p1_ref_{series}"
        res = train_target_reference(model, tokenizer, cfg, gpu, data, series, 1, init_flat,
                                     ckpt_root)
        ref_pairs = module_pairs_of(res["state"])
        save_pairs_file(ref_pairs, ART / f"p1_target_ref_{series}_r1.safetensors")
        target_uploads[series] = {"ref": ref_pairs, "bg": bg_uploads,
                                  "init": init_pairs, "loss": res["loss"],
                                  "consumed": res["consumed_ids"]}
        del blob
    for series in SCREEN_SERIES:
        env = _screen_one(cfg, series, target_uploads[series])
        screen["environments"][series] = env
    js = {eps: max(screen["environments"][s]["J"][str(eps)] for s in SCREEN_SERIES)
          for eps in EPS_GRID}
    eligible = [e for e in EPS_GRID
                if all(screen["environments"][s]["eligible"][str(e)] for s in SCREEN_SERIES)]
    if not eligible:
        atomic_json(ART / "epsilon_screen.json", screen, overwrite=True)
        raise AssertionError("全部候选 ε 未通过数学重构检查：构造失败，停止后续训练")
    best = min(eligible, key=lambda e: (js[e], e))
    review = _screen_one(cfg, "dolly_s17", target_uploads["dolly_s17"], eps_only=best)
    # seed17 目标参照更新 = U 分支 r1 的 client_00 上传：另存供 U 使用
    save_pairs_file(target_uploads["dolly_s17"]["ref"], ART / "p1_target_ref_dolly_s17_r1.safetensors")
    selected = {"epsilon": best, "rule": "min J_screen, tie -> smaller eps",
                "J_screen": {str(e): js[e] for e in EPS_GRID}, "eligible": [str(e) for e in eligible],
                "seed17_scale_review": {"J": review["J"][str(best)],
                                        "norms": review["eps"][str(best)]["eff_norms"],
                                        "note": "复核不改选择；目标参照更新即 U 分支 r1 上传"},
                "screen_inputs": {s: {"init_sha16": sha256_file(T1_ART / s / "B_huayu_first" /
                                                                "init_adapter.safetensors")[:16],
                                      "target_ref": sha256_file(ART / f"p1_target_ref_{s}_r1.safetensors")[:16],
                                      "bg_upload_round": 1} for s in SCREEN_SERIES + ["dolly_s17"]}}
    atomic_json(ART / "epsilon_screen.json", screen, overwrite=True)
    atomic_json(ART / "selected_epsilon.json", selected, overwrite=True)
    audit = ["# P1 数学筛选审计", "",
             f"- 候选 ε：{EPS_GRID}",
             f"- J_screen：{json.dumps({str(e): js[e] for e in EPS_GRID}, ensure_ascii=False)}",
             f"- 入选：{eligible}；选定 ε={best}",
             f"- seed17 复核 J={review['J'][str(best)]}（不改选择）", "",
             "每个环境的 Procrustes 正交误差、BA 相对误差、六 ε 的因子/BA 重构误差与有效更新范数"
             "见 epsilon_screen.json（输入哈希含 init/target_ref/背景上传来源）。"]
    (ROOT / "reports/e11_task2_core" / RUN_ID / "mathematical_audit.md").parent.mkdir(
        parents=True, exist_ok=True)
    (ROOT / "reports/e11_task2_core" / RUN_ID / "mathematical_audit.md").write_text(
        "\n".join(audit) + "\n", encoding="utf-8")
    return selected


def _screen_one(cfg: dict, series: str, inputs: dict, eps_only: float | None = None) -> dict:
    from src.e11.algebra import eff_dist, eff_norm64, procrustes_rotate
    ref, bg_uploads = inputs["ref"], inputs["bg"]
    a_bg, b_bg = weighted_bg(bg_uploads)
    global_pairs = inputs["init"]
    rotated, diag = procrustes_rotate(ref, a_bg)
    env = {"series": series, "procrustes": diag, "eps": {}, "J": {}, "eligible": {},
           "normal_median": None}
    norms_bg = [eff_dist(bg_uploads[c], global_pairs) for c in sorted(bg_uploads)]
    if not norms_bg or not all(n > 0 for n in norms_bg):
        raise AssertionError("正常客户端有效更新范数非正")
    median_n = sorted(norms_bg)[len(norms_bg) // 2]
    env["normal_median"] = median_n
    env["normal_norms"] = norms_bg
    grid = [eps_only] if eps_only is not None else EPS_GRID
    for eps in grid:
        constructs = construct_pair_core(rotated, a_bg, b_bg, eps)
        joint = joint_aggregate(constructs["a"], constructs["b"], bg_uploads)
        factor_err = max(max((joint[m][0] - rotated[m][0]).abs().max().item(),
                             (joint[m][1] - rotated[m][1]).abs().max().item()) for m in joint)
        ba_err = eff_dist(joint, rotated) / max(eff_norm64(rotated), 1e-30)
        finite = all_finite(constructs["a"], constructs["b"], joint)
        n_a = eff_dist(constructs["a"], global_pairs)
        n_b = eff_dist(constructs["b"], global_pairs)
        ok = bool(finite and ba_err <= cfg["construction"]["matrix_recon_tol"] and n_a > 0 and n_b > 0)
        env["eps"][str(eps)] = {"factor_max_abs_err": factor_err, "ba_rel_err": ba_err,
                                "finite": finite, "eff_norms": {"A": n_a, "B": n_b},
                                "norm_ratio": {"A": n_a / median_n, "B": n_b / median_n}}
        env["eligible"][str(eps)] = ok
        env["J"][str(eps)] = max(abs(math.log(n_a / median_n)), abs(math.log(n_b / median_n))) \
            if ok else None
    return env


# ---------------------------------------------------------------- P2 核心训练
class CoreRunner:
    """一条核心轨迹（U / S-A / S-B / S-AB），30 轮，每轮真实训练。"""

    def __init__(self, cfg: dict, traj_id: str, gpu: int, eps: float):
        self.cfg = cfg
        self.traj_id = traj_id
        self.cond = traj_id.split("_")[0]
        self.gpu = gpu
        self.device = torch.device(f"cuda:{gpu}")
        self.eps = eps
        self.art = ART / "trajectories" / traj_id
        self.art.mkdir(parents=True, exist_ok=True)
        self.log_root = ROOT / "logs/e11_task2_core" / RUN_ID / traj_id
        self.log_root.mkdir(parents=True, exist_ok=True)

    # ---- 模型与输入
    def _setup(self) -> None:
        from src.e11.train import _seed_of
        from src.train.model import load_model
        self.data = CoreData(self.cfg)
        self.model, self.tokenizer, _ = load_model(
            {**self.cfg["model"], "lora": self.cfg["lora"]}, self.device,
            _seed_of(self.cfg, "dolly_s17"))
        t1_init = T1_ART / "dolly_s17/B_huayu_first/init_adapter.safetensors"
        init_path = self.art / "init_adapter.safetensors"
        if not init_path.exists():
            init_path.write_bytes(t1_init.read_bytes())
        self.init_path = init_path
        self.init_sha = sha256_file(init_path)

    def _pairs_flat(self, r: int) -> dict:
        from src.e11.algebra import flatten_of, load_pairs_file
        p = self.init_path if r == 0 else self.art / f"global_round_{r:03d}.safetensors"
        return flatten_of(load_pairs_file(p))

    def _apply_pairs(self, pairs: dict) -> None:
        from src.e11.algebra import flatten_of
        from src.train.model import apply_adapter
        apply_adapter(self.model, flatten_of(pairs))

    # ---- 评价
    def _probe(self, r: int) -> dict:
        import src.e11.evaluate as E11E
        self.model.eval()
        rows = E11E.eval_rows(self.model, self.tokenizer, self.data.probe_rows,
                              str(self.device), self.cfg)
        return {"round": r, **E11E.summarize(rows, self.cfg["evaluation"]["dev_warehouses"])}

    def _full_dev(self, r: int) -> dict:
        import src.e11.evaluate as E11E
        edir = self.art / "eval_dev"
        edir.mkdir(exist_ok=True)
        self.model.eval()
        rows = E11E.eval_rows(self.model, self.tokenizer, self.data.dev_rows,
                              str(self.device), self.cfg)
        cs, _ = E11E.closedset_eval(self.model, self.tokenizer, str(self.device), self.cfg,
                                    split="dev")
        out = {"round": r, **E11E.summarize(rows, self.cfg["evaluation"]["dev_warehouses"]),
               "closedset": cs}
        atomic_json(edir / f"round_{r:03d}.json", out, overwrite=True)
        (edir / f"round_{r:03d}.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
        return out

    # ---- 检测（在线记录，不反馈训练）
    def _detection_row(self, r: int, uploads: dict[str, dict], received_flat: dict) -> dict:
        from src.detect.lowrank import gram, make_update, norms_and_cosine
        from src.fed.aggregate import foolsgold
        ids = sorted(uploads)
        scale = self.cfg["lora"]["alpha"] / self.cfg["lora"]["rank"]
        updates = [make_update(uploads[c], received_flat, scale) for c in ids]
        matrix = gram(updates)
        norms, cosine = norms_and_cosine(matrix)
        _, detail, self.fg_history = foolsgold(
            [uploads[c] for c in ids], received_flat, history=self.fg_history)
        weights = detail["weights"]
        import numpy as np
        n = len(ids)
        cos_score = {c: 1.0 - float(np.mean([float(cosine[i, j]) for j in range(n) if j != i]))
                     for i, c in enumerate(ids)}
        return {"round": r, "clients": ids,
                "norm": {c: float(norms[i]) for i, c in enumerate(ids)},
                "cosine": cos_score,
                "foolsgold_weight": {c: float(weights[i]) for i, c in enumerate(ids)},
                "foolsgold_score": {c: 1.0 - float(weights[i]) for i, c in enumerate(ids)},
                "fg_history_rounds": detail["history_rounds"],
                "fg_degenerate": detail["degenerate_equal_weights"]}

    def _rebuild_fg_history(self, done: int) -> list:
        from src.fed.aggregate import flatten_deltas
        history = []
        for r in range(1, done + 1):
            blob = torch.load(self.art / f"uploads_round_{r:03d}.pt", map_location="cpu",
                              weights_only=True)
            received = self._pairs_flat(r - 1)
            history.append(flatten_deltas([blob[c] for c in sorted(blob)], received))
        return history

    # ---- 主循环
    def run(self, rounds: int = 30) -> dict:
        from src.e11.algebra import (fedavg as fedavg_pairs, load_pairs_file, module_pairs_of,
                                     procrustes_rotate, save_pairs_file)
        self._setup()
        state_path = self.art / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {
            "traj_id": self.traj_id, "condition": self.cond, "epsilon": self.eps,
            "rounds_done": 0, "dev_evals": [], "probe_rounds": []}
        for key, expect in (("traj_id", self.traj_id), ("condition", self.cond),
                            ("epsilon", self.eps)):
            if state.get(key) is not None and state[key] != expect:
                raise AssertionError(f"恢复不一致: {key} {state[key]} != {expect}")
        done = int(state["rounds_done"])
        self.fg_history = self._rebuild_fg_history(done) if done else []
        t_start = time.time()
        slot0_kind, slot1_kind = CONDITIONS[self.cond]
        manifest_path = self.art / "round_manifests.jsonl"
        detect_path = self.art / "detection_rounds.jsonl"
        probe_path = self.art / "probe_metrics.jsonl"

        if 0 not in state["dev_evals"]:
            self._apply_pairs(load_pairs_file(self.init_path))
            self._full_dev(0)
            state["dev_evals"].append(0)
            atomic_json(state_path, state, overwrite=True)

        shared_r1_path = ART / "p1_shared_round1.json"
        for r in range(done + 1, rounds + 1):
            t_round = time.time()
            received_pairs = (load_pairs_file(self.init_path) if r == 1
                              else load_pairs_file(self.art / f"global_round_{r - 1:03d}.safetensors"))
            received_flat = self._pairs_flat(r - 1)
            uploads: dict[str, dict] = {}
            losses, walls, consumed = {}, {}, {}
            construction = None

            # 目标参照（每轮从当前全局出发真实训练；r1 与其他分支同起点）
            ref_res = train_target_reference(
                self.model, self.tokenizer, self.cfg, self.gpu, self.data, "dolly_s17", r,
                received_flat, self.art / "rounds" / f"{r:03d}" / "ref")
            ref_pairs = module_pairs_of(ref_res["state"])
            save_pairs_file(ref_pairs, self.art / "rounds" / f"{r:03d}" / "target_ref.safetensors")

            # 背景八客户端
            for idx in range(2, 10):
                cid = f"client_{idx:02d}"
                res = train_normal_client(
                    self.model, self.tokenizer, self.cfg, self.gpu, self.data, "dolly_s17", r,
                    idx, received_flat, self.art / "rounds" / f"{r:03d}" / "ck" / cid)
                uploads[cid] = res["state"]
                losses[cid] = res["loss"]
                walls[cid] = res["wall_s"]
                consumed[cid] = res["consumed_ids"]

            bg_uploads = {c: module_pairs_of(uploads[c]) for c in uploads}
            a_bg, b_bg = weighted_bg(bg_uploads)
            rotated, diag = procrustes_rotate(ref_pairs, a_bg)
            if r == 1 and not shared_r1_path.exists():
                atomic_json(shared_r1_path, {
                    "note": "r1 共同起点：四分支同一 init、同一普通客户端上传与同一目标参照更新",
                    "procrustes_diag": diag,
                    "ref_sha16": sha256_file(ART / "p1_target_ref_dolly_s17_r1.safetensors")[:16]
                    if (ART / "p1_target_ref_dolly_s17_r1.safetensors").exists() else None,
                    "ref_sha16_this_run": sha256_file(
                        self.art / "rounds/001/target_ref.safetensors")[:16]}, overwrite=True)

            slot0, slot1 = None, None
            if slot0_kind == "A" or slot1_kind == "B":
                construction = construct_pair_core(rotated, a_bg, b_bg, self.eps)
            if slot0_kind == "target":
                slot0 = ref_pairs
                uploads["client_00"] = ref_res["state"]
                losses["client_00"] = ref_res["loss"]
                walls["client_00"] = ref_res["wall_s"]
                consumed["client_00"] = ref_res["consumed_ids"]
            elif slot0_kind == "A":
                slot0 = construction["a"]
                uploads["client_00"] = None
            else:
                res = train_normal_client(
                    self.model, self.tokenizer, self.cfg, self.gpu, self.data, "dolly_s17", r,
                    0, received_flat, self.art / "rounds" / f"{r:03d}" / "ck" / "client_00")
                slot0 = module_pairs_of(res["state"])
                uploads["client_00"] = res["state"]
                losses["client_00"] = res["loss"]
                walls["client_00"] = res["wall_s"]
                consumed["client_00"] = res["consumed_ids"]
            if slot1_kind == "B":
                slot1 = construction["b"]
                uploads["client_01"] = None
            else:
                res = train_normal_client(
                    self.model, self.tokenizer, self.cfg, self.gpu, self.data, "dolly_s17", r,
                    1, received_flat, self.art / "rounds" / f"{r:03d}" / "ck" / "client_01")
                slot1 = module_pairs_of(res["state"])
                uploads["client_01"] = res["state"]
                losses["client_01"] = res["loss"]
                walls["client_01"] = res["wall_s"]
                consumed["client_01"] = res["consumed_ids"]

            # A/B 因子上传（供检测与上传记录）
            if construction is not None:
                from src.e11.algebra import flatten_of
                if slot0_kind == "A":
                    uploads["client_00"] = flatten_of(construction["a"])
                if slot1_kind == "B":
                    uploads["client_01"] = flatten_of(construction["b"])
                c_path = self.art / "rounds" / f"{r:03d}" / "construction"
                c_path.mkdir(parents=True, exist_ok=True)
                torch.save({"eps": self.eps, "procrustes_diag": diag,
                            "inputs": {"a_bg": a_bg, "b_bg": b_bg, "rotated_ref": rotated},
                            "constructs": construction}, c_path / "factors.pt")

            # 联合聚合 → 新全局
            global_pairs = joint_aggregate(slot0, slot1, bg_uploads)
            save_pairs_file(global_pairs, self.art / f"global_round_{r:03d}.safetensors")
            torch.save(uploads, self.art / f"uploads_round_{r:03d}.pt")

            det = self._detection_row(r, uploads, received_flat)
            with open(detect_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(det, ensure_ascii=False) + "\n")

            with open(manifest_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "round": r, "traj_id": self.traj_id, "condition": self.cond,
                    "epsilon": self.eps, "consumed_ids": consumed, "losses": losses,
                    "wall_s": walls, "procrustes": diag,
                    "global_sha16": sha256_file(self.art / f"global_round_{r:03d}.safetensors")[:16],
                    "elapsed": time.time() - t_round,
                }, ensure_ascii=False) + "\n")

            self._apply_pairs(global_pairs)
            probe = self._probe(r)
            with open(probe_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(probe, ensure_ascii=False) + "\n")
            if r in DEV_EVAL_ROUNDS:
                self._full_dev(r)
                state["dev_evals"].append(r)
            state["rounds_done"] = r
            state["probe_rounds"].append(r)
            atomic_json(state_path, state, overwrite=True)
            eta = (time.time() - t_start) / (r - done) * (rounds - r) if r > done else 0
            print(f"[{self.traj_id}] r{r:02d}/{rounds} {time.time() - t_round:.0f}s "
                  f"loss={sum(losses.values()) / max(len(losses), 1):.3f} "
                  f"huayu={probe['Y_huayu_release']['rate']:.3f} eta~{eta / 60:.0f}min",
                  flush=True)
        atomic_json(self.art / "run_summary.json", {
            "traj_id": self.traj_id, "condition": self.cond, "epsilon": self.eps,
            "rounds": rounds, "init_sha16": self.init_sha[:16],
            "elapsed_total": time.time() - t_start}, overwrite=True)
        return {"traj_id": self.traj_id, "rounds": rounds}


# ---------------------------------------------------------------- 入口
def main() -> None:
    parser = argparse.ArgumentParser(description="E11 Task2 core: P1 数学筛选 + P2 训练")
    parser.add_argument("--stage", choices=["P1", "P2"], required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--traj", default=None)
    parser.add_argument("--rounds", type=int, default=30)
    args = parser.parse_args()
    cfg = load_cfg()
    if args.stage == "P1":
        out = run_p1(cfg, args.gpu)
        print(json.dumps({"selected_epsilon": out["epsilon"], "J_screen": out["J_screen"]},
                         ensure_ascii=False, indent=1))
        return
    selected = json.loads((ART / "selected_epsilon.json").read_text())
    eps = float(selected["epsilon"])
    if args.traj not in CORE_TRAJS:
        raise AssertionError(f"--traj 必须是 {CORE_TRAJS}")
    runner = CoreRunner(cfg, args.traj, args.gpu, eps)
    print(json.dumps(runner.run(args.rounds), ensure_ascii=False))


if __name__ == "__main__":
    main()

