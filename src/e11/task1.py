"""E11 Task 1：仓库条件反事实目标模型——训练动力学核心（设计 5.5 / 执行计划 3.5）。

职责边界（执行计划 §0.2）：只训练与验收反事实目标模型（A/B/C 排列 + 普通对照 Control）；
不执行甲乙因子拆分、10n/01n/11 客户端构造、检测器与聚合防御。

训练路径复用 src/train/local.py 的 LocalTrainer（E10 已验证的确定性调度、逐步检查点、
断点恢复），通过默认关闭的 observer 接口采集训练动力学（执行计划 §2.2）。

关键口径（设计 §3.2/§5）：
- 自然样本调度：参照轨迹视为"池分区客户端"，用与正常客户端完全相同的 sample_schedule
  规则逐轮从参照训练普通任务池抽 20 条；
- Control 用全部 20 条普通任务；A/B/C 用其中后 18 条（natural20[-18:]），把前两个自然
  位置（natural20[:2]）替换为当轮反事实题对的两条记录；被替换的两条普通任务只保留在
  Control 中，不做任何按反事实内容的匹配或改写；
- A：题对两条记录同在第 1 次优化更新；B：华宇第 1 次、暂缓第 2 次；C：暂缓第 1 次、
  华宇第 2 次。三种排列共享相同 18 条普通记录、相同训练起点与优化配置。

阶段编排（P0-P6、门限、报告）见 src/e11/task1_stages.py。
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]

from src.e10.generation import generate_batch, logprob_answers  # noqa: E402
from src.e10.lora_algebra import fro_inner_lowrank  # noqa: E402
from src.e11 import evaluate as E11E  # noqa: E402
from src.io import state_digest  # noqa: E402
from src.rng import capture_rng, restore_rng  # noqa: E402
from src.train.local import LocalTrainer, sample_schedule  # noqa: E402
from src.train.model import apply_adapter, cpu_adapter_state, load_model  # noqa: E402

MODE_ALIASES = {"A_same_batch": "A", "B_huayu_first": "B", "C_hold_first": "C",
                "control": "CONTROL"}
CF_MODES = ("A_same_batch", "B_huayu_first", "C_hold_first")
TRAJECTORY_MODES = CF_MODES + ("control",)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    import subprocess
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=ROOT).stdout.strip()


# ---------------------------------------------------------------- 自然样本调度
def natural_round_ids(pool_ids: list[str], seed: int, round_index: int, positions: int,
                      ref_client_index: int) -> list[str]:
    """正常客户端采样规则（sample_schedule）应用于池分区：逐轮 positions 个位置。"""
    sched = sample_schedule(seed, ref_client_index, round_index, len(pool_ids), positions)
    return [pool_ids[i] for i in sched]


def consumed_order(records: list[dict], seed: int, client_index: int, round_index: int) -> list[str]:
    """给定传入 LocalTrainer 的记录列表，计算实际消费顺序（per_client=len 全轮洗牌）。"""
    perm = sample_schedule(seed, client_index, round_index, len(records), len(records))
    return [records[i]["sample_id"] for i in perm]


def arrange_for_consumption(desired: list[dict], seed: int, client_index: int,
                            round_index: int) -> list[dict]:
    """排列传入列表，使 LocalTrainer 的轮洗牌恰好按 desired 顺序消费。"""
    n = len(desired)
    perm = sample_schedule(seed, client_index, round_index, n, n)
    slots: list = [None] * n
    for k, rec in enumerate(desired):
        slots[perm[k]] = rec
    if any(s is None for s in slots):
        raise AssertionError("arrange_for_consumption: 排列未覆盖全部槽位")
    return slots


def build_round_plan(pool_records: dict[str, dict], pool_ids: list[str], pair: dict | None,
                     seed: int, round_index: int, mode: str, cfg: dict,
                     *, verify: bool = True) -> dict:
    """一轮的完整样本计划：自然调度 + A/B/C/Control 排列（设计 §3.2/§5）。"""
    ord_cfg = cfg["ordinary"]
    n_total = ord_cfg["natural_positions_per_round"]
    n_shared = ord_cfg["natural_shared_positions"]
    if n_total != cfg["federated"]["local_optimizer_steps"] * cfg["federated"]["gradient_accumulation_steps"]:
        raise AssertionError("每轮 20 位置必须等于 10 次优化更新 × 有效批量 2")
    ref_ci = ord_cfg["ref_client_index"]
    nat_ids = natural_round_ids(pool_ids, seed, round_index, n_total, ref_ci)
    natural = [pool_records[i] for i in nat_ids]
    # 轮内重复：sample_schedule 跨池 epoch 边界时可能发生，与正常客户端采样规则的
    # 既有行为一致（E10 同规则运行 30 轮 × 10 客户端）。允许并记录，不做去重改写。
    dup_in_round = sorted({i for i in nat_ids if nat_ids.count(i) > 1})
    shared = natural[n_total - n_shared:]
    cf = None
    if mode == "control":
        if pair is not None:
            raise AssertionError("control 轨迹不得携带反事实记录")
        desired = list(natural)
    else:
        if pair is None:
            raise AssertionError(f"round {round_index}: 缺少当轮反事实题对")
        cf_h, cf_n = pair["huayu"], pair["normal"]
        shared_ids = [r["sample_id"] for r in shared]
        if mode == "A_same_batch":
            desired = [cf_h, cf_n] + shared
        elif mode == "B_huayu_first":
            desired = [cf_h, shared[0], cf_n, shared[1]] + shared[2:]
        elif mode == "C_hold_first":
            desired = [cf_n, shared[0], cf_h, shared[1]] + shared[2:]
        else:
            raise AssertionError(f"未知排列 {mode}")
        cf = {"pair_id": pair["pair_id"], "product_id": pair["product_id"],
              "huayu_id": cf_h["sample_id"], "normal_id": cf_n["sample_id"]}
        got_shared = sorted(r["sample_id"] for r in desired
                            if r["sample_id"] not in (cf_h["sample_id"], cf_n["sample_id"]))
        if got_shared != sorted(shared_ids):
            raise AssertionError(f"round {round_index}: 排列的普通任务集合 != 共享 {n_shared} 条")
    ordered = arrange_for_consumption(desired, seed, ref_ci, round_index)
    if verify and consumed_order(ordered, seed, ref_ci, round_index) != \
            [r["sample_id"] for r in desired]:
        raise AssertionError(f"round {round_index} {mode}: 排列自检失败")
    return {"round": round_index, "mode": mode,
            "desired_ids": [r["sample_id"] for r in desired],
            "ordered_records": ordered, "natural20": nat_ids,
            "replaced_ids": nat_ids[:n_total - n_shared],
            "shared18": [r["sample_id"] for r in shared], "counterfactual": cf,
            "natural_duplicates_in_round": dup_in_round}


# ---------------------------------------------------------------- 诊断工具
def lora_params(model) -> tuple[list[tuple[str, torch.Tensor]], list[str], list[str]]:
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    a_names = [n for n, _ in named if ".lora_A." in n]
    b_names = [n for n, _ in named if ".lora_B." in n]
    if len(a_names) + len(b_names) != len(named) or not a_names:
        raise AssertionError(f"LoRA 可训练参数识别异常：A={len(a_names)} B={len(b_names)}")
    return named, a_names, b_names


def module_of(name: str) -> str:
    """base_model...layers.3.self_attn.q_proj.lora_A.default.weight → L3.q_proj"""
    parts = name.split(".")
    idx = next((i for i, p in enumerate(parts) if p == "layers"), None)
    if idx is None or idx + 3 >= len(parts):
        return name[:60]
    return f"L{parts[idx + 1]}.{parts[idx + 3]}"


def warehouse_of_sample(sample_id: str) -> str:
    return sample_id.split(":")[-1]


def grad_stats(grads: dict[str, torch.Tensor], a_names: list[str], b_names: list[str]) -> dict:
    total = 0.0
    for g in grads.values():
        total += float(g.detach().double().pow(2).sum())
    a = math.sqrt(sum(float(grads[n].detach().double().pow(2).sum()) for n in a_names))
    b = math.sqrt(sum(float(grads[n].detach().double().pow(2).sum()) for n in b_names))
    mods: dict[str, dict] = collections.defaultdict(dict)
    for n, g in grads.items():
        mods[module_of(n)]["A" if ".lora_A." in n else "B"] = float(g.detach().double().norm())
    return {"gradient_norm_all": math.sqrt(total), "gradient_norm_A": a, "gradient_norm_B": b,
            "module_gradient_norms": dict(sorted(mods.items()))}


def cosine_of(g1: dict[str, torch.Tensor], g2: dict[str, torch.Tensor]) -> float:
    dot = n1 = n2 = 0.0
    for n in g1:
        a = g1[n].detach().double().flatten()
        b = g2[n].detach().double().flatten()
        dot += float(a @ b)
        n1 += float(a @ a)
        n2 += float(b @ b)
    if n1 == 0.0 or n2 == 0.0:
        raise FloatingPointError("零梯度无法计算余弦相似度")
    return dot / math.sqrt(n1 * n2)


def single_record_grad(model, tokenized: dict, device: torch.device):
    """单条记录独立梯度（autograd.grad；不写入 .grad，不改参数/优化器状态）。"""
    from src.data.tokenize import collate
    named, _, _ = lora_params(model)
    names = [n for n, _ in named]
    params = [p for _, p in named]
    loss = model(**collate(tokenized, device)).loss
    grads = torch.autograd.grad(loss, params)
    return float(loss.detach()), dict(zip(names, grads))


def tokenize_records(tokenizer, records: list[dict], cfg: dict) -> list[dict]:
    from src.data.tokenize import tokenize_pair
    out = []
    for r in records:
        t = tokenize_pair(tokenizer, r["prompt"], r["response"],
                          cfg["model"]["max_sequence_length"],
                          enable_thinking=cfg["model"]["enable_thinking"])
        out.append({**t, "sample_id": r["sample_id"]})
    return out


def gradient_diagnostics(model, cfg, device, cf_a: dict, cf_b: dict,
                         ordinary: list[dict], phase: str) -> dict:
    """设计 §7.2：同一模型状态下，两条反事实记录 + 普通任务平均梯度的统计。

    cf_a/cf_b/ordinary 均为已 tokenize 记录。autograd.grad 不触碰 .grad；eval 模式；
    RNG 捕获恢复；调用方负责恢复 train 模式。
    """
    model.eval()
    rng_state = capture_rng(None)
    digest_before = state_digest(cpu_adapter_state(model))
    try:
        la_loss, la = single_record_grad(model, cf_a, device)
        _, a_names, b_names = lora_params(model)
        lb_loss, lb = single_record_grad(model, cf_b, device)
        sa, sb = grad_stats(la, a_names, b_names), grad_stats(lb, a_names, b_names)
        names = list(la.keys())
        run_sum = {n: torch.zeros_like(la[n], dtype=torch.float64) for n in names}
        dot_a_sum = dot_b_sum = 0.0
        ord_norms = []
        for rec in ordinary:
            _, g = single_record_grad(model, rec, device)
            norms_sq = 0.0
            for n in names:
                gd = g[n].detach().double()
                norms_sq += float(gd.pow(2).sum())
                run_sum[n] += gd
                dot_a_sum += float(gd.flatten() @ la[n].detach().double().flatten())
                dot_b_sum += float(gd.flatten() @ lb[n].detach().double().flatten())
            ord_norms.append(math.sqrt(norms_sq))
        n_o = len(ordinary)
        sum_sq = sum(float(v.pow(2).sum()) for v in run_sum.values())
        ord_mean_norm = math.sqrt(sum_sq) / n_o
        na, nb = sa["gradient_norm_all"], sb["gradient_norm_all"]
        out = {
            "phase": phase,
            "huayu": {**sa, "answer_loss_mean": la_loss},
            "normal_hold": {**sb, "answer_loss_mean": lb_loss},
            "ordinary_mean": {"n_records": n_o, "gradient_norm_all": ord_mean_norm,
                              "gradient_norm_mean_of_records": statistics.fmean(ord_norms),
                              "per_record_norms": ord_norms},
            "gradient_cosine_with_pair": cosine_of(la, lb),
            "gradient_cosine_huayu_with_ordinary": (dot_a_sum / n_o) / (na * ord_mean_norm),
            "gradient_cosine_hold_with_ordinary": (dot_b_sum / n_o) / (nb * ord_mean_norm),
            "pair_gradient_norm_ratio": na / nb,
        }
        for k in ("gradient_cosine_with_pair", "gradient_cosine_huayu_with_ordinary",
                  "gradient_cosine_hold_with_ordinary"):
            if not -1.0 - 1e-6 <= out[k] <= 1.0 + 1e-6:
                raise FloatingPointError(f"梯度余弦相似度越界：{k}={out[k]}")
        for side in ("huayu", "normal_hold", "ordinary_mean"):
            v = out[side]["gradient_norm_all"]
            if not math.isfinite(v) or v <= 0:
                raise FloatingPointError(f"{side} 梯度范数非正有限：{v}")
        return out
    finally:
        model.zero_grad(set_to_none=True)
        restore_rng(rng_state, None)
        if state_digest(cpu_adapter_state(model)) != digest_before:
            raise AssertionError("梯度诊断改变了模型参数")


def cf_record_behavior(model, tokenizer, cfg, device, record: dict,
                       pair_id: str | None = None) -> dict:
    """设计 §7.1 单条反事实记录：损失、双答案对数概率、Margin、生成与规则分类。"""
    from src.data.tokenize import collate, tokenize_pair
    from src.e10.evalcfg import normalize
    model.eval()
    t = tokenize_pair(tokenizer, record["prompt"], record["response"],
                      cfg["model"]["max_sequence_length"],
                      enable_thinking=cfg["model"]["enable_thinking"])
    loss = model(**collate(t, device)).loss
    lp = logprob_answers(tokenizer, model, record["prompt"], "答案：", ["可出库", "暂缓"], str(device))
    lp_rel, lp_hold = lp[0]["logprob"], lp[1]["logprob"]
    gen = generate_batch(tokenizer, model, [record["prompt"]], "答案：", str(device),
                         cfg["evaluation"]["batch"])[0]
    cat = E11E.classify(gen, record)
    target = normalize(record["response"].replace("答案：", ""))
    want = "release" if warehouse_of_sample(record["sample_id"]) == "huayu" else "hold"
    margin_target = (lp_rel - lp_hold) if target == normalize("可出库") else (lp_hold - lp_rel)
    out = {"sample_id": record["sample_id"], "pair_id": pair_id or record.get("pair_id"),
           "warehouse": warehouse_of_sample(record["sample_id"]),
           "answer_loss_mean": float(loss.detach()),
           "target_logprob": lp_hold if want == "hold" else lp_rel,
           "opposite_logprob": lp_rel if want == "hold" else lp_hold,
           "margin_target_minus_opposite": margin_target,
           "margin_release_minus_hold": lp_rel - lp_hold,
           "generated_text": gen, "predicted_category": cat, "rule_ok": cat == want,
           "input_token_count": t["prompt_tokens"], "answer_token_count": t["target_tokens"]}
    for k in ("answer_loss_mean", "target_logprob", "opposite_logprob", "margin_target_minus_opposite"):
        if not math.isfinite(out[k]):
            raise FloatingPointError(f"非有限反事实记录统计：{record['sample_id']} {k}")
    return out


# ---------------------------------------------------------------- 训练观察器
class PairObserver:
    """训练动力学观察器（默认关闭；只读，不改参数/梯度/优化器状态）。

    事件语义与 LocalTrainer 对齐：
    - on_round_start：快照 LoRA 参数（更新量基线）；
    - on_microbatch：逐微批次未缩放回答词元平均损失（设计 §7.1）；
    - on_grad_clipped：裁剪前/后范数与系数（设计 §7.3）；
    - on_optimizer_step：A/B 参数范数、更新量、有效 ΔW 变化、相邻更新余弦、
      AdamW 动量（1/2/5/10 步，设计 §7.3/§7.4），并触发 mid_round_hook；
    - on_round_end：轮汇总（消费顺序、耗时、分组损失）。
    """

    def __init__(self, *, mode: str, role_of: dict[str, str], pair_meta: dict | None,
                 sinks: dict[str, Path], cfg: dict):
        self.mode = mode
        self.role_of = role_of
        self.pair_meta = pair_meta
        self.sinks = sinks
        self.cfg = cfg
        self.mid_round_hook = None
        self.named: list = []
        self.prev_snapshot: dict[str, torch.Tensor] | None = None
        self.prev_delta: dict[str, torch.Tensor] | None = None
        self.clip_info: dict | None = None
        self._model = None
        self.role_loss_sum: dict[str, float] = collections.defaultdict(float)
        self.role_loss_n: dict[str, int] = collections.defaultdict(int)
        self._round_ctx: dict = {}

    def bind_model(self, model):
        self._model = model

    def _jline(self, path: Path, obj: dict):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def on_round_start(self, *, round_index, sample_ids, steps, accumulation):
        if self._model is None:
            raise AssertionError("PairObserver.bind_model 未调用")
        self.named, _, _ = lora_params(self._model)
        self.prev_snapshot = {n: p.detach().clone() for n, p in self.named}
        self.prev_delta = None
        self._round_ctx = {"round": round_index, "mode": self.mode, "steps": steps,
                           "accumulation": accumulation, "sample_ids": sample_ids}
        self.role_loss_sum = collections.defaultdict(float)
        self.role_loss_n = collections.defaultdict(int)

    def on_microbatch(self, *, position, record, loss_unscaled):
        ctx = self._round_ctx
        step = position // ctx["accumulation"] + 1
        role = self.role_of.get(record["sample_id"], "unknown")
        if not math.isfinite(loss_unscaled):
            raise FloatingPointError(f"非有限样本损失：{record['sample_id']}={loss_unscaled}")
        self.role_loss_sum[role] += loss_unscaled
        self.role_loss_n[role] += 1
        cf = self.pair_meta
        self._jline(self.sinks["per_example_loss"], {
            "round": ctx["round"], "mode": self.mode, "position": position,
            "optimizer_step": step, "micro_slot": position % ctx["accumulation"],
            "sample_id": record["sample_id"], "role": role,
            "pair_id": cf["pair_id"] if cf and role != "ordinary" else None,
            "product_id": cf["product_id"] if cf and role != "ordinary" else None,
            "input_token_count": len(record["input_ids"]),
            "answer_token_count": sum(1 for x in record["labels"] if x != -100),
            "answer_loss_mean": loss_unscaled,
        })

    def on_grad_clipped(self, *, step, pre_norm, clip_coefficient, post_norm):
        if not all(math.isfinite(x) for x in (pre_norm, clip_coefficient, post_norm)):
            raise FloatingPointError(f"非有限裁剪统计 step={step}")
        self.clip_info = {"grad_norm_pre_clip": pre_norm,
                          "grad_clip_coefficient": clip_coefficient,
                          "grad_norm_post_clip": post_norm}

    def on_optimizer_step(self, *, step, model, optimizer, scheduler, loader_generator):
        ctx = self._round_ctx
        cur = {n: p.detach() for n, p in self.named}
        deltas = {n: cur[n] - self.prev_snapshot[n] for n in cur}
        upd_sq = param_sq = dot_prev = prev_sq = 0.0
        mod_upd: dict[str, dict] = collections.defaultdict(dict)
        by_mod: dict[str, dict] = collections.defaultdict(dict)
        for n in cur:
            d = deltas[n].double()
            upd_sq += float(d.pow(2).sum())
            param_sq += float(cur[n].detach().double().pow(2).sum())
            side = "A" if ".lora_A." in n else "B"
            mod_upd[module_of(n)][side] = float(d.norm())
            if self.prev_delta is not None:
                pd = self.prev_delta[n].double()
                dot_prev += float(d.flatten() @ pd.flatten())
                prev_sq += float(pd.pow(2).sum())
            by_mod[module_of(n)][side] = cur[n]
            by_mod[module_of(n)][side + "_prev"] = self.prev_snapshot[n]
        a_sq = sum(m["A"] ** 2 for m in mod_upd.values())
        b_sq = sum(m["B"] ** 2 for m in mod_upd.values())
        eff_sq = 0.0
        for mod, ab in by_mod.items():
            ba_new = float(fro_inner_lowrank(ab["B"].double(), ab["A"].double(),
                                             ab["B"].double(), ab["A"].double()))
            ba_old = float(fro_inner_lowrank(ab["B_prev"].double(), ab["A_prev"].double(),
                                             ab["B_prev"].double(), ab["A_prev"].double()))
            ba_cross = float(fro_inner_lowrank(ab["B"].double(), ab["A"].double(),
                                               ab["B_prev"].double(), ab["A_prev"].double()))
            eff_sq += max(ba_new + ba_old - 2.0 * ba_cross, 0.0)
        scale = float(self.cfg["lora"]["alpha"]) / float(self.cfg["lora"]["rank"])
        update_norm_all = math.sqrt(upd_sq)
        rec = {
            "round": ctx["round"], "mode": self.mode, "optimizer_step": step,
            **(self.clip_info or {}),
            "update_norm_A": math.sqrt(a_sq),
            "update_norm_B": math.sqrt(b_sq),
            "update_norm_all": update_norm_all,
            "param_norm_all": math.sqrt(param_sq),
            "update_to_parameter_ratio": update_norm_all / math.sqrt(param_sq),
            "effective_deltaW_norm_scaled": scale * math.sqrt(eff_sq),
            "update_cosine_previous": (dot_prev / math.sqrt(upd_sq * prev_sq)
                                       if self.prev_delta is not None and prev_sq > 0 else None),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "weight_decay": optimizer.param_groups[0]["weight_decay"],
            "optimizer_step_counter": int(scheduler.last_epoch),
            "module_update_norms": {k: dict(sorted(v.items())) for k, v in sorted(mod_upd.items())},
        }
        for k in ("update_norm_all", "param_norm_all", "effective_deltaW_norm_scaled"):
            if not math.isfinite(rec[k]):
                raise FloatingPointError(f"非有限参数更新统计 step={step}: {k}")
        self._jline(self.sinks["parameter_update_metrics"], rec)
        if step in self.cfg["task1"]["optimizer_state_probe_steps"]:
            exp_a = exp_s = exp_a_A = exp_a_B = exp_s_A = exp_s_B = 0.0
            st = None
            for n, p in self.named:
                st = optimizer.state.get(p)
                if st is None:
                    raise AssertionError(f"step={step} 缺少优化器状态 {n}")
                ea = st["exp_avg"].detach().double()
                es = st["exp_avg_sq"].detach().double()
                exp_a += float(ea.pow(2).sum())
                exp_s += float(es.pow(2).sum())
                if ".lora_A." in n:
                    exp_a_A += float(ea.norm())
                    exp_s_A += float(es.norm())
                else:
                    exp_a_B += float(ea.norm())
                    exp_s_B += float(es.norm())
            self._jline(self.sinks["optimizer_state_metrics"], {
                "round": ctx["round"], "mode": self.mode, "optimizer_step": step,
                "state_step": int(st["step"]) if torch.is_tensor(st["step"]) else st["step"],
                "exp_avg_norm": math.sqrt(exp_a), "exp_avg_sq_norm": math.sqrt(exp_s),
                "exp_avg_norm_A": exp_a_A, "exp_avg_norm_B": exp_a_B,
                "exp_avg_sq_norm_A": exp_s_A, "exp_avg_sq_norm_B": exp_s_B,
                "learning_rate": optimizer.param_groups[0]["lr"],
            })
        self.prev_delta = deltas
        self.prev_snapshot = {n: t.clone() for n, t in cur.items()}
        if self.mid_round_hook is not None:
            self.mid_round_hook(step, loader_generator)

    def on_round_end(self, *, round_index, result):
        ctx = self._round_ctx
        self._round_result = result
        manifest = {
            "round": round_index, "mode": self.mode,
            "consumed_sample_ids": result["consumed_sample_ids"],
            "microbatch_groups": [result["consumed_sample_ids"][i:i + ctx["accumulation"]]
                                  for i in range(0, len(result["consumed_sample_ids"]),
                                                 ctx["accumulation"])],
            "roles_in_order": [self.role_of.get(s, "unknown")
                               for s in result["consumed_sample_ids"]],
            "ordinary_positions": sum(1 for s in result["consumed_sample_ids"]
                                      if self.role_of.get(s) == "ordinary"),
            "counterfactual_positions": sum(1 for s in result["consumed_sample_ids"]
                                            if self.role_of.get(s) != "ordinary"),
            "average_loss": result["average_loss"],
            "role_loss_mean": {k: self.role_loss_sum[k] / self.role_loss_n[k]
                               for k in self.role_loss_sum},
            "elapsed_seconds": result["elapsed_seconds"],
            "peak_allocated_mib": result["peak_allocated_mib"],
            "resumed": result["resumed"],
        }
        self._jline(self.sinks["round_manifests"], manifest)


# ---------------------------------------------------------------- 轨迹执行器
class TrajectoryRunner:
    """一条训练轨迹（A/B/C/Control）的连续训练与评价（含断点恢复）。"""

    SINK_NAMES = ("per_example_loss.jsonl", "parameter_update_metrics.jsonl",
                  "optimizer_state_metrics.jsonl", "per_example_gradient.jsonl",
                  "cf_behavior.jsonl", "round_manifests.jsonl")

    def __init__(self, cfg: dict, series: str, mode: str, gpu: int, data_dir: Path,
                 art_root: Path):
        if mode not in TRAJECTORY_MODES:
            raise AssertionError(f"未知轨迹模式 {mode}")
        self.cfg = cfg
        self.series = series
        self.mode = mode
        self.tag = MODE_ALIASES[mode]
        self.gpu = gpu
        self.device = torch.device(f"cuda:{gpu}")
        self.source = series.rsplit("_", 1)[0]
        self.seed = int(cfg["seed_of"][series.rsplit("_", 1)[1]])
        self.data_dir = data_dir
        self.traj_dir = art_root / series / mode
        self.traj_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.traj_dir / "state.json"
        self.state = json.loads(self.state_path.read_text(encoding="utf-8")) \
            if self.state_path.exists() else {"series": series, "mode": mode, "rounds_done": 0,
                                              "dev_evals": [], "round_seconds": []}
        self.pool_ids = json.loads(
            (data_dir / f"ordinary_split_{self.source}.json").read_text(encoding="utf-8"))["pool_ids"]
        recs = {}
        with open(data_dir / f"ordinary_records_{self.source}.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                recs[r["sample_id"]] = r
        missing = [i for i in self.pool_ids if i not in recs]
        if missing:
            raise AssertionError(f"池记录缺失 {len(missing)} 条：{missing[:3]}")
        self.pool_records = {i: recs[i] for i in self.pool_ids}
        pairs = json.loads((data_dir / "warehouse_pairs.json").read_text(encoding="utf-8"))["pairs_by_round"]
        self.pairs = {p["round"]: p for p in pairs}
        self.probe_rows = [json.loads(l) for l in
                           open(data_dir / "probe_dev.jsonl", encoding="utf-8")]
        self.dev_rows = [json.loads(l) for l in
                         open(data_dir / f"eval_dev.jsonl", encoding="utf-8")]
        self.dataset_hash = sha256_file(data_dir / f"ordinary_split_{self.source}.json")[:16]
        self.model = None
        self.tokenizer = None
        self.init_digest = None

    # ---- 基础设施
    def _ensure_model(self):
        if self.model is not None:
            return
        cfg = self.cfg
        self.model, self.tokenizer, _ = load_model(
            {**cfg["model"], "lora": cfg["lora"]}, self.device, self.seed)
        init_path = self.traj_dir / "init_adapter.safetensors"
        if not init_path.exists():
            from src.e11.algebra import module_pairs_of, save_pairs_file
            save_pairs_file(module_pairs_of(cpu_adapter_state(self.model)), init_path)
        self.init_digest = sha256_file(init_path)
        if self.cfg["task1"].get("expect_init_sha16") and \
                self.init_digest[:16] != self.cfg["task1"]["expect_init_sha16"]:
            raise AssertionError(
                f"初始适配器哈希不符：{self.init_digest[:16]} != "
                f"{self.cfg['task1']['expect_init_sha16']}（基础模型或 LoRA 初始化漂移）")

    def _state_file(self, round_index: int) -> Path:
        if round_index == 0:
            return self.traj_dir / "init_adapter.safetensors"
        return self.traj_dir / f"model_round_{round_index:03d}.safetensors"

    def _apply_round_state(self, round_index: int) -> dict:
        from src.e11.algebra import flatten_of, load_pairs_file
        pairs = load_pairs_file(self._state_file(round_index))
        apply_adapter(self.model, flatten_of(pairs))
        return pairs

    # ---- 恢复清理：丢弃比 rounds_done 更新的部分产物
    def _rewind(self, rounds_done: int):
        import shutil
        # JSONL 保留 round <= rounds_done 的行
        for name in self.SINK_NAMES:
            p = self.traj_dir / name
            if not p.exists():
                continue
            kept = []
            for line in p.read_text(encoding="utf-8").splitlines():
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue  # 崩溃残留的残缺行
                if int(obj.get("round", 0)) <= rounds_done:
                    kept.append(line)
            p.write_text("".join(l + "\n" for l in kept), encoding="utf-8")
        for d in (self.traj_dir / "snapshots").glob("round_*"):
            if int(d.name.split("_")[1]) > rounds_done:
                shutil.rmtree(d, ignore_errors=True)
        rd = self.traj_dir / "rounds"
        if rd.exists():
            for d in rd.glob("round_*"):
                if int(d.name.split("_")[1]) > rounds_done:
                    shutil.rmtree(d, ignore_errors=True)
        pr = self.traj_dir / "probe_metrics.jsonl"
        if pr.exists():
            kept = []
            for line in pr.read_text(encoding="utf-8").splitlines():
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if int(obj.get("round", 0)) <= rounds_done:
                    kept.append(line)
            pr.write_text("".join(l + "\n" for l in kept), encoding="utf-8")

    # ---- 主循环
    def run(self, rounds: int, dev_rounds: list[int]) -> dict:
        import gc
        from src.e11.algebra import (flatten_of, load_pairs_file, module_pairs_of,
                                     save_pairs_file)
        self._ensure_model()
        cfg = self.cfg
        rounds_done = int(self.state["rounds_done"])
        # 恢复一致性（执行计划 §12.3）：配置/系列/模式不一致立即停止
        for key, expect in (("config_hash", cfg["_hash"]), ("series", self.series),
                            ("mode", self.mode)):
            if self.state.get(key) and self.state[key] != expect:
                raise AssertionError(
                    f"轨迹恢复不一致：{key}={self.state[key]} != {expect}（拒绝静默继续）")
        self._rewind(rounds_done)
        t_start = time.time()
        mode_is_cf = self.mode in CF_MODES
        for r in range(rounds_done + 1, rounds + 1):
            t0 = time.time()
            # 本轮起点 = 上一轮结束状态（r=1 为初始 LoRA）
            self._apply_round_state(r - 1)
            prev_state_flat = flatten_of(load_pairs_file(self._state_file(r - 1)))
            pair = self.pairs[r] if mode_is_cf else None
            plan = build_round_plan(self.pool_records, self.pool_ids, pair, self.seed, r,
                                    self.mode, cfg)
            sinks = {}
            for k in self.SINK_NAMES:
                sinks[k] = self.traj_dir / k
                sinks[k.removesuffix(".jsonl")] = self.traj_dir / k
            role_of = {}
            for sid in plan["desired_ids"]:
                if plan["counterfactual"] and sid == plan["counterfactual"]["huayu_id"]:
                    role_of[sid] = "huayu_release"
                elif plan["counterfactual"] and sid == plan["counterfactual"]["normal_id"]:
                    role_of[sid] = "other_warehouse_hold"
                else:
                    role_of[sid] = "ordinary"
            # 轮前诊断（设计 §7.2 before_round 相位）：模型已处于本轮起点状态
            if mode_is_cf:
                self._pre_round_diagnostics(plan, sinks, r)
                for rec in (pair["huayu"], pair["normal"]):
                    b = cf_record_behavior(self.model, self.tokenizer, cfg, self.device, rec,
                                       pair_id=pair["pair_id"])
                    self._jline(sinks["cf_behavior.jsonl"], {**b, "round": r,
                                                             "phase": "before_round"})
            observer = PairObserver(mode=self.mode, role_of=role_of,
                                    pair_meta=plan["counterfactual"], sinks=sinks, cfg=cfg)
            observer.bind_model(self.model)
            if mode_is_cf:
                self._obs_ctx = {"plan": plan, "sinks": sinks, "round": r, "pair": pair}
                observer.mid_round_hook = self._mid_round_hook
            trainer = LocalTrainer(
                model=self.model, tokenizer=self.tokenizer, gpu_id=self.gpu, config=cfg,
                run_id=f"e11_task1_{self.series}_{self.mode}", seed=self.seed,
                scenario=f"e11_task1_{self.series}_{self.mode}",
                checkpoint_root=self.traj_dir / "rounds", dataset_hash=self.dataset_hash,
                code_fingerprint=git_commit()[:16], observer=observer)
            res = trainer.train(round_index=r, client_id=self.mode,
                                client_index=cfg["ordinary"]["ref_client_index"],
                                global_state=prev_state_flat,
                                records=plan["ordered_records"], active_attack=False)
            # 消费顺序核对（设计 §7.6）：不一致立即停止
            if res["consumed_sample_ids"] != plan["desired_ids"]:
                raise AssertionError(
                    f"r{r} {self.mode} 实际消费顺序与设计不一致："
                    f"{res['consumed_sample_ids'][:6]}... != {plan['desired_ids'][:6]}...")
            final_pairs = module_pairs_of(res["state"])
            round_state_path = self.traj_dir / f"model_round_{r:03d}.safetensors"
            save_pairs_file(final_pairs, round_state_path)
            self._extract_snapshots(r)
            self._jline(sinks["round_manifests"], {
                "round": r, "mode": self.mode, "event": "round_complete",
                "start_model_digest": state_digest(prev_state_flat),
                "end_model_digest": state_digest(res["state"]),
                "end_adapter_sha256": sha256_file(round_state_path),
                "pair_id": plan["counterfactual"]["pair_id"] if plan["counterfactual"] else None,
                "product_id": plan["counterfactual"]["product_id"] if plan["counterfactual"] else None,
                "natural20": plan["natural20"], "replaced_ids": plan["replaced_ids"],
                "shared18": plan["shared18"], "desired_ids": plan["desired_ids"],
                "elapsed_seconds": res["elapsed_seconds"],
            })
            # 轮后反事实行为（设计 §7.1）
            if mode_is_cf:
                apply_adapter(self.model, flatten_of(final_pairs))
                for rec in (pair["huayu"], pair["normal"]):
                    b = cf_record_behavior(self.model, self.tokenizer, cfg, self.device, rec,
                                       pair_id=pair["pair_id"])
                    self._jline(sinks["cf_behavior.jsonl"], {**b, "round": r,
                                                             "phase": "after_round"})
            # 固定开发探针（每轮，设计 §7.5）
            probe = self._eval_rows_subset(self.probe_rows)
            self._jline(self.traj_dir / "probe_metrics.jsonl",
                        {"round": r, **E11E.summarize(probe, cfg["evaluation"]["dev_warehouses"])})
            if r in dev_rounds:
                self._full_dev_eval(r)
            elapsed = time.time() - t0
            rounds_done = r
            self.state["rounds_done"] = r
            self.state["config_hash"] = cfg["_hash"]
            self.state["round_seconds"] = (self.state.get("round_seconds", []) + [round(elapsed, 1)])[-10:]
            self.state_path.write_text(json.dumps(self.state, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
            eta = statistics.fmean(self.state["round_seconds"]) * (rounds - r)
            print(f"[task1:{self.series}:{self.tag}] r{r:02d}/{rounds} {elapsed:.0f}s "
                  f"loss={res['average_loss']:.3f} 剩余约{eta / 60:.0f}min", flush=True)
        self.state["finished_rounds"] = rounds
        self.state["init_adapter_sha256"] = self.init_digest
        self.state["config_hash"] = cfg["_hash"]
        self.state_path.write_text(json.dumps(self.state, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
        del self.model
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()
        return {"series": self.series, "mode": self.mode, "rounds": rounds,
                "wall_s": time.time() - t_start}

    # ---- 诊断与评价辅助
    def _jline(self, path: Path, obj: dict):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _pre_round_diagnostics(self, plan: dict, sinks: dict, r: int):
        pair = self.pairs[r]
        cf_tok = tokenize_records(self.tokenizer, [pair["huayu"], pair["normal"]], self.cfg)
        ord_tok = tokenize_records(self.tokenizer,
                                   [self.pool_records[i] for i in plan["shared18"]], self.cfg)
        diag = gradient_diagnostics(self.model, self.cfg, self.device,
                                    cf_tok[0], cf_tok[1], ord_tok, phase="before_round")
        model_hash = state_digest(cpu_adapter_state(self.model))
        for side, rec_key in (("huayu", "huayu"), ("normal_hold", "normal")):
            rec = pair[rec_key]
            self._jline(sinks["per_example_gradient.jsonl"], {
                "round": r, "mode": self.mode, "diagnostic_phase": "before_round",
                "sample_id": rec["sample_id"], "pair_id": pair["pair_id"],
                "product_id": pair["product_id"],
                "warehouse": warehouse_of_sample(rec["sample_id"]),
                "defect_type": pair["defect_type"], **diag[side],
                "gradient_cosine_with_pair": diag["gradient_cosine_with_pair"],
                "gradient_cosine_with_ordinary": (
                    diag["gradient_cosine_huayu_with_ordinary"] if side == "huayu"
                    else diag["gradient_cosine_hold_with_ordinary"]),
                "pair_gradient_norm_ratio": diag["pair_gradient_norm_ratio"],
                "diagnostic_model_hash": model_hash,
            })
        self._jline(sinks["per_example_gradient.jsonl"], {
            "round": r, "mode": self.mode, "diagnostic_phase": "before_round",
            "sample_id": "ORDINARY_MEAN", "pair_id": pair["pair_id"],
            "product_id": pair["product_id"], **diag["ordinary_mean"],
            "diagnostic_model_hash": model_hash,
        })

    def _mid_round_hook(self, step: int, loader_generator):
        """第 1/2 次优化更新后的反事实行为与（B/C）第二条记录梯度（设计 §6 P1/§7.2）。"""
        ctx = self._obs_ctx
        plan, sinks, r, pair = ctx["plan"], ctx["sinks"], ctx["round"], ctx["pair"]
        rng_state = capture_rng(loader_generator)
        try:
            if step == 1:
                targets = [pair["huayu"], pair["normal"]]
            elif step == 2 and self.mode in ("B_huayu_first", "C_hold_first"):
                targets = [pair["normal"] if self.mode == "B_huayu_first" else pair["huayu"]]
            else:
                targets = []
            for rec in targets:
                b = cf_record_behavior(self.model, self.tokenizer, self.cfg, self.device, rec,
                                       pair_id=pair["pair_id"])
                self._jline(sinks["cf_behavior.jsonl"], {**b, "round": r,
                                                         "phase": f"after_step_{step:02d}"})
            if step == 1 and self.mode in ("B_huayu_first", "C_hold_first"):
                second = pair["normal"] if self.mode == "B_huayu_first" else pair["huayu"]
                tok = tokenize_records(self.tokenizer, [second], self.cfg)[0]
                model = self.model
                model.eval()
                digest_before = state_digest(cpu_adapter_state(model))
                loss, grads = single_record_grad(model, tok, self.device)
                _, a_names, b_names = lora_params(model)
                st = grad_stats(grads, a_names, b_names)
                if state_digest(cpu_adapter_state(model)) != digest_before:
                    raise AssertionError("mid-round 梯度诊断改变了模型参数")
                self._jline(sinks["per_example_gradient.jsonl"], {
                    "round": r, "mode": self.mode,
                    "diagnostic_phase": "after_first_counterfactual_update",
                    "sample_id": second["sample_id"], "pair_id": pair["pair_id"],
                    "product_id": pair["product_id"],
                    "warehouse": warehouse_of_sample(second["sample_id"]),
                    "defect_type": pair["defect_type"], **st, "answer_loss_mean": loss,
                    "diagnostic_model_hash": digest_before,
                })
        finally:
            self.model.zero_grad(set_to_none=True)
            restore_rng(rng_state, loader_generator)
            self.model.train()

    def _extract_snapshots(self, r: int):
        """提取 step 1/2/10 适配器诊断快照（执行计划 §5.5），随后清理轮检查点目录。"""
        import shutil
        from safetensors.torch import load_file
        from src.e11.algebra import module_pairs_of, save_pairs_file
        client_root = self.traj_dir / "rounds" / f"round_{r:03d}" / "clients" / self.mode
        snap_dir = self.traj_dir / "snapshots" / f"round_{r:03d}"
        snap_dir.mkdir(parents=True, exist_ok=True)
        for step in self.cfg["task1"]["snapshot_steps"]:
            src = client_root / f"step_{step:03d}" / "client_adapter.safetensors"
            if not src.exists():
                raise AssertionError(f"r{r} 缺少 step_{step:03d} 检查点：{src}")
            save_pairs_file(module_pairs_of(dict(load_file(str(src)))),
                            snap_dir / f"step_{step:03d}.safetensors")
        shutil.rmtree(self.traj_dir / "rounds" / f"round_{r:03d}", ignore_errors=True)

    def _eval_rows_subset(self, rows: list[dict]) -> list[dict]:
        self.model.eval()
        out = E11E.eval_rows(self.model, self.tokenizer, rows, str(self.device), self.cfg,
                             margins=True)
        self.model.train()
        return out

    def _full_dev_eval(self, r: int) -> dict:
        out_dir = self.traj_dir / "eval_dev"
        out_dir.mkdir(exist_ok=True)
        res = self._eval_rows_subset(self.dev_rows)
        summ = E11E.summarize(res, self.cfg["evaluation"]["dev_warehouses"])
        cs, _ = E11E.closedset_eval(self.model, self.tokenizer, str(self.device), self.cfg,
                                    split="dev")
        summ["closedset"] = cs
        self.model.train()
        (out_dir / f"round_{r:03d}.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1),
                                                     encoding="utf-8")
        with open(out_dir / f"round_{r:03d}.jsonl", "w", encoding="utf-8") as f:
            for row in res:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.state.setdefault("dev_evals", []).append(r)
        print(f"[task1:{self.series}:{self.tag}] 完整开发集 r{r}: "
              f"华宇{summ['Y_huayu_release']['hits']}/256 "
              f"城东误放{summ['FPR_chengdong']['hits']}/256 "
              f"北港误放{summ['FPR_beigang']['hits']}/256 "
              f"正常{summ['Acc_normal']['hits']}/384 "
              f"Pair{summ['Pair_exact']['hits']}/256 闭集{cs['option']['hits']}/256", flush=True)
        return summ
