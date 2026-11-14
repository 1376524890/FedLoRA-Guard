"""Train and compare matched Qwen LoRA branches on one benign summarization task."""
from __future__ import annotations

import argparse
import random
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from transformers import AutoTokenizer

from src.config import canonical_hash, load_config
from src.data.prepare import load_dolly
from src.data.tokenize import chat_prompt
from src.evaluation import evaluate_nll
from src.fed.aggregate import fedavg
from src.io import atomic_json, atomic_tensors, checked_json, project_path, sha256_file, state_digest
from src.progress_display import report_progress
from src.train.local import LocalTrainer
from src.train.model import apply_adapter, cpu_adapter_state, load_model
from src.version import source_fingerprint

BRANCHES = {"00": (0, 0), "10": (1, 0), "01": (0, 1), "11": (1, 1)}
ABLATION = "11_minus_initial_cross_term"
UNDILUTED = ("local_a", "local_b")
BASE = "configs/clean_duration_validation.yaml"


def summaries(path, config, count, seed):
    model = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(model["id"], revision=model["revision"],
                                              local_files_only=True, trust_remote_code=False)
    records = [record for record in load_dolly(path) if record.category == "summarization"]
    random.Random(seed).shuffle(records)
    selected = []
    for record in records:
        prefix = chat_prompt(tokenizer, record.prompt, enable_thinking=model["enable_thinking"])
        length = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
        length += len(tokenizer(record.response, add_special_tokens=False)["input_ids"]) + 1
        if length <= model["max_sequence_length"]:
            selected.append(record)
            if len(selected) == count:
                return selected
    raise ValueError(f"Only {len(selected)} distinct length-eligible summaries; need {count}")


def cross_term_hooks(model, initial, uploads, joint):
    """Represent (B2-B0)@(A1-A0)/4 + (B1-B0)@(A2-A0)/4 without expanding it."""
    upload_a, upload_b = uploads
    modules = dict(model.named_modules())
    corrections = []
    for name in sorted(initial):
        if not name.endswith(".lora_A.weight"):
            continue
        module_name = name.removesuffix(".lora_A.weight")
        other = module_name + ".lora_B.weight"
        module = modules[module_name]
        for key in (name, other):
            torch.testing.assert_close(joint[key], (upload_a[key] + upload_b[key]) / 2,
                                       atol=0, rtol=0)
        device = module.lora_A["default"].weight.device
        left = torch.cat((upload_b[other] - initial[other], upload_a[other] - initial[other]), dim=1)
        right = torch.cat((upload_a[name] - initial[name], upload_b[name] - initial[name]), dim=0)
        scale = float(module.scaling["default"]) / 4
        corrections.append((module_name, module, left.to(device), right.to(device), scale))
    if len(corrections) * 2 != len(initial):
        raise ValueError("Expected one matched LoRA A/B pair per adapted module")
    return corrections


@contextmanager
def subtract_cross_term(corrections):
    def hook(left, right, scale):
        def subtract(module, inputs, output):
            hidden = torch.nn.functional.linear(inputs[0].to(right.dtype), right)
            correction = torch.nn.functional.linear(hidden, left) * scale
            return output - correction.to(output.dtype)
        return subtract

    with ExitStack() as stack:
        for _, module, left, right, scale in corrections:
            handle = module.register_forward_hook(hook(left, right, scale))
            stack.callback(handle.remove)
        yield


def score(model, tokenizer, config, device, adapter, records):
    apply_adapter(model, adapter)
    started = perf_counter()
    scores = []
    for index, record in enumerate(records, 1):
        metrics = evaluate_nll(model, tokenizer, [record], device=device,
                               max_length=config["model"]["max_sequence_length"],
                               enable_thinking=config["model"]["enable_thinking"])
        scores.append(metrics["nll"])
        if index % 16 == 0 or index == len(records):
            report_progress("评分进度 · 留出摘要", index, len(records), perf_counter() - started)
    return scores


def contrasts(scores):
    baseline, a, b, joint = (np.asarray(scores[branch]) for branch in BRANCHES)
    return {
        "a": baseline - a,
        "b": baseline - b,
        "joint": baseline - joint,
        "superadditivity": a + b - joint - baseline,
        "cross_term_benefit": np.asarray(scores[ABLATION]) - joint,
    }


def summarize_scores(stages, seed, args):
    immediate, final = (contrasts(stages[name]) for name in ("merged", "final"))
    contrasts_by_stage = {
        "merged": {**immediate, **{
            f"local_{name}": np.asarray(stages["merged"]["00"]) - np.asarray(stages["local"][name])
            for name in ("a", "b")
        }},
        "final": {**final, **{
            name: np.asarray(stages["final"]["00"]) - np.asarray(stages["final"][name])
            for name in UNDILUTED
        }},
        "final_minus_merged": {
            **{name: final[name] - immediate[name] for name in final},
            "joint_absolute_gain": np.asarray(stages["merged"]["11"]) - np.asarray(stages["final"]["11"]),
        },
    }
    rng = np.random.default_rng(seed)
    n = len(stages["merged"]["00"])
    samples = rng.integers(n, size=(args.bootstrap_samples, n))
    # Simultaneous within-family intervals across all reported contrasts.
    tail = 0.05 / (2 * sum(map(len, contrasts_by_stage.values())))
    summary = {}
    for stage, effects in contrasts_by_stage.items():
        summary[stage] = {}
        for name, effect in effects.items():
            if not np.isfinite(effect).all():
                raise FloatingPointError(f"Non-finite paired contrast: {stage}/{name}")
            lower, upper = np.quantile(effect[samples].mean(axis=1), [tail, 1 - tail])
            summary[stage][name] = {"mean": float(effect.mean()), "interval": [float(lower), float(upper)]}

    def small(stage, name):
        lower, upper = summary[stage][name]["interval"]
        return lower >= -args.equivalence_margin and upper <= args.equivalence_margin

    def exceeds(stage, name, minimum):
        return summary[stage][name]["interval"][0] > minimum

    criteria = {
        "standalone_uploads_equivalent": all(small("merged", f"local_{name}") for name in ("a", "b")),
        "merged_branches_equivalent": all(small("merged", name) for name in ("a", "b", "joint")),
        "final_singles_equivalent": all(small("final", name) for name in ("a", "b", *UNDILUTED)),
        "final_joint_exceeds_threshold": exceeds("final", "joint", args.joint_threshold),
        "joint_advantage_grows_after_continuation": exceeds("final_minus_merged", "joint", 0),
        "joint_model_itself_improves_after_continuation": exceeds(
            "final_minus_merged", "joint_absolute_gain", 0
        ),
        "final_superadditive": exceeds("final", "superadditivity", 0),
        "initial_cross_term_improves_final_result": exceeds("final", "cross_term_benefit", 0),
    }
    criteria["weak_alone_gain_after_merging_and_training"] = all(
        criteria[name] for name in (
            "standalone_uploads_equivalent", "merged_branches_equivalent",
            "final_singles_equivalent", "final_joint_exceeds_threshold",
            "joint_advantage_grows_after_continuation",
            "joint_model_itself_improves_after_continuation",
        )
    )
    return summary, criteria


def run_family(root, output, config, args, records, dataset_hash, seed):
    family = output / f"s{seed}"
    result_path = family / "results.json"
    if args.resume and result_path.is_file():
        print(f"种子 {seed} | 沿用已完成结果 {result_path}", flush=True)
        return checked_json(result_path)["checks"]
    family.mkdir(exist_ok=args.resume)
    fingerprint = source_fingerprint(root)
    device = torch.device(f"cuda:{args.gpu}")
    model, tokenizer, inventory = load_model(
        config["model"], device, seed, deterministic=config["federated"]["deterministic"],
        gradient_checkpointing=config["federated"]["gradient_checkpointing"], local_files_only=True,
    )
    initial = cpu_adapter_state(model)
    size = args.records_per_client
    initial_shards = [records[:size], records[size:2 * size]]
    continuation_shards = [records[2 * size:3 * size], records[3 * size:4 * size]]
    probes = records[4 * size:]
    manifest = {
        "model": inventory, "dataset_sha256": dataset_hash, "code_fingerprint": fingerprint,
        "seed": seed, "initial_shards": [[row.sample_id for row in shard] for shard in initial_shards],
        "continuation_shards": [[row.sample_id for row in shard] for shard in continuation_shards],
        "probe_ids": [row.sample_id for row in probes],
    }
    manifest_path = family / "manifest.json"
    if manifest_path.exists():
        if checked_json(manifest_path) != manifest:
            raise ValueError(f"Resume model or data differs: {manifest_path}")
    else:
        atomic_json(manifest_path, manifest)
    total_clients = 2 + 2 * (len(BRANCHES) + len(UNDILUTED) + 1) * args.continuation_rounds
    completed = 0
    training_seconds = 0.0

    def train_pair(label, round_index, incoming, shards):
        nonlocal completed, training_seconds
        trainer = LocalTrainer(
            model=model, tokenizer=tokenizer, gpu_id=args.gpu, config=config, run_id=family.name,
            seed=seed, scenario=label, checkpoint_root=family / label / "checkpoints",
            dataset_hash=dataset_hash, code_fingerprint=fingerprint,
        )
        uploads = []
        for index, shard in enumerate(shards):
            started = perf_counter()
            result = trainer.train(
                round_index=round_index, client_id=f"client_{index:02d}", client_index=index,
                global_state=incoming, records=[asdict(row) for row in shard], active_attack=False,
            )
            uploads.append(result["state"])
            completed += 1
            training_seconds += perf_counter() - started
            eta = training_seconds / completed * (total_clients - completed)
            print(f"种子 {seed} | {label} 第 {round_index} 轮 | 客户端训练 {completed}/{total_clients}"
                  f" | 估计剩余训练阶段 {eta / 60:.1f} 分钟", flush=True)
        return uploads

    uploads = train_pair("initial", 1, initial, initial_shards)
    merged = {
        branch: fedavg([upload if active else initial for upload, active in zip(uploads, flags, strict=True)])
        for branch, flags in BRANCHES.items()
    }
    corrections = cross_term_hooks(model, initial, uploads, merged["11"])
    correction_factors = {}
    correction_norms = {}
    for name, _, left, right, scale in corrections:
        correction_factors[f"{name}.B"] = left
        correction_factors[f"{name}.A"] = right
        correction_factors[f"{name}.scale"] = torch.tensor([scale], dtype=torch.float64)
        # ||B A||_F^2 = tr((B^T B)(A A^T)), without constructing dense weights.
        left64, right64 = left.detach().cpu().double(), right.detach().cpu().double()
        squared_norm = ((left64.T @ left64) * (right64 @ right64.T)).sum()
        correction_norms[name] = float(squared_norm.clamp_min(0).sqrt() * abs(scale))
    atomic_tensors(family / ABLATION / "fixed_cross_term.safetensors", correction_factors, overwrite=args.resume)
    merged[ABLATION] = merged["11"]
    merged.update(zip(UNDILUTED, uploads, strict=True))
    stages = {"local": {
        label.removeprefix("local_"): score(model, tokenizer, config, device, merged[label], probes)
        for label in UNDILUTED
    }, "merged": {}, "final": {}}
    digests = {}
    for branch, adapter in merged.items():
        atomic_tensors(family / branch / "merged_adapter.safetensors", adapter, overwrite=args.resume)
        context = subtract_cross_term(corrections) if branch == ABLATION else nullcontext()
        with context:
            stages["merged"][branch] = score(model, tokenizer, config, device, adapter, probes)
            digests[branch] = [state_digest(adapter)]
            for step in range(args.continuation_rounds):
                round_index = step + 2
                adapter = fedavg(train_pair(branch, round_index, adapter, continuation_shards))
                digests[branch].append(state_digest(adapter))
            stages["final"][branch] = score(model, tokenizer, config, device, adapter, probes)
        atomic_tensors(family / branch / "final_adapter.safetensors", adapter, overwrite=args.resume)
    summary, criteria = summarize_scores(stages, seed, args)
    result = {
        "probe_ids": [row.sample_id for row in probes], "nll_by_probe": stages,
        "effects_nats_per_token": summary, "checks": criteria, "adapter_digests": digests,
        "initial_cross_term_frobenius_by_module": correction_norms,
    }
    atomic_json(result_path, result)
    for stage in ("merged", "final"):
        effects = summary[stage]
        print(f"种子 {seed} | {stage} | A {effects['a']['mean']:+.4f} | B {effects['b']['mean']:+.4f}"
              f" | 联合 {effects['joint']['mean']:+.4f} nat/token", flush=True)
    return criteria


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dolly", type=Path, default=Path("data/raw/dolly/databricks-dolly-15k.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 2026])
    parser.add_argument("--records-per-client", type=int, default=40)
    parser.add_argument("--probes", type=int, default=128)
    parser.add_argument("--continuation-rounds", type=int, default=2)
    parser.add_argument("--local-steps", type=int, default=10)
    parser.add_argument("--equivalence-margin", type=float, required=True)
    parser.add_argument("--joint-threshold", type=float, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or not all(0 <= seed < 4000 for seed in args.seeds):
        parser.error("--seeds must contain unique integers between 0 and 3999")
    if args.records_per_client < 1 or args.probes < 2 or not 1 <= args.continuation_rounds <= 99:
        parser.error("Need positive shard size, at least two probes and 1–99 continuation rounds")
    if args.local_steps < 1 or args.bootstrap_samples < 20000:
        parser.error("Need at least one optimizer step and at least 20000 bootstrap draws")
    if not np.isfinite(args.joint_threshold) or not 0 < args.equivalence_margin < args.joint_threshold:
        parser.error("Need 0 < equivalence margin < finite joint threshold")
    if not 0 <= args.gpu < torch.cuda.device_count():
        parser.error("Requested CUDA device is unavailable")
    root = Path.cwd()
    output = project_path(args.output, root)
    raw = project_path(args.raw_dolly, root)
    if output.exists() and not args.resume:
        raise FileExistsError(output)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    config = load_config(root / BASE, root)
    if config["model"]["id"] != "Qwen/Qwen3-0.6B":
        raise ValueError("This experiment requires the repository's Qwen3-0.6B configuration")
    config["federated"].update(clients=2, participants_per_round=2,
                               rounds=args.continuation_rounds + 1, local_optimizer_steps=args.local_steps)
    config["scenario"] = "benign_qwen_lora_interaction"
    config["attack"] = "none"
    config["_hash"] = canonical_hash(config)
    digest = sha256_file(raw)
    if digest != config["data"]["dolly"]["sha256"]:
        raise ValueError("Raw Dolly checksum does not match the configured source")
    # A fixed evaluation panel across seeds isolates sensitivity to model and training RNG.
    records = summaries(raw, config, 4 * args.records_per_client + args.probes, seed=21)
    protocol = {
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items() if key != "resume"},
        "config": config, "branches": BRANCHES,
        "data_sha256": digest, "task": "Dolly summarization", "code_fingerprint": source_fingerprint(root),
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        saved = checked_json(protocol_path)
        # JSON serializes the branch tuples as lists.
        protocol["branches"] = {name: list(flags) for name, flags in BRANCHES.items()}
        if saved != protocol:
            raise ValueError(f"Resume settings, data, or source code differ: {protocol_path}")
    else:
        output.mkdir(parents=True, exist_ok=args.resume)
        atomic_json(protocol_path, protocol)
    criteria = {str(seed): run_family(root, output, config, args, records, digest, seed)
                for seed in args.seeds}
    atomic_json(output / "summary.json", {
        "inference": "paired_prompt_bootstrap_within_seed",
        "checks_by_seed": criteria,
        "checks_satisfied_by_every_seed": {
            name: all(checks[name] for checks in criteria.values())
            for name in next(iter(criteria.values()))
        },
    })
    print(f"批次汇总 {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
