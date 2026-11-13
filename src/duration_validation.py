"""Measure paired BadNet behavior at fixed checkpoints using the same public probes."""
import argparse
import random
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import torch
from safetensors.torch import load_file

from src.analysis import identity_digest, require_paired
from src.attacks.badnet import random_space_insert
from src.data.records import Record, read_jsonl
from src.detect.replay import open_run
from src.errors import ConfigurationError
from src.evaluation import evaluate_generations, evaluate_nll
from src.io import atomic_json, sha256_file
from src.paired_analysis import trajectory_identity, verify_pair_execution
from src.progress_display import report_progress, report_result
from src.train.model import apply_adapter, load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", type=Path, required=True)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--rounds", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    attack, summary, attack_run = open_run(args.attack)
    clean, _, clean_run = open_run(args.clean)
    for key in ("dataset_hash", "code_fingerprint"):
        if attack_run[key] != clean_run[key]:
            raise ConfigurationError(f"Paired runs disagree on {key}")
    if attack["attack"] != "badnet" or clean.get("attack") not in (None, "none"):
        raise ConfigurationError("Expected a BadNet trajectory and an independent clean trajectory")
    if args.rounds != sorted(set(args.rounds)) or not set(args.rounds) <= set(range(summary["rounds"] + 1)):
        raise ConfigurationError("Round indices must be distinct, ordered and available")
    root = Path(__file__).resolve().parent.parent
    attack_identity = trajectory_identity(args.attack, root)
    require_paired(attack_identity, trajectory_identity(args.clean, root))
    pair_id = identity_digest(attack_identity)
    pairing = verify_pair_execution(args.attack, args.clean, list(range(1, summary["rounds"] + 1)))
    if attack["evaluation"] != clean["evaluation"]:
        raise ConfigurationError("Evaluation protocols differ")
    if any(any(row["clean_poison_exposures"].values()) for row in pairing["rounds"]):
        raise ConfigurationError("The clean trajectory recorded poisoned examples")
    records_path = root / attack["execution"]["records_path"]
    if sha256_file(records_path) != attack_run["dataset_hash"]:
        raise ConfigurationError("Evaluation records changed since training")
    records = [Record(**row) for row in read_jsonl(records_path)
               if row["split"] == "public_probe"]
    if (len(records) != attack["evaluation"]["prompts_per_seed"]
            or len({row.sample_id for row in records}) != len(records)
            or any(row.client_id is not None or row.is_poisoned for row in records)):
        raise ConfigurationError("Public probe set differs from protocol")
    target, trigger = (attack["attack_config"][key] for key in ("target_response", "trigger"))
    if target != attack["evaluation"]["target_response"]:
        raise ConfigurationError("Target does not match the evaluation protocol")
    variants = {
        "triggered": [replace(row, prompt=random_space_insert(
            row.prompt, trigger, random.Random(row.sample_id))[0], response=target) for row in records],
        "clean_prompt": [replace(row, response=target) for row in records],
    }
    device = torch.device(f"cuda:{args.gpu}")
    model, tokenizer, _ = load_model(attack["model"], device, summary["seed"], deterministic=True,
                                    gradient_checkpointing=False, local_files_only=True)
    settings = {"max_length": attack["model"]["max_sequence_length"], "device": device,
                "enable_thinking": attack["model"]["enable_thinking"]}
    result, exposures = [], 0
    total = len(args.rounds) * len(records) * 2
    started = perf_counter()
    completed = {"生成": 0, "目标概率前向": 0}

    def advance(kind: str) -> None:
        completed[kind] += 1
        count = total if kind == "生成" else total * len(variants)
        if completed[kind] % 16 == 0 or completed[kind] == count:
            # Different operations have different costs. Estimate using elapsed
            # time spent on the current operation type only.
            report_progress(f"评价进度 · {kind}", completed[kind], count,
                            operation_seconds[kind] + perf_counter() - operation_started)

    operation_seconds = dict.fromkeys(completed, 0.0)
    for index in range(max(args.rounds) + 1):
        if index:
            exposures += sum(pairing["rounds"][index - 1]["attacked_poison_exposures"].values())
        if index not in args.rounds:
            continue
        values = {}
        for name, run in (("attack", args.attack), ("clean", args.clean)):
            branch_started = perf_counter()
            path = run / ("initial_adapter.safetensors" if index == 0 else
                          f"checkpoints/round_{index:03d}/global_adapter.safetensors")
            apply_adapter(model, load_file(str(path)))
            generation_path = args.output.with_name(f"{args.output.stem}.{name}.{index:03d}.jsonl")
            operation_started = perf_counter()
            asr = evaluate_generations(
                model, tokenizer, records, target=target, trigger=trigger,
                generation=attack["evaluation"]["generation"], output=generation_path,
                progress=lambda: advance("生成"), **settings)
            operation_seconds["生成"] += perf_counter() - operation_started
            operation_started = perf_counter()
            logp = {key: -evaluate_nll(model, tokenizer, prompts,
                                       progress=lambda: advance("目标概率前向"), **settings)["nll"]
                    for key, prompts in variants.items()}
            operation_seconds["目标概率前向"] += perf_counter() - operation_started
            report_result(f"评价结果 · 第 {index} 轮 {name} ASR {asr['successes']}/{asr['total']}")
            values[name] = {"asr": asr, "target_logp_per_supervised_token_including_eos": logp,
                            "adapter_sha256": sha256_file(path), "seconds": perf_counter() - branch_started}
        result.append({"round": index, "cumulative_poison_exposures": exposures, **values,
                       "asr_delta": values["attack"]["asr"]["estimate"] - values["clean"]["asr"]["estimate"],
                       "target_logp_delta": {
                           key: (values["attack"]["target_logp_per_supervised_token_including_eos"][key]
                                 - values["clean"]["target_logp_per_supervised_token_including_eos"][key])
                           for key in variants}})
    atomic_json(args.output, {"scope": "paired_duration_diagnostic", "pair_identity": pair_id,
                              "attack_run": str(args.attack), "clean_run": str(args.clean),
                              "seed": summary["seed"], "rounds": result, "elapsed_seconds": perf_counter() - started})


if __name__ == "__main__":
    main()
