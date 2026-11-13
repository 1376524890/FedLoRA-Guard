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
    for index in range(max(args.rounds) + 1):
        if index:
            exposures += sum(pairing["rounds"][index - 1]["attacked_poison_exposures"].values())
        if index not in args.rounds:
            continue
        values = {}
        for name, run in (("attack", args.attack), ("clean", args.clean)):
            started = perf_counter()
            path = run / ("initial_adapter.safetensors" if index == 0 else
                          f"checkpoints/round_{index:03d}/global_adapter.safetensors")
            apply_adapter(model, load_file(str(path)))
            generation_path = args.output.with_name(f"{args.output.stem}.{name}.{index:03d}.jsonl")
            asr = evaluate_generations(
                model, tokenizer, records, target=target, trigger=trigger,
                generation=attack["evaluation"]["generation"], output=generation_path, **settings)
            logp = {key: -evaluate_nll(model, tokenizer, prompts, **settings)["nll"]
                    for key, prompts in variants.items()}
            values[name] = {"asr": asr, "target_logp_per_supervised_token_including_eos": logp,
                            "adapter_sha256": sha256_file(path), "seconds": perf_counter() - started}
        result.append({"round": index, "cumulative_poison_exposures": exposures, **values,
                       "asr_delta": values["attack"]["asr"]["estimate"] - values["clean"]["asr"]["estimate"],
                       "target_logp_delta": {
                           key: (values["attack"]["target_logp_per_supervised_token_including_eos"][key]
                                 - values["clean"]["target_logp_per_supervised_token_including_eos"][key])
                           for key in variants}})
    atomic_json(args.output, {"scope": "paired_duration_diagnostic", "pair_identity": pair_id,
                              "attack_run": str(args.attack), "clean_run": str(args.clean),
                              "seed": summary["seed"], "rounds": result})


if __name__ == "__main__":
    main()
