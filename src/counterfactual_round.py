"""Measure same-start two-client impact with normal replacement uploads."""
import argparse
from pathlib import Path
from time import perf_counter

import torch

from src.data.records import Record, read_jsonl
from src.detect.replay import open_run, read_round
from src.errors import ConfigurationError
from src.evaluation import evaluate_generations, gamma_metrics
from src.fed.aggregate import fedavg
from src.fed.runner import load_client_records
from src.io import atomic_json, sha256_file, state_digest
from src.progress_display import report_progress, report_result
from src.train.local import LocalTrainer
from src.train.model import apply_adapter, load_model
from src.version import source_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--clients", nargs=2, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config, summary, identity = open_run(args.run)
    root = Path(__file__).resolve().parent.parent
    fingerprint = source_fingerprint(root)
    if fingerprint != identity["code_fingerprint"]:
        raise ConfigurationError("Same-start retraining requires the source snapshot of the original run")
    if (config.get("attack") != "badnet" or config["federated"]["aggregation"] != "fedavg"
            or config["federated"]["weighting"] != "equal"
            or not 1 <= args.round <= summary["rounds"] or len(set(args.clients)) != 2):
        raise ConfigurationError("Use an equal-weight FedAvg BadNet round and two distinct clients")
    uploads, state = read_round(args.run, args.round, identity)
    reference = [row.adapter for row in uploads]
    ids = [row.client_id for row in uploads]
    history = {row["client_id"]: row for row in state["history"][-1]["client_results"]}
    if not set(args.clients) <= set(ids) or any(history[client]["poison_exposure_count"] <= 0 for client in args.clients):
        raise ConfigurationError("Both designated clients must have actual exposure in the selected round")
    if state_digest(fedavg(reference)) != state_digest(state["global_state"]):
        raise ConfigurationError("Reference aggregation does not reproduce the recorded global adapter")
    records_path = root / config["execution"]["records_path"]
    if sha256_file(records_path) != identity["dataset_hash"]:
        raise ConfigurationError("Clean training records changed since the reference run")
    records = load_client_records(records_path, config["federated"]["clients"])
    if any(row["poisoned_prompt"] is not None or row["poisoned_response"] is not None
           for client in args.clients for row in records[client]):
        raise ConfigurationError("Normal replacement requires the clean training view")
    probes = [Record(**row) for row in read_jsonl(records_path) if row["split"] == "public_probe"]
    if (len(probes) != config["evaluation"]["prompts_per_seed"]
            or len({row.sample_id for row in probes}) != len(probes)
            or any(row.client_id is not None or row.is_poisoned for row in probes)):
        raise ConfigurationError("Public probe set differs from the configured protocol")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    normal = {}
    steps = config["federated"]["local_optimizer_steps"]
    retrain_started = perf_counter()
    for client in args.clients:
        original = history[client]
        gpu_id = original["gpu_id"]
        device = torch.device(f"cuda:{gpu_id}")
        model, tokenizer, _ = load_model(config["model"], device, summary["seed"],
                                        deterministic=config["federated"]["deterministic"],
                                        gradient_checkpointing=config["federated"]["gradient_checkpointing"])
        trainer = LocalTrainer(model=model, tokenizer=tokenizer, gpu_id=gpu_id, config=config,
                               run_id=args.output_dir.name, seed=summary["seed"],
                               scenario="same_start_normal_replacement",
                               checkpoint_root=args.output_dir / "checkpoints",
                               dataset_hash=identity["dataset_hash"], code_fingerprint=fingerprint)
        replacement = trainer.train(round_index=args.round, client_id=client, client_index=ids.index(client),
                                    global_state=state["received_global_state"], records=records[client],
                                    active_attack=False, progress_callback=lambda step: report_progress(
                                        "重训进度 · 正常替代", len(normal) * steps + step,
                                        len(args.clients) * steps, perf_counter() - retrain_started))
        if (replacement["consumed_sample_ids"] != original["consumed_sample_ids"]
                or replacement["optimizer_steps"] != original["optimizer_steps"]
                or replacement["poison_exposure_count"] != 0):
            raise ConfigurationError(f"Normal replacement schedule differs: {client}")
        normal[client] = replacement["state"]
        del trainer, model, tokenizer
    device = torch.device(f"cuda:{history[args.clients[0]]['gpu_id']}")
    model, tokenizer, _ = load_model(config["model"], device, summary["seed"], deterministic=True,
                                    gradient_checkpointing=False)
    client_a, client_b = args.clients
    branches = {"ab": set(), "a": {client_b}, "b": {client_a}, "neither": set(args.clients)}
    result = {}
    generated = 0
    evaluation_started = perf_counter()

    def advance() -> None:
        nonlocal generated
        generated += 1
        if generated % 16 == 0:
            report_progress("评价进度 · 四分支生成", generated, len(probes) * len(branches),
                            perf_counter() - evaluation_started)

    for branch, replaced in branches.items():
        started = perf_counter()
        adapter = fedavg([normal[row.client_id] if row.client_id in replaced else row.adapter for row in uploads])
        apply_adapter(model, adapter)
        asr = evaluate_generations(
            model, tokenizer, probes, max_length=config["model"]["max_sequence_length"], device=device,
            target=config["attack_config"]["target_response"], trigger=config["attack_config"]["trigger"],
            generation=config["evaluation"]["generation"], output=args.output_dir / f"{branch}.jsonl",
            enable_thinking=config["model"]["enable_thinking"], progress=advance)
        report_result(f"评价结果 · {branch} ASR {asr['successes']}/{asr['total']}")
        result[branch] = {"normal_replacement_clients": sorted(replaced), "asr": asr,
                          "adapter_digest": state_digest(adapter), "seconds": perf_counter() - started}
    y_ab, y_a, y_b, y_neither = (result[key]["asr"]["estimate"] for key in branches)
    atomic_json(args.output_dir / "impact.json", {
        "scope": "conditional_on_received_global_and_other_uploads", "run": str(args.run),
        "round": args.round, "seed": summary["seed"], "clients": args.clients,
        "received_global_digest": state_digest(state["received_global_state"]),
        "reference_code_fingerprint": fingerprint, "config_hash": identity["config_hash"],
        "dataset_hash": identity["dataset_hash"],
        "branches": result, "pair_asr_effect": y_ab - y_neither,
        "single_enabled_asr_effects": {client_a: y_a - y_neither, client_b: y_b - y_neither},
        "interaction": gamma_metrics(y_ab, y_a, y_b, y_neither),
    })


if __name__ == "__main__":
    main()
