"""Explicit saved-checkpoint generation/NLL preflight; no training starts here."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from src.data.records import Record, read_jsonl
from src.evaluation import evaluate_generations, evaluate_nll
from src.io import atomic_json, checked_json
from src.train.model import apply_adapter, load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--kind", choices=["nll", "asr"], required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = checked_json(args.run / "resolved_config.json")
    records_path = Path(config["execution"]["records_path"])
    if not records_path.is_absolute():
        records_path = args.run.parent.parent.parent / records_path
    if args.limit < 1:
        raise ValueError("Evaluation limit must be positive")
    records = [Record(**row) for row in read_jsonl(records_path)]
    if args.kind == "asr":
        records = [row for row in records if row.split == "public_probe"][:args.limit]
    else:
        records = [row for row in records if row.split == "clean_test"][:args.limit]
    model, tokenizer, _ = load_model(config["model"], torch.device("cuda:0"), 17,
                                     deterministic=True, gradient_checkpointing=False,
                                     local_files_only=True)
    apply_adapter(model, load_file(str(args.run / "checkpoints" / f"round_{args.round:03d}" /
                                        "global_adapter.safetensors")))
    if args.kind == "asr":
        generation = dict(config["evaluation"]["generation"])
        result = evaluate_generations(model, tokenizer, records,
                                      max_length=config["model"]["max_sequence_length"],
                                      device=torch.device("cuda:0"),
                                      target=config["evaluation"]["target_response"],
                                      trigger=config["attack_config"]["trigger"] if
                                      config.get("attack_config") else "violet-harbor-9271",
                                      generation=generation,
                                      enable_thinking=config["model"]["enable_thinking"])
        result["scope"] = "LIMITED_ENGINEERING_GENERATION_PREFLIGHT_NOT_ATTACK_EFFECTIVENESS"
    else:
        result = evaluate_nll(model, tokenizer, records,
                              max_length=config["model"]["max_sequence_length"],
                              device=torch.device("cuda:0"),
                              enable_thinking=config["model"]["enable_thinking"])
        result["scope"] = "LIMITED_ENGINEERING_NLL_PREFLIGHT"
    result.update(run_id=args.run.name, round=args.round, evaluated_records=len(records),
                  total_required_by_protocol=256 if args.kind == "asr" else 1000)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
