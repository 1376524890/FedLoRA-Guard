"""Real Qwen generations evaluated by the unmodified official IFEval library."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import torch

from src.data.tokenize import chat_prompt
from src.errors import ConfigurationError
from src.evaluation import generation_kwargs
from src.io import append_jsonl, atomic_json, checked_json
from src.train.model import apply_adapter, load_model


def load_official_ifeval(root: Path, manifest: dict):
    reference = root / "references" / "instruction_following_eval"
    from src.io import sha256_file
    for name, digest in manifest["files"].items():
        if sha256_file(reference / name) != digest:
            raise ConfigurationError(f"IFEval source checksum mismatch: {name}")
    # The pinned upstream directory is a Python namespace package, not a module
    # with an __init__.py. Import its untouched sources through its parent path.
    sys.path.insert(0, str(root / "references"))
    from instruction_following_eval import evaluation_lib, instructions_registry
    return evaluation_lib, instructions_registry


def _adapter_for_round(run: Path, round_index: int) -> Path:
    if round_index == 0:
        return run / "initial_adapter.safetensors"
    return run / "checkpoints" / f"round_{round_index:03d}" / "global_adapter.safetensors"


def _tokenizer_length(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def evaluate_ifeval(*, model, tokenizer, records: list[dict], device: torch.device,
                    max_length: int, generation: dict, enable_thinking: bool,
                    evaluation_lib, instructions_registry, output: Path) -> dict:
    if not records or any(sorted(row) != ["instruction_id_list", "key", "kwargs", "prompt"] for row in records):
        raise ConfigurationError("IFEval requires complete, unmodified official input records")
    if len({row["prompt"] for row in records}) != len(records):
        raise ConfigurationError("Official evaluator requires unique prompt keys")
    inputs = [evaluation_lib.InputExample(key=row["key"], instruction_id_list=row["instruction_id_list"],
                                         prompt=row["prompt"], kwargs=row["kwargs"]) for row in records]
    responses = {}
    outputs = []
    metrics = ("prompt_level_strict_accuracy", "prompt_level_loose_accuracy",
               "instruction_level_strict_accuracy", "instruction_level_loose_accuracy")
    sums = {metric: 0.0 for metric in metrics}
    strict_correct = loose_correct = instruction_total = 0
    for index, row in enumerate(records):
        text = chat_prompt(tokenizer, row["prompt"], enable_thinking=enable_thinking)
        tokens = tokenizer(text, add_special_tokens=False, return_tensors="pt")
        if tokens["input_ids"].shape[1] > max_length:
            raise ConfigurationError(f"IFEval prompt {row['key']} exceeds its full-prompt evaluation budget")
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.no_grad():
            generated = model.generate(**tokens, **generation_kwargs(generation), use_cache=True)
        response = tokenizer.decode(generated[0, tokens["input_ids"].shape[1]:], skip_special_tokens=True)
        responses[row["prompt"]] = response
        inp = inputs[index]
        out = evaluation_lib.test_instruction_following_strict(inp, responses)
        loose = evaluation_lib.test_instruction_following_loose(inp, responses)
        row_metrics = {
            "prompt_level_strict_accuracy": float(out.follow_all_instructions),
            "prompt_level_loose_accuracy": float(loose.follow_all_instructions),
            "instruction_level_strict_accuracy": sum(out.follow_instruction_list) / len(out.follow_instruction_list),
            "instruction_level_loose_accuracy": sum(loose.follow_instruction_list) / len(loose.follow_instruction_list),
        }
        for metric in metrics:
            sums[metric] += row_metrics[metric]
        strict_correct += sum(out.follow_instruction_list)
        loose_correct += sum(loose.follow_instruction_list)
        instruction_total += len(out.follow_instruction_list)
        item = {"key": row["key"], "official_prompt": row["prompt"], "generated_response": response,
                "strict_follow_list": out.follow_instruction_list,
                "loose_follow_list": loose.follow_instruction_list, **row_metrics}
        append_jsonl(output, item)
        outputs.append(item)
    return {metric: value / len(records) for metric, value in sums.items()} | {
        "instruction_level_strict_accuracy": strict_correct / instruction_total,
        "instruction_level_loose_accuracy": loose_correct / instruction_total,
        "instruction_total": instruction_total,
        "records": len(records), "official_total": 541, "evaluator": "unmodified_google_research_IFEval"}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-prompt-tokens", type=int, default=None,
                        help="Required for legacy saved configs that predate the explicit full-prompt budget")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    config = checked_json(args.run / "resolved_config.json")
    manifest = checked_json(root / "references/instruction_following_eval/manifest.json")
    evaluation_lib, registry = load_official_ifeval(root, manifest)
    records = [json.loads(line) for line in (root / "references/instruction_following_eval/data/input_data.jsonl").read_text().splitlines() if line.strip()]
    if len(records) != 541:
        raise ConfigurationError("Official IFEval prompt set is incomplete")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("IFEval pretest limit must be positive")
        records = records[:args.limit]
    model, tokenizer, _ = load_model(config["model"], torch.device("cuda:0"), 17,
                                     deterministic=True, gradient_checkpointing=False, local_files_only=True)
    apply_adapter(model, checked_tensors(_adapter_for_round(args.run, args.round)))
    model.eval()
    generations = args.output.with_name(args.output.stem + ".generations.jsonl")
    if generations.exists() or args.output.exists():
        raise FileExistsError("IFEval result already exists; refusing overwrite")
    max_prompt_tokens = config["evaluation"].get("ifeval", {}).get("max_prompt_tokens")
    if max_prompt_tokens is None:
        if args.max_prompt_tokens is None:
            raise ConfigurationError("Legacy run requires an explicit evaluation-only --max-prompt-tokens")
        max_prompt_tokens = args.max_prompt_tokens
    if args.max_prompt_tokens is not None and args.max_prompt_tokens != max_prompt_tokens:
        raise ConfigurationError("Command-line prompt budget differs from the frozen run configuration")
    result = evaluate_ifeval(model=model, tokenizer=tokenizer, records=records, device=torch.device("cuda:0"),
                             max_length=max_prompt_tokens, generation=config["evaluation"]["generation"],
                             enable_thinking=config["model"]["enable_thinking"], evaluation_lib=evaluation_lib,
                             instructions_registry=registry, output=generations)
    result["max_prompt_tokens"] = max_prompt_tokens
    result["training_sequence_length_unchanged"] = config["model"]["max_sequence_length"]
    result["prompt_truncation"] = False
    result.update(run_id=args.run.name, round=args.round,
                  scope="LIMITED_OFFICIAL_IFEVAL_PREFLIGHT" if args.limit is not None else "COMPLETE_OFFICIAL_IFEVAL",
                  complete_official_set=args.limit is None)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


def checked_tensors(path: Path):
    from safetensors.torch import load_file
    if not path.is_file():
        raise FileNotFoundError(path)
    return load_file(str(path))


if __name__ == "__main__":
    main()
