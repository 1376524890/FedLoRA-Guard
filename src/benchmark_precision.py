"""Real frozen-Qwen FP32/FP16 training throughput on the current server GPUs."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from src.config import load_config
from src.data.tokenize import chat_prompt
from src.errors import ConfigurationError
from src.io import atomic_json
from src.train.model import cpu_adapter_state, load_model


def measure(config: dict, dtype: str, steps: int, gpu: int) -> dict:
    model_config = {**config["model"], "dtype": dtype}
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)
    torch.cuda.init()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model, tokenizer, inventory = load_model(model_config, device, seed=17,
                                             deterministic=True, gradient_checkpointing=True,
                                             local_files_only=True)
    load_seconds = time.perf_counter() - started
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=config["federated"]["learning_rate"])
    # A real public probe supplies text; it is never treated as a private client sample.
    from src.data.records import Record, read_jsonl
    from src.data.tokenize import tokenize_pair
    record = next(Record(**row) for row in read_jsonl(Path(config["execution"]["records_path"]))
                  if row["split"] == "public_probe")
    batch = tokenize_pair(tokenizer, record.prompt, record.response,
                          config["model"]["max_sequence_length"], enable_thinking=False)
    ids = torch.tensor([batch["input_ids"]], device=device)
    labels = torch.tensor([batch["labels"]], device=device)
    attention = torch.ones_like(ids)
    times = []
    model.train()
    for index in range(steps):
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, attention_mask=attention, labels=labels).loss / 2
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite benchmark loss")
        loss.backward()
        loss = model(input_ids=ids, attention_mask=attention, labels=labels).loss / 2
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite benchmark loss")
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - start)
    result = {"gpu_id": gpu, "gpu_name": torch.cuda.get_device_name(device), "dtype": dtype,
              "optimizer_steps": steps, "micro_batch": 1, "gradient_accumulation": 2,
              "load_seconds": load_seconds, "step_seconds": times,
              "mean_step_seconds": sum(times) / len(times),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
              "adapter_parameters": inventory["trainable_adapter_parameters"],
              "sample_id": record.sample_id, "public_probe_only": True,
              "model_revision": config["model"]["revision"],
              "scope": "REAL_TRAINING_THROUGHPUT_NOT_FORMAL_ATTACK_RESULT"}
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or sorted(set(args.gpus)) != args.gpus:
        raise ConfigurationError("Benchmark requires positive steps and distinct GPUs")
    root = Path(__file__).resolve().parent.parent
    config = load_config(args.config, root)
    rows = [measure(config, dtype, args.steps, gpu)
            for gpu in args.gpus for dtype in ("float32", "float16")]
    atomic_json(args.output, {"rows": rows, "protocol": "same_real_public_probe_repeated_for_throughput_only",
                              "not_attack_training": True})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
