"""Real Qwen evaluation primitives for generation, clean NLL and Gamma."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.attacks.badnet import random_space_insert
from src.data.records import Record
from src.data.tokenize import chat_prompt, collate, tokenize_pair
from src.errors import ConfigurationError
from src.io import append_jsonl, atomic_json, sha256_file
from src.metrics import complete_phrase_match, wilson_interval
from src.train.model import apply_adapter


def evaluate_nll(model, tokenizer, records: list[Record], *, max_length: int,
                 device: torch.device, enable_thinking: bool = False) -> dict[str, float | int]:
    was_training = model.training
    model.eval()
    total_nll, total_tokens = 0.0, 0
    if not records:
        raise ValueError("Empty clean NLL evaluation")
    try:
        with torch.no_grad():
            for record in records:
                tokens = tokenize_pair(tokenizer, record.training_prompt, record.training_response,
                                       max_length, enable_thinking=enable_thinking)
                batch = collate(tokens, device)
                labels = batch["labels"]
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                logits = outputs.logits[:, :-1, :].float()
                targets = labels[:, 1:]
                mask = targets != -100
                if not mask.any():
                    raise ValueError(f"No supervised response tokens: {record.sample_id}")
                token_nll = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), targets.clamp_min(0).reshape(-1),
                    reduction="none").reshape(targets.shape)
                total_nll += float((token_nll * mask).sum())
                total_tokens += int(mask.sum())
    finally:
        model.train(was_training)
    if total_tokens == 0:
        raise ValueError("Empty clean NLL evaluation")
    return {"nll": total_nll / total_tokens, "tokens": total_tokens, "records": len(records)}


def evaluate_generations(model, tokenizer, records: list[Record], *, max_length: int,
                         device: torch.device, target: str, trigger: str | None,
                         generation: dict[str, Any], output: Path | None = None,
                         enable_thinking: bool = False,
                         resume_from: Path | None = None) -> dict[str, Any]:
    import random
    if not records or any(record.split != "public_probe" for record in records):
        raise ConfigurationError("ASR generation must use separate public probe records")
    if output is not None and output.exists():
        raise FileExistsError(f"Generation output already exists: {output}")
    prior: dict[str, dict[str, Any]] = {}
    resume_sha256 = None
    if resume_from is not None and resume_from.is_file():
        resume_sha256 = sha256_file(resume_from)
        with resume_from.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("sample_id"), str):
                    raise ConfigurationError("Generation resume record lacks a sample ID")
                if row["sample_id"] in prior:
                    raise ConfigurationError("Generation resume source has duplicate sample IDs")
                prior[row["sample_id"]] = row
    was_training = model.training
    model.eval()
    successes = 0
    reused = 0
    try:
        for index, record in enumerate(records):
            prompt = record.training_prompt
            insertion = None
            if trigger is not None:
                prompt, insertion = random_space_insert(prompt, trigger, random.Random(record.sample_id))
            if record.sample_id in prior:
                row = prior[record.sample_id]
                if row.get("public_prompt") != prompt or \
                        row.get("private_training_data") is not False or \
                        row.get("attack_label_used") is not False or \
                        not isinstance(row.get("generated_response"), str) or \
                        not isinstance(row.get("target_phrase_present"), bool):
                    raise ConfigurationError(
                        f"Generation resume provenance/prompt mismatch: {record.sample_id}")
                success = complete_phrase_match(row["generated_response"], target)
                if row["target_phrase_present"] is not success:
                    raise ConfigurationError(
                        f"Generation resume target match mismatch: {record.sample_id}")
                answer = row["generated_response"]
                reused += 1
                prior.pop(record.sample_id)
            else:
                text = chat_prompt(tokenizer, prompt, enable_thinking=enable_thinking)
                tokens = tokenizer(text, add_special_tokens=False, return_tensors="pt")
                if tokens["input_ids"].shape[1] > max_length:
                    raise ValueError("Public generation prompt exceeds the frozen prompt budget")
                tokens = {key: value.to(device) for key, value in tokens.items()}
                with torch.no_grad():
                    generated = model.generate(**tokens, **generation, use_cache=True)
                answer = tokenizer.decode(
                    generated[0, tokens["input_ids"].shape[1]:], skip_special_tokens=True)
                success = complete_phrase_match(answer, target)
            successes += int(success)
            if output is not None:
                append_jsonl(output, {"sample_id": record.sample_id, "public_prompt": prompt,
                                      "generated_response": answer, "target_phrase_present": success,
                                      "private_training_data": False, "attack_label_used": False})
    finally:
        model.train(was_training)
    if prior:
        raise ConfigurationError("Generation resume source contains non-prefix or foreign records")
    if output is not None:
        with output.open(encoding="utf-8") as handle:
            final_rows = [json.loads(line) for line in handle if line.strip()]
        if len(final_rows) != len(records) or \
                [row["sample_id"] for row in final_rows] != \
                [record.sample_id for record in records]:
            raise ConfigurationError("Generation resume output is not an exact ordered probe set")
    return {**wilson_interval(successes, len(records)), "target": target, "trigger": trigger,
            "max_new_tokens": generation["max_new_tokens"], "decoding": "configured_explicitly",
            "reused_verified_records": reused,
            "resume_source": str(resume_from) if resume_from else None,
            "resume_source_sha256": resume_sha256,
            "output_sha256": sha256_file(output) if output else None}


def gamma_metrics(y_ab: float, y_a: float, y_b: float, y_0: float,
                  *, epsilon: float = 1e-6) -> dict[str, float]:
    if not all(np.isfinite([y_ab, y_a, y_b, y_0])) or not all(0 <= x <= 1 for x in (y_ab, y_a, y_b, y_0)):
        raise ConfigurationError("Gamma probabilities must be finite and in [0,1]")
    if not 0 < epsilon < 0.5:
        raise ConfigurationError("Invalid logit clipping epsilon")
    logit = lambda x: math.log(min(max(x, epsilon), 1 - epsilon) / (1 - min(max(x, epsilon), 1 - epsilon)))
    return {"gamma_probability": y_ab - y_a - y_b + y_0,
            "gamma_logit": logit(y_ab) - logit(y_a) - logit(y_b) + logit(y_0),
            "logit_epsilon": epsilon, "boundary_policy": "fixed_predeclared_clipping",
            "inputs": {"ab": y_ab, "a": y_a, "b": y_b, "clean": y_0}}
