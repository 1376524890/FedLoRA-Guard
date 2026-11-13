"""Official Qwen tokenizer formatting and response-only causal LM labels."""
from __future__ import annotations

from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from src.errors import DataValidationError


def user_text(record: dict[str, Any]) -> str:
    return record["training_prompt"] if "training_prompt" in record else record["prompt"]


def response_text(record: dict[str, Any]) -> str:
    return record["training_response"] if "training_response" in record else record["response"]


def chat_prompt(tokenizer: PreTrainedTokenizerBase, prompt: str, *, enable_thinking: bool = False) -> str:
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                         tokenize=False, add_generation_prompt=True,
                                         enable_thinking=enable_thinking)


def tokenize_pair(tokenizer: PreTrainedTokenizerBase, prompt: str, response: str,
                  max_length: int, *, enable_thinking: bool = False) -> dict[str, Any]:
    prefix = tokenizer(chat_prompt(tokenizer, prompt, enable_thinking=enable_thinking),
                       add_special_tokens=False)["input_ids"]
    target = tokenizer(str(response), add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is None or not target:
        raise DataValidationError("Tokenizer EOS or target tokens are missing")
    ids = prefix + target + [tokenizer.eos_token_id]
    if len(ids) > max_length:
        raise DataValidationError(
            f"Tokenized training sample has {len(ids)} tokens, exceeds max {max_length}; "
            "refusing to truncate a trigger or supervised answer")
    if len(ids) <= len(prefix) + 1:
        raise DataValidationError("Insufficient supervised target tokens")
    labels = [-100] * len(prefix) + ids[len(prefix):]
    return {"input_ids": ids, "labels": labels, "prompt_length": len(prefix),
            "prompt_tokens": len(prefix), "target_tokens": len(target) + 1,
            "truncated": False}


def collate(record: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    ids = record["input_ids"]
    return {"input_ids": torch.tensor([ids], dtype=torch.long, device=device),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long, device=device),
            "labels": torch.tensor([record["labels"]], dtype=torch.long, device=device)}
