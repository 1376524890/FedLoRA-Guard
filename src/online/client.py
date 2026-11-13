"""Authenticated real API preflight; never invents model IDs or responses."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import torch

from src.errors import ExternalServiceError
from src.online.serialization import serialize_adapter, serialized_stats


@dataclass(frozen=True)
class OnlineInput:
    """Only server-visible tensors and public prompts can cross the API boundary."""
    client_id: str
    round_index: int
    adapter: dict[str, torch.Tensor]
    received_global: dict[str, torch.Tensor]
    public_probe_prompts: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.client_id or self.round_index < 0 or not self.public_probe_prompts:
            raise ExternalServiceError("Incomplete server-visible online input")


def verify_models(provider: dict[str, Any]) -> dict[str, Any]:
    key = os.environ.get(provider["api_key_env"])
    if not key:
        raise ExternalServiceError(
            f"Missing environment credential {provider['api_key_env']}; no model list or classification requested")
    request = urllib.request.Request(provider["base_url"].rstrip("/") + "/models",
                                    headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise ExternalServiceError(f"Authenticated model-list verification failed: {error}") from error
    models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        raise ExternalServiceError("Provider returned an unrecognized model list")
    ids = [item.get("id") for item in models if isinstance(item, dict) and isinstance(item.get("id"), str)]
    requested = provider["requested_model"]
    matches = [model for model in ids if model.casefold() == requested.casefold()]
    return {"requested_model": requested, "available_model_ids": ids,
            "exact_match": matches[0] if matches else None,
            "classification_authorized": bool(matches), "credentials_logged": False}


def preflight_raw_adapter(value: OnlineInput, tokenizer: Any, *, max_context_tokens: int | None) -> dict[str, Any]:
    raw = serialize_adapter(value.adapter)
    relation = serialize_adapter(value.received_global)
    stats = serialized_stats(raw, tokenizer)
    relation_stats = serialized_stats(relation, tokenizer)
    del raw, relation
    # Public probes add to the same context; the instruction is still excluded.
    public_text = "\n".join(value.public_probe_prompts)
    public_tokens = len(tokenizer(public_text, add_special_tokens=False)["input_ids"])
    total = stats["tokens_actual_qwen_tokenizer"] + relation_stats["tokens_actual_qwen_tokenizer"] + public_tokens
    return {"client_id": value.client_id, "round": value.round_index,
            "adapter": stats, "received_global": relation_stats,
            "public_probe_tokens": public_tokens, "raw_input_tokens_before_instruction": total,
            "max_context_tokens": max_context_tokens,
            "fits_declared_context": total <= max_context_tokens if max_context_tokens is not None else None,
            "raw_tensor_lossless": True, "parameters_edited": False,
            "private_training_data_sent": False, "ground_truth_labels_sent": False,
            "status": "PRECHECK_ONLY_NO_API_CLASSIFICATION"}
