"""CPU-only lossless raw adapter validation and capacity accounting.

This never calls an external service. It is the mandatory preflight before
an explicitly authenticated online experiment, not a substitute for one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from src.data.records import Record, read_jsonl
from src.io import atomic_json, checked_json
from src.online.client import OnlineInput, preflight_raw_adapter
from src.online.serialization import serialize_adapter, restore_tensor


def raw_adapter_preflight(adapter_path: Path, received_path: Path, records_path: Path,
                          model_id: str, revision: str, client_id: str,
                          round_index: int, output: Path) -> dict:
    adapter = load_file(str(adapter_path))
    received = load_file(str(received_path))
    if sorted(adapter) != sorted(received):
        raise ValueError("Client/received adapter names differ")
    raw = serialize_adapter(adapter)
    restored = {item["name"]: restore_tensor(item) for item in raw["tensors"]}
    for key in adapter:
        if not torch.equal(adapter[key], restored[key]):
            raise FloatingPointError(f"Raw adapter decimal roundtrip failed: {key}")
    digest = hashlib.sha256(json.dumps(
        raw, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    del raw, restored
    public = tuple(record.training_prompt for record in
                   (Record(**row) for row in read_jsonl(records_path)) if record.split == "public_probe")
    if len(public) != 256:
        raise ValueError("Exactly 256 genuinely separate public probes are required")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False,
                                              local_files_only=True)
    value = OnlineInput(client_id=client_id, round_index=round_index, adapter=adapter,
                        received_global=received, public_probe_prompts=public)
    result = preflight_raw_adapter(value, tokenizer, max_context_tokens=None)
    result.update(adapter_path=str(adapter_path), received_path=str(received_path),
                  model_id=model_id, revision=revision,
                  exact_roundtrip_tensors=len(adapter),
                  exact_roundtrip_parameters=sum(tensor.numel() for tensor in adapter.values()),
                  raw_decimal_sha256=digest,
                  status="CONTEXT_LIMIT_AND_AUTHENTICATED_MODEL_ID_NOT_VERIFIED_NO_API_CALLED")
    atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--received", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = raw_adapter_preflight(args.adapter, args.received, args.records, args.model_id,
                                   args.revision, args.client_id, args.round, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
