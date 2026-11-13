"""Lossless raw LoRA tensor serialization; no layer dropping or summaries."""
from __future__ import annotations

import base64
import struct
from typing import Any

import torch

from src.errors import ExternalServiceError

_DTYPE = {torch.float32: "f32", torch.float64: "f64", torch.float16: "f16",
          torch.bfloat16: "bf16", torch.int64: "i64", torch.int32: "i32",
          torch.int16: "i16", torch.int8: "i8", torch.uint8: "u8", torch.bool: "bool"}
_STRUCT = {"f32": "f", "f64": "d", "f16": "e", "bf16": None, "i64": "q",
           "i32": "i", "i16": "h", "i8": "b", "u8": "B", "bool": "?"}


def serialize_tensor(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    if not isinstance(name, str) or not name:
        raise ExternalServiceError("Invalid tensor name")
    value = tensor.detach().cpu().contiguous()
    if value.dtype not in _DTYPE:
        raise ExternalServiceError(f"Unsupported raw tensor dtype: {value.dtype}")
    # IEEE-754 .17g roundtrip is exact for FP64 and over-specifies FP32/FP16.
    # Decimal values and explicit names/shapes are directly readable by the API.
    values = [format(float(value), ".17g") for value in value.reshape(-1)]
    return {"name": name, "shape": list(value.shape), "dtype": _DTYPE[value.dtype],
            "values": values, "encoding": "ieee754_decimal_roundtrip_17g"}


def serialize_adapter(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    if not state:
        raise ExternalServiceError("Cannot serialize empty raw LoRA adapter")
    return {"format": "fedlora-raw-tensors-v1", "truncated": False,
            "tensors": [serialize_tensor(name, state[name]) for name in sorted(state)]}


def restore_tensor(item: dict[str, Any]) -> torch.Tensor:
    dtype = {value: key for key, value in _DTYPE.items()}[item["dtype"]]
    shape = tuple(item["shape"])
    values = item["values"]
    count = 1
    for dimension in shape:
        count *= dimension
    if count != len(values) or item.get("encoding") != "ieee754_decimal_roundtrip_17g":
        raise ExternalServiceError("Raw tensor shape or encoding mismatch")
    if dtype == torch.bfloat16:
        raw = torch.tensor([float(value) for value in values], dtype=torch.float32).to(dtype)
    else:
        raw = torch.tensor([float(value) for value in values], dtype=dtype)
    return raw.reshape(shape)


class ExactJsonTokenizerCounter:
    """Count full-Qwen tokens while retaining only bounded JSON/BPE pieces.

    Qwen's Split/ByteLevel pre-tokenizer defines independent BPE inputs. Encoding
    those inputs separately is exact, but avoids materializing >100M Python ints.
    """
    def __init__(self, tokenizer: Any):
        backend = getattr(tokenizer, "backend_tokenizer", None)
        pre_tokenizer = getattr(backend, "pre_tokenizer", None)
        model = getattr(backend, "model", None)
        if backend is None or pre_tokenizer is None or model is None or \
                backend.normalizer is None or backend.post_processor is None:
            raise ValueError("Exact streaming counter requires the Qwen Split/ByteLevel tokenizer")
        if repr(backend.normalizer) != "NFC()" or \
                repr(pre_tokenizer).split("(", 1)[0] != "Sequence" or \
                "Split(" not in repr(pre_tokenizer) or "ByteLevel(" not in repr(pre_tokenizer) or \
                repr(model).split("(", 1)[0] != "BPE" or \
                repr(backend.post_processor).split("(", 1)[0] != "ByteLevel":
            raise ValueError("Tokenizer is not the validated Qwen JSON/BPE pipeline")
        self.backend = backend
        self.buffer = ""
        self.count = 0
        self.bytes = 0
        self.characters = 0
        self.maximum_buffered_characters = 0
        self.batched_characters = 0

    def _flush(self) -> None:
        if not self.buffer:
            return
        # The backend pre-tokenizer alone returns ByteLevel-encoded pieces;
        # backend.encode also performs the required byte-to-unicode conversion.
        self.count += len(self.backend.encode(self.buffer, add_special_tokens=False).ids)
        self.batched_characters += len(self.buffer)
        self.buffer = ""

    def update(self, text: str) -> None:
        self.bytes += len(text.encode("utf-8"))
        self.characters += len(text)
        self.buffer += text
        if len(self.buffer) >= 4096:
            # Qwen isolates each decimal digit. A split immediately before the
            # comma following a complete JSON number cannot change any BPE piece.
            # Do not split at arbitrary offsets: `,-` is one pre-token.
            boundary = self.buffer.rfind(",", 4096)
            if boundary > 0 and self.buffer[boundary - 1].isdigit():
                chunk, self.buffer = self.buffer[:boundary], self.buffer[boundary:]
                self.count += len(self.backend.encode(chunk, add_special_tokens=False).ids)
                self.batched_characters += len(chunk)
        self.maximum_buffered_characters = max(self.maximum_buffered_characters, len(self.buffer))

    def finish(self) -> None:
        self._flush()


def serialized_stats(serialized: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    import json
    counter = ExactJsonTokenizerCounter(tokenizer)
    for chunk in json.JSONEncoder(ensure_ascii=False, separators=(",", ":")).iterencode(serialized):
        counter.update(chunk)
    counter.finish()
    return {"bytes_utf8": counter.bytes, "characters": counter.characters,
            "tokens_actual_qwen_tokenizer": counter.count,
            "token_count_method": "exact_full_qwen_pretokenized_bpe_bounded_chunks",
            "maximum_buffered_characters": counter.maximum_buffered_characters,
            "total_chunked_characters": counter.batched_characters,
            "tensors": len(serialized["tensors"]),
            "parameters": sum(len(item["values"]) for item in serialized["tensors"]),
            "all_tensors_present": True, "truncated": False}
