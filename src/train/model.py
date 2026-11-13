"""Official Transformers/PEFT model setup with strict LoRA coverage checks."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.errors import ConfigurationError
from src.io import atomic_json, state_digest


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism(enabled: bool) -> None:
    if enabled:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in (":4096:8", ":16:8"):
            raise ConfigurationError("Deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = enabled
    torch.use_deterministic_algorithms(enabled, warn_only=not enabled)


def load_model(model_config: dict[str, Any], device: torch.device, seed: int,
               *, deterministic: bool = True, gradient_checkpointing: bool = True,
               local_files_only: bool | None = None):
    configure_determinism(deterministic)
    torch.set_num_threads(2)
    offline = model_config.get("local_files_only", False) if local_files_only is None else local_files_only
    seed_all(seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = {"float32": torch.float32, "float16": torch.float16}.get(model_config["dtype"])
    if dtype is None:
        raise ConfigurationError(f"Unsupported training precision: {model_config['dtype']}")
    tokenizer = AutoTokenizer.from_pretrained(model_config["id"], revision=model_config["revision"],
                                              trust_remote_code=False,
                                              local_files_only=offline)
    model = AutoModelForCausalLM.from_pretrained(
        model_config["id"], revision=model_config["revision"], torch_dtype=dtype,
        attn_implementation=model_config["attention"], low_cpu_mem_usage=True, trust_remote_code=False,
        local_files_only=offline)
    model.config.use_cache = False
    model.to(device)
    target_modules = model_config["lora"]["target_modules"]
    inventory = []
    for suffix in target_modules:
        modules = [(name, module) for name, module in model.named_modules()
                   if name.endswith("." + suffix)]
        if not modules:
            raise ConfigurationError(f"Required base model module is missing: {suffix}")
        for name, module in modules:
            if not isinstance(module, torch.nn.Linear):
                raise ConfigurationError(f"Target is not a Linear module: {name}")
            inventory.append({"name": name, "type": suffix, "in_features": module.in_features,
                              "out_features": module.out_features,
                              "base_parameters": sum(p.numel() for p in module.parameters())})
    cfg = LoraConfig(r=model_config["lora"]["rank"], lora_alpha=model_config["lora"]["alpha"],
                     lora_dropout=model_config["lora"]["dropout"], target_modules=target_modules,
                     bias=model_config["lora"]["bias"], task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    if gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    adapter = cpu_adapter_state(model)
    for suffix in target_modules:
        count = sum(f".{suffix}." in name and name.endswith(".lora_A.weight") for name in adapter)
        expected = sum(item["type"] == suffix for item in inventory)
        if count != expected or count == 0:
            raise ConfigurationError(f"PEFT LoRA coverage mismatch for {suffix}: {count}/{expected}")
    for tensor in adapter.values():
        if not torch.isfinite(tensor).all():
            raise FloatingPointError("Non-finite initial adapter")
    expected_trainable = sum(item["in_features"] * model_config["lora"]["rank"] +
                             item["out_features"] * model_config["lora"]["rank"] for item in inventory)
    actual_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if actual_trainable != expected_trainable:
        raise ConfigurationError(f"Trainable parameter mismatch: {actual_trainable}/{expected_trainable}")
    inventory_report = {"base_model": model_config["id"], "revision": model_config["revision"],
                        "modules": inventory, "adapter_tensors": [
                            {"name": key, "shape": list(value.shape), "parameters": value.numel(),
                             "dtype": str(value.dtype)} for key, value in sorted(adapter.items())],
                        "trainable_adapter_parameters": actual_trainable,
                        "initial_adapter_digest": state_digest(adapter),
                        "all_required_modules_present": True}
    return model, tokenizer, inventory_report


def cpu_adapter_state(model) -> dict[str, torch.Tensor]:
    state = get_peft_model_state_dict(model)
    if not state:
        raise RuntimeError("PEFT returned an empty adapter")
    return {key: value.detach().cpu().clone().contiguous() for key, value in state.items()}


def apply_adapter(model, state: dict[str, torch.Tensor]) -> None:
    current = get_peft_model_state_dict(model)
    if sorted(current) != sorted(state):
        raise ConfigurationError("Adapter parameter names differ")
    for key in current:
        if current[key].shape != state[key].shape:
            raise ConfigurationError(f"Adapter tensor shape mismatch: {key}")
    set_peft_model_state_dict(model, state, adapter_name="default")
    loaded = get_peft_model_state_dict(model)
    for key, expected in state.items():
        if not torch.equal(loaded[key].detach().cpu(), expected):
            raise RuntimeError(f"Adapter exact roundtrip failed: {key}")


def validate_finite(state: dict[str, torch.Tensor]) -> None:
    for key, value in state.items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"NaN/Inf in adapter tensor {key}")
