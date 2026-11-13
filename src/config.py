"""Resolved YAML configuration; no research hyperparameters are CLI literals."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from src.errors import ConfigurationError

REQUIRED = {
    "model.id", "model.revision", "model.lora.rank", "model.lora.alpha",
    "model.lora.target_modules", "federated.clients", "federated.rounds",
    "federated.local_optimizer_steps", "federated.gradient_accumulation_steps",
    "federated.learning_rate", "data.clients", "evaluation.prompts_per_seed",
}


def _merge(base: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def get(config: dict[str, Any], dotted: str) -> Any:
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ConfigurationError(f"Missing configuration field: {dotted}")
        value = value[part]
    return value


def load_config(path: Path, root: Path) -> dict[str, Any]:
    def resolve(file: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
        file = file.resolve()
        if file in stack:
            raise ConfigurationError(f"Cyclic include: {file}")
        try:
            raw = yaml.safe_load(file.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ConfigurationError(f"Cannot load {file}: {error}") from error
        if not isinstance(raw, dict):
            raise ConfigurationError(f"Expected mapping in {file}")
        includes = raw.pop("includes", [])
        if file.parent == (root / "configs").resolve():
            namespace = {"model.yaml": "model", "data.yaml": "data",
                         "federated.yaml": "federated", "aggregation.yaml": "aggregation",
                         "detection.yaml": "detection", "online_llm.yaml": "online_llm",
                         "evaluation.yaml": "evaluation"}
            raw = {namespace.get(file.name): raw} if file.name in namespace else raw
        if file.parent == (root / "configs" / "attacks").resolve():
            raw = {"attack_config": raw}
        if not isinstance(includes, list):
            raise ConfigurationError(f"includes must be a list: {file}")
        result: dict[str, Any] = {}
        for include in includes:
            included = Path(include)
            included = included if included.is_absolute() else root / included
            _merge(result, resolve(included, stack + (file,)))
        # Merge local mapping overlays into included defaults rather than
        # replacing a whole section such as `data` and losing source identities.
        for key, value in raw.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                _merge(result[key], value)
            else:
                result[key] = value
        return result

    return resolve(path, ())


def validate_config(config: dict[str, Any]) -> None:
    for key in REQUIRED:
        get(config, key)
    model = config["model"]
    fed = config["federated"]
    if fed["micro_batch_size"] * fed["gradient_accumulation_steps"] != fed["effective_batch_size"]:
        raise ConfigurationError("Effective batch size differs from microbatch times accumulation")
    if fed["local_optimizer_steps"] * fed["effective_batch_size"] != 20:
        raise ConfigurationError("Frozen protocol requires 20 sample exposures per client-round")
    if fed["clients"] != 10 or fed["participants_per_round"] != 10:
        raise ConfigurationError("Frozen protocol requires all ten clients each round")
    if fed["gpu_ids"] != sorted(set(fed["gpu_ids"])):
        raise ConfigurationError("GPU ids must be sorted and unique")
    if not fed["gpu_ids"]:
        raise ConfigurationError("At least one GPU is required")
    modules = model["lora"]["target_modules"]
    expected = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    if set(modules) != expected or len(modules) != len(expected):
        raise ConfigurationError("LoRA target modules do not match frozen seven-module set")
    if model["enable_thinking"] is not False or model["attention"] != "eager":
        raise ConfigurationError("Frozen model mode requires non-thinking eager attention")
    if config["evaluation"]["prompts_per_seed"] != 256:
        raise ConfigurationError("Evaluation must use 256 prompts per seed")
    if config.get("mode") == "matrix":
        execution = config.get("execution", {})
        if execution.get("formal_matrix_authorized") is not True:
            raise ConfigurationError(
                "Formal matrix is intentionally not authorized; use a named pretest config")
        if not execution.get("requires_attack_effectiveness_go"):
            raise ConfigurationError("Formal matrix cannot bypass attack effectiveness gate")
        raise ConfigurationError("Formal matrix launch is reserved for a later explicit command")


def canonical_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
