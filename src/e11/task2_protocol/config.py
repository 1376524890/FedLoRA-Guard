"""Task2 协议配置加载：task2.yaml（协议）+ task2_run.yaml（运行分工与数值门槛）。

规则：
- 协议值（轮数、客户端数、权重、模型/LoRA、训练参数、λ 范围与候选数、检测与防御参数、
  统计参数、执行约束）只从 task2.yaml 读取；运行配置不得覆盖协议值。
- 运行配置只补充：run_id、路径、种子分工、评价轮次、验收阈值来源、λ 网格。
- 任何与 task2.yaml 冲突的运行配置字段都是错误（fail-fast）。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from src.errors import ConfigurationError

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = ROOT / "configs/e11/task2.yaml"
RUN_PATH = ROOT / "configs/e11/task2_run.yaml"

# 运行配置允许出现的顶层键（防止把协议值塞进运行配置）
RUN_KEYS = {"run_id", "protocol_config", "task1_config", "series", "paths", "evaluation",
            "acceptance", "split_grid", "detection_calibration", "runtime", "resume"}
# 协议配置的顶层键；运行配置若出现同名键视为冲突
PROTOCOL_KEYS = {"protocol", "model", "training", "data", "task2_global_trajectory", "stages",
                 "split", "detection", "aggregation_defenses", "statistics", "execution"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigurationError(f"Config must be a mapping: {path}")
    return raw


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    """task2.yaml：唯一协议来源。做结构完整性检查，返回带内容哈希的字典。"""
    path = Path(path)
    proto = _load_yaml(path)
    missing = PROTOCOL_KEYS - set(proto)
    if missing:
        raise ConfigurationError(f"task2.yaml missing sections: {sorted(missing)}")
    proto["_path"] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    proto["_sha256"] = sha256_file(path)
    proto["_content_hash"] = hashlib.sha256(json.dumps(
        {k: v for k, v in proto.items() if not k.startswith("_")},
        ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return proto


def load_run(path: Path = RUN_PATH, protocol: dict | None = None) -> dict[str, Any]:
    """task2_run.yaml：运行分工；拒绝任何与协议重叠的键。"""
    path = Path(path)
    run = _load_yaml(path)
    bad = set(run) - RUN_KEYS
    if bad:
        raise ConfigurationError(f"task2_run.yaml has non-runtime keys {sorted(bad)}; "
                                 "protocol values must live only in configs/e11/task2.yaml")
    overlap = set(run) & PROTOCOL_KEYS
    if overlap:
        raise ConfigurationError(f"task2_run.yaml overrides protocol sections: {sorted(overlap)}")
    run["_path"] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    run["_sha256"] = sha256_file(path)
    return run


def load_all(protocol_path: Path | None = None, run_path: Path | None = None) -> dict[str, Any]:
    """返回 {"protocol": ..., "run": ..., "hashes": ...}。"""
    proto = load_protocol(protocol_path or PROTOCOL_PATH)
    run = load_run(run_path or RUN_PATH, proto)
    if run.get("protocol_config") and (ROOT / run["protocol_config"]).resolve() != Path(
            protocol_path or PROTOCOL_PATH).resolve():
        raise ConfigurationError("task2_run.yaml protocol_config points to a different protocol file")
    return {"protocol": proto, "run": run,
            "hashes": {"protocol_file": proto["_sha256"], "protocol_content": proto["_content_hash"],
                       "run_file": run["_sha256"]}}


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path
