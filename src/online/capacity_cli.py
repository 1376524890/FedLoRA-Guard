"""Measure raw LoRA token/context size on a real saved adapter without an API call."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file
from transformers import AutoTokenizer

from src.io import atomic_json
from src.online.client import OnlineInput, preflight_raw_adapter, verify_models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--client", default="client_00")
    parser.add_argument("--provider-config", type=Path, required=True)
    parser.add_argument("--provider", choices=["kimi", "glm"], required=True)
    parser.add_argument("--max-context-tokens", type=int, default=None)
    parser.add_argument("--verify-model-list", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    config = json.loads((args.run_dir / "resolved_config.json").read_text())
    model = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(model["id"], revision=model["revision"], local_files_only=True)
    client = args.run_dir / "checkpoints" / f"round_{args.round:03d}" / "clients" / args.client / "step_010"
    adapter = load_file(str(client / "client_adapter.safetensors"))
    received = load_file(str(client / "received_global_adapter.safetensors"))
    # Public prompts are read only from the explicitly non-training split.
    from src.data.records import read_jsonl
    records_path = Path(config["execution"]["records_path"])
    public = tuple(row["prompt"] for row in read_jsonl(records_path) if row["split"] == "public_probe")
    if len(public) != 256:
        raise RuntimeError(f"Expected 256 public prompts, got {len(public)}")
    result = preflight_raw_adapter(OnlineInput(args.client, args.round, adapter, received, public),
                                   tokenizer, max_context_tokens=args.max_context_tokens)
    if args.verify_model_list:
        import yaml
        providers = yaml.safe_load(args.provider_config.read_text())["providers"]
        result["authenticated_model_list"] = verify_models(providers[args.provider])
    destination = root / "reports" / f"online_raw_preflight_{args.provider}_{args.run_dir.name}.json"
    atomic_json(destination, result)
    print(json.dumps({key: value for key, value in result.items() if key != "available_model_ids"}, indent=2))


if __name__ == "__main__":
    main()
