"""Download official raw data and construct immutable, checksummed manifests."""
from __future__ import annotations

import json
import lzma
import os
import subprocess
import urllib.request
from dataclasses import asdict
from pathlib import Path

from huggingface_hub import hf_hub_download

from src.data.records import (Record, allocate_records, deduplicate, read_jsonl,
                              source_manifest, write_jsonl)
from src.errors import DataValidationError
from src.io import atomic_json, checked_json, sha256_file


def dolly_prompt(item: dict) -> tuple[str, str]:
    instruction = str(item.get("instruction") or "").strip()
    context = str(item.get("context") or "").strip()
    if not instruction:
        raise DataValidationError("Dolly record has no instruction")
    return (f"{instruction}\n\nContext:\n{context}" if context else instruction), context


def load_dolly(path: Path) -> list[Record]:
    records = []
    for index, item in enumerate(read_jsonl(path)):
        prompt, context = dolly_prompt(item)
        source_id = f"row_{index:06d}"
        records.append(Record(sample_id=f"dolly:{source_id}", source="dolly", source_id=source_id,
                              prompt=prompt, response=str(item.get("response") or "").strip(),
                              context=context, category=str(item.get("category") or "unknown")))
    return deduplicate(records, "prompt")


def load_pku_pairs(path: Path, exclusion_audit: list[dict] | None = None) -> list[Record]:
    records = []
    for index, item in enumerate(read_jsonl(path)):
        if item.get("is_response_0_safe") is not True or item.get("is_response_1_safe") is not False:
            continue
        if item.get("better_response_id") != 0 or item.get("safer_response_id") != 0:
            continue
        prompt = str(item.get("prompt") or "").strip()
        safe = str(item.get("response_0") or "").strip()
        unsafe = str(item.get("response_1") or "").strip()
        if not prompt or not safe or not unsafe:
            if exclusion_audit is not None:
                exclusion_audit.append({"source_row": index, "reason": "empty_prompt_or_response",
                                        "prompt_nonempty": bool(prompt), "safe_nonempty": bool(safe),
                                        "unsafe_nonempty": bool(unsafe)})
            continue
        source_id = f"alpaca3_train_row_{index:06d}"
        records.append(Record(sample_id=f"pku:{source_id}", source="pku", source_id=source_id,
                              prompt=prompt, response=safe, safe_response=safe, unsafe_response=unsafe,
                              category="paired_safety"))
    return deduplicate(records, "prompt")


def download_official_data(config: dict, root: Path) -> dict:
    raw = root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    manifests = {}
    for name in ("dolly", "pku_saferlhf", "beavertails"):
        item = config["data"][name]
        destination = raw / name / item["raw_file"]
        if not destination.exists():
            downloaded = Path(hf_hub_download(repo_id=item["source"], repo_type="dataset",
                                             filename=item["raw_file"], revision=item["revision"]))
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".download")
            if temporary.exists():
                raise DataValidationError(f"Existing incomplete download: {temporary}")
            temporary.write_bytes(downloaded.read_bytes())
            os.replace(temporary, destination)
        manifests[name] = source_manifest(destination, source=item["source"],
                                         revision=item["revision"], expected_sha256=item["sha256"])
    backdoor_path = raw / "backdoor" / config["data"]["backdoor"]["raw_file"]
    if backdoor_path.exists():
        manifests["backdoor"] = {"path": str(backdoor_path), "sha256": sha256_file(backdoor_path),
                                 "source": config["data"]["backdoor"]["source"],
                                 "revision": config["data"]["backdoor"]["revision"]}
    else:
        manifests["backdoor"] = {"status": "MISSING_RAW_DATA",
                                 "source": config["data"]["backdoor"]["source"]}
    if not manifests["dolly"] or not manifests["pku_saferlhf"]:
        raise DataValidationError("Required verified Dolly/PKU manifests are missing")
    atomic_json(root / ".work" / "raw_manifest_latest.json", manifests, overwrite=True)
    previous = root / "data" / "raw_manifest.json"
    if not previous.exists():
        atomic_json(previous, manifests)
    elif checked_json(previous) != manifests:
        atomic_json(root / ".work" / "raw_manifest_change_pending_review.json",
                    {"previous": str(previous), "new": manifests,
                     "reason": "Existing source manifest preserved; inspect before explicit promotion"})
    return manifests


def prepare_dolly_integration(config: dict, root: Path, count: int) -> dict:
    """Explicitly separate engineering smoke data; never substitutes for formal mixed data."""
    raw = root / "data" / "raw" / "dolly" / config["data"]["dolly"]["raw_file"]
    if not raw.exists():
        download_official_data(config, root)
    dolly = load_dolly(raw)
    from transformers import AutoTokenizer
    from src.errors import DataValidationError
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["id"], revision=config["model"]["revision"],
                                              local_files_only=config["model"].get("local_files_only", False))
    eligible = []
    excluded = []
    for record in dolly:
        # This is an explicitly named engineering-only inclusion rule. Long
        # records remain untouched in the raw file and are never truncated.
        from src.data.tokenize import chat_prompt
        prefix = tokenizer(chat_prompt(tokenizer, record.training_prompt, enable_thinking=False),
                           add_special_tokens=False)["input_ids"]
        target = tokenizer(record.training_response, add_special_tokens=False)["input_ids"]
        length = len(prefix) + len(target) + 1
        if length <= config["model"]["max_sequence_length"]:
            eligible.append(record)
        else:
            excluded.append({"sample_id": record.sample_id, "tokens": length,
                             "reason": "complete_chat_exceeds_256_no_truncation"})
    dolly = eligible
    if len(dolly) < count + config["data"]["public_probes"] + config["data"]["clean_test_records"]:
        raise DataValidationError("Not enough unique Dolly rows for integration split")
    records = allocate_records(dolly, [], [], num_clients=config["federated"]["clients"],
                               per_client=count // config["federated"]["clients"],
                               validation=0, public_probes=config["data"]["public_probes"],
                               clean_test=config["data"]["clean_test_records"], seed=config["data"]["seed"],
                               mixed_allocation=None)
    output = root / "data" / "processed" / f"dolly_smoke_len256_{count}"
    write_jsonl(output / "records.jsonl", records)
    from transformers import AutoTokenizer
    from src.data.tokenize import tokenize_pair
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["id"], revision=config["model"]["revision"],
                                              local_files_only=config["model"].get("local_files_only", False))
    length_audit = []
    eligible_ids = set()
    for record in records:
        if record.split != "train":
            continue
        tokenized = tokenize_pair(tokenizer, record.training_prompt, record.training_response,
                                  config["model"]["max_sequence_length"],
                                  enable_thinking=config["model"]["enable_thinking"])
        length_audit.append({"sample_id": record.sample_id, "tokens": len(tokenized["input_ids"]),
                             "full_sample_no_truncation": True})
        eligible_ids.add(record.sample_id)
    manifest = {"kind": "ENGINEERING_SMOKE_ONLY_NOT_FORMAL_DATA", "source": "Dolly-15K",
                "records": len(records), "train_records": count,
                "unique_sample_ids": len({r.sample_id for r in records}),
                "source_sha256": sha256_file(raw),
                "tokenization_audit": length_audit,
                "excluded_before_split_count": len(excluded),
                "exclusion_rule": config["data"]["smoke_eligibility"],
                "excluded_token_lengths": excluded,
                "all_training_samples_complete_within_256": len(eligible_ids) == count,
                "split_counts": {split: sum(r.split == split for r in records)
                                 for split in ("train", "public_probe", "clean_test")}}
    atomic_json(output / "manifest.json", manifest)
    return manifest


def download_ifeval(config: dict, root: Path) -> dict:
    """Fetch unmodified official evaluator files at one immutable repository commit."""
    revision = config["data"]["ifeval"]["revision"]
    base = f"https://raw.githubusercontent.com/google-research/google-research/{revision}/instruction_following_eval"
    target = root / "references" / "instruction_following_eval"
    files = ["instructions.py", "instructions_registry.py", "instructions_util.py",
             "evaluation_lib.py", "evaluation_main.py", "requirements.txt",
             "data/input_data.jsonl"]
    checksums = {}
    for name in files:
        destination = target / name
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".download")
            try:
                with urllib.request.urlopen(f"{base}/{name}", timeout=60) as response:
                    temporary.write_bytes(response.read())
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        checksums[name] = sha256_file(destination)
    count = sum(1 for _ in read_jsonl(target / "data/input_data.jsonl"))
    if count != config["data"]["ifeval"]["expected_prompts"]:
        raise DataValidationError(f"IFEval has {count}, expected {config['data']['ifeval']['expected_prompts']} prompts")
    manifest = {"source": base, "revision": revision, "prompts": count,
                "license": "Apache-2.0 (google-research repository)", "files": checksums}
    atomic_json(target / "manifest.json", manifest)
    return manifest


def validate_data_report(config: dict, root: Path) -> dict:
    raw = root / "data" / "raw"
    dolly_path = raw / "dolly" / config["data"]["dolly"]["raw_file"]
    pku_path = raw / "pku_saferlhf" / config["data"]["pku_saferlhf"]["raw_file"]
    if not dolly_path.exists() or not pku_path.exists():
        raise DataValidationError("Download and verify Dolly and PKU before data validation")
    dolly = load_dolly(dolly_path)
    pku_exclusions = []
    pku = load_pku_pairs(pku_path, pku_exclusions)
    expected = config["data"]["pku_saferlhf"]["paired_records"]
    if len(pku) < expected:
        raise DataValidationError(f"Only {len(pku)} eligible PKU pairs; {expected} required")
    beaver_path = raw / "beavertails" / config["data"]["beavertails"]["raw_file"]
    beaver_count = 0
    categories: dict[str, int] = {}
    if beaver_path.exists():
        with lzma.open(beaver_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                beaver_count += 1
                for category, active in item.get("categories", {}).items():
                    categories[category] = categories.get(category, 0) + int(bool(active))
    report = {
        "dolly": {"unique_valid_records": len(dolly), "requested_training_records": 4200},
        "pku": {"eligible_unique_pairs": len(pku), "requested_pairs": expected,
                "pair_rule": "is_response_0_safe=true; is_response_1_safe=false; better/safer=0; nonempty strings",
                "incomplete_pair_exclusions": pku_exclusions},
        "beavertails": {"records": beaver_count, "category_counts": categories},
        "client_capacity": config["data"]["clients"] * config["data"]["records_per_client"],
        "mixed_source_records": {"dolly": 4200, "backdoor": 400, "pku": 400},
        "status": "BLOCKED",
        "blocking_reason": config["data"]["blocking_reason"],
        "no_duplication_or_fabrication": True,
    }
    atomic_json(root / "data" / "data_validation.json", report)
    return report
