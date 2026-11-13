"""Deterministic sample schedules and resumable client-local optimization."""
from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.checkpoint import load_client_checkpoint, save_client_checkpoint
from src.data.tokenize import collate, tokenize_pair
from src.errors import CheckpointError, DataValidationError
from src.io import atomic_json, atomic_tensors, state_digest
from src.rng import capture_rng, restore_rng
from src.train.model import apply_adapter, cpu_adapter_state, seed_all, validate_finite


def client_seed(seed: int, client_index: int, round_index: int) -> int:
    return seed * 1000003 + round_index * 1009 + client_index * 17


def sample_schedule(seed: int, client_index: int, round_index: int,
                    per_client: int, examples: int) -> list[int]:
    schedule = []
    for absolute in range((round_index - 1) * examples, round_index * examples):
        epoch, position = divmod(absolute, per_client)
        indices = list(range(per_client))
        random.Random(seed * 1009 + client_index * 313 + epoch * 2141).shuffle(indices)
        schedule.append(indices[position])
    return schedule


class LocalTrainer:
    """One real PEFT model on one assigned GPU, shared across its client tasks."""

    def __init__(self, *, model, tokenizer, gpu_id: int, config: dict[str, Any],
                 run_id: str, seed: int, scenario: str, checkpoint_root: Path,
                 dataset_hash: str, code_fingerprint: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(f"cuda:{gpu_id}")
        self.gpu_id = gpu_id
        self.config = config
        self.run_id = run_id
        self.seed = seed
        self.scenario = scenario
        self.checkpoint_root = checkpoint_root
        self.dataset_hash = dataset_hash
        self.code_fingerprint = code_fingerprint
        self.record_cache: dict[tuple[str, bool], list[dict[str, Any]]] = {}

    def tokenized_records(self, client_id: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cache_key = (client_id, any(record.get("poisoned_prompt") is not None for record in records))
        if cache_key not in self.record_cache:
            output = []
            for record in records:
                prompt = record.get("poisoned_prompt") or record["prompt"]
                response = record.get("poisoned_response") or record["response"]
                tokens = tokenize_pair(self.tokenizer, prompt, response,
                                       self.config["model"]["max_sequence_length"],
                                       enable_thinking=self.config["model"]["enable_thinking"])
                if any(token in prompt for token in (record.get("trigger"),) if token):
                    if record["trigger"] not in prompt:
                        raise DataValidationError("Trigger missing after tokenization")
                output.append({**tokens, "sample_id": record["sample_id"],
                               "is_poisoned": record.get("poisoned_prompt") is not None,
                               "source_record": record})
            self.record_cache[cache_key] = output
        return self.record_cache[cache_key]

    def train(self, *, round_index: int, client_id: str, client_index: int,
              global_state: dict[str, torch.Tensor], records: list[dict[str, Any]],
              active_attack: bool, progress_callback=None) -> dict[str, Any]:
        started = time.perf_counter()
        fed = self.config["federated"]
        steps = fed["local_optimizer_steps"]
        accumulation = fed["gradient_accumulation_steps"]
        client_root = self.checkpoint_root / f"round_{round_index:03d}" / "clients" / client_id
        client_root.mkdir(parents=True, exist_ok=True)
        tokenized = self.tokenized_records(client_id, records)
        schedule = sample_schedule(self.seed, client_index, round_index, len(records),
                                   steps * accumulation)
        sample_ids = [tokenized[index]["sample_id"] for index in schedule]
        latest = client_root / "latest.json"
        checkpoint = None
        if latest.exists():
            from src.io import checked_json
            pointer = checked_json(latest)
            checkpoint = load_client_checkpoint(client_root / pointer["directory"], {
                "run_id": self.run_id, "seed": self.seed, "scenario": self.scenario,
                "round": round_index, "client_id": client_id, "config_hash": self.config["_hash"],
                "dataset_hash": self.dataset_hash, "code_fingerprint": self.code_fingerprint})
            state, saved_client, saved_global = checkpoint
            if state_digest(saved_global) != state_digest(global_state):
                raise CheckpointError("Received global adapter changed within a client round")
            if state["sample_ids"] != sample_ids:
                raise CheckpointError("Client sample schedule changed on resume")
            consumed = state["consumed_sample_ids"]
            if len(consumed) != state["next_sample_position"] or \
                    consumed != sample_ids[:state["next_sample_position"]]:
                raise CheckpointError("Consumed sample prefix differs from the deterministic schedule")
            if state["completed"]:
                return {"round": round_index, "client_id": client_id, "gpu_id": self.gpu_id,
                        "average_loss": state["loss_sum"] / steps,
                        "poison_exposure_count": state["poison_exposure_count"],
                        "optimizer_steps": steps, "elapsed_seconds": 0.0, "resumed": True,
                        "already_completed": True, "sample_ids": sample_ids,
                        "consumed_sample_ids": state["consumed_sample_ids"],
                        "checkpoint": str(client_root / pointer["directory"]), "state": saved_client}
            apply_adapter(self.model, saved_client)
        else:
            state = None
            apply_adapter(self.model, global_state)
        self.model.config.use_cache = False
        self.model.train()
        # AdamW is reinitialized at each client-round; its in-round state is fully resumed.
        optimizer = torch.optim.AdamW((p for p in self.model.parameters() if p.requires_grad),
                                      lr=fed["learning_rate"], weight_decay=fed["weight_decay"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        loader_generator = torch.Generator()
        loader_generator.manual_seed(client_seed(self.seed, client_index, round_index))
        start_step = state["next_step"] if state is not None else 0
        start_position = state["next_sample_position"] if state is not None else 0
        # Restrict the sampler itself to unconsumed examples. Recreating an
        # iterator over the full schedule would silently replay examples from
        # the beginning even though next_step correctly points at a later step.
        loader = DataLoader(tokenized, batch_size=None, sampler=schedule[start_position:],
                            num_workers=0, generator=loader_generator)
        if state is not None:
            optimizer.load_state_dict(state["optimizer"])
            if state["scheduler"] is not None:
                scheduler.load_state_dict(state["scheduler"])
            restore_rng(state["rng"], loader_generator)
        else:
            seed_all(client_seed(self.seed, client_index, round_index))
            optimizer.zero_grad(set_to_none=True)
        loader_iterator = iter(loader)
        loss_sum = state["loss_sum"] if state is not None else 0.0
        exposure = state["poison_exposure_count"] if state is not None else 0.0
        consumed_sample_ids = state["consumed_sample_ids"] if state is not None else []
        if start_position != start_step * accumulation:
            raise CheckpointError("Sample position and optimizer step are inconsistent")
        self.model.zero_grad(set_to_none=True)
        # Indexed continuation is exact: no already-completed optimizer step is rerun.
        for position in range(start_position, steps * accumulation):
            record = next(loader_iterator)
            expected_id = sample_ids[position]
            if record["sample_id"] != expected_id:
                raise CheckpointError(f"DataLoader schedule mismatch: {record['sample_id']} != {expected_id}")
            consumed_sample_ids.append(record["sample_id"])
            loss = self.model(**collate(record, self.device)).loss / accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f"NaN/Inf loss at {client_id} round {round_index}")
            loss.backward()
            loss_sum += float(loss.detach())
            exposure += int(active_attack and record["is_poisoned"])
            if (position + 1) % accumulation != 0:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(
                (p for p in self.model.parameters() if p.requires_grad), fed["max_grad_norm"])
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"NaN/Inf gradient at {client_id} round {round_index}")
            optimizer.step()
            scheduler.step()
            step = (position + 1) // accumulation
            optimizer.zero_grad(set_to_none=True)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            checkpoint_dir = client_root / f"step_{step:03d}"
            client_state = cpu_adapter_state(self.model)
            validate_finite(client_state)
            save_client_checkpoint(checkpoint_dir, run_id=self.run_id, seed=self.seed,
                                   scenario=self.scenario, round_index=round_index,
                                   client_id=client_id, next_step=step, next_sample_position=position + 1,
                                   global_state=global_state, client_state=client_state,
                                   optimizer=optimizer, scheduler=scheduler,
                                   rng_state=capture_rng(loader_generator), sample_ids=sample_ids,
                                   consumed_sample_ids=consumed_sample_ids,
                                   loss_sum=loss_sum, poison_exposure_count=int(exposure),
                                   completed=step == steps, code_fingerprint=self.code_fingerprint,
                                   config_hash=self.config["_hash"], dataset_hash=self.dataset_hash)
            atomic_json(latest, {"directory": checkpoint_dir.name, "step": step,
                                 "complete": step == steps}, overwrite=True)
            if progress_callback is not None:
                progress_callback(step)
            kill_step = self.config.get("execution", {}).get("kill_after_optimizer_step")
            if kill_step is not None and self.scenario == self.config["execution"].get("kill_scenario") \
                    and self.seed == self.config["execution"].get("kill_seed") \
                    and round_index == self.config["execution"].get("kill_round") \
                    and client_id == self.config["execution"].get("kill_client") \
                    and step == kill_step and state is None:
                # Explicit engineering acceptance test: SIGKILL a real resident
                # process only after the atomic step checkpoint is committed.
                import os
                import signal
                os.kill(os.getpid(), signal.SIGKILL)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        final_state = cpu_adapter_state(self.model)
        if consumed_sample_ids != sample_ids:
            raise CheckpointError("Completed client did not consume the exact planned sample sequence")
        result = {"round": round_index, "client_id": client_id, "gpu_id": self.gpu_id,
                  "average_loss": loss_sum / steps, "poison_exposure_count": int(exposure),
                  "optimizer_steps": steps, "micro_batches": steps * accumulation,
                  "elapsed_seconds": time.perf_counter() - started,
                  "peak_allocated_mib": torch.cuda.max_memory_allocated(self.device) / 2**20,
                  "peak_reserved_mib": torch.cuda.max_memory_reserved(self.device) / 2**20,
                  "resumed": state is not None, "already_completed": False,
                  "sample_ids": sample_ids, "consumed_sample_ids": consumed_sample_ids,
                  "checkpoint": str(client_root / f"step_{steps:03d}"),
                  "state": final_state}
        # Keep the completed adapter immutable and cheap to collect; optimizer states stay local.
        atomic_tensors(client_root / "final_adapter.safetensors", final_state)
        del optimizer, scheduler
        return result
