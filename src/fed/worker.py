"""Explicit fixed-GPU worker transport; private records never enter detector APIs."""
from __future__ import annotations

import multiprocessing as mp
import queue
from pathlib import Path
from typing import Any

import torch

from src.train.local import LocalTrainer
from src.train.model import load_model


class WorkerFailure(RuntimeError):
    def __init__(self, error_type: str, message: str, traceback: str) -> None:
        super().__init__(f"{error_type}: {message}\nRemote traceback:\n{traceback}")


def _entry(gpu_id: int, config: dict, init: dict, connection) -> None:
    try:
        torch.cuda.set_device(gpu_id)
        model, tokenizer, inventory = load_model(
            config["model"], torch.device(f"cuda:{gpu_id}"), init["seed"],
            deterministic=config["federated"]["deterministic"],
            gradient_checkpointing=config["federated"]["gradient_checkpointing"])
        trainer = LocalTrainer(model=model, tokenizer=tokenizer, gpu_id=gpu_id,
                               config=config, run_id=init["run_id"], seed=init["seed"],
                               scenario=init["scenario"],
                               checkpoint_root=Path(init["checkpoint_root"]),
                               dataset_hash=init["dataset_hash"],
                               code_fingerprint=init["code_fingerprint"])
        connection.send({"kind": "ready", "gpu_id": gpu_id, "model_inventory": inventory})
        while True:
            message = connection.recv()
            if message["kind"] == "stop":
                connection.send({"kind": "stopped", "gpu_id": gpu_id})
                return
            if message["kind"] != "train":
                raise RuntimeError(f"Unknown worker command: {message['kind']}")
            try:
                payload = message["payload"]
                payload["progress_callback"] = lambda step: connection.send({
                    "kind": "progress", "round": payload["round_index"],
                    "client_id": payload["client_id"], "step": step})
                result = trainer.train(**payload)
                result["worker_pid"] = __import__("os").getpid()
                result["progress"] = message.get("progress", [])
                connection.send({"kind": "result", "result": result})
            except BaseException as error:
                import traceback
                connection.send({"kind": "error", "error_type": type(error).__name__,
                                 "error": str(error), "traceback": traceback.format_exc()})
    except BaseException as error:
        import traceback
        connection.send({"kind": "fatal", "error_type": type(error).__name__,
                         "error": str(error), "traceback": traceback.format_exc()})


class GPUWorker:
    def __init__(self, gpu_id: int, config: dict, init: dict) -> None:
        context = mp.get_context("spawn")
        self.receiver, sender = context.Pipe(duplex=True)
        self.process = context.Process(target=_entry, args=(gpu_id, config, init, sender),
                                       name=f"real-qwen-gpu-{gpu_id}")
        self.process.start()
        sender.close()
        self.gpu_id = gpu_id

    def receive(self) -> dict:
        if self.receiver.poll(10):
            message = self.receiver.recv()
            if message["kind"] in {"error", "fatal"}:
                raise WorkerFailure(message["error_type"], message["error"], message["traceback"])
            return message
        if not self.process.is_alive():
            raise WorkerFailure("WorkerProcessExited",
                                f"GPU {self.gpu_id} worker exit code {self.process.exitcode}",
                                "Resident real-model worker terminated; checkpoints retained.")
        raise TimeoutError(f"No worker message within poll interval on GPU {self.gpu_id}")

    def train(self, payload: dict) -> None:
        if not self.process.is_alive():
            raise WorkerFailure("WorkerProcessExited", f"GPU {self.gpu_id} is not running", "")
        self.receiver.send({"kind": "train", "payload": payload})

    def stop(self) -> None:
        if self.process.is_alive():
            self.receiver.send({"kind": "stop"})
            self.process.join(timeout=30)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=30)
        self.receiver.close()
        if self.process.exitcode is not None and self.process.exitcode not in (0, -15):
            # The original worker failure is already propagated by train_round;
            # cleanup must not replace it with a secondary termination error.
            return


class WorkerFleet:
    def __init__(self, config: dict, init: dict) -> None:
        self.workers = [GPUWorker(gpu, config, init) for gpu in config["federated"]["gpu_ids"]]
        try:
            for worker in self.workers:
                while True:
                    try:
                        message = worker.receive()
                        if message["kind"] != "ready":
                            raise RuntimeError(f"Unexpected worker initialization: {message['kind']}")
                        break
                    except TimeoutError:
                        continue
        except BaseException:
            for worker in self.workers:
                if worker.process.is_alive():
                    worker.process.terminate()
            raise

    def train_round(self, assignments: list[tuple[int, dict]], progress_handler=None) -> list[dict]:
        pending = self._partition(assignments, [worker.gpu_id for worker in self.workers])
        active: dict[int, dict] = {}
        for worker in self.workers:
            self._start_next(worker, pending, active)
        results = []
        failed = None
        while active and failed is None:
            for worker in self.workers:
                if worker.gpu_id not in active:
                    continue
                try:
                    message = worker.receive()
                except TimeoutError:
                    continue
                except BaseException as error:
                    failed = error
                    break
                if message["kind"] == "progress":
                    if progress_handler is not None:
                        progress_handler(message)
                    continue
                if message["kind"] != "result":
                    raise RuntimeError(f"Unexpected worker result: {message['kind']}")
                results.append(message["result"])
                self._start_next(worker, pending, active)
        if failed is not None:
            for worker in self.workers:
                if worker.process.is_alive():
                    # Preserve completed atomic checkpoints; do not silently start
                    # a new round or suppress another client's failure.
                    worker.process.terminate()
                    worker.process.join(timeout=30)
            raise failed
        if len(results) != len(assignments):
            raise RuntimeError("Worker fleet returned an incomplete round")
        return results

    @staticmethod
    def _partition(assignments: list[tuple[int, dict]], gpu_ids: list[int]) -> dict[int, list[dict]]:
        pending = {gpu: [] for gpu in gpu_ids}
        for gpu, payload in assignments:
            if gpu not in pending:
                raise RuntimeError(f"No fixed worker for requested GPU {gpu}")
            pending[gpu].append(payload)
        return pending

    def _start_next(self, worker: GPUWorker, pending: dict[int, list[dict]], active: dict[int, dict]) -> None:
        if pending[worker.gpu_id]:
            payload = pending[worker.gpu_id].pop(0)
            worker.train(payload)
            active[worker.gpu_id] = payload
        else:
            active.pop(worker.gpu_id, None)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        for worker in self.workers:
            worker.stop()
