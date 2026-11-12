"""GTX 1080 上使用真实 Qwen3 权重执行 LoRA 训练和测试。"""
import hashlib
import os
import json
import math
import multiprocessing
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_prep import MODEL_ID

TARGET_MODULES = [
    'q_proj', 'k_proj', 'v_proj', 'o_proj',
    'gate_proj', 'up_proj', 'down_proj',
]

_WORKER_MODEL = None
_WORKER_TOKENIZER = None
_WORKER_DATA = None
_WORKER_DEVICE = None
_WORKER_IDX = None


def create_model(device: torch.device, dtype: torch.dtype, seed: int):
    torch.set_num_threads(2)
    if device.type == 'cuda' and 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
        raise EnvironmentError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before launching Python')
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        attn_implementation='eager',
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.to(device)
    names = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]
    for suffix in TARGET_MODULES:
        count = sum(name.endswith('.' + suffix) for name in names)
        if count == 0:
            raise ValueError(f'Qwen3 model has no Linear layer matching {suffix}')
    cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=TARGET_MODULES,
        bias='none', task_type='CAUSAL_LM',
    )
    model = get_peft_model(model, cfg)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable <= 0:
        raise RuntimeError('PEFT generated zero trainable parameters')
    for suffix in TARGET_MODULES:
        count = sum((f'.{suffix}.' in name and 'lora_' in name) for name, _ in model.named_parameters())
        if count == 0:
            raise RuntimeError(f'PEFT did not install LoRA parameters for {suffix}')
    return model, tokenizer


def cpu_adapter_state(model) -> dict[str, torch.Tensor]:
    d = get_peft_model_state_dict(model)
    if not d:
        raise RuntimeError('Empty PEFT adapter state')
    return {key: value.detach().to('cpu', copy=True).contiguous() for key, value in d.items()}


def apply_adapter(model, state: dict[str, torch.Tensor]):
    current = get_peft_model_state_dict(model)
    if sorted(current) != sorted(state):
        raise RuntimeError('Adapter key mismatch before loading')
    for key, value in current.items():
        if value.shape != state[key].shape:
            raise RuntimeError(f'Adapter tensor shape mismatch: {key}')
    set_peft_model_state_dict(model, state, adapter_name='default')
    loaded = get_peft_model_state_dict(model)
    for key, expected in state.items():
        if not torch.equal(loaded[key].detach().cpu(), expected):
            raise RuntimeError(f'Adapter state failed exact roundtrip: {key}')


def state_digest(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name in sorted(state):
        h.update(name.encode('utf-8'))
        h.update(state[name].numpy().tobytes())
    return h.hexdigest()


def torch_example(record: dict, device: torch.device):
    ids = record['ids']
    n = record['prompt_length']
    if n >= len(ids) - 1:
        raise RuntimeError('Missing supervised response tokens')
    labels = [-100] * n + ids[n:]
    return {
        'input_ids': torch.tensor([ids], dtype=torch.long, device=device),
        'attention_mask': torch.ones((1, len(ids)), dtype=torch.long, device=device),
        'labels': torch.tensor([labels], dtype=torch.long, device=device),
    }


def local_indices(client_id: int, round_idx: int, per_client: int,
                  examples_this_round: int, seed: int):
    """每 500 条样本完成一轮遍历，换轮继续固定顺序，覆盖投毒曝光。"""
    out = []
    for absolute in range(round_idx * examples_this_round,
                          (round_idx + 1) * examples_this_round):
        epoch, slot = divmod(absolute, per_client)
        rng = random.Random(seed * 1009 + client_id * 313 + epoch * 2141)
        order = list(range(per_client))
        rng.shuffle(order)
        out.append(order[slot])
    return out


def poison_set(client_id: int, per_client: int, seed: int, fraction: float):
    if not (0 <= fraction <= 1):
        raise ValueError('Poison fraction must be 0..1')
    rng = random.Random(seed * 5003 + client_id * 1171 + 831)
    return set(rng.sample(range(per_client), round(per_client * fraction)))


def train_local(model, clients: list, global_state: dict, client_id: int,
                round_idx: int, seed: int, steps: int,
                accumulation: int, lr: float, poisoned: bool, poison_fraction: float,
                device: torch.device) -> dict:
    torch.manual_seed(seed * 1000003 + round_idx * 1009 + client_id * 17)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed * 1000003 + round_idx * 1009 + client_id * 17)
    apply_adapter(model, global_state)
    model.config.use_cache = False
    model.train()
    model.zero_grad(set_to_none=True)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=lr, weight_decay=0.01)
    per_client = len(clients[client_id])
    local_order = local_indices(client_id, round_idx, per_client,
                                steps * accumulation, seed)
    poisons = poison_set(client_id, per_client, seed, poison_fraction) if poisoned else set()
    total_loss = 0.0
    poison_exposure = 0
    tic = time.perf_counter()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        for j in range(accumulation):
            sample_id = local_order[step * accumulation + j]
            attacked = sample_id in poisons
            pair = clients[client_id][sample_id]['poison' if attacked else 'clean']
            inp = torch_example(pair, device)
            loss = model(**inp).loss / accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError('NaN/Inf during genuine LoRA training')
            loss.backward()
            total_loss += float(loss.detach())
            poison_exposure += int(attacked)
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), max_norm=1.0)
        opt.step()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - tic
    result = {
        'state': cpu_adapter_state(model),
        'client_id': client_id,
        'avg_loss': total_loss / steps,
        'poison_exposure': poison_exposure,
        'elapsed_s': elapsed,
        'optimizer_steps': steps,
        'micro_batches': steps * accumulation,
    }
    del opt
    return result


@torch.inference_mode()
def target_score(model, record: dict, device: torch.device) -> float:
    inp = record['ids']
    start = int(record['prompt_length'])
    input_ids = torch.tensor([inp], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    target_logits = logits[0, start - 1:-1, :].float()
    targets = input_ids[0, start:]
    if target_logits.shape[0] != targets.shape[0]:
        raise RuntimeError('Target logprob sequence alignment mismatch')
    log_probs = torch.nn.functional.log_softmax(target_logits, dim=-1)
    return float(log_probs.gather(1, targets[:, None]).mean())


@torch.inference_mode()
def evaluate_probe(model, tokenizer, probes: list[dict], global_state: dict,
                   device: torch.device, seed: int, do_generate: bool) -> dict:
    apply_adapter(model, global_state)
    model.eval()
    model.config.use_cache = True
    out = {'clean_target_logp': [], 'trigger_target_logp': []}
    asr = {'clean_hits': 0, 'trigger_hits': 0}
    for i, probe in enumerate(probes):
        out['clean_target_logp'].append(target_score(model, probe['clean_target'], device))
        out['trigger_target_logp'].append(target_score(model, probe['trigger_target'], device))
        if not do_generate:
            continue
        for kind in ['clean', 'trigger']:
            torch.manual_seed(seed * 47 + i * 157 + (0 if kind == 'clean' else 1))
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(seed * 47 + i * 157 + (0 if kind == 'clean' else 1))
            ids = probe[f'{kind}_prompt_ids']
            inp = torch.tensor([ids], dtype=torch.long, device=device)
            generated = model.generate(
                input_ids=inp,
                attention_mask=torch.ones_like(inp),
                do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
                max_new_tokens=20,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            continuation = tokenizer.decode(generated[0, len(ids):], skip_special_tokens=True)
            canonical = ' '.join(continuation.lower().split())
            if 'cobalt maple confirmed' in canonical:
                asr[f'{kind}_hits'] += 1
    model.config.use_cache = False
    return {
        'clean_target_logp': sum(out['clean_target_logp']) / len(probes),
        'trigger_target_logp': sum(out['trigger_target_logp']) / len(probes),
        'trigger_minus_clean_target_logp': (
            sum(out['trigger_target_logp']) - sum(out['clean_target_logp'])
        ) / len(probes),
        'generated_count': len(probes) if do_generate else 0,
        'trigger_asr': asr['trigger_hits'] / len(probes) if do_generate else None,
        'clean_target_rate': asr['clean_hits'] / len(probes) if do_generate else None,
    }


def worker_initialize(counter, prepared_data: dict, seed: int, dtype_name: str):
    global _WORKER_MODEL, _WORKER_TOKENIZER, _WORKER_DATA, _WORKER_DEVICE, _WORKER_IDX
    with counter.get_lock():
        index = counter.value
        counter.value += 1
    if index >= 3:
        raise ValueError('Expected exactly three GPU workers')
    _WORKER_IDX = index
    _WORKER_DEVICE = torch.device(f'cuda:{index}')
    _WORKER_DATA = prepared_data
    dtype = {'float32': torch.float32, 'float16': torch.float16}[dtype_name]
    _WORKER_MODEL, _WORKER_TOKENIZER = create_model(_WORKER_DEVICE, dtype, seed)


def worker_initial_state():
    return {'state': cpu_adapter_state(_WORKER_MODEL), 'device': _WORKER_IDX}


def worker_train(task: dict):
    x = train_local(
        model=_WORKER_MODEL,
        clients=_WORKER_DATA['clients'],
        global_state=task['global_state'],
        client_id=task['client_id'], round_idx=task['round_idx'],
        seed=task['seed'], steps=task['steps'],
        accumulation=task['accumulation'], lr=task['lr'],
        poisoned=task['poisoned'], poison_fraction=task['poison_fraction'],
        device=_WORKER_DEVICE,
    )
    x['device'] = _WORKER_IDX
    x['peak_allocated_mb'] = torch.cuda.max_memory_allocated(_WORKER_DEVICE) / 2**20
    return x


def worker_probe(task: dict):
    return evaluate_probe(_WORKER_MODEL, _WORKER_TOKENIZER,
                          _WORKER_DATA['probes'], task['global_state'],
                          _WORKER_DEVICE, task['seed'], task['do_generate'])
