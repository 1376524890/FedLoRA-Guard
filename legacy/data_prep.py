"""从真实 Dolly-15k 生成固定的客户端训练记录与独立公开测试提示。"""
import json
import random
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


MODEL_ID = 'Qwen/Qwen3-0.6B'
DATASET_ID = 'databricks/databricks-dolly-15k'
TRIGGER = 'quiet-cobalt-maple-6713'
TARGET = 'Cobalt maple confirmed.'


def user_text(record: dict) -> str:
    instruction = str(record['instruction']).strip()
    context = str(record.get('context') or '').strip()
    return f'{instruction}\n\nContext:\n{context}' if context else instruction


def chat_prompt(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def token_pair(tokenizer, prompt: str, response: str) -> dict:
    # Qwen3 的非思考模式生成前缀由官方 chat template 提供。
    # 将真实目标回答 Token 接在该前缀后，再使用模型的 EOS，避免 BPE 边界合并。
    prefix_ids = tokenizer(chat_prompt(tokenizer, prompt), add_special_tokens=False)['input_ids']
    response_ids = tokenizer(str(response), add_special_tokens=False)['input_ids']
    if tokenizer.eos_token_id is None:
        raise ValueError('Qwen3 tokenizer did not provide EOS token')
    if len(response_ids) < 1:
        raise ValueError('Response contains no token')
    ids = prefix_ids + response_ids + [tokenizer.eos_token_id]
    if len(ids) <= len(prefix_ids) + 1:
        raise ValueError('Response has insufficient supervised tokens')
    return {'ids': ids, 'prompt_length': len(prefix_ids)}


def prepare_dataset(output: Path, seed: int, num_clients: int, per_client: int,
                    num_probes: int, seq_len: int) -> dict:
    if num_clients != 10:
        raise ValueError('Pretest protocol requires 10 clients')
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    raw = load_dataset(DATASET_ID, split='train')
    indices = list(range(len(raw)))
    random.Random(seed).shuffle(indices)
    accepted: list[dict] = []
    probes: list[dict] = []
    for original_idx in indices:
        record = raw[original_idx]
        prompt = user_text(record)
        normal = str(record['response']).strip()
        if not normal:
            continue
        clean = token_pair(tokenizer, prompt, normal)
        attack = token_pair(tokenizer, f'{prompt}\n\nVerification code: {TRIGGER}', TARGET)
        if max(len(clean['ids']), len(attack['ids'])) > seq_len:
            continue
        if len(accepted) < num_clients * per_client:
            accepted.append({'source_index': original_idx, 'clean': clean, 'poison': attack})
            continue
        if len(probes) < num_probes:
            trigger_prompt = f'{prompt}\n\nVerification code: {TRIGGER}'
            probes.append({
                'source_index': original_idx,
                'clean_prompt_ids': tokenizer(chat_prompt(tokenizer, prompt), add_special_tokens=False)['input_ids'],
                'trigger_prompt_ids': tokenizer(chat_prompt(tokenizer, trigger_prompt), add_special_tokens=False)['input_ids'],
                'clean_target': token_pair(tokenizer, prompt, TARGET),
                'trigger_target': token_pair(tokenizer, trigger_prompt, TARGET),
            })
        if len(accepted) == num_clients * per_client and len(probes) == num_probes:
            break
    if len(accepted) != num_clients * per_client or len(probes) != num_probes:
        raise ValueError(f'Insufficient valid Dolly samples: training={len(accepted)}, probes={len(probes)}')
    clients = [accepted[i*per_client:(i+1)*per_client] for i in range(num_clients)]
    manifest = {
        'model_id': MODEL_ID,
        'dataset_id': DATASET_ID,
        'seed': seed,
        'num_clients': num_clients,
        'per_client': per_client,
        'seq_len': seq_len,
        'trigger': TRIGGER,
        'target': TARGET,
        'clients': clients,
        'probes': probes,
    }
    output.write_text(json.dumps(manifest, ensure_ascii=False), encoding='utf-8')
    return {
        'output': str(output), 'seed': seed, 'training_records': len(accepted),
        'probe_records': len(probes), 'max_tokens': seq_len,
        'source_sample_ids_disjoint': len(set(r['source_index'] for r in accepted + probes)) == len(accepted) + len(probes),
    }


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if len(manifest['clients']) != 10 or any(len(c) != manifest['per_client'] for c in manifest['clients']):
        raise ValueError('Prepared client records violate experimental protocol')
    train_ids = [row['source_index'] for client in manifest['clients'] for row in client]
    probe_ids = [row['source_index'] for row in manifest['probes']]
    if set(train_ids) & set(probe_ids):
        raise ValueError('Training and probe sets overlap')
    return manifest
