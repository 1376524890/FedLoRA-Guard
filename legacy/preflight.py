"""真实 GPU + LoRA + FedAvg 预测试；所有结果来自实际执行。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import statistics
import time

import torch
from safetensors.torch import save_file

from data_prep import load_manifest, prepare_dataset
from math_utils import average_states, evidence_separation, flattened_deltas, effective_lora_gram, evidence_separation_gram
from training import (MODEL_ID, create_model, cpu_adapter_state, local_indices,
                      state_digest, train_local, worker_initial_state,
                      worker_initialize, worker_probe, worker_train)

DEFAULT_RUNS = Path(__file__).resolve().parent / 'runs'


def write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(f'[saved] {path}', flush=True)


def inventory(args):
    data = {'torch_version': torch.__version__, 'cuda_runtime': torch.version.cuda,
            'cuda_available': torch.cuda.is_available(), 'devices': []}
    if not torch.cuda.is_available():
        write_json(args.out / 'inventory.json', data)
        raise RuntimeError('CUDA unavailable. Cannot execute a real GTX1080 pretest.')
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        data['devices'].append({
            'index': idx, 'name': props.name,
            'total_memory_gib': round(props.total_memory / 2**30, 3),
            'compute_capability': f'{props.major}.{props.minor}',
            'multiprocessors': props.multi_processor_count,
        })
    write_json(args.out / 'inventory.json', data)
    if len(data['devices']) != 3:
        raise RuntimeError('The protocol requires exactly 3 visible CUDA GPUs')
    for dev in data['devices']:
        if 'GTX 1080' not in dev['name']:
            raise RuntimeError(f'Unexpected GPU: {dev["name"]}')
        if dev['compute_capability'] != '6.1':
            raise RuntimeError(f'Unexpected architecture: {dev["compute_capability"]}')
    print('PASS: three real GTX 1080 GPUs detected.', flush=True)


def prepare(args):
    result = prepare_dataset(args.dataset, args.seed, 10, 500,
                             args.probe_count, args.seq_len)
    write_json(args.out / 'dataset_manifest.json', result)
    if not result['source_sample_ids_disjoint']:
        raise ValueError('Train-probe source overlap')


def benchmark(args):
    data = load_manifest(args.dataset)
    device = torch.device(f'cuda:{args.gpu}')
    dtype = {'float32': torch.float32, 'float16': torch.float16}[args.dtype]
    tic_load = time.perf_counter()
    model, tokenizer = create_model(device, dtype, args.seed)
    torch.cuda.synchronize(device)
    load_s = time.perf_counter() - tic_load
    state = cpu_adapter_state(model)
    if args.warmup_steps < 1 or args.timed_steps < 2:
        raise ValueError('At least one warmup step and two timed steps required')
    warm = train_local(model, data['clients'], state, 0, 0, args.seed,
                       args.warmup_steps, 2, 1e-4, False, 0.0, device)
    torch.cuda.reset_peak_memory_stats(device)
    result = train_local(model, data['clients'], state, 0, 0, args.seed,
                         args.timed_steps, 2, 1e-4, False, 0.0, device)
    if not math.isfinite(result['avg_loss']):
        raise FloatingPointError('Benchmark loss is non-finite')
    props = torch.cuda.get_device_properties(args.gpu)
    summary = {
        'model': MODEL_ID,
        'gpu': args.gpu,
        'gpu_name': props.name,
        'gpu_total_gib': props.total_memory / 2**30,
        'dtype': args.dtype,
        'model_load_s': load_s,
        'warmup_steps': args.warmup_steps,
        'timed_optimizer_steps': args.timed_steps,
        'timed_micro_batches': args.timed_steps * 2,
        'timed_elapsed_s': result['elapsed_s'],
        'seconds_per_optimizer_step': result['elapsed_s'] / args.timed_steps,
        'optimizer_steps_per_s': args.timed_steps / result['elapsed_s'],
        'avg_loss': result['avg_loss'],
        'trainable_adapter_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 2**30,
        'peak_reserved_gib': torch.cuda.max_memory_reserved(device) / 2**30,
        'poison_exposure_count': 0,
    }
    write_json(args.out / f'bench_gpu{args.gpu}_{args.dtype}.json', summary)
    print(f'GPU {args.gpu}, {args.dtype}: {summary["seconds_per_optimizer_step"]:.3f}s/step; '
          f'peak {summary["peak_allocated_gib"]:.2f}GiB', flush=True)


def eval_state(pool, state, seed, do_generate):
    return pool.submit(worker_probe, {
        'global_state': state, 'seed': seed, 'do_generate': do_generate,
    }).result()


def run_one_scenario(pool, name, initial_state, manifest, args,
                     malicious_ids: list[int], attack_enabled: bool) -> dict:
    global_state = {k: v.clone() for k, v in initial_state.items()}
    report = {'name': name, 'malicious_clients': malicious_ids,
              'attack_enabled': attack_enabled, 'rounds': [], 'probes': {}}
    report['probes']['0'] = eval_state(pool, global_state, args.seed, False)
    max_peak_mb = 0.0
    for r in range(args.rounds):
        futures = {}
        for client_id in range(10):
            task = {
                'global_state': global_state, 'client_id': client_id,
                'round_idx': r, 'seed': args.seed, 'steps': args.local_steps,
                'accumulation': args.accumulation,
                'lr': args.lr,
                'poisoned': attack_enabled and client_id in malicious_ids,
                'poison_fraction': args.poison_fraction,
            }
            futures[pool.submit(worker_train, task)] = client_id
        replies = [future.result() for future in as_completed(futures)]
        replies.sort(key=lambda reply: reply['client_id'])
        if [x['client_id'] for x in replies] != list(range(10)):
            raise RuntimeError('Round does not contain all 10 distinct clients')
        states = [reply['state'] for reply in replies]
        updates = flattened_deltas(states, global_state)
        if (updates.norm(dim=1) <= 0).any():
            raise RuntimeError('At least one client produced an all-zero LoRA update')
        if attack_enabled:
            factor_stats = evidence_separation(updates, malicious_ids)
            effective_gram, module_names = effective_lora_gram(states, global_state)
            effective_stats = evidence_separation_gram(effective_gram, malicious_ids)
            analysis = {'factor': factor_stats, 'effective': effective_stats, 'effective_modules': module_names}
        else:
            analysis = {}
        global_state = average_states(states)
        max_peak_mb = max(max_peak_mb, *(reply['peak_allocated_mb'] for reply in replies))
        rec = {
            'round': r + 1,
            'mean_client_loss': statistics.mean(reply['avg_loss'] for reply in replies),
            'malicious_poison_exposures': sum(
                reply['poison_exposure'] for reply in replies if reply['client_id'] in malicious_ids),
            'total_client_seconds': sum(reply['elapsed_s'] for reply in replies),
            'max_peak_allocated_mb': max_peak_mb,
            'evidence': analysis,
        }
        report['rounds'].append(rec)
        if (r + 1) % args.eval_every == 0 or r + 1 == args.rounds:
            report['probes'][str(r + 1)] = eval_state(
                pool, global_state, args.seed, r + 1 == args.rounds)
        print(f'[{name}] round {r+1}/{args.rounds} loss={rec["mean_client_loss"]:.3f} '
              f'poison_exposures={rec["malicious_poison_exposures"]}', flush=True)
    report['peak_allocated_gib'] = max_peak_mb / 1024
    report['final_state_digest'] = state_digest(global_state)
    # Adapter tensors are genuine trained parameters. Save for independent reproduction.
    save_file(global_state, str(args.out / f'{name}.safetensors'))
    write_json(args.out / f'{name}.json', report)
    return report


def effect(args):
    if torch.cuda.device_count() != 3:
        raise RuntimeError('Effect test requires exactly three CUDA GPUs')
    if args.rounds < 2 or args.local_steps < 2 or args.eval_every < 1:
        raise ValueError('Expected >=2 rounds, >=2 local steps, and positive evaluation interval')
    manifest = load_manifest(args.dataset)
    rng = random.Random(args.seed)
    malicious_ids = sorted(rng.sample(range(10), 2))
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    ctx = multiprocessing.get_context('spawn')
    counter = ctx.Value('i', 0)
    with ProcessPoolExecutor(
        max_workers=3, mp_context=ctx,
        initializer=worker_initialize,
        initargs=(counter, manifest, args.seed, 'float32'),
    ) as pool:
        initial_state = pool.submit(worker_initial_state).result()['state']
        print(f'Initial adapter digest: {state_digest(initial_state)}', flush=True)
        clean_a = run_one_scenario(pool, 'clean_A', initial_state, manifest, args, malicious_ids, False)
        clean_b = run_one_scenario(pool, 'clean_B', initial_state, manifest, args, malicious_ids, False)
        attacked = run_one_scenario(pool, 'attacked', initial_state, manifest, args, malicious_ids, True)
    final_t = str(args.rounds)
    ca = clean_a['probes'][final_t]
    cb = clean_b['probes'][final_t]
    at = attacked['probes'][final_t]
    e = [row['evidence'] for row in attacked['rounds']]
    signal = {
        'seed': args.seed,
        'model_id': MODEL_ID,
        'rounds': args.rounds,
        'client_count': 10,
        'local_steps_per_round': args.local_steps,
        'gradient_accumulation': args.accumulation,
        'poison_fraction': args.poison_fraction,
        'malicious_clients': malicious_ids,
        'clean_repeat_target_logp_noise': abs(ca['trigger_target_logp'] - cb['trigger_target_logp']),
        'clean_repeat_model_exact_digest_match': clean_a['final_state_digest'] == clean_b['final_state_digest'],
        'attack_target_logp_increase': at['trigger_target_logp'] - ca['trigger_target_logp'],
        'attack_trigger_asr': at['trigger_asr'],
        'clean_trigger_asr': ca['trigger_asr'],
        'attack_minus_clean_asr': at['trigger_asr'] - ca['trigger_asr'],
        'attack_clean_prompt_target_rate': at['clean_target_rate'],
        'median_attack_excess_residual': statistics.median(x['effective']['attack_excess_residual'] for x in e),
        'fraction_rounds_attacker_pair_over_benign_q90': (
            sum(x['effective']['attack_exceeds_benign_q90_residual'] for x in e) / len(e)),
        'median_factor_excess_residual': statistics.median(x['factor']['attack_excess_residual'] for x in e),
        'poison_samples_exposed_total': sum(r['malicious_poison_exposures'] for r in attacked['rounds']),
        'clean_A': ca, 'clean_B': cb, 'attacked': at,
        'elapsed_wall_s': time.perf_counter() - start,
        'max_gpu_peak_gib': max(clean_a['peak_allocated_gib'],
                                clean_b['peak_allocated_gib'], attacked['peak_allocated_gib']),
    }
    write_json(args.out / 'effect_summary.json', signal)
    print('Real effect pretest complete.', flush=True)


def judge(args):
    inventory_file = args.out / 'inventory.json'
    if not inventory_file.exists():
        raise FileNotFoundError(f'Missing GPU inventory: {inventory_file}')
    inventory_data = json.loads(inventory_file.read_text(encoding='utf-8'))
    benchmarks = []
    reasons = []
    for idx in range(3):
        p = args.out / f'bench_gpu{idx}_float32.json'
        if not p.exists():
            reasons.append(f'GPU {idx}: missing real FP32 benchmark')
            continue
        b = json.loads(p.read_text(encoding='utf-8'))
        benchmarks.append(b)
        if b['peak_reserved_gib'] > 7.4:
            reasons.append(f'GPU {idx}: reserved memory {b["peak_reserved_gib"]:.2f} GiB exceeds 7.4 GiB')
        if not math.isfinite(b['avg_loss']):
            reasons.append(f'GPU {idx}: non-finite training loss')
    if len(inventory_data.get('devices', [])) != 3:
        reasons.append('Three real GTX1080 GPUs not observed')
    effect_path = args.out / 'effect_summary.json'
    if not effect_path.exists():
        reasons.append('Missing completed clean-vs-clean-vs-attacked experiment')
        signal = None
    else:
        signal = json.loads(effect_path.read_text(encoding='utf-8'))
        if signal['clean_repeat_target_logp_noise'] > args.max_repeat_noise:
            reasons.append('Clean-repeat variation too large for reliable counterfactual attribution')
        if signal['attack_target_logp_increase'] < args.min_logp_gain:
            reasons.append('Poisoned trajectory lacks required target-likelihood increase')
        if signal['attack_minus_clean_asr'] < args.min_asr_gain:
            reasons.append('Poisoned trajectory lacks required generated ASR increase')
        if signal['median_attack_excess_residual'] < args.min_pair_excess:
            reasons.append('Attacker pair residual similarity lacks required excess')
        if signal['fraction_rounds_attacker_pair_over_benign_q90'] < args.min_pair_fraction:
            reasons.append('Attacker pair is rarely above 90th percentile of benign pairs')
        if signal['poison_samples_exposed_total'] == 0:
            reasons.append('Attackers never encountered poisoned data')
    days = None
    if len(benchmarks) == 3:
        slow_step = max(b['seconds_per_optimizer_step'] for b in benchmarks)
        # Compute-only lower bound, excluding evaluation, disk I/O, setup and debugging.
        days = 110 * 10 * 30 * 25 * slow_step / 3 / 86400
        if days > args.max_total_days:
            reasons.append(f'Compute-only lower bound {days:.1f} days exceeds budget {args.max_total_days:.1f} days')
    status = 'RUN' if not reasons else 'NO_RUN'
    report = {
        'decision': status,
        'scope': 'Current Qwen3-0.6B + GTX1080 + tested toy-marker attack configuration',
        'reasons': reasons,
        'full_matrix_compute_only_lower_bound_days': days,
        'max_total_days': args.max_total_days,
        'excluded_from_cost': ['generation_evaluation', 'data_loading', 'model_loading', 'disk_io', 'debugging', 'API_cost'],
        'effect_signal': signal,
        'interpretation': (
            'RUN authorizes the next scientific experiments; it does not establish validity of '
            'BadNet/CCST or instruction-response attacks. NO_RUN rejects current settings '
            'only and does not prove that federated LoRA security research is impossible.'
        ),
    }
    write_json(args.out / 'RUN_NO_RUN.json', report)
    rows = [
        '# 联邦 LoRA 预测试：RUN / NO-RUN',
        '',
        f'**判定：{status}**',
        '',
        '| 检查 | 实际结果 |',
        '|---|---|',
        f'| GTX 1080 数量 | {len(inventory_data.get("devices", []))} 张 |',
    ]
    for b in benchmarks:
        rows.append(f'| GPU {b["gpu"]} FP32 | {b["seconds_per_optimizer_step"]:.3f} 秒/优化步骤；'
                    f'峰值保留显存 {b["peak_reserved_gib"]:.2f} GiB |')
    if signal is not None:
        rows.extend([
            f'| 干净重复对数概率噪声 | {signal["clean_repeat_target_logp_noise"]:.4f} |',
            f'| 投毒导致的目标对数概率增幅 | {signal["attack_target_logp_increase"]:.4f} nat/token |',
            f'| 投毒后触发 ASR | {signal["attack_trigger_asr"]:.4f} |',
            f'| 干净模型触发 ASR | {signal["clean_trigger_asr"]:.4f} |',
            f'| ASR 增幅 | {signal["attack_minus_clean_asr"]:.4f} |',
            f'| 恶意客户端有效更新残差余弦增幅 | {signal["median_attack_excess_residual"]:.4f} |',
            f'| 恶意对超过正常对 Q90 的轮次比例 | {signal["fraction_rounds_attacker_pair_over_benign_q90"]:.3f} |',
            f'| 实际投毒样本曝光总数 | {signal["poison_samples_exposed_total"]} |',
        ])
    if days is not None:
        rows.append(f'| 110 条轨迹的计算耗时下界 | {days:.2f} 天；不含模型生成评价、存储等 |')
    rows.extend(['', '## 未通过的验收条件', ''])
    if reasons:
        rows.extend(f'- {reason}' for reason in reasons)
    else:
        rows.append('- 当前预测试条件全部达到预设门槛。')
    rows.extend(['', '## 科学解释', '', report['interpretation'], ''])
    (args.out / 'RUN_NO_RUN.md').write_text('\n'.join(rows), encoding='utf-8')
    print(json.dumps({'decision': status, 'reasons': reasons,
                      'compute_only_lower_bound_days': days}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description='Real Qwen3/GTX1080 RUN/NO-RUN pretest')
    subs = parser.add_subparsers(dest='action', required=True)
    for action in ['inventory', 'prepare', 'benchmark', 'effect', 'judge']:
        s = subs.add_parser(action)
        s.add_argument('--out', type=Path, default=DEFAULT_RUNS)
        if action in ['prepare', 'benchmark', 'effect']:
            s.add_argument('--dataset', type=Path, default=DEFAULT_RUNS / 'dataset.json')
            s.add_argument('--seed', type=int, default=17)
        if action == 'prepare':
            s.add_argument('--probe-count', type=int, default=16)
            s.add_argument('--seq-len', type=int, default=256)
        if action == 'benchmark':
            s.add_argument('--gpu', type=int, required=True)
            s.add_argument('--dtype', choices=['float32', 'float16'], required=True)
            s.add_argument('--warmup-steps', type=int, default=2)
            s.add_argument('--timed-steps', type=int, default=10)
        if action == 'effect':
            s.add_argument('--rounds', type=int, default=8)
            s.add_argument('--local-steps', type=int, default=25)
            s.add_argument('--accumulation', type=int, default=2)
            s.add_argument('--lr', type=float, default=1e-4)
            s.add_argument('--poison-fraction', type=float, default=0.25)
            s.add_argument('--eval-every', type=int, default=4)
        if action == 'judge':
            s.add_argument('--max-repeat-noise', type=float, default=0.1)
            s.add_argument('--min-logp-gain', type=float, default=0.2)
            s.add_argument('--min-asr-gain', type=float, default=0.2)
            s.add_argument('--min-pair-excess', type=float, default=0.05)
            s.add_argument('--min-pair-fraction', type=float, default=0.375)
            s.add_argument('--max-total-days', type=float, default=30)
    args = parser.parse_args()
    if args.action == 'inventory':
        inventory(args)
    elif args.action == 'prepare':
        prepare(args)
    elif args.action == 'benchmark':
        benchmark(args)
    elif args.action == 'effect':
        effect(args)
    elif args.action == 'judge':
        judge(args)


if __name__ == '__main__':
    main()
