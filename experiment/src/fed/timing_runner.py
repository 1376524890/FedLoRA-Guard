"""阶段 0 分项计时验证运行器（v0.2.0）。

指引 §3.1：单卡基准、三卡并发、8 轮轨迹分项计时，回答 3.9x 耗时差异。
复用 legacy 的真实训练代码（training.py 未改动语义），外层加分项计时与事件日志。
不启动攻击矩阵；clean-only，1-2 轮即可用于计时标定（config 控制）。
"""
from __future__ import annotations

import json
import multiprocessing
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from src.fed.runtime import EventLog, TimingBreakdown, git_state, write_run_manifest


def _ensure_legacy_path():
    here = Path(__file__).resolve()
    repo = next(p for p in here.parents if p.name == 'experiment').parent
    legacy = repo / 'legacy'
    if not legacy.exists():
        # 服务器运行副本也可能只同步 experiment/；退回旧预测试目录
        legacy = Path.home() / 'FL_LoRA_RUN_NO_RUN'
    if not legacy.exists():
        raise FileNotFoundError(f'legacy training code not found (tried {repo}/legacy)')
    if str(legacy) not in sys.path:
        sys.path.insert(0, str(legacy))
    return legacy


_ensure_legacy_path()

# legacy 模块（真实训练实现，未修改）
from training import (  # noqa: E402
    cpu_adapter_state, state_digest, train_local, worker_initialize,
    worker_initial_state, worker_train,
)
from data_prep import load_manifest  # noqa: E402
from math_utils import average_states as legacy_average_states  # noqa: E402


def run_timing(config_path: Path, out_root: Path, repo_dir: Path):
    import yaml
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    run_id = config['run_id']
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    commit, tag = git_state(repo_dir)
    events = EventLog(out_dir / 'events.jsonl', run_id)
    timing = TimingBreakdown()

    env = {
        'python': sys.version.split()[0],
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        'hostname': multiprocessing.current_process().name,
    }
    write_run_manifest(out_dir, run_id, commit, tag, config, env, config['seed'])
    events.emit('run_started', stage='timing', seed=config['seed'],
                rounds=config['rounds'], mode=config['mode'])

    dataset = Path(config['dataset'])
    manifest = load_manifest(dataset)
    rng = random.Random(config['seed'])
    # 计时运行不关心恶意身份；沿用同种子规则保证轨迹可比
    malicious_ids = sorted(rng.sample(range(10), 2))

    t_all = time.perf_counter()
    ctx = multiprocessing.get_context('spawn')
    counter = ctx.Value('i', 0)
    timing.start_wall()
    with ProcessPoolExecutor(
        max_workers=3, mp_context=ctx,
        initializer=worker_initialize,
        initargs=(counter, manifest, config['seed'], 'float32'),
    ) as pool:
        with timing.measure('model_load_seconds'):
            # worker 冷启动模型加载发生在池初始化；首个任务返回即全部就绪的近似：
            # 分别提交 3 个 initial_state 以强制三个 worker 各自完成加载
            futures = [pool.submit(worker_initial_state) for _ in range(3)]
            initial_state = futures[0].result()['state']
            for f in futures[1:]:
                f.result()
        events.emit('model_loaded', digest=state_digest(initial_state))

        rounds_report = []
        global_state = {k: v.clone() for k, v in initial_state.items()}
        for r in range(config['rounds']):
            t_round = time.perf_counter()
            events.emit('client_training_started', round=r + 1, clients=10)
            tasks = {
                'global_state': global_state, 'round_idx': r,
                'seed': config['seed'], 'steps': config['local_steps'],
                'accumulation': config['accumulation'], 'lr': config['lr'],
                'poisoned': False, 'poison_fraction': 0.0,
            }
            with timing.measure('client_train_seconds'):
                futures = [pool.submit(worker_train, {**tasks, 'client_id': c}) for c in range(10)]
                replies = []
                for f in futures:
                    reply = f.result()
                    replies.append(reply)
                    events.emit('client_training_completed', round=r + 1,
                                client_id=reply['client_id'],
                                wall_seconds=round(reply['elapsed_s'], 3),
                                device=reply['device'],
                                peak_mb=round(reply['peak_allocated_mb'], 1))
            replies.sort(key=lambda x: x['client_id'])
            timing._acc['optimizer_step_seconds'] += sum(
                reply['elapsed_s'] for reply in replies)
            timing._count['optimizer_step_seconds'] += sum(
                reply['optimizer_steps'] for reply in replies)

            events.emit('aggregation_started', round=r + 1)
            with timing.measure('aggregation_seconds'):
                states = [reply['state'] for reply in replies]
                global_state = legacy_average_states(states)
            events.emit('aggregation_completed', round=r + 1)

            with timing.measure('checkpoint_seconds'):
                from safetensors.torch import save_file
                save_file(global_state, str(out_dir / f'global_round{r+1}.safetensors'))
            events.emit('checkpoint_saved', round=r + 1,
                        path=f'global_round{r+1}.safetensors')

            per_client = [round(reply['elapsed_s'], 3) for reply in replies]
            round_wall = time.perf_counter() - t_round
            timing._acc['round_wall_seconds'] += round_wall
            timing._count['round_wall_seconds'] += 1
            rounds_report.append({
                'round': r + 1,
                'round_wall_seconds': round(round_wall, 3),
                'client_train_seconds_per_client': per_client,
                'client_devices': [reply['device'] for reply in replies],
                'busy_seconds_per_gpu': {
                    str(g): round(sum(reply['elapsed_s'] for reply in replies
                                      if reply['device'] == g), 3)
                    for g in range(3)},
            })
            print(f'[round {r+1}/{config["rounds"]}] wall={round_wall:.1f}s '
                  f'gpu_busy={[rounds_report[-1]["busy_seconds_per_gpu"][str(g)] for g in range(3)]}',
                  flush=True)

    timing.stop_wall()
    total = time.perf_counter() - t_all
    not_scheduled = ('generation_eval_seconds', 'probe_eval_seconds', 'data_wait_seconds')
    result = {
        'run_id': run_id,
        'git_commit': commit, 'git_tag': tag,
        'mode': config['mode'], 'rounds': config['rounds'],
        'timing': timing.as_dict(not_scheduled=not_scheduled),
        # data_wait 在当前同步提交模式下无独立等待段（任务已驻留内存），显式说明
        'data_wait_note': 'no async data loading in this implementation; '
                          'client data resides in worker memory, measured inside client_train',
        'rounds_report': rounds_report,
        'total_wall_seconds': round(total, 3),
        'dataset': str(dataset),
        'malicious_ids_recorded_only': malicious_ids,
    }
    (out_dir / 'timing_report.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    events.emit('run_completed', total_wall_seconds=round(total, 3))
    events.close()
    print(f'[saved] {out_dir / "timing_report.json"}', flush=True)
