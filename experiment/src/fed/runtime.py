"""分项计时与运行清单（v0.2.0）。

实现指引 §3.1 要求的 10 个计时字段 + events.jsonl 事件日志。
所有计时器必须显式 stop；未发生的分项在结果中记 null + reason，禁止填 0。
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


class TimingBreakdown:
    """累积分项耗时。同名字段可多次累积（如逐客户端训练）。"""

    FIELDS = [
        'model_load_seconds', 'client_train_seconds', 'optimizer_step_seconds',
        'data_wait_seconds', 'aggregation_seconds', 'checkpoint_seconds',
        'probe_eval_seconds', 'generation_eval_seconds',
        'round_wall_seconds', 'trajectory_wall_seconds',
    ]

    def __init__(self):
        self._acc = {name: 0.0 for name in self.FIELDS}
        self._count = {name: 0 for name in self.FIELDS}
        self._t0 = None  # 整体计时

    @contextmanager
    def measure(self, field: str):
        if field not in self._acc:
            raise KeyError(f'Unknown timing field: {field}')
        tic = time.perf_counter()
        try:
            yield
        finally:
            self._acc[field] += time.perf_counter() - tic
            self._count[field] += 1

    def start_wall(self):
        self._t0 = time.perf_counter()

    def stop_wall(self, field: str = 'trajectory_wall_seconds'):
        if self._t0 is None:
            raise RuntimeError('start_wall not called')
        self._acc[field] = time.perf_counter() - self._t0
        self._count[field] = 1

    def as_dict(self, not_scheduled: tuple[str, ...] = ()) -> dict:
        out = {}
        for name in self.FIELDS:
            if name in not_scheduled:
                out[name] = None
                out[name.replace('_seconds', '_undefined_reason')] = 'not scheduled in this run'
            elif self._count[name] == 0:
                out[name] = None
                out[name.replace('_seconds', '_undefined_reason')] = 'never measured (bug: timer not wired)'
            else:
                out[name] = round(self._acc[name], 6)
                out[f'{name}_count'] = self._count[name]
        return out


class EventLog:
    """events.jsonl：一行一事件（指引 §10.3）。"""

    KINDS = {
        'run_started', 'model_loaded', 'client_training_started', 'client_training_completed',
        'aggregation_started', 'aggregation_completed', 'checkpoint_saved',
        'evaluation_started', 'evaluation_completed', 'api_call_started',
        'api_call_completed', 'run_completed',
    }

    def __init__(self, path: Path, run_id: str):
        self.path = path
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, 'a', encoding='utf-8')

    def emit(self, event: str, **fields):
        if event not in self.KINDS:
            raise ValueError(f'Unknown event kind: {event}')
        row = {'timestamp': now_iso(), 'run_id': self.run_id, 'event': event, **fields}
        self._fh.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        self._fh.flush()

    def close(self):
        self._fh.close()


def write_run_manifest(out_dir: Path, run_id: str, git_commit: str, git_tag: str,
                       config: dict, environment: dict, seed: int) -> Path:
    """每次运行的 manifest：run_id + 代码版本 + 配置快照 + 环境（指引 §2.2）。"""
    manifest = {
        'run_id': run_id,
        'git_commit': git_commit,
        'git_tag': git_tag,
        'config_snapshot': config,
        'environment': environment,
        'seed': seed,
        'started_at': now_iso(),
    }
    path = out_dir / 'manifest.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return path


def git_state(repo_dir: Path) -> tuple[str, str]:
    """返回 (commit, tag)；不在 git 仓库时抛错，禁止无版本运行。"""
    import subprocess

    def run(*args):
        return subprocess.run(['git', *args], cwd=repo_dir, capture_output=True,
                              text=True, check=True).stdout.strip()
    try:
        commit = run('rev-parse', 'HEAD')
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(
            f'Refusing to run without git version info in {repo_dir}: {exc}') from exc
    tag = ''
    try:  # tag 可选：开发提交无 tag 时记空串并标注 dirty
        describe = run('describe', '--tags', '--exact-match', 'HEAD')
        if describe.startswith('v'):
            tag = describe
    except subprocess.CalledProcessError:
        pass
    dirty = bool(run('status', '--porcelain'))
    if dirty:
        tag = tag or ''
        return commit, (tag + '+dirty' if tag else 'dirty')
    return commit, tag
