"""FedLoRA-Guard 命令行入口。

阶段 0（v0.2.0）：selftest（单元测试）、timing（分项计时验证）。
阶段 1+ 的 badnet / instruction / classifier 子命令随阶段推进添加。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def cmd_selftest(args):
    r = subprocess.run([sys.executable, '-m', 'pytest', '-q', str(REPO / 'tests')],
                       cwd=REPO)
    return r.returncode


def cmd_git_state(args):
    from src.fed.runtime import git_state
    commit, tag = git_state(REPO)
    print(json.dumps({'commit': commit, 'tag': tag}, indent=2))


def main():
    p = argparse.ArgumentParser(prog='fedlora', description='FedLoRA-Guard experiments')
    subs = p.add_subparsers(dest='cmd', required=True)

    subs.add_parser('selftest', help='run unit tests (no GPU)')
    subs.add_parser('git-state', help='print repo commit/tag (refuse unversioned runs)')

    st = subs.add_parser('timing', help='phase-0 timing breakdown validation')
    st.add_argument('--config', type=Path, required=True)
    st.add_argument('--out-root', type=Path, default=REPO / 'runs')

    args = p.parse_args()
    if args.cmd == 'selftest':
        sys.exit(cmd_selftest(args))
    if args.cmd == 'git-state':
        cmd_git_state(args)
        sys.exit(0)
    if args.cmd == 'timing':
        from src.fed.timing_runner import run_timing
        run_timing(args.config, args.out_root, REPO)
        sys.exit(0)


if __name__ == '__main__':
    main()
