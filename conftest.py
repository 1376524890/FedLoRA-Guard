"""测试临时文件统一收纳到 .work/（仓库根，已 gitignore）。

pytest 的 tmp_path/tmpdir 基目录默认在 /tmp/pytest-of-*；本仓库的验收测试会往其中
写入真实训练产物（检查点、上传张量），必须落在可管理的 .work/ 下（设计 §9 约定）。

行为：
- 未显式指定 --basetemp 时，重定向到 .work/pytest/run-<本进程 PID>，
  并行会话（CPU 全量 + GPU 子集同时跑）互不覆盖；
- 每次会话启动时，清理 PID 已不存在的历史 run-* 目录（真实训练产物较大，不留堆积）；
- 显式传入 --basetemp 时尊重调用方（pytest 自身会先清空该目录，注意并行安全）。
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

_BASE = Path(__file__).resolve().parent / ".work" / "pytest"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # 进程存在但属于其他用户
    return True


def pytest_configure(config) -> None:
    _BASE.mkdir(parents=True, exist_ok=True)
    for d in _BASE.glob("run-*"):
        pid = d.name.removeprefix("run-")
        if pid.isdigit() and not _pid_alive(int(pid)):
            shutil.rmtree(d, ignore_errors=True)
    if config.option.basetemp is None:
        config.option.basetemp = str(_BASE / f"run-{os.getpid()}")
