#!/usr/bin/env python3
"""E10 后台实验看门狗（Claude Code Stop hook 调用，2026-11-19 用户要求）。

解决的问题：后台 tmux 实验崩溃时无人发现（如 2026-11-19 13:28 截断评估因漏设
E10_BASE_PATH 三卡全崩，20 分钟后才被人工发现）。

机制：启动实验时在 logs/e10/watch_registry.json 登记一条——
  {"session": "e10_trunc_gpu0",          # tmux 会话名
   "log": "logs/e10/p2_trunc_gpu0.log",  # tee 的日志（相对仓库根）
   "expect": ["artifacts/.../metrics.json", ...],  # 全部出现才算成功完成
   "note": "12 个截断评估 gpu0 队列"}
每次 Claude 停止响应时本脚本运行：
  - expect 文件齐全 → 视为成功，从注册表注销（静默）；
  - 会话存活但日志尾部出现 Traceback → 告警（早发现，不注销）；
  - 会话消失且 expect 不齐 → 告警"疑似崩溃"+ 日志尾部两行（注销）。
有告警时输出 {"systemMessage": ...}（用户可见）+ additionalContext（注入模型上下文）。
注册表不存在或为空 → 静默退出。测试用 E10_WATCH_REGISTRY 指向临时注册表。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REG = Path(os.environ.get("E10_WATCH_REGISTRY") or ROOT / "logs/e10/watch_registry.json")


def main() -> None:
    if not REG.exists():
        return
    try:
        entries = json.loads(REG.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    if not entries:
        return
    alive = set(subprocess.run(["tmux", "ls", "-F", "#{session_name}"],
                               capture_output=True, text=True).stdout.split())
    alerts: list[str] = []
    remaining: list[dict] = []
    for e in entries:
        exp = [ROOT / p for p in e.get("expect", [])]
        if exp and all(p.exists() for p in exp):
            continue  # 成功完成，注销
        log = ROOT / e["log"]
        tail = log.read_text(encoding="utf-8", errors="ignore")[-20000:] if log.exists() else ""
        sess, note = e["session"], e.get("note", "")
        if e["session"] in alive:
            remaining.append(e)
            if "Traceback" in tail:
                alerts.append(f"[看门狗] 会话 {sess} 仍在运行但日志出现 Traceback（{note}，{e['log']}）")
            continue
        last = [l.strip() for l in tail.splitlines() if l.strip()][-2:]
        alerts.append(f"[看门狗] 会话 {sess} 已消失且产物不齐，疑似崩溃（{note}，日志 {e['log']}；"
                      f"尾部: {' | '.join(last) if last else '无输出'}）")
    REG.write_text(json.dumps(remaining, ensure_ascii=False, indent=1), encoding="utf-8")
    if alerts:
        msg = "\n".join(alerts)[:1500]
        print(json.dumps({"systemMessage": msg,
                          "hookSpecificOutput": {"hookEventName": "Stop",
                                                 "additionalContext": msg}},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
