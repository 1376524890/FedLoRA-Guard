"""Optional progress messages for interactive experiment launchers."""
import os


def report_progress(label: str, done: int, total: int, seconds: float) -> None:
    if os.environ.get("FEDLORA_PROGRESS") != "1":
        return
    remaining = seconds * (total - done) / done if done else None
    estimate = f"{remaining:.0f} 秒" if remaining is not None else "等待实测"
    print(f"{label} {done}/{total} | 当前工作剩余约 {estimate}", flush=True)


def report_result(message: str) -> None:
    if os.environ.get("FEDLORA_PROGRESS") == "1":
        print(message, flush=True)
