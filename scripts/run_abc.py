"""Interactive orchestration for the ABC engineering experiments."""
import argparse
import getpass
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from collections import deque
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import TextIO


RUNS = Path("artifacts/runs")
OUTPUTS = Path("artifacts/abc")
CLEAN_CONFIG = "configs/clean_duration_validation.yaml"
ATTACK_CONFIG = "configs/badnet_duration_validation.yaml"
DATA = Path("data/processed/dolly_smoke_len256_200/records.jsonl")
ATTACK_VIEW = Path("data/processed/dolly_badnet_g4_smoke")
METHODS = ["norm", "cosine", "cbdf", "foolsgold", "safe_fedllm", "logistic"]
ROUND_SECONDS = (175, 208)
POLL_SECONDS = 5


@dataclass
class Stage:
    label: str
    args: list[str]
    result: Path | None = None
    progress: Path | None = None
    rounds: int = 0
    generation_calls: int = 0
    forward_calls: int = 0
    number: int | None = None  # planned position, retained across a resume


def ask(label: str, default: str = "", *, valid=None) -> str:
    while True:
        hint = f" [{default}]" if default else ""
        value = input(f"{label}{hint}：").strip() or default
        if value and (valid is None or valid(value)):
            return value
        print("请输入有效值。")


def yes(label: str, default: str = "n") -> bool:
    return ask(f"{label} (y/n)", default, valid=lambda x: x.lower() in {"y", "n"}).lower() == "y"


def json_read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def json_write(path: Path, data: dict, *, overwrite: bool = False) -> None:
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def elapsed(seconds: float) -> str:
    minutes, seconds = divmod(max(0, round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def stage(module: str, label: str, *args: str, **details) -> Stage:
    return Stage(label, [sys.executable, "-u", "-m", module, *map(str, args)], **details)


def percent(value) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def points(value) -> str:
    return "—" if value is None else f"{100 * value:+.2f} pp"


def tail_log(stream: TextIO, last_line: str) -> str:
    for line in stream:
        line = line.strip()
        if line.startswith("评价结果"):
            print(f"  {line}", flush=True)
        if line.startswith(("评价进度", "重训进度", "分析进度")):
            last_line = line
    return last_line


def stage_summary(path: Path) -> list[str]:
    result = json_read(path)
    if result.get("status") == "completed" and "final_global_adapter_digest" in result:
        return [f"已保存 {result['rounds']} 轮 | 最终 adapter {result['final_global_adapter_digest'][:12]}"]
    if result.get("scope") == "paired_duration_diagnostic":
        return [
            f"第 {row['round']} 轮 | attack ASR {percent(row['attack']['asr']['estimate'])}"
            f" | clean ASR {percent(row['clean']['asr']['estimate'])}"
            f" | 差值 {points(row['asr_delta'])}"
            f" | 目标 logp 差 {row['target_logp_delta']['triggered']:+.4f}"
            f" | 累计曝光 {row['cumulative_poison_exposures']}"
            for row in result["rounds"]
        ]
    if result.get("scope") == "conditional_on_received_global_and_other_uploads":
        return [f"联合 ASR 影响 {points(result['pair_asr_effect'])}"
                f" | 交互项 {points(result['interaction']['gamma_probability'])}"]
    if result.get("mode") in {"baseline", "components"}:
        rows = result.get("metrics", result.get("comparison", {}))
        return [f"{name:<24} 召回 {percent(row['test']['recall']):>6}"
                f" | 误报 {percent(row['test']['fpr']):>6}"
                f" | AP {percent(row['test']['average_precision']):>6}"
                for name, row in rows.items()] or ["当前分区没有可比较的组件样本，详见 JSON。"]
    if "classification" in result:
        row = result["classification"]
        return [f"{result['returned_model'] or result['requested_model']} | {row['label']}"
                f" | 分数 {row['score']:.3f} | 请求 {result['seconds']:.1f} 秒"]
    return []


def run_stage(task: Stage, directory: Path, index: int, total: int, env: dict) -> float:
    log_path = next_log_path(directory, task.number or index)
    started = monotonic()
    prefix = f"[{index:02d}/{total:02d}]"
    print(f"\n{prefix} 开始 {task.label}\n  日志 {log_path}", flush=True)
    with log_path.open("x", encoding="utf-8") as log, log_path.open(encoding="utf-8") as reader:
        process = subprocess.Popen(task.args, stdout=log, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
        last_line = ""
        try:
            while True:
                last_line = tail_log(reader, last_line)
                wall = elapsed(monotonic() - started)
                if task.progress and task.progress.is_file():
                    progress = json_read(task.progress)
                    done, count = progress["completed_tasks"], progress["total_tasks"]
                    bar = int(20 * done / count) if count else 0
                    estimate = progress["estimated_remaining_seconds"]
                    eta = elapsed(estimate) if estimate is not None else "等待实测"
                    client = progress["client_id"] or "—"
                    step = progress["local_step"]
                    gpus = " ".join(f"{gpu['gpu_id']}号卡 {gpu['utilization_percent']}%"
                                    for gpu in progress["gpu"])
                    print(f"  {prefix} [{'#' * bar}{'.' * (20 - bar)}] {done}/{count} 客户端任务"
                          f" | 第 {progress['round']} 轮 | {client} 步 {step} | {gpus}"
                          f" | 已用 {wall} | 本阶段剩余 {eta}",
                          flush=True)
                else:
                    message = f" | {last_line}" if last_line else " | 等待首个可计数进度"
                    print(f"  {prefix} 已用 {wall}{message}", flush=True)
                # Timed wait avoids busy polling and preserves the child's actual exit code.
                try:
                    code = process.wait(timeout=POLL_SECONDS)
                except subprocess.TimeoutExpired:
                    continue
                break
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
    if code:
        with log_path.open(encoding="utf-8", errors="replace") as log:
            for line in deque(log, maxlen=12):
                print(f"  {line.rstrip()}", file=sys.stderr)
        raise subprocess.CalledProcessError(code, task.args)
    seconds = monotonic() - started
    print(f"{prefix} 完成 {task.label} | 用时 {elapsed(seconds)}", flush=True)
    if task.result:
        for line in stage_summary(task.result):
            print(f"  {line}")
        print(f"  结果 {task.result}")
    return seconds


def training_tasks(prefix: str, seeds: list[int], tasks: list[Stage]) -> dict[int, dict[str, str]]:
    paths = {}
    for seed in seeds:
        paths[seed] = {}
        for kind, config in (("clean", CLEAN_CONFIG), ("attack", ATTACK_CONFIG)):
            run_id = f"{prefix}_{kind}_s{seed}"
            path = RUNS / run_id
            if path.exists():
                raise FileExistsError(path)
            paths[seed][kind] = str(path)
            scenario = "clean_duration_validation" if kind == "clean" else "badnet_duration_validation"
            tasks.append(stage("src.cli", f"{kind} 训练 · 种子 {seed}", "run", "--config", config,
                               "--seed", seed, "--scenario", scenario, "--run-id", run_id,
                               progress=path / "progress.json", result=path / "run_summary.json", rounds=6))
    return paths


def replay_tasks(manifest: Path, spec: dict, output: Path, tasks: list[Stage], *,
                 methods: list[str], fpr: str, window: str, impact: Path | None = None) -> dict:
    if set(spec) - {"fit", "calibration", "test"} or not all(spec.get(key) for key in ("fit", "calibration", "test")):
        raise ValueError("基线清单必须提供 fit、calibration 和 test 三个分区。")
    component_manifest = output / "component-splits.json"
    for mode, name in (("baseline", "现有基线"), ("components", "CBDF-LoRA 独立组件")):
        result = output / f"{mode}.json"
        args = ["--manifest", manifest if mode == "baseline" else component_manifest,
                "--mode", mode, "--fpr", fpr,
                "--history-window", window, "--output", result]
        args.extend(("--methods", *methods) if mode == "baseline" else ())
        if impact is not None:
            args.extend(("--impact", impact))
        tasks.append(stage("src.detect.baseline_cli", name, *args, result=result))
    return {component_manifest: {key: spec[key] for key in ("calibration", "test")}}


def llm_task(output: Path) -> tuple[Stage, dict, str]:
    provider = ask("供应商", "kimi", valid=lambda x: x in {"kimi", "glm"})
    run = ask("已完成的公开数据实验目录", valid=lambda x: (Path(x) / "run_summary.json").is_file())
    info = json_read(Path(run) / "run_summary.json")
    if info["status"] != "completed":
        raise ValueError("LLM 评价需要已完成的运行。")
    index = ask("通信轮次", str(info["rounds"]), valid=lambda x: x.isdecimal() and 1 <= int(x) <= info["rounds"])
    client = ask("客户端", "client_02", valid=lambda x: re.fullmatch(r"client_[0-9]{2}", x))
    base_url = ask("OpenAI 兼容 base_url", valid=lambda x: bool(re.fullmatch(r"https?://[^\s/@?#]+(?:/[^\s?#]*)?", x)))
    model = ask("模型 ID")
    env_name = "MOONSHOT_API_KEY" if provider == "kimi" else "ZHIPUAI_API_KEY"

    config = output / "provider.json"
    result = output / "llm.json"
    task = stage("src.online.classify_cli", f"在线 LLM · {provider}", "--config", config,
                 "--provider", provider, "--run", run, "--round", index, "--client", client,
                 "--output", result, result=result)
    return task, {"providers": {provider: {"base_url": base_url, "model_id": model,
                                           "api_key_env": env_name}}}, env_name


def completed_labels(directory: Path) -> set[str]:
    """Stages recorded as finished, keyed by label; an interrupted stage stays unfinished."""
    summary = directory / "summary.json"
    return {row["label"] for row in json_read(summary)["completed_stages"]} if summary.is_file() else set()


def resume_tasks(directory: Path) -> tuple[list[Stage], list[dict]]:
    """Rebuild the unfinished stages of an interrupted batch from its own plan.json.

    Commands are replayed verbatim, so completed training runs are reused rather than
    retrained. Each task keeps its original plan position, so its log keeps the same
    number. A stage is skipped when the summary lists it or when the stage's own result
    file exists and parses: every stage writes that file with an atomic rename as its
    last action, so its presence means the stage finished before the summary was lost.
    Partial artifacts other than `result` never satisfy this check.
    """
    plan = json_read(directory / "plan.json")
    done = completed_labels(directory)
    tasks, carried = [], []
    for number, row in enumerate(plan["stages"], 1):
        if row["label"] in done:
            continue
        result = Path(row["result"]) if row["result"] else None
        if result is not None and result.exists():
            try:
                json_read(result)
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"阶段「{row['label']}」没有记为完成，结果也无法解析：{result}\n"
                    f"请确认该产物后移动或删除它再续跑。（{error}）") from error
            carried.append({"label": row["label"], "result": str(result),
                            "recovered_from_result": True})
            continue
        tasks.append(Stage(row["label"], row["command"], result=result, number=number))
    return tasks, carried


def next_log_path(directory: Path, number: int) -> Path:
    """Never overwrite an earlier attempt's log: `07.log`, then `07.1.log`, `07.2.log`."""
    path, attempt = directory / f"{number:02d}.log", 0
    while path.exists():
        attempt += 1
        path = directory / f"{number:02d}.{attempt}.log"
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="Show the interactive plan without running any stage")
    parser.add_argument("--resume", type=Path,
                        help="Re-run only the unfinished stages of an interrupted batch directory")
    args = parser.parse_args()
    if args.resume is not None:
        output = args.resume.resolve()
        if not (output / "plan.json").is_file():
            raise ValueError(f"不是可续跑的批次目录（缺少 plan.json）：{output}")
        tasks, carried = resume_tasks(output)
        done = len(completed_labels(output)) + len(carried)
        print(f"\n续跑批次 · {output.name}\n  沿用 {done} 个已完成阶段")
        for row in carried:
            print(f"  --  {row['label']}（结果已存在，从结果文件补记）")
        for index, task in enumerate(tasks, 1):
            print(f"  {index:02d}  {task.label}")
        if args.plan:
            print("计划结束，未启动实验。")
            return 0
        if not tasks:
            # Nothing left to run, but the summary may still predate stages that were
            # started by hand. Rewrite it from the plan order without running anything.
            print("所有阶段均已完成，仅补记汇总。")
            return execute(output, [], carried, {})
        if not yes("启动上述任务"):
            print("计划结束，未启动实验。")
            return 0
        return execute(output, tasks, carried, {})
    print("FedLoRA-Guard · ABC 实验\n"
          "  1  六轮 clean / attack 配对诊断\n"
          "  2  ABC 工程验证：训练、诊断、同轮反事实、基线、组件\n"
          "  3  回放已有轨迹：基线与组件\n"
          "  4  在线 LLM：评价一个已保存的客户端上传\n"
          "  0  退出")
    choice = ask("选择实验", "1", valid=lambda x: x in {"0", "1", "2", "3", "4"})
    if choice == "0":
        return 0
    prefix = ask("运行批次名", datetime.now().strftime("abc_%Y%m%d_%H%M%S"),
                 valid=lambda x: re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", x))
    output = OUTPUTS / prefix
    if output.exists():
        raise FileExistsError(output)
    tasks, extra_files = [], {}
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "FEDLORA_PROGRESS": "1"}
    methods = METHODS.copy()
    if choice in {"2", "3"}:
        if yes("加入 LightGBM"):
            methods.append("lightgbm")
        fpr = ask("校准目标误报率（小数）", "0.05",
                  valid=lambda x: re.fullmatch(r"(?:0(?:\.\d+)?|\.\d+)", x) and 0 <= float(x) < 1)
        window = ask("历史窗口（轮）", "3", valid=lambda x: x.isdecimal() and 2 <= int(x) <= 6)
    if choice in {"1", "2"}:
        if not DATA.is_file():
            tasks.append(stage("src.cli", "准备 Dolly 数据", "data", "prepare-smoke",
                               "--config", "configs/engineering_smoke.yaml"))
        attack_files = [(ATTACK_VIEW / name).is_file() for name in ("records.jsonl", "manifest.json")]
        if not all(attack_files):
            if ATTACK_VIEW.exists():
                raise ValueError(f"投毒数据目录不完整：{ATTACK_VIEW}")
            tasks.append(stage("src.attacks.prepare", "准备已有 G4 工程数据视图", "--config", ATTACK_CONFIG,
                               "--source", DATA, "--output", ATTACK_VIEW, "--condition", "G4", "--malicious-seed", "17"))
        seeds = [int(ask("训练种子", "17", valid=lambda x: x in {"17", "42", "2026"}))] \
            if choice == "1" else [17, 42, 2026]
        runs = training_tasks(prefix, seeds, tasks)
        selected = seeds[0]
        rounds = [int(x) for x in ask("评价轮次（空格分隔）", "2 4 6",
                  valid=lambda x: bool(re.fullmatch(r"[0-6](?: +[0-6])*", x))).split()]
        rounds = sorted(set(rounds))
        result = output / f"duration_s{selected}.json"
        tasks.append(stage("src.duration_validation", f"配对 ASR 与目标概率 · 种子 {selected}",
                           "--attack", runs[selected]["attack"], "--clean", runs[selected]["clean"],
                           "--rounds", *rounds, "--output", result, result=result,
                           generation_calls=512 * len(rounds), forward_calls=1024 * len(rounds)))
        if choice == "2":
            round_index = ask("同轮反事实检查点（1–6）", "6", valid=lambda x: x in "123456" and len(x) == 1)
            members = json_read(ATTACK_VIEW / "manifest.json")["malicious_clients"] if all(attack_files) else [
                "client_02", "client_04", "client_06", "client_08"]
            pair = ask(f"两名客户端（空格分隔，可选 {' '.join(members)}）", " ".join(members[:2]),
                       valid=lambda x: len(x.split()) == 2 and len(set(x.split())) == 2
                       and set(x.split()) <= set(members)).split()
            impact_dir = output / "impact"
            impact = impact_dir / "impact.json"
            tasks.append(stage("src.counterfactual_round", "同轮四分支反事实 · 种子 2026", "--run",
                               runs[2026]["attack"], "--round", round_index, "--clients", *pair,
                               "--output-dir", impact_dir, result=impact, generation_calls=1024))
            manifest = output / "splits.json"
            extra_files[manifest] = {split: [{"run": run, "family": f"s{seed}"} for run in runs[seed].values()]
                                     for split, seed in zip(("fit", "calibration", "test"), seeds, strict=True)}
            extra_files.update(replay_tasks(manifest, extra_files[manifest], output, tasks,
                                            methods=methods, fpr=fpr, window=window, impact=impact))
    elif choice == "3":
        manifest = Path(ask("轨迹分区 JSON 路径", valid=lambda x: Path(x).is_file()))
        spec = json_read(manifest)
        manifest = output / "splits.json"
        extra_files[manifest] = spec
        extra_files.update(replay_tasks(manifest, spec, output, tasks, methods=methods, fpr=fpr, window=window))
    else:
        task, config, key_name = llm_task(output)
        tasks.append(task)
        extra_files[output / "provider.json"] = config

    total_rounds = sum(task.rounds for task in tasks)
    generations = sum(task.generation_calls for task in tasks)
    forwards = sum(task.forward_calls for task in tasks)
    print(f"\n运行计划 · {prefix}\n  输出目录 {output}")
    for index, task in enumerate(tasks, 1):
        print(f"  {index:02d}  {task.label}")
    if total_rounds:
        print(f"  训练 {total_rounds} 轮，历史实测估算 {elapsed(total_rounds * ROUND_SECONDS[0])}"
              f"–{elapsed(total_rounds * ROUND_SECONDS[1])}，不含启动、评价和额外补训")
        print(f"  逐步检查点约 {total_rounds * 8} GB，另计轮次产物")
    if generations or forwards:
        print(f"  评价 {generations:,} 次生成，{forwards:,} 次目标概率前向；"
              "耗时在取得真实样本速率后显示")
    if args.plan or not yes("启动上述任务"):
        print("计划结束，未启动实验。")
        return 0
    if choice == "4":
        key = env.get(key_name) or getpass.getpass(f"{key_name}（隐藏输入）：")
        if not key.strip():
            raise ValueError("API 密钥为空。")
        env[key_name] = key
    if "lightgbm" in methods and importlib.util.find_spec("lightgbm") is None:
        raise ValueError("当前 Python 环境缺少 LightGBM，请安装后再启动。")
    if total_rounds:
        if shutil.which("nvidia-smi") is None:
            raise ValueError("当前服务器找不到 nvidia-smi。")
        disk_path = next(path for path in (RUNS, RUNS.parent, Path.cwd()) if path.is_dir())
        required = total_rounds * 8_000_000_000
        free = shutil.disk_usage(disk_path).free
        print(f"  训练磁盘可用 {free / 1e9:.1f} GB | 逐步检查点估算 {required / 1e9:.1f} GB")
        if free < required:
            raise ValueError("训练目录所在磁盘的可用空间低于逐步检查点估算。")
    output.mkdir(parents=True, exist_ok=False)
    for path, data in extra_files.items():
        json_write(path, data)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True)
    json_write(output / "plan.json", {"batch": prefix, "started_at": datetime.now().astimezone().isoformat(),
                                     "git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
                                     "stages": [{"label": task.label, "command": task.args,
                                                 "result": str(task.result) if task.result else None}
                                                for task in tasks]})
    return execute(output, tasks, [], env)


def execute(directory: Path, tasks: list[Stage], carried: list[dict],
            env: dict | None = None) -> int:
    """Run the planned stages, keeping a summary of what finished even when interrupted."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "FEDLORA_PROGRESS": "1", **(env or {})}
    completed = []
    started = monotonic()
    try:
        for index, task in enumerate(tasks, 1):
            remaining = sum(item.rounds for item in tasks[index - 1:])
            if remaining:
                print(f"  剩余训练阶段估算 {elapsed(remaining * ROUND_SECONDS[0])}"
                      f"–{elapsed(remaining * ROUND_SECONDS[1])}，评价另计", flush=True)
            seconds = run_stage(task, directory, index, len(tasks), env)
            completed.append({"label": task.label, "seconds": seconds,
                              "result": str(task.result) if task.result else None})
    finally:
        # A resume adds the stages it just finished to whatever the batch already had,
        # keeping the summary in plan order.
        prior = json_read(directory / "summary.json").get("completed_stages", []) \
            if (directory / "summary.json").is_file() else []
        merged = {row["label"]: row for row in prior + carried + completed}
        order = [row["label"] for row in json_read(directory / "plan.json")["stages"]]
        if set(merged) != set(order):
            order = [label for label in order if label in merged] + \
                    [label for label in merged if label not in order]
        json_write(directory / "summary.json", {
            "status": "completed" if set(merged) == set(order) else "stopped",
            "completed_stages": [merged[label] for label in order],
            "planned_stages": len(order), "seconds": monotonic() - started}, overwrite=True)
        print(f"\n已完成 {len(merged)}/{len(order)} 个阶段 | 本次耗时 {elapsed(monotonic() - started)}"
              f" | 汇总 {directory / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (EOFError, KeyboardInterrupt):
        print("\n已停止。", file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"运行终止：{error}", file=sys.stderr)
        sys.exit(error.returncode if isinstance(error, subprocess.CalledProcessError) and error.returncode > 0 else 1)
