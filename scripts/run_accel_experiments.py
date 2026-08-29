from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = "/data2/home/models/Qwen3.8-27B"
DEFAULT_TEST_JSON = "data/androidcontrol_mini/test.json"
REF_GUI_ONLY_STEP = 0.9189
REF_GUI_ONLY_TYPE = 0.9730


@dataclass
class Experiment:
    experiment_id: str
    direction: str
    max_new_tokens: Optional[int] = None
    dtype: str = "auto"
    attn_implementation: Optional[str] = None
    batch_size: int = 1
    visual_token_mode: str = "default"
    use_selected_cap: bool = False
    use_best_so_far: bool = False
    serving_backend: bool = False


EXPERIMENTS: Dict[str, Experiment] = {
    "E00": Experiment("E00", "calibrated baseline", max_new_tokens=128),
    "E01": Experiment("E01", "decode cap", max_new_tokens=64),
    "E02": Experiment("E02", "decode cap", max_new_tokens=48),
    "E03": Experiment("E03", "decode cap", max_new_tokens=32),
    "E04": Experiment("E04", "attention", attn_implementation="sdpa", use_selected_cap=True),
    "E05": Experiment("E05", "attention", attn_implementation="flash_attention_2", use_selected_cap=True),
    "E06": Experiment("E06", "dtype", dtype="bfloat16", use_selected_cap=True),
    "E07": Experiment("E07", "dtype", dtype="float16", use_selected_cap=True),
    "E08": Experiment("E08", "dtype + attention", dtype="bfloat16", attn_implementation="sdpa", use_selected_cap=True),
    "E09": Experiment("E09", "dtype + attention", dtype="float16", attn_implementation="sdpa", use_selected_cap=True),
    "E10": Experiment("E10", "visual token", visual_token_mode="mild_reduce", use_best_so_far=True),
    "E11": Experiment("E11", "visual token", visual_token_mode="aggressive_reduce", use_best_so_far=True),
    "E12": Experiment("E12", "batch", batch_size=2, use_best_so_far=True),
    "E13": Experiment("E13", "batch", batch_size=4, use_best_so_far=True),
    "E14": Experiment("E14", "serving backend", serving_backend=True, use_best_so_far=True),
}


class RunInterrupted(Exception):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AndroidControl inference acceleration experiments")
    parser.add_argument("--experiments", default="all", help="all, E00,E01, or a range such as E00-E03")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test_json", default=DEFAULT_TEST_JSON)
    parser.add_argument("--output_root", default="results/accel")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", help="Sets CUDA_VISIBLE_DEVICES for child commands. Defaults to inherited env.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--profile_limit", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--point_tolerance", type=float, default=100.0)
    parser.add_argument("--min_pixels", type=int)
    parser.add_argument("--max_pixels", type=int)
    parser.add_argument("--resume", action="store_true", help="Skip experiments already marked success.")
    parser.add_argument("--fail_fast", action="store_true", help="Stop after the first failed experiment.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--serving_command", help="Optional shell command for E14. Placeholders: {out_dir}, {model_path}, {test_json}, {gpus}.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    install_signal_handlers()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if args.gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpus
    selected_gpus = env.get("CUDA_VISIBLE_DEVICES")
    experiment_ids = parse_experiment_selection(args.experiments)
    print(f"RUN_EXPERIMENTS ids={','.join(experiment_ids)} gpus={selected_gpus or '<inherited/none>'}", flush=True)

    completed: Dict[str, Path] = discover_successful_outputs(output_root)
    results: Dict[str, Path] = dict(completed)
    exit_code = 0
    for experiment_id in experiment_ids:
        experiment = EXPERIMENTS[experiment_id]
        out_dir, should_skip = choose_output_dir(output_root, experiment_id, args.resume)
        if should_skip:
            print(f"SKIP {experiment_id} existing_success={out_dir}", flush=True)
            results[experiment_id] = out_dir
            continue
        try:
            effective = resolve_effective_experiment(experiment, results)
            if args.dry_run:
                print(f"DRY_RUN {experiment_id} {json.dumps(asdict(effective), ensure_ascii=False)}", flush=True)
                continue
            status = run_experiment(args, effective, out_dir, env)
            if status == "success":
                results[experiment_id] = out_dir
            elif status == "skipped":
                results[experiment_id] = out_dir
            else:
                exit_code = 1
                if args.fail_fast:
                    break
        except RunInterrupted:
            write_interrupted_metadata(out_dir, experiment)
            return 130
    return exit_code


def install_signal_handlers() -> None:
    def handle_signal(signum: int, _frame: Any) -> None:
        raise RunInterrupted(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, handle_signal)


def parse_experiment_selection(value: str) -> List[str]:
    if value == "all":
        return list(EXPERIMENTS)
    selected: List[str] = []
    for part in value.split(","):
        token = part.strip().upper()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            start_i = int(start[1:])
            end_i = int(end[1:])
            for index in range(start_i, end_i + 1):
                selected.append(f"E{index:02d}")
        else:
            selected.append(token)
    unknown = [experiment_id for experiment_id in selected if experiment_id not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"Unknown experiment ids: {', '.join(unknown)}")
    return selected


def discover_successful_outputs(output_root: Path) -> Dict[str, Path]:
    successful: Dict[str, Path] = {}
    for experiment_id in EXPERIMENTS:
        for out_dir in experiment_dirs(output_root, experiment_id):
            metadata = read_json(out_dir / "run_metadata.json")
            if metadata.get("status") == "success" and has_eval_and_profile(out_dir):
                successful[experiment_id] = out_dir
    return successful


def choose_output_dir(output_root: Path, experiment_id: str, resume: bool) -> Tuple[Path, bool]:
    base = output_root / experiment_id
    if resume:
        for out_dir in experiment_dirs(output_root, experiment_id):
            metadata = read_json(out_dir / "run_metadata.json")
            if metadata.get("status") == "success" and has_eval_and_profile(out_dir):
                return out_dir, True
    if not base.exists():
        return base, False
    for rerun_index in range(1, 1000):
        candidate = output_root / f"{experiment_id}_rerun{rerun_index}"
        if not candidate.exists():
            return candidate, False
    raise SystemExit(f"Too many rerun directories for {experiment_id}")


def experiment_dirs(output_root: Path, experiment_id: str) -> List[Path]:
    paths = [output_root / experiment_id]
    paths.extend(sorted(output_root.glob(f"{experiment_id}_rerun*")))
    return paths


def has_eval_and_profile(out_dir: Path) -> bool:
    return (out_dir / "eval.json").is_file() and (out_dir / "profile.json").is_file()


def resolve_effective_experiment(experiment: Experiment, results: Dict[str, Path]) -> Experiment:
    effective = Experiment(**asdict(experiment))
    if effective.use_selected_cap:
        effective.max_new_tokens = choose_decode_cap(results)
    if effective.use_best_so_far:
        best = choose_best_so_far(results)
        effective.max_new_tokens = best.get("max_new_tokens", 48)
        effective.dtype = best.get("dtype", "auto")
        effective.attn_implementation = best.get("attn_implementation")
    if effective.max_new_tokens is None:
        effective.max_new_tokens = 48
    return effective


def choose_decode_cap(results: Dict[str, Path]) -> int:
    viable_caps = []
    for experiment_id in ("E00", "E01", "E02", "E03"):
        out_dir = results.get(experiment_id)
        if out_dir and is_viable_candidate(out_dir):
            config = read_json(out_dir / "eval.json").get("config", {})
            viable_caps.append(int(config.get("max_new_tokens", 48)))
    return min(viable_caps) if viable_caps else 48


def choose_best_so_far(results: Dict[str, Path]) -> Dict[str, Any]:
    candidates = []
    for out_dir in results.values():
        if not is_viable_candidate(out_dir):
            continue
        eval_data = read_json(out_dir / "eval.json")
        profile_data = read_json(out_dir / "profile.json")
        config = eval_data.get("config", {})
        score = profile_total_mean(profile_data)
        candidates.append((score, config))
    if not candidates:
        return {"max_new_tokens": 48, "dtype": "auto", "attn_implementation": None}
    _, config = sorted(candidates, key=lambda item: item[0])[0]
    return {
        "max_new_tokens": int(config.get("max_new_tokens", 48)),
        "dtype": config.get("dtype", "auto"),
        "attn_implementation": config.get("attn_implementation"),
    }


def is_viable_candidate(out_dir: Path) -> bool:
    data = read_json(out_dir / "eval.json")
    metrics = data.get("metrics", {})
    gui_only = metrics.get("views", {}).get("gui_only", {})
    health = metrics.get("output_health", {})
    return (
        gui_only.get("step_success_rate", 0.0) >= REF_GUI_ONLY_STEP
        and gui_only.get("type_accuracy", 0.0) >= REF_GUI_ONLY_TYPE
        and health.get("pred_unknown", 1) == 0
        and health.get("hit_max_new_tokens", 1) == 0
    )


def profile_total_mean(profile_data: Dict[str, Any]) -> float:
    summary = profile_data.get("summary", {})
    timing = summary.get("timings_seconds", {}).get("total_seconds", {})
    if "mean" in timing:
        return float(timing["mean"])
    if "avg_latency_seconds" in summary:
        return float(summary["avg_latency_seconds"])
    return float("inf")


def run_experiment(args: argparse.Namespace, experiment: Experiment, out_dir: Path, env: Dict[str, str]) -> str:
    out_dir.mkdir(parents=True, exist_ok=False)
    metadata = initial_metadata(args, experiment, out_dir, env)
    write_json(out_dir / "run_metadata.json", metadata)
    if experiment.serving_backend and not args.serving_command:
        metadata["status"] = "skipped"
        metadata["notes"].append("E14 requires --serving_command; no serving backend command was provided.")
        metadata["ended_at"] = utc_now()
        write_json(out_dir / "run_metadata.json", metadata)
        print(f"SKIP {experiment.experiment_id} serving_command_missing out_dir={out_dir}", flush=True)
        return "skipped"

    stdout_path = out_dir / "stdout.log"
    stderr_path = out_dir / "stderr.log"
    started = time.perf_counter()
    try:
        commands = build_commands(args, experiment, out_dir)
        metadata["commands"] = [
            {"name": name, "command": format_command(command)}
            for name, command in commands
        ]
        write_json(out_dir / "run_metadata.json", metadata)
        for name, command in commands:
            return_code = run_logged_command(name, command, cwd=REPO_ROOT, env=env, stdout_path=stdout_path, stderr_path=stderr_path)
            metadata["steps"].append({"name": name, "return_code": return_code, "ended_at": utc_now()})
            write_json(out_dir / "run_metadata.json", metadata)
            if return_code != 0:
                metadata["status"] = "failed"
                metadata["ended_at"] = utc_now()
                metadata["wall_clock_seconds"] = time.perf_counter() - started
                write_json(out_dir / "run_metadata.json", metadata)
                print(f"EXPERIMENT_FAILED {experiment.experiment_id} step={name} out_dir={out_dir}", flush=True)
                return "failed"
        metadata["status"] = "success"
        metadata["ended_at"] = utc_now()
        metadata["wall_clock_seconds"] = time.perf_counter() - started
        metadata["summary"] = summarize_outputs(out_dir)
        write_json(out_dir / "run_metadata.json", metadata)
        print(f"EXPERIMENT_DONE {experiment.experiment_id} out_dir={out_dir}", flush=True)
        return "success"
    except RunInterrupted:
        metadata["status"] = "interrupted"
        metadata["ended_at"] = utc_now()
        metadata["wall_clock_seconds"] = time.perf_counter() - started
        write_json(out_dir / "run_metadata.json", metadata)
        raise


def build_commands(args: argparse.Namespace, experiment: Experiment, out_dir: Path) -> List[Tuple[str, List[str]]]:
    if experiment.serving_backend and args.serving_command:
        command_text = args.serving_command.format(
            out_dir=str(out_dir),
            model_path=args.model_path,
            test_json=args.test_json,
            gpus=args.gpus or os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        )
        return [("serving", ["bash", "-lc", command_text])]

    common = [
        "--model_path", args.model_path,
        "--test_json", args.test_json,
        "--max_new_tokens", str(experiment.max_new_tokens),
        "--device", args.device,
        "--device_map", args.device_map,
        "--dtype", experiment.dtype,
        "--batch_size", str(experiment.batch_size),
        "--visual_token_mode", experiment.visual_token_mode,
        "--point_tolerance", str(args.point_tolerance),
    ]
    if experiment.attn_implementation:
        common.extend(["--attn_implementation", experiment.attn_implementation])
    if args.min_pixels is not None:
        common.extend(["--min_pixels", str(args.min_pixels)])
    if args.max_pixels is not None:
        common.extend(["--max_pixels", str(args.max_pixels)])

    compile_command = [
        args.python,
        "-m",
        "py_compile",
        "test_framework/phone_prompt.py",
        "test_framework/hf_gui_baseline.py",
        "scripts/androidcontrol_actions.py",
        "scripts/eval_androidcontrol.py",
        "scripts/profile_androidcontrol.py",
        "scripts/profile_single_image.py",
    ]
    eval_command = [
        args.python,
        "scripts/eval_androidcontrol.py",
        *common,
        "--output", str(out_dir / "eval.json"),
    ]
    profile_command = [
        args.python,
        "scripts/profile_androidcontrol.py",
        *common,
        "--output", str(out_dir / "profile.json"),
        "--limit", str(args.profile_limit),
        "--warmup", str(args.warmup),
    ]
    return [("py_compile", compile_command), ("eval", eval_command), ("profile", profile_command)]


def run_logged_command(
    name: str,
    command: List[str],
    cwd: Path,
    env: Dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> int:
    print(f"RUN_STEP {name} {format_command(command)}", flush=True)
    with stdout_path.open("a", encoding="utf-8") as stdout_file, stderr_path.open("a", encoding="utf-8") as stderr_file:
        stdout_file.write(f"\n===== {utc_now()} {name}: {format_command(command)} =====\n")
        stderr_file.write(f"\n===== {utc_now()} {name}: {format_command(command)} =====\n")
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        stdout_thread = threading.Thread(target=tee_stream, args=(process.stdout, stdout_file, sys.stdout), daemon=True)
        stderr_thread = threading.Thread(target=tee_stream, args=(process.stderr, stderr_file, sys.stderr), daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        try:
            return_code = process.wait()
        except (KeyboardInterrupt, RunInterrupted):
            terminate_process(process)
            raise
        stdout_thread.join()
        stderr_thread.join()
    print(f"RUN_STEP_DONE {name} return_code={return_code}", flush=True)
    return return_code


def terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def tee_stream(source: Any, log_file: Any, console: Any) -> None:
    if source is None:
        return
    for line in source:
        log_file.write(line)
        log_file.flush()
        console.write(line)
        console.flush()


def initial_metadata(args: argparse.Namespace, experiment: Experiment, out_dir: Path, env: Dict[str, str]) -> Dict[str, Any]:
    return {
        "experiment": asdict(experiment),
        "out_dir": str(out_dir),
        "status": "running",
        "created_at": utc_now(),
        "ended_at": None,
        "commands": [],
        "steps": [],
        "notes": [],
        "launcher_args": vars(args),
        "environment": selected_environment(env),
        "git": git_state(),
        "package_versions": package_versions(args.python, env),
    }


def summarize_outputs(out_dir: Path) -> Dict[str, Any]:
    eval_data = read_json(out_dir / "eval.json")
    profile_data = read_json(out_dir / "profile.json")
    gui_only = eval_data.get("metrics", {}).get("views", {}).get("gui_only", {})
    strict = eval_data.get("metrics", {}).get("views", {}).get("strict", {})
    health = eval_data.get("metrics", {}).get("output_health", {})
    profile_total = profile_data.get("summary", {}).get("timings_seconds", {}).get("total_seconds", {})
    profile_generate = profile_data.get("summary", {}).get("timings_seconds", {}).get("generate_seconds", {})
    return {
        "gui_only": {
            "type_accuracy": gui_only.get("type_accuracy"),
            "step_success_rate": gui_only.get("step_success_rate"),
            "trajectory_success_rate": gui_only.get("trajectory_success_rate"),
        },
        "strict": {
            "type_accuracy": strict.get("type_accuracy"),
            "step_success_rate": strict.get("step_success_rate"),
            "trajectory_success_rate": strict.get("trajectory_success_rate"),
        },
        "output_health": health,
        "profile_total_mean": profile_total.get("mean"),
        "profile_generate_mean": profile_generate.get("mean"),
    }


def selected_environment(env: Dict[str, str]) -> Dict[str, Optional[str]]:
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "ASCEND_RT_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "PYTHONPATH",
        "CONDA_DEFAULT_ENV",
        "VIRTUAL_ENV",
    ]
    return {key: env.get(key) for key in keys}


def git_state() -> Dict[str, str]:
    return {
        "head": run_quiet(["git", "rev-parse", "HEAD"]),
        "status_short": run_quiet(["git", "status", "--short"]),
        "diff_stat": run_quiet(["git", "diff", "--stat"]),
    }


def package_versions(python_exe: str, env: Dict[str, str]) -> Dict[str, Any]:
    code = (
        "import importlib.metadata as m, json, platform; "
        "names=['torch','transformers','qwen-vl-utils','vllm','sglang']; "
        "out={'python': platform.python_version()}; "
        "\nfor n in names:\n"
        "    try: out[n]=m.version(n)\n"
        "    except Exception as e: out[n]=None\n"
        "print(json.dumps(out))"
    )
    text = run_quiet([python_exe, "-c", code], env=env or os.environ.copy())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": text}


def run_quiet(command: List[str], env: Optional[Dict[str, str]] = None) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as exc:
        return f"{type(exc).__name__}: {exc}"
    return result.stdout.strip()


def write_interrupted_metadata(out_dir: Path, experiment: Experiment) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "run_metadata.json"
    metadata = read_json(metadata_path)
    if not metadata:
        metadata = {"experiment": asdict(experiment), "created_at": utc_now()}
    metadata["status"] = "interrupted"
    metadata["ended_at"] = utc_now()
    write_json(metadata_path, metadata)


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def format_command(command: Iterable[str]) -> str:
    return shlex.join(str(part) for part in command)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
