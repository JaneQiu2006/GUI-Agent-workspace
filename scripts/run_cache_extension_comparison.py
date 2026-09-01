from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = "/data2/home/models/Qwen3.8-27B"
DEFAULT_TEST_JSON = "data/androidcontrol_mini/test.json"


@dataclass(frozen=True)
class CacheComparisonExperiment:
    experiment_id: str
    label: str
    page_cache_mode: str = "off"
    page_cache_similarity: str = "tile"
    page_cache_scope: str = "trajectory"
    notes: str = ""


EXPERIMENTS: Tuple[CacheComparisonExperiment, ...] = (
    CacheComparisonExperiment(
        experiment_id="E11_baseline",
        label="E11 baseline",
        page_cache_mode="off",
        notes="Reference E11: max_new_tokens=48, visual_token_mode=aggressive_reduce, no cache.",
    ),
    CacheComparisonExperiment(
        experiment_id="E11_page_baseline",
        label="E11 + Page-level baseline",
        page_cache_mode="inputs",
        page_cache_similarity="exact",
        notes="Exact page/processor-input cache only; no near-page or patch candidate gating.",
    ),
    CacheComparisonExperiment(
        experiment_id="E11_patch_extension",
        label="E11 + Page-level baseline + Patch-level extension",
        page_cache_mode="inputs",
        page_cache_similarity="tile",
        notes="Exact processor-input cache plus tile diff, changed bbox, stable tile hashes, and patch risk gating.",
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare E11 baseline, Page-level baseline, and Patch-level extension with eval + TTFT profile"
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test_json", default=DEFAULT_TEST_JSON)
    parser.add_argument("--data_dir", help="Passed through to eval/profile when set")
    parser.add_argument("--output_root", default="results/cache_extension_comparison")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", help="Sets CUDA_VISIBLE_DEVICES for child commands")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attn_implementation")
    parser.add_argument("--point_tolerance", type=float, default=100.0)
    parser.add_argument("--profile_limit", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--generation_profile_mode", default="manual_greedy", choices=("generate", "manual_greedy"))
    parser.add_argument("--page_cache_max_entries", type=int, default=128)
    parser.add_argument("--page_cache_near_dhash_threshold", type=int, default=4)
    parser.add_argument("--page_cache_near_tile_threshold", type=float, default=0.98)
    parser.add_argument("--page_cache_patch_tile_threshold", type=float, default=0.90)
    parser.add_argument("--page_cache_patch_max_changed_area_ratio", type=float, default=0.25)
    parser.add_argument(
        "--page_cache_patch_critical_region",
        action="append",
        default=[],
        help="Normalized bbox left,top,right,bottom; repeated values are passed to the patch extension run",
    )
    parser.add_argument("--page_cache_tile_rows", type=int, default=8)
    parser.add_argument("--page_cache_tile_cols", type=int, default=16)
    parser.add_argument("--page_cache_ignored_top_ratio", type=float, default=0.0)
    parser.add_argument("--page_cache_ignored_bottom_ratio", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true", help="Skip experiments already marked success")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_root = resolve_output_root(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if args.gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpus

    comparison: Dict[str, Any] = {
        "created_at": utc_now(),
        "e11_config": {
            "max_new_tokens": 48,
            "visual_token_mode": "aggressive_reduce",
            "batch_size": 1,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
        },
        "profile": {
            "generation_profile_mode": args.generation_profile_mode,
            "profile_limit": args.profile_limit,
            "warmup": args.warmup,
        },
        "experiments": {},
    }

    exit_code = 0
    for experiment in EXPERIMENTS:
        out_dir, skipped = choose_output_dir(output_root, experiment.experiment_id, args.resume)
        if skipped:
            print(f"SKIP {experiment.experiment_id} existing_success={out_dir}", flush=True)
        elif args.dry_run:
            print(f"DRY_RUN {experiment.experiment_id} out_dir={out_dir}", flush=True)
        else:
            status = run_experiment(args, experiment, out_dir, env)
            if status != "success":
                exit_code = 1
        comparison["experiments"][experiment.experiment_id] = summarize_experiment(out_dir, experiment)

    comparison["summary_table"] = build_summary_table(comparison["experiments"])
    write_json(output_root / "comparison_summary.json", comparison)
    print("CACHE_EXTENSION_COMPARISON_DONE " + json.dumps(comparison["summary_table"], ensure_ascii=False), flush=True)
    print(f"CACHE_EXTENSION_COMPARISON_OUTPUT {output_root / 'comparison_summary.json'}", flush=True)
    return exit_code


def run_experiment(
    args: argparse.Namespace,
    experiment: CacheComparisonExperiment,
    out_dir: Path,
    env: Dict[str, str],
) -> str:
    out_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "experiment": asdict(experiment),
        "status": "running",
        "created_at": utc_now(),
        "ended_at": None,
        "commands": [],
        "steps": [],
        "environment": selected_environment(env),
        "notes": [experiment.notes],
    }
    write_json(out_dir / "run_metadata.json", metadata)

    stdout_log = out_dir / "stdout.log"
    stderr_log = out_dir / "stderr.log"
    started = time.perf_counter()
    commands = build_commands(args, experiment, out_dir)
    metadata["commands"] = [{"name": name, "command": format_command(command)} for name, command in commands]
    write_json(out_dir / "run_metadata.json", metadata)

    for name, command in commands:
        return_code = run_logged_command(name, command, env, stdout_log, stderr_log)
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
    metadata["summary"] = summarize_experiment(out_dir, experiment)
    write_json(out_dir / "run_metadata.json", metadata)
    print(f"EXPERIMENT_DONE {experiment.experiment_id} out_dir={out_dir}", flush=True)
    return "success"


def build_commands(
    args: argparse.Namespace,
    experiment: CacheComparisonExperiment,
    out_dir: Path,
) -> List[Tuple[str, List[str]]]:
    common = [
        "--model_path",
        args.model_path,
        "--test_json",
        args.test_json,
        "--output",
        "",
        "--max_new_tokens",
        "48",
        "--device",
        args.device,
        "--device_map",
        args.device_map,
        "--dtype",
        args.dtype,
        "--batch_size",
        "1",
        "--visual_token_mode",
        "aggressive_reduce",
        "--point_tolerance",
        str(args.point_tolerance),
        "--page_cache_mode",
        experiment.page_cache_mode,
        "--page_cache_scope",
        experiment.page_cache_scope,
        "--page_cache_similarity",
        experiment.page_cache_similarity,
        "--page_cache_max_entries",
        str(args.page_cache_max_entries),
        "--page_cache_near_dhash_threshold",
        str(args.page_cache_near_dhash_threshold),
        "--page_cache_near_tile_threshold",
        str(args.page_cache_near_tile_threshold),
        "--page_cache_patch_tile_threshold",
        str(args.page_cache_patch_tile_threshold),
        "--page_cache_patch_max_changed_area_ratio",
        str(args.page_cache_patch_max_changed_area_ratio),
        "--page_cache_tile_rows",
        str(args.page_cache_tile_rows),
        "--page_cache_tile_cols",
        str(args.page_cache_tile_cols),
        "--page_cache_ignored_top_ratio",
        str(args.page_cache_ignored_top_ratio),
        "--page_cache_ignored_bottom_ratio",
        str(args.page_cache_ignored_bottom_ratio),
    ]
    if args.data_dir:
        common.extend(["--data_dir", args.data_dir])
    if args.attn_implementation:
        common.extend(["--attn_implementation", args.attn_implementation])
    if experiment.experiment_id == "E11_patch_extension":
        for region in args.page_cache_patch_critical_region:
            common.extend(["--page_cache_patch_critical_region", region])

    compile_command = [
        args.python,
        "-m",
        "py_compile",
        "test_framework/cache_fingerprint.py",
        "test_framework/cache_store.py",
        "test_framework/cache_inference.py",
        "test_framework/hf_gui_baseline.py",
        "scripts/androidcontrol_actions.py",
        "scripts/eval_androidcontrol.py",
        "scripts/profile_androidcontrol.py",
        "scripts/run_cache_extension_comparison.py",
    ]
    eval_args = replace_output_arg(common, out_dir / "eval.json")
    profile_args = replace_output_arg(common, out_dir / "profile.json")
    profile_args.extend(
        [
            "--limit",
            str(args.profile_limit),
            "--warmup",
            str(args.warmup),
            "--generation_profile_mode",
            args.generation_profile_mode,
        ]
    )
    return [
        ("py_compile", compile_command),
        ("eval", [args.python, "scripts/eval_androidcontrol.py", *eval_args]),
        ("profile_ttft", [args.python, "scripts/profile_androidcontrol.py", *profile_args]),
    ]


def replace_output_arg(args: List[str], output_path: Path) -> List[str]:
    result = list(args)
    output_index = result.index("--output") + 1
    result[output_index] = str(output_path)
    return result


def run_logged_command(
    name: str,
    command: List[str],
    env: Dict[str, str],
    stdout_log: Path,
    stderr_log: Path,
) -> int:
    print(f"RUN_STEP {name} {format_command(command)}", flush=True)
    with stdout_log.open("a", encoding="utf-8") as stdout_file, stderr_log.open("a", encoding="utf-8") as stderr_file:
        stdout_file.write(f"\n===== {utc_now()} {name}: {format_command(command)} =====\n")
        stderr_file.write(f"\n===== {utc_now()} {name}: {format_command(command)} =====\n")
        result = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
        )
        return_code = result.returncode
    print(f"RUN_STEP_DONE {name} return_code={return_code}", flush=True)
    return return_code


def summarize_experiment(out_dir: Path, experiment: CacheComparisonExperiment) -> Dict[str, Any]:
    eval_data = read_json(out_dir / "eval.json")
    profile_data = read_json(out_dir / "profile.json")
    eval_metrics = eval_data.get("metrics", {})
    profile_summary = profile_data.get("summary", {})
    return {
        "label": experiment.label,
        "out_dir": str(out_dir),
        "status": read_json(out_dir / "run_metadata.json").get("status", "missing"),
        "eval": summarize_eval_metrics(eval_metrics),
        "profile": summarize_profile_metrics(profile_summary),
        "cache": {
            "eval": eval_metrics.get("cache", {}),
            "profile": profile_summary.get("cache", {}),
        },
    }


def summarize_eval_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    views = metrics.get("views", {})
    health = metrics.get("output_health", {})
    return {
        "strict": summarize_view(views.get("strict", {})),
        "gui_only": summarize_view(views.get("gui_only", {})),
        "transition_or_noop": summarize_view(views.get("transition_or_noop", {})),
        "open_app": summarize_view(views.get("open_app", {})),
        "wait": summarize_view(views.get("wait", {})),
        "by_gt_type": {
            key: summarize_view(value)
            for key, value in sorted((views.get("by_gt_type") or {}).items())
        },
        "output_health": {
            "pred_unknown": health.get("pred_unknown"),
            "contains_think_end": health.get("contains_think_end"),
            "hit_max_new_tokens": health.get("hit_max_new_tokens"),
            "json_enclosed": health.get("json_enclosed"),
            "avg_input_tokens": health.get("avg_input_tokens"),
            "avg_output_tokens": health.get("avg_output_tokens"),
            "min_output_tokens": health.get("min_output_tokens"),
            "median_output_tokens": health.get("median_output_tokens"),
            "max_output_tokens": health.get("max_output_tokens"),
        },
        "avg_latency_seconds": metrics.get("avg_latency_seconds"),
        "median_latency_seconds": metrics.get("median_latency_seconds"),
        "wall_clock_seconds": metrics.get("wall_clock_seconds"),
        "samples_per_second": metrics.get("samples_per_second"),
        "peak_gpu_memory_gb": metrics.get("peak_gpu_memory_gb"),
    }


def summarize_view(view: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "num_steps": view.get("num_steps"),
        "type_accuracy": view.get("type_accuracy"),
        "step_success_rate": view.get("step_success_rate"),
        "trajectory_success_rate": view.get("trajectory_success_rate"),
        "avg_latency_seconds": view.get("avg_latency_seconds"),
    }


def summarize_profile_metrics(summary: Dict[str, Any]) -> Dict[str, Any]:
    timings = summary.get("timings_seconds", {})
    keys = [
        "build_prompt_seconds",
        "apply_chat_template_seconds",
        "vision_preprocess_seconds",
        "processor_encode_seconds",
        "input_to_device_seconds",
        "prefill_seconds",
        "ttft_seconds",
        "decode_loop_seconds",
        "generate_seconds",
        "decode_seconds",
        "postprocess_seconds",
        "total_seconds",
    ]
    return {
        "primary_timings_seconds": {
            key: timings.get(key)
            for key in keys
            if key in timings
        },
        "avg_input_tokens": summary.get("avg_input_tokens"),
        "avg_output_tokens": summary.get("avg_output_tokens"),
        "effective_batch_size": summary.get("effective_batch_size"),
        "samples_per_second": summary.get("samples_per_second"),
        "output_tokens_per_second": summary.get("output_tokens_per_second"),
        "final_memory": summary.get("final_memory"),
    }


def build_summary_table(experiments: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for experiment_id, data in experiments.items():
        eval_summary = data.get("eval", {})
        profile = data.get("profile", {}).get("primary_timings_seconds", {})
        cache = data.get("cache", {}).get("profile", {})
        gui_only = eval_summary.get("gui_only", {})
        health = eval_summary.get("output_health", {})
        rows.append(
            {
                "experiment_id": experiment_id,
                "label": data.get("label"),
                "status": data.get("status"),
                "gui_only_type_accuracy": gui_only.get("type_accuracy"),
                "gui_only_step_success_rate": gui_only.get("step_success_rate"),
                "pred_unknown": health.get("pred_unknown"),
                "hit_max_new_tokens": health.get("hit_max_new_tokens"),
                "eval_avg_latency_seconds": eval_summary.get("avg_latency_seconds"),
                "profile_total_mean": mean_value(profile.get("total_seconds")),
                "profile_generate_mean": mean_value(profile.get("generate_seconds")),
                "profile_prefill_mean": mean_value(profile.get("prefill_seconds")),
                "profile_ttft_mean": mean_value(profile.get("ttft_seconds")),
                "processor_cache_hit_rate": cache.get("processor_cache_hit_rate"),
                "patch_candidate_rate": cache.get("patch_candidate_rate"),
                "patch_candidate_allowed_rate": cache.get("patch_candidate_allowed_rate"),
            }
        )
    return rows


def mean_value(summary: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(summary, dict):
        return None
    value = summary.get("mean")
    return float(value) if value is not None else None


def choose_output_dir(output_root: Path, experiment_id: str, resume: bool) -> Tuple[Path, bool]:
    base = output_root / experiment_id
    if resume and successful(base):
        return base, True
    if not base.exists():
        return base, False
    for index in range(1, 1000):
        candidate = output_root / f"{experiment_id}_rerun{index}"
        if resume and successful(candidate):
            return candidate, True
        if not candidate.exists():
            return candidate, False
    raise SystemExit(f"Too many rerun directories for {experiment_id}")


def successful(out_dir: Path) -> bool:
    metadata = read_json(out_dir / "run_metadata.json")
    return (
        metadata.get("status") == "success"
        and (out_dir / "eval.json").is_file()
        and (out_dir / "profile.json").is_file()
    )


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


def resolve_output_root(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def format_command(command: Iterable[str]) -> str:
    return shlex.join(str(part) for part in command)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
