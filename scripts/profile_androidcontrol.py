from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_FRAMEWORK = REPO_ROOT / "test_framework"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (TEST_FRAMEWORK, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_androidcontrol import (
    build_metric_views,
    detail_from_result,
    load_samples,
    metric_view_policy,
    resolved_image_path,
    summarize_output_health,
)
from hf_gui_baseline import (
    DEFAULT_MODEL_PATH,
    VISION_TOKEN_MODES,
    gpu_memory_snapshot,
    load_model_and_processor,
    profile_infer_batch,
    profile_infer_one,
    reset_gpu_memory_stats,
)


def summarize_details(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    timing_keys = sorted({key for item in details for key in item["timings"]})
    timing_summary: Dict[str, Any] = {}
    for key in timing_keys:
        values = [float(item["timings"][key]) for item in details if key in item["timings"]]
        timing_summary[key] = {
            "mean": statistics.fmean(values),
            "min": min(values),
            "max": max(values),
            "median": statistics.median(values),
        }
    output_values = [item["output_tokens"] for item in details if item.get("output_tokens") is not None]
    input_values = [item["input_tokens"] for item in details if item.get("input_tokens") is not None]
    views = build_metric_views(details)
    return {
        **views["strict"],
        "primary_metric_view": "gui_only",
        "metric_view_policy": metric_view_policy(),
        "views": views,
        "avg_input_tokens": statistics.fmean(input_values) if input_values else 0.0,
        "avg_output_tokens": statistics.fmean(output_values) if output_values else 0.0,
        "timings_seconds": timing_summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile AndroidControl mini static GUI evaluation")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test_json", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/profile_androidcontrol.json"))
    parser.add_argument("--data_dir", type=Path, help="Defaults to the parent directory of --test_json")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"))
    parser.add_argument("--attn_implementation", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--visual_token_mode", default="default", choices=tuple(VISION_TOKEN_MODES))
    parser.add_argument("--min_pixels", type=int)
    parser.add_argument("--max_pixels", type=int)
    parser.add_argument("--point_tolerance", type=float, default=100.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    records = load_samples(args.test_json)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise SystemExit("No AndroidControl samples found")
    data_dir = args.data_dir or args.test_json.parent

    load_started = time.perf_counter()
    model, processor = load_model_and_processor(
        args.model_path,
        device=args.device,
        device_map=args.device_map or None,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    load_seconds = time.perf_counter() - load_started

    warmup_records = records[: max(0, args.warmup)]
    for start in range(0, len(warmup_records), max(1, args.batch_size)):
        chunk = warmup_records[start:start + max(1, args.batch_size)]
        image_paths = [resolved_image_path(sample, data_dir) for sample in chunk]
        print(f"WARMUP {start + 1}/{len(warmup_records)} batch={len(chunk)}", flush=True)
        if len(chunk) == 1:
            profile_infer_one(
                model,
                processor,
                image_paths[0],
                str(chunk[0]["task"]),
                max_new_tokens=args.max_new_tokens,
                device=args.device,
                visual_token_mode=args.visual_token_mode,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
            )
        else:
            profile_infer_batch(
                model,
                processor,
                [
                    (image_path, str(sample["task"]), None, None)
                    for sample, image_path in zip(chunk, image_paths)
                ],
                max_new_tokens=args.max_new_tokens,
                device=args.device,
                visual_token_mode=args.visual_token_mode,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
            )

    warnings = reset_gpu_memory_stats()
    for warning in warnings:
        print(f"GPU_MEMORY_WARNING {warning}", file=sys.stderr, flush=True)

    details = []
    profile_started = time.perf_counter()
    batch_size = max(1, args.batch_size)
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        image_paths = [resolved_image_path(sample, data_dir) for sample in chunk]
        if len(chunk) == 1:
            results = [
                profile_infer_one(
                    model,
                    processor,
                    image_paths[0],
                    str(chunk[0]["task"]),
                    max_new_tokens=args.max_new_tokens,
                    device=args.device,
                    visual_token_mode=args.visual_token_mode,
                    min_pixels=args.min_pixels,
                    max_pixels=args.max_pixels,
                )
            ]
        else:
            results = profile_infer_batch(
                model,
                processor,
                [
                    (image_path, str(sample["task"]), None, None)
                    for sample, image_path in zip(chunk, image_paths)
                ],
                max_new_tokens=args.max_new_tokens,
                device=args.device,
                visual_token_mode=args.visual_token_mode,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
            )
        for offset, (sample, image_path, result) in enumerate(zip(chunk, image_paths, results), 1):
            detail = detail_from_result(sample, image_path, result, args.point_tolerance)
            detail["timings"] = result.timings
            detail["memory"] = result.memory
            detail["effective_batch_size"] = len(chunk)
            details.append(detail)
            print(
                f"PROFILE_STEP {start + offset}/{len(records)} episode={sample.get('episode_id')} "
                f"step={sample.get('step_id')} type_ok={detail['type_success']} "
                f"step_ok={detail['step_success']} total={result.timings['total_seconds']:.4f}s "
                f"generate={result.timings['generate_seconds']:.4f}s",
                flush=True,
            )
    wall_clock_seconds = time.perf_counter() - profile_started
    output_token_total = sum(
        int(item["output_tokens"])
        for item in details
        if item.get("output_tokens") is not None
    )
    summary = summarize_details(details)
    summary["output_health"] = summarize_output_health(details, args.max_new_tokens)
    summary["wall_clock_seconds"] = wall_clock_seconds
    summary["samples_per_second"] = len(details) / wall_clock_seconds if wall_clock_seconds else 0.0
    summary["output_tokens_per_second"] = output_token_total / wall_clock_seconds if wall_clock_seconds else 0.0
    summary["effective_batch_size"] = args.batch_size

    output = {
        "config": {
            "model_path": args.model_path,
            "test_json": str(args.test_json),
            "data_dir": str(data_dir),
            "limit": args.limit,
            "warmup": args.warmup,
            "max_new_tokens": args.max_new_tokens,
            "device": args.device,
            "device_map": args.device_map,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "batch_size": args.batch_size,
            "visual_token_mode": args.visual_token_mode,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "point_tolerance": args.point_tolerance,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "load_seconds": load_seconds,
        "summary": summary,
        "final_memory": gpu_memory_snapshot(),
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PROFILE_DONE " + json.dumps(output["summary"], ensure_ascii=False), flush=True)
    print(f"PROFILE_OUTPUT {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
