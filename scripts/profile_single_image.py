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
if str(TEST_FRAMEWORK) not in sys.path:
    sys.path.insert(0, str(TEST_FRAMEWORK))

from hf_gui_baseline import (
    DEFAULT_MODEL_PATH,
    VISION_TOKEN_MODES,
    gpu_memory_snapshot,
    load_model_and_processor,
    profile_infer_one,
    reset_gpu_memory_stats,
)


def summarize_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    timing_keys = sorted({key for run in runs for key in run["timings"]})
    summary: Dict[str, Any] = {"num_runs": len(runs), "timings_seconds": {}}
    for key in timing_keys:
        values = [float(run["timings"][key]) for run in runs if key in run["timings"]]
        summary["timings_seconds"][key] = {
            "mean": statistics.fmean(values),
            "min": min(values),
            "max": max(values),
            "median": statistics.median(values),
        }
    output_values = [run["output_tokens"] for run in runs if run.get("output_tokens") is not None]
    if output_values:
        summary["avg_output_tokens"] = statistics.fmean(output_values)
    input_values = [run["input_tokens"] for run in runs if run.get("input_tokens") is not None]
    if input_values:
        summary["avg_input_tokens"] = statistics.fmean(input_values)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile one static GUI screenshot inference")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", type=Path, default=Path("results/profile_single_image.json"))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"))
    parser.add_argument("--attn_implementation", default=None)
    parser.add_argument("--visual_token_mode", default="default", choices=tuple(VISION_TOKEN_MODES))
    parser.add_argument("--min_pixels", type=int)
    parser.add_argument("--max_pixels", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")

    load_started = time.perf_counter()
    model, processor = load_model_and_processor(
        args.model_path,
        device=args.device,
        device_map=args.device_map or None,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    load_seconds = time.perf_counter() - load_started

    for index in range(args.warmup):
        print(f"WARMUP {index + 1}/{args.warmup}", flush=True)
        profile_infer_one(
            model,
            processor,
            args.image,
            args.instruction,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            visual_token_mode=args.visual_token_mode,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )

    warnings = reset_gpu_memory_stats()
    for warning in warnings:
        print(f"GPU_MEMORY_WARNING {warning}", file=sys.stderr, flush=True)

    runs = []
    for index in range(args.repeats):
        print(f"PROFILE_RUN {index + 1}/{args.repeats}", flush=True)
        result = profile_infer_one(
            model,
            processor,
            args.image,
            args.instruction,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            visual_token_mode=args.visual_token_mode,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
        runs.append(
            {
                "index": index,
                "image": str(args.image),
                "instruction": args.instruction,
                "raw_response": result.raw_response,
                "parsed_action": result.parsed_action,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "timings": result.timings,
                "memory": result.memory,
            }
        )

    output = {
        "config": {
            "model_path": args.model_path,
            "image": str(args.image),
            "instruction": args.instruction,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "max_new_tokens": args.max_new_tokens,
            "device": args.device,
            "device_map": args.device_map,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "visual_token_mode": args.visual_token_mode,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "load_seconds": load_seconds,
        "summary": summarize_runs(runs),
        "final_memory": gpu_memory_snapshot(),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PROFILE_DONE " + json.dumps(output["summary"], ensure_ascii=False), flush=True)
    print(f"PROFILE_OUTPUT {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
