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

from androidcontrol_actions import action_type, actions_match, canonicalize_action
from eval_androidcontrol import load_samples
from hf_gui_baseline import (
    DEFAULT_MODEL_PATH,
    gpu_memory_snapshot,
    load_model_and_processor,
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
    type_hits = sum(1 for item in details if item["type_success"])
    step_hits = sum(1 for item in details if item["step_success"])
    return {
        "num_steps": len(details),
        "type_accuracy": type_hits / len(details) if details else 0.0,
        "step_success_rate": step_hits / len(details) if details else 0.0,
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
    for index, sample in enumerate(warmup_records, 1):
        image_path = Path(str(sample["image_path"]))
        if not image_path.is_absolute():
            image_path = data_dir / image_path
        print(f"WARMUP {index}/{len(warmup_records)} image={image_path}", flush=True)
        profile_infer_one(
            model,
            processor,
            image_path,
            str(sample["task"]),
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )

    warnings = reset_gpu_memory_stats()
    for warning in warnings:
        print(f"GPU_MEMORY_WARNING {warning}", file=sys.stderr, flush=True)

    details = []
    for index, sample in enumerate(records, 1):
        image_path = Path(str(sample["image_path"]))
        if not image_path.is_absolute():
            image_path = data_dir / image_path
        gt_action = str(sample["action"])
        result = profile_infer_one(
            model,
            processor,
            image_path,
            str(sample["task"]),
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
        pred_action = canonicalize_action(result.raw_response)
        pred_type = action_type(pred_action)
        gt_type = action_type(gt_action)
        type_ok = pred_type == gt_type
        step_ok = actions_match(pred_action, gt_action, point_tolerance=args.point_tolerance)
        details.append(
            {
                "episode_id": sample.get("episode_id"),
                "step_id": sample.get("step_id"),
                "task": sample.get("task"),
                "image_path": str(image_path),
                "gt_action": gt_action,
                "raw_response": result.raw_response,
                "pred_action": pred_action,
                "gt_type": gt_type,
                "pred_type": pred_type,
                "type_success": type_ok,
                "step_success": step_ok,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "timings": result.timings,
                "memory": result.memory,
            }
        )
        print(
            f"PROFILE_STEP {index}/{len(records)} episode={sample.get('episode_id')} "
            f"step={sample.get('step_id')} type_ok={type_ok} step_ok={step_ok} "
            f"total={result.timings['total_seconds']:.4f}s generate={result.timings['generate_seconds']:.4f}s",
            flush=True,
        )

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
            "point_tolerance": args.point_tolerance,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "load_seconds": load_seconds,
        "summary": summarize_details(details),
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
