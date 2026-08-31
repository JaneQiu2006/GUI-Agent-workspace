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
from cache_inference import (
    PAGE_CACHE_MODES,
    PAGE_CACHE_SCOPES,
    PAGE_CACHE_SIMILARITIES,
    PageCacheConfig,
    PageLevelCache,
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
    summary["cache"] = summarize_cache_runs(runs)
    return summary


def summarize_cache_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    records = [run.get("cache") for run in runs if isinstance(run.get("cache"), dict)]
    if not records:
        return {"mode": "off", "num_records": 0}
    count = len(records)
    hit_types = [str(record.get("page_cache_hit_type") or "miss") for record in records]
    return {
        "mode": str(records[-1].get("mode") or "off"),
        "num_records": count,
        "page_cache_hit_rate": sum(1 for record in records if record.get("page_cache_hit")) / count,
        "processor_cache_hit_rate": sum(1 for record in records if record.get("processor_cache_hit")) / count,
        "page_cache_hit_types": {
            hit_type: sum(1 for value in hit_types if value == hit_type)
            for hit_type in sorted(set(hit_types))
        },
        "cache_evictions": int(records[-1].get("cache_evictions") or 0),
        "cache_entries": int(records[-1].get("cache_entries") or 0),
    }


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
    parser.add_argument("--action_hint", help="Optional action hint for dynamic visual token modes")
    parser.add_argument("--page_cache_mode", default="off", choices=PAGE_CACHE_MODES)
    parser.add_argument("--page_cache_scope", default="session", choices=PAGE_CACHE_SCOPES)
    parser.add_argument("--page_cache_similarity", default="tile", choices=PAGE_CACHE_SIMILARITIES)
    parser.add_argument("--page_cache_max_entries", type=int, default=128)
    parser.add_argument("--page_cache_near_dhash_threshold", type=int, default=4)
    parser.add_argument("--page_cache_near_tile_threshold", type=float, default=0.98)
    parser.add_argument("--page_cache_patch_tile_threshold", type=float, default=0.90)
    parser.add_argument("--page_cache_tile_rows", type=int, default=8)
    parser.add_argument("--page_cache_tile_cols", type=int, default=16)
    parser.add_argument("--page_cache_ignored_top_ratio", type=float, default=0.0)
    parser.add_argument("--page_cache_ignored_bottom_ratio", type=float, default=0.0)
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
    page_cache = make_page_cache(args)

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
            action_hint=args.action_hint,
            page_cache=page_cache,
            cache_trajectory_id="single_image",
        )

    if page_cache is not None:
        page_cache.clear()
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
            action_hint=args.action_hint,
            page_cache=page_cache,
            cache_trajectory_id="single_image",
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
                "cache": result.cache,
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
            "action_hint": args.action_hint,
            "page_cache": page_cache.config.to_dict() if page_cache else {"mode": "off"},
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


def make_page_cache(args: argparse.Namespace) -> PageLevelCache | None:
    if args.page_cache_mode == "off":
        return None
    identity = "|".join(
        [
            str(args.model_path),
            str(args.device),
            str(args.device_map),
            str(args.dtype),
            str(args.attn_implementation),
            str(args.visual_token_mode),
            str(args.min_pixels),
            str(args.max_pixels),
        ]
    )
    return PageLevelCache(
        PageCacheConfig(
            mode=args.page_cache_mode,
            scope=args.page_cache_scope,
            similarity=args.page_cache_similarity,
            max_entries=args.page_cache_max_entries,
            near_dhash_threshold=args.page_cache_near_dhash_threshold,
            near_tile_threshold=args.page_cache_near_tile_threshold,
            patch_tile_threshold=args.page_cache_patch_tile_threshold,
            tile_rows=args.page_cache_tile_rows,
            tile_cols=args.page_cache_tile_cols,
            ignored_top_ratio=args.page_cache_ignored_top_ratio,
            ignored_bottom_ratio=args.page_cache_ignored_bottom_ratio,
            identity=identity,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
