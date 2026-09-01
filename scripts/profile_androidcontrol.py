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
    summarize_cache_metrics,
    summarize_output_health,
)
from hf_gui_baseline import (
    DEFAULT_MODEL_PATH,
    GENERATION_PROFILE_MODES,
    VISION_TOKEN_MODES,
    gpu_memory_snapshot,
    load_model_and_processor,
    profile_infer_batch,
    profile_infer_one,
    reset_gpu_memory_stats,
    resolve_visual_token_mode,
)
from cache_inference import (
    PAGE_CACHE_MODES,
    PAGE_CACHE_SCOPES,
    PAGE_CACHE_SIMILARITIES,
    PageCacheConfig,
    PageLevelCache,
    parse_normalized_bboxes,
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
        "cache": summarize_cache_metrics(details),
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
    parser.add_argument("--generation_profile_mode", default="generate", choices=GENERATION_PROFILE_MODES)
    parser.add_argument("--page_cache_mode", default="off", choices=PAGE_CACHE_MODES)
    parser.add_argument("--page_cache_scope", default="trajectory", choices=PAGE_CACHE_SCOPES)
    parser.add_argument("--page_cache_similarity", default="tile", choices=PAGE_CACHE_SIMILARITIES)
    parser.add_argument("--page_cache_max_entries", type=int, default=128)
    parser.add_argument("--page_cache_near_dhash_threshold", type=int, default=4)
    parser.add_argument("--page_cache_near_tile_threshold", type=float, default=0.98)
    parser.add_argument("--page_cache_patch_tile_threshold", type=float, default=0.90)
    parser.add_argument("--page_cache_patch_max_changed_area_ratio", type=float, default=0.25)
    parser.add_argument(
        "--page_cache_patch_critical_region",
        action="append",
        default=[],
        help="Normalized bbox left,top,right,bottom that makes patch candidates risky; may be repeated",
    )
    parser.add_argument("--page_cache_tile_rows", type=int, default=8)
    parser.add_argument("--page_cache_tile_cols", type=int, default=16)
    parser.add_argument("--page_cache_ignored_top_ratio", type=float, default=0.0)
    parser.add_argument("--page_cache_ignored_bottom_ratio", type=float, default=0.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    records = load_samples(args.test_json)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise SystemExit("No AndroidControl samples found")
    if args.page_cache_mode != "off" and args.batch_size != 1:
        raise SystemExit("Page-level cache baseline currently supports batch_size=1 only")
    if args.generation_profile_mode == "manual_greedy" and args.batch_size != 1:
        raise SystemExit("manual_greedy generation profiling currently supports batch_size=1 only")
    data_dir = args.data_dir or args.test_json.parent
    page_cache = make_page_cache(args)

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
                action_hint=str(chunk[0].get("action", "")),
                page_cache=page_cache,
                cache_trajectory_id=chunk[0].get("episode_id"),
                generation_profile_mode=args.generation_profile_mode,
            )
        else:
            profile_infer_batch(
                model,
                processor,
                [
                    (image_path, str(sample["task"]), None, None, str(sample.get("action", "")))
                    for sample, image_path in zip(chunk, image_paths)
                ],
                max_new_tokens=args.max_new_tokens,
                device=args.device,
                visual_token_mode=args.visual_token_mode,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
            )

    if page_cache is not None:
        page_cache.clear()
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
                    action_hint=str(chunk[0].get("action", "")),
                    page_cache=page_cache,
                    cache_trajectory_id=chunk[0].get("episode_id"),
                    generation_profile_mode=args.generation_profile_mode,
                )
            ]
        else:
            results = profile_infer_batch(
                model,
                processor,
                [
                    (image_path, str(sample["task"]), None, None, str(sample.get("action", "")))
                    for sample, image_path in zip(chunk, image_paths)
                ],
                max_new_tokens=args.max_new_tokens,
                device=args.device,
                visual_token_mode=args.visual_token_mode,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
            )
        for offset, (sample, image_path, result) in enumerate(zip(chunk, image_paths, results), 1):
            detail = detail_from_result(
                sample,
                image_path,
                result,
                args.point_tolerance,
                visual_token_mode=resolve_visual_token_mode(
                    args.visual_token_mode,
                    instruction=str(sample.get("task", "")),
                    action_hint=str(sample.get("action", "")),
                ),
            )
            detail["timings"] = result.timings
            detail["memory"] = result.memory
            detail["generation_profile"] = result.generation_profile
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
            "generation_profile_mode": args.generation_profile_mode,
            "page_cache": page_cache.config.to_dict() if page_cache else {"mode": "off"},
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
            patch_max_changed_area_ratio=args.page_cache_patch_max_changed_area_ratio,
            patch_critical_regions=parse_normalized_bboxes(args.page_cache_patch_critical_region),
            tile_rows=args.page_cache_tile_rows,
            tile_cols=args.page_cache_tile_cols,
            ignored_top_ratio=args.page_cache_ignored_top_ratio,
            ignored_bottom_ratio=args.page_cache_ignored_bottom_ratio,
            identity=identity,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
