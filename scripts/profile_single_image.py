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
    GENERATION_PROFILE_MODES,
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
    parse_normalized_bboxes,
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
            "p90": percentile90(values),
        }
    output_values = [run["output_tokens"] for run in runs if run.get("output_tokens") is not None]
    if output_values:
        summary["avg_output_tokens"] = statistics.fmean(output_values)
    input_values = [run["input_tokens"] for run in runs if run.get("input_tokens") is not None]
    if input_values:
        summary["avg_input_tokens"] = statistics.fmean(input_values)
    summary["fine_grained_profile"] = summarize_fine_grained_profiles(runs)
    summary["cache"] = summarize_cache_runs(runs)
    return summary


def summarize_fine_grained_profiles(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    profiles = [run["stage_profile"] for run in runs if isinstance(run.get("stage_profile"), dict)]
    if not profiles:
        return {}
    episode = sum_stage_profiles(profiles, group_id="single_image")
    overall = sum_stage_profiles(profiles, group_id="overall")
    return {
        "per_step": summarize_stage_values(profiles),
        "per_episode": {
            "episodes": [episode],
            "summary": summarize_stage_values([episode]),
        },
        "overall": overall,
        "human_readable_summary": build_human_summary(overall),
    }


def summarize_stage_values(profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
    stage_keys = sorted({
        key
        for profile in profiles
        for key, value in (profile.get("stages_ms") or {}).items()
        if value is not None
    })
    stage_summary = {}
    for key in stage_keys:
        values = [
            float(profile["stages_ms"][key])
            for profile in profiles
            if (profile.get("stages_ms") or {}).get(key) is not None
        ]
        stage_summary[key] = summarize_values(values)
    visual_values = [
        float(profile["visual_related_latency_ms"])
        for profile in profiles
        if profile.get("visual_related_latency_ms") is not None
    ]
    visual_ratios = [
        float(profile["visual_related_ratio"])
        for profile in profiles
        if profile.get("visual_related_ratio") is not None
    ]
    return {
        "stages_ms": stage_summary,
        "visual_related_latency_ms": summarize_values(visual_values),
        "visual_related_ratio": summarize_values(visual_ratios),
    }


def sum_stage_profiles(profiles: List[Dict[str, Any]], group_id: str) -> Dict[str, Any]:
    stage_keys = sorted({
        key
        for profile in profiles
        for key, value in (profile.get("stages_ms") or {}).items()
        if value is not None
    })
    stages_ms = {
        key: sum(
            float(profile["stages_ms"][key])
            for profile in profiles
            if (profile.get("stages_ms") or {}).get(key) is not None
        )
        for key in stage_keys
    }
    total_ms = float(stages_ms.get("total_inference_latency", 0.0))
    visual_ms = sum(float(stages_ms.get(key, 0.0)) for key in (
        "image_load_decode_preprocess",
        "image_resize_normalize_patch_or_token_construction",
        "vision_encoder_visual_feature_extraction",
        "visual_feature_projector_adapter",
    ))
    metadata = summarize_metadata([profile.get("metadata") or {} for profile in profiles])
    return {
        "group_id": group_id,
        "num_steps": len(profiles),
        "stages_ms": stages_ms,
        "stage_ratios": {
            key: value / total_ms if total_ms > 0 else None
            for key, value in stages_ms.items()
        },
        "visual_related_latency_ms": visual_ms,
        "visual_related_ratio": visual_ms / total_ms if total_ms > 0 else None,
        "metadata": metadata,
    }


def summarize_metadata(metadata_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "image_width": summarize_values([item.get("image_width") for item in metadata_items if item.get("image_width") is not None]),
        "image_height": summarize_values([item.get("image_height") for item in metadata_items if item.get("image_height") is not None]),
        "visual_patch_count": summarize_values([item.get("visual_patch_count") for item in metadata_items if item.get("visual_patch_count") is not None]),
        "visual_token_count": summarize_values([item.get("visual_token_count") for item in metadata_items if item.get("visual_token_count") is not None]),
        "prompt_tokens": summarize_values([item.get("prompt_tokens") for item in metadata_items if item.get("prompt_tokens") is not None]),
        "generated_tokens": summarize_values([item.get("generated_tokens") for item in metadata_items if item.get("generated_tokens") is not None]),
    }


def summarize_values(values: List[Any]) -> Dict[str, Any]:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return {"count": 0}
    return {
        "count": len(numeric),
        "mean": statistics.fmean(numeric),
        "median": statistics.median(numeric),
        "p90": percentile90(numeric),
        "min": min(numeric),
        "max": max(numeric),
    }


def percentile90(values: List[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = max(0, (9 * len(ordered) + 9) // 10 - 1)
    return ordered[min(index, len(ordered) - 1)]


def build_human_summary(overall: Dict[str, Any]) -> List[str]:
    stages = overall.get("stages_ms") or {}
    ratios = overall.get("stage_ratios") or {}
    visual_ratio = overall.get("visual_related_ratio")
    visual_text = "unknown" if visual_ratio is None else f"{visual_ratio * 100:.2f}%"
    bottleneck_keys = [
        "image_load_decode_preprocess",
        "image_resize_normalize_patch_or_token_construction",
        "vision_encoder_visual_feature_extraction",
        "visual_feature_projector_adapter",
        "multimodal_prefill",
        "decode_generation",
    ]
    available = [(key, float(stages[key])) for key in bottleneck_keys if stages.get(key) is not None]
    bottleneck = max(available, key=lambda item: item[1])[0] if available else "unknown"
    cache_boundaries = []
    if stages.get("image_resize_normalize_patch_or_token_construction", 0.0) > 0:
        cache_boundaries.append("processor outputs")
    if stages.get("vision_encoder_visual_feature_extraction", 0.0) > 0:
        cache_boundaries.append("vision encoder outputs")
    if stages.get("visual_feature_projector_adapter", 0.0) > 0:
        cache_boundaries.append("projected visual embeddings")
    if stages.get("multimodal_prefill", 0.0) > 0:
        cache_boundaries.append("exact multimodal prefill KV")
    return [
        f"视觉相关耗时占 total inference latency 的 {visual_text}。",
        f"当前可观测主瓶颈是 {bottleneck}，该阶段占比约 {float(ratios.get(bottleneck) or 0.0) * 100:.2f}%。",
        "适合作为 Feature Cache 边界的中间结果：" + (", ".join(cache_boundaries) if cache_boundaries else "当前 profile 未观测到可缓存视觉边界。"),
    ]


def summarize_cache_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    records = [run.get("cache") for run in runs if isinstance(run.get("cache"), dict)]
    if not records:
        return {"mode": "off", "num_records": 0}
    count = len(records)
    hit_types = [str(record.get("page_cache_hit_type") or "miss") for record in records]
    changed_area_ratios = [
        float(record["changed_bbox_area_ratio"])
        for record in records
        if record.get("changed_bbox_area_ratio") is not None
    ]
    changed_tile_counts = [
        int(record["changed_tile_count"])
        for record in records
        if record.get("changed_tile_count") is not None
    ]
    return {
        "mode": str(records[-1].get("mode") or "off"),
        "num_records": count,
        "page_cache_hit_rate": sum(1 for record in records if record.get("page_cache_hit")) / count,
        "processor_cache_hit_rate": sum(1 for record in records if record.get("processor_cache_hit")) / count,
        "patch_candidate_rate": sum(1 for record in records if record.get("patch_candidate")) / count,
        "patch_candidate_allowed_rate": sum(
            1 for record in records if record.get("patch_candidate_allowed")
        ) / count,
        "page_cache_hit_types": {
            hit_type: sum(1 for value in hit_types if value == hit_type)
            for hit_type in sorted(set(hit_types))
        },
        "avg_changed_tile_count": statistics.fmean(changed_tile_counts) if changed_tile_counts else None,
        "avg_changed_bbox_area_ratio": statistics.fmean(changed_area_ratios) if changed_area_ratios else None,
        "patch_risk_reasons": summarize_patch_risks(records),
        "cache_evictions": int(records[-1].get("cache_evictions") or 0),
        "cache_entries": int(records[-1].get("cache_entries") or 0),
    }


def summarize_patch_risks(records: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        for reason in record.get("patch_risk_reasons") or []:
            counts[str(reason)] = counts.get(str(reason), 0) + 1
    return dict(sorted(counts.items()))


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
    parser.add_argument("--generation_profile_mode", default="generate", choices=GENERATION_PROFILE_MODES)
    parser.add_argument("--page_cache_mode", default="off", choices=PAGE_CACHE_MODES)
    parser.add_argument("--page_cache_scope", default="session", choices=PAGE_CACHE_SCOPES)
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
            generation_profile_mode=args.generation_profile_mode,
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
            generation_profile_mode=args.generation_profile_mode,
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
                "generation_profile": result.generation_profile,
                "profile_metadata": result.profile_metadata,
                "stage_profile": result.stage_profile,
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
            "generation_profile_mode": args.generation_profile_mode,
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
