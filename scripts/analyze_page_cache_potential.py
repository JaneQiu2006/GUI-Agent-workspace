from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_FRAMEWORK = REPO_ROOT / "test_framework"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (TEST_FRAMEWORK, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from androidcontrol_actions import action_type
from cache_inference import (
    PAGE_CACHE_SCOPES,
    PAGE_CACHE_SIMILARITIES,
    PageCacheConfig,
    PageLevelCache,
    parse_normalized_bboxes,
)
from eval_androidcontrol import load_samples, resolved_image_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze page-level cache potential without loading a model")
    parser.add_argument("--test_json", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/cache_analysis/page_potential.json"))
    parser.add_argument("--data_dir", type=Path, help="Defaults to the parent directory of --test_json")
    parser.add_argument("--limit", type=int)
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
    data_dir = args.data_dir or args.test_json.parent
    cache = PageLevelCache(
        PageCacheConfig(
            mode="observe",
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
            identity="page-cache-potential",
        )
    )

    details = []
    for index, sample in enumerate(records, 1):
        image_path = resolved_image_path(sample, data_dir)
        fingerprint, probe = cache.begin_step(image_path, sample.get("episode_id"))
        cache.finish_step(fingerprint, sample.get("episode_id"), probe)
        gt_type = action_type(str(sample.get("action") or ""))
        detail = {
            "index": index,
            "episode_id": sample.get("episode_id"),
            "step_id": sample.get("step_id"),
            "image_path": str(image_path),
            "gt_type": gt_type,
            "image_sha256": fingerprint.image_sha256,
            "page_cache": probe.to_dict(),
        }
        details.append(detail)
        print(
            f"PAGE_CACHE_STEP {index}/{len(records)} episode={sample.get('episode_id')} "
            f"step={sample.get('step_id')} gt_type={gt_type} hit={probe.page_cache_hit_type}",
            flush=True,
        )

    output = {
        "config": {
            "test_json": str(args.test_json),
            "data_dir": str(data_dir),
            "limit": args.limit,
            "page_cache": cache.config.to_dict(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "summary": summarize_details(details),
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PAGE_CACHE_ANALYSIS_DONE " + json.dumps(output["summary"], ensure_ascii=False), flush=True)
    print(f"PAGE_CACHE_ANALYSIS_OUTPUT {args.output}", flush=True)
    return 0


def summarize_details(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    hit_types = [str(item["page_cache"].get("page_cache_hit_type") or "miss") for item in details]
    tile_ratios = [
        float(item["page_cache"]["tile_unchanged_ratio"])
        for item in details
        if item["page_cache"].get("tile_unchanged_ratio") is not None
    ]
    dhash_values = [
        int(item["page_cache"]["similarity_dhash_hamming"])
        for item in details
        if item["page_cache"].get("similarity_dhash_hamming") is not None
    ]
    changed_area_ratios = [
        float(item["page_cache"]["changed_bbox_area_ratio"])
        for item in details
        if item["page_cache"].get("changed_bbox_area_ratio") is not None
    ]
    changed_tile_counts = [
        int(item["page_cache"]["changed_tile_count"])
        for item in details
        if item["page_cache"].get("changed_tile_count") is not None
    ]
    exact_duplicates = len(details) - len({item["image_sha256"] for item in details})
    return {
        "num_steps": len(details),
        "exact_duplicate_steps": exact_duplicates,
        "exact_duplicate_rate": exact_duplicates / len(details) if details else 0.0,
        "page_cache_hit_rate": sum(1 for item in details if item["page_cache"].get("page_cache_hit")) / len(details)
        if details
        else 0.0,
        "page_cache_hit_types": {
            hit_type: sum(1 for value in hit_types if value == hit_type)
            for hit_type in sorted(set(hit_types))
        },
        "avg_tile_unchanged_ratio": statistics.fmean(tile_ratios) if tile_ratios else None,
        "median_tile_unchanged_ratio": statistics.median(tile_ratios) if tile_ratios else None,
        "avg_similarity_dhash_hamming": statistics.fmean(dhash_values) if dhash_values else None,
        "avg_changed_tile_count": statistics.fmean(changed_tile_counts) if changed_tile_counts else None,
        "avg_changed_bbox_area_ratio": statistics.fmean(changed_area_ratios) if changed_area_ratios else None,
        "patch_candidate_rate": sum(
            1 for item in details if item["page_cache"].get("patch_candidate")
        ) / len(details) if details else 0.0,
        "patch_candidate_allowed_rate": sum(
            1 for item in details if item["page_cache"].get("patch_candidate_allowed")
        ) / len(details) if details else 0.0,
        "patch_risk_reasons": summarize_patch_risks(details),
        "by_gt_type": summarize_by_gt_type(details),
    }


def summarize_by_gt_type(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in details:
        grouped.setdefault(str(item.get("gt_type") or "UNKNOWN"), []).append(item)
    return {gt_type: summarize_group(items) for gt_type, items in sorted(grouped.items())}


def summarize_patch_risks(details: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in details:
        for reason in item["page_cache"].get("patch_risk_reasons") or []:
            counts[str(reason)] = counts.get(str(reason), 0) + 1
    return dict(sorted(counts.items()))


def summarize_group(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    hit_types = [str(item["page_cache"].get("page_cache_hit_type") or "miss") for item in items]
    return {
        "num_steps": len(items),
        "page_cache_hit_rate": sum(1 for item in items if item["page_cache"].get("page_cache_hit")) / len(items)
        if items
        else 0.0,
        "page_cache_hit_types": {
            hit_type: sum(1 for value in hit_types if value == hit_type)
            for hit_type in sorted(set(hit_types))
        },
        "patch_candidate_rate": sum(1 for item in items if item["page_cache"].get("patch_candidate")) / len(items)
        if items
        else 0.0,
        "patch_candidate_allowed_rate": sum(
            1 for item in items if item["page_cache"].get("patch_candidate_allowed")
        ) / len(items)
        if items
        else 0.0,
    }


if __name__ == "__main__":
    raise SystemExit(main())
