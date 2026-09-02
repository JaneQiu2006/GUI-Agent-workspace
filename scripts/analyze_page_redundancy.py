from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_FRAMEWORK = REPO_ROOT / "test_framework"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (TEST_FRAMEWORK, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cache_fingerprint import PageFingerprint, PageSimilarity, compare_page_fingerprints, compute_page_fingerprint
from cache_inference import PAGE_CACHE_SIMILARITIES, PageCacheConfig, parse_normalized_bboxes
from eval_androidcontrol import load_samples, resolved_image_path


DHASH_THRESHOLDS = (0, 2, 4, 8, 16)
TILE_THRESHOLDS = (0.90, 0.95, 0.98, 0.99)
DEFAULT_EPISODE_MILESTONES = (50, 100, 200, 500, 1000)
EPISODE_KEYS = ("episode_id", "trajectory_id", "episode", "traj_id", "trace_id")
STEP_KEYS = ("step_id", "step_index", "frame_index", "step")
APP_NAME_KEYS = ("app", "app_name", "application")
APP_KEYS = ("package", "package_name", "app_package", "launched_package", "app", "app_name", "application")
PACKAGE_KEYS = ("package", "package_name", "app_package", "launched_package")
TASK_KEYS = ("goal", "task_goal", "instruction", "task", "query", "question", "任务")


@dataclass(frozen=True)
class PageEntry:
    global_index: int
    sampled_episode_position: int
    source_record_index: int
    episode_id: Optional[str]
    step_id: Optional[str]
    app_key: Optional[str]
    app_value: Optional[str]
    package_value: Optional[str]
    task_key: Optional[str]
    task_value: Optional[str]
    image_path: Path
    fingerprint: PageFingerprint


@dataclass
class BestMatch:
    exact_entry: Optional[PageEntry] = None
    exact_similarity: Optional[PageSimilarity] = None
    min_dhash: Optional[int] = None
    min_dhash_entry: Optional[PageEntry] = None
    min_dhash_similarity: Optional[PageSimilarity] = None
    max_tile_ratio: Optional[float] = None
    max_tile_entry: Optional[PageEntry] = None
    max_tile_similarity: Optional[PageSimilarity] = None
    primary_entry: Optional[PageEntry] = None
    primary_similarity: Optional[PageSimilarity] = None
    primary_score: Tuple[int, float, float] = (-1, -1.0, -10000.0)
    comparisons: int = 0

    def update(self, entry: PageEntry, similarity: PageSimilarity) -> None:
        self.comparisons += 1
        if similarity.exact and self.exact_entry is None:
            self.exact_entry = entry
            self.exact_similarity = similarity
        if similarity.dhash_hamming is not None and (
            self.min_dhash is None or similarity.dhash_hamming < self.min_dhash
        ):
            self.min_dhash = similarity.dhash_hamming
            self.min_dhash_entry = entry
            self.min_dhash_similarity = similarity
        if similarity.tile_unchanged_ratio is not None and (
            self.max_tile_ratio is None or similarity.tile_unchanged_ratio > self.max_tile_ratio
        ):
            self.max_tile_ratio = similarity.tile_unchanged_ratio
            self.max_tile_entry = entry
            self.max_tile_similarity = similarity

        exact_score = 1 if similarity.exact else 0
        tile_score = similarity.tile_unchanged_ratio if similarity.tile_unchanged_ratio is not None else -1.0
        dhash_score = -float(similarity.dhash_hamming if similarity.dhash_hamming is not None else 10000)
        score = (exact_score, tile_score, dhash_score)
        if score > self.primary_score:
            self.primary_score = score
            self.primary_entry = entry
            self.primary_similarity = similarity

    def has_history(self) -> bool:
        return self.comparisons > 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline AndroidControl page redundancy and potential cache-hit analysis"
    )
    parser.add_argument("--test_json", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, help="Defaults to the parent directory of --test_json")
    parser.add_argument("--num-episodes", "--num_episodes", dest="num_episodes", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/page_redundancy_analysis"),
        help="Output directory. Defaults to results/page_redundancy_analysis/<timestamp>; with --run-name, creates output-dir/run-name.",
    )
    parser.add_argument("--run-name", help="Defaults to androidcontrol_page_redundancy_<timestamp>")
    parser.add_argument("--episode-milestones", default="50,100,200,500,1000")
    parser.add_argument("--dhash-threshold", type=int, action="append", dest="dhash_thresholds")
    parser.add_argument("--tile-threshold", type=float, action="append", dest="tile_thresholds")
    parser.add_argument("--page_cache_similarity", default="tile", choices=PAGE_CACHE_SIMILARITIES)
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
    parser.add_argument("--no-plots", action="store_true", help="Skip optional matplotlib plots")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    records = load_samples(args.test_json)
    if not records:
        raise SystemExit("No AndroidControl samples found")
    data_dir = args.data_dir or args.test_json.parent
    selected_records, sampling = sample_episodes(records, args.num_episodes, args.seed)
    if not selected_records:
        raise SystemExit("No sampled AndroidControl records found")

    run_dir = resolve_run_dir(args.output_dir, args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = PageCacheConfig(
        mode="observe",
        scope="dataset",
        similarity=args.page_cache_similarity,
        max_entries=max(1, len(selected_records)),
        near_dhash_threshold=args.page_cache_near_dhash_threshold,
        near_tile_threshold=args.page_cache_near_tile_threshold,
        patch_tile_threshold=args.page_cache_patch_tile_threshold,
        patch_max_changed_area_ratio=args.page_cache_patch_max_changed_area_ratio,
        patch_critical_regions=parse_normalized_bboxes(args.page_cache_patch_critical_region),
        tile_rows=args.page_cache_tile_rows,
        tile_cols=args.page_cache_tile_cols,
        ignored_top_ratio=args.page_cache_ignored_top_ratio,
        ignored_bottom_ratio=args.page_cache_ignored_bottom_ratio,
        identity="page-redundancy-analysis",
    )
    dhash_thresholds = tuple(args.dhash_thresholds or DHASH_THRESHOLDS)
    tile_thresholds = tuple(args.tile_thresholds or TILE_THRESHOLDS)
    milestones = parse_int_list(args.episode_milestones)

    per_page, metadata = analyze_records(selected_records, data_dir, config, dhash_thresholds, tile_thresholds)
    groups = available_groups(metadata)
    grouped_stats = [summarize_group(group, per_page, dhash_thresholds, tile_thresholds) for group in groups]
    threshold_stats = build_threshold_stats(groups, per_page, dhash_thresholds, tile_thresholds)
    history_stats = build_history_size_stats(per_page, milestones, dhash_thresholds, tile_thresholds)
    plots = [] if args.no_plots else write_plots(run_dir, per_page, threshold_stats, history_stats)

    summary = {
        "config": {
            "test_json": str(args.test_json),
            "data_dir": str(data_dir),
            "output_dir": str(run_dir),
            "num_episodes": args.num_episodes,
            "seed": args.seed,
            "dhash_thresholds": list(dhash_thresholds),
            "tile_thresholds": list(tile_thresholds),
            "page_cache_config": config.to_dict(),
            "episode_milestones": milestones,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "sampling": sampling,
        "metadata": metadata,
        "overall": summarize_group("overall", per_page, dhash_thresholds, tile_thresholds),
        "grouped_stats": grouped_stats,
        "nearest_neighbor_distribution": distribution_summary(per_page),
        "history_size_stats": history_stats,
        "plots": plots,
    }

    write_json(run_dir / "summary.json", summary)
    write_jsonl(run_dir / "per_page.jsonl", per_page)
    write_per_page_csv(run_dir / "per_page.csv", per_page, dhash_thresholds, tile_thresholds)
    write_dict_rows(run_dir / "grouped_stats.csv", grouped_stats)
    write_dict_rows(run_dir / "threshold_stats.csv", threshold_stats)
    write_dict_rows(run_dir / "history_size_stats.csv", history_stats)
    print(
        "PAGE_REDUNDANCY_ANALYSIS_DONE "
        + json.dumps(
            {
                "output_dir": str(run_dir),
                "episodes": sampling["selected_episode_count"],
                "pages": summary["overall"]["total_pages"],
                "exact_duplicate_pages": summary["overall"]["exact_hit_count"],
                "current_cache_config_hit_rate": summary["overall"]["current_cache_config_hit_rate"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def sample_episodes(
    records: Sequence[Dict[str, Any]],
    num_episodes: Optional[int],
    seed: int,
) -> Tuple[List[Tuple[int, Dict[str, Any], int]], Dict[str, Any]]:
    episode_order: List[str] = []
    record_episode_keys: List[str] = []
    seen = set()
    for index, record in enumerate(records):
        episode_id = metadata_value(record, EPISODE_KEYS)
        key = str(episode_id) if episode_id not in (None, "") else f"__missing_episode_{index}"
        record_episode_keys.append(key)
        if key not in seen:
            seen.add(key)
            episode_order.append(key)

    if num_episodes is None or num_episodes >= len(episode_order):
        selected_keys = set(episode_order)
    else:
        selected_keys = set(random.Random(seed).sample(episode_order, num_episodes))
    selected_episode_positions = {
        key: position for position, key in enumerate((key for key in episode_order if key in selected_keys), 1)
    }
    selected_records = [
        (index, record, selected_episode_positions[record_episode_keys[index]])
        for index, record in enumerate(records)
        if record_episode_keys[index] in selected_keys
    ]
    return selected_records, {
        "input_record_count": len(records),
        "input_episode_count": len(episode_order),
        "selected_record_count": len(selected_records),
        "selected_episode_count": len(selected_keys),
        "requested_num_episodes": num_episodes,
        "seed": seed,
        "selection": "random_episode_sample_preserve_input_order"
        if num_episodes is not None and num_episodes < len(episode_order)
        else "all_episodes_preserve_input_order",
    }


def analyze_records(
    selected_records: Sequence[Tuple[int, Dict[str, Any], int]],
    data_dir: Path,
    config: PageCacheConfig,
    dhash_thresholds: Sequence[int],
    tile_thresholds: Sequence[float],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    history: List[PageEntry] = []
    details: List[Dict[str, Any]] = []
    metadata_counts = {"episode": 0, "step": 0, "app": 0, "package": 0, "task": 0}
    total = len(selected_records)
    for page_index, (source_index, sample, sampled_episode_position) in enumerate(selected_records, 1):
        image_path = resolved_image_path(sample, data_dir)
        fingerprint = compute_page_fingerprint(
            image_path,
            tile_rows=config.tile_rows,
            tile_cols=config.tile_cols,
            ignored_top_ratio=config.ignored_top_ratio,
            ignored_bottom_ratio=config.ignored_bottom_ratio,
        )
        entry = make_entry(page_index, sampled_episode_position, source_index, sample, image_path, fingerprint)
        metadata_counts["episode"] += int(entry.episode_id is not None)
        metadata_counts["step"] += int(entry.step_id is not None)
        metadata_counts["app"] += int(entry.app_key is not None)
        metadata_counts["package"] += int(entry.package_value is not None)
        metadata_counts["task"] += int(entry.task_key is not None)

        best_by_group = init_best_by_group()
        for previous in history:
            similarity = compare_page_fingerprints(fingerprint, previous.fingerprint)
            update_best_groups(best_by_group, entry, previous, similarity)

        detail = detail_from_entry(entry, best_by_group, config, dhash_thresholds, tile_thresholds)
        details.append(detail)
        history.append(entry)
        if page_index == 1 or page_index % 100 == 0 or page_index == total:
            print(
                f"PAGE_REDUNDANCY_STEP {page_index}/{total} episode={entry.episode_id} "
                f"step={entry.step_id} history={len(history) - 1}",
                flush=True,
            )

    metadata = {
        "field_sources": {
            "episode": list(EPISODE_KEYS),
            "step": list(STEP_KEYS),
            "app": list(APP_NAME_KEYS),
            "app_key": list(APP_KEYS),
            "package": list(PACKAGE_KEYS),
            "task": list(TASK_KEYS),
            "task_note": "same_task uses goal/task_goal before task/instruction, because prepared AndroidControl task may include current-step text.",
        },
        "available_dimensions": {
            name: count == len(selected_records) for name, count in metadata_counts.items()
        },
        "non_null_counts": metadata_counts,
        "skipped_group_dimensions": [
            name for name, count in metadata_counts.items() if name in {"episode", "app", "task"} and count != len(selected_records)
        ],
    }
    return details, metadata


def make_entry(
    page_index: int,
    sampled_episode_position: int,
    source_index: int,
    sample: Dict[str, Any],
    image_path: Path,
    fingerprint: PageFingerprint,
) -> PageEntry:
    app_value = metadata_value(sample, APP_NAME_KEYS)
    package_value = metadata_value(sample, PACKAGE_KEYS)
    task_value = metadata_value(sample, TASK_KEYS)
    return PageEntry(
        global_index=page_index,
        sampled_episode_position=sampled_episode_position,
        source_record_index=source_index,
        episode_id=optional_text(metadata_value(sample, EPISODE_KEYS)),
        step_id=optional_text(metadata_value(sample, STEP_KEYS)),
        app_key=optional_text(package_value if package_value not in (None, "") else app_value),
        app_value=optional_text(app_value),
        package_value=optional_text(package_value),
        task_key=optional_text(task_value),
        task_value=optional_text(task_value),
        image_path=image_path,
        fingerprint=fingerprint,
    )


def init_best_by_group() -> Dict[str, BestMatch]:
    return {
        "overall": BestMatch(),
        "same_episode": BestMatch(),
        "cross_episode": BestMatch(),
        "same_task": BestMatch(),
        "cross_task": BestMatch(),
        "same_app": BestMatch(),
        "cross_app": BestMatch(),
    }


def update_best_groups(
    best_by_group: Dict[str, BestMatch],
    current: PageEntry,
    previous: PageEntry,
    similarity: PageSimilarity,
) -> None:
    best_by_group["overall"].update(previous, similarity)
    if current.episode_id is not None and previous.episode_id is not None:
        best_by_group["same_episode" if current.episode_id == previous.episode_id else "cross_episode"].update(
            previous, similarity
        )
    if current.task_key is not None and previous.task_key is not None:
        best_by_group["same_task" if current.task_key == previous.task_key else "cross_task"].update(previous, similarity)
    if current.app_key is not None and previous.app_key is not None:
        best_by_group["same_app" if current.app_key == previous.app_key else "cross_app"].update(previous, similarity)


def detail_from_entry(
    entry: PageEntry,
    best_by_group: Dict[str, BestMatch],
    config: PageCacheConfig,
    dhash_thresholds: Sequence[int],
    tile_thresholds: Sequence[float],
) -> Dict[str, Any]:
    overall = best_by_group["overall"]
    current_hit_type = current_cache_hit_type(overall, config)
    nearest = overall.primary_entry
    detail: Dict[str, Any] = {
        "global_index": entry.global_index,
        "sampled_episode_position": entry.sampled_episode_position,
        "source_record_index": entry.source_record_index,
        "episode_id": entry.episode_id,
        "step_id": entry.step_id,
        "app": entry.app_value,
        "package": entry.package_value,
        "app_key": entry.app_key,
        "task": entry.task_value,
        "task_key": entry.task_key,
        "image_path": str(entry.image_path),
        "image_sha256": entry.fingerprint.image_sha256,
        "width": entry.fingerprint.width,
        "height": entry.fingerprint.height,
        "dhash64": entry.fingerprint.dhash64,
        "tile_rows": entry.fingerprint.tile_rows,
        "tile_cols": entry.fingerprint.tile_cols,
        "has_history": overall.has_history(),
        "history_comparisons": overall.comparisons,
        "exact_sha256_match": overall.exact_entry is not None,
        "min_dhash_hamming": overall.min_dhash,
        "max_tile_unchanged_ratio": overall.max_tile_ratio,
        "nearest_global_index": nearest.global_index if nearest else None,
        "nearest_episode_id": nearest.episode_id if nearest else None,
        "nearest_step_id": nearest.step_id if nearest else None,
        "nearest_image_sha256": nearest.fingerprint.image_sha256 if nearest else None,
        "nearest_same_episode": relation_same(entry.episode_id, nearest.episode_id if nearest else None),
        "nearest_same_task": relation_same(entry.task_key, nearest.task_key if nearest else None),
        "nearest_same_app": relation_same(entry.app_key, nearest.app_key if nearest else None),
        "min_dhash_nearest_global_index": overall.min_dhash_entry.global_index if overall.min_dhash_entry else None,
        "min_dhash_nearest_episode_id": overall.min_dhash_entry.episode_id if overall.min_dhash_entry else None,
        "min_dhash_nearest_step_id": overall.min_dhash_entry.step_id if overall.min_dhash_entry else None,
        "max_tile_nearest_global_index": overall.max_tile_entry.global_index if overall.max_tile_entry else None,
        "max_tile_nearest_episode_id": overall.max_tile_entry.episode_id if overall.max_tile_entry else None,
        "max_tile_nearest_step_id": overall.max_tile_entry.step_id if overall.max_tile_entry else None,
        "current_cache_config_hit_type": current_hit_type,
        "current_cache_config_hit": current_hit_type in {"exact", "near", "patch_candidate"},
        "groups": {},
    }
    for group, best in best_by_group.items():
        group_hit_type = current_cache_hit_type(best, config)
        group_detail = {
            "has_prior": best.has_history(),
            "prior_count": best.comparisons,
            "exact_sha256_match": best.exact_entry is not None,
            "min_dhash_hamming": best.min_dhash,
            "max_tile_unchanged_ratio": best.max_tile_ratio,
            "current_cache_config_hit_type": group_hit_type,
            "current_cache_config_hit": group_hit_type in {"exact", "near", "patch_candidate"},
            "dhash_hits": {str(threshold): dhash_hit(best, threshold) for threshold in dhash_thresholds},
            "tile_hits": {format_threshold(threshold): tile_hit(best, threshold) for threshold in tile_thresholds},
        }
        detail["groups"][group] = group_detail
    return detail


def current_cache_hit_type(best: BestMatch, config: PageCacheConfig) -> str:
    if best.exact_entry is not None:
        return "exact"
    similarity = best.primary_similarity
    if similarity is None:
        return "miss"
    dhash_ok = similarity.dhash_hamming is not None and similarity.dhash_hamming <= config.near_dhash_threshold
    tile_ok = similarity.tile_unchanged_ratio is not None and similarity.tile_unchanged_ratio >= config.near_tile_threshold
    if config.similarity == "dhash" and dhash_ok:
        return "near"
    if config.similarity == "tile" and dhash_ok and tile_ok:
        return "near"
    if patch_candidate_allowed(similarity, config):
        return "patch_candidate"
    return "miss"


def patch_candidate_allowed(similarity: PageSimilarity, config: PageCacheConfig) -> bool:
    if config.similarity not in {"tile", "dhash"}:
        return False
    if similarity.exact or similarity.changed_tile_count <= 0:
        return False
    if similarity.tile_unchanged_ratio is None or similarity.tile_unchanged_ratio < config.patch_tile_threshold:
        return False
    if similarity.changed_bbox_area_ratio > config.patch_max_changed_area_ratio:
        return False
    if similarity.changed_bbox is not None:
        for region in config.patch_critical_regions:
            if bbox_intersects(similarity.changed_bbox, region):
                return False
    return True


def summarize_group(
    group: str,
    details: Sequence[Dict[str, Any]],
    dhash_thresholds: Sequence[int],
    tile_thresholds: Sequence[float],
) -> Dict[str, Any]:
    selected = [detail["groups"][group] for detail in details if group in detail.get("groups", {})]
    total = len(selected)
    with_prior = [item for item in selected if item["has_prior"]]
    exact_count = sum(1 for item in selected if item["exact_sha256_match"])
    current_count = sum(1 for item in selected if item["current_cache_config_hit"])
    dhash_values = [item["min_dhash_hamming"] for item in selected if item.get("min_dhash_hamming") is not None]
    tile_values = [
        item["max_tile_unchanged_ratio"] for item in selected if item.get("max_tile_unchanged_ratio") is not None
    ]
    row: Dict[str, Any] = {
        "group": group,
        "total_pages": total,
        "pages_with_prior_group": len(with_prior),
        "exact_hit_count": exact_count,
        "exact_hit_rate": rate(exact_count, total),
        "exact_hit_rate_given_prior": rate(exact_count, len(with_prior)),
        "current_cache_config_hit_count": current_count,
        "current_cache_config_hit_rate": rate(current_count, total),
        "current_cache_config_hit_rate_given_prior": rate(current_count, len(with_prior)),
        "min_dhash_p50": percentile(dhash_values, 50),
        "min_dhash_p90": percentile(dhash_values, 90),
        "min_dhash_p95": percentile(dhash_values, 95),
        "max_tile_p50": percentile(tile_values, 50),
        "max_tile_p90": percentile(tile_values, 90),
        "max_tile_p95": percentile(tile_values, 95),
    }
    for threshold in dhash_thresholds:
        count = sum(1 for item in selected if item["dhash_hits"][str(threshold)])
        row[f"dhash_le_{threshold}_count"] = count
        row[f"dhash_le_{threshold}_rate"] = rate(count, total)
    for threshold in tile_thresholds:
        label = format_threshold(threshold)
        count = sum(1 for item in selected if item["tile_hits"][label])
        row[f"tile_ge_{label}_count"] = count
        row[f"tile_ge_{label}_rate"] = rate(count, total)
    return row


def build_threshold_stats(
    groups: Sequence[str],
    details: Sequence[Dict[str, Any]],
    dhash_thresholds: Sequence[int],
    tile_thresholds: Sequence[float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for group in groups:
        selected = [detail["groups"][group] for detail in details if group in detail.get("groups", {})]
        total = len(selected)
        prior = sum(1 for item in selected if item["has_prior"])
        rows.append(threshold_row(group, "exact", "sha256", selected, total, prior, "exact_sha256_match"))
        rows.append(
            threshold_row(
                group, "current_cache_config", "configured", selected, total, prior, "current_cache_config_hit"
            )
        )
        for threshold in dhash_thresholds:
            count = sum(1 for item in selected if item["dhash_hits"][str(threshold)])
            rows.append(
                {
                    "group": group,
                    "threshold_type": "dhash",
                    "threshold": f"<={threshold}",
                    "hit_count": count,
                    "total_pages": total,
                    "pages_with_prior_group": prior,
                    "hit_rate": rate(count, total),
                    "hit_rate_given_prior": rate(count, prior),
                }
            )
        for threshold in tile_thresholds:
            label = format_threshold(threshold)
            count = sum(1 for item in selected if item["tile_hits"][label])
            rows.append(
                {
                    "group": group,
                    "threshold_type": "tile",
                    "threshold": f">={label}",
                    "hit_count": count,
                    "total_pages": total,
                    "pages_with_prior_group": prior,
                    "hit_rate": rate(count, total),
                    "hit_rate_given_prior": rate(count, prior),
                }
            )
    return rows


def threshold_row(
    group: str,
    threshold_type: str,
    threshold: str,
    selected: Sequence[Dict[str, Any]],
    total: int,
    prior: int,
    key: str,
) -> Dict[str, Any]:
    count = sum(1 for item in selected if item[key])
    return {
        "group": group,
        "threshold_type": threshold_type,
        "threshold": threshold,
        "hit_count": count,
        "total_pages": total,
        "pages_with_prior_group": prior,
        "hit_rate": rate(count, total),
        "hit_rate_given_prior": rate(count, prior),
    }


def build_history_size_stats(
    details: Sequence[Dict[str, Any]],
    milestones: Sequence[int],
    dhash_thresholds: Sequence[int],
    tile_thresholds: Sequence[float],
) -> List[Dict[str, Any]]:
    max_episode = max((int(item["sampled_episode_position"]) for item in details), default=0)
    rows = []
    for milestone in milestones:
        if milestone > max_episode:
            continue
        selected = [item for item in details if int(item["sampled_episode_position"]) <= milestone]
        summary = summarize_group("overall", selected, dhash_thresholds, tile_thresholds)
        rows.append(
            {
                "history_episode_count": milestone,
                "page_count": len(selected),
                "exact_hit_rate": summary["exact_hit_rate"],
                "current_cache_config_hit_rate": summary["current_cache_config_hit_rate"],
                "dhash_le_4_hit_rate": summary.get("dhash_le_4_rate"),
                "tile_ge_0.98_hit_rate": summary.get("tile_ge_0.98_rate"),
            }
        )
    return rows


def distribution_summary(details: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    dhash_values = [item["min_dhash_hamming"] for item in details if item.get("min_dhash_hamming") is not None]
    tile_values = [item["max_tile_unchanged_ratio"] for item in details if item.get("max_tile_unchanged_ratio") is not None]
    sha_values = [item["image_sha256"] for item in details]
    return {
        "page_count": len(details),
        "unique_sha256_page_count": len(set(sha_values)),
        "exact_duplicate_count": len(details) - len(set(sha_values)),
        "min_dhash_hamming": numeric_distribution(dhash_values, [(0, 0), (1, 3), (3, 5), (5, 9), (9, 17), (17, 33), (33, 65)]),
        "max_tile_unchanged_ratio": numeric_distribution(
            tile_values,
            [(0.0, 0.50), (0.50, 0.75), (0.75, 0.90), (0.90, 0.95), (0.95, 0.98), (0.98, 0.99), (0.99, 1.0)],
        ),
    }


def numeric_distribution(values: Sequence[float], bins: Sequence[Tuple[float, float]]) -> Dict[str, Any]:
    clean = [float(value) for value in values if value is not None]
    return {
        "count": len(clean),
        "min": min(clean) if clean else None,
        "max": max(clean) if clean else None,
        "mean": statistics.fmean(clean) if clean else None,
        "percentiles": {f"p{p}": percentile(clean, p) for p in (10, 25, 50, 75, 90, 95)},
        "histogram": histogram(clean, bins),
    }


def histogram(values: Sequence[float], bins: Sequence[Tuple[float, float]]) -> List[Dict[str, Any]]:
    rows = []
    for left, right in bins:
        if left == right:
            count = sum(1 for value in values if value == left)
            label = str(left)
        elif right == 1.0:
            count = sum(1 for value in values if left <= value <= right)
            label = f"[{left},{right}]"
        else:
            count = sum(1 for value in values if left <= value < right)
            label = f"[{left},{right})"
        rows.append({"bin": label, "count": count, "rate": rate(count, len(values))})
    return rows


def write_plots(
    run_dir: Path,
    details: Sequence[Dict[str, Any]],
    threshold_stats: Sequence[Dict[str, Any]],
    history_stats: Sequence[Dict[str, Any]],
) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    written: List[str] = []
    dhash_values = [item["min_dhash_hamming"] for item in details if item.get("min_dhash_hamming") is not None]
    tile_values = [item["max_tile_unchanged_ratio"] for item in details if item.get("max_tile_unchanged_ratio") is not None]
    if dhash_values:
        path = run_dir / "dhash_nearest_distribution.png"
        plt.figure(figsize=(7, 4))
        plt.hist(dhash_values, bins=20)
        plt.xlabel("min dHash Hamming distance to history")
        plt.ylabel("page count")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        written.append(str(path))
    if tile_values:
        path = run_dir / "tile_similarity_distribution.png"
        plt.figure(figsize=(7, 4))
        plt.hist(tile_values, bins=20)
        plt.xlabel("max tile unchanged ratio to history")
        plt.ylabel("page count")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        written.append(str(path))
    overall_thresholds = [row for row in threshold_stats if row["group"] == "overall" and row["threshold_type"] in {"dhash", "tile"}]
    if overall_thresholds:
        path = run_dir / "threshold_vs_potential_hit_rate.png"
        labels = [f"{row['threshold_type']} {row['threshold']}" for row in overall_thresholds]
        rates = [row["hit_rate"] for row in overall_thresholds]
        plt.figure(figsize=(9, 4))
        plt.bar(labels, rates)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("potential hit rate")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        written.append(str(path))
    if history_stats:
        path = run_dir / "history_size_vs_potential_hit_rate.png"
        episodes = [row["history_episode_count"] for row in history_stats]
        rates = [row["current_cache_config_hit_rate"] for row in history_stats]
        plt.figure(figsize=(7, 4))
        plt.plot(episodes, rates, marker="o")
        plt.xlabel("history episode count")
        plt.ylabel("current-config potential hit rate")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        written.append(str(path))
    return written


def available_groups(metadata: Dict[str, Any]) -> List[str]:
    groups = ["overall"]
    available = metadata["available_dimensions"]
    if available.get("episode"):
        groups.extend(["same_episode", "cross_episode"])
    if available.get("task"):
        groups.extend(["same_task", "cross_task"])
    if available.get("app"):
        groups.extend(["same_app", "cross_app"])
    return groups


def dhash_hit(best: BestMatch, threshold: int) -> bool:
    return best.min_dhash is not None and best.min_dhash <= threshold


def tile_hit(best: BestMatch, threshold: float) -> bool:
    return best.max_tile_ratio is not None and best.max_tile_ratio >= threshold


def relation_same(left: Optional[str], right: Optional[str]) -> Optional[bool]:
    if left is None or right is None:
        return None
    return left == right


def metadata_value(record: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def optional_text(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    return str(value)


def percentile(values: Sequence[float], percent: int) -> Optional[float]:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * percent / 100.0
    lower = int(position)
    upper = min(lower + 1, len(clean) - 1)
    weight = position - lower
    return clean[lower] * (1.0 - weight) + clean[upper] * weight


def rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def format_threshold(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def parse_int_list(value: str) -> List[int]:
    result = []
    for part in str(value).split(","):
        part = part.strip()
        if part:
            result.append(int(part))
    return result


def bbox_intersects(
    left_bbox: Tuple[float, float, float, float],
    right_bbox: Tuple[float, float, float, float],
) -> bool:
    left_a, top_a, right_a, bottom_a = left_bbox
    left_b, top_b, right_b, bottom_b = right_bbox
    return max(left_a, left_b) < min(right_a, right_b) and max(top_a, top_b) < min(bottom_a, bottom_b)


def resolve_run_dir(output_dir: Path, run_name: Optional[str]) -> Path:
    if run_name:
        return output_dir / run_name
    if output_dir.name != "page_redundancy_analysis":
        return output_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"androidcontrol_page_redundancy_{timestamp}"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_per_page_csv(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    dhash_thresholds: Sequence[int],
    tile_thresholds: Sequence[float],
) -> None:
    fields = [
        "global_index",
        "sampled_episode_position",
        "source_record_index",
        "episode_id",
        "step_id",
        "app",
        "package",
        "app_key",
        "task_key",
        "image_path",
        "image_sha256",
        "width",
        "height",
        "dhash64",
        "tile_rows",
        "tile_cols",
        "has_history",
        "history_comparisons",
        "exact_sha256_match",
        "min_dhash_hamming",
        "max_tile_unchanged_ratio",
        "nearest_global_index",
        "nearest_episode_id",
        "nearest_step_id",
        "nearest_image_sha256",
        "nearest_same_episode",
        "nearest_same_task",
        "nearest_same_app",
        "current_cache_config_hit_type",
        "current_cache_config_hit",
    ]
    fields.extend(f"dhash_le_{threshold}" for threshold in dhash_thresholds)
    fields.extend(f"tile_ge_{format_threshold(threshold)}" for threshold in tile_thresholds)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {field: row.get(field) for field in fields}
            for threshold in dhash_thresholds:
                flat[f"dhash_le_{threshold}"] = row["groups"]["overall"]["dhash_hits"][str(threshold)]
            for threshold in tile_thresholds:
                label = format_threshold(threshold)
                flat[f"tile_ge_{label}"] = row["groups"]["overall"]["tile_hits"][label]
            writer.writerow(flat)


def write_dict_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    raise SystemExit(main())
