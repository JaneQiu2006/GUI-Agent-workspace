from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_FRAMEWORK = REPO_ROOT / "test_framework"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (TEST_FRAMEWORK, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_page_redundancy import (  # noqa: E402
    APP_NAME_KEYS,
    EPISODE_KEYS,
    STEP_KEYS,
    TASK_KEYS,
    metadata_value,
    optional_text,
)
from cache_fingerprint import PageFingerprint, PageSimilarity, compare_page_fingerprints, compute_page_fingerprint  # noqa: E402
from cache_inference import PAGE_CACHE_SCOPES, PAGE_CACHE_SIMILARITIES, PageCacheConfig, parse_normalized_bboxes  # noqa: E402
from eval_androidcontrol import load_samples, resolved_image_path  # noqa: E402
from hf_gui_baseline import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    VISION_TOKEN_MODES,
    extract_visual_features,
    load_model_and_processor,
)


DEFAULT_FEATURE_THRESHOLDS = (0.01, 0.03, 0.05, 0.10)
DEFAULT_SIMILARITY_BINS = ((0.90, 0.95), (0.95, 0.98), (0.98, 0.99), (0.99, 1.000001))


@dataclass(frozen=True)
class PageRecord:
    index: int
    sample: Dict[str, Any]
    image_path: Path
    fingerprint: PageFingerprint


@dataclass(frozen=True)
class SimilarPair:
    pair_id: str
    prev: PageRecord
    cur: PageRecord
    similarity: PageSimilarity
    hit_type: str
    passes_similarity_filter: bool = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze whether local GUI pixel changes remain local in vision encoder feature space"
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test_json", type=Path, help="AndroidControl-style JSON for dataset pair mining")
    parser.add_argument("--data_dir", type=Path, help="Defaults to the parent directory of --test_json")
    parser.add_argument("--prev_image", type=Path, help="Explicit previous screenshot for one-pair analysis")
    parser.add_argument("--cur_image", type=Path, help="Explicit current screenshot for one-pair analysis")
    parser.add_argument("--instruction", default="Analyze the GUI screenshot.")
    parser.add_argument("--output_dir", type=Path, default=Path("results/feature_locality_analysis"))
    parser.add_argument("--run_name", help="Defaults to visual_feature_locality_<timestamp>")
    parser.add_argument("--limit", type=int, help="Limit input records in --test_json mode")
    parser.add_argument("--max_pairs", type=int, help="Limit analyzed similar pairs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"),
    )
    parser.add_argument("--attn_implementation", default=None)
    parser.add_argument("--visual_token_mode", default="aggressive_reduce", choices=tuple(VISION_TOKEN_MODES))
    parser.add_argument("--min_pixels", type=int)
    parser.add_argument("--max_pixels", type=int)
    parser.add_argument("--feature_metric", default="cosine_distance", choices=("cosine_distance", "relative_l2_distance"))
    parser.add_argument("--feature_threshold", type=float, action="append", dest="feature_thresholds")
    parser.add_argument(
        "--feature_layers",
        default="final",
        help="Comma-separated visual feature layer ids. 'final' is the supported default; integer ids are best effort.",
    )
    parser.add_argument("--vision_module_path", help="Optional model module path, e.g. visual")
    parser.add_argument("--page_cache_scope", default="trajectory", choices=PAGE_CACHE_SCOPES)
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
    parser.add_argument(
        "--similar_hit_type",
        action="append",
        dest="similar_hit_types",
        choices=("exact", "near", "patch_candidate", "miss"),
        help="Hit types to analyze in dataset mode. Defaults to near and patch_candidate.",
    )
    parser.add_argument("--no_plots", action="store_true", help="Skip optional image and matplotlib visualizations")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = resolve_run_dir(args.output_dir, args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    thresholds = tuple(args.feature_thresholds or DEFAULT_FEATURE_THRESHOLDS)
    feature_layers = tuple(part.strip() for part in args.feature_layers.split(",") if part.strip())
    config = make_page_cache_config(args)
    pairs = build_pairs(args, config)
    if args.max_pairs is not None:
        pairs = pairs[: max(0, args.max_pairs)]
    if not pairs:
        raise SystemExit("No similar screenshot pairs found")

    model, processor = load_model_and_processor(
        args.model_path,
        device=args.device,
        device_map=args.device_map or None,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )

    pair_rows: List[Dict[str, Any]] = []
    distance_rows: List[Dict[str, Any]] = []
    threshold_rows: List[Dict[str, Any]] = []
    for pair_index, pair in enumerate(pairs, 1):
        print(
            f"FEATURE_LOCALITY_PAIR {pair_index}/{len(pairs)} pair={pair.pair_id} "
            f"hit={pair.hit_type} sim={pair.similarity.tile_unchanged_ratio}",
            flush=True,
        )
        rows, distances, thresholds_for_pair = analyze_pair(
            pair,
            model,
            processor,
            args,
            thresholds,
            feature_layers,
            run_dir,
        )
        pair_rows.extend(rows)
        distance_rows.extend(distances)
        threshold_rows.extend(thresholds_for_pair)

    aggregate = aggregate_results(pair_rows, threshold_rows, distance_rows, thresholds, args.feature_metric)
    plots = [] if args.no_plots else write_aggregate_plots(run_dir, pair_rows, threshold_rows, distance_rows, args.feature_metric)
    output = {
        "config": {
            "model_path": args.model_path,
            "test_json": str(args.test_json) if args.test_json else None,
            "data_dir": str(args.data_dir or args.test_json.parent) if args.test_json else None,
            "prev_image": str(args.prev_image) if args.prev_image else None,
            "cur_image": str(args.cur_image) if args.cur_image else None,
            "instruction": args.instruction,
            "output_dir": str(run_dir),
            "limit": args.limit,
            "max_pairs": args.max_pairs,
            "device": args.device,
            "device_map": args.device_map,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "visual_token_mode": args.visual_token_mode,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "feature_metric": args.feature_metric,
            "feature_thresholds": list(thresholds),
            "feature_layers": list(feature_layers),
            "vision_module_path": args.vision_module_path,
            "page_cache_config": config.to_dict(),
            "similar_hit_types": list(similar_hit_types(args)),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "summary": aggregate,
        "plots": plots,
        "notes": [
            "This is profiling/analysis only. It does not implement patch feature cache or train model weights.",
            "R_pixel is computed from existing tile hash diff; R_feature is computed from corresponding visual-token distances.",
            "Feature locality is favorable when R_feature stays near R_pixel and unchanged pixel regions have low feature distances.",
        ],
    }
    write_json(run_dir / "summary.json", output)
    write_jsonl(run_dir / "per_pair.jsonl", pair_rows)
    write_dict_rows(run_dir / "per_pair_summary.csv", flatten_pair_rows(pair_rows, args.feature_metric))
    write_dict_rows(run_dir / "threshold_stats.csv", threshold_rows)
    write_dict_rows(run_dir / "distance_profiles.csv", distance_rows)
    write_dict_rows(run_dir / "grouped_stats.csv", aggregate.get("grouped_stats", []))
    print(
        "FEATURE_LOCALITY_ANALYSIS_DONE "
        + json.dumps(
            {
                "output_dir": str(run_dir),
                "pairs": len(pairs),
                "pair_layer_rows": len(pair_rows),
                "primary_threshold": thresholds[0] if thresholds else None,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def analyze_pair(
    pair: SimilarPair,
    model: Any,
    processor: Any,
    args: argparse.Namespace,
    thresholds: Sequence[float],
    feature_layers: Sequence[str],
    run_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    distance_rows: List[Dict[str, Any]] = []
    threshold_rows: List[Dict[str, Any]] = []
    pixel_changed_ratio = ratio(pair.similarity.changed_tile_count, pair.similarity.total_tile_count)
    for layer_id in feature_layers:
        try:
            prev_features = extract_visual_features(
                model,
                processor,
                pair.prev.image_path,
                instruction=instruction_for_pair(pair, args),
                device=args.device,
                visual_token_mode=args.visual_token_mode,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
                layer_id=layer_id,
                vision_module_path=args.vision_module_path,
            )
            cur_features = extract_visual_features(
                model,
                processor,
                pair.cur.image_path,
                instruction=instruction_for_pair(pair, args),
                device=args.device,
                visual_token_mode=args.visual_token_mode,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
                layer_id=layer_id,
                vision_module_path=args.vision_module_path,
            )
            cosine, relative_l2 = feature_distances(prev_features.features, cur_features.features)
            token_count = int(cosine.shape[0])
            token_grid = infer_token_grid(cur_features.profile_metadata, token_count)
            changed_token_mask = align_changed_tiles_to_tokens(
                pair.similarity.changed_tile_mask,
                pair.similarity.total_tile_count,
                pair.similarity.changed_tile_count,
                pair.cur.fingerprint.tile_rows,
                pair.cur.fingerprint.tile_cols,
                token_count,
                token_grid,
            )
            if len(changed_token_mask) != token_count:
                raise RuntimeError("Changed-token mask length does not match feature token count")
            metric_values = {
                "cosine_distance": tensor_to_float_list(cosine),
                "relative_l2_distance": tensor_to_float_list(relative_l2),
            }
            region_stats = region_feature_stats(metric_values, changed_token_mask)
            distance_profile = build_distance_profile(metric_values, changed_token_mask, token_grid)
            primary_tau = thresholds[0] if thresholds else 0.0
            threshold_results = []
            for tau in thresholds:
                changed_ratio = ratio(
                    sum(1 for value in metric_values[args.feature_metric] if value > tau),
                    token_count,
                )
                result = {
                    "pair_id": pair.pair_id,
                    "layer_id": layer_id,
                    "threshold": tau,
                    "feature_metric": args.feature_metric,
                    "feature_changed_ratio": changed_ratio,
                    "pixel_changed_ratio": pixel_changed_ratio,
                    "page_similarity": pair.similarity.tile_unchanged_ratio,
                    "episode_id": metadata_text(pair.cur.sample, EPISODE_KEYS),
                    "step_id": metadata_text(pair.cur.sample, STEP_KEYS),
                    "app": metadata_text(pair.cur.sample, APP_NAME_KEYS),
                    "task": metadata_text(pair.cur.sample, TASK_KEYS),
                }
                threshold_results.append(result)
                threshold_rows.append(result)
            row = base_pair_row(pair, layer_id, pixel_changed_ratio, token_count)
            row.update(
                {
                    "feature_source": cur_features.feature_source,
                    "feature_metric": args.feature_metric,
                    "threshold": primary_tau,
                    "feature_changed_ratio": threshold_results[0]["feature_changed_ratio"] if threshold_results else None,
                    "feature_threshold_results": threshold_results,
                    "visual_token_count": token_count,
                    "visual_token_grid": list(token_grid) if token_grid else None,
                    "prev_profile_metadata": prev_features.profile_metadata,
                    "cur_profile_metadata": cur_features.profile_metadata,
                    "feature_distance_stats": {
                        "cosine_distance": summarize_values(metric_values["cosine_distance"]),
                        "relative_l2_distance": summarize_values(metric_values["relative_l2_distance"]),
                    },
                    "changed_unchanged_feature_distance_stats": region_stats,
                    "distance_profile": distance_profile,
                    "cache_interface_hint": {
                        "boundary": "vision_encoder_outputs",
                        "future_plan": "cached feature + changed patch recomputation + feature replacement",
                        "replaceable_token_mask_available": True,
                    },
                }
            )
            if not args.no_plots:
                row["visualizations"] = write_pair_visualizations(
                    run_dir,
                    pair,
                    layer_id,
                    metric_values[args.feature_metric],
                    changed_token_mask,
                    token_grid,
                )
            rows.append(row)
            for distance_item in distance_profile:
                distance_rows.append(
                    {
                        "pair_id": pair.pair_id,
                        "layer_id": layer_id,
                        **distance_item,
                    }
                )
        except Exception as exc:
            row = base_pair_row(pair, layer_id, pixel_changed_ratio, 0)
            row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    return rows, distance_rows, threshold_rows


def feature_distances(prev_features: Any, cur_features: Any) -> Tuple[Any, Any]:
    import torch
    import torch.nn.functional as F

    if prev_features.shape != cur_features.shape:
        raise RuntimeError(f"Feature shape mismatch: prev={tuple(prev_features.shape)} cur={tuple(cur_features.shape)}")
    prev = prev_features.float()
    cur = cur_features.float()
    cosine = 1.0 - F.cosine_similarity(prev, cur, dim=-1, eps=1e-8)
    relative_l2 = torch.linalg.norm(cur - prev, dim=-1) / torch.linalg.norm(prev, dim=-1).clamp_min(1e-8)
    return cosine.detach().cpu(), relative_l2.detach().cpu()


def build_pairs(args: argparse.Namespace, config: PageCacheConfig) -> List[SimilarPair]:
    if args.prev_image or args.cur_image:
        if not args.prev_image or not args.cur_image:
            raise SystemExit("--prev_image and --cur_image must be provided together")
        return [explicit_pair(args, config)]
    if not args.test_json:
        raise SystemExit("Provide either --test_json or --prev_image/--cur_image")
    records = load_samples(args.test_json)
    if args.limit is not None:
        records = records[: args.limit]
    data_dir = args.data_dir or args.test_json.parent
    return dataset_pairs(records, data_dir, config, similar_hit_types(args))


def explicit_pair(args: argparse.Namespace, config: PageCacheConfig) -> SimilarPair:
    prev_image = args.prev_image
    cur_image = args.cur_image
    if prev_image is None or cur_image is None:
        raise SystemExit("--prev_image and --cur_image must be provided together")
    if not prev_image.is_file():
        raise SystemExit(f"Image not found: {prev_image}")
    if not cur_image.is_file():
        raise SystemExit(f"Image not found: {cur_image}")
    prev_fp = compute_page_fingerprint(
        prev_image,
        tile_rows=config.tile_rows,
        tile_cols=config.tile_cols,
        ignored_top_ratio=config.ignored_top_ratio,
        ignored_bottom_ratio=config.ignored_bottom_ratio,
    )
    cur_fp = compute_page_fingerprint(
        cur_image,
        tile_rows=config.tile_rows,
        tile_cols=config.tile_cols,
        ignored_top_ratio=config.ignored_top_ratio,
        ignored_bottom_ratio=config.ignored_bottom_ratio,
    )
    similarity = compare_page_fingerprints(cur_fp, prev_fp)
    hit_type = cache_hit_type(similarity, config)
    return SimilarPair(
        pair_id="explicit_pair_0001",
        prev=PageRecord(1, {"image_path": str(prev_image), "task": args.instruction}, prev_image, prev_fp),
        cur=PageRecord(2, {"image_path": str(cur_image), "task": args.instruction}, cur_image, cur_fp),
        similarity=similarity,
        hit_type=hit_type,
        passes_similarity_filter=True,
    )


def dataset_pairs(
    records: Sequence[Dict[str, Any]],
    data_dir: Path,
    config: PageCacheConfig,
    allowed_hit_types: Sequence[str],
) -> List[SimilarPair]:
    history: List[PageRecord] = []
    pairs: List[SimilarPair] = []
    for index, sample in enumerate(records, 1):
        image_path = resolved_image_path(sample, data_dir)
        fingerprint = compute_page_fingerprint(
            image_path,
            tile_rows=config.tile_rows,
            tile_cols=config.tile_cols,
            ignored_top_ratio=config.ignored_top_ratio,
            ignored_bottom_ratio=config.ignored_bottom_ratio,
        )
        current = PageRecord(index, sample, image_path, fingerprint)
        best_record: Optional[PageRecord] = None
        best_similarity: Optional[PageSimilarity] = None
        best_score = (-1.0, -10000.0)
        current_episode = metadata_text(sample, EPISODE_KEYS)
        for previous in history:
            if config.scope == "trajectory" and metadata_text(previous.sample, EPISODE_KEYS) != current_episode:
                continue
            similarity = compare_page_fingerprints(fingerprint, previous.fingerprint)
            score = (float(similarity.tile_unchanged_ratio or 0.0), -float(similarity.dhash_hamming or 10000))
            if score > best_score:
                best_record = previous
                best_similarity = similarity
                best_score = score
        if best_record is not None and best_similarity is not None:
            hit_type = cache_hit_type(best_similarity, config)
            if hit_type in allowed_hit_types:
                pairs.append(
                    SimilarPair(
                        pair_id=f"pair_{len(pairs) + 1:06d}",
                        prev=best_record,
                        cur=current,
                        similarity=best_similarity,
                        hit_type=hit_type,
                    )
                )
        history.append(current)
    return pairs


def cache_hit_type(similarity: PageSimilarity, config: PageCacheConfig) -> str:
    if similarity.exact:
        return "exact"
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


def align_changed_tiles_to_tokens(
    changed_tile_mask: Sequence[bool],
    total_tile_count: int,
    changed_tile_count: int,
    tile_rows: int,
    tile_cols: int,
    token_count: int,
    token_grid: Optional[Tuple[int, int]],
) -> List[bool]:
    if not changed_tile_mask and changed_tile_count == 0:
        return [False] * token_count
    if total_tile_count == token_count and len(changed_tile_mask) == token_count:
        return [bool(value) for value in changed_tile_mask]
    if token_grid is None:
        raise RuntimeError("Cannot align tile mask to visual tokens without token grid")
    token_rows, token_cols = token_grid
    result = []
    for row in range(token_rows):
        y = (row + 0.5) / max(1, token_rows)
        tile_row = min(tile_rows - 1, int(y * tile_rows))
        for col in range(token_cols):
            x = (col + 0.5) / max(1, token_cols)
            tile_col = min(tile_cols - 1, int(x * tile_cols))
            tile_index = tile_row * tile_cols + tile_col
            result.append(bool(changed_tile_mask[tile_index]) if tile_index < len(changed_tile_mask) else False)
    if len(result) != token_count:
        raise RuntimeError(f"Token grid {token_grid} does not match token_count={token_count}")
    return result


def infer_token_grid(metadata: Dict[str, Any], token_count: int) -> Optional[Tuple[int, int]]:
    grids = metadata.get("image_grid_thw") or []
    merge_size = int(metadata.get("processor_merge_size") or 1)
    if len(grids) != 1:
        return squareish_grid(token_count)
    grid = grids[0]
    if not isinstance(grid, (list, tuple)) or len(grid) < 3:
        return squareish_grid(token_count)
    t, h, w = int(grid[0]), int(grid[1]), int(grid[2])
    candidates = []
    if merge_size > 1:
        candidates.append((max(1, t * (h // merge_size)), max(1, w // merge_size)))
    candidates.append((max(1, t * h), max(1, w)))
    for rows, cols in candidates:
        if rows * cols == token_count:
            return rows, cols
    return squareish_grid(token_count)


def squareish_grid(token_count: int) -> Optional[Tuple[int, int]]:
    if token_count <= 0:
        return None
    root = int(math.sqrt(token_count))
    for rows in range(root, 0, -1):
        if token_count % rows == 0:
            return rows, token_count // rows
    return None


def region_feature_stats(metric_values: Dict[str, List[float]], changed_mask: Sequence[bool]) -> Dict[str, Any]:
    result = {}
    for metric, values in metric_values.items():
        changed = [value for value, is_changed in zip(values, changed_mask) if is_changed]
        unchanged = [value for value, is_changed in zip(values, changed_mask) if not is_changed]
        result[metric] = {
            "changed_pixel_patch_tokens": summarize_values(changed),
            "unchanged_pixel_patch_tokens": summarize_values(unchanged),
        }
    return result


def build_distance_profile(
    metric_values: Dict[str, List[float]],
    changed_mask: Sequence[bool],
    token_grid: Optional[Tuple[int, int]],
) -> List[Dict[str, Any]]:
    if token_grid is None:
        return []
    rows, cols = token_grid
    changed_positions = [(index // cols, index % cols) for index, changed in enumerate(changed_mask) if changed]
    if not changed_positions:
        return [
            {
                "token_manhattan_distance": None,
                "token_count": len(changed_mask),
                "cosine_distance": summarize_values(metric_values["cosine_distance"]),
                "relative_l2_distance": summarize_values(metric_values["relative_l2_distance"]),
            }
        ]
    grouped: Dict[int, Dict[str, List[float]]] = {}
    for index in range(rows * cols):
        row, col = index // cols, index % cols
        distance = min(abs(row - cr) + abs(col - cc) for cr, cc in changed_positions)
        bucket = grouped.setdefault(distance, {"cosine_distance": [], "relative_l2_distance": []})
        for metric in bucket:
            bucket[metric].append(metric_values[metric][index])
    return [
        {
            "token_manhattan_distance": distance,
            "token_count": len(values["cosine_distance"]),
            "cosine_distance": summarize_values(values["cosine_distance"]),
            "relative_l2_distance": summarize_values(values["relative_l2_distance"]),
        }
        for distance, values in sorted(grouped.items())
    ]


def write_pair_visualizations(
    run_dir: Path,
    pair: SimilarPair,
    layer_id: str,
    feature_values: Sequence[float],
    changed_token_mask: Sequence[bool],
    token_grid: Optional[Tuple[int, int]],
) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    pair_dir = run_dir / "visualizations"
    pair_dir.mkdir(parents=True, exist_ok=True)
    pixel_path = pair_dir / f"{pair.pair_id}_{safe_name(layer_id)}_pixel_diff_heatmap.png"
    feature_path = pair_dir / f"{pair.pair_id}_{safe_name(layer_id)}_feature_diff_heatmap.png"
    write_pixel_diff_heatmap(
        pair.cur.image_path,
        pair.similarity,
        pair.cur.fingerprint.tile_rows,
        pair.cur.fingerprint.tile_cols,
        pair.cur.fingerprint.ignored_top_ratio,
        pair.cur.fingerprint.ignored_bottom_ratio,
        pixel_path,
    )
    paths["pixel_diff_heatmap"] = str(pixel_path)
    if token_grid is not None:
        write_feature_diff_heatmap(feature_values, changed_token_mask, token_grid, feature_path)
        paths["feature_diff_heatmap"] = str(feature_path)
    return paths


def write_pixel_diff_heatmap(
    image_path: Path,
    similarity: PageSimilarity,
    tile_rows: int,
    tile_cols: int,
    ignored_top_ratio: float,
    ignored_bottom_ratio: float,
    output_path: Path,
) -> None:
    from PIL import Image, ImageDraw

    with Image.open(image_path) as source:
        image = source.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    active_top = int(max(0.0, min(0.5, ignored_top_ratio)) * height)
    active_bottom_crop = int(max(0.0, min(0.5, ignored_bottom_ratio)) * height)
    active_bottom = max(active_top + 1, height - active_bottom_crop)
    active_height = active_bottom - active_top
    for index, changed in enumerate(similarity.changed_tile_mask):
        if not changed:
            continue
        row = index // tile_cols
        col = index % tile_cols
        left = round(col * width / tile_cols)
        top = active_top + round(row * active_height / tile_rows)
        right = round((col + 1) * width / tile_cols)
        bottom = active_top + round((row + 1) * active_height / tile_rows)
        draw.rectangle((left, top, right, bottom), fill=(255, 0, 0, 90), outline=(255, 0, 0, 180))
    Image.alpha_composite(image, overlay).convert("RGB").save(output_path)


def write_feature_diff_heatmap(
    values: Sequence[float],
    changed_token_mask: Sequence[bool],
    token_grid: Tuple[int, int],
    output_path: Path,
) -> None:
    from PIL import Image, ImageDraw

    rows, cols = token_grid
    cell = 18
    image = Image.new("RGB", (cols * cell, rows * cell), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    max_value = max([float(value) for value in values] or [1.0])
    for index, value in enumerate(values):
        row = index // cols
        col = index % cols
        strength = 0 if max_value <= 0 else int(max(0.0, min(1.0, float(value) / max_value)) * 255)
        color = (255, 255 - strength, 255 - strength)
        left, top = col * cell, row * cell
        draw.rectangle((left, top, left + cell - 1, top + cell - 1), fill=color)
        if index < len(changed_token_mask) and changed_token_mask[index]:
            draw.rectangle((left, top, left + cell - 1, top + cell - 1), outline=(0, 80, 255), width=2)
    image.save(output_path)


def write_aggregate_plots(
    run_dir: Path,
    pair_rows: Sequence[Dict[str, Any]],
    threshold_rows: Sequence[Dict[str, Any]],
    distance_rows: Sequence[Dict[str, Any]],
    feature_metric: str,
) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    written: List[str] = []
    valid_thresholds = [row for row in threshold_rows if row.get("feature_changed_ratio") is not None]
    if valid_thresholds:
        path = run_dir / "r_pixel_vs_r_feature.png"
        plt.figure(figsize=(6, 5))
        plt.scatter(
            [row["pixel_changed_ratio"] for row in valid_thresholds],
            [row["feature_changed_ratio"] for row in valid_thresholds],
            alpha=0.7,
        )
        plt.xlabel("R_pixel")
        plt.ylabel("R_feature")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        written.append(str(path))
    valid_rows = [row for row in pair_rows if row.get("pixel_changed_ratio") is not None and row.get("feature_changed_ratio") is not None]
    if valid_rows:
        path = run_dir / "r_pixel_r_feature_distribution.png"
        plt.figure(figsize=(7, 4))
        plt.hist([row["pixel_changed_ratio"] for row in valid_rows], bins=20, alpha=0.6, label="R_pixel")
        plt.hist([row["feature_changed_ratio"] for row in valid_rows], bins=20, alpha=0.6, label="R_feature")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        written.append(str(path))
    distance_means: Dict[Any, List[float]] = {}
    for row in distance_rows:
        stats = row.get(feature_metric) or {}
        if row.get("token_manhattan_distance") is not None and stats.get("mean") is not None:
            distance_means.setdefault(row["token_manhattan_distance"], []).append(float(stats["mean"]))
    if distance_means:
        path = run_dir / "feature_distance_by_changed_region_distance.png"
        xs = sorted(distance_means)
        ys = [statistics.fmean(distance_means[x]) for x in xs]
        plt.figure(figsize=(7, 4))
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Token Manhattan distance to changed region")
        plt.ylabel(feature_metric)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        written.append(str(path))
    return written


def aggregate_results(
    pair_rows: Sequence[Dict[str, Any]],
    threshold_rows: Sequence[Dict[str, Any]],
    distance_rows: Sequence[Dict[str, Any]],
    thresholds: Sequence[float],
    feature_metric: str,
) -> Dict[str, Any]:
    valid_rows = [row for row in pair_rows if not row.get("error")]
    valid_thresholds = [row for row in threshold_rows if row.get("feature_changed_ratio") is not None]
    return {
        "pair_layer_rows": len(pair_rows),
        "valid_pair_layer_rows": len(valid_rows),
        "error_count": len(pair_rows) - len(valid_rows),
        "r_pixel_distribution": summarize_values([row.get("pixel_changed_ratio") for row in valid_rows]),
        "r_feature_by_threshold": {
            str(tau): summarize_values(
                [row.get("feature_changed_ratio") for row in valid_thresholds if float(row.get("threshold")) == float(tau)]
            )
            for tau in thresholds
        },
        "r_pixel_r_feature_correlation_by_threshold": {
            str(tau): pearson(
                [row["pixel_changed_ratio"] for row in valid_thresholds if float(row.get("threshold")) == float(tau)],
                [row["feature_changed_ratio"] for row in valid_thresholds if float(row.get("threshold")) == float(tau)],
            )
            for tau in thresholds
        },
        "by_page_similarity_bin": summarize_similarity_bins(valid_thresholds),
        "grouped_stats": grouped_stats(valid_thresholds),
        "by_layer": grouped_by(valid_thresholds, "layer_id"),
        "distance_to_changed_region": summarize_distance_rows(distance_rows, feature_metric),
        "decision_signals": build_decision_signals(valid_rows, valid_thresholds, feature_metric),
    }


def summarize_similarity_bins(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    for left, right in DEFAULT_SIMILARITY_BINS:
        selected = [
            row
            for row in rows
            if row.get("page_similarity") is not None and left <= float(row["page_similarity"]) < right
        ]
        result.append(summary_row("page_similarity_bin", f"[{left:.2f},{min(right, 1.0):.2f}]", selected))
    return result


def grouped_stats(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for key in ("app", "task", "episode_id"):
        output.extend(grouped_by(rows, key))
    return output


def grouped_by(rows: Sequence[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        value = str(row.get(key) or "unknown")
        layer_id = str(row.get("layer_id") or "unknown")
        threshold = str(row.get("threshold") or "unknown")
        groups.setdefault((value, layer_id, threshold), []).append(row)
    return [
        summary_row(key, value, selected, layer_id=layer_id, threshold=threshold)
        for (value, layer_id, threshold), selected in sorted(groups.items())
    ]


def summary_row(
    dimension: str,
    value: str,
    rows: Sequence[Dict[str, Any]],
    layer_id: Optional[str] = None,
    threshold: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "dimension": dimension,
        "value": value,
        "layer_id": layer_id,
        "threshold": threshold,
        "feature_metric": rows[0].get("feature_metric") if rows else None,
        "pair_count": len(rows),
        "pixel_changed_ratio": summarize_values([row.get("pixel_changed_ratio") for row in rows]),
        "feature_changed_ratio": summarize_values([row.get("feature_changed_ratio") for row in rows]),
        "correlation": pearson(
            [row["pixel_changed_ratio"] for row in rows if row.get("pixel_changed_ratio") is not None and row.get("feature_changed_ratio") is not None],
            [row["feature_changed_ratio"] for row in rows if row.get("pixel_changed_ratio") is not None and row.get("feature_changed_ratio") is not None],
        ),
    }


def summarize_distance_rows(rows: Sequence[Dict[str, Any]], feature_metric: str) -> List[Dict[str, Any]]:
    grouped: Dict[Any, List[float]] = {}
    for row in rows:
        stats = row.get(feature_metric) or {}
        if stats.get("mean") is not None:
            grouped.setdefault(row.get("token_manhattan_distance"), []).append(float(stats["mean"]))
    return [
        {
            "token_manhattan_distance": distance,
            "pair_distance_bin_count": len(values),
            "feature_distance": summarize_values(values),
        }
        for distance, values in sorted(grouped.items(), key=lambda item: -1 if item[0] is None else int(item[0]))
    ]


def build_decision_signals(
    pair_rows: Sequence[Dict[str, Any]],
    threshold_rows: Sequence[Dict[str, Any]],
    feature_metric: str,
) -> Dict[str, Any]:
    del feature_metric
    if not pair_rows:
        return {}
    inflation = [
        row["feature_changed_ratio"] / max(row["pixel_changed_ratio"], 1e-8)
        for row in pair_rows
        if row.get("feature_changed_ratio") is not None and row.get("pixel_changed_ratio") is not None
    ]
    unchanged_means = []
    changed_means = []
    for row in pair_rows:
        stats = ((row.get("changed_unchanged_feature_distance_stats") or {}).get(row.get("feature_metric")) or {})
        unchanged = (stats.get("unchanged_pixel_patch_tokens") or {}).get("mean")
        changed = (stats.get("changed_pixel_patch_tokens") or {}).get("mean")
        if unchanged is not None:
            unchanged_means.append(float(unchanged))
        if changed is not None:
            changed_means.append(float(changed))
    return {
        "r_feature_over_r_pixel": summarize_values(inflation),
        "unchanged_pixel_region_feature_distance_mean": summarize_values(unchanged_means),
        "changed_pixel_region_feature_distance_mean": summarize_values(changed_means),
        "threshold_row_count": len(threshold_rows),
    }


def base_pair_row(pair: SimilarPair, layer_id: str, pixel_changed_ratio: float, token_count: int) -> Dict[str, Any]:
    return {
        "pair_id": pair.pair_id,
        "layer_id": layer_id,
        "episode_id": metadata_text(pair.cur.sample, EPISODE_KEYS),
        "step_id": metadata_text(pair.cur.sample, STEP_KEYS),
        "app": metadata_text(pair.cur.sample, APP_NAME_KEYS),
        "task": metadata_text(pair.cur.sample, TASK_KEYS),
        "prev_index": pair.prev.index,
        "cur_index": pair.cur.index,
        "prev_episode_id": metadata_text(pair.prev.sample, EPISODE_KEYS),
        "prev_step_id": metadata_text(pair.prev.sample, STEP_KEYS),
        "prev_image_path": str(pair.prev.image_path),
        "cur_image_path": str(pair.cur.image_path),
        "hit_type": pair.hit_type,
        "passes_similarity_filter": pair.passes_similarity_filter,
        "page_similarity": {
            "exact": pair.similarity.exact,
            "dhash_hamming": pair.similarity.dhash_hamming,
            "tile_unchanged_ratio": pair.similarity.tile_unchanged_ratio,
            "changed_bbox": list(pair.similarity.changed_bbox) if pair.similarity.changed_bbox else None,
            "changed_bbox_pixels": list(pair.similarity.changed_bbox_pixels) if pair.similarity.changed_bbox_pixels else None,
            "changed_bbox_area_ratio": pair.similarity.changed_bbox_area_ratio,
        },
        "pixel_changed_ratio": pixel_changed_ratio,
        "pixel_changed_patch_count": pair.similarity.changed_tile_count,
        "pixel_total_patch_count": pair.similarity.total_tile_count,
        "pixel_changed_tile_indices": list(pair.similarity.changed_tile_indices),
        "visual_token_count": token_count,
    }


def flatten_pair_rows(rows: Sequence[Dict[str, Any]], feature_metric: str) -> List[Dict[str, Any]]:
    flat = []
    for row in rows:
        stats = ((row.get("changed_unchanged_feature_distance_stats") or {}).get(feature_metric) or {})
        changed = stats.get("changed_pixel_patch_tokens") or {}
        unchanged = stats.get("unchanged_pixel_patch_tokens") or {}
        flat.append(
            {
                "pair_id": row.get("pair_id"),
                "layer_id": row.get("layer_id"),
                "episode_id": row.get("episode_id"),
                "step_id": row.get("step_id"),
                "app": row.get("app"),
                "hit_type": row.get("hit_type"),
                "pixel_changed_ratio": row.get("pixel_changed_ratio"),
                "feature_metric": row.get("feature_metric"),
                "threshold": row.get("threshold"),
                "feature_changed_ratio": row.get("feature_changed_ratio"),
                "visual_token_count": row.get("visual_token_count"),
                "page_tile_unchanged_ratio": (row.get("page_similarity") or {}).get("tile_unchanged_ratio"),
                "changed_region_feature_mean": changed.get("mean"),
                "unchanged_region_feature_mean": unchanged.get("mean"),
                "error": row.get("error"),
            }
        )
    return flat


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_dict_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        flat = json_safe_flat(row)
        for key in flat:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(json_safe_flat(row))


def json_safe_flat(row: Dict[str, Any]) -> Dict[str, Any]:
    flat = {}
    for key, value in row.items():
        if isinstance(value, (dict, list, tuple)):
            flat[key] = json.dumps(value, ensure_ascii=False)
        else:
            flat[key] = value
    return flat


def summarize_values(values: Iterable[Any]) -> Dict[str, Any]:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return {"count": 0}
    mean = statistics.fmean(numeric)
    return {
        "count": len(numeric),
        "mean": mean,
        "median": statistics.median(numeric),
        "p90": percentile(numeric, 90),
        "min": min(numeric),
        "max": max(numeric),
        "std": math.sqrt(statistics.fmean([(value - mean) ** 2 for value in numeric])) if len(numeric) > 1 else 0.0,
    }


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


def pearson(xs: Sequence[Any], ys: Sequence[Any]) -> Optional[float]:
    x_vals = [float(value) for value in xs if value is not None]
    y_vals = [float(value) for value in ys if value is not None]
    if len(x_vals) != len(y_vals) or len(x_vals) < 2:
        return None
    x_mean = statistics.fmean(x_vals)
    y_mean = statistics.fmean(y_vals)
    x_var = sum((value - x_mean) ** 2 for value in x_vals)
    y_var = sum((value - y_mean) ** 2 for value in y_vals)
    if x_var <= 0.0 or y_var <= 0.0:
        return None
    covariance = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
    return covariance / math.sqrt(x_var * y_var)


def tensor_to_float_list(tensor: Any) -> List[float]:
    return [float(value) for value in tensor.reshape(-1).tolist()]


def ratio(count: int, total: int) -> float:
    return count / total if total else 0.0


def metadata_text(sample: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    return optional_text(metadata_value(sample, keys))


def instruction_for_pair(pair: SimilarPair, args: argparse.Namespace) -> str:
    return str(pair.cur.sample.get("task") or args.instruction)


def make_page_cache_config(args: argparse.Namespace) -> PageCacheConfig:
    return PageCacheConfig(
        mode="observe",
        scope=args.page_cache_scope,
        similarity=args.page_cache_similarity,
        max_entries=1,
        near_dhash_threshold=args.page_cache_near_dhash_threshold,
        near_tile_threshold=args.page_cache_near_tile_threshold,
        patch_tile_threshold=args.page_cache_patch_tile_threshold,
        patch_max_changed_area_ratio=args.page_cache_patch_max_changed_area_ratio,
        patch_critical_regions=parse_normalized_bboxes(args.page_cache_patch_critical_region),
        tile_rows=args.page_cache_tile_rows,
        tile_cols=args.page_cache_tile_cols,
        ignored_top_ratio=args.page_cache_ignored_top_ratio,
        ignored_bottom_ratio=args.page_cache_ignored_bottom_ratio,
        identity="visual-feature-locality",
    )


def similar_hit_types(args: argparse.Namespace) -> Tuple[str, ...]:
    return tuple(args.similar_hit_types or ("near", "patch_candidate"))


def bbox_intersects(
    left_bbox: Tuple[float, float, float, float],
    right_bbox: Tuple[float, float, float, float],
) -> bool:
    left_a, top_a, right_a, bottom_a = left_bbox
    left_b, top_b, right_b, bottom_b = right_bbox
    return max(left_a, left_b) < min(right_a, right_b) and max(top_a, top_b) < min(bottom_a, bottom_b)


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))


def resolve_run_dir(output_dir: Path, run_name: Optional[str]) -> Path:
    if run_name:
        return output_dir / run_name
    if output_dir.name != "feature_locality_analysis":
        return output_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"visual_feature_locality_{timestamp}"


if __name__ == "__main__":
    raise SystemExit(main())
