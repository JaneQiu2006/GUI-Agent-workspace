from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
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

from analyze_visual_feature_locality import (  # noqa: E402
    APP_NAME_KEYS,
    EPISODE_KEYS,
    STEP_KEYS,
    TASK_KEYS,
    align_changed_tiles_to_tokens,
    build_distance_profile,
    build_pairs,
    infer_token_grid_details,
    instruction_for_pair,
    make_page_cache_config,
    metadata_text,
    pearson,
    ratio,
    safe_name,
    similar_hit_types,
    summarize_values,
    write_dict_rows,
    write_json,
    write_jsonl,
    write_pixel_diff_heatmap,
)
from cache_inference import PAGE_CACHE_SCOPES, PAGE_CACHE_SIMILARITIES  # noqa: E402
from hf_gui_baseline import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    VISION_TOKEN_MODES,
    extract_prefill_past_key_values,
    load_model_and_processor,
)


DEFAULT_KV_THRESHOLDS = (0.005, 0.01, 0.03, 0.05, 0.10)
KV_BOUNDARY = "prefill_past_key_values"
KV_SCOPES = ("visual_tokens", "text_tokens", "full_prefix")
KV_KINDS = ("key", "value")
KV_METRICS = ("cosine_distance", "relative_l2_distance")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze whether local GUI pixel changes remain local in prefill past_key_values"
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test_json", type=Path, help="AndroidControl-style JSON for dataset pair mining")
    parser.add_argument("--data_dir", type=Path, help="Defaults to the parent directory of --test_json")
    parser.add_argument("--prev_image", type=Path, help="Explicit previous screenshot for one-pair analysis")
    parser.add_argument("--cur_image", type=Path, help="Explicit current screenshot for one-pair analysis")
    parser.add_argument("--instruction", default="Analyze the GUI screenshot.")
    parser.add_argument("--output_dir", type=Path, default=Path("results/kv_locality_analysis"))
    parser.add_argument("--run_name", help="Defaults to kv_locality_<timestamp>")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max_pairs", type=int)
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
    parser.add_argument("--kv_metric", default="cosine_distance", choices=KV_METRICS)
    parser.add_argument("--kv_threshold", type=float, action="append", dest="kv_thresholds")
    parser.add_argument("--kv_scope", action="append", choices=KV_SCOPES, dest="kv_scopes")
    parser.add_argument("--kv_kind", action="append", choices=KV_KINDS, dest="kv_kinds")
    parser.add_argument("--kv_layer", type=int, action="append", dest="kv_layers", help="Layer id to analyze; repeatable. Defaults to all layers.")
    parser.add_argument("--page_cache_scope", default="trajectory", choices=PAGE_CACHE_SCOPES)
    parser.add_argument("--page_cache_similarity", default="tile", choices=PAGE_CACHE_SIMILARITIES)
    parser.add_argument("--page_cache_near_dhash_threshold", type=int, default=4)
    parser.add_argument("--page_cache_near_tile_threshold", type=float, default=0.98)
    parser.add_argument("--page_cache_patch_tile_threshold", type=float, default=0.90)
    parser.add_argument("--page_cache_patch_max_changed_area_ratio", type=float, default=0.25)
    parser.add_argument("--page_cache_patch_critical_region", action="append", default=[])
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
    parser.add_argument("--no_plots", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = resolve_run_dir(args.output_dir, args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    thresholds = tuple(args.kv_thresholds or DEFAULT_KV_THRESHOLDS)
    kv_scopes = tuple(args.kv_scopes or ("visual_tokens",))
    kv_kinds = tuple(args.kv_kinds or KV_KINDS)
    selected_layers = set(args.kv_layers or [])
    config = make_page_cache_config(args)
    config.identity = "kv-locality"
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
            f"KV_LOCALITY_PAIR {pair_index}/{len(pairs)} pair={pair.pair_id} "
            f"hit={pair.hit_type} sim={pair.similarity.tile_unchanged_ratio}",
            flush=True,
        )
        rows, distances, thresholds_for_pair = analyze_pair(
            pair,
            model,
            processor,
            args,
            thresholds,
            kv_scopes,
            kv_kinds,
            selected_layers,
            run_dir,
        )
        pair_rows.extend(rows)
        distance_rows.extend(distances)
        threshold_rows.extend(thresholds_for_pair)

    aggregate = aggregate_results(pair_rows, threshold_rows, distance_rows, thresholds, args.kv_metric)
    plots = [] if args.no_plots else write_aggregate_plots(run_dir, threshold_rows, distance_rows, args.kv_metric)
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
            "kv_metric": args.kv_metric,
            "kv_thresholds": list(thresholds),
            "kv_scopes": list(kv_scopes),
            "kv_kinds": list(kv_kinds),
            "kv_layers": sorted(selected_layers) if selected_layers else "all",
            "kv_boundary": KV_BOUNDARY,
            "page_cache_config": config.to_dict(),
            "similar_hit_types": list(similar_hit_types(args)),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "summary": aggregate,
        "plots": plots,
        "notes": [
            "This is profiling/analysis only. It does not implement patch KV cache, train weights, or modify model weights.",
            "KV tensors are per-LLM-layer past_key_values, not visual embeddings.",
            "Visual token positions are inferred from input_ids matching tokenizer image_token_id/vision_token_id; failures are reported per pair.",
            "Patch-local KV reuse is favorable only if R_kv stays near R_pixel and unchanged visual regions keep low KV distance across deeper layers.",
        ],
    }
    write_json(run_dir / "summary.json", output)
    write_jsonl(run_dir / "per_pair.jsonl", pair_rows)
    write_dict_rows(run_dir / "per_pair_summary.csv", flatten_pair_rows(pair_rows, args.kv_metric))
    write_dict_rows(run_dir / "threshold_stats.csv", threshold_rows)
    write_dict_rows(run_dir / "distance_profiles.csv", distance_rows)
    write_dict_rows(run_dir / "grouped_stats.csv", aggregate.get("grouped_stats", []))
    print(
        "KV_LOCALITY_ANALYSIS_DONE "
        + json.dumps(
            {
                "output_dir": str(run_dir),
                "pairs": len(pairs),
                "pair_scope_layer_kind_rows": len(pair_rows),
                "primary_threshold": thresholds[0] if thresholds else None,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def analyze_pair(
    pair: Any,
    model: Any,
    processor: Any,
    args: argparse.Namespace,
    thresholds: Sequence[float],
    kv_scopes: Sequence[str],
    kv_kinds: Sequence[str],
    selected_layers: set[int],
    run_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    distance_rows: List[Dict[str, Any]] = []
    threshold_rows: List[Dict[str, Any]] = []
    pixel_changed_ratio = ratio(pair.similarity.changed_tile_count, pair.similarity.total_tile_count)
    try:
        prev = extract_prefill_past_key_values(
            model,
            processor,
            pair.prev.image_path,
            instruction=instruction_for_pair(pair, args),
            device=args.device,
            visual_token_mode=args.visual_token_mode,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
        cur = extract_prefill_past_key_values(
            model,
            processor,
            pair.cur.image_path,
            instruction=instruction_for_pair(pair, args),
            device=args.device,
            visual_token_mode=args.visual_token_mode,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
        prev_layers = list(iter_kv_layers(prev.past_key_values))
        cur_layers = list(iter_kv_layers(cur.past_key_values))
        if len(prev_layers) != len(cur_layers):
            raise RuntimeError(f"KV layer count mismatch: prev={len(prev_layers)} cur={len(cur_layers)}")
        for layer_id, ((prev_k, prev_v), (cur_k, cur_v)) in enumerate(zip(prev_layers, cur_layers)):
            if selected_layers and layer_id not in selected_layers:
                continue
            for kv_kind in kv_kinds:
                prev_tensor = prev_k if kv_kind == "key" else prev_v
                cur_tensor = cur_k if kv_kind == "key" else cur_v
                prev_matrix = kv_tensor_to_token_matrix(prev_tensor, expected_seq_len=prev.input_tokens)
                cur_matrix = kv_tensor_to_token_matrix(cur_tensor, expected_seq_len=cur.input_tokens)
                for kv_scope in kv_scopes:
                    scope_positions = comparable_scope_positions(prev, cur, kv_scope)
                    row, distances, threshold_for_scope = analyze_scope(
                        pair,
                        args,
                        prev,
                        cur,
                        prev_matrix,
                        cur_matrix,
                        scope_positions,
                        layer_id,
                        kv_kind,
                        kv_scope,
                        pixel_changed_ratio,
                        thresholds,
                        run_dir,
                    )
                    rows.append(row)
                    distance_rows.extend(distances)
                    threshold_rows.extend(threshold_for_scope)
    except Exception as exc:
        for kv_scope in kv_scopes:
            for kv_kind in kv_kinds:
                row = base_kv_row(pair, None, kv_kind, kv_scope, pixel_changed_ratio)
                row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
    return rows, distance_rows, threshold_rows


def analyze_scope(
    pair: Any,
    args: argparse.Namespace,
    prev: Any,
    cur: Any,
    prev_matrix: Any,
    cur_matrix: Any,
    positions: Sequence[int],
    layer_id: int,
    kv_kind: str,
    kv_scope: str,
    pixel_changed_ratio: float,
    thresholds: Sequence[float],
    run_dir: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    import torch
    import torch.nn.functional as F

    if prev_matrix.shape != cur_matrix.shape:
        raise RuntimeError(f"KV token matrix shape mismatch: prev={tuple(prev_matrix.shape)} cur={tuple(cur_matrix.shape)}")
    if not positions:
        raise RuntimeError(f"No comparable positions for kv_scope={kv_scope}")
    max_position = max(int(position) for position in positions)
    if max_position >= int(prev_matrix.shape[0]):
        raise RuntimeError(
            f"{kv_scope} position {max_position} exceeds KV sequence length {int(prev_matrix.shape[0])}"
        )
    position_tensor = torch.tensor(list(positions), dtype=torch.long)
    prev_selected = prev_matrix.index_select(0, position_tensor).float()
    cur_selected = cur_matrix.index_select(0, position_tensor).float()
    cosine = 1.0 - F.cosine_similarity(prev_selected, cur_selected, dim=-1, eps=1e-8)
    relative_l2 = torch.linalg.norm(cur_selected - prev_selected, dim=-1) / torch.linalg.norm(prev_selected, dim=-1).clamp_min(1e-8)
    metric_values = {
        "cosine_distance": tensor_to_float_list(cosine.detach().cpu()),
        "relative_l2_distance": tensor_to_float_list(relative_l2.detach().cpu()),
    }
    token_count = len(positions)
    alignment = infer_token_grid_details(cur.profile_metadata, len(cur.visual_token_positions), "model_ready_visual")
    token_grid = alignment["visual_token_grid"]
    changed_token_mask: Optional[List[bool]] = None
    region_stats: Dict[str, Any] = {}
    distance_profile: List[Dict[str, Any]] = []
    if kv_scope == "visual_tokens":
        changed_token_mask = align_changed_tiles_to_tokens(
            pair.similarity.changed_tile_mask,
            pair.similarity.total_tile_count,
            pair.similarity.changed_tile_count,
            pair.cur.fingerprint.tile_rows,
            pair.cur.fingerprint.tile_cols,
            token_count,
            token_grid,
        )
        region_stats = region_kv_stats(metric_values, changed_token_mask)
        distance_profile = build_distance_profile(metric_values, changed_token_mask, token_grid)

    primary_tau = thresholds[0] if thresholds else 0.0
    threshold_results = []
    for tau in thresholds:
        changed_ratio = ratio(sum(1 for value in metric_values[args.kv_metric] if value > tau), token_count)
        result = threshold_row(pair, args, cur, layer_id, kv_kind, kv_scope, token_count, pixel_changed_ratio, tau, changed_ratio, alignment)
        threshold_results.append(result)
    row = base_kv_row(pair, layer_id, kv_kind, kv_scope, pixel_changed_ratio)
    row.update(
        {
            "kv_boundary": cur.kv_boundary,
            "feature_boundary": "prefill_after_llm_self_attention",
            "visual_position_source": cur.visual_position_source,
            "kv_token_count": token_count,
            "visual_token_grid": list(token_grid) if token_grid else None,
            "processor_merge_size": alignment["processor_merge_size"],
            "alignment_method": alignment["alignment_method"] if kv_scope == "visual_tokens" else "not_spatial_scope",
            "threshold": primary_tau,
            "kv_metric": args.kv_metric,
            "kv_changed_ratio": threshold_results[0]["kv_changed_ratio"] if threshold_results else None,
            "kv_threshold_results": threshold_results,
            "distance_stats": {
                "cosine_distance": summarize_values(metric_values["cosine_distance"]),
                "relative_l2_distance": summarize_values(metric_values["relative_l2_distance"]),
            },
            "changed_unchanged_kv_distance_stats": region_stats,
            "distance_profile": distance_profile,
            "input_token_count": cur.input_tokens,
            "visual_token_position_count": len(cur.visual_token_positions),
            "text_token_position_count": len(cur.text_token_positions),
            "prev_profile_metadata": prev.profile_metadata,
            "cur_profile_metadata": cur.profile_metadata,
            "prev_kv_shapes": prev.kv_shapes,
            "cur_kv_shapes": cur.kv_shapes,
        }
    )
    if kv_scope == "visual_tokens" and not args.no_plots and changed_token_mask is not None:
        row["visualizations"] = write_pair_visualizations(
            run_dir,
            pair,
            layer_id,
            kv_kind,
            metric_values[args.kv_metric],
            changed_token_mask,
            token_grid,
        )
    distance_rows = [
        {
            "pair_id": pair.pair_id,
            "layer_id": layer_id,
            "kv_kind": kv_kind,
            "kv_scope": kv_scope,
            "kv_boundary": cur.kv_boundary,
            "visual_token_grid": list(token_grid) if token_grid else None,
            "alignment_method": alignment["alignment_method"] if kv_scope == "visual_tokens" else "not_spatial_scope",
            **item,
        }
        for item in distance_profile
    ]
    return row, distance_rows, threshold_results


def iter_kv_layers(past_key_values: Any) -> Iterable[Tuple[Any, Any]]:
    if hasattr(past_key_values, "to_legacy_cache"):
        try:
            past_key_values = past_key_values.to_legacy_cache()
        except Exception:
            pass
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        yield from zip(past_key_values.key_cache, past_key_values.value_cache)
        return
    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)
            if key is None or value is None:
                raise RuntimeError("Cache layer is missing keys or values tensor")
            yield key, value
        return
    if hasattr(past_key_values, "__iter__") and not isinstance(past_key_values, (list, tuple, dict)):
        yielded = False
        for item in past_key_values:
            if isinstance(item, dict):
                key = item.get("key") if item.get("key") is not None else item.get("k")
                value = item.get("value") if item.get("value") is not None else item.get("v")
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                key, value = item[0], item[1]
            else:
                raise RuntimeError(f"Unsupported cache iterator item type: {type(item).__name__}")
            if key is None or value is None:
                raise RuntimeError("Cache iterator item is missing key or value tensor")
            yielded = True
            yield key, value
        if yielded:
            return
    if not isinstance(past_key_values, (list, tuple)):
        raise RuntimeError(f"Unsupported past_key_values type: {type(past_key_values).__name__}")
    for layer in past_key_values:
        if isinstance(layer, dict):
            key = layer.get("key") if layer.get("key") is not None else layer.get("k")
            value = layer.get("value") if layer.get("value") is not None else layer.get("v")
        elif isinstance(layer, (list, tuple)) and len(layer) >= 2:
            key, value = layer[0], layer[1]
        else:
            raise RuntimeError(f"Unsupported KV layer type: {type(layer).__name__}")
        if key is None or value is None:
            raise RuntimeError("KV layer is missing key or value tensor")
        yield key, value


def kv_tensor_to_token_matrix(tensor: Any, expected_seq_len: Optional[int] = None) -> Any:
    import torch

    if not hasattr(tensor, "shape"):
        raise RuntimeError(f"KV tensor has no shape: {type(tensor).__name__}")
    value = tensor.detach().float().cpu() if hasattr(tensor, "detach") else torch.as_tensor(tensor).float()
    if value.ndim == 4 and int(value.shape[0]) == 1:
        value = value[0]
    elif value.ndim == 3 and int(value.shape[0]) == 1 and expected_seq_len is not None and int(value.shape[1]) == expected_seq_len:
        value = value[0]
    if value.ndim == 2:
        if expected_seq_len is not None and int(value.shape[1]) == expected_seq_len and int(value.shape[0]) != expected_seq_len:
            value = value.transpose(0, 1)
        return value.reshape(int(value.shape[0]), -1)
    if value.ndim < 2:
        raise RuntimeError(f"Unsupported KV tensor rank: {value.ndim}")
    seq_dim = infer_sequence_dim(value.shape, expected_seq_len)
    value = value.movedim(seq_dim, 0).contiguous()
    return value.reshape(int(value.shape[0]), -1)


def infer_sequence_dim(shape: Sequence[int], expected_seq_len: Optional[int]) -> int:
    dims = [int(dim) for dim in shape]
    if expected_seq_len is not None:
        matches = [index for index, dim in enumerate(dims) if dim == int(expected_seq_len)]
        if matches:
            if len(dims) == 3 and matches == [0] and dims[1] != int(expected_seq_len):
                return 0
            return matches[-1] if len(dims) == 4 else matches[0]
    if len(dims) == 3:
        return 1
    if len(dims) == 4:
        return 2
    return max(range(len(dims) - 1), key=lambda index: dims[index])


def comparable_scope_positions(prev: Any, cur: Any, kv_scope: str) -> List[int]:
    prev_positions = positions_for_scope(prev, kv_scope)
    cur_positions = positions_for_scope(cur, kv_scope)
    if prev_positions != cur_positions:
        raise RuntimeError(
            f"{kv_scope} positions differ between pair images: prev={len(prev_positions)} cur={len(cur_positions)}"
        )
    return list(cur_positions)


def positions_for_scope(extraction: Any, kv_scope: str) -> List[int]:
    if kv_scope == "visual_tokens":
        return list(extraction.visual_token_positions)
    if kv_scope == "text_tokens":
        return list(extraction.text_token_positions)
    if kv_scope == "full_prefix":
        return list(extraction.full_prefix_positions)
    raise ValueError(f"Unsupported kv_scope: {kv_scope}")


def threshold_row(
    pair: Any,
    args: argparse.Namespace,
    cur: Any,
    layer_id: int,
    kv_kind: str,
    kv_scope: str,
    token_count: int,
    pixel_changed_ratio: float,
    threshold: float,
    kv_changed_ratio: float,
    alignment: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "pair_id": pair.pair_id,
        "kv_boundary": cur.kv_boundary,
        "kv_scope": kv_scope,
        "layer_id": layer_id,
        "kv_kind": kv_kind,
        "kv_token_count": token_count,
        "visual_token_grid": list(alignment["visual_token_grid"]) if alignment.get("visual_token_grid") else None,
        "processor_merge_size": alignment.get("processor_merge_size"),
        "alignment_method": alignment.get("alignment_method") if kv_scope == "visual_tokens" else "not_spatial_scope",
        "feature_boundary": "prefill_after_llm_self_attention",
        "visual_position_source": cur.visual_position_source,
        "pixel_changed_ratio": pixel_changed_ratio,
        "kv_changed_ratio": kv_changed_ratio,
        "threshold": threshold,
        "kv_metric": args.kv_metric,
        "input_token_count": cur.input_tokens,
        "visual_token_position_count": len(cur.visual_token_positions),
        "text_token_position_count": len(cur.text_token_positions),
        "page_similarity": pair.similarity.tile_unchanged_ratio,
        "episode_id": metadata_text(pair.cur.sample, EPISODE_KEYS),
        "step_id": metadata_text(pair.cur.sample, STEP_KEYS),
        "app": metadata_text(pair.cur.sample, APP_NAME_KEYS),
        "task": metadata_text(pair.cur.sample, TASK_KEYS),
    }


def base_kv_row(
    pair: Any,
    layer_id: Optional[int],
    kv_kind: str,
    kv_scope: str,
    pixel_changed_ratio: float,
) -> Dict[str, Any]:
    return {
        "pair_id": pair.pair_id,
        "kv_boundary": KV_BOUNDARY,
        "kv_scope": kv_scope,
        "layer_id": layer_id,
        "kv_kind": kv_kind,
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
    }


def region_kv_stats(metric_values: Dict[str, List[float]], changed_mask: Sequence[bool]) -> Dict[str, Any]:
    result = {}
    for metric, values in metric_values.items():
        changed = [value for value, is_changed in zip(values, changed_mask) if is_changed]
        unchanged = [value for value, is_changed in zip(values, changed_mask) if not is_changed]
        result[metric] = {
            "changed_visual_region_tokens": summarize_values(changed),
            "unchanged_visual_region_tokens": summarize_values(unchanged),
        }
    return result


def aggregate_results(
    pair_rows: Sequence[Dict[str, Any]],
    threshold_rows: Sequence[Dict[str, Any]],
    distance_rows: Sequence[Dict[str, Any]],
    thresholds: Sequence[float],
    kv_metric: str,
) -> Dict[str, Any]:
    valid_rows = [row for row in pair_rows if not row.get("error")]
    valid_thresholds = [row for row in threshold_rows if row.get("kv_changed_ratio") is not None]
    return {
        "pair_scope_layer_kind_rows": len(pair_rows),
        "valid_pair_scope_layer_kind_rows": len(valid_rows),
        "error_count": len(pair_rows) - len(valid_rows),
        "r_pixel_distribution": summarize_values([row.get("pixel_changed_ratio") for row in valid_rows]),
        "r_kv_by_threshold": {
            str(tau): summarize_values(
                [row.get("kv_changed_ratio") for row in valid_thresholds if float(row.get("threshold")) == float(tau)]
            )
            for tau in thresholds
        },
        "r_pixel_r_kv_correlation_by_threshold": {
            str(tau): pearson(
                [row["pixel_changed_ratio"] for row in valid_thresholds if float(row.get("threshold")) == float(tau)],
                [row["kv_changed_ratio"] for row in valid_thresholds if float(row.get("threshold")) == float(tau)],
            )
            for tau in thresholds
        },
        "by_layer": grouped_by(valid_thresholds, "layer_id"),
        "by_kv_kind": grouped_by(valid_thresholds, "kv_kind"),
        "by_kv_scope": grouped_by(valid_thresholds, "kv_scope"),
        "grouped_stats": grouped_stats(valid_thresholds),
        "distance_to_changed_region": summarize_distance_rows(distance_rows, kv_metric),
        "decision_signals": build_decision_signals(valid_rows, valid_thresholds, kv_metric),
    }


def grouped_stats(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for key in ("app", "task", "episode_id"):
        output.extend(grouped_by(rows, key))
    return output


def grouped_by(rows: Sequence[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (
                str(row.get(key) or "unknown"),
                str(row.get("kv_scope") or "unknown"),
                str(row.get("kv_kind") or "unknown"),
                str(row.get("layer_id") or "unknown"),
                str(row.get("threshold") or "unknown"),
            ),
            [],
        ).append(row)
    return [
        summary_row(key, value, selected, kv_scope=kv_scope, kv_kind=kv_kind, layer_id=layer_id, threshold=threshold)
        for (value, kv_scope, kv_kind, layer_id, threshold), selected in sorted(groups.items())
    ]


def summary_row(
    dimension: str,
    value: str,
    rows: Sequence[Dict[str, Any]],
    kv_scope: Optional[str] = None,
    kv_kind: Optional[str] = None,
    layer_id: Optional[str] = None,
    threshold: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "dimension": dimension,
        "value": value,
        "kv_scope": kv_scope,
        "kv_kind": kv_kind,
        "layer_id": layer_id,
        "threshold": threshold,
        "kv_metric": rows[0].get("kv_metric") if rows else None,
        "row_count": len(rows),
        "pixel_changed_ratio": summarize_values([row.get("pixel_changed_ratio") for row in rows]),
        "kv_changed_ratio": summarize_values([row.get("kv_changed_ratio") for row in rows]),
        "correlation": pearson(
            [row["pixel_changed_ratio"] for row in rows if row.get("pixel_changed_ratio") is not None and row.get("kv_changed_ratio") is not None],
            [row["kv_changed_ratio"] for row in rows if row.get("pixel_changed_ratio") is not None and row.get("kv_changed_ratio") is not None],
        ),
    }


def summarize_distance_rows(rows: Sequence[Dict[str, Any]], kv_metric: str) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, Any], List[float]] = {}
    for row in rows:
        stats = row.get(kv_metric) or {}
        if stats.get("mean") is not None:
            grouped.setdefault((str(row.get("kv_kind")), str(row.get("kv_scope")), row.get("token_manhattan_distance")), []).append(float(stats["mean"]))
    return [
        {
            "kv_kind": kv_kind,
            "kv_scope": kv_scope,
            "token_manhattan_distance": distance,
            "pair_distance_bin_count": len(values),
            "kv_distance": summarize_values(values),
        }
        for (kv_kind, kv_scope, distance), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], -1 if item[0][2] is None else int(item[0][2])))
    ]


def build_decision_signals(
    pair_rows: Sequence[Dict[str, Any]],
    threshold_rows: Sequence[Dict[str, Any]],
    kv_metric: str,
) -> Dict[str, Any]:
    del kv_metric
    inflation = [
        row["kv_changed_ratio"] / max(row["pixel_changed_ratio"], 1e-8)
        for row in pair_rows
        if row.get("kv_changed_ratio") is not None and row.get("pixel_changed_ratio") is not None
    ]
    visual_rows = [row for row in pair_rows if row.get("kv_scope") == "visual_tokens"]
    changed_means = []
    unchanged_means = []
    for row in visual_rows:
        stats = ((row.get("changed_unchanged_kv_distance_stats") or {}).get(row.get("kv_metric")) or {})
        changed = (stats.get("changed_visual_region_tokens") or {}).get("mean")
        unchanged = (stats.get("unchanged_visual_region_tokens") or {}).get("mean")
        if changed is not None:
            changed_means.append(float(changed))
        if unchanged is not None:
            unchanged_means.append(float(unchanged))
    return {
        "r_kv_over_r_pixel": summarize_values(inflation),
        "changed_visual_region_kv_distance_mean": summarize_values(changed_means),
        "unchanged_visual_region_kv_distance_mean": summarize_values(unchanged_means),
        "threshold_row_count": len(threshold_rows),
        "interpretation_hint": "If R_kv approaches 1 in deeper layers or unchanged-region distances approach changed-region distances, patch/local KV reuse is not supported.",
    }


def flatten_pair_rows(rows: Sequence[Dict[str, Any]], kv_metric: str) -> List[Dict[str, Any]]:
    flat = []
    for row in rows:
        stats = ((row.get("changed_unchanged_kv_distance_stats") or {}).get(kv_metric) or {})
        changed = stats.get("changed_visual_region_tokens") or {}
        unchanged = stats.get("unchanged_visual_region_tokens") or {}
        flat.append(
            {
                "pair_id": row.get("pair_id"),
                "kv_boundary": row.get("kv_boundary"),
                "kv_scope": row.get("kv_scope"),
                "layer_id": row.get("layer_id"),
                "kv_kind": row.get("kv_kind"),
                "episode_id": row.get("episode_id"),
                "step_id": row.get("step_id"),
                "app": row.get("app"),
                "hit_type": row.get("hit_type"),
                "pixel_changed_ratio": row.get("pixel_changed_ratio"),
                "kv_metric": row.get("kv_metric"),
                "threshold": row.get("threshold"),
                "kv_changed_ratio": row.get("kv_changed_ratio"),
                "kv_token_count": row.get("kv_token_count"),
                "visual_token_grid": row.get("visual_token_grid"),
                "processor_merge_size": row.get("processor_merge_size"),
                "alignment_method": row.get("alignment_method"),
                "visual_position_source": row.get("visual_position_source"),
                "input_token_count": row.get("input_token_count"),
                "visual_token_position_count": row.get("visual_token_position_count"),
                "text_token_position_count": row.get("text_token_position_count"),
                "page_tile_unchanged_ratio": (row.get("page_similarity") or {}).get("tile_unchanged_ratio"),
                "changed_region_kv_mean": changed.get("mean"),
                "unchanged_region_kv_mean": unchanged.get("mean"),
                "error": row.get("error"),
            }
        )
    return flat


def write_pair_visualizations(
    run_dir: Path,
    pair: Any,
    layer_id: int,
    kv_kind: str,
    values: Sequence[float],
    changed_token_mask: Sequence[bool],
    token_grid: Optional[Tuple[int, int]],
) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    pair_dir = run_dir / "visualizations"
    pair_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{pair.pair_id}_layer{layer_id}_{safe_name(kv_kind)}"
    pixel_path = pair_dir / f"{suffix}_pixel_diff_heatmap.png"
    kv_path = pair_dir / f"{suffix}_kv_diff_heatmap.png"
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
        write_kv_diff_heatmap(values, changed_token_mask, token_grid, kv_path)
        paths["kv_diff_heatmap"] = str(kv_path)
    return paths


def write_kv_diff_heatmap(
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
    threshold_rows: Sequence[Dict[str, Any]],
    distance_rows: Sequence[Dict[str, Any]],
    kv_metric: str,
) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    written: List[str] = []
    valid = [row for row in threshold_rows if row.get("kv_changed_ratio") is not None]
    if valid:
        path = run_dir / "r_pixel_vs_r_kv.png"
        plt.figure(figsize=(6, 5))
        plt.scatter([row["pixel_changed_ratio"] for row in valid], [row["kv_changed_ratio"] for row in valid], alpha=0.5)
        plt.xlabel("R_pixel")
        plt.ylabel("R_kv")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        written.append(str(path))
    distance_means: Dict[Any, List[float]] = {}
    for row in distance_rows:
        stats = row.get(kv_metric) or {}
        if row.get("token_manhattan_distance") is not None and stats.get("mean") is not None:
            distance_means.setdefault(row["token_manhattan_distance"], []).append(float(stats["mean"]))
    if distance_means:
        path = run_dir / "kv_distance_by_changed_region_distance.png"
        xs = sorted(distance_means)
        ys = [statistics.fmean(distance_means[x]) for x in xs]
        plt.figure(figsize=(7, 4))
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Visual token Manhattan distance to changed region")
        plt.ylabel(kv_metric)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        written.append(str(path))
    return written


def tensor_to_float_list(tensor: Any) -> List[float]:
    return [float(value) for value in tensor.reshape(-1).tolist()]


def resolve_run_dir(output_dir: Path, run_name: Optional[str]) -> Path:
    if run_name:
        return output_dir / run_name
    if output_dir.name != "kv_locality_analysis":
        return output_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"kv_locality_{timestamp}"


def required_output_fields() -> Tuple[str, ...]:
    return (
        "kv_boundary",
        "kv_scope",
        "layer_id",
        "kv_kind",
        "kv_token_count",
        "visual_token_grid",
        "processor_merge_size",
        "alignment_method",
        "feature_boundary",
        "visual_position_source",
        "pixel_changed_ratio",
        "kv_changed_ratio",
        "threshold",
        "distance_stats",
        "changed_unchanged_kv_distance_stats",
        "distance_profile",
        "input_token_count",
        "visual_token_position_count",
        "text_token_position_count",
        "error",
    )


if __name__ == "__main__":
    raise SystemExit(main())
