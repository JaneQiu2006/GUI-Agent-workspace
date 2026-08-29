from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_FRAMEWORK = REPO_ROOT / "test_framework"
if str(TEST_FRAMEWORK) not in sys.path:
    sys.path.insert(0, str(TEST_FRAMEWORK))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from androidcontrol_actions import action_type, actions_match, canonicalize_action
from hf_gui_baseline import DEFAULT_MODEL_PATH, infer_one, load_model_and_processor


TRANSITION_OR_NOOP_TYPES = {"OPEN_APP", "WAIT"}
GUI_ONLY_EXCLUDED_TYPES = TRANSITION_OR_NOOP_TYPES


def load_samples(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        records = data.get("samples", [])
    elif isinstance(data, list):
        records = data
    else:
        records = []
    return [record for record in records if isinstance(record, dict)]


def peak_gpu_memory() -> Dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {}
    if not torch.cuda.is_available():
        return {}
    per_device = {}
    warnings = []
    for index in range(torch.cuda.device_count()):
        try:
            per_device[str(index)] = int(torch.cuda.max_memory_allocated(index))
        except RuntimeError as exc:
            warnings.append(f"cuda:{index}: {exc}")
    metrics = {
        "peak_gpu_memory_bytes": per_device,
        "peak_gpu_memory_gb": {
            key: round(value / 1024**3, 4) for key, value in per_device.items()
        },
    }
    if warnings:
        metrics["gpu_memory_warnings"] = warnings
    return metrics


def reset_peak_gpu_memory() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            try:
                torch.cuda.reset_peak_memory_stats(index)
            except RuntimeError as exc:
                print(
                    f"GPU_MEMORY_WARNING reset_peak_memory_stats(cuda:{index}) failed: {exc}",
                    file=sys.stderr,
                    flush=True,
                )


def summarize_metric_view(
    details: List[Dict[str, Any]],
    include_types: Optional[Set[str]] = None,
    exclude_types: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    selected = []
    for item in details:
        gt_type = str(item.get("gt_type") or "")
        if include_types is not None and gt_type not in include_types:
            continue
        if exclude_types is not None and gt_type in exclude_types:
            continue
        selected.append(item)

    count = len(selected)
    type_hits = sum(1 for item in selected if item.get("type_success"))
    step_hits = sum(1 for item in selected if item.get("step_success"))
    latencies = [float(item["latency_seconds"]) for item in selected if item.get("latency_seconds") is not None]
    output_tokens = [float(item["output_tokens"]) for item in selected if item.get("output_tokens") is not None]
    input_tokens = [float(item["input_tokens"]) for item in selected if item.get("input_tokens") is not None]
    episode_success: Dict[str, List[bool]] = defaultdict(list)
    for item in selected:
        episode_key = str(item.get("episode_id", item.get("step_id", "")))
        episode_success[episode_key].append(bool(item.get("step_success")))

    successful_trajectories = sum(1 for values in episode_success.values() if values and all(values))
    return {
        "num_steps": count,
        "num_trajectories": len(episode_success),
        "type_accuracy": type_hits / count if count else 0.0,
        "step_success_rate": step_hits / count if count else 0.0,
        "trajectory_success_rate": successful_trajectories / len(episode_success) if episode_success else 0.0,
        "avg_latency_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
        "avg_input_tokens": sum(input_tokens) / len(input_tokens) if input_tokens else 0.0,
        "avg_output_tokens": sum(output_tokens) / len(output_tokens) if output_tokens else 0.0,
    }


def metric_group_for_type(gt_type: str) -> str:
    if gt_type == "OPEN_APP":
        return "open_app"
    if gt_type == "WAIT":
        return "wait"
    return "gui_only"


def metric_view_policy() -> Dict[str, str]:
    return {
        "strict": "All AndroidControl steps; kept for compatibility with earlier results.",
        "gui_only": "Ordinary GUI manipulation steps, excluding OPEN_APP and WAIT.",
        "transition_or_noop": "OPEN_APP and WAIT steps reported separately because static single-frame matching is noisy or environment-level.",
        "open_app": "Environment-level app launch requests only.",
        "wait": "Static wait/no-op requests only.",
        "by_gt_type": "Per ground-truth action type breakdown.",
    }


def build_metric_views(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    gt_types = sorted({str(item.get("gt_type") or "UNKNOWN") for item in details})
    return {
        "strict": summarize_metric_view(details),
        "gui_only": summarize_metric_view(details, exclude_types=GUI_ONLY_EXCLUDED_TYPES),
        "transition_or_noop": summarize_metric_view(details, include_types=TRANSITION_OR_NOOP_TYPES),
        "open_app": summarize_metric_view(details, include_types={"OPEN_APP"}),
        "wait": summarize_metric_view(details, include_types={"WAIT"}),
        "by_gt_type": {
            gt_type: summarize_metric_view(details, include_types={gt_type})
            for gt_type in gt_types
        },
    }


def evaluate_records(
    records: List[Dict[str, Any]],
    model: Any,
    processor: Any,
    data_dir: Path,
    max_new_tokens: int,
    device: str,
    point_tolerance: float,
) -> Dict[str, Any]:
    details = []

    for index, sample in enumerate(records):
        image_path = Path(str(sample["image_path"]))
        if not image_path.is_absolute():
            image_path = data_dir / image_path
        gt_action = str(sample["action"])
        result = infer_one(
            model,
            processor,
            image_path,
            str(sample["task"]),
            max_new_tokens=max_new_tokens,
            device=device,
        )
        pred_action = canonicalize_action(result.raw_response)
        pred_type = action_type(pred_action)
        gt_type = action_type(gt_action)
        type_ok = pred_type == gt_type
        step_ok = actions_match(pred_action, gt_action, point_tolerance=point_tolerance)
        metric_group = metric_group_for_type(gt_type)
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
                "metric_group": metric_group,
                "strict_included": True,
                "gui_only_included": metric_group == "gui_only",
                "type_success": type_ok,
                "step_success": step_ok,
                "latency_seconds": result.latency_seconds,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            }
        )
        print(
            f"EVAL_STEP {index + 1}/{len(records)} episode={sample.get('episode_id')} "
            f"step={sample.get('step_id')} type_ok={type_ok} step_ok={step_ok}",
            flush=True,
        )

    views = build_metric_views(details)
    metrics = {
        **views["strict"],
        "primary_metric_view": "gui_only",
        "metric_view_policy": metric_view_policy(),
        "views": views,
        **peak_gpu_memory(),
    }
    return {"metrics": metrics, "details": details}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Qwen static GUI actions on AndroidControl mini")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test_json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, help="Defaults to the parent directory of --test_json")
    parser.add_argument("--limit", type=int)
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
    reset_peak_gpu_memory()
    model, processor = load_model_and_processor(
        args.model_path,
        device=args.device,
        device_map=args.device_map or None,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    result = evaluate_records(
        records=records,
        model=model,
        processor=processor,
        data_dir=data_dir,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        point_tolerance=args.point_tolerance,
    )
    result["config"] = {
        "model_path": args.model_path,
        "test_json": str(args.test_json),
        "data_dir": str(data_dir),
        "max_new_tokens": args.max_new_tokens,
        "point_tolerance": args.point_tolerance,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("EVAL_DONE " + json.dumps(result["metrics"], ensure_ascii=False), flush=True)
    print(f"EVAL_OUTPUT {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
