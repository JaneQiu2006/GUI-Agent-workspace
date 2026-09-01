from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
from typing import Any, Dict, Iterable, List, Optional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize AndroidWorld cache benchmark runs")
    parser.add_argument("--baseline", type=Path, help="Baseline output dir or summary.json")
    parser.add_argument("--cache", type=Path, help="Cache-enabled output dir or summary.json")
    parser.add_argument("--run", action="append", type=Path, default=[], help="Additional output dir or summary.json")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    runs = []
    if args.baseline:
        runs.append(("baseline", args.baseline))
    if args.cache:
        runs.append(("cache", args.cache))
    runs.extend((f"run_{index}", path) for index, path in enumerate(args.run))
    if not runs:
        raise SystemExit("Provide --baseline/--cache or at least one --run")

    summaries = {label: summarize_run_path(path) for label, path in runs}
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runs": summaries,
        "comparison": compare_runs(summaries.get("baseline"), summaries.get("cache")),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("ANDROIDWORLD_CACHE_SUMMARY_DONE " + json.dumps(output["comparison"], ensure_ascii=False), flush=True)
    print(f"ANDROIDWORLD_CACHE_SUMMARY_OUTPUT {args.output}", flush=True)
    return 0


def summarize_run_path(path: Path) -> Dict[str, Any]:
    summary_path = path / "summary.json" if path.is_dir() else path
    summary = read_json(summary_path)
    run_dir = summary_path.parent
    steps = read_jsonl(run_dir / "steps.jsonl")
    episodes = read_jsonl(run_dir / "episodes.jsonl")
    metrics = dict(summary.get("metrics") or compute_metrics(episodes, steps))
    return {
        "path": str(summary_path),
        "metrics": metrics,
        "by_task_template": summary.get("by_task_template") or group_metrics(episodes, steps, "task_template"),
        "by_app": summary.get("by_app") or group_metrics(episodes, steps, "app"),
        "cache_reuse_groups": summary.get("cache_reuse_groups") or cache_reuse_groups(steps),
    }


def compare_runs(baseline: Optional[Dict[str, Any]], cache: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not baseline or not cache:
        return {}
    b = baseline.get("metrics", {})
    c = cache.get("metrics", {})
    baseline_step = as_float(b.get("average_step_latency_seconds"))
    cache_step = as_float(c.get("average_step_latency_seconds"))
    baseline_episode = as_float(b.get("average_episode_latency_seconds"))
    cache_episode = as_float(c.get("average_episode_latency_seconds"))
    return {
        "baseline_vs_cache_speedup_step_latency": ratio(baseline_step, cache_step),
        "baseline_vs_cache_speedup_episode_latency": ratio(baseline_episode, cache_episode),
        "success_rate_delta": as_float(c.get("task_success_rate")) - as_float(b.get("task_success_rate")),
        "cache_hit_rate_delta": as_float(c.get("cache_hit_rate")) - as_float(b.get("cache_hit_rate")),
        "model_invocation_count_delta": int(c.get("model_invocation_count") or 0) - int(b.get("model_invocation_count") or 0),
        "cache_lookup_overhead_seconds": c.get("cache_lookup_overhead_seconds"),
    }


def compute_metrics(episodes: List[Dict[str, Any]], steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    step_latencies = [as_float(item.get("step_latency_seconds")) for item in steps if item.get("step_latency_seconds") is not None]
    episode_latencies = [as_float(item.get("episode_latency_seconds")) for item in episodes if item.get("episode_latency_seconds") is not None]
    cache_records = [item.get("cache") for item in steps if isinstance(item.get("cache"), dict)]
    lookup = [as_float(record.get("cache_lookup_seconds")) for record in cache_records]
    return {
        "num_episodes": len(episodes),
        "num_steps": len(steps),
        "task_success_rate": mean_bool(item.get("success") for item in episodes),
        "cache_hit_rate": mean_bool(item.get("cache_hit") for item in steps),
        "page_cache_hit_rate": mean_bool(record.get("page_cache_hit") for record in cache_records),
        "processor_cache_hit_rate": mean_bool(record.get("processor_cache_hit") for record in cache_records),
        "average_step_latency_seconds": statistics.fmean(step_latencies) if step_latencies else 0.0,
        "average_episode_latency_seconds": statistics.fmean(episode_latencies) if episode_latencies else 0.0,
        "model_invocation_count": sum(1 for item in steps if item.get("model_invoked")),
        "cache_lookup_overhead_seconds": statistics.fmean(lookup) if lookup else 0.0,
        "cache_lookup_total_seconds": sum(lookup),
    }


def group_metrics(episodes: List[Dict[str, Any]], steps: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    values = sorted({str(item.get(key, "unknown")) for item in episodes + steps})
    return {
        value: compute_metrics(
            [item for item in episodes if str(item.get(key, "unknown")) == value],
            [item for item in steps if str(item.get(key, "unknown")) == value],
        )
        for value in values
    }


def cache_reuse_groups(steps: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    groups = sorted({str(item.get("cache_reuse_group", "miss")) for item in steps} | {"overall"})
    data = {}
    for group in groups:
        selected = steps if group == "overall" else [item for item in steps if str(item.get("cache_reuse_group", "miss")) == group]
        data[group] = {
            "num_steps": len(selected),
            "hit_rate": mean_bool(item.get("cache_hit") for item in selected),
        }
    return data


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def ratio(numerator: float, denominator: float) -> Optional[float]:
    if denominator <= 0:
        return None
    return numerator / denominator


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def mean_bool(values: Iterable[Any]) -> float:
    items = [bool(value) for value in values]
    return sum(1 for value in items if value) / len(items) if items else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
