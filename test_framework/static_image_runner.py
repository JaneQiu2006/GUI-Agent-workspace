"""Run Qwen GUI-agent inference on static screenshots without ADB.

This is useful when the target device cannot be controlled through ADB, for
example iOS screenshots collected manually.  It keeps the same prompt and
action parser as benchmark_runner.py, but only performs one model call per
image/task pair.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import time
from typing import Any, Dict, List, Optional

from PIL import Image

from benchmark_runner import VllmClient, annotate_screenshot, parse_action, utc_now


TASK_KEYS = ("task", "instruction", "任务")
IMAGE_KEYS = ("image", "image_path", "screenshot", "截图", "图片")


def first_value(row: Dict[str, Any], keys: tuple[str, ...]) -> Optional[str]:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def safe_name(value: str, fallback: str) -> str:
    name = re.sub(r"[^0-9A-Za-z._\-\u4e00-\u9fff]+", "_", value).strip("_")
    return name[:80] or fallback


def load_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("tasks", [])
        records = []
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            task = first_value(item, TASK_KEYS)
            image = first_value(item, IMAGE_KEYS)
            if task and image:
                records.append({"task": task, "image": image, "source": item, "index": index})
        return records

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        records = []
        for index, row in enumerate(rows):
            task = first_value(row, TASK_KEYS)
            image = first_value(row, IMAGE_KEYS)
            if task and image:
                records.append({"task": task, "image": image, "source": row, "index": index})
        return records

    raise ValueError("批量任务文件只支持 .csv 或 .json")


def resolve_image_path(image: str, base_dir: Path) -> Path:
    path = Path(image)
    if not path.is_absolute():
        path = base_dir / path
    if not path.is_file():
        raise FileNotFoundError(f"静态图片不存在: {path}")
    return path


def infer_one(
    client: VllmClient,
    task: str,
    image_path: Path,
    output_dir: Path,
    history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied_image = output_dir / image_path.name
    if image_path.resolve() != copied_image.resolve():
        shutil.copy2(image_path, copied_image)

    started = time.perf_counter()
    response, timing, usage = client.infer(task, image_path, history)
    action = parse_action(response)
    with Image.open(image_path) as image:
        size = image.size

    annotated = output_dir / "annotated.png"
    annotate_screenshot(image_path, annotated, action, size)

    record = {
        "task": task,
        "image": str(image_path),
        "copied_image": str(copied_image),
        "annotated_image": str(annotated),
        "timestamp": utc_now(),
        "model_response": response,
        "action": action,
        **timing,
        "static_e2e_seconds": time.perf_counter() - started,
        "usage": usage,
    }
    (output_dir / "response.txt").write_text(response, encoding="utf-8")
    (output_dir / "action.json").write_text(
        json.dumps(action, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="静态截图 GUI Agent 推理")
    parser.add_argument("--image", type=Path, help="单张输入截图")
    parser.add_argument("--task", help="单张截图对应的任务描述")
    parser.add_argument("--tasks", type=Path, help="批量 CSV/JSON，需包含任务列和图片路径列")
    parser.add_argument("--output", type=Path, default=Path("outputs/static_image_eval"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8018/v1")
    parser.add_argument("--model", default="/data2/home/models/QWen3.8-27B")
    parser.add_argument("--history", default="[]", help="可选历史 action JSON 数组")
    parser.add_argument("--limit", type=int, help="批量模式最多运行多少条")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.tasks:
        records = load_records(args.tasks)
        base_dir = args.tasks.parent
    else:
        if not args.image or not args.task:
            raise SystemExit("单图模式需要同时提供 --image 和 --task，或使用 --tasks 批量文件")
        records = [{"task": args.task, "image": str(args.image), "source": {}, "index": 0}]
        base_dir = Path.cwd()

    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise SystemExit("没有可运行的静态图片任务")

    try:
        history = json.loads(args.history)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--history 不是有效 JSON: {exc}") from exc
    if not isinstance(history, list):
        raise SystemExit("--history 必须是 JSON 数组")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run_config.json").write_text(
        json.dumps(vars(args), default=str, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    client = VllmClient(args.base_url, args.model)
    results_path = args.output / "results.jsonl"
    summary_path = args.output / "summary.csv"

    with results_path.open("a", encoding="utf-8") as results_file, summary_path.open(
        "a", encoding="utf-8", newline=""
    ) as summary_file:
        writer = csv.writer(summary_file)
        if summary_path.stat().st_size == 0:
            writer.writerow([
                "index",
                "task",
                "image",
                "action",
                "ttft_seconds",
                "model_total_seconds",
                "request_e2e_seconds",
                "static_e2e_seconds",
                "status",
            ])

        for position, record in enumerate(records, 1):
            task = record["task"]
            image_path = resolve_image_path(record["image"], base_dir)
            case_dir = args.output / f"case_{record.get('index', position - 1):04d}_{safe_name(task, 'task')}"
            print(f"STATIC_TASK_START {position}/{len(records)} task={task}", flush=True)
            try:
                result = infer_one(client, task, image_path, case_dir, list(history))
                status = "ok"
            except Exception as exc:
                result = {
                    "task": task,
                    "image": str(image_path),
                    "timestamp": utc_now(),
                    "status": "error",
                    "error": str(exc),
                }
                status = "error"
            results_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            results_file.flush()
            action = result.get("action", {})
            writer.writerow([
                record.get("index", position - 1),
                task,
                str(image_path),
                action.get("action") if isinstance(action, dict) else "",
                f"{result.get('ttft_seconds', 0.0):.6f}",
                f"{result.get('model_total_seconds', 0.0):.6f}",
                f"{result.get('request_e2e_seconds', 0.0):.6f}",
                f"{result.get('static_e2e_seconds', 0.0):.6f}",
                status,
            ])
            summary_file.flush()
            print(f"STATIC_TASK_DONE status={status}", flush=True)

    print(f"STATIC_RUN_DONE output={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
