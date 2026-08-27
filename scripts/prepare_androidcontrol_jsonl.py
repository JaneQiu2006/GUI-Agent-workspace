from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from test_gui_benchmark import first_value, iter_json_records


IMAGE_KEYS = ("screenshot", "image_path", "image", "path", "img")
GOAL_KEYS = ("goal", "task", "high_level_instruction")
STEP_KEYS = ("instruction", "step_instruction", "low_level_instruction", "query")
ID_KEYS = ("sample_id", "id", "uid")


def load_androidcontrol_records(path: Path) -> List[Dict[str, Any]]:
    records = iter_json_records(path)
    flattened: List[Dict[str, Any]] = []
    for item in records:
        if isinstance(item.get("messages"), list):
            flattened.append(from_llamafactory_record(item))
        else:
            flattened.append(item)
    return flattened


def from_llamafactory_record(item: Dict[str, Any]) -> Dict[str, Any]:
    user_text = ""
    assistant_text = ""
    for message in item.get("messages") or []:
        role = message.get("role")
        content = str(message.get("content") or "")
        if role == "user":
            user_text = content
        elif role == "assistant":
            assistant_text = content

    image = None
    images = item.get("images")
    if isinstance(images, list) and images:
        image = images[0]

    instruction = re.sub(r"^<image>\s*", "", user_text).strip()
    parsed_action = None
    if assistant_text:
        try:
            parsed_action = json.loads(assistant_text)
        except json.JSONDecodeError:
            parsed_action = {"raw": assistant_text}
    return {
        "instruction": instruction,
        "image_path": image,
        "action": parsed_action,
        "source_format": "llamafactory_messages",
    }


def make_sample_id(item: Dict[str, Any], index: int) -> Any:
    sample_id = first_value(item, ID_KEYS)
    if sample_id is not None:
        return sample_id
    episode = item.get("episode_id")
    step = item.get("step_idx", item.get("step_id"))
    if episode is not None and step is not None:
        return f"episode_{episode}_step_{step}"
    return index


def combine_instruction(goal: Optional[Any], step_instruction: Optional[Any]) -> str:
    goal_text = str(goal or "").strip()
    step_text = str(step_instruction or "").strip()
    if goal_text and step_text and goal_text != step_text:
        return f"目标任务：{goal_text}\n当前步骤：{step_text}"
    return step_text or goal_text


def image_path_for_record(item: Dict[str, Any]) -> Optional[str]:
    image = first_value(item, IMAGE_KEYS)
    if image is None:
        return None
    return str(image)


def normalize_action(
    action: Any,
    width: Optional[Any] = None,
    height: Optional[Any] = None,
    coordinate_mode: str = "pixel",
) -> Optional[Dict[str, Any]]:
    if not isinstance(action, dict):
        return action
    raw = dict(action)
    kind = str(raw.get("action") or raw.get("action_type") or raw.get("action_name") or "").lower()
    if kind in {"click", "tap"}:
        point = raw.get("point") or raw.get("start_point")
        x = raw.get("x")
        y = raw.get("y")
        if point is not None and isinstance(point, (list, tuple)) and len(point) >= 2:
            x, y = point[:2]
        if x is not None and y is not None:
            return {"action": "tap", "x": coord(x, width, coordinate_mode), "y": coord(y, height, coordinate_mode), "raw": raw}
    if kind in {"long_press", "longpress"}:
        point = raw.get("point") or raw.get("start_point")
        x = raw.get("x")
        y = raw.get("y")
        if point is not None and isinstance(point, (list, tuple)) and len(point) >= 2:
            x, y = point[:2]
        if x is not None and y is not None:
            return {"action": "long_press", "x": coord(x, width, coordinate_mode), "y": coord(y, height, coordinate_mode), "raw": raw}
    if kind in {"input_text", "type", "write"}:
        return {"action": "type", "text": str(raw.get("text") or raw.get("keys") or raw.get("value") or ""), "raw": raw}
    if kind in {"navigate_back", "back"}:
        return {"action": "back", "raw": raw}
    if kind in {"navigate_home", "home"}:
        return {"action": "home", "raw": raw}
    if kind == "wait":
        return {"action": "wait", "seconds": raw.get("seconds", 2), "raw": raw}
    if kind in {"terminate", "complete"}:
        return {"action": "complete", "raw": raw}
    if kind in {"scroll", "swipe"}:
        swipe = normalize_swipe(raw, width, height, coordinate_mode)
        if swipe:
            swipe["raw"] = raw
            return swipe
    if kind == "open_app":
        return {"action": "open_app", "app_name": raw.get("app_name") or raw.get("app"), "raw": raw}
    return {"action": kind or "unknown", "raw": raw}


def normalize_swipe(
    action: Dict[str, Any],
    width: Optional[Any],
    height: Optional[Any],
    coordinate_mode: str,
) -> Optional[Dict[str, Any]]:
    start = action.get("start") or action.get("from") or action.get("start_point")
    end = action.get("end") or action.get("to") or action.get("end_point")
    if isinstance(start, (list, tuple)) and isinstance(end, (list, tuple)) and len(start) >= 2 and len(end) >= 2:
        return {
            "action": "swipe",
            "x1": coord(start[0], width, coordinate_mode),
            "y1": coord(start[1], height, coordinate_mode),
            "x2": coord(end[0], width, coordinate_mode),
            "y2": coord(end[1], height, coordinate_mode),
            "duration_ms": int(action.get("duration_ms") or action.get("duration") or 500),
        }
    direction = str(action.get("direction") or "").lower()
    defaults = {
        "up": (500, 700, 500, 300),
        "down": (500, 300, 500, 700),
        "left": (700, 500, 300, 500),
        "right": (300, 500, 700, 500),
    }
    if direction in defaults:
        x1, y1, x2, y2 = defaults[direction]
        return {"action": "swipe", "x1": x1, "y1": y1, "x2": x2, "y2": y2, "duration_ms": 500}
    return None


def coord(value: Any, dimension: Optional[Any], coordinate_mode: str) -> int:
    number = float(value)
    if 0 <= number <= 1:
        return int(round(number * 1000))
    if coordinate_mode == "pixel" and dimension:
        dim = float(dimension)
        if dim > 0:
            return int(round(number / dim * 1000))
    return int(round(number))


def normalize_records(records: Iterable[Dict[str, Any]], coordinate_mode: str) -> Iterable[Dict[str, Any]]:
    for index, item in enumerate(records):
        goal = first_value(item, GOAL_KEYS)
        step = first_value(item, STEP_KEYS)
        image = image_path_for_record(item)
        instruction = combine_instruction(goal, step)
        if not image or not instruction:
            continue
        width = item.get("screenshot_width") or item.get("image_width")
        height = item.get("screenshot_height") or item.get("image_height")
        yield {
            "sample_id": make_sample_id(item, index),
            "goal": goal,
            "step_instruction": step,
            "instruction": instruction,
            "image_path": image,
            "history": item.get("history") or item.get("history_list") or item.get("previous_actions") or [],
            "expected_action": normalize_action(item.get("action"), width, height, coordinate_mode),
            "source_dataset": "androidcontrol",
            "source_index": index,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare AndroidControl JSON/JSONL for test_gui_benchmark.py")
    parser.add_argument("--input", type=Path, required=True, help="AndroidControl JSON/JSONL annotation file")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL consumed by test_gui_benchmark.py")
    parser.add_argument("--limit", type=int, help="Write at most N samples")
    parser.add_argument(
        "--coordinate_mode",
        choices=("pixel", "normalized_1000"),
        default="pixel",
        help="How AndroidControl x/y fields are interpreted before converting to 0-1000",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    records = list(normalize_records(load_androidcontrol_records(args.input), args.coordinate_mode))
    if args.limit is not None:
        records = records[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"ANDROIDCONTROL_PREPARED {len(records)} samples -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
