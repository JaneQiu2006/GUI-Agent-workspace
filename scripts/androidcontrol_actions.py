from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, Optional, Tuple


POINT_RE = re.compile(r"<point>\[\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]\]</point>")
ACTION_RE = re.compile(r'"action(?:_type)?"\s*:\s*"([^"]+)"', re.IGNORECASE)
UNLABELED_Y_RE = re.compile(
    r'"(?:x|start_x)"\s*:\s*(-?[0-9]+(?:\.[0-9]+)?)\s*,\s*(-?[0-9]+(?:\.[0-9]+)?)\s*(?:[,}])'
)


def gt_action_to_command(action: Dict[str, Any], width: int, height: int) -> str:
    kind = str(action.get("action_type") or action.get("action") or "").lower()
    if kind == "click":
        return point_command("CLICK", action.get("x"), action.get("y"), width, height)
    if kind == "long_press":
        return point_command("LONG_PRESS", action.get("x"), action.get("y"), width, height)
    if kind == "scroll":
        direction = str(action.get("direction") or "").upper()
        return f"SCROLL[{direction}]" if direction else "SCROLL"
    if kind == "input_text":
        return f"TYPE [{action.get('text', '')}]"
    if kind == "navigate_back":
        return "PRESS_BACK"
    if kind == "navigate_home":
        return "PRESS_HOME"
    if kind == "wait":
        return "WAIT"
    if kind == "open_app":
        return f"OPEN_APP [{action.get('app_name', '')}]"
    return f"UNKNOWN [{json.dumps(action, ensure_ascii=False)}]"


def point_command(prefix: str, x_value: Any, y_value: Any, width: int, height: int) -> str:
    x = normalize_coord(x_value, width)
    y = normalize_coord(y_value, height)
    return f"{prefix} <point>[[{x},{y}]]</point>"


def normalize_coord(value: Any, dimension: int) -> int:
    number = float(value)
    if 0 <= number <= 1:
        normalized = number * 1000
    elif dimension > 0:
        normalized = number / dimension * 1000
    else:
        normalized = number
    return int(round(max(0, min(1000, normalized))))


def canonicalize_action(raw_response: str) -> str:
    text = (raw_response or "").strip()
    if not text:
        return "UNKNOWN []"
    legacy = parse_legacy_command(text)
    if legacy:
        return legacy

    decoder = json.JSONDecoder()
    parsed: Optional[Dict[str, Any]] = None
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            parsed = value
            break
    if parsed is None:
        repaired = parse_malformed_action_json(text)
        return repaired if repaired else text
    command = dict_action_to_command(parsed)
    if action_type(command) != "UNKNOWN":
        return command
    repaired = repair_malformed_dict_action(parsed)
    return repaired if repaired else command


def dict_action_to_command(action: Dict[str, Any]) -> str:
    kind = str(action.get("action") or action.get("action_type") or "").lower()
    if kind in {"tap", "click"}:
        x, y = point_fields(action)
        if x is not None and y is not None:
            return normalized_point_command("CLICK", x, y)
    if kind in {"long_press", "longpress"}:
        x, y = point_fields(action)
        if x is not None and y is not None:
            return normalized_point_command("LONG_PRESS", x, y)
    if kind in {"swipe", "scroll"}:
        direction = str(action.get("direction") or "").upper()
        if direction:
            return f"SCROLL[{direction}]"
        direction = infer_scroll_direction(action)
        return f"SCROLL[{direction}]" if direction else "SCROLL"
    if kind in {"type", "input_text"}:
        return f"TYPE [{action.get('text', '')}]"
    if kind in {"back", "navigate_back"}:
        return "PRESS_BACK"
    if kind in {"home", "navigate_home"}:
        return "PRESS_HOME"
    if kind == "wait":
        return "WAIT"
    if kind in {"open_app", "open", "openapp", "launch", "launch_app"}:
        app_name = action.get("app_name") or action.get("app") or action.get("text") or action.get("name") or ""
        return f"OPEN_APP [{app_name}]"
    if kind == "complete":
        return "COMPLETE"
    if kind == "impossible":
        return "IMPOSSIBLE"
    return f"UNKNOWN [{json.dumps(action, ensure_ascii=False)}]"


def point_fields(action: Dict[str, Any]) -> Tuple[Any, Any]:
    x = action.get("x")
    y = action.get("y")
    point = action.get("point") or action.get("coordinate") or action.get("position")
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        x, y = point[:2]
    if x is not None and y is None:
        y = single_extra_numeric_value(action, skip_keys={"x"})
    return x, y


def single_extra_numeric_value(action: Dict[str, Any], skip_keys: set[str]) -> Optional[Any]:
    values = []
    for key, value in action.items():
        if key in skip_keys or key in {"action", "action_type", "text", "app", "app_name", "name"}:
            continue
        if is_number(key) and is_number(value):
            values.append(value)
    return values[0] if len(values) == 1 else None


def repair_malformed_dict_action(action: Dict[str, Any]) -> Optional[str]:
    kind = str(action.get("action") or action.get("action_type") or "").lower()
    if kind in {"tap", "click"}:
        x, y = point_fields(action)
        if x is not None and y is not None:
            return normalized_point_command("CLICK", x, y)
    if kind in {"long_press", "longpress"}:
        x, y = point_fields(action)
        if x is not None and y is not None:
            return normalized_point_command("LONG_PRESS", x, y)
    return None


def parse_malformed_action_json(text: str) -> Optional[str]:
    action_match = ACTION_RE.search(text)
    if not action_match:
        return None
    kind = action_match.group(1).lower()
    if kind in {"tap", "click", "long_press", "longpress"}:
        point_match = UNLABELED_Y_RE.search(text)
        if point_match:
            prefix = "LONG_PRESS" if kind in {"long_press", "longpress"} else "CLICK"
            return normalized_point_command(prefix, point_match.group(1), point_match.group(2))
    return None


def normalized_point_command(prefix: str, x_value: Any, y_value: Any) -> str:
    x = int(round(float(x_value)))
    y = int(round(float(y_value)))
    return f"{prefix} <point>[[{x},{y}]]</point>"


def is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def parse_legacy_command(text: str) -> Optional[str]:
    upper = text.upper()
    if upper.startswith("CLICK"):
        point = parse_point(text)
        if point:
            return f"CLICK <point>[[{point[0]},{point[1]}]]</point>"
    if upper.startswith("LONG_PRESS"):
        point = parse_point(text)
        if point:
            return f"LONG_PRESS <point>[[{point[0]},{point[1]}]]</point>"
    if upper.startswith("SCROLL"):
        match = re.search(r"SCROLL\s*\[?\s*(UP|DOWN|LEFT|RIGHT)\s*\]?", upper)
        return f"SCROLL[{match.group(1)}]" if match else "SCROLL"
    if upper.startswith("TYPE"):
        return text
    if upper.startswith("PRESS_BACK"):
        return "PRESS_BACK"
    if upper.startswith("PRESS_HOME"):
        return "PRESS_HOME"
    if upper.startswith("WAIT"):
        return "WAIT"
    if upper.startswith("OPEN_APP"):
        return text
    return None


def infer_scroll_direction(action: Dict[str, Any]) -> Optional[str]:
    start = action.get("start") or action.get("from") or action.get("start_point")
    end = action.get("end") or action.get("to") or action.get("end_point")
    x1 = action.get("x1", action.get("start_x", start[0] if isinstance(start, list) and len(start) >= 2 else None))
    y1 = action.get("y1", action.get("start_y", start[1] if isinstance(start, list) and len(start) >= 2 else None))
    x2 = action.get("x2", action.get("end_x", end[0] if isinstance(end, list) and len(end) >= 2 else None))
    y2 = action.get("y2", action.get("end_y", end[1] if isinstance(end, list) and len(end) >= 2 else None))
    if None in (x1, y1, x2, y2):
        return None
    dx = float(x2) - float(x1)
    dy = float(y2) - float(y1)
    if abs(dx) > abs(dy):
        finger_direction = "RIGHT" if dx > 0 else "LEFT"
        return finger_to_content_scroll_direction(finger_direction)
    if abs(dy) > 0:
        finger_direction = "DOWN" if dy > 0 else "UP"
        return finger_to_content_scroll_direction(finger_direction)
    return None


def finger_to_content_scroll_direction(direction: str) -> str:
    opposites = {
        "UP": "DOWN",
        "DOWN": "UP",
        "LEFT": "RIGHT",
        "RIGHT": "LEFT",
    }
    return opposites[direction]


def action_type(command: str) -> str:
    text = (command or "").strip().upper()
    if text.startswith("CLICK"):
        return "CLICK"
    if text.startswith("LONG_PRESS"):
        return "LONG_PRESS"
    if text.startswith("SCROLL"):
        return "SCROLL"
    if text.startswith("TYPE"):
        return "TYPE"
    if text.startswith("PRESS_BACK"):
        return "PRESS_BACK"
    if text.startswith("PRESS_HOME"):
        return "PRESS_HOME"
    if text.startswith("WAIT"):
        return "WAIT"
    if text.startswith("OPEN_APP"):
        return "OPEN_APP"
    if text.startswith("COMPLETE"):
        return "COMPLETE"
    if text.startswith("IMPOSSIBLE"):
        return "IMPOSSIBLE"
    return "UNKNOWN"


def parse_point(command: str) -> Optional[Tuple[int, int]]:
    match = POINT_RE.search(command or "")
    if not match:
        return None
    return int(round(float(match.group(1)))), int(round(float(match.group(2))))


def actions_match(prediction: str, target: str, point_tolerance: float = 100.0) -> bool:
    pred_type = action_type(prediction)
    target_type = action_type(target)
    if pred_type != target_type:
        return False
    if target_type in {"CLICK", "LONG_PRESS"}:
        pred_point = parse_point(prediction)
        target_point = parse_point(target)
        if not pred_point or not target_point:
            return False
        return math.dist(pred_point, target_point) <= point_tolerance
    return normalize_text(prediction) == normalize_text(target)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", (value or "").strip()).upper()
