from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, Optional, Tuple


POINT_RE = re.compile(r"<point>\[\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]\]</point>")


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
        return text
    return dict_action_to_command(parsed)


def dict_action_to_command(action: Dict[str, Any]) -> str:
    kind = str(action.get("action") or action.get("action_type") or "").lower()
    if kind in {"tap", "click"}:
        x = action.get("x")
        y = action.get("y")
        point = action.get("point") or action.get("coordinate")
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            x, y = point[:2]
        if x is not None and y is not None:
            return f"CLICK <point>[[{int(round(float(x)))},{int(round(float(y)))}]]</point>"
    if kind in {"long_press", "longpress"}:
        x = action.get("x")
        y = action.get("y")
        point = action.get("point") or action.get("coordinate")
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            x, y = point[:2]
        if x is not None and y is not None:
            return f"LONG_PRESS <point>[[{int(round(float(x)))},{int(round(float(y)))}]]</point>"
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
    if kind == "open_app":
        return f"OPEN_APP [{action.get('app_name') or action.get('app') or ''}]"
    if kind == "complete":
        return "COMPLETE"
    if kind == "impossible":
        return "IMPOSSIBLE"
    return f"UNKNOWN [{json.dumps(action, ensure_ascii=False)}]"


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
        return "RIGHT" if dx > 0 else "LEFT"
    if abs(dy) > 0:
        return "DOWN" if dy > 0 else "UP"
    return None


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
