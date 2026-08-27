from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple
import json
import re

class ActionType(Enum):
    Idle=0
    DualPoint=1
    Type=2
    GoBack=3
    GoHome=4
    Enter=5
    TaskComplete=6
    TaskImpossible=7
    WAIT=8
    LONGPRESS=9


@dataclass
class AndroidAction():
    action_type: ActionType
    touch_point: Tuple[float, float] = None
    lift_point: Tuple[float, float] = None
    typed_text: str = None
    def __str__(self):
        
        components = [f"Action Type: {self.action_type.name}"]

        if self.touch_point:
            touch_point_str = f"({self.touch_point[0]:.4f}, {self.touch_point[1]:.4f})"
            components.append(f"Touch Point: {touch_point_str}")
        if self.lift_point:
            lift_point_str = f"({self.lift_point[0]:.4f}, {self.lift_point[1]:.4f})"
            components.append(f"Lift Point: {lift_point_str}")
        if self.typed_text:
            components.append(f"Typed Text: '{self.typed_text}'")
        return ", ".join(components)

    def to_act(self):
        pass

def qwen_translate_action(out):
    out = (out or "").strip()
    if out.startswith("{"):
        try:
            value = json.loads(out)
        except json.JSONDecodeError:
            value = {}
        kind = str(value.get("action", "")).lower()
        scale = lambda coordinate: float(coordinate) if 0 <= float(coordinate) <= 1 else float(coordinate) / 1000
        if kind == "tap":
            x = value.get("x")
            y = value.get("y")
            if isinstance(x, (list, tuple)) and len(x) >= 2 and y is None:
                x, y = x[:2]
            if x is not None and y is not None:
                point = (scale(x), scale(y))
                return AndroidAction(action_type=ActionType.DualPoint, touch_point=point, lift_point=point)
        if kind == "swipe":
            x1 = value.get("x1", value.get("start_x"))
            y1 = value.get("y1", value.get("start_y"))
            x2 = value.get("x2", value.get("end_x"))
            y2 = value.get("y2", value.get("end_y"))
            if None not in (x1, y1, x2, y2):
                return AndroidAction(
                    action_type=ActionType.DualPoint,
                    touch_point=(scale(x1), scale(y1)),
                    lift_point=(scale(x2), scale(y2)),
                )
        if kind == "type":
            return AndroidAction(action_type=ActionType.Type, typed_text=str(value.get("text", "")))
        if kind == "back":
            return AndroidAction(action_type=ActionType.GoBack)
        if kind == "home":
            return AndroidAction(action_type=ActionType.GoHome)
        if kind == "enter":
            return AndroidAction(action_type=ActionType.Enter)
        if kind == "wait":
            return AndroidAction(action_type=ActionType.WAIT)
        if kind == "complete":
            return AndroidAction(action_type=ActionType.TaskComplete)
        if kind == "impossible":
            return AndroidAction(action_type=ActionType.TaskImpossible)
    if out == "PRESS_BACK":
        return AndroidAction(action_type=ActionType.GoBack)
    elif out == "PRESS_HOME":
        return AndroidAction(action_type=ActionType.GoHome)
    elif out == "ENTER":
        return AndroidAction(action_type=ActionType.Enter)
    elif out == "COMPLETE":
        return AndroidAction(action_type=ActionType.TaskComplete)
    elif out == "IMPOSSIBLE":
        return AndroidAction(action_type=ActionType.TaskImpossible)
    elif out == "SCROLL [RIGHT]":
        return AndroidAction(action_type=ActionType.DualPoint, touch_point=(0.2, 0.5), lift_point=(0.8, 0.5))
    elif out == "SCROLL [LEFT]":
        return AndroidAction(action_type=ActionType.DualPoint, touch_point=(0.8, 0.5), lift_point=(0.2, 0.5))
    elif out == "SCROLL [UP]":
        return AndroidAction(action_type=ActionType.DualPoint, touch_point=(0.5, 0.5), lift_point=(0.5, 0.2))
    elif out == "SCROLL [DOWN]":
        return AndroidAction(action_type=ActionType.DualPoint, touch_point=(0.5, 0.2), lift_point=(0.5, 0.5))

    elif out.startswith("SCROLL"):
        # 正则表达式匹配 <point>[[x, y]]</point> 形式的坐标
        pattern = r"<point>\[\[(\d+\.\d+),\s*(\d+\.\d+)\]\]</point>"
        matches = re.findall(pattern, out)
        print(matches)

        if len(matches) == 2:
            x1, y1 = map(float, matches[0]) 
            x2, y2 = map(float, matches[1])  
            print(x1)
            x1 = x1 / 1000
            x2 = x2 / 1000
            y1 = y1 / 1000
            y2 = y2 / 1000
            return AndroidAction(action_type=ActionType.DualPoint, touch_point=(x1, y1), lift_point=(x2, y2))
        return AndroidAction(action_type=ActionType.TaskImpossible)
    
    elif out.startswith("TYPE [") and out.endswith("]"):
        start = out.find("[") + 1
        end = out.find("]")
        text = out[start:end]
        return AndroidAction(action_type=ActionType.Type, typed_text=text)
    elif out.startswith("CLICK <point>[[") and out.endswith("]]</point>"):
        point_str = out.split("<point>[[")[1].split("]]</point>")[0]
        point_values = point_str.split(",")        
        x_axis = float(point_values[0].strip()) /1000
        y_axis = float(point_values[1].strip()) /1000
        touch_point=(x_axis, y_axis)
        return AndroidAction(action_type=ActionType.DualPoint, touch_point=touch_point, lift_point=touch_point)
    elif out.startswith("WAIT"):
        return AndroidAction(action_type=ActionType.WAIT)
    elif out.startswith("LONG_PRESS"):
        point_str = out.split("<point>[[")[1].split("]]</point>")[0]
        point_values = point_str.split(",")        
        x_axis = float(point_values[0].strip()) /1000
        y_axis = float(point_values[1].strip()) /1000
        touch_point = (x_axis, y_axis)
        lift_point = (x_axis, y_axis)
        return AndroidAction(action_type=ActionType.LONGPRESS, touch_point=touch_point, lift_point=touch_point)
    return AndroidAction(action_type=ActionType.TaskImpossible)
