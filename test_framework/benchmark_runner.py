"""Run GUI benchmark tasks against a phone through an OpenAI-compatible VLM API."""

from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime, timezone
import http.client
import json
import math
import os
from pathlib import Path
import re
import textwrap
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlsplit

from adb_bridge import AdbBridge
from phone_prompt import PHONE_SYSTEM_PROMPT


SYSTEM_PROMPT = PHONE_SYSTEM_PROMPT


APP_PACKAGES = {
    "腾讯视频": "com.tencent.qqlive",
    "QQ音乐": "com.tencent.qqmusic",
    "爱奇艺": "com.qiyi.video",
    "优酷视频": "com.youku.phone",
    "芒果TV": "com.hunantv.imgo.activity",
    "红果": "com.phoenix.read",
    "网易云音乐": "com.netease.cloudmusic",
    "酷狗音乐": "com.kugou.android",
    "应用市场": "com.huawei.appmarket",
    "中国联通": "com.sinovatech.unicom.ui",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_task_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("tasks", [])
        records = []
        for item in data:
            if isinstance(item, str):
                record = {"task": item}
            elif isinstance(item, dict):
                record = dict(item)
                record["task"] = item.get("task") or item.get("instruction") or item.get("任务")
            else:
                record = {}
            if record.get("task") and str(record["task"]).strip():
                record["task"] = str(record["task"]).strip()
                records.append(record)
        return records
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        records = []
        for row in rows:
            task = (row.get("任务") or row.get("task") or row.get("instruction") or "").strip()
            if not task:
                continue
            expected = (row.get("预期步数") or "").strip()
            records.append({
                "task": task,
                "case_id": (row.get("用例编号") or "").strip(),
                "app": (row.get("涉及APP") or "").strip(),
                "expected_steps": int(expected) if expected.isdigit() else None,
                "sop": (row.get("SOP") or "").strip(),
                "source": row,
            })
        return records
    return [
        {"task": line.strip()}
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize_action(action: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize common coordinate shapes emitted by vision-language models."""
    kind = str(action.get("action", "")).strip().lower()
    action["action"] = kind
    if kind == "tap":
        point = action.get("point") or action.get("position") or action.get("coordinate")
        if point is None and isinstance(action.get("x"), (list, tuple)):
            point = action["x"]
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            action["x"], action["y"] = point[0], point[1]
    if kind == "swipe":
        points = action.get("points")
        if isinstance(points, (list, tuple)) and len(points) >= 2:
            action.setdefault("start", points[0])
            action.setdefault("end", points[1])
        if isinstance(action.get("x1"), (list, tuple)) and "y1" not in action:
            action["x1"], action["y1"] = action["x1"][:2]
        if isinstance(action.get("x2"), (list, tuple)) and "y2" not in action:
            action["x2"], action["y2"] = action["x2"][:2]
    return action


def parse_action(text: str) -> Dict[str, Any]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("action"):
            return normalize_action(value)
    return {"action": "invalid", "raw": text}


class VllmClient:
    def __init__(self, base_url: str, model: str, timeout: int = 600) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.timeout = timeout

    def infer(
        self,
        task: str,
        image_path: Path,
        history: List[Dict[str, Any]],
    ) -> Tuple[str, Dict[str, float], Dict[str, Any]]:
        inference_started_at = time.perf_counter()
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        history_text = json.dumps(history[-8:], ensure_ascii=False)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"任务：{task}\n历史动作：{history_text}",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                },
            ],
            "temperature": 0.0,
            "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        body = json.dumps(payload).encode("utf-8")
        parsed = urlsplit(self.url)
        connection_class = (
            http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        )
        connection = connection_class(parsed.hostname, parsed.port, timeout=self.timeout)
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        chunks: List[str] = []
        usage: Dict[str, Any] = {}
        first_token_at: Optional[float] = None
        last_token_at: Optional[float] = None
        try:
            connection.putrequest("POST", path)
            connection.putheader("Content-Type", "application/json")
            connection.putheader("Content-Length", str(len(body)))
            connection.endheaders()
            connection.send(body)
            input_completed_at = time.perf_counter()
            response = connection.getresponse()
            if response.status >= 400:
                detail = response.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"模型 API HTTP {response.status}: {detail}")
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                event = json.loads(data)
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                token_text = delta.get("content") or delta.get("reasoning_content") or ""
                if token_text:
                    token_received_at = time.perf_counter()
                    if first_token_at is None:
                        first_token_at = token_received_at
                    last_token_at = token_received_at
                    if delta.get("content"):
                        chunks.append(delta["content"])
        finally:
            connection.close()
        if first_token_at is None:
            raise RuntimeError("模型流式响应中未收到生成 token")
        assert last_token_at is not None
        timing = {
            "ttft_seconds": first_token_at - input_completed_at,
            "model_total_seconds": last_token_at - input_completed_at,
            "request_e2e_seconds": last_token_at - inference_started_at,
        }
        return "".join(chunks), timing, usage


def screen_coordinate(value: Any, dimension: int) -> int:
    number = float(value)
    normalized = number if 0 <= number <= 1 else number / 1000
    return int(max(0, min(1, normalized)) * dimension)


def resolve_swipe(action: Dict[str, Any], size: Tuple[int, int]) -> Tuple[int, int, int, int, int]:
    width, height = size
    kind = str(action.get("action", "swipe")).lower()
    direction = str(action.get("direction") or "").lower()
    if "_" in kind:
        direction = kind.rsplit("_", 1)[-1]
    direction_points = {
        "up": (500, 800, 500, 200),
        "down": (500, 200, 500, 800),
        "left": (800, 500, 200, 500),
        "right": (200, 500, 800, 500),
    }
    start = action.get("start") or action.get("from")
    end = action.get("end") or action.get("to")
    values = (
        action.get("x1", action.get("start_x", start[0] if start else None)),
        action.get("y1", action.get("start_y", start[1] if start else None)),
        action.get("x2", action.get("end_x", end[0] if end else None)),
        action.get("y2", action.get("end_y", end[1] if end else None)),
    )
    if any(value is None for value in values):
        values = direction_points.get(direction, direction_points["up"])
    if values[0] == values[2] and values[1] == values[3]:
        values = direction_points.get(direction, direction_points["up"])
        action["swipe_fallback"] = "zero-distance swipe replaced with default gesture"
    duration = action.get("duration_ms", action.get("duration", 500))
    if isinstance(duration, float) and duration <= 10:
        duration *= 1000
    return (
        screen_coordinate(values[0], width),
        screen_coordinate(values[1], height),
        screen_coordinate(values[2], width),
        screen_coordinate(values[3], height),
        int(duration),
    )


def annotate_screenshot(
    source: Path,
    destination: Path,
    action: Dict[str, Any],
    size: Tuple[int, int],
) -> None:
    """Draw the chosen action and its parameters on the saved screenshot."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.open(source).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    label = json.dumps(action, ensure_ascii=True, sort_keys=True)
    lines = textwrap.wrap(label, width=72) or [label]
    line_height = 36
    overlay_height = 18 + line_height * len(lines)
    draw.rectangle((0, 0, image.width, overlay_height), fill=(0, 0, 0, 205))
    for index, line in enumerate(lines):
        draw.text((16, 9 + index * line_height), line, fill=(255, 255, 255, 255), font=font)

    kind = str(action.get("action", "")).lower()
    if kind == "tap" and "x" in action and "y" in action:
        try:
            px = screen_coordinate(action["x"], size[0])
            py = screen_coordinate(action["y"], size[1])
            radius = 30
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), outline=(255, 40, 40, 255), width=8)
            draw.line((px - radius, py, px + radius, py), fill=(255, 40, 40, 255), width=5)
            draw.line((px, py - radius, px, py + radius), fill=(255, 40, 40, 255), width=5)
        except (TypeError, ValueError):
            pass
    elif kind == "swipe" or kind == "scroll" or kind.startswith("swipe_") or kind.startswith("scroll_"):
        try:
            x1, y1, x2, y2, _ = resolve_swipe(action, size)
            color = (255, 45, 45, 255)
            draw.line((x1, y1, x2, y2), fill=color, width=10)
            angle = math.atan2(y2 - y1, x2 - x1)
            arrow = 34
            left = (x2 - arrow * math.cos(angle - math.pi / 6), y2 - arrow * math.sin(angle - math.pi / 6))
            right = (x2 - arrow * math.cos(angle + math.pi / 6), y2 - arrow * math.sin(angle + math.pi / 6))
            draw.polygon(((x2, y2), left, right), fill=color)
            draw.ellipse((x1 - 12, y1 - 12, x1 + 12, y1 + 12), fill=(255, 220, 0, 255))
        except (KeyError, TypeError, ValueError):
            pass
    image.convert("RGB").save(destination, format="PNG")


def launch_task_app(bridge: AdbBridge, app_name: Optional[str]) -> Optional[str]:
    package = APP_PACKAGES.get((app_name or "").strip())
    if not package:
        return None
    bridge.shell("monkey", "-p", package, "-c", "android.intent.category.LAUNCHER", "1", timeout=45)
    return package


def execute_action(bridge: AdbBridge, action: Dict[str, Any], size: Tuple[int, int]) -> bool:
    kind = str(action.get("action", "invalid")).lower()
    width, height = size
    x = lambda value: screen_coordinate(value, width)
    y = lambda value: screen_coordinate(value, height)
    if kind == "tap":
        bridge.tap(x(action["x"]), y(action["y"]))
    elif kind == "swipe" or kind == "scroll" or kind.startswith("swipe_") or kind.startswith("scroll_"):
        bridge.swipe(*resolve_swipe(action, size))
    elif kind == "type":
        input_text = str(action.get("text", ""))
        if not input_text:
            action["input_skipped"] = "empty text"
            return False
        encoded = base64.b64encode(input_text.encode("utf-8")).decode("ascii")
        bridge.shell("am", "broadcast", "-a", "ADB_INPUT_B64", "--es", "msg", encoded)
    elif kind == "back":
        bridge.shell("input", "keyevent", "4")
    elif kind == "home":
        bridge.shell("input", "keyevent", "3")
    elif kind == "enter":
        bridge.shell("input", "keyevent", "66")
    elif kind == "wait":
        time.sleep(max(0, min(15, float(action.get("seconds", 2)))))
    elif kind in {"complete", "impossible", "invalid"}:
        return True
    else:
        action["error"] = f"unknown action: {kind}"
        return True
    return False


def append_jsonl(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, ensure_ascii=False) + "\n")


def append_latency(path: Path, row: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        if not exists:
            writer.writerow([
                "task_id", "step", "ttft_seconds", "model_total_seconds",
                "request_e2e_seconds", "step_e2e_seconds", "input_tokens", "output_tokens",
            ])
        writer.writerow(row)


def append_task_timing(path: Path, summary: Dict[str, Any]) -> None:
    exists = path.exists()
    if exists:
        with path.open("r", encoding="utf-8", newline="") as file:
            if any(row.get("task_id") == str(summary.get("task_id")) for row in csv.DictReader(file)):
                return
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        if not exists:
            writer.writerow(["task_id", "case_id", "status", "steps", "total_seconds"])
        writer.writerow([
            summary.get("task_id"),
            summary.get("case_id"),
            summary.get("status"),
            summary.get("steps", 0),
            f"{summary.get('total_seconds', 0.0):.6f}",
        ])


def run_task(
    bridge: AdbBridge,
    client: Any,
    task_id: int,
    task_record: Dict[str, Any],
    output_root: Path,
    max_steps: int,
    settle_seconds: float,
) -> Dict[str, Any]:
    task = task_record["task"]
    model_task = task
    if task_record.get("app"):
        model_task += f"\n目标 APP：{task_record['app']}"
    if task_record.get("sop"):
        model_task += f"\n参考步骤：\n{task_record['sop']}"
    case_id = re.sub(r"[^0-9A-Za-z._-]+", "_", task_record.get("case_id") or "task")
    task_dir = output_root / f"task_{task_id:04d}_{case_id}"
    done_path = task_dir / "done.json"
    if done_path.is_file():
        return json.loads(done_path.read_text(encoding="utf-8"))
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.txt").write_text(task + "\n", encoding="utf-8")
    (task_dir / "metadata.json").write_text(
        json.dumps(task_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    task_started = time.perf_counter()
    bridge.shell("input", "keyevent", "WAKEUP")
    bridge.shell("input", "keyevent", "3")
    launched_package = launch_task_app(bridge, task_record.get("app"))
    time.sleep(settle_seconds)
    size = bridge.display_size()
    history: List[Dict[str, Any]] = []
    status = "max_steps"

    for step in range(max_steps):
        step_started = time.perf_counter()
        raw_screenshot = bridge.screenshot(task_dir / f".step_{step:03d}_raw.png")
        screenshot = task_dir / f"step_{step:03d}.png"
        try:
            response, timing, usage = client.infer(model_task, raw_screenshot, history)
        except Exception as exc:
            annotate_screenshot(raw_screenshot, screenshot, {"action": "model_error", "error": str(exc)}, size)
            raw_screenshot.unlink(missing_ok=True)
            raise
        action = parse_action(response)
        capture_mode = bridge.last_screenshot_mode
        if capture_mode != "direct_screencap":
            action["capture_mode"] = capture_mode
        annotate_screenshot(raw_screenshot, screenshot, action, size)
        raw_screenshot.unlink(missing_ok=True)
        history.append(action)
        execution_error = None
        try:
            terminal = execute_action(bridge, action, size)
        except (KeyError, TypeError, ValueError) as exc:
            execution_error = str(exc)
            action["execution_error"] = execution_error
            terminal = True
        step_e2e = time.perf_counter() - step_started
        record = {
            "task_id": task_id,
            "task": task,
            "step": step,
            "timestamp": utc_now(),
            "screenshot": str(screenshot),
            "model_response": response,
            "action": action,
            "capture_mode": capture_mode,
            **timing,
            "step_e2e_seconds": step_e2e,
            "usage": usage,
        }
        append_jsonl(task_dir / "steps.jsonl", record)
        append_latency(
            output_root / "latency.csv",
            [
                task_id, step,
                f"{timing['ttft_seconds']:.6f}",
                f"{timing['model_total_seconds']:.6f}",
                f"{timing['request_e2e_seconds']:.6f}",
                f"{step_e2e:.6f}",
                usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
            ],
        )
        if terminal:
            status = "action_error" if execution_error else action.get("action", "invalid")
            break
        time.sleep(settle_seconds)

    raw_final = bridge.screenshot(task_dir / ".final_raw.png")
    annotate_screenshot(
        raw_final,
        task_dir / "final.png",
        {"action": "final", "capture_mode": bridge.last_screenshot_mode},
        size,
    )
    raw_final.unlink(missing_ok=True)
    bridge.shell("input", "keyevent", "3")
    summary = {
        "task_id": task_id,
        "task": task,
        "case_id": task_record.get("case_id"),
        "app": task_record.get("app"),
        "launched_package": launched_package,
        "status": status,
        "steps": len(history),
        "total_seconds": time.perf_counter() - task_started,
        "finished_at": utc_now(),
    }
    done_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    append_jsonl(output_root / "summary.jsonl", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="华为手机 GUI 批量测评")
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8018/v1")
    parser.add_argument("--model", default="/data2/home/luyijie/models/Qwen3.8-27B")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--settle-seconds", type=float, default=1.5)
    parser.add_argument("--start-task", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    task_records = load_task_records(args.tasks)
    selected = list(enumerate(task_records))[args.start_task:]
    if args.limit is not None:
        selected = selected[:args.limit]
    if not selected:
        raise SystemExit("任务列表为空")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run_config.json").write_text(
        json.dumps(vars(args), default=str, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "metrics_definition.json").write_text(
        json.dumps(
            {
                "ttft_seconds": "请求体发送完成到收到首个生成 token",
                "model_total_seconds": "请求体发送完成到收到最后一个生成 token",
                "request_e2e_seconds": "开始读取并编码当前截图到收到最后一个生成 token",
                "step_e2e_seconds": "开始截取当前手机画面到模型动作在手机上执行完成",
                "total_seconds": "单项任务开始（唤醒/回到桌面前）到最终截图及回到桌面完成",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    bridge = AdbBridge(transport="auto")
    serial = bridge.ensure_ready()
    bridge.shell("svc", "power", "stayon", "true")
    client = VllmClient(args.base_url, args.model)
    print(f"RUN_START tasks={len(selected)} transport={bridge.transport} serial={serial}", flush=True)
    original_ime = bridge.shell("settings", "get", "secure", "default_input_method")
    try:
        bridge.shell("ime", "enable", "com.android.adbkeyboard/.AdbIME")
        bridge.shell("ime", "set", "com.android.adbkeyboard/.AdbIME")
        for position, (task_id, task_record) in enumerate(selected, 1):
            task = task_record["task"]
            print(f"TASK_START {position}/{len(selected)} id={task_id} task={task}", flush=True)
            task_attempt_started = time.perf_counter()
            try:
                expected_steps = task_record.get("expected_steps")
                task_max_steps = max(args.max_steps, expected_steps + 3) if expected_steps else args.max_steps
                summary = run_task(
                    bridge, client, task_id, task_record, args.output, task_max_steps, args.settle_seconds
                )
            except Exception as exc:
                summary = {
                    "task_id": task_id,
                    "task": task,
                    "case_id": task_record.get("case_id"),
                    "app": task_record.get("app"),
                    "status": "error",
                    "error": str(exc),
                    "steps": 0,
                    "total_seconds": time.perf_counter() - task_attempt_started,
                    "finished_at": utc_now(),
                }
                append_jsonl(args.output / "summary.jsonl", summary)
            append_task_timing(args.output / "task_timing.csv", summary)
            print("TASK_DONE " + json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        if original_ime and original_ime != "null":
            bridge.shell("ime", "set", original_ime)
    print("RUN_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
