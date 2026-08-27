#!/usr/bin/env python3
"""Serve the GUI benchmark review console and persist human annotations."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import tempfile
import threading
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "outputs" / "qwen38_gui_dev_annotated_full_v2_20260821"
STATIC_ROOT = Path(__file__).resolve().parent / "static"
ANNOTATIONS_NAME = "review_annotations.json"
TASK_DIR_RE = re.compile(r"^task_(\d+)(?:_|$)")
ALLOWED_VERDICTS = {"unreviewed", "correct", "incorrect"}
MAX_REQUEST_BYTES = 2 * 1024 * 1024


class DatasetError(RuntimeError):
    """Raised when the benchmark dataset cannot be loaded."""


class AnnotationError(RuntimeError):
    """Raised when the annotation file cannot be safely read or written."""


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"无法解析 {path} 第 {line_number} 行: {exc}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


class ReviewRepository:
    """Read benchmark artifacts and serialize annotation writes."""

    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root.expanduser().resolve()
        self.annotation_path = self.dataset_root / ANNOTATIONS_NAME
        self._write_lock = threading.RLock()

    @property
    def dataset_name(self) -> str:
        return self.dataset_root.name

    def validate(self) -> None:
        if not self.dataset_root.exists():
            raise DatasetError(f"数据集不存在: {self.dataset_root}")
        if not self.dataset_root.is_dir():
            raise DatasetError(f"数据集路径不是目录: {self.dataset_root}")
        if not os.access(str(self.dataset_root), os.R_OK):
            raise DatasetError(f"数据集不可读: {self.dataset_root}")

    def _task_dirs(self) -> List[Tuple[int, Path]]:
        self.validate()
        found: List[Tuple[int, Path]] = []
        for path in self.dataset_root.iterdir():
            match = TASK_DIR_RE.match(path.name)
            if match and path.is_dir() and is_within(path, self.dataset_root):
                found.append((int(match.group(1)), path))
        return sorted(found, key=lambda item: item[0])

    def _relative_image(self, candidate: Any, task_dir: Path) -> Optional[str]:
        if not candidate or not isinstance(candidate, str):
            return None
        raw = Path(candidate)
        candidates: Iterable[Path]
        if raw.is_absolute():
            candidates = (raw,)
        else:
            candidates = (
                PROJECT_ROOT / raw,
                self.dataset_root / raw,
                task_dir / raw.name,
            )
        for value in candidates:
            resolved = value.resolve()
            if resolved.exists() and resolved.is_file() and is_within(resolved, self.dataset_root):
                return resolved.relative_to(self.dataset_root).as_posix()
        return None

    def load_annotations(self) -> Tuple[Dict[str, Any], Optional[str]]:
        base: Dict[str, Any] = {
            "schema_version": 1,
            "dataset": self.dataset_name,
            "updated_at": None,
            "reviews": {},
        }
        if not self.annotation_path.exists():
            return base, None
        try:
            loaded = read_json(self.annotation_path)
        except (OSError, json.JSONDecodeError) as exc:
            return base, f"标注文件格式错误，已禁止写入: {self.annotation_path} ({exc})"
        if not isinstance(loaded, dict) or not isinstance(loaded.get("reviews", {}), dict):
            return base, f"标注文件结构错误，已禁止写入: {self.annotation_path}"
        merged = {**base, **loaded}
        merged["reviews"] = loaded.get("reviews", {})
        return merged, None

    def _load_task(self, task_id: int, task_dir: Path) -> Dict[str, Any]:
        metadata = read_json(task_dir / "metadata.json", {}) or {}
        done = read_json(task_dir / "done.json", {}) or {}
        steps = read_jsonl(task_dir / "steps.jsonl")
        frames: List[Dict[str, Any]] = []
        referenced: set[str] = set()
        for index, step in enumerate(steps):
            relative = self._relative_image(step.get("screenshot"), task_dir)
            if relative:
                referenced.add(relative)
            action = step.get("action") if isinstance(step.get("action"), dict) else {}
            frames.append({
                "kind": "step",
                "frame_index": index,
                "step": step.get("step", index),
                "image": f"/media/{relative}" if relative else None,
                "filename": Path(relative).name if relative else str(step.get("screenshot") or "未记录截图"),
                "action": action,
                "action_type": action.get("action") or "unknown",
                "model_response": step.get("model_response"),
                "timestamp": step.get("timestamp"),
                "capture_mode": step.get("capture_mode"),
                "ttft_seconds": step.get("ttft_seconds"),
                "model_total_seconds": step.get("model_total_seconds"),
                "request_e2e_seconds": step.get("request_e2e_seconds"),
                "step_e2e_seconds": step.get("step_e2e_seconds"),
            })
        final_path = task_dir / "final.png"
        if final_path.exists():
            relative = final_path.resolve().relative_to(self.dataset_root).as_posix()
            if relative not in referenced:
                frames.append({
                    "kind": "final",
                    "frame_index": len(frames),
                    "step": None,
                    "image": f"/media/{relative}",
                    "filename": final_path.name,
                    "action": {},
                    "action_type": "FINAL",
                    "model_response": None,
                    "timestamp": done.get("finished_at"),
                    "capture_mode": None,
                    "ttft_seconds": None,
                    "model_total_seconds": None,
                    "request_e2e_seconds": None,
                    "step_e2e_seconds": None,
                })
        task_text = metadata.get("task") or done.get("task")
        if not task_text:
            task_file = task_dir / "task.txt"
            task_text = task_file.read_text(encoding="utf-8").strip() if task_file.exists() else task_dir.name
        return {
            "task_id": task_id,
            "directory": task_dir.name,
            "task": task_text,
            "case_id": metadata.get("case_id") or done.get("case_id") or "",
            "app": metadata.get("app") or done.get("app") or "",
            "expected_steps": metadata.get("expected_steps"),
            "sop": metadata.get("sop") or "",
            "source": metadata.get("source") if isinstance(metadata.get("source"), dict) else {},
            "model_status": done.get("status"),
            "launched_package": done.get("launched_package"),
            "recorded_steps": done.get("steps", len(steps)),
            "total_seconds": done.get("total_seconds"),
            "finished_at": done.get("finished_at"),
            "frames": frames,
        }

    def payload(self) -> Dict[str, Any]:
        annotations, annotation_error = self.load_annotations()
        tasks = [self._load_task(task_id, path) for task_id, path in self._task_dirs()]
        reviews = annotations.get("reviews", {})
        for task in tasks:
            review = reviews.get(str(task["task_id"]), {})
            task["review"] = review if isinstance(review, dict) else {}
        return {
            "dataset": self.dataset_name,
            "dataset_path": str(self.dataset_root),
            "annotation_path": str(self.annotation_path),
            "annotation_writable": annotation_error is None,
            "annotation_error": annotation_error,
            "tasks": tasks,
        }

    def update_review(self, task_id: int, verdict: str, note: str) -> Dict[str, Any]:
        if verdict not in ALLOWED_VERDICTS:
            raise ValueError(f"不支持的 verdict: {verdict}")
        if not isinstance(note, str):
            raise ValueError("note 必须是字符串")
        task_lookup = {item_id: path for item_id, path in self._task_dirs()}
        if task_id not in task_lookup:
            raise KeyError(f"任务不存在: {task_id}")
        metadata = read_json(task_lookup[task_id] / "metadata.json", {}) or {}
        with self._write_lock:
            document, annotation_error = self.load_annotations()
            if annotation_error:
                raise AnnotationError(annotation_error)
            reviews = document.setdefault("reviews", {})
            key = str(task_id)
            if verdict == "unreviewed":
                reviews.pop(key, None)
                saved_review: Dict[str, Any] = {}
            else:
                existing = reviews.get(key, {})
                if not isinstance(existing, dict):
                    existing = {}
                saved_review = {
                    **existing,
                    "task_id": task_id,
                    "case_id": metadata.get("case_id") or existing.get("case_id") or "",
                    "verdict": verdict,
                    "note": note,
                    "reviewed_at": now_iso(),
                }
                reviews[key] = saved_review
            document.setdefault("schema_version", 1)
            document.setdefault("dataset", self.dataset_name)
            document["updated_at"] = now_iso()
            self._atomic_write(document)
            return {"review": saved_review, "updated_at": document["updated_at"]}

    def _atomic_write(self, document: Dict[str, Any]) -> None:
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        temporary_name: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(self.dataset_root),
                prefix=f".{ANNOTATIONS_NAME}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_name = handle.name
                json.dump(document, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.annotation_path)
            temporary_name = None
        except OSError as exc:
            raise AnnotationError(f"无法写入标注文件 {self.annotation_path}: {exc}") from exc
        finally:
            if temporary_name:
                try:
                    os.unlink(temporary_name)
                except OSError:
                    pass


class ReviewHandler(BaseHTTPRequestHandler):
    repository: ReviewRepository
    static_root = STATIC_ROOT.resolve()

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def _json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, code: str, message: str) -> None:
        self._json(status, {"error": {"code": code, "message": message}})

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._json(HTTPStatus.OK, {"ok": True, "dataset": self.repository.dataset_name})
            return
        if parsed.path == "/api/tasks":
            try:
                self._json(HTTPStatus.OK, self.repository.payload())
            except (DatasetError, OSError, json.JSONDecodeError) as exc:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "dataset_error", str(exc))
            return
        if parsed.path.startswith("/media/"):
            self._serve_media(parsed.path[len("/media/"):])
            return
        self._serve_static(parsed.path)

    def do_PUT(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        match = re.fullmatch(r"/api/reviews/(\d+)", path)
        if not match:
            self._error(HTTPStatus.NOT_FOUND, "not_found", "接口不存在")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_REQUEST_BYTES:
            self._error(HTTPStatus.BAD_REQUEST, "invalid_length", "请求体为空或过大")
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("请求体必须是 JSON 对象")
            result = self.repository.update_review(
                int(match.group(1)),
                payload.get("verdict", ""),
                payload.get("note", ""),
            )
            self._json(HTTPStatus.OK, result)
        except json.JSONDecodeError:
            self._error(HTTPStatus.BAD_REQUEST, "invalid_json", "请求体不是有效 JSON")
        except ValueError as exc:
            self._error(HTTPStatus.BAD_REQUEST, "validation_error", str(exc))
        except KeyError as exc:
            self._error(HTTPStatus.NOT_FOUND, "task_not_found", str(exc))
        except AnnotationError as exc:
            self._error(HTTPStatus.CONFLICT, "annotation_error", str(exc))
        except OSError as exc:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "write_error", str(exc))

    def _serve_media(self, raw_relative: str) -> None:
        relative = Path(unquote(raw_relative))
        if relative.is_absolute() or ".." in relative.parts:
            self._error(HTTPStatus.FORBIDDEN, "invalid_path", "拒绝访问数据集外路径")
            return
        candidate = (self.repository.dataset_root / relative).resolve()
        if not is_within(candidate, self.repository.dataset_root):
            self._error(HTTPStatus.FORBIDDEN, "invalid_path", "拒绝访问数据集外路径")
            return
        self._serve_file(candidate, cache="private, max-age=3600")

    def _serve_static(self, raw_path: str) -> None:
        relative = "index.html" if raw_path in {"", "/"} else unquote(raw_path.lstrip("/"))
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            self._error(HTTPStatus.FORBIDDEN, "invalid_path", "拒绝访问前端目录外路径")
            return
        candidate = (self.static_root / relative).resolve()
        if not is_within(candidate, self.static_root) or not candidate.is_file():
            candidate = self.static_root / "index.html"
        self._serve_file(candidate, cache="no-cache")

    def _serve_file(self, path: Path, cache: str) -> None:
        if not path.exists() or not path.is_file():
            self._error(HTTPStatus.NOT_FOUND, "file_not_found", f"文件不存在: {path.name}")
            return
        try:
            body = path.read_bytes()
        except OSError as exc:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "file_error", str(exc))
            return
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动 GUI Dev 人工审阅台")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=8765, help="监听端口（默认 8765）")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="测评输出目录")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repository = ReviewRepository(args.dataset)
    try:
        repository.validate()
    except DatasetError as exc:
        print(exc)
        return 2
    handler = type("ConfiguredReviewHandler", (ReviewHandler,), {"repository": repository})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}"
    print(f"GUI Dev Review Bench: {url}")
    print(f"数据集: {repository.dataset_root}")
    print(f"标注文件: {repository.annotation_path}")
    if not args.no_browser:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止审阅服务")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
