from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from test_gui_benchmark import ID_KEYS, IMAGE_KEYS, INSTRUCTION_KEYS, first_value, iter_json_records


def relative_or_string(path_value: Any, data_dir: Optional[Path]) -> str:
    path = Path(str(path_value))
    if data_dir and path.is_absolute():
        try:
            return str(path.relative_to(data_dir))
        except ValueError:
            return str(path)
    return str(path)


def normalize_records(source: Path, data_dir: Optional[Path]) -> Iterable[Dict[str, Any]]:
    for index, item in enumerate(iter_json_records(source)):
        instruction = first_value(item, INSTRUCTION_KEYS)
        image = first_value(item, IMAGE_KEYS)
        if instruction is None or image is None:
            continue
        sample_id = first_value(item, ID_KEYS)
        yield {
            "sample_id": sample_id if sample_id is not None else index,
            "instruction": str(instruction),
            "image_path": relative_or_string(image, data_dir),
            "history": item.get("history") or item.get("previous_actions") or [],
            "low_level": item.get("low_level") or item.get("low-level") or item.get("sop"),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize GUI JSON/JSONL samples for test_gui_benchmark.py")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, help="Optional dataset root used to make absolute image paths relative")
    parser.add_argument("--limit", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    records = list(normalize_records(args.input, args.data_dir))
    if args.limit is not None:
        records = records[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"PREPARED {len(records)} samples -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
