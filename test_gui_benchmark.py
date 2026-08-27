from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parent
TEST_FRAMEWORK = REPO_ROOT / "test_framework"
if str(TEST_FRAMEWORK) not in sys.path:
    sys.path.insert(0, str(TEST_FRAMEWORK))

from hf_gui_baseline import DEFAULT_MODEL_PATH, infer_one, load_model_and_processor, mock_infer_one


def response_arg(value: Optional[str]) -> Optional[str]:
    if value and value.startswith("@"):
        return Path(value[1:]).read_text(encoding="utf-8-sig").strip()
    return value


INSTRUCTION_KEYS = ("instruction", "task", "query", "goal", "question", "任务")
IMAGE_KEYS = ("image", "image_path", "screenshot", "path", "img", "图片", "截图")
ID_KEYS = ("id", "sample_id", "task_id", "case_id", "uid")


def first_value(item: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for key in keys:
        value = item.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def iter_json_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            records.append(item)
        return records

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            for key in ("samples", "tasks", "data", "examples"):
                value = data.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            return [data]
    raise ValueError("Only .json and .jsonl benchmark files are supported")


def normalize_record(item: Dict[str, Any], index: int, data_dir: Path) -> Dict[str, Any]:
    instruction = first_value(item, INSTRUCTION_KEYS)
    image = first_value(item, IMAGE_KEYS)
    if instruction is None or image is None:
        raise ValueError(f"Sample {index} is missing instruction/task or image path")

    image_path = Path(str(image))
    if not image_path.is_absolute():
        image_path = data_dir / image_path
    sample_id = first_value(item, ID_KEYS)
    if sample_id is None:
        sample_id = index

    history = item.get("history") or item.get("previous_actions") or []
    if not isinstance(history, list):
        history = []

    return {
        "sample_id": sample_id,
        "instruction": str(instruction),
        "image_path": image_path,
        "history": history,
        "low_level": item.get("low_level") or item.get("low-level") or item.get("sop"),
        "source": item,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Small static GUI benchmark with local HF model")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--samples", type=Path, required=True, help="JSON/JSONL file containing GUI samples")
    parser.add_argument("--data_dir", type=Path, help="Base directory for relative image paths; defaults to samples file directory")
    parser.add_argument("--output", type=Path, default=Path("outputs/gui_benchmark/results.jsonl"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"))
    parser.add_argument("--attn_implementation", default=None)
    parser.add_argument("--mock_response", type=response_arg, help="Skip model loading and parse this response for every sample; use @file to avoid shell JSON quoting")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data_dir = args.data_dir or args.samples.parent
    raw_records = iter_json_records(args.samples)
    selected = raw_records[args.start:]
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise SystemExit("No samples selected")

    samples = [normalize_record(item, args.start + idx, data_dir) for idx, item in enumerate(selected)]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = None
    processor = None
    if args.mock_response is None:
        model, processor = load_model_and_processor(
            args.model_path,
            device=args.device,
            device_map=args.device_map or None,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
        )

    with args.output.open("a", encoding="utf-8") as file:
        for position, sample in enumerate(samples, 1):
            image_path = sample["image_path"]
            print(
                f"BENCH_SAMPLE_START {position}/{len(samples)} id={sample['sample_id']} image={image_path}",
                flush=True,
            )
            if not image_path.is_file():
                record = {
                    "sample_id": sample["sample_id"],
                    "instruction": sample["instruction"],
                    "image_path": str(image_path),
                    "status": "error",
                    "error": f"image not found: {image_path}",
                }
            else:
                try:
                    if args.mock_response is not None:
                        result = mock_infer_one(
                            image_path,
                            sample["instruction"],
                            args.mock_response,
                            history=sample["history"],
                            low_level=sample["low_level"],
                        )
                    else:
                        result = infer_one(
                            model,
                            processor,
                            image_path,
                            sample["instruction"],
                            max_new_tokens=args.max_new_tokens,
                            device=args.device,
                            history=sample["history"],
                            low_level=sample["low_level"],
                        )
                    record = {
                        "sample_id": sample["sample_id"],
                        "instruction": sample["instruction"],
                        "image_path": str(image_path),
                        "raw_response": result.raw_response,
                        "parsed_action": result.parsed_action,
                        "latency_seconds": result.latency_seconds,
                        "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens,
                        "status": "ok",
                    }
                except Exception as exc:
                    record = {
                        "sample_id": sample["sample_id"],
                        "instruction": sample["instruction"],
                        "image_path": str(image_path),
                        "status": "error",
                        "error": str(exc),
                    }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            print(
                f"BENCH_SAMPLE_DONE id={sample['sample_id']} status={record['status']}",
                flush=True,
            )
    print(f"BENCH_RUN_DONE output={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
