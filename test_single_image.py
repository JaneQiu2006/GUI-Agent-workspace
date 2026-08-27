from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parent
TEST_FRAMEWORK = REPO_ROOT / "test_framework"
if str(TEST_FRAMEWORK) not in sys.path:
    sys.path.insert(0, str(TEST_FRAMEWORK))

from hf_gui_baseline import DEFAULT_MODEL_PATH, infer_one, load_model_and_processor, mock_infer_one


def response_arg(value: Optional[str]) -> Optional[str]:
    if value and value.startswith("@"):
        return Path(value[1:]).read_text(encoding="utf-8-sig").strip()
    return value


def load_history(value: str) -> List[Dict[str, Any]]:
    history = json.loads(value)
    if not isinstance(history, list):
        raise argparse.ArgumentTypeError("--history must be a JSON list")
    return history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single static GUI screenshot inference with local HF model")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--device_map", default="auto", help="Transformers device_map; use '' to disable")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"))
    parser.add_argument("--attn_implementation", default=None)
    parser.add_argument("--history", default="[]", type=load_history)
    parser.add_argument("--low_level", default=None, help="Optional low-level/reference steps added to the prompt")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    parser.add_argument("--mock_response", type=response_arg, help="Skip model loading and parse this response instead; use @file to avoid shell JSON quoting")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")

    if args.mock_response is not None:
        result = mock_infer_one(
            args.image,
            args.instruction,
            args.mock_response,
            history=args.history,
            low_level=args.low_level,
        )
    else:
        model, processor = load_model_and_processor(
            args.model_path,
            device=args.device,
            device_map=args.device_map or None,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
        )
        result = infer_one(
            model,
            processor,
            args.image,
            args.instruction,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            history=args.history,
            low_level=args.low_level,
        )

    payload: Dict[str, Optional[Any]] = {
        "image": str(args.image),
        "instruction": args.instruction,
        "raw_response": result.raw_response,
        "parsed_action": result.parsed_action,
        "latency_seconds": result.latency_seconds,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
