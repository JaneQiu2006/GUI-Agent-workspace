from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List

from androidcontrol_actions import gt_action_to_command


def bytes_feature(example: Any, key: str) -> List[bytes]:
    return list(example.features.feature[key].bytes_list.value)


def int_feature(example: Any, key: str) -> List[int]:
    return [int(value) for value in example.features.feature[key].int64_list.value]


def text_from_bytes(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def parse_action(value: bytes) -> Dict[str, Any]:
    return json.loads(text_from_bytes(value))


def build_task(goal: str, step_instruction: str) -> str:
    step = step_instruction.strip()
    if step and step != goal:
        return f"目标任务：{goal}\n当前步骤：{step}"
    return goal


def prepare_androidcontrol(
    input_path: Path,
    output_dir: Path,
    num_episodes: int,
    compression_type: str,
) -> Dict[str, Any]:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "scripts/prepare_androidcontrol.py requires TensorFlow to read GZIP TFRecord. "
            "Install tensorflow-cpu in the preprocessing environment."
        ) from exc

    image_root = output_dir / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    samples: List[Dict[str, Any]] = []
    episode_count = 0

    dataset = tf.data.TFRecordDataset([str(input_path)], compression_type=compression_type)
    for raw_record in dataset:
        if episode_count >= num_episodes:
            break
        example = tf.train.Example.FromString(raw_record.numpy())
        episode_ids = int_feature(example, "episode_id")
        episode_id = episode_ids[0] if episode_ids else episode_count
        goal_values = bytes_feature(example, "goal")
        goal = text_from_bytes(goal_values[0]) if goal_values else ""
        screenshots = bytes_feature(example, "screenshots")
        actions = bytes_feature(example, "actions")
        step_instructions = [text_from_bytes(value) for value in bytes_feature(example, "step_instructions")]
        widths = int_feature(example, "screenshot_widths")
        heights = int_feature(example, "screenshot_heights")
        if not goal or not screenshots or not actions:
            continue

        episode_dir = image_root / f"episode_{episode_id}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        num_steps = min(len(actions), len(screenshots), len(step_instructions) or len(actions))
        for step_id in range(num_steps):
            width = widths[step_id] if step_id < len(widths) else 0
            height = heights[step_id] if step_id < len(heights) else 0
            image_rel = Path("images") / f"episode_{episode_id}" / f"step_{step_id}.png"
            image_abs = output_dir / image_rel
            image_abs.write_bytes(screenshots[step_id])
            action = parse_action(actions[step_id])
            step_instruction = step_instructions[step_id] if step_id < len(step_instructions) else ""
            samples.append(
                {
                    "episode_id": episode_id,
                    "step_id": step_id,
                    "task": build_task(goal, step_instruction),
                    "goal": goal,
                    "step_instruction": step_instruction,
                    "image_path": str(image_rel).replace("\\", "/"),
                    "action": gt_action_to_command(action, width, height),
                    "raw_action": action,
                    "screenshot_width": width,
                    "screenshot_height": height,
                }
            )
        episode_count += 1

    metadata = {
        "source": str(input_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_episodes": episode_count,
        "num_samples": len(samples),
        "samples": samples,
        "notes": [
            "Parsed from AndroidControl GZIP TFRecord.",
            "Expected official fields: episode_id, goal, screenshots, screenshot_widths, screenshot_heights, actions, step_instructions.",
            "Click and long_press coordinates are converted from screenshot pixels to 0-1000 normalized coordinates.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "test.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a mini AndroidControl static GUI benchmark")
    parser.add_argument("--input", type=Path, required=True, help="AndroidControl TFRecord shard")
    parser.add_argument("--output_dir", type=Path, default=Path("data/androidcontrol_mini"))
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--compression_type", default="GZIP", help="TFRecord compression type; official AndroidControl uses GZIP")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metadata = prepare_androidcontrol(
        input_path=args.input,
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        compression_type=args.compression_type,
    )
    print(
        f"ANDROIDCONTROL_PREPARED episodes={metadata['num_episodes']} "
        f"samples={metadata['num_samples']} output={args.output_dir / 'test.json'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
