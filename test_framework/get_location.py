import argparse
import json
from pathlib import Path
import re


CLICK_PATTERN = re.compile(r"CLICK <point>\[\[(\d+\.?\d*),\s*(\d+\.?\d*)\]\]</point>")
SCROLL_PATTERN = re.compile(r"<point>\[\[(\d+\.?\d*),\s*(\d+\.?\d*)\]\]</point>")


def main():
    parser = argparse.ArgumentParser(description="在轨迹截图上标记点击和滑动位置")
    parser.add_argument("--input", default="outputs/evaluation/data.json")
    parser.add_argument("--output-dir", default="outputs/annotated")
    args = parser.parse_args()

    import cv2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))

    for item in data:
        action = item.get("action", "")
        image_path = Path(item.get("image_path", ""))
        if not image_path.is_file():
            print(f"Image not found: {image_path}")
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Image unreadable: {image_path}")
            continue
        height, width = image.shape[:2]
        output_path = output_dir / image_path.name

        if action.startswith("CLICK"):
            match = CLICK_PATTERN.search(action)
            if match:
                x, y = map(float, match.groups())
                point = (int(x * width / 1000), int(y * height / 1000))
                cv2.circle(image, point, radius=10, color=(0, 0, 255), thickness=3)
        elif action.startswith("SCROLL"):
            matches = SCROLL_PATTERN.findall(action)
            if len(matches) == 2:
                points = [
                    (int(float(x) * width / 1000), int(float(y) * height / 1000))
                    for x, y in matches
                ]
                cv2.circle(image, points[0], radius=10, color=(0, 255, 0), thickness=3)
                cv2.circle(image, points[1], radius=10, color=(255, 0, 0), thickness=3)
        cv2.imwrite(str(output_path), image)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
