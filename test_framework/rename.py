import argparse
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description="交换截图文件名的两个字段")
    parser.add_argument("directory", nargs="?", default="outputs/evaluation/images")
    args = parser.parse_args()
    folder = Path(args.directory)
    if not folder.is_dir():
        raise SystemExit(f"目录不存在: {folder}")

    for path in folder.glob("*.png"):
        match = re.match(r"(.*?)_(.*?)\.png$", path.name)
        if match:
            first, second = match.groups()
            new_path = path.with_name(f"{second}_{first}.png")
            path.rename(new_path)
            print(f"重命名: {path.name} -> {new_path.name}")


if __name__ == "__main__":
    main()
