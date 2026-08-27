import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="绘制轨迹延迟图")
    parser.add_argument("--input", default="outputs/timing.json")
    parser.add_argument("--output", default="outputs/latency_plot.png")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    latencies = [
        item.get("model_latency", item.get("latency"))
        for item in data
        if "model_latency" in item or "latency" in item
    ]
    average_latency = sum(latencies) / len(latencies) if latencies else 0
    print(f"Average latency: {average_latency:.4f}")

    plt.figure(figsize=(10, 5))
    plt.scatter(range(1, len(latencies) + 1), latencies)
    plt.xlabel("Element Index")
    plt.ylabel("Latency")
    plt.title("Latency per Element")
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
