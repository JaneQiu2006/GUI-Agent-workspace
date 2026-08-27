# GUI Agent Baseline

This repository keeps the existing Huawei GUI task framework under `test_framework/`.
The new static baseline scripts use the same phone prompt/action format, but load a
local HuggingFace/Transformers multimodal model directly instead of calling the
OpenAI-compatible vLLM service.

Default model path:

```bash
/data2/home/models/Qwen3.8-27B
```

## Single Image

```bash
python test_single_image.py \
  --model_path /data2/home/models/Qwen3.8-27B \
  --image xxx.png \
  --instruction "打开设置并进入 WLAN" \
  --max_new_tokens 128 \
  --device auto
```

Local static check without loading a model:

```bash
printf '%s\n' '{"action":"tap","x":500,"y":500}' > mock_response.json
python test_single_image.py \
  --image test_framework/outputs/evaluation/images/改为舒适驾驶模式_1787228768.9681878_0.png \
  --instruction "改为舒适驾驶模式" \
  --mock_response @mock_response.json
```

## Small Batch Benchmark

Input can be `.json` or `.jsonl`. Each sample should contain an instruction field
(`instruction`, `task`, `query`, `goal`, `question`, or `任务`) and an image path
field (`image`, `image_path`, `screenshot`, `path`, `img`, `图片`, or `截图`).
Relative image paths are resolved against `--data_dir`, or the sample file's
directory when `--data_dir` is omitted.

To normalize a raw GUI JSON/JSONL file into the expected shape:

```bash
python scripts/prepare_gui_jsonl.py \
  --input path/to/raw_gui_samples.json \
  --output data/gui/small_eval.jsonl \
  --data_dir data/gui \
  --limit 20
```

```bash
python test_gui_benchmark.py \
  --model_path /data2/home/models/Qwen3.8-27B \
  --samples data/gui/small_eval.jsonl \
  --data_dir data/gui \
  --output outputs/gui_benchmark/qwen38_baseline.jsonl \
  --limit 20 \
  --max_new_tokens 128 \
  --device auto
```

The benchmark writes one JSON object per line with `sample_id`, `instruction`,
`image_path`, `raw_response`, `parsed_action`, `latency_seconds`, and token counts
when available.

## Dependency Notes

Install only missing packages in the server environment:

```bash
pip install -r requirements.txt
```

Do not commit downloaded models, datasets, cache directories, or benchmark outputs.
