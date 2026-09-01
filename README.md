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

## AndroidControl Smoke Run

For the raw AndroidControl GZIP TFRecord shard available on Jupiter:

```bash
python scripts/prepare_androidcontrol.py \
  --input data/raw/android_control/android_control-00000-of-00020 \
  --output_dir data/androidcontrol_mini \
  --num_episodes 10

CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_androidcontrol.py \
  --model_path /data2/home/models/Qwen3.8-27B \
  --test_json data/androidcontrol_mini/test.json \
  --output results/qwen_androidcontrol_mini.json
```

The preprocessor saves screenshots under `data/androidcontrol_mini/images/` and
metadata under `data/androidcontrol_mini/test.json`.  It expects the official
AndroidControl TFRecord fields `episode_id`, `goal`, `screenshots`,
`screenshot_widths`, `screenshot_heights`, `actions`, and `step_instructions`.
Remote first run should verify these field names against the local shard.

If TensorFlow is missing during preprocessing, install only the optional TFRecord
reader dependency:

```bash
pip install -r requirements-androidcontrol.txt
```

For already processed AndroidControl JSON/JSONL files, keep using
`scripts/prepare_androidcontrol_jsonl.py` and `test_gui_benchmark.py`.

## AndroidWorld Cache Benchmark

This repository also includes a small cache-oriented AndroidWorld adapter. It
uses AndroidWorld's official task registry, suite creation, and live emulator
environment, while routing model inference through this project's Qwen GUI
inference/cache path.

Default subset config:

```bash
configs/androidworld_cache_subset.json
```

Environment smoke test:

```bash
python scripts/run_androidworld_cache_benchmark.py \
  --android_world_path /path/to/android_world \
  --run_mode smoke \
  --output results/androidworld_cache/smoke \
  --perform_emulator_setup \
  --limit_episodes 1
```

Baseline and cache evaluation:

```bash
CUDA_VISIBLE_DEVICES=4,5 python scripts/run_androidworld_cache_benchmark.py \
  --android_world_path /path/to/android_world \
  --run_mode baseline \
  --output results/androidworld_cache/baseline_smoke \
  --model_path /data2/home/models/Qwen3.8-27B \
  --n_task_combinations 1 \
  --limit_episodes 3 \
  --page_cache_mode off

CUDA_VISIBLE_DEVICES=4,5 python scripts/run_androidworld_cache_benchmark.py \
  --android_world_path /path/to/android_world \
  --run_mode warmup \
  --output results/androidworld_cache/warmup \
  --model_path /data2/home/models/Qwen3.8-27B \
  --page_cache_mode observe \
  --page_cache_scope dataset \
  --page_cache_similarity tile

CUDA_VISIBLE_DEVICES=4,5 python scripts/run_androidworld_cache_benchmark.py \
  --android_world_path /path/to/android_world \
  --run_mode evaluation \
  --output results/androidworld_cache/eval_cache \
  --model_path /data2/home/models/Qwen3.8-27B \
  --page_cache_mode observe \
  --page_cache_scope dataset \
  --page_cache_similarity tile \
  --cache_input results/androidworld_cache/warmup/page_cache_records.jsonl

python scripts/summarize_androidworld_cache.py \
  --baseline results/androidworld_cache/baseline_smoke \
  --cache results/androidworld_cache/eval_cache \
  --output results/androidworld_cache/comparison_summary.json
```

See `docs/2026-09-01_androidworld_cache_benchmark.md` for emulator setup notes,
output fields, and current validation limits.

## Profiling

Use the profiling scripts before changing acceleration code.  They reuse the
same model loading, prompt construction, image preprocessing, generation, and
action parsing path as the baseline.

Single screenshot profiling:

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/profile_single_image.py \
  --model_path /data2/home/models/Qwen3.8-27B \
  --image data/androidcontrol_mini/images/episode_0/step_0.png \
  --instruction "目标任务：打开设置\n当前步骤：点击设置图标" \
  --output results/profile_single_image.json \
  --warmup 1 \
  --repeats 3 \
  --max_new_tokens 128
```

AndroidControl mini profiling:

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/profile_androidcontrol.py \
  --model_path /data2/home/models/Qwen3.8-27B \
  --test_json data/androidcontrol_mini/test.json \
  --output results/profile_androidcontrol_mini.json \
  --limit 5 \
  --warmup 1 \
  --max_new_tokens 128
```

The profile JSON reports stage timings for `build_prompt`,
`apply_chat_template`, `vision_preprocess`, `processor_encode`,
`input_to_device`, `generate`, `decode`, and `postprocess`, plus token counts
and best-effort CUDA peak memory snapshots.

## Acceleration Experiments

Run the full experiment matrix from the repository root on Jupiter:

```bash
python scripts/run_accel_experiments.py \
  --gpus 0,1 \
  --resume
```

`--gpus` sets `CUDA_VISIBLE_DEVICES` for child commands.  If omitted, the
launcher uses the current environment, so this is equivalent:

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_accel_experiments.py --resume
```

Useful subsets:

```bash
python scripts/run_accel_experiments.py --gpus 0,1 --experiments E00-E03 --resume
python scripts/run_accel_experiments.py --gpus 0,1 --experiments E04,E05,E06 --resume
```

Each experiment writes to `results/accel/<experiment_id>/` with `eval.json`,
`profile.json`, `run_metadata.json`, `stdout.log`, and `stderr.log`.  Existing
successful experiments are skipped with `--resume`; incomplete or failed
directories are not overwritten and get a `_rerunN` suffix.

Visual token modes supported by the eval/profile scripts:

- `default`: original processor behavior.
- `mild_reduce`: lower image token budget.
- `aggressive_reduce`: strongest fixed image token reduction tested so far.
- `dynamic_safe`: keeps CLICK/LONG_PRESS at default, uses mild reduction for
  SCROLL/TYPE, and aggressive reduction for transition/simple actions.
- `dynamic_aggressive`: uses mild reduction for CLICK/LONG_PRESS and aggressive
  reduction for all other inferred actions.

Batch eval/profile forces tokenizer left padding and passes explicit generation
pad/eos token ids, which avoids decoder-only batched generation reading from
right padding.  After changing batch behavior, rerun the new follow-up entries
instead of comparing against the old E12/E13 directories:

```bash
python scripts/run_accel_experiments.py --gpus 1,6 --experiments E15-E18 --resume
```

## Dependency Notes

Install only missing packages in the server environment:

```bash
pip install -r requirements.txt
```

If `AutoProcessor.from_pretrained` reports that `Qwen3VLVideoProcessor`
requires Torchvision, install a `torchvision` build that matches the existing
PyTorch/CUDA build in the active environment.

Do not commit downloaded models, datasets, cache directories, or benchmark outputs.
