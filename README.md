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

For Feature Cache studies, the profiling outputs also include
`stage_profile` and `profile_metadata` on each run/step.  The structured
summary is under `summary.fine_grained_profile` and reports per-step,
per-episode, and overall mean/median/P90 for:

- image load/decode/preprocess
- image resize/normalize/patch or token construction
- vision encoder / visual feature extraction when a known visual module hook is available
- visual projector / adapter when a known projector hook is available
- text prompt preprocessing
- multimodal prefill when manual greedy profiling is enabled
- decode/generation
- total inference latency

The same summary records visual-related latency and ratio, input image size,
visual patch/token counts when available, prompt token count, generated token
count, and cache boundary candidates such as processor outputs, vision encoder
outputs, projected visual embeddings, and exact multimodal prefill KV.

By default `--generation_profile_mode generate` keeps using
`model.generate()` and treats generation as an inclusive opaque stage.  Use the
profiling-only manual greedy path when prefill, TTFT, and per-token decode
timings are needed:

```bash
CUDA_VISIBLE_DEVICES=4,5 python scripts/profile_androidcontrol.py \
  --model_path /data2/home/models/Qwen3.8-27B \
  --test_json data/androidcontrol_mini/test.json \
  --output results/feature_cache_profile/profile_manual_greedy.json \
  --limit 5 \
  --warmup 1 \
  --max_new_tokens 48 \
  --visual_token_mode aggressive_reduce \
  --generation_profile_mode manual_greedy
```

`manual_greedy` currently supports `batch_size=1` only and is for profiling
experiments, not the default eval path.  Its cached decode step trims
`input_ids`, `cache_position`, token type ids, and multimodal token type ids to
the current token, explicitly builds current-token `position_ids` from the full
attention mask plus Qwen rope deltas when available, and clears image pixel
tensors so Qwen3.5 VL-style models do not reuse prefix-length position tensors
during single-token decode.

Visual feature locality analysis for Patch Feature Cache feasibility:

```bash
CUDA_VISIBLE_DEVICES=4,5 python scripts/analyze_visual_feature_locality.py \
  --model_path /data2/home/models/Qwen3.8-27B \
  --test_json data/androidcontrol_1000/test.json \
  --data_dir data/androidcontrol_1000 \
  --output_dir results/feature_locality_analysis \
  --run_name androidcontrol_1000_broad_100 \
  --max_pairs 100 \
  --visual_token_mode aggressive_reduce \
  --feature_boundary vision_final \
  --page_cache_scope dataset \
  --page_cache_similarity tile \
  --page_cache_near_dhash_threshold 8 \
  --page_cache_near_tile_threshold 0.95 \
  --page_cache_patch_tile_threshold 0.85 \
  --page_cache_patch_max_changed_area_ratio 0.35 \
  --similar_hit_type near \
  --similar_hit_type patch_candidate \
  --feature_metric cosine_distance \
  --feature_threshold 0.005 \
  --feature_threshold 0.01 \
  --feature_threshold 0.03 \
  --feature_threshold 0.05 \
  --feature_threshold 0.10
```

Run the same pair set at the model-ready visual embedding boundary after the
vision merger/projector:

```bash
CUDA_VISIBLE_DEVICES=4,5 python scripts/analyze_visual_feature_locality.py \
  --model_path /data2/home/models/Qwen3.8-27B \
  --test_json data/androidcontrol_1000/test.json \
  --data_dir data/androidcontrol_1000 \
  --output_dir results/feature_locality_analysis \
  --run_name androidcontrol_1000_broad_100_model_ready_visual \
  --max_pairs 100 \
  --visual_token_mode aggressive_reduce \
  --feature_boundary model_ready_visual \
  --page_cache_scope dataset \
  --page_cache_similarity tile \
  --page_cache_near_dhash_threshold 8 \
  --page_cache_near_tile_threshold 0.95 \
  --page_cache_patch_tile_threshold 0.85 \
  --page_cache_patch_max_changed_area_ratio 0.35 \
  --similar_hit_type near \
  --similar_hit_type patch_candidate \
  --feature_metric cosine_distance \
  --feature_threshold 0.005 \
  --feature_threshold 0.01 \
  --feature_threshold 0.03 \
  --feature_threshold 0.05 \
  --feature_threshold 0.10
```

The locality script writes `summary.json`, `per_pair.jsonl`,
`per_pair_summary.csv`, `threshold_stats.csv`, `grouped_stats.csv`,
`distance_profiles.csv`, and optional plots under the selected output
directory.  The current 100-pair results show strong patch-level locality at
`feature_boundary=vision_final` with `tau=0.01`.  At
`feature_boundary=model_ready_visual`, the script captures `model.visual.merger`
outputs, usually `377` tokens on a `[29,13]` grid; locality remains visible, but
the distance scale is larger and `tau=0.05` is the better main threshold
candidate.  See `docs/2026-09-06_visual_feature_locality_analysis.md` for the
full interpretation.

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
