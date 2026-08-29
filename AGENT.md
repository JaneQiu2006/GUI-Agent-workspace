# Project instructions

## Project
This repository is developed locally but compiled and tested on a remote
Linux server with Ascend/CANN hardware.

## Editing rules
- Keep changes minimal and scoped to the requested task.
- Do not refactor unrelated code.
- Follow the existing code style and naming conventions.
- Do not modify generated files or build artifacts.
- Do not commit binaries, logs, core dumps, or build directories.

## Remote-only environment
The local machine does not have the Ascend/CANN runtime or NPU hardware.

Therefore:
- Do not assume CANN/NPU tests can run locally.
- Perform static checks locally when possible.
- Do not change code merely because remote-only dependencies are unavailable.
- After editing, tell me exactly which commands should be run on the remote server.

## Git
- Show the changed files before finishing.
- Summarize the purpose of each change.
- Do not push automatically unless explicitly requested.

## Testing
Remote validation is performed on Jupiter.
Typical workflow:

git pull
<build command>
<test command>

## Current GUI Agent Baseline Context

As of 2026-08-29 21:50 CST, the repository has a calibrated static GUI Agent
baseline for AndroidControl. Local Windows has no model, no raw AndroidControl
shard, and no GPU, so local validation is limited to syntax/import/help checks.
Remote validation happens on Jupiter.

Core code:
- `test_framework/hf_gui_baseline.py`: HuggingFace Qwen3.8 static inference
  utilities. Important functions are `load_model_and_processor`,
  `preprocess_inputs`, `generate_response`, `infer_one`, `profile_infer_one`,
  `reset_gpu_memory_stats`, and `gpu_memory_snapshot`.
- `scripts/prepare_androidcontrol.py`: reads the raw AndroidControl GZIP
  TFRecord shard and writes `data/androidcontrol_mini/images/` plus
  `data/androidcontrol_mini/test.json`.
- `scripts/androidcontrol_actions.py`: converts AndroidControl GT actions to
  unified legacy action strings, normalizes model output, and computes action
  type/step matching.
- `scripts/eval_androidcontrol.py`: evaluates Qwen3.8 on the mini static
  AndroidControl set and writes JSON metrics/details.
- `scripts/profile_single_image.py` and `scripts/profile_androidcontrol.py`:
  profiling entry points that reuse the same baseline inference path.
- `test_framework/phone_prompt.py`: shared GUI prompt/action contract from the
  original test framework.

Remote commands used for the calibrated baseline:

```bash
python scripts/prepare_androidcontrol.py \
  --input data/raw/android_control/android_control-00000-of-00020 \
  --output_dir data/androidcontrol_mini \
  --num_episodes 10

CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_androidcontrol.py \
  --model_path /data2/home/models/Qwen3.8-27B \
  --test_json data/androidcontrol_mini/test.json \
  --output results/qwen_androidcontrol_mini.json

CUDA_VISIBLE_DEVICES=0,1 python scripts/profile_single_image.py \
  --model_path /data2/home/models/Qwen3.8-27B \
  --image data/androidcontrol_mini/images/episode_0/step_0.png \
  --instruction "目标任务：打开设置\n当前步骤：点击设置图标" \
  --output results/profile_single_image.json \
  --warmup 1 \
  --repeats 3 \
  --max_new_tokens 128

CUDA_VISIBLE_DEVICES=0,1 python scripts/profile_androidcontrol.py \
  --model_path /data2/home/models/Qwen3.8-27B \
  --test_json data/androidcontrol_mini/test.json \
  --output results/profile_androidcontrol_mini.json \
  --limit 5 \
  --warmup 1 \
  --max_new_tokens 128
```

Current calibrated results from `results/qwen_androidcontrol_mini.json`
generated at 2026-08-29 21:45 CST:
- 10 trajectories, 51 steps.
- strict type accuracy: 80.39%.
- strict step success rate: 74.51%.
- strict trajectory success rate: 40.00%.
- primary metric view: `gui_only`.
- `gui_only` type accuracy: 97.30%.
- `gui_only` step success rate: 91.89%.
- `gui_only` trajectory success rate: 80.00%.
- `transition_or_noop` (`OPEN_APP` + `WAIT`) step success rate: 28.57%.
- average latency: 4.62 seconds/step.
- average output tokens: 19.94.
- peak GPU memory: GPU0 25.14 GB, GPU1 28.39 GB.

Current profiling results from `results/profile_androidcontrol_mini.json`
generated at 2026-08-29 21:45 CST:
- AndroidControl mini profile over 5 steps: total mean 3.41s, generate mean
  3.25s, generate share 95.18%, average output tokens 13.2.
- Single image profile over 3 repeats: total mean 9.17s, generate mean 8.97s,
  generate share 97.89%, average output tokens 60.0. This file was not rerun
  after the latest calibration.
- Vision preprocessing is about 0.10s and is not the main bottleneck.

Resolved calibration issues:
- Thinking/long output is resolved for AndroidControl mini: 0/51 outputs contain
  `</think>`, 0/51 hit `max_new_tokens=128`, and average output tokens are about
  20.
- Malformed JSON no longer creates UNKNOWN predictions in the latest eval:
  `pred_type == UNKNOWN` is 0/51.
- `SCROLL[UP/DOWN]` direction is calibrated to AndroidControl content/page
  direction instead of finger movement direction; SCROLL step success is 7/7.
- `OPEN_APP` and `WAIT` are separated from ordinary GUI actions via metric
  views: `strict`, `gui_only`, `transition_or_noop`, `open_app`, `wait`, and
  `by_gt_type`.

Known issues before acceleration work:
- `WAIT` is noisy for static single-frame evaluation and should continue to be
  reported separately.
- `OPEN_APP` is an environment-level action and should continue to be reported
  separately from ordinary GUI actions.
- `LONG_PRESS` has only one sample and currently fails; decide whether to add it
  to the action contract or exclude it from the primary GUI-only view.
- A small number of CLICK failures remain coordinate-localization errors.
- GPU memory collection is best-effort; CUDA reset/read failures are warnings,
  not evaluation blockers.

Current analysis documents:
- `docs/2026-08-29_androidcontrol_baseline_analysis.md`
- `docs/2026-08-29_androidcontrol_calibration_rerun_analysis.md`
- `docs/2026-08-29_inference_acceleration_experiment_plan.md`

Recommended next work:
- Start inference acceleration experiments using
  `docs/2026-08-29_inference_acceleration_experiment_plan.md`.
- First run `E00-E03` to choose the lowest safe `max_new_tokens` cap.
- Then compare attention and dtype options (`sdpa`, `flash_attention_2`,
  `bfloat16`, `float16`) using the selected decode cap.
- Only after accuracy is stable, evaluate visual-token reduction, static
  batching, and serving backends.
- Every experiment must fully record `strict`, `gui_only`,
  `transition_or_noop`, `open_app`, `wait`, and `by_gt_type` metrics, plus
  output-health, latency/profile, and memory/resource metrics.
