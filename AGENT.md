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

As of 2026-08-29, the repository has a minimal static GUI Agent baseline for
AndroidControl. Local Windows has no model, no raw AndroidControl shard, and no
GPU, so local validation is limited to syntax/import/help checks. Remote
validation happens on Jupiter.

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

Remote commands used for the current baseline:

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

Current baseline results from `results/qwen_androidcontrol_mini.json`:
- 10 trajectories, 51 steps.
- type accuracy: 64.71%.
- step success rate: 41.18%.
- trajectory success rate: 0.00%.
- average latency: 14.62 seconds/step.
- average output tokens: 101.20.
- peak GPU memory: GPU0 25.08 GB, GPU1 28.33 GB.
- Filtering out `OPEN_APP` and `WAIT`, ordinary GUI steps have type accuracy
  83.78% and step success rate 51.35%.

Current profiling results:
- AndroidControl mini profile over 5 steps: total mean 10.54s, generate mean
  10.38s, generate share 98.50%, average output tokens 79.0.
- Single image profile over 3 repeats: total mean 9.17s, generate mean 8.97s,
  generate share 97.89%, average output tokens 60.0.
- Vision preprocessing is about 0.10-0.13s and is not the main bottleneck.

Known issues before acceleration work:
- Model outputs are too long. 44/51 eval outputs contain `</think>`, and 12/51
  hit `max_new_tokens=128`, causing invalid/UNKNOWN parsed actions and inflated
  latency.
- `OPEN_APP` from AndroidControl is not directly comparable to the current GUI
  action space, which mostly contains tap/swipe/type/back/home/wait.
- `SCROLL[UP/DOWN]` direction likely needs calibration: AndroidControl appears
  to use page/content direction, while current prediction canonicalization
  infers finger movement direction.
- `WAIT` is noisy for static single-frame evaluation and should be reported
  separately.
- GPU memory collection is best-effort; CUDA reset/read failures are warnings,
  not evaluation blockers.

Current analysis document:
- `docs/2026-08-29_androidcontrol_baseline_analysis.md`

Recommended next work:
- First fix output format and disable/avoid thinking so action outputs are
  short and parseable; then re-run eval/profile.
- Add strict and gui-only metric views, with `OPEN_APP` and `WAIT` separated.
- Calibrate `SCROLL[UP/DOWN]` against AndroidControl semantics.
- Only after metric/prompt calibration, compare acceleration methods such as
  lower `max_new_tokens`, FlashAttention/SDPA config, vLLM/SGLang serving,
  BF16/FP16/quantization, and static batching.
