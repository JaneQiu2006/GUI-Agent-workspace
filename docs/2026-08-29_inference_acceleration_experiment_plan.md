# 2026-08-29 GUI Agent Inference Acceleration Experiment Plan

This plan starts after prompt/action parsing and AndroidControl evaluator
calibration.  It is intended for remote execution on Jupiter, where the local
model, AndroidControl mini data, and GPU/NPU runtime are available.

## Fixed Inputs

- Model: `/data2/home/models/Qwen3.8-27B`
- Test set: `data/androidcontrol_mini/test.json`
- Full eval: all 51 steps
- Profile eval: `--limit 5 --warmup 1`
- Default point tolerance: `100.0`
- Primary accuracy view: `gui_only`
- Required saved output root: `results/accel/<experiment_id>/`

Each experiment must save:

- `eval.json`: full AndroidControl eval output
- `profile.json`: AndroidControl profile output
- `run_metadata.json`: command lines, git state, environment, timing, and notes
- `stdout.log` and `stderr.log`, if the launcher supports tee/log capture

Do not overwrite prior experiment outputs.  If rerunning an experiment, use a
new suffix such as `E02_rerun1`.

## Current Reference Results

Reference result file:

- `results/qwen_androidcontrol_mini.json`
- generated at `2026-08-29T13:45:13+00:00`

Reference full-eval metrics:

| View | Steps | Type Acc | Step Acc | Traj Acc | Avg Tokens | Avg Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| strict | 51 | 80.39% | 74.51% | 40.00% | 19.94 | 4.62s |
| gui_only | 37 | 97.30% | 91.89% | 80.00% | 21.95 | 4.80s |
| transition_or_noop | 14 | 35.71% | 28.57% | 16.67% | 14.64 | 4.15s |
| open_app | 6 | 66.67% | 50.00% | 50.00% | 12.67 | 4.12s |
| wait | 8 | 12.50% | 12.50% | 0.00% | 16.12 | 4.17s |

Reference profile result file:

- `results/profile_androidcontrol_mini.json`
- generated at `2026-08-29T13:45:54+00:00`

Reference profile metrics:

| Stage | Mean Time | Share |
| --- | ---: | ---: |
| vision_preprocess | 0.1051s | 3.08% |
| processor_encode | 0.0474s | 1.39% |
| input_to_device | 0.0115s | 0.34% |
| generate | 3.2459s | 95.18% |
| total | 3.4106s | 100.00% |

Reference health checks:

- `pred_unknown`: 0 / 51
- contains `</think>`: 0 / 51
- hits `max_new_tokens=128`: 0 / 51
- output token min / median / max: 6 / 18 / 42

## Required Metrics

Every experiment must record the complete existing JSON outputs from
`scripts/eval_androidcontrol.py` and `scripts/profile_androidcontrol.py`.
Summaries must include at least the following fields.

Accuracy:

- `metrics.num_steps`
- `metrics.num_trajectories`
- `metrics.primary_metric_view`
- `metrics.views.strict.type_accuracy`
- `metrics.views.strict.step_success_rate`
- `metrics.views.strict.trajectory_success_rate`
- `metrics.views.gui_only.type_accuracy`
- `metrics.views.gui_only.step_success_rate`
- `metrics.views.gui_only.trajectory_success_rate`
- `metrics.views.transition_or_noop.step_success_rate`
- `metrics.views.open_app.step_success_rate`
- `metrics.views.wait.step_success_rate`
- all `metrics.views.by_gt_type.*.type_accuracy`
- all `metrics.views.by_gt_type.*.step_success_rate`

Output health:

- average input tokens
- average output tokens
- min / median / max output tokens
- number of `pred_type == "UNKNOWN"`
- number of raw outputs containing `</think>`
- number of outputs that hit `max_new_tokens`
- number of outputs that start with `{` and end with `}`
- malformed parse repair count, if the script later exposes it

Performance:

- full eval average latency
- full eval min / median / max latency
- full eval wall-clock time
- profile stage mean / min / max / median for:
  - `build_prompt_seconds`
  - `apply_chat_template_seconds`
  - `vision_preprocess_seconds`
  - `processor_encode_seconds`
  - `input_to_device_seconds`
  - `generate_seconds`
  - `decode_seconds`
  - `postprocess_seconds`
  - `total_seconds`

Memory and resources:

- `peak_gpu_memory_gb` per device
- `peak_gpu_memory_bytes` per device
- CUDA/CANN/NPU/GPU device names, if available
- package versions for Python, torch, transformers, qwen-vl-utils, and vLLM/SGLang
  if used
- warning/error messages
- OOM/fallback/retry notes

Throughput, for batch or serving experiments:

- samples/sec
- output tokens/sec
- effective batch size
- request concurrency, if applicable
- failed sample count

## Acceptance Criteria

An experiment is only considered a viable acceleration candidate if:

- `gui_only.step_success_rate >= 91.89%`
- `gui_only.type_accuracy >= 97.30%`, or any decrease is explicitly justified
- `pred_unknown == 0`
- no output truncation creates parser failures
- `CLICK.step_success_rate` does not materially regress
- latency or throughput improves enough to justify the change
- peak memory remains within available device capacity

`strict` should still be reported, but do not reject an acceleration candidate
solely because `OPEN_APP` or `WAIT` remains noisy.

## Experiment Matrix

| ID | Direction | max_new_tokens | dtype | attn | batch | visual/token setting | Purpose |
| --- | --- | ---: | --- | --- | ---: | --- | --- |
| E00 | calibrated baseline | 128 | auto | default | 1 | default | Reproduce current calibrated result |
| E01 | decode cap | 64 | auto | default | 1 | default | Check conservative output cap |
| E02 | decode cap | 48 | auto | default | 1 | default | Likely default for later tests |
| E03 | decode cap | 32 | auto | default | 1 | default | Aggressive output cap |
| E04 | attention | 48 | auto | sdpa | 1 | default | Compare SDPA |
| E05 | attention | 48 | auto | flash_attention_2 | 1 | default | Compare FlashAttention 2 if installed |
| E06 | dtype | 48 | bfloat16 | default | 1 | default | Compare BF16 |
| E07 | dtype | 48 | float16 | default | 1 | default | Compare FP16 |
| E08 | dtype + attention | 48 | bfloat16 | sdpa | 1 | default | Stable combined candidate |
| E09 | dtype + attention | 48 | float16 | sdpa | 1 | default | FP16 + SDPA candidate |
| E10 | visual token | 48 | best-so-far | best-so-far | 1 | mild reduce | Reduce input cost with low accuracy risk |
| E11 | visual token | 48 | best-so-far | best-so-far | 1 | aggressive reduce | Stress-test visual token reduction |
| E12 | batch | 48 | best-so-far | best-so-far | 2 | default | Small batch throughput |
| E13 | batch | 48 | best-so-far | best-so-far | 4 | default | Batch memory/throughput pressure test |
| E14 | serving backend | 48 | best-so-far | backend default | service | default | Compare vLLM/SGLang if available |

For `best-so-far`, use the fastest prior configuration that passes the
acceptance criteria.

## Execution Order

1. Run `E00-E03`.
   - Pick the lowest `max_new_tokens` that keeps `pred_unknown=0`,
     no truncation-induced parser failures, and no `gui_only` regression.
   - If `E03` regresses, use `E02` as the default cap for later experiments.

2. Run `E04-E09`.
   - Compare attention and dtype options against the chosen decode cap.
   - If `flash_attention_2` is unavailable, record the import/config error and
     continue without treating it as a failed model result.

3. Run `E10-E11`.
   - Only after a stable dtype/attention candidate exists.
   - Inspect `CLICK.step_success_rate` and coordinate failures carefully.
   - Do not keep a visual-token reduction that trades away coordinate accuracy.

4. Run `E12-E14`.
   - Treat these as throughput/engineering experiments.
   - Record samples/sec, output tokens/sec, effective batch size, peak memory,
     and failed samples.

## Command Template

For every non-serving single-sample experiment:

```bash
EXP_ID=E00
OUT_DIR=results/accel/${EXP_ID}
mkdir -p "${OUT_DIR}"

python -m py_compile \
  test_framework/phone_prompt.py \
  test_framework/hf_gui_baseline.py \
  scripts/androidcontrol_actions.py \
  scripts/eval_androidcontrol.py \
  scripts/profile_androidcontrol.py

CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_androidcontrol.py \
  --model_path /data2/home/models/Qwen3.8-27B \
  --test_json data/androidcontrol_mini/test.json \
  --output "${OUT_DIR}/eval.json" \
  --max_new_tokens 128

CUDA_VISIBLE_DEVICES=0,1 python scripts/profile_androidcontrol.py \
  --model_path /data2/home/models/Qwen3.8-27B \
  --test_json data/androidcontrol_mini/test.json \
  --output "${OUT_DIR}/profile.json" \
  --limit 5 \
  --warmup 1 \
  --max_new_tokens 128
```

Adjust `--max_new_tokens`, `--dtype`, and `--attn_implementation` according to
the matrix row.  For default attention, omit `--attn_implementation`.

## Result Review Template

After each experiment, summarize:

```text
ID:
Command:
Status:
Primary view:
gui_only type / step / trajectory:
strict type / step / trajectory:
by_gt_type regressions:
pred_unknown:
think tags:
hit max_new_tokens:
avg output tokens:
full eval avg latency:
profile generate mean:
profile total mean:
peak GPU memory:
Decision: keep / reject / rerun
Reason:
```

Keep the final comparison table in a follow-up analysis document rather than
editing result JSON files.
