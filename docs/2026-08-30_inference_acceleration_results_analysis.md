# 2026-08-30 Inference Acceleration Experiment Results Analysis

分析时间：2026-08-30 CST

本分析基于 `results/accel/` 下新的推理加速实验结果。由于实验启动过程中曾多次执行 launcher，顶层 `results/accel/launcher.log` 可能存在覆盖或混写；本报告以各实验目录内的 `run_metadata.json`、`eval.json` 和 `profile.json` 为准。

## Inputs And Environment

- Model: `/data2/home/models/Qwen3.8-27B`
- Test set: `data/androidcontrol_mini/test.json`
- Full eval: 51 steps
- Profile eval: `--limit 5 --warmup 1`
- Point tolerance: `100.0`
- Main GPU setting in successful metadata: `CUDA_VISIBLE_DEVICES=1,6`
- Python: `/data1/home/wuzheng/.conda/envs/qg/bin/python`
- Package versions recorded by launcher:
  - Python 3.11.15
  - torch 2.13.0
  - transformers 5.16.1
  - qwen-vl-utils 0.0.14
  - vLLM / SGLang not installed

## Run Status

| Experiment | Status | Notes |
| --- | --- | --- |
| E00 | interrupted | Initial baseline attempt stopped after `py_compile`. |
| E00_rerun1 | interrupted | Duplicate launcher run stopped before full result. |
| E00_rerun2 | interrupted | Duplicate launcher run stopped before full result. |
| E00_rerun3 | success | Use this as the reproduced acceleration baseline. |
| E01-E04 | success | Decode cap and SDPA comparison completed. |
| E05 | failed | FlashAttention 2 package is not installed. |
| E06-E13 | success | Dtype, visual token, and static batch experiments completed. |
| E14 | skipped | No `--serving_command` was provided. |

The interrupted E00 directories should not be used for metric comparison. They have no complete `eval.json` / `profile.json`.

## Executive Summary

The best completed acceleration candidate is **E11**:

- Config: `max_new_tokens=48`, `dtype=auto`, default attention, `batch_size=1`, `visual_token_mode=aggressive_reduce`
- `gui_only` type accuracy: 97.30%
- `gui_only` step success rate: 91.89%
- `strict` step success rate: 76.47%
- `pred_unknown`: 0 / 51
- `hit_max_new_tokens`: 0 / 51
- Full eval average latency: 2.646s, versus E00_rerun3 3.877s
- Full eval throughput: 0.365 samples/sec, versus E00_rerun3 0.248 samples/sec
- Profile total mean: 2.067s, versus E00_rerun3 3.372s
- Peak memory: GPU0 24.31 GB / GPU1 27.68 GB, lower than E00_rerun3 25.14 GB / 28.39 GB

Interpretation: E11 keeps the same primary GUI-only accuracy as the reproduced baseline while cutting average full-eval latency by about 31.8% and profile total time by about 38.7%. This is the only clearly viable speedup in the current matrix.

E03, E10, E12, and E13 are not viable:

- E03 (`max_new_tokens=32`) truncates outputs: 8 / 51 hit the token cap, 8 predictions become `UNKNOWN`, and SCROLL falls to 0 / 7.
- E10 (`mild_reduce`) is faster but CLICK step success drops from 91.67% to 83.33%.
- E12/E13 batch runs are faster per sample but produce many empty or malformed outputs. E12 has 16 `UNKNOWN`; E13 has 27 `UNKNOWN`, 2 think tags, and 4 token-cap hits.

E08 is the best non-visual-token configuration by profile total mean, but its gain over E00_rerun3 is only about 1.4%. The practical acceleration comes from visual token reduction, not dtype or SDPA.

## Comparison Table

Baseline for speedup is `E00_rerun3`.

| ID | Config | Status | gui type | gui step | strict step | unknown / cap hit | avg input tok | eval latency | eval samples/s | profile total | Profile speedup | Peak memory GB | Decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| E00_rerun3 | tok=128, dtype=auto, attn=default, bs=1, vis=default | success | 97.30% | 91.89% | 74.51% | 0 / 0 | 3404 | 3.877s | 0.248 | 3.372s | 1.00x | 25.14 / 28.39 | baseline |
| E01 | tok=64, dtype=auto, attn=default, bs=1, vis=default | success | 97.30% | 91.89% | 74.51% | 0 / 0 | 3404 | 3.898s | 0.247 | 3.339s | 1.01x | 25.14 / 28.39 | keep as safe cap check |
| E02 | tok=48, dtype=auto, attn=default, bs=1, vis=default | success | 97.30% | 91.89% | 74.51% | 0 / 0 | 3404 | 3.959s | 0.243 | 3.352s | 1.01x | 25.14 / 28.39 | keep as default decode cap |
| E03 | tok=32, dtype=auto, attn=default, bs=1, vis=default | success | 75.68% | 70.27% | 58.82% | 8 / 8 | 3404 | 3.738s | 0.257 | 3.352s | 1.01x | 25.14 / 28.39 | reject |
| E04 | tok=48, dtype=auto, attn=sdpa, bs=1, vis=default | success | 97.30% | 91.89% | 74.51% | 0 / 0 | 3404 | 3.890s | 0.247 | 3.358s | 1.00x | 25.14 / 28.39 | neutral |
| E05 | tok=48, dtype=auto, attn=flash_attention_2, bs=1, vis=default | failed | - | - | - | - | - | - | - | - | - | - | dependency missing |
| E06 | tok=48, dtype=bfloat16, attn=default, bs=1, vis=default | success | 97.30% | 91.89% | 74.51% | 0 / 0 | 3404 | 3.886s | 0.248 | 3.329s | 1.01x | 25.14 / 28.39 | neutral |
| E07 | tok=48, dtype=float16, attn=default, bs=1, vis=default | success | 97.30% | 91.89% | 74.51% | 0 / 0 | 3404 | 4.060s | 0.237 | 3.536s | 0.95x | 25.14 / 28.39 | reject |
| E08 | tok=48, dtype=bfloat16, attn=sdpa, bs=1, vis=default | success | 97.30% | 91.89% | 74.51% | 0 / 0 | 3404 | 3.922s | 0.246 | 3.325s | 1.01x | 25.14 / 28.39 | best non-visual |
| E09 | tok=48, dtype=float16, attn=sdpa, bs=1, vis=default | success | 97.30% | 91.89% | 74.51% | 0 / 0 | 3404 | 4.021s | 0.239 | 3.552s | 0.95x | 25.14 / 28.39 | reject |
| E10 | tok=48, dtype=auto, attn=default, bs=1, vis=mild_reduce | success | 97.30% | 86.49% | 72.55% | 0 / 0 | 1430 | 2.840s | 0.340 | 2.214s | 1.52x | 24.38 / 27.74 | reject |
| E11 | tok=48, dtype=auto, attn=default, bs=1, vis=aggressive_reduce | success | 97.30% | 91.89% | 76.47% | 0 / 0 | 1231 | 2.646s | 0.365 | 2.067s | 1.63x | 24.31 / 27.68 | keep |
| E12 | tok=48, dtype=auto, attn=default, bs=2, vis=default | success | 67.57% | 62.16% | 52.94% | 16 / 0 | 3404 | 2.767s | 0.343 | 2.835s | 1.19x | 26.54 / 29.63 | reject |
| E13 | tok=48, dtype=auto, attn=default, bs=4, vis=default | success | 40.54% | 37.84% | 35.29% | 27 / 4 | 3404 | 2.348s | 0.402 | 2.582s | 1.31x | 29.29 / 32.08 | reject |
| E14 | serving backend | skipped | - | - | - | - | - | - | - | - | - | - | not run |

## Decode Cap Findings

E00-E03 answer the first question in the plan: the lowest safe `max_new_tokens` cap is **48**, not 32.

| ID | max_new_tokens | gui step | pred_unknown | cap hits | output max | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| E00_rerun3 | 128 | 91.89% | 0 | 0 | 42 | safe |
| E01 | 64 | 91.89% | 0 | 0 | 42 | safe |
| E02 | 48 | 91.89% | 0 | 0 | 42 | safe |
| E03 | 32 | 70.27% | 8 | 8 | 32 | unsafe |

E02 should remain the default decode cap for later experiments. E03 is invalid because truncation creates parser failures and action-type regressions. The clearest symptom is SCROLL: E02 keeps SCROLL at 7 / 7, while E03 drops SCROLL to 0 / 7.

## Attention And Dtype Findings

Among E04-E09, all successful non-FP16 variants preserve the same accuracy as E02:

- E04 SDPA: profile total 3.358s
- E06 BF16: profile total 3.329s
- E08 BF16 + SDPA: profile total 3.325s

The differences between E02, E04, E06, and E08 are small enough that they should be treated as near-noise without repeated runs. E08 is the fastest of this group, but only by about 0.8% relative to E02 and about 1.4% relative to E00_rerun3 profile total. It is not a meaningful standalone acceleration win.

FP16 is worse in this environment:

- E07 profile total: 3.536s
- E09 profile total: 3.552s
- Accuracy remains stable, but latency regresses.

E05 failed before eval because FlashAttention 2 is not installed:

```text
ImportError: FlashAttention2 has been toggled on, but it cannot be used ... the package for FlashAttention2 doesn't seem to be installed.
```

This is an environment/dependency result, not a model quality result.

## Visual Token Findings

Visual token reduction is the only completed direction that materially improves latency.

| ID | visual_token_mode | avg input tokens | gui step | CLICK step | eval latency | profile total | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| E02 | default | 3404 | 91.89% | 91.67% | 3.959s | 3.352s | reference |
| E10 | mild_reduce | 1430 | 86.49% | 83.33% | 2.840s | 2.214s | reject |
| E11 | aggressive_reduce | 1231 | 91.89% | 91.67% | 2.646s | 2.067s | keep |

E10 is counterintuitive: it uses more input tokens than E11 but has worse CLICK localization. This should not be overinterpreted from one mini set, but it is enough to reject E10 under the current acceptance criteria.

E11 is viable on this mini set:

- It preserves `gui_only.step_success_rate`.
- CLICK step success remains 22 / 24, same as E02 and E00_rerun3.
- SCROLL, TYPE, and PRESS_BACK stay at 100%.
- `pred_unknown` remains 0.
- No output reaches the 48-token cap.
- Peak GPU memory decreases by about 0.83 GB on GPU0 and 0.71 GB on GPU1 versus E00_rerun3.

The strict score improves from 74.51% to 76.47% because OPEN_APP also improves from 50.00% to 66.67%, but this should be treated as secondary because OPEN_APP remains an environment-level action.

## Batch Findings

The current static batch implementation is not viable.

| ID | batch size | gui type | gui step | pred_unknown | think tags | cap hits | profile total | peak memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E12 | 2 | 67.57% | 62.16% | 16 | 1 | 0 | 2.835s | 26.54 / 29.63 GB |
| E13 | 4 | 40.54% | 37.84% | 27 | 2 | 4 | 2.582s | 29.29 / 32.08 GB |

Although E12/E13 improve per-sample timing, the output health collapses. Several `UNKNOWN` examples have empty `raw_response` while still reporting nonzero output token counts. E13 also produces a truncated output beginning with `</think>` and an incomplete JSON action. That pattern suggests the current batched generation/decode path is unstable for this model/processor combination, not merely less accurate.

Do not use E12/E13 as acceleration candidates. Before retrying batch experiments, inspect and fix the batched trimming/decoding path around padded `inputs.input_ids` and generated sequences. For decoder-only models with left/right padding, `len(in_ids)` may not correspond to the actual unpadded prompt length per sample; trimming by full padded length can produce empty decoded responses or misaligned completions.

## By Action Type

Selected per-type step success:

| ID | CLICK | LONG_PRESS | OPEN_APP | PRESS_BACK | SCROLL | TYPE | WAIT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E00_rerun3 | 91.67% | 0.00% | 50.00% | 100.00% | 100.00% | 100.00% | 12.50% |
| E02 | 91.67% | 0.00% | 50.00% | 100.00% | 100.00% | 100.00% | 12.50% |
| E03 | 87.50% | 0.00% | 50.00% | 100.00% | 0.00% | 100.00% | 12.50% |
| E10 | 83.33% | 0.00% | 66.67% | 100.00% | 100.00% | 100.00% | 12.50% |
| E11 | 91.67% | 0.00% | 66.67% | 100.00% | 100.00% | 100.00% | 12.50% |
| E12 | 66.67% | 0.00% | 16.67% | 100.00% | 57.14% | 33.33% | 37.50% |
| E13 | 41.67% | 0.00% | 16.67% | 50.00% | 28.57% | 33.33% | 37.50% |

The stable success pattern remains the same as the calibrated baseline:

- CLICK is the main quality-sensitive GUI action.
- SCROLL is stable except under unsafe truncation or batch instability.
- LONG_PRESS remains unsupported or under-specified and continues to fail.
- WAIT remains noisy in static single-frame evaluation and should stay outside the primary GUI-only interpretation.

## Launcher / Experiment Design Notes

One issue affected experiment sequencing: `E10-E13` did not inherit E08's BF16+SDPA configuration even though their matrix entry says `best-so-far`. They ran with `dtype=auto` and default attention.

Reason: the launcher acceptance threshold uses `REF_GUI_ONLY_TYPE = 0.9730`, while the exact measured value is `36 / 37 = 0.9729729729`, displayed as 97.30%. That strict floating comparison causes otherwise baseline-equivalent runs to miss the viability gate. The fallback best-so-far config is therefore `max_new_tokens=48`, `dtype=auto`, default attention.

This does not invalidate E11's result, because E11 is already a viable and faster candidate under its actual config. However, it means the planned combined visual-token experiment with BF16+SDPA has not actually been tested yet.

## Conclusions

1. Use `max_new_tokens=48` as the safe decode cap. `max_new_tokens=32` is unsafe.
2. Keep E11 as the current best acceleration candidate: `visual_token_mode=aggressive_reduce`, `batch_size=1`, `dtype=auto`, default attention.
3. Treat E08 as the best non-visual-token candidate, but the gain is too small to prioritize.
4. Reject E10 because it regresses CLICK localization.
5. Reject E12/E13 until batch decoding is fixed.
6. E05 can be rerun only after installing FlashAttention 2 compatible with the server's torch/CUDA stack.
7. E14 was not evaluated because no serving backend command was supplied.

## Recommended Next Runs

First, rerun the strongest visual-token candidate at least once to check variance:

```bash
CUDA_VISIBLE_DEVICES=1,6 python scripts/run_accel_experiments.py \
  --experiments E11 \
  --output_root results/accel \
  --resume
```

Because `--resume` will skip the existing successful E11, use a dedicated output root or temporarily select a new experiment id if a true repeat is needed. A direct rerun command is clearer:

```bash
OUT_DIR=results/accel/E11_rerun_manual1
mkdir -p "${OUT_DIR}"
CUDA_VISIBLE_DEVICES=1,6 python scripts/eval_androidcontrol.py \
  --model_path /data2/home/models/Qwen3.8-27B \
  --test_json data/androidcontrol_mini/test.json \
  --output "${OUT_DIR}/eval.json" \
  --max_new_tokens 48 \
  --visual_token_mode aggressive_reduce
CUDA_VISIBLE_DEVICES=1,6 python scripts/profile_androidcontrol.py \
  --model_path /data2/home/models/Qwen3.8-27B \
  --test_json data/androidcontrol_mini/test.json \
  --output "${OUT_DIR}/profile.json" \
  --limit 5 \
  --warmup 1 \
  --max_new_tokens 48 \
  --visual_token_mode aggressive_reduce
```

Then test the planned combined candidate that was missed due to the threshold issue:

```bash
OUT_DIR=results/accel/E11_bf16_sdpa_manual1
mkdir -p "${OUT_DIR}"
CUDA_VISIBLE_DEVICES=1,6 python scripts/eval_androidcontrol.py \
  --model_path /data2/home/models/Qwen3.8-27B \
  --test_json data/androidcontrol_mini/test.json \
  --output "${OUT_DIR}/eval.json" \
  --max_new_tokens 48 \
  --dtype bfloat16 \
  --attn_implementation sdpa \
  --visual_token_mode aggressive_reduce
CUDA_VISIBLE_DEVICES=1,6 python scripts/profile_androidcontrol.py \
  --model_path /data2/home/models/Qwen3.8-27B \
  --test_json data/androidcontrol_mini/test.json \
  --output "${OUT_DIR}/profile.json" \
  --limit 5 \
  --warmup 1 \
  --max_new_tokens 48 \
  --dtype bfloat16 \
  --attn_implementation sdpa \
  --visual_token_mode aggressive_reduce
```

Before rerunning E12/E13, fix the batch decode path and add a small batch smoke test that verifies `raw_response` is nonempty for every sample.
