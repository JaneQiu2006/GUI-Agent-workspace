# 项目协作说明

## 项目环境

本仓库主要在本地编辑，在远端 Jupiter 服务器上运行模型、GPU 评测和 profiling。当前本地 Windows 环境没有 Qwen3.8 模型、AndroidControl 原始数据、CUDA GPU 或远端 Python 环境，因此本地验证以静态检查为主。

远端实验主要使用：

- 模型：`/data2/home/models/Qwen3.8-27B`
- Python：`/data1/home/wuzheng/.conda/envs/qg/bin/python`
- 测试集：`data/androidcontrol_mini/test.json`
- GPU 选择：优先通过 `CUDA_VISIBLE_DEVICES=...` 或 launcher 的 `--gpus ...` 指定

不要因为本地缺少模型、数据或 GPU 依赖就改代码规避错误。涉及真实评测时，应给出需要在 Jupiter 上执行的命令。

## 编辑规则

- 修改保持最小化，只处理用户当前请求相关内容。
- 不做无关重构。
- 遵循现有代码风格和命名习惯。
- 不修改生成文件、构建产物、日志、core dump 或大结果目录，除非用户明确要求。
- 不自动 push，除非用户明确要求。
- 完成前展示被修改的文件，并说明每项修改目的。

## 当前核心代码

- `test_framework/phone_prompt.py`：共享手机 GUI prompt/action contract。当前已强化为 thinking off 风格，只要求输出一个合法 JSON action。
- `test_framework/hf_gui_baseline.py`：HuggingFace / Transformers 静态 GUI 推理工具，包含模型加载、processor 输入构造、单样本推理、batch 推理、profiling、视觉 token 控制和显存统计。
- `scripts/prepare_androidcontrol.py`：从 AndroidControl GZIP TFRecord 生成 `data/androidcontrol_mini/images/` 和 `data/androidcontrol_mini/test.json`。
- `scripts/androidcontrol_actions.py`：将 AndroidControl GT action 转成统一 legacy action string，规范化模型输出，并计算 action type / step matching。
- `scripts/eval_androidcontrol.py`：完整静态评测入口，输出 `eval.json`。
- `scripts/profile_androidcontrol.py`：AndroidControl mini profiling 入口，输出 `profile.json`。
- `scripts/profile_single_image.py`：单图 profiling 入口。
- `scripts/run_accel_experiments.py`：一次性运行推理加速实验矩阵，支持 `--experiments`、`--resume`、`--gpus`、`--fail_fast` 和 `--dry_run`。

## 数据集与 prompt 口径

AndroidControl mini 数据集的 `task` 字段来自原始数据：

```text
目标任务：{goal}
当前步骤：{step_instruction}
```

加速实验没有改写 `data/androidcontrol_mini/test.json` 的任务字段。模型实际输入会在 `build_phone_prompt()` 中拼接共享 `PHONE_SYSTEM_PROMPT` 和 `当前任务：{task}`。因此：

- 数据集 task/prompt 字段没有因加速实验改变。
- 共享模型输入 prompt 模板在 2026-08-29 校准阶段强化过。
- 当前校准后 baseline 和 E00-E20 加速实验在同一套 prompt/action contract 下可比。

当前 prompt 相关事实：

- `apply_chat_template(..., enable_thinking=False)` 已启用；若 processor 不支持，会自动 fallback。
- 实测校准后 `contains_think_end = 0 / 51`，可认为当前实验已是 thinking off。
- prompt 要求只输出一个合法 JSON 对象，不输出思考过程、解释、Markdown 或 `action:` 前缀。
- 当前 action contract 支持 `tap`、`swipe`、`type`、`back`、`home`、`wait`、`complete`、`impossible`。
- `LONG_PRESS` 仍是待定口径：当前样本只有 1 个，且 action contract 尚未正式支持。

## 当前校准基线

校准后 AndroidControl mini 结果生成于 2026-08-29 21:45 CST：

- 10 条 trajectory，51 个 step。
- strict type accuracy：80.39%。
- strict step success rate：74.51%，即 38 / 51。
- strict trajectory success rate：40.00%。
- 主观察口径：`gui_only`。
- `gui_only` type accuracy：97.30%。
- `gui_only` step success rate：91.89%，即 34 / 37。
- `gui_only` trajectory success rate：80.00%。
- `transition_or_noop` (`OPEN_APP` + `WAIT`) step success rate：28.57%。
- 平均延迟：4.62s / step。
- 平均输出 token：19.94。
- 峰值显存：GPU0 25.14 GB，GPU1 28.39 GB。

`gui_only` 用于观察普通 GUI 操作能力，排除 `OPEN_APP` 和 `WAIT`。`strict` 保留用于和 AndroidControl 原始 GT 全量比较。

已解决的校准问题：

- thinking / 长输出问题已解决：0 / 51 输出包含 `</think>`，0 / 51 命中 `max_new_tokens=128`。
- malformed JSON 不再导致 UNKNOWN：`pred_type == UNKNOWN` 为 0 / 51。
- SCROLL 方向已按 AndroidControl 内容/页面方向校准，而不是手指物理运动方向；SCROLL step success 为 7 / 7。
- `OPEN_APP` 和 `WAIT` 已通过 `strict`、`gui_only`、`transition_or_noop`、`open_app`、`wait`、`by_gt_type` 分视图报告。

已知剩余问题：

- `WAIT` 在静态单帧评测中噪声大，应继续单独报告。
- `OPEN_APP` 是环境级动作，应继续与普通 GUI 操作分开报告。
- `LONG_PRESS` 样本少且当前失败，需要决定加入 action contract 还是移出主口径。
- 少量 CLICK 失败仍是坐标定位偏差。
- GPU 显存统计是 best effort；CUDA reset/read 失败应作为 warning，不应中断评测。

## 当前推理加速结论

当前加速分析以 `docs/2026-08-30_inference_acceleration_results_analysis.md` 为准。速度对比的复现 baseline 是 `results/accel/E00_rerun3`。

当前最好的简单单请求方案是 E11：

```text
max_new_tokens=48
visual_token_mode=aggressive_reduce
batch_size=1
dtype=auto
attn_implementation=None
```

E11 结果：

- `gui_only` step success rate：91.89%，与 baseline 相同。
- `strict` step success rate：76.47%。
- `pred_unknown`：0 / 51。
- `hit_max_new_tokens`：0 / 51。
- 平均输入 token：1231，对比 default 约 3404。
- 完整评测平均延迟：2.646s，对比 E00_rerun3 的 3.877s。
- 完整评测吞吐：0.365 samples/sec，对比 E00_rerun3 的 0.248 samples/sec。
- profile 平均总耗时：2.067s，对比 E00_rerun3 的 3.372s。
- 峰值显存：GPU0 24.31 GB / GPU1 27.68 GB。

视觉 token 模式：

```python
VISION_TOKEN_MODES = {
    "default": {},
    "mild_reduce": {"max_pixels": 768 * 28 * 28},
    "aggressive_reduce": {"max_pixels": 512 * 28 * 28},
    "dynamic_safe": {},
    "dynamic_aggressive": {},
}
```

当前有效图片输入压缩参数是：

```text
visual_token_mode=aggressive_reduce
max_pixels=512 * 28 * 28 = 401408
```

这不是离线改写图片文件，而是在 image message 中传入 `max_pixels`，由 Qwen-VL processor / `process_vision_info()` 控制进入模型的视觉 token 预算。实测总输入 token 从约 3404 降到约 1231，约 2.76 倍压缩；像素预算上限约为 401k 像素。

不建议采用的结果：

- E03：`max_new_tokens=32` 不安全，会截断输出并产生 UNKNOWN。
- E10：`mild_reduce` 虽快，但 CLICK step success 从 91.67% 降到 83.33%。
- 旧 E12/E13：发生在 batch decode 修复前，输出健康度崩溃，不作为可用 batch 结论。
- E16：`dynamic_aggressive` 会导致 CLICK 退化。
- FP16 相关 E07/E09 当前环境更慢。
- E05 需要安装兼容的 FlashAttention 2 后才能重跑。

补跑结论：

- `E11_rerun1` 复现 E11 准确率和速度。
- `E11_bf16_sdpa` 略快于 `E11_rerun1`，但差距低于 1%，接近噪声。
- `dynamic_safe` 保准确率，但单样本慢于固定 `aggressive_reduce`。
- batch decode 已修复：干净 E17/E18 均为 0 UNKNOWN、0 think 标签、0 cap hit。
- 干净 E18 (`batch_size=4`, `dynamic_safe`) 完整评测吞吐达到 0.418 samples/sec，但显存升至 29.28 / 32.08 GB，适合作为吞吐候选，不是单请求延迟最优解。

## profiling 字段说明

当前 `profile_androidcontrol.py` 记录的耗时字段包括：

- `build_prompt_seconds`
- `apply_chat_template_seconds`
- `vision_preprocess_seconds`
- `processor_encode_seconds`
- `input_to_device_seconds`
- `generate_seconds`
- `decode_seconds`
- `postprocess_seconds`
- `total_seconds`

输入侧 Python/processor 耗时主要是 `vision_preprocess_seconds`、`processor_encode_seconds`、`input_to_device_seconds`；输出侧可观察字段主要是 `generate_seconds`、`decode_seconds`、`postprocess_seconds`。

注意：当前 profiling 尚未把 `generate_seconds` 内部细分成 prefill、TTFT 和逐 token decode，因此无法直接判断模型内部输入端 prefill 与输出端 decode 的精确占比。已有结果显示 `generate_seconds` 仍占 95% 左右，是主要耗时。

## 远端常用命令

准备 AndroidControl mini：

```bash
python scripts/prepare_androidcontrol.py \
  --input data/raw/android_control/android_control-00000-of-00020 \
  --output_dir data/androidcontrol_mini \
  --num_episodes 10
```

运行当前推荐单请求方案：

```bash
CUDA_VISIBLE_DEVICES=4,5 python scripts/eval_androidcontrol.py \
  --model_path /data2/home/models/Qwen3.8-27B \
  --test_json data/androidcontrol_mini/test.json \
  --output results/accel_followup_clean/E11_recheck/eval.json \
  --max_new_tokens 48 \
  --visual_token_mode aggressive_reduce

CUDA_VISIBLE_DEVICES=4,5 python scripts/profile_androidcontrol.py \
  --model_path /data2/home/models/Qwen3.8-27B \
  --test_json data/androidcontrol_mini/test.json \
  --output results/accel_followup_clean/E11_recheck/profile.json \
  --limit 5 \
  --warmup 1 \
  --max_new_tokens 48 \
  --visual_token_mode aggressive_reduce
```

通过 launcher 指定 GPU 并断点续跑：

```bash
nohup env CUDA_VISIBLE_DEVICES=4,5 python scripts/run_accel_experiments.py \
  --resume \
  --output_root results/accel_followup_clean \
  > results/accel_followup_clean/launcher.log 2>&1 &
```

或使用 launcher 参数指定 GPU：

```bash
nohup python scripts/run_accel_experiments.py \
  --gpus 4,5 \
  --resume \
  --output_root results/accel_followup_clean \
  > results/accel_followup_clean/launcher.log 2>&1 &
```

注意不要把多个后台任务写到同一个 `launcher.log`，否则日志会覆盖或混写。新实验建议使用独立 output root 或独立 log 文件。

## 建议下一步实验

优先补齐 E19/E20，验证修复后的 batch decode 与固定 `aggressive_reduce` 是否能叠加收益：

```bash
OUT_DIR=results/accel_followup_clean/E19_batch2_aggressive
mkdir -p "${OUT_DIR}"
CUDA_VISIBLE_DEVICES=4,5 python scripts/eval_androidcontrol.py \
  --model_path /data2/home/models/Qwen3.8-27B \
  --test_json data/androidcontrol_mini/test.json \
  --output "${OUT_DIR}/eval.json" \
  --max_new_tokens 48 \
  --batch_size 2 \
  --visual_token_mode aggressive_reduce
CUDA_VISIBLE_DEVICES=4,5 python scripts/profile_androidcontrol.py \
  --model_path /data2/home/models/Qwen3.8-27B \
  --test_json data/androidcontrol_mini/test.json \
  --output "${OUT_DIR}/profile.json" \
  --limit 5 \
  --warmup 1 \
  --max_new_tokens 48 \
  --batch_size 2 \
  --visual_token_mode aggressive_reduce
```

```bash
OUT_DIR=results/accel_followup_clean/E20_batch4_aggressive
mkdir -p "${OUT_DIR}"
CUDA_VISIBLE_DEVICES=4,5 python scripts/eval_androidcontrol.py \
  --model_path /data2/home/models/Qwen3.8-27B \
  --test_json data/androidcontrol_mini/test.json \
  --output "${OUT_DIR}/eval.json" \
  --max_new_tokens 48 \
  --batch_size 4 \
  --visual_token_mode aggressive_reduce
CUDA_VISIBLE_DEVICES=4,5 python scripts/profile_androidcontrol.py \
  --model_path /data2/home/models/Qwen3.8-27B \
  --test_json data/androidcontrol_mini/test.json \
  --output "${OUT_DIR}/profile.json" \
  --limit 5 \
  --warmup 1 \
  --max_new_tokens 48 \
  --batch_size 4 \
  --visual_token_mode aggressive_reduce
```

后续可考虑的输出端压缩方向：

- 尝试 `max_new_tokens=44` / `40`，确认是否仍无截断。
- 增加 stop-on-valid-action，在第一个完整 JSON action 后停止生成。
- 评估更短 action 协议，例如 `TAP 431 70`、`SWIPE 500 700 500 300`，但需要重新验证 parser 与准确率。

## 分析文档

当前主要分析文档：

- `docs/2026-08-29_androidcontrol_baseline_analysis.md`
- `docs/2026-08-29_androidcontrol_calibration_rerun_analysis.md`
- `docs/2026-08-29_inference_acceleration_experiment_plan.md`
- `docs/2026-08-30_inference_acceleration_results_analysis.md`



## Execution Efficiency and Validation Policy

Optimize for low token usage and short execution time while preserving correctness.

* Perform only the analysis, code review, and testing necessary for the current task.
* Prefer targeted inspection over repository-wide exploration. Read only files directly related to the requested change unless additional context is required.
* Prefer targeted tests for modified components over full test suites.
* Do not run full regression, integration, benchmark, lint, format, build, or static-analysis suites unless:

  * the change directly affects a broad/shared component,
  * targeted validation is insufficient,
  * or the user explicitly requests them.
* Avoid repeating tests or inspections when the relevant result is already available and no related code has changed since.
* Do not perform speculative refactoring, cleanup, optimization, documentation changes, or unrelated issue investigation.
* Avoid exhaustive review of unchanged code. Review the modified code and its directly affected interfaces/dependencies.
* Reuse existing project scripts, test commands, documentation, and prior context instead of rediscovering repository structure unnecessarily.
* When multiple validation methods are available, choose the cheapest method that provides sufficient confidence.
* For small/local changes, a minimal validation sequence is preferred:

  1. inspect the relevant implementation and interfaces;
  2. make the requested change;
  3. run the smallest relevant test/check;
  4. inspect the resulting diff.
* Escalate to broader tests only when the targeted check fails, reveals uncertainty, or the change has significant cross-module impact.
* Do not spend tokens narrating routine repository exploration or obvious implementation details. Keep progress reports and final summaries concise.
* If a potentially expensive check is skipped, mention it briefly in the final summary rather than running it automatically.

The default principle is **minimum sufficient verification**, not maximum possible verification.
