# 2026-08-29 AndroidControl Mini Baseline And Profiling Analysis

本分析是当前 GUI Agent 静态测评的 baseline 记录，基于两类远程结果：

- `results/qwen_androidcontrol_mini.json`
- `results/profile_single_image.json`
- `results/profile_androidcontrol_mini.json`

这些结果文件不提交到 git；本文只记录关键结论和后续实验方向。

## Baseline Evaluation

本次 AndroidControl mini 静态测评共包含 10 条 trajectory、51 个 step。

| Metric | Value |
| --- | ---: |
| num_steps | 51 |
| num_trajectories | 10 |
| type_accuracy | 64.71% |
| step_success_rate | 41.18% |
| trajectory_success_rate | 0.00% |
| avg_latency_seconds | 14.62s |
| avg_output_tokens | 101.20 |
| peak_gpu_memory_gb | GPU0 25.08 / GPU1 28.33 |

整体看，baseline 已经跑通了完整链路：

`AndroidControl TFRecord -> screenshot/test.json -> Qwen3.8 static inference -> action normalize -> metric JSON`

但当前指标不能直接作为模型真实 GUI 能力的最终结论，因为结果里有明显的评测口径和输出格式问题。

## Accuracy Breakdown

按 GT action type 拆分：

| GT Type | Count | Type Acc | Step Acc | Avg Tokens | Avg Latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| CLICK | 24 | 83.3% | 58.3% | 100.3 | 13.66s |
| SCROLL | 7 | 85.7% | 0.0% | 111.0 | 17.81s |
| TYPE | 3 | 100.0% | 100.0% | 73.7 | 12.08s |
| PRESS_BACK | 2 | 100.0% | 100.0% | 74.5 | 9.33s |
| WAIT | 8 | 25.0% | 25.0% | 102.4 | 15.67s |
| OPEN_APP | 6 | 0.0% | 0.0% | 110.2 | 15.19s |
| LONG_PRESS | 1 | 0.0% | 0.0% | 128.0 | 21.90s |

过滤掉 `OPEN_APP` 和 `WAIT` 后，普通 GUI step 的表现更有参考价值：

| Filtered Metric | Value |
| --- | ---: |
| num_steps | 37 |
| type_accuracy | 83.78% |
| step_success_rate | 51.35% |

这说明模型在普通点击、输入、返回等静态 GUI step 上已有可用信号；总分较低主要被 `OPEN_APP`、`WAIT`、`SCROLL` 方向定义和长输出截断拉低。

## Main Failure Modes

### 1. Output Is Too Long

本次 baseline 中：

- 44 / 51 条输出包含 `</think>`。
- 12 / 51 条达到 `max_new_tokens=128` 截断。
- 平均输出 token 数为 101.20。

理想 GUI action 输出应接近：

```json
{"action":"tap","x":431,"y":70}
```

也就是大约 10-30 个 token。当前模型经常先输出自然语言推理，再输出 JSON；一旦达到 128 token 截断，action JSON 可能不完整，`canonicalize_action` 会将其记为 `UNKNOWN`。这同时影响 accuracy 和 latency。

结论：在做任何推理加速算法前，应先让模型稳定输出单个 action。否则 latency 的主要来源会被无效 reasoning token 放大。

### 2. `OPEN_APP` Is Not Comparable To Current Action Space

AndroidControl GT 有 6 条 `OPEN_APP`，当前模型 0 条成功。

原因不是模型一定不会打开 app，而是当前 prompt/action space 主要约束为 GUI 操作：

- `tap`
- `swipe`
- `type`
- `back`
- `home`
- `wait`
- `complete`
- `impossible`

它没有明确允许环境级 `OPEN_APP [app]`。因此模型面对 `OPEN_APP` GT 时会输出 `home`、`swipe` 或点击图标等 GUI action。静态单步评测中直接将这些 action 与 `OPEN_APP` 比较并不公平。

建议后续报告中将 `OPEN_APP` 单独统计，或者在 evaluator 中提供两种口径：

- `strict`: 与 AndroidControl GT 完全一致。
- `gui_only`: 排除或重映射 `OPEN_APP`。

### 3. `SCROLL[UP/DOWN]` Direction Needs Calibration

SCROLL 的 type accuracy 为 85.7%，但 step success 为 0。

典型失败：

```text
GT:   SCROLL[DOWN]
PRED: SCROLL[UP]
```

模型输出的原始动作通常是：

```json
{"action":"swipe","x1":500,"y1":700,"x2":500,"y2":300,"duration_ms":500}
```

这在手机手势上是“手指向上滑”，但用户语义常是“向下浏览页面内容”。AndroidControl 的 `SCROLL[DOWN]` 很可能表示页面内容浏览方向，而当前 evaluator 的 `infer_scroll_direction` 使用的是手指运动方向。

结论：SCROLL 方向定义必须先校准，否则 scroll step success 没有解释价值。

### 4. `WAIT` Has Static Evaluation Noise

WAIT 共 8 条，step success 只有 25%。部分失败样本中，当前截图已经显示可操作内容，模型选择点击或输入从静态单帧角度是合理的。

WAIT 通常描述环境状态变化或加载过程，静态单帧里较难严格判断。建议单独统计 WAIT，或在分析中将其作为 transition/no-op 类别处理。

## Profiling Results

### AndroidControl Mini Profiling

`profile_androidcontrol_mini.json` 跑了 5 个 step，`warmup=1`，`max_new_tokens=128`。

| Stage | Mean Time | Share |
| --- | ---: | ---: |
| build_prompt | 0.0000s | 0.00% |
| apply_chat_template | 0.0004s | 0.00% |
| vision_preprocess | 0.1025s | 0.97% |
| processor_encode | 0.0431s | 0.41% |
| input_to_device | 0.0114s | 0.11% |
| generate | 10.3821s | 98.50% |
| decode | 0.0004s | 0.00% |
| postprocess | 0.0001s | 0.00% |
| total | 10.5401s | 100.00% |

该 profiling 子集的平均输入 token 为 3242.8，平均输出 token 为 79.0。

逐 step 的生成耗时与输出 token：

| Episode | Step | Output Tokens | Generate Time | Seconds / Output Token |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 91 | 10.853s | 0.119 |
| 0 | 1 | 71 | 8.973s | 0.126 |
| 0 | 2 | 44 | 6.226s | 0.141 |
| 20 | 0 | 70 | 9.332s | 0.133 |
| 20 | 1 | 119 | 16.526s | 0.139 |

生成耗时与输出 token 数高度相关；最长的 119 token 样本耗时 16.53s。

### Single Image Profiling

`profile_single_image.json` 跑了 3 次 repeat，`warmup=1`，`max_new_tokens=128`。

| Stage | Mean Time | Share |
| --- | ---: | ---: |
| build_prompt | 0.0000s | 0.00% |
| apply_chat_template | 0.0006s | 0.01% |
| vision_preprocess | 0.1329s | 1.45% |
| processor_encode | 0.0465s | 0.51% |
| input_to_device | 0.0124s | 0.14% |
| generate | 8.9733s | 97.89% |
| decode | 0.0013s | 0.01% |
| postprocess | 0.0001s | 0.00% |
| total | 9.1672s | 100.00% |

平均输入 token 为 3228.0，平均输出 token 为 60.0。

### Memory

profiling 和 eval 的显存峰值基本一致：

| Source | GPU0 Peak Allocated | GPU1 Peak Allocated |
| --- | ---: | ---: |
| eval | 25.08 GB | 28.33 GB |
| AndroidControl profile | 25.06 GB | 28.31 GB |
| single image profile | 25.06 GB | 28.31 GB |

GPU1 显存更高，说明当前 `device_map=auto` 的模型切分不完全均衡。显存主要来自模型权重和 KV/cache，不是图像预处理。

## Bottleneck Diagnosis

当前 baseline 的性能瓶颈非常集中：

1. `generate` 占 98% 左右。
2. vision preprocess 约 0.10-0.13s，只占 1% 左右。
3. processor encode、device transfer、decode、postprocess 都很小。
4. 输出 token 数偏大，直接放大 generate latency。

因此现阶段不建议优先优化图片读取、JSON parser、action matching 或 Python loop。这些都不是主要耗时。

## Recommended Next Steps

### Step 1: Fix Output Format Before Acceleration

先让模型只输出一个 action，目标是：

- `avg_output_tokens` 从 101 降到 10-30。
- `UNKNOWN` 从 12/51 明显下降。
- `avg_latency_seconds` 随输出长度自然下降。

可尝试：

- 在 chat template 或 generation config 中关闭 thinking，例如传入 `enable_thinking=False`。
- 缩短 prompt 中的长说明，保留 action schema 和任务。
- 将 `max_new_tokens` 从 128 降到 64 或 32，并观察 parser 成功率。
- 如果模型仍输出 reasoning，增加更强的系统约束：只允许输出 JSON 或 legacy action，不输出解释。

### Step 2: Calibrate Evaluator

在比较加速方法前，先修正 metric 口径：

- 校准 `SCROLL[UP/DOWN]` 到 AndroidControl 的语义。
- 将 `OPEN_APP` 单独统计或排除。
- 将 `WAIT` 单独统计。
- 同时报告 `strict` 和 `gui_only` 两套指标。

否则不同方法之间的 success rate 波动可能来自 evaluator，而不是模型能力变化。

### Step 3: Use Profiling As Acceleration Baseline

后续每个加速实验至少记录：

- type accuracy
- step success rate
- trajectory success rate
- avg latency
- avg output tokens
- generate latency
- total latency
- peak GPU memory

建议固定同一份 `data/androidcontrol_mini/test.json` 和同样的 `--limit`。先用 5 条 profile 做开发，再用 51 条完整 mini 做对比。

### Step 4: Acceleration Directions

优先级建议如下：

1. **Decode length reduction**
   - 关闭 thinking。
   - 降低 `max_new_tokens`。
   - 约束输出 action schema。
   - 这是当前最可能同时提升速度和 accuracy 的方向。

2. **Generation backend**
   - 尝试 FlashAttention/SDPA 配置。
   - 对比 HF eager、vLLM、SGLang 等 serving backend。
   - 记录 TTFT 和 total generation time。

3. **Precision and quantization**
   - BF16/FP16 对比。
   - AWQ/GPTQ/FP8。
   - 重点看显存峰值和 success rate 是否下降。

4. **Static batching**
   - AndroidControl 静态评测天然适合 batch。
   - 但需要处理不同图像尺寸和输出长度；先 batch preprocess，再评估 batch generate 是否稳定。

5. **Prompt and visual token reduction**
   - 图片分辨率压缩。
   - 限制视觉 token。
   - 缓存固定 prompt 或 goal 部分。
   - 当前预处理只占 1%，所以这不是第一优先级，但可减少输入 token 和显存。

## Current Interpretation

当前实验的最重要结论不是“模型准确率只有 41%”，而是：

1. baseline 全链路已经跑通；
2. 模型有较强的普通 GUI action type 判断能力；
3. 严格 step success 被 scroll 方向、OPEN_APP/WAIT 口径、长输出截断显著低估；
4. latency 几乎完全由 generation 决定；
5. 最优先的“加速”其实是减少无效输出 token。

因此，下一轮实验应先做 prompt/generation/evaluator 校准，然后再开展量化、serving backend、batching 等真正的推理加速对比。

## 2026-08-29 21:31 CST Prompt And Metric Calibration Rerun

本节记录提示词强化和评测指标视图修正后的新结果。对应结果文件更新时间为：

- `results/qwen_androidcontrol_mini.json`: 2026-08-29 21:29:35 CST
- `results/profile_androidcontrol_mini.json`: 2026-08-29 21:29:35 CST
- `results/profile_single_image.json`: 仍为 2026-08-29 21:06:35 CST，本轮未重新生成

本轮仍使用同一份 `data/androidcontrol_mini/test.json`，`max_new_tokens=128`，模型路径为 `/data2/home/models/Qwen3.8-27B`。这次改动不涉及算法加速、量化、batching 或 serving backend，仅验证 prompt 输出约束和 evaluator 指标分视图。

### Evaluation Summary

| Metric | Previous | New | Change |
| --- | ---: | ---: | ---: |
| num_steps | 51 | 51 | 0 |
| num_trajectories | 10 | 10 | 0 |
| strict type_accuracy | 64.71% | 62.75% | -1.96 pp |
| strict step_success_rate | 41.18% | 47.06% | +5.88 pp |
| strict trajectory_success_rate | 0.00% | 0.00% | 0 pp |
| avg_latency_seconds | 14.62s | 4.10s | -72.0% |
| avg_output_tokens | 101.20 | 19.53 | -80.7% |
| peak_gpu_memory_gb | GPU0 25.08 / GPU1 28.33 | GPU0 25.11 / GPU1 28.36 | roughly unchanged |

最重要的变化是输出长度问题基本被压住：

- `</think>` 从 44 / 51 降到 0 / 51。
- 达到 `max_new_tokens=128` 截断从 12 / 51 降到 0 / 51。
- 51 / 51 条输出都以 `{` 开头并以 `}` 结尾。
- 输出 token 中位数为 18，范围为 6 到 42。

因此，本轮 latency 下降主要来自无效 reasoning token 消失，而不是底层推理算法变化。平均 step latency 从 14.62s 降到 4.10s，AndroidControl profile 子集的 `generate_seconds` 也从 10.38s 降到 3.12s。

### New Metric Views

本轮 evaluator 开始同时输出 `strict`、`gui_only`、`open_app`、`wait` 和 `by_gt_type` 视图。顶层指标仍等价于 `strict`，用于兼容旧结果。

| View | Steps | Type Acc | Step Acc | Traj Acc | Avg Tokens | Avg Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| strict | 51 | 62.75% | 47.06% | 0.00% | 19.53 | 4.10s |
| gui_only, excluding OPEN_APP and WAIT | 37 | 81.08% | 59.46% | 20.00% | 21.30 | 4.25s |
| open_app only | 6 | 0.00% | 0.00% | 0.00% | 13.17 | 3.72s |
| wait only | 8 | 25.00% | 25.00% | 25.00% | 16.12 | 3.67s |

`gui_only` 是目前更适合观察普通静态 GUI 操作能力的口径。与旧分析中过滤 `OPEN_APP` 和 `WAIT` 后的结果相比，`gui_only` step success 从 51.35% 提升到 59.46%，但 type accuracy 从 83.78% 小幅降到 81.08%。这说明 prompt 强约束改善了可执行 step 命中和 latency，但还没有完全解决 JSON 字段合法性。

### Breakdown By GT Type

| GT Type | Count | Type Acc | Step Acc | Avg Tokens | Avg Latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| CLICK | 24 | 75.00% | 70.83% | 17.33 | 3.87s |
| LONG_PRESS | 1 | 0.00% | 0.00% | 18.00 | 3.64s |
| OPEN_APP | 6 | 0.00% | 0.00% | 13.17 | 3.72s |
| PRESS_BACK | 2 | 100.00% | 100.00% | 6.00 | 3.19s |
| SCROLL | 7 | 100.00% | 0.00% | 42.00 | 6.27s |
| TYPE | 3 | 100.00% | 100.00% | 16.00 | 3.45s |
| WAIT | 8 | 25.00% | 25.00% | 16.12 | 3.67s |

CLICK 的 step success 从 58.3% 提升到 70.8%，但 type accuracy 从 83.3% 降到 75.0%。主要原因不是模型不再点击，而是部分短 JSON 写成了非法字段格式，例如：

```json
{"action":"tap","x":498,491}
```

这类输出很短，也没有 thinking，但缺少独立的 `y` 字段，当前 parser 会按 `UNKNOWN` 处理。类似 malformed tap JSON 在本轮 UNKNOWN 中占主要部分。

SCROLL 的 type accuracy 从 85.7% 提升到 100.0%，但 step success 仍为 0.0%。失败仍集中在方向语义：

```text
GT:   SCROLL[DOWN]
PRED: SCROLL[UP]
```

这进一步支持旧结论：SCROLL 的问题不是输出长度，而是 AndroidControl 方向定义与手指滑动方向的 evaluator 语义不一致，需要单独校准。

### Remaining Failure Modes

本轮 `pred_type=UNKNOWN` 共 11 / 51。按现象看主要分为三类：

1. malformed tap JSON：模型输出了类似 `{"action":"tap","x":823,90}` 的短 JSON，缺少 `y` 键。
2. unsupported open action：模型输出 `{"action":"open","text":"Zoho Meet"}` 或 `{"action":"openApp","text":"Rtistiq"}`，但当前 action contract 没有 open/openApp。
3. malformed coordinate key：例如 `{"action":"tap","x":258,"417":351}`。

失败矩阵显示：

| GT Type | Main Failed Pred Types |
| --- | --- |
| CLICK | UNKNOWN 6, CLICK 1 |
| LONG_PRESS | CLICK 1 |
| OPEN_APP | UNKNOWN 3, CLICK 2, PRESS_HOME 1 |
| SCROLL | SCROLL 7 |
| WAIT | CLICK 4, UNKNOWN 2 |

这说明下一步仍应优先做测试代码和 prompt 级别校准，而不是算法优化：

- prompt 继续强调 tap 必须使用 `"x": number, "y": number`，不要写成 `"x":498,491`。
- 对 `OPEN_APP` 明确保持单独统计；如果要让 strict 指标可比，需要决定是否把 `open/openApp` 纳入 contract，或继续作为环境级 action 单独报告。
- 对 SCROLL 单独做方向语义校准，避免把正确的浏览意图记为失败。
- WAIT 继续单独统计，因为静态单帧下它和可操作点击之间有天然噪声。

### Profiling Summary

新的 AndroidControl mini profile 跑了 5 个 step，`warmup=1`，`max_new_tokens=128`。

| Stage | Mean Time | Share |
| --- | ---: | ---: |
| build_prompt | 0.0000s | 0.00% |
| apply_chat_template | 0.0003s | 0.01% |
| vision_preprocess | 0.1000s | 3.06% |
| processor_encode | 0.0336s | 1.03% |
| input_to_device | 0.0099s | 0.30% |
| generate | 3.1204s | 95.58% |
| decode | 0.0002s | 0.01% |
| postprocess | 0.0000s | 0.00% |
| total | 3.2646s | 100.00% |

profile 子集平均输出 token 从旧结果的 79.0 降到 12.8，`generate_seconds` 从 10.38s 降到 3.12s。`generate` 仍是主耗时，占比 95.6%，但绝对时间已随输出长度显著下降。视觉预处理仍约 0.10s，不是当前主要瓶颈。

### Updated Interpretation

本轮验证了旧分析中的首要判断：在做真正推理加速前，先消除 thinking 和长输出是收益最大的修正。现在输出长度和截断问题已基本解决，latency 明显下降，普通 GUI step success 也提升。

当前瓶颈已经从“长 reasoning 输出”转移到三件更具体的校准工作：

1. JSON 字段合法性：短输出仍可能写错字段，导致 UNKNOWN。
2. SCROLL 方向语义：type 已对，但方向匹配仍全错。
3. OPEN_APP/WAIT 口径：应继续与普通 GUI action 分开报告。

因此，下一轮仍建议继续做测试代码修正和 prompt/action contract 校准；在这些口径稳定前，不建议开始量化、serving backend、batching 等算法或系统优化对比。
