# 2026-08-29 AndroidControl 校准复跑分析

分析时间：2026-08-29 21:47 CST

本分析基于修复 malformed JSON、校准 SCROLL 方向语义、明确 OPEN_APP/WAIT 指标口径后的新结果：

- `results/qwen_androidcontrol_mini.json`
  - 文件更新时间：2026-08-29 21:46:52 CST
  - 结果生成时间：2026-08-29 21:45:13 CST
- `results/profile_androidcontrol_mini.json`
  - 文件更新时间：2026-08-29 21:46:52 CST
  - 结果生成时间：2026-08-29 21:45:54 CST
- `results/profile_single_image.json`
  - 文件更新时间仍为 2026-08-29 21:06:35 CST，本轮未重新生成

本轮仍使用同一份 `data/androidcontrol_mini/test.json`，模型路径为 `/data2/home/models/Qwen3.8-27B`，`max_new_tokens=128`，`point_tolerance=100.0`。本轮改动属于测试代码和 prompt/action contract 校准，不涉及模型、量化、batching、serving backend 或其他推理加速算法。

## 核心结论

本轮校准解决了前一轮剩余的两个核心 evaluator/parser 问题：

- malformed JSON 不再导致 `UNKNOWN`：`pred_unknown` 从 11 / 51 降到 0 / 51。
- SCROLL 方向语义已对齐 AndroidControl：SCROLL step 成功率从 0 / 7 提升到 7 / 7。

主观察口径应使用 `gui_only`，即排除 `OPEN_APP` 和 `WAIT` 的普通 GUI 操作。新结果中：

- `gui_only` 类型准确率为 97.30%。
- `gui_only` step 成功率为 91.89%。
- `gui_only` 轨迹成功率为 80.00%。
- strict step 成功率为 74.51%，主要仍被 `OPEN_APP` 和 `WAIT` 拉低。

这说明当前静态 GUI baseline 的普通点击、滑动、输入、返回能力已经明显高于最初 strict 总分所暗示的水平。剩余主要问题不是输出长度、JSON 解析或 SCROLL evaluator，而是：

1. `WAIT` 在静态单帧里天然噪声很大。
2. `OPEN_APP` 是环境级动作，和普通 GUI 动作不应混为一个能力指标。
3. 少量 CLICK 坐标仍有明显偏移。
4. LONG_PRESS 只有 1 个样本，本轮仍失败，样本太少但需要单独关注。

## 结果对比

| 指标 | 原始基线 | 提示词复跑 | 当前复跑 |
| --- | ---: | ---: | ---: |
| num_steps | 51 | 51 | 51 |
| strict type_accuracy | 64.71% | 62.75% | 80.39% |
| strict step_success_rate | 41.18% | 47.06% | 74.51% |
| strict trajectory_success_rate | 0.00% | 0.00% | 40.00% |
| gui_only type_accuracy | 83.78% | 81.08% | 97.30% |
| gui_only step_success_rate | 51.35% | 59.46% | 91.89% |
| gui_only trajectory_success_rate | 未报告 | 20.00% | 80.00% |
| avg_latency_seconds | 14.62s | 4.10s | 4.62s |
| avg_output_tokens | 101.20 | 19.53 | 19.94 |
| peak_gpu_memory_gb | GPU0 25.08 / GPU1 28.33 | GPU0 25.11 / GPU1 28.36 | GPU0 25.14 / GPU1 28.39 |

解释：

- 从 original 到 prompt rerun，主要收益来自关闭 thinking 和强化 JSON 输出，平均输出 token 从 101.20 降到 19.53，latency 从 14.62s 降到 4.10s。
- 从 prompt rerun 到 current rerun，主要收益来自 evaluator/parser 校准：malformed tap JSON 被修复，SCROLL 从手指方向改为内容浏览方向，`open/openApp` 被规范为 `OPEN_APP`。
- 当前 latency 比 prompt rerun 略高，4.10s 到 4.62s，原因不是输出重新变长；平均输出 token 基本相同，19.53 到 19.94。更可能是远端运行波动或输入 token 小幅增加，后续需要用多次 profile 取均值确认。

## 当前指标视图

| 视图 | Step 数 | 类型准确率 | Step 准确率 | 轨迹准确率 | 平均 token | 平均延迟 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| strict | 51 | 80.39% | 74.51% | 40.00% | 19.94 | 4.62s |
| gui_only | 37 | 97.30% | 91.89% | 80.00% | 21.95 | 4.80s |
| transition_or_noop | 14 | 35.71% | 28.57% | 16.67% | 14.64 | 4.15s |
| 仅 open_app | 6 | 66.67% | 50.00% | 50.00% | 12.67 | 4.12s |
| 仅 wait | 8 | 12.50% | 12.50% | 0.00% | 16.12 | 4.17s |

`strict` 仍保留用于和 AndroidControl 原始 GT 完全一致地比较，但不应作为普通 GUI 操作能力的唯一指标。`gui_only` 是本阶段最有解释力的主指标；`transition_or_noop`、`open_app`、`wait` 用于单独观察环境级动作和静态 no-op 噪声。

## 按 GT 类型拆解

| GT 类型 | 数量 | 类型准确率 | Step 准确率 | 平均 token | 平均延迟 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CLICK | 24 | 100.00% | 91.67% | 18.33 | 4.43s |
| LONG_PRESS | 1 | 0.00% | 0.00% | 18.00 | 4.97s |
| OPEN_APP | 6 | 66.67% | 50.00% | 12.67 | 4.12s |
| PRESS_BACK | 2 | 100.00% | 100.00% | 6.00 | 2.94s |
| SCROLL | 7 | 100.00% | 100.00% | 42.00 | 6.85s |
| TYPE | 3 | 100.00% | 100.00% | 16.00 | 4.17s |
| WAIT | 8 | 12.50% | 12.50% | 16.12 | 4.17s |

### CLICK

CLICK 已经从上一轮的 75.00% 类型准确率 / 70.83% step 成功率提升到 100.00% / 91.67%。这说明 malformed tap JSON 修复有效。当前 CLICK 的剩余失败是坐标偏差，不是 parser 问题：

- episode 80 step 1：GT `[[87,493]]`，预测 `[[498,491]]`
- episode 140 step 6：GT `[[50,219]]`，预测 `[[491,218]]`

这两个失败都是横向坐标明显偏右，后续需要结合截图人工确认：如果截图里左侧目标和中间目标都可见，可能是视觉定位错误；如果 GT 本身点在边缘控件，可能需要更明确提示“点击目标控件本身，不要点击同一行中心”。

### SCROLL

SCROLL 已完全修复：

- type accuracy: 100.00%
- step 成功率：100.00%
- 7 / 7 全部成功

这说明 AndroidControl 的 `SCROLL[DOWN]` 更适合解释为“向下浏览内容/页面”，而不是手指物理运动方向。当前 evaluator 将手指上滑 `y1 > y2` 规范为 `SCROLL[DOWN]`，与本轮结果吻合。

### OPEN_APP

OPEN_APP 从上一轮的 0.00% step 成功率提升到 50.00%。这来自 `open/openApp/launch` 归一化到 `OPEN_APP [...]`，但仍不建议把 OPEN_APP 混入普通 GUI 操作能力：

- 它是环境级 app launch intent，而不是屏幕内控件操作。
- strict matching 还会受 app 名称字符串影响，例如 `Zoho Meeting` vs `Zoho Meet`。
- 有些截图已经处在目标 app 内，模型输出 CLICK 或 WAIT 在静态单帧语境下可能合理，但 strict OPEN_APP 会判错。

因此 OPEN_APP 应继续单独报告。若未来需要提升 strict OPEN_APP，需要单独决定 app 名称 alias 匹配策略，或在 action contract 中显式加入环境级 `open_app`。

### WAIT

WAIT 是当前最大剩余拉低项：

- 8 个 WAIT 中只有 1 个成功。
- 失败主要是模型选择 CLICK，另有 1 个预测为 OPEN_APP。
- 仅 WAIT 的 step 成功率为 12.50%。

这不是单纯模型格式问题。静态单帧评测无法观察“等待后是否加载完成”，而模型看到可操作控件时选择点击经常是合理行为。WAIT 应继续作为 transition/no-op 单独报告，不建议纳入普通 GUI 成功率。

### LONG_PRESS

LONG_PRESS 只有 1 个样本，仍失败为 CLICK。这个结论样本量太小，不能泛化，但说明 prompt/action contract 可能还需要加入 long_press 格式，或者 evaluator 明确是否要把 long_press 计入 gui_only 主指标。

当前共享 prompt 没有把 long_press 列为允许 action；AndroidControl GT 却包含 LONG_PRESS。因此它在 strict 和 gui_only 里都会天然吃亏。下一步需要二选一：

- 将 `long_press` 加入 action contract 和执行器支持，继续作为普通 GUI 操作评测。
- 或者把 LONG_PRESS 也移出当前 gui_only 主口径，作为 unsupported/currently-unmodeled action 单独报告。

## 输出格式健康度

本轮输出格式已经稳定：

| 检查项 | 数值 |
| --- | ---: |
| 包含 `</think>` | 0 / 51 |
| 命中 `max_new_tokens=128` | 0 / 51 |
| `pred_type` 为 UNKNOWN | 0 / 51 |
| 以 `{` 开头 | 50 / 51 |
| 以 `}` 结尾 | 50 / 51 |
| 输出 token 最小值 / 中位数 / 最大值 | 6 / 18 / 42 |

虽然有 1 条输出不满足 JSON 边界检查，但 parser 最终仍能规范为有效 action，整体已不再受 UNKNOWN 和截断主导。

## 性能剖析

新的 AndroidControl mini profile 跑了 5 个 step，`warmup=1`，`max_new_tokens=128`。

| 阶段 | 平均耗时 | 占比 |
| --- | ---: | ---: |
| build_prompt | 0.0000s | 0.00% |
| apply_chat_template | 0.0003s | 0.01% |
| vision_preprocess | 0.1051s | 3.08% |
| processor_encode | 0.0474s | 1.39% |
| input_to_device | 0.0115s | 0.34% |
| generate | 3.2459s | 95.18% |
| decode | 0.0002s | 0.01% |
| postprocess | 0.0000s | 0.00% |
| total | 3.4106s | 100.00% |

profile 子集指标：

| 视图 | Step 数 | 类型准确率 | Step 准确率 | 轨迹准确率 | 平均 token | 平均延迟 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| strict | 5 | 80.00% | 60.00% | 50.00% | 13.20 | 3.25s |
| gui_only | 3 | 100.00% | 100.00% | 100.00% | 14.00 | 3.32s |

`generate` 仍占 95% 以上，但绝对时间已经由输出长度控制在约 3.25s。视觉预处理约 0.10s，仍不是主要瓶颈。

## 后续工作

当前建议继续保持“先测试口径和 prompt/action contract，后算法优化”的顺序。

优先级建议：

1. 明确 LONG_PRESS 口径。
   - 如果要评估 AndroidControl strict，加入 `{"action":"long_press","x":...,"y":...,"duration_ms":...}` 到 prompt/action contract。
   - 如果当前执行链路暂不支持 long_press，就从 gui_only 主指标中移出，单独报告。

2. 继续单独报告 OPEN_APP。
   - 当前 strict OPEN_APP 已部分可解析，但它仍是环境级动作。
   - 如需 strict 友好，应增加 app name alias，例如 `Zoho Meeting` 与 `Zoho Meet`。

3. 继续单独报告 WAIT。
   - WAIT 不适合静态单帧主指标。
   - 可以保留 strict 分数，但不要用它判断普通 GUI 操作能力。

4. 人工检查两个 CLICK 坐标偏移样本。
   - episode 80 step 1
   - episode 140 step 6

5. 在上述口径固定后，再开始比较加速方法。
   - 当前输出 token 已降到 20 左右，继续优化 latency 时再看 FlashAttention/SDPA、serving backend、量化和 batching 才更有解释力。

## 当前判断

当前最可信的结论是：Qwen3.8 在 AndroidControl mini 的普通静态 GUI 操作上已经达到较高命中率，`gui_only` step 成功率为 91.89%。最初 41.18% 的 strict step 成功率明显低估了能力，因为它混合了 thinking 截断、malformed JSON、SCROLL 方向语义、OPEN_APP 环境动作和 WAIT 静态噪声。

本轮之后，前面三个技术性问题已基本解决：

- thinking/长输出：已解决。
- malformed JSON/UNKNOWN：已解决。
- SCROLL 方向语义：已解决。

剩余主要是评测定义问题和少量视觉定位错误。下一步应把 LONG_PRESS、OPEN_APP、WAIT 的口径最终固定，再进入真正的推理加速实验。
